"""Executa os comandos do parser, chamando os módulos de cálculo.

Dois avaliadores trabalham sobre a mesma árvore:
- numérico: calcula valores exatos (números, raízes, vetores, matrizes);
- linear: monta expressões com incógnitas (resolver, T(...) = ..., subespaco).
"""
from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field
from fractions import Fraction

from lae import matrices as mt
from lae import vectors as vc
from lae.determinants import det
from lae.exact import Root
from lae.linear_systems import Solution, rank, rref
from lae.linear_systems import solve as solve_system
from lae.parser import (
    Assign, BinOp, Call, Command, Define, ImplicitSubspace, Line, Mat, Neg, Node, Num,
    ParamSubspace, Program, Show, Solve, Title, Var, Vec, parse_program,
)
from lae.transformations import Transformation, evaluate, from_formula, image, kernel
from lae.vector_spaces import Affine, check_implicit, check_parametric, coordinates, is_independent

Form = dict[str, Fraction]  # expressão linear: {"x": 2, "": 1} representa 2x + 1
Linear = Form | tuple[Form, ...]

CALLS = {  # função: (tipos dos argumentos: v = vetor, m = matriz, função de cálculo)
    "dot": ("vv", vc.dot), "norma": ("v", vc.norm), "dist": ("vv", vc.distance),
    "cos": ("vv", vc.cosine), "det": ("m", det), "transposta": ("m", mt.transpose),
    "posto": ("m", lambda a: Fraction(rank(a))), "escalonar": ("m", rref),
}
KIND_CODES = {"v": "vetor", "m": "matriz"}


def kind(value: object) -> str:
    """Nome do tipo de um valor, usado nas mensagens de erro e no passo a passo."""
    if isinstance(value, Fraction):
        return "número"
    if isinstance(value, Root):
        return "raiz"
    if isinstance(value, tuple) and value:
        return "matriz" if isinstance(value[0], tuple) else "vetor"
    return type(value).__name__


def simplify(value: object) -> object:
    """Raiz racional vira Fraction, para seguir nas contas comuns."""
    return value.as_fraction() if isinstance(value, Root) and value.is_rational else value


# ---------- aritmética de expressões lineares ----------

def _form(value: Fraction) -> Form:
    return {"": value} if value else {}


def _constant(f: Form) -> Fraction | None:
    """Valor da forma se ela não tem incógnitas; senão None."""
    return f.get("", Fraction(0)) if set(f) <= {""} else None


def _add(a: Form, b: Form, sign: int = 1) -> Form:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, Fraction(0)) + sign * v
    return {k: v for k, v in out.items() if v != 0}


def _scale(k: Fraction, f: Form) -> Form:
    return {n: k * v for n, v in f.items() if k * v != 0}


def _mul(a: Form, b: Form) -> Form:
    if (k := _constant(a)) is not None:
        return _scale(k, b)
    if (k := _constant(b)) is not None:
        return _scale(k, a)
    raise ValueError("produto entre incógnitas: a expressão não é linear")


@dataclass
class Engine:
    """Estado do programa: variáveis e transformações definidas até agora."""

    vars: dict[str, object] = field(default_factory=dict)
    transforms: dict[str, Transformation] = field(default_factory=dict)

    def snapshot(self) -> Engine:
        """Cópia do estado, para o passo a passo reavaliar subexpressões."""
        return Engine(dict(self.vars), dict(self.transforms))

    def lookup(self, name: str) -> object:
        if name in self.vars:
            return self.vars[name]
        if name in self.transforms:
            return self.transforms[name]
        raise ValueError(f"variável '{name}' não definida")

    def scalar(self, value: object) -> Fraction:
        value = simplify(value)
        if not isinstance(value, Fraction):
            raise ValueError(f"esperado um número, encontrado {kind(value)}")
        return value

    # ----- avaliador numérico -----

    def eval(self, node: Node) -> object:
        match node:
            case Num(value):
                return value
            case Var(name):
                return self.lookup(name)
            case Vec(items):
                return vc.vector([self.scalar(self.eval(i)) for i in items])
            case Mat(rows):
                return mt.matrix([[self.scalar(self.eval(x)) for x in r] for r in rows])
            case Neg(operand):
                return self.binop("*", Fraction(-1), self.eval(operand))
            case BinOp(op, left, right):
                return self.binop(op, self.eval(left), self.eval(right))
            case Call(name, args):
                return simplify(self.call(name, args))
        raise ValueError("expressão desconhecida")

    def binop(self, op: str, a: object, b: object) -> object:
        a, b = simplify(a), simplify(b)
        ka, kb = kind(a), kind(b)
        if op in "+-":
            if ka != kb or ka not in ("número", "vetor", "matriz"):
                raise ValueError(f"não é possível somar/subtrair {ka} e {kb}")
            if op == "-":
                b = self.binop("*", Fraction(-1), b)
            if ka == "número":
                return a + b
            return vc.add(a, b) if ka == "vetor" else mt.add(a, b)
        if kb == "número" and ka != "número":
            a, b, ka, kb = b, a, kb, ka  # escalar sempre à esquerda
        if ka == "número":
            if kb in ("número", "raiz"):
                return simplify(b * a) if kb == "raiz" else a * b
            if kb in ("vetor", "matriz"):
                return vc.scale(a, b) if kb == "vetor" else mt.scale(a, b)
        if ka == kb == "raiz":
            return simplify(a * b)
        if ka == "matriz" and kb in ("matriz", "vetor"):
            return mt.mul(a, b) if kb == "matriz" else mt.apply(a, b)
        hint = " (para produto escalar use dot(u, v))" if ka == kb == "vetor" else ""
        raise ValueError(f"não é possível multiplicar {ka} por {kb}{hint}")

    def call(self, name: str, args: tuple[Node, ...]) -> object:
        if name in self.transforms:
            vals = [self.eval(a) for a in args]
            single = len(vals) == 1 and kind(vals[0]) == "vetor"
            v = vals[0] if single else vc.vector([self.scalar(x) for x in vals])
            return evaluate(self.transforms[name], v)
        if name in ("nucleo", "imagem"):
            t = self.as_transformation(args[0])
            return kernel(t) if name == "nucleo" else image(t)
        vals = [simplify(self.eval(a)) for a in args]
        if name in ("independente", "coordenadas"):
            self.check_kinds(name, vals, "v" * len(vals))
            return is_independent(vals) if name == "independente" else coordinates(vals[0], vals[1:])
        codes, fn = CALLS[name]
        self.check_kinds(name, vals, codes)
        return fn(*vals)

    def check_kinds(self, name: str, vals: list[object], codes: str) -> None:
        for i, (v, c) in enumerate(zip(vals, codes), start=1):
            if kind(v) != KIND_CODES[c]:
                raise ValueError(f"{name}: argumento {i} deve ser {KIND_CODES[c]}, recebeu {kind(v)}")

    def as_transformation(self, node: Node) -> Transformation:
        """Aceita o nome de uma transformação ou uma matriz qualquer."""
        if isinstance(node, Var) and node.name in self.transforms:
            return self.transforms[node.name]
        a = self.eval(node)
        if kind(a) != "matriz":
            raise ValueError(f"esperada transformação ou matriz, recebeu {kind(a)}")
        return Transformation([f"x{i + 1}" for i in range(mt.shape(a)[1])], a)

    # ----- avaliador linear (incógnitas como símbolos) -----

    def linear(self, node: Node, symbols: Collection[str]) -> Linear:
        match node:
            case Num(value):
                return _form(value)
            case Var(name) if name in symbols:
                return {name: Fraction(1)}
            case Var(name):
                value = simplify(self.lookup(name))
                if kind(value) == "número":
                    return _form(value)
                if kind(value) == "vetor":
                    return tuple(_form(x) for x in value)
                raise ValueError(f"'{name}' ({kind(value)}) não pode aparecer numa expressão linear")
            case Vec(items):
                parts = [self.linear(i, symbols) for i in items]
                if not all(isinstance(p, dict) for p in parts):
                    raise ValueError("coordenadas de um vetor devem ser números")
                return tuple(parts)
            case Neg(operand):
                return self.linear(BinOp("*", Num(Fraction(-1)), operand), symbols)
            case BinOp(op, left, right):
                return self.linear_op(op, self.linear(left, symbols), self.linear(right, symbols))
        raise ValueError("em expressões com incógnitas só valem +, - e * por número")

    def linear_op(self, op: str, a: Linear, b: Linear) -> Linear:
        if op == "*":
            if isinstance(a, tuple) and isinstance(b, tuple):
                raise ValueError("não é possível multiplicar vetor por vetor")
            if isinstance(a, tuple):
                return tuple(_mul(x, b) for x in a)
            return tuple(_mul(a, x) for x in b) if isinstance(b, tuple) else _mul(a, b)
        sign = 1 if op == "+" else -1
        if isinstance(a, dict) and isinstance(b, dict):
            return _add(a, b, sign)
        if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
            return tuple(_add(x, y, sign) for x, y in zip(a, b))
        raise ValueError("dimensões incompatíveis na expressão")

    def affine(self, node: Node, symbols: Collection[str]) -> Affine:
        f = self.linear(node, symbols)
        if not isinstance(f, dict):
            raise ValueError("cada coordenada deve ser uma expressão numérica, não um vetor")
        return Affine({k: v for k, v in f.items() if k}, f.get("", 0))

    # ----- comandos -----

    def execute(self, command: Command) -> object:
        match command:
            case Title():
                return None
            case Assign(name, expr):
                self.vars[name] = self.eval(expr)
                return self.vars[name]
            case Show(expr):
                return self.eval(expr)
            case Solve(unknowns, left, right):
                return self.solve(unknowns, left, right)
            case Define(name, params, outputs):
                t = from_formula(params, [self.affine(o, params) for o in outputs])
                self.transforms[name] = t
                self.vars.pop(name, None)
                return t
            case ParamSubspace(params, coords):
                return check_parametric(params, [self.affine(c, params) for c in coords])
            case ImplicitSubspace(names, equations):
                return check_implicit(names, [self.affine(BinOp("-", l, r), names) for l, r in equations])
        raise ValueError("comando desconhecido")

    def solve(self, unknowns: tuple[str, ...], left: Node, right: Node) -> Solution:
        """Passa tudo para um lado (esquerda - direita = 0) e monta A x = b.

        Se as incógnitas estão só à direita, os lados são trocados antes, para o
        sistema sair como no caderno: 3x + 2y = -6, e não -3x - 2y = 6.
        """
        lhs = self.linear(left, unknowns)
        if all(_constant(f) is not None for f in (lhs if isinstance(lhs, tuple) else (lhs,))):
            left, right = right, left
        diff = self.linear(BinOp("-", left, right), unknowns)
        equations = diff if isinstance(diff, tuple) else (diff,)
        a = mt.matrix([[eq.get(u, Fraction(0)) for u in unknowns] for eq in equations])
        b = vc.vector([-eq.get("", Fraction(0)) for eq in equations])
        sol = solve_system(a, b)
        if sol.kind == "SPD":
            self.vars.update(zip(unknowns, sol.particular))
        return sol


@dataclass
class Outcome:
    """Resultado de uma linha: valor ou erro, e o estado antes dela rodar."""

    line: Line
    value: object = None
    error: str | None = None
    scope: Engine = field(default_factory=Engine)


@dataclass
class Report:
    outcomes: list[Outcome]
    warnings: list[str]


def run(program: Program) -> Report:
    """Executa linha a linha; um erro não interrompe as linhas seguintes."""
    engine = Engine()
    outcomes = []
    for line in program.lines:
        scope = engine.snapshot()
        try:
            outcomes.append(Outcome(line, engine.execute(line.command), None, scope))
        except (ValueError, ArithmeticError) as exc:
            outcomes.append(Outcome(line, None, f"linha {line.number}: {exc}", scope))
    return Report(outcomes, program.warnings)


def solve_text(text: str) -> Report:
    """Atalho: lê e executa um programa da mini-linguagem."""
    return run(parse_program(text))