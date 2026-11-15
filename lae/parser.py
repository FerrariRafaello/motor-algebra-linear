"""Mini-linguagem do motor: texto -> comandos estruturados.

O parser só reconhece números, nomes, operadores e os comandos da linguagem.
Ele nunca usa eval/exec: nenhum texto de entrada vira código Python.
Linhas com instruções suspeitas (prompt injection) são ignoradas e viram aviso.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from fractions import Fraction
from typing import NamedTuple, NoReturn


class ParseError(ValueError):
    """Erro de sintaxe, sempre com o número da linha."""

    def __init__(self, line: int, message: str) -> None:
        super().__init__(f"linha {line}: {message}")
        self.line = line


# ---------- árvore das expressões ----------

@dataclass(frozen=True)
class Num:
    value: Fraction


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class Vec:
    items: tuple[Node, ...]


@dataclass(frozen=True)
class Mat:
    rows: tuple[tuple[Node, ...], ...]


@dataclass(frozen=True)
class BinOp:
    op: str  # "+", "-" ou "*"
    left: Node
    right: Node


@dataclass(frozen=True)
class Neg:
    operand: Node


@dataclass(frozen=True)
class Call:
    name: str
    args: tuple[Node, ...]


Node = Num | Var | Vec | Mat | BinOp | Neg | Call


# ---------- comandos (um por linha) ----------

@dataclass(frozen=True)
class Title:
    text: str


@dataclass(frozen=True)
class Assign:
    name: str
    expr: Node


@dataclass(frozen=True)
class Show:
    expr: Node


@dataclass(frozen=True)
class Solve:
    unknowns: tuple[str, ...]
    left: Node
    right: Node


@dataclass(frozen=True)
class Define:
    name: str
    params: tuple[str, ...]
    outputs: tuple[Node, ...]


@dataclass(frozen=True)
class ParamSubspace:
    params: tuple[str, ...]
    coords: tuple[Node, ...]


@dataclass(frozen=True)
class ImplicitSubspace:
    names: tuple[str, ...]
    equations: tuple[tuple[Node, Node], ...]


Command = Title | Assign | Show | Solve | Define | ParamSubspace | ImplicitSubspace


@dataclass(frozen=True)
class Line:
    number: int
    source: str
    command: Command


@dataclass
class Program:
    lines: list[Line]
    warnings: list[str]


# ---------- vocabulário ----------

KEYWORDS = {"resolver", "subespaco"}
FUNCTIONS = {  # nome: (mínimo, máximo) de argumentos
    "dot": (2, 2), "norma": (1, 1), "dist": (2, 2), "cos": (2, 2),
    "det": (1, 1), "transposta": (1, 1), "posto": (1, 1), "escalonar": (1, 1),
    "independente": (1, 99), "coordenadas": (2, 99), "nucleo": (1, 1), "imagem": (1, 1),
}
RESERVED = KEYWORDS | FUNCTIONS.keys()

SUSPICIOUS = [  # aplicados ao texto minúsculo, sem acentos e sem pontuação
    r"\bse (voce|vc|tu) (for|e|es) (uma? )?(ia|ai|inteligencia artificial|assistente|modelo)\b",
    r"\bignor\w* (\w+ ){0,3}instruc",
    r"\bignore (\w+ ){0,3}instructions?\b",
    r"\bnao (resolva|responda|calcule|forneca)\b",
    r"\balter\w* (\w+ ){0,2}(resultado|resposta|calculo)",
    r"\b(resposta|resultado)s? (incorreta|errada|falsa|aleatoria|diferente)",
]


class Token(NamedTuple):
    kind: str  # "num", "name" ou "op"
    text: str
    pos: int
    end: int


TOKEN_RE = re.compile(r"""
    (?P<num>\d+(?:\.\d+)?(?:/\d+)?)
  | (?P<name>[^\W\d]\w*)
  | (?P<op>[-+*=(),\[\]{}|:])
  | (?P<space>\s+)
  | (?P<bad>.)
""", re.VERBOSE)


def tokenize(code: str, line: int = 0) -> list[Token]:
    """Quebra o texto em tokens; qualquer caractere fora da linguagem é erro."""
    tokens = []
    for m in TOKEN_RE.finditer(code):
        if m.lastgroup == "bad":
            raise ParseError(line, f"caractere inválido '{m.group()}'")
        if m.lastgroup != "space":
            tokens.append(Token(m.lastgroup, m.group(), m.start(), m.end()))
    return tokens


def is_suspicious(text: str) -> bool:
    """True se o texto parece uma instrução embutida (prompt injection)."""
    plain = unicodedata.normalize("NFD", text.lower())
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    plain = " ".join(re.sub(r"\W+", " ", plain).split())
    return any(re.search(p, plain) for p in SUSPICIOUS)


class _Parser:
    """Analisador descendente recursivo de uma única linha."""

    def __init__(self, code: str, line: int, transforms: dict[str, int]) -> None:
        self.tokens = tokenize(code, line)
        self.i, self.line, self.transforms = 0, line, transforms

    def fail(self, message: str) -> NoReturn:
        raise ParseError(self.line, message)

    def peek(self, k: int = 0) -> Token | None:
        j = self.i + k
        return self.tokens[j] if j < len(self.tokens) else None

    def at(self, text: str) -> bool:
        t = self.peek()
        return t is not None and t.text == text

    def take(self) -> Token:
        t = self.peek()
        if t is None:
            self.fail("a linha terminou antes do esperado")
        self.i += 1
        return t

    def accept(self, text: str) -> bool:
        if self.at(text):
            self.i += 1
            return True
        return False

    def expect(self, text: str) -> None:
        t = self.take()
        if t.text != text:
            self.fail(f"esperado '{text}', encontrado '{t.text}'")

    def name(self) -> str:
        t = self.take()
        if t.kind != "name":
            self.fail(f"esperado um nome, encontrado '{t.text}'")
        if t.text in RESERVED:
            self.fail(f"'{t.text}' é palavra reservada")
        return t.text

    def name_list(self) -> tuple[str, ...]:
        names = [self.name()]
        while self.accept(","):
            names.append(self.name())
        if len(set(names)) != len(names):
            self.fail("nomes repetidos")
        return tuple(names)

    # ----- comandos -----

    def parse(self) -> Command:
        command = self.statement()
        if self.at("="):
            self.fail("igualdade fora de 'resolver' (use: resolver x, y: ... = ...)")
        if self.peek() is not None:
            self.fail(f"símbolo inesperado '{self.peek().text}'")
        return command

    def statement(self) -> Command:
        first, second = self.peek(), self.peek(1)
        second_text = second.text if second else ""
        if self.accept("resolver"):
            unknowns = self.name_list()
            self.expect(":")
            left = self.expr()
            self.expect("=")
            return Solve(unknowns, left, self.expr())
        if self.accept("subespaco"):
            return self.subspace()
        if first.kind == "name" and second_text == "=":
            name = self.name()
            self.take()
            if name in self.transforms:
                self.fail(f"'{name}' já é uma transformação")
            return Assign(name, self.expr())
        if first.kind == "name" and second_text == "(" and any(t.text == "=" for t in self.tokens):
            return self.define()
        return Show(self.expr())

    def define(self) -> Define:
        name = self.name()
        self.expect("(")
        params = self.name_list()
        self.expect(")")
        self.expect("=")
        outputs = self.parens() if self.at("(") else (self.expr(),)
        self.transforms[name] = len(params)
        return Define(name, params, outputs)

    def subspace(self) -> Command:
        self.expect("{")
        coords = self.parens()
        self.expect("|")
        params: list[Node] = []
        equations: list[tuple[Node, Node]] = []
        while True:
            left = self.expr()
            if self.accept("="):
                equations.append((left, self.expr()))
            else:
                params.append(left)
            if not self.accept(","):
                break
        self.expect("}")
        if params and equations:
            self.fail("após '|' use só parâmetros (x, y) ou só equações (x + y = 0)")
        names = [n.name for n in (coords if equations else params) if isinstance(n, Var)]
        expected = len(coords) if equations else len(params)
        if len(names) != expected or len(set(names)) != len(names):
            hint = "o vetor deve ter só nomes distintos, como (x, y, z)" if equations \
                else "após '|' liste os parâmetros, como: x, y"
            self.fail(hint)
        if equations:
            return ImplicitSubspace(tuple(names), tuple(equations))
        return ParamSubspace(tuple(names), coords)

    # ----- expressões: expr = termo (+|- termo)*, termo = unário (* unário)* -----

    def expr(self) -> Node:
        node = self.term()
        while self.at("+") or self.at("-"):
            op = self.take().text
            node = BinOp(op, node, self.term())
        return node

    def term(self) -> Node:
        node = self.unary()
        while self.accept("*"):
            node = BinOp("*", node, self.unary())
        return node

    def unary(self) -> Node:
        if self.accept("-"):
            return Neg(self.unary())
        if self.accept("+"):
            return self.unary()
        return self.primary()

    def primary(self) -> Node:
        t = self.take()
        if t.kind == "num":
            node = Num(self.number(t.text))
            nxt = self.peek()
            if nxt is not None and nxt.kind == "name" and nxt.pos == t.end:
                return BinOp("*", node, self.primary())  # multiplicação implícita: 12x
            return node
        if t.kind == "name":
            return self.call_or_var(t.text)
        if t.text in "([":
            self.i -= 1
            if t.text == "[":
                return self.matrix()
            items = self.parens()
            return items[0] if len(items) == 1 else Vec(items)
        self.fail(f"símbolo inesperado '{t.text}'")

    def number(self, text: str) -> Fraction:
        num, _, den = text.partition("/")
        try:
            return Fraction(num) / Fraction(den or 1)
        except ZeroDivisionError:
            self.fail(f"divisão por zero em '{text}'")

    def call_or_var(self, name: str) -> Node:
        if name in KEYWORDS:
            self.fail(f"'{name}' só pode aparecer no início da linha")
        if not self.at("("):
            if name in FUNCTIONS:
                self.fail(f"'{name}' é função: use {name}(...)")
            return Var(name)
        args = self.parens()
        if name in FUNCTIONS:
            lo, hi = FUNCTIONS[name]
            if not lo <= len(args) <= hi:
                qtd = str(lo) if lo == hi else f"pelo menos {lo}"
                self.fail(f"{name} recebe {qtd} argumento(s), recebeu {len(args)}")
        elif name in self.transforms:
            n = self.transforms[name]
            if len(args) not in (1, n):
                self.fail(f"{name} recebe {n} coordenadas ou 1 vetor, recebeu {len(args)}")
        else:
            self.fail(f"função desconhecida '{name}' (para multiplicar, use {name}*(...))")
        return Call(name, args)

    def comma_exprs(self) -> tuple[Node, ...]:
        items = [self.expr()]
        while self.accept(","):
            items.append(self.expr())
        return tuple(items)

    def parens(self) -> tuple[Node, ...]:
        self.expect("(")
        items = self.comma_exprs()
        self.expect(")")
        return items

    def matrix(self) -> Mat:
        self.expect("[")
        rows = []
        while True:
            self.expect("[")
            rows.append(self.comma_exprs())
            self.expect("]")
            if not self.accept(","):
                break
        self.expect("]")
        return Mat(tuple(rows))


def parse_program(text: str) -> Program:
    """Lê o programa inteiro. Trechos suspeitos nunca entram no cálculo."""
    lines: list[Line] = []
    warnings: list[str] = []
    transforms: dict[str, int] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if is_suspicious(stripped):
            short = stripped if len(stripped) <= 60 else stripped[:57] + "..."
            warnings.append(f'linha {number}: instrução suspeita ignorada: "{short}"')
        if stripped.startswith("##"):
            title = stripped[2:].strip()
            if title and not is_suspicious(title):
                lines.append(Line(number, stripped, Title(title)))
            continue
        code = stripped.partition("#")[0].strip()
        if code and not is_suspicious(code):
            lines.append(Line(number, code, _Parser(code, number, transforms).parse()))
    return Program(lines, warnings)