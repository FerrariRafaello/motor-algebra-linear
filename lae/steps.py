"""Texto de saída: resultado curto (modo resolver) ou passo a passo (modo explicar).

Tudo em texto simples: raiz(205), x^2, matrizes com colunas alinhadas.
"""
from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

from lae.determinants import det, minor
from lae.exact import approx, fmt, sqrt
from lae.linear_systems import KINDS, Rref, Solution
from lae.matrices import fmt_matrix
from lae.parser import (
    Assign, BinOp, Call, Define, ImplicitSubspace, Mat, Neg, Node, Num,
    ParamSubspace, Show, Solve, Title, Var, Vec,
)
from lae.solver import Engine, Outcome, Report, kind, simplify
from lae.transformations import Transformation
from lae.vector_spaces import SubspaceResult
from lae.vectors import dot, fmt_vector, sub

Checker = Callable[[Outcome], list[str]]


# ---------- formatação de valores ----------

def fmt_value(value: object) -> str:
    """Texto de qualquer valor do motor (pode ter várias linhas, no caso de matrizes)."""
    k = kind(value)
    if k == "número":
        return fmt(value)
    if k == "raiz":
        return f"{value}  (aprox. {approx(value)})"
    if k == "vetor":
        return fmt_vector(value)
    if k == "matriz":
        return fmt_matrix(value)
    if isinstance(value, bool):
        return "sim (linearmente independentes)" if value else "não (linearmente dependentes)"
    if isinstance(value, list):
        return _set(value)
    if isinstance(value, Rref):
        return fmt_matrix(value.matrix)
    if isinstance(value, Transformation):
        return f"T: R^{value.domain} -> R^{value.codomain}, matriz canônica:\n{fmt_matrix(value.matrix)}"
    if isinstance(value, SubspaceResult):
        return subspace_text(value)
    if isinstance(value, Solution):
        return solution_text(value, [f"x{i + 1}" for i in range(len(value.augmented[0]) - 1)])
    return str(value)


def _set(vectors: list) -> str:
    return "{" + ", ".join(fmt_vector(v) for v in vectors) + "}" if vectors else "{0} (só o vetor nulo)"


def _eq(label: str, value: object) -> list[str]:
    """'label = valor', com matrizes em várias linhas recuadas."""
    text = fmt_value(value)
    if "\n" not in text:
        return [f"{label} = {text}"]
    indent = " " * (len(label) - len(label.lstrip()) + 2)
    return [f"{label} ="] + [indent + line for line in text.splitlines()]


def signed(values: list[Fraction]) -> str:
    """Soma escrita com sinais: [6, -15, 0] -> '6 - 15 + 0'."""
    text = fmt(values[0])
    for v in values[1:]:
        text += f" {'-' if v < 0 else '+'} {fmt(abs(v), paren=True)}"
    return text


def linear_text(coefs: list[Fraction], names: list[str]) -> str:
    """Combinação escrita como no caderno: [3, -1, 0] com (x, y, z) -> '3x - y'."""
    text = ""
    for c, n in zip(coefs, names):
        if c == 0:
            continue
        k = "" if abs(c) == 1 else fmt(abs(c), paren=True)
        sign = ("-" if c < 0 else "") if not text else (" - " if c < 0 else " + ")
        text += f"{sign}{k}{n}"
    return text or "0"


def solution_text(sol: Solution, names: list[str]) -> str:
    if sol.kind == "SI":
        return "sem solução (sistema impossível)"
    if sol.kind == "SPD":
        return ", ".join(f"{n} = {fmt(v)}" for n, v in zip(names, sol.particular))
    free = [names[f] for f in sol.free]
    terms = " + ".join(f"{n}*{fmt_vector(d)}" for n, d in zip(free, sol.directions))
    return f"({', '.join(names)}) = {fmt_vector(sol.particular)} + {terms}, com {', '.join(free)} livre(s)"


def subspace_text(r: SubspaceResult, n: int | None = None) -> str:
    if not r.is_subspace:
        return f"não é subespaço: {r.reason}"
    where = f" de R^{n}" if n else ""
    return f"é subespaço{where}; dimensão = {r.dimension}; base = {_set(r.basis) if r.basis else '{} (subespaço nulo)'}"


def unparse(node: Node) -> str:
    """Árvore -> texto da mini-linguagem, para nomear passos intermediários."""
    match node:
        case Num(v):
            return fmt(v)
        case Var(name):
            return name
        case Vec(items):
            return "(" + ", ".join(map(unparse, items)) + ")"
        case Mat(rows):
            return "[" + ", ".join("[" + ", ".join(map(unparse, r)) + "]" for r in rows) + "]"
        case Neg(x):
            return "-" + _wrap(x)
        case BinOp("*", Num(v), Var(name)):
            return f"{fmt(v, paren=True)}{name}"
        case BinOp("*", left, right):
            return f"{_wrap(left)}*{_wrap(right)}"
        case BinOp(op, left, right):
            r = _wrap(right) if op == "-" else unparse(right)
            return f"{unparse(left)} {op} {r}"
        case Call(name, args):
            return f"{name}({', '.join(map(unparse, args))})"
    return "?"


def _wrap(node: Node) -> str:
    text = unparse(node)
    return f"({text})" if isinstance(node, BinOp) and node.op in "+-" else text


# ---------- explicações ----------

def _terms(node: Node, sign: int = 1) -> list[tuple[int, Node]]:
    """Separa a soma em parcelas com sinal: 2u + 3v - 5w -> [(+,2u), (+,3v), (-,5w)]."""
    match node:
        case BinOp("+", left, right):
            return _terms(left, sign) + _terms(right, sign)
        case BinOp("-", left, right):
            return _terms(left, sign) + _terms(right, -sign)
        case Neg(x):
            return _terms(x, -sign)
    return [(sign, node)]


def _products(a: tuple, b: tuple) -> str:
    return " + ".join(f"{fmt(x, paren=True)}*{fmt(y, paren=True)}" for x, y in zip(a, b))


def _explain_term(node: Node, value: object, scope: Engine) -> list[str]:
    """Mostra como uma parcela foi calculada."""
    label = unparse(node)
    if not isinstance(node, BinOp) or node.op != "*":
        return _eq(label, value)
    a, b = simplify(scope.eval(node.left)), simplify(scope.eval(node.right))
    ka, kb = kind(a), kind(b)
    if ka == "número" and kb == "vetor":
        each = ", ".join(f"{fmt(a, paren=True)}*{fmt(x, paren=True)}" for x in b)
        return [f"{label} = {fmt(a, paren=True)}*{fmt_vector(b)} = ({each}) = {fmt_vector(value)}"]
    if ka == "número" and kb == "matriz":
        return [f"{label}: multiplica cada elemento por {fmt(a)}"] + _eq(f"  {label}", value)
    if ka == "matriz" and kb in ("matriz", "vetor"):
        cols = list(zip(*b)) if kb == "matriz" else [b]
        lines = [f"{label}: elemento (i, j) = linha i de {unparse(node.left)} . coluna j de {unparse(node.right)}"]
        for i, row in enumerate(a, start=1):
            for j, col in enumerate(cols, start=1):
                pos = f"{i}{j}" if kb == "matriz" else f"{i}"
                lines.append(f"  c{pos} = {_products(row, col)} = {fmt(dot(row, col))}")
        return lines + _eq(f"  {label}", value)
    return _eq(label, value)


def _explain_sum(label: str, node: Node, scope: Engine, value: object) -> list[str]:
    terms = _terms(node)
    parts = [(s, simplify(scope.eval(t))) for s, t in terms]
    lines: list[str] = []
    for (_, t), (_, v) in zip(terms, parts):
        if not isinstance(t, (Num, Var, Vec, Mat)):
            lines += _explain_term(t, v, scope)
    if len(parts) > 1:
        k = kind(value)
        if k == "número":
            lines.append(f"{label} = {signed([s * v for s, v in parts])}")
        elif k == "vetor":
            coords = [signed([s * v[i] for s, v in parts]) for i in range(len(value))]
            lines.append(f"Somando coordenada a coordenada: ({', '.join(coords)})")
        elif k == "matriz":
            lines.append("Somando elemento a elemento:")
            for i, row in enumerate(value):
                for j, x in enumerate(row):
                    soma = signed([s * v[i][j] for s, v in parts])
                    lines.append(f"  [{i + 1},{j + 1}]: {soma} = {fmt(x)}")
    return lines


def _norm_line(label: str, v: tuple) -> str:
    total = dot(v, v)
    squares = " + ".join(f"{fmt(x, paren=True)}^2" for x in v)
    text = f"|{label}| = raiz({squares}) = raiz({signed([x * x for x in v])}) = raiz({fmt(total)})"
    root = str(sqrt(total))
    return text + (f" = {root}" if root != f"raiz({fmt(total)})" else "")


def _explain_metric(name: str, args: tuple[Node, ...], scope: Engine) -> list[str]:
    vals = [scope.eval(a) for a in args]
    labels = [_wrap(a) for a in args]
    if name == "norma":
        return ["Fórmula: |v| = raiz(v . v)", _norm_line(labels[0], vals[0])]
    u, v = vals
    lu, lv = labels
    if name == "dist":
        diff = sub(u, v)
        coords = ", ".join(f"{fmt(a)} - {fmt(b, paren=True)}" for a, b in zip(u, v))
        return [f"Fórmula: d({lu}, {lv}) = |{lu} - {lv}|",
                f"{lu} - {lv} = ({coords}) = {fmt_vector(diff)}",
                _norm_line(f"{lu} - {lv}", diff)]
    prods = [a * b for a, b in zip(u, v)]
    dot_line = f"{lu} . {lv} = {_products(u, v)} = {signed(prods)} = {fmt(sum(prods))}"
    if name == "dot":
        return [f"Fórmula: {lu} . {lv} = soma dos produtos das coordenadas", dot_line]
    nu, nv = sqrt(dot(u, u)), sqrt(dot(v, v))
    lines = ["Fórmula: cos(t) = (u . v) / (|u| * |v|)", dot_line, _norm_line(lu, u), _norm_line(lv, v)]
    sub_line = f"cos(t) = {fmt(sum(prods))} / ({nu} * {nv})"
    if nu.is_rational and nv.is_rational:
        sub_line += f" = {fmt(sum(prods))} / {fmt(nu.as_fraction() * nv.as_fraction())}"
    return lines + [sub_line]


def _explain_det(arg: Node, scope: Engine) -> list[str]:
    a = scope.eval(arg)
    lines: list[str] = []
    if isinstance(arg, Call) and arg.name == "transposta":
        lines += [f"Transposta: as linhas de {unparse(arg.args[0])} viram colunas"]
        lines += _eq(f"  {unparse(arg)}", a)
    n = len(a)
    if n == 1:
        return lines + [f"det = {fmt(a[0][0])}"]
    if n == 2:
        (p, q), (r, s) = a
        return lines + [f"det = a11*a22 - a12*a21 = {fmt(p, paren=True)}*{fmt(s, paren=True)} - "
                        f"{fmt(q, paren=True)}*{fmt(r, paren=True)} = {signed([p * s, -q * r])}"]
    if n > 3:
        return lines + ["det calculado por eliminação de Gauss (produto dos pivôs, com sinal das trocas)"]
    lines.append("Expansão por cofatores na 1a linha: det = a11*det(M11) - a12*det(M12) + a13*det(M13)")
    minors = [det(minor(a, 0, j)) for j in range(3)]
    for j in range(3):
        (p, q), (r, s) = minor(a, 0, j)
        lines.append(f"  det(M1{j + 1}) = {fmt(p, paren=True)}*{fmt(s, paren=True)} - "
                     f"{fmt(q, paren=True)}*{fmt(r, paren=True)} = {fmt(minors[j])}")
    terms = [(-1) ** j * a[0][j] * minors[j] for j in range(3)]
    expansion = " - ".join(f"{fmt(a[0][j], paren=True)}*{fmt(minors[j], paren=True)}" for j in range(2))
    lines.append(f"det = {expansion} + {fmt(a[0][2], paren=True)}*{fmt(minors[2], paren=True)}")
    lines.append(f"det = {signed(terms)}")
    return lines


def _explain_apply(t: Transformation, args: tuple[Node, ...], scope: Engine) -> list[str]:
    vals = [simplify(scope.eval(x)) for x in args]
    v = vals[0] if len(vals) == 1 and kind(vals[0]) == "vetor" else vals
    formula = ", ".join(linear_text(list(row), t.variables) for row in t.matrix)
    subst = ", ".join(f"{n} = {fmt(x)}" for n, x in zip(t.variables, v))
    rows = []
    for row in t.matrix:
        terms = [(c, x) for c, x in zip(row, v) if c != 0]
        text = ""
        for c, x in terms:
            piece = fmt(x, paren=True) if abs(c) == 1 else f"{fmt(abs(c), paren=True)}*{fmt(x, paren=True)}"
            text += (("-" if c < 0 else "") if not text else (" - " if c < 0 else " + ")) + piece
        rows.append(text or "0")
    return [f"Substituindo {subst} em ({formula}):", f"  = ({', '.join(rows)})"]


def _explain_show(label: str, node: Node, scope: Engine, value: object) -> list[str]:
    if isinstance(node, Call):
        if node.name in ("norma", "dist", "cos", "dot"):
            return _explain_metric(node.name, node.args, scope)
        if node.name == "det":
            return _explain_det(node.args[0], scope)
        if node.name in scope.transforms:
            return _explain_apply(scope.transforms[node.name], node.args, scope)
        return []
    return _explain_sum(label, node, scope, value)


def _explain_solve(cmd: Solve, sol: Solution) -> list[str]:
    names = list(cmd.unknowns)
    lines = [f"Sistema nas incógnitas {', '.join(names)}:"]
    lines += [f"  {linear_text(list(row[:-1]), names)} = {fmt(row[-1])}" for row in sol.augmented]
    lines += ["Matriz aumentada [A | b] (última coluna = b):"]
    lines += ["  " + ln for ln in fmt_matrix(sol.augmented).splitlines()]
    lines.append("Escalonamento (Gauss-Jordan):")
    for op, m in sol.reduced.history:
        lines += [f"  {op}"] + ["    " + ln for ln in fmt_matrix(m).splitlines()]
    lines.append(f"Posto de A = {sol.rank_a}, posto de [A | b] = {sol.rank_aug}, incógnitas = {len(names)}")
    lines.append(f"Classificação: {sol.kind}, {KINDS[sol.kind]}")
    return lines


def _explain_param(cmd: ParamSubspace, r: SubspaceResult, scope: Engine) -> list[str]:
    params = list(cmd.params)
    offset = tuple(scope.affine(c, params).const for c in cmd.coords)
    generic = "(" + ", ".join(unparse(c) for c in cmd.coords) + ")"
    combo = " + ".join(f"{p}*{fmt_vector(g)}" for p, g in zip(params, r.generators))
    has_offset = any(offset)
    lines = ["Vetor genérico como combinação dos parâmetros:",
             f"  {generic} = " + (f"{fmt_vector(offset)} + " if has_offset else "") + combo]
    if not r.is_subspace:
        return lines + ["O termo constante impede que algum valor dos parâmetros dê o vetor nulo,",
                        "e todo subespaço precisa conter o vetor nulo."]
    if has_offset:
        lines.append("O termo constante está no espaço gerado, então o conjunto é o próprio espaço gerado.")
    else:
        lines.append(f"Com {' = '.join(params)} = 0 obtemos o vetor nulo, que portanto pertence ao conjunto.")
    lines += ["Todo conjunto da forma {combinações de geradores} é fechado para soma e para",
              "multiplicação por escalar (somar ou escalar combinações dá outra combinação).",
              f"Geradores: {_set(r.generators)}; posto da matriz deles = {len(r.basis)}"]
    if len(r.basis) < len(r.generators):
        lines.append("Há geradores dependentes; a base fica só com os independentes.")
    else:
        lines.append("Posto = número de geradores: são linearmente independentes, formam uma base.")
    return lines


def _explain_implicit(cmd: ImplicitSubspace, r: SubspaceResult, scope: Engine) -> list[str]:
    names = list(cmd.names)
    eqs = [scope.affine(BinOp("-", left, right), names) for left, right in cmd.equations]
    lines = ["Equações com tudo de um lado:"]
    lines += [f"  {linear_text([e.coef(n) for n in names], names)} = {fmt(-e.const)}" for e in eqs]
    if not r.is_subspace:
        return lines + ["Lado direito diferente de 0: o vetor nulo não satisfaz as equações."]
    return lines + ["Sistema homogêneo: o vetor nulo é solução, e soma ou múltiplo de soluções",
                    "continua sendo solução. O conjunto é o núcleo do sistema.",
                    "Base: uma solução para cada variável livre do escalonamento."]


def _explain(o: Outcome) -> list[str]:
    cmd, value, scope = o.line.command, o.value, o.scope
    match cmd:
        case Assign(name, Num() | Vec() | Mat()):
            first, *rest = _eq(name, value)
            return [f"Dado: {first}"] + [f"      {line}" for line in rest]
        case Assign(name, expr):
            body, label = _explain_show(name, expr, scope, value), name
        case Show(expr):
            body, label = _explain_show(o.line.source, expr, scope, value), o.line.source
        case Solve():
            body = _explain_solve(cmd, value)
            result = [f"Resultado: {solution_text(value, list(cmd.unknowns))}"]
            return [f">> {o.line.source}"] + ["  " + ln for ln in body + result]
        case Define(name, params):
            formula = ", ".join(linear_text(list(row), list(params)) for row in value.matrix)
            body = [f"{name}: R^{value.domain} -> R^{value.codomain}, {name}({', '.join(params)}) = ({formula})",
                    "Matriz canônica (linha i = coeficientes da coordenada i):"]
            body += ["  " + ln for ln in fmt_matrix(value.matrix).splitlines()]
            return [f">> {o.line.source}"] + ["  " + ln for ln in body]
        case ParamSubspace() | ImplicitSubspace():
            n = len(cmd.coords) if isinstance(cmd, ParamSubspace) else len(cmd.names)
            body = (_explain_param(cmd, value, scope) if isinstance(cmd, ParamSubspace)
                    else _explain_implicit(cmd, value, scope))
            result = [f"Resultado: {subspace_text(value, n)}"]
            return [f">> {o.line.source}"] + ["  " + ln for ln in body + result]
        case _:
            return []
    result = ["Resultado: " + ln if i == 0 else "  " + ln for i, ln in enumerate(_eq(label, value))]
    return [f">> {o.line.source}"] + ["  " + ln for ln in body + result]


def _short(o: Outcome) -> list[str]:
    cmd, value = o.line.command, o.value
    match cmd:
        case Assign(name):
            return _eq(name, value)
        case Show():
            return _eq(o.line.source, value)
        case Solve(unknowns):
            return [f"{o.line.source}  ->  {solution_text(value, list(unknowns))}"]
        case ParamSubspace(coords=coords):
            return [f"{o.line.source}  ->  {subspace_text(value, len(coords))}"]
        case ImplicitSubspace(names):
            return [f"{o.line.source}  ->  {subspace_text(value, len(names))}"]
    return [o.line.source]


def render(report: Report, explain: bool = True, checks: Checker | None = None) -> str:
    """Monta o texto final. `checks` (do validator) acrescenta a verificação no modo explicar."""
    out: list[str] = []
    if report.warnings:
        out += ["AVISOS (trechos ignorados no cálculo):"] + [f"  {w}" for w in report.warnings] + [""]
    for o in report.outcomes:
        if isinstance(o.line.command, Title):
            out += ["", f"===== {o.line.command.text} =====", ""]
        elif o.error:
            out += [f"ERRO {o.error}"] + ([""] if explain else [])
        elif not explain:
            out += _short(o)
        else:
            lines = _explain(o)
            verified = checks(o) if checks else []
            if verified:
                lines += ["  Verificação:"] + [f"    {c}" for c in verified]
            out += lines + ([""] if lines and lines[0].startswith(">>") else [])
    return "\n".join(out).strip("\n") + "\n"