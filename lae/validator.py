"""Validação independente: confere cada resultado por outro caminho.

Cada verificação devolve linhas "[ok] ..." ou "[FALHOU] ...", que o steps.py
mostra no fim da resolução. Uma falha indica erro no próprio motor.
"""
from __future__ import annotations

from fractions import Fraction

from lae.determinants import det_cofactor, det_elimination
from lae.exact import fmt
from lae.linear_systems import rank, solve
from lae.matrices import from_columns
from lae.parser import (
    Assign, BinOp, Call, Define, ImplicitSubspace, Mat, Neg, Node, Num,
    ParamSubspace, Show, Solve, Var, Vec,
)
from lae.solver import Engine, Outcome, kind, simplify
from lae.steps import fmt_value, signed, unparse
from lae.transformations import evaluate
from lae.vectors import dot, linear_combination, vector

TOLERANCE = 1e-9  # diferença máxima aceita na conferência em ponto flutuante


def mark(ok: bool, text: str) -> str:
    return f"[ok] {text}" if ok else f"[FALHOU] {text}"


# ---------- conferência em ponto flutuante (aritmética independente) ----------

class _Unsupported(Exception):
    """Expressão fora do que a conferência em float sabe recalcular."""


def _to_float(value: object) -> object:
    if isinstance(value, tuple):
        return [_to_float(x) for x in value]
    if isinstance(value, Fraction):
        return float(value)
    raise _Unsupported


def _f_add(a: object, b: object, sign: int) -> object:
    if isinstance(a, list):
        return [_f_add(x, y, sign) for x, y in zip(a, b)]
    return a + sign * b


def _f_mul(a: object, b: object) -> object:
    if isinstance(b, float) and not isinstance(a, float):
        a, b = b, a
    if isinstance(a, float):
        return a * b if isinstance(b, float) else [_f_mul(a, x) for x in b]
    if isinstance(b[0], list):  # matriz * matriz
        return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]
    return [sum(x * y for x, y in zip(row, b)) for row in a]  # matriz * vetor


def float_eval(node: Node, scope: Engine) -> object:
    match node:
        case Num(v):
            return float(v)
        case Var(name):
            return _to_float(simplify(scope.lookup(name)))
        case Vec(items):
            return [float_eval(i, scope) for i in items]
        case Mat(rows):
            return [[float_eval(x, scope) for x in r] for r in rows]
        case Neg(x):
            return _f_mul(-1.0, float_eval(x, scope))
        case BinOp("*", left, right):
            return _f_mul(float_eval(left, scope), float_eval(right, scope))
        case BinOp(op, left, right):
            return _f_add(float_eval(left, scope), float_eval(right, scope), 1 if op == "+" else -1)
    raise _Unsupported


def _close(exact: object, approx: object) -> bool:
    if isinstance(exact, tuple):
        return len(exact) == len(approx) and all(_close(x, y) for x, y in zip(exact, approx))
    return abs(float(exact) - approx) <= TOLERANCE * max(1.0, abs(approx))


# ---------- verificações por tipo de problema ----------

def _check_metric(name: str, args: tuple[Node, ...], scope: Engine, value: object) -> list[str]:
    vals = [scope.eval(a) for a in args]
    if name == "norma":
        sq = dot(vals[0], vals[0])
        return [mark(simplify(value * value) == sq, f"|v|^2 = v . v = {fmt(sq)}")]
    u, v = vals
    uv, uu, vv = dot(u, v), dot(u, u), dot(v, v)
    if name == "dot":
        return [mark(dot(v, u) == uv, f"comutatividade: v . u = u . v = {fmt(uv)}")]
    if name == "dist":
        d2 = simplify(value * value)
        return [mark(d2 == uu + vv - 2 * uv,
                     f"lei dos cossenos: |u - v|^2 = |u|^2 + |v|^2 - 2(u . v)  ->  "
                     f"{fmt(d2)} = {signed([uu, vv, -2 * uv])}")]
    c2 = simplify(value * value)
    same_sign = (float(value) > 0) == (uv > 0) and (float(value) < 0) == (uv < 0)
    return [mark(c2 <= 1, "-1 <= cos(t) <= 1"),
            mark(same_sign, "o sinal de cos(t) é o mesmo de u . v"),
            mark(c2 == uv * uv / (uu * vv),
                 f"cos^2 = (u . v)^2 / (|u|^2 * |v|^2)  ->  {fmt(c2)} = {fmt(uv * uv)}/{fmt(uu * vv)}")]


def _check_det(arg: Node, scope: Engine, value: Fraction) -> list[str]:
    a = scope.eval(arg)
    lines = [mark(det_cofactor(a) == det_elimination(a) == value,
                  f"cofatores e eliminação de Gauss dão o mesmo valor: {fmt(value)}")]
    if isinstance(arg, Call) and arg.name == "transposta":
        original = scope.eval(arg.args[0])
        lines.append(mark(det_elimination(original) == value,
                          f"propriedade det(M) = det(M^T): det({unparse(arg.args[0])}) = {fmt(value)}"))
    return lines


def _check_apply(name: str, args: tuple[Node, ...], scope: Engine, value: tuple) -> list[str]:
    t = scope.transforms[name]
    vals = [simplify(scope.eval(x)) for x in args]
    v = vals[0] if len(vals) == 1 and kind(vals[0]) == "vetor" else vector(vals)
    columns = list(zip(*t.matrix))
    combo = linear_combination(list(v), columns)
    return [mark(combo == value, f"linearidade: {name}(v) = soma de v_i * {name}(e_i) = {fmt_value(combo)}")]


def _check_expression(node: Node, scope: Engine, value: object) -> list[str]:
    if isinstance(node, Call):
        if node.name in ("norma", "dist", "cos", "dot"):
            return _check_metric(node.name, node.args, scope, value)
        if node.name == "det":
            return _check_det(node.args[0], scope, value)
        if node.name in scope.transforms:
            return _check_apply(node.name, node.args, scope, value)
        return []
    if isinstance(node, (Num, Vec, Mat)):
        return []
    try:
        approx = float_eval(node, scope)
    except _Unsupported:
        return []
    return [mark(_close(value, approx), f"recalculado em ponto flutuante (tolerância {TOLERANCE:g})")]


def _check_solve(cmd: Solve, scope: Engine, sol) -> list[str]:
    if sol.kind == "SI":
        return [mark(sol.rank_a < sol.rank_aug,
                     f"Rouché-Capelli: posto(A) = {sol.rank_a} < posto([A | b]) = {sol.rank_aug}")]
    candidates = [sol.particular]
    candidates += [tuple(p + d for p, d in zip(sol.particular, dv)) for dv in sol.directions]
    lines = []
    for x in candidates:
        test = scope.snapshot()
        test.vars.update(zip(cmd.unknowns, x))
        left, right = simplify(test.eval(cmd.left)), simplify(test.eval(cmd.right))
        subst = ", ".join(f"{n} = {fmt(v)}" for n, v in zip(cmd.unknowns, x))
        lines.append(mark(left == right, f"substituindo {subst}: os dois lados dão {fmt_value(left)}"))
    return lines


def _member(test: Engine, params: tuple[str, ...], coords: tuple[Node, ...], values: list[int]) -> tuple:
    test.vars.update(zip(params, (Fraction(x) for x in values)))
    return tuple(simplify(test.eval(c)) for c in coords)


def _check_param(cmd: ParamSubspace, scope: Engine, r) -> list[str]:
    offset = tuple(scope.affine(c, cmd.params).const for c in cmd.coords)
    target = tuple(-x for x in offset)
    zero_ok = bool(r.generators) and solve(from_columns(r.generators), target).kind != "SI"
    if not r.is_subspace:
        return [mark(not zero_ok, "nenhuma escolha dos parâmetros produz o vetor nulo")]
    k = len(cmd.params)
    test = scope.snapshot()
    m1 = _member(test, cmd.params, cmd.coords, [1] * k)
    m2 = _member(test, cmd.params, cmd.coords, [i + 2 for i in range(k)])

    def in_span(w: tuple) -> bool:
        return not r.basis or solve(from_columns(r.basis), w).kind != "SI"

    total, triple = tuple(a + b for a, b in zip(m1, m2)), tuple(3 * a for a in m1)
    return [mark(zero_ok, "o vetor nulo pertence ao conjunto"),
            mark(in_span(total) and in_span(triple),
                 f"fechamento: {fmt_value(m1)} + {fmt_value(m2)} e 3*{fmt_value(m1)} continuam no conjunto"),
            mark(rank(from_columns(r.basis)) == r.dimension == len(r.basis),
                 f"a base tem {len(r.basis)} vetores linearmente independentes")]


def _check_implicit(cmd: ImplicitSubspace, scope: Engine, r) -> list[str]:
    eqs = [scope.affine(BinOp("-", left, right), cmd.names) for left, right in cmd.equations]
    if not r.is_subspace:
        return [mark(any(e.const != 0 for e in eqs), "com o vetor nulo, alguma equação dá constante != 0")]
    satisfied = all(sum(e.coef(n) * x for n, x in zip(cmd.names, b)) == 0 for e in eqs for b in r.basis)
    coefs = [[e.coef(n) for n in cmd.names] for e in eqs]
    rk = rank(tuple(tuple(row) for row in coefs)) if coefs else 0
    n = len(cmd.names)
    return [mark(satisfied, "cada vetor da base satisfaz as equações"),
            mark(r.dimension == n - rk, f"dimensão = n - posto = {n} - {rk} = {n - rk}")]


def check(o: Outcome) -> list[str]:
    """Verificações da linha; lista vazia quando não há o que conferir."""
    if o.error:
        return []
    cmd, scope, value = o.line.command, o.scope, o.value
    match cmd:
        case Assign(_, expr) | Show(expr):
            return _check_expression(expr, scope, value)
        case Solve():
            return _check_solve(cmd, scope, value)
        case Define(name):
            zero = evaluate(value, tuple(Fraction(0) for _ in range(value.domain)))
            return [mark(all(x == 0 for x in zero), f"{name}(0) = {fmt_value(zero)}: leva o vetor nulo no vetor nulo")]
        case ParamSubspace():
            return _check_param(cmd, scope, value)
        case ImplicitSubspace():
            return _check_implicit(cmd, scope, value)
    return []