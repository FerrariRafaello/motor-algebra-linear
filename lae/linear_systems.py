"""Escalonamento (Gauss-Jordan), posto, núcleo e resolução de sistemas lineares."""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from lae.exact import fmt
from lae.matrices import Matrix, shape
from lae.vectors import DimensionError, Vector

KINDS = {
    "SPD": "sistema possível e determinado (solução única)",
    "SPI": "sistema possível e indeterminado (infinitas soluções)",
    "SI": "sistema impossível (sem solução)",
}


@dataclass
class Rref:
    """Forma escalonada reduzida, colunas pivô e histórico das operações."""

    matrix: Matrix
    pivots: list[int]
    history: list[tuple[str, Matrix]] = field(default_factory=list)


@dataclass
class Solution:
    """Resultado de um sistema A x = b, com tudo que o passo a passo precisa."""

    kind: str                     # "SPD", "SPI" ou "SI"
    augmented: Matrix             # matriz aumentada [A | b]
    reduced: Rref                 # escalonamento da aumentada
    rank_a: int                   # posto de A
    rank_aug: int                 # posto de [A | b]
    particular: Vector | None     # uma solução (None se SI)
    directions: list[Vector]      # base do núcleo: solução geral = particular + t1*d1 + ...
    free: list[int]               # índices das variáveis livres


def _combine_text(i: int, f: Fraction, j: int) -> str:
    """Texto da operação Li = Li - f*Lj, já com o sinal ajustado."""
    op = "-" if f > 0 else "+"
    k = "" if abs(f) == 1 else f"{fmt(abs(f), paren=True)}*"
    return f"L{i + 1} = L{i + 1} {op} {k}L{j + 1}"


def rref(a: Matrix) -> Rref:
    """Escalona até a forma escalonada reduzida, registrando cada operação."""
    rows, cols = shape(a)
    m = [list(row) for row in a]
    pivots: list[int] = []
    history: list[tuple[str, Matrix]] = []

    def snap() -> Matrix:
        """Foto imutável do estado atual da matriz."""
        return tuple(tuple(row) for row in m)

    r = 0
    for c in range(cols):
        if r == rows:
            break
        p = next((i for i in range(r, rows) if m[i][c] != 0), None)
        if p is None:
            continue
        if p != r:
            m[r], m[p] = m[p], m[r]
            history.append((f"L{r + 1} <-> L{p + 1}", snap()))
        piv = m[r][c]
        if piv != 1:
            m[r] = [x / piv for x in m[r]]
            history.append((f"L{r + 1} = {fmt(1 / piv, paren=True)}*L{r + 1}", snap()))
        for i in range(rows):
            f = m[i][c]
            if i != r and f != 0:
                m[i] = [x - f * y for x, y in zip(m[i], m[r])]
                history.append((_combine_text(i, f, r), snap()))
        pivots.append(c)
        r += 1
    return Rref(snap(), pivots, history)


def rank(a: Matrix) -> int:
    """Posto: número de pivôs na forma escalonada."""
    return len(rref(a).pivots)


def _kernel(reduced: Matrix, pivots: list[int], n: int) -> list[Vector]:
    """Base do núcleo a partir da forma reduzida: um vetor por variável livre."""
    basis = []
    for f in (j for j in range(n) if j not in pivots):
        v = [Fraction(0)] * n
        v[f] = Fraction(1)
        for row, pc in enumerate(pivots):
            v[pc] = -reduced[row][f]
        basis.append(tuple(v))
    return basis


def null_space(a: Matrix) -> list[Vector]:
    """Base do núcleo de A (soluções de A x = 0)."""
    red = rref(a)
    return _kernel(red.matrix, red.pivots, shape(a)[1])


def solve(a: Matrix, b: Vector) -> Solution:
    """Resolve A x = b por Gauss-Jordan na matriz aumentada e classifica o sistema
    pelo teorema de Rouché-Capelli (comparando posto de A e de [A | b])."""
    rows, n = shape(a)
    if len(b) != rows:
        raise DimensionError(f"A tem {rows} linhas, mas b tem {len(b)} coordenadas")
    aug = tuple(row + (bi,) for row, bi in zip(a, b))
    red = rref(aug)
    rank_aug = len(red.pivots)
    pivots = [p for p in red.pivots if p < n]
    if n in red.pivots:
        return Solution("SI", aug, red, len(pivots), rank_aug, None, [], [])
    x = [Fraction(0)] * n
    for row, pc in enumerate(pivots):
        x[pc] = red.matrix[row][n]
    free = [j for j in range(n) if j not in pivots]
    kind = "SPD" if not free else "SPI"
    return Solution(kind, aug, red, len(pivots), rank_aug, tuple(x), _kernel(red.matrix, pivots, n), free)