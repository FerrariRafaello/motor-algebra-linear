"""Operações com matrizes usando aritmética exata."""
from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from lae.exact import Number, fmt, to_fraction
from lae.vectors import DimensionError, Vector, check_same_dim, dot

Matrix = tuple[tuple[Fraction, ...], ...]


def matrix(rows: Sequence[Sequence[Number]]) -> Matrix:
    """Cria uma matriz exata a partir de linhas: matrix([[3, 7], [8, 4]])."""
    if len(rows) == 0 or len(rows[0]) == 0:
        raise DimensionError("matriz precisa ter pelo menos uma linha e uma coluna")
    if len({len(r) for r in rows}) != 1:
        raise DimensionError("todas as linhas da matriz devem ter o mesmo tamanho")
    return tuple(tuple(to_fraction(x) for x in r) for r in rows)


def shape(a: Matrix) -> tuple[int, int]:
    """Dimensões (linhas, colunas) da matriz."""
    return len(a), len(a[0])


def from_columns(columns: Sequence[Vector]) -> Matrix:
    """Monta a matriz cujas colunas são vetores dados."""
    check_same_dim(*columns)
    return tuple(zip(*columns))


def transpose(a: Matrix) -> Matrix:
    """Transposta: as linhas viram colunas."""
    return tuple(zip(*a))


def add(a: Matrix, b: Matrix) -> Matrix:
    """Soma elemento a elemento: A + B (exige mesmas dimensões)."""
    if shape(a) != shape(b):
        raise DimensionError(f"soma exige mesmas dimensões: {shape(a)} e {shape(b)}")
    return tuple(tuple(x + y for x, y in zip(ra, rb)) for ra, rb in zip(a, b))


def scale(k: Number, a: Matrix) -> Matrix:
    """Multiplicação por escalar: k * A."""
    k = to_fraction(k)
    return tuple(tuple(k * x for x in r) for r in a)


def mul(a: Matrix, b: Matrix) -> Matrix:
    """Produto matricial A * B: elemento (i, j) = linha i de A . coluna j de B."""
    if shape(a)[1] != shape(b)[0]:
        raise DimensionError(f"produto exige colunas de A = linhas de B: {shape(a)} e {shape(b)}")
    cols = transpose(b)
    return tuple(tuple(dot(r, c) for c in cols) for r in a)


def apply(a: Matrix, v: Vector) -> Vector:
    """Produto matriz por vetor: A * v."""
    if shape(a)[1] != len(v):
        raise DimensionError(f"matriz {shape(a)} não aplica em vetor de dimensão {len(v)}")
    return tuple(dot(r, v) for r in a)


def fmt_matrix(a: Matrix) -> str:
    """Formata a matriz em linhas e colunas alinhadas à direita:
    | 12  331 |
    | 32  252 |
    """
    cells = [[fmt(x) for x in r] for r in a]
    widths = [max(len(r[j]) for r in cells) for j in range(len(cells[0]))]
    lines = ("| " + "  ".join(c.rjust(w) for c, w in zip(r, widths)) + " |" for r in cells)
    return "\n".join(lines)
