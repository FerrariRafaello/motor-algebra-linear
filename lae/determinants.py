"""Determinantes por dois métodos independentes: eliminação e cofatores"""
from __future__ import annotations

from fractions import Fraction

from lae.matrices import Matrix, shape
from lae.vectors import DimensionError


def check_square(a: Matrix) -> int:
    """Confere que a matriz é quadrada e devolve sua ordem n."""
    rows, cols = shape(a)
    if rows != cols:
        raise DimensionError(f"determinante exige matriz quadrada, recebida {rows}x{cols}")
    return rows


def minor(a: Matrix, i: int, j: int) -> Matrix:
    """Submatriz obtida removendo a linha i e a coluna j."""
    return tuple(
        tuple(x for c, x in enumerate(row) if c != j)
        for r, row in enumerate(a) if r != i
    )


def det_cofactor(a: Matrix) -> Fraction:
    """Determinante por expansão de cofatores na primeira linha (Laplace).
    
    Custo cresce como n!, então serve para matrizes pequenas e para validação.
    """
    n = check_square(a)
    if n == 1:
        return a[0][0]
    return sum(
        ((-1) ** j * a[0][j] * det_cofactor(minor(a, 0, j)) for j in range(n)),
        Fraction(0),
    )


def det_elimination(a: Matrix) -> Fraction:
    """Determinante por eliminação de Gauss: produto dos pivôs, trocando o sinal a cada troca de linhas."""
    n = check_square(a)
    m = [list(row) for row in a]
    sign, result = 1, Fraction(1)
    for c in range(n):
        p = next((r for r in range(c, n) if m[r][c] != 0), None)
        if p is None:
            return Fraction(0) # a coluna sem pivô: matriz singular
        if p != c:
            m[c], m[p] = m[p], m[c]
            sign = -sign
        for r in range(c + 1, n):
            f = m[r][c] / m[c][c]
            m[r] = [x - f * y for x, y in zip(m[r], m[c])]
        result *= m[c][c]
    return sign * result


def det(a: Matrix) -> Fraction:
    """Determinante (método padrão: eliminação, eficiente para qualquer n)."""
    return det_elimination(a)


def is_singular(a: Matrix) -> bool:
    """True se a matriz não tem inversa (determinante zero)."""
    return det(a) == 0
