"""Operações com vetores de R^n usando aritmética exata."""
from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from lae.exact import Number, Root, fmt, sqrt, to_fraction

Vector = tuple[Fraction, ...] # ... significa que a tupla pode ter qualquer tamanho


class DimensionError(ValueError):
    """Operação entre objetos de dimensões incompatíveis."""


def vector(values: Sequence[Number]) -> Vector:
    """Cria um vetor exato a partir de números: vector([3, '1/2', -2.5])"""
    if len(values) == 0:
        raise DimensionError("vetor precisa ter pelo menos uma coordenada")
    return tuple(to_fraction(x) for x in values)


def check_same_dim(*vectors: Vector) -> int:
    """Confere que todos os vetores têm a mesma dimensão e a devolve."""
    dims = {len(v) for v in vectors}
    if len(dims) != 1:
        raise DimensionError(f"dimensões incompatíveis: {sorted(dims)}")
    return dims.pop()


def add(u: Vector, v: Vector) -> Vector:
    """Soma coordenada a coordenada: u + v."""
    check_same_dim(u, v)
    return tuple(a + b for a, b in zip(u, v))


def sub(u: Vector, v: Vector) -> Vector:
    """Subtração coordenada a coordenada: u -v."""
    check_same_dim(u, v)
    return tuple(a - b for a, b in zip(u, v))


def scale(k: Number, v: Vector) -> Vector:
    """Multiplicação por escalar: k * v."""
    k = to_fraction(k)
    return tuple(k * a for a in v)


def linear_combination(coefs: Sequence[Number], vectors: Sequence[Vector]) -> Vector:
    """Combinação linear c1*v1 + c2*v2 + ... + cn*vn."""
    if len(coefs) != len(vectors) or not vectors:
        raise ValueError("é preciso um coeficiente para cada vetor")
    check_same_dim(*vectors)
    result = scale(coefs[0], vectors[0])
    for c, v in zip(coefs[1:], vectors[1:]):
        result = add(result, scale(c, v))
    return result


def dot(u: Vector, v: Vector) -> Fraction:
    """Produto escalar: u . v = u1*v1 + ... + un*vn."""
    check_same_dim(u, v)
    return sum((a*b for a, b in zip(u,v)), Fraction(0))


def is_zero(v: Vector) -> bool:
    """True se todas as coordenadas forem zero."""
    return all(a == 0 for a in v)


def norm(v: Vector) -> Root:
    """Norma euclidiana exata: |v| = raiz(v . v)."""
    return sqrt(dot(v, v))


def distance(u: Vector, v: Vector) -> Root:
    """Distância entre vetores: d(u, v) = |u - v|."""
    return norm(sub(u, v))


def cosine(u: Vector, v: Vector) -> Root:
    """Cosseno do ângulo entre u e v: (u . v) / (|u| * |v|)."""
    if is_zero(u) or is_zero(v):
        raise ValueError("ângulo indefinido: um dos vetores é nulo")
    return Root(dot(u, v)) / (norm(u) * norm(v))


def fmt_vector(v: Vector) -> str:
    """Formata um vetor como texto: (6, 3, -8)"""
    return "(" + ", ".join(fmt(a) for a in v) + ")"
