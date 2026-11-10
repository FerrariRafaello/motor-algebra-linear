#
# Este módulo é a base de todo o resto. Ele guarda frações exatas e representa raízes de forma simplificada,
# então raiz(200) vira 10*raiz(2) automaticamente. Ele também racionaliza denominadores (1/raiz(2) vira raiz(2)/2)
# e formata números para o passo, colocando negativos entre parênteses em substituições, como 6*(-8).
# Um dos testes já confere o cosseno do exercício 3: -12/(raiz(81)*raiz(100)) dá exatamente -2/15.
# 

"""Aritmética exata: números racionais (Fraction) e raízes quadradas simplificadas."""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

Number = int | float | str | Fraction


def to_fraction(value: Number) -> Fraction:
    """Converte int, float, Fraction ou texto ('3', '-2.5', '1/3') em Fraction exata."""
    if isinstance(value, bool):
        raise TypeError("valor booleano não é número")
    if isinstance(value, float):
        value = repr(value) # 0.1 vira '0.1' e depois 1/10, sem lixo binário
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError, TypeError) as exc:
        raise ValueError(f"número inválido: {value!r}") from exc


def fmt(value: Number, paren: bool = False) -> str:
    """Formata um racional como text ('-2/15'). Com paren=True, negativos e frações saem entre parênteses, para uso em substituições: 6*(-8)."""
    q = to_fraction(value)
    text = str(q)
    if paren and (q < 0 or q.denominator != 1):
        return f"({text})"
    return text


def _split_square(n: int) -> tuple[int, int]:
    """Escreve n como a^2 * b, com b livre de quadrados. Retorna (a, b)."""
    a, b, d = 1, n, 2
    while d * d <= b:
        while b % (d * d) == 0:
            b //= d * d
            a *= d
        d += 1
    return a, b


@dataclass(frozen=True)
class Root:
    """Número real exato da forma coef * raiz(radicand).
    
    Sempre normalizado: radicand é inteiro livre de quadrados (1 = racional), então raiz(200)
    é guardado como 10*raiz(2) e comparações são exatas.
    """

    coef: Fraction
    radicand: int = 1

    def __post_init__(self) -> None:
        coef = to_fraction(self.coef)
        if not isinstance(self.radicand, int) or self.radicand < 0:
            raise ValueError("radicando deve ser inteiro não negativo")
        if coef == 0 or self.radicand == 0:
            coef, a, b = Fraction(0), 1, 1
        else:
            a, b = _split_square(self.radicand)
        object.__setattr__(self, "coef", coef * a)
        object.__setattr__(self, "radicand", b)

    @property
    def is_rational(self) -> bool:
        "True quando o número não tem raiz irracional."
        return self.radicand == 1

    def as_fraction(self) -> Fraction:
        """Devolve o valor como Fraction: falha se for irracional"""
        if not self.is_rational:
            raise ValueError(f"{self} é irracional")
        return self.coef

    def __float__(self) -> float:
        return float(self.coef) * math.sqrt(self.radicand)

    def __neg__(self) -> Root:
        return Root(-self.coef, self.radicand)

    def __mul__(self, other: Root | Number) -> Root:
        other = other if isinstance(other, Root) else Root(to_fraction(other))
        return Root(self.coef * other.coef, self.radicand * other.radicand)

    __rmul__ = __mul__

    def __truediv__(self, other: Root | Number) -> Root:
        other = other if isinstance(other, Root) else Root(to_fraction(other))
        if other.coef == 0:
            raise ZeroDivisionError("divisão por zero")
        # a*raiz(r) / (b*raiz(s)) = a/(b*s) * raiz(r*s) (racionaliza o denominador)
        return Root(self.coef / (other.coef * other.radicand), self.radicand * other.radicand)

    def __rtruediv__(self, other: Number) -> Root:
        return Root(to_fraction(other))/self

    def __str__(self) -> str:
        if self.is_rational:
            return fmt(self.coef)
        num, den = self.coef.numerator, self.coef.denominator
        head = {1: "", -1: "-"}.get(num, f"{num}*")
        text = f"{head}raiz({self.radicand})"
        return text if den == 1 else f"{text}/{den}"


def sqrt(value: Number) -> Root:
    """Raiz quadrada exata de um racional: raiz(p/q) = raiz(p*q)/q."""
    q = to_fraction(value)
    if q < 0:
        raise ValueError(f"raiz de número negativo não é real: {fmt(q)}")
    return Root(Fraction(1, q.denominator), q.numerator * q.denominator)


def approx(value: Root | Number, digits: int = 4) -> str:
    """Valor decimal aproximado, só para exibição: approx(raiz(2025)) = '14.3178'."""
    number = float(value) if isinstance(value, Root) else float(to_fraction(value))
    return f"{number:.{digits}f}"
