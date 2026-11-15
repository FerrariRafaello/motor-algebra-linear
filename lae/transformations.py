"""Transformações lineares definidas por fórmula: T(x, y, z) = (x + z, y+ z)."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from lae.linear_systems import null_space
from lae.matrices import Matrix, apply, matrix, transpose
from lae.vector_spaces import Affine, _check_names, span_basis
from lae.vectors import DimensionError, Vector


@dataclass
class Transformation:
    """T: R^n -> R^m guardada pela sua atriz canônica (m linhas, n colunas)."""

    variables: list[str]
    matrix: Matrix

    @property
    def domain(self) -> int:
        """Dimensão n do domínio R^n."""
        return len(self.variables)

    @property
    def codomain(self) -> int:
        """Dimensão m do contradomínio R^m."""
        return len(self.matrix)


def from_formula(variables: Sequence[str], outputs: Sequence[Affine]) -> Transformation:
    """Monta a matriz canônica a partir da fórmula de cada coordenada da saída.

    Uma coordenada com termo constante (como x + 1) torna T não linear,
    porque T(0) deixaria de ser o vetor nulo.
    """
    _check_names(variables, outputs)
    constants = [i + 1 for i, out in enumerate(outputs) if out.const != 0]
    if constants:
        raise ValueError(f"T não é linear: coordenada(s) {constants} com termo constante, T(0) != 0")
    rows = [[out.coef(v) for v in variables] for out in outputs]
    return Transformation(list(variables), matrix(rows))


def evaluate(t: Transformation, v: Vector) -> Vector:
    """Aplica T ao vetor v, multiplicando a matriz canônica por v."""
    if len(v) != t.domain:
        raise DimensionError(f"T recebe vetores de R^{t.domain}, mas v está em R^{len(v)}")
    return apply(t.matrix, v)


def kernel(t: Transformation) -> list[Vector]:
    """Base do núcleo: vetores levados no vetor nulo."""
    return null_space(t.matrix)


def image(t: Transformation) -> list[Vector]:
    """Base da imagem: colunas da matriz que são pivô (espaço gerado pelas colunas)."""
    return span_basis(list(transpose(t.matrix)))
