"""Espaços vetoriais: espaço gerado, independência linear, bases, coordenadas e subespaços."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction

from lae.exact import Number, to_fraction
from lae.linear_systems import null_space, rank, rref, solve
from lae.matrices import from_columns, matrix
from lae.vectors import Vector, is_zero


@dataclass(frozen=True)
class Affine:
    """Expressão afim c1*p1 + ... + cn*pn + const, como 12x ou x + z - 1.

    Os coeficientes ficam num dicionário {nome: coeficiente}; nomes ausentes valem 0.
    """

    coefs: dict[str, Number] = field(default_factory=dict)
    const: Number = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefs", {k: to_fraction(v) for k, v in self.coefs.items()})
        object.__setattr__(self, "const", to_fraction(self.const))

    def coef(self, name: str) -> Fraction:
        """Coeficiente da variável `name` (0 se ela não aparece)."""
        return self.coefs.get(name, Fraction(0))


@dataclass
class SubspaceResult:
    """Resultado da verificação de subespaço."""

    is_subspace: bool
    reason: str                 # justificativa curta
    generators: list[Vector]    # vetores que geram o conjunto
    basis: list[Vector]         # base (vazia se não for subespaço ou se for {0})
    dimension: int | None       # None quando não é subespaço


def _check_names(allowed: Sequence[str], exprs: Sequence[Affine]) -> None:
    """Garante que as expressões só usam variáveis declaradas."""
    extra = {k for e in exprs for k in e.coefs} - set(allowed)
    if extra:
        raise ValueError(f"variáveis não declaradas: {sorted(extra)}")


def span_basis(vectors: Sequence[Vector]) -> list[Vector]:
    """Base do espaço gerado: os vetores originais que ocupam colunas pivô."""
    if not vectors:
        return []
    return [vectors[p] for p in rref(from_columns(vectors)).pivots]


def is_independent(vectors: Sequence[Vector]) -> bool:
    """Linearmente independentes se o posto for igual à quantidade de vetores."""
    return not vectors or rank(from_columns(vectors)) == len(vectors)


def coordinates(v: Vector, basis: Sequence[Vector]) -> Vector:
    """Coordenadas de v na base dada: os escalares c com v = c1*b1 + ... + cn*bn."""
    if not basis or not is_independent(basis):
        raise ValueError("os vetores dados não formam uma base (são dependentes)")
    sol = solve(from_columns(basis), v)
    if sol.kind == "SI":
        raise ValueError("o vetor não pertence ao espaço gerado pela base")
    return sol.particular


def check_parametric(params: Sequence[str], coords: Sequence[Affine]) -> SubspaceResult:
    """W = {(f1, ..., fn) | parâmetros reais}, com cada fi afim nos parâmetros.

    Cada parâmetro contribui com um gerador (seus coeficientes). Se houver termo
    constante, W só é subespaço quando o vetor nulo ainda pertence a ele.
    """
    _check_names(params, coords)
    gens = [tuple(c.coef(p) for c in coords) for p in params]
    offset = tuple(c.const for c in coords)
    if not is_zero(offset):
        target = tuple(-x for x in offset)
        if not gens or solve(from_columns(gens), target).kind == "SI":
            return SubspaceResult(False, "o vetor nulo não pertence ao conjunto", gens, [], None)
    basis = span_basis(gens)
    return SubspaceResult(True, "é o espaço gerado pelos vetores dos parâmetros", gens, basis, len(basis))


def check_implicit(names: Sequence[str], equations: Sequence[Affine]) -> SubspaceResult:
    """W = {(x1, ..., xn) | equações afins = 0}.

    Com termo constante, o vetor nulo não satisfaz a equação: não é subespaço.
    Sem ele, W é o núcleo de um sistema homogêneo, sempre subespaço.
    """
    _check_names(names, equations)
    if any(eq.const != 0 for eq in equations):
        return SubspaceResult(False, "o vetor nulo não satisfaz as equações", [], [], None)
    n = len(names)
    if not equations:
        basis = [tuple(Fraction(int(i == j)) for j in range(n)) for i in range(n)]
    else:
        basis = null_space(matrix([[eq.coef(x) for x in names] for eq in equations]))
    return SubspaceResult(True, "é o conjunto solução de um sistema homogêneo", basis, basis, len(basis))
