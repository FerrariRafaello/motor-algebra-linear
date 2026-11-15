"""Testes de lae.transformations: fórmula -> matriz, aplicação, núcleo e imagem."""
import pytest

from lae import matrices as mt
from lae.transformations import evaluate, from_formula, image, kernel
from lae.vector_spaces import Affine
from lae.vectors import DimensionError, vector

# T(x, y, z) = (x + z, y + z) , a transformação do exercício 6
T6 = from_formula(["x", "y", "z"], [Affine({"x": 1, "z": 1}), Affine({"y": 1, "z": 1})])


class TestMontagem:
    def test_matriz_canonica_exercicio_6(self):
        assert T6.matrix == mt.matrix([[1, 0, 1], [0, 1, 1]])
        assert (T6.domain, T6.codomain) == (3, 2)

    def test_variavel_ausente_vira_coluna_zero(self):
        t = from_formula(["x", "y"], [Affine({"x": 2}), Affine({"x": -1})])
        assert t.matrix == mt.matrix([[2, 0], [-1, 0]])

    def test_termo_constante_nao_e_linear(self):
        with pytest.raises(ValueError, match="não é linear"):
            from_formula(["x"], [Affine({"x": 1}, 1)])

    def test_variavel_nao_declarada(self):
        with pytest.raises(ValueError):
            from_formula(["x"], [Affine({"w": 1})])


class TestAplicacao:
    def test_exercicio_6(self):
        assert evaluate(T6, vector([1, 2, 10])) == (11, 12)

    def test_vetor_nulo_vai_no_nulo(self):
        assert evaluate(T6, vector([0, 0, 0])) == (0, 0)

    def test_dimensao_errada(self):
        with pytest.raises(DimensionError):
            evaluate(T6, vector([1, 2]))


class TestNucleoEImagem:
    def test_nucleo_e_imagem_exercicio_6(self):
        nuc, img = kernel(T6), image(T6)
        assert nuc == [(-1, -1, 1)]
        assert evaluate(T6, nuc[0]) == (0, 0)
        assert len(img) == 2

    def test_teorema_do_nucleo_e_imagem(self):
        # dim N(T) + dim Im(T) = dimensão do domínio
        t = from_formula(["x", "y", "z"], [Affine({"x": 1, "y": 2}), Affine({"x": 2, "y": 4})])
        assert len(kernel(t)) + len(image(t)) == t.domain
        assert len(image(t)) == 1
