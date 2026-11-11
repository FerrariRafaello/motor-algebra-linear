"""Testes de lae.matrices: criação, operações, dimensões e formatação."""
from fractions import Fraction

import pytest

from lae import matrices as mt
from lae.vectors import DimensionError, vector


class TestCriacao:
    def test_fracoes_e_negativos(self):
        a = mt.matrix([[1, "-1/2"], ["0.25", -3]])
        assert a == ((1, Fraction(-1, 2)), (Fraction(1, 4), -3))
        assert mt.shape(a) == (2, 2)

    def test_matriz_irregular_ou_vazia(self):
        with pytest.raises(DimensionError):
            mt.matrix([[1, 2], [3]])
        with pytest.raises(DimensionError):
            mt.matrix([])

    def test_a_partir_de_colunas(self):
        cols = [vector([3, 4]), vector([2, 3])]
        assert mt.from_columns(cols) == mt.matrix([[3, 2], [4, 3]])


class TestOperacoes:
    def test_transposta_nao_quadrada(self):
        a = mt.matrix([[1, 2, 3], [4, 5, 6]])
        assert mt.transpose(a) == mt.matrix([[1, 4], [2, 5], [3, 6]])

    def test_soma_dimensoes_incompativeis(self):
        with pytest.raises(DimensionError):
            mt.add(mt.matrix([[1, 2]]), mt.matrix([[1], [2]]))

    def test_produto_nao_quadrado(self):
        a = mt.matrix([[1, 2, 3], [4, 5, 6]])
        b = mt.matrix([[1, 0], [0, 1], [1, 1]])
        assert mt.mul(a, b) == mt.matrix([[4, 5], [10, 11]])

    def test_produto_incompativel(self):
        with pytest.raises(DimensionError):
            mt.mul(mt.matrix([[1, 2]]), mt.matrix([[1, 2]]))

    def test_produto_nao_comutativo(self):
        a, b = mt.matrix([[1, 2], [0, 1]]), mt.matrix([[1, 0], [3, 1]])
        assert mt.mul(a, b) != mt.mul(b, a)

    def test_expressao_exercicio_7(self):
        a = mt.matrix([[3, 7], [8, 4]])
        b = mt.matrix([[2, 10], [0, 41]])
        c = mt.add(mt.scale(2, a), mt.mul(a, b))
        assert c == mt.matrix([[12, 331], [32, 252]])

    def test_matriz_vezes_vetor_exercicio_6(self):
        t = mt.matrix([[1, 0, 1], [0, 1, 1]])
        assert mt.apply(t, vector([1, 2, 10])) == (11, 12)
        with pytest.raises(DimensionError):
            mt.apply(t, vector([1, 2]))


class TestFormatacao:
    def test_colunas_alinhadas(self):
        texto = mt.fmt_matrix(mt.matrix([[12, 331], [32, 252]]))
        assert texto == "| 12  331 |\n| 32  252 |"

    def test_alinhamento_com_negativos_e_fracoes(self):
        texto = mt.fmt_matrix(mt.matrix([[-2, "1/3"], [10, 4]]))
        assert texto == "| -2  1/3 |\n| 10    4 |"