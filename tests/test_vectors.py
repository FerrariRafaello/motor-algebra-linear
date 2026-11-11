"""Testes de lae.vectors: operações, dimensões e casos especiais."""
from fractions import Fraction

import pytest

from lae.vectors import (
    DimensionError, add, cosine, distance, dot, fmt_vector, is_zero,
    linear_combination, norm, scale, sub, vector
)


class TestCriacao:
    def test_aceita_inteiros_fracoes_e_decimais(self):
        assert vector([3, "1/2", "-2.5"]) == (3, Fraction(1, 2), Fraction(-5, 2))

    def test_vetor_vazio(self):
        with pytest.raises(DimensionError):
            vector([])

    def test_formatacao(self):
        assert fmt_vector(vector([6, "-1/3", 0])) == "(6, -1/3, 0)"


class TestOperacoes:
    def test_soma_e_subtracao(self):
        u, v = vector([1, 2]), vector([3,-5])
        assert add(u, v) == (4, -3)
        assert sub(u, v) == (-2, 7)

    def test_escalar_fracionario(self):
        assert scale("1/2", vector([4, -3])) == (2, Fraction(-3, 2))

    def test_dimensoes_incompativeis(self):
        with pytest.raises(DimensionError):
            add(vector([1, 2]), vector([1, 2, 3]))
        with pytest.raises(DimensionError):
            dot(vector([1]), vector([1, 2]))

    def test_combinacao_linear_exercicio_1(self):
        u, v, w = vector([3, 0, 1]), vector([-5, 1, 2]), vector([0, 1, 4])
        assert linear_combination([2,3, -5], [u, v, w]) == (-9, -2, -12)

    def test_combinacao_com_numero_erraro_de_coeficientes(self):
        with pytest.raises(ValueError):
            linear_combination([1, 2], [vector([1, 0])])


class TestMetricas:
    def test_norma_racional_e_irracional(self):
        assert norm(vector([6, 0, -8])).as_fraction() == 10
        assert str(norm(vector([1, 1]))) == "raiz(2)"
 
    def test_distancia_exercicio_3(self):
        assert str(distance(vector([6, 3, 6]), vector([6, 0, -8]))) == "raiz(205)"
 
    def test_cosseno_exercicio_3(self):
        cos = cosine(vector([6, 3, 6]), vector([6, 0, -8]))
        assert cos.as_fraction() == Fraction(-2, 15)
 
    def test_cosseno_irracional(self):
        assert str(cosine(vector([1, 0]), vector([1, 1]))) == "raiz(2)/2"
 
    def test_vetor_nulo(self):
        zero = vector([0, 0, 0])
        assert is_zero(zero)
        assert norm(zero).as_fraction() == 0
        with pytest.raises(ValueError):
            cosine(zero, vector([1, 2, 3]))
