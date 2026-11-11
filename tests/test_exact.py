"""Teste de lae.exact: conversão, formatação e raízes exata."""
from fractions import Fraction

import pytest

from lae.exact import Root, approx, fmt, sqrt, to_fraction


class TestToFraction:
    def test_inteiro_decimal_e_fracao(self):
        assert to_fraction(3) == 3
        assert to_fraction("-2.5") == Fraction(-5, 2)
        assert to_fraction("1/3") == Fraction(1, 3)
        assert to_fraction(Fraction(2, 4)) == Fraction(1, 2)

    def test_float_sem_erro_binario(self):
        assert to_fraction(0.1) == Fraction(1, 10)

    def test_entradas_invalidas(self):
        with pytest.raises(TypeError):
            to_fraction(True)
        with pytest.raises(ValueError):
            to_fraction("abc")
        with pytest.raises(ValueError):
            to_fraction("1/0")


class TestFmt:
    def test_formato_simples(self):
        assert fmt(Fraction(-2, 15)) == "-2/15"
        assert fmt(7) == "7"

    def test_parenteses_em_substituicao(self):
        assert fmt(-8, paren=True) == "(-8)"
        assert fmt(Fraction(1,3), paren=True) == "(1/3)"
        assert fmt(3, paren=True) == "3"


class TestRoot:
    def test_simplificacao(self):
        assert str(sqrt(205)) == "raiz(205)"
        assert str(sqrt(200)) == "10*raiz(2)"
        assert Root(1, 8) == Root(2, 2)

    def test_raiz_racional(self):
        assert sqrt(81).as_fraction() == 9
        assert sqrt(Fraction(1,4)).as_fraction() == Fraction(1, 2)
        assert str(sqrt(0)) == "0"

    def test_raiz_de_fracao_irracional(self):
        assert str(sqrt(Fraction(1,2))) == "raiz(2)/2"

    def test_negativo_e_irracional_como_fracao(self):
        with pytest.raises(ValueError):
            sqrt(-4)
        with pytest.raises(ValueError):
            sqrt(2).as_fraction()

    def test_produto_e_divisao(self):
        assert (sqrt(2) * sqrt(2)).as_fraction() == 2
        assert str(sqrt(2) * sqrt(3)) == "raiz(6)"
        assert str(1 / sqrt(2)) == "raiz(2)/2"
        assert str(-sqrt(2) * Fraction(3, 2)) == "-3*raiz(2)/2"

    def test_coseno_do_exercicio_3(self):
        # cos = -12 / (|u| * |v|) com |u| = raiz(81), |v| = raiz(100)
        cos = Root(-12) / (sqrt(81) * sqrt(100))
        assert cos.as_fraction() == Fraction(-2, 15)

    def test_divisao_por_zero(self):
        with pytest.raises(ZeroDivisionError):
            sqrt(2) / sqrt(0)

    def test_aproximacao(self):
        assert approx(sqrt(205)) == "14.3178"
        assert approx(Fraction(-2, 15)) == "-0.1333"
