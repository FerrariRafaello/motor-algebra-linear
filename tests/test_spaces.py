"""Testes de lae.vector_spaces: independência, bases, coordenadas e subespaços."""
from fractions import Fraction

import pytest

from lae.vector_spaces import (
    Affine, check_implicit, check_parametric, coordinates, is_independent, span_basis
)
from lae.vectors import vector


class TestIndependencia:
    def test_independentes_e_dependentes(self):
        assert is_independent([vector([1, 0]), vector([0, 1])])
        assert not is_independent([vector([0, 0, 0]), vector([1, 2, 3])])

    def test_conjunto_com_vetor_nulo_e_dependente(self):
        assert not is_independent([vector([0, 0, 0]), vector([1, 2, 3])])

    def test_base_do_gerado_descarta_redundantes(self):
        vs = [vector([1, 0, 1]), vector([2, 0, 2]), vector([0, 1, 0])]
        assert span_basis(vs) == [vs[0], vs[2]]


class TestCoordenadas:
    def test_coordenadas_exercicio_4(self):
        base = [vector([-2, 1]), vector([5, 2])]
        assert coordinates(vector([-3, -2]), base) == (Fraction(-4, 9), Fraction(-7, 9))

    def test_vetores_dependentes_nao_formam_base(self):
        with pytest.raises(ValueError):
            coordinates(vector([1, 1]), [vector([1, 1]), vector([2, 2])])

    def test_vetor_fora_do_gerado(self):
        with pytest.raises(ValueError):
            coordinates(vector([0, 0, 1]), [vector([1, 0, 0])])


class TestSubespacoParametrico:
    def test_exercicio_5(self):
        # H2 = {(x, 12x, y) | x, y, reais}
        coords = [Affine({"x": 1}), Affine({"x": 12}), Affine({"y": 1})]
        r = check_parametric(["x", "y"], coords)
        assert r.is_subspace
        assert r.dimension == 2
        assert r.basis == [(1, 12, 0), (0, 0, 1)]

    def test_termo_constante_exclui_o_nulo(self):
        # {(x, x + 1) | x real}: reta não passa pela origem
        r = check_parametric(["x"], [Affine({"x": 1}), Affine({"x": 1}, 1)])
        assert not r.is_subspace
        assert r.dimension is None

    def test_constante_que_ainda_contem_o_nulo(self):
        # {(x + 1, y) + x, y reais} é o próprio R^2
        r = check_parametric(["x", "y"], [Affine({"x": 1}, 1), Affine({"y": 1})])
        assert r.is_subspace and r.dimension == 2

    def test_parametros_redundantes(self):
        # {(x + y, 2x + 2y) | x, y}: dois parâmetros, mas dimensão 1
        r = check_parametric(["x", "y"], [Affine({"x": 1, "y": 1}), Affine({"x": 2, "y": 2})])
        assert r.dimension == 1

    def test_variavel_nao_declarada(self):
        with pytest.raises(ValueError):
            check_parametric(["x"], [Affine({"z": 1})])


class TestSubespacoImplicito:
    def test_plano_pela_origem(self):
        # {(x, y, z) | x + y - z = 0}
        r = check_implicit(["x", "y", "z"], [Affine({"x": 1, "y": 1, "z": -1})])
        assert r.is_subspace and r.dimension == 2

    def test_plano_fora_da_origem(self):
        r = check_implicit(["x", "y", "z"], [Affine({"x": 1}, -3)])
        assert not r.is_subspace

    def test_subespaco_nulo(self):
        eqs = [Affine({"x": 1}), Affine({"y": 1})]
        r = check_implicit(["x", "y"], eqs)
        assert r.is_subspace and r.dimension == 0 and r.basis == []
