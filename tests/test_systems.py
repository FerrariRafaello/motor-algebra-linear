"""Testes de lae.linear_systems: escalonamento, posto, núcleo e classificação."""
from fractions import Fraction

import pytest

from lae import matrices as mt
from lae.linear_systems import null_space, rank, rref, solve
from lae.vectors import DimensionError, vector


def confere(a, sol, b):
    """Toda solução geral (particular + direções) precisa satisfazer A x = b."""
    assert mt.apply(a, sol.particular) == b
    for d in sol.directions:
        assert mt.apply(a, d) == (0,) * len(b)


class TestEscalonamento:
    def test_matriz_invertivel_vira_identidade(self):
        red = rref(mt.matrix([[2, 1], [1, 3]]))
        assert red.matrix == mt.matrix([[1, 0], [0, 1]])
        assert red.pivots == [0, 1]

    def test_historico_de_operacoes(self):
        red = rref(mt.matrix([[0, 1], [2, 4]]))
        ops = [texto for texto, _ in red.history]
        assert ops == ["L1 <-> L2", "L1 = (1/2)*L1", "L1 = L1 - 2*L2"]

    def test_posto(self):
        assert rank(mt.matrix([[1, 2, 3], [2, 4, 6]])) == 1
        assert rank(mt.matrix([[0, 0], [0, 0]])) == 0
        assert rank(mt.matrix([[1, 0, 2], [0, 1, 3]])) == 2

    def test_nucleo(self):
        assert null_space(mt.matrix([[1, 2], [3, 4]])) == []
        assert null_space(mt.matrix([[1, 1, -1]])) == [(-1, 1, 0), (1, 0, 1)]


class TestClassificacao:
    def test_spd_exercicio_2(self):
        # (-6, 7) = x(3, 4) + y(2, 3)
        a = mt.from_columns([vector([3, 4]), vector([2, 3])])
        b = vector([-6, 7])
        sol = solve(a, b)
        assert sol.kind == "SPD"
        assert sol.particular == (-32, 45)
        assert sum(sol.particular) == 13
        confere(a, sol, b)

    def test_spd_com_fracoes_exercicio_4(self):
        # w = a*u + b*v com u = (-2, 1), v = (5, 2), w = (-3, -2)
        a = mt.from_columns([vector([-2, 1]), vector([5, 2])])
        sol = solve(a, vector([-3, -2]))
        assert sol.particular == (Fraction(-4, 9), Fraction(-7, 9))

    def test_sistema_impossivel(self):
        sol = solve(mt.matrix([[1, 1], [1, 1]]), vector([1, 2]))
        assert sol.kind == "SI"
        assert sol.particular is None
        assert (sol.rank_a, sol.rank_aug) == (1, 2)

    def test_infinitas_solucoes(self):
        a = mt.matrix([[1, 1, 1], [2, 2, 2]])
        b = vector([1, 2])
        sol = solve(a, b)
        assert sol.kind == "SPI"
        assert sol.free == [1, 2]
        assert len(sol.directions) == 2
        confere(a, sol, b)

    def test_mais_equacoes_que_incognitas(self):
        a = mt.matrix([[1, 0], [0, 1], [1, 1]])
        assert solve(a, vector([2, 3, 5])).kind == "SPD"
        assert solve(a, vector([2, 3, 6])).kind == "SI"

    def test_homogeneo_e_negativos(self):
        a = mt.matrix([[-1, 2], [2, -4]])
        sol = solve(a, vector([0, 0]))
        assert sol.kind == "SPI"
        confere(a, sol, (0, 0))

    def test_dimensao_de_b_incompativel(self):
        with pytest.raises(DimensionError):
            solve(mt.matrix([[1, 2], [3, 4]]), vector([1, 2, 3]))