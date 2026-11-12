"""Testes de lae.determinants: os dois métodos devem sempre concordar."""
from fractions import Fraction
 
import pytest
 
from lae import matrices as mt
from lae.determinants import det, det_cofactor, det_elimination, is_singular, minor
from lae.vectors import DimensionError
 
 
def ambos(a):
    """Calcula pelos dois métodos, confere que concordam e devolve o valor."""
    d1, d2 = det_elimination(a), det_cofactor(a)
    assert d1 == d2
    return d1
 
 
class TestCasosBasicos:
    def test_ordem_1_e_2(self):
        assert ambos(mt.matrix([[-7]])) == -7
        assert ambos(mt.matrix([[3, 7], [8, 4]])) == 3 * 4 - 7 * 8
 
    def test_fracoes(self):
        assert ambos(mt.matrix([["1/2", "1/3"], ["1/4", "1/5"]])) == Fraction(1, 60)
 
    def test_pivo_zero_exige_troca_de_linhas(self):
        assert ambos(mt.matrix([[0, 1], [1, 0]])) == -1
        assert ambos(mt.matrix([[0, 2, 1], [3, 0, 0], [0, 0, 5]])) == -30
 
    def test_nao_quadrada(self):
        with pytest.raises(DimensionError):
            det(mt.matrix([[1, 2, 3], [4, 5, 6]]))
 
    def test_menor(self):
        a = mt.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert minor(a, 0, 1) == mt.matrix([[4, 6], [7, 9]])
 
 
class TestSingular:
    def test_linhas_proporcionais(self):
        a = mt.matrix([[1, 2, 3], [2, 4, 6], [1, 0, 1]])
        assert ambos(a) == 0
        assert is_singular(a)
 
    def test_coluna_nula(self):
        assert ambos(mt.matrix([[0, 5], [0, 3]])) == 0
 
 
class TestPropriedades:
    A = mt.matrix([[-2, 4, 2], [-2, 3, 1], [-1, 2, 0]])
 
    def test_exercicio_8(self):
        assert ambos(self.A) == -2
        assert ambos(mt.transpose(self.A)) == -2
 
    def test_det_do_produto(self):
        b = mt.matrix([[1, 0, 2], [0, 3, -1], [4, 1, 1]])
        assert det(mt.mul(self.A, b)) == det(self.A) * det(b)
 
    def test_ordem_4(self):
        a = mt.matrix([[2, -1, 0, 3], [1, 0, 4, -2], [0, 5, 1, 1], [-3, 2, 2, 0]])
        assert ambos(a) == det_elimination(mt.transpose(a))
 