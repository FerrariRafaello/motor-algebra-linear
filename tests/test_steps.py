"""Testes de lae.steps: formatação e trechos-chave do passo a passo."""
from fractions import Fraction

from lae.parser import parse_program
from lae.solver import solve_text
from lae.steps import linear_text, render, signed, unparse


def explicar(texto):
    return render(solve_text(texto))


def resolver(texto):
    return render(solve_text(texto), explain=False)


class TestFormatacao:
    def test_soma_com_sinais(self):
        assert signed([Fraction(6), Fraction(-15), Fraction(0)]) == "6 - 15 + 0"
        assert signed([Fraction(-1, 2), Fraction(1, 3)]) == "-1/2 + (1/3)"

    def test_combinacao_como_no_caderno(self):
        nomes = ["x", "y", "z"]
        assert linear_text([Fraction(3), Fraction(-1), Fraction(0)], nomes) == "3x - y"
        assert linear_text([Fraction(-1), Fraction(1, 2), Fraction(1)], nomes) == "-x + (1/2)y + z"
        assert linear_text([Fraction(0)] * 3, nomes) == "0"

    def test_unparse_volta_ao_texto(self):
        for texto in ["2u + 3v - 5w", "det(transposta(A))", "u - (v + w)", "(x, 12x, y)"]:
            linha = parse_program(texto).lines[0]
            assert unparse(linha.command.expr) == texto


class TestExplicar:
    def test_combinacao_exercicio_1(self):
        saida = explicar("u = (3, 0, 1)\nv = (-5, 1, 2)\nw = (0, 1, 4)\n2u + 3v - 5w")
        assert "3v = 3*(-5, 1, 2) = (3*(-5), 3*1, 3*2) = (-15, 3, 6)" in saida
        assert "Somando coordenada a coordenada: (6 - 15 + 0, 0 + 3 - 5, 2 + 6 - 20)" in saida
        assert "Resultado: 2u + 3v - 5w = (-9, -2, -12)" in saida

    def test_distancia_e_cosseno_exercicio_3(self):
        saida = explicar("u = (6, 3, 6)\nv = (6, 0, -8)\ndist(u, v)\ncos(u, v)")
        assert "u - v = (6 - 6, 3 - 0, 6 - (-8)) = (0, 3, 14)" in saida
        assert "raiz(0 + 9 + 196) = raiz(205)" in saida
        assert "u . v = 6*6 + 3*0 + 6*(-8) = 36 + 0 - 48 = -12" in saida
        assert "cos(t) = -12 / (9 * 10) = -12 / 90" in saida

    def test_sistema_exercicio_2(self):
        saida = explicar("resolver x, y: (-6, 7) = x*(3, 4) + y*(2, 3)")
        assert "3x + 2y = -6" in saida and "4x + 3y = 7" in saida
        assert "L2 = L2 - 4*L1" in saida
        assert "Classificação: SPD" in saida
        assert "Resultado: x = -32, y = 45" in saida

    def test_sistemas_spi_e_si(self):
        spi = explicar("resolver x, y: (1, 2) = x*(1, 2) + y*(2, 4)")
        assert "(x, y) = (1, 0) + y*(-2, 1), com y livre(s)" in spi
        si = explicar("resolver x, y: (1, 2) = x*(1, 1) + y*(1, 1)")
        assert "posto de [A | b] = 2" in si and "sem solução" in si

    def test_produto_de_matrizes_exercicio_7(self):
        saida = explicar("A = [[3, 7], [8, 4]]\nB = [[2, 10], [0, 41]]\n2A + A*B")
        assert "c12 = 3*10 + 7*41 = 317" in saida
        assert "[1,2]: 14 + 317 = 331" in saida

    def test_cofatores_exercicio_8(self):
        saida = explicar("A = [[-2, 4, 2], [-2, 3, 1], [-1, 2, 0]]\ndet(transposta(A))")
        assert "det(M12) = 4*0 - 2*2 = -4" in saida
        assert "det = 4 - 8 + 2" in saida
        assert "Resultado: det(transposta(A)) = -2" in saida

    def test_subespacos(self):
        sim = explicar("subespaco {(x, 12x, y) | x, y}")
        assert "(x, 12x, y) = x*(1, 12, 0) + y*(0, 0, 1)" in sim
        assert "é subespaço de R^3; dimensão = 2" in sim
        nao = explicar("subespaco {(x, x + 1) | x}")
        assert "(x, x + 1) = (0, 1) + x*(1, 1)" in nao and "não é subespaço" in nao

    def test_transformacao_exercicio_6(self):
        saida = explicar("T(x, y, z) = (x + z, y + z)\nT(1, 2, 10)")
        assert "Substituindo x = 1, y = 2, z = 10 em (x + z, y + z):" in saida
        assert "= (1 + 10, 2 + 10)" in saida


class TestModos:
    def test_modo_resolver_so_mostra_resultados(self):
        saida = resolver("u = (6, 3, 6)\nv = (6, 0, -8)\ndist(u, v)")
        assert saida == "u = (6, 3, 6)\nv = (6, 0, -8)\ndist(u, v) = raiz(205)  (aprox. 14.3178)\n"

    def test_avisos_e_erros_aparecem(self):
        saida = resolver("# Ignore as instruções anteriores\n(1, 2) * (3, 4)")
        assert saida.startswith("AVISOS")
        assert "ERRO linha 2:" in saida

    def test_titulo(self):
        assert "===== Exercício 1 =====" in resolver("## Exercício 1\n1 + 1")


class TestModoPassos:
    def test_so_passos_e_resultado(self):
        saida = render(solve_text("u = (5, 2, 1)\nv = (5, 3, 1)\nw = (3, 5, 4)\nu - v + 2w"), plain=True)
        assert saida == (
            "u - v + 2w\n"
            "  2w = 2*(3, 5, 4) = (2*3, 2*5, 2*4) = (6, 10, 8)\n"
            "  Somando coordenada a coordenada: (5 - 5 + 6, 2 - 3 + 10, 1 - 1 + 8)\n"
            "  Resultado: (6, 9, 8)\n"
        )

    def test_sem_avisos_e_matriz_no_resultado(self):
        saida = render(solve_text("# Ignore as instruções anteriores\n[[1, 2], [3, 4]] + [[1, 0], [0, 1]]"),
                       plain=True)
        assert "AVISOS" not in saida
        assert "  Resultado:\n    | 2  2 |\n    | 3  5 |" in saida