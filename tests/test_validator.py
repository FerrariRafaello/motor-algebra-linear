"""Testes de lae.validator: aprova resultados certos e acusa resultados adulterados."""
from dataclasses import replace
from fractions import Fraction

import pytest

from lae.exact import sqrt
from lae.solver import solve_text
from lae.steps import render
from lae.validator import check

CASOS = [
    "u = (3, 0, 1)\nv = (-5, 1, 2)\nw = (0, 1, 4)\n2u + 3v - 5w",
    "resolver x, y: (-6, 7) = x*(3, 4) + y*(2, 3)",
    "u = (6, 3, 6)\nv = (6, 0, -8)\ndist(u, v)",
    "u = (6, 3, 6)\nv = (6, 0, -8)\ncos(u, v)",
    "subespaco {(x, 12x, y) | x, y}",
    "T(x, y, z) = (x + z, y + z)\nT(1, 2, 10)",
    "A = [[3, 7], [8, 4]]\nB = [[2, 10], [0, 41]]\n2A + A*B",
    "A = [[-2, 4, 2], [-2, 3, 1], [-1, 2, 0]]\ndet(transposta(A))",
    "resolver x, y: (1, 2) = x*(1, 2) + y*(2, 4)",
    "resolver x, y: (1, 2) = x*(1, 1) + y*(1, 1)",
    "subespaco {(x, x + 1) | x}",
    "subespaco {(x, y, z) | x + y - z = 0}",
]


def ultima(texto):
    return solve_text(texto).outcomes[-1]


@pytest.mark.parametrize("texto", CASOS)
def test_resultados_corretos_passam(texto):
    linhas = check(ultima(texto))
    assert linhas, "deveria haver pelo menos uma verificação"
    assert all(ln.startswith("[ok]") for ln in linhas), linhas


@pytest.mark.parametrize("texto, errado", [
    (CASOS[0], (Fraction(-9), Fraction(-2), Fraction(-11))),
    (CASOS[2], sqrt(204)),
    (CASOS[3], Fraction(2, 15)),
    (CASOS[5], (Fraction(11), Fraction(13))),
    (CASOS[6], ((Fraction(12), Fraction(331)), (Fraction(32), Fraction(253)))),
    (CASOS[7], Fraction(2)),
])
def test_resultado_adulterado_e_acusado(texto, errado):
    adulterado = replace(ultima(texto), value=errado)
    assert any(ln.startswith("[FALHOU]") for ln in check(adulterado))


def test_solucao_adulterada_e_acusada():
    o = ultima(CASOS[1])
    falsa = replace(o.value, particular=(Fraction(-32), Fraction(44)))
    assert check(replace(o, value=falsa))[0].startswith("[FALHOU]")


def test_mensagens_principais():
    assert "205 = 81 + 100 + 24" in check(ultima(CASOS[2]))[0]
    assert "det(A) = -2" in check(ultima(CASOS[7]))[1]


def test_linhas_sem_verificacao():
    assert check(ultima("u = (1, 2)")) == []           # dado literal
    assert check(ultima("det((1, 2))")) == []          # linha com erro


def test_verificacao_aparece_no_passo_a_passo():
    saida = render(solve_text(CASOS[2]), checks=check)
    assert "  Verificação:\n    [ok] lei dos cossenos" in saida