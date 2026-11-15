"""Interface de terminal.

Uso:
    python -m lae.cli explicar exemplos/aps.lae    (passo a passo com verificação)
    python -m lae.cli resolver exemplos/aps.lae    (só os resultados)
    python -m lae.cli resolver                     (lê do teclado; termine com Ctrl+D)
"""
from __future__ import annotations

import argparse
import sys

from lae.parser import ParseError, Title, parse_program
from lae.solver import run
from lae.steps import render
from lae.validator import check


def main(argv: list[str] | None = None) -> int:
    """Executa o motor. Retorna 0 (tudo certo), 1 (erro ou verificação falhou) ou 2 (entrada inválida)."""
    ap = argparse.ArgumentParser(prog="lae", description="Motor de álgebra linear com aritmética exata.")
    ap.add_argument("modo", choices=["resolver", "explicar"],
                    help="resolver: só os resultados; explicar: passo a passo com verificação")
    ap.add_argument("arquivo", nargs="?", default="-",
                    help="arquivo .lae; sem arquivo (ou '-') lê do teclado")
    args = ap.parse_args(argv)

    try:
        if args.arquivo == "-":
            text = sys.stdin.read()
        else:
            with open(args.arquivo, encoding="utf-8") as f:
                text = f.read()
    except OSError as exc:
        print(f"ERRO: não foi possível ler '{args.arquivo}': {exc.strerror}", file=sys.stderr)
        return 2
    try:
        program = parse_program(text)
    except ParseError as exc:
        print(f"ERRO DE SINTAXE {exc}", file=sys.stderr)
        return 2

    report = run(program)
    explain = args.modo == "explicar"
    print(render(report, explain=explain, checks=check if explain else None), end="")

    commands = [o for o in report.outcomes if not isinstance(o.line.command, Title)]
    errors = sum(1 for o in commands if o.error)
    verified = [c for o in commands for c in check(o)]
    failed = sum(1 for c in verified if c.startswith("[FALHOU]"))
    if explain:
        print(f"\nResumo: {len(commands)} comandos, {errors} erro(s), "
              f"{len(verified) - failed} verificações ok, {failed} falha(s)")
    elif failed:
        print(f"\nATENÇÃO: {failed} verificação(ões) falharam; rode no modo explicar para ver quais.")
    return 1 if errors or failed else 0


if __name__ == "__main__":
    sys.exit(main())