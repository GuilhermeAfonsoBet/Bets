#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Divide o Relatorio_BayesGlobal_Mesa_Profissional em 2 PDFs:
1) Operacional: resultado da semana vigente + portfólio da próxima semana
2) Estrutural: todo o restante (metodologia e análises)

Heurística robusta:
- Varre texto por página e seleciona páginas que contenham os headers-alvo:
  - "2. Portfólio otimizado" (regras semana mais recente)
  - "2.B Portfólio sugerido para a próxima semana"
  - "3.1.C Auditoria" e/ou "Tabela — PnL semanal OOS"
- Inclui também a capa (página 1) no PDF operacional para contexto.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader, PdfWriter


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="PDF de entrada (full report)")
    ap.add_argument("--out-weekly", required=True, help="PDF de saída (operacional)")
    ap.add_argument("--out-struct", required=True, help="PDF de saída (estrutural)")
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        raise FileNotFoundError(str(inp))

    r = PdfReader(str(inp))
    n = len(r.pages)

    needles = [
        "2. Portfólio otimizado",
        "2.B Portfólio sugerido para a próxima semana",
        "3.1.C Auditoria",
        "Tabela — PnL semanal OOS",
        "Tabela - PnL semanal OOS",
    ]

    selected = set()
    for i in range(n):
        try:
            txt = (r.pages[i].extract_text() or "").replace("\u00a0", " ")
        except Exception:
            txt = ""
        if any(s in txt for s in needles):
            selected.add(i)

    # sempre incluir capa no operacional
    selected.add(0)

    weekly = PdfWriter()
    struct = PdfWriter()
    for i in range(n):
        if i in selected:
            weekly.add_page(r.pages[i])
        else:
            struct.add_page(r.pages[i])

    out_weekly = Path(args.out_weekly)
    out_struct = Path(args.out_struct)
    out_weekly.parent.mkdir(parents=True, exist_ok=True)
    out_struct.parent.mkdir(parents=True, exist_ok=True)

    with out_weekly.open("wb") as f:
        weekly.write(f)
    with out_struct.open("wb") as f:
        struct.write(f)

    print(str(out_weekly))
    print(str(out_struct))
    print(f"selected_pages_operational={sorted(selected)} total_pages={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

