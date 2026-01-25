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
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def _cover_pdf_bytes(title: str, lines: list[str]) -> bytes:
    """
    Gera um PDF simples (capa/introdução) em memória.
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2.0 * cm, rightMargin=2.0 * cm, topMargin=2.0 * cm, bottomMargin=2.0 * cm)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    story.append(Spacer(1, 0.4 * cm))
    for ln in lines:
        story.append(Paragraph(ln, styles["BodyText"]))
        story.append(Spacer(1, 0.2 * cm))
    doc.build(story)
    return buf.getvalue()


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

    # Regras de split robustas:
    # - Operacional = tudo antes do início da seção 3 (inclui capa + seção 2.*).
    # - Estrutural  = seção 3 em diante (começa com header 3., evitando "tabelas quebradas" no início).
    idx3 = None
    needles_3 = [
        "3. Métricas estatísticas e de negócio",
        "3. Métricas",
        "3. M\u00e9tricas",
    ]
    for i in range(n):
        try:
            txt = (r.pages[i].extract_text() or "").replace("\u00a0", " ")
        except Exception:
            txt = ""
        if any(s in txt for s in needles_3):
            idx3 = i
            break
    if idx3 is None:
        # fallback conservador: se não achou o header da seção 3, mantém split antigo (capa + páginas “alvo”)
        idx3 = max(1, n)

    # Capa/introdução própria em cada PDF
    # tenta inferir data do nome do arquivo (padrão ..._YYYY-MM-DD.pdf)
    stem = inp.stem
    asof = stem.split("_")[-1] if stem.split("_")[-1].count("-") == 2 else "—"

    weekly_cover = PdfReader(BytesIO(_cover_pdf_bytes(
        title=f"Relatório Bayes Global — Semanal (Operacional) — {asof}",
        lines=[
            "Conteúdo: portfólio da semana vigente, recomendação forward-looking (próxima semana) e blocos operacionais.",
            "Nota: este PDF é gerado a partir do relatório completo, mas com uma capa/introdução própria para leitura operacional.",
        ],
    )))
    struct_cover = PdfReader(BytesIO(_cover_pdf_bytes(
        title=f"Relatório Bayes Global — Estrutural — {asof}",
        lines=[
            "Conteúdo: análises estruturais (estatística, calibração, risco, escala, slippage e diagnósticos).",
            "Este PDF começa na Seção 3 (Métricas) para evitar páginas iniciando no meio de tabelas.",
            "Exclui: detalhamento operacional da semana e portfólio forward-looking (que ficam no PDF Semanal).",
        ],
    )))

    weekly = PdfWriter()
    struct = PdfWriter()

    # adicionar capas
    weekly.add_page(weekly_cover.pages[0])
    struct.add_page(struct_cover.pages[0])

    # copiar páginas do relatório completo
    for i in range(0, idx3):
        weekly.add_page(r.pages[i])
    for i in range(idx3, n):
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
    print(f"split_idx3={idx3} total_pages_full={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

