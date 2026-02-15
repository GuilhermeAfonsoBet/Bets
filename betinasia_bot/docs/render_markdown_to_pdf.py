#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Renderiza um markdown simples para PDF com foco em legibilidade executiva.

Suporta:
- Títulos (#, ##, ###)
- Parágrafos
- Listas simples iniciadas por "-"
- Tabelas markdown simples com "|" (inclui quebra de linha em células)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List
from xml.sax.saxutils import escape as _xml_escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)


def build_styles():
    styles = getSampleStyleSheet()
    return {
        "h1": ParagraphStyle(
            "h1",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            spaceBefore=10,
            spaceAfter=8,
            textColor=colors.HexColor("#0f172a"),
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            spaceBefore=10,
            spaceAfter=6,
            textColor=colors.HexColor("#111827"),
        ),
        "h3": ParagraphStyle(
            "h3",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            spaceBefore=8,
            spaceAfter=4,
            textColor=colors.HexColor("#1f2937"),
        ),
        "body": ParagraphStyle(
            "body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            spaceBefore=2,
            spaceAfter=2,
            textColor=colors.HexColor("#111827"),
            wordWrap="CJK",
        ),
        "quote": ParagraphStyle(
            "quote",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=9.5,
            leading=13,
            leftIndent=10,
            spaceBefore=3,
            spaceAfter=3,
            textColor=colors.HexColor("#111827"),
            backColor=colors.HexColor("#f8fafc"),
            borderColor=colors.HexColor("#cbd5e1"),
            borderWidth=0.7,
            borderPadding=6,
            wordWrap="CJK",
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            leftIndent=12,
            bulletIndent=2,
            spaceBefore=1,
            spaceAfter=1,
            textColor=colors.HexColor("#111827"),
            wordWrap="CJK",
        ),
        "code": ParagraphStyle(
            "code",
            parent=styles["BodyText"],
            fontName="Courier",
            fontSize=8.8,
            leading=11.2,
            leftIndent=8,
            rightIndent=8,
            spaceBefore=4,
            spaceAfter=6,
            backColor=colors.HexColor("#0b1020"),
            textColor=colors.HexColor("#e5e7eb"),
            borderPadding=6,
        ),
    }


def normalize_inline(text: str) -> str:
    """
    Converte markdown *simples* para markup do ReportLab Paragraph.
    - Escapa XML
    - **bold**
    - `inline code`
    """
    text = (text or "").strip()
    if text == "---" or text == "":
        return ""

    # Escape primeiro para evitar quebrar markup do ReportLab.
    text = _xml_escape(text)

    # **negrito**
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)

    # `código inline` -> fonte monoespaçada
    # (não suporta múltiplas linhas)
    text = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', text)
    return text


def normalize_cell(text: str) -> str:
    return normalize_inline(text)


def strip_html_like(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def is_table_line(line: str) -> bool:
    return line.strip().startswith("|") and line.strip().endswith("|")


def parse_table_block(lines: List[str], start_idx: int):
    block = []
    idx = start_idx
    while idx < len(lines) and is_table_line(lines[idx]):
        block.append(lines[idx].strip())
        idx += 1
    return block, idx


def table_to_flowable(table_lines: List[str], available_width: float, body_style: ParagraphStyle):
    rows = []
    for ln in table_lines:
        cells = [normalize_cell(c) for c in ln.strip().strip("|").split("|")]
        rows.append(cells)

    # Remove linha separadora tipo |---|---:|
    clean_rows = []
    for r in rows:
        if all(c.replace("-", "").replace(":", "").strip() == "" for c in r):
            continue
        clean_rows.append(r)

    if not clean_rows:
        return None

    n_cols = max(len(r) for r in clean_rows)
    padded_rows = [r + [""] * (n_cols - len(r)) for r in clean_rows]

    # Distribui largura por "peso" de conteúdo para melhorar legibilidade.
    col_weights = []
    for c in range(n_cols):
        max_len = 6
        for row in padded_rows:
            cell = strip_html_like(row[c])
            max_len = max(max_len, len(cell))
        col_weights.append(float(max_len))

    total_weight = sum(col_weights) if sum(col_weights) > 0 else float(n_cols)
    raw_widths = [(w / total_weight) * available_width for w in col_weights]

    # Largura mínima evita esmagar colunas pequenas.
    # Porém, quando n_cols é grande, `n_cols * min_col` pode estourar a página e quebrar o layout.
    # Ajustamos min_col dinamicamente para sempre caber.
    base_min = min(75.0, max(48.0, available_width * 0.08))
    # Se a tabela é "larga", reduz min_col para caber.
    if n_cols * base_min > available_width:
        base_min = max(14.0, (available_width / float(n_cols)) * 0.98)
    min_col = float(base_min)
    col_widths = [max(min_col, float(w)) for w in raw_widths]
    width_over = sum(col_widths) - available_width
    if width_over > 0:
        # Ajuste proporcional sem quebrar mínimos.
        flex_indices = [i for i, w in enumerate(col_widths) if w > min_col]
        if flex_indices:
            flex_total = sum(col_widths[i] - min_col for i in flex_indices)
            if flex_total > 0:
                for i in flex_indices:
                    flex_part = (col_widths[i] - min_col) / flex_total
                    col_widths[i] -= width_over * flex_part

    # Header style: precisa ser branco, pois TableStyle(TEXTCOLOR) não sobrescreve
    # a cor já definida dentro do ParagraphStyle.
    header_style = ParagraphStyle(
        "table_header",
        parent=body_style,
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        textColor=colors.white,
        wordWrap="CJK",
    )

    table_data = []
    for i, r in enumerate(padded_rows):
        style = header_style if i == 0 else body_style
        row_cells = [Paragraph(c if c else "&nbsp;", style) for c in r]
        table_data.append(row_cells)

    table = Table(table_data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    body_font_size = 9 if n_cols <= 7 else (8 if n_cols <= 10 else (7.5 if n_cols <= 14 else 6.8))
    header_font_size = 9 if n_cols <= 10 else (8 if n_cols <= 14 else 6.8)
    pad_lr = 5 if n_cols <= 10 else (3 if n_cols <= 14 else 1.5)
    pad_tb = 4 if n_cols <= 10 else (3 if n_cols <= 14 else 2)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b73")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), header_font_size),
                ("FONTSIZE", (0, 1), (-1, -1), body_font_size),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), pad_lr),
                ("RIGHTPADDING", (0, 0), (-1, -1), pad_lr),
                ("TOPPADDING", (0, 0), (-1, -1), pad_tb),
                ("BOTTOMPADDING", (0, 0), (-1, -1), pad_tb),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ]
        )
    )

    # Zebra striping no corpo da tabela.
    for r_idx in range(1, len(table_data)):
        bg = colors.HexColor("#f8fafc") if (r_idx % 2 == 1) else colors.white
        table.setStyle(TableStyle([("BACKGROUND", (0, r_idx), (-1, r_idx), bg)]))

    return table


def render_markdown(md_text: str, output_pdf: Path):
    styles = build_styles()
    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title="Relatório Analítico (BetinAsia)",
        author="BetinAsia Operação",
    )

    def _footer(canvas, doc_):
        canvas.saveState()
        w, h = doc_.pagesize
        canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
        canvas.setLineWidth(0.6)
        canvas.line(doc_.leftMargin, doc_.bottomMargin - 6, w - doc_.rightMargin, doc_.bottomMargin - 6)
        canvas.setFont("Helvetica", 8.5)
        canvas.setFillColor(colors.HexColor("#334155"))
        canvas.drawString(doc_.leftMargin, doc_.bottomMargin - 18, "BetinAsia • Relatório estatístico/operacional")
        canvas.drawRightString(w - doc_.rightMargin, doc_.bottomMargin - 18, f"Página {canvas.getPageNumber()}")
        canvas.restoreState()

    flow = []
    lines = md_text.splitlines()
    available_width = doc.pagesize[0] - doc.leftMargin - doc.rightMargin

    i = 0
    in_code = False
    code_buf: List[str] = []
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()

        # Code fences (``` ... ```)
        if line.startswith("```"):
            if not in_code:
                in_code = True
                code_buf = []
            else:
                in_code = False
                code_text = "\n".join(code_buf).rstrip()
                if code_text:
                    flow.append(Preformatted(code_text, styles["code"]))
                    flow.append(Spacer(1, 6))
            i += 1
            continue

        if in_code:
            code_buf.append(raw.rstrip("\n"))
            i += 1
            continue

        if not line:
            flow.append(Spacer(1, 6))
            i += 1
            continue

        if line == "---":
            flow.append(Spacer(1, 8))
            i += 1
            continue

        if is_table_line(line):
            table_lines, i = parse_table_block(lines, i)
            table = table_to_flowable(table_lines, available_width, styles["body"])
            if table:
                flow.append(table)
                flow.append(Spacer(1, 8))
            continue

        if line.startswith("# "):
            flow.append(Paragraph(normalize_inline(line[2:].strip()), styles["h1"]))
            i += 1
            continue
        if line.startswith("## "):
            flow.append(Paragraph(normalize_inline(line[3:].strip()), styles["h2"]))
            i += 1
            continue
        if line.startswith("### "):
            flow.append(Paragraph(normalize_inline(line[4:].strip()), styles["h3"]))
            i += 1
            continue

        if line.startswith("> "):
            flow.append(Paragraph(normalize_inline(line[2:].strip()), styles["quote"]))
            i += 1
            continue

        if line.startswith("- "):
            flow.append(Paragraph(f"• {normalize_cell(line[2:].strip())}", styles["bullet"]))
            i += 1
            continue

        flow.append(Paragraph(normalize_cell(line), styles["body"]))
        i += 1

    doc.build(flow, onFirstPage=_footer, onLaterPages=_footer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_md", help="Arquivo markdown de entrada")
    parser.add_argument("output_pdf", help="Arquivo PDF de saída")
    args = parser.parse_args()

    md_path = Path(args.input_md)
    pdf_path = Path(args.output_pdf)

    text = md_path.read_text(encoding="utf-8")
    render_markdown(text, pdf_path)
    print(f"PDF gerado: {pdf_path}")


if __name__ == "__main__":
    main()

