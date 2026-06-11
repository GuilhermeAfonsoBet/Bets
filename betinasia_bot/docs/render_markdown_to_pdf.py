#!/usr/bin/env python3
"""
Renderizador simples de Markdown para PDF.

Uso:
  python3 render_markdown_to_pdf.py input.md output.pdf

Dependencia:
  reportlab
"""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def markdown_to_story(markdown_text: str):
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        spaceAfter=4,
    )
    h1 = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        spaceBefore=8,
        spaceAfter=8,
    )
    h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        spaceBefore=6,
        spaceAfter=6,
    )
    mono = ParagraphStyle(
        "Mono",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=8.5,
        leading=11,
    )

    story = []
    lines = markdown_text.splitlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            story.append(Spacer(1, 3))
            i += 1
            continue

        if stripped.startswith("# "):
            story.append(Paragraph(_escape_html(stripped[2:].strip()), h1))
            i += 1
            continue

        if stripped.startswith("## "):
            story.append(Paragraph(_escape_html(stripped[3:].strip()), h2))
            i += 1
            continue

        if stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            story.append(Preformatted("\n".join(table_lines), mono))
            story.append(Spacer(1, 3))
            continue

        if stripped.startswith("- "):
            bullets = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                bullets.append(lines[i].strip()[2:])
                i += 1
            for b in bullets:
                story.append(Paragraph(f"• {_escape_html(b)}", body))
            story.append(Spacer(1, 2))
            continue

        story.append(Paragraph(_escape_html(stripped), body))
        i += 1

    return story


def render_markdown_to_pdf(input_md: Path, output_pdf: Path) -> None:
    text = input_md.read_text(encoding="utf-8")
    story = markdown_to_story(text)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title=input_md.stem,
    )
    doc.build(story)


def main() -> int:
    if len(sys.argv) != 3:
        print("Uso: python3 render_markdown_to_pdf.py <input.md> <output.pdf>", file=sys.stderr)
        return 2

    input_md = Path(sys.argv[1]).resolve()
    output_pdf = Path(sys.argv[2]).resolve()

    if not input_md.exists():
        print(f"Arquivo markdown nao encontrado: {input_md}", file=sys.stderr)
        return 2

    try:
        render_markdown_to_pdf(input_md, output_pdf)
    except Exception as exc:
        print(f"Falha ao renderizar PDF: {exc}", file=sys.stderr)
        return 1

    print(f"PDF gerado: {output_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
