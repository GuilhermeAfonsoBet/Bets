"""PDF renderer for Daily V2 PREVIEW (never uses V1 filenames/paths)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate

from .preview_labels import PREVIEW_FOOTER, ensure_preview_markdown_banner, markdown_has_preview_label


def _load_v1_renderer_module(root: Optional[Path] = None):
    """Load docs/render_markdown_to_pdf.py without requiring package install."""
    candidates = []
    if root:
        candidates.append(Path(root) / "docs" / "render_markdown_to_pdf.py")
    # relative to this file: ops/daily_v2 -> ../../docs
    here = Path(__file__).resolve()
    candidates.append(here.parents[2] / "docs" / "render_markdown_to_pdf.py")
    candidates.append(Path.cwd() / "docs" / "render_markdown_to_pdf.py")
    for p in candidates:
        if p.exists():
            spec = importlib.util.spec_from_file_location("betinasia_render_md_pdf", p)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = mod
                spec.loader.exec_module(mod)
                return mod
    raise FileNotFoundError("docs/render_markdown_to_pdf.py not found")


def render_preview_pdf(
    md_text: str,
    pdf_path: Path,
    *,
    root: Optional[Path] = None,
) -> Path:
    """Render markdown to PDF with PREVIEW footer on every page."""
    md = ensure_preview_markdown_banner(md_text)
    if not markdown_has_preview_label(md):
        raise ValueError("PREVIEW_LABEL_VALIDATION_FAILED")

    mod = _load_v1_renderer_module(root)
    # Re-use parsing/styles from V1 renderer but inject preview footer.
    styles = mod.build_styles()
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.8 * cm,
        title=PREVIEW_FOOTER,
    )

    def _footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(colors.HexColor("#b45309"))
        w, _h = A4
        canvas.drawString(doc_.leftMargin, doc_.bottomMargin - 12, PREVIEW_FOOTER)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#6b7280"))
        canvas.drawRightString(w - doc_.rightMargin, doc_.bottomMargin - 12, f"Página {canvas.getPageNumber()}")
        # header banner
        canvas.setFont("Helvetica-Bold", 9)
        canvas.setFillColor(colors.HexColor("#b45309"))
        canvas.drawString(doc_.leftMargin, _h - 1.0 * cm, PREVIEW_FOOTER)
        canvas.restoreState()

    # Build flow by temporarily swapping footer via local copy of render loop.
    # Call internal pieces: easiest path — write temp md and monkeypatch footer in a local build.
    flow = []
    lines = md.splitlines()
    available_width = doc.pagesize[0] - doc.leftMargin - doc.rightMargin
    i = 0
    in_code = False
    code_buf = []
    from reportlab.platypus import Paragraph, Preformatted, Spacer

    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
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
        if mod.is_table_line(line):
            table_lines, i = mod.parse_table_block(lines, i)
            table = mod.table_to_flowable(table_lines, available_width, styles["body"])
            if table:
                flow.append(table)
                flow.append(Spacer(1, 8))
            continue
        if line.startswith("# "):
            flow.append(Paragraph(mod.normalize_inline(line[2:].strip()), styles["h1"]))
            i += 1
            continue
        if line.startswith("## "):
            flow.append(Paragraph(mod.normalize_inline(line[3:].strip()), styles["h2"]))
            i += 1
            continue
        if line.startswith("### "):
            flow.append(Paragraph(mod.normalize_inline(line[4:].strip()), styles["h3"]))
            i += 1
            continue
        if line.startswith("> "):
            flow.append(Paragraph(mod.normalize_inline(line[2:].strip()), styles["quote"]))
            i += 1
            continue
        if line.startswith("- "):
            flow.append(Paragraph(f"• {mod.normalize_cell(line[2:].strip())}", styles["bullet"]))
            i += 1
            continue
        flow.append(Paragraph(mod.normalize_cell(line), styles["body"]))
        i += 1

    doc.build(flow, onFirstPage=_footer, onLaterPages=_footer)
    return pdf_path


def pdf_contains_preview_label(pdf_path: Path) -> bool:
    """Best-effort: check raw PDF bytes for PREVIEW label string."""
    try:
        data = Path(pdf_path).read_bytes()
    except Exception:
        return False
    if b"PREVIEW" not in data:
        return False
    # Accented "NÃO" may be fragmented in PDF streams; accept OFICIAL + PREVIEW.
    return b"OFICIAL" in data or "NÃO OFICIAL".encode("utf-8") in data or b"NAO OFICIAL" in data