#!/usr/bin/env python3
"""Gera relatorio executivo V2 em PDF (layout visual simples).

Uso:
  python3 reports/generate_relatorio_executivo_v2.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fpdf import FPDF


OUT_DIR = Path("reports")
OUT_PDF = OUT_DIR / "relatorio_executivo_backpre_20260604_v2.pdf"
OUT_NOTE = OUT_DIR / "relatorio_executivo_backpre_20260604_v2.txt"


class ReportPDF(FPDF):
    def header(self) -> None:
        self.set_y(8)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(80, 80, 80)
        self.cell(0, 5, "Back Pre | Executive Report V2", 0, 1, "R")
        self.ln(1)

    def footer(self) -> None:
        self.set_y(-10)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 5, f"Pagina {self.page_no()}", 0, 0, "C")


def txt_line(text: str, *, bold: bool = False, size: int = 10, h: float = 5.5) -> None:
    pdf = txt_line._pdf  # type: ignore[attr-defined]
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B" if bold else "", size)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(190, h, text)


def section_title(pdf: ReportPDF, title: str) -> None:
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(25, 25, 25)
    pdf.multi_cell(0, 7, title)
    pdf.ln(0.5)


def bullets(pdf: ReportPDF, items: list[str]) -> None:
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(30, 30, 30)
    for it in items:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 5.5, f"- {it}")
    pdf.ln(1)


def table(pdf: ReportPDF, headers: list[str], rows: list[list[str]], widths: list[float]) -> None:
    pdf.set_fill_color(235, 241, 248)
    pdf.set_text_color(20, 20, 20)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_x(pdf.l_margin)
    for h, w in zip(headers, widths):
        pdf.cell(w, 7, h, border=1, fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    fill = False
    for row in rows:
        if fill:
            pdf.set_fill_color(250, 250, 250)
        else:
            pdf.set_fill_color(255, 255, 255)
        for val, w in zip(row, widths):
            pdf.cell(w, 6.2, str(val), border=1, fill=True)
        pdf.ln()
        fill = not fill
    pdf.ln(1)


def kpi_row(pdf: ReportPDF, kpis: list[tuple[str, str]]) -> None:
    box_w = 62
    box_h = 18
    x0 = pdf.l_margin
    y0 = pdf.get_y()
    for i, (label, value) in enumerate(kpis):
        x = x0 + i * (box_w + 2)
        pdf.set_xy(x, y0)
        pdf.set_draw_color(210, 210, 210)
        pdf.set_fill_color(245, 248, 252)
        pdf.rect(x, y0, box_w, box_h, style="DF")
        pdf.set_xy(x + 2, y0 + 2)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(70, 70, 70)
        pdf.cell(box_w - 4, 4, label, 0, 2)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(20, 20, 20)
        pdf.cell(box_w - 4, 8, value, 0, 0)
    pdf.set_y(y0 + box_h + 3)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    txt_line._pdf = pdf  # type: ignore[attr-defined]

    # Cover / executive snapshot
    txt_line("Relatorio Executivo V2 - Back Pre (slippage_pre_pct < 0)", bold=True, size=16, h=9)
    txt_line(
        f"Emissao: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | "
        "Escopo: periodo pos 2026-04-04, P&L real por ledger (order_id).",
        size=10,
        h=5.5,
    )
    pdf.ln(2)

    kpi_row(
        pdf,
        [
            ("ROI real total", "2.02%"),
            ("ROI ultimos 10d", "9.72%"),
            ("Cobertura max stake", "60.45%"),
        ],
    )

    section_title(pdf, "Resumo executivo")
    bullets(
        pdf,
        [
            "A reconciliacao de resultado do daily esta alinhada com accounting ledger por order_id.",
            "Nao existe max_stake canonico persistido em 100% do historico estrito.",
            "Foi definido um proxy conservador de max_stake_confiavel para projecoes de escala.",
            "Nao ha evidencia estatistica de deseconomia de escala na amostra coberta.",
        ],
    )

    section_title(pdf, "Tabela A - ROI real por janela (ledger)")
    table(
        pdf,
        ["Janela", "N", "Turnover", "P&L", "ROI"],
        [
            ["desde 2026-04-04", "445", "5,320.00", "107.62", "2.02%"],
            ["ultimos 30d", "194", "3,322.00", "71.85", "2.16%"],
            ["ultimos 10d", "61", "1,220.00", "118.62", "9.72%"],
            ["ultimos 7d", "29", "580.00", "78.08", "13.46%"],
        ],
        [64, 18, 34, 30, 20],
    )

    section_title(pdf, "Tabela B - Escala (real vs projecao)")
    table(
        pdf,
        ["Cenario", "N", "Turnover", "P&L", "ROI"],
        [
            ["REAL_total", "445", "5,320.00", "107.62", "2.02%"],
            ["REAL_coberto_maxstake", "269", "3,162.50", "160.21", "5.07%"],
            ["REAL_nao_coberto", "176", "2,157.50", "-52.59", "-2.44%"],
            ["PROJ_maxstake_confiavel", "269", "229,160.53", "18,562.11", "8.10%"],
            ["PROJ_stake_fixo_alto_p75", "269", "226,998.64", "11,509.84", "5.07%"],
            ["PROJ_stake_fixo_alto_capado", "269", "93,581.05", "4,933.05", "5.27%"],
        ],
        [84, 16, 32, 28, 20],
    )
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(90, 90, 90)
    pdf.multi_cell(
        0,
        4.8,
        "Nota: projecoes de max stake refletem apenas o subset coberto (60.45%), "
        "com risco de viés para cima.",
    )
    pdf.ln(1)

    section_title(pdf, "Deseconomia de escala (teste formal)")
    bullets(
        pdf,
        [
            "Amostra coberta: 269 apostas.",
            "Spearman(max_stake, roi_real_pct) = +0.0518.",
            "Spearman winsorizado (1%-99%) = +0.0520.",
            "Pearson(log(max_stake), roi_winsor) = +0.0494.",
            "Conclusao: sem evidencia de piora de ROI com stakes maximas maiores nesta amostra.",
        ],
    )

    section_title(pdf, "Distribuicao de stake maximo (subset coberto)")
    table(
        pdf,
        ["Quantil", "Max stake"],
        [
            ["p10", "19.20"],
            ["p25", "53.06"],
            ["p50", "220.92"],
            ["p75", "843.86"],
            ["p90", "1,623.13"],
            ["p95", "2,553.40"],
            ["p99", "10,509.85"],
        ],
        [36, 40],
    )

    # scenarios page
    pdf.add_page()
    section_title(pdf, "Matriz de cenarios solicitada (ROI fixo 5% por aposta)")
    table(
        pdf,
        ["Cap USD", "Apostas/dia", "Turnover mensal", "P&L mensal", "Faixa P&L p5-p95"],
        [
            ["800", "10", "101,025.62", "5,051.28", "4,396.90 a 5,695.53"],
            ["800", "16", "161,640.99", "8,082.05", "7,275.26 a 8,922.40"],
            ["800", "25", "252,564.05", "12,628.20", "11,457.64 a 13,553.61"],
            ["1200", "10", "124,728.66", "6,236.43", "5,395.46 a 7,136.25"],
            ["1200", "16", "199,565.86", "9,978.29", "8,885.76 a 11,075.62"],
            ["1200", "25", "311,821.65", "15,591.08", "14,142.50 a 16,847.84"],
            ["1600", "10", "140,957.83", "7,047.89", "6,047.05 a 8,080.70"],
            ["1600", "16", "225,532.53", "11,276.63", "9,987.59 a 12,593.37"],
            ["1600", "25", "352,394.58", "17,619.73", "15,898.01 a 19,165.63"],
        ],
        [20, 24, 38, 30, 62],
    )

    section_title(pdf, "Leitura de negocio")
    bullets(
        pdf,
        [
            "Aumento de cap eleva turnover e P&L esperado, mas com ganho marginal decrescente.",
            "Volume diario e o principal driver de resultado mensal no curto prazo.",
            "Sugestao de governanca: manter 3 cenarios (conservador/base/agressivo) e revisar mensalmente.",
            "Prioridade tecnica: persistir max_stake canonico por order_id para eliminar zona cinzenta de proxy.",
        ],
    )

    section_title(pdf, "Limitacoes")
    bullets(
        pdf,
        [
            "Max stake historico nao esta 100% persistido; parte da escala depende de proxy conservador.",
            "Janelas curtas (7-10 dias) sofrem maior volatilidade de ROI.",
            "Cenarios usam ROI fixo de 5% por premissa; nao representam garantia de performance.",
        ],
    )

    pdf.output(str(OUT_PDF))
    OUT_NOTE.write_text(
        "PDF V2 gerado em reports/relatorio_executivo_backpre_20260604_v2.pdf\\n",
        encoding="utf-8",
    )
    print(f"ok: {OUT_PDF}")


if __name__ == "__main__":
    main()
