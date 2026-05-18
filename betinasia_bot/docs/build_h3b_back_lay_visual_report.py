#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera PDF de leitura com graficos da analise H3B (Back + Lay).

Entrada esperada (arquivos no workspace):
  - 03_ws_vs_bs.log
  - 04_h3b_comprehensive.log
  - 05_lay_bucket_analysis.txt
  - 06_lay_target_summary.txt

Saida:
  - betinasia_bot/docs/analise_h3b_back_lay_visual_2026-02-12.pdf
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.shapes import Drawing, Line, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_clv_block_ws_bs(text: str, label: str) -> Dict[str, float]:
    pattern = re.compile(
        rf"{re.escape(label)}:\s*"
        r"\n\s*N = (?P<n>\d+)"
        r"\n\s*CLV adicional = (?P<mean>[+-]?\d+\.\d+)%"
        r"\n\s*Erro padrão\s*= (?P<se>\d+\.\d+)%"
        r"\n\s*IC 90%\s*= \[(?P<lo>[+-]?\d+\.\d+)%, (?P<hi>[+-]?\d+\.\d+)%\]",
        re.MULTILINE,
    )
    m = pattern.search(text)
    if not m:
        raise ValueError(f"Nao foi possivel extrair bloco: {label}")
    return {
        "n": int(m.group("n")),
        "mean": float(m.group("mean")),
        "se": float(m.group("se")),
        "lo": float(m.group("lo")),
        "hi": float(m.group("hi")),
    }


def parse_mean_metric(text: str, title: str) -> Dict[str, float]:
    pattern = re.compile(
        rf"{re.escape(title)}:\s*"
        r"\n\s*N = (?P<n>\d+)"
        r"\n\s*Media\s+= (?P<mean>[+-]?\d+\.\d+)%"
        r".*?"
        r"\n\s*IC 90% media = \[(?P<lo>[+-]?\d+\.\d+)%, (?P<hi>[+-]?\d+\.\d+)%\]",
        re.MULTILINE | re.DOTALL,
    )
    m = pattern.search(text)
    if not m:
        raise ValueError(f"Nao foi possivel extrair metrica: {title}")
    return {
        "n": int(m.group("n")),
        "mean": float(m.group("mean")),
        "lo": float(m.group("lo")),
        "hi": float(m.group("hi")),
    }


def parse_bucket_block(text: str, label: str) -> Dict[str, float]:
    pattern = re.compile(
        rf"=== {re.escape(label)} \(N=(?P<n_total>\d+)\) ==="
        r"(?P<body>.*?)(?:\n\s*=== |\n={70,})",
        re.DOTALL,
    )
    m = pattern.search(text + "\n=== ")
    if not m:
        raise ValueError(f"Nao foi possivel extrair bucket: {label}")
    body = m.group("body")
    clv = parse_mean_metric(body, "CLV Betslip (pre-match)")
    roi = parse_mean_metric(body, "ROI Betslip (todos)")
    return {
        "n_total": int(m.group("n_total")),
        "clv_n": clv["n"],
        "clv_mean": clv["mean"],
        "roi_n": roi["n"],
        "roi_mean": roi["mean"],
    }


def section_between(text: str, start: str, end: str) -> str:
    a = text.find(start)
    if a < 0:
        raise ValueError(f"Marcador de inicio nao encontrado: {start}")
    b = text.find(end, a + len(start))
    if b < 0:
        raise ValueError(f"Marcador de fim nao encontrado: {end}")
    return text[a:b]


def parse_model_metrics(text: str) -> Dict[str, Dict[str, float]]:
    api_sec = section_between(text, "=== API (2-4s) ===", "=== DOM (15-30s) ===")
    dom_sec = section_between(text, "=== DOM (15-30s) ===", "DIAGNOSTICO DE QUALIDADE")

    def parse_sec(sec: str, label: str) -> Dict[str, float]:
        lag_m = re.search(r"Lag medio:\s*(\d+)ms", sec)
        if not lag_m:
            raise ValueError(f"Lag medio ausente em {label}")
        lag_ms = int(lag_m.group(1))
        clv_add = parse_mean_metric(sec, f"CLV Adicional BS Pre-Match ({label})")
        diff = parse_mean_metric(sec, f"Diff BS vs WS ({label})")
        roi = parse_mean_metric(sec, f"ROI Betslip ({label})")
        plus2 = re.search(r"BS > WS \+2%:\s*(\d+)/(\d+)\s*\((\d+\.\d+)%\)", sec)
        if not plus2:
            raise ValueError(f"BS>WS +2% ausente em {label}")
        return {
            "lag_ms": lag_ms,
            "clv_add_mean": clv_add["mean"],
            "diff_mean": diff["mean"],
            "roi_mean": roi["mean"],
            "bs_ws_plus2_pct": float(plus2.group(3)),
            "bs_ws_plus2_n": int(plus2.group(1)),
            "bs_ws_plus2_total": int(plus2.group(2)),
        }

    return {
        "API (2-4s)": parse_sec(api_sec, "API (2-4s)"),
        "DOM (15-30s)": parse_sec(dom_sec, "DOM (15-30s)"),
    }


def parse_lay_buckets(text: str) -> List[Dict[str, Optional[float]]]:
    rows: List[Dict[str, Optional[float]]] = []
    for ln in text.splitlines():
        if "|" not in ln:
            continue
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) < 5:
            continue
        regime = parts[0]
        if regime not in {"IN_MATCH", "PRE_MATCH"}:
            continue
        n_total = int(parts[2])
        n_lay = int(parts[3])
        clv_raw = parts[4]
        clv = float(clv_raw) if clv_raw else None
        rows.append(
            {
                "regime": regime,
                "bucket": parts[1],
                "n_total": n_total,
                "n_lay": n_lay,
                "coverage_pct": (100.0 * n_lay / n_total) if n_total else 0.0,
                "lay_clv_mean": clv,
            }
        )
    return rows


def parse_summary_counts(text: str) -> Dict[str, int]:
    patterns = {
        "total_match_kickoff": r"Total com match\+kickoff:\s*(\d+)",
        "com_betslip": r"Com betslip:\s*(\d+)",
        "com_clv_bs": r"Com CLV BS bruto:\s*(\d+)",
        "com_roi_bs": r"Com ROI BS:\s*(\d+)",
        "pre_match": r"Pre-match:\s*(\d+)",
        "in_match": r"In-match:\s*(\d+)",
    }
    out: Dict[str, int] = {}
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if not m:
            raise ValueError(f"Resumo nao encontrado: {key}")
        out[key] = int(m.group(1))
    return out


def nice_axis_bounds(values: List[float]) -> tuple[float, float, float]:
    vmin = min(values + [0.0])
    vmax = max(values + [0.0])
    span = max(1.0, vmax - vmin)
    pad = max(0.5, span * 0.15)
    y0 = vmin - pad
    y1 = vmax + pad
    step = 1.0
    for candidate in (0.5, 1, 2, 5, 10, 20, 50):
        if (y1 - y0) / candidate <= 8:
            step = float(candidate)
            break
    return y0, y1, step


def make_grouped_bar_chart(
    title: str,
    categories: List[str],
    series: List[List[float]],
    series_names: List[str],
    percent: bool = True,
    width: int = 520,
    height: int = 260,
) -> Drawing:
    d = Drawing(width, height)
    d.add(String(width / 2, height - 14, title, textAnchor="middle", fontSize=11))

    chart = VerticalBarChart()
    chart.x = 48
    chart.y = 40
    chart.height = height - 90
    chart.width = width - 86
    chart.data = series
    chart.categoryAxis.categoryNames = categories
    chart.categoryAxis.labels.angle = 20
    chart.categoryAxis.labels.fontSize = 8
    chart.categoryAxis.labels.dy = -10
    chart.barLabels.fontSize = 7
    chart.barLabels.nudge = 6
    chart.barLabels.visible = True
    chart.barLabelFormat = "%.2f"
    chart.groupSpacing = 8
    chart.barSpacing = 2

    values = [v for row in series for v in row]
    y0, y1, step = nice_axis_bounds(values)
    chart.valueAxis.valueMin = y0
    chart.valueAxis.valueMax = y1
    chart.valueAxis.valueStep = step
    chart.valueAxis.labels.fontSize = 8

    palette = [
        colors.HexColor("#2F80ED"),
        colors.HexColor("#F2994A"),
        colors.HexColor("#27AE60"),
        colors.HexColor("#9B51E0"),
    ]
    for i in range(len(series)):
        chart.bars[i].fillColor = palette[i % len(palette)]

    d.add(chart)

    if y0 < 0 < y1:
        zero_y = chart.y + ((0 - y0) / (y1 - y0)) * chart.height
        d.add(Line(chart.x, zero_y, chart.x + chart.width, zero_y, strokeColor=colors.grey, strokeWidth=0.8))

    legend_x = chart.x + 2
    legend_y = 16
    for i, name in enumerate(series_names):
        x = legend_x + i * 130
        d.add(Line(x, legend_y, x + 10, legend_y, strokeColor=palette[i % len(palette)], strokeWidth=6))
        d.add(String(x + 14, legend_y - 3, name, fontSize=8))

    unit = "%" if percent else ""
    d.add(String(width - 12, height - 28, unit, fontSize=8, textAnchor="end"))
    return d


def money_table_style() -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f2f5")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111111")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
    )


def build_pdf(
    output_pdf: Path,
    summary: Dict[str, int],
    ws_up: Dict[str, float],
    ws_down: Dict[str, float],
    bs_up: Dict[str, float],
    back_buckets: Dict[str, Dict[str, float]],
    lay_rows: List[Dict[str, Optional[float]]],
    model_metrics: Dict[str, Dict[str, float]],
) -> None:
    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
    )
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=20, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15, spaceBefore=8, spaceAfter=6)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName="Helvetica", fontSize=9, leading=12)

    story = []
    story.append(Paragraph("Analise H3B Back + Lay (atualizada, com graficos)", h1))
    story.append(
        Paragraph(
            "Base: logs consolidados (03/04/05/06). Objetivo: leitura executiva de valor em Back e Lay, "
            "com foco em distribuicao por bucket e comparativo operacional API vs DOM.",
            body,
        )
    )
    story.append(Spacer(1, 6))

    story.append(Paragraph("1) Cobertura da base e qualidade", h2))
    coverage_tbl = Table(
        [
            ["Metrica", "Valor"],
            ["Total com match + kickoff", f"{summary['total_match_kickoff']}"],
            ["Com betslip", f"{summary['com_betslip']}"],
            ["Com CLV BS bruto", f"{summary['com_clv_bs']}"],
            ["Com ROI BS", f"{summary['com_roi_bs']}"],
            ["Pre-match", f"{summary['pre_match']}"],
            ["In-match", f"{summary['in_match']}"],
        ],
        colWidths=[95 * mm, 40 * mm],
    )
    coverage_tbl.setStyle(money_table_style())
    story.append(coverage_tbl)
    story.append(Spacer(1, 8))

    story.append(Paragraph("2) Back: CLV adicional (WebSocket vs Betslip)", h2))
    chart_clv = make_grouped_bar_chart(
        title="CLV adicional por fonte (UP/DOWN)",
        categories=["WS UP", "WS DOWN", "BS UP"],
        series=[[ws_up["mean"], ws_down["mean"], bs_up["mean"]]],
        series_names=["CLV adicional"],
        percent=True,
    )
    story.append(chart_clv)
    story.append(Spacer(1, 4))
    clv_tbl = Table(
        [
            ["Bloco", "N", "Media", "IC 90%"],
            ["WS UP", f"{ws_up['n']}", f"{ws_up['mean']:+.3f}%", f"[{ws_up['lo']:+.3f}%, {ws_up['hi']:+.3f}%]"],
            ["WS DOWN", f"{ws_down['n']}", f"{ws_down['mean']:+.3f}%", f"[{ws_down['lo']:+.3f}%, {ws_down['hi']:+.3f}%]"],
            ["BS UP", f"{bs_up['n']}", f"{bs_up['mean']:+.3f}%", f"[{bs_up['lo']:+.3f}%, {bs_up['hi']:+.3f}%]"],
        ],
        colWidths=[30 * mm, 16 * mm, 26 * mm, 63 * mm],
    )
    clv_tbl.setStyle(money_table_style())
    story.append(clv_tbl)
    story.append(Spacer(1, 10))

    story.append(Paragraph("3) Back por bucket de diff (BS vs WS)", h2))
    bucket_order = ["BS < WS (-10% a -2%)", "BS ~ WS (-2% a +2%)", "BS > WS (+2% a +10%)"]
    bucket_short = ["BS<WS", "BS~WS", "BS>WS"]
    clv_vals = [back_buckets[b]["clv_mean"] for b in bucket_order]
    roi_vals = [back_buckets[b]["roi_mean"] for b in bucket_order]
    chart_buckets = make_grouped_bar_chart(
        title="Back por bucket: CLV pre-match x ROI",
        categories=bucket_short,
        series=[clv_vals, roi_vals],
        series_names=["CLV", "ROI"],
        percent=True,
    )
    story.append(chart_buckets)
    bucket_tbl_rows = [["Bucket", "N total", "N CLV", "CLV", "N ROI", "ROI"]]
    for b in bucket_order:
        row = back_buckets[b]
        bucket_tbl_rows.append(
            [
                b,
                f"{row['n_total']}",
                f"{row['clv_n']}",
                f"{row['clv_mean']:+.3f}%",
                f"{row['roi_n']}",
                f"{row['roi_mean']:+.3f}%",
            ]
        )
    bucket_tbl = Table(bucket_tbl_rows, colWidths=[50 * mm, 18 * mm, 18 * mm, 22 * mm, 18 * mm, 22 * mm])
    bucket_tbl.setStyle(money_table_style())
    story.append(bucket_tbl)
    story.append(Spacer(1, 10))

    story.append(Paragraph("4) Lay por bucket (cobertura e CLV)", h2))
    pre_rows = [r for r in lay_rows if r["regime"] == "PRE_MATCH"]
    pre_rows_sorted = sorted(pre_rows, key=lambda x: x["bucket"])
    lay_categories = [r["bucket"].replace("PRE_MATCH", "").strip() for r in pre_rows_sorted]
    lay_cov = [r["coverage_pct"] for r in pre_rows_sorted]
    chart_lay_cov = make_grouped_bar_chart(
        title="Lay coverage no pre-match (N_lay / N_total)",
        categories=lay_categories,
        series=[lay_cov],
        series_names=["Coverage"],
        percent=True,
    )
    story.append(chart_lay_cov)

    clv_rows = [r for r in pre_rows_sorted if r["lay_clv_mean"] is not None]
    if clv_rows:
        chart_lay_clv = make_grouped_bar_chart(
            title="Lay CLV medio por bucket (somente buckets com closing)",
            categories=[r["bucket"] for r in clv_rows],
            series=[[float(r["lay_clv_mean"]) for r in clv_rows]],
            series_names=["Lay CLV"],
            percent=True,
        )
        story.append(Spacer(1, 4))
        story.append(chart_lay_clv)

    lay_tbl_rows = [["Regime", "Bucket", "N total", "N lay", "Cobertura", "Lay CLV"]]
    for row in lay_rows:
        lay_tbl_rows.append(
            [
                row["regime"],
                row["bucket"],
                str(int(row["n_total"])),
                str(int(row["n_lay"])),
                f"{row['coverage_pct']:.1f}%",
                "-" if row["lay_clv_mean"] is None else f"{float(row['lay_clv_mean']):+.3f}%",
            ]
        )
    lay_tbl = Table(lay_tbl_rows, colWidths=[23 * mm, 30 * mm, 18 * mm, 16 * mm, 22 * mm, 22 * mm])
    lay_tbl.setStyle(money_table_style())
    story.append(lay_tbl)
    story.append(Spacer(1, 10))

    story.append(Paragraph("5) API vs DOM (operacional + qualidade do preco)", h2))
    models = ["API (2-4s)", "DOM (15-30s)"]
    lag_sec = [model_metrics[m]["lag_ms"] / 1000.0 for m in models]
    plus2 = [model_metrics[m]["bs_ws_plus2_pct"] for m in models]
    diff_vals = [model_metrics[m]["diff_mean"] for m in models]
    chart_lag = make_grouped_bar_chart(
        title="Lag medio por modelo (segundos)",
        categories=["API", "DOM"],
        series=[lag_sec],
        series_names=["Lag s"],
        percent=False,
    )
    story.append(chart_lag)
    chart_plus2 = make_grouped_bar_chart(
        title="Share BS > WS +2% (proxy de janela de valor)",
        categories=["API", "DOM"],
        series=[plus2],
        series_names=["BS>WS+2%"],
        percent=True,
    )
    story.append(chart_plus2)
    model_tbl = Table(
        [
            ["Modelo", "Lag medio", "Diff BS-WS", "CLV add BS", "ROI BS", "BS>WS+2%"],
            [
                "API (2-4s)",
                f"{model_metrics['API (2-4s)']['lag_ms']} ms",
                f"{model_metrics['API (2-4s)']['diff_mean']:+.3f}%",
                f"{model_metrics['API (2-4s)']['clv_add_mean']:+.3f}%",
                f"{model_metrics['API (2-4s)']['roi_mean']:+.3f}%",
                f"{model_metrics['API (2-4s)']['bs_ws_plus2_pct']:.1f}%",
            ],
            [
                "DOM (15-30s)",
                f"{model_metrics['DOM (15-30s)']['lag_ms']} ms",
                f"{model_metrics['DOM (15-30s)']['diff_mean']:+.3f}%",
                f"{model_metrics['DOM (15-30s)']['clv_add_mean']:+.3f}%",
                f"{model_metrics['DOM (15-30s)']['roi_mean']:+.3f}%",
                f"{model_metrics['DOM (15-30s)']['bs_ws_plus2_pct']:.1f}%",
            ],
        ],
        colWidths=[32 * mm, 24 * mm, 22 * mm, 22 * mm, 22 * mm, 22 * mm],
    )
    model_tbl.setStyle(money_table_style())
    story.append(model_tbl)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Leitura executiva", h2))
    dominant_bucket = bucket_order[max(range(len(bucket_order)), key=lambda i: back_buckets[bucket_order[i]]["roi_mean"])]
    story.append(
        Paragraph(
            f"Back: o bucket com melhor ROI medio nesta janela foi <b>{dominant_bucket}</b>, "
            "mas com intervalo amplo e N ainda limitado para inferencia robusta. "
            f"No Lay pre-match, a cobertura cresce forte no bucket E (>= +2%), mas o CLV medio observado nesse bucket foi negativo "
            f"({next((r['lay_clv_mean'] for r in pre_rows_sorted if r['bucket'] == 'E >= +2%'), None):+.3f}%). "
            "A conclusao operacional e que o ganho real depende de aumentar N com telemetria fina de etapa e leitura temporal Back/Lay.",
            body,
        )
    )

    doc.build(story)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/workspace", help="Diretorio com logs 03/04/05/06")
    parser.add_argument(
        "--output-pdf",
        default="/workspace/betinasia_bot/docs/analise_h3b_back_lay_visual_2026-02-12.pdf",
        help="PDF de saida",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_pdf = Path(args.output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    ws_text = read_text(input_dir / "03_ws_vs_bs.log")
    comp_text = read_text(input_dir / "04_h3b_comprehensive.log")
    lay_text = read_text(input_dir / "05_lay_bucket_analysis.txt")

    ws_up = parse_clv_block_ws_bs(ws_text, "WEBSOCKET - REVERSÃO UP")
    ws_down = parse_clv_block_ws_bs(ws_text, "WEBSOCKET - REVERSÃO DOWN")
    bs_up = parse_clv_block_ws_bs(ws_text, "BETSLIP - REVERSÃO UP")

    summary = parse_summary_counts(comp_text)
    back_buckets = {
        "BS < WS (-10% a -2%)": parse_bucket_block(comp_text, "BS < WS (-10% a -2%)"),
        "BS ~ WS (-2% a +2%)": parse_bucket_block(comp_text, "BS ~ WS (-2% a +2%)"),
        "BS > WS (+2% a +10%)": parse_bucket_block(comp_text, "BS > WS (+2% a +10%)"),
    }
    model_metrics = parse_model_metrics(comp_text)
    lay_rows = parse_lay_buckets(lay_text)

    build_pdf(
        output_pdf=output_pdf,
        summary=summary,
        ws_up=ws_up,
        ws_down=ws_down,
        bs_up=bs_up,
        back_buckets=back_buckets,
        lay_rows=lay_rows,
        model_metrics=model_metrics,
    )
    print(f"PDF gerado: {output_pdf}")


if __name__ == "__main__":
    main()
