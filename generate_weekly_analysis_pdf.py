#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera um PDF curto de análise de uma semana específica.

Default: semana 2026-01-12 .. 2026-01-18 (W-SUN), conforme pedido.

Entrada:
- analysis_proba_raw/pro_portfolio_all/oos_walkforward_global_bayes_weekly.csv
- analysis_proba_raw/pro_portfolio_all/oos_walkforward_global_bayes_daily.csv
- analysis_proba_raw/pro_portfolio_all/oos_walkforward_global_bayes_selected_rules.csv

Saída:
- analysis_proba_raw/pro_portfolio_all/Analise_Semana_<YYYY-MM-DD>_<YYYY-MM-DD>.pdf
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")
WF_WEEKLY = OUT_DIR / "oos_walkforward_global_bayes_weekly.csv"
WF_DAILY = OUT_DIR / "oos_walkforward_global_bayes_daily.csv"
WF_RULES = OUT_DIR / "oos_walkforward_global_bayes_selected_rules.csv"


@dataclass(frozen=True)
class WeekSpec:
    start: str  # YYYY-MM-DD
    end: str    # YYYY-MM-DD

    @property
    def key(self) -> str:
        return f"{self.start}/{self.end}"


def fmt_money(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x:,.2f}"


def fmt_pct(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{100.0 * x:.2f}%"


def tbl(data: List[List[object]], col_widths=None) -> Table:
    t = Table(data, colWidths=col_widths)
    t.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(0.98, 0.98, 0.98)]),
            ]
        )
    )
    return t


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ws = WeekSpec(start="2026-01-12", end="2026-01-18")
    out_pdf = OUT_DIR / f"Analise_Semana_{ws.start}_{ws.end}.pdf"

    weekly = pd.read_csv(WF_WEEKLY)
    daily = pd.read_csv(WF_DAILY, parse_dates=["date"])
    rules = pd.read_csv(WF_RULES)

    w = weekly[weekly["week"].astype(str) == ws.key].copy()
    if w.empty:
        raise SystemExit(f"Semana não encontrada em {WF_WEEKLY}: {ws.key}")
    w0 = w.iloc[0]

    # daily breakdown for this week
    d = daily[daily["week"].astype(str) == ws.key].copy()
    if not d.empty:
        d = d.sort_values("date")
        d["roi"] = 0.0
        s = pd.to_numeric(d["stake_usd"], errors="coerce").to_numpy(float)
        p = pd.to_numeric(d["profit_cap2_usd"], errors="coerce").to_numpy(float)
        np.divide(p, s, out=d["roi"].to_numpy(), where=(s > 0))
        d["cum_profit"] = pd.to_numeric(d["profit_cap2_usd"], errors="coerce").cumsum()

    # active rules
    r = rules[rules["test_week"].astype(str) == ws.key].copy()
    r["stake_frac"] = pd.to_numeric(r["stake_frac"], errors="coerce")
    r_act = r[(r["status"].astype(str) == "ok") & (r["stake_frac"] > 0)].copy()
    r_act = r_act.sort_values(["bet_type", "dow_pt"])

    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Análise da Semana (12–18/01/2026)", styles["Title"]))
    story.append(Paragraph(f"Semana (W-SUN): <b>{ws.key}</b>", styles["Normal"]))
    story.append(Paragraph(f"Gerado em: <b>{date.today().isoformat()}</b>", styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Resumo da performance (OOS global_bayes, cap2)", styles["Heading2"]))
    summary = [
        ["Métrica", "Valor"],
        ["alpha_global", f"{float(w0['alpha_global']):.3f}"],
        ["n_bets", f"{int(w0['n_bets'])}"],
        ["stake_usd", fmt_money(float(w0["stake_usd"]))],
        ["profit_cap2_usd", fmt_money(float(w0["profit_cap2_usd"]))],
        ["ROI on stake (cap2)", fmt_pct(float(w0["roi_on_stake_cap2"]))],
    ]
    story.append(tbl(summary, col_widths=[220, 260]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Quebra diária (OOS)", styles["Heading2"]))
    if d.empty:
        story.append(Paragraph("Sem dias registrados nessa semana (stake=0).", styles["Normal"]))
    else:
        dd = [
            ["date", "stake_usd", "profit_cap2_usd", "ROI dia", "cum_profit"],
        ]
        for _, row in d.iterrows():
            dd.append(
                [
                    row["date"].date().isoformat(),
                    fmt_money(float(row["stake_usd"])),
                    fmt_money(float(row["profit_cap2_usd"])),
                    fmt_pct(float(row["roi"])),
                    fmt_money(float(row["cum_profit"])),
                ]
            )
        story.append(tbl(dd, col_widths=[90, 100, 110, 80, 100]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Segmentos ativos (regras aplicadas nessa semana)", styles["Heading2"]))
    if r_act.empty:
        story.append(Paragraph("Nenhum segmento ativo (stake_frac=0 ou status!=ok).", styles["Normal"]))
    else:
        rr = [["bet_type", "dow_pt", "score_col", "cutoff", "stake_frac", "status"]]
        for _, row in r_act.iterrows():
            rr.append(
                [
                    str(row["bet_type"]),
                    str(row["dow_pt"]),
                    str(row["score_col"]),
                    f"{float(row['cutoff']):.3f}",
                    fmt_pct(float(row["stake_frac"])),
                    str(row["status"]),
                ]
            )
        story.append(tbl(rr, col_widths=[55, 95, 155, 55, 70, 60]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Notas", styles["Heading2"]))
    story.append(
        Paragraph(
            "Este PDF usa os arquivos gerados pelo walk-forward (`oos_walkforward_global_bayes_*`). "
            "O objetivo é uma leitura objetiva da semana (retorno/risco e quais segmentos estavam ativos).",
            styles["Normal"],
        )
    )

    doc = SimpleDocTemplate(str(out_pdf), pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    doc.build(story)
    print(str(out_pdf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

