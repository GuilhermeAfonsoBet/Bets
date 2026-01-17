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
FC_WEEK = OUT_DIR / "forecast_calibration_global_bayes.csv"
FC_ONLINE = OUT_DIR / "forecast_calibration_global_bayes_online_bias.csv"


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
    fc = pd.read_csv(FC_WEEK) if FC_WEEK.exists() else pd.DataFrame()
    fco = pd.read_csv(FC_ONLINE) if FC_ONLINE.exists() else pd.DataFrame()

    w = weekly[weekly["week"].astype(str) == ws.key].copy()
    if w.empty:
        raise SystemExit(f"Semana não encontrada em {WF_WEEKLY}: {ws.key}")
    w0 = w.iloc[0]
    fcw = fc[fc["week"].astype(str) == ws.key].copy() if (not fc.empty and "week" in fc.columns) else pd.DataFrame()
    fco_w = fco[fco["week"].astype(str) == ws.key].copy() if (not fco.empty and "week" in fco.columns) else pd.DataFrame()

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

    story.append(Paragraph("Previsto vs realizado (OOS teórico)", styles["Heading2"]))
    if fcw.empty:
        story.append(Paragraph("Sem dados de forecast para essa semana (arquivo de calibração ausente).", styles["Normal"]))
    else:
        r0 = fcw.iloc[0]
        realized = float(r0.get("pnl_theoretical", float("nan")))
        pred_mean = float(r0.get("pred_mean", float("nan")))
        pred_p10 = float(r0.get("pred_p10", float("nan")))
        pred_p50 = float(r0.get("pred_p50", float("nan")))
        pred_p90 = float(r0.get("pred_p90", float("nan")))
        online = float("nan")
        if not fco_w.empty and "pred_mean_bias_adj_online" in fco_w.columns:
            online = float(fco_w.iloc[0].get("pred_mean_bias_adj_online", float("nan")))
        tdata = [
            ["Métrica", "Valor"],
            ["Realizado OOS (PnL teórico)", fmt_money(realized)],
            ["Previsto (média, sem correção)", fmt_money(pred_mean)],
            ["Previsto (p10 / p50 / p90)", f"{fmt_money(pred_p10)} / {fmt_money(pred_p50)} / {fmt_money(pred_p90)}"],
            ["Previsto (média, bias-adjusted on-line)", fmt_money(online)],
            ["Δ Realizado - Previsto (média)", fmt_money(realized - pred_mean) if np.isfinite(realized) and np.isfinite(pred_mean) else "—"],
            ["Δ Realizado - Previsto (média on-line)", fmt_money(realized - online) if np.isfinite(realized) and np.isfinite(online) else "—"],
        ]
        story.append(tbl(tdata, col_widths=[220, 260]))
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
            "<b>Sobre α (alpha_global)</b>: é um fator (0..1) aplicado a todos os stakes na semana, "
            "calculado no treino de cada passo para satisfazer constraints de risco globais (exposição diária e drawdown). "
            "Na prática, o stake efetivo por aposta é <b>min(banca × stake_frac × α, house_cap)</b>.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "<b>Interpretação (por que pode divergir do previsto)</b>: mesmo quando o scoring está correto, "
            "o PnL semanal tem alta variância e depende de: (i) poucas observações na semana, (ii) distribuição de odds/ROI "
            "capada (cap2), (iii) mudança de regime (mercado), (iv) mudança do conjunto de segmentos ativos (política dinâmica), "
            "e (v) α reduzir/exacerbar exposição conforme restrições globais.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "<b>Aplicabilidade do modelo</b>: este relatório avalia a política como geradora de decisões. "
            "A aderência operacional deve ser checada via `bet_id` (payload + log JSONL + score gravado). "
            "Se a aderência for 100%, divergências entre previsto e realizado passam a ser atribuídas ao risco/variância do mercado "
            "e à incerteza estatística (não a erro de pipeline).",
            styles["Normal"],
        )
    )

    doc = SimpleDocTemplate(str(out_pdf), pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    doc.build(story)
    print(str(out_pdf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

