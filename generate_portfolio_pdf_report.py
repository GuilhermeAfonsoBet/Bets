#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera um PDF com:
1) Data
2) Critérios utilizados
3) Portfólio selecionado (DoW x Tipo x Score x Stake)
4) Métricas estatísticas globais do portfólio
5) Mesmas métricas por combinação

Fontes:
 - /workspace/analysis_proba_raw/pro_portfolio_all/portfolio_pro_all.json
 - /workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv

Saída:
 - /workspace/analysis_proba_raw/pro_portfolio_all/Relatorio_Portfolio_Mesa_Profissional.pdf
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors


JSON_PATH = Path("/workspace/analysis_proba_raw/pro_portfolio_all/portfolio_pro_all.json")
CSV_PATH = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")
OUT_PDF = Path("/workspace/analysis_proba_raw/pro_portfolio_all/Relatorio_Portfolio_Mesa_Profissional.pdf")

# Período de treino (para as métricas deste relatório)
TRAIN_START = pd.Timestamp("2025-10-01")
TRAIN_END = pd.Timestamp("2025-12-31 23:59:59")

# Bootstrap / sims globais
N_BOOT_ANNUAL = 50_000
N_BOOT_DD = 20_000
SEED = 7


WEEKDAY_PT = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo"]


def safe_cap(x) -> float:
    try:
        v = float(x)
    except Exception:
        return float("inf")
    if not np.isfinite(v) or v <= 0:
        return float("inf")
    return v


def roi_cap(arr: np.ndarray, cap: float) -> np.ndarray:
    return np.minimum(arr.astype(float), float(cap))


def week_key(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts).dt.to_period("W-SUN").astype(str)


def date_key(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts).dt.date.astype(str)


def apply_rules(df: pd.DataFrame, portfolio: Dict, bankroll: float, roi_mode: str) -> pd.DataFrame:
    """
    Aplica todas as combinações do portfólio (FT+FH) e retorna apostas selecionadas.
    roi_mode: 'cap2' ou 'cap1' ou 'raw'
    """
    if roi_mode == "raw":
        roi = pd.to_numeric(df["ROI Real"], errors="coerce").to_numpy(dtype=float)
    elif roi_mode == "cap2":
        roi = roi_cap(pd.to_numeric(df["ROI Real"], errors="coerce").to_numpy(dtype=float), 2.0)
    elif roi_mode == "cap1":
        roi = roi_cap(pd.to_numeric(df["ROI Real"], errors="coerce").to_numpy(dtype=float), 1.0)
    else:
        raise ValueError(roi_mode)
    roi_s = pd.Series(roi, index=df.index)

    out_rows = []
    for bet_type in ["FT", "FH"]:
        for dow in WEEKDAY_PT:
            rule = portfolio[bet_type][dow]
            if float(rule["stake_frac"]) <= 0:
                continue
            sc = rule["score_col"]
            cut = float(rule["cutoff"])
            frac = float(rule["stake_frac"])
            x = df[(df["dow_pt"] == dow) & (df["bet_type"] == bet_type)].copy()
            if x.empty:
                continue
            score = pd.to_numeric(x[sc], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(score) & (score >= cut)
            if not np.any(m):
                continue
            stake0 = bankroll * frac
            capv = x["house_cap"].to_numpy(dtype=float)
            stake = np.minimum(stake0, capv[m])
            profit = stake * roi_s.loc[x.index].to_numpy(dtype=float)[m]
            x2 = x.iloc[np.where(m)[0]].copy()
            x2["stake_usd"] = stake
            x2["profit_usd"] = profit
            x2["portfolio_dow"] = dow
            x2["portfolio_type"] = bet_type
            x2["portfolio_score_col"] = sc
            x2["portfolio_cutoff"] = cut
            x2["portfolio_stake_frac"] = frac
            out_rows.append(x2)

    if not out_rows:
        return pd.DataFrame()
    return pd.concat(out_rows, axis=0, ignore_index=True)


@dataclass(frozen=True)
class GlobalMetrics:
    weeks: int
    mean_week: float
    std_week: float
    sharpe_week: float
    sharpe_ann: float
    ann_profit_mean: float
    ann_profit_p05: float
    ann_profit_p50: float
    ann_profit_p95: float
    dd_p95: float
    max_daily_loss_p05: float
    p_daily_loss_ge_25pct: float


def bootstrap_annual_profit(weekly_pnl: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = np.asarray(weekly_pnl, dtype=float)
    w = w[np.isfinite(w)]
    idx = rng.integers(0, len(w), size=(N_BOOT_ANNUAL, 52))
    return w[idx].sum(axis=1)


def bootstrap_annual_sharpe(weekly_pnl: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)
    w = np.asarray(weekly_pnl, dtype=float)
    w = w[np.isfinite(w)]
    idx = rng.integers(0, len(w), size=(N_BOOT_ANNUAL, len(w)))
    samples = w[idx]
    means = samples.mean(axis=1)
    stds = samples.std(axis=1, ddof=1)
    sharpe_week = np.where(stds > 0, means / stds, np.nan)
    return sharpe_week * np.sqrt(52.0)


def bootstrap_drawdown_p95(weekly_pnl: np.ndarray, bankroll0: float, seed: int) -> float:
    rng = np.random.default_rng(seed + 2)
    w = np.asarray(weekly_pnl, dtype=float)
    w = w[np.isfinite(w)]
    idx = rng.integers(0, len(w), size=(N_BOOT_DD, 52))
    pnl = w[idx]
    bank = bankroll0 + np.cumsum(pnl, axis=1)
    peak = np.maximum.accumulate(bank, axis=1)
    dd = (peak - bank).max(axis=1)
    return float(np.quantile(dd, 0.95))


def compute_global_metrics(df_sel: pd.DataFrame, bankroll: float) -> GlobalMetrics:
    df_sel = df_sel.copy()
    df_sel["week"] = week_key(df_sel["BIA_ApostaUTC"])
    weekly = df_sel.groupby("week")["profit_usd"].sum().sort_index()
    w = weekly.to_numpy(dtype=float)
    mean_w = float(w.mean()) if w.size else 0.0
    std_w = float(w.std(ddof=1)) if w.size >= 2 else 0.0
    sharpe_w = float(mean_w / std_w) if std_w > 0 else (float("inf") if mean_w > 0 else 0.0)
    sharpe_a = float(sharpe_w * np.sqrt(52.0)) if np.isfinite(sharpe_w) else sharpe_w

    ann = bootstrap_annual_profit(w, seed=SEED)
    dd_p95 = bootstrap_drawdown_p95(w, bankroll0=bankroll, seed=SEED)

    # daily loss metrics (cap2 etc já embutido em profit_usd)
    df_sel["date"] = date_key(df_sel["BIA_ApostaUTC"])
    daily = df_sel.groupby("date")["profit_usd"].sum()
    losses = daily.to_numpy(dtype=float)
    max_daily_loss_p05 = float(np.quantile(losses, 0.05)) if losses.size else 0.0
    p_loss_25 = float((losses <= (-0.25 * bankroll)).mean()) if losses.size else 0.0

    return GlobalMetrics(
        weeks=int(w.size),
        mean_week=mean_w,
        std_week=std_w,
        sharpe_week=sharpe_w,
        sharpe_ann=sharpe_a,
        ann_profit_mean=float(ann.mean()),
        ann_profit_p05=float(np.quantile(ann, 0.05)),
        ann_profit_p50=float(np.quantile(ann, 0.50)),
        ann_profit_p95=float(np.quantile(ann, 0.95)),
        dd_p95=dd_p95,
        max_daily_loss_p05=max_daily_loss_p05,
        p_daily_loss_ge_25pct=p_loss_25,
    )


def table_from_dict_rows(rows: List[Dict], col_order: List[str], title_map: Dict[str, str]) -> Table:
    data = [[title_map.get(c, c) for c in col_order]]
    for r in rows:
        data.append([r.get(c, "") for c in col_order])
    t = Table(data, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def main() -> int:
    j = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    portfolio = j["portfolio"]
    stability = j["stability"]
    criteria = j.get("daily_risk_constraints", {})
    bankroll = float(j["bankroll"])

    df = pd.read_csv(CSV_PATH, parse_dates=["BIA_ApostaUTC"])
    df = df[(df["BIA_ApostaUTC"] >= TRAIN_START) & (df["BIA_ApostaUTC"] <= TRAIN_END)].copy()
    df["house_cap"] = df["house_cap"].apply(safe_cap)
    # ROI Real pode ter NaNs; o apply_rules filtra via score + profit calc.

    # Portfolio global metrics (cap2 as principal) + raw for reference
    sel_cap2 = apply_rules(df, portfolio, bankroll=bankroll, roi_mode="cap2")
    sel_cap1 = apply_rules(df, portfolio, bankroll=bankroll, roi_mode="cap1")
    sel_raw = apply_rules(df, portfolio, bankroll=bankroll, roi_mode="raw")

    gm_cap2 = compute_global_metrics(sel_cap2, bankroll)
    gm_cap1 = compute_global_metrics(sel_cap1, bankroll)
    gm_raw = compute_global_metrics(sel_raw, bankroll)

    # Per-combo table using stored stability metrics
    combo_rows = []
    for bet_type in ["FT", "FH"]:
        for dow in WEEKDAY_PT:
            rule = portfolio[bet_type][dow]
            rep = stability[bet_type].get(dow, {})
            if float(rule.get("stake_frac", 0.0)) <= 0:
                continue
            combo_rows.append(
                {
                    "Tipo": bet_type,
                    "Dia": dow,
                    "Score": rule["score_col"],
                    "Cut": f"{float(rule['cutoff']):.2f}",
                    "Stake%": f"{float(rule['stake_frac'])*100:.1f}%",
                    "mean_week_cap2": f"{rep.get('cap2_mean_week', 0):.1f}",
                    "std_week_cap2": f"{rep.get('cap2_std_week', 0):.1f}",
                    "sharpe_week_cap2": f"{rep.get('sharpe_week_cap2', 0):.2f}",
                    "pneg_week_cap2": f"{rep.get('cap2_pneg_week', 0)*100:.1f}%",
                    "VaR10d": f"{rep.get('daily_var_q', 0):.1f}",
                    "P(d<=-25%)": f"{rep.get('p_daily_dd', 0)*100:.1f}%",
                    "bins+": f"{rep.get('score_bins_cap2', {}).get('pos', '')}/{rep.get('score_bins_cap2', {}).get('n', '')}",
                }
            )

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT_PDF), pagesize=A4, rightMargin=1.5 * cm, leftMargin=1.5 * cm, topMargin=1.2 * cm, bottomMargin=1.2 * cm)
    styles = getSampleStyleSheet()
    story = []

    today = dt.datetime.now().date().isoformat()
    story.append(Paragraph("Relatório — Portfólio Mesa Profissional (proba_raw)", styles["Title"]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(f"<b>Data:</b> {today}", styles["Normal"]))
    story.append(Paragraph(f"<b>Janela de análise (treino):</b> {TRAIN_START.date()} .. {TRAIN_END.date()}", styles["Normal"]))
    story.append(Paragraph(f"<b>Banca:</b> USD {bankroll:,.0f}", styles["Normal"]))
    story.append(Spacer(1, 0.3 * cm))

    # Criteria section
    story.append(Paragraph("Critérios utilizados na seleção", styles["Heading2"]))
    crit_lines = [
        "- Score: proba_raw (operacional, clipped)",
        "- Stress tests: cap2 (ROI<=2) e cap1 (ROI<=1)",
        f"- Regra diária: VaR{int(criteria.get('daily_var_quantile', 0.1)*100)}% do PnL diário >= -{int(criteria.get('max_daily_drawdown_frac', 0.25)*100)}% banca",
        f"- Regra diária: P(PnL_dia <= -25% banca) <= {int(criteria.get('max_p_daily_dd', 0.10)*100)}%",
        f"- Sharpe semanal mínimo (cap2): {criteria.get('min_weekly_sharpe_cap2', 0.1)}",
        "- Estabilidade por score (cap2): >= 4/5 bins positivos (quantis acima do cutoff)",
        "- Bootstrap semanal: VaR/CVaR anual (52 semanas) e simulação de drawdown (p95)",
    ]
    story.append(Paragraph("<br/>".join(crit_lines), styles["BodyText"]))
    story.append(Spacer(1, 0.3 * cm))

    # Portfolio section
    story.append(Paragraph("Portfólio selecionado (DoW × Tipo × Score × Stake)", styles["Heading2"]))
    rows_rules = []
    for bet_type in ["FT", "FH"]:
        for dow in WEEKDAY_PT:
            r = portfolio[bet_type][dow]
            if float(r["stake_frac"]) <= 0:
                continue
            rows_rules.append(
                {
                    "Tipo": bet_type,
                    "Dia": dow,
                    "Score": r["score_col"],
                    "Cut": f"{float(r['cutoff']):.2f}",
                    "Stake%": f"{float(r['stake_frac'])*100:.1f}%",
                    "Stake USD": f"{bankroll*float(r['stake_frac']):.0f}",
                }
            )
    rules_table = table_from_dict_rows(
        rows_rules,
        ["Tipo", "Dia", "Score", "Cut", "Stake%", "Stake USD"],
        {"Stake USD": "Stake (USD)"},
    )
    story.append(rules_table)
    story.append(PageBreak())

    # Global metrics
    def gm_block(title: str, gm: GlobalMetrics) -> List:
        b = []
        b.append(Paragraph(title, styles["Heading2"]))
        b.append(
            Paragraph(
                f"- Semanas: <b>{gm.weeks}</b><br/>"
                f"- PnL semanal médio: <b>USD {gm.mean_week:.1f}</b> | std: <b>USD {gm.std_week:.1f}</b> | Sharpe semanal: <b>{gm.sharpe_week:.2f}</b><br/>"
                f"- Sharpe anual (aprox): <b>{gm.sharpe_ann:.2f}</b><br/>"
                f"- Lucro anual (bootstrap 52s): mean <b>USD {gm.ann_profit_mean:.0f}</b>, p05 <b>USD {gm.ann_profit_p05:.0f}</b>, p50 <b>USD {gm.ann_profit_p50:.0f}</b>, p95 <b>USD {gm.ann_profit_p95:.0f}</b><br/>"
                f"- Drawdown p95 (paths 52s): <b>USD {gm.dd_p95:.0f}</b><br/>"
                f"- PnL diário p05: <b>USD {gm.max_daily_loss_p05:.0f}</b> | P(PnL_dia <= -25% banca): <b>{gm.p_daily_loss_ge_25pct*100:.1f}%</b>",
                styles["BodyText"],
            )
        )
        return b

    story.extend(gm_block("Métricas globais — cenário principal (cap2)", gm_cap2))
    story.append(Spacer(1, 0.2 * cm))
    story.extend(gm_block("Métricas globais — stress (cap1)", gm_cap1))
    story.append(Spacer(1, 0.2 * cm))
    story.extend(gm_block("Métricas globais — referência (raw)", gm_raw))
    story.append(PageBreak())

    # Per combo metrics table
    story.append(Paragraph("Métricas por combinação (cap2)", styles["Heading2"]))
    combo_table = table_from_dict_rows(
        combo_rows,
        [
            "Tipo",
            "Dia",
            "Score",
            "Cut",
            "Stake%",
            "mean_week_cap2",
            "std_week_cap2",
            "sharpe_week_cap2",
            "pneg_week_cap2",
            "VaR10d",
            "P(d<=-25%)",
            "bins+",
        ],
        {
            "mean_week_cap2": "mean_w",
            "std_week_cap2": "std_w",
            "sharpe_week_cap2": "Sharpe_w",
            "pneg_week_cap2": "P(w<0)",
            "VaR10d": "VaR10% dia",
            "P(d<=-25%)": "P(d<=-25%)",
            "bins+": "bins +",
        },
    )
    story.append(combo_table)

    doc.build(story)
    print(str(OUT_PDF))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

