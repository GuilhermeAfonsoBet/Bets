#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walk-forward semanal (expanding window) para reduzir viés de otimização in-sample.

Para cada semana t no período de treino:
  - otimiza cutoffs por dia usando apenas semanas < t
  - aplica na semana t e registra PnL

Faz isso para cenários de ROI: raw / cap2 / cap1.

Outputs:
  /workspace/analysis_proba_raw/robust_weekly/walkforward_weekly.csv
  /workspace/analysis_proba_raw/robust_weekly/walkforward_summary.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw.csv")
CFG = json.loads(Path("/workspace/analysis_proba_raw/portfolio_proba_raw_reoptimized.json").read_text(encoding="utf-8"))
OUT_DIR = Path("/workspace/analysis_proba_raw/robust_weekly")

TRAIN_START = pd.Timestamp("2025-10-01")
TRAIN_END = pd.Timestamp("2025-12-31 23:59:59")

BANK = float(CFG["bankroll"])
MAX_FRAC = float(CFG["max_frac_per_bet"])

RNG_SEED = 7


def is_ft(tipo: str) -> bool:
    return "first half" not in str(tipo).lower()


def safe_house_cap(x) -> float:
    try:
        v = float(x)
    except Exception:
        return float("inf")
    if not np.isfinite(v) or v <= 0:
        return float("inf")
    return v


def cap_roi(a: np.ndarray, cap: float) -> np.ndarray:
    return np.minimum(a, cap)


def weekly_key(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts).dt.to_period("W-SUN").astype(str)


def optimize_cutoff_for_day(
    df_train: pd.DataFrame,
    day: str,
    score_col: str,
    stake_frac: float,
    min_total_bets: int,
    min_bets_per_week: int,
) -> float:
    """
    Otimiza cutoff por semana (objetivo: mean - 0.25*std do PnL semanal).
    """
    x = df_train[df_train["dow_pt"] == day].copy()
    if x.empty:
        return 1.0

    cutoffs = np.round(np.arange(0.05, 0.951, 0.01), 2)
    best_cut = 1.0
    best_obj = -np.inf

    stake0 = BANK * stake_frac

    for c in cutoffs:
        s = x[x[score_col] >= c]
        if s.empty:
            continue
        # robustez por volume
        n_total = int(len(s))
        if n_total < min_total_bets:
            continue
        g = s.groupby("week").size()
        if g.size < 4:  # pelo menos 4 semanas históricas
            continue
        if int(g.min()) < min_bets_per_week:
            continue

        stake = np.minimum(stake0, s["house_cap"].to_numpy(dtype=float))
        pnl = stake * s["roi"].to_numpy(dtype=float)
        wp = pd.Series(pnl, index=s["week"]).groupby(level=0).sum().sort_index()
        mean = float(wp.mean())
        std = float(wp.std(ddof=1)) if wp.size >= 2 else 0.0
        obj = mean - 0.25 * std
        if obj > best_obj:
            best_obj = obj
            best_cut = float(c)

    return best_cut


def run_walkforward(df: pd.DataFrame, roi_mode: str) -> pd.DataFrame:
    # stake_frac: como na solução final, acabou 7% em todos os dias; fixamos para reduzir dimensão
    stake_frac = MAX_FRAC

    # score cols por dia (como você pediu)
    score_map = {
        "segunda-feira": "proba_raw_segunda",
        "terça-feira": "proba_raw_terca",
        "quarta-feira": "proba_raw_quarta",
        "quinta-feira": "proba_raw_segqui",
    }

    # preparar ROI conforme cenário
    roi = pd.to_numeric(df["ROI Real"], errors="coerce").to_numpy(dtype=float)
    if roi_mode == "raw":
        roi2 = roi
    elif roi_mode == "cap2":
        roi2 = cap_roi(roi, 2.0)
    elif roi_mode == "cap1":
        roi2 = cap_roi(roi, 1.0)
    else:
        raise ValueError(roi_mode)

    work = df.copy()
    work["roi"] = roi2
    work["week"] = weekly_key(work["BIA_ApostaUTC"])

    # semanas no período de treino
    in_train = (work["BIA_ApostaUTC"] >= TRAIN_START) & (work["BIA_ApostaUTC"] <= TRAIN_END)
    weeks = sorted(work.loc[in_train, "week"].unique().tolist())

    rows = []
    for i, wk in enumerate(weeks):
        # exige histórico mínimo antes de começar a testar
        hist_weeks = weeks[:i]
        if len(hist_weeks) < 4:
            continue
        train_df = work[in_train & work["week"].isin(hist_weeks)].copy()
        test_df = work[in_train & (work["week"] == wk)].copy()

        # otimizar cutoffs com histórico
        cuts = {}
        for day, sc in score_map.items():
            # robustez por dia
            if day in ("segunda-feira", "quarta-feira"):
                min_total, min_per_week = 40, 3
            else:
                min_total, min_per_week = 25, 2
            cuts[day] = optimize_cutoff_for_day(train_df, day, sc, stake_frac, min_total, min_per_week)

        # aplicar na semana wk
        stake0 = BANK * stake_frac
        pnl_parts = []
        n_bets = 0
        for day, sc in score_map.items():
            x = test_df[test_df["dow_pt"] == day]
            if x.empty:
                continue
            x = x[np.isfinite(x["roi"].to_numpy(dtype=float))]
            x = x[np.isfinite(pd.to_numeric(x[sc], errors="coerce").to_numpy(dtype=float))]
            x = x[pd.to_numeric(x[sc], errors="coerce").to_numpy(dtype=float) >= cuts[day]]
            if x.empty:
                continue
            stake = np.minimum(stake0, x["house_cap"].to_numpy(dtype=float))
            pnl = stake * x["roi"].to_numpy(dtype=float)
            pnl_parts.append(pnl.sum())
            n_bets += int(len(x))

        profit_week = float(np.sum(pnl_parts)) if pnl_parts else 0.0
        rows.append(
            {
                "roi_mode": roi_mode,
                "week": wk,
                "profit_usd": profit_week,
                "n_bets": n_bets,
                "cut_seg": cuts["segunda-feira"],
                "cut_ter": cuts["terça-feira"],
                "cut_qua": cuts["quarta-feira"],
                "cut_qui": cuts["quinta-feira"],
            }
        )

    return pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])
    df["house_cap"] = df["house_cap"].apply(safe_house_cap)
    df = df[df["Tipo Aposta"].apply(is_ft)].copy()
    df = df[np.isfinite(pd.to_numeric(df["ROI Real"], errors="coerce").to_numpy(dtype=float))].copy()

    all_res = pd.concat([run_walkforward(df, m) for m in ["raw", "cap2", "cap1"]], axis=0, ignore_index=True)
    all_res.to_csv(OUT_DIR / "walkforward_weekly.csv", index=False)

    # resumo
    lines = []
    lines.append("## Walk-forward semanal (expanding window) — treino (out–dez/2025)\n")
    lines.append(f"- stake_frac fixo: {MAX_FRAC*100:.1f}% da banca (USD {BANK*MAX_FRAC:.0f})\n")
    for mode in ["raw", "cap2", "cap1"]:
        x = all_res[all_res["roi_mode"] == mode]
        if x.empty:
            continue
        p = x["profit_usd"].to_numpy(dtype=float)
        lines.append(f"\n### {mode}\n")
        lines.append(f"- semanas testadas: {len(x)}\n")
        lines.append(f"- PnL semanal médio: USD {p.mean():.0f}\n")
        lines.append(f"- PnL semanal std: USD {p.std(ddof=1) if len(p)>=2 else 0:.0f}\n")
        lines.append(f"- P(semana<0): {(p<0).mean()*100:.1f}%\n")
        lines.append(f"- p05: {np.quantile(p,0.05):.0f} | p50: {np.quantile(p,0.5):.0f} | p95: {np.quantile(p,0.95):.0f}\n")

    (OUT_DIR / "walkforward_summary.md").write_text("".join(lines), encoding="utf-8")
    print(str(OUT_DIR / "walkforward_summary.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

