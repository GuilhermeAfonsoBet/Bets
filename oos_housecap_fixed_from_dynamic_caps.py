#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constrói um cap_max FIXO por segmento usando caps dinâmicos (OOS) de um período inicial,
e avalia a estratégia (re-otimizando cutoff/stake semanalmente) com esse cap fixo aplicado.

Motivação: cap fixo por segmento (menos overfit) + hipótese de que cap torna alguns segmentos viáveis.

Procedimento:
1) Usa `oos_walkforward_housecap_gating_rules.csv` (caps dinâmicos já calculados) para derivar,
   por segmento, um cap fixo (mediana dos caps finitos) usando apenas semanas iniciais (cap-train).
2) Roda walk-forward p10_p70 a partir da semana seguinte, aplicando o filtro house_cap<=cap_fixed
   tanto no treino quanto no teste (por segmento).

Saídas:
- analysis_proba_raw/pro_portfolio_all/oos_walkforward_housecap_capfixed2_weekly.csv
- analysis_proba_raw/pro_portfolio_all/oos_walkforward_housecap_capfixed2_selected_rules.csv
- analysis_proba_raw/pro_portfolio_all/oos_walkforward_housecap_capfixed2_summary.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import evaluate_oos_walkforward_strategy as wf


OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")
SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")
MODE_BASE = "global_bayes_roll12_robust_p10_p70"
WF_WEEKLY = OUT_DIR / f"oos_walkforward_{MODE_BASE}_weekly.csv"
WF_RULES = OUT_DIR / f"oos_walkforward_{MODE_BASE}_selected_rules.csv"
DYN_CAPS = OUT_DIR / "oos_walkforward_housecap_gating_rules.csv"

TRAIN_WINDOW_WEEKS = 12
CAP_TRAIN_WEEKS = 8  # usar primeiras N semanas do WF para inferir caps fixos


def _ensure_calibrated(df: pd.DataFrame) -> pd.DataFrame:
    import json

    df = df.copy()
    df["proba_cal_segqui"] = df.get("proba_cal_segqui", np.nan)
    df["proba_cal_sexdom"] = df.get("proba_cal_sexdom", np.nan)
    if "proba_raw_segqui" in df.columns and wf.CALIB_SEGQUI.exists():
        obj = json.loads(wf.CALIB_SEGQUI.read_text(encoding="utf-8"))
        x = np.asarray(obj.get("isotonic", {}).get("x", []), float)
        y = np.asarray(obj.get("isotonic", {}).get("y", []), float)
        p = pd.to_numeric(df["proba_raw_segqui"], errors="coerce").to_numpy(float)
        df["proba_cal_segqui"] = wf._apply_isotonic_vec(p, x=x, y=y, floor=wf.CALIB_FLOOR)
    if "proba_raw_sexdom" in df.columns and wf.CALIB_SEXDOM.exists():
        obj = json.loads(wf.CALIB_SEXDOM.read_text(encoding="utf-8"))
        x = np.asarray(obj.get("isotonic", {}).get("x", []), float)
        y = np.asarray(obj.get("isotonic", {}).get("y", []), float)
        p = pd.to_numeric(df["proba_raw_sexdom"], errors="coerce").to_numpy(float)
        df["proba_cal_sexdom"] = wf._apply_isotonic_vec(p, x=x, y=y, floor=wf.CALIB_FLOOR)
    return df


def compute_fixed_caps(weeks: List[str]) -> Dict[str, float]:
    if not DYN_CAPS.exists():
        raise FileNotFoundError(f"Arquivo de caps dinâmicos não encontrado: {DYN_CAPS}")
    dyn = pd.read_csv(DYN_CAPS)
    train_weeks = weeks[: min(CAP_TRAIN_WEEKS, len(weeks))]
    d0 = dyn[dyn["test_week"].astype(str).isin(train_weeks)].copy()
    out: Dict[str, float] = {}
    for rk, g in d0.groupby("rule_key", sort=False):
        a = pd.to_numeric(g["cap_max"], errors="coerce").to_numpy(float)
        a = a[np.isfinite(a) & (a > 0) & (a < 1e12)]
        if a.size:
            out[str(rk)] = float(np.median(a))
        else:
            out[str(rk)] = float("inf")
    # ensure all segments present
    for bt in ("FT", "FH"):
        for dow in wf.WEEKDAY_PT:
            rk = f"{bt}|{dow}"
            out.setdefault(rk, float("inf"))
    return out


def run_wf(df: pd.DataFrame, weeks: List[str], caps: Dict[str, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # set p10_p70
    wf.POST_Q_OBJ = 0.10
    wf.MIN_POST_P_MEAN_POS = 0.70
    wf.ROBUST_CUTOFF_ENABLED = True
    wf.HYSTERESIS_ENABLED = True
    wf.ROBUST_CUTOFF_DELTA = 0.02
    wf.HYST_P_SWITCH = 0.90
    wf.BAYES_N = 2000

    rules_rows = []
    weekly_rows = []
    prev_rules: Dict[str, wf.Rule] = {}

    for i in range(max(wf.MIN_GLOBAL_TRAIN_WEEKS, CAP_TRAIN_WEEKS), len(weeks)):
        w_test = weeks[i]
        train_weeks = weeks[max(0, i - TRAIN_WINDOW_WEEKS) : i]
        df_train = df[df["week"].isin(train_weeks)].copy()
        df_test = df[df["week"] == w_test].copy()

        rules: Dict[str, wf.Rule] = {}
        for bt in ("FT", "FH"):
            for dow in wf.WEEKDAY_PT:
                rk = f"{bt}|{dow}"
                cap_max = float(caps.get(rk, float("inf")))
                sc = wf.segment_score_col(dow)
                x = df_train[(df_train["bet_type"] == bt) & (df_train["dow_pt"] == dow)].copy()
                if np.isfinite(cap_max):
                    x = x[np.isfinite(x["house_cap"]) & (x["house_cap"] <= cap_max)].copy()
                if x.empty:
                    rule = wf.Rule(bt, dow, sc, 1.0, 0.0, "no_data", cap_max=cap_max)
                else:
                    prev = prev_rules.get(rk)
                    r0 = wf.optimize_segment_train(x, sc, bayes_select=True, prev_rule=prev, roi_bias_adj=0.0)
                    rule = wf.Rule(r0.bet_type, r0.dow, r0.score_col, r0.cutoff, r0.stake_frac, r0.status, cap_max=cap_max)
                rules[rk] = rule

        alpha, _, _ = wf.find_alpha_global(df_train, rules)
        bets = wf.apply_rules_on_df(df_test, rules, alpha=float(alpha))
        stake = float(bets["stake_eff"].sum()) if not bets.empty else 0.0
        pnl = float(bets["profit_cap2"].sum()) if not bets.empty else 0.0
        weekly_rows.append({"week": w_test, "alpha_global": float(alpha), "stake_usd": stake, "profit_cap2_usd": pnl, "n_bets": int(len(bets))})

        for key, r in rules.items():
            rules_rows.append({"test_week": w_test, "rule_key": key, "bet_type": r.bet_type, "dow_pt": r.dow, "score_col": r.score_col, "cutoff": float(r.cutoff), "stake_frac": float(r.stake_frac), "cap_max": float(r.cap_max), "status": r.status, "alpha_global": float(alpha)})

        prev_rules = rules.copy()

    return pd.DataFrame(weekly_rows), pd.DataFrame(rules_rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_week = pd.read_csv(WF_WEEKLY)
    weeks = base_week["week"].astype(str).tolist()
    caps = compute_fixed_caps(weeks)

    df = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])
    df["week"] = wf.week_key(df["BIA_ApostaUTC"]).astype(str)
    df["date"] = wf.date_key(df["BIA_ApostaUTC"])
    df["house_cap"] = df["house_cap"].apply(wf.safe_cap)
    df["roi_raw"] = pd.to_numeric(df["roi_calc"], errors="coerce").to_numpy(float)
    df["roi_cap2"] = np.minimum(df["roi_raw"].to_numpy(float), 2.0)
    df["roi_cap1"] = np.minimum(df["roi_raw"].to_numpy(float), 1.0)
    df = _ensure_calibrated(df)

    weekly_df, rules_df = run_wf(df, weeks, caps)

    # compare on eval weeks only (weeks after CAP_TRAIN_WEEKS)
    eval_weeks = weeks[max(wf.MIN_GLOBAL_TRAIN_WEEKS, CAP_TRAIN_WEEKS) :]
    base_eval = base_week[base_week["week"].astype(str).isin(eval_weeks)]
    new_eval = weekly_df[weekly_df["week"].astype(str).isin(eval_weeks)]
    stake_b = float(base_eval["stake_usd"].sum())
    pnl_b = float(base_eval["profit_cap2_usd"].sum())
    stake_n = float(new_eval["stake_usd"].sum())
    pnl_n = float(new_eval["profit_cap2_usd"].sum())
    summ = pd.DataFrame(
        [
            {"name": "baseline_eval", "profit_cap2_total": pnl_b, "stake_total": stake_b, "roi_total_cap2": pnl_b / stake_b if stake_b > 0 else np.nan, "weeks": int(len(base_eval))},
            {"name": "capfixed_eval", "profit_cap2_total": pnl_n, "stake_total": stake_n, "roi_total_cap2": pnl_n / stake_n if stake_n > 0 else np.nan, "weeks": int(len(new_eval))},
        ]
    )

    out_week = OUT_DIR / "oos_walkforward_housecap_capfixed2_weekly.csv"
    out_rules = OUT_DIR / "oos_walkforward_housecap_capfixed2_selected_rules.csv"
    out_sum = OUT_DIR / "oos_walkforward_housecap_capfixed2_summary.csv"
    weekly_df.to_csv(out_week, index=False)
    rules_df.to_csv(out_rules, index=False)
    summ.to_csv(out_sum, index=False)
    print(str(out_sum))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

