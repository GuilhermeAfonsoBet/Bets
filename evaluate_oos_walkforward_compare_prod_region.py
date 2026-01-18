#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compara walk-forward p10_p70 usando:
 - score atual (segment_score_col)
 - score produção-like (payload-only) + score produção-like (payload+região)

Scores externos são gerados por:
  build_score_prod_weekly_scores_payload_region.py

Saídas:
  /workspace/analysis_proba_raw/pro_portfolio_all/oos_walkforward_score_compare_prod_region_wf12_p10_p70_summary.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import evaluate_oos_walkforward_strategy as wf


SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")
SCORES = Path("/workspace/analysis_proba_raw/pro_portfolio_all/scored_with_score_prod_payload_region_wf12.csv")
OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")

TRAIN_WINDOW_WEEKS = 12


def _ensure_calibrated_cols(df: pd.DataFrame) -> pd.DataFrame:
    import json

    df = df.copy()
    if "proba_cal_segqui" not in df.columns:
        df["proba_cal_segqui"] = np.nan
    if "proba_cal_sexdom" not in df.columns:
        df["proba_cal_sexdom"] = np.nan
    if "proba_raw_segqui" in df.columns and wf.CALIB_SEGQUI.exists():
        calib = json.loads(wf.CALIB_SEGQUI.read_text(encoding="utf-8"))
        x = np.asarray(calib.get("isotonic", {}).get("x", []), dtype=float)
        y = np.asarray(calib.get("isotonic", {}).get("y", []), dtype=float)
        p_raw = pd.to_numeric(df["proba_raw_segqui"], errors="coerce").to_numpy(dtype=float)
        df["proba_cal_segqui"] = wf._apply_isotonic_vec(p_raw, x=x, y=y, floor=wf.CALIB_FLOOR)
    if "proba_raw_sexdom" in df.columns and wf.CALIB_SEXDOM.exists():
        calib = json.loads(wf.CALIB_SEXDOM.read_text(encoding="utf-8"))
        x = np.asarray(calib.get("isotonic", {}).get("x", []), dtype=float)
        y = np.asarray(calib.get("isotonic", {}).get("y", []), dtype=float)
        p_raw = pd.to_numeric(df["proba_raw_sexdom"], errors="coerce").to_numpy(dtype=float)
        df["proba_cal_sexdom"] = wf._apply_isotonic_vec(p_raw, x=x, y=y, floor=wf.CALIB_FLOOR)
    return df


def _run_wf(df: pd.DataFrame, score_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    weeks = sorted(df["week"].astype(str).unique().tolist())
    weekly_rows = []
    prev_rules: Dict[str, wf.Rule] = {}

    for i in range(wf.MIN_GLOBAL_TRAIN_WEEKS, len(weeks)):
        w_test = weeks[i]
        train_weeks = weeks[max(0, i - TRAIN_WINDOW_WEEKS) : i]
        df_train = df[df["week"].astype(str).isin(train_weeks)].copy()
        df_test = df[df["week"].astype(str) == w_test].copy()

        rules: Dict[str, wf.Rule] = {}
        for bet_type in ("FT", "FH"):
            for dow in wf.WEEKDAY_PT:
                sc = score_col if score_col != "score_current" else wf.segment_score_col(dow)
                x = df_train[(df_train["dow_pt"] == dow) & (df_train["bet_type"] == bet_type)].copy()
                if x.empty or sc not in x.columns:
                    rule = wf.Rule(bet_type=bet_type, dow=dow, score_col=sc, cutoff=1.0, stake_frac=0.0, status="no_data")
                else:
                    prev = prev_rules.get(f"{bet_type}|{dow}")
                    rule = wf.optimize_segment_train(x, sc, bayes_select=True, prev_rule=prev, roi_bias_adj=0.0)
                rules[f"{bet_type}|{dow}"] = rule

        alpha, _, _ = wf.find_alpha_global(df_train, rules)
        bets = wf.apply_rules_on_df(df_test, rules, alpha=float(alpha))

        stake_sum = float(bets["stake_eff"].sum()) if len(bets) else 0.0
        pnl_sum = float(bets["profit_cap2"].sum()) if len(bets) else 0.0
        n_bets = int(len(bets))
        weekly_rows.append(
            {
                "week": w_test,
                "train_weeks": int(len(train_weeks)),
                "alpha_global": float(alpha),
                "n_bets": n_bets,
                "stake_usd": stake_sum,
                "profit_cap2_usd": pnl_sum,
                "roi_on_stake_cap2": float(pnl_sum / stake_sum) if stake_sum > 0 else float("nan"),
            }
        )
        prev_rules = rules.copy()

    wk = pd.DataFrame(weekly_rows)

    stake = float(wk["stake_usd"].sum())
    pnl = float(wk["profit_cap2_usd"].sum())
    roi = float(pnl / stake) if stake > 0 else float("nan")
    w_nonzero = wk.loc[wk["stake_usd"] > 0, "profit_cap2_usd"].to_numpy(float)
    summ = pd.DataFrame(
        [{
            "name": score_col,
            "profit_cap2_total": pnl,
            "stake_total": stake,
            "roi_total_cap2": roi,
            "weeks_total": int(len(wk)),
            "weeks_with_stake": int((wk["stake_usd"] > 0).sum()),
            "mean_weekly_cap2_nonzero": float(np.mean(w_nonzero)) if w_nonzero.size else float("nan"),
            "pneg_weeks_nonzero": float((w_nonzero < 0).mean()) if w_nonzero.size else float("nan"),
        }]
    )
    return wk, summ


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # p10_p70
    wf.POST_Q_OBJ = 0.10
    wf.MIN_POST_P_MEAN_POS = 0.70
    wf.ROBUST_CUTOFF_ENABLED = True
    wf.ROBUST_CUTOFF_DELTA = 0.02
    wf.HYSTERESIS_ENABLED = True
    wf.HYST_P_SWITCH = 0.90

    df = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])
    df["house_cap"] = df["house_cap"].apply(wf.safe_cap)
    df["week"] = wf.week_key(df["BIA_ApostaUTC"]).astype(str)
    df["date"] = wf.date_key(df["BIA_ApostaUTC"])
    df["roi_raw"] = pd.to_numeric(df["roi_calc"], errors="coerce").astype(float)
    df["roi_cap2"] = np.minimum(df["roi_raw"].to_numpy(dtype=float), 2.0)
    df["roi_cap1"] = np.minimum(df["roi_raw"].to_numpy(dtype=float), 1.0)
    df = _ensure_calibrated_cols(df)

    sc = pd.read_csv(SCORES, parse_dates=["BIA_ApostaUTC"])
    sc = sc[["ID Aposta", "score_prod_payload_logit_wf12", "score_prod_payload_region_logit_wf12"]].copy()
    df = df.merge(sc, how="left", on="ID Aposta")

    wk_cur, s_cur = _run_wf(df, "score_current")
    wk_p, s_p = _run_wf(df, "score_prod_payload_logit_wf12")
    wk_pr, s_pr = _run_wf(df, "score_prod_payload_region_logit_wf12")

    out_sum = OUT_DIR / "oos_walkforward_score_compare_prod_region_wf12_p10_p70_summary.csv"
    pd.concat([s_cur, s_p, s_pr], axis=0, ignore_index=True).to_csv(out_sum, index=False)

    wk_cur.to_csv(OUT_DIR / "oos_walkforward_score_current_p10_p70_weekly.csv", index=False)
    wk_p.to_csv(OUT_DIR / "oos_walkforward_score_prod_payload_logit_wf12_p10_p70_weekly.csv", index=False)
    wk_pr.to_csv(OUT_DIR / "oos_walkforward_score_prod_payload_region_logit_wf12_p10_p70_weekly.csv", index=False)

    print(str(out_sum))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

