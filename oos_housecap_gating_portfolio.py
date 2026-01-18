#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimento OOS: otimizar um cap_max (house_cap máximo permitido) por segmento/semana,
mantendo (cutoff, stake_frac, score_col, alpha) do portfólio base (p10_p70).

Por que assim?
- É 100% OOS (cap_max escolhido só com semanas anteriores)
- Evita re-otimizar toda a grade de cutoff/stake (muito caro)
- Produz uma regra operacional simples: além de score>=cutoff, exigir house_cap<=cap_max

Saídas:
- analysis_proba_raw/pro_portfolio_all/oos_walkforward_housecap_gating_weekly.csv
- analysis_proba_raw/pro_portfolio_all/oos_walkforward_housecap_gating_rules.csv
- analysis_proba_raw/pro_portfolio_all/oos_walkforward_housecap_gating_summary.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import evaluate_oos_walkforward_strategy as wf


OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")
MODE = "global_bayes_roll12_robust_p10_p70"
SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")
WF_RULES = OUT_DIR / f"oos_walkforward_{MODE}_selected_rules.csv"
WF_WEEKLY = OUT_DIR / f"oos_walkforward_{MODE}_weekly.csv"

TRAIN_WINDOW_WEEKS = 12
CAP_QS = [0.60, 0.80, 0.90]  # candidatos por quantis no treino condicional ao cutoff
MIN_BETS_FOR_CAP_OPT = 60
MIN_NONZERO_WEEKS = 6
MIN_SELECTED_BETS = 6

# seleção Bayes (leve)
BAYES_N = 2000
MIN_POST_P_MEAN_POS = 0.70
POST_Q_OBJ = 0.10


def _ensure_calibrated_cols(df: pd.DataFrame) -> pd.DataFrame:
    import json

    df = df.copy()
    floor = 0.005
    if "proba_cal_segqui" not in df.columns:
        df["proba_cal_segqui"] = np.nan
    if "proba_cal_sexdom" not in df.columns:
        df["proba_cal_sexdom"] = np.nan
    if "proba_raw_segqui" in df.columns and wf.CALIB_SEGQUI.exists():
        obj = json.loads(wf.CALIB_SEGQUI.read_text(encoding="utf-8"))
        x = np.asarray(obj["isotonic"]["x"], float)
        y = np.asarray(obj["isotonic"]["y"], float)
        p = pd.to_numeric(df["proba_raw_segqui"], errors="coerce").to_numpy(float)
        out = np.interp(p, x, y, left=float(y[0]), right=float(y[-1]))
        out = np.maximum(out, floor)
        df["proba_cal_segqui"] = np.clip(out, 0.0, 1.0)
    if "proba_raw_sexdom" in df.columns and wf.CALIB_SEXDOM.exists():
        obj = json.loads(wf.CALIB_SEXDOM.read_text(encoding="utf-8"))
        x = np.asarray(obj["isotonic"]["x"], float)
        y = np.asarray(obj["isotonic"]["y"], float)
        p = pd.to_numeric(df["proba_raw_sexdom"], errors="coerce").to_numpy(float)
        out = np.interp(p, x, y, left=float(y[0]), right=float(y[-1]))
        out = np.maximum(out, floor)
        df["proba_cal_sexdom"] = np.clip(out, 0.0, 1.0)
    return df


def _bb_weights(n_weeks: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(n_weeks), size=BAYES_N)


def _weekly_series(profit: np.ndarray, weeks: np.ndarray, weeks_all: List[str]) -> np.ndarray:
    return (
        pd.Series(profit, index=weeks)
        .groupby(level=0)
        .sum()
        .reindex(weeks_all, fill_value=0.0)
        .to_numpy(float)
    )


def choose_cap_max(
    df_train_seg: pd.DataFrame,
    score_col: str,
    cutoff: float,
    stake0: float,
    train_weeks: List[str],
    seed: int,
) -> Tuple[float, str]:
    """
    Retorna (cap_max escolhido, status).
    """
    if df_train_seg.empty or not train_weeks:
        return float("inf"), "no_data"
    score = pd.to_numeric(df_train_seg[score_col], errors="coerce").to_numpy(float)
    roi2 = df_train_seg["roi_cap2"].to_numpy(float)
    cap = df_train_seg["house_cap"].to_numpy(float)
    wk = df_train_seg["week"].to_numpy()
    m0 = np.isfinite(score) & (score >= float(cutoff)) & np.isfinite(roi2) & np.isfinite(cap) & (cap > 0)
    if int(np.sum(m0)) < MIN_BETS_FOR_CAP_OPT:
        return float("inf"), "too_few_bets_for_capopt"

    cap0 = cap[m0]
    cand = [float("inf")]
    qs = np.unique(np.quantile(cap0, CAP_QS))
    qs = qs[np.isfinite(qs) & (qs > 0)]
    cand += [float(v) for v in qs.tolist()]
    cand = sorted(set(cand))
    if len(cand) <= 1:
        return float("inf"), "no_cap_candidates"

    bb = _bb_weights(len(train_weeks), seed=seed)
    best_obj = -np.inf
    best_cap = float("inf")
    best_status = "no_candidate"

    for cmax in cand:
        m = m0 & (cap <= float(cmax) if np.isfinite(cmax) else True)
        if int(np.sum(m)) < MIN_SELECTED_BETS:
            continue
        # semanas com trade
        nonzero = int(pd.Series(np.ones(int(np.sum(m))), index=wk[m]).groupby(level=0).sum().shape[0])
        if nonzero < MIN_NONZERO_WEEKS:
            continue
        stake_eff = np.minimum(stake0, cap[m])
        profit = stake_eff * roi2[m]
        w2 = _weekly_series(profit, wk[m], train_weeks)
        post_means = bb @ w2.astype(float)
        p_pos = float(np.mean(post_means > 0))
        if p_pos < MIN_POST_P_MEAN_POS:
            continue
        q_obj = float(np.quantile(post_means, POST_Q_OBJ))
        if q_obj > best_obj:
            best_obj = q_obj
            best_cap = float(cmax)
            best_status = "ok"

    return best_cap, best_status


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rules = pd.read_csv(WF_RULES)
    weekly_base = pd.read_csv(WF_WEEKLY)
    weeks = weekly_base["week"].astype(str).tolist()

    df = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])
    df["week"] = wf.week_key(df["BIA_ApostaUTC"]).astype(str)
    df["date"] = wf.date_key(df["BIA_ApostaUTC"])
    df["roi_raw"] = pd.to_numeric(df["roi_calc"], errors="coerce").to_numpy(float)
    df["roi_cap2"] = np.minimum(df["roi_raw"].to_numpy(float), 2.0)
    df["house_cap"] = pd.to_numeric(df["house_cap"], errors="coerce").to_numpy(float)
    df = _ensure_calibrated_cols(df)

    out_week = []
    out_rules = []

    for i, wk_test in enumerate(weeks):
        rw = rules[rules["test_week"].astype(str) == wk_test].copy()
        if rw.empty:
            continue
        train_weeks = weeks[max(0, i - TRAIN_WINDOW_WEEKS) : i]
        df_train = df[df["week"].isin(train_weeks)].copy()
        df_test = df[df["week"] == wk_test].copy()

        # alpha efetivo da semana (já inclui overlay, se houver)
        alpha = float(rw["alpha_effective"].iloc[0]) if "alpha_effective" in rw.columns and np.isfinite(float(rw["alpha_effective"].iloc[0])) else float(rw["alpha_global"].iloc[0])

        bets_rows = []
        for _, r in rw.iterrows():
            if str(r.get("status")) != "ok":
                continue
            frac = float(r.get("stake_frac", 0.0))
            if frac <= 0:
                continue
            bt = str(r["bet_type"])
            dow = str(r["dow_pt"])
            score_col = str(r["score_col"])
            cutoff = float(r["cutoff"])
            stake0 = wf.BANKROLL * frac * float(alpha)

            seg_train = df_train[(df_train["bet_type"] == bt) & (df_train["dow_pt"] == dow)].copy()
            cap_max, cap_status = choose_cap_max(seg_train, score_col=score_col, cutoff=cutoff, stake0=stake0, train_weeks=train_weeks, seed=7 + i * 17)

            out_rules.append(
                {
                    "test_week": wk_test,
                    "bet_type": bt,
                    "dow_pt": dow,
                    "rule_key": str(r.get("rule_key", f"{bt}|{dow}")),
                    "score_col": score_col,
                    "cutoff": cutoff,
                    "stake_frac": frac,
                    "alpha_effective": float(alpha),
                    "cap_max": float(cap_max),
                    "cap_status": cap_status,
                }
            )

            seg_test = df_test[(df_test["bet_type"] == bt) & (df_test["dow_pt"] == dow)].copy()
            if seg_test.empty or score_col not in seg_test.columns:
                continue
            score = pd.to_numeric(seg_test[score_col], errors="coerce").to_numpy(float)
            roi2 = seg_test["roi_cap2"].to_numpy(float)
            cap = seg_test["house_cap"].to_numpy(float)
            m = np.isfinite(score) & (score >= cutoff) & np.isfinite(roi2) & np.isfinite(cap) & (cap > 0)
            if np.isfinite(cap_max):
                m = m & (cap <= float(cap_max))
            if not np.any(m):
                continue
            stake_eff = np.minimum(stake0, cap[m])
            profit = stake_eff * roi2[m]
            bets_rows.append(pd.DataFrame({"stake_eff": stake_eff, "profit_cap2": profit}))

        if bets_rows:
            b = pd.concat(bets_rows, ignore_index=True)
            stake_sum = float(b["stake_eff"].sum())
            pnl_sum = float(b["profit_cap2"].sum())
            n_bets = int(len(b))
        else:
            stake_sum = 0.0
            pnl_sum = 0.0
            n_bets = 0

        out_week.append(
            {
                "week": wk_test,
                "alpha_effective": float(alpha),
                "n_bets": n_bets,
                "stake_usd": stake_sum,
                "profit_cap2_usd": pnl_sum,
                "roi_on_stake_cap2": float(pnl_sum / stake_sum) if stake_sum > 0 else float("nan"),
            }
        )

    out_week_df = pd.DataFrame(out_week)
    out_rules_df = pd.DataFrame(out_rules)

    out_week_path = OUT_DIR / "oos_walkforward_housecap_gating_weekly.csv"
    out_rules_path = OUT_DIR / "oos_walkforward_housecap_gating_rules.csv"
    out_week_df.to_csv(out_week_path, index=False)
    out_rules_df.to_csv(out_rules_path, index=False)

    # summary vs baseline
    base = weekly_base.copy()
    base["week"] = base["week"].astype(str)
    stake_b = float(base["stake_usd"].sum())
    pnl_b = float(base["profit_cap2_usd"].sum())
    stake_g = float(out_week_df["stake_usd"].sum())
    pnl_g = float(out_week_df["profit_cap2_usd"].sum())
    summ = pd.DataFrame(
        [
            {"name": "baseline", "profit_cap2_total": pnl_b, "stake_total": stake_b, "roi_total_cap2": pnl_b / stake_b if stake_b > 0 else np.nan, "weeks": int(len(base)), "weeks_with_stake": int((base["stake_usd"] > 0).sum())},
            {"name": "housecap_gating", "profit_cap2_total": pnl_g, "stake_total": stake_g, "roi_total_cap2": pnl_g / stake_g if stake_g > 0 else np.nan, "weeks": int(len(out_week_df)), "weeks_with_stake": int((out_week_df["stake_usd"] > 0).sum())},
        ]
    )
    summ_path = OUT_DIR / "oos_walkforward_housecap_gating_summary.csv"
    summ.to_csv(summ_path, index=False)
    print(str(summ_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

