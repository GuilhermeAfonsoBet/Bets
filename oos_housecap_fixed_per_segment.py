#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Escolhe um cap_max FIXO por segmento (DoW x FT/FH) e avalia OOS.

Plano:
1) Treino de caps (fixos): usa um período inicial (primeiras N semanas do WF) para escolher cap_max por segmento,
   rodando uma busca leve por candidatos (inf, q60, q80, q90) e escolhendo o cap que maximiza um objetivo Bayesiano:
     - aplica optimize_segment_train no dataset filtrado por cap<=cap_max
     - avalia o p10 do lucro semanal médio (cap2) no treino, usando Bayesian bootstrap.

2) Avaliação OOS:
   para semanas posteriores, roda o otimizador semanal p10_p70 (cutoff/stake_frac/α) mas com o dataset
   de cada segmento filtrado pelo cap_max FIXO escolhido no passo 1.

Saídas:
- analysis_proba_raw/pro_portfolio_all/oos_walkforward_housecap_capfixed_weekly.csv
- analysis_proba_raw/pro_portfolio_all/oos_walkforward_housecap_capfixed_selected_rules.csv
- analysis_proba_raw/pro_portfolio_all/oos_walkforward_housecap_capfixed_summary.csv
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

TRAIN_WINDOW_WEEKS = 12
CAP_QS = [0.60, 0.80, 0.90]
CAP_MIN_BETS = 30
CAP_MIN_WEEKS = 4

# Cap selection Bayesian bootstrap
CAP_BAYES_N = 2000

# Cap training period: first N weeks of the WF timeline
CAP_TRAIN_WEEKS = 12


def _cap_candidates(cap: np.ndarray) -> List[float]:
    cap = np.asarray(cap, float)
    cap = cap[np.isfinite(cap) & (cap > 0)]
    out = [float("inf")]
    if cap.size:
        qs = np.unique(np.quantile(cap, CAP_QS))
        qs = qs[np.isfinite(qs) & (qs > 0)]
        out += [float(v) for v in qs.tolist()]
    return sorted(set(out))


def _posterior_obj_p10(w_pnl: np.ndarray, n_draws: int, seed: int) -> Tuple[float, float]:
    """
    Return (obj=p10(mean_week), p_mean_pos) where mean_week is the weekly pnl mean under BB.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(w_pnl, float)
    if x.size == 0:
        return -np.inf, 0.0
    bb = rng.dirichlet(np.ones(x.size), size=int(n_draws))
    post_means = bb @ x.astype(float)
    ppos = float(np.mean(post_means > 0))
    obj = float(np.quantile(post_means, 0.10))
    return obj, ppos


def _train_weekly_series(df_bets: pd.DataFrame, weeks_all: List[str]) -> np.ndarray:
    if df_bets.empty:
        return np.zeros(len(weeks_all), float)
    s = df_bets.groupby("week")["profit_cap2"].sum().reindex(weeks_all, fill_value=0.0).to_numpy(float)
    return s


def choose_fixed_caps(df: pd.DataFrame, cap_train_weeks: List[str]) -> Dict[str, float]:
    """
    Escolhe cap_max fixo por rule_key (FT|dow, FH|dow) usando apenas cap_train_weeks.
    """
    out: Dict[str, float] = {}
    for bt in ("FT", "FH"):
        for dow in wf.WEEKDAY_PT:
            rk = f"{bt}|{dow}"
            sc = wf.segment_score_col(dow)
            x = df[(df["week"].isin(cap_train_weeks)) & (df["bet_type"] == bt) & (df["dow_pt"] == dow)].copy()
            if x.empty:
                out[rk] = float("inf")
                continue
            cap_cands = _cap_candidates(x["house_cap"].to_numpy(float))
            weeks_all = sorted(x["week"].unique().tolist())
            if len(weeks_all) < CAP_MIN_WEEKS:
                out[rk] = float("inf")
                continue

            best_cap = float("inf")
            best_obj = -np.inf
            for cap_max in cap_cands:
                xx = x if not np.isfinite(cap_max) else x[np.isfinite(x["house_cap"]) & (x["house_cap"] <= float(cap_max))].copy()
                if xx.empty or int(len(xx)) < CAP_MIN_BETS or int(xx["week"].nunique()) < CAP_MIN_WEEKS:
                    continue
                # optimize cutoff/stake within this filtered set (p10_p70 config)
                rule = wf.optimize_segment_train(xx, sc, bayes_select=True, prev_rule=None, roi_bias_adj=0.0)
                if rule.status != "ok" or rule.stake_frac <= 0:
                    continue
                # evaluate objective p10 of mean_week on train
                bets = wf.apply_rule_on_week(xx, wf.Rule(rule.bet_type, rule.dow, rule.score_col, rule.cutoff, rule.stake_frac, rule.status, cap_max=float(cap_max)))
                w_pnl = _train_weekly_series(bets, weeks_all)
                obj, ppos = _posterior_obj_p10(w_pnl, n_draws=CAP_BAYES_N, seed=7 + hash(rk) % 10000)
                if ppos < 0.55:
                    continue
                if obj > best_obj:
                    best_obj = float(obj)
                    best_cap = float(cap_max)
            out[rk] = float(best_cap)
    return out


def run_wf_with_fixed_caps(df: pd.DataFrame, fixed_caps: Dict[str, float], weeks: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Roda WF p10_p70 em todas as semanas, mas filtrando cada segmento por cap_max fixo.
    """
    # align p10_p70
    wf.POST_Q_OBJ = 0.10
    wf.MIN_POST_P_MEAN_POS = 0.70
    wf.ROBUST_CUTOFF_ENABLED = True
    wf.HYSTERESIS_ENABLED = True
    wf.ROBUST_CUTOFF_DELTA = 0.02
    wf.HYST_P_SWITCH = 0.90

    rules_rows = []
    weekly_rows = []
    prev_rules: Dict[str, wf.Rule] = {}

    for i in range(wf.MIN_GLOBAL_TRAIN_WEEKS, len(weeks)):
        w_test = weeks[i]
        train_weeks = weeks[max(0, i - TRAIN_WINDOW_WEEKS) : i]
        df_train = df[df["week"].isin(train_weeks)].copy()
        df_test = df[df["week"] == w_test].copy()

        rules: Dict[str, wf.Rule] = {}
        for bt in ("FT", "FH"):
            for dow in wf.WEEKDAY_PT:
                rk = f"{bt}|{dow}"
                sc = wf.segment_score_col(dow)
                cap_max = float(fixed_caps.get(rk, float("inf")))
                x = df_train[(df_train["bet_type"] == bt) & (df_train["dow_pt"] == dow)].copy()
                if np.isfinite(cap_max):
                    x = x[np.isfinite(x["house_cap"]) & (x["house_cap"] <= cap_max)].copy()
                if x.empty:
                    rule = wf.Rule(bt, dow, sc, 1.0, 0.0, "no_data", cap_max=cap_max)
                else:
                    prev = prev_rules.get(rk)
                    rule0 = wf.optimize_segment_train(x, sc, bayes_select=True, prev_rule=prev, roi_bias_adj=0.0)
                    rule = wf.Rule(rule0.bet_type, rule0.dow, rule0.score_col, rule0.cutoff, rule0.stake_frac, rule0.status, cap_max=cap_max)
                rules[rk] = rule

        alpha, _, _ = wf.find_alpha_global(df_train, rules)
        bets = wf.apply_rules_on_df(df_test, rules, alpha=float(alpha))
        stake = float(bets["stake_eff"].sum()) if not bets.empty else 0.0
        pnl = float(bets["profit_cap2"].sum()) if not bets.empty else 0.0
        weekly_rows.append({"week": w_test, "alpha_global": float(alpha), "stake_usd": stake, "profit_cap2_usd": pnl, "n_bets": int(len(bets))})

        for key, r in rules.items():
            rules_rows.append({"test_week": w_test, "bet_type": r.bet_type, "dow_pt": r.dow, "rule_key": key, "cutoff": float(r.cutoff), "stake_frac": float(r.stake_frac), "cap_max": float(r.cap_max), "status": r.status, "alpha_global": float(alpha)})

        prev_rules = rules.copy()

    return pd.DataFrame(weekly_rows), pd.DataFrame(rules_rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])
    df["week"] = wf.week_key(df["BIA_ApostaUTC"]).astype(str)
    df["date"] = wf.date_key(df["BIA_ApostaUTC"])
    df["house_cap"] = df["house_cap"].apply(wf.safe_cap)
    df["roi_raw"] = pd.to_numeric(df["roi_calc"], errors="coerce").to_numpy(float)
    df["roi_cap2"] = np.minimum(df["roi_raw"].to_numpy(float), 2.0)
    df["roi_cap1"] = np.minimum(df["roi_raw"].to_numpy(float), 1.0)

    # ensure calibrated cols exist (use wf main logic already in file when imported)
    if "proba_cal_segqui" not in df.columns or "proba_cal_sexdom" not in df.columns:
        # quick re-run: mimic evaluate script’s calibration
        import json
        df["proba_cal_segqui"] = np.nan
        if "proba_raw_segqui" in df.columns and wf.CALIB_SEGQUI.exists():
            obj = json.loads(wf.CALIB_SEGQUI.read_text(encoding="utf-8"))
            x = np.asarray(obj.get("isotonic", {}).get("x", []), float)
            y = np.asarray(obj.get("isotonic", {}).get("y", []), float)
            p = pd.to_numeric(df["proba_raw_segqui"], errors="coerce").to_numpy(float)
            df["proba_cal_segqui"] = wf._apply_isotonic_vec(p, x=x, y=y, floor=wf.CALIB_FLOOR)
        df["proba_cal_sexdom"] = np.nan
        if "proba_raw_sexdom" in df.columns and wf.CALIB_SEXDOM.exists():
            obj = json.loads(wf.CALIB_SEXDOM.read_text(encoding="utf-8"))
            x = np.asarray(obj.get("isotonic", {}).get("x", []), float)
            y = np.asarray(obj.get("isotonic", {}).get("y", []), float)
            p = pd.to_numeric(df["proba_raw_sexdom"], errors="coerce").to_numpy(float)
            df["proba_cal_sexdom"] = wf._apply_isotonic_vec(p, x=x, y=y, floor=wf.CALIB_FLOOR)

    weeks = sorted(df["week"].unique().tolist())
    cap_train_weeks = weeks[: min(CAP_TRAIN_WEEKS, len(weeks))]
    fixed_caps = choose_fixed_caps(df, cap_train_weeks=cap_train_weeks)

    weekly_df, rules_df = run_wf_with_fixed_caps(df, fixed_caps=fixed_caps, weeks=weeks)

    # summary vs baseline
    base_weekly = pd.read_csv(OUT_DIR / f"oos_walkforward_{MODE_BASE}_weekly.csv")
    stake_b = float(base_weekly["stake_usd"].sum())
    pnl_b = float(base_weekly["profit_cap2_usd"].sum())
    stake_f = float(weekly_df["stake_usd"].sum())
    pnl_f = float(weekly_df["profit_cap2_usd"].sum())
    summ = pd.DataFrame(
        [
            {"name": "baseline", "profit_cap2_total": pnl_b, "stake_total": stake_b, "roi_total_cap2": pnl_b / stake_b if stake_b > 0 else np.nan},
            {"name": "capfixed", "profit_cap2_total": pnl_f, "stake_total": stake_f, "roi_total_cap2": pnl_f / stake_f if stake_f > 0 else np.nan},
        ]
    )

    out_week = OUT_DIR / "oos_walkforward_housecap_capfixed_weekly.csv"
    out_rules = OUT_DIR / "oos_walkforward_housecap_capfixed_selected_rules.csv"
    out_sum = OUT_DIR / "oos_walkforward_housecap_capfixed_summary.csv"
    weekly_df.to_csv(out_week, index=False)
    rules_df.to_csv(out_rules, index=False)
    summ.to_csv(out_sum, index=False)
    print(str(out_sum))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

