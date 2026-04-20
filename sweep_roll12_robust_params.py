#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep (grid pequeno) em torno da melhor família:
  global_bayes + rolling window + robust cutoff + histerese

Objetivo: comparar configs de forma honesta (walk-forward) sem gerar PDFs,
e sem bootstrap pesado — apenas métricas agregadas OOS.

Saída:
- /workspace/analysis_proba_raw/pro_portfolio_all/sweep_roll12_robust_params.csv
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import evaluate_oos_walkforward_strategy as wf


SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")
OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")
OUT_CSV = OUT_DIR / "sweep_roll12_robust_params.csv"


@dataclass(frozen=True)
class Cfg:
    name: str
    train_window_weeks: int
    post_q_obj: float
    min_post_p_mean_pos: float
    robust_delta: float
    hyst_p_switch: float
    bayes_n: int = 3000


def _load_df() -> pd.DataFrame:
    df = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])
    df["house_cap"] = df["house_cap"].apply(wf.safe_cap)
    df["week"] = wf.week_key(df["BIA_ApostaUTC"])
    df["date"] = wf.date_key(df["BIA_ApostaUTC"])
    if "roi_calc" not in df.columns:
        raise KeyError("Coluna roi_calc ausente no scored_dedup_proba_raw_all.csv")
    df["roi_raw"] = pd.to_numeric(df["roi_calc"], errors="coerce").astype(float)
    df["roi_cap2"] = np.minimum(df["roi_raw"].to_numpy(dtype=float), 2.0)
    df["roi_cap1"] = np.minimum(df["roi_raw"].to_numpy(dtype=float), 1.0)

    # scores calibrados (qui, sex/sab/dom) igual ao WF
    df["proba_cal_segqui"] = np.nan
    if "proba_raw_segqui" in df.columns and wf.CALIB_SEGQUI.exists():
        calib = json.loads(wf.CALIB_SEGQUI.read_text(encoding="utf-8"))
        x = np.asarray(calib.get("isotonic", {}).get("x", []), dtype=float)
        y = np.asarray(calib.get("isotonic", {}).get("y", []), dtype=float)
        p_raw = pd.to_numeric(df["proba_raw_segqui"], errors="coerce").to_numpy(dtype=float)
        df["proba_cal_segqui"] = wf._apply_isotonic_vec(p_raw, x=x, y=y, floor=wf.CALIB_FLOOR)

    df["proba_cal_sexdom"] = np.nan
    if "proba_raw_sexdom" in df.columns and wf.CALIB_SEXDOM.exists():
        calib = json.loads(wf.CALIB_SEXDOM.read_text(encoding="utf-8"))
        x = np.asarray(calib.get("isotonic", {}).get("x", []), dtype=float)
        y = np.asarray(calib.get("isotonic", {}).get("y", []), dtype=float)
        p_raw = pd.to_numeric(df["proba_raw_sexdom"], errors="coerce").to_numpy(dtype=float)
        df["proba_cal_sexdom"] = wf._apply_isotonic_vec(p_raw, x=x, y=y, floor=wf.CALIB_FLOOR)

    return df


def _run_cfg(df: pd.DataFrame, cfg: Cfg) -> Dict[str, float | int | str]:
    prev = {
        "BAYES_N": wf.BAYES_N,
        "POST_Q_OBJ": wf.POST_Q_OBJ,
        "MIN_POST_P_MEAN_POS": wf.MIN_POST_P_MEAN_POS,
        "ROBUST_CUTOFF_ENABLED": wf.ROBUST_CUTOFF_ENABLED,
        "ROBUST_CUTOFF_DELTA": wf.ROBUST_CUTOFF_DELTA,
        "HYSTERESIS_ENABLED": wf.HYSTERESIS_ENABLED,
        "HYST_P_SWITCH": wf.HYST_P_SWITCH,
    }
    try:
        wf.BAYES_N = int(cfg.bayes_n)
        wf.POST_Q_OBJ = float(cfg.post_q_obj)
        wf.MIN_POST_P_MEAN_POS = float(cfg.min_post_p_mean_pos)
        wf.ROBUST_CUTOFF_ENABLED = True
        wf.ROBUST_CUTOFF_DELTA = float(cfg.robust_delta)
        wf.HYSTERESIS_ENABLED = True
        wf.HYST_P_SWITCH = float(cfg.hyst_p_switch)

        weeks = sorted(df["week"].unique().tolist())
        weekly_rows = []
        prev_rules: Dict[str, wf.Rule] = {}

        for i in range(wf.MIN_GLOBAL_TRAIN_WEEKS, len(weeks)):
            w_test = weeks[i]
            train_weeks = weeks[max(0, i - int(cfg.train_window_weeks)) : i]
            df_train = df[df["week"].isin(train_weeks)].copy()
            df_test = df[df["week"] == w_test].copy()

            # otimiza regras no treino (para cada segmento)
            rules: Dict[str, wf.Rule] = {}
            for bet_type in ("FT", "FH"):
                for dow in wf.WEEKDAY_PT:
                    sc = wf.segment_score_col(dow)
                    x = df_train[(df_train["dow_pt"] == dow) & (df_train["bet_type"] == bet_type)].copy()
                    if x.empty:
                        rule = wf.Rule(bet_type=bet_type, dow=dow, score_col=sc, cutoff=1.0, stake_frac=0.0, status="no_data")
                    else:
                        prev_rule = prev_rules.get(f"{bet_type}|{dow}")
                        rule = wf.optimize_segment_train(x, sc, bayes_select=True, prev_rule=prev_rule, roi_bias_adj=0.0)
                    rules[f"{bet_type}|{dow}"] = rule

            # risco global: ajustar alpha
            alpha, _, _ = wf.find_alpha_global(df_train, rules)

            # aplica no teste
            bets = wf.apply_rules_on_df(df_test, rules, alpha=float(alpha))
            stake_sum = float(bets["stake_eff"].sum()) if len(bets) else 0.0
            pnl_sum = float(bets["profit_cap2"].sum()) if len(bets) else 0.0
            n_bets = int(len(bets))
            weekly_rows.append({"week": w_test, "stake_usd": stake_sum, "profit_cap2_usd": pnl_sum, "n_bets": n_bets, "alpha_global": float(alpha)})

            prev_rules = rules.copy()

        weekly = pd.DataFrame(weekly_rows)
        stake_tot = float(weekly["stake_usd"].sum())
        pnl_tot = float(weekly["profit_cap2_usd"].sum())
        roi_tot = float(pnl_tot / stake_tot) if stake_tot > 0 else float("nan")
        w_nonzero = weekly.loc[weekly["stake_usd"] > 0, "profit_cap2_usd"].to_numpy(dtype=float)
        return {
            "name": cfg.name,
            "train_window_weeks": cfg.train_window_weeks,
            "post_q_obj": cfg.post_q_obj,
            "min_post_p_mean_pos": cfg.min_post_p_mean_pos,
            "robust_delta": cfg.robust_delta,
            "hyst_p_switch": cfg.hyst_p_switch,
            "bayes_n": cfg.bayes_n,
            "weeks_total": int(len(weekly)),
            "weeks_with_stake": int((weekly["stake_usd"] > 0).sum()),
            "profit_cap2_total": pnl_tot,
            "stake_total": stake_tot,
            "roi_total_cap2": roi_tot,
            "mean_weekly_cap2_nonzero": float(np.mean(w_nonzero)) if w_nonzero.size else float("nan"),
            "pneg_weeks_nonzero": float((w_nonzero < 0).mean()) if w_nonzero.size else float("nan"),
        }
    finally:
        for k, v in prev.items():
            setattr(wf, k, v)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_df()

    # Grid pequeno ao redor do roll12_robust atual
    grid = [
        Cfg("base", 12, 0.05, 0.80, 0.02, 0.90),
        Cfg("p10", 12, 0.10, 0.80, 0.02, 0.90),
        Cfg("p20", 12, 0.20, 0.80, 0.02, 0.90),
        Cfg("p05_p70", 12, 0.05, 0.70, 0.02, 0.90),
        Cfg("p10_p70", 12, 0.10, 0.70, 0.02, 0.90),
        Cfg("delta04", 12, 0.05, 0.80, 0.04, 0.90),
        Cfg("hyst85", 12, 0.05, 0.80, 0.02, 0.85),
        Cfg("win10", 10, 0.05, 0.80, 0.02, 0.90),
        Cfg("win16", 16, 0.05, 0.80, 0.02, 0.90),
    ]

    rows = []
    for cfg in grid:
        rows.append(_run_cfg(df, cfg))

    out = pd.DataFrame(rows).sort_values(["profit_cap2_total"], ascending=False)
    out.to_csv(OUT_CSV, index=False)
    print(str(OUT_CSV))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

