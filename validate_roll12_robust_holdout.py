#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validação meta-OOS (holdout temporal) para hiperparâmetros do modo:
  global_bayes + rolling window + robust cutoff + histerese

Objetivo:
- Reduzir risco de overfit ao escolher hiperparâmetros.
- Escolher config com base APENAS no período de tuning e avaliar no holdout final.

Saídas:
- /workspace/analysis_proba_raw/pro_portfolio_all/holdout_roll12_robust_results.csv
- /workspace/analysis_proba_raw/pro_portfolio_all/holdout_roll12_robust_results.md
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
OUT_CSV = OUT_DIR / "holdout_roll12_robust_results.csv"
OUT_MD = OUT_DIR / "holdout_roll12_robust_results.md"


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


def _run_cfg_weekly(df: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
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

            alpha, _, _ = wf.find_alpha_global(df_train, rules)
            bets = wf.apply_rules_on_df(df_test, rules, alpha=float(alpha))
            stake_sum = float(bets["stake_eff"].sum()) if len(bets) else 0.0
            pnl_sum = float(bets["profit_cap2"].sum()) if len(bets) else 0.0
            weekly_rows.append({"week": w_test, "stake_usd": stake_sum, "profit_cap2_usd": pnl_sum, "alpha": float(alpha)})
            prev_rules = rules.copy()

        out = pd.DataFrame(weekly_rows).sort_values("week").reset_index(drop=True)
        out["cfg"] = cfg.name
        return out
    finally:
        for k, v in prev.items():
            setattr(wf, k, v)


def _agg(df: pd.DataFrame) -> Dict[str, float | int]:
    stake = float(df["stake_usd"].sum())
    pnl = float(df["profit_cap2_usd"].sum())
    roi = float(pnl / stake) if stake > 0 else float("nan")
    w = df.loc[df["stake_usd"] > 0, "profit_cap2_usd"].to_numpy(float)
    return {
        "weeks": int(len(df)),
        "weeks_with_stake": int((df["stake_usd"] > 0).sum()),
        "stake_total": stake,
        "pnl_total": pnl,
        "roi_total": roi,
        "mean_week_nonzero": float(np.mean(w)) if w.size else float("nan"),
        "pneg_week_nonzero": float((w < 0).mean()) if w.size else float("nan"),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_df()
    weeks_all = sorted(df["week"].unique().tolist())

    # Holdout = últimas 4 semanas do arquivo (temporal)
    holdout_n = 4
    holdout_weeks = weeks_all[-holdout_n:]

    # Candidatos (vindos do sweep)
    cfgs = [
        Cfg("base_p05_p80", 12, 0.05, 0.80, 0.02, 0.90),
        Cfg("p05_p70", 12, 0.05, 0.70, 0.02, 0.90),
        Cfg("p10_p70", 12, 0.10, 0.70, 0.02, 0.90),
    ]

    rows = []
    weekly_all = []
    for cfg in cfgs:
        w = _run_cfg_weekly(df, cfg)
        weekly_all.append(w)
        tune = w[~w["week"].isin(holdout_weeks)].copy()
        hold = w[w["week"].isin(holdout_weeks)].copy()
        a_t = _agg(tune)
        a_h = _agg(hold)
        rows.append(
            {
                "cfg": cfg.name,
                "holdout_weeks": ",".join(holdout_weeks),
                "tune_pnl": a_t["pnl_total"],
                "tune_roi": a_t["roi_total"],
                "tune_stake": a_t["stake_total"],
                "tune_weeks_with_stake": a_t["weeks_with_stake"],
                "tune_pneg": a_t["pneg_week_nonzero"],
                "hold_pnl": a_h["pnl_total"],
                "hold_roi": a_h["roi_total"],
                "hold_stake": a_h["stake_total"],
                "hold_weeks_with_stake": a_h["weeks_with_stake"],
                "hold_pneg": a_h["pneg_week_nonzero"],
            }
        )

    out = pd.DataFrame(rows).sort_values(["hold_pnl"], ascending=False)
    out.to_csv(OUT_CSV, index=False)

    # markdown curto
    lines = []
    lines.append("## Holdout temporal — roll12 + robust cutoff + histerese\n\n")
    lines.append(f"- Holdout (últimas {holdout_n} semanas): `{', '.join(holdout_weeks)}`\n")
    lines.append("- Critério: comparar configs com tuning vs holdout (cap2, ROI via roi_calc).\n\n")
    lines.append("Arquivo: `analysis_proba_raw/pro_portfolio_all/holdout_roll12_robust_results.csv`\n")
    OUT_MD.write_text("".join(lines), encoding="utf-8")

    print(str(OUT_CSV))
    print(str(OUT_MD))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

