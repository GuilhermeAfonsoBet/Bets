#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibração de forecast para o cenário "Operação no máximo":
  stake_eff_max = house_cap (banca não limita stake).

Para cada fold semanal do walk-forward:
- usa as regras θ_t do global_bayes (cutoff/stake_frac/alpha_global)
- aplica em semanas de treino e gera série semanal de PnL_max (stake=house_cap)
- gera distribuição preditiva via Bayesian bootstrap (Dirichlet) sobre semanas
- compara com PnL_max teórico realizado na semana teste

Saídas:
- analysis_proba_raw/pro_portfolio_all/forecast_calibration_global_bayes_max.csv
- analysis_proba_raw/pro_portfolio_all/forecast_calibration_global_bayes_max.md
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import json


OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")
SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")
RULES = OUT_DIR / "oos_walkforward_global_bayes_selected_rules.csv"
WEEKLY = OUT_DIR / "oos_walkforward_global_bayes_weekly.csv"
CALIB_SEXDOM = Path("/workspace/clv_calib_SexDom.json")
CALIB_FLOOR_SEXDOM = 0.005
CALIB_SEGQUI = Path("/workspace/clv_calib_SegQui.json")
CALIB_FLOOR_SEGQUI = 0.005

N_DRAWS = 10_000
SEED = 7


def _apply_isotonic_vec(p: np.ndarray, x: np.ndarray, y: np.ndarray, floor: float | None) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if x.size and y.size and x.size == y.size:
        out = np.interp(p, x, y, left=float(y[0]), right=float(y[-1]))
    else:
        out = p.copy()
    if floor is not None:
        out = np.maximum(out, float(floor))
    return np.clip(out, 0.0, 1.0)


def posterior_predictive_weekly_joint(
    w_pnl: np.ndarray, w_stake: np.ndarray, rng: np.random.Generator, n_draws: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(w_pnl, dtype=float)
    s = np.asarray(w_stake, dtype=float)
    n = p.size
    if n == 0:
        z = np.zeros(n_draws, dtype=float)
        return z, z, z
    pnl = np.empty(n_draws, dtype=float)
    stk = np.empty(n_draws, dtype=float)
    roi = np.empty(n_draws, dtype=float)
    for i in range(n_draws):
        w = rng.dirichlet(np.ones(n))
        idx = rng.choice(n, p=w)
        pnl[i] = p[idx]
        stk[i] = s[idx]
        roi[i] = (pnl[i] / stk[i]) if stk[i] > 0 else 0.0
    return pnl, stk, roi


def crps_from_sample(draws: np.ndarray, y: float) -> float:
    x = np.asarray(draws, dtype=float)
    if x.size == 0:
        return float("nan")
    a = float(np.mean(np.abs(x - y)))
    m = min(2000, x.size)
    xs = x[:m]
    xp = np.roll(xs, 1)
    b = float(np.mean(np.abs(xs - xp)))
    return a - 0.5 * b


def apply_rules_max(df: pd.DataFrame, rules_week: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in rules_week.iterrows():
        if str(r.get("status")) != "ok":
            continue
        bt = str(r["bet_type"])
        dow = str(r["dow_pt"])
        score_col = str(r["score_col"])
        cutoff = float(r["cutoff"])

        x = df[(df["bet_type"] == bt) & (df["dow_pt"] == dow)].copy()
        if x.empty:
            continue
        score = pd.to_numeric(x[score_col], errors="coerce").to_numpy(dtype=float)
        roi2 = x["roi_cap2"].to_numpy(dtype=float)
        cap = x["house_cap"].to_numpy(dtype=float)
        m = np.isfinite(score) & (score >= cutoff) & np.isfinite(roi2) & np.isfinite(cap)
        if not np.any(m):
            continue
        x = x.iloc[np.where(m)[0]].copy()
        x["stake_eff_max"] = x["house_cap"].to_numpy(dtype=float)
        x["profit_cap2_max"] = x["stake_eff_max"].to_numpy(dtype=float) * x["roi_cap2"].to_numpy(dtype=float)
        x["rule_key"] = str(r.get("rule_key", f"{bt}|{dow}"))
        rows.append(x[["week", "stake_eff_max", "profit_cap2_max", "rule_key"]])
    return pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame(columns=["week", "stake_eff_max", "profit_cap2_max", "rule_key"])


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rules = pd.read_csv(RULES)
    wf_week = pd.read_csv(WEEKLY)

    df = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])
    df["roi_raw"] = pd.to_numeric(df["ROI Real"], errors="coerce").astype(float)
    df["roi_cap2"] = np.minimum(df["roi_raw"].to_numpy(dtype=float), 2.0)
    df["house_cap"] = pd.to_numeric(df["house_cap"], errors="coerce").astype(float)
    df["week"] = pd.to_datetime(df["BIA_ApostaUTC"]).dt.to_period("W-SUN").astype(str)

    # garantir coluna calibrada para Sex/Sáb/Dom se necessária (regras podem referenciar proba_cal_sexdom)
    if "proba_cal_sexdom" not in df.columns:
        df["proba_cal_sexdom"] = np.nan
    if "proba_raw_sexdom" in df.columns and CALIB_SEXDOM.exists():
        try:
            calib = json.loads(CALIB_SEXDOM.read_text(encoding="utf-8"))
            x = np.asarray(calib.get("isotonic", {}).get("x", []), dtype=float)
            y = np.asarray(calib.get("isotonic", {}).get("y", []), dtype=float)
            p_raw = pd.to_numeric(df["proba_raw_sexdom"], errors="coerce").to_numpy(dtype=float)
            df["proba_cal_sexdom"] = _apply_isotonic_vec(p_raw, x=x, y=y, floor=CALIB_FLOOR_SEXDOM)
        except Exception:
            pass

    # garantir coluna calibrada para Qui (regras podem referenciar proba_cal_segqui)
    if "proba_cal_segqui" not in df.columns:
        df["proba_cal_segqui"] = np.nan
    if "proba_raw_segqui" in df.columns and CALIB_SEGQUI.exists():
        try:
            calib = json.loads(CALIB_SEGQUI.read_text(encoding="utf-8"))
            x = np.asarray(calib.get("isotonic", {}).get("x", []), dtype=float)
            y = np.asarray(calib.get("isotonic", {}).get("y", []), dtype=float)
            p_raw = pd.to_numeric(df["proba_raw_segqui"], errors="coerce").to_numpy(dtype=float)
            df["proba_cal_segqui"] = _apply_isotonic_vec(p_raw, x=x, y=y, floor=CALIB_FLOOR_SEGQUI)
        except Exception:
            pass

    weeks_sorted = sorted(df["week"].unique().tolist())
    week_to_i = {w: i for i, w in enumerate(weeks_sorted)}
    rng = np.random.default_rng(SEED)

    out_rows = []
    for w_test in wf_week["week"].astype(str).tolist():
        if w_test not in week_to_i:
            continue
        i = week_to_i[w_test]
        train_weeks = weeks_sorted[:i]
        df_train = df[df["week"].isin(train_weeks)].copy()
        df_test = df[df["week"] == w_test].copy()

        rw = rules[rules["test_week"].astype(str) == w_test].copy()
        if rw.empty:
            continue

        bets_train = apply_rules_max(df_train, rw)
        w_pnl = (
            bets_train.groupby("week")["profit_cap2_max"].sum().reindex(train_weeks, fill_value=0.0).to_numpy(dtype=float)
            if len(train_weeks)
            else np.array([], dtype=float)
        )
        w_stake = (
            bets_train.groupby("week")["stake_eff_max"].sum().reindex(train_weeks, fill_value=0.0).to_numpy(dtype=float)
            if len(train_weeks)
            else np.array([], dtype=float)
        )

        draws_pnl, draws_stake, draws_roi = posterior_predictive_weekly_joint(w_pnl, w_stake, rng=rng, n_draws=N_DRAWS)
        pred_mean = float(np.mean(draws_pnl))
        pred_p10, pred_p50, pred_p90 = (float(np.quantile(draws_pnl, q)) for q in (0.10, 0.50, 0.90))
        pred_p05, pred_p95 = (float(np.quantile(draws_pnl, q)) for q in (0.05, 0.95))
        pred_prob_pos = float(np.mean(draws_pnl > 0))

        pred_stake_mean = float(np.mean(draws_stake))
        pred_roi_mean = float(np.mean(draws_roi))

        bets_test = apply_rules_max(df_test, rw)
        y = float(bets_test["profit_cap2_max"].sum()) if not bets_test.empty else 0.0
        stake = float(bets_test["stake_eff_max"].sum()) if not bets_test.empty else 0.0
        roi = float(y / stake) if stake > 0 else float("nan")

        pit = float(np.mean(draws_pnl <= y))
        crps = crps_from_sample(draws_pnl, y=y)

        out_rows.append(
            {
                "week": w_test,
                "n_train_weeks": int(len(train_weeks)),
                "pnl_max_theoretical": y,
                "stake_max_theoretical": stake,
                "roi_on_stake_max_theoretical": roi,
                "pred_mean": pred_mean,
                "pred_p05": pred_p05,
                "pred_p10": pred_p10,
                "pred_p50": pred_p50,
                "pred_p90": pred_p90,
                "pred_p95": pred_p95,
                "pred_prob_pos": pred_prob_pos,
                "pred_stake_mean": pred_stake_mean,
                "pred_roi_mean": pred_roi_mean,
                "pit": pit,
                "crps": crps,
                "error": y - pred_mean,
            }
        )

    out = pd.DataFrame(out_rows)
    out_path = OUT_DIR / "forecast_calibration_global_bayes_max.csv"
    out.to_csv(out_path, index=False)

    # summary md
    err = out["error"].to_numpy(dtype=float)
    bias = float(np.mean(err))
    cov80 = float(np.mean((out["pnl_max_theoretical"] >= out["pred_p10"]) & (out["pnl_max_theoretical"] <= out["pred_p90"]))) if not out.empty else float("nan")
    pred_mean_mean = float(np.mean(out["pred_mean"].to_numpy(dtype=float))) if not out.empty else float("nan")
    pred_mean_cal = float(pred_mean_mean + bias) if np.isfinite(pred_mean_mean) and np.isfinite(bias) else float("nan")

    lines: List[str] = []
    lines.append("## Calibração forecast — Operação no máximo (stake_eff_max = house_cap)\n\n")
    lines.append(f"- Folds: **{out.shape[0]}**\n")
    lines.append(f"- E[pred_mean] (máx): **USD {pred_mean_mean:,.1f}**\n")
    lines.append(f"- Bias (y - pred) (máx): **USD {bias:,.1f}**\n")
    lines.append(f"- E[pred_mean]+Bias (máx): **USD {pred_mean_cal:,.1f}**\n")
    lines.append(f"- Coverage 80% (p10..p90): **{cov80*100:.1f}%**\n")
    (OUT_DIR / "forecast_calibration_global_bayes_max.md").write_text("".join(lines), encoding="utf-8")
    print(str(out_path))
    print(str(OUT_DIR / "forecast_calibration_global_bayes_max.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

