#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Avalia a "qualidade do modelo completo" como previsão:
  PnL Previsto -> PnL Realizado Teórico

Definições:
- Para cada fold semanal do walk-forward (test_week):
  - Regras θ_t (cutoff/stake_frac por segmento + alpha_global) já são as que foram otimizadas usando apenas
    o histórico anterior.
  - PnL Realizado Teórico: PnL na semana teste aplicando θ_t (já disponível em oos_walkforward_*_weekly.csv).
  - PnL Previsto: distribuição posterior preditiva de PnL de UMA semana, usando apenas as semanas de treino,
    aplicando θ_t nessas semanas e gerando posterior preditiva via Bayesian bootstrap (Dirichlet).

Saídas:
- analysis_proba_raw/pro_portfolio_all/forecast_calibration_global_bayes.csv
- analysis_proba_raw/pro_portfolio_all/forecast_calibration_global_bayes.md
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")
SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")

RULES = OUT_DIR / "oos_walkforward_global_bayes_selected_rules.csv"
WEEKLY = OUT_DIR / "oos_walkforward_global_bayes_weekly.csv"

BANKROLL = 2300.0

N_DRAWS = 10_000  # posterior predictive draws per fold
SEED = 7


def week_order(df: pd.DataFrame) -> List[str]:
    w = df["week"].astype(str).unique().tolist()
    w_sorted = sorted(w)
    return w_sorted


def apply_rules_to_df(df: pd.DataFrame, rules_week: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """
    Aplica regras (por segmento) em df, retorna DataFrame de bets com stake/profit e week.
    """
    rows = []
    for _, r in rules_week.iterrows():
        if str(r.get("status")) != "ok":
            continue
        stake_frac = float(r.get("stake_frac", 0.0))
        if stake_frac <= 0:
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
        stake0 = BANKROLL * stake_frac * float(alpha)
        x["stake_eff"] = np.minimum(stake0, x["house_cap"].to_numpy(dtype=float))
        x["profit_cap2"] = x["stake_eff"].to_numpy(dtype=float) * x["roi_cap2"].to_numpy(dtype=float)
        x["rule_key"] = str(r["rule_key"])
        rows.append(x[["week", "stake_eff", "profit_cap2", "rule_key"]])
    return pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame(columns=["week", "stake_eff", "profit_cap2", "rule_key"])


def posterior_predictive_weekly(w_train: np.ndarray, rng: np.random.Generator, n_draws: int) -> np.ndarray:
    """
    Bayesian bootstrap posterior predictive for a single future week:
    - sample Dirichlet weights over historical weeks
    - sample one week index from those weights
    """
    w = np.asarray(w_train, dtype=float)
    n = w.size
    if n == 0:
        return np.zeros(n_draws, dtype=float)
    draws = np.empty(n_draws, dtype=float)
    for i in range(n_draws):
        p = rng.dirichlet(np.ones(n))
        idx = rng.choice(n, p=p)
        draws[i] = w[idx]
    return draws


def crps_from_sample(draws: np.ndarray, y: float) -> float:
    """
    CRPS approximation using sample draws:
    CRPS(F,y) = E|X-y| - 0.5 E|X-X'|
    """
    x = np.asarray(draws, dtype=float)
    if x.size == 0:
        return float("nan")
    a = float(np.mean(np.abs(x - y)))
    # approximate E|X-X'| using a subsample for speed
    m = min(2000, x.size)
    xs = x[:m]
    # E|X-X'| = mean over all pairs; approximate by pairing with shuffled
    xp = np.roll(xs, 1)
    b = float(np.mean(np.abs(xs - xp)))
    return a - 0.5 * b


def main() -> int:
    rules = pd.read_csv(RULES)
    wf_week = pd.read_csv(WEEKLY)

    df = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])
    df["roi_raw"] = pd.to_numeric(df["ROI Real"], errors="coerce").astype(float)
    df["roi_cap2"] = np.minimum(df["roi_raw"].to_numpy(dtype=float), 2.0)
    df["house_cap"] = pd.to_numeric(df["house_cap"], errors="coerce").astype(float)
    df["week"] = pd.to_datetime(df["BIA_ApostaUTC"]).dt.to_period("W-SUN").astype(str)

    weeks_sorted = sorted(df["week"].unique().tolist())
    week_to_i = {w: i for i, w in enumerate(weeks_sorted)}

    out_rows = []
    rng = np.random.default_rng(SEED)

    for w_test in wf_week["week"].astype(str).tolist():
        if w_test not in week_to_i:
            continue
        i = week_to_i[w_test]
        train_weeks = weeks_sorted[:i]
        df_train = df[df["week"].isin(train_weeks)].copy()

        # rules of this test week
        rw = rules[rules["test_week"].astype(str) == w_test].copy()
        alpha = float(rw["alpha_global"].iloc[0]) if (not rw.empty and "alpha_global" in rw.columns) else 1.0

        bets_train = apply_rules_to_df(df_train, rw, alpha=alpha)
        # weekly pnl series for training weeks (include 0 for no-trade weeks)
        w_pnl = bets_train.groupby("week")["profit_cap2"].sum().reindex(train_weeks, fill_value=0.0).to_numpy(dtype=float) if len(train_weeks) else np.array([], dtype=float)
        w_stake = bets_train.groupby("week")["stake_eff"].sum().reindex(train_weeks, fill_value=0.0).to_numpy(dtype=float) if len(train_weeks) else np.array([], dtype=float)

        # predictive draws
        draws = posterior_predictive_weekly(w_pnl, rng=rng, n_draws=N_DRAWS)
        pred_mean = float(np.mean(draws))
        pred_p10, pred_p50, pred_p90 = (float(np.quantile(draws, q)) for q in (0.10, 0.50, 0.90))
        pred_p05, pred_p95 = (float(np.quantile(draws, q)) for q in (0.05, 0.95))
        pred_prob_pos = float(np.mean(draws > 0))

        # realized theoretical
        row = wf_week[wf_week["week"].astype(str) == w_test].iloc[0]
        y = float(row["profit_cap2_usd"])
        stake = float(row["stake_usd"])
        roi = float(y / stake) if stake > 0 else float("nan")

        pit = float(np.mean(draws <= y))
        crps = crps_from_sample(draws, y=y)

        out_rows.append(
            {
                "week": w_test,
                "n_train_weeks": int(len(train_weeks)),
                "alpha": float(alpha),
                "pnl_theoretical": y,
                "stake_theoretical": stake,
                "roi_on_stake_theoretical": roi,
                "pred_mean": pred_mean,
                "pred_p05": pred_p05,
                "pred_p10": pred_p10,
                "pred_p50": pred_p50,
                "pred_p90": pred_p90,
                "pred_p95": pred_p95,
                "pred_prob_pos": pred_prob_pos,
                "pit": pit,
                "crps": crps,
                "error": y - pred_mean,
                "abs_error": abs(y - pred_mean),
            }
        )

    out = pd.DataFrame(out_rows)
    out_path = OUT_DIR / "forecast_calibration_global_bayes.csv"
    out.to_csv(out_path, index=False)

    # aggregate diagnostics
    err = out["error"].to_numpy(dtype=float)
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(err * err)))
    cov80 = float(np.mean((out["pnl_theoretical"] >= out["pred_p10"]) & (out["pnl_theoretical"] <= out["pred_p90"])))
    cov90 = float(np.mean((out["pnl_theoretical"] >= out["pred_p05"]) & (out["pnl_theoretical"] <= out["pred_p95"])))
    mean_pit = float(np.mean(out["pit"]))
    crps_mean = float(np.mean(out["crps"]))

    md = []
    md.append("## Calibração do modelo completo — PnL Previsto -> PnL Teórico Realizado (global_bayes)\n\n")
    md.append(f"- Folds (semanas WF): **{out.shape[0]}**\n")
    md.append(f"- Draws por fold (posterior preditivo): **{N_DRAWS}**\n\n")
    md.append("### Erros (ponto: média prevista)\n")
    md.append(f"- Bias (média do erro): **USD {bias:,.1f}**\n")
    md.append(f"- MAE: **USD {mae:,.1f}**\n")
    md.append(f"- RMSE: **USD {rmse:,.1f}**\n\n")
    md.append("### Calibração probabilística\n")
    md.append(f"- Coverage 80% (p10..p90): **{cov80*100:.1f}%** (ideal ~80%)\n")
    md.append(f"- Coverage 90% (p05..p95): **{cov90*100:.1f}%** (ideal ~90%)\n")
    md.append(f"- PIT médio: **{mean_pit:.3f}** (ideal ~0.5)\n")
    md.append(f"- CRPS médio (aprox): **{crps_mean:,.1f}** (menor é melhor)\n\n")
    md.append("### Arquivos\n")
    md.append(f"- CSV: `analysis_proba_raw/pro_portfolio_all/{out_path.name}`\n")
    (OUT_DIR / "forecast_calibration_global_bayes.md").write_text("".join(md), encoding="utf-8")
    print(str(out_path))
    print(str(OUT_DIR / "forecast_calibration_global_bayes.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

