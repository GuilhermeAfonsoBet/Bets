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
import json
from pathlib import Path
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")
SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")
CALIB_SEXDOM = Path("/workspace/clv_calib_SexDom.json")
CALIB_FLOOR_SEXDOM = 0.005
CALIB_SEGQUI = Path("/workspace/clv_calib_SegQui.json")
CALIB_FLOOR_SEGQUI = 0.005

def _safe_mode(s: str) -> str:
    s = str(s).strip()
    return "".join(ch for ch in s if ch.isalnum() or ch in {"_", "-"}).strip("_")


def _paths_for_mode(mode: str) -> tuple[Path, Path, Path, Path]:
    m = _safe_mode(mode)
    rules = OUT_DIR / f"oos_walkforward_{m}_selected_rules.csv"
    weekly = OUT_DIR / f"oos_walkforward_{m}_weekly.csv"
    out_csv = OUT_DIR / f"forecast_calibration_{m}.csv"
    out_md = OUT_DIR / f"forecast_calibration_{m}.md"
    return rules, weekly, out_csv, out_md

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
    """
    Posterior preditivo Bayesiano (bootstrap) para UMA semana futura, preservando a dependência PnL<->Stake:
    - amostra pesos Dirichlet sobre semanas históricas
    - amostra um índice de semana de acordo com esses pesos
    - retorna draws de (pnl, stake, roi=pnl/stake)
    """
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="global_bayes", help="Prefixo do modo do WF (ex.: global_bayes_roll12_robust_p10_p70)")
    args = ap.parse_args()

    RULES, WEEKLY, OUT_CSV, OUT_MD = _paths_for_mode(args.mode)
    rules = pd.read_csv(RULES)
    wf_week = pd.read_csv(WEEKLY)

    df = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])
    if "roi_calc" not in df.columns:
        raise KeyError("Coluna roi_calc ausente. Regerar scored_dedup_proba_raw_all.csv antes de rodar forecast calibration.")
    df["roi_raw"] = pd.to_numeric(df["roi_calc"], errors="coerce").astype(float)
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

        # predictive draws (joint: pnl, stake, roi)
        draws_pnl, draws_stake, draws_roi = posterior_predictive_weekly_joint(w_pnl, w_stake, rng=rng, n_draws=N_DRAWS)
        pred_mean = float(np.mean(draws_pnl))
        pred_p10, pred_p50, pred_p90 = (float(np.quantile(draws_pnl, q)) for q in (0.10, 0.50, 0.90))
        pred_p05, pred_p95 = (float(np.quantile(draws_pnl, q)) for q in (0.05, 0.95))
        pred_prob_pos = float(np.mean(draws_pnl > 0))

        pred_stake_mean = float(np.mean(draws_stake))
        pred_stake_p10, pred_stake_p50, pred_stake_p90 = (float(np.quantile(draws_stake, q)) for q in (0.10, 0.50, 0.90))

        pred_roi_mean = float(np.mean(draws_roi))
        pred_roi_p10, pred_roi_p50, pred_roi_p90 = (float(np.quantile(draws_roi, q)) for q in (0.10, 0.50, 0.90))

        # realized theoretical
        row = wf_week[wf_week["week"].astype(str) == w_test].iloc[0]
        y = float(row["profit_cap2_usd"])
        stake = float(row["stake_usd"])
        roi = float(y / stake) if stake > 0 else float("nan")

        pit = float(np.mean(draws_pnl <= y))
        crps = crps_from_sample(draws_pnl, y=y)

        # decomposição do erro: PnL = Stake * ROI
        # Nota: como a previsão de PnL usa Phat = E[S*R] (draws conjuntos), precisamos expor também o termo
        # de covariância: cov = E[S*R] - E[S]E[R], para fechar a conta.
        stake_theo = float(stake)
        roi_theo = float(roi) if np.isfinite(float(roi)) else 0.0
        dS = stake_theo - pred_stake_mean
        dR = roi_theo - pred_roi_mean
        err_stake = dS * pred_roi_mean
        err_roi = pred_stake_mean * dR
        err_interaction = dS * dR
        cov_term = pred_mean - (pred_stake_mean * pred_roi_mean)  # E[S*R] - E[S]E[R]
        err_decomp_total = err_stake + err_roi + err_interaction - cov_term  # == (y - E[S*R]) = error
        err_vs_prodmeans = (y - (pred_stake_mean * pred_roi_mean))  # y - E[S]E[R]

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
                "pred_stake_mean": pred_stake_mean,
                "pred_stake_p10": pred_stake_p10,
                "pred_stake_p50": pred_stake_p50,
                "pred_stake_p90": pred_stake_p90,
                "pred_roi_mean": pred_roi_mean,
                "pred_roi_p10": pred_roi_p10,
                "pred_roi_p50": pred_roi_p50,
                "pred_roi_p90": pred_roi_p90,
                "pit": pit,
                "crps": crps,
                "error": y - pred_mean,
                "abs_error": abs(y - pred_mean),
                "error_stake_component": err_stake,
                "error_roi_component": err_roi,
                "error_interaction": err_interaction,
                "pred_cov_term": cov_term,
                "error_vs_prodmeans": err_vs_prodmeans,
                "error_decomp_total": err_decomp_total,
            }
        )

    out = pd.DataFrame(out_rows)
    out_path = OUT_CSV
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
    md.append("### Decomposição do erro (stake vs ROI)\n")
    md.append("- Identidade: PnL = Stake * ROI.\n")
    md.append("- Definições no preditivo: P̂ = E[S*R], Ŝ = E[S], R̂ = E[R].\n")
    md.append("- Termo de dependência (cov): cov = E[S*R] - E[S]E[R].\n")
    md.append("- Decomposição que fecha a conta do erro y - P̂:\n")
    md.append("  y - P̂ = (S-Ŝ)R̂ + Ŝ(R-R̂) + (S-Ŝ)(R-R̂) - cov.\n")
    md.append(f"- Média componente **stake** ((S-Ŝ)R̂): **USD {float(out['error_stake_component'].mean()):,.1f}**\n")
    md.append(f"- Média componente **ROI** (Ŝ(R-R̂)): **USD {float(out['error_roi_component'].mean()):,.1f}**\n")
    md.append(f"- Média **interação** ((S-Ŝ)(R-R̂)): **USD {float(out['error_interaction'].mean()):,.1f}**\n")
    md.append(f"- Média **cov** (E[S*R]-E[S]E[R]): **USD {float(out['pred_cov_term'].mean()):,.1f}**\n")
    md.append(f"- Média total (decomp): **USD {float(out['error_decomp_total'].mean()):,.1f}** (deve bater com Bias)\n\n")

    # diagnóstico direto de stake e ROI (somente semanas com stake>0)
    nz = out[out["stake_theoretical"] > 0].copy()
    if not nz.empty:
        stake_bias = float((nz["stake_theoretical"] - nz["pred_stake_mean"]).mean())
        roi_bias = float((nz["roi_on_stake_theoretical"] - nz["pred_roi_mean"]).mean())
        md.append("### Diagnóstico direto (stake e ROI, semanas com trade)\n")
        md.append(f"- Stake: média (real - previsto): **USD {stake_bias:,.1f}** (positivo => o previsto estava menor que o realizado)\n")
        md.append(f"- ROI: média (real - previsto): **{roi_bias:.5f}** (negativo => o previsto estava maior que o realizado)\n\n")
    md.append("### Calibração probabilística\n")
    md.append(f"- Coverage 80% (p10..p90): **{cov80*100:.1f}%** (ideal ~80%)\n")
    md.append(f"- Coverage 90% (p05..p95): **{cov90*100:.1f}%** (ideal ~90%)\n")
    md.append(f"- PIT médio: **{mean_pit:.3f}** (ideal ~0.5)\n")
    md.append(f"- CRPS médio (aprox): **{crps_mean:,.1f}** (menor é melhor)\n\n")
    md.append("### Arquivos\n")
    md.append(f"- CSV: `analysis_proba_raw/pro_portfolio_all/{out_path.name}`\n")
    OUT_MD.write_text("".join(md), encoding="utf-8")
    print(str(out_path))
    print(str(OUT_MD))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

