#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibração por combinação (rule_key = bet_type|dow):
  Forecast (distribuição preditiva) -> Realizado Teórico, por segmento.

Para cada fold semanal do walk-forward (test_week):
- Usa as regras θ_t daquele test_week (cutoff, stake_frac, alpha_global).
- Aplica θ_t nas semanas de treino (expanding) para obter séries semanais:
    (pnl_week, stake_week) por segmento
- Gera distribuição preditiva conjunta (PnL, Stake, ROI) via Bayesian bootstrap:
    - Dirichlet weights sobre semanas históricas
    - amostra um índice de semana segundo esses pesos
    - draws de (pnl, stake, roi=pnl/stake)
- Realizado teórico: aplica θ_t na semana teste e agrega por segmento.

Saídas:
- analysis_proba_raw/pro_portfolio_all/forecast_calibration_global_bayes_by_rule.csv
- analysis_proba_raw/pro_portfolio_all/forecast_calibration_global_bayes_by_rule_summary.csv
- analysis_proba_raw/pro_portfolio_all/forecast_calibration_global_bayes_by_rule.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import json


OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")
SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")

RULES = OUT_DIR / "oos_walkforward_global_bayes_selected_rules.csv"
WEEKLY = OUT_DIR / "oos_walkforward_global_bayes_weekly.csv"

BANKROLL = 2300.0
N_DRAWS = 10_000
SEED = 7
CALIB_SEXDOM = Path("/workspace/clv_calib_SexDom.json")
CALIB_FLOOR_SEXDOM = 0.005
CALIB_SEGQUI = Path("/workspace/clv_calib_SegQui.json")
CALIB_FLOOR_SEGQUI = 0.005


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


def apply_single_rule(df: pd.DataFrame, bet_type: str, dow: str, score_col: str, cutoff: float, stake_frac: float, alpha: float) -> pd.DataFrame:
    if stake_frac <= 0:
        return df.iloc[:0].copy()
    x = df[(df["bet_type"] == bet_type) & (df["dow_pt"] == dow)].copy()
    if x.empty:
        return x
    score = pd.to_numeric(x[score_col], errors="coerce").to_numpy(dtype=float)
    roi2 = x["roi_cap2"].to_numpy(dtype=float)
    m = np.isfinite(score) & (score >= float(cutoff)) & np.isfinite(roi2)
    if not np.any(m):
        return x.iloc[:0].copy()
    x = x.iloc[np.where(m)[0]].copy()
    stake0 = BANKROLL * float(stake_frac) * float(alpha)
    x["stake_eff"] = np.minimum(stake0, x["house_cap"].to_numpy(dtype=float))
    x["profit_cap2"] = x["stake_eff"].to_numpy(dtype=float) * x["roi_cap2"].to_numpy(dtype=float)
    x["rule_key"] = f"{bet_type}|{dow}"
    return x[["week", "stake_eff", "profit_cap2", "rule_key"]]


def _empirical_bayes_shrink(means: np.ndarray, se2: np.ndarray) -> Tuple[float, float, np.ndarray]:
    m = np.asarray(means, dtype=float)
    v = np.asarray(se2, dtype=float)
    ok = np.isfinite(m) & np.isfinite(v) & (v > 0)
    m = m[ok]
    v = v[ok]
    if m.size == 0:
        return float("nan"), 0.0, np.array([], dtype=float)
    w = 1.0 / v
    mu0 = float(np.sum(w * m) / np.sum(w))
    var_m = float(np.var(m, ddof=1)) if m.size > 1 else 0.0
    tau2 = float(max(0.0, var_m - float(np.mean(v))))
    if tau2 <= 1e-12:
        post = np.full(m.size, mu0, dtype=float)
    else:
        post = (m / v + mu0 / tau2) / (1.0 / v + 1.0 / tau2)
    return mu0, tau2, post


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

    out_rows: List[Dict] = []

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
        alpha = float(rw["alpha_global"].iloc[0]) if "alpha_global" in rw.columns else 1.0

        for _, r in rw.iterrows():
            if str(r.get("status")) != "ok":
                continue
            stake_frac = float(r.get("stake_frac", 0.0))
            if stake_frac <= 0:
                continue
            bet_type = str(r["bet_type"])
            dow = str(r["dow_pt"])
            score_col = str(r["score_col"])
            cutoff = float(r["cutoff"])
            rule_key = str(r.get("rule_key", f"{bet_type}|{dow}"))

            # training weekly series for this rule
            bets_train = apply_single_rule(df_train, bet_type, dow, score_col, cutoff, stake_frac, alpha=alpha)
            w_pnl = (
                bets_train.groupby("week")["profit_cap2"].sum().reindex(train_weeks, fill_value=0.0).to_numpy(dtype=float)
                if len(train_weeks)
                else np.array([], dtype=float)
            )
            w_stake = (
                bets_train.groupby("week")["stake_eff"].sum().reindex(train_weeks, fill_value=0.0).to_numpy(dtype=float)
                if len(train_weeks)
                else np.array([], dtype=float)
            )

            draws_pnl, draws_stake, draws_roi = posterior_predictive_weekly_joint(w_pnl, w_stake, rng=rng, n_draws=N_DRAWS)
            pred_mean = float(np.mean(draws_pnl))
            pred_p10, pred_p50, pred_p90 = (float(np.quantile(draws_pnl, q)) for q in (0.10, 0.50, 0.90))
            pred_p05, pred_p95 = (float(np.quantile(draws_pnl, q)) for q in (0.05, 0.95))

            pred_stake_mean = float(np.mean(draws_stake))
            pred_roi_mean = float(np.mean(draws_roi))

            # realized on test week
            bets_test = apply_single_rule(df_test, bet_type, dow, score_col, cutoff, stake_frac, alpha=alpha)
            y = float(bets_test["profit_cap2"].sum()) if not bets_test.empty else 0.0
            s = float(bets_test["stake_eff"].sum()) if not bets_test.empty else 0.0
            roi = float(y / s) if s > 0 else float("nan")

            pit = float(np.mean(draws_pnl <= y))
            crps = crps_from_sample(draws_pnl, y=y)

            out_rows.append(
                {
                    "week": w_test,
                    "rule_key": rule_key,
                    "alpha": float(alpha),
                    "cutoff": float(cutoff),
                    "stake_frac": float(stake_frac),
                    "n_train_weeks": int(len(train_weeks)),
                    "pnl_theoretical": y,
                    "stake_theoretical": s,
                    "roi_theoretical": roi,
                    "pred_pnl_mean": pred_mean,
                    "pred_pnl_p05": pred_p05,
                    "pred_pnl_p10": pred_p10,
                    "pred_pnl_p50": pred_p50,
                    "pred_pnl_p90": pred_p90,
                    "pred_pnl_p95": pred_p95,
                    "pred_stake_mean": pred_stake_mean,
                    "pred_roi_mean": pred_roi_mean,
                    "error_pnl": y - pred_mean,
                    "error_stake": s - pred_stake_mean,
                    "error_roi": (roi - pred_roi_mean) if np.isfinite(roi) else float("nan"),
                    "pit": pit,
                    "crps": crps,
                }
            )

    out = pd.DataFrame(out_rows)
    out_path = OUT_DIR / "forecast_calibration_global_bayes_by_rule.csv"
    out.to_csv(out_path, index=False)

    # summary + shrinkage
    g = out.groupby("rule_key", as_index=False).agg(
        n_obs=("week", "count"),
        bias_pnl=("error_pnl", "mean"),
        mae_pnl=("error_pnl", lambda x: float(np.mean(np.abs(np.asarray(x, dtype=float))))),
        bias_roi=("error_roi", "mean"),
        bias_stake=("error_stake", "mean"),
        pit_mean=("pit", "mean"),
        crps_mean=("crps", "mean"),
    )

    # shrinkage for ROI bias
    # se2: var(error)/n
    all_roi = out["error_roi"].to_numpy(dtype=float)
    all_roi = all_roi[np.isfinite(all_roi)]
    global_var_roi = float(np.var(all_roi, ddof=1)) if all_roi.size > 1 else 0.0
    all_pnl = out["error_pnl"].to_numpy(dtype=float)
    all_pnl = all_pnl[np.isfinite(all_pnl)]
    global_var_pnl = float(np.var(all_pnl, ddof=1)) if all_pnl.size > 1 else 0.0
    roi_stats = []
    pnl_stats = []
    for rk, gg in out.groupby("rule_key"):
        a = gg["error_roi"].to_numpy(dtype=float)
        a = a[np.isfinite(a)]
        if a.size:
            v = float(np.var(a, ddof=1)) if a.size > 1 else global_var_roi
            v = float(v if v > 1e-12 else 1e-4)
            roi_stats.append((rk, float(np.mean(a)), float(v / a.size)))
        b = gg["error_pnl"].to_numpy(dtype=float)
        b = b[np.isfinite(b)]
        if b.size:
            v2 = float(np.var(b, ddof=1)) if b.size > 1 else global_var_pnl
            v2 = float(v2 if v2 > 1e-12 else 1.0)
            pnl_stats.append((rk, float(np.mean(b)), float(v2 / b.size)))

    roi_stats = sorted(roi_stats, key=lambda t: t[0])
    pnl_stats = sorted(pnl_stats, key=lambda t: t[0])

    if roi_stats:
        keys = [k for k, _, _ in roi_stats]
        means = np.array([m for _, m, _ in roi_stats], dtype=float)
        se2 = np.array([v for _, _, v in roi_stats], dtype=float)
        mu0, tau2, post = _empirical_bayes_shrink(means, se2)
        post_map = {k: float(v) for k, v in zip(keys, post)}
        g["bias_roi_shrunk"] = g["rule_key"].map(post_map).astype(float)
        g["bias_roi_shrunk_mu0"] = float(mu0)
        g["bias_roi_shrunk_tau2"] = float(tau2)
    else:
        g["bias_roi_shrunk"] = float("nan")
        g["bias_roi_shrunk_mu0"] = float("nan")
        g["bias_roi_shrunk_tau2"] = float("nan")

    if pnl_stats:
        keys = [k for k, _, _ in pnl_stats]
        means = np.array([m for _, m, _ in pnl_stats], dtype=float)
        se2 = np.array([v for _, _, v in pnl_stats], dtype=float)
        mu0, tau2, post = _empirical_bayes_shrink(means, se2)
        post_map = {k: float(v) for k, v in zip(keys, post)}
        g["bias_pnl_shrunk"] = g["rule_key"].map(post_map).astype(float)
        g["bias_pnl_shrunk_mu0"] = float(mu0)
        g["bias_pnl_shrunk_tau2"] = float(tau2)
    else:
        g["bias_pnl_shrunk"] = float("nan")
        g["bias_pnl_shrunk_mu0"] = float("nan")
        g["bias_pnl_shrunk_tau2"] = float("nan")

    sum_path = OUT_DIR / "forecast_calibration_global_bayes_by_rule_summary.csv"
    g.sort_values(["rule_key"]).to_csv(sum_path, index=False)

    # markdown report
    lines = []
    lines.append("## Calibração por combinação (rule_key) — global_bayes\n\n")
    lines.append(f"- Observações (linhas): **{out.shape[0]}**\n")
    lines.append(f"- Segmentos (rule_key): **{g.shape[0]}**\n\n")
    lines.append("### Interpretação rápida\n")
    lines.append("- `error_roi = ROI_real - ROI_previsto`: negativo => previsão otimista de ROI.\n")
    lines.append("- `bias_roi_shrunk`: bias de ROI com shrinkage (pooling) entre segmentos.\n\n")
    lines.append("### Segmentos com ROI mais otimista (bias shrunken mais negativo)\n")
    top = g.sort_values("bias_roi_shrunk").head(7)
    for _, r in top.iterrows():
        lines.append(f"- **{r['rule_key']}**: bias_roi_shrunk={float(r['bias_roi_shrunk']):.5f}, n={int(r['n_obs'])}\n")
    lines.append("\n### Arquivos\n")
    lines.append(f"- CSV (por semana e segmento): `analysis_proba_raw/pro_portfolio_all/{out_path.name}`\n")
    lines.append(f"- CSV (resumo + shrinkage): `analysis_proba_raw/pro_portfolio_all/{sum_path.name}`\n")

    (OUT_DIR / "forecast_calibration_global_bayes_by_rule.md").write_text("".join(lines), encoding="utf-8")
    print(str(out_path))
    print(str(sum_path))
    print(str(OUT_DIR / "forecast_calibration_global_bayes_by_rule.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

