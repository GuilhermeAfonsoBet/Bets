#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera um quadro comparativo "antes vs depois" da correção por calibração por combinação.

Antes = baseline commit (a206d56): global_bayes sem correção por combinação.
Depois = HEAD atual: global_bayes com correção conservadora por combinação (shrinkage em bias de ROI).

Saídas:
- analysis_proba_raw/pro_portfolio_all/before_after_global_comparison.csv
- analysis_proba_raw/pro_portfolio_all/before_after_rule_comparison.csv
- analysis_proba_raw/pro_portfolio_all/before_after_comparison.md
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


BASELINE_COMMIT = "a206d56"

OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")
SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")

WF_WEEKLY = OUT_DIR / "oos_walkforward_global_bayes_weekly.csv"
WF_SEG = OUT_DIR / "oos_walkforward_global_bayes_weekly_by_segment.csv"
WF_RULES = OUT_DIR / "oos_walkforward_global_bayes_selected_rules.csv"
FC_GLOBAL = OUT_DIR / "forecast_calibration_global_bayes.csv"
FC_RULE_SUM_AFTER = OUT_DIR / "forecast_calibration_global_bayes_by_rule_summary.csv"


def _git_read_csv(commit: str, rel_path: str) -> pd.DataFrame:
    s = subprocess.check_output(["git", "show", f"{commit}:{rel_path}"]).decode("utf-8")
    return pd.read_csv(io.StringIO(s))


def _weekly_stats(weekly_df: pd.DataFrame) -> Dict[str, float]:
    w = weekly_df["profit_cap2_usd"].to_numpy(dtype=float)
    mean = float(np.mean(w))
    std = float(np.std(w, ddof=1)) if w.size > 1 else 0.0
    sharpe_ann = float((mean * 52.0) / (std * np.sqrt(52.0))) if std > 0 else float("nan")
    profit = float(np.sum(w))
    stake = float(weekly_df["stake_usd"].sum())
    roi = float(profit / stake) if stake > 0 else float("nan")
    pneg = float(np.mean(w < 0))
    return {"weeks": int(w.size), "profit": profit, "stake": stake, "roi_on_stake": roi, "mean_week": mean, "std_week": std, "pneg_week": pneg, "sharpe_annual": sharpe_ann}


def _forecast_stats(fc: pd.DataFrame) -> Dict[str, float]:
    err = fc["error"].to_numpy(dtype=float)  # y - pred_mean
    bias = float(np.mean(err))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    cov80 = float(np.mean((fc["pnl_theoretical"] >= fc["pred_p10"]) & (fc["pnl_theoretical"] <= fc["pred_p90"])))
    cov90 = float(np.mean((fc["pnl_theoretical"] >= fc["pred_p05"]) & (fc["pnl_theoretical"] <= fc["pred_p95"])))
    pit = float(np.mean(fc["pit"].to_numpy(dtype=float)))
    crps = float(np.mean(fc["crps"].to_numpy(dtype=float)))
    return {"bias": bias, "mae": mae, "rmse": rmse, "cov80": cov80, "cov90": cov90, "pit_mean": pit, "crps_mean": crps}


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


def _apply_single_rule(df: pd.DataFrame, bet_type: str, dow: str, score_col: str, cutoff: float, stake_frac: float, alpha: float, bankroll: float = 2300.0) -> pd.DataFrame:
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
    stake0 = float(bankroll) * float(stake_frac) * float(alpha)
    x["stake_eff"] = np.minimum(stake0, x["house_cap"].to_numpy(dtype=float))
    x["profit_cap2"] = x["stake_eff"].to_numpy(dtype=float) * x["roi_cap2"].to_numpy(dtype=float)
    x["rule_key"] = f"{bet_type}|{dow}"
    return x[["week", "stake_eff", "profit_cap2", "rule_key"]]


def _roi_pred_weekly_mean(df_train: pd.DataFrame, rule_row: Dict, train_weeks: List[str]) -> float:
    if not train_weeks:
        return 0.0
    bets = _apply_single_rule(
        df_train,
        bet_type=str(rule_row["bet_type"]),
        dow=str(rule_row["dow_pt"]),
        score_col=str(rule_row["score_col"]),
        cutoff=float(rule_row["cutoff"]),
        stake_frac=float(rule_row["stake_frac"]),
        alpha=float(rule_row["alpha_global"]),
    )
    if bets.empty:
        return 0.0
    g = bets.groupby("week", as_index=False).agg(stake=("stake_eff", "sum"), pnl=("profit_cap2", "sum"))
    gm = g.set_index("week").reindex(train_weeks, fill_value=0.0)
    stake = gm["stake"].to_numpy(dtype=float)
    pnl = gm["pnl"].to_numpy(dtype=float)
    roi_w = np.zeros_like(stake, dtype=float)
    np.divide(pnl, stake, out=roi_w, where=(stake > 0))
    return float(np.mean(roi_w))


def _compute_rule_bias_summary_from_rules(
    df_all: pd.DataFrame, rules_df: pd.DataFrame, test_weeks: List[str]
) -> pd.DataFrame:
    """
    Constrói uma tabela tipo *_by_rule_summary.csv (bias ROI/PnL + shrinkage),
    usando pred = média semanal no treino (inclui semanas sem trade como 0).
    """
    week_to_i = {w: i for i, w in enumerate(test_weeks)}
    out = []
    for w_test in sorted(rules_df["test_week"].astype(str).unique().tolist()):
        if w_test not in week_to_i:
            continue
        i = week_to_i[w_test]
        train_weeks = test_weeks[:i]
        df_train = df_all[df_all["week"].isin(train_weeks)].copy()
        df_test = df_all[df_all["week"] == w_test].copy()
        rw = rules_df[rules_df["test_week"].astype(str) == w_test].copy()
        for _, r in rw.iterrows():
            if str(r.get("status")) != "ok":
                continue
            if float(r.get("stake_frac", 0.0)) <= 0:
                continue
            roi_pred = _roi_pred_weekly_mean(df_train, r.to_dict(), train_weeks=train_weeks)
            bets_test = _apply_single_rule(
                df_test,
                bet_type=str(r["bet_type"]),
                dow=str(r["dow_pt"]),
                score_col=str(r["score_col"]),
                cutoff=float(r["cutoff"]),
                stake_frac=float(r["stake_frac"]),
                alpha=float(r["alpha_global"]),
            )
            stake_t = float(bets_test["stake_eff"].sum()) if not bets_test.empty else 0.0
            pnl_t = float(bets_test["profit_cap2"].sum()) if not bets_test.empty else 0.0
            if stake_t <= 0:
                continue
            roi_real = float(pnl_t / stake_t)
            out.append(
                {
                    "week": w_test,
                    "rule_key": str(r.get("rule_key", f"{r['bet_type']}|{r['dow_pt']}")),
                    "error_roi": float(roi_real - roi_pred),
                    "error_pnl": float(pnl_t - (roi_pred * stake_t)),
                }
            )
    if not out:
        return pd.DataFrame(columns=["rule_key"])
    o = pd.DataFrame(out)
    g = o.groupby("rule_key", as_index=False).agg(
        n_obs=("week", "count"),
        bias_roi=("error_roi", "mean"),
        bias_pnl=("error_pnl", "mean"),
    )
    # shrinkage ROI and PnL
    all_roi = o["error_roi"].to_numpy(dtype=float)
    all_roi = all_roi[np.isfinite(all_roi)]
    global_var_roi = float(np.var(all_roi, ddof=1)) if all_roi.size > 1 else 0.0
    stats = []
    for rk, gg in o.groupby("rule_key"):
        a = gg["error_roi"].to_numpy(dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            continue
        v = float(np.var(a, ddof=1)) if a.size > 1 else global_var_roi
        v = float(v if v > 1e-12 else 1e-4)
        stats.append((rk, float(np.mean(a)), float(v / a.size)))
    stats = sorted(stats, key=lambda t: t[0])
    if stats:
        keys = [k for k, _, _ in stats]
        means = np.array([m for _, m, _ in stats], dtype=float)
        se2 = np.array([v for _, _, v in stats], dtype=float)
        _, _, post = _empirical_bayes_shrink(means, se2)
        post_map = {k: float(v) for k, v in zip(keys, post)}
        g["bias_roi_shrunk"] = g["rule_key"].map(post_map).astype(float)
    else:
        g["bias_roi_shrunk"] = float("nan")

    all_pnl = o["error_pnl"].to_numpy(dtype=float)
    all_pnl = all_pnl[np.isfinite(all_pnl)]
    global_var_pnl = float(np.var(all_pnl, ddof=1)) if all_pnl.size > 1 else 0.0
    stats = []
    for rk, gg in o.groupby("rule_key"):
        a = gg["error_pnl"].to_numpy(dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            continue
        v = float(np.var(a, ddof=1)) if a.size > 1 else global_var_pnl
        v = float(v if v > 1e-12 else 1.0)
        stats.append((rk, float(np.mean(a)), float(v / a.size)))
    stats = sorted(stats, key=lambda t: t[0])
    if stats:
        keys = [k for k, _, _ in stats]
        means = np.array([m for _, m, _ in stats], dtype=float)
        se2 = np.array([v for _, _, v in stats], dtype=float)
        _, _, post = _empirical_bayes_shrink(means, se2)
        post_map = {k: float(v) for k, v in zip(keys, post)}
        g["bias_pnl_shrunk"] = g["rule_key"].map(post_map).astype(float)
    else:
        g["bias_pnl_shrunk"] = float("nan")

    return g


def _rule_oos_metrics(weekly: pd.DataFrame, weekly_by_seg: pd.DataFrame) -> pd.DataFrame:
    weeks = weekly["week"].astype(str).tolist()
    total_weeks = len(weeks)
    seg = weekly_by_seg.copy()
    seg["week"] = seg["week"].astype(str)
    out = []
    for rk, g in seg.groupby("rule_key"):
        pnl_map = g.set_index("week")["profit_cap2_usd"].to_dict()
        stake_map = g.set_index("week")["stake_usd"].to_dict()
        pnl = np.array([float(pnl_map.get(w, 0.0)) for w in weeks], dtype=float)
        stake = np.array([float(stake_map.get(w, 0.0)) for w in weeks], dtype=float)
        mean = float(np.mean(pnl))
        std = float(np.std(pnl, ddof=1)) if pnl.size > 1 else 0.0
        active = int(np.sum(stake > 0))
        profit = float(np.sum(pnl))
        stake_tot = float(np.sum(stake))
        roi = float(profit / stake_tot) if stake_tot > 0 else float("nan")
        out.append(
            {
                "rule_key": str(rk),
                "weeks_total": int(total_weeks),
                "weeks_active": int(active),
                "active_rate": float(active / total_weeks) if total_weeks else float("nan"),
                "profit": profit,
                "stake": stake_tot,
                "roi_on_stake": roi,
                "mean_week": mean,
                "std_week": std,
            }
        )
    return pd.DataFrame(out)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # AFTER
    w_after = pd.read_csv(WF_WEEKLY)
    seg_after = pd.read_csv(WF_SEG) if WF_SEG.exists() else pd.DataFrame(columns=["week", "rule_key", "stake_usd", "profit_cap2_usd"])
    fc_after = pd.read_csv(FC_GLOBAL)
    rule_sum_after = pd.read_csv(FC_RULE_SUM_AFTER) if FC_RULE_SUM_AFTER.exists() else pd.DataFrame(columns=["rule_key"])

    # BEFORE
    w_before = _git_read_csv(BASELINE_COMMIT, "analysis_proba_raw/pro_portfolio_all/oos_walkforward_global_bayes_weekly.csv")
    seg_before = _git_read_csv(BASELINE_COMMIT, "analysis_proba_raw/pro_portfolio_all/oos_walkforward_global_bayes_weekly_by_segment.csv")
    fc_before = _git_read_csv(BASELINE_COMMIT, "analysis_proba_raw/pro_portfolio_all/forecast_calibration_global_bayes.csv")
    rules_before = _git_read_csv(BASELINE_COMMIT, "analysis_proba_raw/pro_portfolio_all/oos_walkforward_global_bayes_selected_rules.csv")

    # global table
    gb = {**_weekly_stats(w_before), **{f"fc_{k}": v for k, v in _forecast_stats(fc_before).items()}}
    ga = {**_weekly_stats(w_after), **{f"fc_{k}": v for k, v in _forecast_stats(fc_after).items()}}
    global_comp = pd.DataFrame(
        [
            {"scenario": "before", **gb},
            {"scenario": "after", **ga},
        ]
    )
    global_comp.to_csv(OUT_DIR / "before_after_global_comparison.csv", index=False)

    # per-rule OOS metrics before/after
    oos_before = _rule_oos_metrics(w_before, seg_before)
    oos_after = _rule_oos_metrics(w_after, seg_after)
    oos_before = oos_before.rename(columns={c: f"{c}_before" for c in oos_before.columns if c != "rule_key"})
    oos_after = oos_after.rename(columns={c: f"{c}_after" for c in oos_after.columns if c != "rule_key"})

    # per-rule bias summary before (compute) and after (existing)
    df_all = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])
    df_all["roi_raw"] = pd.to_numeric(df_all["ROI Real"], errors="coerce").astype(float)
    df_all["roi_cap2"] = np.minimum(df_all["roi_raw"].to_numpy(dtype=float), 2.0)
    df_all["house_cap"] = pd.to_numeric(df_all["house_cap"], errors="coerce").astype(float)
    df_all["week"] = pd.to_datetime(df_all["BIA_ApostaUTC"]).dt.to_period("W-SUN").astype(str)

    test_weeks = sorted(df_all["week"].unique().tolist())
    rule_sum_before = _compute_rule_bias_summary_from_rules(df_all, rules_before, test_weeks=test_weeks)
    rule_sum_before = rule_sum_before.rename(columns={c: f"{c}_before" for c in rule_sum_before.columns if c != "rule_key"})

    rule_sum_after2 = rule_sum_after.copy()
    # normalize column names to match
    keep_cols = ["rule_key", "n_obs", "bias_roi", "bias_pnl", "bias_roi_shrunk", "bias_pnl_shrunk"]
    for c in keep_cols:
        if c not in rule_sum_after2.columns:
            rule_sum_after2[c] = float("nan")
    rule_sum_after2 = rule_sum_after2[keep_cols].copy()
    rule_sum_after2 = rule_sum_after2.rename(columns={c: f"{c}_after" for c in rule_sum_after2.columns if c != "rule_key"})

    comp = oos_before.merge(oos_after, on="rule_key", how="outer").merge(rule_sum_before, on="rule_key", how="left").merge(rule_sum_after2, on="rule_key", how="left")
    # deltas
    comp["delta_mean_week"] = comp["mean_week_after"] - comp["mean_week_before"]
    comp["delta_roi_on_stake"] = comp["roi_on_stake_after"] - comp["roi_on_stake_before"]
    comp["delta_bias_roi_shrunk"] = comp["bias_roi_shrunk_after"] - comp["bias_roi_shrunk_before"]
    comp["delta_active_rate"] = comp["active_rate_after"] - comp["active_rate_before"]
    comp.to_csv(OUT_DIR / "before_after_rule_comparison.csv", index=False)

    # markdown summary
    lines = []
    lines.append("## Comparação antes vs depois — calibração por combinação (global_bayes)\n\n")
    lines.append(f"- Baseline commit: `{BASELINE_COMMIT}`\n\n")
    lines.append("### Global (portfólio)\n")
    # compact summary
    b = global_comp[global_comp["scenario"] == "before"].iloc[0].to_dict()
    a = global_comp[global_comp["scenario"] == "after"].iloc[0].to_dict()
    lines.append(f"- **Lucro total (OOS WF)**: antes={b['profit']:.1f}, depois={a['profit']:.1f}\n")
    lines.append(f"- **Lucro médio semanal**: antes={b['mean_week']:.1f}, depois={a['mean_week']:.1f}\n")
    lines.append(f"- **Std semanal**: antes={b['std_week']:.1f}, depois={a['std_week']:.1f}\n")
    lines.append(f"- **Sharpe anualizado**: antes={b['sharpe_annual']:.3f}, depois={a['sharpe_annual']:.3f}\n")
    lines.append(f"- **ROI por $ (turnover)**: antes={b['roi_on_stake']:.4f}, depois={a['roi_on_stake']:.4f}\n")
    lines.append(f"- **Forecast Bias (y - pred)**: antes={b['fc_bias']:.1f}, depois={a['fc_bias']:.1f}\n")
    lines.append(f"- **Coverage 80%**: antes={b['fc_cov80']*100:.1f}%, depois={a['fc_cov80']*100:.1f}%\n\n")

    lines.append("### Top mudanças por combinação\n")
    top_drop = comp.sort_values("delta_mean_week").head(7)
    lines.append("**Maior queda de lucro médio semanal por combinação**\n")
    for _, r in top_drop.iterrows():
        lines.append(
            f"- **{r['rule_key']}**: mean_week {r['mean_week_before']:.1f} -> {r['mean_week_after']:.1f} (Δ {r['delta_mean_week']:.1f}); "
            f"active_rate {r['active_rate_before']:.2f} -> {r['active_rate_after']:.2f}\n"
        )
    lines.append("\n")
    top_bias = comp.sort_values("bias_roi_shrunk_after").head(7)
    lines.append("**Combinações mais otimistas em ROI (após shrinkage)**\n")
    for _, r in top_bias.iterrows():
        broi = r.get("bias_roi_shrunk_after", float("nan"))
        nobs = r.get("n_obs_after", float("nan"))
        lines.append(f"- **{r['rule_key']}**: bias_roi_shrunk={broi:.5f}, n_obs={int(nobs) if np.isfinite(nobs) else 'nan'}\n")

    lines.append("\n### Arquivos\n")
    lines.append("- `analysis_proba_raw/pro_portfolio_all/before_after_global_comparison.csv`\n")
    lines.append("- `analysis_proba_raw/pro_portfolio_all/before_after_rule_comparison.csv`\n")

    (OUT_DIR / "before_after_comparison.md").write_text("".join(lines), encoding="utf-8")
    print(str(OUT_DIR / "before_after_global_comparison.csv"))
    print(str(OUT_DIR / "before_after_rule_comparison.csv"))
    print(str(OUT_DIR / "before_after_comparison.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

