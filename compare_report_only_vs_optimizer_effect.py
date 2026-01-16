#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Separa o impacto da calibração em:
1) Reporte (ajustar expectativas/valores reportados sem mudar o OOS realizado)
2) Otimizador (mudar seleção/cutoff/stake, portanto muda o OOS realizado)

Produz 3 cenários:
- baseline: commit a206d56 (sem correção por combinação no otimizador)
- report_only: mesmo OOS do baseline, mas com projeções ajustadas por calibração (bias)
- optimizer: HEAD atual (com correção por combinação no otimizador)

Saídas:
- analysis_proba_raw/pro_portfolio_all/report_vs_optimizer_global.csv
- analysis_proba_raw/pro_portfolio_all/report_vs_optimizer_rules.csv
- analysis_proba_raw/pro_portfolio_all/report_vs_optimizer.md
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


BASELINE_COMMIT = "a206d56"
OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")


def _git_csv(commit: str, rel_path: str) -> pd.DataFrame:
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
    pred_mean_mean = float(np.mean(fc["pred_mean"].to_numpy(dtype=float))) if "pred_mean" in fc.columns else float("nan")
    return {
        "bias": bias,
        "mae": mae,
        "rmse": rmse,
        "cov80": cov80,
        "cov90": cov90,
        "pit_mean": pit,
        "crps_mean": crps,
        "pred_mean_mean": pred_mean_mean,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # baseline
    w_base = _git_csv(BASELINE_COMMIT, "analysis_proba_raw/pro_portfolio_all/oos_walkforward_global_bayes_weekly.csv")
    seg_base = _git_csv(BASELINE_COMMIT, "analysis_proba_raw/pro_portfolio_all/oos_walkforward_global_bayes_weekly_by_segment.csv")
    fc_base = _git_csv(BASELINE_COMMIT, "analysis_proba_raw/pro_portfolio_all/forecast_calibration_global_bayes.csv")

    # optimizer (after)
    w_opt = pd.read_csv(OUT_DIR / "oos_walkforward_global_bayes_weekly.csv")
    seg_opt = pd.read_csv(OUT_DIR / "oos_walkforward_global_bayes_weekly_by_segment.csv")
    fc_opt = pd.read_csv(OUT_DIR / "forecast_calibration_global_bayes.csv")

    gb = {**_weekly_stats(w_base), **{f"fc_{k}": v for k, v in _forecast_stats(fc_base).items()}}
    go = {**_weekly_stats(w_opt), **{f"fc_{k}": v for k, v in _forecast_stats(fc_opt).items()}}

    # report-only = mesmo OOS realizado do baseline, mas com projeções ajustadas:
    # - forecast mean (média prevista) = E[pred_mean]
    # - forecast mean corrigida = E[pred_mean] + Bias
    gr = dict(gb)
    gr["forecast_mean_week"] = float(gb["fc_pred_mean_mean"])
    gr["forecast_mean_week_cal"] = float(gb["fc_pred_mean_mean"] + gb["fc_bias"]) if np.isfinite(gb["fc_pred_mean_mean"]) and np.isfinite(gb["fc_bias"]) else float("nan")
    gr["roi_bank_week_forecast"] = float(gr["forecast_mean_week"] / 2300.0) if np.isfinite(gr["forecast_mean_week"]) else float("nan")
    gr["roi_bank_week_forecast_cal"] = float(gr["forecast_mean_week_cal"] / 2300.0) if np.isfinite(gr["forecast_mean_week_cal"]) else float("nan")

    # baseline/optimizer: forecast mean e corrigida (para mostrar que isso é "reporte", não muda realizado)
    gb["forecast_mean_week"] = float(gb["fc_pred_mean_mean"])
    gb["forecast_mean_week_cal"] = float(gb["fc_pred_mean_mean"] + gb["fc_bias"]) if np.isfinite(gb["fc_pred_mean_mean"]) and np.isfinite(gb["fc_bias"]) else float("nan")
    go["forecast_mean_week"] = float(go["fc_pred_mean_mean"])
    go["forecast_mean_week_cal"] = float(go["fc_pred_mean_mean"] + go["fc_bias"]) if np.isfinite(go["fc_pred_mean_mean"]) and np.isfinite(go["fc_bias"]) else float("nan")

    global_df = pd.DataFrame(
        [
            {"scenario": "baseline", **gb},
            {"scenario": "report_only", **gr},
            {"scenario": "optimizer", **go},
        ]
    )
    global_df.to_csv(OUT_DIR / "report_vs_optimizer_global.csv", index=False)

    # per-rule realized metrics: baseline vs optimizer (report-only == baseline realized)
    def rule_metrics(weekly: pd.DataFrame, seg: pd.DataFrame) -> pd.DataFrame:
        weeks = weekly["week"].astype(str).tolist()
        seg = seg.copy()
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
                    "weeks_total": int(len(weeks)),
                    "weeks_active": int(active),
                    "active_rate": float(active / len(weeks)) if len(weeks) else float("nan"),
                    "profit": profit,
                    "stake": stake_tot,
                    "roi_on_stake": roi,
                    "mean_week": mean,
                    "std_week": std,
                }
            )
        return pd.DataFrame(out)

    rb = rule_metrics(w_base, seg_base).rename(columns={c: f"{c}_baseline" for c in rule_metrics(w_base, seg_base).columns if c != "rule_key"})
    ro = rule_metrics(w_opt, seg_opt).rename(columns={c: f"{c}_optimizer" for c in rule_metrics(w_opt, seg_opt).columns if c != "rule_key"})
    rr = rb.copy()
    # report-only realized == baseline
    rr = rr.rename(columns={c: c.replace("_baseline", "_report_only") for c in rr.columns if c != "rule_key"})
    rules = rb.merge(rr, on="rule_key", how="outer").merge(ro, on="rule_key", how="outer")
    rules["delta_mean_week_optimizer_vs_baseline"] = rules["mean_week_optimizer"] - rules["mean_week_baseline"]
    rules["delta_active_rate_optimizer_vs_baseline"] = rules["active_rate_optimizer"] - rules["active_rate_baseline"]
    rules.to_csv(OUT_DIR / "report_vs_optimizer_rules.csv", index=False)

    # markdown summary
    lines = []
    lines.append("## Separação de efeito: Reporte vs Otimizador (global_bayes)\n\n")
    lines.append(f"- Baseline commit: `{BASELINE_COMMIT}`\n\n")
    b = global_df[global_df["scenario"] == "baseline"].iloc[0].to_dict()
    r = global_df[global_df["scenario"] == "report_only"].iloc[0].to_dict()
    o = global_df[global_df["scenario"] == "optimizer"].iloc[0].to_dict()
    lines.append("### Global\n")
    lines.append(f"- **Baseline realizado**: mean/sem={b['mean_week']:.1f}, Sharpe_ann={b['sharpe_annual']:.3f}, ROI/$={b['roi_on_stake']:.4f}\n")
    lines.append(f"- **Reporte (baseline)**: forecast mean={b['forecast_mean_week']:.1f}; forecast corrigido={b['forecast_mean_week_cal']:.1f}\n")
    lines.append(f"- **Otimizador ajustado (novo OOS)**: mean/sem={o['mean_week']:.1f}, Sharpe_ann={o['sharpe_annual']:.3f}, ROI/$={o['roi_on_stake']:.4f}\n\n")
    lines.append("### Principais diferenças\n")
    lines.append("- Reporte ajustado altera apenas expectativas/projeções; não muda o realizado OOS.\n")
    lines.append("- Otimizador ajustado muda seleção/stakes/cutoffs; portanto muda o realizado OOS.\n\n")
    lines.append("### Arquivos\n")
    lines.append("- `analysis_proba_raw/pro_portfolio_all/report_vs_optimizer_global.csv`\n")
    lines.append("- `analysis_proba_raw/pro_portfolio_all/report_vs_optimizer_rules.csv`\n")

    (OUT_DIR / "report_vs_optimizer.md").write_text("".join(lines), encoding="utf-8")
    print(str(OUT_DIR / "report_vs_optimizer_global.csv"))
    print(str(OUT_DIR / "report_vs_optimizer_rules.csv"))
    print(str(OUT_DIR / "report_vs_optimizer.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

