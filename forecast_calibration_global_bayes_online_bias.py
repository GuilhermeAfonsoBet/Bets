#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bias-adjusted on-line (walk-forward) para o forecast de PnL semanal.

Entrada:
- analysis_proba_raw/pro_portfolio_all/forecast_calibration_global_bayes.csv
  (contém, por semana, pnl_theoretical, pred_mean, quantis e métricas do preditivo)

Saídas:
- analysis_proba_raw/pro_portfolio_all/forecast_calibration_global_bayes_online_bias.csv
- analysis_proba_raw/pro_portfolio_all/forecast_calibration_global_bayes_online_bias.md

Definições:
Para cada semana t:
  y_t = PnL teórico realizado
  mu_t = pred_mean (modelo cru)
  e_t = y_t - mu_t

Bias on-line (somente passado):
  bias_t = mean(e_{t-1}, e_{t-2}, ..., e_{t-K})  (rolling window)
onde K = BIAS_WINDOW (default 8). Se não houver histórico suficiente, usa os disponíveis (ou 0 na 1ª semana).

Previsão corrigida:
  mu_t_adj = mu_t + bias_t

Colunas “prática vs modelo” pedidas:
  1) mu_t (modelo cru)
  2) mu_t_adj (bias-adjusted on-line)
  3) y_t - mu_t
  4) y_t - mu_t_adj
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")
IN_CSV = OUT_DIR / "forecast_calibration_global_bayes.csv"

# rolling window (semanas passadas) para estimar bias on-line
BIAS_WINDOW = 8


def main() -> int:
    if not IN_CSV.exists():
        raise SystemExit(f"Arquivo não encontrado: {IN_CSV}")

    df = pd.read_csv(IN_CSV)
    if df.empty:
        raise SystemExit("CSV de calibração está vazio.")

    # garantir ordem temporal (week string já é ordenável; mas usamos a ordem do arquivo por segurança)
    df = df.copy()
    df["week"] = df["week"].astype(str)

    y = df["pnl_theoretical"].to_numpy(dtype=float)
    mu = df["pred_mean"].to_numpy(dtype=float)
    err = y - mu

    bias = np.zeros_like(err, dtype=float)
    for i in range(err.size):
        if i == 0:
            bias[i] = 0.0
            continue
        lo = max(0, i - int(BIAS_WINDOW))
        hist = err[lo:i]
        hist = hist[np.isfinite(hist)]
        bias[i] = float(np.mean(hist)) if hist.size else 0.0

    mu_adj = mu + bias
    err_adj = y - mu_adj

    out = df.copy()
    out["pred_mean_raw"] = mu
    out["bias_online"] = bias
    out["pred_mean_bias_adj_online"] = mu_adj
    out["diff_real_minus_model_raw"] = err
    out["diff_real_minus_model_bias_adj"] = err_adj
    out["abs_diff_raw"] = np.abs(err)
    out["abs_diff_bias_adj"] = np.abs(err_adj)

    # também ajustar quantis por shift aditivo (mesmo bias_online)
    for q in ["pred_p05", "pred_p10", "pred_p50", "pred_p90", "pred_p95"]:
        if q in out.columns:
            out[f"{q}_bias_adj_online"] = out[q].to_numpy(dtype=float) + bias

    out_path = OUT_DIR / "forecast_calibration_global_bayes_online_bias.csv"
    out.to_csv(out_path, index=False)

    # resumo md
    def _m(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        return float(np.mean(x)) if x.size else float("nan")

    mae_raw = _m(np.abs(err))
    mae_adj = _m(np.abs(err_adj))
    bias_raw = _m(err)
    bias_adj = _m(err_adj)

    lines = []
    lines.append("## Bias-adjusted on-line (walk-forward) — forecast de PnL semanal (global_bayes)\n\n")
    lines.append(f"- Janela de bias (rolling): **{BIAS_WINDOW}** semanas passadas\n\n")
    lines.append("### Métricas (OOS, por semana)\n")
    lines.append(f"- Bias (real - pred), modelo cru: **USD {bias_raw:,.1f}**\n")
    lines.append(f"- Bias (real - pred), bias-adjusted on-line: **USD {bias_adj:,.1f}**\n")
    lines.append(f"- MAE, modelo cru: **USD {mae_raw:,.1f}**\n")
    lines.append(f"- MAE, bias-adjusted on-line: **USD {mae_adj:,.1f}**\n\n")
    lines.append("### Arquivos\n")
    lines.append(f"- CSV: `analysis_proba_raw/pro_portfolio_all/{out_path.name}`\n")
    (OUT_DIR / "forecast_calibration_global_bayes_online_bias.md").write_text("".join(lines), encoding="utf-8")

    print(str(out_path))
    print(str(OUT_DIR / "forecast_calibration_global_bayes_online_bias.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

