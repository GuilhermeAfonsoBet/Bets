#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análise empírica: ROI vs tamanho de stake (sem penalização arbitrária).

Ideia:
1) Para cada segmento ativo do portfólio (DoW x FT/FH), no conjunto selecionado (score>=cutoff),
   comparamos ROI_cap2 entre:
     - apostas "uncapped" (house_cap >= stake0, ou seja, onde conseguiríamos executar o stake cheio)
     - apostas "capped"   (house_cap < stake0, onde seríamos limitados pela casa)
   Reportamos delta = mean(uncapped) - mean(capped) com IC95% (bootstrap) e p-valor (permutação).

2) Também medimos a relação ROI_cap2 vs stake histórico ("Stake Aposta Realizada (USD)")
   dentro do conjunto selecionado, via Spearman (rho) + p-valor (permutação), usando apenas stake_hist>0.

Saídas:
  - /workspace/analysis_proba_raw/pro_portfolio_all/stake_sensitivity.csv
  - /workspace/analysis_proba_raw/pro_portfolio_all/stake_sensitivity.md

Observação:
- Isto NÃO prova causalidade (stake histórico é escolha/execução, não um experimento randomizado),
  mas é um check objetivo para detectar sinais fortes de degradação de ROI em stakes maiores.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd


SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")
PORT = Path("/workspace/analysis_proba_raw/pro_portfolio_all/portfolio_pro_all.json")
OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")

TRAIN_START = pd.Timestamp("2025-10-01")
TRAIN_END = pd.Timestamp("2025-12-31 23:59:59")
OOS_START = pd.Timestamp("2026-01-01")

BANKROLL = 2300.0

N_BOOT = 10_000
N_PERM = 5_000
SEED = 7


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    # rank with average ties
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    rx = rx - np.mean(rx)
    ry = ry - np.mean(ry)
    den = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    return float(np.sum(rx * ry) / den) if den > 0 else float("nan")


def perm_pvalue(x: np.ndarray, y: np.ndarray, stat_fn: Callable[[np.ndarray, np.ndarray], float], n_perm: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 10:
        return float("nan"), float("nan")
    obs = float(stat_fn(x, y))
    if not np.isfinite(obs):
        return obs, float("nan")
    cnt = 0
    for _ in range(int(n_perm)):
        yp = rng.permutation(y)
        s = float(stat_fn(x, yp))
        if np.abs(s) >= np.abs(obs):
            cnt += 1
    p = (cnt + 1.0) / (n_perm + 1.0)
    return obs, float(p)


def bootstrap_delta_mean(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 5 or b.size < 5:
        return float("nan"), float("nan"), float("nan")
    obs = float(np.mean(a) - np.mean(b))
    ia = rng.integers(0, a.size, size=(n_boot, a.size))
    ib = rng.integers(0, b.size, size=(n_boot, b.size))
    boot = np.mean(a[ia], axis=1) - np.mean(b[ib], axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return obs, float(lo), float(hi)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(PORT.read_text(encoding="utf-8"))
    portfolio: Dict[str, Dict[str, Dict]] = cfg["portfolio"]

    df = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])
    if "roi_calc" not in df.columns:
        raise KeyError("Coluna roi_calc ausente. Regerar scored_dedup_proba_raw_all.csv antes de rodar este script.")
    df["roi_raw"] = pd.to_numeric(df["roi_calc"], errors="coerce").astype(float)
    df["roi_cap2"] = np.minimum(df["roi_raw"].to_numpy(dtype=float), 2.0)
    df["house_cap"] = pd.to_numeric(df["house_cap"], errors="coerce").astype(float)
    df["stake_hist"] = pd.to_numeric(df["Stake Aposta Realizada (USD)"], errors="coerce").astype(float)

    def run_period(tag: str, dfx: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for bet_type in ("FT", "FH"):
            for dow, rule in portfolio[bet_type].items():
                frac = float(rule["stake_frac"])
                if frac <= 0:
                    continue
                score_col = str(rule["score_col"])
                cutoff = float(rule["cutoff"])
                stake0 = BANKROLL * frac

                x = dfx[(dfx["dow_pt"] == dow) & (dfx["bet_type"] == bet_type)].copy()
                x["score"] = pd.to_numeric(x[score_col], errors="coerce").astype(float)
                x = x[np.isfinite(x["score"]) & np.isfinite(x["roi_cap2"])].copy()
                x = x[x["score"] >= cutoff].copy()
                if x.empty:
                    continue

                unc = x[x["house_cap"] >= stake0]["roi_cap2"].to_numpy(dtype=float)
                capd = x[x["house_cap"] < stake0]["roi_cap2"].to_numpy(dtype=float)
                delta, d_lo, d_hi = bootstrap_delta_mean(unc, capd, n_boot=N_BOOT, seed=SEED + 13)

                # teste por permutação do delta de médias (se tiver dois grupos)
                p_delta = float("nan")
                if np.isfinite(delta) and unc.size >= 5 and capd.size >= 5:
                    rng = np.random.default_rng(SEED + 17)
                    y = np.concatenate([unc, capd])
                    g = np.concatenate([np.ones_like(unc), np.zeros_like(capd)])
                    obs = float(np.mean(y[g == 1]) - np.mean(y[g == 0]))
                    cnt = 0
                    for _ in range(N_PERM):
                        gp = rng.permutation(g)
                        s = float(np.mean(y[gp == 1]) - np.mean(y[gp == 0]))
                        if abs(s) >= abs(obs):
                            cnt += 1
                    p_delta = (cnt + 1.0) / (N_PERM + 1.0)

                # correlação Spearman ROI vs stake_hist, apenas stake_hist>0
                xh = x[np.isfinite(x["stake_hist"]) & (x["stake_hist"] > 0)].copy()
                rho, p_rho = perm_pvalue(
                    xh["stake_hist"].to_numpy(dtype=float),
                    xh["roi_cap2"].to_numpy(dtype=float),
                    stat_fn=spearman_rho,
                    n_perm=N_PERM,
                    seed=SEED + 19,
                )

                rows.append(
                    {
                        "period": tag,
                        "bet_type": bet_type,
                        "dow_pt": dow,
                        "score_col": score_col,
                        "cutoff": cutoff,
                        "stake_frac": frac,
                        "stake0_usd": stake0,
                        "n_selected": int(len(x)),
                        "n_uncapped": int(np.sum(x["house_cap"] >= stake0)),
                        "n_capped": int(np.sum(x["house_cap"] < stake0)),
                        "mean_roi_cap2": float(np.mean(x["roi_cap2"])),
                        "mean_roi_uncapped": float(np.mean(unc)) if unc.size else float("nan"),
                        "mean_roi_capped": float(np.mean(capd)) if capd.size else float("nan"),
                        "delta_unc_minus_capped": float(delta),
                        "delta_ci95_lo": float(d_lo),
                        "delta_ci95_hi": float(d_hi),
                        "perm_p_delta": float(p_delta),
                        "n_with_stake_hist_gt0": int(len(xh)),
                        "spearman_rho_roi_vs_stake_hist": float(rho),
                        "spearman_perm_p": float(p_rho),
                    }
                )
        return pd.DataFrame(rows)

    train = df[(df["BIA_ApostaUTC"] >= TRAIN_START) & (df["BIA_ApostaUTC"] <= TRAIN_END)].copy()
    oos = df[df["BIA_ApostaUTC"] >= OOS_START].copy()

    res = pd.concat([run_period("train", train), run_period("oos", oos)], ignore_index=True)
    res.to_csv(OUT_DIR / "stake_sensitivity.csv", index=False)

    # Markdown curto e “executivo”
    lines = []
    lines.append("## Stake sensitivity (empírico)\n")
    lines.append(
        "- Métrica de ROI usada: **ROI_cap2** (ROI Real capado em 2.0).\n"
        "- Comparação principal: **uncapped vs capped** para o stake0 do próprio segmento.\n"
        "- Relação adicional: Spearman(ROI_cap2, stake_hist) no conjunto selecionado com stake_hist>0.\n"
        "\n"
    )

    if res.empty:
        lines.append("_Sem dados suficientes para análise._\n")
    else:
        for tag in ("train", "oos"):
            r = res[res["period"] == tag].copy()
            lines.append(f"### Período: **{tag}**\n")
            if r.empty:
                lines.append("_Sem linhas._\n")
                continue
            for _, row in r.sort_values(["bet_type", "dow_pt"]).iterrows():
                bt = row["bet_type"]
                dow = row["dow_pt"]
                frac = row["stake_frac"]
                stake0 = row["stake0_usd"]
                n = int(row["n_selected"])
                nu = int(row["n_uncapped"])
                nc = int(row["n_capped"])
                mu = row["mean_roi_uncapped"]
                mc = row["mean_roi_capped"]
                d = row["delta_unc_minus_capped"]
                dlo = row["delta_ci95_lo"]
                dhi = row["delta_ci95_hi"]
                pdel = row["perm_p_delta"]
                rho = row["spearman_rho_roi_vs_stake_hist"]
                prho = row["spearman_perm_p"]
                nst = int(row["n_with_stake_hist_gt0"])

                lines.append(
                    f"- **{bt} | {dow}** — stake0 **USD {stake0:.0f}** ({frac*100:.1f}%), n={n} "
                    f"(uncapped={nu}, capped={nc}). "
                    f"ROI_cap2 uncapped={mu:.3f}, capped={mc:.3f}; "
                    f"Δ={d:.3f} (IC95% {dlo:.3f}..{dhi:.3f}), p_perm={pdel:.3f}. "
                    f"Spearman(ROI, stake_hist>0): rho={rho:.3f} (p_perm={prho:.3f}, n={nst}).\n"
                )

            lines.append("\n")

    (OUT_DIR / "stake_sensitivity.md").write_text("".join(lines), encoding="utf-8")
    print(str(OUT_DIR / "stake_sensitivity.md"))
    print(str(OUT_DIR / "stake_sensitivity.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

