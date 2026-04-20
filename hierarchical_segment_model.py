#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelo hierárquico Bayesiano (partial pooling) para estimar edge por segmento.

Contexto:
- Usamos as séries OOS do walk-forward já geradas em:
    analysis_proba_raw/pro_portfolio_all/oos_walkforward_<mode>_weekly_by_segment.csv
  onde cada linha contém o resultado OOS daquela semana para um segmento (rule_key=FT|terça-feira etc).

Objetivo:
- Estimar, para cada segmento s, o ROI médio por dólar (cap2) no OOS: mu_s
- Aplicar shrinkage (partial pooling) entre segmentos para reduzir overfit em segmentos com poucos dados.

Modelo (hierárquico, com pesos):
  y_{s,t} = ROI_on_stake_{s,t} ~ Normal(mu_s, sigma^2 / w_{s,t})
  mu_s ~ Normal(mu0, tau^2)
  sigma^2 ~ InvGamma(a_sig, b_sig)
  tau^2 ~ InvGamma(a_tau, b_tau)
  mu0 ~ Normal(m0, v0)

Onde:
- w_{s,t} = n_bets_{s,t} (peso por informação; mais apostas => menor variância do ROI semanal).

Saídas:
- analysis_proba_raw/pro_portfolio_all/hierarchical_<mode>_segment_posterior.csv
- analysis_proba_raw/pro_portfolio_all/hierarchical_<mode>_segment_report.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


BASE = Path("/workspace/analysis_proba_raw/pro_portfolio_all")

# defaults
MODE = "global_classic"  # ou global_bayes, segment_classic, segment_bayes

# MCMC
SEED = 7
BURN = 2000
SAMPLES = 8000
THIN = 1

# Priors (fracas / conservadoras)
m0 = 0.0
v0 = 1.0  # prior sd=1 (ROI semanal por $ geralmente muito menor que 1)
a_sig = 2.0
b_sig = 0.10
a_tau = 2.0
b_tau = 0.05


def parse_mode(argv: List[str]) -> str:
    if len(argv) >= 2 and argv[1].strip():
        return argv[1].strip()
    return MODE


def invgamma_sample(rng: np.random.Generator, a: float, b: float) -> float:
    # If X ~ InvGamma(a,b), then 1/X ~ Gamma(a, rate=b)
    g = rng.gamma(shape=a, scale=1.0 / b)
    return float(1.0 / g) if g > 0 else 1e-9


@dataclass
class PosteriorSummary:
    rule_key: str
    n_weeks: int
    sum_w: float
    mean_roi_obs: float
    post_mean: float
    ci05: float
    ci50: float
    ci95: float
    p_pos: float


def main(argv: List[str]) -> int:
    mode = parse_mode(argv)
    path = BASE / f"oos_walkforward_{mode}_weekly_by_segment.csv"
    if not path.exists():
        raise SystemExit(f"Arquivo não encontrado: {path}")

    df = pd.read_csv(path)
    # required cols: week, rule_key, n_bets, stake_usd, profit_cap2_usd
    for c in ("week", "rule_key", "n_bets", "stake_usd", "profit_cap2_usd"):
        if c not in df.columns:
            raise SystemExit(f"Coluna ausente em {path}: {c}")

    # ROI por semana/segmento (somente quando stake>0)
    df["stake_usd"] = pd.to_numeric(df["stake_usd"], errors="coerce").astype(float)
    df["profit_cap2_usd"] = pd.to_numeric(df["profit_cap2_usd"], errors="coerce").astype(float)
    df["n_bets"] = pd.to_numeric(df["n_bets"], errors="coerce").fillna(0).astype(int)
    df = df[(df["stake_usd"] > 0) & (df["n_bets"] > 0)].copy()
    if df.empty:
        raise SystemExit("Sem linhas com stake>0 e n_bets>0.")

    df["roi"] = df["profit_cap2_usd"] / df["stake_usd"]
    df["w"] = df["n_bets"].astype(float)

    # map segments to integer ids
    segs = sorted(df["rule_key"].unique().tolist())
    S = len(segs)
    seg2i = {s: i for i, s in enumerate(segs)}

    # pack observations
    y = df["roi"].to_numpy(dtype=float)
    w = df["w"].to_numpy(dtype=float)
    sid = df["rule_key"].map(seg2i).to_numpy(dtype=int)
    N = y.size

    # Precompute per-segment index lists
    idxs: List[np.ndarray] = [np.where(sid == i)[0] for i in range(S)]
    sumw = np.array([float(w[ix].sum()) for ix in idxs], dtype=float)
    sumwy = np.array([float((w[ix] * y[ix]).sum()) for ix in idxs], dtype=float)

    # init
    rng = np.random.default_rng(SEED)
    mu0_cur = float(np.mean(y))
    tau2_cur = 0.05
    sig2_cur = 0.10
    mu_s = np.array([float(sumwy[i] / sumw[i]) if sumw[i] > 0 else mu0_cur for i in range(S)], dtype=float)

    draws_mu0 = []
    draws_tau2 = []
    draws_sig2 = []
    draws_mu = np.zeros((SAMPLES // THIN, S), dtype=float)
    keep = 0

    total_iters = BURN + SAMPLES * THIN
    for it in range(total_iters):
        # sample mu_s (conditionally independent)
        for i in range(S):
            prec = (sumw[i] / sig2_cur) + (1.0 / tau2_cur)
            var = 1.0 / prec
            mean = var * ((sumwy[i] / sig2_cur) + (mu0_cur / tau2_cur))
            mu_s[i] = float(rng.normal(loc=mean, scale=np.sqrt(var)))

        # sample mu0
        prec0 = (S / tau2_cur) + (1.0 / v0)
        var0 = 1.0 / prec0
        mean0 = var0 * ((mu_s.sum() / tau2_cur) + (m0 / v0))
        mu0_cur = float(rng.normal(loc=mean0, scale=np.sqrt(var0)))

        # sample tau^2 (InvGamma)
        ss_tau = float(np.sum((mu_s - mu0_cur) ** 2))
        tau2_cur = invgamma_sample(rng, a_tau + S / 2.0, b_tau + 0.5 * ss_tau)

        # sample sigma^2 (InvGamma), weighted residuals
        ss_sig = 0.0
        for i in range(S):
            ix = idxs[i]
            if ix.size == 0:
                continue
            ss_sig += float(np.sum(w[ix] * (y[ix] - mu_s[i]) ** 2))
        sig2_cur = invgamma_sample(rng, a_sig + N / 2.0, b_sig + 0.5 * ss_sig)

        # store
        if it >= BURN and ((it - BURN) % THIN == 0):
            draws_mu[keep, :] = mu_s
            draws_mu0.append(mu0_cur)
            draws_tau2.append(tau2_cur)
            draws_sig2.append(sig2_cur)
            keep += 1

    # summarize per segment
    rows: List[PosteriorSummary] = []
    for s, i in seg2i.items():
        d = draws_mu[:, i]
        ci05, ci50, ci95 = (float(np.quantile(d, q)) for q in (0.05, 0.50, 0.95))
        p_pos = float(np.mean(d > 0))
        # observed mean (weighted)
        g = df[df["rule_key"] == s]
        mean_obs = float(np.average(g["roi"].to_numpy(dtype=float), weights=g["w"].to_numpy(dtype=float)))
        rows.append(
            PosteriorSummary(
                rule_key=s,
                n_weeks=int(g["week"].nunique()),
                sum_w=float(g["w"].sum()),
                mean_roi_obs=mean_obs,
                post_mean=float(d.mean()),
                ci05=ci05,
                ci50=ci50,
                ci95=ci95,
                p_pos=p_pos,
            )
        )

    out = pd.DataFrame([r.__dict__ for r in rows]).sort_values(["p_pos", "post_mean"], ascending=False)
    out_path = BASE / f"hierarchical_{mode}_segment_posterior.csv"
    out.to_csv(out_path, index=False)

    # executive MD
    mu0 = float(np.mean(draws_mu0))
    tau = float(np.sqrt(np.mean(draws_tau2)))
    sig = float(np.sqrt(np.mean(draws_sig2)))
    top = out.head(10)

    md = []
    md.append(f"## Modelo hierárquico (partial pooling) — segmentos — `{mode}`\n")
    md.append(f"- Observações: ROI semanal por segmento (cap2), ponderado por **n_bets**.\n")
    md.append(f"- Posterior: Gibbs (burn={BURN}, draws={keep}).\n")
    md.append(f"- Hyperparams (médias posteriores): mu0={mu0:.4f}, tau={tau:.4f}, sigma={sig:.4f}\n")
    md.append("\n### Top 10 segmentos por P(mu>0)\n")
    for _, r in top.iterrows():
        md.append(
            f"- **{r['rule_key']}**: P(mu>0)={r['p_pos']*100:.1f}%, "
            f"mu_post={r['post_mean']:.4f}, CI90% [{r['ci05']:.4f}..{r['ci95']:.4f}], "
            f"mean_obs={r['mean_roi_obs']:.4f}, n_weeks={int(r['n_weeks'])}, sum_w={r['sum_w']:.0f}\n"
        )

    md.append("\n### Arquivos\n")
    md.append(f"- CSV: `analysis_proba_raw/pro_portfolio_all/{out_path.name}`\n")
    md.append(f"- Fonte: `analysis_proba_raw/pro_portfolio_all/{path.name}`\n")

    md_path = BASE / f"hierarchical_{mode}_segment_report.md"
    md_path.write_text("".join(md), encoding="utf-8")

    print(str(md_path))
    print(str(out_path))
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv))

