#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testes estatísticos formais para o "ponto de virada" ao escalar banca.

Objetivos:
1) Para cada banca B, testar H0: E[PnL_sem(B)] <= 0 vs H1: >0
   - IC95% bootstrap do mean semanal
   - p_perm (sign-flip) para mean>0
   - P_boot(mean>0)

2) Testes pareados entre bancas (mesmas semanas):
   - delta_t = PnL_t(B2) - PnL_t(B1)
   - IC95% bootstrap de mean(delta)
   - p_perm(sign-flip) para mean(delta)>0

3) Estimar o breakpoint (banca onde mean cruza 0) por bootstrap:
   - para cada bootstrap (resample semanas), calcula mean(B) no grid e interpola onde cruza 0.

Entrada:
- scored_dedup_proba_raw_all.csv (roi_calc + house_cap + scores)
- oos_walkforward_{MODE}_selected_rules.csv (cutoff, stake_frac, alpha_effective por semana/segmento)
- oos_walkforward_{MODE}_weekly.csv (lista de semanas OOS)

Saídas:
- analysis_proba_raw/pro_portfolio_all/stat_tests_scaling_breakpoint.md
- analysis_proba_raw/pro_portfolio_all/stat_tests_scaling_breakpoint.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import evaluate_oos_walkforward_strategy as wf


OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")
MODE = "global_bayes_roll12_robust_p10_p70"
SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")
RULES = OUT_DIR / f"oos_walkforward_{MODE}_selected_rules.csv"
WEEKLY = OUT_DIR / f"oos_walkforward_{MODE}_weekly.csv"

# bancos a testar (grid grosso)
BANKROLLS = np.array([2300, 5000, 10000, 20000, 30000, 40000, 50000, 63205], dtype=float)

N_BOOT = 50000
N_PERM = 80000
SEED = 7


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def bootstrap_mean_ci(x: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float, float, float]:
    """
    Retorna (mean, lo95, hi95, p_boot_pos) onde p_boot_pos = P_boot(mean>0).
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    r = _rng(seed)
    idx = r.integers(0, x.size, size=(n_boot, x.size))
    means = np.mean(x[idx], axis=1)
    return float(np.mean(x)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), float(np.mean(means > 0))


def signflip_perm_p_mean_gt0(x: np.ndarray, n_perm: int, seed: int) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    obs = float(np.mean(x))
    r = _rng(seed)
    cnt = 0
    for _ in range(int(n_perm)):
        s = r.choice([-1.0, 1.0], size=x.size)
        m = float(np.mean(x * s))
        if m >= obs:
            cnt += 1
    return float((cnt + 1.0) / (n_perm + 1.0))


def _ensure_calibrated_cols(df: pd.DataFrame) -> pd.DataFrame:
    import json

    df = df.copy()
    floor = 0.005
    if "proba_cal_segqui" not in df.columns:
        df["proba_cal_segqui"] = np.nan
    if "proba_cal_sexdom" not in df.columns:
        df["proba_cal_sexdom"] = np.nan
    if "proba_raw_segqui" in df.columns and wf.CALIB_SEGQUI.exists():
        obj = json.loads(wf.CALIB_SEGQUI.read_text(encoding="utf-8"))
        x = np.asarray(obj["isotonic"]["x"], float)
        y = np.asarray(obj["isotonic"]["y"], float)
        p = pd.to_numeric(df["proba_raw_segqui"], errors="coerce").to_numpy(float)
        out = np.interp(p, x, y, left=float(y[0]), right=float(y[-1]))
        out = np.maximum(out, floor)
        df["proba_cal_segqui"] = np.clip(out, 0.0, 1.0)
    if "proba_raw_sexdom" in df.columns and wf.CALIB_SEXDOM.exists():
        obj = json.loads(wf.CALIB_SEXDOM.read_text(encoding="utf-8"))
        x = np.asarray(obj["isotonic"]["x"], float)
        y = np.asarray(obj["isotonic"]["y"], float)
        p = pd.to_numeric(df["proba_raw_sexdom"], errors="coerce").to_numpy(float)
        out = np.interp(p, x, y, left=float(y[0]), right=float(y[-1]))
        out = np.maximum(out, floor)
        df["proba_cal_sexdom"] = np.clip(out, 0.0, 1.0)
    return df


def pnl_week_for_bankroll(df_week: pd.DataFrame, rules_week: pd.DataFrame, bankroll: float) -> float:
    """
    Aplica regras da semana usando stake_eff = min(bankroll*stake_frac*alpha_effective, house_cap).
    Retorna PnL cap2 da semana.
    """
    alpha = float(rules_week["alpha_effective"].iloc[0]) if "alpha_effective" in rules_week.columns and np.isfinite(float(rules_week["alpha_effective"].iloc[0])) else float(rules_week["alpha_global"].iloc[0])
    pnl = 0.0
    for _, r in rules_week.iterrows():
        if str(r.get("status")) != "ok":
            continue
        frac = float(r.get("stake_frac", 0.0))
        if frac <= 0:
            continue
        bt = str(r["bet_type"])
        dow = str(r["dow_pt"])
        sc = str(r["score_col"])
        cutoff = float(r["cutoff"])
        x = df_week[(df_week["bet_type"] == bt) & (df_week["dow_pt"] == dow)].copy()
        if x.empty or sc not in x.columns:
            continue
        score = pd.to_numeric(x[sc], errors="coerce").to_numpy(float)
        roi2 = x["roi_cap2"].to_numpy(float)
        cap = x["house_cap"].to_numpy(float)
        m = np.isfinite(score) & (score >= cutoff) & np.isfinite(roi2) & np.isfinite(cap) & (cap > 0)
        if not np.any(m):
            continue
        stake0 = float(bankroll) * frac * float(alpha)
        stake_eff = np.minimum(stake0, cap[m])
        pnl += float(np.sum(stake_eff * roi2[m]))
    return float(pnl)


def breakpoint_from_means(B: np.ndarray, m: np.ndarray) -> float:
    """
    Dado grid B crescente e mean m(B), retorna B* onde cruza 0 (interpolação linear).
    Se não cruza, retorna NaN.
    """
    B = np.asarray(B, float)
    m = np.asarray(m, float)
    if not (np.all(np.isfinite(B)) and np.all(np.isfinite(m))):
        return float("nan")
    for i in range(len(B) - 1):
        if (m[i] >= 0 and m[i + 1] <= 0) or (m[i] <= 0 and m[i + 1] >= 0):
            if m[i + 1] == m[i]:
                return float(B[i])
            t = (0.0 - m[i]) / (m[i + 1] - m[i])
            return float(B[i] + t * (B[i + 1] - B[i]))
    return float("nan")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])
    df["week"] = wf.week_key(df["BIA_ApostaUTC"]).astype(str)
    df["roi_raw"] = pd.to_numeric(df["roi_calc"], errors="coerce").to_numpy(float)
    df["roi_cap2"] = np.minimum(df["roi_raw"].to_numpy(float), 2.0)
    df["house_cap"] = pd.to_numeric(df["house_cap"], errors="coerce").to_numpy(float)
    df = _ensure_calibrated_cols(df)

    rules = pd.read_csv(RULES)
    weeks = pd.read_csv(WEEKLY)["week"].astype(str).tolist()

    # build pnl matrix: weeks x bankrolls
    pnl_mat = np.zeros((len(weeks), len(BANKROLLS)), float)
    for wi, wk in enumerate(weeks):
        rw = rules[rules["test_week"].astype(str) == wk].copy()
        xw = df[df["week"] == wk].copy()
        for bi, b in enumerate(BANKROLLS):
            pnl_mat[wi, bi] = pnl_week_for_bankroll(xw, rw, bankroll=float(b))

    # per-bankroll stats
    rows = []
    for bi, b in enumerate(BANKROLLS):
        pnl = pnl_mat[:, bi]
        mean, lo, hi, pboot = bootstrap_mean_ci(pnl, n_boot=N_BOOT, seed=SEED + 10 + bi)
        pperm = signflip_perm_p_mean_gt0(pnl, n_perm=N_PERM, seed=SEED + 100 + bi)
        rows.append({"bankroll": float(b), "mean_week": mean, "ci95_lo": lo, "ci95_hi": hi, "p_boot_mean_gt0": pboot, "p_perm_mean_gt0": pperm})

    # paired deltas vs 2.3k
    base = pnl_mat[:, 0]
    for bi in range(1, len(BANKROLLS)):
        b = BANKROLLS[bi]
        delta = pnl_mat[:, bi] - base
        mean, lo, hi, pboot = bootstrap_mean_ci(delta, n_boot=N_BOOT, seed=SEED + 200 + bi)
        pperm = signflip_perm_p_mean_gt0(delta, n_perm=N_PERM, seed=SEED + 300 + bi)
        rows.append({"bankroll": float(b), "compare_to": float(BANKROLLS[0]), "delta_mean_week": mean, "delta_ci95_lo": lo, "delta_ci95_hi": hi, "delta_p_boot_gt0": pboot, "delta_p_perm_gt0": pperm})

    out_csv = OUT_DIR / "stat_tests_scaling_breakpoint.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # breakpoint bootstrap
    r = _rng(SEED + 999)
    bp = np.full(N_BOOT, np.nan, float)
    for k in range(N_BOOT):
        idx = r.integers(0, len(weeks), size=len(weeks))
        means = np.mean(pnl_mat[idx, :], axis=0)
        bp[k] = breakpoint_from_means(BANKROLLS, means)
    bp_ok = bp[np.isfinite(bp)]
    bp_p10 = float(np.quantile(bp_ok, 0.10)) if bp_ok.size else float("nan")
    bp_p50 = float(np.quantile(bp_ok, 0.50)) if bp_ok.size else float("nan")
    bp_p90 = float(np.quantile(bp_ok, 0.90)) if bp_ok.size else float("nan")

    md = []
    md.append(f"## Scaling: testes formais (MODE={MODE})\\n\\n")
    md.append("### Por que 'operacional' vs 'estatístico'?\\n")
    md.append("- 'Operacional': baseado no **ponto estimado** (mean) no grid, sem rejeição formal de H0.\\n")
    md.append("- 'Estatístico': exige IC95% do mean > 0 (ou p<0.05).\\n\\n")
    md.append("### 1) Teste de E[PnL_sem(B)]>0 por banca\\n")
    for bi, b in enumerate(BANKROLLS):
        pnl = pnl_mat[:, bi]
        mean, lo, hi, pboot = bootstrap_mean_ci(pnl, n_boot=20000, seed=SEED + 10 + bi)
        md.append(f"- B={b:,.0f}: mean={mean:,.1f}, IC95%=[{lo:,.1f},{hi:,.1f}], p_perm={signflip_perm_p_mean_gt0(pnl, n_perm=20000, seed=SEED+100+bi):.3f}, P_boot(mean>0)={pboot:.3f}\\n")
    md.append("\\n")
    md.append("### 2) Testes pareados: Δ(B)-Δ(2.3k)\\n")
    for bi in range(1, len(BANKROLLS)):
        delta = pnl_mat[:, bi] - pnl_mat[:, 0]
        mean, lo, hi, pboot = bootstrap_mean_ci(delta, n_boot=20000, seed=SEED + 200 + bi)
        md.append(f"- Δ({BANKROLLS[bi]:,.0f} - 2,300): mean={mean:,.1f}, IC95%=[{lo:,.1f},{hi:,.1f}], p_perm={signflip_perm_p_mean_gt0(delta, n_perm=20000, seed=SEED+300+bi):.3f}\\n")
    md.append("\\n")
    md.append("### 3) Breakpoint (banca onde mean cruza 0) via bootstrap\\n")
    md.append(f"- p10/p50/p90 do breakpoint: {bp_p10:,.0f} / {bp_p50:,.0f} / {bp_p90:,.0f} (NaN se não cruzar em muitos bootstraps)\\n")

    out_md = OUT_DIR / "stat_tests_scaling_breakpoint.md"
    out_md.write_text("".join(md), encoding="utf-8")
    print(str(out_md))
    print(str(out_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

