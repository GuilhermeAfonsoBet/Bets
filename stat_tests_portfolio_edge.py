#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testes estatísticos formais para:
1) evidência de edge (banca 2.3k) no PnL semanal OOS (cap2)
2) qualidade do modelo (bias-adjusted on-line) vs realizado no cenário max (house_cap)
3) até que banca (grid) o edge permanece estatisticamente suportado
4) hipótese de ROI maior em stakes menores (ROI vs house_cap)

Saídas:
- analysis_proba_raw/pro_portfolio_all/stat_tests_portfolio_edge.md
- analysis_proba_raw/pro_portfolio_all/stat_tests_bankroll_scaling.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")
MODE = "global_bayes_roll12_robust_p10_p70"

WF_WEEKLY = OUT_DIR / f"oos_walkforward_{MODE}_weekly.csv"
WF_RULES = OUT_DIR / f"oos_walkforward_{MODE}_selected_rules.csv"
SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")
FC_MAX = OUT_DIR / f"forecast_calibration_{MODE}_max.csv"

BANKROLL_BASE = 2300.0


def _rng(seed: int = 7) -> np.random.Generator:
    return np.random.default_rng(seed)


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 20000, seed: int = 7) -> Tuple[float, float, float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    r = _rng(seed)
    idx = r.integers(0, x.size, size=(n_boot, x.size))
    means = np.mean(x[idx], axis=1)
    # também retornamos P_boot(mean>0)
    return float(np.mean(x)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), float(np.mean(means > 0))


def sign_test_pos(x: np.ndarray) -> Tuple[int, int, float]:
    """
    Teste de sinais unilateral para H0: P(x>0) <= 0.5 vs H1: P(x>0) > 0.5
    Retorna (k_pos, n_nonzero, p_value_exact).
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    # remove zeros (ties)
    nz = x[x != 0]
    n = int(nz.size)
    if n == 0:
        return 0, 0, float("nan")
    k = int(np.sum(nz > 0))
    # p = P[X >= k] where X~Bin(n,0.5)
    # exact sum
    from math import comb

    p = 0.0
    for j in range(k, n + 1):
        p += comb(n, j) * (0.5 ** n)
    return k, n, float(p)


def signflip_permutation_pvalue_mean(x: np.ndarray, n_perm: int = 50000, seed: int = 7) -> Tuple[float, float]:
    """
    Permutação por flip de sinais (teste unilateral) para H0: mean<=0 vs H1: mean>0,
    assumindo simetria sob H0.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    obs = float(np.mean(x))
    r = _rng(seed)
    cnt = 0
    for _ in range(int(n_perm)):
        s = r.choice([-1.0, 1.0], size=x.size)
        m = float(np.mean(x * s))
        if m >= obs:
            cnt += 1
    p = (cnt + 1.0) / (n_perm + 1.0)
    return obs, float(p)


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    # pandas>=3 pode devolver arrays read-only; evitar operações in-place
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float, copy=True)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float, copy=True)
    rx = rx - float(np.mean(rx))
    ry = ry - float(np.mean(ry))
    den = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    return float(np.sum(rx * ry) / den) if den > 0 else float("nan")


def perm_pvalue_stat(x: np.ndarray, y: np.ndarray, stat_fn, n_perm: int = 20000, seed: int = 7, two_sided: bool = True) -> Tuple[float, float]:
    r = _rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 20:
        return float("nan"), float("nan")
    obs = float(stat_fn(x, y))
    cnt = 0
    for _ in range(int(n_perm)):
        yp = r.permutation(y)
        s = float(stat_fn(x, yp))
        if two_sided:
            if abs(s) >= abs(obs):
                cnt += 1
        else:
            if s <= obs:
                cnt += 1
    p = (cnt + 1.0) / (n_perm + 1.0)
    return obs, float(p)


def online_bias_adjust(mu: np.ndarray, y: np.ndarray, window: int = 8) -> np.ndarray:
    mu = np.asarray(mu, float)
    y = np.asarray(y, float)
    err = y - mu
    bias = np.zeros_like(err, float)
    for i in range(err.size):
        if i == 0:
            bias[i] = 0.0
            continue
        lo = max(0, i - int(window))
        hist = err[lo:i]
        hist = hist[np.isfinite(hist)]
        bias[i] = float(np.mean(hist)) if hist.size else 0.0
    return mu + bias


def _ensure_calibrated_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garante proba_cal_segqui e proba_cal_sexdom para aplicar regras.
    """
    import json

    df = df.copy()
    floor = 0.005
    if "proba_cal_segqui" not in df.columns:
        df["proba_cal_segqui"] = np.nan
    if "proba_cal_sexdom" not in df.columns:
        df["proba_cal_sexdom"] = np.nan
    # segqui
    calib_segqui = Path("/workspace/clv_calib_SegQui.json")
    if "proba_raw_segqui" in df.columns and calib_segqui.exists():
        obj = json.loads(calib_segqui.read_text(encoding="utf-8"))
        x = np.asarray(obj["isotonic"]["x"], float)
        y = np.asarray(obj["isotonic"]["y"], float)
        p = pd.to_numeric(df["proba_raw_segqui"], errors="coerce").to_numpy(float)
        out = np.interp(p, x, y, left=float(y[0]), right=float(y[-1]))
        out = np.maximum(out, floor)
        df["proba_cal_segqui"] = np.clip(out, 0.0, 1.0)
    # sexdom
    calib_sexdom = Path("/workspace/clv_calib_SexDom.json")
    if "proba_raw_sexdom" in df.columns and calib_sexdom.exists():
        obj = json.loads(calib_sexdom.read_text(encoding="utf-8"))
        x = np.asarray(obj["isotonic"]["x"], float)
        y = np.asarray(obj["isotonic"]["y"], float)
        p = pd.to_numeric(df["proba_raw_sexdom"], errors="coerce").to_numpy(float)
        out = np.interp(p, x, y, left=float(y[0]), right=float(y[-1]))
        out = np.maximum(out, floor)
        df["proba_cal_sexdom"] = np.clip(out, 0.0, 1.0)
    return df


def apply_rules_week_scaled(df_week: pd.DataFrame, rules_week: pd.DataFrame, bankroll: float, use_max: bool) -> pd.DataFrame:
    rows = []
    alpha = float(rules_week["alpha_effective"].iloc[0]) if "alpha_effective" in rules_week.columns and np.isfinite(float(rules_week["alpha_effective"].iloc[0])) else float(rules_week["alpha_global"].iloc[0])
    for _, r in rules_week.iterrows():
        if str(r.get("status")) != "ok":
            continue
        bt = str(r["bet_type"])
        dow = str(r["dow_pt"])
        sc = str(r["score_col"])
        cutoff = float(r["cutoff"])
        frac = float(r["stake_frac"])
        if frac <= 0:
            continue
        x = df_week[(df_week["bet_type"] == bt) & (df_week["dow_pt"] == dow)].copy()
        if x.empty or sc not in x.columns:
            continue
        score = pd.to_numeric(x[sc], errors="coerce").to_numpy(float)
        roi2 = x["roi_cap2"].to_numpy(float)
        cap = x["house_cap"].to_numpy(float)
        m = np.isfinite(score) & (score >= cutoff) & np.isfinite(roi2) & np.isfinite(cap) & (cap > 0)
        if not np.any(m):
            continue
        cap_sel = cap[m]
        roi_sel = roi2[m]
        stake0 = float(bankroll) * frac * float(alpha)
        stake_eff = cap_sel if use_max else np.minimum(stake0, cap_sel)
        profit = stake_eff * roi_sel
        rows.append(pd.DataFrame({"stake_eff": stake_eff, "profit_cap2": profit, "house_cap": cap_sel, "roi_cap2": roi_sel}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["stake_eff", "profit_cap2", "house_cap", "roi_cap2"])


def bankroll_curve(df: pd.DataFrame, rules: pd.DataFrame, weeks: List[str], bankroll_grid: np.ndarray) -> pd.DataFrame:
    out_rows = []
    for b in bankroll_grid:
        wk_rows = []
        for wk in weeks:
            rw = rules[rules["test_week"].astype(str) == wk].copy()
            if rw.empty:
                continue
            xw = df[df["week"] == wk].copy()
            bets = apply_rules_week_scaled(xw, rw, bankroll=float(b), use_max=False)
            stake = float(bets["stake_eff"].sum()) if len(bets) else 0.0
            pnl = float(bets["profit_cap2"].sum()) if len(bets) else 0.0
            wk_rows.append((stake, pnl))
        if not wk_rows:
            continue
        stake = np.array([s for s, _ in wk_rows], float)
        pnl = np.array([p for _, p in wk_rows], float)
        mean, lo, hi, p_boot = bootstrap_mean_ci(pnl, n_boot=20000, seed=7)
        _, p_perm = signflip_permutation_pvalue_mean(pnl, n_perm=50000, seed=7)
        k, n, p_sign = sign_test_pos(pnl)
        out_rows.append(
            {
                "bankroll": float(b),
                "weeks": int(len(pnl)),
                "weeks_with_stake": int(np.sum(stake > 0)),
                "stake_total": float(stake.sum()),
                "pnl_total": float(pnl.sum()),
                "roi_on_stake": float(pnl.sum() / stake.sum()) if stake.sum() > 0 else float("nan"),
                "mean_week": float(np.mean(pnl)),
                "boot_ci95_lo": float(lo),
                "boot_ci95_hi": float(hi),
                "p_perm_mean_gt0": float(p_perm),
                "p_boot_mean_gt0": float(p_boot),
                "sign_k_pos": int(k),
                "sign_n": int(n),
                "p_sign_pos": float(p_sign),
            }
        )
    return pd.DataFrame(out_rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wf_week = pd.read_csv(WF_WEEKLY)
    rules = pd.read_csv(WF_RULES)

    w = wf_week["profit_cap2_usd"].to_numpy(float)
    stake = wf_week["stake_usd"].to_numpy(float)

    # Edge tests (inclui semanas sem trade = 0)
    mean_w, lo_w, hi_w, pboot_w = bootstrap_mean_ci(w, n_boot=50000, seed=7)
    obs_w, p_perm_w = signflip_permutation_pvalue_mean(w, n_perm=80000, seed=7)
    k_w, n_w, p_sign_w = sign_test_pos(w)

    # Condicional: apenas semanas com stake>0
    w_tr = wf_week.loc[wf_week["stake_usd"] > 0, "profit_cap2_usd"].to_numpy(float)
    mean_tr, lo_tr, hi_tr, pboot_tr = bootstrap_mean_ci(w_tr, n_boot=50000, seed=7)
    obs_tr, p_perm_tr = signflip_permutation_pvalue_mean(w_tr, n_perm=80000, seed=7)
    k_tr, n_tr, p_sign_tr = sign_test_pos(w_tr)

    # Modelo max: online bias-adjusted vs realizado
    fcmax = pd.read_csv(FC_MAX)
    y_max = fcmax["pnl_max_theoretical"].to_numpy(float)
    mu_max = fcmax["pred_mean"].to_numpy(float)
    mu_max_online = online_bias_adjust(mu_max, y_max, window=8)
    err_max_online = y_max - mu_max_online
    mean_e, lo_e, hi_e, pboot_e = bootstrap_mean_ci(err_max_online, n_boot=50000, seed=7)
    # aqui testamos se erro médio é <0 (viés otimista); usamos signflip em -erro para H1: mean(-err)>0
    _, p_perm_e = signflip_permutation_pvalue_mean(-err_max_online, n_perm=80000, seed=7)
    k_e, n_e, p_sign_e = sign_test_pos(-err_max_online)  # “sucessos” = err<0

    # ROI vs escala (max-selected bets): Spearman(roi_cap2, house_cap)
    df = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])
    df["week"] = pd.to_datetime(df["BIA_ApostaUTC"]).dt.to_period("W-SUN").astype(str)
    df["roi_raw"] = pd.to_numeric(df["roi_calc"], errors="coerce").to_numpy(float)
    df["roi_cap2"] = np.minimum(df["roi_raw"].to_numpy(float), 2.0)
    df["house_cap"] = pd.to_numeric(df["house_cap"], errors="coerce").to_numpy(float)
    df = _ensure_calibrated_cols(df)

    weeks = wf_week["week"].astype(str).tolist()
    rows = []
    for wk in weeks:
        rw = rules[rules["test_week"].astype(str) == wk].copy()
        xw = df[df["week"] == wk].copy()
        if rw.empty or xw.empty:
            continue
        bets = apply_rules_week_scaled(xw, rw, bankroll=BANKROLL_BASE, use_max=True)  # max-selected bets
        if len(bets):
            rows.append(bets[["house_cap", "roi_cap2"]])
    bets_all = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["house_cap", "roi_cap2"])
    rho, p_rho = perm_pvalue_stat(np.log1p(bets_all["house_cap"].to_numpy(float)), bets_all["roi_cap2"].to_numpy(float), stat_fn=spearman_rho, n_perm=40000, seed=7, two_sided=True)

    # bankroll threshold: curva de significância (perm p<0.05)
    bankroll_grid = np.array([2300, 5000, 10000, 20000, 30000, 40000, 50000, 63205], float)
    curve = bankroll_curve(df=df, rules=rules, weeks=weeks, bankroll_grid=bankroll_grid)
    curve_path = OUT_DIR / "stat_tests_bankroll_scaling.csv"
    curve.to_csv(curve_path, index=False)

    # Sensibilidade: excluir a última semana OOS (muito útil quando o relatório é gerado com semana corrente parcial)
    curve_excl_path = OUT_DIR / "stat_tests_bankroll_scaling_excl_lastweek.csv"
    if len(weeks) >= 2:
        weeks_excl = weeks[:-1]
        curve_excl = bankroll_curve(df=df, rules=rules, weeks=weeks_excl, bankroll_grid=bankroll_grid)
        curve_excl.to_csv(curve_excl_path, index=False)

    # decide max bankroll “estatisticamente confiável” sob critério p_perm_mean_gt0<0.05
    b_ok = curve[np.isfinite(curve["p_perm_mean_gt0"]) & (curve["p_perm_mean_gt0"] < 0.05)]
    b_max_ok = float(b_ok["bankroll"].max()) if not b_ok.empty else float("nan")

    # markdown
    md = []
    md.append(f"## Testes estatísticos — edge e escala (MODE={MODE})\n\n")
    md.append("### 1) Evidência formal de edge (banca 2.3k, PnL semanal OOS cap2)\n")
    md.append(f"- Semanas (inclui semanas sem trade=0): n={len(w)}\n")
    md.append(f"  - mean={mean_w:,.1f}, IC95%(bootstrap)=[{lo_w:,.1f}, {hi_w:,.1f}]\n")
    md.append(f"  - p_perm(mean>0) (sign-flip)={p_perm_w:.4f}\n")
    md.append(f"  - sign-test: k_pos={k_w}/{n_w}, p={p_sign_w:.4f}\n\n")
    md.append(f"- Apenas semanas com trade (stake>0): n={len(w_tr)}\n")
    md.append(f"  - mean={mean_tr:,.1f}, IC95%(bootstrap)=[{lo_tr:,.1f}, {hi_tr:,.1f}]\n")
    md.append(f"  - p_perm(mean>0) (sign-flip)={p_perm_tr:.4f}\n")
    md.append(f"  - sign-test: k_pos={k_tr}/{n_tr}, p={p_sign_tr:.4f}\n\n")

    md.append("### 2) Max (house_cap): variância vs modelo inadequado (online bias-adjusted)\n")
    md.append("Teste formal aqui é sobre o **erro** (y - mu_online) no cenário max.\n")
    md.append(f"- mean(error_online)={mean_e:,.1f}, IC95%=[{lo_e:,.1f}, {hi_e:,.1f}]\n")
    md.append(f"- p_perm(mean_error<0) (sign-flip em -erro)={p_perm_e:.4f}\n")
    md.append(f"- sign-test (erro<0): k={k_e}/{n_e}, p={p_sign_e:.4f}\n\n")

    md.append("### 3) Até que banca o edge permanece estatisticamente suportado?\n")
    md.append(f"- Critério (simples): p_perm(mean_week>0)<0.05 na curva de banca.\n")
    md.append(f"- Maior banca que passou no grid: {b_max_ok if np.isfinite(b_max_ok) else '—'}\n\n")
    md.append(f"CSV: `{curve_path.name}`\n\n")
    if curve_excl_path.exists():
        md.append(f"CSV (sensibilidade, exclui última semana): `{curve_excl_path.name}`\n\n")

    md.append("### 4) ROI maior em stakes menores (teste formal)\n")
    md.append("Usamos apostas selecionadas no cenário max e testamos associação monotônica entre ROI_cap2 e log(1+house_cap).\n")
    md.append(f"- Spearman rho={rho:.3f}, p_perm(two-sided)={p_rho:.4f}\n\n")

    out_md = OUT_DIR / "stat_tests_portfolio_edge.md"
    out_md.write_text("".join(md), encoding="utf-8")
    print(str(out_md))
    print(str(curve_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

