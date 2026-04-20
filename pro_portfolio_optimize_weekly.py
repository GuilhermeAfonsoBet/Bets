#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Versão "mesa profissional" do portfólio:
- Score operacional (proba_raw)
- Legs: Seg/Ter/Qua (modelos separados) + Qui (SegQui)

Objetivo:
  Selecionar (por dia) um cutoff e stake_frac (<= 7%) que maximizem um
  objetivo conservador, sob stress tests e walk-forward semanal.

Abordagem:
  Para cada perna (dia):
    - grid de (cutoff, stake_frac)
    - calcula série semanal de PnL no treino (2025-10..12) em 3 cenários:
        raw, cap2 (ROI capado em 2.0), cap1 (ROI capado em 1.0)
    - calcula estabilidade:
        mean/std/p_neg semanal; bootstrap 52w VaR/CVaR; drawdown p95 (paths)
    - calcula walk-forward semanal (expanding window) no treino, em cap2/cap1:
        mean/std/p_neg/p05
    - aplica constraints "mesa profissional"
  Se uma perna não tiver candidato viável, ela é removida (stake=0, cutoff=1.0).

Outputs:
  /workspace/analysis_proba_raw/pro_portfolio/portfolio_pro.json
  /workspace/analysis_proba_raw/pro_portfolio/report_pro.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw.csv")
OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio")

TRAIN_START = pd.Timestamp("2025-10-01")
TRAIN_END = pd.Timestamp("2025-12-31 23:59:59")
OOS_START = pd.Timestamp("2026-01-01")

BANKROLL = 2300.0
MAX_FRAC = 0.07

N_BOOT = 50_000
N_BOOT_DD = 10_000
SEED = 7


def is_ft(x: str) -> bool:
    return "first half" not in str(x).lower()


def safe_cap(x) -> float:
    try:
        v = float(x)
    except Exception:
        return float("inf")
    if not np.isfinite(v) or v <= 0:
        return float("inf")
    return v


def week_key(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts).dt.to_period("W-SUN").astype(str)


def roi_mode_arr(roi: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return roi
    if mode == "cap2":
        return np.minimum(roi, 2.0)
    if mode == "cap1":
        return np.minimum(roi, 1.0)
    raise ValueError(mode)


def bootstrap_sum(w: np.ndarray, H: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(w), size=(N_BOOT, H))
    sums = w[idx].sum(axis=1)
    var05 = float(np.quantile(sums, 0.05))
    cvar05 = float(np.mean(sums[sums <= var05]))
    return var05, cvar05


def bootstrap_dd(w: np.ndarray, bankroll0: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(w), size=(N_BOOT_DD, 52))
    pnl = w[idx]
    bank = bankroll0 + np.cumsum(pnl, axis=1)
    peak = np.maximum.accumulate(bank, axis=1)
    dd = (peak - bank).max(axis=1)
    return float(np.quantile(dd, 0.95))


def walkforward_weekly(
    df: pd.DataFrame,
    day: str,
    score_col: str,
    cutoff_grid: np.ndarray,
    stake_frac: float,
    roi_mode: str,
    min_hist_weeks: int = 4,
) -> np.ndarray:
    """
    Walk-forward simples para um único dia:
      - para cada semana t: otimiza cutoff em semanas < t (objetivo mean-0.25*std)
      - aplica na semana t
    Retorna série de PnL por semana (test weeks).
    """
    x = df[df["dow_pt"] == day].copy()
    if x.empty:
        return np.array([], dtype=float)

    roi = x["roi"].to_numpy(dtype=float)
    score = x[score_col].to_numpy(dtype=float)
    cap = x["house_cap"].to_numpy(dtype=float)
    wk = x["week"].to_numpy()

    # ordenar semanas
    weeks = sorted(pd.unique(wk).tolist())
    stake0 = BANKROLL * stake_frac

    out = []
    for i, w_test in enumerate(weeks):
        hist_weeks = weeks[:i]
        if len(hist_weeks) < min_hist_weeks:
            continue

        hist_mask = np.isin(wk, hist_weeks)
        test_mask = wk == w_test
        if not np.any(test_mask):
            continue

        best_cut = 1.0
        best_obj = -np.inf
        for c in cutoff_grid:
            m = hist_mask & (score >= c)
            if not np.any(m):
                continue
            # weekly pnl in hist
            stake = np.minimum(stake0, cap[m])
            pnl = stake * roi[m]
            wp = pd.Series(pnl, index=wk[m]).groupby(level=0).sum().to_numpy(dtype=float)
            if wp.size < 2:
                continue
            obj = float(wp.mean() - 0.25 * wp.std(ddof=1))
            if obj > best_obj:
                best_obj = obj
                best_cut = float(c)

        # aplicar em semana teste
        m2 = test_mask & (score >= best_cut)
        if not np.any(m2):
            out.append(0.0)
        else:
            stake = np.minimum(stake0, cap[m2])
            out.append(float((stake * roi[m2]).sum()))

    return np.asarray(out, dtype=float)


@dataclass(frozen=True)
class Candidate:
    cutoff: float
    stake_frac: float
    # primary metrics (cap2)
    cap2_mean: float
    cap2_std: float
    cap2_pneg: float
    cap2_var05_52w: float
    cap2_cvar05_52w: float
    cap2_dd_p95_52w: float
    # stress (cap1)
    cap1_mean: float
    cap1_pneg: float
    cap1_dd_p95_52w: float
    # walk-forward (cap2)
    wf_cap2_mean: float
    wf_cap2_pneg: float
    wf_cap2_p05: float
    # objective value
    obj: float


def evaluate_candidate(
    df_day: pd.DataFrame,
    score_col: str,
    cutoff: float,
    stake_frac: float,
    weeks_all: List[str],
    seed: int,
) -> Candidate | None:
    x = df_day.copy()
    score = x[score_col].to_numpy(dtype=float)
    m = score >= cutoff
    if not np.any(m):
        return None

    stake0 = BANKROLL * stake_frac
    cap = x["house_cap"].to_numpy(dtype=float)
    wk = x["week"].to_numpy()

    # build weekly pnl series for each scenario, aligned to all weeks (fill 0)
    out = {}
    for mode in ("cap2", "cap1"):
        roi = roi_mode_arr(x["roi_raw"].to_numpy(dtype=float), mode)
        stake = np.minimum(stake0, cap[m])
        pnl = stake * roi[m]
        s = pd.Series(pnl, index=wk[m]).groupby(level=0).sum()
        s = s.reindex(weeks_all, fill_value=0.0)
        out[mode] = s.to_numpy(dtype=float)

    w2 = out["cap2"]
    w1 = out["cap1"]

    # basic stats
    cap2_mean = float(w2.mean())
    cap2_std = float(w2.std(ddof=1)) if w2.size >= 2 else 0.0
    cap2_pneg = float((w2 < 0).mean())
    cap1_mean = float(w1.mean())
    cap1_pneg = float((w1 < 0).mean())

    # bootstrap risk (52w sums) and drawdown p95
    cap2_var05, cap2_cvar05 = bootstrap_sum(w2, 52, seed=seed + 1)
    cap2_dd_p95 = bootstrap_dd(w2, BANKROLL, seed=seed + 2)
    cap1_dd_p95 = bootstrap_dd(w1, BANKROLL, seed=seed + 3)

    # walk-forward (cap2) on this day
    # create a working frame with roi already set to cap2
    df_wf = x.copy()
    df_wf["roi"] = roi_mode_arr(df_wf["roi_raw"].to_numpy(dtype=float), "cap2")
    wf = walkforward_weekly(df_wf, x["dow_pt"].iloc[0], score_col, np.round(np.arange(0.05, 0.951, 0.01), 2), stake_frac, "cap2")
    if wf.size == 0:
        wf_mean = 0.0
        wf_pneg = 0.0
        wf_p05 = 0.0
    else:
        wf_mean = float(wf.mean())
        wf_pneg = float((wf < 0).mean())
        wf_p05 = float(np.quantile(wf, 0.05))

    # objective: cap2 weekly mean penalized by volatility and drawdown
    obj = cap2_mean - 0.25 * cap2_std - 0.01 * cap2_dd_p95

    return Candidate(
        cutoff=float(cutoff),
        stake_frac=float(stake_frac),
        cap2_mean=cap2_mean,
        cap2_std=cap2_std,
        cap2_pneg=cap2_pneg,
        cap2_var05_52w=cap2_var05,
        cap2_cvar05_52w=cap2_cvar05,
        cap2_dd_p95_52w=cap2_dd_p95,
        cap1_mean=cap1_mean,
        cap1_pneg=cap1_pneg,
        cap1_dd_p95_52w=cap1_dd_p95,
        wf_cap2_mean=wf_mean,
        wf_cap2_pneg=wf_pneg,
        wf_cap2_p05=wf_p05,
        obj=obj,
    )


def pro_constraints(c: Candidate) -> bool:
    """
    Constraints "mesa profissional" (conservadoras):
    - cap2: EV semanal > 0
    - cap2: P(semana<0) <= 35%
    - cap2: VaR5% anual (52w) > 0  (5% de chance de ano negativo)
    - cap2: drawdown p95 (52w) <= 60% da banca (USD 1.380)
    - walk-forward cap2: EV semanal > 0 e p05 não muito negativo
    - cap1: não pode ser desastroso (EV não muito negativo e DD p95 limitado)
    """
    if c.cap2_mean <= 0:
        return False
    if c.cap2_pneg > 0.35:
        return False
    if c.cap2_var05_52w <= 0:
        return False
    if c.cap2_dd_p95_52w > 0.60 * BANKROLL:
        return False
    if c.wf_cap2_mean <= 0:
        return False
    if c.wf_cap2_p05 < -0.5 * BANKROLL * MAX_FRAC * 4:  # ~ perda de 2 semanas ruins
        return False
    if c.cap1_mean < -0.10 * c.cap2_mean:  # cap1 não pode inverter completamente
        return False
    if c.cap1_dd_p95_52w > 1.5 * BANKROLL:
        return False
    return True


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])
    df["house_cap"] = df["house_cap"].apply(safe_cap)
    df = df[(df["BIA_ApostaUTC"] >= TRAIN_START) & (df["BIA_ApostaUTC"] <= TRAIN_END)].copy()
    df = df[df["Tipo Aposta"].apply(is_ft)].copy()

    # base fields
    df["week"] = week_key(df["BIA_ApostaUTC"])
    df["roi_raw"] = pd.to_numeric(df["ROI Real"], errors="coerce")
    # drop invalid ROI/score later per-day

    legs = {
        "segunda-feira": "proba_raw_segunda",
        "terça-feira": "proba_raw_terca",
        "quarta-feira": "proba_raw_quarta",
        "quinta-feira": "proba_raw_segqui",
    }

    stake_fracs = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
    cutoffs = np.round(np.arange(0.05, 0.951, 0.01), 2)

    weeks_all = sorted(df["week"].unique().tolist())

    chosen = {}
    per_leg_report = {}

    for day, sc in legs.items():
        xd = df[df["dow_pt"] == day].copy()
        xd["score"] = pd.to_numeric(xd[sc], errors="coerce")
        xd = xd[np.isfinite(xd["roi_raw"]) & np.isfinite(xd["score"])].copy()
        if xd.empty:
            chosen[day] = {"type": "FT", "score_col": sc, "cutoff": 1.0, "stake_frac": 0.0}
            per_leg_report[day] = {"status": "no_data"}
            continue

        # ensure columns exist
        xd[sc] = xd["score"].astype(float)
        xd["roi_raw"] = xd["roi_raw"].astype(float)
        xd["house_cap"] = xd["house_cap"].astype(float)

        best: Candidate | None = None
        best_reason = "no_candidate"

        for f in stake_fracs:
            for c in cutoffs:
                cand = evaluate_candidate(xd, sc, float(c), float(f), weeks_all, seed=SEED + hash((day, c, f)) % 10000)
                if cand is None:
                    continue
                if not pro_constraints(cand):
                    continue
                if best is None or cand.obj > best.obj:
                    best = cand
                    best_reason = "ok"

        if best is None:
            chosen[day] = {"type": "FT", "score_col": sc, "cutoff": 1.0, "stake_frac": 0.0}
            per_leg_report[day] = {"status": best_reason}
        else:
            chosen[day] = {"type": "FT", "score_col": sc, "cutoff": best.cutoff, "stake_frac": best.stake_frac}
            per_leg_report[day] = {
                "status": "ok",
                "cap2_mean_week": best.cap2_mean,
                "cap2_std_week": best.cap2_std,
                "cap2_pneg_week": best.cap2_pneg,
                "cap2_var05_52w": best.cap2_var05_52w,
                "cap2_cvar05_52w": best.cap2_cvar05_52w,
                "cap2_dd_p95_52w": best.cap2_dd_p95_52w,
                "cap1_mean_week": best.cap1_mean,
                "cap1_pneg_week": best.cap1_pneg,
                "cap1_dd_p95_52w": best.cap1_dd_p95_52w,
                "wf_cap2_mean": best.wf_cap2_mean,
                "wf_cap2_pneg": best.wf_cap2_pneg,
                "wf_cap2_p05": best.wf_cap2_p05,
                "obj": best.obj,
            }

    out = {
        "bankroll": BANKROLL,
        "max_frac_per_bet": MAX_FRAC,
        "train_period": {"start": str(TRAIN_START.date()), "end": str(TRAIN_END.date())},
        "definition": {"type_filter": "FT (exclude first half)", "score": "proba_raw (operational, clipped)"},
        "constraints": {
            "cap2_mean_week_gt_0": True,
            "cap2_pneg_week_le": 0.35,
            "cap2_var05_52w_gt_0": True,
            "cap2_dd_p95_52w_le_bankroll_frac": 0.60,
            "wf_cap2_mean_gt_0": True,
            "wf_cap2_p05_ge": -0.5 * BANKROLL * MAX_FRAC * 4,
            "cap1_mean_not_too_negative": "cap1_mean >= -0.10 * cap2_mean",
            "cap1_dd_p95_le": 1.5 * BANKROLL,
        },
        "portfolio": chosen,
        "per_leg_stability": per_leg_report,
    }

    (OUT_DIR / "portfolio_pro.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # markdown summary
    lines = []
    lines.append("## Portfólio (mesa profissional) — resumo\n")
    lines.append(f"- Treino: {TRAIN_START.date()}..{TRAIN_END.date()}\n")
    lines.append(f"- Banca: USD {BANKROLL:,.0f}; stake máximo: {MAX_FRAC*100:.1f}% (USD {BANKROLL*MAX_FRAC:.0f})\n")
    lines.append("\n### Regras\n")
    for day in ("segunda-feira", "terça-feira", "quarta-feira", "quinta-feira"):
        r = chosen[day]
        if r["stake_frac"] <= 0:
            lines.append(f"- **{day}**: (removido)\n")
        else:
            lines.append(f"- **{day}**: FT, `{r['score_col']}` ≥ **{r['cutoff']:.2f}**, stake **{r['stake_frac']*100:.1f}%**\n")

    lines.append("\n### Estabilidade por perna (cap2 + walk-forward cap2)\n")
    for day in ("segunda-feira", "terça-feira", "quarta-feira", "quinta-feira"):
        rep = per_leg_report.get(day, {})
        if rep.get("status") != "ok":
            lines.append(f"- **{day}**: sem candidato viável pelos constraints.\n")
            continue
        lines.append(
            f"- **{day}**: mean_week={rep['cap2_mean_week']:.0f}, std_week={rep['cap2_std_week']:.0f}, "
            f"P(week<0)={rep['cap2_pneg_week']*100:.1f}%, VaR5%(52w)={rep['cap2_var05_52w']:.0f}, "
            f"DD_p95(52w)={rep['cap2_dd_p95_52w']:.0f}; wf_mean={rep['wf_cap2_mean']:.0f}, wf_p05={rep['wf_cap2_p05']:.0f}\n"
        )

    (OUT_DIR / "report_pro.md").write_text("".join(lines), encoding="utf-8")
    print(str(OUT_DIR / "report_pro.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

