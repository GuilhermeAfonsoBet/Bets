#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Versão "mesa profissional" para:
  1) FT em todos os dias (Seg..Dom)
  2) FH em todos os dias (Seg..Dom)

Com: stress tests cap2/cap1, walk-forward semanal e restrição explícita de
exposição diária (porque volume por dia muda o risco do staking).

Saídas:
  /workspace/analysis_proba_raw/pro_portfolio_all/portfolio_pro_all.json
  /workspace/analysis_proba_raw/pro_portfolio_all/report_pro_all.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")
OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")

TRAIN_START = pd.Timestamp("2025-10-01")
TRAIN_END = pd.Timestamp("2025-12-31 23:59:59")

BANKROLL = 2300.0
MAX_FRAC = 0.07

# Volume-aware risk control (mesa): p95 de exposição diária <= 25% da banca
MAX_DAILY_EXPOSURE_FRAC_P95 = 0.25

N_BOOT = 20_000
N_BOOT_DD = 5_000
SEED = 7


WEEKDAY_PT = [
    "segunda-feira",
    "terça-feira",
    "quarta-feira",
    "quinta-feira",
    "sexta-feira",
    "sábado",
    "domingo",
]


def is_fh(tipo: str) -> bool:
    return "first half" in str(tipo).lower()


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


def date_key(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts).dt.date.astype(str)


def roi_mode_arr(roi: np.ndarray, mode: str) -> np.ndarray:
    if mode == "cap2":
        return np.minimum(roi, 2.0)
    if mode == "cap1":
        return np.minimum(roi, 1.0)
    raise ValueError(mode)


def bootstrap_sum(w: np.ndarray, H: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(w), size=(N_BOOT, H))
    sums = w[idx].sum(axis=1)
    return float(np.quantile(sums, 0.05))


def bootstrap_dd_p95(w: np.ndarray, bankroll0: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(w), size=(N_BOOT_DD, 52))
    pnl = w[idx]
    bank = bankroll0 + np.cumsum(pnl, axis=1)
    peak = np.maximum.accumulate(bank, axis=1)
    dd = (peak - bank).max(axis=1)
    return float(np.quantile(dd, 0.95))


def walkforward_weekly_pnl(
    df: pd.DataFrame,
    cutoff_grid: np.ndarray,
    stake_frac: float,
    score_col: str,
) -> np.ndarray:
    """
    Walk-forward (expanding window) no treino, otimiza cutoff com mean-0.25*std
    e aplica na próxima semana.
    """
    weeks = sorted(df["week"].unique().tolist())
    stake0 = BANKROLL * stake_frac
    out = []
    for i, wk_test in enumerate(weeks):
        hist_weeks = weeks[:i]
        if len(hist_weeks) < 4:
            continue
        hist = df[df["week"].isin(hist_weeks)]
        test = df[df["week"] == wk_test]
        if test.empty:
            continue

        best_cut = 1.0
        best_obj = -np.inf
        for c in cutoff_grid:
            sel = hist[hist[score_col] >= c]
            if sel.empty:
                continue
            pnl = np.minimum(stake0, sel["house_cap"].to_numpy(dtype=float)) * sel["roi_cap2"].to_numpy(dtype=float)
            wp = pd.Series(pnl, index=sel["week"]).groupby(level=0).sum().to_numpy(dtype=float)
            if wp.size < 2:
                continue
            obj = float(wp.mean() - 0.25 * wp.std(ddof=1))
            if obj > best_obj:
                best_obj = obj
                best_cut = float(c)

        sel_t = test[test[score_col] >= best_cut]
        if sel_t.empty:
            out.append(0.0)
        else:
            pnl_t = np.minimum(stake0, sel_t["house_cap"].to_numpy(dtype=float)) * sel_t["roi_cap2"].to_numpy(dtype=float)
            out.append(float(pnl_t.sum()))
    return np.asarray(out, dtype=float)


@dataclass(frozen=True)
class Candidate:
    cutoff: float
    stake_frac: float
    # stats cap2 weekly
    cap2_mean_week: float
    cap2_std_week: float
    cap2_pneg_week: float
    cap2_var05_52w: float
    cap2_dd_p95_52w: float
    # stress cap1
    cap1_mean_week: float
    cap1_pneg_week: float
    cap1_dd_p95_52w: float
    # daily exposure
    p95_daily_exposure: float
    p95_daily_bets: float
    # walk-forward cap2
    wf_mean: float
    wf_pneg: float
    wf_p05: float
    obj: float


def evaluate_candidate(
    df_seg: pd.DataFrame,
    score_col: str,
    cutoff: float,
    stake_frac: float,
    weeks_all: List[str],
    seed: int,
) -> Candidate | None:
    stake0 = BANKROLL * stake_frac
    sel = df_seg[df_seg[score_col] >= cutoff]
    if sel.empty:
        return None

    # weekly pnl series aligned
    pnl2 = np.minimum(stake0, sel["house_cap"].to_numpy(dtype=float)) * sel["roi_cap2"].to_numpy(dtype=float)
    w2 = pd.Series(pnl2, index=sel["week"]).groupby(level=0).sum().reindex(weeks_all, fill_value=0.0).to_numpy(dtype=float)
    pnl1 = np.minimum(stake0, sel["house_cap"].to_numpy(dtype=float)) * sel["roi_cap1"].to_numpy(dtype=float)
    w1 = pd.Series(pnl1, index=sel["week"]).groupby(level=0).sum().reindex(weeks_all, fill_value=0.0).to_numpy(dtype=float)

    cap2_mean = float(w2.mean())
    cap2_std = float(w2.std(ddof=1)) if w2.size >= 2 else 0.0
    cap2_pneg = float((w2 < 0).mean())
    cap1_mean = float(w1.mean())
    cap1_pneg = float((w1 < 0).mean())

    var05_52 = bootstrap_sum(w2, 52, seed=seed + 1)
    dd2 = bootstrap_dd_p95(w2, BANKROLL, seed=seed + 2)
    dd1 = bootstrap_dd_p95(w1, BANKROLL, seed=seed + 3)

    # daily exposure distribution (cap2 scenario; exposure = stake0 * n_bets_day)
    dcnt = sel.groupby("date").size().to_numpy(dtype=float)
    if dcnt.size == 0:
        return None
    p95_bets = float(np.quantile(dcnt, 0.95))
    p95_exp = float(p95_bets * stake0)

    # walk-forward
    wf = walkforward_weekly_pnl(df_seg, np.round(np.arange(0.05, 0.951, 0.01), 2), stake_frac, score_col)
    if wf.size == 0:
        wf_mean = 0.0
        wf_pneg = 0.0
        wf_p05 = 0.0
    else:
        wf_mean = float(wf.mean())
        wf_pneg = float((wf < 0).mean())
        wf_p05 = float(np.quantile(wf, 0.05))

    # objective: conservative + penalty for daily exposure p95
    obj = cap2_mean - 0.25 * cap2_std - 0.01 * dd2 - 0.001 * p95_exp

    return Candidate(
        cutoff=float(cutoff),
        stake_frac=float(stake_frac),
        cap2_mean_week=cap2_mean,
        cap2_std_week=cap2_std,
        cap2_pneg_week=cap2_pneg,
        cap2_var05_52w=var05_52,
        cap2_dd_p95_52w=dd2,
        cap1_mean_week=cap1_mean,
        cap1_pneg_week=cap1_pneg,
        cap1_dd_p95_52w=dd1,
        p95_daily_exposure=p95_exp,
        p95_daily_bets=p95_bets,
        wf_mean=wf_mean,
        wf_pneg=wf_pneg,
        wf_p05=wf_p05,
        obj=obj,
    )


def pro_constraints(c: Candidate) -> bool:
    # cap2: edge e risco
    if c.cap2_mean_week <= 0:
        return False
    if c.cap2_pneg_week > 0.40:
        return False
    if c.cap2_var05_52w <= 0:
        return False
    if c.cap2_dd_p95_52w > 0.60 * BANKROLL:
        return False

    # daily exposure constraint
    if c.p95_daily_exposure > MAX_DAILY_EXPOSURE_FRAC_P95 * BANKROLL:
        return False

    # walk-forward
    if c.wf_mean <= 0:
        return False
    if c.wf_p05 < -0.5 * BANKROLL * MAX_FRAC * 4:
        return False

    # cap1 sanity
    if c.cap1_mean_week < -0.10 * c.cap2_mean_week:
        return False
    if c.cap1_dd_p95_52w > 1.5 * BANKROLL:
        return False
    return True


def segment_score_col(dow: str) -> str:
    if dow == "segunda-feira":
        return "proba_raw_segunda"
    if dow == "terça-feira":
        return "proba_raw_terca"
    if dow == "quarta-feira":
        return "proba_raw_quarta"
    if dow == "quinta-feira":
        return "proba_raw_segqui"
    # sexdom
    return "proba_raw_sexdom"


def optimize_for_segment(df: pd.DataFrame, dow: str, bet_type: str) -> Tuple[Dict, Dict]:
    sc = segment_score_col(dow)
    x = df[(df["dow_pt"] == dow) & (df["bet_type"] == bet_type)].copy()
    if x.empty:
        return {"type": bet_type, "score_col": sc, "cutoff": 1.0, "stake_frac": 0.0}, {"status": "no_data"}

    x["score"] = pd.to_numeric(x[sc], errors="coerce")
    x = x[np.isfinite(x["roi_raw"]) & np.isfinite(x["score"])].copy()
    if x.empty:
        return {"type": bet_type, "score_col": sc, "cutoff": 1.0, "stake_frac": 0.0}, {"status": "no_valid_rows"}
    x[sc] = x["score"].astype(float)

    weeks_all = sorted(x["week"].unique().tolist())
    if len(weeks_all) < 6:
        return {"type": bet_type, "score_col": sc, "cutoff": 1.0, "stake_frac": 0.0}, {"status": "too_few_weeks"}

    stake_fracs = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
    cutoffs = np.round(np.arange(0.05, 0.951, 0.01), 2)

    best = None
    for f in stake_fracs:
        for c in cutoffs:
            cand = evaluate_candidate(x, sc, float(c), float(f), weeks_all, seed=SEED + hash((dow, bet_type, c, f)) % 10000)
            if cand is None:
                continue
            if not pro_constraints(cand):
                continue
            if best is None or cand.obj > best.obj:
                best = cand

    if best is None:
        return {"type": bet_type, "score_col": sc, "cutoff": 1.0, "stake_frac": 0.0}, {"status": "no_candidate"}

    rule = {"type": bet_type, "score_col": sc, "cutoff": best.cutoff, "stake_frac": best.stake_frac}
    rep = {
        "status": "ok",
        "cap2_mean_week": best.cap2_mean_week,
        "cap2_std_week": best.cap2_std_week,
        "cap2_pneg_week": best.cap2_pneg_week,
        "cap2_var05_52w": best.cap2_var05_52w,
        "cap2_dd_p95_52w": best.cap2_dd_p95_52w,
        "cap1_mean_week": best.cap1_mean_week,
        "cap1_pneg_week": best.cap1_pneg_week,
        "cap1_dd_p95_52w": best.cap1_dd_p95_52w,
        "p95_daily_bets": best.p95_daily_bets,
        "p95_daily_exposure": best.p95_daily_exposure,
        "wf_mean": best.wf_mean,
        "wf_pneg": best.wf_pneg,
        "wf_p05": best.wf_p05,
        "obj": best.obj,
    }
    return rule, rep


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])
    df = df[(df["BIA_ApostaUTC"] >= TRAIN_START) & (df["BIA_ApostaUTC"] <= TRAIN_END)].copy()

    df["house_cap"] = df["house_cap"].apply(safe_cap)
    df["week"] = week_key(df["BIA_ApostaUTC"])
    df["date"] = date_key(df["BIA_ApostaUTC"])
    df["roi_raw"] = pd.to_numeric(df["ROI Real"], errors="coerce")
    df["roi_cap2"] = np.minimum(df["roi_raw"].to_numpy(dtype=float), 2.0)
    df["roi_cap1"] = np.minimum(df["roi_raw"].to_numpy(dtype=float), 1.0)

    # bet type already present in scored file; but enforce
    df["bet_type"] = df.get("bet_type")
    if df["bet_type"].isna().any():
        df["bet_type"] = np.where(df["Tipo Aposta"].astype(str).str.lower().str.contains("first half"), "FH", "FT")

    portfolio = {}
    stability = {}

    for bet_type in ("FT", "FH"):
        portfolio[bet_type] = {}
        stability[bet_type] = {}
        for dow in WEEKDAY_PT:
            rule, rep = optimize_for_segment(df, dow, bet_type)
            portfolio[bet_type][dow] = rule
            stability[bet_type][dow] = rep

    out = {
        "bankroll": BANKROLL,
        "max_frac_per_bet": MAX_FRAC,
        "train_period": {"start": str(TRAIN_START.date()), "end": str(TRAIN_END.date())},
        "daily_exposure_constraint": {"p95_daily_exposure_le_frac_bankroll": MAX_DAILY_EXPOSURE_FRAC_P95},
        "portfolio": portfolio,
        "stability": stability,
    }
    (OUT_DIR / "portfolio_pro_all.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # markdown summary (compact)
    lines = []
    lines.append("## Portfólio mesa profissional — FT (Seg..Dom) e FH (Seg..Dom)\n")
    lines.append(f"- Banca: USD {BANKROLL:,.0f}; max por aposta: {MAX_FRAC*100:.1f}%\n")
    lines.append(f"- Constraint volume: p95 exposição diária <= {MAX_DAILY_EXPOSURE_FRAC_P95*100:.0f}% da banca\n")

    for bet_type in ("FT", "FH"):
        lines.append(f"\n### {bet_type}\n")
        for dow in WEEKDAY_PT:
            r = portfolio[bet_type][dow]
            if r["stake_frac"] <= 0:
                lines.append(f"- **{dow}**: (removido)\n")
            else:
                lines.append(f"- **{dow}**: `{r['score_col']}` ≥ **{r['cutoff']:.2f}**, stake **{r['stake_frac']*100:.1f}%**\n")

    (OUT_DIR / "report_pro_all.md").write_text("".join(lines), encoding="utf-8")
    print(str(OUT_DIR / "report_pro_all.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

