#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfólio "mesa profissional" para:
  - FT em Seg..Dom
  - FH em Seg..Dom

Robustez:
  - stress tests cap2/cap1 no ROI
  - constraints de risco (VaR/CVaR via bootstrap semanal, drawdown via paths)
  - constraint explícita de exposição diária (p95)
  - walk-forward semanal (expanding window) apenas nos top candidatos

Otimização é feita em 2 estágios para rodar rápido:
  1) varredura barata (mean/std/p_neg + exposição diária)
  2) validação pesada (bootstrap + drawdown + walk-forward) só nos top-K
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

# Mesa: p95 da exposição diária (stake somado) <= 25% da banca
MAX_DAILY_EXPOSURE_FRAC_P95 = 0.25

# Estágio 1 (barato)
STAKE_FRACS = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
CUTOFFS = np.round(np.arange(0.05, 0.951, 0.02), 2)  # mais grosso para velocidade
TOP_K = 8

# Estágio 2 (pesado)
N_BOOT = 8_000
N_BOOT_DD = 3_000
SEED = 7
N_SCORE_BINS = 5
MIN_POS_BINS_CAP2 = 4

WEEKDAY_PT = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo"]


def week_key(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts).dt.to_period("W-SUN").astype(str)


def date_key(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts).dt.date.astype(str)


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


def segment_score_col(dow: str) -> str:
    if dow == "segunda-feira":
        return "proba_raw_segunda"
    if dow == "terça-feira":
        return "proba_raw_terca"
    if dow == "quarta-feira":
        return "proba_raw_quarta"
    if dow == "quinta-feira":
        return "proba_raw_segqui"
    return "proba_raw_sexdom"


def bootstrap_var_cvar(w: np.ndarray, H: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(w), size=(N_BOOT, H))
    sums = w[idx].sum(axis=1)
    var05 = float(np.quantile(sums, 0.05))
    cvar05 = float(np.mean(sums[sums <= var05]))
    return var05, cvar05


def bootstrap_drawdown_p95(w: np.ndarray, bankroll0: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(w), size=(N_BOOT_DD, 52))
    pnl = w[idx]
    bank = bankroll0 + np.cumsum(pnl, axis=1)
    peak = np.maximum.accumulate(bank, axis=1)
    dd = (peak - bank).max(axis=1)
    return float(np.quantile(dd, 0.95))


def walkforward_weekly_cap2(df: pd.DataFrame, score_col: str, stake_eff: np.ndarray, cutoff_grid: np.ndarray) -> np.ndarray:
    """
    Walk-forward no treino (cap2), escolhendo cutoff por mean-0.25*std em semanas passadas.
    stake_eff é por aposta (já com min(stake0, cap)).
    """
    weeks = sorted(df["week"].unique().tolist())
    out = []
    score = df[score_col].to_numpy(dtype=float)
    wk = df["week"].to_numpy()
    roi2 = df["roi_cap2"].to_numpy(dtype=float)
    for i, w_test in enumerate(weeks):
        hist_weeks = weeks[:i]
        if len(hist_weeks) < 4:
            continue
        hist_mask = np.isin(wk, hist_weeks)
        test_mask = wk == w_test
        best_cut = 1.0
        best_obj = -np.inf
        for c in cutoff_grid:
            m = hist_mask & (score >= c)
            if not np.any(m):
                continue
            pnl = stake_eff[m] * roi2[m]
            wp = pd.Series(pnl, index=wk[m]).groupby(level=0).sum().to_numpy(dtype=float)
            if wp.size < 2:
                continue
            obj = float(wp.mean() - 0.25 * wp.std(ddof=1))
            if obj > best_obj:
                best_obj = obj
                best_cut = float(c)
        m2 = test_mask & (score >= best_cut)
        out.append(float((stake_eff[m2] * roi2[m2]).sum()) if np.any(m2) else 0.0)
    return np.asarray(out, dtype=float)


@dataclass(frozen=True)
class Cand1:
    cutoff: float
    stake_frac: float
    mean_w: float
    std_w: float
    pneg_w: float
    cap1_mean_w: float
    p95_daily_exposure: float
    obj: float


@dataclass(frozen=True)
class CandFinal:
    cutoff: float
    stake_frac: float
    cap2_mean_w: float
    cap2_std_w: float
    cap2_pneg_w: float
    cap2_var05_52w: float
    cap2_cvar05_52w: float
    cap2_dd_p95_52w: float
    cap1_mean_w: float
    cap1_pneg_w: float
    cap1_dd_p95_52w: float
    p95_daily_exposure: float
    p95_daily_bets: float
    wf_mean: float
    wf_pneg: float
    wf_p05: float
    # score-bin stability (cap2)
    n_bins: int
    pos_bins: int
    obj: float


def stage1_candidates(df: pd.DataFrame, score_col: str) -> List[Cand1]:
    weeks_all = sorted(df["week"].unique().tolist())
    out: List[Cand1] = []

    score = df[score_col].to_numpy(dtype=float)
    roi2 = df["roi_cap2"].to_numpy(dtype=float)
    roi1 = df["roi_cap1"].to_numpy(dtype=float)
    cap = df["house_cap"].to_numpy(dtype=float)
    wk = df["week"].to_numpy()
    d = df["date"].to_numpy()

    for f in STAKE_FRACS:
        stake0 = BANKROLL * float(f)
        stake_eff = np.minimum(stake0, cap)
        for c in CUTOFFS:
            m = score >= c
            if not np.any(m):
                continue
            # weekly pnl (cap2)
            pnl2 = stake_eff[m] * roi2[m]
            s2 = pd.Series(pnl2, index=wk[m]).groupby(level=0).sum().reindex(weeks_all, fill_value=0.0)
            w2 = s2.to_numpy(dtype=float)
            mean = float(w2.mean())
            std = float(w2.std(ddof=1)) if w2.size >= 2 else 0.0
            pneg = float((w2 < 0).mean())
            if mean <= 0:
                continue
            if pneg > 0.40:
                continue

            # cap1 mean (sanidade barata)
            pnl1 = stake_eff[m] * roi1[m]
            s1 = pd.Series(pnl1, index=wk[m]).groupby(level=0).sum().reindex(weeks_all, fill_value=0.0)
            mean1 = float(s1.to_numpy(dtype=float).mean())
            if mean1 < -0.10 * mean:
                continue

            # daily exposure p95 (stake somado) e p95 bets/day
            stake_day = pd.Series(stake_eff[m], index=d[m]).groupby(level=0).sum().to_numpy(dtype=float)
            if stake_day.size == 0:
                continue
            p95_exp = float(np.quantile(stake_day, 0.95))
            if p95_exp > MAX_DAILY_EXPOSURE_FRAC_P95 * BANKROLL:
                continue

            # objective (barato): mean - 0.25 std - penalty exposure
            obj = mean - 0.25 * std - 0.001 * p95_exp
            out.append(Cand1(cutoff=float(c), stake_frac=float(f), mean_w=mean, std_w=std, pneg_w=pneg, cap1_mean_w=mean1, p95_daily_exposure=p95_exp, obj=obj))

    out.sort(key=lambda z: z.obj, reverse=True)
    return out[:TOP_K]


def stage2_validate(df: pd.DataFrame, score_col: str, c1: Cand1, seed: int) -> CandFinal | None:
    weeks_all = sorted(df["week"].unique().tolist())
    score = df[score_col].to_numpy(dtype=float)
    roi2 = df["roi_cap2"].to_numpy(dtype=float)
    roi1 = df["roi_cap1"].to_numpy(dtype=float)
    cap = df["house_cap"].to_numpy(dtype=float)
    wk = df["week"].to_numpy()
    d = df["date"].to_numpy()

    stake0 = BANKROLL * c1.stake_frac
    stake_eff = np.minimum(stake0, cap)
    m = score >= c1.cutoff

    # weekly pnl aligned
    pnl2 = stake_eff[m] * roi2[m]
    w2 = pd.Series(pnl2, index=wk[m]).groupby(level=0).sum().reindex(weeks_all, fill_value=0.0).to_numpy(dtype=float)
    pnl1 = stake_eff[m] * roi1[m]
    w1 = pd.Series(pnl1, index=wk[m]).groupby(level=0).sum().reindex(weeks_all, fill_value=0.0).to_numpy(dtype=float)

    cap2_mean = float(w2.mean())
    cap2_std = float(w2.std(ddof=1)) if w2.size >= 2 else 0.0
    cap2_pneg = float((w2 < 0).mean())
    cap1_mean = float(w1.mean())
    cap1_pneg = float((w1 < 0).mean())

    var05, cvar05 = bootstrap_var_cvar(w2, 52, seed=seed + 1)
    if var05 <= 0:
        return None
    dd2 = bootstrap_drawdown_p95(w2, BANKROLL, seed=seed + 2)
    if dd2 > 0.60 * BANKROLL:
        return None
    dd1 = bootstrap_drawdown_p95(w1, BANKROLL, seed=seed + 3)
    if dd1 > 1.5 * BANKROLL:
        return None

    # daily exposure and bet counts
    stake_day = pd.Series(stake_eff[m], index=d[m]).groupby(level=0).sum().to_numpy(dtype=float)
    p95_exp = float(np.quantile(stake_day, 0.95)) if stake_day.size else 0.0
    if p95_exp > MAX_DAILY_EXPOSURE_FRAC_P95 * BANKROLL:
        return None
    cnt_day = pd.Series(np.ones_like(stake_eff[m], dtype=int), index=d[m]).groupby(level=0).sum().to_numpy(dtype=float)
    p95_bets = float(np.quantile(cnt_day, 0.95)) if cnt_day.size else 0.0

    # walk-forward
    wf = walkforward_weekly_cap2(df, score_col, stake_eff, cutoff_grid=CUTOFFS)
    if wf.size == 0:
        return None
    wf_mean = float(wf.mean())
    wf_pneg = float((wf < 0).mean())
    wf_p05 = float(np.quantile(wf, 0.05))
    if wf_mean <= 0:
        return None
    if wf_p05 < -0.5 * BANKROLL * MAX_FRAC * 4:
        return None

    # -----------------------------------------
    # Filtro de estabilidade por bins de score
    # -----------------------------------------
    score_sel = score[m]
    profit_sel_cap2 = stake_eff[m] * roi2[m]
    if score_sel.size == 0:
        return None

    # bins por quantis dentro do conjunto selecionado
    edges = np.unique(np.quantile(score_sel, np.linspace(0.0, 1.0, N_SCORE_BINS + 1)))
    # se não tem spread suficiente, vira 1 bin
    if edges.size < 3:
        n_bins = 1
        pos_bins = 1 if float(np.mean(profit_sel_cap2)) > 0 else 0
    else:
        bins = []
        for a, b in zip(edges[:-1], edges[1:]):
            if b == edges[-1]:
                sel = (score_sel >= a) & (score_sel <= b)
            else:
                sel = (score_sel >= a) & (score_sel < b)
            if not np.any(sel):
                continue
            bins.append(sel)
        n_bins = len(bins)
        pos_bins = sum(1 for sel in bins if float(np.mean(profit_sel_cap2[sel])) > 0)

    # regra: exigir >=4/5 quando temos 5 bins; relaxa proporcionalmente se n_bins<5
    if n_bins >= N_SCORE_BINS:
        if pos_bins < MIN_POS_BINS_CAP2:
            return None
    elif n_bins == 4:
        if pos_bins < 3:
            return None
    elif n_bins == 3:
        if pos_bins < 2:
            return None
    else:
        # 1-2 bins: exigir todos positivos
        if pos_bins < n_bins:
            return None

    # objective final
    obj = cap2_mean - 0.25 * cap2_std - 0.01 * dd2 - 0.001 * p95_exp
    return CandFinal(
        cutoff=c1.cutoff,
        stake_frac=c1.stake_frac,
        cap2_mean_w=cap2_mean,
        cap2_std_w=cap2_std,
        cap2_pneg_w=cap2_pneg,
        cap2_var05_52w=var05,
        cap2_cvar05_52w=cvar05,
        cap2_dd_p95_52w=dd2,
        cap1_mean_w=cap1_mean,
        cap1_pneg_w=cap1_pneg,
        cap1_dd_p95_52w=dd1,
        p95_daily_exposure=p95_exp,
        p95_daily_bets=p95_bets,
        wf_mean=wf_mean,
        wf_pneg=wf_pneg,
        wf_p05=wf_p05,
        n_bins=int(n_bins),
        pos_bins=int(pos_bins),
        obj=obj,
    )


def optimize_segment(df: pd.DataFrame, dow: str, bet_type: str) -> Tuple[Dict, Dict]:
    sc = segment_score_col(dow)
    x = df[(df["dow_pt"] == dow) & (df["bet_type"] == bet_type)].copy()
    if x.empty:
        return {"type": bet_type, "score_col": sc, "cutoff": 1.0, "stake_frac": 0.0}, {"status": "no_data"}

    x["score"] = pd.to_numeric(x[sc], errors="coerce")
    x = x[np.isfinite(x["roi_raw"]) & np.isfinite(x["score"])].copy()
    if x.empty:
        return {"type": bet_type, "score_col": sc, "cutoff": 1.0, "stake_frac": 0.0}, {"status": "no_valid_rows"}
    x[sc] = x["score"].astype(float)

    weeks = sorted(x["week"].unique().tolist())
    if len(weeks) < 6:
        return {"type": bet_type, "score_col": sc, "cutoff": 1.0, "stake_frac": 0.0}, {"status": "too_few_weeks"}

    # stage 1: get top K cheap candidates
    cands1 = stage1_candidates(x, sc)
    if not cands1:
        return {"type": bet_type, "score_col": sc, "cutoff": 1.0, "stake_frac": 0.0}, {"status": "no_candidate_stage1"}

    best: CandFinal | None = None
    for j, c1 in enumerate(cands1):
        cand = stage2_validate(x, sc, c1, seed=SEED + hash((dow, bet_type, j)) % 10000)
        if cand is None:
            continue
        if best is None or cand.obj > best.obj:
            best = cand

    if best is None:
        return {"type": bet_type, "score_col": sc, "cutoff": 1.0, "stake_frac": 0.0}, {"status": "no_candidate_stage2"}

    rule = {"type": bet_type, "score_col": sc, "cutoff": best.cutoff, "stake_frac": best.stake_frac}
    rep = {
        "status": "ok",
        "cap2_mean_week": best.cap2_mean_w,
        "cap2_std_week": best.cap2_std_w,
        "cap2_pneg_week": best.cap2_pneg_w,
        "cap2_var05_52w": best.cap2_var05_52w,
        "cap2_cvar05_52w": best.cap2_cvar05_52w,
        "cap2_dd_p95_52w": best.cap2_dd_p95_52w,
        "cap1_mean_week": best.cap1_mean_w,
        "cap1_pneg_week": best.cap1_pneg_w,
        "cap1_dd_p95_52w": best.cap1_dd_p95_52w,
        "p95_daily_bets": best.p95_daily_bets,
        "p95_daily_exposure": best.p95_daily_exposure,
        "wf_mean": best.wf_mean,
        "wf_pneg": best.wf_pneg,
        "wf_p05": best.wf_p05,
        "score_bins_cap2": {"pos": best.pos_bins, "n": best.n_bins},
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
    df["roi_raw"] = pd.to_numeric(df["ROI Real"], errors="coerce").astype(float)
    df["roi_cap2"] = np.minimum(df["roi_raw"].to_numpy(dtype=float), 2.0)
    df["roi_cap1"] = np.minimum(df["roi_raw"].to_numpy(dtype=float), 1.0)
    df["bet_type"] = df.get("bet_type")
    if df["bet_type"].isna().any():
        df["bet_type"] = np.where(df["Tipo Aposta"].astype(str).str.lower().str.contains("first half"), "FH", "FT")

    portfolio: Dict[str, Dict[str, Dict]] = {"FT": {}, "FH": {}}
    stability: Dict[str, Dict[str, Dict]] = {"FT": {}, "FH": {}}

    for bet_type in ("FT", "FH"):
        for dow in WEEKDAY_PT:
            rule, rep = optimize_segment(df, dow, bet_type)
            portfolio[bet_type][dow] = rule
            stability[bet_type][dow] = rep

    out = {
        "bankroll": BANKROLL,
        "max_frac_per_bet": MAX_FRAC,
        "train_period": {"start": str(TRAIN_START.date()), "end": str(TRAIN_END.date())},
        "daily_exposure_constraint": {"p95_daily_exposure_le_frac_bankroll": MAX_DAILY_EXPOSURE_FRAC_P95},
        "grid": {"stake_fracs": STAKE_FRACS.tolist(), "cutoffs": CUTOFFS.tolist(), "top_k": TOP_K},
        "bootstrap": {"n_boot": N_BOOT, "n_boot_dd": N_BOOT_DD},
        "portfolio": portfolio,
        "stability": stability,
    }
    (OUT_DIR / "portfolio_pro_all.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # compact markdown
    lines = []
    lines.append("## Portfólio mesa profissional — FT e FH (todos os dias)\n")
    lines.append(f"- Banca: USD {BANKROLL:,.0f}; max por aposta: {MAX_FRAC*100:.1f}%\n")
    lines.append(f"- Constraint volume: p95 exposição diária <= {MAX_DAILY_EXPOSURE_FRAC_P95*100:.0f}% da banca\n")

    for bet_type in ("FT", "FH"):
        lines.append(f"\n### {bet_type}\n")
        for dow in WEEKDAY_PT:
            r = portfolio[bet_type][dow]
            rep = stability[bet_type][dow]
            if r["stake_frac"] <= 0:
                lines.append(f"- **{dow}**: (removido) — {rep.get('status')}\n")
            else:
                lines.append(
                    f"- **{dow}**: `{r['score_col']}` ≥ **{r['cutoff']:.2f}**, stake **{r['stake_frac']*100:.1f}%** "
                    f"(p95 exp dia ~USD {rep.get('p95_daily_exposure',0):.0f}; wf_mean {rep.get('wf_mean',0):.0f})\n"
                )

    (OUT_DIR / "report_pro_all.md").write_text("".join(lines), encoding="utf-8")
    print(str(OUT_DIR / "report_pro_all.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

