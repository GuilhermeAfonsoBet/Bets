#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compara, em OOS walk-forward (global_bayes), duas escalas de score para Sex/Sáb/Dom:
  - weekend_score = "raw": usa proba_raw_sexdom (como no estudo atual)
  - weekend_score = "cal": cria proba_cal_sexdom aplicando clv_calib_SexDom.json (isotonic + calib_floor)

Mantém a estratégia igual no resto:
  - Seg/Ter/Qua/Qui usam as colunas proba_raw_* já presentes no dataset scored.
  - Mesmos filtros/constraints/objetivo do evaluate_oos_walkforward_strategy.py (global_bayes).

Saídas (em analysis_proba_raw/pro_portfolio_all/):
  - oos_walkforward_global_bayes_weekend_raw_weekly.csv
  - oos_walkforward_global_bayes_weekend_raw_daily.csv
  - oos_walkforward_global_bayes_weekend_raw_selected_rules.csv
  - oos_walkforward_global_bayes_weekend_raw_strategy.md
  - (idem para weekend_cal)
  - oos_walkforward_global_bayes_weekend_score_comparison.md
"""

from __future__ import annotations

import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")
OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")
CALIB_SEXDOM = Path("/workspace/clv_calib_SexDom.json")

# sizing / constraints (iguais ao global_bayes atual)
BANKROLL = 2300.0
MAX_FRAC = 0.07
STAKE_FRACS = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
CUTOFFS = np.round(np.arange(0.05, 0.951, 0.02), 2)

DAILY_EXPOSURE_Q = 0.80
MAX_DAILY_EXPOSURE_FRAC_Q = 0.70
MAX_DAILY_DRAWDOWN_FRAC = 0.25
DAILY_VAR_Q = 0.10
MAX_P_DAILY_DD = 0.10
MIN_WEEKLY_SHARPE_CAP2 = 0.10

N_SCORE_BINS = 5
MIN_POS_BINS_CAP2 = 4
ENABLE_SCORE_BIN_STABILITY = True

MIN_SELECTED_BETS = 6
MIN_BETS_PER_BIN = 20
MIN_BINS_FOR_STABILITY = 3
MIN_NONZERO_WEEKS = 6

MIN_GLOBAL_TRAIN_WEEKS = 10
MIN_SEG_TRAIN_WEEKS = 6

# seleção bayesiana (como no global_bayes)
BAYES_N = 8_000
MIN_POST_P_MEAN_POS = 0.80
POST_Q_OBJ = 0.05
EXPOSURE_PENALTY = 0.001

SEED = 7

WEEKDAY_PT = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo"]
WEEKEND_PT = {"sexta-feira", "sábado", "domingo"}


def week_key(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts).dt.to_period("W-SUN").astype(str)


def date_key(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts).dt.date.astype(str)


def safe_cap(x) -> float:
    try:
        v = float(x)
    except Exception:
        return float("inf")
    if not np.isfinite(v) or v <= 0:
        return float("inf")
    return v


def segment_score_col(dow: str, weekend_score: str) -> str:
    if dow == "segunda-feira":
        return "proba_raw_segunda"
    if dow == "terça-feira":
        return "proba_raw_terca"
    if dow == "quarta-feira":
        return "proba_raw_quarta"
    if dow == "quinta-feira":
        return "proba_raw_segqui"
    # sex/sab/dom:
    return "proba_cal_sexdom" if weekend_score == "cal" else "proba_raw_sexdom"


def apply_isotonic_vec(p: np.ndarray, x: np.ndarray, y: np.ndarray, floor: float | None = None) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.interp(p, x, y, left=float(y[0]), right=float(y[-1])) if x.size and y.size else p.copy()
    if floor is not None:
        out = np.maximum(out, float(floor))
    return np.clip(out, 0.0, 1.0)


@dataclass(frozen=True)
class Rule:
    bet_type: str
    dow: str
    score_col: str
    cutoff: float
    stake_frac: float
    status: str


def score_bin_ok(score_sel: np.ndarray, profit_sel: np.ndarray) -> Tuple[int, int, bool]:
    score_sel = np.asarray(score_sel, dtype=float)
    profit_sel = np.asarray(profit_sel, dtype=float)
    if score_sel.size == 0:
        return 0, 0, False
    max_bins_by_n = int(score_sel.size // max(MIN_BETS_PER_BIN, 1))
    n_bins_target = int(min(N_SCORE_BINS, max_bins_by_n))
    if n_bins_target < MIN_BINS_FOR_STABILITY:
        return 0, 0, False
    edges = np.unique(np.quantile(score_sel, np.linspace(0.0, 1.0, n_bins_target + 1)))
    if edges.size < 3:
        n_bins = 1
        pos_bins = 1 if float(np.mean(profit_sel)) > 0 else 0
        return n_bins, pos_bins, pos_bins == n_bins
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
    pos_bins = sum(1 for sel in bins if float(np.mean(profit_sel[sel])) > 0)
    if n_bins >= N_SCORE_BINS:
        ok = pos_bins >= MIN_POS_BINS_CAP2
    elif n_bins == 4:
        ok = pos_bins >= 3
    elif n_bins == 3:
        ok = pos_bins >= 2
    else:
        ok = pos_bins == n_bins
    return int(n_bins), int(pos_bins), bool(ok)


def optimize_segment_train(x: pd.DataFrame, score_col: str) -> Rule:
    weeks_all = sorted(x["week"].unique().tolist())
    if len(weeks_all) < MIN_SEG_TRAIN_WEEKS:
        return Rule(bet_type=str(x["bet_type"].iloc[0]), dow=str(x["dow_pt"].iloc[0]), score_col=score_col, cutoff=1.0, stake_frac=0.0, status="too_few_weeks")

    score = pd.to_numeric(x[score_col], errors="coerce").to_numpy(dtype=float)
    roi2 = x["roi_cap2"].to_numpy(dtype=float)
    roi1 = x["roi_cap1"].to_numpy(dtype=float)
    cap = x["house_cap"].to_numpy(dtype=float)
    wk = x["week"].to_numpy()
    d = x["date"].to_numpy()

    rng = np.random.default_rng(SEED + 123 + hash((str(x["dow_pt"].iloc[0]), str(x["bet_type"].iloc[0]), score_col)) % 10_000)
    bb_weights = rng.dirichlet(np.ones(len(weeks_all)), size=BAYES_N)

    def eval_obj_for_cutoff(stake_eff: np.ndarray, cutoff: float) -> Tuple[bool, float, np.ndarray | None]:
        m = np.isfinite(score) & (score >= float(cutoff)) & np.isfinite(roi2)
        if not np.any(m):
            return False, -np.inf, None
        if int(np.sum(m)) < MIN_SELECTED_BETS:
            return False, -np.inf, None
        nonzero_weeks = int(pd.Series(np.ones(int(np.sum(m))), index=wk[m]).groupby(level=0).sum().shape[0])
        if nonzero_weeks < MIN_NONZERO_WEEKS:
            return False, -np.inf, None

        pnl2 = stake_eff[m] * roi2[m]
        w2 = (
            pd.Series(pnl2, index=wk[m])
            .groupby(level=0)
            .sum()
            .reindex(weeks_all, fill_value=0.0)
            .to_numpy(dtype=float)
        )
        mean = float(w2.mean())
        if mean <= 0:
            return False, -np.inf, None
        std = float(w2.std(ddof=1)) if w2.size >= 2 else 0.0
        pneg = float((w2 < 0).mean())
        if pneg > 0.40:
            return False, -np.inf, None
        sharpe = float(mean / std) if std > 0 else (float("inf") if mean > 0 else -float("inf"))
        if np.isfinite(sharpe) and sharpe < MIN_WEEKLY_SHARPE_CAP2:
            return False, -np.inf, None

        pnl1 = stake_eff[m] * roi1[m]
        w1 = (
            pd.Series(pnl1, index=wk[m])
            .groupby(level=0)
            .sum()
            .reindex(weeks_all, fill_value=0.0)
            .to_numpy(dtype=float)
        )
        mean1 = float(w1.mean())
        if mean1 < -0.10 * mean:
            return False, -np.inf, None

        pnl_day = pd.Series(stake_eff[m] * roi2[m], index=d[m]).groupby(level=0).sum().to_numpy(dtype=float)
        if pnl_day.size == 0:
            return False, -np.inf, None
        daily_var = float(np.quantile(pnl_day, DAILY_VAR_Q))
        p_dd = float((pnl_day <= (-MAX_DAILY_DRAWDOWN_FRAC * BANKROLL)).mean())
        if daily_var < -MAX_DAILY_DRAWDOWN_FRAC * BANKROLL:
            return False, -np.inf, None
        if p_dd > MAX_P_DAILY_DD:
            return False, -np.inf, None

        stake_day = pd.Series(stake_eff[m], index=d[m]).groupby(level=0).sum().to_numpy(dtype=float)
        if stake_day.size == 0:
            return False, -np.inf, None
        p80_exp = float(np.quantile(stake_day, DAILY_EXPOSURE_Q))
        if p80_exp > MAX_DAILY_EXPOSURE_FRAC_Q * BANKROLL:
            return False, -np.inf, None
        p95_exp = float(np.quantile(stake_day, 0.95))

        if ENABLE_SCORE_BIN_STABILITY:
            n_bins, pos_bins, ok = score_bin_ok(score[m], stake_eff[m] * roi2[m])
            if not ok:
                return False, -np.inf, None

        post_means = bb_weights @ w2.astype(float)
        p_mean_pos = float(np.mean(post_means > 0))
        if p_mean_pos < MIN_POST_P_MEAN_POS:
            return False, -np.inf, None
        q_obj = float(np.quantile(post_means, POST_Q_OBJ))
        return True, float(q_obj - EXPOSURE_PENALTY * p95_exp), post_means

    best_obj = -np.inf
    best = None
    for f in STAKE_FRACS:
        stake0 = BANKROLL * float(f)
        stake_eff = np.minimum(stake0, cap)
        for c in CUTOFFS:
            ok0, obj0, _ = eval_obj_for_cutoff(stake_eff, float(c))
            if not ok0:
                continue
            if obj0 > best_obj:
                best_obj = float(obj0)
                best = (float(c), float(f))

    if best is None:
        return Rule(bet_type=str(x["bet_type"].iloc[0]), dow=str(x["dow_pt"].iloc[0]), score_col=score_col, cutoff=1.0, stake_frac=0.0, status="no_candidate")
    return Rule(bet_type=str(x["bet_type"].iloc[0]), dow=str(x["dow_pt"].iloc[0]), score_col=score_col, cutoff=best[0], stake_frac=best[1], status="ok")


def apply_rules_on_df(df_any: pd.DataFrame, rules: Dict[str, Rule], alpha: float) -> pd.DataFrame:
    rows = []
    for rule in rules.values():
        if rule.stake_frac <= 0:
            continue
        stake0 = BANKROLL * rule.stake_frac * float(alpha)
        if stake0 <= 0:
            continue
        x = df_any[(df_any["dow_pt"] == rule.dow) & (df_any["bet_type"] == rule.bet_type)].copy()
        if x.empty:
            continue
        score = pd.to_numeric(x[rule.score_col], errors="coerce").to_numpy(dtype=float)
        roi2 = x["roi_cap2"].to_numpy(dtype=float)
        m = np.isfinite(score) & (score >= rule.cutoff) & np.isfinite(roi2)
        if not np.any(m):
            continue
        x = x.iloc[np.where(m)[0]].copy()
        x["stake_eff"] = np.minimum(stake0, x["house_cap"].to_numpy(dtype=float))
        x["profit_cap2"] = x["stake_eff"].to_numpy(dtype=float) * x["roi_cap2"].to_numpy(dtype=float)
        x["rule_key"] = f"{rule.bet_type}|{rule.dow}"
        rows.append(x[["date", "week", "stake_eff", "profit_cap2", "rule_key"]])
    if not rows:
        return df_any.iloc[:0].copy()
    return pd.concat(rows, axis=0, ignore_index=True)


def portfolio_global_constraints_ok(df_train: pd.DataFrame, rules: Dict[str, Rule], alpha: float) -> Tuple[bool, Dict[str, float]]:
    bets = apply_rules_on_df(df_train, rules, alpha=alpha)
    if bets.empty:
        return True, {"p80_exp": 0.0, "daily_var10": 0.0, "p_dd": 0.0, "n_days": 0}
    stake_day = bets.groupby("date")["stake_eff"].sum().to_numpy(dtype=float)
    pnl_day = bets.groupby("date")["profit_cap2"].sum().to_numpy(dtype=float)
    p80_exp = float(np.quantile(stake_day, DAILY_EXPOSURE_Q)) if stake_day.size else 0.0
    daily_var = float(np.quantile(pnl_day, DAILY_VAR_Q)) if pnl_day.size else 0.0
    p_dd = float((pnl_day <= (-MAX_DAILY_DRAWDOWN_FRAC * BANKROLL)).mean()) if pnl_day.size else 0.0
    ok = True
    if p80_exp > MAX_DAILY_EXPOSURE_FRAC_Q * BANKROLL:
        ok = False
    if daily_var < -MAX_DAILY_DRAWDOWN_FRAC * BANKROLL:
        ok = False
    if p_dd > MAX_P_DAILY_DD:
        ok = False
    return ok, {"p80_exp": p80_exp, "daily_var10": daily_var, "p_dd": p_dd, "n_days": int(pnl_day.size)}


def find_alpha_global(df_train: pd.DataFrame, rules: Dict[str, Rule]) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    ok1, m1 = portfolio_global_constraints_ok(df_train, rules, alpha=1.0)
    if ok1:
        return 1.0, m1, m1
    lo, hi = 0.0, 1.0
    best = 0.0
    best_m = None
    for _ in range(24):
        mid = (lo + hi) / 2.0
        ok, mm = portfolio_global_constraints_ok(df_train, rules, alpha=mid)
        if ok:
            best = mid
            best_m = mm
            lo = mid
        else:
            hi = mid
    if best_m is None:
        ok0, m0 = portfolio_global_constraints_ok(df_train, rules, alpha=0.0)
        return 0.0, m1, m0
    return float(best), m1, best_m


def compute_weekly_stats(w: np.ndarray) -> Dict[str, float]:
    x = np.asarray(w, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    m = float(x.mean())
    s = float(x.std(ddof=1)) if x.size > 1 else 0.0
    med = float(np.median(x))
    pneg = float((x < 0).mean())
    sharpe_ann = float((m * 52.0) / (s * math.sqrt(52.0))) if s > 0 else float("nan")
    return {"n": int(x.size), "mean": m, "std": s, "median": med, "pneg": pneg, "sharpe_annual": sharpe_ann}


def run_mode(df_all: pd.DataFrame, weeks: List[str], weekend_score: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_rules_rows = []
    weekly_rows = []
    daily_rows = []
    prev_rules: Dict[str, Rule] = {}

    for i in range(MIN_GLOBAL_TRAIN_WEEKS, len(weeks)):
        w_test = weeks[i]
        train_weeks = weeks[:i]
        df_train = df_all[df_all["week"].isin(train_weeks)].copy()
        df_test = df_all[df_all["week"] == w_test].copy()

        rules: Dict[str, Rule] = {}
        for bet_type in ("FT", "FH"):
            for dow in WEEKDAY_PT:
                sc = segment_score_col(dow, weekend_score=weekend_score)
                x = df_train[(df_train["dow_pt"] == dow) & (df_train["bet_type"] == bet_type)].copy()
                if x.empty:
                    rule = Rule(bet_type=bet_type, dow=dow, score_col=sc, cutoff=1.0, stake_frac=0.0, status="no_data")
                else:
                    rule = optimize_segment_train(x, sc)
                rules[f"{bet_type}|{dow}"] = rule

        alpha, m_at1, m_ata = find_alpha_global(df_train, rules)

        for key, rule in rules.items():
            all_rules_rows.append(
                {
                    "test_week": w_test,
                    "train_weeks": len(train_weeks),
                    "bet_type": rule.bet_type,
                    "dow_pt": rule.dow,
                    "score_col": rule.score_col,
                    "cutoff": rule.cutoff,
                    "stake_frac": rule.stake_frac,
                    "alpha_global": float(alpha),
                    "status": rule.status,
                    "rule_key": key,
                    "train_global_p80_exp_at1": float(m_at1.get("p80_exp", float("nan"))),
                    "train_global_var10_at1": float(m_at1.get("daily_var10", float("nan"))),
                    "train_global_p_dd_at1": float(m_at1.get("p_dd", float("nan"))),
                    "train_global_p80_exp_at_alpha": float(m_ata.get("p80_exp", float("nan"))),
                    "train_global_var10_at_alpha": float(m_ata.get("daily_var10", float("nan"))),
                    "train_global_p_dd_at_alpha": float(m_ata.get("p_dd", float("nan"))),
                }
            )

        bets = apply_rules_on_df(df_test, rules, alpha=alpha)
        stake_sum = float(bets["stake_eff"].sum()) if len(bets) else 0.0
        pnl_sum = float(bets["profit_cap2"].sum()) if len(bets) else 0.0
        roi_on_stake = float(pnl_sum / stake_sum) if stake_sum > 0 else float("nan")
        weekly_rows.append({"week": w_test, "train_weeks": len(train_weeks), "alpha_global": float(alpha), "n_bets": int(len(bets)), "stake_usd": stake_sum, "profit_cap2_usd": pnl_sum, "roi_on_stake_cap2": roi_on_stake})

        if len(bets):
            dd = bets.groupby("date", as_index=False).agg(stake_usd=("stake_eff", "sum"), profit_cap2_usd=("profit_cap2", "sum"))
            dd["week"] = w_test
            dd["alpha_global"] = float(alpha)
            daily_rows.append(dd)

        prev_rules = rules.copy()

    rules_df = pd.DataFrame(all_rules_rows)
    weekly_df = pd.DataFrame(weekly_rows)
    daily_df = pd.concat(daily_rows, axis=0, ignore_index=True) if daily_rows else pd.DataFrame(columns=["date", "stake_usd", "profit_cap2_usd", "week", "alpha_global"])
    return rules_df, weekly_df, daily_df


def write_strategy_md(mode_name: str, weekly_df: pd.DataFrame, rules_df: pd.DataFrame) -> None:
    w = weekly_df["profit_cap2_usd"].to_numpy(dtype=float)
    st = compute_weekly_stats(w)
    rules_df = rules_df.copy()
    rules_df["active"] = (rules_df["status"] == "ok") & (rules_df["stake_frac"] > 0)
    act = rules_df.groupby("rule_key")["active"].mean().sort_values(ascending=False)
    lines = []
    lines.append(f"## {mode_name}\n\n")
    lines.append(f"- Semanas: **{int(st.get('n',0))}**\n")
    lines.append(f"- Lucro total: **USD {float(np.sum(w)):,.1f}**\n")
    lines.append(f"- Lucro médio semanal: **USD {st.get('mean', float('nan')):,.1f}**\n")
    lines.append(f"- Std semanal: **USD {st.get('std', float('nan')):,.1f}**\n")
    lines.append(f"- Sharpe anualizado: **{st.get('sharpe_annual', float('nan')):.3f}**\n")
    lines.append(f"- P(semana<0): **{st.get('pneg', float('nan'))*100:.1f}%**\n\n")
    lines.append("### Taxa de ativação por segmento (mean active)\n\n")
    for k, v in act.items():
        lines.append(f"- {k}: {v*100:.1f}%\n")
    (OUT_DIR / f"oos_walkforward_{mode_name}_strategy.md").write_text("".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])
    df["house_cap"] = df["house_cap"].apply(safe_cap)
    df["week"] = week_key(df["BIA_ApostaUTC"])
    df["date"] = date_key(df["BIA_ApostaUTC"])
    df["roi_raw"] = pd.to_numeric(df["ROI Real"], errors="coerce").astype(float)
    df["roi_cap2"] = np.minimum(df["roi_raw"].to_numpy(dtype=float), 2.0)
    df["roi_cap1"] = np.minimum(df["roi_raw"].to_numpy(dtype=float), 1.0)

    # cria proba_cal_sexdom se possível
    if CALIB_SEXDOM.exists() and "proba_raw_sexdom" in df.columns:
        calib = json.loads(CALIB_SEXDOM.read_text(encoding="utf-8"))
        x = np.asarray(calib.get("isotonic", {}).get("x", []), dtype=float)
        y = np.asarray(calib.get("isotonic", {}).get("y", []), dtype=float)
        df["proba_cal_sexdom"] = apply_isotonic_vec(pd.to_numeric(df["proba_raw_sexdom"], errors="coerce").to_numpy(dtype=float), x=x, y=y, floor=0.005)
    else:
        df["proba_cal_sexdom"] = np.nan

    weeks = sorted(df["week"].unique().tolist())
    if len(weeks) < (MIN_GLOBAL_TRAIN_WEEKS + 3):
        raise SystemExit(f"Poucas semanas no dataset: {len(weeks)}")

    outputs = {}
    for weekend_score in ("raw", "cal"):
        mode = f"global_bayes_weekend_{weekend_score}"
        rules_df, weekly_df, daily_df = run_mode(df, weeks, weekend_score=weekend_score)
        rules_df.to_csv(OUT_DIR / f"oos_walkforward_{mode}_selected_rules.csv", index=False)
        weekly_df.to_csv(OUT_DIR / f"oos_walkforward_{mode}_weekly.csv", index=False)
        daily_df.to_csv(OUT_DIR / f"oos_walkforward_{mode}_daily.csv", index=False)
        write_strategy_md(mode, weekly_df, rules_df)
        outputs[weekend_score] = weekly_df

    # comparison md
    lines = []
    lines.append("## Comparação OOS (global_bayes): fim de semana em proba_raw vs proba_cal\n\n")
    for k in ("raw", "cal"):
        w = outputs[k]["profit_cap2_usd"].to_numpy(dtype=float)
        st = compute_weekly_stats(w)
        lines.append(f"### weekend_score={k}\n")
        lines.append(f"- Lucro total: USD {float(np.sum(w)):,.1f}\n")
        lines.append(f"- Mean semanal: USD {st.get('mean', float('nan')):,.1f}\n")
        lines.append(f"- Std semanal: USD {st.get('std', float('nan')):,.1f}\n")
        lines.append(f"- Sharpe anualizado: {st.get('sharpe_annual', float('nan')):.3f}\n")
        lines.append(f"- P(semana<0): {st.get('pneg', float('nan'))*100:.1f}%\n\n")
    (OUT_DIR / "oos_walkforward_global_bayes_weekend_score_comparison.md").write_text("".join(lines), encoding="utf-8")

    print(str(OUT_DIR / "oos_walkforward_global_bayes_weekend_score_comparison.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

