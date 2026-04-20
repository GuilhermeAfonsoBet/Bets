#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parte 2 — Thompson Sampling / bandit (sem misturar com o modelo hierárquico MCMC).

Objetivo:
- Usar um bandit com Thompson Sampling para decidir "quanto apostar" em cada segmento (FT/FH × DoW),
  de forma adaptativa no tempo, com shrinkage (empirical Bayes) e desconto temporal (não-estacionariedade).
- Manter o processo de geração de regras por segmento (cutoff + stake_frac_base) igual ao otimizador
  (bayes_select ou clássico), e deixar o bandit atuar APENAS nos pesos/escala por segmento.
- Respeitar o risco global do portfólio via fator α (0..1) encontrado no treino de cada passo.

Entrada:
- /workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv

Saídas:
- /workspace/analysis_proba_raw/pro_portfolio_all/thompson_global_weekly.csv
- /workspace/analysis_proba_raw/pro_portfolio_all/thompson_global_daily.csv
- /workspace/analysis_proba_raw/pro_portfolio_all/thompson_global_weights.csv
- /workspace/analysis_proba_raw/pro_portfolio_all/thompson_global_report.md

Notas importantes:
- "Contextual": aqui o contexto é a própria incerteza/volume (n_bets) e recência (decay),
  que entram na posterior empírica por segmento. Não usa features exógenas (porque não temos
  um contexto observável ex-ante de cada semana além do histórico).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import math
import numpy as np
import pandas as pd


SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")
OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")

# sizing / constraints
BANKROLL = 2300.0
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

MIN_GLOBAL_TRAIN_WEEKS = 10
MIN_SEG_TRAIN_WEEKS = 6

SEED = 7

# regra base por segmento (cutoff/stake): usar seleção bayesiana conservadora da etapa 1
BAYES_RULE_SELECT = True
BAYES_N = 4_000
MIN_POST_P_MEAN_POS = 0.75
POST_Q_OBJ = 0.05
EXPOSURE_PENALTY = 0.001

# Bandit / Thompson
DECAY = 0.97  # desconto por semana (recência). 1.0 = sem desconto.
P_POS_MIN = 0.55  # só liga segmento se P(mu>0) >= este limiar
MU_SCALE = 0.20   # escala para mapear mu_sample -> peso [0..1] via clip(mu/MU_SCALE)

N_BOOT = 20_000  # CI bootstrap para lucro semanal do portfólio TS

WEEKDAY_PT = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo"]


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


@dataclass(frozen=True)
class Rule:
    bet_type: str
    dow: str
    score_col: str
    cutoff: float
    stake_frac: float
    status: str


def score_bin_ok(score_sel: np.ndarray, profit_sel: np.ndarray) -> bool:
    score_sel = np.asarray(score_sel, dtype=float)
    profit_sel = np.asarray(profit_sel, dtype=float)
    if score_sel.size == 0:
        return False
    edges = np.unique(np.quantile(score_sel, np.linspace(0.0, 1.0, N_SCORE_BINS + 1)))
    if edges.size < 3:
        return float(np.mean(profit_sel)) > 0
    bins = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b == edges[-1]:
            sel = (score_sel >= a) & (score_sel <= b)
        else:
            sel = (score_sel >= a) & (score_sel < b)
        if np.any(sel):
            bins.append(sel)
    n_bins = len(bins)
    pos_bins = sum(1 for sel in bins if float(np.mean(profit_sel[sel])) > 0)
    if n_bins >= N_SCORE_BINS:
        return pos_bins >= MIN_POS_BINS_CAP2
    if n_bins == 4:
        return pos_bins >= 3
    if n_bins == 3:
        return pos_bins >= 2
    return pos_bins == n_bins


def optimize_segment_train(x: pd.DataFrame, score_col: str, bayes_select: bool) -> Rule:
    weeks_all = sorted(x["week"].unique().tolist())
    if len(weeks_all) < MIN_SEG_TRAIN_WEEKS:
        return Rule(bet_type=str(x["bet_type"].iloc[0]), dow=str(x["dow_pt"].iloc[0]), score_col=score_col, cutoff=1.0, stake_frac=0.0, status="too_few_weeks")

    score = pd.to_numeric(x[score_col], errors="coerce").to_numpy(dtype=float)
    roi2 = x["roi_cap2"].to_numpy(dtype=float)
    roi1 = x["roi_cap1"].to_numpy(dtype=float)
    cap = x["house_cap"].to_numpy(dtype=float)
    wk = x["week"].to_numpy()
    d = x["date"].to_numpy()

    best_obj = -np.inf
    best = None
    for f in STAKE_FRACS:
        stake0 = BANKROLL * float(f)
        stake_eff = np.minimum(stake0, cap)
        for c in CUTOFFS:
            m = np.isfinite(score) & (score >= c) & np.isfinite(roi2)
            if not np.any(m):
                continue

            pnl2 = stake_eff[m] * roi2[m]
            w2 = (
                pd.Series(pnl2, index=wk[m]).groupby(level=0).sum().reindex(weeks_all, fill_value=0.0).to_numpy(dtype=float)
            )
            mean = float(w2.mean())
            if mean <= 0:
                continue
            std = float(w2.std(ddof=1)) if w2.size >= 2 else 0.0
            pneg = float((w2 < 0).mean())
            if pneg > 0.40:
                continue
            sharpe = float(mean / std) if std > 0 else (float("inf") if mean > 0 else -float("inf"))
            if np.isfinite(sharpe) and sharpe < MIN_WEEKLY_SHARPE_CAP2:
                continue

            pnl1 = stake_eff[m] * roi1[m]
            w1 = (
                pd.Series(pnl1, index=wk[m]).groupby(level=0).sum().reindex(weeks_all, fill_value=0.0).to_numpy(dtype=float)
            )
            mean1 = float(w1.mean())
            if mean1 < -0.10 * mean:
                continue

            pnl_day = pd.Series(stake_eff[m] * roi2[m], index=d[m]).groupby(level=0).sum().to_numpy(dtype=float)
            if pnl_day.size == 0:
                continue
            daily_var = float(np.quantile(pnl_day, DAILY_VAR_Q))
            p_dd = float((pnl_day <= (-MAX_DAILY_DRAWDOWN_FRAC * BANKROLL)).mean())
            if daily_var < -MAX_DAILY_DRAWDOWN_FRAC * BANKROLL:
                continue
            if p_dd > MAX_P_DAILY_DD:
                continue

            stake_day = pd.Series(stake_eff[m], index=d[m]).groupby(level=0).sum().to_numpy(dtype=float)
            if stake_day.size == 0:
                continue
            p80_exp = float(np.quantile(stake_day, DAILY_EXPOSURE_Q))
            if p80_exp > MAX_DAILY_EXPOSURE_FRAC_Q * BANKROLL:
                continue
            p95_exp = float(np.quantile(stake_day, 0.95))

            if ENABLE_SCORE_BIN_STABILITY and not score_bin_ok(score[m], stake_eff[m] * roi2[m]):
                continue

            if not bayes_select:
                obj = mean - 0.25 * std - EXPOSURE_PENALTY * p95_exp
            else:
                rng = np.random.default_rng(SEED + 123)
                W = w2.astype(float)
                weights = rng.dirichlet(np.ones(W.size), size=BAYES_N)
                post_means = weights @ W
                p_mean_pos = float(np.mean(post_means > 0))
                if p_mean_pos < MIN_POST_P_MEAN_POS:
                    continue
                q_obj = float(np.quantile(post_means, POST_Q_OBJ))
                obj = q_obj - EXPOSURE_PENALTY * p95_exp

            if obj > best_obj:
                best_obj = obj
                best = (float(c), float(f))

    if best is None:
        return Rule(bet_type=str(x["bet_type"].iloc[0]), dow=str(x["dow_pt"].iloc[0]), score_col=score_col, cutoff=1.0, stake_frac=0.0, status="no_candidate")
    return Rule(bet_type=str(x["bet_type"].iloc[0]), dow=str(x["dow_pt"].iloc[0]), score_col=score_col, cutoff=best[0], stake_frac=best[1], status="ok")


def apply_rules_on_df(df_any: pd.DataFrame, rules: Dict[str, Rule], weights: Dict[str, float], alpha: float) -> pd.DataFrame:
    rows = []
    for k, rule in rules.items():
        if rule.stake_frac <= 0:
            continue
        wgt = float(weights.get(k, 0.0))
        if wgt <= 0:
            continue
        stake0 = BANKROLL * rule.stake_frac * wgt * float(alpha)
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
        x["rule_key"] = k
        rows.append(x[["date", "week", "stake_eff", "profit_cap2", "rule_key"]])
    return pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame(columns=["date", "week", "stake_eff", "profit_cap2", "rule_key"])


def portfolio_constraints(df_train: pd.DataFrame, rules: Dict[str, Rule], weights: Dict[str, float], alpha: float) -> bool:
    bets = apply_rules_on_df(df_train, rules, weights, alpha=alpha)
    if bets.empty:
        return True
    stake_day = bets.groupby("date")["stake_eff"].sum().to_numpy(dtype=float)
    pnl_day = bets.groupby("date")["profit_cap2"].sum().to_numpy(dtype=float)
    p80_exp = float(np.quantile(stake_day, DAILY_EXPOSURE_Q)) if stake_day.size else 0.0
    daily_var = float(np.quantile(pnl_day, DAILY_VAR_Q)) if pnl_day.size else 0.0
    p_dd = float((pnl_day <= (-MAX_DAILY_DRAWDOWN_FRAC * BANKROLL)).mean()) if pnl_day.size else 0.0
    if p80_exp > MAX_DAILY_EXPOSURE_FRAC_Q * BANKROLL:
        return False
    if daily_var < -MAX_DAILY_DRAWDOWN_FRAC * BANKROLL:
        return False
    if p_dd > MAX_P_DAILY_DD:
        return False
    return True


def find_alpha(df_train: pd.DataFrame, rules: Dict[str, Rule], weights: Dict[str, float]) -> float:
    if portfolio_constraints(df_train, rules, weights, alpha=1.0):
        return 1.0
    lo, hi = 0.0, 1.0
    best = 0.0
    for _ in range(24):
        mid = (lo + hi) / 2.0
        if portfolio_constraints(df_train, rules, weights, alpha=mid):
            best = mid
            lo = mid
        else:
            hi = mid
    return float(best)


def bootstrap_ci_mean(x: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    obs = float(a.mean())
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    boot = a[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return obs, float(lo), float(hi)


def fit_empirical_bayes(obs: Dict[str, List[Tuple[float, float]]]) -> Tuple[float, float, float, Dict[str, Tuple[float, float]]]:
    """
    obs[k] = list of (roi_week, w_info) já ponderado por recência (decay) e volume.
    Returns (mu0, tau2, sig2, post_params[k]=(mean,var)).
    """
    keys = list(obs.keys())
    # compute per-key sufficient stats
    sumw = {}
    sumwy = {}
    for k in keys:
        arr = obs[k]
        sw = float(sum(w for _, w in arr))
        sy = float(sum(y * w for y, w in arr))
        sumw[k] = sw
        sumwy[k] = sy

    # global mean (weighted)
    SW = float(sum(sumw.values()))
    if SW <= 0:
        mu0 = 0.0
    else:
        mu0 = float(sum(sumwy.values()) / SW)

    # per-segment means
    means = []
    ws = []
    for k in keys:
        if sumw[k] > 0:
            means.append(sumwy[k] / sumw[k])
            ws.append(sumw[k])
    means = np.asarray(means, float)
    ws = np.asarray(ws, float)

    # estimate sigma2 from within-segment residuals
    sse = 0.0
    n_eff = 0.0
    for k in keys:
        for y, w in obs[k]:
            mk = (sumwy[k] / sumw[k]) if sumw[k] > 0 else mu0
            sse += float(w * (y - mk) ** 2)
            n_eff += float(w)
    sig2 = float(sse / n_eff) if n_eff > 0 else 0.10
    sig2 = max(sig2, 1e-6)

    # between variance tau2 (method-of-moments rough)
    if means.size >= 2 and float(ws.sum()) > 0:
        mu_bar = float(np.average(means, weights=ws))
        vb = float(np.average((means - mu_bar) ** 2, weights=ws))
        # subtract an average sampling variance component
        avg_var = float(np.average(sig2 / ws, weights=ws))
        tau2 = max(vb - avg_var, 1e-6)
    else:
        tau2 = 0.05

    post = {}
    for k in keys:
        sw = sumw[k]
        if sw <= 0:
            post[k] = (mu0, tau2)
            continue
        prec = (sw / sig2) + (1.0 / tau2)
        var = 1.0 / prec
        mean = var * ((sumwy[k] / sig2) + (mu0 / tau2))
        post[k] = (float(mean), float(var))
    return float(mu0), float(tau2), float(sig2), post


def segment_weekly_roi(df_train: pd.DataFrame, rule_key: str, rule: Rule) -> pd.DataFrame:
    """
    Retorna série semanal (week) com ROI_week (profit/stake), n_bets e stake_usd
    para um único segmento, usando a regra base (cutoff + stake_frac).
    """
    if rule.stake_frac <= 0 or rule.status != "ok":
        return pd.DataFrame(columns=["week", "roi_week", "n_bets", "stake_usd"])
    x = df_train[(df_train["dow_pt"] == rule.dow) & (df_train["bet_type"] == rule.bet_type)].copy()
    if x.empty:
        return pd.DataFrame(columns=["week", "roi_week", "n_bets", "stake_usd"])
    score = pd.to_numeric(x[rule.score_col], errors="coerce").to_numpy(dtype=float)
    roi2 = x["roi_cap2"].to_numpy(dtype=float)
    m = np.isfinite(score) & (score >= rule.cutoff) & np.isfinite(roi2)
    if not np.any(m):
        return pd.DataFrame(columns=["week", "roi_week", "n_bets", "stake_usd"])
    x = x.iloc[np.where(m)[0]].copy()
    stake0 = BANKROLL * rule.stake_frac
    stake_eff = np.minimum(stake0, x["house_cap"].to_numpy(dtype=float))
    profit = stake_eff * x["roi_cap2"].to_numpy(dtype=float)
    tmp = pd.DataFrame({"week": x["week"].to_numpy(), "stake": stake_eff, "profit": profit})
    g = tmp.groupby("week", as_index=False).agg(stake_usd=("stake", "sum"), profit_usd=("profit", "sum"), n_bets=("profit", "size"))
    g["roi_week"] = np.where(g["stake_usd"] > 0, g["profit_usd"] / g["stake_usd"], np.nan)
    return g[["week", "roi_week", "n_bets", "stake_usd"]]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])
    df["house_cap"] = df["house_cap"].apply(safe_cap)
    df["week"] = week_key(df["BIA_ApostaUTC"])
    df["date"] = date_key(df["BIA_ApostaUTC"])
    df["roi_raw"] = pd.to_numeric(df["ROI Real"], errors="coerce").astype(float)
    df["roi_cap2"] = np.minimum(df["roi_raw"].to_numpy(dtype=float), 2.0)
    df["roi_cap1"] = np.minimum(df["roi_raw"].to_numpy(dtype=float), 1.0)

    weeks = sorted(df["week"].unique().tolist())
    if len(weeks) < (MIN_GLOBAL_TRAIN_WEEKS + 3):
        raise SystemExit(f"Poucas semanas no dataset: {len(weeks)}")

    rng = np.random.default_rng(SEED)

    weekly_rows = []
    daily_rows = []
    weight_rows = []

    # walk-forward
    for i in range(MIN_GLOBAL_TRAIN_WEEKS, len(weeks)):
        w_test = weeks[i]
        train_weeks = weeks[:i]
        df_train = df[df["week"].isin(train_weeks)].copy()
        df_test = df[df["week"] == w_test].copy()

        # 1) generate base rules per segment from train (no leak)
        rules: Dict[str, Rule] = {}
        for bet_type in ("FT", "FH"):
            for dow in WEEKDAY_PT:
                sc = segment_score_col(dow)
                x = df_train[(df_train["dow_pt"] == dow) & (df_train["bet_type"] == bet_type)].copy()
                key = f"{bet_type}|{dow}"
                if x.empty:
                    rules[key] = Rule(bet_type=bet_type, dow=dow, score_col=sc, cutoff=1.0, stake_frac=0.0, status="no_data")
                else:
                    rules[key] = optimize_segment_train(x, sc, bayes_select=BAYES_RULE_SELECT)

        # 2) construir observações (ROI semanal por segmento) no treino e aplicar decay por recência
        # age=0 => semana imediatamente anterior ao teste
        obs: Dict[str, List[Tuple[float, float]]] = {k: [] for k in rules.keys()}
        week_to_idx = {w: j for j, w in enumerate(train_weeks)}
        for k, rule in rules.items():
            g = segment_weekly_roi(df_train, k, rule)
            if g.empty:
                continue
            for _, r in g.iterrows():
                wk = str(r["week"])
                if wk not in week_to_idx:
                    continue
                age = (len(train_weeks) - 1) - week_to_idx[wk]
                decay = float(DECAY ** age)
                y = float(r["roi_week"])
                w_info = float(r["n_bets"]) * decay
                if np.isfinite(y) and w_info > 0:
                    obs[k].append((y, w_info))

        # 3) fit EB hierarchical posterior from obs
        mu0, tau2, sig2, post = fit_empirical_bayes(obs)

        # 4) Thompson draw and convert to weights
        weights: Dict[str, float] = {}
        for k in rules:
            m, v = post.get(k, (mu0, tau2))
            mu_samp = float(rng.normal(loc=m, scale=np.sqrt(max(v, 1e-9))))
            # analytic P(mu>0) under Normal approx
            z = m / np.sqrt(max(v, 1e-9))
            p_pos = float(0.5 * (1.0 + math.erf(float(z) / float(np.sqrt(2.0)))))
            if p_pos < P_POS_MIN:
                wgt = 0.0
            else:
                wgt = float(np.clip(mu_samp / MU_SCALE, 0.0, 1.0))
            weights[k] = wgt
            weight_rows.append(
                {
                    "week": w_test,
                    "rule_key": k,
                    "post_mean": m,
                    "post_var": v,
                    "p_pos": p_pos,
                    "mu_sample": mu_samp,
                    "weight": wgt,
                    "rule_cutoff": rules[k].cutoff,
                    "rule_stake_frac_base": rules[k].stake_frac,
                    "rule_status": rules[k].status,
                }
            )

        # 5) find global alpha to satisfy constraints on TRAIN
        alpha = find_alpha(df_train, rules, weights)

        # 6) apply on TEST
        bets = apply_rules_on_df(df_test, rules, weights, alpha=alpha)
        stake_sum = float(bets["stake_eff"].sum()) if len(bets) else 0.0
        pnl_sum = float(bets["profit_cap2"].sum()) if len(bets) else 0.0
        n_bets = int(len(bets))
        roi_on_stake = float(pnl_sum / stake_sum) if stake_sum > 0 else float("nan")
        weekly_rows.append(
            {
                "week": w_test,
                "train_weeks": len(train_weeks),
                "alpha_global": alpha,
                "mu0": mu0,
                "tau": float(np.sqrt(tau2)),
                "sigma": float(np.sqrt(sig2)),
                "n_bets": n_bets,
                "stake_usd": stake_sum,
                "profit_cap2_usd": pnl_sum,
                "roi_on_stake_cap2": roi_on_stake,
            }
        )

        if len(bets):
            dd = bets.groupby("date", as_index=False).agg(stake_usd=("stake_eff", "sum"), profit_cap2_usd=("profit_cap2", "sum"))
            dd["week"] = w_test
            dd["alpha_global"] = alpha
            daily_rows.append(dd)

            # Nota: não atualizamos "história" incremental; o posterior é sempre recalculado
            # a partir do treino expanding + decay. Isso evita cold-start e mantém TS honesto no tempo.

    weekly_df = pd.DataFrame(weekly_rows)
    daily_df = pd.concat(daily_rows, axis=0, ignore_index=True) if daily_rows else pd.DataFrame(columns=["date", "stake_usd", "profit_cap2_usd", "week", "alpha_global"])
    weights_df = pd.DataFrame(weight_rows)

    weekly_df.to_csv(OUT_DIR / "thompson_global_weekly.csv", index=False)
    daily_df.to_csv(OUT_DIR / "thompson_global_daily.csv", index=False)
    weights_df.to_csv(OUT_DIR / "thompson_global_weights.csv", index=False)

    # summary
    w = weekly_df["profit_cap2_usd"].to_numpy(dtype=float)
    mean_w, lo_w, hi_w = bootstrap_ci_mean(w, n_boot=N_BOOT, seed=SEED + 99)
    pneg = float((w < 0).mean())
    std = float(np.std(w, ddof=1)) if w.size >= 2 else 0.0
    stake_tot = float(weekly_df["stake_usd"].sum())
    pnl_tot = float(weekly_df["profit_cap2_usd"].sum())
    roi_tot = float(pnl_tot / stake_tot) if stake_tot > 0 else float("nan")

    # OOS risk
    if not daily_df.empty:
        stake_day = daily_df["stake_usd"].to_numpy(dtype=float)
        pnl_day = daily_df["profit_cap2_usd"].to_numpy(dtype=float)
        oos_p80_exp = float(np.quantile(stake_day, DAILY_EXPOSURE_Q)) if stake_day.size else float("nan")
        oos_var10 = float(np.quantile(pnl_day, DAILY_VAR_Q)) if pnl_day.size else float("nan")
        oos_p_dd = float((pnl_day <= (-MAX_DAILY_DRAWDOWN_FRAC * BANKROLL)).mean()) if pnl_day.size else float("nan")
    else:
        oos_p80_exp, oos_var10, oos_p_dd = float("nan"), float("nan"), float("nan")

    # selection stability: avg weight & active rate
    weights_df["active"] = weights_df["weight"] > 0
    agg = (
        weights_df.groupby("rule_key", as_index=False)
        .agg(active_rate=("active", "mean"), mean_weight=("weight", "mean"), mean_p_pos=("p_pos", "mean"))
        .sort_values(["active_rate", "mean_weight"], ascending=False)
    )
    agg.to_csv(OUT_DIR / "thompson_global_weights_summary.csv", index=False)

    lines = []
    lines.append("## Thompson Sampling (bandit) — portfólio com risco global\n")
    lines.append(f"- Decay={DECAY:.3f}; limiar P(mu>0)>={P_POS_MIN:.2f}; MU_SCALE={MU_SCALE:.2f}\n")
    lines.append(f"- Regras base por segmento: bayes_select={BAYES_RULE_SELECT}\n")
    lines.append("\n### Performance OOS (cap2)\n")
    lines.append(f"- PnL semanal médio (bootstrap IC95%): USD {mean_w:.1f} (IC95% {lo_w:.1f}..{hi_w:.1f})\n")
    lines.append(f"- Std semanal: USD {std:.1f}; P(semana<0)={pneg*100:.1f}%\n")
    lines.append(f"- ROI on stake (ponderado): {roi_tot:.4f}\n")
    lines.append("\n### Risco no OOS (teste)\n")
    lines.append(
        f"- p80(soma stakes/dia)=USD {oos_p80_exp:.0f} (limite=USD {MAX_DAILY_EXPOSURE_FRAC_Q*BANKROLL:.0f})\n"
        f"- VaR10%(PnL diário)=USD {oos_var10:.1f} (limite >= USD {-MAX_DAILY_DRAWDOWN_FRAC*BANKROLL:.0f})\n"
        f"- P(PnL diário <= -25% banca)={oos_p_dd*100:.1f}% (limite <= {MAX_P_DAILY_DD*100:.0f}%)\n"
    )
    lines.append("\n### Segmentos mais usados (peso médio)\n")
    for _, r in agg.head(10).iterrows():
        lines.append(
            f"- **{r['rule_key']}**: active_rate={r['active_rate']*100:.1f}%, mean_weight={r['mean_weight']:.3f}, mean_Ppos={r['mean_p_pos']*100:.1f}%\n"
        )
    lines.append("\n### Arquivos\n")
    lines.append("- `analysis_proba_raw/pro_portfolio_all/thompson_global_weekly.csv`\n")
    lines.append("- `analysis_proba_raw/pro_portfolio_all/thompson_global_daily.csv`\n")
    lines.append("- `analysis_proba_raw/pro_portfolio_all/thompson_global_weights.csv`\n")
    lines.append("- `analysis_proba_raw/pro_portfolio_all/thompson_global_weights_summary.csv`\n")

    (OUT_DIR / "thompson_global_report.md").write_text("".join(lines), encoding="utf-8")
    print(str(OUT_DIR / "thompson_global_report.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

