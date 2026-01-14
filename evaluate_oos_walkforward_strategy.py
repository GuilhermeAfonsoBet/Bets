#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validação OOS robusta (walk-forward) da estratégia COMPLETA por combinação:
  (DoW x FT/FH x score>=cutoff x stake_frac)

Como funciona (honesto no tempo):
- Ordena semanas disponíveis (W-SUN).
- Para cada semana t (como "teste"):
    - Treino = todas as semanas anteriores (expanding window), com mínimo de MIN_TRAIN_WEEKS.
    - Para cada segmento (DoW x tipo FT/FH), re-otimiza (cutoff, stake_frac) em grid,
      usando apenas dados do treino daquele segmento, com as mesmas constraints baratas:
        - mean_week_cap2 > 0
        - pneg_week_cap2 <= 0.40
        - cap1_mean_week >= -0.10 * mean_week_cap2 (sanidade)
        - VaR10%(PnL_dia) >= -25% banca e P(PnL_dia <= -25% banca) <= 10%
        - p80(soma_stakes_no_dia) <= 70% banca
        - sharpe semanal (cap2) >= 0.10 (se std>0)
        - estabilidade por bins de score (mesma regra do otimizador) opcional (ligada por default)
      Objetivo: mean_week - 0.25*std_week - 0.001*p95_daily_exposure
    - Aplica as regras escolhidas na semana t e soma o PnL OOS por segmento e total.

Saídas (em /workspace/analysis_proba_raw/pro_portfolio_all/):
  - oos_walkforward_weekly.csv           (PnL semanal OOS do portfólio + stake/ROI)
  - oos_walkforward_weekly_by_segment.csv (PnL semanal OOS por segmento)
  - oos_walkforward_selected_rules.csv    (regras escolhidas em cada semana)
  - oos_walkforward_strategy.md           (resumo executivo com métricas e ICs por bootstrap)
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

# sizing / constraints (iguais ao portfólio pro_all)
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

# Para o walk-forward: exigimos um mínimo de histórico global para começar,
# mas cada segmento pode ter menos semanas (como no otimizador original, que exigia ~6).
MIN_GLOBAL_TRAIN_WEEKS = 10
MIN_SEG_TRAIN_WEEKS = 6

N_BOOT = 20_000
SEED = 7

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


def score_bin_ok(score_sel: np.ndarray, profit_sel: np.ndarray) -> Tuple[int, int, bool]:
    # returns (n_bins, pos_bins, ok)
    score_sel = np.asarray(score_sel, dtype=float)
    profit_sel = np.asarray(profit_sel, dtype=float)
    if score_sel.size == 0:
        return 0, 0, False
    edges = np.unique(np.quantile(score_sel, np.linspace(0.0, 1.0, N_SCORE_BINS + 1)))
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
    """
    x já vem filtrado para um único (dow, bet_type) e apenas semanas de treino.
    Retorna a melhor regra (ou stake_frac=0 com status).
    """
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

            # weekly pnl (cap2) aligned to weeks_all (fill 0 for empty weeks)
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
                continue
            std = float(w2.std(ddof=1)) if w2.size >= 2 else 0.0
            pneg = float((w2 < 0).mean())
            if pneg > 0.40:
                continue
            sharpe = float(mean / std) if std > 0 else (float("inf") if mean > 0 else -float("inf"))
            if np.isfinite(sharpe) and sharpe < MIN_WEEKLY_SHARPE_CAP2:
                continue

            # cap1 sanity
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
                continue

            # daily risk constraints (cap2)
            pnl_day = pd.Series(stake_eff[m] * roi2[m], index=d[m]).groupby(level=0).sum().to_numpy(dtype=float)
            if pnl_day.size == 0:
                continue
            daily_var = float(np.quantile(pnl_day, DAILY_VAR_Q))
            p_dd = float((pnl_day <= (-MAX_DAILY_DRAWDOWN_FRAC * BANKROLL)).mean())
            if daily_var < -MAX_DAILY_DRAWDOWN_FRAC * BANKROLL:
                continue
            if p_dd > MAX_P_DAILY_DD:
                continue

            # exposure constraint (p80 sum stakes/day)
            stake_day = pd.Series(stake_eff[m], index=d[m]).groupby(level=0).sum().to_numpy(dtype=float)
            if stake_day.size == 0:
                continue
            p80_exp = float(np.quantile(stake_day, DAILY_EXPOSURE_Q))
            if p80_exp > MAX_DAILY_EXPOSURE_FRAC_Q * BANKROLL:
                continue
            p95_exp = float(np.quantile(stake_day, 0.95))

            # score-bin stability (cap2), opcional
            if ENABLE_SCORE_BIN_STABILITY:
                n_bins, pos_bins, ok = score_bin_ok(score[m], stake_eff[m] * roi2[m])
                if not ok:
                    continue

            obj = mean - 0.25 * std - 0.001 * p95_exp
            if obj > best_obj:
                best_obj = obj
                best = (float(c), float(f))

    if best is None:
        return Rule(bet_type=str(x["bet_type"].iloc[0]), dow=str(x["dow_pt"].iloc[0]), score_col=score_col, cutoff=1.0, stake_frac=0.0, status="no_candidate")

    return Rule(bet_type=str(x["bet_type"].iloc[0]), dow=str(x["dow_pt"].iloc[0]), score_col=score_col, cutoff=best[0], stake_frac=best[1], status="ok")


def apply_rule_on_week(df_week: pd.DataFrame, rule: Rule) -> pd.DataFrame:
    if rule.stake_frac <= 0:
        return df_week.iloc[:0].copy()
    stake0 = BANKROLL * rule.stake_frac
    x = df_week[(df_week["dow_pt"] == rule.dow) & (df_week["bet_type"] == rule.bet_type)].copy()
    if x.empty:
        return x
    score = pd.to_numeric(x[rule.score_col], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(score) & (score >= rule.cutoff) & np.isfinite(x["roi_cap2"].to_numpy(dtype=float))
    x = x.iloc[np.where(m)[0]].copy()
    if x.empty:
        return x
    x["stake_eff"] = np.minimum(stake0, x["house_cap"].to_numpy(dtype=float))
    x["profit_cap2"] = x["stake_eff"].to_numpy(dtype=float) * x["roi_cap2"].to_numpy(dtype=float)
    x["rule_cutoff"] = rule.cutoff
    x["rule_stake_frac"] = rule.stake_frac
    x["rule_score_col"] = rule.score_col
    x["rule_status"] = rule.status
    x["rule_key"] = f"{rule.bet_type}|{rule.dow}"
    return x


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

    all_rules_rows = []
    weekly_rows = []
    weekly_seg_rows = []

    # walk-forward por semana disponível
    for i in range(MIN_GLOBAL_TRAIN_WEEKS, len(weeks)):
        w_test = weeks[i]
        train_weeks = weeks[:i]

        df_train = df[df["week"].isin(train_weeks)].copy()
        df_test = df[df["week"] == w_test].copy()

        # otimiza regras no treino (para cada segmento)
        rules: Dict[str, Rule] = {}
        for bet_type in ("FT", "FH"):
            for dow in WEEKDAY_PT:
                sc = segment_score_col(dow)
                x = df_train[(df_train["dow_pt"] == dow) & (df_train["bet_type"] == bet_type)].copy()
                if x.empty:
                    rule = Rule(bet_type=bet_type, dow=dow, score_col=sc, cutoff=1.0, stake_frac=0.0, status="no_data")
                else:
                    rule = optimize_segment_train(x, sc)
                rules[f"{bet_type}|{dow}"] = rule
                all_rules_rows.append(
                    {
                        "test_week": w_test,
                        "train_weeks": len(train_weeks),
                        "bet_type": bet_type,
                        "dow_pt": dow,
                        "score_col": sc,
                        "cutoff": rule.cutoff,
                        "stake_frac": rule.stake_frac,
                        "status": rule.status,
                    }
                )

        # aplica no teste e agrega
        bets_rows = []
        for key, rule in rules.items():
            xb = apply_rule_on_week(df_test, rule)
            if xb.empty:
                continue
            bets_rows.append(xb)

        if bets_rows:
            bets = pd.concat(bets_rows, axis=0, ignore_index=True)
        else:
            bets = df_test.iloc[:0].copy()
            bets["stake_eff"] = []
            bets["profit_cap2"] = []
            bets["rule_key"] = []

        # total semanal
        stake_sum = float(bets["stake_eff"].sum()) if len(bets) else 0.0
        pnl_sum = float(bets["profit_cap2"].sum()) if len(bets) else 0.0
        roi_on_stake = float(pnl_sum / stake_sum) if stake_sum > 0 else float("nan")
        n_bets = int(len(bets))

        weekly_rows.append(
            {
                "week": w_test,
                "train_weeks": len(train_weeks),
                "n_bets": n_bets,
                "stake_usd": stake_sum,
                "profit_cap2_usd": pnl_sum,
                "roi_on_stake_cap2": roi_on_stake,
            }
        )

        # por segmento
        if len(bets):
            g = bets.groupby("rule_key", as_index=False).agg(n_bets=("profit_cap2", "size"), stake_usd=("stake_eff", "sum"), profit_cap2_usd=("profit_cap2", "sum"))
            for _, r in g.iterrows():
                weekly_seg_rows.append(
                    {
                        "week": w_test,
                        "rule_key": r["rule_key"],
                        "n_bets": int(r["n_bets"]),
                        "stake_usd": float(r["stake_usd"]),
                        "profit_cap2_usd": float(r["profit_cap2_usd"]),
                        "roi_on_stake_cap2": float(r["profit_cap2_usd"] / r["stake_usd"]) if float(r["stake_usd"]) > 0 else float("nan"),
                    }
                )
        else:
            # sem bets: ainda registramos zeros para facilitar plots posteriores (opcional)
            pass

    rules_df = pd.DataFrame(all_rules_rows)
    weekly_df = pd.DataFrame(weekly_rows)
    weekly_seg_df = pd.DataFrame(weekly_seg_rows)

    rules_df.to_csv(OUT_DIR / "oos_walkforward_selected_rules.csv", index=False)
    weekly_df.to_csv(OUT_DIR / "oos_walkforward_weekly.csv", index=False)
    weekly_seg_df.to_csv(OUT_DIR / "oos_walkforward_weekly_by_segment.csv", index=False)

    # resumo robusto
    w = weekly_df["profit_cap2_usd"].to_numpy(dtype=float)
    mean_w, lo_w, hi_w = bootstrap_ci_mean(w, n_boot=N_BOOT, seed=SEED)
    pneg = float((w < 0).mean())
    std = float(np.std(w, ddof=1)) if w.size >= 2 else 0.0

    # mesma coisa, mas excluindo semanas com stake=0 (sem trades)
    w2 = weekly_df.loc[weekly_df["stake_usd"] > 0, "profit_cap2_usd"].to_numpy(dtype=float)
    mean_w2, lo_w2, hi_w2 = bootstrap_ci_mean(w2, n_boot=N_BOOT, seed=SEED + 1)
    pneg2 = float((w2 < 0).mean()) if w2.size else float("nan")
    std2 = float(np.std(w2, ddof=1)) if w2.size >= 2 else 0.0

    # frequência de ativação por segmento (stake>0 e status ok)
    rules_df["active"] = (rules_df["stake_frac"] > 0) & (rules_df["status"] == "ok")
    act = (
        rules_df.groupby(["bet_type", "dow_pt"], as_index=False)
        .agg(active_rate=("active", "mean"), mean_stake_frac=("stake_frac", "mean"), mean_cutoff=("cutoff", "mean"), ok_rate=("status", lambda s: float(np.mean(np.asarray(s) == "ok"))))
        .sort_values(["bet_type", "dow_pt"])
    )

    lines: List[str] = []
    lines.append("## OOS walk-forward (semana-a-semana) — validação da estratégia completa\n")
    lines.append(f"- Dataset: `{SCORED}`\n")
    lines.append(
        f"- Semanas totais no dataset: **{len(weeks)}**; semanas testadas OOS (WF): **{len(weekly_df)}** "
        f"(a partir de {MIN_GLOBAL_TRAIN_WEEKS} semanas globais de treino; por-segmento exige >= {MIN_SEG_TRAIN_WEEKS})\n"
    )
    lines.append(
        f"- Constraints por segmento (no treino de cada passo): p80 soma stakes/dia <= {MAX_DAILY_EXPOSURE_FRAC_Q*100:.0f}% banca; "
        f"VaR{int(DAILY_VAR_Q*100)}% do PnL diário >= -{MAX_DAILY_DRAWDOWN_FRAC*100:.0f}% banca e P(loss>=25%)<=10%; Sharpe semanal cap2 >= {MIN_WEEKLY_SHARPE_CAP2:.2f}.\n"
    )
    lines.append("\n### Performance OOS (cap2) — portfólio agregado\n")
    lines.append(f"- **PnL semanal médio (bootstrap IC95%)**: **USD {mean_w:.1f}** (IC95% {lo_w:.1f}..{hi_w:.1f})\n")
    lines.append(f"- **Desvio padrão semanal**: USD {std:.1f}\n")
    lines.append(f"- **P(semana < 0)**: {pneg*100:.1f}%\n")
    lines.append("\n### Performance OOS (cap2) — excluindo semanas sem trades (stake=0)\n")
    lines.append(f"- **PnL semanal médio (bootstrap IC95%)**: **USD {mean_w2:.1f}** (IC95% {lo_w2:.1f}..{hi_w2:.1f})\n")
    lines.append(f"- **Desvio padrão semanal**: USD {std2:.1f}\n")
    lines.append(f"- **P(semana < 0)**: {pneg2*100:.1f}%\n")
    lines.append(f"- Série semanal OOS: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_weekly.csv`\n")

    # ROI on stake (médio, ponderado por stake)
    stake_tot = float(weekly_df["stake_usd"].sum())
    pnl_tot = float(weekly_df["profit_cap2_usd"].sum())
    roi_tot = float(pnl_tot / stake_tot) if stake_tot > 0 else float("nan")
    lines.append(f"- **ROI on stake agregado (ponderado)**: {roi_tot:.4f}\n")

    lines.append("\n### Estabilidade OOS da decisão por segmento (frequência de ativação)\n")
    lines.append("- Arquivo: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_selected_rules.csv`\n\n")
    for _, r in act.iterrows():
        lines.append(
            f"- **{r['bet_type']} | {r['dow_pt']}**: active_rate={r['active_rate']*100:.1f}%, ok_rate={r['ok_rate']*100:.1f}%, "
            f"stake_frac_médio={r['mean_stake_frac']*100:.2f}%, cutoff_médio={r['mean_cutoff']:.2f}\n"
        )

    lines.append("\n### Observação importante\n")
    lines.append(
        "- Este WF valida **a estratégia completa por combinação** (cutoff+stake por segmento) de forma honesta no tempo. "
        "Ele ainda não impõe um constraint de risco **global do portfólio** no treino de cada passo (as constraints são por segmento, como no otimizador atual); "
        "se você quiser, eu adapto para otimizar/filtrar também por risco global diário/semanal do portfólio no passo de treino.\n"
    )

    (OUT_DIR / "oos_walkforward_strategy.md").write_text("".join(lines), encoding="utf-8")
    print(str(OUT_DIR / "oos_walkforward_strategy.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

