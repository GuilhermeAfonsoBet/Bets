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
CALIB_SEXDOM = Path("/workspace/clv_calib_SexDom.json")
CALIB_SEGQUI = Path("/workspace/clv_calib_SegQui.json")
CALIB_FLOOR = 0.005  # mesmo piso do RPA/CLI

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

# Confiança estatística mínima (pedido):
# - mínimo de apostas selecionadas por candidato (para evitar regras com 1-2 bets)
MIN_SELECTED_BETS = 6
# - bins de score: exigir pelo menos ~20 apostas por bin para estimar médias com menor variância
MIN_BETS_PER_BIN = 20
MIN_BINS_FOR_STABILITY = 3  # se não dá para formar ao menos 3 bins, consideramos evidência insuficiente
MIN_NONZERO_WEEKS = 6  # mínimo de semanas com >=1 aposta selecionada (confiança temporal)

# Para o walk-forward: exigimos um mínimo de histórico global para começar,
# mas cada segmento pode ter menos semanas (como no otimizador original, que exigia ~6).
MIN_GLOBAL_TRAIN_WEEKS = 10
MIN_SEG_TRAIN_WEEKS = 6

# Seleção Bayesiana (robusta) no treino de cada passo:
# - posterior via Bayesian bootstrap sobre semanas (Dirichlet)
# - objetivo conservador: maximizar p05 do lucro semanal médio (cap2),
#   com penalidade de exposição diária (p95) para desempate
# - requer "confiança": P(mean>0) >= MIN_POST_P_MEAN_POS
BAYES_SELECT = True  # default; main() roda tanto bayes quanto clássico
BAYES_N = 8_000
MIN_POST_P_MEAN_POS = 0.80
POST_Q_OBJ = 0.05  # p05 do mean semanal (posterior)
EXPOSURE_PENALTY = 0.001  # mesmo scale do otimizador

# Calibração por combinação (rule_key) para reduzir otimismo sistemático:
# - Mantém um histórico de erros de ROI por segmento no walk-forward (usando apenas passado)
# - Estima um bias por segmento com shrinkage (pooling) entre segmentos (Empirical Bayes)
# - Aplica correção conservadora na etapa de seleção (apenas penaliza; não "premia" otimismo positivo)
SEGMENT_CALIB_ENABLED = False
SEGMENT_CALIB_ONLY_PENALIZE = True
SEGMENT_CALIB_MIN_TOTAL_OBS = 8  # mínimo de observações (somadas em todos segmentos) para ativar shrinkage

# Experimentos (apenas para estudo): rodar modos adicionais sem sobrescrever baseline.
BIAS_EXPERIMENTS_ENABLED = True
BIAS_DISABLE_ENABLED = True
BIAS_DISABLE_TOP_K = 2  # desliga os K segmentos com maior viés negativo
BIAS_DISABLE_MIN_OBS_PER_SEG = 3  # só considera segmento para desligar se tiver >= este nº de observações históricas

# (Removido a pedido do usuário): lower bound no p05 do PnL semanal.

# Robustez do cutoff (sensibilidade) — DESLIGADO por ora.
ROBUST_CUTOFF_ENABLED = False
ROBUST_CUTOFF_DELTA = 0.02  # 1 passo do grid, se habilitar no futuro

# Histerese / custo de mudança (reduzir churn) — DESLIGADO por ora.
HYSTERESIS_ENABLED = False
HYST_P_SWITCH = 0.90  # se habilitar no futuro

# Gating conservador adicional (experimentos):
# Exigir que o quantil POST_Q_OBJ (p05 por default) do lucro semanal médio seja > 0.
REQUIRE_POST_Q_OBJ_POS = False

# Gating alternativo (menos/more conservador) sem mexer no objetivo:
# Exigir que o quantil POST_Q_GATE do lucro semanal médio posterior seja > 0.
POST_Q_GATE = 0.30
REQUIRE_POST_Q_GATE_POS = False

# Gating por house_cap (novo experimento):
# - Otimiza também um cap máximo permitido (house_cap <= cap_max) por segmento.
# - Intuição: se ROI médio degrada em stakes maiores, podemos restringir execução a caps menores.
CAP_GATING_ENABLED = False
CAP_QS = np.array([0.60, 0.80, 0.90])
CAP_MIN_UNIQUE = 3  # se não houver diversidade de caps, não otimiza cap_max

# Alternativa (mais robusta contra overfit): cap_max fixo por segmento.
# - Escolhe cap_max uma única vez por segmento quando houver dados suficientes (usando apenas passado),
#   e mantém esse cap_max fixo no restante do walk-forward.
CAP_FIXED_PER_SEGMENT_ENABLED = False
CAP_FIXED_MIN_TRAIN_BETS = 60
CAP_FIXED_MIN_NONZERO_WEEKS = 6

N_BOOT = 20_000
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


def _apply_isotonic_vec(p: np.ndarray, x: np.ndarray, y: np.ndarray, floor: float | None) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if x.size and y.size and x.size == y.size:
        out = np.interp(p, x, y, left=float(y[0]), right=float(y[-1]))
    else:
        out = p.copy()
    if floor is not None:
        out = np.maximum(out, float(floor))
    return np.clip(out, 0.0, 1.0)


def segment_score_col(dow: str) -> str:
    if dow == "segunda-feira":
        return "proba_raw_segunda"
    if dow == "terça-feira":
        return "proba_raw_terca"
    if dow == "quarta-feira":
        return "proba_raw_quarta"
    if dow == "quinta-feira":
        # Quinta: no RPA (score_logit_by_dow_cli) é subset SegQui + isotonic + floor
        return "proba_cal_segqui"
    # Sex/Sáb/Dom: usar score calibrado (mesmo do RPA) para alinhar execução com estudo
    return "proba_cal_sexdom"


@dataclass(frozen=True)
class Rule:
    bet_type: str
    dow: str
    score_col: str
    cutoff: float
    stake_frac: float
    status: str
    cap_max: float = float("inf")  # house_cap máximo permitido (inf = sem restrição)


def score_bin_ok(score_sel: np.ndarray, profit_sel: np.ndarray) -> Tuple[int, int, bool]:
    # returns (n_bins, pos_bins, ok)
    score_sel = np.asarray(score_sel, dtype=float)
    profit_sel = np.asarray(profit_sel, dtype=float)
    if score_sel.size == 0:
        return 0, 0, False

    # Exigir evidência mínima para avaliar estabilidade por bins:
    # se não há apostas suficientes para formar ao menos 3 bins com ~MIN_BETS_PER_BIN cada, falha.
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


def optimize_segment_train(
    x: pd.DataFrame,
    score_col: str,
    bayes_select: bool,
    prev_rule: Rule | None,
    roi_bias_adj: float = 0.0,
) -> Rule:
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
    # ajuste conservador de ROI (calibração por segmento):
    # roi_bias_adj é erro (ROI_real - ROI_previsto); se negativo => modelo otimista => reduzir ROI esperado.
    roi_bias = float(roi_bias_adj)
    roi2_adj = roi2 + roi_bias
    roi1_adj = roi1 + roi_bias
    cap = x["house_cap"].to_numpy(dtype=float)
    wk = x["week"].to_numpy()
    d = x["date"].to_numpy()

    # Bayesian bootstrap weights (fixos por chamada, para comparabilidade entre candidatos)
    bb_weights = None
    if bayes_select:
        rng = np.random.default_rng(SEED + 123 + hash((str(x["dow_pt"].iloc[0]), str(x["bet_type"].iloc[0]))) % 10_000)
        bb_weights = rng.dirichlet(np.ones(len(weeks_all)), size=BAYES_N)

    def eval_obj_for_cutoff(stake_eff: np.ndarray, cutoff: float, cap_max: float) -> Tuple[bool, float, np.ndarray | None]:
        """
        Avalia um cutoff fixo (stake_eff já definido pelo stake_frac).
        Retorna (ok, obj, post_means) onde post_means só existe no modo bayes_select.
        """
        m = np.isfinite(score) & (score >= float(cutoff)) & np.isfinite(roi2_adj)
        if CAP_GATING_ENABLED and np.isfinite(float(cap_max)):
            m = m & np.isfinite(cap) & (cap <= float(cap_max))
        if not np.any(m):
            return False, -np.inf, None
        if int(np.sum(m)) < MIN_SELECTED_BETS:
            return False, -np.inf, None
        nonzero_weeks = int(pd.Series(np.ones(int(np.sum(m))), index=wk[m]).groupby(level=0).sum().shape[0])
        if nonzero_weeks < MIN_NONZERO_WEEKS:
            return False, -np.inf, None

        pnl2 = stake_eff[m] * roi2_adj[m]
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

        pnl1 = stake_eff[m] * roi1_adj[m]
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

        pnl_day = pd.Series(stake_eff[m] * roi2_adj[m], index=d[m]).groupby(level=0).sum().to_numpy(dtype=float)
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
            n_bins, pos_bins, ok = score_bin_ok(score[m], stake_eff[m] * roi2_adj[m])
            if not ok:
                return False, -np.inf, None

        if not bayes_select:
            return True, float(mean - 0.25 * std - EXPOSURE_PENALTY * p95_exp), None

        assert bb_weights is not None
        post_means = bb_weights @ w2.astype(float)
        p_mean_pos = float(np.mean(post_means > 0))
        if p_mean_pos < MIN_POST_P_MEAN_POS:
            return False, -np.inf, None
        q_obj = float(np.quantile(post_means, POST_Q_OBJ))
        if REQUIRE_POST_Q_OBJ_POS and q_obj <= 0:
            return False, -np.inf, None
        if REQUIRE_POST_Q_GATE_POS:
            q_gate = float(np.quantile(post_means, POST_Q_GATE))
            if q_gate <= 0:
                return False, -np.inf, None
        return True, float(q_obj - EXPOSURE_PENALTY * p95_exp), post_means

    best_obj = -np.inf
    best = None  # (cutoff, stake_frac)
    best_post = None

    # Passo 1 (sempre): otimiza cutoff+stake sem cap gating (cap_max=inf)
    for f in STAKE_FRACS:
        stake0 = BANKROLL * float(f)
        stake_eff = np.minimum(stake0, cap)
        for c in CUTOFFS:
            ok0, obj0, post0 = eval_obj_for_cutoff(stake_eff, float(c), float("inf"))
            if not ok0:
                continue

            obj_use = obj0
            if bayes_select and ROBUST_CUTOFF_ENABLED:
                worst = obj0
                for cc in (float(c) - ROBUST_CUTOFF_DELTA, float(c) + ROBUST_CUTOFF_DELTA):
                    if cc < float(CUTOFFS.min()) or cc > float(CUTOFFS.max()):
                        continue
                    okn, objn, _ = eval_obj_for_cutoff(stake_eff, float(cc), float("inf"))
                    if not okn:
                        worst = -np.inf
                        break
                    worst = min(worst, objn)
                obj_use = worst
                if not np.isfinite(obj_use):
                    continue

            if obj_use > best_obj:
                best_obj = float(obj_use)
                best = (float(c), float(f))
                best_post = post0

    if best is None:
        return Rule(bet_type=str(x["bet_type"].iloc[0]), dow=str(x["dow_pt"].iloc[0]), score_col=score_col, cutoff=1.0, stake_frac=0.0, cap_max=float("inf"), status="no_candidate")

    # Passo 2 (se habilitado): dado o melhor cutoff+stake, escolhe cap_max por grid de quantis (rápido).
    best_cap = float("inf")
    if CAP_GATING_ENABLED:
        cap_ok = cap[np.isfinite(cap) & (cap > 0)]
        cap_candidates = []
        if cap_ok.size > 0:
            qs = np.unique(np.quantile(cap_ok, CAP_QS))
            qs = qs[np.isfinite(qs) & (qs > 0)]
            if qs.size >= CAP_MIN_UNIQUE:
                cap_candidates = [float(v) for v in qs.tolist()]
        # avalia apenas caps finitos; inf já é o baseline (best_obj/best_post)
        c_best, f_best = best
        stake0 = BANKROLL * float(f_best)
        stake_eff = np.minimum(stake0, cap)
        for cap_max in sorted(set(cap_candidates)):
            ok0, obj0, post0 = eval_obj_for_cutoff(stake_eff, float(c_best), float(cap_max))
            if not ok0:
                continue
            obj_use = obj0
            if bayes_select and ROBUST_CUTOFF_ENABLED:
                worst = obj0
                for cc in (float(c_best) - ROBUST_CUTOFF_DELTA, float(c_best) + ROBUST_CUTOFF_DELTA):
                    if cc < float(CUTOFFS.min()) or cc > float(CUTOFFS.max()):
                        continue
                    okn, objn, _ = eval_obj_for_cutoff(stake_eff, float(cc), float(cap_max))
                    if not okn:
                        worst = -np.inf
                        break
                    worst = min(worst, objn)
                obj_use = worst
            if np.isfinite(obj_use) and float(obj_use) > float(best_obj):
                best_obj = float(obj_use)
                best_cap = float(cap_max)
                best_post = post0

    # Histerese: comparar contra a regra anterior, se válida
    if HYSTERESIS_ENABLED and bayes_select and prev_rule is not None and prev_rule.status == "ok" and prev_rule.stake_frac > 0 and best_post is not None:
        # Reavaliar a regra anterior no treino atual
        stake0_prev = BANKROLL * float(prev_rule.stake_frac)
        stake_eff_prev = np.minimum(stake0_prev, cap)
        mprev = np.isfinite(score) & (score >= float(prev_rule.cutoff)) & np.isfinite(roi2)
        if CAP_GATING_ENABLED and np.isfinite(float(prev_rule.cap_max)):
            mprev = mprev & np.isfinite(cap) & (cap <= float(prev_rule.cap_max))
        ok_prev = True
        if not np.any(mprev) or int(np.sum(mprev)) < MIN_SELECTED_BETS:
            ok_prev = False
        else:
            nonzero_prev = int(pd.Series(np.ones(int(np.sum(mprev))), index=wk[mprev]).groupby(level=0).sum().shape[0])
            if nonzero_prev < MIN_NONZERO_WEEKS:
                ok_prev = False
        if ok_prev:
            pnl2_prev = stake_eff_prev[mprev] * roi2[mprev]
            w2_prev = (
                pd.Series(pnl2_prev, index=wk[mprev])
                .groupby(level=0)
                .sum()
                .reindex(weeks_all, fill_value=0.0)
                .to_numpy(dtype=float)
            )
            assert bb_weights is not None
            post_prev = bb_weights @ w2_prev.astype(float)
            p_switch = float(np.mean(best_post > post_prev))
            if p_switch < HYST_P_SWITCH:
                return Rule(bet_type=str(x["bet_type"].iloc[0]), dow=str(x["dow_pt"].iloc[0]), score_col=score_col, cutoff=float(prev_rule.cutoff), stake_frac=float(prev_rule.stake_frac), cap_max=float(prev_rule.cap_max), status="ok")

    return Rule(bet_type=str(x["bet_type"].iloc[0]), dow=str(x["dow_pt"].iloc[0]), score_col=score_col, cutoff=best[0], stake_frac=best[1], cap_max=float(best_cap), status="ok")


def apply_rule_on_week(df_week: pd.DataFrame, rule: Rule) -> pd.DataFrame:
    if rule.stake_frac <= 0:
        return df_week.iloc[:0].copy()
    stake0 = BANKROLL * rule.stake_frac
    x = df_week[(df_week["dow_pt"] == rule.dow) & (df_week["bet_type"] == rule.bet_type)].copy()
    if x.empty:
        return x
    score = pd.to_numeric(x[rule.score_col], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(score) & (score >= rule.cutoff) & np.isfinite(x["roi_cap2"].to_numpy(dtype=float))
    if np.isfinite(float(rule.cap_max)):
        cap = x["house_cap"].to_numpy(dtype=float)
        m = m & np.isfinite(cap) & (cap <= float(rule.cap_max))
    x = x.iloc[np.where(m)[0]].copy()
    if x.empty:
        return x
    x["stake_eff"] = np.minimum(stake0, x["house_cap"].to_numpy(dtype=float))
    x["profit_cap2"] = x["stake_eff"].to_numpy(dtype=float) * x["roi_cap2"].to_numpy(dtype=float)
    x["rule_cutoff"] = rule.cutoff
    x["rule_stake_frac"] = rule.stake_frac
    x["rule_cap_max"] = float(rule.cap_max)
    x["rule_score_col"] = rule.score_col
    x["rule_status"] = rule.status
    x["rule_key"] = f"{rule.bet_type}|{rule.dow}"
    return x


def apply_rules_on_df(df_any: pd.DataFrame, rules: Dict[str, Rule], alpha: float) -> pd.DataFrame:
    """
    Aplica um conjunto de regras (para vários segmentos) num DataFrame qualquer (treino ou teste),
    com um fator global alpha multiplicando os stake_fracs.
    Retorna apostas selecionadas com stake/profit cap2.
    """
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
        if np.isfinite(float(rule.cap_max)):
            cap = x["house_cap"].to_numpy(dtype=float)
            m = m & np.isfinite(cap) & (cap <= float(rule.cap_max))
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


def _cap_candidates_from_x(cap: np.ndarray) -> List[float]:
    cap_ok = np.asarray(cap, dtype=float)
    cap_ok = cap_ok[np.isfinite(cap_ok) & (cap_ok > 0)]
    out = [float("inf")]
    if cap_ok.size:
        qs = np.unique(np.quantile(cap_ok, CAP_QS))
        qs = qs[np.isfinite(qs) & (qs > 0)]
        out += [float(v) for v in qs.tolist()]
    return sorted(set(out))


def _cap_select_obj_p10(
    x: pd.DataFrame, rule: Rule, weeks_all: List[str], seed: int
) -> Tuple[float, float]:
    """
    Objetivo simples para escolher cap_max em treino:
    - posterior p10 do lucro semanal médio (cap2) via Bayesian bootstrap em semanas.
    - retorna (obj, p_mean_pos)
    """
    if x.empty or rule.status != "ok" or rule.stake_frac <= 0:
        return -np.inf, 0.0
    score = pd.to_numeric(x[rule.score_col], errors="coerce").to_numpy(float)
    roi2 = x["roi_cap2"].to_numpy(float)
    wk = x["week"].to_numpy()
    cap = x["house_cap"].to_numpy(float)
    m = np.isfinite(score) & (score >= float(rule.cutoff)) & np.isfinite(roi2) & np.isfinite(cap) & (cap > 0)
    if not np.any(m):
        return -np.inf, 0.0
    stake0 = BANKROLL * float(rule.stake_frac)
    stake_eff = np.minimum(stake0, cap[m])
    pnl = stake_eff * roi2[m]
    w2 = (
        pd.Series(pnl, index=wk[m])
        .groupby(level=0)
        .sum()
        .reindex(weeks_all, fill_value=0.0)
        .to_numpy(float)
    )
    rng = np.random.default_rng(seed)
    bb = rng.dirichlet(np.ones(len(weeks_all)), size=int(BAYES_N))
    post = bb @ w2.astype(float)
    ppos = float(np.mean(post > 0))
    if ppos < MIN_POST_P_MEAN_POS:
        return -np.inf, ppos
    return float(np.quantile(post, POST_Q_OBJ)), ppos


def _empirical_bayes_shrink(means: np.ndarray, se2: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    Shrinkage Empirical Bayes para estimar um efeito por grupo:
      m_i ~ Normal(mu0, tau^2 + se_i^2)
    Retorna (mu0, tau2, post_mean_i).
    """
    m = np.asarray(means, dtype=float)
    v = np.asarray(se2, dtype=float)
    ok = np.isfinite(m) & np.isfinite(v) & (v > 0)
    m = m[ok]
    v = v[ok]
    if m.size == 0:
        return float("nan"), 0.0, np.array([], dtype=float)
    w = 1.0 / v
    mu0 = float(np.sum(w * m) / np.sum(w))
    # método de momentos (aprox): var(m) ≈ tau2 + mean(se2)
    var_m = float(np.var(m, ddof=1)) if m.size > 1 else 0.0
    tau2 = float(max(0.0, var_m - float(np.mean(v))))
    if tau2 <= 1e-12:
        post = np.full(m.size, mu0, dtype=float)
    else:
        post = (m / v + mu0 / tau2) / (1.0 / v + 1.0 / tau2)
    return mu0, tau2, post


def _segment_roi_bias_shrunk(roi_err_hist: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Estima bias de ROI por segmento usando apenas histórico observado:
    - bias = ROI_real - ROI_previsto (negativo => otimismo)
    - aplica shrinkage entre segmentos para reduzir ruído em amostras pequenas
    """
    keys = sorted(roi_err_hist.keys())
    all_err = np.array([e for k in keys for e in roi_err_hist.get(k, []) if np.isfinite(float(e))], dtype=float)
    total_obs = int(all_err.size)
    if total_obs < SEGMENT_CALIB_MIN_TOTAL_OBS:
        return {}
    global_var = float(np.var(all_err, ddof=1)) if all_err.size > 1 else 0.0
    means = []
    se2 = []
    used_keys = []
    for k in keys:
        a = np.asarray(roi_err_hist.get(k, []), dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            continue
        m = float(np.mean(a))
        vv = float(np.var(a, ddof=1)) if a.size > 1 else global_var
        # se global_var==0 e n==1, ainda precisamos de um se2>0
        vv = float(vv if vv > 1e-12 else 1e-6)
        means.append(m)
        se2.append(vv / float(a.size))
        used_keys.append(k)

    if len(used_keys) < 2:
        return {}
    mu0, tau2, post = _empirical_bayes_shrink(np.asarray(means), np.asarray(se2))
    out = {}
    for k, pm in zip(used_keys, post):
        val = float(pm)
        if SEGMENT_CALIB_ONLY_PENALIZE:
            val = min(0.0, val)
        out[k] = val
    return out


def _segment_roi_bias_raw_mean(
    roi_err_hist: Dict[str, List[float]], min_obs_per_seg: int
) -> Dict[str, float]:
    """
    Bias bruto por segmento: média(ROI_real - ROI_previsto) no histórico.
    Usado apenas para escolher quais segmentos 'desligar' (evita arbitrariedade quando tau2~0).
    """
    out: Dict[str, float] = {}
    for k, xs in roi_err_hist.items():
        a = np.asarray([float(v) for v in xs if np.isfinite(float(v))], dtype=float)
        if a.size < int(min_obs_per_seg):
            continue
        out[str(k)] = float(a.mean())
    return out

def _rule_weekly_roi_mean(df_train: pd.DataFrame, rule: Rule, train_weeks: List[str]) -> float:
    """
    ROI semanal médio no treino, incluindo semanas sem trade como ROI=0.
    Usado como "ROI previsto" para calibração por segmento.
    """
    if not train_weeks:
        return 0.0
    bets = apply_rules_on_df(df_train, {f"{rule.bet_type}|{rule.dow}": rule}, alpha=1.0)
    if bets.empty:
        return 0.0
    g = bets.groupby("week", as_index=False).agg(stake=("stake_eff", "sum"), pnl=("profit_cap2", "sum"))
    gm = g.set_index("week").reindex(train_weeks, fill_value=0.0)
    stake = gm["stake"].to_numpy(dtype=float)
    pnl = gm["pnl"].to_numpy(dtype=float)
    roi_w = np.zeros_like(stake, dtype=float)
    np.divide(pnl, stake, out=roi_w, where=(stake > 0))
    return float(np.mean(roi_w))


def portfolio_global_constraints_ok(df_train: pd.DataFrame, rules: Dict[str, Rule], alpha: float) -> Tuple[bool, Dict[str, float]]:
    """
    Checa constraints no portfólio agregado (treino):
      - p80 soma stakes/dia <= 70% banca
      - VaR10%(PnL_dia) >= -25% banca
      - P(PnL_dia <= -25% banca) <= 10%
    Retorna (ok, metrics)
    """
    bets = apply_rules_on_df(df_train, rules, alpha=alpha)
    if bets.empty:
        # sem trades -> trivialmente ok, mas não desejável
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
    """
    Encontra o maior alpha in [0,1] que satisfaz constraints globais no treino.
    Retorna (alpha, metrics_at_1, metrics_at_alpha).
    """
    ok1, m1 = portfolio_global_constraints_ok(df_train, rules, alpha=1.0)
    if ok1:
        return 1.0, m1, m1
    lo, hi = 0.0, 1.0
    best = 0.0
    best_m = None
    # busca binária monotônica (alpha menor => risco/exposição menor)
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
        # mesmo alpha~0 não achou trades? ou ainda viola (deveria não violar)
        ok0, m0 = portfolio_global_constraints_ok(df_train, rules, alpha=0.0)
        return 0.0, m1, m0
    return float(best), m1, best_m


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
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--only-mode", default="", help="Se preenchido, roda apenas este modo (ex.: global_bayes_roll12_robust_p10_p70_capgate)")
    args = ap.parse_args()
    only_mode = str(args.only_mode).strip()
    df = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])
    df["house_cap"] = df["house_cap"].apply(safe_cap)
    df["week"] = week_key(df["BIA_ApostaUTC"])
    df["date"] = date_key(df["BIA_ApostaUTC"])
    # ROI: usar ROI calculado via odds+resultado (mais consistente que a coluna \"ROI Real\" da planilha)
    if "roi_calc" not in df.columns:
        raise KeyError("Coluna roi_calc ausente. Regerar /workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")
    df["roi_raw"] = pd.to_numeric(df["roi_calc"], errors="coerce").astype(float)
    df["roi_cap2"] = np.minimum(df["roi_raw"].to_numpy(dtype=float), 2.0)
    df["roi_cap1"] = np.minimum(df["roi_raw"].to_numpy(dtype=float), 1.0)

    # Construir scores calibrados para SegQui (qui) e SexDom (sex/sab/dom) a partir dos proba_raw_* + isotonic.
    # Replica a lógica do CLI (apply_isotonic + calib_floor).
    df["proba_cal_segqui"] = np.nan
    if "proba_raw_segqui" in df.columns and CALIB_SEGQUI.exists():
        try:
            calib = json.loads(CALIB_SEGQUI.read_text(encoding="utf-8"))
            x = np.asarray(calib.get("isotonic", {}).get("x", []), dtype=float)
            y = np.asarray(calib.get("isotonic", {}).get("y", []), dtype=float)
            p_raw = pd.to_numeric(df["proba_raw_segqui"], errors="coerce").to_numpy(dtype=float)
            df["proba_cal_segqui"] = _apply_isotonic_vec(p_raw, x=x, y=y, floor=CALIB_FLOOR)
        except Exception:
            pass

    df["proba_cal_sexdom"] = np.nan
    if "proba_raw_sexdom" in df.columns and CALIB_SEXDOM.exists():
        try:
            calib = json.loads(CALIB_SEXDOM.read_text(encoding="utf-8"))
            x = np.asarray(calib.get("isotonic", {}).get("x", []), dtype=float)
            y = np.asarray(calib.get("isotonic", {}).get("y", []), dtype=float)
            p_raw = pd.to_numeric(df["proba_raw_sexdom"], errors="coerce").to_numpy(dtype=float)
            df["proba_cal_sexdom"] = _apply_isotonic_vec(p_raw, x=x, y=y, floor=CALIB_FLOOR)
        except Exception:
            pass

    weeks = sorted(df["week"].unique().tolist())
    if len(weeks) < (MIN_GLOBAL_TRAIN_WEEKS + 3):
        raise SystemExit(f"Poucas semanas no dataset: {len(weeks)}")

    def run_walkforward(
        global_risk: bool,
        bayes_select: bool,
        segment_calib: bool,
        disable_top_k: int = 0,
        train_window_weeks: int | None = None,
        regime_lookback_weeks: int | None = None,
        regime_alpha_bad: float = 1.0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        all_rules_rows = []
        weekly_rows = []
        weekly_seg_rows = []
        weekly_global_rows = []
        daily_rows = []
        prev_rules: Dict[str, Rule] = {}
        fixed_cap_by_key: Dict[str, float] = {}
        # histórico de calibração por segmento (só usa passado)
        seg_roi_err_hist: Dict[str, List[float]] = {}

        for i in range(MIN_GLOBAL_TRAIN_WEEKS, len(weeks)):
            w_test = weeks[i]
            if train_window_weeks is None:
                train_weeks = weeks[:i]
            else:
                w = int(train_window_weeks)
                train_weeks = weeks[max(0, i - w) : i]

            df_train = df[df["week"].isin(train_weeks)].copy()
            df_test = df[df["week"] == w_test].copy()

            roi_bias_adj_map: Dict[str, float] = {}
            if bayes_select and segment_calib:
                roi_bias_adj_map = _segment_roi_bias_shrunk(seg_roi_err_hist)

            disabled_keys: set[str] = set()
            if bayes_select and segment_calib and int(disable_top_k) > 0:
                bias_raw = _segment_roi_bias_raw_mean(seg_roi_err_hist, min_obs_per_seg=BIAS_DISABLE_MIN_OBS_PER_SEG)
                worst = sorted(bias_raw.items(), key=lambda kv: kv[1])[: int(disable_top_k)]
                disabled_keys = {k for k, _ in worst}

            # otimiza regras no treino (para cada segmento)
            rules: Dict[str, Rule] = {}
            for bet_type in ("FT", "FH"):
                for dow in WEEKDAY_PT:
                    sc = segment_score_col(dow)
                    x = df_train[(df_train["dow_pt"] == dow) & (df_train["bet_type"] == bet_type)].copy()
                    if x.empty:
                        rule = Rule(bet_type=bet_type, dow=dow, score_col=sc, cutoff=1.0, stake_frac=0.0, status="no_data")
                    else:
                        prev = prev_rules.get(f"{bet_type}|{dow}")
                        rk = f"{bet_type}|{dow}"
                        if rk in disabled_keys:
                            rule = Rule(bet_type=bet_type, dow=dow, score_col=sc, cutoff=1.0, stake_frac=0.0, status="disabled_bias")
                        else:
                            roi_adj = float(roi_bias_adj_map.get(rk, 0.0))
                            # cap_max fixo por segmento (escolhido uma vez e mantido)
                            cap_fixed = fixed_cap_by_key.get(rk) if CAP_FIXED_PER_SEGMENT_ENABLED else None
                            if cap_fixed is not None and np.isfinite(float(cap_fixed)):
                                x_use = x[np.isfinite(x["house_cap"]) & (x["house_cap"] <= float(cap_fixed))].copy()
                            else:
                                x_use = x

                            # se ainda não temos cap fixo, tentamos escolhê-lo UMA vez (em treino) testando poucos candidatos
                            if CAP_FIXED_PER_SEGMENT_ENABLED and (rk not in fixed_cap_by_key):
                                score0 = pd.to_numeric(x["house_cap"], errors="coerce").to_numpy(float)
                                # só tenta se há volume suficiente
                                if int(np.sum(np.isfinite(score0) & (score0 > 0))) >= CAP_FIXED_MIN_TRAIN_BETS:
                                    weeks_all = sorted(x["week"].unique().tolist())
                                    best_cap = float("inf")
                                    best_obj = -np.inf
                                    best_rule = None
                                    for cap_cand in _cap_candidates_from_x(x["house_cap"].to_numpy(float)):
                                        if np.isfinite(cap_cand):
                                            xx = x[np.isfinite(x["house_cap"]) & (x["house_cap"] <= float(cap_cand))].copy()
                                        else:
                                            xx = x
                                        # exige diversidade temporal mínima
                                        nonzero_weeks = int(xx["week"].nunique())
                                        if nonzero_weeks < CAP_FIXED_MIN_NONZERO_WEEKS:
                                            continue
                                        rr = optimize_segment_train(xx, sc, bayes_select=bayes_select, prev_rule=None, roi_bias_adj=roi_adj)
                                        if rr.status != "ok" or rr.stake_frac <= 0:
                                            continue
                                        # define cap no rule apenas para avaliação do objetivo
                                        rr = Rule(bet_type=rr.bet_type, dow=rr.dow, score_col=rr.score_col, cutoff=rr.cutoff, stake_frac=rr.stake_frac, status=rr.status, cap_max=float(cap_cand))
                                        obj, _pp = _cap_select_obj_p10(xx, rr, weeks_all=weeks_all, seed=SEED + 991 + hash(rk) % 10_000)
                                        if obj > best_obj:
                                            best_obj = float(obj)
                                            best_cap = float(cap_cand)
                                            best_rule = rr
                                    fixed_cap_by_key[rk] = float(best_cap)
                                    if best_rule is not None and np.isfinite(best_cap):
                                        x_use = x[np.isfinite(x["house_cap"]) & (x["house_cap"] <= float(best_cap))].copy()

                            rule0 = optimize_segment_train(x_use, sc, bayes_select=bayes_select, prev_rule=prev, roi_bias_adj=roi_adj)
                            cap_final = float(fixed_cap_by_key.get(rk, float("inf"))) if CAP_FIXED_PER_SEGMENT_ENABLED else float("inf")
                            rule = Rule(bet_type=rule0.bet_type, dow=rule0.dow, score_col=rule0.score_col, cutoff=rule0.cutoff, stake_frac=rule0.stake_frac, status=rule0.status, cap_max=cap_final)
                    rules[f"{bet_type}|{dow}"] = rule

            # se pedir risco global, ajustar stakes por alpha
            if global_risk:
                alpha, m_at1, m_ata = find_alpha_global(df_train, rules)
            else:
                alpha, m_at1, m_ata = 1.0, {"p80_exp": float("nan"), "daily_var10": float("nan"), "p_dd": float("nan"), "n_days": 0}, {"p80_exp": float("nan"), "daily_var10": float("nan"), "p_dd": float("nan"), "n_days": 0}

            # Regime overlay (usa apenas passado): se o ROI recente do portfólio for negativo, reduzir alpha.
            alpha_regime = 1.0
            if regime_lookback_weeks is not None and int(regime_lookback_weeks) > 0:
                tw = list(train_weeks)
                recent = tw[-int(regime_lookback_weeks) :] if len(tw) else []
                if recent:
                    df_recent = df_train[df_train["week"].isin(recent)].copy()
                    bets_recent = apply_rules_on_df(df_recent, rules, alpha=float(alpha))
                    stake_r = float(bets_recent["stake_eff"].sum()) if not bets_recent.empty else 0.0
                    pnl_r = float(bets_recent["profit_cap2"].sum()) if not bets_recent.empty else 0.0
                    roi_r = float(pnl_r / stake_r) if stake_r > 0 else 0.0
                    if roi_r < 0:
                        alpha_regime = float(regime_alpha_bad)
            alpha_use = float(alpha) * float(alpha_regime)

            # salvar regras daquela semana (com alpha)
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
                        "cap_max": float(rule.cap_max),
                        "alpha_global": float(alpha),
                        "status": rule.status,
                        "rule_key": key,
                        "roi_bias_adj_used": float(roi_bias_adj_map.get(key, 0.0)) if (bayes_select and segment_calib) else 0.0,
                        "alpha_regime": float(alpha_regime),
                        "alpha_effective": float(alpha_use),
                    }
                )

            weekly_global_rows.append(
                {
                    "week": w_test,
                    "train_weeks": len(train_weeks),
                    "alpha_global": float(alpha),
                    "alpha_regime": float(alpha_regime),
                    "alpha_effective": float(alpha_use),
                    "train_global_p80_exp_at1": float(m_at1.get("p80_exp", float("nan"))),
                    "train_global_var10_at1": float(m_at1.get("daily_var10", float("nan"))),
                    "train_global_p_dd_at1": float(m_at1.get("p_dd", float("nan"))),
                    "train_global_p80_exp_at_alpha": float(m_ata.get("p80_exp", float("nan"))),
                    "train_global_var10_at_alpha": float(m_ata.get("daily_var10", float("nan"))),
                    "train_global_p_dd_at_alpha": float(m_ata.get("p_dd", float("nan"))),
                    "train_global_n_days": int(m_ata.get("n_days", 0)),
                }
            )

            # aplica no teste e agrega (usando alpha)
            bets = apply_rules_on_df(df_test, rules, alpha=alpha_use)

            stake_sum = float(bets["stake_eff"].sum()) if len(bets) else 0.0
            pnl_sum = float(bets["profit_cap2"].sum()) if len(bets) else 0.0
            roi_on_stake = float(pnl_sum / stake_sum) if stake_sum > 0 else float("nan")
            n_bets = int(len(bets))

            weekly_rows.append(
                {
                    "week": w_test,
                    "train_weeks": len(train_weeks),
                    "alpha_global": float(alpha),
                    "n_bets": n_bets,
                    "stake_usd": stake_sum,
                    "profit_cap2_usd": pnl_sum,
                    "roi_on_stake_cap2": roi_on_stake,
                }
            )

            if len(bets):
                # daily series OOS (para métricas globais no teste)
                dd = bets.groupby("date", as_index=False).agg(stake_usd=("stake_eff", "sum"), profit_cap2_usd=("profit_cap2", "sum"))
                dd["week"] = w_test
                dd["alpha_global"] = float(alpha)
                daily_rows.append(dd)

                g = bets.groupby("rule_key", as_index=False).agg(n_bets=("profit_cap2", "size"), stake_usd=("stake_eff", "sum"), profit_cap2_usd=("profit_cap2", "sum"))
                for _, r in g.iterrows():
                    weekly_seg_rows.append(
                        {
                            "week": w_test,
                            "alpha_global": float(alpha),
                            "rule_key": r["rule_key"],
                            "n_bets": int(r["n_bets"]),
                            "stake_usd": float(r["stake_usd"]),
                            "profit_cap2_usd": float(r["profit_cap2_usd"]),
                            "roi_on_stake_cap2": float(r["profit_cap2_usd"] / r["stake_usd"]) if float(r["stake_usd"]) > 0 else float("nan"),
                        }
                    )

            # atualizar histórico de calibração por segmento (para próximos passos)
            # usa ROI previsto no treino (média semanal, incluindo semanas sem trade como 0) e ROI realizado no teste
            if bayes_select and segment_calib:
                for key, rule in rules.items():
                    if rule.status != "ok" or rule.stake_frac <= 0:
                        continue
                    roi_pred = _rule_weekly_roi_mean(df_train, rule, train_weeks=train_weeks)
                    b_test = apply_rules_on_df(df_test, {key: rule}, alpha=1.0)
                    stake_t = float(b_test["stake_eff"].sum()) if not b_test.empty else 0.0
                    pnl_t = float(b_test["profit_cap2"].sum()) if not b_test.empty else 0.0
                    if stake_t <= 0:
                        continue
                    roi_real = float(pnl_t / stake_t)
                    err_roi = float(roi_real - roi_pred)
                    seg_roi_err_hist.setdefault(key, []).append(err_roi)

            # atualizar regras anteriores (histerese) para o próximo passo
            prev_rules = rules.copy()

        daily_df = pd.concat(daily_rows, axis=0, ignore_index=True) if daily_rows else pd.DataFrame(columns=["date", "stake_usd", "profit_cap2_usd", "week", "alpha_global"])
        return (
            pd.DataFrame(all_rules_rows),
            pd.DataFrame(weekly_rows),
            pd.DataFrame(weekly_seg_rows),
            pd.DataFrame(weekly_global_rows),
            daily_df,
        )

    results_summary_rows: List[Dict[str, float | int | str]] = []

    def _summarize_mode(mode_name: str, weekly_df: pd.DataFrame) -> None:
        stake_tot = float(weekly_df["stake_usd"].sum())
        pnl_tot = float(weekly_df["profit_cap2_usd"].sum())
        roi_tot = float(pnl_tot / stake_tot) if stake_tot > 0 else float("nan")
        w_nonzero = weekly_df.loc[weekly_df["stake_usd"] > 0, "profit_cap2_usd"].to_numpy(dtype=float)
        results_summary_rows.append(
            {
                "mode": mode_name,
                "weeks_total": int(len(weekly_df)),
                "weeks_with_stake": int((weekly_df["stake_usd"] > 0).sum()),
                "profit_cap2_total": pnl_tot,
                "stake_total": stake_tot,
                "roi_total_cap2": roi_tot,
                "mean_weekly_cap2_nonzero": float(np.mean(w_nonzero)) if w_nonzero.size else float("nan"),
                "pneg_weeks_nonzero": float((w_nonzero < 0).mean()) if w_nonzero.size else float("nan"),
            }
        )

    # ------------------------
    # Baseline: 4 modos (como antes)
    # ------------------------
    for mode_name, global_risk, bayes_select in (
        ("segment_classic", False, False),
        ("global_classic", True, False),
        ("segment_bayes", False, True),
        ("global_bayes", True, True),
    ):
        rules_df, weekly_df, weekly_seg_df, weekly_global_df, daily_df = run_walkforward(
            global_risk=global_risk,
            bayes_select=bayes_select,
            segment_calib=SEGMENT_CALIB_ENABLED,
            disable_top_k=0,
            train_window_weeks=None,
            regime_lookback_weeks=None,
            regime_alpha_bad=1.0,
        )

        rules_df.to_csv(OUT_DIR / f"oos_walkforward_{mode_name}_selected_rules.csv", index=False)
        weekly_df.to_csv(OUT_DIR / f"oos_walkforward_{mode_name}_weekly.csv", index=False)
        weekly_seg_df.to_csv(OUT_DIR / f"oos_walkforward_{mode_name}_weekly_by_segment.csv", index=False)
        weekly_global_df.to_csv(OUT_DIR / f"oos_walkforward_{mode_name}_train_global_metrics.csv", index=False)
        daily_df.to_csv(OUT_DIR / f"oos_walkforward_{mode_name}_daily.csv", index=False)
        _summarize_mode(mode_name, weekly_df)

        # resumo robusto
        w = weekly_df["profit_cap2_usd"].to_numpy(dtype=float)
        mean_w, lo_w, hi_w = bootstrap_ci_mean(w, n_boot=N_BOOT, seed=SEED + (11 if global_risk else 0))
        pneg = float((w < 0).mean())
        std = float(np.std(w, ddof=1)) if w.size >= 2 else 0.0

        # mesma coisa, mas excluindo semanas com stake=0
        w2 = weekly_df.loc[weekly_df["stake_usd"] > 0, "profit_cap2_usd"].to_numpy(dtype=float)
        mean_w2, lo_w2, hi_w2 = bootstrap_ci_mean(w2, n_boot=N_BOOT, seed=SEED + 1 + (11 if global_risk else 0))
        pneg2 = float((w2 < 0).mean()) if w2.size else float("nan")
        std2 = float(np.std(w2, ddof=1)) if w2.size >= 2 else 0.0

        # ROI on stake agregado
        stake_tot = float(weekly_df["stake_usd"].sum())
        pnl_tot = float(weekly_df["profit_cap2_usd"].sum())
        roi_tot = float(pnl_tot / stake_tot) if stake_tot > 0 else float("nan")

        # métricas de risco no OOS (teste) — portfólio agregado
        if not daily_df.empty:
            stake_day = daily_df["stake_usd"].to_numpy(dtype=float)
            pnl_day = daily_df["profit_cap2_usd"].to_numpy(dtype=float)
            oos_p80_exp = float(np.quantile(stake_day, DAILY_EXPOSURE_Q)) if stake_day.size else float("nan")
            oos_var10 = float(np.quantile(pnl_day, DAILY_VAR_Q)) if pnl_day.size else float("nan")
            oos_p_dd = float((pnl_day <= (-MAX_DAILY_DRAWDOWN_FRAC * BANKROLL)).mean()) if pnl_day.size else float("nan")
        else:
            oos_p80_exp, oos_var10, oos_p_dd = float("nan"), float("nan"), float("nan")

        # frequência de ativação por segmento
        rules_df["active"] = (rules_df["stake_frac"] > 0) & (rules_df["status"] == "ok")
        act = (
            rules_df.groupby(["bet_type", "dow_pt"], as_index=False)
            .agg(
                active_rate=("active", "mean"),
                mean_stake_frac=("stake_frac", "mean"),
                mean_cutoff=("cutoff", "mean"),
                ok_rate=("status", lambda s: float(np.mean(np.asarray(s) == "ok"))),
            )
            .sort_values(["bet_type", "dow_pt"])
        )

        # alpha summary
        alpha = weekly_df["alpha_global"].to_numpy(dtype=float)
        a_mean = float(np.mean(alpha))
        a_p10, a_p50, a_p90 = (float(np.quantile(alpha, q)) for q in (0.10, 0.50, 0.90))
        a_lt1 = float(np.mean(alpha < 0.999))

        # estabilidade por segmento no OOS (semana-a-semana)
        # alinhamos semanas testadas; faltas => 0
        seg_rows = []
        weeks_test = weekly_df["week"].tolist()
        for rk in sorted(weekly_seg_df["rule_key"].unique().tolist()) if not weekly_seg_df.empty else []:
            s = weekly_seg_df[weekly_seg_df["rule_key"] == rk].set_index("week")["profit_cap2_usd"]
            v = np.array([float(s.get(w, 0.0)) for w in weeks_test], dtype=float)
            m, lo, hi = bootstrap_ci_mean(v, n_boot=N_BOOT, seed=SEED + 31)
            ppos = float(np.mean(v > 0))
            pnegw = float(np.mean(v < 0))
            seg_rows.append({"rule_key": rk, "mean_week_profit": m, "ci95_lo": lo, "ci95_hi": hi, "p_week_pos": ppos, "p_week_neg": pnegw})
        seg_stab = pd.DataFrame(seg_rows).sort_values("mean_week_profit", ascending=False) if seg_rows else pd.DataFrame()
        seg_stab.to_csv(OUT_DIR / f"oos_walkforward_{mode_name}_segment_stability.csv", index=False)

        lines: List[str] = []
        lines.append(f"## OOS walk-forward ({mode_name}) — estratégia completa\n")
        lines.append(f"- Dataset: `{SCORED}`\n")
        lines.append(
            f"- Semanas totais no dataset: **{len(weeks)}**; semanas testadas OOS (WF): **{len(weekly_df)}** "
            f"(a partir de {MIN_GLOBAL_TRAIN_WEEKS} semanas globais de treino; por-segmento exige >= {MIN_SEG_TRAIN_WEEKS})\n"
        )
        if global_risk:
            lines.append(
                f"- **Modo global**: aplica um fator α (0..1) multiplicando todos os stakes para satisfazer constraints do portfólio agregado no treino de cada passo.\n"
            )
        else:
            lines.append("- **Modo segment**: constraints avaliadas por segmento (equivalente ao que já rodamos antes), α=1.\n")

        lines.append("\n### Performance OOS (cap2) — portfólio agregado\n")
        lines.append(f"- **PnL semanal médio (bootstrap IC95%)**: **USD {mean_w:.1f}** (IC95% {lo_w:.1f}..{hi_w:.1f})\n")
        lines.append(f"- **Desvio padrão semanal**: USD {std:.1f}\n")
        lines.append(f"- **P(semana < 0)**: {pneg*100:.1f}%\n")
        lines.append("\n### Performance OOS (cap2) — excluindo semanas sem trades (stake=0)\n")
        lines.append(f"- **PnL semanal médio (bootstrap IC95%)**: **USD {mean_w2:.1f}** (IC95% {lo_w2:.1f}..{hi_w2:.1f})\n")
        lines.append(f"- **Desvio padrão semanal**: USD {std2:.1f}\n")
        lines.append(f"- **P(semana < 0)**: {pneg2*100:.1f}%\n")
        lines.append(f"- **ROI on stake agregado (ponderado)**: {roi_tot:.4f}\n")
        lines.append("\n### Risco no OOS (teste) — portfólio agregado\n")
        lines.append(
            f"- p80(soma stakes/dia) = USD {oos_p80_exp:.0f} (limite=USD {MAX_DAILY_EXPOSURE_FRAC_Q*BANKROLL:.0f})\n"
            f"- VaR10%(PnL diário) = USD {oos_var10:.1f} (limite >= USD {-MAX_DAILY_DRAWDOWN_FRAC*BANKROLL:.0f})\n"
            f"- P(PnL diário <= -25% banca) = {oos_p_dd*100:.1f}% (limite <= {MAX_P_DAILY_DD*100:.0f}%)\n"
        )

        lines.append("\n### Ajuste de stake global (α)\n")
        lines.append(f"- α médio={a_mean:.3f}; p10={a_p10:.3f}; p50={a_p50:.3f}; p90={a_p90:.3f}; P(α<1)={a_lt1*100:.1f}%\n")

        lines.append("\n### Estabilidade OOS da decisão por segmento (frequência de ativação)\n")
        lines.append(f"- Arquivo regras: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_{mode_name}_selected_rules.csv`\n\n")
        for _, r in act.iterrows():
            lines.append(
                f"- **{r['bet_type']} | {r['dow_pt']}**: active_rate={r['active_rate']*100:.1f}%, ok_rate={r['ok_rate']*100:.1f}%, "
                f"stake_frac_médio={r['mean_stake_frac']*100:.2f}%, cutoff_médio={r['mean_cutoff']:.2f}\n"
            )

        if not seg_stab.empty:
            lines.append("\n### Segmentos mais estáveis no OOS (por lucro semanal)\n")
            lines.append(f"- CSV: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_{mode_name}_segment_stability.csv`\n")
            top = seg_stab.head(8)
            for _, rr in top.iterrows():
                lines.append(
                    f"- **{rr['rule_key']}**: mean_week={rr['mean_week_profit']:.1f} (IC95% {rr['ci95_lo']:.1f}..{rr['ci95_hi']:.1f}), "
                    f"P(semana>0)={rr['p_week_pos']*100:.1f}%\n"
                )

        (OUT_DIR / f"oos_walkforward_{mode_name}_strategy.md").write_text("".join(lines), encoding="utf-8")

    # ------------------------
    # Experimentos: Portfolio Bayes Global com balanceamento por viés
    # ------------------------
    if BIAS_EXPERIMENTS_ENABLED:
        for mode_name, segment_calib, disable_top_k in (
            ("global_bayes_biaspen", True, 0),
            ("global_bayes_biasdisable", True, BIAS_DISABLE_TOP_K if BIAS_DISABLE_ENABLED else 0),
        ):
            if mode_name.endswith("biasdisable") and not BIAS_DISABLE_ENABLED:
                continue
            rules_df, weekly_df, weekly_seg_df, weekly_global_df, daily_df = run_walkforward(
                global_risk=True,
                bayes_select=True,
                segment_calib=segment_calib,
                disable_top_k=int(disable_top_k),
                train_window_weeks=None,
                regime_lookback_weeks=None,
                regime_alpha_bad=1.0,
            )
            rules_df.to_csv(OUT_DIR / f"oos_walkforward_{mode_name}_selected_rules.csv", index=False)
            weekly_df.to_csv(OUT_DIR / f"oos_walkforward_{mode_name}_weekly.csv", index=False)
            weekly_seg_df.to_csv(OUT_DIR / f"oos_walkforward_{mode_name}_weekly_by_segment.csv", index=False)
            weekly_global_df.to_csv(OUT_DIR / f"oos_walkforward_{mode_name}_train_global_metrics.csv", index=False)
            daily_df.to_csv(OUT_DIR / f"oos_walkforward_{mode_name}_daily.csv", index=False)
            _summarize_mode(mode_name, weekly_df)

        pd.DataFrame(results_summary_rows).to_csv(OUT_DIR / "bias_balance_experiment_summary.csv", index=False)

    # ------------------------
    # Experimentos adicionais (recência + gating + robustez)
    # ------------------------
    def _with_globals(**kwargs):
        class _Ctx:
            def __enter__(self0):
                self0.prev = {
                    "ROBUST_CUTOFF_ENABLED": globals().get("ROBUST_CUTOFF_ENABLED"),
                    "HYSTERESIS_ENABLED": globals().get("HYSTERESIS_ENABLED"),
                    "REQUIRE_POST_Q_OBJ_POS": globals().get("REQUIRE_POST_Q_OBJ_POS"),
                    "REQUIRE_POST_Q_GATE_POS": globals().get("REQUIRE_POST_Q_GATE_POS"),
                    "POST_Q_OBJ": globals().get("POST_Q_OBJ"),
                    "MIN_POST_P_MEAN_POS": globals().get("MIN_POST_P_MEAN_POS"),
                }
                for k, v in kwargs.items():
                    globals()[k] = v
                return self0

            def __exit__(self0, exc_type, exc, tb):
                for k, v in self0.prev.items():
                    globals()[k] = v
                return False

        return _Ctx()

    # 12 semanas rolling (mais peso no recente por construção)
    for exp_name, win, robust, hyst, qpos in (
        ("global_bayes_roll12", 12, False, False, False),
        ("global_bayes_roll12_robust", 12, True, True, False),
        ("global_bayes_roll12_robust_qpos", 12, True, True, True),
        ("global_bayes_roll12_robust_qgate30", 12, True, True, False),
    ):
        if only_mode and exp_name != only_mode:
            continue
        req_gate = (exp_name == "global_bayes_roll12_robust_qgate30")
        with _with_globals(
            ROBUST_CUTOFF_ENABLED=bool(robust),
            HYSTERESIS_ENABLED=bool(hyst),
            REQUIRE_POST_Q_OBJ_POS=bool(qpos),
            REQUIRE_POST_Q_GATE_POS=bool(req_gate),
        ):
            rules_df, weekly_df, weekly_seg_df, weekly_global_df, daily_df = run_walkforward(
                global_risk=True,
                bayes_select=True,
                segment_calib=False,
                disable_top_k=0,
                train_window_weeks=int(win),
                regime_lookback_weeks=None,
                regime_alpha_bad=1.0,
            )
            rules_df.to_csv(OUT_DIR / f"oos_walkforward_{exp_name}_selected_rules.csv", index=False)
            weekly_df.to_csv(OUT_DIR / f"oos_walkforward_{exp_name}_weekly.csv", index=False)
            weekly_seg_df.to_csv(OUT_DIR / f"oos_walkforward_{exp_name}_weekly_by_segment.csv", index=False)
            weekly_global_df.to_csv(OUT_DIR / f"oos_walkforward_{exp_name}_train_global_metrics.csv", index=False)
            daily_df.to_csv(OUT_DIR / f"oos_walkforward_{exp_name}_daily.csv", index=False)
            _summarize_mode(exp_name, weekly_df)

    pd.DataFrame(results_summary_rows).to_csv(OUT_DIR / "bias_balance_experiment_summary.csv", index=False)

    # ------------------------
    # Modo escolhido (pedido): p10_p70 em cima de roll12 + robust + histerese
    # ------------------------
    with _with_globals(
        ROBUST_CUTOFF_ENABLED=True,
        HYSTERESIS_ENABLED=True,
        REQUIRE_POST_Q_OBJ_POS=False,
        REQUIRE_POST_Q_GATE_POS=False,
        POST_Q_OBJ=0.10,
        MIN_POST_P_MEAN_POS=0.70,
    ):
        exp_name = "global_bayes_roll12_robust_p10_p70"
        if (not only_mode) or (only_mode == exp_name):
            rules_df, weekly_df, weekly_seg_df, weekly_global_df, daily_df = run_walkforward(
                global_risk=True,
                bayes_select=True,
                segment_calib=False,
                disable_top_k=0,
                train_window_weeks=12,
                regime_lookback_weeks=None,
                regime_alpha_bad=1.0,
            )
            rules_df.to_csv(OUT_DIR / f"oos_walkforward_{exp_name}_selected_rules.csv", index=False)
            weekly_df.to_csv(OUT_DIR / f"oos_walkforward_{exp_name}_weekly.csv", index=False)
            weekly_seg_df.to_csv(OUT_DIR / f"oos_walkforward_{exp_name}_weekly_by_segment.csv", index=False)
            weekly_global_df.to_csv(OUT_DIR / f"oos_walkforward_{exp_name}_train_global_metrics.csv", index=False)
            daily_df.to_csv(OUT_DIR / f"oos_walkforward_{exp_name}_daily.csv", index=False)
            _summarize_mode(exp_name, weekly_df)

    # ------------------------
    # Experimento: cap gating por segmento (otimiza também cap_max)
    # ------------------------
    with _with_globals(
        ROBUST_CUTOFF_ENABLED=True,
        HYSTERESIS_ENABLED=True,
        REQUIRE_POST_Q_OBJ_POS=False,
        REQUIRE_POST_Q_GATE_POS=False,
        POST_Q_OBJ=0.10,
        MIN_POST_P_MEAN_POS=0.70,
        CAP_GATING_ENABLED=True,
        BAYES_N=2000,
    ):
        exp_name = "global_bayes_roll12_robust_p10_p70_capgate"
        if only_mode == exp_name:
            rules_df, weekly_df, weekly_seg_df, weekly_global_df, daily_df = run_walkforward(
                global_risk=True,
                bayes_select=True,
                segment_calib=False,
                disable_top_k=0,
                train_window_weeks=12,
                regime_lookback_weeks=None,
                regime_alpha_bad=1.0,
            )
            rules_df.to_csv(OUT_DIR / f"oos_walkforward_{exp_name}_selected_rules.csv", index=False)
            weekly_df.to_csv(OUT_DIR / f"oos_walkforward_{exp_name}_weekly.csv", index=False)
            weekly_seg_df.to_csv(OUT_DIR / f"oos_walkforward_{exp_name}_weekly_by_segment.csv", index=False)
            weekly_global_df.to_csv(OUT_DIR / f"oos_walkforward_{exp_name}_train_global_metrics.csv", index=False)
            daily_df.to_csv(OUT_DIR / f"oos_walkforward_{exp_name}_daily.csv", index=False)
            _summarize_mode(exp_name, weekly_df)

    # ------------------------
    # Experimento: cap_max fixo por segmento (escolhe uma vez e mantém)
    # ------------------------
    with _with_globals(
        ROBUST_CUTOFF_ENABLED=True,
        HYSTERESIS_ENABLED=True,
        REQUIRE_POST_Q_OBJ_POS=False,
        REQUIRE_POST_Q_GATE_POS=False,
        POST_Q_OBJ=0.10,
        MIN_POST_P_MEAN_POS=0.70,
        CAP_FIXED_PER_SEGMENT_ENABLED=True,
        BAYES_N=2000,
    ):
        exp_name = "global_bayes_roll12_robust_p10_p70_capfixed"
        if only_mode == exp_name:
            rules_df, weekly_df, weekly_seg_df, weekly_global_df, daily_df = run_walkforward(
                global_risk=True,
                bayes_select=True,
                segment_calib=False,
                disable_top_k=0,
                train_window_weeks=12,
                regime_lookback_weeks=None,
                regime_alpha_bad=1.0,
            )
            rules_df.to_csv(OUT_DIR / f"oos_walkforward_{exp_name}_selected_rules.csv", index=False)
            weekly_df.to_csv(OUT_DIR / f"oos_walkforward_{exp_name}_weekly.csv", index=False)
            weekly_seg_df.to_csv(OUT_DIR / f"oos_walkforward_{exp_name}_weekly_by_segment.csv", index=False)
            weekly_global_df.to_csv(OUT_DIR / f"oos_walkforward_{exp_name}_train_global_metrics.csv", index=False)
            daily_df.to_csv(OUT_DIR / f"oos_walkforward_{exp_name}_daily.csv", index=False)
            _summarize_mode(exp_name, weekly_df)

    pd.DataFrame(results_summary_rows).to_csv(OUT_DIR / "bias_balance_experiment_summary.csv", index=False)

    # Regime overlay: se ROI recente (últimas 4 semanas do treino) < 0, reduzir alpha.
    for exp_name, alpha_bad in (
        ("global_bayes_roll12_robust_regime4_off", 0.0),
        ("global_bayes_roll12_robust_regime4_half", 0.5),
    ):
        if only_mode and exp_name != only_mode:
            continue
        with _with_globals(ROBUST_CUTOFF_ENABLED=True, HYSTERESIS_ENABLED=True, REQUIRE_POST_Q_OBJ_POS=False, REQUIRE_POST_Q_GATE_POS=False):
            rules_df, weekly_df, weekly_seg_df, weekly_global_df, daily_df = run_walkforward(
                global_risk=True,
                bayes_select=True,
                segment_calib=False,
                disable_top_k=0,
                train_window_weeks=12,
                regime_lookback_weeks=4,
                regime_alpha_bad=float(alpha_bad),
            )
            rules_df.to_csv(OUT_DIR / f"oos_walkforward_{exp_name}_selected_rules.csv", index=False)
            weekly_df.to_csv(OUT_DIR / f"oos_walkforward_{exp_name}_weekly.csv", index=False)
            weekly_seg_df.to_csv(OUT_DIR / f"oos_walkforward_{exp_name}_weekly_by_segment.csv", index=False)
            weekly_global_df.to_csv(OUT_DIR / f"oos_walkforward_{exp_name}_train_global_metrics.csv", index=False)
            daily_df.to_csv(OUT_DIR / f"oos_walkforward_{exp_name}_daily.csv", index=False)
            _summarize_mode(exp_name, weekly_df)

    pd.DataFrame(results_summary_rows).to_csv(OUT_DIR / "bias_balance_experiment_summary.csv", index=False)

    print(str(OUT_DIR / "oos_walkforward_global_strategy.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

