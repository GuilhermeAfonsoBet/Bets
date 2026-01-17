#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reotimização do portfólio usando o score operacional do robô:
`proba_raw` = model.predict_proba(...) com clip em [calib_floor, 1-calib_floor].

Objetivo: encontrar (por dia) um cutoff de score e um stake-fraction (<= max_frac)
que maximizem Sharpe (média/std) do lucro mensal no período de treino.

Inclui Quinta-feira aplicando, como aproximação, um dos modelos existentes
(segunda/terça/quarta). O modelo de quinta é escolhido por performance no treino.

Saídas:
 - analysis_proba_raw/scored_dedup_proba_raw.csv
 - analysis_proba_raw/portfolio_proba_raw_reoptimized.json
 - analysis_proba_raw/summary_proba_raw.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib


EXCEL_PATH_DEFAULT = "/workspace/ResumoApostas_PBI_final_17.01.2026.xlsx"
SHEET_NAME_DEFAULT = "ResumoApostas (2)"

BANKROLL_DEFAULT = 2300.0
MAX_FRAC_DEFAULT = 0.07
CALIB_FLOOR_DEFAULT = 0.005

# Períodos (seguindo a base onde a massa está em 2025-10..2025-12 e OOS em 2026-01)
TRAIN_START_DEFAULT = "2025-10-01"
TRAIN_END_DEFAULT = "2025-12-31"
TEST_START_DEFAULT = "2026-01-01"


WEEKDAY_PT = [
    "segunda-feira",
    "terça-feira",
    "quarta-feira",
    "quinta-feira",
    "sexta-feira",
    "sábado",
    "domingo",
]


def infer_turno_utc(dt: pd.Timestamp) -> str:
    h = int(dt.hour)
    if 6 <= h <= 11:
        return "manhã"
    if 12 <= h <= 17:
        return "tarde"
    return "noite"


# ---------------------------------------------------------------------
# Compat patch (sklearn 1.4.x models running in sklearn 1.8.x)
# ---------------------------------------------------------------------
def _walk_estimators(est):
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
    except Exception:
        Pipeline = ()
        ColumnTransformer = ()

    yield est
    if isinstance(est, Pipeline):
        for _, sub in est.steps:
            yield from _walk_estimators(sub)
    elif isinstance(est, ColumnTransformer):
        for _, sub, _ in est.transformers_:
            if sub in ("drop", "passthrough"):
                continue
            yield from _walk_estimators(sub)
    elif hasattr(est, "estimators"):
        for _, sub in getattr(est, "estimators", []):
            yield from _walk_estimators(sub)


def patch_sklearn_compat(est):
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OneHotEncoder
    except Exception:
        return est

    for obj in _walk_estimators(est):
        # Pipeline (sklearn 1.8 adicionou atributo)
        try:
            if isinstance(obj, Pipeline) and not hasattr(obj, "transform_input"):
                setattr(obj, "transform_input", None)
        except Exception:
            pass

        # SimpleImputer: mudaram atributos internos
        try:
            if isinstance(obj, SimpleImputer):
                if not hasattr(obj, "keep_empty_features"):
                    setattr(obj, "keep_empty_features", False)
                if not hasattr(obj, "_fill_dtype") and hasattr(obj, "_fit_dtype"):
                    setattr(obj, "_fill_dtype", getattr(obj, "_fit_dtype"))
        except Exception:
            pass

        # OneHotEncoder: sparse_output e atributo interno
        try:
            if isinstance(obj, OneHotEncoder):
                if not hasattr(obj, "sparse_output"):
                    setattr(obj, "sparse_output", bool(getattr(obj, "sparse", True)))
                if not hasattr(obj, "_drop_idx_after_grouping"):
                    setattr(obj, "_drop_idx_after_grouping", None)
        except Exception:
            pass

    return est


# ---------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------
def load_base(path: str, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    if df.empty:
        raise ValueError(f"Sheet vazia: {sheet!r} em {path}")
    return df


def dedup_by_id_aposta(df: pd.DataFrame) -> pd.DataFrame:
    # Mantém a última linha por ID Aposta.
    # Ordenação prioriza timestamp de aposta e, em seguida, o índice original.
    df = df.copy()
    if "BIA_ApostaUTC" not in df.columns:
        raise ValueError("Coluna 'BIA_ApostaUTC' não encontrada.")
    if "ID Aposta" not in df.columns:
        raise ValueError("Coluna 'ID Aposta' não encontrada.")

    df["_idx0"] = np.arange(len(df), dtype=int)
    df = df.sort_values(["ID Aposta", "BIA_ApostaUTC", "_idx0"], ascending=[True, True, True])
    out = df.groupby("ID Aposta", as_index=False).tail(1).drop(columns=["_idx0"])
    return out.reset_index(drop=True)


def build_features_weekday_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói as colunas esperadas pelos modelos `model_logit_{segunda,terca,quarta}.joblib`
    (nomes *exatos*).
    """
    required_map = {
        "Número de casas disponíveis no momento da aposta": "ApostaLive.Número de casas disponíveis no momento da aposta",
        "Dif percent maior odd e segunda maior": "ApostaLive.Dif % maior odd e segunda maior",
        "Dif percent maior odd e odd mediana": "ApostaLive.Dif % maior odd e odd mediana",
        "Dif Odds RB E BIA": "Dif Odds RB & BIA",
        "MinutesToMatchStart": "RebelBetting.MinutesToMatchStart",
        "TempoApostas.Tempo total bot": "TempoApostas.Tempo total bot",
        "Subtipo da Aposta": "Subtipo da Aposta",
        "Casa aposta vencedora": "ApostaLive.Casa aposta vencedora",
    }
    missing = [src for src in required_map.values() if src not in df.columns]
    if missing:
        raise ValueError(f"Faltam colunas para scoring: {missing}")

    out = pd.DataFrame(index=df.index)
    for dst, src in required_map.items():
        out[dst] = df[src]

    # Dia/turno em UTC a partir de BIA_ApostaUTC
    dt = pd.to_datetime(df["BIA_ApostaUTC"], errors="coerce")
    out["Dia Semana Aposta (UTC)"] = dt.dt.weekday.map(lambda x: WEEKDAY_PT[int(x)] if pd.notna(x) else None)
    out["Turno Aposta (UTC)"] = dt.apply(lambda x: infer_turno_utc(x) if pd.notna(x) else None)

    return out


def build_features_segqui(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói as colunas esperadas por `model_logit_SegQui.joblib` (nomes *exatos*).
    """
    required_map = {
        "Número de casas disponíveis no momento da aposta": "ApostaLive.Número de casas disponíveis no momento da aposta",
        "Dif % maior odd e segunda maior": "ApostaLive.Dif % maior odd e segunda maior",
        "Dif % maior odd e odd mediana": "ApostaLive.Dif % maior odd e odd mediana",
        "Dif Odds RB & BIA": "Dif Odds RB & BIA",
        "MinutesToMatchStart": "RebelBetting.MinutesToMatchStart",
        "TempoApostas.Tempo total bot": "TempoApostas.Tempo total bot",
        "Subtipo da Aposta": "Subtipo da Aposta",
        "Casa aposta vencedora": "ApostaLive.Casa aposta vencedora",
    }
    missing = [src for src in required_map.values() if src not in df.columns]
    if missing:
        raise ValueError(f"Faltam colunas para scoring SegQui: {missing}")

    out = pd.DataFrame(index=df.index)
    for dst, src in required_map.items():
        out[dst] = df[src]

    # Dia/turno em UTC a partir de BIA_ApostaUTC
    dt = pd.to_datetime(df["BIA_ApostaUTC"], errors="coerce")
    out["Dia Semana Aposta (UTC)"] = dt.dt.weekday.map(lambda x: WEEKDAY_PT[int(x)] if pd.notna(x) else None)
    out["Turno Aposta (UTC)"] = dt.apply(lambda x: infer_turno_utc(x) if pd.notna(x) else None)

    return out


def is_ft(tipo: str) -> bool:
    if tipo is None or (isinstance(tipo, float) and np.isnan(tipo)):
        return False
    s = str(tipo).lower()
    return "first half" not in s


def stake_cap_from_house(row: pd.Series) -> float:
    cap = row.get("ApostaLive.Stake máximo da casa da aposta (USD)", np.nan)
    try:
        cap = float(cap)
    except Exception:
        cap = float("nan")
    if not np.isfinite(cap) or cap <= 0:
        return float("inf")
    return cap


# ---------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Models:
    segunda: object
    terca: object
    quarta: object
    segqui: object


def load_models() -> Models:
    m_seg = patch_sklearn_compat(joblib.load("/workspace/model_logit_segunda.joblib"))
    m_ter = patch_sklearn_compat(joblib.load("/workspace/model_logit_terca.joblib"))
    m_qua = patch_sklearn_compat(joblib.load("/workspace/model_logit_quarta.joblib"))
    m_segqui = patch_sklearn_compat(joblib.load("/workspace/model_logit_SegQui.joblib"))
    return Models(segunda=m_seg, terca=m_ter, quarta=m_qua, segqui=m_segqui)


def clip_proba(p: np.ndarray, floor: float) -> np.ndarray:
    lo = float(floor)
    hi = 1.0 - float(floor)
    return np.clip(p.astype(float), lo, hi)


def score_proba_raw(df_feat: pd.DataFrame, models: Models, calib_floor: float) -> pd.DataFrame:
    """
    Retorna DataFrame com colunas:
      - proba_raw_segunda / _terca / _quarta: score se aplicado o modelo correspondente
      - proba_raw_operacional: score operacional por DOW (seg->seg, ter->ter, qua->qua, qui->(placeholder))
    Para quinta-feira, NÃO existe modelo no runtime atual; por isso geramos as 3 alternativas.
    """
    feat = df_feat.copy()

    # Força Dia Semana Aposta (UTC) compatível com cada modelo (cada um viu só um DOW)
    feat_seg = feat.copy()
    feat_seg["Dia Semana Aposta (UTC)"] = "segunda-feira"
    feat_ter = feat.copy()
    feat_ter["Dia Semana Aposta (UTC)"] = "terça-feira"
    feat_qua = feat.copy()
    feat_qua["Dia Semana Aposta (UTC)"] = "quarta-feira"

    p_seg = clip_proba(models.segunda.predict_proba(feat_seg)[:, 1], calib_floor)
    p_ter = clip_proba(models.terca.predict_proba(feat_ter)[:, 1], calib_floor)
    p_qua = clip_proba(models.quarta.predict_proba(feat_qua)[:, 1], calib_floor)

    out = pd.DataFrame(
        {
            "proba_raw_segunda": p_seg,
            "proba_raw_terca": p_ter,
            "proba_raw_quarta": p_qua,
        },
        index=feat.index,
    )

    # Score operacional: usa o modelo do dia; quinta fica NaN (vamos escolher na otimização)
    dow = feat["Dia Semana Aposta (UTC)"].astype("string")
    proba_oper = np.full(len(feat), np.nan, dtype=float)
    proba_oper[dow == "segunda-feira"] = p_seg[dow == "segunda-feira"]
    proba_oper[dow == "terça-feira"] = p_ter[dow == "terça-feira"]
    proba_oper[dow == "quarta-feira"] = p_qua[dow == "quarta-feira"]
    out["proba_raw_operacional"] = proba_oper
    return out


def score_proba_raw_segqui(df_feat_segqui: pd.DataFrame, models: Models, calib_floor: float) -> pd.Series:
    """
    Score do modelo SegQui (Seg..Qui), no formato operacional (predict_proba com clip).
    """
    p = clip_proba(models.segqui.predict_proba(df_feat_segqui)[:, 1], calib_floor)
    return pd.Series(p, index=df_feat_segqui.index, name="proba_raw_segqui")


# ---------------------------------------------------------------------
# Optimization / metrics
# ---------------------------------------------------------------------
def monthly_profit_series(
    df: pd.DataFrame,
    score_col: str,
    cutoff: float,
    stake_frac: float,
    bankroll: float,
    only_ft: bool = True,
) -> pd.Series:
    """
    Retorna série (index: mês) com lucro em USD no mês.
    Usa ROI Real como multiplicador de lucro: lucro = stake * ROI.
    """
    x = df.copy()
    if only_ft:
        x = x[x["is_ft"]]
    x = x[np.isfinite(x[score_col].to_numpy(dtype=float))]
    x = x[x[score_col] >= float(cutoff)]
    x = x[np.isfinite(x["ROI Real"].to_numpy(dtype=float))]

    if x.empty:
        return pd.Series(dtype=float)

    stake0 = float(stake_frac) * float(bankroll)
    # house cap (inf quando cap<=0)
    stake = np.minimum(stake0, x["house_cap"].to_numpy(dtype=float))
    profit = stake * x["ROI Real"].to_numpy(dtype=float)

    month = pd.to_datetime(x["BIA_ApostaUTC"]).dt.to_period("M").astype(str).to_numpy()
    s = pd.Series(profit, index=month).groupby(level=0).sum()
    s.index.name = "month"
    s.name = "profit_usd"
    return s.sort_index()


def monthly_profit_and_counts(
    df: pd.DataFrame,
    score_col: str,
    cutoff: float,
    stake_frac: float,
    bankroll: float,
    only_ft: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    """
    Igual ao monthly_profit_series, mas também retorna contagem de apostas por mês.
    """
    x = df.copy()
    if only_ft:
        x = x[x["is_ft"]]
    x = x[np.isfinite(x[score_col].to_numpy(dtype=float))]
    x = x[x[score_col] >= float(cutoff)]
    x = x[np.isfinite(x["ROI Real"].to_numpy(dtype=float))]

    if x.empty:
        return pd.Series(dtype=float), pd.Series(dtype=int)

    stake0 = float(stake_frac) * float(bankroll)
    stake = np.minimum(stake0, x["house_cap"].to_numpy(dtype=float))
    profit = stake * x["ROI Real"].to_numpy(dtype=float)
    month = pd.to_datetime(x["BIA_ApostaUTC"]).dt.to_period("M").astype(str).to_numpy()

    profit_s = pd.Series(profit, index=month).groupby(level=0).sum().sort_index()
    profit_s.index.name = "month"
    profit_s.name = "profit_usd"

    count_s = pd.Series(np.ones_like(profit, dtype=int), index=month).groupby(level=0).sum().sort_index()
    count_s.index.name = "month"
    count_s.name = "n_bets"
    return profit_s, count_s


@dataclass(frozen=True)
class OptResult:
    day: str
    score_col: str
    cutoff: float
    stake_frac: float
    train_mean: float
    train_std: float
    train_sharpe: float


def optimize_day(
    df_train: pd.DataFrame,
    day: str,
    score_cols: List[str],
    bankroll: float,
    max_frac: float,
) -> OptResult:
    # grid simples
    stake_fracs = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, float(max_frac)])
    cutoffs = np.round(np.arange(0.05, 0.951, 0.01), 2)

    best: Optional[OptResult] = None
    subset = df_train[df_train["dow_pt"] == day]

    # Regras de robustez: evitar \"ótimos\" com 1 aposta/mês.
    min_bets_total = 30 if day in ("segunda-feira", "quarta-feira") else 15
    min_bets_per_month = 5 if day in ("segunda-feira", "quarta-feira") else 3
    min_months_covered = 2

    for sc in score_cols:
        # garante que a coluna existe
        if sc not in subset.columns:
            continue

        for f in stake_fracs:
            for c in cutoffs:
                ms, cs_month = monthly_profit_and_counts(subset, sc, c, f, bankroll, only_ft=True)
                if ms.empty:
                    continue

                # robustez mínima
                total_bets = int(cs_month.sum()) if not cs_month.empty else 0
                months_cov = int(cs_month.size)
                min_month_bets = int(cs_month.min()) if months_cov > 0 else 0
                if total_bets < min_bets_total:
                    continue
                if months_cov < min_months_covered:
                    continue
                if min_month_bets < min_bets_per_month:
                    continue

                mean = float(ms.mean())
                std = float(ms.std(ddof=1)) if ms.size >= 2 else 0.0
                sharpe = float(mean / std) if std > 0 else (float("inf") if mean > 0 else -float("inf"))
                # objetivo mais conservador: penaliza volatilidade
                score_obj = mean - 0.25 * std

                cand = OptResult(day=day, score_col=sc, cutoff=float(c), stake_frac=float(f), train_mean=mean, train_std=std, train_sharpe=sharpe)
                if best is None:
                    best = cand
                else:
                    # prioridade: score_obj, depois mean
                    best_score_obj = best.train_mean - 0.25 * best.train_std
                    if (score_obj > best_score_obj) or (score_obj == best_score_obj and cand.train_mean > best.train_mean):
                        best = cand

    if best is None:
        # fallback: nada selecionável
        return OptResult(day=day, score_col=score_cols[0], cutoff=1.0, stake_frac=float(max_frac), train_mean=0.0, train_std=0.0, train_sharpe=-float("inf"))
    return best


def eval_portfolio(df: pd.DataFrame, cfg: Dict[str, Dict[str, float]], bankroll: float) -> Dict[str, float]:
    """
    Avalia portfólio agregado (Seg/Ter/Qua/Qui) em df.
    Retorna métricas agregadas simples.
    """
    profits = []
    for day, params in cfg.items():
        sc = params["score_col"]
        cutoff = float(params["cutoff"])
        frac = float(params["stake_frac"])
        m = monthly_profit_series(df[df["dow_pt"] == day], sc, cutoff, frac, bankroll, only_ft=True)
        if not m.empty:
            profits.append(m)

    if not profits:
        return {"mean_monthly_profit": 0.0, "std_monthly_profit": 0.0, "p_monthly_negative": 0.0, "n_months": 0}

    total = pd.concat(profits, axis=1).fillna(0.0).sum(axis=1)
    mean = float(total.mean())
    std = float(total.std(ddof=1)) if total.size >= 2 else 0.0
    pneg = float((total < 0).mean()) if total.size >= 1 else 0.0
    return {
        "mean_monthly_profit": mean,
        "std_monthly_profit": std,
        "p_monthly_negative": pneg,
        "n_months": int(total.size),
    }


def main() -> int:
    out_dir = Path("/workspace/analysis_proba_raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    df0 = load_base(EXCEL_PATH_DEFAULT, SHEET_NAME_DEFAULT)
    df0 = df0[pd.to_datetime(df0["BIA_ApostaUTC"], errors="coerce") >= pd.Timestamp("2024-01-01")]
    df = dedup_by_id_aposta(df0)

    # flags e colunas auxiliares
    df["is_ft"] = df["Tipo Aposta"].apply(is_ft)
    dt = pd.to_datetime(df["BIA_ApostaUTC"], errors="coerce")
    df["dow_pt"] = dt.dt.weekday.map(lambda x: WEEKDAY_PT[int(x)] if pd.notna(x) else None)
    df["house_cap"] = df.apply(stake_cap_from_house, axis=1)

    # features + scoring
    feat_week = build_features_weekday_models(df)
    feat_segqui = build_features_segqui(df)
    models = load_models()
    scored_week = score_proba_raw(feat_week, models, CALIB_FLOOR_DEFAULT)
    scored_segqui = score_proba_raw_segqui(feat_segqui, models, CALIB_FLOOR_DEFAULT)

    df_scored = pd.concat(
        [
            df.reset_index(drop=True),
            feat_week.reset_index(drop=True),
            scored_week.reset_index(drop=True),
            scored_segqui.reset_index(drop=True),
        ],
        axis=1,
    )

    # splits
    train_start = pd.Timestamp(TRAIN_START_DEFAULT)
    train_end = pd.Timestamp(TRAIN_END_DEFAULT) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    test_start = pd.Timestamp(TEST_START_DEFAULT)

    m_train = (pd.to_datetime(df_scored["BIA_ApostaUTC"]) >= train_start) & (pd.to_datetime(df_scored["BIA_ApostaUTC"]) <= train_end)
    m_test = pd.to_datetime(df_scored["BIA_ApostaUTC"]) >= test_start

    df_train = df_scored[m_train].copy()
    df_test = df_scored[m_test].copy()

    # otimização por dia
    results: Dict[str, OptResult] = {}
    # seg/ter/qua usam seu próprio score; qui usa SegQui
    results["segunda-feira"] = optimize_day(df_train, "segunda-feira", ["proba_raw_segunda"], BANKROLL_DEFAULT, MAX_FRAC_DEFAULT)
    results["terça-feira"] = optimize_day(df_train, "terça-feira", ["proba_raw_terca"], BANKROLL_DEFAULT, MAX_FRAC_DEFAULT)
    results["quarta-feira"] = optimize_day(df_train, "quarta-feira", ["proba_raw_quarta"], BANKROLL_DEFAULT, MAX_FRAC_DEFAULT)
    results["quinta-feira"] = optimize_day(
        df_train,
        "quinta-feira",
        ["proba_raw_segqui"],
        BANKROLL_DEFAULT,
        MAX_FRAC_DEFAULT,
    )

    cfg = {
        day: {"score_col": r.score_col, "cutoff": r.cutoff, "stake_frac": r.stake_frac}
        for day, r in results.items()
    }

    # métricas agregadas (treino e OOS)
    train_metrics = eval_portfolio(df_train, cfg, BANKROLL_DEFAULT)
    test_metrics = eval_portfolio(df_test, cfg, BANKROLL_DEFAULT)

    # salvar base scored deduplicada
    df_scored.to_csv(out_dir / "scored_dedup_proba_raw.csv", index=False)

    # salvar cfg
    cfg_out = {
        "bankroll": BANKROLL_DEFAULT,
        "max_frac_per_bet": MAX_FRAC_DEFAULT,
        "calib_floor": CALIB_FLOOR_DEFAULT,
        "train_period": {"start": TRAIN_START_DEFAULT, "end": TRAIN_END_DEFAULT},
        "test_period": {"start": TEST_START_DEFAULT},
        "per_day": cfg,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
    (out_dir / "portfolio_proba_raw_reoptimized.json").write_text(json.dumps(cfg_out, ensure_ascii=False, indent=2), encoding="utf-8")

    # summary md
    lines = []
    lines.append("## Portfólio reotimizado (proba_raw)\n")
    lines.append(f"- Banca considerada: **USD {BANKROLL_DEFAULT:,.0f}**\n")
    lines.append(f"- Max por aposta: **{MAX_FRAC_DEFAULT*100:.1f}%** (USD {BANKROLL_DEFAULT*MAX_FRAC_DEFAULT:,.0f})\n")
    lines.append(f"- Clip do score (calib_floor): **{CALIB_FLOOR_DEFAULT}**\n")
    lines.append(f"- Treino: **{TRAIN_START_DEFAULT}..{TRAIN_END_DEFAULT}**; OOS: **>= {TEST_START_DEFAULT}**\n")
    lines.append("\n### Regras por dia (FT)\n")
    for day in ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira"]:
        r = results[day]
        lines.append(f"- **{day}**: usar `{r.score_col}`, cutoff **{r.cutoff:.2f}**, stake **{r.stake_frac*100:.1f}%**\n")
    lines.append("\n### Métricas agregadas (treino)\n")
    lines.append(f"- **lucro mensal médio**: USD {train_metrics['mean_monthly_profit']:.2f}\n")
    lines.append(f"- **desvio-padrão mensal**: USD {train_metrics['std_monthly_profit']:.2f}\n")
    lines.append(f"- **P(mês < 0)**: {train_metrics['p_monthly_negative']*100:.1f}% (n={train_metrics['n_months']})\n")
    lines.append("\n### Métricas agregadas (OOS)\n")
    lines.append(f"- **lucro mensal médio**: USD {test_metrics['mean_monthly_profit']:.2f}\n")
    lines.append(f"- **desvio-padrão mensal**: USD {test_metrics['std_monthly_profit']:.2f}\n")
    lines.append(f"- **P(mês < 0)**: {test_metrics['p_monthly_negative']*100:.1f}% (n={test_metrics['n_months']})\n")
    lines.append("\n### Nota operacional (quinta-feira)\n")
    lines.append(
        "- Para quinta-feira, este portfólio usa o modelo **`model_logit_SegQui.joblib`** (score `proba_raw_segqui`). "
        "Isso permite operar quinta sem precisar \"fingir\" que é quarta no payload.\n"
    )

    (out_dir / "summary_proba_raw.md").write_text("".join(lines), encoding="utf-8")

    print(str(out_dir / "summary_proba_raw.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
