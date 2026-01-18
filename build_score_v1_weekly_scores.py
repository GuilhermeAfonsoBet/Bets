#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera scores v1 de forma HONESTA no tempo (walk-forward por semana):
- Para cada semana w (a partir da MIN_GLOBAL_TRAIN_WEEKS), treina usando somente as 12 semanas anteriores
  e prevê apenas as linhas de w.

Saída:
- /workspace/analysis_proba_raw/pro_portfolio_all/scored_with_score_v1_wf12.csv (somente colunas id+scores)

Objetivo:
Ter um score por linha, ex-ante, sem vazamento temporal, para plugar no otimizador p10_p70.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import evaluate_oos_walkforward_strategy as wf


SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")
OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")
OUT = OUT_DIR / "scored_with_score_v1_wf12.csv"

TRAIN_WINDOW_WEEKS = 12
MAX_WEEKS = 10_000  # gerar scores para todo o período OOS


def _safe_get(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] if col in df.columns else pd.Series(np.nan, index=df.index)


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Conjunto ex-ante conservador (sem resultado/ROI/CLV/odd fechamento).
    """
    X = pd.DataFrame(index=df.index)
    # odds snapshot
    X["odd_rb"] = pd.to_numeric(_safe_get(df, "Odd Indicada no RB"), errors="coerce")
    X["odd_rb2"] = pd.to_numeric(_safe_get(df, "RebelBetting.Odds"), errors="coerce")
    X["odd_got"] = pd.to_numeric(_safe_get(df, "BetinAsia.got price"), errors="coerce")
    X["dif_odds_rb_bia"] = pd.to_numeric(_safe_get(df, "Dif Odds RB & BIA"), errors="coerce")
    X["rb_percentage"] = pd.to_numeric(_safe_get(df, "RebelBetting.Percentage"), errors="coerce")

    # market snapshot
    X["n_books"] = pd.to_numeric(_safe_get(df, "ApostaLive.Número de casas disponíveis no momento da aposta"), errors="coerce")
    X["stake_max_house"] = pd.to_numeric(_safe_get(df, "ApostaLive.Stake máximo da casa da aposta (USD)"), errors="coerce")
    X["dif_top2"] = pd.to_numeric(_safe_get(df, "ApostaLive.Dif % maior odd e segunda maior"), errors="coerce")
    X["dif_med"] = pd.to_numeric(_safe_get(df, "ApostaLive.Dif % maior odd e odd mediana"), errors="coerce")
    X["aux1_maior_odd"] = pd.to_numeric(_safe_get(df, "ApostaLive.Aux1 - maior odd"), errors="coerce")

    # timing
    X["mins_to_start"] = pd.to_numeric(_safe_get(df, "RebelBetting.MinutesToMatchStart"), errors="coerce")
    X["bot_total"] = pd.to_numeric(_safe_get(df, "TempoApostas.Tempo total bot"), errors="coerce")

    # categorical
    X["tipo_aposta"] = _safe_get(df, "Tipo Aposta").astype(str)
    X["subtipo"] = _safe_get(df, "Subtipo da Aposta").astype(str)
    X["jogo_int_ou_intervalo"] = _safe_get(df, "Jogo inteiro / intervalo").astype(str)
    X["esporte"] = _safe_get(df, "Esporte").astype(str)
    X["bet_type"] = _safe_get(df, "bet_type").astype(str)
    X["dow_pt"] = _safe_get(df, "dow_pt").astype(str)
    X["book"] = _safe_get(df, "ApostaLive.Casa aposta vencedora").astype(str)

    # time features (UTC)
    dt = pd.to_datetime(df["BIA_ApostaUTC"], errors="coerce")
    X["hour_utc"] = dt.dt.hour
    X["weekday_utc"] = dt.dt.weekday
    return X


def _make_preprocess(X: pd.DataFrame):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    return ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cat_cols),
        ]
    )


def _fit_predict_models(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retorna:
    - score_cls_logit: P(roi>0)
    - score_cls_hgb:   P(roi>0) (desligado nesta iteração por performance; retorna NaN)
    """
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import Pipeline

    roi_tr = pd.to_numeric(train_df["roi_calc"], errors="coerce").to_numpy(float)
    ok = np.isfinite(roi_tr)
    if ok.sum() < 200:
        n = len(test_df)
        return (np.full(n, np.nan), np.full(n, np.nan))

    ytr = (roi_tr[ok] > 0).astype(int)
    if len(np.unique(ytr)) < 2:
        n = len(test_df)
        return (np.full(n, np.nan), np.full(n, np.nan))

    Xtr = _build_features(train_df.loc[ok].copy())
    Xte = _build_features(test_df.copy())
    pre = _make_preprocess(Xtr)

    # SGD(log_loss) é bem mais rápido/estável que LBFGS em alta dimensão.
    logit = Pipeline(
        [
            ("pre", pre),
            ("clf", SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=2000, tol=1e-3, random_state=7)),
        ]
    )

    logit.fit(Xtr, ytr)

    p_logit = logit.predict_proba(Xte)[:, 1].astype(float)
    p_hgb = np.full(len(test_df), np.nan, dtype=float)
    return p_logit, p_hgb


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])
    df["week"] = wf.week_key(df["BIA_ApostaUTC"])

    weeks = sorted(df["week"].astype(str).unique().tolist())
    out_logit = np.full(len(df), np.nan, dtype=float)
    out_hgb = np.full(len(df), np.nan, dtype=float)

    start_i = wf.MIN_GLOBAL_TRAIN_WEEKS
    end_i = min(len(weeks), start_i + MAX_WEEKS)
    for i in range(start_i, end_i):
        w_test = weeks[i]
        w_train = weeks[max(0, i - TRAIN_WINDOW_WEEKS) : i]
        train_df = df[df["week"].astype(str).isin(w_train)].copy()
        te_idx = df.index[df["week"].astype(str) == w_test]
        test_df = df.loc[te_idx].copy()
        if test_df.empty or train_df.empty:
            continue
        p_logit, p_hgb = _fit_predict_models(train_df, test_df)
        out_logit[df.index.get_indexer(te_idx)] = p_logit
        out_hgb[df.index.get_indexer(te_idx)] = p_hgb

    out = pd.DataFrame(
        {
            "ID Aposta": df["ID Aposta"] if "ID Aposta" in df.columns else np.arange(len(df)),
            "BIA_ApostaUTC": df["BIA_ApostaUTC"],
            "week": df["week"].astype(str),
            "score_v1_cls_logit_wf12": out_logit,
            "score_v1_cls_hgb_wf12": out_hgb,
        }
    )
    out.to_csv(OUT, index=False)
    print(str(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

