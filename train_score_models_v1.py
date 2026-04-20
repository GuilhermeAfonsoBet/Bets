#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""\
Treino e avaliação (temporal) de modelos de score financeiros.

Target principal:
- y_cls = 1[roi_calc > 0] (probabilidade de ROI positivo)
- y_reg = roi_calc_cap2 (E[ROI_cap2])

Modelos:
- Classificação: LogisticRegression (baseline), HistGradientBoostingClassifier
- Regressão: Ridge, HistGradientBoostingRegressor

Estratégia temporal:
- Split por semana (W-SUN).
- Para cada "grupo de dia":
    - segunda, terça, quarta, quinta, sexdom
  treinar com histórico e testar nas últimas K semanas.

Saídas:
- /workspace/analysis_proba_raw/pro_portfolio_all/score_modeling_v1_metrics.csv
- /workspace/models_score_v1/*.joblib

Obs: usa apenas features ex-ante (sem Result/ROI/etc).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

SCORED = Path('/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv')
OUT_DIR = Path('/workspace/analysis_proba_raw/pro_portfolio_all')
MODELS_DIR = Path('/workspace/models_score_v1')

TEST_WEEKS = 4  # holdout final por grupo
RANDOM_STATE = 7

GROUPS = {
  'segunda': {'dows': {'segunda-feira'}},
  'terca': {'dows': {'terça-feira'}},
  'quarta': {'dows': {'quarta-feira'}},
  'quinta': {'dows': {'quinta-feira'}},
  'sexdom': {'dows': {'sexta-feira','sábado','domingo'}},
}

# Remover variáveis pós-evento / leakage explícito
LEAKY_COL_SUBSTR = [
  'result',
  'roi ',
  'roi_',
  'clv',
  'fechamento',
  'oddfechamento',
  'closing',
  'stake aposta realizada',
  'balance',
  'status',
  'settled',
  'bet id',
]
LEAKY_EXACT = {
  'RebelBetting.Result',
  'ROI Real',
  'roi_calc',
  'roi_calc_cap1',
  'roi_calc_cap2',
}


def week_key(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts).dt.to_period('W-SUN').astype(str)


def is_exante_feature(col: str) -> bool:
    c = str(col)
    if c in LEAKY_EXACT:
        return False
    lc = c.lower()
    for s in LEAKY_COL_SUBSTR:
        if s in lc:
            return False
    return True


def choose_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Escolhe um conjunto fixo de colunas ex-ante para um dataset (grupo) inteiro,
    para garantir que treino e teste tenham a mesma matriz de features.
    """
    cols = []
    for c in df.columns:
        if not is_exante_feature(c):
            continue
        lc = str(c).lower()
        if any(k in lc for k in ["evento", "time ", "eventokey", "matchstart", "bet id", "id rb", "id "]):
            continue
        cols.append(c)

    keep = []
    for c in cols:
        if c.startswith("ApostaLive."):
            keep.append(c)
        elif c.startswith("RebelBetting.") and ("result" not in c.lower()) and ("outcome" not in c.lower()):
            keep.append(c)
        elif c in {
            "Odd Indicada no RB",
            "Odd Aposta Realizada",
            "Dif Odds RB & BIA",
            "TempoApostas.Tempo total bot",
            "Subtipo da Aposta",
            "Tipo Aposta",
            "Casa aposta vencedora",
        }:
            keep.append(c)

    # remover colunas totalmente vazias no grupo
    all_na = [c for c in keep if (c in df.columns and df[c].isna().all())]
    keep = [c for c in keep if c not in set(all_na)]
    return keep


def build_feature_frame(df: pd.DataFrame, keep_cols: List[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in keep_cols:
        X[c] = df[c] if c in df.columns else np.nan
    if "BIA_ApostaUTC" in df.columns:
        dt = pd.to_datetime(df["BIA_ApostaUTC"], errors="coerce")
        X["hour_utc"] = dt.dt.hour
        X["weekday_utc"] = dt.dt.weekday
    return X


def split_cols(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num = []
    cat = []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            num.append(c)
        else:
            cat.append(c)
    return num, cat


def make_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    num, cat = split_cols(X)
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler(with_mean=False)),
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # HistGradientBoosting exige matriz densa; manter OHE denso (dataset pequeno)
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([
        ('num', num_pipe, num),
        ('cat', cat_pipe, cat),
    ])


@dataclass(frozen=True)
class ModelSpec:
    name: str
    task: str  # 'cls' | 'reg'
    est: object


def eval_cls(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    out = {}
    try:
        out['auc'] = float(roc_auc_score(y_true, p))
    except Exception:
        out['auc'] = float('nan')
    try:
        out['brier'] = float(brier_score_loss(y_true, p))
    except Exception:
        out['brier'] = float('nan')
    out['p_mean'] = float(np.nanmean(p))
    out['y_mean'] = float(np.nanmean(y_true))
    return out


def eval_reg(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    out = {}
    mse = mean_squared_error(y_true, y_pred)
    out['rmse'] = float(math.sqrt(mse))
    out['y_mean'] = float(np.nanmean(y_true))
    out['pred_mean'] = float(np.nanmean(y_pred))
    out['bias'] = float(out['pred_mean'] - out['y_mean'])
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SCORED, parse_dates=['BIA_ApostaUTC'])
    df['week'] = week_key(df['BIA_ApostaUTC'])

    # target
    roi = pd.to_numeric(df['roi_calc'], errors='coerce').to_numpy(float)
    df['y_cls'] = (roi > 0).astype(int)
    df['y_reg'] = np.minimum(roi, 2.0)

    # drop unknown group
    df = df[df['dow_pt'].notna()].copy()

    rows = []

    specs = [
        ModelSpec('logit_l2', 'cls', LogisticRegression(max_iter=500, n_jobs=None, C=1.0, solver='lbfgs')),
        ModelSpec('hgb_cls', 'cls', HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, random_state=RANDOM_STATE)),
        ModelSpec('ridge', 'reg', Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ModelSpec('hgb_reg', 'reg', HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=400, random_state=RANDOM_STATE)),
    ]

    for gname, g in GROUPS.items():
        dows = set(g['dows'])
        xg = df[df['dow_pt'].isin(dows)].copy()
        # require finite y
        xg = xg[np.isfinite(xg['y_reg'].to_numpy(float))].copy()
        weeks = sorted(xg['week'].unique().tolist())
        if len(weeks) < (TEST_WEEKS + 6):
            continue
        test_weeks = weeks[-TEST_WEEKS:]
        train_weeks = weeks[:-TEST_WEEKS]

        tr = xg[xg['week'].isin(train_weeks)].copy()
        te = xg[xg['week'].isin(test_weeks)].copy()

        keep_cols = choose_feature_columns(pd.concat([tr, te], axis=0, ignore_index=True))
        Xtr = build_feature_frame(tr, keep_cols)
        Xte = build_feature_frame(te, keep_cols)

        pre = make_preprocess(pd.concat([Xtr, Xte], axis=0, ignore_index=True))

        for sp in specs:
            if sp.task == 'cls':
                ytr = tr['y_cls'].to_numpy(int)
                yte = te['y_cls'].to_numpy(int)
                pipe = Pipeline([('pre', pre), ('model', sp.est)])
                pipe.fit(Xtr, ytr)
                p = pipe.predict_proba(Xte)[:,1]
                m = eval_cls(yte, p)
                out = {
                    'group': gname,
                    'task': 'cls',
                    'model': sp.name,
                    'train_weeks': len(train_weeks),
                    'test_weeks': len(test_weeks),
                    'n_train': int(len(tr)),
                    'n_test': int(len(te)),
                    **m,
                }
                rows.append(out)
                joblib.dump(pipe, MODELS_DIR / f'score_{gname}_{sp.name}.joblib')
            else:
                ytr = tr['y_reg'].to_numpy(float)
                yte = te['y_reg'].to_numpy(float)
                pipe = Pipeline([('pre', pre), ('model', sp.est)])
                pipe.fit(Xtr, ytr)
                yp = pipe.predict(Xte)
                m = eval_reg(yte, yp)
                out = {
                    'group': gname,
                    'task': 'reg',
                    'model': sp.name,
                    'train_weeks': len(train_weeks),
                    'test_weeks': len(test_weeks),
                    'n_train': int(len(tr)),
                    'n_test': int(len(te)),
                    **m,
                }
                rows.append(out)
                joblib.dump(pipe, MODELS_DIR / f'score_{gname}_{sp.name}.joblib')

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / 'score_modeling_v1_metrics.csv', index=False)
    print(str(OUT_DIR / 'score_modeling_v1_metrics.csv'))
    print(str(MODELS_DIR))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
