#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera scores "produção-like" (LogisticRegression + preprocess parecido com os joblib atuais),
usando exatamente o conjunto de features do payload do CLI, com opção de adicionar RegiaoEvento.

HONESTO no tempo (weekly wf12):
- Para cada semana w, treina com as 12 semanas anteriores e prevê somente w.
- Gera scores para todas as semanas a partir de um warmup (i>=4) para abastecer o treino do WF externo.

Saída:
  /workspace/analysis_proba_raw/pro_portfolio_all/scored_with_score_prod_payload_region_wf12.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_oos_walkforward_strategy as wf


SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")
OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")
OUT = OUT_DIR / "scored_with_score_prod_payload_region_wf12.csv"

TRAIN_WINDOW_WEEKS = 12
MIN_TRAIN_ROWS = 600  # ligeiramente maior para estabilidade


def _turno_utc_from_hour(h: float) -> str:
    if not np.isfinite(h):
        return "missing"
    hh = int(h)
    if 0 <= hh <= 5:
        return "madrugada"
    if 6 <= hh <= 11:
        return "manha"
    if 12 <= hh <= 17:
        return "tarde"
    if 18 <= hh <= 23:
        return "noite"
    return "missing"


def infer_region(text: str) -> str:
    if not text:
        return "desconhecida"
    s = str(text).lower()

    if any(k in s for k in ["australia", "austrália", "new zealand", "nova zelandia", "nova zelândia"]):
        return "oceania"
    if any(k in s for k in ["saudi", "arábia", "arabia", "qatar", "catar", "uae", "emirates", "emirados", "iran", "iraq", "israel", "turkey", "turquia"]):
        return "oriente_medio"
    if any(k in s for k in ["japan", "japão", "korea", "coreia", "china", "índia", "india", "thailand", "tailand", "viet", "malaysia", "singapore", "indonesia"]):
        return "asia"

    west = [
        "england","inglaterra","spain","espanha","france","frança","italy","itália","germany","alemanha",
        "netherlands","holanda","belgium","bélgica","portugal","switzerland","suíça","austria","áustria",
        "scotland","escócia","ireland","irlanda","norway","noruega","sweden","suécia","denmark","dinamarca",
    ]
    east = [
        "poland","polônia","czech","tcheca","slovakia","eslováquia","hungary","hungria","romania","romênia",
        "bulgaria","bulgária","serbia","sérvia","croatia","croácia","ukraine","ucrânia","russia","rússia",
        "greece","grécia",
    ]
    if any(k in s for k in west):
        return "europa_ocidental"
    if any(k in s for k in east):
        return "europa_oriental"

    if any(k in s for k in ["mexico", "méxico", "costa rica", "honduras", "guatemala", "panama", "panamá", "el salvador", "nicaragua", "nicarágua", "jamaica", "haiti", "haití", "dominican", "república dominicana"]):
        return "america_central"
    if any(k in s for k in ["usa", "united states", "estados unidos", "canada", "canadá"]):
        return "america_norte"
    if any(k in s for k in ["brazil","brasil","argentina","chile","colombia","colômbia","peru","uruguay","uruguai","paraguay","paraguai","ecuador","venezuela","bolivia","bolívia"]):
        return "america_sul"

    return "desconhecida"


def build_payload_features(df: pd.DataFrame, with_region: bool, with_tempo_bot: bool) -> pd.DataFrame:
    """
    Replica o schema do payload do CLI:
    NUM:
      - Número de casas disponíveis no momento da aposta
      - Dif % maior odd e segunda maior
      - Dif % maior odd e odd mediana
      - Dif Odds RB & BIA
      - MinutesToMatchStart
      - TempoApostas.Tempo total bot
    CAT:
      - Subtipo da Aposta
      - Dia Semana Aposta (UTC)
      - Turno Aposta (UTC)
      - Casa aposta vencedora
    + RegiaoEvento (opcional)
    """
    X = pd.DataFrame(index=df.index)
    X["Número de casas disponíveis no momento da aposta"] = pd.to_numeric(
        df.get("ApostaLive.Número de casas disponíveis no momento da aposta"), errors="coerce"
    )
    X["Dif % maior odd e segunda maior"] = pd.to_numeric(df.get("ApostaLive.Dif % maior odd e segunda maior"), errors="coerce")
    X["Dif % maior odd e odd mediana"] = pd.to_numeric(df.get("ApostaLive.Dif % maior odd e odd mediana"), errors="coerce")
    X["Dif Odds RB & BIA"] = pd.to_numeric(df.get("Dif Odds RB & BIA"), errors="coerce")
    X["MinutesToMatchStart"] = pd.to_numeric(df.get("RebelBetting.MinutesToMatchStart"), errors="coerce")
    if with_tempo_bot:
        X["TempoApostas.Tempo total bot"] = pd.to_numeric(df.get("TempoApostas.Tempo total bot"), errors="coerce")

    X["Subtipo da Aposta"] = df.get("Subtipo da Aposta", pd.Series("missing", index=df.index)).astype("string").fillna("missing")
    X["Dia Semana Aposta (UTC)"] = df.get("dow_pt", pd.Series("missing", index=df.index)).astype("string").fillna("missing")
    dt = pd.to_datetime(df.get("BIA_ApostaUTC"), errors="coerce")
    X["Turno Aposta (UTC)"] = dt.dt.hour.astype(float).apply(_turno_utc_from_hour).astype("string")
    X["Casa aposta vencedora"] = df.get("ApostaLive.Casa aposta vencedora", pd.Series("missing", index=df.index)).astype("string").fillna("missing")

    if with_region:
        comp = df.get("BetinAsia.event info competition name", pd.Series("", index=df.index)).astype(str)
        ev = df.get("Evento", pd.Series("", index=df.index)).astype(str)
        txt = (comp.fillna("") + " | " + ev.fillna("")).astype(str)
        X["RegiaoEvento"] = txt.apply(infer_region).astype("string")

    return X


def fit_predict_prod_logit(Xtr: pd.DataFrame, ytr: np.ndarray, Xte: pd.DataFrame) -> np.ndarray:
    """
    Pipeline similar ao observado nos .joblib:
      ColumnTransformer(num: SimpleImputer+StandardScaler, cat: SimpleImputer+OneHotEncoder)
      LogisticRegression(lbfgs, l2, max_iter=1000)
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num_cols = [c for c in Xtr.columns if pd.api.types.is_numeric_dtype(Xtr[c])]
    cat_cols = [c for c in Xtr.columns if c not in num_cols]
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", penalty="l2", C=1.0)
    pipe = Pipeline([("prep", pre), ("clf", clf)])
    pipe.fit(Xtr, ytr)
    return pipe.predict_proba(Xte)[:, 1].astype(float)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])
    df["week"] = wf.week_key(df["BIA_ApostaUTC"]).astype(str)

    weeks = sorted(df["week"].unique().tolist())
    s_payload = np.full(len(df), np.nan, dtype=float)
    s_payload_region = np.full(len(df), np.nan, dtype=float)
    s_payload_notempo = np.full(len(df), np.nan, dtype=float)
    s_payload_region_notempo = np.full(len(df), np.nan, dtype=float)

    # começamos em i>=4 para ter alguma história; o treino é a janela de até 12 semanas anteriores.
    for i in range(4, len(weeks)):
        w_test = weeks[i]
        w_train = weeks[max(0, i - TRAIN_WINDOW_WEEKS) : i]
        tr = df[df["week"].isin(w_train)].copy()
        te_idx = df.index[df["week"] == w_test]
        te = df.loc[te_idx].copy()
        if tr.empty or te.empty:
            continue

        roi_tr = pd.to_numeric(tr.get("roi_calc"), errors="coerce").to_numpy(float)
        ok = np.isfinite(roi_tr)
        if ok.sum() < MIN_TRAIN_ROWS:
            continue
        y = (roi_tr[ok] > 0).astype(int)
        if len(np.unique(y)) < 2:
            continue

        Xtr = build_payload_features(tr.loc[ok].copy(), with_region=False, with_tempo_bot=True)
        Xte = build_payload_features(te.copy(), with_region=False, with_tempo_bot=True)
        p = fit_predict_prod_logit(Xtr, y, Xte)
        s_payload[df.index.get_indexer(te_idx)] = p

        Xtr2 = build_payload_features(tr.loc[ok].copy(), with_region=True, with_tempo_bot=True)
        Xte2 = build_payload_features(te.copy(), with_region=True, with_tempo_bot=True)
        p2 = fit_predict_prod_logit(Xtr2, y, Xte2)
        s_payload_region[df.index.get_indexer(te_idx)] = p2

        # Variante sem `TempoApostas.Tempo total bot`
        Xtr3 = build_payload_features(tr.loc[ok].copy(), with_region=False, with_tempo_bot=False)
        Xte3 = build_payload_features(te.copy(), with_region=False, with_tempo_bot=False)
        p3 = fit_predict_prod_logit(Xtr3, y, Xte3)
        s_payload_notempo[df.index.get_indexer(te_idx)] = p3

        Xtr4 = build_payload_features(tr.loc[ok].copy(), with_region=True, with_tempo_bot=False)
        Xte4 = build_payload_features(te.copy(), with_region=True, with_tempo_bot=False)
        p4 = fit_predict_prod_logit(Xtr4, y, Xte4)
        s_payload_region_notempo[df.index.get_indexer(te_idx)] = p4

    out = pd.DataFrame(
        {
            "ID Aposta": df.get("ID Aposta", pd.Series(np.arange(len(df)), index=df.index)),
            "BIA_ApostaUTC": df.get("BIA_ApostaUTC"),
            "week": df["week"],
            "score_prod_payload_logit_wf12": s_payload,
            "score_prod_payload_region_logit_wf12": s_payload_region,
            "score_prod_payload_logit_notempo_wf12": s_payload_notempo,
            "score_prod_payload_region_logit_notempo_wf12": s_payload_region_notempo,
        }
    )
    out.to_csv(OUT, index=False)
    print(str(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

