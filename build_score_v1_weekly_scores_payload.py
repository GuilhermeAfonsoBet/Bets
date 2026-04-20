#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Score v1 (leve) com:
  A) somente variáveis do payload (NUM+CAT) e
  B) payload + região do evento (feature categórica, sem IDs únicos).

Treino/score HONESTO no tempo:
- para cada semana w, treina usando as 12 semanas anteriores e prevê somente w.
- faz isso para TODAS as semanas (inclusive pré-OOS), para que as semanas de treino
  do walk-forward tenham score disponível sem vazamento.

Saída:
  /workspace/analysis_proba_raw/pro_portfolio_all/scored_with_score_v1_payload_wf12.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import evaluate_oos_walkforward_strategy as wf


SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")
OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")
OUT = OUT_DIR / "scored_with_score_v1_payload_wf12.csv"

TRAIN_WINDOW_WEEKS = 12
MIN_TRAIN_ROWS = 400


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
    """
    Heurística simples e determinística: busca países/regiões no texto.
    Retorna uma das categorias pedidas pelo usuário.
    """
    if not text:
        return "desconhecida"
    s = str(text).lower()

    # Oceania
    if any(k in s for k in ["australia", "austrália", "new zealand", "nova zelandia", "nova zelândia"]):
        return "oceania"

    # Oriente Médio
    if any(k in s for k in ["saudi", "arábia", "arabia", "qatar", "catar", "uae", "emirates", "emirados", "iran", "iraq", "israel", "turkey", "turquia"]):
        return "oriente_medio"

    # Ásia (fora oriente médio)
    if any(k in s for k in ["japan", "japão", "korea", "coreia", "china", "índia", "india", "thailand", "tailand", "viet", "malaysia", "singapore", "indonesia"]):
        return "asia"

    # Europa Ocidental/Oriental (simplificado)
    west = ["england", "inglaterra", "spain", "espanha", "france", "frança", "italy", "itália", "germany", "alemanha",
            "netherlands", "holanda", "belgium", "bélgica", "portugal", "switzerland", "suíça", "austria", "áustria",
            "scotland", "escócia", "ireland", "irlanda", "norway", "noruega", "sweden", "suécia", "denmark", "dinamarca"]
    east = ["poland", "polônia", "czech", "tcheca", "slovakia", "eslováquia", "hungary", "hungria", "romania", "romênia",
            "bulgaria", "bulgária", "serbia", "sérvia", "croatia", "croácia", "ukraine", "ucrânia", "russia", "rússia",
            "greece", "grécia", "turkey", "turquia"]
    if any(k in s for k in west):
        return "europa_ocidental"
    if any(k in s for k in east):
        return "europa_oriental"

    # Américas
    if any(k in s for k in ["usa", "united states", "estados unidos", "canada", "canadá", "mexico", "méxico"]):
        # México é mais "central", mas deixamos para o bloco abaixo (prioridade)
        if any(k in s for k in ["mexico", "méxico"]):
            return "america_central"
        if any(k in s for k in ["canada", "canadá", "usa", "united states", "estados unidos"]):
            return "america_norte"

    if any(k in s for k in ["costa rica", "honduras", "guatemala", "panama", "panamá", "el salvador", "nicaragua", "nicarágua", "jamaica", "haiti", "haití", "dominican", "república dominicana"]):
        return "america_central"

    if any(k in s for k in ["brazil", "brasil", "argentina", "chile", "colombia", "colômbia", "peru", "uruguay", "uruguai", "paraguay", "paraguai", "ecuador", "venezuela", "bolivia", "bolívia"]):
        return "america_sul"

    return "desconhecida"


def build_payload_frame(df: pd.DataFrame, with_region: bool) -> pd.DataFrame:
    """
    Constrói exatamente as variáveis do payload (NUM+CAT) a partir do dataset scored.
    """
    X = pd.DataFrame(index=df.index)

    # NUM (payload)
    X["Número de casas disponíveis no momento da aposta"] = pd.to_numeric(
        df.get("ApostaLive.Número de casas disponíveis no momento da aposta"), errors="coerce"
    )
    X["Dif % maior odd e segunda maior"] = pd.to_numeric(df.get("ApostaLive.Dif % maior odd e segunda maior"), errors="coerce")
    X["Dif % maior odd e odd mediana"] = pd.to_numeric(df.get("ApostaLive.Dif % maior odd e odd mediana"), errors="coerce")
    X["Dif Odds RB & BIA"] = pd.to_numeric(df.get("Dif Odds RB & BIA"), errors="coerce")
    X["MinutesToMatchStart"] = pd.to_numeric(df.get("RebelBetting.MinutesToMatchStart"), errors="coerce")
    X["TempoApostas.Tempo total bot"] = pd.to_numeric(df.get("TempoApostas.Tempo total bot"), errors="coerce")

    # CAT (payload)
    X["Subtipo da Aposta"] = df.get("Subtipo da Aposta", pd.Series("missing", index=df.index)).astype("string").fillna("missing")
    X["Dia Semana Aposta (UTC)"] = df.get("dow_pt", pd.Series("missing", index=df.index)).astype("string").fillna("missing")

    dt = pd.to_datetime(df.get("BIA_ApostaUTC"), errors="coerce")
    hour = dt.dt.hour.astype(float)
    X["Turno Aposta (UTC)"] = hour.apply(_turno_utc_from_hour).astype("string")

    X["Casa aposta vencedora"] = df.get("ApostaLive.Casa aposta vencedora", pd.Series("missing", index=df.index)).astype("string").fillna("missing")

    if with_region:
        # texto ex-ante (sem IDs únicos): competição/evento
        comp = df.get("BetinAsia.event info competition name", pd.Series("", index=df.index)).astype(str)
        ev = df.get("Evento", pd.Series("", index=df.index)).astype(str)
        txt = (comp.fillna("") + " | " + ev.fillna("")).astype(str)
        X["RegiaoEvento"] = txt.apply(infer_region).astype("string")

    return X


def fit_predict_sgd(Xtr: pd.DataFrame, ytr: np.ndarray, Xte: pd.DataFrame) -> np.ndarray:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    num_cols = [c for c in Xtr.columns if pd.api.types.is_numeric_dtype(Xtr[c])]
    cat_cols = [c for c in Xtr.columns if c not in num_cols]
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True))]), cat_cols),
        ]
    )
    clf = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=2000, tol=1e-3, random_state=7)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(Xtr, ytr)
    return pipe.predict_proba(Xte)[:, 1].astype(float)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])
    df["week"] = wf.week_key(df["BIA_ApostaUTC"]).astype(str)

    weeks = sorted(df["week"].unique().tolist())
    score_payload = np.full(len(df), np.nan, dtype=float)
    score_payload_region = np.full(len(df), np.nan, dtype=float)

    # gera scores desde cedo (para alimentar treino do WF), usando janela variável até 12
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

        # payload-only
        Xtr = build_payload_frame(tr.loc[ok].copy(), with_region=False)
        Xte = build_payload_frame(te.copy(), with_region=False)
        p = fit_predict_sgd(Xtr, y, Xte)
        score_payload[df.index.get_indexer(te_idx)] = p

        # payload + region
        Xtr2 = build_payload_frame(tr.loc[ok].copy(), with_region=True)
        Xte2 = build_payload_frame(te.copy(), with_region=True)
        p2 = fit_predict_sgd(Xtr2, y, Xte2)
        score_payload_region[df.index.get_indexer(te_idx)] = p2

    out = pd.DataFrame(
        {
            "ID Aposta": df.get("ID Aposta", pd.Series(np.arange(len(df)), index=df.index)),
            "BIA_ApostaUTC": df.get("BIA_ApostaUTC"),
            "week": df["week"],
            "score_v1_payload_sgd_wf12": score_payload,
            "score_v1_payload_region_sgd_wf12": score_payload_region,
        }
    )
    out.to_csv(OUT, index=False)
    print(str(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

