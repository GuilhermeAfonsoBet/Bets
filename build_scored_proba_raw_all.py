#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera base deduplicada (por ID Aposta) com scores proba_raw para:
- Seg/Ter/Qua: modelos separados (model_logit_segunda/terca/quarta)
- SegQui: modelo model_logit_SegQui (Seg..Qui)
- SexDom: modelo model_logit_prod_SexDom (Sex..Dom)

Saída:
  /workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv

Observações:
- Scores são "operacionais": predict_proba com clip em [calib_floor, 1-calib_floor].
- A base inclui FH e FT; o consumidor deve segmentar por Tipo Aposta.
"""

from __future__ import annotations

import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


EXCEL_CANDIDATES = [
    "/workspace/ResumoApostas_PBI_final_20.01.2026.xlsx",
    "/workspace/ResumoApostas_PBI_final_17.01.2026.xlsx",
    "/workspace/main_snapshot_latest/ResumoApostas_PBI_final_17.01.2026.xlsx",
    "/workspace/main_snapshot_latest/ResumoApostas_PBI_final_14.01.2026.xlsx",
]
SHEET = "ResumoApostas (2)"
OUT = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")

CALIB_FLOOR = 0.005

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


def walk(est):
    yield est
    if isinstance(est, Pipeline):
        for _, s in est.steps:
            yield from walk(s)
    elif isinstance(est, ColumnTransformer):
        for _, s, _ in est.transformers_:
            if s in ("drop", "passthrough"):
                continue
            yield from walk(s)
    elif hasattr(est, "estimators"):
        for _, s in getattr(est, "estimators", []):
            yield from walk(s)


def patch_sklearn_compat(est):
    for obj in walk(est):
        if isinstance(obj, Pipeline) and not hasattr(obj, "transform_input"):
            setattr(obj, "transform_input", None)
        if isinstance(obj, SimpleImputer):
            if not hasattr(obj, "keep_empty_features"):
                setattr(obj, "keep_empty_features", False)
            if not hasattr(obj, "_fill_dtype") and hasattr(obj, "_fit_dtype"):
                setattr(obj, "_fill_dtype", getattr(obj, "_fit_dtype"))
        if isinstance(obj, OneHotEncoder):
            if not hasattr(obj, "sparse_output"):
                setattr(obj, "sparse_output", bool(getattr(obj, "sparse", True)))
            if not hasattr(obj, "_drop_idx_after_grouping"):
                setattr(obj, "_drop_idx_after_grouping", None)
    return est


def clip(p: np.ndarray) -> np.ndarray:
    return np.clip(p.astype(float), CALIB_FLOOR, 1.0 - CALIB_FLOOR)


def house_cap_series(df: pd.DataFrame) -> pd.Series:
    x = pd.to_numeric(df.get("ApostaLive.Stake máximo da casa da aposta (USD)"), errors="coerce")
    x = x.where((x > 0) & np.isfinite(x), np.inf)
    return x


def dedup_last(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_i"] = np.arange(len(df), dtype=int)
    df = df.sort_values(["ID Aposta", "BIA_ApostaUTC", "_i"], ascending=[True, True, True])
    out = df.groupby("ID Aposta", as_index=False).tail(1).drop(columns=["_i"]).reset_index(drop=True)
    return out


def _roi_from_result_and_odds(result: object, odds: float) -> float:
    """
    ROI por stake (lucro/ stake), baseado em odds decimais e resultado.
    Ex.: Win @2.10 => +1.10; Lose => -1.0; HalfWin => +(odds-1)/2; HalfLose => -0.5; Push/Void => 0.
    """
    if not np.isfinite(odds) or odds <= 1e-12:
        return float("nan")
    s = str(result).strip().lower()
    if s == "win":
        return float(odds - 1.0)
    if s in {"lose", "loss"}:
        return -1.0
    if s in {"halfwin", "half win"}:
        return float((odds - 1.0) / 2.0)
    if s in {"halfloss", "halflose", "half loss", "half lose"}:
        return -0.5
    if s in {"push", "void", "refund", "cancelled", "canceled"}:
        return 0.0
    return float("nan")


def compute_roi_calc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona:
    - roi_calc: ROI por aposta calculado via odds+resultado
    - roi_calc_cap1 / roi_calc_cap2: versões capadas (para robustez)
    """
    out = df.copy()

    def col(name: str) -> pd.Series:
        return out[name] if name in out.columns else pd.Series(np.nan, index=out.index)

    odds = pd.to_numeric(col("Odd Aposta Realizada"), errors="coerce")
    for nm in ["BetinAsia.got price", "BetinAsia.Odds", "RebelBetting.Odds", "Odd Indicada no RB"]:
        odds = odds.combine_first(pd.to_numeric(col(nm), errors="coerce"))
    odds = odds.to_numpy(dtype=float)

    res = col("RebelBetting.Result")
    roi_calc = np.array([_roi_from_result_and_odds(r, o) for r, o in zip(res, odds)], dtype=float)

    out["roi_calc"] = roi_calc
    out["roi_calc_cap2"] = np.minimum(roi_calc, 2.0)
    out["roi_calc_cap1"] = np.minimum(roi_calc, 1.0)
    return out


def build_features_weekday_models(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["Número de casas disponíveis no momento da aposta"] = df["ApostaLive.Número de casas disponíveis no momento da aposta"]
    out["Dif percent maior odd e segunda maior"] = df["ApostaLive.Dif % maior odd e segunda maior"]
    out["Dif percent maior odd e odd mediana"] = df["ApostaLive.Dif % maior odd e odd mediana"]
    # IMPORTANT: `Dif Odds RB & BIA` na planilha pode usar BetinAsia.got price (ex-post) como proxy de BIA.
    # Para emular o operacional (payload), recalculamos esse campo via Aux1 (maior odd) / 1000.
    out["Dif Odds RB E BIA"] = df["Dif Odds RB & BIA_op_sim"]
    out["MinutesToMatchStart"] = df["RebelBetting.MinutesToMatchStart"]
    out["TempoApostas.Tempo total bot"] = df["TempoApostas.Tempo total bot"]
    out["Subtipo da Aposta"] = df["Subtipo da Aposta"]
    out["Casa aposta vencedora"] = df["ApostaLive.Casa aposta vencedora"]

    dt = pd.to_datetime(df["BIA_ApostaUTC"], errors="coerce")
    out["Dia Semana Aposta (UTC)"] = dt.dt.weekday.map(lambda x: WEEKDAY_PT[int(x)] if pd.notna(x) else None)
    out["Turno Aposta (UTC)"] = dt.apply(lambda x: infer_turno_utc(x) if pd.notna(x) else None)
    return out


def build_features_segqui_sexdom(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features esperadas por SegQui e SexDom: usam % e & nos nomes.
    """
    out = pd.DataFrame(index=df.index)
    out["Subtipo da Aposta"] = df["Subtipo da Aposta"]
    out["Número de casas disponíveis no momento da aposta"] = df["ApostaLive.Número de casas disponíveis no momento da aposta"]
    out["Dif % maior odd e segunda maior"] = df["ApostaLive.Dif % maior odd e segunda maior"]
    out["Dif % maior odd e odd mediana"] = df["ApostaLive.Dif % maior odd e odd mediana"]
    # idem weekdays: usar versão op_sim (sem got price ex-post)
    out["Dif Odds RB & BIA"] = df["Dif Odds RB & BIA_op_sim"]
    out["MinutesToMatchStart"] = df["RebelBetting.MinutesToMatchStart"]
    out["TempoApostas.Tempo total bot"] = df["TempoApostas.Tempo total bot"]
    dt = pd.to_datetime(df["BIA_ApostaUTC"], errors="coerce")
    out["Dia Semana Aposta (UTC)"] = dt.dt.weekday.map(lambda x: WEEKDAY_PT[int(x)] if pd.notna(x) else None)
    out["Turno Aposta (UTC)"] = dt.apply(lambda x: infer_turno_utc(x) if pd.notna(x) else None)
    out["Casa aposta vencedora"] = df["ApostaLive.Casa aposta vencedora"]
    return out


def compute_dif_odds_rb_bia_op_sim(df: pd.DataFrame) -> pd.Series:
    """
    Recalcula `Dif Odds RB & BIA` para emular o operacional (payload).

    Definição (observada no operacional):
      Odd_RB  := RebelBetting.Odds (fallback: Odd Indicada no RB)
      Odd_BIA := ApostaLive.Aux1 - maior odd / 1000 (com tolerância a já estar em escala decimal)
      Dif     := (Odd_RB - Odd_BIA) / Odd_RB

    Observação: evita usar `BetinAsia.got price` (ex-post).
    """
    # odd RB
    odd_rb = pd.to_numeric(df.get("RebelBetting.Odds"), errors="coerce")
    odd_rb = odd_rb.combine_first(pd.to_numeric(df.get("Odd Indicada no RB"), errors="coerce"))
    odd_rb = odd_rb.astype(float)

    # odd BIA proxy: Aux1 / 1000
    aux1 = pd.to_numeric(df.get("ApostaLive.Aux1 - maior odd"), errors="coerce").astype(float)
    # heurística de escala: se Aux1 > 10, interpretamos como milésimos (ex.: 1665 -> 1.665)
    odd_bia = np.where(np.isfinite(aux1.to_numpy(float)) & (aux1.to_numpy(float) > 10.0), aux1.to_numpy(float) / 1000.0, aux1.to_numpy(float))
    odd_bia = pd.Series(odd_bia, index=df.index, dtype=float)

    dif = (odd_rb - odd_bia) / odd_rb
    dif = dif.where(np.isfinite(dif) & np.isfinite(odd_rb) & (odd_rb > 1e-12), np.nan)
    return dif.astype(float)


def main() -> int:
    Path("/workspace/analysis_proba_raw").mkdir(parents=True, exist_ok=True)

    excel = next((p for p in EXCEL_CANDIDATES if Path(p).exists()), None)
    if excel is None:
        raise FileNotFoundError("Nenhum arquivo Excel encontrado em: " + " | ".join(EXCEL_CANDIDATES))
    df = pd.read_excel(excel, sheet_name=SHEET, engine="openpyxl")
    # filtra datas válidas
    df = df[pd.to_datetime(df["BIA_ApostaUTC"], errors="coerce") >= pd.Timestamp("2024-01-01")].copy()
    df = dedup_last(df)

    df["house_cap"] = house_cap_series(df)
    dt = pd.to_datetime(df["BIA_ApostaUTC"], errors="coerce")
    df["dow_pt"] = dt.dt.weekday.map(lambda x: WEEKDAY_PT[int(x)] if pd.notna(x) else None)
    df["bet_is_fh"] = df["Tipo Aposta"].astype(str).str.lower().str.contains("first half")
    df["bet_type"] = np.where(df["bet_is_fh"], "FH", "FT")
    df = compute_roi_calc(df)
    # Recalcular Dif Odds para emular o operacional (sem got price ex-post).
    df["Dif Odds RB & BIA_op_sim"] = compute_dif_odds_rb_bia_op_sim(df)

    # load models
    m_seg = patch_sklearn_compat(joblib.load("/workspace/model_logit_segunda.joblib"))
    m_ter = patch_sklearn_compat(joblib.load("/workspace/model_logit_terca.joblib"))
    m_qua = patch_sklearn_compat(joblib.load("/workspace/model_logit_quarta.joblib"))
    m_segqui = patch_sklearn_compat(joblib.load("/workspace/model_logit_SegQui.joblib"))
    m_sexdom = patch_sklearn_compat(joblib.load("/workspace/model_logit_prod_SexDom.joblib"))

    # features
    Xw = build_features_weekday_models(df)
    Xsd = build_features_segqui_sexdom(df)

    # weekday models require forcing DOW to their training day
    X_seg = Xw.copy()
    X_seg["Dia Semana Aposta (UTC)"] = "segunda-feira"
    X_ter = Xw.copy()
    X_ter["Dia Semana Aposta (UTC)"] = "terça-feira"
    X_qua = Xw.copy()
    X_qua["Dia Semana Aposta (UTC)"] = "quarta-feira"

    df["proba_raw_segunda"] = clip(m_seg.predict_proba(X_seg)[:, 1])
    df["proba_raw_terca"] = clip(m_ter.predict_proba(X_ter)[:, 1])
    df["proba_raw_quarta"] = clip(m_qua.predict_proba(X_qua)[:, 1])
    df["proba_raw_segqui"] = clip(m_segqui.predict_proba(Xsd)[:, 1])
    df["proba_raw_sexdom"] = clip(m_sexdom.predict_proba(Xsd)[:, 1])

    # score operacional \"por dia\" (para referência)
    df["proba_raw_operacional"] = np.nan
    df.loc[df["dow_pt"] == "segunda-feira", "proba_raw_operacional"] = df.loc[df["dow_pt"] == "segunda-feira", "proba_raw_segunda"]
    df.loc[df["dow_pt"] == "terça-feira", "proba_raw_operacional"] = df.loc[df["dow_pt"] == "terça-feira", "proba_raw_terca"]
    df.loc[df["dow_pt"] == "quarta-feira", "proba_raw_operacional"] = df.loc[df["dow_pt"] == "quarta-feira", "proba_raw_quarta"]
    df.loc[df["dow_pt"] == "quinta-feira", "proba_raw_operacional"] = df.loc[df["dow_pt"] == "quinta-feira", "proba_raw_segqui"]
    df.loc[df["dow_pt"].isin(["sexta-feira", "sábado", "domingo"]), "proba_raw_operacional"] = df.loc[
        df["dow_pt"].isin(["sexta-feira", "sábado", "domingo"]),
        "proba_raw_sexdom",
    ]

    df.to_csv(OUT, index=False)
    print(str(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

