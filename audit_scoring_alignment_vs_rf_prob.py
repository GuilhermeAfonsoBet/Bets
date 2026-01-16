#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audita alinhamento de scoragem (SegQui/SexDom) vs coluna ApostaLive.rf_prob.

Objetivo:
- Recalcular retroativamente as probabilidades usando os mesmos artefatos do CLI `score_logit_by_dow_cli.py`:
  - Modelos: model_logit_SegQui.joblib (Seg..Qui) e model_logit_prod_SexDom.joblib (Sex..Dom)
  - Calibração isotônica: clv_calib_SegQui.json e clv_calib_SexDom.json
  - Piso: 0.005 (mesmo do RPA)
- Construir uma coluna "proba_cli_like" que replica o score que o CLI retornaria (proba_cal por subset).
- Comparar "proba_cli_like" com "ApostaLive.rf_prob" (rf_proba) global e por dia-da-semana.

Saídas:
- analysis_proba_raw/pro_portfolio_all/scoring_alignment_vs_rf_prob.csv
- analysis_proba_raw/pro_portfolio_all/scoring_alignment_vs_rf_prob.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import joblib


SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")
OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")

MODEL_SEGQUI = Path("/workspace/model_logit_SegQui.joblib")
MODEL_SEXDOM = Path("/workspace/model_logit_prod_SexDom.joblib")
CALIB_SEGQUI = Path("/workspace/clv_calib_SegQui.json")
CALIB_SEXDOM = Path("/workspace/clv_calib_SexDom.json")

CALIB_FLOOR = 0.005

WEEKDAY_PT = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo"]


def _walk_estimators(est):
    """Itera recursivamente estimadores em Pipeline/ColumnTransformer/afins."""
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
    """
    Define atributos que mudaram entre versões (1.1.x -> 1.4+/1.8),
    evitando AttributeError em SimpleImputer e OneHotEncoder.
    """
    try:
        from sklearn.impute import SimpleImputer
    except Exception:
        SimpleImputer = ()
    try:
        from sklearn.preprocessing import OneHotEncoder
    except Exception:
        OneHotEncoder = ()

    for obj in _walk_estimators(est):
        try:
            if isinstance(obj, SimpleImputer) and not hasattr(obj, "keep_empty_features"):
                setattr(obj, "keep_empty_features", False)
            # algumas versões referenciam _fill_dtype
            if isinstance(obj, SimpleImputer) and not hasattr(obj, "_fill_dtype") and hasattr(obj, "_fit_dtype"):
                setattr(obj, "_fill_dtype", getattr(obj, "_fit_dtype"))
        except Exception:
            pass
        try:
            if isinstance(obj, OneHotEncoder):
                if not hasattr(obj, "sparse_output"):
                    setattr(obj, "sparse_output", bool(getattr(obj, "sparse", True)))
                if not hasattr(obj, "_drop_idx_after_grouping"):
                    setattr(obj, "_drop_idx_after_grouping", None)
        except Exception:
            pass
    return est


def infer_turno_utc(dt: pd.Timestamp) -> str:
    h = int(dt.hour)
    if 6 <= h <= 11:
        return "manhã"
    if 12 <= h <= 17:
        return "tarde"
    return "noite"


def build_features_segqui_sexdom(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["Subtipo da Aposta"] = df["Subtipo da Aposta"]
    out["Número de casas disponíveis no momento da aposta"] = df["ApostaLive.Número de casas disponíveis no momento da aposta"]
    out["Dif % maior odd e segunda maior"] = df["ApostaLive.Dif % maior odd e segunda maior"]
    out["Dif % maior odd e odd mediana"] = df["ApostaLive.Dif % maior odd e odd mediana"]
    out["Dif Odds RB & BIA"] = df["Dif Odds RB & BIA"]
    out["MinutesToMatchStart"] = df["RebelBetting.MinutesToMatchStart"]
    out["TempoApostas.Tempo total bot"] = df["TempoApostas.Tempo total bot"]
    dt = pd.to_datetime(df["BIA_ApostaUTC"], errors="coerce", utc=True)
    out["Dia Semana Aposta (UTC)"] = dt.dt.weekday.map(lambda x: WEEKDAY_PT[int(x)] if pd.notna(x) else None)
    out["Turno Aposta (UTC)"] = dt.apply(lambda x: infer_turno_utc(x) if pd.notna(x) else None)
    out["Casa aposta vencedora"] = df["ApostaLive.Casa aposta vencedora"]
    # coerção de tipos como no CLI
    for c in ["Subtipo da Aposta", "Dia Semana Aposta (UTC)", "Turno Aposta (UTC)", "Casa aposta vencedora"]:
        out[c] = out[c].astype("string").fillna("missing")
    for c in [
        "Número de casas disponíveis no momento da aposta",
        "Dif % maior odd e segunda maior",
        "Dif % maior odd e odd mediana",
        "Dif Odds RB & BIA",
        "MinutesToMatchStart",
        "TempoApostas.Tempo total bot",
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def clip_floor(p: np.ndarray, floor: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.clip(p, float(floor), 1.0 - float(floor))


def load_isotonic(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    d = json.loads(path.read_text(encoding="utf-8"))
    iso = d.get("isotonic", {})
    x = np.asarray(iso.get("x", []), dtype=float)
    y = np.asarray(iso.get("y", []), dtype=float)
    return x, y


def apply_isotonic_vec(p: np.ndarray, x: np.ndarray, y: np.ndarray, floor: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if x.size and y.size and x.size == y.size:
        out = np.interp(p, x, y, left=float(y[0]), right=float(y[-1]))
    else:
        out = p.copy()
    out = np.maximum(out, float(floor))
    return np.clip(out, 0.0, 1.0)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(m)) < 3:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])

    if "ApostaLive.rf_prob" not in df.columns:
        raise SystemExit("Coluna ApostaLive.rf_prob não encontrada no dataset.")

    # features + models
    X = build_features_segqui_sexdom(df)
    if not MODEL_SEGQUI.exists() or not MODEL_SEXDOM.exists():
        raise SystemExit("Modelos SegQui/SexDom não encontrados no workspace.")

    m_segqui = patch_sklearn_compat(joblib.load(MODEL_SEGQUI))
    m_sexdom = patch_sklearn_compat(joblib.load(MODEL_SEXDOM))

    p_segqui_raw = clip_floor(m_segqui.predict_proba(X)[:, 1], CALIB_FLOOR)
    p_sexdom_raw = clip_floor(m_sexdom.predict_proba(X)[:, 1], CALIB_FLOOR)

    # isotonic calibration (como no CLI: floor aplicado após isotonic)
    xq, yq = load_isotonic(CALIB_SEGQUI)
    xs, ys = load_isotonic(CALIB_SEXDOM)
    p_segqui_cal = apply_isotonic_vec(p_segqui_raw, xq, yq, CALIB_FLOOR)
    p_sexdom_cal = apply_isotonic_vec(p_sexdom_raw, xs, ys, CALIB_FLOOR)

    dow = df["dow_pt"].astype(str)
    is_weekend = dow.isin(["sexta-feira", "sábado", "domingo"])
    proba_cli_like = np.where(is_weekend.to_numpy(), p_sexdom_cal, p_segqui_cal).astype(float)

    out = pd.DataFrame(
        {
            "BIA_ApostaUTC": pd.to_datetime(df["BIA_ApostaUTC"], errors="coerce"),
            "dow_pt": dow,
            "bet_type": df.get("bet_type", pd.Series([None] * len(df))).astype(str),
            "rf_prob": pd.to_numeric(df["ApostaLive.rf_prob"], errors="coerce"),
            "proba_segqui_raw": p_segqui_raw,
            "proba_sexdom_raw": p_sexdom_raw,
            "proba_segqui_cal": p_segqui_cal,
            "proba_sexdom_cal": p_sexdom_cal,
            "proba_cli_like": proba_cli_like,
        }
    )

    # sanidade: rf_prob deve ser probabilidade em [0,1]. Valores fora desse range são tratados como inválidos.
    out["rf_prob_raw"] = out["rf_prob"]
    out.loc[~np.isfinite(out["rf_prob"]), "rf_prob"] = np.nan
    out.loc[(out["rf_prob"] < 0.0) | (out["rf_prob"] > 1.0), "rf_prob"] = np.nan

    out["diff_rf_minus_cli_like"] = out["rf_prob"] - out["proba_cli_like"]
    out["abs_diff_rf_minus_cli_like"] = np.abs(out["diff_rf_minus_cli_like"])

    # também trazer scores já existentes no dataset (para diagnóstico de alinhamento)
    for c in [
        "proba_raw_segunda",
        "proba_raw_terca",
        "proba_raw_quarta",
        "proba_raw_segqui",
        "proba_raw_sexdom",
        "proba_raw_operacional",
    ]:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            out[c] = np.nan

    out_path = OUT_DIR / "scoring_alignment_vs_rf_prob.csv"
    out.to_csv(out_path, index=False)

    # summary
    lines = []
    lines.append("## Auditoria — alinhamento de scoring vs `ApostaLive.rf_prob`\n\n")
    lines.append("Este relatório recalcula `proba_cli_like` usando os mesmos artefatos do CLI `score_logit_by_dow_cli.py`:\n")
    lines.append("- Seg..Qui: modelo SegQui + calibração isotônica\n")
    lines.append("- Sex..Dom: modelo SexDom + calibração isotônica\n")
    lines.append(f"- Piso (calib_floor): {CALIB_FLOOR}\n\n")

    a = out["rf_prob"].to_numpy(dtype=float)
    b = out["proba_cli_like"].to_numpy(dtype=float)
    invalid = int(np.sum(~np.isfinite(out["rf_prob"].to_numpy(dtype=float))))
    lines.append(f"- Observações: **{len(out)}** (rf_prob inválido/fora [0,1]: **{invalid}**)\n")
    lines.append(f"- Correlação global rf_prob vs proba_cli_like: **{corr(a,b):.3f}**\n")
    lines.append(f"- MAE global |rf_prob - proba_cli_like|: **{float(np.nanmean(out['abs_diff_rf_minus_cli_like'])):.4f}**\n\n")

    lines.append("### Por dia-da-semana\n")
    for d0 in WEEKDAY_PT:
        g = out[out["dow_pt"] == d0]
        if g.empty:
            continue
        aa = g["rf_prob"].to_numpy(dtype=float)
        bb = g["proba_cli_like"].to_numpy(dtype=float)
        mae = float(np.nanmean(np.abs(aa - bb)))
        lines.append(f"- **{d0}**: n={len(g)}, corr={corr(aa,bb):.3f}, MAE={mae:.4f}\n")

    # melhor alinhamento por dia (comparando rf_prob com candidatos disponíveis)
    lines.append("\n### Diagnóstico: qual score mais se parece com rf_prob?\n")
    candidates = [
        "proba_cli_like",
        "proba_raw_operacional",
        "proba_raw_segunda",
        "proba_raw_terca",
        "proba_raw_quarta",
        "proba_raw_segqui",
        "proba_raw_sexdom",
        "proba_segqui_cal",
        "proba_sexdom_cal",
    ]
    for d0 in WEEKDAY_PT:
        g = out[out["dow_pt"] == d0].copy()
        if g.empty:
            continue
        best = None
        for c in candidates:
            if c not in g.columns:
                continue
            cc = corr(g["rf_prob"].to_numpy(dtype=float), g[c].to_numpy(dtype=float))
            if not np.isfinite(cc):
                continue
            if (best is None) or (cc > best[1]):
                best = (c, float(cc))
        if best is None:
            lines.append(f"- **{d0}**: sem candidato válido\n")
        else:
            lines.append(f"- **{d0}**: melhor corr com **{best[0]}** = **{best[1]:.3f}**\n")

    lines.append("\n### Arquivos\n")
    lines.append(f"- CSV: `analysis_proba_raw/pro_portfolio_all/{out_path.name}`\n")
    (OUT_DIR / "scoring_alignment_vs_rf_prob.md").write_text("".join(lines), encoding="utf-8")

    print(str(out_path))
    print(str(OUT_DIR / "scoring_alignment_vs_rf_prob.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

