#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera um dataset "minipipeline" (apenas payloads) para auditoria de alinhamento:
  payload -> (pré-processamento do estudo) -> score -> comparação com logs operacionais

Foco:
  - 2026-01-21 (quarta, weekdays): proba_raw_quarta (com clip CALIB_FLOOR)
  - 2026-01-22 (quinta, SegQui):  proba_raw_segqui (clip) e proba_cal_segqui (isotonic + floor)

Saída:
  /workspace/analysis_proba_raw/pro_portfolio_all/minipipeline_payload_scores_2026-01-21_22.csv

Observação:
  - Usamos o MESMO pré-processamento dos CLIs (normalização de Subtipo e tempo_bot=NaN),
    para garantir alinhamento operacional.
  - O dataset do estudo completo (scored_dedup_proba_raw_all.csv) aqui para em 2026-01-20,
    então este script cria uma versão mínima apenas para 21/01 e 22/01 a partir de payloads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib


ROOT = Path("/workspace")
OUT = ROOT / "analysis_proba_raw" / "pro_portfolio_all" / "minipipeline_payload_scores_2026-01-21_22.csv"

PAYLOADS_21_22_DIR = ROOT / "payloads_21_01_22_01"
PAYLOADS_22_DIR = ROOT / "payloads_22_01_26"

LOG_WEEKDAYS = ROOT / "scoring_weekdays.jsonl"
LOG_BY_DOW = ROOT / "scoring.jsonl"

MODEL_QUARTA = ROOT / "model_logit_quarta.joblib"
MODEL_SEGQUI = ROOT / "model_logit_SegQui.joblib"
CALIB_SEGQUI = ROOT / "clv_calib_SegQui.json"

CALIB_FLOOR = 0.005


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return pd.DataFrame()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return pd.DataFrame(rows)


def _norm_bet_id(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    s = s.lstrip("0")
    return s if s else None


def _clip(p: float, floor: float = CALIB_FLOOR) -> float:
    return float(np.clip(float(p), float(floor), 1.0 - float(floor)))


def _apply_isotonic(p: float, calib: Dict[str, Any], floor: Optional[float]) -> float:
    iso = calib.get("isotonic", {}) if isinstance(calib, dict) else {}
    x = np.asarray(iso.get("x", []), dtype=float)
    y = np.asarray(iso.get("y", []), dtype=float)
    if x.size and y.size and x.size == y.size:
        out = float(np.interp(float(p), x, y, left=float(y[0]), right=float(y[-1])))
    else:
        out = float(p)
    if floor is not None:
        out = max(out, float(floor))
    return float(np.clip(out, 0.0, 1.0))


def _load_cli_modules():
    """
    Importa os CLIs como módulos para reusar exatamente:
      - preparar_payload() e _payload_hash() do weekdays
      - coerce_df_types() e normalize_subtipo_aposta() do by_dow
    """
    import importlib.util

    wk_spec = importlib.util.spec_from_file_location("wkcli", str(ROOT / "score_logit_weekdays_cli.py"))
    if wk_spec is None or wk_spec.loader is None:
        raise RuntimeError("Falha ao importar score_logit_weekdays_cli.py")
    wkcli = importlib.util.module_from_spec(wk_spec)
    wk_spec.loader.exec_module(wkcli)  # type: ignore[attr-defined]

    bd_spec = importlib.util.spec_from_file_location("bdcli", str(ROOT / "score_logit_by_dow_cli.py"))
    if bd_spec is None or bd_spec.loader is None:
        raise RuntimeError("Falha ao importar score_logit_by_dow_cli.py")
    bdcli = importlib.util.module_from_spec(bd_spec)
    bd_spec.loader.exec_module(bdcli)  # type: ignore[attr-defined]

    return wkcli, bdcli


def _payload_paths_by_id(payload_root: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not payload_root.exists():
        return out
    for p in payload_root.rglob("payload*.csv"):
        stem = p.stem.replace("payload", "").strip().lstrip("0")
        if stem.isdigit():
            out[stem] = p
    return out


def _score_quarta_from_payload(wkcli, model_qua, payload_path: Path) -> Tuple[float, int, str]:
    """
    Retorna (proba_clip, decision_dummy, payload_hash) para o último registro do CSV.
    """
    df_raw = pd.read_csv(payload_path, sep=";")
    df_p = wkcli.preparar_payload(df_raw)
    row = df_p.iloc[-1]
    X_row = row[wkcli.NUM_FEATURES + wkcli.CAT_FEATURES].to_frame().T
    proba = float(model_qua.predict_proba(X_row)[0, 1])
    p = _clip(proba, CALIB_FLOOR)
    decision = int(p >= 0.0)  # cutoff no log de 21/01 está 0.0; aqui não é usado no estudo
    ph = str(wkcli._payload_hash(row))
    return p, decision, ph


def _score_segqui_from_payload(bdcli, model_segqui, calib: Dict[str, Any], payload_path: Path) -> Tuple[float, float, int]:
    """
    Retorna (proba_raw_clip, proba_cal, decision) para o 1º registro do CSV.
    """
    df_raw = bdcli.read_one_record_from_csv(str(payload_path))
    df = bdcli.coerce_df_types(df_raw.copy())
    X = df  # já está em ALL_COLS, coerçadas e com tempo_bot=NaN
    proba_raw = float(model_segqui.predict_proba(X)[:, 1][0])
    proba_raw_clip = _clip(proba_raw, CALIB_FLOOR)
    # calibra a partir do raw (sem clip) e aplica floor (como no RPA)
    proba_cal = _apply_isotonic(proba_raw, calib, floor=CALIB_FLOOR)
    decision = int(proba_cal >= 0.62)  # cutoff operacional padrão SegQui
    return proba_raw_clip, proba_cal, decision


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    wkcli, bdcli = _load_cli_modules()

    # logs
    log_wk = _read_jsonl(LOG_WEEKDAYS)
    log_bd = _read_jsonl(LOG_BY_DOW)

    log_wk["ts"] = pd.to_datetime(log_wk.get("ts"), errors="coerce", utc=True)
    log_wk["date"] = log_wk["ts"].dt.date.astype(str)
    log_wk = log_wk[log_wk["date"] == "2026-01-21"].copy()
    log_wk["bet_id_norm"] = log_wk.get("bet_id").map(_norm_bet_id)
    log_wk = log_wk.dropna(subset=["bet_id_norm"])

    log_bd["ts"] = pd.to_datetime(log_bd.get("ts"), errors="coerce", utc=True)
    log_bd["date"] = log_bd["ts"].dt.date.astype(str)
    log_bd = log_bd[(log_bd["date"] == "2026-01-22") & (log_bd.get("subset") == "SegQui") & (log_bd.get("status") == "ok")].copy()
    log_bd["bet_id_norm"] = log_bd.get("bet_id").map(_norm_bet_id)
    log_bd = log_bd.dropna(subset=["bet_id_norm"])

    # payloads
    p21 = _payload_paths_by_id(PAYLOADS_21_22_DIR)
    p22 = _payload_paths_by_id(PAYLOADS_22_DIR)

    # modelos
    if not MODEL_QUARTA.exists():
        raise FileNotFoundError(str(MODEL_QUARTA))
    if not MODEL_SEGQUI.exists():
        raise FileNotFoundError(str(MODEL_SEGQUI))
    model_qua = wkcli.patch_sklearn_compat(joblib.load(MODEL_QUARTA))
    model_segqui = bdcli.patch_sklearn_compat(joblib.load(MODEL_SEGQUI))

    calib = json.loads(CALIB_SEGQUI.read_text(encoding="utf-8")) if CALIB_SEGQUI.exists() else {}

    rows: List[Dict[str, Any]] = []

    # -------------------------
    # 2026-01-21 (quarta)
    # -------------------------
    for _, r in log_wk.iterrows():
        bid = str(r["bet_id_norm"])
        payload_path = p21.get(bid)
        if payload_path is None:
            continue
        p_qua, d_qua, ph = _score_quarta_from_payload(wkcli, model_qua, payload_path)
        p_log = float(r.get("proba")) if pd.notna(r.get("proba")) else np.nan
        rows.append(
            {
                "date": "2026-01-21",
                "dow_pt": "quarta-feira",
                "subset": "weekdays",
                "bet_id": int(bid),
                "payload_path": str(payload_path),
                "payload_hash_minipipeline": ph,
                "payload_hash_log": r.get("payload_hash"),
                "match_payload_hash": int(str(r.get("payload_hash")) == ph),
                "proba_raw_quarta": p_qua,
                "proba_raw_segqui": np.nan,
                "proba_cal_segqui": np.nan,
                "score_portfolio": p_qua,  # portfólio usa proba_raw_quarta na quarta
                "score_portfolio_col": "proba_raw_quarta",
                "log_score_ref": p_log,
                "log_score_ref_col": "scoring_weekdays.proba",
                "abs_diff_score": float(abs(p_qua - p_log)) if np.isfinite(p_log) else np.nan,
                "match6_score": int(np.isfinite(p_log) and (round(p_qua, 6) == round(p_log, 6))),
            }
        )

    # -------------------------
    # 2026-01-22 (quinta / SegQui)
    # -------------------------
    for _, r in log_bd.iterrows():
        bid = str(r["bet_id_norm"])
        payload_path = p22.get(bid)
        if payload_path is None:
            continue
        p_raw_clip, p_cal, d = _score_segqui_from_payload(bdcli, model_segqui, calib, payload_path)
        p_cal_log = float(r.get("proba_cal")) if pd.notna(r.get("proba_cal")) else np.nan
        p_raw_log = float(r.get("proba_raw")) if pd.notna(r.get("proba_raw")) else np.nan
        rows.append(
            {
                "date": "2026-01-22",
                "dow_pt": "quinta-feira",
                "subset": "SegQui",
                "bet_id": int(bid),
                "payload_path": str(payload_path),
                "payload_hash_minipipeline": np.nan,
                "payload_hash_log": np.nan,
                "match_payload_hash": np.nan,
                "proba_raw_quarta": np.nan,
                "proba_raw_segqui": p_raw_clip,
                "proba_cal_segqui": p_cal,
                "score_portfolio": p_cal,  # portfólio usa proba_cal_segqui na quinta
                "score_portfolio_col": "proba_cal_segqui",
                "log_score_ref": p_cal_log,
                "log_score_ref_col": "scoring.jsonl.proba_cal",
                "abs_diff_score": float(abs(p_cal - p_cal_log)) if np.isfinite(p_cal_log) else np.nan,
                "match6_score": int(np.isfinite(p_cal_log) and (round(p_cal, 6) == round(p_cal_log, 6))),
                # diagnósticos adicionais:
                "log_proba_raw": p_raw_log,
                "abs_diff_raw": float(abs(p_raw_clip - p_raw_log)) if np.isfinite(p_raw_log) else np.nan,
                "match6_raw": int(np.isfinite(p_raw_log) and (round(p_raw_clip, 6) == round(p_raw_log, 6))),
                "decision_minipipeline": d,
                "decision_log": int(r.get("decision")) if pd.notna(r.get("decision")) else np.nan,
                "match_decision": int(pd.notna(r.get("decision")) and int(d) == int(r.get("decision"))),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["date", "bet_id"]).reset_index(drop=True)
    out.to_csv(OUT, index=False)

    # resumo
    if out.empty:
        print("dataset vazio (nenhuma interseção payload<->log).")
    else:
        for d, g in out.groupby("date"):
            match6 = float(np.mean(g["match6_score"].astype(float)) * 100.0)
            print(d, "n", len(g), "match6_score%", round(match6, 1))
        if "match6_raw" in out.columns:
            g = out[out["date"] == "2026-01-22"].copy()
            if not g.empty:
                mraw = float(np.mean(g["match6_raw"].astype(float)) * 100.0)
                mdec = float(np.mean(g["match_decision"].astype(float)) * 100.0)
                print("2026-01-22 raw_match6%", round(mraw, 1), "decision_match%", round(mdec, 1))

    print(str(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

