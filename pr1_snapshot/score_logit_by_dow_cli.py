#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CLI de scoring por dia-da-semana (SegQui vs SexDom) com compat patch sklearn.
Saída em CSV: header "proba,decision" e uma linha (1º registro do CSV/JSON).
"""

from __future__ import annotations
import argparse, sys, os, json, datetime, warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from joblib import load

# Silenciar warnings chatos do sklearn ao unpickle
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Trying to unpickle estimator.*")


# ============================================================================
# PATCH DE COMPATIBILIDADE (carregar modelos 1.1.x em runtime 1.4+ / Py3.12)
# ============================================================================
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
    elif hasattr(est, "estimators"):  # Stacking/FeatureUnion/etc
        for _, sub in getattr(est, "estimators", []):
            yield from _walk_estimators(sub)


def patch_sklearn_compat(est):
    """
    Define atributos que mudaram entre versões (1.1.x -> 1.4.x),
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
        # SimpleImputer: novas versões referenciam keep_empty_features
        try:
            if isinstance(obj, SimpleImputer) and not hasattr(obj, "keep_empty_features"):
                setattr(obj, "keep_empty_features", False)
        except Exception:
            pass
        # OneHotEncoder: 1.2+ usa sparse_output; alguns modelos antigos tinham sparse
        try:
            if isinstance(obj, OneHotEncoder):
                if not hasattr(obj, "sparse_output"):
                    setattr(obj, "sparse_output", bool(getattr(obj, "sparse", True)))
                # algumas versões checam esse atributo interno
                if not hasattr(obj, "_drop_idx_after_grouping"):
                    setattr(obj, "_drop_idx_after_grouping", None)
        except Exception:
            pass
    return est


# ============================================================================
# Definições de schema e utilitários de parsing
# ============================================================================
NUM_COLS = [
    "Número de casas disponíveis no momento da aposta",
    "Dif % maior odd e segunda maior",
    "Dif % maior odd e odd mediana",
    "Dif Odds RB & BIA",
    "MinutesToMatchStart",
    "TempoApostas.Tempo total bot",
]
CAT_COLS = [
    "Subtipo da Aposta",
    "Dia Semana Aposta (UTC)",
    "Turno Aposta (UTC)",
    "Casa aposta vencedora",
]
ALL_COLS = NUM_COLS + CAT_COLS

ALIASES = {
    # % ↔︎ percent
    "Dif percent maior odd e segunda maior": "Dif % maior odd e segunda maior",
    "Dif percent maior odd e odd mediana": "Dif % maior odd e odd mediana",
    # & ↔︎ E (varios acidentais que já vimos)
    "Dif Odds RB E BIA": "Dif Odds RB & BIA",
    "Dif Oddds RB E BIA": "Dif Odds RB & BIA",
    # caracteres com acentuação podem vir alterados de algumas origens (normalmente ok)
}

DOW_PT_TO_IDX = {
    "seg": 0, "segunda": 0,
    "ter": 1, "terça": 1, "terca": 1,
    "qua": 2, "quarta": 2,
    "qui": 3, "quinta": 3,
    "sex": 4, "sexta": 4,
    "sáb": 5, "sab": 5, "sábado": 5, "sabado": 5,
    "dom": 6, "domingo": 6,
}


def echo_schema_and_exit():
    payload = {
        "expected_features": (
            [{"name": c, "type": "float"} for c in NUM_COLS]
            + [{"name": c, "type": "string"} for c in CAT_COLS]
        ),
        "note": "Numéricas aceitam ponto OU vírgula; aliases p/ % e & são tolerados."
    }
    print(json.dumps(payload, ensure_ascii=False))
    sys.exit(0)


def detect_subset_from_row(row: pd.Series) -> Tuple[str, int]:
    """
    Define subset "SegQui" ou "SexDom" e retorna (subset, weekday_idx 0..6).
    Usa 'Dia Semana Aposta (UTC)' se presente, senão weekday(UTC) atual.
    """
    wd_idx: Optional[int] = None
    val = row.get("Dia Semana Aposta (UTC)")
    if pd.notna(val):
        s = str(val).strip().lower()
        # usa só as 3 primeiras (ex.: "Sex", "Seg", "Qui"), mapeando
        key = s[:3]
        wd_idx = DOW_PT_TO_IDX.get(key)
    if wd_idx is None:
        wd_idx = datetime.datetime.now(datetime.timezone.utc).weekday()
    subset = "SegQui" if wd_idx in (0, 1, 2, 3) else "SexDom"
    return subset, wd_idx


def coerce_decimal_string(x: Any) -> Any:
    """
    Converte strings com vírgula decimal e (quando detectável) '.' como milhar.
    Regras:
      - se tem exatamente 1 vírgula e >1 ponto => pontos são milhares; vírgula é decimal
      - se tem 1 vírgula e 0 pontos       => vírgula é decimal
      - senão, tenta float direto
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan
    # caso típico BR com muitos pontos e 1 vírgula: 4.975.124 ... ,000000
    if s.count(",") == 1 and s.count(".") > 1:
        s_clean = s.replace(".", "").replace(",", ".")
        try:
            return float(s_clean)
        except Exception:
            return np.nan
    # vírgula decimal simples
    if s.count(",") == 1 and s.count(".") == 0:
        try:
            return float(s.replace(",", "."))
        except Exception:
            return np.nan
    # ponto decimal comum
    try:
        return float(s)
    except Exception:
        return np.nan


def coerce_df_types(df: pd.DataFrame) -> pd.DataFrame:
    # renomeia por aliases
    ren = {k: v for k, v in ALIASES.items() if k in df.columns}
    if ren:
        df = df.rename(columns=ren)
    # garante todas as colunas
    for c in ALL_COLS:
        if c not in df.columns:
            df[c] = np.nan
    # numéricas
    for c in NUM_COLS:
        df[c] = df[c].apply(coerce_decimal_string)
    # categóricas: string, substitui NaN por "missing"
    for c in CAT_COLS:
        df[c] = df[c].astype("string").fillna("missing")
    # mantém apenas o que precisamos, na ordem
    return df[ALL_COLS]


def read_one_record_from_csv(path: str) -> pd.DataFrame:
    # detecta delimitador olhando a 1ª linha
    with open(path, "r", encoding="utf-8") as f:
        head = f.readline()
    sep = ";" if head.count(";") >= head.count(",") else ","
    df = pd.read_csv(path, sep=sep, dtype=str, encoding="utf-8")
    if df.empty:
        raise ValueError("CSV sem linhas.")
    return df.iloc[[0]]  # 1ª linha


def read_one_record_from_json_str(s: str) -> pd.DataFrame:
    data = json.loads(s)
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        if not data:
            raise ValueError("JSON vazio.")
        df = pd.DataFrame([data[0]])
    else:
        raise ValueError("JSON deve ser objeto ou lista de objetos.")
    return df.iloc[[0]]


def load_json_record(args) -> Optional[pd.DataFrame]:
    if args.json:
        return read_one_record_from_json_str(args.json)
    if args.jsonfile:
        txt = Path(args.jsonfile).read_text(encoding="utf-8")
        return read_one_record_from_json_str(txt)
    return None


def apply_isotonic(p: float, calib: Dict[str, Any], floor: Optional[float], ceil: Optional[float]) -> float:
    """
    Aplica calibrador isotônico salvo em JSON (campos: isotonic.x, isotonic.y).
    Extrapolação: constante em bordas. Aplica floor/ceil, se fornecidos.
    """
    iso = calib.get("isotonic", {})
    x = np.asarray(iso.get("x", []), dtype=float)
    y = np.asarray(iso.get("y", []), dtype=float)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        # calibrador inválido → identidade
        out = p
    else:
        out = float(np.interp(p, x, y, left=float(y[0]), right=float(y[-1])))
    if floor is not None:
        out = max(out, float(floor))
    if ceil is not None:
        out = min(out, float(ceil))
    # clipe final por segurança
    return float(np.clip(out, 0.0, 1.0))


def log_jsonl(path: Optional[str], payload: Dict[str, Any]) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# ============================================================================
# MAIN
# ============================================================================
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", required=True, help="Diretório com os modelos .joblib")
    g = parser.add_mutually_exclusive_group(required=False)
    g.add_argument("--csvin", help="Arquivo CSV (1ª linha lida)")
    g.add_argument("--json", help="Registro JSON (string)")
    g.add_argument("--jsonfile", help="Arquivo com JSON")
    parser.add_argument("--cutoff", type=float, default=0.62, help="Cutoff p/ decisão (default 0.62)")
    parser.add_argument("--logfile", default=None, help="Arquivo JSONL p/ logging")
    parser.add_argument("--echo-schema", action="store_true", help="Imprime schema esperado e sai")
    parser.add_argument("--skip-calib", action="store_true", help="Não aplicar calibração mesmo se existir")
    parser.add_argument("--calib-floor", type=float, default=None, help="Piso para probas calibradas (ex.: 0.005)")
    parser.add_argument("--calib-ceil", type=float, default=None, help="Teto para probas calibradas")
    parser.add_argument("--debug", action="store_true", help="Erros propagam no stdout/stderr")
    args = parser.parse_args()

    if args.echo_schema or args.echo_schema:  # tolera --echo-schema com underscore
        echo_schema_and_exit()

    version = "2025-10-18c"

    try:
        # 1) Ler 1 registro
        if args.csvin:
            df_raw = read_one_record_from_csv(args.csvin)
        else:
            df_raw = load_json_record(args)
            if df_raw is None:
                raise ValueError("Forneça --csvin OU --json/--jsonfile.")

        # 2) Coerção de tipos/aliases
        df = coerce_df_types(df_raw.copy())

        # 3) Subset (SegQui x SexDom)
        subset, wd_idx = detect_subset_from_row(df.iloc[0])

        # 4) Carregar modelo
        models_dir = Path(args.models_dir)
        # nomes esperados:
        model_path = models_dir / f"model_logit_{subset}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(str(model_path))
        pipe = load(model_path)
        pipe = patch_sklearn_compat(pipe)

        # 5) Scoring bruto
        X = df  # possui exatamente ALL_COLS, coerçadas
        proba_raw = float(pipe.predict_proba(X)[:, 1][0])

        # 6) Calibração (se existir/permitida)
        proba_iso = proba_raw
        proba_cal = proba_raw
        calib_file = None
        if not args.skip_calib:
            calib_path = models_dir / f"clv_calib_{subset}.json"
            if calib_path.exists():
                calib_file = str(calib_path)
                calib = json.loads(calib_path.read_text(encoding="utf-8"))
                proba_iso = apply_isotonic(proba_raw, calib, floor=None, ceil=None)
                proba_cal = apply_isotonic(proba_raw, calib, floor=args.calib_floor, ceil=args.calib_ceil)
            else:
                calib_file = None  # sem calibrador → proba_cal = proba_raw
                proba_iso = proba_raw
                proba_cal = proba_raw

        decision = bool(proba_cal >= float(args.cutoff))

        # 7) Output CSV único (stdout)
        print("proba,decision")
        print(f"{proba_cal:.6f},{str(decision)}")

        # 8) Log JSONL
        payload = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "ok",
            "version": version,
            "subset": subset,
            "weekday": wd_idx,
            "model_path": str(model_path),
            "received_cols": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "sample_row": {c: (None if pd.isna(df.iloc[0][c]) else df.iloc[0][c]) for c in df.columns},
            "proba_raw": proba_raw,
            "proba_iso": proba_iso,
            "proba_cal": proba_cal,
            "cutoff": float(args.cutoff),
            "decision": decision,
            "calibration_file": calib_file,
            "calib_floor": args.calib_floor,
            "calib_ceil": args.calib_ceil,
        }
        log_jsonl(args.logfile, payload)
        return 0

    except Exception as e:
        # saída CSV "vazia" porém bem-formada (0 e False) para o PAD não quebrar
        print("proba,decision")
        print("0.000000,False")

        tb = None
        if args.debug:
            import traceback
            tb = traceback.format_exc()
        else:
            tb = getattr(e, "traceback", None)

        # tentar salvar contexto mínimo
        ctx_cols = None
        ctx_dtypes = None
        ctx_row = None
        try:
            if "df" in locals():
                ctx_cols = list(df.columns)
                ctx_dtypes = {c: str(df[c].dtype) for c in df.columns}
                ctx_row = {c: (None if pd.isna(df.iloc[0][c]) else df.iloc[0][c]) for c in df.columns}
        except Exception:
            pass

        payload = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "error",
            "version": "2025-10-18c",
            "subset": locals().get("subset", None),
            "weekday": locals().get("wd_idx", None),
            "model_path": str(locals().get("model_path", "")),
            "error": str(e),
            "traceback": tb,
            "received_cols": ctx_cols,
            "dtypes": ctx_dtypes,
            "sample_row": ctx_row,
        }
        log_jsonl(getattr(args, "logfile", None), payload)
        if args.debug:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
