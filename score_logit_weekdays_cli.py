# -*- coding: utf-8 -*-
# score_logit_weekdays_cli.py

import argparse
import sys
import json
import datetime
import hashlib
from pathlib import Path
import re
import unicodedata

import pandas as pd
import numpy as np
import joblib


# ============================================================================
# PATCH DE COMPATIBILIDADE (carregar modelos antigos em sklearn novo)
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
    Define atributos que mudaram entre versões (1.1.x -> 1.4+),
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
        # SimpleImputer: versões novas referenciam keep_empty_features
        try:
            if isinstance(obj, SimpleImputer):
                if not hasattr(obj, "keep_empty_features"):
                    setattr(obj, "keep_empty_features", False)
                # algumas versões novas esperam _fill_dtype (modelos antigos têm _fit_dtype)
                if not hasattr(obj, "_fill_dtype") and hasattr(obj, "_fit_dtype"):
                    try:
                        setattr(obj, "_fill_dtype", getattr(obj, "_fit_dtype"))
                    except Exception:
                        pass
        except Exception:
            pass
        # OneHotEncoder: 1.2+ usa sparse_output; modelos antigos tinham sparse
        try:
            if isinstance(obj, OneHotEncoder):
                if not hasattr(obj, "sparse_output"):
                    setattr(obj, "sparse_output", bool(getattr(obj, "sparse", True)))
                if not hasattr(obj, "_drop_idx_after_grouping"):
                    setattr(obj, "_drop_idx_after_grouping", None)
        except Exception:
            pass
    return est

# Colunas esperadas no PAYLOAD (as mesmas que usamos no treino)
NUM_FEATURES = [
    "Número de casas disponíveis no momento da aposta",
    "Dif percent maior odd e segunda maior",
    "Dif percent maior odd e odd mediana",
    "Dif Odds RB E BIA",
    "MinutesToMatchStart",
    "TempoApostas.Tempo total bot",
]

CAT_FEATURES = [
    "Subtipo da Aposta",
    "Dia Semana Aposta (UTC)",
    "Turno Aposta (UTC)",
    "Casa aposta vencedora",
]

DOW_COL = "Dia Semana Aposta (UTC)"

# Coluna de ID da aposta (opcional no payload). Se existir, será logada.
BET_ID_COL_CANDIDATES = [
    "IDAposta",
    "ID Aposta",
    "ID_Aposta",
    "betID",
    "BetID",
    "idAposta",
    "id_aposta",
]

# Nomes dos arquivos de modelo (dentro de --models-dir)
MODEL_FILENAMES = {
    # chaves são DOW normalizados (lowercase, sem acento)
    "segunda-feira": "model_logit_segunda.joblib",
    "terca-feira":   "model_logit_terca.joblib",
    "quarta-feira":  "model_logit_quarta.joblib",
}


def _strip_accents(s: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn"
    )


def normalize_dow(x) -> str:
    """
    Normaliza o dia da semana vindo do payload para casar com MODEL_FILENAMES.
    Aceita variações comuns: maiúsculas, sem acento (terca), abreviações (seg/ter/qua).
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = _strip_accents(s)  # terça -> terca
    # normaliza abreviações comuns
    s3 = s[:3]
    if s3 in {"seg"}:
        return "segunda-feira"
    if s3 in {"ter"}:
        return "terca-feira"
    if s3 in {"qua"}:
        return "quarta-feira"
    # formas completas
    if s.startswith("segunda"):
        return "segunda-feira"
    if s.startswith("terca") or s.startswith("terça"):
        return "terca-feira"
    if s.startswith("quarta"):
        return "quarta-feira"
    return s


def _payload_hash(row: pd.Series) -> str:
    """
    Hash estável do registro para auditoria (sem expor PII).
    Usa apenas as features do modelo, já após coerção em preparar_payload().
    """
    parts = []
    for c in (NUM_FEATURES + CAT_FEATURES):
        v = row.get(c, "")
        if pd.isna(v):
            v = ""
        parts.append(f"{c}={v}")
    s = "|".join(parts).encode("utf-8", errors="ignore")
    return hashlib.sha256(s).hexdigest()[:16]


def get_bet_id(row: pd.Series):
    for c in BET_ID_COL_CANDIDATES:
        if c in row.index:
            v = row.get(c)
            if pd.isna(v):
                return None
            s = str(v).strip()
            return s if s else None
    return None


def coerce_decimal_string(x):
    """
    Converte números que podem vir como string em formato BR/EN.
    Regras:
      - se tem exatamente 1 vírgula e >1 ponto => pontos são milhares; vírgula é decimal
      - se tem 1 vírgula e 0 pontos           => vírgula é decimal
      - senão, tenta float direto (ponto decimal comum)
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan
    # caso típico BR com milhares (.) e decimal (,)
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


def normalize_subtipo_aposta(x) -> str:
    """
    Normaliza `Subtipo da Aposta` para casar com as categorias usadas no treino.

    Observação importante: os modelos foram treinados com valores em PT-BR e, em muitos
    casos, com sinal explícito, por exemplo: '+0,75', '-0,5', '+1,25', além de inteiros
    sem sinal ('-1', '-2', '0', '1').

    O operacional/payload às vezes envia variações como '0.75'/'-0.75' (ponto decimal) ou
    sem sinal. Como o modelo usa OneHotEncoder, essas variações viram categorias
    desconhecidas e mudam o score. Esta função converte para o formato canônico.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "missing"
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return "missing"

    # Já está no formato canônico (ex.: '+0,75', '-0,5', '0', '1', '-2')
    # Mantém como está (apenas remove espaços redundantes).
    s0 = re.sub(r"\s+", "", s)
    if re.fullmatch(r"[+-]?\d+(,\d+)?", s0):
        # Se houver decimal e não houver sinal, pode ser ambíguo; lidamos abaixo via parse.
        if "," in s0 and not (s0.startswith("+") or s0.startswith("-")):
            pass
        else:
            return s0

    # Parse robusto para float aceitando vírgula/ponto
    v = coerce_decimal_string(s0.replace(",", ".")) if "," in s0 and "." in s0 else coerce_decimal_string(s0)
    if not np.isfinite(v):
        return s0  # fallback: mantém string original

    # Inteiros: no treino aparecem sem sinal para positivos (ex.: '1') e com '-' para negativos.
    if abs(v - int(round(v))) < 1e-12:
        return str(int(round(v)))

    # Decimais: no treino aparecem com sinal explícito (+/-) e vírgula decimal.
    sign = "+" if v > 0 else "-"
    mag = abs(float(v))
    # limita precisão (evita '0,7500000001')
    mag = round(mag, 2)
    mag_str = f"{mag:.2f}".rstrip("0").rstrip(".")
    mag_str = mag_str.replace(".", ",")
    return f"{sign}{mag_str}"


def preparar_payload(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]

    # converter colunas numéricas vindas como texto (aceita ponto OU vírgula decimal)
    for col in NUM_FEATURES:
        if col in df.columns:
            # aplica coerção robusta em qualquer dtype (string/object/num)
            df[col] = df[col].apply(coerce_decimal_string)

    # normaliza Subtipo para casar com o treino (OneHotEncoder é sensível a string exata)
    if "Subtipo da Aposta" in df.columns:
        df["Subtipo da Aposta"] = df["Subtipo da Aposta"].apply(normalize_subtipo_aposta)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--csvin", required=True)
    parser.add_argument("--cutoff", required=True, type=float)
    parser.add_argument("--calib-floor", required=False, type=float, default=0.0)
    parser.add_argument("--logfile", required=False)
    # Compatibilidade com PAD legado: por padrão, imprimir apenas 1 linha (último registro do payload).
    # Use --stdout-all-rows para imprimir todas as linhas (modo auditoria).
    parser.add_argument("--stdout-all-rows", action="store_true", help="Imprime proba/decision para todas as linhas do CSV (padrão: apenas última linha).")

    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    csvin      = Path(args.csvin)
    cutoff     = float(args.cutoff)
    calib_floor = float(args.calib_floor)

    # Carregar payload (separador ; conforme seu exemplo)
    df = pd.read_csv(csvin, sep=";")
    df = preparar_payload(df)

    # Prepara estruturas de saída
    probas    = []
    decisions = []

    # Cache de modelos por dia (para não carregar várias vezes)
    model_cache = {}

    logfile_handle = None
    if args.logfile:
        log_path = Path(args.logfile)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logfile_handle = open(log_path, "a", encoding="utf-8")

    # Loop nas linhas do payload, na ordem
    for idx, row in df.iterrows():
        bet_id = get_bet_id(row)
        dow_raw = row.get(DOW_COL, None)
        dow_norm = normalize_dow(dow_raw)

        if pd.isna(dow_raw) or not dow_norm:
            p = 0.0
            d = 0
        else:
            if dow_norm in MODEL_FILENAMES:
                # carrega modelo correspondente, se ainda não carregou
                if dow_norm not in model_cache:
                    model_path = models_dir / MODEL_FILENAMES[dow_norm]
                    if not model_path.exists():
                        # se não existir modelo, não aprova
                        model_cache[dow_norm] = None
                    else:
                        model = joblib.load(model_path)
                        model_cache[dow_norm] = patch_sklearn_compat(model)

                model = model_cache[dow_norm]
                if model is None:
                    p = 0.0
                    d = 0
                else:
                    X_row = row[NUM_FEATURES + CAT_FEATURES].to_frame().T
                    proba = model.predict_proba(X_row)[0, 1]
                    # aplica piso de calibração
                    p = float(np.clip(proba, calib_floor, 1.0 - calib_floor))
                    d = int(p >= cutoff)
            else:
                # dia sem modelo (quinta, sexta, sábado, domingo)
                p = 0.0
                d = 0

        probas.append(p)
        decisions.append(d)

        # logging opcional
        if logfile_handle is not None:
            model_path = None
            try:
                model_path = str(models_dir / MODEL_FILENAMES.get(dow_norm, "")) if dow_norm else None
            except Exception:
                model_path = None
            mstat = None
            try:
                if model_path and Path(model_path).exists():
                    st = Path(model_path).stat()
                    mstat = {"mtime": st.st_mtime, "size": st.st_size}
            except Exception:
                mstat = None
            log_entry = {
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "bet_id": bet_id,
                "idx": int(idx),
                "dia_semana_raw": (None if pd.isna(dow_raw) else str(dow_raw)),
                "dia_semana_norm": dow_norm,
                "proba": p,
                "decision": d,
                "cutoff": cutoff,
                "calib_floor": calib_floor,
                "model_path": model_path,
                "model_stat": mstat,
                "payload_hash": _payload_hash(row),
            }
            logfile_handle.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    if logfile_handle is not None:
        logfile_handle.close()

    # Imprime CSV no stdout.
    # IMPORTANTE (PAD): formato legado é exatamente 2 linhas:
    #   proba,decision
    #   <proba>,<decision>
    out = sys.stdout
    out.write("proba,decision\n")
    if args.stdout_all_rows:
        for p, d in zip(probas, decisions):
            out.write(f"{p:.6f},{d}\n")
    else:
        # legado: só o último registro
        if probas:
            out.write(f"{float(probas[-1]):.6f},{int(decisions[-1])}\n")


if __name__ == "__main__":
    main()
