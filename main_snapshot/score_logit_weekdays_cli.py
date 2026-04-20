# -*- coding: utf-8 -*-
# score_logit_weekdays_cli.py

import argparse
import sys
import json
from pathlib import Path
import re

import pandas as pd
import numpy as np
import joblib

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

# Nomes dos arquivos de modelo (dentro de --models-dir)
MODEL_FILENAMES = {
    "segunda-feira": "model_logit_segunda.joblib",
    "terça-feira":   "model_logit_terca.joblib",
    "quarta-feira":  "model_logit_quarta.joblib",
}


def preparar_payload(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]

    # converter colunas numéricas com vírgula decimal, se vierem como texto
    for col in NUM_FEATURES:
        if col in df.columns and df[col].dtype == object:
            s = (
                df[col]
                .astype(str)
                .str.replace(".", "", regex=False)   # remove separador de milhar
                .str.replace(",", ".", regex=False)  # vírgula -> ponto decimal
            )
            df[col] = pd.to_numeric(s, errors="coerce")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--csvin", required=True)
    parser.add_argument("--cutoff", required=True, type=float)
    parser.add_argument("--calib-floor", required=False, type=float, default=0.0)
    parser.add_argument("--logfile", required=False)

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
        dow = row.get(DOW_COL, None)

        if pd.isna(dow):
            p = 0.0
            d = 0
        else:
            dow_str = str(dow).strip()
            if dow_str in MODEL_FILENAMES:
                # carrega modelo correspondente, se ainda não carregou
                if dow_str not in model_cache:
                    model_path = models_dir / MODEL_FILENAMES[dow_str]
                    if not model_path.exists():
                        # se não existir modelo, não aprova
                        model_cache[dow_str] = None
                    else:
                        model_cache[dow_str] = joblib.load(model_path)

                model = model_cache[dow_str]
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
            log_entry = {
                "idx": int(idx),
                "dia_semana": str(row.get(DOW_COL, "")),
                "proba": p,
                "decision": d,
                "cutoff": cutoff,
                "calib_floor": calib_floor,
            }
            logfile_handle.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    if logfile_handle is not None:
        logfile_handle.close()

    # Imprime CSV no stdout
    out = sys.stdout
    out.write("proba,decision\n")
    for p, d in zip(probas, decisions):
        out.write(f"{p:.6f},{d}\n")


if __name__ == "__main__":
    main()
