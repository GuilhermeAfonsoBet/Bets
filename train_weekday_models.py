# -*- coding: utf-8 -*-
# train_weekday_models.py
#
# Treina 1 modelo por dia (segunda, terça, quarta)
# usando as mesmas features conceituais do modelo SexDom,
# só na metade final da base histórica.
#
# Salva:
#   C:\Bets\ModelosEstatísticos\model_logit_segunda.joblib
#   C:\Bets\ModelosEstatísticos\model_logit_terca.joblib
#   C:\Bets\ModelosEstatísticos\model_logit_quarta.joblib

import pandas as pd
import numpy as np
import joblib
import re
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

# ==========================
# CONFIGURAÇÕES
# ==========================

BASE_DIR    = Path(r"C:\Bets\ModelosEstatísticos")
ARQUIVO_DF  = BASE_DIR / "modeloEstatísticoChatGPT_30.10.25_modificado.xlsx"
MODELS_DIR  = BASE_DIR

DATE_COL   = "Data/Hora Aposta Realizada no BetinAsia"
DOW_COL    = "Dia Semana Aposta (UTC)"
TARGET_COL = "CLV Real"

# ---- mapeamento histórico -> payload (do seu de-para) ----
HIST_TO_PAYLOAD = {
    "Subtipo da Aposta": "Subtipo da Aposta",
    "ApostaLive.Número de casas disponíveis no momento da aposta":
        "Número de casas disponíveis no momento da aposta",
    "ApostaLive.Dif % maior odd e segunda maior":
        "Dif percent maior odd e segunda maior",
    "ApostaLive.Dif % maior odd e odd mediana":
        "Dif percent maior odd e odd mediana",
    "Dif Odds RB & BIA":
        "Dif Odds RB E BIA",
    "RebelBetting.MinutesToMatchStart":
        "MinutesToMatchStart",
    "TempoApostas.Tempo até o processamento":
        "TempoApostas.Tempo total bot",
    "Dia Semana Aposta (UTC)":
        "Dia Semana Aposta (UTC)",
    "Turno Aposta (UTC)":
        "Turno Aposta (UTC)",
    "ApostaLive.Casa aposta vencedora":
        "Casa aposta vencedora",
}

# Features no FORMATO DO PAYLOAD
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

# Melhor tipo de modelo por dia (da análise OOS com essas features, metade final)
BEST_MODEL_BY_DIA = {
    "segunda-feira": "isotonic",        # logit + calibração isotônica
    "terça-feira":   "sem_calibracao",  # logit puro
    "quarta-feira":  "sem_calibracao",  # logit puro
}

MODEL_FILENAMES = {
    "segunda-feira": "model_logit_segunda.joblib",
    "terça-feira":   "model_logit_terca.joblib",
    "quarta-feira":  "model_logit_quarta.joblib",
}

# ==========================
# FUNÇÕES AUXILIARES
# ==========================

def carregar_e_preparar_df(caminho: Path) -> pd.DataFrame:
    """Carrega a base histórica, renomeia colunas para o padrão do payload
    e retorna apenas a metade final (cronológica)."""
    df = pd.read_excel(caminho)
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]

    # renomear histórico -> payload onde for possível
    rename_dict = {
        hist: payload for hist, payload in HIST_TO_PAYLOAD.items()
        if hist in df.columns
    }
    df = df.rename(columns=rename_dict)

    # ordenar por data e pegar só metade final
    df = df.sort_values(by=DATE_COL).reset_index(drop=True)
    cut_idx = len(df) // 2
    cut_time = df.loc[cut_idx, DATE_COL]
    df = df[df[DATE_COL] >= cut_time].reset_index(drop=True)

    # converter col numéricas com vírgula decimal, se necessário
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


def criar_preprocessador():
    """Mesmo estilo de pré-processamento do modelo SexDom."""
    preprocess = ColumnTransformer(
        transformers=[
            ("num",
             Pipeline([
                 ("imp", SimpleImputer(strategy="median")),
                 ("sc", StandardScaler()),
             ]),
             NUM_FEATURES),
            ("cat",
             Pipeline([
                 ("imp", SimpleImputer(strategy="most_frequent")),
                 ("oh", OneHotEncoder(handle_unknown="ignore")),
             ]),
             CAT_FEATURES),
        ]
    )
    return preprocess


def criar_pipeline(tipo_modelo: str) -> Pipeline:
    """Cria pipeline com pré-processamento + modelo final."""
    preprocess = criar_preprocessador()
    base_lr = LogisticRegression(max_iter=1000)

    if tipo_modelo == "sem_calibracao":
        clf = base_lr

    elif tipo_modelo == "platt":
        # compatível com a versão do scikit-learn que usa 'estimator'
        clf = CalibratedClassifierCV(
            estimator=base_lr,
            method="sigmoid",
            cv=3,
        )

    elif tipo_modelo == "isotonic":
        clf = CalibratedClassifierCV(
            estimator=base_lr,
            method="isotonic",
            cv=3,
        )

    else:
        raise ValueError(f"Tipo de modelo desconhecido: {tipo_modelo}")

    return Pipeline([
        ("prep", preprocess),
        ("clf", clf),
    ])


def avaliar_auc_temporal(X: pd.DataFrame, y: pd.Series,
                         tipo_modelo: str, n_splits: int = 3) -> float:
    """Validação temporal (TimeSeriesSplit) para estimar AUC OOS."""
    n = len(X)
    if n < 30:
        return float("nan")

    tscv = TimeSeriesSplit(
        n_splits=min(n_splits, max(2, n // 20))
    )
    aucs = []
    for tr, te in tscv.split(X):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]

        pipe = criar_pipeline(tipo_modelo)
        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_te)[:, 1]
        if len(np.unique(y_te)) > 1:
            aucs.append(roc_auc_score(y_te, p))

    return float(np.mean(aucs)) if aucs else float("nan")


# ==========================
# MAIN
# ==========================

def main():
    print(f"Lendo base: {ARQUIVO_DF}")
    df = carregar_e_preparar_df(ARQUIVO_DF)

    y_bin = (df[TARGET_COL] > 0).astype(int)
    X_all = df[NUM_FEATURES + CAT_FEATURES].copy()
    dates = pd.to_datetime(df[DATE_COL])
    dow   = df[DOW_COL]

    resultados = []

    for dia, tipo in BEST_MODEL_BY_DIA.items():
        mask = (dow == dia)
        X_dia = X_all.loc[mask]
        y_dia = y_bin.loc[mask]
        dates_dia = dates.loc[mask]

        if len(X_dia) < 30:
            print(f"[{dia}] poucos dados ({len(X_dia)}) – não vou treinar.")
            continue

        # ordenar temporalmente
        order = np.argsort(dates_dia.values)
        X_dia = X_dia.iloc[order]
        y_dia = y_dia.iloc[order]

        # AUC temporal OOS (diagnóstico)
        auc_oos = avaliar_auc_temporal(X_dia, y_dia, tipo)
        print(f"[{dia}] AUC temporal OOS ≈ {auc_oos:.3f} (n={len(X_dia)})")

        # treinar modelo final com TODOS os dados desse dia
        modelo_final = criar_pipeline(tipo)
        modelo_final.fit(X_dia, y_dia)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = MODELS_DIR / MODEL_FILENAMES[dia]
        joblib.dump(modelo_final, out_path)
        print(f"[{dia}] modelo salvo em: {out_path}")

        resultados.append((dia, tipo, auc_oos, len(X_dia)))

    print("\nResumo:")
    for dia, tipo, auc, nobs in resultados:
        print(f"- {dia}: {tipo}, AUC≈{auc:.3f}, n={nobs}")


if __name__ == "__main__":
    main()
