#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constrói uma coluna de região "ex-ante" para toda a base, treinando um classificador
apenas com features ex-ante (texto do evento/times/bookmaker) e usando como rótulo
uma heurística de região aplicada ao texto de competição da BetinAsia (quando existir).

Motivação:
- `BetinAsia.event info competition name` é ex-post (só aparece em parte das apostas executadas),
  então não pode ser usado diretamente no operacional.
- Porém, podemos usar essa parcela como "fonte de rótulo" para aprender um mapeamento a partir de
  informações ex-ante presentes em ~100% da base (`Evento`, `Time Home`, `Time Away`, `RebelBetting.Bookmaker`),
  e aplicar o classificador retroativamente para toda a base.

Saídas:
- /workspace/analysis_proba_raw/pro_portfolio_all/region_exante_pred.csv
  colunas: ID Aposta, BIA_ApostaUTC, region_label (quando houver), region_pred, region_pred_pmax

Obs:
- O modelo NÃO usa ROI/resultado.
- Para auditoria rápida, imprime métricas simples (holdout temporal) no stdout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")
OUT = Path("/workspace/analysis_proba_raw/pro_portfolio_all/region_exante_pred.csv")


def infer_region_from_competition(text: str) -> str:
    """
    Heurística determinística (mesma ideia usada no projeto) aplicada ao nome da competição,
    que costuma conter país/região (ex.: 'Sweden Superettan', 'Japan J-League Division 1').
    """
    if not text or str(text).strip() == "" or str(text).lower() in {"nan", "none", "null"}:
        return "desconhecida"
    s = str(text).lower()

    if any(k in s for k in ["australia", "austrália", "new zealand", "nova zelandia", "nova zelândia"]):
        return "oceania"
    # África / confederações / competições globais-regionais
    if any(k in s for k in ["caf ", "caf-", "africa", "afric", "african", "nations cup", "afcon"]):
        return "africa"
    if any(k in s for k in ["conmebol", "copa libertadores", "sudamericana"]):
        return "america_sul"
    if any(k in s for k in ["concacaf"]):
        return "america_norte"
    if any(k in s for k in ["uefa"]):
        # UEFA cobre toda Europa; mantemos como "europa" (ocidental) para não fragmentar demais
        return "europa_ocidental"
    if any(k in s for k in ["afc "]):
        return "asia"
    if any(
        k in s
        for k in [
            "saudi",
            "arábia",
            "arabia",
            "qatar",
            "catar",
            "uae",
            "emirates",
            "emirados",
            "iran",
            "iraq",
            "israel",
            "turkey",
            "turquia",
        ]
    ):
        return "oriente_medio"
    if any(
        k in s
        for k in [
            "japan",
            "japão",
            "korea",
            "coreia",
            "china",
            "índia",
            "india",
            "thailand",
            "tailand",
            "viet",
            "malaysia",
            "singapore",
            "indonesia",
        ]
    ):
        return "asia"

    west = [
        "england",
        "inglaterra",
        "spain",
        "espanha",
        "france",
        "frança",
        "italy",
        "itália",
        "germany",
        "alemanha",
        "netherlands",
        "holanda",
        "belgium",
        "bélgica",
        "portugal",
        "switzerland",
        "suíça",
        "austria",
        "áustria",
        "scotland",
        "escócia",
        "ireland",
        "irlanda",
        "norway",
        "noruega",
        "sweden",
        "suécia",
        "denmark",
        "dinamarca",
    ]
    east = [
        "poland",
        "polônia",
        "czech",
        "tcheca",
        "slovakia",
        "eslováquia",
        "hungary",
        "hungria",
        "romania",
        "romênia",
        "bulgaria",
        "bulgária",
        "serbia",
        "sérvia",
        "croatia",
        "croácia",
        "ukraine",
        "ucrânia",
        "russia",
        "rússia",
        "greece",
        "grécia",
    ]
    if any(k in s for k in west):
        return "europa_ocidental"
    if any(k in s for k in east):
        return "europa_oriental"

    if any(
        k in s
        for k in [
            "mexico",
            "méxico",
            "costa rica",
            "honduras",
            "guatemala",
            "panama",
            "panamá",
            "el salvador",
            "nicaragua",
            "nicarágua",
            "jamaica",
            "haiti",
            "haití",
            "dominican",
            "república dominicana",
        ]
    ):
        return "america_central"
    if any(k in s for k in ["usa", "united states", "estados unidos", "canada", "canadá"]):
        return "america_norte"
    if any(
        k in s
        for k in [
            "brazil",
            "brasil",
            "argentina",
            "chile",
            "colombia",
            "colômbia",
            "peru",
            "uruguay",
            "uruguai",
            "paraguay",
            "paraguai",
            "ecuador",
            "venezuela",
            "bolivia",
            "bolívia",
        ]
    ):
        return "america_sul"

    return "desconhecida"


def _text(s: Any) -> str:
    if s is None:
        return ""
    try:
        x = str(s)
    except Exception:
        return ""
    if x.lower() in {"nan", "none", "null"}:
        return ""
    return x


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(
        SCORED,
        parse_dates=["BIA_ApostaUTC"],
        usecols=[
            "ID Aposta",
            "BIA_ApostaUTC",
            "Evento",
            "Time Home",
            "Time Away",
            "RebelBetting.Bookmaker",
            "BetinAsia.event info competition name",
        ],
    )

    comp = df["BetinAsia.event info competition name"].astype(str)
    has_label = ~comp.str.lower().isin(["nan", "none", "null", ""])
    df["region_label"] = "desconhecida"
    df.loc[has_label, "region_label"] = comp[has_label].apply(infer_region_from_competition).astype(str)

    # Features ex-ante: texto do evento/times/bookmaker (sem BetinAsia)
    feat = (
        df["Time Home"].map(_text)
        + " | "
        + df["Time Away"].map(_text)
        + " | "
        + df["Evento"].map(_text)
        + " | bk="
        + df["RebelBetting.Bookmaker"].map(_text)
    )

    # treino só onde há label != desconhecida
    m = has_label & (df["region_label"].astype(str) != "desconhecida")
    n_lab = int(m.sum())
    if n_lab < 200:
        raise SystemExit(f"Poucos rótulos para treinar (n={n_lab}).")

    # split temporal simples: 80% antigo / 20% recente (para sanity)
    idx = np.where(m.to_numpy(bool))[0]
    idx = idx[np.argsort(df.loc[idx, "BIA_ApostaUTC"].to_numpy())]
    cut = int(max(1, round(0.8 * len(idx))))
    tr_idx = idx[:cut]
    te_idx = idx[cut:]

    Xtr = feat.iloc[tr_idx].to_numpy()
    ytr = df.loc[tr_idx, "region_label"].astype(str).to_numpy()
    Xte = feat.iloc[te_idx].to_numpy()
    yte = df.loc[te_idx, "region_label"].astype(str).to_numpy()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, f1_score

    pipe = Pipeline(
        [
            # char ngram funciona melhor para nomes próprios/abreviações e idiomas mistos
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)),
            # balancear classes (rotulação é bem desbalanceada)
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")),
        ]
    )
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    acc = float(accuracy_score(yte, pred))
    f1m = float(f1_score(yte, pred, average="macro"))
    print(f"region_exante clf holdout: n_train={len(tr_idx)} n_test={len(te_idx)} acc={acc:.3f} f1_macro={f1m:.3f}")

    # fit full labeled set and predict all
    pipe.fit(feat.iloc[idx].to_numpy(), df.loc[idx, "region_label"].astype(str).to_numpy())
    proba = pipe.predict_proba(feat.to_numpy())
    classes = pipe.classes_
    pmax = proba.max(axis=1)
    pred_all = classes[np.argmax(proba, axis=1)]

    out = pd.DataFrame(
        {
            "ID Aposta": df["ID Aposta"],
            "BIA_ApostaUTC": df["BIA_ApostaUTC"],
            "region_label": df["region_label"],
            "region_pred": pred_all,
            "region_pred_pmax": pmax.astype(float),
        }
    )
    out.to_csv(OUT, index=False)
    print(str(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

