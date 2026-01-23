#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera região ex-ante determinística usando `RebelBetting.EventName` (do Excel original),
que frequentemente contém o país/competição (ex.: "Egypt Division 2 B").

Saída (compatível com o gating existente):
  /workspace/analysis_proba_raw/pro_portfolio_all/region_exante_pred.csv
    - ID Aposta
    - region_pred              (categoria de região)
    - region_pred_pmax         (1.0 quando determinado; 0.0 quando desconhecida)
    - region_source            ("eventname_country", "eventname_keyword", "unknown")

Observações:
- Não usa nenhum dado ex-post, ROI, resultado, nem BetinAsia.
- Mapeia país -> continente (via pycountry + pycountry_convert) e então para buckets do projeto.
- Para casos como "UEFA Champions League" ou "Club Friendly", tenta keywords; senão, cai em "desconhecida".
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

import pycountry
import pycountry_convert as pcv


OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")
EVENTNAME_MAP = OUT_DIR / "eventname_by_id_from_excel_17.01.2026.csv"
OUT = OUT_DIR / "region_exante_pred.csv"


MIDDLE_EAST = {
    "AE", "SA", "QA", "KW", "BH", "OM", "YE", "IR", "IQ", "IL", "JO", "LB", "SY", "TR", "PS"
}

# Central America / Caribbean (parte cai em NA no continente)
CENTRAL_AM = {
    "MX", "GT", "BZ", "SV", "HN", "NI", "CR", "PA",
    "CU", "DO", "HT", "JM", "TT", "BS", "BB", "GD", "LC", "VC", "AG", "DM", "KN",
}

EASTERN_EU = {
    "PL", "CZ", "SK", "HU", "RO", "BG", "RS", "HR", "UA", "RU", "GR", "BA", "AL", "MK", "ME", "SI", "BY", "MD",
    "LT", "LV", "EE",
}

# Non-sovereign football "countries"
SPECIAL_COUNTRY_TO_ALPHA2 = {
    "england": "GB",
    "scotland": "GB",
    "wales": "GB",
    "northern ireland": "GB",
    "czech republic": "CZ",
    "usa": "US",
    "uae": "AE",
    "u.a.e.": "AE",
}


def _clean_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _alpha2_to_region(alpha2: str) -> str:
    a2 = (alpha2 or "").upper()
    if not a2 or len(a2) != 2:
        return "desconhecida"

    # middle east override (AS)
    if a2 in MIDDLE_EAST:
        return "oriente_medio"

    try:
        cont = pcv.country_alpha2_to_continent_code(a2)
    except Exception:
        return "desconhecida"

    if cont == "AF":
        return "africa"
    if cont == "OC":
        return "oceania"
    if cont == "SA":
        return "america_sul"
    if cont == "NA":
        return "america_central" if a2 in CENTRAL_AM else "america_norte"
    if cont == "EU":
        return "europa_oriental" if a2 in EASTERN_EU else "europa_ocidental"
    if cont == "AS":
        return "asia"
    return "desconhecida"


def _lookup_country_alpha2(country: str) -> Optional[str]:
    s = _clean_text(country).lower()
    if not s:
        return None
    if s in SPECIAL_COUNTRY_TO_ALPHA2:
        return SPECIAL_COUNTRY_TO_ALPHA2[s]
    # pycountry lookup (robusto)
    try:
        c = pycountry.countries.lookup(country)
        return getattr(c, "alpha_2", None)
    except Exception:
        return None


def _extract_country_prefix(event_name: str) -> Optional[str]:
    """
    Extrai candidato de país do início do EventName:
      - tenta casar o maior prefixo (até 4 palavras) que seja reconhecido como país.
    """
    s = _clean_text(event_name)
    if not s:
        return None
    # normaliza: mantém letras, espaços e hífens
    s2 = re.sub(r"[^A-Za-zÀ-ÿ\\-\\s\\.]", " ", s).strip()
    s2 = re.sub(r"\\s+", " ", s2)
    parts = s2.split(" ")
    if not parts:
        return None
    # tenta do maior para o menor
    max_w = min(4, len(parts))
    for k in range(max_w, 0, -1):
        cand = " ".join(parts[:k]).strip()
        # remove pontuação residual
        cand2 = cand.replace(".", "").strip()
        if _lookup_country_alpha2(cand2):
            return cand2
    return None


def _keyword_region(event_name: str) -> Optional[str]:
    s = _clean_text(event_name).lower()
    if not s:
        return None
    # confederações / competições globais
    if "caf" in s or "africa" in s or "african" in s:
        return "africa"
    if "conmebol" in s or "libertadores" in s or "sudamericana" in s:
        return "america_sul"
    if "concacaf" in s:
        return "america_central"
    if "uefa" in s:
        return "europa_ocidental"
    if re.search(r"\\bafc\\b", s):
        return "asia"
    return None


def classify_region_from_eventname(event_name: str) -> Tuple[str, float, str]:
    # 1) tentar extrair país
    c = _extract_country_prefix(event_name)
    if c:
        a2 = _lookup_country_alpha2(c)
        if a2:
            reg = _alpha2_to_region(a2)
            if reg != "desconhecida":
                return reg, 1.0, "eventname_country"
    # 2) fallback por keywords
    kw = _keyword_region(event_name)
    if kw:
        return kw, 0.9, "eventname_keyword"
    return "desconhecida", 0.0, "unknown"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not EVENTNAME_MAP.exists():
        raise FileNotFoundError(f"Arquivo ausente: {EVENTNAME_MAP}")
    df = pd.read_csv(EVENTNAME_MAP, usecols=["ID Aposta", "RebelBetting.EventName"])
    df["ID Aposta"] = pd.to_numeric(df["ID Aposta"], errors="coerce")
    df = df.dropna(subset=["ID Aposta"]).copy()
    df["ID Aposta"] = df["ID Aposta"].astype(int)
    df["RebelBetting.EventName"] = df["RebelBetting.EventName"].astype(str)

    regs = []
    pmax = []
    src = []
    for s in df["RebelBetting.EventName"].tolist():
        r, p, so = classify_region_from_eventname(s)
        regs.append(r)
        pmax.append(float(p))
        src.append(so)

    out = pd.DataFrame(
        {
            "ID Aposta": df["ID Aposta"],
            "region_pred": regs,
            "region_pred_pmax": np.asarray(pmax, dtype=float),
            "region_source": src,
        }
    )
    out.to_csv(OUT, index=False)
    print(str(OUT))
    # quick stats
    vc = out["region_pred"].value_counts()
    print("region_pred value_counts (top):")
    print(vc.head(15).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

