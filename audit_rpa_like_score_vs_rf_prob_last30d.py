#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checagem objetiva de aderência entre:
- score "RPA-like" reconstruído a partir das colunas do dataset (probas logit + calibração isotônica SexDom)
- coluna `ApostaLive.rf_prob`

Regras do score RPA-like (conforme scripts do usuário):
- segunda-feira: proba_raw_segunda  (weekdays_cli: modelo por dia + clip 0.005)
- terça-feira:   proba_raw_terca
- quarta-feira:  proba_raw_quarta
- sexta/sábado/domingo: proba_cal_sexdom = isotonic(clv_calib_SexDom) aplicado em proba_raw_sexdom + floor 0.005
- quinta-feira: ignorar (a pedido)

Relatório restrito aos últimos 30 dias (a partir da data máxima no dataset).

Saídas:
- analysis_proba_raw/pro_portfolio_all/rpa_like_vs_rf_prob_last30d.csv (amostra e diffs)
- analysis_proba_raw/pro_portfolio_all/rpa_like_vs_rf_prob_last30d.md  (resumo)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")
OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")
CALIB_SEXDOM = Path("/workspace/clv_calib_SexDom.json")
CALIB_FLOOR = 0.005


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])

    if "ApostaLive.rf_prob" not in df.columns:
        raise SystemExit("Coluna ApostaLive.rf_prob não encontrada.")

    # compute proba_cal_sexdom
    if not CALIB_SEXDOM.exists():
        raise SystemExit("Arquivo clv_calib_SexDom.json não encontrado.")
    cal = json.loads(CALIB_SEXDOM.read_text(encoding="utf-8"))
    x = np.asarray(cal.get("isotonic", {}).get("x", []), dtype=float)
    y = np.asarray(cal.get("isotonic", {}).get("y", []), dtype=float)
    raw = pd.to_numeric(df["proba_raw_sexdom"], errors="coerce").to_numpy(dtype=float)
    iso = np.interp(raw, x, y, left=float(y[0]), right=float(y[-1])) if x.size and y.size else raw
    proba_cal_sexdom = np.clip(np.maximum(iso, CALIB_FLOOR), 0.0, 1.0)

    dow = df["dow_pt"].astype(str).to_numpy()
    score = np.full(len(df), np.nan, dtype=float)
    pseg = pd.to_numeric(df["proba_raw_segunda"], errors="coerce").to_numpy(dtype=float)
    pter = pd.to_numeric(df["proba_raw_terca"], errors="coerce").to_numpy(dtype=float)
    pqua = pd.to_numeric(df["proba_raw_quarta"], errors="coerce").to_numpy(dtype=float)
    m = dow == "segunda-feira"
    score[m] = pseg[m]
    m = dow == "terça-feira"
    score[m] = pter[m]
    m = dow == "quarta-feira"
    score[m] = pqua[m]
    m = np.isin(dow, ["sexta-feira", "sábado", "domingo"])
    score[m] = proba_cal_sexdom[m]

    rf = pd.to_numeric(df["ApostaLive.rf_prob"], errors="coerce").to_numpy(dtype=float)
    rf[(~np.isfinite(rf)) | (rf < 0.0) | (rf > 1.0)] = np.nan

    # last 30 days
    maxdt = df["BIA_ApostaUTC"].max()
    cut = maxdt - pd.Timedelta(days=30)
    dt = df["BIA_ApostaUTC"].to_numpy("datetime64[ns]")
    mask = (dt >= np.datetime64(cut)) & (dow != "quinta-feira") & np.isfinite(rf) & np.isfinite(score)

    out = pd.DataFrame(
        {
            "BIA_ApostaUTC": df.loc[mask, "BIA_ApostaUTC"].reset_index(drop=True),
            "dow_pt": pd.Series(dow[mask]),
            "rf_prob": rf[mask],
            "score_rpa_like": score[mask],
        }
    )
    out["abs_diff"] = np.abs(out["rf_prob"] - out["score_rpa_like"])
    out["match_6dec"] = (np.round(out["rf_prob"], 6) == np.round(out["score_rpa_like"], 6)).astype(int)

    out_path = OUT_DIR / "rpa_like_vs_rf_prob_last30d.csv"
    out.to_csv(out_path, index=False)

    # summary md
    n = len(out)
    mae = float(out["abs_diff"].mean()) if n else float("nan")
    maxd = float(out["abs_diff"].max()) if n else float("nan")
    match = float(out["match_6dec"].mean()) if n else float("nan")
    corr = float(np.corrcoef(out["rf_prob"], out["score_rpa_like"])[0, 1]) if n >= 3 else float("nan")

    lines = []
    lines.append("## Checagem (últimos 30 dias) — score RPA-like vs `ApostaLive.rf_prob`\n\n")
    lines.append(f"- Janela: **{cut} .. {maxdt}**\n")
    lines.append("- Quinta-feira: **excluída** (a pedido)\n\n")
    lines.append(f"- N: **{n}**\n")
    lines.append(f"- Correlação: **{corr:.3f}**\n")
    lines.append(f"- MAE |rf - score|: **{mae:.4f}**\n")
    lines.append(f"- Máx |rf - score|: **{maxd:.4f}**\n")
    lines.append(f"- % match exato (arredondado 6 casas): **{match*100:.1f}%**\n\n")

    lines.append("### Por dia\n")
    for d0 in ["segunda-feira", "terça-feira", "quarta-feira", "sexta-feira", "sábado", "domingo"]:
        g = out[out["dow_pt"] == d0]
        if len(g) < 5:
            continue
        c = float(np.corrcoef(g["rf_prob"], g["score_rpa_like"])[0, 1])
        mae0 = float(np.mean(np.abs(g["rf_prob"] - g["score_rpa_like"])))
        m0 = float(np.mean(g["match_6dec"]))
        lines.append(f"- **{d0}**: n={len(g)}, corr={c:.3f}, MAE={mae0:.4f}, match6={m0*100:.1f}%\n")

    lines.append("\n### Arquivos\n")
    lines.append(f"- CSV: `analysis_proba_raw/pro_portfolio_all/{out_path.name}`\n")
    (OUT_DIR / "rpa_like_vs_rf_prob_last30d.md").write_text("".join(lines), encoding="utf-8")

    print(str(out_path))
    print(str(OUT_DIR / "rpa_like_vs_rf_prob_last30d.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

