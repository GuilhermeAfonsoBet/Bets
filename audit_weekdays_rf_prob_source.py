#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auditoria forense da origem do `ApostaLive.rf_prob` (weekdays).

Objetivo:
- Verificar se `ApostaLive.rf_prob` na base corresponde ao `proba_txt` impresso no stdout
  pelo `score_logit_weekdays_cli.py` (predict_proba + clip calib_floor, 6 casas decimais).
- Opcionalmente, quando disponível um `scoring_weekdays.jsonl` NOVO (com `payload_hash`),
  fazer o join determinístico base <-> log e auditar linha-a-linha.

Uso típico (local/estudo):
  python3 audit_weekdays_rf_prob_source.py \\
    --base-csv /workspace/pr1_snapshot/dedup_scored_base.csv \\
    --models-dir /workspace \\
    --outdir /workspace/analysis_proba_raw/pro_portfolio_all \\
    --last-days 30

Para auditar com o log real do RPA (Windows) após atualizar o CLI:
  python3 audit_weekdays_rf_prob_source.py \\
    --base-csv /workspace/pr1_snapshot/dedup_scored_base.csv \\
    --models-dir /workspace \\
    --log-jsonl /path/to/scoring_weekdays.jsonl \\
    --outdir /workspace/analysis_proba_raw/pro_portfolio_all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

# Reutiliza exatamente as funções/constantes do CLI para não divergir.
import score_logit_weekdays_cli as cli


def _clip_floor(p: np.ndarray, floor: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.clip(p, float(floor), 1.0 - float(floor))


def _load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return pd.DataFrame(rows)


def _safe_numeric(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x.where((x >= 0.0) & (x <= 1.0))


def _coerce_decimal_string(x: object) -> float:
    """
    Coerção robusta para numéricos vindos como string.
    - Se tem muitos pontos e uma vírgula: pontos são milhares, vírgula é decimal
    - Se tem uma vírgula e nenhum ponto: vírgula é decimal
    - Caso contrário: tenta float direto (ponto decimal)
    """
    if x is None:
        return float("nan")
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return float("nan")
    if s.count(",") == 1 and s.count(".") > 1:
        s = s.replace(".", "").replace(",", ".")
        try:
            return float(s)
        except Exception:
            return float("nan")
    if s.count(",") == 1 and s.count(".") == 0:
        try:
            return float(s.replace(",", "."))
        except Exception:
            return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")


def _prepare_payload(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Prepara payload para scoring.

    mode:
      - 'weekdays_cli': replica exatamente o `score_logit_weekdays_cli.preparar_payload`
      - 'legacy_buggy': replica o parsing antigo (bugado) que removia '.' sempre
      - 'robust': coerção robusta (aceita ponto OU vírgula decimal sem destruir ponto decimal)
    """
    if mode == "weekdays_cli":
        return cli.preparar_payload(df)
    if mode == "legacy_buggy":
        out = df.copy()
        out.columns = [__import__("re").sub(r"\s+", " ", str(c)).strip() for c in out.columns]
        for col in cli.NUM_FEATURES:
            if col in out.columns:
                s = out[col].astype(str)
                s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
                out[col] = pd.to_numeric(s, errors="coerce")
        return out
    if mode != "robust":
        raise ValueError("parse-mode inválido (use 'weekdays_cli', 'legacy_buggy' ou 'robust').")

    out = df.copy()
    # normaliza nomes de colunas
    out.columns = [__import__("re").sub(r"\s+", " ", str(c)).strip() for c in out.columns]
    # numéricas
    for col in cli.NUM_FEATURES:
        if col in out.columns:
            out[col] = out[col].apply(_coerce_decimal_string)
    return out


def _pick_dt_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("BIA_ApostaUTC", "ApostaLive.ApostaUTC", "ApostaUTC", "ApostaLive.DataHoraUTC"):
        if c in df.columns:
            return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-csv", default="/workspace/pr1_snapshot/dedup_scored_base.csv")
    ap.add_argument("--models-dir", default="/workspace")
    ap.add_argument("--log-jsonl", default=None)
    ap.add_argument("--outdir", default="/workspace/analysis_proba_raw/pro_portfolio_all")
    ap.add_argument("--last-days", type=int, default=30)
    ap.add_argument("--calib-floor", type=float, default=0.005)
    ap.add_argument(
        "--parse-mode",
        choices=["weekdays_cli", "legacy_buggy", "robust"],
        default="weekdays_cli",
        help="Como coagir numéricos do payload (robust aceita ponto decimal).",
    )
    ap.add_argument(
        "--compare-legacy",
        action="store_true",
        help="Também recalcula probas com parsing antigo (bugado) e compara com rf_prob.",
    )
    args = ap.parse_args()

    base_csv = Path(args.base_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(base_csv, dtype=str)
    df = _prepare_payload(df_raw, mode=args.parse_mode)

    # filtro last N dias (se houver coluna de data)
    dt_col = _pick_dt_col(df)
    if dt_col is not None:
        dt = pd.to_datetime(df[dt_col], errors="coerce", utc=False)
        maxdt = dt.max()
        if pd.notna(maxdt):
            cut = maxdt - pd.Timedelta(days=int(args.last_days))
            df = df.loc[dt >= cut].copy()

    # dow normalizado e modelos apenas Seg/Ter/Qua
    df["dow_norm"] = df[cli.DOW_COL].apply(cli.normalize_dow)
    mask_model = df["dow_norm"].isin(list(cli.MODEL_FILENAMES.keys()))
    df = df.loc[mask_model].copy()

    # calcula proba "como no CLI"
    floor = float(args.calib_floor)
    proba_cli = np.full(len(df), np.nan, dtype=float)

    models_dir = Path(args.models_dir)
    model_cache: Dict[str, Any] = {}

    for dow_norm, filename in cli.MODEL_FILENAMES.items():
        m = (df["dow_norm"] == dow_norm).to_numpy()
        if not m.any():
            continue
        if dow_norm not in model_cache:
            model_path = models_dir / filename
            model = cli.patch_sklearn_compat(__import__("joblib").load(model_path))
            model_cache[dow_norm] = model
        model = model_cache[dow_norm]
        X = df.loc[m, cli.NUM_FEATURES + cli.CAT_FEATURES]
        p = model.predict_proba(X)[:, 1]
        proba_cli[m] = _clip_floor(p, floor)

    df["proba_cli"] = proba_cli
    df["proba_txt"] = df["proba_cli"].map(lambda x: (f"{float(x):.6f}" if pd.notna(x) else ""))

    # Opcional: calcula proba com parsing antigo (bugado) para explicar divergências históricas
    if args.compare_legacy:
        df_legacy = _prepare_payload(df_raw.loc[df.index], mode="legacy_buggy")
        df_legacy["dow_norm"] = df_legacy[cli.DOW_COL].apply(cli.normalize_dow)
        proba_legacy = np.full(len(df_legacy), np.nan, dtype=float)
        model_cache2: Dict[str, Any] = {}
        for dow_norm, filename in cli.MODEL_FILENAMES.items():
            m = (df_legacy["dow_norm"] == dow_norm).to_numpy()
            if not m.any():
                continue
            if dow_norm not in model_cache2:
                model_path = models_dir / filename
                model = cli.patch_sklearn_compat(__import__("joblib").load(model_path))
                model_cache2[dow_norm] = model
            model = model_cache2[dow_norm]
            X = df_legacy.loc[m, cli.NUM_FEATURES + cli.CAT_FEATURES]
            p = model.predict_proba(X)[:, 1]
            proba_legacy[m] = _clip_floor(p, floor)
        df["proba_cli_legacy_buggy"] = proba_legacy
        df["proba_txt_legacy_buggy"] = df["proba_cli_legacy_buggy"].map(
            lambda x: (f"{float(x):.6f}" if pd.notna(x) else "")
        )

    # rf_prob
    if "ApostaLive.rf_prob" not in df.columns:
        raise SystemExit("Coluna `ApostaLive.rf_prob` não encontrada na base.")
    df["rf_prob"] = _safe_numeric(df["ApostaLive.rf_prob"])

    # métricas de aderência do rf_prob vs proba_cli
    ok = df["rf_prob"].notna() & df["proba_cli"].notna()
    g = df.loc[ok].copy()
    if g.empty:
        raise SystemExit("Sem linhas válidas para comparar (rf_prob/proba_cli).")

    g["abs_diff"] = (g["rf_prob"] - g["proba_cli"]).abs()
    g["match6"] = (np.round(g["rf_prob"], 6) == np.round(g["proba_cli"], 6)).astype(int)

    if args.compare_legacy and "proba_cli_legacy_buggy" in g.columns:
        g["abs_diff_legacy"] = (g["rf_prob"] - g["proba_cli_legacy_buggy"]).abs()
        g["match6_legacy"] = (
            np.round(g["rf_prob"], 6) == np.round(g["proba_cli_legacy_buggy"], 6)
        ).astype(int)
        g["closer_to_legacy"] = (g["abs_diff_legacy"] < g["abs_diff"]).astype(int)

    # por dia
    per_day = []
    for d0 in ["segunda-feira", "terca-feira", "quarta-feira"]:
        gg = g[g["dow_norm"] == d0]
        if gg.empty:
            continue
        corr = float(np.corrcoef(gg["rf_prob"], gg["proba_cli"])[0, 1]) if len(gg) >= 3 else float("nan")
        row = {
            "dow_norm": d0,
            "n": int(len(gg)),
            "mae": float(gg["abs_diff"].mean()),
            "p95_abs_diff": float(gg["abs_diff"].quantile(0.95)),
            "corr": corr,
            "match6_pct": float(100.0 * gg["match6"].mean()),
        }
        if args.compare_legacy and "abs_diff_legacy" in gg.columns:
            row.update(
                {
                    "mae_legacy": float(gg["abs_diff_legacy"].mean()),
                    "match6_legacy_pct": float(100.0 * gg["match6_legacy"].mean()),
                    "pct_rows_closer_to_legacy": float(100.0 * gg["closer_to_legacy"].mean()),
                }
            )
        per_day.append(row)
    per_day_df = pd.DataFrame(per_day).sort_values("dow_norm")

    # salvar CSV de mismatches (top 200)
    worst = g.sort_values("abs_diff", ascending=False).head(200)
    cols_out = [
        c
        for c in [
            dt_col,
            cli.DOW_COL,
            "dow_norm",
            "ApostaLive.rf_prob",
            "rf_prob",
            "proba_cli",
            "proba_txt",
            "proba_cli_legacy_buggy",
            "proba_txt_legacy_buggy",
            "abs_diff",
            "match6",
            "abs_diff_legacy",
            "match6_legacy",
            "closer_to_legacy",
        ]
        if c is not None and c in worst.columns
    ]
    out_csv = outdir / "audit_weekdays_rf_prob_vs_cli.csv"
    worst[cols_out].to_csv(out_csv, index=False)

    # se log-jsonl foi fornecido e tem payload_hash, faz join determinístico
    log_md_extra = ""
    if args.log_jsonl:
        log_path = Path(args.log_jsonl)
        if log_path.exists():
            logs = _load_jsonl(log_path)
            if "payload_hash" in logs.columns:
                # calcula hash do payload na base
                df["payload_hash"] = df.apply(lambda r: cli._payload_hash(r), axis=1)
                logs = logs.rename(columns={"dia_semana_norm": "dow_norm", "proba": "proba_log"})
                # normaliza caso o log tenha outro nome de coluna
                if "dow_norm" not in logs.columns and "dia_semana_norm" in logs.columns:
                    logs["dow_norm"] = logs["dia_semana_norm"]
                join_cols = ["payload_hash"]
                if "dow_norm" in logs.columns:
                    join_cols.append("dow_norm")
                merged = df.merge(logs, on=join_cols, how="inner", suffixes=("", "_log"))
                if not merged.empty and "proba_log" in merged.columns:
                    merged["proba_log"] = pd.to_numeric(merged["proba_log"], errors="coerce")
                    merged["abs_diff_rf_vs_log"] = (pd.to_numeric(merged["rf_prob"], errors="coerce") - merged["proba_log"]).abs()
                    merged["abs_diff_cli_vs_log"] = (pd.to_numeric(merged["proba_cli"], errors="coerce") - merged["proba_log"]).abs()
                    log_md_extra = (
                        "\n### Join com `scoring_weekdays.jsonl` (via `payload_hash`)\n"
                        f"- Linhas casadas: **{len(merged)}**\n"
                        f"- MAE |rf - log|: **{float(merged['abs_diff_rf_vs_log'].mean()):.6f}**\n"
                        f"- MAE |cli - log|: **{float(merged['abs_diff_cli_vs_log'].mean()):.6f}**\n"
                    )
                    # salva amostra do join
                    merged_out = outdir / "audit_weekdays_join_log.csv"
                    keep = [c for c in ["payload_hash", "dow_norm", "rf_prob", "proba_cli", "proba_log", "abs_diff_rf_vs_log", "abs_diff_cli_vs_log"] if c in merged.columns]
                    merged[keep].head(5000).to_csv(merged_out, index=False)
            else:
                log_md_extra = (
                    "\n### Join com `scoring_weekdays.jsonl`\n"
                    "- O arquivo fornecido **não contém `payload_hash`**.\n"
                    "- Para auditoria determinística linha-a-linha, gere o log com o `score_logit_weekdays_cli.py` atualizado.\n"
                )

    # markdown resumo
    md = []
    md.append("## Auditoria — `ApostaLive.rf_prob` vs stdout do `score_logit_weekdays_cli.py`\n\n")
    md.append(f"- Base: `{base_csv}`\n")
    md.append(f"- Modelos: `{models_dir}`\n")
    md.append(f"- `calib_floor` (clip): **{floor}**\n")
    md.append(f"- parse-mode: **{args.parse_mode}**\n")
    if dt_col is not None:
        md.append(f"- Janela: últimos **{int(args.last_days)}** dias (coluna `{dt_col}`, se disponível)\n")
    md.append("\n### Resumo (rf_prob vs proba_cli)\n")
    md.append(f"- N (válidos): **{len(g)}**\n")
    md.append(f"- MAE |rf - cli|: **{float(g['abs_diff'].mean()):.6f}**\n")
    md.append(f"- P95 |rf - cli|: **{float(g['abs_diff'].quantile(0.95)):.6f}**\n")
    md.append(f"- % match exato (@6 casas): **{100.0*float(g['match6'].mean()):.2f}%**\n\n")
    if args.compare_legacy and "abs_diff_legacy" in g.columns:
        md.append("### Comparação com parsing antigo (bugado)\n")
        md.append(f"- MAE |rf - legacy|: **{float(g['abs_diff_legacy'].mean()):.6f}**\n")
        md.append(f"- % match exato legacy (@6 casas): **{100.0*float(g['match6_legacy'].mean()):.2f}%**\n")
        md.append(
            f"- % linhas em que legacy é MAIS perto do rf (|rf-legacy| < |rf-cli|): **{100.0*float(g['closer_to_legacy'].mean()):.2f}%**\n\n"
        )
    md.append("### Por dia\n")
    for _, r in per_day_df.iterrows():
        line = (
            f"- **{r['dow_norm']}**: n={int(r['n'])}, corr={r['corr']:.3f}, "
            f"MAE={r['mae']:.6f}, p95={r['p95_abs_diff']:.6f}, match6={r['match6_pct']:.2f}%"
        )
        if args.compare_legacy and "mae_legacy" in r:
            line += (
                f" | legacy_MAE={r['mae_legacy']:.6f}, legacy_match6={r['match6_legacy_pct']:.2f}%, "
                f"legacy_closer={r['pct_rows_closer_to_legacy']:.2f}%"
            )
        md.append(line + "\n")
    md.append("\n### Arquivos\n")
    md.append(f"- Mismatches (top 200): `{out_csv}`\n")
    if args.log_jsonl:
        md.append(f"- Log usado: `{args.log_jsonl}`\n")
        md.append("- Join (se disponível): `audit_weekdays_join_log.csv`\n")
    md.append(log_md_extra)

    out_md = outdir / "audit_weekdays_rf_prob_vs_cli.md"
    out_md.write_text("".join(md), encoding="utf-8")

    print(str(out_md))
    print(str(out_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

