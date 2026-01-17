#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audita artefatos exportados do PAD/RPA:

- payload%betID%.csv (1 linha, com header)
- stdout.csv (betID ; "stdout bruto do CLI")

O objetivo é responder: o número gravado (stdout) bate com o que o CLI
deveria produzir para aquele payload e models-dir?

Uso:
  python3 audit_rpa_artifacts.py \
    --input-dir /path/to/dir \
    --models-dir-weekdays "C:\\Bets\\ModelosEstatísticos" \
    --models-dir-by-dow "C:\\Bets\\ModelosEstatísticos" \
    --cutoff-weekdays 0 \
    --cutoff-by-dow 0.62 \
    --calib-floor 0.005
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def parse_stdout_csv(path: Path) -> pd.DataFrame:
    # Formato observado: betID ; "proba,decision\r\n0.123456,False\r\n"
    df = pd.read_csv(
        path,
        sep=";",
        header=None,
        names=["bet_id", "stdout_raw"],
        engine="python",
        quotechar='"',
    )
    df["bet_id"] = df["bet_id"].astype(str).str.strip().str.zfill(6)

    def extract_proba(s: str):
        m = re.search(r"\n([0-9.]+),", str(s))
        return float(m.group(1)) if m else np.nan

    def extract_decision(s: str):
        m = re.search(r"\n[0-9.]+,([A-Za-z0-9]+)", str(s))
        return m.group(1) if m else None

    df["stdout_proba"] = df["stdout_raw"].map(extract_proba)
    df["stdout_decision"] = df["stdout_raw"].map(extract_decision)
    return df


def choose_mode(dow: str) -> str:
    s = str(dow).strip().lower()
    # Heurística operacional padrão:
    # - sex/sáb/dom: by_dow (SexDom calibrado)
    # - qui: costuma usar by_dow (SegQui calibrado) em muitas versões
    # - seg/ter/qua: weekdays (modelos por dia)
    if s.startswith("sexta") or s.startswith("sáb") or s.startswith("sab") or s.startswith("dom"):
        return "by_dow"
    if s.startswith("qui"):
        return "by_dow"
    return "weekdays"


def run_cli(cli_path: Path, models_dir: str, csvin: Path, cutoff: float, calib_floor: float) -> tuple[float, str]:
    cmd = [
        "python3",
        str(cli_path),
        "--models-dir",
        str(models_dir),
        "--csvin",
        str(csvin),
        "--cutoff",
        str(cutoff),
        "--calib-floor",
        str(calib_floor),
    ]
    out = subprocess.check_output(cmd, text=True)
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    # header: proba,decision
    proba_s, decision_s = lines[1].split(",", 1)
    return float(proba_s), decision_s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--stdout-csv", default="stdout.csv")
    ap.add_argument("--models-dir-weekdays", required=True)
    ap.add_argument("--models-dir-by-dow", required=True)
    ap.add_argument("--cli-weekdays", default="/workspace/score_logit_weekdays_cli.py")
    ap.add_argument("--cli-by-dow", default="/workspace/score_logit_by_dow_cli.py")
    ap.add_argument("--cutoff-weekdays", type=float, default=0.0)
    ap.add_argument("--cutoff-by-dow", type=float, default=0.62)
    ap.add_argument("--calib-floor", type=float, default=0.005)
    ap.add_argument("--out-csv", default="audit_rpa_artifacts_report.csv")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    stdout_path = in_dir / args.stdout_csv
    if not stdout_path.exists():
        raise SystemExit(f"stdout.csv não encontrado: {stdout_path}")

    df_stdout = parse_stdout_csv(stdout_path)
    stdout_map = dict(zip(df_stdout["bet_id"], df_stdout["stdout_proba"]))

    payload_files = sorted([p for p in in_dir.glob("payload*.csv") if p.name != "payload.csv"])
    if not payload_files:
        raise SystemExit(f"Nenhum payload%betID%.csv encontrado em {in_dir}")

    rows = []
    for p in payload_files:
        d = pd.read_csv(p, sep=";", dtype=str)
        if d.empty:
            continue
        bet_id = str(d.iloc[0].get("IDAposta", "")).strip().zfill(6)
        dow = str(d.iloc[0].get("Dia Semana Aposta (UTC)", "")).strip()
        mode = choose_mode(dow)

        if mode == "weekdays":
            proba_cli, decision_cli = run_cli(
                Path(args.cli_weekdays),
                args.models_dir_weekdays,
                p,
                args.cutoff_weekdays,
                args.calib_floor,
            )
        else:
            proba_cli, decision_cli = run_cli(
                Path(args.cli_by_dow),
                args.models_dir_by_dow,
                p,
                args.cutoff_by_dow,
                args.calib_floor,
            )

        proba_stdout = stdout_map.get(bet_id, np.nan)
        diff = proba_cli - proba_stdout if np.isfinite(proba_stdout) else np.nan
        match6 = (
            bool(np.isfinite(proba_stdout))
            and (round(float(proba_cli), 6) == round(float(proba_stdout), 6))
        )

        rows.append(
            {
                "bet_id": bet_id,
                "dow": dow,
                "mode": mode,
                "payload_file": p.name,
                "stdout_proba": proba_stdout,
                "cli_proba": proba_cli,
                "diff": diff,
                "match6": int(match6),
                "cli_decision": decision_cli,
            }
        )

    rep = pd.DataFrame(rows).sort_values(["dow", "bet_id"])
    out_csv = in_dir / args.out_csv
    rep.to_csv(out_csv, index=False)

    n = len(rep)
    n_with_stdout = int(np.isfinite(rep["stdout_proba"]).sum())
    match_rate = float(rep.loc[np.isfinite(rep["stdout_proba"]), "match6"].mean()) if n_with_stdout else float("nan")
    print(f"Report: {out_csv}")
    print(f"N payloads: {n} | com stdout: {n_with_stdout} | match6: {match_rate*100:.1f}%")
    if n_with_stdout:
        bad = rep[(np.isfinite(rep["stdout_proba"])) & (rep["match6"] == 0)]
        if not bad.empty:
            print("Primeiras divergências:")
            print(bad.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

