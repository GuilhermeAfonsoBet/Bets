#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runner robusto para avaliacao ROI Back Pre (slippage < 0).

Objetivo: minimizar erros operacionais de CLI
- escolhe automaticamente um CSV enriquecido nao-vazio em /tmp
- detecta colunas existentes por sinonimos
- só passa overrides de colunas que realmente existem
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).strip().lower() if ch.isalnum())


def _pick_col(fieldnames: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    by_norm = {_norm(f): f for f in fieldnames}
    for c in candidates:
        hit = by_norm.get(_norm(c))
        if hit:
            return hit
    return None


def count_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rd = csv.DictReader(f)
            return sum(1 for _ in rd)
    except Exception:
        return -1


def choose_input_csv(explicit: str) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise RuntimeError(f"--input-csv nao existe: {p}")
        n = count_rows(p)
        if n <= 0:
            raise RuntimeError(f"--input-csv sem linhas validas: {p}")
        return p

    cands: List[Path] = list(Path("/tmp").glob("projecao_por_aposta_enriquecido__regen_db_*.csv"))
    p0 = Path("/tmp/projecao_por_aposta_enriquecido.csv")
    if p0.exists():
        cands.append(p0)
    if not cands:
        raise RuntimeError("Nenhum CSV enriquecido encontrado em /tmp.")

    ranked: List[Tuple[Path, int, float]] = []
    for p in cands:
        ranked.append((p, count_rows(p), p.stat().st_mtime))
    ranked.sort(key=lambda t: t[2], reverse=True)

    print("[DIAG] candidatos (mais novo -> mais antigo):")
    for p, n, _ in ranked[:10]:
        print(f"  - {p} | rows={n}")

    valid = [p for p, n, _ in ranked if n > 0]
    if not valid:
        raise RuntimeError("Todos os CSVs candidatos estao vazios.")
    return valid[0]


def get_headers(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        return list(rd.fieldnames or [])


def main() -> int:
    ap = argparse.ArgumentParser(description="Runner seguro para avaliacao ROI Back Pre")
    ap.add_argument("--input-csv", default="", help="CSV enriquecido explicito (opcional)")
    ap.add_argument("--bootstrap-iters", type=int, default=10000)
    ap.add_argument("--perm-iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    in_csv = choose_input_csv(args.input_csv)
    fields = get_headers(in_csv)
    if not fields:
        raise RuntimeError(f"CSV sem cabecalho: {in_csv}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_json = Path(f"/tmp/roi_backpre_robusto_{ts}.json")
    out_md = Path(f"/tmp/roi_backpre_robusto_{ts}.md")

    mapping: Dict[str, Optional[str]] = {
        "event": _pick_col(fields, ["event_id", "match_id", "fixture_id", "game_id", "order_id", "audit_id", "id"]),
        "league": _pick_col(fields, ["league", "league_name", "competition", "tournament"]),
        "timestamp": _pick_col(fields, ["audited_at", "timestamp", "created_at", "updated_at", "date_utc"]),
        "stake": _pick_col(fields, ["stake_real", "stake", "exposure_real", "exposure", "stake_liq", "stake_liquidado"]),
        "pnl": _pick_col(fields, ["pnl_real", "pnl", "profit", "pl", "result"]),
        "roi": _pick_col(fields, ["roi_real_pct", "roi_pct", "roi"]),
        "slippage": _pick_col(fields, ["slippage_pre_pct", "slippage_raw_pct", "slippage"]),
        "side": _pick_col(fields, ["side", "exec_side", "direction"]),
        "regime": _pick_col(fields, ["regime", "market_regime", "market_phase", "phase", "is_live"]),
    }

    cmd: List[str] = [
        "python3",
        "betinasia_bot/avaliar_roi_backpre_slipneg_robusto.py",
        "--input-csv",
        str(in_csv),
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
        "--bootstrap-iters",
        str(args.bootstrap_iters),
        "--perm-iters",
        str(args.perm_iters),
        "--seed",
        str(args.seed),
        "--topk",
        "1,3,5,10",
    ]

    if mapping["event"]:
        cmd += ["--event-col", mapping["event"]]
    if mapping["league"]:
        cmd += ["--league-col", mapping["league"]]
    if mapping["timestamp"]:
        cmd += ["--timestamp-col", mapping["timestamp"]]
    if mapping["stake"]:
        cmd += ["--stake-col", mapping["stake"]]
    else:
        cmd += ["--allow-unit-stake-fallback", "1"]
    if mapping["pnl"]:
        cmd += ["--pnl-col", mapping["pnl"]]
    elif mapping["roi"]:
        cmd += ["--roi-col", mapping["roi"]]
    if mapping["slippage"]:
        cmd += ["--slippage-col", mapping["slippage"]]
    else:
        cmd += ["--no-enforce-slip-neg"]
    if mapping["side"]:
        cmd += ["--side-col", mapping["side"]]
    if mapping["regime"]:
        cmd += ["--regime-col", mapping["regime"]]

    print(f"[CSV] {in_csv}")
    print(f"[MAP] {mapping}")
    print("[RUN] " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[OK] JSON: {out_json}")
    print(f"[OK] MD:   {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

