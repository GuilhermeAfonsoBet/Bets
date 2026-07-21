#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orquestrador "one-shot" para 5Ms:
- escolhe (ou regenera) base de resultado robusta
- valida cobertura pre/pos split-date
- roda avaliacao 5Ms com e sem World Cup (aliases)
- gera artefatos em /tmp

Objetivo: evitar "patinar" na montagem da base e reduzir ciclos manuais.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


def _pdt(v: str) -> Optional[datetime]:
    s = str(v or "").strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        d = datetime.fromisoformat(s)
    except Exception:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc)


def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).strip().lower() if ch.isalnum())


def _pick_col(fields: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    by = {_norm(f): f for f in fields}
    for c in candidates:
        got = by.get(_norm(c))
        if got:
            return got
    return None


@dataclass
class Profile:
    path: Path
    rows_total: int
    rows_window: int
    rows_pre: int
    rows_pos: int
    min_ts: Optional[datetime]
    max_ts: Optional[datetime]
    ts_col: Optional[str]


def profile_csv(path: Path, split_dt: datetime, end_dt: Optional[datetime]) -> Optional[Profile]:
    if not path.exists() or not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            rd = csv.DictReader(f)
            fields = list(rd.fieldnames or [])
            if not fields:
                return None
            ts_col = _pick_col(
                fields,
                ["audited_at", "timestamp", "created_at", "executed_at", "updated_at", "post date", "post_date", "date_utc"],
            )
            if not ts_col:
                return None
            total = 0
            pre = 0
            pos = 0
            mn = None
            mx = None
            for row in rd:
                total += 1
                dt = _pdt(row.get(ts_col, ""))
                if dt is None:
                    continue
                if mn is None or dt < mn:
                    mn = dt
                if mx is None or dt > mx:
                    mx = dt
                if end_dt is not None and dt > end_dt:
                    continue
                if dt < split_dt:
                    pre += 1
                else:
                    pos += 1
            return Profile(
                path=path,
                rows_total=total,
                rows_window=pre + pos,
                rows_pre=pre,
                rows_pos=pos,
                min_ts=mn,
                max_ts=mx,
                ts_col=ts_col,
            )
    except Exception:
        return None


def discover_candidates(extra_patterns: Iterable[str]) -> List[Path]:
    pats = [
        "/tmp/base_5ms_real_ate_*inferred*.csv",
        "/tmp/projecao_por_aposta_enriquecido.csv",
        "/tmp/projecao_por_aposta_enriquecido__regen_db_*.csv",
        "/tmp/base_5ms_from_bets_*.csv",
    ]
    pats.extend(list(extra_patterns))
    out: List[Path] = []
    seen = set()
    for pat in pats:
        for p in glob.glob(str(pat)):
            pp = Path(p)
            sp = str(pp.resolve())
            if sp in seen:
                continue
            seen.add(sp)
            out.append(pp)
    return out


def run_cmd(cmd: Sequence[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(list(cmd), check=True)


def maybe_regen_base(search_roots: Sequence[str], max_depth: int) -> None:
    cmd = [
        "python3",
        "betinasia_bot/regen_proj_from_balance_or_db.py",
        "--max-depth",
        str(max_depth),
        "--write-default",
        "0",
    ]
    for r in search_roots:
        rr = str(r or "").strip()
        if rr:
            cmd += ["--search-root", rr]
    run_cmd(cmd)


def choose_best(
    profiles: Sequence[Profile],
    split_dt: datetime,
    end_dt: Optional[datetime],
    min_pre: int,
    min_pos: int,
) -> Optional[Profile]:
    if not profiles:
        return None

    def score(p: Profile) -> Tuple[int, int, int, int, float]:
        max_covers_end = int(end_dt is not None and p.max_ts is not None and p.max_ts >= end_dt)
        min_ok = int(p.rows_pre >= min_pre and p.rows_pos >= min_pos)
        # Prioriza cobertura de janela e equilíbrio entre pre/pos.
        balance = min(p.rows_pre, p.rows_pos)
        mtime = p.path.stat().st_mtime if p.path.exists() else 0.0
        return (max_covers_end, min_ok, p.rows_window, balance, mtime)

    ranked = sorted(profiles, key=score, reverse=True)
    return ranked[0]


def fmt_dt(d: Optional[datetime]) -> str:
    return d.isoformat() if d else "NA"


def main() -> int:
    ap = argparse.ArgumentParser(description="Orquestra base + 5Ms pre/pos + World Cup em um comando")
    ap.add_argument("--input-csv", default="", help="Forca CSV de entrada (pula selecao automatica)")
    ap.add_argument("--split-date", default="2026-05-25T00:00:00+00:00")
    ap.add_argument("--end-date", default="", help="Limite superior opcional para cobertura da base")
    ap.add_argument("--min-pre", type=int, default=120, help="Minimo de linhas pre para considerar base adequada")
    ap.add_argument("--min-pos", type=int, default=60, help="Minimo de linhas pos para considerar base adequada")
    ap.add_argument("--regen-if-needed", type=int, default=1, help="Se 1, roda regen_proj quando nenhuma base adequada")
    ap.add_argument(
        "--search-root",
        action="append",
        default=[
            "/home/betbot/Bets/betinasia_bot/logs/accounting",
            "/workspace/betinasia_bot/logs/accounting",
        ],
        help="Raiz extra para busca de balance CSV na regen",
    )
    ap.add_argument("--max-depth", type=int, default=7)
    ap.add_argument("--bootstrap-iters", type=int, default=10000)
    ap.add_argument("--perm-iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--world-cup-aliases",
        default="FIFA World Cup,World Cup,Copa do Mundo,FIFA Club World Cup,Club World Cup,Mundial de Clubes",
    )
    ap.add_argument("--run-ms-segment", type=int, default=1, help="Se 1, roda ms_por_segmento_completo.py tambem")
    args = ap.parse_args()

    split_dt = _pdt(args.split_date)
    if split_dt is None:
        raise RuntimeError(f"split-date invalido: {args.split_date}")
    end_dt = _pdt(args.end_date) if str(args.end_date or "").strip() else None

    chosen: Optional[Profile] = None

    if str(args.input_csv or "").strip():
        p = Path(str(args.input_csv).strip())
        pr = profile_csv(p, split_dt, end_dt)
        if pr is None:
            raise RuntimeError(f"input-csv invalido/inlegivel: {p}")
        chosen = pr
        print("[INFO] CSV forçado selecionado.")
    else:
        cands = discover_candidates([])
        profiles = [pr for pr in (profile_csv(c, split_dt, end_dt) for c in cands) if pr is not None]
        print("[INFO] candidatos encontrados:", len(profiles))
        for p in sorted(profiles, key=lambda x: x.path.stat().st_mtime if x.path.exists() else 0.0, reverse=True)[:20]:
            print(
                f"  - {p.path} | window={p.rows_window} pre={p.rows_pre} pos={p.rows_pos} "
                f"period={fmt_dt(p.min_ts)}->{fmt_dt(p.max_ts)} ts_col={p.ts_col}"
            )
        chosen = choose_best(profiles, split_dt, end_dt, int(args.min_pre), int(args.min_pos))
        if chosen and not (chosen.rows_pre >= int(args.min_pre) and chosen.rows_pos >= int(args.min_pos)):
            chosen = None

        if chosen is None and int(args.regen_if_needed) == 1:
            print("[WARN] Nenhuma base adequada; executando regen_proj_from_balance_or_db.py ...")
            maybe_regen_base(args.search_root, int(args.max_depth))
            cands2 = discover_candidates([])
            profiles2 = [pr for pr in (profile_csv(c, split_dt, end_dt) for c in cands2) if pr is not None]
            chosen = choose_best(profiles2, split_dt, end_dt, int(args.min_pre), int(args.min_pos))

    if chosen is None:
        raise RuntimeError("Nao foi possivel selecionar/gerar base adequada para pre/pos.")

    print(
        f"[OK] base_escolhida={chosen.path} | window={chosen.rows_window} pre={chosen.rows_pre} pos={chosen.rows_pos} "
        f"| period={fmt_dt(chosen.min_ts)}->{fmt_dt(chosen.max_ts)}"
    )

    # 1) run_roi_backpre_safe -> avaliar_roi_backpre_slipneg_robusto (com segmentacao)
    roi_cmd = [
        "python3",
        "betinasia_bot/run_roi_backpre_safe.py",
        "--input-csv",
        str(chosen.path),
        "--bootstrap-iters",
        str(args.bootstrap_iters),
        "--perm-iters",
        str(args.perm_iters),
        "--seed",
        str(args.seed),
        "--split-date",
        str(args.split_date),
        "--split-world-cup",
        "1",
        "--world-cup-aliases",
        str(args.world_cup_aliases),
    ]
    run_cmd(roi_cmd)

    # 2) opcional: relatorio Ms por segmento completo
    if int(args.run_ms_segment) == 1:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_json = Path(f"/tmp/ms_por_segmento_{ts}.json")
        out_md = Path(f"/tmp/ms_por_segmento_{ts}.md")
        out_pdf = Path(f"/tmp/ms_por_segmento_{ts}.pdf")
        ms_cmd = [
            "python3",
            "betinasia_bot/ms_por_segmento_completo.py",
            "--input-csv",
            str(chosen.path),
            "--split-date",
            str(args.split_date),
            "--bootstrap-iters",
            str(args.bootstrap_iters),
            "--perm-iters",
            str(args.perm_iters),
            "--seed",
            str(args.seed),
            "--world-cup-aliases",
            str(args.world_cup_aliases),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
            "--out-pdf",
            str(out_pdf),
            "--skip-pdf-if-missing",
            "1",
        ]
        run_cmd(ms_cmd)
        print(f"[OK] ms_segment_json={out_json}")
        print(f"[OK] ms_segment_md={out_md}")
        if out_pdf.exists():
            print(f"[OK] ms_segment_pdf={out_pdf}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

