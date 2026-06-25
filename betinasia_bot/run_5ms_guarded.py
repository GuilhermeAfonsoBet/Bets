#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline guardado para 5Ms (pre/pos + com/sem World Cup).

Objetivo: execução robusta e auditável, reduzindo erros operacionais.

Modos:
- doctor: valida pré-requisitos (DB, heartbeat, balance, scripts)
- run: executa regen (opcional), escolhe base, aplica asserts, roda 5Ms e gera manifesto
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import glob
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


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


def _run(cmd: Sequence[str], *, timeout_sec: Optional[int] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(cmd),
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
    )


def _run_stream(cmd: Sequence[str], *, timeout_sec: Optional[int] = None) -> int:
    p = subprocess.Popen(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert p.stdout is not None
    start = datetime.now(timezone.utc)
    while True:
        line = p.stdout.readline()
        if line:
            print(line.rstrip("\n"))
        if p.poll() is not None:
            break
        if timeout_sec is not None:
            if (datetime.now(timezone.utc) - start).total_seconds() > timeout_sec:
                p.kill()
                return 124
    return int(p.returncode or 0)


def _latest_balance_csv() -> Optional[Path]:
    cands = sorted(glob.glob("/home/betbot/Bets/betinasia_bot/logs/accounting/*balance*.csv"), key=os.path.getmtime)
    if not cands:
        cands = sorted(glob.glob("/workspace/betinasia_bot/logs/accounting/*balance*.csv"), key=os.path.getmtime)
    return Path(cands[-1]) if cands else None


def _max_post_date(balance_csv: Path) -> Optional[datetime]:
    with balance_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        rd = csv.DictReader(f)
        fields = list(rd.fieldnames or [])
        col = _pick_col(fields, ["post date", "post_date", "created_at", "created at", "timestamp"])
        if not col:
            return None
        mx = None
        for r in rd:
            dt = _pdt(r.get(col, ""))
            if dt and (mx is None or dt > mx):
                mx = dt
        return mx


@dataclass
class CsvProfile:
    path: Path
    rows: int
    pre: int
    pos: int
    min_ts: Optional[datetime]
    max_ts: Optional[datetime]
    ts_col: Optional[str]


def _profile_csv(path: Path, split_dt: datetime) -> Optional[CsvProfile]:
    if not path.exists() or not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            rd = csv.DictReader(f)
            fields = list(rd.fieldnames or [])
            ts_col = _pick_col(fields, ["audited_at", "timestamp", "created_at", "executed_at", "updated_at", "date_utc"])
            if not ts_col:
                return None
            rows = 0
            pre = 0
            pos = 0
            mn = None
            mx = None
            for r in rd:
                dt = _pdt(r.get(ts_col, ""))
                if dt is None:
                    continue
                rows += 1
                if mn is None or dt < mn:
                    mn = dt
                if mx is None or dt > mx:
                    mx = dt
                if dt < split_dt:
                    pre += 1
                else:
                    pos += 1
            return CsvProfile(path=path, rows=rows, pre=pre, pos=pos, min_ts=mn, max_ts=mx, ts_col=ts_col)
    except Exception:
        return None


def _discover_bases() -> List[Path]:
    pats = [
        "/tmp/projecao_por_aposta_enriquecido__regen_db_*.csv",
        "/tmp/base_5ms_real_ate_*inferred*.csv",
        "/tmp/projecao_por_aposta_enriquecido.csv",
    ]
    out = []
    seen = set()
    for pat in pats:
        for p in glob.glob(pat):
            rp = str(Path(p).resolve())
            if rp in seen:
                continue
            seen.add(rp)
            out.append(Path(p))
    return out


def _doctor() -> int:
    ok = True
    print("=== DOCTOR ===")
    db = os.environ.get("DATABASE_URL", "").strip()
    if not db:
        print("[ERRO] DATABASE_URL nao definido.")
        return 2

    hb = _run(
        [
            "psql",
            db,
            "-At",
            "-c",
            "SELECT (SELECT max(audited_at) FROM betslip_audit_results), (SELECT max(created_at) FROM executor_bridge_seen);",
        ]
    )
    if hb.returncode != 0:
        ok = False
        print("[ERRO] psql heartbeat falhou:", hb.stderr.strip())
    else:
        print("[OK] DB heartbeat:", hb.stdout.strip())

    bal = _latest_balance_csv()
    if not bal:
        ok = False
        print("[ERRO] Nenhum balance CSV encontrado.")
    else:
        mx = _max_post_date(bal)
        print(f"[OK] latest_balance={bal}")
        print(f"[OK] max_post_date={mx}")

    must = [
        Path("betinasia_bot/regen_proj_from_balance_or_db.py"),
        Path("betinasia_bot/build_5ms_base_bridge_exec.py"),
        Path("betinasia_bot/diagnose_5ms_chain_coverage.py"),
        Path("betinasia_bot/run_5ms_prepos_worldcup_fast.py"),
        Path("betinasia_bot/run_roi_backpre_safe.py"),
        Path("betinasia_bot/ms_por_segmento_completo.py"),
    ]
    for p in must:
        if p.exists():
            print(f"[OK] script={p}")
        else:
            ok = False
            print(f"[ERRO] faltando script={p}")
    return 0 if ok else 3


def _profile_check(chosen: CsvProfile, end_ts: datetime, args: argparse.Namespace) -> tuple[bool, float, list[str]]:
    reasons: list[str] = []
    if chosen.rows < int(args.min_rows):
        reasons.append(f"rows={chosen.rows} < min_rows={args.min_rows}")
    if chosen.pre < int(args.min_pre):
        reasons.append(f"pre={chosen.pre} < min_pre={args.min_pre}")
    if chosen.pos < int(args.min_pos):
        reasons.append(f"pos={chosen.pos} < min_pos={args.min_pos}")
    if chosen.max_ts is None:
        reasons.append("max_ts ausente")
        return False, float("inf"), reasons
    lag_h = (end_ts - chosen.max_ts).total_seconds() / 3600.0
    if lag_h > float(args.max_lag_hours):
        reasons.append(f"lag_h={lag_h:.2f} > max_lag_hours={args.max_lag_hours}")
    return len(reasons) == 0, lag_h, reasons


def _run_pipeline(args: argparse.Namespace) -> int:
    lock_path = Path(args.lock_file)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"[ERRO] lock ativo: {lock_path}")
        return 10
    lock_file.write(str(os.getpid()))
    lock_file.flush()

    split_dt = _pdt(args.split_date)
    if split_dt is None:
        print(f"[ERRO] split-date invalido: {args.split_date}")
        return 11

    bal = _latest_balance_csv()
    if not bal:
        print("[ERRO] balance CSV nao encontrado.")
        return 12
    end_ts = _max_post_date(bal)
    if end_ts is None:
        print("[ERRO] nao consegui determinar max_post_date do balance.")
        return 13
    print(f"[INFO] latest_balance={bal}")
    print(f"[INFO] end_ts_real={end_ts.isoformat()}")

    chosen: Optional[CsvProfile] = None
    used_bridge_fallback = False
    bridge_base_path: Optional[str] = None
    if str(args.input_csv or "").strip():
        pr = _profile_csv(Path(args.input_csv), split_dt)
        if pr is None:
            print(f"[ERRO] input-csv invalido/inlegivel: {args.input_csv}")
            return 14
        chosen = pr
    else:
        if int(args.regen_first) == 1:
            print("[INFO] executando regen...")
            regen_cmd = [
                "python3",
                "-u",
                "betinasia_bot/regen_proj_from_balance_or_db.py",
                "--balance-csv",
                str(bal),
                "--max-depth",
                str(args.max_depth),
                "--write-default",
                "0",
            ]
            for r in args.search_root:
                rr = str(r or "").strip()
                if rr:
                    regen_cmd += ["--search-root", rr]
            rc = _run_stream(regen_cmd, timeout_sec=int(args.regen_timeout_sec))
            if rc != 0:
                print(f"[WARN] regen terminou com rc={rc}")

        profiles = [p for p in (_profile_csv(x, split_dt) for x in _discover_bases()) if p is not None]
        if not profiles:
            print("[ERRO] nenhum CSV base elegivel encontrado.")
            return 15
        profiles.sort(
            key=lambda p: (
                int(p.max_ts is not None and p.max_ts >= end_ts),
                p.rows,
                min(p.pre, p.pos),
                p.path.stat().st_mtime,
            ),
            reverse=True,
        )
        chosen = profiles[0]

    assert chosen is not None
    print(
        f"[INFO] base={chosen.path} rows={chosen.rows} pre={chosen.pre} pos={chosen.pos} "
        f"period={chosen.min_ts}->{chosen.max_ts}"
    )

    ok_profile, lag_h, reasons = _profile_check(chosen, end_ts, args)

    # fallback estrutural: audit->exec->order->ledger
    if (not ok_profile) and str(args.input_csv or "").strip() == "" and int(args.bridge_fallback) == 1:
        print(f"[WARN] base inicial reprovada: {'; '.join(reasons)}")
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        bridge_out = Path(f"/tmp/base_5ms_real_ate_{end_ts.strftime('%Y%m%d')}_inferred_bridge_{ts}.csv")
        bridge_cmd = [
            "python3",
            "betinasia_bot/build_5ms_base_bridge_exec.py",
            "--start-ts",
            str(args.start_ts),
            "--end-ts",
            end_ts.isoformat(),
            "--balance-csv",
            str(bal),
            "--max-log-files",
            str(args.bridge_max_log_files),
            "--out-csv",
            str(bridge_out),
        ]
        for r in args.executor_root:
            rr = str(r or "").strip()
            if rr:
                bridge_cmd += ["--executor-root", rr]
        print("[INFO] executando fallback bridge/exec/order ...")
        rc_fb = _run_stream(bridge_cmd, timeout_sec=int(args.bridge_timeout_sec))
        if rc_fb == 0 and bridge_out.exists():
            pr_fb = _profile_csv(bridge_out, split_dt)
            if pr_fb is not None:
                chosen = pr_fb
                ok_profile, lag_h, reasons = _profile_check(chosen, end_ts, args)
                used_bridge_fallback = True
                bridge_base_path = str(bridge_out)
                print(
                    f"[INFO] base fallback={chosen.path} rows={chosen.rows} pre={chosen.pre} pos={chosen.pos} "
                    f"period={chosen.min_ts}->{chosen.max_ts}"
                )
        else:
            print(f"[WARN] fallback bridge retornou rc={rc_fb}")

    if not ok_profile:
        print(f"[ERRO] base reprovada apos tentativas: {'; '.join(reasons)}")
        return 24

    start = datetime.now(timezone.utc)
    run_cmd = [
        "python3",
        "betinasia_bot/run_5ms_prepos_worldcup_fast.py",
        "--input-csv",
        str(chosen.path),
        "--split-date",
        args.split_date,
        "--end-date",
        end_ts.isoformat(),
        "--bootstrap-iters",
        str(args.bootstrap_iters),
        "--perm-iters",
        str(args.perm_iters),
        "--seed",
        str(args.seed),
        "--world-cup-aliases",
        args.world_cup_aliases,
        "--run-ms-segment",
        "1" if int(args.run_ms_segment) == 1 else "0",
        "--regen-if-needed",
        "0",
    ]
    rc = _run_stream(run_cmd, timeout_sec=int(args.pipeline_timeout_sec))
    if rc != 0:
        print(f"[ERRO] pipeline 5ms retornou rc={rc}")
        return 30

    # encontrar artefatos novos
    def latest_new(pat: str) -> Optional[str]:
        cands = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
        for p in cands:
            mt = datetime.fromtimestamp(os.path.getmtime(p), tz=timezone.utc)
            if mt >= start - timedelta(seconds=5):
                return p
        return cands[0] if cands else None

    roi_json = latest_new("/tmp/roi_backpre_robusto_*.json")
    ms_json = latest_new("/tmp/ms_por_segmento_*.json")
    ms_md = latest_new("/tmp/ms_por_segmento_*.md")
    ms_pdf = latest_new("/tmp/ms_por_segmento_*.pdf")

    manifest = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "selected_base": str(chosen.path),
        "selected_base_profile": {
            "rows": chosen.rows,
            "pre": chosen.pre,
            "pos": chosen.pos,
            "min_ts": chosen.min_ts.isoformat() if chosen.min_ts else None,
            "max_ts": chosen.max_ts.isoformat() if chosen.max_ts else None,
            "ts_col": chosen.ts_col,
        },
        "balance_csv": str(bal),
        "balance_max_post_date": end_ts.isoformat(),
        "asserts": {
            "min_rows": int(args.min_rows),
            "min_pre": int(args.min_pre),
            "min_pos": int(args.min_pos),
            "max_lag_hours": float(args.max_lag_hours),
            "lag_hours_observed": lag_h,
        },
        "fallback": {
            "bridge_fallback_enabled": int(args.bridge_fallback) == 1,
            "bridge_fallback_used": used_bridge_fallback,
            "bridge_base_path": bridge_base_path,
        },
        "artifacts": {
            "roi_json": roi_json,
            "ms_json": ms_json,
            "ms_md": ms_md,
            "ms_pdf": ms_pdf,
        },
        "status": "ok",
    }
    out_manifest = Path(args.manifest_out)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] manifest={out_manifest}")
    print(f"[OK] artifacts={manifest['artifacts']}")
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Runner guardado para 5Ms")
    ap.add_argument("--mode", choices=["doctor", "run"], default="run")
    ap.add_argument("--input-csv", default="")
    ap.add_argument("--split-date", default="2026-05-25T00:00:00+00:00")
    ap.add_argument("--start-ts", default="2026-01-01T00:00:00+00:00")
    ap.add_argument("--regen-first", type=int, default=1)
    ap.add_argument("--regen-timeout-sec", type=int, default=900)
    ap.add_argument("--bridge-fallback", type=int, default=1)
    ap.add_argument("--bridge-timeout-sec", type=int, default=1800)
    ap.add_argument("--bridge-max-log-files", type=int, default=1500)
    ap.add_argument("--pipeline-timeout-sec", type=int, default=1800)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument(
        "--search-root",
        action="append",
        default=["/home/betbot/Bets/betinasia_bot/logs/accounting"],
    )
    ap.add_argument(
        "--executor-root",
        action="append",
        default=[
            "/home/betbot/Bets/betinasia_bot/logs",
            "/workspace/betinasia_bot/logs",
        ],
    )
    ap.add_argument("--min-rows", type=int, default=120)
    ap.add_argument("--min-pre", type=int, default=40)
    ap.add_argument("--min-pos", type=int, default=40)
    ap.add_argument("--max-lag-hours", type=float, default=36.0)
    ap.add_argument("--bootstrap-iters", type=int, default=10000)
    ap.add_argument("--perm-iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--run-ms-segment", type=int, default=1)
    ap.add_argument(
        "--world-cup-aliases",
        default="FIFA World Cup,World Cup,Copa do Mundo,FIFA Club World Cup,Club World Cup,Mundial de Clubes",
    )
    ap.add_argument("--lock-file", default="/tmp/run_5ms_guarded.lock")
    ap.add_argument("--manifest-out", default=f"/tmp/run_5ms_guarded_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "doctor":
        return _doctor()
    return _run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())

