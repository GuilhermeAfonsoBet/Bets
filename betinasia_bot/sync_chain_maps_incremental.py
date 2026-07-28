#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sincroniza mapas persistentes da cadeia de reconciliacao:
- audit_id -> execution_id
- execution_id -> order_id

Objetivo:
1) reduzir dependencia de janela curta de logs
2) permitir backfill incremental/diario sem perder mapeamentos historicos
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


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


def _norm_compact(s: str) -> str:
    return "".join(ch for ch in str(s or "").strip().lower() if ch.isalnum())


def _run(cmd: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(list(cmd), capture_output=True, text=True, check=False)


def _psql(db: str, sql: str, *, at: bool = True) -> str:
    cmd = ["psql", db, "-v", "ON_ERROR_STOP=1"]
    if at:
        cmd.append("-At")
    cmd += ["-c", sql]
    p = _run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"psql falhou: {p.stderr.strip()}")
    return p.stdout


def _qident(x: str) -> str:
    return '"' + str(x).replace('"', '""') + '"'


def _split_table_ref(table_ref: str, *, default_schema: str = "public") -> Tuple[str, str]:
    raw = str(table_ref or "").strip()
    if not raw:
        raise RuntimeError("table_ref vazio")
    if "." in raw:
        sc, tb = raw.split(".", 1)
        sc = sc.strip() or default_schema
        tb = tb.strip()
    else:
        sc = default_schema
        tb = raw
    if not tb:
        raise RuntimeError(f"table_ref invalido: {table_ref}")
    return sc, tb


def _pick(cols: Sequence[str], cands: Sequence[str]) -> Optional[str]:
    by = {_norm_compact(c): c for c in cols}
    for c in cands:
        hit = by.get(_norm_compact(c))
        if hit:
            return hit
    return None


def _table_cols(db: str, table: str) -> Tuple[str, List[str]]:
    sql = f"""
    SELECT table_schema, column_name
    FROM information_schema.columns
    WHERE table_name = '{table}'
      AND table_schema NOT IN ('pg_catalog','information_schema')
    ORDER BY CASE WHEN table_schema='public' THEN 0 ELSE 1 END, ordinal_position;
    """
    out = _psql(db, sql, at=True).strip()
    if not out:
        raise RuntimeError(f"tabela nao encontrada: {table}")
    schema = ""
    cols: List[str] = []
    for ln in out.splitlines():
        parts = ln.split("|")
        if len(parts) != 2:
            continue
        sc, c = parts
        if not schema:
            schema = sc
        cols.append(c)
    if not schema:
        raise RuntimeError(f"sem schema para tabela: {table}")
    return schema, cols


def _latest_balance_csv() -> Path:
    cands = sorted(glob.glob("/home/betbot/Bets/betinasia_bot/logs/accounting/*balance*.csv"), key=os.path.getmtime)
    if not cands:
        cands = sorted(glob.glob("/workspace/betinasia_bot/logs/accounting/*balance*.csv"), key=os.path.getmtime)
    if not cands:
        raise RuntimeError("nenhum balance csv encontrado")
    return Path(cands[-1])


def _max_post_date(balance_csv: Path) -> Optional[datetime]:
    with balance_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        rd = csv.DictReader(f)
        fields = list(rd.fieldnames or [])
        col = _pick(fields, ["post date", "post_date", "created_at", "created at", "timestamp"])
        if not col:
            return None
        mx = None
        for r in rd:
            dt = _pdt(r.get(col, ""))
            if dt and (mx is None or dt > mx):
                mx = dt
        return mx


def _read_database_url_from_env_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                s = ln.strip()
                if not s or s.startswith("#"):
                    continue
                if s.startswith("export "):
                    s = s[len("export ") :].strip()
                if not s.startswith("DATABASE_URL="):
                    continue
                v = s.split("=", 1)[1].strip()
                if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    v = v[1:-1].strip()
                return v
    except Exception:
        return ""
    return ""


def _resolve_database_url(cli_value: str) -> str:
    if str(cli_value or "").strip():
        return str(cli_value).strip()
    env = os.environ.get("DATABASE_URL", "").strip()
    if env:
        return env
    env_candidates = [
        Path.cwd() / "betinasia_bot/.env",
        Path.cwd() / ".env",
        Path("/home/betbot/Bets/betinasia_bot/.env"),
        Path("/home/betbot/Bets/.env"),
        Path("/workspace/betinasia_bot/.env"),
        Path("/workspace/.env"),
    ]
    for p in env_candidates:
        db = _read_database_url_from_env_file(p)
        if db:
            print(f"[INFO] DATABASE_URL carregado de {p}")
            return db
    return ""


def _ensure_tables(db: str, audit_exec_table: str, exec_order_table: str) -> None:
    sc1, tb1 = _split_table_ref(audit_exec_table)
    sc2, tb2 = _split_table_ref(exec_order_table)
    q1 = f"{_qident(sc1)}.{_qident(tb1)}"
    q2 = f"{_qident(sc2)}.{_qident(tb2)}"
    idx1 = f"idx_{tb1}_execution_id"
    idx2 = f"idx_{tb2}_order_id"
    sql = f"""
    CREATE TABLE IF NOT EXISTS {q1} (
      audit_id text PRIMARY KEY,
      execution_id text NOT NULL,
      last_seen_at timestamptz NULL,
      source text NOT NULL DEFAULT 'bridge',
      hit_count bigint NOT NULL DEFAULT 1,
      updated_at timestamptz NOT NULL DEFAULT now()
    );
    ALTER TABLE {q1} ADD COLUMN IF NOT EXISTS last_seen_at timestamptz NULL;
    ALTER TABLE {q1} ADD COLUMN IF NOT EXISTS source text;
    ALTER TABLE {q1} ADD COLUMN IF NOT EXISTS hit_count bigint;
    ALTER TABLE {q1} ADD COLUMN IF NOT EXISTS updated_at timestamptz;
    ALTER TABLE {q1} ALTER COLUMN source SET DEFAULT 'bridge';
    ALTER TABLE {q1} ALTER COLUMN hit_count SET DEFAULT 1;
    ALTER TABLE {q1} ALTER COLUMN updated_at SET DEFAULT now();
    UPDATE {q1}
       SET source = COALESCE(NULLIF(source,''), 'bridge'),
           hit_count = COALESCE(hit_count, 1),
           updated_at = COALESCE(updated_at, now())
     WHERE source IS NULL OR source = '' OR hit_count IS NULL OR updated_at IS NULL;
    CREATE INDEX IF NOT EXISTS {_qident(idx1)} ON {q1} (execution_id);

    CREATE TABLE IF NOT EXISTS {q2} (
      execution_id text PRIMARY KEY,
      order_id text NOT NULL,
      last_seen_at timestamptz NULL,
      source text NOT NULL DEFAULT 'logs',
      hit_count bigint NOT NULL DEFAULT 1,
      updated_at timestamptz NOT NULL DEFAULT now()
    );
    ALTER TABLE {q2} ADD COLUMN IF NOT EXISTS last_seen_at timestamptz NULL;
    ALTER TABLE {q2} ADD COLUMN IF NOT EXISTS source text;
    ALTER TABLE {q2} ADD COLUMN IF NOT EXISTS hit_count bigint;
    ALTER TABLE {q2} ADD COLUMN IF NOT EXISTS updated_at timestamptz;
    ALTER TABLE {q2} ALTER COLUMN source SET DEFAULT 'logs';
    ALTER TABLE {q2} ALTER COLUMN hit_count SET DEFAULT 1;
    ALTER TABLE {q2} ALTER COLUMN updated_at SET DEFAULT now();
    UPDATE {q2}
       SET source = COALESCE(NULLIF(source,''), 'logs'),
           hit_count = COALESCE(hit_count, 1),
           updated_at = COALESCE(updated_at, now())
     WHERE source IS NULL OR source = '' OR hit_count IS NULL OR updated_at IS NULL;
    CREATE INDEX IF NOT EXISTS {_qident(idx2)} ON {q2} (order_id);
    """
    _psql(db, sql, at=False)


def _export_bridge(db: str, out_csv: Path, start_ts: str, end_ts: str) -> int:
    schema, cols = _table_cols(db, "executor_bridge_seen")
    qtbl = f"{_qident(schema)}.{_qident('executor_bridge_seen')}"
    c_created = _pick(cols, ["created_at", "ts", "timestamp"])
    if not c_created:
        raise RuntimeError("executor_bridge_seen sem created_at")

    json_cols = [c for c in cols if _norm_compact(c) in {_norm_compact(x) for x in ["meta", "payload_json", "row_json", "payload", "message", "data"]}]
    expr_exec_terms = []
    expr_audit_terms = []
    c_ex = _pick(cols, ["execution_id"])
    c_aid = _pick(cols, ["audit_id"])
    c_src = _pick(cols, ["src_id", "source_id", "sourceid"])
    if c_ex:
        expr_exec_terms.append(f"NULLIF({_qident(c_ex)}::text,'')")
    if c_aid:
        expr_audit_terms.append(f"NULLIF({_qident(c_aid)}::text,'')")
    if c_src:
        expr_audit_terms.append(f"NULLIF({_qident(c_src)}::text,'')")

    for jc in json_cols:
        qj = _qident(jc)
        expr_exec_terms += [
            f"NULLIF({qj}::jsonb #>> '{{execution_id}}','')",
            f"NULLIF({qj}::jsonb #>> '{{executionId}}','')",
            f"NULLIF({qj}::jsonb #>> '{{exec_id}}','')",
            f"NULLIF({qj}::jsonb #>> '{{execution,id}}','')",
            f"NULLIF(jsonb_path_query_first({qj}::jsonb, '$.**.execution_id') #>> '{{}}','')",
            f"NULLIF(jsonb_path_query_first({qj}::jsonb, '$.**.executionId') #>> '{{}}','')",
            f"NULLIF(jsonb_path_query_first({qj}::jsonb, '$.**.exec_id') #>> '{{}}','')",
            f"NULLIF(jsonb_path_query_first({qj}::jsonb, '$.**.execution.id') #>> '{{}}','')",
        ]
        expr_audit_terms += [
            f"NULLIF({qj}::jsonb #>> '{{audit_id}}','')",
            f"NULLIF({qj}::jsonb #>> '{{auditId}}','')",
            f"NULLIF({qj}::jsonb #>> '{{audit,id}}','')",
            f"NULLIF({qj}::jsonb #>> '{{src_id}}','')",
            f"NULLIF({qj}::jsonb #>> '{{srcId}}','')",
            f"NULLIF({qj}::jsonb #>> '{{source_id}}','')",
            f"NULLIF({qj}::jsonb #>> '{{sourceId}}','')",
            f"NULLIF(jsonb_path_query_first({qj}::jsonb, '$.**.audit_id') #>> '{{}}','')",
            f"NULLIF(jsonb_path_query_first({qj}::jsonb, '$.**.auditId') #>> '{{}}','')",
            f"NULLIF(jsonb_path_query_first({qj}::jsonb, '$.**.audit.id') #>> '{{}}','')",
            f"NULLIF(jsonb_path_query_first({qj}::jsonb, '$.**.src_id') #>> '{{}}','')",
            f"NULLIF(jsonb_path_query_first({qj}::jsonb, '$.**.srcId') #>> '{{}}','')",
            f"NULLIF(jsonb_path_query_first({qj}::jsonb, '$.**.source_id') #>> '{{}}','')",
            f"NULLIF(jsonb_path_query_first({qj}::jsonb, '$.**.sourceId') #>> '{{}}','')",
        ]

    expr_exec = "COALESCE(" + ",".join(expr_exec_terms + ["NULL::text"]) + ")"
    expr_audit = "COALESCE(" + ",".join(expr_audit_terms + ["NULL::text"]) + ")"
    start2 = (_pdt(start_ts) - timedelta(days=2)).isoformat()
    end2 = (_pdt(end_ts) + timedelta(days=2)).isoformat()
    sql = f"""
    SELECT
      {expr_audit} AS audit_id,
      {expr_exec} AS execution_id,
      {_qident(c_created)}::text AS created_at
    FROM {qtbl}
    WHERE {_qident(c_created)} >= '{start2}'::timestamptz
      AND {_qident(c_created)} <= '{end2}'::timestamptz
      AND {expr_audit} IS NOT NULL
      AND {expr_exec} IS NOT NULL
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    dst = str(out_csv).replace("'", "''")
    stmt = f"\\copy ({sql}) TO '{dst}' WITH CSV HEADER"
    p = _run(["psql", db, "-v", "ON_ERROR_STOP=1", "-c", stmt])
    if p.returncode != 0:
        raise RuntimeError(f"psql copy falhou: {p.stderr.strip()}")
    with out_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def _bridge_to_rows(bridge_csv: Path) -> List[Dict[str, str]]:
    best: Dict[str, Tuple[datetime, str, int]] = {}
    pair_count: Dict[Tuple[str, str], int] = defaultdict(int)
    with bridge_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            aid = str(r.get("audit_id", "")).strip()
            ex = str(r.get("execution_id", "")).strip()
            dt = _pdt(r.get("created_at", ""))
            if not aid or not ex:
                continue
            if dt is None:
                dt = datetime.now(timezone.utc)
            pair_count[(aid, ex)] += 1
            prev = best.get(aid)
            if prev is None or dt > prev[0]:
                best[aid] = (dt, ex, 0)

    rows: List[Dict[str, str]] = []
    for aid, (dt, ex, _) in best.items():
        rows.append(
            {
                "audit_id": aid,
                "execution_id": ex,
                "last_seen_at": dt.isoformat(),
                "source": "bridge",
                "hit_count": str(pair_count.get((aid, ex), 1)),
            }
        )
    return rows


def _iter_lines(path: Path):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                yield ln
    else:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                yield ln


def _extract_ids(obj: object, exec_ids: set[str], order_ids: set[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            kn = _norm_compact(k)
            if kn in {"executionid", "execution_id", "execid", "exec_id"}:
                sv = str(v or "").strip()
                if sv:
                    exec_ids.add(sv)
            if kn in {"orderid", "order_id", "ticketid", "betid"}:
                sv = str(v or "").strip()
                if sv:
                    order_ids.add(sv)
            _extract_ids(v, exec_ids, order_ids)
    elif isinstance(obj, list):
        for it in obj:
            _extract_ids(it, exec_ids, order_ids)


def _build_exec_order_pairs_from_logs(log_roots: Sequence[str], max_files: int) -> Tuple[Dict[Tuple[str, str], int], int]:
    files: List[Path] = []
    for root in log_roots:
        rr = Path(str(root))
        if not rr.exists():
            continue
        patterns = [
            "*executor*jsonl",
            "*executor*jsonl.*",
            "*executor*ndjson*",
            "*executor*json*",
            "*executor*log*",
            "*executor_live*",
        ]
        for pat in patterns:
            files += list(rr.rglob(pat))
    uniq = {}
    for p in files:
        if not p.is_file():
            continue
        try:
            rp = str(p.resolve())
        except Exception:
            rp = str(p)
        uniq[rp] = p
    files2 = sorted(uniq.values(), key=lambda p: p.stat().st_mtime, reverse=True)
    if max_files > 0:
        files2 = files2[:max_files]
    print(f"[INFO] logs executor candidatos: {len(files2)}")

    pair_count: Dict[Tuple[str, str], int] = defaultdict(int)
    rx_exec = re.compile(r'"execution[_ ]?id"\s*:\s*"([^"]+)"', re.I)
    rx_order = re.compile(r'"order[_ ]?id"\s*:\s*"([^"]+)"', re.I)
    for i, p in enumerate(files2, start=1):
        if i % 100 == 0:
            print(f"[INFO] parsing logs: {i}/{len(files2)}")
        try:
            for ln in _iter_lines(p):
                ln = ln.strip()
                if not ln:
                    continue
                exec_ids: set[str] = set()
                order_ids: set[str] = set()
                try:
                    obj = json.loads(ln)
                    _extract_ids(obj, exec_ids, order_ids)
                except Exception:
                    for m in rx_exec.findall(ln):
                        if m:
                            exec_ids.add(m.strip())
                    for m in rx_order.findall(ln):
                        if m:
                            order_ids.add(m.strip())
                if not exec_ids or not order_ids:
                    continue
                for ex in exec_ids:
                    for od in order_ids:
                        if ex and od:
                            pair_count[(ex, od)] += 1
        except Exception:
            continue
    return pair_count, len(files2)


def _build_exec_order_pairs_from_balance_ref(balance_csv: Path) -> Dict[Tuple[str, str], int]:
    with balance_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        rd = csv.DictReader(f)
        fields = list(rd.fieldnames or [])
        oid_col = _pick(fields, ["order id", "order_id", "orderid", "bet id", "bet_id", "ticket id", "ticket_id"])
        ref_col = _pick(fields, ["transaction id", "transaction_id", "reference id", "reference_id", "ref_id", "execution_id", "execution id"])
        if not oid_col or not ref_col:
            return {}
        pair_count: Dict[Tuple[str, str], int] = defaultdict(int)
        for r in rd:
            od = str(r.get(oid_col, "")).strip()
            ex = str(r.get(ref_col, "")).strip()
            if ex and od:
                pair_count[(ex, od)] += 1
        return pair_count


def _choose_best_exec_order_rows(log_pairs: Dict[Tuple[str, str], int], bal_pairs: Dict[Tuple[str, str], int]) -> List[Dict[str, str]]:
    by_exec: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)
    for (ex, od), c in log_pairs.items():
        by_exec[ex].append((c, bal_pairs.get((ex, od), 0), od))
    for (ex, od), c in bal_pairs.items():
        if (ex, od) not in log_pairs:
            by_exec[ex].append((0, c, od))
    rows: List[Dict[str, str]] = []
    for ex, arr in by_exec.items():
        arr.sort(key=lambda t: (t[0] + t[1], t[0], t[1], len(t[2])), reverse=True)
        c_logs, c_bal, od = arr[0]
        if c_logs > 0 and c_bal > 0:
            src = "logs+balance_ref"
        elif c_logs > 0:
            src = "logs"
        else:
            src = "balance_ref"
        now_iso = datetime.now(timezone.utc).isoformat()
        rows.append(
            {
                "execution_id": ex,
                "order_id": od,
                "last_seen_at": now_iso,
                "source": src,
                "hit_count": str(c_logs + c_bal),
            }
        )
    return rows


def _upsert_audit_exec_rows(db: str, table_ref: str, rows: List[Dict[str, str]]) -> int:
    if not rows:
        return 0
    sc, tb = _split_table_ref(table_ref)
    qtbl = f"{_qident(sc)}.{_qident(tb)}"
    tmp = Path(f"/tmp/sync_audit_exec_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["audit_id", "execution_id", "last_seen_at", "source", "hit_count"])
        wr.writeheader()
        wr.writerows(rows)
    tmp_esc = str(tmp).replace("'", "''")
    create_tmp = "CREATE TEMP TABLE tmp_audit_exec (audit_id text, execution_id text, last_seen_at timestamptz, source text, hit_count bigint);"
    copy_tmp = f"\\copy tmp_audit_exec FROM '{tmp_esc}' WITH CSV HEADER"
    upsert = f"""
    UPDATE {qtbl} t
       SET execution_id = s.execution_id,
           last_seen_at = COALESCE(GREATEST(t.last_seen_at, s.last_seen_at), s.last_seen_at, t.last_seen_at),
           source = COALESCE(NULLIF(s.source,''), 'bridge'),
           hit_count = COALESCE(t.hit_count, 0) + COALESCE(s.hit_count, 1),
           updated_at = now()
      FROM tmp_audit_exec s
     WHERE t.audit_id = s.audit_id;

    INSERT INTO {qtbl} (audit_id, execution_id, last_seen_at, source, hit_count, updated_at)
    SELECT s.audit_id, s.execution_id, s.last_seen_at, COALESCE(NULLIF(s.source,''),'bridge'), COALESCE(s.hit_count,1), now()
      FROM tmp_audit_exec s
     WHERE NOT EXISTS (
       SELECT 1
         FROM {qtbl} t
        WHERE t.audit_id = s.audit_id
     );
    """
    p = _run(["psql", db, "-v", "ON_ERROR_STOP=1", "-c", create_tmp, "-c", copy_tmp, "-c", upsert])
    if p.returncode != 0:
        raise RuntimeError(f"upsert audit_exec falhou: {p.stderr.strip()}")
    return len(rows)


def _upsert_exec_order_rows(db: str, table_ref: str, rows: List[Dict[str, str]]) -> int:
    if not rows:
        return 0
    sc, tb = _split_table_ref(table_ref)
    qtbl = f"{_qident(sc)}.{_qident(tb)}"
    tmp = Path(f"/tmp/sync_exec_order_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["execution_id", "order_id", "last_seen_at", "source", "hit_count"])
        wr.writeheader()
        wr.writerows(rows)
    tmp_esc = str(tmp).replace("'", "''")
    create_tmp = "CREATE TEMP TABLE tmp_exec_order (execution_id text, order_id text, last_seen_at timestamptz, source text, hit_count bigint);"
    copy_tmp = f"\\copy tmp_exec_order FROM '{tmp_esc}' WITH CSV HEADER"
    upsert = f"""
    UPDATE {qtbl} t
       SET order_id = s.order_id,
           last_seen_at = COALESCE(GREATEST(t.last_seen_at, s.last_seen_at), s.last_seen_at, t.last_seen_at),
           source = COALESCE(NULLIF(s.source,''), 'logs'),
           hit_count = COALESCE(t.hit_count, 0) + COALESCE(s.hit_count, 1),
           updated_at = now()
      FROM tmp_exec_order s
     WHERE t.execution_id = s.execution_id;

    INSERT INTO {qtbl} (execution_id, order_id, last_seen_at, source, hit_count, updated_at)
    SELECT s.execution_id, s.order_id, s.last_seen_at, COALESCE(NULLIF(s.source,''),'logs'), COALESCE(s.hit_count,1), now()
      FROM tmp_exec_order s
     WHERE NOT EXISTS (
       SELECT 1
         FROM {qtbl} t
        WHERE t.execution_id = s.execution_id
     );
    """
    p = _run(["psql", db, "-v", "ON_ERROR_STOP=1", "-c", create_tmp, "-c", copy_tmp, "-c", upsert])
    if p.returncode != 0:
        raise RuntimeError(f"upsert exec_order falhou: {p.stderr.strip()}")
    return len(rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sincroniza mapas persistentes da cadeia 5Ms")
    ap.add_argument("--start-ts", default="2026-01-01T00:00:00+00:00")
    ap.add_argument("--end-ts", default="")
    ap.add_argument("--database-url", default="", help="Override do DATABASE_URL")
    ap.add_argument("--balance-csv", default="")
    ap.add_argument("--executor-root", action="append", default=["/home/betbot/Bets/betinasia_bot/logs", "/workspace/betinasia_bot/logs"])
    ap.add_argument("--max-log-files", type=int, default=20000)
    ap.add_argument("--audit-exec-map-table", default="public.audit_execution_map")
    ap.add_argument("--exec-order-map-table", default="public.execution_order_map")
    ap.add_argument("--skip-bridge", action="store_true")
    ap.add_argument("--skip-logs", action="store_true")
    ap.add_argument("--skip-balance-ref", action="store_true")
    ap.add_argument("--out-summary", default=f"/tmp/sync_chain_maps_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    db = _resolve_database_url(args.database_url)
    if not db:
        raise SystemExit("DATABASE_URL nao definido (exporte no shell, passe --database-url ou configure .env)")
    bal = Path(args.balance_csv).expanduser() if str(args.balance_csv or "").strip() else _latest_balance_csv()
    if not bal.exists():
        raise SystemExit(f"balance csv nao encontrado: {bal}")
    end_ts = str(args.end_ts or "").strip()
    if not end_ts:
        mx = _max_post_date(bal)
        if mx is None:
            end_ts = datetime.now(timezone.utc).isoformat()
        else:
            end_ts = mx.isoformat()

    print(f"[INFO] balance={bal}")
    print(f"[INFO] start_ts={args.start_ts} end_ts={end_ts}")
    print(f"[INFO] audit_exec_map_table={args.audit_exec_map_table}")
    print(f"[INFO] exec_order_map_table={args.exec_order_map_table}")

    _ensure_tables(db, args.audit_exec_map_table, args.exec_order_map_table)

    out = {
        "start_ts": args.start_ts,
        "end_ts": end_ts,
        "balance_csv": str(bal),
        "audit_exec_rows_bridge": 0,
        "audit_exec_rows_upserted": 0,
        "exec_order_pairs_logs": 0,
        "exec_order_pairs_balance_ref": 0,
        "exec_order_rows_upserted": 0,
        "executor_logs_scanned": 0,
    }

    if not args.skip_bridge:
        bridge_csv = Path(f"/tmp/sync_chain_bridge_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv")
        n_bridge = _export_bridge(db, bridge_csv, args.start_ts, end_ts)
        bridge_rows = _bridge_to_rows(bridge_csv)
        n_up = _upsert_audit_exec_rows(db, args.audit_exec_map_table, bridge_rows)
        out["audit_exec_rows_bridge"] = n_bridge
        out["audit_exec_rows_upserted"] = n_up
        print(f"[OK] bridge rows: {n_bridge}")
        print(f"[OK] audit->exec upserted: {n_up}")

    log_pairs: Dict[Tuple[str, str], int] = {}
    bal_pairs: Dict[Tuple[str, str], int] = {}
    if not args.skip_logs:
        log_pairs, n_logs = _build_exec_order_pairs_from_logs(args.executor_root, int(args.max_log_files))
        out["exec_order_pairs_logs"] = len(log_pairs)
        out["executor_logs_scanned"] = n_logs
        print(f"[OK] exec->order pairs (logs): {len(log_pairs)}")
    if not args.skip_balance_ref:
        bal_pairs = _build_exec_order_pairs_from_balance_ref(bal)
        out["exec_order_pairs_balance_ref"] = len(bal_pairs)
        print(f"[OK] exec->order pairs (balance_ref): {len(bal_pairs)}")

    rows_ex_order = _choose_best_exec_order_rows(log_pairs, bal_pairs)
    n_up_ex = _upsert_exec_order_rows(db, args.exec_order_map_table, rows_ex_order)
    out["exec_order_rows_upserted"] = n_up_ex
    print(f"[OK] exec->order upserted: {n_up_ex}")

    out_path = Path(args.out_summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] summary_json: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

