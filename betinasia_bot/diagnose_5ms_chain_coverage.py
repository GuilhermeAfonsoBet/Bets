#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostico forense da cadeia de reconciliacao para base 5Ms:
audit -> execution_id (bridge) -> order_id (logs executor) -> pnl (balance)

Saidas:
- CSV por dia com cobertura em cada etapa
- JSON resumo com gargalo identificado
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
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


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


def _pf(v: object) -> Optional[float]:
    s = str(v or "").strip().replace(" ", "")
    if not s:
        return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        x = float(s)
    except Exception:
        return None
    if x != x or x in (float("inf"), float("-inf")):
        return None
    return x


def _norm(s: str) -> str:
    return "".join(ch for ch in str(s or "").strip().lower() if ch.isalnum())


def _qident(x: str) -> str:
    return '"' + str(x).replace('"', '""') + '"'


def _pick(cols: Sequence[str], cands: Sequence[str]) -> Optional[str]:
    by = {_norm(c): c for c in cols}
    for c in cands:
        hit = by.get(_norm(c))
        if hit:
            return hit
    return None


def _run(cmd: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(list(cmd), text=True, capture_output=True, check=False)


def _psql(db: str, sql: str, *, at: bool = True) -> str:
    cmd = ["psql", db, "-v", "ON_ERROR_STOP=1"]
    if at:
        cmd.append("-At")
    cmd += ["-c", sql]
    p = _run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"psql failed: {p.stderr.strip()}")
    return p.stdout


def _psql_copy(db: str, query: str, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    dst = str(out_csv).replace("'", "''")
    stmt = f"\\copy ({query}) TO '{dst}' WITH CSV HEADER"
    p = _run(["psql", db, "-v", "ON_ERROR_STOP=1", "-c", stmt])
    if p.returncode != 0:
        raise RuntimeError(f"psql copy failed: {p.stderr.strip()}")


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
        raise RuntimeError(f"table not found: {table}")
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
        raise RuntimeError(f"no schema for table {table}")
    return schema, cols


def _latest_balance_csv() -> Path:
    cands = sorted(glob.glob("/home/betbot/Bets/betinasia_bot/logs/accounting/*balance*.csv"), key=os.path.getmtime)
    if not cands:
        cands = sorted(glob.glob("/workspace/betinasia_bot/logs/accounting/*balance*.csv"), key=os.path.getmtime)
    if not cands:
        raise RuntimeError("no balance csv found")
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


def _build_order_pnl_keys(balance_csv: Path) -> set[str]:
    with balance_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        rd = csv.DictReader(f)
        fields = list(rd.fieldnames or [])
        oid_col = _pick(fields, ["order id", "order_id", "orderid", "bet id", "bet_id", "ticket id", "ticket_id"])
        val_col = _pick(fields, ["amount", "pnl_real", "pnl", "profit", "pl", "result", "net"])
        if not oid_col or not val_col:
            return set()
        out = set()
        for r in rd:
            oid = str(r.get(oid_col, "")).strip()
            if not oid:
                continue
            v = _pf(r.get(val_col))
            if v is None:
                continue
            out.add(oid)
            oc = _norm(oid)
            if oc:
                out.add(oc)
            od = "".join(ch for ch in oid if ch.isdigit())
            if od:
                out.add(od)
        return out


def _export_audit(db: str, out_csv: Path, start_ts: str, end_ts: str, hypothesis_type: str, reversal_direction: str, slippage_max: float) -> int:
    schema, cols = _table_cols(db, "betslip_audit_results")
    qtbl = f"{_qident(schema)}.{_qident('betslip_audit_results')}"
    c_id = _pick(cols, ["id", "audit_id"])
    c_ts = _pick(cols, ["audited_at", "created_at", "updated_at"])
    c_hyp = _pick(cols, ["hypothesis_details"])
    c_hyp_type = _pick(cols, ["hypothesis_type"])
    c_rev = _pick(cols, ["reversal_direction"])
    if not c_id or not c_ts or not c_hyp:
        raise RuntimeError("betslip_audit_results missing required columns")

    qh = _qident(c_hyp)
    slip_expr = (
        "COALESCE("
        f"NULLIF({qh} #>> '{{value_sizing,slippage_pre_pct}}','')::double precision,"
        f"NULLIF({qh} #>> '{{finance,value_sizing,slippage_pre_pct}}','')::double precision,"
        f"NULLIF({qh} #>> '{{slippage_pre_pct}}','')::double precision"
        ")"
    )
    query = f"""
    SELECT
      {_qident(c_id)}::text AS audit_id,
      {_qident(c_ts)}::text AS audited_at
    FROM {qtbl}
    WHERE {_qident(c_ts)} >= '{start_ts}'::timestamptz
      AND {_qident(c_ts)} <= '{end_ts}'::timestamptz
      AND {_qident(c_hyp_type)} = '{hypothesis_type}'
      AND {_qident(c_rev)} = '{reversal_direction}'
      AND {slip_expr} < {float(slippage_max)}
    """
    _psql_copy(db, query, out_csv)
    return _count_rows(out_csv)


def _export_bridge(db: str, out_csv: Path, start_ts: str, end_ts: str) -> int:
    schema, cols = _table_cols(db, "executor_bridge_seen")
    qtbl = f"{_qident(schema)}.{_qident('executor_bridge_seen')}"
    c_created = _pick(cols, ["created_at", "timestamp", "ts"])
    if not c_created:
        raise RuntimeError("executor_bridge_seen missing created_at")
    json_cols = [c for c in cols if _norm(c) in {_norm(x) for x in ["meta", "payload_json", "row_json", "payload", "message", "data"]}]

    audit_terms = []
    exec_terms = []
    c_aid = _pick(cols, ["audit_id"])
    c_src = _pick(cols, ["src_id", "source_id", "sourceid"])
    c_ex = _pick(cols, ["execution_id"])
    if c_aid:
        audit_terms.append(f"NULLIF({_qident(c_aid)}::text,'')")
    if c_src:
        audit_terms.append(f"NULLIF({_qident(c_src)}::text,'')")
    if c_ex:
        exec_terms.append(f"NULLIF({_qident(c_ex)}::text,'')")

    for jc in json_cols:
        qj = _qident(jc)
        audit_terms += [
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
        exec_terms += [
            f"NULLIF({qj}::jsonb #>> '{{execution_id}}','')",
            f"NULLIF({qj}::jsonb #>> '{{executionId}}','')",
            f"NULLIF({qj}::jsonb #>> '{{exec_id}}','')",
            f"NULLIF({qj}::jsonb #>> '{{execution,id}}','')",
            f"NULLIF(jsonb_path_query_first({qj}::jsonb, '$.**.execution_id') #>> '{{}}','')",
            f"NULLIF(jsonb_path_query_first({qj}::jsonb, '$.**.executionId') #>> '{{}}','')",
            f"NULLIF(jsonb_path_query_first({qj}::jsonb, '$.**.exec_id') #>> '{{}}','')",
            f"NULLIF(jsonb_path_query_first({qj}::jsonb, '$.**.execution.id') #>> '{{}}','')",
        ]
    audit_expr = "COALESCE(" + ",".join(audit_terms + ["NULL::text"]) + ")"
    exec_expr = "COALESCE(" + ",".join(exec_terms + ["NULL::text"]) + ")"
    start2 = (_pdt(start_ts) - timedelta(days=2)).isoformat()
    end2 = (_pdt(end_ts) + timedelta(days=2)).isoformat()
    query = f"""
    SELECT
      {audit_expr} AS audit_id,
      {exec_expr} AS execution_id,
      {_qident(c_created)}::text AS created_at
    FROM {qtbl}
    WHERE {_qident(c_created)} >= '{start2}'::timestamptz
      AND {_qident(c_created)} <= '{end2}'::timestamptz
      AND {audit_expr} IS NOT NULL
      AND {exec_expr} IS NOT NULL
    """
    _psql_copy(db, query, out_csv)
    return _count_rows(out_csv)


def _count_rows(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        rd = csv.DictReader(f)
        for _ in rd:
            n += 1
    return n


def _audit_day_map(audit_csv: Path) -> Tuple[Dict[str, str], Dict[str, int]]:
    aid_day: Dict[str, str] = {}
    day_cnt: Dict[str, int] = defaultdict(int)
    with audit_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            aid = str(r.get("audit_id", "")).strip()
            dt = _pdt(r.get("audited_at", ""))
            if not aid or dt is None:
                continue
            day = dt.date().isoformat()
            aid_day[aid] = day
            day_cnt[day] += 1
    return aid_day, day_cnt


def _bridge_latest_map(bridge_csv: Path) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    latest_raw: Dict[str, Tuple[datetime, str]] = {}
    latest_norm: Dict[str, Tuple[datetime, str]] = {}
    latest_dig: Dict[str, Tuple[datetime, str]] = {}
    with bridge_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            aid = str(r.get("audit_id", "")).strip()
            ex = str(r.get("execution_id", "")).strip()
            dt = _pdt(r.get("created_at", ""))
            if not aid or not ex or dt is None:
                continue
            prev_raw = latest_raw.get(aid)
            if prev_raw is None or dt > prev_raw[0]:
                latest_raw[aid] = (dt, ex)
            an = _norm(aid)
            if an:
                prev_norm = latest_norm.get(an)
                if prev_norm is None or dt > prev_norm[0]:
                    latest_norm[an] = (dt, ex)
            ad = "".join(ch for ch in aid if ch.isdigit())
            if ad:
                prev_dig = latest_dig.get(ad)
                if prev_dig is None or dt > prev_dig[0]:
                    latest_dig[ad] = (dt, ex)
    return (
        {k: v[1] for k, v in latest_raw.items()},
        {k: v[1] for k, v in latest_norm.items()},
        {k: v[1] for k, v in latest_dig.items()},
    )


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
            kn = _norm(k)
            if kn in {"executionid", "execution_id"}:
                sv = str(v or "").strip()
                if sv:
                    exec_ids.add(sv)
            if kn in {"orderid", "order_id", "ticketid", "betid"}:
                sv = str(v or "").strip()
                if sv:
                    order_ids.add(sv)
            _extract_ids(v, exec_ids, order_ids)
    elif isinstance(obj, list):
        for x in obj:
            _extract_ids(x, exec_ids, order_ids)


def _build_exec_order_map(roots: Sequence[str], max_files: int) -> Tuple[Dict[str, str], int]:
    files: List[Path] = []
    for root in roots:
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
            "*live*jsonl*",
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

    pair_count: Dict[Tuple[str, str], int] = defaultdict(int)
    rx_exec = re.compile(r'"execution[_ ]?id"\s*:\s*"([^"]+)"', re.I)
    rx_order = re.compile(r'"order[_ ]?id"\s*:\s*"([^"]+)"', re.I)
    for p in files2:
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
                        exec_ids.add(m.strip())
                    for m in rx_order.findall(ln):
                        order_ids.add(m.strip())
                if not exec_ids or not order_ids:
                    continue
                for ex in exec_ids:
                    for od in order_ids:
                        if ex and od:
                            pair_count[(ex, od)] += 1
        except Exception:
            continue

    by_exec: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for (ex, od), c in pair_count.items():
        by_exec[ex].append((c, od))
    out = {}
    for ex, arr in by_exec.items():
        arr.sort(key=lambda t: (t[0], len(t[1])), reverse=True)
        out[ex] = arr[0][1]
    return out, len(files2)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Diagnose chain coverage for 5Ms base")
    ap.add_argument("--start-ts", default="2026-01-01T00:00:00+00:00")
    ap.add_argument("--end-ts", default="")
    ap.add_argument("--balance-csv", default="")
    ap.add_argument("--hypothesis-type", default="H3B")
    ap.add_argument("--reversal-direction", default="up")
    ap.add_argument("--slippage-max", type=float, default=0.0)
    ap.add_argument("--executor-root", action="append", default=["/home/betbot/Bets/betinasia_bot/logs", "/workspace/betinasia_bot/logs"])
    ap.add_argument("--max-log-files", type=int, default=2000)
    ap.add_argument("--out-prefix", default=f"/tmp/diag_5ms_chain_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    db = os.environ.get("DATABASE_URL", "").strip()
    if not db:
        raise SystemExit("DATABASE_URL nao definido")

    bal = Path(args.balance_csv) if str(args.balance_csv or "").strip() else _latest_balance_csv()
    if not bal.exists():
        raise SystemExit(f"balance nao encontrado: {bal}")
    end_ts = str(args.end_ts or "").strip()
    if not end_ts:
        mx = _max_post_date(bal)
        if mx is None:
            raise SystemExit("nao consegui inferir end_ts do balance")
        end_ts = mx.isoformat()

    print(f"[INFO] balance={bal}")
    print(f"[INFO] start_ts={args.start_ts} end_ts={end_ts}")

    out_prefix = Path(args.out_prefix)
    audit_csv = Path(str(out_prefix) + "_audit.csv")
    bridge_csv = Path(str(out_prefix) + "_bridge.csv")
    cov_csv = Path(str(out_prefix) + "_coverage_by_day.csv")
    summary_json = Path(str(out_prefix) + "_summary.json")

    n_audit = _export_audit(db, audit_csv, args.start_ts, end_ts, args.hypothesis_type, args.reversal_direction, args.slippage_max)
    n_bridge = _export_bridge(db, bridge_csv, args.start_ts, end_ts)
    aid_day, day_total = _audit_day_map(audit_csv)
    aid_exec_raw, aid_exec_norm, aid_exec_dig = _bridge_latest_map(bridge_csv)
    ex_order, n_logs = _build_exec_order_map(args.executor_root, int(args.max_log_files))
    ledger_keys = _build_order_pnl_keys(bal)

    day_exec: Dict[str, int] = defaultdict(int)
    day_order: Dict[str, int] = defaultdict(int)
    day_pnl: Dict[str, int] = defaultdict(int)
    missing_exec_after = []
    missing_order_after = []
    missing_pnl_after = []
    end_dt = _pdt(end_ts)
    cutoff = datetime(2026, 5, 31, 23, 59, 59, tzinfo=timezone.utc)

    ex_comp = {_norm(k): v for k, v in ex_order.items() if _norm(k)}
    ex_dig = {"".join(ch for ch in k if ch.isdigit()): v for k, v in ex_order.items() if "".join(ch for ch in k if ch.isdigit())}

    def lookup_order(ex: str) -> Optional[str]:
        if not ex:
            return None
        return ex_order.get(ex) or ex_comp.get(_norm(ex)) or ex_dig.get("".join(ch for ch in ex if ch.isdigit()))

    def lookup_exec(aid: str) -> str:
        if not aid:
            return ""
        return aid_exec_raw.get(aid, "") or aid_exec_norm.get(_norm(aid), "") or aid_exec_dig.get("".join(ch for ch in aid if ch.isdigit()), "")

    for aid, day in aid_day.items():
        ex = lookup_exec(aid)
        if ex:
            day_exec[day] += 1
        else:
            dt = _pdt(day + "T00:00:00+00:00")
            if dt and dt > cutoff and len(missing_exec_after) < 30:
                missing_exec_after.append(aid)
            continue
        od = lookup_order(ex)
        if od:
            day_order[day] += 1
        else:
            dt = _pdt(day + "T00:00:00+00:00")
            if dt and dt > cutoff and len(missing_order_after) < 30:
                missing_order_after.append(ex)
            continue
        ok = od in ledger_keys or _norm(od) in ledger_keys or "".join(ch for ch in od if ch.isdigit()) in ledger_keys
        if ok:
            day_pnl[day] += 1
        else:
            dt = _pdt(day + "T00:00:00+00:00")
            if dt and dt > cutoff and len(missing_pnl_after) < 30:
                missing_pnl_after.append(od)

    days = sorted(day_total.keys())
    cov_csv.parent.mkdir(parents=True, exist_ok=True)
    with cov_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(
            [
                "day",
                "audit_total",
                "with_exec",
                "with_order",
                "with_pnl",
                "cov_exec_pct",
                "cov_order_pct",
                "cov_pnl_pct",
            ]
        )
        for d in days:
            a = day_total.get(d, 0)
            e = day_exec.get(d, 0)
            o = day_order.get(d, 0)
            p = day_pnl.get(d, 0)
            wr.writerow(
                [
                    d,
                    a,
                    e,
                    o,
                    p,
                    round(100.0 * e / a, 2) if a else 0.0,
                    round(100.0 * o / a, 2) if a else 0.0,
                    round(100.0 * p / a, 2) if a else 0.0,
                ]
            )

    total_a = sum(day_total.values())
    total_e = sum(day_exec.values())
    total_o = sum(day_order.values())
    total_p = sum(day_pnl.values())

    last_day_any_pnl = None
    for d in days:
        if day_pnl.get(d, 0) > 0:
            last_day_any_pnl = d

    summary = {
        "balance_csv": str(bal),
        "start_ts": args.start_ts,
        "end_ts": end_ts,
        "audit_rows": n_audit,
        "bridge_rows": n_bridge,
        "executor_logs_scanned": n_logs,
        "exec_order_pairs": len(ex_order),
        "ledger_keys": len(ledger_keys),
        "coverage_total": {
            "audit_total": total_a,
            "with_exec": total_e,
            "with_order": total_o,
            "with_pnl": total_p,
            "cov_exec_pct": round(100.0 * total_e / total_a, 2) if total_a else 0.0,
            "cov_order_pct": round(100.0 * total_o / total_a, 2) if total_a else 0.0,
            "cov_pnl_pct": round(100.0 * total_p / total_a, 2) if total_a else 0.0,
        },
        "last_day_with_any_pnl_coverage": last_day_any_pnl,
        "samples_missing_after_2026_05_31": {
            "audit_without_exec": missing_exec_after,
            "exec_without_order": missing_order_after,
            "order_without_pnl": missing_pnl_after,
        },
        "outputs": {
            "audit_csv": str(audit_csv),
            "bridge_csv": str(bridge_csv),
            "coverage_by_day_csv": str(cov_csv),
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] coverage_by_day:", cov_csv)
    print("[OK] summary_json:", summary_json)
    print("[OK] coverage_total:", summary["coverage_total"])
    print("[OK] last_day_with_any_pnl_coverage:", last_day_any_pnl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

