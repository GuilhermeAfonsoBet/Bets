#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build robust base for 5Ms via chain:
audit_id -> execution_id (executor_bridge_seen) -> order_id (executor logs) -> pnl (balance CSV)

Motivacao:
- quando join direto audit/order retorna matches=0/0, este pipeline costuma recuperar cobertura.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import subprocess
from collections import defaultdict
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


def _norm_compact(s: str) -> str:
    return "".join(ch for ch in str(s or "").strip().lower() if ch.isalnum())


def _norm_digits(s: str) -> str:
    return "".join(ch for ch in str(s or "") if ch.isdigit())


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


def _psql_copy(db: str, sql_query: str, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    path = str(out_csv).replace("'", "''")
    stmt = f"\\copy ({sql_query}) TO '{path}' WITH CSV HEADER"
    p = _run(["psql", db, "-v", "ON_ERROR_STOP=1", "-c", stmt])
    if p.returncode != 0:
        raise RuntimeError(f"psql copy falhou: {p.stderr.strip()}")


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


def _pick(cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    by = {_norm_compact(c): c for c in cols}
    for cand in candidates:
        hit = by.get(_norm_compact(cand))
        if hit:
            return hit
    return None


def _qident(x: str) -> str:
    return '"' + str(x).replace('"', '""') + '"'


def _latest_balance_csv() -> Path:
    cands = sorted(glob_list("/home/betbot/Bets/betinasia_bot/logs/accounting/*balance*.csv"), key=os.path.getmtime)
    if not cands:
        cands = sorted(glob_list("/workspace/betinasia_bot/logs/accounting/*balance*.csv"), key=os.path.getmtime)
    if not cands:
        raise RuntimeError("nenhum balance csv encontrado")
    return Path(cands[-1])


def glob_list(pattern: str) -> List[str]:
    import glob

    return glob.glob(pattern)


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


def _build_ledger_maps(balance_csv: Path) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    with balance_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        rd = csv.DictReader(f)
        fields = list(rd.fieldnames or [])
        oid_col = _pick(fields, ["order id", "order_id", "orderid", "bet id", "bet_id", "ticket id", "ticket_id"])
        val_col = _pick(fields, ["amount", "pnl_real", "pnl", "profit", "pl", "result", "net"])
        odd_col = _pick(fields, ["got price", "got_price", "price", "odd", "odds", "matched price"])
        if not oid_col or not val_col:
            raise RuntimeError(f"balance sem colunas esperadas: oid_col={oid_col} val_col={val_col}")
        pnl_raw: Dict[str, float] = defaultdict(float)
        odd_vals: Dict[str, List[float]] = defaultdict(list)
        for r in rd:
            oid = str(r.get(oid_col, "")).strip()
            if not oid:
                continue
            v = _pf(r.get(val_col))
            if v is not None:
                pnl_raw[oid] += float(v)
            if odd_col:
                od = _pf(r.get(odd_col))
                if od is not None and od > 1:
                    odd_vals[oid].append(float(od))
    odd_raw: Dict[str, float] = {}
    for k, arr in odd_vals.items():
        if not arr:
            continue
        arr2 = sorted(arr)
        n = len(arr2)
        odd_raw[k] = arr2[n // 2] if n % 2 == 1 else 0.5 * (arr2[n // 2 - 1] + arr2[n // 2])

    pnl_comp = {_norm_compact(k): v for k, v in pnl_raw.items() if _norm_compact(k)}
    pnl_dig = {_norm_digits(k): v for k, v in pnl_raw.items() if _norm_digits(k)}
    odd_comp = {_norm_compact(k): v for k, v in odd_raw.items() if _norm_compact(k)}
    odd_dig = {_norm_digits(k): v for k, v in odd_raw.items() if _norm_digits(k)}
    return pnl_raw, pnl_comp, pnl_dig, odd_raw, odd_comp, odd_dig


def _export_audit_universe(db: str, out_csv: Path, start_ts: str, end_ts: str, hypothesis_type: str, reversal_direction: str, slippage_max: float) -> int:
    schema, cols = _table_cols(db, "betslip_audit_results")
    qtbl = f"{_qident(schema)}.{_qident('betslip_audit_results')}"
    c_id = _pick(cols, ["id", "audit_id"])
    c_event = _pick(cols, ["event_id", "match_id", "fixture_id", "game_id"])
    c_league = _pick(cols, ["league", "league_name", "competition", "tournament"])
    c_ts = _pick(cols, ["audited_at", "created_at", "updated_at"])
    c_hyp_type = _pick(cols, ["hypothesis_type"])
    c_rev = _pick(cols, ["reversal_direction"])
    c_hyp = _pick(cols, ["hypothesis_details"])
    c_stake = _pick(cols, ["stake_real", "stake"])
    c_side = _pick(cols, ["side"])
    c_regime = _pick(cols, ["regime", "market_regime"])

    if not c_id or not c_ts:
        raise RuntimeError("betslip_audit_results sem colunas minimas id/timestamp")

    def col_or_null(c: Optional[str], alias: str) -> str:
        return f"{_qident(c)}::text AS {_qident(alias)}" if c else f"NULL::text AS {_qident(alias)}"

    if c_hyp:
        qh = _qident(c_hyp)
        slip_expr = (
            "COALESCE("
            f"NULLIF({qh} #>> '{{value_sizing,slippage_pre_pct}}','')::double precision,"
            f"NULLIF({qh} #>> '{{finance,value_sizing,slippage_pre_pct}}','')::double precision,"
            f"NULLIF({qh} #>> '{{slippage_pre_pct}}','')::double precision"
            ")"
        )
        if c_stake:
            stake_expr = f"COALESCE(NULLIF({_qident(c_stake)}::text,'')::double precision, NULLIF({qh} #>> '{{value_sizing,stake}}','')::double precision, NULLIF({qh} #>> '{{finance,value_sizing,stake}}','')::double precision)"
        else:
            stake_expr = f"COALESCE(NULLIF({qh} #>> '{{value_sizing,stake}}','')::double precision, NULLIF({qh} #>> '{{finance,value_sizing,stake}}','')::double precision)"
    else:
        slip_expr = "NULL::double precision"
        stake_expr = f"NULLIF({_qident(c_stake)}::text,'')::double precision" if c_stake else "NULL::double precision"

    where_parts = [f"{_qident(c_ts)} >= '{start_ts}'::timestamptz", f"{_qident(c_ts)} <= '{end_ts}'::timestamptz"]
    if c_hyp_type:
        where_parts.append(f"{_qident(c_hyp_type)} = '{hypothesis_type}'")
    if c_rev:
        where_parts.append(f"{_qident(c_rev)} = '{reversal_direction}'")
    where_parts.append(f"{slip_expr} < {float(slippage_max)}")
    where_sql = " AND ".join(where_parts)

    sql = f"""
    SELECT
      {_qident(c_id)}::text AS audit_id,
      {col_or_null(c_event, 'event_id')},
      {col_or_null(c_league, 'league')},
      {_qident(c_ts)}::text AS audited_at,
      {slip_expr} AS slippage_pre_pct,
      {stake_expr} AS stake_real,
      {col_or_null(c_side, 'side')},
      {col_or_null(c_regime, 'regime')}
    FROM {qtbl}
    WHERE {where_sql}
    """
    _psql_copy(db, sql, out_csv)
    return _count_rows(out_csv)


def _export_bridge_audit_exec(db: str, out_csv: Path, start_ts: str, end_ts: str) -> int:
    schema, cols = _table_cols(db, "executor_bridge_seen")
    qtbl = f"{_qident(schema)}.{_qident('executor_bridge_seen')}"
    c_created = _pick(cols, ["created_at", "ts", "timestamp"])
    if not c_created:
        raise RuntimeError("executor_bridge_seen sem created_at")

    json_cols = [c for c in cols if _norm_compact(c) in {_norm_compact(x) for x in ["meta", "payload_json", "row_json", "payload", "message", "data"]}]
    expr_exec_terms = []
    expr_audit_terms = []
    if _pick(cols, ["execution_id"]):
        expr_exec_terms.append(f"NULLIF({_qident(_pick(cols, ['execution_id']))}::text,'')")
    if _pick(cols, ["audit_id"]):
        expr_audit_terms.append(f"NULLIF({_qident(_pick(cols, ['audit_id']))}::text,'')")
    if _pick(cols, ["src_id"]):
        expr_audit_terms.append(f"NULLIF({_qident(_pick(cols, ['src_id']))}::text,'')")

    for jc in json_cols:
        qj = _qident(jc)
        expr_exec_terms += [
            f"NULLIF({qj}::jsonb #>> '{{execution_id}}','')",
            f"NULLIF({qj}::jsonb #>> '{{executionId}}','')",
            f"NULLIF(jsonb_path_query_first({qj}::jsonb, '$.**.execution_id') #>> '{{}}','')",
            f"NULLIF(jsonb_path_query_first({qj}::jsonb, '$.**.executionId') #>> '{{}}','')",
        ]
        expr_audit_terms += [
            f"NULLIF({qj}::jsonb #>> '{{audit_id}}','')",
            f"NULLIF(jsonb_path_query_first({qj}::jsonb, '$.**.audit_id') #>> '{{}}','')",
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
    _psql_copy(db, sql, out_csv)
    return _count_rows(out_csv)


def _count_rows(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        rd = csv.DictReader(f)
        for _ in rd:
            n += 1
    return n


def _build_latest_audit_exec_map(path: Path) -> Dict[str, str]:
    latest: Dict[str, Tuple[datetime, str]] = {}
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            aid = str(r.get("audit_id", "")).strip()
            ex = str(r.get("execution_id", "")).strip()
            dt = _pdt(r.get("created_at", ""))
            if not aid or not ex or dt is None:
                continue
            prev = latest.get(aid)
            if prev is None or dt > prev[0]:
                latest[aid] = (dt, ex)
    return {k: v[1] for k, v in latest.items()}


def _extract_ids_from_obj(obj: object, exec_ids: set[str], order_ids: set[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            kn = _norm_compact(k)
            if kn in {"executionid", "execution_id"}:
                sv = str(v or "").strip()
                if sv:
                    exec_ids.add(sv)
            if kn in {"orderid", "order_id", "orderidid", "ticketid", "betid"}:
                sv = str(v or "").strip()
                if sv:
                    order_ids.add(sv)
            _extract_ids_from_obj(v, exec_ids, order_ids)
    elif isinstance(obj, list):
        for it in obj:
            _extract_ids_from_obj(it, exec_ids, order_ids)


def _iter_log_lines(path: Path):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                yield ln
    else:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                yield ln


def _build_exec_order_map(log_roots: Sequence[str], max_files: int) -> Dict[str, str]:
    files: List[Path] = []
    for root in log_roots:
        rr = Path(str(root))
        if not rr.exists():
            continue
        files += list(rr.rglob("*executor*jsonl"))
        files += list(rr.rglob("*executor*jsonl.gz"))
        files += list(rr.rglob("*executor_live*.log"))
        files += list(rr.rglob("*executor_live*.log.gz"))
    uniq = {}
    for p in files:
        uniq[str(p.resolve())] = p
    files2 = sorted(uniq.values(), key=lambda p: p.stat().st_mtime, reverse=True)
    if max_files > 0:
        files2 = files2[:max_files]
    print(f"[INFO] logs executor candidatos: {len(files2)}")

    pair_count: Dict[Tuple[str, str], int] = defaultdict(int)
    rx_exec = re.compile(r'"execution[_ ]?id"\s*:\s*"([^"]+)"', re.I)
    rx_order = re.compile(r'"order[_ ]?id"\s*:\s*"([^"]+)"', re.I)
    for i, p in enumerate(files2, start=1):
        if i % 50 == 0:
            print(f"[INFO] parsing logs: {i}/{len(files2)}")
        try:
            for ln in _iter_log_lines(p):
                ln = ln.strip()
                if not ln:
                    continue
                exec_ids: set[str] = set()
                order_ids: set[str] = set()
                try:
                    obj = json.loads(ln)
                    _extract_ids_from_obj(obj, exec_ids, order_ids)
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

    by_exec: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for (ex, od), c in pair_count.items():
        by_exec[ex].append((c, od))
    out = {}
    for ex, arr in by_exec.items():
        arr.sort(key=lambda t: (t[0], len(t[1])), reverse=True)
        out[ex] = arr[0][1]
    print(f"[OK] exec->order mapeados: {len(out)}")
    return out


def _lookup_order(exec_id: str, ex_raw: Dict[str, str], ex_comp: Dict[str, str], ex_dig: Dict[str, str]) -> Optional[str]:
    if not exec_id:
        return None
    return ex_raw.get(exec_id) or ex_comp.get(_norm_compact(exec_id)) or ex_dig.get(_norm_digits(exec_id))


def _lookup_pnl_odd(order_id: str, pnl_raw, pnl_comp, pnl_dig, odd_raw, odd_comp, odd_dig) -> Tuple[Optional[float], Optional[float]]:
    if not order_id:
        return None, None
    pnl = pnl_raw.get(order_id)
    if pnl is None:
        pnl = pnl_comp.get(_norm_compact(order_id))
    if pnl is None:
        pnl = pnl_dig.get(_norm_digits(order_id))

    odd = odd_raw.get(order_id)
    if odd is None:
        odd = odd_comp.get(_norm_compact(order_id))
    if odd is None:
        odd = odd_dig.get(_norm_digits(order_id))
    return pnl, odd


def build_base(
    db: str,
    start_ts: str,
    end_ts: str,
    balance_csv: Path,
    executor_roots: Sequence[str],
    max_log_files: int,
    out_csv: Path,
    hypothesis_type: str,
    reversal_direction: str,
    slippage_max: float,
) -> Tuple[int, Optional[datetime], Optional[datetime]]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    audit_csv = Path(f"/tmp/base_audit_{ts}.csv")
    bridge_csv = Path(f"/tmp/base_bridge_audit_exec_{ts}.csv")

    n_audit = _export_audit_universe(db, audit_csv, start_ts, end_ts, hypothesis_type, reversal_direction, slippage_max)
    print(f"[OK] audit rows: {n_audit} ({audit_csv})")
    n_bridge = _export_bridge_audit_exec(db, bridge_csv, start_ts, end_ts)
    print(f"[OK] bridge rows: {n_bridge} ({bridge_csv})")

    audit_to_exec = _build_latest_audit_exec_map(bridge_csv)
    print(f"[OK] audit->exec mapeados: {len(audit_to_exec)}")

    ex_map_raw = _build_exec_order_map(executor_roots, max_log_files)
    ex_map_comp = {_norm_compact(k): v for k, v in ex_map_raw.items() if _norm_compact(k)}
    ex_map_dig = {_norm_digits(k): v for k, v in ex_map_raw.items() if _norm_digits(k)}

    pnl_raw, pnl_comp, pnl_dig, odd_raw, odd_comp, odd_dig = _build_ledger_maps(balance_csv)
    print(f"[OK] ledger keys: raw={len(pnl_raw)}")

    rows_out = 0
    n_with_exec = 0
    n_with_order = 0
    n_with_pnl = 0
    mn = None
    mx = None
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with audit_csv.open("r", encoding="utf-8", errors="ignore", newline="") as fin, out_csv.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        rd = csv.DictReader(fin)
        headers = [
            "audit_id",
            "event_id",
            "league",
            "audited_at",
            "slippage_pre_pct",
            "execution_id",
            "order_id",
            "pnl_real",
            "stake",
            "side",
            "regime",
        ]
        wr = csv.DictWriter(fout, fieldnames=headers)
        wr.writeheader()
        for i, r in enumerate(rd, start=1):
            aid = str(r.get("audit_id", "")).strip()
            ex = audit_to_exec.get(aid, "")
            if not ex:
                continue
            n_with_exec += 1
            od = _lookup_order(ex, ex_map_raw, ex_map_comp, ex_map_dig)
            if not od:
                continue
            n_with_order += 1
            pnl, odd = _lookup_pnl_odd(od, pnl_raw, pnl_comp, pnl_dig, odd_raw, odd_comp, odd_dig)
            if pnl is None:
                continue
            n_with_pnl += 1
            st = _pf(r.get("stake_real", ""))
            if st is None or st <= 0:
                if pnl < 0:
                    st = -float(pnl)
                elif pnl > 0 and odd is not None and odd > 1:
                    st = float(pnl) / (float(odd) - 1.0)
            if st is None or st <= 0:
                continue
            dt = _pdt(r.get("audited_at", ""))
            if dt:
                if mn is None or dt < mn:
                    mn = dt
                if mx is None or dt > mx:
                    mx = dt
            wr.writerow(
                {
                    "audit_id": aid,
                    "event_id": str(r.get("event_id", "")).strip(),
                    "league": str(r.get("league", "")).strip(),
                    "audited_at": str(r.get("audited_at", "")).strip(),
                    "slippage_pre_pct": str(r.get("slippage_pre_pct", "")).strip(),
                    "execution_id": ex,
                    "order_id": od,
                    "pnl_real": f"{float(pnl):.10f}",
                    "stake": f"{float(st):.10f}",
                    "side": str(r.get("side", "back")).strip() or "back",
                    "regime": str(r.get("regime", "pre")).strip() or "pre",
                }
            )
            rows_out += 1
            if i % 50000 == 0:
                print(f"[INFO] progresso audit={i} rows_out={rows_out}")

    print(f"[OK] rows_out={rows_out}")
    print(f"[OK] coverage audit->exec={n_with_exec} exec->order={n_with_order} order->pnl={n_with_pnl}")
    print(f"[OK] period={mn} -> {mx}")
    return rows_out, mn, mx


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build base 5Ms via audit->exec->order->ledger")
    ap.add_argument("--start-ts", default="2026-01-01T00:00:00+00:00")
    ap.add_argument("--end-ts", default="")
    ap.add_argument("--balance-csv", default="")
    ap.add_argument("--hypothesis-type", default="H3B")
    ap.add_argument("--reversal-direction", default="up")
    ap.add_argument("--slippage-max", type=float, default=0.0, help="Filtro slippage_pre_pct < slippage-max")
    ap.add_argument(
        "--executor-root",
        action="append",
        default=[
            "/home/betbot/Bets/betinasia_bot/logs",
            "/workspace/betinasia_bot/logs",
        ],
    )
    ap.add_argument("--max-log-files", type=int, default=1500)
    ap.add_argument("--out-csv", default="")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    db = os.environ.get("DATABASE_URL", "").strip()
    if not db:
        raise SystemExit("DATABASE_URL nao definido")

    bal = Path(args.balance_csv).expanduser() if str(args.balance_csv or "").strip() else _latest_balance_csv()
    if not bal.exists():
        raise SystemExit(f"balance csv nao encontrado: {bal}")
    end_ts = str(args.end_ts or "").strip()
    if not end_ts:
        mx = _max_post_date(bal)
        if mx is None:
            raise SystemExit("nao consegui inferir end-ts do balance")
        end_ts = mx.isoformat()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_csv = Path(args.out_csv) if str(args.out_csv or "").strip() else Path(
        f"/tmp/base_5ms_real_ate_{end_ts[:10].replace('-', '')}_inferred_{ts}.csv"
    )
    print(f"[INFO] balance={bal}")
    print(f"[INFO] start_ts={args.start_ts} end_ts={end_ts}")
    print(f"[INFO] out_csv={out_csv}")

    rows, mn, mx = build_base(
        db=db,
        start_ts=args.start_ts,
        end_ts=end_ts,
        balance_csv=bal,
        executor_roots=args.executor_root,
        max_log_files=int(args.max_log_files),
        out_csv=out_csv,
        hypothesis_type=args.hypothesis_type,
        reversal_direction=args.reversal_direction,
        slippage_max=float(args.slippage_max),
    )
    if rows <= 0:
        raise SystemExit("rows_out=0 (sem base util via bridge/exec/order)")
    print(f"[OK] base pronta: {out_csv}")
    print(f"[OK] period: {mn} -> {mx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

