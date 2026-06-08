#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regenera projecao_por_aposta com fallback robusto:
1) tenta extrair P&L por order_id de balance/ledger CSV
2) se nao encontrar CSV valido, tenta derivar de tabela de ledger no DB
3) cruza com betslip_audit_results e gera CSVs timestampados em /tmp

Nao sobrescreve /tmp/projecao_por_aposta.csv por padrao.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SYSTEM_SCHEMAS = {"pg_catalog", "information_schema"}
LEDGER_VALUE_COL_PRIORITY = [
    "amount",
    "pnl_real",
    "pnl",
    "profit",
    "pl",
    "result",
    "net_amount",
    "net",
]
TYPE_COL_CANDIDATES = [
    "type",
    "transaction_type",
    "entry_type",
    "kind",
    "description",
    "reason",
    "source",
]
ORDER_COL_CANDIDATES = [
    "order_id",
    "orderid",
    "order",
    "id_order",
    "external_order_id",
    "bet_id",
]
CSV_VALUE_COL_CANDIDATES = [
    "amount",
    "pnl_real",
    "pnl",
    "profit",
    "pl",
    "result",
    "net",
    "value",
    "delta",
]
EXCLUDE_TX_PAT = re.compile(r"(deposit|withdraw|transfer|fee|commission|bonus|rebate)", re.IGNORECASE)
NUMERIC_TYPES = {
    "smallint",
    "integer",
    "bigint",
    "numeric",
    "real",
    "double precision",
    "decimal",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerar projecao_por_aposta de forma robusta")
    p.add_argument("--balance-csv", default="", help="Caminho explicito para balance/ledger CSV")
    p.add_argument(
        "--search-root",
        action="append",
        default=[],
        help="Raiz adicional para procurar balance/ledger CSV (pode repetir)",
    )
    p.add_argument("--max-depth", type=int, default=7, help="Profundidade maxima da busca recursiva por CSV")
    p.add_argument(
        "--prefer-db-ledger",
        type=int,
        default=0,
        help="Se 1, tenta primeiro ledger no banco antes de procurar CSV",
    )
    p.add_argument(
        "--write-default",
        type=int,
        default=0,
        help="Se 1, copia a saida principal para /tmp/projecao_por_aposta.csv",
    )
    return p.parse_args()


def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).strip().lower() if ch.isalnum())


def parse_float(v: object) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    s = s.replace(" ", "")
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


def run_cmd(args: Sequence[str]) -> str:
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Falha comando ({' '.join(args)}): {proc.stderr.strip()}")
    return proc.stdout


def run_psql_query(db_url: str, sql: str) -> str:
    return run_cmd(["psql", db_url, "-At", "-v", "ON_ERROR_STOP=1", "-c", sql])


def psql_copy_csv(db_url: str, sql_query: str, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    copy_stmt = f"\\copy ({sql_query}) TO '{str(out_csv)}' WITH CSV HEADER"
    run_cmd(["psql", db_url, "-v", "ON_ERROR_STOP=1", "-c", copy_stmt])


def quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def list_table_columns(db_url: str, table_name: str) -> Tuple[str, Dict[str, str]]:
    sql = f"""
    SELECT table_schema, column_name, data_type
    FROM information_schema.columns
    WHERE table_name = '{table_name}'
      AND table_schema NOT IN ('pg_catalog','information_schema')
    ORDER BY CASE WHEN table_schema='public' THEN 0 ELSE 1 END, ordinal_position;
    """
    out = run_psql_query(db_url, sql)
    schema = ""
    cols: Dict[str, str] = {}
    for ln in out.splitlines():
        parts = ln.split("|")
        if len(parts) != 3:
            continue
        sc, col, typ = parts
        if not schema:
            schema = sc
        cols[col] = typ
    if not schema or not cols:
        raise RuntimeError(f"Tabela {table_name} nao encontrada no DB.")
    return schema, cols


def pick_existing(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    by_norm = {_norm(c): c for c in columns}
    for cand in candidates:
        c = by_norm.get(_norm(cand))
        if c:
            return c
    return None


def walk_with_depth(root: Path, max_depth: int) -> Iterable[Path]:
    if not root.exists():
        return
    root = root.resolve()
    for cur_root, dirs, files in os.walk(root):
        rel = Path(cur_root).relative_to(root)
        if len(rel.parts) >= max_depth:
            dirs[:] = []
        for fn in files:
            yield Path(cur_root) / fn


def discover_balance_candidates(explicit_csv: str, extra_roots: List[str], max_depth: int) -> List[Path]:
    out: List[Path] = []
    if explicit_csv:
        p = Path(explicit_csv)
        if p.exists() and p.is_file():
            return [p.resolve()]
        raise RuntimeError(f"--balance-csv informado mas arquivo nao existe: {p}")

    roots = [
        "/home/betbot/Bets/betinasia_bot/logs",
        "/home/betbot/Bets/betinasia_bot",
        "/home/betbot/Bets",
        "/home/betbot",
        "/tmp",
        ".",
    ]
    roots.extend(extra_roots)
    seen = set()
    for rs in roots:
        rp = Path(rs).expanduser()
        if not rp.exists():
            continue
        for f in walk_with_depth(rp, max_depth=max_depth):
            if f.suffix.lower() != ".csv":
                continue
            low = f.name.lower()
            if "balance" not in low and "ledger" not in low:
                continue
            rr = str(f.resolve())
            if rr in seen:
                continue
            seen.add(rr)
            out.append(f.resolve())
    out.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return out


def sniff_reader(path: Path) -> csv.DictReader:
    sample = path.read_text(encoding="utf-8", errors="ignore")[:8192]
    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return csv.DictReader(lines, dialect=dialect)


def choose_csv_ledger_columns(path: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    try:
        rd = sniff_reader(path)
    except Exception:
        return None, None, None
    fields = list(rd.fieldnames or [])
    if not fields:
        return None, None, None
    order_col = pick_existing(fields, ORDER_COL_CANDIDATES)
    pnl_col = pick_existing(fields, CSV_VALUE_COL_CANDIDATES)
    type_col = pick_existing(fields, TYPE_COL_CANDIDATES)
    return order_col, pnl_col, type_col


def normalize_ledger_from_csv(in_csv: Path, out_csv: Path) -> int:
    rd = sniff_reader(in_csv)
    fields = list(rd.fieldnames or [])
    order_col = pick_existing(fields, ORDER_COL_CANDIDATES)
    pnl_col = pick_existing(fields, CSV_VALUE_COL_CANDIDATES)
    type_col = pick_existing(fields, TYPE_COL_CANDIDATES)
    if not order_col or not pnl_col:
        raise RuntimeError(
            f"CSV {in_csv} nao parece ledger valido. order_col={order_col}, value_col={pnl_col}"
        )

    agg: Dict[str, float] = defaultdict(float)
    for row in rd:
        oid = str(row.get(order_col, "")).strip()
        if not oid:
            continue
        if type_col:
            tx = str(row.get(type_col, "")).strip().lower()
            if tx and EXCLUDE_TX_PAT.search(tx):
                continue
        pnl = parse_float(row.get(pnl_col))
        if pnl is None:
            continue
        agg[oid] += pnl

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["order_id", "pnl_real"])
        for oid, pnl in agg.items():
            wr.writerow([oid, f"{pnl:.10f}"])
    return len(agg)


def discover_ledger_table(db_url: str) -> Tuple[str, str, Optional[str]]:
    sql = """
    SELECT table_schema, table_name, column_name, data_type
    FROM information_schema.columns
    WHERE table_schema NOT IN ('pg_catalog','information_schema')
    ORDER BY table_schema, table_name, ordinal_position;
    """
    out = run_psql_query(db_url, sql)
    tbl_cols: Dict[Tuple[str, str], Dict[str, str]] = defaultdict(dict)
    for ln in out.splitlines():
        parts = ln.split("|")
        if len(parts) != 4:
            continue
        sc, tb, col, typ = parts
        tbl_cols[(sc, tb)][col] = typ

    scored: List[Tuple[int, str, str, str, Optional[str]]] = []
    for (sc, tb), cols in tbl_cols.items():
        colnames = list(cols.keys())
        order_col = pick_existing(colnames, ORDER_COL_CANDIDATES)
        if not order_col:
            continue
        value_col = None
        for cand in LEDGER_VALUE_COL_PRIORITY:
            c = pick_existing(colnames, [cand])
            if c and cols.get(c, "").lower() in NUMERIC_TYPES:
                value_col = c
                break
        if not value_col:
            for c in colnames:
                if cols.get(c, "").lower() not in NUMERIC_TYPES:
                    continue
                if _norm(c) in {_norm(x) for x in LEDGER_VALUE_COL_PRIORITY}:
                    value_col = c
                    break
        if not value_col:
            continue
        type_col = pick_existing(colnames, TYPE_COL_CANDIDATES)
        score = 0
        n = tb.lower()
        if "ledger" in n:
            score += 40
        if "balance" in n:
            score += 30
        if "account" in n or "transaction" in n:
            score += 20
        if sc == "public":
            score += 10
        scored.append((score, sc, tb, value_col, type_col))

    if not scored:
        raise RuntimeError("Nao encontrei tabela de ledger no DB com (order_id + coluna numerica de valor).")
    scored.sort(reverse=True)
    _, sc, tb, val_col, type_col = scored[0]
    return f"{sc}.{tb}", val_col, type_col


def normalize_ledger_from_db(db_url: str, out_csv: Path) -> Tuple[str, int]:
    table, value_col, type_col = discover_ledger_table(db_url)
    schema, tb = table.split(".", 1)
    q_table = f"{quote_ident(schema)}.{quote_ident(tb)}"
    q_oid = quote_ident("order_id")
    q_val = quote_ident(value_col)

    where_parts = [f"{q_oid} IS NOT NULL"]
    if type_col:
        q_type = quote_ident(type_col)
        where_parts.append(
            f"(COALESCE(lower({q_type}::text),'') !~ '(deposit|withdraw|transfer|fee|commission|bonus|rebate)')"
        )
    where_sql = " AND ".join(where_parts)

    sql = f"""
    SELECT
      {q_oid}::text AS order_id,
      SUM(COALESCE({q_val}::double precision,0)) AS pnl_real
    FROM {q_table}
    WHERE {where_sql}
    GROUP BY 1
    """
    psql_copy_csv(db_url, sql, out_csv)
    count = 0
    with out_csv.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for _ in rd:
            count += 1
    return table, count


def export_audit_csv(db_url: str, out_csv: Path) -> None:
    schema, cols = list_table_columns(db_url, "betslip_audit_results")
    colnames = list(cols.keys())
    q_table = f"{quote_ident(schema)}.{quote_ident('betslip_audit_results')}"

    order_col = pick_existing(colnames, ["order_id"])
    if not order_col:
        raise RuntimeError("betslip_audit_results sem coluna order_id.")
    audit_col = pick_existing(colnames, ["audit_id", "id"]) or "audit_id"
    event_col = pick_existing(colnames, ["event_id", "match_id", "fixture_id", "game_id"])
    league_col = pick_existing(colnames, ["league", "league_name", "competition", "tournament"])
    ts_col = pick_existing(colnames, ["audited_at", "created_at", "updated_at"])
    side_col = pick_existing(colnames, ["side", "direction", "exec_side", "pick_side"])
    regime_col = pick_existing(colnames, ["regime", "market_regime", "market_phase", "phase"])
    status_col = pick_existing(colnames, ["status"])
    slip_col = pick_existing(colnames, ["slippage_pre_pct", "slippage_raw_pct", "slippage"])
    json_col = pick_existing(colnames, ["hypothesis_details"])

    stake_candidates = [c for c in ["stake_real", "exposure_real", "exposure", "stake"] if c in colnames]
    stake_terms = [f"NULLIF({quote_ident(c)}::double precision,0)" for c in stake_candidates]
    if json_col:
        q_json = quote_ident(json_col)
        stake_terms.extend(
            [
                f"NULLIF(({q_json} #>> '{{value_sizing,stake_real}}')::double precision,0)",
                f"NULLIF(({q_json} #>> '{{value_sizing,stake}}')::double precision,0)",
                f"NULLIF(({q_json} #>> '{{value_sizing,exposure}}')::double precision,0)",
            ]
        )
    stake_expr = f"COALESCE({', '.join(stake_terms)}, 0.0)"

    if slip_col:
        slip_expr = f"{quote_ident(slip_col)}::double precision"
    elif json_col:
        q_json = quote_ident(json_col)
        slip_expr = f"({q_json} #>> '{{value_sizing,slippage_pre_pct}}')::double precision"
    else:
        slip_expr = "NULL::double precision"

    def col_or_null(c: Optional[str], alias: str, cast: str = "text") -> str:
        if c:
            return f"{quote_ident(c)}::{cast} AS {quote_ident(alias)}"
        return f"NULL::{cast} AS {quote_ident(alias)}"

    where_parts = [f"{quote_ident(order_col)} IS NOT NULL", f"{stake_expr} > 0"]
    if status_col:
        where_parts.append(f"UPPER({quote_ident(status_col)}::text)='OK'")
    where_sql = " AND ".join(where_parts)

    sql = f"""
    SELECT
      {col_or_null(audit_col, 'audit_id')},
      {quote_ident(order_col)}::text AS "order_id",
      {stake_expr} AS "stake_real",
      {col_or_null(event_col, 'event_id')},
      {col_or_null(league_col, 'league')},
      {col_or_null(ts_col, 'audited_at')},
      {col_or_null(side_col, 'side')},
      {col_or_null(regime_col, 'regime')},
      {slip_expr} AS "slippage_pre_pct"
    FROM {q_table}
    WHERE {where_sql}
    """
    psql_copy_csv(db_url, sql, out_csv)


def join_audit_and_ledger(audit_csv: Path, ledger_csv: Path, out_csv: Path, out_enriched_csv: Path) -> Tuple[int, float, float]:
    pnl_by_oid: Dict[str, float] = {}
    with ledger_csv.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            oid = str(row.get("order_id", "")).strip()
            pnl = parse_float(row.get("pnl_real"))
            if oid and pnl is not None:
                pnl_by_oid[oid] = pnl

    rows = []
    with audit_csv.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            oid = str(row.get("order_id", "")).strip()
            if oid not in pnl_by_oid:
                continue
            st = parse_float(row.get("stake_real"))
            if st is None or st <= 0:
                continue
            pnl = pnl_by_oid[oid]
            roi = 100.0 * pnl / st
            rows.append(
                {
                    "audit_id": str(row.get("audit_id", "")).strip(),
                    "order_id": oid,
                    "stake_real": f"{st:.10f}",
                    "pnl_real": f"{pnl:.10f}",
                    "roi_real_pct": f"{roi:.10f}",
                    "event_id": str(row.get("event_id", "")).strip(),
                    "league": str(row.get("league", "")).strip(),
                    "audited_at": str(row.get("audited_at", "")).strip(),
                    "side": str(row.get("side", "")).strip(),
                    "regime": str(row.get("regime", "")).strip(),
                    "slippage_pre_pct": str(row.get("slippage_pre_pct", "")).strip(),
                }
            )

    headers = [
        "audit_id",
        "order_id",
        "stake_real",
        "pnl_real",
        "roi_real_pct",
        "event_id",
        "league",
        "audited_at",
        "side",
        "regime",
        "slippage_pre_pct",
    ]
    for target in [out_csv, out_enriched_csv]:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=headers)
            wr.writeheader()
            wr.writerows(rows)

    ts_vals = [r["audited_at"] for r in rows if r["audited_at"]]
    ts_min = min(ts_vals) if ts_vals else ""
    ts_max = max(ts_vals) if ts_vals else ""
    return len(rows), ts_min, ts_max


def main() -> int:
    args = parse_args()
    db = os.environ.get("DATABASE_URL", "").strip()
    if not db:
        raise SystemExit("DATABASE_URL nao definido")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    old = Path("/tmp/projecao_por_aposta.csv")
    bak = Path(f"/tmp/projecao_por_aposta__backup_{ts}.csv")
    out = Path(f"/tmp/projecao_por_aposta__regen_db_{ts}.csv")
    out_enr = Path(f"/tmp/projecao_por_aposta_enriquecido__regen_db_{ts}.csv")
    ledger_norm = Path(f"/tmp/ledger_norm_{ts}.csv")
    audit_csv = Path(f"/tmp/audit_for_proj_{ts}.csv")

    if old.exists():
        shutil.copy2(old, bak)
        print(f"[OK] backup: {bak}")

    ledger_source = ""
    used_db_fallback = False

    try_csv_first = int(args.prefer_db_ledger) != 1
    if try_csv_first:
        cands = discover_balance_candidates(args.balance_csv, args.search_root, args.max_depth)
        for cand in cands:
            order_col, value_col, _ = choose_csv_ledger_columns(cand)
            if not order_col or not value_col:
                continue
            try:
                n = normalize_ledger_from_csv(cand, ledger_norm)
            except Exception:
                continue
            if n > 0:
                ledger_source = str(cand)
                print(f"[OK] ledger CSV: {cand} (order_col={order_col}, value_col={value_col}, n={n})")
                break

    if not ledger_source:
        table, n = normalize_ledger_from_db(db, ledger_norm)
        ledger_source = f"db:{table}"
        used_db_fallback = True
        print(f"[OK] ledger DB fallback: {table} (n={n})")

    export_audit_csv(db, audit_csv)
    print(f"[OK] auditoria exportada: {audit_csv}")

    nrows, ts_min, ts_max = join_audit_and_ledger(audit_csv, ledger_norm, out, out_enr)
    if nrows <= 0:
        raise RuntimeError("Join auditoria x ledger gerou 0 linhas.")

    print(f"[OK] saida: {out}")
    print(f"[OK] enriquecido: {out_enr}")
    print(f"[OK] linhas: {nrows}")
    print(f"[OK] periodo: {ts_min or 'NA'} -> {ts_max or 'NA'}")
    print(f"[OK] fonte_ledger: {ledger_source}")
    if used_db_fallback:
        print("[WARN] balance/ledger CSV nao encontrado; usado fallback via tabela DB.")

    if int(args.write_default) == 1:
        shutil.copy2(out, old)
        print(f"[OK] atualizado /tmp/projecao_por_aposta.csv com {out.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
