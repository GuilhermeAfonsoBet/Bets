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
    "provider_order_id",
    "exchange_order_id",
    "book_order_id",
    "ticket_id",
    "ticket",
    "external_id",
    "bet_id",
]
AUDIT_REF_COL_CANDIDATES = [
    "audit_id",
    "auditid",
    "audit_result_id",
    "auditresultid",
    "betslip_audit_id",
    "betslipauditid",
    "betslip_audit_result_id",
    "betslipauditresultid",
    "id_audit",
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
TEXT_TYPES = {"text", "character varying", "character"}


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
    p.add_argument(
        "--allow-auditid-fallback",
        type=int,
        default=0,
        help="Se 1, permite usar audit_id como fallback de order_id na auditoria.",
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


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for _ in rd:
            n += 1
    return n


def count_csv_nonempty(path: Path, field: str) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            if str(row.get(field, "")).strip():
                n += 1
    return n


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


def pick_orderish_column(columns: Iterable[str]) -> Optional[str]:
    best = None
    best_score = -1
    for c in columns:
        n = _norm(c)
        if "order" not in n:
            continue
        score = 10
        if "id" in n:
            score += 10
        if "external" in n or "provider" in n:
            score += 3
        if score > best_score:
            best_score = score
            best = c
    return best


def pick_auditref_column(columns: Iterable[str]) -> Optional[str]:
    hit = pick_existing(columns, AUDIT_REF_COL_CANDIDATES)
    if hit:
        return hit
    best = None
    best_score = -1
    for c in columns:
        n = _norm(c)
        if "audit" not in n:
            continue
        score = 10
        if "id" in n:
            score += 10
        if score > best_score:
            best_score = score
            best = c
    return best


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
    scored: List[Tuple[int, float, Path]] = []
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
            pf = f.resolve()
            name = pf.name.lower()
            score = 0
            # prioriza fonte "real" de balance em relacao a artefatos temporarios gerados por este script
            if "balance" in name:
                score += 50
            if "accounting" in str(pf).lower():
                score += 20
            if "ledger_norm_" in name:
                score -= 40
            scored.append((score, pf.stat().st_mtime, pf))

    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [p for _, __, p in scored]


def sniff_reader(path: Path) -> csv.DictReader:
    sample = path.read_text(encoding="utf-8", errors="ignore")[:8192]
    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return csv.DictReader(lines, dialect=dialect)


def choose_csv_ledger_columns(path: Path) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    try:
        rd = sniff_reader(path)
    except Exception:
        return None, None, None, None
    fields = list(rd.fieldnames or [])
    if not fields:
        return None, None, None, None
    order_col = pick_existing(fields, ORDER_COL_CANDIDATES) or pick_orderish_column(fields)
    pnl_col = pick_existing(fields, CSV_VALUE_COL_CANDIDATES)
    type_col = pick_existing(fields, TYPE_COL_CANDIDATES)
    audit_col = pick_existing(fields, AUDIT_REF_COL_CANDIDATES) or pick_auditref_column(fields)
    if not audit_col and "id" in fields and "id" != order_col:
        audit_col = "id"
    return order_col, pnl_col, type_col, audit_col


def normalize_ledger_from_csv(in_csv: Path, out_csv: Path) -> Tuple[int, int]:
    rd = sniff_reader(in_csv)
    fields = list(rd.fieldnames or [])
    order_col = pick_existing(fields, ORDER_COL_CANDIDATES) or pick_orderish_column(fields)
    pnl_col = pick_existing(fields, CSV_VALUE_COL_CANDIDATES)
    type_col = pick_existing(fields, TYPE_COL_CANDIDATES)
    audit_col = pick_existing(fields, AUDIT_REF_COL_CANDIDATES) or pick_auditref_column(fields)
    if not audit_col and "id" in fields and "id" != order_col:
        audit_col = "id"
    if not pnl_col:
        raise RuntimeError(
            f"CSV {in_csv} nao parece ledger valido. value_col={pnl_col}"
        )

    agg_order: Dict[str, float] = defaultdict(float)
    agg_audit: Dict[str, float] = defaultdict(float)
    for row in rd:
        if type_col:
            tx = str(row.get(type_col, "")).strip().lower()
            if tx and EXCLUDE_TX_PAT.search(tx):
                continue
        pnl = parse_float(row.get(pnl_col))
        if pnl is None:
            continue
        if order_col:
            oid = str(row.get(order_col, "")).strip()
            if oid:
                agg_order[oid] += pnl
        if audit_col:
            aid = str(row.get(audit_col, "")).strip()
            if aid:
                agg_audit[aid] += pnl

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["order_id", "audit_id", "pnl_real"])
        for oid, pnl in agg_order.items():
            wr.writerow([oid, "", f"{pnl:.10f}"])
        for aid, pnl in agg_audit.items():
            wr.writerow(["", aid, f"{pnl:.10f}"])
    return len(agg_order), len(agg_audit)


def discover_ledger_table(db_url: str) -> Tuple[str, str, str, Optional[str], Optional[str]]:
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

    scored: List[Tuple[int, str, str, str, str, Optional[str], Optional[str]]] = []
    for (sc, tb), cols in tbl_cols.items():
        colnames = list(cols.keys())
        order_col = pick_existing(colnames, ORDER_COL_CANDIDATES) or pick_orderish_column(colnames)
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
        audit_col = pick_existing(colnames, AUDIT_REF_COL_CANDIDATES) or pick_auditref_column(colnames)
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
        scored.append((score, sc, tb, order_col, value_col, type_col, audit_col))

    if not scored:
        raise RuntimeError("Nao encontrei tabela de ledger no DB com (order_id + coluna numerica de valor).")
    scored.sort(reverse=True)
    _, sc, tb, ord_col, val_col, type_col, audit_col = scored[0]
    return f"{sc}.{tb}", ord_col, val_col, type_col, audit_col


def normalize_ledger_from_db(db_url: str, out_csv: Path) -> Tuple[str, int]:
    table, order_col, value_col, type_col, audit_col = discover_ledger_table(db_url)
    schema, tb = table.split(".", 1)
    q_table = f"{quote_ident(schema)}.{quote_ident(tb)}"
    q_oid = quote_ident(order_col)
    q_val = quote_ident(value_col)
    aid_expr = f"{quote_ident(audit_col)}::text" if audit_col else "NULL::text"

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
      {aid_expr} AS audit_id,
      SUM(COALESCE({q_val}::double precision,0)) AS pnl_real
    FROM {q_table}
    WHERE {where_sql}
    GROUP BY 1,2
    """
    psql_copy_csv(db_url, sql, out_csv)
    count = 0
    with out_csv.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for _ in rd:
            count += 1
    return table, count


def export_audit_csv(
    db_url: str,
    out_csv: Path,
    *,
    allow_auditid_fallback: bool = False,
    require_order: bool = True,
    require_positive_stake: bool = True,
    require_status_ok: bool = True,
) -> int:
    schema, cols = list_table_columns(db_url, "betslip_audit_results")
    colnames = list(cols.keys())
    q_table = f"{quote_ident(schema)}.{quote_ident('betslip_audit_results')}"

    order_col = pick_existing(colnames, ORDER_COL_CANDIDATES) or pick_orderish_column(colnames)
    audit_col = pick_existing(colnames, ["audit_id", "id"]) or pick_auditref_column(colnames)
    event_col = pick_existing(colnames, ["event_id", "match_id", "fixture_id", "game_id"])
    league_col = pick_existing(colnames, ["league", "league_name", "competition", "tournament"])
    ts_col = pick_existing(colnames, ["audited_at", "created_at", "updated_at"])
    side_col = pick_existing(colnames, ["side", "direction", "exec_side", "pick_side"])
    regime_col = pick_existing(colnames, ["regime", "market_regime", "market_phase", "phase"])
    status_col = pick_existing(colnames, ["status"])
    slip_col = pick_existing(colnames, ["slippage_pre_pct", "slippage_raw_pct", "slippage"])
    json_cols = [c for c, typ in cols.items() if typ in {"json", "jsonb"}]
    if "hypothesis_details" in json_cols:
        json_cols.remove("hypothesis_details")
        json_cols.insert(0, "hypothesis_details")
    text_blob_cols = [
        c
        for c, typ in cols.items()
        if typ in TEXT_TYPES and any(k in _norm(c) for k in ["detail", "payload", "meta", "raw", "json", "response"])
    ]
    json_col = json_cols[0] if json_cols else None

    stake_candidates = [
        c
        for c in [
            "stake_real",
            "exposure_real",
            "exposure",
            "stake",
            "stake_usd",
            "bet_amount",
            "wager",
            "wager_amount",
            "size",
            "amount",
        ]
        if c in colnames
    ]
    for c, typ in cols.items():
        if typ not in NUMERIC_TYPES:
            continue
        n = _norm(c)
        if any(tok in n for tok in ["stake", "exposure", "wager"]):
            if c not in stake_candidates:
                stake_candidates.append(c)
    stake_terms = [f"NULLIF({quote_ident(c)}::double precision,0)" for c in stake_candidates]
    limit_col = pick_existing(colnames, ["betslip_limit", "limit"])
    stake_pct_col = pick_existing(colnames, ["stake_pct_of_limit", "stake_pct", "stake_percentage"])
    if limit_col and stake_pct_col:
        q_lim = quote_ident(limit_col)
        q_pct = quote_ident(stake_pct_col)
        stake_terms.append(
            f"NULLIF({q_lim}::double precision * "
            f"(CASE WHEN abs({q_pct}::double precision)<=1 THEN {q_pct}::double precision "
            f"ELSE {q_pct}::double precision/100.0 END), 0)"
        )
    if json_col:
        q_json = quote_ident(json_col)
        lim_sql = f"{quote_ident(limit_col)}::double precision" if limit_col else "NULL::double precision"
        stake_terms.extend(
            [
                f"NULLIF(({q_json} #>> '{{value_sizing,stake_real}}')::double precision,0)",
                f"NULLIF(({q_json} #>> '{{value_sizing,stake}}')::double precision,0)",
                f"NULLIF(({q_json} #>> '{{value_sizing,exposure}}')::double precision,0)",
                f"NULLIF(({q_json} #>> '{{finance,stake}}')::double precision,0)",
                f"NULLIF(({q_json} #>> '{{finance,stake_real}}')::double precision,0)",
                f"NULLIF(({q_json} #>> '{{finance,exposure}}')::double precision,0)",
                f"NULLIF( COALESCE({lim_sql}, ({q_json} #>> '{{finance,limit}}')::double precision, ({q_json} #>> '{{value_sizing,limit}}')::double precision ) "
                f"* (CASE WHEN abs(COALESCE(({q_json} #>> '{{finance,stake_pct_of_limit}}')::double precision, ({q_json} #>> '{{value_sizing,stake_pct_of_limit}}')::double precision, 0))<=1 "
                f"THEN COALESCE(({q_json} #>> '{{finance,stake_pct_of_limit}}')::double precision, ({q_json} #>> '{{value_sizing,stake_pct_of_limit}}')::double precision, 0) "
                f"ELSE COALESCE(({q_json} #>> '{{finance,stake_pct_of_limit}}')::double precision, ({q_json} #>> '{{value_sizing,stake_pct_of_limit}}')::double precision, 0)/100.0 END), 0)",
            ]
        )
    if stake_terms:
        stake_expr = f"COALESCE({', '.join(stake_terms)}, 0.0)"
    else:
        stake_expr = "0.0::double precision"

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

    def order_expr() -> str:
        if order_col:
            return f"NULLIF({quote_ident(order_col)}::text, '')"
        terms: List[str] = []
        for jcol in json_cols:
            qj = quote_ident(jcol)
            qj_b = f"({qj})::jsonb"
            # caminhos comuns + busca recursiva por chave
            terms.extend(
                [
                    f"NULLIF({qj} #>> '{{order_id}}', '')",
                    f"NULLIF({qj} #>> '{{orderId}}', '')",
                    f"NULLIF({qj} #>> '{{order,id}}', '')",
                    f"NULLIF({qj} #>> '{{execution,order_id}}', '')",
                    f"NULLIF({qj} #>> '{{execution,orderId}}', '')",
                    f"NULLIF({qj} #>> '{{bridge,order_id}}', '')",
                    f"NULLIF({qj} #>> '{{bridge,orderId}}', '')",
                    f"NULLIF({qj} #>> '{{bet,order_id}}', '')",
                    f"NULLIF({qj} #>> '{{bet,orderId}}', '')",
                    f"NULLIF({qj} #>> '{{order id}}', '')",
                    f"NULLIF({qj} #>> '{{value_sizing,order_id}}', '')",
                    f"NULLIF({qj} #>> '{{value_sizing,orderId}}', '')",
                    f"NULLIF({qj} #>> '{{execution_result,order_id}}', '')",
                    f"NULLIF({qj} #>> '{{execution_result,orderId}}', '')",
                    f"NULLIF({qj} #>> '{{provider,order_id}}', '')",
                    f"NULLIF({qj} #>> '{{provider,orderId}}', '')",
                    f"NULLIF({qj} #>> '{{metadata,order_id}}', '')",
                    f"NULLIF({qj} #>> '{{metadata,orderId}}', '')",
                    f"NULLIF(jsonb_path_query_first({qj_b}, '$.**.order_id') #>> '{{}}', '')",
                    f"NULLIF(jsonb_path_query_first({qj_b}, '$.**.orderId') #>> '{{}}', '')",
                    f"NULLIF(jsonb_path_query_first({qj_b}, '$.**.\"order id\"') #>> '{{}}', '')",
                    f"NULLIF(substring({qj}::text from '(?i)\"order[_ ]?id\"\\s*:\\s*\"([^\"]+)\"'), '')",
                    f"NULLIF(substring({qj}::text from '(?i)\"order[_ ]?id\"\\s*:\\s*([0-9]+)'), '')",
                ]
            )
        for tcol in text_blob_cols:
            qt = quote_ident(tcol)
            terms.extend(
                [
                    f"NULLIF(substring({qt}::text from '(?i)\"order[_ ]?id\"\\s*[:=]\\s*\"([^\"]+)\"'), '')",
                    f"NULLIF(substring({qt}::text from '(?i)\"order[_ ]?id\"\\s*[:=]\\s*([0-9]+)'), '')",
                ]
            )
        if allow_auditid_fallback and audit_col:
            # fallback extremo: usa audit_id como chave de ordem
            terms.append(f"NULLIF({quote_ident(audit_col)}::text, '')")
        if not terms:
            raise RuntimeError(
                "Nao consegui montar chave de order para betslip_audit_results "
                "(sem coluna order e sem hypothesis_details/audit_id)."
            )
        return f"COALESCE({', '.join(terms)})"

    order_expr_sql = order_expr()

    where_parts: List[str] = []
    if require_order:
        where_parts.append(f"{order_expr_sql} IS NOT NULL")
    if require_positive_stake:
        where_parts.append(f"{stake_expr} > 0")
    if require_status_ok and status_col:
        where_parts.append(f"UPPER({quote_ident(status_col)}::text)='OK'")
    where_sql = " AND ".join(where_parts) if where_parts else "TRUE"

    sql = f"""
    SELECT
      {col_or_null(audit_col, 'audit_id')},
      {order_expr_sql} AS "order_id",
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
    return count_csv_rows(out_csv)


def discover_and_export_audit_order_map(db_url: str, out_csv: Path) -> Tuple[Optional[str], int]:
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

    candidates: List[Tuple[int, str, str, str, str]] = []
    for (sc, tb), cols in tbl_cols.items():
        if _norm(tb) == _norm("betslip_audit_results"):
            continue
        colnames = list(cols.keys())
        audit_ref_col = pick_auditref_column(colnames)
        if not audit_ref_col:
            continue
        order_col = pick_existing(colnames, ORDER_COL_CANDIDATES) or pick_orderish_column(colnames)
        if not order_col:
            continue
        score = 0
        n = tb.lower()
        if "bridge" in n:
            score += 30
        if "executor" in n:
            score += 20
        if "audit" in n:
            score += 15
        if "order" in n:
            score += 10
        if sc == "public":
            score += 10
        candidates.append((score, sc, tb, audit_ref_col, order_col))

    candidates.sort(reverse=True)
    for _, sc, tb, audit_col, order_col in candidates[:20]:
        q_table = f"{quote_ident(sc)}.{quote_ident(tb)}"
        q_a = quote_ident(audit_col)
        q_o = quote_ident(order_col)
        q_count = f"SELECT COUNT(*) FROM {q_table} WHERE {q_a} IS NOT NULL AND {q_o} IS NOT NULL"
        try:
            n_pairs = int((run_psql_query(db_url, q_count).strip() or "0"))
        except Exception:
            continue
        if n_pairs <= 0:
            continue
        q_export = f"""
        SELECT
          {q_a}::text AS audit_id,
          {q_o}::text AS order_id
        FROM {q_table}
        WHERE {q_a} IS NOT NULL
          AND {q_o} IS NOT NULL
        """
        try:
            psql_copy_csv(db_url, q_export, out_csv)
        except Exception:
            continue
        n = count_csv_rows(out_csv)
        if n > 0:
            src = f"{sc}.{tb} ({audit_col}->{order_col})"
            return src, n
    return None, 0


def _key_norm_compact(v: str) -> str:
    return _norm(v)


def _key_norm_digits(v: str) -> str:
    return "".join(ch for ch in str(v) if ch.isdigit())


def _build_unique_map(
    rows: Iterable[Dict[str, object]],
    key_fn,
    *,
    id_field: str,
) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    dup = set()
    for r in rows:
        k = key_fn(str(r.get(id_field, "")).strip())
        if not k:
            continue
        if k in out:
            # chave ambigua -> descarta para evitar match errado
            dup.add(k)
            continue
        out[k] = r
    for k in dup:
        out.pop(k, None)
    return out


def join_audit_and_ledger(
    audit_csv: Path,
    ledger_csv: Path,
    out_csv: Path,
    out_enriched_csv: Path,
    audit_order_map_csv: Optional[Path] = None,
) -> Tuple[int, float, float]:
    pnl_by_oid: Dict[str, float] = defaultdict(float)
    pnl_by_aid: Dict[str, float] = defaultdict(float)
    with ledger_csv.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            oid = str(row.get("order_id", "")).strip()
            aid = str(row.get("audit_id", "")).strip()
            pnl = parse_float(row.get("pnl_real"))
            if pnl is None:
                continue
            if oid:
                pnl_by_oid[oid] += pnl
            if aid:
                pnl_by_aid[aid] += pnl

    ledger_rows = [{"order_id": oid, "pnl_real": pnl} for oid, pnl in pnl_by_oid.items()]
    ledger_audit_rows = [{"audit_id": aid, "pnl_real": pnl} for aid, pnl in pnl_by_aid.items()]

    audit_rows = []
    with audit_csv.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            audit_rows.append(dict(row))

    if audit_order_map_csv and audit_order_map_csv.exists():
        map_rows = []
        with audit_order_map_csv.open("r", encoding="utf-8", newline="") as f:
            rd = csv.DictReader(f)
            for r in rd:
                map_rows.append(dict(r))
        aid_to_order = _build_unique_map(map_rows, lambda s: str(s).strip(), id_field="audit_id")
        filled = 0
        for row in audit_rows:
            oid = str(row.get("order_id", "")).strip()
            if oid:
                continue
            aid = str(row.get("audit_id", "")).strip()
            m = aid_to_order.get(aid)
            if not m:
                continue
            moid = str(m.get("order_id", "")).strip()
            if not moid:
                continue
            row["order_id"] = moid
            filled += 1
        if filled > 0:
            print(f"[OK] order_id preenchido via mapa audit->order: +{filled} linhas")

    strategies = {
        "raw": lambda s: str(s).strip(),
        "compact": _key_norm_compact,
        "digits": _key_norm_digits,
    }

    best_name = ""
    best_matches = -1
    best_audit_map: Dict[str, Dict[str, object]] = {}
    best_ledger_map: Dict[str, Dict[str, object]] = {}
    for name, fn in strategies.items():
        a_map = _build_unique_map(audit_rows, fn, id_field="order_id")
        l_map = _build_unique_map(ledger_rows, fn, id_field="order_id")
        n_match = sum(1 for k in a_map.keys() if k in l_map)
        if n_match > best_matches:
            best_matches = n_match
            best_name = name
            best_audit_map = a_map
            best_ledger_map = l_map

    best_aid_name = ""
    best_aid_matches = -1
    best_aid_audit_map: Dict[str, Dict[str, object]] = {}
    best_aid_ledger_map: Dict[str, Dict[str, object]] = {}
    for name, fn in strategies.items():
        a_map = _build_unique_map(audit_rows, fn, id_field="audit_id")
        l_map = _build_unique_map(ledger_audit_rows, fn, id_field="audit_id")
        n_match = sum(1 for k in a_map.keys() if k in l_map)
        if n_match > best_aid_matches:
            best_aid_matches = n_match
            best_aid_name = name
            best_aid_audit_map = a_map
            best_aid_ledger_map = l_map

    rows = []
    matched_by_order = 0
    matched_by_audit = 0
    for row in audit_rows:
        st = parse_float(row.get("stake_real"))
        if st is None or st <= 0:
            continue

        pnl = None
        oid = str(row.get("order_id", "")).strip()
        if oid:
            ok = strategies.get(best_name, lambda s: str(s).strip())(oid)
            led = best_ledger_map.get(ok)
            if led:
                pnl = float(led["pnl_real"])
                matched_by_order += 1
        if pnl is None:
            aid = str(row.get("audit_id", "")).strip()
            if aid:
                ak = strategies.get(best_aid_name, lambda s: str(s).strip())(aid)
                led = best_aid_ledger_map.get(ak)
                if led:
                    pnl = float(led["pnl_real"])
                    matched_by_audit += 1
        if pnl is None:
            continue
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

    if best_name != "raw":
        print(f"[WARN] join por normalizacao de chave: strategy={best_name}, matched_keys={best_matches}")
    if best_aid_name and best_aid_name != "raw":
        print(f"[WARN] join(audit_id) por normalizacao: strategy={best_aid_name}, matched_keys={best_aid_matches}")
    if matched_by_audit > 0:
        print(f"[OK] join via audit_id: {matched_by_audit} linhas")
    if matched_by_order > 0:
        print(f"[OK] join via order_id: {matched_by_order} linhas")

    if not rows:
        a_samp = [str(r.get("order_id", "")).strip() for r in audit_rows[:5]]
        a_aid_samp = [str(r.get("audit_id", "")).strip() for r in audit_rows[:5]]
        l_samp = [str(r.get("order_id", "")).strip() for r in ledger_rows[:5]]
        l_aid_samp = [str(r.get("audit_id", "")).strip() for r in ledger_audit_rows[:5]]
        m_raw = sum(
            1
            for k in _build_unique_map(audit_rows, strategies["raw"], id_field="order_id").keys()
            if k in _build_unique_map(ledger_rows, strategies["raw"], id_field="order_id")
        )
        m_compact = sum(
            1
            for k in _build_unique_map(audit_rows, strategies["compact"], id_field="order_id").keys()
            if k in _build_unique_map(ledger_rows, strategies["compact"], id_field="order_id")
        )
        m_digits = sum(
            1
            for k in _build_unique_map(audit_rows, strategies["digits"], id_field="order_id").keys()
            if k in _build_unique_map(ledger_rows, strategies["digits"], id_field="order_id")
        )
        am_raw = sum(
            1
            for k in _build_unique_map(audit_rows, strategies["raw"], id_field="audit_id").keys()
            if k in _build_unique_map(ledger_audit_rows, strategies["raw"], id_field="audit_id")
        )
        am_compact = sum(
            1
            for k in _build_unique_map(audit_rows, strategies["compact"], id_field="audit_id").keys()
            if k in _build_unique_map(ledger_audit_rows, strategies["compact"], id_field="audit_id")
        )
        am_digits = sum(
            1
            for k in _build_unique_map(audit_rows, strategies["digits"], id_field="audit_id").keys()
            if k in _build_unique_map(ledger_audit_rows, strategies["digits"], id_field="audit_id")
        )
        raise RuntimeError(
            "Join auditoria x ledger gerou 0 linhas. "
            f"matches(raw/compact/digits)=({m_raw}/{m_compact}/{m_digits}). "
            f"matches_auditid(raw/compact/digits)=({am_raw}/{am_compact}/{am_digits}). "
            f"sample_audit_order={a_samp} sample_ledger_order={l_samp} "
            f"sample_audit_id={a_aid_samp} sample_ledger_audit_id={l_aid_samp}"
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
    audit_order_map_csv = Path(f"/tmp/audit_order_map_{ts}.csv")

    if old.exists():
        shutil.copy2(old, bak)
        print(f"[OK] backup: {bak}")

    ledger_source = ""
    used_db_fallback = False

    try_csv_first = int(args.prefer_db_ledger) != 1
    if try_csv_first:
        cands = discover_balance_candidates(args.balance_csv, args.search_root, args.max_depth)
        for cand in cands:
            order_col, value_col, _, audit_ref_col = choose_csv_ledger_columns(cand)
            if not value_col:
                continue
            try:
                n_order, n_auditref = normalize_ledger_from_csv(cand, ledger_norm)
            except Exception:
                continue
            n_total = int(n_order) + int(n_auditref)
            if n_total > 0:
                ledger_source = str(cand)
                print(
                    f"[OK] ledger CSV: {cand} "
                    f"(order_col={order_col}, audit_ref_col={audit_ref_col}, value_col={value_col}, "
                    f"n_order={n_order}, n_audit_ref={n_auditref})"
                )
                break

    if not ledger_source:
        table, n = normalize_ledger_from_db(db, ledger_norm)
        ledger_source = f"db:{table}"
        used_db_fallback = True
        print(f"[OK] ledger DB fallback: {table} (n={n})")

    n_audit = export_audit_csv(
        db,
        audit_csv,
        allow_auditid_fallback=bool(int(args.allow_auditid_fallback)),
        require_order=True,
        require_positive_stake=True,
        require_status_ok=True,
    )
    if n_audit == 0:
        print("[WARN] auditoria strict=0 linhas; tentando export relaxado (sem stake/status).")
        n_audit = export_audit_csv(
            db,
            audit_csv,
            allow_auditid_fallback=bool(int(args.allow_auditid_fallback)),
            require_order=True,
            require_positive_stake=False,
            require_status_ok=False,
        )
    if n_audit == 0:
        print("[WARN] auditoria ainda 0 linhas; tentando sem exigir order no SQL (diagnostico).")
        n_audit = export_audit_csv(
            db,
            audit_csv,
            allow_auditid_fallback=bool(int(args.allow_auditid_fallback)),
            require_order=False,
            require_positive_stake=False,
            require_status_ok=False,
        )
    print(f"[OK] auditoria exportada: {audit_csv} (n={n_audit})")
    order_nonempty = count_csv_nonempty(audit_csv, "order_id")
    stake_nonempty = count_csv_nonempty(audit_csv, "stake_real")
    print(f"[OK] auditoria com order_id preenchido: {order_nonempty}")
    print(f"[OK] auditoria com stake_real preenchido: {stake_nonempty}")

    need_map = (order_nonempty == 0) or (n_audit > 0 and order_nonempty < max(100, int(0.05 * n_audit)))
    if need_map:
        map_src, map_n = discover_and_export_audit_order_map(db, audit_order_map_csv)
        if map_src and map_n > 0:
            print(f"[OK] mapa audit->order: {audit_order_map_csv} (n={map_n}, source={map_src})")
        else:
            print("[WARN] nao encontrei tabela auxiliar de mapeamento audit->order.")

    nrows, ts_min, ts_max = join_audit_and_ledger(
        audit_csv,
        ledger_norm,
        out,
        out_enr,
        audit_order_map_csv=audit_order_map_csv if audit_order_map_csv.exists() else None,
    )
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
