"""Settlement contract for Friendly analysis.

Statuses: OPEN | SETTLED_DECIDED | VOID_PUSH | MISSING | UNRECONCILED

roi_resolved = pnl_resolved / stake_resolved_total  (void in denominator)
roi_decided_ex_void = pnl_decided_ex_void / stake_decided_ex_void
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    from ops.accounting_status import order_id_key
except Exception:  # pragma: no cover
    def order_id_key(x: Any) -> str:  # type: ignore
        return str(x or "").strip()


VOID_EPS = 1e-9


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or str(x).strip() == "":
            return None
        return float(str(x).replace(",", "."))
    except Exception:
        return None


def load_pnl_by_order(balance_paths: Iterable[Path]) -> Dict[str, float]:
    """Aggregate amount by order_id across one or more balance CSVs. Missing ≠ 0."""
    out: Dict[str, float] = {}
    for path in balance_paths:
        if not path or not Path(path).exists():
            continue
        with Path(path).open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                oid = order_id_key(row.get("order id") or row.get("order_id") or row.get("orderid"))
                if not oid:
                    continue
                # Exclude deposits/withdrawals heuristics
                typ = str(row.get("type") or "").strip().lower()
                note = str(row.get("note") or "").lower()
                if typ in {"deposit", "withdrawal", "withdraw"}:
                    continue
                if "deposit" in note and "order" not in note:
                    continue
                amt = safe_float(row.get("amount"))
                if amt is None:
                    continue
                out[oid] = float(out.get(oid, 0.0)) + float(amt)
    return out


def load_open_oids(open_paths: Iterable[Path]) -> Set[str]:
    ids: Set[str] = set()
    for path in open_paths:
        if not path or not Path(path).exists():
            continue
        with Path(path).open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                oid = order_id_key(row.get("order id") or row.get("order_id"))
                if oid:
                    ids.add(oid)
    return ids


def classify_settlement(
    *,
    order_id: str,
    pnl: Optional[float],
    in_open: bool,
    has_accounting_row: bool,
) -> Tuple[str, Optional[float]]:
    """Return (settlement_status, pnl_or_none). Never coerce missing to 0."""
    if not order_id:
        return "UNRECONCILED", None
    if in_open:
        return "OPEN", None
    if has_accounting_row and pnl is not None:
        if abs(float(pnl)) <= VOID_EPS:
            return "VOID_PUSH", 0.0
        return "SETTLED_DECIDED", float(pnl)
    if has_accounting_row and pnl is None:
        return "UNRECONCILED", None
    return "MISSING", None


def attach_settlement(
    orders: List[Dict[str, Any]],
    *,
    pnl_by_oid: Dict[str, float],
    open_oids: Set[str],
) -> List[Dict[str, Any]]:
    out = []
    for o in orders:
        oid = order_id_key(o.get("order_id"))
        has = oid in pnl_by_oid
        pnl = pnl_by_oid.get(oid) if has else None
        status, pnl_out = classify_settlement(
            order_id=oid,
            pnl=pnl,
            in_open=oid in open_oids,
            has_accounting_row=has,
        )
        row = dict(o)
        row["settlement_status"] = status
        row["pnl"] = pnl_out  # None when OPEN/MISSING/UNRECONCILED
        row["accounting_joined"] = bool(has)
        out.append(row)
    return out


def performance_block(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute stake/pnl/roi metrics. Missing pnl never treated as zero loss."""
    n = len(rows)
    events = {str(r.get("event_id") or "") for r in rows if r.get("event_id")}
    stake_placed = sum(float(r["stake"]) for r in rows if r.get("stake") is not None)
    open_rows = [r for r in rows if r.get("settlement_status") == "OPEN"]
    decided = [r for r in rows if r.get("settlement_status") == "SETTLED_DECIDED"]
    voids = [r for r in rows if r.get("settlement_status") == "VOID_PUSH"]
    missing = [r for r in rows if r.get("settlement_status") == "MISSING"]
    unrec = [r for r in rows if r.get("settlement_status") == "UNRECONCILED"]

    stake_open = sum(float(r["stake"]) for r in open_rows if r.get("stake") is not None)
    stake_void = sum(float(r["stake"]) for r in voids if r.get("stake") is not None)
    stake_decided = sum(float(r["stake"]) for r in decided if r.get("stake") is not None)
    pnl_decided = sum(float(r["pnl"]) for r in decided if r.get("pnl") is not None)
    # resolved = decided + void
    stake_resolved = stake_decided + stake_void
    pnl_resolved = pnl_decided + 0.0  # void pnl = 0

    roi_resolved = (pnl_resolved / stake_resolved) if stake_resolved > 0 else None
    roi_ex_void = (pnl_decided / stake_decided) if stake_decided > 0 else None

    n_accounted = sum(1 for r in rows if r.get("accounting_joined"))
    accounting_coverage = (n_accounted / n) if n else None

    n_resolved = len(decided) + len(voids)
    maturity = (
        "FULLY_SETTLED"
        if n and not open_rows and not missing and not unrec
        else (
            "OPEN_COHORT"
            if open_rows and not decided and not voids
            else ("PARTIALLY_SETTLED" if n else "EMPTY")
        )
    )

    return {
        "n_live_ok": n,
        "n_events": len(events),
        "stake_placed": stake_placed,
        "n_open": len(open_rows),
        "stake_open": stake_open,
        "n_settled_decided": len(decided),
        "n_void_push": len(voids),
        "n_missing": len(missing),
        "n_unreconciled": len(unrec),
        "stake_resolved_total": stake_resolved,
        "stake_decided_ex_void": stake_decided,
        "stake_void": stake_void,
        "pnl_resolved": pnl_resolved if (decided or voids) else None,
        "pnl_decided_ex_void": pnl_decided if decided else None,
        "roi_resolved": roi_resolved,
        "roi_decided_ex_void": roi_ex_void,
        "accounting_coverage": accounting_coverage,
        "maturity": maturity,
        "n_resolved": n_resolved,
        "notes": [
            "roi_resolved includes void stake in denominator (official contract)",
            "OPEN excluded from P&L (not counted as loss)",
            "MISSING pnl is null — never coerced to zero",
        ],
    }


def sample_gate(n_settled: int) -> str:
    if n_settled < 30:
        return "VERY_LOW_N"
    if n_settled < 100:
        return "INSUFFICIENT_N"
    if n_settled < 250:
        return "FIRST_READING"
    return "RELIABLE_READING_CANDIDATE"
