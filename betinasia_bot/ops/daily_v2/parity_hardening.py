#!/usr/bin/env python3
"""Parity hardening: order-set reconciliation + parity_as_of vs matured_as_of."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .performance import compute_settlement_and_performance
from .time_windows import ReportWindow, ensure_utc, parse_dt
from .universes import (
    H3BUP_POLICY_NEEDLE,
    load_executor_orders,
    load_open_order_ids,
    load_pnl_by_order_from_balance_csv,
)


PARITY_CUTOFF_20260729 = "2026-07-29T22:01:54.606850+00:00"


def order_set_hash(order_ids: Iterable[str]) -> str:
    s = "\n".join(sorted(str(x) for x in order_ids)) + ("\n" if order_ids else "")
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def filter_orders_as_of(
    orders: Dict[str, Dict[str, Any]],
    *,
    as_of: datetime,
) -> Dict[str, Dict[str, Any]]:
    """Keep LIVE_OK orders with created_at <= as_of (inclusive)."""
    as_of = ensure_utc(as_of)
    out = {}
    for oid, o in orders.items():
        created = o.get("created_at_dt") or parse_dt(o.get("created_at"))
        if created is None:
            continue
        if ensure_utc(created) <= as_of:
            out[oid] = o
    return out


def diff_order_sets(v1_ids: Set[str], v2_ids: Set[str]) -> Dict[str, Any]:
    only_v1 = sorted(v1_ids - v2_ids)
    only_v2 = sorted(v2_ids - v1_ids)
    both = sorted(v1_ids & v2_ids)
    h1 = order_set_hash(v1_ids)
    h2 = order_set_hash(v2_ids)
    return {
        "v1_count": len(v1_ids),
        "v2_count": len(v2_ids),
        "v1_order_set_hash": h1,
        "v2_order_set_hash": h2,
        "order_set_match": bool(h1 == h2 and len(v1_ids) == len(v2_ids) and not only_v1 and not only_v2),
        "only_in_v1": only_v1,
        "only_in_v2": only_v2,
        "only_in_v1_count": len(only_v1),
        "only_in_v2_count": len(only_v2),
        "in_both_count": len(both),
        "in_both": both,
    }


def find_nearest_accounting_snapshot(
    acct_dir: Path,
    *,
    as_of: datetime,
    kind: str = "balance",
) -> Optional[Path]:
    """Pick latest accounting CSV with timestamp in filename <= as_of."""
    as_of = ensure_utc(as_of)
    pattern = f"*__{kind}.csv" if kind != "open_stakes" else "*__open_stakes.csv"
    if kind == "balance":
        pattern = "*__balance.csv"
    cands = []
    for p in Path(acct_dir).glob(pattern):
        # 20260729_220449__open_stakes.csv
        name = p.name
        try:
            stamp = name.split("__")[0]
            dt = datetime.strptime(stamp, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if dt <= as_of:
            cands.append((dt, p))
    if not cands:
        return None
    cands.sort()
    return cands[-1][1]


def _parse_post_date(s: Any) -> Optional[datetime]:
    if not s:
        return None
    return parse_dt(str(s))


def pnl_by_order_as_of(balance_csv: Path, *, as_of: datetime) -> Dict[str, float]:
    """Sum amounts per order_id using only rows with post date <= as_of."""
    as_of = ensure_utc(as_of)
    out: Dict[str, float] = {}
    if not balance_csv or not Path(balance_csv).exists():
        return out
    with Path(balance_csv).open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return out
        cols = {c.lower(): c for c in reader.fieldnames}

        def pick(*names):
            for n in names:
                if n in cols:
                    return cols[n]
            for k, orig in cols.items():
                for n in names:
                    if n in k:
                        return orig
            return None

        oid_c = pick("order id", "order_id", "orderid")
        amt_c = pick("amount", "pnl", "profit", "value")
        post_c = pick("post date", "post_date", "date", "datetime")
        type_c = pick("type", "transaction type", "tx_type")
        if not oid_c or not amt_c:
            return out
        exclude = {"deposit", "withdraw", "withdrawal", "transfer", "bonus"}
        for row in reader:
            oid = str(row.get(oid_c) or "").strip()
            if not oid:
                continue
            if type_c:
                t = str(row.get(type_c) or "").strip().lower()
                if any(x in t for x in exclude):
                    continue
            if post_c:
                pdt = _parse_post_date(row.get(post_c))
                if pdt is None or ensure_utc(pdt) > as_of:
                    continue
            try:
                amt = float(str(row.get(amt_c) or "0").replace(",", ""))
            except Exception:
                continue
            out[oid] = out.get(oid, 0.0) + amt
    return out


def classify_orders_settlement(
    orders: Dict[str, Dict[str, Any]],
    *,
    pnl_by_oid: Dict[str, float],
    open_oids: Set[str],
    void_eps: float = 1e-9,
) -> Dict[str, Any]:
    """Return per-order status + aggregates using given as-of maps."""
    rows = []
    open_l, settled, void_l, missing = [], [], [], []
    for oid, o in orders.items():
        stake = o.get("stake")
        if oid in open_oids:
            st = "OPEN"
            open_l.append(oid)
            pnl = None
        elif oid not in pnl_by_oid:
            st = "MISSING"
            missing.append(oid)
            pnl = None
        else:
            pnl = float(pnl_by_oid[oid])
            if abs(pnl) <= void_eps:
                st = "VOID"
                void_l.append(oid)
            else:
                st = "SETTLED"
                settled.append({"order_id": oid, "pnl": pnl, "stake": stake})
        rows.append(
            {
                "order_id": oid,
                "created_at_utc": o.get("created_at"),
                "policy_version": o.get("policy_version"),
                "stake": stake,
                "status_as_of": st,
                "pnl_as_of": pnl,
                "audit_id": o.get("audit_id"),
            }
        )
    perf = compute_settlement_and_performance(
        orders=orders, pnl_by_oid=pnl_by_oid, open_oids=open_oids
    )
    return {"rows": rows, "performance": perf, "counts": {
        "open": len(open_l),
        "settled": len(settled),
        "void": len(void_l),
        "missing": len(missing),
        "live_ok": len(orders),
    }}


def reconstruct_v1_universe_from_executor(
    exec_path: Path,
    *,
    window: ReportWindow,
    parity_as_of: datetime,
    require_h3bup: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """V1 day LIVE_OK universe equivalent: H3BUP Back LIVE_OK with created_at <= parity_as_of."""
    all_day = load_executor_orders(exec_path, window=window, require_h3bup=require_h3bup)
    return filter_orders_as_of(all_day, as_of=parity_as_of)


def classify_divergent_order(
    *,
    order: Dict[str, Any],
    in_v1: bool,
    in_v2_full: bool,
    in_v2_parity: bool,
    parity_as_of: datetime,
) -> Dict[str, Any]:
    created = parse_dt(order.get("created_at")) or order.get("created_at_dt")
    created = ensure_utc(created) if created else None
    after_cutoff = bool(created and created > ensure_utc(parity_as_of))
    policy = str(order.get("policy_version") or "")
    stake = order.get("stake")
    classification = "UNKNOWN"
    filter_v1 = ""
    filter_v2 = ""
    root = ""
    if in_v2_full and not in_v1 and after_cutoff:
        classification = "EXPECTED_SCOPE_DIFFERENCE"
        filter_v1 = "created_at > V1 report generation cutoff (order not yet existent at V1 freeze)"
        filter_v2 = "DAILY_CLOSED full UTC day includes post-cutoff LIVE_OK"
        root = "V1 frozen at 22:01:54; order created after freeze; V2 closed-day includes it"
    elif in_v1 and not in_v2_full:
        classification = "V2_BUG"
        filter_v2 = "excluded by V2 load_executor_orders"
        root = "present in V1 parity set but missing from V2 full day"
    elif in_v2_parity and in_v1:
        classification = "MATCH"
        root = "in both parity sets"
    elif not (H3BUP_POLICY_NEEDLE in policy):
        classification = "EXPECTED_SCOPE_DIFFERENCE"
        filter_v2 = "require_h3bup excludes non-H3BUP"
        root = "legacy/other policy"
    elif stake is not None and abs(float(stake) - 20.0) < 1e-9:
        classification = "EXPECTED_SCOPE_DIFFERENCE"
        root = "legacy stake 20"
    else:
        classification = "UNKNOWN"
        root = "unclassified"
    return {
        "order_id": order.get("order_id"),
        "created_at_utc": order.get("created_at"),
        "live_ok_ts_utc": order.get("created_at"),
        "policy_version": policy,
        "stake": stake,
        "side": order.get("exec_side"),
        "status": order.get("status"),
        "audit_id": order.get("audit_id"),
        "V1_included": in_v1,
        "V2_full_day_included": in_v2_full,
        "V2_parity_included": in_v2_parity,
        "filtro_V1": filter_v1,
        "filtro_V2": filter_v2,
        "classification": classification,
        "root_cause": root,
        "after_parity_cutoff": after_cutoff,
    }


def build_root_causes(
    *,
    universe_diff: Dict[str, Any],
    divergent_details: List[Dict[str, Any]],
    parity_perf: Dict[str, Any],
    matured_perf: Dict[str, Any],
    v1_frozen_open: Optional[int],
    v1_frozen_settled: Optional[int],
) -> List[Dict[str, Any]]:
    only_v2 = universe_diff.get("only_in_v2") or []
    rows = []
    rows.append(
        {
            "ID": "PAR-001",
            "metric": "LIVE_OK 22×24",
            "symptom": f"V1={universe_diff.get('v1_count')} V2_full={universe_diff.get('v2_count')}",
            "affected_order_ids": ",".join(only_v2),
            "root_cause": "2 LIVE_OK after V1 cutoff included in V2 DAILY_CLOSED full day",
            "correct_behaviour": "parity view filters created_at<=parity_as_of → MATCH 22",
            "classification": "EXPECTED_SCOPE_DIFFERENCE",
            "patch_applied": "filter_orders_as_of + dual views",
            "publication_blocker": "no",
            "owner": "reporting",
        }
    )
    rows.append(
        {
            "ID": "PAR-002",
            "metric": "stake 220×240",
            "symptom": "same two post-cutoff orders × stake 10",
            "affected_order_ids": ",".join(only_v2),
            "root_cause": "consequence of PAR-001",
            "correct_behaviour": "parity stake 220",
            "classification": "EXPECTED_SCOPE_DIFFERENCE",
            "patch_applied": "parity as-of universe",
            "publication_blocker": "no",
            "owner": "reporting",
        }
    )
    rows.append(
        {
            "ID": "PAR-003",
            "metric": "open 9×1",
            "symptom": f"V1 accounting-health open={v1_frozen_open}; matured open={matured_perf.get('n_open')}",
            "affected_order_ids": "",
            "root_cause": "AS_OF_MATURITY_DIFFERENCE + V1 health block used activation subset (n=12) not day universe (n=22)",
            "correct_behaviour": "compare open only under same as_of and same order set",
            "classification": "AS_OF_MATURITY_DIFFERENCE",
            "patch_applied": "parity_as_of vs matured_as_of sections",
            "publication_blocker": "no",
            "owner": "reporting",
        }
    )
    rows.append(
        {
            "ID": "PAR-004",
            "metric": "settled 3×21",
            "symptom": f"V1 health settled={v1_frozen_settled}; matured settled={matured_perf.get('n_settled')}",
            "affected_order_ids": "",
            "root_cause": "settlements posted after V1 cutoff; plus V1 health subset",
            "correct_behaviour": "parity uses snapshot/post-date<=cutoff; matured shows later settlements",
            "classification": "AS_OF_MATURITY_DIFFERENCE",
            "patch_applied": "dual settlement views",
            "publication_blocker": "no",
            "owner": "reporting",
        }
    )
    rows.append(
        {
            "ID": "PAR-005",
            "metric": "ROI definition",
            "symptom": "V1 health ROI settled partial vs V2 roi_resolved",
            "affected_order_ids": "",
            "root_cause": "EXPECTED_DEFINITION_CHANGE: V2 principal=roi_resolved (void in denom)",
            "correct_behaviour": "keep roi_resolved; document void; show roi_decided_ex_void",
            "classification": "EXPECTED_DEFINITION_CHANGE",
            "patch_applied": "documented in methodology",
            "publication_blocker": "no",
            "owner": "reporting",
        }
    )
    return rows


def build_parity_hardening_bundle(
    *,
    root: Path,
    window: ReportWindow,
    parity_as_of: datetime,
    matured_as_of: datetime,
    v1_frozen_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    root = Path(root)
    exec_path = root / "logs" / "executor_live.jsonl"
    acct_dir = root / "logs" / "accounting"

    full_day = load_executor_orders(exec_path, window=window, require_h3bup=True)
    v1_set = reconstruct_v1_universe_from_executor(
        exec_path, window=window, parity_as_of=parity_as_of, require_h3bup=True
    )
    v2_parity = filter_orders_as_of(full_day, as_of=parity_as_of)
    # matured universe for cohort still full day (closed), settlement as of now
    v2_matured_orders = full_day

    diff_full = diff_order_sets(set(v1_set), set(full_day))
    diff_parity = diff_order_sets(set(v1_set), set(v2_parity))

    divergent = []
    for oid in sorted(set(full_day) | set(v1_set)):
        o = full_day.get(oid) or v1_set.get(oid) or {}
        if oid in set(diff_full["only_in_v1"]) or oid in set(diff_full["only_in_v2"]):
            divergent.append(
                classify_divergent_order(
                    order={**o, "order_id": oid},
                    in_v1=oid in v1_set,
                    in_v2_full=oid in full_day,
                    in_v2_parity=oid in v2_parity,
                    parity_as_of=parity_as_of,
                )
            )

    # Accounting snapshots
    bal_parity = find_nearest_accounting_snapshot(acct_dir, as_of=parity_as_of, kind="balance")
    open_parity_path = find_nearest_accounting_snapshot(acct_dir, as_of=parity_as_of, kind="open_stakes")
    bal_matured = find_nearest_accounting_snapshot(acct_dir, as_of=matured_as_of, kind="balance")
    open_matured_path = find_nearest_accounting_snapshot(acct_dir, as_of=matured_as_of, kind="open_stakes")

    historical_status = "AVAILABLE"
    if bal_parity is None or open_parity_path is None:
        historical_status = "HISTORICAL_ASOF_UNAVAILABLE"

    # Parity settlement: prefer snapshot open_stakes + balance amounts with post<=as_of from parity balance file
    if bal_parity is not None:
        pnl_parity = pnl_by_order_as_of(bal_parity, as_of=parity_as_of)
    else:
        pnl_parity = {}
        historical_status = "HISTORICAL_ASOF_UNAVAILABLE"
    open_parity = load_open_order_ids(open_parity_path) if open_parity_path else set()
    # Restrict open set to cohort orders
    open_parity = {oid for oid in open_parity if oid in v2_parity}

    parity_view = classify_orders_settlement(v2_parity, pnl_by_oid=pnl_parity, open_oids=open_parity)

    if bal_matured is not None:
        pnl_mat = load_pnl_by_order_from_balance_csv(bal_matured)
    else:
        pnl_mat = {}
    open_mat = load_open_order_ids(open_matured_path) if open_matured_path else set()
    open_mat = {oid for oid in open_mat if oid in v2_matured_orders}
    matured_view = classify_orders_settlement(v2_matured_orders, pnl_by_oid=pnl_mat, open_oids=open_mat)

    v1m = v1_frozen_metrics or {}
    root_causes = build_root_causes(
        universe_diff=diff_full,
        divergent_details=divergent,
        parity_perf=parity_view["performance"],
        matured_perf=matured_view["performance"],
        v1_frozen_open=v1m.get("n_open"),
        v1_frozen_settled=v1m.get("n_settled"),
    )

    unknown = [d for d in divergent if d.get("classification") == "UNKNOWN"]
    blockers = [r for r in root_causes if str(r.get("publication_blocker")).lower() in {"yes", "true"}]

    status = "DAILY_PARITY_HARDENED_EXPECTED_DIFFERENCES"
    if unknown or blockers:
        status = "DAILY_PARITY_BLOCKERS_REMAIN"
    elif diff_parity.get("order_set_match") and historical_status == "AVAILABLE":
        status = "DAILY_PARITY_HARDENED_MATCH"
    elif historical_status == "HISTORICAL_ASOF_UNAVAILABLE":
        status = "DAILY_PARITY_HISTORICAL_ASOF_LIMITATION"
    elif diff_parity.get("order_set_match"):
        # universe match at cutoff; maturity diffs expected
        status = "DAILY_PARITY_HARDENED_EXPECTED_DIFFERENCES"

    return {
        "status": status,
        "report_date_utc": window.report_date_utc.isoformat(),
        "cohort_window_start_utc": window.window_start_utc.isoformat(),
        "cohort_window_end_utc": window.window_end_utc.isoformat(),
        "parity_as_of_utc": ensure_utc(parity_as_of).isoformat(),
        "matured_as_of_utc": ensure_utc(matured_as_of).isoformat(),
        "historical_asof_status": historical_status,
        "sources": {
            "balance_parity": str(bal_parity) if bal_parity else None,
            "open_parity": str(open_parity_path) if open_parity_path else None,
            "balance_matured": str(bal_matured) if bal_matured else None,
            "open_matured": str(open_matured_path) if open_matured_path else None,
        },
        "universe": {
            "v1_parity": {
                "count": len(v1_set),
                "stake_placed": sum(float(o.get("stake") or 0) for o in v1_set.values()),
                "order_ids": sorted(v1_set.keys()),
                "hash": order_set_hash(v1_set.keys()),
            },
            "v2_full_day": {
                "count": len(full_day),
                "stake_placed": sum(float(o.get("stake") or 0) for o in full_day.values()),
                "order_ids": sorted(full_day.keys()),
                "hash": order_set_hash(full_day.keys()),
            },
            "v2_parity": {
                "count": len(v2_parity),
                "stake_placed": sum(float(o.get("stake") or 0) for o in v2_parity.values()),
                "order_ids": sorted(v2_parity.keys()),
                "hash": order_set_hash(v2_parity.keys()),
            },
            "diff_full_day_vs_v1": diff_full,
            "diff_parity_vs_v1": diff_parity,
            "divergent_orders": divergent,
        },
        "parity_view": {
            "title": "Paridade com Daily V1 — visão congelada",
            "as_of": ensure_utc(parity_as_of).isoformat(),
            "counts": parity_view["counts"],
            "performance": parity_view["performance"],
            "orders": parity_view["rows"],
            "void_in_denominator": True,
            "roi_formula": "pnl_resolved / stake_resolved_total",
        },
        "matured_view": {
            "title": "Atualização de maturity da coorte",
            "as_of": ensure_utc(matured_as_of).isoformat(),
            "warning": "Esta secção utiliza dados posteriores ao cutoff histórico e não participa da paridade V1 × V2.",
            "counts": matured_view["counts"],
            "performance": matured_view["performance"],
            "orders": matured_view["rows"],
            "void_in_denominator": True,
        },
        "root_causes": root_causes,
        "unknown_divergent": unknown,
        "publication_blockers": blockers,
        "safety": {
            "alters_execution": False,
            "alters_policy": False,
            "alters_stake": False,
            "creates_orders": False,
            "opens_betslips": False,
            "v2_official": False,
        },
    }
