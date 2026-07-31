"""Performance / ROI / ROIw contracts for Daily V2."""

from __future__ import annotations

from typing import Any, Dict, Optional, Set

from .statuses import metric_envelope


def compute_settlement_and_performance(
    *,
    orders: Dict[str, Dict[str, Any]],
    pnl_by_oid: Dict[str, float],
    open_oids: Set[str],
    accounting_health_status: Optional[str] = None,
    void_eps: float = 1e-9,
) -> Dict[str, Any]:
    """Separate open / settled / void / missing; ROI settled; ROIw v1 and v2."""
    if accounting_health_status in {"STALE", "FAILED"}:
        stale_block = metric_envelope(
            value=None,
            unit="fraction",
            n=0,
            status="UNAVAILABLE_STALE",
            metric_version="v2.0",
            source="accounting",
            notes=[f"accounting_health={accounting_health_status}"],
        )
        return {
            "live_ok_total": len(orders),
            "open": [],
            "settled": [],
            "void_push": [],
            "missing_accounting": [],
            "roi_settled": stale_block,
            "roiw_total_v1": metric_envelope(
                status="UNAVAILABLE_STALE", unit="percent", metric_version="v1.0", source="accounting"
            ),
            "roiw_total_v2": metric_envelope(
                status="UNAVAILABLE_STALE", unit="percent", metric_version="v2.0", source="accounting"
            ),
            "maturity_status": "OPEN_COHORT" if orders else "FULLY_SETTLED",
        }

    open_list = []
    settled = []
    void_push = []
    missing = []

    for oid, o in orders.items():
        stake = o.get("stake")
        if oid in open_oids:
            open_list.append(oid)
            continue
        if oid not in pnl_by_oid:
            missing.append(oid)
            continue
        pnl = float(pnl_by_oid[oid])
        if abs(pnl) <= void_eps:
            void_push.append({"order_id": oid, "pnl": pnl, "stake": stake})
        else:
            settled.append({"order_id": oid, "pnl": pnl, "stake": stake})

    # ROI settled: settled + void (stake in denom, pnl 0 for void)
    settled_like = settled + void_push
    pnl_sum = 0.0
    stake_sum = 0.0
    n_stake = 0
    for row in settled_like:
        pnl_sum += float(row["pnl"])
        if row.get("stake") is not None:
            stake_sum += float(row["stake"])
            n_stake += 1

    if not pnl_by_oid and orders:
        roi_status = "MISSING"
        roi = metric_envelope(status="MISSING", unit="fraction", n=0, source="accounting", notes=["no pnl map"])
    elif stake_sum > 0:
        roi_val = pnl_sum / stake_sum
        roi = metric_envelope(
            value=roi_val,
            unit="fraction",
            n=len(settled_like),
            numerator=pnl_sum,
            denominator=stake_sum,
            coverage_pct=(100.0 * len(settled_like) / len(orders)) if orders else None,
            status="AVAILABLE" if not open_list and not missing else "PARTIAL",
            metric_version="v2.0",
            source="executor+accounting",
            notes=["open excluded from denominator", "void/push: pnl~0 stake included"],
        )
    elif not settled_like:
        roi = metric_envelope(
            value=None if (open_list or missing or not orders) else 0.0,
            unit="fraction",
            n=0,
            status="AVAILABLE" if (not orders) else ("PARTIAL" if open_list else "MISSING"),
            metric_version="v2.0",
            source="executor+accounting",
            notes=["empty settled set" if orders else "empty cohort"],
        )
    else:
        roi = metric_envelope(status="MISSING", unit="fraction", n=len(settled_like), notes=["stake missing"])

    # ROIw Total v1 (legacy): all LIVE_OK with accounting join, includes open if in ledger
    exp_v1 = 0.0
    pnl_v1 = 0.0
    n_v1 = 0
    for oid, o in orders.items():
        if oid not in pnl_by_oid:
            continue
        if o.get("stake") is None:
            continue
        exp_v1 += float(o["stake"])
        pnl_v1 += float(pnl_by_oid[oid])
        n_v1 += 1
    if exp_v1 > 0:
        roiw_v1 = metric_envelope(
            value=(pnl_v1 / exp_v1) * 100.0,
            unit="percent",
            n=n_v1,
            numerator=pnl_v1,
            denominator=exp_v1,
            status="AVAILABLE",
            metric_version="v1.0",
            source="daily_v1_contract",
            notes=["legacy: may include open if present in ledger", "w = exposure-weighted"],
        )
    else:
        roiw_v1 = metric_envelope(
            value=None,
            unit="percent",
            n=0,
            status="MISSING" if orders else "AVAILABLE",
            metric_version="v1.0",
            source="daily_v1_contract",
        )

    # ROIw Total v2: settled-aware (same formula on settled_like only)
    if stake_sum > 0:
        roiw_v2 = metric_envelope(
            value=(pnl_sum / stake_sum) * 100.0,
            unit="percent",
            n=len(settled_like),
            numerator=pnl_sum,
            denominator=stake_sum,
            status="PARTIAL" if open_list or missing else "AVAILABLE",
            metric_version="v2.0",
            source="executor+accounting",
            notes=["principal complementary to roi_settled", "open excluded"],
        )
    else:
        roiw_v2 = metric_envelope(
            value=None,
            unit="percent",
            n=0,
            status="PARTIAL" if open_list else ("MISSING" if orders else "AVAILABLE"),
            metric_version="v2.0",
            source="executor+accounting",
        )

    stake_void = sum(float(r["stake"]) for r in void_push if r.get("stake") is not None)
    stake_decided_ex_void = sum(float(r["stake"]) for r in settled if r.get("stake") is not None)
    pnl_decided_ex_void = sum(float(r["pnl"]) for r in settled)
    stake_open = sum(float(orders[oid]["stake"]) for oid in open_list if orders[oid].get("stake") is not None)
    stake_placed = sum(float(o["stake"]) for o in orders.values() if o.get("stake") is not None)

    roi_decided = None
    if stake_decided_ex_void > 0:
        roi_decided = metric_envelope(
            value=pnl_decided_ex_void / stake_decided_ex_void,
            unit="fraction",
            n=len(settled),
            numerator=pnl_decided_ex_void,
            denominator=stake_decided_ex_void,
            status="AVAILABLE" if not open_list and not missing else "PARTIAL",
            metric_version="v2.0",
            notes=["void excluded from numerator and denominator"],
        )
    else:
        roi_decided = metric_envelope(status="MISSING" if settled_like or orders else "AVAILABLE", unit="fraction", n=0)

    # alias: roi_resolved == roi including void in denominator (principal)
    roi_resolved = roi
    if isinstance(roi_resolved, dict):
        roi_resolved = dict(roi_resolved)
        roi_resolved["notes"] = list(roi_resolved.get("notes") or []) + [
            "principal_metric=roi_resolved",
            "void/push stake included in denominator; pnl≈0",
        ]

    if not orders:
        maturity = "FULLY_SETTLED"
    elif open_list and settled_like:
        maturity = "PARTIALLY_SETTLED"
    elif open_list and not settled_like:
        maturity = "OPEN_COHORT"
    elif missing:
        maturity = "PARTIALLY_SETTLED"
    else:
        maturity = "FULLY_SETTLED"

    return {
        "live_ok_total": len(orders),
        "n_open": len(open_list),
        "n_settled": len(settled),
        "n_void_push": len(void_push),
        "n_missing_accounting": len(missing),
        "open_order_ids": open_list,
        "missing_order_ids": missing,
        "stake_placed_sum": stake_placed,
        "stake_settled_sum": stake_sum,  # legacy alias = stake_resolved_total
        "pnl_settled_sum": pnl_sum,
        "stake_placed": stake_placed,
        "stake_resolved_total": stake_sum,
        "stake_decided_ex_void": stake_decided_ex_void,
        "stake_void": stake_void,
        "stake_open": stake_open,
        "pnl_resolved": pnl_sum,
        "pnl_decided_ex_void": pnl_decided_ex_void,
        "roi_settled": roi,  # legacy alias
        "roi_resolved": roi_resolved,
        "roi_decided_ex_void": roi_decided,
        "roiw_total_v1": roiw_v1,
        "roiw_total_v2": roiw_v2,
        "maturity_status": maturity,
        "principal_metric": "roi_resolved",
        "complementary_metric": "roi_decided_ex_void",
        "parity_legacy_metric": "roiw_total_v1",
        "formulas": {
            "roi_resolved": "pnl_resolved / stake_resolved_total (void stake in denom)",
            "roi_decided_ex_void": "pnl_decided_ex_void / stake_decided_ex_void",
        },
    }
