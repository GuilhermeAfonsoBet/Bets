"""Data quality metrics and alerts."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .metrics import group_by_class
from .settlement import performance_block, sample_gate


def data_quality_report(
    rows: Sequence[Dict[str, Any]],
    *,
    mapping_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    n = len(rows)
    by_cls = group_by_class(rows)
    class_counts = {c: len(by_cls[c]) for c in by_cls}
    classified = class_counts["FRIENDLY"] + class_counts["NON_FRIENDLY"]
    classification_coverage = (classified / n) if n else None

    per_class: Dict[str, Any] = {}
    for c, lst in by_cls.items():
        per_class[c] = {
            "n": len(lst),
            "missing_order_id": sum(1 for r in lst if not r.get("order_id")),
            "missing_event_id": sum(1 for r in lst if not r.get("event_id")),
            "missing_league": sum(1 for r in lst if not (r.get("league_name") or r.get("league"))),
            "missing_competition": sum(
                1 for r in lst if not (r.get("competition_name") or r.get("competition"))
            ),
            "missing_kickoff": sum(1 for r in lst if not r.get("kickoff_utc")),
            "accounting_coverage": performance_block(lst).get("accounting_coverage"),
            "clv_closing_coverage": (
                sum(1 for r in lst if r.get("clv_closing_valid_strict")) / len(lst) if lst else None
            ),
            "source_missing": sum(1 for r in lst if r.get("clv_source_missing")),
            "line_mismatch": sum(1 for r in lst if r.get("clv_line_mismatch")),
        }

    oids = [r.get("order_id") for r in mapping_rows if r.get("order_id")]
    dup_rate = (1 - (len(set(oids)) / len(oids))) if oids else 0.0

    return {
        "n_universe": n,
        "class_counts": class_counts,
        "classification_coverage": classification_coverage,
        "unclassified_pct": (class_counts["UNCLASSIFIED"] / n) if n else None,
        "conflict_pct": (class_counts["CONFLICT"] / n) if n else None,
        "per_class": per_class,
        "duplicate_rate": dup_rate,
        "mapping_reconciles_universe": len(mapping_rows) == n,
    }


def build_alerts(
    rows: Sequence[Dict[str, Any]],
    *,
    dq: Dict[str, Any],
    concentration: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []

    def add(code: str, severity: str, detail: str) -> None:
        alerts.append({"alert": code, "severity": severity, "detail": detail})

    cov = dq.get("classification_coverage")
    if cov is not None and cov < 0.80:
        add("FRIENDLY_CLASSIFICATION_LOW_COVERAGE", "high", f"classification_coverage={cov:.3f}")
    if (dq.get("conflict_pct") or 0) > 0:
        add("FRIENDLY_CLASSIFICATION_CONFLICT", "medium", f"conflict_pct={dq.get('conflict_pct')}")
    if (dq.get("unclassified_pct") or 0) > 0.20:
        add("FRIENDLY_UNCLASSIFIED_HIGH", "medium", f"unclassified_pct={dq.get('unclassified_pct')}")

    pc = dq.get("per_class") or {}
    acc_f = (pc.get("FRIENDLY") or {}).get("accounting_coverage")
    acc_nf = (pc.get("NON_FRIENDLY") or {}).get("accounting_coverage")
    if acc_f is not None and acc_nf is not None and abs(acc_f - acc_nf) > 0.10:
        add("ACCOUNTING_COVERAGE_DIFFERENCE", "medium", f"friendly={acc_f} non_friendly={acc_nf}")
    clv_f = (pc.get("FRIENDLY") or {}).get("clv_closing_coverage")
    clv_nf = (pc.get("NON_FRIENDLY") or {}).get("clv_closing_coverage")
    if clv_f is not None and clv_nf is not None and abs(clv_f - clv_nf) > 0.15:
        add("CLV_COVERAGE_DIFFERENCE", "medium", f"friendly={clv_f} non_friendly={clv_nf}")

    for c in concentration:
        if c.get("class") == "FRIENDLY" and (c.get("top1_event_stake_share") or 0) > 0.25:
            add(
                "FRIENDLY_RESULT_CONCENTRATED",
                "medium",
                f"top1_event_stake_share={c.get('top1_event_stake_share')}",
            )
        if c.get("class") == "NON_FRIENDLY" and (c.get("top1_event_stake_share") or 0) > 0.25:
            add(
                "NON_FRIENDLY_RESULT_CONCENTRATED",
                "medium",
                f"top1_event_stake_share={c.get('top1_event_stake_share')}",
            )

    n_res = performance_block(list(rows)).get("n_resolved") or 0
    gate = sample_gate(int(n_res))
    if gate == "VERY_LOW_N":
        add("VERY_LOW_N", "high", f"n_resolved={n_res}")
    elif gate == "INSUFFICIENT_N":
        add("INSUFFICIENT_N", "medium", f"n_resolved={n_res}")

    return alerts
