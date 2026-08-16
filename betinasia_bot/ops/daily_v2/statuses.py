"""Canonical status vocabularies for Daily V2 metrics and health."""

from __future__ import annotations

METRIC_STATUSES = frozenset(
    {
        "AVAILABLE",
        "NOT_DUE",
        "NOT_APPLICABLE",
        "MISSING",
        "STALE",
        "PARTIAL",
        "UNRECONCILED",
        "INSUFFICIENT_N",
        "NOT_IMPLEMENTED",
        "FAILED",
        "UNAVAILABLE_STALE",
        "WATCH",
        "OK",
    }
)

SOURCE_HEALTH = frozenset(
    {"HEALTHY", "WATCH", "STALE", "PARTIAL", "FAILED", "NOT_AVAILABLE"}
)

REPORT_HEALTH = frozenset(
    {"HEALTHY", "WATCH", "PARTIAL", "CRITICAL", "FAILED"}
)

MATURITY = frozenset(
    {
        "OPEN_COHORT",
        "PARTIALLY_SETTLED",
        "FULLY_SETTLED",
        "CLV_PARTIAL",
        "CLV_COMPLETE",
        "FINALIZED",
    }
)

ALERT_SEVERITY = frozenset({"INFO", "WATCH", "WARNING", "CRITICAL"})


def metric_envelope(
    *,
    value=None,
    unit: str | None = None,
    n: int = 0,
    numerator=None,
    denominator=None,
    coverage_pct=None,
    status: str = "MISSING",
    metric_version: str = "v2.0",
    source: str | None = None,
    notes: list | None = None,
) -> dict:
    if status not in METRIC_STATUSES and status not in {"OK"}:
        status = "FAILED"
    # Zero is only valid when status is AVAILABLE (or OK) and value is literally 0.
    if value == 0 and status in {"MISSING", "STALE", "FAILED", "NOT_IMPLEMENTED", "NOT_DUE", "NOT_APPLICABLE"}:
        value = None
    return {
        "value": value,
        "unit": unit,
        "n": int(n),
        "numerator": numerator,
        "denominator": denominator,
        "coverage_pct": coverage_pct,
        "status": status,
        "metric_version": metric_version,
        "source": source,
        "notes": list(notes or []),
    }
