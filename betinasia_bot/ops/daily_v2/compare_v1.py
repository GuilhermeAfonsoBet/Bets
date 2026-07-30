"""Compare Daily V1 sample metrics vs V2 snapshot (read-only). P0 parity table."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _parse_roiw_from_v1_md(md: str, day: str) -> Optional[float]:
    for line in md.splitlines():
        if day in line and "ROIw" not in line and line.strip().startswith("|"):
            parts = [p.strip() for p in line.split("|")]
            pcts = []
            for p in parts:
                m = re.fullmatch(r"(-?\d+(?:\.\d+)?)%", p)
                if m:
                    pcts.append(float(m.group(1)))
            if pcts:
                return pcts[0] if len(pcts) == 1 else pcts[min(3, len(pcts) - 1)]
    return None


def _classify_delta(metric: str, v1: Any, v2: Any, cause: str) -> str:
    if v1 is None or v2 is None:
        return "PARITY_UNAVAILABLE" if "unavailable" in cause.lower() or v1 is None else "UNKNOWN"
    try:
        if abs(float(v1) - float(v2)) < 1e-9:
            return "MATCH"
    except Exception:
        if str(v1) == str(v2):
            return "MATCH"
    if "definition" in cause.lower() or "contrato" in cause.lower() or "version" in cause.lower():
        return "EXPECTED_DEFINITION_CHANGE"
    if "cutoff" in cause.lower():
        return "EXPECTED_CUTOFF_DIFFERENCE"
    if "filter" in cause.lower() or "universe" in cause.lower() or "coorte" in cause.lower():
        return "FILTER_DIFFERENCE"
    if "join" in cause.lower():
        return "JOIN_DIFFERENCE"
    if "source" in cause.lower():
        return "SOURCE_DIFFERENCE"
    if "v1" in cause.lower() and "bug" in cause.lower():
        return "V1_BUG"
    if "v2" in cause.lower() and "bug" in cause.lower():
        return "V2_BUG"
    return "UNKNOWN"


def compare_snapshots(
    *,
    v2: Dict[str, Any],
    v1_md: Optional[str] = None,
    v1_metrics: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    v1 = dict(v1_metrics or {})
    day = str(v2.get("report_date_utc") or "")
    if v1_md and "roiw_total" not in v1:
        val = _parse_roiw_from_v1_md(v1_md, day)
        if val is not None:
            v1["roiw_total"] = val

    sett = v2.get("settlement") or {}
    perf = v2.get("performance") or {}
    lat = v2.get("latency") or {}
    clv = v2.get("clv") or {}
    e2e = v2.get("e2e") or {}
    fun = v2.get("execution_funnel") or {}

    def add(metric: str, v1v: Any, v2v: Any, cause: str, correct: str = "v2"):
        status = _classify_delta(metric, v1v, v2v, cause)
        delta = None
        try:
            if v1v is not None and v2v is not None:
                delta = float(v2v) - float(v1v)
        except Exception:
            delta = None
        rows.append(
            {
                "metric": metric,
                "v1": v1v,
                "v2": v2v,
                "delta": delta,
                "status": status,
                "cause": cause,
                "correct_version": correct,
            }
        )

    live_v2 = ((fun.get("live_ok") or {}).get("value"))
    add("LIVE_OK", v1.get("live_ok"), live_v2, "V2 filters H3BUP_vNext + created_at UTC closed cohort")
    add("stake placed", v1.get("stake_placed"), sett.get("stake_placed") or sett.get("stake_placed_sum"), "same universe when H3BUP filtered")
    add("open", v1.get("n_open"), sett.get("n_open"), "accounting join")
    add("settled", v1.get("n_settled"), sett.get("n_settled"), "accounting join; V2 separates void")
    add("void", v1.get("n_void"), sett.get("n_void_push"), "V2 explicit void/push")
    add("missing", v1.get("n_missing"), sett.get("n_missing_accounting"), "join difference if ledger incomplete")
    add("P&L", v1.get("pnl"), sett.get("pnl_resolved") or sett.get("pnl_settled_sum"), "resolved vs mixed account")
    add(
        "ROI",
        v1.get("roi_settled"),
        ((perf.get("roi_resolved") or perf.get("roi_settled") or {}).get("value")),
        "V2 principal=roi_resolved (void in denom)",
    )
    add("fast <=6s", v1.get("fast_le_6s"), ((lat.get("daily_fast_le_6s") or {}).get("value")), "V1 dual 5s/6s; V2 Daily fixed <=6s — expected definition change")
    add("study fast <4s", v1.get("study_fast_lt_4s"), ((lat.get("study_fast_lt_4s") or {}).get("value")), "exploratory study contract")
    add(
        "POST_5M strict",
        v1.get("post_5m_strict"),
        ((clv.get("post_5m_valid_strict") or {}).get("value")),
        "VALID_STRICT only",
    )
    add(
        "POST_15M strict",
        v1.get("post_15m_strict"),
        ((clv.get("post_15m_valid_strict") or {}).get("value")),
        "VALID_STRICT only",
    )
    add(
        "CLOSING strict",
        v1.get("closing_strict"),
        ((clv.get("closing_valid_strict") or {}).get("value")),
        "VALID_STRICT only",
    )
    add("E2E LIVE_OK", v1.get("e2e_live_ok"), e2e.get("n_live_ok"), "E2E all-traces vs cohort window — source/window difference")
    add(
        "ROIw Total",
        v1.get("roiw_total"),
        ((perf.get("roiw_total_v1") or {}).get("value")),
        "Same formula when universe matches; V2 also emits settled-aware roiw_total_v2",
        "both_versioned",
    )
    add(
        "fair_edge",
        v1.get("fair_edge", "omitted_or_zero_risk"),
        ((clv.get("fair_edge") or {}).get("status")),
        "V2 explicit NOT_IMPLEMENTED",
    )
    add(
        "cohort_timestamp",
        "mixed created_at + post date tables",
        "created_at UTC only",
        "V1 dual attribution — expected definition change",
    )
    add(
        "report_date",
        v1.get("report_date") or day,
        v2.get("report_date_utc"),
        "parity requires same report_date_utc",
    )
    return rows


def write_comparison_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["metric", "v1", "v2", "delta", "status", "cause", "correct_version"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
