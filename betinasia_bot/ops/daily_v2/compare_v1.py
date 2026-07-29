"""Compare Daily V1 sample metrics vs V2 snapshot (read-only)."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _parse_roiw_from_v1_md(md: str, day: str) -> Optional[float]:
    # Table row: | 2026-07-28 | ... | -49.85% | ...
    for line in md.splitlines():
        if day in line and "ROIw" not in line and line.strip().startswith("|"):
            parts = [p.strip() for p in line.split("|")]
            # find percent-like token that looks like ROIw Total column — fragile; best-effort
            pcts = []
            for p in parts:
                m = re.fullmatch(r"(-?\d+(?:\.\d+)?)%", p)
                if m:
                    pcts.append(float(m.group(1)))
            if pcts:
                # In V1 sample table ROIw Total is often 4th numeric percent-ish after pnls
                return pcts[0] if len(pcts) == 1 else pcts[min(3, len(pcts) - 1)]
    return None


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

    live_v2 = ((v2.get("execution_funnel") or {}).get("live_ok") or {}).get("value")
    rows.append(
        {
            "metric": "LIVE_OK",
            "v1": v1.get("live_ok"),
            "v2": live_v2,
            "delta": None if v1.get("live_ok") is None or live_v2 is None else (live_v2 - v1.get("live_ok")),
            "cause": "V2 filters H3BUP_vNext + created_at UTC closed cohort; V1 mixes windows",
            "correct_version": "v2",
        }
    )

    roiw_v1_legacy = ((v2.get("performance") or {}).get("roiw_total_v1") or {}).get("value")
    rows.append(
        {
            "metric": "ROIw Total",
            "v1": v1.get("roiw_total"),
            "v2": roiw_v1_legacy,
            "delta": None
            if v1.get("roiw_total") is None or roiw_v1_legacy is None
            else (roiw_v1_legacy - v1.get("roiw_total")),
            "cause": "Same formula when universe matches; V2 also emits settled-aware roiw_total_v2",
            "correct_version": "both_versioned",
        }
    )

    rows.append(
        {
            "metric": "ROI settled",
            "v1": v1.get("roi_settled"),
            "v2": ((v2.get("performance") or {}).get("roi_settled") or {}).get("value"),
            "delta": None,
            "cause": "V1 often omits explicit roi_settled in executive; V2 principal metric",
            "correct_version": "v2",
        }
    )

    rows.append(
        {
            "metric": "DAILY_FAST_LE_6S",
            "v1": v1.get("fast_le_6s"),
            "v2": ((v2.get("latency") or {}).get("daily_fast_le_6s") or {}).get("value"),
            "delta": None,
            "cause": "V1 uses dual 5s/6s thresholds by thesis day; V2 Daily fixed <=6s",
            "correct_version": "v2_for_daily_contract",
        }
    )

    rows.append(
        {
            "metric": "fair_edge",
            "v1": v1.get("fair_edge", "omitted_or_zero_risk"),
            "v2": ((v2.get("clv") or {}).get("fair_edge") or {}).get("status"),
            "delta": None,
            "cause": "V2 explicit NOT_IMPLEMENTED",
            "correct_version": "v2",
        }
    )

    rows.append(
        {
            "metric": "cohort_timestamp",
            "v1": "mixed created_at + post date tables",
            "v2": "created_at UTC only",
            "delta": None,
            "cause": "V1 dual attribution",
            "correct_version": "v2",
        }
    )
    return rows


def write_comparison_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["metric", "v1", "v2", "delta", "cause", "correct_version"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
