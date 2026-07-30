"""P0 CLV forward section from health JSON + snapshots JSONL (read-only)."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

from .statuses import metric_envelope

WINDOWS = ("POST_5M", "POST_15M", "CLOSING")


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def build_clv_section(root: Path) -> Dict[str, Any]:
    health = _load_json(root / "logs" / "h3bup_clv_health.json")
    snaps = list(_iter_jsonl(root / "logs" / "h3bup_clv_snapshots.jsonl"))

    by_win: Dict[str, List[dict]] = {w: [] for w in WINDOWS}
    for s in snaps:
        w = str(s.get("window_name") or "")
        if w in by_win and str(s.get("quality_status") or "") == "VALID_STRICT":
            by_win[w].append(s)

    windows_table = []
    perf_table = []
    for w in WINDOWS:
        expected = health.get(f"{w.lower()}_expected")
        if expected is None:
            # map POST_5M -> post_5m_expected
            key = {"POST_5M": "post_5m", "POST_15M": "post_15m", "CLOSING": "closing"}[w]
            expected = health.get(f"{key}_expected")
            attempted = health.get(f"{key}_attempted")
            strict = health.get(f"{key}_valid_strict")
        else:
            attempted = health.get(f"{w.lower()}_attempted")
            strict = health.get(f"{w.lower()}_valid_strict")
        key = {"POST_5M": "post_5m", "POST_15M": "post_15m", "CLOSING": "closing"}[w]
        expected = health.get(f"{key}_expected")
        attempted = health.get(f"{key}_attempted")
        strict = health.get(f"{key}_valid_strict")
        due = expected  # until due tracked separately, approximate
        cov = None
        try:
            if expected and float(expected) > 0 and strict is not None:
                cov = 100.0 * float(strict) / float(expected)
        except Exception:
            cov = None
        windows_table.append(
            {
                "window": w,
                "expected": expected,
                "due": due,
                "attempted": attempted,
                "strict_valid": strict,
                "coverage_pct": cov,
            }
        )

        vals = []
        dists = []
        for s in by_win[w]:
            try:
                vals.append(float(s.get("clv_raw_pct")))
            except Exception:
                pass
            try:
                if s.get("snapshot_distance_sec") is not None:
                    dists.append(abs(float(s.get("snapshot_distance_sec"))))
            except Exception:
                pass
        n = len(vals)
        mean = statistics.fmean(vals) if vals else None
        med = statistics.median(vals) if vals else None
        pos = (100.0 * sum(1 for v in vals if v > 0) / n) if n else None
        st = "INSUFFICIENT_N" if n < 30 else ("AVAILABLE" if n else "MISSING")
        # still show observed values with INSUFFICIENT_N
        perf_table.append(
            {
                "window": w,
                "n": n,
                "clv_mean_pct": mean,
                "clv_median_pct": med,
                "positive_pct": pos,
                "status": st,
                "snapshot_distance_median_sec": (statistics.median(dists) if dists else None),
                "snapshot_distance_p95_sec": (
                    sorted(dists)[int(0.95 * (len(dists) - 1))] if len(dists) >= 2 else (dists[0] if dists else None)
                ),
            }
        )

    return {
        "collection_status": health.get("status"),
        "collection_started_at_utc": health.get("collection_started_at_utc"),
        "source_priority": health.get("source_priority"),
        "collector_status": health.get("collector_status"),
        "fair_edge": metric_envelope(value=None, status="NOT_IMPLEMENTED", notes=["Phase 2D not started"]),
        "funnel": {
            "live_ok_after_activation": health.get("live_ok_after_activation"),
            "obligations_expected": health.get("obligations_expected"),
            "obligations_created": health.get("obligations_created"),
            "source_missing": health.get("source_missing"),
            "line_mismatch": health.get("line_mismatch"),
            "side_mismatch": health.get("side_mismatch"),
            "period_mismatch": health.get("period_mismatch"),
            "kickoff_missing": health.get("kickoff_missing"),
            "kickoff_conflict": health.get("kickoff_conflict"),
            "snapshot_after_kickoff": health.get("snapshot_after_kickoff"),
            "snapshot_too_far": health.get("snapshot_too_far"),
            "retry_backlog": health.get("retry_backlog"),
        },
        "windows": windows_table,
        "performance": {p["window"]: p for p in perf_table},
        "performance_rows": perf_table,
        # legacy compact fields for older renderers
        "post_5m_valid_strict": metric_envelope(
            value=health.get("post_5m_valid_strict"),
            unit="count",
            n=int(health.get("post_5m_valid_strict") or 0),
            status="INSUFFICIENT_N" if int(health.get("live_ok_after_activation") or 0) < 30 else "AVAILABLE",
        ),
        "post_15m_valid_strict": metric_envelope(
            value=health.get("post_15m_valid_strict"),
            unit="count",
            n=int(health.get("post_15m_valid_strict") or 0),
            status="INSUFFICIENT_N" if int(health.get("live_ok_after_activation") or 0) < 30 else "AVAILABLE",
        ),
        "closing_valid_strict": metric_envelope(
            value=health.get("closing_valid_strict"),
            unit="count",
            n=int(health.get("closing_valid_strict") or 0),
            status="INSUFFICIENT_N" if int(health.get("live_ok_after_activation") or 0) < 30 else "AVAILABLE",
        ),
    }
