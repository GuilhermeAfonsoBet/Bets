"""CLV join — VALID_STRICT only for official metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

WINDOWS = ("POST_5M", "POST_15M", "CLOSING")
WINDOW_TO_FIELD = {
    "POST_5M": "clv_post_5m",
    "POST_15M": "clv_post_15m",
    "CLOSING": "clv_closing",
}


def iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if isinstance(o, dict):
                yield o


def load_clv_by_order(snapshots_path: Path) -> Dict[str, Dict[str, Any]]:
    """Map order_id -> window metrics. Official values only when VALID_STRICT."""
    out: Dict[str, Dict[str, Any]] = {}
    for s in iter_jsonl(snapshots_path) or []:
        oid = str(s.get("order_id") or "").strip()
        if not oid:
            continue
        w = str(s.get("window_name") or s.get("window") or "").strip()
        if w not in WINDOW_TO_FIELD:
            continue
        field = WINDOW_TO_FIELD[w]
        valid = str(s.get("quality_status") or "") == "VALID_STRICT"
        rec = out.setdefault(
            oid,
            {
                "clv_post_5m": None,
                "clv_post_5m_valid_strict": False,
                "clv_post_15m": None,
                "clv_post_15m_valid_strict": False,
                "clv_closing": None,
                "clv_closing_valid_strict": False,
                "clv_failure_reason": None,
                "clv_source_missing": False,
                "clv_line_mismatch": False,
                "snapshot_distance_sec": {},
            },
        )
        fail = str(s.get("failure_reason") or s.get("clv_failure_reason") or "")
        if "source_missing" in fail.lower() or str(s.get("quality_status") or "") == "SOURCE_MISSING":
            rec["clv_source_missing"] = True
        if "line_mismatch" in fail.lower() or str(s.get("quality_status") or "") == "LINE_MISMATCH":
            rec["clv_line_mismatch"] = True
        if valid:
            try:
                rec[field] = float(s.get("clv_raw_pct"))
                rec[f"{field}_valid_strict"] = True
            except Exception:
                pass
            try:
                if s.get("snapshot_distance_sec") is not None:
                    rec["snapshot_distance_sec"][w] = abs(float(s["snapshot_distance_sec"]))
            except Exception:
                pass
        else:
            if not rec.get("clv_failure_reason") and fail:
                rec["clv_failure_reason"] = fail
    return out


def attach_clv(orders: List[Dict[str, Any]], clv_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for o in orders:
        oid = str(o.get("order_id") or "")
        row = dict(o)
        c = clv_map.get(oid) or {}
        for k in (
            "clv_post_5m",
            "clv_post_5m_valid_strict",
            "clv_post_15m",
            "clv_post_15m_valid_strict",
            "clv_closing",
            "clv_closing_valid_strict",
            "clv_failure_reason",
            "clv_source_missing",
            "clv_line_mismatch",
        ):
            row[k] = c.get(k) if k in c else (False if k.endswith("_strict") or k.endswith("_missing") or k.endswith("_mismatch") else None)
        # distances
        dists = c.get("snapshot_distance_sec") or {}
        row["clv_post_5m_snapshot_distance_sec"] = dists.get("POST_5M")
        row["clv_post_15m_snapshot_distance_sec"] = dists.get("POST_15M")
        row["clv_closing_snapshot_distance_sec"] = dists.get("CLOSING")
        out.append(row)
    return out


def _percentile(vals: List[float], p: float) -> Optional[float]:
    if not vals:
        return None
    xs = sorted(vals)
    if len(xs) == 1:
        return xs[0]
    idx = int(round(p * (len(xs) - 1)))
    idx = max(0, min(len(xs) - 1, idx))
    return xs[idx]


def _winsorized_mean(vals: List[float], p: float = 0.05) -> Optional[float]:
    if not vals:
        return None
    xs = sorted(vals)
    n = len(xs)
    k = int(n * p)
    if n - 2 * k <= 0:
        return sum(xs) / n
    clipped = xs[k : n - k] if k else xs
    return sum(clipped) / len(clipped)


def clv_summary_for_group(rows: List[Dict[str, Any]], *, group: str) -> List[Dict[str, Any]]:
    out = []
    for w, field, valid_f, dist_f in (
        ("POST_5M", "clv_post_5m", "clv_post_5m_valid_strict", "clv_post_5m_snapshot_distance_sec"),
        ("POST_15M", "clv_post_15m", "clv_post_15m_valid_strict", "clv_post_15m_snapshot_distance_sec"),
        ("CLOSING", "clv_closing", "clv_closing_valid_strict", "clv_closing_snapshot_distance_sec"),
    ):
        expected = len(rows)
        strict_vals = []
        dists = []
        source_missing = 0
        line_mismatch = 0
        for r in rows:
            if r.get("clv_source_missing"):
                source_missing += 1
            if r.get("clv_line_mismatch"):
                line_mismatch += 1
            if r.get(valid_f) and r.get(field) is not None:
                try:
                    strict_vals.append(float(r[field]))
                except Exception:
                    pass
                if r.get(dist_f) is not None:
                    try:
                        dists.append(float(r[dist_f]))
                    except Exception:
                        pass
        n = len(strict_vals)
        cov = (100.0 * n / expected) if expected else None
        from statistics import fmean, median

        mean = fmean(strict_vals) if strict_vals else None
        med = median(strict_vals) if strict_vals else None
        pos = (100.0 * sum(1 for v in strict_vals if v > 0) / n) if n else None
        if n < 30:
            status = "VERY_LOW_N" if n < 30 else "INSUFFICIENT_N"
            status = "VERY_LOW_N" if n < 30 else status
        if n < 30:
            status = "VERY_LOW_N"
        elif n < 100:
            status = "INSUFFICIENT_N"
        elif n < 250:
            status = "FIRST_READING"
        else:
            status = "RELIABLE_READING_CANDIDATE"
        out.append(
            {
                "group": group,
                "window": w,
                "expected": expected,
                "due": expected,
                "attempted": expected,  # diagnostic; true attempted may be lower
                "strict_valid": n,
                "coverage_pct": cov,
                "n": n,
                "mean": mean,
                "median": med,
                "winsorized_mean": _winsorized_mean(strict_vals),
                "p25": _percentile(strict_vals, 0.25),
                "p75": _percentile(strict_vals, 0.75),
                "positive_pct": pos,
                "source_missing": source_missing,
                "line_mismatch": line_mismatch,
                "retry_backlog": None,
                "snapshot_distance_median": (median(dists) if dists else None),
                "snapshot_distance_p95": _percentile(dists, 0.95),
                "status": status,
                "metric_contract": "VALID_STRICT_ONLY",
            }
        )
    return out
