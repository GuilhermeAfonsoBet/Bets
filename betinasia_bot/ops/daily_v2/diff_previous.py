"""Diff versus previous V2 snapshot (same cohort preferred)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_previous_snapshot(out_dir: Path, *, report_date: str, current_run_id: str) -> Optional[Path]:
    day = report_date.replace("-", "")
    cands = sorted(out_dir.glob(f"h3bup_daily_snapshot_{day}_*.json"))
    prev = None
    for p in cands:
        if current_run_id in p.name:
            continue
        prev = p
    return prev


def diff_snapshots(prev: dict, cur: dict) -> Dict[str, Any]:
    def g(*path, default=None):
        def _get(obj, keys):
            for k in keys:
                if not isinstance(obj, dict):
                    return default
                obj = obj.get(k)
            return obj if obj is not None else default

        return _get(prev, path), _get(cur, path)

    rows = []

    def add(name, a, b):
        delta = None
        try:
            if a is not None and b is not None:
                delta = float(b) - float(a)
        except Exception:
            delta = None
        rows.append({"metric": name, "anterior": a, "atual": b, "delta": delta})

    add("LIVE_OK", *((g("execution_funnel", "live_ok", "value"))))
    sett_keys = [
        ("open", "n_open"),
        ("settled", "n_settled"),
        ("void", "n_void_push"),
        ("missing", "n_missing_accounting"),
        ("stake resolved", "stake_resolved_total"),
        ("P&L resolved", "pnl_resolved"),
    ]
    for label, key in sett_keys:
        add(label, prev.get("settlement", {}).get(key), cur.get("settlement", {}).get(key))
    add(
        "ROI resolved",
        ((prev.get("performance") or {}).get("roi_resolved") or (prev.get("performance") or {}).get("roi_settled") or {}).get("value"),
        ((cur.get("performance") or {}).get("roi_resolved") or (cur.get("performance") or {}).get("roi_settled") or {}).get("value"),
    )
    for w in ("POST_5M", "POST_15M", "CLOSING"):
        add(
            f"{w} valid",
            ((prev.get("clv") or {}).get("performance") or {}).get(w, {}).get("n"),
            ((cur.get("clv") or {}).get("performance") or {}).get(w, {}).get("n"),
        )
    add(
        "CLV backlog",
        ((prev.get("clv") or {}).get("funnel") or {}).get("retry_backlog"),
        ((cur.get("clv") or {}).get("funnel") or {}).get("retry_backlog"),
    )
    add("alertas ativos", len(prev.get("exceptions") or []), len(cur.get("exceptions") or []))

    prev_alerts = {a.get("alert_id") for a in (prev.get("exceptions") or []) if isinstance(a, dict)}
    cur_alerts = {a.get("alert_id") for a in (cur.get("exceptions") or []) if isinstance(a, dict)}
    return {
        "previous_run_id": prev.get("run_id"),
        "current_run_id": cur.get("run_id"),
        "rows": rows,
        "new_alerts": sorted(cur_alerts - prev_alerts),
        "resolved_alerts": sorted(prev_alerts - cur_alerts),
    }
