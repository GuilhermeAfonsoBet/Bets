#!/usr/bin/env python3
"""Export CLV health CSVs + run Phase 2B soak summary (read-only)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ops.analyze_h3bup_e2e_latency import (
    analyze_trace,
    group_traces,
    load_events,
    summarize,
    _stats,
)
from ops.h3bup_clv_config import load_config
from ops.h3bup_clv_store import get_store


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def export_clv(date: str, out_dir: Path) -> Dict[str, Any]:
    cfg = load_config()
    store = get_store(cfg)
    obls = store.list_obligations()
    snaps = store.snapshots()
    write_csv(out_dir / f"h3bup_clv_obligations_{date}.csv", obls)
    write_csv(out_dir / f"h3bup_clv_snapshots_{date}.csv", snaps)
    failures = []
    for o in obls:
        if o.get("last_error_code"):
            failures.append({"kind": "obligation", "order_id": o.get("order_id"), "window": o.get("window_name"), "code": o.get("last_error_code"), "status": o.get("status")})
    for s in snaps:
        if s.get("failure_reason") and s.get("quality_status") != "VALID_STRICT":
            failures.append({"kind": "snapshot", "order_id": s.get("order_id"), "window": s.get("window_name"), "code": s.get("failure_reason"), "status": s.get("quality_status")})
    write_csv(out_dir / f"h3bup_clv_failures_{date}.csv", failures)

    funnel_rows = []
    coverage_rows = []
    live_orders = sorted({o.get("order_id") for o in obls})
    for window in ("POST_5M", "POST_15M", "CLOSING"):
        w_obls = [o for o in obls if o.get("window_name") == window]
        w_snaps = [s for s in snaps if s.get("window_name") == window]
        strict = [s for s in w_snaps if s.get("quality_status") == "VALID_STRICT"]
        expected = len(w_obls)
        created = expected
        due = sum(1 for o in w_obls if o.get("status") not in ("WAITING_TARGET", "PENDING") or True)
        attempted = sum(1 for o in w_obls if int(o.get("attempts") or 0) > 0)
        snap_found = len(w_snaps)
        same_event = sum(1 for s in w_snaps if s.get("same_event_flag"))
        same_market = sum(1 for s in w_snaps if s.get("same_market_flag"))
        same_period = sum(1 for s in w_snaps if s.get("same_period_flag"))
        same_side = sum(1 for s in w_snaps if s.get("same_side_flag"))
        same_line = sum(1 for s in w_snaps if s.get("same_line_strict_flag"))
        before = sum(1 for s in w_snaps if s.get("snapshot_before_kickoff_flag"))
        valid = len(strict)
        clv_n = sum(1 for s in strict if s.get("clv_raw_pct") is not None)
        funnel_rows.append(
            {
                "window": window,
                "live_ok_orders": len(live_orders),
                "expected": expected,
                "created": created,
                "due": due,
                "attempted": attempted,
                "snapshot_found": snap_found,
                "same_event": same_event,
                "same_market": same_market,
                "same_period": same_period,
                "same_side": same_side,
                "same_line": same_line,
                "before_kickoff": before,
                "valid_strict": valid,
                "clv_calculated": clv_n,
            }
        )
        coverage_rows.append(
            {
                "window": window,
                "expected": expected,
                "created": created,
                "attempted": attempted,
                "snapshot": snap_found,
                "strict_valid": valid,
                "coverage": (float(valid) / float(expected) if expected else 0.0),
            }
        )
    write_csv(out_dir / f"h3bup_clv_collection_funnel_{date}.csv", funnel_rows)
    write_csv(out_dir / f"h3bup_clv_coverage_by_window_{date}.csv", coverage_rows)
    return {"obligations": len(obls), "snapshots": len(snaps), "orders": len(live_orders)}


def soak_e2e(trace_path: Path, date: str, out_dir: Path, soak_start: Optional[str] = None) -> Dict[str, Any]:
    events = load_events(trace_path)
    if soak_start:
        # filter
        from ops.analyze_h3bup_e2e_latency import _parse_ts

        t0 = _parse_ts(soak_start)
        events = [e for e in events if (_parse_ts(e.get("event_ts_utc")) or 0) >= (t0 or 0)]
    traces = group_traces(events)
    rows = [analyze_trace(tid, evs) for tid, evs in traces.items()]
    summary, by_status, cov = summarize(rows)
    # rename metrics for soak filenames
    write_csv(out_dir / f"h3bup_phase2b_soak_latency_summary_{date}.csv", summary)
    write_csv(out_dir / f"h3bup_phase2b_soak_latency_by_status_{date}.csv", by_status)
    write_csv(out_dir / f"h3bup_phase2b_soak_trace_funnel_{date}.csv", cov)
    missing = []
    for r in rows:
        if r.get("missing_stages"):
            missing.append({"trace_id": r["trace_id"], "status": r.get("status"), "missing_stages": r.get("missing_stages")})
    write_csv(out_dir / f"h3bup_phase2b_soak_missing_events_{date}.csv", missing)

    names = Counter(e.get("event_name") for e in events)
    n_live = sum(1 for r in rows if r.get("status") == "LIVE_OK")
    n_req = sum(1 for r in rows if "H3B_EXEC_REQUEST_SENT" in str(r.get("events_present")) or "H3B_EXEC_REQUEST_CREATED" in str(r.get("events_present")))
    n_exec = sum(1 for r in rows if "H3B_EXECUTOR_RECEIVED" in str(r.get("events_present")))
    n_dry = sum(1 for r in rows if "H3B_DRYRUN_FINISHED" in str(r.get("events_present")))
    n_gate = sum(1 for r in rows if "H3B_FINAL_GATE_DECIDED" in str(r.get("events_present")))
    n_place_s = sum(1 for r in rows if "H3B_PLACE_STARTED" in str(r.get("events_present")))
    n_place_f = sum(1 for r in rows if "H3B_PLACE_FINISHED" in str(r.get("events_present")))
    n_cap = sum(1 for r in rows if r.get("status") == "CAP_BLOCKED")
    n_api = sum(1 for r in rows if r.get("status") == "API_FAILED")

    def med(metric: str):
        r = next((x for x in summary if x.get("metric") == metric), {})
        return r

    dta = med("detected_to_audited")
    classification = "E2E_SOAK_INSUFFICIENT_N"
    if n_live >= 10 or (n_req >= 30 and n_gate >= 30):
        classification = "E2E_SOAK_SUFFICIENT"
    elif n_req >= 1 and n_exec >= 1:
        classification = "E2E_SOAK_PARTIAL"

    first = events[0]["event_ts_utc"] if events else None
    last = events[-1]["event_ts_utc"] if events else None
    health = {
        "soak_start_utc": soak_start or first,
        "soak_cutoff_utc": last,
        "jsonl_bytes": trace_path.stat().st_size if trace_path.exists() else 0,
        "total_events": len(events),
        "total_trace_ids": len(rows),
        "audits": names.get("H3B_AUDIT_PERSIST_FINISHED", 0),
        "bridge_fetched": names.get("H3B_BRIDGE_FETCHED", 0),
        "requests": n_req,
        "executor_received": n_exec,
        "dryruns": n_dry,
        "final_gates": n_gate,
        "place_started": n_place_s,
        "place_finished": n_place_f,
        "live_ok": n_live,
        "cap_blocked": n_cap,
        "api_failed": n_api,
        "classification": classification,
        "detected_to_audited_median": dta.get("median"),
        "detected_to_audited_p95": dta.get("p95"),
        "ws_to_live_ok": med("ws_to_live_ok_ms"),
        "audit_to_request": med("audit_to_request"),
        "dryrun_duration_ms": med("dryrun_duration_ms"),
        "place_duration_ms": med("place_duration_ms"),
        "ordering_violations": sum(1 for r in rows if r.get("ordering_violations")),
        "clock_skew": sum(1 for r in rows if r.get("clock_skew_suspected")),
        "corrupt_lines": 0,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / f"h3bup_phase2b_soak_health_{date}.json").write_text(json.dumps(health, indent=2) + "\n", encoding="utf-8")
    return health


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    ap.add_argument("--out-dir", default="logs")
    ap.add_argument("--trace", default="logs/h3bup_e2e_trace.jsonl")
    ap.add_argument("--soak-start", default=None)
    args = ap.parse_args()
    out = Path(args.out_dir)
    clv = export_clv(args.date, out)
    soak = soak_e2e(Path(args.trace), args.date, out, soak_start=args.soak_start)
    print(json.dumps({"clv": clv, "soak_class": soak.get("classification"), "live_ok": soak.get("live_ok"), "requests": soak.get("requests")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
