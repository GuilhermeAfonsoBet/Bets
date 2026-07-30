"""P0 E2E + execution funnel builders from h3bup_e2e_trace.jsonl (read-only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .statuses import metric_envelope


FUNNEL_STEPS = [
    ("audit_persisted", "H3B_AUDIT_PERSIST_FINISHED"),
    ("bridge_fetched", "H3B_BRIDGE_FETCHED"),
    ("policy_evaluated", "H3B_POLICY_EVAL_FINISHED"),
    ("execution_request_created", "H3B_EXEC_REQUEST_CREATED"),
    ("executor_received", "H3B_EXECUTOR_RECEIVED"),
    ("dryrun_started", "H3B_DRYRUN_STARTED"),
    ("dryrun_finished", "H3B_DRYRUN_FINISHED"),
    ("final_gate_decided", "H3B_FINAL_GATE_DECIDED"),
    ("place_started", "H3B_PLACE_STARTED"),
    ("place_finished", "H3B_PLACE_FINISHED"),
    ("live_ok", "LIVE_OK"),
]

SEGMENT_MAP = [
    ("ws_to_detect", "ws_to_detect_ms"),
    ("detect_to_audit", "detected_to_audited"),
    ("audit_to_bridge", "audit_persist_to_bridge_fetch_ms"),
    ("bridge_to_request", "audit_to_request"),
    ("request_to_executor", "request_send_to_executor_receive_ms"),
    ("executor_to_dryrun", "executor_receive_to_dryrun_start_ms"),
    ("dryrun_duration", "dryrun_duration_ms"),
    ("dryrun_to_gate", "dryrun_to_final_gate_ms"),
    ("gate_to_place", "final_gate_to_place_start_ms"),
    ("place_duration", "place_duration_ms"),
    ("ws_to_live_ok", "ws_to_live_ok_ms"),
]


def build_e2e_and_funnel(root: Path, *, window_label: str = "all_traces_available") -> Dict[str, Any]:
    """Load E2E analyzer if available; fail-open to empty structure."""
    out: Dict[str, Any] = {
        "window_label": window_label,
        "available": False,
        "n_traces": 0,
        "n_live_ok": 0,
        "funnel": [],
        "block_reasons": [],
        "segments": {},
        "ordering_violations": 0,
        "clock_skew": 0,
        "trace_events_dropped": 0,
        "dominant_stage": None,
        "detect_to_audit_overhead": metric_envelope(status="WATCH", unit="ms", notes=["overhead remains WATCH"]),
    }
    try:
        from ops.analyze_h3bup_e2e_latency import (
            FUNNEL,
            FUNNEL_ALIASES,
            analyze_trace,
            group_traces,
            load_events,
            summarize,
        )
    except Exception as e:
        out["error"] = str(e)[:200]
        return out

    tpath = root / "logs" / "h3bup_e2e_trace.jsonl"
    # also honor env via analyzer default if exists
    import os

    tpath = Path(os.getenv("H3BUP_E2E_TRACE_PATH", str(tpath)))
    if not tpath.exists():
        out["error"] = "e2e_trace_missing"
        return out

    evs = load_events(tpath)
    trs = group_traces(evs)
    rows = [analyze_trace(tid, evs2) for tid, evs2 in trs.items()]
    summary, _by_st, cov = summarize(rows)
    n_traces = len(rows)
    n_live = sum(1 for r in rows if r.get("status") == "LIVE_OK")
    out["available"] = True
    out["n_traces"] = n_traces
    out["n_live_ok"] = n_live
    out["ordering_violations"] = sum(1 for r in rows if r.get("ordering_violations"))
    out["clock_skew"] = sum(1 for r in rows if r.get("clock_skew_suspected"))

    # coverage funnel counts
    cov_map = {str(c.get("stage") or c.get("event") or c.get("name")): c for c in (cov or []) if isinstance(c, dict)}

    def _count_for(event: str) -> int:
        aliases = FUNNEL_ALIASES.get(event, (event,))
        for a in aliases:
            for k, c in cov_map.items():
                if a in k or k == a:
                    return int(c.get("n") or c.get("count") or 0)
        # fallback: count traces that have event
        n = 0
        for r in rows:
            evset = set(r.get("events_present") or r.get("events") or [])
            if not evset and isinstance(r.get("by_event"), dict):
                evset = set(r["by_event"].keys())
            if event in evset or any(a in evset for a in aliases):
                n += 1
            elif event == "LIVE_OK" and r.get("status") == "LIVE_OK":
                n += 1
        return n

    # Prefer cov list order from analyzer
    funnel_rows = []
    prev = None
    initial = None
    # Use FUNNEL + LIVE_OK
    steps = [(s, s) for s in FUNNEL] + [("LIVE_OK", "LIVE_OK")]
    # remap display names
    display = {
        "H3B_AUDIT_PERSIST_FINISHED": "audit persisted",
        "H3B_BRIDGE_FETCHED": "bridge fetched",
        "H3B_POLICY_EVAL_FINISHED": "policy evaluated",
        "H3B_EXEC_REQUEST_CREATED": "execution request created",
        "H3B_EXECUTOR_RECEIVED": "executor received",
        "H3B_DRYRUN_STARTED": "dry-run started",
        "H3B_DRYRUN_FINISHED": "dry-run finished",
        "H3B_FINAL_GATE_DECIDED": "final gate decided",
        "H3B_PLACE_STARTED": "place started",
        "H3B_PLACE_FINISHED": "place finished",
        "LIVE_OK": "LIVE_OK",
        "H3B_WS_RECEIVED": "WS received",
        "H3B_DETECTED": "detected",
    }
    for ev, _ in steps:
        # skip WS/DETECTED from required table? user wants audit→LIVE_OK but also listed those.
        n = _count_for(ev)
        if initial is None and ev in {"H3B_WS_RECEIVED", "H3B_DETECTED", "H3B_AUDIT_PERSIST_FINISHED"}:
            if n > 0:
                initial = n
        if initial is None:
            initial = max(n, 1)
        pct_prev = (100.0 * n / prev) if prev else None
        pct_init = (100.0 * n / initial) if initial else None
        funnel_rows.append(
            {
                "step": display.get(ev, ev),
                "event": ev,
                "n": n,
                "pct_prev": pct_prev,
                "pct_initial": pct_init,
                "status": "AVAILABLE" if n_traces else "MISSING",
            }
        )
        prev = n if n else prev

    # Ensure policy/place_started appear even if 0 when missing from FUNNEL
    out["funnel"] = funnel_rows

    # block reasons from final gate / status
    reasons: Dict[str, int] = {}
    for r in rows:
        st = str(r.get("status") or "OTHER")
        if st in {"LIVE_OK", "OK", "SUCCESS"}:
            continue
        # gate reason if present
        gr = r.get("final_gate_reason") or r.get("block_reason") or st
        reasons[str(gr)] = reasons.get(str(gr), 0) + 1
    # also scan CAP etc in status
    for key in ("CAP_BLOCKED", "API_FAILED", "NO_SESSION", "STALE", "LIVE_PRECHECK_FAILED", "LIVE_PLACE_FAILED"):
        reasons.setdefault(key, sum(1 for r in rows if key in str(r.get("status") or "") or key in str(r.get("final_gate_reason") or "")))
    req_n = _count_for("H3B_EXEC_REQUEST_CREATED") or 1
    out["block_reasons"] = [
        {"reason": k, "n": v, "pct_requests": (100.0 * v / req_n) if req_n else None}
        for k, v in sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))
        if v > 0
    ]

    # segments
    sum_map = {str(s.get("metric") or s.get("name")): s for s in (summary or []) if isinstance(s, dict)}
    segs = {}
    dom = None
    dom_v = -1.0
    for label, metric in SEGMENT_MAP:
        s = sum_map.get(metric) or {}
        n = int(s.get("n_calculable") or s.get("n") or 0)
        med = s.get("median")
        p95 = s.get("p95")
        covg = s.get("coverage")
        st = "INSUFFICIENT_N" if n == 0 or n < 30 else ("AVAILABLE" if med is not None else "MISSING")
        if n == 0:
            med = None
            p95 = None
        segs[label] = metric_envelope(
            value=med,
            unit="ms",
            n=n,
            coverage_pct=(100.0 * float(covg) if covg is not None and covg <= 1 else covg),
            status=st,
            numerator=med,
            notes=[f"p95={p95}", f"metric={metric}"],
        )
        segs[label]["p95"] = p95
        if med is not None and label not in {"ws_to_live_ok"} and float(med) > dom_v:
            dom_v = float(med)
            dom = label
    out["segments"] = segs
    out["dominant_stage"] = dom
    out["full_trace_coverage_pct"] = (100.0 * n_live / n_traces) if n_traces else None
    # overhead watch
    det = segs.get("detect_to_audit") or {}
    out["detect_to_audit_overhead"] = metric_envelope(
        value=det.get("value"),
        unit="ms",
        n=det.get("n") or 0,
        status="WATCH",
        notes=["overhead remains WATCH until re-evaluation", f"median={det.get('value')}"],
    )
    return out
