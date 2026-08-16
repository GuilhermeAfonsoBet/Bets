#!/usr/bin/env python3
"""Read-only analyzer for logs/h3bup_e2e_trace.jsonl (Fase 2B)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


EVENT_ORDER = [
    "H3B_WS_RECEIVED",
    "H3B_DETECTED",
    "H3B_AUDIT_ENQUEUED",
    "H3B_AUDIT_GATE_DECIDED",
    "H3B_AUDIT_PERSIST_STARTED",
    "H3B_AUDIT_PERSIST_FINISHED",
    "H3B_BRIDGE_FETCHED",
    "H3B_BRIDGE_SEEN_RESERVED",
    "H3B_POLICY_EVAL_STARTED",
    "H3B_POLICY_EVAL_FINISHED",
    "H3B_EXEC_REQUEST_CREATED",
    "H3B_EXEC_REQUEST_SENT",
    "H3B_EXECUTOR_RECEIVED",
    "H3B_DRYRUN_STARTED",
    "H3B_DRYRUN_FINISHED",
    "H3B_FINAL_GATE_DECIDED",
    "H3B_PLACE_STARTED",
    "H3B_PLACE_FINISHED",
    "H3B_RESULT_PERSIST_STARTED",
    "H3B_RESULT_PERSIST_FINISHED",
]

LATENCY_SPECS: List[Tuple[str, str, str]] = [
    ("ws_to_detect_ms", "H3B_WS_RECEIVED", "H3B_DETECTED"),
    ("detect_to_audit_gate_ms", "H3B_DETECTED", "H3B_AUDIT_GATE_DECIDED"),
    ("audit_gate_to_persist_ms", "H3B_AUDIT_GATE_DECIDED", "H3B_AUDIT_PERSIST_FINISHED"),
    ("audit_persist_to_bridge_fetch_ms", "H3B_AUDIT_PERSIST_FINISHED", "H3B_BRIDGE_FETCHED"),
    ("bridge_fetch_to_policy_ms", "H3B_BRIDGE_FETCHED", "H3B_POLICY_EVAL_FINISHED"),
    ("policy_to_request_ms", "H3B_POLICY_EVAL_FINISHED", "H3B_EXEC_REQUEST_CREATED"),
    ("request_send_to_executor_receive_ms", "H3B_EXEC_REQUEST_SENT", "H3B_EXECUTOR_RECEIVED"),
    ("executor_receive_to_dryrun_start_ms", "H3B_EXECUTOR_RECEIVED", "H3B_DRYRUN_STARTED"),
    ("dryrun_duration_ms", "H3B_DRYRUN_STARTED", "H3B_DRYRUN_FINISHED"),
    ("dryrun_to_final_gate_ms", "H3B_DRYRUN_FINISHED", "H3B_FINAL_GATE_DECIDED"),
    ("final_gate_to_place_start_ms", "H3B_FINAL_GATE_DECIDED", "H3B_PLACE_STARTED"),
    ("place_duration_ms", "H3B_PLACE_STARTED", "H3B_PLACE_FINISHED"),
    ("place_to_result_persist_ms", "H3B_PLACE_FINISHED", "H3B_RESULT_PERSIST_FINISHED"),
    ("ws_to_live_ok_ms", "H3B_WS_RECEIVED", "H3B_PLACE_FINISHED"),
    ("detect_to_live_ok_ms", "H3B_DETECTED", "H3B_PLACE_FINISHED"),
    ("audit_to_live_ok_ms", "H3B_AUDIT_PERSIST_FINISHED", "H3B_PLACE_FINISHED"),
    ("request_to_live_ok_ms", "H3B_EXEC_REQUEST_CREATED", "H3B_PLACE_FINISHED"),
    ("executor_to_live_ok_ms", "H3B_EXECUTOR_RECEIVED", "H3B_PLACE_FINISHED"),
    # legacy / phase1 comparables
    ("detected_to_audited", "H3B_DETECTED", "H3B_AUDIT_PERSIST_FINISHED"),
    ("audit_to_request", "H3B_AUDIT_PERSIST_FINISHED", "H3B_EXEC_REQUEST_CREATED"),
    ("request_to_finished", "H3B_EXEC_REQUEST_CREATED", "H3B_RESULT_PERSIST_FINISHED"),
    ("audit_to_finished", "H3B_AUDIT_PERSIST_FINISHED", "H3B_RESULT_PERSIST_FINISHED"),
]

FUNNEL = [
    "H3B_WS_RECEIVED",
    "H3B_DETECTED",
    "H3B_AUDIT_PERSIST_FINISHED",
    "H3B_BRIDGE_FETCHED",
    "H3B_EXEC_REQUEST_CREATED",
    "H3B_EXECUTOR_RECEIVED",
    "H3B_DRYRUN_FINISHED",
    "H3B_FINAL_GATE_DECIDED",
    "H3B_PLACE_FINISHED",
]

# Compatibility: early builds emitted SENT without CREATED when policy_version
# was still the bridge legacy string (filtered by only_h3bup).
FUNNEL_ALIASES = {
    "H3B_EXEC_REQUEST_CREATED": ("H3B_EXEC_REQUEST_CREATED", "H3B_EXEC_REQUEST_SENT"),
}

INSUFFICIENT_N = 30


def _parse_ts(s: Any) -> Optional[float]:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    txt = str(s).strip()
    if not txt:
        return None
    try:
        if txt.endswith("Z"):
            txt = txt[:-1] + "+00:00"
        return datetime.fromisoformat(txt).timestamp()
    except Exception:
        return None


def _pct(sorted_vals: List[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)


def _stats(vals: List[float], *, total_n: int) -> Dict[str, Any]:
    negs = [v for v in vals if v < 0]
    pos = [v for v in vals if v >= 0]
    pos_s = sorted(pos)
    out = {
        "n_total": int(total_n),
        "n_calculable": int(len(vals)),
        "coverage": (float(len(vals)) / float(total_n) if total_n else 0.0),
        "missing": int(max(0, total_n - len(vals))),
        "negatives": int(len(negs)),
        "mean": (float(statistics.fmean(pos_s)) if pos_s else None),
        "median": (float(statistics.median(pos_s)) if pos_s else None),
        "p75": _pct(pos_s, 75),
        "p90": _pct(pos_s, 90),
        "p95": _pct(pos_s, 95),
        "p99": _pct(pos_s, 99),
        "min": (float(pos_s[0]) if pos_s else None),
        "max": (float(pos_s[-1]) if pos_s else None),
        "stdev": (float(statistics.pstdev(pos_s)) if len(pos_s) >= 2 else (0.0 if len(pos_s) == 1 else None)),
        "outliers": 0,
        "status_stat": ("INSUFFICIENT_N" if len(pos_s) < INSUFFICIENT_N else "OK"),
    }
    if pos_s and out["median"] is not None:
        med = float(out["median"])
        # simple outlier: > median + 5*IQR-like using p75-p25 approx via p75-median
        thr = med + 10.0 * max(1.0, (float(out["p75"] or med) - med))
        out["outliers"] = int(sum(1 for v in pos_s if v > thr))
    return out


def load_events(path: Path) -> List[Dict[str, Any]]:
    evs: List[Dict[str, Any]] = []
    if not path.exists():
        return evs
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and obj.get("event_name"):
                evs.append(obj)
    return evs


def group_traces(events: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    g: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in events:
        tid = str(e.get("trace_id") or "").strip()
        if not tid:
            continue
        g[tid].append(e)
    for tid in g:
        g[tid].sort(key=lambda x: (_parse_ts(x.get("event_ts_utc")) or 0.0, int(x.get("monotonic_ns") or 0)))
    return dict(g)


def first_event(evs: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    for e in evs:
        if e.get("event_name") == name:
            return e
    return None


def last_event(evs: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    found = None
    for e in evs:
        if e.get("event_name") == name:
            found = e
    return found


def duration_ms(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]], *, same_process: bool = False) -> Optional[float]:
    if not a or not b:
        return None
    if same_process and a.get("process_id") == b.get("process_id") and a.get("monotonic_ns") and b.get("monotonic_ns"):
        try:
            return (int(b["monotonic_ns"]) - int(a["monotonic_ns"])) / 1e6
        except Exception:
            pass
    ta = _parse_ts(a.get("event_ts_utc"))
    tb = _parse_ts(b.get("event_ts_utc"))
    if ta is None or tb is None:
        return None
    return (tb - ta) * 1000.0


def analyze_trace(tid: str, evs: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_name = Counter(e.get("event_name") for e in evs)
    duplicates = {k: v for k, v in by_name.items() if v > 1 and k in ("H3B_WS_RECEIVED", "H3B_DETECTED", "H3B_PLACE_FINISHED", "H3B_AUDIT_PERSIST_FINISHED")}
    order_idx = {n: i for i, n in enumerate(EVENT_ORDER)}
    ordering_violations: List[str] = []
    last_i = -1
    last_name = None
    for e in evs:
        n = str(e.get("event_name"))
        i = order_idx.get(n)
        if i is None:
            continue
        if last_i >= 0 and i < last_i:
            ordering_violations.append(f"{last_name}->{n}")
        last_i = i
        last_name = n

    audit_id = None
    execution_id = None
    order_id = None
    status = None
    event_id = None
    market = None
    side = None
    line = None
    for e in evs:
        audit_id = audit_id or e.get("audit_id")
        execution_id = execution_id or e.get("execution_id")
        order_id = order_id or e.get("order_id")
        event_id = event_id or e.get("event_id")
        market = market or e.get("market_type")
        side = side or e.get("side")
        line = line or e.get("line")
        if e.get("event_name") == "H3B_PLACE_FINISHED" and e.get("status"):
            status = e.get("status")
        elif e.get("event_name") == "H3B_FINAL_GATE_DECIDED" and e.get("status") in ("CAP_BLOCKED", "PASS"):
            if status is None or e.get("status") == "CAP_BLOCKED":
                status = e.get("status") if e.get("status") != "PASS" else status
        elif e.get("event_name") == "H3B_RESULT_PERSIST_FINISHED" and e.get("status"):
            status = e.get("status")
    if status is None:
        status = "UNKNOWN"

    missing_stages = [n for n in FUNNEL if first_event(evs, n) is None]
    if status == "LIVE_OK" and first_event(evs, "H3B_PLACE_FINISHED") is None:
        missing_stages.append("LIVE_OK")

    durations: Dict[str, Any] = {}
    clock_skew = False
    for key, a_name, b_name in LATENCY_SPECS:
        # live_ok metrics only when LIVE_OK
        if key.endswith("live_ok_ms") and status != "LIVE_OK":
            durations[key] = None
            continue
        a = first_event(evs, a_name)
        b = first_event(evs, b_name) if b_name != "H3B_AUDIT_GATE_DECIDED" else last_event(evs, b_name)
        # fallbacks for request timestamps
        if key in ("audit_to_request", "policy_to_request_ms") and b is None:
            b = first_event(evs, "H3B_EXEC_REQUEST_SENT")
        if key.startswith("request_") and a is None and a_name == "H3B_EXEC_REQUEST_CREATED":
            a = first_event(evs, "H3B_EXEC_REQUEST_SENT")
        same_service = bool(a and b and a.get("service") == b.get("service") and a.get("process_id") == b.get("process_id"))
        d = duration_ms(a, b, same_process=same_service)
        durations[key] = d
        if d is not None and d < 0:
            clock_skew = True
            durations[f"{key}_clock_skew_suspected"] = True

    present = {e.get("event_name") for e in evs}
    return {
        "trace_id": tid,
        "audit_id": audit_id,
        "execution_id": execution_id,
        "order_id": order_id,
        "status": status,
        "event_id": event_id,
        "market": market,
        "side": side,
        "line": line,
        "n_events": len(evs),
        "missing_stages": "|".join(missing_stages),
        "ordering_violations": "|".join(ordering_violations),
        "duplicate_events": json.dumps(duplicates, ensure_ascii=False),
        "clock_skew_suspected": clock_skew,
        "events_present": "|".join(sorted(str(x) for x in present)),
        **durations,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fieldnames or sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summarize(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    total = len(rows)
    summary = []
    by_status_rows = []
    for key, _, _ in LATENCY_SPECS:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        st = _stats(vals, total_n=total)
        summary.append({"metric": key, **st})
        # by status
        groups: Dict[str, List[float]] = defaultdict(list)
        group_n: Dict[str, int] = Counter(str(r.get("status") or "UNKNOWN") for r in rows)
        for r in rows:
            if r.get(key) is not None:
                groups[str(r.get("status") or "UNKNOWN")].append(float(r[key]))
        for status, vs in sorted(groups.items()):
            st2 = _stats(vs, total_n=int(group_n.get(status, 0)))
            by_status_rows.append({"metric": key, "status": status, **st2})
    # coverage funnel
    cov = []
    if total:
        for stage in FUNNEL + ["LIVE_OK"]:
            if stage == "LIVE_OK":
                n = sum(1 for r in rows if r.get("status") == "LIVE_OK")
            else:
                aliases = FUNNEL_ALIASES.get(stage, (stage,))
                n = sum(
                    1
                    for r in rows
                    if any(a in str(r.get("events_present") or "") for a in aliases)
                )
            cov.append({"stage": stage, "n": n, "pct": (100.0 * n / total) if total else 0.0})
    return summary, by_status_rows, cov


def render_daily_section(summary: List[Dict[str, Any]], cov: List[Dict[str, Any]], *, health: Dict[str, Any], n_traces: int, n_live: int) -> str:
    def m(name: str, field: str = "median") -> str:
        for r in summary:
            if r.get("metric") == name:
                v = r.get(field)
                return "n/a" if v is None else f"{float(v):.3f}"
        return "n/a"

    # dominant stage among segment medians (excluding end-to-end)
    segs = [
        "ws_to_detect_ms",
        "detect_to_audit_gate_ms",
        "audit_gate_to_persist_ms",
        "audit_persist_to_bridge_fetch_ms",
        "bridge_fetch_to_policy_ms",
        "policy_to_request_ms",
        "request_send_to_executor_receive_ms",
        "executor_receive_to_dryrun_start_ms",
        "dryrun_duration_ms",
        "dryrun_to_final_gate_ms",
        "final_gate_to_place_start_ms",
        "place_duration_ms",
    ]
    dom = "n/a"
    dom_v = -1.0
    for r in summary:
        if r.get("metric") in segs and r.get("median") is not None:
            if float(r["median"]) > dom_v:
                dom_v = float(r["median"])
                dom = str(r["metric"])

    cov_ws = next((c for c in cov if c.get("stage") == "LIVE_OK"), None)
    lines = [
        "## H3BUP End-to-End Latency",
        "",
        "| Métrica | Valor |",
        "|---|---|",
        f"| tracing status | {'ENABLED' if health.get('enabled') else 'DISABLED / NO_FILE'} |",
        f"| schema version | {health.get('schema_version', 1)} |",
        f"| traces totais | {n_traces} |",
        f"| traces LIVE_OK | {n_live} |",
        f"| coverage WS→LIVE_OK | {(100.0 * n_live / n_traces) if n_traces else 0.0:.1f}% |",
        f"| mediana WS→LIVE_OK | {m('ws_to_live_ok_ms')} ms |",
        f"| p95 WS→LIVE_OK | {m('ws_to_live_ok_ms', 'p95')} ms |",
        f"| mediana audit→request | {m('audit_to_request')} ms |",
        f"| mediana request→LIVE_OK | {m('request_to_live_ok_ms')} ms |",
        f"| mediana dry-run | {m('dryrun_duration_ms')} ms |",
        f"| mediana place | {m('place_duration_ms')} ms |",
        f"| etapa dominante | {dom} |",
        f"| trace events dropped | {health.get('trace_events_dropped', 0)} |",
        f"| clock skew violations | {health.get('clock_skew', 0)} |",
        f"| ordering violations | {health.get('ordering_violations', 0)} |",
        f"| status estatístico | {'INSUFFICIENT_N' if n_live < INSUFFICIENT_N else 'OK'} |",
        "",
        "### Funil de cobertura",
        "",
        "| Etapa | N | % |",
        "|---|---:|---:|",
    ]
    for c in cov:
        lines.append(f"| {c['stage']} | {c['n']} | {c['pct']:.1f}% |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.getenv("H3BUP_E2E_TRACE_PATH", "logs/h3bup_e2e_trace.jsonl"))
    ap.add_argument("--out-dir", default="logs")
    ap.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    ap.add_argument("--print-daily", action="store_true")
    args = ap.parse_args()

    inp = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date = args.date

    try:
        events = load_events(inp)
    except Exception:
        events = []

    traces = group_traces(events)
    rows = [analyze_trace(tid, evs) for tid, evs in traces.items()]
    summary, by_status, cov = summarize(rows)

    ordering_rows = []
    for r in rows:
        if r.get("ordering_violations"):
            ordering_rows.append(
                {
                    "trace_id": r["trace_id"],
                    "status": r.get("status"),
                    "ordering_violations": r.get("ordering_violations"),
                    "duplicate_events": r.get("duplicate_events"),
                }
            )

    write_csv(out_dir / f"h3bup_e2e_latency_trace_level_{date}.csv", rows)
    write_csv(out_dir / f"h3bup_e2e_latency_summary_{date}.csv", summary)
    write_csv(out_dir / f"h3bup_e2e_latency_by_status_{date}.csv", by_status)
    write_csv(out_dir / f"h3bup_e2e_latency_coverage_{date}.csv", cov)
    write_csv(out_dir / f"h3bup_e2e_ordering_violations_{date}.csv", ordering_rows)

    n_live = sum(1 for r in rows if r.get("status") == "LIVE_OK")
    clock_skew = sum(1 for r in rows if r.get("clock_skew_suspected"))
    health = {
        "enabled": inp.exists(),
        "schema_version": 1,
        "path": str(inp),
        "events": len(events),
        "traces": len(rows),
        "live_ok": n_live,
        "trace_events_dropped": 0,
        "clock_skew": clock_skew,
        "ordering_violations": len(ordering_rows),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        from ops.h3bup_e2e_trace import get_metrics

        health.update(get_metrics())
    except Exception:
        pass
    (out_dir / f"h3bup_e2e_trace_health_{date}.json").write_text(
        json.dumps(health, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    section = render_daily_section(summary, cov, health=health, n_traces=len(rows), n_live=n_live)
    if args.print_daily:
        print(section)
    else:
        print(json.dumps({"traces": len(rows), "live_ok": n_live, "events": len(events)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
