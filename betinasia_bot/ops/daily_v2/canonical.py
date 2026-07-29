"""Build canonical Daily V2 snapshot JSON."""

from __future__ import annotations

import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from . import SCHEMA_VERSION
from .extract import extract_source_manifest
from .performance import compute_settlement_and_performance
from .statuses import metric_envelope
from .time_windows import ReportWindow, resolve_window
from .universes import (
    classify_fast_buckets,
    load_executor_orders,
    load_open_order_ids,
    load_pnl_by_order_from_balance_csv,
)


def _git_commit(root: Path) -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    import json

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"_error": "parse_failed"}


def build_snapshot(
    *,
    root: Path,
    window: Optional[ReportWindow] = None,
    report_type: str = "DAILY_CLOSED",
    report_date=None,
    require_h3bup: bool = True,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    root = Path(root)
    win = window or resolve_window(report_type=report_type, report_date=report_date)
    run_id = run_id or uuid.uuid4().hex[:12]
    generated_at = datetime.now(timezone.utc)

    manifest = extract_source_manifest(win, root=root)
    exec_path = Path(manifest.get("executor_live", {}).get("path") or root / "logs/executor_live.jsonl")
    orders = load_executor_orders(exec_path, window=win, require_h3bup=require_h3bup)

    bal_meta = manifest.get("accounting_balance") or {}
    open_meta = manifest.get("accounting_open_stakes") or {}
    pnl_by_oid = load_pnl_by_order_from_balance_csv(Path(bal_meta["path"])) if bal_meta.get("exists") else {}
    open_oids = load_open_order_ids(Path(open_meta["path"])) if open_meta.get("exists") else set()

    acct_h = str((manifest.get("accounting_health") or {}).get("status") or "NOT_AVAILABLE")
    perf = compute_settlement_and_performance(
        orders=orders,
        pnl_by_oid=pnl_by_oid,
        open_oids=open_oids,
        accounting_health_status=acct_h,
    )
    fast = classify_fast_buckets(orders)

    # Health aggregates
    critical_sources = ["executor_live", "accounting_health"]
    report_health = "HEALTHY"
    for cs in critical_sources:
        st = (manifest.get(cs) or {}).get("status")
        if st in {"STALE", "FAILED"}:
            report_health = "CRITICAL" if cs.startswith("accounting") and report_type else "PARTIAL"
            if st == "FAILED":
                report_health = "CRITICAL"
        elif st in {"WATCH", "PARTIAL"} and report_health == "HEALTHY":
            report_health = "WATCH"
        elif st == "NOT_AVAILABLE" and cs == "executor_live":
            report_health = "CRITICAL"

    e2e_health = _load_json(Path((manifest.get("e2e_trace") or {}).get("path") or "missing"))
    # e2e_trace is jsonl — health from existence/freshness
    e2e_status = (manifest.get("e2e_trace") or {}).get("status") or "NOT_AVAILABLE"
    clv = _load_json(Path((manifest.get("clv_health") or {}).get("path") or "missing"))

    exceptions = []
    for oid, o in orders.items():
        if o.get("stake") is not None and abs(float(o["stake"]) - 10.0) > 1e-6:
            # current H3BUP stake must be 10; flag mismatch (legacy 20 separated)
            sev = "CRITICAL" if abs(float(o["stake"]) - 20.0) > 1e-6 else "WARNING"
            exceptions.append(
                {
                    "alert_id": f"stake_mismatch:{oid}",
                    "severity": sev,
                    "evidence": {"order_id": oid, "stake": o.get("stake")},
                    "affected_metrics": ["stake_placed_sum"],
                    "status": "OPEN",
                }
            )
        if o.get("policy_version") and "H3BUP_vNext" not in str(o.get("policy_version")):
            exceptions.append(
                {
                    "alert_id": f"policy_mix:{oid}",
                    "severity": "CRITICAL",
                    "evidence": {"order_id": oid, "policy_version": o.get("policy_version")},
                    "affected_metrics": ["live_ok_count"],
                    "status": "OPEN",
                }
            )

    # Latency section from fast buckets + E2E placeholder reading analyzer optionally
    latency = {
        "daily_fast_le_6s": metric_envelope(
            value=fast["DAILY_FAST_LE_6S"]["n"],
            unit="count",
            n=fast["DAILY_FAST_LE_6S"]["n"],
            denominator=fast["n_with_pre_submit_ms"],
            coverage_pct=(
                100.0 * fast["n_with_pre_submit_ms"] / len(orders) if orders else None
            ),
            status="AVAILABLE",
            metric_version="v2.0",
            source="executor_live.jsonl",
            notes=["threshold pre_submit_ms <= 6000", "DAILY_FAST_LE_6S"],
        ),
        "study_fast_lt_4s": metric_envelope(
            value=fast["STUDY_FAST_LT_4S"]["n"],
            unit="count",
            n=fast["STUDY_FAST_LT_4S"]["n"],
            status="AVAILABLE",
            metric_version="v2.0",
            source="executor_live.jsonl",
            notes=["exploratory only", "STUDY_FAST_LT_4S pre_submit_ms < 4000"],
        ),
        "pre_submit_ms_na": metric_envelope(
            value=fast["PRE_SUBMIT_MS_NA"]["n"],
            unit="count",
            n=fast["PRE_SUBMIT_MS_NA"]["n"],
            status="AVAILABLE",
            notes=["missing not coerced to slow"],
        ),
        "e2e_source_status": e2e_status,
        "detect_to_audit_overhead": metric_envelope(
            value=None,
            status="WATCH",
            metric_version="v2.0",
            source="h3bup_e2e_trace",
            notes=["overhead remains WATCH until re-evaluation"],
        ),
    }

    # Try E2E analyzer if available
    try:
        from ops.analyze_h3bup_e2e_latency import analyze_trace, group_traces, load_events, summarize

        tpath = Path((manifest.get("e2e_trace") or {}).get("path") or "")
        if tpath.exists():
            evs = load_events(tpath)
            trs = group_traces(evs)
            rows = [analyze_trace(tid, evs2) for tid, evs2 in trs.items()]
            summary, _by, cov = summarize(rows)
            n_live = sum(1 for r in rows if r.get("status") == "LIVE_OK")
            # pick ws_to_live_ok if present
            med = None
            p95 = None
            nseg = 0
            for s in summary or []:
                if str(s.get("segment") or s.get("name") or "").lower() in {
                    "ws_to_live_ok",
                    "ws→live_ok",
                    "e2e_total",
                }:
                    med = s.get("p50") or s.get("median")
                    p95 = s.get("p95") or s.get("p90")
                    nseg = int(s.get("n") or 0)
            latency["e2e_ws_to_live_ok"] = metric_envelope(
                value=med,
                unit="ms",
                n=nseg or n_live,
                status="INSUFFICIENT_N" if (nseg or n_live) == 0 else "AVAILABLE",
                metric_version="v2.0",
                source="h3bup_e2e_trace.jsonl",
                notes=[f"n_traces={len(rows)}", f"n_live={n_live}"],
            )
            latency["e2e_coverage"] = cov
        else:
            latency["e2e_ws_to_live_ok"] = metric_envelope(
                status="MISSING", unit="ms", n=0, source="h3bup_e2e_trace.jsonl"
            )
    except Exception as e:
        latency["e2e_ws_to_live_ok"] = metric_envelope(
            status="FAILED", unit="ms", n=0, notes=[str(e)[:160]]
        )

    clv_section = {
        "collection_status": clv.get("status") or (manifest.get("clv_health") or {}).get("status"),
        "collection_started_at_utc": clv.get("collection_started_at_utc"),
        "post_5m_valid_strict": metric_envelope(
            value=clv.get("post_5m_valid_strict"),
            unit="count",
            n=int(clv.get("post_5m_valid_strict") or 0) if clv.get("post_5m_valid_strict") is not None else 0,
            status=(
                "INSUFFICIENT_N"
                if int(clv.get("live_ok_after_activation") or 0) < 30
                else ("AVAILABLE" if clv else "MISSING")
            )
            if clv
            else "MISSING",
            notes=["forward-only", "POST_5M"],
        ),
        "post_15m_valid_strict": metric_envelope(
            value=clv.get("post_15m_valid_strict"),
            unit="count",
            status="INSUFFICIENT_N" if int(clv.get("live_ok_after_activation") or 0) < 30 else ("AVAILABLE" if clv else "MISSING"),
            notes=["POST_15M"],
        ),
        "closing_valid_strict": metric_envelope(
            value=clv.get("closing_valid_strict"),
            unit="count",
            status="INSUFFICIENT_N" if int(clv.get("live_ok_after_activation") or 0) < 30 else ("AVAILABLE" if clv else "MISSING"),
            notes=["CLOSING", "requires pre-kickoff snapshot"],
        ),
        "fair_edge": metric_envelope(
            value=None,
            status="NOT_IMPLEMENTED",
            metric_version="v2.0",
            notes=["Phase 2D not started"],
        ),
        "funnel": {
            "live_ok_after_activation": clv.get("live_ok_after_activation"),
            "obligations_expected": clv.get("obligations_expected"),
            "obligations_created": clv.get("obligations_created"),
            "source_missing": clv.get("source_missing"),
            "kickoff_missing": clv.get("kickoff_missing"),
        },
    }

    policy_id = "H3BUP_vNext"
    policy_version = os.getenv("H3BUP_POLICY_VERSION", "H3BUP_vNext_20260629")

    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "report_type": win.report_type,
        "report_date_utc": win.report_date_utc.isoformat(),
        "window_start_utc": win.window_start_utc.isoformat(),
        "window_end_utc": win.window_end_utc.isoformat(),
        "report_cutoff_utc": win.report_cutoff_utc.isoformat(),
        "generated_at_utc": generated_at.isoformat(),
        "git_commit": _git_commit(root),
        "policy_id": policy_id,
        "policy_version": policy_version,
        "policy_fingerprint": None,
        "source_manifest": manifest,
        "report_health": {
            "status": report_health,
            "notes": ["REPORT_HEALTH distinct from STRATEGY_OPERATIONS_HEALTH"],
        },
        "operations_health": {
            "status": "WATCH",
            "notes": ["derived from services externally; Daily V2 does not restart services"],
        },
        "data_quality": {
            "accounting": acct_h,
            "e2e": e2e_status,
            "clv": clv_section["collection_status"],
        },
        "statistical_readiness": {
            "roi": perf["roi_settled"]["status"],
            "clv": clv_section["post_5m_valid_strict"]["status"],
            "e2e": (latency.get("e2e_ws_to_live_ok") or {}).get("status"),
        },
        "execution_funnel": {
            "live_ok": metric_envelope(
                value=len(orders),
                unit="count",
                n=len(orders),
                status="AVAILABLE" if (manifest.get("executor_live") or {}).get("status") not in {"FAILED", "NOT_AVAILABLE"} else "FAILED",
                source="executor_live.jsonl",
            ),
            "order_ids": sorted(orders.keys()),
            "fast_buckets": {
                k: {kk: vv for kk, vv in v.items() if kk != "order_ids"} for k, v in fast.items() if isinstance(v, dict)
            },
        },
        "settlement": {
            "maturity_status": perf["maturity_status"],
            "n_open": perf["n_open"],
            "n_settled": perf["n_settled"],
            "n_void_push": perf["n_void_push"],
            "n_missing_accounting": perf["n_missing_accounting"],
            "stake_placed_sum": perf["stake_placed_sum"],
            "stake_settled_sum": perf["stake_settled_sum"],
            "pnl_settled_sum": perf["pnl_settled_sum"],
        },
        "performance": {
            "roi_settled": perf["roi_settled"],
            "roiw_total_v1": perf["roiw_total_v1"],
            "roiw_total_v2": perf["roiw_total_v2"],
            "principal_metric": perf["principal_metric"],
            "complementary_metric": perf["complementary_metric"],
        },
        "latency": latency,
        "clv": clv_section,
        "concentration": {
            "status": "INSUFFICIENT_N" if len(orders) < 30 else "AVAILABLE",
            "notes": ["only emitted when N sufficient"],
        },
        "exceptions": exceptions,
        "methodology": {
            "cohort_timestamp": "created_at UTC",
            "post_date_usage": "accounting freshness / settlement metadata only",
            "daily_fast": "DAILY_FAST_LE_6S: pre_submit_ms <= 6000",
            "study_fast": "STUDY_FAST_LT_4S: pre_submit_ms < 4000 (exploratory)",
            "roi_settled": "sum(pnl_confirmed_settled)/sum(stake_confirmed_settled)",
            "roiw_total_v1": "(sum pnl / sum exposure)*100; may include open if in ledger",
            "absence_policy": "missing/stale/not_calculable must not appear as zero",
            "fair_edge": "NOT_IMPLEMENTED",
        },
    }
    # Serialize orders without datetime objects
    for oid in list(orders.keys()):
        orders[oid].pop("created_at_dt", None)
    snapshot["execution_funnel"]["orders_sample"] = {
        k: orders[k] for k in list(sorted(orders.keys()))[:50]
    }
    return snapshot
