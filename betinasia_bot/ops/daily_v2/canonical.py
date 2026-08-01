"""Build canonical Daily V2 snapshot JSON (P0 enriched)."""

from __future__ import annotations

import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from . import SCHEMA_VERSION
from .clv_section import build_clv_section
from .diff_previous import diff_snapshots, find_previous_snapshot
from .e2e_funnel import build_e2e_and_funnel
from .extract import extract_source_manifest
from .friendly_section import build_friendly_section
from .health_model import build_health_model, derive_alerts, evaluate_config_file
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
    previous_snapshot: Optional[Dict[str, Any]] = None,
    out_dir: Optional[Path] = None,
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
    friendly_section = build_friendly_section(
        root=root,
        orders=orders,
        pnl_by_oid=pnl_by_oid,
        open_oids=open_oids,
        accounting_health_status=acct_h,
    )

    # Config fingerprint (NOT mtime STALE)
    policy_path = Path((manifest.get("policy_current") or {}).get("path") or root / "logs/wf_policy_current.json")
    risk_path = Path((manifest.get("risk_params") or {}).get("path") or root / "logs/bridge_risk_params.json")
    config_eval = {
        "policy": evaluate_config_file(policy_path),
        "risk_params": evaluate_config_file(risk_path),
    }
    # Overlay config status onto manifest (replace age-based STALE)
    for key, ce in (("policy_current", config_eval["policy"]), ("risk_params", config_eval["risk_params"])):
        if key in manifest and isinstance(manifest[key], dict):
            manifest[key] = dict(manifest[key])
            manifest[key]["status"] = ce.get("drift") or "UNVERIFIED"
            manifest[key]["config_eval"] = ce
            manifest[key]["notes"] = ["mtime age ignored; fingerprint/content used"]

    e2e_pack = build_e2e_and_funnel(root, window_label="all_traces_available")
    clv_section = build_clv_section(root)

    # Fast latency envelopes (speed buckets — separate from funnel)
    n_orders = len(orders)
    n_pre = int(fast.get("n_with_pre_submit_ms") or 0)
    latency = {
        "daily_fast_le_6s": metric_envelope(
            value=fast["DAILY_FAST_LE_6S"]["n"],
            unit="count",
            n=fast["DAILY_FAST_LE_6S"]["n"],
            denominator=n_pre,
            coverage_pct=(100.0 * n_pre / n_orders) if n_orders else None,
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
            coverage_pct=(100.0 * fast["PRE_SUBMIT_MS_NA"]["n"] / n_orders) if n_orders else None,
            status="AVAILABLE",
            notes=["missing not coerced to slow", f"coverage_missing={fast['PRE_SUBMIT_MS_NA']['n']}/{n_orders}"],
        ),
        "e2e_source_status": (manifest.get("e2e_trace") or {}).get("status") or "NOT_AVAILABLE",
        "segments": e2e_pack.get("segments") or {},
        "n_traces": e2e_pack.get("n_traces"),
        "n_live_ok_traces": e2e_pack.get("n_live_ok"),
        "full_trace_coverage_pct": e2e_pack.get("full_trace_coverage_pct"),
        "dominant_stage": e2e_pack.get("dominant_stage"),
        "ordering_violations": e2e_pack.get("ordering_violations") or 0,
        "clock_skew": e2e_pack.get("clock_skew") or 0,
        "trace_events_dropped": e2e_pack.get("trace_events_dropped") or 0,
        "detect_to_audit_overhead": e2e_pack.get("detect_to_audit_overhead")
        or metric_envelope(status="WATCH", unit="ms", notes=["overhead remains WATCH"]),
    }
    # Convenience top-level E2E metric
    ws = (e2e_pack.get("segments") or {}).get("ws_to_live_ok")
    if ws:
        latency["e2e_ws_to_live_ok"] = ws
    else:
        latency["e2e_ws_to_live_ok"] = metric_envelope(
            status="MISSING" if not e2e_pack.get("available") else "INSUFFICIENT_N",
            unit="ms",
            n=0,
            source="h3bup_e2e_trace.jsonl",
        )

    settlement = {
        "maturity_status": perf["maturity_status"],
        "live_ok_total": perf.get("live_ok_total", n_orders),
        "n_open": perf["n_open"],
        "n_settled": perf["n_settled"],
        "n_void_push": perf["n_void_push"],
        "n_missing_accounting": perf["n_missing_accounting"],
        "stake_placed_sum": perf["stake_placed_sum"],
        "stake_settled_sum": perf.get("stake_settled_sum"),
        "pnl_settled_sum": perf.get("pnl_settled_sum"),
        "stake_placed": perf.get("stake_placed"),
        "stake_resolved_total": perf.get("stake_resolved_total"),
        "stake_decided_ex_void": perf.get("stake_decided_ex_void"),
        "stake_void": perf.get("stake_void"),
        "stake_open": perf.get("stake_open"),
        "pnl_resolved": perf.get("pnl_resolved"),
        "pnl_decided_ex_void": perf.get("pnl_decided_ex_void"),
    }

    health = build_health_model(
        manifest=manifest,
        settlement=settlement,
        clv=clv_section,
        latency=latency,
        config_eval=config_eval,
        artifacts_ok=True,
        schema_ok=True,
    )

    # Stake/policy mismatch exceptions (keep existing contract for tests)
    exceptions = []
    for oid, o in orders.items():
        if o.get("stake") is not None and abs(float(o["stake"]) - 10.0) > 1e-6:
            sev = "CRITICAL" if abs(float(o["stake"]) - 20.0) > 1e-6 else "WARNING"
            exceptions.append(
                {
                    "alert_id": f"stake_mismatch:{oid}",
                    "severity": sev,
                    "evidence": {"order_id": oid, "stake": o.get("stake")},
                    "affected_metrics": ["stake_placed_sum"],
                    "status": "OPEN",
                    "message": f"stake mismatch order {oid}",
                    "first_seen_utc": generated_at.isoformat(),
                    "last_seen_utc": generated_at.isoformat(),
                    "resolution_hint": "legacy stake or misconfigured bridge",
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
                    "message": f"policy mix order {oid}",
                    "first_seen_utc": generated_at.isoformat(),
                    "last_seen_utc": generated_at.isoformat(),
                    "resolution_hint": "exclude non-H3BUP from cohort",
                }
            )

    health_alerts = derive_alerts(
        health=health,
        settlement=settlement,
        clv=clv_section,
        latency=latency,
        parity_status=None,
        now_iso=generated_at.isoformat(),
    )
    # Merge: stake/policy first, then health-derived (dedupe by alert_id)
    seen = {e["alert_id"] for e in exceptions}
    for a in health_alerts:
        if a["alert_id"] in seen:
            continue
        exceptions.append(a)
        seen.add(a["alert_id"])

    # Execution funnel: E2E stages + LIVE_OK cohort count overlay
    funnel_rows = list(e2e_pack.get("funnel") or [])
    # Ensure LIVE_OK cohort count appears even if E2E missing
    if not any(r.get("event") == "LIVE_OK" for r in funnel_rows):
        funnel_rows.append(
            {
                "step": "LIVE_OK",
                "event": "LIVE_OK",
                "n": n_orders,
                "pct_prev": None,
                "pct_initial": None,
                "status": "AVAILABLE",
            }
        )
    else:
        for r in funnel_rows:
            if r.get("event") == "LIVE_OK" and n_orders:
                # prefer cohort LIVE_OK for closed day when e2e is all-traces
                r["cohort_live_ok"] = n_orders

    policy_id = "H3BUP_vNext"
    policy_version = os.getenv("H3BUP_POLICY_VERSION", "H3BUP_vNext_20260629")

    snapshot: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "report_kind": "H3BUP_DAILY_V2",
        "official_status": "PREVIEW_NOT_OFFICIAL",
        "report_type": win.report_type,
        "report_date_utc": win.report_date_utc.isoformat(),
        "window_start_utc": win.window_start_utc.isoformat(),
        "window_end_utc": win.window_end_utc.isoformat(),
        "report_cutoff_utc": win.report_cutoff_utc.isoformat(),
        "generated_at_utc": generated_at.isoformat(),
        "cutoffs": {
            "cohort_window_start_utc": win.window_start_utc.isoformat(),
            "cohort_window_end_utc": win.window_end_utc.isoformat(),
            "performance_as_of_utc": win.report_cutoff_utc.isoformat(),
            "v1_report_cutoff_utc": None,  # filled by __main__ parity
            "v2_comparison_cutoff_utc": None,
            "v2_generated_at_utc": generated_at.isoformat(),
        },
        "git_commit": _git_commit(root),
        "policy_id": policy_id,
        "policy_version": policy_version,
        "policy_fingerprint": (config_eval.get("policy") or {}).get("fingerprint"),
        "source_manifest": manifest,
        "report_health": health.get("report_health") or {"status": "HEALTHY"},
        "operations_health": health.get("operations_health") or {"status": "WATCH"},
        "data_quality": health.get("data_quality") or {"status": "WATCH"},
        "statistical_readiness": health.get("statistical_readiness") or {"status": "INSUFFICIENT_N"},
        "config": config_eval,
        "execution_funnel": {
            "window_label": e2e_pack.get("window_label") or "cohort_created_at_utc",
            "live_ok": metric_envelope(
                value=n_orders,
                unit="count",
                n=n_orders,
                status=(
                    "AVAILABLE"
                    if (manifest.get("executor_live") or {}).get("status") not in {"FAILED", "NOT_AVAILABLE"}
                    else "FAILED"
                ),
                source="executor_live.jsonl",
            ),
            "order_ids": sorted(orders.keys()),
            "stages": funnel_rows,
            "block_reasons": e2e_pack.get("block_reasons") or [],
            "fast_buckets": {
                k: {kk: vv for kk, vv in v.items() if kk != "order_ids"}
                for k, v in fast.items()
                if isinstance(v, dict)
            },
        },
        "settlement": settlement,
        "performance": {
            "roi_settled": perf["roi_settled"],
            "roi_resolved": perf.get("roi_resolved") or perf["roi_settled"],
            "roi_decided_ex_void": perf.get("roi_decided_ex_void"),
            "roiw_total_v1": perf["roiw_total_v1"],
            "roiw_total_v2": perf["roiw_total_v2"],
            "principal_metric": perf.get("principal_metric") or "roi_resolved",
            "complementary_metric": perf.get("complementary_metric") or "roi_decided_ex_void",
            "parity_legacy_metric": perf.get("parity_legacy_metric") or "roiw_total_v1",
            "formulas": perf.get("formulas")
            or {
                "roi_resolved": "pnl_resolved / stake_resolved_total (void stake in denom)",
                "roi_decided_ex_void": "pnl_decided_ex_void / stake_decided_ex_void",
            },
        },
        "latency": latency,
        "e2e": {
            "available": e2e_pack.get("available"),
            "n_traces": e2e_pack.get("n_traces"),
            "n_live_ok": e2e_pack.get("n_live_ok"),
            "segments": e2e_pack.get("segments") or {},
            "dominant_stage": e2e_pack.get("dominant_stage"),
            "full_trace_coverage_pct": e2e_pack.get("full_trace_coverage_pct"),
            "ordering_violations": e2e_pack.get("ordering_violations") or 0,
            "clock_skew": e2e_pack.get("clock_skew") or 0,
            "trace_events_dropped": e2e_pack.get("trace_events_dropped") or 0,
            "detect_to_audit_overhead": latency.get("detect_to_audit_overhead"),
            "error": e2e_pack.get("error"),
        },
        "clv": clv_section,
        "friendly_breakdown": friendly_section,
        "concentration": {
            "status": "INSUFFICIENT_N" if n_orders < 30 else "AVAILABLE",
            "notes": ["only emitted when N sufficient"],
        },
        "exceptions": exceptions,
        "previous_diff": None,
        "methodology": {
            "cohort_timestamp": "created_at UTC",
            "post_date_usage": "accounting freshness / settlement metadata only",
            "daily_fast": "DAILY_FAST_LE_6S: pre_submit_ms <= 6000",
            "study_fast": "STUDY_FAST_LT_4S: pre_submit_ms < 4000 (exploratory)",
            "principal_metric": "roi_resolved = pnl_resolved / stake_resolved_total (void in denom)",
            "roi_settled": "legacy alias of roi_resolved",
            "roi_decided_ex_void": "pnl_decided_ex_void / stake_decided_ex_void",
            "roiw_total_v1": "(sum pnl / sum exposure)*100; may include open if in ledger — appendix parity only",
            "roiw_total_v2": "settled-aware percent complementary",
            "absence_policy": "missing/stale/not_calculable must not appear as zero",
            "fair_edge": "NOT_IMPLEMENTED",
            "config_stale_policy": "static config uses fingerprint/drift, not mtime age",
            "friendly_breakdown": (
                "shadow diagnostic FRIENDLY_CLASS_V1 — not an operational filter; "
                "UNCLASSIFIED != NON_FRIENDLY"
            ),
        },
        "safety": {
            "alters_execution": False,
            "alters_policy": False,
            "alters_stake": False,
            "creates_orders": False,
            "opens_betslips": False,
            "alters_friendly_filter": False,
        },
    }

    # Previous V2 diff
    prev = previous_snapshot
    if prev is None and out_dir is not None:
        prev_path = find_previous_snapshot(Path(out_dir), report_date=snapshot["report_date_utc"], current_run_id=run_id)
        if prev_path:
            prev = _load_json(prev_path)
    if prev and isinstance(prev, dict) and prev.get("run_id"):
        snapshot["previous_diff"] = diff_snapshots(prev, snapshot)

    for oid in list(orders.keys()):
        orders[oid].pop("created_at_dt", None)
    snapshot["execution_funnel"]["orders_sample"] = {k: orders[k] for k in list(sorted(orders.keys()))[:50]}
    return snapshot
