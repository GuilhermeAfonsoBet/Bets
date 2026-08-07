"""P0 health model: REPORT / OPERATIONS / DATA_QUALITY / STATISTICAL_READINESS + alerts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


EXPECTED_H3BUP = {
    "policy_id": "H3BUP_vNext",
    "policy_version_prefix": "H3BUP_vNext",
    "stake": 10.0,
    "odd_lo": 1.85,
    "odd_hi": 2.15,
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_fingerprint(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except Exception:
        return None


def evaluate_config_file(path: Path, *, expected_policy_needle: str = "H3BUP_vNext") -> Dict[str, Any]:
    """Config health by content/fingerprint — NOT mtime age."""
    if not path.exists():
        return {"file_status": "MISSING", "runtime_status": "UNVERIFIED", "fingerprint": None, "drift": "MISSING"}
    fp = file_fingerprint(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"file_status": "INVALID", "runtime_status": "UNVERIFIED", "fingerprint": fp, "drift": "INVALID"}
    blob = json.dumps(data, sort_keys=True, default=str)
    has_h3b = expected_policy_needle in blob
    # Without live runtime introspection, matched file that parses = CURRENT_UNCHANGED
    status = "CURRENT_UNCHANGED" if has_h3b or path.name.startswith("bridge") or path.name.startswith("wf") else "CURRENT_MATCHED"
    # If explicitly empty / wrong shape
    if data is None or (isinstance(data, dict) and not data):
        status = "UNVERIFIED"
    return {
        "file_status": "OK",
        "runtime_status": "UNVERIFIED",  # runtime match not probed without process inspect
        "fingerprint": fp,
        "drift": status,
        "notes": ["mtime age ignored; fingerprint/content used"],
    }


def build_health_model(
    *,
    manifest: Dict[str, Any],
    settlement: Dict[str, Any],
    clv: Dict[str, Any],
    latency: Dict[str, Any],
    config_eval: Dict[str, Any],
    artifacts_ok: bool = True,
    schema_ok: bool = True,
) -> Dict[str, Any]:
    # REPORT_HEALTH: generation integrity
    report = "HEALTHY" if artifacts_ok and schema_ok else "FAILED"

    # OPERATIONS: writers/services proxies via health files
    ops = "HEALTHY"
    for key in ("executor_live", "accounting_health", "e2e_trace", "clv_health"):
        st = str((manifest.get(key) or {}).get("status") or "NOT_AVAILABLE")
        if st == "FAILED":
            ops = "FAILED"
            break
        if st in {"STALE", "WATCH", "PARTIAL"} and ops == "HEALTHY":
            ops = "WATCH"
        if st == "NOT_AVAILABLE" and key in {"executor_live", "accounting_health"} and ops == "HEALTHY":
            ops = "WATCH"

    # DATA_QUALITY
    dq = "HEALTHY"
    for key, meta in (manifest or {}).items():
        if key in {"policy_current", "risk_params"}:
            continue  # handled via config fingerprint
        st = str((meta or {}).get("status") or "")
        if st == "FAILED":
            dq = "FAILED"
            break
        if st == "STALE" and dq in {"HEALTHY", "WATCH"}:
            dq = "STALE"
        elif st in {"WATCH", "PARTIAL"} and dq == "HEALTHY":
            dq = "WATCH"
    if int(settlement.get("n_missing_accounting") or 0) > 0 and dq == "HEALTHY":
        dq = "PARTIAL"
    if str((clv or {}).get("collection_status") or "").upper() == "WATCH" and dq == "HEALTHY":
        dq = "WATCH"
    # config drift elevates
    for _name, ce in (config_eval or {}).items():
        if (ce or {}).get("drift") == "CONFIG_DRIFT":
            dq = "FAILED"
        elif (ce or {}).get("drift") == "INVALID" and dq == "HEALTHY":
            dq = "WATCH"

    # STATISTICAL_READINESS
    n_live = int(settlement.get("live_ok_total") or settlement.get("n_live_ok") or 0)
    n_settled = int(settlement.get("n_settled") or 0) + int(settlement.get("n_void_push") or 0)
    clv_n = 0
    try:
        clv_n = int((((clv.get("performance") or {}).get("POST_5M") or {}).get("n")) or 0)
    except Exception:
        clv_n = int((clv.get("post_5m_valid_strict") or {}).get("value") or clv.get("post_5m_valid_strict") or 0 or 0)
    e2e_n = 0
    try:
        e2e_n = int((((latency.get("segments") or {}).get("ws_to_live_ok") or {}).get("n")) or 0)
    except Exception:
        pass
    if n_live < 30 or n_settled < 30 or clv_n < 30 or e2e_n < 30:
        stats = "INSUFFICIENT_N"
    else:
        stats = "AVAILABLE"
    if int(settlement.get("n_open") or 0) > 0 and stats == "AVAILABLE":
        stats = "PARTIAL"

    return {
        "report_health": {"status": report},
        "operations_health": {"status": ops},
        "data_quality": {"status": dq},
        "statistical_readiness": {"status": stats},
        "config": config_eval,
    }


def derive_alerts(
    *,
    health: Dict[str, Any],
    settlement: Dict[str, Any],
    clv: Dict[str, Any],
    latency: Dict[str, Any],
    parity_status: Optional[str] = None,
    now_iso: Optional[str] = None,
) -> List[Dict[str, Any]]:
    now = now_iso or _utcnow()
    alerts: List[Dict[str, Any]] = []

    def add(alert_id: str, severity: str, message: str, affected: List[str], evidence: Any = None, hint: str = ""):
        alerts.append(
            {
                "alert_id": alert_id,
                "severity": severity,
                "status": "OPEN",
                "first_seen_utc": now,
                "last_seen_utc": now,
                "message": message,
                "affected_metrics": affected,
                "evidence": evidence or {},
                "resolution_hint": hint,
            }
        )

    if (health.get("statistical_readiness") or {}).get("status") == "INSUFFICIENT_N":
        add("CLV_INSUFFICIENT_N", "INFO", "N estatístico insuficiente para inferência", ["clv", "roi"], hint="aguardar amostra")
    if str((latency.get("detect_to_audit_overhead") or {}).get("status") or "") == "WATCH":
        add("E2E_OVERHEAD_WATCH", "WATCH", "Overhead detect→audit acima da baseline / em observação", ["e2e"], latency.get("detect_to_audit_overhead"))
    if (health.get("data_quality") or {}).get("status") == "STALE":
        add("SOURCE_STALE", "WARNING", "Fonte de dados operacional stale", ["data_quality"])
    for name, ce in ((health.get("config") or {}) or {}).items():
        if (ce or {}).get("drift") == "CONFIG_DRIFT":
            add("CONFIG_DRIFT", "CRITICAL", f"Drift de configuração em {name}", ["policy", "risk_params"], ce)
    if int(settlement.get("n_missing_accounting") or 0) > 0:
        add("MISSING_ACCOUNTING", "WARNING", "Ordens sem join de accounting", ["roi"], {"n": settlement.get("n_missing_accounting")})
    if int(settlement.get("n_open") or 0) > 0:
        add("SETTLEMENT_PARTIAL", "INFO", "Coorte parcialmente liquidada", ["roi", "maturity"], {"n_open": settlement.get("n_open")})
    if str((clv or {}).get("collection_status") or "").upper() == "WATCH":
        add("CLV_INSUFFICIENT_N", "WATCH", "CLV collection em WATCH / amostra insuficiente", ["clv"])
    backlog = (clv.get("funnel") or {}).get("retry_backlog")
    if backlog is None:
        backlog = clv.get("retry_backlog")
    try:
        if int(backlog or 0) > 0:
            add("CLV_BACKLOG", "WATCH", "Retry backlog CLV > 0", ["clv"], {"retry_backlog": backlog})
    except Exception:
        pass
    if int((clv.get("funnel") or {}).get("source_missing") or clv.get("source_missing") or 0) > 0:
        add("CLV_SOURCE_MISSING", "WARNING", "Snapshots CLV com source missing", ["clv"])
    if int((clv.get("funnel") or {}).get("kickoff_missing") or clv.get("kickoff_missing") or 0) > 0:
        add("KICKOFF_MISSING", "WARNING", "Kickoff missing em obligations CLV", ["clv", "closing"])
    if int(latency.get("ordering_violations") or 0) > 0:
        add("TRACE_ORDERING_VIOLATIONS", "WATCH", "Ordering violations no E2E", ["e2e"], {"n": latency.get("ordering_violations")})
    if int(latency.get("clock_skew") or 0) > 0:
        add("TRACE_CLOCK_SKEW", "WATCH", "Clock skew / negative durations no E2E", ["e2e"], {"n": latency.get("clock_skew")})
    if parity_status and parity_status not in {"CUTOFF_ALIGNED", "MATCH"}:
        if "UNAVAILABLE" in str(parity_status):
            add("PARITY_UNAVAILABLE", "INFO", f"Paridade V1×V2: {parity_status}", ["parity"])
        else:
            add("PARITY_DIFFERENCE", "WATCH", f"Paridade V1×V2: {parity_status}", ["parity"])

    # Deduplicate by alert_id (keep first)
    seen = set()
    out = []
    for a in alerts:
        if a["alert_id"] in seen:
            continue
        seen.add(a["alert_id"])
        out.append(a)
    return out
