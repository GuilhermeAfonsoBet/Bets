"""Source extraction layer for Daily V2 (single report_cutoff_utc)."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .time_windows import ReportWindow, ensure_utc


def _file_meta(path: Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False, "mtime_utc": None, "size": None, "sha256_prefix": None}
    st = p.stat()
    mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
    h = hashlib.sha256()
    # checksum prefix only (first 1MB) for large jsonl
    with p.open("rb") as f:
        chunk = f.read(1024 * 1024)
        h.update(chunk)
    return {
        "path": str(p),
        "exists": True,
        "mtime_utc": mtime.isoformat(),
        "size": st.st_size,
        "sha256_prefix": h.hexdigest()[:16],
        "source_cutoff_utc": mtime.isoformat(),
    }


def _age_seconds(mtime_iso: Optional[str], cutoff: datetime) -> Optional[float]:
    if not mtime_iso:
        return None
    try:
        mt = datetime.fromisoformat(mtime_iso.replace("Z", "+00:00"))
        return (ensure_utc(cutoff) - ensure_utc(mt)).total_seconds()
    except Exception:
        return None


def classify_freshness(age_s: Optional[float], *, stale_after_s: float, watch_after_s: float) -> str:
    if age_s is None:
        return "NOT_AVAILABLE"
    if age_s < 0:
        return "WATCH"  # source cutoff after report cutoff
    if age_s <= watch_after_s:
        return "HEALTHY"
    if age_s <= stale_after_s:
        return "WATCH"
    return "STALE"


def extract_source_manifest(window: ReportWindow, *, root: Path | None = None) -> Dict[str, Any]:
    root = Path(root or os.getcwd())
    paths = {
        "executor_live": root / os.getenv("EXECUTOR_JSONL", "logs/executor_live.jsonl"),
        "accounting_health": root / os.getenv("ACCOUNTING_HEALTH_JSON", "logs/accounting/accounting_health.json"),
        "accounting_daily_report": root / os.getenv("ACCOUNTING_DAILY_JSON", "logs/accounting_daily_report.json"),
        "e2e_trace": root / os.getenv("H3BUP_E2E_TRACE_PATH", "logs/h3bup_e2e_trace.jsonl"),
        "clv_health": root / os.getenv("H3BUP_CLV_HEALTH_PATH", "logs/h3bup_clv_health.json"),
        "clv_obligations": root / os.getenv("H3BUP_CLV_OBLIGATIONS_PATH", "logs/h3bup_clv_obligations.jsonl"),
        "policy_current": root / os.getenv("DAILY_WF_POLICY_CURRENT", "logs/wf_policy_current.json"),
        "risk_params": root / os.getenv("BRIDGE_RISK_PARAMS_JSON", "logs/bridge_risk_params.json"),
    }
    # latest balance / open stakes
    acct_dir = root / os.getenv("ACCOUNTING_OUT_DIR", "logs/accounting")
    balance = None
    open_stakes = None
    if acct_dir.exists():
        bals = sorted(acct_dir.glob("*__balance.csv")) + sorted(acct_dir.glob("*balance*.csv"))
        opens = sorted(acct_dir.glob("*__open_stakes.csv")) + sorted(acct_dir.glob("*open_stakes*.csv"))
        balance = bals[-1] if bals else None
        open_stakes = opens[-1] if opens else None
    if balance:
        paths["accounting_balance"] = balance
    if open_stakes:
        paths["accounting_open_stakes"] = open_stakes

    manifest: Dict[str, Any] = {}
    cutoff = window.report_cutoff_utc
    stale_map = {
        "executor_live": (6 * 3600, 3600),
        "accounting_health": (6 * 3600, 2 * 3600),
        "accounting_balance": (36 * 3600, 12 * 3600),
        "accounting_open_stakes": (36 * 3600, 12 * 3600),
        "accounting_daily_report": (36 * 3600, 12 * 3600),
        "e2e_trace": (6 * 3600, 3600),
        "clv_health": (6 * 3600, 3600),
        "clv_obligations": (6 * 3600, 3600),
        "policy_current": (7 * 86400, 86400),
        "risk_params": (7 * 86400, 86400),
    }
    for name, path in paths.items():
        meta = _file_meta(path)
        age = _age_seconds(meta.get("mtime_utc"), cutoff)
        stale_after, watch_after = stale_map.get(name, (24 * 3600, 6 * 3600))
        status = classify_freshness(age, stale_after_s=stale_after, watch_after_s=watch_after)
        if not meta["exists"]:
            status = "NOT_AVAILABLE"
        # Prefer explicit health JSON status when present
        if name.endswith("health") and meta["exists"]:
            try:
                hj = json.loads(Path(path).read_text(encoding="utf-8"))
                hs = str(hj.get("status") or "").upper()
                if hs in {"HEALTHY", "WATCH", "STALE", "PARTIAL", "FAILED"}:
                    status = hs
                meta["health_json_status"] = hs or None
            except Exception:
                pass
        meta["age_seconds"] = age
        meta["status"] = status
        manifest[name] = meta
    return manifest
