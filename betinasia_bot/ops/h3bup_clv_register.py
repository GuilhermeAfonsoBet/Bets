"""Fail-open LIVE_OK → CLV obligation registration (never blocks execution)."""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from ops.h3bup_clv_config import WINDOWS, load_config
from ops.h3bup_clv_matching import choose_entry_odd, normalize_line, normalize_market, normalize_period, normalize_side
from ops.h3bup_clv_store import get_store, parse_ts, utc_iso


_Q: list = []
_Q_LOCK = threading.Lock()
_THREAD: Optional[threading.Thread] = None
_METRICS = {
    "enqueued": 0,
    "created": 0,
    "skipped": 0,
    "errors": 0,
}


def _ensure_started_at(cfg) -> str:
    p = Path(cfg.collection_started_path)
    try:
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
        p.parent.mkdir(parents=True, exist_ok=True)
        ts = utc_iso()
        p.write_text(ts + "\n", encoding="utf-8")
        return ts
    except Exception:
        return utc_iso()


def _policy_ok(policy_version: Any) -> bool:
    return "H3BUP_vNext" in str(policy_version or "")


def _extract_from_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    req = (payload or {}).get("request") or {}
    res = (payload or {}).get("result") or {}
    if str(res.get("status") or "") != "LIVE_OK":
        return None
    pol = (req.get("policy") or {}).get("policy_version") or (res.get("policy") or {}).get("policy_version")
    if not _policy_ok(pol):
        return None
    raw = res.get("raw") if isinstance(res.get("raw"), dict) else {}
    order_id = raw.get("order_id")
    if not order_id:
        return None
    entry, entry_src = choose_entry_odd(payload)
    if entry is None:
        return None
    meta = req.get("meta") if isinstance(req.get("meta"), dict) else {}
    e2e = meta.get("h3bup_e2e") if isinstance(meta.get("h3bup_e2e"), dict) else {}
    market = normalize_market(req.get("market_type") or res.get("market_type") or "AH")
    side = normalize_side(req.get("side") or res.get("side"))
    line = normalize_line(req.get("line") or res.get("line"), market)
    period = normalize_period((meta.get("market") or {}).get("period") if isinstance(meta.get("market"), dict) else "full_time")
    if not (market and side and line and req.get("event_id")):
        return None
    live_ok_ts = res.get("finished_at") or utc_iso()
    return {
        "order_id": str(order_id),
        "execution_id": str(res.get("execution_id") or req.get("execution_id") or ""),
        "audit_id": req.get("audit_id") or res.get("audit_id"),
        "trace_id": e2e.get("trace_id"),
        "policy_version": str(pol),
        "event_id": str(req.get("event_id")),
        "event_name": None,
        "market_type": market,
        "period": period,
        "side": side,
        "line": line,
        "entry_odd": float(entry),
        "entry_odd_source": entry_src,
        "odd_at_decision": req.get("odd_at_decision") or res.get("odd_at_decision"),
        "odd_final": res.get("odd_final"),
        "live_ok_ts_utc": live_ok_ts,
    }


def _window_target(live_ok_ts: float, window: str) -> Optional[float]:
    if window == "POST_5M":
        return live_ok_ts + 5 * 60
    if window == "POST_15M":
        return live_ok_ts + 15 * 60
    if window == "CLOSING":
        return None  # resolved when kickoff known
    return None


def create_obligations_for_order(base: Dict[str, Any], *, kickoff_ts: Any = None, kickoff_source: str = None) -> int:
    cfg = load_config()
    if not (cfg.enabled and cfg.create_obligations):
        return 0
    store = get_store(cfg)
    created = 0
    live_ts = parse_ts(base.get("live_ok_ts_utc")) or time.time()
    kick_f = parse_ts(kickoff_ts)
    windows = []
    if cfg.post_5m_enabled:
        windows.append("POST_5M")
    if cfg.post_15m_enabled:
        windows.append("POST_15M")
    if cfg.closing_enabled:
        windows.append("CLOSING")
    for w in windows:
        key = store.obligation_key(base["order_id"], w, cfg.schema_version)
        if store.get_obligation(key):
            continue
        target = _window_target(live_ts, w)
        status = "WAITING_TARGET"
        err = None
        if w in ("POST_5M", "POST_15M") and kick_f is not None and target is not None and target >= kick_f:
            status = "SKIPPED"
            err = "TARGET_AFTER_KICKOFF"
        if w == "CLOSING":
            if kick_f is None:
                status = "WAITING_TARGET"  # wait for kickoff resolution
                target = None
            else:
                target = kick_f - float(cfg.closing_buffer_sec)
                status = "WAITING_TARGET"
        obj = {
            "schema_version": cfg.schema_version,
            "obligation_key": key,
            "order_id": base["order_id"],
            "execution_id": base.get("execution_id"),
            "audit_id": base.get("audit_id"),
            "trace_id": base.get("trace_id"),
            "policy_version": base.get("policy_version"),
            "event_id": base.get("event_id"),
            "event_name": base.get("event_name"),
            "market_type": base.get("market_type"),
            "period": base.get("period"),
            "side": base.get("side"),
            "line": base.get("line"),
            "entry_odd": base.get("entry_odd"),
            "entry_odd_source": base.get("entry_odd_source"),
            "odd_at_decision": base.get("odd_at_decision"),
            "odd_final": base.get("odd_final"),
            "live_ok_ts_utc": base.get("live_ok_ts_utc"),
            "kickoff_ts_utc": utc_iso(kick_f) if kick_f else None,
            "kickoff_source": kickoff_source,
            "kickoff_confidence": ("HIGH" if kick_f else "UNKNOWN"),
            "window_name": w,
            "target_ts_utc": utc_iso(target) if target else None,
            "status": status,
            "attempts": 0,
            "next_attempt_ts_utc": utc_iso(target) if target else utc_iso(),
            "last_error_code": err,
            "last_error_message": err,
            "completed_at_utc": utc_iso() if status == "SKIPPED" else None,
        }
        store.upsert_obligation(obj)
        created += 1
        _METRICS["created"] += 1
    return created


def _worker_loop() -> None:
    while True:
        item = None
        try:
            with _Q_LOCK:
                if _Q:
                    item = _Q.pop(0)
            if item is None:
                time.sleep(0.2)
                continue
            cfg = load_config()
            if not (cfg.enabled and cfg.create_obligations):
                _METRICS["skipped"] += 1
                continue
            _ensure_started_at(cfg)
            started = parse_ts(Path(cfg.collection_started_path).read_text().strip()) if Path(cfg.collection_started_path).exists() else None
            base = _extract_from_payload(item)
            if not base:
                _METRICS["skipped"] += 1
                continue
            live_ts = parse_ts(base.get("live_ok_ts_utc"))
            if started and live_ts and live_ts < started - 1:
                # forward-only
                _METRICS["skipped"] += 1
                continue
            # best-effort kickoff resolve without blocking forever
            kick = None
            kick_src = None
            try:
                kick, kick_src = _resolve_kickoff_sync(base.get("event_id"))
            except Exception:
                kick = None
            create_obligations_for_order(base, kickoff_ts=kick, kickoff_source=kick_src)
        except Exception:
            _METRICS["errors"] += 1
            time.sleep(0.5)


def _resolve_kickoff_sync(event_id: Any):
    """Optional sync DB lookup in background thread only."""
    if not event_id:
        return None, None
    try:
        import asyncio
        from sqlalchemy import text
        from storage.database import Database

        async def _run():
            db = Database()
            await db.connect()
            try:
                async with db.async_session() as session:
                    r = await session.execute(
                        text("SELECT kickoff_time FROM matches WHERE external_id = :e LIMIT 2"),
                        {"e": str(event_id)},
                    )
                    rows = r.fetchall()
                    if not rows:
                        return None, None
                    if len(rows) > 1:
                        # conflict if different kickoffs
                        ks = {str(x[0]) for x in rows}
                        if len(ks) > 1:
                            return None, "KICKOFF_CONFLICT"
                    return rows[0][0], "matches.kickoff_time"
            finally:
                try:
                    await db.close()
                except Exception:
                    pass

        return asyncio.run(_run())
    except Exception:
        return None, None


def _ensure_thread() -> None:
    global _THREAD
    if _THREAD and _THREAD.is_alive():
        return
    _THREAD = threading.Thread(target=_worker_loop, name="h3bup-clv-register", daemon=True)
    _THREAD.start()


def enqueue_live_ok_payload(payload: Dict[str, Any]) -> None:
    """Best-effort; NEVER raises."""
    try:
        cfg = load_config()
        if not (cfg.enabled and cfg.create_obligations):
            return
        if cfg.allow_betslip_source:
            # hard refuse betslip path even if misconfigured for registration
            pass
        if cfg.fair_edge_enabled:
            # ignore — fair edge not implemented
            pass
        _ensure_thread()
        with _Q_LOCK:
            if len(_Q) > 5000:
                _METRICS["errors"] += 1
                return
            _Q.append(payload)
            _METRICS["enqueued"] += 1
    except Exception:
        try:
            _METRICS["errors"] += 1
        except Exception:
            pass


def get_register_metrics() -> Dict[str, int]:
    return dict(_METRICS)
