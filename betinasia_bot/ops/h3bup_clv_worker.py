#!/usr/bin/env python3
"""H3BUP CLV async worker — analytics-only, no betslip, no executor calls."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, ".")

from ops.h3bup_clv_config import load_config
from ops.h3bup_clv_matching import clv_raw, evaluate_match, normalize_period
from ops.h3bup_clv_register import create_obligations_for_order, _extract_from_payload, _ensure_started_at
from ops.h3bup_clv_sources import find_boh_closing, find_boh_nearest, find_passive_nearest, resolve_match
from ops.h3bup_clv_store import ensure_postgres_schema, get_store, parse_ts, utc_iso


def write_health(cfg, extra: Dict[str, Any]) -> None:
    store = get_store(cfg)
    obls = store.list_obligations()
    snaps = store.snapshots()
    by_w = {"POST_5M": [], "POST_15M": [], "CLOSING": []}
    for o in obls:
        by_w.setdefault(o.get("window_name"), []).append(o)
    strict = [s for s in snaps if s.get("quality_status") == "VALID_STRICT"]

    def cov(window: str):
        items = by_w.get(window) or []
        expected = len(items)
        attempted = sum(1 for o in items if int(o.get("attempts") or 0) > 0 or o.get("status") in ("COMPLETED", "FAILED_FINAL", "RETRYABLE", "PROCESSING"))
        valid = sum(1 for s in strict if s.get("window_name") == window)
        return expected, attempted, valid

    p5e, p5a, p5v = cov("POST_5M")
    p15e, p15a, p15v = cov("POST_15M")
    ce, ca, cv = cov("CLOSING")
    live_ok = len({o.get("order_id") for o in obls})
    started = None
    try:
        started = Path(cfg.collection_started_path).read_text().strip()
    except Exception:
        started = None
    fail_codes = {}
    for o in obls:
        c = o.get("last_error_code")
        if c:
            fail_codes[c] = fail_codes.get(c, 0) + 1
    for s in snaps:
        c = s.get("failure_reason")
        if c:
            fail_codes[c] = fail_codes.get(c, 0) + 1
    retry_backlog = sum(1 for o in obls if o.get("status") in ("RETRYABLE", "WAITING_TARGET", "READY", "PENDING"))
    failed_final = sum(1 for o in obls if o.get("status") == "FAILED_FINAL")
    status = "HEALTHY"
    if extra.get("worker_consecutive_failures", 0) >= 5:
        status = "CRITICAL"
    elif live_ok < 30 or (p5e and p5v == 0 and p5a > 0):
        status = "WATCH"
    if not cfg.enabled or not cfg.worker_enabled:
        status = "WATCH"
    payload = {
        "checked_at_utc": utc_iso(),
        "status": status,
        "enabled": bool(cfg.enabled),
        "collection_started_at_utc": started,
        "source_priority": list(cfg.source_priority),
        "live_ok_after_activation": live_ok,
        "obligations_expected": live_ok * sum([cfg.post_5m_enabled, cfg.post_15m_enabled, cfg.closing_enabled]),
        "obligations_created": len(obls),
        "post_5m_expected": p5e,
        "post_5m_attempted": p5a,
        "post_5m_valid_strict": p5v,
        "post_15m_expected": p15e,
        "post_15m_attempted": p15a,
        "post_15m_valid_strict": p15v,
        "closing_expected": ce,
        "closing_attempted": ca,
        "closing_valid_strict": cv,
        "source_missing": fail_codes.get("SOURCE_MISSING", 0),
        "line_mismatch": fail_codes.get("LINE_NOT_FOUND", 0) + fail_codes.get("LINE_CHANGED", 0),
        "side_mismatch": fail_codes.get("SIDE_NOT_FOUND", 0),
        "period_mismatch": fail_codes.get("PERIOD_NOT_FOUND", 0),
        "kickoff_missing": fail_codes.get("KICKOFF_MISSING", 0),
        "kickoff_conflict": fail_codes.get("KICKOFF_CONFLICT", 0),
        "snapshot_after_kickoff": fail_codes.get("SNAPSHOT_AFTER_KICKOFF", 0),
        "snapshot_too_far": fail_codes.get("SNAPSHOT_TOO_FAR", 0),
        "retry_backlog": retry_backlog,
        "failed_final": failed_final,
        "worker_last_success_utc": extra.get("worker_last_success_utc"),
        "worker_consecutive_failures": int(extra.get("worker_consecutive_failures") or 0),
        "collector_status": extra.get("collector_status", "DISABLED" if not cfg.passive_collector_enabled else "UNKNOWN"),
        "error": extra.get("error"),
        "betslip_source_allowed": bool(cfg.allow_betslip_source),
        "fair_edge_enabled": bool(cfg.fair_edge_enabled),
    }
    path = Path(cfg.health_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


async def process_one(session, cfg, obl: Dict[str, Any]) -> Dict[str, Any]:
    store = get_store(cfg)
    now = time.time()
    window = obl.get("window_name")
    live_ok_ts = parse_ts(obl.get("live_ok_ts_utc")) or 0.0
    kick = parse_ts(obl.get("kickoff_ts_utc"))
    err = None

    # refresh kickoff if missing
    if kick is None and obl.get("event_id"):
        mid, kick_dt, ecode = await resolve_match(session, str(obl.get("event_id")))
        if ecode == "KICKOFF_CONFLICT":
            obl["status"] = "FAILED_FINAL"
            obl["last_error_code"] = "KICKOFF_CONFLICT"
            obl["last_error_message"] = "KICKOFF_CONFLICT"
            obl["completed_at_utc"] = utc_iso()
            return store.upsert_obligation(obl)
        if kick_dt is not None:
            kick = kick_dt.timestamp() if hasattr(kick_dt, "timestamp") else parse_ts(kick_dt)
            obl["kickoff_ts_utc"] = utc_iso(kick)
            obl["kickoff_source"] = "matches.kickoff_time"
            obl["kickoff_confidence"] = "HIGH"
            if window == "CLOSING":
                obl["target_ts_utc"] = utc_iso(kick - float(cfg.closing_buffer_sec))
            if window in ("POST_5M", "POST_15M"):
                target = parse_ts(obl.get("target_ts_utc"))
                if target and kick and target >= kick:
                    obl["status"] = "SKIPPED"
                    obl["last_error_code"] = "TARGET_AFTER_KICKOFF"
                    obl["last_error_message"] = "TARGET_AFTER_KICKOFF"
                    obl["completed_at_utc"] = utc_iso()
                    return store.upsert_obligation(obl)
        elif ecode:
            # keep waiting for kickoff on CLOSING; for timed windows may still proceed without kickoff if target not past
            if window == "CLOSING":
                obl["status"] = "RETRYABLE"
                obl["last_error_code"] = ecode or "KICKOFF_MISSING"
                obl["last_error_message"] = ecode or "KICKOFF_MISSING"
                obl["attempts"] = int(obl.get("attempts") or 0) + 1
                obl["next_attempt_ts_utc"] = utc_iso(now + cfg.retry_base_sec * max(1, obl["attempts"]))
                if obl["attempts"] >= cfg.max_attempts:
                    obl["status"] = "FAILED_FINAL"
                    obl["completed_at_utc"] = utc_iso()
                return store.upsert_obligation(obl)

    target = parse_ts(obl.get("target_ts_utc"))
    if window == "CLOSING":
        if kick is None:
            obl["status"] = "RETRYABLE"
            obl["last_error_code"] = "KICKOFF_MISSING"
            obl["attempts"] = int(obl.get("attempts") or 0) + 1
            obl["next_attempt_ts_utc"] = utc_iso(now + cfg.retry_base_sec)
            if obl["attempts"] >= cfg.max_attempts:
                obl["status"] = "FAILED_FINAL"
                obl["completed_at_utc"] = utc_iso()
            return store.upsert_obligation(obl)
        target = kick - float(cfg.closing_buffer_sec)
        obl["target_ts_utc"] = utc_iso(target)
        # only attempt closing after cutoff reached (enough time for last pre-kickoff scrape)
        if now < target:
            obl["status"] = "WAITING_TARGET"
            obl["next_attempt_ts_utc"] = utc_iso(target)
            return store.upsert_obligation(obl)
    else:
        if target is None:
            obl["status"] = "FAILED_FINAL"
            obl["last_error_code"] = "INTERNAL_ERROR"
            obl["completed_at_utc"] = utc_iso()
            return store.upsert_obligation(obl)
        if now < target:
            obl["status"] = "WAITING_TARGET"
            obl["next_attempt_ts_utc"] = utc_iso(target)
            return store.upsert_obligation(obl)

    obl["status"] = "PROCESSING"
    obl["attempts"] = int(obl.get("attempts") or 0) + 1
    store.upsert_obligation(obl)

    hit = None
    err = None
    mid = None
    if obl.get("event_id"):
        mid, kick_dt, ecode = await resolve_match(session, str(obl.get("event_id")))
        if ecode == "EVENT_NOT_FOUND":
            err = "EVENT_NOT_FOUND"
        elif ecode == "KICKOFF_CONFLICT":
            err = "KICKOFF_CONFLICT"
        if kick is None and kick_dt is not None:
            kick = kick_dt.timestamp() if hasattr(kick_dt, "timestamp") else parse_ts(kick_dt)

    for src in cfg.source_priority:
        if err in ("EVENT_NOT_FOUND", "KICKOFF_CONFLICT"):
            break
        if src == "best_odds_history" and mid is not None:
            if window == "CLOSING":
                hit, err = await find_boh_closing(
                    session,
                    match_id=mid,
                    market_type=str(obl.get("market_type")),
                    side=str(obl.get("side")),
                    line=str(obl.get("line")),
                    kickoff_ts=float(kick),
                    closing_buffer_sec=cfg.closing_buffer_sec,
                    closing_max_age_sec=cfg.closing_max_age_sec,
                    event_id=str(obl.get("event_id")),
                    period=str(obl.get("period") or "full_time"),
                )
            else:
                tol_b = cfg.post_5m_tol_before_sec if window == "POST_5M" else cfg.post_15m_tol_before_sec
                tol_a = cfg.post_5m_tol_after_sec if window == "POST_5M" else cfg.post_15m_tol_after_sec
                hit, err = await find_boh_nearest(
                    session,
                    match_id=mid,
                    market_type=str(obl.get("market_type")),
                    side=str(obl.get("side")),
                    line=str(obl.get("line")),
                    target_ts=float(target),
                    tol_before=tol_b,
                    tol_after=tol_a,
                    live_ok_ts=float(live_ok_ts),
                    kickoff_ts=kick,
                    period=str(obl.get("period") or "full_time"),
                    event_id=str(obl.get("event_id")),
                )
            if hit:
                break
        if src == "passive_collector":
            if window == "CLOSING":
                # passive closing: nearest before cutoff
                hit2, err2 = find_passive_nearest(
                    cfg.passive_jsonl,
                    order_id=str(obl.get("order_id")),
                    event_id=str(obl.get("event_id")),
                    side=str(obl.get("side")),
                    line=str(obl.get("line")),
                    market_type=str(obl.get("market_type")),
                    target_ts=float(target),
                    tol_before=float(cfg.closing_max_age_sec),
                    tol_after=0.0,
                    live_ok_ts=float(live_ok_ts),
                    kickoff_ts=kick,
                )
            else:
                tol_b = cfg.post_5m_tol_before_sec if window == "POST_5M" else cfg.post_15m_tol_before_sec
                tol_a = cfg.post_5m_tol_after_sec if window == "POST_5M" else cfg.post_15m_tol_after_sec
                hit2, err2 = find_passive_nearest(
                    cfg.passive_jsonl,
                    order_id=str(obl.get("order_id")),
                    event_id=str(obl.get("event_id")),
                    side=str(obl.get("side")),
                    line=str(obl.get("line")),
                    market_type=str(obl.get("market_type")),
                    target_ts=float(target),
                    tol_before=tol_b,
                    tol_after=tol_a,
                    live_ok_ts=float(live_ok_ts),
                    kickoff_ts=kick,
                )
            if hit2:
                hit, err = hit2, None
                break
            if not hit:
                err = err2 or err

    # hard forbid betslip
    if cfg.allow_betslip_source:
        # still do not use it
        pass

    flags = None
    quality = "MISSING"
    clv_dec = None
    clv_pct = None
    failure = err or "SOURCE_MISSING"
    snap_odd = None
    snap_ts = None
    dist = None
    src_name = None
    src_id = None

    if hit:
        flags = evaluate_match(
            want_event=obl.get("event_id"),
            got_event=hit.event_id,
            want_market=obl.get("market_type"),
            got_market=hit.market_type,
            want_period=obl.get("period") or "full_time",
            got_period=hit.period,
            want_side=obl.get("side"),
            got_side=hit.side,
            want_line=obl.get("line"),
            got_line=hit.line,
            snapshot_ts=hit.snapshot_ts,
            kickoff_ts=kick if kick is not None else 1e18,
        )
        snap_odd = hit.odd
        snap_ts = hit.snapshot_ts
        dist = (hit.snapshot_ts - float(target)) if target is not None else None
        src_name = hit.source
        src_id = hit.source_record_id
        if flags.is_strict:
            try:
                clv_dec, clv_pct = clv_raw(float(obl.get("entry_odd")), float(hit.odd))
                quality = "VALID_STRICT"
                failure = None
            except Exception:
                quality = "INVALID_ODD"
                failure = "INVALID_ODD"
        else:
            quality = "VALID_NON_STRICT_DIAGNOSTIC"
            if not flags.same_line_strict:
                failure = "LINE_CHANGED"
            elif not flags.same_side:
                failure = "SIDE_NOT_FOUND"
            elif not flags.same_period:
                failure = "PERIOD_NOT_FOUND"
            elif not flags.same_market:
                failure = "MARKET_NOT_FOUND"
            elif not flags.same_event:
                failure = "EVENT_NOT_FOUND"
            elif not flags.snapshot_before_kickoff:
                failure = "SNAPSHOT_AFTER_KICKOFF"
            else:
                failure = "INVALID_TIME"

    snap = {
        "obligation_id": obl.get("id"),
        "obligation_key": obl.get("obligation_key"),
        "order_id": obl.get("order_id"),
        "execution_id": obl.get("execution_id"),
        "audit_id": obl.get("audit_id"),
        "trace_id": obl.get("trace_id"),
        "policy_version": obl.get("policy_version"),
        "window_name": window,
        "target_ts_utc": obl.get("target_ts_utc"),
        "snapshot_ts_utc": utc_iso(snap_ts) if snap_ts else None,
        "snapshot_distance_sec": dist,
        "kickoff_ts_utc": obl.get("kickoff_ts_utc"),
        "source": src_name,
        "source_record_id": src_id,
        "event_id": obl.get("event_id"),
        "market_type": obl.get("market_type"),
        "period": obl.get("period"),
        "side": obl.get("side"),
        "line": obl.get("line"),
        "entry_odd": obl.get("entry_odd"),
        "entry_odd_source": obl.get("entry_odd_source"),
        "snapshot_odd": snap_odd,
        "clv_raw_decimal": clv_dec,
        "clv_raw_pct": clv_pct,
        "same_event_flag": (flags.same_event if flags else False),
        "same_market_flag": (flags.same_market if flags else False),
        "same_period_flag": (flags.same_period if flags else False),
        "same_side_flag": (flags.same_side if flags else False),
        "same_line_flag": (flags.same_line if flags else False),
        "same_line_strict_flag": (flags.same_line_strict if flags else False),
        "snapshot_before_kickoff_flag": (flags.snapshot_before_kickoff if flags else False),
        "quality_status": quality,
        "failure_reason": failure,
    }
    store.append_snapshot(snap)

    if quality == "VALID_STRICT":
        obl["status"] = "COMPLETED"
        obl["last_error_code"] = None
        obl["last_error_message"] = None
        obl["completed_at_utc"] = utc_iso()
    else:
        obl["last_error_code"] = failure
        obl["last_error_message"] = failure
        if int(obl.get("attempts") or 0) >= cfg.max_attempts:
            obl["status"] = "FAILED_FINAL"
            obl["completed_at_utc"] = utc_iso()
        else:
            obl["status"] = "RETRYABLE"
            obl["next_attempt_ts_utc"] = utc_iso(now + cfg.retry_base_sec * max(1, int(obl.get("attempts") or 1)))
    return store.upsert_obligation(obl)


async def reconcile_live_ok(cfg) -> int:
    """Scan executor_live.jsonl for LIVE_OK H3BUP without obligations (forward-only)."""
    path = Path(os.getenv("EXECUTOR_JSONL", "logs/executor_live.jsonl"))
    if not path.exists():
        return 0
    store = get_store(cfg)
    started = None
    try:
        started = parse_ts(Path(cfg.collection_started_path).read_text().strip())
    except Exception:
        started = None
    n = 0
    # read tail only for performance
    try:
        data = path.read_bytes()
        if len(data) > 8_000_000:
            data = data[-8_000_000:]
        text = data.decode("utf-8", errors="ignore")
    except Exception:
        return 0
    for line in text.splitlines():
        line = line.strip()
        if not line or "LIVE_OK" not in line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        base = _extract_from_payload(payload)
        if not base:
            continue
        live_ts = parse_ts(base.get("live_ok_ts_utc"))
        if started and live_ts and live_ts < started - 1:
            continue
        if store.has_order(base["order_id"]):
            continue
        create_obligations_for_order(base)
        n += 1
    return n


async def run_worker_once(cfg) -> int:
    from storage.database import Database

    db = Database()
    await db.connect()
    processed = 0
    try:
        await ensure_postgres_schema(db.engine)
        store = get_store(cfg)
        now = time.time()
        cand = []
        for o in store.list_obligations():
            if o.get("status") in ("COMPLETED", "SKIPPED", "FAILED_FINAL", "CANCELLED"):
                continue
            nxt = parse_ts(o.get("next_attempt_ts_utc")) or 0.0
            if nxt <= now + 1:
                cand.append(o)
            elif o.get("status") == "WAITING_TARGET" and (parse_ts(o.get("target_ts_utc")) or 1e18) <= now:
                cand.append(o)
        cand = cand[: int(cfg.batch_size)]
        async with db.async_session() as session:
            for o in cand:
                await process_one(session, cfg, o)
                processed += 1
    finally:
        try:
            await db.close()
        except Exception:
            pass
    return processed


async def main_loop() -> int:
    cfg = load_config()
    if not cfg.enabled or not cfg.worker_enabled:
        print("CLV worker disabled by flags")
        write_health(cfg, {"error": "disabled", "worker_consecutive_failures": 0, "collector_status": "DISABLED"})
        return 0
    if cfg.allow_betslip_source:
        print("WARN: H3BUP_CLV_ALLOW_BETSLIP_SOURCE ignored (always off)")
    _ensure_started_at(cfg)
    failures = 0
    last_ok = None
    while True:
        cfg = load_config()
        if not cfg.enabled or not cfg.worker_enabled:
            write_health(cfg, {"error": "disabled", "worker_consecutive_failures": failures, "collector_status": "DISABLED"})
            await asyncio.sleep(cfg.poll_sec)
            continue
        try:
            await reconcile_live_ok(cfg)
            n = await run_worker_once(cfg)
            last_ok = utc_iso()
            failures = 0
            write_health(
                cfg,
                {
                    "worker_last_success_utc": last_ok,
                    "worker_consecutive_failures": 0,
                    "collector_status": ("ENABLED" if cfg.passive_collector_enabled else "DISABLED"),
                    "error": None,
                    "last_batch": n,
                },
            )
        except Exception as e:
            failures += 1
            write_health(
                cfg,
                {
                    "worker_last_success_utc": last_ok,
                    "worker_consecutive_failures": failures,
                    "collector_status": ("ENABLED" if cfg.passive_collector_enabled else "DISABLED"),
                    "error": str(e)[:240],
                },
            )
            traceback.print_exc()
        await asyncio.sleep(max(1.0, float(cfg.poll_sec)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if args.once:
        async def _once():
            _ensure_started_at(cfg)
            if cfg.enabled and cfg.worker_enabled:
                await reconcile_live_ok(cfg)
                n = await run_worker_once(cfg)
            else:
                n = 0
            write_health(cfg, {"worker_last_success_utc": utc_iso(), "worker_consecutive_failures": 0, "collector_status": "DISABLED", "last_batch": n})
            print(json.dumps({"processed": n}))

        asyncio.run(_once())
        return 0
    asyncio.run(main_loop())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
