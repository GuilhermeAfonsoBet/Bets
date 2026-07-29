from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ops.h3bup_clv_matching import (
    choose_entry_odd,
    clv_raw,
    evaluate_match,
    normalize_line,
    normalize_side,
)
from ops.h3bup_clv_register import create_obligations_for_order, _extract_from_payload
from ops.h3bup_clv_store import ClvJsonlStore
from ops.h3bup_clv_config import ClvConfig


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("H3BUP_CLV_ENABLED", "1")
    monkeypatch.setenv("H3BUP_CLV_CREATE_OBLIGATIONS", "1")
    monkeypatch.setenv("H3BUP_CLV_WORKER_ENABLED", "0")
    monkeypatch.setenv("H3BUP_CLV_ALLOW_BETSLIP_SOURCE", "0")
    monkeypatch.setenv("H3BUP_CLV_FAIR_EDGE_ENABLED", "0")
    c = ClvConfig(
        enabled=True,
        create_obligations=True,
        worker_enabled=False,
        allow_betslip_source=False,
        fair_edge_enabled=False,
        obligations_jsonl=str(tmp_path / "obl.jsonl"),
        snapshots_jsonl=str(tmp_path / "snap.jsonl"),
        collection_started_path=str(tmp_path / "started.txt"),
        health_path=str(tmp_path / "health.json"),
    )
    (tmp_path / "started.txt").write_text("2026-07-29T15:00:00+00:00\n", encoding="utf-8")
    return c


def _live_payload(**over):
    base = {
        "request": {
            "execution_id": "e1",
            "audit_id": 1,
            "event_id": "2026-07-29,1,2",
            "market_type": "AH",
            "side": "home",
            "line": "-0.5",
            "odd_at_decision": 2.0,
            "policy": {"policy_version": "H3BUP_vNext_20260629", "stake_requested": 10},
            "meta": {"h3bup_e2e": {"trace_id": "h3bup:1:aaaaaaaaaaaa"}, "market": {"period": "full_time"}},
        },
        "result": {
            "execution_id": "e1",
            "status": "LIVE_OK",
            "odd_final": 1.98,
            "finished_at": "2026-07-29T16:00:00+00:00",
            "raw": {"order_id": "1937000001", "sent": {"price": 1.97, "stake": 10}},
        },
    }
    base.update(over)
    return base


def test_entry_odd_prefers_sent_price():
    odd, src = choose_entry_odd(_live_payload())
    assert odd == 1.97 and src == "sent.price"


def test_entry_odd_fallback_odd_final():
    p = _live_payload()
    p["result"]["raw"]["sent"] = {}
    odd, src = choose_entry_odd(p)
    assert odd == 1.98 and src == "odd_final"


def test_clv_positive_and_negative_b808():
    dec, pct = clv_raw(2.0, 1.9)
    assert dec > 0 and pct > 0
    dec2, pct2 = clv_raw(1.9, 2.0)
    assert dec2 < 0 and pct2 < 0
    # B808 equivalence: (entry-closing)/closing
    assert abs(pct - ((2.0 - 1.9) / 1.9 * 100)) < 1e-9


def test_same_line_strict_and_rejects_different():
    ok = evaluate_match(
        want_event="e", got_event="e", want_market="AH", got_market="AH",
        want_period="full_time", got_period="FT", want_side="home", got_side="HOME",
        want_line="-0.5", got_line="-0.50", snapshot_ts=100.0, kickoff_ts=200.0,
    )
    assert ok.is_strict
    bad = evaluate_match(
        want_event="e", got_event="e", want_market="AH", got_market="AH",
        want_period="full_time", got_period="full_time", want_side="home", got_side="home",
        want_line="-0.5", got_line="-0.75", snapshot_ts=100.0, kickoff_ts=200.0,
    )
    assert not bad.same_line_strict


def test_side_market_period_mismatch():
    m = evaluate_match(
        want_event="e", got_event="e", want_market="AH", got_market="OU",
        want_period="full_time", got_period="first_half", want_side="home", got_side="away",
        want_line="0.5", got_line="0.5", snapshot_ts=100.0, kickoff_ts=200.0,
    )
    assert not m.same_market and not m.same_period and not m.same_side


def test_snapshot_after_kickoff_invalid():
    m = evaluate_match(
        want_event="e", got_event="e", want_market="AH", got_market="AH",
        want_period="ft", got_period="ft", want_side="home", got_side="home",
        want_line="0", got_line="0.0", snapshot_ts=250.0, kickoff_ts=200.0,
    )
    assert not m.snapshot_before_kickoff


def test_only_live_ok_h3bup_creates(cfg, monkeypatch):
    store = ClvJsonlStore(cfg)
    monkeypatch.setattr("ops.h3bup_clv_register.get_store", lambda _c=None: store)
    monkeypatch.setattr("ops.h3bup_clv_register.load_config", lambda: cfg)
    base = _extract_from_payload(_live_payload())
    assert base is not None
    n = create_obligations_for_order(base, kickoff_ts="2026-07-29T18:00:00+00:00", kickoff_source="test")
    assert n == 3
    assert len(store.list_obligations()) == 3
    # idempotent
    n2 = create_obligations_for_order(base, kickoff_ts="2026-07-29T18:00:00+00:00")
    assert n2 == 0
    assert len(store.list_obligations()) == 3


def test_cap_blocked_no_obligation():
    p = _live_payload()
    p["result"]["status"] = "CAP_BLOCKED"
    assert _extract_from_payload(p) is None


def test_api_failed_no_obligation():
    p = _live_payload()
    p["result"]["status"] = "API_FAILED"
    assert _extract_from_payload(p) is None


def test_other_policy_no_obligation():
    p = _live_payload()
    p["request"]["policy"]["policy_version"] = "bridge_h3b_live_v0"
    assert _extract_from_payload(p) is None


def test_post15_skipped_if_after_kickoff(cfg, monkeypatch):
    store = ClvJsonlStore(cfg)
    monkeypatch.setattr("ops.h3bup_clv_register.get_store", lambda _c=None: store)
    monkeypatch.setattr("ops.h3bup_clv_register.load_config", lambda: cfg)
    base = _extract_from_payload(_live_payload())
    # live at 16:00, kickoff 16:10 → POST_15M target 16:15 after kickoff
    create_obligations_for_order(base, kickoff_ts="2026-07-29T16:10:00+00:00")
    by_w = {o["window_name"]: o for o in store.list_obligations()}
    assert by_w["POST_15M"]["status"] == "SKIPPED"
    assert by_w["POST_15M"]["last_error_code"] == "TARGET_AFTER_KICKOFF"


def test_line_normalize_variants():
    assert normalize_line("0.50", "AH") == normalize_line(0.5, "AH")
    assert normalize_side("HOME") == "home"


def test_flag_off_no_create(tmp_path, monkeypatch):
    monkeypatch.setenv("H3BUP_CLV_ENABLED", "0")
    from ops.h3bup_clv_config import load_config

    c = load_config()
    c.obligations_jsonl = str(tmp_path / "o.jsonl")
    store = ClvJsonlStore(c)
    monkeypatch.setattr("ops.h3bup_clv_register.get_store", lambda _c=None: store)
    monkeypatch.setattr("ops.h3bup_clv_register.load_config", lambda: c)
    base = _extract_from_payload(_live_payload())
    assert create_obligations_for_order(base) == 0


def test_store_enqueue_does_not_raise(monkeypatch):
    monkeypatch.setenv("H3BUP_CLV_ENABLED", "0")
    from ops.h3bup_clv_register import enqueue_live_ok_payload

    enqueue_live_ok_payload(_live_payload())  # must not raise when disabled
    monkeypatch.setenv("H3BUP_CLV_ENABLED", "1")
    monkeypatch.setenv("H3BUP_CLV_CREATE_OBLIGATIONS", "1")
    # enabled path still fail-open even if store broken
    enqueue_live_ok_payload(_live_payload())


def test_executor_store_contains_clv_hook():
    src = (ROOT / "executor" / "store.py").read_text(encoding="utf-8")
    assert "enqueue_live_ok_payload" in src
    assert "Fase 2C" in src
    assert "LIVE_OK" in src


def test_no_fair_edge_and_no_betslip_defaults():
    from ops.h3bup_clv_config import load_config
    import os

    os.environ.pop("H3BUP_CLV_ALLOW_BETSLIP_SOURCE", None)
    os.environ.pop("H3BUP_CLV_FAIR_EDGE_ENABLED", None)
    c = load_config()
    assert c.allow_betslip_source is False
    assert c.fair_edge_enabled is False


def test_policy_stake_untouched_in_bridge_worker_sources():
    bridge = (ROOT / "ops" / "executor_bridge_audit.py").read_text(encoding="utf-8")
    worker = (ROOT / "executor" / "worker.py").read_text(encoding="utf-8")
    assert 'POLICY_VERSION_H3BUP_VNEXT = "H3BUP_vNext_20260629"' in bridge
    assert "req.policy.stake_requested = 10.0" in bridge
    assert "1.85 <= float(odd_val) <= 2.15" in worker
