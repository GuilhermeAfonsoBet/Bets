from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ops.accounting_io import atomic_write_bytes, atomic_write_json
from ops.accounting_status import (
    ACCOUNTING_AUTH_FAILED,
    ACCOUNTING_EMPTY_RESPONSE,
    ACCOUNTING_OK,
    ACCOUNTING_PARTIAL,
    ACCOUNTING_SCHEMA_CHANGED,
    ACCOUNTING_TIMEOUT,
    CRITICAL,
    HEALTHY,
    WATCH,
    FreshnessLimits,
    classify_exception,
    classify_health,
    cycle_status,
    normalize_jsonl_path,
    order_id_key,
    validate_csv_schema,
)
from ops.h3bup_accounting_reconcile import classify_row, reconcile


def test_files_none_not_ok():
    st = cycle_status(balance_ok=False, open_ok=False, error_type=None)
    assert st != ACCOUNTING_OK
    assert st == ACCOUNTING_EMPTY_RESPONSE


def test_partial_balance_only():
    assert cycle_status(balance_ok=True, open_ok=False) == ACCOUNTING_PARTIAL


def test_auth_failure_classified():
    assert classify_exception("LOGIN_FAILED auth required") == ACCOUNTING_AUTH_FAILED
    assert cycle_status(balance_ok=False, open_ok=False, error_type=ACCOUNTING_AUTH_FAILED) == ACCOUNTING_AUTH_FAILED


def test_timeout_classified():
    assert classify_exception("Timeout 20000ms exceeded while waiting") == ACCOUNTING_TIMEOUT


def test_schema_changed():
    ok, err = validate_csv_schema(["amount", "balance"])
    assert not ok
    assert "missing" in (err or "")
    assert cycle_status(balance_ok=False, open_ok=False, error_type=ACCOUNTING_SCHEMA_CHANGED) == ACCOUNTING_SCHEMA_CHANGED


def test_atomic_write(tmp_path: Path):
    p = tmp_path / "x.csv"
    atomic_write_bytes(p, b"a,b\n1,2\n")
    assert p.read_bytes() == b"a,b\n1,2\n"
    assert not list(tmp_path.glob("*.partial"))


def test_last_valid_preserved_on_failure(tmp_path: Path):
    p = tmp_path / "20260728_220133__balance.csv"
    p.write_text("order id,amount,type,post date\n1,1,bet,2026-07-28\n", encoding="utf-8")
    before = p.read_text(encoding="utf-8")
    atomic_write_json(tmp_path / "accounting_health.json", {"status": ACCOUNTING_EMPTY_RESPONSE})
    assert p.read_text(encoding="utf-8") == before


def test_order_id_large_string():
    assert order_id_key("1933822208") == "1933822208"
    assert order_id_key(1933822208) == "1933822208"
    assert order_id_key("1933822208.0") == "1933822208"


def test_join_and_duplicate(tmp_path: Path):
    bal = tmp_path / "b.csv"
    bal.write_text(
        "order id,amount,type,post date,got price,note,status\n"
        "1931674091,-9.97,settlement,2026-07-28 17:57:35,1.86,Settlement of Bet,settled\n"
        "1932353274,0.00,settlement,2026-07-28 21:06:11,1.87,Settlement of Bet,settled\n",
        encoding="utf-8",
    )
    open_csv = tmp_path / "o.csv"
    open_csv.write_text("order id,amount,type,post date\n999,1,bet,2026-07-29\n", encoding="utf-8")
    live = [
        {"order_id": "1931674091", "stake": 10, "live_ok_ts": "2026-07-28T15:30:30+00:00", "kickoff_ts": "2026-07-28T16:00:00+00:00"},
        {"order_id": "1931674091", "stake": 10, "live_ok_ts": "2026-07-28T15:31:00+00:00", "kickoff_ts": "2026-07-28T16:00:00+00:00"},
        {"order_id": "1932353274", "stake": 10, "live_ok_ts": "2026-07-28T17:58:48+00:00", "kickoff_ts": "2026-07-28T19:00:00+00:00"},
        {"order_id": "1939999999", "stake": 10, "live_ok_ts": "2026-07-29T12:00:00+00:00", "kickoff_ts": "2026-07-30T18:00:00+00:00"},
    ]
    rows, summary = reconcile(
        live_rows=live,
        balance_path=bal,
        open_path=open_csv,
        snapshot_ts=datetime(2026, 7, 29, 13, 0, tzinfo=timezone.utc),
        now=datetime(2026, 7, 29, 13, 0, tzinfo=timezone.utc),
    )
    assert summary["n_live_ok"] == 3
    by = {r["order_id"]: r for r in rows}
    assert by["1931674091"]["reconciliation_status"] == "SETTLED_ACCOUNTING_OK"
    assert by["1932353274"]["reconciliation_status"] == "VOID_OR_PUSH"
    assert by["1939999999"]["reconciliation_status"] == "EVENT_NOT_STARTED"


def test_void_push_and_partial():
    now = datetime(2026, 7, 29, 12, tzinfo=timezone.utc)
    st, _ = classify_row(
        order_id="1",
        kickoff_ts="2026-07-28T10:00:00+00:00",
        now=now,
        balance={"amount_sum": 0.0, "notes": ["void"], "types": ["settlement"], "n_rows": 1, "post_dates": []},
        in_open=False,
        snapshot_ts=now,
        prev_snapshot_ts=None,
    )
    assert st == "VOID_OR_PUSH"
    st2, _ = classify_row(
        order_id="2",
        kickoff_ts="2026-07-28T10:00:00+00:00",
        now=now,
        balance={"amount_sum": -5.0, "notes": ["partial fill"], "types": ["settlement"], "n_rows": 2, "post_dates": []},
        in_open=False,
        snapshot_ts=now,
        prev_snapshot_ts=None,
    )
    assert st2 == "PARTIAL_SETTLEMENT"


def test_event_not_started():
    now = datetime(2026, 7, 29, 12, tzinfo=timezone.utc)
    st, _ = classify_row(
        order_id="3",
        kickoff_ts="2026-07-30T18:00:00+00:00",
        now=now,
        balance=None,
        in_open=False,
        snapshot_ts=now,
        prev_snapshot_ts=None,
    )
    assert st == "EVENT_NOT_STARTED"


def test_health_watch_and_critical():
    lim = FreshnessLimits(warn_stale_sec=100, critical_stale_sec=500, max_consecutive_failures=3)
    assert classify_health(status=ACCOUNTING_OK, balance_age_sec=10, open_age_sec=10, consecutive_failures=0, limits=lim) == HEALTHY
    assert classify_health(status=ACCOUNTING_OK, balance_age_sec=150, open_age_sec=10, consecutive_failures=0, limits=lim) == WATCH
    assert classify_health(status=ACCOUNTING_OK, balance_age_sec=600, open_age_sec=10, consecutive_failures=0, limits=lim) == CRITICAL
    assert classify_health(status=ACCOUNTING_EMPTY_RESPONSE, balance_age_sec=600, open_age_sec=600, consecutive_failures=3, limits=lim) == CRITICAL
    assert classify_health(status=ACCOUNTING_PARTIAL, balance_age_sec=10, open_age_sec=10, consecutive_failures=0, limits=lim) == WATCH


def test_jsonl_typo_normalized():
    assert "accounring" not in normalize_jsonl_path("logs/accounring_snapshots.jsonl")
    assert normalize_jsonl_path("logs/accounring_snapshots.jsonl").endswith("accounting_snapshots.jsonl")


def test_no_policy_stake_mutation_constants():
    text = (ROOT / "ops" / "accounting_monitor.py").read_text(encoding="utf-8")
    assert "place_order" not in text
    assert "EXECUTOR_LIVE_STAKE" not in text
    assert "stake_usd" not in text


def test_no_betslip_order_apis_in_accounting_modules():
    for name in ("accounting_monitor.py", "accounting_status.py", "h3bup_accounting_reconcile.py", "accounting_io.py"):
        t = (ROOT / "ops" / name).read_text(encoding="utf-8")
        assert "open_betslip" not in t.lower()
        assert "place_order" not in t
        assert "cancel_order" not in t
