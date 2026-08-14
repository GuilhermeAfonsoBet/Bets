"""Mandatory tests for H3BUP Friendly vs Non-Friendly analysis (read-only)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ops.h3bup_friendly_analysis.classification import (
    FRIENDLY_NAME_RULES,
    build_classification_mapping,
    classify_entity,
    mapping_checksum,
    normalize_text,
    write_freeze_artifacts,
)
from ops.h3bup_friendly_analysis.clv_join import attach_clv, load_clv_by_order
from ops.h3bup_friendly_analysis.run import run_analysis
from ops.h3bup_friendly_analysis.settlement import (
    attach_settlement,
    classify_settlement,
    performance_block,
    sample_gate,
)
from ops.h3bup_friendly_analysis.stats import clustered_bootstrap_diff, run_stat_tests
from ops.h3bup_friendly_analysis.universe import load_primary_h3bup_universe, load_secondary_historical_comparable


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _live_ok(
    *,
    order_id: str,
    created: str,
    league: str,
    event_id: str,
    policy_version: str = "H3BUP_vNext_20260629",
    stake: float = 10.0,
    exec_side: str = "back",
    status: str = "LIVE_OK",
    is_live: bool = False,
    competition_type=None,
    event_name: str = "Team A vs Team B",
):
    return {
        "request": {
            "created_at": created,
            "execution_id": f"ex_{order_id}",
            "audit_id": f"a_{order_id}",
            "event_id": event_id,
            "event_name": event_name,
            "league": league,
            "league_name": league,
            "competition_type": competition_type,
            "exec_side": exec_side,
            "side": "home",
            "line": -0.5,
            "is_live": is_live,
            "odd_at_decision": 2.0,
            "policy": {"policy_version": policy_version, "stake_requested": stake},
            "kickoff": "2026-07-01T18:00:00+00:00",
        },
        "result": {
            "status": status,
            "created_at": created,
            "finished_at": created,
            "execution_id": f"ex_{order_id}",
            "audit_id": f"a_{order_id}",
            "event_id": event_id,
            "event_name": event_name,
            "league": league,
            "league_name": league,
            "exec_side": exec_side,
            "side": "home",
            "line": -0.5,
            "is_live": is_live,
            "odd_at_decision": 2.0,
            "odd_final": 1.98,
            "limit_final": 150,
            "bookie_final": "Pinnacle",
            "policy": {"policy_version": policy_version, "stake_requested": stake},
            "raw": {
                "sent": {"stake": stake, "odd": 1.98},
                "order_resp": {"data": {"order_id": order_id}},
                "value_sizing": {"slippage_pre_pct": -0.5, "pre_submit_ms": 1200},
            },
        },
    }


@pytest.fixture
def fixture_root(tmp_path: Path) -> Path:
    rows = [
        _live_ok(order_id="1001", created="2026-06-30T10:00:00+00:00", league="Club Friendly", event_id="e1"),
        _live_ok(order_id="1002", created="2026-07-01T10:00:00+00:00", league="England Premier League", event_id="e2"),
        _live_ok(order_id="1003", created="2026-07-02T10:00:00+00:00", league="International Friendlies", event_id="e3"),
        _live_ok(order_id="1004", created="2026-07-03T10:00:00+00:00", league="Spain La Liga", event_id="e4"),
        # exclusions
        _live_ok(order_id="2001", created="2026-07-01T10:00:00+00:00", league="X", event_id="ex1", status="DRY_OK"),
        _live_ok(order_id="2002", created="2026-07-01T10:00:00+00:00", league="X", event_id="ex2", exec_side="lay"),
        # Back In via explicit market_regime (executor is_live means LIVE mode, not in-play)
        {
            **_live_ok(order_id="2003", created="2026-07-01T10:00:00+00:00", league="X", event_id="ex3"),
            "request": {
                **_live_ok(order_id="2003", created="2026-07-01T10:00:00+00:00", league="X", event_id="ex3")["request"],
                "market_regime": "inplay",
            },
            "result": {
                **_live_ok(order_id="2003", created="2026-07-01T10:00:00+00:00", league="X", event_id="ex3")["result"],
                "market_regime": "inplay",
            },
        },
        _live_ok(
            order_id="2004",
            created="2026-07-01T10:00:00+00:00",
            league="X",
            event_id="ex4",
            policy_version="legacy_bridge_h3b_live_v0",
        ),
        _live_ok(order_id="2005", created="2026-07-01T10:00:00+00:00", league="X", event_id="ex5", stake=20.0),
        # duplicate order_id
        _live_ok(order_id="1002", created="2026-07-01T11:00:00+00:00", league="England Premier League", event_id="e2"),
    ]
    _write_jsonl(tmp_path / "logs" / "executor_live.jsonl", rows)

    # balance
    acc = tmp_path / "logs" / "accounting"
    acc.mkdir(parents=True)
    bal = acc / "20260703_120000__balance.csv"
    bal.write_text(
        "order id,amount,type,note,post date,status\n"
        "1001,-10,settled,loss,2026-06-30T20:00:00+00:00,ok\n"
        "1002,9.8,settled,win,2026-07-01T20:00:00+00:00,ok\n"
        "1003,0,settled,void,2026-07-02T20:00:00+00:00,ok\n"
        "1004,-10,settled,loss,2026-07-03T20:00:00+00:00,ok\n",
        encoding="utf-8",
    )
    (acc / "20260703_120000__open_stakes.csv").write_text("order id\n", encoding="utf-8")

    # CLV
    snaps = []
    for oid, clv in (("1001", -1.0), ("1002", 0.5), ("1003", 0.1), ("1004", -0.2)):
        for w in ("POST_5M", "POST_15M", "CLOSING"):
            snaps.append(
                {
                    "order_id": oid,
                    "window_name": w,
                    "quality_status": "VALID_STRICT",
                    "clv_raw_pct": clv,
                    "snapshot_distance_sec": 30,
                }
            )
    _write_jsonl(tmp_path / "logs" / "h3bup_clv_snapshots.jsonl", snaps)
    (tmp_path / "logs" / "wf_policy_current.json").write_text('{"policy":"H3BUP_vNext"}\n', encoding="utf-8")
    (tmp_path / "logs" / "bridge_risk_params.json").write_text('{"ok":true}\n', encoding="utf-8")
    return tmp_path


def test_01_primary_only_h3bup(fixture_root):
    rows, meta = load_primary_h3bup_universe(
        fixture_root / "logs" / "executor_live.jsonl",
        cutoff=datetime(2026, 7, 10, tzinfo=timezone.utc),
    )
    assert meta["n"] == 4
    assert all(r["policy_version"] == "H3BUP_vNext_20260629" for r in rows)


def test_02_legacy_excluded(fixture_root):
    rows, _ = load_primary_h3bup_universe(
        fixture_root / "logs" / "executor_live.jsonl",
        cutoff=datetime(2026, 7, 10, tzinfo=timezone.utc),
    )
    assert "2004" not in {r["order_id"] for r in rows}


def test_03_stake20_excluded(fixture_root):
    rows, meta = load_primary_h3bup_universe(
        fixture_root / "logs" / "executor_live.jsonl",
        cutoff=datetime(2026, 7, 10, tzinfo=timezone.utc),
    )
    assert meta["excluded"]["stake_20_legacy"] >= 1
    assert "2005" not in {r["order_id"] for r in rows}


def test_04_back_in_excluded(fixture_root):
    rows, meta = load_primary_h3bup_universe(
        fixture_root / "logs" / "executor_live.jsonl",
        cutoff=datetime(2026, 7, 10, tzinfo=timezone.utc),
    )
    assert meta["excluded"]["not_pre"] >= 1


def test_05_lay_excluded(fixture_root):
    rows, meta = load_primary_h3bup_universe(
        fixture_root / "logs" / "executor_live.jsonl",
        cutoff=datetime(2026, 7, 10, tzinfo=timezone.utc),
    )
    assert meta["excluded"]["not_back"] >= 1


def test_06_dry_ok_excluded(fixture_root):
    rows, meta = load_primary_h3bup_universe(
        fixture_root / "logs" / "executor_live.jsonl",
        cutoff=datetime(2026, 7, 10, tzinfo=timezone.utc),
    )
    assert meta["excluded"]["dry_ok"] >= 1


def test_07_order_id_dedup(fixture_root):
    rows, meta = load_primary_h3bup_universe(
        fixture_root / "logs" / "executor_live.jsonl",
        cutoff=datetime(2026, 7, 10, tzinfo=timezone.utc),
    )
    assert len([r for r in rows if r["order_id"] == "1002"]) == 1
    assert meta["excluded"]["duplicates_collapsed"] >= 1


def test_08_09_10_classification_no_pnl_settlement_clv():
    # classify_entity signature has no pnl/settlement/clv params
    import inspect

    sig = inspect.signature(classify_entity)
    for forbidden in ("pnl", "settlement", "clv", "roi", "odd_final"):
        assert forbidden not in sig.parameters


def test_11_structured_flag_priority():
    r = classify_entity(structured_flag="official", league_name="Club Friendly")
    assert r.friendly_class == "CONFLICT"


def test_12_league_name_fallback():
    r = classify_entity(league_name="Club Friendlies")
    assert r.friendly_class == "FRIENDLY"
    assert r.friendly_source == "league_name"


def test_13_event_name_last_fallback():
    r = classify_entity(event_name="Friendly Match: A vs B")
    assert r.friendly_class == "FRIENDLY"
    assert r.friendly_source == "event_name"


def test_14_unknown_unclassified():
    r = classify_entity(event_name="Team A vs Team B")
    assert r.friendly_class == "UNCLASSIFIED"


def test_15_conflict():
    r = classify_entity(competition_type="official", competition_name="International Friendly")
    assert r.friendly_class == "CONFLICT"


def test_16_unclassified_not_non_friendly():
    r = classify_entity()
    assert r.friendly_class == "UNCLASSIFIED"
    assert r.friendly_class != "NON_FRIENDLY"


def test_17_regex_word_boundary():
    # "unfriendly" should NOT match \bfriendly\b alone as a club name false positive differently;
    # "NotFriendlyLeague" without word boundary wouldn't match — our patterns use \bfriendly\b
    r = classify_entity(league_name="Super League")
    assert r.friendly_class == "NON_FRIENDLY"
    # ensure pattern has word boundary
    assert any("\\b" in pat.pattern for _, pat in FRIENDLY_NAME_RULES)


def test_18_original_text_preserved():
    r = classify_entity(league_name="  Club Friendly  ")
    assert r.friendly_raw_value == "  Club Friendly  "
    assert normalize_text(r.friendly_raw_value) == "club friendly"


def test_19_mapping_checksum(tmp_path):
    rows = build_classification_mapping(
        [
            {"order_id": "1", "league_name": "Club Friendly", "event_id": "e"},
            {"order_id": "2", "league_name": "La Liga", "event_id": "f"},
        ]
    )
    arts = write_freeze_artifacts(tmp_path, rows, run_id="t")
    assert arts["checksum"].exists()
    assert mapping_checksum(rows) in arts["checksum"].read_text()


def test_20_21_22_23_classes_stake_pnl_settlement_reconcile(fixture_root):
    bundle = run_analysis(
        root=fixture_root,
        cutoff=datetime(2026, 7, 10, tzinfo=timezone.utc),
        run_id="testfix",
        n_boot=50,
        n_perm=50,
    )
    rows = bundle["order_rows"]
    assert len(rows) == 4
    assert sum(1 for r in rows if r["friendly_class"] == "FRIENDLY") == 2
    assert sum(1 for r in rows if r["friendly_class"] == "NON_FRIENDLY") == 2
    total_stake = sum(float(r["stake"]) for r in rows)
    assert abs(total_stake - 40.0) < 1e-9
    pb = performance_block(rows)
    assert pb["pnl_resolved"] is not None
    assert abs(pb["pnl_resolved"] - (-10 + 9.8 + 0 + -10)) < 1e-9


def test_24_void_in_denominator(fixture_root):
    bundle = run_analysis(root=fixture_root, cutoff=datetime(2026, 7, 10, tzinfo=timezone.utc), run_id="void", n_boot=20, n_perm=20)
    pb = performance_block(bundle["order_rows"])
    # void stake 10 included
    assert abs(pb["stake_resolved_total"] - 40.0) < 1e-9
    assert "void" in " ".join(pb["notes"]).lower()


def test_25_open_not_loss():
    st, pnl = classify_settlement(order_id="1", pnl=None, in_open=True, has_accounting_row=False)
    assert st == "OPEN"
    assert pnl is None


def test_26_missing_not_zero():
    st, pnl = classify_settlement(order_id="1", pnl=None, in_open=False, has_accounting_row=False)
    assert st == "MISSING"
    assert pnl is None


def test_27_28_clv_valid_strict_windows(fixture_root):
    bundle = run_analysis(root=fixture_root, cutoff=datetime(2026, 7, 10, tzinfo=timezone.utc), run_id="clv", n_boot=20, n_perm=20)
    windows = {r["window"] for r in bundle["clv_summary"]}
    assert windows == {"POST_5M", "POST_15M", "CLOSING"}
    assert all(r.get("metric_contract") == "VALID_STRICT_ONLY" for r in bundle["clv_summary"])


def test_29_30_source_missing_line_mismatch_separated(tmp_path):
    path = tmp_path / "s.jsonl"
    _write_jsonl(
        path,
        [
            {"order_id": "1", "window_name": "CLOSING", "quality_status": "SOURCE_MISSING", "failure_reason": "source_missing"},
            {"order_id": "2", "window_name": "CLOSING", "quality_status": "LINE_MISMATCH", "failure_reason": "line_mismatch"},
            {"order_id": "3", "window_name": "CLOSING", "quality_status": "VALID_STRICT", "clv_raw_pct": 1.0},
        ],
    )
    m = load_clv_by_order(path)
    assert m["1"]["clv_source_missing"] is True
    assert m["2"]["clv_line_mismatch"] is True
    assert m["3"]["clv_closing_valid_strict"] is True


def test_31_bootstrap_clustered():
    a = [{"event_id": "e1", "order_id": "1", "settlement_status": "SETTLED_DECIDED", "pnl": -10, "stake": 10, "friendly_class": "FRIENDLY"}]
    b = [{"event_id": "e2", "order_id": "2", "settlement_status": "SETTLED_DECIDED", "pnl": 5, "stake": 10, "friendly_class": "NON_FRIENDLY"}]
    # expand a bit
    a = a * 5
    b = b * 5
    for i, r in enumerate(a):
        r = dict(r)
        r["order_id"] = f"a{i}"
        a[i] = r
    for i, r in enumerate(b):
        r = dict(r)
        r["order_id"] = f"b{i}"
        b[i] = r
    out = clustered_bootstrap_diff(a, b, stat_fn=lambda rs: performance_block(rs).get("roi_resolved"), n_boot=100)
    assert "estimate" in out
    assert out.get("n_events_a") is not None or out.get("status") == "INSUFFICIENT_N"


def test_32_low_n_insufficient():
    assert sample_gate(10) == "VERY_LOW_N"
    assert sample_gate(50) == "INSUFFICIENT_N"
    stats = run_stat_tests(
        [
            {"friendly_class": "FRIENDLY", "event_id": "1", "settlement_status": "SETTLED_DECIDED", "pnl": -1, "stake": 10},
            {"friendly_class": "NON_FRIENDLY", "event_id": "2", "settlement_status": "SETTLED_DECIDED", "pnl": 1, "stake": 10},
        ],
        n_boot=20,
        n_perm=20,
    )
    assert stats["tests"]["roi_resolved_diff_friendly_minus_non"]["status"] in {"VERY_LOW_N", "INSUFFICIENT_N"}


def test_33_34_concentration_and_lolo(fixture_root):
    bundle = run_analysis(root=fixture_root, cutoff=datetime(2026, 7, 10, tzinfo=timezone.utc), run_id="conc", n_boot=20, n_perm=20)
    assert bundle["concentration"]
    assert bundle["leave_one_league_out"]


def test_35_secondary_separated(fixture_root):
    bundle = run_analysis(root=fixture_root, cutoff=datetime(2026, 7, 10, tzinfo=timezone.utc), run_id="sec", n_boot=20, n_perm=20)
    assert bundle["secondary_meta"].get("universe") == "HISTORICAL_COMPARABLE_BACK_PRE" or bundle["secondary_meta"].get("diagnostic_only")
    # primary ROI not merged: secondary meta marked diagnostic
    assert bundle["primary_meta"]["universe"] == "H3BUP_vNext_exact"


def test_balance_snapshot_not_multiplied(tmp_path: Path):
    """Full ledger dumps must not be summed N times (ROI << -100% bug)."""
    from ops.h3bup_friendly_analysis.settlement import load_pnl_by_order

    acc = tmp_path / "logs" / "accounting"
    acc.mkdir(parents=True)
    header = "transaction_id,order id,amount,type,note\n"
    row = "tx1,999,-10,bet,Settlement of Bet x\n"
    paths = []
    for i in range(5):
        p = acc / f"2026070{i}_120000__balance.csv"
        p.write_text(header + row, encoding="utf-8")
        paths.append(p)
    pnl = load_pnl_by_order(paths)
    assert abs(pnl["999"] - (-10.0)) < 1e-9


def test_single_latest_balance_used_in_runner_contract():
    import inspect
    from ops.h3bup_friendly_analysis import run as runmod

    src = inspect.getsource(runmod.run_analysis)
    assert "bal_all[-1:]" in src or "balance_paths = bal_all[-1:]" in src
