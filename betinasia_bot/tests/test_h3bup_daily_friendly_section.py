"""Tests for Daily Friendly vs Non-Friendly shadow section."""

from __future__ import annotations

from pathlib import Path

from ops.daily_v2.friendly_section import build_friendly_section, render_friendly_markdown
from ops.h3bup_friendly_analysis import FRIENDLY_CLASSIFICATION_VERSION


def test_friendly_section_splits_classes(tmp_path: Path):
    # league map for enrichment
    (tmp_path / "logs").mkdir(parents=True)
    (tmp_path / "logs" / "h3bup_friendly_league_map.csv").write_text(
        "event_id,league,league_name,competition,competition_name\n"
        "e1,Club Friendly,Club Friendly,Club Friendly,Club Friendly\n"
        "e2,England Premier League,England Premier League,EPL,EPL\n",
        encoding="utf-8",
    )
    orders = {
        "1": {"order_id": "1", "stake": 10.0, "event_id": "e1"},
        "2": {"order_id": "2", "stake": 10.0, "event_id": "e2"},
        "3": {"order_id": "3", "stake": 10.0, "event_id": "e_unknown"},
    }
    pnl = {"1": -10.0, "2": 5.0}
    open_oids = set()
    sec = build_friendly_section(
        root=tmp_path,
        orders=orders,
        pnl_by_oid=pnl,
        open_oids=open_oids,
    )
    assert sec["classification_version"] == FRIENDLY_CLASSIFICATION_VERSION
    assert sec["official_filter"] is False
    assert sec["n_friendly"] == 1
    assert sec["n_non_friendly"] == 1
    assert sec["n_unclassified"] == 1
    assert sec["n_conflict"] == 0
    by = {r["class"]: r for r in sec["rows"]}
    assert by["FRIENDLY"]["n_live_ok"] == 1
    assert by["NON_FRIENDLY"]["n_live_ok"] == 1
    assert by["UNCLASSIFIED"]["n_live_ok"] == 1


def test_unclassified_not_merged_into_non_friendly(tmp_path: Path):
    (tmp_path / "logs").mkdir(parents=True)
    orders = {"1": {"order_id": "1", "stake": 10.0, "event_id": ""}}
    sec = build_friendly_section(root=tmp_path, orders=orders, pnl_by_oid={}, open_oids=set())
    assert sec["n_unclassified"] == 1
    assert sec["n_non_friendly"] == 0


def test_render_contains_shadow_disclaimer(tmp_path: Path):
    (tmp_path / "logs").mkdir(parents=True)
    orders = {
        "1": {"order_id": "1", "stake": 10.0, "league_name": "Club Friendly"},
        "2": {"order_id": "2", "stake": 10.0, "league_name": "Spain La Liga"},
    }
    sec = build_friendly_section(
        root=tmp_path,
        orders=orders,
        pnl_by_oid={"1": -10.0, "2": -10.0},
        open_oids=set(),
    )
    md = render_friendly_markdown(sec)
    assert "não é filtro operacional" in md
    assert "FRIENDLY" in md
    assert "NON_FRIENDLY" in md
    assert FRIENDLY_CLASSIFICATION_VERSION in md


def test_canonical_includes_friendly_breakdown(tmp_path: Path):
    import json
    from datetime import date

    from ops.daily_v2.canonical import build_snapshot
    from ops.daily_v2.time_windows import resolve_window

    logs = tmp_path / "logs"
    logs.mkdir()
    # minimal executor line
    line = {
        "request": {
            "created_at": "2026-07-30T12:00:00+00:00",
            "exec_side": "back",
            "event_id": "ev1",
            "policy": {"policy_version": "H3BUP_vNext_20260629", "stake_requested": 10},
        },
        "result": {
            "status": "LIVE_OK",
            "created_at": "2026-07-30T12:00:00+00:00",
            "exec_side": "back",
            "event_id": "ev1",
            "policy": {"policy_version": "H3BUP_vNext_20260629", "stake_requested": 10},
            "raw": {"sent": {"stake": 10}, "order_resp": {"data": {"order_id": "oid1"}}, "value_sizing": {"pre_submit_ms": 1000}},
        },
    }
    (logs / "executor_live.jsonl").write_text(json.dumps(line) + "\n", encoding="utf-8")
    (logs / "h3bup_friendly_league_map.csv").write_text(
        "event_id,league_name\nev1,Club Friendly\n", encoding="utf-8"
    )
    (logs / "wf_policy_current.json").write_text('{"ok":1}\n', encoding="utf-8")
    (logs / "bridge_risk_params.json").write_text('{"ok":1}\n', encoding="utf-8")
    acct = logs / "accounting"
    acct.mkdir()
    (acct / "20260730_120000__balance.csv").write_text(
        "order id,amount,type,note\noid1,-10,bet,loss\n", encoding="utf-8"
    )
    (acct / "20260730_120000__open_stakes.csv").write_text("order id\n", encoding="utf-8")
    (acct / "accounting_health.json").write_text('{"status":"HEALTHY"}\n', encoding="utf-8")

    win = resolve_window(report_type="DAILY_CLOSED", report_date=date(2026, 7, 30))
    snap = build_snapshot(root=tmp_path, window=win, require_h3bup=True)
    assert "friendly_breakdown" in snap
    fb = snap["friendly_breakdown"]
    assert fb.get("official_filter") is False
    assert snap["safety"].get("alters_friendly_filter") is False
    assert fb.get("n_friendly") == 1

    from ops.daily_v2.render import render_markdown

    md = render_markdown(snap)
    assert "Friendly vs Non-Friendly" in md
    assert "não é filtro operacional" in md
