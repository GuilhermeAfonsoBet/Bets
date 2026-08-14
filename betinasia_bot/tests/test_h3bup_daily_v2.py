"""Mandatory Daily V2 contract tests (Phase 2R)."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from ops.daily_v2 import DAILY_FAST_LE_6S_MS, STUDY_FAST_LT_4S_MS
from ops.daily_v2.io_atomic import atomic_write_text
from ops.daily_v2.performance import compute_settlement_and_performance
from ops.daily_v2.render import render_markdown
from ops.daily_v2.statuses import metric_envelope
from ops.daily_v2.time_windows import closed_day_window, execution_day_utc, in_half_open, resolve_window
from ops.daily_v2.universes import classify_fast_buckets, load_executor_orders


def test_cohort_uses_created_at_utc():
    dt = datetime(2026, 7, 28, 23, 30, tzinfo=timezone.utc)
    assert execution_day_utc(dt).isoformat() == "2026-07-28"


def test_post_date_does_not_change_execution_day():
    # post date next day must not redefine cohort
    created = datetime(2026, 7, 28, 1, 0, tzinfo=timezone.utc)
    assert execution_day_utc(created).isoformat() == "2026-07-28"
    # settlement day is independent metadata
    post = date(2026, 7, 29)
    assert post != execution_day_utc(created)


def test_utc_midnight_boundaries():
    start, end = closed_day_window(date(2026, 7, 28))
    assert in_half_open(datetime(2026, 7, 28, 0, 0, tzinfo=timezone.utc), start, end)
    assert not in_half_open(datetime(2026, 7, 29, 0, 0, tzinfo=timezone.utc), start, end)
    assert in_half_open(datetime(2026, 7, 28, 23, 59, 59, tzinfo=timezone.utc), start, end)


def test_settlement_days_later_still_same_cohort():
    win = resolve_window(report_type="DAILY_CLOSED", report_date=date(2026, 7, 20))
    assert win.report_date_utc.isoformat() == "2026-07-20"
    assert win.window_end_utc.isoformat().startswith("2026-07-21")


def test_open_not_in_roi_settled():
    orders = {
        "1": {"stake": 10.0},
        "2": {"stake": 10.0},
    }
    pnl = {"1": 5.0, "2": -10.0}
    out = compute_settlement_and_performance(orders=orders, pnl_by_oid=pnl, open_oids={"2"})
    assert out["n_open"] == 1
    assert out["roi_settled"]["denominator"] == 10.0
    assert out["roi_settled"]["numerator"] == 5.0


def test_void_push():
    orders = {"1": {"stake": 10.0}}
    pnl = {"1": 0.0}
    out = compute_settlement_and_performance(orders=orders, pnl_by_oid=pnl, open_oids=set())
    assert out["n_void_push"] == 1
    assert out["roi_settled"]["value"] == 0.0
    assert out["roi_settled"]["status"] == "AVAILABLE"


def test_missing_accounting():
    orders = {"1": {"stake": 10.0}}
    out = compute_settlement_and_performance(orders=orders, pnl_by_oid={}, open_oids=set())
    assert out["n_missing_accounting"] == 1
    assert out["roi_settled"]["status"] in {"MISSING", "PARTIAL"}


def test_accounting_stale():
    orders = {"1": {"stake": 10.0}}
    out = compute_settlement_and_performance(
        orders=orders, pnl_by_oid={"1": 1.0}, open_oids=set(), accounting_health_status="STALE"
    )
    assert out["roi_settled"]["status"] == "UNAVAILABLE_STALE"
    assert out["roi_settled"]["value"] is None


def test_true_zero_vs_missing():
    z = metric_envelope(value=0, status="AVAILABLE", unit="count")
    m = metric_envelope(value=0, status="MISSING", unit="count")
    assert z["value"] == 0
    assert m["value"] is None


def test_source_failed_status():
    m = metric_envelope(status="FAILED", value=None)
    assert m["status"] == "FAILED"
    assert m["value"] is None


def _write_jsonl(path: Path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _live_ok_row(*, created_at: str, order_id, policy_version: str, exec_side: str = "Back", pre_submit_ms=1000, stake=10.0):
    return {
        "request": {
            "created_at": created_at,
            "exec_side": exec_side,
            "policy": {"policy_version": policy_version, "stake_requested": stake},
        },
        "result": {
            "status": "LIVE_OK",
            "created_at": created_at,
            "exec_side": exec_side,
            "policy": {"policy_version": policy_version, "stake_requested": stake},
            "raw": {
                "order_resp": {"data": {"order_id": order_id}},
                "sent": {"stake": stake},
                "value_sizing": {"pre_submit_ms": pre_submit_ms},
            },
        },
    }


def test_heartbeat_excluded(tmp_path):
    p = tmp_path / "exec.jsonl"
    _write_jsonl(
        p,
        [
            _live_ok_row(created_at="2026-07-28T12:00:00+00:00", order_id="100", policy_version="H3BUP_vNext_20260629", pre_submit_ms=3000),
            {
                "heartbeat": True,
                "note": "auth_guard_heartbeat",
                "request": {"created_at": "2026-07-28T12:01:00+00:00"},
                "result": {"status": "INFO", "created_at": "2026-07-28T12:01:00+00:00"},
            },
        ],
    )
    win = resolve_window(report_type="DAILY_CLOSED", report_date=date(2026, 7, 28))
    orders = load_executor_orders(p, window=win, require_h3bup=True)
    assert list(orders.keys()) == ["100"]


def test_retry_dedup_keeps_latest(tmp_path):
    p = tmp_path / "exec.jsonl"
    _write_jsonl(
        p,
        [
            _live_ok_row(created_at="2026-07-28T10:00:00+00:00", order_id="55", policy_version="H3BUP_vNext_20260629", pre_submit_ms=1000),
            _live_ok_row(created_at="2026-07-28T11:00:00+00:00", order_id="55", policy_version="H3BUP_vNext_20260629", pre_submit_ms=2000),
        ],
    )
    win = resolve_window(report_type="DAILY_CLOSED", report_date=date(2026, 7, 28))
    orders = load_executor_orders(p, window=win)
    assert orders["55"]["pre_submit_ms"] == 2000


def test_order_id_as_string(tmp_path):
    p = tmp_path / "exec.jsonl"
    _write_jsonl(
        p,
        [_live_ok_row(created_at="2026-07-28T12:00:00+00:00", order_id=999001, policy_version="H3BUP_vNext_20260629")],
    )
    win = resolve_window(report_type="DAILY_CLOSED", report_date=date(2026, 7, 28))
    orders = load_executor_orders(p, window=win)
    assert "999001" in orders


def test_policy_different_excluded(tmp_path):
    p = tmp_path / "exec.jsonl"
    _write_jsonl(
        p,
        [_live_ok_row(created_at="2026-07-28T12:00:00+00:00", order_id="7", policy_version="OTHER_POLICY")],
    )
    win = resolve_window(report_type="DAILY_CLOSED", report_date=date(2026, 7, 28))
    assert load_executor_orders(p, window=win, require_h3bup=True) == {}


def test_lay_excluded(tmp_path):
    p = tmp_path / "exec.jsonl"
    _write_jsonl(
        p,
        [_live_ok_row(created_at="2026-07-28T12:00:00+00:00", order_id="8", policy_version="H3BUP_vNext_20260629", exec_side="Lay")],
    )
    win = resolve_window(report_type="DAILY_CLOSED", report_date=date(2026, 7, 28))
    assert load_executor_orders(p, window=win) == {}


def test_daily_fast_and_study_fast_boundary():
    orders = {
        "a": {"pre_submit_ms": 4000},
        "b": {"pre_submit_ms": 3999},
        "c": {"pre_submit_ms": 6000},
        "d": {"pre_submit_ms": 6001},
        "e": {"pre_submit_ms": None},
    }
    b = classify_fast_buckets(orders)
    assert "a" in b["DAILY_FAST_LE_6S"]["order_ids"]
    assert "a" not in b["STUDY_FAST_LT_4S"]["order_ids"]
    assert "b" in b["STUDY_FAST_LT_4S"]["order_ids"]
    assert "c" in b["DAILY_FAST_LE_6S"]["order_ids"]
    assert "d" in b["DAILY_SLOW_GT_6S"]["order_ids"]
    assert "e" in b["PRE_SUBMIT_MS_NA"]["order_ids"]
    assert DAILY_FAST_LE_6S_MS == 6000
    assert STUDY_FAST_LT_4S_MS == 4000


def test_roiw_total_contract():
    orders = {"1": {"stake": 10.0}, "2": {"stake": 10.0}}
    pnl = {"1": 5.0, "2": -1.0}
    out = compute_settlement_and_performance(orders=orders, pnl_by_oid=pnl, open_oids=set())
    assert out["roiw_total_v1"]["value"] == pytest.approx(20.0)
    assert out["principal_metric"] == "roi_resolved"
    assert out["complementary_metric"] == "roi_decided_ex_void"
    assert out["parity_legacy_metric"] == "roiw_total_v1"
    assert out["roi_resolved"]["value"] == pytest.approx(0.2)


def test_fair_edge_not_implemented():
    m = metric_envelope(value=None, status="NOT_IMPLEMENTED")
    assert m["value"] is None
    assert m["status"] == "NOT_IMPLEMENTED"


def test_e2e_n0_insufficient():
    m = metric_envelope(value=None, n=0, status="INSUFFICIENT_N", unit="ms")
    assert m["value"] is None
    assert m["n"] == 0


def test_atomic_write(tmp_path):
    p = tmp_path / "x.md"
    atomic_write_text(p, "hello")
    assert p.read_text() == "hello"


def test_renderer_does_not_recompute():
    snap = {
        "schema_version": 2,
        "run_id": "t",
        "report_type": "DAILY_CLOSED",
        "report_date_utc": "2026-07-28",
        "window_start_utc": "2026-07-28T00:00:00+00:00",
        "window_end_utc": "2026-07-29T00:00:00+00:00",
        "report_cutoff_utc": "2026-07-29T12:00:00+00:00",
        "generated_at_utc": "2026-07-29T12:00:00+00:00",
        "git_commit": None,
        "policy_id": "H3BUP_vNext",
        "policy_version": "H3BUP_vNext_20260629",
        "source_manifest": {},
        "report_health": {"status": "HEALTHY"},
        "operations_health": {},
        "data_quality": {},
        "statistical_readiness": {},
        "execution_funnel": {"live_ok": metric_envelope(value=2, status="AVAILABLE", unit="count", n=2), "fast_buckets": {}},
        "settlement": {"maturity_status": "FULLY_SETTLED", "n_open": 0, "n_settled": 2, "n_void_push": 0, "n_missing_accounting": 0},
        "performance": {
            "roi_settled": metric_envelope(value=0.1, status="AVAILABLE", unit="fraction", n=2),
            "roiw_total_v1": metric_envelope(value=10.0, status="AVAILABLE", unit="percent", n=2),
            "roiw_total_v2": metric_envelope(value=10.0, status="AVAILABLE", unit="percent", n=2),
            "principal_metric": "roi_settled",
        },
        "latency": {
            "daily_fast_le_6s": metric_envelope(value=1, status="AVAILABLE"),
            "study_fast_lt_4s": metric_envelope(value=0, status="AVAILABLE"),
            "e2e_ws_to_live_ok": metric_envelope(status="INSUFFICIENT_N"),
            "detect_to_audit_overhead": metric_envelope(status="WATCH"),
        },
        "clv": {"fair_edge": metric_envelope(status="NOT_IMPLEMENTED"), "funnel": {}},
        "concentration": {"status": "INSUFFICIENT_N"},
        "exceptions": [],
        "methodology": {"cohort_timestamp": "created_at UTC"},
    }
    md = render_markdown(snap)
    assert "NOT_IMPLEMENTED" in md
    assert "DAILY_FAST_LE_6S" in md or "daily_fast" in md.lower()
    assert "0.1" in md or "10.00%" in md


def test_closed_vs_intraday_separated():
    c = resolve_window(report_type="DAILY_CLOSED", report_date=date(2026, 7, 28))
    i = resolve_window(
        report_type="INTRADAY",
        report_date=date(2026, 7, 29),
        cutoff_utc=datetime(2026, 7, 29, 15, 0, tzinfo=timezone.utc),
    )
    assert c.report_type == "DAILY_CLOSED"
    assert i.report_type == "INTRADAY"
    assert i.window_end_utc.hour == 15


def test_empty_healthy_cohort(tmp_path):
    p = tmp_path / "exec.jsonl"
    p.write_text("", encoding="utf-8")
    win = resolve_window(report_type="DAILY_CLOSED", report_date=date(2026, 7, 28))
    orders = load_executor_orders(p, window=win)
    out = compute_settlement_and_performance(orders=orders, pnl_by_oid={}, open_oids=set())
    assert out["live_ok_total"] == 0
    assert out["roi_settled"]["status"] == "AVAILABLE"


def test_stake20_legacy_flagged_in_exceptions(tmp_path):
    from ops.daily_v2.canonical import build_snapshot
    from ops.daily_v2.time_windows import resolve_window
    from datetime import date
    # minimal fixture root
    root = tmp_path
    (root / "logs").mkdir()
    exec_p = root / "logs" / "executor_live.jsonl"
    rows = [
        _live_ok_row(created_at="2026-07-28T12:00:00+00:00", order_id="1", policy_version="H3BUP_vNext_20260629", stake=20.0, pre_submit_ms=1000),
        _live_ok_row(created_at="2026-07-28T13:00:00+00:00", order_id="2", policy_version="H3BUP_vNext_20260629", stake=10.0, pre_submit_ms=1000),
    ]
    _write_jsonl(exec_p, rows)
    snap = build_snapshot(root=root, window=resolve_window(report_type="DAILY_CLOSED", report_date=date(2026,7,28)), require_h3bup=True)
    alerts = [e["alert_id"] for e in snap.get("exceptions") or []]
    assert any(a.startswith("stake_mismatch:1") for a in alerts)


def test_json_markdown_consistent_live_ok(tmp_path):
    from ops.daily_v2.canonical import build_snapshot
    from ops.daily_v2.render import render_markdown
    from ops.daily_v2.time_windows import resolve_window
    from datetime import date
    root = tmp_path
    (root / "logs").mkdir()
    _write_jsonl(root / "logs" / "executor_live.jsonl", [
        _live_ok_row(created_at="2026-07-28T12:00:00+00:00", order_id="9", policy_version="H3BUP_vNext_20260629")
    ])
    snap = build_snapshot(root=root, window=resolve_window(report_type="DAILY_CLOSED", report_date=date(2026,7,28)))
    md = render_markdown(snap)
    assert str(((snap.get("execution_funnel") or {}).get("live_ok") or {}).get("value")) in md


def test_missing_not_zero_in_envelope():
    m = metric_envelope(value=0, status="STALE", unit="count")
    assert m["value"] is None


def test_clv_windows_keys_separated(tmp_path):
    from ops.daily_v2.canonical import build_snapshot
    from ops.daily_v2.time_windows import resolve_window
    from datetime import date
    root = tmp_path
    (root / "logs").mkdir()
    (root / "logs" / "executor_live.jsonl").write_text("")
    health = {
        "status": "WATCH",
        "collection_started_at_utc": "2026-07-29T20:07:50+00:00",
        "live_ok_after_activation": 2,
        "post_5m_valid_strict": 1,
        "post_15m_valid_strict": 0,
        "closing_valid_strict": 0,
    }
    (root / "logs" / "h3bup_clv_health.json").write_text(json.dumps(health))
    snap = build_snapshot(root=root, window=resolve_window(report_type="DAILY_CLOSED", report_date=date(2026,7,28)))
    clv = snap["clv"]
    assert "post_5m_valid_strict" in clv and "post_15m_valid_strict" in clv and "closing_valid_strict" in clv
    assert clv["fair_edge"]["status"] == "NOT_IMPLEMENTED"


def test_publish_flag_default_off_in_cli(monkeypatch, tmp_path):
    from ops.daily_v2 import __main__ as m
    monkeypatch.setenv("H3BUP_DAILY_V2_ENABLED", "1")
    monkeypatch.setenv("H3BUP_DAILY_V2_PUBLISH", "0")
    monkeypatch.setenv("H3BUP_DAILY_V2_FAIL_OPEN", "1")
    root = tmp_path
    (root / "logs").mkdir()
    (root / "logs" / "executor_live.jsonl").write_text("")
    out = root / "out"
    rc = m.run(["--root", str(root), "--out-dir", str(out), "--report-type", "DAILY_CLOSED", "--report-date", "2026-07-28"])
    assert rc == 0
    assert not (out / "published").exists()


def test_preview_banner_in_markdown():
    from ops.daily_v2.render import render_markdown
    from ops.daily_v2.statuses import metric_envelope
    snap = {
        "schema_version": 2,
        "run_id": "abc",
        "report_type": "DAILY_CLOSED",
        "report_date_utc": "2026-07-28",
        "window_start_utc": "2026-07-28T00:00:00+00:00",
        "window_end_utc": "2026-07-29T00:00:00+00:00",
        "report_cutoff_utc": "2026-07-28T22:01:00+00:00",
        "generated_at_utc": "2026-07-28T22:10:00+00:00",
        "git_commit": None,
        "policy_id": "H3BUP_vNext",
        "policy_version": "H3BUP_vNext_20260629",
        "source_manifest": {},
        "report_health": {"status": "HEALTHY"},
        "operations_health": {},
        "data_quality": {},
        "statistical_readiness": {},
        "execution_funnel": {"live_ok": metric_envelope(value=0, status="AVAILABLE", unit="count"), "fast_buckets": {}},
        "settlement": {"maturity_status": "FULLY_SETTLED", "n_open": 0, "n_settled": 0, "n_void_push": 0, "n_missing_accounting": 0},
        "performance": {
            "roi_settled": metric_envelope(status="AVAILABLE", value=0, unit="fraction"),
            "roiw_total_v1": metric_envelope(status="AVAILABLE", value=0, unit="percent"),
            "roiw_total_v2": metric_envelope(status="AVAILABLE", value=0, unit="percent"),
            "principal_metric": "roi_settled",
        },
        "latency": {
            "daily_fast_le_6s": metric_envelope(value=0, status="AVAILABLE"),
            "study_fast_lt_4s": metric_envelope(value=0, status="AVAILABLE"),
            "e2e_ws_to_live_ok": metric_envelope(status="INSUFFICIENT_N"),
            "detect_to_audit_overhead": metric_envelope(status="WATCH"),
        },
        "clv": {"fair_edge": metric_envelope(status="NOT_IMPLEMENTED"), "funnel": {}},
        "concentration": {"status": "INSUFFICIENT_N"},
        "exceptions": [],
        "methodology": {},
        "parity": {"parity_status": "CUTOFF_ALIGNED", "v1_report_cutoff_utc": "2026-07-28T22:01:00+00:00", "v2_comparison_cutoff_utc": "2026-07-28T22:01:00+00:00"},
    }
    md = render_markdown(snap)
    assert "DAILY V2 — PREVIEW / NÃO OFICIAL" in md


def test_preview_filename_contract():
    from ops.daily_v2.preview_labels import preview_pdf_filename, validate_preview_artifacts
    name = preview_pdf_filename("20260728", "deadbeef")
    assert name.startswith("H3BUP_DAILY_V2_PREVIEW_")
    ok, _ = validate_preview_artifacts(md="# DAILY V2 — PREVIEW / NÃO OFICIAL\n", pdf_name=name)
    assert ok
    ok2, reason = validate_preview_artifacts(md="# no label", pdf_name=name)
    assert not ok2


def test_telegram_caption_contains_preview():
    from ops.daily_v2.preview_labels import build_telegram_caption
    cap = build_telegram_caption(
        report_date_utc="2026-07-28",
        report_cutoff_utc="2026-07-28T22:01:00+00:00",
        generated_at_utc="2026-07-28T22:10:00+00:00",
        schema_version=2,
        run_id="abc",
        parity_status="CUTOFF_ALIGNED",
    )
    assert "PREVIEW / NÃO OFICIAL" in cap
    assert "Não substitui o Daily V1 oficial" in cap


def test_official_flag_refused(monkeypatch, tmp_path):
    from ops.daily_v2 import __main__ as m
    monkeypatch.setenv("H3BUP_DAILY_V2_ENABLED", "1")
    monkeypatch.setenv("H3BUP_DAILY_V2_OFFICIAL", "1")
    monkeypatch.setenv("H3BUP_DAILY_V2_FAIL_OPEN", "1")
    root = tmp_path
    (root / "logs").mkdir()
    (root / "logs" / "executor_live.jsonl").write_text("")
    rc = m.run(["--root", str(root), "--out-dir", str(root / "out"), "--report-date", "2026-07-28"])
    assert rc == 0
    assert not list((root / "out").glob("H3BUP_DAILY_V2_PREVIEW_*.pdf"))


def test_parity_cutoff_alignment():
    from ops.daily_v2.cutoff import resolve_parity_cutoffs
    from datetime import datetime, timezone
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        day_dir = root / "logs" / "daily_reports" / "20260728"
        day_dir.mkdir(parents=True)
        (day_dir / "report_daily.md").write_text(
            "# Daily\n\n- Dia do relatório (UTC): `20260728`\n- Gerado em (UTC): `2026-07-28T22:01:08+00:00`\n",
            encoding="utf-8",
        )
        out = resolve_parity_cutoffs(
            root=root,
            report_date_utc="2026-07-28",
            v2_generated_at=datetime(2026, 7, 28, 22, 10, tzinfo=timezone.utc),
        )
        assert out["parity_status"] == "CUTOFF_ALIGNED"
        assert out["v1_report_cutoff_utc"].startswith("2026-07-28T22:01:08")
        assert out["v2_comparison_cutoff_utc"] == out["v1_report_cutoff_utc"]
        assert out["cutoffs_equal"] is True


def test_pdf_preview_label(tmp_path):
    pytest.importorskip("reportlab")
    from ops.daily_v2.pdf_preview import render_preview_pdf, pdf_contains_preview_label
    # ensure renderer available via workspace docs path
    md = "# DAILY V2 — PREVIEW / NÃO OFICIAL\n\nHello\n\n## Sec\n\n- a\n"
    pdf = tmp_path / "H3BUP_DAILY_V2_PREVIEW_20260728_test.pdf"
    # docs path relative to cwd when tests run from betinasia_bot
    render_preview_pdf(md, pdf, root=Path('.').resolve())
    assert pdf.exists() and pdf.stat().st_size > 100
    assert pdf_contains_preview_label(pdf)


def test_token_not_in_telegram_evidence(tmp_path, monkeypatch):
    from ops.daily_v2.telegram_preview import maybe_send_telegram_preview
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:FAKE_SECRET_TOKEN_VALUE")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "999")
    # force skip by missing preview label in md / bad pdf
    pdf = tmp_path / "bad.pdf"
    pdf.write_text("x")
    evid = maybe_send_telegram_preview(
        root=tmp_path,
        out_dir=tmp_path,
        day_s="20260728",
        run_id="r1",
        snap={"generated_at_utc": "2026-07-28T22:10:00+00:00", "report_date_utc": "2026-07-28", "schema_version": 2, "parity": {}},
        md_text="no banner",
        pdf_path=pdf,
        parity_status="X",
        telegram_preview_enabled=True,
        official=False,
    )
    blob = str(evid)
    assert "FAKE_SECRET_TOKEN_VALUE" not in blob
    assert evid["telegram_status"] == "PREVIEW_LABEL_VALIDATION_FAILED"
