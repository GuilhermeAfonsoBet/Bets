"""P0 Daily V1/V2 correction tests (FASE 2R-P0)."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from ops.daily_v2.canonical import build_snapshot
from ops.daily_v2.clv_section import build_clv_section
from ops.daily_v2.compare_v1 import compare_snapshots
from ops.daily_v2.diff_previous import diff_snapshots
from ops.daily_v2.formatters import fmt_age, fmt_money, fmt_pct, fmt_ts
from ops.daily_v2.health_model import build_health_model, derive_alerts, evaluate_config_file
from ops.daily_v2.performance import compute_settlement_and_performance
from ops.daily_v2.render import render_markdown
from ops.daily_v2.statuses import metric_envelope
from ops.daily_v2.time_windows import resolve_window
from ops.daily_v2.universes import load_executor_orders


def _write_jsonl(path: Path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _live_ok_row(*, created_at: str, order_id, policy_version: str = "H3BUP_vNext_20260629", exec_side: str = "Back", pre_submit_ms=1000, stake=10.0):
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


def test_p0_same_report_date_parity_fields():
    from ops.daily_v2.cutoff import resolve_parity_cutoffs

    root = Path("/tmp")  # unused when v1 missing
    out = resolve_parity_cutoffs(
        root=root,
        report_date_utc="2026-07-28",
        v2_generated_at=datetime(2026, 7, 28, 22, 10, tzinfo=timezone.utc),
    )
    assert out["v2_generated_at_utc"].startswith("2026-07-28")


def test_p0_cohort_created_at_utc(tmp_path):
    root = tmp_path
    (root / "logs").mkdir()
    _write_jsonl(
        root / "logs" / "executor_live.jsonl",
        [
            _live_ok_row(created_at="2026-07-28T23:50:00+00:00", order_id="1"),
            _live_ok_row(created_at="2026-07-29T00:01:00+00:00", order_id="2"),
        ],
    )
    snap = build_snapshot(root=root, window=resolve_window(report_type="DAILY_CLOSED", report_date=date(2026, 7, 28)))
    assert snap["report_date_utc"] == "2026-07-28"
    assert snap["window_start_utc"].startswith("2026-07-28T00:00:00")
    assert snap["window_end_utc"].startswith("2026-07-29T00:00:00")
    assert ((snap["execution_funnel"]["live_ok"] or {}).get("value")) == 1


def test_p0_parity_cutoff_equal_when_v1_present(tmp_path):
    from ops.daily_v2.cutoff import resolve_parity_cutoffs

    day_dir = tmp_path / "logs" / "daily_reports" / "20260728"
    day_dir.mkdir(parents=True)
    (day_dir / "report_daily.md").write_text(
        "# Daily\n\n- Dia do relatório (UTC): `20260728`\n- Gerado em (UTC): `2026-07-28T22:01:08+00:00`\n",
        encoding="utf-8",
    )
    out = resolve_parity_cutoffs(
        root=tmp_path,
        report_date_utc="2026-07-28",
        v2_generated_at=datetime(2026, 7, 28, 22, 10, tzinfo=timezone.utc),
    )
    assert out["parity_status"] == "CUTOFF_ALIGNED"
    assert out["v1_report_cutoff_utc"] == out["v2_comparison_cutoff_utc"]


def test_p0_policy_legacy_excluded(tmp_path):
    root = tmp_path
    (root / "logs").mkdir()
    _write_jsonl(
        root / "logs" / "executor_live.jsonl",
        [
            _live_ok_row(created_at="2026-07-28T12:00:00+00:00", order_id="1", policy_version="bridge_h3b_live_v0"),
            _live_ok_row(created_at="2026-07-28T13:00:00+00:00", order_id="2"),
        ],
    )
    orders = load_executor_orders(
        root / "logs" / "executor_live.jsonl",
        window=resolve_window(report_type="DAILY_CLOSED", report_date=date(2026, 7, 28)),
        require_h3bup=True,
    )
    assert list(orders.keys()) == ["2"]


def test_p0_stake20_flagged(tmp_path):
    root = tmp_path
    (root / "logs").mkdir()
    _write_jsonl(
        root / "logs" / "executor_live.jsonl",
        [_live_ok_row(created_at="2026-07-28T12:00:00+00:00", order_id="1", stake=20.0)],
    )
    snap = build_snapshot(root=root, window=resolve_window(report_type="DAILY_CLOSED", report_date=date(2026, 7, 28)))
    assert any(str(e.get("alert_id")).startswith("stake_mismatch") for e in snap["exceptions"])


def test_p0_void_separated_and_roi_resolved():
    orders = {"1": {"stake": 10.0}, "2": {"stake": 10.0}}
    pnl = {"1": 5.0, "2": 0.0}
    out = compute_settlement_and_performance(orders=orders, pnl_by_oid=pnl, open_oids=set())
    assert out["n_void_push"] == 1
    assert out["stake_void"] == 10.0
    assert out["stake_resolved_total"] == 20.0
    assert out["principal_metric"] == "roi_resolved"
    assert out["roi_resolved"]["value"] == pytest.approx(0.25)
    assert out["roi_decided_ex_void"]["value"] == pytest.approx(0.5)


def test_p0_health_four_dimensions():
    health = build_health_model(
        manifest={"executor_live": {"status": "HEALTHY"}, "accounting_health": {"status": "HEALTHY"}},
        settlement={"n_open": 1, "n_settled": 0, "n_void_push": 0, "n_missing_accounting": 0, "live_ok_total": 5},
        clv={"collection_status": "WATCH", "performance": {"POST_5M": {"n": 2}}, "funnel": {}},
        latency={"detect_to_audit_overhead": {"status": "WATCH"}, "segments": {"ws_to_live_ok": {"n": 2}}, "ordering_violations": 0, "clock_skew": 0},
        config_eval={"policy": {"drift": "CURRENT_UNCHANGED"}, "risk_params": {"drift": "CURRENT_UNCHANGED"}},
    )
    assert "status" in health["report_health"]
    assert "status" in health["operations_health"]
    assert "status" in health["data_quality"]
    assert health["statistical_readiness"]["status"] == "INSUFFICIENT_N"


def test_p0_config_unchanged_not_stale(tmp_path):
    p = tmp_path / "wf_policy_current.json"
    p.write_text(json.dumps({"policy_version": "H3BUP_vNext_20260629", "stake": 10}), encoding="utf-8")
    # backdate mtime
    import os
    import time

    old = time.time() - 30 * 86400
    os.utime(p, (old, old))
    ce = evaluate_config_file(p)
    assert ce["drift"] in {"CURRENT_UNCHANGED", "CURRENT_MATCHED"}
    assert ce["drift"] != "STALE"


def test_p0_config_drift_critical(tmp_path):
    # simulate by forcing drift key in derive_alerts
    alerts = derive_alerts(
        health={
            "report_health": {"status": "HEALTHY"},
            "operations_health": {"status": "HEALTHY"},
            "data_quality": {"status": "HEALTHY"},
            "statistical_readiness": {"status": "AVAILABLE"},
            "config": {"policy": {"drift": "CONFIG_DRIFT"}},
        },
        settlement={"n_open": 0, "n_missing_accounting": 0},
        clv={"collection_status": "HEALTHY", "funnel": {}},
        latency={"detect_to_audit_overhead": {"status": "AVAILABLE"}, "ordering_violations": 0, "clock_skew": 0},
    )
    assert any(a["alert_id"] == "CONFIG_DRIFT" and a["severity"] == "CRITICAL" for a in alerts)


def test_p0_watch_generates_alert():
    alerts = derive_alerts(
        health={
            "report_health": {"status": "HEALTHY"},
            "operations_health": {"status": "HEALTHY"},
            "data_quality": {"status": "WATCH"},
            "statistical_readiness": {"status": "INSUFFICIENT_N"},
            "config": {},
        },
        settlement={"n_open": 2, "n_missing_accounting": 0},
        clv={"collection_status": "WATCH", "funnel": {"retry_backlog": 3}},
        latency={"detect_to_audit_overhead": {"status": "WATCH"}, "ordering_violations": 1, "clock_skew": 0},
        parity_status="CUTOFF_ALIGNED",
    )
    ids = {a["alert_id"] for a in alerts}
    assert "E2E_OVERHEAD_WATCH" in ids
    assert "CLV_BACKLOG" in ids
    assert "SETTLEMENT_PARTIAL" in ids
    assert "TRACE_ORDERING_VIOLATIONS" in ids


def test_p0_no_nenhum_when_alerts(tmp_path):
    root = tmp_path
    (root / "logs").mkdir()
    (root / "logs" / "executor_live.jsonl").write_text("")
    (root / "logs" / "h3bup_clv_health.json").write_text(json.dumps({"status": "WATCH", "live_ok_after_activation": 1, "retry_backlog": 2}))
    snap = build_snapshot(root=root, window=resolve_window(report_type="DAILY_CLOSED", report_date=date(2026, 7, 28)))
    md = render_markdown(snap)
    assert snap["exceptions"]
    assert "- nenhum" not in md.split("## 9)")[1].split("## 10)")[0]


def test_p0_roiw_v1_out_of_summary(tmp_path):
    root = tmp_path
    (root / "logs").mkdir()
    (root / "logs" / "executor_live.jsonl").write_text("")
    snap = build_snapshot(root=root, window=resolve_window(report_type="DAILY_CLOSED", report_date=date(2026, 7, 28)))
    md = render_markdown(snap)
    summary = md.split("## 1)")[1].split("## 2)")[0]
    assert "ROIw Total v1" not in summary or "apêndice" in summary.lower()
    assert "ROIw Total v1" in md  # still in appendix


def test_p0_formatters():
    assert "US$" in fmt_money(-0.4300000000000015)
    assert "0,43" in fmt_money(-0.4300000000000015)
    assert fmt_age(208.580494) == "3m29s"
    assert "29/07/2026" in fmt_ts("2026-07-29T22:07:27.963644+00:00")
    assert fmt_pct(0.1234) == "12.34%"


def test_p0_clv_means(tmp_path):
    root = tmp_path
    (root / "logs").mkdir()
    (root / "logs" / "h3bup_clv_health.json").write_text(
        json.dumps(
            {
                "status": "WATCH",
                "post_5m_expected": 2,
                "post_5m_attempted": 2,
                "post_5m_valid_strict": 2,
                "post_15m_expected": 0,
                "post_15m_attempted": 0,
                "post_15m_valid_strict": 0,
                "closing_expected": 0,
                "closing_attempted": 0,
                "closing_valid_strict": 0,
                "retry_backlog": 0,
            }
        )
    )
    _write_jsonl(
        root / "logs" / "h3bup_clv_snapshots.jsonl",
        [
            {"window_name": "POST_5M", "quality_status": "VALID_STRICT", "clv_raw_pct": 1.0, "snapshot_distance_sec": 10},
            {"window_name": "POST_5M", "quality_status": "VALID_STRICT", "clv_raw_pct": -1.0, "snapshot_distance_sec": 20},
            {"window_name": "POST_5M", "quality_status": "MISSING", "clv_raw_pct": 99.0},
        ],
    )
    clv = build_clv_section(root)
    p5 = clv["performance"]["POST_5M"]
    assert p5["n"] == 2
    assert p5["clv_mean_pct"] == pytest.approx(0.0)
    assert p5["clv_median_pct"] == pytest.approx(0.0)
    assert p5["positive_pct"] == pytest.approx(50.0)
    assert clv["fair_edge"]["status"] == "NOT_IMPLEMENTED"
    assert clv["windows"][0]["coverage_pct"] == pytest.approx(100.0)


def test_p0_e2e_n0_no_percentiles():
    m = metric_envelope(value=None, n=0, status="INSUFFICIENT_N", unit="ms")
    assert m["value"] is None
    assert m["n"] == 0


def test_p0_previous_diff_and_parity_separated():
    prev = {
        "run_id": "aaa",
        "execution_funnel": {"live_ok": {"value": 2}},
        "settlement": {"n_open": 1, "n_settled": 1, "n_void_push": 0, "n_missing_accounting": 0, "stake_resolved_total": 10, "pnl_resolved": 1},
        "performance": {"roi_resolved": {"value": 0.1}},
        "clv": {"performance": {"POST_5M": {"n": 1}}, "funnel": {"retry_backlog": 0}},
        "exceptions": [{"alert_id": "OLD"}],
    }
    cur = {
        "run_id": "bbb",
        "execution_funnel": {"live_ok": {"value": 3}},
        "settlement": {"n_open": 0, "n_settled": 3, "n_void_push": 0, "n_missing_accounting": 0, "stake_resolved_total": 30, "pnl_resolved": 3},
        "performance": {"roi_resolved": {"value": 0.1}},
        "clv": {"performance": {"POST_5M": {"n": 2}, "POST_15M": {"n": 0}, "CLOSING": {"n": 0}}, "funnel": {"retry_backlog": 1}},
        "exceptions": [{"alert_id": "NEW"}],
    }
    d = diff_snapshots(prev, cur)
    assert d["rows"]
    assert "NEW" in d["new_alerts"]
    assert "OLD" in d["resolved_alerts"]
    rows = compare_snapshots(v2=cur, v1_metrics={"live_ok": 3})
    assert any(r["metric"] == "LIVE_OK" for r in rows)
    assert any(r.get("status") for r in rows)


def test_p0_v2_preview_label_and_official_guard():
    from ops.daily_v2.preview_labels import PREVIEW_BANNER

    assert "PREVIEW" in PREVIEW_BANNER
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
        "operations_health": {"status": "HEALTHY"},
        "data_quality": {"status": "WATCH"},
        "statistical_readiness": {"status": "INSUFFICIENT_N"},
        "config": {"policy": {"file_status": "OK", "runtime_status": "UNVERIFIED", "fingerprint": "abc", "drift": "CURRENT_UNCHANGED"}},
        "execution_funnel": {
            "live_ok": metric_envelope(value=0, status="AVAILABLE", unit="count"),
            "stages": [{"step": "LIVE_OK", "event": "LIVE_OK", "n": 0, "pct_prev": None, "pct_initial": None, "status": "AVAILABLE"}],
            "block_reasons": [{"reason": "CAP_BLOCKED", "n": 1, "pct_requests": 10.0}],
            "fast_buckets": {},
        },
        "settlement": {
            "maturity_status": "FULLY_SETTLED",
            "n_open": 0,
            "n_settled": 0,
            "n_void_push": 0,
            "n_missing_accounting": 0,
            "stake_placed": 0,
            "stake_resolved_total": 0,
            "stake_void": 0,
            "pnl_resolved": 0,
        },
        "performance": {
            "roi_resolved": metric_envelope(value=0, status="AVAILABLE", unit="fraction"),
            "roi_settled": metric_envelope(value=0, status="AVAILABLE", unit="fraction"),
            "roiw_total_v1": metric_envelope(value=0, status="AVAILABLE", unit="percent"),
            "roiw_total_v2": metric_envelope(value=0, status="AVAILABLE", unit="percent"),
            "principal_metric": "roi_resolved",
        },
        "latency": {
            "daily_fast_le_6s": metric_envelope(value=0, status="AVAILABLE"),
            "study_fast_lt_4s": metric_envelope(value=0, status="AVAILABLE"),
            "pre_submit_ms_na": metric_envelope(value=0, status="AVAILABLE"),
            "e2e_ws_to_live_ok": metric_envelope(status="INSUFFICIENT_N"),
            "detect_to_audit_overhead": metric_envelope(status="WATCH"),
            "segments": {},
        },
        "e2e": {"n_traces": 0, "n_live_ok": 0, "segments": {}, "ordering_violations": 0, "clock_skew": 0},
        "clv": {
            "fair_edge": metric_envelope(status="NOT_IMPLEMENTED"),
            "funnel": {},
            "windows": [],
            "performance_rows": [],
            "collection_status": "WATCH",
        },
        "concentration": {"status": "INSUFFICIENT_N"},
        "exceptions": [{"alert_id": "E2E_OVERHEAD_WATCH", "severity": "WATCH", "status": "OPEN", "message": "overhead"}],
        "methodology": {},
        "parity": {"parity_status": "CUTOFF_ALIGNED"},
        "previous_diff": {"rows": [], "new_alerts": [], "resolved_alerts": []},
        "safety": {"alters_execution": False, "alters_policy": False, "alters_stake": False, "creates_orders": False, "opens_betslips": False},
    }
    md = render_markdown(snap)
    assert "DAILY V2 — PREVIEW / NÃO OFICIAL" in md
    assert "CAP_BLOCKED" in md
    assert "REPORT_HEALTH" in md
    assert "Funil operacional" in md or "Funil" in md
    assert snap["safety"]["creates_orders"] is False


def test_p0_v1_summary_helper_imports():
    from ops.daily_v2.v1_h3bup_summary import render_h3bup_vnext_official_summary

    md = render_h3bup_vnext_official_summary(Path("."))
    assert "H3BUP_vNext — Resumo Oficial" in md
    assert "não representam necessariamente" in md
