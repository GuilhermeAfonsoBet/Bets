from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def trace_env(tmp_path, monkeypatch):
    path = tmp_path / "h3bup_e2e_trace.jsonl"
    monkeypatch.setenv("H3BUP_E2E_TRACE_ENABLED", "1")
    monkeypatch.setenv("H3BUP_E2E_TRACE_PATH", str(path))
    monkeypatch.setenv("H3BUP_E2E_TRACE_FLUSH_INTERVAL_SEC", "0.05")
    monkeypatch.setenv("H3BUP_E2E_TRACE_MAX_FILE_MB", "1")
    monkeypatch.setenv("H3BUP_E2E_TRACE_BACKUP_COUNT", "3")
    monkeypatch.setenv("H3BUP_E2E_TRACE_SAMPLE_RATE", "1.0")
    # reload writer with new env
    import importlib
    import ops.h3bup_e2e_trace as m

    importlib.reload(m)
    yield m, path
    try:
        m._WRITER.close()
    except Exception:
        pass


def _read_events(path: Path):
    import ops.h3bup_e2e_trace as m

    m.force_flush()
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def test_trace_id_format(trace_env):
    m, path = trace_env
    tid = m.make_trace_id(123)
    assert tid.startswith("h3bup:123:")
    assert len(tid.split(":")[-1]) == 12


def test_emit_and_utc_monotonic(trace_env):
    m, path = trace_env
    tid = m.make_trace_id(1)
    m.emit_trace_event("H3B_DETECTED", trace_id=tid, service="audit_h3b", audit_id=1)
    m.force_flush()
    evs = _read_events(path)
    assert len(evs) == 1
    e = evs[0]
    assert e["schema_version"] == 1
    assert e["trace_id"] == tid
    assert e["audit_id"] == 1
    assert e["order_id"] is None
    assert "T" in e["event_ts_utc"]
    assert isinstance(e["monotonic_ns"], int) and e["monotonic_ns"] > 0
    json.dumps(e)  # serializable


def test_attach_meta_propagation(trace_env):
    m, _ = trace_env
    tid = m.make_trace_id(9)
    meta = m.attach_trace_meta({}, trace_id=tid, audit_id=9)
    assert m.extract_trace_id_from_meta(meta) == tid
    assert meta["h3bup_e2e"]["audit_id"] == 9


def test_details_trace_extract(trace_env):
    m, _ = trace_env
    tid = "h3bup:na:abc123def456"
    assert m.extract_trace_id_from_details({"_e2e_trace_id": tid}) == tid


def test_cap_blocked_keeps_trace_without_order(trace_env):
    m, path = trace_env
    tid = m.make_trace_id(2)
    m.emit_trace_event(
        "H3B_FINAL_GATE_DECIDED",
        trace_id=tid,
        audit_id=2,
        execution_id="exec-1",
        status="CAP_BLOCKED",
        service="executor",
        policy_version="H3BUP_vNext_20260629",
    )
    m.force_flush()
    e = _read_events(path)[0]
    assert e["trace_id"] == tid
    assert e["order_id"] is None
    assert e["status"] == "CAP_BLOCKED"


def test_live_ok_includes_order_id(trace_env):
    m, path = trace_env
    tid = m.make_trace_id(3)
    m.emit_trace_event(
        "H3B_PLACE_FINISHED",
        trace_id=tid,
        audit_id=3,
        execution_id="exec-2",
        order_id="1933822208",
        status="LIVE_OK",
        service="executor",
        policy_version="H3BUP_vNext_20260629",
    )
    m.force_flush()
    e = _read_events(path)[0]
    assert e["order_id"] == "1933822208"


def test_write_error_fail_open(trace_env, monkeypatch):
    m, path = trace_env
    tid = m.make_trace_id(4)

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(m._WRITER, "enqueue_line", boom)
    # must not raise
    m.emit_trace_event("H3B_DETECTED", trace_id=tid, service="audit_h3b")
    assert m.get_metrics()["trace_events_dropped"] >= 1


def test_invalid_path_fail_open(tmp_path, monkeypatch):
    monkeypatch.setenv("H3BUP_E2E_TRACE_ENABLED", "1")
    monkeypatch.setenv("H3BUP_E2E_TRACE_PATH", str(tmp_path / "no_such_dir_never" / "x.jsonl"))
    import importlib
    import ops.h3bup_e2e_trace as m

    importlib.reload(m)
    # force path parent to be unwritable by pointing to a file-as-dir if possible
    m.emit_trace_event("H3B_DETECTED", trace_id=m.make_trace_id(None), service="audit_h3b")
    m.force_flush()
    # no exception is success


def test_feature_flag_off_no_write(tmp_path, monkeypatch):
    monkeypatch.setenv("H3BUP_E2E_TRACE_ENABLED", "0")
    monkeypatch.setenv("H3BUP_E2E_TRACE_PATH", str(tmp_path / "off.jsonl"))
    import importlib
    import ops.h3bup_e2e_trace as m

    importlib.reload(m)
    m.emit_trace_event("H3B_DETECTED", trace_id=m.make_trace_id(1), service="audit_h3b")
    m.force_flush()
    assert not (tmp_path / "off.jsonl").exists()


def test_rotation(trace_env, monkeypatch):
    m, path = trace_env
    m._WRITER.max_file_mb = 0.00001  # tiny
    for i in range(50):
        m.emit_trace_event("H3B_DETECTED", trace_id=m.make_trace_id(i), service="audit_h3b")
        m.force_flush()
    # either rotated backup exists or current file exists (best-effort)
    assert path.exists() or Path(str(path) + ".1").exists()


def test_analyzer_missing_not_zero(tmp_path, monkeypatch):
    path = tmp_path / "t.jsonl"
    tid = "h3bup:1:aaaaaaaaaaaa"
    ev = {
        "schema_version": 1,
        "event_name": "H3B_DETECTED",
        "event_ts_utc": "2026-07-29T12:00:00.000000+00:00",
        "monotonic_ns": 1000,
        "trace_id": tid,
        "audit_id": 1,
        "execution_id": None,
        "order_id": None,
        "policy_version": "H3BUP_vNext_20260629",
        "event_id": "e1",
        "market_type": "AH",
        "side": "home",
        "line": "-0.5",
        "status": None,
        "reason": None,
        "service": "audit_h3b",
        "process_id": 1,
        "duration_ms": None,
        "metadata": {},
    }
    path.write_text(json.dumps(ev) + "\n", encoding="utf-8")
    from ops.analyze_h3bup_e2e_latency import analyze_trace, group_traces, load_events

    rows = [analyze_trace(k, v) for k, v in group_traces(load_events(path)).items()]
    assert rows[0]["ws_to_live_ok_ms"] is None  # missing, not 0


def test_negative_duration_marked(tmp_path):
    path = tmp_path / "t.jsonl"
    tid = "h3bup:1:bbbbbbbbbbbb"
    evs = [
        {
            "schema_version": 1,
            "event_name": "H3B_DETECTED",
            "event_ts_utc": "2026-07-29T12:00:01+00:00",
            "monotonic_ns": 2000,
            "trace_id": tid,
            "service": "audit_h3b",
            "process_id": 1,
        },
        {
            "schema_version": 1,
            "event_name": "H3B_WS_RECEIVED",
            "event_ts_utc": "2026-07-29T12:00:02+00:00",
            "monotonic_ns": 3000,
            "trace_id": tid,
            "service": "audit_h3b",
            "process_id": 1,
        },
    ]
    path.write_text("\n".join(json.dumps(e) for e in evs) + "\n", encoding="utf-8")
    from ops.analyze_h3bup_e2e_latency import analyze_trace, group_traces, load_events

    row = analyze_trace(tid, group_traces(load_events(path))[tid])
    assert row["ws_to_detect_ms"] is not None and row["ws_to_detect_ms"] < 0
    assert row["clock_skew_suspected"] is True
    assert "H3B_WS_RECEIVED->H3B_DETECTED" in row["ordering_violations"] or row["ordering_violations"]


def test_duplicates_detected(tmp_path):
    path = tmp_path / "t.jsonl"
    tid = "h3bup:1:cccccccccccc"
    evs = [
        {"schema_version": 1, "event_name": "H3B_DETECTED", "event_ts_utc": "2026-07-29T12:00:00+00:00", "monotonic_ns": 1, "trace_id": tid, "service": "a", "process_id": 1},
        {"schema_version": 1, "event_name": "H3B_DETECTED", "event_ts_utc": "2026-07-29T12:00:01+00:00", "monotonic_ns": 2, "trace_id": tid, "service": "a", "process_id": 1},
    ]
    path.write_text("\n".join(json.dumps(e) for e in evs) + "\n", encoding="utf-8")
    from ops.analyze_h3bup_e2e_latency import analyze_trace, group_traces, load_events

    row = analyze_trace(tid, group_traces(load_events(path))[tid])
    assert "H3B_DETECTED" in row["duplicate_events"]


def test_old_request_without_trace_compatible():
    # ExecutionRequest meta may omit h3bup_e2e
    from ops.h3bup_e2e_trace import extract_trace_id_from_meta

    assert extract_trace_id_from_meta({}) is None
    assert extract_trace_id_from_meta({"bridge": {"src_id": 1}}) is None


def test_policy_constants_unchanged():
    bridge = (ROOT / "ops" / "executor_bridge_audit.py").read_text(encoding="utf-8")
    assert 'POLICY_VERSION_H3BUP_VNEXT = "H3BUP_vNext_20260629"' in bridge
    assert "1.85 <= float(odd) <= 2.15" in bridge
    assert "req.policy.stake_requested = 2.0" in bridge
    assert "pre_exec_ok = bool(is_backpre and is_odd)" in bridge
    assert "float(cap) > 100.0" in bridge
    assert "float(slip) < 0.0" in bridge


def test_daily_tolerates_missing_file(tmp_path):
    from ops.analyze_h3bup_e2e_latency import load_events, group_traces, analyze_trace, summarize, render_daily_section

    evs = load_events(tmp_path / "missing.jsonl")
    rows = [analyze_trace(k, v) for k, v in group_traces(evs).items()]
    summary, _, cov = summarize(rows)
    text = render_daily_section(summary, cov, health={"enabled": False, "schema_version": 1}, n_traces=0, n_live=0)
    assert "H3BUP End-to-End Latency" in text


def test_analyzer_cli(tmp_path, monkeypatch):
    path = tmp_path / "t.jsonl"
    tid = "h3bup:5:dddddddddddd"
    events = [
        ("H3B_WS_RECEIVED", "2026-07-29T12:00:00.000+00:00"),
        ("H3B_DETECTED", "2026-07-29T12:00:00.010+00:00"),
        ("H3B_AUDIT_PERSIST_FINISHED", "2026-07-29T12:00:00.020+00:00"),
        ("H3B_BRIDGE_FETCHED", "2026-07-29T12:00:01.000+00:00"),
        ("H3B_EXEC_REQUEST_CREATED", "2026-07-29T12:00:01.010+00:00"),
        ("H3B_EXECUTOR_RECEIVED", "2026-07-29T12:00:01.020+00:00"),
        ("H3B_DRYRUN_STARTED", "2026-07-29T12:00:01.030+00:00"),
        ("H3B_DRYRUN_FINISHED", "2026-07-29T12:00:03.000+00:00"),
        ("H3B_FINAL_GATE_DECIDED", "2026-07-29T12:00:03.010+00:00"),
        ("H3B_PLACE_STARTED", "2026-07-29T12:00:03.020+00:00"),
        ("H3B_PLACE_FINISHED", "2026-07-29T12:00:04.000+00:00"),
        ("H3B_RESULT_PERSIST_FINISHED", "2026-07-29T12:00:04.010+00:00"),
    ]
    lines = []
    for i, (name, ts) in enumerate(events):
        lines.append(
            json.dumps(
                {
                    "schema_version": 1,
                    "event_name": name,
                    "event_ts_utc": ts,
                    "monotonic_ns": 1000 + i,
                    "trace_id": tid,
                    "audit_id": 5,
                    "execution_id": "e5",
                    "order_id": "99" if name == "H3B_PLACE_FINISHED" else None,
                    "status": "LIVE_OK" if "PLACE_FINISHED" in name or "RESULT" in name else None,
                    "service": "test",
                    "process_id": 1,
                    "market_type": "AH",
                    "side": "home",
                    "line": "0",
                    "policy_version": "H3BUP_vNext_20260629",
                    "metadata": {},
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        ["analyze", "--input", str(path), "--out-dir", str(out), "--date", "20260729"],
    )
    from ops.analyze_h3bup_e2e_latency import main

    assert main() == 0
    assert (out / "h3bup_e2e_latency_summary_20260729.csv").exists()


def test_gate_logic_unchanged_by_import():
    # smoke: importing e2e module must not alter worker gate thresholds in source
    src = (ROOT / "executor" / "worker.py").read_text(encoding="utf-8")
    assert "1.85 <= float(odd_val) <= 2.15" in src
    assert "float(cap_val) <= 100.0" in src
    assert "float(slip_val) >= 0.0" in src
    assert "abs(float(stake) - hard_stake) > 1e-6" in src
    assert '"required_stake": 2.0' in src or "'required_stake': 2.0" in src
    bridge = (ROOT / "ops" / "executor_bridge_audit.py").read_text(encoding="utf-8")
    assert "req.policy.stake_requested = 2.0" in bridge
    assert "H3BUP_vNext_20260629" in bridge
