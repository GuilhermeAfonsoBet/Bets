#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _load_env_file(path: Path) -> None:
    try:
        if not path.exists():
            return
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            if not k or k in os.environ:
                continue
            os.environ[k] = v.strip()
    except Exception:
        return


def _parse_iso_utc(s: Any) -> Optional[datetime]:
    try:
        if s is None:
            return None
        t = str(s).strip()
        if not t:
            return None
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _pctl(xs: List[float], p: float) -> Optional[float]:
    if not xs:
        return None
    ys = sorted(xs)
    k = (len(ys) - 1) * float(p)
    f = int(k)
    c = min(len(ys) - 1, f + 1)
    if f == c:
        return float(ys[f])
    return float(ys[f] + (k - f) * (ys[c] - ys[f]))


def _tail_jsonl(path: Path, *, max_bytes: int = 8_000_000, max_lines: int = 120_000) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            end = f.tell()
            size = min(int(max_bytes), end)
            f.seek(max(0, end - size))
            raw = f.read().decode("utf-8", errors="ignore")
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for ln in raw.splitlines()[-int(max_lines):]:
        ln = ln.strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _run_health_monitor(window_minutes: int) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "ops.health_monitor",
        "--since-minutes",
        str(int(window_minutes)),
        "--autopilot",
    ]
    try:
        p = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return {
            "cmd": " ".join(cmd),
            "exit_code": int(p.returncode),
            "output_tail": "\n".join((p.stdout or "").splitlines()[-120:]),
        }
    except Exception as e:
        return {"cmd": " ".join(cmd), "exit_code": 99, "output_tail": f"health_monitor_exec_error: {e}"}


@dataclass
class WindowMetrics:
    hours: float
    nonhb_n: int
    live_ok_n: int
    fail_n: int
    api_failed_n: int
    no_session_n: int
    internal_error_n: int
    call_p50_ms: Optional[float]
    call_p90_ms: Optional[float]
    post_p50_ms: Optional[float]
    queue_p50_ms: Optional[float]
    live_ok_age_sec: Optional[int]
    nonhb_age_sec: Optional[int]


def _collect_window_metrics(exec_jsonl: Path, *, hours: float) -> WindowMetrics:
    rows = _tail_jsonl(exec_jsonl)
    cut = _utcnow() - timedelta(hours=float(hours))

    nonhb_n = 0
    live_ok_n = 0
    api_failed_n = 0
    no_session_n = 0
    internal_error_n = 0
    call_ms: List[float] = []
    post_ms: List[float] = []
    queue_ms: List[float] = []
    last_live_ok: Optional[datetime] = None
    last_nonhb: Optional[datetime] = None

    for o in rows:
        req = o.get("request") if isinstance(o.get("request"), dict) else {}
        res = o.get("result") if isinstance(o.get("result"), dict) else {}
        st = str(res.get("status") or "UNKNOWN").strip()
        ts = _parse_iso_utc(res.get("created_at") or req.get("created_at"))
        if ts is None or ts < cut:
            continue

        if st != "HEARTBEAT":
            nonhb_n += 1
            if last_nonhb is None or ts > last_nonhb:
                last_nonhb = ts

        if st == "LIVE_OK":
            live_ok_n += 1
            if last_live_ok is None or ts > last_live_ok:
                last_live_ok = ts
            timing = res.get("timing") if isinstance(res.get("timing"), dict) else {}
            c = timing.get("call_to_done_ms")
            p = timing.get("post_ms")
            q = timing.get("queue_delay_ms")
            try:
                if c is not None and float(c) > 0:
                    call_ms.append(float(c))
            except Exception:
                pass
            try:
                if p is not None and float(p) > 0:
                    post_ms.append(float(p))
            except Exception:
                pass
            try:
                if q is not None and float(q) >= 0:
                    queue_ms.append(float(q))
            except Exception:
                pass
        elif st == "API_FAILED":
            api_failed_n += 1
        elif st == "NO_SESSION":
            no_session_n += 1
        elif st == "INTERNAL_ERROR":
            internal_error_n += 1

    fail_n = int(api_failed_n + no_session_n + internal_error_n)
    now = _utcnow()
    live_ok_age_sec = int((now - last_live_ok).total_seconds()) if last_live_ok else None
    nonhb_age_sec = int((now - last_nonhb).total_seconds()) if last_nonhb else None
    return WindowMetrics(
        hours=float(hours),
        nonhb_n=int(nonhb_n),
        live_ok_n=int(live_ok_n),
        fail_n=int(fail_n),
        api_failed_n=int(api_failed_n),
        no_session_n=int(no_session_n),
        internal_error_n=int(internal_error_n),
        call_p50_ms=_pctl(call_ms, 0.50),
        call_p90_ms=_pctl(call_ms, 0.90),
        post_p50_ms=_pctl(post_ms, 0.50),
        queue_p50_ms=_pctl(queue_ms, 0.50),
        live_ok_age_sec=live_ok_age_sec,
        nonhb_age_sec=nonhb_age_sec,
    )


def _load_state(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _evaluate(metrics: WindowMetrics, *, p50_target_ms: int, p90_target_ms: int, max_fail_rate_pct: float, min_nonhb: int) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    c1 = (metrics.live_ok_n > 0) if metrics.nonhb_n >= int(min_nonhb) else True
    checks.append(
        {
            "name": "live_ok_presence",
            "ok": bool(c1),
            "detail": f"live_ok_n={metrics.live_ok_n} nonhb_n={metrics.nonhb_n} min_nonhb={min_nonhb}",
        }
    )

    c2 = True
    if metrics.call_p50_ms is not None:
        c2 = bool(float(metrics.call_p50_ms) < float(p50_target_ms))
    checks.append(
        {
            "name": "latency_p50_target",
            "ok": bool(c2),
            "detail": f"call_p50_ms={metrics.call_p50_ms} target<{p50_target_ms}",
        }
    )

    c3 = True
    if metrics.call_p90_ms is not None:
        c3 = bool(float(metrics.call_p90_ms) < float(p90_target_ms))
    checks.append(
        {
            "name": "latency_p90_target",
            "ok": bool(c3),
            "detail": f"call_p90_ms={metrics.call_p90_ms} target<{p90_target_ms}",
        }
    )

    fail_rate = (float(metrics.fail_n) / float(metrics.nonhb_n) * 100.0) if metrics.nonhb_n > 0 else 0.0
    c4 = bool(fail_rate <= float(max_fail_rate_pct))
    checks.append(
        {
            "name": "fail_rate_control",
            "ok": bool(c4),
            "detail": f"fail_rate_pct={fail_rate:.2f} max={max_fail_rate_pct:.2f}",
        }
    )

    overall_ok = all(bool(c.get("ok")) for c in checks)
    return {"overall_ok": bool(overall_ok), "checks": checks, "fail_rate_pct": fail_rate}


def main() -> int:
    ap = argparse.ArgumentParser(description="Checagem D+1 de estabilidade operacional (30m/2h/6h).")
    ap.add_argument("--env-file", default=os.getenv("ENV_FILE", ".env"))
    ap.add_argument("--executor-jsonl", default=os.getenv("EXECUTOR_JSONL", "logs/executor_live.jsonl"))
    ap.add_argument("--state-file", default=os.getenv("OPS_AUTOPILOT_STATE_FILE", "logs/ops_autopilot_state.json"))
    ap.add_argument("--out", default="logs/stability_d1_check.json")
    ap.add_argument("--run-health-monitor", action="store_true", help="Executa health_monitor --autopilot para 30m/120m/360m.")
    ap.add_argument("--target-p50-ms", type=int, default=5000)
    ap.add_argument("--target-p90-ms", type=int, default=8000)
    ap.add_argument("--max-fail-rate-pct", type=float, default=25.0)
    ap.add_argument("--min-nonhb", type=int, default=20)
    args = ap.parse_args()

    _load_env_file(Path(str(args.env_file)))
    exec_path = Path(str(args.executor_jsonl)).expanduser()
    state_path = Path(str(args.state_file)).expanduser()

    windows = [0.5, 2.0, 6.0]
    out: Dict[str, Any] = {
        "ts_utc": _utcnow().isoformat(),
        "executor_jsonl": str(exec_path),
        "state_file": str(state_path),
        "targets": {
            "p50_ms": int(args.target_p50_ms),
            "p90_ms": int(args.target_p90_ms),
            "max_fail_rate_pct": float(args.max_fail_rate_pct),
            "min_nonhb": int(args.min_nonhb),
        },
        "windows": {},
        "state_snapshot": {},
    }

    hm_runs: Dict[str, Any] = {}
    if bool(args.run_health_monitor):
        for mins in (30, 120, 360):
            hm_runs[f"{mins}m"] = _run_health_monitor(int(mins))
    out["health_monitor_runs"] = hm_runs

    for h in windows:
        m = _collect_window_metrics(exec_path, hours=float(h))
        ev = _evaluate(
            m,
            p50_target_ms=int(args.target_p50_ms),
            p90_target_ms=int(args.target_p90_ms),
            max_fail_rate_pct=float(args.max_fail_rate_pct),
            min_nonhb=int(args.min_nonhb),
        )
        key = f"{int(h*60)}m"
        out["windows"][key] = {
            "metrics": {
                "nonhb_n": m.nonhb_n,
                "live_ok_n": m.live_ok_n,
                "fail_n": m.fail_n,
                "api_failed_n": m.api_failed_n,
                "no_session_n": m.no_session_n,
                "internal_error_n": m.internal_error_n,
                "call_p50_ms": m.call_p50_ms,
                "call_p90_ms": m.call_p90_ms,
                "post_p50_ms": m.post_p50_ms,
                "queue_p50_ms": m.queue_p50_ms,
                "live_ok_age_sec": m.live_ok_age_sec,
                "nonhb_age_sec": m.nonhb_age_sec,
            },
            "evaluation": ev,
        }

    st = _load_state(state_path)
    out["state_snapshot"] = {
        "last_overall_code": st.get("last_overall_code"),
        "last_overall_utc": st.get("last_overall_utc"),
        "latency_degrade": st.get("latency_degrade"),
        "last_bridge_seen_total": st.get("last_bridge_seen_total"),
        "last_bridge_executed_total": st.get("last_bridge_executed_total"),
        "last_bridge_audits_n": st.get("last_bridge_audits_n"),
    }

    out_path = Path(str(args.out)).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
