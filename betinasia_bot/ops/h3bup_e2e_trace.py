"""H3BUP end-to-end latency telemetry (append-only, fail-open).

Telemetry failures NEVER raise to callers and NEVER alter execution.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Optional


SCHEMA_VERSION_DEFAULT = 1
DEFAULT_PATH = "logs/h3bup_e2e_trace.jsonl"

_EVENT_NAMES = frozenset(
    {
        "H3B_WS_RECEIVED",
        "H3B_DETECTED",
        "H3B_AUDIT_ENQUEUED",
        "H3B_AUDIT_GATE_DECIDED",
        "H3B_AUDIT_PERSIST_STARTED",
        "H3B_AUDIT_PERSIST_FINISHED",
        "H3B_BRIDGE_FETCHED",
        "H3B_BRIDGE_SEEN_RESERVED",
        "H3B_POLICY_EVAL_STARTED",
        "H3B_POLICY_EVAL_FINISHED",
        "H3B_EXEC_REQUEST_CREATED",
        "H3B_EXEC_REQUEST_SENT",
        "H3B_EXECUTOR_RECEIVED",
        "H3B_DRYRUN_STARTED",
        "H3B_DRYRUN_FINISHED",
        "H3B_FINAL_GATE_DECIDED",
        "H3B_PLACE_STARTED",
        "H3B_PLACE_FINISHED",
        "H3B_RESULT_PERSIST_STARTED",
        "H3B_RESULT_PERSIST_FINISHED",
    }
)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return int(default)


def utc_now_iso() -> str:
    # microseconds for inter-process correlation
    return datetime.now(timezone.utc).isoformat()


def make_trace_id(audit_id: Any = None) -> str:
    """Stable unique id: h3bup:<audit_id|na>:<uuid12>."""
    aid = "na"
    try:
        if audit_id is not None and str(audit_id).strip() not in ("", "0", "None"):
            aid = str(int(audit_id)) if str(audit_id).isdigit() else str(audit_id).strip()[:32]
    except Exception:
        aid = "na"
    return f"h3bup:{aid}:{uuid.uuid4().hex[:12]}"


def derive_trace_id_from_audit(audit_id: Any) -> str:
    """Deterministic-ish pre-persist id (before DB id exists use provisional)."""
    return make_trace_id(audit_id)


class _TraceMetrics:
    __slots__ = (
        "trace_events_written",
        "trace_events_dropped",
        "trace_write_errors",
        "trace_serialization_errors",
        "trace_missing_trace_id",
        "trace_out_of_order_detected",
        "_lock",
    )

    def __init__(self) -> None:
        self.trace_events_written = 0
        self.trace_events_dropped = 0
        self.trace_write_errors = 0
        self.trace_serialization_errors = 0
        self.trace_missing_trace_id = 0
        self.trace_out_of_order_detected = 0
        self._lock = threading.Lock()

    def inc(self, name: str, n: int = 1) -> None:
        try:
            with self._lock:
                setattr(self, name, int(getattr(self, name, 0)) + int(n))
        except Exception:
            pass

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return {
                "trace_events_written": int(self.trace_events_written),
                "trace_events_dropped": int(self.trace_events_dropped),
                "trace_write_errors": int(self.trace_write_errors),
                "trace_serialization_errors": int(self.trace_serialization_errors),
                "trace_missing_trace_id": int(self.trace_missing_trace_id),
                "trace_out_of_order_detected": int(self.trace_out_of_order_detected),
            }


METRICS = _TraceMetrics()


class TraceWriter:
    def __init__(self) -> None:
        self.enabled = _env_bool("H3BUP_E2E_TRACE_ENABLED", False)
        self.path = Path(os.getenv("H3BUP_E2E_TRACE_PATH", DEFAULT_PATH))
        self.schema_version = _env_int("H3BUP_E2E_TRACE_SCHEMA_VERSION", SCHEMA_VERSION_DEFAULT)
        self.flush_interval_sec = max(0.05, _env_float("H3BUP_E2E_TRACE_FLUSH_INTERVAL_SEC", 1.0))
        self.max_file_mb = max(1.0, _env_float("H3BUP_E2E_TRACE_MAX_FILE_MB", 100.0))
        self.backup_count = max(1, _env_int("H3BUP_E2E_TRACE_BACKUP_COUNT", 10))
        self.sample_rate = min(1.0, max(0.0, _env_float("H3BUP_E2E_TRACE_SAMPLE_RATE", 1.0)))
        self.only_h3bup = _env_bool("H3BUP_E2E_TRACE_ONLY_H3BUP", True)
        self._q: Deque[str] = deque(maxlen=_env_int("H3BUP_E2E_TRACE_QUEUE_MAX", 20000))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_error_log = 0.0
        self._dropped_full = 0
        if self.enabled:
            self._start()

    def _start(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="h3bup-e2e-trace-writer", daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def reload_enabled(self) -> None:
        """Hot-ish: if flag flipped on, start writer."""
        now_en = _env_bool("H3BUP_E2E_TRACE_ENABLED", False)
        if now_en and not self.enabled:
            self.enabled = True
            self._start()
        self.enabled = now_en

    def close(self) -> None:
        try:
            self._stop.set()
            self._flush_once()
        except Exception:
            pass

    def _rotate_if_needed(self) -> None:
        try:
            if not self.path.exists():
                return
            max_bytes = int(self.max_file_mb * 1024 * 1024)
            if self.path.stat().st_size < max_bytes:
                return
            # rotate: path -> path.1 ... path.N
            for i in range(int(self.backup_count) - 1, 0, -1):
                src = Path(f"{self.path}.{i}")
                dst = Path(f"{self.path}.{i+1}")
                if src.exists():
                    try:
                        if dst.exists():
                            dst.unlink()
                        src.rename(dst)
                    except Exception:
                        pass
            bak = Path(f"{self.path}.1")
            try:
                if bak.exists():
                    bak.unlink()
                self.path.rename(bak)
            except Exception:
                pass
        except Exception:
            METRICS.inc("trace_write_errors")

    def _flush_once(self) -> None:
        batch: list[str] = []
        with self._lock:
            while self._q:
                batch.append(self._q.popleft())
        if not batch:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate_if_needed()
            with self.path.open("a", encoding="utf-8") as f:
                for line in batch:
                    f.write(line)
                    if not line.endswith("\n"):
                        f.write("\n")
            METRICS.inc("trace_events_written", len(batch))
        except Exception:
            METRICS.inc("trace_write_errors")
            METRICS.inc("trace_events_dropped", len(batch))
            now = time.time()
            if now - self._last_error_log > 30.0:
                self._last_error_log = now
                try:
                    from loguru import logger

                    logger.warning("[h3bup_e2e] trace write failed (fail-open)")
                except Exception:
                    pass

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._flush_once()
            except Exception:
                METRICS.inc("trace_write_errors")
            self._stop.wait(self.flush_interval_sec)
        try:
            self._flush_once()
        except Exception:
            pass

    def enqueue_line(self, line: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            if len(self._q) >= int(self._q.maxlen or 20000):
                self._dropped_full += 1
                METRICS.inc("trace_events_dropped")
                return
            self._q.append(line)


_WRITER = TraceWriter()


def get_metrics() -> Dict[str, int]:
    snap = METRICS.snapshot()
    snap["queue_dropped_full"] = int(getattr(_WRITER, "_dropped_full", 0))
    snap["enabled"] = 1 if _WRITER.enabled else 0
    return snap


def is_enabled() -> bool:
    try:
        _WRITER.reload_enabled()
    except Exception:
        pass
    return bool(_WRITER.enabled)


def _should_sample(trace_id: str) -> bool:
    rate = float(_WRITER.sample_rate)
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    try:
        h = abs(hash(trace_id)) % 10000
        return h < int(rate * 10000)
    except Exception:
        return True


def _policy_is_h3bup(policy_version: Any) -> bool:
    try:
        return "H3BUP_vNext" in str(policy_version or "")
    except Exception:
        return False


def emit_trace_event(
    event_name: str,
    *,
    trace_id: Optional[str] = None,
    audit_id: Any = None,
    execution_id: Any = None,
    order_id: Any = None,
    policy_version: Any = None,
    event_id: Any = None,
    market_type: Any = None,
    side: Any = None,
    line: Any = None,
    status: Any = None,
    reason: Any = None,
    service: str = "unknown",
    duration_ms: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
    event_ts_utc: Optional[str] = None,
) -> None:
    """Best-effort emit. Never raises."""
    try:
        if not is_enabled():
            return
        if event_name not in _EVENT_NAMES:
            # allow forward-compatible names but still write
            pass
        if _WRITER.only_h3bup and policy_version is not None and not _policy_is_h3bup(policy_version):
            # Still allow audit-side events without policy yet
            if str(service) not in ("audit_h3b", "audit-ws-gate-back"):
                return
        tid = str(trace_id or "").strip()
        if not tid:
            METRICS.inc("trace_missing_trace_id")
            tid = make_trace_id(audit_id)
        if not _should_sample(tid):
            METRICS.inc("trace_events_dropped")
            return
        payload = {
            "schema_version": int(_WRITER.schema_version),
            "event_name": str(event_name),
            "event_ts_utc": event_ts_utc or utc_now_iso(),
            "monotonic_ns": int(time.monotonic_ns()),
            "trace_id": tid,
            "audit_id": (None if audit_id in (None, "", 0, "0") else audit_id),
            "execution_id": (None if execution_id in (None, "") else str(execution_id)),
            "order_id": (None if order_id in (None, "") else str(order_id)),
            "policy_version": (None if policy_version in (None, "") else str(policy_version)),
            "event_id": (None if event_id in (None, "") else str(event_id)),
            "market_type": (None if market_type in (None, "") else str(market_type)),
            "side": (None if side in (None, "") else str(side)),
            "line": (None if line in (None, "") else str(line)),
            "status": (None if status in (None, "") else str(status)),
            "reason": (None if reason in (None, "") else str(reason)[:240]),
            "service": str(service),
            "process_id": int(os.getpid()),
            "duration_ms": (None if duration_ms is None else float(duration_ms)),
            "metadata": dict(metadata or {}),
        }
        try:
            line_s = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
        except Exception:
            METRICS.inc("trace_serialization_errors")
            METRICS.inc("trace_events_dropped")
            return
        _WRITER.enqueue_line(line_s)
    except Exception:
        try:
            METRICS.inc("trace_write_errors")
            METRICS.inc("trace_events_dropped")
        except Exception:
            pass


def extract_trace_id_from_meta(meta: Any) -> Optional[str]:
    try:
        if not isinstance(meta, dict):
            return None
        e2e = meta.get("h3bup_e2e") if isinstance(meta.get("h3bup_e2e"), dict) else meta
        tid = e2e.get("trace_id") if isinstance(e2e, dict) else None
        return str(tid) if tid else None
    except Exception:
        return None


def attach_trace_meta(meta: Optional[Dict[str, Any]], *, trace_id: str, **extra: Any) -> Dict[str, Any]:
    out = dict(meta or {})
    try:
        block = dict(out.get("h3bup_e2e") or {})
        block["trace_id"] = trace_id
        for k, v in extra.items():
            if v is not None:
                block[k] = v
        out["h3bup_e2e"] = block
    except Exception:
        pass
    return out


def extract_trace_id_from_details(details: Any) -> Optional[str]:
    """Read _e2e_trace_id from hypothesis_details (audit persist)."""
    try:
        if isinstance(details, str) and details.strip():
            details = json.loads(details)
        if not isinstance(details, dict):
            return None
        tid = details.get("_e2e_trace_id") or details.get("e2e_trace_id")
        if tid:
            return str(tid)
        e2e = details.get("h3bup_e2e")
        if isinstance(e2e, dict) and e2e.get("trace_id"):
            return str(e2e.get("trace_id"))
        return None
    except Exception:
        return None


def wall_ts_to_iso(ts: Any) -> Optional[str]:
    try:
        if ts is None:
            return None
        x = float(ts)
        if x <= 0:
            return None
        return datetime.fromtimestamp(x, tz=timezone.utc).isoformat()
    except Exception:
        return None


def map_exec_status(status: Any, error: Any = None) -> str:
    s = str(status or "").upper()
    if s in ("LIVE_OK", "CAP_BLOCKED", "API_FAILED", "STALE", "NO_SESSION", "TIMEOUT", "REJECTED"):
        return s
    if "LIVE_OK" in s:
        return "LIVE_OK"
    if "CAP_BLOCKED" in s or "GATE" in s:
        return "CAP_BLOCKED"
    if "STALE" in s:
        return "STALE"
    if "TIMEOUT" in s:
        return "TIMEOUT"
    if "DRY_OK" in s or s == "OK":
        return "OK"
    if "API" in s or "FAILED" in s:
        return "API_FAILED"
    err = str(error or "").upper()
    if "NO_SESSION" in err or "NO_ROOT_SESSION" in err:
        return "NO_SESSION"
    if s:
        return "UNKNOWN"
    return "UNKNOWN"


def force_flush() -> None:
    """Test helper: flush writer queue immediately."""
    try:
        _WRITER._flush_once()
    except Exception:
        pass
