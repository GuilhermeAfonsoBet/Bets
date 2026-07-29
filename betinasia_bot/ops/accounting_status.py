"""Pure helpers for accounting monitor status / freshness (no I/O side effects)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple


ACCOUNTING_OK = "ACCOUNTING_OK"
ACCOUNTING_PARTIAL = "ACCOUNTING_PARTIAL"
ACCOUNTING_AUTH_FAILED = "ACCOUNTING_AUTH_FAILED"
ACCOUNTING_PROXY_FAILED = "ACCOUNTING_PROXY_FAILED"
ACCOUNTING_API_FAILED = "ACCOUNTING_API_FAILED"
ACCOUNTING_RATE_LIMIT = "ACCOUNTING_RATE_LIMIT"
ACCOUNTING_TIMEOUT = "ACCOUNTING_TIMEOUT"
ACCOUNTING_SCHEMA_CHANGED = "ACCOUNTING_SCHEMA_CHANGED"
ACCOUNTING_PARSE_FAILED = "ACCOUNTING_PARSE_FAILED"
ACCOUNTING_WRITE_FAILED = "ACCOUNTING_WRITE_FAILED"
ACCOUNTING_EMPTY_RESPONSE = "ACCOUNTING_EMPTY_RESPONSE"
ACCOUNTING_BROWSER_DEAD = "ACCOUNTING_BROWSER_DEAD"
ACCOUNTING_UNKNOWN_FAILURE = "ACCOUNTING_UNKNOWN_FAILURE"

HEALTHY = "HEALTHY"
WATCH = "WATCH"
CRITICAL = "CRITICAL"

REQUIRED_CSV_COLS = ("order id", "amount", "type", "post date")


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return int(default)


def normalize_jsonl_path(raw: str) -> str:
    """
    Correct known typo ACCOUNTING_JSONL=logs/accounring_snapshots.jsonl
    without touching shared .env; call-site may override via systemd.
    """
    s = (raw or "").strip() or "logs/accounting_snapshots.jsonl"
    if os.getenv("ACCOUNTING_FIX_JSONL_TYPO", "1").strip() in ("0", "false", "False", "no", "NO"):
        return s
    return s.replace("accounring_snapshots", "accounting_snapshots")


def classify_exception(exc: BaseException | str) -> str:
    msg = str(exc or "")
    low = msg.lower()
    if any(
        x in low
        for x in (
            "login_failed",
            "auth required",
            "unauthorized",
            "401",
            "no_session",
            "sessão inválida",
            "session expired",
            "sessão expirada",
            "api auth",
        )
    ):
        return ACCOUNTING_AUTH_FAILED
    if any(x in low for x in ("proxy", "err_proxy", "tunnel", "econnreset via proxy")):
        return ACCOUNTING_PROXY_FAILED
    if any(x in low for x in ("429", "rate limit", "too many requests", "api_backoff")):
        return ACCOUNTING_RATE_LIMIT
    if any(x in low for x in ("timeout", "timed out", "exceeded while waiting")):
        return ACCOUNTING_TIMEOUT
    if any(x in low for x in ("pipe closed", "target closed", "browser has been closed", "page.goto", "page is closed", "connection closed")):
        return ACCOUNTING_BROWSER_DEAD
    if any(x in low for x in ("schema", "missing column", "fieldnames")):
        return ACCOUNTING_SCHEMA_CHANGED
    if any(x in low for x in ("parse", "csv", "decode")):
        return ACCOUNTING_PARSE_FAILED
    if any(x in low for x in ("permission", "read-only", "errno 28", "no space", "disk", "write")):
        return ACCOUNTING_WRITE_FAILED
    if any(x in low for x in ("empty", "no body", "0 bytes")):
        return ACCOUNTING_EMPTY_RESPONSE
    if any(x in low for x in ("http", "api", "status")):
        return ACCOUNTING_API_FAILED
    return ACCOUNTING_UNKNOWN_FAILURE


def validate_csv_schema(cols: list[str] | None) -> Tuple[bool, Optional[str]]:
    have = {str(c or "").strip().lower() for c in (cols or [])}
    missing = [c for c in REQUIRED_CSV_COLS if c not in have]
    if missing:
        return False, f"missing columns: {missing}"
    return True, None


def cycle_status(*, balance_ok: bool, open_ok: bool, error_type: Optional[str] = None) -> str:
    """ACCOUNTING_OK only when both sources validated and written."""
    if error_type in (
        ACCOUNTING_AUTH_FAILED,
        ACCOUNTING_PROXY_FAILED,
        ACCOUNTING_RATE_LIMIT,
        ACCOUNTING_TIMEOUT,
        ACCOUNTING_SCHEMA_CHANGED,
        ACCOUNTING_PARSE_FAILED,
        ACCOUNTING_WRITE_FAILED,
        ACCOUNTING_EMPTY_RESPONSE,
        ACCOUNTING_BROWSER_DEAD,
        ACCOUNTING_API_FAILED,
    ) and not (balance_ok or open_ok):
        return error_type
    if balance_ok and open_ok:
        return ACCOUNTING_OK
    if balance_ok or open_ok:
        return ACCOUNTING_PARTIAL
    if error_type:
        return error_type
    return ACCOUNTING_EMPTY_RESPONSE


@dataclass
class FreshnessLimits:
    warn_stale_sec: float = 900.0
    critical_stale_sec: float = 3600.0
    max_consecutive_failures: int = 3

    @classmethod
    def from_env(cls) -> "FreshnessLimits":
        return cls(
            warn_stale_sec=env_float("ACCOUNTING_WARN_STALE_SEC", 900.0),
            critical_stale_sec=env_float("ACCOUNTING_CRITICAL_STALE_SEC", 3600.0),
            max_consecutive_failures=env_int("ACCOUNTING_MAX_CONSECUTIVE_FAILURES", 3),
        )


def classify_health(
    *,
    status: str,
    balance_age_sec: Optional[float],
    open_age_sec: Optional[float],
    consecutive_failures: int,
    limits: Optional[FreshnessLimits] = None,
) -> str:
    lim = limits or FreshnessLimits.from_env()
    ages = [a for a in (balance_age_sec, open_age_sec) if a is not None]
    max_age = max(ages) if ages else None

    if status in (
        ACCOUNTING_AUTH_FAILED,
        ACCOUNTING_SCHEMA_CHANGED,
        ACCOUNTING_PARSE_FAILED,
        ACCOUNTING_BROWSER_DEAD,
    ):
        return CRITICAL
    if status != ACCOUNTING_OK and status != ACCOUNTING_PARTIAL:
        # both files missing / hard failure
        if consecutive_failures >= lim.max_consecutive_failures:
            return CRITICAL
        return CRITICAL if status in (ACCOUNTING_EMPTY_RESPONSE, ACCOUNTING_WRITE_FAILED) else WATCH
    if max_age is not None and max_age > lim.critical_stale_sec:
        return CRITICAL
    if status == ACCOUNTING_PARTIAL:
        return WATCH
    if consecutive_failures > 0:
        return WATCH
    if max_age is not None and max_age > lim.warn_stale_sec:
        return WATCH
    if status == ACCOUNTING_OK:
        return HEALTHY
    return WATCH


def sanitize_error_message(msg: Any, *, limit: int = 400) -> str:
    s = str(msg or "")
    # strip obvious secrets
    s = re.sub(r"(?i)(password|passwd|token|cookie|authorization|proxy_password)\s*[:=]\s*\S+", r"\1=***", s)
    s = re.sub(r"(?i)//[^:@\s]+:[^@\s]+@", "//***:***@", s)
    return s[:limit]


def order_id_key(raw: Any) -> str:
    """Preserve large order ids as exact strings (never float)."""
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".", 1)[0]
    return s
