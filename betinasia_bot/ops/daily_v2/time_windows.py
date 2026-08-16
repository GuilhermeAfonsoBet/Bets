"""Time-window and cohort contracts for Daily V2.

Mandatory:
  execution_day_utc = civil date of created_at in UTC
  daily window = [day_start_utc, next_day_start_utc)

Accounting post date is NEVER the primary operational cohort key.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional, Tuple


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return ensure_utc(value)
    s = str(value).strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return ensure_utc(datetime.fromisoformat(s))
    except Exception:
        return None


def execution_day_utc(created_at: datetime) -> date:
    return ensure_utc(created_at).date()


def closed_day_window(day: date) -> Tuple[datetime, datetime]:
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start, end


def in_half_open(ts: datetime, start: datetime, end: datetime) -> bool:
    t = ensure_utc(ts)
    return start <= t < end


@dataclass(frozen=True)
class ReportWindow:
    report_type: str  # DAILY_CLOSED | INTRADAY
    report_date_utc: date
    window_start_utc: datetime
    window_end_utc: datetime
    report_cutoff_utc: datetime

    def to_dict(self) -> dict:
        return {
            "report_type": self.report_type,
            "report_date_utc": self.report_date_utc.isoformat(),
            "window_start_utc": self.window_start_utc.isoformat(),
            "window_end_utc": self.window_end_utc.isoformat(),
            "report_cutoff_utc": self.report_cutoff_utc.isoformat(),
        }


def resolve_window(
    *,
    now_utc: Optional[datetime] = None,
    report_date: Optional[date] = None,
    report_type: str = "DAILY_CLOSED",
    cutoff_utc: Optional[datetime] = None,
) -> ReportWindow:
    now = ensure_utc(now_utc or datetime.now(timezone.utc))
    if report_type == "DAILY_CLOSED":
        # Last complete UTC day (or explicit date).
        day = report_date or (now.date() - timedelta(days=1))
        start, end = closed_day_window(day)
        cutoff = ensure_utc(cutoff_utc or now)
        return ReportWindow("DAILY_CLOSED", day, start, end, cutoff)
    if report_type == "INTRADAY":
        day = report_date or now.date()
        start, _ = closed_day_window(day)
        cutoff = ensure_utc(cutoff_utc or now)
        end = cutoff  # partial up to cutoff
        return ReportWindow("INTRADAY", day, start, end, cutoff)
    raise ValueError(f"unknown report_type={report_type}")
