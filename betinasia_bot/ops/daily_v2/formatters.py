"""P0 formatting helpers for human-readable Markdown/PDF (JSON keeps precision)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional


def fmt_money(x: Any, *, currency: str = "US$") -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except Exception:
        return "—"
    sign = "-" if v < 0 else ""
    return f"{sign}{currency} {abs(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_pct(x: Any, *, already_percent: bool = False, nd: int = 2) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except Exception:
        return "—"
    if not already_percent:
        v = v * 100.0
    return f"{v:.{nd}f}%"


def fmt_ms(x: Any, nd: int = 1) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):,.{nd}f} ms".replace(",", " ")
    except Exception:
        return "—"


def fmt_age(seconds: Any) -> str:
    if seconds is None:
        return "—"
    try:
        s = float(seconds)
    except Exception:
        return "—"
    if s < 0:
        return f"cutoff+{fmt_age(-s)}"
    if s < 60:
        return f"{int(round(s))}s"
    if s < 3600:
        m = int(s // 60)
        r = int(round(s % 60))
        return f"{m}m{r:02d}s"
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    return f"{h}h{m:02d}m"


def fmt_ts(iso: Any) -> str:
    if not iso:
        return "—"
    try:
        s = str(iso).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt.strftime("%d/%m/%Y %H:%M UTC")
    except Exception:
        return str(iso)[:19]


def fmt_int(x: Any) -> str:
    if x is None:
        return "—"
    try:
        return str(int(x))
    except Exception:
        return "—"


def metric_cell(m: Any, *, as_pct: bool = False, as_money: bool = False, as_ms: bool = False) -> str:
    if not isinstance(m, dict):
        return "—" if m is None else str(m)
    st = m.get("status")
    val = m.get("value")
    if val is None:
        return f"`{st or 'MISSING'}`"
    if as_money:
        body = fmt_money(val)
    elif as_pct or m.get("unit") in {"fraction", "percent"}:
        body = fmt_pct(val, already_percent=(m.get("unit") == "percent"))
    elif as_ms or m.get("unit") == "ms":
        body = fmt_ms(val)
    else:
        body = str(val)
    return f"{body} [{st}]"
