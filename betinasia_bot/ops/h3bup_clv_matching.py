"""Same-line strict matching + CLV raw formula (B808-compatible)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple


def normalize_side(side: Any) -> Optional[str]:
    if side is None:
        return None
    s = str(side).strip().lower()
    aliases = {
        "home": "home",
        "h": "home",
        "1": "home",
        "side_a": "home",
        "away": "away",
        "a": "away",
        "2": "away",
        "side_b": "away",
        "over": "over",
        "o": "over",
        "under": "under",
        "u": "under",
    }
    return aliases.get(s, s or None)


def normalize_period(period: Any) -> str:
    if period is None or str(period).strip() == "":
        return "full_time"
    p = str(period).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ft": "full_time",
        "full": "full_time",
        "fulltime": "full_time",
        "full_time": "full_time",
        "1h": "first_half",
        "first_half": "first_half",
        "fh": "first_half",
        "2h": "second_half",
        "second_half": "second_half",
    }
    return aliases.get(p, p)


def normalize_market(market: Any) -> Optional[str]:
    if market is None:
        return None
    m = str(market).strip().upper()
    if m in ("AH", "ASIAN", "ASIAN_HANDICAP", "HANDICAP"):
        return "AH"
    if m in ("OU", "O/U", "OVER_UNDER", "TOTALS"):
        return "OU"
    if m in ("1X2", "ML", "MONEYLINE"):
        return "1X2"
    return m or None


def normalize_line(line: Any, market_type: Any = "AH") -> Optional[str]:
    """Canonical line string after allowed normalizations only."""
    if line is None:
        return None
    raw = str(line).strip()
    if not raw:
        return None
    mt = normalize_market(market_type) or "AH"
    if mt == "OU":
        if raw.upper().startswith("OU_"):
            raw = raw[3:]
        try:
            v = float(raw.replace(",", "."))
        except Exception:
            return None
        # canonical without trailing .0 noise but keep .25/.5/.75
        if abs(v - round(v)) < 1e-9:
            return f"OU_{int(round(v))}.0"
        return f"OU_{v}"
    if mt == "1X2":
        return "1X2"
    # AH
    try:
        v = float(raw.replace(",", ".").replace("+", ""))
        if raw.strip().startswith("-"):
            v = -abs(v)
        elif "+" in str(line):
            v = abs(v)
        # preserve sign for nonzero; zero as 0.0
        if abs(v) < 1e-12:
            return "0.0"
        # format: drop useless trailing zeros but keep quarter lines
        s = f"{v:.4f}".rstrip("0").rstrip(".")
        if "." not in s:
            s = f"{s}.0"
        return s
    except Exception:
        return raw


def boh_ah_line(line: Any, market_type: Any = "AH") -> Optional[str]:
    """Form expected by best_odds_history.ah_line."""
    mt = normalize_market(market_type) or "AH"
    canon = normalize_line(line, mt)
    if canon is None:
        return None
    if mt == "AH":
        # BOH often stores without forced + prefix; try canonical as-is
        return canon
    return canon


def line_variants(line: Any, market_type: Any = "AH") -> Tuple[str, ...]:
    """Allowed string variants that represent THE SAME line (not different lines)."""
    mt = normalize_market(market_type) or "AH"
    canon = normalize_line(line, mt)
    if canon is None:
        return tuple()
    out = {canon}
    if mt == "AH":
        try:
            v = float(canon)
        except Exception:
            return tuple(out)
        forms = {canon, f"{v}", f"{v:.1f}", f"{v:.2f}", f"{v:.0f}"}
        if v > 0:
            forms |= {f"+{v}", f"+{v:.1f}", f"+{v:.2f}"}
        if abs(v - int(v)) < 1e-9:
            forms |= {str(int(v)), f"+{int(v)}" if v > 0 else str(int(v))}
        out |= {x for x in forms if x}
    elif mt == "OU":
        out.add(canon)
        if canon.startswith("OU_"):
            out.add(canon[3:])
    return tuple(sorted(out))


@dataclass
class MatchFlags:
    same_event: bool
    same_market: bool
    same_period: bool
    same_side: bool
    same_line: bool
    same_line_strict: bool
    snapshot_before_kickoff: bool

    @property
    def is_strict(self) -> bool:
        return (
            self.same_event
            and self.same_market
            and self.same_period
            and self.same_side
            and self.same_line_strict
            and self.snapshot_before_kickoff
        )


def evaluate_match(
    *,
    want_event: Any,
    got_event: Any,
    want_market: Any,
    got_market: Any,
    want_period: Any,
    got_period: Any,
    want_side: Any,
    got_side: Any,
    want_line: Any,
    got_line: Any,
    snapshot_ts: Any,
    kickoff_ts: Any,
) -> MatchFlags:
    same_event = str(want_event or "") == str(got_event or "") and bool(want_event)
    same_market = normalize_market(want_market) == normalize_market(got_market) and normalize_market(want_market) is not None
    same_period = normalize_period(want_period) == normalize_period(got_period)
    same_side = normalize_side(want_side) == normalize_side(got_side) and normalize_side(want_side) is not None
    wl = normalize_line(want_line, want_market)
    gl = normalize_line(got_line, got_market or want_market)
    same_line = wl is not None and gl is not None and wl == gl
    before = False
    try:
        if snapshot_ts is not None and kickoff_ts is not None:
            before = float(snapshot_ts) < float(kickoff_ts)
    except Exception:
        before = False
    return MatchFlags(
        same_event=bool(same_event),
        same_market=bool(same_market),
        same_period=bool(same_period),
        same_side=bool(same_side),
        same_line=bool(same_line),
        same_line_strict=bool(same_line),
        snapshot_before_kickoff=bool(before),
    )


def choose_entry_odd(payload: dict) -> Tuple[Optional[float], Optional[str]]:
    """Priority: sent.price → odd_final → odd_at_decision."""
    req = (payload or {}).get("request") or {}
    res = (payload or {}).get("result") or {}
    raw = res.get("raw") if isinstance(res.get("raw"), dict) else {}
    sent = raw.get("sent") if isinstance(raw.get("sent"), dict) else {}
    for source, val in (
        ("sent.price", sent.get("price")),
        ("odd_final", res.get("odd_final")),
        ("odd_at_decision", res.get("odd_at_decision") or req.get("odd_at_decision")),
    ):
        try:
            if val is None:
                continue
            x = float(val)
            if x > 1.0:
                return x, source
        except Exception:
            continue
    return None, None


def clv_raw(entry_odd: float, snapshot_odd: float) -> Tuple[float, float]:
    """B808-compatible Back CLV.

    clv_raw_decimal = (entry - snapshot) / snapshot
    clv_raw_pct = clv_raw_decimal * 100

    Positive ⇒ entry better than snapshot for Back (higher price taken).
    Equivalent to entry/snapshot - 1.
    """
    e = float(entry_odd)
    s = float(snapshot_odd)
    if s <= 0 or e <= 0:
        raise ValueError("INVALID_ODD")
    dec = (e / s) - 1.0
    return dec, dec * 100.0
