"""Price sources for H3BUP CLV: BestOddsHistory + passive JSONL (no betslip)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ops.h3bup_clv_matching import line_variants, normalize_side


@dataclass
class PriceHit:
    odd: float
    snapshot_ts: float
    source: str
    source_record_id: Optional[str]
    event_id: str
    market_type: str
    period: str
    side: str
    line: str
    match_id: Optional[int] = None


def _side_column(side: str) -> str:
    s = normalize_side(side) or ""
    if s in ("home", "over", "side_a"):
        return "home"
    return "away"


async def resolve_match(session, event_id: str) -> Tuple[Optional[int], Optional[datetime], Optional[str]]:
    """Returns match_id, kickoff, error_code."""
    from sqlalchemy import text

    r = await session.execute(
        text("SELECT id, kickoff_time FROM matches WHERE external_id = :e"),
        {"e": str(event_id)},
    )
    rows = r.fetchall()
    if not rows:
        return None, None, "EVENT_NOT_FOUND"
    if len(rows) > 1:
        kicks = {str(x[1]) for x in rows}
        if len(kicks) > 1:
            return None, None, "KICKOFF_CONFLICT"
    mid, kick = rows[0][0], rows[0][1]
    if kick is None:
        return int(mid), None, "KICKOFF_MISSING"
    return int(mid), kick, None


async def find_boh_nearest(
    session,
    *,
    match_id: int,
    market_type: str,
    side: str,
    line: str,
    target_ts: float,
    tol_before: float,
    tol_after: float,
    live_ok_ts: float,
    kickoff_ts: Optional[float],
    period: str = "full_time",
    event_id: str = "",
) -> Tuple[Optional[PriceHit], Optional[str]]:
    """Nearest BOH snapshot within tolerance; same-line variants only."""
    from sqlalchemy import text

    if period and str(period) not in ("full_time", "ft", "full"):
        # BOH does not distinguish period; only FT supported strictly
        if normalize_period_local(period) != "full_time":
            return None, "PERIOD_NOT_FOUND"

    variants = list(line_variants(line, market_type))
    if not variants:
        return None, "LINE_NOT_FOUND"
    col = _side_column(side)
    odd_col = "best_home_odds" if col == "home" else "best_away_odds"
    # query window
    lo = datetime.fromtimestamp(target_ts - tol_before, tz=timezone.utc)
    hi = datetime.fromtimestamp(target_ts + tol_after, tz=timezone.utc)
    entry = datetime.fromtimestamp(live_ok_ts, tz=timezone.utc)
    q = text(
        f"""
        SELECT id, ah_line, {odd_col} AS odd, scraped_at
        FROM best_odds_history
        WHERE match_id = :mid
          AND ah_line = ANY(:lines)
          AND scraped_at >= :lo AND scraped_at <= :hi
          AND scraped_at >= :entry
        ORDER BY scraped_at
        """
    )
    r = await session.execute(q, {"mid": match_id, "lines": variants, "lo": lo, "hi": hi, "entry": entry})
    rows = r.fetchall()
    if not rows:
        # distinguish missing line vs missing time
        r2 = await session.execute(
            text("SELECT count(*) FROM best_odds_history WHERE match_id=:mid AND ah_line = ANY(:lines)"),
            {"mid": match_id, "lines": variants},
        )
        cnt = int(r2.scalar() or 0)
        if cnt <= 0:
            return None, "LINE_NOT_FOUND"
        return None, "SOURCE_MISSING"

    best = None
    best_dist = None
    for row in rows:
        rid, ah, odd, scraped = row
        if odd is None or float(odd) <= 1.0:
            continue
        ts = scraped.timestamp() if hasattr(scraped, "timestamp") else float(scraped)
        if kickoff_ts is not None and ts >= float(kickoff_ts):
            continue
        if ts < float(live_ok_ts):
            continue
        dist = abs(ts - float(target_ts))
        if best is None or dist < best_dist:
            best_dist = dist
            best = PriceHit(
                odd=float(odd),
                snapshot_ts=ts,
                source="best_odds_history",
                source_record_id=str(rid),
                event_id=event_id,
                market_type=market_type,
                period="full_time",
                side=normalize_side(side) or side,
                line=str(ah),
                match_id=match_id,
            )
    if best is None:
        return None, "SNAPSHOT_AFTER_KICKOFF"
    return best, None


def normalize_period_local(period: Any) -> str:
    from ops.h3bup_clv_matching import normalize_period

    return normalize_period(period)


async def find_boh_closing(
    session,
    *,
    match_id: int,
    market_type: str,
    side: str,
    line: str,
    kickoff_ts: float,
    closing_buffer_sec: float,
    closing_max_age_sec: float,
    event_id: str = "",
    period: str = "full_time",
) -> Tuple[Optional[PriceHit], Optional[str]]:
    from sqlalchemy import text

    if normalize_period_local(period) != "full_time":
        return None, "PERIOD_NOT_FOUND"
    variants = list(line_variants(line, market_type))
    if not variants:
        return None, "LINE_NOT_FOUND"
    col = _side_column(side)
    odd_col = "best_home_odds" if col == "home" else "best_away_odds"
    cutoff = datetime.fromtimestamp(float(kickoff_ts) - float(closing_buffer_sec), tz=timezone.utc)
    q = text(
        f"""
        SELECT id, ah_line, {odd_col} AS odd, scraped_at
        FROM best_odds_history
        WHERE match_id = :mid
          AND ah_line = ANY(:lines)
          AND scraped_at <= :cutoff
        ORDER BY scraped_at DESC
        LIMIT 1
        """
    )
    r = await session.execute(q, {"mid": match_id, "lines": variants, "cutoff": cutoff})
    row = r.fetchone()
    if not row:
        r2 = await session.execute(
            text("SELECT count(*) FROM best_odds_history WHERE match_id=:mid AND ah_line = ANY(:lines)"),
            {"mid": match_id, "lines": variants},
        )
        if int(r2.scalar() or 0) <= 0:
            return None, "LINE_NOT_FOUND"
        return None, "SOURCE_MISSING"
    rid, ah, odd, scraped = row
    if odd is None or float(odd) <= 1.0:
        return None, "INVALID_ODD"
    ts = scraped.timestamp() if hasattr(scraped, "timestamp") else float(scraped)
    age = float(kickoff_ts) - ts
    if age > float(closing_max_age_sec):
        return None, "SNAPSHOT_TOO_FAR"
    if ts >= float(kickoff_ts):
        return None, "SNAPSHOT_AFTER_KICKOFF"
    return (
        PriceHit(
            odd=float(odd),
            snapshot_ts=ts,
            source="best_odds_history",
            source_record_id=str(rid),
            event_id=event_id,
            market_type=market_type,
            period="full_time",
            side=normalize_side(side) or side,
            line=str(ah),
            match_id=match_id,
        ),
        None,
    )


def find_passive_nearest(
    path: str,
    *,
    order_id: str,
    event_id: str,
    side: str,
    line: str,
    market_type: str,
    target_ts: float,
    tol_before: float,
    tol_after: float,
    live_ok_ts: float,
    kickoff_ts: Optional[float],
) -> Tuple[Optional[PriceHit], Optional[str]]:
    p = Path(path)
    if not p.exists():
        return None, "COLLECTOR_UNAVAILABLE"
    variants = set(line_variants(line, market_type))
    want_side = normalize_side(side)
    best = None
    best_dist = None
    any_line = False
    with p.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                o = json.loads(raw)
            except Exception:
                continue
            if str(o.get("event_id") or "") != str(event_id):
                continue
            if normalize_side(o.get("side")) != want_side:
                continue
            if str(o.get("line") or "") not in variants and str(o.get("line") or "") not in {line}:
                # also accept normalized
                from ops.h3bup_clv_matching import normalize_line

                if normalize_line(o.get("line"), market_type) not in {normalize_line(line, market_type)}:
                    continue
            any_line = True
            try:
                ts = datetime.fromisoformat(str(o.get("observed_ts_utc")).replace("Z", "+00:00")).timestamp()
                odd = float(o.get("observed_odd"))
            except Exception:
                continue
            if odd <= 1.0:
                continue
            if ts < live_ok_ts:
                continue
            if kickoff_ts is not None and ts >= kickoff_ts:
                continue
            if ts < target_ts - tol_before or ts > target_ts + tol_after:
                continue
            dist = abs(ts - target_ts)
            if best is None or dist < best_dist:
                best_dist = dist
                best = PriceHit(
                    odd=odd,
                    snapshot_ts=ts,
                    source="passive_collector",
                    source_record_id=str(o.get("id") or ""),
                    event_id=event_id,
                    market_type=market_type,
                    period=str(o.get("period") or "full_time"),
                    side=want_side or side,
                    line=str(o.get("line")),
                )
    if best:
        return best, None
    if not any_line:
        return None, "LINE_NOT_FOUND"
    return None, "SOURCE_MISSING"
