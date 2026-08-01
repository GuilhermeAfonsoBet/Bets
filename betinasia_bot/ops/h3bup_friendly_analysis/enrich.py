"""Optional league/competition enrichment (read-only).

Executor LIVE_OK rows often lack league fields; join from:
  1. fields already on the order
  2. optional mapping CSV (event_id/audit_id → league)
  3. optional read-only SQL (DATABASE_URL) against betslip_audit_results / matches

Never consults P&L/CLV.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_league_map_csv(path: Path) -> Dict[str, Dict[str, Any]]:
    """Key by event_id primarily; also audit_id / match_id."""
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            payload = {
                "league_name": row.get("league_name") or row.get("league") or "",
                "competition_name": row.get("competition_name") or row.get("competition") or "",
                "competition_type": row.get("competition_type") or row.get("league_type") or "",
                "league_type": row.get("league_type") or "",
                "is_friendly": row.get("is_friendly") or "",
                "tournament_name": row.get("tournament_name") or "",
                "country": row.get("country") or row.get("competition_country") or "",
                "event_name": row.get("event_name") or "",
            }
            for k in ("event_id", "audit_id", "match_id"):
                key = str(row.get(k) or "").strip()
                if key:
                    out[f"{k}:{key}"] = payload
    return out


def enrich_orders(
    orders: List[Dict[str, Any]],
    *,
    league_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    league_map = league_map or {}
    out = []
    for o in orders:
        row = dict(o)
        for key_type, val in (
            ("event_id", row.get("event_id")),
            ("audit_id", row.get("audit_id")),
            ("match_id", row.get("match_id")),
        ):
            if not val:
                continue
            payload = league_map.get(f"{key_type}:{val}")
            if not payload:
                continue
            for fk, fv in payload.items():
                if fv and not row.get(fk):
                    row[fk] = fv
            break
        out.append(row)
    return out


def try_sql_league_map(database_url: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Best-effort read-only SELECT. Returns {} if unavailable."""
    url = database_url or os.environ.get("DATABASE_URL") or ""
    if not url:
        return {}
    try:
        import psycopg2  # type: ignore
    except Exception:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    try:
        conn = psycopg2.connect(url)
        conn.set_session(readonly=True, autocommit=True)
        cur = conn.cursor()
        # betslip_audit_results
        try:
            cur.execute(
                """
                SELECT event_id::text, league, status
                FROM betslip_audit_results
                WHERE event_id IS NOT NULL
                ORDER BY audited_at DESC NULLS LAST
                LIMIT 200000
                """
            )
            for event_id, league, _status in cur.fetchall():
                eid = str(event_id or "").strip()
                if not eid or f"event_id:{eid}" in out:
                    continue
                out[f"event_id:{eid}"] = {
                    "league_name": league or "",
                    "competition_name": league or "",
                    "competition_type": "",
                    "league_type": "",
                    "is_friendly": "",
                    "tournament_name": "",
                    "country": "",
                    "event_name": "",
                }
        except Exception:
            pass
        try:
            cur.execute(
                """
                SELECT external_id::text, league, home_team, away_team
                FROM matches
                WHERE external_id IS NOT NULL
                LIMIT 200000
                """
            )
            for external_id, league, home, away in cur.fetchall():
                eid = str(external_id or "").strip()
                if not eid:
                    continue
                key = f"event_id:{eid}"
                if key not in out:
                    out[key] = {
                        "league_name": league or "",
                        "competition_name": league or "",
                        "competition_type": "",
                        "league_type": "",
                        "is_friendly": "",
                        "tournament_name": "",
                        "country": "",
                        "event_name": f"{home} vs {away}" if home and away else "",
                    }
                elif not out[key].get("league_name") and league:
                    out[key]["league_name"] = league
        except Exception:
            pass
        cur.close()
        conn.close()
    except Exception:
        return {}
    return out
