#!/usr/bin/env python3
"""Passive CLV collector — copies relevant BestOddsHistory rows for active obligations.

No external bookmaker requests, no betslip, fail-open, does not touch audit WS path.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Set, Tuple

sys.path.insert(0, ".")

from ops.h3bup_clv_config import load_config
from ops.h3bup_clv_matching import line_variants, normalize_side
from ops.h3bup_clv_store import get_store, utc_iso


def write_passive_health(cfg, payload: Dict[str, Any]) -> None:
    path = Path(cfg.passive_health_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


async def collect_once(cfg) -> Dict[str, Any]:
    store = get_store(cfg)
    active = [
        o
        for o in store.list_obligations()
        if o.get("status") not in ("COMPLETED", "SKIPPED", "FAILED_FINAL", "CANCELLED")
    ]
    keys: Set[Tuple[str, str, str]] = set()
    for o in active:
        keys.add((str(o.get("event_id")), str(o.get("line")), str(normalize_side(o.get("side")))))
    stats = {"active_obligations": len(active), "keys": len(keys), "persisted": 0, "errors": 0}
    if not keys:
        return stats
    from sqlalchemy import text
    from storage.database import Database

    db = Database()
    await db.connect()
    out_path = Path(cfg.passive_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen_local: Set[str] = set()
    # load recent ids to dedupe lightly
    if out_path.exists():
        try:
            for line in out_path.open():
                try:
                    o = json.loads(line)
                    seen_local.add(f"{o.get('source_sequence')}")
                except Exception:
                    continue
                if len(seen_local) > 200000:
                    break
        except Exception:
            pass
    try:
        async with db.async_session() as session:
            for event_id, line, side in list(keys)[:500]:
                r = await session.execute(
                    text("SELECT id, kickoff_time FROM matches WHERE external_id=:e LIMIT 1"),
                    {"e": event_id},
                )
                row = r.fetchone()
                if not row:
                    continue
                mid, kick = int(row[0]), row[1]
                variants = list(line_variants(line, "AH"))
                if not variants:
                    continue
                odd_col = "best_home_odds" if side in ("home", "over") else "best_away_odds"
                r2 = await session.execute(
                    text(
                        f"""
                        SELECT id, ah_line, {odd_col} AS odd, scraped_at
                        FROM best_odds_history
                        WHERE match_id=:mid AND ah_line = ANY(:lines)
                          AND scraped_at > now() - interval '6 hours'
                        ORDER BY scraped_at DESC
                        LIMIT 20
                        """
                    ),
                    {"mid": mid, "lines": variants},
                )
                for rid, ah, odd, scraped in r2.fetchall():
                    seq = f"boh:{rid}"
                    if seq in seen_local:
                        continue
                    if odd is None or float(odd) <= 1.0:
                        continue
                    rec = {
                        "id": str(uuid.uuid4()),
                        "obligation_id": None,
                        "order_id": None,
                        "event_id": event_id,
                        "market_type": "AH",
                        "period": "full_time",
                        "side": side,
                        "line": str(ah),
                        "observed_odd": float(odd),
                        "observed_ts_utc": scraped.isoformat() if hasattr(scraped, "isoformat") else str(scraped),
                        "source": "best_odds_history_passive_copy",
                        "source_sequence": seq,
                        "kickoff_ts_utc": kick.isoformat() if kick is not None and hasattr(kick, "isoformat") else None,
                        "created_at_utc": utc_iso(),
                    }
                    with out_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    seen_local.add(seq)
                    stats["persisted"] += 1
                    if stats["persisted"] >= 500:
                        return stats
    except Exception:
        stats["errors"] += 1
    finally:
        try:
            await db.close()
        except Exception:
            pass
    return stats


async def main_loop() -> None:
    while True:
        cfg = load_config()
        if not (cfg.enabled and cfg.passive_collector_enabled):
            write_passive_health(
                cfg,
                {
                    "checked_at_utc": utc_iso(),
                    "status": "DISABLED",
                    "enabled": False,
                    "error": None,
                },
            )
            await asyncio.sleep(30)
            continue
        try:
            st = await collect_once(cfg)
            write_passive_health(
                cfg,
                {
                    "checked_at_utc": utc_iso(),
                    "status": "HEALTHY",
                    "enabled": True,
                    "external_requests": 0,
                    "betslip_opens": 0,
                    **st,
                    "error": None,
                },
            )
        except Exception as e:
            write_passive_health(
                cfg,
                {
                    "checked_at_utc": utc_iso(),
                    "status": "WATCH",
                    "enabled": True,
                    "error": str(e)[:240],
                    "external_requests": 0,
                    "betslip_opens": 0,
                },
            )
        await asyncio.sleep(max(15.0, float(cfg.poll_sec)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    if args.once:
        cfg = load_config()

        async def _o():
            if cfg.enabled and cfg.passive_collector_enabled:
                st = await collect_once(cfg)
            else:
                st = {"skipped": True}
            write_passive_health(cfg, {"checked_at_utc": utc_iso(), "status": "OK", **st})
            print(json.dumps(st))

        asyncio.run(_o())
        return 0
    asyncio.run(main_loop())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
