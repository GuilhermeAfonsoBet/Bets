from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from storage.database import Database


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class AuditStatusCfg:
    hours: float = 24.0
    direction: str = "up"
    hypothesis_type: str = "H3B"


async def compute_audit_status_kpis(cfg: AuditStatusCfg) -> Dict[str, Any]:
    """
    KPIs operacionais diretamente do DB (betslip_audit_results):
    - contagem por audit_version e status (inclui no-OK: gate/api/stale/etc.)
    - quantos OK têm betslip_odd preenchido e is_valid_opportunity=true
    """
    db = Database()
    await db.connect()

    since = _utcnow() - timedelta(hours=float(cfg.hours))

    q = text(
        """
        SELECT
          audit_version,
          status,
          COUNT(*)::bigint AS n,
          SUM(CASE WHEN status='OK' THEN 1 ELSE 0 END)::bigint AS n_ok,
          SUM(CASE WHEN status='OK' AND betslip_odd IS NOT NULL THEN 1 ELSE 0 END)::bigint AS ok_with_bs,
          SUM(CASE WHEN status='OK' AND is_valid_opportunity = TRUE THEN 1 ELSE 0 END)::bigint AS ok_valid,
          MIN(audited_at) AS first_ts,
          MAX(audited_at) AS last_ts
        FROM betslip_audit_results
        WHERE hypothesis_type = :hyp
          AND reversal_direction = :direction
          AND audited_at >= :since
        GROUP BY 1,2
        ORDER BY n DESC;
        """
    )

    rows: List[Dict[str, Any]] = []
    async with db.async_session() as session:
        r = await session.execute(q, {"hyp": cfg.hypothesis_type, "direction": cfg.direction, "since": since})
        for x in r.fetchall() or []:
            rows.append(dict(x._mapping))

    # pivot leve: por version
    by_version: Dict[str, Dict[str, Any]] = {}
    for it in rows:
        ver = str(it.get("audit_version") or "NA")
        st = str(it.get("status") or "NA")
        v = by_version.setdefault(
            ver,
            {
                "audit_version": ver,
                "total": 0,
                "status_counts": {},
                "ok_with_bs": 0,
                "ok_valid": 0,
                "first_ts": None,
                "last_ts": None,
            },
        )
        n = int(it.get("n") or 0)
        v["total"] += n
        v["status_counts"][st] = int(v["status_counts"].get(st, 0)) + n
        v["ok_with_bs"] += int(it.get("ok_with_bs") or 0)
        v["ok_valid"] += int(it.get("ok_valid") or 0)

        # manter bounds por version
        f = it.get("first_ts")
        l = it.get("last_ts")
        if f is not None:
            fs = str(f)
            if v["first_ts"] is None or fs < str(v["first_ts"]):
                v["first_ts"] = f
        if l is not None:
            ls = str(l)
            if v["last_ts"] is None or ls > str(v["last_ts"]):
                v["last_ts"] = l

    out = {
        "ts_utc": _utcnow().isoformat(),
        "since_utc": since.isoformat(),
        "hours": float(cfg.hours),
        "direction": cfg.direction,
        "hypothesis_type": cfg.hypothesis_type,
        "rows": rows,
        "by_version": sorted(by_version.values(), key=lambda x: int(x.get("total") or 0), reverse=True),
    }
    try:
        await db.close()
    except Exception:
        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="KPIs de auditoria (DB): status/no-OK por audit_version + qualidade do OK.")
    ap.add_argument("--hours", type=float, default=float(os.getenv("DAILY_AUDIT_KPI_HOURS", "24")))
    ap.add_argument("--direction", default=os.getenv("DAILY_OOS_DIRECTION", "up"))
    ap.add_argument("--hypothesis-type", default="H3B")
    ap.add_argument("--out", default=os.getenv("DAILY_AUDIT_KPI_OUT", "").strip() or None)
    args = ap.parse_args()

    import asyncio

    rep = asyncio.run(
        compute_audit_status_kpis(
            AuditStatusCfg(hours=float(args.hours), direction=str(args.direction), hypothesis_type=str(args.hypothesis_type))
        )
    )
    if args.out:
        from pathlib import Path

        p = Path(str(args.out))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

