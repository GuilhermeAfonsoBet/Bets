from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from storage.database import Database


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _json_safe(x: Any) -> Any:
    if isinstance(x, datetime):
        return x.isoformat()
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x


async def run_diag(*, hours: float, direction: str, hypothesis_type: str) -> Dict[str, Any]:
    db = Database()
    await db.connect()
    since = _utcnow() - timedelta(hours=float(hours))

    ver_expr = "COALESCE(audit_version, hypothesis_details->>'audit_version', 'NA')"
    side_expr = "COALESCE(hypothesis_details->>'exec_side_hint', '')"

    q = text(
        f"""
        WITH b AS (
          SELECT
            audited_at,
            status,
            is_valid_opportunity,
            {ver_expr} AS ver,
            {side_expr} AS side_hint
          FROM betslip_audit_results
          WHERE audited_at >= :since
            AND hypothesis_type = :hyp
            AND reversal_direction = :direction
        )
        SELECT
          COUNT(*)::bigint AS n_total,
          COUNT(*) FILTER (WHERE status='OK')::bigint AS n_ok,
          COUNT(*) FILTER (WHERE status='OK' AND is_valid_opportunity=TRUE)::bigint AS n_ok_valid,

          COUNT(*) FILTER (WHERE lower(ver) LIKE '%back%')::bigint AS n_ver_back,
          COUNT(*) FILTER (WHERE lower(ver) LIKE '%lay%')::bigint AS n_ver_lay,

          COUNT(*) FILTER (WHERE lower(side_hint)='back')::bigint AS n_hint_back,
          COUNT(*) FILTER (WHERE lower(side_hint)='lay')::bigint AS n_hint_lay,

          MIN(audited_at) AS first_ts,
          MAX(audited_at) AS last_ts
        FROM b;
        """
    )

    q_by_ver = text(
        f"""
        SELECT
          {ver_expr} AS ver,
          COUNT(*)::bigint AS n,
          COUNT(*) FILTER (WHERE status='OK')::bigint AS ok,
          COUNT(*) FILTER (WHERE status='OK' AND is_valid_opportunity=TRUE)::bigint AS ok_valid,
          MIN(audited_at) AS first_ts,
          MAX(audited_at) AS last_ts
        FROM betslip_audit_results
        WHERE audited_at >= :since
          AND hypothesis_type = :hyp
          AND reversal_direction = :direction
        GROUP BY 1
        ORDER BY n DESC;
        """
    )

    q_seen = text(
        """
        SELECT
          action,
          COUNT(*)::bigint AS n,
          SUM(CASE WHEN (meta->>'reason')='not_active' THEN 1 ELSE 0 END)::bigint AS n_not_active,
          SUM(CASE WHEN (meta->>'accepted')='true' THEN 1 ELSE 0 END)::bigint AS n_accepted,
          MIN(created_at) AS first_ts,
          MAX(created_at) AS last_ts
        FROM executor_bridge_seen
        WHERE created_at >= :since
        GROUP BY 1
        ORDER BY n DESC;
        """
    )

    out: Dict[str, Any] = {
        "ts_utc": _utcnow().isoformat(),
        "since_utc": since.isoformat(),
        "hours": float(hours),
        "direction": str(direction),
        "hypothesis_type": str(hypothesis_type),
        "summary": {},
        "by_version": [],
        "bridge_seen": [],
    }

    try:
        async with db.async_session() as s:
            r = await s.execute(q, {"since": since, "hyp": hypothesis_type, "direction": direction})
            row = r.first()
            out["summary"] = _json_safe(dict(row._mapping)) if row else {}

        async with db.async_session() as s:
            r = await s.execute(q_by_ver, {"since": since, "hyp": hypothesis_type, "direction": direction})
            out["by_version"] = _json_safe([dict(x._mapping) for x in (r.fetchall() or [])])

        async with db.async_session() as s:
            r = await s.execute(q_seen, {"since": since})
            out["bridge_seen"] = _json_safe([dict(x._mapping) for x in (r.fetchall() or [])])
    finally:
        try:
            await db.close()
        except Exception:
            pass

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnóstico rápido: presença de auditorias Back no DB + bridge_seen.")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--direction", default="up")
    ap.add_argument("--hypothesis-type", default="H3B")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import asyncio

    rep = asyncio.run(run_diag(hours=float(args.hours), direction=str(args.direction), hypothesis_type=str(args.hypothesis_type)))
    txt = json.dumps(rep, ensure_ascii=False, indent=2)
    if str(args.out or "").strip():
        from pathlib import Path

        p = Path(str(args.out))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

