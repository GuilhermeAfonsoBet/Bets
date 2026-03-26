from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from storage.database import Database


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _fmt_err(s: Optional[str], n: int = 160) -> str:
    t = (s or "").replace("\n", " ").replace("\r", " ").strip()
    return (t[:n].rstrip() + "…") if len(t) > n else t


async def _run(*, minutes: int, audit_version: str, limit: int, only_errors: bool) -> int:
    t0 = _utcnow() - timedelta(minutes=int(minutes))

    qk = text(
        """
        SELECT
          CASE
            WHEN (hypothesis_details->>'api_error') ILIKE '%No PMMs received%' THEN 'NO_PMMS'
            WHEN (hypothesis_details->>'api_error') ILIKE '%HTTP_401%' OR (hypothesis_details->>'api_error') ILIKE '%auth_error%' THEN 'HTTP_401_AUTH'
            WHEN (hypothesis_details->>'api_error') ILIKE '%NO_ROOT_SESSION_COOKIE%' THEN 'NO_ROOT_SESSION_COOKIE'
            WHEN (hypothesis_details->>'api_error') ILIKE '%RATE_LIMIT%' THEN 'RATE_LIMIT'
            WHEN (hypothesis_details->>'api_error') ILIKE '%STALE_QUEUE_WAIT%' THEN 'STALE_QUEUE_WAIT'
            WHEN (hypothesis_details->>'api_error') ILIKE '%No betslip_id%' THEN 'NO_BETSLIP_ID'
            ELSE 'OTHER'
          END AS kind,
          COUNT(*)::bigint AS n
        FROM betslip_audit_results
        WHERE audited_at >= :t0
          AND hypothesis_details->>'audit_version' = :audit_version
          AND (:only_errors = FALSE OR COALESCE(hypothesis_details->>'api_error','') <> '')
        GROUP BY 1
        ORDER BY n DESC;
        """
    )

    qs = text(
        """
        SELECT
          audited_at,
          hypothesis_details->>'api_error' AS api_error,
          hypothesis_details->'telemetry'->>'auth_401' AS auth_401,
          hypothesis_details->'telemetry'->>'back_pmm_count' AS back_pmm_count,
          hypothesis_details->'telemetry'->>'back_pmm_wait_s' AS back_pmm_wait_s,
          hypothesis_details->'telemetry'->>'back_ws_age_ms' AS back_ws_age_ms,
          hypothesis_details->'telemetry'->>'back_ws_msg_count' AS back_ws_msg_count,
          hypothesis_details->'telemetry'->>'back_post_ms' AS back_post_ms,
          hypothesis_details->'telemetry'->>'back_total_ms' AS back_total_ms
        FROM betslip_audit_results
        WHERE audited_at >= :t0
          AND hypothesis_details->>'audit_version' = :audit_version
          AND (:only_errors = FALSE OR COALESCE(hypothesis_details->>'api_error','') <> '')
        ORDER BY audited_at DESC
        LIMIT :lim;
        """
    )

    db = Database()
    await db.connect()
    try:
        async with db.async_session() as s:
            r = await s.execute(qk, {"t0": t0, "audit_version": audit_version, "only_errors": bool(only_errors)})
            kinds: List[Dict[str, Any]] = [dict(x._mapping) for x in (r.fetchall() or [])]
            r = await s.execute(
                qs,
                {"t0": t0, "audit_version": audit_version, "only_errors": bool(only_errors), "lim": int(limit)},
            )
            rows: List[Dict[str, Any]] = [dict(x._mapping) for x in (r.fetchall() or [])]
    finally:
        await db.close()

    print(f"audit_version={audit_version} minutes={minutes} only_errors={int(bool(only_errors))}")
    print("kinds:", kinds)
    print("samples:")
    for row in rows:
        print(
            row.get("audited_at"),
            "|",
            _fmt_err(row.get("api_error")),
            "| auth_401=",
            row.get("auth_401"),
            "| pmm=",
            row.get("back_pmm_count"),
            "wait_s=",
            row.get("back_pmm_wait_s"),
            "ws_age_ms=",
            row.get("back_ws_age_ms"),
            "ws_msg=",
            row.get("back_ws_msg_count"),
            "post_ms=",
            row.get("back_post_ms"),
            "total_ms=",
            row.get("back_total_ms"),
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Debug rápido de api_error/telemetry do betslip_audit_results.")
    ap.add_argument("--minutes", type=int, default=30)
    ap.add_argument("--audit-version", default="v5.2-api-back")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--only-errors", action="store_true", default=True)
    args = ap.parse_args()
    return asyncio.run(
        _run(
            minutes=int(args.minutes),
            audit_version=str(args.audit_version),
            limit=int(args.limit),
            only_errors=bool(args.only_errors),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())

