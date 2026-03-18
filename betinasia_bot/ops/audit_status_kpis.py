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


def _json_safe(x: Any) -> Any:
    if isinstance(x, datetime):
        return x.isoformat()
    if isinstance(x, timedelta):
        return x.total_seconds()
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x


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
            rows.append(_json_safe(dict(x._mapping)))

    # Diagnóstico dos OK por versão: buckets de |difference_pct| para explicar queda OK_with_bs -> OK_valid
    q_okdiff = text(
        """
        SELECT
          audit_version,
          SUM(CASE WHEN status='OK' AND difference_pct IS NULL THEN 1 ELSE 0 END)::bigint AS ok_diff_null,
          SUM(CASE WHEN status='OK' AND difference_pct IS NOT NULL AND abs(difference_pct) < 2.0 THEN 1 ELSE 0 END)::bigint AS ok_absdiff_lt2,
          SUM(CASE WHEN status='OK' AND difference_pct IS NOT NULL AND abs(difference_pct) >= 2.0 AND abs(difference_pct) <= 10.0 THEN 1 ELSE 0 END)::bigint AS ok_absdiff_2_10,
          SUM(CASE WHEN status='OK' AND difference_pct IS NOT NULL AND abs(difference_pct) > 10.0 THEN 1 ELSE 0 END)::bigint AS ok_absdiff_gt10
        FROM betslip_audit_results
        WHERE hypothesis_type = :hyp
          AND reversal_direction = :direction
          AND audited_at >= :since
        GROUP BY 1
        """
    )
    okdiff_map: Dict[str, Dict[str, Any]] = {}
    async with db.async_session() as session:
        r = await session.execute(q_okdiff, {"hyp": cfg.hypothesis_type, "direction": cfg.direction, "since": since})
        for x in r.fetchall() or []:
            d = _json_safe(dict(x._mapping))
            okdiff_map[str(d.get("audit_version") or "NA")] = d

    # Top erros (api_error) por versão/status para entender "por que não-OK"
    q_err = text(
        """
        SELECT
          audit_version,
          status,
          COALESCE(hypothesis_details->>'api_error', '') AS api_error,
          COUNT(*)::bigint AS n
        FROM betslip_audit_results
        WHERE hypothesis_type = :hyp
          AND reversal_direction = :direction
          AND audited_at >= :since
          AND status <> 'OK'
          AND hypothesis_details IS NOT NULL
          AND (hypothesis_details::jsonb ? 'api_error')
        GROUP BY 1,2,3
        ORDER BY n DESC;
        """
    )
    err_rows: List[Dict[str, Any]] = []
    async with db.async_session() as session:
        r = await session.execute(q_err, {"hyp": cfg.hypothesis_type, "direction": cfg.direction, "since": since})
        for x in r.fetchall() or []:
            err_rows.append(_json_safe(dict(x._mapping)))

    top_errors_by_version: Dict[str, List[Dict[str, Any]]] = {}
    for it in err_rows:
        ver = str(it.get("audit_version") or "NA")
        top_errors_by_version.setdefault(ver, []).append(it)
    # keep only top 8 per version
    for ver, xs in list(top_errors_by_version.items()):
        top_errors_by_version[ver] = xs[:8]

    # PMM consults (denominador) vs "No PMMs received" (numerador)
    # Evita cast numérico frágil: conta "consult" quando existe telemetry.parallel_fetch_ms no JSON.
    # Isso captura apenas os casos em que de fato houve tentativa de fetch (abre ticket / espera WS).
    pmm_by_version: List[Dict[str, Any]] = []
    pmm_tot = {"pmm_consults": 0, "no_pmms": 0, "no_pmms_rate_pct": None, "error": None}
    try:
        q_pmm = text(
            """
            SELECT
              audit_version,
              COUNT(*) FILTER (
                WHERE (hypothesis_details::jsonb ? 'telemetry')
                  AND ((hypothesis_details::jsonb->'telemetry') ? 'parallel_fetch_ms')
              )::bigint AS pmm_consults,
              COUNT(*) FILTER (
                WHERE (hypothesis_details::jsonb ? 'telemetry')
                  AND ((hypothesis_details::jsonb->'telemetry') ? 'parallel_fetch_ms')
                  AND COALESCE(hypothesis_details->>'api_error', '') ILIKE '%No PMMs received%'
              )::bigint AS no_pmms
            FROM betslip_audit_results
            WHERE hypothesis_type = :hyp
              AND reversal_direction = :direction
              AND audited_at >= :since
              AND hypothesis_details IS NOT NULL
            GROUP BY 1
            ORDER BY pmm_consults DESC;
            """
        )
        async with db.async_session() as session:
            r = await session.execute(q_pmm, {"hyp": cfg.hypothesis_type, "direction": cfg.direction, "since": since})
            for x in r.fetchall() or []:
                d = _json_safe(dict(x._mapping))
                try:
                    denom = int(d.get("pmm_consults") or 0)
                    num = int(d.get("no_pmms") or 0)
                    d["no_pmms_rate_pct"] = float(num / denom * 100.0) if denom > 0 else None
                except Exception:
                    d["no_pmms_rate_pct"] = None
                pmm_by_version.append(d)
        pmm_tot["pmm_consults"] = int(sum(int(x.get("pmm_consults") or 0) for x in pmm_by_version))
        pmm_tot["no_pmms"] = int(sum(int(x.get("no_pmms") or 0) for x in pmm_by_version))
        if int(pmm_tot["pmm_consults"]) > 0:
            pmm_tot["no_pmms_rate_pct"] = float(int(pmm_tot["no_pmms"]) / int(pmm_tot["pmm_consults"]) * 100.0)
    except Exception as e:
        # não falha o relatório inteiro por causa desse KPI; reporta erro para debug
        pmm_tot["error"] = str(e)[:200]
        pmm_by_version = []

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

    # merge okdiff diagnostics
    for ver, v in by_version.items():
        d = okdiff_map.get(ver) or {}
        v["ok_diff_null"] = int(d.get("ok_diff_null") or 0)
        v["ok_absdiff_lt2"] = int(d.get("ok_absdiff_lt2") or 0)
        v["ok_absdiff_2_10"] = int(d.get("ok_absdiff_2_10") or 0)
        v["ok_absdiff_gt10"] = int(d.get("ok_absdiff_gt10") or 0)

    out = {
        "ts_utc": _utcnow().isoformat(),
        "since_utc": since.isoformat(),
        "hours": float(cfg.hours),
        "direction": cfg.direction,
        "hypothesis_type": cfg.hypothesis_type,
        "rows": rows,
        "error_rows": err_rows,
        "top_errors_by_version": top_errors_by_version,
        "by_version": sorted(by_version.values(), key=lambda x: int(x.get("total") or 0), reverse=True),
        "pmm": {"total": pmm_tot, "by_version": pmm_by_version},
    }
    out = _json_safe(out)
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

