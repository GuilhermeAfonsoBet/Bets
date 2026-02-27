from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from sqlalchemy import text

sys.path.insert(0, ".")

from storage.database import Database
from executor.client import submit_execution
from executor.contracts import ExecSide, ExecutionRequest, MarketType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(s: str) -> Optional[datetime]:
    try:
        if not s:
            return None
        t = str(s).strip()
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _norm_line(line: str) -> str:
    return (str(line or "").strip()).replace(",", ".").replace("−", "-")


@dataclass
class BridgeConfig:
    poll_sec: float = 2.0
    lookback_sec: int = 120
    max_per_cycle: int = 3
    mode: str = "shadow"  # shadow|live
    exec_side: ExecSide = ExecSide.BACK
    stake: float = 3.0
    unix_socket: str = "/tmp/betinasia-exec.sock"
    http_url: Optional[str] = None
    only_hypothesis: str = "H3B"
    only_prematch: bool = True


DDL_SEEN = """
CREATE TABLE IF NOT EXISTS executor_bridge_seen (
  id BIGSERIAL PRIMARY KEY,
  src_table TEXT NOT NULL,
  src_id BIGINT NOT NULL,
  action TEXT NOT NULL,
  execution_id UUID NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  meta JSONB NULL,
  UNIQUE (src_table, src_id, action)
);
"""


async def _ensure_seen_table(db: Database) -> None:
    async with db.engine.begin() as conn:
        await conn.execute(text(DDL_SEEN))


async def _fetch_candidates(
    db: Database,
    *,
    since: datetime,
    cfg: BridgeConfig,
) -> List[Dict[str, Any]]:
    # Nota: betslip_audit_results está em models_hypothesis (tabela criada pela connect()).
    q = """
    SELECT
      r.id,
      r.hypothesis_type,
      r.event_id,
      r.market_type,
      r.line,
      r.side,
      r.is_live,
      r.websocket_odd,
      r.betslip_odd,
      r.betslip_limit,
      r.hypothesis_details,
      r.audited_at
    FROM betslip_audit_results r
    LEFT JOIN executor_bridge_seen s
      ON s.src_table='betslip_audit_results' AND s.src_id=r.id AND s.action=:action
    WHERE r.audited_at >= :since
      AND s.id IS NULL
      AND r.is_valid_opportunity = TRUE
      AND r.event_id IS NOT NULL AND r.event_id <> ''
      AND upper(r.market_type) = 'AH'
      AND r.hypothesis_type = :hyp
    ORDER BY r.audited_at ASC
    LIMIT :lim
    """
    params = {
        "since": since,
        "lim": int(cfg.max_per_cycle),
        "action": f"{cfg.mode}:{cfg.exec_side.value}",
        "hyp": str(cfg.only_hypothesis),
    }
    if cfg.only_prematch:
        q = q.replace("AND r.hypothesis_type = :hyp", "AND r.hypothesis_type = :hyp AND (r.is_live IS NULL OR r.is_live = FALSE)")
    async with db.async_session() as session:
        r = await session.execute(text(q), params)
        rows = r.fetchall() or []
        return [dict(x._mapping) for x in rows]


async def _mark_seen(
    db: Database,
    *,
    src_id: int,
    action: str,
    execution_id: Optional[str],
    meta: Dict[str, Any],
) -> None:
    q = """
    INSERT INTO executor_bridge_seen (src_table, src_id, action, execution_id, meta)
    VALUES ('betslip_audit_results', :src_id, :action, :execution_id, (:meta)::jsonb)
    ON CONFLICT (src_table, src_id, action) DO NOTHING
    """
    async with db.async_session() as session:
        await session.execute(
            text(q),
            {
                "src_id": int(src_id),
                "action": str(action),
                "execution_id": execution_id,
                "meta": json.dumps(meta, ensure_ascii=False),
            },
        )
        await session.commit()


def _build_request(row: Dict[str, Any], cfg: BridgeConfig) -> ExecutionRequest:
    odd_dec = None
    try:
        odd_dec = float(row.get("websocket_odd") or 0) or None
    except Exception:
        odd_dec = None
    if odd_dec is None:
        try:
            odd_dec = float(row.get("betslip_odd") or 0) or None
        except Exception:
            odd_dec = None

    req = ExecutionRequest(
        created_at=_utcnow(),
        audit_id=int(row.get("id") or 0),
        event_id=str(row.get("event_id")),
        market_type=MarketType.AH,
        side=str(row.get("side") or "").strip(),
        line=_norm_line(str(row.get("line") or "")),
        exec_side=cfg.exec_side,
        is_live=(cfg.mode == "live"),
        odd_at_decision=odd_dec,
    )
    req.policy.policy_version = f"bridge_{cfg.only_hypothesis.lower()}_{cfg.mode}_v0"
    if cfg.exec_side == ExecSide.LAY:
        req.policy.stake_requested = float(cfg.stake)
    else:
        req.policy.stake_requested = float(cfg.stake)
    # meta útil para auditoria
    req.meta["bridge"] = {
        "src": "betslip_audit_results",
        "src_id": int(row.get("id") or 0),
        "hypothesis_type": str(row.get("hypothesis_type") or ""),
        "audited_at": str(row.get("audited_at") or ""),
        "betslip_limit": row.get("betslip_limit"),
    }
    return req


async def run_bridge(cfg: BridgeConfig) -> int:
    db = Database()
    await db.connect()
    await _ensure_seen_table(db)

    logger.info(
        f"[bridge] started mode={cfg.mode} exec_side={cfg.exec_side.value} "
        f"poll_sec={cfg.poll_sec} lookback_sec={cfg.lookback_sec} max_per_cycle={cfg.max_per_cycle} "
        f"hyp={cfg.only_hypothesis} prematch_only={cfg.only_prematch}"
    )

    while True:
        t0 = time.time()
        since = _utcnow() - timedelta(seconds=int(cfg.lookback_sec))
        rows = await _fetch_candidates(db, since=since, cfg=cfg)
        if not rows:
            await asyncio.sleep(float(cfg.poll_sec))
            continue

        for row in rows:
            src_id = int(row.get("id") or 0)
            action = f"{cfg.mode}:{cfg.exec_side.value}"
            try:
                req = _build_request(row, cfg)
                res = await submit_execution(req=req, unix_socket=cfg.unix_socket, http_base=cfg.http_url)
                eid = str(res.get("execution_id") or "")
                accepted = bool(res.get("accepted"))
                logger.info(f"[bridge] submit src_id={src_id} accepted={accepted} execution_id={eid}")
                await _mark_seen(db, src_id=src_id, action=action, execution_id=(eid or None), meta={"accepted": accepted, "resp": res})
            except Exception as e:
                logger.exception(f"[bridge] failed src_id={src_id}: {e}")
                await _mark_seen(db, src_id=src_id, action=action, execution_id=None, meta={"error": str(e)[:500]})

        dt = time.time() - t0
        # evita loop muito agressivo
        await asyncio.sleep(max(0.1, float(cfg.poll_sec) - dt))


def main() -> int:
    ap = argparse.ArgumentParser(description="Bridge: DB audit -> Executor (/execute).")
    ap.add_argument("--mode", default=os.getenv("BRIDGE_MODE", "shadow"), choices=["shadow", "live"])
    ap.add_argument("--exec-side", default=os.getenv("BRIDGE_EXEC_SIDE", "Back"), choices=["Back", "Lay"])
    ap.add_argument("--stake", type=float, default=float(os.getenv("BRIDGE_STAKE", "3.0")))
    ap.add_argument("--poll-sec", type=float, default=float(os.getenv("BRIDGE_POLL_SEC", "2.0")))
    ap.add_argument("--lookback-sec", type=int, default=int(os.getenv("BRIDGE_LOOKBACK_SEC", "120")))
    ap.add_argument("--max-per-cycle", type=int, default=int(os.getenv("BRIDGE_MAX_PER_CYCLE", "3")))
    ap.add_argument("--unix-socket", default=os.getenv("EXECUTOR_UNIX_SOCKET", "/tmp/betinasia-exec.sock"))
    ap.add_argument("--http-url", default=os.getenv("EXECUTOR_HTTP_URL", "").strip() or None)
    ap.add_argument("--hypothesis", default=os.getenv("BRIDGE_HYPOTHESIS", "H3B"))
    ap.add_argument("--prematch-only", action="store_true", default=(os.getenv("BRIDGE_PREMATCH_ONLY", "1").strip() not in ("0", "false", "False", "no", "NO")))
    args = ap.parse_args()

    cfg = BridgeConfig(
        poll_sec=float(args.poll_sec),
        lookback_sec=int(args.lookback_sec),
        max_per_cycle=int(args.max_per_cycle),
        mode=str(args.mode),
        exec_side=ExecSide(str(args.exec_side)),
        stake=float(args.stake),
        unix_socket=str(args.unix_socket),
        http_url=(str(args.http_url) if args.http_url else None),
        only_hypothesis=str(args.hypothesis),
        only_prematch=bool(args.prematch_only),
    )

    logger.remove()
    logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))
    try:
        asyncio.run(run_bridge(cfg))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

