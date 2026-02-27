from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
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
    # Policy OOS (export do report walk-forward)
    policy_json: Optional[str] = None
    policy_reload_sec: float = 5.0
    policy_use_base: bool = False
    # Guardrails simples
    min_limit: float = 0.0


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

DDL_SEEN_KEYS = """
CREATE TABLE IF NOT EXISTS executor_bridge_seen_keys (
  id BIGSERIAL PRIMARY KEY,
  src_table TEXT NOT NULL,
  src_key TEXT NOT NULL,
  action TEXT NOT NULL,
  execution_id UUID NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  meta JSONB NULL,
  UNIQUE (src_table, src_key, action)
);
"""


async def _ensure_seen_table(db: Database) -> None:
    async with db.engine.begin() as conn:
        await conn.execute(text(DDL_SEEN))
        await conn.execute(text(DDL_SEEN_KEYS))


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _event_key(row: Dict[str, Any], cfg: BridgeConfig) -> str:
    event_id = str(row.get("event_id") or "").strip()
    market = str(row.get("market_type") or "AH").strip().upper()
    line = _norm_line(str(row.get("line") or ""))
    side = str(row.get("side") or "").strip().lower()
    is_live = bool(row.get("is_live")) if row.get("is_live") is not None else False
    hyp = str(row.get("hypothesis_type") or "").strip().upper()
    regime = "in" if is_live else "pre"
    return f"{event_id}|{market}|{line}|{side}|{cfg.exec_side.value}|{cfg.mode}|{regime}|{hyp}"


async def _reserve_seen_key(
    db: Database,
    *,
    src_key: str,
    action: str,
    meta: Dict[str, Any],
) -> bool:
    q = """
    INSERT INTO executor_bridge_seen_keys (src_table, src_key, action, execution_id, meta)
    VALUES ('betslip_audit_results', :src_key, :action, NULL, (:meta)::jsonb)
    ON CONFLICT (src_table, src_key, action) DO NOTHING
    RETURNING id
    """
    async with db.async_session() as session:
        r = await session.execute(
            text(q),
            {"src_key": str(src_key), "action": str(action), "meta": json.dumps(meta, ensure_ascii=False)},
        )
        row = r.fetchone()
        await session.commit()
        return bool(row and row[0])


async def _finalize_seen_key(
    db: Database,
    *,
    src_key: str,
    action: str,
    execution_id: Optional[str],
) -> None:
    if not execution_id:
        return
    q = """
    UPDATE executor_bridge_seen_keys
    SET execution_id = :execution_id
    WHERE src_table='betslip_audit_results' AND src_key=:src_key AND action=:action
    """
    async with db.async_session() as session:
        await session.execute(
            text(q),
            {"execution_id": str(execution_id), "src_key": str(src_key), "action": str(action)},
        )
        await session.commit()


def _load_policy_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        p = Path(path)
        if not p.exists():
            return None
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _combo_key_from_row(row: Dict[str, Any], cfg: BridgeConfig, policy: Dict[str, Any]) -> str:
    is_live = bool(row.get("is_live")) if row.get("is_live") is not None else False
    regime = "In" if is_live else "Pre"

    if cfg.exec_side == ExecSide.BACK:
        comb = f"Back_{regime}_Any"
    else:
        hyp = str(row.get("hypothesis_type") or "").strip().upper()
        details = row.get("hypothesis_details")
        if isinstance(details, str):
            try:
                details = json.loads(details)
            except Exception:
                details = None
        had_rev: Optional[bool] = None
        if hyp == "H3B":
            had_rev = True
        elif isinstance(details, dict) and "had_reversal" in details:
            had_rev = bool(details.get("had_reversal"))
        else:
            had_rev = bool(str(row.get("reversal_direction") or "").strip())
        rev = "Yes" if had_rev else "No"
        comb = f"Lay_{regime}_{rev}"

    wf = policy.get("wf") if isinstance(policy.get("wf"), dict) else {}
    key_by_league = bool(wf.get("key_by_league"))
    scope = str(wf.get("key_by_league_scope") or "pre").strip().lower()
    if key_by_league and (scope == "all" or regime == "Pre"):
        league = str(row.get("league") or "").strip()
        if league:
            comb = f"{comb}__{league}"
    return comb


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
      r.league,
      r.reversal_direction,
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

    policy: Optional[Dict[str, Any]] = None
    policy_mtime: Optional[float] = None
    policy_last_check = 0.0
    active_keys: Optional[set] = None
    active_keys_base: Optional[set] = None

    logger.info(
        f"[bridge] started mode={cfg.mode} exec_side={cfg.exec_side.value} "
        f"poll_sec={cfg.poll_sec} lookback_sec={cfg.lookback_sec} max_per_cycle={cfg.max_per_cycle} "
        f"hyp={cfg.only_hypothesis} prematch_only={cfg.only_prematch} "
        f"policy_json={cfg.policy_json or '-'} use_base={cfg.policy_use_base} min_limit={cfg.min_limit}"
    )

    while True:
        t0 = time.time()
        # reload policy (se configurado)
        if cfg.policy_json and (time.time() - policy_last_check) >= float(cfg.policy_reload_sec):
            policy_last_check = time.time()
            try:
                p = Path(cfg.policy_json)
                mtime = p.stat().st_mtime if p.exists() else None
                if mtime and (policy_mtime is None or float(mtime) > float(policy_mtime)):
                    pol = _load_policy_json(cfg.policy_json)
                    if pol:
                        policy = pol
                        policy_mtime = float(mtime)
                        steps = pol.get("steps") if isinstance(pol.get("steps"), list) else []
                        last = steps[-1] if steps else None
                        if isinstance(last, dict):
                            active_keys = set(last.get("active_keys") or [])
                            active_keys_base = set(last.get("active_keys_base") or [])
                            logger.info(
                                f"[bridge] policy reloaded mtime={policy_mtime:.0f} "
                                f"active_keys={len(active_keys)} active_base={len(active_keys_base)}"
                            )
            except Exception as e:
                logger.warning(f"[bridge] policy reload failed: {e}")

        since = _utcnow() - timedelta(seconds=int(cfg.lookback_sec))
        rows = await _fetch_candidates(db, since=since, cfg=cfg)
        if not rows:
            await asyncio.sleep(float(cfg.poll_sec))
            continue

        for row in rows:
            src_id = int(row.get("id") or 0)
            action = f"{cfg.mode}:{cfg.exec_side.value}"
            try:
                # guardrail: limit mínimo
                lim = _safe_float(row.get("betslip_limit"))
                if cfg.min_limit and lim is not None and float(lim) < float(cfg.min_limit):
                    await _mark_seen(
                        db,
                        src_id=src_id,
                        action=action,
                        execution_id=None,
                        meta={"skipped": True, "reason": "min_limit", "betslip_limit": lim, "min_limit": cfg.min_limit},
                    )
                    continue

                # policy OOS: só executa se combinação estiver ativa
                if policy and active_keys is not None:
                    comb = _combo_key_from_row(row, cfg, policy)
                    ok = False
                    if cfg.policy_use_base and active_keys_base is not None and active_keys_base:
                        ok = str(comb).split("__", 1)[0] in active_keys_base
                    else:
                        ok = comb in active_keys
                    if not ok:
                        await _mark_seen(
                            db,
                            src_id=src_id,
                            action=action,
                            execution_id=None,
                            meta={"skipped": True, "reason": "not_active", "combo": comb},
                        )
                        continue

                # dedupe por chave operacional
                skey = _event_key(row, cfg)
                reserved = await _reserve_seen_key(
                    db,
                    src_key=skey,
                    action=action,
                    meta={"src_id": src_id, "audited_at": str(row.get("audited_at") or ""), "event_id": row.get("event_id")},
                )
                if not reserved:
                    await _mark_seen(
                        db,
                        src_id=src_id,
                        action=action,
                        execution_id=None,
                        meta={"skipped": True, "reason": "dup_key", "src_key": skey},
                    )
                    continue

                req = _build_request(row, cfg)
                res = await submit_execution(req=req, unix_socket=cfg.unix_socket, http_base=cfg.http_url)
                eid = str(res.get("execution_id") or "")
                accepted = bool(res.get("accepted"))
                logger.info(f"[bridge] submit src_id={src_id} accepted={accepted} execution_id={eid}")
                await _finalize_seen_key(db, src_key=skey, action=action, execution_id=(eid or None))
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
    ap.add_argument("--policy-json", default=os.getenv("BRIDGE_POLICY_JSON", "").strip() or None, help="Path para WF policy exportado (JSON).")
    ap.add_argument("--policy-reload-sec", type=float, default=float(os.getenv("BRIDGE_POLICY_RELOAD_SEC", "5.0")))
    ap.add_argument(
        "--policy-use-base",
        action="store_true",
        default=(os.getenv("BRIDGE_POLICY_USE_BASE", "0").strip() in ("1", "true", "True", "yes", "YES")),
        help="Se true, usa active_keys_base (ignora sufixo de liga).",
    )
    ap.add_argument("--min-limit", type=float, default=float(os.getenv("BRIDGE_MIN_LIMIT", "0.0")), help="Se >0, exige betslip_limit >= este mínimo.")
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
        policy_json=(str(args.policy_json) if args.policy_json else None),
        policy_reload_sec=float(args.policy_reload_sec),
        policy_use_base=bool(args.policy_use_base),
        min_limit=float(args.min_limit),
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

