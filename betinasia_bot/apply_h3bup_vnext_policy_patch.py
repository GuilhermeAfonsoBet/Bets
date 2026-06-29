#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aplica H3BUP_vNext no ambiente vivo da VPS.

O checkout operacional possui betinasia_bot/ops e betinasia_bot/executor fora do
conjunto rastreado principal em algumas maquinas. Este patch versiona a
alteracao operacional de forma reaplicavel:

- Bridge Back Pre passa a aprovar H3BUP_vNext sem filtro de ligas:
  Back Pre + odd 1.85..2.15 + capacity/limit > 100.
- Executor continua sendo o gate final de slippage_pre_pct < 0.
- Stake H3BUP_vNext passa a 30 e max_stake operacional minimo a 30.
- Cria registro shadow amplo BackPre_Shadow_All em tabela backpre_shadow_all.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


POLICY_ID = "H3BUP_vNext"
POLICY_VERSION = "H3BUP_vNext_20260629"
POLICY_STARTED_AT = "2026-06-29T00:00:00+00:00"


def _backup(path: Path) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path.with_name(f"{path.name}.bak_h3bup_vnext_{ts}").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def patch_bridge(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    changed = False
    if "DDL_BACKPRE_SHADOW_ALL" not in text:
        anchor = '''DDL_POSITIONS_IDX = [
    "CREATE INDEX IF NOT EXISTS idx_bridge_pos_match ON executor_bridge_positions (match_key, exec_side, created_at);",
    "CREATE INDEX IF NOT EXISTS idx_bridge_pos_event ON executor_bridge_positions (event_id, exec_side, created_at);",
    "CREATE INDEX IF NOT EXISTS idx_bridge_pos_execid ON executor_bridge_positions (execution_id);",
]
'''
        insert = '''DDL_BACKPRE_SHADOW_ALL = """
CREATE TABLE IF NOT EXISTS backpre_shadow_all (
  id BIGSERIAL PRIMARY KEY,
  shadow_id TEXT UNIQUE NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  detected_at TIMESTAMPTZ NULL,
  decision_at TIMESTAMPTZ NULL,
  submit_at TIMESTAMPTZ NULL,
  post_date TIMESTAMPTZ NULL,
  policy_id TEXT NOT NULL,
  policy_version TEXT NOT NULL,
  policy_started_at TIMESTAMPTZ NOT NULL,
  event_id TEXT NULL,
  market_id TEXT NULL,
  league TEXT NULL,
  competition TEXT NULL,
  is_world_cup BOOLEAN NULL,
  market_type TEXT NULL,
  selection TEXT NULL,
  ah_line TEXT NULL,
  odd_at_decision DOUBLE PRECISION NULL,
  odd_winner DOUBLE PRECISION NULL,
  bookie_winner TEXT NULL,
  odd_second DOUBLE PRECISION NULL,
  bookie_second TEXT NULL,
  odd_median DOUBLE PRECISION NULL,
  num_bookies INTEGER NULL,
  best_vs_second_pct DOUBLE PRECISION NULL,
  best_vs_median_pct DOUBLE PRECISION NULL,
  odds_dispersion DOUBLE PRECISION NULL,
  slippage_pre_pct DOUBLE PRECISION NULL,
  capacity DOUBLE PRECISION NULL,
  max_stake DOUBLE PRECISION NULL,
  aggregate_liquidity DOUBLE PRECISION NULL,
  approval_reason TEXT NULL,
  rejection_reason TEXT NULL,
  non_execution_reason TEXT NULL,
  is_backpre BOOLEAN NOT NULL DEFAULT TRUE,
  is_slippage_negativo BOOLEAN NULL,
  is_odd_185_215 BOOLEAN NULL,
  is_capacity_gt_100 BOOLEAN NULL,
  is_h3bup_vnext BOOLEAN NULL,
  is_executed BOOLEAN NOT NULL DEFAULT FALSE,
  is_shadow_only BOOLEAN NOT NULL DEFAULT TRUE,
  is_rejected_by_slippage BOOLEAN NULL,
  is_rejected_by_odd BOOLEAN NULL,
  is_rejected_by_capacity BOOLEAN NULL,
  is_rejected_by_operational_issue BOOLEAN NULL,
  is_rejected_by_bankroll BOOLEAN NULL,
  is_rejected_by_other BOOLEAN NULL,
  status_operacional TEXT NULL,
  source_table TEXT NULL,
  source_id BIGINT NULL,
  execution_id UUID NULL,
  order_id TEXT NULL,
  raw JSONB NULL
);
"""

DDL_BACKPRE_SHADOW_IDX = [
    "CREATE INDEX IF NOT EXISTS idx_backpre_shadow_event ON backpre_shadow_all (event_id, created_at);",
    "CREATE INDEX IF NOT EXISTS idx_backpre_shadow_policy ON backpre_shadow_all (policy_id, policy_version, created_at);",
    "CREATE INDEX IF NOT EXISTS idx_backpre_shadow_flags ON backpre_shadow_all (is_h3bup_vnext, is_executed, is_shadow_only);",
]

''' + anchor
        if anchor not in text:
            raise SystemExit("anchor DDL_POSITIONS_IDX nao encontrado")
        text = text.replace(anchor, insert, 1)
        changed = True
    if "await conn.execute(text(DDL_BACKPRE_SHADOW_ALL))" not in text:
        text = text.replace("        await conn.execute(text(DDL_POSITIONS))\n", "        await conn.execute(text(DDL_POSITIONS))\n        await conn.execute(text(DDL_BACKPRE_SHADOW_ALL))\n", 1)
        changed = True
    if "for stmt in DDL_BACKPRE_SHADOW_IDX" not in text:
        text = text.replace(
            "        for stmt in DDL_POSITIONS_IDX:\n            await conn.execute(text(stmt))\n",
            "        for stmt in DDL_POSITIONS_IDX:\n            await conn.execute(text(stmt))\n        for stmt in DDL_BACKPRE_SHADOW_IDX:\n            await conn.execute(text(stmt))\n",
            1,
        )
        changed = True
    if "def _h3bup_vnext_eval" not in text:
        anchor = "\n\nasync def _fetch_candidates(\n"
        helper = f'''

POLICY_ID_H3BUP_VNEXT = "{POLICY_ID}"
POLICY_VERSION_H3BUP_VNEXT = "{POLICY_VERSION}"
POLICY_STARTED_AT_H3BUP_VNEXT = "{POLICY_STARTED_AT}"


def _details_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    d = row.get("hypothesis_details")
    if isinstance(d, dict):
        return d
    if isinstance(d, str) and d.strip():
        try:
            x = json.loads(d)
            return x if isinstance(x, dict) else {{}}
        except Exception:
            return {{}}
    return {{}}


def _h3bup_vnext_eval(row: Dict[str, Any]) -> Dict[str, Any]:
    details = _details_dict(row)
    vs = details.get("value_sizing") if isinstance(details.get("value_sizing"), dict) else {{}}
    slip = _safe_float(vs.get("slippage_pre_pct"))
    odd = _safe_float(row.get("websocket_odd")) or _safe_float(row.get("betslip_odd"))
    cap = _safe_float(row.get("betslip_limit"))
    if cap is None:
        cap = _safe_float(vs.get("max_stake")) or _safe_float(vs.get("capacity")) or _safe_float(vs.get("limit"))
    is_pre = not bool(row.get("is_live"))
    is_backpre = bool(is_pre)
    is_odd = bool(odd is not None and 1.85 <= float(odd) <= 2.15)
    is_cap = bool(cap is not None and float(cap) > 100.0)
    is_slip_neg = None if slip is None else bool(float(slip) < 0.0)
    # Slippage final e confiavel e aplicado no executor imediatamente antes do submit.
    pre_exec_ok = bool(is_backpre and is_odd and is_cap)
    final_ok_if_slip_known = bool(pre_exec_ok and is_slip_neg is True)
    rejection = []
    if not is_backpre:
        rejection.append("not_backpre")
    if not is_odd:
        rejection.append("odd_outside_1.85_2.15")
    if not is_cap:
        rejection.append("capacity_lte_100")
    if is_slip_neg is False:
        rejection.append("slippage_non_negative")
    return {{
        "policy_id": POLICY_ID_H3BUP_VNEXT,
        "policy_version": POLICY_VERSION_H3BUP_VNEXT,
        "policy_started_at": POLICY_STARTED_AT_H3BUP_VNEXT,
        "odd_at_decision": odd,
        "slippage_pre_pct": slip,
        "capacity": cap,
        "is_backpre": is_backpre,
        "is_slippage_negativo": is_slip_neg,
        "is_odd_185_215": is_odd,
        "is_capacity_gt_100": is_cap,
        "is_h3bup_vnext_pre_exec": pre_exec_ok,
        "is_h3bup_vnext": final_ok_if_slip_known,
        "rejection_reason": ",".join(rejection) if rejection else None,
    }}


async def _record_backpre_shadow(db: Database, row: Dict[str, Any], action: str, ev: Dict[str, Any], *, status_operacional: str, execution_id: Optional[str] = None, is_executed: bool = False, non_execution_reason: Optional[str] = None, raw_extra: Optional[Dict[str, Any]] = None) -> None:
    try:
        shadow_id = f"{{POLICY_VERSION_H3BUP_VNEXT}}:{{int(row.get('id') or 0)}}:{{action}}"
        league = str(row.get("league") or "")
        is_wc = "world cup" in league.lower() or "copa do mundo" in league.lower() or "mundial" in league.lower()
        raw = {{
            "row": {{
                "id": row.get("id"),
                "audited_at": str(row.get("audited_at") or ""),
                "websocket_odd": row.get("websocket_odd"),
                "betslip_odd": row.get("betslip_odd"),
                "betslip_limit": row.get("betslip_limit"),
                "hypothesis_details": row.get("hypothesis_details"),
            }},
            "eval": ev,
            "extra": raw_extra or {{}},
        }}
        q = """
        INSERT INTO backpre_shadow_all (
          shadow_id, detected_at, decision_at, policy_id, policy_version, policy_started_at,
          event_id, market_id, league, competition, is_world_cup, market_type, selection, ah_line,
          odd_at_decision, odd_winner, max_stake, capacity, aggregate_liquidity, slippage_pre_pct,
          approval_reason, rejection_reason, non_execution_reason,
          is_backpre, is_slippage_negativo, is_odd_185_215, is_capacity_gt_100, is_h3bup_vnext,
          is_executed, is_shadow_only, is_rejected_by_slippage, is_rejected_by_odd,
          is_rejected_by_capacity, is_rejected_by_operational_issue, is_rejected_by_bankroll, is_rejected_by_other,
          status_operacional, source_table, source_id, execution_id, raw
        )
        VALUES (
          :shadow_id, :detected_at, now(), :policy_id, :policy_version, :policy_started_at,
          :event_id, :market_id, :league, :competition, :is_world_cup, :market_type, :selection, :ah_line,
          :odd_at_decision, :odd_winner, :max_stake, :capacity, :aggregate_liquidity, :slippage_pre_pct,
          :approval_reason, :rejection_reason, :non_execution_reason,
          :is_backpre, :is_slippage_negativo, :is_odd_185_215, :is_capacity_gt_100, :is_h3bup_vnext,
          :is_executed, :is_shadow_only, :is_rejected_by_slippage, :is_rejected_by_odd,
          :is_rejected_by_capacity, :is_rejected_by_operational_issue, :is_rejected_by_bankroll, :is_rejected_by_other,
          :status_operacional, 'betslip_audit_results', :source_id, :execution_id, (:raw)::jsonb
        )
        ON CONFLICT (shadow_id) DO UPDATE
        SET decision_at=EXCLUDED.decision_at,
            status_operacional=EXCLUDED.status_operacional,
            execution_id=COALESCE(EXCLUDED.execution_id, backpre_shadow_all.execution_id),
            is_executed=backpre_shadow_all.is_executed OR EXCLUDED.is_executed,
            is_shadow_only=EXCLUDED.is_shadow_only,
            non_execution_reason=COALESCE(EXCLUDED.non_execution_reason, backpre_shadow_all.non_execution_reason),
            raw=EXCLUDED.raw;
        """
        params = {{
            "shadow_id": shadow_id,
            "detected_at": row.get("audited_at"),
            "policy_id": ev.get("policy_id"),
            "policy_version": ev.get("policy_version"),
            "policy_started_at": ev.get("policy_started_at"),
            "event_id": str(row.get("event_id") or ""),
            "market_id": str(row.get("event_id") or ""),
            "league": league,
            "competition": league,
            "is_world_cup": bool(is_wc),
            "market_type": str(row.get("market_type") or ""),
            "selection": str(row.get("side") or ""),
            "ah_line": str(row.get("line") or ""),
            "odd_at_decision": ev.get("odd_at_decision"),
            "odd_winner": ev.get("odd_at_decision"),
            "max_stake": ev.get("capacity"),
            "capacity": ev.get("capacity"),
            "aggregate_liquidity": ev.get("capacity"),
            "slippage_pre_pct": ev.get("slippage_pre_pct"),
            "approval_reason": "H3BUP_vNext_pre_exec_pass" if ev.get("is_h3bup_vnext_pre_exec") else None,
            "rejection_reason": ev.get("rejection_reason"),
            "non_execution_reason": non_execution_reason,
            "is_backpre": bool(ev.get("is_backpre")),
            "is_slippage_negativo": ev.get("is_slippage_negativo"),
            "is_odd_185_215": bool(ev.get("is_odd_185_215")),
            "is_capacity_gt_100": bool(ev.get("is_capacity_gt_100")),
            "is_h3bup_vnext": bool(ev.get("is_h3bup_vnext_pre_exec")),
            "is_executed": bool(is_executed),
            "is_shadow_only": not bool(is_executed),
            "is_rejected_by_slippage": ("slippage" in str(ev.get("rejection_reason") or "")),
            "is_rejected_by_odd": ("odd_" in str(ev.get("rejection_reason") or "")),
            "is_rejected_by_capacity": ("capacity" in str(ev.get("rejection_reason") or "")),
            "is_rejected_by_operational_issue": bool(non_execution_reason in ("downtime", "session", "PMM", "API")),
            "is_rejected_by_bankroll": bool(non_execution_reason == "bankroll"),
            "is_rejected_by_other": False,
            "status_operacional": status_operacional,
            "source_id": int(row.get("id") or 0),
            "execution_id": execution_id,
            "raw": json.dumps(raw, ensure_ascii=False, default=str),
        }}
        async with db.async_session() as session:
            await session.execute(text(q), params)
            await session.commit()
    except Exception as e:
        logger.warning(f"[bridge] backpre_shadow_record failed src_id={{row.get('id')}}: {{e}}")
'''
        if anchor not in text:
            raise SystemExit("anchor _fetch_candidates nao encontrado")
        text = text.replace(anchor, helper + anchor, 1)
        changed = True
    if "h3bup_vnext_eval = _h3bup_vnext_eval(row)" not in text:
        anchor = '''            src_id = int(row.get("id") or 0)
            action = f"{cfg.mode}:{cfg.exec_side.value}"
            skey = ""
'''
        repl = anchor + '''            h3bup_vnext_eval = _h3bup_vnext_eval(row) if cfg.exec_side == ExecSide.BACK else {}
            if cfg.exec_side == ExecSide.BACK:
                await _record_backpre_shadow(
                    db,
                    row,
                    action,
                    h3bup_vnext_eval,
                    status_operacional="shadow_detected",
                    non_execution_reason=(None if h3bup_vnext_eval.get("is_h3bup_vnext_pre_exec") else "rule_rejected"),
                )
'''
        if anchor not in text:
            raise SystemExit("anchor row action nao encontrado")
        text = text.replace(anchor, repl, 1)
        changed = True
    if "h3bup_vnext bypasses league policy" not in text:
        old = '''                    if not ok:
                        await _mark_seen(
'''
        new = '''                    if (not ok) and cfg.exec_side == ExecSide.BACK and h3bup_vnext_eval.get("is_h3bup_vnext_pre_exec"):
                        ok = True
                        logger.info(f"[bridge] H3BUP_vNext bypasses league policy src_id={src_id} combo={comb}")
                    if not ok:
                        await _record_backpre_shadow(
                            db,
                            row,
                            action,
                            h3bup_vnext_eval,
                            status_operacional="rejected",
                            non_execution_reason="league",
                            raw_extra={"combo": comb},
                        )
                        await _mark_seen(
'''
        if old not in text:
            raise SystemExit("anchor policy not ok nao encontrado")
        text = text.replace(old, new, 1)
        changed = True
    if "H3BUP_vNext_force_stake_30" not in text:
        anchor = '''                # Gate de slippage (enforced no executor antes do LIVE):
'''
        insert = '''                # H3BUP_vNext: policy versionada, sem filtro de ligas, stake 30.
                try:
                    if cfg.exec_side == ExecSide.BACK and h3bup_vnext_eval.get("is_h3bup_vnext_pre_exec"):
                        req.policy.stake_requested = 30.0
                        req.policy.liability_requested = None
                        req.policy.policy_version = POLICY_VERSION_H3BUP_VNEXT
                        req.meta.setdefault("policy", {})
                        req.meta["policy"].update({
                            "policy_id": POLICY_ID_H3BUP_VNEXT,
                            "policy_version": POLICY_VERSION_H3BUP_VNEXT,
                            "policy_started_at": POLICY_STARTED_AT_H3BUP_VNEXT,
                            "criteria": {
                                "back_pre": True,
                                "slippage_pre_pct": "<0 enforced in executor pre-submit",
                                "odd_range": [1.85, 2.15],
                                "capacity_gt": 100,
                                "league_filter": "disabled",
                            },
                            "parameters": {"stake": 30.0},
                            "change_reason": "H3BUP_vNext: slippage<0 + odd 1.85-2.15 + capacity>100; remover filtro de ligas e manter shadow amplo",
                            "H3BUP_vNext_force_stake_30": True,
                        })
                except Exception:
                    pass

''' + anchor
        if anchor not in text:
            raise SystemExit("anchor gate slippage nao encontrado")
        text = text.replace(anchor, insert, 1)
        changed = True
    if "status_operacional=\"submitted\"" not in text:
        old = '''                await _finalize_seen_key(db, src_key=skey, action=action, execution_id=(eid or None))
                await _mark_seen(db, src_id=src_id, action=action, execution_id=(eid or None), meta={"accepted": accepted, "resp": res})
'''
        new = '''                if cfg.exec_side == ExecSide.BACK:
                    await _record_backpre_shadow(
                        db,
                        row,
                        action,
                        h3bup_vnext_eval,
                        status_operacional="submitted" if accepted else "approved_not_executed",
                        execution_id=(eid or None),
                        is_executed=bool(accepted),
                        non_execution_reason=(None if accepted else ("API" if int(hs or 0) >= 400 else "other")),
                        raw_extra={"submit_response": res, "accepted": accepted},
                    )
                await _finalize_seen_key(db, src_key=skey, action=action, execution_id=(eid or None))
                await _mark_seen(db, src_id=src_id, action=action, execution_id=(eid or None), meta={"accepted": accepted, "resp": res, "policy": req.meta.get("policy") if isinstance(req.meta, dict) else None})
'''
        if old not in text:
            raise SystemExit("anchor final mark seen nao encontrado")
        text = text.replace(old, new, 1)
        changed = True
    if changed:
        _backup(path)
        path.write_text(text, encoding="utf-8")
    return changed


def patch_executor(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    changed = False
    if "H3BUP_vNext_min_max_stake_30" not in text:
        old = '        max_stake = float(os.getenv("EXECUTOR_LIVE_MAX_STAKE", "5.0"))\n'
        new = old + '''        if "H3BUP_vNext" in str(getattr(req.policy, "policy_version", "")):
            max_stake = max(float(max_stake), 30.0)  # H3BUP_vNext_min_max_stake_30
'''
        if old not in text:
            raise SystemExit("executor max_stake anchor nao encontrado")
        text = text.replace(old, new, 1)
        changed = True
    if "H3BUP_vNext_force_stake_30" not in text:
        old = "                stake = float(stake_pre_fast if is_pre_fast else stake_back_default)\n"
        new = '''                if "H3BUP_vNext" in str(getattr(req.policy, "policy_version", "")):
                    stake = 30.0  # H3BUP_vNext_force_stake_30
                else:
                    stake = float(stake_pre_fast if is_pre_fast else stake_back_default)
'''
        if old not in text:
            raise SystemExit("executor stake sizing anchor nao encontrado")
        text = text.replace(old, new, 1)
        changed = True
    if changed:
        _backup(path)
        path.write_text(text, encoding="utf-8")
    return changed


def main() -> int:
    bridge = Path("betinasia_bot/ops/executor_bridge_audit.py")
    worker = Path("betinasia_bot/executor/worker.py")
    if not bridge.exists():
        raise SystemExit(f"arquivo nao encontrado: {bridge}")
    if not worker.exists():
        raise SystemExit(f"arquivo nao encontrado: {worker}")
    cb = patch_bridge(bridge)
    cw = patch_executor(worker)
    print(f"[OK] bridge_changed={cb} worker_changed={cw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
