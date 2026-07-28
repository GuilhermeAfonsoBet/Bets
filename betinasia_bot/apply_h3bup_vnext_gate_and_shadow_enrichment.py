#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aplica correcoes incrementais no bridge/executor vivo para H3BUP_vNext:

1. Bridge faz hard-reject pre-exec de Back que nao passe Back Pre + odd 1.85-2.15.
2. Capacity 0/NULL no audit instantaneo e tratada como desconhecida pre-exec,
   pois a capacidade confiavel vem do executor ao abrir/capturar betslip.
3. Executor aplica gate final de H3BUP_vNext antes de submit:
   odd 1.85-2.15, capacity >100, slippage_pre_pct <0.
4. Shadow e atualizado para oportunidades aprovadas mas nao executadas por
   disable_back/banca/no_base_exposure/budget.
"""

from __future__ import annotations

from pathlib import Path


def patch_bridge(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    before = text
    text = text.replace(
        '''    cap = _safe_float(row.get("betslip_limit"))
    if cap is None:
        cap = _safe_float(vs.get("max_stake")) or _safe_float(vs.get("capacity")) or _safe_float(vs.get("limit"))
''',
        '''    cap = _safe_float(row.get("betslip_limit"))
    if cap is None or float(cap or 0) <= 0:
        cap = _safe_float(vs.get("max_stake")) or _safe_float(vs.get("capacity")) or _safe_float(vs.get("limit"))
''',
    )
    text = text.replace(
        '''    is_cap = bool(cap is not None and float(cap) > 100.0)
    is_slip_neg = None if slip is None else bool(float(slip) < 0.0)
    # Slippage final e confiavel e aplicado no executor imediatamente antes do submit.
    pre_exec_ok = bool(is_backpre and is_odd and is_cap)
    final_ok_if_slip_known = bool(pre_exec_ok and is_slip_neg is True)
''',
        '''    # Capacity e slippage final sao medidos de forma confiavel no executor.
    # No bridge, capacity ausente/0 em audit instantaneo fica como desconhecida.
    is_cap = None if cap is None or float(cap) <= 0 else bool(float(cap) > 100.0)
    is_slip_neg = None if slip is None else bool(float(slip) < 0.0)
    pre_exec_ok = bool(is_backpre and is_odd)
    final_ok_if_slip_known = bool(pre_exec_ok and is_slip_neg is True and (is_cap is True or is_cap is None))
''',
    )
    text = text.replace(
        '''    if not is_cap:
        rejection.append("capacity_lte_100")
''',
        '''    if is_cap is False:
        rejection.append("capacity_lte_100")
    elif is_cap is None:
        rejection.append("capacity_unknown_pre_exec")
''',
    )
    anchor = '''            if cfg.exec_side == ExecSide.BACK:
                await _record_backpre_shadow(
                    db,
                    row,
                    action,
                    h3bup_vnext_eval,
                    status_operacional="shadow_detected",
                    non_execution_reason=(None if h3bup_vnext_eval.get("is_h3bup_vnext_pre_exec") else "rule_rejected"),
                )
'''
    insert = anchor + '''                if not h3bup_vnext_eval.get("is_h3bup_vnext_pre_exec"):
                    await _mark_seen(
                        db,
                        src_id=src_id,
                        action=action,
                        execution_id=None,
                        meta={"skipped": True, "reason": "h3bup_vnext_pre_exec_rejected", "policy": h3bup_vnext_eval},
                    )
                    continue
'''
    if "h3bup_vnext_pre_exec_rejected" not in text and anchor in text:
        text = text.replace(anchor, insert, 1)
    old = '''                    if cfg.exec_side == ExecSide.BACK and bool(_rp_bool(risk_params, "disable_back")):
                        await _mark_seen(
                            db,
                            src_id=src_id,
                            action=action,
                            execution_id=None,
                            meta={"skipped": True, "reason": "disabled_back", "risk_params_json": cfg.risk_params_json},
                        )
                        await _unreserve_seen_key(db, src_key=skey, action=action)
                        continue
'''
    new = '''                    if cfg.exec_side == ExecSide.BACK and bool(_rp_bool(risk_params, "disable_back")):
                        await _record_backpre_shadow(
                            db,
                            row,
                            action,
                            h3bup_vnext_eval,
                            status_operacional="approved_not_executed" if h3bup_vnext_eval.get("is_h3bup_vnext_pre_exec") else "rejected",
                            non_execution_reason="operational_disabled_back",
                            raw_extra={"reason": "disabled_back", "risk_params_json": cfg.risk_params_json},
                        )
                        await _mark_seen(
                            db,
                            src_id=src_id,
                            action=action,
                            execution_id=None,
                            meta={"skipped": True, "reason": "disabled_back", "risk_params_json": cfg.risk_params_json},
                        )
                        await _unreserve_seen_key(db, src_key=skey, action=action)
                        continue
'''
    if "operational_disabled_back" not in text and old in text:
        text = text.replace(old, new, 1)
    if text != before:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_worker(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    before = text
    old = '''            is_back_pre = bool(req.exec_side == ExecSide.BACK and not bool(market_is_live))
            slip_val = None
            try:
                slip_val = float((dry.raw.get("value_sizing") or {}).get("slippage_pre_pct"))
            except Exception:
                slip_val = None
            block_slip = bool(gate_enabled and is_back_pre and ((slip_val is None and gate_fail_closed) or (slip_val is not None and float(slip_val) >= 0.0)))
            if block_slip:
'''
    new = '''            is_back_pre = bool(req.exec_side == ExecSide.BACK and not bool(market_is_live))
            is_h3bup_vnext_req = "H3BUP_vNext" in str(getattr(req.policy, "policy_version", ""))
            slip_val = None
            try:
                slip_val = float((dry.raw.get("value_sizing") or {}).get("slippage_pre_pct"))
            except Exception:
                slip_val = None
            odd_val = None
            try:
                odd_val = float(req.odd_at_decision) if req.odd_at_decision is not None else None
            except Exception:
                odd_val = None
            cap_val = None
            try:
                cap_val = float(limit_final) if limit_final is not None else None
            except Exception:
                cap_val = None
            h3bup_block_reasons = []
            if is_h3bup_vnext_req:
                if odd_val is None or not (1.85 <= float(odd_val) <= 2.15):
                    h3bup_block_reasons.append("odd_outside_1.85_2.15")
                if cap_val is None or float(cap_val) <= 100.0:
                    h3bup_block_reasons.append("capacity_lte_100")
                if slip_val is None:
                    h3bup_block_reasons.append("slippage_missing")
                elif float(slip_val) >= 0.0:
                    h3bup_block_reasons.append("slippage_non_negative")
            block_slip = bool(
                gate_enabled
                and is_back_pre
                and (
                    (is_h3bup_vnext_req and bool(h3bup_block_reasons))
                    or ((not is_h3bup_vnext_req) and ((slip_val is None and gate_fail_closed) or (slip_val is not None and float(slip_val) >= 0.0)))
                )
            )
            if block_slip:
'''
    if old in text:
        text = text.replace(old, new, 1)
    text = text.replace(
        '''                dry.error = (
                    "SLIPPAGE_GATE_BACK_PRE_MISSING"
                    if slip_val is None
                    else f"SLIPPAGE_GATE_BACK_PRE slippage_pre_pct={float(slip_val):.6f} requires < 0"
                )
''',
        '''                dry.error = (
                    f"H3BUP_VNEXT_GATE {'|'.join(h3bup_block_reasons)}"
                    if is_h3bup_vnext_req and h3bup_block_reasons
                    else ("SLIPPAGE_GATE_BACK_PRE_MISSING" if slip_val is None else f"SLIPPAGE_GATE_BACK_PRE slippage_pre_pct={float(slip_val):.6f} requires < 0")
                )
''',
        1,
    )
    text = text.replace(
        '''                    "slippage_pre_pct": (float(slip_val) if slip_val is not None else None),
                    "threshold": 0.0,
''',
        '''                    "slippage_pre_pct": (float(slip_val) if slip_val is not None else None),
                    "odd_at_decision": (float(odd_val) if odd_val is not None else None),
                    "capacity": (float(cap_val) if cap_val is not None else None),
                    "h3bup_vnext": bool(is_h3bup_vnext_req),
                    "block_reasons": list(h3bup_block_reasons),
                    "threshold": 0.0,
''',
        1,
    )
    if text != before:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> int:
    bridge = Path("betinasia_bot/ops/executor_bridge_audit.py")
    worker = Path("betinasia_bot/executor/worker.py")
    print(f"[OK] bridge_changed={patch_bridge(bridge)} worker_changed={patch_worker(worker)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
