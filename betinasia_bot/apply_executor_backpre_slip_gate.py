#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aplica gate operacional no executor vivo:

Back Pre so deve postar ordem quando o slippage medido imediatamente antes do
submit for favoravel (slippage_pre_pct < 0). Mesmo quando bloquear, preserva
telemetria em raw.value_sizing/raw.slippage_gate para analise posterior.

Este patch existe porque, em algumas VPS, betinasia_bot/executor ainda nao esta
rastreado no checkout Git principal.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


ANCHOR = """        # valida max stake após sizing
        if stake > max_stake:
            dry.error = f"LIVE_STAKE_TOO_HIGH stake={stake} max={max_stake}"
            return dry

        # 3) Place order (com 1 retry via relogin se 401)
"""


PATCH = """        # Gate operacional Back Pre: so permite postar ordem se o slippage medido
        # imediatamente antes do submit for favoravel (< 0). A telemetria fica em
        # dry.raw/value_sizing e dry.raw/slippage_gate para analise posterior.
        try:
            gate_enabled = str(os.getenv("EXECUTOR_BACK_PRE_SLIP_NEG_GATE", "1")).strip().lower() not in {"0", "false", "no", "off"}
            gate_fail_closed = str(os.getenv("EXECUTOR_BACK_PRE_SLIP_NEG_FAIL_CLOSED", "1")).strip().lower() not in {"0", "false", "no", "off"}
            is_back_pre = bool(req.exec_side == ExecSide.BACK and not bool(market_is_live))
            slip_val = None
            try:
                slip_val = float((dry.raw.get("value_sizing") or {}).get("slippage_pre_pct"))
            except Exception:
                slip_val = None
            block_slip = bool(gate_enabled and is_back_pre and ((slip_val is None and gate_fail_closed) or (slip_val is not None and float(slip_val) >= 0.0)))
            if block_slip:
                try:
                    await asyncio.wait_for(self._api.close_betslip(betslip_id), timeout=float(os.getenv("EXECUTOR_LIVE_CLOSE_TIMEOUT_SEC", "1.2")))
                except Exception:
                    pass
                dry.status = ExecStatus.CAP_BLOCKED
                dry.http_status = 200
                dry.error = (
                    "SLIPPAGE_GATE_BACK_PRE_MISSING"
                    if slip_val is None
                    else f"SLIPPAGE_GATE_BACK_PRE slippage_pre_pct={float(slip_val):.6f} requires < 0"
                )
                dry.raw = dict(dry.raw or {})
                dry.raw["slippage_gate"] = {
                    "enabled": True,
                    "rule": "back_pre_requires_executor_slippage_pre_pct_lt_0",
                    "slippage_pre_pct": (float(slip_val) if slip_val is not None else None),
                    "threshold": 0.0,
                    "fail_closed": bool(gate_fail_closed),
                    "blocked": True,
                }
                try:
                    dry.timing.call_to_done_ms = _ms(max(0.0, time.time() - float(req.created_at.timestamp())))
                except Exception:
                    pass
                dry.finished_at = _now_utc()
                return dry
        except Exception as e:
            if str(os.getenv("EXECUTOR_BACK_PRE_SLIP_NEG_FAIL_CLOSED", "1")).strip().lower() not in {"0", "false", "no", "off"}:
                try:
                    await asyncio.wait_for(self._api.close_betslip(betslip_id), timeout=float(os.getenv("EXECUTOR_LIVE_CLOSE_TIMEOUT_SEC", "1.2")))
                except Exception:
                    pass
                dry.status = ExecStatus.CAP_BLOCKED
                dry.http_status = 200
                dry.error = f"SLIPPAGE_GATE_BACK_PRE_EXCEPTION: {e}"
                dry.raw = dict(dry.raw or {})
                dry.raw["slippage_gate"] = {
                    "enabled": True,
                    "rule": "back_pre_requires_executor_slippage_pre_pct_lt_0",
                    "blocked": True,
                    "exception": str(e),
                    "fail_closed": True,
                }
                dry.finished_at = _now_utc()
                return dry

        # valida max stake após sizing
        if stake > max_stake:
            dry.error = f"LIVE_STAKE_TOO_HIGH stake={stake} max={max_stake}"
            return dry

        # 3) Place order (com 1 retry via relogin se 401)
"""


def main() -> int:
    path = Path("betinasia_bot/executor/worker.py")
    if not path.exists():
        raise SystemExit(f"executor worker nao encontrado: {path}")
    text = path.read_text(encoding="utf-8")
    if "back_pre_requires_executor_slippage_pre_pct_lt_0" in text:
        print("[OK] gate ja aplicado")
        return 0
    if ANCHOR not in text:
        raise SystemExit("anchor nao encontrado; worker.py pode ter mudado")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}.bak_backpre_slip_gate_{ts}")
    backup.write_text(text, encoding="utf-8")
    path.write_text(text.replace(ANCHOR, PATCH, 1), encoding="utf-8")
    print(f"[OK] patch aplicado: {path}")
    print(f"[OK] backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
