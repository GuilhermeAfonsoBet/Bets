#!/usr/bin/env python3
"""Insere hard-cap: Back Pre LIVE so com H3BUP e stake==10."""
from pathlib import Path

MARKER = "hard_stake_cap_h3bup_only_10"
PATH = Path("/home/betbot/Bets/betinasia_bot/executor/worker.py")

OLD = """        # valida max stake após sizing
        if stake > max_stake:
            dry.error = f"LIVE_STAKE_TOO_HIGH stake={stake} max={max_stake}"
            return dry
"""

NEW = """        # HARD CAP operacional H3BUP: Back Pre LIVE so stake==10; rejeita qualquer outro sizing.
        try:
            is_back_pre_hard = bool(req.exec_side == ExecSide.BACK and not bool(market_is_live))
            pol_v = str(getattr(req.policy, "policy_version", "") or "")
            is_h3b = "H3BUP_vNext" in pol_v
            if is_back_pre_hard:
                if (not is_h3b) or (stake is None) or (abs(float(stake) - 10.0) > 1e-6):
                    dry.status = ExecStatus.CAP_BLOCKED
                    dry.http_status = 200
                    dry.error = f"H3BUP_STAKE_HARD_CAP hard_stake_cap_h3bup_only_10 pol={pol_v!r} stake={stake}"
                    dry.raw = dict(dry.raw or {})
                    dry.raw["stake_hard_cap"] = {
                        "policy_version": pol_v,
                        "stake": (float(stake) if stake is not None else None),
                        "required_policy": "H3BUP_vNext*",
                        "required_stake": 10.0,
                    }
                    try:
                        await asyncio.wait_for(self._api.close_betslip(betslip_id), timeout=float(os.getenv("EXECUTOR_LIVE_CLOSE_TIMEOUT_SEC", "1.2")))
                    except Exception:
                        pass
                    return dry
                stake = 10.0
        except Exception:
            pass

        # valida max stake após sizing
        if stake > max_stake:
            dry.error = f"LIVE_STAKE_TOO_HIGH stake={stake} max={max_stake}"
            return dry
"""


def main() -> int:
    text = PATH.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"[OK] ja presente: {PATH}")
        return 0
    if OLD not in text:
        raise SystemExit("anchor valida max stake nao encontrado")
    PATH.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"[OK] hard cap inserido: {PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
