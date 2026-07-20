#!/usr/bin/env python3
"""Aplica defesa em profundidade no executor: Back Pre LIVE so com H3BUP_vNext.

Motivo operacional (2026-07-20):
o servico betinasia-executor-bridge-dt rodava em paralelo como Back/H3B live
(com codigo/env legado) e submetia apostas com policy stake_requested=1.5.

O executor ainda bumpava essas ordens para sent.stake=20 via
EXECUTOR_BACKPRE_FAST_STAKE_HI=20 (drop-in zz-999-backpre-fast20.conf).
Este gate bloqueia qualquer Back Pre LIVE sem policy H3BUP_vNext.
Complementar: ops/align_h3bup_stake_overrides.sh desliga o fast-path e fixa
LIVE_STAKE/MAX_STAKE=10.
"""

from __future__ import annotations

from pathlib import Path


OLD = """            is_back_pre = bool(req.exec_side == ExecSide.BACK and not bool(market_is_live))
            is_h3bup_vnext_req = \"H3BUP_vNext\" in str(getattr(req.policy, \"policy_version\", \"\"))
            slip_val = None
"""

NEW = """            is_back_pre = bool(req.exec_side == ExecSide.BACK and not bool(market_is_live))
            is_h3bup_vnext_req = \"H3BUP_vNext\" in str(getattr(req.policy, \"policy_version\", \"\"))
            # Defesa em profundidade: nenhum Back Pre LIVE fora de H3BUP_vNext
            # (bloqueia bridge legado / DT / stake 1.5).
            if is_back_pre and (not is_h3bup_vnext_req):
                dry.status = DryRunStatus.CAP_BLOCKED
                dry.error = \"H3BUP_VNEXT_REQUIRED non_h3bup_backpre_rejected\"
                dry.raw[\"slippage_gate\"] = {
                    \"blocked\": True,
                    \"rule\": \"back_pre_requires_h3bup_vnext_policy\",
                    \"policy_version\": str(getattr(req.policy, \"policy_version\", \"\")),
                    \"stake_requested\": getattr(req.policy, \"stake_requested\", None),
                }
                return dry
            slip_val = None
"""

SIZING_OLD = """                if \"H3BUP_vNext\" in str(getattr(req.policy, \"policy_version\", \"\")):
                    stake = 10.0  # H3BUP_vNext_force_stake_10
                else:
                    stake = float(stake_pre_fast if is_pre_fast else stake_back_default)
"""

SIZING_NEW = """                if \"H3BUP_vNext\" in str(getattr(req.policy, \"policy_version\", \"\")):
                    stake = 10.0  # H3BUP_vNext_force_stake_10
                else:
                    # Nunca bumpa stake legado (ex.: 1.5 -> 20). Gate H3BUP rejeita depois.
                    stake = float(stake_back_default)  # no_fast_bump_without_h3bup
"""


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    path = root / "executor" / "worker.py"
    text = path.read_text(encoding="utf-8")
    changed = False

    if "non_h3bup_backpre_rejected" not in text:
        if OLD not in text:
            raise SystemExit(f"anchor gate nao encontrado em {path}")
        text = text.replace(OLD, NEW, 1)
        changed = True
        print(f"[OK] gate H3BUP aplicado: {path}")
    else:
        print(f"[OK] gate H3BUP ja presente: {path}")

    if "no_fast_bump_without_h3bup" not in text:
        if SIZING_OLD not in text:
            raise SystemExit(f"anchor sizing nao encontrado em {path}")
        text = text.replace(SIZING_OLD, SIZING_NEW, 1)
        changed = True
        print(f"[OK] defesa sizing (sem bump 20) aplicada: {path}")
    else:
        print(f"[OK] defesa sizing ja presente: {path}")

    if changed:
        path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
