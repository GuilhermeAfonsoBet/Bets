#!/usr/bin/env python3
"""Aplica defesa em profundidade no executor: Back Pre LIVE so com H3BUP_vNext.

Motivo operacional (2026-07-20):
o servico betinasia-executor-bridge-dt rodava em paralelo como Back/H3B live
(com codigo/env legado) e submetia apostas stake=1.5 fora da policy H3BUP_vNext.
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


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    path = root / "executor" / "worker.py"
    text = path.read_text(encoding="utf-8")
    if "non_h3bup_backpre_rejected" in text:
        print(f"[OK] ja aplicado: {path}")
        return 0
    if OLD not in text:
        raise SystemExit(f"anchor nao encontrado em {path}")
    path.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"[OK] patch aplicado: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
