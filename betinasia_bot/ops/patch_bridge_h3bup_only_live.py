#!/usr/bin/env python3
"""Bridge: Back LIVE so submete se H3BUP_vNext pre-exec passar."""
from pathlib import Path

PATH = Path("/home/betbot/Bets/betinasia_bot/ops/executor_bridge_audit.py")
MARKER = "h3bup_vnext_live_submit_required"

OLD = """                except Exception:
                    pass

                # Gate de slippage (enforced no executor antes do LIVE):
"""

NEW = """                except Exception:
                    pass

                # HARD GATE: Back LIVE so com H3BUP_vNext pre-exec.
                # Impede o path legado bridge_h3b_live_v0 (stake 1.5/20) quando H3BUP falha.
                if cfg.mode == "live" and cfg.exec_side == ExecSide.BACK:
                    if not bool(h3bup_vnext_eval.get("is_h3bup_vnext_pre_exec")):
                        await _record_backpre_shadow(
                            db,
                            row,
                            action,
                            h3bup_vnext_eval,
                            status_operacional="rejected",
                            non_execution_reason="non_h3bup_live_blocked",
                            raw_extra={"gate": "h3bup_vnext_live_submit_required"},
                        )
                        await _mark_seen(
                            db,
                            src_id=src_id,
                            action=action,
                            execution_id=None,
                            meta={
                                "skipped": True,
                                "reason": "non_h3bup_live_blocked",
                                "policy_version": str((h3bup_vnext_eval or {}).get("policy_version")),
                                "rejection_reason": (h3bup_vnext_eval or {}).get("rejection_reason"),
                                "gate": "h3bup_vnext_live_submit_required",
                            },
                        )
                        continue

                # Gate de slippage (enforced no executor antes do LIVE):
"""


def main() -> int:
    text = PATH.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"[OK] ja presente: {PATH}")
        return 0
    if OLD not in text:
        raise SystemExit("anchor nao encontrado")
    PATH.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"[OK] hard gate bridge aplicado: {PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
