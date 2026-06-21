#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry point dedicado para a estrategia Downward Trend.

Este wrapper fixa defaults operacionais de DT antes de delegar para o motor de
auditoria WebSocket/API reaproveitado. A intencao e evitar que um service DT
rode acidentalmente com defaults historicos de H3B/H3BUP.
"""

from __future__ import annotations

import os
import sys


def _set_default(name: str, value: str) -> None:
    if not str(os.getenv(name, "") or "").strip():
        os.environ[name] = value


# Defaults estritos da estrategia nova.
_set_default("AUDIT_HYPOTHESIS_TYPE", "DT")
_set_default("AUDIT_MODE", "ws_gate_back")
_set_default("AUDIT_API_SIDES", "back")
_set_default("GATE_BACK_ENFORCE_RISE_FILTER", "0")

# Conceitualizacao inicial: 3 quedas consecutivas acima da sensibilidade.
_set_default("DT_MIN_CONSEC_DOWNS", "3")
_set_default("DT_MIN_STEP_DROP_PCT", "0.20")
_set_default("DT_MIN_CUM_DROP_PCT", "0.80")
_set_default("DT_MAX_STEP_GAP_SEC", "30")
_set_default("DT_SIGNAL_COOLDOWN_SEC", "45")


def _arg_value(name: str) -> str | None:
    try:
        argv = list(sys.argv[1:])
        if name in argv:
            i = argv.index(name)
            if i + 1 < len(argv):
                return str(argv[i + 1])
        prefix = f"{name}="
        for arg in argv:
            if str(arg).startswith(prefix):
                return str(arg).split("=", 1)[1]
    except Exception:
        return None
    return None


def _validate_dt_args() -> None:
    hyp = _arg_value("--hypothesis-type")
    if hyp is not None and str(hyp).strip().upper() != "DT":
        raise SystemExit("audit_downward_trend_api.py aceita apenas --hypothesis-type DT")

    direction = _arg_value("--direction")
    if direction is not None and str(direction).strip().lower() not in ("down", "all"):
        raise SystemExit("audit_downward_trend_api.py aceita apenas --direction down ou all")

    api_sides = _arg_value("--api-sides")
    if api_sides is not None and str(api_sides).strip().lower() != "back":
        raise SystemExit("audit_downward_trend_api.py aceita apenas --api-sides back")

from audit_h3b_api import main  # noqa: E402  (env defaults must be set first)


if __name__ == "__main__":
    _validate_dt_args()
    raise SystemExit(main())
