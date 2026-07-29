#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corrige o gate H3BUP_vNext no executor live:

Bug: em `_execute_unlocked`, o gate lia `limit_final` (variavel inexistente nesse
escopo). O `except` engolia o NameError e `capacity` ficava sempre None, gerando
`capacity_lte_100` mesmo com `dry.limit_final` > 100.

Fix: usar `dry.limit_final`.
"""

from __future__ import annotations

import argparse
from pathlib import Path


OLD = '''            cap_val = None
            try:
                cap_val = float(limit_final) if limit_final is not None else None
            except Exception:
                cap_val = None
'''

NEW = '''            cap_val = None
            try:
                # Em _execute_unlocked a capacidade vem do dry-run (dry.limit_final).
                # limit_final solto nao existe neste escopo (NameError -> capacity sempre null).
                cap_val = float(dry.limit_final) if getattr(dry, "limit_final", None) is not None else None
            except Exception:
                cap_val = None
'''


def patch(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if "float(dry.limit_final) if getattr(dry, \"limit_final\"" in text or "float(dry.limit_final) if getattr(dry, 'limit_final'" in text:
        return "already_patched"
    if OLD not in text:
        # tolerate already partially edited variants
        if "cap_val = float(dry.limit_final)" in text:
            return "already_patched"
        return "pattern_not_found"
    path.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    return "patched"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--worker",
        default="executor/worker.py",
        help="Path to executor/worker.py",
    )
    args = ap.parse_args()
    path = Path(args.worker)
    if not path.exists():
        print(f"[ERR] missing {path}")
        return 2
    status = patch(path)
    print(f"[OK] {path} status={status}")
    return 0 if status in {"patched", "already_patched"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
