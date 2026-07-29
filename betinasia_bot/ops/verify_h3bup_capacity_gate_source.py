#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verifica se o worker usa dry.limit_final no gate H3BUP (nao limit_final solto)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def check_worker(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    uses_dry = bool(re.search(r"cap_val\s*=\s*float\(\s*dry\.limit_final\s*\)", text))
    uses_bare = bool(re.search(r"cap_val\s*=\s*float\(\s*limit_final\s*\)", text))
    has_gate = "H3BUP_VNEXT_GATE" in text and "capacity_lte_100" in text
    ok = bool(has_gate and uses_dry and not uses_bare)
    return {
        "worker": str(path),
        "has_h3bup_gate": has_gate,
        "uses_dry_limit_final": uses_dry,
        "uses_bare_limit_final": uses_bare,
        "ok": ok,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default="executor/worker.py")
    args = ap.parse_args()
    out = check_worker(Path(args.worker))
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
