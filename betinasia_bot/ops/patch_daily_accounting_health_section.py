#!/usr/bin/env python3
"""Idempotent patch: inject Accounting Health — H3BUP section into daily_full_report.py."""

from __future__ import annotations

import re
import sys
from pathlib import Path

MARKER_BEGIN = "# BEGIN H3BUP_ACCOUNTING_HEALTH_SECTION"
MARKER_END = "# END H3BUP_ACCOUNTING_HEALTH_SECTION"

SNIPPET = '''
    {begin}
    try:
        from .accounting_health_report import load_health, render_accounting_health_h3bup_section
        _hpath = Path(os.getenv("ACCOUNTING_HEALTH_JSON", "logs/accounting/accounting_health.json"))
        _health = load_health(_hpath)
        _sum = {{}}
        try:
            _sum_path = Path(os.getenv("H3BUP_ACCOUNTING_SUMMARY_JSON", "logs/h3bup_accounting_summary_latest.json"))
            if _sum_path.exists():
                import json as _json
                _sum = _json.loads(_sum_path.read_text(encoding="utf-8"))
        except Exception:
            _sum = {{}}
        s0.append(render_accounting_health_h3bup_section(health=_health, reconcile_summary=_sum))
        s0.append("\\n")
    except Exception as _e_acc_health:
        s0.append("## Accounting Health — H3BUP\\n\\n")
        s0.append(f"_indisponível: {{str(_e_acc_health)[:160]}}_\\n\\n")
    {end}
'''.format(begin=MARKER_BEGIN, end=MARKER_END)


def patch(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if MARKER_BEGIN in text:
        return False
    # Insert near early markdown assembly after accounting availability note if present
    anchors = [
        "s0.append(\"- **Accounting**: indisponível (ver apêndice 99.1)\\n\")",
        "out_lines.append(\"## 0.",
        "# Accounting: série por dia/mês",
    ]
    idx = -1
    anchor_used = None
    for a in anchors:
        idx = text.find(a)
        if idx >= 0:
            anchor_used = a
            break
    if idx < 0:
        # fallback: before final return of markdown builder if we can find a unique marker
        m = re.search(r"\n\s*return \"\".join\(out_lines\)", text)
        if not m:
            raise SystemExit(f"no insertion anchor in {path}")
        insert_at = m.start()
        new = text[:insert_at] + "\n" + SNIPPET + text[insert_at:]
    else:
        # insert after the anchor line
        line_end = text.find("\n", idx)
        insert_at = line_end + 1
        new = text[:insert_at] + SNIPPET + text[insert_at:]
    path.write_text(new, encoding="utf-8")
    print(f"patched {path} after={anchor_used!r}")
    return True


if __name__ == "__main__":
    p = Path(sys.argv[1] if len(sys.argv) > 1 else "ops/daily_full_report.py")
    changed = patch(p)
    print("changed" if changed else "already_patched")
