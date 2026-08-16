#!/usr/bin/env python3
"""Idempotent patch: inject H3BUP End-to-End Latency section into daily_full_report.py."""

from __future__ import annotations

import re
import sys
from pathlib import Path

MARKER_BEGIN = "# BEGIN H3BUP_E2E_LATENCY_SECTION"
MARKER_END = "# END H3BUP_E2E_LATENCY_SECTION"

SNIPPET = '''
    {begin}
    try:
        from pathlib import Path as _Path
        import os as _os
        from .analyze_h3bup_e2e_latency import load_events, group_traces, analyze_trace, summarize, render_daily_section
        _tpath = _Path(_os.getenv("H3BUP_E2E_TRACE_PATH", "logs/h3bup_e2e_trace.jsonl"))
        _evs = load_events(_tpath) if _tpath.exists() else []
        _trs = group_traces(_evs)
        _rows = [analyze_trace(tid, evs) for tid, evs in _trs.items()]
        _summary, _by_st, _cov = summarize(_rows)
        _health = {{
            "enabled": bool(_tpath.exists()),
            "schema_version": 1,
            "trace_events_dropped": 0,
            "clock_skew": sum(1 for r in _rows if r.get("clock_skew_suspected")),
            "ordering_violations": sum(1 for r in _rows if r.get("ordering_violations")),
        }}
        s0.append(render_daily_section(
            _summary, _cov, health=_health,
            n_traces=len(_rows),
            n_live=sum(1 for r in _rows if r.get("status") == "LIVE_OK"),
        ))
        s0.append("\\n")
    except Exception as _e_e2e:
        s0.append("## H3BUP End-to-End Latency\\n\\n")
        s0.append(f"_indisponível (fail-open): {{str(_e_e2e)[:160]}}_\\n\\n")
    {end}
'''.format(begin=MARKER_BEGIN, end=MARKER_END)


def patch(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if MARKER_BEGIN in text:
        return False
    anchors = [
        "# BEGIN H3BUP_ACCOUNTING_HEALTH_SECTION",
        "Accounting Health — H3BUP",
        "return \"\".join(out_lines)",
    ]
    idx = -1
    anchor_used = None
    for a in anchors:
        idx = text.find(a)
        if idx >= 0:
            anchor_used = a
            break
    if idx < 0:
        m = re.search(r"\n\s*return \"\".join\(out_lines\)", text)
        if not m:
            raise SystemExit(f"no insertion anchor in {path}")
        insert_at = m.start()
        new = text[:insert_at] + "\n" + SNIPPET + text[insert_at:]
    else:
        if anchor_used == "# BEGIN H3BUP_ACCOUNTING_HEALTH_SECTION":
            # insert after accounting section end
            end = text.find("# END H3BUP_ACCOUNTING_HEALTH_SECTION", idx)
            if end >= 0:
                line_end = text.find("\n", end)
                insert_at = line_end + 1
            else:
                line_end = text.find("\n", idx)
                insert_at = line_end + 1
        else:
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
