#!/usr/bin/env python3
"""Idempotent Daily patch: H3BUP CLV Forward Collection section."""

from __future__ import annotations

import re
import sys
from pathlib import Path

MARKER_BEGIN = "# BEGIN H3BUP_CLV_FORWARD_SECTION"
MARKER_END = "# END H3BUP_CLV_FORWARD_SECTION"

SNIPPET = '''
    {begin}
    try:
        import json as _json
        from pathlib import Path as _Path
        import os as _os
        _hp = _Path(_os.getenv("H3BUP_CLV_HEALTH_PATH", "logs/h3bup_clv_health.json"))
        _h = _json.loads(_hp.read_text(encoding="utf-8")) if _hp.exists() else {{"status": "WATCH", "enabled": False}}
        out_lines.append("## H3BUP CLV Forward Collection\\n\\n")
        out_lines.append("| Métrica | Valor |\\n|---|---|\\n")
        for k in [
            ("collection status", _h.get("status")),
            ("collection started at", _h.get("collection_started_at_utc")),
            ("source priority", ",".join(_h.get("source_priority") or [])),
            ("passive collector status", _h.get("collector_status")),
            ("LIVE_OK após activação", _h.get("live_ok_after_activation")),
            ("obligations esperadas", _h.get("obligations_expected")),
            ("obligations criadas", _h.get("obligations_created")),
            ("POST_5M strict válidas", _h.get("post_5m_valid_strict")),
            ("POST_15M strict válidas", _h.get("post_15m_valid_strict")),
            ("CLOSING strict válidas", _h.get("closing_valid_strict")),
            ("source missing", _h.get("source_missing")),
            ("line mismatch", _h.get("line_mismatch")),
            ("kickoff missing", _h.get("kickoff_missing")),
            ("retry backlog", _h.get("retry_backlog")),
            ("status estatístico", ("INSUFFICIENT_N" if int(_h.get("live_ok_after_activation") or 0) < 30 else "OK")),
        ]:
            out_lines.append(f"| {{k[0]}} | {{k[1]}} |\\n")
        out_lines.append("\\n")
    except Exception as _e_clv:
        out_lines.append("## H3BUP CLV Forward Collection\\n\\n")
        out_lines.append(f"_indisponível (fail-open): {{str(_e_clv)[:160]}}_\\n\\n")
    {end}
'''.format(begin=MARKER_BEGIN, end=MARKER_END)


def patch(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if MARKER_BEGIN in text:
        return False
    anchors = ["# BEGIN H3BUP_E2E_LATENCY_SECTION", "# BEGIN H3BUP_ACCOUNTING_HEALTH_SECTION", 'return "".join(out_lines)']
    idx = -1
    used = None
    for a in anchors:
        idx = text.find(a)
        if idx >= 0:
            used = a
            break
    if idx < 0:
        m = re.search(r"\n\s*return \"\".join\(out_lines\)", text)
        if not m:
            raise SystemExit(f"no anchor in {path}")
        insert_at = m.start()
        new = text[:insert_at] + "\n" + SNIPPET + text[insert_at:]
    else:
        if used and used.startswith("# BEGIN"):
            end_marker = used.replace("BEGIN", "END")
            end = text.find(end_marker, idx)
            insert_at = (text.find("\n", end) + 1) if end >= 0 else (text.find("\n", idx) + 1)
        else:
            insert_at = text.find("\n", idx) + 1
        new = text[:insert_at] + SNIPPET + text[insert_at:]
    path.write_text(new, encoding="utf-8")
    print(f"patched {path} after={used!r}")
    return True


if __name__ == "__main__":
    p = Path(sys.argv[1] if len(sys.argv) > 1 else "ops/daily_full_report.py")
    print("changed" if patch(p) else "already_patched")
