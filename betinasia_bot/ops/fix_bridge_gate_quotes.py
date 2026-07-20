#!/usr/bin/env python3
from pathlib import Path

p = Path("/home/betbot/Bets/betinasia_bot/ops/executor_bridge_audit.py")
t = p.read_text(encoding="utf-8")
t2 = t.replace(
    '{"gate": h3bup_vnext_live_submit_required}',
    '{"gate": "h3bup_vnext_live_submit_required"}',
).replace(
    '"gate": h3bup_vnext_live_submit_required,',
    '"gate": "h3bup_vnext_live_submit_required",',
)
if t2 == t:
    i = t.find("h3bup_vnext_live_submit_required")
    print("no change; context:", repr(t[i - 60 : i + 100]))
else:
    p.write_text(t2, encoding="utf-8")
    print("fixed")

import py_compile

py_compile.compile(str(p), doraise=True)
print("compile OK")
i = p.read_text().find("non_h3bup_live_blocked")
print(p.read_text()[i - 200 : i + 550])
