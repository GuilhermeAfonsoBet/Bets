#!/usr/bin/env bash
# Resume H3BUP_vNext live placement (disable_back=false). Use only if intentionally restarting.
set -euo pipefail
ROOT="${1:-/home/betbot/Bets/betinasia_bot}"
RISK="$ROOT/logs/bridge_risk_params.json"
python3 - "$RISK" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
data = json.loads(p.read_text())
before = bool(data.get("disable_back"))
data["disable_back"] = False
data.pop("_h3bup_stop_note", None)
p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
print(f"disable_back: {before} -> False")
PY
echo "RESUME_APPLIED (wait ~5s for bridge reload)"
