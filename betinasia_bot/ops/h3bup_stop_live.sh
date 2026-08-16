#!/usr/bin/env bash
# Pause H3BUP_vNext live placement via bridge kill-switch (disable_back=true).
# Reversible. Does NOT stop executor/accounting/CLV/daily. Does NOT create orders.
set -euo pipefail
ROOT="${1:-/home/betbot/Bets/betinasia_bot}"
RISK="$ROOT/logs/bridge_risk_params.json"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP="$ROOT/logs/bridge_risk_params.json.bak_stop_${STAMP}"

if [[ ! -f "$RISK" ]]; then
  echo "ERROR: missing $RISK" >&2
  exit 2
fi

cp -a "$RISK" "$BACKUP"
python3 - "$RISK" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
data = json.loads(p.read_text())
before = bool(data.get("disable_back"))
data["disable_back"] = True
data["_h3bup_stop_note"] = "paused_by_ops_h3bup_stop_live"
p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
print(f"disable_back: {before} -> True")
print(f"wrote {p}")
PY

echo "backup=$BACKUP"
echo "waiting 6s for bridge reload..."
sleep 6
echo "--- risk params ---"
cat "$RISK"
echo "--- recent bridge reasons (if log exists) ---"
grep -E "disabled_back|operational_disabled_back|disable_back" \
  "$ROOT/logs/executor_bridge_back.log" 2>/dev/null | tail -5 || true
echo "STOP_APPLIED"
