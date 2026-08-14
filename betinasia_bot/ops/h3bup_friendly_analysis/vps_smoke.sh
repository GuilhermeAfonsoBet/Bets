#!/usr/bin/env bash
# Read-only VPS smoke for H3BUP Friendly analysis.
# Does NOT alter policy/stake/executor/accounting/CLV/timers/Telegram.
set -euo pipefail
ROOT="${1:-/home/betbot/Bets/betinasia_bot}"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Optional: dump league map from local Postgres (read-only)
if [[ -n "${DATABASE_URL:-}" ]]; then
  ./venv/bin/python - <<'PY' || true
import csv, os
from pathlib import Path
try:
    import psycopg2
except Exception as e:
    print("psycopg2 unavailable", e); raise SystemExit(0)
url=os.environ["DATABASE_URL"]
conn=psycopg2.connect(url); conn.set_session(readonly=True, autocommit=True)
cur=conn.cursor()
out=Path("logs/h3bup_friendly_league_map.csv")
out.parent.mkdir(parents=True, exist_ok=True)
rows=[]
try:
    cur.execute("SELECT event_id::text, league FROM betslip_audit_results WHERE event_id IS NOT NULL ORDER BY audited_at DESC NULLS LAST LIMIT 200000")
    seen=set()
    for eid, league in cur.fetchall():
        eid=str(eid or "").strip()
        if not eid or eid in seen: continue
        seen.add(eid)
        rows.append({"event_id": eid, "league": league or "", "league_name": league or "", "competition": league or ""})
except Exception as e:
    print("audit query skip", e)
with out.open("w", newline="", encoding="utf-8") as f:
    w=csv.DictWriter(f, fieldnames=["event_id","league","league_name","competition"])
    w.writeheader(); w.writerows(rows)
print("wrote", out, "n=", len(rows))
PY
fi
./venv/bin/python -m ops.h3bup_friendly_analysis.run --root "$ROOT" --n-boot 1000 --n-perm 1000
echo "FRIENDLY_SMOKE_DONE"
