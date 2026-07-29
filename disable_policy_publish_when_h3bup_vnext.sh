#!/usr/bin/env bash
set -euo pipefail

# Desabilita a publicação cron de wf_policy_current.json quando a regra
# operacional vigente é H3BUP_vNext. A regra H3BUP_vNext é aplicada no bridge
# com bypass de ligas, mas manter o publisher escrevendo a cada 15min causa
# risco operacional e confusão de auditoria.

ROOT="${ROOT:-/home/betbot/Bets}"
LOCK_FILE="$ROOT/betinasia_bot/logs/H3BUP_VNEXT_POLICY_LOCK"
WRAPPER="$ROOT/run_publish_policy_frozen.sh"

mkdir -p "$(dirname "$LOCK_FILE")"
cat > "$LOCK_FILE" <<'EOF'
policy_id=H3BUP_vNext
policy_version=H3BUP_vNext_20260629
reason=H3BUP_vNext removes league filter; do not overwrite operational policy via frozen league publisher.
EOF

if [[ -f "$WRAPPER" ]] && ! grep -q "H3BUP_VNEXT_POLICY_LOCK" "$WRAPPER"; then
  tmp="$(mktemp)"
  awk 'NR==1{print; next} NR==2{print; print ""; print "LOCK_FILE=\"/home/betbot/Bets/betinasia_bot/logs/H3BUP_VNEXT_POLICY_LOCK\""; print "if [[ -f \"$LOCK_FILE\" ]]; then"; print "  echo \"[$(date -u +%FT%TZ)] SKIP publish_policy_frozen: H3BUP_VNEXT_POLICY_LOCK present\" >> /home/betbot/Bets/betinasia_bot/logs/publish_policy_cron.ok.log"; print "  exit 0"; print "fi"; next} {print}' "$WRAPPER" > "$tmp"
  cat "$tmp" > "$WRAPPER"
  rm -f "$tmp"
  chmod +x "$WRAPPER"
fi

# Remove a entrada do cron que publica policy congelada a cada 15min.
if crontab -l >/tmp/current_cron_h3bup 2>/dev/null; then
  grep -v 'run_publish_policy_frozen.sh' /tmp/current_cron_h3bup > /tmp/new_cron_h3bup || true
  crontab /tmp/new_cron_h3bup
fi

echo "[OK] H3BUP_vNext policy lock active: $LOCK_FILE"
echo "[OK] run_publish_policy_frozen guarded and cron entry removed if present."
