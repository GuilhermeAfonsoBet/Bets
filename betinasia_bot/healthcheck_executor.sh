#!/bin/bash
# ============================================================
# Health Check - Executor (dryrun/live)
# ============================================================
#
# Verifica:
# - systemd service ativo
# - endpoint /health responde (via unix socket ou http)
#
# Uso (cron):
#   */2 * * * * /home/betbot/Bets/betinasia_bot/healthcheck_executor.sh
#
# ============================================================

set -euo pipefail

SVC="betinasia-executor"
SOCK="${EXECUTOR_UNIX_SOCKET:-/tmp/betinasia-exec.sock}"
HTTP_URL="${EXECUTOR_HTTP_URL:-http://127.0.0.1:8089}"

if ! systemctl is-active --quiet "$SVC"; then
  echo "[$(date)] ALERTA: $SVC NÃO está rodando!"
  sudo systemctl restart "$SVC" || true
  exit 1
fi

OK=0
if [ -S "$SOCK" ]; then
  if curl -sS --unix-socket "$SOCK" "http://localhost/health" >/dev/null 2>&1; then
    OK=1
  fi
else
  if curl -sS "$HTTP_URL/health" >/dev/null 2>&1; then
    OK=1
  fi
fi

if [ "$OK" -ne 1 ]; then
  echo "[$(date)] ALERTA: $SVC ativo, mas /health não responde (sock=$SOCK http=$HTTP_URL)"
  sudo systemctl restart "$SVC" || true
  exit 1
fi

echo "[$(date)] OK - $SVC ativo e /health respondeu"
exit 0

