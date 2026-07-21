#!/usr/bin/env bash
set -euo pipefail

# Aplica novo PROXY_SERVER[/USER/PASS], limpa sessao BetinAsia e reinicia
# collector/audit/executor para forcar relogin pelo proxy (pais permitido).
#
# Uso:
#   PROXY_SERVER=http://HOST:PORT \
#   PROXY_USERNAME=user \
#   PROXY_PASSWORD=pass \
#   ./betinasia_bot/ops/apply_new_proxy_and_relogin.sh
#
# Ou:
#   ./betinasia_bot/ops/apply_new_proxy_and_relogin.sh --proxy-line HOST:PORT:USER:PASS

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BOT_DIR="${BOT_DIR:-$ROOT_DIR/betinasia_bot}"
ENV_FILE="${ENV_FILE:-$BOT_DIR/.env}"
SESSION_FILE="${SESSION_FILE:-$BOT_DIR/betinasia_session.json}"
RISK_FILE="${RISK_FILE:-$BOT_DIR/logs/bridge_risk_params.json}"
REENABLE_BACK="${REENABLE_BACK:-0}"

PROXY_LINE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --proxy-line)
      PROXY_LINE="${2:-}"; shift 2 ;;
    --reenable-back)
      REENABLE_BACK=1; shift ;;
    *)
      echo "[ERRO] argumento desconhecido: $1" >&2
      exit 2 ;;
  esac
done

if [[ -n "$PROXY_LINE" ]]; then
  IFS=':' read -r H P U PW <<<"$PROXY_LINE"
  if [[ -z "${H:-}" || -z "${P:-}" ]]; then
    echo "[ERRO] --proxy-line deve ser HOST:PORT ou HOST:PORT:USER:PASS" >&2
    exit 2
  fi
  PROXY_SERVER="http://${H}:${P}"
  PROXY_USERNAME="${U:-}"
  PROXY_PASSWORD="${PW:-}"
fi

if [[ -z "${PROXY_SERVER:-}" ]]; then
  echo "[ERRO] Defina PROXY_SERVER ou --proxy-line HOST:PORT:USER:PASS" >&2
  exit 2
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "[ERRO] .env nao encontrado: $ENV_FILE" >&2
  exit 2
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
cp -a "$ENV_FILE" "${ENV_FILE}.bak_before_proxy_${TS}"

python3 - "$ENV_FILE" "$PROXY_SERVER" "${PROXY_USERNAME:-}" "${PROXY_PASSWORD:-}" <<'PY'
from pathlib import Path
import sys

env_path = Path(sys.argv[1])
server, user, password = sys.argv[2], sys.argv[3], sys.argv[4]
lines = env_path.read_text(encoding="utf-8").splitlines(True)
keys = {
    "PROXY_SERVER": server,
    "PROXY_USERNAME": user,
    "PROXY_PASSWORD": password,
}
seen = set()
out = []
for line in lines:
    raw = line
    if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
        out.append(raw)
        continue
    k, _, _ = line.partition("=")
    k = k.strip()
    if k in keys:
        out.append(f"{k}={keys[k]}\n")
        seen.add(k)
    else:
        out.append(raw)
for k, v in keys.items():
    if k not in seen:
        out.append(f"{k}={v}\n")
env_path.write_text("".join(out), encoding="utf-8")
print(f"[OK] .env atualizado PROXY_SERVER={server}")
PY

if [[ -f "$SESSION_FILE" ]]; then
  cp -a "$SESSION_FILE" "${SESSION_FILE}.bak_${TS}"
  rm -f "$SESSION_FILE"
  echo "[OK] sessao removida (backup ${SESSION_FILE}.bak_${TS})"
fi

# Precheck proxy before restarting browsers
hostport="${PROXY_SERVER#*://}"
hostport="${hostport%%/*}"
host="${hostport%%:*}"
port="${hostport##*:}"
if ! timeout 10 bash -c "echo >/dev/tcp/${host}/${port}"; then
  echo "[ERRO] Proxy TCP falhou em ${host}:${port}. Abortando restart." >&2
  exit 2
fi

PROXY_AUTH_URL="$PROXY_SERVER"
if [[ -n "${PROXY_USERNAME:-}" && -n "${PROXY_PASSWORD:-}" ]]; then
  proto="${PROXY_SERVER%%://*}"
  PROXY_AUTH_URL="${proto}://${PROXY_USERNAME}:${PROXY_PASSWORD}@${host}:${port}"
fi
pcode="$(curl -sS -o /dev/null -w '%{http_code}' --connect-timeout 15 --max-time 30 -x "$PROXY_AUTH_URL" https://black.betinasia.com/login || true)"
if [[ "$pcode" != "200" ]]; then
  echo "[ERRO] Proxy HTTP falhou (code=$pcode) em black.betinasia.com/login. Abortando." >&2
  exit 2
fi
echo "[OK] proxy precheck HTTP 200"

systemctl restart betinasia-collector.service
systemctl restart betinasia-audit-ws-gate-back.service
systemctl restart betinasia-executor.service
sleep 8
systemctl is-active betinasia-collector.service betinasia-audit-ws-gate-back.service betinasia-executor.service

if [[ "$REENABLE_BACK" == "1" && -f "$RISK_FILE" ]]; then
  python3 - "$RISK_FILE" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

p = Path(sys.argv[1])
data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
data["disable_back"] = False
data["reenable_back_set_at"] = now
data["reenable_back_reason"] = "proxy restored + session relogin via apply_new_proxy_and_relogin.sh"
p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
print("[OK] disable_back=false")
PY
fi

echo "[OK] Proxy aplicado e servicos reiniciados. Monitore executor_error.log por 'Sessao valida' e ausencia de login_country_not_allowed."
echo "     Depois valide com: ./betinasia_bot/ops/check_proxy_and_country.sh"
