#!/usr/bin/env bash
set -euo pipefail

# Diagnostico rapido de proxy + bloqueio geografico BetinAsia.
# Exit codes:
#   0 = proxy OK (ou no-proxy permitido) e sem bloqueio de pais nos logs recentes
#   1 = WARN (proxy ausente / degradado, mas login direto funciona)
#   2 = FAIL (proxy morto e/ou login_country_not_allowed)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BOT_DIR="${BOT_DIR:-$ROOT_DIR/betinasia_bot}"
ENV_FILE="${ENV_FILE:-$BOT_DIR/.env}"
EXECUTOR_JSONL="${EXECUTOR_JSONL:-$BOT_DIR/logs/executor_live.jsonl}"
LOOKBACK_MIN="${PROXY_HEALTH_LOOKBACK_MIN:-180}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

if [[ "$EXECUTOR_JSONL" != /* ]]; then
  EXECUTOR_JSONL="$BOT_DIR/$EXECUTOR_JSONL"
fi

echo "============================================================"
echo "Proxy / country health"
echo "============================================================"
echo "PROXY_SERVER=${PROXY_SERVER:-<empty>}"
echo "EXECUTOR_JSONL=$EXECUTOR_JSONL"
echo

PROXY_OK=0
DIRECT_OK=0
COUNTRY_BLOCK=0
EXIT_CODE=0

echo ">>> 1) Direct access to BetinAsia login"
if curl -sS -o /tmp/betinasia_direct_login.html -w "direct_http=%{http_code} time=%{time_total}\n" \
  --connect-timeout 10 --max-time 20 https://black.betinasia.com/login; then
  code="$(curl -sS -o /dev/null -w '%{http_code}' --connect-timeout 10 --max-time 20 https://black.betinasia.com/login || true)"
  if [[ "$code" == "200" ]]; then
    DIRECT_OK=1
  fi
else
  echo "direct_fail"
fi

echo ">>> 2) Proxy TCP/HTTP"
if [[ -z "${PROXY_SERVER:-}" ]]; then
  echo "[WARN] PROXY_SERVER vazio (modo no-proxy)."
  EXIT_CODE=1
else
  hostport="${PROXY_SERVER#*://}"
  hostport="${hostport%%/*}"
  host="${hostport%%:*}"
  port="${hostport##*:}"
  if [[ -n "$host" && -n "$port" ]]; then
    if timeout 8 bash -c "echo >/dev/tcp/${host}/${port}" 2>/dev/null; then
      echo "proxy_tcp=OK ${host}:${port}"
      PROXY_AUTH_URL="$PROXY_SERVER"
      if [[ -n "${PROXY_USERNAME:-}" && -n "${PROXY_PASSWORD:-}" ]]; then
        proto="${PROXY_SERVER%%://*}"
        PROXY_AUTH_URL="${proto}://${PROXY_USERNAME}:${PROXY_PASSWORD}@${host}:${port}"
      fi
      if curl -sS -o /tmp/betinasia_proxy_login.html -w "proxy_http=%{http_code} time=%{time_total}\n" \
        --connect-timeout 12 --max-time 25 -x "$PROXY_AUTH_URL" https://black.betinasia.com/login; then
        pcode="$(curl -sS -o /dev/null -w '%{http_code}' --connect-timeout 12 --max-time 25 -x "$PROXY_AUTH_URL" https://black.betinasia.com/login || true)"
        if [[ "$pcode" == "200" ]]; then
          PROXY_OK=1
        fi
      else
        echo "proxy_http=FAIL"
      fi
    else
      echo "proxy_tcp=FAIL ${host}:${port}"
      EXIT_CODE=2
    fi
  else
    echo "[ERRO] PROXY_SERVER invalido: $PROXY_SERVER"
    EXIT_CODE=2
  fi
fi

echo ">>> 3) Recent login_country_not_allowed in executor JSONL"
set +e
python3 - "$EXECUTOR_JSONL" "$LOOKBACK_MIN" <<'PY'
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

path = Path(sys.argv[1])
lookback = int(sys.argv[2])
cutoff = datetime.now(timezone.utc) - timedelta(minutes=lookback)
n = 0
last = None
if not path.exists():
    print(f"[WARN] jsonl ausente: {path}")
    raise SystemExit(0)
with path.open(encoding="utf-8", errors="ignore") as f:
    f.seek(0, 2)
    size = f.tell()
    f.seek(max(0, size - 2_000_000))
    f.readline()
    for line in f:
        if "login_country_not_allowed" not in line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        res = o.get("result") if isinstance(o.get("result"), dict) else {}
        ts = str(res.get("created_at") or "")
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if dt < cutoff:
            continue
        n += 1
        last = ts
print(f"country_block_hits={n} last={last}")
if n:
    raise SystemExit(3)
raise SystemExit(0)
PY
country_rc=$?
set -e
if [[ "$country_rc" == "3" ]]; then
  COUNTRY_BLOCK=1
  EXIT_CODE=2
fi

echo
echo "SUMMARY: DIRECT_OK=$DIRECT_OK PROXY_OK=$PROXY_OK COUNTRY_BLOCK=$COUNTRY_BLOCK"
if [[ "$PROXY_OK" -eq 0 && "$DIRECT_OK" -eq 1 && -z "${PROXY_SERVER:-}" ]]; then
  echo "[WARN] Sem proxy: login direto pode funcionar, mas LIVE tende a falhar com login_country_not_allowed (ex.: SG)."
  EXIT_CODE=1
fi
if [[ "$COUNTRY_BLOCK" -eq 1 ]]; then
  echo "[FAIL] Apostas bloqueadas por pais. Reconfigure PROXY_SERVER de pais permitido, limpe sessao e relogin via proxy."
fi
if [[ "$PROXY_OK" -eq 0 && -n "${PROXY_SERVER:-}" ]]; then
  echo "[FAIL] Proxy configurado mas inacessivel."
  EXIT_CODE=2
fi
exit "$EXIT_CODE"
