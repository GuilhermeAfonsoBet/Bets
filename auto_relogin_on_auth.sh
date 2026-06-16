#!/usr/bin/env bash
set -euo pipefail

# Guardiao operacional para sessao/auth:
# - Detecta sinais de NO_SESSION/NO_ROOT_SESSION_COOKIE no executor JSONL
# - Opcionalmente executa relogin e restart via comandos configuraveis
# - Aplica cooldown para evitar loops de auto-recuperacao

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOT_DIR="${BOT_DIR:-$ROOT_DIR/betinasia_bot}"
ENV_FILE_CANDIDATE="${ENV_FILE:-$BOT_DIR/.env}"

if [[ -f "$ENV_FILE_CANDIDATE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE_CANDIDATE"
  set +a
fi

EXECUTOR_JSONL="${EXECUTOR_JSONL:-$BOT_DIR/logs/executor_live.jsonl}"
LOOKBACK_MIN="${AUTH_GUARD_LOOKBACK_MIN:-20}"
COOLDOWN_SEC="${AUTH_GUARD_COOLDOWN_SEC:-900}"
MIN_ROWS="${AUTH_GUARD_MIN_ROWS:-20}"
NO_SESSION_MIN="${AUTH_GUARD_NO_SESSION_MIN:-1}"
NO_ROOT_COOKIE_MIN="${AUTH_GUARD_NO_ROOT_COOKIE_MIN:-1}"
AUTH_401_MIN="${AUTH_GUARD_AUTH401_MIN:-1}"
REQUIRE_NO_LIVE_OK="${AUTH_GUARD_REQUIRE_NO_LIVE_OK:-1}"
HEARTBEAT_ONLY_ENABLE="${AUTH_GUARD_HEARTBEAT_ONLY_ENABLE:-1}"
HEARTBEAT_ONLY_REQUIRE_DB_STALE="${AUTH_GUARD_HEARTBEAT_ONLY_REQUIRE_DB_STALE:-1}"
DRY_RUN="${AUTH_GUARD_DRY_RUN:-0}"
DB_STALE_ENABLE="${AUTH_GUARD_DB_STALE_ENABLE:-1}"
DB_STALE_MIN="${AUTH_GUARD_DB_STALE_MIN:-90}"
DB_STALE_IGNORE_COOLDOWN="${AUTH_GUARD_DB_STALE_IGNORE_COOLDOWN:-1}"

RELOGIN_CMD="${AUTH_GUARD_RELOGIN_CMD:-}"
POST_RELOGIN_CMD="${AUTH_GUARD_POST_RELOGIN_CMD:-}"

STATE_FILE="${AUTH_GUARD_STATE_FILE:-$BOT_DIR/logs/auth_guard_state.json}"
ACTIONS_LOG="${AUTH_GUARD_ACTIONS_LOG:-$BOT_DIR/logs/auth_guard_actions.jsonl}"
LOCK_DIR="${AUTH_GUARD_LOCK_DIR:-/tmp/auth_guard.lock}"

# Resolve caminhos relativos para BOT_DIR (evita depender do cwd do cron/shell).
if [[ "$EXECUTOR_JSONL" != /* ]]; then
  EXECUTOR_JSONL="$BOT_DIR/$EXECUTOR_JSONL"
fi
if [[ "$STATE_FILE" != /* ]]; then
  STATE_FILE="$BOT_DIR/$STATE_FILE"
fi
if [[ "$ACTIONS_LOG" != /* ]]; then
  ACTIONS_LOG="$BOT_DIR/$ACTIONS_LOG"
fi

for v in LOOKBACK_MIN COOLDOWN_SEC MIN_ROWS NO_SESSION_MIN NO_ROOT_COOKIE_MIN AUTH_401_MIN REQUIRE_NO_LIVE_OK HEARTBEAT_ONLY_ENABLE HEARTBEAT_ONLY_REQUIRE_DB_STALE DRY_RUN DB_STALE_ENABLE DB_STALE_MIN DB_STALE_IGNORE_COOLDOWN; do
  if ! [[ "${!v}" =~ ^[0-9]+$ ]]; then
    echo "[ERRO] $v invalido: ${!v}" >&2
    exit 2
  fi
done

mkdir -p "$(dirname "$STATE_FILE")" "$(dirname "$ACTIONS_LOG")"

if [[ ! -f "$EXECUTOR_JSONL" ]]; then
  echo "[WARN] EXECUTOR_JSONL nao encontrado: $EXECUTOR_JSONL"
  exit 0
fi

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "[INFO] Lock ativo em $LOCK_DIR; outra execucao em andamento."
  exit 0
fi
cleanup() {
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT

METRICS_JSON="$(
python3 - "$EXECUTOR_JSONL" "$LOOKBACK_MIN" <<'PY'
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

src = Path(sys.argv[1])
lookback_min = int(sys.argv[2])
cut = datetime.now(timezone.utc) - timedelta(minutes=lookback_min)

def parse_dt(v):
    s = str(v or "").strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

status = Counter()
reasons = Counter()
total = 0
live_ok = 0

for ln in src.read_text(encoding="utf-8", errors="ignore").splitlines():
    try:
        o = json.loads(ln)
    except Exception:
        continue
    req = o.get("request") if isinstance(o.get("request"), dict) else {}
    res = o.get("result") if isinstance(o.get("result"), dict) else {}
    dt = (
        parse_dt(o.get("ts"))
        or parse_dt(res.get("created_at"))
        or parse_dt(req.get("created_at"))
    )
    if dt is None or dt < cut:
        continue

    total += 1
    st = str(res.get("status") or o.get("status") or "<none>").upper()
    status[st] += 1
    if st == "LIVE_OK":
        live_ok += 1

    for key in ("reason", "skip_reason"):
        v = res.get(key)
        if v is not None and str(v).strip():
            reasons[str(v).strip()] += 1
    for key in ("reason", "skip_reason"):
        v = o.get(key)
        if v is not None and str(v).strip():
            reasons[str(v).strip()] += 1

no_session = int(status.get("NO_SESSION", 0))
heartbeat = int(status.get("HEARTBEAT", 0))
no_root_cookie = sum(c for k, c in reasons.items() if "NO_ROOT_SESSION_COOKIE" in str(k).upper())
auth_401 = sum(c for k, c in reasons.items() if "401" in str(k))

non_heartbeat_statuses = [k for k in status.keys() if k != "HEARTBEAT"]
heartbeat_only = int(total > 0 and len(non_heartbeat_statuses) == 0)

print(json.dumps({
    "total_rows": total,
    "live_ok": live_ok,
    "no_session": no_session,
    "heartbeat": heartbeat,
    "no_root_cookie": int(no_root_cookie),
    "auth_401": int(auth_401),
    "heartbeat_only": heartbeat_only,
    "status_top": status.most_common(10),
    "reasons_top": reasons.most_common(10),
}, ensure_ascii=False))
PY
)"

if [[ -z "$METRICS_JSON" ]]; then
  echo "[ERRO] Falha ao calcular metricas do auth guard."
  exit 3
fi

eval "$(
python3 - "$METRICS_JSON" <<'PY'
import json
import sys

d = json.loads(sys.argv[1])
print(f"TOTAL_ROWS={int(d.get('total_rows') or 0)}")
print(f"LIVE_OK={int(d.get('live_ok') or 0)}")
print(f"NO_SESSION={int(d.get('no_session') or 0)}")
print(f"NO_ROOT_COOKIE={int(d.get('no_root_cookie') or 0)}")
print(f"AUTH_401={int(d.get('auth_401') or 0)}")
print(f"HEARTBEAT_ONLY={int(d.get('heartbeat_only') or 0)}")
PY
)"

INCIDENT_REASONS=()
if [[ "$NO_SESSION" -ge "$NO_SESSION_MIN" && "$NO_SESSION_MIN" -gt 0 ]]; then
  INCIDENT_REASONS+=("NO_SESSION>=$NO_SESSION_MIN")
fi
if [[ "$NO_ROOT_COOKIE" -ge "$NO_ROOT_COOKIE_MIN" && "$NO_ROOT_COOKIE_MIN" -gt 0 ]]; then
  INCIDENT_REASONS+=("NO_ROOT_SESSION_COOKIE>=$NO_ROOT_COOKIE_MIN")
fi
if [[ "$AUTH_401" -ge "$AUTH_401_MIN" && "$AUTH_401_MIN" -gt 0 ]]; then
  INCIDENT_REASONS+=("AUTH_401>=$AUTH_401_MIN")
fi
HEARTBEAT_ONLY_CANDIDATE=0
if [[ "$HEARTBEAT_ONLY_ENABLE" == "1" && "$TOTAL_ROWS" -ge "$MIN_ROWS" && "$HEARTBEAT_ONLY" == "1" ]]; then
  HEARTBEAT_ONLY_CANDIDATE=1
fi
if [[ "$HEARTBEAT_ONLY_ENABLE" == "0" && "$TOTAL_ROWS" -ge "$MIN_ROWS" && "$HEARTBEAT_ONLY" == "1" ]]; then
  echo "[WARN] HEARTBEAT_ONLY detectado com total_rows=$TOTAL_ROWS, mas AUTH_GUARD_HEARTBEAT_ONLY_ENABLE=0 (sinal suprimido)."
fi
if [[ "$HEARTBEAT_ONLY_ENABLE" == "1" && "$HEARTBEAT_ONLY" == "1" && "$TOTAL_ROWS" -lt "$MIN_ROWS" ]]; then
  echo "[INFO] HEARTBEAT_ONLY observado, mas total_rows=$TOTAL_ROWS < MIN_ROWS=$MIN_ROWS (sem incidente por este criterio)."
fi
if [[ "$REQUIRE_NO_LIVE_OK" == "1" && "$LIVE_OK" -gt 0 ]]; then
  INCIDENT_REASONS=()
fi

DB_STALE_INCIDENT=0
DB_STALE_BY_LAG=0
DB_LAG_VALID=0
AUDIT_LAG_SEC=-1
BRIDGE_LAG_SEC=-1
NEED_DB_LAG_CHECK=0
if [[ "$DB_STALE_ENABLE" == "1" || ( "$HEARTBEAT_ONLY_ENABLE" == "1" && "$HEARTBEAT_ONLY_REQUIRE_DB_STALE" == "1" && "$HEARTBEAT_ONLY_CANDIDATE" == "1" ) ]]; then
  NEED_DB_LAG_CHECK=1
fi
if [[ "$NEED_DB_LAG_CHECK" == "1" && -z "${DATABASE_URL:-}" ]]; then
  echo "[WARN] DATABASE_URL ausente; sem validacao de lag de DB para DB_STALE/HEARTBEAT_ONLY."
fi
if [[ "$NEED_DB_LAG_CHECK" == "1" && -n "${DATABASE_URL:-}" ]]; then
  DB_STALE_RAW="$(
  psql "$DATABASE_URL" -At -v ON_ERROR_STOP=1 -c "
  SELECT
    COALESCE(EXTRACT(EPOCH FROM (now() - (SELECT max(audited_at) FROM betslip_audit_results))), 1e12)::bigint AS audit_lag_sec,
    COALESCE(EXTRACT(EPOCH FROM (now() - (SELECT max(created_at) FROM executor_bridge_seen))), 1e12)::bigint AS bridge_lag_sec;
  " 2>/dev/null || true
  )"
  if [[ -n "$DB_STALE_RAW" ]]; then
    AUDIT_LAG_SEC="$(echo "$DB_STALE_RAW" | awk -F'|' 'NR==1 {print $1}' | tr -d '[:space:]')"
    BRIDGE_LAG_SEC="$(echo "$DB_STALE_RAW" | awk -F'|' 'NR==1 {print $2}' | tr -d '[:space:]')"
    if [[ "$AUDIT_LAG_SEC" =~ ^[0-9]+$ && "$BRIDGE_LAG_SEC" =~ ^[0-9]+$ ]]; then
      DB_LAG_VALID=1
      DB_STALE_SEC=$((DB_STALE_MIN * 60))
      if [[ "$AUDIT_LAG_SEC" -ge "$DB_STALE_SEC" && "$BRIDGE_LAG_SEC" -ge "$DB_STALE_SEC" ]]; then
        DB_STALE_BY_LAG=1
        if [[ "$DB_STALE_ENABLE" == "1" ]]; then
          DB_STALE_INCIDENT=1
          if [[ "$REQUIRE_NO_LIVE_OK" == "1" && "$LIVE_OK" -gt 0 ]]; then
            DB_STALE_INCIDENT=0
          fi
        fi
      fi
    fi
  fi
fi

if [[ "$HEARTBEAT_ONLY_CANDIDATE" == "1" ]]; then
  if [[ "$HEARTBEAT_ONLY_REQUIRE_DB_STALE" == "1" ]]; then
    if [[ "$DB_LAG_VALID" == "1" && "$DB_STALE_BY_LAG" == "1" ]]; then
      INCIDENT_REASONS+=("HEARTBEAT_ONLY_WITH_ROWS>=$MIN_ROWS")
    elif [[ "$DB_LAG_VALID" == "1" ]]; then
      echo "[INFO] HEARTBEAT_ONLY suprimido por DB recente (audit_lag_sec=$AUDIT_LAG_SEC bridge_lag_sec=$BRIDGE_LAG_SEC threshold=${DB_STALE_MIN}m)."
    else
      echo "[WARN] HEARTBEAT_ONLY candidato suprimido: lag de DB indisponivel."
    fi
  else
    INCIDENT_REASONS+=("HEARTBEAT_ONLY_WITH_ROWS>=$MIN_ROWS")
  fi
fi

if [[ "$DB_STALE_INCIDENT" == "1" ]]; then
  INCIDENT_REASONS+=("DB_STALE>=${DB_STALE_MIN}m")
fi

INCIDENT=0
if [[ "${#INCIDENT_REASONS[@]}" -gt 0 ]]; then
  INCIDENT=1
fi

LAST_ACTION_EPOCH="$(
python3 - "$STATE_FILE" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
if not p.exists():
    print(0)
    raise SystemExit(0)
try:
    d = json.loads(p.read_text(encoding="utf-8"))
except Exception:
    print(0)
    raise SystemExit(0)
print(int(d.get("last_action_epoch") or 0))
PY
)"

NOW_EPOCH="$(date +%s)"
COOLDOWN_LEFT=0
IGNORE_COOLDOWN=0
if [[ "$LAST_ACTION_EPOCH" -gt 0 ]]; then
  NEXT_ALLOWED=$((LAST_ACTION_EPOCH + COOLDOWN_SEC))
  if [[ "$NOW_EPOCH" -lt "$NEXT_ALLOWED" ]]; then
    COOLDOWN_LEFT=$((NEXT_ALLOWED - NOW_EPOCH))
  fi
fi
if [[ "$DB_STALE_INCIDENT" == "1" && "$DB_STALE_IGNORE_COOLDOWN" == "1" ]]; then
  IGNORE_COOLDOWN=1
  COOLDOWN_LEFT=0
fi

ACTION_TAKEN=0
RELOGIN_EXIT=""
POST_EXIT=""

run_cmd() {
  local label="$1"
  local cmd="$2"
  if [[ -z "$cmd" ]]; then
    echo "[INFO] $label: comando vazio (skip)"
    return 0
  fi
  echo "[INFO] $label: $cmd"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[INFO] $label: DRY_RUN=1 (nao executado)"
    return 0
  fi
  set +e
  bash -lc "$cmd"
  local ec=$?
  set -e
  return $ec
}

if [[ "$INCIDENT" == "0" ]]; then
  echo "[INFO] Auth guard: sem incidente no recorte de ${LOOKBACK_MIN} min."
elif [[ "$COOLDOWN_LEFT" -gt 0 ]]; then
  echo "[WARN] Auth guard: incidente detectado (${INCIDENT_REASONS[*]}), mas em cooldown (${COOLDOWN_LEFT}s restantes)."
else
  if [[ -z "$RELOGIN_CMD" && -z "$POST_RELOGIN_CMD" ]]; then
    echo "[ERRO] Incidente detectado (${INCIDENT_REASONS[*]}), mas AUTH_GUARD_RELOGIN_CMD/POST_RELOGIN_CMD nao configurados." >&2
  else
    if [[ "$RELOGIN_CMD" == "true" || "$RELOGIN_CMD" == ":" ]]; then
      echo "[WARN] AUTH_GUARD_RELOGIN_CMD='$RELOGIN_CMD' parece no-op; apenas POST_RELOGIN_CMD pode surtir efeito."
    fi
    if [[ "$IGNORE_COOLDOWN" == "1" ]]; then
      echo "[WARN] DB stale incidente com cooldown ignorado por AUTH_GUARD_DB_STALE_IGNORE_COOLDOWN=1."
    fi
    ACTION_TAKEN=1
    if run_cmd "relogin_cmd" "$RELOGIN_CMD"; then
      RELOGIN_EXIT="0"
    else
      RELOGIN_EXIT="$?"
    fi
    if run_cmd "post_relogin_cmd" "$POST_RELOGIN_CMD"; then
      POST_EXIT="0"
    else
      POST_EXIT="$?"
    fi
  fi
fi

python3 - "$STATE_FILE" "$ACTIONS_LOG" "$ACTION_TAKEN" "$NOW_EPOCH" "$COOLDOWN_LEFT" "$RELOGIN_EXIT" "$POST_EXIT" "$METRICS_JSON" "${INCIDENT_REASONS[*]}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

state_file = Path(sys.argv[1])
actions_log = Path(sys.argv[2])
action_taken = int(sys.argv[3])
now_epoch = int(sys.argv[4])
cooldown_left = int(sys.argv[5])
relogin_exit = sys.argv[6]
post_exit = sys.argv[7]
metrics = json.loads(sys.argv[8])
incident_reasons = [x for x in sys.argv[9].split() if x]

ts = datetime.now(timezone.utc).isoformat()
record = {
    "ts": ts,
    "action_taken": bool(action_taken),
    "cooldown_left_sec": cooldown_left,
    "incident_reasons": incident_reasons,
    "metrics": metrics,
    "relogin_exit": relogin_exit,
    "post_relogin_exit": post_exit,
}

actions_log.parent.mkdir(parents=True, exist_ok=True)
with actions_log.open("a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")

if action_taken:
    state = {
        "last_action_epoch": now_epoch,
        "last_action_ts": ts,
        "last_record": record,
    }
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
PY

echo "[INFO] Auth guard finalizado."
echo "[INFO] metricas: total_rows=$TOTAL_ROWS live_ok=$LIVE_OK no_session=$NO_SESSION no_root_cookie=$NO_ROOT_COOKIE auth_401=$AUTH_401 heartbeat_only=$HEARTBEAT_ONLY"
if [[ "$AUDIT_LAG_SEC" =~ ^[0-9]+$ && "$BRIDGE_LAG_SEC" =~ ^[0-9]+$ ]]; then
  echo "[INFO] db_lag_sec: audit=$AUDIT_LAG_SEC bridge=$BRIDGE_LAG_SEC threshold=${DB_STALE_MIN}m"
fi
if [[ "$INCIDENT" == "1" ]]; then
  echo "[WARN] incidente_detectado=1 reasons=${INCIDENT_REASONS[*]}"
else
  echo "[INFO] incidente_detectado=0"
fi

