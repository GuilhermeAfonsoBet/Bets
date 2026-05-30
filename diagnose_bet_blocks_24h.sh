#!/usr/bin/env bash
set -euo pipefail

# Diagnostico rapido de travas de aposta (ultimas N horas).
# Foco: policy ativa, funil auditor->bridge->executor e motivos de bloqueio.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOT_DIR="${BOT_DIR:-$ROOT_DIR/betinasia_bot}"
ENV_FILE_CANDIDATE="${ENV_FILE:-$BOT_DIR/.env}"
USER_POLICY_CURRENT_RAW="${POLICY_CURRENT-__UNSET__}"
USER_EXECUTOR_JSONL_RAW="${EXECUTOR_JSONL-__UNSET__}"

if [[ -z "${DATABASE_URL:-}" && -f "$ENV_FILE_CANDIDATE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE_CANDIDATE"
  set +a
fi

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "[ERRO] DATABASE_URL nao definido (nem via ambiente, nem via .env)." >&2
  exit 2
fi

LOOKBACK_HOURS="${LOOKBACK_HOURS:-24}"
if ! [[ "$LOOKBACK_HOURS" =~ ^[0-9]+$ ]]; then
  echo "[ERRO] LOOKBACK_HOURS invalido: $LOOKBACK_HOURS" >&2
  exit 2
fi

if [[ "$USER_POLICY_CURRENT_RAW" != "__UNSET__" ]]; then
  POLICY_CURRENT="$USER_POLICY_CURRENT_RAW"
else
  POLICY_CURRENT="${POLICY_CURRENT:-$BOT_DIR/logs/wf_policy_current.json}"
fi
if [[ "$USER_EXECUTOR_JSONL_RAW" != "__UNSET__" ]]; then
  EXECUTOR_JSONL="$USER_EXECUTOR_JSONL_RAW"
else
  EXECUTOR_JSONL="${EXECUTOR_JSONL:-$BOT_DIR/logs/executor_live.jsonl}"
fi

# Se vier caminho relativo (comum em .env), ancora no BOT_DIR para evitar ambiguidades.
if [[ "$POLICY_CURRENT" != /* ]]; then
  POLICY_CURRENT="$BOT_DIR/$POLICY_CURRENT"
fi
if [[ "$EXECUTOR_JSONL" != /* ]]; then
  EXECUTOR_JSONL="$BOT_DIR/$EXECUTOR_JSONL"
fi

echo "============================================================"
echo "Diagnostico de bloqueio de apostas (${LOOKBACK_HOURS}h)"
echo "============================================================"
echo "DATABASE_URL: [definido]"
echo "POLICY_CURRENT: $POLICY_CURRENT"
echo "EXECUTOR_JSONL: $EXECUTOR_JSONL"
echo

echo ">>> 1) Policy ativa (resumo)"
python3 - "$POLICY_CURRENT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print(f"[WARN] policy nao encontrada: {path}")
    raise SystemExit(0)

try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception as e:
    print(f"[ERRO] falha ao ler policy JSON: {e}")
    raise SystemExit(0)

wf = data.get("wf") if isinstance(data, dict) else {}
steps = data.get("steps") if isinstance(data, dict) else []
last = steps[-1] if isinstance(steps, list) and steps else {}

active_keys = last.get("active_keys") if isinstance(last, dict) else []
approved = last.get("approved_leagues") if isinstance(last, dict) else []

print(f"generated_at: {data.get('generated_at')}")
print(f"wf.selection_mode: {wf.get('selection_mode')}")
print(f"wf.pre_activation_mode: {wf.get('pre_activation_mode')}")
print(f"wf.roi_min_activate: {wf.get('roi_min_activate')}")
print(f"wf.backpre_slip_field: {wf.get('backpre_slip_field')}")
print(f"wf.backpre_slip_max: {wf.get('backpre_slip_max')}")
print(f"steps: {len(steps) if isinstance(steps, list) else 0}")
print(f"active_n(last): {last.get('active_n') if isinstance(last, dict) else None}")
print(f"n_ev_elig_pre(last): {last.get('n_ev_elig_pre') if isinstance(last, dict) else None}")

if isinstance(active_keys, list):
    print(f"active_keys_sample: {active_keys[:10]}")
if isinstance(approved, list):
    print(f"approved_leagues_sample: {approved[:10]}")
PY
echo

echo ">>> 2) Auditor (betslip_audit_results) - status nas ultimas ${LOOKBACK_HOURS}h"
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -c "
SELECT
  UPPER(COALESCE(status, '<null>')) AS status,
  COUNT(*) AS n
FROM betslip_audit_results
WHERE audited_at >= now() - interval '${LOOKBACK_HOURS} hours'
GROUP BY 1
ORDER BY n DESC, status;
"
echo

echo ">>> 3) Auditor - top ligas com status=OK nas ultimas ${LOOKBACK_HOURS}h"
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -c "
SELECT
  COALESCE(NULLIF(league,''), '<sem_liga>') AS league,
  COUNT(*) AS n
FROM betslip_audit_results
WHERE audited_at >= now() - interval '${LOOKBACK_HOURS} hours'
  AND UPPER(COALESCE(status,'')) = 'OK'
GROUP BY 1
ORDER BY n DESC, league
LIMIT 25;
"
echo

ACTIVE_LEAGUES_SQL="$(
python3 - "$POLICY_CURRENT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(0)
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(0)
steps = data.get("steps") if isinstance(data, dict) else []
last = steps[-1] if isinstance(steps, list) and steps else {}
approved = last.get("approved_leagues") if isinstance(last, dict) else []
if not isinstance(approved, list):
    raise SystemExit(0)

vals = []
for lg in approved:
    s = str(lg or "").strip()
    if not s:
        continue
    vals.append("'" + s.replace("'", "''") + "'")

print(",".join(vals))
PY
)"

if [[ -n "$ACTIVE_LEAGUES_SQL" ]]; then
  echo ">>> 3.1) Cobertura dos OK vs ligas ativas na policy"
  psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -c "
  WITH active AS (
    SELECT lower(trim(x)) AS league
    FROM unnest(ARRAY[$ACTIVE_LEAGUES_SQL]::text[]) AS t(x)
  ),
  src AS (
    SELECT lower(trim(COALESCE(league,''))) AS league
    FROM betslip_audit_results
    WHERE audited_at >= now() - interval '${LOOKBACK_HOURS} hours'
      AND UPPER(COALESCE(status,'')) = 'OK'
  )
  SELECT
    COUNT(*) FILTER (WHERE league <> '' AND league IN (SELECT league FROM active)) AS ok_in_active,
    COUNT(*) FILTER (WHERE league <> '' AND league NOT IN (SELECT league FROM active)) AS ok_outside_active,
    COUNT(*) FILTER (WHERE league = '') AS ok_sem_liga,
    COUNT(*) AS ok_total
  FROM src;
  "
  echo

  echo ">>> 3.2) Top ligas OK fora da policy ativa"
  psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -c "
  WITH active AS (
    SELECT lower(trim(x)) AS league
    FROM unnest(ARRAY[$ACTIVE_LEAGUES_SQL]::text[]) AS t(x)
  ),
  src AS (
    SELECT COALESCE(NULLIF(league,''), '<sem_liga>') AS league_raw,
           lower(trim(COALESCE(league,''))) AS league_norm
    FROM betslip_audit_results
    WHERE audited_at >= now() - interval '${LOOKBACK_HOURS} hours'
      AND UPPER(COALESCE(status,'')) = 'OK'
  )
  SELECT league_raw AS league, COUNT(*) AS n
  FROM src
  WHERE league_norm <> ''
    AND league_norm NOT IN (SELECT league FROM active)
  GROUP BY 1
  ORDER BY n DESC, league
  LIMIT 25;
  "
  echo
fi

echo ">>> 4) Bridge - motivos de skip/bloqueio (se tabela existir)"
BRIDGE_EXISTS="$(psql "$DATABASE_URL" -At -v ON_ERROR_STOP=1 -c "SELECT to_regclass('public.executor_bridge_seen') IS NOT NULL;")"
if [[ "$BRIDGE_EXISTS" == "t" ]]; then
  BRIDGE_COLS="$(psql "$DATABASE_URL" -At -v ON_ERROR_STOP=1 -c "SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='executor_bridge_seen';")"

  has_col() {
    local name="$1"
    while IFS= read -r c; do
      [[ "$c" == "$name" ]] && return 0
    done <<< "$BRIDGE_COLS"
    return 1
  }

  TS_COL=""
  for c in created_at seen_at ts updated_at audited_at processed_at inserted_at; do
    if has_col "$c"; then TS_COL="$c"; break; fi
  done
  REASON_COL=""
  for c in skip_reason reason status_reason block_reason decision_reason outcome_reason skip_reason_raw; do
    if has_col "$c"; then REASON_COL="$c"; break; fi
  done
  LEAGUE_COL=""
  for c in league competition league_name league_text tournament; do
    if has_col "$c"; then LEAGUE_COL="$c"; break; fi
  done

  echo "bridge.ts_col=$TS_COL bridge.reason_col=$REASON_COL bridge.league_col=$LEAGUE_COL"
  if [[ -n "$TS_COL" && -n "$REASON_COL" ]]; then
    psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -c "
    SELECT
      COALESCE(NULLIF(${REASON_COL}::text,''), '<none>') AS reason,
      COUNT(*) AS n
    FROM executor_bridge_seen
    WHERE ${TS_COL} >= now() - interval '${LOOKBACK_HOURS} hours'
    GROUP BY 1
    ORDER BY n DESC, reason
    LIMIT 40;
    "
    echo
    if [[ -n "$LEAGUE_COL" ]]; then
      psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -c "
      SELECT
        COALESCE(NULLIF(${LEAGUE_COL}::text,''), '<sem_liga>') AS league,
        COUNT(*) AS n
      FROM executor_bridge_seen
      WHERE ${TS_COL} >= now() - interval '${LOOKBACK_HOURS} hours'
        AND COALESCE(NULLIF(${REASON_COL}::text,''), '<none>') = 'not_active'
      GROUP BY 1
      ORDER BY n DESC, league
      LIMIT 25;
      "
    fi
  else
    echo "[WARN] Colunas explicitas de motivo nao encontradas. Tentando fallback por JSON/row dump..."
    if [[ -n "$TS_COL" ]]; then
      TMP_BRIDGE_JSON="$(mktemp /tmp/bridge_rows.XXXXXX.jsonl)"
      psql "$DATABASE_URL" -At -v ON_ERROR_STOP=1 -c "
      SELECT row_to_json(t)::text
      FROM (
        SELECT *
        FROM executor_bridge_seen
        WHERE ${TS_COL} >= now() - interval '${LOOKBACK_HOURS} hours'
        ORDER BY ${TS_COL} DESC
        LIMIT 8000
      ) t;
      " > "$TMP_BRIDGE_JSON"

      python3 - "$TMP_BRIDGE_JSON" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

src = Path(sys.argv[1])
if not src.exists():
    print("[WARN] fallback bridge: sem dados.")
    raise SystemExit(0)

reason_keys = [
    "skip_reason", "reason", "status_reason", "block_reason",
    "decision_reason", "outcome_reason", "decision", "status"
]
league_keys = ["league", "competition", "league_name", "league_text", "tournament"]

reasons = Counter()
not_active_leagues = Counter()
rows = 0

def get_val(obj, keys):
    if not isinstance(obj, dict):
        return None
    for k in keys:
        v = obj.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None

for ln in src.read_text(encoding="utf-8", errors="ignore").splitlines():
    ln = ln.strip()
    if not ln:
        continue
    try:
        row = json.loads(ln)
    except Exception:
        continue
    rows += 1

    reason = get_val(row, reason_keys)
    league = get_val(row, league_keys) or "<sem_liga>"

    if reason is None:
        for v in row.values():
            if isinstance(v, dict):
                reason = get_val(v, reason_keys) or reason
                if league == "<sem_liga>":
                    league = get_val(v, league_keys) or league
            elif isinstance(v, str):
                s = v.strip()
                if s.startswith("{") and s.endswith("}"):
                    try:
                        vv = json.loads(s)
                    except Exception:
                        continue
                    if isinstance(vv, dict):
                        reason = get_val(vv, reason_keys) or reason
                        if league == "<sem_liga>":
                            league = get_val(vv, league_keys) or league

    if reason:
        reasons[reason] += 1
        if reason == "not_active":
            not_active_leagues[league] += 1

print(f"fallback_rows={rows}")
print(f"fallback_reasons_top={reasons.most_common(40)}")
if not_active_leagues:
    print(f"fallback_not_active_leagues_top={not_active_leagues.most_common(25)}")
PY
      rm -f "$TMP_BRIDGE_JSON"
    else
      echo "[WARN] Nao foi possivel detectar coluna temporal na executor_bridge_seen."
    fi
  fi
else
  echo "[WARN] Tabela public.executor_bridge_seen nao existe."
fi
echo

echo ">>> 5) Executor JSONL (status/motivos) nas ultimas ${LOOKBACK_HOURS}h"
python3 - "$EXECUTOR_JSONL" "$LOOKBACK_HOURS" <<'PY'
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

src = Path(sys.argv[1])
hours = int(sys.argv[2])

if not src.exists():
    print(f"[WARN] JSONL nao encontrado: {src}")
    raise SystemExit(0)

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

cut = datetime.now(timezone.utc) - timedelta(hours=hours)
status = Counter()
reasons = Counter()
live_ok = 0
total = 0

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
    r = (
        res.get("reason")
        or res.get("skip_reason")
        or o.get("reason")
        or o.get("skip_reason")
    )
    if r is not None and str(r).strip():
        reasons[str(r).strip()] += 1

print(f"linhas_no_recorte={total}")
print(f"status_top={status.most_common(20)}")
print(f"live_ok={live_ok}")
print(f"reasons_top={reasons.most_common(20)}")
PY
echo

echo "[OK] Diagnostico concluido."
