#!/usr/bin/env bash
set -u
set -o pipefail

# Healthcheck v4 - BetinAsia (collector + API audit)
# - Verifica servicos
# - Verifica erros recentes de journal
# - Verifica frescor dos JSONL de telemetria
# - Verifica cobertura no banco (lay_t0 / back_temporal / lay_temporal / telemetry)
# - Imprime resumo PASS/WARN/FAIL

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR" || exit 2

WINDOW_MINUTES=30
JOURNAL_MINUTES=15
TELEMETRY_MAX_AGE_SEC=1200
MIN_AUDITS_FOR_STRICT_TEMPORAL=5

DB_NAME="${DB_NAME:-betinasia_bot}"
DB_USER="${DB_USER:-betbot}"
USE_SUDO_PSQL=1
STRICT_MODE=0

COLLECTOR_SERVICE="${COLLECTOR_SERVICE:-betinasia-collector}"
AUDIT_SERVICE="${AUDIT_SERVICE:-betinasia-audit-api}"

AUDIT_TELEMETRY_FILE="${AUDIT_TELEMETRY_FILE:-logs/audit_api_telemetry.jsonl}"
COLLECTOR_TELEMETRY_FILE="${COLLECTOR_TELEMETRY_FILE:-logs/collector_telemetry.jsonl}"

PASS_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0

is_tty=0
if [[ -t 1 ]]; then
  is_tty=1
fi

color() {
  local code="$1"
  local text="$2"
  if [[ "$is_tty" -eq 1 ]]; then
    printf "\033[%sm%s\033[0m" "$code" "$text"
  else
    printf "%s" "$text"
  fi
}

report_pass() {
  PASS_COUNT=$((PASS_COUNT + 1))
  printf "%s %s\n" "$(color "32" "[PASS]")" "$1"
}

report_warn() {
  WARN_COUNT=$((WARN_COUNT + 1))
  printf "%s %s\n" "$(color "33" "[WARN]")" "$1"
}

report_fail() {
  FAIL_COUNT=$((FAIL_COUNT + 1))
  printf "%s %s\n" "$(color "31" "[FAIL]")" "$1"
}

usage() {
  cat <<'EOF'
Uso: bash healthcheck_v4.sh [opcoes]

Opcoes:
  --strict                         Trata WARN como falha no exit code
  --window-minutes N              Janela de auditoria no banco (default: 30)
  --journal-minutes N             Janela de logs systemd (default: 15)
  --telemetry-max-age-sec N       Idade maxima aceitavel do ultimo JSONL (default: 1200)
  --db-name NOME                  Banco (default: betinasia_bot)
  --db-user USUARIO               Usuario para psql (default: betbot)
  --no-sudo-psql                  Nao usar sudo -u no psql
  --help                          Mostra ajuda

Variaveis opcionais:
  COLLECTOR_SERVICE, AUDIT_SERVICE
  AUDIT_TELEMETRY_FILE, COLLECTOR_TELEMETRY_FILE
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --strict)
      STRICT_MODE=1
      shift
      ;;
    --window-minutes)
      WINDOW_MINUTES="${2:-}"
      shift 2
      ;;
    --journal-minutes)
      JOURNAL_MINUTES="${2:-}"
      shift 2
      ;;
    --telemetry-max-age-sec)
      TELEMETRY_MAX_AGE_SEC="${2:-}"
      shift 2
      ;;
    --db-name)
      DB_NAME="${2:-}"
      shift 2
      ;;
    --db-user)
      DB_USER="${2:-}"
      shift 2
      ;;
    --no-sudo-psql)
      USE_SUDO_PSQL=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Opcao invalida: $1"
      usage
      exit 2
      ;;
  esac
done

now_utc="$(date -u +'%Y-%m-%d %H:%M:%S UTC')"
echo "====================================================================="
echo "HEALTHCHECK V4 - ${now_utc}"
echo "Servico collector: ${COLLECTOR_SERVICE}"
echo "Servico audit API: ${AUDIT_SERVICE}"
echo "Janela DB: ${WINDOW_MINUTES} min | Janela journal: ${JOURNAL_MINUTES} min"
echo "====================================================================="

run_psql() {
  local sql="$1"
  if [[ "$USE_SUDO_PSQL" -eq 1 ]] && [[ "$(id -un)" != "$DB_USER" ]] && command -v sudo >/dev/null 2>&1; then
    sudo -u "$DB_USER" psql "$DB_NAME" -tA -c "$sql"
  else
    psql "$DB_NAME" -tA -c "$sql"
  fi
}

service_state() {
  local svc="$1"
  local active sub restarts
  active="$(systemctl show "$svc" -p ActiveState --value 2>/dev/null || true)"
  sub="$(systemctl show "$svc" -p SubState --value 2>/dev/null || true)"
  restarts="$(systemctl show "$svc" -p NRestarts --value 2>/dev/null || true)"
  echo "${active:-unknown}|${sub:-unknown}|${restarts:-unknown}"
}

count_journal_errors() {
  local svc="$1"
  local raw cnt
  raw="$(journalctl -u "$svc" --since "-${JOURNAL_MINUTES} min" --no-pager -l 2>/dev/null || true)"
  cnt="$(printf "%s" "$raw" | grep -E -c "Erro na coleta|ERROR|Traceback|UndefinedColumnError|ERR_TUNNEL|Failed with result|Exception" || true)"
  echo "${cnt:-0}"
}

telemetry_file_info() {
  local file="$1"
  python3 - "$file" <<'PY'
import json
import datetime as dt
from pathlib import Path
import sys

p = Path(sys.argv[1])
if not p.exists():
    print("MISSING|0|999999")
    raise SystemExit(0)

lines = [ln.strip() for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
if not lines:
    print("EMPTY|0|999999")
    raise SystemExit(0)

line_count = len(lines)
last = lines[-1]
try:
    payload = json.loads(last)
except Exception:
    print(f"INVALID_JSON|{line_count}|999999")
    raise SystemExit(0)

ts = payload.get("ts_utc") or payload.get("timestamp") or payload.get("ts")
if not ts:
    print(f"NO_TS|{line_count}|999999")
    raise SystemExit(0)

try:
    if isinstance(ts, str) and ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(ts)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    age = int((dt.datetime.now(dt.timezone.utc) - parsed).total_seconds())
    print(f"OK|{line_count}|{age}")
except Exception:
    print(f"BAD_TS|{line_count}|999999")
PY
}

echo
echo "1) Servicos"
collector_state="$(service_state "$COLLECTOR_SERVICE")"
audit_state="$(service_state "$AUDIT_SERVICE")"

IFS='|' read -r c_active c_sub c_restarts <<<"$collector_state"
IFS='|' read -r a_active a_sub a_restarts <<<"$audit_state"

if [[ "$c_active" == "active" && "$c_sub" == "running" ]]; then
  report_pass "Collector ativo (${c_active}/${c_sub}, restarts=${c_restarts})"
else
  report_fail "Collector fora do esperado (${c_active}/${c_sub}, restarts=${c_restarts})"
fi

if [[ "$a_active" == "active" && "$a_sub" == "running" ]]; then
  report_pass "Audit API ativo (${a_active}/${a_sub}, restarts=${a_restarts})"
else
  report_fail "Audit API fora do esperado (${a_active}/${a_sub}, restarts=${a_restarts})"
fi

echo
echo "2) Erros recentes no journal (${JOURNAL_MINUTES} min)"
c_err="$(count_journal_errors "$COLLECTOR_SERVICE")"
a_err="$(count_journal_errors "$AUDIT_SERVICE")"

if [[ "$c_err" -eq 0 ]]; then
  report_pass "Collector sem erros recentes"
else
  report_warn "Collector com ${c_err} ocorrencia(s) de erro recente"
fi

if [[ "$a_err" -eq 0 ]]; then
  report_pass "Audit API sem erros recentes"
else
  report_warn "Audit API com ${a_err} ocorrencia(s) de erro recente"
fi

echo
echo "3) Telemetria JSONL"
audit_info="$(telemetry_file_info "$AUDIT_TELEMETRY_FILE")"
collector_info="$(telemetry_file_info "$COLLECTOR_TELEMETRY_FILE")"

IFS='|' read -r audit_status audit_lines audit_age <<<"$audit_info"
IFS='|' read -r collector_status collector_lines collector_age <<<"$collector_info"

if [[ "$audit_status" == "OK" ]]; then
  if [[ "$audit_age" -le "$TELEMETRY_MAX_AGE_SEC" ]]; then
    report_pass "audit_api_telemetry.jsonl atualizado (linhas=${audit_lines}, age=${audit_age}s)"
  else
    report_warn "audit_api_telemetry.jsonl antigo (linhas=${audit_lines}, age=${audit_age}s)"
  fi
else
  report_fail "audit_api_telemetry.jsonl invalido/ausente (${audit_status})"
fi

if [[ "$collector_status" == "OK" ]]; then
  if [[ "$collector_age" -le "$TELEMETRY_MAX_AGE_SEC" ]]; then
    report_pass "collector_telemetry.jsonl atualizado (linhas=${collector_lines}, age=${collector_age}s)"
  else
    report_warn "collector_telemetry.jsonl antigo (linhas=${collector_lines}, age=${collector_age}s)"
  fi
else
  report_fail "collector_telemetry.jsonl invalido/ausente (${collector_status})"
fi

echo
echo "4) Banco - cobertura da auditoria (${WINDOW_MINUTES} min)"

read -r -d '' SQL_COVERAGE <<SQL || true
WITH base AS (
  SELECT *
  FROM betslip_audit_results
  WHERE audit_version='v4.0-api'
    AND audited_at >= now() - interval '${WINDOW_MINUTES} minutes'
)
SELECT
  COUNT(*) AS total_30m,
  COUNT(*) FILTER (WHERE status='OK') AS ok_30m,
  COUNT(*) FILTER (WHERE (hypothesis_details::jsonb -> 'lay') IS NOT NULL) AS lay_t0_30m,
  COUNT(*) FILTER (WHERE (hypothesis_details::jsonb -> 'temporal') IS NOT NULL) AS back_temporal_30m,
  COUNT(*) FILTER (WHERE (hypothesis_details::jsonb -> 'lay_temporal') IS NOT NULL) AS lay_temporal_30m,
  COUNT(*) FILTER (WHERE (hypothesis_details::jsonb -> 'telemetry') IS NOT NULL) AS telemetry_30m
FROM base;
SQL

coverage_ok=0
coverage_raw="$(run_psql "$SQL_COVERAGE" 2>/tmp/healthcheck_v4_psql.err || true)"
if [[ -n "$coverage_raw" ]] && [[ "$coverage_raw" == *"|"* ]]; then
  coverage_ok=1
  IFS='|' read -r total_30m ok_30m lay_t0_30m back_temporal_30m lay_temporal_30m telemetry_30m <<<"$coverage_raw"
  echo "   total=${total_30m} ok=${ok_30m} lay_t0=${lay_t0_30m} back_temporal=${back_temporal_30m} lay_temporal=${lay_temporal_30m} telemetry=${telemetry_30m}"

  if [[ "$total_30m" -gt 0 ]]; then
    report_pass "Ha auditorias recentes na janela"
  else
    report_warn "Sem auditorias recentes na janela"
  fi

  if [[ "$telemetry_30m" -gt 0 ]]; then
    report_pass "Telemetria persistida no banco"
  else
    report_fail "Telemetria ausente no banco na janela"
  fi

  if [[ "$lay_t0_30m" -gt 0 ]]; then
    report_pass "Lay T+0 presente"
  else
    if [[ "$total_30m" -ge "$MIN_AUDITS_FOR_STRICT_TEMPORAL" ]]; then
      report_warn "Lay T+0 zerado apesar de volume recente"
    else
      report_warn "Lay T+0 zerado (amostra pequena)"
    fi
  fi

  if [[ "$back_temporal_30m" -gt 0 ]]; then
    report_pass "Back temporal presente"
  else
    if [[ "$total_30m" -ge "$MIN_AUDITS_FOR_STRICT_TEMPORAL" ]]; then
      report_warn "Back temporal zerado com volume recente"
    else
      report_warn "Back temporal zerado (amostra pequena)"
    fi
  fi

  if [[ "$lay_temporal_30m" -gt 0 ]]; then
    report_pass "Lay temporal presente"
  else
    if [[ "$total_30m" -ge "$MIN_AUDITS_FOR_STRICT_TEMPORAL" ]]; then
      report_warn "Lay temporal zerado com volume recente"
    else
      report_warn "Lay temporal zerado (amostra pequena)"
    fi
  fi
else
  report_fail "Falha ao consultar cobertura no banco (ver /tmp/healthcheck_v4_psql.err)"
fi

echo
echo "5) Banco - tempos medios por etapa (24h)"
read -r -d '' SQL_TIMES <<'SQL' || true
SELECT
  ROUND(AVG((hypothesis_details::jsonb -> 'telemetry' ->> 'queue_wait_ms')::numeric),1),
  ROUND(AVG((hypothesis_details::jsonb -> 'telemetry' ->> 'back_post_ms')::numeric),1),
  ROUND(AVG((hypothesis_details::jsonb -> 'telemetry' ->> 'back_pmm_ms')::numeric),1),
  ROUND(AVG((hypothesis_details::jsonb -> 'telemetry' ->> 'lay_post_ms')::numeric),1),
  ROUND(AVG((hypothesis_details::jsonb -> 'telemetry' ->> 'lay_pmm_ms')::numeric),1),
  ROUND(AVG((hypothesis_details::jsonb -> 'telemetry' ->> 'temporal_total_ms')::numeric),1),
  ROUND(AVG((hypothesis_details::jsonb -> 'telemetry' ->> 'db_save_ms')::numeric),1),
  ROUND(AVG((hypothesis_details::jsonb -> 'telemetry' ->> 'pipeline_total_ms')::numeric),1)
FROM betslip_audit_results
WHERE audit_version='v4.0-api'
  AND audited_at >= now() - interval '24 hours'
  AND (hypothesis_details::jsonb -> 'telemetry') IS NOT NULL;
SQL

times_raw="$(run_psql "$SQL_TIMES" 2>/tmp/healthcheck_v4_psql.err || true)"
if [[ -n "$times_raw" ]] && [[ "$times_raw" == *"|"* ]]; then
  IFS='|' read -r t_queue t_back_post t_back_pmm t_lay_post t_lay_pmm t_temporal t_db t_pipeline <<<"$times_raw"
  echo "   queue_wait_ms=${t_queue:-null}"
  echo "   back_post_ms=${t_back_post:-null} | back_pmm_ms=${t_back_pmm:-null}"
  echo "   lay_post_ms=${t_lay_post:-null} | lay_pmm_ms=${t_lay_pmm:-null}"
  echo "   temporal_total_ms=${t_temporal:-null} | db_save_ms=${t_db:-null} | pipeline_total_ms=${t_pipeline:-null}"

  if [[ -n "${t_pipeline}" ]]; then
    report_pass "Tempos por etapa disponiveis"
  else
    report_warn "Sem dados suficientes de tempos por etapa"
  fi
else
  report_warn "Falha ao consultar tempos por etapa (ver /tmp/healthcheck_v4_psql.err)"
fi

echo
echo "====================================================================="
echo "RESUMO FINAL"
echo "  PASS: ${PASS_COUNT}"
echo "  WARN: ${WARN_COUNT}"
echo "  FAIL: ${FAIL_COUNT}"
echo "====================================================================="

if [[ "$FAIL_COUNT" -gt 0 ]]; then
  exit 2
fi

if [[ "$STRICT_MODE" -eq 1 ]] && [[ "$WARN_COUNT" -gt 0 ]]; then
  exit 1
fi

exit 0
