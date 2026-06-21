#!/usr/bin/env bash
set -u
set -o pipefail

# Snapshot rapido de andamento da versao com multiplos workers
# - foco no audit-api (pool T+0 + pool temporal)
# - inclui sinais do collector para freshness

WINDOW_MINUTES=60
AUDIT_VERSION="v4.0-api"
DB_NAME="${DB_NAME:-betinasia_bot}"
DB_USER="${DB_USER:-betbot}"
USE_SUDO_PSQL=1
AUDIT_SERVICE="${AUDIT_SERVICE:-betinasia-audit-api}"
COLLECTOR_SERVICE="${COLLECTOR_SERVICE:-betinasia-collector}"

usage() {
  cat <<'EOF'
Uso: bash check_workers_progress_v4.sh [opcoes]

Opcoes:
  --window-minutes N   Janela principal para leitura do andamento (default: 60)
  --audit-version V    Versao de auditoria (default: v4.0-api)
  --db-name N          Banco (default: betinasia_bot)
  --db-user U          Usuario psql (default: betbot)
  --no-sudo-psql       Nao usar sudo -u para psql
  --help               Ajuda
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --window-minutes) WINDOW_MINUTES="${2:-}"; shift 2 ;;
    --audit-version) AUDIT_VERSION="${2:-}"; shift 2 ;;
    --db-name) DB_NAME="${2:-}"; shift 2 ;;
    --db-user) DB_USER="${2:-}"; shift 2 ;;
    --no-sudo-psql) USE_SUDO_PSQL=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Opcao invalida: $1" >&2; usage; exit 2 ;;
  esac
done

run_psql() {
  local sql="$1"
  if [[ "$USE_SUDO_PSQL" -eq 1 ]] && [[ "$(id -un)" != "$DB_USER" ]] && command -v sudo >/dev/null 2>&1; then
    sudo -u "$DB_USER" psql "$DB_NAME" -c "$sql"
  else
    psql "$DB_NAME" -c "$sql"
  fi
}

echo "====================================================================="
echo "CHECK WORKERS V4 | $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
echo "window=${WINDOW_MINUTES} min | audit_version=${AUDIT_VERSION}"
echo "====================================================================="
echo

echo "1) STATUS DE SERVICOS"
systemctl show "$AUDIT_SERVICE" -p ActiveState -p SubState -p NRestarts --no-pager 2>/dev/null || true
systemctl show "$COLLECTOR_SERVICE" -p ActiveState -p SubState -p NRestarts --no-pager 2>/dev/null || true
echo

echo "2) ENV E DROP-IN DO AUDIT"
systemctl show "$AUDIT_SERVICE" -p Environment -p DropInPaths --no-pager 2>/dev/null || true
echo

echo "3) LOGS RECENTES (audit) - sinais de workers"
LOG_TODAY="logs/audit_api_$(date +%F).log"
if [[ -f "$LOG_TODAY" ]]; then
  tail -n 160 "$LOG_TODAY" | grep -E "Executores T\\+0 ativos|Temporal worker iniciado|\\[TEMPORAL\\]|\\[STATS\\]" || true
else
  echo "Arquivo nao encontrado: $LOG_TODAY"
fi
echo

echo "4) RESUMO OPERACIONAL DA JANELA (${WINDOW_MINUTES} min)"
run_psql "
WITH b AS (
  SELECT
    id,
    audited_at,
    status,
    hypothesis_details::jsonb -> 'telemetry' AS t
  FROM betslip_audit_results
  WHERE audit_version='${AUDIT_VERSION}'
    AND audited_at >= now() - interval '${WINDOW_MINUTES} minutes'
),
s AS (
  SELECT
    *,
    NULLIF(t ->> 'queue_wait_ms','')::numeric AS q_ms
  FROM b
)
SELECT
  COUNT(*) AS n_total,
  COUNT(*) FILTER (WHERE status='OK') AS n_ok,
  COUNT(*) FILTER (WHERE status<>'OK') AS n_fail,
  COUNT(*) FILTER (WHERE NULLIF(t ->> 'worker_id','') IS NOT NULL) AS n_t0_worker_tag,
  COUNT(*) FILTER (WHERE NULLIF(t ->> 'temporal_worker_id','') IS NOT NULL) AS n_temporal_worker_tag,
  COUNT(*) FILTER (WHERE (t ->> 'temporal_deferred')='true') AS n_temporal_deferred,
  COUNT(*) FILTER (
    WHERE (t ->> 'temporal_deferred')='true'
      AND NULLIF(t ->> 'temporal_worker_id','') IS NULL
      AND audited_at < now() - interval '2 minutes'
  ) AS n_temporal_pending_gt_2m,
  ROUND(AVG(q_ms),1) AS q_avg_ms,
  (
    SELECT ROUND((PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY q_ms))::numeric,1)
    FROM s
    WHERE q_ms IS NOT NULL
  ) AS q_p90_ms,
  (
    SELECT ROUND((PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY q_ms))::numeric,1)
    FROM s
    WHERE q_ms IS NOT NULL
  ) AS q_p95_ms,
  ROUND(AVG(NULLIF(t ->> 'queue_depth_at_enqueue','')::numeric),2) AS q_depth_enq_avg
FROM s;
"
echo

echo "5) DISTRIBUICAO POR WORKER (T+0)"
run_psql "
WITH b AS (
  SELECT hypothesis_details::jsonb -> 'telemetry' AS t
  FROM betslip_audit_results
  WHERE audit_version='${AUDIT_VERSION}'
    AND audited_at >= now() - interval '${WINDOW_MINUTES} minutes'
)
SELECT
  COALESCE(NULLIF(t ->> 'worker_id',''),'na') AS t0_worker,
  COUNT(*) AS n
FROM b
GROUP BY 1
ORDER BY n DESC, 1;
"
echo

echo "6) DISTRIBUICAO POR WORKER (TEMPORAL)"
run_psql "
WITH b AS (
  SELECT hypothesis_details::jsonb -> 'telemetry' AS t
  FROM betslip_audit_results
  WHERE audit_version='${AUDIT_VERSION}'
    AND audited_at >= now() - interval '${WINDOW_MINUTES} minutes'
)
SELECT
  COALESCE(NULLIF(t ->> 'temporal_worker_id',''),'na') AS temporal_worker,
  COUNT(*) AS n
FROM b
GROUP BY 1
ORDER BY n DESC, 1;
"
echo

echo "7) THROUGHPUT (bucket 10 min)"
run_psql "
WITH b AS (
  SELECT audited_at, status
  FROM betslip_audit_results
  WHERE audit_version='${AUDIT_VERSION}'
    AND audited_at >= now() - interval '${WINDOW_MINUTES} minutes'
)
SELECT
  date_trunc('hour', audited_at)
    + (floor(extract(minute from audited_at)/10)::int * interval '10 minutes') AS bucket_utc,
  COUNT(*) AS n_total,
  COUNT(*) FILTER (WHERE status='OK') AS n_ok
FROM b
GROUP BY 1
ORDER BY 1 DESC;
"
echo

echo "8) AMOSTRA RECENTE (ultimos 15 registros)"
run_psql "
SELECT
  id,
  audited_at,
  status,
  hypothesis_details::jsonb -> 'telemetry' ->> 'worker_id' AS t0_worker,
  hypothesis_details::jsonb -> 'telemetry' ->> 'temporal_worker_id' AS temporal_worker,
  hypothesis_details::jsonb -> 'telemetry' ->> 'temporal_deferred' AS temporal_deferred,
  hypothesis_details::jsonb -> 'telemetry' ->> 'queue_depth_at_enqueue' AS q_enq,
  hypothesis_details::jsonb -> 'telemetry' ->> 'queue_wait_ms' AS q_ms
FROM betslip_audit_results
WHERE audit_version='${AUDIT_VERSION}'
ORDER BY id DESC
LIMIT 15;
"
echo

echo "9) FRESHNESS DO COLLECTOR (sinal de coleta viva)"
run_psql "
SELECT
  now() AT TIME ZONE 'UTC' AS now_utc,
  max(scraped_at) AS last_scraped_at,
  COALESCE(EXTRACT(EPOCH FROM (now() - max(scraped_at)))::int,999999) AS freshness_sec,
  COUNT(*) FILTER (WHERE scraped_at >= now() - interval '15 minutes') AS odds_15m,
  COUNT(*) FILTER (WHERE scraped_at >= now() - interval '60 minutes') AS odds_60m
FROM best_odds_history;
"
echo

echo "====================================================================="
echo "Fim do check de andamento."
echo "====================================================================="
