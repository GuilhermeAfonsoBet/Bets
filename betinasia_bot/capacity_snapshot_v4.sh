#!/usr/bin/env bash
set -u
set -o pipefail

# Snapshot de capacidade e sanidade operacional (v4.0-api)
# - Serviços e freshness
# - Cobertura de captura (Back/Lay/temporal/telemetria)
# - N por recortes (regime, direção, mercado, bucket BS vs WS)
# - N por dia + projeção mensal
# - N por tabela de hipóteses (H1/H3/H3b/H6), se existirem

WINDOW_HOURS=24
LOOKBACK_DAYS=7
AUDIT_VERSION="v4.0-api"
DB_NAME="${DB_NAME:-betinasia_bot}"
DB_USER="${DB_USER:-betbot}"
USE_SUDO_PSQL=1

COLLECTOR_SERVICE="${COLLECTOR_SERVICE:-betinasia-collector}"
AUDIT_SERVICE="${AUDIT_SERVICE:-betinasia-audit-api}"

usage() {
  cat <<'EOF'
Uso: bash capacity_snapshot_v4.sh [opcoes]

Opcoes:
  --window-hours N        Janela para snapshot principal (default: 24)
  --lookback-days N       Janela para N/dia e projecao mensal (default: 7)
  --audit-version V       Versao auditada (default: v4.0-api)
  --db-name N             Banco (default: betinasia_bot)
  --db-user U             Usuario psql (default: betbot)
  --no-sudo-psql          Nao usar sudo -u para psql
  --help                  Ajuda
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --window-hours)
      WINDOW_HOURS="${2:-}"; shift 2 ;;
    --lookback-days)
      LOOKBACK_DAYS="${2:-}"; shift 2 ;;
    --audit-version)
      AUDIT_VERSION="${2:-}"; shift 2 ;;
    --db-name)
      DB_NAME="${2:-}"; shift 2 ;;
    --db-user)
      DB_USER="${2:-}"; shift 2 ;;
    --no-sudo-psql)
      USE_SUDO_PSQL=0; shift ;;
    --help|-h)
      usage; exit 0 ;;
    *)
      echo "Opcao invalida: $1" >&2
      usage
      exit 2 ;;
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

run_psql_raw() {
  local sql="$1"
  if [[ "$USE_SUDO_PSQL" -eq 1 ]] && [[ "$(id -un)" != "$DB_USER" ]] && command -v sudo >/dev/null 2>&1; then
    sudo -u "$DB_USER" psql "$DB_NAME" -tA -c "$sql"
  else
    psql "$DB_NAME" -tA -c "$sql"
  fi
}

table_exists() {
  local t="$1"
  local e
  e="$(run_psql_raw "SELECT to_regclass('$t') IS NOT NULL;" 2>/dev/null || echo "f")"
  [[ "$e" == "t" ]]
}

echo "====================================================================="
echo "CAPACITY SNAPSHOT V4 | $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
echo "audit_version=${AUDIT_VERSION} | window=${WINDOW_HOURS}h | lookback=${LOOKBACK_DAYS}d"
echo "====================================================================="
echo

echo "1) STATUS DE SERVICOS"
systemctl show "$COLLECTOR_SERVICE" -p ActiveState -p SubState -p NRestarts --no-pager 2>/dev/null || true
systemctl show "$AUDIT_SERVICE" -p ActiveState -p SubState -p NRestarts --no-pager 2>/dev/null || true
echo

echo "2) FRESHNESS DE INGESTAO (collector + audit)"
run_psql "
SELECT
  now() AT TIME ZONE 'UTC' AS now_utc,
  (SELECT max(scraped_at) FROM best_odds_history) AS last_scraped_at,
  COALESCE(EXTRACT(EPOCH FROM (now() - (SELECT max(scraped_at) FROM best_odds_history)))::int, 999999) AS freshness_collector_sec,
  (SELECT max(audited_at) FROM betslip_audit_results WHERE audit_version='${AUDIT_VERSION}') AS last_audited_at,
  COALESCE(EXTRACT(EPOCH FROM (now() - (SELECT max(audited_at) FROM betslip_audit_results WHERE audit_version='${AUDIT_VERSION}')))::int, 999999) AS freshness_audit_sec;
"
echo

echo "3) COBERTURA DE CAPTURA (janela ${WINDOW_HOURS}h)"
run_psql "
WITH base AS (
  SELECT *
  FROM betslip_audit_results
  WHERE audit_version='${AUDIT_VERSION}'
    AND audited_at >= now() - interval '${WINDOW_HOURS} hours'
)
SELECT
  COUNT(*) AS n_total,
  COUNT(*) FILTER (WHERE status='OK') AS n_ok,
  COUNT(*) FILTER (WHERE status<>'OK') AS n_fail,
  COUNT(*) FILTER (WHERE is_live IS TRUE) AS n_in_match,
  COUNT(*) FILTER (WHERE is_live IS FALSE OR is_live IS NULL) AS n_pre_match,
  COUNT(*) FILTER (WHERE (hypothesis_details::jsonb -> 'lay') IS NOT NULL) AS n_lay_t0,
  COUNT(*) FILTER (WHERE (hypothesis_details::jsonb -> 'temporal') IS NOT NULL) AS n_back_temporal,
  COUNT(*) FILTER (WHERE (hypothesis_details::jsonb -> 'lay_temporal') IS NOT NULL) AS n_lay_temporal,
  COUNT(*) FILTER (WHERE (hypothesis_details::jsonb -> 'telemetry') IS NOT NULL) AS n_telemetry
FROM base;
"
echo

echo "4) N POR DIRECAO / MERCADO / REGIME (janela ${WINDOW_HOURS}h)"
run_psql "
WITH base AS (
  SELECT *
  FROM betslip_audit_results
  WHERE audit_version='${AUDIT_VERSION}'
    AND audited_at >= now() - interval '${WINDOW_HOURS} hours'
)
SELECT
  COALESCE(reversal_direction, 'na') AS direction,
  market_type,
  market_period,
  CASE WHEN is_live THEN 'IN_MATCH' ELSE 'PRE_MATCH' END AS regime,
  COUNT(*) AS n
FROM base
GROUP BY 1,2,3,4
ORDER BY n DESC, 1,2,3,4;
"
echo

echo "5) N POR BUCKET BS vs WS (janela ${WINDOW_HOURS}h)"
run_psql "
WITH base AS (
  SELECT *
  FROM betslip_audit_results
  WHERE audit_version='${AUDIT_VERSION}'
    AND audited_at >= now() - interval '${WINDOW_HOURS} hours'
    AND status='OK'
    AND difference_pct IS NOT NULL
)
SELECT
  CASE
    WHEN difference_pct <= -10 THEN 'A <= -10%'
    WHEN difference_pct <= -5  THEN 'B (-10,-5]'
    WHEN difference_pct <= -2  THEN 'C (-5,-2]'
    WHEN difference_pct < 2    THEN 'D (-2,+2)'
    ELSE 'E >= +2%'
  END AS bucket,
  COUNT(*) AS n_total,
  COUNT(*) FILTER (WHERE (hypothesis_details::jsonb -> 'lay') IS NOT NULL) AS n_lay_t0,
  COUNT(*) FILTER (WHERE (hypothesis_details::jsonb -> 'lay_temporal') IS NOT NULL) AS n_lay_temporal,
  COUNT(*) FILTER (WHERE (hypothesis_details::jsonb -> 'temporal') IS NOT NULL) AS n_back_temporal
FROM base
GROUP BY 1
ORDER BY 1;
"
echo

echo "6) N/DIA E PROJECAO MENSAL (lookback ${LOOKBACK_DAYS}d)"
run_psql "
WITH daily AS (
  SELECT
    date(audited_at) AS d,
    COUNT(*) AS n_total,
    COUNT(*) FILTER (WHERE status='OK') AS n_ok,
    COUNT(*) FILTER (WHERE (hypothesis_details::jsonb -> 'lay_temporal') IS NOT NULL) AS n_lay_temporal,
    COUNT(*) FILTER (WHERE (hypothesis_details::jsonb -> 'telemetry') IS NOT NULL) AS n_telemetry
  FROM betslip_audit_results
  WHERE audit_version='${AUDIT_VERSION}'
    AND audited_at >= now() - interval '${LOOKBACK_DAYS} days'
  GROUP BY 1
)
SELECT * FROM daily ORDER BY d DESC;
"

run_psql "
WITH daily AS (
  SELECT
    date(audited_at) AS d,
    COUNT(*) AS n_total,
    COUNT(*) FILTER (WHERE status='OK') AS n_ok,
    COUNT(*) FILTER (WHERE (hypothesis_details::jsonb -> 'lay_temporal') IS NOT NULL) AS n_lay_temporal,
    COUNT(*) FILTER (WHERE (hypothesis_details::jsonb -> 'telemetry') IS NOT NULL) AS n_telemetry
  FROM betslip_audit_results
  WHERE audit_version='${AUDIT_VERSION}'
    AND audited_at >= now() - interval '${LOOKBACK_DAYS} days'
  GROUP BY 1
)
SELECT
  COUNT(*) AS dias_com_dado,
  COALESCE(SUM(n_total),0) AS n_total_periodo,
  ROUND(COALESCE(AVG(n_total),0),2) AS n_total_dia,
  ROUND(COALESCE(AVG(n_total),0)*30,0) AS n_total_mes_proj,
  ROUND(COALESCE(AVG(n_lay_temporal),0),2) AS n_lay_temporal_dia,
  ROUND(COALESCE(AVG(n_lay_temporal),0)*30,0) AS n_lay_temporal_mes_proj,
  ROUND(COALESCE(AVG(n_telemetry),0),2) AS n_telemetry_dia
FROM daily;
"
echo

echo "7) TEMPOS MEDIOS POR ETAPA (24h, onde houver telemetry)"
run_psql "
WITH t AS (
  SELECT
    (hypothesis_details::jsonb -> 'telemetry' ->> 'queue_wait_ms')::numeric AS queue_wait_ms,
    NULLIF((hypothesis_details::jsonb -> 'telemetry' ->> 'queue_depth_at_enqueue'),'')::numeric AS queue_depth_at_enqueue,
    NULLIF((hypothesis_details::jsonb -> 'telemetry' ->> 'queue_depth_after_dequeue'),'')::numeric AS queue_depth_after_dequeue,
    (hypothesis_details::jsonb -> 'telemetry' ->> 'back_post_ms')::numeric AS back_post_ms,
    (hypothesis_details::jsonb -> 'telemetry' ->> 'back_pmm_ms')::numeric AS back_pmm_ms,
    (hypothesis_details::jsonb -> 'telemetry' ->> 'lay_post_ms')::numeric AS lay_post_ms,
    (hypothesis_details::jsonb -> 'telemetry' ->> 'lay_pmm_ms')::numeric AS lay_pmm_ms,
    (hypothesis_details::jsonb -> 'telemetry' ->> 'temporal_total_ms')::numeric AS temporal_total_ms,
    NULLIF((hypothesis_details::jsonb -> 'telemetry' ->> 'db_save_ms'),'')::numeric AS db_save_ms,
    NULLIF((hypothesis_details::jsonb -> 'telemetry' ->> 'pipeline_total_ms'),'')::numeric AS pipeline_total_ms
  FROM betslip_audit_results
  WHERE audit_version='${AUDIT_VERSION}'
    AND audited_at >= now() - interval '24 hours'
    AND (hypothesis_details::jsonb -> 'telemetry') IS NOT NULL
),
calc AS (
  SELECT
    *,
    (GREATEST(COALESCE(back_post_ms,0)+COALESCE(back_pmm_ms,0), COALESCE(lay_post_ms,0)+COALESCE(lay_pmm_ms,0)) + COALESCE(temporal_total_ms,0)) AS service_ms_est,
    CASE
      WHEN (GREATEST(COALESCE(back_post_ms,0)+COALESCE(back_pmm_ms,0), COALESCE(lay_post_ms,0)+COALESCE(lay_pmm_ms,0)) + COALESCE(temporal_total_ms,0)) > 0
      THEN queue_wait_ms / (GREATEST(COALESCE(back_post_ms,0)+COALESCE(back_pmm_ms,0), COALESCE(lay_post_ms,0)+COALESCE(lay_pmm_ms,0)) + COALESCE(temporal_total_ms,0))
      ELSE NULL
    END AS queue_jobs_ahead_est
  FROM t
)
SELECT
  COUNT(*) AS n_com_telemetry,
  ROUND(AVG(queue_wait_ms),1) AS queue_wait_ms,
  ROUND(AVG(queue_depth_at_enqueue),2) AS queue_depth_enq_avg,
  ROUND((PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY queue_depth_at_enqueue))::numeric,2) AS queue_depth_enq_p90,
  ROUND(AVG(queue_depth_after_dequeue),2) AS queue_depth_deq_avg,
  ROUND(AVG(back_post_ms),1) AS back_post_ms,
  ROUND(AVG(back_pmm_ms),1) AS back_pmm_ms,
  ROUND(AVG(lay_post_ms),1) AS lay_post_ms,
  ROUND(AVG(lay_pmm_ms),1) AS lay_pmm_ms,
  ROUND(AVG(temporal_total_ms),1) AS temporal_total_ms,
  ROUND(AVG(db_save_ms),1) AS db_save_ms,
  ROUND(AVG(pipeline_total_ms),1) AS pipeline_total_ms,
  ROUND(AVG(queue_jobs_ahead_est),2) AS queue_jobs_ahead_est_avg,
  ROUND((PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY queue_jobs_ahead_est))::numeric,2) AS queue_jobs_ahead_est_p90
FROM calc;
"
echo

echo "8) N POR TABELA DE HIPOTESE (se existir)"
for t in h1_pricing_events h3_line_monotonicity_events h3b_temporal_reversal_events h6_correlation_lag_events; do
  if table_exists "$t"; then
    run_psql "
    SELECT
      '${t}' AS tabela,
      COUNT(*) AS n_total,
      COUNT(*) FILTER (WHERE detected_at >= now() - interval '${WINDOW_HOURS} hours') AS n_window,
      COUNT(*) FILTER (WHERE is_live IS TRUE AND detected_at >= now() - interval '${WINDOW_HOURS} hours') AS n_live_window,
      COUNT(*) FILTER (WHERE (is_live IS FALSE OR is_live IS NULL) AND detected_at >= now() - interval '${WINDOW_HOURS} hours') AS n_pre_window
    FROM ${t};
    "
  else
    echo "Tabela ausente: ${t}"
  fi
done

echo
echo "====================================================================="
echo "Fim do snapshot. Use com janela maior para capacidade: "
echo "  bash capacity_snapshot_v4.sh --window-hours 72 --lookback-days 14"
echo "====================================================================="
