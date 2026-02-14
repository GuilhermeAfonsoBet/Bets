#!/usr/bin/env bash
set -u
set -o pipefail

# Análise robusta de performance por versão do robô
# - Operacional por versão (OK/fail/latência/cobertura)
# - Coorte Back (BS >> WS) e Lay (BS << WS)
# - Risco de cauda para Lay (liability p95/p99 + ES95)
# - Separação por audit_version
#
# Obs:
# - Não altera execução do robô; é apenas análise offline via SQL.
# - Para métricas financeiras, usa blocos hypothesis_details.finance quando existir.
#   Se ausente, usa fallback baseado em % do limite disponível.

LOOKBACK_DAYS=14
AUDIT_VERSION_FILTER=""
BACK_DIFF_MIN=2.0
LAY_DIFF_MAX=-2.0
FALLBACK_STAKE_PCT="${FALLBACK_STAKE_PCT:-0.25}"
LIABILITY_BUCKET_MIN=5
MAX_BUCKETS=20

DB_NAME="${DB_NAME:-betinasia_bot}"
DB_USER="${DB_USER:-betbot}"
USE_SUDO_PSQL=1

usage() {
  cat <<'EOF'
Uso: bash hypothesis_performance_robust.sh [opcoes]

Opcoes:
  --lookback-days N       Janela de análise (default: 14)
  --audit-version V       Filtra uma versão específica (default: todas)
  --back-diff-min X       Corte Back BS>>WS (default: 2.0)
  --lay-diff-max X        Corte Lay BS<<WS (default: -2.0)
  --fallback-stake-pct X  Stake fallback (% do limite), default: 0.25
  --liability-bucket-min N Bucket de agregação de exposição lay (default: 5)
  --max-buckets N         Top buckets mais expostos para imprimir (default: 20)
  --db-name N             Banco (default: betinasia_bot)
  --db-user U             Usuário psql (default: betbot)
  --no-sudo-psql          Não usar sudo -u no psql
  --help                  Ajuda
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lookback-days) LOOKBACK_DAYS="${2:-}"; shift 2 ;;
    --audit-version) AUDIT_VERSION_FILTER="${2:-}"; shift 2 ;;
    --back-diff-min) BACK_DIFF_MIN="${2:-}"; shift 2 ;;
    --lay-diff-max) LAY_DIFF_MAX="${2:-}"; shift 2 ;;
    --fallback-stake-pct) FALLBACK_STAKE_PCT="${2:-}"; shift 2 ;;
    --liability-bucket-min) LIABILITY_BUCKET_MIN="${2:-}"; shift 2 ;;
    --max-buckets) MAX_BUCKETS="${2:-}"; shift 2 ;;
    --db-name) DB_NAME="${2:-}"; shift 2 ;;
    --db-user) DB_USER="${2:-}"; shift 2 ;;
    --no-sudo-psql) USE_SUDO_PSQL=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Opcao invalida: $1" >&2; usage; exit 2 ;;
  esac
done

run_psql() {
  local sql="$1"
  local out rc
  if [[ "$USE_SUDO_PSQL" -eq 1 ]] && [[ "$(id -un)" != "$DB_USER" ]] && command -v sudo >/dev/null 2>&1; then
    out="$(sudo -u "$DB_USER" psql -v ON_ERROR_STOP=1 "$DB_NAME" -c "$sql" 2>&1)"
    rc=$?
  else
    out="$(psql -v ON_ERROR_STOP=1 "$DB_NAME" -c "$sql" 2>&1)"
    rc=$?
  fi
  if [[ "$rc" -ne 0 ]]; then
    echo "[ERRO SQL] consulta falhou"
    echo "$out"
    return "$rc"
  fi
  printf "%s\n" "$out"
}

AUDIT_FILTER_SQL="TRUE"
if [[ -n "$AUDIT_VERSION_FILTER" ]]; then
  AUDIT_FILTER_SQL="audit_version='${AUDIT_VERSION_FILTER}'"
fi

BUCKET_SEC=$(( LIABILITY_BUCKET_MIN * 60 ))

echo "====================================================================="
echo "ANALISE ROBUSTA DE PERFORMANCE / HIPOTESES | $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
echo "lookback=${LOOKBACK_DAYS}d | back_cut>=${BACK_DIFF_MIN}% | lay_cut<=${LAY_DIFF_MAX}%"
if [[ -n "$AUDIT_VERSION_FILTER" ]]; then
  echo "audit_version=${AUDIT_VERSION_FILTER}"
else
  echo "audit_version=TODAS"
fi
echo "fallback_stake_pct=${FALLBACK_STAKE_PCT} | liability_bucket=${LIABILITY_BUCKET_MIN}m"
echo "====================================================================="
echo

echo "1) PERFIL OPERACIONAL POR VERSAO"
run_psql "
WITH base AS (
  SELECT
    audit_version,
    hypothesis_type,
    status,
    difference_pct,
    hypothesis_details::jsonb AS h,
    CASE
      WHEN COALESCE(hypothesis_details::jsonb -> 'telemetry' ->> 'queue_wait_ms','') ~ '^[+-]?[0-9]+([.][0-9]+)?$'
      THEN (hypothesis_details::jsonb -> 'telemetry' ->> 'queue_wait_ms')::numeric
      ELSE NULL
    END AS q_ms,
    CASE
      WHEN COALESCE(hypothesis_details::jsonb -> 'telemetry' ->> 'pipeline_total_ms','') ~ '^[+-]?[0-9]+([.][0-9]+)?$'
      THEN (hypothesis_details::jsonb -> 'telemetry' ->> 'pipeline_total_ms')::numeric
      ELSE NULL
    END AS pipe_ms
  FROM betslip_audit_results
  WHERE audited_at >= now() - interval '${LOOKBACK_DAYS} days'
    AND ${AUDIT_FILTER_SQL}
)
SELECT
  audit_version,
  hypothesis_type,
  COUNT(*) AS n_total,
  COUNT(*) FILTER (WHERE status='OK') AS n_ok,
  ROUND(100.0 * COUNT(*) FILTER (WHERE status='OK') / NULLIF(COUNT(*),0), 1) AS ok_pct,
  COUNT(*) FILTER (WHERE status<>'OK') AS n_fail,
  COUNT(*) FILTER (WHERE status='OK' AND difference_pct >= ${BACK_DIFF_MIN}) AS n_back_edge,
  COUNT(*) FILTER (WHERE status='OK' AND difference_pct <= ${LAY_DIFF_MAX}) AS n_lay_edge,
  ROUND(100.0 * COUNT(*) FILTER (WHERE (h -> 'lay') IS NOT NULL) / NULLIF(COUNT(*),0), 1) AS lay_t0_cov_pct,
  ROUND(100.0 * COUNT(*) FILTER (WHERE (h -> 'lay_temporal') IS NOT NULL) / NULLIF(COUNT(*),0), 1) AS lay_temporal_cov_pct,
  ROUND(100.0 * COUNT(*) FILTER (WHERE (h -> 'finance') IS NOT NULL) / NULLIF(COUNT(*),0), 1) AS finance_cov_pct,
  ROUND(AVG(COALESCE(q_ms,0)),1) AS q_avg_ms,
  ROUND((PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY COALESCE(q_ms,0)))::numeric,1) AS q_p95_ms,
  ROUND(AVG(COALESCE(pipe_ms,0)),1) AS pipe_avg_ms,
  ROUND((PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY COALESCE(pipe_ms,0)))::numeric,1) AS pipe_p95_ms
FROM base
GROUP BY 1,2
ORDER BY 1 DESC, 2;
"
echo

echo "2) COORTE BACK (BS >> WS) POR VERSAO"
run_psql "
WITH b AS (
  SELECT
    audit_version,
    difference_pct,
    betslip_odd,
    betslip_limit,
    hypothesis_details::jsonb AS h,
    CASE
      WHEN COALESCE(hypothesis_details::jsonb -> 'finance' -> 'back' ->> 'suggested_stake','') ~ '^[+-]?[0-9]+([.][0-9]+)?$'
      THEN (hypothesis_details::jsonb -> 'finance' -> 'back' ->> 'suggested_stake')::numeric
      ELSE NULL
    END AS back_stake_fin,
    CASE
      WHEN COALESCE(hypothesis_details::jsonb -> 'finance' -> 'back' ->> 'profit_if_win','') ~ '^[+-]?[0-9]+([.][0-9]+)?$'
      THEN (hypothesis_details::jsonb -> 'finance' -> 'back' ->> 'profit_if_win')::numeric
      ELSE NULL
    END AS back_win_fin
  FROM betslip_audit_results
  WHERE audited_at >= now() - interval '${LOOKBACK_DAYS} days'
    AND ${AUDIT_FILTER_SQL}
    AND status='OK'
    AND difference_pct >= ${BACK_DIFF_MIN}
    AND betslip_odd IS NOT NULL
),
x AS (
  SELECT
    audit_version,
    difference_pct,
    betslip_odd,
    betslip_limit,
    COALESCE(back_stake_fin, GREATEST(COALESCE(betslip_limit,0),0) * ${FALLBACK_STAKE_PCT}) AS back_stake_est,
    COALESCE(back_win_fin, COALESCE(back_stake_fin, GREATEST(COALESCE(betslip_limit,0),0) * ${FALLBACK_STAKE_PCT}) * GREATEST(COALESCE(betslip_odd,0)-1,0)) AS back_profit_if_win_est
  FROM b
)
SELECT
  audit_version,
  COUNT(*) AS n,
  ROUND(AVG(difference_pct)::numeric,2) AS diff_avg_pct,
  ROUND((PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY difference_pct))::numeric,2) AS diff_p50_pct,
  ROUND((PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY difference_pct))::numeric,2) AS diff_p90_pct,
  ROUND(AVG(betslip_odd)::numeric,3) AS odd_avg,
  ROUND(AVG(betslip_limit)::numeric,2) AS limit_avg,
  ROUND((PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY betslip_limit))::numeric,2) AS limit_p90,
  ROUND(AVG(back_stake_est)::numeric,2) AS stake_est_avg,
  ROUND((PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY back_stake_est))::numeric,2) AS stake_est_p90,
  ROUND(AVG(back_profit_if_win_est)::numeric,2) AS profit_if_win_avg
FROM x
GROUP BY 1
ORDER BY 1 DESC;
"
echo

echo "3) COORTE LAY (BS << WS) POR VERSAO + RISCO DE CAUDA"
run_psql "
WITH b AS (
  SELECT
    audit_version,
    difference_pct,
    hypothesis_details::jsonb AS h,
    CASE
      WHEN COALESCE(hypothesis_details::jsonb -> 'lay' ->> 'odd','') ~ '^[+-]?[0-9]+([.][0-9]+)?$'
      THEN (hypothesis_details::jsonb -> 'lay' ->> 'odd')::numeric
      ELSE NULL
    END AS lay_odd,
    CASE
      WHEN COALESCE(hypothesis_details::jsonb -> 'lay' ->> 'limit','') ~ '^[+-]?[0-9]+([.][0-9]+)?$'
      THEN (hypothesis_details::jsonb -> 'lay' ->> 'limit')::numeric
      ELSE NULL
    END AS lay_limit,
    CASE
      WHEN COALESCE(hypothesis_details::jsonb -> 'finance' -> 'lay' ->> 'suggested_stake','') ~ '^[+-]?[0-9]+([.][0-9]+)?$'
      THEN (hypothesis_details::jsonb -> 'finance' -> 'lay' ->> 'suggested_stake')::numeric
      ELSE NULL
    END AS lay_stake_fin,
    CASE
      WHEN COALESCE(hypothesis_details::jsonb -> 'finance' -> 'lay' ->> 'liability_if_lose','') ~ '^[+-]?[0-9]+([.][0-9]+)?$'
      THEN (hypothesis_details::jsonb -> 'finance' -> 'lay' ->> 'liability_if_lose')::numeric
      ELSE NULL
    END AS lay_liability_fin
  FROM betslip_audit_results
  WHERE audited_at >= now() - interval '${LOOKBACK_DAYS} days'
    AND ${AUDIT_FILTER_SQL}
    AND status='OK'
    AND difference_pct <= ${LAY_DIFF_MAX}
),
x AS (
  SELECT
    audit_version,
    difference_pct,
    lay_odd,
    lay_limit,
    COALESCE(lay_stake_fin, GREATEST(COALESCE(lay_limit,0),0) * ${FALLBACK_STAKE_PCT}) AS lay_stake_est,
    COALESCE(
      lay_liability_fin,
      COALESCE(lay_stake_fin, GREATEST(COALESCE(lay_limit,0),0) * ${FALLBACK_STAKE_PCT}) * GREATEST(COALESCE(lay_odd,0)-1,0)
    ) AS lay_liability_est
  FROM b
  WHERE lay_odd IS NOT NULL
)
SELECT
  audit_version,
  COUNT(*) AS n,
  ROUND(AVG(difference_pct)::numeric,2) AS diff_avg_pct,
  ROUND((PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY difference_pct))::numeric,2) AS diff_p50_pct,
  ROUND((PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY difference_pct))::numeric,2) AS diff_p10_pct,
  ROUND(AVG(lay_odd)::numeric,3) AS lay_odd_avg,
  ROUND(AVG(lay_limit)::numeric,2) AS lay_limit_avg,
  ROUND(AVG(lay_stake_est)::numeric,2) AS lay_stake_avg,
  ROUND(AVG(lay_liability_est)::numeric,2) AS liability_avg,
  ROUND((PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY lay_liability_est))::numeric,2) AS liability_p90,
  ROUND((PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY lay_liability_est))::numeric,2) AS liability_p95,
  ROUND((PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY lay_liability_est))::numeric,2) AS liability_p99,
  ROUND(MAX(lay_liability_est)::numeric,2) AS liability_max,
  ROUND(AVG(CASE WHEN lay_stake_est > 0 THEN lay_liability_est / lay_stake_est END)::numeric,2) AS liab_to_stake_avg
FROM x
GROUP BY 1
ORDER BY 1 DESC;
"
echo

echo "4) LAY TAIL RISK - ES95 E EXPOSICAO AGREGADA POR BUCKET"
run_psql "
WITH lay AS (
  SELECT
    audit_version,
    audited_at,
    COALESCE(
      CASE
        WHEN COALESCE(hypothesis_details::jsonb -> 'finance' -> 'lay' ->> 'liability_if_lose','') ~ '^[+-]?[0-9]+([.][0-9]+)?$'
        THEN (hypothesis_details::jsonb -> 'finance' -> 'lay' ->> 'liability_if_lose')::numeric
        ELSE NULL
      END,
      (
        CASE
          WHEN COALESCE(hypothesis_details::jsonb -> 'finance' -> 'lay' ->> 'suggested_stake','') ~ '^[+-]?[0-9]+([.][0-9]+)?$'
          THEN (hypothesis_details::jsonb -> 'finance' -> 'lay' ->> 'suggested_stake')::numeric
          WHEN COALESCE(hypothesis_details::jsonb -> 'lay' ->> 'limit','') ~ '^[+-]?[0-9]+([.][0-9]+)?$'
          THEN (hypothesis_details::jsonb -> 'lay' ->> 'limit')::numeric * ${FALLBACK_STAKE_PCT}
          ELSE NULL
        END
      ) * GREATEST(
        COALESCE(
          CASE
            WHEN COALESCE(hypothesis_details::jsonb -> 'lay' ->> 'odd','') ~ '^[+-]?[0-9]+([.][0-9]+)?$'
            THEN (hypothesis_details::jsonb -> 'lay' ->> 'odd')::numeric
            ELSE 0
          END, 0
        ) - 1, 0
      )
    ) AS liability
  FROM betslip_audit_results
  WHERE audited_at >= now() - interval '${LOOKBACK_DAYS} days'
    AND ${AUDIT_FILTER_SQL}
    AND status='OK'
    AND difference_pct <= ${LAY_DIFF_MAX}
),
lay_clean AS (
  SELECT *
  FROM lay
  WHERE liability IS NOT NULL AND liability > 0
),
p AS (
  SELECT
    audit_version,
    (PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY liability))::numeric AS p95_single,
    (PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY liability))::numeric AS p99_single
  FROM lay_clean
  GROUP BY 1
),
es AS (
  SELECT
    l.audit_version,
    AVG(l.liability) AS es95_single
  FROM lay_clean l
  JOIN p ON p.audit_version = l.audit_version
  WHERE l.liability >= p.p95_single
  GROUP BY 1
),
buckets AS (
  SELECT
    audit_version,
    to_timestamp(floor(extract(epoch from audited_at) / ${BUCKET_SEC}) * ${BUCKET_SEC}) AS bucket_utc,
    COUNT(*) AS n_lay,
    SUM(liability) AS liability_bucket
  FROM lay_clean
  GROUP BY 1,2
),
bucket_stats AS (
  SELECT
    audit_version,
    ROUND(AVG(liability_bucket)::numeric,2) AS bucket_liability_avg,
    ROUND((PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY liability_bucket))::numeric,2) AS bucket_liability_p95,
    ROUND((PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY liability_bucket))::numeric,2) AS bucket_liability_p99,
    ROUND(MAX(liability_bucket)::numeric,2) AS bucket_liability_max
  FROM buckets
  GROUP BY 1
)
SELECT
  p.audit_version,
  ROUND(p.p95_single,2) AS single_liability_p95,
  ROUND(p.p99_single,2) AS single_liability_p99,
  ROUND(es.es95_single,2) AS single_liability_es95,
  bs.bucket_liability_avg,
  bs.bucket_liability_p95,
  bs.bucket_liability_p99,
  bs.bucket_liability_max
FROM p
LEFT JOIN es ON es.audit_version = p.audit_version
LEFT JOIN bucket_stats bs ON bs.audit_version = p.audit_version
ORDER BY 1 DESC;
"
echo

echo "5) TOP BUCKETS DE EXPOSICAO LAY (maior soma de liability)"
run_psql "
WITH lay AS (
  SELECT
    audit_version,
    audited_at,
    COALESCE(
      CASE
        WHEN COALESCE(hypothesis_details::jsonb -> 'finance' -> 'lay' ->> 'liability_if_lose','') ~ '^[+-]?[0-9]+([.][0-9]+)?$'
        THEN (hypothesis_details::jsonb -> 'finance' -> 'lay' ->> 'liability_if_lose')::numeric
        ELSE NULL
      END,
      (
        CASE
          WHEN COALESCE(hypothesis_details::jsonb -> 'finance' -> 'lay' ->> 'suggested_stake','') ~ '^[+-]?[0-9]+([.][0-9]+)?$'
          THEN (hypothesis_details::jsonb -> 'finance' -> 'lay' ->> 'suggested_stake')::numeric
          WHEN COALESCE(hypothesis_details::jsonb -> 'lay' ->> 'limit','') ~ '^[+-]?[0-9]+([.][0-9]+)?$'
          THEN (hypothesis_details::jsonb -> 'lay' ->> 'limit')::numeric * ${FALLBACK_STAKE_PCT}
          ELSE NULL
        END
      ) * GREATEST(
        COALESCE(
          CASE
            WHEN COALESCE(hypothesis_details::jsonb -> 'lay' ->> 'odd','') ~ '^[+-]?[0-9]+([.][0-9]+)?$'
            THEN (hypothesis_details::jsonb -> 'lay' ->> 'odd')::numeric
            ELSE 0
          END, 0
        ) - 1, 0
      )
    ) AS liability
  FROM betslip_audit_results
  WHERE audited_at >= now() - interval '${LOOKBACK_DAYS} days'
    AND ${AUDIT_FILTER_SQL}
    AND status='OK'
    AND difference_pct <= ${LAY_DIFF_MAX}
),
b AS (
  SELECT
    audit_version,
    to_timestamp(floor(extract(epoch from audited_at) / ${BUCKET_SEC}) * ${BUCKET_SEC}) AS bucket_utc,
    COUNT(*) AS n_lay,
    ROUND(SUM(liability)::numeric,2) AS liability_bucket
  FROM lay
  WHERE liability IS NOT NULL AND liability > 0
  GROUP BY 1,2
)
SELECT *
FROM b
ORDER BY liability_bucket DESC
LIMIT ${MAX_BUCKETS};
"
echo

echo "6) COBERTURA DE RESULTADO REALIZADO (CLV/P&L) NAS TABELAS DE HIPOTESE"
run_psql "
SELECT 'h1_pricing_events' AS tabela,
       COUNT(*) FILTER (WHERE detected_at >= now() - interval '${LOOKBACK_DAYS} days') AS n_total,
       COUNT(*) FILTER (WHERE detected_at >= now() - interval '${LOOKBACK_DAYS} days' AND clv_pct IS NOT NULL) AS n_clv,
       COUNT(*) FILTER (WHERE detected_at >= now() - interval '${LOOKBACK_DAYS} days' AND profit_loss IS NOT NULL) AS n_pl
FROM h1_pricing_events
UNION ALL
SELECT 'h3_line_monotonicity_events',
       COUNT(*) FILTER (WHERE detected_at >= now() - interval '${LOOKBACK_DAYS} days'),
       COUNT(*) FILTER (WHERE detected_at >= now() - interval '${LOOKBACK_DAYS} days' AND clv_pct IS NOT NULL),
       COUNT(*) FILTER (WHERE detected_at >= now() - interval '${LOOKBACK_DAYS} days' AND profit_loss IS NOT NULL)
FROM h3_line_monotonicity_events
UNION ALL
SELECT 'h3b_temporal_reversal_events',
       COUNT(*) FILTER (WHERE detected_at >= now() - interval '${LOOKBACK_DAYS} days'),
       COUNT(*) FILTER (WHERE detected_at >= now() - interval '${LOOKBACK_DAYS} days' AND clv_pct IS NOT NULL),
       COUNT(*) FILTER (WHERE detected_at >= now() - interval '${LOOKBACK_DAYS} days' AND profit_loss IS NOT NULL)
FROM h3b_temporal_reversal_events
UNION ALL
SELECT 'h6_correlation_lag_events',
       COUNT(*) FILTER (WHERE detected_at >= now() - interval '${LOOKBACK_DAYS} days'),
       COUNT(*) FILTER (WHERE detected_at >= now() - interval '${LOOKBACK_DAYS} days' AND clv_pct IS NOT NULL),
       COUNT(*) FILTER (WHERE detected_at >= now() - interval '${LOOKBACK_DAYS} days' AND profit_loss IS NOT NULL)
FROM h6_correlation_lag_events;
"
echo

echo "7) INFERENCIA ESTATISTICA (audit diff_pct por coorte)"
run_psql "
WITH cohorts AS (
  SELECT
    audit_version,
    CASE
      WHEN status='OK' AND difference_pct >= ${BACK_DIFF_MIN} THEN 'BACK_EDGE'
      WHEN status='OK' AND difference_pct <= ${LAY_DIFF_MAX} THEN 'LAY_EDGE'
      ELSE NULL
    END AS cohort,
    difference_pct::numeric AS metric
  FROM betslip_audit_results
  WHERE audited_at >= now() - interval '${LOOKBACK_DAYS} days'
    AND ${AUDIT_FILTER_SQL}
    AND status='OK'
    AND difference_pct IS NOT NULL
),
stats AS (
  SELECT
    audit_version,
    cohort,
    COUNT(*) AS n,
    AVG(metric) AS mean_metric,
    STDDEV_SAMP(metric) AS sd_metric
  FROM cohorts
  WHERE cohort IS NOT NULL
  GROUP BY 1,2
),
calc AS (
  SELECT
    *,
    CASE WHEN n >= 2 AND sd_metric IS NOT NULL THEN sd_metric / SQRT(n::numeric) END AS se_metric
  FROM stats
)
SELECT
  audit_version,
  cohort,
  n,
  ROUND(mean_metric::numeric, 3) AS mean_diff_pct,
  ROUND(sd_metric::numeric, 3) AS sd_diff_pct,
  ROUND(se_metric::numeric, 3) AS se_diff_pct,
  ROUND((mean_metric - 1.645 * se_metric)::numeric, 3) AS ci90_low,
  ROUND((mean_metric + 1.645 * se_metric)::numeric, 3) AS ci90_high,
  ROUND((mean_metric - 1.960 * se_metric)::numeric, 3) AS ci95_low,
  ROUND((mean_metric + 1.960 * se_metric)::numeric, 3) AS ci95_high,
  ROUND((mean_metric / NULLIF(se_metric,0))::numeric, 2) AS t_stat_vs_0,
  CASE
    WHEN se_metric IS NULL THEN 'NA'
    WHEN (mean_metric - 1.960 * se_metric) > 0 OR (mean_metric + 1.960 * se_metric) < 0 THEN 'YES'
    ELSE 'NO'
  END AS sig_95
FROM calc
ORDER BY audit_version DESC, cohort;
"
echo

echo "8) INFERENCIA DE RESULTADO REALIZADO (profit_loss e CLV)"
run_psql "
WITH data AS (
  SELECT 'h1_pricing_events' AS tabela, profit_loss::numeric AS profit_loss, clv_pct::numeric AS clv_pct
  FROM h1_pricing_events
  WHERE detected_at >= now() - interval '${LOOKBACK_DAYS} days'
  UNION ALL
  SELECT 'h3_line_monotonicity_events', profit_loss::numeric, clv_pct::numeric
  FROM h3_line_monotonicity_events
  WHERE detected_at >= now() - interval '${LOOKBACK_DAYS} days'
  UNION ALL
  SELECT 'h3b_temporal_reversal_events', profit_loss::numeric, clv_pct::numeric
  FROM h3b_temporal_reversal_events
  WHERE detected_at >= now() - interval '${LOOKBACK_DAYS} days'
  UNION ALL
  SELECT 'h6_correlation_lag_events', profit_loss::numeric, clv_pct::numeric
  FROM h6_correlation_lag_events
  WHERE detected_at >= now() - interval '${LOOKBACK_DAYS} days'
),
agg AS (
  SELECT
    tabela,
    COUNT(*) AS n_total,
    COUNT(profit_loss) AS n_pl,
    ROUND(100.0 * COUNT(profit_loss) / NULLIF(COUNT(*),0), 1) AS pl_cov_pct,
    AVG(profit_loss) AS pl_mean,
    STDDEV_SAMP(profit_loss) AS pl_sd,
    COUNT(clv_pct) AS n_clv,
    ROUND(100.0 * COUNT(clv_pct) / NULLIF(COUNT(*),0), 1) AS clv_cov_pct,
    AVG(clv_pct) AS clv_mean,
    STDDEV_SAMP(clv_pct) AS clv_sd,
    AVG(
      CASE
        WHEN profit_loss IS NULL THEN NULL
        WHEN profit_loss > 0 THEN 1.0
        ELSE 0.0
      END
    ) AS win_rate
  FROM data
  GROUP BY 1
),
calc AS (
  SELECT
    *,
    CASE WHEN n_pl >= 2 AND pl_sd IS NOT NULL THEN pl_sd / SQRT(n_pl::numeric) END AS pl_se,
    CASE WHEN n_clv >= 2 AND clv_sd IS NOT NULL THEN clv_sd / SQRT(n_clv::numeric) END AS clv_se,
    CASE
      WHEN n_pl >= 2 AND win_rate IS NOT NULL
      THEN SQRT(win_rate * (1 - win_rate) / n_pl::numeric)
      ELSE NULL
    END AS wr_se
  FROM agg
)
SELECT
  tabela,
  n_total,
  n_pl,
  pl_cov_pct,
  ROUND(pl_mean::numeric,4) AS pl_mean_u,
  ROUND((pl_mean - 1.645 * pl_se)::numeric,4) AS pl_ci90_low,
  ROUND((pl_mean + 1.645 * pl_se)::numeric,4) AS pl_ci90_high,
  ROUND((pl_mean - 1.960 * pl_se)::numeric,4) AS pl_ci95_low,
  ROUND((pl_mean + 1.960 * pl_se)::numeric,4) AS pl_ci95_high,
  n_clv,
  clv_cov_pct,
  ROUND(clv_mean::numeric,4) AS clv_mean_pct,
  ROUND((clv_mean - 1.960 * clv_se)::numeric,4) AS clv_ci95_low,
  ROUND((clv_mean + 1.960 * clv_se)::numeric,4) AS clv_ci95_high,
  ROUND((win_rate * 100)::numeric,2) AS win_rate_pct,
  ROUND(((win_rate - 1.960 * wr_se) * 100)::numeric,2) AS win_rate_ci95_low,
  ROUND(((win_rate + 1.960 * wr_se) * 100)::numeric,2) AS win_rate_ci95_high
FROM calc
ORDER BY tabela;
"
echo

echo "9) NOTAS DE LEITURA"
echo "- Back e Lay devem ser analisados separados."
echo "- Lay tem cauda mais pesada: use p95/p99/ES95 de liability e bucket exposure."
echo "- A seção 7 testa diferença média (diff_pct) vs 0 com IC e t_stat."
echo "- A seção 8 usa apenas registros liquidados (profit_loss/clv não nulos)."
echo "- Se pl_cov_pct for baixo, IC de ROI/drawdown realizado fica frágil (amostra efetiva pequena)."
echo
echo "====================================================================="
echo "Fim da análise robusta."
echo "====================================================================="
