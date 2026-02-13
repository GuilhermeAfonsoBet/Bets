#!/usr/bin/env bash
set -u
set -o pipefail

# Snapshot de prontidão financeira do pipeline v4
# Objetivo:
#  - confirmar se os insumos econômicos estão sendo persistidos
#  - medir cobertura para análises futuras de turnover/lucro/ROI/drawdown

WINDOW_MINUTES=60
LOOKBACK_DAYS=14
AUDIT_VERSION="v4.0-api"
DB_NAME="${DB_NAME:-betinasia_bot}"
DB_USER="${DB_USER:-betbot}"
USE_SUDO_PSQL=1

usage() {
  cat <<'EOF'
Uso: bash finance_readiness_v4.sh [opcoes]

Opcoes:
  --window-minutes N     Janela recente de auditoria (default: 60)
  --lookback-days N      Janela histórica de liquidação (default: 14)
  --audit-version V      Versão audit (default: v4.0-api)
  --db-name N            Banco (default: betinasia_bot)
  --db-user U            Usuário psql (default: betbot)
  --no-sudo-psql         Não usar sudo -u no psql
  --help                 Ajuda
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --window-minutes) WINDOW_MINUTES="${2:-}"; shift 2 ;;
    --lookback-days) LOOKBACK_DAYS="${2:-}"; shift 2 ;;
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

echo "====================================================================="
echo "FINANCE READINESS V4 | $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
echo "audit_version=${AUDIT_VERSION} | window=${WINDOW_MINUTES}m | lookback=${LOOKBACK_DAYS}d"
echo "====================================================================="
echo

echo "1) COBERTURA DE INSUMOS FINANCEIROS (janela ${WINDOW_MINUTES}m)"
run_psql "
WITH b AS (
  SELECT *
  FROM betslip_audit_results
  WHERE audit_version='${AUDIT_VERSION}'
    AND audited_at >= now() - interval '${WINDOW_MINUTES} minutes'
)
SELECT
  COUNT(*) AS n_total,
  COUNT(*) FILTER (WHERE status='OK') AS n_ok,
  ROUND(100.0 * COUNT(*) FILTER (WHERE status='OK') / NULLIF(COUNT(*),0), 1) AS ok_pct,
  COUNT(*) FILTER (WHERE betslip_odd IS NOT NULL) AS n_back_odd,
  COUNT(*) FILTER (WHERE betslip_limit IS NOT NULL AND betslip_limit > 0) AS n_back_limit,
  COUNT(*) FILTER (WHERE (hypothesis_details::jsonb -> 'lay' ->> 'odd') IS NOT NULL) AS n_lay_odd,
  COUNT(*) FILTER (
    WHERE (
      CASE
        WHEN COALESCE(hypothesis_details::jsonb -> 'lay' ->> 'limit','') ~ '^-?[0-9]+([.][0-9]+)?$'
        THEN (hypothesis_details::jsonb -> 'lay' ->> 'limit')::numeric
        ELSE 0
      END
    ) > 0
  ) AS n_lay_limit,
  COUNT(*) FILTER (WHERE (hypothesis_details::jsonb -> 'finance') IS NOT NULL) AS n_finance_block,
  ROUND(100.0 * COUNT(*) FILTER (WHERE (hypothesis_details::jsonb -> 'finance') IS NOT NULL) / NULLIF(COUNT(*),0), 1) AS finance_cov_pct,
  COUNT(*) FILTER (WHERE (hypothesis_details::jsonb -> 'telemetry') IS NOT NULL) AS n_telemetry
FROM b;
"
echo

echo "2) QUALIDADE DOS CAMPOS ECONOMICOS (janela ${WINDOW_MINUTES}m)"
run_psql "
WITH b AS (
  SELECT hypothesis_details::jsonb AS h
  FROM betslip_audit_results
  WHERE audit_version='${AUDIT_VERSION}'
    AND audited_at >= now() - interval '${WINDOW_MINUTES} minutes'
    AND (hypothesis_details::jsonb -> 'finance') IS NOT NULL
)
SELECT
  COUNT(*) AS n_finance,
  ROUND(AVG(
    CASE WHEN COALESCE(h -> 'finance' -> 'back' ->> 'suggested_stake','') ~ '^-?[0-9]+([.][0-9]+)?$'
         THEN (h -> 'finance' -> 'back' ->> 'suggested_stake')::numeric END
  ),2) AS back_stake_avg,
  ROUND(AVG(
    CASE WHEN COALESCE(h -> 'finance' -> 'lay' ->> 'suggested_stake','') ~ '^-?[0-9]+([.][0-9]+)?$'
         THEN (h -> 'finance' -> 'lay' ->> 'suggested_stake')::numeric END
  ),2) AS lay_stake_avg,
  ROUND(AVG(
    CASE WHEN COALESCE(h -> 'finance' -> 'back' ->> 'profit_if_win','') ~ '^-?[0-9]+([.][0-9]+)?$'
         THEN (h -> 'finance' -> 'back' ->> 'profit_if_win')::numeric END
  ),2) AS back_profit_if_win_avg,
  ROUND(AVG(
    CASE WHEN COALESCE(h -> 'finance' -> 'lay' ->> 'liability_if_lose','') ~ '^-?[0-9]+([.][0-9]+)?$'
         THEN (h -> 'finance' -> 'lay' ->> 'liability_if_lose')::numeric END
  ),2) AS lay_liability_avg
FROM b;
"
echo

echo "3) COBERTURA DE CLV/P&L REALIZADO NAS TABELAS DE HIPOTESE (${LOOKBACK_DAYS}d)"
run_psql "
WITH h3b AS (
  SELECT
    COUNT(*) AS n_total,
    COUNT(*) FILTER (WHERE clv_pct IS NOT NULL) AS n_clv,
    COUNT(*) FILTER (WHERE profit_loss IS NOT NULL) AS n_pl
  FROM h3b_temporal_reversal_events
  WHERE detected_at >= now() - interval '${LOOKBACK_DAYS} days'
)
SELECT
  n_total,
  n_clv,
  ROUND(100.0 * n_clv / NULLIF(n_total,0), 1) AS clv_cov_pct,
  n_pl,
  ROUND(100.0 * n_pl / NULLIF(n_total,0), 1) AS pl_cov_pct
FROM h3b;
"
echo

echo "4) LEITURA OPERACIONAL"
echo "- Para lucro/ROI/drawdown robustos precisamos de 2 camadas:"
echo "  (a) insumo de stake/liability no audit (agora salvo em hypothesis_details.finance)"
echo "  (b) resultado liquidado (profit_loss/clv) nas tabelas de hipótese."
echo
echo "====================================================================="
echo "Fim do finance readiness."
echo "====================================================================="
