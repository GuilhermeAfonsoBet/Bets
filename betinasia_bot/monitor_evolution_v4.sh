#!/usr/bin/env bash
set -u
set -o pipefail

# Monitor de evolução operacional da versão v4.0-api
# - Mostra tendência desde um marco temporal (deploy/restart/since custom)
# - Foco em throughput, qualidade, fila T+0, pipeline, temporal e sinais de estratégia
# - Inclui evolução do collector via JSONL de telemetria

AUDIT_VERSION="v4.0-api"
DB_NAME="${DB_NAME:-betinasia_bot}"
DB_USER="${DB_USER:-betbot}"
USE_SUDO_PSQL=1

AUDIT_SERVICE="${AUDIT_SERVICE:-betinasia-audit-api}"
COLLECTOR_SERVICE="${COLLECTOR_SERVICE:-betinasia-collector}"
COLLECTOR_TELEMETRY_FILE="${COLLECTOR_TELEMETRY_FILE:-logs/collector_telemetry.jsonl}"

SINCE=""
SINCE_HOURS=24
USE_SERVICE_RESTART=1
BUCKET_MINUTES=10
MAX_BUCKETS=48
PENDING_THRESHOLD_MIN=3

usage() {
  cat <<'EOF'
Uso: bash monitor_evolution_v4.sh [opcoes]

Opcoes:
  --since "YYYY-MM-DD HH:MM:SS"   Marco inicial em UTC (prioridade mais alta)
  --since-hours N                  Lookback em horas se nao usar --since (default: 24)
  --since-service-restart          Usa inicio do servico audit como marco (default: ligado)
  --no-since-service-restart       Ignora inicio do servico e usa --since-hours
  --bucket-minutes N               Bucket para series temporais (default: 10)
  --max-buckets N                  Limita linhas na serie (default: 48)
  --pending-threshold-min N        SLA para pendencia temporal (default: 3)
  --audit-version V                Versao de auditoria (default: v4.0-api)
  --collector-telemetry-file P     JSONL de telemetria do collector
  --db-name N                      Banco (default: betinasia_bot)
  --db-user U                      Usuario psql (default: betbot)
  --no-sudo-psql                   Nao usar sudo -u no psql
  --help                           Ajuda
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --since) SINCE="${2:-}"; shift 2 ;;
    --since-hours) SINCE_HOURS="${2:-}"; shift 2 ;;
    --since-service-restart) USE_SERVICE_RESTART=1; shift ;;
    --no-since-service-restart) USE_SERVICE_RESTART=0; shift ;;
    --bucket-minutes) BUCKET_MINUTES="${2:-}"; shift 2 ;;
    --max-buckets) MAX_BUCKETS="${2:-}"; shift 2 ;;
    --pending-threshold-min) PENDING_THRESHOLD_MIN="${2:-}"; shift 2 ;;
    --audit-version) AUDIT_VERSION="${2:-}"; shift 2 ;;
    --collector-telemetry-file) COLLECTOR_TELEMETRY_FILE="${2:-}"; shift 2 ;;
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

resolve_since_utc() {
  if [[ -n "$SINCE" ]]; then
    date -u -d "$SINCE" +'%Y-%m-%d %H:%M:%S+00' 2>/dev/null || return 1
    return 0
  fi

  if [[ "$USE_SERVICE_RESTART" -eq 1 ]]; then
    local restart_raw
    restart_raw="$(systemctl show "$AUDIT_SERVICE" -p ActiveEnterTimestamp --value 2>/dev/null || true)"
    if [[ -n "$restart_raw" ]] && [[ "$restart_raw" != "n/a" ]]; then
      date -u -d "$restart_raw" +'%Y-%m-%d %H:%M:%S+00' 2>/dev/null && return 0
    fi
  fi

  date -u -d "-${SINCE_HOURS} hours" +'%Y-%m-%d %H:%M:%S+00'
  return 0
}

SINCE_UTC="$(resolve_since_utc)" || {
  echo "Falha ao resolver --since. Exemplo valido: --since \"2026-02-13 16:12:00\"" >&2
  exit 2
}

NOW_UTC="$(date -u +'%Y-%m-%d %H:%M:%S+00')"
BUCKET_SEC=$((BUCKET_MINUTES * 60))

echo "====================================================================="
echo "MONITOR DE EVOLUCAO V4 | $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
echo "audit_version=${AUDIT_VERSION}"
echo "since_utc=${SINCE_UTC} | now_utc=${NOW_UTC}"
echo "bucket=${BUCKET_MINUTES} min | max_buckets=${MAX_BUCKETS}"
echo "collector_telemetry=${COLLECTOR_TELEMETRY_FILE}"
echo "====================================================================="
echo

echo "1) ESTADO DOS SERVICOS"
systemctl show "$AUDIT_SERVICE" -p ActiveState -p SubState -p NRestarts --no-pager 2>/dev/null || true
systemctl show "$COLLECTOR_SERVICE" -p ActiveState -p SubState -p NRestarts --no-pager 2>/dev/null || true
echo

echo "2) RESUMO ACUMULADO DESDE O MARCO"
run_psql "
WITH base AS (
  SELECT *
  FROM betslip_audit_results
  WHERE audit_version='${AUDIT_VERSION}'
    AND audited_at >= timestamptz '${SINCE_UTC}'
),
s AS (
  SELECT
    *,
    NULLIF((hypothesis_details::jsonb -> 'telemetry' ->> 'queue_wait_ms'),'')::numeric AS q_ms,
    NULLIF((hypothesis_details::jsonb -> 'telemetry' ->> 'pipeline_total_ms'),'')::numeric AS pipe_ms,
    NULLIF((hypothesis_details::jsonb -> 'telemetry' ->> 'temporal_async_latency_ms'),'')::numeric AS t_async_ms,
    COALESCE((hypothesis_details::jsonb -> 'telemetry' ->> 'temporal_deferred'),'false')::boolean AS temporal_deferred
  FROM base
)
SELECT
  COUNT(*) AS n_total,
  COUNT(*) FILTER (WHERE status='OK') AS n_ok,
  ROUND(100.0 * COUNT(*) FILTER (WHERE status='OK') / NULLIF(COUNT(*),0), 1) AS ok_rate_pct,
  COUNT(*) FILTER (WHERE status<>'OK') AS n_fail,
  COUNT(*) FILTER (WHERE difference_pct >= 2 AND status='OK') AS n_back_bs_gt_ws_2p,
  COUNT(*) FILTER (WHERE difference_pct <= -2 AND status='OK') AS n_lay_bs_lt_ws_2p,
  COUNT(*) FILTER (WHERE (hypothesis_details::jsonb -> 'lay') IS NOT NULL) AS n_lay_t0,
  COUNT(*) FILTER (WHERE (hypothesis_details::jsonb -> 'temporal') IS NOT NULL) AS n_back_temporal,
  COUNT(*) FILTER (WHERE (hypothesis_details::jsonb -> 'lay_temporal') IS NOT NULL) AS n_lay_temporal,
  COUNT(*) FILTER (WHERE (hypothesis_details::jsonb -> 'telemetry') IS NOT NULL) AS n_telemetry,
  ROUND(AVG(q_ms), 1) AS q_avg_ms,
  ROUND((PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY q_ms))::numeric, 1) AS q_p95_ms,
  ROUND(AVG(pipe_ms), 1) AS pipe_avg_ms,
  ROUND((PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY pipe_ms))::numeric, 1) AS pipe_p95_ms,
  ROUND(AVG(t_async_ms), 1) AS temporal_async_avg_ms,
  ROUND((PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY t_async_ms))::numeric, 1) AS temporal_async_p95_ms,
  COUNT(*) FILTER (WHERE temporal_deferred) AS n_temporal_deferred
FROM s;
"
echo

echo "3) EVOLUCAO POR BUCKET (${BUCKET_MINUTES} min)"
run_psql "
WITH base AS (
  SELECT *
  FROM betslip_audit_results
  WHERE audit_version='${AUDIT_VERSION}'
    AND audited_at >= timestamptz '${SINCE_UTC}'
),
s AS (
  SELECT
    to_timestamp(
      floor(extract(epoch from audited_at) / ${BUCKET_SEC}) * ${BUCKET_SEC}
    ) AS bucket_utc,
    status,
    difference_pct,
    (hypothesis_details::jsonb -> 'lay') IS NOT NULL AS has_lay_t0,
    (hypothesis_details::jsonb -> 'lay_temporal') IS NOT NULL AS has_lay_temporal,
    (hypothesis_details::jsonb -> 'telemetry') IS NOT NULL AS has_telemetry,
    NULLIF((hypothesis_details::jsonb -> 'telemetry' ->> 'queue_wait_ms'),'')::numeric AS q_ms,
    NULLIF((hypothesis_details::jsonb -> 'telemetry' ->> 'pipeline_total_ms'),'')::numeric AS pipe_ms
  FROM base
)
SELECT
  bucket_utc,
  COUNT(*) AS n_total,
  COUNT(*) FILTER (WHERE status='OK') AS n_ok,
  ROUND(100.0 * COUNT(*) FILTER (WHERE status='OK') / NULLIF(COUNT(*),0), 1) AS ok_pct,
  COUNT(*) FILTER (WHERE status='OK' AND difference_pct >= 2) AS n_back_edge_2p,
  COUNT(*) FILTER (WHERE status='OK' AND difference_pct <= -2) AS n_lay_edge_2p,
  ROUND(100.0 * COUNT(*) FILTER (WHERE has_lay_t0) / NULLIF(COUNT(*),0), 1) AS lay_t0_cov_pct,
  ROUND(100.0 * COUNT(*) FILTER (WHERE has_lay_temporal) / NULLIF(COUNT(*),0), 1) AS lay_temporal_cov_pct,
  ROUND(100.0 * COUNT(*) FILTER (WHERE has_telemetry) / NULLIF(COUNT(*),0), 1) AS telemetry_cov_pct,
  ROUND(AVG(q_ms), 1) AS q_avg_ms,
  ROUND((PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY q_ms))::numeric, 1) AS q_p95_ms,
  ROUND(AVG(pipe_ms), 1) AS pipe_avg_ms,
  ROUND((PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY pipe_ms))::numeric, 1) AS pipe_p95_ms
FROM s
GROUP BY 1
ORDER BY 1 DESC
LIMIT ${MAX_BUCKETS};
"
echo

echo "4) COMPARATIVO INICIO vs FIM DO PERIODO"
run_psql "
WITH base AS (
  SELECT *
  FROM betslip_audit_results
  WHERE audit_version='${AUDIT_VERSION}'
    AND audited_at >= timestamptz '${SINCE_UTC}'
),
bounds AS (
  SELECT
    MIN(audited_at) AS min_ts,
    MAX(audited_at) AS max_ts
  FROM base
),
tagged AS (
  SELECT
    CASE
      WHEN b.max_ts IS NULL OR b.min_ts IS NULL THEN 'SEM_DADOS'
      WHEN x.audited_at < b.min_ts + ((b.max_ts - b.min_ts) / 2) THEN 'INICIO_50%'
      ELSE 'FIM_50%'
    END AS faixa,
    x.status,
    x.difference_pct,
    NULLIF((x.hypothesis_details::jsonb -> 'telemetry' ->> 'queue_wait_ms'),'')::numeric AS q_ms,
    NULLIF((x.hypothesis_details::jsonb -> 'telemetry' ->> 'pipeline_total_ms'),'')::numeric AS pipe_ms
  FROM base x
  CROSS JOIN bounds b
)
SELECT
  faixa,
  COUNT(*) AS n_total,
  COUNT(*) FILTER (WHERE status='OK') AS n_ok,
  ROUND(100.0 * COUNT(*) FILTER (WHERE status='OK') / NULLIF(COUNT(*),0), 1) AS ok_pct,
  COUNT(*) FILTER (WHERE status='OK' AND difference_pct >= 2) AS n_back_edge_2p,
  COUNT(*) FILTER (WHERE status='OK' AND difference_pct <= -2) AS n_lay_edge_2p,
  ROUND(AVG(q_ms), 1) AS q_avg_ms,
  ROUND((PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY q_ms))::numeric, 1) AS q_p95_ms,
  ROUND(AVG(pipe_ms), 1) AS pipe_avg_ms,
  ROUND((PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY pipe_ms))::numeric, 1) AS pipe_p95_ms
FROM tagged
GROUP BY 1
ORDER BY faixa;
"
echo

echo "5) FALHAS DESDE O MARCO (status + amostra)"
run_psql "
WITH base AS (
  SELECT *
  FROM betslip_audit_results
  WHERE audit_version='${AUDIT_VERSION}'
    AND audited_at >= timestamptz '${SINCE_UTC}'
)
SELECT
  status,
  COUNT(*) AS n,
  ROUND(100.0 * COUNT(*) / NULLIF(SUM(COUNT(*)) OVER(),0), 1) AS pct_das_falhas
FROM base
WHERE status <> 'OK'
GROUP BY 1
ORDER BY n DESC, 1;
"
run_psql "
SELECT
  id,
  audited_at,
  status,
  home_team,
  away_team,
  market_type,
  line,
  side,
  websocket_odd,
  betslip_odd,
  difference_pct
FROM betslip_audit_results
WHERE audit_version='${AUDIT_VERSION}'
  AND audited_at >= timestamptz '${SINCE_UTC}'
  AND status <> 'OK'
ORDER BY audited_at DESC
LIMIT 15;
"
echo

echo "6) SAUDE DO TEMPORAL ASSINCRONO (SLA=${PENDING_THRESHOLD_MIN}m)"
run_psql "
WITH base AS (
  SELECT *
  FROM betslip_audit_results
  WHERE audit_version='${AUDIT_VERSION}'
    AND audited_at >= timestamptz '${SINCE_UTC}'
),
s AS (
  SELECT
    audited_at,
    COALESCE((hypothesis_details::jsonb -> 'telemetry' ->> 'temporal_deferred'),'false')::boolean AS temporal_deferred,
    (hypothesis_details::jsonb -> 'temporal') IS NOT NULL AS has_back_temporal,
    (hypothesis_details::jsonb -> 'lay_temporal') IS NOT NULL AS has_lay_temporal,
    NULLIF((hypothesis_details::jsonb -> 'telemetry' ->> 'temporal_async_latency_ms'),'')::numeric AS temporal_async_ms
  FROM base
)
SELECT
  COUNT(*) AS n_total,
  COUNT(*) FILTER (WHERE temporal_deferred) AS n_temporal_deferred,
  COUNT(*) FILTER (
    WHERE temporal_deferred
      AND NOT has_back_temporal
      AND audited_at < now() - interval '${PENDING_THRESHOLD_MIN} minutes'
  ) AS pending_back_gt_sla,
  COUNT(*) FILTER (
    WHERE temporal_deferred
      AND NOT has_lay_temporal
      AND audited_at < now() - interval '${PENDING_THRESHOLD_MIN} minutes'
  ) AS pending_lay_gt_sla,
  ROUND(AVG(temporal_async_ms),1) AS temporal_async_avg_ms,
  ROUND((PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY temporal_async_ms))::numeric,1) AS temporal_async_p95_ms
FROM s;
"
echo

echo "7) EVOLUCAO DO COLLECTOR (JSONL)"
python3 - "${COLLECTOR_TELEMETRY_FILE}" "${SINCE_UTC}" "${BUCKET_MINUTES}" "${MAX_BUCKETS}" <<'PY'
import datetime as dt
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

path = Path(sys.argv[1])
since_raw = sys.argv[2]
bucket_minutes = int(sys.argv[3])
max_rows = int(sys.argv[4])

if since_raw.endswith("+00"):
    since_raw = since_raw + ":00"
since_dt = dt.datetime.fromisoformat(since_raw)

if not path.exists():
    print(f"Arquivo ausente: {path}")
    sys.exit(0)

def parse_ts(v):
    if not v:
        return None
    if isinstance(v, str) and v.endswith("Z"):
        v = v[:-1] + "+00:00"
    try:
        d = dt.datetime.fromisoformat(v)
    except Exception:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d

def p95(values):
    vals = sorted([x for x in values if x is not None])
    if not vals:
        return None
    k = int(math.ceil(0.95 * len(vals))) - 1
    k = max(0, min(k, len(vals) - 1))
    return vals[k]

bucket_sec = bucket_minutes * 60
buckets = defaultdict(lambda: {
    "n_cycles": 0,
    "matches_saved": 0,
    "save_errors": 0,
    "collect_ms": [],
    "save_ms": [],
    "cycle_total_ms": [],
})

total_lines = 0
valid_lines = 0

with path.open("r", encoding="utf-8", errors="ignore") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        total_lines += 1
        try:
            obj = json.loads(line)
        except Exception:
            continue

        ts = parse_ts(obj.get("ts_utc") or obj.get("timestamp") or obj.get("ts"))
        if ts is None or ts < since_dt:
            continue

        valid_lines += 1
        epoch = int(ts.timestamp())
        bucket_epoch = (epoch // bucket_sec) * bucket_sec
        bdt = dt.datetime.fromtimestamp(bucket_epoch, tz=dt.timezone.utc)

        b = buckets[bdt]
        b["n_cycles"] += 1
        b["matches_saved"] += int(obj.get("matches_saved") or 0)
        b["save_errors"] += int(obj.get("save_errors") or 0)
        b["collect_ms"].append(int(obj.get("collect_ms") or 0))
        b["save_ms"].append(int(obj.get("save_ms") or 0))
        b["cycle_total_ms"].append(int(obj.get("cycle_total_ms") or 0))

print(f"linhas_total={total_lines} | linhas_desde_marco={valid_lines}")
if not buckets:
    print("Sem dados de collector no periodo.")
    sys.exit(0)

rows = sorted(buckets.items(), key=lambda x: x[0], reverse=True)[:max_rows]
print("bucket_utc               | ciclos | matches_saved | save_errors | collect_avg/p95_ms | save_avg/p95_ms | cycle_avg/p95_ms")
for ts, data in rows:
    collect_avg = round(sum(data["collect_ms"]) / len(data["collect_ms"]), 1) if data["collect_ms"] else None
    save_avg = round(sum(data["save_ms"]) / len(data["save_ms"]), 1) if data["save_ms"] else None
    cycle_avg = round(sum(data["cycle_total_ms"]) / len(data["cycle_total_ms"]), 1) if data["cycle_total_ms"] else None
    collect_p95 = p95(data["collect_ms"])
    save_p95 = p95(data["save_ms"])
    cycle_p95 = p95(data["cycle_total_ms"])
    print(
        f"{ts.strftime('%Y-%m-%d %H:%M:%S+00')} | "
        f"{data['n_cycles']:>6} | "
        f"{data['matches_saved']:>12} | "
        f"{data['save_errors']:>11} | "
        f"{collect_avg}/{collect_p95} | "
        f"{save_avg}/{save_p95} | "
        f"{cycle_avg}/{cycle_p95}"
    )
PY
echo

echo "====================================================================="
echo "Dica de uso para acompanhar mudancas recentes:"
echo "  bash monitor_evolution_v4.sh --since-service-restart --bucket-minutes 10 --max-buckets 72"
echo "Ou para comparar um deploy especifico:"
echo "  bash monitor_evolution_v4.sh --since \"2026-02-13 16:12:00\" --bucket-minutes 10"
echo "====================================================================="
