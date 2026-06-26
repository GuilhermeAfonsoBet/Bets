#!/usr/bin/env bash
set -euo pipefail

# Sincroniza slippage_pre_pct do EXECUTOR_JSONL para betslip_audit_results.hypothesis_details.
# Objetivo: manter OOS/queries (que leem do banco) alinhados com o daily (que lê JSONL).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOT_DIR="${BOT_DIR:-$ROOT_DIR/betinasia_bot}"
ENV_FILE_CANDIDATE="${ENV_FILE:-$BOT_DIR/.env}"

if [[ -z "${DATABASE_URL:-}" && -f "$ENV_FILE_CANDIDATE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE_CANDIDATE"
  set +a
fi

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "[ERRO] DATABASE_URL nao definido." >&2
  exit 2
fi

JSONL_PATH="${EXECUTOR_JSONL:-$BOT_DIR/logs/executor_live.jsonl}"
LOOKBACK_DAYS="${SLIPPAGE_SYNC_LOOKBACK_DAYS:-45}"
HYP_TYPE="${SLIPPAGE_SYNC_HYPOTHESIS_TYPE:-H3B}"
REV_DIR="${SLIPPAGE_SYNC_REVERSAL_DIRECTION:-up}"

if ! [[ "$LOOKBACK_DAYS" =~ ^[0-9]+$ ]]; then
  echo "[ERRO] SLIPPAGE_SYNC_LOOKBACK_DAYS invalido: $LOOKBACK_DAYS" >&2
  exit 2
fi

if [[ ! -f "$JSONL_PATH" ]]; then
  echo "[WARN] EXECUTOR_JSONL nao encontrado: $JSONL_PATH (nada para sincronizar)"
  exit 0
fi

TMP_TSV="$(mktemp /tmp/slippage_backfill.XXXXXX.tsv)"
trap 'rm -f "$TMP_TSV"' EXIT

python3 - "$JSONL_PATH" "$TMP_TSV" <<'PY'
import json
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

def parse_dt(x):
    s = str(x or "").strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None

latest = OrderedDict()
for ln in src.read_text(encoding="utf-8", errors="ignore").splitlines():
    try:
        o = json.loads(ln)
    except Exception:
        continue
    req = o.get("request") if isinstance(o.get("request"), dict) else {}
    res = o.get("result") if isinstance(o.get("result"), dict) else {}
    status = str(res.get("status", "")).upper()
    if status not in {"LIVE_OK", "CAP_BLOCKED"}:
        continue
    side = str(res.get("exec_side") or req.get("exec_side") or "").lower()
    if side != "back":
        continue

    aid = res.get("audit_id") or req.get("audit_id")
    if aid is None:
        continue
    try:
        aid = int(aid)
    except Exception:
        continue

    raw = res.get("raw") if isinstance(res.get("raw"), dict) else {}
    vs = raw.get("value_sizing") if isinstance(raw.get("value_sizing"), dict) else {}
    slip = vs.get("slippage_pre_pct")
    if slip is None:
        continue
    try:
        slip = float(slip)
    except Exception:
        continue

    pre_ms = vs.get("pre_submit_ms")
    try:
        pre_ms = int(pre_ms) if pre_ms is not None else None
    except Exception:
        pre_ms = None

    source = str(vs.get("source") or "")
    created = parse_dt(res.get("created_at") or req.get("created_at")) or datetime.now(timezone.utc)
    prev = latest.get(aid)
    if prev is None or created >= prev[0]:
        latest[aid] = (created, slip, pre_ms, source)

with dst.open("w", encoding="utf-8") as f:
    for aid, (_dt, slip, pre_ms, source) in latest.items():
        pre_s = "" if pre_ms is None else str(pre_ms)
        f.write(f"{aid}\t{slip}\t{pre_s}\t{source}\n")

print(f"rows={len(latest)}")
PY

ROWS="$(wc -l < "$TMP_TSV" | tr -d ' ')"
if [[ "$ROWS" == "0" ]]; then
  echo "[INFO] Nenhuma linha com slippage_pre_pct no JSONL para sincronizar."
  exit 0
fi

echo "[INFO] Sincronizando $ROWS linhas de slippage para hypothesis_details..."

psql "$DATABASE_URL" -v ON_ERROR_STOP=1 <<SQL
CREATE TEMP TABLE tmp_slip_backfill (
  audit_id BIGINT PRIMARY KEY,
  slippage_pre_pct DOUBLE PRECISION,
  pre_submit_ms INTEGER,
  source TEXT
);
\copy tmp_slip_backfill FROM '$TMP_TSV' WITH (FORMAT text, DELIMITER E'\t', NULL '')

UPDATE betslip_audit_results a
SET hypothesis_details = (
  jsonb_set(
    jsonb_set(
      jsonb_set(
        jsonb_set(
          COALESCE(a.hypothesis_details::jsonb, '{}'::jsonb),
          '{value_sizing}',
          CASE
            WHEN jsonb_typeof(COALESCE(a.hypothesis_details::jsonb, '{}'::jsonb)->'value_sizing') = 'object'
              THEN COALESCE(a.hypothesis_details::jsonb, '{}'::jsonb)->'value_sizing'
            ELSE '{}'::jsonb
          END,
          true
        ),
        '{value_sizing,slippage_pre_pct}',
        to_jsonb(t.slippage_pre_pct),
        true
      ),
      '{value_sizing,pre_submit_ms}',
      to_jsonb(t.pre_submit_ms),
      true
    ),
    '{value_sizing,source}',
    to_jsonb(t.source),
    true
  )
)::json
FROM tmp_slip_backfill t
WHERE a.id = t.audit_id
  AND a.hypothesis_type = '$HYP_TYPE'
  AND a.reversal_direction = '$REV_DIR'
  AND a.audited_at >= now() - interval '$LOOKBACK_DAYS days';

SELECT
  COUNT(*) FILTER (
    WHERE NULLIF(hypothesis_details #>> '{value_sizing,slippage_pre_pct}','') IS NOT NULL
  ) AS has_slippage_pre_pct_window
FROM betslip_audit_results
WHERE hypothesis_type = '$HYP_TYPE'
  AND reversal_direction = '$REV_DIR'
  AND audited_at >= now() - interval '$LOOKBACK_DAYS days';
SQL

echo "[OK] Sincronizacao de slippage concluida."
