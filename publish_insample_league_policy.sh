#!/usr/bin/env bash
set -euo pipefail

# Publica policy operacional (wf_policy_current.json) baseada em ROI in-sample por liga.
# Critério default: ROI > 2% (sem N mínimo), Back Pre, slippage_pre_pct < 0.

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

LOOKBACK_DAYS="${INSAMPLE_LOOKBACK_DAYS:-45}"
ROI_MIN="${INSAMPLE_ROI_MIN:-2}"
N_MIN="${INSAMPLE_N_MIN:-0}"
HYP_TYPE="${INSAMPLE_HYPOTHESIS_TYPE:-H3B}"
REV_DIR="${INSAMPLE_REVERSAL_DIRECTION:-up}"
REQ_SLIP_NEG="${INSAMPLE_REQUIRE_SLIPPAGE_NEG:-1}"

POLICY_CURRENT="${INSAMPLE_POLICY_CURRENT:-$BOT_DIR/logs/wf_policy_current.json}"
POLICY_HISTORY_DIR="${INSAMPLE_POLICY_HISTORY_DIR:-$BOT_DIR/logs/policy_history}"
POLICY_HISTORY_JSONL="${INSAMPLE_POLICY_HISTORY_JSONL:-$BOT_DIR/logs/wf_policy_history.jsonl}"
APPROVED_CSV="${INSAMPLE_APPROVED_CSV:-$BOT_DIR/logs/approved_leagues_insample_45d.csv}"
NOT_APPROVED_CSV="${INSAMPLE_NOT_APPROVED_CSV:-$BOT_DIR/logs/not_approved_leagues_insample_45d.csv}"
TMP_ALL_CSV="$(mktemp /tmp/league_roi_all.XXXXXX.csv)"

trap 'rm -f "$TMP_ALL_CSV"' EXIT

for v in LOOKBACK_DAYS N_MIN; do
  if ! [[ "${!v}" =~ ^[0-9]+$ ]]; then
    echo "[ERRO] $v invalido: ${!v}" >&2
    exit 2
  fi
done

mkdir -p "$(dirname "$POLICY_CURRENT")" "$POLICY_HISTORY_DIR" "$(dirname "$POLICY_HISTORY_JSONL")" "$(dirname "$APPROVED_CSV")"

if [[ "$REQ_SLIP_NEG" == "1" ]]; then
  SLIP_FILTER_SQL="AND COALESCE(
      NULLIF(a.hypothesis_details #>> '{value_sizing,slippage_pre_pct}','')::double precision,
      NULLIF(a.hypothesis_details #>> '{finance,value_sizing,slippage_pre_pct}','')::double precision,
      NULLIF(a.hypothesis_details #>> '{slippage_pre_pct}','')::double precision
    ) < 0"
else
  SLIP_FILTER_SQL=""
fi

psql "$DATABASE_URL" -v ON_ERROR_STOP=1 <<SQL
DROP TABLE IF EXISTS tmp_league_roi;
CREATE TEMP TABLE tmp_league_roi AS
WITH src AS (
  SELECT
    a.league,
    lower(coalesce(a.side,'')) AS side_lc,
    replace(replace(a.line::text, ',', '.'), '−', '-') AS raw_line,
    a.betslip_odd AS bs_odd,
    upper(coalesce(a.status,'')) AS status_u,
    lower(coalesce(m.status,'')) AS match_status,
    m.kickoff_time,
    m.home_score,
    m.away_score
  FROM betslip_audit_results a
  JOIN matches m ON m.external_id = a.event_id
  WHERE a.hypothesis_type = '$HYP_TYPE'
    AND a.reversal_direction = '$REV_DIR'
    AND a.audited_at >= now() - interval '$LOOKBACK_DAYS days'
    AND a.audited_at < now()
    AND a.audited_at < m.kickoff_time
    AND upper(a.status) IN ('OK','GATE_NOT_ELIGIBLE','GATE_BLOCKED_CAP','GATE_BLOCKED_BACKOFF')
    AND lower(coalesce(m.status,'')) IN ('finished','ended','closed','settled','full_time','fulltime','ft')
    $SLIP_FILTER_SQL
    AND a.betslip_odd IS NOT NULL AND a.betslip_odd > 0
    AND m.home_score IS NOT NULL AND m.away_score IS NOT NULL
),
calc AS (
  SELECT
    *,
    (home_score::int - away_score::int) AS goal_diff,
    CASE
      WHEN raw_line ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' THEN
        CASE
          WHEN raw_line ~ '^[+-]' THEN raw_line::double precision
          WHEN side_lc='home' THEN raw_line::double precision
          WHEN side_lc='away' THEN -raw_line::double precision
          ELSE NULL
        END
      ELSE NULL
    END AS home_handicap
  FROM src
  WHERE side_lc IN ('home','away')
),
roi_events AS (
  SELECT
    league,
    CASE
      WHEN side_lc='home' THEN goal_diff + home_handicap
      WHEN side_lc='away' THEN -goal_diff - home_handicap
      ELSE NULL
    END AS adjusted,
    bs_odd
  FROM calc
  WHERE home_handicap IS NOT NULL
)
SELECT
  league,
  COUNT(*) AS n,
  AVG(
    CASE
      WHEN adjusted >  0.25 THEN (bs_odd - 1.0) * 100.0
      WHEN adjusted =  0.25 THEN (bs_odd - 1.0) *  50.0
      WHEN adjusted =  0.00 THEN 0.0
      WHEN adjusted = -0.25 THEN -50.0
      ELSE -100.0
    END
  ) AS roiw_pct
FROM roi_events
GROUP BY league;

\copy (
  SELECT league, n, ROUND(roiw_pct::numeric,6) AS roiw_pct
  FROM tmp_league_roi
  ORDER BY roiw_pct DESC, n DESC
) TO '$TMP_ALL_CSV' CSV HEADER

\copy (
  SELECT league, n, ROUND(roiw_pct::numeric,6) AS roiw_pct
  FROM tmp_league_roi
  WHERE n >= $N_MIN AND roiw_pct > $ROI_MIN
  ORDER BY roiw_pct DESC, n DESC
) TO '$APPROVED_CSV' CSV HEADER

\copy (
  SELECT
    league,
    n,
    ROUND(roiw_pct::numeric,6) AS roiw_pct,
    CASE
      WHEN n < $N_MIN THEN 'N<$N_MIN'
      WHEN roiw_pct <= $ROI_MIN THEN 'ROI<=$ROI_MIN'
      ELSE 'OUTRO'
    END AS motivo
  FROM tmp_league_roi
  WHERE NOT (n >= $N_MIN AND roiw_pct > $ROI_MIN)
  ORDER BY n DESC, roiw_pct DESC
) TO '$NOT_APPROVED_CSV' CSV HEADER
SQL

python3 - "$APPROVED_CSV" "$TMP_ALL_CSV" "$POLICY_CURRENT" "$POLICY_HISTORY_DIR" "$POLICY_HISTORY_JSONL" "$LOOKBACK_DAYS" "$ROI_MIN" "$N_MIN" "$REQ_SLIP_NEG" <<'PY'
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

approved_csv = Path(sys.argv[1])
all_csv = Path(sys.argv[2])
policy_current = Path(sys.argv[3])
policy_history_dir = Path(sys.argv[4])
policy_history_jsonl = Path(sys.argv[5])
lookback_days = int(sys.argv[6])
roi_min = float(sys.argv[7])
n_min = int(sys.argv[8])
require_slip = str(sys.argv[9]) == "1"

rows_all = list(csv.DictReader(all_csv.open("r", encoding="utf-8")))
rows_ok = list(csv.DictReader(approved_csv.open("r", encoding="utf-8")))

def norm_league(s: str) -> str:
    x = (s or "").strip()
    if not x:
        return "—"
    x = x.replace("|", "/").replace("\n", " ").replace("\r", " ").strip()
    if len(x) > 48:
        x = x[:48].rstrip() + "…"
    return x

approved_leagues = [norm_league(r.get("league", "")) for r in rows_ok]
active_keys = [f"Back_Pre_Any__{lg}" for lg in approved_leagues]
active_counts = {k: 1 for k in active_keys}

generated_at = datetime.now(timezone.utc).isoformat()
step = {
    "train": f"last_{lookback_days}d",
    "test": "IN_SAMPLE_SQL",
    "selection_mode": "insample_sql",
    "active_keys": active_keys,
    "active_n": len(active_keys),
    "active_keys_base": (["Back_Pre_Any"] if active_keys else []),
    "active_n_base": (1 if active_keys else 0),
    "approved_leagues": approved_leagues,
    "n_ev_elig_pre": sum(int(float(r.get("n", 0) or 0)) for r in rows_ok),
    "diag": {
        r.get("league", ""): {
            "ok": True,
            "reason": f"INSAMPLE_SQL: n>={n_min} and ROI>{roi_min}",
            "train_matches_roi": int(float(r.get("n", 0) or 0)),
            "roi_mean": float(r.get("roiw_pct", 0) or 0),
            "roi_mean_eff": float(r.get("roiw_pct", 0) or 0),
        }
        for r in rows_ok
    },
}

policy = {
    "generated_at": generated_at,
    "report_out": None,
    "lookback_days": lookback_days,
    "versions": [],
    "walkforward": True,
    "wf": {
        "train_mode": "expanding",
        "selection_mode": "insample_sql",
        "insample_days": lookback_days,
        "train_days": lookback_days,
        "test_days": 1,
        "step_days": 1,
        "min_matches": n_min,
        "pre_activation_mode": "roi_only",
        "roi_min_activate": roi_min,
        "roi_require_finished": True,
        "sides": "back",
        "regimes": "pre",
        "backpre_slip_max": (0 if require_slip else None),
        "backpre_slip_field": ("slippage_pre_pct" if require_slip else "disabled"),
        "backpre_fast_max_lag_ms": None,
        "key_by_league": True,
        "key_by_league_scope": "pre",
        "liquidity_mode": "none",
        "liquidity_scope": "pre",
        "liquidity_min_limit": 0.0,
        "ah_max_abs_line": 0.0,
        "ah_scope": "all",
        "scheme_pre": "KELLY_0.25",
        "scheme_in": "FLAT",
        "match_budget": True,
        "budget_back_frac": 0.01,
        "budget_lay_frac": 0.005,
        "budget_cap_signal_frac": 0.33,
        "budget_risk_mode": "fixed",
        "shrinkage": False,
        "exclude_exec_buckets": "",
        "exclude_exec_buckets_back": "",
        "exclude_exec_buckets_lay": "",
    },
    "steps": [step],
    "active_counts": active_counts,
    "insample_sql_summary": {
        "approved_n": len(rows_ok),
        "all_leagues_n": len(rows_all),
    },
}

policy_current.parent.mkdir(parents=True, exist_ok=True)
policy_current.write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")

policy_history_dir.mkdir(parents=True, exist_ok=True)
ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
hist_file = policy_history_dir / f"wf_policy_{ts}.json"
hist_file.write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")

policy_history_jsonl.parent.mkdir(parents=True, exist_ok=True)
with policy_history_jsonl.open("a", encoding="utf-8") as f:
    f.write(json.dumps({
        "ts": generated_at,
        "policy_current": str(policy_current),
        "policy_history_file": str(hist_file),
        "approved_n": len(rows_ok),
        "all_leagues_n": len(rows_all),
        "mode": "insample_sql",
        "roi_min": roi_min,
        "n_min": n_min,
        "lookback_days": lookback_days,
    }, ensure_ascii=False) + "\n")

print(f"approved_n={len(rows_ok)}")
print(f"policy_current={policy_current}")
print(f"policy_history={hist_file}")
PY

echo "[OK] Policy in-sample SQL publicada."
echo "[OK] approved_csv=$APPROVED_CSV"
echo "[OK] not_approved_csv=$NOT_APPROVED_CSV"
