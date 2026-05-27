#!/usr/bin/env bash
set -euo pipefail

# Wrapper "fail-closed" para rodar o daily no modo ROI-only Back Pre
# sem depender de CWD e sem depender de defaults frágeis.
#
# Uso rápido:
#   bash vps_run_daily_roi_only.sh
#   ENV_FILE=/caminho/.env DAILY_REPORT_OUT_DIR=logs/daily_reports bash vps_run_daily_roi_only.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOT_DIR_DEFAULT="$ROOT_DIR/betinasia_bot"
BOT_DIR="${BOT_DIR:-$BOT_DIR_DEFAULT}"
ENV_FILE_CANDIDATE="${ENV_FILE:-}"

if [[ ! -d "$BOT_DIR" ]]; then
  echo "[ERRO] Diretório do bot não encontrado: $BOT_DIR" >&2
  echo "       Defina BOT_DIR=/caminho/para/betinasia_bot" >&2
  exit 2
fi

if [[ -z "$ENV_FILE_CANDIDATE" ]]; then
  if [[ -f "$BOT_DIR/.env" ]]; then
    ENV_FILE_CANDIDATE="$BOT_DIR/.env"
  elif [[ -f "$ROOT_DIR/.env" ]]; then
    ENV_FILE_CANDIDATE="$ROOT_DIR/.env"
  else
    ENV_FILE_CANDIDATE="$BOT_DIR/.env"
  fi
fi

mkdir -p "$BOT_DIR/logs"

export ENV_FILE="$ENV_FILE_CANDIDATE"

# Hardening de parâmetros críticos para o cenário pedido:
# - OOS expanding
# - ativação pre-match por ROI_ONLY
# - Back Pre apenas
# - slippage <= 0 (proxy diff_pct)
export DAILY_WF_TRAIN_MODE="${DAILY_WF_TRAIN_MODE:-expanding}"
export DAILY_WF_PRE_ACTIVATION_MODE="${DAILY_WF_PRE_ACTIVATION_MODE:-roi_only}"
export DAILY_WF_ROI_MIN_ACTIVATE="${DAILY_WF_ROI_MIN_ACTIVATE:-0}"
export DAILY_WF_SIDES="${DAILY_WF_SIDES:-back}"
export DAILY_WF_REGIMES="${DAILY_WF_REGIMES:-pre}"
export DAILY_WF_KEY_BY_LEAGUE="${DAILY_WF_KEY_BY_LEAGUE:-1}"
export DAILY_WF_KEY_BY_LEAGUE_SCOPE="${DAILY_WF_KEY_BY_LEAGUE_SCOPE:-pre}"
export DAILY_WF_BACKPRE_SLIP_MAX="${DAILY_WF_BACKPRE_SLIP_MAX:-0}"
export DAILY_WF_BACKPRE_SLIP_FIELD="${DAILY_WF_BACKPRE_SLIP_FIELD:-diff_pct}"

echo "[INFO] BOT_DIR=$BOT_DIR"
echo "[INFO] ENV_FILE=$ENV_FILE"
echo "[INFO] DAILY_WF_TRAIN_MODE=$DAILY_WF_TRAIN_MODE"
echo "[INFO] DAILY_WF_PRE_ACTIVATION_MODE=$DAILY_WF_PRE_ACTIVATION_MODE"
echo "[INFO] DAILY_WF_ROI_MIN_ACTIVATE=$DAILY_WF_ROI_MIN_ACTIVATE"
echo "[INFO] DAILY_WF_SIDES=$DAILY_WF_SIDES"
echo "[INFO] DAILY_WF_REGIMES=$DAILY_WF_REGIMES"
echo "[INFO] DAILY_WF_KEY_BY_LEAGUE=$DAILY_WF_KEY_BY_LEAGUE"
echo "[INFO] DAILY_WF_KEY_BY_LEAGUE_SCOPE=$DAILY_WF_KEY_BY_LEAGUE_SCOPE"
echo "[INFO] DAILY_WF_BACKPRE_SLIP_MAX=$DAILY_WF_BACKPRE_SLIP_MAX"
echo "[INFO] DAILY_WF_BACKPRE_SLIP_FIELD=$DAILY_WF_BACKPRE_SLIP_FIELD"

RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
OUT_JSON="$BOT_DIR/logs/daily_roi_only_run_${RUN_TS}.json"
POLICY_JSON="${DAILY_WF_POLICY_CURRENT:-$BOT_DIR/logs/wf_policy_current.json}"

echo "[INFO] Rodando daily_full_report..."
(
  cd "$BOT_DIR"
  python3 -m ops.daily_full_report \
    --out-dir "${DAILY_REPORT_OUT_DIR:-logs/daily_reports}" \
    >"$OUT_JSON"
)
echo "[OK] Saída do daily salva em: $OUT_JSON"

if [[ ! -f "$POLICY_JSON" ]]; then
  echo "[ERRO] Policy não encontrada após execução: $POLICY_JSON" >&2
  exit 3
fi

echo "[INFO] Validando policy efetiva em $POLICY_JSON ..."
python3 - "$POLICY_JSON" <<'PY'
import json
import sys
from pathlib import Path

policy_path = Path(sys.argv[1])
data = json.loads(policy_path.read_text(encoding="utf-8"))
wf = data.get("wf") if isinstance(data, dict) else None
if not isinstance(wf, dict):
    raise SystemExit("[ERRO] policy.wf ausente/ inválido")

checks = {
    "pre_activation_mode": ("roi_only", str(wf.get("pre_activation_mode", "")).strip().lower()),
    "sides": ("back", str(wf.get("sides", "")).strip().lower()),
    "regimes": ("pre", str(wf.get("regimes", "")).strip().lower()),
    "train_mode": ("expanding", str(wf.get("train_mode", "")).strip().lower()),
    "backpre_slip_field": ("diff_pct", str(wf.get("backpre_slip_field", "")).strip().lower()),
}

failed = []
for k, (want, got) in checks.items():
    if got != want:
        failed.append(f"{k}: esperado={want} obtido={got}")

slip_max = wf.get("backpre_slip_max")
try:
    slip_val = float(slip_max)
except Exception:
    failed.append(f"backpre_slip_max inválido: {slip_max!r}")
else:
    if slip_val > 0.0:
        failed.append(f"backpre_slip_max esperado <=0, obtido={slip_val}")

if failed:
    msg = "\n".join(f"- {x}" for x in failed)
    raise SystemExit("[ERRO] Policy fora do esperado:\n" + msg)

print("[OK] Policy validada com sucesso (ROI_ONLY Back Pre + slippage<=0).")
PY

echo "[SUCESSO] Execução concluída e policy validada."
