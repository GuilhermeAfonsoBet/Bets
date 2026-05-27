#!/usr/bin/env bash
set -euo pipefail

# Wrapper fail-closed para rodar o daily no modo ROI-only Back Pre.
# Resiliente a layouts diferentes de repositorio e com bootstrap opcional de deps.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUESTED_BOT_DIR="${BOT_DIR:-$ROOT_DIR/betinasia_bot}"
REQUESTED_ENV_FILE="${ENV_FILE:-}"

if [[ ! -d "$REQUESTED_BOT_DIR" ]]; then
  echo "[ERRO] Diretorio BOT_DIR nao encontrado: $REQUESTED_BOT_DIR" >&2
  echo "       Defina BOT_DIR para a pasta correta do projeto." >&2
  exit 2
fi

# Hardening dos parametros criticos do cenario pedido.
export DAILY_WF_TRAIN_MODE="${DAILY_WF_TRAIN_MODE:-expanding}"
export DAILY_WF_PRE_ACTIVATION_MODE="${DAILY_WF_PRE_ACTIVATION_MODE:-roi_only}"
export DAILY_WF_ROI_MIN_ACTIVATE="${DAILY_WF_ROI_MIN_ACTIVATE:-0}"
export DAILY_WF_SIDES="${DAILY_WF_SIDES:-back}"
export DAILY_WF_REGIMES="${DAILY_WF_REGIMES:-pre}"
export DAILY_WF_KEY_BY_LEAGUE="${DAILY_WF_KEY_BY_LEAGUE:-1}"
export DAILY_WF_KEY_BY_LEAGUE_SCOPE="${DAILY_WF_KEY_BY_LEAGUE_SCOPE:-pre}"
export DAILY_WF_BACKPRE_SLIP_MAX="${DAILY_WF_BACKPRE_SLIP_MAX:-0}"
export DAILY_WF_BACKPRE_SLIP_FIELD="${DAILY_WF_BACKPRE_SLIP_FIELD:-diff_pct}"

echo "[INFO] REQUESTED_BOT_DIR=$REQUESTED_BOT_DIR"
echo "[INFO] REQUESTED_ENV_FILE=${REQUESTED_ENV_FILE:-<vazio>}"
echo "[INFO] DAILY_WF_TRAIN_MODE=$DAILY_WF_TRAIN_MODE"
echo "[INFO] DAILY_WF_PRE_ACTIVATION_MODE=$DAILY_WF_PRE_ACTIVATION_MODE"
echo "[INFO] DAILY_WF_ROI_MIN_ACTIVATE=$DAILY_WF_ROI_MIN_ACTIVATE"
echo "[INFO] DAILY_WF_SIDES=$DAILY_WF_SIDES"
echo "[INFO] DAILY_WF_REGIMES=$DAILY_WF_REGIMES"
echo "[INFO] DAILY_WF_KEY_BY_LEAGUE=$DAILY_WF_KEY_BY_LEAGUE"
echo "[INFO] DAILY_WF_KEY_BY_LEAGUE_SCOPE=$DAILY_WF_KEY_BY_LEAGUE_SCOPE"
echo "[INFO] DAILY_WF_BACKPRE_SLIP_MAX=$DAILY_WF_BACKPRE_SLIP_MAX"
echo "[INFO] DAILY_WF_BACKPRE_SLIP_FIELD=$DAILY_WF_BACKPRE_SLIP_FIELD"

ROOT_A="$REQUESTED_BOT_DIR"
ROOT_B="$ROOT_DIR"
ROOT_C="$(dirname "$REQUESTED_BOT_DIR")"

DAILY_SCRIPT="$({
  python3 - "$ROOT_A" "$ROOT_B" "$ROOT_C" <<'PY_FIND_DAILY'
import os
import sys

roots = []
seen = set()
for r in sys.argv[1:]:
    rr = os.path.abspath(r)
    if rr in seen or not os.path.isdir(rr):
        continue
    seen.add(rr)
    roots.append(rr)

skip = {".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache", ".pytest_cache"}
cands = []
for root in roots:
    for cur, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip]
        if "daily_full_report.py" not in files:
            continue
        p = os.path.join(cur, "daily_full_report.py")
        rel = os.path.relpath(p, root).replace("\\", "/")
        score = 0
        if rel.endswith("ops/daily_full_report.py"):
            score -= 20
        npath = p.replace("\\", "/")
        if "/betinasia_bot/ops/" in npath:
            score -= 10
        if "/archive/" in npath or "/old/" in npath:
            score += 30
        cands.append((score, len(p), p))

if cands:
    cands.sort()
    print(cands[0][2])
PY_FIND_DAILY
} || true)"

if [[ -z "$DAILY_SCRIPT" || ! -f "$DAILY_SCRIPT" ]]; then
  echo "[ERRO] Nao encontrei daily_full_report.py por caminho de arquivo." >&2
  echo "       Roots pesquisados:" >&2
  echo "       - $ROOT_A" >&2
  echo "       - $ROOT_B" >&2
  echo "       - $ROOT_C" >&2
  echo "[DICA] Esta branch pode nao conter o pipeline operacional (betinasia_bot/ops)." >&2
  echo "[DICA] Troque para uma branch com o bot completo, ex.:" >&2
  echo "       git checkout cursor/wf-root-cause-fix-dc34" >&2
  echo "       # ou git checkout cursor/strict-policy-latency-regime-dc34" >&2
  exit 4
fi

RUN_CWD="$(dirname "$(dirname "$DAILY_SCRIPT")")"
WORK_ROOT="$RUN_CWD"
RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$WORK_ROOT/logs"
OUT_JSON="$WORK_ROOT/logs/daily_roi_only_run_${RUN_TS}.json"
ERR_LOG="$WORK_ROOT/logs/daily_roi_only_run_${RUN_TS}.stderr.log"

# Resolve ENV_FILE final com fallback, sem falhar se inexistente.
ENV_FILE_CANDIDATE=""
if [[ -n "$REQUESTED_ENV_FILE" && -f "$REQUESTED_ENV_FILE" ]]; then
  ENV_FILE_CANDIDATE="$REQUESTED_ENV_FILE"
elif [[ -f "$REQUESTED_BOT_DIR/.env" ]]; then
  ENV_FILE_CANDIDATE="$REQUESTED_BOT_DIR/.env"
elif [[ -f "$RUN_CWD/.env" ]]; then
  ENV_FILE_CANDIDATE="$RUN_CWD/.env"
elif [[ -f "$ROOT_DIR/.env" ]]; then
  ENV_FILE_CANDIDATE="$ROOT_DIR/.env"
fi
if [[ -n "$ENV_FILE_CANDIDATE" ]]; then
  export ENV_FILE="$ENV_FILE_CANDIDATE"
  echo "[INFO] ENV_FILE=$ENV_FILE"
else
  echo "[WARN] .env nao encontrado automaticamente; seguindo sem ENV_FILE explicito."
fi

# Bootstrap de runtime Python (opcional, mas ligado por padrao).
USE_VENV="${USE_VENV:-1}"
AUTO_INSTALL_DEPS="${AUTO_INSTALL_DEPS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${PY_VENV_DIR:-$WORK_ROOT/.venv-roi-only}"
REQ_FILE=""

if [[ "$USE_VENV" == "1" ]]; then
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "[INFO] Criando virtualenv em $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
  fi
  PYTHON_BIN="$VENV_DIR/bin/python"
fi

for rf in \
  "$WORK_ROOT/requirements.txt" \
  "$REQUESTED_BOT_DIR/requirements.txt" \
  "$WORK_ROOT/betinasia_bot/requirements.txt"
do
  if [[ -f "$rf" ]]; then
    REQ_FILE="$rf"
    break
  fi
done

if [[ "$AUTO_INSTALL_DEPS" == "1" && -n "$REQ_FILE" ]]; then
  if command -v sha256sum >/dev/null 2>&1; then
    REQ_HASH="$(sha256sum "$REQ_FILE" | awk '{print $1}')"
  else
    REQ_HASH="$(wc -c < "$REQ_FILE" | awk '{print $1}')"
  fi
  STAMP_FILE="$VENV_DIR/.deps_${REQ_HASH}.ok"
  if [[ ! -f "$STAMP_FILE" ]]; then
    echo "[INFO] Instalando dependencias de $REQ_FILE ..."
    "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
    "$PYTHON_BIN" -m pip install -r "$REQ_FILE"
    touch "$STAMP_FILE"
  else
    echo "[INFO] Dependencias ja instaladas para hash $REQ_HASH"
  fi
elif [[ "$AUTO_INSTALL_DEPS" == "1" && -z "$REQ_FILE" ]]; then
  echo "[WARN] requirements.txt nao encontrado; seguindo sem instalacao automatica de deps."
fi

PARENT_CWD="$(dirname "$RUN_CWD")"
export PYTHONPATH="$RUN_CWD:$PARENT_CWD:$REQUESTED_BOT_DIR:${PYTHONPATH:-}"
echo "[INFO] PYTHON_BIN=$PYTHON_BIN"

if [[ -n "${DAILY_WF_POLICY_CURRENT:-}" ]]; then
  if [[ "$DAILY_WF_POLICY_CURRENT" = /* ]]; then
    POLICY_JSON="$DAILY_WF_POLICY_CURRENT"
  else
    POLICY_JSON="$RUN_CWD/$DAILY_WF_POLICY_CURRENT"
  fi
else
  POLICY_JSON="$WORK_ROOT/logs/wf_policy_current.json"
fi

echo "[INFO] Rodando daily_full_report..."
echo "[INFO] RUN_CWD=$RUN_CWD"
echo "[INFO] DAILY_MODE=auto(module->script)"
echo "[INFO] DAILY_ENTRY=$DAILY_SCRIPT"
echo "[INFO] PYTHONPATH=$PYTHONPATH"
(
  cd "$RUN_CWD"
  OUT_DIR_ARG="${DAILY_REPORT_OUT_DIR:-$WORK_ROOT/logs/daily_reports}"
  RUN_OK=0

  MODULE_CANDIDATES=()
  if [[ -f "$RUN_CWD/ops/__init__.py" ]]; then
    MODULE_CANDIDATES+=("ops.daily_full_report")
  fi
  if [[ -f "$PARENT_CWD/betinasia_bot/ops/__init__.py" ]]; then
    MODULE_CANDIDATES+=("betinasia_bot.ops.daily_full_report")
  fi

  for MOD in "${MODULE_CANDIDATES[@]}"; do
    echo "[INFO] Tentando modulo: $MOD"
    if "$PYTHON_BIN" -m "$MOD" --out-dir "$OUT_DIR_ARG" >"$OUT_JSON" 2>"$ERR_LOG"; then
      echo "[INFO] Execucao por modulo OK: $MOD"
      RUN_OK=1
      break
    fi
  done

  if [[ "$RUN_OK" != "1" ]]; then
    echo "[INFO] Fallback para execucao por script: $DAILY_SCRIPT"
    if "$PYTHON_BIN" "$DAILY_SCRIPT" --out-dir "$OUT_DIR_ARG" >"$OUT_JSON" 2>"$ERR_LOG"; then
      RUN_OK=1
    fi
  fi

  if [[ "$RUN_OK" != "1" ]]; then
    echo "[ERRO] Falha ao executar daily_full_report.py (modulo e script)." >&2
    echo "[ERRO] stderr tail:" >&2
    tail -n 120 "$ERR_LOG" >&2 || true
    exit 5
  fi
)
echo "[OK] Saida do daily salva em: $OUT_JSON"

if [[ ! -f "$POLICY_JSON" ]]; then
  POLICY_JSON_FALLBACK="$({
    python3 - "$ROOT_A" "$ROOT_B" "$ROOT_C" <<'PY_FIND_POLICY'
import os
import sys

roots = []
seen = set()
for r in sys.argv[1:]:
    rr = os.path.abspath(r)
    if rr in seen or not os.path.isdir(rr):
        continue
    seen.add(rr)
    roots.append(rr)

best = None
for root in roots:
    for cur, _, files in os.walk(root):
        if "wf_policy_current.json" not in files:
            continue
        p = os.path.join(cur, "wf_policy_current.json")
        try:
            mt = os.path.getmtime(p)
        except Exception:
            continue
        if best is None or mt > best[0]:
            best = (mt, p)

if best:
    print(best[1])
PY_FIND_POLICY
  } || true)"
  if [[ -n "$POLICY_JSON_FALLBACK" && -f "$POLICY_JSON_FALLBACK" ]]; then
    POLICY_JSON="$POLICY_JSON_FALLBACK"
    echo "[WARN] Policy encontrada por fallback: $POLICY_JSON"
  fi
fi

if [[ ! -f "$POLICY_JSON" ]]; then
  echo "[ERRO] Policy nao encontrada apos execucao: $POLICY_JSON" >&2
  exit 3
fi

echo "[INFO] Validando policy efetiva em $POLICY_JSON ..."
"$PYTHON_BIN" - "$POLICY_JSON" <<'PY3'
import json
import sys
from pathlib import Path

policy_path = Path(sys.argv[1])
data = json.loads(policy_path.read_text(encoding="utf-8"))
wf = data.get("wf") if isinstance(data, dict) else None
if not isinstance(wf, dict):
    raise SystemExit("[ERRO] policy.wf ausente/ invalido")

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
    failed.append(f"backpre_slip_max invalido: {slip_max!r}")
else:
    if slip_val > 0.0:
        failed.append(f"backpre_slip_max esperado <=0, obtido={slip_val}")

if failed:
    msg = "\n".join(f"- {x}" for x in failed)
    raise SystemExit("[ERRO] Policy fora do esperado:\n" + msg)

print("[OK] Policy validada com sucesso (ROI_ONLY Back Pre + slippage<=0).")
PY3

echo "[SUCESSO] Execucao concluida e policy validada."
