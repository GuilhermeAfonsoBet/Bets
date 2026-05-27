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
export DAILY_WF_BACKPRE_SLIP_FIELD="${DAILY_WF_BACKPRE_SLIP_FIELD:-slippage_pre_pct}"

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

# Aviso de configuracao contraditoria: diff_pct<=0 pode conflitar com gate de edge Back.
if [[ "$DAILY_WF_BACKPRE_SLIP_FIELD" == "diff_pct" ]]; then
  if python3 - <<'PY_WARN_DIFF'
import os
try:
    v = float(os.getenv("DAILY_WF_BACKPRE_SLIP_MAX", "0"))
except Exception:
    v = 0.0
raise SystemExit(0 if v <= 0 else 1)
PY_WARN_DIFF
  then
    echo "[WARN] DAILY_WF_BACKPRE_SLIP_FIELD=diff_pct com DAILY_WF_BACKPRE_SLIP_MAX<=0 pode zerar oportunidades." >&2
    echo "[WARN] Para regra de slippage<0, prefira DAILY_WF_BACKPRE_SLIP_FIELD=slippage_pre_pct." >&2
  fi
fi

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
RUN_START_EPOCH="$(date +%s)"
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

# Prioriza policy candidata gerada neste run (quando disponível),
# para evitar validar arquivo stale em wf_policy_current.
if [[ -f "$OUT_JSON" ]]; then
  POLICY_FROM_RUN="$({
    "$PYTHON_BIN" - "$OUT_JSON" "$RUN_START_EPOCH" "$RUN_CWD" "$WORK_ROOT" "$REQUESTED_BOT_DIR" "$ROOT_DIR" <<'PY_PICK_POLICY'
import json
import os
import sys
from pathlib import Path

out_json = Path(sys.argv[1])
run_start_epoch = int(float(sys.argv[2]))
base_dirs = [Path(x).resolve() for x in sys.argv[3:] if x]

def _fresh(path: Path, start_epoch: int, tolerance_sec: int = 300) -> bool:
    try:
        return path.stat().st_mtime >= (start_epoch - tolerance_sec)
    except Exception:
        return False

raw = out_json.read_text(encoding='utf-8', errors='ignore')
data = None
try:
    data = json.loads(raw)
except Exception:
    i = raw.find('{')
    j = raw.rfind('}')
    if i >= 0 and j > i:
        try:
            data = json.loads(raw[i:j+1])
        except Exception:
            data = None

if not isinstance(data, dict):
    raise SystemExit(0)

def _resolve_existing(path_str: str):
    if not path_str or not str(path_str).strip():
        return None
    p = Path(str(path_str).strip())
    if p.is_absolute() and p.is_file():
        return p.resolve()
    for b in base_dirs:
        q = (b / p).resolve()
        if q.is_file():
            return q
    return None

cands = []
pp = data.get('policy_publish') if isinstance(data, dict) else None
if isinstance(pp, dict):
    cands.extend([
        pp.get('candidate_path'),
        pp.get('effective_path'),
        pp.get('policy_current'),
        pp.get('policy_path'),
    ])

oos = data.get('oos_run') if isinstance(data, dict) else None
if isinstance(oos, dict):
    cands.extend([
        oos.get('policy_path'),
        oos.get('policy_current'),
    ])

cands.extend([
    data.get('policy_current') if isinstance(data, dict) else None,
    data.get('policy_path') if isinstance(data, dict) else None,
])

seen = set()
resolved = []
for c in cands:
    if not isinstance(c, str):
        continue
    c = c.strip()
    if not c or c in seen:
        continue
    seen.add(c)
    r = _resolve_existing(c)
    if r:
        resolved.append(r)

# escolhe primeira policy fresca; se não houver, não devolve nada.
for r in resolved:
    if _fresh(r, run_start_epoch):
        print(str(r))
        break
PY_PICK_POLICY
  } || true)"
  if [[ -n "$POLICY_FROM_RUN" && -f "$POLICY_FROM_RUN" ]]; then
    POLICY_JSON="$POLICY_FROM_RUN"
    echo "[INFO] Policy selecionada a partir do output do run (fresca): $POLICY_JSON"
  else
    echo "[WARN] Nao foi possivel resolver policy fresca a partir do output do run; usando fallback fresco." >&2
  fi
fi

if [[ ! -f "$POLICY_JSON" ]]; then
  POLICY_JSON_FALLBACK="$({
    "$PYTHON_BIN" - "$RUN_START_EPOCH" "$ROOT_A" "$ROOT_B" "$ROOT_C" <<'PY_FIND_POLICY'
import os
import sys
from pathlib import Path

run_start_epoch = int(float(sys.argv[1]))
roots = []
seen = set()
for r in sys.argv[2:]:
    rr = os.path.abspath(r)
    if rr in seen or not os.path.isdir(rr):
        continue
    seen.add(rr)
    roots.append(rr)

def fresh(p: Path, start_epoch: int, tolerance_sec: int = 300) -> bool:
    try:
        return p.stat().st_mtime >= (start_epoch - tolerance_sec)
    except Exception:
        return False

cands = []
for root in roots:
    for cur, _, files in os.walk(root):
        if 'wf_policy_current.json' in files:
            cands.append(Path(cur) / 'wf_policy_current.json')
        for f in files:
            if f.startswith('wf_policy_') and f.endswith('.json'):
                cands.append(Path(cur) / f)

# prioriza frescos e mais novos
cands = [p.resolve() for p in cands if p.is_file()]
cands = sorted(set(cands), key=lambda p: p.stat().st_mtime, reverse=True)
for p in cands:
    if fresh(p, run_start_epoch):
        print(str(p))
        break
PY_FIND_POLICY
  } || true)"
  if [[ -n "$POLICY_JSON_FALLBACK" && -f "$POLICY_JSON_FALLBACK" ]]; then
    POLICY_JSON="$POLICY_JSON_FALLBACK"
    echo "[WARN] Policy fresca encontrada por fallback: $POLICY_JSON"
  fi
fi

if [[ ! -f "$POLICY_JSON" ]]; then
  echo "[ERRO] Policy fresca nao encontrada apos execucao (run_start=$RUN_START_EPOCH)." >&2
  exit 3
fi

EXPECT_PRE_ACT="${DAILY_WF_PRE_ACTIVATION_MODE:-roi_only}"
EXPECT_SIDES="${DAILY_WF_SIDES:-back}"
EXPECT_REGIMES="${DAILY_WF_REGIMES:-pre}"
EXPECT_TRAIN_MODE="${DAILY_WF_TRAIN_MODE:-expanding}"
EXPECT_SLIP_FIELD="${DAILY_WF_BACKPRE_SLIP_FIELD:-slippage_pre_pct}"
EXPECT_SLIP_MAX="${DAILY_WF_BACKPRE_SLIP_MAX:-0}"

echo "[INFO] Validando policy efetiva em $POLICY_JSON ..."
"$PYTHON_BIN" - "$POLICY_JSON" "$RUN_START_EPOCH" "$EXPECT_PRE_ACT" "$EXPECT_SIDES" "$EXPECT_REGIMES" "$EXPECT_TRAIN_MODE" "$EXPECT_SLIP_FIELD" "$EXPECT_SLIP_MAX" <<'PY3'
import json
import sys
from pathlib import Path

primary = Path(sys.argv[1]).resolve()
run_start_epoch = int(float(sys.argv[2]))
checks_expected = {
    'pre_activation_mode': str(sys.argv[3]).strip().lower(),
    'sides': str(sys.argv[4]).strip().lower(),
    'regimes': str(sys.argv[5]).strip().lower(),
    'train_mode': str(sys.argv[6]).strip().lower(),
    'backpre_slip_field': str(sys.argv[7]).strip().lower(),
}
try:
    expect_slip_max = float(sys.argv[8])
except Exception:
    expect_slip_max = 0.0

def fresh(path: Path, start_epoch: int, tolerance_sec: int = 300) -> bool:
    try:
        return path.stat().st_mtime >= (start_epoch - tolerance_sec)
    except Exception:
        return False

def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None

def _validate(path: Path):
    data = _load_json(path)
    if not isinstance(data, dict):
        return [f'arquivo nao e JSON objeto: {path}']
    wf = data.get('wf')
    if not isinstance(wf, dict):
        return ['policy.wf ausente/ invalido']

    failed = []
    for k, want in checks_expected.items():
        got = str(wf.get(k, '')).strip().lower()
        if got != want:
            failed.append(f'{k}: esperado={want} obtido={got}')

    slip_max = wf.get('backpre_slip_max')
    try:
        slip_val = float(slip_max)
    except Exception:
        failed.append(f'backpre_slip_max invalido: {slip_max!r}')
    else:
        if slip_val > expect_slip_max:
            failed.append(f'backpre_slip_max esperado <= {expect_slip_max}, obtido={slip_val}')

    return failed

candidates = [primary]
base = primary.parent
extra = []
try:
    extra.extend(base.glob('wf_policy_*.json'))
except Exception:
    pass
try:
    extra.extend((base / 'policy_history').glob('wf_policy_*.json'))
except Exception:
    pass
extra = [p.resolve() for p in extra if p.is_file() and p.resolve() != primary]
extra.sort(key=lambda p: p.stat().st_mtime, reverse=True)
candidates.extend(extra)

seen = set()
ordered = []
for c in candidates:
    if c in seen:
        continue
    seen.add(c)
    ordered.append(c)

fresh_only = [c for c in ordered if fresh(c, run_start_epoch)]
if not fresh_only:
    raise SystemExit(
        '[ERRO] Nenhuma policy fresca para este run; abortando para evitar falso positivo. '
        f'(run_start={run_start_epoch}, primary={primary})'
    )

best_fail = None
best_path = None
for c in fresh_only:
    fail = _validate(c)
    if not fail:
        print(f'[OK] Policy validada com sucesso (ROI_ONLY Back Pre + slippage<=0): {c}')
        raise SystemExit(0)
    if best_fail is None:
        best_fail = fail
        best_path = c

msg = '\n'.join(f'- {x}' for x in (best_fail or ['falha desconhecida']))
raise SystemExit(
    '[ERRO] Policy fresca fora do esperado (melhor candidata: '
    + str(best_path)
    + '):\n'
    + msg
)
PY3

echo "[SUCESSO] Execucao concluida e policy validada."
