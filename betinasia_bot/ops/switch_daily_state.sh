#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
MAY21_REF="${DAILY_STATE_MAY21_REF:-6535bd8}"
CURRENT_REF="${DAILY_STATE_CURRENT_REF:-b99281a}"

FILES=(
  "betinasia_bot/ops/daily_full_report.py"
  "betinasia_bot/analyze_contexto_operacao_b808_robust_report.py"
  "betinasia_bot/.env.example"
)

usage() {
  cat <<'USAGE'
Uso:
  ops/switch_daily_state.sh status
  ops/switch_daily_state.sh may21-test
  ops/switch_daily_state.sh current-hardened

Descrição:
  - may21-test: restaura arquivos do snapshot estável (ref padrão: 6535bd8)
  - current-hardened: restaura arquivos do estado atual hardening (ref padrão: b99281a)
  - status: mostra qual perfil está mais próximo para os arquivos alvo

Variáveis opcionais:
  DAILY_STATE_MAY21_REF=<git-ref>
  DAILY_STATE_CURRENT_REF=<git-ref>
USAGE
}

require_git_repo() {
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    echo "ERRO: execute dentro de um repositório git." >&2
    exit 1
  }
}

ensure_file_in_ref() {
  local ref="$1"
  local file="$2"
  git cat-file -e "${ref}:${file}" 2>/dev/null
}

sha_in_ref_or_na() {
  local ref="$1"
  local file="$2"
  if git cat-file -e "${ref}:${file}" 2>/dev/null; then
    git rev-parse "${ref}:${file}"
  else
    echo "N/A"
  fi
}

status_mode() {
  echo "Refs:"
  echo "  may21-test      => ${MAY21_REF}"
  echo "  current-hardened => ${CURRENT_REF}"
  echo

  local file cur may now
  for file in "${FILES[@]}"; do
    cur="$(git rev-parse "HEAD:${file}" 2>/dev/null || echo "N/A")"
    may="$(sha_in_ref_or_na "${MAY21_REF}" "${file}")"
    now="$(sha_in_ref_or_na "${CURRENT_REF}" "${file}")"
    echo "${file}"
    echo "  HEAD            ${cur}"
    echo "  may21-test      ${may}"
    echo "  current-hardened ${now}"
    if [[ "${cur}" == "${may}" && "${cur}" != "N/A" ]]; then
      echo "  -> estado: may21-test"
    elif [[ "${cur}" == "${now}" && "${cur}" != "N/A" ]]; then
      echo "  -> estado: current-hardened"
    else
      echo "  -> estado: misto/custom"
    fi
    echo
  done
}

switch_mode() {
  local target_ref="$1"
  local backup_root
  backup_root="betinasia_bot/logs/state_switch_backups/$(date -u +%Y%m%d_%H%M%S)"
  mkdir -p "${backup_root}"

  local file dst
  for file in "${FILES[@]}"; do
    if [[ -f "${file}" ]]; then
      dst="${backup_root}/${file}"
      mkdir -p "$(dirname "${dst}")"
      cp "${file}" "${dst}"
    fi
  done

  local checkout_files=()
  for file in "${FILES[@]}"; do
    if ensure_file_in_ref "${target_ref}" "${file}"; then
      checkout_files+=("${file}")
    fi
  done

  if [[ ${#checkout_files[@]} -eq 0 ]]; then
    echo "ERRO: nenhum arquivo-alvo existe em ${target_ref}." >&2
    exit 1
  fi

  git checkout "${target_ref}" -- "${checkout_files[@]}"

  echo "OK: arquivos restaurados de ${target_ref}"
  echo "Backup local salvo em: ${backup_root}"
  echo
  git status -sb
}

require_git_repo

case "${MODE}" in
  status)
    status_mode
    ;;
  may21-test)
    switch_mode "${MAY21_REF}"
    ;;
  current-hardened)
    switch_mode "${CURRENT_REF}"
    ;;
  ""|-h|--help|help)
    usage
    ;;
  *)
    echo "Modo inválido: ${MODE}" >&2
    usage
    exit 1
    ;;
esac

