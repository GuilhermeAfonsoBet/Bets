#!/usr/bin/env bash
# Alinha overrides systemd do executor com H3BUP_vNext (stake fixo 10).
#
# Causa (2026-07-20): zz-999-backpre-fast20.conf definia
#   EXECUTOR_BACKPRE_FAST_STAKE_HI=20 e EXECUTOR_LIVE_MAX_STAKE=20
# O bridge legado pedia stake_requested=1.5, mas o value_sizing do executor
# bumpava para stake_chosen/sent=20 no "pre fast path".
#
# Uso (na VPS, como root):
#   bash betinasia_bot/ops/align_h3bup_stake_overrides.sh
set -euo pipefail

DROP_IN_DIR="/etc/systemd/system/betinasia-executor.service.d"
BACKUP_DIR="/root/backups/executor-stake-overrides-$(date -u +%Y%m%dT%H%M%SZ)"
FAST20="${DROP_IN_DIR}/zz-999-backpre-fast20.conf"
FORCE_LIVE="${DROP_IN_DIR}/zz-999-force-live-stake.conf"
STAKE_GUARD="${DROP_IN_DIR}/99-stake-guard.conf"
H3BUP_CONF="${DROP_IN_DIR}/zz-999-h3bup-stake10.conf"

mkdir -p "$BACKUP_DIR" "$DROP_IN_DIR"

for f in "$FAST20" "$FORCE_LIVE" "$STAKE_GUARD"; do
  if [[ -f "$f" ]]; then
    cp -a "$f" "$BACKUP_DIR/"
  fi
done

# Desativa o drop-in que forçava HI/MAX=20 (mantém arquivo .bak no backup).
if [[ -f "$FAST20" ]]; then
  mv -f "$FAST20" "${FAST20}.disabled_$(date -u +%Y%m%dT%H%M%SZ)"
fi

# LIVE_STAKE legado 1.5 não pode sobrescrever o alinhamento H3BUP.
if [[ -f "$FORCE_LIVE" ]]; then
  mv -f "$FORCE_LIVE" "${FORCE_LIVE}.disabled_$(date -u +%Y%m%dT%H%M%SZ)"
fi

cat >"$H3BUP_CONF" <<'EOF'
[Service]
# H3BUP_vNext: stake operacional fixo 10.
# Desliga o fast-path que bumpava stake_requested=1.5 -> sent=20.
Environment="EXECUTOR_BACKPRE_FAST_STAKE_ENABLE=0"
Environment="EXECUTOR_BACK_STAKE_SIZING_ENABLE=0"
Environment="EXECUTOR_BACKPRE_FAST_STAKE_HI=10"
Environment="EXECUTOR_BACKPRE_FAST_STAKE=10"
Environment="EXECUTOR_BACKPRE_FAST_STAKE_LO=10"
Environment="EXECUTOR_BACK_STAKE_DEFAULT=10"
Environment="EXECUTOR_LIVE_STAKE=10"
Environment="EXECUTOR_LIVE_MAX_STAKE=10"
EOF

# Harmoniza o stake-guard antigo (HI=6) para o mesmo teto H3BUP.
if [[ -f "$STAKE_GUARD" ]]; then
  cat >"$STAKE_GUARD" <<'EOF'
[Service]
Environment="EXECUTOR_BACKPRE_FAST_STAKE_ENABLE=0"
Environment="EXECUTOR_BACKPRE_FAST_MAX_PRE_SUBMIT_MS=6000"
Environment="EXECUTOR_BACKPRE_FAST_STAKE_HI=10"
Environment="EXECUTOR_BACKPRE_FAST_STAKE=10"
Environment="EXECUTOR_BACKPRE_FAST_STAKE_LO=10"
Environment="EXECUTOR_BACK_STAKE_DEFAULT=10"
Environment="EXECUTOR_BACK_STAKE_SIZING_ENABLE=0"
Environment="EXECUTOR_LIVE_STAKE=10"
Environment="EXECUTOR_LIVE_MAX_STAKE=10"
EOF
fi

# Critico: o processo herda EnvironmentFile=.env; so drop-in systemd NAO basta.
ENV_FILE="/home/betbot/Bets/betinasia_bot/.env"
if [[ -f "$ENV_FILE" ]]; then
  cp -a "$ENV_FILE" "$BACKUP_DIR/dotenv.env"
  python3 - <<'PY'
from pathlib import Path
p = Path("/home/betbot/Bets/betinasia_bot/.env")
replacements = {
    "EXECUTOR_LIVE_STAKE": "10",
    "EXECUTOR_LIVE_MAX_STAKE": "10",
    "EXECUTOR_BACKPRE_FAST_STAKE_ENABLE": "0",
    "EXECUTOR_BACKPRE_FAST_STAKE_ENFORCE": "0",
    "EXECUTOR_BACKPRE_FAST_STAKE": "10",
    "EXECUTOR_BACKPRE_FAST_STAKE_HI": "10",
    "EXECUTOR_BACKPRE_FAST_STAKE_LO": "10",
    "EXECUTOR_BACK_STAKE_DEFAULT": "10",
    "EXECUTOR_BACK_STAKE_SIZING_ENABLE": "0",
    "BRIDGE_STAKE": "10",
}
out = []
seen = set()
for line in p.read_text().splitlines(True):
    if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
        out.append(line)
        continue
    k, _ = line.split("=", 1)
    k = k.strip()
    if k in replacements:
        out.append(f"{k}={replacements[k]}\n")
        seen.add(k)
    else:
        out.append(line)
for k, v in replacements.items():
    if k not in seen:
        out.append(f"{k}={v}\n")
p.write_text("".join(out))
print("[OK] .env stake keys alinhados a H3BUP stake10")
PY
fi

systemctl daemon-reload
systemctl restart betinasia-executor.service

echo "[OK] backups em: $BACKUP_DIR"
echo "[OK] ativo: $H3BUP_CONF"
echo "[OK] /proc environ efetivo (stake):"
pid="$(systemctl show -p MainPID --value betinasia-executor)"
tr '\0' '\n' <"/proc/$pid/environ" \
  | grep -E 'EXECUTOR_(BACKPRE_FAST_STAKE|LIVE_STAKE|LIVE_MAX|BACK_STAKE)|BRIDGE_STAKE' \
  | sort || true
systemctl is-active betinasia-executor.service
