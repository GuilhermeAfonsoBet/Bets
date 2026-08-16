# Acesso VPS necessário — Friendly analysis

A análise histórica precisa de leitura em:

- `logs/executor_live.jsonl`
- `logs/accounting/*__balance.csv` / `*__open_stakes.csv`
- `logs/h3bup_clv_snapshots.jsonl`
- (opcional) Postgres `betslip_audit_results` / `matches` para liga

O SSH desta sessão está **Permission denied** para `root@178.128.55.30`
e `betbot@178.128.55.30` (chave `cursor-agent-20260728` revogada/ausente).

## Chave pública a instalar

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOZUf7Qxh8vyYdIjU0s5X12svvStmtYGapaf1nSV9flG cursor-agent-20260731-friendly
```

No VPS (como root ou betbot):

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOZUf7Qxh8vyYdIjU0s5X12svvStmtYGapaf1nSV9flG cursor-agent-20260731-friendly' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

Depois do acesso, o smoke read-only é:

```bash
cd /home/betbot/Bets/betinasia_bot
bash ops/h3bup_friendly_analysis/vps_smoke.sh
```

Outputs: `logs/h3bup_friendly_analysis/<YYYYMMDD>/<run_id>/`
