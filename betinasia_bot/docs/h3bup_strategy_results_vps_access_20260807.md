# Acesso VPS — refresh análise de resultados H3BUP (2026-08-07)

SSH desta sessão: `Permission denied` para `root@178.128.55.30`.

Sem acesso não é possível incluir:
- liquidação dos 18 opens do freeze 2026-08-01
- cohort **stake=2** (desde ~2026-08-01 12:07Z)
- LIVE_OK 2026-08-01 → 2026-08-07

## Chave pública a instalar

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFH5MuVrvyEQOKlYnbD/GDgVKENHsK+84mN8x8QNOfPt cursor-agent-20260807-h3bup-results
```

Na VPS (root):

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFH5MuVrvyEQOKlYnbD/GDgVKENHsK+84mN8x8QNOfPt cursor-agent-20260807-h3bup-results' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

Depois: rerun Friendly analysis + `ops.h3bup_strategy_results.analyze_freeze`.
