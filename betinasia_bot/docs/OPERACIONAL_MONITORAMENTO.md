## Operacional — controle, resiliência e monitoramento (VPS)

### Objetivo (SLO)
- **Collector**: odds sendo gravadas continuamente (sem “hang silencioso”).
- **Audit API**: auditorias contínuas com fila baixa e telemetria fresca.
- **DB**: `best_odds_history` e `betslip_audit_results` com timestamps recentes.

---

## 1) Checagens rápidas (manual)

### Serviços
```bash
sudo systemctl status --no-pager -n 80 betinasia-collector betinasia-audit-api
```

### Saldo + P&L (Accounting)

Há um monitor paralelo que baixa os CSVs de:
- `https://black.betinasia.com/accounting/balance`
- `https://black.betinasia.com/accounting/open-stakes`

Ele grava:
- CSVs em `logs/accounting/`
- snapshots em `logs/accounting_snapshots.jsonl`

Instalar/rodar:

```bash
sudo cp -v "$(git ls-files | grep -m1 'betinasia-accounting-monitor.service')" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now betinasia-accounting-monitor
sudo systemctl status --no-pager -n 80 betinasia-accounting-monitor
tail -n 50 logs/accounting_monitor.log
```

### Importante (systemd): `.env` vs overrides (`service.d/`)
Se você ajustou variáveis no `.env` (ex.: `AUDIT_EXECUTOR_WORKERS=4`) e **não refletiu** no serviço, cheque se existe algum **drop-in** em `/etc/systemd/system/<service>.service.d/*.conf` sobrescrevendo `Environment=`. Esses overrides **têm precedência** sobre o `EnvironmentFile=...`.

```bash
sudo systemctl show betinasia-audit-api -p EnvironmentFiles --no-pager
sudo systemctl show betinasia-audit-api -p Environment --no-pager
sudo systemctl cat betinasia-audit-api --no-pager | sed -n '1,140p'
```

Para corrigir, edite/remova o drop-in e recarregue o systemd:

```bash
sudo ls -la /etc/systemd/system/betinasia-audit-api.service.d/ || true
sudo nano /etc/systemd/system/betinasia-audit-api.service.d/workers.conf
sudo systemctl daemon-reload
sudo systemctl restart betinasia-audit-api
```

### Frescor (telemetria)
```bash
tail -n 3 logs/collector_telemetry.jsonl
tail -n 3 logs/audit_api_telemetry.jsonl
```

### Frescor (DB)
```bash
psql "$DATABASE_URL" -c "
SELECT
  now() AT TIME ZONE 'utc' AS now_utc,
  (SELECT max(scraped_at) FROM best_odds_history) AS last_best_odds_utc,
  (SELECT max(audited_at) FROM betslip_audit_results) AS last_audit_utc;
"
```

---

## 2) Monitor automático (timer)

O monitor executa a cada ~2 minutos e envia alerta no Telegram quando houver WARN/FAIL.

> Nota: o `ops.health_monitor` usa **exit codes** (0=PASS, 1=WARN, 2=FAIL).  
> Nos units `betinasia-ops-monitor.service` e `betinasia-ops-autopilot.service` nós já incluímos `SuccessExitStatus=1 2` para o systemd **não** marcar o unit como "failed" em WARN/FAIL (o que é esperado e não significa que o timer quebrou).

### Instalar systemd unit + timer
```bash
# Alguns ambientes têm layout com pasta duplicada (ex.: betinasia_bot/betinasia_bot/...).
# Para não errar o path, localize os arquivos via git:
MON_SVC="$(git ls-files | grep -m1 'betinasia-ops-monitor.service')"
MON_TMR="$(git ls-files | grep -m1 'betinasia-ops-monitor.timer')"
echo "monitor: $MON_SVC | $MON_TMR"
sudo cp -v "$MON_SVC" /etc/systemd/system/
sudo cp -v "$MON_TMR" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now betinasia-ops-monitor.timer
systemctl list-timers --all | grep -i betinasia-ops-monitor || true
```

### Rodar manualmente
```bash
PYTHONPATH=betinasia_bot python3 -m ops.health_monitor --since-minutes 30 --telegram
```

Variáveis úteis (no `.env`):
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
- `OPS_TELEMETRY_MAX_AGE_SEC` (default 600)

### Telegram (como “cadastrar” corretamente)
Não é pelo telefone. Você precisa do **chat_id** do seu Telegram para o seu bot.

1) Crie um bot no Telegram via `@BotFather` e pegue o token.  
2) No seu celular (o número `+55...`), abra o bot e mande `/start`.  
3) No VPS, com `TELEGRAM_BOT_TOKEN` no `.env`, rode:

```bash
PYTHONPATH=betinasia_bot python3 -m ops.telegram_setup
```

Ele vai imprimir o `chat_id`. Coloque no `.env` como `TELEGRAM_CHAT_ID=...`.

---

## 2b) Auto-pilot seguro (timer com restart controlado)

O auto-pilot roda a cada ~2 minutos e:
- envia alerta Telegram em WARN/FAIL
- **só reinicia** serviços quando houver **FAIL por 2 execuções seguidas** (default)
- aplica **cooldown** e **rate limit** para evitar flapping

### Instalar
```bash
AUTO_SVC="$(git ls-files | grep -m1 'betinasia-ops-autopilot.service')"
AUTO_TMR="$(git ls-files | grep -m1 'betinasia-ops-autopilot.timer')"
echo "autopilot: $AUTO_SVC | $AUTO_TMR"
sudo cp -v "$AUTO_SVC" /etc/systemd/system/
sudo cp -v "$AUTO_TMR" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now betinasia-ops-autopilot.timer
systemctl list-timers --all | grep -i betinasia-ops-autopilot || true
```

### Configurar parâmetros (no `.env`)
- `OPS_FAILS_TO_RESTART=2`
- `OPS_RESTART_COOLDOWN_SEC=1800`  (30min)
- `OPS_MAX_RESTARTS_PER_HOUR=2`
- `OPS_AUTOPILOT_STATE_FILE=logs/ops_autopilot_state.json`

### Importante: sudo sem prompt (para o timer conseguir reiniciar)
O restart usa `sudo systemctl restart ...`. Configure uma regra NOPASSWD (exemplo):

```bash
sudo visudo
```

Adicione (ajuste o path do systemctl se necessário, às vezes é `/usr/bin/systemctl`):
```
betbot ALL=(root) NOPASSWD: /bin/systemctl restart betinasia-collector, /bin/systemctl restart betinasia-audit-api
betbot ALL=(root) NOPASSWD: /usr/bin/systemctl restart betinasia-collector, /usr/bin/systemctl restart betinasia-audit-api
```

---

## 2c) Status on-demand via Telegram (/status)

Além dos alertas automáticos, você pode **perguntar o status completo** pelo Telegram:

- mande `/status` para o bot
- ele responde com:
  - systemd (collector + audit)
  - frescor de telemetria/DB
  - resumo da telemetria do collector (inclui `save_errors`, timeouts e ciclos com 0 odds úteis)

### Instalar o serviço do bot de status
```bash
sudo cp -v "$(git ls-files | grep -m1 'betinasia-ops-telegram-bot.service')" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now betinasia-ops-telegram-bot.service
sudo systemctl status --no-pager -n 80 betinasia-ops-telegram-bot.service
```

---

## 3) Procedimento “restart limpo” (quando travar)

Collector:
```bash
sudo systemctl stop betinasia-collector
pkill -f "collector.continuous_collector" || true
pkill -f "chrome-headless-shell" || true
pkill -f "playwright/driver" || true
sudo systemctl start betinasia-collector
sleep 10
tail -n 3 logs/collector_telemetry.jsonl
```

Audit:
```bash
sudo systemctl restart betinasia-audit-api
tail -n 3 logs/audit_api_telemetry.jsonl
```

