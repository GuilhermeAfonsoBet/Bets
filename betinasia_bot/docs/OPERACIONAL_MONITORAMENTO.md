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

### Instalar systemd unit + timer
```bash
sudo cp -v betinasia_bot/ops/systemd/betinasia-ops-monitor.service /etc/systemd/system/
sudo cp -v betinasia_bot/ops/systemd/betinasia-ops-monitor.timer   /etc/systemd/system/
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
sudo cp -v betinasia_bot/ops/systemd/betinasia-ops-autopilot.service /etc/systemd/system/
sudo cp -v betinasia_bot/ops/systemd/betinasia-ops-autopilot.timer   /etc/systemd/system/
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

