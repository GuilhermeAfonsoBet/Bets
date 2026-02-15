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
sudo cp -v ops/systemd/betinasia-ops-monitor.service /etc/systemd/system/
sudo cp -v ops/systemd/betinasia-ops-monitor.timer   /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now betinasia-ops-monitor.timer
systemctl list-timers --all | rg betinasia-ops-monitor
```

### Rodar manualmente
```bash
python3 -m ops.health_monitor --since-minutes 30 --telegram
```

Variáveis úteis (no `.env`):
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
- `OPS_TELEMETRY_MAX_AGE_SEC` (default 600)

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

