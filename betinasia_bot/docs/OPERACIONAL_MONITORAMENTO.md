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

### Auditoria Back + Lay em paralelo (2 audits)
Para executar **Back e Lay** sob policy OOS, você precisa gerar oportunidades executáveis para ambos:
- `betinasia-audit-api` (já existente): `--mode ws_gate_lay` (Lay via gate)
- `betinasia-audit-api-back` (novo): `--mode api --api-sides back` (Back via API, back-only)

Instalar o unit do Back:

```bash
sudo cp -v "$(git ls-files | grep -m1 'betinasia-audit-api-back.service')" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now betinasia-audit-api-back
sudo systemctl status --no-pager -n 80 betinasia-audit-api-back
```

Notas:
- O Back-only grava como `audit_version=v5.2-api-back` (não mistura com `v4.0-api` antigo).
- Inclua `v5.2-api-back` em `DAILY_OOS_VERSIONS` para aparecer nos relatórios e no OOS.

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
sudo cp -v "$(git ls-files | grep -m1 'betinasia-accounting-daily.service')" /etc/systemd/system/
sudo cp -v "$(git ls-files | grep -m1 'betinasia-accounting-daily.timer')" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now betinasia-accounting-monitor
sudo systemctl enable --now betinasia-accounting-daily.timer
sudo systemctl status --no-pager -n 80 betinasia-accounting-monitor
tail -n 50 logs/accounting_monitor.log
```

Rodar manual “snapshot do momento”:

```bash
python3 -m ops.accounting_monitor --once
```

Resumo de P&L (best-effort) a partir de um CSV exportado:

```bash
python3 -m ops.accounting_report --csv logs/accounting/<arquivo.csv>
```

Comando único (snapshot + relatório):

```bash
python3 -m ops.accounting_daily_report
```

KPIs de execução (lag/slippage/status) a partir do JSONL do executor:

```bash
python3 -m ops.execution_kpis --jsonl logs/executor_live.jsonl --last 5000
```

Como ler as tabelas:
- **Status (all)**: contagem por `result.status` em `logs/executor_live.jsonl`.
  - `LIVE_OK`: aposta real enviada/confirmada (`EXECUTOR_ALLOW_LIVE=1` + request `is_live=true`)
  - `DRY_OK`: abriu betslip e capturou odd final, **sem** apostar (dry-run)
  - `NO_SESSION`: sem sessão/logado (ou sessão expirada)
  - `API_FAILED`: falha de API/timeout/erro ao abrir betslip ou confirmar
  - `RATE_LIMIT`: rate limit/backoff
- **Erros típicos (quando `API_FAILED`)**:
  - `HTTP_403 ... code=too_many_open_betslips`: a conta/sessão acumulou **betslips abertos demais**. Mitigação: reduzir cache + fechar betslips no executor/audit.
    - Ajustes recomendados no `.env`: `EXECUTOR_BETSLIP_CACHE_MAX_KEYS=0` (shadow) e reiniciar executor.
  - `No PMMs received`: betslip abriu, mas não chegaram mensagens PMM no WS dentro do timeout. Mitigação: aumentar timeouts (`EXECUTOR_PMM_TIMEOUT_SEC`) e reduzir saturação (menos abertura concorrente/menos filas velhas).
- **Latência (somente LIVE_OK/DRY_OK)**:
  - `queue_delay_ms`: tempo na fila do executor antes do worker iniciar
  - `call_to_done_ms`: tempo total observado desde `created_at` até o resultado (wall clock)
  - `post_ms`: tempo do request principal (ex.: abrir betslip / place order)
- **Slippage** só aparece quando o JSONL tem `odd_at_decision` **e** `odd_final` na mesma linha.

Relatório diário completo (OOS + execução + accounting + PDF + Telegram):

```bash
python3 -m ops.daily_full_report
```

Saídas importantes do daily:
- `logs/daily_reports/YYYYMMDD/report_daily.pdf`: relatório completo (inclui seção 99)
- `logs/wf_policy_current.json`: policy “corrente” (export do walk-forward, usado pelo bridge)
- `logs/policy_history/wf_policy_YYYYMMDD.json`: histórico diário do export
- `logs/daily_reports/YYYYMMDD/oos_adherence.json`: aderência (portfolio por dia × execução × ROI por placar)

### Policy OOS ativa (mecânica de atualização)
O `ops.daily_full_report` roda o walk-forward e faz:
- export diário da policy (JSON) em `logs/policy_history/`
- atualização **atômica** do “ponteiro” `logs/wf_policy_current.json`

O bridge (`ops/executor_bridge_audit.py`) aplica essa policy quando você setar:
- `BRIDGE_POLICY_JSON=logs/wf_policy_current.json`
- `BRIDGE_POLICY_RELOAD_SEC=5` (recarrega automaticamente quando o arquivo muda)
- `BRIDGE_POLICY_USE_BASE=1` (opcional: ignora sufixo de liga, usa `active_keys_base`)

### Anti-trava operacional do bridge (recomendado)
Para reduzir risco de “accepted=0” por lock/dedupe antigo:

- `BRIDGE_SINGLETON_LOCK_PATH`  
  Garante **uma única instância** do `ops.executor_bridge_audit` por modo/lado no host.
  Evita concorrência acidental (service + processo órfão/manual).
- `BRIDGE_SINGLETON_KILL_COMPETITORS=1`  
  No startup, encerra processos concorrentes do mesmo modo/lado encontrados em `/proc`
  (self-healing para órfãos após deploy/restart).
- `BRIDGE_SINGLETON_ASSUME_UNKNOWN_MODE_COMPETITOR=1` e `...SIDE...=1`  
  Quando um processo órfão não tem `BRIDGE_MODE/BRIDGE_EXEC_SIDE` no ambiente
  (run manual), ele também é tratado como concorrente e encerrado.
- `BRIDGE_SEEN_KEY_GC_SEC` + `BRIDGE_SEEN_KEY_TTL_SEC`  
  Limpam reservas órfãs (sem `execution_id`) antigas.
- `BRIDGE_SEEN_KEY_HARD_TTL_SEC`  
  Limpa chaves de dedupe muito antigas, evitando `dup_key` indefinido por histórico velho.

Defaults operacionais sugeridos:

```env
BRIDGE_SEEN_KEY_GC_SEC=300
BRIDGE_SEEN_KEY_TTL_SEC=3600
BRIDGE_SEEN_KEY_HARD_TTL_SEC=86400
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

### Importante (systemd): `ExecStopPost` NÃO é shell
Alguns deployments usam `ExecStopPost` para matar `chrome-headless-shell`/Playwright ao reiniciar o executor.  
**Atenção:** `ExecStopPost=` não roda em shell por padrão — então escrever `... || true` vira **argumento** do comando e pode quebrar (ex.: `pkill: only one pattern can be provided`), deixando processos zumbis e piorando travamentos.

O jeito correto é usar o prefixo `-` (ignora erro) ou envolver em `/bin/bash -lc`.

Exemplo recomendado:

```ini
ExecStopPost=-/usr/bin/pkill -9 -u betbot -f chrome-headless-shell
ExecStopPost=-/usr/bin/pkill -9 -u betbot -f "playwright/driver"
ExecStopPost=-/usr/bin/rm -f /tmp/betinasia-exec.sock
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
- `OPS_PIPELINE_MIN_BEST_ODDS_FOR_ZERO_AUDIT` (default 1000)
- `OPS_PIPELINE_ZERO_AUDIT_FAIL` (default 1)
- `OPS_AUDIT_GATE_BACK_VERSION` (default `v5.3-ws-gate-back`)
- `OPS_AUDIT_ZERO_VALID_MIN_AUDITS` (default 40)
- `OPS_AUDIT_ZERO_VALID_LEVEL` (`off|warn|fail`, default `warn`)
- `OPS_PREMATCH_MISALIGN_MIN_AUDITS` (default 40)
- `OPS_PREMATCH_MISALIGN_LEVEL` (`off|warn|fail`, default `warn`)
- `BRIDGE_PREMATCH_FALLBACK_ENABLE` (default `0`)
- `BRIDGE_PREMATCH_FALLBACK_WINDOW_MIN` (default `60`)
- `BRIDGE_PREMATCH_FALLBACK_MIN_TOTAL_AUDITS` (default `40`)
- `BRIDGE_PREMATCH_FALLBACK_MIN_LIVE_VALID` (default `1`)

Essas duas últimas ajudam a detectar “travamento funcional” do audit/bridge:
se o collector segue produzindo (`best_odds_n` alto) mas `audits_n=0` na janela do monitor,
o health_monitor sobe WARN/FAIL e envia Telegram.

As três variáveis `OPS_AUDIT_*` detectam um cenário diferente:
auditoria rodando, porém sem elegibilidade (`valid=0`) por janela longa
(ex.: só `GATE_NOT_ELIGIBLE`). Isso evita confundir “sem setup” com “travado”.

As variáveis `OPS_PREMATCH_*` detectam desalinhamento de regime:
`BRIDGE_PREMATCH_ONLY=1` com `valid_live>0` e `valid_pre=0` na janela.
Nesse caso a execução fica zerada por desenho (não é crash), e o monitor alerta.

### Opções operacionais para PREMATCH_ONLY (A e B)

Quando o bridge está em `BRIDGE_PREMATCH_ONLY=1`, há dois modos de operar:

- **Opção A (estrita PRE, sem fallback)**  
  - `BRIDGE_PREMATCH_ONLY=1`
  - `BRIDGE_PREMATCH_FALLBACK_ENABLE=0`  
  O monitor passa a imprimir o regime ativo (`bridge-regime`) e alerta desalinhamento quando houver `valid_live>0` e `valid_pre=0`.

- **Opção B (fallback LIVE controlado)**  
  - `BRIDGE_PREMATCH_ONLY=1`
  - `BRIDGE_PREMATCH_FALLBACK_ENABLE=1`  
  O bridge ativa LIVE automaticamente quando, na janela configurada, `valid_pre=0` e `valid_live` atingir o mínimo.  
  Quando `valid_pre` volta, o fallback desliga sozinho e o bridge retorna ao comportamento PRE-only.

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

