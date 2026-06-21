# Downward Trend (DT) - operacao dedicada

## Objetivo

Esta estrategia substitui o fluxo operacional H3BUP por um sinal de
**Downward Trend**. O codigo reaproveita a infraestrutura validada de BetinAsia
(WebSocket, betslip API, bridge, executor, reports e systemd), mas a hipotese
operacional salva e consumida pelo bridge deve ser sempre `DT`.

## Regra inicial do sinal

O detector DT opera por mercado/lado:

```text
event_id + market_type + market_period + line + side
```

Um sinal e emitido quando:

1. existem pelo menos `DT_MIN_CONSEC_DOWNS` quedas consecutivas;
2. cada queda individual e menor/igual a `-DT_MIN_STEP_DROP_PCT`;
3. a queda acumulada da sequencia e menor/igual a `-DT_MIN_CUM_DROP_PCT`;
4. o intervalo entre passos nao excede `DT_MAX_STEP_GAP_SEC`;
5. depois de um sinal, a mesma chave respeita `DT_SIGNAL_COOLDOWN_SEC`.

Defaults iniciais:

```env
DT_MIN_CONSEC_DOWNS=3
DT_MIN_STEP_DROP_PCT=0.20
DT_MIN_CUM_DROP_PCT=0.80
DT_MAX_STEP_GAP_SEC=30
DT_SIGNAL_COOLDOWN_SEC=45
```

Exemplo de sinal:

```text
2.00 -> 1.99 -> 1.98 -> 1.97
```

Com os defaults acima, a terceira queda qualificada emite um evento `DT`.

## Separacao de H3B/H3BUP

Para evitar resquicio operacional de H3BUP:

- o entrypoint DT e `audit_downward_trend_api.py`;
- o service DT chama esse entrypoint, nao `audit_h3b_api.py` diretamente;
- `AUDIT_HYPOTHESIS_TYPE=DT` e fixado no service;
- `BRIDGE_HYPOTHESIS=DT` e fixado no bridge DT;
- a policy OOS DT usa arquivos separados:
  - `logs/wf_policy_current_dt.json`
  - `logs/policy_history_dt/`
  - `logs/wf_policy_history_dt.jsonl`

Ainda existem nomes internos historicos em alguns modulos compartilhados porque
eles carregam infraestrutura comum do auditor antigo. A aderencia runtime da
estrategia DT e garantida pelo detector `DTDownwardTrendDetector`, pelo
`hypothesis_type=DT` e pelos services dedicados.

## Services principais

Instalar na VPS DT:

```bash
sudo cp betinasia-executor.service /etc/systemd/system/
sudo cp betinasia-audit-ws-gate-dt.service /etc/systemd/system/
sudo cp betinasia-executor-bridge-dt.service /etc/systemd/system/
sudo cp betinasia-daily-dt-report.service /etc/systemd/system/
sudo cp betinasia-daily-dt-report.timer /etc/systemd/system/
sudo systemctl daemon-reload
```

Ordem recomendada em shadow:

```bash
sudo systemctl enable --now betinasia-executor
sudo systemctl enable --now betinasia-audit-ws-gate-dt
sudo systemctl enable --now betinasia-executor-bridge-dt
sudo systemctl enable --now betinasia-daily-dt-report.timer
```

Para LIVE, alterar no `.env` local da VPS:

```env
EXECUTOR_ALLOW_LIVE=1
BRIDGE_MODE=live
```

## Credenciais e proxy

Credenciais BetinAsia, proxy dedicado e Telegram devem ficar apenas no `.env` da
VPS DT. Use `betinasia_bot/.env.downward-trend.example` como template.

Nunca commite:

- `BETINASIA_PASSWORD`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `PROXY_USERNAME`
- `PROXY_PASSWORD`

## Telegram separado

Recomendado criar um bot/grupo especifico para DT:

1. criar bot via `@BotFather`;
2. adicionar o bot ao grupo Telegram DT;
3. enviar `/start` ou qualquer mensagem no grupo;
4. rodar na VPS:

```bash
PYTHONPATH=betinasia_bot python3 -m ops.telegram_setup
```

5. copiar o `chat_id` para o `.env` DT.

## Proxima etapa de latencia (<5s)

O fluxo atual ja reduz polling com `BRIDGE_POLL_SEC=0.25` no exemplo DT. A
otimizacao estrutural recomendada e separar o executor em duas fases:

```text
audit DT detecta sinal
  |-- executor /prepare abre betslip e captura PMM sem apostar
  |-- bridge valida policy/saldo/stake em paralelo
       |-- OK: executor /commit confirma ordem
       |-- NOK: executor /cancel fecha betslip
```

Isso remove a serializacao bridge -> abrir betslip -> place order e deve ser a
proxima mudanca para perseguir e2e consistentemente abaixo de 5s.
