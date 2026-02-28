# Executor v1 (Dry Run) — BetinAsia

Objetivo: simular a **execução real** (abrir ticket/betslip e capturar odd final) com o menor overhead possível, sem confirmar aposta.

## O que ele mede
- Latência (fila → POST betslip → PMMs → total)
- Odd no momento da decisão (fornecida pelo Decision Engine) vs odd final do betslip (slippage)
- Rate limit / backoff / cap de aberturas

## Como rodar

### Servidor HTTP local

```bash
cd /home/betbot/Bets/betinasia_bot
source venv/bin/activate

python3 run_executor_dryrun.py --host 127.0.0.1 --port 8089 --workers 1
```

### Alternativa (mais rápida): Unix socket

```bash
python3 run_executor_dryrun.py --unix-socket /tmp/betinasia-exec.sock --workers 1
```

### Modo LIVE (aposta real) — v0

**Atenção**: isso confirma aposta via `POST /v1/orders/`. Use stake pequeno e caps conservadores.

- Para habilitar no executor: `EXECUTOR_ALLOW_LIVE=1`
- No request: `is_live=true`
- Suporta **Back e Lay** (Lay usa `betslip_type=lay` + `bet_type=against,...`).

Variáveis úteis:

```bash
export EXECUTOR_ALLOW_LIVE=1
export EXECUTOR_LIVE_CCY=USD
export EXECUTOR_LIVE_STAKE=3.0
export EXECUTOR_LIVE_MAX_STAKE=3.0
```

## Modo ultra-rápido (recomendado para aproximar do real)

O fluxo "abrir betslip e esperar todos PMMs" pode levar 2–3s. Para reduzir:
- o executor ativa **FAST_PMM** por padrão (usa timeouts menores e para assim que existe preço utilizável);
- usa **cache de betslip_id** por (event_id, bet_type, betslip_type) e passa a usar `/refresh/` nas execuções seguintes.

Variáveis:

```bash
export EXECUTOR_FAST_PMM=1
export EXECUTOR_PMM_TIMEOUT_SEC=0.8
export EXECUTOR_PMM_MIN_WAIT_SEC=0.0
export EXECUTOR_PMM_IDLE_TIMEOUT_SEC=0.12
```

Isso troca um pouco de "best odd" por velocidade (o que é desejável quando tempo destrói edge).

## API

### `POST /execute`
Retorna `202` com `execution_id`. O resultado é persistido em memória + JSONL e pode ser consultado via `/result/{execution_id}`.

Exemplo:

```bash
curl -sS http://127.0.0.1:8089/execute -H 'Content-Type: application/json' -d '{
  "created_at": "2026-02-25T12:34:56Z",
  "audit_id": 123,
  "match_id": 456,
  "event_id": "2026-02-08,176,178",
  "market_type": "AH",
  "side": "home",
  "line": "-1",
  "exec_side": "Back",
  "is_live": false,
  "odd_at_decision": 2.05,
  "max_late_ms": 8000,
  "policy": {
    "policy_version": "risk_sqrt_eq4_cap33_v1",
    "bankroll_ref": 10000,
    "bud_back_frac": 0.04,
    "bud_lay_frac": 0.04,
    "cap_signal_frac": 0.33,
    "risk_mode": "signals_sqrt",
    "stake_requested": 80.0
  }
}'
```

### `GET /result/{execution_id}`

```bash
curl -sS http://127.0.0.1:8089/result/<execution_id>
```

### `GET /health`

```bash
curl -sS http://127.0.0.1:8089/health
```

### `GET /account` (saldo + P&L best-effort)

Retorna um snapshot operacional:
- tenta descobrir um endpoint de saldo/conta (varia)
- puxa as últimas ordens de `GET /v1/orders/` e resume P&L quando houver `profit_loss`

```bash
curl -sS http://127.0.0.1:8089/account?page_size=50
```

Via unix socket:

```bash
curl -sS --unix-socket /tmp/betinasia-exec.sock "http://localhost/account?page_size=50"
```

## Client CLI (exemplo)

Dentro de `betinasia_bot/`:

```bash
python -m executor.client \
  --event-id 2026-02-28,23624,173 \
  --side away \
  --line -0.25 \
  --exec-side Back \
  --live \
  --stake 3
```

## Bridge / Decision Engine v0 (DB audit -> Executor)

O bridge (`ops/executor_bridge_audit.py`) faz polling da tabela `betslip_audit_results` e dispara `POST /execute` no executor.

### Rodar (shadow ou live)

```bash
cd /home/betbot/Bets/betinasia_bot
source venv/bin/activate

# shadow (não aposta de verdade)
BRIDGE_MODE=shadow BRIDGE_EXEC_SIDE=Back BRIDGE_STAKE=3 python3 ops/executor_bridge_audit.py

# live (aposta real) — requer EXECUTOR_ALLOW_LIVE=1 no serviço do executor
BRIDGE_MODE=live BRIDGE_EXEC_SIDE=Lay BRIDGE_STAKE=3 python3 ops/executor_bridge_audit.py
```

### Produção (todas combinações): bridge Back + bridge Lay em paralelo

Para operar Back e Lay ao mesmo tempo, rode **dois serviços** (um para cada lado):
- `betinasia-executor-bridge-back.service`
- `betinasia-executor-bridge-lay.service`

Eles leem o mesmo `.env` e sobrescrevem `BRIDGE_EXEC_SIDE` no unit.

### Gate por OOS (policy export do walk-forward)

O relatório (`analyze_contexto_operacao_b808_robust_report.py`) pode exportar um JSON com `active_keys` do walk-forward.
O bridge pode carregar esse JSON e **só executar quando a combinação estiver ativa**.

```bash
# 1) gerar/exportar policy (exemplo)
python3 analyze_contexto_operacao_b808_robust_report.py --walkforward \
  --wf-export-policy-json logs/wf_policy.json \
  --out logs/relatorio.md

# 2) ligar o bridge com policy
BRIDGE_POLICY_JSON=logs/wf_policy.json BRIDGE_POLICY_RELOAD_SEC=5 python3 ops/executor_bridge_audit.py
```

## Logs e DB
- JSONL padrão: `logs/executor_dryrun.jsonl` (config: `EXECUTOR_JSONL`)
- Opcional: gravar no Postgres com `--save-to-db` (ou `EXECUTOR_SAVE_TO_DB=1`).

## Descoberta do endpoint de "place/confirm" (aposta real)

Para implementarmos o "place bet" via API interna, rode a captura e faça **1 aposta manual de ~$3**:

```bash
cd /home/betbot/Bets/betinasia_bot
source venv/bin/activate

python3 betinasia_bot/executor/capture_place_confirm.py --out logs/place_confirm_capture.jsonl
```

O script vai abrir o browser, fazer login e gravar requests/responses/WS relevantes (com redaction).

