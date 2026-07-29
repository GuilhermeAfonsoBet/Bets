# Design futuro — h3bup_e2e_trace (20260729)

**Não implementar nesta fase.**

## Objetivo
Tornar mensurável WS→LIVE_OK com coverage alta e join estável.

## Tabela proposta (append-only)
`h3bup_e2e_trace` com campos mínimos pedidos:
trace_id, audit_id, execution_id, order_id, event_id, market, side, line,
ws_received_ts, audit_started_ts, audit_decision_ts, audit_persisted_ts,
bridge_fetched_ts, policy_eval_ts, request_created_ts, executor_received_ts,
dryrun_started_ts, dryrun_finished_ts, final_gate_ts, place_started_ts,
place_finished_ts, live_ok_ts, persisted_ts, status, failure_reason.

## Onde inserir (sem código agora)
| Campo | Onde | Impacto | Migration? |
|---|---|---|---|
| ws_received_ts | audit WS on_message / state update | baixo | opcional |
| audit_started/decision/persisted | `_execute_ws_gate_back` / `_save_result` | baixo | sim se tabela |
| bridge_fetched/policy_eval | `_fetch_candidates` loop / `_h3bup_vnext_eval` | baixo | sim |
| request_created | já existe; copiar p/ trace | nenhum | — |
| dryrun_*/gate/place_* | `worker._execute_unlocked` | baixo–médio | sim |
| ids | propagar audit_id/execution_id/order_id | baixo | — |

## Compatibilidade
- Preferir JSONL append `logs/h3bup_e2e_trace.jsonl` **antes** de migration DB para reduzir risco.
- Manter feature flag off por default.

## O que já dá para medir hoje
- detected→audited (~4ms median LIVE_OK)
- audited→request (~3.1s median)
- request→finished (~5.6s median LIVE_OK)
- audited→LIVE_OK (~8.5s median)
- **WS message receive dedicado: não mensurável**
