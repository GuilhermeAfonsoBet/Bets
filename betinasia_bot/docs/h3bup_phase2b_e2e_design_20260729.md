# H3BUP Fase 2B — Design E2E Latency Tracing (2026-07-29)

## Objectivo

Instrumentação append-only, fail-open, da latência ponta a ponta H3BUP_vNext **sem** alterar policy, stake, gates, routing, audit logic, execução LIVE ou accounting.

## Escolha de `trace_id`

Formato: `h3bup:<audit_id|na>:<uuid12>`

- Nasce no audit no momento da detecção/enqueue (`audit_id` ainda `na`).
- `audit_id` numérico é propagado nos eventos após `H3B_AUDIT_PERSIST_FINISHED`.
- Relação audit↔trace: campo `audit_id` em cada evento + `hypothesis_details._e2e_trace_id` na row.
- Não substitui `audit_id`, `execution_id` nem `order_id`.
- Requests antigos sem `meta.h3bup_e2e.trace_id` continuam válidos.

## Inventário de timestamps (antes → depois)

| Etapa | Timestamp existente | Fonte | TZ | Precisão | Cobertura | Reutilizável? |
|---|---|---|---|---|---|---|
| WS receive | `_last_ws_time` | audit_h3b_api | epoch local→UTC | s | todas msgs | Sim → `ws_received_ts` |
| Detect | `detected_at` | audit queue item | epoch→UTC | s/ms | H3B detect | Sim → `detected_ts` |
| Audit gate | `telemetry.execution_ms` | `_execute_ws_gate_back` | relativo | ms | gate back | Parcial |
| Audit persist | `audited_at` (DB) | betslip_audit_results | UTC | us | DB rows | Sim (fim) |
| Bridge | poll loop | executor_bridge_audit | UTC | s | candidates | Novo `bridge_fetched_ts` |
| Policy | shadow `decision_at` | backpre_shadow_all | UTC | us | shadow | Novo eval started/finished |
| Request | `request.created_at` | ExecutionRequest | UTC | us | submits | Sim + `request_*_ts` |
| Executor recv | `received_ts` local | service.submit | epoch | s | queue | Sim → evento |
| Dry-run | timing.* | worker | ms | parcial | dry | Novos start/finish |
| Final gate | — | worker H3BUP_VNEXT_GATE | — | — | live path | Novo |
| Place | `order_post_ms` | worker | ms | duração | LIVE | Novos start/finish |
| Result | `finished_at` | executor_live.jsonl | UTC | us | results | + persist events |

## Eventos (schema_version=1)

Lista exacta em `ops/h3bup_e2e_trace.py` (`_EVENT_NAMES`). Storage: `logs/h3bup_e2e_trace.jsonl` com writer thread, rotação por tamanho, sample_rate, métricas dropped/errors.

## Fail-open

Toda emissão em `try/except`; nunca altera status/reason/retry/place. Flag `H3BUP_E2E_TRACE_ENABLED` (default off no código; on via drop-in systemd após testes).

## Analyzer / Daily

- `ops/analyze_h3bup_e2e_latency.py` (read-only)
- Secção Daily via `ops/patch_daily_h3bup_e2e_latency_section.py` (fail-open se ficheiro ausente)

## Fora de âmbito

CLV, fair edge, Fase 2C, migrations DB, alteração de policy/stake/thresholds.
