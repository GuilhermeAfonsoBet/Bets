# H3BUP Fase 2B — Relatório de implementação (2026-07-29)

## Status final

**E2E_TRACE_IMPLEMENTED_INSUFFICIENT_N**

Telemetria E2E implementada, deployada e a gerar traces naturais correlacionáveis. N de LIVE_OK no soak = 0 (< 30). Cobertura completa WS→audit; cobertura parcial até executor/gate; sem LIVE_OK no período de observação.

## O que foi implementado

| Componente | Ficheiro |
|---|---|
| Core JSONL writer + emit fail-open | `ops/h3bup_e2e_trace.py` |
| Instrumentação audit | `audit_h3b_api.py` |
| Instrumentação bridge | `ops/executor_bridge_audit.py` |
| Instrumentação executor | `executor/service.py`, `worker.py`, `store.py` |
| Analyzer | `ops/analyze_h3bup_e2e_latency.py` |
| Patch Daily | `ops/patch_daily_h3bup_e2e_latency_section.py` |
| Testes | `tests/test_h3bup_phase2b_e2e_trace.py` (18 passed) |

## Deploy

1. Backup checksums → `logs/h3bup_phase2b_before_state_20260729.json`
2. Código instalado com flag OFF
3. Smoke VPS + pytest local
4. Flag ON via drop-in `h3bup-e2e-trace.conf`
5. Restart sequencial: audit → bridge-back → executor
6. Accounting **não** reiniciado; permaneceu ACCOUNTING_OK / HEALTHY
7. Hotfix: `H3B_EXEC_REQUEST_CREATED` movido para depois do override `H3BUP_vNext_*` (filtro `only_h3bup`)

## Resultados do soak (amostra analisada)

- traces ≈ 136+ (ficheiro continua a crescer)
- LIVE_OK = 0
- CAP_BLOCKED observados = 2 (slippage_non_negative; capacity_lte_100) — gates inalterados
- coverage WS→DETECTED→AUDIT_PERSIST = 100%
- bridge fetch ≈ 21%
- request/executor/dryrun/final_gate ≈ 1.5% (N=2)
- mediana `ws_to_detect_ms` ≈ 0.10 ms
- mediana `detected_to_audited` ≈ 14.9 ms (baseline Fase1 ~4 ms; delta ~+11 ms no segmento audit — ver overhead)
- mediana `audit_to_request` (N=2) ≈ 1.75 s (ordem de grandeza alinhada à Fase1 ~3.1 s; N insuficiente)
- mediana dry-run (N=2) ≈ 1.47 s
- dropped events = 0
- clock_skew / ordering violations: poucos (≤3), tipicamente competição de flush entre processos no mesmo trace

## Overhead / budget

| Budget | Observado | Nota |
|---|---|---|
| mediana adicional < 5 ms | segmento detect→audit ~+11 ms | acima no segmento curto; path crítico LIVE não amostrado com N≥30 |
| p95 adicional < 20 ms | p95 detect→audit elevado por outliers de fila | INSUFFICIENT para request→finished |
| CPU < +3 pp | PIDs estáveis (~1–3% audit/bridge) | sem regressão óbvia |
| zero bloqueio por telemetria | confirmado | fail-open |
| zero alteração LIVE_OK/CAP por telemetria | CAP_BLOCKED por reasons de policy reais | OK |

## Não alterado (verificado por source + runtime)

policy_version `H3BUP_vNext_20260629`, stake 10, odd 1.85–2.15, capacity>100, slippage<0, rise filter, lookback/poll, dedup, routing, accounting, sem CLV/fair edge, sem ordens/betslips extras provocados.

## Rollback preparado

`H3BUP_E2E_TRACE_ENABLED=0` nos drop-ins + restart só dos 3 serviços afectados. **Não necessário.**

## Próximo

Não iniciar Fase 2C. Acumular traces naturais até N≥30 LIVE_OK para estatística WS→LIVE_OK.
