# H3BUP Fase 2B — Sumário executivo (2026-07-29)

## Veredicto

**E2E_TRACE_IMPLEMENTED_INSUFFICIENT_N** — tracing ponta a ponta activo e fail-open; amostra natural sem LIVE_OK suficiente para estatística WS→LIVE_OK.

## TRACE

1. Nasce no audit (detecção/enqueue).
2. Propagado: result/hypothesis_details → bridge → `ExecutionRequest.meta.h3bup_e2e` → executor → JSONL live/result events.
3. `audit_id` preservado (campo dedicado).
4. `execution_id` preservado.
5. `order_id` só quando existe (LIVE_OK); CAP_BLOCKED sem order_id — OK.
6. Requests antigos compatíveis (meta opcional).
7. schema_version = **1**.
8. path = `logs/h3bup_e2e_trace.jsonl`.
9. Rotação: sim (`MAX_FILE_MB` + `BACKUP_COUNT`).
10. Contador dropped: sim (`trace_events_dropped`).

## TIMESTAMPS

11–21: implementados/reutilizados conforme design (`ws_received_ts`, `detected_at`, persist start/finish, bridge fetch, policy eval, executor received, dryrun start/finish, final gate, place start/finish, live_ok metadata em PLACE_FINISHED, result persist finish).

## LATÊNCIA (soak)

22. traces ≈ 136+ (ficheiro >1400 eventos no after-state)
23. LIVE_OK = **0**
24. coverage WS→LIVE_OK = **0%**
25–26. mediana/p95 WS→LIVE_OK = n/a
27. mediana audit→request ≈ **1748 ms** (N=2)
28. mediana request→LIVE_OK = n/a
29. mediana dry-run ≈ **1475 ms** (N=2)
30. mediana place = n/a
31. etapa dominante (amostra parcial): **espera audit→bridge/request** (+ dry-run quando executa)
32. durations negativas: poucas (clock skew marcado)
33. ordering violations: poucas (≤3)
34. duplicados: monitorizados pelo analyzer
35. traces incompletos: maioria pára em audit (não elegível / não seleccionada pelo bridge)
36. missing principal: **não chega a request/LIVE** (filtros naturais H3BUP + volume), não falha de telemetria

## OVERHEAD

37–38. detect→audit mediana ~15 ms vs baseline ~4 ms; path LIVE N insuficiente
39–40. CPU/mem sem regressão óbvia nos PIDs observados
41. budget estrito de +5 ms no segmento curto: **parcialmente excedido**; sem evidência de bloqueio
42. Execução bloqueada por telemetria? **Não**

## SEGURANÇA OPERACIONAL (42–58)

43–58: **Não** (policy/stake/odd/capacity/slippage/rise/poll/lookback/dedup/routing/betslips/ordens/cancel/accounting/CLV/fair edge inalterados quanto ao comportamento permitido).

## DEPLOY

59. Reiniciados: audit-ws-gate-back, executor-bridge-back, executor (sequencial).
60. Accounting HEALTHY? **Sim**
61. Rollback preparado? **Sim**
62. Rollback necessário? **Não**
63. Status final: **E2E_TRACE_IMPLEMENTED_INSUFFICIENT_N**
