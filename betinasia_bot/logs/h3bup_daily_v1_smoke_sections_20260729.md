# Daily Report BetinAsia

- Dia do relatório (UTC): `20260729`
- Gerado em (UTC): `2026-07-29T21:15:09.258625+00:00`

## 0) Resumo e conclusões (executivo)

**Status do OOS (walk-forward)**

- **OOS**: **FAILED** — `OOS_FAILED: returncode=1`
- Log: `logs/daily_reports_smoke_2r/20260729/oos_run.log`

- **Accounting**: indisponível (ver apêndice 99.1)
## Accounting Health — H3BUP

| Métrica | Valor |
|---|---|
| status | ACCOUNTING_OK / HEALTHY |
| último sucesso UTC | 2026-07-29T21:12:30.838226+00:00 |
| balance age | 7.114010572433472 |
| open_stakes age | 0.05135488510131836 |
| falhas consecutivas | 0 |
| última falha | None |
| LIVE_OK total | 12 |
| settled reconciliado | 3 |
| não iniciados | — |
| abertos | 9 |
| missing accounting | — |
| coverage accounting | 0.25 |
| stake settled | 30.0 |
| P&L settled | -0.4300000000000015 |
| ROI settled | -0.014333333333333384 |

_Disclaimer: ROI settled é parcial (N baixo e/ou coverage/health insuficientes); não é ROI total da estratégia._

## H3BUP End-to-End Latency

| Métrica | Valor |
|---|---|
| tracing status | ENABLED |
| schema version | 1 |
| traces totais | 2195 |
| traces LIVE_OK | 12 |
| coverage WS→LIVE_OK | 0.5% |
| mediana WS→LIVE_OK | 8809.345 ms |
| p95 WS→LIVE_OK | 12558.715 ms |
| mediana audit→request | 3178.442 ms |
| mediana request→LIVE_OK | 5787.158 ms |
| mediana dry-run | 1375.311 ms |
| mediana place | 4228.545 ms |
| etapa dominante | place_duration_ms |
| trace events dropped | 0 |
| clock skew violations | 74 |
| ordering violations | 74 |
| status estatístico | INSUFFICIENT_N |

### Funil de cobertura

| Etapa | N | % |
|---|---:|---:|
| H3B_WS_RECEIVED | 2195 | 100.0% |
| H3B_DETECTED | 2195 | 100.0% |
| H3B_AUDIT_PERSIST_FINISHED | 2195 | 100.0% |
| H3B_BRIDGE_FETCHED | 654 | 29.8% |
| H3B_EXEC_REQUEST_CREATED | 73 | 3.3% |
| H3B_EXECUTOR_RECEIVED | 73 | 3.3% |
| H3B_DRYRUN_FINISHED | 73 | 3.3% |
| H3B_FINAL_GATE_DECIDED | 66 | 3.0% |
| H3B_PLACE_FINISHED | 12 | 0.5% |
| LIVE_OK | 12 | 0.5% |

## H3BUP CLV Forward Collection

| Métrica | Valor |
|---|---|
| collection status | WATCH |
| collection started at | 2026-07-29T20:07:50+00:00 |
| source priority | best_odds_history,passive_collector |
| passive collector status | ENABLED |
| LIVE_OK após activação | 2 |
| obligations esperadas | 6 |
| obligations criadas | 6 |
| POST_5M strict válidas | 2 |
| POST_15M strict válidas | 2 |
| CLOSING strict válidas | 0 |
| source missing | 0 |
| line mismatch | 0 |
| kickoff missing | 0 |
| retry backlog | 2 |
| status estatístico | INSUFFICIENT_N |
