# H3BUP Fase 2B — Soak overhead (2026-07-29)

## Período
- start: 2026-07-29T15:53:09+00:00
- cutoff: 2026-07-29T20:07:33.532169+00:00
- classificação: **E2E_SOAK_SUFFICIENT**

## Detect→audit vs baseline Fase 1
| Métrica | Baseline Fase1 | Soak | Delta |
|---|---:|---:|---:|
| mediana detected_to_audited | ~4 ms | 13.867324 ms | ~+10 ms |
| p95 detected_to_audited | ~7 ms | 118.32473425 ms | elevado (fila/outliers) |

## Budget
- mediana adicional < 5 ms no segmento curto: **parcialmente excedido** (~+10 ms)
- sem bloqueio de execução
- CPU/serviços estáveis; accounting HEALTHY; tracing activo
- dropped/write errors críticos: não observados no soak health

## Conclusão overhead
E2E_SOAK_SUFFICIENT com WATCH leve no overhead detect→audit; sem rollback.
