# MANUAL VALIDATION — Executive Summary

**Status:** `DAILY_MANUAL_VALIDATION_WITH_WARNINGS`

## Execução

| # | Pergunta | Resposta |
|---|----------|----------|
| 1 | run_id | `99f5924aec32` |
| 2 | commit executado | runner `33d628a` (branch `cursor/h3bup-daily-p0-47ee`); VPS git HEAD local `d4a3d42` (árvore suja, sem checkout) |
| 3 | report_date | `2026-07-29` |
| 4 | cohort window | `2026-07-29T00:00:00+00:00` → `2026-07-30T00:00:00+00:00` (created_at UTC) |
| 5 | parity cutoff | `2026-07-29T22:01:54.606850+00:00` (V1 oficial) |
| 6 | performance_as_of | `2026-07-29T22:01:54.606850+00:00` |
| 7 | generated_at | `2026-07-30T19:47:25.884636+00:00` |
| 8 | V1 exit code | `0` |
| 9 | V2 exit code | `0` |

## Outputs

| # | Artefacto | OK |
|---|-----------|----|
| 10 | V1 PDF | Sim — `H3BUP_DAILY_V1_MANUAL_VALIDATION_20260729_99f5924aec32.pdf` |
| 11 | V2 PDF | Sim — `H3BUP_DAILY_V2_MANUAL_PREVIEW_20260729_99f5924aec32.pdf` |
| 12 | JSON canónico | Sim |
| 13 | CSV paridade | Sim |
| 14 | Health 4D | Sim |
| 15 | Alertas | Sim (6) |
| 16 | CLV CSV | Sim |
| 17 | E2E CSV | Sim |
| 18 | Funil | Sim |
| 19 | Outputs oficiais preservados | Sim (checksums before=after) |

## Paridade (mesma coorte / cutoff)

| Métrica | V1 | V2 | Status |
|---------|----|----|--------|
| report_date | 2026-07-29 | 2026-07-29 | MATCH |
| LIVE_OK | 22 | 24 | FILTER_DIFFERENCE |
| stake placed | 220 | 240 | FILTER_DIFFERENCE |
| open | 9 | 1 | JOIN_DIFFERENCE |
| settled | 3 | 21 | JOIN_DIFFERENCE |
| void | — | 2 | PARITY_UNAVAILABLE (V1 não separava) |
| ROI | accounting-block | roi_resolved 2.87% | EXPECTED_DEFINITION_CHANGE |

- Divergências UNKNOWN em LIVE_OK/stake/settlement/P&L/ROI: **nenhuma**
- Divergências esperadas: filtro H3BUP_vNext (V2), join accounting mais maduro as-of-now vs bloco V1, ROI resolved vs ROI parcial do health V1

## H3BUP_vNext (V2)

- LIVE_OK: 24 · stake_placed: US$ 240 · stake_resolved_total: US$ 230 · stake_void: US$ 20
- open/settled/void/missing: 1 / 21 / 2 / 0
- pnl_resolved: US$ 6,60 · roi_resolved: 2,87% (PARTIAL) · maturity: PARTIALLY_SETTLED
- Fórmula: `roi_resolved = pnl_resolved / stake_resolved_total` (**void no denominador**)

## Health 4D

- REPORT_HEALTH=HEALTHY
- OPERATIONS_HEALTH=WATCH
- DATA_QUALITY=WATCH
- STATISTICAL_READINESS=INSUFFICIENT_N

Alertas: CLV_INSUFFICIENT_N, E2E_OVERHEAD_WATCH, SETTLEMENT_PARTIAL, CLV_BACKLOG, TRACE_ORDERING_VIOLATIONS, TRACE_CLOCK_SKEW

## Segurança (30–41)

Todas **Não**: Telegram oficial, sobrescrita V1/V2/latest, timers, policy, stake, accounting, E2E, CLV, ordens, betslips.

Timers `betinasia-daily-full-report` e `betinasia-daily-v2-shadow` permanecem **active**.

## Directório

`logs/daily_p0/manual_validation/20260729/99f5924aec32/`
