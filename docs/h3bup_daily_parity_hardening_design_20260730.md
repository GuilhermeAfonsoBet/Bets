# H3BUP Daily Parity Hardening Design — 2026-07-30

## Objectivo
Reconciliar V1×V2 por **conjunto exacto de order_id**, separar `parity_as_of` de `matured_as_of`, e eliminar ambiguidade FILTER/JOIN sem root cause.

## Hipótese confirmada (2026-07-29)
Dois LIVE_OK após o cutoff oficial do V1 (`2026-07-29T22:01:54.606850Z`):

| order_id | created_at | stake | policy |
|----------|------------|------:|--------|
| 1938082582 | 22:21:51Z | 10 | H3BUP_vNext_20260629 |
| 1938105954 | 22:30:18Z | 10 | H3BUP_vNext_20260629 |

→ V1 freeze = 22 / 220; V2 full closed day = 24 / 240.

## Contratos
- **COHORT_WINDOW**: `[day, next_day)` created_at UTC
- **PARITY_AS_OF**: cutoff V1 oficial — só `created_at <= parity_as_of`
- **MATURED_AS_OF**: instante da execução manual — settlement actual

## Classificação
`EXPECTED_SCOPE_DIFFERENCE` (não V1_BUG / não V2_BUG): V1 é snapshot pré-fecho do dia; V2 `DAILY_CLOSED` inclui o dia completo.

## Segurança
Reporting-only; V2 permanece PREVIEW; sem Telegram/latest/timers/policy/stake.
