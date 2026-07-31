# H3BUP Daily Parity Hardening Report — 2026-07-30

## Status
`DAILY_PARITY_HARDENED_MATCH`

run_id: `964f1a41abe4`  
parity_as_of: `2026-07-29T22:01:54.606850+00:00`  
matured_as_of: `2026-07-30T20:23:40.223866+00:00`

## PAR-001 / PAR-002 — Universo 22×24
| order_id | created_at | stake | policy | classificação |
|----------|------------|------:|--------|---------------|
| 1938082582 | 2026-07-29T22:21:51Z | 10 | H3BUP_vNext_20260629 | EXPECTED_SCOPE_DIFFERENCE |
| 1938105954 | 2026-07-29T22:30:18Z | 10 | H3BUP_vNext_20260629 | EXPECTED_SCOPE_DIFFERENCE |

Root cause: V1 congelado às 22:01:54; V2 `DAILY_CLOSED` inclui o dia UTC completo.  
Correcção: visão parity filtra `created_at <= parity_as_of` → **LIVE_OK 22 / stake 220 / hashes iguais**.

Não é V1_BUG nem V2_BUG.

## PAR-003 / PAR-004 — Open/Settled
- V1 health block (subset activação n=12): open=9 settled=3  
- Parity as-of (universo 22 + snapshot accounting ≤cutoff): open=6 settled=15 void=1  
- Matured as-of: open=1 settled=21 void=2  

Classificação: `AS_OF_MATURITY_DIFFERENCE` (+ subset do health V1).

## PAR-005 — ROI
`roi_resolved = pnl_resolved / stake_resolved_total` com **void no denominador**.  
Também exposto `roi_decided_ex_void`. `EXPECTED_DEFINITION_CHANGE`.

## Segurança
Official latest/V1 preservados; Telegram não usado; timers active; policy/stake/E2E/CLV/ordens inalterados.
