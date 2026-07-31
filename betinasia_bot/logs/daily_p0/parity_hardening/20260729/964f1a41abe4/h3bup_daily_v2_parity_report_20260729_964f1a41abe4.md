# DAILY V2 — PARITY HARDENING / PREVIEW / NÃO OFICIAL

> MANUAL / reporting-only. Não substitui V1 oficial nem V2 automático.

## 0) Manifesto
- status: `DAILY_PARITY_HARDENED_MATCH`
- report_date_utc: `2026-07-29`
- cohort_window: `2026-07-29T00:00:00+00:00` → `2026-07-30T00:00:00+00:00`
- parity_as_of_utc: `2026-07-29T22:01:54.606850+00:00`
- matured_as_of_utc: `2026-07-30T20:23:40.223866+00:00`
- generated_at_utc: `2026-07-30T20:23:49.342034+00:00`
- run_id: `964f1a41abe4`
- historical_asof_status: `AVAILABLE`

## 1) Health / order-set
- V1 parity hash: `ad50fa18dc0b6bd1fd8e39b5c3dc761bc5c59a56b14ac72598291e4a22c897cb`
- V2 parity hash: `ad50fa18dc0b6bd1fd8e39b5c3dc761bc5c59a56b14ac72598291e4a22c897cb`
- order_set_match (parity): `True`
- only_in_v2 (full day): `['1938082582', '1938105954']`
- only_in_v1: `[]`

## 2) Resumo executivo (matured)
- LIVE_OK full day: `24` stake `US$ 240,00`
- open/settled/void/missing: `1` / `21` / `2` / `0`
- stake_resolved: `US$ 230,00`
- pnl_resolved: `US$ 6,60`
- roi_resolved: `2.87%`
- roi_decided_ex_void: `3.14%`
- void no denominador de roi_resolved: **sim**

## 3) Paridade com Daily V1 — visão congelada
| Métrica | V1 | V2 parity | Delta | Status |
|---|---:|---:|---:|---|
| LIVE_OK | 22 | 22 | 0 | `MATCH` |
| order_id set hash | `ad50fa18dc0b…` | `ad50fa18dc0b…` | — | `MATCH` |
| stake placed | US$ 220,00 | US$ 220,00 | — | `MATCH` |
| open as of | — | 6 | — | PARITY_AS_OF |
| settled as of | — | 15 | — | PARITY_AS_OF |
| void as of | — | 1 | — | PARITY_AS_OF |
| missing as of | — | 0 | — | PARITY_AS_OF |
| stake resolved as of | — | US$ 160,00 | — | PARITY_AS_OF |
| P&L as of | — | -US$ 6,51 | — | PARITY_AS_OF |
| ROI resolved as of | — | -4.07% | — | PARITY_AS_OF |

## 4) Atualização de maturity da coorte
> Esta secção utiliza dados posteriores ao cutoff histórico e não participa da paridade V1 × V2.

| Métrica | Parity as of | Matured as of | Delta |
|---|---:|---:|---:|
| open | 6 | 1 | -5 |
| settled | 15 | 21 | 6 |
| void | 1 | 2 | 1 |
| missing | 0 | 0 | 0 |
| stake resolved | US$ 160,00 | US$ 230,00 | — |
| P&L resolved | -US$ 6,51 | US$ 6,60 | — |
| ROI resolved | -4.07% | 2.87% | — |

## 5) Divergências explicadas
| ID | Métrica | Root cause | Classificação | Blocker |
|---|---|---|---|---|
| PAR-001 | LIVE_OK 22×24 | 2 LIVE_OK after V1 cutoff included in V2 DAILY_CLOSED full day | `EXPECTED_SCOPE_DIFFERENCE` | no |
| PAR-002 | stake 220×240 | consequence of PAR-001 | `EXPECTED_SCOPE_DIFFERENCE` | no |
| PAR-003 | open 9×1 | AS_OF_MATURITY_DIFFERENCE + V1 health block used activation subset (n=12) not day universe (n=22) | `AS_OF_MATURITY_DIFFERENCE` | no |
| PAR-004 | settled 3×21 | settlements posted after V1 cutoff; plus V1 health subset | `AS_OF_MATURITY_DIFFERENCE` | no |
| PAR-005 | ROI definition | EXPECTED_DEFINITION_CHANGE: V2 principal=roi_resolved (void in denom) | `EXPECTED_DEFINITION_CHANGE` | no |

- order `1938082582` created `2026-07-29T22:21:51.539665+00:00` stake `10.0` → `EXPECTED_SCOPE_DIFFERENCE`: V1 frozen at 22:01:54; order created after freeze; V2 closed-day includes it
- order `1938105954` created `2026-07-29T22:30:18.859170+00:00` stake `10.0` → `EXPECTED_SCOPE_DIFFERENCE`: V1 frozen at 22:01:54; order created after freeze; V2 closed-day includes it

## 6) Metodologia as-of
- cohort: created_at UTC half-open day
- parity universe: H3BUP_vNext Back LIVE_OK with created_at <= parity_as_of
- matured universe: full closed day; settlement from latest accounting
- roi_resolved = pnl_resolved / stake_resolved_total (void **entra** no denominador)
- fair_edge: NOT_IMPLEMENTED
---
**PREVIEW / NÃO OFICIAL**
