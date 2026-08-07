# Relatório — Friendly vs Non-Friendly (H3BUP_vNext)

Data: 2026-08-01  
Run id: `78c9f53d95df` (corrige run `ae996053a99a`)  
Cutoff: `2026-08-01T01:31:39Z`  
Classificação: `FRIENDLY_CLASS_V1_20260731`  
Checksum: `1a9706023c55df4244dd2d15a1593137aae073339d2e612728c85972c2df0d8e`  
Status final: **`NON_FRIENDLY_WORSE_PRELIMINARY`**

> Análise histórica read-only. Não altera policy/stake/filtros. Não é recomendação operacional.

## Correção de P&L (importante)

O run anterior (`ae996053a99a`) somava **5 snapshots completos** de `balance.csv`,
multiplicando o P&L ≈5× e produzindo ROI absurdo (ex.: Non-Friendly −119%).

Contrato corrigido: usar **apenas o ledger mais recente** + dedupe por `transaction_id`.
ROI por ordem agora está na banda esperada de Back (sem perdas < −100%).

## Universo PRIMÁRIO (só H3BUP_vNext_20260629)

| Item | Valor |
|---|---|
| LIVE_OK | **74** |
| Friendly | **39** |
| Non-Friendly | **35** |
| Unclassified / Conflict | **0 / 0** |
| Stake | 10 em 74/74 |

## Performance (corrigida)

| Métrica | Friendly | Non-Friendly | Total |
|---|---:|---:|---:|
| Stake placed | 390 | 350 | 740 |
| Open | 1 | **17** | 18 |
| Settled decided | 35 | 17 | 52 |
| Void/push | 3 | 1 | 4 |
| Stake resolved | 380 | 180 | 560 |
| P&L resolved | **−12.54** | **−42.88** | **−55.42** |
| ROI resolved | **−3.3%** | **−23.8%** | **−9.9%** |
| ROI ex-void | −3.6% | −25.2% | −10.7% |
| Accounting coverage | **97.4%** | **51.4%** | 75.7% |
| Sample gate | INSUFFICIENT_N | VERY_LOW_N | INSUFFICIENT_N |

### Leitura

1. Friendly **negativo leve** (−3.3%).
2. Non-Friendly **mais negativo** no resolvido (−23.8%), mas com só 180 de stake resolved vs 380 Friendly e **17 open**.
3. Maior parte da perda resolved: **Non-Friendly** (−42.9 vs −12.5).
4. Evidência ainda **INSUFFICIENT_N**; cobertura accounting assimétrica.
5. Não usar isto como filtro operacional.

## Segurança

Policy / stake / executor / accounting / CLV / timers / Telegram / ordens / betslips: **Não alterados**.

## Outputs

`logs/h3bup_friendly_analysis/20260801/78c9f53d95df/`
