# Executive — ROI acumulado + bootstrap H3BUP_vNext (policy-exact)

- **Status:** `BOOTSTRAP_ANALYSIS_COMPLETE_WITH_WARNINGS`
- **Classificação:** `NO_CLEAR_ROI_EDGE`
- **statistical_readiness:** `RELIABLE_READING_CANDIDATE`
- Void/push no denominador do ROI resolved: **sim**
- Friendly class: `FRIENDLY_CLASS_V1_20260731`
- Universo: `H3BUP_vNext` / `H3BUP_vNext_20260629` / `LIVE_OK` only

## Reconciliação

| # | Item | Valor |
|---|---|---|
| 1 | Primeiro LIVE_OK | 2026-07-28T15:30:25.309321+00:00 |
| 2 | Cutoff | 2026-08-14T13:48:03Z |
| 3 | LIVE_OK total | 331 |
| 4 | order_id únicos | 331 |
| 5 | event_id únicos | 258 (resolvidos: 233) |
| 6a | settled decided | 277 |
| 6b | void/push | 25 |
| 6c | open | 17 |
| 6d | missing | 12 |
| 6e | unreconciled | 0 |
| 7 | Stake colocada | 1350.00 |
| 8 | Stake resolvida | 1292.00 |
| 9 | Stake aberta | 34.00 |
| 10 | P&L resolvido | -25.96 |
| 11 | Accounting coverage | 91.24% |

Reconciliação OK: **True**

## ROI acumulado (desde o 1º LIVE_OK)

- N resolved: **302**
- Eventos únicos (resolvidos): **233**
- Stake resolved: **1292.00**
- P&L resolved: **-25.96**
- **ROI resolved acumulado: -2.01%**
- ROI decided ex-void: **-2.17%**

| Período | N resolvido | Stake | P&L | ROI |
|---|---:|---:|---:|---:|
| desde_inicio | 302 | 1292.00 | -25.96 | -2.01% |
| ultimos_7d | 110 | 220.00 | +2.60 | 1.18% |
| ultimas_50_resolvidas | 50 | 100.00 | -3.66 | -3.66% |
| ultimas_100_resolvidas | 100 | 200.00 | +4.51 | 2.25% |

## Bootstrap principal — cluster `event_id`

| Métrica | Resultado |
|---|---|
| N ordens | 302 |
| N eventos | 233 |
| ROI observado | -2.01% |
| Bootstrap mean | -1.95% |
| Bootstrap median | -1.97% |
| IC80 | [-8.96%, 5.05%] |
| IC90 | [-10.87%, 7.04%] |
| IC95 | [-12.55%, 8.70%] |
| IC99 | [-14.50%, 10.70%] |
| **P(ROI > 0%)** | **35.82%** |
| P(ROI > 2%) | 23.43% |
| P(ROI > 5%) | 10.15% |
| P(ROI > 10%) | 1.38% |
| P(ROI < 0%) | 64.18% |

> `P(ROI>0)` é a massa bootstrap acima de zero — **não** é automaticamente um p-value frequentista clássico.

### Order-level (secundário)
P(ROI>0) ordem = **37.93%** · mean=-2.01% · IC95=[-14.97%, 10.86%]

## Concentração / robustez

- Share P&L positivo top1/3/5 eventos: 3.06% / 9.00% / 14.70%
- Share |P&L| top1/3/5 eventos: 1.48% / 4.34% / 7.09%

| Cenário | N | Eventos | Stake | P&L | ROI | IC90 | IC95 | P(ROI>0) |
|---|---:|---:|---:|---:|---:|---|---|---:|
| A_drop_top1_winning_event | 301 | 232 | 1282.00 | -36.86 | -2.88% | [-11.74%, 6.16%] | [-13.48%, 7.92%] | 29.84% |
| B_drop_top3_winning_events | 299 | 230 | 1262.00 | -57.97 | -4.59% | [-13.38%, 4.26%] | [-15.07%, 5.92%] | 19.79% |
| C_drop_top5_winning_events | 297 | 228 | 1242.00 | -78.23 | -6.30% | [-14.98%, 2.55%] | [-16.70%, 4.30%] | 11.96% |
| D_drop_top1_positive_league | 290 | 225 | 1236.00 | -46.34 | -3.75% | [-12.91%, 5.54%] | [-14.63%, 7.39%] | 25.17% |
| E_drop_top3_positive_leagues | 260 | 202 | 1080.00 | -68.86 | -6.38% | [-15.84%, 3.27%] | [-17.61%, 5.09%] | 13.61% |

## Evolução temporal

| N | ROI | IC90 | IC95 | P(ROI>0) |
|---:|---:|---|---|---:|
| 25 | -3.38% | [-25.82%, 20.40%] | [-30.30%, 25.33%] | 40.51% |
| 50 | -8.84% | [-26.04%, 8.39%] | [-29.43%, 11.87%] | 19.65% |
| 75 | -1.00% | [-14.28%, 12.60%] | [-16.67%, 15.38%] | 45.05% |
| 100 | -3.86% | [-16.30%, 8.62%] | [-18.63%, 11.00%] | 30.37% |
| 150 | -3.26% | [-14.47%, 8.13%] | [-16.60%, 10.24%] | 31.53% |
| 200 | -2.70% | [-12.97%, 7.73%] | [-14.88%, 9.80%] | 33.41% |
| 250 | -1.71% | [-11.31%, 8.03%] | [-13.15%, 9.83%] | 38.19% |
| 302 | -2.01% | [-10.95%, 7.05%] | [-12.62%, 8.86%] | 36.12% |

Rolling:
- últimas 25: ROI 5.76% · P(>0) 64.22%
- últimas 50: ROI -3.66% · P(>0) 36.43%
- últimas 100: ROI 2.25% · P(>0) 60.95%
- Leitura temporal: **estável / oscilante sem tendência clara**

## Friendly vs Non-Friendly

| Grupo | N | Eventos | Stake | P&L | ROI | IC90 | IC95 | P(>0) | P(>5%) |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| FRIENDLY | 135 | 107 | 654.00 | -4.45 | -0.68% | [-12.95%, 12.12%] | [-15.38%, 14.60%] | 46.53% | 22.79% |
| NON_FRIENDLY | 167 | 126 | 638.00 | -21.51 | -3.37% | [-16.25%, 9.68%] | [-18.69%, 12.15%] | 33.59% | 14.54% |

- delta_ROI (NF − F) = **-2.69%**
- P(delta_ROI > 0) = **40.28%** (diagnóstico; não altera filtro)

## CLV (VALID_STRICT)

| Janela | N | Eventos | Coverage | Média | Mediana | IC90 mean | IC95 mean | P(mean>0) |
|---|---:|---:|---:|---:|---:|---|---|---:|
| POST_5M | 92 | 78 | 27.79% | -1.45% | -1.02% | [-1.83%, -1.08%] | [-1.90%, -1.01%] | 0.00% |
| POST_15M | 84 | 72 | 25.38% | -1.68% | -1.28% | [-2.16%, -1.22%] | [-2.26%, -1.13%] | 0.00% |
| CLOSING | 95 | 80 | 28.70% | -1.99% | -1.76% | [-2.71%, -1.29%] | [-2.86%, -1.15%] | 0.00% |

ROI realizado e CLV mesma direção? **sim**

## Tabela executiva obrigatória

| Pergunta | Resposta |
|---|---|
| Primeiro LIVE_OK | 2026-07-28T15:30:25.309321+00:00 |
| Cutoff | 2026-08-14T13:48:03Z |
| LIVE_OK total | 331 |
| Ordens resolvidas | 302 |
| Eventos únicos | 233 resolvidos / 258 LIVE_OK |
| Stake resolvida | 1292.00 |
| P&L acumulado | -25.96 |
| ROI acumulado | -2.01% |
| IC90 ROI cluster | [-10.87%, 7.04%] |
| IC95 ROI cluster | [-12.55%, 8.70%] |
| P(ROI > 0) | **35.82%** |
| P(ROI > 2%) | 23.43% |
| P(ROI > 5%) | 10.15% |
| P(ROI > 10%) | 1.38% |
| P(ROI < 0%) | 64.18% |
| P(ROI>0) sem maior evento vencedor | 29.84% |
| P(ROI>0) sem top 3 eventos | 19.79% |
| Friendly ROI / P(ROI>0) | -0.68% / 46.53% |
| Non-Friendly ROI / P(ROI>0) | -3.37% / 33.59% |
| CLV closing médio | -1.99% |
| CLV closing mediano | -1.76% |
| P(CLV closing médio > 0) | 0.00% |
| Statistical readiness | `RELIABLE_READING_CANDIDATE` |

## Respostas simples

1. ROI acumulado real H3BUP_vNext: **-2.01%** (void no denominador).
2. P(ROI>0) bootstrap por ordem: **37.93%**.
3. P(ROI>0) bootstrap cluster evento (principal): **35.82%**.
4. IC90 cluster: **[-10.87%, 7.04%]**.
5. IC95 cluster: **[-12.55%, 8.70%]**.
6. P(ROI>5%) cluster: **10.15%**.
7. Sem principais eventos vencedores: ROI permanece positivo? **não** (P>0 sem top1=29.84%, sem top3=19.79%).
8. Friendly vs Non-Friendly: F=-0.68% vs NF=-3.37%; P(delta>0)=40.28% — não diferem de forma clara.
9. CLV: média closing=-1.99%, P(mean>0)=0.00% — confirma a direção do ROI.
10. Evidência atual: **NO_CLEAR_ROI_EDGE** (`RELIABLE_READING_CANDIDATE`) — ainda inconclusiva para edge positivo operacional.

## Avisos
- partial_settlement open=17 missing=12 coverage=0.912

## Segurança
READ-ONLY. Policy/stake/bridge/executor/accounting/CLV/Telegram inalterados. 0 ordens / 0 betslips.
CLV 5m P(mean>0)=0.00% · 15m=0.00%.
Day-cluster N=18 P(>0)=24.45%.
