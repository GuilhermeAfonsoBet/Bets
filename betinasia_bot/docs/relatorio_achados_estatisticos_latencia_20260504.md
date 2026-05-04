# Achados estatísticos — Latência de execução vs ROI (Back)

Gerado em: **2026-05-04 (UTC)**

## Escopo e objetivo

- Objetivo: avaliar ROI por buckets de latência de execução e diferença estatística entre grupos.
- Janela: `since_day = 2026-04-20` (UTC).
- Buckets disjuntos de latência (`call_to_done_ms`):
  - `<4s`
  - `4-6s`
  - `>6s`
- Universo principal de interesse operacional: **Back PRE estrito** (`market_regime='pre'`).
- Também foi mostrado o universo **ALL Back** (PRE + IN) para contraste.

## Base de dados utilizada

- `executor_jsonl`: `betinasia_bot/logs/executor_live.jsonl`
- `balance_csv`: `betinasia_bot/logs/accounting/20260503_220214__balance.csv`
- `n_exec_back_live_ok`: **1663**
- `n_orders no balance (após filtro de data)`: **1510**
- `n_join_total (executor x balance por order_id)`: **1428**
- `n_join_pre_estrito`: **292**

## Composição por regime (JOIN)

| Regime | n | % |
|---|---:|---:|
| In | 1136 | 79.6% |
| Pre | 292 | 20.4% |

> Conclusão de composição: o universo ALL Back está fortemente dominado por Back In; por isso, inferência para a tese de latência em Back Pre deve priorizar o recorte PRE estrito.

## Resultado A — ALL Back (PRE + IN)

| Bucket | n | Stake total | Stake médio | Slippage_pre médio | ROIw | ROI mean | IC95 mean | ROI mediana | IC95 mediana |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| <4s | 261 | 765.00 | 2.93 | 1.74% | 1.20% | -9.48% | -19.37% .. 0.24% | 0.00% | -49.67% .. 0.00% |
| 4-6s | 407 | 1464.00 | 3.60 | 0.31% | -6.15% | -7.11% | -15.06% .. 0.73% | 0.00% | -47.33% .. 0.00% |
| >6s | 760 | 2092.50 | 2.75 | 2.54% | -3.85% | -4.23% | -10.03% .. 1.54% | 0.00% | 0.00% .. 0.00% |

### Diferença estatística (delta de ROI mean; bootstrap)

| Comparação | Delta ROI mean | IC90 | IC95 |
|---|---:|---:|---:|
| <4s - >6s | -5.33% | -14.61% .. 4.30% | -16.49% .. 6.43% |
| 4-6s - >6s | -2.77% | -11.28% .. 5.67% | -12.63% .. 7.14% |
| <4s - 4-6s | -2.54% | -13.07% .. 7.74% | -14.87% .. 9.80% |

Leitura: sem evidência robusta de diferença de média no ALL Back (intervalos cruzam zero).

## Resultado B — PRE estrito (`market_regime='pre'`)

| Bucket | n | Stake total | Stake médio | Slippage_pre médio | ROIw | ROI mean | IC95 mean | ROI mediana | IC95 mediana |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| <4s | 41 | 435.00 | 10.61 | -0.49% | 11.77% | 7.99% | -17.51% .. 34.33% | 0.00% | -49.67% .. 88.25% |
| 4-6s | 95 | 996.00 | 10.48 | -0.44% | -5.31% | -4.32% | -22.20% .. 13.57% | 0.00% | -97.58% .. 55.83% |
| >6s | 156 | 1186.50 | 7.61 | -0.55% | -4.73% | -10.22% | -23.43% .. 3.27% | -49.71% | -50.17% .. 0.00% |

### Diferença estatística (delta de ROI mean; bootstrap)

| Comparação | Delta ROI mean | IC90 | IC95 |
|---|---:|---:|---:|
| <4s - >6s | 18.65% | -6.69% .. 43.78% | -12.03% .. 48.90% |
| 4-6s - >6s | 5.72% | -13.44% .. 24.21% | -16.99% .. 27.87% |
| <4s - 4-6s | 12.50% | -14.29% .. 39.36% | -19.95% .. 44.06% |

Leitura: há sinal direcional favorável a menor latência no PRE, porém sem significância estatística robusta no recorte atual (ICs amplos e cruzando zero).

## Resultado C — PRE estrito com controle de covariáveis

Padronização por estratos (`stake_bin x slippage_bin`) para aproximar comparação ceteris paribus.

- `n_pre_total`: **292**
- `n_pre_usado_no_controle`: **221**
- `n_estratos_comuns`: **2**

### ROI mean padronizado

| Bucket | ROI mean padronizado |
|---|---:|
| <4s | 9.41% |
| 4-6s | -4.34% |
| >6s | -2.70% |

### Deltas padronizados (bootstrap)

| Comparação | Delta ROI mean padronizado | IC90 | IC95 |
|---|---:|---:|---:|
| <4s - >6s | 11.59% | -20.08% .. 44.09% | -26.12% .. 49.80% |
| 4-6s - >6s | -1.71% | -22.89% .. 19.95% | -26.66% .. 23.76% |

Leitura: após controle por stake/slippage, a incerteza permanece alta; ainda não há evidência estatística conclusiva de efeito causal isolado da latência no recorte atual.

## Interpretação executiva

- A diferença entre ALL Back e PRE estrito é material: ALL Back está contaminado por In.
- No PRE estrito, o bucket `<4s` apresenta ponto estimado melhor no ROI mean, mas com IC95 amplo.
- Medianas próximas de `0.00%` em vários buckets sugerem massa relevante de ordens com resultado próximo a zero (incluindo possíveis ordens neutras/void-like no ledger).
- Para decisão operacional forte, recomenda-se ampliar janela temporal e/ou aumentar N efetivo por bucket no PRE estrito.

## Recomendação de próximos passos (estatísticos)

- Repetir este relatório com janela maior (ex.: desde `2026-03-20`) mantendo PRE estrito.
- Incluir estratos adicionais (ex.: odds bucket e minuto relativo ao kickoff) para reduzir viés de composição.
- Monitorar estabilidade rolling (ex.: janelas móveis de 7 dias) para detectar regime shift.

