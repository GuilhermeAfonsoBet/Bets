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

- **Lucro esperado (com gate de slippage; exec c/ placar)**: `26.84` (base `34.05`, Δ `-7.21`)

**Conversão (últimas 24h; auditoria DB)**

- OK/total: **4101/4748** (86.4%)
- OK_valid/total: **4101/4748** (86.4%)

**Saúde do executor (amostra lida do JSONL; não é 24h)**

- Janela: `2026-06-24T18:26:47.110876+00:00` → `2026-07-29T21:14:23.000197+00:00` (n=50000)
- Maior gap: `107,022.3s` | gaps>5min: `1`

**Saúde do executor (últimas 24h; proxy por gaps no JSONL)**

- Janela: `2026-07-28T21:15:48.686919+00:00` → `2026-07-29T21:15:48.686947+00:00` (n=1597)
- Maior gap: `248.2s` | gaps>15min: `0` | silêncio>15min (est.): `0s` (0.00%)

**Recursos da VPS (snapshot)**

- MemAvailable: `3,427 MiB`
- vCPUs (os.cpu_count): `4`

**Atividade recente (executor)**

- Último `LIVE_OK`: `2026-07-29T20:37:54.942782+00:00` | `LIVE_OK` (1h/6h/24h): `2/13/23`

**Falhas pós-accepted (executor, 24h)**

| Métrica | Valor |
|---|---:|
| accepted | 172 |
| LIVE_OK | 23 (13.4%) |
| accepted sem LIVE_OK | 149 (86.6%) |
| precheck fail (`LIVE_PRECHECK_FAILED`) | 19 |
| place fail (`LIVE_PLACE_FAILED`) | 0 |
| API_FAILED | 16 |
| NO_SESSION | 3 |
| RATE_LIMIT | 0 |
| CAP_BLOCKED | 130 |
| No PMMs received | 12 |
| Execution context destroyed/target closed | 2 |
| Auth 401 / NO_ROOT_SESSION_COOKIE | 3 |
| p50/p90 `pmm_wait_ms` (precheck fail) | — / — |
| p50/p90 `ws_age_ms` (precheck fail) | — / — |

- Top erros pós-accepted:
  - ×86: `H3BUP_VNEXT_GATE capacity_lte_100`
  - ×24: `H3BUP_VNEXT_GATE capacity_lte_100|slippage_non_negative`
  - ×20: `H3BUP_VNEXT_GATE slippage_non_negative`
  - ×11: `No PMMs received (waited 1.2s) | LIVE_PRECHECK_FAILED`
  - ×3: `NO_ROOT_SESSION_COOKIE | LIVE_PRECHECK_FAILED`
  - ×2: `NO_VALID_BOOKMAKER_PRICES | LIVE_PRECHECK_FAILED`

**Latência ponta a ponta (24h; WS → executor_done)**

- Cobertura: `n_jsonl_24h=1596`, `com_audit_id=172`, `com_hypothesis_detected_at=172`, `e2e_all=172`, `e2e_success=23`.
| Etapa | p50 | p90 | p99 | mean |
|---|---:|---:|---:|---:|
| e2e_total | 8,676 | 11,596 | 13,560 | 8,893 |
| detect_to_submit | 3,196 | 3,330 | 4,915 | 2,962 |
| audit_total | 0 | 5 | 11 | 2 |
| audit_detect_to_click | 0 | 4 | 4 | 1 |
| audit_click_to_betslip | 0 | 0 | 0 | 0 |
| audit_queue_wait | 0 | 4 | 4 | 1 |
| audit_parallel_fetch | — | — | — | — |
| audit_temporal_total | — | — | — | — |
| audit_execution | 0 | 0 | 0 | 0 |
| audit_pipeline_overhead | 0 | 0 | 0 | 0 |
| audit_db_save | 11 | 33 | 1,181 | 66 |
| audit_gate_wait | — | — | — | — |
| bridge_wait | 3,196 | 3,330 | 4,915 | 2,961 |
| executor_submit_to_done | 5,578 | 9,025 | 10,230 | 5,930 |
| executor_queue_delay | 0 | 2 | 45 | 2 |
| executor_post | 821 | 1,711 | 4,428 | 1,199 |
| executor_total_api | 1,357 | 2,211 | 4,779 | 1,693 |


**Prontidão para LIVE (go/no-go)**

| Critério | Atual | Alvo | Status |
|---|---:|---:|---|
| Live liberado (`EXECUTOR_ALLOW_LIVE`) | `False` | `True` | **FAIL** |
| OK_valid/total (24h, DB) | 86.4% | ≥5.0% | **OK** |
| API_FAILED/total (24h, DB) | 0.0% | ≤20.0% | **OK** |
| STALE_QUEUE_WAIT/total (24h, DB) | 0.0% | ≤10.0% | **OK** |
| `No PMMs received` (24h, DB) | 0 | ≤0 | **OK** |
| `No PMMs` / `PMM-consults` (24h, DB) | — | — | — |
| `too_many_open_betslips` (24h, DB) | 0 | ≤0 | **OK** |
| Latência p90 `call_to_done_ms` (24h; sucessos) | 8,900ms | ≤8000ms | **FAIL** |
| Latência p50 `call_to_done_ms` (24h; sucessos) | 5,578ms | — | — |
| n sucessos no JSONL (24h) | 23 | — | — |
| Gaps >15min no executor_jsonl (24h; proxy) | 0 | ≤8 | **OK** |

**Veredito**: **NÃO APTO**

**Motivos (prioridade)**

- LIVE bloqueado (`EXECUTOR_ALLOW_LIVE=0`)

**Próximos passos recomendados (para destravar LIVE)**

- Atacar `No PMMs received` (timeout/min_wait/idle + estabilidade de sessão) antes de aumentar volume.
- Zerar `too_many_open_betslips` (caps/janelas + cleanup agressivo) para evitar bloqueio global.
- Reduzir `STALE_QUEUE_WAIT` (fila/concurrency) para não operar atrasado.


**Conclusões operacionais (prioridades)**

- **Objetivo 1 (conversão)**: reduzir `API_FAILED` (especialmente `No PMMs received`) e `STALE_QUEUE_WAIT` para aumentar taxa de execução sem inflar risco.
- **Objetivo 2 (governança de risco)**: consolidar sizing/limites (banca teórica vs banca real) e travas para evitar picos (`too_many_open_betslips`, rate limit, backoff).
- **Objetivo 3 (qualidade de entrada)**: acompanhar slippage **com sinal** e seu impacto em ROI por bucket (negativo/flat/positivo) para validar edge e execução.

## 1) Resultados reais (shadow/live)

_Sem série de accounting disponível para métricas diárias/Sharpe/DD (ver 99.1)._ 

**Execução — métricas mínimas por tipo (Back/Lay × Pre/In; janela curta)**

| Tipo | #ordens | #eventos_jsonl | #linhas_api | #jogos | Valor em risco ($) | Ticket médio ($/ordem) | Stake total ($) | #liq | #pend | P&L (liq, $) | ROI% (liq) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Back Pre | 23 | 23 | 0 | 18 | 230.00 | 10.00 | 230.00 | 0 | 23 | 0.00 | — |
| Back In | 0 | 0 | 0 | 0 | 0.00 | — | 0.00 | 0 | 0 | 0.00 | — |
| Lay Pre | 0 | 0 | 0 | 0 | 0.00 | — | 0.00 | 0 | 0 | 0.00 | — |
| Lay In | 0 | 0 | 0 | 0 | 0.00 | — | 0.00 | 0 | 0 | 0.00 | — |
| **TOTAL** | **23** | **23** | **0** | **18** | **230.00** | **10.00** | **230.00** | **0** | **23** | **0.00** | **—** |

**Execução (últimos dias; executor_jsonl + placares quando disponíveis)**

| Dia | Exec rows | Sucessos | LIVE_OK | DRY_OK | API_FAILED | N Back | N Lay | Apostado Back ($) | Apostado Lay stake ($) | Apostado Lay liab ($) | P&L total (acct; post date UTC) | ROI/$ (acct) | P&L Back (acct; oid join) | P&L Back Pre (acct; oid) | P&L Back In (acct; oid) | Δ (acct_total - acct_back_oid) | Cobertura oids% (Back) | P&L (placar) | ROI/$ (placar) | P&L Back | ROI Back | P&L Lay | ROI Lay/liab | ROI Lay/stake |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-07-23 | 68 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | — | — | — | — | — | — | — | 0.00 | — | 0.00 | — | 0.00 | — | — |
| 2026-07-24 | 167 | 0 | 0 | 0 | 6 | 0 | 0 | 0.00 | 0.00 | 0.00 | — | — | — | — | — | — | — | 0.00 | — | 0.00 | — | 0.00 | — | — |
| 2026-07-25 | 111 | 0 | 0 | 0 | 10 | 0 | 0 | 0.00 | 0.00 | 0.00 | — | — | — | — | — | — | — | 0.00 | — | 0.00 | — | 0.00 | — | — |
| 2026-07-26 | 38 | 0 | 0 | 0 | 3 | 0 | 0 | 0.00 | 0.00 | 0.00 | — | — | — | — | — | — | — | 0.00 | — | 0.00 | — | 0.00 | — | — |
| 2026-07-27 | 52 | 0 | 0 | 0 | 3 | 0 | 0 | 0.00 | 0.00 | 0.00 | — | — | — | — | — | — | — | 0.00 | — | 0.00 | — | 0.00 | — | — |
| 2026-07-28 | 23 | 3 | 3 | 0 | 1 | 3 | 0 | 30.00 | 0.00 | 0.00 | — | — | — | — | — | — | — | 26.84 | 89.47% | 26.84 | 89.47% | 0.00 | — | — |
| 2026-07-29 | 171 | 22 | 22 | 0 | 16 | 22 | 0 | 220.00 | 0.00 | 0.00 | — | — | — | — | — | — | — | 7.21 | 24.03% | 7.21 | 24.03% | 0.00 | — | — |

_Nota: `P&L total (acct)` é calculado por **post date UTC** diretamente do `balance.csv` quando disponível (exclui depósitos/saques/etc.). `P&L Back Pre/In (acct; order_id)` é **Back-only** via join `order_id` (ledger ↔ executor_jsonl) e inclui tipos P&L-like (ex.: void/refund) quando existirem. Se o CSV não tiver `order_id`, esses campos podem ficar vazios._

**Cobertura de placar (somente entre execuções bem-sucedidas)**

| Dia | Back n_cov/n_success | Back stake_cov/stake | Back jogos_cov/jogos_success | Lay n_cov/n_success | Lay stake_cov/stake | Lay jogos_cov/jogos_success |
|---|---:|---:|---:|---:|---:|---:|
| 2026-07-23 | — | — | — | — | — | — |
| 2026-07-24 | — | — | — | — | — | — |
| 2026-07-25 | — | — | — | — | — | — |
| 2026-07-26 | — | — | — | — | — | — |
| 2026-07-27 | — | — | — | — | — | — |
| 2026-07-28 | 3/3 (100.0%) | 30.00/30.00 (100.0%) | 3/3 (100.0%) | — | — | — |
| 2026-07-29 | 3/22 (13.6%) | 30.00/220.00 (13.6%) | 2/17 (11.8%) | — | — | — |

**Quebra (placar): Back/Lay × Pre/In (somente cobertos por ROI)**

| Dia | P&L Back Pre | ROI Back Pre | P&L Back In | ROI Back In | P&L Lay Pre | ROI Lay Pre/liab | P&L Lay In | ROI Lay In/liab |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-07-23 | 0.00 | — | 0.00 | — | 0.00 | — | 0.00 | — |
| 2026-07-24 | 0.00 | — | 0.00 | — | 0.00 | — | 0.00 | — |
| 2026-07-25 | 0.00 | — | 0.00 | — | 0.00 | — | 0.00 | — |
| 2026-07-26 | 0.00 | — | 0.00 | — | 0.00 | — | 0.00 | — |
| 2026-07-27 | 0.00 | — | 0.00 | — | 0.00 | — | 0.00 | — |
| 2026-07-28 | 26.84 | 89.47% | 0.00 | — | 0.00 | — | 0.00 | — |
| 2026-07-29 | 7.21 | 24.03% | 0.00 | — | 0.00 | — | 0.00 | — |

**Latência × ROI (Back Pre/In) — acumulado (call_to_done_ms)**

- Latência por execução vem de result.timing.call_to_done_ms no executor_jsonl. ROI/placar usa somente odd executada (odd_final). Se odd_final estiver ausente, a execução não entra no subconjunto coberto.

- **Back Pre (ROI por stake)**

| Bucket call_to_done_ms | n | ROI mean (SE; IC95) |
|---|---:|---:|
| < 5s | 611 | 20.71% (SE 3.82%) [13.23%, 28.18%] | ROIw 21.91% (odd~1.94, exp~20.00) |
| 5-10s | 634 | 12.99% (SE 3.52%) [6.10%, 19.88%] | ROIw 13.54% (odd~1.94, exp~12.00) |
| 10-20s | 194 | 12.99% (SE 6.19%) [0.86%, 25.11%] | ROIw 21.85% (odd~1.94, exp~3.00) |
| 20-40s | 22 | 12.68% (SE 17.60%) [-21.83%, 47.18%] | ROIw 8.13% (odd~1.95, exp~3.00) |
| > 40s | 8 | -6.65% (SE 30.59%) [-66.61%, 53.31%] | ROIw 42.87% (odd~1.87, exp~1.50) |

- **Back In (ROI por stake)**

| Bucket call_to_done_ms | n | ROI mean (SE; IC95) |
|---|---:|---:|
| < 5s | 1008 | 34.87% (SE 4.89%) [25.29%, 44.46%] | ROIw 50.74% (odd~1.93, exp~3.00) |
| 5-10s | 1179 | 37.01% (SE 3.85%) [29.47%, 44.55%] | ROIw 36.71% (odd~1.95, exp~3.00) |
| 10-20s | 701 | 32.23% (SE 4.66%) [23.11%, 41.36%] | ROIw 31.33% (odd~1.95, exp~3.00) |
| 20-40s | 265 | 44.84% (SE 9.66%) [25.91%, 63.77%] | ROIw 38.53% (odd~1.94, exp~3.00) |
| > 40s | 25 | 43.75% (SE 16.41%) [11.58%, 75.92%] | ROIw 38.96% (odd~2.00, exp~3.00) |

**Slippage × Latência (Back Pre/In) — acumulado (call_to_done_ms)**

- Slippage_raw_pct vs latência usa execuções com ROI via placar e odd_final presente; slippage_raw_pct=(odd_final-odd_at_decision)/odd_at_decision.

- **Back Pre (slippage_raw_pct por stake)**

| Bucket call_to_done_ms | n | Slippage_raw mean (SE; IC95) | Slippage_raw mediana | Slippage_raw_w (por exp.) | ROIw (por exp.) |
|---|---:|---:|---:|---:|---:|
| < 5s | 611 | 0.29% (SE 0.55%) [-0.78%, 1.37%] | -0.44% | -0.14% | 21.91% |
| 5-10s | 634 | -0.47% (SE 0.07%) [-0.61%, -0.34%] | -0.46% | -0.41% | 13.54% |
| 10-20s | 194 | -0.64% (SE 0.19%) [-1.03%, -0.26%] | -0.52% | -0.25% | 21.85% |
| 20-40s | 22 | -0.20% (SE 0.25%) [-0.70%, 0.30%] | -0.24% | -0.35% | 8.13% |
| > 40s | 8 | -1.87% (SE 1.24%) [-4.30%, 0.56%] | -1.69% | -3.30% | 42.87% |

- **Back In (slippage_raw_pct por stake)**

| Bucket call_to_done_ms | n | Slippage_raw mean (SE; IC95) | Slippage_raw mediana | Slippage_raw_w (por exp.) | ROIw (por exp.) |
|---|---:|---:|---:|---:|---:|
| < 5s | 1008 | 8.34% (SE 1.93%) [4.56%, 12.13%] | 0.00% | 26.82% | 50.74% |
| 5-10s | 1179 | 33.25% (SE 21.31%) [-8.51%, 75.01%] | 0.00% | 172.03% | 36.71% |
| 10-20s | 701 | 23.85% (SE 15.12%) [-5.80%, 53.49%] | 0.00% | 95.01% | 31.33% |
| 20-40s | 265 | 0.96% (SE 0.81%) [-0.63%, 2.54%] | 0.00% | 1.20% | 38.53% |
| > 40s | 25 | 1.28% (SE 1.08%) [-0.85%, 3.40%] | 0.16% | 0.07% | 38.96% |

**Slippage × ROI por bucket (raw, com sinal) — acumulado (range: `2026-02-26` → `2026-07-29`; span_days=`154`)**

- **Back (ROI por stake)**

| Bucket slippage_raw_pct | n | ROI mean (SE; IC95) |
|---|---:|---:|
| <= -2% | 695 | 36.26% (SE 3.69%) [29.03%, 43.49%] | ROIw 38.50% (odd~1.90, exp~3.00) |
| (-2, 2] | 3202 | 21.71% (SE 1.64%) [18.49%, 24.92%] | ROIw 18.98% (odd~1.94, exp~3.00) |
| > 2% | 750 | 57.71% (SE 8.40%) [41.24%, 74.18%] | ROIw 67.97% (odd~2.04, exp~3.00) |

- **Lay (ROI por liability)**

| Bucket slippage_raw_pct | n | ROI mean (SE; IC95) |
|---|---:|---:|
| <= -2% | 223 | -2.72% (SE 45.24%) [-91.40%, 85.95%] | ROIw -61.79% (odd~1.08, exp~0.19) |
| (-2, 2] | 30 | 7.84% (SE 17.68%) [-26.82%, 42.50%] | ROIw 10.83% (odd~1.88, exp~2.24) |
| > 2% | 233 | 84.97% (SE 51.65%) [-16.27%, 186.21%] | ROIw -20.87% (odd~3.46, exp~3.68) |

- **Back Pre (ROI por stake)**

| Bucket slippage_raw_pct | n | ROI mean (SE; IC95) |
|---|---:|---:|
| <= -2% | 116 | 15.91% (SE 7.81%) [0.61%, 31.22%] | ROIw 24.01% (odd~1.88, exp~11.00) |
| (-2, 2] | 1300 | 15.85% (SE 2.45%) [11.04%, 20.66%] | ROIw 17.52% (odd~1.94, exp~12.00) |
| > 2% | 53 | 22.25% (SE 19.51%) [-15.99%, 60.48%] | ROIw 25.53% (odd~2.00, exp~20.00) |

- **Back In (ROI por stake)**

| Bucket slippage_raw_pct | n | ROI mean (SE; IC95) |
|---|---:|---:|
| <= -2% | 579 | 40.33% (SE 4.13%) [32.25%, 48.42%] | ROIw 47.39% (odd~1.91, exp~3.00) |
| (-2, 2] | 1902 | 25.71% (SE 2.19%) [21.42%, 30.00%] | ROIw 22.01% (odd~1.93, exp~3.00) |
| > 2% | 697 | 60.41% (SE 8.91%) [42.94%, 77.88%] | ROIw 77.57% (odd~2.05, exp~3.00) |

- Nota: `ROIw` é o **ROI ponderado por exposição** (peso=stake no Back; peso=liability no Lay). Em prática, dentro de um bucket, `ROIw ≈ (∑P&L)/(∑exposição)`; já o `ROI mean` é a média simples por linha/sinal.

- Nota importante (reconciliação): as tabelas **Slippage × ROI** usam **somente execuções cobertas por ROI via placar** (precisa audit+placar+odd). Isso é um subconjunto e pode ter viés (ex.: jogos ainda não liquidaram, falta de odds finais, etc.). Já o **accounting ledger** inclui todo o resultado financeiro (incluindo void/refund/cancel quando existirem) por `post date`.

- **Lay (ROI por stake; bounded)**

| Bucket slippage_raw_pct | n | ROI mean (SE; IC95) |
|---|---:|---:|
| <= -2% | 223 | -4.00% (SE 2.04%) [-8.00%, 0.00%] | ROIw -6.34% (odd~1.08, exp~1.00) |
| (-2, 2] | 30 | 2.32% (SE 17.04%) [-31.09%, 35.72%] | ROIw 9.47% (odd~1.88, exp~3.00) |
| > 2% | 233 | -114.78% (SE 26.33%) [-166.38%, -63.18%] | ROIw -68.29% (odd~3.46, exp~1.00) |

**Contrafactual (placar): aplicar filtro de slippage**

- Regra: **Back** pula `slippage_raw_pct <= -2%`; **Lay** pula `slippage_raw_pct > 2%`.
- Observação: usa somente execuções com ROI via placar; não é o P&L do accounting.

| Lado | n (base) | P&L (base) | Exposição (base) | n (após filtro) | P&L (após) | Exposição (após) |
|---|---:|---:|---:|---:|---:|---:|
| Back | 4647 | 8,709.06 | 31,340.50 | 3952 | 7,317.16 | 27,725.50 |
| Lay (liab) | 486 | -699.00 | 3,012.27 | 440 | -541.85 | 2,453.18 |
| **Total** | — | 8,010.06 | — | — | 6,775.31 | — |

**Diagnóstico AH (linha) observado na execução**

- Policy: `ah_max_abs_line=0.0` | `ah_scope=all`
- Execuções (todas): `n=8933` | `max|line|=10.00` | `n_over=7679`
- Execuções com placar/ROI: `n=6276` | `max|line|=10.00` | `n_over=5344`

**Slippage × ROI por combinação (top 2 por volume; acumulado)**

- **Back**

| Combinação | n | ROI<=-2% | n | ROI(-2..2] | n | ROI>2% | n | corr(slip_raw,ROI) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Back_In_Any | 3178 | 40.33% (SE 4.13%) [32.25%, 48.42%] | 579 | 25.71% (SE 2.19%) [21.42%, 30.00%] | 1902 | 60.41% (SE 8.91%) [42.94%, 77.88%] | 697 | 0.06 |
| Back_Pre_Any | 1469 | 15.91% (SE 7.81%) [0.61%, 31.22%] | 116 | 15.85% (SE 2.45%) [11.04%, 20.66%] | 1300 | 22.25% (SE 19.51%) [-15.99%, 60.48%] | 53 | 0.20 |

**Slippage × ROI por combinação (top 2 por volume; acumulado)**

- **Lay**

| Combinação | n | ROI<=-2% | n | ROI(-2..2] | n | ROI>2% | n | corr(slip_raw,ROI) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Lay_In_Yes | 404 | -6.99% (SE 52.13%) [-109.17%, 95.20%] | 188 | 11.56% (SE 17.89%) [-23.52%, 46.63%] | 29 | 112.61% (SE 64.19%) [-13.20%, 238.43%] | 187 | 0.06 |
| Lay_Pre_Yes | 82 | 20.18% (SE 69.55%) [-116.14%, 156.50%] | 35 | -100.00% | 1 | -27.42% (SE 8.95%) [-44.96%, -9.88%] | 46 | -0.05 |

**Slippage × Latência (Back Pre/In) — pós-início (>= 2026-04-04)**

- Slippage_raw_pct vs latência usa execuções com ROI via placar e odd_final presente; slippage_raw_pct=(odd_final-odd_at_decision)/odd_at_decision.

- **Back Pre (slippage_raw_pct por stake)**

| Bucket call_to_done_ms | n | Slippage_raw mean (SE; IC95) | Slippage_raw mediana | Slippage_raw_w (por exp.) | ROIw (por exp.) |
|---|---:|---:|---:|---:|---:|
| < 5s | 585 | -0.50% (SE 0.06%) [-0.63%, -0.38%] | -0.44% | -0.52% | 22.39% |
| 5-10s | 598 | -0.48% (SE 0.07%) [-0.63%, -0.34%] | -0.47% | -0.43% | 15.02% |
| 10-20s | 184 | -0.70% (SE 0.20%) [-1.09%, -0.31%] | -0.52% | -0.45% | 19.07% |
| 20-40s | 21 | -0.17% (SE 0.27%) [-0.69%, 0.35%] | -0.05% | -0.15% | 12.65% |
| > 40s | 5 | -1.27% (SE 1.64%) [-4.49%, 1.95%] | -1.80% | -1.27% | -13.46% |

- **Back In (slippage_raw_pct por stake)**

| Bucket call_to_done_ms | n | Slippage_raw mean (SE; IC95) | Slippage_raw mediana | Slippage_raw_w (por exp.) | ROIw (por exp.) |
|---|---:|---:|---:|---:|---:|
| < 5s | 905 | 2.88% (SE 0.95%) [1.01%, 4.75%] | 0.00% | 3.55% | 33.04% |
| 5-10s | 1097 | 2.93% (SE 1.06%) [0.85%, 5.02%] | 0.00% | 2.94% | 37.64% |
| 10-20s | 664 | 12.30% (SE 10.10%) [-7.49%, 32.09%] | 0.00% | 13.90% | 31.66% |
| 20-40s | 262 | 0.95% (SE 0.82%) [-0.65%, 2.56%] | 0.00% | 1.19% | 47.10% |
| > 40s | 22 | 1.75% (SE 1.16%) [-0.53%, 4.02%] | 0.35% | 1.75% | 46.78% |

**Slippage × ROI por bucket (raw, com sinal) — pós-início (>= 2026-04-04) (range: `2026-04-04` → `2026-07-29`; span_days=`117`)**

- **Back (ROI por stake)**

| Bucket slippage_raw_pct | n | ROI mean (SE; IC95) |
|---|---:|---:|
| <= -2% | 649 | 35.82% (SE 3.74%) [28.48%, 43.15%] | ROIw 29.93% (odd~1.91, exp~3.00) |
| (-2, 2] | 3030 | 22.28% (SE 1.68%) [18.98%, 25.58%] | ROIw 21.14% (odd~1.94, exp~3.00) |
| > 2% | 664 | 50.69% (SE 8.40%) [34.22%, 67.16%] | ROIw 46.05% (odd~2.03, exp~3.00) |

- **Back Pre (ROI por stake)**

| Bucket slippage_raw_pct | n | ROI mean (SE; IC95) |
|---|---:|---:|
| <= -2% | 112 | 14.06% (SE 8.00%) [-1.61%, 29.74%] | ROIw 19.83% (odd~1.88, exp~10.00) |
| (-2, 2] | 1237 | 16.37% (SE 2.52%) [11.44%, 21.30%] | ROIw 19.09% (odd~1.94, exp~12.00) |
| > 2% | 44 | 4.84% (SE 13.71%) [-22.04%, 31.72%] | ROIw 16.83% (odd~1.98, exp~12.00) |

- **Back In (ROI por stake)**

| Bucket slippage_raw_pct | n | ROI mean (SE; IC95) |
|---|---:|---:|
| <= -2% | 537 | 40.35% (SE 4.18%) [32.16%, 48.54%] | ROIw 39.83% (odd~1.92, exp~3.00) |
| (-2, 2] | 1793 | 26.36% (SE 2.25%) [21.95%, 30.77%] | ROIw 27.58% (odd~1.94, exp~3.00) |
| > 2% | 620 | 53.94% (SE 8.93%) [36.43%, 71.45%] | ROIw 56.26% (odd~2.04, exp~3.00) |

- Nota: `ROIw` é o **ROI ponderado por exposição** (peso=stake no Back; peso=liability no Lay). Em prática, dentro de um bucket, `ROIw ≈ (∑P&L)/(∑exposição)`; já o `ROI mean` é a média simples por linha/sinal.

- Nota importante (reconciliação): as tabelas **Slippage × ROI** usam **somente execuções cobertas por ROI via placar** (precisa audit+placar+odd). Isso é um subconjunto e pode ter viés (ex.: jogos ainda não liquidaram, falta de odds finais, etc.). Já o **accounting ledger** inclui todo o resultado financeiro (incluindo void/refund/cancel quando existirem) por `post date`.

**Contrafactual (placar): aplicar filtro de slippage**

- Regra: **Back** pula `slippage_raw_pct <= -2%`; **Lay** pula `slippage_raw_pct > 2%`.
- Observação: usa somente execuções com ROI via placar; não é o P&L do accounting.

| Lado | n (base) | P&L (base) | Exposição (base) | n (após filtro) | P&L (após) | Exposição (após) |
|---|---:|---:|---:|---:|---:|---:|
| Back | 4343 | 5,688.59 | 23,323.50 | 3694 | 4,930.88 | 20,791.50 |
| Lay (liab) | 0 | 0.00 | 0.00 | 0 | 0.00 | 0.00 |
| **Total** | — | 5,688.59 | — | — | 4,930.88 | — |

**Diagnóstico AH (linha) observado na execução**

- Policy: `ah_max_abs_line=0.0` | `ah_scope=all`
- Execuções (todas): `n=8934` | `max|line|=10.00` | `n_over=7680`
- Execuções com placar/ROI: `n=6276` | `max|line|=10.00` | `n_over=5344`

**Slippage × ROI por combinação (top 2 por volume; acumulado)**

- **Back**

| Combinação | n | ROI<=-2% | n | ROI(-2..2] | n | ROI>2% | n | corr(slip_raw,ROI) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Back_In_Any | 2950 | 40.35% (SE 4.18%) [32.16%, 48.54%] | 537 | 26.36% (SE 2.25%) [21.95%, 30.77%] | 1793 | 53.94% (SE 8.93%) [36.43%, 71.45%] | 620 | 0.03 |
| Back_Pre_Any | 1393 | 14.06% (SE 8.00%) [-1.61%, 29.74%] | 112 | 16.37% (SE 2.52%) [11.44%, 21.30%] | 1237 | 4.84% (SE 13.71%) [-22.04%, 31.72%] | 44 | 0.02 |

**Funil de oportunidades (últimas 24h; auditoria DB)**

| audit_version | total | OK | OK_valid | GATE_NOT_ELIGIBLE | API_FAILED | STALE_QUEUE_WAIT |
|---|---:|---:|---:|---:|---:|---:|
| v5.3-ws-gate-back | 4101 | 4101 | 4101 | 0 | 0 | 0 |
| v1.0 | 647 | 0 | 0 | 0 | 0 | 0 |

**Oportunidades identificadas / melhorias propostas (curto prazo)**

- **PMM/timeout**: se `No PMMs received` dominar, aumentar timeout efetivo e reduzir bursts (workers/queue) tende a elevar conversão sem mexer na estratégia.
- **Betslips abertos**: `too_many_open_betslips` é um gargalo de throughput; manter caps/janelas e garantir cleanup rápido evita bloqueio global.
- **Fila**: `STALE_QUEUE_WAIT` indica atraso interno; atacar latência/concorrência antes de aumentar volume/seleção.

**Portfólio OOS: vigente vs histórico recente**

| ts | n_active_keys |
|---|---:|
| 2026-06-29T11:30:03.161126+00:00 | — |
| 2026-06-29T11:45:03.568376+00:00 | — |
| 2026-06-29T12:00:03.170738+00:00 | — |
| 2026-06-29T12:15:04.115571+00:00 | — |
| 2026-06-29T12:30:02.651953+00:00 | — |
| 2026-06-29T12:45:03.010166+00:00 | — |
| 2026-06-29T13:00:05.927827+00:00 | — |
| 2026-06-29T13:15:03.295033+00:00 | — |

**Parâmetros vigentes (visão executiva)**

- **Combinações ativas (OOS)**: ver `99.3` (active_keys) e o bloco `2) OOS`.
- **Stake sizing operacional (real)**: hoje é **FLAT** via `BRIDGE_STAKE` (ver `99.3` e `99.6`).
- **Parâmetros técnicos efetivos** (executor/audit/bridge): ver `99.6 Filtros ativos`.

**Critérios de seleção (OOS) e critérios do real (bridge/executor)**

- **OOS (walk-forward)** decide o portfólio `active_keys`.
  - **Chave por liga**: `True` (scope=`pre`) ⇒ em pre-match a chave pode virar `...__<League>`.
  - **Filtro de AH ativo?**: `True` (max_abs_line=`2.00`; scope=`pre`) ⇒ remove eventos com `abs(line)` acima do limiar.
  - **Mínimo de jogos no treino**: `wf_min_matches=0` (0 = desligado).
  - **Regra de decisão (por combinação, no treino)**:
    - Se `ROI` for **significativamente negativo** (IC90 inteiro < 0): **bloqueia**.
    - Se `ROI` for **significativamente positivo** (IC90 inteiro > 0): **ativa**.
    - Se `ROI` > 0 mas **não significativo**:
      - **Pre-match**: ativa apenas se **CLV > 0** (CLV não precisa ser sig.).
      - **In-match**: ativa se **ROI > 0** (CLV não se aplica).
  - Operacionalmente, o OOS também pode excluir buckets de execução (ex.: `wf_exclude_exec_buckets_back`).
- **Real (shadow/live)**:
  - O bridge só envia oportunidades cuja chave esteja em `active_keys` (policy current).
  - `DRY_OK` = **shadow** (não apostou); `LIVE_OK` = **efetivo** (apostou).

**Este período está rodando shadow ou efetivo?**

- Predominantemente **efetivo**: `LIVE_OK=210` (e `DRY_OK=0`).

**Aspectos técnicos (latência/estabilidade)**

- Latência detalhada: ver `99.2` (p50/p90/p99 por etapa).
- Gaps no `executor_jsonl` (proxy de downtime/restart/sem tráfego): max `107,022.3s`, gaps>5min `1`, gaps>15min `1`.



## 99) Operacional — saldo, P&L e execução

### 99.1 Accounting (saldo + P&L)

- Arquivo: `logs/daily_reports_smoke_2r/20260729/accounting_daily_report.json`
- **Erro**: **ACCOUNTING_SKIPPED (DAILY_SKIP_ACCOUNTING=1)**
- Saldo atual: **None**
- P&L hoje/semana/mês: **None / None / None**

Meses fechados:

| Mês | P&L |
|---|---:|

### 99.2 Execução (KPIs)

- Fonte: `logs/executor_live.jsonl`
- Nota: métricas abaixo vêm do JSONL; se ele estiver **stale** ou incompleto, podem divergir do volume “24h, DB”.

**Status (all)**

| Status | N |
|---|---:|
| CAP_BLOCKED | 1025 |
| LIVE_OK | 210 |
| API_FAILED | 132 |
| STALE | 31 |
| NO_SESSION | 25 |

**Latência (somente LIVE_OK/DRY_OK) — ms**

| Métrica | n | p50 | p90 | p99 | mean |
|---|---:|---:|---:|---:|---:|
| queue_delay | 210 | 0.0 | 2.0 | 5762.269999999998 | 201.67142857142858 |
| call_to_done | 210 | 5133.5 | 8743.699999999999 | 18781.449999999997 | 6145.72380952381 |
| post | 210 | 837.5 | 2462.599999999998 | 7091.279999999996 | 1341.5333333333333 |

**Latência (últimas 24h; somente LIVE_OK/DRY_OK) — ms**

| Métrica | n | p50 | p90 | p99 | mean |
|---|---:|---:|---:|---:|---:|
| queue_delay | 23 | 0.0 | 1.8000000000000007 | 35.54000000000005 | 2.4347826086956523 |
| call_to_done | 23 | 5578.0 | 8900.0 | 10155.2 | 5930.347826086957 |
| post | 23 | 821.0 | 1708.6 | 4084.3600000000015 | 1198.9130434782608 |

**Slippage (somente LIVE_OK/DRY_OK, quando houver odd_at_decision)**

- Definição: `slippage = odd_final - odd_at_decision` (em odds decimais) e `slippage_pct = slippage/odd_at_decision`.
- Interpretação depende do lado:
  - **Back**: slippage_pct **negativo** = piorou (odd caiu); **positivo** = melhorou.
  - **Lay**: slippage_pct **positivo** = piorou (odd subiu); **negativo** = melhorou.

| Tipo | n | p50 | p90 | p99 | mean |
|---|---:|---:|---:|---:|---:|
| abs | 210 | -0.010999999999999788 | -0.0020000000000000018 | 0.005459999999999783 | -0.02321904761904763 |
| pct | 210 | -0.5398778922692229 | -0.10985731440276907 | 0.29166666666665503 | -1.141568412614914 |

**Slippage por lado (Back vs Lay)**

| Lado | Métrica | n | p50 | p90 | p99 | mean |
|---|---|---:|---:|---:|---:|---:|
| Back | slippage_pct (raw) | 210 | -0.5398778922692229 | -0.10985731440276907 | 0.29166666666665503 | -1.141568412614914 |
| Back | slippage_pct (custo, >=0) | 210 | 0.5398778922692229 | 3.2885423223034667 | 8.735382057118866 | 1.2461717192595967 |

_Nota: o p90/p99 de `call_to_done_ms` explode quando inclui `NO_SESSION/API_FAILED` (timeouts/relogin). Por isso reportamos também o recorte apenas de sucessos._


### 99.6 Filtros ativos (config efetiva)

_Nota: esta seção reflete as variáveis carregadas pelo `daily_full_report` (via `.env`). Services do systemd podem ter overrides (`Environment=`) que não aparecem aqui; use `systemctl show` para confirmar no VPS._

**Executor**

| chave | valor |
|---|---|
| EXECUTOR_ALLOW_LIVE | `` |
| EXECUTOR_WORKERS | `` |
| EXECUTOR_QUEUE_MAX | `` |
| EXECUTOR_CAP_WINDOW_SEC | `` |
| EXECUTOR_CAP_MAX | `` |
| EXECUTOR_BACKPRE_FAST_STAKE_ENABLE | `` |
| EXECUTOR_BACK_STAKE_SIZING_ENABLE | `` |
| EXECUTOR_BACK_STAKE_SLIP_NEG_LIMIT_PCT | `` |
| EXECUTOR_BACK_STAKE_SLIP_POS_LIMIT_PCT | `` |
| EXECUTOR_BACK_STAKE_SLIP_NEG | `` |
| EXECUTOR_BACK_STAKE_SLIP_MID | `` |
| EXECUTOR_BACK_STAKE_SLIP_POS | `` |
| EXECUTOR_BACK_LATENCY_GATE_ENABLE | `` |
| EXECUTOR_BACK_LATENCY_GATE_MAX_SEC | `` |
| EXECUTOR_FAST_PMM | `` |
| EXECUTOR_PMM_TIMEOUT_SEC | `` |
| EXECUTOR_PMM_MIN_WAIT_SEC | `` |
| EXECUTOR_PMM_IDLE_TIMEOUT_SEC | `` |
| EXECUTOR_BETSLIP_CACHE_MAX_KEYS | `` |

**Audit H3B**

| chave | valor |
|---|---|
| AUDIT_MODE | `` |
| AUDIT_API_SIDES | `` |
| AUDIT_EXECUTOR_WORKERS | `` |
| AUDIT_TEMPORAL_WORKERS | `` |
| AUDIT_MAX_QUEUE_DEPTH | `` |
| AUDIT_MAX_QUEUE_WAIT_MS | `` |
| WS_SAMPLE_OFFSETS_SEC | `` |
| GATE_DROP_OFFSET_SEC | `` |
| GATE_DROP_RATIO | `` |
| GATE_RISE_OFFSET_SEC | `` |
| GATE_RISE_RATIO | `` |
| GATE_OPEN_WINDOW_SEC | `` |
| GATE_OPEN_MAX | `` |
| GATE_MAX_LATE_SEC | `` |
| GATE_LAY_REFRESH_TIMES_SEC | `` |

_Nota (Back vs Lay): o `AUDIT_MODE` acima costuma refletir o serviço principal (ex.: `ws_gate_lay`). Em operação real, o **Back** pode vir de um serviço separado (ex.: `betinasia-audit-api-back`, `audit_version=v5.2-api-back`) ou de uma variante `ws_gate_back` (dependendo do deploy). Para confirmar o que rodou nas últimas 24h, veja `99.5 Auditoria (DB)`._

**Interpretação operacional (timing de entrada)**

| Item | Regra efetiva |
|---|---|
| Back (mais cedo possível) | Depende do executor: `EXECUTOR_FAST_PMM`, `EXECUTOR_PMM_MIN_WAIT_SEC`, `EXECUTOR_PMM_TIMEOUT_SEC` (ver tabela Executor). |
| Lay (reversão vs fim) | Depende do `AUDIT_MODE`/audit_version: `ws_gate_lay` abre Lay só quando o gate em `t+GATE_DROP_OFFSET_SEC` passa; `ws_reversal_lay` tende a entrar no pós-reversal; `ws_only` usa a série WS (offsets até o último ponto, tipicamente 30s). |

**Bridge**

| chave | valor |
|---|---|
| BRIDGE_MODE | `` |
| BRIDGE_EXEC_SIDE | `` |
| BRIDGE_STAKE | `` |
| BRIDGE_POLL_SEC | `` |
| BRIDGE_LOOKBACK_SEC | `` |
| BRIDGE_MAX_PER_CYCLE | `` |
| BRIDGE_PREMATCH_ONLY | `` |
| BRIDGE_POLICY_JSON | `` |
| BRIDGE_POLICY_RELOAD_SEC | `` |
| BRIDGE_POLICY_USE_BASE | `` |
| BRIDGE_MIN_LIMIT | `` |

**OOS / Walk-forward (daily)**

| chave | valor |
|---|---|
| DAILY_OOS_DIRECTION | `` |
| DAILY_OOS_VERSIONS | `` |
| DAILY_OOS_LOOKBACK_DAYS | `` |
| DAILY_WF_TRAIN_MODE | `` |
| DAILY_WF_TRAIN_DAYS | `` |
| DAILY_WF_TEST_DAYS | `` |
| DAILY_WF_STEP_DAYS | `` |
| DAILY_WF_SIDES | `` |
| DAILY_WF_REGIMES | `` |
| DAILY_WF_BACKPRE_SLIP_MAX | `` |
| DAILY_WF_BACKPRE_SLIP_FIELD | `` |
| DAILY_WF_BACKPRE_FAST_MAX_LAG_MS | `` |
| DAILY_WF_KEY_BY_LEAGUE | `` |
| DAILY_WF_KEY_BY_LEAGUE_SCOPE | `` |
| DAILY_WF_AH_MAX_ABS_LINE | `` |
| DAILY_WF_AH_SCOPE | `` |
| DAILY_WF_EXCLUDE_EXEC_BUCKETS_BACK | `` |


### 99.4 Aderência OOS (portfolio por dia × execução)

- Arquivo (curto): `logs/daily_reports_smoke_2r/20260729/oos_adherence_short.json`
- Arquivo (acumulado/slippage): `logs/daily_reports_smoke_2r/20260729/oos_adherence_long.json`
- Policy current: `logs/wf_policy_current.json`

**Resumo (últimos dias)**

| Dia | Ativas (keys) | Bridge rows | Skipped(not_active) | Exec rows | LIVE_OK | DRY_OK | Back bloqueadas (slip<=-2%; cov) | Lay bloqueadas (slip>2%; cov) | ΔP&L cf (placar; cov) | P&L Back | ROI Back | P&L Lay | ROI Lay/liab | P&L total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-07-23 | 22 | 1663 | 0 | 68 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | — | 0.00 | — | 0.00 |
| 2026-07-24 | 22 | 742 | 0 | 167 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | — | 0.00 | — | 0.00 |
| 2026-07-25 | 22 | 402 | 0 | 111 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | — | 0.00 | — | 0.00 |
| 2026-07-26 | 22 | 212 | 0 | 38 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | — | 0.00 | — | 0.00 |
| 2026-07-27 | 22 | 811 | 0 | 52 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | — | 0.00 | — | 0.00 |
| 2026-07-28 | 22 | 851 | 0 | 23 | 3 | 0 | 0 | 0 | 0.00 | 26.84 | 89.47% | 0.00 | — | 26.84 |
| 2026-07-29 | 22 | 1440 | 0 | 171 | 22 | 0 | 3 | 0 | -7.21 | 7.21 | 24.03% | 0.00 | — | 7.21 |

**Slippage × ROI (raw, com sinal; 3 buckets) — acumulado (range: `2026-02-26` → `2026-07-29`; span_days=`154`; cut=`None`)**

- **Back (ROI por stake)**

| Bucket slippage_raw_pct | n | ROI mean |
|---|---:|---:|
| <= -2% | 695 | 36.26% |
| (-2, 2] | 3202 | 21.71% |
| > 2% | 750 | 57.71% |

- **Lay (ROI por liability)**

| Bucket slippage_raw_pct | n | ROI mean |
|---|---:|---:|
| <= -2% | 223 | -2.72% |
| (-2, 2] | 30 | 7.84% |
| > 2% | 233 | 84.97% |


### 99.5 Auditoria (DB) — motivos de no-OK (por versão)

- Arquivo: `logs/daily_reports_smoke_2r/20260729/audit_status_kpis.json`
- Janela: últimas **24.0h** (desde `2026-07-28T21:15:41.113950+00:00`)

**Definições (colunas)**

- **OK**: `status='OK'` no `betslip_audit_results` (a auditoria concluiu com sucesso).
- **OK com betslip_odd**: subset de OK em que `betslip_odd` está preenchido (houve snapshot do ticket/odds).
- **OK valid**: subset de OK em que `is_valid_opportunity=true` (passou o critério operacional de “oportunidade executável”).
  - Na prática, o `is_valid_opportunity` tende a cair quando `difference_pct` está fora do range aceito (edge muito pequeno <2% ou mismatch >10%) ou quando campos essenciais do ticket estão ausentes.

**Glossário rápido (`audit_version`)**

| padrão | significado |
|---|---|
| `v5.2-api-back` | Back via API (serviço back-only); tende a abrir betslip e medir limites/odds. |
| `v5.1-ws-gate-lay` | Lay via WS gate (queda em 5s); só abre ticket quando o gate passa. |
| `v5.4-ws-reversal-lay` | Lay no pós-reversal; volume baixo pode ser “evento raro” (depende de reversões). |
| `v5.3-ws-gate-back` | Back via WS gate; se `OK` é baixo, costuma indicar gate muito restritivo, parse/click falhando, ou credenciais/sessão instável. |
| `v4.*` / `v1.*` | versões antigas/legadas do pipeline (API/WS), úteis para comparação histórica. |

| audit_version | total | OK | OK com betslip_odd | OK valid | top no-OK |
|---|---:|---:|---:|---:|---|
| v5.3-ws-gate-back | 4101 | 4101 | 0 | 4101 | — |
| v1.0 | 647 | 0 | 0 | 0 | LINE_NOT_AVAILABLE=445, GAME_NOT_FOUND=192, MAJOR_DIFF=10 |

**Diagnóstico dos OK (por versão): buckets de |difference_pct|**

_Leitura: `OK valid` tende a ser aproximadamente o bucket `2% ≤ |difference_pct| ≤ 10%` (dependendo da regra vigente)._

| audit_version | OK diff nulo | OK |diff|<2% | OK 2–10% | OK |diff|>10% |
|---|---:|---:|---:|---:|
| v5.3-ws-gate-back | 4101 | 0 | 0 | 0 |
| v1.0 | 0 | 0 | 0 | 0 |



## Anexo B) Ajuste operacional (slippage gate × capacidade)


### Ajuste operacional: Sensibilidade por banca com gate de slippage (contrafactual)

_Leitura: aplica a regra `Back: pula slippage_raw_pct<=-2%` e `Lay: pula slippage_raw_pct>2%` como um ajuste de capacidade, usando a evidência contrafactual nas execuções cobertas por placar. O ajuste é um **proxy**: usa exposição observada (Back=stake, Lay=liability) para estimar redução de N/turnover e mudança de ROI._

- Fonte OOS (curvas por banca): `logs/daily_reports_smoke_2r/20260729/wf_bank_sensitivity.json` (existe=não; sens_ok=não).

_Aviso: não foi possível aplicar o ajuste na sensibilidade por banca porque o export `wf_bank_sensitivity.json` está ausente/vazio/ilegível. Isso não afeta o OOS em si; apenas impede esta tabela ajustada. Se persistir, verifique se o daily está rodando a versão mais recente do `analyze_contexto_operacao_b808_robust_report.py` com `--wf-export-bank-sensitivity-json` habilitado._

