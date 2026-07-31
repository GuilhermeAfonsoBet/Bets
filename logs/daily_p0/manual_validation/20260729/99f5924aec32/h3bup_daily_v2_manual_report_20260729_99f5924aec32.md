# DAILY V2 — MANUAL VALIDATION / PREVIEW / NÃO OFICIAL

> MANUAL VALIDATION — Execução manual para revisão. Não substitui o Daily V1 oficial nem o V2 automático.

# DAILY V2 — PREVIEW / NÃO OFICIAL

> Este relatório está em validação shadow. Não substitui o Daily V1 oficial das 22:00 UTC.

# H3BUP Daily V2 — 2026-07-29
## 0) Manifesto
- status: `SHADOW / PREVIEW / NÃO OFICIAL`
- report_type: `DAILY_CLOSED`
- report_date_utc: `2026-07-29`
- window: `29/07/2026 00:00 UTC` → `30/07/2026 00:00 UTC`
- report_cutoff_utc: `29/07/2026 22:01 UTC`
- v1_report_cutoff_utc: `29/07/2026 22:01 UTC`
- v2_comparison_cutoff_utc: `29/07/2026 22:01 UTC`
- parity_status: `CUTOFF_ALIGNED`
- generated_at_utc: `30/07/2026 19:47 UTC`
- schema_version: `2`
- run_id: `99f5924aec32`
- git_commit: `d4a3d42d501cd44597401376de3f599d8cde9038`
- policy: `H3BUP_vNext` / `H3BUP_vNext_20260629`
- REPORT_HEALTH: `HEALTHY`
- OPERATIONS_HEALTH: `WATCH`
- DATA_QUALITY: `WATCH`
- STATISTICAL_READINESS: `INSUFFICIENT_N`

## 1) Resumo executivo
- LIVE_OK: 24 [AVAILABLE]
- maturity: `PARTIALLY_SETTLED`
- open / settled decided / void / missing: `1` / `21` / `2` / `0`
- stake placed: `US$ 240,00`
- stake resolved total: `US$ 230,00`
- stake void: `US$ 20,00`
- P&L resolved: `US$ 6,60`
- ROI principal (`roi_resolved`): 2.87% [PARTIAL]
- ROI decided ex-void: 3.14% [PARTIAL]
- ROIw Total v2 (complementar): 2.87% [PARTIAL]
- ROIw Total v1: ver apêndice de paridade V1×V2 (fora do resumo principal).
- Nenhuma conclusão de edge sem evidência estatística suficiente.

## 2) Health (4 dimensões)
| Dimensão | Status |
|---|---|
| REPORT_HEALTH | `HEALTHY` |
| OPERATIONS_HEALTH | `WATCH` |
| DATA_QUALITY | `WATCH` |
| STATISTICAL_READINESS | `INSUFFICIENT_N` |

## 3) Configuração (fingerprint / drift)
| Config | File status | Runtime status | Fingerprint | Drift |
|---|---|---|---|---|
| policy | `OK` | `UNVERIFIED` | `8009c8fa4b67b75f` | `CURRENT_UNCHANGED` |
| risk_params | `OK` | `UNVERIFIED` | `6a38c6ddd893a215` | `CURRENT_UNCHANGED` |

> `CURRENT_UNCHANGED` não é warning. `CONFIG_DRIFT` é CRITICAL.

## 4) Data health (fontes)
| Fonte | Status | Cutoff | Age |
|---|---|---|---|
| executor_live | `HEALTHY` | `30/07/2026 19:46 UTC` | 43s |
| accounting_health | `HEALTHY` | `30/07/2026 19:42 UTC` | 4m54s |
| accounting_daily_report | `WATCH` | `29/07/2026 22:02 UTC` | 21h44m |
| e2e_trace | `HEALTHY` | `30/07/2026 19:47 UTC` | 13s |
| clv_health | `WATCH` | `30/07/2026 19:47 UTC` | 22s |
| clv_obligations | `HEALTHY` | `30/07/2026 19:45 UTC` | 1m53s |
| policy_current | `CURRENT_UNCHANGED` | `29/06/2026 13:15 UTC` | 750h32m |
| risk_params | `CURRENT_UNCHANGED` | `20/07/2026 15:16 UTC` | 244h31m |
| accounting_balance | `HEALTHY` | `30/07/2026 19:42 UTC` | 4m58s |
| accounting_open_stakes | `HEALTHY` | `30/07/2026 19:42 UTC` | 4m54s |

## 5) Funil operacional
- Universo temporal: `all_traces_available`

| Etapa | N | % etapa anterior | % inicial | Status |
|---|---:|---:|---:|---|
| WS received | 5549 | — | 100.0% | `AVAILABLE` |
| detected | 5549 | 100.0% | 100.0% | `AVAILABLE` |
| audit persisted | 5549 | 100.0% | 100.0% | `AVAILABLE` |
| bridge fetched | 2670 | 48.1% | 48.1% | `AVAILABLE` |
| policy evaluated | 0 | 0.0% | 0.0% | `AVAILABLE` |
| execution request created | 186 | 7.0% | 3.4% | `AVAILABLE` |
| executor received | 186 | 100.0% | 3.4% | `AVAILABLE` |
| dry-run started | 0 | 0.0% | 0.0% | `AVAILABLE` |
| dry-run finished | 185 | 99.5% | 3.3% | `AVAILABLE` |
| final gate decided | 145 | 78.4% | 2.6% | `AVAILABLE` |
| place started | 0 | 0.0% | 0.0% | `AVAILABLE` |
| place finished | 23 | 15.9% | 0.4% | `AVAILABLE` |
| LIVE_OK | 23 | 100.0% | 0.4% | `AVAILABLE` |

### Reasons / bloqueios
| Reason/status | N | % requests |
|---|---:|---:|
| CAP_BLOCKED | 122 | 65.6% |
| API_FAILED | 37 | 19.9% |
| NO_SESSION | 0 | 0.0% |
| STALE | 3 | 1.6% |
| LIVE_PRECHECK_FAILED | 0 | 0.0% |
| LIVE_PLACE_FAILED | 0 | 0.0% |
| UNKNOWN | 5364 | 2883.9% |

### Buckets de velocidade
- DAILY_FAST_LE_6S: `24` de operações com pre_submit ≤ 6s
- STUDY_FAST_LT_4S (exploratório): `23` com pre_submit < 4s
- PRE_SUBMIT_MS_NA: N=`0` · coverage missing=`0 [AVAILABLE]`

## 6) Settlement e performance
| Métrica | Valor |
|---|---|
| stake_placed | US$ 240,00 |
| stake_resolved_total | US$ 230,00 |
| stake_decided_ex_void | US$ 210,00 |
| stake_void | US$ 20,00 |
| stake_open | US$ 10,00 |
| pnl_resolved | US$ 6,60 |
| pnl_decided_ex_void | US$ 6,60 |
| roi_resolved (principal) | 2.87% [PARTIAL] |
| roi_decided_ex_void | 3.14% [PARTIAL] |
| ROIw Total v2 | 2.87% [PARTIAL] |
| maturity | `PARTIALLY_SETTLED` |

> Fórmulas: `roi_resolved = pnl_resolved / stake_resolved_total` (void no denominador); `roi_decided_ex_void = pnl_decided_ex_void / stake_decided_ex_void`.

## 7) Qualidade de preço / CLV forward
- collection: `WATCH` started `29/07/2026 20:07 UTC`
- source priority: `['best_odds_history', 'passive_collector']`
- collector: `ENABLED`
- fair edge: `NOT_IMPLEMENTED`

### Funil CLV (diagnóstico)
- live_ok_after_activation: `13`
- obligations_expected: `39`
- obligations_created: `39`
- source_missing: `0`
- line_mismatch: `0`
- side_mismatch: `0`
- period_mismatch: `0`
- kickoff_missing: `0`
- kickoff_conflict: `0`
- snapshot_after_kickoff: `0`
- snapshot_too_far: `0`
- retry_backlog: `2`

### Cobertura por janela (VALID_STRICT)
| Janela | Expected | Due | Attempted | Strict valid | Coverage |
|---|---:|---:|---:|---:|---:|
| POST_5M | 13 | 13 | 13 | 7 | 53.8% |
| POST_15M | 13 | 13 | 12 | 6 | 46.2% |
| CLOSING | 13 | 13 | 11 | 8 | 61.5% |

### Performance CLV (VALID_STRICT)
| Janela | N | CLV médio | Mediana | Positivo % | Status |
|---|---:|---:|---:|---:|---|
| POST_5M | 7 | -1.78% | -0.64% | 28.57% | `INSUFFICIENT_N` |
| POST_15M | 6 | -2.32% | -0.43% | 33.33% | `INSUFFICIENT_N` |
| CLOSING | 8 | -2.82% | -1.68% | 25.00% | `INSUFFICIENT_N` |

## 8) Latência E2E
- traces totais: `5549`
- traces LIVE_OK: `23`
- full-trace coverage: `0.41%`
- etapa dominante: `place_duration`
- ordering violations: `187`
- clock skew: `187`
- detect→audit overhead: 14.1 ms [WATCH]

| Métrica | N | Coverage | Mediana | p95 | Status |
|---|---:|---:|---:|---:|---|
| WS→detect | 5549 | 100.0% | 0.1 ms | 0.2 ms | `AVAILABLE` |
| detect→audit | 5549 | 100.0% | 14.1 ms | 145.3 ms | `AVAILABLE` |
| audit→bridge | 2670 | 48.1% | 173.8 ms | 405.1 ms | `AVAILABLE` |
| bridge→request | 186 | 3.4% | 3 187.5 ms | 3 419.3 ms | `AVAILABLE` |
| request→executor | 186 | 3.4% | — | — | `MISSING` |
| executor→dry-run | 186 | 3.4% | 0.6 ms | 6.7 ms | `AVAILABLE` |
| dry-run duration | 185 | 3.3% | 1 365.4 ms | 9 582.6 ms | `AVAILABLE` |
| dry-run→gate | 145 | 2.6% | 1 202.5 ms | 1 249.8 ms | `AVAILABLE` |
| gate→place | 23 | 0.4% | 0.1 ms | 40.9 ms | `INSUFFICIENT_N` |
| place duration | 23 | 0.4% | 3 998.1 ms | 7 172.1 ms | `INSUFFICIENT_N` |
| WS→LIVE_OK | 23 | 0.4% | 8 686.2 ms | 13 384.2 ms | `INSUFFICIENT_N` |

## 9) Excepções e alertas
| alert_id | severity | status | message |
|---|---|---|---|
| `CLV_INSUFFICIENT_N` | `INFO` | `OPEN` | N estatístico insuficiente para inferência |
| `E2E_OVERHEAD_WATCH` | `WATCH` | `OPEN` | Overhead detect→audit acima da baseline / em observação |
| `SETTLEMENT_PARTIAL` | `INFO` | `OPEN` | Coorte parcialmente liquidada |
| `CLV_BACKLOG` | `WATCH` | `OPEN` | Retry backlog CLV > 0 |
| `TRACE_ORDERING_VIOLATIONS` | `WATCH` | `OPEN` | Ordering violations no E2E |
| `TRACE_CLOCK_SKEW` | `WATCH` | `OPEN` | Clock skew / negative durations no E2E |

## 10) Mudanças versus V2 anterior
- previous_run_id: `c80914813bbe` → current `99f5924aec32`

| Métrica | Anterior | Atual | Delta |
|---|---:|---:|---:|
| LIVE_OK | 22 | 24 | 2.0 |
| open | 6 | 1 | -5.0 |
| settled | 15 | 21 | 6.0 |
| void | 1 | 2 | 1.0 |
| missing | 0 | 0 | 0.0 |
| stake resolved | None | 230.0 | None |
| P&L resolved | None | 6.600000000000001 | None |
| ROI resolved | -0.0406875 | 0.02869565217391305 | 0.06938315217391305 |
| POST_5M valid | None | 7 | None |
| POST_15M valid | None | 6 | None |
| CLOSING valid | None | 8 | None |
| CLV backlog | None | 2 | None |
| alertas ativos | 0 | 6 | 6.0 |

- novos alertas: `['CLV_BACKLOG', 'CLV_INSUFFICIENT_N', 'E2E_OVERHEAD_WATCH', 'SETTLEMENT_PARTIAL', 'TRACE_CLOCK_SKEW', 'TRACE_ORDERING_VIOLATIONS']`
- alertas resolvidos: `[]`

## 11) Paridade V1 × V2
| Campo | V1 | V2 | Status |
|---|---|---|---|
| report_date | `2026-07-29` | `2026-07-29` | `CUTOFF_ALIGNED` |
| cohort start | `29/07/2026 00:00 UTC` | `29/07/2026 00:00 UTC` | — |
| cohort end | `30/07/2026 00:00 UTC` | `30/07/2026 00:00 UTC` | — |
| parity cutoff | `29/07/2026 22:01 UTC` | `29/07/2026 22:01 UTC` | `CUTOFF_ALIGNED` |
| policy | `H3BUP_vNext` | `H3BUP_vNext_20260629` | — |
| LIVE_OK universe | `—` | 24 [AVAILABLE] | — |

### Apêndice métricas legado (paridade)
| Métrica | Fórmula | Universo | Inclui open? | Uso |
|---|---|---|---|---|
| ROI principal | pnl_resolved/stake_resolved_total | settled+void | não | oficial V2 |
| ROIw v2 | settled-aware % | settled+void | não | complementar |
| ROIw v1 | 2.87% [AVAILABLE] | ledger join | potencialmente | paridade legado |

## 12) Metodologia e linhagem
- **cohort_timestamp**: created_at UTC
- **post_date_usage**: accounting freshness / settlement metadata only
- **daily_fast**: DAILY_FAST_LE_6S: pre_submit_ms <= 6000
- **study_fast**: STUDY_FAST_LT_4S: pre_submit_ms < 4000 (exploratory)
- **principal_metric**: roi_resolved = pnl_resolved / stake_resolved_total (void in denom)
- **roi_settled**: legacy alias of roi_resolved
- **roi_decided_ex_void**: pnl_decided_ex_void / stake_decided_ex_void
- **roiw_total_v1**: (sum pnl / sum exposure)*100; may include open if in ledger — appendix parity only
- **roiw_total_v2**: settled-aware percent complementary
- **absence_policy**: missing/stale/not_calculable must not appear as zero
- **fair_edge**: NOT_IMPLEMENTED
- **config_stale_policy**: static config uses fingerprint/drift, not mtime age

---

**DAILY V2 — PREVIEW / NÃO OFICIAL** — Uso: validação técnica e metodológica. Não utilizar este preview como substituto do relatório oficial.
