# DAILY V2 — PREVIEW / NÃO OFICIAL

> Este relatório está em validação shadow. Não substitui o Daily V1 oficial das 22:00 UTC.

# H3BUP Daily V2 — 2026-07-31
## 0) Manifesto
- status: `SHADOW / PREVIEW / NÃO OFICIAL`
- report_type: `DAILY_CLOSED`
- report_date_utc: `2026-07-31`
- window: `31/07/2026 00:00 UTC` → `01/08/2026 00:00 UTC`
- report_cutoff_utc: `01/08/2026 01:38 UTC`
- v1_report_cutoff_utc: `—`
- v2_comparison_cutoff_utc: `—`
- parity_status: `—`
- generated_at_utc: `01/08/2026 01:38 UTC`
- schema_version: `2`
- run_id: `friendlydaily01`
- git_commit: `d4a3d42d501cd44597401376de3f599d8cde9038`
- policy: `H3BUP_vNext` / `H3BUP_vNext_20260629`
- REPORT_HEALTH: `HEALTHY`
- OPERATIONS_HEALTH: `HEALTHY`
- DATA_QUALITY: `HEALTHY`
- STATISTICAL_READINESS: `INSUFFICIENT_N`

## 1) Resumo executivo
- LIVE_OK: 31 [AVAILABLE]
- maturity: `PARTIALLY_SETTLED`
- open / settled decided / void / missing: `15` / `15` / `1` / `0`
- stake placed: `US$ 310,00`
- stake resolved total: `US$ 160,00`
- stake void: `US$ 10,00`
- P&L resolved: `-US$ 28,70`
- ROI principal (`roi_resolved`): -17.94% [PARTIAL]
- ROI decided ex-void: -19.13% [PARTIAL]
- ROIw Total v2 (complementar): -17.94% [PARTIAL]
- ROIw Total v1: ver apêndice de paridade V1×V2 (fora do resumo principal).
- Nenhuma conclusão de edge sem evidência estatística suficiente.

## 2) Health (4 dimensões)
| Dimensão | Status |
|---|---|
| REPORT_HEALTH | `HEALTHY` |
| OPERATIONS_HEALTH | `HEALTHY` |
| DATA_QUALITY | `HEALTHY` |
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
| executor_live | `HEALTHY` | `01/08/2026 01:38 UTC` | 42s |
| accounting_health | `HEALTHY` | `01/08/2026 01:36 UTC` | 2m29s |
| accounting_daily_report | `HEALTHY` | `31/07/2026 22:02 UTC` | 3h36m |
| e2e_trace | `HEALTHY` | `01/08/2026 01:37 UTC` | 1m42s |
| clv_health | `HEALTHY` | `01/08/2026 01:38 UTC` | 14s |
| clv_obligations | `HEALTHY` | `01/08/2026 01:35 UTC` | 3m16s |
| policy_current | `CURRENT_UNCHANGED` | `29/06/2026 13:15 UTC` | 780h23m |
| risk_params | `CURRENT_UNCHANGED` | `20/07/2026 15:16 UTC` | 274h22m |
| accounting_balance | `HEALTHY` | `01/08/2026 01:36 UTC` | 2m32s |
| accounting_open_stakes | `HEALTHY` | `01/08/2026 01:36 UTC` | 2m29s |

## 5) Funil operacional
- Universo temporal: `all_traces_available`

| Etapa | N | % etapa anterior | % inicial | Status |
|---|---:|---:|---:|---|
| WS received | 11079 | — | 100.0% | `AVAILABLE` |
| detected | 11079 | 100.0% | 100.0% | `AVAILABLE` |
| audit persisted | 11079 | 100.0% | 100.0% | `AVAILABLE` |
| bridge fetched | 5949 | 53.7% | 53.7% | `AVAILABLE` |
| policy evaluated | 0 | 0.0% | 0.0% | `AVAILABLE` |
| execution request created | 530 | 8.9% | 4.8% | `AVAILABLE` |
| executor received | 530 | 100.0% | 4.8% | `AVAILABLE` |
| dry-run started | 0 | 0.0% | 0.0% | `AVAILABLE` |
| dry-run finished | 528 | 99.6% | 4.8% | `AVAILABLE` |
| final gate decided | 443 | 83.9% | 4.0% | `AVAILABLE` |
| place started | 0 | 0.0% | 0.0% | `AVAILABLE` |
| place finished | 61 | 13.8% | 0.6% | `AVAILABLE` |
| LIVE_OK | 61 | 100.0% | 0.6% | `AVAILABLE` |

### Reasons / bloqueios
| Reason/status | N | % requests |
|---|---:|---:|
| CAP_BLOCKED | 382 | 72.1% |
| API_FAILED | 82 | 15.5% |
| NO_SESSION | 0 | 0.0% |
| STALE | 3 | 0.6% |
| LIVE_PRECHECK_FAILED | 0 | 0.0% |
| LIVE_PLACE_FAILED | 0 | 0.0% |
| UNKNOWN | 10551 | 1990.8% |

### Buckets de velocidade
- DAILY_FAST_LE_6S: `31` de operações com pre_submit ≤ 6s
- STUDY_FAST_LT_4S (exploratório): `30` com pre_submit < 4s
- PRE_SUBMIT_MS_NA: N=`0` · coverage missing=`0 [AVAILABLE]`

## 6) Settlement e performance
| Métrica | Valor |
|---|---|
| stake_placed | US$ 310,00 |
| stake_resolved_total | US$ 160,00 |
| stake_decided_ex_void | US$ 150,00 |
| stake_void | US$ 10,00 |
| stake_open | US$ 150,00 |
| pnl_resolved | -US$ 28,70 |
| pnl_decided_ex_void | -US$ 28,70 |
| roi_resolved (principal) | -17.94% [PARTIAL] |
| roi_decided_ex_void | -19.13% [PARTIAL] |
| ROIw Total v2 | -17.94% [PARTIAL] |
| maturity | `PARTIALLY_SETTLED` |

> Fórmulas: `roi_resolved = pnl_resolved / stake_resolved_total` (void no denominador); `roi_decided_ex_void = pnl_decided_ex_void / stake_decided_ex_void`.

### Friendly vs Non-Friendly (diagnóstico / shadow)

> Diagnóstico shadow · `classification_version=FRIENDLY_CLASS_V1_20260731` · **não é filtro operacional** · status=`AVAILABLE`

- coverage classificação: 100.0% (F=15 · NF=16 · U=0 · C=0)

| Classe | N | Open | Settled | Void | Stake placed | Stake resolved | P&L resolved | ROI resolved | Maturity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FRIENDLY | 15 | 1 | 13 | 1 | US$ 150,00 | US$ 140,00 | -US$ 28,08 | -20.06% | `PARTIALLY_SETTLED` |
| NON_FRIENDLY | 16 | 14 | 2 | 0 | US$ 160,00 | US$ 20,00 | -US$ 0,62 | -3.10% | `PARTIALLY_SETTLED` |
| UNCLASSIFIED | 0 | 0 | 0 | 0 | US$ 0,00 | US$ 0,00 | US$ 0,00 | — | `FULLY_SETTLED` |
| CONFLICT | 0 | 0 | 0 | 0 | US$ 0,00 | US$ 0,00 | US$ 0,00 | — | `FULLY_SETTLED` |

> `roi_resolved = pnl_resolved / stake_resolved_total` (void no denominador). Comparar classes só com maturity/coverage visíveis.

## 7) Qualidade de preço / CLV forward
- collection: `HEALTHY` started `29/07/2026 20:07 UTC`
- source priority: `['best_odds_history', 'passive_collector']`
- collector: `ENABLED`
- fair edge: `NOT_IMPLEMENTED`

### Funil CLV (diagnóstico)
- live_ok_after_activation: `51`
- obligations_expected: `153`
- obligations_created: `153`
- source_missing: `33`
- line_mismatch: `489`
- side_mismatch: `0`
- period_mismatch: `0`
- kickoff_missing: `0`
- kickoff_conflict: `0`
- snapshot_after_kickoff: `0`
- snapshot_too_far: `0`
- retry_backlog: `18`

### Cobertura por janela (VALID_STRICT)
| Janela | Expected | Due | Attempted | Strict valid | Coverage |
|---|---:|---:|---:|---:|---:|
| POST_5M | 51 | 51 | 50 | 21 | 41.2% |
| POST_15M | 51 | 51 | 45 | 20 | 39.2% |
| CLOSING | 51 | 51 | 34 | 12 | 23.5% |

### Performance CLV (VALID_STRICT)
| Janela | N | CLV médio | Mediana | Positivo % | Status |
|---|---:|---:|---:|---:|---|
| POST_5M | 21 | -0.54% | -0.48% | 23.81% | `INSUFFICIENT_N` |
| POST_15M | 20 | -0.91% | -0.44% | 25.00% | `INSUFFICIENT_N` |
| CLOSING | 12 | -2.82% | -1.68% | 25.00% | `INSUFFICIENT_N` |

## 8) Latência E2E
- traces totais: `11079`
- traces LIVE_OK: `61`
- full-trace coverage: `0.55%`
- etapa dominante: `place_duration`
- ordering violations: `542`
- clock skew: `542`
- detect→audit overhead: 13.5 ms [WATCH]

| Métrica | N | Coverage | Mediana | p95 | Status |
|---|---:|---:|---:|---:|---|
| WS→detect | 11079 | 100.0% | 0.1 ms | 0.2 ms | `AVAILABLE` |
| detect→audit | 11079 | 100.0% | 13.5 ms | 139.5 ms | `AVAILABLE` |
| audit→bridge | 5949 | 53.7% | 168.6 ms | 312.7 ms | `AVAILABLE` |
| bridge→request | 530 | 4.8% | 3 175.1 ms | 3 346.6 ms | `AVAILABLE` |
| request→executor | 530 | 4.8% | — | — | `MISSING` |
| executor→dry-run | 530 | 4.8% | 0.5 ms | 4.0 ms | `AVAILABLE` |
| dry-run duration | 528 | 4.8% | 1 299.4 ms | 4 662.1 ms | `AVAILABLE` |
| dry-run→gate | 443 | 4.0% | 1 202.4 ms | 1 249.2 ms | `AVAILABLE` |
| gate→place | 61 | 0.6% | 0.1 ms | 0.2 ms | `AVAILABLE` |
| place duration | 61 | 0.6% | 3 946.7 ms | 7 411.4 ms | `AVAILABLE` |
| WS→LIVE_OK | 61 | 0.6% | 8 932.5 ms | 13 567.6 ms | `AVAILABLE` |

## 9) Excepções e alertas
| alert_id | severity | status | message |
|---|---|---|---|
| `CLV_INSUFFICIENT_N` | `INFO` | `OPEN` | N estatístico insuficiente para inferência |
| `E2E_OVERHEAD_WATCH` | `WATCH` | `OPEN` | Overhead detect→audit acima da baseline / em observação |
| `SETTLEMENT_PARTIAL` | `INFO` | `OPEN` | Coorte parcialmente liquidada |
| `CLV_BACKLOG` | `WATCH` | `OPEN` | Retry backlog CLV > 0 |
| `CLV_SOURCE_MISSING` | `WARNING` | `OPEN` | Snapshots CLV com source missing |
| `TRACE_ORDERING_VIOLATIONS` | `WATCH` | `OPEN` | Ordering violations no E2E |
| `TRACE_CLOCK_SKEW` | `WATCH` | `OPEN` | Clock skew / negative durations no E2E |

## 10) Mudanças versus V2 anterior
- sem snapshot V2 anterior comparável para esta coorte.

## 11) Paridade V1 × V2
| Campo | V1 | V2 | Status |
|---|---|---|---|
| report_date | `2026-07-31` | `2026-07-31` | `—` |
| cohort start | `31/07/2026 00:00 UTC` | `31/07/2026 00:00 UTC` | — |
| cohort end | `01/08/2026 00:00 UTC` | `01/08/2026 00:00 UTC` | — |
| parity cutoff | `—` | `—` | `None` |
| policy | `H3BUP_vNext` | `H3BUP_vNext_20260629` | — |
| LIVE_OK universe | `—` | 31 [AVAILABLE] | — |

### Apêndice métricas legado (paridade)
| Métrica | Fórmula | Universo | Inclui open? | Uso |
|---|---|---|---|---|
| ROI principal | pnl_resolved/stake_resolved_total | settled+void | não | oficial V2 |
| ROIw v2 | settled-aware % | settled+void | não | complementar |
| ROIw v1 | -17.94% [AVAILABLE] | ledger join | potencialmente | paridade legado |

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
- **friendly_breakdown**: shadow diagnostic FRIENDLY_CLASS_V1 — not an operational filter; UNCLASSIFIED != NON_FRIENDLY

---

**DAILY V2 — PREVIEW / NÃO OFICIAL** — Uso: validação técnica e metodológica. Não utilizar este preview como substituto do relatório oficial.
