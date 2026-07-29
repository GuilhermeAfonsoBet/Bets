# H3BUP Daily V2 — 2026-07-28
## 0) Manifesto
- report_type: `DAILY_CLOSED`
- report_date_utc: `2026-07-28`
- window: `2026-07-28T00:00:00+00:00` → `2026-07-29T00:00:00+00:00`
- report_cutoff_utc: `2026-07-29T21:00:56.063225+00:00`
- generated_at_utc: `2026-07-29T21:00:56.063307+00:00`
- schema_version: `2`
- run_id: `3adbd07716f9`
- git_commit: `None`
- policy: `H3BUP_vNext` / `H3BUP_vNext_20260629`
- report_health: `HEALTHY`

## 1) Resumo executivo
- LIVE_OK: 3 [AVAILABLE]
- maturity: `FULLY_SETTLED`
- open/settled/void/missing: `0` / `2` / `1` / `0`
- ROI settled (principal): -1.43% [AVAILABLE]
- ROIw Total v1 (complementar): -1.43% [AVAILABLE]
- ROIw Total v2: -1.43% [AVAILABLE]
- Nenhuma conclusão de edge sem evidência estatística suficiente.

## 2) Policy e configuração efectiva
- policy_id: `H3BUP_vNext`
- policy_version: `H3BUP_vNext_20260629`
- stake alvo H3BUP: `10` USD
- odd band: `1.85–2.15`; capacity `dry.limit_final > 100`; slippage_pre_pct `< 0`

## 3) Data health
| Fonte | Status | Cutoff | Age(s) |
|---|---|---|---:|
| executor_live | `HEALTHY` | `2026-07-29T21:00:22.979953+00:00` | 33.083272 |
| accounting_health | `HEALTHY` | `2026-07-29T20:57:28.719775+00:00` | 207.34345 |
| accounting_daily_report | `WATCH` | `2026-07-28T22:02:42.108069+00:00` | 82693.955156 |
| e2e_trace | `HEALTHY` | `2026-07-29T21:00:55.989163+00:00` | 0.074062 |
| clv_health | `WATCH` | `2026-07-29T21:00:37.244342+00:00` | 18.818883 |
| clv_obligations | `HEALTHY` | `2026-07-29T20:54:03.279857+00:00` | 412.783368 |
| policy_current | `STALE` | `2026-06-29T13:15:03.295096+00:00` | 2619952.768129 |
| risk_params | `STALE` | `2026-07-20T15:16:08.267440+00:00` | 798287.795785 |
| accounting_balance | `HEALTHY` | `2026-07-29T20:57:24.490353+00:00` | 211.572872 |
| accounting_open_stakes | `HEALTHY` | `2026-07-29T20:57:28.692773+00:00` | 207.370452 |

## 4) Funil operacional
- LIVE_OK (coorte created_at UTC): 3 [AVAILABLE]
- DAILY_FAST_LE_6S: `{'n': 3, 'threshold_ms': 6000, 'op': '<='}`
- STUDY_FAST_LT_4S (exploratório): `{'n': 3, 'threshold_ms': 4000, 'op': '<', 'label': 'exploratory_only'}`
- PRE_SUBMIT_MS_NA: `{'n': 0}`

## 5) Settlement e performance
- stake placed: `30.0`
- stake settled: `30.0`
- pnl settled: `-0.4300000000000015`
- ROI settled: -1.43% [AVAILABLE]
- ROIw Total v1: -1.43% [AVAILABLE]
- ROIw Total v2: -1.43% [AVAILABLE]
- principal_metric: `roi_settled`

## 6) Qualidade de preço / CLV
- collection: `WATCH` started `2026-07-29T20:07:50+00:00`
- POST_5M strict: 2 [INSUFFICIENT_N]
- POST_15M strict: 2 [INSUFFICIENT_N]
- CLOSING strict: 0 [INSUFFICIENT_N]
- fair edge: `NOT_IMPLEMENTED`
- funnel: `{'live_ok_after_activation': 2, 'obligations_expected': 6, 'obligations_created': 6, 'source_missing': 0, 'kickoff_missing': 0}`

## 7) Latência E2E
- DAILY_FAST_LE_6S: 3 [AVAILABLE]
- STUDY_FAST_LT_4S: 3 [AVAILABLE]
- E2E WS→LIVE_OK: `AVAILABLE`
- detect→audit overhead: `WATCH`
- e2e_source_status: `HEALTHY`

## 8) Concentração
- `{'status': 'INSUFFICIENT_N', 'notes': ['only emitted when N sufficient']}`

## 9) Excepções e alertas
- nenhum

## 10) Mudanças vs relatório anterior
- _comparação incremental gerida pelo runner shadow / compare_v1_

## 11) Metodologia e linhagem
- **cohort_timestamp**: created_at UTC
- **post_date_usage**: accounting freshness / settlement metadata only
- **daily_fast**: DAILY_FAST_LE_6S: pre_submit_ms <= 6000
- **study_fast**: STUDY_FAST_LT_4S: pre_submit_ms < 4000 (exploratory)
- **roi_settled**: sum(pnl_confirmed_settled)/sum(stake_confirmed_settled)
- **roiw_total_v1**: (sum pnl / sum exposure)*100; may include open if in ledger
- **absence_policy**: missing/stale/not_calculable must not appear as zero
- **fair_edge**: NOT_IMPLEMENTED

