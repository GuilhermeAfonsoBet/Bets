# H3BUP Daily P0 Design — 2026-07-30

## Objectivo
Correcções P0 mensuráveis nos Dailies V1 (oficial) e V2 (PREVIEW), sem reescrita integral e sem alterar policy/stake/execução.

## Invariantes
- V1 oficial @ 22:00 UTC; V2 PREVIEW @ 22:10 UTC
- `H3BUP_DAILY_V2_OFFICIAL=0`; Telegram preview via `H3BUP_DAILY_V2_TELEGRAM_PREVIEW`
- Reporting-only; fail-open; sem impacto em executor/bridge/audit/accounting/CLV worker

## P0 map
| ID | Tema | Módulo |
|----|------|--------|
| P0-1 | Coorte/cutoff paridade | `cutoff.py`, `canonical.py`, render §11 |
| P0-2 | Resumo H3BUP isolado V1 | `v1_h3bup_summary.py`, `daily_full_report.py` |
| P0-3 | Funil operacional | `e2e_funnel.py` |
| P0-4 | E2E N/coverage/p50/p95 | `e2e_funnel.py` |
| P0-5 | CLV completo | `clv_section.py` |
| P0-6 | Health 4D | `health_model.py` |
| P0-7 | Config ≠ STALE por mtime | `evaluate_config_file`, `extract.py` |
| P0-8 | Alertas derivados | `derive_alerts` |
| P0-9 | Stake/void/ROI | `performance.py` (`roi_resolved`) |
| P0-10 | ROIw V1 fora do resumo | `render.py` |
| P0-11 | Diff vs V2 anterior | `diff_previous.py` |
| P0-12 | Formatação PDF/MD | `formatters.py` |
| P0-13 | V1 sem recomendações de risco | `daily_full_report.py` |

## Métrica principal
`principal_metric = roi_resolved = pnl_resolved / stake_resolved_total` (void no denominador).
