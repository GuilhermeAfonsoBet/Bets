# Daily V2 — evidência de deploy shadow — 20260729

## Flags efectivas
- `H3BUP_DAILY_V2_ENABLED=1`
- `H3BUP_DAILY_V2_PUBLISH=0`
- `H3BUP_DAILY_V2_COMPARE_V1=1`
- `H3BUP_DAILY_V2_FAIL_OPEN=1`

## VPS
- Código: `/home/betbot/Bets/betinasia_bot/ops/daily_v2/`
- Timer shadow: `betinasia-daily-v2-shadow.timer` @ **22:10 UTC** (após V1 22:00)
- Service: `betinasia-daily-v2-shadow.service` (SuccessExitStatus=0 1)
- V1 permanece: `betinasia-daily-full-report.timer` @ 22:00 UTC → Telegram

## Hotfix V1 (P0)
- Secções H3BUP injectadas usavam `out_lines` dentro de `run_daily_full` (buffer correcto: `s0`)
- Corrigido para `s0.append` em `daily_full_report.py` + patch scripts
- Deployed na VPS antes da corrida 22:00 UTC de 2026-07-29

## Shadow run (exemplo)
- Cohort `DAILY_CLOSED` `2026-07-28` run_id `3adbd07716f9`
- LIVE_OK H3BUP=3; stake=30; ROI settled=-1.43% AVAILABLE
- fair_edge=`NOT_IMPLEMENTED`
- published=`false`
- elapsed ≈ 5.1s
- Outputs: `logs/daily_v2/h3bup_daily_{snapshot,report,health,exceptions}_20260728_3adbd07716f9.*`
- LKG + latest symlinks sob `logs/daily_v2/`

## Testes
- `tests/test_h3bup_daily_v2.py` → **23 passed**
- Log: `logs/h3bup_daily_v2_tests_20260729.txt`

## Decisão de publicação
- **NÃO publicado** (`PUBLISH=0`)
- Motivo: paridade V1/V2 ainda com divergências de janela esperadas e explicadas; gates de publicação não todos verdes
- Rollback: desligar timer shadow ou `H3BUP_DAILY_V2_PUBLISH=0` (já é o default)

## Impacto operacional
- Execução LIVE: não afectada
- Policy/stake: não alterados
- Ordens/betslips: nenhum criado

## Continuação 2026-07-29T21:15Z

### Smoke-test Daily V1 (pré-22:00)
- Corrida segura: `DAILY_SKIP_ACCOUNTING=1` `DAILY_REPORT_TELEGRAM=0` → `logs/daily_reports_smoke_2r/20260729/`
- Resultado: **DAILY_OK** (exit 0), PDF gerado
- Secções presentes no markdown:
  - `## Accounting Health — H3BUP`
  - `## H3BUP End-to-End Latency`
  - `## H3BUP CLV Forward Collection`
- Sem `NameError` / sem referência a `out_lines`
- Conclusão: hotfix `s0.append` validado end-to-end antes do timer oficial

### Replay matrix (V2)
Ver `logs/h3bup_daily_v2_replay_matrix_20260729.json`.

| Dia | LIVE_OK H3BUP_vNext | Nota |
|---|---:|---|
| 2026-07-20 | 0 (12 se sem filtro) | Policy `bridge_h3b_live_v0`, stake **20** — leak legado; excluído do universo H3BUP_vNext |
| 2026-07-28 | 3 | Pós capacity-fix; stake 10; ROI settled -1.43% |
| 2026-07-29 | 22 (parcial/intraday até cutoff) | 6 open → ROI PARTIAL; todos DAILY_FAST_LE_6S |

### Testes
- `tests/test_h3bup_daily_v2.py` → **28 passed** (antes 23)

### Publicação
- Continua **OFF** (`PUBLISH=0`)
- V1 permanece oficial às 22:00 UTC
