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
