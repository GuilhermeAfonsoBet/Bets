# Deploy evidence — Daily P0 20260730

- Method: SCP of reporting modules only (no executor/bridge/audit restart)
- Tests VPS: 53 passed
- V2 smoke: 2026-07-29 run a053b4a189ca TELEGRAM SENT 78212
- V1: remained official; PDF copied from logs/daily_reports/20260729/report_daily.pdf
- Flags: H3BUP_DAILY_V2_OFFICIAL=0 TELEGRAM_PREVIEW=1
- Rollback: restore ops/daily_v2 + daily_full_report.py; TELEGRAM_PREVIEW=0 stops preview only
