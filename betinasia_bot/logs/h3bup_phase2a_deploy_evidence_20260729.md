# Fase 2A — Deploy evidence (2026-07-29)

## Antes
- Último balance CSV monitor útil: `20260728_131617__balance.csv` (13:17Z)
- Último CSV via daily: `20260728_220133__balance.csv` / `20260728_220220__open_stakes.csv`
- Monitor logava `files=None` como `snapshot ok`
- Bridge/executor md5: `3ccdc438...` / `298f8073...`

## Acções de deploy
1. Backup em `/home/betbot/Bets/betinasia_bot/backups/accounting_phase2a_*`
2. Deploy módulos `ops/accounting_*` + reconcile/health
3. Patch scraper login SPA + save_session guard
4. Drop-ins:
   - `betinasia-accounting-monitor.service.d/90-phase2a-health.conf`
   - `betinasia-accounting-daily.service.d/90-phase2a-health.conf`
5. Patch daily_full_report secção Health H3BUP
6. Smoke `--once` → `ACCOUNTING_OK`
7. `systemctl start betinasia-accounting-monitor` apenas

## Depois
- Snapshot fresco: balance `20260729_140752__balance.csv` (5444 rows), open `20260729_140816__open_stakes.csv` (69 rows)
- Health: `ACCOUNTING_OK` / `HEALTHY` (`checked_at_utc=2026-07-29T14:08:16Z`)
- Monitor ActiveEnter: `2026-07-29 14:06:55 UTC`
- Executor PID/ActiveEnter inalterado nesta fase de deploy: `3682164` / `2026-07-29 13:30:07 UTC`
- Bridge PID/ActiveEnter inalterado: `3682166` / `2026-07-29 13:30:07 UTC`
- md5 bridge/worker inalterados

## Rollback
- Restaurar `ops/accounting_monitor.py` + `scraper/betinasia.py` a partir de `backups/accounting_phase2a_*`
- Remover drop-ins `90-phase2a-health.conf`
- `systemctl daemon-reload && systemctl restart betinasia-accounting-monitor`
- Não requer restart de executor/bridge

## Rollback validado?
Validado por existência de backups e diff conceptual; **não** foi executado rollback completo (serviço ficou healthy).
