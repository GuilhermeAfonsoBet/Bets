# H3BUP Accounting Health — 20260729

**Corte:** 2026-07-29T12:31:52Z  
**READ-ONLY**

## Estado do serviço
- `betinasia-accounting-monitor.service`: **active/running** (PID observado; uptime ~1d5h)
- Último CSV balance/open: **2026-07-28T22:01/22:02Z**
- Defasagem vs agora: **~14.5h**
- `accounting_daily_report.json` mtime: 2026-07-28T22:02Z
- Timer daily: `betinasia-accounting-daily.timer` (próxima execução observada ~2026-07-29 22:01Z)

## Achado crítico
Logs recentes em `logs/accounting_monitor_error.log`:
```
[acct] snapshot ok dt=0.0s files={'balance': None, 'open_stakes': None}
```
repetido a cada ~5 minutos.

**Interpretação (FATO observado):** o monitor está em loop e reporta "snapshot ok", mas **não produz arquivos** balance/open_stakes. Isso explica a lacuna 2/8 (agora 2/12 liquidados no CSV antigo) melhor do que "apenas baixo N".

## Histórico relevante
- 2026-07-08: OOM kill do accounting-monitor (journal).
- 2026-07-28 06:45Z: restart do monitor (coincide com janela de restore/ops).
- Após 22:02Z 28/07: sem novos CSVs apesar do processo "active".

## Implicações para P&L H3BUP
- Não é seguro declarar ROI da estratégia com coverage atual.
- Ordens com kickoff futuro (~9) ainda não deveriam estar settled.
- 1 ordem com kickoff 2026-07-29T00:00Z está em `SETTLED_MISSING_ACCOUNTING` sob snapshot stale — provavelmente **ACCOUNTING_LAG** operacional.

## Recomendações (sem implementar nesta fase)
1. Diagnosticar por que snapshot retorna None (sessão Playwright/API/proxy) em fase posterior.
2. Não usar `balance_current` da conta como P&L H3BUP.
3. Recalcular settlement somente após snapshots válidos.
