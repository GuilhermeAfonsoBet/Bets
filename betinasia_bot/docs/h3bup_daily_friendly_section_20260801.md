# Daily — Friendly vs Non-Friendly (shadow)

Data: 2026-08-01

## O que mudou

A coorte diária H3BUP_vNext passa a mostrar uma tabela **Friendly / Non-Friendly /
Unclassified / Conflict** no:

- **Daily V2** (PREVIEW): secção 6 — Settlement e performance
- **Daily V1** resumo H3BUP: após CLV, bloco diagnóstico

## Regras

- Classificação: `FRIENDLY_CLASS_V1_20260731` (mesmo contrato da análise histórica)
- Enrichment: `logs/h3bup_friendly_league_map.csv` + SQL read-only opcional
- `UNCLASSIFIED` ≠ `NON_FRIENDLY`
- Label explícito: **diagnóstico / shadow · não é filtro operacional**
- `safety.alters_friendly_filter = false`

## O que NÃO muda

Policy, stake, odd band, capacity, slippage gate, executor, bridge, accounting,
CLV worker, timers, Telegram oficial, filtros de execução.

## Módulo

`ops/daily_v2/friendly_section.py` → `canonical.build_snapshot` → `render` / `v1_h3bup_summary`
