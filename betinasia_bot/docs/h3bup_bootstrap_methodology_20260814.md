# Metodologia — FASE 2E-A Bootstrap H3BUP_vNext

## Universo
- `policy_id` = `H3BUP_vNext`
- `policy_version` = `H3BUP_vNext_20260629`
- status execução = `LIVE_OK` (freeze Friendly)
- Janela: primeiro LIVE_OK até cutoff `2026-08-14T13:48:03Z`
- Deduplicação: `order_id`
- Resolvidas: `SETTLED_DECIDED` + `VOID_PUSH`
- Excluídas do P&L: `OPEN`, `MISSING`, `UNRECONCILED` (não preenchidas com zero)

## ROI
```
roi_resolved = sum(pnl SETTLED_DECIDED) / sum(stake SETTLED_DECIDED + VOID_PUSH)
```
**Void entra no denominador** (pnl void = 0).

## Bootstrap
- Replicações: `100000`
- Seed: `20260814`
- Order-level: amostragem com reposição de ordens resolvidas
- **Preferencial:** cluster por `event_id` (amostra eventos; inclui todas as ordens do evento)
- Dia UTC: cluster por data de `created_at_utc`
- Quantis empíricos da distribuição bootstrap

## Interpretação
- `P(ROI>0)` = fração das replicações bootstrap com ROI>0
- **Não** é automaticamente um p-value frequentista clássico
- Teste sign-flip por evento é **exploratório**

## Segurança
Análise read-only. Sem criação de ordens, betslips, nem alteração de policy/stake/executor/bridge/CLV/Telegram.

## Freeze
- run_id fonte: `7d6fa310a00b`
- orders CSV: `/workspace/betinasia_bot/logs/h3bup_friendly_analysis/20260814/7d6fa310a00b/h3bup_friendly_order_level_7d6fa310a00b.csv`
- generated: `2026-08-14T13:49:37Z`
