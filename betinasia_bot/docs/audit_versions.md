# Versões de Auditoria - BetinAsia Bot

Este documento descreve as versões do script de auditoria de betslip e suas características.

## v1.0 (2026-02-08)

**Descrição**: Primeira versão de auditoria com tracking completo de lag times.

### Características:
- Auditoria H3B (reversões temporais)
- Filtro de direção: up, down, all
- Filtro de linhas extremas: |AH| > 5 são ignoradas
- Suporte a aliases de times (Wolves, Athletic Bilbao, etc)

### Campos capturados:
- Odds: websocket_odd, betslip_odd, difference_pct, difference_absolute
- Timing: hypothesis_detected_at, lag_detection_to_click_ms, lag_click_to_betslip_ms, audit_total_duration_ms
- Mercado: market_type, market_period, line, side, bet_description
- Jogo: home_team, away_team, match_info, sport
- Status: IDENTICAL, OK, MINOR_DIFF, MAJOR_DIFF, LINE_NOT_AVAILABLE, GAME_NOT_FOUND, ERROR

### Regra is_valid_opportunity:
```
is_valid = betslip_odd IS NOT NULL AND abs(difference_pct) < 2.0
```

### Limitações:
- Apenas AH Full-Time
- Não suporta AH Half-Time
- Não suporta OU (Over/Under)
- Auditoria não é em paralelo ao scraping

---

## Próximas versões planejadas

### v1.1 (planejado)
- Adicionar suporte a AH Half-Time
- Adicionar suporte a OU Full-Time
- Adicionar suporte a OU Half-Time

### v2.0 (planejado)
- Auditoria em paralelo ao scraping
- Redução do lag time
- Execução de ordens de aposta

---

## Merge com outras tabelas

Para fazer merge dos resultados de auditoria com as tabelas de hipóteses:

```sql
-- Merge H3B audit com h3b_temporal_reversal_events
SELECT 
    a.*,
    h.clv_pct as hypothesis_clv,
    h.bet_result
FROM betslip_audit_results a
LEFT JOIN h3b_temporal_reversal_events h 
    ON a.hypothesis_event_id = h.id
WHERE a.hypothesis_type = 'H3B';
```

**Importante**: O campo `hypothesis_event_id` precisa ser preenchido durante a auditoria para permitir o merge.
