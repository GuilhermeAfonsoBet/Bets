# Design futuro — CLV H3BUP (20260729)

**Não implementar nesta fase.**

```text
LIVE_OK H3BUP
→ register CLV obligation (order_id, audit_id, execution_id, policy_version, market/side/line, entry odds, kickoff)
→ post_5m snapshot (optional)
→ post_15m snapshot (optional)
→ closing snapshot (pre-kickoff last / at kickoff-ε)
→ same-line validation
→ opposite side (if needed for overround)
→ overround / de-vig (se definido)
→ fair edge
→ Daily section
```

## Campos mínimos da obrigação
order_id, execution_id, audit_id, policy_version, event_id, market, side, line,
entry_odd, odd_at_decision, odd_final, entry_ts, kickoff_ts, kickoff_source,
kickoff_confidence, stake.

## Reuso
- `get_closing_odd` + `BestOddsHistory` para closing offline (preferível: **sem betslip**).
- Evitar abrir betslip só para CLV enquanto accounting monitor estiver quebrado / sem daily cap.
- Se betslip for necessário: dry-run + cancel obrigatório + cap diário + kill switch.

## Riscos
- Misturar métricas DT/H3BUP
- Explosão de betslips
- Kickoff errado → closing após início
