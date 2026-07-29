# H3BUP Fase 2C — Design CLV Forward Collection (2026-07-29)

## Objectivo
Criar obligations POST_5M / POST_15M / CLOSING por cada LIVE_OK H3BUP (forward-only), worker assíncrono, matching same-line strict, CLV raw B808-compatible, sem betslip e sem fair edge.

## Fontes
1. **best_odds_history** (SOURCE 1) — alimentado por `betinasia-collector` / `continuous_collector`
2. **passive_collector** (SOURCE 2) — cópia passiva de BOH só para obligations activas (sem requests externos)
3. **MISSING** — failure_reason explícito
4. Betslip **proibido** (`H3BUP_CLV_ALLOW_BETSLIP_SOURCE=0`)

## Fórmula CLV (confirmada B808 / update_hypothesis_results)
```
clv_raw_decimal = entry_odd / snapshot_odd - 1
clv_raw_pct = clv_raw_decimal * 100
# equivalente: (entry - snapshot) / snapshot * 100
```
Sinal Back: **positivo** quando entry > snapshot (melhor preço Back).

## Odd de entrada
1. `sent.price` 2. `odd_final` 3. `odd_at_decision`

## Storage
JSONL primário (`logs/h3bup_clv_obligations.jsonl`, `logs/h3bup_clv_snapshots.jsonl`) + DDL opcional Postgres `h3bup_clv_*`.
Unique key: `order_id|window_name|schema_version`

## Hook
`executor/store.py` após persist LIVE_OK → `enqueue_live_ok_payload` (thread daemon, fail-open).

## Flags
OFF por defeito; activar via systemd drop-in.
