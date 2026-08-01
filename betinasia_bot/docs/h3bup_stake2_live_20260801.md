# H3BUP — stake live = 2 (2026-08-01)

## Mudança
Stake operacional LIVE H3BUP_vNext passa de **10** para **2**.

## Pontos de enforcement
1. `ops/executor_bridge_audit.py` — `req.policy.stake_requested = 2.0` + meta `H3BUP_vNext_force_stake_2`
2. `executor/worker.py` — force sizing + hard-cap Back Pre = `EXECUTOR_LIVE_STAKE` (default 2)
3. systemd / `.env` na VPS — `EXECUTOR_LIVE_STAKE=2`, `EXECUTOR_LIVE_MAX_STAKE=2`, `BRIDGE_STAKE=2`

## Daily
`EXPECTED_H3BUP["stake"] = 2.0`. Ordens legadas com stake 10 ou 20 geram `stake_mismatch` WARNING.

## Fora de âmbito
Sem alteração de filtros (odd/slippage/capacity), policy version ou Telegram oficial.
