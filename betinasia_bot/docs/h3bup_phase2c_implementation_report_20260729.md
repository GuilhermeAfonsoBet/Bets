# H3BUP Fase 2C — Implementation Report (2026-07-29)

## Status CLV: CLV_COLLECTION_HEALTHY_INSUFFICIENT_N
## Status E2E soak: E2E_SOAK_SUFFICIENT
## Status geral: PHASE2C_COMPLETE_E2E_SOAK_SUFFICIENT

(CLV forward activo; N LIVE_OK após activação CLV = 0 no corte — WATCH/INSUFFICIENT_N para estatística CLV; soak 2B suficiente.)

## Implementado
- matching/fórmula/store/register/sources/worker/passive/export/Daily patch
- hook fail-open em `executor/store.py`
- systemd: `betinasia-h3bup-clv-worker`, `betinasia-h3bup-clv-passive`
- collection_started_at: 2026-07-29T20:07:50+00:00

## Health no corte
```json
{
  "checked_at_utc": "2026-07-29T20:09:04.179324+00:00",
  "status": "WATCH",
  "enabled": true,
  "collection_started_at_utc": "2026-07-29T20:07:50+00:00",
  "source_priority": [
    "best_odds_history",
    "passive_collector"
  ],
  "live_ok_after_activation": 0,
  "obligations_expected": 0,
  "obligations_created": 0,
  "post_5m_expected": 0,
  "post_5m_attempted": 0,
  "post_5m_valid_strict": 0,
  "post_15m_expected": 0,
  "post_15m_attempted": 0,
  "post_15m_valid_strict": 0,
  "closing_expected": 0,
  "closing_attempted": 0,
  "closing_valid_strict": 0,
  "source_missing": 0,
  "line_mismatch": 0,
  "side_mismatch": 0,
  "period_mismatch": 0,
  "kickoff_missing": 0,
  "kickoff_conflict": 0,
  "snapshot_after_kickoff": 0,
  "snapshot_too_far": 0,
  "retry_backlog": 0,
  "failed_final": 0,
  "worker_last_success_utc": "2026-07-29T20:09:04.179031+00:00",
  "worker_consecutive_failures": 0,
  "collector_status": "ENABLED",
  "error": null,
  "betslip_source_allowed": false,
  "fair_edge_enabled": false
}
```

## Não alterado
policy/stake/thresholds/audit/bridge/execution/accounting; sem fair edge; sem betslip CLV.
