# H3BUP_vNext — Análise Friendly vs Non-Friendly

- status: `NON_FRIENDLY_WORSE_PRELIMINARY`
- run_id: `78c9f53d95df`
- generated_at_utc: `2026-08-01T01:32:19.568125+00:00`
- cutoff_utc: `2026-08-01T01:31:39.129924+00:00`
- friendly_classification_version: `FRIENDLY_CLASS_V1_20260731`
- classification_checksum: `1a9706023c55df4244dd2d15a1593137aae073339d2e612728c85972c2df0d8e`

## Universo PRIMÁRIO (H3BUP_vNext exact)

```json
{
  "universe": "H3BUP_vNext_exact",
  "policy_id": "H3BUP_vNext",
  "policy_version": "H3BUP_vNext_20260629",
  "n": 74,
  "n_with_order_id": 74,
  "n_fallback_identity": 0,
  "excluded": {
    "not_live_ok": 195057,
    "legacy_policy": 5288,
    "wrong_policy_version": 0,
    "not_back": 0,
    "not_pre": 0,
    "before_start": 0,
    "after_cutoff": 0,
    "stake_20_legacy": 0,
    "dry_ok": 966,
    "heartbeat": 0,
    "duplicates_collapsed": 0
  },
  "cutoff_utc": "2026-08-01T01:31:39.129924+00:00",
  "policy_start_utc": "2026-06-29T00:00:00+00:00"
}
```

## Tabela principal de performance

| Métrica | Friendly | Non-Friendly | Unclassified | Conflict | Total |
|---|---:|---:|---:|---:|---:|
| LIVE_OK | 39 | 35 | 0 | 0 | 74 |
| eventos_unicos | 28 | 30 | 0 | 0 | 58 |
| stake_placed | 390.0000 | 350.0000 | 0 | 0 | 740.0000 |
| open | 1 | 17 | 0 | 0 | 18 |
| settled_decided | 35 | 17 | 0 | 0 | 52 |
| void_push | 3 | 1 | 0 | 0 | 4 |
| missing | 0 | 0 | 0 | 0 | 0 |
| stake_resolved | 380.0000 | 180.0000 | 0 | 0 | 560.0000 |
| pnl_resolved | -12.5400 | -42.8800 | — | — | -55.4200 |
| roi_resolved | -0.0330 | -0.2382 | — | — | -0.0990 |
| roi_ex_void | -0.0358 | -0.2522 | — | — | -0.1066 |
| accounting_coverage | 0.9744 | 0.5143 | — | — | 0.7568 |
| maturity | PARTIALLY_SETTLED | PARTIALLY_SETTLED | EMPTY | EMPTY | PARTIALLY_SETTLED |
| sample_gate | INSUFFICIENT_N | VERY_LOW_N | VERY_LOW_N | VERY_LOW_N | INSUFFICIENT_N |

## CLV (VALID_STRICT)

| Grupo | Janela | N | Coverage | Média | Mediana | Positivo % | Status |
|---|---|---:|---:|---:|---:|---:|---|
| FRIENDLY | POST_5M | 5 | 12.8 | -1.7862 | -1.3706 | 20.0 | VERY_LOW_N |
| FRIENDLY | POST_15M | 5 | 12.8 | -3.4717 | -2.8014 | 20.0 | VERY_LOW_N |
| FRIENDLY | CLOSING | 6 | 15.4 | -3.5568 | -2.7004 | 33.3 | VERY_LOW_N |
| NON_FRIENDLY | POST_5M | 16 | 45.7 | -0.1541 | -0.4367 | 25.0 | VERY_LOW_N |
| NON_FRIENDLY | POST_15M | 14 | 40.0 | -0.0293 | -0.3509 | 28.6 | VERY_LOW_N |
| NON_FRIENDLY | CLOSING | 6 | 17.1 | -2.0900 | -1.6768 | 16.7 | VERY_LOW_N |
| UNCLASSIFIED | POST_5M | 0 | — | — | — | — | VERY_LOW_N |
| UNCLASSIFIED | POST_15M | 0 | — | — | — | — | VERY_LOW_N |
| UNCLASSIFIED | CLOSING | 0 | — | — | — | — | VERY_LOW_N |
| CONFLICT | POST_5M | 0 | — | — | — | — | VERY_LOW_N |
| CONFLICT | POST_15M | 0 | — | — | — | — | VERY_LOW_N |
| CONFLICT | CLOSING | 0 | — | — | — | — | VERY_LOW_N |

## Execução / preço

| Métrica | Friendly | Non-Friendly | Delta |
|---|---:|---:|---:|
| odd_mediana | 1.9800 | 1.9520 | 0.0280 |
| slippage_mediana | -3.1184 | -0.6391 | -2.4793 |
| pre_submit_p50 | 1456.0000 | 1262.0000 | 194.0000 |
| pre_submit_p95 | 3237.0000 | 4813.0000 | -1576.0000 |
| place_p50 | — | — | — |
| capacity_mediana | 170.2474 | 200.6285 | -30.3811 |

## Cenários contrafactuais (diagnóstico)

> Cenários históricos não representam resultado out-of-sample e não devem ser interpretados como recomendação operacional.

- `A_H3BUP_completa`: N=74 stake=740.00 P&L=-55.4200 ROI=-0.0990
- `B_apenas_Friendly`: N=39 stake=390.00 P&L=-12.5400 ROI=-0.0330
- `C_apenas_non_Friendly`: N=35 stake=350.00 P&L=-42.8800 ROI=-0.2382
- `D_non_Friendly_plus_Unclassified`: N=35 stake=350.00 P&L=-42.8800 ROI=-0.2382
- `E_confirmed_classes_only`: N=74 stake=740.00 P&L=-55.4200 ROI=-0.0990

## Alertas

- `ACCOUNTING_COVERAGE_DIFFERENCE` (medium): friendly=0.9743589743589743 non_friendly=0.5142857142857142
- `INSUFFICIENT_N` (medium): n_resolved=56

## Segurança

- checksums_unchanged: `True`
- policy_altered: `False` → deve ser Não
- telegram_used: `False`
- orders_created: `False`
- betslip_opened: `False`

## Universo SECUNDÁRIO (apêndice)

```json
{
  "universe": "HISTORICAL_COMPARABLE_BACK_PRE",
  "diagnostic_only": true,
  "n": 0,
  "note": "Never consolidate with H3BUP_vNext exact ROI."
}
```

Nunca consolidar ROI primário + secundário numa única linha.
