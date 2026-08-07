# H3BUP_vNext — Análise Friendly vs Non-Friendly

- status: `NO_CLEAR_FRIENDLY_DIFFERENCE`
- run_id: `a27c1dc4ab52`
- generated_at_utc: `2026-08-07T14:25:32.325470+00:00`
- cutoff_utc: `2026-08-07T14:25:06+00:00`
- friendly_classification_version: `FRIENDLY_CLASS_V1_20260731`
- classification_checksum: `1427e59fab1bb1642b76385e4aec3ed1ebd89bba64ce7bf87021eb4f4680d6b2`

## Universo PRIMÁRIO (H3BUP_vNext exact)

```json
{
  "universe": "H3BUP_vNext_exact",
  "policy_id": "H3BUP_vNext",
  "policy_version": "H3BUP_vNext_20260629",
  "n": 208,
  "n_with_order_id": 208,
  "n_fallback_identity": 0,
  "excluded": {
    "not_live_ok": 205381,
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
  "cutoff_utc": "2026-08-07T14:25:06+00:00",
  "policy_start_utc": "2026-06-29T00:00:00+00:00"
}
```

## Tabela principal de performance

| Métrica | Friendly | Non-Friendly | Unclassified | Conflict | Total |
|---|---:|---:|---:|---:|---:|
| LIVE_OK | 109 | 99 | 0 | 0 | 208 |
| eventos_unicos | 82 | 76 | 0 | 0 | 158 |
| stake_placed | 602.0000 | 502.0000 | 0 | 0 | 1104.0000 |
| open | 5 | 17 | 0 | 0 | 22 |
| settled_decided | 90 | 68 | 0 | 0 | 158 |
| void_push | 7 | 11 | 0 | 0 | 18 |
| missing | 7 | 3 | 0 | 0 | 10 |
| stake_resolved | 578.0000 | 462.0000 | 0 | 0 | 1040.0000 |
| pnl_resolved | -15.0500 | -15.1700 | — | — | -30.2200 |
| roi_resolved | -0.0260 | -0.0328 | — | — | -0.0291 |
| roi_ex_void | -0.0283 | -0.0358 | — | — | -0.0316 |
| accounting_coverage | 0.8899 | 0.7980 | — | — | 0.8462 |
| maturity | PARTIALLY_SETTLED | PARTIALLY_SETTLED | EMPTY | EMPTY | PARTIALLY_SETTLED |
| sample_gate | INSUFFICIENT_N | INSUFFICIENT_N | VERY_LOW_N | VERY_LOW_N | FIRST_READING |

## CLV (VALID_STRICT)

| Grupo | Janela | N | Coverage | Média | Mediana | Positivo % | Status |
|---|---|---:|---:|---:|---:|---:|---|
| FRIENDLY | POST_5M | 20 | 18.3 | -2.6973 | -2.1030 | 5.0 | VERY_LOW_N |
| FRIENDLY | POST_15M | 18 | 16.5 | -3.3075 | -2.8818 | 11.1 | VERY_LOW_N |
| FRIENDLY | CLOSING | 23 | 21.1 | -2.9171 | -1.7729 | 21.7 | VERY_LOW_N |
| NON_FRIENDLY | POST_5M | 46 | 46.5 | -0.7582 | -0.5077 | 17.4 | INSUFFICIENT_N |
| NON_FRIENDLY | POST_15M | 42 | 42.4 | -0.8594 | -0.9201 | 19.0 | INSUFFICIENT_N |
| NON_FRIENDLY | CLOSING | 39 | 39.4 | -1.9660 | -1.7619 | 35.9 | INSUFFICIENT_N |
| UNCLASSIFIED | POST_5M | 0 | — | — | — | — | VERY_LOW_N |
| UNCLASSIFIED | POST_15M | 0 | — | — | — | — | VERY_LOW_N |
| UNCLASSIFIED | CLOSING | 0 | — | — | — | — | VERY_LOW_N |
| CONFLICT | POST_5M | 0 | — | — | — | — | VERY_LOW_N |
| CONFLICT | POST_15M | 0 | — | — | — | — | VERY_LOW_N |
| CONFLICT | CLOSING | 0 | — | — | — | — | VERY_LOW_N |

## Execução / preço

| Métrica | Friendly | Non-Friendly | Delta |
|---|---:|---:|---:|
| odd_mediana | 1.9400 | 1.9520 | -0.0120 |
| slippage_mediana | -2.8500 | -0.6391 | -2.2109 |
| pre_submit_p50 | 1369.0000 | 1254.0000 | 115.0000 |
| pre_submit_p95 | 4451.0000 | 8919.0000 | -4468.0000 |
| place_p50 | — | — | — |
| capacity_mediana | 193.2715 | 248.4149 | -55.1434 |

## Cenários contrafactuais (diagnóstico)

> Cenários históricos não representam resultado out-of-sample e não devem ser interpretados como recomendação operacional.

- `A_H3BUP_completa`: N=208 stake=1104.00 P&L=-30.2200 ROI=-0.0291
- `B_apenas_Friendly`: N=109 stake=602.00 P&L=-15.0500 ROI=-0.0260
- `C_apenas_non_Friendly`: N=99 stake=502.00 P&L=-15.1700 ROI=-0.0328
- `D_non_Friendly_plus_Unclassified`: N=99 stake=502.00 P&L=-15.1700 ROI=-0.0328
- `E_confirmed_classes_only`: N=208 stake=1104.00 P&L=-30.2200 ROI=-0.0291

## Alertas

- `CLV_COVERAGE_DIFFERENCE` (medium): friendly=0.21100917431192662 non_friendly=0.3939393939393939

## Segurança

- checksums_unchanged: `True`
- policy_altered: `False` → deve ser Não
- telegram_used: `False`
- orders_created: `False`
- betslip_opened: `False`

## Universo SECUNDÁRIO (apêndice)

```json
{
  "skipped": true
}
```

Nunca consolidar ROI primário + secundário numa única linha.
