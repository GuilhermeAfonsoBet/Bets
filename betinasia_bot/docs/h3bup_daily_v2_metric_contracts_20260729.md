# H3BUP Daily V2 — Contratos de métricas — 20260729

Catálogo normativo alinhado a `ops/daily_v2/contracts.py` e testes `tests/test_h3bup_daily_v2.py`.  
Export runtime: `logs/h3bup_daily_metric_catalog_v2_{YYYYMMDD}.csv`.

---

## 1. Regras transversais

| Regra | Contrato |
|---|---|
| Coorte operacional | `created_at` UTC; janela half-open |
| Post date | metadata only |
| Policy | `H3BUP_vNext` required por default |
| Dedupe | latest `created_at` por `order_id` |
| Heartbeat | excluído |
| Zero vs missing | zero só com status AVAILABLE/OK |
| Open em ROI settled | **proibido** no denominador |
| Fair edge | `NOT_IMPLEMENTED` |
| Daily fast vs study | IDs separados; study nunca substitui daily |

Constantes:

```text
DAILY_FAST_LE_6S_MS = 6000   # pre_submit_ms <= 6000
STUDY_FAST_LT_4S_MS = 4000   # pre_submit_ms <  4000
SCHEMA_VERSION = 2
```

---

## 2. Contratos (resumo tabular)

### 2.1 `live_ok_count`

- Universo: LIVE_OK Back H3BUP; exclude heartbeat.
- Fórmula: `count(distinct order_id)`.
- Null: MISSING se fonte FAILED; 0 só se HEALTHY e coorte vazia.

### 2.2 `roi_settled` (principal)

- Universo: LIVE_OK ∩ settled accounting; open excluído.
- Fórmula: `sum(pnl_confirmed_settled)/sum(stake_confirmed_settled)` (fraction).
- Void/push: pnl~0, stake no den.
- STALE accounting → `UNAVAILABLE_STALE` value null.
- PARTIAL se ainda há open/missing.

### 2.3 `roiw_total_v1` (legado)

- Fórmula: `(sum pnl / sum exposure)*100`.
- Pode incluir open se no ledger.
- Complementar; não substitui `roi_settled`.

### 2.4 `roiw_total_v2`

- Mesma fórmula em settled-like; open excluído; status PARTIAL se open/missing.

### 2.5 `daily_fast_le_6s`

- `pre_submit_ms <= 6000`.
- Missing ms → bucket `PRE_SUBMIT_MS_NA` (nunca coercido a slow).

### 2.6 `study_fast_lt_4s`

- `pre_submit_ms < 4000`.
- Label obrigatório: exploratory_only.

### 2.7 `clv_post_5m_strict` (+ 15M / CLOSING)

- Forward-only; strict valid; `INSUFFICIENT_N` se n_live_after_activation < 30.
- NOT_DUE / MISSING ≠ 0.

### 2.8 `fair_edge`

- `status=NOT_IMPLEMENTED`, `value=null`.
- Proibido default 0.

### 2.9 `e2e_ws_to_live_ok`

- `live_ok_ts - ws_ts` via trace schema_version=1.
- n=0 → INSUFFICIENT_N; sem percentis vazios.

---

## 3. Campos obrigatórios por envelope

```json
{
  "value": null,
  "unit": "fraction|percent|count|ms|null",
  "n": 0,
  "numerator": null,
  "denominator": null,
  "coverage_pct": null,
  "status": "AVAILABLE|...",
  "metric_version": "v2.0",
  "source": "...",
  "notes": []
}
```

---

## 4. Matriz de testes mandatórios (Phase 2R)

| Teste | Garante |
|---|---|
| `test_cohort_uses_created_at_utc` | dia exec |
| `test_post_date_does_not_change_execution_day` | dual cohort isolado |
| `test_utc_midnight_boundaries` | half-open |
| `test_open_not_in_roi_settled` | open fora |
| `test_void_push` | void pnl0 |
| `test_missing_accounting` | missing status |
| `test_accounting_stale` | UNAVAILABLE_STALE |
| `test_true_zero_vs_missing` | coerção zero |
| `test_heartbeat_excluded` | heartbeat |
| `test_retry_dedup_keeps_latest` | dedupe |
| `test_daily_fast_le_6s` / `test_study_fast_lt_4s` / boundaries 4s | fast split |
| `test_fair_edge_not_implemented` | fair edge |
| atomic / render smoke | IO + md |

Resultado local audit: **23 passed** (`logs/h3bup_daily_v2_tests_20260729.txt`).

---

## 5. Anti-contratos V1 (não reintroduzir)

1. Usar `call_to_done_ms` como se fosse `DAILY_FAST_LE_6S`.
2. Misturar limiar 5000 ms pós-tese no ID Daily fast sem versionar métrica.
3. Filtrar ROIw sem H3BUP e chamar de H3BUP.
4. Tratar `pnl_today` BRT como ROI settled do dia UTC.
5. Omitir fair edge sem status.
