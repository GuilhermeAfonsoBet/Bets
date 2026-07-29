# Auditoria ROI / ROIw — H3BUP Daily — 20260729

## 1. Duas famílias de métricas (não intercambiáveis)

| Família | Definição | Open | Void-like | Missing oid | Unidade típica | Onde no V1 |
|---|---|---|---|---|---|---|
| **ROIw Total** | `(∑pnl_ledger / ∑exposure_exec) * 100` | Pode **incluir** se oid está no ledger | pnl≈0 **entra** nos somatórios | **Excluído** de num e den | **percent** | Tabela principal por dia exec UTC |
| **ROI settled** | `∑pnl_confirmed_settled / ∑stake_confirmed_settled` | **Excluído** | pnl≈0 com stake no den | contado à parte (missing) | **fraction** | Accounting Health H3BUP / reconcile |

**Principal (contrato Phase 2R / V2):** `roi_settled`.  
**Complementar (legado V1):** `roiw_total` / `roiw_total_v1`.

## 2. Dual cohort que alimenta confusão

1. **P&L accounting por post date UTC** — série do ledger; útil para cashflow do dia de posting.
2. **ROIw Total por `created_at` UTC + join `order_id`** — coorte operacional de execução.

Um mesmo “dia” no PDF pode misturar linhas post-date e linhas exec-date. `pnl_today` (REPORT_TZ BRT) é ainda uma terceira janela civil.

## 3. Fórmula ROIw (código V1)

Agregador interno documentado:

```text
ROIw = (sum pnl) / (sum exposure) * 100
w = exposure-weighted  (Back: stake; Lay legado: liability quando aplicável)
```

Notas auditadas:

- exposição no ROIw Total Back vem tipicamente do stake do executor;
- P&L vem do ledger accounting por `order_id`;
- depósitos/withdrawals devem ser excluídos do mapa P&L (best-effort por tipo).

## 4. ROI settled (health / reconcile)

Em `ops/h3bup_accounting_reconcile.py`:

- universo LIVE_OK reconciliado;
- open / event not started / in progress **fora** do ROI;
- void/push: stake no denominador, pnl 0;
- `roi_settled` como fraction;
- disclaimers low_n (<30) e low_coverage (<0.95).

Render em `ops/accounting_health_report.py` na secção **Accounting Health — H3BUP**.

## 5. Policy filter H3BUP

**Gap:** a tabela principal de ROIw V1 **não** aplica de forma consistente o filtro `policy_version` contém `H3BUP_vNext`.  
V2 default `require_h3bup=True` em `load_executor_orders`.

## 6. V2 side-by-side

| metric_id | Papel |
|---|---|
| `roi_settled` | principal |
| `roiw_total_v1` | legado parity |
| `roiw_total_v2` | settled-aware percent; PARTIAL se open/missing |

Accounting `STALE`/`FAILED` → envelopes `UNAVAILABLE_STALE` (value `null`, nunca 0 falso).

## 7. Contrafactuais (não são ROI “oficial”)

Colunas “Após slippage… / Após lat≤6s / Após ambos” recalculam P&L/ROIw sob filtros. O limiar de latência usa **`call_to_done_ms≤6000`**, não o contrato `DAILY_FAST_LE_6S` (`pre_submit_ms`). Tratar como estudo.

## 8. Recomendações de leitura operacional

1. Para performance settled H3BUP: **ROI settled** + coverage + health.
2. Para comparar com histórico V1: ROIw Total, ciente de open/void/policy mix.
3. Nunca igualar `pnl_today` (BRT) a ROIw do dia exec UTC.
4. N≤30 settled → `INSUFFICIENT_N` / disclaimer; sem conclusão de edge.
