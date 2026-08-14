# Auditoria — Semântica temporal Daily H3BUP — 20260729

## 1. Mapa de relógios

| Relógio | Uso V1 | Uso V2 | Risco |
|---|---|---|---|
| **UTC generation `ts`** | `day` pasta `YYYYMMDD`; cabeçalho “Dia do relatório (UTC)” | `generated_at_utc` | Confundir com coorte fechada |
| **UTC execution `created_at`** | ROIw Total por dia; tese fast | **coorte obrigatória** `execution_day_utc` half-open `[D, D+1)` | Correcto para ops |
| **UTC post date** | Séries P&L acct / algumas tabelas CF | Apenas metadata settlement/freshness — **nunca** cohort key | Dual cohort |
| **REPORT_TZ America/Sao_Paulo** | `pnl_today` / week / month | Não é base de coorte V2 | “Hoje” BRT ≠ pasta UTC |
| **Rolling 24h** | KPIs LIVE_OK / latências / CAP_BLOCKED | Diagnóstico opcional; não substitui DAILY_CLOSED | Mistura com dia civil |
| **Aderência 7d UTC** | Blocos adherence | Fora do snapshot core mínimo | Terceira janela |
| **report_cutoff_utc** | **Ausente V1** | Obrigatório no snapshot | As-of indefinido V1 |

## 2. Dia do relatório V1 (facto)

```python
ts = now(UTC)
day = ts.strftime("%Y%m%d")
```

Timer 22:00 UTC ⇒ pasta do **mesmo** dia civil UTC. Não é estritamente “ontem fechado”. Reruns no mesmo UTC day overwrite.

## 3. Contrato V2 DAILY_CLOSED

```text
report_date_utc = D-1 completo (ou --report-date)
window_start_utc = D 00:00:00Z
window_end_utc   = (D+1) 00:00:00Z   # half-open
report_cutoff_utc = now (ou override)
```

`execution_day_utc(created_at) = date(created_at_utc)`.

Post date **não** redefine o dia de execução (testes Phase 2R).

## 4. INTRADAY V2

- `report_date_utc = hoje UTC` (ou override);
- `window_end_utc = report_cutoff_utc` (parcial);
- maturity tipicamente `OPEN_COHORT` / `PARTIALLY_SETTLED`.

## 5. Fast thresholds vs tempo

| Nome | Campo temporal de latência | Limiar | Papel |
|---|---|---|---|
| Fast old V1 | `pre_submit_ms` | ≤6000 | legado pré-tese |
| Fast post-tese V1 | `pre_submit_ms` | ≤5000 | executor env |
| CF lat V1 | `call_to_done_ms` | ≤6000 | contrafactual |
| **DAILY_FAST_LE_6S** | `pre_submit_ms` | ≤6000 | contrato Daily |
| **STUDY_FAST_LT_4S** | `pre_submit_ms` | <4000 | estudo apenas |

Latência de wall-clock do pedido (`call_to_done`) ≠ latência pré-submit. Misturá-las invalida comparações.

## 6. Settlement lag

Ordens criadas no dia D podem liquidar em D+n.  
Correcto: manter coorte por `created_at` D; medir ROI settled **as-of** `report_cutoff`.  
Incorrecto: mover a ordem para o dia do post date e chamar isso “ROI do dia de execução”.

## 7. CLV windows

POST_5M / POST_15M / CLOSING são janelas **forward** após activação/LIVE_OK; `NOT_DUE` até maturarem. Não usar zeros por default.

## 8. Recomendações

1. Todo gráfico/tabela deve declarar `time_basis` no cabeçalho.
2. Proibir agregações que misturem post_date e created_at sem label.
3. Cutover V2: Telegram caption deve mostrar `report_date_utc` + `report_type` + `run_id`.
4. Manter BRT apenas para labels de banca local, nunca como cohort key H3BUP.
