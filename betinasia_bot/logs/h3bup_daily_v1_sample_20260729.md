# Daily Report BetinAsia

- Dia do relatório (UTC): `20260728`
- Gerado em (UTC): `2026-07-28T22:01:08.644407+00:00`

## 0) Resumo e conclusões (executivo)

- **Policy publish**: `BLOQUEADO` (`skipped`) — mantendo policy anterior em `logs/wf_policy_current.json`.
- **Banca real (saldo atual)**: `1106.4`
- **P&L (hoje / semana / mês)**: `-9.97 / -9.97 / -176.85999999999984`
- **Lucro esperado (com gate de slippage; exec c/ placar)**: `8.62` (base `8.62`, Δ `0.00`)

**Conversão (últimas 24h; auditoria DB)**

- OK/total: **1302/1993** (65.3%)
- OK_valid/total: **1302/1993** (65.3%)

**Saúde do executor (amostra lida do JSONL; não é 24h)**

- Janela: `2026-06-23T16:49:43.118494+00:00` → `2026-07-28T22:02:05.718173+00:00` (n=50000)
- Maior gap: `107,022.3s` | gaps>5min: `1`

**Saúde do executor (últimas 24h; proxy por gaps no JSONL)**

- Janela: `2026-07-27T22:03:13.647488+00:00` → `2026-07-28T22:03:13.647514+00:00` (n=1459)
- Maior gap: `191.0s` | gaps>15min: `0` | silêncio>15min (est.): `0s` (0.00%)

**Recursos da VPS (snapshot)**

- MemAvailable: `3,316 MiB`
- vCPUs (os.cpu_count): `4`

**Atividade recente (executor)**

- Último `LIVE_OK`: `2026-07-28T17:58:48.334722+00:00` | `LIVE_OK` (1h/6h/24h): `0/1/2`

**Falhas pós-accepted (executor, 24h)**

| Métrica | Valor |
|---|---:|
| accepted | 22 |
| LIVE_OK | 2 (9.1%) |
| accepted sem LIVE_OK | 20 (90.9%) |
| precheck fail (`LIVE_PRECHECK_FAILED`) | 1 |
| place fail (`LIVE_PLACE_FAILED`) | 0 |
| API_FAILED | 1 |
| NO_SESSION | 0 |
| RATE_LIMIT | 0 |
| CAP_BLOCKED | 19 |
| No PMMs received | 0 |
| Execution context destroyed/target closed | 0 |
| Auth 401 / NO_ROOT_SESSION_COOKIE | 0 |
| p50/p90 `pmm_wait_ms` (precheck fail) | — / — |
| p50/p90 `ws_age_ms` (precheck fail) | — / — |

- Top erros pós-accepted:
  - ×16: `H3BUP_VNEXT_GATE capacity_lte_100`
  - ×3: `H3BUP_VNEXT_GATE capacity_lte_100|slippage_non_negative`
  - ×1: `NO_VALID_BOOKMAKER_PRICES | LIVE_PRECHECK_FAILED`

**Latência ponta a ponta (24h; WS → executor_done)**

- Cobertura: `n_jsonl_24h=1458`, `com_audit_id=22`, `com_hypothesis_detected_at=22`, `e2e_all=22`, `e2e_success=2`.
| Etapa | p50 | p90 | p99 | mean |
|---|---:|---:|---:|---:|
| e2e_total | 8,334 | 9,676 | 9,676 | 9,005 |
| detect_to_submit | 3,109 | 3,203 | 3,203 | 3,156 |
| audit_total | 0 | 0 | 0 | 0 |
| audit_detect_to_click | 0 | 0 | 0 | 0 |
| audit_click_to_betslip | 0 | 0 | 0 | 0 |
| audit_queue_wait | 0 | 0 | 0 | 0 |
| audit_parallel_fetch | — | — | — | — |
| audit_temporal_total | — | — | — | — |
| audit_execution | 0 | 0 | 0 | 0 |
| audit_pipeline_overhead | 0 | 0 | 0 | 0 |
| audit_db_save | 11 | 16 | 16 | 14 |
| audit_gate_wait | — | — | — | — |
| bridge_wait | 3,109 | 3,203 | 3,203 | 3,156 |
| executor_submit_to_done | 5,225 | 6,474 | 6,474 | 5,850 |
| executor_queue_delay | 0 | 1 | 1 | 0 |
| executor_post | 797 | 843 | 843 | 820 |
| executor_total_api | 1,122 | 1,211 | 1,211 | 1,166 |


**Prontidão para LIVE (go/no-go)**

| Critério | Atual | Alvo | Status |
|---|---:|---:|---|
| Live liberado (`EXECUTOR_ALLOW_LIVE`) | `True` | `True` | **OK** |
| OK_valid/total (24h, DB) | 65.3% | ≥5.0% | **OK** |
| API_FAILED/total (24h, DB) | 0.0% | ≤20.0% | **OK** |
| STALE_QUEUE_WAIT/total (24h, DB) | 0.0% | ≤10.0% | **OK** |
| `No PMMs received` (24h, DB) | 0 | ≤0 | **OK** |
| `No PMMs` / `PMM-consults` (24h, DB) | — | — | — |
| `too_many_open_betslips` (24h, DB) | 0 | ≤0 | **OK** |
| Latência p90 `call_to_done_ms` (24h; sucessos) | 6,349ms | ≤8000ms | **OK** |
| Latência p50 `call_to_done_ms` (24h; sucessos) | 5,849ms | — | — |
| n sucessos no JSONL (24h) | 2 | — | — |
| Gaps >15min no executor_jsonl (24h; proxy) | 0 | ≤8 | **OK** |

**Veredito**: **APTO (com cautela)**


**Conclusões operacionais (prioridades)**

- **Objetivo 1 (conversão)**: reduzir `API_FAILED` (especialmente `No PMMs received`) e `STALE_QUEUE_WAIT` para aumentar taxa de execução sem inflar risco.
- **Objetivo 2 (governança de risco)**: consolidar sizing/limites (banca teórica vs banca real) e travas para evitar picos (`too_many_open_betslips`, rate limit, backoff).
- **Objetivo 3 (qualidade de entrada)**: acompanhar slippage **com sinal** e seu impacto em ROI por bucket (negativo/flat/positivo) para validar edge e execução.

## 1) Resultados reais (shadow/live)

**P&L real por dia (semana corrente)**

| Dia | P&L |
|---|---:|
| 2026-07-28 | -9.97 |

**Regras efetivas (seleção + sizing) — aplicadas na execução**

| Risk params (manual) | Valor |

_(excerpt; full report on VPS logs/daily_reports/20260728/report_daily.md)_
