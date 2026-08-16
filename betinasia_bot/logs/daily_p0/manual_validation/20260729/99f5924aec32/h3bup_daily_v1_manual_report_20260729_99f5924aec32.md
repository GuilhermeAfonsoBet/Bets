# DAILY V1 — MANUAL VALIDATION

> Execução manual controlada para revisão técnica. **Não substitui** o Daily V1 oficial das 22:00 UTC.

- report_date_utc: `2026-07-29`
- cohort: `2026-07-29T00:00:00+00:00` → `2026-07-30T00:00:00+00:00` (created_at UTC)
- parity_cutoff_utc (do V1 oficial): `2026-07-29T22:01:54.606850+00:00`
- generated_at_utc: `2026-07-30T19:47:25.884636+00:00`
- run_id: `99f5924aec32`
- source_official_v1: `/home/betbot/Bets/betinasia_bot/logs/daily_reports/20260729/report_daily.md`
- git_commit: `d4a3d42d501cd44597401376de3f599d8cde9038`

## H3BUP_vNext — Resumo Oficial da Estratégia

> Os valores de banca, P&L semanal/mensal da conta e estudos históricos não representam necessariamente a performance da H3BUP_vNext.

| Métrica | Valor | Status |
|---|---|---|
| policy_version | `H3BUP_vNext_20260629` | CURRENT |
| LIVE_OK da coorte | 24 | AVAILABLE |
| stake placed | US$ 240,00 | AVAILABLE |
| open | 1 | AVAILABLE |
| settled decided | 21 | AVAILABLE |
| void/push | 2 | AVAILABLE |
| missing accounting | 0 | OK |
| stake resolved total | US$ 230,00 | AVAILABLE |
| stake decided ex-void | US$ 210,00 | AVAILABLE |
| stake void | US$ 20,00 | AVAILABLE |
| P&L resolved | US$ 6,60 | AVAILABLE |
| ROI resolved | 2.87% | `PARTIAL` |
| ROI decided ex-void | 3.14% | `PARTIAL` |
| accounting coverage | 95.8% | AVAILABLE |
| maturity | `PARTIALLY_SETTLED` | AVAILABLE |
| CLV collection status | `WATCH` | `WATCH` |
| E2E status | `AVAILABLE` | `AVAILABLE` |
| data quality | ver Accounting/CLV/E2E | SEPARATED |
| statistical readiness | `INSUFFICIENT_N` | `INSUFFICIENT_N` |

### CLV forward (VALID_STRICT) — H3BUP_vNext

| Janela | N | CLV médio | Mediana | Positivo % | Status |
|---|---:|---:|---:|---:|---|
| POST_5M | 7 | -1.78% | -0.64% | 28.57% | `INSUFFICIENT_N` |
| POST_15M | 6 | -2.32% | -0.43% | 33.33% | `INSUFFICIENT_N` |
| CLOSING | 8 | -2.82% | -1.68% | 25.00% | `INSUFFICIENT_N` |

| Janela | Expected | Due | Attempted | Strict valid | Coverage |
|---|---:|---:|---:|---:|---:|
| POST_5M | 13 | 13 | 13 | 7 | 53.8% |
| POST_15M | 13 | 13 | 12 | 6 | 46.2% |
| CLOSING | 13 | 13 | 11 | 8 | 61.5% |

- fair edge: `NOT_IMPLEMENTED`
- coorte: `29/07/2026 00:00 UTC` → `30/07/2026 00:00 UTC` (created_at UTC)

---

### Separação visual de universos

1. **H3BUP_vNext** — estratégia oficial deste resumo
2. **CONTA TOTAL** — banca/P&L da conta (abaixo)
3. **POLICIES LEGADAS** — fora deste resumo
4. **ESTUDOS HISTÓRICOS / CONTRAFACTUAIS** — apêndice de pesquisa (não operacional)


---

## Conteúdo do V1 oficial (20260729) — preservado para revisão

# Daily Report BetinAsia

- Dia do relatório (UTC): `20260729`
- Gerado em (UTC): `2026-07-29T22:01:54.606850+00:00`

## 0) Resumo e conclusões (executivo)

- **Policy publish**: `BLOQUEADO` (`skipped`) — mantendo policy anterior em `logs/wf_policy_current.json`.
- **Banca real (saldo atual)**: `1109.42`
- **P&L (hoje / semana / mês)**: `-6.510000000000003 / -6.940000000000005 / -173.82999999999987`
## Accounting Health — H3BUP

| Métrica | Valor |
|---|---|
| status | ACCOUNTING_OK / HEALTHY |
| último sucesso UTC | 2026-07-29T22:04:49.706301+00:00 |
| balance age | 7.057508230209351 |
| open_stakes age | 0.019872188568115234 |
| falhas consecutivas | 0 |
| última falha | None |
| LIVE_OK total | 12 |
| settled reconciliado | 3 |
| não iniciados | — |
| abertos | 9 |
| missing accounting | — |
| coverage accounting | 0.25 |
| stake settled | 30.0 |
| P&L settled | -0.4300000000000015 |
| ROI settled | -0.014333333333333384 |

_Disclaimer: ROI settled é parcial (N baixo e/ou coverage/health insuficientes); não é ROI total da estratégia._

## H3BUP End-to-End Latency

| Métrica | Valor |
|---|---|
| tracing status | ENABLED |
| schema version | 1 |
| traces totais | 2287 |
| traces LIVE_OK | 12 |
| coverage WS→LIVE_OK | 0.5% |
| mediana WS→LIVE_OK | 8809.345 ms |
| p95 WS→LIVE_OK | 12558.715 ms |
| mediana audit→request | 3178.442 ms |
| mediana request→LIVE_OK | 5787.158 ms |
| mediana dry-run | 1363.359 ms |
| mediana place | 4228.545 ms |
| etapa dominante | place_duration_ms |
| trace events dropped | 0 |
| clock skew violations | 78 |
| ordering violations | 78 |
| status estatístico | INSUFFICIENT_N |

### Funil de cobertura

| Etapa | N | % |
|---|---:|---:|
| H3B_WS_RECEIVED | 2287 | 100.0% |
| H3B_DETECTED | 2287 | 100.0% |
| H3B_AUDIT_PERSIST_FINISHED | 2287 | 100.0% |
| H3B_BRIDGE_FETCHED | 713 | 31.2% |
| H3B_EXEC_REQUEST_CREATED | 77 | 3.4% |
| H3B_EXECUTOR_RECEIVED | 77 | 3.4% |
| H3B_DRYRUN_FINISHED | 77 | 3.4% |
| H3B_FINAL_GATE_DECIDED | 70 | 3.1% |
| H3B_PLACE_FINISHED | 12 | 0.5% |
| LIVE_OK | 12 | 0.5% |

## H3BUP CLV Forward Collection

| Métrica | Valor |
|---|---|
| collection status | WATCH |
| collection started at | 2026-07-29T20:07:50+00:00 |
| source priority | best_odds_history,passive_collector |
| passive collector status | ENABLED |
| LIVE_OK após activação | 2 |
| obligations esperadas | 6 |
| obligations criadas | 6 |
| POST_5M strict válidas | 2 |
| POST_15M strict válidas | 2 |
| CLOSING strict válidas | 1 |
| source missing | 0 |
| line mismatch | 0 |
| kickoff missing | 0 |
| retry backlog | 1 |
| status estatístico | INSUFFICIENT_N |

- **Lucro esperado (com gate de slippage; exec c/ placar)**: `26.84` (base `34.05`, Δ `-7.21`)

**Conversão (últimas 24h; auditoria DB)**

- OK/total: **4176/4805** (86.9%)
- OK_valid/total: **4176/4805** (86.9%)

**Saúde do executor (amostra lida do JSONL; não é 24h)**

- Janela: `2026-06-24T19:20:47.250822+00:00` → `2026-07-29T22:04:23.163565+00:00` (n=50000)
- Maior gap: `107,022.3s` | gaps>5min: `1`

**Saúde do executor (últimas 24h; proxy por gaps no JSONL)**

- Janela: `2026-07-28T22:05:38.004287+00:00` → `2026-07-29T22:05:38.004390+00:00` (n=1602)
- Maior gap: `248.2s` | gaps>15min: `0` | silêncio>15min (est.): `0s` (0.00%)

**Recursos da VPS (snapshot)**

- MemAvailable: `3,461 MiB`
- vCPUs (os.cpu_count): `4`

**Atividade recente (executor)**

- Último `LIVE_OK`: `2026-07-29T20:37:54.942782+00:00` | `LIVE_OK` (1h/6h/24h): `0/12/23`

**Falhas pós-accepted (executor, 24h)**

| Métrica | Valor |
|---|---:|
| accepted | 177 |
| LIVE_OK | 23 (13.0%) |
| accepted sem LIVE_OK | 154 (87.0%) |
| precheck fail (`LIVE_PRECHECK_FAILED`) | 19 |
| place fail (`LIVE_PLACE_FAILED`) | 0 |
| API_FAILED | 16 |
| NO_SESSION | 3 |
| RATE_LIMIT | 0 |
| CAP_BLOCKED | 135 |
| No PMMs received | 12 |
| Execution context destroyed/target closed | 2 |
| Auth 401 / NO_ROOT_SESSION_COOKIE | 3 |
| p50/p90 `pmm_wait_ms` (precheck fail) | — / — |
| p50/p90 `ws_age_ms` (precheck fail) | — / — |

- Top erros pós-accepted:
  - ×89: `H3BUP_VNEXT_GATE capacity_lte_100`
  - ×25: `H3BUP_VNEXT_GATE capacity_lte_100|slippage_non_negative`
  - ×21: `H3BUP_VNEXT_GATE slippage_non_negative`
  - ×11: `No PMMs received (waited 1.2s) | LIVE_PRECHECK_FAILED`
  - ×3: `NO_ROOT_SESSION_COOKIE | LIVE_PRECHECK_FAILED`
  - ×2: `NO_VALID_BOOKMAKER_PRICES | LIVE_PRECHECK_FAILED`

**Latência ponta a ponta (24h; WS → executor_done)**

- Cobertura: `n_jsonl_24h=1602`, `com_audit_id=177`, `com_hypothesis_detected_at=177`, `e2e_all=177`, `e2e_success=23`.
| Etapa | p50 | p90 | p99 | mean |
|---|---:|---:|---:|---:|
| e2e_total | 8,676 | 11,596 | 13,560 | 8,893 |
| detect_to_submit | 3,196 | 3,330 | 4,915 | 2,962 |
| audit_total | 0 | 5 | 11 | 2 |
| audit_detect_to_click | 0 | 4 | 4 | 1 |
| audit_click_to_betslip | 0 | 0 | 0 | 0 |
| audit_queue_wait | 0 | 4 | 4 | 1 |
| audit_parallel_fetch | — | — | — | — |
| audit_temporal_total | — | — | — | — |
| audit_execution | 0 | 0 | 0 | 0 |
| audit_pipeline_overhead | 0 | 0 | 0 | 0 |
| audit_db_save | 11 | 33 | 1,181 | 66 |
| audit_gate_wait | — | — | — | — |
| bridge_wait | 3,196 | 3,330 | 4,915 | 2,961 |
| executor_submit_to_done | 5,578 | 9,025 | 10,230 | 5,930 |
| executor_queue_delay | 0 | 2 | 45 | 2 |
| executor_post | 821 | 1,711 | 4,428 | 1,199 |
| executor_total_api | 1,357 | 2,211 | 4,779 | 1,693 |


**Prontidão para LIVE (go/no-go)**

| Critério | Atual | Alvo | Status |
|---|---:|---:|---|
| Live liberado (`EXECUTOR_ALLOW_LIVE`) | `True` | `True` | **OK** |
| OK_valid/total (24h, DB) | 86.9% | ≥5.0% | **OK** |
| API_FAILED/total (24h, DB) | 0.0% | ≤20.0% | **OK** |
| STALE_QUEUE_WAIT/total (24h, DB) | 0.0% | ≤10.0% | **OK** |
| `No PMMs received` (24h, DB) | 0 | ≤0 | **OK** |
| `No PMMs` / `PMM-consults` (24h, DB) | — | — | — |
| `too_many_open_betslips` (24h, DB) | 0 | ≤0 | **OK** |
| Latência p90 `call_to_done_ms` (24h; sucessos) | 8,900ms | ≤8000ms | **FAIL** |
| Latência p50 `call_to_done_ms` (24h; sucessos) | 5,578ms | — | — |
| n sucessos no JSONL (24h) | 23 | — | — |
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
| 2026-07-28 | -0.43 |
| 2026-07-29 | -6.51 |

**Regras efetivas (seleção + sizing) — aplicadas na execução**

| Risk params (manual) | Valor |
|---|---|

| Runtime | Valor |
|---|---|
| EXECUTOR_ALLOW_LIVE | `1` |
| BRIDGE_USE_WF_BUDGET | `0` |
| BRIDGE_ENFORCE_WF_FILTERS | `1` |
| BRIDGE_WF_RISK_MODE_OVERRIDE | `fixed` |
| BRIDGE_BANKROLL_REF | `3000` |
| BRIDGE_POLICY_JSON | `logs/wf_policy_current.json` |
| BRIDGE_RISK_PARAMS_JSON | `logs/bridge_risk_params.json` |

_Nota: filtro **AH** é por **|linha|** (ex.: `ah_max_abs_line=2.0` significa |line|≤2.0), não por odds; odds médias >2 podem ocorrer mesmo com AH válido._

**Risco/consistência (a partir do P&L diário)**

| Métrica | Valor |
|---|---:|
| Max drawdown (diário, monetário) | 807.56 |
| Max drawdown (semanal, monetário) | 618.09 |
| Max drawdown (mensal, monetário) | 618.09 |
| Janela do DD | 2026-03-16 → 2026-07-29 |
| Sharpe anualizado (vs banca real) | 0.88 |
| ROI/banca real (semana) | -0.63% |
| ROI/banca real (mês) | -15.67% |
| Sharpe anualizado (vs banca teórica) | 0.88 |
| ROI/banca teórica (semana; ref=10,000) | -0.07% |
| ROI/banca teórica (mês; ref=10,000) | -1.74% |

**Semanas anteriores fechadas (mês corrente)**

| Semana (start) | P&L |
|---|---:|
| 2026-07-06 | -85.82 |
| 2026-07-20 | -74.04 |

**Execução — métricas mínimas por tipo (Back/Lay × Pre/In; janela curta)**

| Tipo | #ordens | #eventos_jsonl | #linhas_api | #jogos | Valor em risco ($) | Ticket médio ($/ordem) | Stake total ($) | #liq | #pend | P&L (liq, $) | ROI% (liq) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Back Pre | 23 | 23 | 0 | 18 | 230.00 | 10.00 | 230.00 | 0 | 23 | 0.00 | — |
| Back In | 0 | 0 | 0 | 0 | 0.00 | — | 0.00 | 0 | 0 | 0.00 | — |
| Lay Pre | 0 | 0 | 0 | 0 | 0.00 | — | 0.00 | 0 | 0 | 0.00 | — |
| Lay In | 0 | 0 | 0 | 0 | 0.00 | — | 0.00 | 0 | 0 | 0.00 | — |
| **TOTAL** | **23** | **23** | **0** | **18** | **230.00** | **10.00** | **230.00** | **0** | **23** | **0.00** | **—** |

**Execução (últimos dias; executor_jsonl + placares quando disponíveis)**

| Dia | Exec rows | Sucessos | LIVE_OK | DRY_OK | API_FAILED | N Back | N Lay | Apostado Back ($) | Apostado Lay stake ($) | Apostado Lay liab ($) | P&L total (acct; post date UTC) | ROI/$ (acct) | P&L Back (acct; oid join) | P&L Back Pre (acct; oid) | P&L Back In (acct; oid) | Δ (acct_total - acct_back_oid) | Cobertura oids% (Back) | P&L (placar) | ROI/$ (placar) | P&L Back | ROI Back | P&L Lay | ROI Lay/liab | ROI Lay/stake |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-07-23 | 68 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | -32.27 | — | -32.27 | -32.27 | 0.00 | 0.00 | 0.1% | 0.00 | — | 0.00 | — | 0.00 | — | — |
| 2026-07-24 | 167 | 0 | 0 | 0 | 6 | 0 | 0 | 0.00 | 0.00 | 0.00 | — | — | — | — | — | — | — | 0.00 | — | 0.00 | — | 0.00 | — | — |
| 2026-07-25 | 111 | 0 | 0 | 0 | 10 | 0 | 0 | 0.00 | 0.00 | 0.00 | — | — | — | — | — | — | — | 0.00 | — | 0.00 | — | 0.00 | — | — |
| 2026-07-26 | 38 | 0 | 0 | 0 | 3 | 0 | 0 | 0.00 | 0.00 | 0.00 | — | — | — | — | — | — | — | 0.00 | — | 0.00 | — | 0.00 | — | — |
| 2026-07-27 | 52 | 0 | 0 | 0 | 3 | 0 | 0 | 0.00 | 0.00 | 0.00 | — | — | — | — | — | — | — | 0.00 | — | 0.00 | — | 0.00 | — | — |
| 2026-07-28 | 23 | 3 | 3 | 0 | 1 | 3 | 0 | 30.00 | 0.00 | 0.00 | -9.97 | -33.23% | -9.97 | -9.97 | 0.00 | 0.00 | 0.0% | 26.84 | 89.47% | 26.84 | 89.47% | 0.00 | — | — |
| 2026-07-29 | 176 | 22 | 22 | 0 | 16 | 22 | 0 | 220.00 | 0.00 | 0.00 | 3.03 | 1.38% | 3.03 | 3.03 | 0.00 | -0.00 | 0.3% | 7.21 | 24.03% | 7.21 | 24.03% | 0.00 | — | — |

_Nota: `P&L total (acct)` é calculado por **post date UTC** diretamente do `balance.csv` quando disponível (exclui depósitos/saques/etc.). `P&L Back Pre/In (acct; order_id)` é **Back-only** via join `order_id` (ledger ↔ executor_jsonl) e inclui tipos P&L-like (ex.: void/refund) quando existirem. Se o CSV não tiver `order_id`, esses campos podem ficar vazios._

**Accounting (por order_id): P&L por dia de execução (created_at UTC; Back Pre/In)**

| Dia (exec UTC) | P&L Back Pre | P&L Back In | P&L Total | ROIw Total | Cobertura oids% (no dia) | #ordens c/ P&L≈0 (void-like) |
|---|---:|---:|---:|---:|---:|---:|
| 2026-07-28 | -0.43 | 0.00 | -0.43 | -1.43% | 100.0% | 1 |
| 2026-07-29 | -6.51 | 0.00 | -6.51 | -4.07% | 72.7% | 1 |

**Slippage × ROI (accounting; order_id) — Back (janela móvel: 2026-07-23..2026-07-29 (7 dias))**

| Bucket slippage_raw_pct | n | Exposição (∑stake) | P&L (∑acct) | ROIw |
|---|---:|---:|---:|---:|
| <= -2% | 10 | 100.00 | 1.49 | 1.49% |
| (-2, 2] | 9 | 90.00 | -8.43 | -9.37% |

**Slippage × ROI (accounting; order_id) — Back Pre (janela móvel: 2026-07-23..2026-07-29 (7 dias))**

| Bucket slippage_raw_pct | n | Exposição (∑stake) | P&L (∑acct) | ROIw |
|---|---:|---:|---:|---:|
| <= -2% | 10 | 100.00 | 1.49 | 1.49% |
| (-2, 2] | 9 | 90.00 | -8.43 | -9.37% |

**Slippage × ROI (accounting; order_id) — Back (acumulado pós-início >= 2026-04-04)**

| Bucket slippage_raw_pct | n | Exposição (∑stake) | P&L (∑acct) | ROIw |
|---|---:|---:|---:|---:|
| <= -2% | 682 | 2,785.50 | 82.54 | 2.96% |
| (-2, 2] | 3328 | 21,824.00 | -465.43 | -2.13% |
| > 2% | 748 | 2,364.00 | -222.09 | -9.39% |

**Slippage × ROI (accounting; order_id) — Back Pre (acumulado pós-início >= 2026-04-04)**

| Bucket slippage_raw_pct | n | Exposição (∑stake) | P&L (∑acct) | ROIw |
|---|---:|---:|---:|---:|
| <= -2% | 131 | 1,489.50 | 116.87 | 7.85% |
| (-2, 2] | 1376 | 17,000.00 | -369.39 | -2.17% |
| > 2% | 50 | 603.00 | -26.82 | -4.45% |

**Slippage × ROI (accounting; order_id) — Back In (acumulado pós-início >= 2026-04-04)**

| Bucket slippage_raw_pct | n | Exposição (∑stake) | P&L (∑acct) | ROIw |
|---|---:|---:|---:|---:|
| <= -2% | 551 | 1,296.00 | -34.33 | -2.65% |
| (-2, 2] | 1952 | 4,824.00 | -96.04 | -1.99% |
| > 2% | 698 | 1,761.00 | -195.27 | -11.09% |

**Tese: Back Pre fast (pós-início; elegível HI) — performance (accounting; order_id)**

- Critério antigo (`até 2026-04-19`): `stake em [5.00, 14.00]` e `pre_submit_ms<= 6000ms`.
- Critério atual (`desde 2026-04-20`): `stake > 5.00` e `pre_submit_ms<= 5000ms`.
- Stake HI configurado no executor (`EXECUTOR_BACKPRE_FAST_STAKE_HI`): `10.00`.

| Grupo | n_ordens | n_liquidadas | n_abertas | Stake_liquidado (∑) | P&L_liquidado (∑acct) | ROIw_liquidado |
|---|---:|---:|---:|---:|---:|---:|
| Back Pre fast (desde 2026-04-20; stake > 5.00; pre_submit_ms<= 5000ms) | 1046 | 1046 | 0 | 17,534.00 | -321.55 | -1.83% |

**Back Pre slow (pós-início; stake < limiar HI) — performance auxiliar (accounting; order_id)**

- Critério antigo (`até 2026-04-19`): `pre_submit_ms> 6000ms` e `stake < 5.00`.
- Critério atual (`desde 2026-04-20`): `pre_submit_ms> 5000ms` e `stake < 5.00`.

| Grupo | n_ordens | n_liquidadas | n_abertas | Stake_liquidado (∑) | P&L_liquidado (∑acct) | ROIw_liquidado |
|---|---:|---:|---:|---:|---:|---:|
| Back Pre slow (até 2026-04-19; stake < 5.00; pre_submit_ms> 6000ms) | 32 | 32 | 0 | 96.00 | -11.56 | -12.04% |
| Back Pre slow (desde 2026-04-20; stake < 5.00; pre_submit_ms> 5000ms) | 123 | 123 | 0 | 244.50 | 4.76 | 1.95% |

**Tese Back Pre fast — compliance (pós-início; distribuição de stake e pre_submit_ms)**

| Grupo | n_ordens | stake=HI (critério por período) | stake≈10.00 | stake=other/NA |
|---|---:|---:|---:|---:|
| Back Pre fast (critério dinâmico por período) | 1091 | 1046 | 0 | 45 |
| Back Pre slow (critério dinâmico por período) | 179 | 24 | 0 | 155 |
| Back Pre fast (até 2026-04-19; pre_submit_ms<= 6000ms) | 29 | 0 | 0 | 29 |
| Back Pre fast (desde 2026-04-20; pre_submit_ms<= 5000ms) | 1062 | 1046 | 0 | 16 |
| Back Pre slow (até 2026-04-19; pre_submit_ms> 6000ms) | 32 | 0 | 0 | 32 |
| Back Pre slow (desde 2026-04-20; pre_submit_ms> 5000ms) | 147 | 24 | 0 | 123 |
| Back Pre (pre_submit_ms NA) | 293 | 0 | 0 | 293 |
| Back In | 3204 | 0 | 0 | 3204 |

**Tese: Back Pre fast — slippage_pre_pct (bucket 3-way; accounting por order_id)**

| Grupo | n_ordens | slippage_pre_pct mean | slippage_pre_pct mediana | <= -2% | (-2,2] | > 2% | NA |
|---|---:|---:|---:|---:|---:|---:|---:|
| Back Pre fast (critério dinâmico por período) | 1091 | -0.53% | -0.46% | 97 | 957 | 37 | 0 |
| Back Pre slow (critério dinâmico por período) | 179 | -0.78% | -0.56% | 20 | 156 | 3 | 0 |
| Back Pre fast (até 2026-04-19; pre_submit_ms<= 6000ms) | 29 | -0.35% | -0.49% | 0 | 28 | 1 | 0 |
| Back Pre fast (desde 2026-04-20; pre_submit_ms<= 5000ms) | 1062 | -0.53% | -0.46% | 97 | 929 | 36 | 0 |
| Back Pre slow (até 2026-04-19; pre_submit_ms> 6000ms) | 32 | -0.54% | -0.48% | 4 | 27 | 1 | 0 |
| Back Pre slow (desde 2026-04-20; pre_submit_ms> 5000ms) | 147 | -0.83% | -0.60% | 16 | 129 | 2 | 0 |
| Back Pre (pre_submit_ms NA) | 293 | — | — | 0 | 0 | 0 | 293 |
| Back In | 3204 | 6.34% | 0.00% | 389 | 1222 | 394 | 1199 |

**Tese: Back Pre fast vs slow — diferença de ROI mean (por ordem)**

- Critério dinâmico por período (pré: `<= 6000ms`; pós: `<= 5000ms`).
- Amostra líquida: fast=`1091` | slow=`179` | min_n=`25`.
- Delta (fast − slow) IC90 bootstrap: `-13.86% .. 9.20%`.
- Delta (fast − slow) IC95 bootstrap: `-16.25% .. 11.67%`.

**Accounting ledger: Voids/Refunds/Cancels por dia (post date UTC)**

| Dia | Bet (∑amount) | Void/Push (∑amount) | Refund (∑amount) | Cancel (∑amount) | Excluídos (dep/saque/etc.) | Top types (|amt|) |
|---|---:|---:|---:|---:|---:|---|
| 2026-07-23 | -32.27 | 0.00 | 0.00 | 0.00 | 0.00 | `bet`(-32.27) |
| 2026-07-28 | -9.97 | 0.00 | 0.00 | 0.00 | 0.00 | `bet`(-9.97) |
| 2026-07-29 | 3.03 | 0.00 | 0.00 | 0.00 | 0.00 | `bet`(3.03) |

**Cobertura de placar (somente entre execuções bem-sucedidas)**

| Dia | Back n_cov/n_success | Back stake_cov/stake | Back jogos_cov/jogos_success | Lay n_cov/n_success | Lay stake_cov/stake | Lay jogos_cov/jogos_success |
|---|---:|---:|---:|---:|---:|---:|
| 2026-07-23 | — | — | — | — | — | — |
| 2026-07-24 | — | — | — | — | — | — |
| 2026-07-25 | — | — | — | — | — | — |
| 2026-07-26 | — | — | — | — | — | — |
| 2026-07-27 | — | — | — | — | — | — |
| 2026-07-28 | 3/3 (100.0%) | 30.00/30.00 (100.0%) | 3/3 (100.0%) | — | — | — |
| 2026-07-29 | 3/22 (13.6%) | 30.00/220.00 (13.6%) | 2/17 (11.8%) | — | — | — |

**Contrafactual (accounting ledger; por order_id): filtros operacionais (Back)**

_Nota: P&L aqui vem do ledger por `order_id`. Quando o CSV expõe `type`, incluímos todos os tipos **P&L-like** (exclui dep/saque/transfer/etc.), para capturar void/refund/cancel se existirem. Caso contrário, cai no legado `type=bet`._

| Dia (post date UTC) | Base P&L (acct) | Base ROIw | Após slippage_raw_pct<=+2%: P&L | ROIw | Após lat<=6s: P&L | ROIw | Após ambos: P&L | ROIw | Cobertura (orders com acct no dia) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-07-23 | -32.27 | -53.78% | -32.27 | -53.78% | 7.67 | 38.35% | 7.67 | 38.35% | 3 |
| 2026-07-28 | -9.97 | -49.85% | -9.97 | -49.85% | -9.97 | -99.70% | -9.97 | -99.70% | 2 |
| 2026-07-29 | 3.03 | 1.78% | 3.03 | 1.78% | 2.16 | 2.16% | 2.16 | 2.16% | 17 |

**Contrafactual (accounting ledger; por order_id): filtros operacionais (Back In somente)**

| Dia (post date UTC) | Base P&L (acct) | Base ROIw | Após slippage_raw_pct<=+2%: P&L | ROIw | Após lat<=6s: P&L | ROIw | Após ambos: P&L | ROIw | Cobertura (orders Back In com acct no dia) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

**Contrafactual (accounting ledger; por order_id): filtros operacionais (Back Pre somente)**

_Nota: `Base P&L` usa todas as ordens Back Pre com `order_id` no ledger daquele dia. O filtro contrafactual usa `slippage_raw_pct` (pós-execução, `odd_final` vs `odd_at_decision`). Se o gate operacional `slippage_raw_pct<=+2%` já estiver efetivamente aplicado no runtime, `Base` e `Após slippage<=+2%` tendem a coincidir; divergências sugerem ordens fora do gate e/ou diferença entre métricas de slippage usadas no runtime vs relatório._

| Dia (post date UTC) | Base P&L (acct) | Base ROIw | Após slippage_raw_pct<=+2%: P&L | ROIw | Após lat<=6s: P&L | ROIw | Após ambos: P&L | ROIw | Cobertura (orders Back Pre com acct no dia) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-07-23 | -32.27 | -53.78% | -32.27 | -53.78% | 7.67 | 38.35% | 7.67 | 38.35% | 3 |
| 2026-07-28 | -9.97 | -49.85% | -9.97 | -49.85% | -9.97 | -99.70% | -9.97 | -99.70% | 2 |
| 2026-07-29 | 3.03 | 1.78% | 3.03 | 1.78% | 2.16 | 2.16% | 2.16 | 2.16% | 17 |

**Contrafactual (placar; somente cobertos por ROI): filtros operacionais (Back)**

| Dia | Base P&L | Base ROIw | Após slippage_raw_pct<=+2%: P&L | ROIw | Após lat<=6s: P&L | ROIw | Após ambos: P&L | ROIw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-07-23 | 0.00 | — | 0.00 | — | 0.00 | — | 0.00 | — |
| 2026-07-24 | 0.00 | — | 0.00 | — | 0.00 | — | 0.00 | — |
| 2026-07-25 | 0.00 | — | 0.00 | — | 0.00 | — | 0.00 | — |
| 2026-07-26 | 0.00 | — | 0.00 | — | 0.00 | — | 0.00 | — |
| 2026-07-27 | 0.00 | — | 0.00 | — | 0.00 | — | 0.00 | — |
| 2026-07-28 | 26.84 | 89.47% | 26.84 | 89.47% | 18.07 | 90.35% | 18.07 | 90.35% |
| 2026-07-29 | 7.21 | 24.03% | 7.21 | 24.03% | 7.21 | 24.03% | 7.21 | 24.03% |

**Accounting: distribuição de P&L por jogo (event_id; por post date UTC)**

| Dia | #jogos | P&L total (acct; bets) | P&L mediana/jogo | Stake médio/jogo (proxy) | ROI mediana (P&L mediana / stake médio) | P10 | P90 | Concentração P&L (max |abs| / soma |abs|) | Turnover proxy (∑-amount) | Concentração turnover (max share) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-07-23 | 1 | -32.27 | -32.27 | 39.94 | -80.80% | — | — | 100.00% | 39.94 | 100.00% |
| 2026-07-24 | 0 | 0.00 | — | — | — | — | — | — | 0.00 | — |
| 2026-07-25 | 0 | 0.00 | — | — | — | — | — | — | 0.00 | — |
| 2026-07-26 | 0 | 0.00 | — | — | — | — | — | — | 0.00 | — |
| 2026-07-27 | 0 | 0.00 | — | — | — | — | — | — | 0.00 | — |
| 2026-07-28 | 1 | -9.97 | -9.97 | 9.97 | -100.00% | — | — | 100.00% | 9.97 | 100.00% |
| 2026-07-29 | 1 | 3.03 | 3.03 | 68.86 | 4.40% | — | — | 100.00% | 68.86 | 100.00% |

**Risco de cauda por jogo (event_id; accounting)**

| Dia | #jogos | P5 P&L/jogo | CVaR5 P&L/jogo (média piores 5%) | Pior jogo |
|---|---:|---:|---:|---:|
| 2026-07-23 | 1 | — | — | — |
| 2026-07-24 | 0 | — | — | — |
| 2026-07-25 | 0 | — | — | — |
| 2026-07-26 | 0 | — | — | — |
| 2026-07-27 | 0 | — | — | — |
| 2026-07-28 | 1 | — | — | — |
| 2026-07-29 | 1 | — | — | — |

**Top jogos por exposição (proxy) — concentração operacional**

| Dia | event_id | event_name | Exposição proxy (∑-amount) | Share da exposição do dia | P&L por jogo |
|---|---|---|---:|---:|---:|
| 2026-07-23 | __NO_EVENT_ID__ |  | 39.94 | 100.00% | -32.27 |
| 2026-07-28 | __NO_EVENT_ID__ |  | 9.97 | 100.00% | -9.97 |
| 2026-07-29 | __NO_EVENT_ID__ |  | 68.86 | 100.00% | 3.03 |

**Accounting: coberto vs não-coberto (placar), por jogo/event_id (mesmo dia UTC)**

| Dia | Back jogos_success | Back jogos_cov | Back jogos_uncov | P&L acct cov | P&L acct uncov | Turnover proxy cov | Turnover proxy uncov |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-07-28 | 3 | 3 | 0 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2026-07-29 | 17 | 2 | 15 | 0.00 | 0.00 | 0.00 | 0.00 |

**Quebra (placar): Back/Lay × Pre/In (somente cobertos por ROI)**

| Dia | P&L Back Pre | ROI Back Pre | P&L Back In | ROI Back In | P&L Lay Pre | ROI Lay Pre/liab | P&L Lay In | ROI Lay In/liab |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-07-23 | 0.00 | — | 0.00 | — | 0.00 | — | 0.00 | — |
| 2026-07-24 | 0.00 | — | 0.00 | — | 0.00 | — | 0.00 | — |
| 2026-07-25 | 0.00 | — | 0.00 | — | 0.00 | — | 0.00 | — |
| 2026-07-26 | 0.00 | — | 0.00 | — | 0.00 | — | 0.00 | — |
| 2026-07-27 | 0.00 | — | 0.00 | — | 0.00 | — | 0.00 | — |
| 2026-07-28 | 26.84 | 89.47% | 0.00 | — | 0.00 | — | 0.00 | — |
| 2026-07-29 | 7.21 | 24.03% | 0.00 | — | 0.00 | — | 0.00 | — |

**Latência × ROI (Back Pre/In) — acumulado (call_to_done_ms)**

- Latência por execução vem de result.timing.call_to_done_ms no executor_jsonl. ROI/placar usa somente odd executada (odd_final). Se odd_final estiver ausente, a execução não entra no subconjunto coberto.

- **Back Pre (ROI por stake)**

| Bucket call_to_done_ms | n | ROI mean (SE; IC95) |
|---|---:|---:|
| < 5s | 608 | 19.52% (SE 3.61%) [12.44%, 26.59%] | ROIw 21.67% (odd~1.94, exp~20.00) |
| 5-10s | 634 | 12.99% (SE 3.52%) [6.10%, 19.88%] | ROIw 13.54% (odd~1.94, exp~12.00) |
| 10-20s | 194 | 12.99% (SE 6.19%) [0.86%, 25.11%] | ROIw 21.85% (odd~1.94, exp~3.00) |
| 20-40s | 22 | 12.68% (SE 17.60%) [-21.83%, 47.18%] | ROIw 8.13% (odd~1.95, exp~3.00) |
| > 40s | 8 | -6.65% (SE 30.59%) [-66.61%, 53.31%] | ROIw 42.87% (odd~1.87, exp~1.50) |

- **Back In (ROI por stake)**

| Bucket call_to_done_ms | n | ROI mean (SE; IC95) |
|---|---:|---:|
| < 5s | 985 | 35.27% (SE 4.91%) [25.64%, 44.90%] | ROIw 51.24% (odd~1.93, exp~3.00) |
| 5-10s | 1175 | 36.10% (SE 3.75%) [28.76%, 43.44%] | ROIw 36.05% (odd~1.95, exp~3.00) |
| 10-20s | 700 | 30.41% (SE 4.29%) [22.00%, 38.81%] | ROIw 29.97% (odd~1.95, exp~3.00) |
| 20-40s | 265 | 44.84% (SE 9.66%) [25.91%, 63.77%] | ROIw 38.53% (odd~1.94, exp~3.00) |
| > 40s | 24 | 45.57% (SE 17.01%) [12.24%, 78.91%] | ROIw 39.88% (odd~2.01, exp~3.00) |

**Slippage × Latência (Back Pre/In) — acumulado (call_to_done_ms)**

- Slippage_raw_pct vs latência usa execuções com ROI via placar e odd_final presente; slippage_raw_pct=(odd_final-odd_at_decision)/odd_at_decision.

- **Back Pre (slippage_raw_pct por stake)**

| Bucket call_to_done_ms | n | Slippage_raw mean (SE; IC95) | Slippage_raw mediana | Slippage_raw_w (por exp.) | ROIw (por exp.) |
|---|---:|---:|---:|---:|---:|
| < 5s | 608 | -0.38% (SE 0.10%) [-0.58%, -0.18%] | -0.44% | -0.28% | 21.67% |
| 5-10s | 634 | -0.47% (SE 0.07%) [-0.61%, -0.34%] | -0.46% | -0.41% | 13.54% |
| 10-20s | 194 | -0.64% (SE 0.19%) [-1.03%, -0.26%] | -0.52% | -0.25% | 21.85% |
| 20-40s | 22 | -0.20% (SE 0.25%) [-0.70%, 0.30%] | -0.24% | -0.35% | 8.13% |
| > 40s | 8 | -1.87% (SE 1.24%) [-4.30%, 0.56%] | -1.69% | -3.30% | 42.87% |

- **Back In (slippage_raw_pct por stake)**

| Bucket call_to_done_ms | n | Slippage_raw mean (SE; IC95) | Slippage_raw mediana | Slippage_raw_w (por exp.) | ROIw (por exp.) |
|---|---:|---:|---:|---:|---:|
| < 5s | 985 | 6.36% (SE 1.61%) [3.21%, 9.50%] | 0.00% | 25.81% | 51.24% |
| 5-10s | 1175 | 25.63% (SE 20.11%) [-13.79%, 65.04%] | 0.00% | 166.86% | 36.05% |
| 10-20s | 700 | 23.18% (SE 15.13%) [-6.48%, 52.84%] | 0.00% | 94.59% | 29.97% |
| 20-40s | 265 | 0.96% (SE 0.81%) [-0.63%, 2.54%] | 0.00% | 1.20% | 38.53% |
| > 40s | 24 | 1.47% (SE 1.11%) [-0.71%, 3.65%] | 0.35% | 0.15% | 39.88% |

**Slippage × ROI por bucket (raw, com sinal) — acumulado (range: `2026-03-20` → `2026-07-29`; span_days=`132`)**

- **Back (ROI por stake)**

| Bucket slippage_raw_pct | n | ROI mean (SE; IC95) |
|---|---:|---:|
| <= -2% | 684 | 37.10% (SE 3.74%) [29.78%, 44.42%] | ROIw 39.01% (odd~1.90, exp~3.00) |
| (-2, 2] | 3199 | 21.70% (SE 1.64%) [18.48%, 24.91%] | ROIw 18.97% (odd~1.94, exp~3.00) |
| > 2% | 732 | 53.94% (SE 8.15%) [37.96%, 69.92%] | ROIw 66.10% (odd~2.04, exp~3.00) |

- **Lay (ROI por liability)**

| Bucket slippage_raw_pct | n | ROI mean (SE; IC95) |
|---|---:|---:|
| <= -2% | 2 | -100.00% (SE 0.00%) [-100.00%, -100.00%] | ROIw -100.00% (odd~1.18, exp~20.08) |
| (-2, 2] | 0 | — |
| > 2% | 0 | — |

- **Back Pre (ROI por stake)**

| Bucket slippage_raw_pct | n | ROI mean (SE; IC95) |
|---|---:|---:|
| <= -2% | 116 | 15.91% (SE 7.81%) [0.61%, 31.22%] | ROIw 24.01% (odd~1.88, exp~11.00) |
| (-2, 2] | 1299 | 15.79% (SE 2.46%) [10.98%, 20.60%] | ROIw 17.51% (odd~1.94, exp~12.00) |
| > 2% | 51 | 9.60% (SE 13.09%) [-16.06%, 35.26%] | ROIw 23.03% (odd~1.99, exp~20.00) |

- **Back In (ROI por stake)**

| Bucket slippage_raw_pct | n | ROI mean (SE; IC95) |
|---|---:|---:|
| <= -2% | 568 | 41.43% (SE 4.19%) [33.22%, 49.63%] | ROIw 48.34% (odd~1.92, exp~3.00) |
| (-2, 2] | 1900 | 25.74% (SE 2.19%) [21.45%, 30.03%] | ROIw 22.02% (odd~1.93, exp~3.00) |
| > 2% | 681 | 57.26% (SE 8.70%) [40.21%, 74.30%] | ROIw 75.90% (odd~2.05, exp~3.00) |

- Nota: `ROIw` é o **ROI ponderado por exposição** (peso=stake no Back; peso=liability no Lay). Em prática, dentro de um bucket, `ROIw ≈ (∑P&L)/(∑exposição)`; já o `ROI mean` é a média simples por linha/sinal.

- Nota importante (reconciliação): as tabelas **Slippage × ROI** usam **somente execuções cobertas por ROI via placar** (precisa audit+placar+odd). Isso é um subconjunto e pode ter viés (ex.: jogos ainda não liquidaram, falta de odds finais, etc.). Já o **accounting ledger** inclui todo o resultado financeiro (incluindo void/refund/cancel quando existirem) por `post date`.

- **Lay (ROI por stake; bounded)**

| Bucket slippage_raw_pct | n | ROI mean (SE; IC95) |
|---|---:|---:|
| <= -2% | 2 | -18.20% (SE 9.90%) [-37.60%, 1.20%] | ROIw -17.52% (odd~1.18, exp~114.63) |
| (-2, 2] | 0 | — |
| > 2% | 0 | — |

**Contrafactual (placar): aplicar filtro de slippage**

- Regra: **Back** pula `slippage_raw_pct <= -2%`; **Lay** pula `slippage_raw_pct > 2%`.
- Observação: usa somente execuções com ROI via placar; não é o P&L do accounting.

| Lado | n (base) | P&L (base) | Exposição (base) | n (após filtro) | P&L (após) | Exposição (após) |
|---|---:|---:|---:|---:|---:|---:|
| Back | 4615 | 8,597.37 | 31,244.50 | 3931 | 7,200.18 | 27,662.50 |
| Lay (liab) | 2 | -40.17 | 40.17 | 2 | -40.17 | 40.17 |
| **Total** | — | 8,557.21 | — | — | 7,160.01 | — |

**Diagnóstico AH (linha) observado na execução**

- Policy: `ah_max_abs_line=0.0` | `ah_scope=all`
- Execuções (todas): `n=8938` | `max|line|=10.00` | `n_over=7684`
- Execuções com placar/ROI: `n=6276` | `max|line|=10.00` | `n_over=5344`

**Slippage × ROI por combinação (top 2 por volume; acumulado)**

- **Back**

| Combinação | n | ROI<=-2% | n | ROI(-2..2] | n | ROI>2% | n | corr(slip_raw,ROI) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Back_In_Any | 3149 | 41.43% (SE 4.19%) [33.22%, 49.63%] | 568 | 25.74% (SE 2.19%) [21.45%, 30.03%] | 1900 | 57.26% (SE 8.70%) [40.21%, 74.30%] | 681 | 0.06 |
| Back_Pre_Any | 1466 | 15.91% (SE 7.81%) [0.61%, 31.22%] | 116 | 15.79% (SE 2.46%) [10.98%, 20.60%] | 1299 | 9.60% (SE 13.09%) [-16.06%, 35.26%] | 51 | 0.03 |

**Slippage × ROI por combinação (top 1 por volume; acumulado)**

- **Lay**

| Combinação | n | ROI<=-2% | n | ROI(-2..2] | n | ROI>2% | n | corr(slip_raw,ROI) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Lay_In_Yes | 2 | -100.00% (SE 0.00%) [-100.00%, -100.00%] | 2 | — | 0 | — | 0 | — |

**Slippage × Latência (Back Pre/In) — pós-início (>= 2026-04-04)**

- Slippage_raw_pct vs latência usa execuções com ROI via placar e odd_final presente; slippage_raw_pct=(odd_final-odd_at_decision)/odd_at_decision.

- **Back Pre (slippage_raw_pct por stake)**

| Bucket call_to_done_ms | n | Slippage_raw mean (SE; IC95) | Slippage_raw mediana | Slippage_raw_w (por exp.) | ROIw (por exp.) |
|---|---:|---:|---:|---:|---:|
| < 5s | 585 | -0.50% (SE 0.06%) [-0.63%, -0.38%] | -0.44% | -0.52% | 22.39% |
| 5-10s | 598 | -0.48% (SE 0.07%) [-0.63%, -0.34%] | -0.47% | -0.43% | 15.02% |
| 10-20s | 184 | -0.70% (SE 0.20%) [-1.09%, -0.31%] | -0.52% | -0.45% | 19.07% |
| 20-40s | 21 | -0.17% (SE 0.27%) [-0.69%, 0.35%] | -0.05% | -0.15% | 12.65% |
| > 40s | 5 | -1.27% (SE 1.64%) [-4.49%, 1.95%] | -1.80% | -1.27% | -13.46% |

- **Back In (slippage_raw_pct por stake)**

| Bucket call_to_done_ms | n | Slippage_raw mean (SE; IC95) | Slippage_raw mediana | Slippage_raw_w (por exp.) | ROIw (por exp.) |
|---|---:|---:|---:|---:|---:|
| < 5s | 905 | 2.88% (SE 0.95%) [1.01%, 4.75%] | 0.00% | 3.55% | 33.04% |
| 5-10s | 1097 | 2.93% (SE 1.06%) [0.85%, 5.02%] | 0.00% | 2.94% | 37.64% |
| 10-20s | 664 | 12.30% (SE 10.10%) [-7.49%, 32.09%] | 0.00% | 13.90% | 31.66% |
| 20-40s | 262 | 0.95% (SE 0.82%) [-0.65%, 2.56%] | 0.00% | 1.19% | 47.10% |
| > 40s | 22 | 1.75% (SE 1.16%) [-0.53%, 4.02%] | 0.35% | 1.75% | 46.78% |

**Slippage × ROI por bucket (raw, com sinal) — pós-início (>= 2026-04-04) (range: `2026-04-04` → `2026-07-29`; span_days=`117`)**

- **Back (ROI por stake)**

| Bucket slippage_raw_pct | n | ROI mean (SE; IC95) |
|---|---:|---:|
| <= -2% | 649 | 35.82% (SE 3.74%) [28.48%, 43.15%] | ROIw 29.93% (odd~1.91, exp~3.00) |
| (-2, 2] | 3030 | 22.28% (SE 1.68%) [18.98%, 25.58%] | ROIw 21.14% (odd~1.94, exp~3.00) |
| > 2% | 664 | 50.69% (SE 8.40%) [34.22%, 67.16%] | ROIw 46.05% (odd~2.03, exp~3.00) |

- **Back Pre (ROI por stake)**

| Bucket slippage_raw_pct | n | ROI mean (SE; IC95) |
|---|---:|---:|
| <= -2% | 112 | 14.06% (SE 8.00%) [-1.61%, 29.74%] | ROIw 19.83% (odd~1.88, exp~10.00) |
| (-2, 2] | 1237 | 16.37% (SE 2.52%) [11.44%, 21.30%] | ROIw 19.09% (odd~1.94, exp~12.00) |
| > 2% | 44 | 4.84% (SE 13.71%) [-22.04%, 31.72%] | ROIw 16.83% (odd~1.98, exp~12.00) |

- **Back In (ROI por stake)**

| Bucket slippage_raw_pct | n | ROI mean (SE; IC95) |
|---|---:|---:|
| <= -2% | 537 | 40.35% (SE 4.18%) [32.16%, 48.54%] | ROIw 39.83% (odd~1.92, exp~3.00) |
| (-2, 2] | 1793 | 26.36% (SE 2.25%) [21.95%, 30.77%] | ROIw 27.58% (odd~1.94, exp~3.00) |
| > 2% | 620 | 53.94% (SE 8.93%) [36.43%, 71.45%] | ROIw 56.26% (odd~2.04, exp~3.00) |

- Nota: `ROIw` é o **ROI ponderado por exposição** (peso=stake no Back; peso=liability no Lay). Em prática, dentro de um bucket, `ROIw ≈ (∑P&L)/(∑exposição)`; já o `ROI mean` é a média simples por linha/sinal.

- Nota importante (reconciliação): as tabelas **Slippage × ROI** usam **somente execuções cobertas por ROI via placar** (precisa audit+placar+odd). Isso é um subconjunto e pode ter viés (ex.: jogos ainda não liquidaram, falta de odds finais, etc.). Já o **accounting ledger** inclui todo o resultado financeiro (incluindo void/refund/cancel quando existirem) por `post date`.

**Contrafactual (placar): aplicar filtro de slippage**

- Regra: **Back** pula `slippage_raw_pct <= -2%`; **Lay** pula `slippage_raw_pct > 2%`.
- Observação: usa somente execuções com ROI via placar; não é o P&L do accounting.

| Lado | n (base) | P&L (base) | Exposição (base) | n (após filtro) | P&L (após) | Exposição (após) |
|---|---:|---:|---:|---:|---:|---:|
| Back | 4343 | 5,688.59 | 23,323.50 | 3694 | 4,930.88 | 20,791.50 |
| Lay (liab) | 0 | 0.00 | 0.00 | 0 | 0.00 | 0.00 |
| **Total** | — | 5,688.59 | — | — | 4,930.88 | — |

**Diagnóstico AH (linha) observado na execução**

- Policy: `ah_max_abs_line=0.0` | `ah_scope=all`
- Execuções (todas): `n=8938` | `max|line|=10.00` | `n_over=7684`
- Execuções com placar/ROI: `n=6276` | `max|line|=10.00` | `n_over=5344`

**Slippage × ROI por combinação (top 2 por volume; acumulado)**

- **Back**

| Combinação | n | ROI<=-2% | n | ROI(-2..2] | n | ROI>2% | n | corr(slip_raw,ROI) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Back_In_Any | 2950 | 40.35% (SE 4.18%) [32.16%, 48.54%] | 537 | 26.36% (SE 2.25%) [21.95%, 30.77%] | 1793 | 53.94% (SE 8.93%) [36.43%, 71.45%] | 620 | 0.03 |
| Back_Pre_Any | 1393 | 14.06% (SE 8.00%) [-1.61%, 29.74%] | 112 | 16.37% (SE 2.52%) [11.44%, 21.30%] | 1237 | 4.84% (SE 13.71%) [-22.04%, 31.72%] | 44 | 0.02 |

**Funil de oportunidades (últimas 24h; auditoria DB)**

| audit_version | total | OK | OK_valid | GATE_NOT_ELIGIBLE | API_FAILED | STALE_QUEUE_WAIT |
|---|---:|---:|---:|---:|---:|---:|
| v5.3-ws-gate-back | 4176 | 4176 | 4176 | 0 | 0 | 0 |
| v1.0 | 629 | 0 | 0 | 0 | 0 | 0 |

**Oportunidades identificadas / melhorias propostas (curto prazo)**

- **PMM/timeout**: se `No PMMs received` dominar, aumentar timeout efetivo e reduzir bursts (workers/queue) tende a elevar conversão sem mexer na estratégia.
- **Betslips abertos**: `too_many_open_betslips` é um gargalo de throughput; manter caps/janelas e garantir cleanup rápido evita bloqueio global.
- **Fila**: `STALE_QUEUE_WAIT` indica atraso interno; atacar latência/concorrência antes de aumentar volume/seleção.

**Portfólio OOS: vigente vs histórico recente**

| ts | n_active_keys |
|---|---:|
| 2026-06-29T11:30:03.161126+00:00 | — |
| 2026-06-29T11:45:03.568376+00:00 | — |
| 2026-06-29T12:00:03.170738+00:00 | — |
| 2026-06-29T12:15:04.115571+00:00 | — |
| 2026-06-29T12:30:02.651953+00:00 | — |
| 2026-06-29T12:45:03.010166+00:00 | — |
| 2026-06-29T13:00:05.927827+00:00 | — |
| 2026-06-29T13:15:03.295033+00:00 | — |

**Parâmetros vigentes (visão executiva)**

- **Combinações ativas (OOS)**: ver `99.3` (active_keys) e o bloco `2) OOS`.
- **Stake sizing operacional (real)**: hoje é **FLAT** via `BRIDGE_STAKE` (ver `99.3` e `99.6`).
- **Parâmetros técnicos efetivos** (executor/audit/bridge): ver `99.6 Filtros ativos`.

**Critérios de seleção (OOS) e critérios do real (bridge/executor)**

- **OOS (walk-forward)** decide o portfólio `active_keys`.
  - **Chave por liga**: `True` (scope=`pre`) ⇒ em pre-match a chave pode virar `...__<League>`.
  - **Filtro de AH ativo?**: `False` (max_abs_line=`0.00`; scope=`pre`) ⇒ remove eventos com `abs(line)` acima do limiar.
  - **Mínimo de jogos no treino**: `wf_min_matches=3` (0 = desligado).
  - **Regra de decisão (por combinação, no treino)**:
    - Se `ROI` for **significativamente negativo** (IC90 inteiro < 0): **bloqueia**.
    - Se `ROI` for **significativamente positivo** (IC90 inteiro > 0): **ativa**.
    - Se `ROI` > 0 mas **não significativo**:
      - **Pre-match**: ativa apenas se **CLV > 0** (CLV não precisa ser sig.).
      - **In-match**: ativa se **ROI > 0** (CLV não se aplica).
  - Operacionalmente, o OOS também pode excluir buckets de execução (ex.: `wf_exclude_exec_buckets_back`).
- **Real (shadow/live)**:
  - O bridge só envia oportunidades cuja chave esteja em `active_keys` (policy current).
  - `DRY_OK` = **shadow** (não apostou); `LIVE_OK` = **efetivo** (apostou).

**Este período está rodando shadow ou efetivo?**

- Predominantemente **efetivo**: `LIVE_OK=209` (e `DRY_OK=0`).

**Aspectos técnicos (latência/estabilidade)**

- Latência detalhada: ver `99.2` (p50/p90/p99 por etapa).
- Gaps no `executor_jsonl` (proxy de downtime/restart/sem tráfego): max `107,022.3s`, gaps>5min `1`, gaps>15min `1`.

## 3) In-sample (detalhe)

# Análise Estatística Robusta — Contexto Operação (b808)
**Data da execução:** 29/07/2026 22:05 UTC  
**Escopo:** H3B (auditoria), com comparação por `audit_version` e inferência robusta por jogo (cluster bootstrap).  
**Nota:** a robustez aqui significa que intervalos de confiança consideram correlação intra-jogo (múltiplas auditorias por partida).
---
### 0) Sumário executivo (leitura rápida)
- **Recorte**: direction=`up`, lookback_days=`35`, versions=`v5.2-api-back,v4.0-api,v5.3-ws-gate-back`.
- **Amostra**: 0 auditorias (jogos únicos=0, média=0.0 obs/jogo); betslip confiável=0.
- **Dias excluídos / missing** (UTC, não tratados como 0): manual=0 [—]; auto(ws-only sem Lay)=31 [2026-06-24, 2026-06-25, 2026-06-26, 2026-06-27, 2026-06-28, 2026-06-29, 2026-06-30, 2026-07-01 ... (+23)]; auto(sem BS/WS/Lay)=0 [—]; missing(sem dados)=5 [2026-07-11, 2026-07-12, 2026-07-13, 2026-07-14, 2026-07-15].
- **Coortes (status=OK, betslip confiável)**: `BS>WS` (diff>=2.0%): **0**; `BS<WS` (diff<=-2.0%): **0**.
- **Coberturas em `hypothesis_details` (OK)**: temporal(BS)=0/0; lay_temporal(BS)=0/0; ws_series(WS)=0/0; finance=0/0.
- **Cobertura de placar (ROI)**: jogos com placar=0/0 (status finished=0).
- **DOM**: sem dados no recorte atual (N=0), então não há comparação API vs DOM aqui.
- **CLV pre-match (Betslip, API)**: média robusta por jogo — (IC90 —), com N=0 eventos (jogos=0).
- **Padrão por bucket (CLV PM)**: `BS < WS` — (N/A), `BS ~ WS` — (N/A), `BS > WS` — (N/A).
- **ROI**: sem cobertura no recorte (N=0). Isso normalmente acontece quando os placares ainda não foram sincronizados para esses jogos.
- **Leitura recomendada**: use este relatório para validar **qualidade de execução/CLV**. Para concluir sobre **ROI**, rode após atualizar resultados e/ou use uma janela com jogos já liquidados.

---


## 99) Operacional — saldo, P&L e execução

### 99.1 Accounting (saldo + P&L)

- Arquivo: `logs/daily_reports/20260729/accounting_daily_report.json`
- Saldo atual: **1109.42**
- P&L hoje/semana/mês: **-6.510000000000003 / -6.940000000000005 / -173.82999999999987**

Meses fechados:

| Mês | P&L |
|---|---:|
| 2026-01 | 58.0 |
| 2026-02 | -7.61 |
| 2026-03 | 1676.53 |
| 2026-04 | -381.4300000000007 |
| 2026-05 | 211.13000000000008 |
| 2026-06 | -273.96000000000004 |

### 99.2 Execução (KPIs)

- Fonte: `logs/executor_live.jsonl`
- Nota: métricas abaixo vêm do JSONL; se ele estiver **stale** ou incompleto, podem divergir do volume “24h, DB”.

**Status (all)**

| Status | N |
|---|---:|
| CAP_BLOCKED | 1030 |
| LIVE_OK | 209 |
| API_FAILED | 132 |
| STALE | 31 |
| NO_SESSION | 25 |

**Latência (somente LIVE_OK/DRY_OK) — ms**

| Métrica | n | p50 | p90 | p99 | mean |
|---|---:|---:|---:|---:|---:|
| queue_delay | 209 | 0.0 | 2.0 | 5769.239999999992 | 202.63636363636363 |
| call_to_done | 209 | 5139.0 | 8771.400000000005 | 18795.399999999983 | 6157.727272727273 |
| post | 209 | 837.0 | 2495.2000000000057 | 7103.359999999985 | 1343.9186602870814 |

**Latência (últimas 24h; somente LIVE_OK/DRY_OK) — ms**

| Métrica | n | p50 | p90 | p99 | mean |
|---|---:|---:|---:|---:|---:|
| queue_delay | 23 | 0.0 | 1.8000000000000007 | 35.54000000000005 | 2.4347826086956523 |
| call_to_done | 23 | 5578.0 | 8900.0 | 10155.2 | 5930.347826086957 |
| post | 23 | 821.0 | 1708.6 | 4084.3600000000015 | 1198.9130434782608 |

**Slippage (somente LIVE_OK/DRY_OK, quando houver odd_at_decision)**

- Definição: `slippage = odd_final - odd_at_decision` (em odds decimais) e `slippage_pct = slippage/odd_at_decision`.
- Interpretação depende do lado:
  - **Back**: slippage_pct **negativo** = piorou (odd caiu); **positivo** = melhorou.
  - **Lay**: slippage_pct **positivo** = piorou (odd subiu); **negativo** = melhorou.

| Tipo | n | p50 | p90 | p99 | mean |
|---|---:|---:|---:|---:|---:|
| abs | 209 | -0.010999999999999677 | -0.0020000000000000018 | 0.005519999999999726 | -0.02326315789473685 |
| pct | 209 | -0.5344995140913351 | -0.1095217913399732 | 0.2948717948717802 | -1.1440090079510126 |

**Slippage por lado (Back vs Lay)**

| Lado | Métrica | n | p50 | p90 | p99 | mean |
|---|---|---:|---:|---:|---:|---:|
| Back | slippage_pct (raw) | 209 | -0.5344995140913351 | -0.1095217913399732 | 0.2948717948717802 | -1.1440090079510126 |
| Back | slippage_pct (custo, >=0) | 209 | 0.5344995140913351 | 3.291868118972298 | 8.741215027203296 | 1.249112808885861 |

_Nota: o p90/p99 de `call_to_done_ms` explode quando inclui `NO_SESSION/API_FAILED` (timeouts/relogin). Por isso reportamos também o recorte apenas de sucessos._


### 99.6 Filtros ativos (config efetiva)

_Nota: esta seção reflete as variáveis carregadas pelo `daily_full_report` (via `.env`). Services do systemd podem ter overrides (`Environment=`) que não aparecem aqui; use `systemctl show` para confirmar no VPS._

**Executor**

| chave | valor |
|---|---|
| EXECUTOR_ALLOW_LIVE | `1` |
| EXECUTOR_WORKERS | `2` |
| EXECUTOR_QUEUE_MAX | `200` |
| EXECUTOR_CAP_WINDOW_SEC | `` |
| EXECUTOR_CAP_MAX | `` |
| EXECUTOR_BACKPRE_FAST_STAKE_ENABLE | `0` |
| EXECUTOR_BACK_STAKE_SIZING_ENABLE | `0` |
| EXECUTOR_BACK_STAKE_SLIP_NEG_LIMIT_PCT | `` |
| EXECUTOR_BACK_STAKE_SLIP_POS_LIMIT_PCT | `` |
| EXECUTOR_BACK_STAKE_SLIP_NEG | `` |
| EXECUTOR_BACK_STAKE_SLIP_MID | `` |
| EXECUTOR_BACK_STAKE_SLIP_POS | `` |
| EXECUTOR_BACK_LATENCY_GATE_ENABLE | `` |
| EXECUTOR_BACK_LATENCY_GATE_MAX_SEC | `` |
| EXECUTOR_FAST_PMM | `1` |
| EXECUTOR_PMM_TIMEOUT_SEC | `1.2` |
| EXECUTOR_PMM_MIN_WAIT_SEC | `0.10` |
| EXECUTOR_PMM_IDLE_TIMEOUT_SEC | `0.25` |
| EXECUTOR_BETSLIP_CACHE_MAX_KEYS | `0` |

**Audit H3B**

| chave | valor |
|---|---|
| AUDIT_MODE | `ws_gate_back` |
| AUDIT_API_SIDES | `back` |
| AUDIT_EXECUTOR_WORKERS | `4` |
| AUDIT_TEMPORAL_WORKERS | `1` |
| AUDIT_MAX_QUEUE_DEPTH | `` |
| AUDIT_MAX_QUEUE_WAIT_MS | `` |
| WS_SAMPLE_OFFSETS_SEC | `` |
| GATE_DROP_OFFSET_SEC | `` |
| GATE_DROP_RATIO | `0.995` |
| GATE_RISE_OFFSET_SEC | `5` |
| GATE_RISE_RATIO | `1.02` |
| GATE_OPEN_WINDOW_SEC | `` |
| GATE_OPEN_MAX | `10` |
| GATE_MAX_LATE_SEC | `0` |
| GATE_LAY_REFRESH_TIMES_SEC | `` |

_Nota (Back vs Lay): o `AUDIT_MODE` acima costuma refletir o serviço principal (ex.: `ws_gate_lay`). Em operação real, o **Back** pode vir de um serviço separado (ex.: `betinasia-audit-api-back`, `audit_version=v5.2-api-back`) ou de uma variante `ws_gate_back` (dependendo do deploy). Para confirmar o que rodou nas últimas 24h, veja `99.5 Auditoria (DB)`._

**Interpretação operacional (timing de entrada)**

| Item | Regra efetiva |
|---|---|
| Back (mais cedo possível) | Depende do executor: `EXECUTOR_FAST_PMM`, `EXECUTOR_PMM_MIN_WAIT_SEC`, `EXECUTOR_PMM_TIMEOUT_SEC` (ver tabela Executor). |
| Lay (reversão vs fim) | Depende do `AUDIT_MODE`/audit_version: `ws_gate_lay` abre Lay só quando o gate em `t+GATE_DROP_OFFSET_SEC` passa; `ws_reversal_lay` tende a entrar no pós-reversal; `ws_only` usa a série WS (offsets até o último ponto, tipicamente 30s). |

**Bridge**

| chave | valor |
|---|---|
| BRIDGE_MODE | `live` |
| BRIDGE_EXEC_SIDE | `Back` |
| BRIDGE_STAKE | `10` |
| BRIDGE_POLL_SEC | `0.3` |
| BRIDGE_LOOKBACK_SEC | `120` |
| BRIDGE_MAX_PER_CYCLE | `10` |
| BRIDGE_PREMATCH_ONLY | `1` |
| BRIDGE_POLICY_JSON | `logs/wf_policy_current.json` |
| BRIDGE_POLICY_RELOAD_SEC | `5.0` |
| BRIDGE_POLICY_USE_BASE | `0` |
| BRIDGE_MIN_LIMIT | `0` |

**OOS / Walk-forward (daily)**

| chave | valor |
|---|---|
| DAILY_OOS_DIRECTION | `up` |
| DAILY_OOS_VERSIONS | `v5.2-api-back,v4.0-api,v5.3-ws-gate-back` |
| DAILY_OOS_LOOKBACK_DAYS | `35` |
| DAILY_WF_TRAIN_MODE | `expanding` |
| DAILY_WF_TRAIN_DAYS | `30` |
| DAILY_WF_TEST_DAYS | `7` |
| DAILY_WF_STEP_DAYS | `7` |
| DAILY_WF_SIDES | `back` |
| DAILY_WF_REGIMES | `pre` |
| DAILY_WF_BACKPRE_SLIP_MAX | `0` |
| DAILY_WF_BACKPRE_SLIP_FIELD | `diff_pct` |
| DAILY_WF_BACKPRE_FAST_MAX_LAG_MS | `` |
| DAILY_WF_KEY_BY_LEAGUE | `1` |
| DAILY_WF_KEY_BY_LEAGUE_SCOPE | `` |
| DAILY_WF_AH_MAX_ABS_LINE | `` |
| DAILY_WF_AH_SCOPE | `pre` |
| DAILY_WF_EXCLUDE_EXEC_BUCKETS_BACK | `` |


### 99.4 Aderência OOS (portfolio por dia × execução)

- Arquivo (curto): `logs/daily_reports/20260729/oos_adherence_short.json`
- Arquivo (acumulado/slippage): `logs/daily_reports/20260729/oos_adherence_long.json`
- Policy current: `logs/wf_policy_current.json`

**Resumo (últimos dias)**

| Dia | Ativas (keys) | Bridge rows | Skipped(not_active) | Exec rows | LIVE_OK | DRY_OK | Back bloqueadas (slip<=-2%; cov) | Lay bloqueadas (slip>2%; cov) | ΔP&L cf (placar; cov) | P&L Back | ROI Back | P&L Lay | ROI Lay/liab | P&L total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-07-23 | 22 | 1663 | 0 | 68 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | — | 0.00 | — | 0.00 |
| 2026-07-24 | 22 | 742 | 0 | 167 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | — | 0.00 | — | 0.00 |
| 2026-07-25 | 22 | 402 | 0 | 111 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | — | 0.00 | — | 0.00 |
| 2026-07-26 | 22 | 212 | 0 | 38 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | — | 0.00 | — | 0.00 |
| 2026-07-27 | 22 | 811 | 0 | 52 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | — | 0.00 | — | 0.00 |
| 2026-07-28 | 22 | 851 | 0 | 23 | 3 | 0 | 0 | 0 | 0.00 | 26.84 | 89.47% | 0.00 | — | 26.84 |
| 2026-07-29 | 22 | 1500 | 0 | 176 | 22 | 0 | 3 | 0 | -7.21 | 7.21 | 24.03% | 0.00 | — | 7.21 |

**Slippage × ROI (raw, com sinal; 3 buckets) — acumulado (range: `2026-03-20` → `2026-07-29`; span_days=`132`; cut=`2026-03-20`)**

- **Back (ROI por stake)**

| Bucket slippage_raw_pct | n | ROI mean |
|---|---:|---:|
| <= -2% | 684 | 37.10% |
| (-2, 2] | 3199 | 21.70% |
| > 2% | 732 | 53.94% |

- **Lay (ROI por liability)**

| Bucket slippage_raw_pct | n | ROI mean |
|---|---:|---:|
| <= -2% | 2 | -100.00% |
| (-2, 2] | 0 | — |
| > 2% | 0 | — |


### 99.5 Auditoria (DB) — motivos de no-OK (por versão)

- Arquivo: `logs/daily_reports/20260729/audit_status_kpis.json`
- Janela: últimas **24.0h** (desde `2026-07-28T22:05:31.546151+00:00`)

**Definições (colunas)**

- **OK**: `status='OK'` no `betslip_audit_results` (a auditoria concluiu com sucesso).
- **OK com betslip_odd**: subset de OK em que `betslip_odd` está preenchido (houve snapshot do ticket/odds).
- **OK valid**: subset de OK em que `is_valid_opportunity=true` (passou o critério operacional de “oportunidade executável”).
  - Na prática, o `is_valid_opportunity` tende a cair quando `difference_pct` está fora do range aceito (edge muito pequeno <2% ou mismatch >10%) ou quando campos essenciais do ticket estão ausentes.

**Glossário rápido (`audit_version`)**

| padrão | significado |
|---|---|
| `v5.2-api-back` | Back via API (serviço back-only); tende a abrir betslip e medir limites/odds. |
| `v5.1-ws-gate-lay` | Lay via WS gate (queda em 5s); só abre ticket quando o gate passa. |
| `v5.4-ws-reversal-lay` | Lay no pós-reversal; volume baixo pode ser “evento raro” (depende de reversões). |
| `v5.3-ws-gate-back` | Back via WS gate; se `OK` é baixo, costuma indicar gate muito restritivo, parse/click falhando, ou credenciais/sessão instável. |
| `v4.*` / `v1.*` | versões antigas/legadas do pipeline (API/WS), úteis para comparação histórica. |

| audit_version | total | OK | OK com betslip_odd | OK valid | top no-OK |
|---|---:|---:|---:|---:|---|
| v5.3-ws-gate-back | 4176 | 4176 | 0 | 4176 | — |
| v1.0 | 629 | 0 | 0 | 0 | LINE_NOT_AVAILABLE=429, GAME_NOT_FOUND=190, MAJOR_DIFF=10 |

**Diagnóstico dos OK (por versão): buckets de |difference_pct|**

_Leitura: `OK valid` tende a ser aproximadamente o bucket `2% ≤ |difference_pct| ≤ 10%` (dependendo da regra vigente)._

| audit_version | OK diff nulo | OK |diff|<2% | OK 2–10% | OK |diff|>10% |
|---|---:|---:|---:|---:|
| v5.3-ws-gate-back | 4176 | 0 | 0 | 0 |
| v1.0 | 0 | 0 | 0 | 0 |

## Anexo A) OOS walk-forward (Seção 12)

### 1) OOS walk-forward (expanding window): seleção e validação
Este relatório é **OOS-first**: começamos pelo walk-forward (OOS) e deixamos as análises in-sample/diagnósticos no apêndice.

- **Pre activation mode**: `roi_only` (`roi_clv` = ROI+CLV no pre; `roi_only` = somente ROI no pre).

- **ROI mínimo de ativação**: `ROI > 2.00%`.

**Regras de entrada (OOS, alinhadas ao robô atual)**:
- Back: `WS@t+5.0s` (gap máx 2.5s)
- Lay: pós-reversal quando existir; senão `~WS@t+30.0s` (gap máx 12.0s)

**Filtro operacional (OOS)**: excluindo exec_bucket apenas no walk-forward (Back=['10-20s']; Lay=—).

### 1.0 Diagnóstico de cobertura OOS (por que N cai)
| Filtro | Jogos únicos |
|---|---:|
| Combinações elegíveis (edge + timing + t0) | 0 |
| Com ROI disponível (precisa de placar) | 0 |
| Com CLV disponível (pre-match + closing) | 0 |

**Calendário do walk-forward (dias únicos)**

| Tipo | Dias |
|---|---:|
| Dias com dados carregados (audited_at) | 0 |
| Dias com eventos OK (qualquer versão, incl. ws-only) | 0 |
| Dias com eventos elegíveis p/ WF (edge) | 0 |
| Dias usados no walk-forward | 0 |

**Diagnóstico por dia (audited_at): cobertura WS vs edge (alinhado ao robô atual)**

| Dia | Auditorias carregadas | OK (total) | Back: WS proxy ok | Lay: série ok | Edge Back/Lay | Edge Pre/In | Status não-OK dominante |
|---|---:|---:|---:|---:|---:|---:|---|

Leitura:
- **Back: WS proxy ok** mede cobertura da regra operacional de entrada (`WS@t+x`, ex.: x=5s).
- **Lay: série ok** mede se há dados suficientes para aplicar a regra de entrada (pós-reversal ou ~fim do período).
- **Edge Back/Lay** é contado a partir do universo OOS (já usando WS/entrada efetiva), então não deve ser interpretado via betslip.


Leitura: o walk-forward mede OOS principalmente por **ROI**, então ele encolhe quando a cobertura de placar é baixa. Além disso, a métrica é agregada por **jogo único** (cluster), então você verá números menores que o N de eventos.

**Parâmetros efetivos (WF)**: dias_unicos=0 | wf_train_days=30 | wf_test_days=7 | wf_step_days=7 | only_oos=OFF

[WARN] Janela curta para walk-forward: dias únicos=0; precisa >= 37.

**Regra de elegibilidade (todas as combinações):** exige `N_ROI >= wf_min_matches` (aqui: 3).


---

### Apêndice — Diagnósticos e in-sample

_Nota: as seções abaixo mantêm a numeração original do relatório completo._

### 1) Contexto do corte (b808)
| Indicador | Valor |
|---|---:|
| Auditorias H3B `UP` (match + kickoff passado) | 0 |
| Betslip bruto | 0 |
| Betslip confiável (diff -10% a +10%) | 0 |
| Descartados no filtro de qualidade | 0 |
| Jogos únicos (geral) | 0 |
| Média de observações por jogo | 0.0 |
| Jogos únicos com betslip confiável | 0 |
| Distribuição por market_type |  |
| Jogos únicos (AH) no recorte | 0 |
| Jogos únicos (AH) com closing_odd disponível | 0 |
| Cobertura closing_odd (AH) | —% |

---
### 2) Base comparativa: API vs DOM
| Métrica | API (v4.0-api) | DOM (v1.0) |
|---|---:|---:|
| Total de observações | 0 | 0 |
| Com betslip confiável | 0 | 0 |
| Com CLV pre-match (betslip) | 0 | 0 |
| Com ROI (betslip) | 0 | 0 |
| Tempo total observado (detecção→betslip, wall/total) | — ms | — ms |
| Tempo instrumentado (detecção→clique→betslip) | — ms | — ms |

---
### 2.0a Glossário de métricas (definições operacionais)
Este glossário existe para eliminar ambiguidades entre **tempo total**, **tempos instrumentados** e **overhead**.

- **`hypothesis_detected_at`**: timestamp (UTC) de detecção do evento que gerou a auditoria.
- **`audited_at`**: timestamp (UTC) em que a auditoria foi concluída/persistida.
- **`lag_total_ms` (tempo total observado / wall)**: proxy de tempo “de parede” do pipeline do evento até o betslip; quando disponível usa wall time (ex.: `audited_at - detected_at`).
- **`lag_det_to_click_ms` (detecção→clique)**: tempo até o robô executar o clique/ação de betslip.
- **`lag_click_to_betslip_ms` (clique→betslip)**: tempo até carregar/obter o payload do betslip após o clique.
- **`lag_e2e_ms` (tempo instrumentado)**: `lag_det_to_click_ms + lag_click_to_betslip_ms`.
- **`audit_total_ms` (duração da auditoria)**: duração instrumentada do ciclo de auditoria (pode diferir de `lag_total_ms` se houver esperas fora do escopo instrumentado).
- **`lag_overhead_ms` (overhead)**: `lag_total_ms - lag_e2e_ms`; agrega espera fora das duas etapas instrumentadas (ex.: fila, retries, pausas, latência externa).
- **`diff_pct` (BS vs WS)**: diferença percentual entre a odd do **betslip no momento da execução** (BS) e a odd do **WebSocket no momento da detecção** (WS): `(BS - WS) / WS * 100`. Importante: **BS e WS são medidos em instantes diferentes**, então este número mede principalmente **drift durante a execução + slippage/atualização** (e não “mispricing contemporâneo”).
- **Betslip confiável**: filtro de qualidade `diff_pct ∈ [-10%, +10%]` para reduzir casos de mismatch/parse incorreto.

---
### 2.0b Decomposição do tempo (detecção→clique→betslip vs. overhead)
Interpretação: `lag_e2e` é o **tempo fim-a-fim** (detecção→clique + clique→betslip). `overhead` = `lag_total` − `lag_e2e` (proxy de fila/retries/esperas fora das 2 etapas instrumentadas).

| Modelo | Métrica | mean (ms) | p50 (ms) | p95 (ms) | N |
|---|---|---:|---:|---:|---:|
| API (2-4s) | lag_det→click | — | — | — | 0 |
| API (2-4s) | lag_click→betslip | — | — | — | 0 |
| API (2-4s) | lag_e2e (soma) | — | — | — | 0 |
| API (2-4s) | audit_total (duração) | — | — | — | 0 |
| API (2-4s) | overhead (total - e2e) | — | — | — | 0 |
| DOM (15-30s) | lag_det→click | — | — | — | 0 |
| DOM (15-30s) | lag_click→betslip | — | — | — | 0 |
| DOM (15-30s) | lag_e2e (soma) | — | — | — | 0 |
| DOM (15-30s) | audit_total (duração) | — | — | — | 0 |
| DOM (15-30s) | overhead (total - e2e) | — | — | — | 0 |

**Diagnóstico de cauda (percentual acima do limiar)**

| Modelo | % det→click > 5s | % det→click > 20s | % total > 10s | % total > 40s | % overhead < 0 |
|---|---:|---:|---:|---:|---:|
| API (2-4s) | —% | —% | —% | —% | —% |
| DOM (15-30s) | —% | —% | —% | —% | —% |

---
### 2.1 Cobertura temporal (pre-match vs in-match)
| Métrica | Pre-match | In-match | Observação |
|---|---:|---:|---|
| Observações totais com classificação temporal | 0 | 0 | Contagem bruta do corte |
| ROI Betslip | 0 | 0 | Amostra com resultado do jogo |
| ROI WebSocket | 0 | 0 | Referência de mercado |
| CLV (apenas pre-match) | 0 | — | CLV vs closing pré-jogo não é interpretável in-match |

---
### 2.2 Performance por regime (pre-match vs in-match)
| Regime | N auditorias | N betslip conf. | N OK | Back edge | Lay edge | Diff médio (OK) |
|---|---:|---:|---:|---:|---:|---:|
| PRE_MATCH | 0 | 0 | 0 | 0 | 0 | — |
| IN_MATCH | 0 | 0 | 0 | 0 | 0 | — |

---
### 2.2c Quebra por liga (top por volume)
Objetivo: detectar não-uniformidade do edge por **liga**. Reporta volume, cobertura de closing (para CLV) e métricas robustas por jogo.

_Sem dados OK+conf suficientes para quebrar por liga._

---
### 2.3 Regimes operacionais por tempo total (bucket)
Objetivo: separar a amostra em **regimes de execução** (tempo total) e medir performance em cada regime. Use isso para detectar **fila/saturação**: regimes lentos tendem a ter edge menor e pior ROI.

| Bucket (tempo total) | N OK | Jogos | lag_total mean (ms) | lag_total p95 (ms) | overhead p95 (ms) | Back edge | Lay edge | CLV PM (mean; IC90) | ROI (mean; IC90) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| < 5s | 0 | 0 | — | — | — | 0 | 0 | — | — |
| 5-10s | 0 | 0 | — | — | — | 0 | 0 | — | — |
| 10-20s | 0 | 0 | — | — | — | 0 | 0 | — | — |
| 20-40s | 0 | 0 | — | — | — | 0 | 0 | — | — |
| > 40s | 0 | 0 | — | — | — | 0 | 0 | — | — |
| Desconhecido | 0 | 0 | — | — | — | 0 | 0 | — | — |

---
### 2.3b Regimes por tempo total — separando coortes por `diff_pct` (BS vs WS)
Nesta tabela, separamos duas coortes operacionais por **delta de execução**: `BS > WS` (diff_pct >= +2%) e `BS < WS` (diff_pct <= -2%). Isso **não** é (por si só) “Back vs Lay”; é um recorte por **melhora/piora do preço** entre detecção (WS) e execução (BS). CLV é reportado apenas em pre‑match.

| Bucket | N OK | N (BS>WS +2%) | N (BS<WS -2%) | CLV PM (BS>WS) | CLV PM (BS<WS) | ROI (BS>WS) | ROI (BS<WS) |
|---|---:|---:|---:|---:|---:|---:|---:|
| < 5s | 0 | 0 | 0 | — | — | — | — |
| 5-10s | 0 | 0 | 0 | — | — | — | — |
| 10-20s | 0 | 0 | 0 | — | — | — | — |
| 20-40s | 0 | 0 | 0 | — | — | — | — |
| > 40s | 0 | 0 | 0 | — | — | — | — |
| Desconhecido | 0 | 0 | 0 | — | — | — | — |

---
### 2.3c Estabilidade temporal (por dia, `audited_at`)
Objetivo: checar se o regime de edge/execução é **time‑dependent**. Se houver dias com comportamento distinto (ex.: collector intermitente, horários/ligas), isso aparecerá aqui.

| Dia (UTC) | Modelo | N OK | Jogos | % BS>WS +2% | % BS<WS -2% | lag_total p50 (ms) | CLV PM (BS>WS) | CLV PM (BS<WS) |
|---|---|---:|---:|---:|---:|---:|---:|---:|

---
### 3) CLV pre-match (núcleo)
### 3.1 CLV com odd do Betslip (execução real)
| Métrica | API | DOM |
|---|---:|---:|
| CLV Bruto BS Pre-Match | — (N/A, N=0, jogos=0) | — (N/A, N=0, jogos=0) |
| CLV Adicional BS Pre-Match | — (N/A, N=0, jogos=0) | — (N/A, N=0, jogos=0) |
| Taxa de CLV > 0 (bruto) | —% | —% |
| Taxa de CLV > 0 (adicional) | —% | —% |

Notas de robustez (IC 90% por jogo):  
- API CLV bruto (cluster): média —; IC90 —  
- DOM CLV bruto (cluster): média —; IC90 —  

---
### 4) ROI por modelo
| Métrica | API | DOM |
|---|---:|---:|
| ROI Betslip | — (N/A, N=0) | — (N/A, N=0) |
| ROI WebSocket | — (N/A, N=0) | — (N/A, N=0) |
| Win rate ROI Betslip | —% | —% |
| Win rate ROI WS | —% | —% |

Notas de robustez (IC 90% por jogo):  
- API ROI betslip (cluster): média —; IC90 —  
- API ROI WS (cluster): média —; IC90 —  

---
### 4.1) Validade do CLV: relação CLV × ROI (pre-match)
Objetivo: avaliar se **CLV** (vs closing) é um bom proxy de **ROI realizado** (por placar), ao menos no regime **pre‑match**.

Regras do recorte desta seção:

- Apenas `status=OK` com betslip confiável (diff ∈ [-10%, +10%])
- Apenas `PRE_MATCH` (`is_live=False`)
- Exige **closing_odd** (para CLV) e **placar** (para ROI)

### 4.1a Estatística global (por jogo)
| Métrica | Valor |
|---|---:|
| Jogos com CLV+ROI | 0 |
| Eventos (auditorias) usados | 0 |
| Correlação Pearson (mean por jogo) | — |
| Correlação Spearman (mean por jogo) | — |

### 4.1b Concordância de sinal (CLV vs ROI)
| CLV (jogo) | ROI (jogo) | Jogos |
|---|---|---:|
| > 0 | > 0 | 0 |
| > 0 | ≤ 0 | 0 |
| ≤ 0 | > 0 | 0 |
| ≤ 0 | ≤ 0 | 0 |

Leitura: CLV e ROI podem divergir por **variância do resultado** (ROI) e por **missingness** (jogos sem closing/sem placar). A correlação acima é um diagnóstico de “alinhamento”, não causalidade.

### 4.1c ROI por bucket de CLV (quintis; por jogo)
Amostra insuficiente (jogos com CLV+ROI < 10) para buckets estáveis.


---
### 5) Diferença de preço BS vs WS
| Métrica | API | DOM |
|---|---:|---:|
| Diff BS vs WS (média) | — (N/A, N=0) | — (N/A, N=0) |
| BS > WS | —% (0/0) | —% (0/0) |
| BS > WS +2% | —% (0/0) | —% (0/0) |

---
### 6) Combinações de valor
### 6.1 Buckets por diferença BS vs WS
Nota de leitura:
- `BS > WS (+2% a +10%)`: preço no betslip ficou **melhor** do que o WS observado na detecção (delta positivo entre instantes).
- `BS < WS (-10% a -2%)`: preço no betslip ficou **pior** do que o WS observado na detecção (delta negativo entre instantes).
- O ROI abaixo é calculado **dentro do bucket** (não mistura buckets), mas ainda é sensível à cobertura de placar.

| Bucket | N bucket | CLV BS PM (média) | IC90 (cluster) | N CLV PM | Jogos CLV PM | ROI BS (todos) | IC90 (cluster) |
|---|---:|---:|---|---:|---:|---:|---|
| BS < WS (-10% a -2%) | 0 | — | — | 0 | 0 | — | — |
| BS ~ WS (-2% a +2%) | 0 | — | — | 0 | 0 | — | — |
| BS > WS (+2% a +10%) | 0 | — | — | 0 | 0 | — | — |

---
### 6.2 Combinação por faixa de linha AH
| Faixa AH | CLV BS PM (média) | IC90 (cluster) | ROI BS (todos) | IC90 (cluster) | Diff BS vs WS (média) |
|---|---:|---|---:|---|---:|
| AH 0-1 (líquida) | — | — | — | — | — |
| AH 1-2 (média) | — | — | — | — | — |
| AH 2+ (extrema) | — | — | — | — | — |

---
### 6.3 Combinação por faixa de lag
| Faixa de lag | CLV BS PM (média) | IC90 (cluster) | N CLV PM | Jogos CLV PM | ROI BS (todos) | IC90 (cluster) | Diff BS vs WS (média) |
|---|---:|---|---:|---:|---:|---|---:|
| < 10s | — | — | 0 | 0 | — | — | — |
| 10-20s | — | — | 0 | 0 | — | — | — |
| 20-30s | — | — | 0 | 0 | — | — | — |
| > 30s | — | — | 0 | 0 | — | — | — |

---
### 7) Estimativa financeira (proxy) e risco
Este bloco usa `hypothesis_details.finance` quando existe; se não existir, usa stake fallback = `stake_pct_of_limit × limit`.

**Política de stake (proxy)**

| Parâmetro | Valor |
|---|---:|
| stake_pct_of_limit | 0.25 |
| stake_cap | 0.00 |
| Cobertura finance (OK, betslip conf.) | 0/0 |

### 7.1 Back (BS >> WS)
| Métrica | Valor |
|---|---:|
| Corte (diff_pct) | >= 2.0% |
| N eventos | 0 |
| Cobertura finance (na coorte) | 0/0 |
| Stake total (estimado) | — |
| Stake médio | — |
| Profit_if_win total (estimado) | — |
| Profit_if_win médio | — |
| N com ROI realizado | 0 (placares ausentes no recorte) |

**Como ler as 3 linhas de ROI (Back)**

- **ROI realizado (ponderado por stake)**: ΣP&L / Σstake (pode ser dominado por stakes grandes).
- **ROI realizado (robusto por jogo)**: média por jogo (cada jogo pesa parecido; reduz dominância de outliers).
- **ROI ponderado por stake (robusto por jogo)**: calcula ROI ponderado dentro do jogo e depois faz média/IC por jogo.

Observação: a Seção 4 reporta ROI médio no **recorte completo** (betslip confiável). Aqui (7.1) é apenas a coorte **Back edge** (diff>=corte) e o ROI também é mostrado ponderado por stake; por isso sinais podem divergir.

### 7.2 Lay (BS << WS) — risco de cauda
| Métrica | Valor |
|---|---:|
| Corte (diff_pct) | <= -2.0% |
| N eventos | 0 |
| Cobertura finance (na coorte) | 0/0 |
| Stake total (estimado) | — |
| Liability total (estimada) | — |
| Liability média | — |
| Liability p95 | — |
| Liability p99 | — |
| ES95 (liability) | — |
| Liability max | — |
| Proxy de banca (>= p99 liability) | — |
| N com ROI realizado | 0 (placares ausentes no recorte) |

### 7.3 Projeção mensal (30 dias fixo) — turnover, lucro e banca
Premissas:
- **Mês = 30 dias fixo**.
- Turnover = soma de stakes (Back e Lay). Para Lay, também reportamos exposição por **liability**.
- **Banca por risco (unitária)** = p99/ES95(exposição por aposta).
- **Banca por liquidez (turnover)** = p95/p99 de capital **simultaneamente travado** (stake/liability em aberto até a liquidação), capturando distribuição desigual de entradas ao longo do tempo.
- Projeção de lucro usa duas visões: (i) **lucro/dia observado** e (ii) **ROI observado × turnover projetado**.

| Bloco | Janela(d) | Turnover 30d | Lucro 30d (direto) | Lucro 30d (ROI×turnover) |
|---|---:|---:|---:|---:|
| Back | 1.0 | 0.00 | — | — |
| Lay (stake) | 1.0 | 0.00 | — | — |
| Total (Back+Lay) | 1.0 | 0.00 | — | — |

**Banca por risco (exposição unitária)**

| Bloco | Banca conservadora (p99) | Banca agressiva (ES95) | ROI/banca 30d (direto) | ROI/banca 30d (ROI×turnover) |
|---|---:|---:|---:|---:|
| Back (stake) | — | — | —% | —% |
| Lay (liability) | — | — | —% | —% |
| Total (soma) | — | — | —% | —% |

**Banca por liquidez (capital simultaneamente travado)**

Definição operacional: cada aposta trava capital de `audited_at` até `kickoff + 2.00h + 2.25h` (grid=5min). A banca recomendada aplica buffer de +10%.

| Bloco | Liquidez mean | Liquidez p95 | Liquidez p99 | Liquidez max | Banca liq p99 (+buffer) |
|---|---:|---:|---:|---:|---:|
| Back (stake) | — | — | — | — | — |
| Lay (liability) | — | — | — | — | — |
| Total (Back+Lay) | — | — | — | — | — |

**Banca recomendada (conservadora)**

| Métrica | Valor |
|---|---:|
| Banca por risco (p99 unitário, soma) | — |
| Banca por liquidez (p99 simultâneo + buffer) | — |
| Banca efetiva (max das duas) | — |
| ROI/banca 30d (direto, banca efetiva) | —% |
| ROI/banca 30d (ROI×turnover, banca efetiva) | —% |
**Diagnóstico de cobertura (placar/ROI)**

| Bloco | Turnover total | Turnover com ROI | Cobertura turnover |
|---|---:|---:|---:|
| Back | 0.00 | 0.00 | —% |
| Lay | 0.00 | 0.00 | —% |

Notas (Lay): exposição 30d por liability (não é turnover) = 0.00; ROI realizado por liability (ponderado) = —%.

---
### 8) Curva temporal (pico, reversão e melhor timing)
Esta seção usa séries temporais coletadas em pontos discretos (t≈0,3,6,10,15,20s). Fontes possíveis:

- **BS-temporal (legado)**: `hypothesis_details.temporal` (Back) e `hypothesis_details.lay_temporal` (Lay)
- **WS-temporal (novo)**: `hypothesis_details.ws_series` (todos os t’s via WebSocket)

Para manter comparabilidade, nesta seção `diff_pct(t)` é sempre calculado contra o **WS do t0** (`ws_odd`): `(odd_t - ws_t0)/ws_t0*100`.

O objetivo é responder: **tempo até o pico/vale**, **% que segue melhorando até t_max**, **% com reversão** e **impacto esperado por timing**. **CLV é reportado somente pre-match** (closing pré-jogo).

### 8.1 Back (pico em diff_pct)
Definições: **pico** = `max(diff_pct)`. **reversão** = após o pico, `diff_pct` cair pelo menos 0.50 p.p. (para Lay, subir 0.50 p.p. após o vale). `t_reversão` é o 1º tempo que cruza esse limiar (logo, não é o pico).

| Regime | N | t_pico médio (s) | t_pico p50 (s) | % melhora até fim | % com reversão | t_reversão médio (s) | Δt (rev - pico) médio (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRE_MATCH | 0 | — | — | — | — | — | — |
| IN_MATCH | 0 | — | — | — | — | — | — |

**Partição 100% (Back)**: categorias exclusivas (somam 100% por regime).

| Regime | % pico no fim, sem reversão | % pico no fim, com reversão (recuperou) | % pico antes, com reversão | % pico antes, sem reversão relevante |
|---|---:|---:|---:|---:|
| PRE_MATCH | — | — | — | — |
| IN_MATCH | — | — | — | — |

**Curva média (Back)**: média de `diff_pct` e `odd` por tempo. `CLV` é reportado **somente pre-match** (closing pré-jogo). `ROI` (quando aparece) é o **ROI realizado** se a aposta fosse feita naquele ponto.

| Tempo | N pts | diff_pct médio | odd média | CLV médio (pre-match) | ROI médio (se houver) |
|---|---:|---:|---:|---:|---:|

### 8.1b Back — impacto por timing (t0 vs pico vs último) e reversão
Leitura: se a estratégia é **entrar no pico**, compare CLV/ROI no pico vs t0 e veja diferença entre **casos com reversão** vs **sem reversão**. Aqui reportamos **média robusta por jogo + IC90**. Observação: **CLV só é calculado pre-match**.

**CLV (somente pre-match) — média robusta por jogo (IC90)**

| Subcoorte | N total | N CLV (PM) | CLV t0 (mean; IC90) | CLV pico (mean; IC90) | CLV último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 0 | 0 | — — | — — | — — |
| COM_REVERSAO | 0 | 0 | — — | — — | — — |

**ROI (stake=1) — média robusta por jogo (IC90)**

| Subcoorte | N total | N ROI | ROI t0 (mean; IC90) | ROI pico (mean; IC90) | ROI último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 0 | 0 | — — | — — | — — |
| COM_REVERSAO | 0 | 0 | — — | — — | — — |

Nota interpretativa: a subcoorte **COM_REVERSAO** pode ter CLV maior no **pico** porque, por definição, ela inclui casos em que o edge atingiu um extremo mais alto antes de reverter. Para entender a hipótese 'sem reversão deveria ser maior', compare também a categoria '**pico no fim, sem reversão**' na partição 100% (tabela 8.1).

**Decomposição (pre-match): entry_odd vs closing_odd (IC90)**

| Subcoorte | N CLV (PM) | entry_odd t0 (mean; IC90) | entry_odd pico (mean; IC90) | closing_odd (mean; IC90) |
|---|---:|---:|---:|---:|
| SEM_REVERSAO | 0 | — — | — — | — — |
| COM_REVERSAO | 0 | — — | — — | — — |

### 8.2 Lay (vale em diff_pct)
| Regime | N | t_vale médio (s) | t_vale p50 (s) | % melhora até fim | % com reversão | t_reversão médio (s) | Δt (rev - vale) médio (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRE_MATCH | 0 | — | — | — | — | — | — |
| IN_MATCH | 0 | — | — | — | — | — | — |

**Partição 100% (Lay)**: categorias exclusivas (somam 100% por regime).

| Regime | % vale no fim, sem reversão | % vale no fim, com reversão (recuperou) | % vale antes, com reversão | % vale antes, sem reversão relevante |
|---|---:|---:|---:|---:|
| PRE_MATCH | — | — | — | — |
| IN_MATCH | — | — | — | — |

**Curva média (Lay)**: usa `odd=lay_odd`. `CLV` é reportado **somente pre-match** (closing pré-jogo). O ROI mostrado aqui é **ROI por liability** (não por stake), porque Lay é governado por risco.

| Tempo | N pts | diff_pct médio | lay_odd média | CLV médio (pre-match) | ROI/liability médio (se houver) |
|---|---:|---:|---:|---:|---:|

### 8.2b Lay — impacto por timing (t0 vs vale vs último) e reversão
Leitura: se a estratégia é **entrar no vale (odd mais baixa)**, compare CLV/ROI no vale vs t0 e veja diferença entre **casos com reversão** vs **sem reversão**. Aqui reportamos **média robusta por jogo + IC90**. Observação: **CLV só é calculado pre-match**.

**CLV (somente pre-match) — média robusta por jogo (IC90)**

| Subcoorte | N total | N CLV (PM) | CLV t0 (mean; IC90) | CLV vale (mean; IC90) | CLV último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 0 | 0 | — — | — — | — — |
| COM_REVERSAO | 0 | 0 | — — | — — | — — |

**ROI/liability — média robusta por jogo (IC90)**

| Subcoorte | N total | N ROI | ROI/liab t0 (mean; IC90) | ROI/liab vale (mean; IC90) | ROI/liab último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 0 | 0 | — — | — — | — — |
| COM_REVERSAO | 0 | 0 | — — | — — | — — |

---
### 8.3 Resumo de estratégias — combinações (Side × Pre/In × Reversal)
Esta tabela resume as combinações possíveis. Observação importante:

- **Back**: a estratégia é **entrar rápido em `t0`**, então **não faz sentido separar por Reversal(Sim/Não)** (agregamos como `Any`).
- **Lay**: entrada **após reversão** quando ela existe (`odd_reversal`), senão no **último ponto** (~t+20s).
- **CLV** aqui é **somente pre‑match** (closing pré‑jogo). Para **Lay**, usamos a convenção unificada `clv_conv = -(entry - closing)/closing`, logo **Lay “bom” tende a CLV_CONV > 0**.
- **ROI** é calculado no **ponto de entrada da estratégia** (se houver placar). Para Lay, ROI é **por liability**.
- **IC90** é bootstrap por jogo (cluster em `match_id`). Para critério “p30” usamos o quantil bootstrap **p30** do estimador.

| Side | Pre/In | Reversal | N | Jogos | CLV t0 (mean; IC90) | ROI (mean; IC90) | ROI p30 | Ativa? (critério) |
|---|---|---|---:|---:|---:|---:|---:|---|
| Back | Pre | Any | 0 | 0 | — — | — — | — | não (ROI>0 (NS) AND CLV>0) |
| Back | In | Any | 0 | 0 | — | — — | — | não (ROI>0) |
| Lay | Pre | Yes | 0 | 0 | — — | — — | — | não (ROI>0 (NS) AND CLV>0) |
| Lay | Pre | No | 0 | 0 | — — | — — | — | não (ROI>0 (NS) AND CLV>0) |
| Lay | In | Yes | 0 | 0 | — | — — | — | não (ROI>0) |
| Lay | In | No | 0 | 0 | — | — — | — | não (ROI>0) |

Notas:
- Se você quiser operar **cada combinação como uma estratégia separada**, o sizing (Kelly/caps) deve ser estimado e governado separadamente por estratégia.
- Esta tabela é **in-sample** na janela (`--lookback-days`). A seção OOS (walk-forward) pode ser habilitada com `--walkforward`.

### 9) Combinações de valor (regime × linha AH × lag)
Tabela rankeada por volume (N). Métrica: diff_pct (OK, betslip confiável).

### 9.1 Back combos (diff_pct >= corte)
| Regime | Linha | Lag | N | Jogos | Mean diff % | IC90 cluster |
|---|---|---|---:|---:|---:|---|

### 9.2 Lay combos (diff_pct <= corte) + risco
| Regime | Linha | Lag | N | Jogos | Mean diff % | IC90 cluster | Liability p95 |
|---|---|---|---:|---:|---:|---|---:|

---
### 9.3 Stake sizing — teoria mínima + calibração empírica
Objetivo: explicar por que **ROI por aposta** pode divergir de **ROI ponderado por stake/liability**, e propor uma política de staking que seja (i) coerente com edge/CLV e (ii) controlada por risco (p99/ES).

**Teoria (resumo prático)**

- **Flat stake**: cada aposta pesa igual. Boa baseline para checar se o sizing atual está piorando resultado.
- **Proporcional ao limite**: útil operacionalmente (capacidade), mas **não é** sizing por edge.
- **Kelly fracionado**: sizing por edge. Para Back, \(f^* \propto \frac{EV}{odds-1}\). Para Lay, o sizing natural é por **liability**.
- **Governança de risco**: impor **cap por aposta** (ex.: 1–2% da banca) e olhar p95/p99/ES95 de exposição.

**Como o Kelly está sendo calculado aqui (detalhado, com premissas)**

Como ainda não temos um modelo explícito de probabilidade \(p\) por aposta, usamos um proxy padrão: **o closing pré‑jogo como melhor estimativa de preço justo**. A partir disso inferimos \(p\) e aplicamos Kelly como aproximação.

Premissas e entradas:

- **Entrada (Back)**: `entry_odd = bs_odd` (odd do betslip no momento de execução).
- **Entrada (Lay)**: `entry_lay_odd = hypothesis_details.lay.odd` (fallback: `bs_odd`).
- **Preço justo (pre‑match)**: `closing_odd` (closing line). Inferimos \(p \approx 1/closing\_odd\).
- **Aplicabilidade**: para `is_live=True` (in‑match), **não usamos** `closing_odd` como benchmark de CLV/Kelly.

Fórmulas (Back):

- Odds decimais \(O\); retorno líquido \(b = O-1\).
- \(p \approx 1/closing\_odd\).
- Valor esperado por unidade de stake: \(EV = O\cdot p - 1\).
- Kelly cheio (fração de banca em **stake**): \(f^* = \frac{EV}{b} = \frac{O\cdot p - 1}{O-1}\).
- No relatório: \(f = \max(0,f^*)\cdot \text{frac}\) com `frac` em {0.10, 0.25, 0.50, 1.00}.

Fórmulas (Lay):

- Para Lay, o “capital em risco” natural é a **liability** \(L\) (perda máxima), não o stake.
- Usamos \(p \approx 1/closing\_odd\) e \(o = entry\_lay\_odd\).
- Kelly em termos de **liability** (proxy): \(f^*_{liab} = 1 - p\cdot o\).
- No relatório: \(f_{liab} = \max(0,f^*_{liab})\cdot \text{frac}\).
- Conversão para stake (apenas para turnover): \(stake = L/(o-1)\).

Derivação rápida (por que \(f^*_{liab}=1-p\cdot o\)):

- Defina \(W\) como banca e escolha alocar \(L=f\cdot W\) como **liability**.
- Se o evento acontece (prob. \(p\)), você perde \(L\): \(W' = W-L = W(1-f)\).
- Se o evento não acontece (prob. \(1-p\)), você ganha o **stake** do Lay, que é \(S=L/(o-1)\): \(W' = W+S = W\left(1+\frac{f}{o-1}\right)\).
- Kelly maximiza \(p\log(1-f) + (1-p)\log\left(1+\frac{f}{o-1}\right)\). Derivando e igualando a zero, obtém-se \(f^* = 1 - p\cdot o\).

Parâmetros de escala (proxy de banca) e caps:

- Por padrão: `back_bank_ref = p99(stake)` e `lay_bank_ref = p99(liability)` observados no sizing **PROXY** da janela.
- Opcional: com `--kelly-bankroll`, usamos `bank_ref = bankroll` para simular capacidade com banca explícita.
- `stake_back = min(f * back_bank_ref, cap_back, cap_evento_limit)`.
- `liab_lay = min(f_liab * lay_bank_ref, cap_lay, cap_evento_limit)`.
- Caps atuais (guardrail): `cap_back = 2.0% * ref`, `cap_lay = 1.0% * ref`. Cap por evento: `max_stake = 100% * limit`.
- **Implicação importante**: se o cap estiver frequentemente ativo, aumentar `frac` (ex.: >0,25×Kelly) **não aumenta** tamanho real — a curva satura.

Limitações: comissão/vigorish não modelados; correlação entre apostas ignorada; closing como preço justo é aproximação; e o `bank_ref` é uma escala interna (proxy) baseada em limits observados.

**Diagnóstico: exposição vs performance (correlação de Pearson; indicativo, não causal)**

- **Back (stake)**: corr(exposição, ROI)=—; corr(exposição, CLV)=— (onde CLV existe).
- **Lay (liability)**: corr(exposição, ROI)=—; corr(exposição, CLV)=— (onde CLV existe).

**Backtest de sizing (apenas eventos com placar; valores em unidade monetária *proxy*)**

| Lado | Scheme | N | Turnover (janela) | Lucro (janela) | ROI/turnover | p99 exposição | ES95 exposição | DD 30d (média) | DD 30d (p95) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Back | FLAT | 0 | 0.00 | 0.00 | —% | — | — | — | — |
| Lay | FLAT | 0 | 0.00 | 0.00 | —% | — | — | — | — |
| Back | PROXY | 0 | 0.00 | 0.00 | —% | — | — | — | — |
| Lay | PROXY | 0 | 0.00 | 0.00 | —% | — | — | — | — |
| Back | KELLY_0.10 | 0 | 0.00 | 0.00 | —% | — | — | — | — |
| Lay | KELLY_0.10 | 0 | 0.00 | 0.00 | —% | — | — | — | — |
| Back | KELLY_0.25 | 0 | 0.00 | 0.00 | —% | — | — | — | — |
| Lay | KELLY_0.25 | 0 | 0.00 | 0.00 | —% | — | — | — | — |
| Back | KELLY_0.50 | 0 | 0.00 | 0.00 | —% | — | — | — | — |
| Lay | KELLY_0.50 | 0 | 0.00 | 0.00 | —% | — | — | — | — |
| Back | KELLY_1.00 | 0 | 0.00 | 0.00 | —% | — | — | — | — |
| Lay | KELLY_1.00 | 0 | 0.00 | 0.00 | —% | — | — | — | — |

**Backtest de sizing por banca (10k/50k/100k/500k; foco em FLAT/PROXY/KELLY)**

**Banca (ref) = 10.000**

| Lado | Scheme | N | Turnover (janela) | Lucro (janela) | ROI/turnover | p99 exposição | ES95 exposição | DD 30d (média) | DD 30d (p95) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Back | FLAT | 0 | 0.00 | 0.00 | —% | — | — | — | — |
| Lay | FLAT | 0 | 0.00 | 0.00 | —% | — | — | — | — |
| Back | PROXY | 0 | 0.00 | 0.00 | —% | — | — | — | — |
| Lay | PROXY | 0 | 0.00 | 0.00 | —% | — | — | — | — |
| Back | KELLY_0.25 | 0 | 0.00 | 0.00 | —% | — | — | — | — |
| Lay | KELLY_0.25 | 0 | 0.00 | 0.00 | —% | — | — | — | — |

**Banca (ref) = 50.000**

| Lado | Scheme | N | Turnover (janela) | Lucro (janela) | ROI/turnover | p99 exposição | ES95 exposição | DD 30d (média) | DD 30d (p95) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Back | FLAT | 0 | 0.00 | 0.00 | —% | — | — | — | — |
| Lay | FLAT | 0 | 0.00 | 0.00 | —% | — | — | — | — |
| Back | PROXY | 0 | 0.00 | 0.00 | —% | — | — | — | — |
| Lay | PROXY | 0 | 0.00 | 0.00 | —% | — | — | — | — |
| Back | KELLY_0.25 | 0 | 0.00 | 0.00 | —% | — | — | — | — |
| Lay | KELLY_0.25 | 0 | 0.00 | 0.00 | —% | — | — | — | — |

**Banca (ref) = 100.000**

| Lado | Scheme | N | Turnover (janela) | Lucro (janela) | ROI/turnover | p99 exposição | ES95 exposição | DD 30d (média) | DD 30d (p95) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Back | FLAT | 0 | 0.00 | 0.00 | —% | — | — | — | — |
| Lay | FLAT | 0 | 0.00 | 0.00 | —% | — | — | — | — |
| Back | PROXY | 0 | 0.00 | 0.00 | —% | — | — | — | — |
| Lay | PROXY | 0 | 0.00 | 0.00 | —% | — | — | — | — |
| Back | KELLY_0.25 | 0 | 0.00 | 0.00 | —% | — | — | — | — |
| Lay | KELLY_0.25 | 0 | 0.00 | 0.00 | —% | — | — | — | — |

**Banca (ref) = 500.000**

| Lado | Scheme | N | Turnover (janela) | Lucro (janela) | ROI/turnover | p99 exposição | ES95 exposição | DD 30d (média) | DD 30d (p95) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Back | FLAT | 0 | 0.00 | 0.00 | —% | — | — | — | — |
| Lay | FLAT | 0 | 0.00 | 0.00 | —% | — | — | — | — |
| Back | PROXY | 0 | 0.00 | 0.00 | —% | — | — | — | — |
| Lay | PROXY | 0 | 0.00 | 0.00 | —% | — | — | — | — |
| Back | KELLY_0.25 | 0 | 0.00 | 0.00 | —% | — | — | — | — |
| Lay | KELLY_0.25 | 0 | 0.00 | 0.00 | —% | — | — | — | — |


Leitura:
- Se `PROXY` piora ROI/turnover vs `FLAT`, isso indica que a política de stake atual está concentrando exposição em pontos com pior performance.
- `KELLY_0.25` tende a ser um bom compromisso quando o edge é estimado por CLV, mas requer **caps** e só é aplicável quando há `closing_odd` (pre‑match).
- Em Lay, é comum observar ROI alto por **liability**, mas sizing menor em **stake**: isso é uma decisão deliberada de governança de risco (liability tem cauda pior).
- DD é estimado por bootstrap i.i.d de dias (aproximação). Para uma curva mais fiel, use bootstrap por dia com blocos maiores.

### 9.3b Stake sizing por estratégia (8 combinações)
Abaixo repetimos o backtest de sizing **separado** por cada combinação `Side × Pre/In × Reversal`. Isso responde diretamente sua necessidade: **se várias combinações tiverem valor, o Kelly/caps deve ser calibrado por estratégia**.

Observações:
- Kelly é calculado **somente pre-match** (depende de `closing_odd`). Em combinações `In`, reportamos apenas `FLAT` e `PROXY`.
- ROI do Lay é por **liability**; turnover é mostrado em stake equivalente.

| Side | Pre/In | Reversal | Scheme | N (placar) | Turnover | Lucro | ROI/turnover | p99 exp | DD30 p95 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Back | Pre | Yes | FLAT | 0 | 0.00 | 0.00 | —% | — | — |
| Back | Pre | Yes | PROXY | 0 | 0.00 | 0.00 | —% | — | — |
| Back | Pre | Yes | KELLY_0.10 | 0 | 0.00 | 0.00 | —% | — | — |
| Back | Pre | Yes | KELLY_0.25 | 0 | 0.00 | 0.00 | —% | — | — |
| Back | Pre | Yes | KELLY_0.50 | 0 | 0.00 | 0.00 | —% | — | — |
| Back | Pre | Yes | KELLY_1.00 | 0 | 0.00 | 0.00 | —% | — | — |
| Back | Pre | No | FLAT | 0 | 0.00 | 0.00 | —% | — | — |
| Back | Pre | No | PROXY | 0 | 0.00 | 0.00 | —% | — | — |
| Back | Pre | No | KELLY_0.10 | 0 | 0.00 | 0.00 | —% | — | — |
| Back | Pre | No | KELLY_0.25 | 0 | 0.00 | 0.00 | —% | — | — |
| Back | Pre | No | KELLY_0.50 | 0 | 0.00 | 0.00 | —% | — | — |
| Back | Pre | No | KELLY_1.00 | 0 | 0.00 | 0.00 | —% | — | — |
| Back | In | Yes | FLAT | 0 | 0.00 | 0.00 | —% | — | — |
| Back | In | Yes | PROXY | 0 | 0.00 | 0.00 | —% | — | — |
| Back | In | No | FLAT | 0 | 0.00 | 0.00 | —% | — | — |
| Back | In | No | PROXY | 0 | 0.00 | 0.00 | —% | — | — |
| Lay | Pre | Yes | FLAT | 0 | 0.00 | 0.00 | —% | — | — |
| Lay | Pre | Yes | PROXY | 0 | 0.00 | 0.00 | —% | — | — |
| Lay | Pre | Yes | KELLY_0.10 | 0 | 0.00 | 0.00 | —% | — | — |
| Lay | Pre | Yes | KELLY_0.25 | 0 | 0.00 | 0.00 | —% | — | — |
| Lay | Pre | Yes | KELLY_0.50 | 0 | 0.00 | 0.00 | —% | — | — |
| Lay | Pre | Yes | KELLY_1.00 | 0 | 0.00 | 0.00 | —% | — | — |
| Lay | Pre | No | FLAT | 0 | 0.00 | 0.00 | —% | — | — |
| Lay | Pre | No | PROXY | 0 | 0.00 | 0.00 | —% | — | — |
| Lay | Pre | No | KELLY_0.10 | 0 | 0.00 | 0.00 | —% | — | — |
| Lay | Pre | No | KELLY_0.25 | 0 | 0.00 | 0.00 | —% | — | — |
| Lay | Pre | No | KELLY_0.50 | 0 | 0.00 | 0.00 | —% | — | — |
| Lay | Pre | No | KELLY_1.00 | 0 | 0.00 | 0.00 | —% | — | — |
| Lay | In | Yes | FLAT | 0 | 0.00 | 0.00 | —% | — | — |
| Lay | In | Yes | PROXY | 0 | 0.00 | 0.00 | —% | — | — |
| Lay | In | No | FLAT | 0 | 0.00 | 0.00 | —% | — | — |
| Lay | In | No | PROXY | 0 | 0.00 | 0.00 | —% | — | — |
### 9.4 Estratégias candidatas (combinações 8.3 + sizing recomendado)
Esta seção foi atualizada para refletir as **combinações** que você está analisando (Back/Lay × Pre/In × Reversal). Ela não assume mais apenas `BackFast` e `LayReversal`.

**Política de entrada**:
- Back: `t0`.
- Lay: **após reversão** quando existir; senão no **último ponto** (~t+20s).

**Política de sizing sugerida** (padrão):
- Pre‑match: `KELLY_0.25` (com caps e cap por evento).
- In‑match: `FLAT` ou `PROXY` capado, até existir um benchmark live (Kelly live não é confiável sem referência).

| Side | Pre/In | Reversal | N (janela) | Jogos | CLV (entry; IC90) | ROI (entry; IC90) | ROI p30 | Observação |
|---|---|---|---:|---:|---:|---:|---:|---|
| Back | Pre | Yes | 0 | 0 | — — | — — | — | pre: Kelly OK |
| Back | Pre | No | 0 | 0 | — — | — — | — | pre: Kelly OK |
| Back | In | Yes | 0 | 0 | — — | — — | — | in: use FLAT/PROXY |
| Back | In | No | 0 | 0 | — — | — — | — | in: use FLAT/PROXY |
| Lay | Pre | Yes | 0 | 0 | — — | — — | — | pre: Kelly OK |
| Lay | Pre | No | 0 | 0 | — — | — — | — | pre: Kelly OK |
| Lay | In | Yes | 0 | 0 | — — | — — | — | in: use FLAT/PROXY |
| Lay | In | No | 0 | 0 | — — | — — | — | in: use FLAT/PROXY |
**Tabela política de sizing sugerida — resumo executivo (30d)**

| Estratégia | Scheme | Turnover 30d | Lucro 30d | Banca rec. (max) | ROI/banca 30d | DD 30d p95 |
|---|---|---:|---:|---:|---:|---:|
| Ativas (PRE, critérios 8.3) | FLAT | 0.00 | 0.00 | — | —% | — |
| Ativas (PRE, critérios 8.3) | KELLY_0.25 | 0.00 | 0.00 | — | —% | — |

**Tabela política de sizing sugerida — detalhe (volume/sizing/risco)**

| Estratégia | Scheme | N Back | N Lay | N Back 30d | N Lay 30d | Stake méd Back | Stake méd Lay | Liab méd Lay |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Ativas (PRE, critérios 8.3) | FLAT | 0 | 0 | 0 | 0 | — | — | — |
| Ativas (PRE, critérios 8.3) | KELLY_0.25 | 0 | 0 | 0 | 0 | — | — | — |

| Estratégia | Scheme | ROI/turnover (janela) | ROI Lay/liab (janela) | Banca risco p99 | Banca liq p99 | Banca rec. (max) |
|---|---|---:|---:|---:|---:|---:|
| Ativas (PRE, critérios 8.3) | FLAT | —% | —% | — | — | — |
| Ativas (PRE, critérios 8.3) | KELLY_0.25 | —% | —% | — | — | — |


Notas:
- **N Back/Lay** na tabela é **na janela observada**. As colunas `N 30d (proj.)` são uma **escala linear** por dias observados.
- Esta linha usa a união das **combinações pre‑match ativas** sob os critérios da 8.3 (é um resumo, não substitui a tabela 8.3).
- `Stake médio` e `Liability média` são **proxies** na unidade monetária do seu limit/finance; valores em **R$** usam fx `--fx-usdbrl`.
- `Banca recomendada` = max( banca por risco p99 (unitária, soma) ; banca por liquidez p99 (+buffer) ).
- **Por que Lay pode ter stake médio menor mesmo com ROI maior**: (i) Lay é governado por **liability** (risco) e usamos cap mais conservador; (ii) stake em Lay é derivado de `liability/(odd-1)` e depende do nível médio de odds; (iii) Kelly depende do **edge vs closing** (gap entry→closing), não do ROI observado isoladamente.
- **Por que não incluímos Back in‑match aqui**: Kelly/CLV usam `closing_odd` pré‑jogo; in‑match requer benchmark diferente (ex.: referência por minuto/VWAP) e governança operacional específica. A seção 2.2 já compara PRE_MATCH vs IN_MATCH.

### 9.4b Curva de capacidade — frações de Kelly e tamanho potencial
Esta tabela é um exercício para estimar **capacidade** (turnover/lucro/risco) ao variar a fração de Kelly. Ela deixa explícito quando o sizing satura por **cap por aposta**.

**Escala Kelly usada nesta curva**: BANKROLL | ref_back=10,000.00 ref_lay=10,000.00 | cap_back=2.0% cap_lay=1.0% | max_stake_event=100%*limit

| Strategy | Scheme | cap_hit Back (%) | limit_hit Back (%) | cap_hit Lay (%) | limit_hit Lay (%) | Turnover 30d (proj.) | Lucro 30d (proj.) | ROI/banca 30d | DD 30d p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ativas (PRE, critérios 8.3) | KELLY_0.10 | —% | —% | —% | —% | 0.00 | 0.00 | —% | — |
| Ativas (PRE, critérios 8.3) | KELLY_0.25 | —% | —% | —% | —% | 0.00 | 0.00 | —% | — |
| Ativas (PRE, critérios 8.3) | KELLY_0.50 | —% | —% | —% | —% | 0.00 | 0.00 | —% | — |
| Ativas (PRE, critérios 8.3) | KELLY_1.00 | —% | —% | —% | —% | 0.00 | 0.00 | —% | — |

Leitura rápida:
- Se `cap_hit` estiver alto, a alavancagem adicional (ex.: 0,50× ou 1,00×) não vira turnover — o cap está “travando” o tamanho.
- Se `limit_hit` estiver alto, você está batendo no **limit operacional** (capacidade da conta/mercado) — aumentar banca ou frac não aumenta turnover.
- Se `cap_hit` estiver baixo, a curva deve escalar quase linearmente com `frac` (até bater em limites reais/operacionais).
- Lay normalmente satura antes por ter cap mais conservador (liability) e por ter risco de cauda mais assimétrico.

### 9.4c Diagnóstico: Back in‑match (por que não está na estratégia candidata)
Aqui reportamos Back in‑match **apenas como diagnóstico**. Não incluímos in‑match na estratégia candidata porque (i) `closing_odd` pré‑jogo não é benchmark in‑match e (ii) o sizing Kelly acima depende desse benchmark.

| Regime | Scheme | N | Turnover (janela) | Lucro (janela) | ROI/turnover |
|---|---|---:|---:|---:|---:|
| IN_MATCH BackFast (<5s) | FLAT | 0 | 0.00 | 0.00 | —% |
| IN_MATCH BackFast (<5s) | PROXY | 0 | 0.00 | 0.00 | —% |

Próximo passo (se quiser operar in‑match): definir um benchmark de preço justo in‑match (ex.: referência por minuto, VWAP, ou odds externas) e calibrar sizing/risco específico do live.

### 10) Diagnóstico: por que o ROI pode estar zerado
ROI aqui é calculado por placar do jogo (`matches.home_score/away_score`). Se os placares não estiverem preenchidos no banco, a cobertura de ROI será 0.

| Indicador | Valor |
|---|---:|
| Jogos únicos no recorte | 0 |
| Jogos com placar disponível (home_score/away_score não nulos) | 0 |
| Jogos com status='finished' no banco | 0 |

### 10.1 Distribuição por data de kickoff (explica janela da API)
- Kickoff (UTC) no recorte: **—** até **—**.

| Kickoff date (UTC) | Jogos | Com placar | Cobertura |
|---|---:|---:|---:|

**Leitura**: se seu recorte inclui muitos jogos com kickoff antigo, a API-Football **free** pode não retornar fixtures dessa data (limitação por janela recente). Nesse cenário, mesmo com o job rodando, `placar disponível` ficará baixo para jogos fora da janela.

Se `placar disponível` estiver 0 (mesmo para datas recentes), isso geralmente indica que o job de resultados não rodou ou está sem chave válida.  
Sugestão (rodar no servidor): `cd betinasia_bot && python3 -m results.auto_update_results --once` (ou configure o serviço para rodar em loop).

---
### 11) Conclusões (visão de investidor), riscos e próximos passos
Esta seção é escrita como se um investidor externo estivesse avaliando a tese: **há edge replicável? o sistema executa? o risco é governável? a mensuração é confiável?**

### 11.1 O que já está forte (e por quê)
- **Evidência de execução (CLV pre‑match)**: CLV robusto por jogo positivo é um dos melhores sinais de edge/execução em janela curta. Diferente de ROI, CLV não depende de amostra grande de jogos liquidados; ele mede **qualidade de entrada**.
- **Controle de latência por regime**: o relatório já separa regimes de execução por tempo total (2.3/2.3b). Isso permite uma regra objetiva de operação (ex.: só operar `exec_bucket < 5s`).
- **Separação Back vs Lay**: Back e Lay têm perfis de risco diferentes. Lay deve ser governado por **liability** (p95/p99/ES), e isso já aparece como métrica de banca e risco.

### 11.2 O que ainda está frágil (e impede captação hoje)
- **ROI ainda não é prova**: mesmo quando ROI aparece, a incerteza por jogo pode ser grande e a cobertura de placar pode ser incompleta. Para captação, um investidor vai pedir **histórico maior**, **pipeline de resultados estável** e **métrica de drawdown** bem definida.
- **Risco de viés por falhas de coleta**: quando o collector fica “active” mas não coleta odds, você perde janelas do mercado de forma não aleatória. Isso impacta a extrapolação para execução.
- **Stake sizing ainda é proxy**: parte do sizing usa limit/finance como aproximação. Para captação, é necessário um sizing governado por risco e consistente com edge (ex.: Kelly fracionado + caps), com auditoria clara.

### 11.3 Avaliação das 2 estratégias candidatas (como um investidor leria)
Você propôs duas teses operacionais coerentes com o mecanismo observado:
1) **BackFast**: operar Back edge apenas quando a execução foi rápida (`< 5s`) e pre‑match.
2) **LayReversal**: operar Lay edge apenas quando há reversão e entrar próximo do vale (t_ext curto).

O relatório quantifica isso na **Seção 9.4** com (i) N na janela, (ii) projeção 30d, (iii) stake/liability médio, (iv) banca p99 e ROI/banca mensal, e (v) drawdown p95.

**Como um investidor decide**: ele vai priorizar uma estratégia com
- sinal de edge (CLV) consistente,
- execução estável (latência controlada),
- sizing governado por risco (caps + banca p99/ES),
- e um perfil de drawdown aceitável no horizonte de caixa.

### 11.4 Stake sizing: recomendação inicial para produção (sem overfitting)
- Use **baseline FLAT** como controle (para detectar se o sizing está degradando performance).
- Para Back, use **Kelly fracionado** (ex.: `KELLY_0.25`) apenas quando houver `closing_odd` (pre‑match), com **cap** por aposta (ex.: 2% da banca p99).
- Para Lay, faça sizing por **liability**, com cap mais conservador (ex.: 1% da banca p99) e monitoramento de cauda (p95/p99/ES95).

A Seção 9.3 compara `FLAT` vs `PROXY` vs `KELLY` (fracionado) no subconjunto com placar, e reporta risco (p99/ES) e drawdown 30d via bootstrap.

### 11.5 Status para captação (checkpoint objetivo)
Se você estivesse captando hoje, um investidor institucional provavelmente pediria:
- **(A)** 30–90 dias de execução estável com SLO de coleta (collector), auditoria e resultados.
- **(B)** KPIs: CLV pre‑match por jogo estável; latência por bucket; taxa de falhas; cobertura de placar.
- **(C)** Política de risco: banca por p99/ES, caps por aposta, limites por janela e mecanismos de stop.
- **(D)** Demonstração de P&L com sizing definido (não só proxy) e drawdown observado/estimado.

Minha leitura: **a tese de edge/execução parece promissora pelo CLV**, mas o projeto ainda está em fase de **consolidação operacional/medição** para uma captação “grande”. Um caminho pragmático é:
- validar BackFast com sizing conservador e risco baixo,
- validar LayReversal com governança de liability,
- e só então ampliar banca.

---
### 12) Como reproduzir
1. Configure `betinasia_bot/.env` com `DATABASE_URL`.  
2. (Opcional) Atualize resultados para ter ROI: `cd betinasia_bot && python3 -m results.auto_update_results --once`.  
3. Execute:

```bash
python3 betinasia_bot/analyze_contexto_operacao_b808_robust_report.py \
  --direction up \
  --versions v4.0-api,v1.0,v1.0-recovered \
  --lookback-days 14 \
  --out betinasia_bot/docs/analise_contexto_operacao_b808_robusta.md \
  --pdf betinasia_bot/docs/analise_contexto_operacao_b808_robusta.pdf
```

### Ajuste operacional: Sensibilidade por banca com gate de slippage (contrafactual)

_Leitura: aplica a regra `Back: pula slippage_raw_pct<=-2%` e `Lay: pula slippage_raw_pct>2%` como um ajuste de capacidade, usando a evidência contrafactual nas execuções cobertas por placar. O ajuste é um **proxy**: usa exposição observada (Back=stake, Lay=liability) para estimar redução de N/turnover e mudança de ROI._

- Fonte OOS (curvas por banca): `logs/daily_reports/20260729/wf_bank_sensitivity.json` (existe=sim; sens_ok=não).

- Aviso do export: `NO_SCENARIOS_EXPORTED`.

_Aviso: não foi possível aplicar o ajuste na sensibilidade por banca porque o export `wf_bank_sensitivity.json` está ausente/vazio/ilegível. Isso não afeta o OOS em si; apenas impede esta tabela ajustada. Se persistir, verifique se o daily está rodando a versão mais recente do `analyze_contexto_operacao_b808_robust_report.py` com `--wf-export-bank-sensitivity-json` habilitado._




> **APÊNDICE DE PESQUISA — NÃO OPERACIONAL** — recomendações de risco/sizing não são operacionais neste Daily.
