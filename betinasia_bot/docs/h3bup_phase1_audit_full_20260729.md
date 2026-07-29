# H3BUP Phase 1 — Auditoria completa READ-ONLY (20260729)

Gerado: 2026-07-29T12:35:23Z  
Corte operacional desta fase: **2026-07-29T12:31:52Z**  
Janela pós-fix capacity: **≥ 2026-07-28T13:19:39Z**

Impacto: **zero alterações operacionais** (checksums/services timestamps verificados antes/depois).

---

## PARTE 1 — Audit H3B

### Diagrama
```text
WS odds message (Playwright page.on websocket)
  → parse markets AH/OU; update _ws_odds_state
  → HypothesisDetector.process_market_update
  → H3bTemporalReversalDetector.update_odd
  → filter direction_after == "up"
  → audit dedup key TTL (AUDIT_DEDUP_TTL_SEC=300)
  → optional prefilter (AUDIT_PREFILTER_*)
  → queue executor workers
  → _execute_ws_gate_back
       (enforce_rise=0: OK @ T+0 if ws key/side/ws0)
       (measure offsets 0,5,10 deferred async)
  → status/reason + is_valid_opportunity
  → _save_result → betslip_audit_results
  → bridge _fetch_candidates (lookback 120s, poll 0.3s)
  → seen-key reserve (TTL 600 / hard 86400)
  → _h3bup_vnext_eval
  → ExecutionRequest / executor gates
```

### Respostas 1–65 (síntese evidenciada)

| # | Resposta |
|---|---|
| 1 | Mensagem WebSocket de odds processada em handler `page.on('websocket')` / parse de mercados |
| 2 | Universo: eventos com odds WS; filtra jogos com kickoff há >9000s; `|AH line|<=10` |
| 3 | Detector processa AH (e OU no detector geral); **bridge consome só `market_type=AH`** |
| 4 | Sides home/away (AH); audit service `AUDIT_API_SIDES=back` |
| 5–6 | `direction=up`: após uma direção prévia, o movimento atual da odd é para cima (`odd > last_odd`) e isso constitui reversão para up |
| 7 | Preço base do gate: `websocket_odd` = `odd_at_reversal` no momento da detecção |
| 8 | Momento zero: `detected_at = time.time()` no enqueue; offsets relativos a `detected_at` |
| 9–12 | Offset 0 = ws0 na detecção; 5 e 10 medidos async (`GATE_BACK_MEASURE_OFFSETS_SEC=0,5,10` + rise offset). Outros offsets WS sample env `0..30` existem para outros modos |
| 13 | Relativos a `detected_at` (abs ts = detected_at+offset), não wall-clock arbitrário |
| 14 | Sim: queue workers + temporal workers async |
| 15–16 | Possível reordenação/atraso de workers; dedup por audit_key reduz flood; multi-worker na mesma key mitigado por TTL map |
| 17 | Dedup `_audited_ts[audit_key]` TTL **300s** (env efetivo) |
| 18–20 | Gaps: estado WS pode ficar stale; há reload em outros modos (`api_ws_stale`); reconnect WS via Playwright — detalhes finos NÃO CONFIRMADO além de hooks existentes |
| 21–22 | `GATE_STALE` só no caminho enforce=1; freshness bridge = lookback 120s |
| 23–24 | Debounce/cooldown audit = TTL dedup 300s; bridge seen TTL 600s |
| 25–28 | Dedup no audit (audit_key) **e** no bridge (seen keys). Bridge src_key/seen; TTL 600 / hard 86400 |
| 29 | Sim, após expirar TTL dedup |
| 30 | Evitado por `executor_bridge_seen` / seen_keys se finalizado; reservas órfãs GC |
| 31–34 | Ver CSV payload dictionary; tabela `betslip_audit_results`; PK `id` = audit_id |
| 35–38 | Status observados v5.3 desde restart: praticamente só `OK`. Código também define GATE_WS_MISSING, GATE_WS_POINT_MISSING, GATE_NOT_ELIGIBLE, GATE_STALE. **OK exacto (enforce=0):** ws_state_key & side & ws0>0 → status OK + is_valid_opportunity True |
| 39–41 | Com enforce=0, medição continua (deferred); filtro 1.02 **não bloqueia**. Se activate=1, exige `ws5 >= 1.02 * ws0` |
| 42 | Sim: mais amplo que “rise 2%”; aprova no T+0 |
| 43–44 | Prematch: bridge `only_prematch`; audit marca is_live via kickoff |
| 45–46 | Sem filtro placar/minuto no gate back |
| 47–49 | Sem filtro liquidez/book/liga no audit OK path; H3BUP bypassa ligas depois |
| 50–52 | Line string do detector; MAX_AH_LINE=10; line movement é a própria reversão |
| 53 | Mercados sem odds válidas skip; suspenso explícito NÃO CONFIRMADO |
| 54–55 | Exige odd>0; sem band 1.85–2.15 no audit (isso é H3BUP) |
| 56–58 | Payload = row DB; bridge usa id, odds, limit, is_live, details.exec_side_hint, league, market/side/line |
| 59–60 | Consome `is_valid_opportunity=TRUE` (não status). SQL em `_fetch_candidates` |
| 61 | `ORDER BY audited_at ASC` (env newest_first **não aplicado**) |
| 62–63 | lookback **120s**; poll **0.3s** |
| 64–65 | Sim: elegível pode sair do lookback; race persistência→fetch mitigada por audited_at e retry |

### Etapas

| Etapa | Arquivo | Função | Input | Output | Timestamp | Status | Falha |
|---|---|---|---|---|---|---|---|
| WS parse | audit_h3b_api.py | WS handler | ws msg | odds state | time.time | — | parse skip |
| Detect | detectors.py | update_odd | OddSnapshot | H3b event | snapshot.ts | — | no reversal |
| Enqueue | audit_h3b_api.py | queue.put | h3b dict | queued | detected_at | — | full queue |
| Gate back | audit_h3b_api.py | _execute_ws_gate_back | h3b | result dict | detected_at | OK/... | missing ws |
| Save | audit_h3b_api.py | _save_result | result | row id | audited_at | persisted | DB error |
| Fetch | executor_bridge_audit.py | _fetch_candidates | since | rows | audited_at | candidates | DB disconnect |
| Seen | executor_bridge_audit.py | _reserve_seen_key | src_key | reserved | now | — | conflict |
| Policy | executor_bridge_audit.py | _h3bup_vnext_eval | row | eval | decision_at | pass/reject | odd/backpre |

---

## PARTE 2 — Timestamps e latência

Ver `logs/h3bup_existing_timestamps_inventory_20260729.csv` e `logs/h3bup_existing_latency_analysis_20260729.csv`.

### Calculável (LIVE_OK n=12)
{
  "audit_to_request_ms": {
    "n": 12,
    "mean": 2758.66,
    "median": 3107.296,
    "p75": 3190.67,
    "p90": 3319.609,
    "p95": 3319.609,
    "p99": 4911.996,
    "max": 4911.996,
    "min": 287.052,
    "neg": 0
  },
  "audit_to_finished_ms": {
    "n": 12,
    "mean": 8502.471,
    "median": 8512.942,
    "p75": 9672.367,
    "p90": 10177.286,
    "p95": 10177.286,
    "p99": 11286.516,
    "max": 11286.516,
    "min": 6298.238,
    "neg": 0
  },
  "detected_to_audited_ms": {
    "n": 12,
    "mean": 4.56,
    "median": 3.965,
    "p75": 4.412,
    "p90": 6.93,
    "p95": 6.93,
    "p99": 7.708,
    "max": 7.708,
    "min": 2.788,
    "neg": 0
  },
  "shadow_to_request_ms": {
    "n": 12,
    "mean": 2330.621,
    "median": 3025.298,
    "p75": 3032.369,
    "p90": 3032.946,
    "p95": 3032.946,
    "p99": 3642.51,
    "max": 3642.51,
    "min": 8.751,
    "neg": 0
  }
}

Request→finished LIVE_OK: {
  "n": 12,
  "mean": 5743.811,
  "median": 5622.845,
  "p75": 6473.531,
  "p90": 9024.813,
  "p95": 9024.813,
  "p99": 9890.234,
  "max": 9890.234,
  "min": 3433.825,
  "neg": 0
}

### Não calculável com dados atuais
WS receive → detect; bridge fetch wall time; dryrun start; place start dedicados.

### Clock
- PostgreSQL: `Etc/UTC`
- Executor JSONL: UTC Z
- Accounting post date: tz não explícito
- Negativos nas latências LIVE_OK analisadas: **0**

Design futuro: `logs/h3bup_future_e2e_trace_design_20260729.md`

---

## PARTE 3 — Settlement / P&L

Corte: 2026-07-29T12:31:52Z. LIVE_OK pós-fix: **12** (cresceu vs 8 do handoff).

Accounting latest file: `20260728_220133__balance.csv` (mtime 2026-07-28T22:01Z).  
Monitor: active mas `files={balance: None, open_stakes: None}`.

| Reconciliation status | N | Notas |
|---|---:|---|
| SETTLED_ACCOUNTING_OK | 1 | order 1931674091 amount -9.97 |
| VOID_OR_PUSH | 1 | 1932353274 amount 0.00 |
| SETTLED_MISSING_ACCOUNTING | 1 | 1933822208 (kickoff 29/07 00:00; CSV stale) |
| EVENT_NOT_STARTED | 9 | kickoffs futuros |
| OPEN_NOT_SETTLED / JOIN_FAILURE | 0 | open CSV também stale |

Funil:
- LIVE_OK 12
- eventos terminados (heurística kickoff+2.5h < now): ~3
- liquidados no CSV: 2
- P&L reconciliado parcial: **-9.97** sobre stake settled 20 → ROI **-49.85%** (**não** usar como ROI da estratégia)
- Coverage settled adequada? **Não**

Classificação frente: **SETTLEMENT_ACCOUNTING_LAG**

Detalhe order-level: `logs/h3bup_settlement_reconciliation_20260729.csv`

---

## PARTE 4 — CLV infra

Ver `logs/h3bup_clv_infrastructure_inventory_20260729.md` e design `logs/h3bup_future_clv_design_20260729.md`.

Conclusão: **CLV_INFRA_PARTIALLY_REUSABLE** (closing offline/analytics) + componentes novos para obligations/scheduler H3BUP.

---

## PARTE 5 — Impacto zero

| Check | Antes | Depois |
|---|---|---|
| executor ActiveEnter | 2026-07-28 13:19:39Z | igual |
| bridge-back | 2026-07-28 06:45:49Z | igual |
| audit-ws-gate-back | 2026-07-29 10:06:45Z | igual |
| md5 audit_h3b_api.py | 6d163e5e... | igual |
| md5 bridge | 3ccdc438... | igual |
| md5 worker | 298f8073... | igual |
| BRIDGE_MODE/STAKE/LIVE_STAKE | live/10/10 | igual |
| disable_back | false | false |

Perguntas 49–57: **Não** (sem alterações/deploys/restarts/jobs/betslips/ordens/migrations).

---

## Verificação final objetiva

### AUDIT H3B
1. WS odds → detector reversão  
2. odd subiu (reversão para up)  
3. 0,5,10 (measure); sample env mais largo noutros modos  
4. rise: `ws5 >= 1.02 * ws0` (só se enforce)  
5. **Não** enforced  
6. enforce=0: key+side+ws0 → OK+valid  
7. OK (+ GATE_* no código); outras versões LINE_NOT_AVAILABLE/GAME_NOT_FOUND/MAJOR_DIFF  
8. `betslip_audit_results`  
9. row DB + hypothesis_details  
10. `_fetch_candidates` SQL (is_valid_opportunity, H3B, AH, lookback)  
11. audit: `event|market|period|line|side`; bridge: seen keys por src_id/action  
12. audit 300s; bridge 600s / hard 86400  
13. Sim após TTL  
14. lookback 120s  
15. **Sim**, mais amplo que H3BUP e que rise filter

### LATÊNCIA
16–17. Ver inventário CSV  
18. WS→LIVE_OK completo: **não**  
19. N LIVE_OK latência=12  
20. audit→OK median≈8513ms; p95≈10177ms; request→OK median≈5623ms  
21. ws_received, bridge_fetched, dryrun/place starts  
22. sem negativos no subset; PG UTC  
23. h3bup_e2e_trace JSONL-first  

### SETTLEMENT
24. 12  
25. ~3 terminados (heurística)  
26–27. 2 no accounting  
28. 0 after previous snapshot no CSV atual (CSV não avançou)  
29. open CSV stale; 9 not started  
30. 0 join failure óbvio  
31. 1 missing (lag)  
32. falha settlement exchange: NÃO CONFIRMADA  
33. **Sim** — monitor sem arquivos  
34. Handoff 2/8 era stale + muitos não liquidados; agora também monitor quebrado  
35. coverage baixa  
36. ROI total: **não** de forma segura  

### CLV
37. scheduler reutilizável betslip: não; analytics closing: sim  
38. post_5m/15m jobs: não encontrados  
39. same-line parcial via ah_line  
40–41. de-vig/fair edge: NÃO CONFIRMADO  
42. kickoff parcial (matches)  
43–45. N/A scheduler  
46. BestOddsHistory + get_closing_odd + B808 CLV raw  
47. obligations/scheduler/caps/fair edge se desejado  
48. risco betslip alto; analytics baixo  

### IMPACTO
49–57. **Não.**

## Status final
- AUDIT_H3B_CLEAR  
- LATENCY_PARTIALLY_MEASURABLE  
- SETTLEMENT_ACCOUNTING_LAG  
- CLV_INFRA_PARTIALLY_REUSABLE  

**PHASE1_COMPLETE_DATA_GAPS**
