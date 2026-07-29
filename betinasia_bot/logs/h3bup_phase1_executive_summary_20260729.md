# Resumo executivo — H3BUP Fase 1 (20260729)

Corte métricas: **2026-07-29T12:31:52Z** · Gerado: 2026-07-29T12:35:23Z  
Escopo: READ-ONLY. **Nenhuma mudança operacional.**

## Achados confirmados
1. **Audit H3B `ws_gate_back` com `GATE_BACK_ENFORCE_RISE_FILTER=0` aprova no T+0**: status=`OK` e `is_valid_opportunity=True` se existem `ws_state_key`, `side` e `websocket_odd>0`. Offsets 0/5/10 são medidos em paralelo (deferred) para analytics, **não bloqueiam**.
2. **`direction=up`**: reversão temporal para movimento de odd para cima no detector `H3bTemporalReversalDetector` (odd atual > odd anterior após direção previa diferente).
3. **Bridge NÃO filtra por `status=OK`**; filtra `is_valid_opportunity=TRUE`, `hypothesis_type=H3B`, `market_type=AH`, lookback 120s, prematch, exec_side_hint=Back. Env `BRIDGE_SOURCE_STATUSES=OK` está setado mas **não é lido pelo código**.
4. **Latência parcial mensurável** (LIVE_OK n=12): audit→finished mediana ~8.5s; request→finished ~5.6s; detected→audited ~4ms. WS receive dedicado **ausente**.
5. **Settlement**: accounting CSVs parados em 28/07 22:02; monitor ativo mas grava `files=None`. Dos 12 LIVE_OK: 2 no CSV (1 loss -9.97, 1 push/void 0), 9 eventos ainda não iniciados, 1 missing sob lag.
6. **CLV**: há analytics/closing via `best_odds_history`; **não** há scheduler H3BUP post_5m/15m/closing. Fair edge/overround **não confirmados** no path B808 inspecionado.

## Inferências
- INFERÊNCIA: monitor de accounting está com sessão/API degradada (snapshot vazio) apesar de process alive.
- INFERÊNCIA: com enforce rise=0, o audit H3B atual é deliberadamente mais amplo que um gate de “subida 2%”.

## Conflitos
- `BRIDGE_SOURCE_STATUSES` / `BRIDGE_FETCH_NEWEST_FIRST` / `BRIDGE_SOURCE_AUDIT_VERSIONS` no env vs ausência no código do bridge.
- Systemd unit text diz “decide Back via WS(t0,t+5)” mas enforce=0 decide no T+0.
- Handoff 2/8 vs agora 12 LIVE_OK (crescimento natural pós-corte).

## Riscos
- ROI H3BUP não reconciliável enquanto accounting não escrever CSVs.
- Audit amplo (OK sem betslip/diff) aumenta carga no bridge; H3BUP gates finais ainda protegem execução.
- Qualquer CLV via betslip sem caps pode estressar PMM/open betslips.

## Lacunas
- Timestamp WS receive; bridge_fetched; dryrun/place starts.
- Fair edge/de-vig produção.
- Kickoff confidence tree completa.
- Confirmar se 1933822208 liquidou na exchange (sem CSV).

## Acções recomendadas (próximas fases; não fazer agora)
1. Corrigir accounting monitor (root-cause de snapshot None) — ops controlada.
2. Instrumentar `h3bup_e2e_trace` (JSONL primeiro).
3. Desenhar obrigação CLV offline via BestOddsHistory antes de betslip jobs.
4. Limpar/documentar env vars mortas do bridge.

## Status por frente
- AUDIT_H3B: **AUDIT_H3B_CLEAR**
- LATENCY: **LATENCY_PARTIALLY_MEASURABLE** / instrumentação requerida para E2E WS
- SETTLEMENT: **SETTLEMENT_ACCOUNTING_LAG** (falha de produção de snapshot confirmada nos logs)
- CLV: **CLV_INFRA_PARTIALLY_REUSABLE** / new components required para scheduler H3BUP

**Status geral:** `PHASE1_COMPLETE_DATA_GAPS`
