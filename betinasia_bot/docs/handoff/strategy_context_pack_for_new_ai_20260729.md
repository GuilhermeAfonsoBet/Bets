# Context Pack — H3BUP_vNext (para outra IA)
**Data:** 2026-07-29 · **Corte operacional VPS:** 2026-07-29T11:03:41Z  
**Idioma:** português  
**Restrição:** este pacote é somente leitura/contexto. Não altera produção.

Você está recebendo o contexto da estratégia **H3BUP_vNext** do projeto BetinAsia (`github.com/GuilhermeAfonsoBet/Bets`), operando na VPS `178.128.55.30` em `/home/betbot/Bets/betinasia_bot`.  
Este texto é autocontido: não assuma acesso à conversa original, ao repositório completo ou aos logs.

Legenda epistêmica usada abaixo:
- **FATO** — observado em código/VPS/logs
- **NÃO CONFIRMADO** — não encontrado com evidência suficiente
- **INFERÊNCIA — não comprovada diretamente**
- **CONFLITO** — fontes discordam

---

## 1. Descrição da estratégia

H3BUP_vNext é uma **policy operacional de execução** para apostas **Back pré-jogo (Back Pre)** no BetinAsia Exchange. Ela não é um detector de “downward trend”. Ela filtra candidatos já produzidos pelo auditor H3B (`ws_gate_back`) e só permite `place_order` quando odd, capacidade e slippage passam nos limiares, com **stake fixa 10 USD**.

Em linguagem simples: “só apostar Back Pre perto de odd 2.00, com liquidez boa no betslip e preço que não piorou desde a decisão”.

Estágio atual (**FATO**): **live controlado** — executor e bridge-back ativos em LIVE; bridge DT desligado; stake 10; gates ativos.

---

## 2. Identidade e policy exata

### Identidade
| Campo | Valor |
|---|---|
| policy_id | `H3BUP_vNext` |
| policy_version | `H3BUP_vNext_20260629` |
| policy_started_at | `2026-06-29T00:00:00+00:00` |
| aliases | H3BUP; H3b (nome de hipótese nos estudos) |
| strategy_id formal | NÃO CONFIRMADO |
| lock file | `logs/H3BUP_VNEXT_POLICY_LOCK` |
| repo | GuilhermeAfonsoBet/Bets |
| PRs recentes | #27 proxy, #28 post-restore review, #29 capacity fix |

### Policy literal (reproduzível)

Uma execução LIVE Back sob H3BUP exige **todas**:

1. `policy_version` contém `H3BUP_vNext`
2. `exec_side == Back` e mercado **não** ao vivo (Back Pre)
3. `1.85 <= odd_at_decision <= 2.15`
4. `capacity = dry.limit_final` e `capacity > 100`
5. `slippage_pre_pct < 0`, onde  
   `slippage_pre_pct = (odd_pre_submit - odd_at_decision) / odd_at_decision * 100`
6. `stake == 10` (USD)
7. filtro de ligas **desabilitado** (bypass no bridge)

**Não são regras H3BUP:** `cum_drop_pct`, `last_share`, non-Friendly, CLV, fair edge, placar, minuto de jogo.

### Split bridge vs executor (**FATO**)
- **Bridge** (`ops/executor_bridge_audit.py`): pre-exec = Back Pre + odd band; capacity 0/NULL = desconhecida (`capacity_unknown_pre_exec`) e **não** bloqueia pre-exec; capacity conhecida ≤100 rejeita; injeta `H3BUP_vNext_20260629` e stake 10; bloqueia live Back não-H3BUP.
- **Executor** (`executor/worker.py`): gate final odd + `dry.limit_final>100` + slip<0; rejeita Back Pre não-H3BUP; hard-cap stake 10.

### Expressão lógica
```text
final_ok =
  H3BUP_vNext in policy_version
  AND Back Pre
  AND odd in [1.85, 2.15]
  AND dry.limit_final > 100
  AND slippage_pre_pct < 0
  AND stake == 10
```

### O que governa produção
Código patchado na VPS + lock de policy + `bridge_risk_params.json` (`disable_back`) + env de stake.  
`logs/wf_policy_current.json` (lista de ligas) **existe mas é bypassada** para H3BUP.

---

## 3. Diferenças versus estratégia de referência

### Referência pedida (usuário)
```text
DT Back Pre
cum_drop_pct > 3%
non-Friendly
last_share >= 0.35
LIVE_OK
```

### CONFLITO importante
No código/docs DT do repo (`cursor/downward-trend-strategy-1c5f`) e na VPS, DT usa defaults do tipo:
- 3 quedas consecutivas
- step drop ≥ 0.20%
- cum drop ≥ **0.80%** (não 3%)
- gap/cooldown
- **sem** `last_share` nem `non-Friendly` no detector documentado

Nesta VPS H3BUP, `grep` por `last_share` não retornou uso operacional.  
Portanto: use a spec do usuário **como referência analítica pedida**, mas **não** afirme que ela é a policy DT ativa nesta máquina.

### Diferenças que realmente importam
1. **Tipo de sinal:** DT = sequência de quedas de odd; H3BUP = filtro de odd/liquidez/slip sobre audits H3B.  
2. **Thresholds sem overlap direto** (cum_drop/last_share vs odd/cap/slip).  
3. **Estado ops:** H3BUP LIVE; DT bridge inactive/disabled; audit DT failed.  
4. **Stake H3BUP:** 10 fixo; stake DT referência NÃO CONFIRMADO na spec do usuário.  
5. **Não compartilhar N/ROI/CLV entre as duas.**

---

## 4. Tese econômica

### MECANISMO PLAUSÍVEL
Selecionar Back Pre com:
- preço ainda na zona ~2.00,
- profundidade (`limit_final`) > 100,
- e slippage pré-envio negativo (preço no betslip não pior que a decisão),

poderia melhorar o ROI versus Back Pre amplo, segundo estudos internos de hipóteses (H3b/H3BUP).

### MECANISMO COMPROVADO
Nenhum mecanismo causal de mercado está comprovado. Está comprovada a **implementação dos filtros** e a existência de estudos históricos.

### Riscos de tese errada
- Slip negativo como artefato de latência/PMM sem valor preditivo  
- Edge concentrado em poucos eventos/ligas (estudos alertam Top-k / World Cup)  
- Policy sem filtro de ligas ampliar ruído  
- Amostra live ainda minúscula

Invalidadores práticos: ROI OOS ≤ 0 com N adequado; ausência de monotonicidade em capacity/slip; concentração frágil.

---

## 5. Arquitetura ponta a ponta

```text
WS odds
 → audit_h3b_api.py --direction up --mode ws_gate_back
   (systemd: betinasia-audit-ws-gate-back)
 → DB audits (status OK)
 → ops.executor_bridge_audit (hypothesis H3B, side Back, mode live efetivo)
 → _h3bup_vnext_eval + shadow backpre_shadow_all + bypass ligas
 → ExecutionRequest (unix socket /tmp/betinasia-exec.sock)
 → executor worker: dryrun betslip → limit_final/odd_final
 → force stake 10
 → H3BUP_VNEXT_GATE
 → hard stake cap
 → place_order
 → LIVE_OK / CAP_BLOCKED / API_FAILED / ...
 → logs/executor_live.jsonl
 → accounting monitor → CSV balance/open_stakes + daily JSON
```

Componentes **não** no caminho crítico H3BUP: scheduler CLV post_5m/15m/closing dedicado; detector DT; prepare/commit split (só planejado em docs DT).

---

## 6. Fontes de dados

| Fonte | Uso H3BUP |
|---|---|
| WebSocket | auditoria H3B |
| Betslip API | odd_final, limit_final, place |
| PostgreSQL | audits, shadow, seen keys |
| executor_live.jsonl | verdade operacional de execução |
| accounting CSV/JSON | settlement/PnL (pode atrasar) |
| bridge_risk_params.json | disable_back |
| wf_policy_current.json | legado ligas (bypass) |
| estudos CSV | pesquisa histórica |

Timezone: timestamps do executor em UTC (`Z`). Accounting `post date` sem tz explícito → cuidado.

---

## 7. Lógica de detecção

H3BUP **não detecta tendência**.  
O “sinal” a montante é o audit H3B `ws_gate_back` (direction up). Rise filter existe (`GATE_RISE_RATIO=1.02` @5s) mas **`GATE_BACK_ENFORCE_RISE_FILTER=0`** (não enforced).

Fórmulas do gate H3BUP:
```text
slippage_pre_pct = (odd_pre_submit - odd_at_decision)/odd_at_decision*100
capacity = dry.limit_final
```

Dedup bridge: seen-key TTL (`BRIDGE_SEEN_KEY_TTL_SEC=600`, hard 86400).  
Cooldownplica/out-of-order WS: detalhes finos NÃO CONFIRMADO.

---

## 8. Execução

Fluxo após candidato aprovado no bridge:
1. Request com `stake_requested=10`, `policy_version=H3BUP_vNext_20260629`
2. Dryrun abre betslip e captura snapshot
3. Gates H3BUP
4. `place_order` com `sent.stake=10`

Status observados no jsonl H3BUP: `LIVE_OK`, `CAP_BLOCKED`, `API_FAILED`, `STALE`, `NO_SESSION`.

Reasons de CAP_BLOCKED comuns:
- `capacity_lte_100`
- `slippage_non_negative`
- (historicamente, antes do fix, capacity era **sempre null** → falso `capacity_lte_100`)

---

## 9. Kickoff

`BRIDGE_PREMATCH_ONLY=1`.  
Árvore completa de precedência de kickoff/best_kickoff/conflitos: **NÃO CONFIRMADO** neste pacote (existe infra em relatórios gerais do monorepo, não revalidada aqui para o path H3BUP).

---

## 10. Stake

**Atual (FATO):**
- 10 USD fixo
- env: `EXECUTOR_LIVE_STAKE=10`, `EXECUTOR_LIVE_MAX_STAKE=10`, `BRIDGE_STAKE=10`
- fast-path enable=0
- hard cap código impede Back Pre LIVE com stake ≠ 10 ou sem H3BUP

**Histórico:** leak de stake 20 em 20/07/2026 (bridge DT / fast-path HI=20), depois contido.

Exposição diária/por liga específica H3BUP: NÃO CONFIRMADO além de seen-keys e disable_back.

---

## 11. CLV e fair edge

**Não fazem parte da policy H3BUP.**  
Existem em analytics (`analyze_contexto_operacao_b808_robust_report.py`) com definições de estudo (ex.: CLV vs closing odd).  

Para os 8 LIVE_OK atuais: CLV/fair edge **NÃO CONFIRMADO** (não calculados neste handoff).  
Não tratar CLV, fair edge, slippage_pre_pct e ROI como equivalentes.

---

## 12. Performance (corte 2026-07-29T11:03:41Z)

### Pós-fix do gate (≥ 2026-07-28T13:19:39Z)
| Métrica | Valor |
|---|---|
| H3BUP attempts | 59 |
| LIVE_OK | **8** |
| CAP_BLOCKED | 49 |
| API_FAILED | 2 |
| stake sent | 10 em 100% dos LIVE_OK |
| notional | 80 USD |
| odd_final média | ~1.936 |
| limit_final | 112.8 … 655.3 |
| slip médio LIVE_OK | ~-1.93% |

### Settlement reconciliado (accounting CSV mais recente no audit: 2026-07-28 22:02Z)
Apenas **2/8** order_ids aparecem:
- `1931674091` → amount **-9.97** (settled)
- `1932353274` → amount **0.00** (settled)

ROI total da estratégia live: **NÃO CONFIRMADO** (amostra liquidada incompleta + CSV stale).  
Não extrapolar ROI a partir de N=2.

### Antes do fix (20/07 18:47Z → 28/07 13:19Z)
0 LIVE_OK H3BUP; centenas de CAP_BLOCKED com `sg.capacity=null` apesar de `limit_final` frequentemente >100 → bug.

### Accounting agregado (JSON 28/07 22:02Z)
`balance_current≈1106.4` — inclui toda a conta, **não** isola H3BUP.

---

## 13. Validação estatística

### Live H3BUP
Classificação: **INSUFFICIENT_N** / **STATISTICALLY_INCONCLUSIVE**.

### Estudos (pesquisa; não idênticos ao live atual)
- `estudo_ligas_odds_backpre.py` define **H3b** = slip<0 + odd 1.85–2.15 + capacity>100  
- `estimar_capacidade_h3bup.py` / `estudo_capacidade_escala_h3bup.py` estudam escala/capacidade  
- `docs/estudo_5ms_pre_pos_20260611.md` avalia Back Pre slip-neg amplo (Pre/Pos 25/05) e falha M1/M2 de robustez no amplo — **não é a policy H3BUP exata**

Números pontuais desses estudos **não foram reexecutados** nesta auditoria com a base atual → não inventar ROI histórico aqui.

Respostas-chave:
- ROI≠0 live? inconclusivo  
- OOS live? insuficiente  
- Correlação CLV/fair↔ROI live? NÃO CONFIRMADO  
- Risco de overfitting de faixa de odd/ligas: reconhecido nos estudos

---

## 14. Riscos

1. **Estatístico:** N live pequeno; ROI não reconciliado.  
2. **Operacional:** dependência de proxy; drop-in systemd shadow vs modo live efetivo (confuso).  
3. **Dados:** accounting pode atrasar → falsos “sem PnL”.  
4. **Conceitual:** confundir H3BUP com DT ou com H3B audit genérico.  
5. **Regressão:** reintroduzir `float(limit_final)` bare no worker quebraria fills de novo.  
6. **Exposição:** ausência confirmada de daily cap dedicado H3BUP (NÃO CONFIRMADO).

---

## 15. Problemas conhecidos

| Item | Status |
|---|---|
| Bug capacity `limit_final` undefined | RESOLVIDO (28/07; usa `dry.limit_final`) |
| Leak stake 20 | RESOLVIDO/contido |
| Proxy SG block | RESOLVIDO (ops) |
| Accounting stale vs fills novos | ABERTO |
| Shadow flag capacity_gt100=0 no bridge | PARCIAL (esperado se capacity unknown pre-exec) |
| DT audit failed / bridge off | ABERTO (fora do H3BUP) |
| CLV scheduler H3BUP | ABERTO / ausente |
| Spec DT usuário vs código DT | CONFLITO documental |

---

## 16. Mudanças recentes (cronologia curta)

- **2026-06-29:** nasce H3BUP_vNext (sem ligas, stake 10, shadow).  
- **2026-07-20:** proxy restaurado; `disable_back=false`; em seguida vazamento stake 20; contenção (DT off, reject non-H3BUP, hard cap).  
- **2026-07-28 13:19Z:** fix capacity gate → primeiros LIVE_OK H3BUP.  
- **2026-07-28→29:** 8 LIVE_OK stake 10 observados até o corte.

---

## 17. Feature flags (valores encontrados)

| Flag | Valor |
|---|---|
| EXECUTOR_ALLOW_LIVE | 1 |
| BRIDGE_MODE efetivo | live |
| BRIDGE_HYPOTHESIS | H3B |
| BRIDGE_PREMATCH_ONLY | 1 |
| disable_back | false |
| EXECUTOR_BACKPRE_FAST_STAKE_ENABLE | 0 |
| EXECUTOR_LIVE_STAKE/MAX | 10 |
| GATE_BACK_ENFORCE_RISE_FILTER | 0 |
| BRIDGE_USE_WF_BUDGET | 0 |
| DRY_RUN | false |

Não há env `H3BUP_*`; ativação por substring em `policy_version`.

---

## 18. Controles

Ativos: reject non-H3BUP Back Pre; hard stake 10; gates odd/cap/slip; policy lock; DT bridge disabled; close betslip on block; accounting monitor; api-error-guard; proxy soak cron.  
`disable_back` disponível como kill switch (atualmente false).  
Daily cap H3BUP dedicado: NÃO CONFIRMADO.

---

## 19. Perguntas abertas (priorizadas)

**Críticas**
1. Settlements completos dos LIVE_OK H3BUP (accounting fresco).  
2. Onde (se existir) vive a regra DT `cum_drop>3% + last_share≥0.35 + non-Friendly`.  
3. Critérios exatos do audit H3B para gravar candidato consumível pelo bridge.  
4. Kickoff resolver/precedência no prematch.

**Importantes**
5. Reexecutar estudos H3b na base atual.  
6. Medir latências ponta a ponta.  
7. Definir coleta CLV forward para H3BUP.  

---

## 20. Fontes principais

1. VPS read-only 2026-07-29: systemd, `.env`, `executor/worker.py`, `ops/executor_bridge_audit.py`, `logs/executor_live.jsonl`, accounting, shadow DB, lock file.  
2. Repo patches: `apply_h3bup_vnext_policy_patch.py`, `apply_h3bup_vnext_gate_and_shadow_enrichment.py`, `ops/patch_h3bup_capacity_from_dry_limit.py`, verifiers/analyzers.  
3. Estudos: `estudo_ligas_odds_backpre.py`, `estimar_capacidade_h3bup.py`, `estudo_capacidade_escala_h3bup.py`, `docs/estudo_5ms_pre_pos_20260611.md`.  
4. DT comparação: branch/PR #26 docs+detector+env; systemd DT na VPS.  
5. Spec de referência do usuário (mensagem 2026-07-29).  
6. Dossiê completo irmão: `strategy_complete_handoff_dossier_20260729.md`.

### Verificação rápida para a IA receptora
- Policy reproduzível? **Sim.**  
- Edge estatístico live comprovado? **Não.**  
- Pode misturar métricas DT e H3BUP? **Não.**  
- Alguma mudança ops foi feita para gerar este pacote? **Não. Nenhuma mudança operacional foi realizada.**

**Status do handoff:** `HANDOFF_MOSTLY_COMPLETE`


---

## Anexo A — Pseudocódigo operacional completo

```text
# Bridge loop (simplificado)
for row in fetch_candidates(source_statuses=OK, newest_first=True):
  if not reserve_seen_key(row): continue
  ev = h3bup_vnext_eval(row)
  record_shadow(row, ev)
  if exec_side != Back: continue
  if not ev.is_h3bup_vnext_pre_exec:
    mark_seen(skipped=h3bup_vnext_pre_exec_rejected); continue
  if risk_params.disable_back:
    shadow(approved_not_executed or rejected); continue
  # league policy bypass for H3BUP
  submit ExecutionRequest(
    policy_version="H3BUP_vNext_20260629",
    stake_requested=10,
    odd_at_decision=row.odd,
    market/side/line/event from row
  )

# Executor _execute_unlocked (Back LIVE)
dry = execute_dryrun(req)  # opens betslip, sets odd_final, limit_final
if dry.status != DRY_OK: return dry + LIVE_PRECHECK_FAILED
price = dry.odd_final or req.odd_at_decision
if H3BUP: stake = 10
slip = (price - odd_at_decision)/odd_at_decision*100
cap = dry.limit_final
reasons = []
if odd not in [1.85,2.15]: reasons += ["odd_outside_1.85_2.15"]
if cap is None or cap <= 100: reasons += ["capacity_lte_100"]
if slip is None: reasons += ["slippage_missing"]
elif slip >= 0: reasons += ["slippage_non_negative"]
if reasons:
  close_betslip; return CAP_BLOCKED("H3BUP_VNEXT_GATE " + join(reasons))
if stake != 10: return CAP_BLOCKED(hard_stake_cap...)
return place_order(stake=10, price=price)
```

## Anexo B — Tabela LIVE_OK pós-fix (jsonl)

Corte: 2026-07-29T11:03:41Z. Todos `policy=H3BUP_vNext_20260629`, `sent.stake=10`.

| finished_at UTC | order_id | event_id | odd_dec→final | limit_final | slip% | audit_id |
|---|---|---|---|---|---|---|
| 2026-07-28T15:30:30Z | 1931674091 | 2026-07-28,60848,652 | 1.882→1.862 | 655.31 | -1.06 | 419662 |
| 2026-07-28T17:58:48Z | 1932353274 | 2026-07-28,324,10003088 | 1.892→1.877 | 321.24 | -0.79 | 419935 |
| 2026-07-28T23:59:02Z | 1933822208 | 2026-07-29,23470,23456 | 1.980→1.945 | 129.59 | -1.77 | 420463 |
| 2026-07-29T04:00:57Z | 1934662052 | 2026-07-30,1583,1624 | 1.952→1.914 | 122.18 | -1.95 | 420839 |
| 2026-07-29T07:51:59Z | 1934994839 | 2026-07-29,23475,29682 | 1.961→1.934 | 120.64 | -1.38 | 421000 |
| 2026-07-29T08:28:34Z | 1935045057 | 2026-07-29,10059479,10088422 | 2.038→2.030 | 112.77 | -0.39 | 421030 |
| 2026-07-29T10:42:51Z | 1935402833 | 2026-07-29,11,70 | 2.068→1.940 | 502.79 | -6.19 | 421392 |
| 2026-07-29T10:59:28Z | 1935460146 | 2026-07-29,184,305 | 2.029→1.990 | 113.92 | -1.92 | 421512 |

Settlement no CSV `20260728_220133__balance.csv` (stale relativo aos fills de 29/07):
- 1931674091 amount=-9.97 settled
- 1932353274 amount=0.00 settled
- demais: **não presentes** nesse snapshot

## Anexo C — Como NÃO confundir nomes

1. **H3B** = hipótese/auditoria legada (`audit_h3b_api`, direction up).  
2. **H3BUP / H3BUP_vNext** = policy de execução (odd/cap/slip/stake10).  
3. **H3b** = nome da hipótese nos estudos estatísticos (= regra H3BUP).  
4. **DT** = Downward Trend (detector de quedas consecutivas).  
5. **Back Pre** = Back em pré-jogo; não confundir com `request.is_live=true` (flag de execução LIVE na API).  
6. **capacity** no gate = `dry.limit_final`, não `last_share`.  
7. **slippage_pre_pct** ≠ CLV ≠ fair edge ≠ ROI.

## Anexo D — Incidentes que moldaram o estado atual

### D.1 Proxy morto (≈ jul/2026 pré-20/07)
Sintoma: `login_country_not_allowed` country=SG.  
Ação: novo proxy `200.234.174.52:12323`; `disable_back=false` em ~2026-07-20T15:16:08Z.

### D.2 Leak stake 20 (20/07)
Mesmo com intenção H3BUP stake 10/1.5, path legado (bridge-DT + fast-path HI=20) enviou stake 20.  
Mitigações: DT stop/disable; reject non-H3BUP Back Pre; force stake 10; hard cap; alinhar `.env` (fonte real do processo) além de drop-ins systemd.

### D.3 Bug capacity (20/07 18:47Z → 28/07 13:19Z)
Gate usava variável `limit_final` inexistente em `_execute_unlocked`; `except` → `cap_val=None` → sempre `capacity_lte_100`.  
Prova: 693 gated com `sg.capacity=null` enquanto `result.limit_final` podia ser >100.  
Fix: `cap_val = float(dry.limit_final)...`.  
Resultado: fills retomam.

## Anexo E — Matriz referência vs H3BUP (completa)

| Dimensão | Referência (spec usuário) | H3BUP_vNext |
|---|---|---|
| tipo de sinal | DT cum drop + share | filtro odd/cap/slip |
| direção | Back | Back |
| momento | Pre | Pre |
| mercado | NÃO CONFIRMADO na spec | AH predominante nos fills |
| threshold | cum_drop>3%; last_share≥0.35 | odd 1.85–2.15; cap>100; slip<0 |
| non-Friendly | sim na spec | não (liga filter off) |
| stake | NÃO CONFIRMADO | 10 |
| executor | infra BetinAsia | mesmo stack |
| estado nesta VPS | DT offline | LIVE controlado |
| CLV no gate | NÃO CONFIRMADO | não |
| maturidade | stack DT em PR/docs | live com 8 fills pós-fix |

## Anexo F — Controles e o que verificar amanhã

Checklist read-only recomendado para a próxima IA:
1. `verify_h3bup_capacity_gate_source.py` → ok?  
2. Contar LIVE_OK H3BUP desde 13:19Z e conferir `sent.stake==10`.  
3. Nos CAP_BLOCKED, `sg.capacity` null? (regressão) ou numérico coerente com `limit_final`?  
4. Puxar accounting fresco e casar order_ids.  
5. Confirmar `disable_back=false`, DT inactive, bridge-back active.  
6. Não misturar PnL da conta inteira com PnL H3BUP.

## Anexo G — Limitações explícitas deste pacote

- Não reexecutou estudos estatísticos H3b com a base CSV atual.  
- Não calculou CLV/fair edge dos 8 LIVE_OK.  
- Não mapeou schema SQL completo de todas as tabelas de audit.  
- Não validou end-to-end o resolver de kickoff.  
- Não afirma que a spec DT do usuário está implementada nesta VPS.  
- Accounting usado para settlement estava defasado em relação aos fills de 29/07.

## Anexo H — Arquivos-chave no repositório (handoff)

- `betinasia_bot/apply_h3bup_vnext_policy_patch.py`  
- `betinasia_bot/apply_h3bup_vnext_gate_and_shadow_enrichment.py`  
- `betinasia_bot/ops/patch_h3bup_capacity_from_dry_limit.py`  
- `betinasia_bot/ops/verify_h3bup_capacity_gate_source.py`  
- `betinasia_bot/ops/verify_h3bup_post_restore.py`  
- `betinasia_bot/ops/analyze_h3bup_post_restore_deep.py`  
- `betinasia_bot/estudo_ligas_odds_backpre.py`  
- `betinasia_bot/estimar_capacidade_h3bup.py`  
- `betinasia_bot/docs/estudo_5ms_pre_pos_20260611.md`  
- `betinasia_bot/docs/handoff/strategy_complete_handoff_dossier_20260729.md`  
- VPS-only: `ops/executor_bridge_audit.py`, `executor/worker.py`

## Anexo I — Respostas objetivas da verificação final

1. Policy reproduzível? Sim.  
2. Policy confirmada em produção? Sim.  
3. Tese documentada? Sim (plausível).  
4. Fluxo técnico mapeado? Sim, com lacunas de schema audit.  
5. Execução confirmada? Sim (8 LIVE_OK).  
6. Campos críticos documentados? Em grande parte.  
7. Timestamps confiáveis? Executor sim; accounting parcial.  
8. Kickoff bem definido? Parcial.  
9. Stake confirmado? Sim (10).  
10–11. CLV/fair edge H3BUP? Não operacionais.  
12. ROI reconciliado? Não completamente.  
13–14. Amostra/OOS suficientes? Não.  
15. Concentração? INSUFFICIENT_N.  
16. Abertos: accounting, CLV, DT spec vs código, kickoff details.  
17. Diferenças importantes vs referência: tipo de sinal e thresholds.  
18. Inferências: mecanismo de mercado e ROI futuro.  
19. Dados faltantes: settlements completos; clarificação DT; reestudo.  
20. Suficiente para análise independente? Sim para entendimento; não para concluir edge.  
21. Fontes: VPS, repo, estudos, PRs, spec usuário.  
22. Conflitos: DT thresholds; BRIDGE_MODE drop-in vs live.  
23. Confiança geral: média–alta técnica; baixa estatística live.  
24. Mudança operacional? **Não. Nenhuma mudança operacional foi realizada.**

**Status:** HANDOFF_MOSTLY_COMPLETE


## Anexo J — Contrato mínimo para a IA receptora

Ao iniciar uma análise nova, a IA receptora deve:

1. Tratar H3BUP_vNext como policy de execução Back Pre com odd∈[1.85,2.15], capacity=`dry.limit_final`>100, slip<0, stake=10, sem filtro de ligas.
2. Não importar regras DT (`cum_drop`, `last_share`, non-Friendly) para H3BUP sem evidência nova.
3. Separar bases: estudos históricos H3b ≠ live pós-28/07 ≠ PnL da conta inteira.
4. Exigir order_id↔accounting antes de declarar ROI.
5. Verificar regressão do gate: `sg.capacity` não pode voltar a ser sempre null.
6. Preservar disciplina: NÃO CONFIRMADO / INFERÊNCIA / CONFLITO.
7. Não recomendar aumento de exposição neste pacote; ele é só transferência de contexto.

Se precisar de detalhe máximo por seção (33 seções), abrir o dossiê completo irmão no mesmo diretório.
