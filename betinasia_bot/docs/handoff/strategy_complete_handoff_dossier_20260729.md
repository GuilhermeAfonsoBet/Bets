# Dossiê completo de handoff — H3BUP_vNext
**Data do pacote:** 2026-07-29  
**Timestamp de corte das métricas operacionais VPS:** 2026-07-29T11:03:41Z  
**Gerado em:** 2026-07-29T11:07:29Z  
**Natureza:** documentação / auditoria / transferência de contexto  
**Mudanças operacionais nesta tarefa:** Não. Nenhuma mudança operacional foi realizada.

---

## 0. Escopo e disciplina epistêmica

Esta estratégia é **H3BUP_vNext** (Back Pre com filtros de odd, capacidade e slippage), operando no stack BetinAsia.

Regras deste documento:
- Fato só com evidência.
- Lacunas marcadas como **NÃO CONFIRMADO**.
- Inferências marcadas como **INFERÊNCIA — não comprovada diretamente**.
- Conflitos entre fontes apresentados com confiança.
- A estratégia de referência DT é usada **apenas para comparação**, nunca como substituto de dados H3BUP.

Fontes principais:
1. Código/patches no repositório `GuilhermeAfonsoBet/Bets` (branches H3BUP e DT).
2. VPS `root@178.128.55.30` `/home/betbot/Bets/betinasia_bot` (systemd, `.env`, worker/bridge, jsonl, accounting, shadow DB) — **leitura**.
3. Conversas/agentes cloud sobre proxy restore, leak stake 20, fix capacity gate (Jul/2026).
4. Estudos locais (`estudo_*`, `estimar_*`, `docs/estudo_5ms_*`).
5. Spec de referência fornecida pelo usuário para comparação: `DT Back Pre | cum_drop_pct > 3% | non-Friendly | last_share >= 0.35 | LIVE_OK`.

---

## 1. RESUMO EXECUTIVO DA ESTRATÉGIA

### Resumo (≤15 linhas)

H3BUP_vNext é uma política operacional de **Back pré-jogo (Back Pre)** no BetinAsia Exchange (mercado típico AH), que só permite LIVE quando: odd de decisão ∈ **[1.85, 2.15]**, capacidade do betslip (**`dry.limit_final`**) **> 100**, e **slippage_pre_pct < 0** medido imediatamente antes do place. Stake fixa **10 USD**. Não usa filtro de ligas. Não usa `cum_drop`, `last_share` nem regras DT. O sinal de origem operacional atual vem do auditor H3B `ws_gate_back` (`audit_h3b_api.py --direction up`), consumido pelo bridge-back, com policy version `H3BUP_vNext_20260629`. Estágio atual: **live controlado** (executor+bridge-back LIVE, stake 10, DT bridge desligado). Pós-fix do gate de capacity (2026-07-28 13:19Z) há **8 LIVE_OK** com stake 10; amostra liquidada reconciliada ainda é pequena/incompleta no último CSV de accounting (corte 28/07 22:02Z). Edge esperado deriva de estudos Back Pre com slip negativo + faixa perto de 2.00 + liquidez; **não** há CLV/fair edge como filtro de entrada.

### Explicação detalhada (perguntas 1–11)

1. **Qual é a estratégia?** Filtro operacional H3BUP_vNext sobre candidatos Back Pre do pipeline audit→bridge→executor.  
2. **Comportamento de mercado:** oportunidades pré-jogo em que a odd ainda está na faixa ~2.00 e o preço no momento do envio não piorou vs decisão (`slippage_pre_pct < 0`), com liquidez suficiente.  
3. **Hipótese econômica:** Back Pre com slippage favorável e liquidez acima de limiar concentra edge histórico observado em estudos de P&L reconciliado; faixa 1.85–2.15 é a zona operacional escolhida (H3b).  
4. **Evento que dispara o sinal:** auditoria H3B WS gate Back (direction up) grava audits elegíveis; bridge seleciona Back Pre e aplica `_h3bup_vnext_eval`. O “sinal” H3BUP não é um detector de queda acumulada.  
5. **Mercado:** principalmente Asian Handicap (AH) via bet type `for,ah,...` — outros mercados podem aparecer no audit genérico; **NÃO CONFIRMADO** se H3BUP restringe estritamente a AH no bridge além do que o audit envia.  
6. **Back / Lay:** **Back** apenas no fluxo H3BUP bridge-back.  
7. **Pré / live / híbrida:** **pré-jogo** (`BRIDGE_PREMATCH_ONLY=1`; gate exige `not market_is_live` / Back Pre). O campo `request.is_live=true` no ExecutionRequest significa “execução LIVE permitida na API”, não “jogo ao vivo”.  
8. **Janela temporal:** pré-kickoff (prematch); janela exata em minutos **NÃO CONFIRMADO** como threshold único além do prematch.  
9. **Expectativa de edge:** estudos H3b/H3BUP estimam ROI positivo em bases históricas filtradas; números pontuais dos estudos **dependem da base CSV usada na execução do script** e **não foram reexecutados nesta auditoria** → valores históricos detalhados: ver seção 16/17; live pós-fix ainda **INSUFFICIENT_N**.  
10. **Principal risco de tese errada:** seleção em liquidez alta/odd perto de 2 sem edge real; ou edge só em subconjunto (ligas/eventos) que a policy atual não filtra; ou slip negativo ser artefato de latência/PMM sem valor preditivo.  
11. **Estágio:** **live controlado** (não shadow; não live pleno sem limites). Controles: stake 10, hard reject non-H3BUP Back Pre, capacity gate, slip gate, DT desligado.

---

## 2. IDENTIDADE DA ESTRATÉGIA

### CONFIGURAÇÃO ATUAL
| Campo | Valor | Evidência | Confiança |
|---|---|---|---|
| Nome oficial | H3BUP_vNext | `apply_h3bup_vnext_policy_patch.py` `POLICY_ID` | alta |
| Nome interno | H3BUP / H3b (estudo) | estudos + conversas | alta |
| Aliases | H3BUP_vNext_20260629; “H3b”; às vezes “H3BUP” vs “H3B” | patches / estudos | alta |
| strategy_id | **NÃO CONFIRMADO** (não há campo strategy_id dedicado; usa policy_id/version) | — | — |
| policy_id | `H3BUP_vNext` | lock + patch | alta |
| Versão atual | `H3BUP_vNext_20260629` | bridge inject + lock | alta |
| Data de criação (policy started) | `2026-06-29T00:00:00+00:00` | `POLICY_STARTED_AT` | alta |
| Data de ativação LIVE recente | proxy restore + `disable_back=false` ~`2026-07-20T15:16:08Z`; corte operacional pós-leak ~`18:47Z`; capacity fix `2026-07-28T13:19:39Z` | risk_params + logs | alta |
| Freeze | lock `logs/H3BUP_VNEXT_POLICY_LOCK` ativo | VPS file | alta |
| Status atual | LIVE controlado, stake 10 | systemd + jsonl | alta |
| Repositório | `github.com/GuilhermeAfonsoBet/Bets` | git remotes | alta |
| Branches relevantes | `cursor/h3bup-capacity-gate-fix-47ee`, `cursor/h3bup-post-restore-review-47ee`, `cursor/roi-only-hardening-47ee`, `cursor/proxy-dead-failover-47ee` | git/gh | alta |
| Serviços | `betinasia-executor`, `betinasia-executor-bridge-back`, `betinasia-audit-ws-gate-back`, accounting monitor/timers | systemd | alta |
| Responsável técnico/operacional | **NÃO CONFIRMADO** (owner email cloud: guimaafonso@msn.com — não prova papel ops) | cloud run-info | baixa |

### CONFIGURAÇÃO HISTÓRICA
- Pré-H3BUP: Back Pre com policy de ligas WF (`wf_policy_current.json`), sizing fast-path (HI até 20).
- Leak 20/07: bridge-DT e/ou fast-path HI=20 enviaram stake 20 fora da intenção H3BUP.
- Bug capacity (até 28/07 13:19Z): gate lia `limit_final` inexistente → sempre `capacity_lte_100`.

### CONFIGURAÇÃO PLANEJADA
- **NÃO CONFIRMADO** roadmap formal além de docs DT (prepare/commit) e estudos de escala.

### NÃO CONFIRMADO
- `strategy_id` formal, dono ops único, data de “freeze estatístico” da amostra de pesquisa.

---

## 3. DEFINIÇÃO EXATA DA POLICY

### A. Linguagem natural (literal)

Uma ordem Back só pode ser enviada LIVE sob H3BUP_vNext se **todas** forem verdadeiras no gate final do executor:
1. Policy version contém `H3BUP_vNext`.
2. Operação é **Back Pre** (Back e mercado não ao vivo).
3. `odd_at_decision` ∈ **[1.85, 2.15]**.
4. Capacidade `dry.limit_final` **> 100**.
5. `slippage_pre_pct < 0`, com  
   `slippage_pre_pct = (odd_pre_submit - odd_at_decision) / odd_at_decision * 100`.
6. Stake enviada **= 10** (force + hard cap).
7. Filtro de ligas **desabilitado** para H3BUP (bypass no bridge).

No bridge, pre-exec exige Back Pre + odd band; capacity ausente/0 é `capacity_unknown_pre_exec` (não bloqueia pre-exec); capacity conhecida ≤100 rejeita; slip final é no executor.

### B. Expressão lógica

```text
is_h3bup_req = "H3BUP_vNext" in policy_version

pre_exec_bridge =
    exec_side == Back
    AND NOT market_is_live
    AND 1.85 <= odd_at_decision <= 2.15
    AND (capacity is NULL OR capacity <= 0 OR capacity > 100)
    # se capacity conhecida e <=100 => reject capacity_lte_100

final_exec_ok =
    is_h3bup_req
    AND exec_side == Back
    AND NOT market_is_live
    AND 1.85 <= odd_at_decision <= 2.15
    AND dry.limit_final is not NULL
    AND dry.limit_final > 100
    AND slippage_pre_pct is not NULL
    AND slippage_pre_pct < 0
    AND stake == 10

live_back_submit_allowed = pre_exec_bridge AND final_exec_ok
# adicional: non-H3BUP Back Pre => rejected (defense in depth)
```

### C. Pseudocódigo

```text
function evaluate_bridge(row):
  odd = odd_at_decision(row)
  cap = betslip_limit(row) or value_sizing.max_stake/capacity/limit
  if cap <= 0: cap = unknown
  is_backpre = side==Back and not is_live_market
  is_odd = odd in [1.85, 2.15]
  is_cap = None if unknown else (cap > 100)
  pre_ok = is_backpre and is_odd
  if not pre_ok: reject/shadow rule_rejected; do not live-submit
  if is_cap is False: reject capacity_lte_100
  bypass league policy
  submit ExecutionRequest(policy_version=H3BUP_vNext_20260629, stake_requested=10)

function evaluate_executor(req, dry):
  if Back Pre and "H3BUP_vNext" not in policy: CAP_BLOCKED non_h3bup_backpre_rejected
  stake = 10 if H3BUP else default_without_fast_bump
  slip = (price - odd_at_decision)/odd_at_decision*100
  cap = dry.limit_final
  reasons = []
  if odd not in [1.85,2.15]: reasons += odd_outside_1.85_2.15
  if cap is None or cap <= 100: reasons += capacity_lte_100
  if slip is None: reasons += slippage_missing
  elif slip >= 0: reasons += slippage_non_negative
  if reasons: CAP_BLOCKED H3BUP_VNEXT_GATE ...
  if stake != 10: CAP_BLOCKED hard_stake_cap_h3bup_only_10
  place_order(stake=10, price=dry.odd_final)
```

### O que governa a policy em produção
1. Código patchado em `ops/executor_bridge_audit.py` e `executor/worker.py` na VPS.  
2. Lock `logs/H3BUP_VNEXT_POLICY_LOCK` impede overwrite por publisher de ligas.  
3. `logs/wf_policy_current.json` existe (ligas) mas é **bypassado** para H3BUP.  
4. `logs/bridge_risk_params.json` (`disable_back`) pode pausar Back.  
5. Env stake: `EXECUTOR_LIVE_STAKE=10`, fast-path enable=0, HI=10.

**Não são filtros H3BUP:** `cum_drop_pct`, `last_share`, non-Friendly, CLV, fair edge, placar, tempo de jogo.

---

## 4. COMPARAÇÃO COM A ESTRATÉGIA DE REFERÊNCIA

### Definição da referência (fornecida pelo usuário)
```text
DT Back Pre
cum_drop_pct > 3%
non-Friendly
last_share >= 0.35
LIVE_OK
```

### CONFLITO DE FONTES — DT

| Versão | Fonte | Data/versão | Conteúdo | Confiança |
|---|---|---|---|---|
| A — Spec usuário (comparação) | mensagem desta tarefa | 2026-07-29 | cum_drop>3%, last_share≥0.35, non-Friendly, LIVE_OK | alta como *pedido de comparação* |
| B — Stack DT no repo/PR #26 | `DOWNWARD_TREND_OPERATION.md`, detector, env example | branch `cursor/downward-trend-strategy-1c5f` | 3 downs, step≥0.20%, cum≥0.80%, gap, cooldown; **sem** last_share/non-Friendly no doc do detector | alta para *código DT documentado* |
| C — VPS DT service | systemd DT (inactive) + zz-dt-params | audit 2026-07-29 | `DT_MIN_CUM_DROP_PCT=0.80`, step 0.20, downs 3, gap 10s, cooldown 45; bridge-dt inactive/disabled | alta para *config presente* |

**Mais confiável para “o que o código DT faz”:** Versão B/C.  
**Usada na matriz abaixo como referência analítica:** Versão A (pedido do usuário), com nota de que A **não foi encontrada** como filtros ativos no tree H3BUP VPS inspecionado (`last_share` grep vazio).

### Matriz comparativa

| Dimensão | Referência (spec usuário) | H3BUP_vNext | Diferença | Impacto provável |
|---|---|---|---|---|
| tipo de sinal | queda acumulada DT | filtro sobre audits H3B Back Pre + gates odd/cap/slip | conceitualmente distinto | não comparáveis 1:1 |
| direção | Back | Back | igual | — |
| momento | Pre | Pre | igual | — |
| mercado | NÃO CONFIRMADO na spec; DT docs AH/linha | AH predominante nos fills | possível overlap | médio |
| threshold | cum_drop>3%; last_share≥0.35 | odd 1.85–2.15; cap>100; slip<0 | totalmente diferente | alto |
| janela temporal | NÃO CONFIRMADO | prematch | ? | ? |
| kickoff | NÃO CONFIRMADO na spec | prematch bridge; resolver detalhado NÃO CONFIRMADO | ? | ? |
| odd | não na spec usuário | 1.85–2.15 | H3BUP restringe odd | alto |
| liquidez | last_share≥0.35 (spec) | limit_final>100 | métricas diferentes | alto |
| last_share | ≥0.35 | **não usado** | crítico | alto |
| cumulative drop | >3% | **não usado** | crítico | alto |
| stake | NÃO CONFIRMADO na spec | 10 fixo | — | — |
| executor | compartilhado (infra) | mesmo stack BetinAsia | reuso | — |
| scheduler | DT report dedicado (docs) | daily ROI/accounting geral | diferente | médio |
| CLV | NÃO CONFIRMADO | não é filtro; analytics genéricos | — | — |
| fair edge | NÃO CONFIRMADO | não é filtro | — | — |
| risco | tendência/momentum de queda | seleção por preço/liquidez/slip | perfis distintos | alto |
| frequência | NÃO CONFIRMADO | baixa pós-gates (8 LIVE_OK ~1d) | — | — |
| variância | NÃO CONFIRMADO | amostra live pequena | — | — |
| dados | DT reports isolados (docs) | jsonl + shadow + accounting | — | — |
| maturidade | PR DT draft; VPS DT inactive/failed audit | H3BUP LIVE controlado | H3BUP mais “em produção” agora | alto |

### Diferenças agrupadas
1. **Conceituais:** DT = sequência de quedas; H3BUP = qualidade de preço/liquidez/slip em Back Pre.  
2. **Policy:** thresholds sem interseção direta.  
3. **Estatísticas:** bases e hipóteses diferentes; não misturar N/ROI.  
4. **Técnicas:** entrypoints (`audit_downward_trend_api.py` vs `audit_h3b_api.py`), hypothesis DT vs H3B/H3BUP.  
5. **Operacionais:** DT bridge inactive; H3BUP bridge-back LIVE.  
6. **Risco:** DT depende de definição de streak; H3BUP de medição de limit_final/slip.  
7. **Qualidade de dados:** H3BUP tem shadow `backpre_shadow_all`; DT tem reports/dirs dedicados.  
8. **Maturidade:** H3BUP live pós-hotfix; DT stack separado ainda não ativo nesta VPS.  
9. **CLV:** nenhum dos dois usa CLV como gate H3BUP; coleta DT-specific NÃO CONFIRMADO nesta VPS.  
10. **Execução:** mesmo executor possível; path de serviço diferente; H3BUP força stake 10.

---

## 5. TESE ECONÔMICA E MECANISMO ESPERADO

### MECANISMO PLAUSÍVEL
- Slippage pré-submit negativo indica que o preço no betslip ainda não piorou vs decisão → menor adverse selection imediata.
- Capacidade >100 seleciona mercados com mais profundidade, historicamente associados a melhor comportamento em estudos de capacidade.
- Faixa 1.85–2.15 concentra perto de even-money, zona estudada como mais estável que Back Pre amplo.

### MECANISMO ESPECULATIVO
- “Quem” move o mercado (sharp/recreacional/MM): **NÃO CONFIRMADO**.
- Persistência temporal do edge pós-sinal: **NÃO CONFIRMADO** em minutos exatos.
- Relação causal limpa slip→ROI vs confounder de latência/PMM.

### MECANISMO COMPROVADO
- **Nenhum mecanismo causal de mercado está comprovado** neste handoff. O que está comprovado é a **implementação dos filtros** e a **existência de estudos estatísticos históricos** sobre Back Pre filtrado.

### MECANISMO REFUTADO
- Capacidade sempre ≤100 no gate (era bug de variável): **refutado como realidade de mercado**; era bug de código (corrigido 28/07).

### Respostas 1–10 (síntese)
1. Edge viria de melhor preço de entrada relativo + seleção de liquidez.  
2. Agente do movimento: **NÃO CONFIRMADO**.  
3. Tipo: mistura plausível de correção de preço / liquidez / comportamento; não classificado empiricamente.  
4. Por que preço ainda incorreto: **INFERÊNCIA** — janela pré-jogo antes de convergência.  
5. Duração do edge: **NÃO CONFIRMADO**.  
6. Desaparecimento: convergência para fair / kickoff / retirada de liquidez — **INFERÊNCIA**.  
7. Invalidadores: ROI OOS≤0 após N adequado; slip<0 sem poder preditivo; concentração extrema.  
8. Edge deveria aparecer primeiro em: estudos enfatizam **ROI realizado**; CLV/fair edge são analytics, não gate.  
9. Alternativas: seleção por latência; bias de amostra WC; overfitting de faixa de odd.  
10. Evidências: estudos `estudo_ligas_odds_backpre` (H3b), `estimar_capacidade_h3bup`, doc 5Ms (Back Pre slip-neg amplo — **não idêntico** a H3BUP).

---

## 6. FLUXO TÉCNICO PONTA A PONTA

```text
WebSocket Odds (BetinAsia)
  ↓
audit_h3b_api.py --direction up --mode ws_gate_back
  (serviço: betinasia-audit-ws-gate-back)
  ↓
Persistência audits (DB betslip_audit* / status OK)   [detalhe schema: parcial]
  ↓
ops.executor_bridge_audit (BRIDGE_HYPOTHESIS=H3B, EXEC_SIDE=Back, MODE=live efetivo)
  ↓
Filtros bridge: source OK, prematch, seen-key TTL, risk_params.disable_back
  ↓
_h3bup_vnext_eval (pre-exec)
  ↓
Shadow backpre_shadow_all (+ bypass league)
  ↓
ExecutionRequest via unix socket /tmp/betinasia-exec.sock
  ↓
betinasia-executor / worker._execute_unlocked
  ↓
execute_dryrun → abre/captura betslip → odd_final, limit_final
  ↓
Sizing force stake=10
  ↓
H3BUP_VNEXT_GATE (odd, dry.limit_final, slip)
  ↓
hard_stake_cap_h3bup_only_10
  ↓
place_order → LIVE_OK / API_FAILED / ...
  ↓
logs/executor_live.jsonl
  ↓
Accounting monitor → balance/open_stakes CSV + accounting_daily_report.json
  ↓
Daily timers (ROI/full report) — genéricos
  ↓
CLV/fair edge: NÃO fazem parte do caminho crítico H3BUP; existem em analytics B808/outros
```

### Por etapa (resumo estruturado)

| Etapa | Módulo/serviço | I/O chave | Falhas | Idempotência |
|---|---|---|---|---|
| Ingestão WS | audit-ws-gate-back | odds snapshots | WS gap, service fail | NÃO CONFIRMADO detalhe |
| Detecção H3B | audit_h3b_api ws_gate_back | audits OK | rise filter opcional off | NÃO CONFIRMADO |
| Bridge poll | executor_bridge_audit | candidates | disable_back, non_h3bup | seen keys TTL 600/86400 |
| Policy pre-exec | `_h3bup_vnext_eval` | bool + reasons | odd/cap unknown | shadow_id versioned |
| Queue/exec | unix socket executor | ExecutionRequest | NO_SESSION, STALE | execution_id UUID |
| Prepare betslip | dryrun API | limit_final, odd_final | NO_VALID_BOOKMAKER_PRICES | betslip cache |
| Gate | worker H3BUP | CAP_BLOCKED reasons | capacity/slip/odd | close betslip on block |
| Place | place_order | LIVE_OK + order_id | HTTP 403 country, etc. | — |
| Persist | jsonl + DB save flag | rows | disk | append-only jsonl |
| Accounting | accounting-monitor/daily | CSV/JSON | stale snapshot | — |

Diagrama textual completo pedido: ver bloco acima. Etapas “register/scheduler CLV/prepare-commit DT” **não** estão no caminho H3BUP atual.

---

## 7. ARQUITETURA DE DADOS

### Fontes

| Fonte | Tipo | Frequência | Confiabilidade | Uso |
|---|---|---|---|---|
| WebSocket odds | stream | contínuo | média–alta (gaps possíveis) | audit H3B |
| Betslip API | request | on-demand | alta para limit/odd no momento | executor |
| PostgreSQL local | DB | contínuo | alta | audits, shadow, bridge seen |
| Bookmaker exchange | BetinAsia | — | alta p/ accounting | settlement |
| logs/executor_live.jsonl | JSONL | cada execução | alta p/ status/stake/slip | auditoria exec |
| logs/executor_bridge_back.log | text | contínuo | média | bypass/debug |
| logs/accounting/*.csv | CSV | periódico | alta mas **snapshot pode atrasar** | PnL |
| logs/accounting_daily_report.json | JSON | daily/monitor | média | balance/pnl agregado |
| logs/bridge_risk_params.json | JSON | reload ~5s | alta | disable_back |
| logs/wf_policy_current.json | JSON | publish (locked) | alta como artefato; **não governa H3BUP** | legado ligas |
| Estudos CSV /tmp | derivados | sob demanda | depende de reconciliação | pesquisa |

### Tabelas/arquivos críticos
- `backpre_shadow_all`: shadow amplo H3BUP; PK lógica `shadow_id`; flags is_h3bup_vnext, capacity, slip, statuses.  
- `executor_live.jsonl`: request/result aninhados; status; raw.sent; raw.value_sizing; raw.slippage_gate; order_id.  
- Accounting balance: `order id`, `amount` (PnL settlement), `got price`, `status`, event fields.  
- Timezone: timestamps executor tipicamente UTC ISO Z; accounting `post date` sem tz explícito nos CSV → tratar com cautela.

### Dicionário (campos críticos)

| Campo | Definição | Fonte | Unidade | NULL? | Confiança |
|---|---|---|---|---|---|
| order_id | id ordem exchange | result.raw.order_id / balance | id | raro em LIVE_OK | alta |
| event_id | chave evento | request.event_id | string | pode | alta |
| match_id | id partida | request | id | frequente null | média |
| audit_id | id audit origem | request.audit_id | int | pode | alta |
| policy_version | versão policy | request.policy | string | heartbeat vazio | alta |
| market_type | AH etc | request | enum | | alta |
| side/line | seleção | request | | | alta |
| exec_side | Back/Lay | request | | | alta |
| odd_at_decision | odd na decisão | request | odd | | alta |
| odd_final / accepted | odd no betslip/envio | result / sent.price | odd | | alta |
| stake_requested | stake policy | policy | USD | | alta |
| sent.stake | stake enviada | raw.sent | USD | | alta |
| limit_final / capacity | liquidez betslip | dry/result | USD lim | pode null se falha precheck | alta pós-fix no gate |
| slippage_pre_pct | (price-dec)/dec*100 | value_sizing | % | | alta |
| pnl / amount | settlement | accounting amount | USD | só liquidado | alta quando presente |
| last_share | — | **não no H3BUP** | — | — | N/A |
| cum_drop_pct | — | **não no H3BUP** | — | — | N/A |
| clv_raw / fair_edge | analytics B808 | estudos | % | cobertura parcial | média p/ analytics; N/A p/ gate |
| kickoff_ts | | fixture/providers | UTC? | | NÃO CONFIRMADO detalhe H3BUP path |
| candidate_id/strategy_id | | | | | NÃO CONFIRMADO |

---

## 8. LÓGICA DE DETECÇÃO DO SINAL

### O que H3BUP “detecta”
H3BUP **não** implementa detector próprio de tendência. Ele **filtra** candidatos já auditados.

### Detector a montante (H3B audit)
- Serviço: `betinasia-audit-ws-gate-back`
- Comando: `audit_h3b_api.py --direction up --mode ws_gate_back`
- Env: `GATE_RISE_OFFSET_SEC=5`, `GATE_RISE_RATIO=1.02`, mas `GATE_BACK_ENFORCE_RISE_FILTER=0` (filtro rise **não enforced**)
- Medidas offsets `0,5,10` com workers async

Respostas objetivas:
1. Acompanhamento inicia no WS audit H3B.  
2. Preço base: odd_at_decision do audit/bridge.  
3. Movimento relevante no **gate H3BUP** é o slip pre-submit, não cum_drop.  
4–9. Debounce/cooldown H3BUP: seen-key no bridge (TTL); cooldown DT **não se aplica**.  
10. Múltiplos steps: N/A ao filtro H3BUP.  
11–12. last_share/cum_drop: **não calculados** no path H3BUP.  
13. Encerra com CAP_BLOCKED/LIVE_OK/API_FAILED etc.  
14. Mesmo evento pode gerar múltiplas tentativas se não marcado seen — mitigado por seen keys.  
15. Dedup: `_mark_seen` / TTL.  
16. Bridge: polling (`BRIDGE_POLL_SEC` default NÃO CONFIRMADO no processo live; DT example 0.25). Audit: event-driven WS.  
17. Latência: fills observados com pre_submit_ms ~1–2s em exemplos; distribuição completa NÃO CONFIRMADO nesta auditoria.  
18. Preços: WS + betslip API.  
19. Normalização de linha: herdada do audit/API — detalhes NÃO CONFIRMADO.  
20. Mercado suspenso: falhas tipo NO_VALID_BOOKMAKER_PRICES observadas.

Fórmulas:
```text
slippage_pre_pct = (odd_pre_submit - odd_at_decision) / odd_at_decision * 100
capacity = dry.limit_final   # pós-fix 2026-07-28
```

---

## 9. POLICY GATE E CRIAÇÃO DO CANDIDATO

- **Momento bridge:** antes do submit live; grava shadow.  
- **Momento executor:** imediatamente antes do place_order, após dryrun.  
- **Aprovação:** ver seção 3.  
- **Rejeições reais observadas/codificadas:**
  - `capacity_lte_100`
  - `slippage_non_negative`
  - `slippage_missing`
  - `odd_outside_1.85_2.15`
  - `non_h3bup_backpre_rejected` / `H3BUP_VNEXT_REQUIRED`
  - `hard_stake_cap_h3bup_only_10`
  - `h3bup_vnext_pre_exec_rejected` / `non_h3bup_live_blocked`
  - `capacity_unknown_pre_exec` (bridge reason; não bloqueia pre-exec)
  - `rule_rejected` (shadow)
  - `disabled_back` / `operational_disabled_back`
  - falhas API: `LIVE_PRECHECK_FAILED`, `NO_SESSION`, `STALE`, `login_country_not_allowed` (histórico proxy)

Race: capacity no audit pode ser unknown; medição confiável só no executor — design intencional.

---

## 10. FLUXO DE EXECUÇÃO

Elegível → ExecutionRequest(stake=10, policy H3BUP) → dryrun betslip → gates → place_order → jsonl.

### Status reais observados (H3BUP jsonl)
`LIVE_OK`, `CAP_BLOCKED`, `API_FAILED`, `STALE`, `NO_SESSION`, `HEARTBEAT` (não-H3BUP/telemetria).

**Não observados nesta auditoria H3BUP (não listar como existentes sem prova):** PARTIAL_FILL, AFTER_KICKOFF, etc. → **NÃO CONFIRMADO** no path atual.

Falhas:
- Antes: rule_rejected, disable_back, non_h3bup block  
- Durante: CAP_BLOCKED gates, STALE, NO_SESSION  
- Após envio: API_FAILED  
- Fill: LIVE_OK com order_id  
- Reconciliação: accounting CSV pode atrasar vs jsonl  

---

## 11. CRONOLOGIA E TIMESTAMPS

| Timestamp | Origem | TZ | Confiabilidade |
|---|---|---|---|
| request.created_at | executor request | UTC Z | alta |
| result.finished_at | executor | UTC Z | alta |
| value_sizing.pre_submit_ms | derivado | ms | alta |
| accounting post date | ledger | **tz implícito** | média |
| shadow created_at | DB | timestamptz | alta |
| risk_params updated_at | JSON | UTC | alta |

Latências agregadas first_seen→LIVE_OK etc.: **NÃO CONFIRMADO** (não calculadas exaustivamente nesta auditoria).  
Exemplo pontual: pre_submit_ms=1222 no primeiro LIVE_OK pós-fix.

---

## 12. KICKOFF E REGRAS TEMPORAIS

- Bridge: `BRIDGE_PREMATCH_ONLY=1`.  
- Árvore completa de best_kickoff / confidence levels no path H3BUP: **NÃO CONFIRMADO** nesta auditoria (existe infra em reports B808/daily, não revalidada aqui).  
- Kickoff NULL: comportamento exato **NÃO CONFIRMADO**.  
- Ordens após kickoff: gate Back Pre usa `market_is_live` do meta; robustez end-to-end **NÃO CONFIRMADO**.

---

## 13. STAKE E EXPOSIÇÃO

### CONFIGURAÇÃO ATUAL
- Stake fixa **10 USD**
- `EXECUTOR_LIVE_STAKE=10`, `EXECUTOR_LIVE_MAX_STAKE=10`, `BRIDGE_STAKE=10`
- Fast-path enable **0**; HI/LO=10
- Hard cap código: Back Pre LIVE só stake==10 com H3BUP
- `disable_back=false`
- Bankroll ref env `BRIDGE_BANKROLL_REF=3000` (uso efetivo de budget WF: `BRIDGE_USE_WF_BUDGET=0`)
- Exposição por liga/evento dedicada H3BUP: **NÃO CONFIRMADO** além de seen-keys

### CONFIGURAÇÃO HISTÓRICA
- Fast HI=20; leak stake 20 em 20/07 via DT/fast-path

### PLANEJADA
- NÃO CONFIRMADO

---

## 14. CLV, CLOSING LINE E FAIR EDGE

**H3BUP não define nem gateia por CLV/fair edge.**

Em analytics B808 (`analyze_contexto_operacao_b808_robust_report.py`):
- CLV exemplo: `(bs_odd - closing_odd)/closing_odd*100` (AH; closing via histórico pré-kickoff) — formulário de estudo, não do executor H3BUP.

Para H3BUP live atual:
- CLV raw médio dos 8 LIVE_OK: **NÃO CONFIRMADO** (não calculado nesta auditoria)
- Fair edge: **NÃO CONFIRMADO**
- Cobertura closing jobs dedicados H3BUP: **NÃO CONFIRMADO / provavelmente ausente como scheduler dedicado**

Separação obrigatória:
- **CLV RAW** ≠ **FAIR EDGE** ≠ **ROI REALIZADO** ≠ **slippage_pre_pct**

---

## 15. SCHEDULER DE CLV E BETSLIPS

### FLUXO ATUAL REAL (H3BUP)
- Sem scheduler post_5m/post_15m/closing dedicado identificado para H3BUP.
- Betslips abertos no dryrun do executor; fechados em CAP_BLOCKED / cleanup.
- Timers gerais: accounting-daily, daily-full-report, api-error-guard, ops-monitor, ops-autopilot.

### FLUXO ORIGINAL PREVISTO / DT
- Docs DT mencionam prepare/commit futuro e daily DT report — **não é H3BUP**.

### PROBLEMAS HISTÓRICOS
- too_many_open_betslips (mitigações de close/timeout existem no executor)
- Proxy morto → login_country_not_allowed SG

### LACUNAS
- Coleta sistemática de closing/CLV para fills H3BUP: aberta.

---

## 16. PERFORMANCE ATUAL

**Corte VPS:** 2026-07-29T11:03:41Z

### Pós-fix capacity (≥ 2026-07-28T13:19:39Z) — base jsonl H3BUP
| Métrica | Valor |
|---|---|
| tentativas H3BUP | 59 |
| LIVE_OK | 8 |
| CAP_BLOCKED | 49 |
| API_FAILED | 2 |
| stake enviada (LIVE_OK) | 10.0 × 8 = 80 USD notional |
| odd_final média LIVE_OK | ~1.9365 |
| limit_final min/max LIVE_OK | 112.77 / 655.31 |
| slip médio LIVE_OK | ~-1.93% |
| liquidadas no CSV accounting 20260728_220133 | 2 de 8 (outras ausentes no snapshot) |
| PnL reconciliado parcial | -9.97 + 0.00 = **-9.97** (apenas 2 ordens) |
| ROI parcial liquidado | -9.97 / ~20 ≈ **-49.9%** (N=2; **não inferir ROI da estratégia**) |
| CLV/fair edge | NÃO CONFIRMADO |
| abertas | NÃO CONFIRMADO no snapshot stale (open CSV sem esses order_ids) |

### Desde corte pós-restore leak (≥ 2026-07-20T18:47Z) — H3BUP
| Métrica | Valor |
|---|---|
| H3BUP rows | ~846 (auditoria anterior) / status mix |
| LIVE_OK antes do fix | 0 |
| CAP_BLOCKED | centenas, quase todos capacity_lte_100 (muitos falsos) |
| false bug signature (cap null & limit>100) | 232 |

### Estudos históricos H3b/H3BUP (pesquisa)
Resultados numéricos exatos dos scripts **não reexecutados** aqui com a base CSV da VPS → **NÃO CONFIRMADO neste pacote como números atuais**. Ver código de relatório em `estimar_capacidade_h3bup.py` / `estudo_ligas_odds_backpre.py`.

### Janelas OOS/in-sample
Definidas nos estudos; **não** recalculadas nesta auditoria.

---

## 17. VALIDAÇÃO ESTATÍSTICA

Testes existentes **no código de estudo** (não necessariamente rerodados agora):

| Teste | Base | N | Resultado | Interpretação | Limitação |
|---|---|---|---|---|---|
| Bootstrap ROI | estudos Back Pre/H3b | variável | via `_bootstrap_events` | prob ROI>0 | depende da base |
| Permutation | idem | variável | `_perm_p` | vs zero | múltiplos testes |
| 5Ms score | doc 2026-06-11 Pre/Pos | 375/206 | M1/M2 FAIL em regimes amplos slip-neg | edge não robusto no amplo | **não é H3BUP exato** |
| OOS temporal | estudo ligas/odds | variável | tabelas no script | estabilidade faixa odd | data mining risk |
| Walk-forward ligas | WF policy | — | lista ligas | legado; H3BUP remove filtro | não governa H3BUP |
| Capacidade vs ROI | estimar_capacidade | variável | correlação reportada pelo script | escala | concentracao |

Perguntas:
1. ROI≠0? Live H3BUP: **INSUFFICIENT_N**. Estudos: depende da regra/base — não consolidado aqui.  
2. ROI acima de meta? **NÃO CONFIRMADO** (meta numérica não fixada neste handoff).  
3–4. Correlação CLV/fair↔ROI para H3BUP live: **NÃO CONFIRMADO**.  
5. Monotonicidade: estudos de buckets capacity — não reavaliada.  
6–8. Leave-one-out eventos/ligas/books: infraestrutura nos estudos; live N/A.  
9. OOS suficiente live: **não**.  
10. Poder estatístico live (n=8): baixo.  
11. IC cruzam zero: estudos 5Ms sim (FAIL).  
12. Múltiplos testes: risco declarado nos estudos de ligas.  
13. Otimização retroativa: estudos tentam hipóteses pré-definidas; risco residual existe.  
14–15. Selection/survivorship: risco em bases LIVE_OK-only.

---

## 18. FAIR EDGE VS ROI

Para H3BUP live atual: **INSUFFICIENT_N** / métricas **NÃO CONFIRMADO**.  
Classificação: **INSUFFICIENT_N**.  
Não usar fair edge para alterar produção (regra do pedido).

---

## 19. CONCENTRAÇÃO E ROBUSTEZ

Live n=8: classificação **INSUFFICIENT_N**.  
Estudos alertam dependência Top eventos / WC — ver `estimar_capacidade_h3bup.py` textos de escala.

Settled parcial: 1 loss (~-10) + 1 zero → concentração trivial, sem valor estatístico.

---

## 20. QUALIDADE E COBERTURA DOS DADOS

| Campo/Métrica | Esperado | Disponível | Cobertura | Qualidade | Risco |
|---|---|---|---|---|---|
| LIVE_OK jsonl | completo | sim | alta | alta | baixo |
| stake sent | 10 | sim nos 8 | 100% fills | alta | baixo |
| capacity no gate | preenchida | sim pós-fix | 3/3 blocks amostrados | alta | bug histórico |
| accounting settlement | todos fills | parcial (2/8 no CSV 28/07 22:02) | baixa–média | CSV stale | alto p/ ROI |
| shadow flags capacity_gt100 | preenchidas | 0 no pós-fix (flag) | baixa utilidade bridge | capacity unknown pre-exec | médio |
| CLV/closing H3BUP | desejável | não operacionalizado | baixa | — | alto p/ validação preço |
| kickoff confiável | sim | parcial | NÃO CONFIRMADO | — | médio |
| last_share/cum_drop | N/A H3BUP | ausentes | N/A | — | — |

---

## 21. PROBLEMAS CONHECIDOS

| Problema | Impacto | Status | Correção | Evidência |
|---|---|---|---|---|
| Gate capacity lia `limit_final` undefined | 0 fills; falsos CAP_BLOCKED | RESOLVIDO | `dry.limit_final` 28/07 | verify ok; 8 LIVE_OK |
| Stake 20 leak (DT/fast-path) | exposição indevida | RESOLVIDO (controles) | disable DT; reject non-H3BUP; stake hard cap; env=10 | commits + services |
| Proxy morto / SG country | LIVE bloqueado | RESOLVIDO (ops) | novo proxy + disable_back false | risk_params |
| Accounting CSV atrasado vs fills | ROI incompleto | ABERTO | — | 6/8 ausentes no último balance |
| Shadow `is_capacity_gt_100` sempre 0 | telemetria bridge fraca | PARCIALMENTE RESOLVIDO | capacity no executor | DB flags |
| Audit WS DT failed | DT offline | ABERTO (fora H3BUP) | — | systemd failed |
| Bridge drop-in shadow vs env live | confusão operacional | SUSPEITO/ Parcial | processo usa live | systemd vs /proc |
| CLV scheduler H3BUP | validação preço | ABERTO | — | não encontrado |
| last_share/cum_drop>3% spec vs código DT | comparação ambígua | ABERTO/doc | — | conflito fontes |

---

## 22. MUDANÇAS RECENTES

| Data | Mudança | Motivo | Resultado | Status |
|---|---|---|---|---|
| 2026-06-29 | Policy H3BUP_vNext + shadow | ativar regra sem ligas | version pin | aplicado |
| 2026-07-20 | Proxy restore; disable_back=false | SG block | LIVE volta | feito |
| 2026-07-20 | Leak stake 20 | DT/fast-path | PnL legado negativo na janela | mitigado depois |
| 2026-07-20~ | Patches reject non-H3BUP, hard stake 10, bridge-only live | contenção | 0 non-H3BUP requests pós-corte | aplicado VPS |
| 2026-07-28 13:19Z | Fix dry.limit_final | capacity sempre null | 8 LIVE_OK | aplicado |
| PRs #27–#29 | proxy/review/capacity docs+patches | hardening | draft | abertos |

---

## 23. FEATURE FLAGS E CONFIGURAÇÃO ATUAL

| Flag | Valor atual | Efeito | Risco |
|---|---|---|---|
| EXECUTOR_ALLOW_LIVE | 1 | permite LIVE | alto se mal gated |
| BRIDGE_MODE | live (efetivo) | submete | alto |
| BRIDGE_HYPOTHESIS | H3B | origem candidatos | médio |
| BRIDGE_PREMATCH_ONLY | 1 | só pre | médio |
| BRIDGE_REQUIRE_POLICY | 1 | exige policy | baixo (bypass H3BUP ligas) |
| disable_back | false | Back on | alto |
| EXECUTOR_BACKPRE_FAST_STAKE_ENABLE | 0 | sem bump | baixo |
| EXECUTOR_LIVE_STAKE/MAX | 10 | cap | médio |
| EXECUTOR_BACK_PRE_SLIP_NEG_GATE | default on | slip gate | médio |
| GATE_BACK_ENFORCE_RISE_FILTER | 0 | rise não força | médio |
| BRIDGE_USE_WF_BUDGET | 0 | sem budget WF | médio |
| DRY_RUN | false | real | alto |
| H3BUP_* env | inexistente | policy no código | — |

Defaults históricos distintos: **NÃO CONFIRMADO** todos; listar só valores encontrados.

---

## 24. CONTROLES OPERACIONAIS

| Controle | Status |
|---|---|
| disable_back kill | IMPLEMENTADO / ATIVO (via JSON; atualmente false) |
| hard stake cap 10 | IMPLEMENTADO / ATIVO |
| reject non-H3BUP Back Pre | IMPLEMENTADO / ATIVO |
| capacity+slip+odd gate | IMPLEMENTADO / ATIVO (pós-fix) |
| policy publish lock | IMPLEMENTADO / ATIVO |
| DT bridge disabled | ATIVO |
| daily cap H3BUP específico | NÃO CONFIRMADO |
| dedup seen keys | IMPLEMENTADO / ATIVO |
| betslip close on block | IMPLEMENTADO |
| accounting monitor | ATIVO |
| api-error-guard timer | ATIVO |
| proxy soak cron | ATIVO |
| testes automatizados H3BUP gate | PARCIAL (scripts verify; não suite CI completa confirmada) |

---

## 25. TESTES AUTOMATIZADOS

| Área | Coberta? | Testes | Lacunas |
|---|---|---|---|
| DT detector | sim (branch DT) | `tests/test_downward_trend_detector.py` | não é H3BUP |
| H3BUP capacity source | script verify | `verify_h3bup_capacity_gate_source.py` | não é pytest CI |
| Post-restore ops | script | `verify_h3bup_post_restore.py` | manual |
| Estatística H3b | scripts estudo | vários | precisam base CSV |
| E2E place order | NÃO CONFIRMADO | — | — |

Última execução CI: **NÃO CONFIRMADO**.

---

## 26. SERVIÇOS, TIMERS E PROCESSOS

| Componente | Função | Status |
|---|---|---|
| betinasia-executor | dryrun/LIVE | active/enabled |
| betinasia-executor-bridge-back | audit→exec Back | active/enabled LIVE efetivo |
| betinasia-executor-bridge-dt | DT | inactive/disabled |
| betinasia-executor-bridge-lay | Lay | active mas unit disabled; LAY off |
| betinasia-audit-ws-gate-back | sinal H3B | active |
| betinasia-audit-ws-gate-dt | sinal DT | failed |
| betinasia-accounting-monitor | accounting | active |
| timers accounting/full-report/api-error-guard/ops-* | ops | active |
| cron results/proxy/slippage/auth | ops | active |

Paths: `/home/betbot/Bets/betinasia_bot`, user `betbot`, logs sob `logs/`.

---

## 27. RELATÓRIOS E OUTPUTS DISPONÍVEIS

| Arquivo | Conteúdo | Período | Confiabilidade | Uso |
|---|---|---|---|---|
| logs/executor_live.jsonl | execuções | contínuo | alta | performance live |
| logs/executor_bridge_back.log | bridge | contínuo | média | debug |
| logs/accounting/* | balance/open | até 2026-07-28 22:02 no audit | alta mas stale | PnL |
| logs/accounting_daily_report.json | agregados | 2026-07-28T22:02Z | média | balance 1106.4 |
| backpre_shadow_all | shadow | contínuo | alta | funil |
| docs/estudo_5ms_pre_pos_20260611.md | 5Ms slip-neg | até ~2026-06-10 | média | contexto histórico ≠ H3BUP exato |
| PRs #27–#29 | mudanças | jul/2026 | alta | rastreio |
| este dossiê | handoff | 2026-07-29 | — | transferência |

---

## 28. EVIDÊNCIAS E RASTREABILIDADE (amostra)

| Afirmação | Evidência | Fonte | Confiança |
|---|---|---|---|
| Policy version H3BUP_vNext_20260629 | constantes + lock + jsonl policy | patch/VPS/jsonl | alta |
| Stake 10 nos LIVE_OK | raw.sent.stake=10 | jsonl pós-fix | alta |
| Fix capacity usa dry.limit_final | worker L1567 + verifier ok | VPS | alta |
| 8 LIVE_OK pós-fix | contagem jsonl | VPS 2026-07-29T11:03:41Z | alta |
| PnL total H3BUP | só 2/8 no CSV | accounting | baixa p/ total |
| DT cum_drop>3%+last_share | pedido usuário; não no código VPS H3BUP | conflito | — |

---

## 29. PERGUNTAS AINDA NÃO RESPONDIDAS

### A. Críticas
1. ROI/settlement completo dos 8+ fills H3BUP? (accounting fresco)  
2. Definição canônica da referência DT com last_share/cum_drop>3% — onde vive no código/dados?  
3. Schema completo do audit H3B e critérios exatos de `status=OK` que alimentam o bridge.  
4. Kickoff resolver usado no prematch bridge (precedência/conflitos).

### B. Importantes
5. Distribuição latência policy→LIVE_OK.  
6. Cobertura CLV forward para H3BUP.  
7. Reexecução dos estudos H3b na base atual.  
8. Exposição agregada/open stakes atual pós-29/07.

### C. Secundárias
9. Donos RACI.  
10. Inventário completo de testes CI.

---

## 30. GLOSSÁRIO

| Termo | Definição | Ambiguidades |
|---|---|---|
| H3BUP / H3BUP_vNext | Policy Back Pre odd/cap/slip stake10 | ≠ H3B audit genérico |
| H3b | Hipótese de estudo = regra H3BUP | nome de pesquisa |
| H3B | Hipótese/auditoria legada direction up | não é a policy final |
| Back Pre | Back em mercado pré-jogo | `is_live` no request pode ser true (API live) |
| dry.limit_final | liquidez capturada no dryrun | “capacity” no gate |
| slippage_pre_pct | variação % odd decisão→pre-submit | ≠ CLV |
| CAP_BLOCKED | bloqueio de envio por gate | não é rejeição exchange |
| LIVE_OK | ordem aceita | ≠ liquidada |
| DT | Downward Trend | thresholds conflitantes entre spec usuário e código |
| last_share | métrica da spec de referência | **ausente no H3BUP** |
| cum_drop_pct | queda acumulada | DT; não H3BUP |
| shadow | oportunidade registrada sem necessariamente executar | backpre_shadow_all |
| WF policy | policy walk-forward de ligas | bypassed por H3BUP |

---

## 31. AVALIAÇÃO FINAL

| Dimensão | Status | Justificativa |
|---|---|---|
| entendimento da tese | MOSTLY_CLEAR | tese plausível; mecanismo causal não comprovado |
| clareza da policy | CLEAR | thresholds literais no código |
| documentação técnica | MOSTLY_CLEAR | patches+VPS; worker não no git main |
| qualidade dos dados | DATA_RISK | accounting stale vs fills |
| execução | MOSTLY_CLEAR | 8 LIVE_OK stake10 |
| scheduler | PARTIALLY_CLEAR | sem CLV scheduler H3BUP |
| kickoff | PARTIALLY_CLEAR | prematch on; detalhes N/C |
| CLV | UNKNOWN | não no path |
| fair edge | UNKNOWN | não no path |
| ROI | STATISTICALLY_INCONCLUSIVE | N liquidado insuficiente |
| estatística | PARTIALLY_CLEAR | estudos existem; live inconclusivo |
| robustez | INSUFFICIENT via DATA_RISK | n pequeno |
| controles | MOSTLY_CLEAR | caps/gates ativos |
| testes | PARTIALLY_CLEAR | verifies; CI N/C |
| risco operacional | TECHNICAL_RISK residual | confusão shadow drop-in; dependência proxy |

---

## 32–33. Ver seção de verificação final e artefatos irmãos
Ver `strategy_context_pack_for_new_ai_20260729.md` e `strategy_handoff_structured_20260729.json`.

### VERIFICAÇÃO FINAL OBRIGATÓRIA
1. Reproduzível? **Sim, policy sim; mecanismo econômico não.**  
2. Policy confirmada? **Sim.**  
3. Tese documentada? **Sim (plausível, não comprovada).**  
4. Fluxo mapeado? **Sim (com lacunas audit schema).**  
5. Execução confirmada? **Sim (8 LIVE_OK).**  
6. Campos críticos documentados? **Majority sim.**  
7. Timestamps confiáveis? **Executor sim; accounting parcial.**  
8. Kickoff bem definido? **Parcial.**  
9. Stake confirmado? **Sim, 10.**  
10. CLV bem definido p/ H3BUP? **Não (N/A operacional).**  
11. Fair edge bem definido p/ H3BUP? **Não.**  
12. ROI reconciliado? **Não completamente.**  
13. Amostra suficiente? **Não.**  
14. OOS suficiente live? **Não.**  
15. Concentração relevante? **INSUFFICIENT_N.**  
16. Problemas abertos? Accounting stale; CLV; DT spec vs código; kickoff details.  
17. Diferenças importantes vs referência? Sinal/thresholds/last_share/cum_drop — conceitualmente distintas.  
18. Inferências? Mecanismo de mercado; duração do edge; ROI futuro.  
19. Dados a obter? Settlements completos; CLV; reestudo H3b; clarificar DT spec.  
20. Pacote suficiente p/ outra IA? **Sim para operar entendimento; não para concluir edge.**  
21. Fontes? Repo, VPS, estudos, PRs, conversas, spec usuário.  
22. Conflitos? DT thresholds usuário vs código; BRIDGE_MODE drop-in shadow vs live efetivo.  
23. Confiança geral handoff? **Média–alta técnica; baixa estatística de edge live.**  
24. Mudança operacional? **Não. Nenhuma mudança operacional foi realizada.**

**Status final:** `HANDOFF_MOSTLY_COMPLETE`
