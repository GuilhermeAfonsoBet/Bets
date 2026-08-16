# Design — Análise histórica Friendly vs Non-Friendly (H3BUP_vNext)

Data: 2026-07-31  
Versão de classificação: `FRIENDLY_CLASS_V1_20260731`  
Escopo: **read-only / reporting-only**

## Objectivo

Decompor a performance agregada da policy `H3BUP_vNext` / `H3BUP_vNext_20260629`
entre `FRIENDLY`, `NON_FRIENDLY`, `UNCLASSIFIED` e `CONFLICT`, sem alterar
execução, policy, stake, filtros, accounting, CLV worker ou timers.

## Universos

### Primário — H3BUP_vNext exact

- `policy_id = H3BUP_vNext`
- `policy_version = H3BUP_vNext_20260629`
- `status = LIVE_OK`
- `side/exec_side = Back`
- `period/regime = Pre`
- `created_at_utc >= 2026-06-29T00:00:00Z`
- `created_at_utc <= cutoff`

Exclusões: bridge legado, stake 20, DT, Lay, Back In, DRY_OK, CAP_BLOCKED,
API_FAILED, heartbeats, retries sem LIVE_OK.

Dedup: `order_id`. Fallback auditável só sem `order_id`:
`execution_id + live_ok_ts + event_id`.

### Secundário — HISTORICAL_COMPARABLE_BACK_PRE

Apêndice diagnóstico. **Nunca** consolidar ROI com o primário.

## Freeze contract

1. Construir universo de identidade (sem P&L/settlement/CLV).
2. Enriquecer liga/competição (JSONL → CSV map → SQL read-only).
3. Classificar com `FRIENDLY_CLASS_V1_20260731`.
4. Gravar mapping + rules + checksum SHA256.
5. **Só depois** fazer join com accounting e CLV VALID_STRICT.

## Hierarquia de classificação

1. flag estruturada  
2. competition_type / league_type  
3. nome liga/competição  
4. torneio  
5. event name (último)  
6. UNCLASSIFIED  

`UNCLASSIFIED` ≠ `NON_FRIENDLY`. Discordância estruturada vs nome → `CONFLICT`.

## Settlement

`OPEN | SETTLED_DECIDED | VOID_PUSH | MISSING | UNRECONCILED`

- `roi_resolved = pnl_resolved / stake_resolved_total` (**void no denominador**)
- `roi_decided_ex_void` exclui void
- OPEN não entra como perda; MISSING não vira zero

## CLV

Métrica oficial: apenas `VALID_STRICT` em POST_5M / POST_15M / CLOSING.
Source missing e line mismatch reportados em separado.

## Estatística

Unidade `order_id`, cluster `event_id`. Bootstrap + permutação clusterizados.
p-value exploratório — não é gate operacional.

## Segurança

Checksums before/after de policy/risk params. Sem Telegram, sem ordens,
sem betslips, sem restart, sem mudança de env.

## Pacote

`betinasia_bot/ops/h3bup_friendly_analysis/`

Runner: `python -m ops.h3bup_friendly_analysis.run --root <bot_root>`

VPS smoke: `ops/h3bup_friendly_analysis/vps_smoke.sh`
