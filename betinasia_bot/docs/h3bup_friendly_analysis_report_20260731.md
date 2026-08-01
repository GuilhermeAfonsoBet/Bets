# Relatório — Friendly vs Non-Friendly (H3BUP_vNext)

Data: 2026-07-31  
Status: **`FRIENDLY_ANALYSIS_FAILED`** (bloqueio de acesso aos dados operacionais na VPS)

## Sumário

O pacote analítico read-only foi implementado, versionado e testado (28 testes
obrigatórios a verde). A execução histórica sobre o universo real
`H3BUP_vNext_20260629` **não pôde ser concluída** nesta sessão porque o SSH à
VPS `178.128.55.30` devolveu `Permission denied (publickey,password)`.

Sem `executor_live.jsonl`, accounting e CLV da VPS, as respostas Q1–Q38 sobre
P&L/ROI/CLV reais permanecem pendentes. Q39–Q47: **Não** (nada operacional foi
alterado).

## O que foi entregue

Pacote `ops/h3bup_friendly_analysis/`:

- classificação `FRIENDLY_CLASS_V1_20260731` com freeze+checksum antes de P&L
- universo primário exact + apêndice secundário separado
- settlement (`roi_resolved` com void no denominador)
- CLV VALID_STRICT (5m/15m/closing)
- temporal, ligas, bookmakers, concentração, LORO, robustez, bootstrap/permutação
- outputs CSV/JSON/MD/PDF no contrato do brief
- smoke VPS: `vps_smoke.sh`

Testes: `tests/test_h3bup_friendly_analysis.py` (28 passed).

## Como desbloquear

Instalar a pubkey em `docs/h3bup_friendly_analysis_vps_access_20260731.md` e
reexecutar o smoke. Em seguida o relatório oficial será regenerado sob
`logs/h3bup_friendly_analysis/<YYYYMMDD>/<run_id>/`.

## Segurança (esta sessão)

| Verificação | Resultado |
|---|---|
| Policy alterada? | Não |
| Stake alterada? | Não |
| Executor alterado? | Não |
| Accounting alterado? | Não |
| CLV alterado? | Não |
| Timer alterado? | Não |
| Telegram usado? | Não |
| Ordem criada? | Não |
| Betslip aberto? | Não |

## Nota

Fixture local (N=4) correu apenas para validar o pipeline; **não** constitui
resultado oficial H3BUP_vNext.
