# Fase 2A — Root cause accounting monitor (2026-07-29)

## Diagnóstico por camada

| Camada | Verificação | Resultado | Evidência |
|---|---|---|---|
| processo | systemd active | vivo, mas degradado | `ActiveEnterTimestamp=2026-07-28 06:45:47Z` até stop controlado |
| timer | poll 300s | a correr | logs `snapshot ok` a cada ~5 min |
| sessão | root-session | perdida em ~13:26Z; login SPA falhava se click antes de Loading sumir | executor `NO_SESSION`; login probe sem auth XHR |
| proxy | TCP + uso | OK | `PROXY_SERVER` operacional; daily 28/07 usou proxy com sucesso |
| API | balances CSV | OK após sessão | endpoints `/v1/customers/{user}/balances/{balance\|stake}/` |
| parser | schema CSV | OK | cols `order id,amount,type,post date` |
| filesystem | disco/perms | OK | 73G livres; dir `logs/accounting` gravável |
| output | files=None | falso sucesso | logger `[acct] snapshot ok ... files={'balance': None...}` |

## Respostas 1–27

1. Balance: `_download_from_accounting_page` / `_download_via_api` (`balance`)
2. Open stakes: idem (`open_stakes`)
3. Externa: UI `black.betinasia.com/accounting/*` e API `/v1/customers/.../balances/...`
4. Último sucesso monitor: `2026-07-28T13:17:48Z`
5. Primeiro files=None contínuo: `2026-07-28T13:21:00Z` (`dt=0.0s`)
6–12. Sem HTTP/auth explícito no ciclo morto; falha = browser/pipe closed (`pipe closed by peer`)
13. Retorno efectivo vazio (None)
14–15. Parser/schema não eram a causa do ciclo morto
16. Excepções engolidas no loop → logadas como OK
17–20. Sem path/perm/disk full
21–23. Sem lock/tmp órfão relevante
24. **Sim** — processo vivo sem dados
25–26. Serviço simples (não timer); frequência efectiva 300s
27. **Causa raiz exacta (composta):**
   - **Primária operacional (28/07 13:21→):** Playwright/browser morto (`pipe closed`) sem reconnect; ciclos instantâneos `files=None` marcados como sucesso.
   - **Secundária de observabilidade:** `ACCOUNTING_JSONL=logs/accounring_snapshots.jsonl` (typo).
   - **Bloqueio de recuperação (29/07):** sessão `root-session` ausente; login Mollybet SPA só funciona após `Loading...` desaparecer; click prematuro = no-op.
   - Daily oneshot de 28/07 22:02 ainda gerou CSVs (browser fresco).

## Correcção aplicada

1. `ops/accounting_monitor.py`: status estruturados, health JSON, atomic writes, recover browser, API-first download, retries.
2. `scraper/betinasia.py` (patch mínimo sessão): espera SPA login + `save_session` aborta sem `root-session`.
3. Drop-in systemd accounting: JSONL correcto + limites freshness.
4. Daily: secção `Accounting Health — H3BUP` (patch idempotente).
5. Reconcile H3BUP order-level.

**Não alterado:** bridge, executor worker, policy, stake, thresholds.
