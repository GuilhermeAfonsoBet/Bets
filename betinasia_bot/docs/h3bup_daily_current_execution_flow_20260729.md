# H3BUP Daily V1 — Fluxo de execução detalhado — 20260729

**Entrypoint oficial:** `python -m ops.daily_full_report`  
**Orquestrador:** `async def run_daily_full(cfg: DailyReportCfg)`  
**Trigger:** `betinasia-daily-full-report.service` ← `betinasia-daily-full-report.timer`

---

## 1. Arranque e configuração

1. systemd inicia o service no slot `OnCalendar=*-*-* 22:00:00 UTC` com `RandomizedDelaySec=180` e `Persistent=true` (catch-up se a máquina esteve down).
2. `main()` carrega `.env` (`ENV_FILE`), configura loguru, instancia `DailyReportCfg(out_dir=DAILY_REPORT_OUT_DIR|logs/daily_reports)`.
3. Defaults relevantes:
   - `report_tz = America/Sao_Paulo` (override `REPORT_TZ`);
   - `send_telegram` de `DAILY_REPORT_TELEGRAM` (default on);
   - `executor_jsonl = EXECUTOR_JSONL|logs/executor_live.jsonl`;
   - OOS/WF knobs via `DAILY_OOS_*` / `DAILY_WF_*`;
   - `skip_accounting` / `skip_oos` via flags.

---

## 2. Identidade do dia e directório

```text
ts  = now(UTC)
day = ts.strftime("%Y%m%d")          # data UTC de GERAÇÃO
day_dir = out_dir / day
day_dir.mkdir(parents=True, exist_ok=True)
```

Implicações:

- um run às 22:00 UTC do dia D escreve em `.../D/`;
- **não** fixa automaticamente “D−1 fechado” como coorte principal do envelope;
- reruns manuais no mesmo UTC day **overwrite** `report_daily.md` / `.pdf`.

---

## 3. Sequência operacional (passos)

### Passo A — Accounting snapshot

- Se `DAILY_SKIP_ACCOUNTING≠1`: chama fluxo `accounting_daily_report` / monitor → CSVs `*__balance.csv`, `*__open_stakes.csv`.
- Calcula `pnl_today` / week / month em **REPORT_TZ** (`America/Sao_Paulo`).
- Persiste `day_dir/accounting_daily_report.json`.

### Passo B — Leitura executor + KPIs

- Lê `executor_live.jsonl` (janela recente / last N).
- KPIs “OK” (`LIVE_OK`/`DRY_OK`) e KPIs rolling **24h**.
- Actividade recente: último LIVE_OK; contagens 1h/6h/24h.
- Pós-accepted funnel: `API_FAILED`, `NO_SESSION`, `RATE_LIMIT`, **`CAP_BLOCKED`**, etc.

### Passo C — Aderência / short-long windows

- Aderência tipicamente **7d UTC** (artefactos JSON por dia).
- Mistura com métricas de execução e contrafactuais de slippage/latência.

### Passo D — OOS / walk-forward (subprocess)

- Salvo `DAILY_SKIP_OOS=1`, lança analyzer OOS/B808 robust report.
- Escreve `wf_policy_YYYYMMDD.json` (canónico do dia) vs reruns `wf_policy_YYYYMMDD_HHMMSS.json`.
- Pode publicar `wf_policy_current.json` sob guards de compatibilidade (`DAILY_WF_COMPAT_*`).
- **Nota:** publish de policy WF é path do monólito V1 legado (multi-hipótese); não é o mesmo que “publicar Daily V2”.

### Passo E — Montagem markdown (secção 0 = `s0`)

Ordem aproximada em `s0`:

1. Cabeçalho executivo / status OOS / policy publish.
2. Banca + `pnl_today/week/month` (REPORT_TZ).
3. **Accounting Health — H3BUP** (patch; `s0.append` pós-fix P0).
4. **H3BUP End-to-End Latency** (patch; fail-open).
5. **H3BUP CLV Forward Collection** (patch; fail-open).
6. Throughput 24h, conversão DB, gaps executor, VPS snapshot.
7. Falhas pós-accepted / CAP_BLOCKED.
8. Checks de latência `call_to_done_ms` p50/p90 24h.
9. Conclusões / checklists.

### Passo F — Secção 1+ (`s1`) resultados reais / ROIw

- Dual cohort:
  - séries P&L acct por **post date UTC**;
  - tabela **ROIw Total** por **dia exec UTC** (`created_at`) com join ledger `order_id`.
- Fórmula ROIw: `(∑pnl / ∑exposure) * 100` (w = exposure-weighted).
- Void-like `pnl≈0` contados; `order_id` missing excluído de num/den.
- Open pode entrar no ROIw Total se presente no ledger (diferente de ROI settled).
- Tese Back Pre fast: dual threshold 6s/5s em `pre_submit_ms`; contrafactuais usam também `call_to_done_ms≤6000`.
- Filtro policy H3BUP **não** aplicado de forma estrita à tabela ROIw principal.

### Passo G — In-sample / anexos / OOS annex

- Texto OOS pode ir como Anexo A.
- Ajuste sensibilidade banca × gate slippage (Anexo B) — best-effort; muitos `except: pass`.

### Passo H — Persistência md/pdf

```text
combined_md = day_dir / "report_daily.md"
combined_md.write_text(...)          # NÃO atómico
subprocess: render_markdown_to_pdf → report_daily.pdf
```

Sem symlink `latest`. Sem last-known-good. Histórico = pastas dia.

### Passo I — Telegram

Se `DAILY_REPORT_TELEGRAM` activo e credenciais presentes:

- `sendDocument` do PDF com caption `Relatório diário BetinAsia ({day})`;
- retries (`DAILY_TELEGRAM_RETRIES`, sleep);
- fallback `sendMessage` se PDF falhar.

---

## 4. Patch scripts e bug P0 (corrigido)

Scripts:

- `ops/patch_daily_accounting_health_section.py`
- `ops/patch_daily_h3bup_e2e_latency_section.py`
- `ops/patch_daily_h3bup_clv_section.py`

Histórico do P0: snippets injectados referiam `out_lines.append(...)` num contexto onde a lista activa era `s0` → `NameError` / crash da montagem se o patch antigo estivesse activo sem rename. **Phase 2R:** snippets e código em `daily_full_report.py` usam `s0.append`.

Ancoragem: markers `# BEGIN/END H3BUP_*_SECTION`; inserção após nota de accounting / secções anteriores.

---

## 5. Tratamento de erros (comportamento auditado)

- Dezenas/centenas de `except Exception: ... pass` no monólito (~88 `pass` associados; ~271 `except Exception`).
- Efeito: secções inteiras desaparecem do PDF **sem** `report_health=FAILED` e sem alerta estruturado.
- Patches H3BUP são fail-open com mensagem `_indisponível_`, mas o resto do relatório continua “verde” visualmente.

---

## 6. Parallelismo com outros jobs

| Job | Hora | Relação |
|---|---|---|
| `betinasia-accounting-daily.timer` | ~22:00 UTC | Pode correr em paralelo; Daily V1 também dispara accounting interno |
| `betinasia-daily-dt-report` | (separado) | **Não misturar** artefactos/métricas DT com H3BUP |
| CLV workers | contínuo | Daily só consome health |

Race possible: CSV balance “latest” muda entre accounting-daily e daily-full — V1 não congela `report_cutoff` único.

---

## 7. Fluxo shadow V2 (não oficial)

Após/ besides V1 (manual ou futuro hook fail-open):

```bash
python -m ops.daily_v2 \
  --report-type DAILY_CLOSED \
  --out-dir logs/daily_v2
```

Flags: `ENABLED=1`, `PUBLISH=0`, `COMPARE_V1=1`, `FAIL_OPEN=1`.  
Saídas: snapshot JSON + md versionados + health + exceptions CSV + compare CSV.  
**Não** envia Telegram oficial. **Não** substitui `report_daily.pdf`.

---

## 8. Checklist operacional pós-run (V1)

1. Existe `logs/daily_reports/{YYYYMMDD}/report_daily.md` e `.pdf`?
2. Telegram `telegram_sent=true` no JSON de retorno?
3. Secções Accounting Health / E2E / CLV presentes (não só “indisponível”)?
4. LIVE_OK 24h e CAP_BLOCKED coerentes com jsonl?
5. ROIw Total vs `pnl_today` — lembrar dual cohort / TZ misturados.
6. Se V2 shadow: ficheiros em `logs/daily_v2/` sem `published/` (PUBLISH=0).
