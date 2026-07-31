# Auditoria — Timers e publicação Daily H3BUP — 20260729

## 1. Timer oficial (full report)

| Campo | Valor auditado |
|---|---|
| Unit | `betinasia-daily-full-report.timer` |
| OnCalendar | `*-*-* 22:00:00 UTC` |
| Equivalente BRT | **19:00** (`America/Sao_Paulo`, sem DST Brasil desde 2019) |
| Persistent | `true` (dispara catch-up após downtime) |
| RandomizedDelaySec | `180` (0–3 min jitter) |
| Service | `betinasia-daily-full-report.service` |
| Exec | `python -m ops.daily_full_report` |

Implicação: o “dia do relatório” UTC coincide tipicamente com a data civil UTC do disparo (~22:00–22:03), **não** com o fecho estrito de D−1 (embora a maior parte do dia D UTC já tenha decorrido).

## 2. Service

- Carrega EnvironmentFile/`.env` (tokens Telegram, paths, knobs OOS/WF).
- Working directory = root do bot.
- Exit code do processo ≠ garantia de secções completas (por causa de `except: pass`).

## 3. Timers adjacentes

| Timer / unit | Horário | Relação com H3BUP Daily |
|---|---|---|
| `betinasia-accounting-daily.timer` | ~22:00 UTC | Accounting paralelo; pode race no CSV “latest” |
| `betinasia-daily-dt-report` | separado | **DT — não misturar** métricas/artefactos |
| CLV passive/worker services | contínuo | Só insumos; Daily lê health |

## 4. Artefactos de saída

```
logs/daily_reports/{YYYYMMDD}/
  report_daily.md
  report_daily.pdf
  accounting_daily_report.json
  + JSON OOS/WF/aderência/KPIs (variável)
```

Propriedades auditadas:

- **Overwrite** same-day (rerun substitui).
- **Sem** escrita atómica para md/pdf.
- **Sem** symlink `latest`.
- **Sem** last-known-good.
- Histórico = pastas dia.

## 5. Publicação Telegram

| Item | Valor |
|---|---|
| Flag | `DAILY_REPORT_TELEGRAM` (default ligado) |
| API | `sendDocument` PDF |
| Caption | `Relatório diário BetinAsia ({day})` |
| Credenciais | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` |
| Retries | `DAILY_TELEGRAM_RETRIES` (default 2) + sleep |
| Fallback | `sendMessage` texto se PDF falhar |

**V2 shadow não publica** Telegram (`H3BUP_DAILY_V2_PUBLISH=0`).

## 6. Side-effects de policy WF (legado V1)

O monólito pode actualizar `logs/wf_policy_current.json` sob guards. Isto é efeito colateral do job de relatório legado multi-versão — **distinto** de “publicar Daily V2”. Phase 2R shadow V2 **não** altera esse path quando corrido isolado; o timer oficial continua a ser V1.

## 7. Checklist de auditoria de publicação

1. `systemctl status betinasia-daily-full-report.timer` — next elapse ~22:00 UTC.
2. Último `day_dir` tem md+pdf com mtime coerente (+/− jitter).
3. JSON de retorno do service: `telegram_sent`, `pdf_size_mb`, `day_dir`.
4. Confirmar ausência de publicação V2 em `logs/daily_v2/published/` (esperado vazio com PUBLISH=0).
5. Não ingerir outputs DT no mesmo pipeline analítico H3BUP.

## 8. Riscos

- Persistent=true após outage longa: múltiplos catch-ups / load.
- Jitter 180s + accounting-daily no mesmo minuto → inconsistência de CSV.
- Overwrite same-day apaga evidência do primeiro run (sem run_id V1).
