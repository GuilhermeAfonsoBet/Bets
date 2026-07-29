# Evidência de deploy — H3BUP Daily V2 Shadow — 20260729

## 1. Identidade

| Item | Valor |
|---|---|
| Data evidência | 2026-07-29 |
| Pacote | Phase 2R Daily redesign + audit docs |
| Modo | **SHADOW** |
| Oficial | **V1** (`python -m ops.daily_full_report`) |

## 2. Artefactos de código presentes no tree

```
ops/daily_v2/
  __init__.py          SCHEMA_VERSION=2; DAILY_FAST_LE_6S_MS; STUDY_FAST_LT_4S_MS
  __main__.py          CLI shadow flags
  extract.py
  time_windows.py
  universes.py
  performance.py
  statuses.py
  contracts.py
  canonical.py
  render.py
  io_atomic.py
  compare_v1.py
schemas/h3bup_daily_v2_schema.json
tests/test_h3bup_daily_v2.py
```

## 3. Flags de runtime (contrato shadow)

| Env | Valor deploy shadow | Efeito |
|---|---|---|
| `H3BUP_DAILY_V2_ENABLED` | `1` | Runner activo |
| `H3BUP_DAILY_V2_PUBLISH` | `0` | Sem `published/`; sem Telegram V2 |
| `H3BUP_DAILY_V2_COMPARE_V1` | `1` | CSV compare |
| `H3BUP_DAILY_V2_FAIL_OPEN` | `1` | Erros → exit 0 + last_error JSON |

## 4. Testes

Ficheiro: `logs/h3bup_daily_v2_tests_20260729.txt`

```
.......................                                                  [100%]
23 passed in 0.09s
```

## 5. Fix V1 P0 (observabilidade) — evidência

Em `ops/daily_full_report.py` markers activos com **`s0.append`**:

- `# BEGIN/END H3BUP_ACCOUNTING_HEALTH_SECTION`
- `# BEGIN/END H3BUP_E2E_LATENCY_SECTION`
- `# BEGIN/END H3BUP_CLV_FORWARD_SECTION`

Patch scripts alinhados (`ops/patch_daily_*_section.py`) geram `s0.append` (não `out_lines`).

## 6. O que **não** foi deployado como oficial

- [ ] Telegram V2
- [ ] Substituição do timer `betinasia-daily-full-report` por V2
- [ ] `PUBLISH=1` em produção como canal humano
- [ ] Remoção do monólito V1
- [ ] Quaisquer mudanças de policy / stake / executor thresholds de runtime como parte do cutover (reporting only)

## 7. Outputs esperados num run shadow

```
logs/daily_v2/h3bup_daily_snapshot_{YYYYMMDD}_{run_id}.json
logs/daily_v2/h3bup_daily_report_{YYYYMMDD}_{run_id}.md
logs/daily_v2/h3bup_daily_health_{YYYYMMDD}_{run_id}.json
logs/daily_v2/h3bup_daily_exceptions_{YYYYMMDD}_{run_id}.csv
logs/daily_v2/latest_snapshot.json          # symlink/pointer
logs/daily_v2/latest_report.md
logs/daily_v2/lkg/last_known_good_*         # se health ≠ FAILED
logs/h3bup_daily_v1_vs_v2_{YYYYMMDD}.csv
logs/h3bup_daily_metric_catalog_v2_{YYYYMMDD}.csv
logs/h3bup_daily_v2_performance_{YYYYMMDD}.json
```

## 8. Status machine-readable

```text
DAILY_AUDIT_TECHNICAL_GAPS=true
DAILY_V2_IMPLEMENTED_SHADOW=true
DAILY_REDESIGN_COMPLETE_SHADOW=true
DAILY_V2_PUBLISHED_OFFICIAL=false
PARITY_GATE=pending
```

## 9. Comando de verificação rápida

```bash
cd /workspace/betinasia_bot
python -m pytest tests/test_h3bup_daily_v2.py -q
python -m ops.daily_v2 --report-type DAILY_CLOSED --report-date 2026-07-28 \
  --v1-md logs/daily_reports/20260728/report_daily.md || true
# confirmar PUBLISH=0 → ausência de logs/daily_v2/published/ novos oficiais
```

## 10. Declaração de não-impacto

Deploy Phase 2R (código V2 shadow + docs/logs + fix append s0) **não** altera:

- decisões de policy;
- envio de ordens;
- sizing/stake runtime;
- workers CLV/E2E além do consumo read-only no relatório.
