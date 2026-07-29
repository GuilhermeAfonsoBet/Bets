# Sample V1 — resumo auditado (base 20260728) — docstamp 20260729

**Fonte:** `logs/daily_reports/20260728/report_daily.md` (e PDF associado), conforme auditoria Phase 2R.  
**Nota:** este ficheiro é um **resumo de evidência**; não republica o PDF completo.

---

## 1. Identidade do sample

| Campo | Valor |
|---|---|
| Pasta dia | `20260728` |
| Dia do relatório (UTC) | `20260728` (geração UTC desse dia) |
| Produto | Daily V1 BetinAsia / monólito `daily_full_report` |
| Publicação | Telegram PDF (canal oficial) |

---

## 2. Achados operacionais destacados

| Sinal | Observação no sample |
|---|---|
| **LIVE_OK 24h** | **2** |
| **CAP_BLOCKED** | Presente no funil — leitura de **capacity** (bloqueio por capacidade), não falha de API genérica |
| Throughput | Baixo volume LIVE_OK na janela 24h |
| Interpretação | Dia operacionalmente magro em fills; capacidade relevante no funil accepted→LIVE_OK |

---

## 3. Secções ROIw / performance

- Blocos / tabelas **ROIw** **existem** no sample (incluindo ROIw Total por dia exec e/ou derivados por bucket).
- Dualidade accounting post-date vs exec-date permanece no template V1 (estrutura do monólito).
- Não há envelope `roi_settled` canónico separado no topo do relatório V1 (ROI settled vive na secção health quando injectada).

---

## 4. Secções H3BUP injectadas — ausentes neste sample

No sample **20260728** auditado:

| Secção | Presente? |
|---|---|
| Accounting Health — H3BUP | **Não** observada |
| H3BUP End-to-End Latency | **Não** observada |
| H3BUP CLV Forward Collection | **Não** observada |

Causas possíveis (não mutuamente exclusivas), alinhadas ao audit:

1. Patches ainda não deployados / markers ausentes na build que gerou 20260728;
2. Bug P0 `out_lines` vs `s0` a impedir montagem (corrigido depois em Phase 2R);
3. `except` a engolir falha de import/fonte sem deixar vestígio claro;
4. Fontes health/e2e/clv em falta no host naquele run.

**Implicação:** o PDF oficial daquele dia **não** documenta health accounting H3BUP, E2E nem CLV — gap de observabilidade.

---

## 5. Envelope / metadados

Ausentes no V1 (confirmado pelo desenho do monólito; sample coerente):

- `schema_version`
- `run_id`
- `report_health`
- `report_cutoff_utc`

Cabeçalho típico: dia UTC + `Gerado em (UTC)`.

---

## 6. Fair edge

Não aparece como métrica implementada (coerente com “not implemented”). Risco residual: leitor interpretar ausência como “sem edge” em vez de “não medido”.

---

## 7. Ligação ao status Phase 2R

| Status | Relação ao sample |
|---|---|
| `DAILY_AUDIT_TECHNICAL_GAPS` | Sample ilustra gaps (secções H3BUP ausentes, envelope fraco, volume baixo + CAP) |
| `DAILY_V2_IMPLEMENTED_SHADOW` | V2 passa a emitir health/latency/CLV/`NOT_IMPLEMENTED` mesmo quando fontes falham |
| `DAILY_REDESIGN_COMPLETE_SHADOW` | Redesign documentado; sample V1 permanece referência de paridade |

---

## 8. Uso deste sample em compare V1↔V2

Para `H3BUP_DAILY_V2_COMPARE_V1=1`:

```bash
python -m ops.daily_v2 --report-date 2026-07-28 \
  --v1-md logs/daily_reports/20260728/report_daily.md
```

Esperado: divergências em policy filter, fast buckets, presença de secções health, e coorte `DAILY_CLOSED` (D fechado) vs pasta V1 de geração.
