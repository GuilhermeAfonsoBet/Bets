# H3BUP Daily V2 — Health Model — 20260729

## 1. Camadas de health (distintas)

| Camada | Vocabulário | Significado |
|---|---|---|
| **SOURCE_HEALTH** | HEALTHY, WATCH, STALE, PARTIAL, FAILED, NOT_AVAILABLE | Estado de **uma fonte** vs `report_cutoff_utc` |
| **METRIC status** | AVAILABLE, MISSING, STALE, PARTIAL, UNAVAILABLE_STALE, INSUFFICIENT_N, NOT_IMPLEMENTED, FAILED, … | Estado de **uma métrica** |
| **REPORT_HEALTH** | HEALTHY, WATCH, PARTIAL, CRITICAL, FAILED | Agregado do **relatório** |
| **operations_health** | (WATCH default no snapshot) | Saúde de serviços runtime — Daily **não** reinicia nada |
| **statistical_readiness** | derivado de métricas | Prontidão para inferência (ex. CLV n≥30) |
| **maturity_status** | OPEN_COHORT, PARTIALLY_SETTLED, FULLY_SETTLED, CLV_*, FINALIZED | Maturidade da coorte |

**Regra de ouro:** `report_health` ≠ “estratégia está a ganhar dinheiro”.  
Um relatório HEALTHY pode ter ROI negativo AVAILABLE; um relatório CRITICAL não deve mostrar ROI como 0.

---

## 2. Freshness (extract)

Defaults (`extract.py`):

| Fonte | watch_after | stale_after |
|---|---|---|
| executor_live | 1h | 6h |
| accounting_health | 2h | 6h |
| balance / open_stakes | 12h | 36h |
| e2e / clv | 1h | 6h |
| policy_current / risk_params | 1d | 7d |

Se ficheiro health JSON traz `status` explícito ∈ {HEALTHY,WATCH,STALE,PARTIAL,FAILED}, prevalece.

---

## 3. Agregação `report_health`

Heurística actual em `canonical.py`:

- `executor_live` FAILED/NOT_AVAILABLE → CRITICAL;
- accounting STALE → PARTIAL/CRITICAL path;
- FAILED em fonte crítica → CRITICAL;
- WATCH/PARTIAL em fontes → WATCH se antes HEALTHY.

LKG: promove snapshot/md se `report_health` **≠ FAILED**.

---

## 4. Métrica ↔ health

| Condição fonte | Efeito métrica |
|---|---|
| accounting STALE/FAILED | `roi_settled` / roiw → `UNAVAILABLE_STALE`, value null |
| executor missing | `live_ok` FAILED/MISSING |
| e2e ausente | `e2e_ws_to_live_ok` MISSING |
| e2e n=0 | INSUFFICIENT_N |
| clv n_live_after_activation < 30 | INSUFFICIENT_N nos strict windows |
| fair edge | sempre NOT_IMPLEMENTED |

---

## 5. Exceptions / alertas

Lista `exceptions[]` no snapshot + CSV:

Campos: `alert_id`, `severity` (INFO|WATCH|WARNING|CRITICAL), `evidence`, `affected_metrics`, `status`.

Exemplos emitidos:

- `stake_mismatch:{oid}` — stake ≠ 10 (WARNING se 20 legado; CRITICAL outros);
- `policy_mix:{oid}` — policy_version sem H3BUP_vNext.

---

## 6. Contraste com V1

| V1 | V2 |
|---|---|
| Falha de secção → `pass` / some | status + exceptions |
| Sem report_health | obrigatório |
| Sem cutoff | `report_cutoff_utc` |
| Health H3BUP só se patch/fonte OK | manifesto sempre + envelopes |
| Sample 20260728 sem health/E2E/CLV | secções com MISSING/WATCH explícitos |

---

## 7. Publicação e health

Mesmo com `PUBLISH=1` futuro: **não** publicar como “oficial saudável” se `report_health=CRITICAL|FAILED` sem disclaimer. Shadow actual grava `health.publish=false`.

---

## 8. Operações vs relatório

| Pergunta | Campo |
|---|---|
| O PDF/snapshot é confiável as-of? | `report_health` |
| O accounting está fresco? | `source_manifest.accounting_health` |
| Posso afirmar edge? | `statistical_readiness` + INSUFFICIENT_N |
| O robô está UP? | `operations_health` (externo; Daily não é source of truth de uptime) |
