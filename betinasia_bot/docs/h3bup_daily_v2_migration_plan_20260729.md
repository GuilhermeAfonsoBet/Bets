# H3BUP Daily V2 — Plano de migração — 20260729

**Estado actual:** shadow completo · V1 oficial · `PUBLISH=0`  
**Objectivo final:** V2 oficial com V1 hot-standby temporário · zero impacto em execução/policy/ordens.

---

## Fase A — Congelar contratos (DONE / Phase 2R)

- [x] Design 3 camadas
- [x] Schema JSON v2
- [x] Contratos ROI/ROIw/fast/CLV/fair_edge
- [x] Health model
- [x] Testes unitários (23 passed)
- [x] Fix P0 `s0.append` nos patches V1 (observabilidade legado)
- [x] Pacote audit docs/logs `*_20260729.*`

**Exit:** `DAILY_REDESIGN_COMPLETE_SHADOW`

---

## Fase B — Shadow soak (IN PROGRESS)

1. Correr `python -m ops.daily_v2` diariamente após (ou em paralelo fail-open ao) V1.
2. Flags: `ENABLED=1 PUBLISH=0 COMPARE=1 FAIL_OPEN=1`.
3. Arquivar:
   - snapshot/md/health/exceptions;
   - `logs/h3bup_daily_v1_vs_v2_{day}.csv`;
   - performance JSON.
4. Revisar divergências esperadas vs bugs:
   - esperadas: coorte DAILY_CLOSED D−1 vs pasta V1 generation day; filtro H3BUP; fast IDs; secções health;
   - bugs: joins oid, freshness false-positive, render drift.
5. Duração sugerida: ≥14 dias UTC com ≥1 dia de volume não trivial.

**Exit:** relatório de paridade assinado (`parity_pending` → `parity_accepted_with_deltas`).

---

## Fase C — Endurecer IO e ops

1. Garantir atomic+LKG em disco de produção.
2. Alertar se `report_health` ∈ {CRITICAL, FAILED} ou se V2 last_error.
3. Documentar runbook: “Telegram ainda é V1”.
4. Opcional: timer **separado** `betinasia-daily-h3bup-v2-shadow.timer` (não misturar DT).

**Exit:** evidência ops sem incidentes de IO.

---

## Fase D — Publish shadow interno (ainda não Telegram)

1. `H3BUP_DAILY_V2_PUBLISH=1` **só** para `logs/daily_v2/published/{day}/`.
2. Revisão humana md V2 vs PDF V1.
3. Manter Telegram = V1.

**Exit:** published/ populado N dias sem regressão.

---

## Fase E — Cutover publicação (GATE)

Pré-condições **todas** verdadeiras:

| Gate | Critério |
|---|---|
| G1 | Paridade aceite documentada |
| G2 | Testes CI/local verdes |
| G3 | Soak ≥14d sem CRITICAL silencioso |
| G4 | Atomic/LKG verificados |
| G5 | Caption Telegram com report_date/run_id/health |
| G6 | Rollback: V1 timer/PDF restaurável em <15 min |
| G7 | Confirmação explícita: sem mudança policy/exec |

Cutover:

1. Publicar PDF/md V2 no Telegram (novo renderer se necessário).
2. V1 continua a gravar pastas (standby) com `DAILY_REPORT_TELEGRAM=0`.
3. Monitor 7 dias.

**Exit:** V2 oficial; status deixa de ser SHADOW.

---

## Fase F — Desligar monólito (opcional, tardio)

- Remover dependência de patch scripts.
- Reduzir OOS/Telegram do monólito ou extrair OOS para job próprio.
- Arquivar V1 como `daily_full_report_legacy`.

---

## Rollback

```text
1. H3BUP_DAILY_V2_PUBLISH=0 (imediato)
2. DAILY_REPORT_TELEGRAM=1 no V1 service
3. Usar LKG V2 apenas para diagnóstico — não como oficial se health FAILED
4. NÃO tocar policy/executor durante rollback de reporting
```

---

## Riscos e mitigações

| Risco | Mitigação |
|---|---|
| Comparar D V1 generation vs D−1 V2 closed | Comparar com `--report-date` alinhado + coluna expected_delta |
| Contaminação DT | Paths e timers separados |
| Falso cutover por entusiasmo | Checklist G1–G7 obrigatório |
| Side-effect WF policy no V1 | Fora do cutover V2; tratar job WF separado a médio prazo |

---

## Respostas-chave de governação

- V2 **não publicado** agora.
- V1 **permanece oficial**.
- Perguntas 74–76 (execução/policy/ordens afectadas): **Não**.
