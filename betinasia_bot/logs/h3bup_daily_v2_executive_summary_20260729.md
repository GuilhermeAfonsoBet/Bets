# H3BUP Daily Phase 2R — Sumário executivo — 20260729

## Veredicto

O Daily oficial continua a ser o **V1** (`python -m ops.daily_full_report` via timer 22:00 UTC / 19:00 BRT → PDF Telegram). A auditoria confirma **gaps técnicos materiais** (tempo misturado, três “fast”, ROIw sem filtro H3BUP, envelope sem health/run_id, IO não atómico, `except: pass`). O **V2** está **implementado em shadow** (3 camadas, contratos, health, atomic/LKG, testes 23/23) com `PUBLISH=0` — **não** substitui V1. Bug P0 `out_lines`→`s0` nas secções H3BUP está **corrigido**. **Nenhuma** alteração de execução/policy/ordens (Q74–76 = Não).

Status: `DAILY_AUDIT_TECHNICAL_GAPS` · `DAILY_V2_IMPLEMENTED_SHADOW` · `DAILY_REDESIGN_COMPLETE_SHADOW`.

---

## Pacote de artefactos (este stamp)

| # | Ficheiro |
|---|---|
| 1 | `docs/h3bup_daily_current_architecture_20260729.md` |
| 2 | `docs/h3bup_daily_current_execution_flow_20260729.md` |
| 3 | `logs/h3bup_daily_source_inventory_20260729.csv` |
| 4 | `logs/h3bup_daily_data_lineage_20260729.csv` |
| 5 | `logs/h3bup_daily_metric_catalog_current_20260729.csv` |
| 6 | `logs/h3bup_daily_current_issues_20260729.csv` |
| 7 | `logs/h3bup_daily_timer_and_publication_audit_20260729.md` |
| 8 | `logs/h3bup_daily_roi_and_roiw_audit_20260729.md` |
| 9 | `logs/h3bup_daily_time_semantics_audit_20260729.md` |
| 10 | `logs/h3bup_daily_v1_sample_20260729.md` |
| 11 | `docs/h3bup_daily_v2_design_20260729.md` |
| 12 | `docs/h3bup_daily_v2_metric_contracts_20260729.md` |
| 13 | `docs/h3bup_daily_v2_health_model_20260729.md` |
| 14 | `docs/h3bup_daily_v2_migration_plan_20260729.md` |
| 15 | `logs/h3bup_daily_v2_deploy_evidence_20260729.md` |
| 16 | `logs/h3bup_daily_v2_executive_summary_20260729.md` (este) |

---

## Respostas de verificação 1–76

### A. Entrypoint, timer, publicação (1–12)

1. Qual o entrypoint oficial do Daily? → `python -m ops.daily_full_report`.
2. Qual o unit systemd service? → `betinasia-daily-full-report.service`.
3. Qual o timer? → `betinasia-daily-full-report.timer`.
4. OnCalendar? → `*-*-* 22:00:00 UTC`.
5. Equivalente BRT? → 19:00 America/Sao_Paulo.
6. Persistent? → Sim (`true`).
7. RandomizedDelaySec? → 180.
8. Canal de publicação oficial? → Telegram PDF (`DAILY_REPORT_TELEGRAM`).
9. Path md/pdf? → `logs/daily_reports/{YYYYMMDD}/report_daily.md|.pdf`.
10. Escrita atómica md/pdf no V1? → **Não**.
11. Existe latest symlink no V1? → **Não**.
12. Existe last-known-good no V1? → **Não**.

### B. Timers adjacentes e perímetro (13–18)

13. Há accounting-daily timer ~22:00 UTC? → Sim (`betinasia-accounting-daily.timer`).
14. Há daily DT report separado? → Sim (`betinasia-daily-dt-report`).
15. Pode misturar DT com H3BUP Daily? → **Não**.
16. Histórico V1 como é guardado? → Pastas por dia UTC de geração.
17. Overwrite same-day? → Sim.
18. V2 publica Telegram agora? → **Não** (`PUBLISH=0`).

### C. Semântica temporal (19–28)

19. Dia da pasta V1 = ? → Data UTC do instante de geração.
20. É estritamente “ontem fechado”? → **Não**.
21. KPIs LIVE_OK principais no resumo usam? → Rolling 24h (entre outras janelas).
22. Aderência típica? → 7d UTC.
23. `pnl_today` em que TZ? → `REPORT_TZ` default `America/Sao_Paulo`.
24. Coorte ROIw Total? → `created_at` UTC + `order_id`.
25. Coorte P&L acct séries? → Post date UTC.
26. V1 tem `report_cutoff_utc`? → **Não**.
27. V2 DAILY_CLOSED window? → Half-open UTC `[D, D+1)` do dia fechado.
28. Post date pode ser cohort key V2? → **Não**.

### D. ROI / ROIw (29–40)

29. Fórmula ROIw Total? → `(∑pnl / ∑exposure)*100`.
30. w significa? → Exposure-weighted.
31. Open pode entrar no ROIw Total? → Sim, se no ledger.
32. Void-like pnl≈0 no ROIw Total? → Incluído.
33. Missing order_id? → Excluído de num/den.
34. ROI settled inclui open? → **Não**.
35. ROI settled unidade no health? → Fraction.
36. Métrica principal V2 performance? → `roi_settled`.
37. ROIw V1 é complementar? → Sim.
38. Filtro H3BUP na tabela ROIw V1 principal? → **Não** (gap).
39. V2 filtra H3BUP por default? → Sim (`require_h3bup`).
40. Accounting STALE no V2? → `UNAVAILABLE_STALE`, value null.

### E. Fast / latência (41–50)

41. Fast pré-tese limiar pre_submit? → ≤6000 ms.
42. Fast pós-tese (env executor)? → ≤5000 ms (`EXECUTOR_BACKPRE_FAST_MAX_PRE_SUBMIT_MS`).
43. Tese start day típica? → ~2026-04-20 (configurável).
44. Contrafactual lat≤6s usa? → `call_to_done_ms≤6000`.
45. Há três conceitos fast misturados no V1? → **Sim**.
46. Contrato Daily utilizador? → `DAILY_FAST_LE_6S` (`pre_submit≤6000`).
47. Contrato study? → `STUDY_FAST_LT_4S` (`pre_submit<4000`).
48. Study pode substituir Daily fast? → **Não**.
49. Missing pre_submit vira slow no V2? → **Não** (bucket NA).
50. E2E vive onde? → Trace jsonl + secção patch / V2 latency.

### F. Envelope, patches, robustez (51–60)

51. V1 tem schema_version? → **Não**.
52. V1 tem run_id? → **Não**.
53. V1 tem report_health? → **Não**.
54. Secções health/E2E/CLV como entraram? → Patch scripts string.
55. Bug P0 variável errada? → `out_lines` vs `s0`.
56. P0 status Phase 2R? → **FIXED** (`s0.append`).
57. `except: pass` massivo? → **Sim** (gap aberto).
58. Fair edge implementado? → **Não**.
59. V2 declara fair_edge como? → `NOT_IMPLEMENTED`.
60. Sample 20260728 LIVE_OK 24h? → **2**.

### G. Sample 20260728 e fontes (61–68)

61. Sample mostrou CAP_BLOCKED? → Sim (capacity).
62. Sample tinha secções ROIw? → Sim.
63. Sample tinha Accounting Health H3BUP? → **Não** (ausente).
64. Sample tinha E2E H3BUP? → **Não**.
65. Sample tinha CLV H3BUP? → **Não**.
66. Fontes principais V1? → executor jsonl, audit DB, balance/open CSV, health JSON, e2e, clv, wf_policy, OOS/B808.
67. V2 camadas? → extract / canonical / render.
68. Schema V2 version const? → 2.

### H. Shadow, gates, governação (69–76)

69. Flags shadow típicas? → `ENABLED=1 PUBLISH=0 COMPARE=1 FAIL_OPEN=1`.
70. V2 está publicado oficialmente? → **Não**.
71. Gate principal pendente? → Paridade V1↔V2.
72. Quem permanece oficial? → V1.
73. Testes V2 no stamp? → 23 passed.
74. Phase 2R afectou **execução** live do robô? → **Não**.
75. Phase 2R afectou **policy** / wf publish como objectivo do redesign Daily V2 shadow? → **Não** (reporting-only; sem mudança intencional de policy de trading H3BUP).
76. Phase 2R afectou **ordens** (placement/sizing runtime)? → **Não**.

---

## Próximo passo recomendado

Abrir Fase B (soak shadow + compare diário) sem Telegram V2; só avançar PUBLISH após gates G1–G7 do plano de migração.
