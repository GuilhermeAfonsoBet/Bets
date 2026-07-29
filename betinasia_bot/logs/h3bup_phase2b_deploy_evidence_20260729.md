# H3BUP Fase 2B — Deploy evidence (2026-07-29)

## Before

Ver `logs/h3bup_phase2b_before_state_20260729.json`

- md5 audit `6d163e5e...`, bridge `3ccdc438...`, worker `298f8073...`
- accounting ACCOUNTING_OK / HEALTHY
- services active

## Steps

1. Backup em `/tmp/h3bup_phase2b_backup_20260729`
2. Install módulos + instrumentação
3. Patch Daily E2E section
4. Drop-ins systemd `*.service.d/h3bup-e2e-trace.conf` (OFF → ON)
5. Tests: workspace pytest 18 passed; VPS smoke SMOKE_OK
6. Restart sequencial:
   - `betinasia-audit-ws-gate-back`
   - `betinasia-executor-bridge-back`
   - `betinasia-executor`
7. Hotfix REQUEST_CREATED + restarts sequenciais
8. Soak natural (sem forçar ordens)

## After

Ver `logs/h3bup_phase2b_after_state_20260729.json`

- TRACE_ENABLED=1 nos 3 serviços
- JSONL a crescer (`logs/h3bup_e2e_trace.jsonl`)
- accounting HEALTHY
- rollback: `H3BUP_E2E_TRACE_ENABLED=0`

## Artefactos

1. `docs/h3bup_phase2b_e2e_design_20260729.md`
2. `docs/h3bup_phase2b_implementation_report_20260729.md`
3. `logs/h3bup_e2e_trace.jsonl`
4. `logs/h3bup_e2e_trace_health_20260729.json`
5. `logs/h3bup_e2e_latency_trace_level_20260729.csv`
6. `logs/h3bup_e2e_latency_summary_20260729.csv`
7. `logs/h3bup_e2e_latency_by_status_20260729.csv`
8. `logs/h3bup_e2e_latency_coverage_20260729.csv`
9. `logs/h3bup_e2e_ordering_violations_20260729.csv`
10. `logs/h3bup_phase2b_tests_20260729.txt`
11. `logs/h3bup_phase2b_before_state_20260729.json`
12. `logs/h3bup_phase2b_after_state_20260729.json`
13. este ficheiro
14. `logs/h3bup_phase2b_executive_summary_20260729.md`
