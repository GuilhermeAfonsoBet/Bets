# Ajustes no robo: Lay temporal + telemetria ponta a ponta

Este pacote implementa os 2 pontos pedidos:

1. **Lay temporal equivalente ao Back** na auditoria API (`audit_h3b_api.py`)
2. **Telemetria completa ponta a ponta e por etapa** no API audit e no collector continuo

## 1) O que foi alterado

### API audit (`betinasia_bot/audit_h3b_api.py`)

- Mantida captura simultanea **T+0** de Back e Lay.
- Adicionado **monitoramento temporal para Lay** nos mesmos checkpoints do Back:
  - `t+3, t+6, t+10, t+15, t+20`
- Persistencia no banco via `hypothesis_details`:
  - `hypothesis_details.temporal` (Back temporal)
  - `hypothesis_details.lay_temporal` (Lay temporal)
  - `hypothesis_details.lay` (snapshot T+0 do lay)
  - `hypothesis_details.telemetry` (tempos detalhados)

Telemetria gravada por auditoria:

- `queue_wait_ms`
- `build_bet_type_ms`
- `parallel_fetch_ms`
- `back_post_ms`, `back_pmm_ms`, `back_total_ms`
- `lay_post_ms`, `lay_pmm_ms`, `lay_total_ms`
- `temporal_total_ms`, `temporal_wait_ms`, `temporal_refresh_mean_ms`
- `temporal_points_back`, `temporal_points_lay`, `temporal_points` (lista por checkpoint)
- `execution_ms`, `end_to_end_ms`, `pipeline_overhead_ms`
- `db_save_ms`, `pipeline_total_ms`, `executor_total_ms`

Tambem foi adicionado JSONL operacional:

- `logs/audit_api_telemetry.jsonl`

### Collector continuo (`betinasia_bot/collector/continuous_collector.py`)

Telemetria por ciclo com escrita em:

- `logs/collector_telemetry.jsonl`

Campos principais por ciclo:

- `cycle_total_ms`, `collect_ms`, `save_ms`, `collect_reported_ms`
- `events_discovered`, `events_with_odds`, `matches_payload`, `matches_saved`
- `prematch_saved`, `live_saved`
- `hypothesis_events_saved`
- `save_errors`

## 2) Como validar rapidamente em producao

### 2.1 Reiniciar servicos

```bash
sudo systemctl restart betinasia-collector
sudo systemctl restart betinasia-audit-api
```

### 2.2 Confirmar escrita de telemetria

```bash
tail -n 20 logs/audit_api_telemetry.jsonl
tail -n 20 logs/collector_telemetry.jsonl
```

### 2.3 Confirmar Lay temporal no banco

```sql
SELECT
  COUNT(*) AS total,
  COUNT(*) FILTER (WHERE (hypothesis_details::jsonb -> 'lay') IS NOT NULL) AS com_lay_t0,
  COUNT(*) FILTER (WHERE (hypothesis_details::jsonb -> 'lay_temporal') IS NOT NULL) AS com_lay_temporal,
  COUNT(*) FILTER (WHERE (hypothesis_details::jsonb -> 'temporal') IS NOT NULL) AS com_back_temporal
FROM betslip_audit_results
WHERE audit_version = 'v4.0-api'
  AND audited_at >= now() - interval '24 hours';
```

### 2.4 Quebra de tempos (ponta a ponta e etapas)

```sql
SELECT
  AVG((hypothesis_details::jsonb -> 'telemetry' ->> 'queue_wait_ms')::float)   AS queue_wait_ms,
  AVG((hypothesis_details::jsonb -> 'telemetry' ->> 'back_post_ms')::float)    AS back_post_ms,
  AVG((hypothesis_details::jsonb -> 'telemetry' ->> 'back_pmm_ms')::float)     AS back_pmm_ms,
  AVG((hypothesis_details::jsonb -> 'telemetry' ->> 'lay_post_ms')::float)     AS lay_post_ms,
  AVG((hypothesis_details::jsonb -> 'telemetry' ->> 'lay_pmm_ms')::float)      AS lay_pmm_ms,
  AVG((hypothesis_details::jsonb -> 'telemetry' ->> 'temporal_total_ms')::float) AS temporal_total_ms,
  AVG((hypothesis_details::jsonb -> 'telemetry' ->> 'db_save_ms')::float)      AS db_save_ms,
  AVG((hypothesis_details::jsonb -> 'telemetry' ->> 'pipeline_total_ms')::float) AS pipeline_total_ms
FROM betslip_audit_results
WHERE audit_version = 'v4.0-api'
  AND audited_at >= now() - interval '24 hours'
  AND (hypothesis_details::jsonb -> 'telemetry') IS NOT NULL;
```

