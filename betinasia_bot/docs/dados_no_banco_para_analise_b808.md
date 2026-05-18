# Dados no banco para a análise (contexto b808)

Este documento resume **quais tabelas/colunas** do Postgres são usadas nas análises estatísticas do repositório (especialmente H3B) e como elas se relacionam.

## 1) Tabelas principais (H3B “API vs DOM”)

### 1.1 `matches`

Origem: `betinasia_bot/storage/models.py`

Campos relevantes para análise:
- `id` (PK)
- `external_id` (**chave** para merge com auditoria)
- `league`, `home_team`, `away_team`
- `kickoff_time`
- `home_score`, `away_score`, `status` (para ROI/retorno)

Uso:
- filtra “jogos com kickoff passado” (`kickoff_time < NOW()`)
- fornece placar para computar ROI (stake=1)

### 1.2 `best_odds_history`

Origem: `betinasia_bot/storage/models.py`

Campos relevantes:
- `match_id` (FK -> `matches.id`)
- `ah_line`
- `best_home_odds`, `best_away_odds`
- `scraped_at`

Uso:
- calcula a **closing line** como o último registro antes do kickoff para a mesma linha:
  - `ORDER BY scraped_at DESC LIMIT 1` com `scraped_at < kickoff_time`
- a `closing_odd` (lado) é usada para CLV:
  - \(CLV = (odd\_entrada - odd\_closing) / odd\_closing \times 100\)

### 1.3 `betslip_audit_results`

Origem: `betinasia_bot/storage/models_hypothesis.py`

Campos relevantes:
- identificação:
  - `hypothesis_type` (H1, H3, **H3B**, H6)
  - `event_id` (ID do evento no WebSocket)
  - `audit_version` (ex.: `v4.0-api`, `v1.0`, `v1.0-recovered`)
  - `reversal_direction` (H3B: `up`/`down`)
- mercado:
  - `market_type`, `market_period`, `line`, `side`
- odds:
  - `websocket_odd`
  - `betslip_odd`
  - `difference_pct` = \((betslip - websocket) / websocket \times 100\)
- qualidade:
  - `status`
  - `is_valid_opportunity`
- regime:
  - `is_live` (pre-match vs in-match)
- latência:
  - `lag_detection_to_click_ms`, `lag_click_to_betslip_ms`, `audit_total_duration_ms`
- limites:
  - `betslip_limit`

Uso:
- base “execução real” (betslip) para CLV/ROI e para medir erosão (`difference_pct`)
- base operacional para buckets por lag / regime / linha

## 2) Chaves de relacionamento usadas nas análises

O padrão de merge usado nos scripts de análise é:
- `betslip_audit_results.event_id = matches.external_id`
- `best_odds_history.match_id = matches.id`

Observação:
- `betslip_audit_results` armazena `line` como string (`"+1"`, `"-0.5"`).
- `best_odds_history.ah_line` pode estar como `"1.0"` etc.
- por isso, os scripts costumam normalizar e testar equivalências (`line`, `line||'.0'`, com/sem `+`).

## 3) Tabelas auxiliares (hipóteses)

Origem: `betinasia_bot/storage/models_hypothesis.py`

Existem tabelas de eventos para análises “por hipótese” (não necessariamente usando betslip):
- `h1_pricing_events`
- `h3_line_monotonicity_events`
- `h3b_temporal_reversal_events`
- `h6_correlation_lag_events`
- `odds_movement_history`

Elas têm:
- `clv_pct`, `bet_result`, `profit_loss`
- `detected_at`, `is_live`

E podem ser mergeadas com `betslip_audit_results` quando `hypothesis_event_id` é preenchido (ver `docs/audit_versions.md`).

## 4) Onde isso aparece no código

- Base abrangente H3B (CLV/ROI + buckets): `betinasia_bot/analyze_h3b_comprehensive.py`
- Comparação WebSocket vs Betslip: `betinasia_bot/analyze_h3b_websocket_vs_betslip.py`
- Análise robusta por hipótese (agrupando por jogo): `betinasia_bot/analyze_hypothesis_robust.py`
- Novo relatório robusto (cluster bootstrap, estilo “v4_somente”): `betinasia_bot/analyze_contexto_operacao_b808_robust_report.py`

