# H3BUP Fase 2C — Source Audit BestOddsHistory (2026-07-29)

## Respostas 1–24

1. Processo: `betinasia-collector.service` → `python -m collector.continuous_collector`
2. Alimentação contínua: **Sim** (serviço active)
3. Cadência observada (3h): ~1 amostra/min agregada; mediana ~521 rows/min; gaps >90s: 3
4. Períodos sem snapshots: gaps curtos ocasionais
5. Distingue evento: via `matches.external_id` ↔ `match_id` — **Sim**
6. Mercado: via `ah_line` prefix (`OU_`, `1X2`, AH raw) — **Parcial/Sim**
7. Período: **Não** (assume full-time)
8. Lado: colunas `best_home_odds` / `best_away_odds` — **Sim**
9. Linha: `ah_line` — **Sim** (string; requer normalização)
10. Bookmaker/source: **Não**
11. Timestamp: `scraped_at` = tempo de scrape/persistência UTC
12. Timezone: **UTC** (`timestamptz`)
13. Kickoff: `matches.kickoff_time` — **Sim**, confiável na tabela matches
14. Closing existente: último snapshot **antes** do kickoff (`get_closing_odd`)
15. Closing exige mesma linha: **Sim** (equality `ah_line`)
16. Fallback linha diferente no get_closing_odd actual: **Não**
17. Risco in-play: collector também grava in-match; closing filtra `scraped_at < kickoff`
18. Fórmula B808: `(entry - closing) / closing * 100`
19. Convenção: positivo = entry > closing
20. Sim — entrada melhor que snapshot para Back
21. Odd entrada H3BUP: preferir `sent.price`, senão `odd_final`
22. POST_5M/15M só com BOH: **Sim, se a linha existir** na janela temporal
23. Cobertura estimada: alta para linhas activamente scrapadas; risco de LINE_NOT_FOUND em linhas raras / mismatch string
24. Gaps: line-string mismatch e period FT-only → collector passivo (cópia BOH) ajuda densificar obligations activas; **não** requer feed externo

## Tabela componentes

| Componente | Existe | Qualidade | Reutilizável | Limitação | Acção |
|---|---|---|---|---|---|
| BestOddsHistory | Sim | Alta volume | Sim SOURCE1 | sem period/bookie; line string | Usar + normalizar |
| continuous_collector | Sim | Contínuo | Sim | in-play também | Manter |
| get_closing_odd | Sim | Boa | Sim lógica | lag gates | Reutilizar padrão |
| B808 CLV % | Sim | Confirmado | Sim fórmula | clip outliers em reports | Usar exact |
| H3BUP obligations | Não | — | — | — | Criar |
| Scheduler H3BUP | Não | — | — | — | Criar worker |
| Fair edge | Não | — | — | — | Não implementar |
