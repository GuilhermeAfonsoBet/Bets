#!/bin/bash
# Script para verificar estatísticas do banco de dados

echo "============================================================"
echo "ESTATÍSTICAS DE COLETA"
echo "============================================================"

psql -U betbot -d betinasia_bot -c "
SELECT 
  COUNT(*) as total_odds,
  COUNT(DISTINCT match_id) as total_matches,
  ROUND(AVG(num_bookmakers), 2) as avg_bookmakers,
  SUM(CASE WHEN num_bookmakers > 1 THEN 1 ELSE 0 END) as odds_com_bk,
  ROUND(100.0 * SUM(CASE WHEN num_bookmakers > 1 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as acuracia_pct
FROM odds_history;
"

echo ""
echo "POR LIGA:"
echo "============================================================"

psql -U betbot -d betinasia_bot -c "
SELECT 
  m.league,
  COUNT(DISTINCT m.id) as jogos,
  COUNT(o.id) as odds,
  ROUND(AVG(o.num_bookmakers), 1) as avg_bk,
  ROUND(100.0 * SUM(CASE WHEN o.num_bookmakers > 1 THEN 1 ELSE 0 END) / NULLIF(COUNT(o.id), 0), 1) as acuracia
FROM matches m
LEFT JOIN odds_history o ON o.match_id = m.id
GROUP BY m.league
ORDER BY jogos DESC;
"

echo ""
echo "ÚLTIMAS 5 PARTIDAS:"
echo "============================================================"

psql -U betbot -d betinasia_bot -c "
SELECT 
  home_team || ' vs ' || away_team as jogo,
  league,
  created_at::timestamp(0)
FROM matches
ORDER BY created_at DESC
LIMIT 5;
"
