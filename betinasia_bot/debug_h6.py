# -*- coding: utf-8 -*-
"""
Debug do detector H6 - Verifica por que não está detectando eventos.
"""

import asyncio
import sys
from datetime import datetime, timezone, timedelta
from sqlalchemy import text

sys.path.insert(0, '.')

from storage.database import Database


async def analyze_h6_conditions():
    """Analisa as condições necessárias para H6 detectar eventos."""
    
    db = Database()
    await db.connect()
    
    print("=" * 70)
    print("DEBUG H6 - Análise de Condições")
    print("=" * 70)
    
    try:
        async with db.async_session() as session:
            
            # 1. Verifica variação de odds entre coletas consecutivas
            print("\n1. VARIAÇÃO DE ODDS ENTRE COLETAS")
            print("-" * 50)
            
            result = await session.execute(text("""
                WITH consecutive_odds AS (
                    SELECT 
                        match_id,
                        ah_line,
                        best_home_odds,
                        best_away_odds,
                        scraped_at,
                        LAG(best_home_odds) OVER (PARTITION BY match_id, ah_line ORDER BY scraped_at) as prev_home,
                        LAG(best_away_odds) OVER (PARTITION BY match_id, ah_line ORDER BY scraped_at) as prev_away,
                        LAG(scraped_at) OVER (PARTITION BY match_id, ah_line ORDER BY scraped_at) as prev_time
                    FROM best_odds_history
                    WHERE scraped_at >= NOW() - INTERVAL '1 hour'
                )
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(CASE WHEN prev_home IS NOT NULL AND best_home_odds != prev_home THEN 1 END) as home_changes,
                    COUNT(CASE WHEN prev_away IS NOT NULL AND best_away_odds != prev_away THEN 1 END) as away_changes,
                    COUNT(CASE WHEN prev_home IS NOT NULL AND ABS(best_home_odds - prev_home) / prev_home >= 0.005 THEN 1 END) as significant_home_changes,
                    COUNT(CASE WHEN prev_away IS NOT NULL AND ABS(best_away_odds - prev_away) / prev_away >= 0.005 THEN 1 END) as significant_away_changes
                FROM consecutive_odds
            """))
            row = result.fetchone()
            if row:
                print(f"   Total registros (1h): {row[0]:,}")
                print(f"   Mudanças home: {row[1]:,}")
                print(f"   Mudanças away: {row[2]:,}")
                print(f"   Mudanças significativas home (>=0.5%): {row[3]:,}")
                print(f"   Mudanças significativas away (>=0.5%): {row[4]:,}")
            
            # 2. Verifica formato das linhas
            print("\n2. FORMATO DAS LINHAS DE AH")
            print("-" * 50)
            
            result = await session.execute(text("""
                SELECT DISTINCT ah_line 
                FROM best_odds_history 
                WHERE ah_line NOT LIKE 'OU_%' AND ah_line != '1X2' AND ah_line != '1X2_DRAW'
                ORDER BY ah_line
                LIMIT 20
            """))
            lines = [row[0] for row in result.fetchall()]
            print(f"   Linhas de AH: {lines[:10]}...")
            
            # 3. Verifica se há linhas adjacentes
            print("\n3. LINHAS ADJACENTES DISPONÍVEIS")
            print("-" * 50)
            
            result = await session.execute(text("""
                SELECT match_id, array_agg(DISTINCT ah_line ORDER BY ah_line) as lines
                FROM best_odds_history
                WHERE ah_line NOT LIKE 'OU_%' 
                  AND ah_line != '1X2' 
                  AND ah_line != '1X2_DRAW'
                  AND scraped_at >= NOW() - INTERVAL '1 hour'
                GROUP BY match_id
                HAVING COUNT(DISTINCT ah_line) >= 3
                LIMIT 5
            """))
            for row in result.fetchall():
                print(f"   Match {row[0]}: {row[1][:8]}...")
            
            # 4. Verifica exemplo de movimento
            print("\n4. EXEMPLOS DE MOVIMENTOS RECENTES")
            print("-" * 50)
            
            result = await session.execute(text("""
                WITH movements AS (
                    SELECT 
                        match_id,
                        ah_line,
                        best_home_odds as current_odd,
                        LAG(best_home_odds) OVER (PARTITION BY match_id, ah_line ORDER BY scraped_at) as prev_odd,
                        scraped_at,
                        LAG(scraped_at) OVER (PARTITION BY match_id, ah_line ORDER BY scraped_at) as prev_time
                    FROM best_odds_history
                    WHERE ah_line NOT LIKE 'OU_%' 
                      AND ah_line != '1X2'
                      AND scraped_at >= NOW() - INTERVAL '1 hour'
                )
                SELECT 
                    match_id, 
                    ah_line, 
                    prev_odd, 
                    current_odd,
                    ROUND(ABS(current_odd - prev_odd) / prev_odd * 100, 3) as pct_change,
                    scraped_at,
                    scraped_at - prev_time as time_diff
                FROM movements
                WHERE prev_odd IS NOT NULL 
                  AND current_odd != prev_odd
                ORDER BY ABS(current_odd - prev_odd) DESC
                LIMIT 10
            """))
            print("   Match | Line | Prev -> Curr | %Change | Time Diff")
            for row in result.fetchall():
                print(f"   {row[0]:5} | {row[1]:6} | {row[2]:.2f} -> {row[3]:.2f} | {row[4]:.2f}% | {row[6]}")
            
            # 5. Verifica intervalo entre coletas
            print("\n5. INTERVALO ENTRE COLETAS")
            print("-" * 50)
            
            result = await session.execute(text("""
                WITH times AS (
                    SELECT 
                        scraped_at,
                        LAG(scraped_at) OVER (ORDER BY scraped_at) as prev_time
                    FROM (
                        SELECT DISTINCT scraped_at 
                        FROM best_odds_history 
                        WHERE scraped_at >= NOW() - INTERVAL '30 minutes'
                        ORDER BY scraped_at
                    ) t
                )
                SELECT 
                    AVG(EXTRACT(EPOCH FROM (scraped_at - prev_time))) as avg_interval,
                    MIN(EXTRACT(EPOCH FROM (scraped_at - prev_time))) as min_interval,
                    MAX(EXTRACT(EPOCH FROM (scraped_at - prev_time))) as max_interval
                FROM times
                WHERE prev_time IS NOT NULL
            """))
            row = result.fetchone()
            if row and row[0]:
                print(f"   Intervalo médio: {row[0]:.1f}s")
                print(f"   Intervalo mínimo: {row[1]:.1f}s")
                print(f"   Intervalo máximo: {row[2]:.1f}s")
            
            # 6. Diagnóstico final
            print("\n" + "=" * 70)
            print("DIAGNÓSTICO")
            print("=" * 70)
            
            if row and row[0]:
                print(f"""
O H6 detecta lag quando:
1. Um mercado (ex: AH -0.5) tem movimento >= 0.5%
2. Um mercado correlacionado (ex: AH -0.75) NÃO moveu nos últimos 120s

Com intervalo de coleta de ~{row[0]:.0f}s:
- Todos os mercados são atualizados no mesmo ciclo
- Não há "lag real" entre mercados porque todos são coletados juntos
- O conceito de H6 faz mais sentido com streaming em tempo real

Sugestões:
1. Reduzir threshold de movimento (atualmente 0.5%)
2. Mudar lógica para comparar se movimento ESPERADO aconteceu
3. Aceitar que H6 pode não ser aplicável com coleta batch
""")
            
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(analyze_h6_conditions())
