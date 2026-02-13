# -*- coding: utf-8 -*-
"""
Debug: Por que CLV não está sendo calculado para H1/H3/H6?
"""

import asyncio
import sys
from sqlalchemy import text

sys.path.insert(0, '.')

from storage.database import Database


async def debug_clv():
    db = Database()
    await db.connect()
    
    print("=" * 70)
    print("DEBUG: Por que CLV não está sendo calculado?")
    print("=" * 70)
    
    try:
        async with db.async_session() as session:
            
            # 1. Jogos finalizados
            print("\n1. JOGOS FINALIZADOS")
            print("-" * 50)
            result = await session.execute(text("""
                SELECT id, home_team, away_team, home_score, away_score, status
                FROM matches 
                WHERE status = 'finished' AND home_score IS NOT NULL
                ORDER BY id DESC
                LIMIT 10
            """))
            finished = result.fetchall()
            finished_ids = [r[0] for r in finished]
            print(f"   IDs dos últimos jogos finalizados: {finished_ids}")
            
            # 2. Eventos H1 - quais jogos?
            print("\n2. EVENTOS H1")
            print("-" * 50)
            result = await session.execute(text("""
                SELECT 
                    h.match_id,
                    m.home_team,
                    m.status,
                    m.home_score,
                    COUNT(*) as cnt,
                    COUNT(CASE WHEN h.clv IS NOT NULL THEN 1 END) as with_clv
                FROM h1_pricing_events h
                JOIN matches m ON h.match_id = m.id
                GROUP BY h.match_id, m.home_team, m.status, m.home_score
                ORDER BY cnt DESC
                LIMIT 10
            """))
            for row in result.fetchall():
                status = f"FINALIZADO ({row[3]})" if row[2] == 'finished' else row[2]
                print(f"   Match {row[0]}: {row[1]} | {status} | {row[4]} eventos | {row[5]} com CLV")
            
            # 3. Eventos H1 de jogos finalizados sem CLV
            print("\n3. EVENTOS H1 DE JOGOS FINALIZADOS SEM CLV")
            print("-" * 50)
            if finished_ids:
                result = await session.execute(text(f"""
                    SELECT 
                        h.id, h.match_id, h.market_type, h.ah_line,
                        h.recommended_side, h.recommended_odd
                    FROM h1_pricing_events h
                    WHERE h.match_id IN ({','.join(map(str, finished_ids))})
                      AND h.clv IS NULL
                    LIMIT 10
                """))
                rows = result.fetchall()
                if rows:
                    for row in rows:
                        print(f"   H1 id={row[0]}: match={row[1]}, {row[2]} {row[3]}, side={row[4]}, odd={row[5]}")
                else:
                    print("   Nenhum evento H1 para jogos finalizados (todos podem já ter CLV)")
            
            # 4. Verifica se há odds de fechamento
            print("\n4. ODDS DE FECHAMENTO PARA JOGOS FINALIZADOS")
            print("-" * 50)
            if finished_ids:
                result = await session.execute(text(f"""
                    SELECT 
                        match_id,
                        COUNT(DISTINCT ah_line) as linhas,
                        MAX(scraped_at) as ultima_coleta
                    FROM best_odds_history
                    WHERE match_id IN ({','.join(map(str, finished_ids))})
                    GROUP BY match_id
                    ORDER BY match_id DESC
                """))
                for row in result.fetchall():
                    print(f"   Match {row[0]}: {row[1]} linhas, última coleta {row[2]}")
            
            # 5. Verificar se eventos H3b têm bet_odd preenchido
            print("\n5. EVENTOS COM CAMPOS DE APOSTA PREENCHIDOS")
            print("-" * 50)
            result = await session.execute(text("""
                SELECT 
                    'H1' as tipo,
                    COUNT(*) as total,
                    COUNT(recommended_odd) as with_odd,
                    COUNT(recommended_side) as with_side
                FROM h1_pricing_events
                UNION ALL
                SELECT 
                    'H3',
                    COUNT(*),
                    COUNT(recommended_odd),
                    COUNT(recommended_line)
                FROM h3_line_monotonicity_events
                UNION ALL
                SELECT 
                    'H3b',
                    COUNT(*),
                    COUNT(bet_odd),
                    COUNT(bet_side)
                FROM h3b_temporal_reversal_events
                UNION ALL
                SELECT 
                    'H6',
                    COUNT(*),
                    COUNT(bet_odd),
                    COUNT(bet_side)
                FROM h6_correlation_lag_events
            """))
            print("   Tipo | Total | Com Odd | Com Side")
            for row in result.fetchall():
                print(f"   {row[0]:4} | {row[1]:5} | {row[2]:7} | {row[3]}")
            
            # 6. Match IDs nos eventos vs jogos finalizados
            print("\n6. EVENTOS POR STATUS DO JOGO")
            print("-" * 50)
            result = await session.execute(text("""
                SELECT 
                    'H1' as tipo,
                    m.status,
                    COUNT(*) as cnt
                FROM h1_pricing_events h
                JOIN matches m ON h.match_id = m.id
                GROUP BY m.status
                UNION ALL
                SELECT 
                    'H3',
                    m.status,
                    COUNT(*)
                FROM h3_line_monotonicity_events h
                JOIN matches m ON h.match_id = m.id
                GROUP BY m.status
                UNION ALL
                SELECT 
                    'H3b',
                    m.status,
                    COUNT(*)
                FROM h3b_temporal_reversal_events h
                JOIN matches m ON h.match_id = m.id
                GROUP BY m.status
                UNION ALL
                SELECT 
                    'H6',
                    m.status,
                    COUNT(*)
                FROM h6_correlation_lag_events h
                JOIN matches m ON h.match_id = m.id
                GROUP BY m.status
                ORDER BY 1, 2
            """))
            print("   Tipo | Status    | Eventos")
            for row in result.fetchall():
                print(f"   {row[0]:4} | {row[1]:9} | {row[2]}")
                
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(debug_clv())
