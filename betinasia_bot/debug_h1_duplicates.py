# -*- coding: utf-8 -*-
"""
Debug H1 - Verifica se está duplicando eventos.
"""

import asyncio
import sys
from sqlalchemy import text

sys.path.insert(0, '.')

from storage.database import Database


async def analyze_h1():
    """Analisa eventos H1 para verificar duplicatas."""
    
    db = Database()
    await db.connect()
    
    print("=" * 70)
    print("DEBUG H1 - Análise de Duplicatas")
    print("=" * 70)
    
    try:
        async with db.async_session() as session:
            
            # 1. Total e distribuição
            print("\n1. DISTRIBUIÇÃO POR TIPO")
            print("-" * 50)
            result = await session.execute(text("""
                SELECT 
                    market_type,
                    is_arb,
                    mispriced_side,
                    COUNT(*) as cnt
                FROM h1_pricing_events
                GROUP BY market_type, is_arb, mispriced_side
                ORDER BY cnt DESC
            """))
            for row in result.fetchall():
                arb_str = "ARB" if row[1] else "MISPRICING"
                print(f"   {row[0]:6} | {arb_str:10} | side={row[2]} | {row[3]:,} eventos")
            
            # 2. Verificar repetição por match/market/line
            print("\n2. ANÁLISE DE REPETIÇÃO")
            print("-" * 50)
            result = await session.execute(text("""
                WITH event_groups AS (
                    SELECT 
                        match_id, market_type, ah_line,
                        COUNT(*) as event_count,
                        COUNT(DISTINCT is_arb::text || COALESCE(mispriced_side, 'none')) as unique_states
                    FROM h1_pricing_events
                    GROUP BY match_id, market_type, ah_line
                )
                SELECT 
                    COUNT(*) as unique_markets,
                    SUM(event_count) as total_events,
                    ROUND(AVG(event_count)::numeric, 1) as avg_events_per_market,
                    MAX(event_count) as max_events
                FROM event_groups
            """))
            row = result.fetchone()
            print(f"""
   Mercados únicos (match+type+line): {row[0]:,}
   Total de eventos registrados: {row[1]:,}
   Média de eventos por mercado: {row[2]}
   Máximo de eventos em um mercado: {row[3]}
   
   NOTA: Se média >> 1, o mesmo mercado está gerando múltiplos eventos
""")
            
            # 3. Top mercados com mais eventos
            print("\n3. MERCADOS COM MAIS EVENTOS")
            print("-" * 50)
            result = await session.execute(text("""
                SELECT 
                    match_id, market_type, ah_line,
                    COUNT(*) as cnt,
                    ROUND(AVG(overround)::numeric, 4) as avg_overround,
                    ROUND(AVG(edge_estimate)::numeric, 4) as avg_edge
                FROM h1_pricing_events
                GROUP BY match_id, market_type, ah_line
                HAVING COUNT(*) > 5
                ORDER BY cnt DESC
                LIMIT 15
            """))
            rows = result.fetchall()
            if rows:
                print("   Match | Market | Line | Eventos | Overround | Edge")
                for row in rows:
                    print(f"   {row[0]:5} | {row[1]:6} | {row[2]:6} | {row[3]:7} | {row[4]:.4f} | {row[5]:.4f}")
            
            # 4. Verificar se odds mudam entre eventos
            print("\n4. VARIAÇÃO DE ODDS NO MESMO MERCADO")
            print("-" * 50)
            result = await session.execute(text("""
                SELECT 
                    match_id, market_type, ah_line,
                    COUNT(DISTINCT ROUND(odd_side_a::numeric, 2)::text || '-' || ROUND(odd_side_b::numeric, 2)::text) as distinct_odds,
                    COUNT(*) as total_events
                FROM h1_pricing_events
                GROUP BY match_id, market_type, ah_line
                HAVING COUNT(*) > 5
                ORDER BY total_events DESC
                LIMIT 10
            """))
            rows = result.fetchall()
            if rows:
                print("   Match | Market | Line | Odds distintas | Total eventos")
                for row in rows:
                    ratio = row[3] / row[4] * 100 if row[4] > 0 else 0
                    print(f"   {row[0]:5} | {row[1]:6} | {row[2]:6} | {row[3]:14} | {row[4]:5} ({ratio:.0f}% únicos)")
            
            print("""
INTERPRETAÇÃO:
- Se "odds distintas" << "total eventos", está duplicando para mesmas odds
- Se "odds distintas" ≈ "total eventos", cada evento tem odds diferentes (OK)
""")
            
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(analyze_h1())
