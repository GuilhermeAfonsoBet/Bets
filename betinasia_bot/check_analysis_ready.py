# -*- coding: utf-8 -*-
"""
Verifica quantos jogos temos disponíveis para análise estatística das hipóteses.
"""

import asyncio
import sys
from sqlalchemy import text

sys.path.insert(0, '.')

from storage.database import Database


async def check_analysis_ready():
    db = Database()
    await db.connect()
    
    print("=" * 70)
    print("JOGOS DISPONÍVEIS PARA ANÁLISE DE HIPÓTESES")
    print("=" * 70)
    
    try:
        async with db.async_session() as session:
            
            # 1. Jogos finalizados com odds pré-match
            print("\n" + "=" * 70)
            print("1. JOGOS PRONTOS PARA ANÁLISE (finalizados + odds pré-match)")
            print("=" * 70)
            
            result = await session.execute(text("""
                WITH analysis_ready AS (
                    SELECT 
                        m.id,
                        m.home_team,
                        m.away_team,
                        m.league,
                        m.home_score,
                        m.away_score,
                        m.kickoff_time,
                        COUNT(DISTINCT CASE WHEN b.scraped_at < m.kickoff_time THEN b.ah_line END) as linhas_pre_match,
                        COUNT(DISTINCT b.ah_line) as linhas_total
                    FROM matches m
                    LEFT JOIN best_odds_history b ON m.id = b.match_id
                    WHERE m.status = 'finished' 
                      AND m.home_score IS NOT NULL
                    GROUP BY m.id, m.home_team, m.away_team, m.league, m.home_score, m.away_score, m.kickoff_time
                )
                SELECT 
                    CASE 
                        WHEN linhas_pre_match > 0 THEN 'COM odds pré-match'
                        ELSE 'SEM odds pré-match'
                    END as status,
                    COUNT(*) as jogos,
                    SUM(linhas_pre_match) as total_linhas_pre
                FROM analysis_ready
                GROUP BY 1
                ORDER BY 1
            """))
            print("\n   Status | Jogos | Linhas Pré-Match")
            for row in result.fetchall():
                print(f"   {row[0]:25} | {row[1]:5} | {row[2] or 0}")
            
            # 2. Lista de jogos prontos para análise
            print("\n" + "=" * 70)
            print("2. JOGOS PRONTOS PARA ANÁLISE (com odds pré-match)")
            print("=" * 70)
            
            result = await session.execute(text("""
                SELECT 
                    m.id,
                    m.home_team,
                    m.away_team,
                    m.league,
                    m.home_score || '-' || m.away_score as score,
                    m.kickoff_time,
                    COUNT(DISTINCT CASE WHEN b.scraped_at < m.kickoff_time THEN b.ah_line END) as linhas_pre
                FROM matches m
                LEFT JOIN best_odds_history b ON m.id = b.match_id
                WHERE m.status = 'finished' 
                  AND m.home_score IS NOT NULL
                GROUP BY m.id, m.home_team, m.away_team, m.league, m.home_score, m.away_score, m.kickoff_time
                HAVING COUNT(DISTINCT CASE WHEN b.scraped_at < m.kickoff_time THEN b.ah_line END) > 0
                ORDER BY m.kickoff_time DESC
            """))
            rows = result.fetchall()
            if rows:
                print(f"\n   Total: {len(rows)} jogos")
                print("\n   ID | Score | Linhas | Jogo")
                for row in rows[:20]:  # Mostra até 20
                    league_short = (row[3] or "")[:15]
                    print(f"   {row[0]:3} | {row[4]:5} | {row[6]:6} | {row[1][:15]} vs {row[2][:15]}")
                if len(rows) > 20:
                    print(f"   ... e mais {len(rows) - 20} jogos")
            else:
                print("\n   ⚠️ Nenhum jogo pronto para análise ainda!")
            
            # 3. Eventos de hipóteses desses jogos
            print("\n" + "=" * 70)
            print("3. EVENTOS DE HIPÓTESES EM JOGOS PRONTOS")
            print("=" * 70)
            
            # IDs dos jogos prontos
            result = await session.execute(text("""
                SELECT m.id
                FROM matches m
                LEFT JOIN best_odds_history b ON m.id = b.match_id
                WHERE m.status = 'finished' 
                  AND m.home_score IS NOT NULL
                GROUP BY m.id
                HAVING COUNT(DISTINCT CASE WHEN b.scraped_at < m.kickoff_time THEN b.ah_line END) > 0
            """))
            ready_ids = [row[0] for row in result.fetchall()]
            
            if ready_ids:
                ids_str = ','.join(map(str, ready_ids))
                
                result = await session.execute(text(f"""
                    SELECT 
                        'H1' as tipo,
                        COUNT(*) as eventos,
                        COUNT(CASE WHEN clv IS NOT NULL THEN 1 END) as com_clv
                    FROM h1_pricing_events WHERE match_id IN ({ids_str})
                    UNION ALL
                    SELECT 'H3', COUNT(*), COUNT(CASE WHEN clv IS NOT NULL THEN 1 END)
                    FROM h3_line_monotonicity_events WHERE match_id IN ({ids_str})
                    UNION ALL
                    SELECT 'H3b', COUNT(*), COUNT(CASE WHEN clv IS NOT NULL THEN 1 END)
                    FROM h3b_temporal_reversal_events WHERE match_id IN ({ids_str})
                    UNION ALL
                    SELECT 'H6', COUNT(*), COUNT(CASE WHEN clv IS NOT NULL THEN 1 END)
                    FROM h6_correlation_lag_events WHERE match_id IN ({ids_str})
                """))
                print("\n   Tipo | Eventos | Com CLV")
                total_eventos = 0
                total_clv = 0
                for row in result.fetchall():
                    print(f"   {row[0]:4} | {row[1]:7} | {row[2]}")
                    total_eventos += row[1]
                    total_clv += row[2]
                print(f"   {'TOTAL':4} | {total_eventos:7} | {total_clv}")
            
            # 4. Por liga
            print("\n" + "=" * 70)
            print("4. JOGOS PRONTOS POR LIGA")
            print("=" * 70)
            
            result = await session.execute(text("""
                SELECT 
                    m.league,
                    COUNT(*) as jogos,
                    SUM(CASE WHEN b_count.pre > 0 THEN 1 ELSE 0 END) as com_pre_match
                FROM matches m
                LEFT JOIN (
                    SELECT 
                        match_id,
                        COUNT(DISTINCT CASE WHEN b.scraped_at < m.kickoff_time THEN b.ah_line END) as pre
                    FROM best_odds_history b
                    JOIN matches m ON b.match_id = m.id
                    GROUP BY match_id
                ) b_count ON m.id = b_count.match_id
                WHERE m.status = 'finished'
                GROUP BY m.league
                HAVING COUNT(*) > 0
                ORDER BY COUNT(*) DESC
                LIMIT 15
            """))
            print("\n   Liga | Jogos | Com Pré-Match")
            for row in result.fetchall():
                league = (row[0] or "Unknown")[:35]
                print(f"   {league:35} | {row[1]:5} | {row[2] or 0}")
            
            # Resumo
            print("\n" + "=" * 70)
            print("RESUMO")
            print("=" * 70)
            
            print(f"""
   Jogos prontos para análise (finalizados + odds pré-match): {len(rows) if rows else 0}
   
   PRÓXIMOS PASSOS:
   1. Rodar: python -m results.update_results (atualizar mais resultados)
   2. Rodar: python -m results.update_hypothesis_results (calcular CLV)
   3. Analisar CLV médio por hipótese
""")
            
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(check_analysis_ready())
