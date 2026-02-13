# -*- coding: utf-8 -*-
"""
Debug: Verificar fluxo de resultados e relação com hipóteses.
"""

import asyncio
import sys
from datetime import datetime, timezone, timedelta
from sqlalchemy import text

sys.path.insert(0, '.')

from storage.database import Database


async def debug_results():
    db = Database()
    await db.connect()
    
    print("=" * 70)
    print("DEBUG: Fluxo de Resultados e Hipóteses")
    print("=" * 70)
    
    try:
        async with db.async_session() as session:
            
            # ============================================================
            # 1. STATUS DOS JOGOS
            # ============================================================
            print("\n" + "=" * 70)
            print("1. STATUS DOS JOGOS NA TABELA MATCHES")
            print("=" * 70)
            
            result = await session.execute(text("""
                SELECT 
                    status,
                    COUNT(*) as total,
                    COUNT(home_score) as with_score,
                    MIN(kickoff_time) as earliest,
                    MAX(kickoff_time) as latest
                FROM matches
                GROUP BY status
                ORDER BY total DESC
            """))
            print("\n   Status     | Total | Com Score | Kickoff Range")
            for row in result.fetchall():
                print(f"   {row[0]:10} | {row[1]:5} | {row[2]:9} | {row[3]} a {row[4]}")
            
            # ============================================================
            # 2. JOGOS QUE DEVERIAM TER RESULTADO
            # ============================================================
            print("\n" + "=" * 70)
            print("2. JOGOS COM KICKOFF NO PASSADO (deveriam ter resultado)")
            print("=" * 70)
            
            result = await session.execute(text("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN home_score IS NOT NULL THEN 1 END) as with_result,
                    COUNT(CASE WHEN status = 'finished' THEN 1 END) as status_finished
                FROM matches
                WHERE kickoff_time < NOW() - INTERVAL '2 hours'
            """))
            row = result.fetchone()
            print(f"\n   Jogos com kickoff > 2h atrás: {row[0]}")
            print(f"   Com resultado (home_score): {row[1]}")
            print(f"   Com status='finished': {row[2]}")
            
            missing = row[0] - row[1]
            if missing > 0:
                print(f"\n   ⚠️ {missing} jogos SEM resultado mas deveriam ter!")
            
            # Exemplos de jogos sem resultado
            result = await session.execute(text("""
                SELECT id, home_team, away_team, kickoff_time, status, home_score
                FROM matches
                WHERE kickoff_time < NOW() - INTERVAL '2 hours'
                  AND home_score IS NULL
                ORDER BY kickoff_time DESC
                LIMIT 10
            """))
            rows = result.fetchall()
            if rows:
                print("\n   Jogos sem resultado (exemplos):")
                for row in rows:
                    print(f"      ID {row[0]}: {row[1]} vs {row[2]} | {row[3]} | status={row[4]}")
            
            # ============================================================
            # 3. COMO OS RESULTADOS SÃO ATUALIZADOS
            # ============================================================
            print("\n" + "=" * 70)
            print("3. PROCESSO DE ATUALIZAÇÃO DE RESULTADOS")
            print("=" * 70)
            
            # Verifica se existe auto_update_results rodando
            result = await session.execute(text("""
                SELECT 
                    id, home_team, away_team, 
                    home_score, away_score, 
                    status, updated_at
                FROM matches
                WHERE status = 'finished'
                ORDER BY updated_at DESC NULLS LAST
                LIMIT 5
            """))
            rows = result.fetchall()
            if rows:
                print("\n   Últimos jogos finalizados (por updated_at):")
                for row in rows:
                    score = f"{row[3]}-{row[4]}" if row[3] is not None else "N/A"
                    print(f"      ID {row[0]}: {row[1]} vs {row[2]} = {score} | updated: {row[6]}")
            else:
                print("\n   ⚠️ Nenhum jogo com status='finished'!")
            
            # ============================================================
            # 4. RELAÇÃO EVENTOS DE HIPÓTESES <-> JOGOS
            # ============================================================
            print("\n" + "=" * 70)
            print("4. RELAÇÃO EVENTOS DE HIPÓTESES <-> JOGOS")
            print("=" * 70)
            
            # Verificar se match_id existe na tabela matches
            result = await session.execute(text("""
                SELECT 
                    'H1' as tipo,
                    COUNT(*) as total_eventos,
                    COUNT(DISTINCT h.match_id) as matches_unicos,
                    COUNT(DISTINCT CASE WHEN m.id IS NOT NULL THEN h.match_id END) as matches_existentes,
                    COUNT(DISTINCT CASE WHEN m.home_score IS NOT NULL THEN h.match_id END) as matches_com_resultado
                FROM h1_pricing_events h
                LEFT JOIN matches m ON h.match_id = m.id
                UNION ALL
                SELECT 
                    'H3',
                    COUNT(*),
                    COUNT(DISTINCT h.match_id),
                    COUNT(DISTINCT CASE WHEN m.id IS NOT NULL THEN h.match_id END),
                    COUNT(DISTINCT CASE WHEN m.home_score IS NOT NULL THEN h.match_id END)
                FROM h3_line_monotonicity_events h
                LEFT JOIN matches m ON h.match_id = m.id
                UNION ALL
                SELECT 
                    'H3b',
                    COUNT(*),
                    COUNT(DISTINCT h.match_id),
                    COUNT(DISTINCT CASE WHEN m.id IS NOT NULL THEN h.match_id END),
                    COUNT(DISTINCT CASE WHEN m.home_score IS NOT NULL THEN h.match_id END)
                FROM h3b_temporal_reversal_events h
                LEFT JOIN matches m ON h.match_id = m.id
                UNION ALL
                SELECT 
                    'H6',
                    COUNT(*),
                    COUNT(DISTINCT h.match_id),
                    COUNT(DISTINCT CASE WHEN m.id IS NOT NULL THEN h.match_id END),
                    COUNT(DISTINCT CASE WHEN m.home_score IS NOT NULL THEN h.match_id END)
                FROM h6_correlation_lag_events h
                LEFT JOIN matches m ON h.match_id = m.id
            """))
            print("\n   Tipo | Eventos | Matches | Match Existe | Match c/ Resultado")
            for row in result.fetchall():
                print(f"   {row[0]:4} | {row[1]:7} | {row[2]:7} | {row[3]:12} | {row[4]}")
            
            # ============================================================
            # 5. VERIFICAR ODDS DE FECHAMENTO
            # ============================================================
            print("\n" + "=" * 70)
            print("5. ODDS DE FECHAMENTO DISPONÍVEIS")
            print("=" * 70)
            
            result = await session.execute(text("""
                WITH finished_matches AS (
                    SELECT id, home_team, kickoff_time
                    FROM matches
                    WHERE status = 'finished'
                ),
                closing_odds AS (
                    SELECT 
                        m.id,
                        m.home_team,
                        COUNT(DISTINCT b.ah_line) as linhas_total,
                        COUNT(DISTINCT CASE WHEN b.scraped_at < m.kickoff_time THEN b.ah_line END) as linhas_pre_match
                    FROM finished_matches m
                    LEFT JOIN best_odds_history b ON m.id = b.match_id
                    GROUP BY m.id, m.home_team
                )
                SELECT 
                    id, home_team, linhas_total, linhas_pre_match
                FROM closing_odds
                ORDER BY id DESC
                LIMIT 10
            """))
            print("\n   ID | Time | Linhas Total | Linhas Pré-Match")
            for row in result.fetchall():
                status = "✅" if row[3] > 0 else "❌ SEM CLOSING"
                print(f"   {row[0]:3} | {row[1][:20]:20} | {row[2]:12} | {row[3]} {status}")
            
            # ============================================================
            # RESUMO
            # ============================================================
            print("\n" + "=" * 70)
            print("RESUMO E DIAGNÓSTICO")
            print("=" * 70)
            
            # Conta jogos que precisam de resultado
            result = await session.execute(text("""
                SELECT COUNT(*) FROM matches 
                WHERE kickoff_time < NOW() - INTERVAL '2 hours'
                  AND home_score IS NULL
            """))
            missing_results = result.scalar()
            
            # Conta eventos sem CLV mas de jogos finalizados
            result = await session.execute(text("""
                SELECT COUNT(*) 
                FROM h1_pricing_events h
                JOIN matches m ON h.match_id = m.id
                WHERE m.status = 'finished' AND h.clv IS NULL
            """))
            events_without_clv = result.scalar()
            
            print(f"""
   Jogos sem resultado (kickoff > 2h): {missing_results}
   Eventos H1 de jogos finalizados sem CLV: {events_without_clv}
   
   AÇÕES NECESSÁRIAS:
""")
            if missing_results > 0:
                print(f"   1. ⚠️ Rodar atualização de resultados:")
                print(f"      python -m results.update_results")
                print(f"      ou")
                print(f"      python -m results.auto_update_results")
            
            if events_without_clv > 0:
                print(f"\n   2. ⚠️ Rodar cálculo de CLV:")
                print(f"      python -m results.update_hypothesis_results")
            
            if missing_results == 0 and events_without_clv == 0:
                print("   ✅ Tudo parece estar funcionando corretamente!")
                
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(debug_results())
