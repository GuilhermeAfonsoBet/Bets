# -*- coding: utf-8 -*-
"""
Verificação completa do sistema - Checa se está pronto para análise estatística.
"""

import asyncio
import sys
from datetime import datetime, timezone, timedelta
from sqlalchemy import text

sys.path.insert(0, '.')

from storage.database import Database


async def verify_system():
    """Verificação completa do sistema."""
    
    db = Database()
    await db.connect()
    
    print("=" * 70)
    print("VERIFICAÇÃO COMPLETA DO SISTEMA")
    print("=" * 70)
    
    issues = []
    
    try:
        async with db.async_session() as session:
            
            # ============================================================
            # 1. COLETA DE ODDS
            # ============================================================
            print("\n" + "=" * 70)
            print("1. COLETA DE ODDS (best_odds_history)")
            print("=" * 70)
            
            result = await session.execute(text("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(DISTINCT match_id) as matches,
                    MIN(scraped_at) as first_record,
                    MAX(scraped_at) as last_record,
                    COUNT(DISTINCT DATE(scraped_at)) as days_collected
                FROM best_odds_history
            """))
            row = result.fetchone()
            print(f"\n   Total de registros: {row[0]:,}")
            print(f"   Jogos únicos: {row[1]:,}")
            print(f"   Primeira coleta: {row[2]}")
            print(f"   Última coleta: {row[3]}")
            print(f"   Dias com dados: {row[4]}")
            
            # Verifica se está coletando recentemente
            if row[3]:
                age = (datetime.now(timezone.utc) - row[3].replace(tzinfo=timezone.utc)).total_seconds()
                if age > 300:  # 5 minutos
                    issues.append(f"⚠️ Última coleta há {age/60:.0f} minutos")
                    print(f"\n   ⚠️ ALERTA: Última coleta há {age/60:.0f} minutos!")
                else:
                    print(f"\n   ✅ Coleta ativa (última há {age:.0f}s)")
            
            # Distribuição por tipo de mercado
            print("\n   Distribuição por tipo de linha:")
            result = await session.execute(text("""
                SELECT 
                    CASE 
                        WHEN ah_line LIKE 'OU_%' THEN 'OU'
                        WHEN ah_line = '1X2' THEN '1X2'
                        WHEN ah_line = '1X2_DRAW' THEN '1X2_DRAW'
                        ELSE 'AH'
                    END as market_type,
                    COUNT(*) as cnt
                FROM best_odds_history
                GROUP BY 1
                ORDER BY 2 DESC
            """))
            for row in result.fetchall():
                print(f"      {row[0]}: {row[1]:,}")
            
            # ============================================================
            # 2. RESULTADOS DOS JOGOS
            # ============================================================
            print("\n" + "=" * 70)
            print("2. RESULTADOS DOS JOGOS (match_results)")
            print("=" * 70)
            
            # Verifica se tabela existe
            result = await session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'match_results'
                )
            """))
            has_results_table = result.scalar()
            
            if has_results_table:
                result = await session.execute(text("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(CASE WHEN home_score IS NOT NULL THEN 1 END) as with_score
                    FROM match_results
                """))
                row = result.fetchone()
                print(f"\n   Total de jogos: {row[0]:,}")
                print(f"   Com resultado: {row[1]:,}")
                
                if row[1] == 0:
                    issues.append("⚠️ Nenhum resultado de jogo coletado")
                    print("\n   ⚠️ ALERTA: Nenhum resultado!")
                else:
                    print(f"\n   ✅ Resultados sendo coletados")
            else:
                # Verifica na tabela matches
                result = await session.execute(text("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(CASE WHEN ft_home IS NOT NULL OR status = 'Ended' THEN 1 END) as with_result
                    FROM matches
                """))
                row = result.fetchone()
                print(f"\n   Total de jogos (tabela matches): {row[0]:,}")
                print(f"   Com resultado (ft_home != NULL ou Ended): {row[1]:,}")
                
                if row[1] == 0:
                    issues.append("⚠️ Nenhum resultado de jogo encontrado")
                    print("\n   ⚠️ ALERTA: Nenhum resultado!")
            
            # ============================================================
            # 3. ODDS DE FECHAMENTO
            # ============================================================
            print("\n" + "=" * 70)
            print("3. ODDS DE FECHAMENTO (para cálculo de CLV)")
            print("=" * 70)
            
            # Verifica se conseguimos pegar odds de fechamento
            result = await session.execute(text("""
                WITH last_odds AS (
                    SELECT 
                        match_id,
                        ah_line,
                        MAX(scraped_at) as last_scrape
                    FROM best_odds_history
                    GROUP BY match_id, ah_line
                )
                SELECT 
                    COUNT(DISTINCT match_id) as matches_with_closing_odds
                FROM last_odds
            """))
            row = result.fetchone()
            print(f"\n   Jogos com odds de fechamento disponíveis: {row[0]:,}")
            
            # Verifica jogos finalizados com odds de fechamento
            result = await session.execute(text("""
                SELECT 
                    m.id,
                    m.home_team,
                    m.away_team,
                    m.ft_home,
                    m.ft_away,
                    (SELECT MAX(scraped_at) FROM best_odds_history WHERE match_id = m.id) as last_odd_time
                FROM matches m
                WHERE m.ft_home IS NOT NULL OR m.status = 'Ended'
                ORDER BY m.id DESC
                LIMIT 5
            """))
            rows = result.fetchall()
            if rows:
                print("\n   Últimos jogos finalizados:")
                for row in rows:
                    score = f"{row[3]}-{row[4]}" if row[3] is not None else "N/A"
                    print(f"      ID {row[0]}: {row[1]} vs {row[2]} = {score} | última odd: {row[5]}")
            
            # ============================================================
            # 4. EVENTOS DE HIPÓTESES
            # ============================================================
            print("\n" + "=" * 70)
            print("4. EVENTOS DE HIPÓTESES")
            print("=" * 70)
            
            tables = [
                ("h1_pricing_events", "H1 - Pricing"),
                ("h3_line_monotonicity_events", "H3 - Linhas"),
                ("h3b_temporal_reversal_events", "H3B - Reversões"),
                ("h6_correlation_lag_events", "H6 - Lag"),
            ]
            
            for table, name in tables:
                result = await session.execute(text(f"""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(DISTINCT match_id) as matches,
                        COUNT(CASE WHEN clv IS NOT NULL THEN 1 END) as with_clv
                    FROM {table}
                """))
                row = result.fetchone()
                clv_status = f"✅ {row[2]} com CLV" if row[2] > 0 else "⏳ CLV pendente"
                print(f"\n   {name}:")
                print(f"      Eventos: {row[0]:,} | Jogos: {row[1]:,} | {clv_status}")
            
            # ============================================================
            # 5. CAMPOS NECESSÁRIOS PARA ANÁLISE
            # ============================================================
            print("\n" + "=" * 70)
            print("5. CAMPOS PARA ANÁLISE ESTATÍSTICA")
            print("=" * 70)
            
            # H1
            result = await session.execute(text("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(recommended_side) as has_rec_side,
                    COUNT(recommended_odd) as has_rec_odd,
                    COUNT(clv) as has_clv,
                    COUNT(bet_result) as has_result
                FROM h1_pricing_events
            """))
            row = result.fetchone()
            print(f"\n   H1 Pricing:")
            print(f"      Total: {row[0]} | recommended_side: {row[1]} | recommended_odd: {row[2]}")
            print(f"      clv: {row[3]} | bet_result: {row[4]}")
            
            # ============================================================
            # 6. ESTRUTURA DA TABELA MATCHES
            # ============================================================
            print("\n" + "=" * 70)
            print("6. ESTRUTURA DA TABELA MATCHES")
            print("=" * 70)
            
            result = await session.execute(text("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'matches'
                ORDER BY ordinal_position
            """))
            cols = result.fetchall()
            print("\n   Colunas disponíveis:")
            for col in cols:
                print(f"      {col[0]}: {col[1]}")
            
            # ============================================================
            # 7. EXEMPLO DE JOGO COMPLETO
            # ============================================================
            print("\n" + "=" * 70)
            print("7. EXEMPLO DE JOGO COMPLETO (para validação)")
            print("=" * 70)
            
            # Busca um jogo finalizado com dados completos
            result = await session.execute(text("""
                SELECT m.id, m.home_team, m.away_team, m.ft_home, m.ft_away, m.status
                FROM matches m
                WHERE m.ft_home IS NOT NULL
                ORDER BY m.id DESC
                LIMIT 1
            """))
            match = result.fetchone()
            
            if match:
                match_id = match[0]
                print(f"\n   Jogo ID {match_id}: {match[1]} vs {match[2]}")
                print(f"   Resultado: {match[3]}-{match[4]} | Status: {match[5]}")
                
                # Odds deste jogo
                result = await session.execute(text(f"""
                    SELECT 
                        ah_line,
                        COUNT(*) as records,
                        MIN(scraped_at) as first,
                        MAX(scraped_at) as last
                    FROM best_odds_history
                    WHERE match_id = {match_id}
                    GROUP BY ah_line
                    ORDER BY ah_line
                    LIMIT 10
                """))
                odds = result.fetchall()
                if odds:
                    print(f"\n   Odds coletadas:")
                    for o in odds:
                        print(f"      {o[0]}: {o[1]} registros ({o[2]} a {o[3]})")
                
                # Eventos deste jogo
                for table, name in tables:
                    result = await session.execute(text(f"""
                        SELECT COUNT(*) FROM {table} WHERE match_id = {match_id}
                    """))
                    cnt = result.scalar()
                    if cnt > 0:
                        print(f"   {name}: {cnt} eventos")
            else:
                issues.append("⚠️ Nenhum jogo finalizado encontrado")
                print("\n   ⚠️ Nenhum jogo finalizado para validar")
            
            # ============================================================
            # RESUMO
            # ============================================================
            print("\n" + "=" * 70)
            print("RESUMO - PRONTO PARA ANÁLISE?")
            print("=" * 70)
            
            if issues:
                print("\n   ⚠️ PROBLEMAS ENCONTRADOS:")
                for issue in issues:
                    print(f"      {issue}")
            else:
                print("\n   ✅ Sistema aparenta estar funcionando corretamente!")
            
            print("""
   Para rodar análise estatística das hipóteses:
   
   1. Aguardar jogos finalizarem (ter ft_home/ft_away preenchido)
   2. Rodar: python -m results.update_hypothesis_results
   3. Analisar: SELECT AVG(clv_pct), COUNT(*) FROM h1_pricing_events WHERE clv IS NOT NULL
   """)
            
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(verify_system())
