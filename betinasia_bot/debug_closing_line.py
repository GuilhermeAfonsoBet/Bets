# -*- coding: utf-8 -*-
"""
Debug: Por que a closing line não está sendo encontrada?
"""

import asyncio
import sys
from sqlalchemy import text

sys.path.insert(0, '.')

from storage.database import Database


async def debug_closing():
    db = Database()
    await db.connect()
    
    print("=" * 70)
    print("DEBUG: Closing Line - Match 356")
    print("=" * 70)
    
    try:
        async with db.async_session() as session:
            
            # 1. Info do jogo
            print("\n1. INFORMAÇÕES DO JOGO 356")
            print("-" * 50)
            result = await session.execute(text("""
                SELECT id, home_team, away_team, kickoff_time, home_score, away_score, status
                FROM matches WHERE id = 356
            """))
            match = result.fetchone()
            if match:
                print(f"   ID: {match[0]}")
                print(f"   {match[1]} vs {match[2]}")
                print(f"   Kickoff: {match[3]}")
                print(f"   Resultado: {match[4]}-{match[5]}")
                print(f"   Status: {match[6]}")
                kickoff = match[3]
            
            # 2. Linhas de odds disponíveis
            print("\n2. LINHAS DE ODDS DISPONÍVEIS")
            print("-" * 50)
            result = await session.execute(text("""
                SELECT DISTINCT ah_line
                FROM best_odds_history
                WHERE match_id = 356
                ORDER BY ah_line
            """))
            lines = [row[0] for row in result.fetchall()]
            print(f"   Linhas: {lines}")
            
            # 3. Eventos H1 para match 356
            print("\n3. EVENTOS H1 DO MATCH 356")
            print("-" * 50)
            result = await session.execute(text("""
                SELECT id, market_type, ah_line, recommended_side, recommended_odd, detected_at
                FROM h1_pricing_events
                WHERE match_id = 356
                LIMIT 10
            """))
            for row in result.fetchall():
                print(f"   id={row[0]}: {row[1]} | linha='{row[2]}' | side={row[3]} | odd={row[4]}")
                print(f"           detected_at={row[5]}")
            
            # 4. Tentar encontrar a closing line para um evento específico
            print("\n4. BUSCA DE CLOSING LINE")
            print("-" * 50)
            
            # Pegar um evento H1
            result = await session.execute(text("""
                SELECT id, market_type, ah_line, recommended_side
                FROM h1_pricing_events
                WHERE match_id = 356
                LIMIT 1
            """))
            event = result.fetchone()
            if event:
                print(f"   Evento: id={event[0]}, market={event[1]}, line='{event[2]}', side={event[3]}")
                
                # Construir linha de busca
                ah_line = event[2]
                market_type = event[1]
                
                if market_type == "OU":
                    search_line = f"OU_{ah_line}"
                elif market_type == "1X2":
                    search_line = "1X2"
                else:
                    search_line = ah_line
                
                print(f"   Linha de busca: '{search_line}'")
                
                # Verificar se existe no banco
                result = await session.execute(text(f"""
                    SELECT COUNT(*), MIN(scraped_at), MAX(scraped_at)
                    FROM best_odds_history
                    WHERE match_id = 356 AND ah_line = '{search_line}'
                """))
                row = result.fetchone()
                print(f"   Registros encontrados: {row[0]}")
                if row[0] > 0:
                    print(f"   Período: {row[1]} a {row[2]}")
                
                # Verificar condição de kickoff
                print(f"\n   Kickoff time: {kickoff}")
                result = await session.execute(text(f"""
                    SELECT COUNT(*)
                    FROM best_odds_history
                    WHERE match_id = 356 
                      AND ah_line = '{search_line}'
                      AND scraped_at < '{kickoff}'
                """))
                before_kickoff = result.scalar()
                print(f"   Registros ANTES do kickoff: {before_kickoff}")
                
                result = await session.execute(text(f"""
                    SELECT COUNT(*)
                    FROM best_odds_history
                    WHERE match_id = 356 
                      AND ah_line = '{search_line}'
                      AND scraped_at >= '{kickoff}'
                """))
                after_kickoff = result.scalar()
                print(f"   Registros DEPOIS do kickoff: {after_kickoff}")
            
            # 5. Comparar formatos de linha
            print("\n5. COMPARAÇÃO DE FORMATOS")
            print("-" * 50)
            result = await session.execute(text("""
                SELECT DISTINCT h.ah_line as evento_linha, b.ah_line as banco_linha
                FROM h1_pricing_events h
                LEFT JOIN best_odds_history b ON h.match_id = b.match_id AND h.ah_line = b.ah_line
                WHERE h.match_id = 356
                LIMIT 10
            """))
            print("   Linha no evento | Linha no banco (match)")
            for row in result.fetchall():
                match_status = "✓" if row[1] else "✗ NÃO ENCONTRADA"
                print(f"   '{row[0]}' | {match_status}")
                
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(debug_closing())
