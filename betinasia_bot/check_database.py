#!/usr/bin/env python3
"""
Verifica os dados salvos no banco de dados.
"""
import asyncio
import sys
from datetime import datetime, timezone, timedelta

sys.path.insert(0, '.')

from storage.database import Database

async def check_database():
    print("\n" + "="*70)
    print("VERIFICAÇÃO DO BANCO DE DADOS")
    print("="*70)
    
    db = Database()
    await db.connect()
    
    try:
        # 1. Conta total de partidas
        from sqlalchemy import text
        
        async with db._session() as session:
            # Total de partidas
            result = await session.execute(text("SELECT COUNT(*) FROM matches"))
            total_matches = result.scalar()
            print(f"\n[1] Total de partidas no banco: {total_matches}")
            
            # Partidas por liga
            result = await session.execute(text("""
                SELECT league, COUNT(*) as count 
                FROM matches 
                GROUP BY league 
                ORDER BY count DESC
            """))
            rows = result.fetchall()
            
            print(f"\n[2] Partidas por liga:")
            for row in rows:
                print(f"    {row[0]}: {row[1]} partidas")
            
            # Total de odds
            result = await session.execute(text("SELECT COUNT(*) FROM odds_history"))
            total_odds = result.scalar()
            print(f"\n[3] Total de registros de odds: {total_odds}")
            
            # Odds por liga
            result = await session.execute(text("""
                SELECT m.league, COUNT(o.id) as count 
                FROM odds_history o
                JOIN matches m ON o.match_id = m.id
                GROUP BY m.league
                ORDER BY count DESC
            """))
            rows = result.fetchall()
            
            print(f"\n[4] Registros de odds por liga:")
            for row in rows:
                print(f"    {row[0]}: {row[1]} odds")
            
            # Últimas partidas inseridas
            result = await session.execute(text("""
                SELECT home_team, away_team, league, kickoff_time, created_at
                FROM matches 
                ORDER BY created_at DESC 
                LIMIT 10
            """))
            rows = result.fetchall()
            
            print(f"\n[5] Últimas 10 partidas inseridas:")
            for row in rows:
                print(f"    {row[0]} vs {row[1]}")
                print(f"        Liga: {row[2]}")
                print(f"        Kickoff: {row[3]}")
                print(f"        Inserido: {row[4]}")
                print()
            
            # Amostra de odds
            result = await session.execute(text("""
                SELECT m.home_team, m.away_team, o.ah_line, o.bookmaker, 
                       o.home_odds, o.away_odds, o.scraped_at
                FROM odds_history o
                JOIN matches m ON o.match_id = m.id
                ORDER BY o.scraped_at DESC
                LIMIT 10
            """))
            rows = result.fetchall()
            
            print(f"\n[6] Últimas 10 odds inseridas:")
            for row in rows:
                print(f"    {row[0]} vs {row[1]}")
                print(f"        AH: {row[2]}, Bookmaker: {row[3]}")
                print(f"        Home: {row[4]:.3f}, Away: {row[5]:.3f}")
                print(f"        Scraped: {row[6]}")
                print()
            
            # Verifica bookmakers únicos
            result = await session.execute(text("""
                SELECT DISTINCT bookmaker 
                FROM odds_history 
                ORDER BY bookmaker
            """))
            rows = result.fetchall()
            
            print(f"\n[7] Bookmakers únicos: {len(rows)}")
            bookmakers = [row[0] for row in rows]
            print(f"    {', '.join(bookmakers)}")
            
            # Verifica se há odds zeradas
            result = await session.execute(text("""
                SELECT COUNT(*) 
                FROM odds_history 
                WHERE home_odds = 0 OR away_odds = 0
            """))
            zero_odds = result.scalar()
            total_odds_check = total_odds if total_odds else 1
            pct_zero = (zero_odds / total_odds_check) * 100
            
            print(f"\n[8] Odds zeradas: {zero_odds} ({pct_zero:.2f}%)")
            
            if pct_zero > 5:
                print(f"    ⚠️ ATENÇÃO: Muitas odds zeradas!")
            else:
                print(f"    ✓ Proporção aceitável de odds zeradas")
            
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
    
    await db.close()
    
    print("\n" + "="*70)
    print("VERIFICAÇÃO CONCLUÍDA")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(check_database())
