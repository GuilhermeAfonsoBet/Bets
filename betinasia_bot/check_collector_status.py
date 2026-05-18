#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Verifica status do coletor e estatisticas do banco.

Uso:
    python check_collector_status.py
"""

import asyncio
from datetime import datetime, timezone, timedelta
from sqlalchemy import select, func, text
from storage.database import Database
from storage.models import Match, BestOddsHistory


async def check_status():
    """Verifica status do coletor e banco de dados."""
    
    db = Database()
    await db.connect()
    
    print("=" * 60)
    print("STATUS DO COLETOR - BetinAsia Bot")
    print("=" * 60)
    print(f"Data/Hora: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print()
    
    async with db.async_session() as session:
        # Total de jogos
        result = await session.execute(select(func.count(Match.id)))
        total_matches = result.scalar()
        print(f"Total de jogos no banco: {total_matches}")
        
        # Jogos por status
        result = await session.execute(
            select(Match.status, func.count(Match.id))
            .group_by(Match.status)
        )
        print("\nJogos por status:")
        for status, count in result:
            print(f"  {status or 'null'}: {count}")
        
        # Total de registros de odds
        result = await session.execute(select(func.count(BestOddsHistory.id)))
        total_odds = result.scalar()
        print(f"\nTotal de registros de odds: {total_odds}")
        
        # Coletas nas ultimas 24 horas
        yesterday = datetime.now(timezone.utc) - timedelta(hours=24)
        result = await session.execute(
            select(func.count(BestOddsHistory.id))
            .where(BestOddsHistory.scraped_at >= yesterday)
        )
        odds_24h = result.scalar()
        print(f"Registros de odds (ultimas 24h): {odds_24h}")
        
        # Ultima coleta
        result = await session.execute(
            select(func.max(BestOddsHistory.scraped_at))
        )
        last_collection = result.scalar()
        if last_collection:
            age = datetime.now(timezone.utc) - last_collection.replace(tzinfo=timezone.utc)
            print(f"\nUltima coleta: {last_collection.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            print(f"Idade da ultima coleta: {age}")
            
            if age > timedelta(minutes=5):
                print("  ⚠️  ALERTA: Ultima coleta ha mais de 5 minutos!")
            else:
                print("  ✅ Coleta ativa")
        else:
            print("\n⚠️  Nenhuma coleta encontrada!")
        
        # Top 10 ligas por numero de jogos
        print("\n" + "=" * 60)
        print("TOP 10 LIGAS (por numero de jogos)")
        print("=" * 60)
        result = await session.execute(
            select(Match.league, func.count(Match.id).label('count'))
            .group_by(Match.league)
            .order_by(func.count(Match.id).desc())
            .limit(10)
        )
        for league, count in result:
            print(f"  {count:4d} jogos: {league}")
        
        # Coletas por hora (ultimas 6 horas)
        print("\n" + "=" * 60)
        print("COLETAS POR HORA (ultimas 6 horas)")
        print("=" * 60)
        
        for hours_ago in range(6):
            start = datetime.now(timezone.utc) - timedelta(hours=hours_ago+1)
            end = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
            
            result = await session.execute(
                select(func.count(BestOddsHistory.id))
                .where(BestOddsHistory.scraped_at >= start)
                .where(BestOddsHistory.scraped_at < end)
            )
            count = result.scalar()
            
            hour_label = f"{hours_ago}h atras" if hours_ago > 0 else "Ultima hora"
            print(f"  {hour_label}: {count:,} registros")
        
        # Estimativa de tamanho do banco
        print("\n" + "=" * 60)
        print("TAMANHO ESTIMADO DO BANCO")
        print("=" * 60)
        
        try:
            result = await session.execute(text("""
                SELECT 
                    relname as table,
                    pg_size_pretty(pg_total_relation_size(relid)) as size
                FROM pg_catalog.pg_statio_user_tables
                WHERE relname IN ('matches', 'best_odds_history', 'odds_history')
                ORDER BY pg_total_relation_size(relid) DESC
            """))
            for table, size in result:
                print(f"  {table}: {size}")
        except Exception as e:
            print(f"  (Erro ao obter tamanho: {e})")
    
    await db.close()
    
    print("\n" + "=" * 60)
    print("Para ver logs do servico:")
    print("  sudo journalctl -u betinasia-collector -f")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(check_status())
