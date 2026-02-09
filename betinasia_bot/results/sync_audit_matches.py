# -*- coding: utf-8 -*-
"""
Sincroniza jogos do audit com a tabela matches.

Para jogos que o audit capturou mas o collector não registrou,
cria registros na tabela matches usando dados do audit.

Depois, o update_results.py pode buscar resultados para esses jogos.

Uso:
    python -m results.sync_audit_matches
"""

import asyncio
import sys
from datetime import datetime, timezone
from loguru import logger

sys.path.insert(0, '.')

from sqlalchemy import text
from storage.database import Database


async def sync():
    db = Database()
    await db.connect()
    
    async with db.async_session() as session:
        # Encontra auditorias sem match na tabela matches
        result = await session.execute(text("""
            SELECT DISTINCT a.event_id, a.home_team, a.away_team, a.league,
                   MIN(a.audited_at) as first_audit
            FROM betslip_audit_results a
            LEFT JOIN matches m ON m.external_id = a.event_id
            WHERE m.id IS NULL
              AND a.event_id IS NOT NULL
              AND a.home_team IS NOT NULL
              AND a.home_team != ''
              AND a.home_team != '?'
            GROUP BY a.event_id, a.home_team, a.away_team, a.league
        """))
        missing = result.fetchall()
        
        if not missing:
            print("Todos os jogos do audit ja estao na tabela matches.")
            await db.close()
            return
        
        print(f"Encontrados {len(missing)} jogos do audit sem match. Inserindo...")
        
        inserted = 0
        for row in missing:
            event_id = row[0]
            home = row[1]
            away = row[2]
            league = row[3] or "Unknown"
            first_audit = row[4]
            
            # Extrai kickoff da data no event_id (formato: 2026-02-08,176,178)
            kickoff = None
            try:
                date_str = event_id.split(',')[0]
                kickoff = datetime.strptime(date_str, "%Y-%m-%d").replace(
                    hour=15, tzinfo=timezone.utc)  # Default 15:00 UTC
            except:
                kickoff = first_audit
            
            try:
                await session.execute(text(
                    "INSERT INTO matches (external_id, home_team, away_team, league, kickoff_time, status) "
                    "VALUES (:eid, :home, :away, :league, :kickoff, 'pending') "
                    "ON CONFLICT (external_id) DO NOTHING"
                ), {
                    'eid': event_id,
                    'home': home,
                    'away': away,
                    'league': league,
                    'kickoff': kickoff,
                })
                inserted += 1
            except Exception as e:
                logger.debug(f"Erro inserindo {event_id}: {e}")
        
        await session.commit()
        print(f"Inseridos {inserted} novos matches.")
    
    await db.close()


if __name__ == "__main__":
    asyncio.run(sync())
