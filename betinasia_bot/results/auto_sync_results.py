#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto sync: sincroniza matches do audit + busca resultados via API-Football.
Roda em loop a cada 6 horas.

Uso:
    python -m results.auto_sync_results
"""

import asyncio
import signal
import sys
from datetime import datetime, timezone, timedelta
from loguru import logger

sys.path.insert(0, '.')

from sqlalchemy import text
from storage.database import Database
from results.api_football import APIFootballClient


class AutoSyncResults:

    INTERVAL = 6 * 3600  # 6 horas

    def __init__(self):
        self.running = True

    async def run(self):
        signal.signal(signal.SIGTERM, lambda s, f: setattr(self, 'running', False))
        signal.signal(signal.SIGINT, lambda s, f: setattr(self, 'running', False))

        logger.info("Auto Sync Results iniciado (a cada 6h)")

        while self.running:
            try:
                await self._sync_matches()
                await self._update_results()
                logger.info(f"Proximo sync em {self.INTERVAL // 3600}h")
            except Exception as e:
                logger.error(f"Erro no sync: {e}")

            # Espera em intervalos menores para responder a sinais
            for _ in range(self.INTERVAL // 60):
                if not self.running:
                    break
                await asyncio.sleep(60)

    async def _sync_matches(self):
        """Sincroniza jogos do audit para tabela matches."""
        db = Database()
        await db.connect()

        try:
            async with db.async_session() as session:
                result = await session.execute(text(
                    "SELECT DISTINCT a.event_id, a.home_team, a.away_team, a.league "
                    "FROM betslip_audit_results a "
                    "LEFT JOIN matches m ON m.external_id = a.event_id "
                    "WHERE m.id IS NULL AND a.event_id IS NOT NULL "
                    "AND a.home_team IS NOT NULL AND a.home_team != '' AND a.home_team != '?'"
                ))
                missing = result.fetchall()

                if not missing:
                    logger.info("Sync matches: todos sincronizados")
                    return

                inserted = 0
                for row in missing:
                    event_id, home, away, league = row
                    kickoff = None
                    try:
                        date_str = event_id.split(',')[0]
                        kickoff = datetime.strptime(date_str, "%Y-%m-%d").replace(
                            hour=15, tzinfo=timezone.utc)
                    except:
                        kickoff = datetime.now(timezone.utc)

                    try:
                        await session.execute(text(
                            "INSERT INTO matches (external_id, home_team, away_team, league, kickoff_time, status) "
                            "VALUES (:eid, :home, :away, :league, :kickoff, 'pending') "
                            "ON CONFLICT (external_id) DO NOTHING"
                        ), {'eid': event_id, 'home': home, 'away': away,
                            'league': league or 'Unknown', 'kickoff': kickoff})
                        inserted += 1
                    except:
                        pass

                await session.commit()
                logger.info(f"Sync matches: {inserted} novos inseridos")
        finally:
            await db.close()

    async def _update_results(self):
        """Busca resultados via API-Football para jogos recentes."""
        db = Database()
        await db.connect()

        try:
            api = APIFootballClient()

            async with db.async_session() as session:
                # Busca datas com jogos sem resultado (últimos 2 dias apenas - limite free)
                result = await session.execute(text(
                    "SELECT DISTINCT DATE(m.kickoff_time) as d "
                    "FROM matches m "
                    "WHERE m.home_score IS NULL "
                    "AND m.kickoff_time BETWEEN NOW() - interval '2 days' AND NOW() "
                    "ORDER BY d"
                ))
                dates = [row[0] for row in result]

                if not dates:
                    logger.info("Update results: nenhuma data pendente")
                    await api.close()
                    return

                logger.info(f"Update results: {len(dates)} datas a consultar")

                total_updated = 0
                for d in dates:
                    try:
                        results = await api.get_results_by_date(str(d))
                        logger.info(f"  {d}: {len(results)} resultados da API")

                        for match_result in results:
                            try:
                                await session.execute(text(
                                    "UPDATE matches SET home_score = :hs, away_score = :as_, "
                                    "status = 'finished' "
                                    "WHERE home_score IS NULL "
                                    "AND home_team ILIKE :home AND away_team ILIKE :away "
                                    "AND DATE(kickoff_time) = :d"
                                ), {
                                    'hs': match_result.home_score,
                                    'as_': match_result.away_score,
                                    'home': f"%{match_result.home_team[:15]}%",
                                    'away': f"%{match_result.away_team[:15]}%",
                                    'd': str(d),
                                })
                                total_updated += 1
                            except Exception as e:
                                logger.debug(f"  Erro atualizando {match_result.home_team}: {e}")
                    except Exception as e:
                        logger.warning(f"  Erro consultando {d}: {e}")

                await session.commit()
                logger.info(f"Update results: {total_updated} jogos atualizados")

            await api.close()
        finally:
            await db.close()


async def main():
    logger.remove()
    logger.add(sys.stderr,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<7}</level> | <level>{message}</level>",
               level="INFO")
    logger.add("logs/auto_sync_{time:YYYY-MM-DD}.log", rotation="00:00", retention="30 days", level="DEBUG")

    sync = AutoSyncResults()
    await sync.run()


if __name__ == "__main__":
    asyncio.run(main())
