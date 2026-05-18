# -*- coding: utf-8 -*-
"""
Jobs agendados do sistema.

Usa APScheduler para agendar tarefas de scraping.
"""

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from config import settings


# Ligas por tier
TIER_1_LEAGUES = [
    "England Premier League",
    "Germany Bundesliga",
    "Spain La Liga",
    "Italy Serie A",
    "France Ligue 1",
]

TIER_2_LEAGUES = [
    "England Championship",
    "Germany 2. Bundesliga",
    "Netherlands Eredivisie",
    "Portugal Primeira Liga",
    "Belgium Pro League",
]

TIER_3_LEAGUES = [
    "Turkey Super Lig",
    "Brazil Serie A",
    "Argentina Primera Division",
    "Mexico Liga MX",
]


def start_scheduler(db, cache) -> AsyncIOScheduler:
    """
    Inicia o scheduler com os jobs configurados.
    
    Args:
        db: Instância do Database
        cache: Instância do OddsCache
        
    Returns:
        Scheduler iniciado
    """
    scheduler = AsyncIOScheduler()
    
    # Job para ligas Tier 1 (alta frequência)
    scheduler.add_job(
        scrape_tier_job,
        trigger=IntervalTrigger(seconds=settings.scrape_interval_tier1),
        args=[1, TIER_1_LEAGUES, db, cache],
        id="scrape_tier1",
        name="Scrape Tier 1 Leagues",
        replace_existing=True,
    )
    
    # Job para ligas Tier 2 (frequência média)
    scheduler.add_job(
        scrape_tier_job,
        trigger=IntervalTrigger(seconds=settings.scrape_interval_tier2),
        args=[2, TIER_2_LEAGUES, db, cache],
        id="scrape_tier2",
        name="Scrape Tier 2 Leagues",
        replace_existing=True,
    )
    
    # Job para ligas Tier 3 (baixa frequência)
    scheduler.add_job(
        scrape_tier_job,
        trigger=IntervalTrigger(seconds=settings.scrape_interval_tier3),
        args=[3, TIER_3_LEAGUES, db, cache],
        id="scrape_tier3",
        name="Scrape Tier 3 Leagues",
        replace_existing=True,
    )
    
    scheduler.start()
    logger.info("Scheduler iniciado")
    
    return scheduler


async def scrape_tier_job(tier: int, leagues: list, db, cache):
    """
    Job de scraping para um tier de ligas.
    
    Este é um placeholder - a implementação real deve:
    1. Fazer login (ou reusar sessão)
    2. Iterar pelas ligas
    3. Detectar oportunidades
    4. Disparar scoring
    5. Executar apostas aprovadas
    """
    logger.debug(f"Executando scrape Tier {tier}")
    
    # TODO: Implementar lógica completa
    # Por enquanto, apenas loga
    for league in leagues:
        logger.debug(f"  - {league}")
