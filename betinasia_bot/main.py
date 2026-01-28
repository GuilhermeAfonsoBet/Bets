#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BetinAsia Bot - Sistema de Scraping e Apostas Automatizadas

Uso:
    python main.py              # Roda em modo normal
    python main.py --dry-run    # Roda sem executar apostas reais
    python main.py --collect    # Modo coleta de dados (sem scoring/execução)
"""

import asyncio
import argparse
from datetime import datetime, timezone
from loguru import logger
import sys

from config import settings
from scraper import BetinAsiaScraper
from storage import Database
from cache.redis_cache import OddsCache
from scheduler.jobs import start_scheduler


# Configuração do logger
def setup_logging():
    """Configura o sistema de logs."""
    
    # Remove handler padrão
    logger.remove()
    
    # Console (colorido)
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=settings.log_level,
        colorize=True,
    )
    
    # Arquivo (rotativo)
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        settings.log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        compression="gz",
    )


async def initialize():
    """Inicializa os componentes do sistema."""
    logger.info("=" * 60)
    logger.info("BETINASIA BOT - Iniciando...")
    logger.info("=" * 60)
    logger.info(f"Ambiente: {settings.environment}")
    logger.info(f"Dry Run: {settings.dry_run}")
    
    # Conecta ao banco de dados
    db = Database()
    await db.connect()
    logger.info("Banco de dados conectado")
    
    # Conecta ao cache
    cache = OddsCache()
    await cache.connect()
    logger.info("Cache conectado")
    
    return db, cache


async def run_single_scrape(league: str, db: Database, cache: OddsCache):
    """
    Executa um único ciclo de scraping para uma liga.
    
    Útil para testes.
    """
    logger.info(f"Iniciando scrape: {league}")
    
    async with BetinAsiaScraper() as scraper:
        # Login
        if not await scraper.login():
            logger.error("Falha no login")
            return
            
        # Scrape
        matches = await scraper.scrape_league(league)
        
        logger.info(f"Encontradas {len(matches)} partidas")
        
        for match in matches:
            # Salva a partida
            match_id = await db.save_match(match)
            
            # Processa cada linha de AH
            for line_str, ah_line in match.ah_lines.items():
                logger.info(f"  {match}: {ah_line}")
                
                # Salva odds no histórico
                for bk_name, bk_odds in ah_line.bookmaker_odds.items():
                    await db.save_odds(
                        match_id=match_id,
                        ah_line=line_str,
                        bookmaker=bk_name,
                        home_odds=bk_odds.home_odds,
                        away_odds=bk_odds.away_odds,
                    )


async def run_collection_mode(db: Database, cache: OddsCache):
    """
    Modo de coleta de dados.
    
    Faz scraping contínuo mas NÃO executa apostas.
    Usado para coletar dados para treinar o modelo.
    """
    logger.info("=" * 60)
    logger.info("MODO COLETA DE DADOS")
    logger.info("Apostas NÃO serão executadas")
    logger.info("=" * 60)
    
    # Lista de ligas para coletar
    leagues = [
        # Tier 1
        "England Premier League",
        "Germany Bundesliga",
        "Spain La Liga",
        "Italy Serie A",
        # Tier 2
        "England Championship",
        "Netherlands Eredivisie",
    ]
    
    async with BetinAsiaScraper() as scraper:
        # Login
        if not await scraper.login():
            logger.error("Falha no login. Encerrando.")
            return
            
        while True:
            try:
                for league in leagues:
                    try:
                        matches = await scraper.scrape_league(league)
                        
                        for match in matches:
                            # Salva no banco
                            match_id = await db.save_match(match)
                            
                            # Salva todas as odds
                            for line_str, ah_line in match.ah_lines.items():
                                for bk_name, bk_odds in ah_line.bookmaker_odds.items():
                                    await db.save_odds(
                                        match_id=match_id,
                                        ah_line=line_str,
                                        bookmaker=bk_name,
                                        home_odds=bk_odds.home_odds,
                                        away_odds=bk_odds.away_odds,
                                    )
                                    
                                # Detecta oportunidades (para registro)
                                if ah_line.num_bookmakers >= 3:
                                    best_bk, best_odds = ah_line.best_home_odds
                                    
                                    await db.save_opportunity(
                                        match_id=match_id,
                                        ah_line=line_str,
                                        side="home",
                                        best_odds=best_odds,
                                        best_bookmaker=best_bk,
                                        num_bookmakers=ah_line.num_bookmakers,
                                        dif_pct_best_second=ah_line.get_dif_best_second_home(),
                                        dif_pct_best_median=ah_line.get_dif_best_median_home(),
                                        minutes_to_kickoff=match.minutes_to_kickoff,
                                    )
                                    
                        logger.info(f"[{league}] {len(matches)} partidas coletadas")
                        
                    except Exception as e:
                        logger.error(f"Erro ao coletar {league}: {e}")
                        
                    # Delay entre ligas
                    await asyncio.sleep(5)
                    
                # Delay entre ciclos completos
                logger.info("Ciclo completo. Aguardando 60 segundos...")
                await asyncio.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("Interrompido pelo usuário")
                break
            except Exception as e:
                logger.error(f"Erro no ciclo de coleta: {e}")
                await asyncio.sleep(30)


async def run_production_mode(db: Database, cache: OddsCache):
    """
    Modo de produção.
    
    Scraping + Scoring + Execução de apostas.
    """
    logger.info("=" * 60)
    logger.info("MODO PRODUÇÃO")
    if settings.dry_run:
        logger.warning("DRY RUN ATIVO - Apostas simuladas")
    else:
        logger.warning("APOSTAS REAIS ATIVADAS")
    logger.info("=" * 60)
    
    # Inicia o scheduler
    scheduler = start_scheduler(db, cache)
    
    try:
        # Mantém o programa rodando
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Encerrando...")
        scheduler.shutdown()


async def main(args):
    """Função principal."""
    
    setup_logging()
    
    try:
        # Inicializa componentes
        db, cache = await initialize()
        
        if args.test:
            # Modo teste: scrape uma liga
            await run_single_scrape("England Premier League", db, cache)
            
        elif args.collect:
            # Modo coleta de dados
            await run_collection_mode(db, cache)
            
        else:
            # Modo produção
            await run_production_mode(db, cache)
            
    except KeyboardInterrupt:
        logger.info("Programa interrompido pelo usuário")
    except Exception as e:
        logger.exception(f"Erro fatal: {e}")
        raise
    finally:
        # Cleanup
        if 'db' in locals():
            await db.close()
        if 'cache' in locals():
            await cache.close()
            
        logger.info("BetinAsia Bot encerrado")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BetinAsia Bot")
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Não executa apostas reais (apenas simula)",
    )
    
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Modo coleta de dados (sem scoring/execução)",
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Modo teste (scrape único)",
    )
    
    args = parser.parse_args()
    
    # Override dry_run se passado na linha de comando
    if args.dry_run:
        settings.dry_run = True
        
    asyncio.run(main(args))
