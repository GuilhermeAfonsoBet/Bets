#!/usr/bin/env python3
"""
Coleta com múltiplas abas usando o scraper original (que funciona).
"""

import asyncio
from datetime import datetime, timezone
from playwright.async_api import async_playwright
from loguru import logger

from storage.database import Database
from scraper.betinasia import BetinAsiaScraper

# Configuração
NUM_WORKERS = 3

# Ligas
LEAGUES = [
    "England Premier League",
    "Germany Bundesliga", 
    "Spain La Liga",
    "Italy Serie A",
    "France Ligue 1",
]


async def scraper_worker(worker_id: int, context, job_queue: asyncio.Queue, db: Database, stats: dict):
    """
    Worker que usa o scraper original.
    """
    logger.info(f"[WORKER-{worker_id}] Iniciado")
    
    # Cria instância do scraper com o contexto compartilhado
    scraper = BetinAsiaScraper()
    scraper._context = context
    scraper._page = await context.new_page()
    scraper._logged_in = True
    scraper._playwright = None  # Não gerencia o playwright
    scraper._browser = None  # Não gerencia o browser
    
    processed = 0
    total_odds = 0
    
    while True:
        try:
            job = await asyncio.wait_for(job_queue.get(), timeout=30.0)
        except asyncio.TimeoutError:
            if job_queue.empty():
                break
            continue
        
        try:
            league = job['league']
            game_url = job['url']
            
            # Usa o método do scraper para processar o jogo
            match = await scraper._scrape_single_match(game_url, league, capture_bookmakers=True)
            
            if match:
                # Salva no banco
                match_id = await db.save_match(match)
                
                odds_count = 0
                for line_str, ah_line in match.ah_lines.items():
                    # Salva métricas HOME
                    home_metrics = ah_line.get_metrics_summary("home")
                    if home_metrics["maior_odd"] > 0:
                        await db.save_odds(
                            match_id=match_id,
                            ah_line=line_str,
                            side="home",
                            best_odds=home_metrics["maior_odd"],
                            best_bookmaker=home_metrics["casa_maior_odd"],
                            second_best_odds=home_metrics["segunda_maior_odd"],
                            second_best_bookmaker=home_metrics["casa_segunda_maior"],
                            median_odds=home_metrics["odd_mediana"],
                            num_bookmakers=home_metrics["num_casas"],
                        )
                        odds_count += 1
                    
                    # Salva métricas AWAY
                    away_metrics = ah_line.get_metrics_summary("away")
                    if away_metrics["maior_odd"] > 0:
                        await db.save_odds(
                            match_id=match_id,
                            ah_line=line_str,
                            side="away",
                            best_odds=away_metrics["maior_odd"],
                            best_bookmaker=away_metrics["casa_maior_odd"],
                            second_best_odds=away_metrics["segunda_maior_odd"],
                            second_best_bookmaker=away_metrics["casa_segunda_maior"],
                            median_odds=away_metrics["odd_mediana"],
                            num_bookmakers=away_metrics["num_casas"],
                        )
                        odds_count += 1
                
                processed += 1
                total_odds += odds_count
                stats[f'worker_{worker_id}'] = processed
                
                logger.info(f"[WORKER-{worker_id}] ✓ {match.home_team} vs {match.away_team}: {len(match.ah_lines)} AH, {odds_count} odds")
            else:
                logger.warning(f"[WORKER-{worker_id}] Falhou: {game_url[:60]}")
            
            job_queue.task_done()
            
        except Exception as e:
            logger.error(f"[WORKER-{worker_id}] Erro: {str(e)[:80]}")
            job_queue.task_done()
    
    # Fecha página do worker
    await scraper._page.close()
    logger.info(f"[WORKER-{worker_id}] Encerrado: {processed} jogos, {total_odds} odds")


async def main():
    """Executa coleta com múltiplos workers."""
    logger.info("="*60)
    logger.info(f"BETINASIA BOT - Coleta Multi ({NUM_WORKERS} workers)")
    logger.info("="*60)
    
    # Banco
    db = Database()
    await db.connect()
    logger.info("Banco de dados conectado")
    
    # Browser
    p = await async_playwright().start()
    browser = await p.chromium.launch(headless=True)
    context = await browser.new_context(
        storage_state="betinasia_session.json",
        viewport={"width": 1920, "height": 1080},
    )
    logger.info("Browser iniciado")
    
    stats = {}
    
    try:
        # Coleta URLs usando scraper
        scraper = BetinAsiaScraper()
        scraper._context = context
        scraper._page = await context.new_page()
        scraper._logged_in = True
        
        # Coleta todas as URLs
        all_jobs = []
        for league in LEAGUES:
            logger.info(f"[INDEX] Coletando URLs: {league}")
            
            league_code = scraper.LEAGUE_CODES.get(league)
            if not league_code:
                continue
            
            url = f"https://black.betinasia.com/sportsbook/football/{league_code}"
            await scraper._page.goto(url)
            await scraper._page.wait_for_load_state("networkidle")
            await scraper._page.wait_for_timeout(3000)
            
            # Expande lista
            await scraper._expand_game_list()
            
            # Coleta URLs
            links = await scraper._page.query_selector_all("a")
            game_urls = []
            
            for link in links:
                href = await link.get_attribute("href")
                if href and "/sportsbook/football/" in href and "," in href:
                    if league_code in href:
                        full_url = f"https://black.betinasia.com{href.split('?')[0]}"
                        if full_url not in game_urls:
                            game_urls.append(full_url)
                            all_jobs.append({'league': league, 'url': full_url})
            
            logger.info(f"[INDEX] {league}: {len(game_urls)} jogos")
        
        await scraper._page.close()
        
        logger.info(f"[INDEX] Total: {len(all_jobs)} jogos")
        
        if not all_jobs:
            logger.error("Nenhum jogo encontrado!")
            return
        
        # Cria fila
        job_queue = asyncio.Queue()
        for job in all_jobs:
            await job_queue.put(job)
        
        start_time = datetime.now()
        logger.info(f"Iniciando {NUM_WORKERS} workers...")
        
        # Inicia workers
        tasks = [
            asyncio.create_task(scraper_worker(i, context, job_queue, db, stats))
            for i in range(NUM_WORKERS)
        ]
        
        await asyncio.gather(*tasks)
        
        # Estatísticas
        duration = (datetime.now() - start_time).total_seconds()
        total = sum(v for k, v in stats.items() if k.startswith('worker_'))
        
        logger.info("="*60)
        logger.info("COLETA CONCLUÍDA")
        logger.info(f"Jogos: {total}")
        logger.info(f"Tempo: {duration:.0f}s ({duration/60:.1f} min)")
        if total > 0:
            logger.info(f"Média: {duration/total:.1f}s/jogo")
            estimated_serial = total * 180  # ~3 min/jogo serial
            speedup = estimated_serial / duration if duration > 0 else 1
            logger.info(f"Speedup: {speedup:.1f}x mais rápido que serial")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.info("Interrompido pelo usuário")
    
    finally:
        await browser.close()
        await p.stop()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
