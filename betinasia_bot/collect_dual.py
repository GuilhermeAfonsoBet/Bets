#!/usr/bin/env python3
"""
Coleta dual: Best odds (rápida) + Bookmakers (profunda) em paralelo.

Estratégia A: Coleta profunda com bookmakers (~3 min/jogo)
- Dados: best/2nd best odds, bookmakers, mediana, num casas
- Tabela: odds_history

Estratégia B: Coleta rápida de best odds (~5 seg/jogo)
- Dados: best odds home/away extraídas do DOM
- Tabela: best_odds_history
"""

import asyncio
import re
from datetime import datetime, timezone
from playwright.async_api import async_playwright
from loguru import logger

from storage.database import Database
from scraper.betinasia import BetinAsiaScraper
from config import settings

# Configurações
MIN_STAKE_THRESHOLD = 20  # Pular linhas com stake máximo <= $20
FAST_SCAN_INTERVAL = 60   # Intervalo entre varreduras rápidas (segundos)
DEEP_SCAN_INTERVAL = 300  # Intervalo entre varreduras profundas (segundos)

# Ligas para coleta
LEAGUES = [
    "England Premier League",
    "Germany Bundesliga",
    "Spain La Liga",
    "Italy Serie A",
    "France Ligue 1",
]


async def extract_best_odds_from_dom(page) -> list:
    """
    Extrai best odds diretamente do DOM (sem cliques).
    Retorna lista de dicts com ah_line, home_odds, away_odds.
    """
    data = await page.evaluate("""
        () => {
            const result = [];
            const pageText = document.body.innerText;
            
            // Padrão: HANDICAP Home ODDS Away ODDS
            const ahPattern = /([+-]?\\d+(?:[.,]\\d+)?)[\\s\\n]+Home[\\s\\n]+(\\d+[.,]\\d+)[\\s\\n]+Away[\\s\\n]+(\\d+[.,]\\d+)/g;
            
            let match;
            while ((match = ahPattern.exec(pageText)) !== null) {
                result.push({
                    ah_line: match[1],
                    home_odds: parseFloat(match[2].replace(',', '.')),
                    away_odds: parseFloat(match[3].replace(',', '.'))
                });
            }
            
            return result;
        }
    """)
    return data


async def fast_scan_worker(context, db: Database, stop_event: asyncio.Event):
    """
    Worker de varredura rápida - extrai best odds do DOM.
    """
    page = await context.new_page()
    logger.info("[FAST] Worker de varredura rápida iniciado")
    
    cycle = 0
    while not stop_event.is_set():
        cycle += 1
        cycle_start = datetime.now(timezone.utc)
        logger.info(f"[FAST] === Ciclo {cycle} iniciado ===")
        
        total_matches = 0
        total_odds = 0
        
        for league in LEAGUES:
            try:
                # Obtém código da liga
                league_code = BetinAsiaScraper.LEAGUE_CODES.get(league)
                if not league_code:
                    continue
                
                # Navega para a página da liga
                url = f"https://black.betinasia.com/sportsbook/football/{league_code}"
                await page.goto(url)
                await page.wait_for_load_state("networkidle")
                await page.wait_for_timeout(2000)
                
                # Encontra URLs dos jogos
                game_urls = []
                links = await page.query_selector_all("a")
                for link in links:
                    href = await link.get_attribute("href")
                    if href and "/sportsbook/football/" in href and "," in href:
                        if league_code in href:
                            full_url = f"https://black.betinasia.com{href.split('?')[0]}"
                            if full_url not in game_urls:
                                game_urls.append(full_url)
                
                logger.debug(f"[FAST] {league}: {len(game_urls)} jogos encontrados")
                
                # Processa cada jogo
                for game_url in game_urls[:20]:  # Limite de 20 jogos por liga
                    try:
                        await page.goto(game_url)
                        await page.wait_for_load_state("networkidle")
                        await page.wait_for_timeout(1500)
                        
                        # Expande linhas AH
                        for _ in range(3):
                            btns = await page.query_selector_all("text='Show all lines'")
                            for btn in btns:
                                try:
                                    if await btn.is_visible():
                                        await btn.click()
                                        await page.wait_for_timeout(500)
                                except:
                                    pass
                        
                        # Extrai dados do DOM
                        ah_lines = await extract_best_odds_from_dom(page)
                        
                        if ah_lines:
                            # Extrai match_id da URL
                            url_match = re.search(r'/(\d{4}-\d{2}-\d{2}),(\d+),(\d+)', game_url)
                            if url_match:
                                external_id = f"match_{url_match.group(2)}_{url_match.group(3)}"
                                
                                # Busca ou cria partida no banco
                                from scraper.models import MatchData
                                match_data = MatchData(
                                    match_id=external_id,
                                    league=league,
                                    home_team="TBD",
                                    away_team="TBD",
                                    kickoff_time=datetime.now(timezone.utc),
                                )
                                match_id = await db.save_match(match_data)
                                
                                # Salva best odds
                                for line in ah_lines:
                                    await db.save_best_odds(
                                        match_id=match_id,
                                        ah_line=line['ah_line'],
                                        best_home_odds=line['home_odds'],
                                        best_away_odds=line['away_odds'],
                                    )
                                    total_odds += 1
                                
                                total_matches += 1
                        
                    except Exception as e:
                        logger.debug(f"[FAST] Erro em jogo: {e}")
                        continue
                
                logger.info(f"[FAST] [{league}] processada")
                
            except Exception as e:
                logger.error(f"[FAST] Erro em {league}: {e}")
        
        cycle_duration = (datetime.now(timezone.utc) - cycle_start).total_seconds()
        logger.info(f"[FAST] Ciclo {cycle} completo: {total_matches} jogos, {total_odds} odds em {cycle_duration:.0f}s")
        
        # Aguarda próximo ciclo
        await asyncio.sleep(FAST_SCAN_INTERVAL)
    
    await page.close()
    logger.info("[FAST] Worker encerrado")


async def deep_scan_worker(context, db: Database, stop_event: asyncio.Event):
    """
    Worker de varredura profunda - captura todos os bookmakers.
    """
    # Usa o scraper existente
    scraper = BetinAsiaScraper()
    scraper._context = context
    scraper._page = await context.new_page()
    scraper._logged_in = True
    
    logger.info("[DEEP] Worker de varredura profunda iniciado")
    
    cycle = 0
    while not stop_event.is_set():
        cycle += 1
        cycle_start = datetime.now(timezone.utc)
        logger.info(f"[DEEP] === Ciclo {cycle} iniciado ===")
        
        total_matches = 0
        total_odds = 0
        
        for league in LEAGUES:
            try:
                matches = await scraper.scrape_league(league)
                
                for match in matches:
                    match_id = await db.save_match(match)
                    
                    for line_str, ah_line in match.ah_lines.items():
                        # Métricas HOME
                        home_metrics = ah_line.get_metrics_summary("home")
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
                        total_odds += 1
                        
                        # Métricas AWAY
                        away_metrics = ah_line.get_metrics_summary("away")
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
                        total_odds += 1
                    
                    total_matches += 1
                
                if matches:
                    logger.info(f"[DEEP] [{league}] {len(matches)} jogos processados")
                
            except Exception as e:
                logger.error(f"[DEEP] Erro em {league}: {e}")
            
            await asyncio.sleep(10)  # Delay entre ligas
        
        cycle_duration = (datetime.now(timezone.utc) - cycle_start).total_seconds()
        logger.info(f"[DEEP] Ciclo {cycle} completo: {total_matches} jogos, {total_odds} odds em {cycle_duration:.0f}s")
        
        # Aguarda próximo ciclo
        await asyncio.sleep(DEEP_SCAN_INTERVAL)
    
    await scraper._page.close()
    logger.info("[DEEP] Worker encerrado")


async def main():
    """
    Executa coleta dual com duas abas.
    """
    logger.info("="*60)
    logger.info("BETINASIA BOT - Coleta Dual")
    logger.info("="*60)
    logger.info("Estratégia A: Coleta profunda (bookmakers)")
    logger.info("Estratégia B: Coleta rápida (best odds)")
    logger.info("="*60)
    
    # Inicializa banco de dados
    db = Database()
    await db.connect()
    logger.info("Banco de dados conectado")
    
    # Inicializa browser
    p = await async_playwright().start()
    browser = await p.chromium.launch(headless=True)
    context = await browser.new_context(
        storage_state="betinasia_session.json",
        viewport={"width": 1920, "height": 1080},
    )
    logger.info("Browser iniciado com sessão salva")
    
    # Evento para parar workers
    stop_event = asyncio.Event()
    
    try:
        # Inicia workers em paralelo
        fast_task = asyncio.create_task(fast_scan_worker(context, db, stop_event))
        deep_task = asyncio.create_task(deep_scan_worker(context, db, stop_event))
        
        logger.info("Workers iniciados - Ctrl+C para parar")
        
        # Aguarda indefinidamente
        await asyncio.gather(fast_task, deep_task)
        
    except KeyboardInterrupt:
        logger.info("Interrupção recebida, encerrando workers...")
        stop_event.set()
        await asyncio.sleep(2)
    
    finally:
        await browser.close()
        await p.stop()
        await db.close()
        logger.info("Bot encerrado")


if __name__ == "__main__":
    asyncio.run(main())
