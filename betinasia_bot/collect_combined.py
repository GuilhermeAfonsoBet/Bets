#!/usr/bin/env python3
"""
Coleta combinada: 1 aba rápida + N abas profundas em paralelo.

Arquitetura:
- ABA 1: Coleta rápida contínua (best odds do DOM, ~5 seg/jogo)
- ABAS 2-4: Coleta profunda paralela (bookmakers, ~3 min/jogo cada)

Resultado:
- Best odds atualizadas a cada ~60-90 segundos
- Coleta profunda 3x mais rápida que serial
"""

import asyncio
import re
from datetime import datetime, timezone
from playwright.async_api import async_playwright, Page
from loguru import logger

from storage.database import Database
from scraper.betinasia import BetinAsiaScraper
from scraper.models import MatchData

# Configuração
NUM_DEEP_TABS = 3  # Abas para coleta profunda
FAST_CYCLE_DELAY = 30  # Segundos entre ciclos rápidos

# Ligas
LEAGUES = [
    "England Premier League",
    "Germany Bundesliga", 
    "Spain La Liga",
    "Italy Serie A",
    "France Ligue 1",
]


async def extract_best_odds_from_dom(page: Page) -> list:
    """Extrai best odds do DOM."""
    return await page.evaluate("""
        () => {
            const result = [];
            const pageText = document.body.innerText;
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


async def get_all_game_urls(page: Page, leagues: list) -> list:
    """Coleta todas as URLs de jogos de todas as ligas."""
    all_games = []
    
    for league in leagues:
        league_code = BetinAsiaScraper.LEAGUE_CODES.get(league)
        if not league_code:
            continue
            
        url = f"https://black.betinasia.com/sportsbook/football/{league_code}"
        await page.goto(url)
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(2000)
        
        # Scroll para carregar todos
        for _ in range(5):
            await page.evaluate("window.scrollBy(0, 1000)")
            await page.wait_for_timeout(300)
        
        # Coleta URLs
        links = await page.query_selector_all("a")
        for link in links:
            href = await link.get_attribute("href")
            if href and "/sportsbook/football/" in href and "," in href:
                if league_code in href:
                    full_url = f"https://black.betinasia.com{href.split('?')[0]}"
                    if full_url not in [g['url'] for g in all_games]:
                        all_games.append({
                            'url': full_url,
                            'league': league,
                        })
        
        logger.debug(f"[INDEX] {league}: {len([g for g in all_games if g['league'] == league])} jogos")
    
    return all_games


async def fast_worker(page: Page, db: Database, stop_event: asyncio.Event, stats: dict):
    """
    Worker de coleta rápida - extrai best odds do DOM continuamente.
    """
    logger.info("[FAST] Worker iniciado - coleta rápida de best odds")
    
    cycle = 0
    while not stop_event.is_set():
        cycle += 1
        cycle_start = datetime.now(timezone.utc)
        total_odds = 0
        total_games = 0
        
        for league in LEAGUES:
            if stop_event.is_set():
                break
                
            try:
                league_code = BetinAsiaScraper.LEAGUE_CODES.get(league)
                if not league_code:
                    continue
                
                # Navega para liga
                url = f"https://black.betinasia.com/sportsbook/football/{league_code}"
                await page.goto(url)
                await page.wait_for_load_state("networkidle")
                await page.wait_for_timeout(1500)
                
                # Scroll
                for _ in range(3):
                    await page.evaluate("window.scrollBy(0, 800)")
                    await page.wait_for_timeout(200)
                
                # Coleta URLs da liga
                game_urls = []
                links = await page.query_selector_all("a")
                for link in links:
                    href = await link.get_attribute("href")
                    if href and "/sportsbook/football/" in href and "," in href:
                        if league_code in href:
                            full_url = f"https://black.betinasia.com{href.split('?')[0]}"
                            if full_url not in game_urls:
                                game_urls.append(full_url)
                
                # Processa cada jogo rapidamente
                for game_url in game_urls[:25]:  # Limite 25 jogos/liga
                    if stop_event.is_set():
                        break
                        
                    try:
                        await page.goto(game_url)
                        await page.wait_for_load_state("networkidle")
                        await page.wait_for_timeout(1000)
                        
                        # Expande AH
                        btns = await page.query_selector_all("text='Show all lines'")
                        for btn in btns[:2]:
                            try:
                                if await btn.is_visible():
                                    await btn.click()
                                    await page.wait_for_timeout(400)
                            except:
                                pass
                        
                        # Extrai dados
                        ah_lines = await extract_best_odds_from_dom(page)
                        
                        if ah_lines:
                            # Extrai info do jogo
                            url_match = re.search(r'/(\d{4}-\d{2}-\d{2}),(\d+),(\d+)', game_url)
                            if url_match:
                                external_id = f"{url_match.group(1)}_{url_match.group(2)}_{url_match.group(3)}"
                                kickoff = datetime.strptime(url_match.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
                            else:
                                external_id = f"game_{hash(game_url) % 100000}"
                                kickoff = datetime.now(timezone.utc)
                            
                            # Extrai times do título
                            title = await page.title()
                            if " vs " in title:
                                parts = title.split(" - ")[0].split(" vs ")
                                home_team = parts[0].strip()
                                away_team = parts[1].strip() if len(parts) > 1 else "TBD"
                            else:
                                home_team = "TBD"
                                away_team = "TBD"
                            
                            # Salva partida
                            match_data = MatchData(
                                match_id=external_id,
                                league=league,
                                home_team=home_team,
                                away_team=away_team,
                                kickoff_time=kickoff,
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
                            
                            total_games += 1
                    
                    except Exception as e:
                        logger.debug(f"[FAST] Erro em jogo: {e}")
                        continue
                
            except Exception as e:
                logger.debug(f"[FAST] Erro em {league}: {e}")
        
        duration = (datetime.now(timezone.utc) - cycle_start).total_seconds()
        stats['fast_cycles'] = cycle
        stats['fast_odds'] = stats.get('fast_odds', 0) + total_odds
        
        logger.info(f"[FAST] Ciclo {cycle}: {total_games} jogos, {total_odds} odds em {duration:.0f}s")
        
        # Aguarda próximo ciclo
        await asyncio.sleep(FAST_CYCLE_DELAY)
    
    logger.info("[FAST] Worker encerrado")


async def deep_worker(worker_id: int, page: Page, job_queue: asyncio.Queue, db: Database, stats: dict):
    """
    Worker de coleta profunda - captura bookmakers.
    """
    logger.info(f"[DEEP-{worker_id}] Worker iniciado")
    
    processed = 0
    while True:
        try:
            game = await asyncio.wait_for(job_queue.get(), timeout=10.0)
        except asyncio.TimeoutError:
            # Verifica se ainda há jobs
            if job_queue.empty():
                break
            continue
        
        try:
            game_url = game['url']
            league = game['league']
            
            # Navega
            await page.goto(game_url)
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(2000)
            
            # Expande AH
            for _ in range(3):
                btns = await page.query_selector_all("text='Show all lines'")
                for btn in btns:
                    try:
                        if await btn.is_visible():
                            await btn.click()
                            await page.wait_for_timeout(500)
                    except:
                        pass
            
            # Extrai info
            title = await page.title()
            if " vs " in title:
                parts = title.split(" - ")[0].split(" vs ")
                home_team = parts[0].strip()
                away_team = parts[1].strip() if len(parts) > 1 else "TBD"
            else:
                home_team = "TBD"
                away_team = "TBD"
            
            url_match = re.search(r'/(\d{4}-\d{2}-\d{2}),(\d+),(\d+)', game_url)
            if url_match:
                external_id = f"{url_match.group(1)}_{url_match.group(2)}_{url_match.group(3)}"
                kickoff = datetime.strptime(url_match.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            else:
                external_id = f"game_{hash(game_url) % 100000}"
                kickoff = datetime.now(timezone.utc)
            
            # Salva partida
            match_data = MatchData(
                match_id=external_id,
                league=league,
                home_team=home_team,
                away_team=away_team,
                kickoff_time=kickoff,
            )
            match_id = await db.save_match(match_data)
            
            # Extrai AH do DOM
            ah_lines = await extract_best_odds_from_dom(page)
            
            # Captura bookmakers para cada linha
            odds_saved = 0
            for line in ah_lines:
                handicap = line['ah_line']
                home_odds_val = str(line['home_odds'])
                
                try:
                    # Encontra elemento de odds
                    elements = await page.query_selector_all(f"span:has-text('{home_odds_val}')")
                    
                    clicked = False
                    for elem in elements:
                        try:
                            parent = await elem.evaluate_handle("el => el.closest('div')")
                            parent_text = await parent.evaluate("el => el.innerText")
                            
                            if handicap.replace(',', '.') in parent_text.replace(',', '.'):
                                await elem.scroll_into_view_if_needed()
                                await page.wait_for_timeout(200)
                                
                                parent_div = await elem.evaluate_handle("el => el.parentElement")
                                await parent_div.click()
                                await page.wait_for_timeout(1500)
                                
                                clicked = True
                                break
                        except:
                            continue
                    
                    if clicked:
                        # Extrai bookmakers
                        panel_text = await page.evaluate("""
                            () => {
                                const panels = document.querySelectorAll('[class*="panel"], [class*="sidebar"], [class*="drawer"]');
                                for (const panel of panels) {
                                    if (panel.innerText.includes('$') && 
                                        (panel.innerText.includes('sbo') || panel.innerText.includes('pin') || 
                                         panel.innerText.includes('bf') || panel.innerText.includes('3et'))) {
                                        return panel.innerText;
                                    }
                                }
                                return document.body.innerText;
                            }
                        """)
                        
                        # Parse
                        bk_pattern = r'(3et|4casters|bdaq|bf|pin88|mbook|sbo|sharp|sing)\s+(\d+[.,]\d+)\s+\$?([\d,]+)'
                        bk_matches = re.findall(bk_pattern, panel_text, re.IGNORECASE)
                        
                        if bk_matches:
                            odds_list = sorted([float(m[1].replace(',', '.')) for m in bk_matches], reverse=True)
                            
                            await db.save_odds(
                                match_id=match_id,
                                ah_line=handicap,
                                side="home",
                                best_odds=odds_list[0] if odds_list else 0,
                                best_bookmaker=bk_matches[0][0],
                                second_best_odds=odds_list[1] if len(odds_list) > 1 else 0,
                                second_best_bookmaker=bk_matches[1][0] if len(bk_matches) > 1 else "",
                                median_odds=odds_list[len(odds_list)//2] if odds_list else 0,
                                num_bookmakers=len(bk_matches),
                            )
                            odds_saved += 1
                        
                        await page.keyboard.press("Escape")
                        await page.wait_for_timeout(300)
                
                except Exception as e:
                    logger.debug(f"[DEEP-{worker_id}] Erro em linha {handicap}: {e}")
            
            processed += 1
            stats[f'deep_{worker_id}'] = processed
            
            logger.info(f"[DEEP-{worker_id}] ✓ {home_team} vs {away_team}: {odds_saved} bookmaker lines")
            
            job_queue.task_done()
            
        except Exception as e:
            logger.error(f"[DEEP-{worker_id}] Erro: {e}")
            job_queue.task_done()
    
    logger.info(f"[DEEP-{worker_id}] Worker encerrado - {processed} jogos")


async def main():
    """
    Executa coleta combinada.
    """
    logger.info("="*60)
    logger.info("BETINASIA BOT - Coleta Combinada")
    logger.info("="*60)
    logger.info(f"  1 aba RÁPIDA (best odds contínuo)")
    logger.info(f"  {NUM_DEEP_TABS} abas PROFUNDAS (bookmakers paralelo)")
    logger.info("="*60)
    
    # Banco de dados
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
    stop_event = asyncio.Event()
    
    try:
        # Cria aba índice
        index_page = await context.new_page()
        
        # Coleta URLs
        logger.info("[INDEX] Coletando URLs de jogos...")
        all_games = await get_all_game_urls(index_page, LEAGUES)
        await index_page.close()
        
        logger.info(f"[INDEX] Total: {len(all_games)} jogos")
        
        if not all_games:
            logger.error("Nenhum jogo encontrado!")
            return
        
        # Fila para workers profundos
        job_queue = asyncio.Queue()
        for game in all_games:
            await job_queue.put(game)
        
        # Cria abas
        fast_page = await context.new_page()
        deep_pages = [await context.new_page() for _ in range(NUM_DEEP_TABS)]
        
        start_time = datetime.now()
        logger.info("Iniciando workers...")
        
        # Inicia tasks
        tasks = [
            asyncio.create_task(fast_worker(fast_page, db, stop_event, stats))
        ]
        for i, page in enumerate(deep_pages):
            tasks.append(asyncio.create_task(deep_worker(i, page, job_queue, db, stats)))
        
        # Aguarda workers profundos terminarem
        deep_tasks = tasks[1:]
        await asyncio.gather(*deep_tasks)
        
        # Para worker rápido
        stop_event.set()
        await asyncio.sleep(2)
        
        # Estatísticas
        duration = (datetime.now() - start_time).total_seconds()
        deep_total = sum(v for k, v in stats.items() if k.startswith('deep_'))
        
        logger.info("="*60)
        logger.info("COLETA CONCLUÍDA")
        logger.info("="*60)
        logger.info(f"Tempo total: {duration:.0f}s ({duration/60:.1f} min)")
        logger.info(f"[FAST] Ciclos: {stats.get('fast_cycles', 0)}, Odds: {stats.get('fast_odds', 0)}")
        logger.info(f"[DEEP] Jogos: {deep_total}")
        if deep_total > 0:
            logger.info(f"[DEEP] Média: {duration/deep_total:.1f} seg/jogo")
        
    except KeyboardInterrupt:
        logger.info("Interrupção recebida...")
        stop_event.set()
    
    finally:
        await fast_page.close()
        for page in deep_pages:
            await page.close()
        await browser.close()
        await p.stop()
        await db.close()
        logger.info("Bot encerrado")


if __name__ == "__main__":
    asyncio.run(main())
