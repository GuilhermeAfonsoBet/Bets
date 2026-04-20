#!/usr/bin/env python3
"""
Coleta paralela com múltiplas abas.

Exemplo com 5 abas profundas:
- Se 1 aba leva ~3 min/jogo
- Com 5 abas paralelas = ~36 seg/jogo (efetivo)
- 100 jogos: de 5h → 1h
"""

import asyncio
import re
from datetime import datetime, timezone
from playwright.async_api import async_playwright, Page
from loguru import logger

from storage.database import Database
from scraper.betinasia import BetinAsiaScraper
from config import settings

# Configuração de paralelismo
NUM_DEEP_TABS = 4  # Número de abas para coleta profunda
NUM_FAST_TABS = 1  # Número de abas para coleta rápida

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
        
        # Scroll para carregar todos os jogos
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
        
        logger.info(f"[INDEX] {league}: {len([g for g in all_games if g['league'] == league])} jogos")
    
    return all_games


async def deep_worker(worker_id: int, page: Page, job_queue: asyncio.Queue, db: Database, results: dict):
    """
    Worker de coleta profunda - processa jogos da fila.
    """
    logger.info(f"[DEEP-{worker_id}] Worker iniciado")
    
    processed = 0
    while True:
        try:
            # Pega próximo jogo da fila (timeout de 5s)
            game = await asyncio.wait_for(job_queue.get(), timeout=5.0)
        except asyncio.TimeoutError:
            # Fila vazia por 5s = terminou
            break
        
        try:
            game_url = game['url']
            league = game['league']
            
            logger.debug(f"[DEEP-{worker_id}] Processando: {game_url}")
            
            # Navega para o jogo
            await page.goto(game_url)
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(2000)
            
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
            
            # Extrai título do jogo
            title = await page.title()
            teams = title.split(" - ")[0] if " - " in title else "Unknown"
            parts = teams.split(" vs ") if " vs " in teams else [teams, "Unknown"]
            home_team = parts[0].strip()
            away_team = parts[1].strip() if len(parts) > 1 else "Unknown"
            
            # Extrai match_id da URL
            url_match = re.search(r'/(\d{4}-\d{2}-\d{2}),(\d+),(\d+)', game_url)
            if url_match:
                external_id = f"{url_match.group(1)}_{url_match.group(2)}_{url_match.group(3)}"
                kickoff_str = url_match.group(1)
                kickoff = datetime.strptime(kickoff_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            else:
                external_id = f"game_{hash(game_url)}"
                kickoff = datetime.now(timezone.utc)
            
            # Salva partida no banco
            from scraper.models import MatchData
            match_data = MatchData(
                match_id=external_id,
                league=league,
                home_team=home_team,
                away_team=away_team,
                kickoff_time=kickoff,
            )
            match_id = await db.save_match(match_data)
            
            # Extrai linhas AH do DOM primeiro
            ah_lines = await extract_best_odds_from_dom(page)
            
            # Salva best odds
            for line in ah_lines:
                await db.save_best_odds(
                    match_id=match_id,
                    ah_line=line['ah_line'],
                    best_home_odds=line['home_odds'],
                    best_away_odds=line['away_odds'],
                )
            
            # Agora captura bookmakers para cada linha
            odds_saved = 0
            for line in ah_lines:
                handicap = line['ah_line']
                home_odds_val = line['home_odds']
                
                # Encontra e clica no elemento de odds
                try:
                    # Procura o elemento com a odd específica
                    odds_selector = f"span:has-text('{home_odds_val}')"
                    odds_elements = await page.query_selector_all(odds_selector)
                    
                    clicked = False
                    for elem in odds_elements:
                        try:
                            # Verifica contexto (deve conter o handicap)
                            parent = await elem.evaluate_handle("el => el.closest('div')")
                            parent_text = await parent.evaluate("el => el.innerText")
                            
                            if handicap in parent_text.replace(',', '.'):
                                # Scroll e clique
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
                        # Extrai bookmakers do painel
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
                        
                        # Parse bookmakers
                        bk_pattern = r'(3et|4casters|bdaq|bf|pin88|mbook|sbo|sharp|sing)\s+(\d+[.,]\d+)\s+\$?([\d,]+)'
                        bk_matches = re.findall(bk_pattern, panel_text, re.IGNORECASE)
                        
                        if bk_matches:
                            # Calcula métricas
                            odds_list = [float(m[1].replace(',', '.')) for m in bk_matches]
                            odds_list.sort(reverse=True)
                            
                            best_odds = odds_list[0] if odds_list else 0
                            second_best = odds_list[1] if len(odds_list) > 1 else 0
                            median = odds_list[len(odds_list)//2] if odds_list else 0
                            
                            best_bk = bk_matches[0][0] if bk_matches else "unknown"
                            second_bk = bk_matches[1][0] if len(bk_matches) > 1 else ""
                            
                            # Salva odds com métricas
                            await db.save_odds(
                                match_id=match_id,
                                ah_line=handicap,
                                side="home",
                                best_odds=best_odds,
                                best_bookmaker=best_bk,
                                second_best_odds=second_best,
                                second_best_bookmaker=second_bk,
                                median_odds=median,
                                num_bookmakers=len(bk_matches),
                            )
                            odds_saved += 1
                        
                        # Fecha painel (ESC)
                        await page.keyboard.press("Escape")
                        await page.wait_for_timeout(300)
                
                except Exception as e:
                    logger.debug(f"[DEEP-{worker_id}] Erro capturando bookmakers: {e}")
            
            processed += 1
            results[worker_id] = processed
            
            logger.info(f"[DEEP-{worker_id}] ✓ {home_team} vs {away_team}: {len(ah_lines)} AH, {odds_saved} bks")
            
            job_queue.task_done()
            
        except Exception as e:
            logger.error(f"[DEEP-{worker_id}] Erro: {e}")
            job_queue.task_done()
    
    logger.info(f"[DEEP-{worker_id}] Worker encerrado - {processed} jogos processados")


async def main():
    """
    Executa coleta paralela com múltiplas abas.
    """
    logger.info("="*60)
    logger.info(f"BETINASIA BOT - Coleta Paralela ({NUM_DEEP_TABS} abas)")
    logger.info("="*60)
    
    # Inicializa banco
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
    logger.info("Browser iniciado")
    
    try:
        # Cria aba índice para coletar URLs
        index_page = await context.new_page()
        
        # Coleta todas as URLs de jogos
        logger.info("[INDEX] Coletando URLs de todos os jogos...")
        all_games = await get_all_game_urls(index_page, LEAGUES)
        await index_page.close()
        
        logger.info(f"[INDEX] Total: {len(all_games)} jogos para processar")
        
        if not all_games:
            logger.error("Nenhum jogo encontrado!")
            return
        
        # Cria fila de jobs
        job_queue = asyncio.Queue()
        for game in all_games:
            await job_queue.put(game)
        
        # Cria abas e workers
        pages = []
        results = {}
        
        for i in range(NUM_DEEP_TABS):
            page = await context.new_page()
            pages.append(page)
            results[i] = 0
        
        logger.info(f"Iniciando {NUM_DEEP_TABS} workers paralelos...")
        start_time = datetime.now()
        
        # Inicia workers
        tasks = [
            asyncio.create_task(deep_worker(i, pages[i], job_queue, db, results))
            for i in range(NUM_DEEP_TABS)
        ]
        
        # Aguarda conclusão
        await asyncio.gather(*tasks)
        
        # Estatísticas
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        total_processed = sum(results.values())
        
        logger.info("="*60)
        logger.info("COLETA CONCLUÍDA")
        logger.info("="*60)
        logger.info(f"Jogos processados: {total_processed}")
        logger.info(f"Tempo total: {duration:.0f} segundos ({duration/60:.1f} min)")
        logger.info(f"Média: {duration/total_processed:.1f} seg/jogo")
        logger.info(f"Workers: {NUM_DEEP_TABS} abas paralelas")
        
        # Comparação
        estimated_serial = total_processed * 180  # ~3 min/jogo serial
        speedup = estimated_serial / duration if duration > 0 else 1
        logger.info(f"Speedup vs serial: {speedup:.1f}x mais rápido")
        
    except KeyboardInterrupt:
        logger.info("Interrupção recebida...")
    
    finally:
        for page in pages:
            await page.close()
        await browser.close()
        await p.stop()
        await db.close()
        logger.info("Bot encerrado")


if __name__ == "__main__":
    asyncio.run(main())
