#!/usr/bin/env python3
"""
Coleta profunda apenas - foca em capturar bookmakers.
Versão simplificada e robusta.
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
NUM_TABS = 3  # Número de abas paralelas

# Ligas
LEAGUES = [
    "England Premier League",
    "Germany Bundesliga", 
    "Spain La Liga",
    "Italy Serie A",
    "France Ligue 1",
]


async def get_all_game_urls(page: Page, leagues: list) -> list:
    """Coleta todas as URLs de jogos."""
    all_games = []
    
    for league in leagues:
        league_code = BetinAsiaScraper.LEAGUE_CODES.get(league)
        if not league_code:
            continue
            
        url = f"https://black.betinasia.com/sportsbook/football/{league_code}"
        await page.goto(url)
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(3000)
        
        # Scroll
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


async def extract_ah_lines(page: Page) -> list:
    """Extrai linhas AH do DOM."""
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


async def extract_bookmakers(page: Page, handicap: str, target_odds: float) -> list:
    """Clica na odd e extrai bookmakers."""
    bookmakers = []
    
    try:
        # Procura elementos de odds usando seletores Playwright
        odds_str = f"{target_odds:.3f}"[:5]  # Ex: "1.923" -> "1.92"
        
        # Tenta encontrar e clicar usando Playwright
        clicked = False
        
        # Busca todos os spans com texto parecido com odds
        spans = await page.query_selector_all("span")
        
        for span in spans:
            try:
                text = await span.inner_text()
                text = text.strip()
                
                # Verifica se parece com uma odd
                if len(text) > 7 or len(text) < 3:
                    continue
                
                # Verifica se contém o valor da odd
                if odds_str[:4] not in text.replace(",", "."):
                    continue
                
                # Verifica contexto - o handicap deve estar próximo
                parent = await span.evaluate_handle("el => el.closest('div')")
                parent_text = await parent.evaluate("el => el.innerText || ''")
                
                # Normaliza handicap para comparação
                hcp_check = handicap.replace("+", "").replace(",", ".")
                if hcp_check not in parent_text.replace(",", "."):
                    continue
                
                # Encontrou! Faz scroll e clique
                await span.scroll_into_view_if_needed()
                await page.wait_for_timeout(200)
                
                # Clica no elemento pai (div que contém a odd)
                parent_elem = await span.evaluate_handle("el => el.parentElement")
                await parent_elem.click()
                await page.wait_for_timeout(2000)
                
                clicked = True
                break
                
            except Exception:
                continue
        
        if not clicked:
            return []
        
        await page.wait_for_timeout(2000)
        
        # Extrai texto do painel
        panel_text = await page.evaluate("""
            () => {
                // Procura paineis com dados de bookmaker
                const allDivs = document.querySelectorAll('div');
                for (const div of allDivs) {
                    const text = div.innerText;
                    if (text.includes('$') && text.length > 100 && text.length < 5000) {
                        if (/sbo|pin|3et|bf|bdaq/i.test(text)) {
                            return text;
                        }
                    }
                }
                return '';
            }
        """)
        
        if not panel_text:
            await page.keyboard.press("Escape")
            return []
        
        # Parse bookmakers
        patterns = [
            r'(3ete?)\s+(\d+[.,]\d+)\s+\$?([\d,]+)',
            r'(4casters?)\s+(\d+[.,]\d+)\s+\$?([\d,]+)',
            r'(bdaq)\s+(\d+[.,]\d+)\s+\$?([\d,]+)',
            r'(bf)\s+(\d+[.,]\d+)\s+\$?([\d,]+)',
            r'(pin88e?)\s+(\d+[.,]\d+)\s+\$?([\d,]+)',
            r'(mbook)\s+(\d+[.,]\d+)\s+\$?([\d,]+)',
            r'(sbo)\s+(\d+[.,]\d+)\s+\$?([\d,]+)',
            r'(sharp)\s+(\d+[.,]\d+)\s+\$?([\d,]+)',
            r'(sing)\s+(\d+[.,]\d+)\s+\$?([\d,]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, panel_text, re.IGNORECASE)
            for m in matches:
                bookmakers.append({
                    'name': m[0].lower(),
                    'odds': float(m[1].replace(",", ".")),
                    'limit': int(m[2].replace(",", "")) if m[2].replace(",", "").isdigit() else 0
                })
        
        # Fecha painel
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(300)
        
    except Exception as e:
        logger.debug(f"Erro extraindo bookmakers: {e}")
        try:
            await page.keyboard.press("Escape")
        except:
            pass
    
    return bookmakers


async def deep_worker(worker_id: int, page: Page, job_queue: asyncio.Queue, db: Database, stats: dict):
    """Worker de coleta profunda."""
    logger.info(f"[DEEP-{worker_id}] Worker iniciado")
    
    processed = 0
    total_bk_lines = 0
    
    while True:
        try:
            game = await asyncio.wait_for(job_queue.get(), timeout=15.0)
        except asyncio.TimeoutError:
            if job_queue.empty():
                break
            continue
        
        game_start = datetime.now()
        
        try:
            game_url = game['url']
            league = game['league']
            
            # Navega
            await page.goto(game_url, timeout=60000)
            await page.wait_for_load_state("networkidle", timeout=30000)
            await page.wait_for_timeout(3000)
            
            # Expande AH
            for _ in range(3):
                btns = await page.query_selector_all("text='Show all lines'")
                for btn in btns:
                    try:
                        if await btn.is_visible():
                            await btn.click()
                            await page.wait_for_timeout(800)
                    except:
                        pass
            
            # Extrai ID e dados da URL
            url_match = re.search(r'/(\d{4}-\d{2}-\d{2}),(\d+),(\d+)', game_url)
            if url_match:
                external_id = f"{url_match.group(1)}_{url_match.group(2)}_{url_match.group(3)}"
                kickoff = datetime.strptime(url_match.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            else:
                external_id = f"game_{hash(game_url) % 100000}"
                kickoff = datetime.now(timezone.utc)
            
            # Tenta extrair nomes dos times
            home_team = "Home"
            away_team = "Away"
            try:
                title = await page.title()
                if " vs " in title:
                    parts = title.split(" - ")[0].split(" vs ")
                    if len(parts) >= 2:
                        home_team = parts[0].strip()[:50]
                        away_team = parts[1].strip()[:50]
            except:
                pass
            
            # Salva partida
            match_data = MatchData(
                match_id=external_id,
                league=league,
                home_team=home_team,
                away_team=away_team,
                kickoff_time=kickoff,
            )
            match_id = await db.save_match(match_data)
            
            # Extrai linhas AH
            ah_lines = await extract_ah_lines(page)
            
            if not ah_lines:
                logger.warning(f"[DEEP-{worker_id}] Sem linhas AH: {game_url[:60]}")
                job_queue.task_done()
                continue
            
            # Processa até 10 linhas
            odds_saved = 0
            for line in ah_lines[:10]:
                handicap = line['ah_line']
                home_odds = line['home_odds']
                
                # Salva best odds primeiro
                await db.save_best_odds(
                    match_id=match_id,
                    ah_line=handicap,
                    best_home_odds=home_odds,
                    best_away_odds=line['away_odds'],
                )
                
                # Extrai bookmakers
                bookmakers = await extract_bookmakers(page, handicap, home_odds)
                
                if bookmakers:
                    bookmakers.sort(key=lambda x: x['odds'], reverse=True)
                    
                    best = bookmakers[0]
                    second = bookmakers[1] if len(bookmakers) > 1 else {'name': '', 'odds': 0}
                    odds_list = [b['odds'] for b in bookmakers]
                    median = odds_list[len(odds_list)//2]
                    
                    await db.save_odds(
                        match_id=match_id,
                        ah_line=handicap,
                        side="home",
                        best_odds=best['odds'],
                        best_bookmaker=best['name'],
                        second_best_odds=second['odds'],
                        second_best_bookmaker=second['name'],
                        median_odds=median,
                        num_bookmakers=len(bookmakers),
                    )
                    odds_saved += 1
                    total_bk_lines += 1
            
            processed += 1
            stats[f'deep_{worker_id}'] = processed
            
            duration = (datetime.now() - game_start).total_seconds()
            logger.info(f"[DEEP-{worker_id}] ✓ {home_team} vs {away_team}: {odds_saved}/{len(ah_lines[:10])} bk ({duration:.0f}s)")
            
            job_queue.task_done()
            
        except Exception as e:
            logger.error(f"[DEEP-{worker_id}] Erro: {str(e)[:100]}")
            job_queue.task_done()
    
    logger.info(f"[DEEP-{worker_id}] Encerrado: {processed} jogos, {total_bk_lines} linhas bk")


async def main():
    """Executa coleta profunda com múltiplas abas."""
    logger.info("="*60)
    logger.info("BETINASIA BOT - Coleta Profunda")
    logger.info(f"  {NUM_TABS} abas paralelas")
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
        # Coleta URLs
        index_page = await context.new_page()
        logger.info("[INDEX] Coletando URLs...")
        all_games = await get_all_game_urls(index_page, LEAGUES)
        await index_page.close()
        
        logger.info(f"[INDEX] Total: {len(all_games)} jogos")
        
        if not all_games:
            logger.error("Nenhum jogo encontrado!")
            return
        
        # Fila
        job_queue = asyncio.Queue()
        for game in all_games:
            await job_queue.put(game)
        
        # Cria abas
        pages = [await context.new_page() for _ in range(NUM_TABS)]
        
        start_time = datetime.now()
        logger.info(f"Iniciando {NUM_TABS} workers...")
        
        # Workers
        tasks = [
            asyncio.create_task(deep_worker(i, pages[i], job_queue, db, stats))
            for i in range(NUM_TABS)
        ]
        
        await asyncio.gather(*tasks)
        
        # Stats
        duration = (datetime.now() - start_time).total_seconds()
        total = sum(v for k, v in stats.items() if k.startswith('deep_'))
        
        logger.info("="*60)
        logger.info("COLETA CONCLUÍDA")
        logger.info(f"Jogos: {total}")
        logger.info(f"Tempo: {duration:.0f}s ({duration/60:.1f} min)")
        if total > 0:
            logger.info(f"Média: {duration/total:.1f}s/jogo")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.info("Interrompido pelo usuário")
    
    finally:
        for page in pages:
            await page.close()
        await browser.close()
        await p.stop()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
