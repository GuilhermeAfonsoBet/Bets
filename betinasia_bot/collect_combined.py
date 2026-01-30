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


async def extract_team_names(page: Page) -> tuple:
    """Extrai nomes dos times da página."""
    try:
        # Tenta extrair do header da página
        header = await page.query_selector("h1, [class*='match-header'], [class*='event-header']")
        if header:
            text = await header.inner_text()
            if " vs " in text or " v " in text:
                sep = " vs " if " vs " in text else " v "
                parts = text.split(sep)
                return parts[0].strip(), parts[1].strip().split("\n")[0]
        
        # Fallback: procura no texto da página
        page_text = await page.evaluate("() => document.body.innerText")
        lines = page_text.split("\n")
        for line in lines[:30]:
            if " vs " in line and len(line) < 100:
                parts = line.split(" vs ")
                return parts[0].strip(), parts[1].strip()
    except:
        pass
    return "Home", "Away"


async def click_and_extract_bookmakers(page: Page, handicap: str, home_odds: float) -> list:
    """Clica na odd e extrai bookmakers do painel."""
    bookmakers = []
    
    try:
        # Encontra todos os spans com odds
        odds_str = f"{home_odds:.3f}"[:5]  # Ex: "1.923" -> "1.92"
        
        elements = await page.query_selector_all("span")
        target_elem = None
        
        for elem in elements:
            try:
                text = await elem.inner_text()
                if not text or len(text) > 10:
                    continue
                    
                # Verifica se é a odd que procuramos
                if odds_str[:4] in text.replace(",", "."):
                    # Verifica contexto (deve ter o handicap)
                    parent = await elem.evaluate_handle("el => el.closest('div')")
                    ctx = await parent.evaluate("el => el.innerText")
                    
                    hcp_normalized = handicap.replace(",", ".").replace("+", "")
                    if hcp_normalized in ctx.replace(",", "."):
                        target_elem = elem
                        break
            except:
                continue
        
        if not target_elem:
            return []
        
        # Scroll e clique
        await target_elem.scroll_into_view_if_needed()
        await page.wait_for_timeout(300)
        
        # Clica no elemento pai (div)
        await page.evaluate("""(elem) => {
            const parent = elem.parentElement;
            if (parent) parent.click();
        }""", target_elem)
        
        await page.wait_for_timeout(2000)
        
        # Extrai texto do painel de bookmakers
        panel_text = await page.evaluate("""
            () => {
                // Procura pelo painel lateral que aparece
                const rightPanels = document.querySelectorAll('[class*="right"], [class*="sidebar"], [class*="panel"], [class*="drawer"]');
                for (const panel of rightPanels) {
                    const text = panel.innerText;
                    if (text.includes('$') && text.length > 50) {
                        // Verifica se tem nomes de bookmakers
                        if (/sbo|pin|3et|bf|bdaq|mbook|4cast|sharp/i.test(text)) {
                            return text;
                        }
                    }
                }
                // Fallback: todo o texto da página
                return document.body.innerText;
            }
        """)
        
        # Parse dos bookmakers
        # Padrões: "3et 1.854 $2,615" ou "pin88 1.862 $8,700"
        patterns = [
            r'(3et|3ete?)\s+(\d+[.,]\d+)\s+\$?([\d,]+)',
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
                bk_name = m[0].lower()
                bk_odds = float(m[1].replace(",", "."))
                bk_limit = m[2].replace(",", "")
                bookmakers.append({
                    'name': bk_name,
                    'odds': bk_odds,
                    'limit': int(bk_limit) if bk_limit.isdigit() else 0
                })
        
        # Fecha painel
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(300)
        
    except Exception as e:
        logger.debug(f"Erro extraindo bookmakers: {e}")
    
    return bookmakers


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
            if job_queue.empty():
                break
            continue
        
        try:
            game_url = game['url']
            league = game['league']
            
            # Navega
            await page.goto(game_url)
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(3000)
            
            # Verifica se carregou conteúdo
            page_text = await page.inner_text("body")
            if len(page_text) < 500:
                logger.warning(f"[DEEP-{worker_id}] Página não carregou corretamente")
                job_queue.task_done()
                continue
            
            # Expande AH
            for attempt in range(3):
                btns = await page.query_selector_all("text='Show all lines'")
                for btn in btns:
                    try:
                        if await btn.is_visible():
                            await btn.click()
                            await page.wait_for_timeout(800)
                    except:
                        pass
                await page.wait_for_timeout(500)
            
            # Extrai nomes dos times
            home_team, away_team = await extract_team_names(page)
            
            # Log de debug
            if home_team == "Home" and away_team == "Away":
                logger.debug(f"[DEEP-{worker_id}] Não encontrou times. URL: {game_url[:60]}...")
            
            # Extrai ID da URL
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
            
            if not ah_lines:
                logger.warning(f"[DEEP-{worker_id}] Nenhuma linha AH encontrada em {game_url[:50]}...")
                # Tenta screenshot para debug
                try:
                    await page.screenshot(path=f"debug_deep_{worker_id}.png")
                except:
                    pass
            
            # Captura bookmakers para cada linha (limitado às principais)
            odds_saved = 0
            lines_to_process = ah_lines[:15]  # Processa até 15 linhas
            
            for line in lines_to_process:
                handicap = line['ah_line']
                home_odds = line['home_odds']
                
                # Extrai bookmakers
                bookmakers = await click_and_extract_bookmakers(page, handicap, home_odds)
                
                if bookmakers:
                    # Ordena por odds
                    bookmakers.sort(key=lambda x: x['odds'], reverse=True)
                    
                    best = bookmakers[0]
                    second = bookmakers[1] if len(bookmakers) > 1 else {'name': '', 'odds': 0}
                    odds_list = [b['odds'] for b in bookmakers]
                    median = odds_list[len(odds_list)//2] if odds_list else 0
                    
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
            
            processed += 1
            stats[f'deep_{worker_id}'] = processed
            
            logger.info(f"[DEEP-{worker_id}] ✓ {home_team} vs {away_team}: {odds_saved}/{len(lines_to_process)} bk lines")
            
            job_queue.task_done()
            
        except Exception as e:
            logger.error(f"[DEEP-{worker_id}] Erro: {e}")
            job_queue.task_done()
    
    logger.info(f"[DEEP-{worker_id}] Worker encerrado - {processed} jogos")


async def verify_session(page: Page) -> bool:
    """Verifica se a sessão está válida."""
    try:
        await page.goto("https://black.betinasia.com/sportsbook")
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(3000)
        
        # Verifica URL - se redirecionou para login, sessão expirou
        current_url = page.url
        if "/login" in current_url:
            logger.warning("Redirecionado para login - sessão expirada")
            return False
        
        # Verifica se tem elementos de odds na página
        odds_elements = await page.query_selector_all("span")
        page_text = await page.inner_text("body")
        
        # Se tem "Football" e números que parecem odds, está OK
        has_football = "Football" in page_text or "football" in page_text
        has_odds_pattern = bool(re.search(r'\d\.\d{2,3}', page_text))
        
        if has_football and has_odds_pattern:
            return True
        
        # Verifica se tem botão de login visível
        login_btn = await page.query_selector("text='Log In'")
        if login_btn and await login_btn.is_visible():
            return False
            
        return True
    except Exception as e:
        logger.error(f"Erro verificando sessão: {e}")
        return False


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
    
    # Verifica sessão
    test_page = await context.new_page()
    session_valid = await verify_session(test_page)
    await test_page.close()
    
    if not session_valid:
        logger.error("❌ Sessão expirada! Execute: python -c \"from scraper.betinasia import BetinAsiaScraper; import asyncio; s=BetinAsiaScraper(); asyncio.run(s.start()); asyncio.run(s.login())\"")
        await browser.close()
        await p.stop()
        return
    
    logger.info("✓ Sessão válida")
    
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
