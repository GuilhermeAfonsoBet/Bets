#!/usr/bin/env python3
"""
Debug: Encontra as URLs corretas de cada liga navegando pelo menu.
"""
import asyncio
from playwright.async_api import async_playwright

LEAGUES_TO_FIND = [
    "England Premier League",
    "England Championship", 
    "Spain La Liga",
    "Italy Serie A",
    "France Ligue 1",
    "Netherlands Eredivisie",
    "Portugal",
    "Belgium",
]

async def find_leagues():
    print("\n" + "="*70)
    print("DEBUG: Encontrando URLs das ligas")
    print("="*70)
    
    p = await async_playwright().start()
    browser = await p.chromium.launch(headless=True)
    context = await browser.new_context(
        storage_state="betinasia_session.json",
        viewport={"width": 1920, "height": 1080},
    )
    page = await context.new_page()
    
    # Acessa página de futebol
    print("\n[1] Acessando página de futebol...")
    await page.goto("https://black.betinasia.com/sportsbook/football")
    await page.wait_for_load_state("networkidle")
    await page.wait_for_timeout(3000)
    
    # Procura links de ligas no menu lateral
    print("\n[2] Procurando ligas no menu...")
    
    # Expande seções do menu
    expand_buttons = await page.query_selector_all("[class*='expand'], [class*='toggle'], [class*='arrow']")
    for btn in expand_buttons[:20]:
        try:
            if await btn.is_visible():
                await btn.click()
                await page.wait_for_timeout(300)
        except:
            pass
    
    await page.wait_for_timeout(1000)
    
    # Procura links que parecem ser de ligas
    all_links = await page.query_selector_all("a")
    
    found_leagues = {}
    
    for link in all_links:
        try:
            href = await link.get_attribute("href")
            text = await link.inner_text()
            text = text.strip()
            
            if href and "/sportsbook/football/" in href and "," not in href:
                # É um link de liga (não de jogo)
                # Extrai o código da liga
                code = href.replace("/sportsbook/football/", "").split("?")[0]
                
                if code and len(code) < 20:  # Código razoável
                    for league_name in LEAGUES_TO_FIND:
                        if league_name.lower() in text.lower():
                            found_leagues[league_name] = {
                                "code": code,
                                "url": href,
                                "text": text[:50]
                            }
        except:
            continue
    
    print("\n[3] Ligas encontradas:")
    for name, info in sorted(found_leagues.items()):
        print(f"    {name}:")
        print(f"        Código: {info['code']}")
        print(f"        URL: {info['url']}")
        print(f"        Texto: {info['text']}")
    
    # Agora vamos clicar em cada liga e verificar a URL final
    print("\n[4] Verificando URLs clicando em cada liga...")
    
    verified_codes = {}
    
    # Lista de ligas para verificar com seletores
    leagues_selectors = [
        ("England Premier League", "a:has-text('Premier League')"),
        ("England Championship", "a:has-text('Championship')"),
        ("Spain La Liga", "a:has-text('La Liga')"),
        ("Italy Serie A", "a:has-text('Serie A')"),
        ("France Ligue 1", "a:has-text('Ligue 1')"),
        ("Netherlands Eredivisie", "a:has-text('Eredivisie')"),
        ("Portugal Primeira Liga", "a:has-text('Portugal')"),
        ("Belgium Pro League", "a:has-text('Belgium')"),
    ]
    
    for league_name, selector in leagues_selectors:
        try:
            # Volta para página de futebol
            await page.goto("https://black.betinasia.com/sportsbook/football")
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(2000)
            
            # Procura e clica no link da liga
            link = await page.query_selector(selector)
            if link and await link.is_visible():
                await link.click()
                await page.wait_for_load_state("networkidle")
                await page.wait_for_timeout(2000)
                
                # Pega a URL final
                final_url = page.url
                code = final_url.replace("https://black.betinasia.com/sportsbook/football/", "").split("?")[0]
                
                # Conta jogos
                game_links = await page.query_selector_all("a")
                game_count = 0
                for gl in game_links:
                    href = await gl.get_attribute("href")
                    if href and "/sportsbook/football/" in href and "," in href:
                        if code in href:
                            game_count += 1
                
                verified_codes[league_name] = {
                    "code": code,
                    "url": final_url,
                    "games": game_count
                }
                
                print(f"    ✓ {league_name}: {code} ({game_count} jogos)")
            else:
                print(f"    ✗ {league_name}: Link não encontrado")
                
        except Exception as e:
            print(f"    ✗ {league_name}: Erro - {e}")
    
    await browser.close()
    await p.stop()
    
    print("\n" + "="*70)
    print("CÓDIGOS VERIFICADOS PARA ATUALIZAR NO SCRAPER:")
    print("="*70)
    
    print("\nLEAGUE_CODES = {")
    for name, info in sorted(verified_codes.items()):
        print(f'    "{name}": "{info["code"]}",  # {info["games"]} jogos')
    print("}")

if __name__ == "__main__":
    asyncio.run(find_leagues())
