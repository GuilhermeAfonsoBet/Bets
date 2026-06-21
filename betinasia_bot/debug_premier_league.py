#!/usr/bin/env python3
"""
Debug: Investiga URLs dos jogos da Premier League.
"""
import asyncio
import re
from playwright.async_api import async_playwright

async def debug_premier():
    print("\n" + "="*70)
    print("DEBUG: Investigando URLs da Premier League")
    print("="*70)
    
    p = await async_playwright().start()
    browser = await p.chromium.launch(headless=True)
    context = await browser.new_context(
        storage_state="betinasia_session.json",
        viewport={"width": 1920, "height": 1080},
    )
    page = await context.new_page()
    
    # Acessa página da Premier League
    print("\n[1] Acessando England Premier League (XE/1)...")
    await page.goto("https://black.betinasia.com/sportsbook/football/XE/1")
    await page.wait_for_load_state("networkidle")
    await page.wait_for_timeout(3000)
    
    # Conta jogos
    print("\n[2] Analisando URLs dos jogos...")
    
    game_links = await page.query_selector_all("a")
    
    all_game_urls = []
    url_codes = {}
    
    for link in game_links:
        href = await link.get_attribute("href")
        if href and "/sportsbook/football/" in href and "," in href:
            base_url = href.split("?")[0]
            if base_url not in all_game_urls:
                all_game_urls.append(base_url)
                
                # Extrai o código da liga do URL
                match = re.search(r'/sportsbook/football/([A-Z]+/\d+|[A-Z]+)/', href)
                if match:
                    code = match.group(1)
                    url_codes[code] = url_codes.get(code, 0) + 1
    
    print(f"\n    Total de URLs de jogos: {len(all_game_urls)}")
    
    print(f"\n[3] Códigos encontrados nas URLs:")
    for code, count in sorted(url_codes.items(), key=lambda x: -x[1]):
        print(f"    {code}: {count} jogos")
    
    # Mostra URLs por código
    print(f"\n[4] Primeiras URLs de cada código:")
    shown_codes = set()
    for url in all_game_urls[:30]:
        match = re.search(r'/sportsbook/football/([A-Z]+/\d+|[A-Z]+)/', url)
        if match:
            code = match.group(1)
            if code not in shown_codes:
                shown_codes.add(code)
                print(f"    {code}: {url}")
    
    # Verifica se há jogos com XE/1
    xe1_games = [u for u in all_game_urls if "XE/1/" in u]
    print(f"\n[5] Jogos com código XE/1: {len(xe1_games)}")
    for url in xe1_games[:10]:
        print(f"    - {url}")
    
    await browser.close()
    await p.stop()
    
    print("\n" + "="*70)
    print("CONCLUSÃO:")
    print("="*70)
    
    if len(xe1_games) == len(all_game_urls):
        print(f"✓ Todos os {len(all_game_urls)} jogos usam código XE/1")
    else:
        print(f"! Apenas {len(xe1_games)} de {len(all_game_urls)} jogos usam XE/1")
        print(f"! Os outros jogos são de OUTRAS ligas mostradas na página")

if __name__ == "__main__":
    asyncio.run(debug_premier())
