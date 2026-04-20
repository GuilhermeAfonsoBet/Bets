#!/usr/bin/env python3
"""
Debug: Por que algumas ligas mostram menos jogos do que deveriam?
Investiga a página da Bundesliga.
"""
import asyncio
from playwright.async_api import async_playwright

async def debug_game_list():
    print("\n" + "="*70)
    print("DEBUG: Investigando lista de jogos da Bundesliga")
    print("="*70)
    
    p = await async_playwright().start()
    browser = await p.chromium.launch(headless=True)
    context = await browser.new_context(
        storage_state="betinasia_session.json",
        viewport={"width": 1920, "height": 1080},
    )
    page = await context.new_page()
    
    # Navega para a Bundesliga
    print("\n[1] Acessando Germany Bundesliga...")
    await page.goto("https://black.betinasia.com/sportsbook/football/XB")
    await page.wait_for_load_state("networkidle")
    await page.wait_for_timeout(3000)
    
    # Conta links ANTES de expandir
    links_before = await page.query_selector_all("a")
    game_urls_before = set()
    for link in links_before:
        href = await link.get_attribute("href")
        if href and "/sportsbook/football/" in href and "," in href:
            base_url = href.split("?")[0]
            game_urls_before.add(base_url)
    
    print(f"\n[2] ANTES de expandir:")
    print(f"    Links totais: {len(links_before)}")
    print(f"    URLs de jogos: {len(game_urls_before)}")
    
    # Screenshot antes
    await page.screenshot(path="debug_bundesliga_before.png")
    
    # Procura elementos que podem expandir a lista
    print("\n[3] Procurando botões de expansão...")
    
    # Tenta vários seletores
    selectors_to_try = [
        "text='Show more'",
        "text='Mostrar mais'",
        "text='Load more'",
        "text='Ver mais'",
        "button:has-text('more')",
        "button:has-text('mais')",
        "[class*='show-more']",
        "[class*='load-more']",
        "[class*='expand']",
    ]
    
    for selector in selectors_to_try:
        try:
            elements = await page.query_selector_all(selector)
            if elements:
                print(f"    ✓ Encontrado: {selector} ({len(elements)} elementos)")
        except:
            pass
    
    # Procura por datas/grupos que podem ser expandidos
    print("\n[4] Procurando grupos por data...")
    date_headers = await page.query_selector_all("[class*='date'], [class*='header'], [class*='group']")
    print(f"    Elementos de data/grupo: {len(date_headers)}")
    
    # Faz scroll extensivo
    print("\n[5] Fazendo scroll extensivo...")
    for i in range(10):
        await page.evaluate("window.scrollBy(0, 1000)")
        await page.wait_for_timeout(1000)
        print(f"    Scroll {i+1}/10...")
    
    # Volta ao topo
    await page.evaluate("window.scrollTo(0, 0)")
    await page.wait_for_timeout(1000)
    
    # Conta links DEPOIS de scroll
    links_after = await page.query_selector_all("a")
    game_urls_after = set()
    for link in links_after:
        href = await link.get_attribute("href")
        if href and "/sportsbook/football/" in href and "," in href:
            base_url = href.split("?")[0]
            game_urls_after.add(base_url)
    
    print(f"\n[6] DEPOIS de scroll:")
    print(f"    Links totais: {len(links_after)}")
    print(f"    URLs de jogos: {len(game_urls_after)}")
    
    # Screenshot depois
    await page.screenshot(path="debug_bundesliga_after.png", full_page=True)
    
    # Mostra as URLs encontradas
    print(f"\n[7] URLs de jogos encontradas:")
    for i, url in enumerate(sorted(game_urls_after), 1):
        print(f"    {i}. {url}")
    
    # Salva o HTML para análise
    html = await page.content()
    with open("debug_bundesliga_page.html", "w") as f:
        f.write(html)
    print(f"\n[8] HTML salvo em debug_bundesliga_page.html")
    
    # Verifica se há tabs/filtros de tempo
    print("\n[9] Procurando filtros de tempo...")
    time_filters = await page.query_selector_all("[class*='filter'], [class*='tab'], [class*='time']")
    for filt in time_filters[:10]:
        text = await filt.inner_text()
        if text and len(text) < 50:
            print(f"    - {text.strip()}")
    
    await browser.close()
    await p.stop()
    
    print("\n" + "="*70)
    print("DEBUG CONCLUÍDO!")
    print("="*70)
    print(f"\nResultado: {len(game_urls_after)} jogos encontrados")
    print("Verifique os screenshots e o HTML para mais detalhes.")

if __name__ == "__main__":
    asyncio.run(debug_game_list())
