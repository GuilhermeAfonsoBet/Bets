#!/usr/bin/env python3
"""
Debug detalhado para investigar por que não estamos capturando todos os jogos da Bundesliga.
"""
import asyncio
import re
from playwright.async_api import async_playwright

async def debug_bundesliga():
    print("\n" + "="*70)
    print("DEBUG DETALHADO: Investigando jogos da Bundesliga")
    print("="*70)
    
    p = await async_playwright().start()
    browser = await p.chromium.launch(headless=True)
    context = await browser.new_context(
        storage_state="betinasia_session.json",
        viewport={"width": 1920, "height": 1080},
    )
    page = await context.new_page()
    
    # 1. Acessa a página da Bundesliga (URL correta: DE/12)
    print("\n[1] Acessando Germany Bundesliga (DE/12)...")
    await page.goto("https://black.betinasia.com/sportsbook/football/DE/12")
    await page.wait_for_load_state("networkidle")
    await page.wait_for_timeout(3000)
    
    # 2. Screenshot inicial
    await page.screenshot(path="debug_bundesliga_1_inicial.png", full_page=True)
    print("    Screenshot: debug_bundesliga_1_inicial.png")
    
    # 3. Verifica filtros de tempo
    print("\n[2] Procurando filtros de tempo...")
    
    # Procura elementos que parecem filtros de tempo/data
    time_filters = await page.query_selector_all(
        "[class*='filter'], [class*='date'], [class*='time'], "
        "[class*='period'], [class*='range'], "
        "button:has-text('Today'), button:has-text('Tomorrow'), "
        "button:has-text('All'), button:has-text('Next'), "
        "button:has-text('7 days'), button:has-text('week'), "
        "a:has-text('Today'), a:has-text('All')"
    )
    
    print(f"    Elementos de filtro encontrados: {len(time_filters)}")
    
    for i, f in enumerate(time_filters[:10]):
        try:
            text = await f.inner_text()
            text = text.strip().replace('\n', ' ')[:50]
            if text:
                print(f"      [{i}] '{text}'")
        except:
            pass
    
    # 4. Procura links de navegação por data
    print("\n[3] Procurando navegação por datas...")
    
    # Procura todas as datas visíveis na página
    page_text = await page.inner_text("body")
    date_patterns = [
        r'\d{2}/\d{2}/\d{4}',  # 30/01/2026
        r'\d{4}-\d{2}-\d{2}',  # 2026-01-30
        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}',  # Jan 30
        r'(Today|Tomorrow|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)',
    ]
    
    found_dates = set()
    for pattern in date_patterns:
        matches = re.findall(pattern, page_text, re.IGNORECASE)
        for m in matches:
            if isinstance(m, tuple):
                m = ' '.join(m)
            found_dates.add(m)
    
    print(f"    Datas encontradas na página: {len(found_dates)}")
    for d in sorted(found_dates)[:15]:
        print(f"      - {d}")
    
    # 5. Conta jogos antes de expandir
    print("\n[4] Contando jogos ANTES de expandir...")
    
    game_links = await page.query_selector_all("a")
    game_urls = set()
    bundesliga_urls = set()
    
    for link in game_links:
        href = await link.get_attribute("href")
        if href and "/sportsbook/football/" in href and "," in href:
            base_url = href.split("?")[0]
            game_urls.add(base_url)
            if "DE/12" in href or "XB/" in href:
                bundesliga_urls.add(base_url)
    
    print(f"    Total de jogos na página: {len(game_urls)}")
    print(f"    Jogos DA BUNDESLIGA (DE/12 ou XB/): {len(bundesliga_urls)}")
    
    # 6. Lista todos os jogos da Bundesliga encontrados
    print("\n[5] URLs da Bundesliga encontradas:")
    for url in sorted(bundesliga_urls):
        print(f"    - {url}")
    
    # 7. Tenta expandir clicando em todos os "Show more"
    print("\n[6] Expandindo todas as seções...")
    
    for attempt in range(5):
        show_more = await page.query_selector_all(
            "text='Show more', text='Load more', text='Mostrar mais', "
            "button:has-text('more'), button:has-text('mais')"
        )
        
        clicked = 0
        for btn in show_more:
            try:
                if await btn.is_visible():
                    await btn.scroll_into_view_if_needed()
                    await btn.click()
                    await page.wait_for_timeout(1500)
                    clicked += 1
            except:
                pass
        
        print(f"    Tentativa {attempt+1}: clicou em {clicked} botões")
        
        if clicked == 0:
            break
    
    # 8. Scroll extensivo
    print("\n[7] Fazendo scroll extensivo...")
    
    for i in range(15):
        await page.evaluate("window.scrollBy(0, 1000)")
        await page.wait_for_timeout(800)
    
    # Volta ao topo
    await page.evaluate("window.scrollTo(0, 0)")
    await page.wait_for_timeout(1000)
    
    # 9. Conta jogos DEPOIS de expandir
    print("\n[8] Contando jogos DEPOIS de expandir...")
    
    game_links = await page.query_selector_all("a")
    game_urls_after = set()
    bundesliga_urls_after = set()
    
    for link in game_links:
        href = await link.get_attribute("href")
        if href and "/sportsbook/football/" in href and "," in href:
            base_url = href.split("?")[0]
            game_urls_after.add(base_url)
            if "DE/12" in href or "XB/" in href:
                bundesliga_urls_after.add(base_url)
    
    print(f"    Total de jogos na página: {len(game_urls_after)}")
    print(f"    Jogos DA BUNDESLIGA (DE/12 ou XB/): {len(bundesliga_urls_after)}")
    
    # 10. Screenshot final
    await page.screenshot(path="debug_bundesliga_2_expandido.png", full_page=True)
    print("\n    Screenshot: debug_bundesliga_2_expandido.png")
    
    # 11. Lista todos os jogos por liga
    print("\n[9] Contagem de jogos por liga/código:")
    
    league_counts = {}
    for url in game_urls_after:
        # Extrai o código da liga do URL
        match = re.search(r'/sportsbook/football/([A-Z]+/\d+|[A-Z]+)/', url)
        if match:
            league_code = match.group(1)
            league_counts[league_code] = league_counts.get(league_code, 0) + 1
    
    for code, count in sorted(league_counts.items(), key=lambda x: -x[1]):
        print(f"    {code}: {count} jogos")
    
    # 12. Verifica se há links para outras páginas/seções
    print("\n[10] Procurando links de navegação (outras páginas)...")
    
    nav_links = await page.query_selector_all(
        "a[href*='page='], a[href*='offset='], "
        "button:has-text('Next'), button:has-text('Próximo'), "
        "[class*='pagination'], [class*='pager']"
    )
    
    print(f"    Elementos de paginação: {len(nav_links)}")
    
    for i, link in enumerate(nav_links[:5]):
        try:
            text = await link.inner_text()
            href = await link.get_attribute("href")
            print(f"      [{i}] '{text[:30]}' -> {href[:50] if href else 'N/A'}")
        except:
            pass
    
    # 13. Salva o HTML completo
    html = await page.content()
    with open("debug_bundesliga_page.html", "w") as f:
        f.write(html)
    print("\n    HTML salvo: debug_bundesliga_page.html")
    
    # 14. Lista completa de jogos da Bundesliga
    print("\n[11] URLs FINAIS da Bundesliga:")
    for i, url in enumerate(sorted(bundesliga_urls_after), 1):
        # Extrai a data do URL
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', url)
        date_str = date_match.group(1) if date_match else "N/A"
        print(f"    {i}. [{date_str}] {url}")
    
    # 15. Verifica texto específico para encontrar jogos
    print("\n[12] Buscando times da Bundesliga no texto...")
    
    bundesliga_teams = [
        "Bayern", "Dortmund", "Leverkusen", "Leipzig", "Frankfurt",
        "Stuttgart", "Freiburg", "Hoffenheim", "Wolfsburg", "Bremen",
        "Gladbach", "Mainz", "Augsburg", "Köln", "Bochum", "Heidenheim",
        "Union Berlin", "Kiel"
    ]
    
    found_teams = []
    for team in bundesliga_teams:
        if team.lower() in page_text.lower():
            found_teams.append(team)
    
    print(f"    Times encontrados: {len(found_teams)}")
    for t in found_teams:
        print(f"      - {t}")
    
    await browser.close()
    await p.stop()
    
    print("\n" + "="*70)
    print("DEBUG CONCLUÍDO!")
    print("="*70)
    
    # Resumo final
    print(f"\n>>> RESUMO:")
    print(f"    Jogos da Bundesliga encontrados: {len(bundesliga_urls_after)}")
    print(f"    (Usuário reporta que há 18 jogos no site)")
    
    if len(bundesliga_urls_after) < 18:
        print(f"\n>>> POSSÍVEIS CAUSAS DO PROBLEMA:")
        print(f"    1. Jogos em datas futuras não carregados")
        print(f"    2. Página precisa de navegação/filtro de tempo")
        print(f"    3. Alguns jogos podem estar em outra seção do site")

if __name__ == "__main__":
    asyncio.run(debug_bundesliga())
