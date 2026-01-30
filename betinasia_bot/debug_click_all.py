#!/usr/bin/env python3
"""
Debug: Tenta clicar no filtro "All" para mostrar todos os jogos.
"""
import asyncio
from playwright.async_api import async_playwright

async def debug_click_all():
    print("\n" + "="*70)
    print("DEBUG: Tentando clicar em 'All' para mostrar todos os jogos")
    print("="*70)
    
    p = await async_playwright().start()
    browser = await p.chromium.launch(headless=True)
    context = await browser.new_context(
        storage_state="betinasia_session.json",
        viewport={"width": 1920, "height": 1080},
    )
    page = await context.new_page()
    
    # 1. Acessa a página da Bundesliga
    print("\n[1] Acessando Germany Bundesliga (XB)...")
    await page.goto("https://black.betinasia.com/sportsbook/football/XB")
    await page.wait_for_load_state("networkidle")
    await page.wait_for_timeout(3000)
    
    # 2. Screenshot inicial
    await page.screenshot(path="debug_all_1_before.png", full_page=True)
    print("    Screenshot: debug_all_1_before.png")
    
    # 3. Conta jogos ANTES
    game_links = await page.query_selector_all("a")
    bundesliga_before = 0
    for link in game_links:
        href = await link.get_attribute("href")
        if href and "DE/12" in href and "," in href:
            bundesliga_before += 1
    print(f"\n[2] Jogos Bundesliga ANTES: {bundesliga_before}")
    
    # 4. Procura e clica no botão "All"
    print("\n[3] Procurando botão 'All'...")
    
    # Diferentes seletores para encontrar o botão "All"
    all_selectors = [
        "text='All'",
        "button:has-text('All')",
        "a:has-text('All')",
        "[class*='filter'] >> text='All'",
        "span:has-text('All')",
        "div:has-text('All')",
    ]
    
    clicked = False
    for selector in all_selectors:
        try:
            elements = await page.query_selector_all(selector)
            print(f"    Selector '{selector}': {len(elements)} elementos")
            
            for i, el in enumerate(elements):
                try:
                    text = await el.inner_text()
                    text = text.strip()[:50]
                    is_visible = await el.is_visible()
                    box = await el.bounding_box() if is_visible else None
                    
                    # Queremos o "All" que está relacionado a filtro de tempo, não "Football" etc
                    if text == "All" and is_visible and box:
                        print(f"      [{i}] '{text}' - visible={is_visible}, pos=({box['x']:.0f}, {box['y']:.0f})")
                        
                        # Clica no elemento
                        await el.click()
                        await page.wait_for_timeout(3000)
                        print(f"      >>> CLICOU!")
                        clicked = True
                        break
                except Exception as e:
                    continue
            
            if clicked:
                break
        except Exception as e:
            continue
    
    if not clicked:
        print("    ERRO: Não conseguiu clicar em 'All'")
        
        # Tenta clicar por posição (se soubermos a posição aproximada)
        print("\n[4] Tentando encontrar filtros de data/tempo...")
        
        # Procura elementos que parecem ser filtros
        page_text = await page.inner_text("body")
        
        # Verifica se há texto como "Today", "Tomorrow", etc. perto de "All"
        if "Today" in page_text:
            print("    Encontrado 'Today' na página")
            
            # Tenta clicar em elementos próximos
            today_els = await page.query_selector_all("text='Today'")
            for el in today_els:
                try:
                    if await el.is_visible():
                        # Encontra o pai que pode ter o "All"
                        parent = await el.evaluate_handle("el => el.parentElement")
                        parent_text = await parent.inner_text()
                        print(f"    Parent de 'Today': '{parent_text[:100]}...'")
                except:
                    pass
    
    # 5. Screenshot após clicar
    await page.screenshot(path="debug_all_2_after.png", full_page=True)
    print("\n    Screenshot: debug_all_2_after.png")
    
    # 6. Conta jogos DEPOIS
    game_links = await page.query_selector_all("a")
    bundesliga_after = 0
    bundesliga_urls = []
    for link in game_links:
        href = await link.get_attribute("href")
        if href and "DE/12" in href and "," in href:
            base_url = href.split("?")[0]
            if base_url not in bundesliga_urls:
                bundesliga_urls.append(base_url)
                bundesliga_after += 1
    
    print(f"\n[5] Jogos Bundesliga DEPOIS: {bundesliga_after}")
    
    if bundesliga_after > bundesliga_before:
        print(f"    >>> SUCESSO! Mais {bundesliga_after - bundesliga_before} jogos encontrados!")
    
    print("\n[6] URLs da Bundesliga:")
    for url in sorted(bundesliga_urls):
        print(f"    - {url}")
    
    # 7. Verifica datas dos jogos
    print("\n[7] Datas dos jogos:")
    import re
    dates = set()
    for url in bundesliga_urls:
        match = re.search(r'(\d{4}-\d{2}-\d{2})', url)
        if match:
            dates.add(match.group(1))
    
    for d in sorted(dates):
        count = sum(1 for u in bundesliga_urls if d in u)
        print(f"    {d}: {count} jogos")
    
    await browser.close()
    await p.stop()
    
    print("\n" + "="*70)
    print("DEBUG CONCLUÍDO!")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(debug_click_all())
