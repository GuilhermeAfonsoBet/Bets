"""
Debug: Por que o clique no HOME de certas linhas não funciona?
Investiga especificamente a linha -0.5 e +2
"""
import asyncio
from playwright.async_api import async_playwright

KNOWN_BKS = ["3et", "4casters", "bdaq", "bf", "mbook", "pin88", "sbo", "sharp"]

def extract_bks(text):
    """Extrai bookmakers do texto."""
    return [bk for bk in KNOWN_BKS if bk in text.lower()]

async def debug():
    print("\n" + "="*70)
    print("DEBUG: Investigando problema de clique no HOME")
    print("="*70)
    
    p = await async_playwright().start()
    browser = await p.chromium.launch(headless=True)
    context = await browser.new_context(
        storage_state="betinasia_session.json",
        viewport={"width": 1920, "height": 1080},
    )
    page = await context.new_page()
    
    # Acessa jogo
    print("\n[1] Acessando jogo Brighton vs Everton...")
    await page.goto("https://black.betinasia.com/sportsbook/football/XE/1/2026-01-31,13,33")
    await page.wait_for_load_state("networkidle")
    await page.wait_for_timeout(3000)
    
    # Expande linhas
    print("[2] Expandindo todas as linhas AH...")
    for _ in range(5):
        btns = await page.query_selector_all("text='Show all lines'")
        for btn in btns:
            try:
                if await btn.is_visible():
                    await btn.click()
                    await page.wait_for_timeout(1500)
            except:
                pass
    await page.wait_for_timeout(2000)
    
    # Pega o texto da página para encontrar as odds atuais
    text = await page.inner_text("body")
    
    # Encontra odds da linha -0.5
    import re
    pattern_05 = r'-0[,.]5\s*\nHome\s*\n(\d+[,.]\d+)\s*\nAway\s*\n(\d+[,.]\d+)'
    match_05 = re.search(pattern_05, text)
    
    if match_05:
        home_odds_05 = match_05.group(1)
        away_odds_05 = match_05.group(2)
        print(f"\n[3] Linha -0.5 encontrada: Home={home_odds_05}, Away={away_odds_05}")
    else:
        print("\n[3] ERRO: Linha -0.5 não encontrada!")
        home_odds_05 = "1.962"  # fallback
        away_odds_05 = "2.000"
    
    # ============================================================
    # TESTE 1: Clique direto no elemento HOME -0.5
    # ============================================================
    print("\n" + "="*70)
    print(f"TESTE 1: Clicando no HOME da linha -0.5 (odds {home_odds_05})")
    print("="*70)
    
    elements = await page.query_selector_all(f"text='{home_odds_05}'")
    print(f"Elementos encontrados com '{home_odds_05}': {len(elements)}")
    
    for i, el in enumerate(elements):
        try:
            # Informações do elemento
            tag = await el.evaluate("el => el.tagName")
            text_content = await el.evaluate("el => el.textContent")
            parent_class = await el.evaluate("el => el.parentElement?.className || 'N/A'")
            grandparent_text = await el.evaluate(
                "el => (el.parentElement?.parentElement?.textContent || '').substring(0, 60)"
            )
            
            print(f"\n--- Elemento [{i}] ---")
            print(f"  Tag: {tag}")
            print(f"  Texto: '{text_content}'")
            print(f"  Classe do pai: {parent_class[:50]}...")
            print(f"  Contexto (avô): '{grandparent_text}'...")
            
            # Verifica se é da seção AH ou 1X2
            is_ah = "-0,5" in grandparent_text or "-0.5" in grandparent_text
            is_1x2 = "draw" in grandparent_text.lower()
            print(f"  É seção AH? {is_ah}")
            print(f"  É seção 1X2? {is_1x2}")
            
            if not await el.is_visible():
                print(f"  SKIP: Elemento não visível")
                continue
            
            # Scroll
            await el.scroll_into_view_if_needed()
            await page.wait_for_timeout(500)
            
            box = await el.bounding_box()
            print(f"  Posição: x={box['x']:.0f}, y={box['y']:.0f}, w={box['width']:.0f}, h={box['height']:.0f}")
            
            # Tenta diferentes métodos de clique
            print(f"\n  >> Tentando CLIQUE NO PAI...")
            parent = await el.evaluate_handle("el => el.parentElement")
            await parent.click()
            await page.wait_for_timeout(2000)  # Espera mais tempo
            
            panel_text = await page.inner_text("body")
            bks = extract_bks(panel_text)
            print(f"  Bookmakers após clique no pai: {len(bks)} - {bks}")
            
            if len(bks) == 0:
                print(f"\n  >> Tentando CLIQUE DIRETO NO ELEMENTO...")
                await page.keyboard.press("Escape")
                await page.wait_for_timeout(500)
                await el.click()
                await page.wait_for_timeout(2000)
                
                panel_text = await page.inner_text("body")
                bks = extract_bks(panel_text)
                print(f"  Bookmakers após clique direto: {len(bks)} - {bks}")
            
            if len(bks) == 0:
                print(f"\n  >> Tentando CLIQUE VIA JAVASCRIPT no pai...")
                await page.keyboard.press("Escape")
                await page.wait_for_timeout(500)
                await el.evaluate("el => el.parentElement.click()")
                await page.wait_for_timeout(2000)
                
                panel_text = await page.inner_text("body")
                bks = extract_bks(panel_text)
                print(f"  Bookmakers após JS click: {len(bks)} - {bks}")
            
            if len(bks) == 0:
                print(f"\n  >> Tentando DOUBLE CLICK...")
                await page.keyboard.press("Escape")
                await page.wait_for_timeout(500)
                await el.dblclick()
                await page.wait_for_timeout(2000)
                
                panel_text = await page.inner_text("body")
                bks = extract_bks(panel_text)
                print(f"  Bookmakers após double click: {len(bks)} - {bks}")
            
            # Tira screenshot
            await page.screenshot(path=f"debug_click_{i}.png")
            print(f"  Screenshot: debug_click_{i}.png")
            
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(500)
            
            if len(bks) > 0:
                print(f"\n  ✓ SUCESSO no elemento [{i}]!")
                break
            else:
                print(f"\n  ✗ FALHOU no elemento [{i}]")
                
        except Exception as e:
            print(f"  ERRO: {e}")
    
    # ============================================================
    # TESTE 2: Clique no AWAY da mesma linha (para comparar)
    # ============================================================
    print("\n" + "="*70)
    print(f"TESTE 2: Clicando no AWAY da linha -0.5 (odds {away_odds_05}) para comparar")
    print("="*70)
    
    elements_away = await page.query_selector_all(f"text='{away_odds_05}'")
    print(f"Elementos encontrados com '{away_odds_05}': {len(elements_away)}")
    
    for i, el in enumerate(elements_away):
        try:
            grandparent_text = await el.evaluate(
                "el => (el.parentElement?.parentElement?.textContent || '').substring(0, 60)"
            )
            print(f"\n--- Elemento [{i}] ---")
            print(f"  Contexto: '{grandparent_text}'...")
            
            await el.scroll_into_view_if_needed()
            await page.wait_for_timeout(500)
            
            parent = await el.evaluate_handle("el => el.parentElement")
            await parent.click()
            await page.wait_for_timeout(2000)
            
            panel_text = await page.inner_text("body")
            bks = extract_bks(panel_text)
            print(f"  Bookmakers: {len(bks)} - {bks}")
            
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(500)
            
            if len(bks) > 0:
                print(f"  ✓ AWAY funcionou!")
                break
                
        except Exception as e:
            print(f"  ERRO: {e}")
    
    # ============================================================
    # TESTE 3: Verifica se é problema de timing
    # ============================================================
    print("\n" + "="*70)
    print("TESTE 3: Testando com tempo de espera maior (5s)")
    print("="*70)
    
    elements = await page.query_selector_all(f"text='{home_odds_05}'")
    for i, el in enumerate(elements):
        grandparent_text = await el.evaluate(
            "el => (el.parentElement?.parentElement?.textContent || '').substring(0, 60)"
        )
        is_ah = "-0,5" in grandparent_text or "-0.5" in grandparent_text
        
        if is_ah:
            print(f"\nTestando elemento AH [{i}] com espera de 5s...")
            await el.scroll_into_view_if_needed()
            await page.wait_for_timeout(1000)
            
            parent = await el.evaluate_handle("el => el.parentElement")
            await parent.click()
            
            # Espera mais tempo
            print("  Esperando 5 segundos...")
            await page.wait_for_timeout(5000)
            
            panel_text = await page.inner_text("body")
            bks = extract_bks(panel_text)
            print(f"  Bookmakers após 5s: {len(bks)} - {bks}")
            
            await page.screenshot(path="debug_click_5s.png")
            await page.keyboard.press("Escape")
            break
    
    await browser.close()
    await p.stop()
    
    print("\n" + "="*70)
    print("DEBUG CONCLUÍDO!")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(debug())
