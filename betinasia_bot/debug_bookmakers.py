# -*- coding: utf-8 -*-
"""
Script de debug para capturar odds de bookmakers.
Testa a interação com o painel que aparece ao clicar em uma linha de AH.
"""

import asyncio
import re
import json
from playwright.async_api import async_playwright


async def debug_bookmakers():
    """
    Debug completo da extração de odds por bookmaker.
    """
    print("\n" + "="*60)
    print("DEBUG: CAPTURA DE ODDS POR BOOKMAKER")
    print("="*60 + "\n")
    
    p = await async_playwright().start()
    browser = await p.chromium.launch(headless=True)
    context = await browser.new_context(
        viewport={"width": 1920, "height": 1080},
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    )
    page = await context.new_page()
    
    try:
        # === 1. LOGIN ===
        print("[1] Fazendo login...")
        await page.goto("https://black.betinasia.com/login")
        await page.wait_for_timeout(2000)
        
        # Preenche credenciais
        inputs = await page.query_selector_all("input")
        for inp in inputs:
            input_type = await inp.get_attribute("type")
            if input_type == "password":
                await inp.fill("Jom1928@")
            elif input_type in ["text", "email", None]:
                await inp.fill("JomanaSilva")
        
        # Clica no botão de login
        btn = await page.query_selector("button[type='submit']")
        if btn:
            await btn.click()
        await page.wait_for_timeout(4000)
        
        # Verifica se logou
        if "login" in page.url.lower():
            print("    ERRO: Falha no login!")
            await page.screenshot(path="debug_login_error.png")
            return
        print("    OK: Login realizado!")
        
        # Salva sessão para uso futuro
        await context.storage_state(path="betinasia_session.json")
        print("    Sessão salva em betinasia_session.json")
        
        # === 2. NAVEGA PARA LISTA DE JOGOS ===
        print("\n[2] Acessando Premier League...")
        await page.goto("https://black.betinasia.com/sportsbook/football/XE/1")
        await page.wait_for_timeout(3000)
        
        # Encontra primeiro jogo disponível
        game_links = await page.query_selector_all("a")
        game_url = None
        
        for link in game_links:
            href = await link.get_attribute("href")
            if href and "/sportsbook/football/" in href and "," in href:
                game_url = f"https://black.betinasia.com{href}" if href.startswith("/") else href
                break
        
        if not game_url:
            print("    ERRO: Nenhum jogo encontrado!")
            await page.screenshot(path="debug_no_games.png")
            return
        
        print(f"    Jogo encontrado: {game_url}")
        
        # === 3. ACESSA PÁGINA DO JOGO ===
        print("\n[3] Acessando página do jogo...")
        await page.goto(game_url)
        await page.wait_for_timeout(3000)
        
        # Extrai nome dos times
        page_text = await page.inner_text("body")
        teams_match = re.search(r'([A-Za-z\s&\.\-\']+)\s+Vs\.\s+([A-Za-z\s&\.\-\']+)', page_text[:2000])
        if teams_match:
            print(f"    Jogo: {teams_match.group(1).strip()} vs {teams_match.group(2).strip()}")
        
        # === 4. EXPANDE LINHAS DE AH ===
        print("\n[4] Expandindo linhas de Asian Handicap...")
        
        # Clica em "Show all lines" ou equivalente
        for _ in range(3):
            expand_btns = await page.query_selector_all(
                "text=Show all lines, text=Mostrar todas as linhas, "
                "text=Show all, text=Mostrar todas"
            )
            for btn in expand_btns:
                try:
                    if await btn.is_visible():
                        await btn.click()
                        await page.wait_for_timeout(500)
                        print("    Clicou em 'Show all lines'")
                except:
                    continue
        
        await page.wait_for_timeout(1000)
        
        # === 5. CONTA LINHAS DE AH VISÍVEIS ===
        print("\n[5] Contando linhas de AH...")
        
        page_text = await page.inner_text("body")
        
        # Padrão: HANDICAP seguido de Home/Away
        ah_pattern = r'([+-]?\d+[,.]?\d*)\s*\n\s*Home\s*\n\s*(\d+[,.]\d+)\s*\n\s*Away\s*\n\s*(\d+[,.]\d+)'
        ah_matches = re.findall(ah_pattern, page_text)
        
        print(f"    Linhas AH encontradas: {len(ah_matches)}")
        for match in ah_matches[:10]:  # Mostra até 10
            handicap = match[0].replace(",", ".")
            home_odds = match[1].replace(",", ".")
            away_odds = match[2].replace(",", ".")
            print(f"      AH {handicap}: Home={home_odds} Away={away_odds}")
        
        if len(ah_matches) > 10:
            print(f"      ... e mais {len(ah_matches) - 10} linhas")
        
        # === 6. CLICA EM UMA LINHA PARA VER BOOKMAKERS ===
        print("\n[6] Clicando em uma linha de AH para abrir painel de bookmakers...")
        
        # Procura por botões/elementos clicáveis com odds
        # As odds são botões clicáveis que abrem o painel direito
        odds_buttons = await page.query_selector_all("button")
        
        clicked = False
        for btn in odds_buttons:
            try:
                btn_text = await btn.inner_text()
                # Procura por botão que contenha um número decimal (odds)
                if re.search(r'^\d+\.\d+$', btn_text.strip()):
                    await btn.click()
                    clicked = True
                    print(f"    Clicou no botão com odds: {btn_text.strip()}")
                    break
            except:
                continue
        
        if not clicked:
            # Tenta clicar diretamente em texto que parece odds
            print("    Tentando seletor alternativo...")
            
            # Busca elementos que contenham odds próximo a "Home"
            home_elements = await page.query_selector_all("text=Home")
            for el in home_elements:
                try:
                    # Pega o próximo elemento irmão que deve ser a odds
                    parent = await el.evaluate_handle("el => el.parentElement")
                    # Clica no parent para ativar
                    await parent.click()
                    clicked = True
                    print("    Clicou em elemento pai de 'Home'")
                    break
                except:
                    continue
        
        await page.wait_for_timeout(2000)
        
        # === 7. CAPTURA PAINEL DE BOOKMAKERS ===
        print("\n[7] Capturando painel de bookmakers...")
        
        # Screenshot
        await page.screenshot(path="debug_bookmakers_panel.png")
        print("    Screenshot: debug_bookmakers_panel.png")
        
        # Texto completo
        full_text = await page.inner_text("body")
        with open("debug_page_text.txt", "w", encoding="utf-8") as f:
            f.write(full_text)
        print("    Texto: debug_page_text.txt")
        
        # === 8. PROCURA BOOKMAKERS NO TEXTO ===
        print("\n[8] Procurando bookmakers no texto...")
        
        # Bookmakers conhecidos do BetinAsia
        bookmakers = [
            "3et", "4casters", "bdaq", "bf", "ibc", "ipm", 
            "isn", "pin", "pinnacle", "sbo", "sing", "mbook",
            "molly", "sharp", "isn88", "pin88"
        ]
        
        found_bookmakers = []
        text_lower = full_text.lower()
        
        for bk in bookmakers:
            if bk in text_lower:
                found_bookmakers.append(bk)
                print(f"      {bk}: ENCONTRADO")
        
        if not found_bookmakers:
            print("    NENHUM bookmaker encontrado no texto!")
            print("    O painel pode não ter aberto corretamente.")
        
        # === 9. EXTRAI ODDS DOS BOOKMAKERS ===
        print("\n[9] Tentando extrair odds dos bookmakers...")
        
        # Padrão: nome do bookmaker seguido de odds
        # Ex: "pin\n1.95\n1.95" ou "3et 2.10 1.80"
        for bk in found_bookmakers:
            # Procura padrão: bookmaker seguido de dois números
            patterns = [
                rf'{bk}\s*\n\s*(\d+[,.]\d+)\s*\n\s*(\d+[,.]\d+)',  # separado por newline
                rf'{bk}\s+(\d+[,.]\d+)\s+(\d+[,.]\d+)',  # separado por espaço
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    odds1 = match.group(1).replace(",", ".")
                    odds2 = match.group(2).replace(",", ".")
                    print(f"      {bk}: {odds1} / {odds2}")
                    break
        
        # === 10. SALVA HTML PARA ANÁLISE ===
        print("\n[10] Salvando HTML para análise...")
        html = await page.content()
        with open("debug_page_html.html", "w", encoding="utf-8") as f:
            f.write(html)
        print("    HTML: debug_page_html.html")
        
        # === 11. INSPECIONA ESTRUTURA DO PAINEL DIREITO ===
        print("\n[11] Inspecionando estrutura do painel direito...")
        
        # O painel de bookmakers geralmente aparece no lado direito
        # Procura por elementos específicos do painel
        
        # Tenta encontrar o container do painel de odds
        panels = await page.query_selector_all("[class*='panel'], [class*='sidebar'], [class*='betslip']")
        print(f"    Encontrados {len(panels)} painéis potenciais")
        
        for i, panel in enumerate(panels):
            try:
                panel_text = await panel.inner_text()
                if len(panel_text) > 50:
                    # Verifica se contém bookmakers
                    has_bk = any(bk in panel_text.lower() for bk in bookmakers)
                    if has_bk:
                        print(f"    PAINEL {i} contém bookmakers!")
                        with open(f"debug_panel_{i}.txt", "w", encoding="utf-8") as f:
                            f.write(panel_text)
                        print(f"    Salvo em debug_panel_{i}.txt")
            except:
                continue
        
        print("\n" + "="*60)
        print("DEBUG CONCLUÍDO!")
        print("="*60)
        print("\nArquivos gerados:")
        print("  - debug_bookmakers_panel.png (screenshot)")
        print("  - debug_page_text.txt (texto da página)")
        print("  - debug_page_html.html (HTML completo)")
        print("  - betinasia_session.json (sessão salva)")
        
    except Exception as e:
        print(f"\n ERRO: {e}")
        await page.screenshot(path="debug_error.png")
        raise
        
    finally:
        await browser.close()
        await p.stop()


if __name__ == "__main__":
    asyncio.run(debug_bookmakers())
