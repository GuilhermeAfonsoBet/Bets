#!/usr/bin/env python3
"""
Teste: Verifica se o BetinAsia permite múltiplas abas no mesmo browser.
"""
import asyncio
import os
from pathlib import Path

import pytest
from playwright.async_api import async_playwright

async def test_multiple_tabs():
    # E2E/integrado: depende de rede, proxy, disponibilidade do site.
    if os.getenv("RUN_BETINASIA_E2E", "0").strip() not in ("1", "true", "True", "yes", "YES"):
        pytest.skip("E2E desabilitado (set RUN_BETINASIA_E2E=1 para rodar)")

    if not Path("betinasia_session.json").exists():
        pytest.skip("Requer betinasia_session.json (session exportada) para rodar este smoke-test.")

    print("\n" + "="*70)
    print("TESTE: Múltiplas abas no BetinAsia")
    print("="*70)
    
    p = await async_playwright().start()
    browser = await p.chromium.launch(headless=True)
    context = await browser.new_context(
        storage_state="betinasia_session.json",
        viewport={"width": 1920, "height": 1080},
    )
    
    # Cria a primeira aba
    print("\n[1] Abrindo primeira aba...")
    page1 = await context.new_page()
    await page1.goto("https://black.betinasia.com/sportsbook/football/XE/1")
    await page1.wait_for_load_state("networkidle")
    await page1.wait_for_timeout(2000)
    
    # Verifica se está logado
    url1 = page1.url
    text1 = await page1.inner_text("body")
    logged_in_1 = "JomanaSilva" in text1 or "sportsbook" in url1.lower()
    print(f"    URL: {url1[:60]}")
    print(f"    Logado: {'✓' if logged_in_1 else '✗'}")
    
    # Cria a segunda aba
    print("\n[2] Abrindo segunda aba...")
    page2 = await context.new_page()
    await page2.goto("https://black.betinasia.com/sportsbook/football/DE/12")
    await page2.wait_for_load_state("networkidle")
    await page2.wait_for_timeout(2000)
    
    # Verifica se está logado na segunda aba
    url2 = page2.url
    text2 = await page2.inner_text("body")
    logged_in_2 = "JomanaSilva" in text2 or "sportsbook" in url2.lower()
    print(f"    URL: {url2[:60]}")
    print(f"    Logado: {'✓' if logged_in_2 else '✗'}")
    
    # Volta para a primeira aba e verifica se ainda funciona
    print("\n[3] Voltando para primeira aba...")
    await page1.bring_to_front()
    await page1.reload()
    await page1.wait_for_load_state("networkidle")
    await page1.wait_for_timeout(2000)
    
    url1_after = page1.url
    text1_after = await page1.inner_text("body")
    logged_in_1_after = "JomanaSilva" in text1_after or "sportsbook" in url1_after.lower()
    print(f"    URL: {url1_after[:60]}")
    print(f"    Ainda logado: {'✓' if logged_in_1_after else '✗'}")
    
    # Tenta fazer operações simultâneas
    print("\n[4] Testando operações simultâneas...")
    
    async def navigate_page1():
        await page1.goto("https://black.betinasia.com/sportsbook/football/XE/1/2026-01-31,13,33")
        await page1.wait_for_load_state("networkidle")
        return "Page1 OK"
    
    async def navigate_page2():
        await page2.goto("https://black.betinasia.com/sportsbook/football/DE/12/2026-01-31,1047,945")
        await page2.wait_for_load_state("networkidle")
        return "Page2 OK"
    
    try:
        results = await asyncio.gather(navigate_page1(), navigate_page2())
        print(f"    Resultado: {results}")
        simultaneous_ok = True
    except Exception as e:
        print(f"    Erro em operações simultâneas: {e}")
        simultaneous_ok = False
    
    # Verifica se ambas as páginas têm conteúdo válido
    print("\n[5] Verificando conteúdo das páginas...")
    
    try:
        text1_final = await page1.inner_text("body")
        text2_final = await page2.inner_text("body")
        
        page1_has_odds = "Asian Handicap" in text1_final or "Home" in text1_final
        page2_has_odds = "Asian Handicap" in text2_final or "Home" in text2_final
        
        print(f"    Página 1 tem dados de odds: {'✓' if page1_has_odds else '✗'}")
        print(f"    Página 2 tem dados de odds: {'✓' if page2_has_odds else '✗'}")
    except Exception as e:
        print(f"    Erro ao verificar conteúdo: {e}")
        page1_has_odds = False
        page2_has_odds = False
    
    # Screenshots
    await page1.screenshot(path="test_tab1.png")
    await page2.screenshot(path="test_tab2.png")
    print("\n    Screenshots salvos: test_tab1.png, test_tab2.png")
    
    await browser.close()
    await p.stop()
    
    print("\n" + "="*70)
    print("RESULTADO DO TESTE")
    print("="*70)
    
    if logged_in_1 and logged_in_2 and logged_in_1_after and simultaneous_ok and page1_has_odds and page2_has_odds:
        print("\n✅ MÚLTIPLAS ABAS FUNCIONAM!")
        print("   Podemos usar duas abas para coleta paralela:")
        print("   - Aba 1: Coleta rápida de best odds")
        print("   - Aba 2: Coleta profunda com bookmakers")
    elif logged_in_1 and logged_in_2 and logged_in_1_after:
        print("\n⚠️ MÚLTIPLAS ABAS FUNCIONAM PARCIALMENTE")
        print("   As abas funcionam, mas operações simultâneas podem ter problemas.")
        print("   Recomendação: Alternar entre abas em vez de operações paralelas.")
    else:
        print("\n❌ MÚLTIPLAS ABAS NÃO FUNCIONAM")
        print("   O BetinAsia parece bloquear múltiplas sessões.")
        print("   Opções:")
        print("   1. Usar outra conta")
        print("   2. Alternar entre tipos de coleta na mesma sessão")

if __name__ == "__main__":
    asyncio.run(test_multiple_tabs())
