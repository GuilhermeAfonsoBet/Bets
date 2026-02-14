#!/usr/bin/env python3
"""
Debug: Extrai dados de odds diretamente do DOM renderizado.
Sem cliques - apenas lê o que está visível na página.
"""
import asyncio
import json
import re
from playwright.async_api import async_playwright

async def extract_from_dom():
    print("\n" + "="*70)
    print("DEBUG: Extração de dados do DOM renderizado")
    print("="*70)
    
    p = await async_playwright().start()
    browser = await p.chromium.launch(headless=True)
    context = await browser.new_context(
        storage_state="betinasia_session.json",
        viewport={"width": 1920, "height": 1080},
    )
    page = await context.new_page()
    
    # Navega para um jogo
    print("\n[1] Navegando para página do jogo...")
    await page.goto("https://black.betinasia.com/sportsbook/football/XE/1/2026-01-31,13,33")
    await page.wait_for_load_state("networkidle")
    await page.wait_for_timeout(3000)
    
    # Expande todas as linhas de AH
    print("\n[2] Expandindo linhas de AH...")
    for _ in range(5):
        btns = await page.query_selector_all("text='Show all lines'")
        for btn in btns:
            try:
                if await btn.is_visible():
                    await btn.click()
                    await page.wait_for_timeout(500)
            except:
                pass
    
    await page.wait_for_timeout(2000)
    
    # Extrai dados via JavaScript
    print("\n[3] Extraindo dados do DOM...")
    
    data = await page.evaluate("""
        () => {
            const result = {
                match: {},
                ah_lines: [],
                raw_elements: []
            };
            
            // Pega título do jogo
            const title = document.querySelector('h1, [class*="title"]');
            if (title) {
                result.match.title = title.textContent.trim();
            }
            
            // Procura por elementos que parecem odds
            const allElements = document.querySelectorAll('*');
            const oddsPattern = /^[12]\\.[0-9]{2,3}$/;
            const handicapPattern = /^[+-]?[0-9]+(\\.[0-9]+)?$/;
            
            let oddsElements = [];
            
            allElements.forEach(el => {
                const text = el.textContent.trim();
                
                // Procura por odds (números como 1.85, 2.10, etc)
                if (oddsPattern.test(text)) {
                    const parent = el.parentElement;
                    const grandparent = parent ? parent.parentElement : null;
                    const context = grandparent ? grandparent.textContent.substring(0, 100) : '';
                    
                    oddsElements.push({
                        odds: parseFloat(text),
                        tag: el.tagName,
                        context: context.replace(/\\s+/g, ' ').trim()
                    });
                }
            });
            
            result.odds_count = oddsElements.length;
            result.odds_sample = oddsElements.slice(0, 20);
            
            // Procura por seção Asian Handicap especificamente
            const pageText = document.body.innerText;
            
            // Extrai linhas de AH usando regex no texto da página
            // Padrão: HANDICAP Home ODDS Away ODDS
            const ahPattern = /([+-]?\\d+(?:[.,]\\d+)?)[\\s\\n]+Home[\\s\\n]+(\\d+[.,]\\d+)[\\s\\n]+Away[\\s\\n]+(\\d+[.,]\\d+)/g;
            
            let match;
            while ((match = ahPattern.exec(pageText)) !== null) {
                result.ah_lines.push({
                    handicap: match[1],
                    home_odds: parseFloat(match[2].replace(',', '.')),
                    away_odds: parseFloat(match[3].replace(',', '.'))
                });
            }
            
            // Também extrai o texto completo da seção Asian Handicap
            const ahIndex = pageText.indexOf('Asian Handicap');
            if (ahIndex > -1) {
                const ahSection = pageText.substring(ahIndex, ahIndex + 2000);
                result.ah_section_preview = ahSection.substring(0, 500);
            }
            
            return result;
        }
    """)
    
    print(f"\n[4] Resultados:")
    print(f"    Match title: {data.get('match', {}).get('title', 'N/A')}")
    print(f"    Odds elements encontrados: {data.get('odds_count', 0)}")
    print(f"    Linhas AH extraídas: {len(data.get('ah_lines', []))}")
    
    if data.get('ah_lines'):
        print(f"\n[5] Linhas de AH encontradas:")
        for line in data['ah_lines'][:10]:
            print(f"    AH {line['handicap']}: Home={line['home_odds']:.3f} Away={line['away_odds']:.3f}")
    
    if data.get('odds_sample'):
        print(f"\n[6] Amostra de odds encontradas:")
        for i, odds in enumerate(data['odds_sample'][:10]):
            print(f"    [{i}] {odds['odds']:.3f} ({odds['tag']}) - ctx: {odds['context'][:50]}...")
    
    if data.get('ah_section_preview'):
        print(f"\n[7] Preview da seção Asian Handicap:")
        print(f"    {data['ah_section_preview'][:300]}...")
    
    # Salva dados completos
    with open("debug_dom_data.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n    Dados salvos em: debug_dom_data.json")
    
    # Agora tenta clicar em UMA odds para ver estrutura do painel de bookmakers
    print("\n[8] Tentando abrir painel de bookmakers...")
    
    odds_elements = await page.query_selector_all("span")
    clicked = False
    
    for el in odds_elements[:100]:
        try:
            text = await el.text_content()
            if text and re.match(r'^[12]\.\d{2,3}$', text.strip()):
                if await el.is_visible():
                    box = await el.bounding_box()
                    if box and box['width'] > 20:
                        # Clica no pai
                        await el.evaluate("el => el.parentElement.click()")
                        await page.wait_for_timeout(2000)
                        clicked = True
                        print(f"    Clicou em odds: {text.strip()}")
                        break
        except:
            continue
    
    if clicked:
        # Extrai dados do painel de bookmakers
        panel_data = await page.evaluate("""
            () => {
                const bookmakers = [];
                const text = document.body.innerText;
                
                // Bookmakers conhecidos
                const knownBks = ['3et', '4casters', 'bdaq', 'bf', 'ibc', 'isn', 
                                  'mbook', 'molly', 'pin88', 'pinnacle', 'sbo', 
                                  'sharp', 'sing'];
                
                // Procura cada bookmaker no texto
                for (const bk of knownBks) {
                    const regex = new RegExp(bk + 'e?\\\\s*\\\\n\\\\s*(\\\\d+[.,]\\\\d+)\\\\s*\\\\n\\\\s*\\\\$?([\\\\d,]+)', 'gi');
                    const match = regex.exec(text);
                    if (match) {
                        bookmakers.push({
                            name: bk,
                            odds: parseFloat(match[1].replace(',', '.')),
                            limit: match[2]
                        });
                    }
                }
                
                return {
                    bookmakers: bookmakers,
                    visible_text_sample: text.substring(0, 1000)
                };
            }
        """)
        
        if panel_data.get('bookmakers'):
            print(f"\n[9] Bookmakers encontrados no painel:")
            for bk in panel_data['bookmakers']:
                print(f"    {bk['name']}: {bk['odds']:.3f} (limit: ${bk['limit']})")
        else:
            print(f"\n[9] Bookmakers não encontrados no painel")
            print(f"    Texto visível: {panel_data.get('visible_text_sample', '')[:200]}...")
        
        # Fecha o painel
        await page.keyboard.press("Escape")
    
    await browser.close()
    await p.stop()
    
    print("\n" + "="*70)
    print("CONCLUSÃO")
    print("="*70)
    
    if len(data.get('ah_lines', [])) > 10:
        print("\n✓ Conseguimos extrair linhas de AH diretamente do DOM!")
        print("  Isso pode ser mais rápido que clicar em cada elemento.")
    else:
        print("\n? Extração parcial - os dados de bookmakers requerem clique.")

if __name__ == "__main__":
    asyncio.run(extract_from_dom())
