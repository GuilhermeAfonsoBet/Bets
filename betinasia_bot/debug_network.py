#!/usr/bin/env python3
"""
Debug: Investiga requisições de rede para encontrar endpoints de bookmakers.
"""
import asyncio
import json
from playwright.async_api import async_playwright

async def debug_network():
    print("\n" + "="*70)
    print("DEBUG: Investigando requisições de rede")
    print("="*70)
    
    p = await async_playwright().start()
    browser = await p.chromium.launch(headless=True)
    context = await browser.new_context(
        storage_state="betinasia_session.json",
        viewport={"width": 1920, "height": 1080},
    )
    page = await context.new_page()
    
    # Lista para armazenar requisições capturadas
    captured_requests = []
    captured_responses = []
    
    # Intercepta todas as requisições
    async def handle_request(request):
        url = request.url
        # Filtra apenas requisições interessantes (não imagens, css, etc)
        if any(x in url for x in ['.png', '.jpg', '.css', '.woff', '.svg', 'google', 'facebook']):
            return
        
        captured_requests.append({
            "url": url,
            "method": request.method,
            "headers": dict(request.headers),
            "post_data": request.post_data,
        })
    
    async def handle_response(response):
        url = response.url
        # Filtra apenas requisições interessantes
        if any(x in url for x in ['.png', '.jpg', '.css', '.woff', '.svg', 'google', 'facebook']):
            return
        
        content_type = response.headers.get("content-type", "")
        
        # Captura apenas JSON
        if "json" in content_type or "api" in url.lower() or "odds" in url.lower():
            try:
                body = await response.text()
                captured_responses.append({
                    "url": url,
                    "status": response.status,
                    "content_type": content_type,
                    "body_preview": body[:500] if body else "",
                })
            except:
                pass
    
    page.on("request", handle_request)
    page.on("response", handle_response)
    
    # 1. Acessa um jogo
    print("\n[1] Acessando um jogo da Premier League...")
    await page.goto("https://black.betinasia.com/sportsbook/football/XE/1/2026-01-31,13,33")
    await page.wait_for_load_state("networkidle")
    await page.wait_for_timeout(3000)
    
    print(f"    Requisições capturadas até agora: {len(captured_requests)}")
    
    # 2. Expande as linhas de AH
    print("\n[2] Expandindo linhas de AH...")
    btns = await page.query_selector_all("text='Show all lines'")
    for btn in btns:
        try:
            if await btn.is_visible():
                await btn.click()
                await page.wait_for_timeout(1000)
        except:
            pass
    
    await page.wait_for_timeout(2000)
    
    # Limpa as requisições anteriores para focar nas novas
    captured_requests.clear()
    captured_responses.clear()
    
    # 3. Clica em uma odds para abrir o painel de bookmakers
    print("\n[3] Clicando em uma odds para ver bookmakers...")
    
    # Encontra um elemento de odds
    odds_element = await page.query_selector("text='1.9'")
    if not odds_element:
        odds_element = await page.query_selector("text='2.0'")
    
    if odds_element:
        parent = await odds_element.evaluate_handle("el => el.parentElement")
        await parent.click()
        await page.wait_for_timeout(3000)
        
        print(f"\n[4] Requisições capturadas após clicar na odds:")
        print(f"    Total: {len(captured_requests)} requisições")
        
        # Mostra requisições capturadas
        for i, req in enumerate(captured_requests[:20]):
            print(f"\n    [{i+1}] {req['method']} {req['url'][:100]}")
            if req['post_data']:
                print(f"        POST data: {req['post_data'][:200]}")
        
        print(f"\n[5] Respostas JSON capturadas:")
        print(f"    Total: {len(captured_responses)} respostas")
        
        for i, resp in enumerate(captured_responses[:10]):
            print(f"\n    [{i+1}] {resp['url'][:80]}")
            print(f"        Status: {resp['status']}, Type: {resp['content_type']}")
            print(f"        Body: {resp['body_preview'][:300]}...")
    else:
        print("    Não encontrou elemento de odds para clicar")
    
    # 4. Verifica se há WebSocket
    print("\n[6] Verificando WebSockets...")
    
    # Captura info de WebSocket
    ws_info = await page.evaluate("""
        () => {
            // Verifica se há WebSocket no window
            let wsInfo = [];
            if (window.WebSocket) {
                wsInfo.push("WebSocket está disponível");
            }
            
            // Tenta encontrar conexões WebSocket ativas
            // (isso é limitado por segurança do browser)
            
            return wsInfo;
        }
    """)
    
    print(f"    WebSocket info: {ws_info}")
    
    # 5. Verifica dados no window/global
    print("\n[7] Verificando dados no JavaScript global...")
    
    js_data = await page.evaluate("""
        () => {
            let data = {};
            
            // Verifica algumas variáveis comuns
            if (window.__INITIAL_STATE__) data['__INITIAL_STATE__'] = 'exists';
            if (window.__DATA__) data['__DATA__'] = 'exists';
            if (window.appData) data['appData'] = 'exists';
            if (window.pageData) data['pageData'] = 'exists';
            if (window.odds) data['odds'] = 'exists';
            if (window.bookmakers) data['bookmakers'] = 'exists';
            
            // Verifica Redux/Vuex stores
            if (window.__REDUX_DEVTOOLS_EXTENSION__) data['Redux'] = 'possibly';
            if (window.__NUXT__) data['Nuxt'] = Object.keys(window.__NUXT__ || {});
            if (window.__NEXT_DATA__) data['Next.js'] = 'exists';
            
            return data;
        }
    """)
    
    print(f"    Dados encontrados: {js_data}")
    
    # 6. Salva todas as requisições para análise
    print("\n[8] Salvando requisições para análise...")
    
    with open("debug_network_requests.json", "w") as f:
        json.dump(captured_requests, f, indent=2)
    
    with open("debug_network_responses.json", "w") as f:
        json.dump(captured_responses, f, indent=2)
    
    print("    Salvo: debug_network_requests.json")
    print("    Salvo: debug_network_responses.json")
    
    await browser.close()
    await p.stop()
    
    print("\n" + "="*70)
    print("ANÁLISE:")
    print("="*70)
    
    if captured_responses:
        print("\n✓ Encontramos respostas JSON - pode haver uma API!")
        print("  Analise os arquivos salvos para encontrar endpoints úteis.")
    else:
        print("\n✗ Não encontramos respostas JSON óbvias.")
        print("  Os dados podem estar renderizados no HTML ou via WebSocket.")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    asyncio.run(debug_network())
