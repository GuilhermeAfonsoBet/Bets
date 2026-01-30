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
    all_requests = []
    all_responses = []
    
    # Intercepta todas as requisições
    def handle_request(request):
        url = request.url
        # Filtra apenas requisições interessantes (não imagens, css, etc)
        if any(x in url for x in ['.png', '.jpg', '.css', '.woff', '.svg', '.ico', 'google', 'facebook', 'analytics']):
            return
        
        all_requests.append({
            "url": url,
            "method": request.method,
            "post_data": request.post_data,
        })
    
    def handle_response(response):
        url = response.url
        # Filtra apenas requisições interessantes
        if any(x in url for x in ['.png', '.jpg', '.css', '.woff', '.svg', '.ico', 'google', 'facebook', 'analytics']):
            return
        
        content_type = response.headers.get("content-type", "")
        
        all_responses.append({
            "url": url,
            "status": response.status,
            "content_type": content_type,
        })
    
    page.on("request", handle_request)
    page.on("response", handle_response)
    
    # 1. Acessa um jogo
    print("\n[1] Acessando um jogo da Premier League...")
    await page.goto("https://black.betinasia.com/sportsbook/football/XE/1/2026-01-31,13,33")
    await page.wait_for_load_state("networkidle")
    await page.wait_for_timeout(3000)
    
    print(f"    Requisições capturadas até agora: {len(all_requests)}")
    
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
    
    # Salva requisições do carregamento inicial
    initial_requests = list(all_requests)
    initial_responses = list(all_responses)
    
    print(f"    Requisições após expansão: {len(all_requests)}")
    
    # Limpa para focar nas novas
    all_requests.clear()
    all_responses.clear()
    
    # 3. Clica em uma odds para abrir o painel de bookmakers
    print("\n[3] Clicando em uma odds para ver bookmakers...")
    
    # Encontra um elemento de odds (busca por padrões comuns de odds)
    odds_patterns = ["1.9", "2.0", "1.8", "2.1", "1.7", "2.2", "1.95", "2.05"]
    odds_element = None
    
    for pattern in odds_patterns:
        elements = await page.query_selector_all(f"text='{pattern}'")
        for el in elements:
            try:
                if await el.is_visible():
                    box = await el.bounding_box()
                    if box and box['width'] > 20:
                        odds_element = el
                        print(f"    Encontrou odds: {pattern}")
                        break
            except:
                continue
        if odds_element:
            break
    
    if odds_element:
        parent = await odds_element.evaluate_handle("el => el.parentElement")
        await parent.click()
        await page.wait_for_timeout(3000)
        
        print(f"\n[4] Requisições capturadas após clicar na odds:")
        print(f"    Total: {len(all_requests)} requisições")
        
        # Mostra requisições capturadas
        for i, req in enumerate(all_requests[:20]):
            print(f"\n    [{i+1}] {req['method']} {req['url'][:100]}")
            if req.get('post_data'):
                print(f"        POST data: {str(req['post_data'])[:200]}")
        
        print(f"\n[5] Respostas capturadas:")
        print(f"    Total: {len(all_responses)} respostas")
        
        for i, resp in enumerate(all_responses[:10]):
            print(f"\n    [{i+1}] {resp['url'][:80]}")
            print(f"        Status: {resp['status']}, Type: {resp.get('content_type', 'N/A')}")
    else:
        print("    Não encontrou elemento de odds para clicar")
        print("    Tentando pegar screenshot para debug...")
        await page.screenshot(path="debug_network_page.png")
        print("    Screenshot salvo: debug_network_page.png")
    
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
    
    # Combina requisições iniciais + após clique
    all_data = {
        "initial_requests": initial_requests,
        "initial_responses": initial_responses,
        "after_click_requests": list(all_requests),
        "after_click_responses": list(all_responses),
    }
    
    with open("debug_network_all.json", "w") as f:
        json.dump(all_data, f, indent=2)
    
    print(f"    Total requisições iniciais: {len(initial_requests)}")
    print(f"    Total respostas iniciais: {len(initial_responses)}")
    print("    Salvo: debug_network_all.json")
    
    # Mostra URLs interessantes
    print("\n[9] URLs potencialmente interessantes:")
    interesting_keywords = ['api', 'odds', 'book', 'price', 'market', 'event', 'match', 'data']
    for req in initial_requests + list(all_requests):
        url_lower = req['url'].lower()
        if any(kw in url_lower for kw in interesting_keywords):
            print(f"    - {req['method']} {req['url'][:100]}")
    
    await browser.close()
    await p.stop()
    
    print("\n" + "="*70)
    print("ANÁLISE:")
    print("="*70)
    
    total_reqs = len(initial_requests) + len(all_requests)
    if total_reqs > 0:
        print(f"\n✓ Capturamos {total_reqs} requisições para análise.")
        print("  Verifique o arquivo debug_network_all.json")
    else:
        print("\n✗ Não capturamos requisições.")
        print("  Os dados podem estar renderizados no HTML ou via WebSocket.")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    asyncio.run(debug_network())
