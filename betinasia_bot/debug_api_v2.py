#!/usr/bin/env python3
"""
Debug: Captura os headers de autenticação das requisições reais do site.
"""
import asyncio
import json
from playwright.async_api import async_playwright

async def debug_api():
    print("\n" + "="*70)
    print("DEBUG: Capturando headers de autenticação reais")
    print("="*70)
    
    p = await async_playwright().start()
    browser = await p.chromium.launch(headless=True)
    context = await browser.new_context(
        storage_state="betinasia_session.json",
        viewport={"width": 1920, "height": 1080},
    )
    page = await context.new_page()
    
    # Captura TODAS as requisições e seus headers
    captured_requests = []
    api_responses = {}
    
    async def capture_request(request):
        url = request.url
        if "black.betinasia.com" in url:
            req_data = {
                'url': url,
                'method': request.method,
                'headers': dict(request.headers),
            }
            captured_requests.append(req_data)
            
            # Se for uma requisição para /web/ ou /v1/, mostra
            if '/web/' in url or '/v1/' in url:
                print(f"\n    [REQ] {request.method} {url[:70]}")
                headers = dict(request.headers)
                for k, v in headers.items():
                    if k.lower() in ['authorization', 'cookie', 'x-csrf-token', 'x-requested-with']:
                        print(f"          {k}: {v[:60]}...")
    
    async def capture_response(response):
        url = response.url
        if '/web/' in url or '/v1/' in url:
            try:
                body = await response.text()
                api_responses[url] = {
                    'status': response.status,
                    'body': body[:10000]
                }
                print(f"    [RES] {response.status} {url[:70]}")
                if response.status == 200 and body:
                    print(f"          Body preview: {body[:100]}...")
            except:
                pass
    
    page.on("request", lambda req: asyncio.create_task(capture_request(req)))
    page.on("response", lambda res: asyncio.create_task(capture_response(res)))
    
    # Navega para a página de um jogo
    print("\n[1] Navegando para página do jogo (capturando todas as requisições)...")
    await page.goto("https://black.betinasia.com/sportsbook/football/XE/1/2026-01-31,13,33")
    await page.wait_for_load_state("networkidle")
    await page.wait_for_timeout(5000)
    
    print(f"\n[2] Total de requisições capturadas: {len(captured_requests)}")
    print(f"    Respostas de API capturadas: {len(api_responses)}")
    
    # Salva todas as requisições
    with open("debug_all_requests.json", "w") as f:
        json.dump(captured_requests, f, indent=2)
    print("\n    Salvo: debug_all_requests.json")
    
    # Salva respostas de API
    with open("debug_api_responses.json", "w") as f:
        json.dump(api_responses, f, indent=2)
    print("    Salvo: debug_api_responses.json")
    
    # Mostra requisições interessantes
    print("\n[3] Requisições para /web/ e /v1/:")
    for req in captured_requests:
        url = req['url']
        if '/web/' in url or '/v1/' in url:
            print(f"\n    URL: {url[:80]}")
            print(f"    Method: {req['method']}")
            
            # Mostra headers importantes
            headers = req['headers']
            important_headers = ['authorization', 'cookie', 'x-csrf-token', 'x-requested-with', 'x-auth-token']
            for h in important_headers:
                if h in headers:
                    print(f"    {h}: {headers[h][:80]}...")
    
    # Verifica localStorage e sessionStorage
    print("\n[4] Verificando localStorage/sessionStorage...")
    
    storage_data = await page.evaluate("""
        () => {
            let data = {};
            
            // localStorage
            data.localStorage = {};
            for (let i = 0; i < localStorage.length; i++) {
                let key = localStorage.key(i);
                let value = localStorage.getItem(key);
                if (value && (key.toLowerCase().includes('token') || 
                              key.toLowerCase().includes('auth') ||
                              key.toLowerCase().includes('session') ||
                              key.toLowerCase().includes('user'))) {
                    data.localStorage[key] = value.substring(0, 200);
                }
            }
            
            // sessionStorage
            data.sessionStorage = {};
            for (let i = 0; i < sessionStorage.length; i++) {
                let key = sessionStorage.key(i);
                let value = sessionStorage.getItem(key);
                if (value && (key.toLowerCase().includes('token') || 
                              key.toLowerCase().includes('auth') ||
                              key.toLowerCase().includes('session') ||
                              key.toLowerCase().includes('user'))) {
                    data.sessionStorage[key] = value.substring(0, 200);
                }
            }
            
            return data;
        }
    """)
    
    print(f"    localStorage keys: {list(storage_data.get('localStorage', {}).keys())}")
    print(f"    sessionStorage keys: {list(storage_data.get('sessionStorage', {}).keys())}")
    
    for key, value in storage_data.get('localStorage', {}).items():
        print(f"    localStorage[{key}]: {value[:100]}...")
    
    for key, value in storage_data.get('sessionStorage', {}).items():
        print(f"    sessionStorage[{key}]: {value[:100]}...")
    
    # Verifica cookies
    print("\n[5] Verificando cookies...")
    cookies = await context.cookies()
    for cookie in cookies:
        name = cookie['name']
        if 'token' in name.lower() or 'auth' in name.lower() or 'session' in name.lower():
            print(f"    Cookie {name}: {cookie['value'][:60]}...")
    
    await browser.close()
    await p.stop()
    
    print("\n" + "="*70)
    print("ANÁLISE COMPLETA")
    print("="*70)
    
    if api_responses:
        print("\n✓ Capturamos respostas de API!")
        for url, data in api_responses.items():
            if data['status'] == 200:
                print(f"\n  URL: {url[:70]}")
                print(f"  Status: {data['status']}")
                try:
                    body = json.loads(data['body'])
                    print(f"  Keys: {list(body.keys()) if isinstance(body, dict) else 'array'}")
                except:
                    pass

if __name__ == "__main__":
    asyncio.run(debug_api())
