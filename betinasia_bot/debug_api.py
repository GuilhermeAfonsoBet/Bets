#!/usr/bin/env python3
"""
Debug: Testa os endpoints de API descobertos.
"""
import asyncio
import json
from playwright.async_api import async_playwright

async def debug_api():
    print("\n" + "="*70)
    print("DEBUG: Testando endpoints de API")
    print("="*70)
    
    p = await async_playwright().start()
    browser = await p.chromium.launch(headless=True)
    context = await browser.new_context(
        storage_state="betinasia_session.json",
        viewport={"width": 1920, "height": 1080},
    )
    page = await context.new_page()
    
    # Captura headers de requisições autenticadas
    auth_headers = {}
    
    def capture_request(request):
        url = request.url
        if "black.betinasia.com" in url and "/web/" in url or "/v1/" in url:
            headers = dict(request.headers)
            if 'authorization' in headers or 'cookie' in headers:
                auth_headers['url'] = url
                auth_headers['headers'] = headers
                print(f"    [CAPTURED] Auth headers from: {url[:60]}")
    
    page.on("request", capture_request)
    
    # Primeiro, acessa o site para ter os cookies/sessão
    print("\n[1] Inicializando sessão...")
    await page.goto("https://black.betinasia.com/sportsbook")
    await page.wait_for_load_state("networkidle")
    await page.wait_for_timeout(2000)
    
    print(f"    Captured auth headers: {bool(auth_headers)}")
    
    # Endpoints para testar
    event_id = "2026-01-31,13,33"  # Brighton vs Everton
    
    endpoints = [
        f"/web/events/external/?event_id={event_id}&base_sport=fb",
        f"/v1/orders/position_by_event/?event_id={event_id}",
        "/v1/customers/JomanaSilva/bookie_accounts/",
        "/api/version",
    ]
    
    # Navega para uma página que faz requisições autenticadas
    print("\n[2] Navegando para capturar headers de autenticação...")
    await page.goto("https://black.betinasia.com/sportsbook/football/XE/1/2026-01-31,13,33")
    await page.wait_for_load_state("networkidle")
    await page.wait_for_timeout(3000)
    
    # Mostra headers capturados
    if auth_headers:
        print(f"\n    Headers capturados de: {auth_headers.get('url', 'N/A')[:60]}")
        captured_h = auth_headers.get('headers', {})
        for k, v in captured_h.items():
            if k.lower() in ['authorization', 'cookie', 'x-csrf-token', 'x-auth-token']:
                print(f"    {k}: {v[:50]}...")
    
    print("\n[3] Testando endpoints com headers capturados...")
    
    results = {}
    
    for endpoint in endpoints:
        url = f"https://black.betinasia.com{endpoint}"
        print(f"\n    Testing: {endpoint[:60]}...")
        
        try:
            # Usa request do Playwright que mantém contexto/cookies
            response = await context.request.get(url, headers={
                'Accept': 'application/json',
            })
            
            status = response.status
            body = await response.text()
            content_type = response.headers.get('content-type', '')
            
            results[endpoint] = {
                'status': status,
                'content_type': content_type,
                'body': body[:5000]
            }
            
            print(f"    Status: {status}")
            print(f"    Content-Type: {content_type}")
            
            if body:
                # Tenta parsear como JSON
                try:
                    data = json.loads(body)
                    print(f"    JSON válido: Sim")
                    print(f"    Keys: {list(data.keys()) if isinstance(data, dict) else 'array'}")
                    
                    # Salva a resposta completa
                    with open(f"api_response_{endpoint.split('?')[0].replace('/', '_')}.json", "w") as f:
                        json.dump(data, f, indent=2)
                    print(f"    Salvo em arquivo!")
                    
                    # Preview dos dados
                    preview = json.dumps(data, indent=2)[:500]
                    print(f"\n    Preview:\n{preview}...")
                    
                except json.JSONDecodeError:
                    print(f"    JSON válido: Não (HTML ou outro)")
                    print(f"    Body preview: {body[:200]}...")
                    
        except Exception as e:
            print(f"    Erro: {e}")
            results[endpoint] = {"error": str(e)}
    
    # Salva todos os resultados
    with open("api_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("[3] Resultados salvos em api_test_results.json")
    print("="*70)
    
    # Análise específica do endpoint de eventos
    events_endpoint = f"/web/events/external/?event_id={event_id}&base_sport=fb"
    if events_endpoint in results and results[events_endpoint].get('status') == 200:
        print("\n🎯 ENDPOINT DE EVENTOS FUNCIONA!")
        print("   Este endpoint pode conter os dados de odds/bookmakers!")
        print("   Verifique o arquivo: api_response__web_events_external_.json")
    else:
        print(f"\n⚠️ Endpoint de eventos retornou status: {results.get(events_endpoint, {}).get('status', 'N/A')}")
    
    await browser.close()
    await p.stop()

if __name__ == "__main__":
    asyncio.run(debug_api())
