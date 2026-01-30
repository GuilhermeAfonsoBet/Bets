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
    
    # Primeiro, acessa o site para ter os cookies/sessão
    print("\n[1] Inicializando sessão...")
    await page.goto("https://black.betinasia.com/sportsbook")
    await page.wait_for_load_state("networkidle")
    await page.wait_for_timeout(2000)
    
    # Endpoints para testar
    event_id = "2026-01-31,13,33"  # Brighton vs Everton
    
    endpoints = [
        f"/web/events/external/?event_id={event_id}&base_sport=fb",
        f"/v1/orders/position_by_event/?event_id={event_id}",
        "/v1/customers/JomanaSilva/bookie_accounts/",
        "/api/version",
    ]
    
    print("\n[2] Testando endpoints...")
    
    results = {}
    
    for endpoint in endpoints:
        url = f"https://black.betinasia.com{endpoint}"
        print(f"\n    Testing: {endpoint[:60]}...")
        
        try:
            # Faz a requisição via JavaScript para manter cookies
            response = await page.evaluate(f"""
                async () => {{
                    try {{
                        const response = await fetch("{url}", {{
                            credentials: 'include',
                            headers: {{
                                'Accept': 'application/json',
                            }}
                        }});
                        const text = await response.text();
                        return {{
                            status: response.status,
                            contentType: response.headers.get('content-type'),
                            body: text.substring(0, 5000),
                            ok: response.ok
                        }};
                    }} catch (e) {{
                        return {{ error: e.message }};
                    }}
                }}
            """)
            
            results[endpoint] = response
            
            print(f"    Status: {response.get('status', 'N/A')}")
            print(f"    Content-Type: {response.get('contentType', 'N/A')}")
            
            body = response.get('body', '')
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
    if events_endpoint in results and results[events_endpoint].get('ok'):
        print("\n🎯 ENDPOINT DE EVENTOS FUNCIONA!")
        print("   Este endpoint pode conter os dados de odds/bookmakers!")
        print("   Verifique o arquivo: api_response__web_events_external_.json")
    
    await browser.close()
    await p.stop()

if __name__ == "__main__":
    asyncio.run(debug_api())
