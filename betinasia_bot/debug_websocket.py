#!/usr/bin/env python3
"""
Debug: Verifica se os dados de odds vêm via WebSocket ou estão no HTML.
"""
import asyncio
import json
import re
from playwright.async_api import async_playwright

async def debug_websocket():
    print("\n" + "="*70)
    print("DEBUG: Investigando WebSocket e dados no HTML")
    print("="*70)
    
    p = await async_playwright().start()
    browser = await p.chromium.launch(headless=True)
    context = await browser.new_context(
        storage_state="betinasia_session.json",
        viewport={"width": 1920, "height": 1080},
    )
    page = await context.new_page()
    
    # Captura mensagens WebSocket
    ws_messages = []
    
    def handle_websocket(ws):
        print(f"\n    [WS] WebSocket conectado: {ws.url[:70]}")
        
        def on_message(msg):
            ws_messages.append({
                'url': ws.url,
                'data': str(msg)[:500]
            })
            # Mostra se contém dados de odds
            msg_str = str(msg).lower()
            if 'odds' in msg_str or 'price' in msg_str or 'bookmaker' in msg_str:
                print(f"    [WS MSG] Possível dado de odds: {str(msg)[:100]}...")
        
        ws.on("framereceived", lambda payload: on_message(payload))
    
    context.on("websocket", handle_websocket)
    
    # Navega para a página
    print("\n[1] Navegando para página do jogo...")
    await page.goto("https://black.betinasia.com/sportsbook/football/XE/1/2026-01-31,13,33")
    await page.wait_for_load_state("networkidle")
    await page.wait_for_timeout(5000)
    
    print(f"\n[2] WebSocket mensagens capturadas: {len(ws_messages)}")
    
    # Salva mensagens WebSocket
    if ws_messages:
        with open("debug_websocket_messages.json", "w") as f:
            json.dump(ws_messages, f, indent=2)
        print("    Salvo: debug_websocket_messages.json")
    
    # Verifica dados no HTML/JavaScript
    print("\n[3] Verificando dados no HTML (Next.js __NEXT_DATA__)...")
    
    next_data = await page.evaluate("""
        () => {
            // Next.js armazena dados em __NEXT_DATA__
            const nextDataEl = document.getElementById('__NEXT_DATA__');
            if (nextDataEl) {
                return nextDataEl.textContent;
            }
            return null;
        }
    """)
    
    if next_data:
        print("    ✓ Encontrado __NEXT_DATA__!")
        try:
            data = json.loads(next_data)
            print(f"    Keys: {list(data.keys())}")
            
            # Salva os dados
            with open("debug_next_data.json", "w") as f:
                json.dump(data, f, indent=2)
            print("    Salvo: debug_next_data.json")
            
            # Procura por odds nos dados
            data_str = json.dumps(data)
            if 'odds' in data_str.lower() or 'bookmaker' in data_str.lower():
                print("    ✓ Dados parecem conter informações de odds!")
            
            # Mostra estrutura do pageProps
            if 'props' in data and 'pageProps' in data['props']:
                page_props = data['props']['pageProps']
                print(f"    pageProps keys: {list(page_props.keys()) if isinstance(page_props, dict) else 'not dict'}")
                
        except json.JSONDecodeError as e:
            print(f"    Erro ao parsear: {e}")
    else:
        print("    ✗ __NEXT_DATA__ não encontrado")
    
    # Verifica window.__INITIAL_STATE__ ou similar
    print("\n[4] Verificando variáveis globais...")
    
    global_vars = await page.evaluate("""
        () => {
            let vars = {};
            
            // Variáveis comuns de estado
            const checkVars = [
                '__INITIAL_STATE__', '__PRELOADED_STATE__', '__REDUX_STATE__',
                '__APP_STATE__', 'window.state', 'window.data',
                '__NUXT__', 'window.odds', 'window.events'
            ];
            
            for (let v of checkVars) {
                try {
                    let value = eval(v);
                    if (value) {
                        vars[v] = typeof value === 'object' ? 
                            Object.keys(value).slice(0, 10) : 
                            String(value).substring(0, 100);
                    }
                } catch {}
            }
            
            return vars;
        }
    """)
    
    print(f"    Variáveis encontradas: {list(global_vars.keys())}")
    for k, v in global_vars.items():
        print(f"    {k}: {v}")
    
    # Extrai o HTML e procura por padrões de odds
    print("\n[5] Procurando padrões de odds no HTML...")
    
    html = await page.content()
    
    # Procura por padrões de odds (números decimais típicos de odds)
    odds_pattern = r'"odds":\s*[\d.]+'
    odds_matches = re.findall(odds_pattern, html)
    print(f"    Padrões 'odds': encontrados: {len(odds_matches)}")
    if odds_matches:
        print(f"    Exemplos: {odds_matches[:5]}")
    
    # Procura por bookmakers
    bookmaker_pattern = r'"bookmaker":\s*"([^"]+)"'
    bk_matches = re.findall(bookmaker_pattern, html)
    print(f"    Padrões 'bookmaker': encontrados: {len(bk_matches)}")
    if bk_matches:
        print(f"    Bookmakers: {list(set(bk_matches))[:10]}")
    
    # Procura por preços
    price_pattern = r'"price":\s*[\d.]+'
    price_matches = re.findall(price_pattern, html)
    print(f"    Padrões 'price': encontrados: {len(price_matches)}")
    
    await browser.close()
    await p.stop()
    
    print("\n" + "="*70)
    print("CONCLUSÃO")
    print("="*70)
    
    if ws_messages:
        print("\n✓ Dados vêm via WebSocket!")
        print("  Precisamos conectar ao WebSocket para receber dados em tempo real.")
    elif next_data and ('odds' in json.dumps(json.loads(next_data)).lower()):
        print("\n✓ Dados vêm via Server-Side Rendering (Next.js)!")
        print("  Os dados estão no HTML inicial - podemos extrair sem clicar.")
    else:
        print("\n? Dados podem estar carregando dinamicamente via JavaScript.")
        print("  Verifique os arquivos salvos para mais detalhes.")

if __name__ == "__main__":
    asyncio.run(debug_websocket())
