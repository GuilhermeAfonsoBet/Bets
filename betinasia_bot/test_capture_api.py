#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Captura tráfego de rede do BetinAsia quando betslip é aberto.

Objetivo: descobrir a API interna que retorna odds do betslip
(All Bookies, TOTAL/AVERAGE/BEST, bookmakers individuais).

Roda no PC local (IP residencial, browser visível).

Uso:
    cd C:\\betinasia_test
    python test_capture_api.py
"""

import asyncio
import json
from datetime import datetime
from playwright.async_api import async_playwright

USERNAME = "JomanaSilva"
PASSWORD = "Jom1928@"
BASE_URL = "https://black.betinasia.com"


async def main():
    print("=" * 70)
    print("CAPTURA DE API — BetinAsia Betslip")
    print("=" * 70)

    p = await async_playwright().start()
    browser = await p.chromium.launch(headless=False)
    context = await browser.new_context(viewport={"width": 1920, "height": 1080})
    page = await context.new_page()

    # Captura TODOS os requests e responses da rede
    captured_requests = []
    captured_responses = []
    ws_frames = []

    def on_request(request):
        captured_requests.append({
            'url': request.url,
            'method': request.method,
            'headers': dict(request.headers),
            'post_data': request.post_data,
            'timestamp': datetime.now().isoformat(),
        })

    def on_response(response):
        captured_responses.append({
            'url': response.url,
            'status': response.status,
            'headers': dict(response.headers),
            'timestamp': datetime.now().isoformat(),
        })

    def on_websocket(ws):
        def on_frame_sent(data):
            ws_frames.append({'direction': 'sent', 'data': str(data)[:500], 'timestamp': datetime.now().isoformat()})
        def on_frame_received(data):
            ws_frames.append({'direction': 'received', 'data': str(data)[:500], 'timestamp': datetime.now().isoformat()})
        ws.on('framesent', on_frame_sent)
        ws.on('framereceived', on_frame_received)

    page.on('request', on_request)
    page.on('response', on_response)
    page.on('websocket', on_websocket)

    # === LOGIN ===
    print("\n[1] Fazendo login...")
    await page.goto(f"{BASE_URL}/login")
    await page.wait_for_load_state("networkidle")
    await page.wait_for_timeout(2000)

    username_input = await page.query_selector("input[type='text'], input[name*='user']")
    password_input = await page.query_selector("input[type='password']")
    if username_input and password_input:
        await username_input.fill(USERNAME)
        await password_input.fill(PASSWORD)
        await page.wait_for_timeout(500)
        login_btn = await page.query_selector("button[type='submit'], button:has-text('Log'), button:has-text('Sign')")
        if login_btn:
            await login_btn.click()
            await page.wait_for_timeout(5000)
    else:
        print("    Faca login manualmente no browser")
        input("    Pressione Enter apos fazer login...")

    if "/login" in page.url:
        print("    Faca login manualmente")
        input("    Pressione Enter apos login...")

    print("    Login OK!")

    # === NAVEGA PARA JOGO ===
    print("\n[2] Navegando para um jogo...")
    await page.goto(f"{BASE_URL}/sportsbook/football")
    await page.wait_for_load_state("networkidle")
    await page.wait_for_timeout(3000)

    # Clica no primeiro jogo
    game_link = await page.evaluate("""
        () => {
            const links = document.querySelectorAll('a');
            for (const link of links) {
                const href = link.getAttribute('href') || '';
                if (href.includes('/sportsbook/football/') && href.includes(',')) {
                    return href;
                }
            }
            return null;
        }
    """)

    if game_link:
        await page.goto(f"{BASE_URL}{game_link}" if game_link.startswith("/") else game_link)
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(3000)
        print(f"    Navegou para: {game_link[:60]}")
    else:
        print("    Navegue manualmente para um jogo")
        input("    Pressione Enter...")

    # Espera Asian Handicap
    try:
        await page.wait_for_selector("text=Asian Handicap", timeout=10000)
    except:
        pass

    # Expande linhas
    await page.evaluate("""
        () => {
            for (const el of document.querySelectorAll('span, button, div')) {
                const t = (el.innerText || '').trim().toLowerCase();
                if ((t === 'show all lines' || t === 'show all') && el.offsetParent !== null) {
                    try { el.click(); } catch(e) {}
                }
            }
        }
    """)
    await page.wait_for_timeout(2000)

    # === LIMPA CAPTURAS ANTERIORES (só queremos o que acontece ao clicar na odd) ===
    print("\n[3] Limpando capturas anteriores...")
    pre_click_count = len(captured_requests)
    pre_click_ws = len(ws_frames)

    # === CLICA NUMA ODD ===
    print("\n[4] Clicando numa odd do Asian Handicap...")
    
    clicked = await page.evaluate("""
        () => {
            // Encontra secao AH
            let sec = null;
            for (const h of document.querySelectorAll('div, span, h3')) {
                const t = (h.innerText || '').trim();
                if (t.includes('Asian Handicap')) {
                    let p = h.parentElement;
                    for (let i = 0; i < 10 && p; i++) {
                        if ((p.innerText || '').includes('Home') && (p.innerText || '').includes('Away')) {
                            sec = p; break;
                        }
                        p = p.parentElement;
                    }
                    if (sec) break;
                }
            }
            if (!sec) sec = document.body;
            
            // Encontra odd clicavel
            for (const el of sec.querySelectorAll('span, div')) {
                const t = (el.innerText || '').trim();
                const dotIdx = t.indexOf('.');
                if (dotIdx > 0 && dotIdx <= 2 && t.length <= 6) {
                    const val = parseFloat(t);
                    if (val >= 1.1 && val <= 10.0) {
                        const rect = el.getBoundingClientRect();
                        if (rect.width > 20 && rect.height > 10 && rect.width < 200) {
                            el.scrollIntoView({ behavior: 'instant', block: 'center' });
                            try { el.parentElement.click(); return {odd: t, method: 'parent'}; }
                            catch(e) {}
                            try { el.click(); return {odd: t, method: 'direct'}; }
                            catch(e) {}
                        }
                    }
                }
            }
            return null;
        }
    """)

    if clicked:
        print(f"    Clicou na odd: {clicked['odd']}")
    else:
        print("    Clique manualmente numa odd")
        input("    Pressione Enter apos clicar...")

    # === ESPERA BETSLIP CARREGAR ===
    print("\n[5] Aguardando betslip carregar (5s)...")
    await page.wait_for_timeout(5000)

    # === ANALISA REQUESTS POS-CLICK ===
    post_click_requests = captured_requests[pre_click_count:]
    post_click_ws = ws_frames[pre_click_ws:]

    print(f"\n{'=' * 70}")
    print(f"RESULTADO DA CAPTURA")
    print(f"{'=' * 70}")

    print(f"\n  Requests HTTP apos click: {len(post_click_requests)}")
    print(f"  Frames WebSocket apos click: {len(post_click_ws)}")

    # Filtra requests relevantes (não são imagens, CSS, etc)
    api_requests = []
    for req in post_click_requests:
        url = req['url']
        # Ignora assets estáticos
        if any(ext in url for ext in ['.png', '.jpg', '.svg', '.css', '.woff', '.gif', '.ico']):
            continue
        if 'google' in url or 'analytics' in url or 'tracking' in url:
            continue
        api_requests.append(req)

    print(f"\n  === REQUESTS API (excl. assets) === ({len(api_requests)})")
    for req in api_requests:
        print(f"\n    [{req['method']}] {req['url'][:120]}")
        if req['post_data']:
            print(f"    POST data: {str(req['post_data'])[:200]}")

    # Tenta capturar response body dos requests de API
    print(f"\n  === TENTANDO CAPTURAR RESPONSE BODIES ===")
    for req in api_requests:
        url = req['url']
        # Procura response correspondente
        for resp in captured_responses:
            if resp['url'] == url:
                content_type = resp['headers'].get('content-type', '')
                print(f"\n    URL: {url[:100]}")
                print(f"    Status: {resp['status']}")
                print(f"    Content-Type: {content_type}")
                break

    # WebSocket frames
    if post_click_ws:
        print(f"\n  === WEBSOCKET FRAMES APOS CLICK === ({len(post_click_ws)})")
        for frame in post_click_ws[:20]:
            print(f"    [{frame['direction']}] {frame['data'][:200]}")

    # Salva tudo em JSON para análise detalhada
    output = {
        'api_requests': api_requests,
        'ws_frames_post_click': post_click_ws[:50],
        'all_responses_post_click': [
            {'url': r['url'], 'status': r['status'], 'content_type': r['headers'].get('content-type', '')}
            for r in captured_responses[pre_click_count:]
        ],
    }

    with open('api_capture.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Dados completos salvos em: api_capture.json")

    # === SCREENSHOT ===
    await page.screenshot(path="api_capture_betslip.png", full_page=False)
    print(f"  Screenshot: api_capture_betslip.png")

    print(f"\n{'=' * 70}")
    print("O browser continua aberto para inspecao manual.")
    print("Abra DevTools (F12) > Network para ver os requests.")
    print("=" * 70)
    input("\nPressione Enter para fechar...")

    await browser.close()
    await p.stop()


if __name__ == "__main__":
    asyncio.run(main())
