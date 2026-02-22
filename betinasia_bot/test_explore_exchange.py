#!/usr/bin/env python3
"""
Abre browser para explorar Exchange mode manualmente.
Captura TODOS os requests de rede enquanto você navega.

Uso:
    python test_explore_exchange.py

Passos manuais no browser:
1. Navegue para um jogo
2. Clique numa odd (abre betslip)
3. Mude de Classic para Exchange
4. Observe se tem Back/Lay
5. Clique em qualquer coisa
6. Volte ao terminal e pressione Enter quando terminar
"""

import asyncio
import json
import os
from datetime import datetime
from playwright.async_api import async_playwright

USERNAME = os.getenv("BETINASIA_USERNAME", "").strip()
PASSWORD = os.getenv("BETINASIA_PASSWORD", "").strip()
BASE_URL = "https://black.betinasia.com"


async def main():
    print("=" * 60)
    print("EXPLORAR EXCHANGE MODE")
    print("=" * 60)

    p = await async_playwright().start()
    browser = await p.chromium.launch(headless=False)
    context = await browser.new_context(viewport={"width": 1920, "height": 1080})
    page = await context.new_page()

    # Captura rede
    all_requests = []
    ws_frames = []

    async def on_req(request):
        url = request.url
        if 'betslip' in url.lower() or '/v1/' in url:
            headers = await request.all_headers()
            all_requests.append({
                'url': url,
                'method': request.method,
                'headers': headers,
                'post_data': request.post_data,
                'time': datetime.now().isoformat(),
            })
            print(f"\n  >>> [{request.method}] {url}")
            if request.post_data:
                print(f"      POST: {request.post_data[:200]}")

    def on_ws(ws):
        def on_frame(data):
            data_str = str(data)
            if 'betslip' in data_str.lower() or 'pmm' in data_str.lower() or 'exchange' in data_str.lower() or 'lay' in data_str.lower():
                ws_frames.append({'data': data_str[:500], 'time': datetime.now().isoformat()})
                print(f"\n  <<< WS: {data_str[:200]}")
        ws.on('framereceived', on_frame)

    page.on('request', on_req)
    page.on('websocket', on_ws)

    # Login
    print("\n[1] Fazendo login...")
    await page.goto(f"{BASE_URL}/login")
    await page.wait_for_load_state("networkidle")
    await page.wait_for_timeout(2000)

    user_input = await page.query_selector("input[type='text']")
    pass_input = await page.query_selector("input[type='password']")
    if USERNAME and PASSWORD and user_input and pass_input:
        await user_input.fill(USERNAME)
        await pass_input.fill(PASSWORD)
        await page.wait_for_timeout(500)
        btn = await page.query_selector("button[type='submit']")
        if btn:
            await btn.click()
            await page.wait_for_timeout(5000)

    if "/login" in page.url:
        input("    Faca login manualmente. Enter quando pronto...")

    print("    Login OK!")
    print("\n[2] Navegando para futebol...")
    await page.goto(f"{BASE_URL}/sportsbook/football")
    await page.wait_for_load_state("networkidle")
    await page.wait_for_timeout(3000)

    print("\n" + "=" * 60)
    print("BROWSER ABERTO — INTERAJA MANUALMENTE:")
    print("=" * 60)
    print("""
    1. Clique num jogo
    2. Clique numa odd AH (abre betslip em Classic)
    3. No betslip, clique em 'Exchange'
    4. Observe se aparece Back/Lay
    5. Se aparecer Lay, clique nele

    Os requests de rede aparecem aqui em tempo real.
    Quando terminar, pressione Enter.
    """)

    input(">>> Pressione Enter quando terminar de explorar...")

    # Salva
    output = {'requests': all_requests, 'ws_frames': ws_frames}
    with open('exchange_capture.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{len(all_requests)} requests capturados")
    print(f"{len(ws_frames)} WS frames capturados")
    print(f"Salvo em exchange_capture.json")

    await browser.close()
    await p.stop()


if __name__ == "__main__":
    asyncio.run(main())
