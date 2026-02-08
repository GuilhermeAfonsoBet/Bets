#!/usr/bin/env python3
"""
Captura headers EXATOS do POST /v1/betslips/ quando odd é clicada.

Objetivo: descobrir qual auth header/token a API precisa
(que causa 401 quando chamamos fetch() manual).

Roda no PC local (browser visível).

Uso:
    cd C:\\betinasia_test
    python test_capture_headers.py
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
    print("CAPTURA DE HEADERS — POST /v1/betslips/")
    print("=" * 70)

    p = await async_playwright().start()
    browser = await p.chromium.launch(headless=False)
    context = await browser.new_context(viewport={"width": 1920, "height": 1080})
    page = await context.new_page()

    # Captura TODOS os requests com headers completos
    betslip_requests = []

    async def on_request(request):
        url = request.url
        if '/v1/betslips' in url or '/betslip' in url.lower():
            headers = await request.all_headers()
            betslip_requests.append({
                'url': url,
                'method': request.method,
                'headers': headers,
                'post_data': request.post_data,
                'timestamp': datetime.now().isoformat(),
            })

    page.on('request', on_request)

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
        print("    Faca login manualmente")
        input("    Pressione Enter apos login...")

    if "/login" in page.url:
        input("    Faca login manualmente. Pressione Enter...")

    print("    Login OK!")

    # === NAVEGA PARA JOGO ===
    print("\n[2] Navegando para jogo...")
    await page.goto(f"{BASE_URL}/sportsbook/football")
    await page.wait_for_load_state("networkidle")
    await page.wait_for_timeout(3000)

    game_link = await page.evaluate("""
        () => {
            for (const link of document.querySelectorAll('a')) {
                const href = link.getAttribute('href') || '';
                if (href.includes('/sportsbook/football/') && href.includes(',')) return href;
            }
            return null;
        }
    """)

    if game_link:
        await page.goto(f"{BASE_URL}{game_link}" if game_link.startswith("/") else game_link)
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(3000)
        print(f"    Jogo: {game_link[:60]}")
    else:
        print("    Navegue para um jogo manualmente")
        input("    Pressione Enter...")

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

    # Limpa capturas anteriores
    betslip_requests.clear()

    # === CLICA NA ODD ===
    print("\n[3] Clicando numa odd AH...")
    clicked = await page.evaluate("""
        () => {
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
            for (const el of sec.querySelectorAll('span, div')) {
                const t = (el.innerText || '').trim();
                const d = t.indexOf('.');
                if (d > 0 && d <= 2 && t.length <= 6) {
                    const v = parseFloat(t);
                    if (v >= 1.1 && v <= 10.0) {
                        const r = el.getBoundingClientRect();
                        if (r.width > 20 && r.height > 10 && r.width < 200) {
                            el.scrollIntoView({ behavior: 'instant', block: 'center' });
                            try { el.parentElement.click(); return t; } catch(e) {}
                        }
                    }
                }
            }
            return null;
        }
    """)
    print(f"    Clicou: {clicked}")

    # Espera betslip e captura
    await page.wait_for_timeout(4000)

    # === MOSTRA HEADERS ===
    print(f"\n{'=' * 70}")
    print(f"HEADERS CAPTURADOS ({len(betslip_requests)} requests)")
    print(f"{'=' * 70}")

    for req in betslip_requests:
        print(f"\n  [{req['method']}] {req['url']}")
        print(f"  POST data: {req['post_data']}")
        print(f"\n  HEADERS:")
        for key, value in sorted(req['headers'].items()):
            # Mascara cookies longos
            if key.lower() == 'cookie' and len(value) > 100:
                print(f"    {key}: {value[:80]}... ({len(value)} chars)")
            else:
                print(f"    {key}: {value}")

    # Captura cookies do contexto
    cookies = await context.cookies()
    print(f"\n{'=' * 70}")
    print(f"COOKIES DO CONTEXTO ({len(cookies)})")
    print(f"{'=' * 70}")
    for c in cookies:
        if 'betinasia' in c.get('domain', ''):
            print(f"  {c['name']}: {c['value'][:50]}... (domain={c['domain']}, httpOnly={c.get('httpOnly')})")

    # Salva tudo
    output = {
        'betslip_requests': betslip_requests,
        'cookies': [{'name': c['name'], 'domain': c['domain'], 'httpOnly': c.get('httpOnly'),
                     'value_preview': c['value'][:30]} for c in cookies if 'betinasia' in c.get('domain', '')],
    }
    with open('headers_capture.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSalvo em: headers_capture.json")

    print(f"\n{'=' * 70}")
    input("Pressione Enter para fechar...")

    await browser.close()
    await p.stop()


if __name__ == "__main__":
    asyncio.run(main())
