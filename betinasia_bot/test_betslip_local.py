#!/usr/bin/env python3
"""
Teste local: verifica se betslip abre completo (All Bookies) no PC com IP residencial.

Abre o browser VISÍVEL, faz login, navega para um jogo, clica numa odd,
tira screenshot do betslip.

Uso (no PowerShell):
    cd C:\betinasia_test
    python test_betslip.py
"""

import asyncio
import time
from playwright.async_api import async_playwright


# === CREDENCIAIS ===
USERNAME = "JomanaSilva"
PASSWORD = "Jom1928@"

# URLs
BASE_URL = "https://black.betinasia.com"
LOGIN_URL = f"{BASE_URL}/login"
FOOTBALL_URL = f"{BASE_URL}/sportsbook/football"


async def main():
    print("=" * 60)
    print("TESTE LOCAL: Betslip com IP residencial")
    print("=" * 60)

    p = await async_playwright().start()

    # Browser VISÍVEL (headless=False) para você ver o que acontece
    browser = await p.chromium.launch(headless=False)
    context = await browser.new_context(viewport={"width": 1920, "height": 1080})
    page = await context.new_page()

    try:
        # === 1. LOGIN ===
        print("\n[1] Fazendo login...")
        await page.goto(LOGIN_URL)
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(2000)

        # Preenche credenciais
        username_input = await page.query_selector("input[type='text'], input[name*='user'], input[name*='email']")
        password_input = await page.query_selector("input[type='password']")

        if username_input and password_input:
            await username_input.fill(USERNAME)
            await password_input.fill(PASSWORD)
            await page.wait_for_timeout(500)

            # Clica no botão de login
            login_btn = await page.query_selector("button[type='submit'], button:has-text('Log'), button:has-text('Sign')")
            if login_btn:
                await login_btn.click()
                await page.wait_for_timeout(5000)
                print("    Login enviado, aguardando...")
            else:
                print("    Botao login nao encontrado, tente fazer login manualmente no browser")
                print("    (o browser está aberto, faça login e pressione Enter aqui)")
                input("    Pressione Enter após fazer login...")
        else:
            print("    Campos de login nao encontrados, tente manualmente")
            print("    (o browser está aberto, faça login e pressione Enter aqui)")
            input("    Pressione Enter após fazer login...")

        # Verifica login
        await page.wait_for_timeout(3000)
        current_url = page.url
        if "/login" in current_url:
            print("    Ainda na pagina de login. Faca login manualmente no browser.")
            input("    Pressione Enter após fazer login...")

        print("    Login OK!")

        # === 2. NAVEGA PARA FUTEBOL ===
        print("\n[2] Navegando para pagina de futebol...")
        await page.goto(FOOTBALL_URL)
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(3000)

        # Encontra o primeiro jogo com odds
        print("\n[3] Procurando um jogo com odds...")

        # Clica no primeiro jogo que tiver Asian Handicap
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
            game_url = f"{BASE_URL}{game_link}" if game_link.startswith("/") else game_link
            print(f"    Navegando para: {game_url}")
            await page.goto(game_url)
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(3000)
        else:
            print("    Nenhum jogo encontrado. Navegue manualmente para um jogo.")
            input("    Pressione Enter quando estiver na pagina de um jogo...")

        # === 3. ESPERA ASIAN HANDICAP CARREGAR ===
        print("\n[4] Esperando Asian Handicap carregar...")
        try:
            await page.wait_for_selector("text=Asian Handicap", timeout=10000)
            print("    Asian Handicap encontrado")
        except:
            print("    Timeout esperando Asian Handicap, continuando...")
        await page.wait_for_timeout(2000)

        # === 4. EXPANDE LINHAS ===
        print("\n[5] Expandindo linhas...")
        expanded = await page.evaluate("""
            () => {
                let clicked = 0;
                const els = document.querySelectorAll('span, button, div, a, [role="button"]');
                for (const el of els) {
                    const text = (el.innerText || '').trim().toLowerCase();
                    if ((text === 'show all lines' || text === 'show all') && el.offsetParent !== null) {
                        try { el.click(); clicked++; } catch(e) {}
                    }
                }
                return clicked;
            }
        """)
        print(f"    Expandiu {expanded} secoes")
        await page.wait_for_timeout(2000)

        # === 5. CLICA NUMA ODD DO ASIAN HANDICAP ===
        print("\n[6] Clicando numa odd do Asian Handicap...")

        clicked_odd = await page.evaluate("""
            () => {
                // Primeiro encontra a seção Asian Handicap
                let ahSection = null;
                const headers = document.querySelectorAll('div, span, h3, h4');
                for (const h of headers) {
                    const text = (h.innerText || '').trim();
                    if (text.includes('Asian Handicap') || text.includes('Handicap')) {
                        let parent = h.parentElement;
                        for (let i = 0; i < 10 && parent; i++) {
                            const pt = parent.innerText || '';
                            if (pt.includes('Home') && pt.includes('Away')) {
                                ahSection = parent;
                                break;
                            }
                            parent = parent.parentElement;
                        }
                        if (ahSection) break;
                    }
                }
                
                if (!ahSection) ahSection = document.body;
                
                // Dentro da seção AH, encontra a primeira odd clicável
                const allEls = ahSection.querySelectorAll('span, div');
                for (const el of allEls) {
                    const t = (el.innerText || '').trim();
                    // Odd: formato X.XXX (1.500 a 9.999)
                    if (/^[1-9]\\d*[.,]\\d{2,3}$/.test(t) && t.length < 8) {
                        const val = parseFloat(t.replace(',', '.'));
                        // Filtra: odds razoáveis (1.1 a 15.0)
                        if (val >= 1.1 && val <= 15.0) {
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

        if clicked_odd:
            print(f"    Clicou na odd: {clicked_odd['odd']} (metodo: {clicked_odd['method']})")
        else:
            print("    Nao encontrou odd AH para clicar. Clique manualmente numa odd no browser.")
            input("    Pressione Enter apos clicar numa odd...")

        # === 6. ESPERA BETSLIP E TIRA SCREENSHOT ===
        print("\n[7] Aguardando betslip abrir...")
        await page.wait_for_timeout(4000)

        # Screenshot da pagina inteira
        await page.screenshot(path="betslip_screenshot_full.png", full_page=False)
        print("    Screenshot salvo: betslip_screenshot_full.png")

        # Dump do texto do betslip
        betslip_text = await page.evaluate("""
            () => {
                const selectors = ['[class*="betslip"]', '[class*="slip"]', '[class*="sidebar"]', '[class*="panel"]', 'aside'];
                let results = [];
                for (const sel of selectors) {
                    for (const el of document.querySelectorAll(sel)) {
                        const t = (el.innerText || '').trim();
                        if (t.length > 20 && t.length < 5000) {
                            results.push({sel: sel, cls: el.className.substring(0, 60), text: t.substring(0, 600)});
                        }
                    }
                }
                return results;
            }
        """)

        print(f"\n    === BETSLIP DUMP ({len(betslip_text)} paineis) ===")
        for panel in betslip_text:
            print(f"\n    Selector: {panel['sel']}")
            print(f"    Class: {panel['cls']}")
            print(f"    Texto:")
            for line in panel['text'].split('\n')[:20]:
                print(f"      {line.strip()}")

        # Verifica se tem "All Bookies"
        all_text = " ".join([p['text'] for p in betslip_text])
        if "All Bookies" in all_text or "All bookies" in all_text or "Todos Os Agentes" in all_text:
            print("\n    ✅ 'All Bookies' ENCONTRADO — betslip completo!")
            print("    O problema é confirmado: IP da VPS está sendo bloqueado.")
        elif "VPN" in all_text:
            print("\n    ❌ Mensagem de VPN detectada MESMO no PC local!")
            print("    O problema pode não ser o IP da VPS.")
        elif "Betslip" in all_text:
            print("\n    ⚠️ Betslip encontrado mas sem 'All Bookies'.")
            print("    Verifique o screenshot para mais detalhes.")
        else:
            print("\n    ⚠️ Betslip não encontrado nos painéis.")
            print("    Verifique o screenshot.")

        print("\n" + "=" * 60)
        print("TESTE CONCLUIDO")
        print("=" * 60)
        print(f"\nScreenshot salvo em: C:\\betinasia_test\\betslip_screenshot_full.png")
        print("Abra o arquivo para ver o estado do betslip.")
        print("\nO browser continua aberto para você inspecionar manualmente.")
        input("\nPressione Enter para fechar o browser...")

    except Exception as e:
        print(f"\nErro: {e}")
        import traceback
        traceback.print_exc()
        input("\nPressione Enter para fechar...")

    finally:
        await browser.close()
        await p.stop()


if __name__ == "__main__":
    asyncio.run(main())
