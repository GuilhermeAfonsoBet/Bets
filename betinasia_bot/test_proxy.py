#!/usr/bin/env python3
"""Teste mínimo: Playwright + Bright Data proxy."""

import asyncio
from playwright.async_api import async_playwright

PROXY = {
    "server": "http://brd.superproxy.io:22225",
    "username": "brd-customer-hl_58c8955f-zone-residential_proxy1",
    "password": "gcicd6bbsw90",
}

async def test():
    async with async_playwright() as p:
        
        # Teste 1: proxy no browser launch
        print("=== Teste 1: proxy no browser launch ===")
        try:
            browser = await p.chromium.launch(headless=True, proxy=PROXY)
            page = await browser.new_page(ignore_https_errors=True)
            await page.goto("https://lumtest.com/myip.json", timeout=15000)
            print(f"OK: {await page.inner_text('body')}")
            await browser.close()
        except Exception as e:
            print(f"FALHOU: {e}")
            try: await browser.close()
            except: pass
        
        # Teste 2: proxy no browser launch + HTTP (não HTTPS)
        print("\n=== Teste 2: proxy com site HTTP ===")
        try:
            browser = await p.chromium.launch(headless=True, proxy=PROXY)
            page = await browser.new_page(ignore_https_errors=True)
            await page.goto("http://lumtest.com/myip.json", timeout=15000)
            print(f"OK: {await page.inner_text('body')}")
            await browser.close()
        except Exception as e:
            print(f"FALHOU: {e}")
            try: await browser.close()
            except: pass
        
        # Teste 3: proxy no context (não no browser)
        print("\n=== Teste 3: proxy no context ===")
        try:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                proxy=PROXY,
                ignore_https_errors=True,
            )
            page = await context.new_page()
            await page.goto("https://lumtest.com/myip.json", timeout=15000)
            print(f"OK: {await page.inner_text('body')}")
            await browser.close()
        except Exception as e:
            print(f"FALHOU: {e}")
            try: await browser.close()
            except: pass
        
        # Teste 4: proxy via chromium args
        print("\n=== Teste 4: proxy via --proxy-server arg ===")
        try:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--proxy-server=http://brd.superproxy.io:22225",
                    "--ignore-certificate-errors",
                ]
            )
            context = await browser.new_context(
                ignore_https_errors=True,
                http_credentials={
                    "username": PROXY["username"],
                    "password": PROXY["password"],
                }
            )
            page = await context.new_page()
            await page.goto("https://lumtest.com/myip.json", timeout=15000)
            print(f"OK: {await page.inner_text('body')}")
            await browser.close()
        except Exception as e:
            print(f"FALHOU: {e}")
            try: await browser.close()
            except: pass
        
        # Teste 5: proxy via chromium args + proxy auth header
        print("\n=== Teste 5: proxy via args + route auth ===")
        try:
            import base64
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--proxy-server=http://brd.superproxy.io:22225",
                    "--ignore-certificate-errors",
                ]
            )
            context = await browser.new_context(ignore_https_errors=True)
            page = await context.new_page()
            
            # Intercepta requests para adicionar Proxy-Authorization header
            auth = base64.b64encode(f"{PROXY['username']}:{PROXY['password']}".encode()).decode()
            await page.set_extra_http_headers({"Proxy-Authorization": f"Basic {auth}"})
            
            await page.goto("https://lumtest.com/myip.json", timeout=15000)
            print(f"OK: {await page.inner_text('body')}")
            await browser.close()
        except Exception as e:
            print(f"FALHOU: {e}")
            try: await browser.close()
            except: pass

asyncio.run(test())
