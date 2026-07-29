#!/usr/bin/env python3
from __future__ import annotations

import asyncio
from scraper.betinasia import BetinAsiaScraper
from config import settings


async def main() -> int:
    s = BetinAsiaScraper()
    await s.start()
    page = s._page
    reqs = []

    def on_request(r):
        u = r.url
        if any(x in u.lower() for x in ("login", "auth", "session", "token", "customers", "v1/")):
            reqs.append(("req", r.method, u[:180]))

    async def on_response(r):
        u = r.url
        if any(x in u.lower() for x in ("login", "auth", "session", "token", "customers", "v1/")):
            try:
                txt = await r.text()
            except Exception:
                txt = ""
            reqs.append(("resp", r.status, u[:180], txt[:160].replace("\n", " ")))

    page.on("request", on_request)
    page.on("response", lambda r: asyncio.create_task(on_response(r)))

    await page.goto(s.LOGIN_URL, wait_until="commit", timeout=120000)
    for _ in range(60):
        if await page.locator("input[type='password']").count() > 0:
            break
        await page.wait_for_timeout(1000)

    # fill all visible text/email + password
    user = str(settings.betinasia_username or "")
    pw = str(settings.betinasia_password or "")
    for loc in await page.locator("input[type='text'], input[type='email']").all():
        try:
            if await loc.is_visible():
                await loc.fill(user)
        except Exception:
            pass
    await page.locator("input[type='password']").first.fill(pw)
    # trigger events
    await page.evaluate(
        """
        () => {
          for (const i of document.querySelectorAll('input')) {
            i.dispatchEvent(new Event('input', {bubbles:true}));
            i.dispatchEvent(new Event('change', {bubbles:true}));
          }
        }
        """
    )
    await page.wait_for_timeout(500)
    await page.locator("button:has-text('Log In')").first.click()
    await page.wait_for_timeout(8000)
    print("url", page.url)
    print("has_root", await s._has_root_session_cookie())
    body = (await page.locator("body").inner_text())[:400]
    print("body", repr(body))
    print("net_events", len(reqs))
    for e in reqs[-40:]:
        print(e)
    await s.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
