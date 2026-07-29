#!/usr/bin/env python3
from __future__ import annotations

import asyncio
from scraper.betinasia import BetinAsiaScraper
from config import settings


async def main() -> int:
    s = BetinAsiaScraper()
    await s.start()
    page = s._page
    events = []

    def on_request(r):
        events.append(("req", r.method, r.url[:200]))

    async def on_response(r):
        try:
            t = await r.text()
        except Exception:
            t = ""
        events.append(("resp", r.status, r.url[:200], t[:120].replace("\n", " ")))

    page.on("request", on_request)
    page.on("response", lambda r: asyncio.create_task(on_response(r)))

    await page.goto(s.LOGIN_URL, wait_until="commit", timeout=120000)
    # wait up to 3 min for Loading to clear and network quiet-ish
    ready = False
    for i in range(90):
        body = (await page.locator("body").inner_text())[:300]
        low = body.lower()
        has_loading = "loading..." in low
        n_pass = await page.locator("input[type='password']").count()
        print(f"i={i} loading={has_loading} pass={n_pass} events={len(events)} url={page.url}")
        if n_pass > 0 and not has_loading:
            ready = True
            break
        # also try networkidle briefly
        try:
            await page.wait_for_load_state("networkidle", timeout=2000)
        except Exception:
            await page.wait_for_timeout(2000)

    print("ready", ready, "total_events", len(events))
    for e in events[-30:]:
        print(e)

    if not ready:
        # force wait more then proceed anyway
        await page.wait_for_timeout(5000)

    user = str(settings.betinasia_username or "")
    pw = str(settings.betinasia_password or "")
    await page.locator("input[type='text'], input[type='email']").first.fill(user)
    await page.locator("input[type='password']").first.fill(pw)
    await page.evaluate(
        """() => { for (const i of document.querySelectorAll('input')) {
          i.dispatchEvent(new Event('input',{bubbles:true}));
          i.dispatchEvent(new Event('change',{bubbles:true}));
        }}"""
    )
    before = len(events)
    await page.locator("button:has-text('Log In')").first.click()
    await page.wait_for_timeout(15000)
    print("after_click_new_events", len(events) - before)
    for e in events[before:]:
        print(e)
    print("url", page.url, "has_root", await s._has_root_session_cookie())
    print("body", repr((await page.locator("body").inner_text())[:300]))
    await s.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
