#!/usr/bin/env python3
from __future__ import annotations

import asyncio
from pathlib import Path

from scraper.betinasia import BetinAsiaScraper
from config import settings


async def main() -> int:
    s = BetinAsiaScraper()
    await s.start()
    page = s._page
    await page.goto(s.LOGIN_URL, wait_until="commit", timeout=120000)
    # wait form
    for _ in range(80):
        n = await page.locator("input[type='password']").count()
        if n > 0:
            break
        await page.wait_for_timeout(1000)
    # wait loading text gone best-effort
    for _ in range(30):
        txt = (await page.locator("body").inner_text())[:300].lower()
        if "loading" not in txt or ("username" in txt and "password" in txt):
            # still may show Loading... overlay
            pass
        # if Log In button enabled
        btn = page.locator("button:has-text('Log In'), button:has-text('log In'), button:has-text('Login')")
        if await btn.count() > 0:
            break
        await page.wait_for_timeout(1000)

    user = str(settings.betinasia_username or "")
    pw = str(settings.betinasia_password or "")
    # fill by placeholder/name
    user_loc = page.locator("input[type='text'], input[type='email'], input[name='username'], input[placeholder*='user' i]").first
    pass_loc = page.locator("input[type='password']").first
    await user_loc.click()
    await user_loc.fill(user)
    await page.wait_for_timeout(400)
    await pass_loc.click()
    await pass_loc.fill(pw)
    await page.wait_for_timeout(400)
    # click
    clicked = False
    for sel in ["button:has-text('Log In')", "button:has-text('log In')", "button:has-text('Login')", "button[type='submit']"]:
        loc = page.locator(sel)
        if await loc.count() > 0:
            await loc.first.click()
            clicked = True
            print("clicked", sel)
            break
    if not clicked:
        await page.keyboard.press("Enter")
        print("pressed_enter")
    # wait root-session
    for i in range(40):
        has = await s._has_root_session_cookie()
        url = page.url
        print(f"t={i} has_root={has} url={url}")
        if has:
            await s.save_session()
            # verify not saving empty root
            import json
            d = json.loads(Path("betinasia_session.json").read_text())
            print("saved_root", any(c.get("name") == "root-session" for c in d.get("cookies") or []))
            chk = await s._api_auth_check_orders(username=user)
            print("api", chk.get("ok"), chk.get("status"), str(chk.get("prefix") or "")[:100])
            await s.close()
            return 0
        await page.wait_for_timeout(1000)
    # dump errors
    body = (await page.locator("body").inner_text())[:500]
    print("FAIL_BODY", repr(body))
    await page.screenshot(path="logs/login_manual_fail.png", full_page=True)
    await s.close()
    return 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
