#!/usr/bin/env python3
"""One-shot accounting session restore probe (no secrets printed)."""
from __future__ import annotations

import asyncio
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from scraper.betinasia import BetinAsiaScraper
from config import settings


async def wait_for_login_ready(page, *, timeout_ms: int = 120000) -> dict:
    t0 = asyncio.get_event_loop().time()
    last = {}
    while (asyncio.get_event_loop().time() - t0) * 1000 < timeout_ms:
        try:
            info = await page.evaluate(
                """
                () => {
                  const inputs = [...document.querySelectorAll('input')].map(i => ({
                    type: i.type, name: i.name, id: i.id, placeholder: i.placeholder,
                    visible: !!(i.offsetWidth || i.offsetHeight)
                  }));
                  const body = (document.body && document.body.innerText || '').slice(0, 200);
                  const hasPass = inputs.some(i => i.type === 'password' && i.visible);
                  const hasUser = inputs.some(i => (i.type === 'text' || i.type === 'email') && i.visible);
                  return {url: location.href, hasPass, hasUser, nInputs: inputs.length, body};
                }
                """
            )
            last = info if isinstance(info, dict) else {"raw": str(info)}
            if last.get("hasPass") and last.get("hasUser"):
                return {"ok": True, **last}
            if "sportsbook" in str(last.get("url") or "").lower() or "trade" in str(last.get("url") or "").lower():
                return {"ok": True, "already": True, **last}
        except Exception as e:
            last = {"error": str(e)[:160]}
        await page.wait_for_timeout(1500)
    return {"ok": False, "timeout": True, **last}


async def main() -> int:
    sess = Path("betinasia_session.json")
    bak = Path(f"betinasia_session.json.bak_phase2a_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    if sess.exists():
        shutil.copy2(sess, bak)
        print("backed_up", bak)

    print("user_set", bool(settings.betinasia_username), "pass_set", bool(settings.betinasia_password))
    scraper = BetinAsiaScraper()
    await scraper.start()
    # navigate login and wait for SPA
    await scraper._page.goto(scraper.LOGIN_URL, wait_until="commit", timeout=120000)
    ready = await wait_for_login_ready(scraper._page, timeout_ms=120000)
    print("ready", {k: ready.get(k) for k in ready if k != "body"}, "body", repr(ready.get("body")))
    ok = await scraper.login(force=True)
    has_root = await scraper._has_root_session_cookie()
    print("login_ok", ok, "has_root", has_root)
    if ok and has_root:
        await scraper.save_session()
        # verify cookies file
        d = json.loads(sess.read_text())
        names = [c.get("name") for c in d.get("cookies") or []]
        print("saved_has_root", "root-session" in names, "n_cookies", len(names))
        chk = await scraper._api_auth_check_orders(username=str(settings.betinasia_username or ""))
        print("api", {"ok": chk.get("ok"), "status": chk.get("status"), "prefix": str(chk.get("prefix") or "")[:120]})
        # try balance api
        from ops.accounting_monitor import _download_via_api, _api_balance_urls
        urls = _api_balance_urls(str(settings.betinasia_username or ""))
        p, meta = await _download_via_api(scraper, name="balance", url=urls["balance"], out_dir=Path("logs/accounting"), timeout_ms=90000)
        print("balance_api", bool(p), meta.get("downloaded_via"), meta.get("http_status"), meta.get("error_type"), meta.get("error"))
        if p:
            print("balance_path", p, "size", p.stat().st_size)
        p2, meta2 = await _download_via_api(scraper, name="open_stakes", url=urls["open_stakes"], out_dir=Path("logs/accounting"), timeout_ms=90000)
        print("open_api", bool(p2), meta2.get("downloaded_via"), meta2.get("http_status"), meta2.get("error_type"), meta2.get("error"))
        if p2:
            print("open_path", p2, "size", p2.stat().st_size)
    else:
        # restore previous file if we didn't get root
        if bak.exists():
            shutil.copy2(bak, sess)
            print("restored_previous_session_file")
    await scraper.close()
    return 0 if ok and has_root else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
