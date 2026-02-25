from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

sys.path.insert(0, ".")

from scraper.betinasia import BetinAsiaScraper


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


SENSITIVE_HEADER_KEYS = {
    "cookie",
    "authorization",
    "proxy-authorization",
    "session",
    "x-xsrf-token",
    "x-csrf-token",
    "x-auth-token",
}


def _redact_headers(h: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (h or {}).items():
        lk = str(k).lower()
        if lk in SENSITIVE_HEADER_KEYS:
            out[k] = "***"
        else:
            out[k] = v
    return out


def _redact_text(s: Optional[str]) -> Optional[str]:
    if not s:
        return s
    t = str(s)
    # session token / cookies
    t = re.sub(r'(root-session=)[^;"]+', r"\1***", t, flags=re.IGNORECASE)
    t = re.sub(r'("sessionToken"\s*:\s*")[^"]+(")', r"\1***\2", t, flags=re.IGNORECASE)
    t = re.sub(r'("session"\s*:\s*")[^"]+(")', r"\1***\2", t, flags=re.IGNORECASE)
    t = re.sub(r"(\bsession=)[^&\s]+", r"\1***", t, flags=re.IGNORECASE)
    # password fields if any
    t = re.sub(r'("password"\s*:\s*")[^"]+(")', r"\1***\2", t, flags=re.IGNORECASE)
    return t


def _truncate(s: Optional[str], n: int = 900) -> Optional[str]:
    if s is None:
        return None
    s = str(s)
    if len(s) <= n:
        return s
    return s[:n] + "…"


def _mask_url(url: str) -> str:
    if not url:
        return ""
    u = str(url)
    u = re.sub(r"(token=)[^&]+", r"\1***", u, flags=re.IGNORECASE)
    return u


def _is_relevant_url(url: str) -> bool:
    u = (url or "").lower()
    if "/v1/" in u:
        return True
    keys = ("betslip", "bet", "order", "place", "confirm", "ticket", "slip", "wager")
    return any(k in u for k in keys)


def _is_relevant_ws_payload(s: str) -> bool:
    if not s:
        return False
    u = s.lower()
    keys = ("betslip", "pmm", "bet", "order", "place", "confirm", "accepted", "rejected", "error")
    return any(k in u for k in keys)


@dataclass
class JsonlWriter:
    path: Path
    _fh: Any = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def open(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    async def close(self):
        try:
            if self._fh:
                self._fh.close()
        except Exception:
            pass
        self._fh = None

    async def write(self, obj: Dict[str, Any]):
        if not self._fh:
            return
        line = json.dumps(obj, ensure_ascii=False)
        async with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()


async def capture_place_confirm(*, out_jsonl: Path, headless: bool, capture_all: bool) -> int:
    w = JsonlWriter(out_jsonl)
    await w.open()

    req_cnt = Counter()
    resp_cnt = Counter()
    ws_cnt = Counter()

    # Start browser
    scraper = BetinAsiaScraper(headless=headless)
    await scraper.start()
    ok_login = await scraper.login()
    if not ok_login:
        logger.error("Login falhou. Verifique BETINASIA_USERNAME/BETINASIA_PASSWORD em `.env`.")
        await scraper.close()
        await w.close()
        return 2

    page = scraper._page
    await page.goto(BetinAsiaScraper.FOOTBALL_URL)
    await page.wait_for_load_state("domcontentloaded")
    await page.wait_for_timeout(3500)

    pending_tasks: List[asyncio.Task] = []

    def on_request(request):
        try:
            url = str(request.url)
            if (not capture_all) and (not _is_relevant_url(url)):
                return
            item = {
                "type": "request",
                "ts": _utc_now_iso(),
                "url": _mask_url(url),
                "method": request.method,
                "resource_type": getattr(request, "resource_type", None),
                "headers": _redact_headers(dict(request.headers or {})),
                "post_data": _truncate(_redact_text(getattr(request, "post_data", None))),
            }
            req_cnt[(request.method, _mask_url(url).split("?", 1)[0])] += 1
            pending_tasks.append(asyncio.create_task(w.write(item)))
        except Exception:
            pass

    async def _capture_response(response):
        try:
            url = str(response.url)
            if (not capture_all) and (not _is_relevant_url(url)):
                return
            status = int(response.status)
            headers = {}
            try:
                headers = dict(response.headers or {})
            except Exception:
                headers = {}
            body_prefix = None
            if "/v1/" in (url or "").lower():
                try:
                    txt = await response.text()
                    body_prefix = _truncate(_redact_text(txt), 1200)
                except Exception:
                    body_prefix = None
            item = {
                "type": "response",
                "ts": _utc_now_iso(),
                "url": _mask_url(url),
                "status": status,
                "headers": _redact_headers(headers),
                "body_prefix": body_prefix,
            }
            resp_cnt[(status, _mask_url(url).split("?", 1)[0])] += 1
            await w.write(item)
        except Exception:
            return

    def on_response(response):
        try:
            pending_tasks.append(asyncio.create_task(_capture_response(response)))
        except Exception:
            pass

    def on_websocket(ws):
        ws_url = _mask_url(getattr(ws, "url", ""))

        def on_frame_received(data):
            try:
                s = str(data)
                if (not capture_all) and (not _is_relevant_ws_payload(s)):
                    return
                item = {
                    "type": "ws",
                    "ts": _utc_now_iso(),
                    "ws_url": ws_url,
                    "direction": "received",
                    "data": _truncate(_redact_text(s), 1200),
                }
                ws_cnt[("received", ws_url)] += 1
                pending_tasks.append(asyncio.create_task(w.write(item)))
            except Exception:
                pass

        def on_frame_sent(data):
            try:
                s = str(data)
                if (not capture_all) and (not _is_relevant_ws_payload(s)):
                    return
                item = {
                    "type": "ws",
                    "ts": _utc_now_iso(),
                    "ws_url": ws_url,
                    "direction": "sent",
                    "data": _truncate(_redact_text(s), 1200),
                }
                ws_cnt[("sent", ws_url)] += 1
                pending_tasks.append(asyncio.create_task(w.write(item)))
            except Exception:
                pass

        ws.on("framereceived", on_frame_received)
        ws.on("framesent", on_frame_sent)

    page.on("request", on_request)
    page.on("response", on_response)
    page.on("websocket", on_websocket)

    logger.info("=" * 80)
    logger.info("CAPTURA PLACE/CONFIRM — INSTRUÇÕES")
    logger.info("- No browser, selecione um jogo (AH), digite stake ~$3 e clique para confirmar/colocar aposta.")
    logger.info("- Volte ao terminal e aperte ENTER para finalizar a captura.")
    logger.info(f"- Saída: {out_jsonl}")
    logger.info("=" * 80)

    # Aguarda ENTER sem travar loop
    await asyncio.to_thread(input, "\nPressione ENTER após finalizar a aposta manual...\n")

    # aguarda flush dos tasks
    if pending_tasks:
        try:
            await asyncio.wait(pending_tasks, timeout=10.0)
        except Exception:
            pass

    # summary
    await w.write(
        {
            "type": "summary",
            "ts": _utc_now_iso(),
            "requests_top": req_cnt.most_common(30),
            "responses_top": resp_cnt.most_common(30),
            "ws_top": ws_cnt.most_common(30),
        }
    )

    await scraper.close()
    await w.close()
    logger.info(f"[OK] Captura concluída: {out_jsonl}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="", help="Arquivo JSONL de saída (default: logs/place_confirm_capture_*.jsonl)")
    parser.add_argument("--headless", action="store_true", help="Roda headless (não recomendado para captura manual).")
    parser.add_argument("--all", action="store_true", help="Captura todo tráfego (pode gerar arquivo grande).")
    args = parser.parse_args()

    if args.out:
        out = Path(str(args.out))
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out = Path("logs") / f"place_confirm_capture_{ts}.jsonl"

    return int(asyncio.run(capture_place_confirm(out_jsonl=out, headless=bool(args.headless), capture_all=bool(args.all))))


if __name__ == "__main__":
    raise SystemExit(main())

