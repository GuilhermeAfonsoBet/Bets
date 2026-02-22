#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descoberta: existe Lay via WebSocket "puro" (sem /v1/betslips/)?

Objetivo
  - Capturar URLs de WebSockets abertas pelo site
  - Contabilizar tipos de mensagens recebidas/enviadas
  - Inspecionar chaves presentes em mensagens `offers*`
  - (Opcional) clicar em uma odd, tentar alternar para "Exchange" e "Lay"

Saída
  - Um JSON com amostras e estatísticas (sem tokens/sessão em claro)

Uso (no VPS, após configurar BETINASIA_USERNAME/BETINASIA_PASSWORD em betinasia_bot/.env):
  DISPLAY=:99 python3 betinasia_bot/discover_lay_websocket.py --out logs/lay_ws_discovery.json
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger

sys.path.insert(0, ".")

from scraper.betinasia import BetinAsiaScraper


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mask_ws_url(url: str) -> str:
    """
    Remove token de URLs de WS:
      wss://.../cpricefeed/?token=XXX&lang=en  -> token=***
    """
    if not isinstance(url, str) or not url:
        return ""
    url = re.sub(r"(token=)[^&]+", r"\1***", url, flags=re.IGNORECASE)
    return url


def _safe_payload(frame: Any) -> Optional[str]:
    """
    Playwright pode entregar `WebSocketFrame`, dict, bytes ou string.
    Retorna uma string (limpa) ou None.
    """
    try:
        payload = None
        if hasattr(frame, "payload"):
            payload = frame.payload
        elif isinstance(frame, dict) and "payload" in frame:
            payload = frame.get("payload")
        else:
            payload = frame
        if payload is None:
            return None
        if isinstance(payload, (bytes, bytearray)):
            try:
                return payload.decode("utf-8", errors="replace")
            except Exception:
                return str(payload)
        return str(payload)
    except Exception:
        return None


def _redact_post_data(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return raw
    s = str(raw)
    # Sanitiza possíveis campos sensíveis se aparecerem em algum payload
    s = re.sub(r'("sessionToken"\s*:\s*")[^"]+(")', r'\1***\2', s, flags=re.IGNORECASE)
    s = re.sub(r'("session"\s*:\s*")[^"]+(")', r"\1***\2", s, flags=re.IGNORECASE)
    s = re.sub(r"(root-session=)[^;]+", r"\1***", s, flags=re.IGNORECASE)
    return s


def _truncate(s: str, n: int = 800) -> str:
    s = s if isinstance(s, str) else str(s)
    if len(s) <= n:
        return s
    return s[:n] + "…"


@dataclass
class WsStats:
    url_masked: str
    connected_at: str = field(default_factory=_utc_now_iso)
    recv_count: int = 0
    sent_count: int = 0
    msg_type_counts: Counter = field(default_factory=Counter)  # top-level item[0]
    api_subtype_counts: Counter = field(default_factory=Counter)  # ex.: pmm, betslip
    offers_keys: Set[str] = field(default_factory=set)
    offers_nested_keys: Set[str] = field(default_factory=set)
    snippets_with_lay: List[dict] = field(default_factory=list)  # amostras de frames com 'lay'
    recv_samples: List[dict] = field(default_factory=list)
    sent_samples: List[dict] = field(default_factory=list)


class LayWsDiscovery:
    def __init__(self, sample_limit: int = 30):
        self.sample_limit = max(5, int(sample_limit))
        self.phase: str = "baseline"

        self.ws_by_url: Dict[str, WsStats] = {}
        self.http_reqs: List[dict] = []
        self.http_reqs_phase_index: Dict[str, int] = {}

        # globais
        self.global_msg_types: Counter = Counter()
        self.global_api_subtypes: Counter = Counter()
        self.global_offers_keys: Set[str] = set()
        self.global_offers_nested_keys: Set[str] = set()

        self._start_ts = time.time()

    def _ensure_ws(self, ws_url: str) -> WsStats:
        k = _mask_ws_url(ws_url)
        if k not in self.ws_by_url:
            self.ws_by_url[k] = WsStats(url_masked=k)
        return self.ws_by_url[k]

    def on_websocket(self, ws):
        url = getattr(ws, "url", "") or ""
        stats = self._ensure_ws(url)

        def _on_recv(frame):
            payload = _safe_payload(frame)
            if payload is None:
                return
            stats.recv_count += 1
            if len(stats.recv_samples) < self.sample_limit:
                stats.recv_samples.append(
                    {"ts": _utc_now_iso(), "phase": self.phase, "payload": _truncate(payload, 1200)}
                )
            self._process_ws_payload(payload, stats, direction="recv")

        def _on_sent(frame):
            payload = _safe_payload(frame)
            if payload is None:
                return
            stats.sent_count += 1
            if len(stats.sent_samples) < self.sample_limit:
                stats.sent_samples.append(
                    {"ts": _utc_now_iso(), "phase": self.phase, "payload": _truncate(payload, 1200)}
                )
            self._process_ws_payload(payload, stats, direction="sent")

        try:
            ws.on("framereceived", _on_recv)
        except Exception:
            pass
        try:
            ws.on("framesent", _on_sent)
        except Exception:
            pass

    def on_request(self, req):
        try:
            url = str(getattr(req, "url", "") or "")
            method = str(getattr(req, "method", "") or "")
            post_data = None
            try:
                post_data = req.post_data
            except Exception:
                post_data = None
            entry = {
                "ts": _utc_now_iso(),
                "phase": self.phase,
                "method": method,
                "url": url.split("#")[0],
                "post_data": _truncate(_redact_post_data(post_data) or "", 1000) if post_data else "",
            }

            # filtra ruído (assets)
            low = url.lower()
            if any(ext in low for ext in (".png", ".jpg", ".jpeg", ".svg", ".gif", ".ico", ".css", ".woff", ".woff2")):
                return
            if any(x in low for x in ("google", "analytics", "gtm", "doubleclick")):
                return

            # só mantém endpoints potencialmente úteis
            if ("/v1/" in low) or ("/api/" in low) or ("betslip" in low) or ("exchange" in low):
                self.http_reqs.append(entry)
        except Exception:
            return

    def _process_ws_payload(self, payload: str, stats: WsStats, direction: str):
        p_low = payload.lower()
        # Snippet rápido para qualquer coisa com "lay"
        if "lay" in p_low and len(stats.snippets_with_lay) < self.sample_limit:
            stats.snippets_with_lay.append(
                {"ts": _utc_now_iso(), "phase": self.phase, "direction": direction, "payload": _truncate(payload, 1200)}
            )

        try:
            msg = json.loads(payload)
        except Exception:
            return

        if not isinstance(msg, list):
            return

        for item in msg:
            if not isinstance(item, list) or not item:
                continue

            msg_type = item[0]
            if isinstance(msg_type, str):
                stats.msg_type_counts[msg_type] += 1
                self.global_msg_types[msg_type] += 1

            # Offers: tentar extrair chaves
            if isinstance(msg_type, str) and msg_type.startswith("offers"):
                if len(item) >= 3 and isinstance(item[2], dict):
                    for k in item[2].keys():
                        stats.offers_keys.add(str(k))
                        self.global_offers_keys.add(str(k))
                    # inspeciona 1 nível adicional (ex.: data["ah"], data["wdw"] etc.)
                    for k, v in (item[2] or {}).items():
                        if isinstance(v, dict):
                            for kk in v.keys():
                                stats.offers_nested_keys.add(f"{k}.{kk}")
                                self.global_offers_nested_keys.add(f"{k}.{kk}")

            # Canal "api": subtipos (pmm/betslip)
            if msg_type == "api" and len(item) >= 2 and isinstance(item[1], dict):
                api_data = item[1].get("data", [])
                if isinstance(api_data, list):
                    for entry in api_data:
                        if isinstance(entry, list) and len(entry) >= 2:
                            sub = entry[0]
                            if isinstance(sub, str):
                                stats.api_subtype_counts[sub] += 1
                                self.global_api_subtypes[sub] += 1

    async def click_flow(self, page) -> dict:
        """
        Faz um fluxo "best effort":
          - navega para uma página de jogo
          - tenta clicar numa odd (abre betslip)
          - tenta alternar para Exchange e depois Lay
        """
        out: Dict[str, Any] = {"game_url": "", "clicked": None, "exchange_clicked": False, "lay_clicked": False}

        base_url = "https://black.betinasia.com"
        await page.goto(f"{base_url}/sportsbook/football")
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(3500)

        game_link = await page.evaluate(
            """
            () => {
              const links = Array.from(document.querySelectorAll('a'));
              for (const link of links) {
                const href = link.getAttribute('href') || '';
                if (href.includes('/sportsbook/football/') && href.includes(',') && !href.includes('/sportsbook/football/')) {
                  // ignorar link "root"
                }
              }
              for (const link of links) {
                const href = link.getAttribute('href') || '';
                if (href.includes('/sportsbook/football/') && href.includes(',')) return href;
              }
              return null;
            }
            """
        )

        if not game_link:
            return out

        game_url = f"{base_url}{game_link}" if str(game_link).startswith("/") else str(game_link)
        out["game_url"] = game_url
        await page.goto(game_url)
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(3500)

        # tenta expandir linhas (não falha se não existir)
        try:
            await page.evaluate(
                """
                () => {
                  for (const el of document.querySelectorAll('span, button, div')) {
                    const t = (el.innerText || '').trim().toLowerCase();
                    if ((t === 'show all lines' || t === 'show all') && el.offsetParent !== null) {
                      try { el.click(); } catch(e) {}
                    }
                  }
                }
                """
            )
            await page.wait_for_timeout(1200)
        except Exception:
            pass

        self.phase = "after_click_attempt"

        clicked = await page.evaluate(
            """
            () => {
              // Estratégia: procurar por um elemento com cara de odd decimal e clicar.
              const candidates = [];
              const els = Array.from(document.querySelectorAll('span, div, button'));
              for (const el of els) {
                const t = (el.innerText || '').trim();
                if (!t || t.length > 6) continue;
                if (!t.includes('.')) continue;
                const v = parseFloat(t);
                if (!isFinite(v)) continue;
                if (v < 1.05 || v > 20.0) continue;
                const r = el.getBoundingClientRect();
                if (!r || r.width < 18 || r.height < 10) continue;
                if (r.width > 240 || r.height > 80) continue;
                candidates.push({t, el});
              }
              // tenta clicar nos primeiros 20 candidatos
              for (let i = 0; i < Math.min(20, candidates.length); i++) {
                const el = candidates[i].el;
                try {
                  el.scrollIntoView({behavior: 'instant', block: 'center'});
                } catch(e) {}
                try { el.click(); return {odd: candidates[i].t, method: 'direct'}; } catch(e) {}
                try { el.parentElement && el.parentElement.click(); return {odd: candidates[i].t, method: 'parent'}; } catch(e) {}
              }
              return null;
            }
            """
        )

        out["clicked"] = clicked
        await page.wait_for_timeout(4000)

        # tenta alternar Exchange / Lay (best effort)
        self.phase = "after_exchange_attempt"
        exchange_clicked = False
        lay_clicked = False

        for sel in ("text=Exchange", "button:has-text('Exchange')", "span:has-text('Exchange')"):
            try:
                await page.click(sel, timeout=2500)
                exchange_clicked = True
                break
            except Exception:
                continue

        await page.wait_for_timeout(2500)

        for sel in ("text=Lay", "button:has-text('Lay')", "span:has-text('Lay')"):
            try:
                await page.click(sel, timeout=2500)
                lay_clicked = True
                break
            except Exception:
                continue

        out["exchange_clicked"] = exchange_clicked
        out["lay_clicked"] = lay_clicked
        await page.wait_for_timeout(2500)
        return out

    def build_report(self, click_info: dict) -> dict:
        elapsed = time.time() - self._start_ts
        ws_summary = []
        for url, st in self.ws_by_url.items():
            ws_summary.append(
                {
                    "url": url,
                    "connected_at": st.connected_at,
                    "recv_count": st.recv_count,
                    "sent_count": st.sent_count,
                    "msg_types_top": st.msg_type_counts.most_common(30),
                    "api_subtypes_top": st.api_subtype_counts.most_common(30),
                    "offers_keys": sorted(st.offers_keys),
                    "offers_nested_keys": sorted(st.offers_nested_keys),
                    "snippets_with_lay": st.snippets_with_lay,
                    "recv_samples": st.recv_samples,
                    "sent_samples": st.sent_samples,
                }
            )

        return {
            "ts": _utc_now_iso(),
            "elapsed_sec": round(elapsed, 3),
            "click_flow": click_info,
            "http_reqs": self.http_reqs,
            "global": {
                "msg_types_top": self.global_msg_types.most_common(50),
                "api_subtypes_top": self.global_api_subtypes.most_common(50),
                "offers_keys": sorted(self.global_offers_keys),
                "offers_nested_keys": sorted(self.global_offers_nested_keys),
            },
            "websockets": ws_summary,
        }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/lay_ws_discovery.json", help="Caminho do JSON de saída.")
    ap.add_argument("--sample-limit", type=int, default=30, help="Máx. de amostras por WS (recv/sent/snippets).")
    ap.add_argument("--baseline-wait-sec", type=float, default=8.0, help="Tempo esperando apenas WS (sem cliques).")
    ap.add_argument("--do-click-flow", action="store_true", help="Tenta clicar em odd e alternar Exchange/Lay.")
    args = ap.parse_args()

    out_path = str(args.out)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    disc = LayWsDiscovery(sample_limit=int(args.sample_limit))
    scraper = BetinAsiaScraper()

    try:
        await scraper.start()
        ok = await scraper.login()
        if not ok:
            raise RuntimeError(
                "Login falhou. Configure BETINASIA_USERNAME/BETINASIA_PASSWORD em betinasia_bot/.env"
            )

        page = scraper._page
        page.on("websocket", disc.on_websocket)
        page.on("request", disc.on_request)

        disc.phase = "baseline"
        await page.goto("https://black.betinasia.com/sportsbook/football")
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(int(float(args.baseline_wait_sec) * 1000))

        click_info = {"skipped": True}
        if bool(args.do_click_flow):
            click_info = await disc.click_flow(page)

        report = disc.build_report(click_info)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"Salvo: {out_path}")
        logger.info(f"WS detectados: {len(report.get('websockets') or [])}")
        logger.info(f"Top msg types: {report.get('global', {}).get('msg_types_top')[:10]}")

    finally:
        try:
            await scraper.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())

