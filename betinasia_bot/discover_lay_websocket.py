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

async def _try_click_any(page, selectors: List[str], *, timeout_ms: int = 2500) -> Tuple[bool, str]:
    """
    Tenta clicar em um dos seletores (em ordem). Retorna (ok, selector_usado).
    Usa `page.click` porque é simples; se falhar, tenta `locator().first.click()`.
    """
    for sel in selectors:
        try:
            await page.click(sel, timeout=timeout_ms)
            return True, sel
        except Exception:
            pass
        try:
            loc = page.locator(sel)
            if await loc.count() > 0:
                await loc.first.click(timeout=timeout_ms)
                return True, sel
        except Exception:
            continue
    return False, ""


async def _screenshot(page, path: str) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        await page.screenshot(path=path, full_page=True)
    except Exception:
        return

async def _get_root_session_token(page) -> str:
    """
    Lê o cookie HttpOnly `root-session` via Playwright, para usar como header `session`
    em endpoints que exigem autenticação.
    """
    try:
        cookies = await page.context.cookies()
    except Exception:
        return ""
    for c in cookies or []:
        try:
            if c.get("name") == "root-session" and isinstance(c.get("value"), str) and c.get("value"):
                return c["value"]
        except Exception:
            continue
    return ""


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
    pmm_bet_prefix_counts: Counter = field(default_factory=Counter)  # for vs against
    pmm_bet_type_counts: Counter = field(default_factory=Counter)  # bet_type completo
    pmm_against_samples: List[dict] = field(default_factory=list)  # amostras de PMM com against, se existir
    pmm_for_samples: List[dict] = field(default_factory=list)  # amostras de PMM com for
    trade_keyword_snippets: List[dict] = field(default_factory=list)  # snippets com palavras de trade (order/bid/ask)
    recv_samples: List[dict] = field(default_factory=list)
    sent_samples: List[dict] = field(default_factory=list)


class LayWsDiscovery:
    def __init__(self, sample_limit: int = 30):
        self.sample_limit = max(5, int(sample_limit))
        self.phase: str = "baseline"

        self.ws_by_url: Dict[str, WsStats] = {}
        self.http_reqs: List[dict] = []
        self.http_resps: List[dict] = []
        self._resp_tasks: List[asyncio.Task] = []
        self.response_hook_errors: int = 0
        self.response_hook_seen: int = 0
        self.response_hook_kept: int = 0
        self.http_reqs_phase_index: Dict[str, int] = {}

        # globais
        self.global_msg_types: Counter = Counter()
        self.global_api_subtypes: Counter = Counter()
        self.global_offers_keys: Set[str] = set()
        self.global_offers_nested_keys: Set[str] = set()
        self.global_pmm_bet_prefix_counts: Counter = Counter()
        self.global_pmm_against_samples: List[dict] = []
        self.global_pmm_bet_type_counts: Counter = Counter()

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
            rtype = ""
            try:
                rtype = str(getattr(req, "resource_type", "") or "")
            except Exception:
                rtype = ""
            post_data = None
            try:
                post_data = req.post_data
            except Exception:
                post_data = None
            entry = {
                "ts": _utc_now_iso(),
                "phase": self.phase,
                "method": method,
                "resource_type": rtype,
                "url": url.split("#")[0],
                "post_data": _truncate(_redact_post_data(post_data) or "", 1000) if post_data else "",
            }

            # filtra ruído (assets)
            low = url.lower()
            if any(ext in low for ext in (".png", ".jpg", ".jpeg", ".svg", ".gif", ".ico", ".css", ".woff", ".woff2")):
                return
            if any(x in low for x in ("google", "analytics", "gtm", "doubleclick")):
                return

            # Mantém endpoints potencialmente úteis:
            # - Qualquer XHR/fetch do domínio (Trade costuma usar rotas específicas)
            # - E também /v1, /api, betslip, exchange
            keep = False
            if rtype in ("xhr", "fetch"):
                if "black.betinasia.com" in low:
                    keep = True
            if ("/v1/" in low) or ("/api/" in low) or ("betslip" in low) or ("exchange" in low) or ("/trade" in low):
                keep = True
            if keep:
                self.http_reqs.append(entry)
        except Exception:
            return

    def on_response(self, resp):
        """
        Captura respostas XHR/fetch úteis do domínio.
        Observação: handler do Playwright não é async; então fazemos create_task.
        """
        try:
            self.response_hook_seen += 1
            # Limita volume
            if len(self.http_resps) >= 60:
                return
            url = str(getattr(resp, "url", "") or "")
            low = url.lower()
            if "black.betinasia.com" not in low:
                return
            # Evita assets
            if any(ext in low for ext in (".png", ".jpg", ".jpeg", ".svg", ".gif", ".ico", ".css", ".woff", ".woff2")):
                return
            req = getattr(resp, "request", None)
            rtype = ""
            try:
                if req:
                    rtype = str(getattr(req, "resource_type", "") or "")
            except Exception:
                rtype = ""
            if rtype not in ("xhr", "fetch"):
                # ainda assim, captura /web/sessions/* que pode ser long-poll
                if "/web/sessions/" not in low:
                    return

            # Marca que passou no filtro e salva um “mínimo viável” (sem await)
            self.response_hook_kept += 1
            try:
                status = int(getattr(resp, "status", 0) or 0)
            except Exception:
                status = 0
            ctype = ""
            try:
                # response.headers é sync
                hdrs = getattr(resp, "headers", None) or {}
                if isinstance(hdrs, dict):
                    ctype = str(hdrs.get("content-type") or hdrs.get("Content-Type") or "")
            except Exception:
                ctype = ""
            self.http_resps.append(
                {
                    "ts": _utc_now_iso(),
                    "phase": self.phase,
                    "status": status,
                    "resource_type": rtype,
                    "url": url.split("#")[0],
                    "content_type": ctype,
                    "json_keys": [],
                    "body_prefix": "",
                    "mode": "min",
                }
            )

            async def _collect():
                try:
                    status = int(getattr(resp, "status", 0) or 0)
                except Exception:
                    status = 0
                ctype = ""
                try:
                    hdrs = await resp.all_headers()
                    ctype = str((hdrs or {}).get("content-type") or "")
                except Exception:
                    ctype = ""
                body_prefix = ""
                json_keys: List[str] = []
                try:
                    # Pode falhar se body não estiver disponível (stream)
                    txt = await asyncio.wait_for(resp.text(), timeout=0.6)
                    if txt:
                        body_prefix = _truncate(_redact_post_data(txt) or "", 1200)
                        try:
                            j = json.loads(txt)
                            if isinstance(j, dict):
                                json_keys = sorted([str(k) for k in j.keys()])[:60]
                        except Exception:
                            pass
                except Exception:
                    pass

                self.http_resps.append(
                    {
                        "ts": _utc_now_iso(),
                        "phase": self.phase,
                        "status": status,
                        "resource_type": rtype,
                        "url": url.split("#")[0],
                        "content_type": ctype,
                        "json_keys": json_keys,
                        "body_prefix": body_prefix,
                        "mode": "body",
                    }
                )

            try:
                try:
                    loop = asyncio.get_running_loop()
                except Exception:
                    loop = None
                if not loop:
                    self.response_hook_errors += 1
                    return
                t = loop.create_task(_collect())
                self._resp_tasks.append(t)
                # evita crescer sem limite
                if len(self._resp_tasks) > 120:
                    self._resp_tasks = [x for x in self._resp_tasks if not x.done()][-80:]
            except Exception:
                self.response_hook_errors += 1
                return
        except Exception:
            return

    async def flush_http_responses(self, timeout_sec: float = 2.0) -> None:
        """Aguarda tasks de coleta de responses (com timeout curto)."""
        tasks = [t for t in (self._resp_tasks or []) if t and not t.done()]
        if not tasks:
            return
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout_sec)
        except Exception:
            return

    def _process_ws_payload(self, payload: str, stats: WsStats, direction: str):
        p_low = payload.lower()
        # Snippet rápido para qualquer coisa com "lay"
        if "lay" in p_low and len(stats.snippets_with_lay) < self.sample_limit:
            stats.snippets_with_lay.append(
                {"ts": _utc_now_iso(), "phase": self.phase, "direction": direction, "payload": _truncate(payload, 1200)}
            )
        # Snippet de trade/orderbook: útil para /trade se o servidor embedar dados no WS
        if any(k in p_low for k in ("order", "bid", "ask", "orderbook", "trade")) and len(stats.trade_keyword_snippets) < 12:
            stats.trade_keyword_snippets.append(
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
                            # PMM: extrair bet_type e procurar "against" (Lay)
                            if sub == "pmm" and isinstance(entry[1], dict):
                                bt = entry[1].get("bet_type")
                                if isinstance(bt, str) and bt:
                                    stats.pmm_bet_type_counts[bt] += 1
                                    self.global_pmm_bet_type_counts[bt] += 1
                                    prefix = bt.split(",", 1)[0].strip().lower()
                                    if prefix:
                                        stats.pmm_bet_prefix_counts[prefix] += 1
                                        self.global_pmm_bet_prefix_counts[prefix] += 1
                                    if prefix == "against":
                                        if len(stats.pmm_against_samples) < 10:
                                            stats.pmm_against_samples.append(
                                                {
                                                    "ts": _utc_now_iso(),
                                                    "phase": self.phase,
                                                    "bet_type": bt,
                                                    "bookie": entry[1].get("bookie"),
                                                    "price0": (entry[1].get("price_list") or [{}])[0].get("effective", {}).get("price")
                                                    if isinstance(entry[1].get("price_list"), list) and entry[1].get("price_list")
                                                    else None,
                                                }
                                            )
                                        if len(self.global_pmm_against_samples) < 10:
                                            self.global_pmm_against_samples.append(
                                                {
                                                    "ts": _utc_now_iso(),
                                                    "phase": self.phase,
                                                    "bet_type": bt,
                                                    "bookie": entry[1].get("bookie"),
                                                }
                                            )
                                    if prefix == "for":
                                        if len(stats.pmm_for_samples) < 10:
                                            stats.pmm_for_samples.append(
                                                {
                                                    "ts": _utc_now_iso(),
                                                    "phase": self.phase,
                                                    "bet_type": bt,
                                                    "bookie": entry[1].get("bookie"),
                                                    "price0": (entry[1].get("price_list") or [{}])[0].get("effective", {}).get("price")
                                                    if isinstance(entry[1].get("price_list"), list) and entry[1].get("price_list")
                                                    else None,
                                                }
                                            )

    async def click_flow(self, page) -> dict:
        """
        Faz um fluxo "best effort":
          - navega para uma página de jogo
          - tenta clicar numa odd (abre betslip)
          - tenta alternar para Exchange e depois Lay
        """
        out: Dict[str, Any] = {
            "game_url": "",
            "clicked": None,
            "exchange_clicked": False,
            "exchange_selector": "",
            "lay_clicked": False,
            "lay_selector": "",
            "exchange_candidates": 0,
            "lay_candidates": 0,
            "pmm_before": 0,
            "pmm_after_odd_click": 0,
            "pmm_after_exchange_click": 0,
            "http_betslip_posts": 0,
            "http_betslip_refresh_posts": 0,
            "exchange_elements": [],
            "classic_elements": [],
            "screenshots": [],
        }

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

        # screenshot antes de clicar em odds
        try:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            p0 = f"logs/lay_ws_screens/{ts}_01_game.png"
            await _screenshot(page, p0)
            out["screenshots"].append(p0)
        except Exception:
            pass

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
        # baseline counters (para sabermos se o click abriu betslip de fato)
        try:
            out["pmm_before"] = int(self.global_api_subtypes.get("pmm", 0))
        except Exception:
            out["pmm_before"] = 0
        http0 = list(self.http_reqs)

        clicked = await page.evaluate(
            """
            () => {
              // Estratégia 1 (preferida): clicar em odds DENTRO da seção Asian Handicap.
              // Isso aumenta muito a chance de existir Exchange/Lay para a seleção.
              function findSectionByHeaderText(headerText) {
                const els = Array.from(document.querySelectorAll('div, span, h1, h2, h3, h4'));
                for (const h of els) {
                  const t = (h.innerText || '').trim();
                  if (!t) continue;
                  if (t.toLowerCase().includes(headerText.toLowerCase())) {
                    let p = h.parentElement;
                    for (let i = 0; i < 10 && p; i++) {
                      const pt = (p.innerText || '');
                      if (pt.includes('Home') && pt.includes('Away')) return p;
                      p = p.parentElement;
                    }
                  }
                }
                return null;
              }

              function clickFirstDecimalOddWithin(root, minOdd, maxOdd) {
                if (!root) return null;
                const els = Array.from(root.querySelectorAll('span, div, button'));
                const cands = [];
                for (const el of els) {
                  const t = (el.innerText || '').trim();
                  if (!t || t.length > 6) continue;
                  if (!t.includes('.')) continue;
                  const v = parseFloat(t);
                  if (!isFinite(v)) continue;
                  if (v < minOdd || v > maxOdd) continue;
                  const r = el.getBoundingClientRect();
                  if (!r || r.width < 18 || r.height < 10) continue;
                  if (r.width > 260 || r.height > 90) continue;
                  // precisa estar visível
                  if (el.offsetParent === null) continue;
                  cands.push({t, v, el});
                }
                // Preferir odds moderadas (perto de 2.0)
                cands.sort((a,b) => Math.abs(a.v - 2.0) - Math.abs(b.v - 2.0));
                for (let i = 0; i < Math.min(25, cands.length); i++) {
                  const el = cands[i].el;
                  try { el.scrollIntoView({behavior: 'instant', block: 'center'}); } catch(e) {}
                  try { el.click(); return {odd: cands[i].t, method: 'AH/direct'}; } catch(e) {}
                  try { el.parentElement && el.parentElement.click(); return {odd: cands[i].t, method: 'AH/parent'}; } catch(e) {}
                }
                return null;
              }

              const ahSec = findSectionByHeaderText('Asian Handicap');
              const clickedAh = clickFirstDecimalOddWithin(ahSec, 1.20, 4.00);
              if (clickedAh) return clickedAh;

              // Estratégia 2 (fallback): qualquer odd no documento, mas limita range para evitar 7.xx etc.
              const candidates = [];
              const els = Array.from(document.querySelectorAll('span, div, button'));
              for (const el of els) {
                const t = (el.innerText || '').trim();
                if (!t || t.length > 6) continue;
                if (!t.includes('.')) continue;
                const v = parseFloat(t);
                if (!isFinite(v)) continue;
                if (v < 1.20 || v > 6.00) continue;
                const r = el.getBoundingClientRect();
                if (!r || r.width < 18 || r.height < 10) continue;
                if (r.width > 260 || r.height > 90) continue;
                if (el.offsetParent === null) continue;
                candidates.push({t, v, el});
              }
              candidates.sort((a,b) => Math.abs(a.v - 2.0) - Math.abs(b.v - 2.0));
              for (let i = 0; i < Math.min(25, candidates.length); i++) {
                const el = candidates[i].el;
                try { el.scrollIntoView({behavior: 'instant', block: 'center'}); } catch(e) {}
                try { el.click(); return {odd: candidates[i].t, method: 'fallback/direct'}; } catch(e) {}
                try { el.parentElement && el.parentElement.click(); return {odd: candidates[i].t, method: 'fallback/parent'}; } catch(e) {}
              }
              return null;
            }
            """
        )

        out["clicked"] = clicked
        await page.wait_for_timeout(4000)

        # delta HTTP (betslip/refresh)
        try:
            new_http = self.http_reqs[len(http0) :]
            out["http_betslip_posts"] = sum(
                1
                for h in new_http
                if h.get("method") == "POST" and "/v1/betslips/" in (h.get("url") or "") and "/refresh/" not in (h.get("url") or "")
            )
            out["http_betslip_refresh_posts"] = sum(
                1
                for h in new_http
                if h.get("method") == "POST" and "/v1/betslips/" in (h.get("url") or "") and "/refresh/" in (h.get("url") or "")
            )
        except Exception:
            out["http_betslip_posts"] = 0
            out["http_betslip_refresh_posts"] = 0

        try:
            out["pmm_after_odd_click"] = int(self.global_api_subtypes.get("pmm", 0))
        except Exception:
            out["pmm_after_odd_click"] = out.get("pmm_before", 0)

        # screenshot após clicar em odd (betslip deve estar aberto)
        try:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            p1 = f"logs/lay_ws_screens/{ts}_02_after_odd_click.png"
            await _screenshot(page, p1)
            out["screenshots"].append(p1)
        except Exception:
            pass

        # tenta alternar Exchange / Lay (best effort)
        self.phase = "after_exchange_attempt"
        exchange_clicked = False
        lay_clicked = False

        # Conta candidatos visíveis (para debug)
        try:
            out["exchange_candidates"] = int(await page.locator("text=/exchange/i").count())
        except Exception:
            out["exchange_candidates"] = 0
        try:
            out["lay_candidates"] = int(await page.locator("text=/\\blay\\b/i").count())
        except Exception:
            out["lay_candidates"] = 0

        # Captura detalhes dos elementos com texto Exchange/Classic (para debug)
        try:
            out["exchange_elements"] = await page.evaluate(
                """
                () => {
                  const re = /\\bexchange\\b/i;
                  const els = Array.from(document.querySelectorAll('button, a, div, span'));
                  const out = [];
                  for (const el of els) {
                    const t = (el.innerText || '').trim();
                    if (!t) continue;
                    if (!re.test(t)) continue;
                    const r = el.getBoundingClientRect();
                    const visible = (el.offsetParent !== null) && r.width > 0 && r.height > 0;
                    out.push({
                      text: t.slice(0, 120),
                      tag: el.tagName,
                      id: el.id || '',
                      cls: (el.className || '').toString().slice(0, 200),
                      role: el.getAttribute('role') || '',
                      aria: el.getAttribute('aria-label') || '',
                      x: Math.round(r.x), y: Math.round(r.y),
                      w: Math.round(r.width), h: Math.round(r.height),
                      visible
                    });
                  }
                  // prioriza visíveis e menores (tabs)
                  out.sort((a,b) => (b.visible - a.visible) || ((a.w*a.h) - (b.w*b.h)));
                  return out.slice(0, 20);
                }
                """
            )
        except Exception:
            out["exchange_elements"] = []

        try:
            out["classic_elements"] = await page.evaluate(
                """
                () => {
                  const re = /\\bclassic\\b/i;
                  const els = Array.from(document.querySelectorAll('button, a, div, span'));
                  const out = [];
                  for (const el of els) {
                    const t = (el.innerText || '').trim();
                    if (!t) continue;
                    if (!re.test(t)) continue;
                    const r = el.getBoundingClientRect();
                    const visible = (el.offsetParent !== null) && r.width > 0 && r.height > 0;
                    out.push({
                      text: t.slice(0, 120),
                      tag: el.tagName,
                      id: el.id || '',
                      cls: (el.className || '').toString().slice(0, 200),
                      role: el.getAttribute('role') || '',
                      aria: el.getAttribute('aria-label') || '',
                      x: Math.round(r.x), y: Math.round(r.y),
                      w: Math.round(r.width), h: Math.round(r.height),
                      visible
                    });
                  }
                  out.sort((a,b) => (b.visible - a.visible) || ((a.w*a.h) - (b.w*b.h)));
                  return out.slice(0, 20);
                }
                """
            )
        except Exception:
            out["classic_elements"] = []

        # Alguns layouts usam "Classic" vs "Exchange". Tentamos Exchange primeiro.
        exchange_selectors = [
            "text=/\\bexchange\\b/i",
            "button:has-text('Exchange')",
            "span:has-text('Exchange')",
            "[data-testid*='exchange']",
            "[class*='exchange']",
            "a:has-text('Exchange')",
        ]
        exchange_clicked, exchange_sel = await _try_click_any(page, exchange_selectors, timeout_ms=3000)
        out["exchange_selector"] = exchange_sel

        # Se o click por seletor falhou, tenta por coordenada (primeiro elemento visível que contenha Exchange)
        if not exchange_clicked and out.get("exchange_elements"):
            try:
                cand = next((e for e in out["exchange_elements"] if e.get("visible")), None)
                if cand and all(k in cand for k in ("x", "y", "w", "h")):
                    cx = float(cand["x"]) + float(cand["w"]) / 2.0
                    cy = float(cand["y"]) + float(cand["h"]) / 2.0
                    # click no centro
                    await page.mouse.click(cx, cy)
                    exchange_clicked = True
                    out["exchange_selector"] = f"mouse@({int(cx)},{int(cy)})"
            except Exception:
                pass

        await page.wait_for_timeout(2500)
        try:
            out["pmm_after_exchange_click"] = int(self.global_api_subtypes.get("pmm", 0))
        except Exception:
            out["pmm_after_exchange_click"] = out.get("pmm_after_odd_click", 0)

        # screenshot após tentativa de ir para Exchange
        try:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            p2 = f"logs/lay_ws_screens/{ts}_03_after_exchange_click.png"
            await _screenshot(page, p2)
            out["screenshots"].append(p2)
        except Exception:
            pass

        lay_selectors = [
            "text=/\\blay\\b/i",
            "button:has-text('Lay')",
            "span:has-text('Lay')",
            "[data-testid*='lay']",
            "[class*='lay']",
            "a:has-text('Lay')",
        ]
        lay_clicked, lay_sel = await _try_click_any(page, lay_selectors, timeout_ms=3000)
        out["lay_selector"] = lay_sel

        out["exchange_clicked"] = exchange_clicked
        out["lay_clicked"] = lay_clicked
        await page.wait_for_timeout(2500)

        # screenshot final
        try:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            p3 = f"logs/lay_ws_screens/{ts}_04_final.png"
            await _screenshot(page, p3)
            out["screenshots"].append(p3)
        except Exception:
            pass
        return out

    async def trade_flow(self, page, *, wait_sec: float = 10.0, interact: bool = False) -> dict:
        """
        Navega para /trade e captura WS/HTTP.
        Opcionalmente tenta interagir (best effort) para forçar carregamento de preços.
        """
        out: Dict[str, Any] = {
            "start_url": "https://black.betinasia.com/trade",
            "wait_sec": float(wait_sec),
            "interact": bool(interact),
            "interaction": {"clicked": []},
            "dom_debug": {
                "url_after": "",
                "link_samples": [],
                "counts": {},
                "event_candidates": [],
                "body_text_prefix": "",
                "scrollables": [],
            },
            "http_delta": {
                "betslip_posts": 0,
                "betslip_refresh_posts": 0,
                "xhr_fetch_urls_top": [],
                "probed": [],
            },
            "screenshots": [],
        }
        http0 = list(self.http_reqs)
        await page.goto(out["start_url"])
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(3000)
        try:
            out["dom_debug"]["url_after"] = getattr(page, "url", "") or ""
        except Exception:
            out["dom_debug"]["url_after"] = ""

        # Captura amostra de links/textos para entender a rota/estrutura do Trade.
        try:
            out["dom_debug"]["link_samples"] = await page.evaluate(
                """
                () => {
                  const out = [];
                  const links = Array.from(document.querySelectorAll('a[href]'));
                  for (const a of links.slice(0, 250)) {
                    const href = (a.getAttribute('href') || '').trim();
                    const t = (a.innerText || '').trim().replace(/\\s+/g,' ').slice(0, 80);
                    if (!href) continue;
                    if (href.startsWith('javascript:')) continue;
                    // prioriza links que pareçam levar a eventos/mercados
                    if (href.includes(',') || href.toLowerCase().includes('trade') || href.toLowerCase().includes('football')) {
                      out.push({href: href.slice(0, 200), text: t});
                    }
                    if (out.length >= 80) break;
                  }
                  return out;
                }
                """
            )
        except Exception:
            out["dom_debug"]["link_samples"] = []

        try:
            out["dom_debug"]["counts"] = await page.evaluate(
                """
                () => {
                  const q = (re) => Array.from(document.querySelectorAll('div,span,button,a,li'))
                    .filter(el => ((el.innerText||'').match(re)||[]).length>0).length;
                  return {
                    hasVs: q(/\\bvs\\b/i),
                    hasBack: q(/\\bback\\b/i),
                    hasLay: q(/\\blay\\b/i),
                    hasOrder: q(/\\border\\b/i),
                    hasPrice: q(/\\bprice\\b/i),
                    hasMarket: q(/\\bmarket\\b/i),
                  };
                }
                """
            )
        except Exception:
            out["dom_debug"]["counts"] = {}

        # Captura um prefixo do texto do body (para ver se a lista está renderizando mesmo sem "vs")
        try:
            out["dom_debug"]["body_text_prefix"] = await page.evaluate(
                """
                () => {
                  const t = (document.body && document.body.innerText) ? document.body.innerText : '';
                  return (t || '').trim().replace(/\\s+/g,' ').slice(0, 500);
                }
                """
            )
        except Exception:
            out["dom_debug"]["body_text_prefix"] = ""

        # Encontra containers scrolláveis (lista virtualizada costuma estar aqui)
        try:
            out["dom_debug"]["scrollables"] = await page.evaluate(
                """
                () => {
                  const out = [];
                  const els = Array.from(document.querySelectorAll('div, main, section'));
                  for (const el of els) {
                    if (!el) continue;
                    const sh = el.scrollHeight || 0;
                    const ch = el.clientHeight || 0;
                    const sw = el.scrollWidth || 0;
                    const cw = el.clientWidth || 0;
                    if (sh <= ch + 120) continue;
                    const r = el.getBoundingClientRect();
                    if (!r || r.width < 240 || r.height < 220) continue;
                    if (el.offsetParent === null) continue;
                    const style = window.getComputedStyle(el);
                    const oy = style.overflowY || '';
                    const ox = style.overflowX || '';
                    out.push({
                      tag: el.tagName,
                      cls: (el.className || '').toString().slice(0, 160),
                      oy, ox,
                      x: Math.round(r.x), y: Math.round(r.y),
                      w: Math.round(r.width), h: Math.round(r.height),
                      scrollHeight: sh, clientHeight: ch
                    });
                    if (out.length >= 10) break;
                  }
                  return out;
                }
                """
            )
        except Exception:
            out["dom_debug"]["scrollables"] = []
        try:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            p0 = f"logs/lay_ws_screens/{ts}_trade_01.png"
            await _screenshot(page, p0)
            out["screenshots"].append(p0)
        except Exception:
            pass

        if interact:
            # Best-effort: clicar em Football e depois em algum item com "vs"
            try:
                ok, sel = await _try_click_any(
                    page,
                    [
                        "text=/\\bfootball\\b/i",
                        "button:has-text('Football')",
                        "a:has-text('Football')",
                    ],
                    timeout_ms=2500,
                )
                if ok:
                    out["interaction"]["clicked"].append({"what": "football", "sel": sel})
                    await page.wait_for_timeout(1500)
            except Exception:
                pass

            # Tenta ir diretamente para a rota principal do Trade Football (full-time)
            try:
                await page.goto("https://black.betinasia.com/trade/football/full-time")
                await page.wait_for_load_state("domcontentloaded")
                await page.wait_for_timeout(2500)
                out["interaction"]["clicked"].append({"what": "goto_trade_football_fulltime"})
            except Exception:
                pass

            # Tenta clicar em "Top" / "Live" caso exista (Trade geralmente tem essas tabs).
            try:
                ok, sel = await _try_click_any(
                    page,
                    ["text=/\\btop\\b/i", "text=/\\blive\\b/i", "button:has-text('Top')", "button:has-text('Live')"],
                    timeout_ms=2000,
                )
                if ok:
                    out["interaction"]["clicked"].append({"what": "top_or_live", "sel": sel})
                    await page.wait_for_timeout(1500)
            except Exception:
                pass

            try:
                # tenta clicar em qualquer linha/list item que contenha "vs"
                clicked = await page.evaluate(
                    """
                    () => {
                      const els = Array.from(document.querySelectorAll('a, button, div, span, li'));
                      for (const el of els) {
                        const t = (el.innerText || '').trim();
                        if (!t) continue;
                        const low = t.toLowerCase();
                        if (!low.includes(' vs') && !low.includes('vs.') && !low.includes(' vs. ')) continue;
                        const r = el.getBoundingClientRect();
                        if (!r || r.width < 80 || r.height < 14) continue;
                        if (el.offsetParent === null) continue;
                        try { el.scrollIntoView({behavior:'instant', block:'center'});} catch(e){}
                        try { el.click(); return {text: t.slice(0,120)}; } catch(e){}
                        try { el.parentElement && el.parentElement.click(); return {text: t.slice(0,120)}; } catch(e){}
                      }
                      return null;
                    }
                    """
                )
                if clicked:
                    out["interaction"]["clicked"].append({"what": "event_vs", **clicked})
                    await page.wait_for_timeout(2000)
            except Exception:
                pass

            # Fallback 2: clicar em qualquer link que pareça ser um evento (href com vírgulas)
            try:
                clicked = await page.evaluate(
                    """
                    () => {
                      const links = Array.from(document.querySelectorAll('a[href]'));
                      for (const a of links) {
                        const href = (a.getAttribute('href') || '');
                        if (!href) continue;
                        if (href.includes(',') && (href.includes('trade') || href.includes('sportsbook'))) {
                          const r = a.getBoundingClientRect();
                          if (!r || r.width < 40 || r.height < 12) continue;
                          if (a.offsetParent === null) continue;
                          try { a.scrollIntoView({behavior:'instant', block:'center'});} catch(e){}
                          try { a.click(); return {href: href.slice(0,200)}; } catch(e){}
                        }
                      }
                      return null;
                    }
                    """
                )
                if clicked:
                    out["interaction"]["clicked"].append({"what": "event_href_comma", **clicked})
                    await page.wait_for_timeout(2500)
            except Exception:
                pass

            # Fallback 2b: scroll para forçar renderização virtualizada de lista
            try:
                await page.mouse.wheel(0, 1400)
                await page.wait_for_timeout(1200)
                await page.mouse.wheel(0, 1400)
                await page.wait_for_timeout(1200)
                out["interaction"]["clicked"].append({"what": "scroll"})
            except Exception:
                pass

            # Se houver container scrollável, tenta scroll via JS para garantir que afeta a lista (não a janela)
            try:
                await page.evaluate(
                    """
                    () => {
                      const els = Array.from(document.querySelectorAll('div, main, section'));
                      let best = null;
                      for (const el of els) {
                        const sh = el.scrollHeight || 0;
                        const ch = el.clientHeight || 0;
                        if (sh <= ch + 120) continue;
                        const r = el.getBoundingClientRect();
                        if (!r || r.width < 240 || r.height < 220) continue;
                        if (el.offsetParent === null) continue;
                        best = el;
                        break;
                      }
                      if (best) {
                        best.scrollTop = Math.min(best.scrollHeight, best.scrollTop + best.clientHeight * 2);
                      }
                    }
                    """
                )
                out["interaction"]["clicked"].append({"what": "scroll_js"})
                await page.wait_for_timeout(1500)
            except Exception:
                pass

            # Tenta clicar em um possível “evento/mercado” por heurística (preços decimais visíveis)
            try:
                candidates = await page.evaluate(
                    """
                    () => {
                      // Procura elementos pequenos com números decimais (preços).
                      // Em Trade, clicar no preço/linha costuma abrir ticket/market.
                      const out = [];
                      const els = Array.from(document.querySelectorAll('div, span, button, a'));
                      for (const el of els) {
                        const t0 = (el.innerText || '').trim();
                        if (!t0) continue;
                        const t = t0.replace(/\\s+/g,' ');
                        if (t.length > 10) continue;
                        if (!t.includes('.')) continue;
                        const v = parseFloat(t);
                        if (!isFinite(v)) continue;
                        // Odds típicas: evita saldo (ex.: 58.00) e números muito grandes
                        if (v < 1.01 || v > 25.0) continue;
                        const r = el.getBoundingClientRect();
                        if (!r || r.width < 18 || r.height < 10) continue;
                        if (r.width > 160 || r.height > 70) continue;
                        if (el.offsetParent === null) continue;
                        // Evita header/topbar
                        if (r.y < 90) continue;
                        out.push({
                          text: t,
                          v,
                          tag: el.tagName,
                          id: el.id || '',
                          cls: (el.className || '').toString().slice(0, 160),
                          role: el.getAttribute('role') || '',
                          x: Math.round(r.x), y: Math.round(r.y),
                          w: Math.round(r.width), h: Math.round(r.height),
                        });
                      }
                      // Preferir preços perto de 2.0
                      out.sort((a,b) => Math.abs(a.v - 2.0) - Math.abs(b.v - 2.0));
                      return out.slice(0, 25);
                    }
                    """
                )
                out["dom_debug"]["event_candidates"] = candidates or []
                if candidates:
                    c = candidates[0]
                    const_x = float(c.get("x", 0)) + float(c.get("w", 0)) / 2.0
                    const_y = float(c.get("y", 0)) + float(c.get("h", 0)) / 2.0
                    try:
                        await page.mouse.click(const_x, const_y)
                        out["interaction"]["clicked"].append(
                            {"what": "price_candidate_click", "text": c.get("text"), "x": int(const_x), "y": int(const_y)}
                        )
                        await page.wait_for_timeout(2500)
                    except Exception:
                        pass
            except Exception:
                out["dom_debug"]["event_candidates"] = out["dom_debug"].get("event_candidates") or []

            # Fallback 3: tentar clicar em um elemento que pareça "order ticket" (Back/Lay)
            try:
                ok, sel = await _try_click_any(
                    page,
                    [
                        "text=/\\bback\\b/i",
                        "text=/\\blay\\b/i",
                        "text=/\\border\\b/i",
                        "text=/\\bprice\\b/i",
                    ],
                    timeout_ms=2000,
                )
                if ok:
                    out["interaction"]["clicked"].append({"what": "ticket_hint", "sel": sel})
                    await page.wait_for_timeout(1500)
            except Exception:
                pass

            try:
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                p1 = f"logs/lay_ws_screens/{ts}_trade_02_after_interact.png"
                await _screenshot(page, p1)
                out["screenshots"].append(p1)
            except Exception:
                pass

            # Recoleta debug após interações
            try:
                out["dom_debug"]["url_after"] = getattr(page, "url", "") or out["dom_debug"].get("url_after", "")
            except Exception:
                pass
            try:
                out["dom_debug"]["counts"] = await page.evaluate(
                    """
                    () => {
                      const q = (re) => Array.from(document.querySelectorAll('div,span,button,a,li'))
                        .filter(el => ((el.innerText||'').match(re)||[]).length>0).length;
                      return {
                        hasVs: q(/\\bvs\\b/i),
                        hasBack: q(/\\bback\\b/i),
                        hasLay: q(/\\blay\\b/i),
                        hasOrder: q(/\\border\\b/i),
                        hasPrice: q(/\\bprice\\b/i),
                        hasMarket: q(/\\bmarket\\b/i),
                      };
                    }
                    """
                )
            except Exception:
                pass
            try:
                out["dom_debug"]["link_samples"] = await page.evaluate(
                    """
                    () => {
                      const out = [];
                      const links = Array.from(document.querySelectorAll('a[href]'));
                      for (const a of links.slice(0, 450)) {
                        const href = (a.getAttribute('href') || '').trim();
                        const t = (a.innerText || '').trim().replace(/\\s+/g,' ').slice(0, 80);
                        if (!href) continue;
                        if (href.startsWith('javascript:')) continue;
                        if (href.includes(',') || href.toLowerCase().includes('trade') || href.toLowerCase().includes('football')) {
                          out.push({href: href.slice(0, 200), text: t});
                        }
                        if (out.length >= 80) break;
                      }
                      return out;
                    }
                    """
                )
            except Exception:
                pass

        # http delta (após interações)
        try:
            new_http = self.http_reqs[len(http0) :]
            out["http_delta"]["betslip_posts"] = sum(
                1
                for h in new_http
                if h.get("method") == "POST" and "/v1/betslips/" in (h.get("url") or "") and "/refresh/" not in (h.get("url") or "")
            )
            out["http_delta"]["betslip_refresh_posts"] = sum(
                1
                for h in new_http
                if h.get("method") == "POST" and "/v1/betslips/" in (h.get("url") or "") and "/refresh/" in (h.get("url") or "")
            )
            # top URLs XHR/fetch
            from collections import Counter as _C
            top = _C()
            for h in new_http:
                if h.get("resource_type") in ("xhr", "fetch"):
                    u = str(h.get("url") or "")
                    # normaliza (sem query)
                    u0 = u.split("?", 1)[0]
                    if "black.betinasia.com" in u0:
                        top[u0] += 1
            out["http_delta"]["xhr_fetch_urls_top"] = top.most_common(30)
        except Exception:
            pass

        # Probe de endpoints para descobrir payloads (sem depender do hook de response)
        try:
            urls = [u for (u, _c) in (out.get("http_delta", {}).get("xhr_fetch_urls_top") or [])]
            def _prio(u: str) -> int:
                ul = (u or "").lower()
                if "/web/sessions/" in ul and "/broadcaster/" in ul:
                    return 0
                if "/web/sessions/" in ul and "/announcement/" in ul:
                    return 1
                if "/web/sessions/" in ul:
                    return 2
                if "/web/" in ul:
                    return 3
                return 9
            cand = sorted([u for u in urls if "black.betinasia.com" in (u or "")], key=_prio)[:6]
            probed = []
            session_token = await _get_root_session_token(page)
            use_session_hdr = bool(session_token)
            for u in cand:
                try:
                    res = await page.evaluate(
                        """
                        async (params) => {
                          try {
                            const headers = {};
                            if (params.session) headers['session'] = params.session;
                            headers['Accept'] = 'application/json, text/plain, */*';
                            const resp = await fetch(params.url, {method:'GET', credentials:'same-origin', headers});
                            const status = resp.status;
                            const ct = resp.headers.get('content-type') || '';
                            let text = '';
                            try { text = await resp.text(); } catch(e) { text = ''; }
                            let jsonKeys = [];
                            try {
                              const j = JSON.parse(text);
                              if (j && typeof j === 'object' && !Array.isArray(j)) {
                                jsonKeys = Object.keys(j).slice(0, 60);
                              }
                            } catch(e) {}
                            const prefix = (text || '').slice(0, 800);
                            return {ok: resp.ok, status, content_type: ct, json_keys: jsonKeys, prefix};
                          } catch(e) {
                            return {ok:false, error: e.message};
                          }
                        }
                        """,
                        {"url": u, "session": session_token},
                    )
                    if isinstance(res, dict):
                        probed.append(
                            {
                                "url": u,
                                "ok": bool(res.get("ok")),
                                "status": res.get("status"),
                                "content_type": res.get("content_type"),
                                "json_keys": res.get("json_keys") or [],
                                "prefix": _truncate(str(res.get("prefix") or ""), 800),
                                "error": res.get("error") or "",
                            }
                        )
                except Exception:
                    continue
            out["http_delta"]["probed"] = probed
            out["http_delta"]["probe_used_session_header"] = use_session_hdr
        except Exception:
            pass

        await page.wait_for_timeout(int(float(wait_sec) * 1000))
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
                    "pmm_bet_prefix_top": st.pmm_bet_prefix_counts.most_common(10),
                    "pmm_bet_types_top": st.pmm_bet_type_counts.most_common(15),
                    "pmm_against_samples": st.pmm_against_samples,
                    "pmm_for_samples": st.pmm_for_samples,
                    "trade_keyword_snippets": st.trade_keyword_snippets,
                    "offers_keys": sorted(st.offers_keys),
                    "offers_nested_keys": sorted(st.offers_nested_keys),
                    "snippets_with_lay": st.snippets_with_lay,
                    "recv_samples": st.recv_samples,
                    "sent_samples": st.sent_samples,
                }
            )

        # Pairing: mostra que "for" e "against" compartilham o MESMO suffix
        # (ex.: for,ah,h,-1  vs against,ah,h,-1). Isso demonstra que "against" não é "away".
        suffix_pair: Dict[str, Dict[str, int]] = {}
        for bt, c in (self.global_pmm_bet_type_counts or {}).items():
            if not isinstance(bt, str) or "," not in bt:
                continue
            parts = [p.strip() for p in bt.split(",") if p is not None]
            if len(parts) < 2:
                continue
            prefix = parts[0].lower()
            suffix = ",".join(parts[1:]).strip()
            if not suffix:
                continue
            d = suffix_pair.setdefault(suffix, {"for": 0, "against": 0, "other": 0})
            if prefix == "for":
                d["for"] += int(c or 0)
            elif prefix == "against":
                d["against"] += int(c or 0)
            else:
                d["other"] += int(c or 0)

        paired_suffixes = []
        for suf, d in suffix_pair.items():
            if d.get("for", 0) > 0 and d.get("against", 0) > 0:
                paired_suffixes.append((suf, int(d["for"]), int(d["against"])))
        paired_suffixes.sort(key=lambda x: -(x[1] + x[2]))

        return {
            "ts": _utc_now_iso(),
            "elapsed_sec": round(elapsed, 3),
            "click_flow": click_info,
            "http_reqs": self.http_reqs,
            "global": {
                "msg_types_top": self.global_msg_types.most_common(50),
                "api_subtypes_top": self.global_api_subtypes.most_common(50),
                "pmm_bet_prefix_top": self.global_pmm_bet_prefix_counts.most_common(10),
                "pmm_against_samples": self.global_pmm_against_samples,
                "pmm_bet_types_top": self.global_pmm_bet_type_counts.most_common(25),
                "pmm_suffix_pair_top": paired_suffixes[:25],
                "offers_keys": sorted(self.global_offers_keys),
                "offers_nested_keys": sorted(self.global_offers_nested_keys),
                "response_hook_errors": int(self.response_hook_errors or 0),
                "response_hook_seen": int(self.response_hook_seen or 0),
                "response_hook_kept": int(self.response_hook_kept or 0),
            },
            "websockets": ws_summary,
            "http_resps": self.http_resps,
        }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/lay_ws_discovery.json", help="Caminho do JSON de saída.")
    ap.add_argument("--sample-limit", type=int, default=30, help="Máx. de amostras por WS (recv/sent/snippets).")
    ap.add_argument("--baseline-wait-sec", type=float, default=8.0, help="Tempo esperando apenas WS (sem cliques).")
    ap.add_argument("--do-click-flow", action="store_true", help="Tenta clicar em odd e alternar Exchange/Lay.")
    ap.add_argument("--start", choices=["football", "trade"], default="football", help="Página inicial para descoberta.")
    ap.add_argument("--trade-wait-sec", type=float, default=12.0, help="Tempo de espera na página /trade.")
    ap.add_argument("--trade-interact", action="store_true", help="Tenta interagir na página /trade (best effort).")
    ap.add_argument(
        "--trade-from-sportsbook",
        action="store_true",
        help="Fluxo alternativo: abre um jogo no Sportsbook e clica no menu 'Trade' (sem clicar em odds).",
    )
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
        page.on("response", disc.on_response)

        click_info: dict = {"skipped": True}
        if str(args.start) == "trade":
            disc.phase = "trade"
            if bool(getattr(args, "trade_from_sportsbook", False)):
                # 1) vai para futebol sportsbook
                await page.goto("https://black.betinasia.com/sportsbook/football")
                await page.wait_for_load_state("domcontentloaded")
                await page.wait_for_timeout(3000)
                # 2) abre 1º jogo encontrado (sem clicar em odds)
                try:
                    game_link = await page.evaluate(
                        """
                        () => {
                          const links = Array.from(document.querySelectorAll('a'));
                          for (const link of links) {
                            const href = link.getAttribute('href') || '';
                            if (href.includes('/sportsbook/football/') && href.includes(',')) return href;
                          }
                          return null;
                        }
                        """
                    )
                    if game_link:
                        base_url = "https://black.betinasia.com"
                        game_url = f"{base_url}{game_link}" if str(game_link).startswith("/") else str(game_link)
                        await page.goto(game_url)
                        await page.wait_for_load_state("domcontentloaded")
                        await page.wait_for_timeout(2500)
                except Exception:
                    pass
                # 3) tenta clicar em "Trade" no menu superior
                disc.phase = "trade_from_sportsbook"
                url_before = getattr(page, "url", "")
                clicked, sel = await _try_click_any(
                    page,
                    ["text=/\\btrade\\b/i", "a:has-text('Trade')", "button:has-text('Trade')"],
                    timeout_ms=3000,
                )
                click_info = {
                    "mode": "trade_from_sportsbook",
                    "trade_clicked": bool(clicked),
                    "trade_selector": sel,
                    "url_before": url_before,
                    "url_after_click": getattr(page, "url", ""),
                }
                # Se não navegou para /trade, força a rota que vimos no DOM.
                forced = False
                try:
                    if "/trade" not in (getattr(page, "url", "") or ""):
                        await page.goto("https://black.betinasia.com/trade/football/full-time")
                        await page.wait_for_load_state("domcontentloaded")
                        await page.wait_for_timeout(2500)
                        forced = True
                except Exception:
                    forced = False
                click_info["forced_goto_trade"] = forced
                click_info["url_after"] = getattr(page, "url", "")
                await page.wait_for_timeout(int(float(args.trade_wait_sec) * 1000))
            else:
                click_info = await disc.trade_flow(
                    page, wait_sec=float(args.trade_wait_sec), interact=bool(args.trade_interact)
                )
        else:
            disc.phase = "baseline"
            await page.goto("https://black.betinasia.com/sportsbook/football")
            await page.wait_for_load_state("domcontentloaded")
            await page.wait_for_timeout(int(float(args.baseline_wait_sec) * 1000))
            if bool(args.do_click_flow):
                click_info = await disc.click_flow(page)

        # garante que tasks de capture de response finalizaram (melhor diagnóstico)
        await disc.flush_http_responses(timeout_sec=2.0)

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

