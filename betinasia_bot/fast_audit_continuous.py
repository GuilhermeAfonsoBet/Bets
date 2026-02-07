#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auditoria H3B Contínua — Versão Rápida (navegação direta)

Arquitetura:
  - Tab MONITOR: WebSocket permanente, detecta H3B em TODAS as ligas
  - Tab EXECUTOR 1-2: Navega direto para URL do jogo, clica odd, extrai betslip
  - Task MANUTENÇÃO: Health check WS, stats periódicas

Mercados: AH Full-Time, AH Half-Time, OU Full-Time, OU Half-Time
Lag estimado: ~7s (navegação direta via proxy)

Uso:
    # Contínuo (produção)
    DISPLAY=:99 python fast_audit_continuous.py

    # Limitado (teste)
    DISPLAY=:99 python fast_audit_continuous.py --num-audits 20
"""

import asyncio
import argparse
import json
import re
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from loguru import logger

sys.path.insert(0, '.')

from playwright.async_api import Page
from scraper.betinasia import BetinAsiaScraper
from scraper.betslip_extractor import BetslipExtractor, BetslipData
from hypothesis.detectors import HypothesisDetector
from sqlalchemy import text
from storage.database import Database
from storage.models_hypothesis import BetslipAuditResult


BASE_URL = "https://black.betinasia.com"
FOOTBALL_URL = f"{BASE_URL}/sportsbook/football"

# Mapa de competition_name (WS) → league_code (URL)
# Chaves normalizadas (lower)
LEAGUE_CODE_MAP = {
    "england premier league": "XE/1",
    "english premier league": "XE/1",
    "england football league championship": "XE/2",
    "england championship": "XE/2",
    "england fa cup": "XE/132",
    "germany bundesliga": "DE/12",
    "germany 2. bundesliga": "DE/13",
    "germany bundesliga 2": "DE/13",
    "spain la liga": "ES/16",
    "spain second division a (liga adelante)": "ES/17",
    "italy serie a": "IT/19",
    "italy serie b": "IT/20",
    "france ligue 1": "FR/38",
    "france ligue 2": "FR/39",
    "netherlands premier division (eredivisie)": "NL/1",
    "netherlands eredivisie": "NL/1",
    "portugal primeira liga": "PT/1",
    "belgium 1st division a": "BE/1",
    "turkey super league": "TR/160",
    "brazil campeonato brasiliero série a": "BR/1",
    "brazil campeonato brasiliero serie a": "BR/1",
    "argentina liga profesional": "AR/1",
    "argentina primera division": "AR/1",
    "uefa champions league": "XE/5",
    "uefa europa league": "XE/6",
    "usa major league soccer": "US/23",
    "colombia primera a": "CO/1",
    "saudi arabia pro league": "SA/1",
    "japan j-league division 1": "JP/1",
    "japan j-league division 2": "JP/36",
    "england league 1": "XE/3",
    "england league 2": "XE/4",
    "conmebol copa sudamericana": "XS/21",
    "fifa world cup": "XW/65",
    "scotland premiership": "XS/1",
}

# Config
WS_HEALTH_CHECK_INTERVAL = 15
WS_RELOAD_INTERVAL = 60
STATS_LOG_INTERVAL = 100  # Log stats a cada N auditorias
MAX_AH_LINE = 2.0
NUM_EXECUTORS = 2
PAGE_LOAD_TIMEOUT = 12000  # ms


@dataclass
class AuditResult:
    timestamp: datetime
    event_id: str
    home_team: str
    away_team: str
    league: str
    market_type: str
    market_period: str
    line: str
    side: str
    websocket_odd: float
    betslip_odd: Optional[float] = None
    betslip_limit: Optional[float] = None
    difference_pct: Optional[float] = None
    status: str = ""
    is_live: Optional[bool] = None
    direction: str = "up"
    lag_total_ms: int = 0
    lag_navigate_ms: int = 0
    lag_expand_ms: int = 0
    lag_click_ms: int = 0
    lag_betslip_ms: int = 0


class FastAuditContinuous:

    def __init__(self, num_audits: int = 0, direction: str = "up", save_to_db: bool = True):
        self.num_audits = num_audits  # 0 = infinito
        self.direction = direction
        self.save_to_db = save_to_db

        self.scraper: Optional[BetinAsiaScraper] = None
        self.monitor_page: Optional[Page] = None
        self.executor_pages: List[Page] = []
        self.context = None
        self.db: Optional[Database] = None

        # WS
        self._ws_messages: List[str] = []
        self._ws_connected = False
        self._last_ws_msg_time: float = 0
        self._ws_msg_count: int = 0
        self._start_time: float = 0

        # Events info cache (from WS)
        self._events_info: Dict[str, dict] = {}

        # Detector
        self.detector = HypothesisDetector()

        # Stats
        self.results: List[AuditResult] = []
        self.events_processed: int = 0
        self.h3b_detected: int = 0
        self.total_errors: int = 0
        self.running = True

        # Lock
        self._executor_locks = []

    async def start(self):
        logger.info("=" * 60)
        logger.info("AUDITORIA H3B CONTÍNUA — NAVEGAÇÃO DIRETA")
        logger.info("=" * 60)
        logger.info(f"Direção: {self.direction}")
        logger.info(f"Executores paralelos: {NUM_EXECUTORS}")
        logger.info(f"Limite auditorias: {'infinito' if self.num_audits == 0 else self.num_audits}")

        # Signals
        signal.signal(signal.SIGTERM, lambda s, f: setattr(self, 'running', False))
        signal.signal(signal.SIGINT, lambda s, f: setattr(self, 'running', False))

        # Browser
        self.scraper = BetinAsiaScraper()
        await self.scraper.start()
        await self.scraper.login()
        self.context = self.scraper._context
        logger.info("Login OK via proxy")

        # DB
        if self.save_to_db:
            self.db = Database()
            await self.db.connect()
            try:
                async with self.db.engine.begin() as conn:
                    await conn.execute(text(
                        "ALTER TABLE betslip_audit_results ADD COLUMN IF NOT EXISTS is_live BOOLEAN"
                    ))
            except:
                pass
            logger.info("Banco conectado")

        # Monitor
        self.monitor_page = self.scraper._page
        self._setup_ws_listener(self.monitor_page)
        await self.monitor_page.goto(FOOTBALL_URL)
        await self.monitor_page.wait_for_load_state("domcontentloaded")
        logger.info("Aguardando WebSocket...")
        await self.monitor_page.wait_for_timeout(5000)
        self._start_time = time.time()
        logger.info(f"WS: {self._ws_msg_count} msgs recebidas")

        # Executors
        for i in range(NUM_EXECUTORS):
            page = await self.context.new_page()
            page.set_default_timeout(PAGE_LOAD_TIMEOUT)
            self.executor_pages.append(page)
            self._executor_locks.append(asyncio.Lock())
        logger.info(f"{NUM_EXECUTORS} executores prontos")

        return True

    def _setup_ws_listener(self, page: Page):
        def on_ws(ws):
            self._ws_connected = True
            def on_frame(data):
                self._ws_messages.append(str(data))
                self._last_ws_msg_time = time.time()
                self._ws_msg_count += 1
            ws.on('framereceived', on_frame)
            ws.on('close', lambda: setattr(self, '_ws_connected', False))
        page.on('websocket', on_ws)

    async def run(self):
        ok = await self.start()
        if not ok:
            return

        audit_queue = asyncio.Queue()
        audited = set()

        tasks = [
            asyncio.create_task(self._monitor_loop(audit_queue, audited)),
            asyncio.create_task(self._maintenance_loop()),
        ]
        # Multiple executors
        for i in range(NUM_EXECUTORS):
            tasks.append(asyncio.create_task(self._executor_loop(i, audit_queue)))

        try:
            while self.running:
                if self.num_audits > 0 and len(self.results) >= self.num_audits:
                    break
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Interrompido")
        finally:
            self.running = False
            for t in tasks:
                t.cancel()

        self._print_summary()
        await self._cleanup()

    async def _cleanup(self):
        for page in self.executor_pages:
            try: await page.close()
            except: pass
        if self.scraper:
            await self.scraper.close()
        if self.db:
            await self.db.close()

    # ================================================================
    # MONITOR LOOP
    # ================================================================
    async def _monitor_loop(self, audit_queue: asyncio.Queue, audited: set):
        logger.info("Monitor loop iniciado")
        last_idx = 0

        while self.running:
            new = self._ws_messages[last_idx:]
            last_idx = len(self._ws_messages)

            if not new:
                await asyncio.sleep(0.05)  # 50ms polling
                continue

            for msg in new:
                try:
                    data = json.loads(msg)
                    if not isinstance(data, list):
                        continue
                    for item in data:
                        if not isinstance(item, list) or len(item) < 2:
                            continue

                        msg_type, msg_meta = item[0], item[1]
                        msg_data = item[2] if len(item) > 2 else {}

                        # Event info
                        if msg_type == 'event' and isinstance(msg_meta, list) and len(msg_meta) >= 2:
                            if msg_meta[0] == 'fb' and 'home' in msg_data:
                                eid = msg_meta[1]
                                kickoff = None
                                if 'start_ts' in msg_data:
                                    try:
                                        kickoff = datetime.fromisoformat(
                                            msg_data['start_ts'].replace('Z', '+00:00'))
                                    except: pass
                                self._events_info[eid] = {
                                    'home': msg_data.get('home', ''),
                                    'away': msg_data.get('away', ''),
                                    'league': msg_data.get('competition_name', ''),
                                    'kickoff': kickoff,
                                }

                        # Odds
                        if msg_type in ['offers_hcap', 'offers_event']:
                            if isinstance(msg_meta, list) and len(msg_meta) >= 3 and msg_meta[1] == 'fb':
                                eid = msg_meta[2]
                                # AH Full-Time
                                if 'ah' in msg_data:
                                    self._process_odds(eid, msg_data['ah'], 'AH', 'full_time',
                                                       audit_queue, audited)
                                # OU Full-Time
                                if 'ahou' in msg_data:
                                    self._process_odds(eid, msg_data['ahou'], 'OU', 'full_time',
                                                       audit_queue, audited, over_under=True)
                                # AH Half-Time
                                if 'ah_ht' in msg_data:
                                    self._process_odds(eid, msg_data['ah_ht'], 'AH', 'half_time',
                                                       audit_queue, audited)
                                # OU Half-Time
                                if 'ou_ht' in msg_data:
                                    self._process_odds(eid, msg_data['ou_ht'], 'OU', 'half_time',
                                                       audit_queue, audited, over_under=True)
                except:
                    continue

            # Periodic status
            if self.events_processed > 0 and self.events_processed % 500 == 0:
                logger.info(
                    f"Processados: {self.events_processed} | H3B: {self.h3b_detected} | "
                    f"Auditados: {len(self.results)} | WS: {self._ws_msg_count}")

    def _process_odds(self, event_id, odds_data, market_type, period,
                      queue, audited, over_under=False):
        lines = []
        if isinstance(odds_data, list) and len(odds_data) >= 2:
            if isinstance(odds_data[0], (int, float)):
                lines = [odds_data]
            elif isinstance(odds_data[0], list):
                lines = odds_data

        home_key = 'o' if over_under else 'h'
        away_key = 'u' if over_under else 'a'

        for line_data in lines:
            if len(line_data) < 2:
                continue
            line_val = line_data[0]
            odds_list = line_data[1] if len(line_data) > 1 else []

            home_odds = away_odds = 0
            if isinstance(odds_list, list):
                for o in odds_list:
                    if isinstance(o, list) and len(o) >= 2:
                        if o[0] == home_key: home_odds = float(o[1])
                        elif o[0] == away_key: away_odds = float(o[1])

            if home_odds <= 0 or away_odds <= 0:
                continue

            self.events_processed += 1

            try:
                if abs(float(line_val)) > MAX_AH_LINE:
                    continue
            except:
                pass

            det = self.detector.process_market_update(
                match_id=hash(event_id) % 1000000,
                market_type=f"{market_type}{'_HT' if period == 'half_time' else ''}",
                line=str(line_val),
                home_odd=home_odds,
                away_odd=away_odds,
            )

            for h3b in det.get("h3b_events", []):
                self.h3b_detected += 1
                if self.direction != "all" and h3b.direction_after != self.direction:
                    continue

                audit_key = f"{event_id}|{market_type}|{period}|{h3b.ah_line}|{h3b.side}"
                if audit_key in audited:
                    continue

                info = self._events_info.get(event_id, {})
                kickoff = info.get('kickoff')
                now = datetime.now(timezone.utc)
                is_live = kickoff <= now if kickoff else None

                queue.put_nowait({
                    'event_id': event_id,
                    'audit_key': audit_key,
                    'home_team': info.get('home', '?'),
                    'away_team': info.get('away', '?'),
                    'league': info.get('league', ''),
                    'kickoff': kickoff,
                    'is_live': is_live,
                    'market_type': market_type,
                    'market_period': period,
                    'line': str(h3b.ah_line),
                    'side': h3b.side,
                    'websocket_odd': h3b.odd_at_reversal,
                    'direction': h3b.direction_after,
                    'detected_at': time.time(),
                })
                audited.add(audit_key)

    # ================================================================
    # EXECUTOR LOOP (navegação direta)
    # ================================================================
    async def _executor_loop(self, executor_id: int, queue: asyncio.Queue):
        logger.info(f"Executor {executor_id} iniciado")
        page = self.executor_pages[executor_id]
        lock = self._executor_locks[executor_id]

        while self.running:
            if self.num_audits > 0 and len(self.results) >= self.num_audits:
                break
            try:
                h3b = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            async with lock:
                result = await self._execute_audit(page, h3b)

            self.results.append(result)
            if self.save_to_db:
                await self._save_result(result)

            # Log
            live_label = "LIVE" if result.is_live else "PRE" if result.is_live is not None else "?"
            if result.status == "OK":
                logger.info(
                    f"[OK][{live_label}] {result.home_team} vs {result.away_team} | "
                    f"{result.market_type} {result.line} {result.side} ({result.market_period}) | "
                    f"ws={result.websocket_odd:.3f} bs={result.betslip_odd:.3f} "
                    f"diff={result.difference_pct:+.2f}% lim=${result.betslip_limit:.0f} | "
                    f"lag={result.lag_total_ms}ms "
                    f"(nav={result.lag_navigate_ms} exp={result.lag_expand_ms} "
                    f"cl={result.lag_click_ms} bs={result.lag_betslip_ms}) | "
                    f"{len(self.results)}")
            else:
                logger.warning(
                    f"[{result.status}][{live_label}] {result.home_team} vs {result.away_team} | "
                    f"{result.market_type} {result.line} {result.side} | "
                    f"ws={result.websocket_odd:.3f} | lag={result.lag_total_ms}ms | "
                    f"{len(self.results)}")

            # Stats periódicas
            if len(self.results) % STATS_LOG_INTERVAL == 0:
                self._log_stats()

    async def _execute_audit(self, page: Page, h3b: dict) -> AuditResult:
        event_id = h3b['event_id']
        detected_at = h3b['detected_at']
        t0 = time.time()

        # Determina URL do jogo
        # Formato: /sportsbook/football/{league_code}/{event_id}
        league_name = h3b.get('league', '').lower().strip()
        league_code = LEAGUE_CODE_MAP.get(league_name)

        # Se não encontrou match exato, tenta match parcial
        if not league_code:
            for key, code in LEAGUE_CODE_MAP.items():
                if key in league_name or league_name in key:
                    league_code = code
                    break

        if league_code:
            game_url = f"{FOOTBALL_URL}/{league_code}/{event_id}"
        else:
            # Fallback: tenta URL genérica (pode não funcionar)
            game_url = f"{FOOTBALL_URL}/-/1/{event_id}"
            logger.debug(f"Liga sem código mapeado: '{h3b.get('league', '')}', usando URL genérica")

        base_result = {
            'timestamp': datetime.now(timezone.utc),
            'event_id': event_id,
            'home_team': h3b['home_team'],
            'away_team': h3b['away_team'],
            'league': h3b.get('league', ''),
            'market_type': h3b['market_type'],
            'market_period': h3b.get('market_period', 'full_time'),
            'line': h3b['line'],
            'side': h3b['side'],
            'websocket_odd': h3b['websocket_odd'],
            'is_live': h3b.get('is_live'),
            'direction': h3b.get('direction', 'up'),
        }

        try:
            # === NAVIGATE ===
            t_nav = time.time()
            await page.goto(game_url, wait_until="domcontentloaded", timeout=PAGE_LOAD_TIMEOUT)
            try:
                await page.wait_for_selector("text=Asian Handicap", timeout=5000)
            except:
                await page.wait_for_timeout(2000)
            lag_nav = int((time.time() - t_nav) * 1000)

            # === EXPAND ===
            t_exp = time.time()
            await page.evaluate("""
                () => {
                    let c = 0;
                    for (const el of document.querySelectorAll('span, button, div, a, [role="button"]')) {
                        const t = (el.innerText || '').trim().toLowerCase();
                        if ((t === 'show all lines' || t === 'show all') && el.offsetParent !== null) {
                            try { el.click(); c++; } catch(e) {}
                        }
                    }
                    return c;
                }
            """)
            await page.wait_for_timeout(1000)
            lag_exp = int((time.time() - t_exp) * 1000)

            # === CLICK ===
            t_click = time.time()
            clicked = await self._click_odd(page, h3b['line'], h3b['side'], h3b['market_type'])
            lag_click = int((time.time() - t_click) * 1000)

            if not clicked:
                lag_total = int((time.time() - detected_at) * 1000)
                return AuditResult(**base_result, status="CLICK_FAILED",
                                   lag_total_ms=lag_total, lag_navigate_ms=lag_nav,
                                   lag_expand_ms=lag_exp, lag_click_ms=lag_click)

            # === BETSLIP ===
            t_bs = time.time()
            await page.wait_for_timeout(2000)
            extractor = BetslipExtractor(page)
            betslip = await extractor.extract_best_odd()
            lag_bs = int((time.time() - t_bs) * 1000)
            lag_total = int((time.time() - detected_at) * 1000)

            await extractor.close_betslip()

            if not betslip or betslip.best_odd <= 0:
                return AuditResult(**base_result, status="EXTRACT_FAILED",
                                   lag_total_ms=lag_total, lag_navigate_ms=lag_nav,
                                   lag_expand_ms=lag_exp, lag_click_ms=lag_click,
                                   lag_betslip_ms=lag_bs)

            ws_odd = h3b['websocket_odd']
            diff = ((betslip.best_odd - ws_odd) / ws_odd) * 100

            return AuditResult(**base_result,
                               betslip_odd=betslip.best_odd,
                               betslip_limit=betslip.best_limit,
                               difference_pct=diff,
                               status="OK",
                               lag_total_ms=lag_total, lag_navigate_ms=lag_nav,
                               lag_expand_ms=lag_exp, lag_click_ms=lag_click,
                               lag_betslip_ms=lag_bs)

        except Exception as e:
            self.total_errors += 1
            lag_total = int((time.time() - detected_at) * 1000)
            logger.debug(f"Executor erro: {e}")
            return AuditResult(**base_result, status=f"ERROR",
                               lag_total_ms=lag_total)

    async def _click_odd(self, page: Page, line: str, side: str, market_type: str = "AH") -> bool:
        """Click robusto (mesma lógica do audit que funciona)."""
        try:
            line_float = float(line.replace(",", "."))
            line_variants = []
            if line_float == int(line_float):
                iv = int(line_float)
                if iv > 0: line_variants = [f"+{iv}", f"+{iv},0", f"+{iv}.0", str(iv)]
                elif iv < 0: line_variants = [str(iv), f"{iv},0", f"{iv}.0"]
                else: line_variants = ["0", "+0", "0,0", "0.0"]
            else:
                lc, ld = line.replace(".", ","), line.replace(",", ".")
                line_variants = [lc, ld]
                if line_float > 0: line_variants += ["+" + lc, "+" + ld]

            if market_type == "OU":
                section_names = ["Over/Under", "Mais/Menos"]
                home_label, away_label = "Over", "Under"
            else:
                section_names = ["Asian Handicap", "Handicap Asiático", "Handicap"]
                home_label, away_label = "Home", "Away"

            result = await page.evaluate("""(p) => {
                const lv=p.lv, side=p.side, sn=p.sn, hl=p.hl, al=p.al;
                function norm(t){return t.trim().replace(/\\s+/g,'').replace('.',',');}
                let sec=null;
                for(const h of document.querySelectorAll('div,span,h3,h4')){
                    const t=(h.innerText||'').trim();
                    if(sn.some(s=>t.includes(s))||t.includes('Handicap')||t.includes('Asian')){
                        let p=h.parentElement;
                        for(let i=0;i<10&&p;i++){
                            const pt=p.innerText||'';
                            if(pt.includes(hl)&&pt.includes(al)){sec=p;break;}
                            p=p.parentElement;
                        }
                        if(sec)break;
                    }
                }
                if(!sec)sec=document.body;
                for(const el of sec.querySelectorAll('span,div')){
                    const et=(el.innerText||'').trim();
                    if(et.length>10)continue;
                    let m=false;
                    for(const v of lv){if(et===v||norm(et)===norm(v)){m=true;break;}}
                    if(!m)continue;
                    let row=el.parentElement;
                    for(let i=0;i<6&&row;i++){
                        const rt=row.innerText||'';
                        if(rt.includes(hl)&&rt.includes(al)&&rt.split('\\n').length<15){
                            let has=false;
                            for(const v of lv){if(rt.includes(v)){has=true;break;}}
                            if(!has){row=row.parentElement;continue;}
                            const odds=[];const seen=new Set();
                            for(const c of row.querySelectorAll('div,span')){
                                const ct=(c.innerText||'').trim();
                                if(/^\\d+[.,]\\d{2,3}$/.test(ct)&&ct.length<10){
                                    const r=c.getBoundingClientRect();
                                    if(r.width>0&&r.height>0&&r.width<200){
                                        const k=Math.round(r.x)+'|'+ct;
                                        if(!seen.has(k)){seen.add(k);odds.push({el:c,x:r.x,text:ct});}
                                    }
                                }
                            }
                            if(odds.length>=2){
                                odds.sort((a,b)=>a.x-b.x);
                                const idx=(side==='home'||side==='over')?0:1;
                                const t=odds[idx];
                                if(t){
                                    t.el.scrollIntoView({behavior:'instant',block:'center'});
                                    try{t.el.parentElement.click();return true;}catch(e){}
                                    try{t.el.click();return true;}catch(e){}
                                }
                            }
                        }
                        row=row.parentElement;
                    }
                }
                return false;
            }""", {"lv": line_variants, "side": side, "sn": section_names,
                    "hl": home_label, "al": away_label})

            if result:
                await page.wait_for_timeout(1500)
            return bool(result)
        except:
            return False

    # ================================================================
    # MAINTENANCE
    # ================================================================
    async def _maintenance_loop(self):
        while self.running:
            await asyncio.sleep(WS_HEALTH_CHECK_INTERVAL)

            ws_age = time.time() - self._last_ws_msg_time if self._last_ws_msg_time > 0 else 999
            uptime = time.time() - self._start_time if self._start_time > 0 else 0
            ok_count = sum(1 for r in self.results if r.status == "OK")

            logger.info(
                f"[STATS] WS: {self._ws_msg_count} msgs, {self._ws_msg_count/max(1,uptime):.1f}/s, "
                f"last {ws_age:.0f}s | "
                f"Auditorias: {len(self.results)} (OK:{ok_count}) | "
                f"H3B: {self.h3b_detected} | Erros: {self.total_errors}")

            if ws_age > WS_RELOAD_INTERVAL:
                logger.warning("WS morto, recarregando monitor...")
                try:
                    self._ws_messages.clear()
                    await self.monitor_page.reload()
                    await self.monitor_page.wait_for_load_state("domcontentloaded")
                    await self.monitor_page.wait_for_timeout(3000)
                except Exception as e:
                    logger.error(f"Erro reload: {e}")

    # ================================================================
    # SAVE & STATS
    # ================================================================
    async def _save_result(self, r: AuditResult):
        if not self.db:
            return
        try:
            record = BetslipAuditResult(
                hypothesis_type="H3B",
                event_id=r.event_id,
                sport="football",
                league=r.league,
                home_team=r.home_team,
                away_team=r.away_team,
                match_info=f"{r.home_team} vs {r.away_team}",
                market_type=r.market_type,
                market_period=r.market_period,
                line=r.line,
                side=r.side,
                bet_description=f"{r.market_type} {r.line} {r.side} {r.market_period}",
                websocket_odd=r.websocket_odd,
                betslip_odd=r.betslip_odd,
                difference_pct=r.difference_pct,
                difference_absolute=(r.betslip_odd - r.websocket_odd) if r.betslip_odd else None,
                betslip_limit=r.betslip_limit,
                status=r.status,
                is_valid_opportunity=r.betslip_odd is not None and r.betslip_odd > 0,
                is_live=r.is_live,
                reversal_direction=r.direction,
                lag_detection_to_click_ms=r.lag_navigate_ms + r.lag_expand_ms + r.lag_click_ms,
                lag_click_to_betslip_ms=r.lag_betslip_ms,
                audit_total_duration_ms=r.lag_total_ms,
                audit_version="v3.0-fast-direct",
            )
            async with self.db.async_session() as session:
                session.add(record)
                await session.commit()
        except Exception as e:
            logger.warning(f"Erro salvando: {e}")

    def _log_stats(self):
        ok = [r for r in self.results if r.status == "OK"]
        if not ok:
            return
        lags = [r.lag_total_ms for r in ok]
        diffs = [r.difference_pct for r in ok if r.difference_pct is not None]

        logger.info("=" * 50)
        logger.info(f"STATS — {len(self.results)} auditorias ({len(ok)} OK)")
        logger.info(f"  Lag total: min={min(lags)}ms med={sorted(lags)[len(lags)//2]}ms max={max(lags)}ms")
        if diffs:
            logger.info(f"  Diff WS/BS: avg={sum(diffs)/len(diffs):+.2f}% med={sorted(diffs)[len(diffs)//2]:+.2f}%")
        logger.info("=" * 50)

    def _print_summary(self):
        print("\n" + "=" * 60)
        print("RESUMO FINAL")
        print("=" * 60)
        ok = [r for r in self.results if r.status == "OK"]
        failed = [r for r in self.results if r.status != "OK"]
        print(f"  Total: {len(self.results)} | OK: {len(ok)} | Falhas: {len(failed)}")
        print(f"  H3B detectados: {self.h3b_detected} | Processados: {self.events_processed}")

        if ok:
            lags = [r.lag_total_ms for r in ok]
            diffs = [r.difference_pct for r in ok if r.difference_pct is not None]
            print(f"\n  Lag (OK): min={min(lags)}ms med={sorted(lags)[len(lags)//2]}ms avg={sum(lags)//len(lags)}ms max={max(lags)}ms")
            if diffs:
                print(f"  Diff: avg={sum(diffs)/len(diffs):+.2f}% med={sorted(diffs)[len(diffs)//2]:+.2f}%")

        if failed:
            by_status = {}
            for r in failed:
                by_status[r.status] = by_status.get(r.status, 0) + 1
            print(f"\n  Falhas:")
            for s, n in sorted(by_status.items(), key=lambda x: -x[1]):
                print(f"    {s}: {n}")


async def main():
    parser = argparse.ArgumentParser(description="Auditoria H3B contínua rápida")
    parser.add_argument("--num-audits", type=int, default=0, help="0 = infinito")
    parser.add_argument("--direction", choices=["up", "down", "all"], default="up")
    parser.add_argument("--no-db", action="store_true")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr,
               format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level:<7}</level> | <level>{message}</level>",
               level="INFO",
               filter=lambda r: "H6:" not in r["message"] and "H1:" not in r["message"])
    logger.add("logs/fast_audit_{time:YYYY-MM-DD}.log", rotation="00:00", retention="60 days", level="DEBUG")

    test = FastAuditContinuous(
        num_audits=args.num_audits,
        direction=args.direction,
        save_to_db=not args.no_db,
    )
    await test.run()


if __name__ == "__main__":
    asyncio.run(main())
