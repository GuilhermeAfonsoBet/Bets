#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TESTE DE VELOCIDADE: Audit H3B com abas pré-abertas (1 liga)

Arquitetura:
  - Tab 0 (MONITOR): Fica em /sportsbook/football, escuta WebSocket permanente
  - Tab 1..N (JOGOS): Uma aba por jogo da liga, pré-aberta e linhas expandidas
  - Task MANUTENÇÃO: Re-expande linhas periodicamente (a cada 60s)
  - Task EXECUTOR: Quando H3B detectado, clica direto na aba do jogo

Fluxo rápido:
  1. Monitor detecta H3B via WS      →  0ms (contínuo)
  2. Switch para aba do jogo          →  ~100ms
  3. Click na odd (já expandida)      →  ~500ms
  4. Betslip carrega                  →  ~1.5s
  5. Extrai dados                     →  ~5ms
  TOTAL ESTIMADO:                     ~2-3s

Uso:
    DISPLAY=:99 python fast_audit_test.py --league "England Premier League"
    DISPLAY=:99 python fast_audit_test.py --league "Spain La Liga" --num-audits 20
"""

import asyncio
import argparse
import json
import re
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


# URLs do BetinAsia
BASE_URL = "https://black.betinasia.com"
FOOTBALL_URL = f"{BASE_URL}/sportsbook/football"

# Mapeamento de ligas
LEAGUE_CODES = {
    "England Premier League": "XE/1",
    "England Championship": "XE/2",
    "Germany Bundesliga": "DE/12",
    "Spain La Liga": "ES/16",
    "Italy Serie A": "IT/19",
    "France Ligue 1": "FR/38",
    "Netherlands Eredivisie": "NL/1",
    "Portugal Primeira Liga": "PT/1",
    "UEFA Champions League": "XE/5",
    "UEFA Europa League": "XE/6",
}

# Config
WS_HEALTH_CHECK_INTERVAL = 10  # Segundos: alerta se WS sem mensagens
WS_RELOAD_INTERVAL = 30  # Segundos: reload se WS morreu (odds mudam a cada segundo)
EXPAND_CHECK_INTERVAL = 15  # Segundos: re-expande linhas nas abas
MAX_AH_LINE = 2.0  # Filtro de linhas extremas


@dataclass
class GameTab:
    """Representa uma aba com um jogo pré-aberto."""
    page: Page
    event_id: str
    home_team: str
    away_team: str
    league: str
    url: str
    kickoff: Optional[datetime] = None
    lines_expanded: bool = False
    last_expand: float = 0


@dataclass
class FastAuditResult:
    """Resultado de uma auditoria rápida."""
    timestamp: datetime
    event_id: str
    home_team: str
    away_team: str
    league: str
    market_type: str
    line: str
    side: str
    websocket_odd: float
    betslip_odd: Optional[float] = None
    betslip_limit: Optional[float] = None
    difference_pct: Optional[float] = None
    status: str = ""
    is_live: Optional[bool] = None
    # Timing granular (ms)
    lag_detect_to_switch_ms: int = 0
    lag_switch_to_click_ms: int = 0
    lag_click_to_betslip_ms: int = 0
    lag_total_ms: int = 0


class FastAuditTest:
    """Teste de velocidade com abas pré-abertas."""

    def __init__(self, league: str, num_audits: int = 50, max_tabs: int = 8, save_to_db: bool = True):
        self.league = league
        self.league_code = LEAGUE_CODES.get(league)
        self.max_tabs = max_tabs
        self.num_audits = num_audits
        self.save_to_db = save_to_db

        self.scraper: Optional[BetinAsiaScraper] = None
        self.monitor_page: Optional[Page] = None
        self.game_tabs: Dict[str, GameTab] = {}  # event_id → GameTab
        self.context = None
        self.db: Optional[Database] = None

        # WebSocket state
        self._ws_messages: List[str] = []
        self._ws_connected = False
        self._last_ws_message_time: float = 0
        self._ws_message_count: int = 0

        # Hypothesis detector
        self.detector = HypothesisDetector()

        # Results
        self.results: List[FastAuditResult] = []
        self.events_processed: int = 0
        self.h3b_detected: int = 0

        # Lock: evita manutenção e executor acessarem mesma aba
        self._audit_lock = asyncio.Lock()

    async def start(self):
        """Inicia browser, login, abre monitor e abas dos jogos."""
        logger.info("=" * 60)
        logger.info(f"TESTE DE VELOCIDADE - {self.league}")
        logger.info("=" * 60)

        # Browser
        self.scraper = BetinAsiaScraper()
        await self.scraper.start()
        await self.scraper.login()
        self.context = self.scraper._context
        logger.info("Login OK")

        # Banco de dados
        if self.save_to_db:
            self.db = Database()
            await self.db.connect()
            try:
                async with self.db.engine.begin() as conn:
                    await conn.execute(
                        text("ALTER TABLE betslip_audit_results ADD COLUMN IF NOT EXISTS is_live BOOLEAN")
                    )
            except:
                pass
            logger.info("Banco conectado")

        # Monitor page (Tab 0) — fica na página de futebol escutando WS
        self.monitor_page = self.scraper._page
        await self._setup_ws_listener(self.monitor_page)
        await self.monitor_page.goto(FOOTBALL_URL)
        await self.monitor_page.wait_for_load_state("domcontentloaded")
        logger.info("Monitor page: aguardando WebSocket...")
        await self.monitor_page.wait_for_timeout(5000)
        self._start_time = time.time()
        logger.info(f"Monitor page: {self._ws_message_count} mensagens WS recebidas")

        # Lista todas as ligas disponíveis (para ajudar na escolha)
        await self._list_available_leagues()

        # Descobre jogos da liga via WebSocket
        game_urls = await self._discover_game_urls()

        if not game_urls:
            logger.error(f"Nenhum jogo encontrado para {self.league}")
            return False

        logger.info(f"Encontrados {len(game_urls)} jogos para {self.league}")

        # Abre abas dos jogos
        await self._open_game_tabs(game_urls)

        logger.info(f"{len(self.game_tabs)} abas de jogos abertas e expandidas")
        return True

    async def _setup_ws_listener(self, page: Page):
        """Configura listener de WebSocket numa página."""
        def on_ws(ws):
            self._ws_connected = True

            def on_frame(data):
                self._ws_messages.append(str(data))
                self._last_ws_message_time = time.time()
                self._ws_message_count += 1

            ws.on('framereceived', on_frame)
            ws.on('close', lambda: setattr(self, '_ws_connected', False))

        page.on('websocket', on_ws)

    async def _list_available_leagues(self):
        """Lista todas as ligas com jogos no WebSocket (ajuda a escolher)."""
        leagues = {}
        for msg in self._ws_messages:
            try:
                data = json.loads(msg)
                if not isinstance(data, list):
                    continue
                for item in data:
                    if not isinstance(item, list) or len(item) < 3:
                        continue
                    if item[0] == 'event' and isinstance(item[1], list) and len(item[1]) >= 2:
                        if item[1][0] == 'fb' and 'competition_name' in item[2]:
                            league = item[2]['competition_name']
                            if league:
                                leagues[league] = leagues.get(league, 0) + 1
            except:
                continue

        sorted_leagues = sorted(leagues.items(), key=lambda x: -x[1])
        logger.info(f"Ligas disponiveis ({len(sorted_leagues)} total):")
        for league, count in sorted_leagues[:25]:
            marker = " <<<" if self.league.lower() in league.lower() or league.lower() in self.league.lower() else ""
            logger.info(f"  {count:3d} jogos: {league}{marker}")

    async def _discover_game_urls(self) -> List[dict]:
        """Descobre URLs dos jogos da liga a partir dos dados do WebSocket."""
        games = []
        events = {}

        # Parseia mensagens WS para encontrar jogos da liga
        for msg in self._ws_messages:
            try:
                data = json.loads(msg)
                if not isinstance(data, list):
                    continue
                for item in data:
                    if not isinstance(item, list) or len(item) < 2:
                        continue
                    msg_type = item[0]
                    msg_meta = item[1]
                    msg_data = item[2] if len(item) > 2 else {}

                    if msg_type == 'event' and isinstance(msg_meta, list) and len(msg_meta) >= 2:
                        if msg_meta[0] == 'fb' and 'home' in msg_data:
                            event_id = msg_meta[1]
                            league = msg_data.get('competition_name', '')
                            kickoff = None
                            if 'start_ts' in msg_data:
                                try:
                                    kickoff = datetime.fromisoformat(
                                        msg_data['start_ts'].replace('Z', '+00:00')
                                    )
                                except:
                                    pass

                            events[event_id] = {
                                'event_id': event_id,
                                'home': msg_data.get('home', ''),
                                'away': msg_data.get('away', ''),
                                'league': league,
                                'kickoff': kickoff,
                            }
            except:
                continue

        # Filtra pela liga — match preciso
        target_league = self.league.lower().strip()
        for eid, info in events.items():
            ws_league = info.get('league', '').lower().strip()
            # Match: nome exato OU nome do WS contém nome do target E vice-versa
            is_match = (
                ws_league == target_league
                or target_league in ws_league
                or ws_league in target_league
            )
            # Exclui segunda divisão se buscamos primeira
            if is_match and 'second' in ws_league and 'second' not in target_league:
                is_match = False
            if is_match and 'segunda' in ws_league and 'segunda' not in target_league:
                is_match = False
            if is_match and 'rfef' in ws_league:
                is_match = False
            if is_match:
                # Constroi URL do jogo
                # Formato: /sportsbook/football/{league_code}/{event_id}
                url = f"{FOOTBALL_URL}/{self.league_code}/{eid}"
                games.append({
                    'event_id': eid,
                    'home': info['home'],
                    'away': info['away'],
                    'league': info['league'],
                    'kickoff': info.get('kickoff'),
                    'url': url,
                })

        return games

    async def _open_game_tabs(self, games: list):
        """Abre uma aba para cada jogo e expande linhas."""
        for i, game in enumerate(games):
            if len(self.game_tabs) >= self.max_tabs:
                logger.info(f"Limite de {self.max_tabs} abas atingido, parando abertura")
                break
            try:
                page = await self.context.new_page()
                page.set_default_timeout(15000)
                await page.goto(game['url'], wait_until="domcontentloaded", timeout=15000)

                # Espera até Asian Handicap aparecer (máx 8s)
                try:
                    await page.wait_for_selector("text=Asian Handicap", timeout=8000)
                except:
                    # Fallback: espera fixa
                    await page.wait_for_timeout(3000)

                body = await page.inner_text("body")
                if "Asian Handicap" not in body and "Over/Under" not in body:
                    logger.warning(f"Jogo nao carregou: {game['home']} vs {game['away']}")
                    await page.close()
                    continue

                # Espera mais um pouco para botões "Show all" renderizarem
                await page.wait_for_timeout(1500)

                # Expande linhas
                expanded = await self._expand_lines(page)

                tab = GameTab(
                    page=page,
                    event_id=game['event_id'],
                    home_team=game['home'],
                    away_team=game['away'],
                    league=game['league'],
                    url=game['url'],
                    kickoff=game.get('kickoff'),
                    lines_expanded=expanded > 0,
                    last_expand=time.time(),
                )
                self.game_tabs[game['event_id']] = tab

                logger.info(f"  Tab aberta: {game['home']} vs {game['away']} ({expanded} seções expandidas)")

            except Exception as e:
                logger.warning(f"Erro abrindo tab {game['home']} vs {game['away']}: {e}")

    async def _expand_lines(self, page: Page) -> int:
        """Expande todas as linhas via JavaScript. Retorna quantas expandiu."""
        try:
            result = await page.evaluate("""
                () => {
                    let clicked = 0;
                    const els = document.querySelectorAll('span, button, div, a, [role="button"]');
                    for (const el of els) {
                        const text = (el.innerText || '').trim().toLowerCase();
                        if ((text === 'show all lines' || text === 'show all' ||
                             text === 'mostrar todas as linhas' || text === 'mostrar') &&
                            el.offsetParent !== null) {
                            try { el.click(); clicked++; } catch(e) {}
                        }
                    }
                    return clicked;
                }
            """)
            if result > 0:
                await page.wait_for_timeout(1000)
            return result
        except:
            return 0

    async def _maintenance_loop(self):
        """Loop de manutenção: re-expande linhas e verifica saúde do WS."""
        while True:
            await asyncio.sleep(EXPAND_CHECK_INTERVAL)

            # Re-expande linhas (apenas se executor não está rodando)
            if not self._audit_lock.locked():
                total_expanded = 0
                for eid, tab in self.game_tabs.items():
                    try:
                        expanded = await self._expand_lines(tab.page)
                        total_expanded += expanded
                        if expanded > 0:
                            logger.info(f"  Re-expandiu {expanded} seções: {tab.home_team} vs {tab.away_team}")
                            tab.lines_expanded = True
                            tab.last_expand = time.time()
                    except Exception as e:
                        logger.warning(f"  Erro re-expandindo {tab.home_team} vs {tab.away_team}: {e}")
                if total_expanded > 0:
                    logger.info(f"  Total re-expandido: {total_expanded} seções")
            else:
                logger.debug("  Manutencao pulada (executor ativo)")

            # Verifica saúde do WebSocket
            ws_age = time.time() - self._last_ws_message_time if self._last_ws_message_time > 0 else 999
            msgs_per_sec = self._ws_message_count / max(1, time.time() - self._start_time) if hasattr(self, '_start_time') else 0
            logger.info(
                f"[MANUTENCAO] WS: {self._ws_message_count} msgs total, "
                f"{msgs_per_sec:.1f} msg/s, ultima msg {ws_age:.0f}s atras | "
                f"Abas: {len(self.game_tabs)} | Auditados: {len(self.results)}/{self.num_audits}"
            )
            if ws_age > WS_HEALTH_CHECK_INTERVAL:
                logger.warning(f"WebSocket sem mensagens ha {ws_age:.0f}s — POSSIVEL MORTE SILENCIOSA")

            # Reload forçado do monitor se WS parou
            if ws_age > WS_RELOAD_INTERVAL:
                logger.warning("WebSocket morto — recarregando monitor...")
                try:
                    self._ws_messages.clear()
                    await self.monitor_page.reload()
                    await self.monitor_page.wait_for_load_state("domcontentloaded")
                    await self.monitor_page.wait_for_timeout(3000)
                    logger.info(f"Monitor recarregado. WS conectado: {self._ws_connected}")
                except Exception as e:
                    logger.error(f"Erro recarregando monitor: {e}")

    async def _monitor_loop(self, audit_queue: asyncio.Queue, audited: set):
        """Loop do monitor: processa WS e detecta H3B continuamente."""
        logger.info("Monitor loop iniciado — escutando WebSocket...")

        last_process_idx = 0

        while len(self.results) < self.num_audits:
            # Processa apenas mensagens novas (desde último check)
            new_messages = self._ws_messages[last_process_idx:]
            last_process_idx = len(self._ws_messages)

            if not new_messages:
                await asyncio.sleep(0.1)  # 100ms polling
                continue

            # Parseia e detecta H3B
            events_info = {}
            for msg in new_messages:
                try:
                    data = json.loads(msg)
                    if not isinstance(data, list):
                        continue
                    for item in data:
                        if not isinstance(item, list) or len(item) < 2:
                            continue

                        msg_type = item[0]
                        msg_meta = item[1]
                        msg_data = item[2] if len(item) > 2 else {}

                        # Info do evento
                        if msg_type == 'event' and isinstance(msg_meta, list) and len(msg_meta) >= 2:
                            if msg_meta[0] == 'fb' and 'home' in msg_data:
                                eid = msg_meta[1]
                                kickoff = None
                                if 'start_ts' in msg_data:
                                    try:
                                        kickoff = datetime.fromisoformat(
                                            msg_data['start_ts'].replace('Z', '+00:00')
                                        )
                                    except:
                                        pass
                                events_info[eid] = {
                                    'home': msg_data.get('home', ''),
                                    'away': msg_data.get('away', ''),
                                    'league': msg_data.get('competition_name', ''),
                                    'kickoff': kickoff,
                                }

                        # Odds AH
                        if msg_type in ['offers_hcap', 'offers_event']:
                            if isinstance(msg_meta, list) and len(msg_meta) >= 3 and msg_meta[1] == 'fb':
                                eid = msg_meta[2]
                                if 'ah' in msg_data:
                                    self._process_ah_odds(eid, msg_data['ah'], events_info, audit_queue, audited)
                except:
                    continue

            # Status periódico
            if self.events_processed % 500 == 0 and self.events_processed > 0:
                logger.info(
                    f"Processados: {self.events_processed} | "
                    f"H3B: {self.h3b_detected} | "
                    f"Auditados: {len(self.results)}/{self.num_audits} | "
                    f"WS msgs: {self._ws_message_count}"
                )

    def _process_ah_odds(self, event_id: str, ah_data, events_info: dict,
                         audit_queue: asyncio.Queue, audited: set):
        """Processa odds AH e detecta H3B."""
        lines = []
        if isinstance(ah_data, list) and len(ah_data) >= 2:
            if isinstance(ah_data[0], (int, float)):
                lines = [ah_data]
            elif isinstance(ah_data[0], list):
                lines = ah_data

        for line_data in lines:
            if len(line_data) < 2:
                continue
            line_val = line_data[0]
            odds_list = line_data[1] if len(line_data) > 1 else []

            home_odds = away_odds = 0
            if isinstance(odds_list, list):
                for o in odds_list:
                    if isinstance(o, list) and len(o) >= 2:
                        if o[0] == 'h': home_odds = float(o[1])
                        elif o[0] == 'a': away_odds = float(o[1])

            if home_odds <= 0 or away_odds <= 0:
                continue

            self.events_processed += 1

            # Filtro de linha extrema
            try:
                if abs(float(line_val)) > MAX_AH_LINE:
                    continue
            except:
                pass

            # Detecta H3B
            det = self.detector.process_market_update(
                match_id=hash(event_id) % 1000000,
                market_type="AH",
                line=str(line_val),
                home_odd=home_odds,
                away_odd=away_odds,
            )

            for h3b in det.get("h3b_events", []):
                self.h3b_detected += 1

                if h3b.direction_after != "up":
                    continue

                # Só audita se temos aba aberta para este jogo
                if event_id not in self.game_tabs:
                    continue

                audit_key = f"{event_id}|AH|{h3b.ah_line}|{h3b.side}"
                if audit_key in audited:
                    continue

                info = events_info.get(event_id, {})
                kickoff = info.get('kickoff') or self.game_tabs[event_id].kickoff
                now = datetime.now(timezone.utc)
                is_live = kickoff <= now if kickoff else None

                audit_queue.put_nowait({
                    'event_id': event_id,
                    'audit_key': audit_key,
                    'line': str(h3b.ah_line),
                    'side': h3b.side,
                    'websocket_odd': h3b.odd_at_reversal,
                    'is_live': is_live,
                    'detected_at': time.time(),
                })
                audited.add(audit_key)

    async def _executor_loop(self, audit_queue: asyncio.Queue):
        """Loop do executor: processa fila de H3Bs, clica nas abas pré-abertas."""
        logger.info("Executor loop iniciado")

        while len(self.results) < self.num_audits:
            try:
                h3b = await asyncio.wait_for(audit_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            async with self._audit_lock:
                result = await self._execute_audit(h3b)
            self.results.append(result)

            # Salva no banco
            if self.save_to_db:
                await self._save_result(result)

            # Log
            if result.status == "OK":
                logger.info(
                    f"[OK] {result.home_team} vs {result.away_team} | "
                    f"AH {result.line} {result.side} | "
                    f"ws={result.websocket_odd:.3f} bs={result.betslip_odd:.3f} diff={result.difference_pct:+.2f}% | "
                    f"lag={result.lag_total_ms}ms "
                    f"(sw={result.lag_detect_to_switch_ms} cl={result.lag_switch_to_click_ms} "
                    f"bs={result.lag_click_to_betslip_ms}) | "
                    f"{len(self.results)}/{self.num_audits}"
                )
            else:
                logger.warning(
                    f"[{result.status}] {result.home_team} vs {result.away_team} | "
                    f"AH {result.line} {result.side} | ws={result.websocket_odd:.3f} | "
                    f"lag={result.lag_total_ms}ms "
                    f"(sw={result.lag_detect_to_switch_ms} cl={result.lag_switch_to_click_ms} "
                    f"bs={result.lag_click_to_betslip_ms}) | "
                    f"{len(self.results)}/{self.num_audits}"
                )

    async def _execute_audit(self, h3b: dict) -> FastAuditResult:
        """Executa auditoria: switch tab → click odd → extract betslip."""
        event_id = h3b['event_id']
        tab = self.game_tabs[event_id]
        detected_at = h3b['detected_at']

        t0 = time.time()

        # === SWITCH para aba do jogo ===
        try:
            await tab.page.bring_to_front()
        except:
            pass
        t_switch = time.time()
        lag_switch = int((t_switch - detected_at) * 1000)

        # === CLICK na odd ===
        line = h3b['line']
        side = h3b['side']
        logger.debug(f"Clicando AH {line} {side} em {tab.home_team} vs {tab.away_team}")
        clicked = await self._click_odd(tab.page, line, side)
        t_click = time.time()
        lag_click = int((t_click - t_switch) * 1000)
        logger.debug(f"Click resultado: {clicked}, lag={lag_click}ms")

        if not clicked:
            lag_total = int((time.time() - detected_at) * 1000)
            return FastAuditResult(
                timestamp=datetime.now(timezone.utc),
                event_id=event_id,
                home_team=tab.home_team,
                away_team=tab.away_team,
                league=tab.league,
                market_type="AH",
                line=line,
                side=side,
                websocket_odd=h3b['websocket_odd'],
                status="CLICK_FAILED",
                is_live=h3b.get('is_live'),
                lag_detect_to_switch_ms=lag_switch,
                lag_switch_to_click_ms=lag_click,
                lag_total_ms=lag_total,
            )

        # === BETSLIP ===
        await tab.page.wait_for_timeout(2000)

        # Dump do conteúdo do betslip para debug
        betslip_dump = await tab.page.evaluate("""
            () => {
                // Procura painéis candidatos
                const selectors = ['[class*="betslip"]', '[class*="slip"]', '[class*="sidebar"]', '[class*="panel"]', 'aside'];
                let found = [];
                for (const sel of selectors) {
                    for (const el of document.querySelectorAll(sel)) {
                        const t = (el.innerText || '').trim();
                        if (t.length > 20 && t.length < 5000) {
                            found.push({selector: sel, className: el.className, length: t.length, text: t.substring(0, 500)});
                        }
                    }
                }
                // Se nada encontrou nos seletores, pega a parte direita da página
                if (found.length === 0) {
                    const body = document.body.innerText || '';
                    const betslipIdx = body.indexOf('Betslip');
                    if (betslipIdx > -1) {
                        found.push({selector: 'body-betslip', className: '', length: 2000, text: body.substring(betslipIdx, betslipIdx + 1000)});
                    } else {
                        // Últimos 1000 chars do body (betslip geralmente está no final)
                        found.push({selector: 'body-tail', className: '', length: body.length, text: body.substring(Math.max(0, body.length - 1000))});
                    }
                }
                return found;
            }
        """)
        logger.info(f"  BETSLIP DUMP ({len(betslip_dump)} painéis encontrados):")
        for panel in betslip_dump[:3]:
            logger.info(f"    Selector: {panel['selector']} | Class: {panel['className'][:50]} | Len: {panel['length']}")
            # Mostra primeiras 200 chars do texto
            text_preview = panel['text'][:200].replace('\n', ' | ')
            logger.info(f"    Texto: {text_preview}")

        # Screenshot para debug
        try:
            screenshot_name = f"logs/betslip_debug_{event_id.replace(',','_')}_{int(time.time())}.png"
            await tab.page.screenshot(path=screenshot_name, full_page=False)
            logger.info(f"  Screenshot salvo: {screenshot_name}")
        except Exception as e:
            logger.debug(f"Erro screenshot: {e}")

        extractor = BetslipExtractor(tab.page)
        betslip = await extractor.extract_best_odd()
        logger.debug(f"Betslip extraido: {betslip is not None}, data={betslip}")
        t_betslip = time.time()
        lag_betslip = int((t_betslip - t_click) * 1000)
        lag_total = int((t_betslip - detected_at) * 1000)

        # Fecha betslip
        await extractor.close_betslip()

        if not betslip:
            return FastAuditResult(
                timestamp=datetime.now(timezone.utc),
                event_id=event_id,
                home_team=tab.home_team,
                away_team=tab.away_team,
                league=tab.league,
                market_type="AH",
                line=line,
                side=side,
                websocket_odd=h3b['websocket_odd'],
                status="EXTRACT_FAILED",
                is_live=h3b.get('is_live'),
                lag_detect_to_switch_ms=lag_switch,
                lag_switch_to_click_ms=lag_click,
                lag_click_to_betslip_ms=lag_betslip,
                lag_total_ms=lag_total,
            )

        ws_odd = h3b['websocket_odd']
        best_odd = betslip.best_odd
        diff_pct = ((best_odd - ws_odd) / ws_odd) * 100

        return FastAuditResult(
            timestamp=datetime.now(timezone.utc),
            event_id=event_id,
            home_team=tab.home_team,
            away_team=tab.away_team,
            league=tab.league,
            market_type="AH",
            line=line,
            side=side,
            websocket_odd=ws_odd,
            betslip_odd=best_odd,
            betslip_limit=betslip.best_limit,
            difference_pct=diff_pct,
            status="OK",
            is_live=h3b.get('is_live'),
            lag_detect_to_switch_ms=lag_switch,
            lag_switch_to_click_ms=lag_click,
            lag_click_to_betslip_ms=lag_betslip,
            lag_total_ms=lag_total,
        )

    async def _click_odd(self, page: Page, line: str, side: str, market_type: str = "AH") -> bool:
        """
        Clica numa odd específica para abrir o betslip.
        CÓPIA EXATA da lógica _click_specific_odd do audit_h3b_betslip.py (que funciona).
        """
        try:
            line_float = float(line.replace(",", "."))
            line_variants = []
            if line_float == int(line_float):
                int_val = int(line_float)
                if int_val > 0:
                    line_variants.extend([f"+{int_val}", f"+{int_val},0", f"+{int_val}.0", str(int_val)])
                elif int_val < 0:
                    line_variants.extend([str(int_val), f"{int_val},0", f"{int_val}.0"])
                else:
                    line_variants.extend(["0", "+0", "0,0", "0.0"])
            else:
                line_comma = line.replace(".", ",")
                line_dot = line.replace(",", ".")
                line_variants.append(line_comma)
                line_variants.append(line_dot)
                if line_float > 0:
                    line_variants.append("+" + line_comma)
                    line_variants.append("+" + line_dot)

            if market_type == "OU":
                section_names = ["Over/Under", "Mais/Menos"]
                home_label = "Over"
                away_label = "Under"
            else:
                section_names = ["Asian Handicap", "Handicap Asiático", "Handicap"]
                home_label = "Home"
                away_label = "Away"

            clicked = await page.evaluate("""
                (params) => {
                    const lineVariants = params.lineVariants;
                    const side = params.side;
                    const sectionNames = params.sectionNames;
                    const homeLabel = params.homeLabel;
                    const awayLabel = params.awayLabel;

                    function normalizeLineText(text) {
                        return text.trim().replace(/\\s+/g, '').replace('.', ',');
                    }

                    function matchesSection(text) {
                        for (const name of sectionNames) {
                            if (text.includes(name)) return true;
                        }
                        return false;
                    }

                    // Encontra seção Asian Handicap ou Over/Under
                    let sectionContainer = null;
                    const headers = document.querySelectorAll('div, span, h3, h4');
                    for (const h of headers) {
                        const text = (h.innerText || '').trim();
                        if (matchesSection(text) || text.includes('Handicap') || text.includes('Asian')) {
                            let parent = h.parentElement;
                            for (let i = 0; i < 10 && parent; i++) {
                                const parentText = parent.innerText || '';
                                if (parentText.includes(homeLabel) && parentText.includes(awayLabel)) {
                                    sectionContainer = parent;
                                    break;
                                }
                                parent = parent.parentElement;
                            }
                            if (sectionContainer) break;
                        }
                    }
                    if (!sectionContainer) sectionContainer = document.body;

                    // Encontra a linha específica
                    const allElements = sectionContainer.querySelectorAll('span, div');
                    for (const el of allElements) {
                        const elText = (el.innerText || '').trim();
                        if (elText.length > 10) continue;

                        let isLineMatch = false;
                        for (const variant of lineVariants) {
                            if (elText === variant || normalizeLineText(elText) === normalizeLineText(variant)) {
                                isLineMatch = true;
                                break;
                            }
                        }
                        if (!isLineMatch) continue;

                        // Busca row container com Home/Away
                        let rowContainer = el.parentElement;
                        for (let i = 0; i < 6 && rowContainer; i++) {
                            const rowText = rowContainer.innerText || '';
                            if (rowText.includes(homeLabel) && rowText.includes(awayLabel) &&
                                rowText.split('\\n').length < 15) {

                                let hasOurLine = false;
                                for (const variant of lineVariants) {
                                    if (rowText.includes(variant)) { hasOurLine = true; break; }
                                }
                                if (!hasOurLine) { rowContainer = rowContainer.parentElement; continue; }

                                // Encontra odds clicáveis
                                const clickableElements = rowContainer.querySelectorAll('div, span');
                                const oddElements = [];
                                for (const child of clickableElements) {
                                    const childText = (child.innerText || '').trim();
                                    if (/^\\d+[.,]\\d{2,3}$/.test(childText) && childText.length < 10) {
                                        const rect = child.getBoundingClientRect();
                                        if (rect.width > 0 && rect.height > 0 && rect.width < 200) {
                                            oddElements.push({ el: child, x: rect.x, text: childText });
                                        }
                                    }
                                }

                                // Remove duplicatas
                                const uniqueOdds = [];
                                const seenKeys = new Set();
                                for (const odd of oddElements) {
                                    const key = Math.round(odd.x) + '|' + odd.text;
                                    if (!seenKeys.has(key)) { seenKeys.add(key); uniqueOdds.push(odd); }
                                }

                                if (uniqueOdds.length >= 2) {
                                    uniqueOdds.sort((a, b) => a.x - b.x);
                                    const targetIdx = (side === 'home' || side === 'over') ? 0 : 1;
                                    const targetEl = uniqueOdds[targetIdx];
                                    if (targetEl) {
                                        targetEl.el.scrollIntoView({ behavior: 'instant', block: 'center' });
                                        try { targetEl.el.parentElement.click(); return { success: true, odd: targetEl.text }; }
                                        catch(e) {}
                                        try { targetEl.el.click(); return { success: true, odd: targetEl.text }; }
                                        catch(e) {}
                                    }
                                }
                            }
                            rowContainer = rowContainer.parentElement;
                        }
                    }
                    return { success: false };
                }
            """, {
                "lineVariants": line_variants,
                "side": side,
                "sectionNames": section_names,
                "homeLabel": home_label,
                "awayLabel": away_label
            })

            if clicked and clicked.get('success'):
                logger.debug(f"Click OK: odd={clicked.get('odd')}")
                await page.wait_for_timeout(1500)
                return True
            return False

        except Exception as e:
            logger.error(f"Erro click: {e}")
            return False

    async def _save_result(self, r: FastAuditResult):
        """Salva resultado no banco."""
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
                market_period="full_time",
                line=r.line,
                side=r.side,
                bet_description=f"AH {r.line} {r.side}",
                websocket_odd=r.websocket_odd,
                betslip_odd=r.betslip_odd,
                difference_pct=r.difference_pct,
                difference_absolute=(r.betslip_odd - r.websocket_odd) if r.betslip_odd else None,
                betslip_limit=r.betslip_limit,
                status=r.status,
                is_valid_opportunity=r.betslip_odd is not None,
                is_live=r.is_live,
                reversal_direction="up",
                lag_detection_to_click_ms=r.lag_detect_to_switch_ms + r.lag_switch_to_click_ms,
                lag_click_to_betslip_ms=r.lag_click_to_betslip_ms,
                audit_total_duration_ms=r.lag_total_ms,
                audit_version="v2.0-fast",
            )
            async with self.db.async_session() as session:
                session.add(record)
                await session.commit()
        except Exception as e:
            logger.warning(f"Erro salvando: {e}")

    def _print_summary(self):
        """Imprime resumo dos resultados."""
        print("\n" + "=" * 70)
        print("RESULTADOS - TESTE DE VELOCIDADE")
        print("=" * 70)

        ok = [r for r in self.results if r.status == "OK"]
        failed = [r for r in self.results if r.status != "OK"]

        print(f"\n  Total auditorias: {len(self.results)}")
        print(f"  Com betslip (OK): {len(ok)}")
        print(f"  Falhas: {len(failed)}")
        print(f"  WS msgs processadas: {self._ws_message_count}")
        print(f"  Eventos processados: {self.events_processed}")
        print(f"  H3B detectados: {self.h3b_detected}")

        if ok:
            lags_total = [r.lag_total_ms for r in ok]
            lags_switch = [r.lag_detect_to_switch_ms for r in ok]
            lags_click = [r.lag_switch_to_click_ms for r in ok]
            lags_betslip = [r.lag_click_to_betslip_ms for r in ok]
            diffs = [r.difference_pct for r in ok if r.difference_pct is not None]

            print(f"\n  --- TIMING (apenas OK, N={len(ok)}) ---")
            print(f"  LAG TOTAL:     min={min(lags_total)}ms  med={sorted(lags_total)[len(lags_total)//2]}ms  max={max(lags_total)}ms  avg={sum(lags_total)/len(lags_total):.0f}ms")
            print(f"  Detect→Switch: min={min(lags_switch)}ms  med={sorted(lags_switch)[len(lags_switch)//2]}ms  max={max(lags_switch)}ms  avg={sum(lags_switch)/len(lags_switch):.0f}ms")
            print(f"  Switch→Click:  min={min(lags_click)}ms  med={sorted(lags_click)[len(lags_click)//2]}ms  max={max(lags_click)}ms  avg={sum(lags_click)/len(lags_click):.0f}ms")
            print(f"  Click→Betslip: min={min(lags_betslip)}ms  med={sorted(lags_betslip)[len(lags_betslip)//2]}ms  max={max(lags_betslip)}ms  avg={sum(lags_betslip)/len(lags_betslip):.0f}ms")

            if diffs:
                print(f"\n  --- DIFERENÇA WS vs BETSLIP (N={len(diffs)}) ---")
                print(f"  Média:  {sum(diffs)/len(diffs):+.3f}%")
                print(f"  Mediana: {sorted(diffs)[len(diffs)//2]:+.3f}%")
                print(f"  Min/Max: {min(diffs):+.3f}% / {max(diffs):+.3f}%")

        if failed:
            by_status = {}
            for r in failed:
                by_status[r.status] = by_status.get(r.status, 0) + 1
            print(f"\n  --- FALHAS ---")
            for s, n in sorted(by_status.items(), key=lambda x: -x[1]):
                print(f"  {s}: {n}")

        print("\n" + "=" * 70)

    async def run(self):
        """Executa o teste completo."""
        ok = await self.start()
        if not ok:
            return

        audit_queue = asyncio.Queue()
        audited = set()

        # Inicia tasks em paralelo
        tasks = [
            asyncio.create_task(self._monitor_loop(audit_queue, audited)),
            asyncio.create_task(self._executor_loop(audit_queue)),
            asyncio.create_task(self._maintenance_loop()),
        ]

        try:
            # Aguarda até completar auditorias
            while len(self.results) < self.num_audits:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Interrompido")
        finally:
            for t in tasks:
                t.cancel()

        self._print_summary()

        # Cleanup
        for tab in self.game_tabs.values():
            try:
                await tab.page.close()
            except:
                pass
        if self.scraper:
            await self.scraper.close()
        if self.db:
            await self.db.close()


async def main():
    parser = argparse.ArgumentParser(description="Teste de velocidade H3B")
    parser.add_argument("--league", default="England Premier League",
                        help="Liga para monitorar")
    parser.add_argument("--num-audits", type=int, default=20,
                        help="Número de auditorias (default: 20)")
    parser.add_argument("--max-tabs", type=int, default=8,
                        help="Máximo de abas de jogos (default: 8)")
    parser.add_argument("--no-db", action="store_true",
                        help="Não salvar no banco")
    args = parser.parse_args()

    logger.remove()
    # Filtra mensagens H6 do console (muito spam)
    logger.add(sys.stderr,
               format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level:<7}</level> | <level>{message}</level>",
               level="INFO",
               filter=lambda record: "H6:" not in record["message"] and "H1:" not in record["message"])
    logger.add("logs/fast_audit_{time:YYYY-MM-DD}.log", rotation="00:00", retention="30 days", level="DEBUG")

    test = FastAuditTest(
        league=args.league,
        num_audits=args.num_audits,
        max_tabs=args.max_tabs,
        save_to_db=not args.no_db,
    )
    await test.run()


if __name__ == "__main__":
    asyncio.run(main())
