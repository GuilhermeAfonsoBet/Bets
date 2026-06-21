# -*- coding: utf-8 -*-
"""
Fast Collector - Coletor ultra-rápido de odds via WebSocket.

Coleta odds das principais ligas em UMA ÚNICA navegação.
~250 jogos em ~10 segundos.

Ligas cobertas automaticamente:
- England Premier League, Championship, League 1, League 2
- Spain La Liga
- Germany Bundesliga
- Italy Serie A
- France Ligue 1
- Portugal Primeira Liga
- Scotland Premier League
- UEFA Champions League, Europa League
- FIFA World Cup
"""

import asyncio
import json
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
from loguru import logger

from .betinasia import BetinAsiaScraper


@dataclass
class AHOdds:
    """Odds de Asian Handicap para uma linha."""
    line: float  # Ex: -1, -0.5, +0.25
    home_odds: float
    away_odds: float


@dataclass
class OUOdds:
    """Odds de Over/Under para uma linha."""
    line: float  # Ex: 2.5, 3.0
    over_odds: float
    under_odds: float


@dataclass 
class MatchOdds:
    """Odds coletadas de um jogo."""
    event_id: str
    sport: str
    home_team: str = ""
    away_team: str = ""
    kickoff_time: Optional[datetime] = None
    league: str = ""
    country: str = ""
    
    # Asian Handicap - todas as linhas
    ah_lines: Dict[float, AHOdds] = field(default_factory=dict)
    
    # Over/Under
    over_under: Dict[float, OUOdds] = field(default_factory=dict)
    
    # Match Odds (1X2)
    match_odds: Dict[str, float] = field(default_factory=dict)  # {"h": 2.1, "d": 3.5, "a": 3.2}
    
    collected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CollectionResult:
    """Resultado da coleta."""
    matches: List[MatchOdds]
    total_events: int  # Total de eventos descobertos
    total_with_odds: int  # Eventos que têm odds
    leagues_with_odds: Set[str]  # Ligas que receberam odds
    collection_time: float  # Tempo em segundos
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FastCollector:
    """
    Coletor ultra-rápido de odds via WebSocket.
    
    Navega para a página principal de futebol e captura todas as odds
    enviadas automaticamente pelo WebSocket para as principais ligas.
    
    Uso:
        collector = FastCollector()
        await collector.start()
        result = await collector.collect_all()
        await collector.close()
    """
    
    # URL da página principal de futebol
    FOOTBALL_URL = "https://black.betinasia.com/sportsbook/football"
    
    # Tempo de espera para WebSocket carregar (em ms)
    WAIT_TIME_MS = 6000
    
    def __init__(self):
        self.scraper: Optional[BetinAsiaScraper] = None
        self._ws_messages: List[str] = []
        
    async def start(self):
        """Inicia o coletor."""
        self.scraper = BetinAsiaScraper()
        await self.scraper.start()
        logged = await self.scraper.login()
        if not logged:
            raise RuntimeError("Falha no login inicial do FastCollector")
        
        # Configura listener de WebSocket
        self.scraper._page.on('websocket', self._on_websocket)
        
        logger.info("Fast Collector iniciado")
        
    async def close(self):
        """Fecha o coletor."""
        if self.scraper:
            await self.scraper.close()
        logger.info("Fast Collector fechado")
        
    def _on_websocket(self, ws):
        """Callback quando WebSocket conecta."""
        # Playwright pode entregar `WebSocketFrame`, dict ou string; queremos sempre
        # o payload textual (JSON) para o parser.
        def _on_frame(frame):
            try:
                payload = None
                # WebSocketFrame (Playwright) costuma ter `.payload`
                if hasattr(frame, "payload"):
                    payload = frame.payload
                elif isinstance(frame, dict) and "payload" in frame:
                    payload = frame.get("payload")
                else:
                    payload = frame
                if payload is None:
                    return
                self._ws_messages.append(str(payload))
            except Exception:
                return

        ws.on('framereceived', _on_frame)
        
    async def collect_all(self) -> CollectionResult:
        """
        Coleta odds de TODAS as principais ligas em uma única navegação.
        
        Returns:
            CollectionResult com todos os dados coletados
        """
        import time
        start_time = time.time()
        
        logger.info("Iniciando coleta rápida de todas as ligas...")
        
        # Limpa mensagens anteriores
        self._ws_messages.clear()
        
        # Navega para página principal de futebol
        await self.scraper._page.goto(
            self.FOOTBALL_URL,
            timeout=self.scraper.DEFAULT_NAV_TIMEOUT_MS,
            wait_until=self.scraper.DEFAULT_GOTO_WAIT_UNTIL,
        )
        await self.scraper._page.wait_for_timeout(self.WAIT_TIME_MS)

        # Sessão expirada pode redirecionar para /login sem levantar exceção.
        if "login" in (self.scraper._page.url or "").lower():
            logger.warning("FastCollector: sessão expirada detectada, refazendo login...")
            relog_ok = await self.scraper.login(force=True)
            if not relog_ok:
                raise RuntimeError("Sessão inválida no collect_all (relogin falhou)")
            await self.scraper._page.goto(
                self.FOOTBALL_URL,
                timeout=self.scraper.DEFAULT_NAV_TIMEOUT_MS,
                wait_until=self.scraper.DEFAULT_GOTO_WAIT_UNTIL,
            )
            await self.scraper._page.wait_for_timeout(self.WAIT_TIME_MS)
        
        # Parseia todas as mensagens
        all_events, events_with_odds = self._parse_all_messages()

        # Em algumas condições (proxy / WS lento), os offers chegam após a primeira janela.
        if len(all_events) > 0 and len(events_with_odds) == 0:
            await self.scraper._page.wait_for_timeout(6000)
            all_events, events_with_odds = self._parse_all_messages()
        
        # Filtra apenas eventos com odds
        matches = [m for m in all_events.values() if m.ah_lines]
        
        # Identifica ligas com odds
        leagues_with_odds = {m.league for m in matches if m.league}
        
        elapsed = time.time() - start_time
        
        result = CollectionResult(
            matches=matches,
            total_events=len(all_events),
            total_with_odds=len(matches),
            leagues_with_odds=leagues_with_odds,
            collection_time=elapsed
        )
        
        logger.info(
            f"Coleta concluída: {result.total_with_odds} jogos com odds "
            f"de {len(result.leagues_with_odds)} ligas em {elapsed:.1f}s"
        )
        
        return result
    
    async def collect_league(self, league_code: str) -> CollectionResult:
        """
        Coleta odds de uma liga específica.
        
        Args:
            league_code: Código da liga (ex: "XE/1" para Premier League)
            
        Returns:
            CollectionResult com dados da liga
        """
        import time
        start_time = time.time()
        
        # Limpa mensagens anteriores
        self._ws_messages.clear()
        
        # Navega para página da liga
        league_url = f"{self.FOOTBALL_URL}/{league_code}"
        await self.scraper._page.goto(
            league_url,
            timeout=self.scraper.DEFAULT_NAV_TIMEOUT_MS,
            wait_until=self.scraper.DEFAULT_GOTO_WAIT_UNTIL,
        )
        await self.scraper._page.wait_for_timeout(self.WAIT_TIME_MS)

        if "login" in (self.scraper._page.url or "").lower():
            logger.warning("FastCollector: sessão expirada em collect_league, refazendo login...")
            relog_ok = await self.scraper.login(force=True)
            if not relog_ok:
                raise RuntimeError("Sessão inválida no collect_league (relogin falhou)")
            await self.scraper._page.goto(
                league_url,
                timeout=self.scraper.DEFAULT_NAV_TIMEOUT_MS,
                wait_until=self.scraper.DEFAULT_GOTO_WAIT_UNTIL,
            )
            await self.scraper._page.wait_for_timeout(self.WAIT_TIME_MS)
        
        # Parseia mensagens
        all_events, _ = self._parse_all_messages()
        matches = [m for m in all_events.values() if m.ah_lines]
        leagues_with_odds = {m.league for m in matches if m.league}
        
        elapsed = time.time() - start_time
        
        return CollectionResult(
            matches=matches,
            total_events=len(all_events),
            total_with_odds=len(matches),
            leagues_with_odds=leagues_with_odds,
            collection_time=elapsed
        )
        
    def _parse_all_messages(self) -> tuple[Dict[str, MatchOdds], Set[str]]:
        """
        Parseia todas as mensagens WebSocket.
        
        Returns:
            (dict de eventos, set de event_ids com odds)
        """
        events: Dict[str, MatchOdds] = {}
        events_with_odds: Set[str] = set()
        
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

                    def _extract_sport_event_id(meta) -> tuple[Optional[str], Optional[str]]:
                        if not isinstance(meta, list) or not meta:
                            return None, None
                        # Padrões observados (variáveis entre releases):
                        # - ['fb', '<event_id>', ...]
                        # - ['offers', 'fb', '<event_id>', ...]
                        # - [<sport>, <event_id>]
                        try:
                            if len(meta) >= 2 and meta[0] == 'fb':
                                return 'fb', meta[1]
                            if len(meta) >= 3 and meta[1] == 'fb':
                                return 'fb', meta[2]
                            if len(meta) >= 2 and meta[0] in ('fb', 'tn', 'bk', 'bb'):
                                return meta[0], meta[1]
                        except Exception:
                            return None, None
                        return None, None
                    
                    # Processa eventos (info dos jogos)
                    if msg_type == 'event' and isinstance(msg_meta, list) and len(msg_meta) >= 2:
                        sport_type = msg_meta[0]
                        event_id = msg_meta[1]
                        
                        # Apenas futebol principal (não HT, HTFT, etc.)
                        if sport_type == 'fb' and 'home' in msg_data:
                            if event_id not in events:
                                events[event_id] = MatchOdds(event_id=event_id, sport="fb")
                            self._parse_event_info(msg_data, events[event_id])
                    
                    # Processa offers (odds)
                    if isinstance(msg_type, str) and msg_type.startswith('offers'):
                        sport_type, event_id = _extract_sport_event_id(msg_meta)
                        if sport_type == 'fb' and event_id:
                            if event_id not in events:
                                events[event_id] = MatchOdds(event_id=event_id, sport="fb")
                            self._parse_offers(msg_data, events[event_id])
                            # Só marca "com odds" se algum mercado foi preenchido
                            if events[event_id].ah_lines or events[event_id].over_under or events[event_id].match_odds:
                                events_with_odds.add(event_id)
                            
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                continue
        
        return events, events_with_odds
    
    def _parse_event_info(self, data: dict, match: MatchOdds):
        """Extrai informações do evento."""
        if 'home' in data:
            match.home_team = data['home']
        if 'away' in data:
            match.away_team = data['away']
        if 'competition_name' in data:
            match.league = data['competition_name']
        if 'country' in data:
            match.country = data['country']
        if 'start_ts' in data:
            try:
                match.kickoff_time = datetime.fromisoformat(
                    data['start_ts'].replace('Z', '+00:00')
                )
            except:
                pass
                
    def _parse_offers(self, data: dict, match: MatchOdds):
        """Parseia dados de offers."""
        
        # Asian Handicap
        if 'ah' in data:
            ah_data = data['ah']
            if isinstance(ah_data, list) and len(ah_data) >= 2:
                if isinstance(ah_data[0], (int, float)):
                    self._parse_ah_line(ah_data, match)
                elif isinstance(ah_data[0], list):
                    for line_data in ah_data:
                        self._parse_ah_line(line_data, match)
                        
        # Over/Under
        if 'ahou' in data:
            ou_data = data['ahou']
            if isinstance(ou_data, list) and len(ou_data) >= 2:
                if isinstance(ou_data[0], (int, float)):
                    self._parse_ou_line(ou_data, match)
                elif isinstance(ou_data[0], list):
                    for line_data in ou_data:
                        self._parse_ou_line(line_data, match)
                        
        # Match Odds (1X2)
        if 'wdw' in data:
            wdw_data = data['wdw']
            if isinstance(wdw_data, list) and len(wdw_data) >= 2:
                odds_list = wdw_data[1]
                if isinstance(odds_list, list):
                    for item in odds_list:
                        if isinstance(item, list) and len(item) >= 2:
                            match.match_odds[item[0]] = item[1]
                            
    def _parse_ah_line(self, line_data: list, match: MatchOdds):
        """Parseia uma linha de AH."""
        if len(line_data) < 2:
            return
            
        line = float(line_data[0]) if line_data[0] is not None else 0
        odds_list = line_data[1]
        
        home_odds = 0.0
        away_odds = 0.0
        
        if isinstance(odds_list, list):
            for item in odds_list:
                if isinstance(item, list) and len(item) >= 2:
                    if item[0] == 'h':
                        home_odds = float(item[1])
                    elif item[0] == 'a':
                        away_odds = float(item[1])
                        
        if home_odds > 0 and away_odds > 0:
            match.ah_lines[line] = AHOdds(
                line=line,
                home_odds=home_odds,
                away_odds=away_odds
            )
            
    def _parse_ou_line(self, line_data: list, match: MatchOdds):
        """Parseia uma linha de Over/Under."""
        if len(line_data) < 2:
            return
            
        line = float(line_data[0]) if line_data[0] is not None else 0
        odds_list = line_data[1]
        
        over_odds = 0.0
        under_odds = 0.0
        
        if isinstance(odds_list, list):
            for item in odds_list:
                if isinstance(item, list) and len(item) >= 2:
                    if item[0] == 'over':
                        over_odds = float(item[1])
                    elif item[0] == 'under':
                        under_odds = float(item[1])
                        
        if over_odds > 0 and under_odds > 0:
            match.over_under[line] = OUOdds(
                line=line,
                over_odds=over_odds,
                under_odds=under_odds
            )


async def test_fast_collector():
    """Testa o coletor rápido."""
    collector = FastCollector()
    
    try:
        await collector.start()
        
        print("=" * 60)
        print("FAST COLLECTOR - Teste de coleta rápida")
        print("=" * 60)
        
        result = await collector.collect_all()
        
        print(f"\n{'='*60}")
        print("RESULTADO")
        print(f"{'='*60}")
        print(f"Tempo de coleta: {result.collection_time:.1f}s")
        print(f"Total eventos descobertos: {result.total_events}")
        print(f"Eventos com odds: {result.total_with_odds}")
        print(f"Ligas com odds: {len(result.leagues_with_odds)}")
        
        print(f"\n{'='*60}")
        print("LIGAS COLETADAS")
        print(f"{'='*60}")
        
        # Agrupa por liga
        by_league: Dict[str, List[MatchOdds]] = {}
        for match in result.matches:
            league = match.league or "Unknown"
            if league not in by_league:
                by_league[league] = []
            by_league[league].append(match)
        
        for league in sorted(by_league.keys(), key=lambda x: -len(by_league[x])):
            matches = by_league[league]
            print(f"\n{league}: {len(matches)} jogos")
            
            # Mostra primeiros 3 jogos
            for match in matches[:3]:
                ah_line_0 = match.ah_lines.get(0) or match.ah_lines.get(0.0)
                if ah_line_0:
                    print(f"  {match.home_team} vs {match.away_team}")
                    print(f"    AH 0: H={ah_line_0.home_odds:.3f} A={ah_line_0.away_odds:.3f}")
                    
            if len(matches) > 3:
                print(f"  ... e mais {len(matches)-3} jogos")
                
    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(test_fast_collector())
