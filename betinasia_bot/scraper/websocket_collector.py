# -*- coding: utf-8 -*-
"""
Coletor de odds via WebSocket.

Coleta best odds agregadas sem rate limit.
Usa mensagens offers_hcap e offers_event do WebSocket.
"""

import asyncio
import json
import re
from typing import Dict, List, Optional, Any
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
class MatchOdds:
    """Odds coletadas de um jogo."""
    event_id: str
    sport: str
    home_team: str = ""
    away_team: str = ""
    kickoff_time: Optional[datetime] = None
    league: str = ""
    
    # Asian Handicap - todas as linhas
    ah_lines: Dict[float, AHOdds] = field(default_factory=dict)
    
    # Outros mercados
    over_under: Dict[float, Dict[str, float]] = field(default_factory=dict)  # {10: {"over": 1.85, "under": 2.0}}
    match_odds: Dict[str, float] = field(default_factory=dict)  # {"h": 2.1, "d": 3.5, "a": 3.2}
    
    collected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class WebSocketCollector:
    """
    Coletor de odds via WebSocket.
    
    Navega para páginas de jogos e captura mensagens offers_hcap/offers_event
    que contêm as best odds agregadas.
    """
    
    def __init__(self):
        self.scraper: Optional[BetinAsiaScraper] = None
        self._ws_messages: List[str] = []
        self._current_event_id: Optional[str] = None
        
    async def start(self):
        """Inicia o coletor."""
        self.scraper = BetinAsiaScraper()
        await self.scraper.start()
        await self.scraper.login()
        
        # Configura listener de WebSocket
        self.scraper._page.on('websocket', self._on_websocket)
        
        logger.info("WebSocket Collector iniciado")
        
    async def close(self):
        """Fecha o coletor."""
        if self.scraper:
            await self.scraper.close()
        logger.info("WebSocket Collector fechado")
        
    def _on_websocket(self, ws):
        """Callback quando WebSocket conecta."""
        ws.on('framereceived', lambda data: self._ws_messages.append(str(data)))
        
    async def collect_league(self, league_name: str) -> List[MatchOdds]:
        """
        Coleta odds de todos os jogos de uma liga em UMA ÚNICA navegação.
        
        O WebSocket já envia todas as odds da liga automaticamente!
        
        Args:
            league_name: Nome da liga (ex: "England Premier League")
            
        Returns:
            Lista de MatchOdds com dados coletados
        """
        logger.info(f"Coletando liga: {league_name}")
        
        # Obtém código da liga
        league_code = self.scraper.LEAGUE_CODES.get(league_name)
        if not league_code:
            logger.warning(f"Liga não mapeada: {league_name}")
            return []
        
        # Limpa mensagens anteriores
        self._ws_messages.clear()
            
        # Navega para a página da liga (UMA ÚNICA VEZ)
        league_url = f"{self.scraper.FOOTBALL_URL}/{league_code}"
        await self.scraper._page.goto(league_url)
        await self.scraper._page.wait_for_load_state("networkidle")
        await self.scraper._page.wait_for_timeout(4000)  # Aguarda WebSocket enviar dados
        
        # Parseia TODAS as mensagens de uma vez
        results = self._parse_all_ws_messages()
        
        logger.info(f"Coletados {len(results)} jogos com odds em UMA navegação")
        return results
    
    def _parse_all_ws_messages(self) -> List[MatchOdds]:
        """
        Parseia todas as mensagens WebSocket e extrai odds de múltiplos jogos.
        """
        # Dicionário para acumular dados por event_id
        matches: Dict[str, MatchOdds] = {}
        
        for msg in self._ws_messages:
            try:
                data = json.loads(msg)
                
                if not isinstance(data, list) or len(data) == 0:
                    continue
                    
                for item in data:
                    if not isinstance(item, list) or len(item) < 2:
                        continue
                        
                    msg_type = item[0]
                    msg_meta = item[1]
                    msg_data = item[2] if len(item) > 2 else {}
                    
                    # Processa eventos (info dos times)
                    if msg_type == 'event' and isinstance(msg_meta, list) and len(msg_meta) >= 2:
                        sport_type = msg_meta[0]
                        event_id = msg_meta[1]
                        
                        # Apenas futebol principal
                        if sport_type == 'fb' and 'home' in msg_data:
                            if event_id not in matches:
                                matches[event_id] = MatchOdds(event_id=event_id, sport="fb")
                            self._parse_event_info(msg_data, matches[event_id])
                    
                    # Processa offers (odds)
                    if msg_type in ['offers_hcap', 'offers_event']:
                        if isinstance(msg_meta, list) and len(msg_meta) >= 3:
                            sport_type = msg_meta[1]
                            event_id = msg_meta[2]
                            
                            # Apenas futebol principal
                            if sport_type == 'fb':
                                if event_id not in matches:
                                    matches[event_id] = MatchOdds(event_id=event_id, sport="fb")
                                self._parse_offers(msg_data, matches[event_id])
                            
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                continue
        
        # Retorna apenas jogos com odds
        return [m for m in matches.values() if m.ah_lines]
        
    async def _find_game_urls(self, league_name: str) -> List[tuple]:
        """Encontra URLs de jogos na página da liga."""
        game_urls = []
        url_patterns = self.scraper.LEAGUE_URL_PATTERNS.get(league_name, [])
        
        links = await self.scraper._page.query_selector_all("a")
        
        for link in links:
            href = await link.get_attribute("href")
            if href and "/sportsbook/football/" in href and "," in href:
                # Filtra por padrões da liga
                if url_patterns:
                    if not any(p in href for p in url_patterns):
                        continue
                        
                # Extrai event_id
                match = re.search(r'/(\d{4}-\d{2}-\d{2},\d+,\d+)', href)
                if match:
                    event_id = match.group(1)
                    full_url = f"{self.scraper.BASE_URL}{href.split('?')[0]}"
                    
                    if (event_id, full_url) not in game_urls:
                        game_urls.append((event_id, full_url))
                        
        return game_urls
        
    async def collect_match(self, event_id: str, url: str) -> Optional[MatchOdds]:
        """
        Coleta odds de um jogo específico.
        
        Args:
            event_id: ID do evento (ex: "2026-02-01,2,12")
            url: URL completa do jogo
            
        Returns:
            MatchOdds com dados coletados
        """
        self._current_event_id = event_id
        self._ws_messages.clear()
        
        # Navega para o jogo
        await self.scraper._page.goto(url)
        await self.scraper._page.wait_for_load_state("networkidle")
        await self.scraper._page.wait_for_timeout(2000)
        
        # Parseia mensagens WebSocket
        match_odds = self._parse_ws_messages(event_id)
            
        return match_odds
        
    def _parse_ws_messages(self, event_id: str) -> Optional[MatchOdds]:
        """Parseia mensagens WebSocket e extrai odds."""
        match_odds = MatchOdds(event_id=event_id, sport="fb")
        
        for msg in self._ws_messages:
            try:
                data = json.loads(msg)
                
                if not isinstance(data, list) or len(data) == 0:
                    continue
                    
                # Processa cada item da mensagem
                for item in data:
                    if not isinstance(item, list) or len(item) < 2:
                        continue
                        
                    msg_type = item[0]
                    msg_meta = item[1] if len(item) > 1 else []
                    msg_data = item[2] if len(item) > 2 else {}
                    
                    # Extrai informações do evento (times, liga, horário)
                    if msg_type == 'event' and isinstance(msg_meta, list) and len(msg_meta) >= 2:
                        sport_type = msg_meta[0]  # "fb", "fb_ht", etc.
                        msg_event_id = msg_meta[1]
                        
                        # Verifica se é o evento correto (futebol principal)
                        if msg_event_id == event_id and sport_type == "fb":
                            self._parse_event_info(msg_data, match_odds)
                    
                    # Verifica se é do evento correto para offers
                    if isinstance(msg_meta, list) and len(msg_meta) >= 3:
                        msg_event_id = msg_meta[2]
                        if msg_event_id != event_id:
                            continue
                            
                    # Processa offers_hcap e offers_event
                    if msg_type in ['offers_hcap', 'offers_event']:
                        self._parse_offers(msg_data, match_odds)
                        
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                continue
                
        return match_odds if match_odds.ah_lines else None
    
    def _parse_event_info(self, data: dict, match_odds: MatchOdds):
        """Extrai informações do evento (times, liga, horário)."""
        try:
            # Nome dos times
            if 'home' in data:
                match_odds.home_team = data['home']
            if 'away' in data:
                match_odds.away_team = data['away']
                
            # Liga
            if 'competition_name' in data:
                match_odds.league = data['competition_name']
                
            # Horário de início
            if 'start_ts' in data:
                try:
                    # Formato: "2026-02-01T20:00:00Z"
                    match_odds.kickoff_time = datetime.fromisoformat(
                        data['start_ts'].replace('Z', '+00:00')
                    )
                except:
                    pass
                    
        except Exception as e:
            logger.debug(f"Erro ao extrair info do evento: {e}")
        
    def _parse_offers(self, data: dict, match_odds: MatchOdds):
        """Parseia dados de offers e adiciona ao MatchOdds."""
        
        # Asian Handicap
        if 'ah' in data:
            ah_data = data['ah']
            
            # Formato: [line, [["a", odds], ["h", odds]]]
            if isinstance(ah_data, list) and len(ah_data) >= 2:
                # Pode ser uma única linha ou múltiplas
                if isinstance(ah_data[0], (int, float)):
                    # Única linha
                    self._parse_ah_line(ah_data, match_odds)
                elif isinstance(ah_data[0], list):
                    # Múltiplas linhas
                    for line_data in ah_data:
                        self._parse_ah_line(line_data, match_odds)
                        
        # Over/Under
        if 'ahou' in data:
            ou_data = data['ahou']
            if isinstance(ou_data, list) and len(ou_data) >= 2:
                if isinstance(ou_data[0], (int, float)):
                    self._parse_ou_line(ou_data, match_odds)
                elif isinstance(ou_data[0], list):
                    for line_data in ou_data:
                        self._parse_ou_line(line_data, match_odds)
                        
        # Match Odds (1X2)
        if 'wdw' in data:
            wdw_data = data['wdw']
            if isinstance(wdw_data, list) and len(wdw_data) >= 2:
                odds_list = wdw_data[1]
                if isinstance(odds_list, list):
                    for item in odds_list:
                        if isinstance(item, list) and len(item) >= 2:
                            match_odds.match_odds[item[0]] = item[1]
                            
    def _parse_ah_line(self, line_data: list, match_odds: MatchOdds):
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
            match_odds.ah_lines[line] = AHOdds(
                line=line,
                home_odds=home_odds,
                away_odds=away_odds
            )
            
    def _parse_ou_line(self, line_data: list, match_odds: MatchOdds):
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
            match_odds.over_under[line] = {
                "over": over_odds,
                "under": under_odds
            }


async def test_collector():
    """Testa o coletor."""
    collector = WebSocketCollector()
    
    try:
        await collector.start()
        
        # Testa com um jogo específico
        print("=== TESTE: Coleta de um jogo ===")
        match = await collector.collect_match(
            "2026-02-01,2,12",
            "https://black.betinasia.com/sportsbook/football/XE/1/2026-02-01,2,12"
        )
        
        if match:
            print(f"\nJogo: {match.home_team} vs {match.away_team}")
            print(f"Liga: {match.league}")
            print(f"Horário: {match.kickoff_time}")
            print(f"Event ID: {match.event_id}")
            print(f"\nAsian Handicap ({len(match.ah_lines)} linhas):")
            for line, odds in sorted(match.ah_lines.items()):
                print(f"  {line:+.2f}: H={odds.home_odds:.3f} A={odds.away_odds:.3f}")
                
            if match.match_odds:
                print(f"\n1X2: {match.match_odds}")
                
            if match.over_under:
                print(f"\nOver/Under ({len(match.over_under)} linhas):")
                for line, odds in sorted(match.over_under.items())[:10]:  # Limita a 10
                    print(f"  {line}: O={odds['over']:.3f} U={odds['under']:.3f}")
        else:
            print("Nenhuma odds coletada")
            
    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(test_collector())
