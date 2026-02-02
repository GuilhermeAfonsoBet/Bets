# -*- coding: utf-8 -*-
"""
Cliente para API-Football (api-sports.io).

Busca resultados de jogos finalizados.
Documentacao: https://www.api-football.com/documentation-v3
"""

import asyncio
import httpx
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from loguru import logger


@dataclass
class MatchResult:
    """Resultado de um jogo."""
    fixture_id: int
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    status: str  # FT, HT, NS, etc.
    kickoff_time: datetime
    league_name: str
    league_country: str
    
    # Scores detalhados
    home_score_ht: Optional[int] = None
    away_score_ht: Optional[int] = None


class APIFootballClient:
    """
    Cliente para API-Football.
    
    Uso:
        client = APIFootballClient(api_key="...")
        results = await client.get_results_by_date("2026-02-01")
    """
    
    BASE_URL = "https://v3.football.api-sports.io"
    
    # Mapeamento de ligas BetinAsia -> API-Football league IDs
    LEAGUE_MAPPING = {
        "England Premier League": 39,
        "England Football League Championship": 40,
        "England League 1": 41,
        "England League 2": 42,
        "Spain La Liga": 140,
        "Germany Bundesliga": 78,
        "Italy Serie A": 135,
        "France Ligue 1": 61,
        "Portugal Primeira Liga": 94,
        "Scotland Premier League": 179,
        "UEFA Champions League": 2,
        "UEFA Europa League": 3,
        "FIFA World Cup": 1,
    }
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client: Optional[httpx.AsyncClient] = None
        self._requests_today = 0
        self._last_request_date: Optional[str] = None
        
    async def start(self):
        """Inicia o cliente HTTP."""
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "x-apisports-key": self.api_key,
            },
            timeout=30.0
        )
        logger.info("API-Football client iniciado")
        
    async def close(self):
        """Fecha o cliente HTTP."""
        if self._client:
            await self._client.aclose()
        logger.info("API-Football client fechado")
        
    async def _request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict:
        """Faz requisicao GET para a API."""
        if not self._client:
            await self.start()
            
        # Controle de requests diarios
        today = datetime.now().strftime("%Y-%m-%d")
        if self._last_request_date != today:
            self._requests_today = 0
            self._last_request_date = today
            
        self._requests_today += 1
        
        if self._requests_today > 95:  # Margem de seguranca
            logger.warning(f"Proximo do limite diario: {self._requests_today}/100 requests")
            
        try:
            response = await self._client.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Log de uso da API
            if "errors" in data and data["errors"]:
                logger.error(f"API Error: {data['errors']}")
                return None
                
            return data
            
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None
            
    async def get_status(self) -> Dict:
        """Verifica status da conta e requests restantes."""
        data = await self._request("/status")
        if data:
            account = data.get("response", {}).get("account", {})
            requests = data.get("response", {}).get("requests", {})
            return {
                "plan": account.get("plan"),
                "requests_today": requests.get("current"),
                "requests_limit": requests.get("limit_day"),
            }
        return None
        
    async def get_results_by_date(
        self, 
        date: str,  # formato: "2026-02-01"
        league_id: int = None
    ) -> List[MatchResult]:
        """
        Busca resultados de jogos de uma data.
        
        Args:
            date: Data no formato "YYYY-MM-DD"
            league_id: ID da liga (opcional)
            
        Returns:
            Lista de MatchResult
        """
        params = {"date": date}
        if league_id:
            params["league"] = league_id
            
        data = await self._request("/fixtures", params)
        
        if not data:
            return []
            
        results = []
        for fixture in data.get("response", []):
            try:
                fixture_data = fixture.get("fixture", {})
                teams = fixture.get("teams", {})
                goals = fixture.get("goals", {})
                score = fixture.get("score", {})
                league = fixture.get("league", {})
                
                # Apenas jogos finalizados
                status = fixture_data.get("status", {}).get("short", "")
                if status not in ["FT", "AET", "PEN"]:  # Full Time, After Extra Time, Penalties
                    continue
                    
                result = MatchResult(
                    fixture_id=fixture_data.get("id"),
                    home_team=teams.get("home", {}).get("name", ""),
                    away_team=teams.get("away", {}).get("name", ""),
                    home_score=goals.get("home", 0) or 0,
                    away_score=goals.get("away", 0) or 0,
                    status=status,
                    kickoff_time=datetime.fromisoformat(
                        fixture_data.get("date", "").replace("Z", "+00:00")
                    ),
                    league_name=league.get("name", ""),
                    league_country=league.get("country", ""),
                    home_score_ht=score.get("halftime", {}).get("home"),
                    away_score_ht=score.get("halftime", {}).get("away"),
                )
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Erro ao parsear fixture: {e}")
                continue
                
        logger.info(f"Encontrados {len(results)} resultados para {date}")
        return results
        
    async def get_results_by_league(
        self,
        league_id: int,
        season: int = 2025,  # Temporada 2025/2026
        last_n: int = 20
    ) -> List[MatchResult]:
        """
        Busca ultimos resultados de uma liga.
        
        Args:
            league_id: ID da liga
            season: Temporada
            last_n: Numero de jogos
            
        Returns:
            Lista de MatchResult
        """
        params = {
            "league": league_id,
            "season": season,
            "last": last_n,
            "status": "FT"  # Apenas finalizados
        }
        
        data = await self._request("/fixtures", params)
        
        if not data:
            return []
            
        results = []
        for fixture in data.get("response", []):
            try:
                fixture_data = fixture.get("fixture", {})
                teams = fixture.get("teams", {})
                goals = fixture.get("goals", {})
                league = fixture.get("league", {})
                
                result = MatchResult(
                    fixture_id=fixture_data.get("id"),
                    home_team=teams.get("home", {}).get("name", ""),
                    away_team=teams.get("away", {}).get("name", ""),
                    home_score=goals.get("home", 0) or 0,
                    away_score=goals.get("away", 0) or 0,
                    status="FT",
                    kickoff_time=datetime.fromisoformat(
                        fixture_data.get("date", "").replace("Z", "+00:00")
                    ),
                    league_name=league.get("name", ""),
                    league_country=league.get("country", ""),
                )
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Erro ao parsear fixture: {e}")
                continue
                
        return results
        
    async def search_fixture(
        self,
        home_team: str,
        away_team: str,
        date: str
    ) -> Optional[MatchResult]:
        """
        Busca um jogo especifico por times e data.
        
        Usa busca fuzzy para encontrar o jogo mesmo com nomes diferentes.
        """
        results = await self.get_results_by_date(date)
        
        # Normaliza nomes para comparacao
        def normalize(s):
            return s.lower().replace("fc", "").replace("cf", "").strip()
            
        home_norm = normalize(home_team)
        away_norm = normalize(away_team)
        
        for result in results:
            result_home = normalize(result.home_team)
            result_away = normalize(result.away_team)
            
            # Match exato ou parcial
            if (home_norm in result_home or result_home in home_norm) and \
               (away_norm in result_away or result_away in away_norm):
                return result
                
        return None


async def test_api():
    """Testa a API."""
    import os
    
    api_key = os.environ.get("API_FOOTBALL_KEY", "2707432f357b84409fd3212f9c1a84a5")
    
    client = APIFootballClient(api_key)
    await client.start()
    
    try:
        # Verifica status
        print("=" * 60)
        print("STATUS DA CONTA")
        print("=" * 60)
        status = await client.get_status()
        if status:
            print(f"Plano: {status['plan']}")
            print(f"Requests hoje: {status['requests_today']}/{status['requests_limit']}")
        
        # Busca resultados de ontem
        print()
        print("=" * 60)
        print("RESULTADOS DE ONTEM")
        print("=" * 60)
        
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        results = await client.get_results_by_date(yesterday)
        
        for r in results[:10]:
            print(f"{r.home_team} {r.home_score} - {r.away_score} {r.away_team}")
            print(f"  Liga: {r.league_name} ({r.league_country})")
            
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(test_api())
