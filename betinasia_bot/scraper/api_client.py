# -*- coding: utf-8 -*-
"""
Cliente de API para BetinAsia.

Chama as APIs diretamente ao invés de fazer scraping via cliques.
Muito mais rápido e eficiente.
"""

import asyncio
import httpx
from typing import Dict, List, Optional, Any
from loguru import logger
from dataclasses import dataclass


@dataclass
class BookmakerOdds:
    """Odds de um bookmaker específico."""
    bookmaker: str
    odds: float
    stake_limit: float = 0.0


@dataclass
class BetslipData:
    """Dados retornados pela API de betslip."""
    event_id: str
    bet_type: str
    bookmakers: List[BookmakerOdds]
    best_odds: float
    best_bookmaker: str
    raw_data: Dict[str, Any] = None


class BetinAsiaAPIClient:
    """
    Cliente para APIs do BetinAsia.
    
    Usa as mesmas APIs que o site usa internamente.
    Requer cookies de sessão válidos (obtidos após login).
    """
    
    BASE_URL = "https://black.betinasia.com"
    
    # Delay entre requisições para evitar rate limit (em segundos)
    REQUEST_DELAY = 5.0
    
    def __init__(self, session_cookies: Dict[str, str] = None):
        """
        Inicializa o cliente.
        
        Args:
            session_cookies: Cookies de sessão do login
        """
        self.cookies = session_cookies or {}
        self._client: Optional[httpx.AsyncClient] = None
        self._last_request_time = 0
        
    async def start(self):
        """Inicia o cliente HTTP."""
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            cookies=self.cookies,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=30.0
        )
        logger.info("API Client iniciado")
        
    async def close(self):
        """Fecha o cliente HTTP."""
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("API Client fechado")
            
    async def _rate_limit_delay(self):
        """Aplica delay entre requisições para evitar rate limit."""
        import time
        elapsed = time.time() - self._last_request_time
        if elapsed < self.REQUEST_DELAY:
            await asyncio.sleep(self.REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()
        
    async def get_betslip(
        self, 
        event_id: str, 
        bet_type: str = "for,d",
        sport: str = "fb"
    ) -> Optional[BetslipData]:
        """
        Obtém dados do betslip (inclui odds de todos os bookmakers).
        
        Args:
            event_id: ID do evento (ex: "2026-02-01,22,94")
            bet_type: Tipo de aposta (ex: "for,d" para AH)
            sport: Esporte (fb = football)
            
        Returns:
            BetslipData com odds de todos os bookmakers, ou None se falhar
        """
        await self._rate_limit_delay()
        
        try:
            response = await self._client.post(
                "/v1/betslips/",
                json={
                    "sport": sport,
                    "event_id": event_id,
                    "bet_type": bet_type,
                    "betslip_type": "normal",
                    "equivalent_bets": True
                }
            )
            
            if response.status_code == 429:
                # Rate limited
                data = response.json()
                retry_after = data.get("data", {}).get("retry_after", 60)
                logger.warning(f"Rate limited! Retry after {retry_after}s")
                return None
                
            if response.status_code != 200:
                logger.warning(f"Erro na API: {response.status_code}")
                return None
                
            data = response.json()
            return self._parse_betslip_response(event_id, bet_type, data)
            
        except Exception as e:
            logger.error(f"Erro ao chamar API betslip: {e}")
            return None
            
    def _parse_betslip_response(
        self, 
        event_id: str, 
        bet_type: str, 
        data: Dict
    ) -> Optional[BetslipData]:
        """
        Parseia a resposta da API de betslip.
        
        A estrutura exata será descoberta quando testarmos.
        Por enquanto, retorna os dados brutos.
        """
        try:
            # TODO: Ajustar parsing baseado na estrutura real da resposta
            # Por enquanto, retorna dados brutos para análise
            
            bookmakers = []
            best_odds = 0.0
            best_bookmaker = "unknown"
            
            # Tenta extrair bookmakers da resposta
            # A estrutura será ajustada após vermos uma resposta real
            if "data" in data:
                raw_data = data["data"]
                
                # Procura por lista de bookmakers/odds
                if isinstance(raw_data, dict):
                    for key, value in raw_data.items():
                        if "book" in key.lower() or "odds" in key.lower():
                            logger.debug(f"Encontrado campo: {key} = {value}")
                            
            return BetslipData(
                event_id=event_id,
                bet_type=bet_type,
                bookmakers=bookmakers,
                best_odds=best_odds,
                best_bookmaker=best_bookmaker,
                raw_data=data
            )
            
        except Exception as e:
            logger.error(f"Erro ao parsear resposta: {e}")
            return None
            
    async def get_event_odds(self, event_id: str) -> Dict[str, BetslipData]:
        """
        Obtém odds de todos os mercados de um evento.
        
        Args:
            event_id: ID do evento
            
        Returns:
            Dict com bet_type -> BetslipData
        """
        # Tipos de aposta para AH
        bet_types = [
            "for,d",      # AH principal
            "for,d,1h",   # AH 1º tempo
            # Adicionar mais conforme descobrirmos
        ]
        
        results = {}
        for bet_type in bet_types:
            data = await self.get_betslip(event_id, bet_type)
            if data:
                results[bet_type] = data
                
        return results


async def test_api_client():
    """Testa o cliente de API."""
    # Este teste precisa de cookies válidos
    # Obter do browser após login
    
    print("Para testar, precisamos dos cookies de sessão.")
    print("Use o scraper para fazer login e extrair os cookies.")
    

if __name__ == "__main__":
    asyncio.run(test_api_client())
