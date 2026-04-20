# -*- coding: utf-8 -*-
"""
Auditoria em Tempo Real via API

Quando detecta um evento H6, chama a API /v1/betslips/ para verificar
se as odds coletadas via WebSocket correspondem às odds reais da API.

Muito mais confiável que clicar em elementos DOM.
"""

import asyncio
import json
import sys
import httpx
from datetime import datetime, timezone
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from loguru import logger

sys.path.insert(0, '.')

from scraper.betinasia import BetinAsiaScraper
from hypothesis.detectors import HypothesisDetector


@dataclass
class AuditResult:
    """Resultado de uma auditoria."""
    timestamp: datetime
    match_info: str
    event_id: str
    market_type: str
    line: str
    side: str
    websocket_odd: float
    api_odd: Optional[float]
    difference_pct: Optional[float]
    status: str
    api_raw: Optional[dict] = None


class RealtimeAPIAuditor:
    """Auditor em tempo real usando API de betslip."""
    
    FOOTBALL_URL = "https://black.betinasia.com/sportsbook/football"
    WAIT_TIME_MS = 6000
    API_DELAY = 1.0  # Delay entre chamadas API para evitar rate limit
    
    def __init__(self, num_audits: int = 10):
        self.scraper: Optional[BetinAsiaScraper] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        self._ws_messages: List[str] = []
        self.hypothesis_detector = HypothesisDetector()
        self.num_audits = num_audits
        self.audit_results: List[AuditResult] = []
        self.events_processed = 0
        self.h6_events_detected = 0
        self.cookies: Dict[str, str] = {}
        
    async def start(self):
        """Inicia o auditor."""
        # Inicia scraper e faz login
        self.scraper = BetinAsiaScraper()
        await self.scraper.start()
        await self.scraper.login()
        
        # Extrai cookies para usar na API
        cookies = await self.scraper._context.cookies()
        self.cookies = {c['name']: c['value'] for c in cookies}
        
        print(f"\nCookies extraídos: {len(self.cookies)}")
        for name in list(self.cookies.keys())[:10]:
            print(f"  - {name}: {self.cookies[name][:20]}...")
        
        # Configura cliente HTTP para API
        # Usa os mesmos headers que o browser
        self.http_client = httpx.AsyncClient(
            base_url="https://black.betinasia.com",
            cookies=self.cookies,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Content-Type": "application/json",
                "Origin": "https://black.betinasia.com",
                "Referer": "https://black.betinasia.com/sportsbook/football",
                "X-Requested-With": "XMLHttpRequest",
            },
            timeout=30.0
        )
        
        # Configura listener de WebSocket
        self.scraper._page.on('websocket', self._on_websocket)
        
        logger.info("RealtimeAPIAuditor iniciado")
        
    async def close(self):
        """Fecha o auditor."""
        if self.http_client:
            await self.http_client.aclose()
        if self.scraper:
            await self.scraper.close()
            
    def _on_websocket(self, ws):
        """Callback quando WebSocket conecta."""
        ws.on('framereceived', lambda data: self._ws_messages.append(str(data)))
        
    async def call_betslip_api(self, event_id: str, bet_type: str = "for,d") -> Optional[dict]:
        """
        Chama a API de betslip para obter odds reais.
        
        Usa o próprio browser (Playwright) para fazer a requisição,
        garantindo que a sessão autenticada seja usada.
        
        Args:
            event_id: ID do evento (ex: "2026-02-01,22,94")
            bet_type: Tipo de aposta
            
        Returns:
            Dados da API ou None
        """
        try:
            await asyncio.sleep(self.API_DELAY)  # Rate limit
            
            # Usa o browser para fazer a requisição via JavaScript
            # Isso garante que todos os cookies e headers da sessão sejam usados
            result = await self.scraper._page.evaluate("""
                async (params) => {
                    try {
                        const response = await fetch('/v1/betslips/', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                                'Accept': 'application/json',
                            },
                            body: JSON.stringify({
                                sport: params.sport,
                                event_id: params.event_id,
                                bet_type: params.bet_type,
                                betslip_type: 'normal',
                                equivalent_bets: true
                            })
                        });
                        
                        const data = await response.json();
                        return {
                            status: response.status,
                            data: data
                        };
                    } catch (e) {
                        return {
                            status: 0,
                            error: e.toString()
                        };
                    }
                }
            """, {"sport": "fb", "event_id": event_id, "bet_type": bet_type})
            
            if result.get("error"):
                logger.warning(f"Erro no fetch: {result['error']}")
                return None
                
            status = result.get("status", 0)
            
            if status == 429:
                logger.warning("Rate limited pela API!")
                return None
                
            if status != 200:
                logger.warning(f"Erro na API: {status}")
                return None
                
            return result.get("data")
            
        except Exception as e:
            logger.error(f"Erro ao chamar API: {e}")
            return None
    
    def extract_best_odds_from_api(self, api_data: dict, line: str, side: str) -> Optional[float]:
        """
        Extrai a best odd da resposta da API.
        
        Args:
            api_data: Resposta da API
            line: Linha (ex: "-0.5", "2.5")
            side: Lado (home/away/over/under)
            
        Returns:
            Best odd ou None
        """
        try:
            # A estrutura exata depende da resposta da API
            # Vamos tentar extrair de várias formas
            
            data = api_data.get("data", {})
            
            # Procura por campo de odds
            # Estrutura típica: data.odds ou data.bookmakers
            
            if "odds" in data:
                odds_data = data["odds"]
                # Procura pela linha e lado específicos
                if isinstance(odds_data, dict):
                    for key, value in odds_data.items():
                        if line in key and side in key.lower():
                            if isinstance(value, (int, float)):
                                return float(value)
                            elif isinstance(value, dict) and "best" in value:
                                return float(value["best"])
            
            # Procura em bookmakers
            if "bookmakers" in data:
                bookmakers = data["bookmakers"]
                if isinstance(bookmakers, list):
                    # Pega a melhor odd entre todos os bookmakers
                    best = 0.0
                    for bk in bookmakers:
                        if isinstance(bk, dict):
                            odds = bk.get("odds", bk.get("price", 0))
                            if odds > best:
                                best = odds
                    if best > 0:
                        return best
            
            # Procura recursivamente por campos com "odd" ou "price"
            def find_odds(obj, target_line, target_side):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if isinstance(value, (int, float)) and value > 1.0 and value < 100:
                            # Pode ser uma odd
                            if target_line in str(key) or target_side in str(key).lower():
                                return float(value)
                        result = find_odds(value, target_line, target_side)
                        if result:
                            return result
                elif isinstance(obj, list):
                    for item in obj:
                        result = find_odds(item, target_line, target_side)
                        if result:
                            return result
                return None
            
            return find_odds(data, line, side)
            
        except Exception as e:
            logger.error(f"Erro ao extrair odds: {e}")
            return None
        
    async def run_audit(self):
        """Executa ciclos de auditoria."""
        await self.start()
        
        print("=" * 70)
        print("AUDITORIA EM TEMPO REAL VIA API")
        print("=" * 70)
        print(f"""
Este script compara:
- Odds recebidas via WebSocket (o que o coletor salva)
- Odds da API /v1/betslips/ (o que você veria no betslip)

Vou auditar {self.num_audits} eventos H6 detectados.
""")
        
        try:
            while len(self.audit_results) < self.num_audits:
                # Coleta dados
                self._ws_messages.clear()
                
                await self.scraper._page.goto(self.FOOTBALL_URL)
                await self.scraper._page.wait_for_load_state("networkidle")
                await self.scraper._page.wait_for_timeout(self.WAIT_TIME_MS)
                
                # Parseia mensagens e detecta eventos
                await self._process_and_audit()
                
                # Status
                print(f"\rProcessados: {self.events_processed} mercados | "
                      f"H6 detectados: {self.h6_events_detected} | "
                      f"Auditados: {len(self.audit_results)}/{self.num_audits}", 
                      end="", flush=True)
                
                # Pausa antes do próximo ciclo
                await asyncio.sleep(5)
                
        finally:
            await self.close()
            
        # Mostra resultados
        self._print_results()
        
    async def _process_and_audit(self):
        """Processa mensagens e audita eventos H6 detectados."""
        events = {}  # event_id -> info
        
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
                    
                    # Captura info do evento
                    if msg_type == 'event' and isinstance(msg_meta, list) and len(msg_meta) >= 2:
                        sport_type = msg_meta[0]
                        event_id = msg_meta[1]
                        
                        if sport_type == 'fb' and 'home' in msg_data:
                            events[event_id] = {
                                'home': msg_data.get('home', ''),
                                'away': msg_data.get('away', ''),
                                'league': msg_data.get('competition_name', ''),
                            }
                    
                    # Processa offers e detecta hipóteses
                    if msg_type in ['offers_hcap', 'offers_event']:
                        if isinstance(msg_meta, list) and len(msg_meta) >= 3:
                            sport_type = msg_meta[1]
                            event_id = msg_meta[2]
                            
                            if sport_type == 'fb':
                                await self._process_offers_and_audit(event_id, msg_data, events)
                                
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                continue
                
    async def _process_offers_and_audit(self, event_id: str, data: dict, events: dict):
        """Processa offers e audita se detectar H6."""
        
        event_info = events.get(event_id, {})
        match_info = f"{event_info.get('home', '?')} vs {event_info.get('away', '?')}"
        
        # Asian Handicap
        if 'ah' in data:
            ah_data = data['ah']
            lines = []
            
            if isinstance(ah_data, list) and len(ah_data) >= 2:
                if isinstance(ah_data[0], (int, float)):
                    lines = [ah_data]
                elif isinstance(ah_data[0], list):
                    lines = ah_data
                    
            for line_data in lines:
                if len(line_data) < 2:
                    continue
                    
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
                    self.events_processed += 1
                    
                    # Detecta hipóteses
                    detector_events = self.hypothesis_detector.process_market_update(
                        match_id=hash(event_id) % 1000000,
                        market_type="AH",
                        line=str(line),
                        home_odd=home_odds,
                        away_odd=away_odds,
                    )
                    
                    # Se detectou H6, audita via API!
                    h6_events = detector_events.get("h6_events", [])
                    if h6_events and len(self.audit_results) < self.num_audits:
                        self.h6_events_detected += len(h6_events)
                        
                        for h6 in h6_events:
                            result = await self._audit_via_api(
                                event_id=event_id,
                                match_info=match_info,
                                market_type="AH",
                                line=str(h6.lagged_line),
                                side=h6.lagged_side,
                                websocket_odd=h6.lagged_current_odd,
                            )
                            self.audit_results.append(result)
                            
                            if len(self.audit_results) >= self.num_audits:
                                return
                                
    async def _audit_via_api(
        self, 
        event_id: str, 
        match_info: str,
        market_type: str, 
        line: str, 
        side: str, 
        websocket_odd: float
    ) -> AuditResult:
        """Audita uma odd via API."""
        
        print(f"\n\n>>> AUDITANDO VIA API: {match_info}")
        print(f"    Event ID: {event_id}")
        print(f"    Mercado: {market_type} {line} {side}")
        print(f"    Odd WebSocket: {websocket_odd:.3f}")
        
        # Chama API
        api_data = await self.call_betslip_api(event_id)
        
        if api_data is None:
            print(f"    ERRO: Não conseguiu chamar API")
            return AuditResult(
                timestamp=datetime.now(timezone.utc),
                match_info=match_info,
                event_id=event_id,
                market_type=market_type,
                line=line,
                side=side,
                websocket_odd=websocket_odd,
                api_odd=None,
                difference_pct=None,
                status="API_ERROR"
            )
        
        # Extrai odd da API
        api_odd = self.extract_best_odds_from_api(api_data, line, side)
        
        if api_odd is None:
            print(f"    AVISO: Não conseguiu extrair odd da API")
            print(f"    Resposta API (primeiros 500 chars):")
            print(f"    {json.dumps(api_data, indent=2)[:500]}")
            return AuditResult(
                timestamp=datetime.now(timezone.utc),
                match_info=match_info,
                event_id=event_id,
                market_type=market_type,
                line=line,
                side=side,
                websocket_odd=websocket_odd,
                api_odd=None,
                difference_pct=None,
                status="PARSE_ERROR",
                api_raw=api_data
            )
        
        # Calcula diferença
        diff_pct = ((api_odd - websocket_odd) / websocket_odd) * 100
        
        status = "OK" if abs(diff_pct) < 1 else ("MINOR_DIFF" if abs(diff_pct) < 3 else "MAJOR_DIFF")
        
        print(f"    Odd API:       {api_odd:.3f}")
        print(f"    Diferença:     {diff_pct:+.2f}%")
        print(f"    Status:        {status}")
        
        return AuditResult(
            timestamp=datetime.now(timezone.utc),
            match_info=match_info,
            event_id=event_id,
            market_type=market_type,
            line=line,
            side=side,
            websocket_odd=websocket_odd,
            api_odd=api_odd,
            difference_pct=diff_pct,
            status=status,
            api_raw=api_data
        )
            
    def _print_results(self):
        """Imprime resultados da auditoria."""
        print("\n\n" + "=" * 70)
        print("RESULTADOS DA AUDITORIA")
        print("=" * 70)
        
        ok_count = 0
        minor_diff_count = 0
        major_diff_count = 0
        error_count = 0
        
        diffs = []
        
        for r in self.audit_results:
            print(f"\n{r.match_info} (ID: {r.event_id})")
            print(f"  {r.market_type} {r.line} {r.side}")
            print(f"  WebSocket: {r.websocket_odd:.3f}")
            
            if r.api_odd:
                print(f"  API:       {r.api_odd:.3f}")
                print(f"  Diff:      {r.difference_pct:+.2f}%")
                diffs.append(r.difference_pct)
                
            emoji = {"OK": "✅", "MINOR_DIFF": "⚠️", "MAJOR_DIFF": "❌"}.get(r.status, "❓")
            print(f"  Status:    {emoji} {r.status}")
            
            if r.status == "OK":
                ok_count += 1
            elif r.status == "MINOR_DIFF":
                minor_diff_count += 1
            elif r.status == "MAJOR_DIFF":
                major_diff_count += 1
            else:
                error_count += 1
        
        print("\n" + "=" * 70)
        print("RESUMO")
        print("=" * 70)
        print(f"  ✅ OK (diff < 1%):      {ok_count}/{len(self.audit_results)}")
        print(f"  ⚠️ Diff pequena (1-3%): {minor_diff_count}/{len(self.audit_results)}")
        print(f"  ❌ Diff grande (>3%):   {major_diff_count}/{len(self.audit_results)}")
        print(f"  ❓ Erros:               {error_count}/{len(self.audit_results)}")
        
        if diffs:
            avg_diff = sum(diffs) / len(diffs)
            max_diff = max(diffs, key=abs)
            print(f"\n  Diferença média: {avg_diff:+.2f}%")
            print(f"  Diferença máxima: {max_diff:+.2f}%")
            
        print("\n" + "=" * 70)
        print("CONCLUSÃO")
        print("=" * 70)
        
        if error_count == len(self.audit_results):
            print("❓ Todas as auditorias falharam - verificar estrutura da API")
        elif ok_count == len(self.audit_results) - error_count:
            print("✅ ODDS CORRESPONDEM! Dados do WebSocket são confiáveis.")
        elif major_diff_count > 0:
            print("❌ DIFERENÇAS SIGNIFICATIVAS ENCONTRADAS!")
            print("   Pode haver defasagem entre WebSocket e odds reais.")
        else:
            print("⚠️ Pequenas diferenças encontradas - provavelmente latência normal.")


async def main():
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    
    auditor = RealtimeAPIAuditor(num_audits=5)
    await auditor.run_audit()


if __name__ == "__main__":
    asyncio.run(main())
