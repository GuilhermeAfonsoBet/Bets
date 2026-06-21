# -*- coding: utf-8 -*-
"""
Auditoria em Tempo Real de Odds

Roda como o coletor normal, mas quando detecta um evento de valor,
IMEDIATAMENTE verifica o betslip para comparar as odds.

Uso:
    python audit_realtime_odds.py

Este script NÃO salva no banco - apenas verifica e reporta diferenças.
"""

import asyncio
import json
import sys
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
    market_type: str
    line: str
    side: str
    collected_odd: float
    betslip_odd: Optional[float]
    difference_pct: Optional[float]
    status: str


class RealtimeAuditor:
    """Auditor em tempo real de odds."""
    
    FOOTBALL_URL = "https://black.betinasia.com/sportsbook/football"
    WAIT_TIME_MS = 6000
    
    def __init__(self, num_audits: int = 10):
        self.scraper: Optional[BetinAsiaScraper] = None
        self._ws_messages: List[str] = []
        self.hypothesis_detector = HypothesisDetector()
        self.num_audits = num_audits  # Quantas auditorias fazer
        self.audit_results: List[AuditResult] = []
        self.events_processed = 0
        self.h6_events_detected = 0
        
    async def start(self):
        """Inicia o auditor."""
        self.scraper = BetinAsiaScraper()
        await self.scraper.start()
        await self.scraper.login()
        
        # Configura listener de WebSocket
        self.scraper._page.on('websocket', self._on_websocket)
        
        logger.info("RealtimeAuditor iniciado")
        
    async def close(self):
        """Fecha o auditor."""
        if self.scraper:
            await self.scraper.close()
            
    def _on_websocket(self, ws):
        """Callback quando WebSocket conecta."""
        ws.on('framereceived', lambda data: self._ws_messages.append(str(data)))
        
    async def run_audit(self):
        """Executa ciclos de auditoria."""
        await self.start()
        
        print("=" * 70)
        print("AUDITORIA EM TEMPO REAL DE ODDS")
        print("=" * 70)
        print(f"Vou auditar {self.num_audits} eventos H6 detectados.")
        print("Cada vez que um evento H6 for detectado, vou verificar o betslip.\n")
        
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
                
                # Pequena pausa antes do próximo ciclo
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
                        match_id=hash(event_id) % 1000000,  # ID fictício
                        market_type="AH",
                        line=str(line),
                        home_odd=home_odds,
                        away_odd=away_odds,
                    )
                    
                    # Se detectou H6, audita!
                    h6_events = detector_events.get("h6_events", [])
                    if h6_events and len(self.audit_results) < self.num_audits:
                        self.h6_events_detected += len(h6_events)
                        
                        for h6 in h6_events:
                            # Faz auditoria imediata
                            result = await self._audit_betslip(
                                event_id=event_id,
                                match_info=match_info,
                                market_type="AH",
                                line=str(h6.lagged_line),
                                side=h6.lagged_side,
                                collected_odd=h6.lagged_current_odd,
                            )
                            self.audit_results.append(result)
                            
                            if len(self.audit_results) >= self.num_audits:
                                return
                                
    async def _audit_betslip(
        self, 
        event_id: str, 
        match_info: str,
        market_type: str, 
        line: str, 
        side: str, 
        collected_odd: float
    ) -> AuditResult:
        """
        Audita uma odd específica clicando no betslip.
        """
        print(f"\n\n>>> AUDITANDO: {match_info} | {market_type} {line} {side} | Odd coletada: {collected_odd:.3f}")
        
        try:
            page = self.scraper._page
            
            # Mapeia side para texto do botão
            side_text = {
                "home": "h",
                "away": "a", 
                "over": "over",
                "under": "under"
            }.get(side, side)
            
            # Tenta encontrar e clicar na odd
            # Primeiro, procura pelo valor da odd na página
            odd_str = f"{collected_odd:.2f}"
            
            # Seletores para encontrar odds
            # Adaptado para estrutura comum de sites de apostas
            selectors = [
                # Por valor da odd
                f'button:has-text("{odd_str}")',
                f'[class*="odd"]:has-text("{odd_str}")',
                f'span:has-text("{odd_str}")',
                # Por atributos data
                f'[data-line="{line}"][data-side="{side_text}"]',
                f'[data-handicap="{line}"]',
            ]
            
            clicked = False
            for selector in selectors:
                try:
                    elements = await page.query_selector_all(selector)
                    for el in elements[:5]:  # Tenta nos primeiros 5 matches
                        try:
                            await el.click()
                            await page.wait_for_timeout(500)
                            clicked = True
                            break
                        except:
                            continue
                    if clicked:
                        break
                except:
                    continue
            
            if not clicked:
                logger.warning(f"Não conseguiu clicar na odd {line} {side}")
                return AuditResult(
                    timestamp=datetime.now(timezone.utc),
                    match_info=match_info,
                    market_type=market_type,
                    line=line,
                    side=side,
                    collected_odd=collected_odd,
                    betslip_odd=None,
                    difference_pct=None,
                    status="NOT_FOUND"
                )
            
            # Espera betslip aparecer e captura valor
            await page.wait_for_timeout(1000)
            
            # Tenta capturar a odd do betslip
            betslip_selectors = [
                '.betslip-selection .odd',
                '.bet-slip .odds',
                '[class*="betslip"] [class*="odd"]',
                '.selection-price',
                '.stake-odds',
            ]
            
            betslip_odd = None
            for selector in betslip_selectors:
                try:
                    el = await page.query_selector(selector)
                    if el:
                        text = await el.inner_text()
                        # Extrai número do texto
                        import re
                        numbers = re.findall(r'\d+\.?\d*', text)
                        if numbers:
                            betslip_odd = float(numbers[0])
                            break
                except:
                    continue
            
            # Fecha betslip
            try:
                close_btns = await page.query_selector_all('[class*="close"], [class*="remove"], .clear-betslip')
                for btn in close_btns:
                    try:
                        await btn.click()
                        break
                    except:
                        continue
            except:
                pass
            
            if betslip_odd is None:
                logger.warning("Não conseguiu capturar odd do betslip")
                return AuditResult(
                    timestamp=datetime.now(timezone.utc),
                    match_info=match_info,
                    market_type=market_type,
                    line=line,
                    side=side,
                    collected_odd=collected_odd,
                    betslip_odd=None,
                    difference_pct=None,
                    status="BETSLIP_NOT_FOUND"
                )
            
            # Calcula diferença
            diff_pct = ((betslip_odd - collected_odd) / collected_odd) * 100
            
            status = "OK" if abs(diff_pct) < 1 else ("MINOR_DIFF" if abs(diff_pct) < 3 else "MAJOR_DIFF")
            
            print(f"    Odd no betslip: {betslip_odd:.3f} | Diferença: {diff_pct:+.2f}% | {status}")
            
            return AuditResult(
                timestamp=datetime.now(timezone.utc),
                match_info=match_info,
                market_type=market_type,
                line=line,
                side=side,
                collected_odd=collected_odd,
                betslip_odd=betslip_odd,
                difference_pct=diff_pct,
                status=status
            )
            
        except Exception as e:
            logger.error(f"Erro na auditoria: {e}")
            return AuditResult(
                timestamp=datetime.now(timezone.utc),
                match_info=match_info,
                market_type=market_type,
                line=line,
                side=side,
                collected_odd=collected_odd,
                betslip_odd=None,
                difference_pct=None,
                status=f"ERROR: {str(e)[:50]}"
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
            print(f"\n{r.match_info}")
            print(f"  {r.market_type} {r.line} {r.side}")
            print(f"  Coletada: {r.collected_odd:.3f}")
            
            if r.betslip_odd:
                print(f"  Betslip:  {r.betslip_odd:.3f}")
                print(f"  Diff:     {r.difference_pct:+.2f}%")
                diffs.append(r.difference_pct)
                
            emoji = {"OK": "✅", "MINOR_DIFF": "⚠️", "MAJOR_DIFF": "❌"}.get(r.status, "❓")
            print(f"  Status:   {emoji} {r.status}")
            
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
        print(f"  ❓ Erros/não encontrado: {error_count}/{len(self.audit_results)}")
        
        if diffs:
            avg_diff = sum(diffs) / len(diffs)
            max_diff = max(diffs, key=abs)
            print(f"\n  Diferença média: {avg_diff:+.2f}%")
            print(f"  Diferença máxima: {max_diff:+.2f}%")
            
        print("\n" + "=" * 70)
        if ok_count == len(self.audit_results):
            print("✅ TODAS AS ODDS CORRESPONDEM! Dados confiáveis.")
        elif error_count > len(self.audit_results) / 2:
            print("⚠️ Muitos erros - seletores CSS podem precisar de ajuste.")
        elif major_diff_count > 0:
            print("❌ DIFERENÇAS SIGNIFICATIVAS ENCONTRADAS!")
            print("   As odds coletadas podem não corresponder às reais.")
        else:
            print("⚠️ Pequenas diferenças encontradas - pode ser latência normal.")


async def main():
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    
    auditor = RealtimeAuditor(num_audits=5)
    await auditor.run_audit()


if __name__ == "__main__":
    asyncio.run(main())
