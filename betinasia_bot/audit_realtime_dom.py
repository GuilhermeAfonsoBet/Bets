# -*- coding: utf-8 -*-
"""
Auditoria em Tempo Real via DOM

Quando detecta um evento H6:
1. Navega para a página do jogo
2. Extrai as odds diretamente do DOM
3. Compara com as odds do WebSocket

Mais simples e confiável que a API.
"""

import asyncio
import json
import sys
import re
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
    dom_odd: Optional[float]
    difference_pct: Optional[float]
    status: str


class RealtimeDOMAuditor:
    """Auditor em tempo real extraindo odds do DOM."""
    
    FOOTBALL_URL = "https://black.betinasia.com/sportsbook/football"
    
    def __init__(self, num_audits: int = 5):
        self.scraper: Optional[BetinAsiaScraper] = None
        self._ws_messages: List[str] = []
        self.hypothesis_detector = HypothesisDetector()
        self.num_audits = num_audits
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
        
        print("Auditor iniciado e logado")
        
    async def close(self):
        """Fecha o auditor."""
        if self.scraper:
            await self.scraper.close()
            
    def _on_websocket(self, ws):
        """Callback quando WebSocket conecta."""
        ws.on('framereceived', lambda data: self._ws_messages.append(str(data)))
        
    async def extract_odds_from_page(self, line: str, side: str) -> Optional[float]:
        """
        Extrai odds do DOM da página atual.
        
        Args:
            line: Linha do mercado (ex: "2.0", "-0.5")
            side: Lado (home/away)
            
        Returns:
            Odd encontrada ou None
        """
        page = self.scraper._page
        
        try:
            # Extrai todas as odds visíveis na página via JavaScript
            odds_data = await page.evaluate("""
                () => {
                    const results = [];
                    
                    // Procura por elementos que contenham odds
                    // Estrutura típica: spans/buttons com valores numéricos como "2.41"
                    const elements = document.querySelectorAll('span, button, div');
                    
                    elements.forEach(el => {
                        const text = el.innerText || el.textContent || '';
                        // Procura por padrão de odd (número com 2-3 dígitos e decimais)
                        const match = text.match(/^(\d{1,2}\.\d{2,3})$/);
                        if (match) {
                            // Tenta pegar contexto (linha, lado)
                            const parent = el.closest('tr, div[class*="line"], div[class*="row"]');
                            let context = '';
                            if (parent) {
                                context = parent.innerText || '';
                            }
                            results.push({
                                odd: parseFloat(match[1]),
                                context: context.substring(0, 200),
                                html: el.outerHTML.substring(0, 200)
                            });
                        }
                    });
                    
                    return results;
                }
            """)
            
            if not odds_data:
                return None
            
            # Procura pela odd que corresponde à linha e lado
            line_float = float(line)
            side_keywords = {
                "home": ["home", "1", "h"],
                "away": ["away", "2", "a"],
                "over": ["over", "o"],
                "under": ["under", "u"]
            }
            keywords = side_keywords.get(side, [side])
            
            # Procura por correspondência no contexto
            for item in odds_data:
                context_lower = item['context'].lower()
                
                # Verifica se o contexto menciona a linha
                if str(line) in context_lower or str(line_float) in context_lower:
                    # Verifica se menciona o lado
                    for kw in keywords:
                        if kw in context_lower:
                            return item['odd']
            
            # Se não encontrou com contexto, retorna a primeira odd próxima
            # (isso é um fallback, não ideal)
            return None
            
        except Exception as e:
            logger.error(f"Erro ao extrair odds do DOM: {e}")
            return None
        
    async def run_audit(self):
        """Executa ciclos de auditoria."""
        await self.start()
        
        print("=" * 70)
        print("AUDITORIA EM TEMPO REAL VIA DOM")
        print("=" * 70)
        print(f"""
Este script:
1. Detecta evento H6 via WebSocket
2. Navega para a página do jogo
3. Extrai odds diretamente do DOM
4. Compara com as odds do WebSocket

Vou auditar {self.num_audits} eventos H6.
""")
        
        audited_events = set()  # Para não auditar o mesmo evento múltiplas vezes
        
        try:
            while len(self.audit_results) < self.num_audits:
                # Coleta dados
                self._ws_messages.clear()
                
                await self.scraper._page.goto(self.FOOTBALL_URL)
                await self.scraper._page.wait_for_load_state("networkidle")
                await self.scraper._page.wait_for_timeout(6000)
                
                # Parseia mensagens e detecta eventos
                h6_to_audit = await self._find_h6_events(audited_events)
                
                # Audita cada evento encontrado
                for h6_info in h6_to_audit:
                    if len(self.audit_results) >= self.num_audits:
                        break
                        
                    result = await self._audit_event(h6_info)
                    self.audit_results.append(result)
                    audited_events.add(h6_info['event_id'])
                
                # Status
                print(f"\rProcessados: {self.events_processed} mercados | "
                      f"H6 detectados: {self.h6_events_detected} | "
                      f"Auditados: {len(self.audit_results)}/{self.num_audits}", 
                      end="", flush=True)
                
                await asyncio.sleep(3)
                
        finally:
            await self.close()
            
        self._print_results()
        
    async def _find_h6_events(self, already_audited: set) -> List[dict]:
        """Encontra eventos H6 nas mensagens WebSocket."""
        events = {}  # event_id -> info
        h6_to_audit = []
        
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
                            }
                    
                    # Processa offers
                    if msg_type in ['offers_hcap', 'offers_event']:
                        if isinstance(msg_meta, list) and len(msg_meta) >= 3:
                            sport_type = msg_meta[1]
                            event_id = msg_meta[2]
                            
                            if sport_type == 'fb' and 'ah' in msg_data:
                                ah_data = msg_data['ah']
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
                                        for odd_item in odds_list:
                                            if isinstance(odd_item, list) and len(odd_item) >= 2:
                                                if odd_item[0] == 'h':
                                                    home_odds = float(odd_item[1])
                                                elif odd_item[0] == 'a':
                                                    away_odds = float(odd_item[1])
                                    
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
                                        
                                        h6_list = detector_events.get("h6_events", [])
                                        if h6_list:
                                            self.h6_events_detected += len(h6_list)
                                            
                                            for h6 in h6_list:
                                                if event_id not in already_audited:
                                                    event_info = events.get(event_id, {})
                                                    h6_to_audit.append({
                                                        'event_id': event_id,
                                                        'match_info': f"{event_info.get('home', '?')} vs {event_info.get('away', '?')}",
                                                        'line': str(h6.lagged_line),
                                                        'side': h6.lagged_side,
                                                        'websocket_odd': h6.lagged_current_odd,
                                                    })
                                                    
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                continue
                
        return h6_to_audit
    
    async def _audit_event(self, h6_info: dict) -> AuditResult:
        """Audita um evento navegando para a página e extraindo odds do DOM."""
        
        event_id = h6_info['event_id']
        match_info = h6_info['match_info']
        line = h6_info['line']
        side = h6_info['side']
        websocket_odd = h6_info['websocket_odd']
        
        print(f"\n\n>>> AUDITANDO: {match_info}")
        print(f"    Event ID: {event_id}")
        print(f"    Mercado: AH {line} {side}")
        print(f"    Odd WebSocket: {websocket_odd:.3f}")
        
        try:
            # Navega para a página do jogo
            game_url = f"https://black.betinasia.com/sportsbook/football/{event_id}"
            await self.scraper._page.goto(game_url)
            await self.scraper._page.wait_for_load_state("networkidle")
            await self.scraper._page.wait_for_timeout(3000)
            
            # Tenta extrair a odd do DOM
            # Primeiro, procura o valor exato da odd no WebSocket
            odd_str = f"{websocket_odd:.2f}"
            
            # Verifica se a odd está visível na página
            dom_odd = await self.scraper._page.evaluate(f"""
                () => {{
                    // Procura pelo valor exato da odd
                    const searchValue = "{odd_str}";
                    const elements = document.querySelectorAll('span, button, div');
                    
                    for (const el of elements) {{
                        const text = (el.innerText || el.textContent || '').trim();
                        if (text === searchValue) {{
                            return parseFloat(text);
                        }}
                    }}
                    
                    // Procura por valor similar (diferença de 0.01)
                    const targetOdd = {websocket_odd};
                    for (const el of elements) {{
                        const text = (el.innerText || el.textContent || '').trim();
                        const match = text.match(/^(\\d{{1,2}}\\.\\d{{2,3}})$/);
                        if (match) {{
                            const odd = parseFloat(match[1]);
                            if (Math.abs(odd - targetOdd) < 0.02) {{
                                return odd;
                            }}
                        }}
                    }}
                    
                    return null;
                }}
            """)
            
            if dom_odd is None:
                print(f"    DOM: Não encontrou odd similar na página")
                
                # Mostra odds que estão na página para debug
                visible_odds = await self.scraper._page.evaluate("""
                    () => {
                        const odds = [];
                        const elements = document.querySelectorAll('span, button, div');
                        elements.forEach(el => {
                            const text = (el.innerText || el.textContent || '').trim();
                            const match = text.match(/^(\\d{1,2}\\.\\d{2,3})$/);
                            if (match) {
                                odds.push(parseFloat(match[1]));
                            }
                        });
                        return [...new Set(odds)].sort((a,b) => a-b).slice(0, 20);
                    }
                """)
                print(f"    Odds visíveis na página: {visible_odds[:10]}...")
                
                return AuditResult(
                    timestamp=datetime.now(timezone.utc),
                    match_info=match_info,
                    event_id=event_id,
                    market_type="AH",
                    line=line,
                    side=side,
                    websocket_odd=websocket_odd,
                    dom_odd=None,
                    difference_pct=None,
                    status="NOT_FOUND"
                )
            
            # Calcula diferença
            diff_pct = ((dom_odd - websocket_odd) / websocket_odd) * 100
            
            status = "OK" if abs(diff_pct) < 0.5 else ("MINOR_DIFF" if abs(diff_pct) < 2 else "MAJOR_DIFF")
            
            print(f"    DOM:          {dom_odd:.3f}")
            print(f"    Diferença:    {diff_pct:+.2f}%")
            print(f"    Status:       {status}")
            
            return AuditResult(
                timestamp=datetime.now(timezone.utc),
                match_info=match_info,
                event_id=event_id,
                market_type="AH",
                line=line,
                side=side,
                websocket_odd=websocket_odd,
                dom_odd=dom_odd,
                difference_pct=diff_pct,
                status=status
            )
            
        except Exception as e:
            logger.error(f"Erro na auditoria: {e}")
            return AuditResult(
                timestamp=datetime.now(timezone.utc),
                match_info=match_info,
                event_id=event_id,
                market_type="AH",
                line=line,
                side=side,
                websocket_odd=websocket_odd,
                dom_odd=None,
                difference_pct=None,
                status=f"ERROR: {str(e)[:30]}"
            )
            
    def _print_results(self):
        """Imprime resultados."""
        print("\n\n" + "=" * 70)
        print("RESULTADOS DA AUDITORIA")
        print("=" * 70)
        
        ok_count = 0
        minor_diff_count = 0
        major_diff_count = 0
        not_found_count = 0
        error_count = 0
        diffs = []
        
        for r in self.audit_results:
            print(f"\n{r.match_info}")
            print(f"  AH {r.line} {r.side}")
            print(f"  WebSocket: {r.websocket_odd:.3f}")
            
            if r.dom_odd:
                print(f"  DOM:       {r.dom_odd:.3f}")
                print(f"  Diff:      {r.difference_pct:+.2f}%")
                diffs.append(abs(r.difference_pct))
                
            emoji = {"OK": "✅", "MINOR_DIFF": "⚠️", "MAJOR_DIFF": "❌", "NOT_FOUND": "🔍"}.get(r.status, "❓")
            print(f"  Status:    {emoji} {r.status}")
            
            if r.status == "OK":
                ok_count += 1
            elif r.status == "MINOR_DIFF":
                minor_diff_count += 1
            elif r.status == "MAJOR_DIFF":
                major_diff_count += 1
            elif r.status == "NOT_FOUND":
                not_found_count += 1
            else:
                error_count += 1
        
        print("\n" + "=" * 70)
        print("RESUMO")
        print("=" * 70)
        total = len(self.audit_results)
        print(f"  ✅ OK (diff < 0.5%):    {ok_count}/{total}")
        print(f"  ⚠️ Diff pequena (0.5-2%): {minor_diff_count}/{total}")
        print(f"  ❌ Diff grande (>2%):   {major_diff_count}/{total}")
        print(f"  🔍 Não encontrado:      {not_found_count}/{total}")
        print(f"  ❓ Erros:               {error_count}/{total}")
        
        if diffs:
            print(f"\n  Diferença média: {sum(diffs)/len(diffs):.2f}%")
            print(f"  Diferença máxima: {max(diffs):.2f}%")
        
        print("\n" + "=" * 70)
        print("CONCLUSÃO")
        print("=" * 70)
        
        found = ok_count + minor_diff_count + major_diff_count
        if found == 0:
            print("🔍 Nenhuma odd foi encontrada na página - verificar estrutura do DOM")
        elif ok_count == found:
            print("✅ ODDS CORRESPONDEM! Dados do WebSocket são confiáveis.")
        elif major_diff_count == 0:
            print("⚠️ Pequenas diferenças encontradas - provavelmente latência normal.")
        else:
            print("❌ DIFERENÇAS SIGNIFICATIVAS! Pode haver defasagem nos dados.")


async def main():
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    
    auditor = RealtimeDOMAuditor(num_audits=5)
    await auditor.run_audit()


if __name__ == "__main__":
    asyncio.run(main())
