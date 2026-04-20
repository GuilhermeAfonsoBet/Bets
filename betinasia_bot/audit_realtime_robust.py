# -*- coding: utf-8 -*-
"""
Auditoria em Tempo Real - Versão Robusta

Baseado na análise do DOM do BetinAsia:
- Odds aparecem em <span> com formato "1.85", "2.10"
- Estrutura AH: HANDICAP + Home + ODD + Away + ODD
- Usa regex testado para extrair linhas de AH
"""

import asyncio
import json
import sys
import re
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
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
    line: str
    side: str
    websocket_odd: float
    dom_odd: Optional[float]
    difference_pct: Optional[float]
    status: str
    all_dom_odds: Optional[Dict] = None  # Para debug


class RobustAuditor:
    """Auditor robusto baseado na estrutura conhecida do BetinAsia."""
    
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
        self.scraper._page.on('websocket', self._on_websocket)
        print("Auditor iniciado e logado")
        
    async def close(self):
        """Fecha o auditor."""
        if self.scraper:
            await self.scraper.close()
            
    def _on_websocket(self, ws):
        """Callback quando WebSocket conecta."""
        ws.on('framereceived', lambda data: self._ws_messages.append(str(data)))

    async def extract_ah_odds_from_page(self) -> Dict[str, Tuple[float, float]]:
        """
        Extrai todas as odds de AH da página atual.
        
        Usa o padrão conhecido:
        HANDICAP
        Home
        1.85
        Away
        2.10
        
        Returns:
            Dict[linha] -> (home_odd, away_odd)
        """
        page = self.scraper._page
        
        try:
            # Primeiro, expande todas as linhas
            for _ in range(3):
                try:
                    btns = await page.query_selector_all("text='Show all lines'")
                    for btn in btns:
                        if await btn.is_visible():
                            await btn.click()
                            await page.wait_for_timeout(300)
                except:
                    pass
            
            await page.wait_for_timeout(500)
            
            # Extrai texto da página
            page_text = await page.inner_text("body")
            
            # Regex para extrair linhas de AH
            # Padrão: HANDICAP seguido de Home + ODD + Away + ODD
            ah_pattern = r'([+-]?\d+[,.]?\d*)\s*\n\s*Home\s*\n\s*(\d+[,.]\d+)\s*\n\s*Away\s*\n\s*(\d+[,.]\d+)'
            
            matches = re.findall(ah_pattern, page_text)
            
            result = {}
            for match in matches:
                line = match[0].replace(",", ".")
                home_odd = float(match[1].replace(",", "."))
                away_odd = float(match[2].replace(",", "."))
                result[line] = (home_odd, away_odd)
            
            return result
            
        except Exception as e:
            logger.error(f"Erro ao extrair odds: {e}")
            return {}
    
    async def run_audit(self):
        """Executa ciclos de auditoria."""
        await self.start()
        
        print("=" * 70)
        print("AUDITORIA EM TEMPO REAL - VERSÃO ROBUSTA")
        print("=" * 70)
        print(f"""
Método:
1. Coleta odds via WebSocket (o que o coletor salva)
2. Navega para página do jogo
3. Expande "Show all lines"
4. Extrai odds do DOM usando regex testado
5. Compara WebSocket vs DOM

Vou auditar {self.num_audits} eventos H6.
""")
        
        audited = set()
        
        try:
            while len(self.audit_results) < self.num_audits:
                self._ws_messages.clear()
                
                await self.scraper._page.goto(self.FOOTBALL_URL)
                await self.scraper._page.wait_for_load_state("networkidle")
                await self.scraper._page.wait_for_timeout(6000)
                
                h6_events = await self._find_h6_events(audited)
                
                for h6 in h6_events:
                    if len(self.audit_results) >= self.num_audits:
                        break
                    result = await self._audit_event(h6)
                    self.audit_results.append(result)
                    audited.add(h6['event_id'])
                
                print(f"\rProcessados: {self.events_processed} | "
                      f"H6: {self.h6_events_detected} | "
                      f"Auditados: {len(self.audit_results)}/{self.num_audits}", 
                      end="", flush=True)
                
                await asyncio.sleep(3)
                
        finally:
            await self.close()
            
        self._print_results()
        
    async def _find_h6_events(self, already_audited: set) -> List[dict]:
        """Encontra eventos H6 nas mensagens WebSocket."""
        events = {}
        h6_list = []
        
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
                            events[msg_meta[1]] = {
                                'home': msg_data.get('home', ''),
                                'away': msg_data.get('away', ''),
                            }
                    
                    if msg_type in ['offers_hcap', 'offers_event']:
                        if isinstance(msg_meta, list) and len(msg_meta) >= 3:
                            if msg_meta[1] == 'fb' and 'ah' in msg_data:
                                event_id = msg_meta[2]
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
                                    
                                    home_odds = away_odds = 0.0
                                    if isinstance(odds_list, list):
                                        for o in odds_list:
                                            if isinstance(o, list) and len(o) >= 2:
                                                if o[0] == 'h':
                                                    home_odds = float(o[1])
                                                elif o[0] == 'a':
                                                    away_odds = float(o[1])
                                    
                                    if home_odds > 0 and away_odds > 0:
                                        self.events_processed += 1
                                        
                                        det = self.hypothesis_detector.process_market_update(
                                            match_id=hash(event_id) % 1000000,
                                            market_type="AH",
                                            line=str(line),
                                            home_odd=home_odds,
                                            away_odd=away_odds,
                                        )
                                        
                                        for h6 in det.get("h6_events", []):
                                            self.h6_events_detected += 1
                                            if event_id not in already_audited:
                                                info = events.get(event_id, {})
                                                h6_list.append({
                                                    'event_id': event_id,
                                                    'match_info': f"{info.get('home', '?')} vs {info.get('away', '?')}",
                                                    'line': str(h6.lagged_line),
                                                    'side': h6.lagged_side,
                                                    'websocket_odd': h6.lagged_current_odd,
                                                })
            except:
                continue
                
        return h6_list
    
    async def _audit_event(self, h6: dict) -> AuditResult:
        """Audita um evento."""
        event_id = h6['event_id']
        match_info = h6['match_info']
        line = h6['line']
        side = h6['side']
        ws_odd = h6['websocket_odd']
        
        print(f"\n\n>>> AUDITANDO: {match_info}")
        print(f"    Event ID: {event_id}")
        print(f"    Linha: AH {line} {side}")
        print(f"    Odd WebSocket: {ws_odd:.3f}")
        
        try:
            # Navega para página do jogo
            game_url = f"https://black.betinasia.com/sportsbook/football/{event_id}"
            await self.scraper._page.goto(game_url)
            await self.scraper._page.wait_for_load_state("networkidle")
            await self.scraper._page.wait_for_timeout(2000)
            
            # Extrai todas as odds de AH
            dom_odds = await self.extract_ah_odds_from_page()
            
            print(f"    Linhas AH no DOM: {len(dom_odds)}")
            
            if not dom_odds:
                print(f"    ERRO: Nenhuma linha AH encontrada no DOM")
                return AuditResult(
                    timestamp=datetime.now(timezone.utc),
                    match_info=match_info,
                    event_id=event_id,
                    line=line,
                    side=side,
                    websocket_odd=ws_odd,
                    dom_odd=None,
                    difference_pct=None,
                    status="NO_AH_FOUND"
                )
            
            # Procura pela linha específica
            # Tenta vários formatos: "2.0", "2", "+2.0", "+2"
            line_float = float(line)
            possible_keys = [
                line,
                str(int(line_float)) if line_float == int(line_float) else None,
                f"+{line}" if line_float > 0 else None,
                f"+{int(line_float)}" if line_float > 0 and line_float == int(line_float) else None,
                str(line_float),
            ]
            possible_keys = [k for k in possible_keys if k]
            
            dom_line_odds = None
            matched_key = None
            for key in possible_keys:
                if key in dom_odds:
                    dom_line_odds = dom_odds[key]
                    matched_key = key
                    break
            
            if not dom_line_odds:
                print(f"    ERRO: Linha {line} não encontrada no DOM")
                print(f"    Linhas disponíveis: {list(dom_odds.keys())[:10]}")
                return AuditResult(
                    timestamp=datetime.now(timezone.utc),
                    match_info=match_info,
                    event_id=event_id,
                    line=line,
                    side=side,
                    websocket_odd=ws_odd,
                    dom_odd=None,
                    difference_pct=None,
                    status="LINE_NOT_FOUND",
                    all_dom_odds=dom_odds
                )
            
            # Pega a odd do lado correto
            dom_odd = dom_line_odds[0] if side == "home" else dom_line_odds[1]
            
            # Calcula diferença
            diff_pct = ((dom_odd - ws_odd) / ws_odd) * 100
            
            if abs(diff_pct) < 0.1:
                status = "IDENTICAL"
            elif abs(diff_pct) < 0.5:
                status = "OK"
            elif abs(diff_pct) < 2:
                status = "MINOR_DIFF"
            else:
                status = "MAJOR_DIFF"
            
            print(f"    DOM ({matched_key} {side}): {dom_odd:.3f}")
            print(f"    Diferença: {diff_pct:+.2f}%")
            print(f"    Status: {status}")
            
            return AuditResult(
                timestamp=datetime.now(timezone.utc),
                match_info=match_info,
                event_id=event_id,
                line=line,
                side=side,
                websocket_odd=ws_odd,
                dom_odd=dom_odd,
                difference_pct=diff_pct,
                status=status,
                all_dom_odds=dom_odds
            )
            
        except Exception as e:
            logger.error(f"Erro: {e}")
            return AuditResult(
                timestamp=datetime.now(timezone.utc),
                match_info=match_info,
                event_id=event_id,
                line=line,
                side=side,
                websocket_odd=ws_odd,
                dom_odd=None,
                difference_pct=None,
                status=f"ERROR: {str(e)[:30]}"
            )
            
    def _print_results(self):
        """Imprime resultados."""
        print("\n\n" + "=" * 70)
        print("RESULTADOS DA AUDITORIA")
        print("=" * 70)
        
        counts = {"IDENTICAL": 0, "OK": 0, "MINOR_DIFF": 0, "MAJOR_DIFF": 0}
        diffs = []
        errors = 0
        
        for r in self.audit_results:
            print(f"\n{r.match_info}")
            print(f"  AH {r.line} {r.side}")
            print(f"  WebSocket: {r.websocket_odd:.3f}")
            
            if r.dom_odd:
                print(f"  DOM:       {r.dom_odd:.3f}")
                print(f"  Diff:      {r.difference_pct:+.2f}%")
                diffs.append(abs(r.difference_pct))
                
            emoji = {
                "IDENTICAL": "✅", "OK": "✅", 
                "MINOR_DIFF": "⚠️", "MAJOR_DIFF": "❌"
            }.get(r.status, "❓")
            print(f"  Status:    {emoji} {r.status}")
            
            if r.status in counts:
                counts[r.status] += 1
            else:
                errors += 1
        
        print("\n" + "=" * 70)
        print("RESUMO")
        print("=" * 70)
        total = len(self.audit_results)
        print(f"  ✅ Idênticas/OK (diff < 0.5%): {counts['IDENTICAL'] + counts['OK']}/{total}")
        print(f"  ⚠️ Diff pequena (0.5-2%):      {counts['MINOR_DIFF']}/{total}")
        print(f"  ❌ Diff grande (>2%):          {counts['MAJOR_DIFF']}/{total}")
        print(f"  ❓ Erros/não encontrado:       {errors}/{total}")
        
        if diffs:
            print(f"\n  Diferença média: {sum(diffs)/len(diffs):.3f}%")
            print(f"  Diferença máxima: {max(diffs):.3f}%")
        
        print("\n" + "=" * 70)
        print("CONCLUSÃO")
        print("=" * 70)
        
        found = counts['IDENTICAL'] + counts['OK'] + counts['MINOR_DIFF'] + counts['MAJOR_DIFF']
        if found == 0:
            print("❓ Nenhuma odd encontrada - verificar estrutura do DOM")
        elif counts['MAJOR_DIFF'] == 0 and counts['MINOR_DIFF'] == 0:
            print("✅ ODDS CORRESPONDEM PERFEITAMENTE!")
            print("   Os dados do WebSocket são confiáveis.")
        elif counts['MAJOR_DIFF'] == 0:
            print("✅ Odds correspondem com pequenas variações (< 2%)")
            print("   Provavelmente latência normal entre WebSocket e DOM.")
        else:
            pct_major = counts['MAJOR_DIFF'] / found * 100
            print(f"⚠️ {pct_major:.0f}% das odds têm diferença significativa (> 2%)")
            print("   Pode haver defasagem nos dados ou mudanças rápidas.")


async def main():
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    
    auditor = RobustAuditor(num_audits=5)
    await auditor.run_audit()


if __name__ == "__main__":
    asyncio.run(main())
