# -*- coding: utf-8 -*-
"""
Auditoria H3B: WebSocket vs Best Odd do Betslip

Compara as odds de eventos H3B (reversão temporal UP) coletadas via WebSocket 
com a BEST ODD real exibida no painel do betslip.

Foco: Reversão UP (odd subiu = melhorou)
"""

import asyncio
import json
import sys
import re
from datetime import datetime, timezone
from typing import Optional, Dict, List
from dataclasses import dataclass
from loguru import logger

sys.path.insert(0, '.')

from scraper.betinasia import BetinAsiaScraper
from scraper.betslip_extractor import BetslipExtractor, BetslipData
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
    betslip_best_odd: Optional[float]
    betslip_limit: Optional[float]
    difference_pct: Optional[float]
    status: str
    reversal_direction: str  # "up" ou "down"
    betslip_data: Optional[BetslipData] = None


class H3BAuditor:
    """Auditor que compara WebSocket vs Betslip Best Odd para eventos H3B."""
    
    FOOTBALL_URL = "https://black.betinasia.com/sportsbook/football"
    
    def __init__(self, num_audits: int = 50, direction_filter: str = "up"):
        """
        Args:
            num_audits: Número de auditorias a realizar
            direction_filter: "up" para reversão UP, "down" para DOWN, "all" para ambas
        """
        self.scraper: Optional[BetinAsiaScraper] = None
        self.extractor: Optional[BetslipExtractor] = None
        self._ws_messages: List[str] = []
        self.hypothesis_detector = HypothesisDetector()
        self.num_audits = num_audits
        self.direction_filter = direction_filter
        self.audit_results: List[AuditResult] = []
        self.events_processed = 0
        self.h3b_events_detected = 0
        
    async def start(self):
        """Inicia o auditor."""
        self.scraper = BetinAsiaScraper()
        await self.scraper.start()
        await self.scraper.login()
        
        self.extractor = BetslipExtractor(self.scraper._page)
        self.scraper._page.on('websocket', self._on_websocket)
        
        print("Auditor H3B iniciado e logado")
        
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
        
        direction_label = {
            "up": "REVERSÃO UP (odd subiu)",
            "down": "REVERSÃO DOWN (odd desceu)",
            "all": "TODAS AS REVERSÕES"
        }.get(self.direction_filter, self.direction_filter)
        
        print("=" * 70)
        print("AUDITORIA H3B: WEBSOCKET vs BEST ODD DO BETSLIP")
        print("=" * 70)
        print(f"""
Este script compara:
- Odd de eventos H3B (reversão temporal) via WebSocket
- Best Odd do Betslip ("Todos Os Agentes De Apostas" → MELHOR)

Filtro: {direction_label}

Processo:
1. Coleta odds via WebSocket
2. Quando detecta H3B {direction_label}, navega para o jogo
3. Clica na odd para abrir betslip
4. Extrai a best odd do painel
5. Compara e reporta diferença

Vou auditar {self.num_audits} eventos.
""")
        
        audited = set()
        
        try:
            while len(self.audit_results) < self.num_audits:
                self._ws_messages.clear()
                
                await self.scraper._page.goto(self.FOOTBALL_URL)
                await self.scraper._page.wait_for_load_state("domcontentloaded")
                await self.scraper._page.wait_for_timeout(8000)
                
                h3b_events = await self._find_h3b_events(audited)
                
                if h3b_events:
                    print(f"\n    → {len(h3b_events)} H3B novos para auditar neste ciclo")
                
                for h3b in h3b_events:
                    if len(self.audit_results) >= self.num_audits:
                        break
                    result = await self._audit_event(h3b)
                    self.audit_results.append(result)
                    audited.add(h3b['audit_key'])
                
                print(f"\rProcessados: {self.events_processed} | "
                      f"H3B: {self.h3b_events_detected} | "
                      f"Auditados: {len(self.audit_results)}/{self.num_audits}", 
                      end="", flush=True)
                
                await asyncio.sleep(3)
                
        finally:
            await self.close()
            
        self._print_results()
        
    async def _find_h3b_events(self, already_audited: set) -> List[dict]:
        """Encontra eventos H3B nas mensagens WebSocket."""
        events = {}
        h3b_list = []
        skipped_already_audited = 0
        skipped_wrong_direction = 0
        
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
                        if msg_meta[0] == 'fb' and 'home' in msg_data:
                            events[msg_meta[1]] = {
                                'home': msg_data.get('home', ''),
                                'away': msg_data.get('away', ''),
                            }
                    
                    # Processa odds de AH e OU
                    if msg_type in ['offers_hcap', 'offers_event']:
                        if isinstance(msg_meta, list) and len(msg_meta) >= 3:
                            if msg_meta[1] == 'fb':
                                event_id = msg_meta[2]
                                
                                # Processa AH
                                if 'ah' in msg_data:
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
                                            
                                        line = line_data[0]
                                        odds_list = line_data[1] if len(line_data) > 1 else []
                                        
                                        home_odds = 0
                                        away_odds = 0
                                        
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
                                            
                                            for h3b in det.get("h3b_events", []):
                                                self.h3b_events_detected += 1
                                                
                                                # Filtro por direção
                                                direction = h3b.direction_after
                                                if self.direction_filter != "all" and direction != self.direction_filter:
                                                    skipped_wrong_direction += 1
                                                    continue
                                                
                                                # Chave única
                                                audit_key = f"{event_id}|AH|{h3b.ah_line}|{h3b.side}"
                                                
                                                if audit_key not in already_audited:
                                                    info = events.get(event_id, {})
                                                    h3b_list.append({
                                                        'event_id': event_id,
                                                        'audit_key': audit_key,
                                                        'match_info': f"{info.get('home', '?')} vs {info.get('away', '?')}",
                                                        'market_type': 'AH',
                                                        'line': str(h3b.ah_line),
                                                        'side': h3b.side,
                                                        'websocket_odd': h3b.odd_at_reversal,
                                                        'direction': direction,
                                                    })
                                                else:
                                                    skipped_already_audited += 1
                                
                                # Processa OU
                                if 'ou' in msg_data:
                                    ou_data = msg_data['ou']
                                    lines = []
                                    
                                    if isinstance(ou_data, list) and len(ou_data) >= 2:
                                        if isinstance(ou_data[0], (int, float)):
                                            lines = [ou_data]
                                        elif isinstance(ou_data[0], list):
                                            lines = ou_data
                                    
                                    for line_data in lines:
                                        if len(line_data) < 2:
                                            continue
                                        
                                        line = line_data[0]
                                        odds_list = line_data[1] if len(line_data) > 1 else []
                                        
                                        over_odds = 0
                                        under_odds = 0
                                        
                                        if isinstance(odds_list, list):
                                            for o in odds_list:
                                                if isinstance(o, list) and len(o) >= 2:
                                                    if o[0] == 'o':
                                                        over_odds = float(o[1])
                                                    elif o[0] == 'u':
                                                        under_odds = float(o[1])
                                        
                                        if over_odds > 0 and under_odds > 0:
                                            self.events_processed += 1
                                            
                                            det = self.hypothesis_detector.process_market_update(
                                                match_id=hash(event_id) % 1000000,
                                                market_type="OU",
                                                line=str(line),
                                                home_odd=over_odds,
                                                away_odd=under_odds,
                                            )
                                            
                                            for h3b in det.get("h3b_events", []):
                                                self.h3b_events_detected += 1
                                                
                                                direction = h3b.direction_after
                                                if self.direction_filter != "all" and direction != self.direction_filter:
                                                    skipped_wrong_direction += 1
                                                    continue
                                                
                                                audit_key = f"{event_id}|OU|{h3b.ah_line}|{h3b.side}"
                                                
                                                if audit_key not in already_audited:
                                                    info = events.get(event_id, {})
                                                    h3b_list.append({
                                                        'event_id': event_id,
                                                        'audit_key': audit_key,
                                                        'match_info': f"{info.get('home', '?')} vs {info.get('away', '?')}",
                                                        'market_type': 'OU',
                                                        'line': str(h3b.ah_line),
                                                        'side': h3b.side,
                                                        'websocket_odd': h3b.odd_at_reversal,
                                                        'direction': direction,
                                                    })
                                                else:
                                                    skipped_already_audited += 1
            except:
                continue
        
        if skipped_already_audited > 0:
            print(f"\n    (Pulou {skipped_already_audited} H3B já auditados)")
        if skipped_wrong_direction > 0:
            print(f"\n    (Pulou {skipped_wrong_direction} H3B com direção diferente de '{self.direction_filter}')")
                
        return h3b_list
    
    async def _audit_event(self, h3b: dict) -> AuditResult:
        """Audita um evento abrindo o betslip."""
        event_id = h3b['event_id']
        match_info = h3b['match_info']
        market_type = h3b['market_type']
        line = h3b['line']
        side = h3b['side']
        ws_odd = h3b['websocket_odd']
        direction = h3b['direction']
        
        print(f"\n\n>>> AUDITANDO H3B ({direction.upper()}): {match_info}")
        print(f"    Event ID: {event_id}")
        print(f"    Mercado: {market_type} {line} {side}")
        print(f"    Odd WebSocket: {ws_odd:.3f}")
        
        try:
            page = self.scraper._page
            
            # Extrai nomes dos times
            teams = match_info.split(' vs ')
            home_team = teams[0].strip() if len(teams) > 0 else ""
            away_team = teams[1].strip() if len(teams) > 1 else ""
            
            print(f"    Buscando jogo: '{home_team}' vs '{away_team}'")
            
            # Busca o jogo na página atual
            game_found = await self._find_and_click_game(home_team, away_team)
            
            if not game_found:
                print(f"    Jogo não encontrado na página atual, tentando navegar...")
                await page.goto("https://black.betinasia.com/sportsbook/football")
                await page.wait_for_load_state("domcontentloaded")
                await page.wait_for_timeout(3000)
                
                await self._expand_all_lines()
                await page.wait_for_timeout(1500)
                
                game_found = await self._find_and_click_game(home_team, away_team)
            
            if not game_found:
                print(f"    ERRO: Jogo não encontrado")
                return AuditResult(
                    timestamp=datetime.now(timezone.utc),
                    match_info=match_info,
                    event_id=event_id,
                    market_type=market_type,
                    line=line,
                    side=side,
                    websocket_odd=ws_odd,
                    betslip_best_odd=None,
                    betslip_limit=None,
                    difference_pct=None,
                    status="GAME_NOT_FOUND",
                    reversal_direction=direction
                )
            
            await page.wait_for_timeout(2000)
            
            print(f"    Expandindo linhas...")
            await self._expand_all_lines()
            await page.wait_for_timeout(1500)
            
            print(f"    Clicando na odd {line} {side}...")
            
            line_display = line.replace(".", ",")
            
            # Tenta clicar com retry
            click_result = None
            max_attempts = 5
            
            for attempt in range(max_attempts):
                click_result = await self._click_specific_odd(line_display, side, market_type)
                
                if click_result == True:
                    break
                
                remaining = max_attempts - attempt - 1
                if remaining > 0:
                    print(f"    Tentativa {attempt + 1}/{max_attempts} falhou, aguardando...")
                    await page.wait_for_timeout(2000)
                    
                    if attempt > 0 and attempt % 2 == 0:
                        print(f"    Recarregando página...")
                        await page.reload()
                        await page.wait_for_load_state("domcontentloaded")
                        await page.wait_for_timeout(2000)
                        
                        game_found = await self._find_and_click_game(home_team, away_team)
                        if not game_found:
                            continue
                        
                        await page.wait_for_timeout(1500)
                    
                    print(f"    Re-expandindo linhas...")
                    await self._expand_all_lines()
                    await page.wait_for_timeout(1000)
            
            if click_result != True:
                print(f"    LINHA CONFIRMADA COMO NÃO DISPONÍVEL após {max_attempts} tentativas")
                return AuditResult(
                    timestamp=datetime.now(timezone.utc),
                    match_info=match_info,
                    event_id=event_id,
                    market_type=market_type,
                    line=line,
                    side=side,
                    websocket_odd=ws_odd,
                    betslip_best_odd=None,
                    betslip_limit=None,
                    difference_pct=None,
                    status="LINE_NOT_AVAILABLE",
                    reversal_direction=direction
                )
            
            await page.wait_for_timeout(2000)
            
            # Extrai dados do betslip
            betslip_data = await self.extractor.extract_best_odd()
            
            if not betslip_data:
                print(f"    ERRO: Não conseguiu extrair dados do betslip")
                return AuditResult(
                    timestamp=datetime.now(timezone.utc),
                    match_info=match_info,
                    event_id=event_id,
                    market_type=market_type,
                    line=line,
                    side=side,
                    websocket_odd=ws_odd,
                    betslip_best_odd=None,
                    betslip_limit=None,
                    difference_pct=None,
                    status="EXTRACT_FAILED",
                    reversal_direction=direction
                )
            
            best_odd = betslip_data.best_odd
            best_limit = betslip_data.best_limit
            
            diff_pct = ((best_odd - ws_odd) / ws_odd) * 100
            
            if abs(diff_pct) < 0.1:
                status = "IDENTICAL"
            elif abs(diff_pct) < 0.5:
                status = "OK"
            elif abs(diff_pct) < 2:
                status = "MINOR_DIFF"
            else:
                status = "MAJOR_DIFF"
            
            print(f"    Betslip Best Odd: {best_odd:.3f}")
            print(f"    Betslip Limite: ${best_limit:,.0f}")
            print(f"    Diferença: {diff_pct:+.2f}%")
            print(f"    Status: {status}")
            
            # Fecha betslip
            await self.extractor.close_betslip()
            
            return AuditResult(
                timestamp=datetime.now(timezone.utc),
                match_info=match_info,
                event_id=event_id,
                market_type=market_type,
                line=line,
                side=side,
                websocket_odd=ws_odd,
                betslip_best_odd=best_odd,
                betslip_limit=best_limit,
                difference_pct=diff_pct,
                status=status,
                reversal_direction=direction,
                betslip_data=betslip_data
            )
            
        except Exception as e:
            logger.error(f"Erro: {e}")
            return AuditResult(
                timestamp=datetime.now(timezone.utc),
                match_info=match_info,
                event_id=event_id,
                market_type=market_type,
                line=line,
                side=side,
                websocket_odd=ws_odd,
                betslip_best_odd=None,
                betslip_limit=None,
                difference_pct=None,
                status=f"ERROR: {str(e)[:30]}",
                reversal_direction=direction
            )
    
    async def _find_and_click_game(self, home_team: str, away_team: str) -> bool:
        """Encontra e clica num jogo específico na página."""
        page = self.scraper._page
        
        try:
            body_text = await page.inner_text("body")
            
            home_words = home_team.split()[:2]
            away_words = away_team.split()[:2]
            
            home_found = any(word in body_text for word in home_words if len(word) > 3)
            away_found = any(word in body_text for word in away_words if len(word) > 3)
            
            if not home_found and not away_found:
                print(f"    Times não encontrados na página")
                return False
            
            print(f"    Times encontrados: home={home_found}, away={away_found}")
            
            for team_name in [home_team, away_team]:
                for name_variant in [team_name, team_name.split()[0] if team_name else ""]:
                    if not name_variant or len(name_variant) < 3:
                        continue
                    
                    try:
                        selectors = [
                            f"a:has-text('{name_variant}')",
                            f"div:has-text('{name_variant}')",
                            f"span:has-text('{name_variant}')",
                        ]
                        
                        for selector in selectors:
                            try:
                                elements = await page.query_selector_all(selector)
                                
                                for el in elements[:5]:
                                    try:
                                        el_text = await el.inner_text()
                                        
                                        if len(el_text) < 200:
                                            await el.scroll_into_view_if_needed()
                                            await el.click()
                                            await page.wait_for_timeout(1500)
                                            
                                            new_text = await page.inner_text("body")
                                            if "Asian Handicap" in new_text or "Over/Under" in new_text:
                                                print(f"    Clicou no jogo '{name_variant}'")
                                                return True
                                    except:
                                        continue
                            except:
                                continue
                    except:
                        continue
            
            if ("Asian Handicap" in body_text or "Over/Under" in body_text) and home_found:
                print(f"    Jogo já está visível na página")
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Erro ao buscar jogo: {e}")
            return False
    
    async def _expand_all_lines(self):
        """Expande todas as linhas clicando em 'Show all lines'."""
        page = self.scraper._page
        
        total_clicked = 0
        
        try:
            for attempt in range(3):
                selectors = [
                    "text='Show all lines'",
                    "text='Mostrar todas as linhas'",
                    "text='Show all'",
                    "button:has-text('Show all')",
                    "button:has-text('Mostrar')",
                    "[role='button']:has-text('Show all')",
                ]
                
                buttons_clicked = 0
                
                for selector in selectors:
                    try:
                        buttons = await page.query_selector_all(selector)
                        for btn in buttons:
                            try:
                                if await btn.is_visible():
                                    await btn.scroll_into_view_if_needed()
                                    await btn.click()
                                    await page.wait_for_timeout(800)
                                    buttons_clicked += 1
                                    total_clicked += 1
                            except:
                                continue
                    except:
                        continue
                
                if buttons_clicked > 0:
                    await page.wait_for_timeout(500)
                else:
                    break
            
            if total_clicked > 0:
                print(f"    Expandiu {total_clicked} seções")
            else:
                print(f"    Nenhum botão 'Show all' encontrado (pode já estar expandido)")
                    
        except Exception as e:
            logger.debug(f"Erro ao expandir linhas: {e}")
    
    async def _click_specific_odd(self, line: str, side: str, market_type: str = "AH") -> bool:
        """Clica numa odd específica para abrir o betslip."""
        page = self.scraper._page
        
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
                line_variants.append(line.replace(".", ","))
                line_variants.append(line)
                if line_float > 0 and not line.startswith("+"):
                    line_variants.append("+" + line.replace(".", ","))
            
            print(f"    Procurando linha: {line_variants}")
            
            body_text = await page.inner_text("body")
            
            # Determina labels baseado no tipo de mercado
            if market_type == "OU":
                side_labels = {"over": "Over", "under": "Under"}
            else:
                side_labels = {"home": "Home", "away": "Away"}
            
            target_odd = None
            for variant in line_variants:
                for text_line in body_text.split('\n'):
                    side_label = side_labels.get(side, side.capitalize())
                    opposite_label = "Away" if side_label == "Home" else "Home"
                    if market_type == "OU":
                        opposite_label = "Under" if side_label == "Over" else "Over"
                    
                    if variant in text_line and side_label in text_line:
                        odds_found = re.findall(r'(\d+[.,]\d{2,3})', text_line)
                        if len(odds_found) >= 2:
                            side_idx = text_line.find(side_label)
                            opposite_idx = text_line.find(opposite_label)
                            
                            target_odd_candidate = None
                            for odd in odds_found:
                                odd_idx = text_line.find(odd)
                                if side_idx < odd_idx < opposite_idx:
                                    target_odd_candidate = odd
                                    break
                                elif odd_idx > side_idx and opposite_idx < side_idx:
                                    target_odd_candidate = odd
                                    break
                            
                            if target_odd_candidate:
                                target_odd = target_odd_candidate
                                print(f"    Odd alvo identificada: {target_odd}")
                                break
                if target_odd:
                    break
            
            if not target_odd:
                available_lines = set()
                for l in body_text.split('\n'):
                    l_stripped = l.strip()
                    if re.match(r'^[+-]\d+([.,]\d{1,2})?$', l_stripped):
                        available_lines.add(l_stripped)
                    elif re.match(r'^0([.,]0+)?$', l_stripped):
                        available_lines.add("0")
                
                def line_sort_key(x):
                    try:
                        return float(x.replace(',', '.').replace('+', ''))
                    except:
                        return 0
                sorted_lines = sorted(available_lines, key=line_sort_key)
                
                print(f"    Linha não encontrada nesta tentativa")
                print(f"    Buscando: {line_variants}")
                print(f"    Linhas disponíveis: {sorted_lines[:15]}")
                
                return False
            
            # Busca e clica no elemento
            elements = await page.query_selector_all('span')
            
            candidates = []
            for el in elements:
                try:
                    el_text = await el.inner_text()
                    if el_text.strip() == target_odd:
                        parent = await el.evaluate_handle("el => el.parentElement.parentElement.parentElement")
                        parent_text = await parent.evaluate("el => el.innerText || ''")
                        
                        for variant in line_variants:
                            if variant in parent_text:
                                box = await el.bounding_box()
                                if box:
                                    candidates.append((el, box['x']))
                                break
                except:
                    continue
            
            if candidates:
                candidates.sort(key=lambda x: x[1])
                
                if side in ['home', 'over'] and len(candidates) >= 1:
                    el = candidates[0][0]
                elif side in ['away', 'under'] and len(candidates) >= 2:
                    el = candidates[1][0]
                elif len(candidates) == 1:
                    el = candidates[0][0]
                else:
                    el = None
                
                if el:
                    print(f"    Encontrou {len(candidates)} candidatos, clicando...")
                    try:
                        await el.scroll_into_view_if_needed()
                        await page.wait_for_timeout(300)
                        
                        parent = await el.evaluate_handle("el => el.parentElement")
                        await parent.click()
                        await page.wait_for_timeout(1500)
                        return True
                    except:
                        try:
                            await el.click()
                            await page.wait_for_timeout(1500)
                            return True
                        except:
                            pass
            
            return False
            
        except Exception as e:
            logger.error(f"Erro ao clicar: {e}")
            return False
            
    def _print_results(self):
        """Imprime resultados."""
        print("\n\n" + "=" * 70)
        print("RESULTADOS DA AUDITORIA H3B")
        print("=" * 70)
        
        counts = {"IDENTICAL": 0, "OK": 0, "MINOR_DIFF": 0, "MAJOR_DIFF": 0, "LINE_NOT_AVAILABLE": 0}
        diffs = []
        errors = 0
        
        by_direction = {"up": [], "down": []}
        
        for r in self.audit_results:
            print(f"\n{r.match_info}")
            print(f"  {r.market_type} {r.line} {r.side} (Reversão {r.reversal_direction.upper()})")
            print(f"  WebSocket:     {r.websocket_odd:.3f}")
            
            if r.betslip_best_odd:
                print(f"  Betslip Best:  {r.betslip_best_odd:.3f}")
                print(f"  Limite:        ${r.betslip_limit:,.0f}")
                print(f"  Diferença:     {r.difference_pct:+.2f}%")
                diffs.append(abs(r.difference_pct))
                
            emoji = {
                "IDENTICAL": "✅", "OK": "✅", 
                "MINOR_DIFF": "⚠️", "MAJOR_DIFF": "❌",
                "LINE_NOT_AVAILABLE": "📉"
            }.get(r.status, "❓")
            print(f"  Status:        {emoji} {r.status}")
            
            if r.status in counts:
                counts[r.status] += 1
            else:
                errors += 1
            
            by_direction[r.reversal_direction].append(r)
        
        print("\n" + "=" * 70)
        print("RESUMO")
        print("=" * 70)
        total = len(self.audit_results)
        print(f"  ✅ Idênticas/OK (diff < 0.5%): {counts['IDENTICAL'] + counts['OK']}/{total}")
        print(f"  ⚠️ Diff pequena (0.5-2%):      {counts['MINOR_DIFF']}/{total}")
        print(f"  ❌ Diff grande (>2%):          {counts['MAJOR_DIFF']}/{total}")
        print(f"  📉 Linha indisponível:         {counts['LINE_NOT_AVAILABLE']}/{total}")
        print(f"  ❓ Outros erros:               {errors}/{total}")
        
        if diffs:
            print(f"\n  Diferença média: {sum(diffs)/len(diffs):.3f}%")
            print(f"  Diferença máxima: {max(diffs):.3f}%")
        
        print("\n" + "=" * 70)
        print("TAXA DE OPORTUNIDADES REAIS - H3B")
        print("=" * 70)
        
        real_opportunities = counts['IDENTICAL'] + counts['OK'] + counts['MINOR_DIFF'] + counts['MAJOR_DIFF']
        false_positives = counts['LINE_NOT_AVAILABLE'] + errors
        
        if total > 0:
            real_rate = real_opportunities / total * 100
            false_rate = false_positives / total * 100
            
            print(f"\n  📊 TAXA DE SUCESSO: {real_rate:.1f}%")
            print(f"     ({real_opportunities} de {total} eventos H3B têm odds reais)")
            print(f"\n  📉 FALSOS POSITIVOS: {false_rate:.1f}%")
            print(f"     ({false_positives} de {total} eventos não existem de fato)")
            
            print(f"\n  💡 PARA ESTIMAR OPORTUNIDADES REAIS:")
            print(f"     Total H3B detectados × {real_rate:.1f}% = oportunidades apostáveis")
            
            if diffs:
                print(f"\n  📈 QUALIDADE DAS ODDS (quando existem):")
                print(f"     Diferença média WebSocket vs Betslip: {sum(diffs)/len(diffs):.3f}%")
                
                accurate = counts['IDENTICAL'] + counts['OK']
                if real_opportunities > 0:
                    accuracy_rate = accurate / real_opportunities * 100
                    print(f"     Taxa de precisão (diff < 0.5%): {accuracy_rate:.1f}%")
        
        # Breakdown por direção
        print("\n" + "=" * 70)
        print("BREAKDOWN POR DIREÇÃO DA REVERSÃO")
        print("=" * 70)
        
        for direction, results in by_direction.items():
            if not results:
                continue
            
            dir_total = len(results)
            dir_real = sum(1 for r in results if r.status in ["IDENTICAL", "OK", "MINOR_DIFF", "MAJOR_DIFF"])
            dir_rate = dir_real / dir_total * 100 if dir_total > 0 else 0
            
            print(f"\n  REVERSÃO {direction.upper()}:")
            print(f"    Total: {dir_total}")
            print(f"    Taxa de sucesso: {dir_rate:.1f}%")


async def main():
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    
    # Audita eventos H3B com reversão UP (odd subiu = melhorou)
    auditor = H3BAuditor(num_audits=50, direction_filter="up")
    await auditor.run_audit()


if __name__ == "__main__":
    asyncio.run(main())
