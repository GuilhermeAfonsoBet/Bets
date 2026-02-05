# -*- coding: utf-8 -*-
"""
Auditoria Final: WebSocket vs Best Odd do Betslip

Compara as odds coletadas via WebSocket com a BEST ODD real
exibida no painel do betslip.

Baseado na estrutura real do BetinAsia:
- "Todos Os Agentes De Apostas" → coluna MELHOR = best odd
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
from storage.database import Database
from storage.models_hypothesis import H6CorrelationLagEvent
from storage.models import Match
from sqlalchemy import select, and_


@dataclass
class AuditResult:
    """Resultado de uma auditoria."""
    timestamp: datetime
    match_info: str
    event_id: str
    line: str
    side: str
    websocket_odd: float
    betslip_best_odd: Optional[float]
    betslip_limit: Optional[float]
    difference_pct: Optional[float]
    status: str
    betslip_data: Optional[BetslipData] = None
    db_match_id: Optional[int] = None  # ID no banco de dados para atualizar


class BetslipAuditor:
    """Auditor que compara WebSocket vs Betslip Best Odd."""
    
    FOOTBALL_URL = "https://black.betinasia.com/sportsbook/football"
    
    def __init__(self, num_audits: int = 5):
        self.scraper: Optional[BetinAsiaScraper] = None
        self.extractor: Optional[BetslipExtractor] = None
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
        
        self.extractor = BetslipExtractor(self.scraper._page)
        self.scraper._page.on('websocket', self._on_websocket)
        
        print("Auditor iniciado e logado")
        
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
        print("AUDITORIA: WEBSOCKET vs BEST ODD DO BETSLIP")
        print("=" * 70)
        print(f"""
Este script compara:
- Odd recebida via WebSocket (o que o coletor salva)
- Best Odd do Betslip ("Todos Os Agentes De Apostas" → MELHOR)

Processo:
1. Coleta odds via WebSocket
2. Quando detecta H6, navega para o jogo
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
                await self.scraper._page.wait_for_timeout(8000)  # Espera WebSocket carregar
                
                h6_events = await self._find_h6_events(audited)
                
                if h6_events:
                    print(f"\n    → {len(h6_events)} H6 novos para auditar neste ciclo")
                
                for h6 in h6_events:
                    if len(self.audit_results) >= self.num_audits:
                        break
                    result = await self._audit_event(h6)
                    self.audit_results.append(result)
                    audited.add(h6['audit_key'])  # Chave única: event_id + linha + lado
                
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
        skipped_already_audited = 0
        
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
                                            # Chave única: event_id + linha + lado
                                            audit_key = f"{event_id}|{h6.lagged_line}|{h6.lagged_side}"
                                            if audit_key not in already_audited:
                                                info = events.get(event_id, {})
                                                h6_list.append({
                                                    'event_id': event_id,
                                                    'audit_key': audit_key,
                                                    'match_info': f"{info.get('home', '?')} vs {info.get('away', '?')}",
                                                    'line': str(h6.lagged_line),
                                                    'side': h6.lagged_side,
                                                    'websocket_odd': h6.lagged_current_odd,
                                                })
                                            else:
                                                skipped_already_audited += 1
            except:
                continue
        
        # Debug: mostra estatísticas
        if skipped_already_audited > 0:
            print(f"\n    (Pulou {skipped_already_audited} H6 de jogos já auditados)")
                
        return h6_list
    
    async def _audit_event(self, h6: dict) -> AuditResult:
        """Audita um evento abrindo o betslip."""
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
            page = self.scraper._page
            
            # Extrai nomes dos times
            teams = match_info.split(' vs ')
            home_team = teams[0].strip() if len(teams) > 0 else ""
            away_team = teams[1].strip() if len(teams) > 1 else ""
            
            print(f"    Buscando jogo: '{home_team}' vs '{away_team}'")
            
            # NOVA ABORDAGEM: Não navega - busca o jogo na página atual
            # O WebSocket já está enviando dados desta página
            
            # Primeiro, tenta encontrar e clicar no jogo específico
            game_found = await self._find_and_click_game(home_team, away_team)
            
            if not game_found:
                print(f"    Jogo não encontrado na página atual, tentando navegar...")
                # Fallback: tenta navegar para página geral de futebol e procurar
                await page.goto("https://black.betinasia.com/sportsbook/football")
                await page.wait_for_load_state("domcontentloaded")
                await page.wait_for_timeout(3000)
                
                # Expande e procura novamente
                await self._expand_all_lines()
                await page.wait_for_timeout(1500)
                
                game_found = await self._find_and_click_game(home_team, away_team)
            
            if not game_found:
                print(f"    ERRO: Jogo não encontrado")
                return AuditResult(
                    timestamp=datetime.now(timezone.utc),
                    match_info=match_info,
                    event_id=event_id,
                    line=line,
                    side=side,
                    websocket_odd=ws_odd,
                    betslip_best_odd=None,
                    betslip_limit=None,
                    difference_pct=None,
                    status="GAME_NOT_FOUND"
                )
            
            # Aguarda carregar os detalhes do jogo
            await page.wait_for_timeout(2000)
            
            # Expande linhas do jogo
            print(f"    Expandindo linhas...")
            await self._expand_all_lines()
            await page.wait_for_timeout(1500)
            
            # Clica na odd para abrir betslip
            print(f"    Clicando na odd {line} {side}...")
            
            # Formata a linha para o padrão do site (usa vírgula)
            line_display = line.replace(".", ",")
            
            # Tenta clicar com retry robusto (refresh + re-expand entre tentativas)
            click_result = None
            max_attempts = 5  # Mais tentativas para dar tempo da linha aparecer
            
            for attempt in range(max_attempts):
                click_result = await self._click_specific_odd(line_display, side)
                
                if click_result == True:
                    break
                
                # Se não encontrou, pode ser delay da plataforma
                remaining = max_attempts - attempt - 1
                if remaining > 0:
                    print(f"    Tentativa {attempt + 1}/{max_attempts} falhou, aguardando e recarregando...")
                    
                    # Aguarda um pouco (pode ser delay da plataforma)
                    await page.wait_for_timeout(2000)
                    
                    # A cada 2 tentativas, faz refresh da página
                    if attempt > 0 and attempt % 2 == 0:
                        print(f"    Recarregando página...")
                        await page.reload()
                        await page.wait_for_load_state("domcontentloaded")
                        await page.wait_for_timeout(2000)
                        
                        # Re-encontra o jogo
                        game_found = await self._find_and_click_game(home_team, away_team)
                        if not game_found:
                            print(f"    Jogo não encontrado após reload")
                            continue
                        
                        await page.wait_for_timeout(1500)
                    
                    # Re-expande as linhas
                    print(f"    Re-expandindo linhas...")
                    await self._expand_all_lines()
                    await page.wait_for_timeout(1000)
            
            if click_result != True:
                # Após todas as tentativas, confirma que a linha não existe
                print(f"    LINHA CONFIRMADA COMO NÃO DISPONÍVEL após {max_attempts} tentativas")
                return AuditResult(
                    timestamp=datetime.now(timezone.utc),
                    match_info=match_info,
                    event_id=event_id,
                    line=line,
                    side=side,
                    websocket_odd=ws_odd,
                    betslip_best_odd=None,
                    betslip_limit=None,
                    difference_pct=None,
                    status="LINE_NOT_AVAILABLE"
                )
            
            await self.scraper._page.wait_for_timeout(2000)
            
            # Extrai dados do betslip
            betslip_data = await self.extractor.extract_best_odd()
            
            if not betslip_data:
                print(f"    ERRO: Não conseguiu extrair dados do betslip")
                return AuditResult(
                    timestamp=datetime.now(timezone.utc),
                    match_info=match_info,
                    event_id=event_id,
                    line=line,
                    side=side,
                    websocket_odd=ws_odd,
                    betslip_best_odd=None,
                    betslip_limit=None,
                    difference_pct=None,
                    status="EXTRACT_FAILED"
                )
            
            best_odd = betslip_data.best_odd
            best_limit = betslip_data.best_limit
            
            # Calcula diferença
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
                line=line,
                side=side,
                websocket_odd=ws_odd,
                betslip_best_odd=best_odd,
                betslip_limit=best_limit,
                difference_pct=diff_pct,
                status=status,
                betslip_data=betslip_data
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
                betslip_best_odd=None,
                betslip_limit=None,
                difference_pct=None,
                status=f"ERROR: {str(e)[:30]}"
            )
    
    async def _find_and_click_game(self, home_team: str, away_team: str) -> bool:
        """
        Encontra e clica num jogo específico na página.
        
        Args:
            home_team: Nome do time da casa
            away_team: Nome do time visitante
            
        Returns:
            True se encontrou e clicou no jogo
        """
        page = self.scraper._page
        
        try:
            body_text = await page.inner_text("body")
            
            # Verifica se o jogo está na página
            # Procura pelos nomes dos times (podem estar parciais)
            home_words = home_team.split()[:2]  # Primeiras 2 palavras
            away_words = away_team.split()[:2]
            
            home_found = any(word in body_text for word in home_words if len(word) > 3)
            away_found = any(word in body_text for word in away_words if len(word) > 3)
            
            if not home_found and not away_found:
                print(f"    Times não encontrados na página")
                return False
            
            print(f"    Times encontrados: home={home_found}, away={away_found}")
            
            # Tenta encontrar um elemento com o nome do time e clicar
            for team_name in [home_team, away_team]:
                # Tenta com nome completo primeiro
                for name_variant in [team_name, team_name.split()[0] if team_name else ""]:
                    if not name_variant or len(name_variant) < 3:
                        continue
                    
                    try:
                        # Busca link ou div clicável com o nome do time
                        selectors = [
                            f"a:has-text('{name_variant}')",
                            f"div:has-text('{name_variant}')",
                            f"span:has-text('{name_variant}')",
                        ]
                        
                        for selector in selectors:
                            try:
                                elements = await page.query_selector_all(selector)
                                
                                for el in elements[:5]:  # Tenta nos primeiros 5
                                    try:
                                        el_text = await el.inner_text()
                                        
                                        # Verifica se é o contexto certo (jogo, não menu)
                                        if len(el_text) < 200:  # Não é um container grande
                                            # Clica para abrir detalhes do jogo
                                            await el.scroll_into_view_if_needed()
                                            await el.click()
                                            await page.wait_for_timeout(1500)
                                            
                                            # Verifica se abriu detalhes do jogo
                                            new_text = await page.inner_text("body")
                                            if "Asian Handicap" in new_text:
                                                print(f"    Clicou no jogo '{name_variant}'")
                                                return True
                                    except:
                                        continue
                            except:
                                continue
                    except:
                        continue
            
            # Se não precisou clicar mas o jogo está visível, retorna True
            if "Asian Handicap" in body_text and home_found:
                print(f"    Jogo já está visível na página")
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Erro ao buscar jogo: {e}")
            return False
    
    async def _expand_all_lines(self):
        """
        Expande todas as linhas AH clicando em 'Show all lines'.
        Essencial para ver linhas como -13.0 que ficam escondidas por padrão.
        """
        page = self.scraper._page
        
        total_clicked = 0
        
        try:
            # Tenta clicar em todos os botões "Show all lines" várias vezes
            for attempt in range(3):
                # Diferentes seletores para o botão
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
                    break  # Não encontrou mais botões
            
            if total_clicked > 0:
                print(f"    Expandiu {total_clicked} seções")
            else:
                print(f"    Nenhum botão 'Show all' encontrado (pode já estar expandido)")
                    
        except Exception as e:
            logger.debug(f"Erro ao expandir linhas: {e}")
    
    async def _click_specific_odd(self, line: str, side: str) -> bool:
        """
        Clica numa odd específica para abrir o betslip.
        
        ESTRATÉGIA ROBUSTA (baseada na estrutura DOM real):
        1. Encontra a seção Asian Handicap
        2. Busca a LINHA específica pelo valor do handicap
        3. Dentro da linha, clica no lado correto por POSIÇÃO (Home/Away)
        
        NÃO busca pelo valor da odd (que pode mudar entre WebSocket e DOM).
        """
        page = self.scraper._page
        
        try:
            line_float = float(line.replace(",", "."))
            
            # Gera variantes da linha (diferentes formatos possíveis)
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
            
            print(f"    Procurando linha: {line_variants}")
            
            # Nomes de seção (PT e EN)
            section_names = ["Asian Handicap", "Handicap Asiático", "Handicap"]
            
            # === ESTRATÉGIA PRINCIPAL: JavaScript robusto baseado na estrutura DOM ===
            # Estrutura: LINHA | Home | ODD_HOME | Away | ODD_AWAY
            
            clicked = await page.evaluate("""
                (params) => {
                    const lineVariants = params.lineVariants;
                    const side = params.side;
                    const sectionNames = params.sectionNames;
                    
                    // Função para normalizar texto de linha
                    function normalizeLineText(text) {
                        return text.trim().replace(/\\s+/g, '').replace('.', ',');
                    }
                    
                    // Função para verificar se texto contém algum dos nomes de seção
                    function matchesSection(text) {
                        for (const name of sectionNames) {
                            if (text.includes(name)) return true;
                        }
                        return false;
                    }
                    
                    // Encontra a seção Asian Handicap (PT ou EN)
                    let sectionContainer = null;
                    const headers = document.querySelectorAll('div, span, h3, h4');
                    
                    for (const h of headers) {
                        const text = (h.innerText || '').trim();
                        if (matchesSection(text) || text.includes('Handicap') || text.includes('Asian')) {
                            let parent = h.parentElement;
                            for (let i = 0; i < 10 && parent; i++) {
                                const parentText = parent.innerText || '';
                                if (parentText.includes('Home') && parentText.includes('Away')) {
                                    sectionContainer = parent;
                                    break;
                                }
                                parent = parent.parentElement;
                            }
                            if (sectionContainer) break;
                        }
                    }
                    
                    if (!sectionContainer) {
                        sectionContainer = document.body;
                    }
                    
                    // Encontra elementos que podem ser a linha de handicap
                    const allElements = sectionContainer.querySelectorAll('span, div');
                    
                    let foundLineText = null;
                    
                    for (const el of allElements) {
                        const elText = (el.innerText || '').trim();
                        
                        // Verifica se este elemento é a linha que procuramos
                        // IMPORTANTE: O texto deve ser EXATAMENTE a linha
                        let isLineMatch = false;
                        for (const variant of lineVariants) {
                            if (elText === variant || normalizeLineText(elText) === normalizeLineText(variant)) {
                                isLineMatch = true;
                                foundLineText = elText;
                                break;
                            }
                        }
                        
                        if (!isLineMatch) continue;
                        
                        // Verifica se este elemento é pequeno (só a linha, não um container grande)
                        if (elText.length > 10) continue;
                        
                        // Encontrou a linha! Busca o container ROW
                        let rowContainer = el.parentElement;
                        for (let i = 0; i < 6 && rowContainer; i++) {
                            const rowText = rowContainer.innerText || '';
                            
                            // Verifica se tem Home e Away e não é muito grande
                            if (rowText.includes('Home') && rowText.includes('Away') &&
                                rowText.split('\\n').length < 15) {
                                
                                // Verifica se contém a nossa linha
                                let hasOurLine = false;
                                for (const variant of lineVariants) {
                                    if (rowText.includes(variant)) {
                                        hasOurLine = true;
                                        break;
                                    }
                                }
                                if (!hasOurLine) {
                                    rowContainer = rowContainer.parentElement;
                                    continue;
                                }
                                
                                // Encontra os elementos clicáveis (odds)
                                const clickableElements = rowContainer.querySelectorAll('div, span');
                                const oddElements = [];
                                
                                for (const child of clickableElements) {
                                    const childText = (child.innerText || '').trim();
                                    
                                    // Verifica se é uma odd (formato X.XXX ou X,XXX)
                                    if (/^\\d+[.,]\\d{2,3}$/.test(childText) && childText.length < 10) {
                                        const rect = child.getBoundingClientRect();
                                        if (rect.width > 0 && rect.height > 0 && rect.width < 200) {
                                            oddElements.push({
                                                el: child,
                                                x: rect.x,
                                                text: childText
                                            });
                                        }
                                    }
                                }
                                
                                // Remove duplicatas
                                const uniqueOdds = [];
                                const seenKeys = new Set();
                                for (const odd of oddElements) {
                                    const key = Math.round(odd.x) + '|' + odd.text;
                                    if (!seenKeys.has(key)) {
                                        seenKeys.add(key);
                                        uniqueOdds.push(odd);
                                    }
                                }
                                
                                if (uniqueOdds.length >= 2) {
                                    // Ordena por posição X
                                    uniqueOdds.sort((a, b) => a.x - b.x);
                                    
                                    // Home = primeiro (esquerda), Away = segundo (direita)
                                    const targetIdx = (side === 'home') ? 0 : 1;
                                    const targetEl = uniqueOdds[targetIdx];
                                    
                                    if (targetEl) {
                                        targetEl.el.scrollIntoView({ behavior: 'instant', block: 'center' });
                                        
                                        try {
                                            const parent = targetEl.el.parentElement;
                                            if (parent) {
                                                parent.click();
                                                return { 
                                                    success: true, 
                                                    clickedOdd: targetEl.text, 
                                                    method: 'parent',
                                                    lineFound: foundLineText,
                                                    allOdds: uniqueOdds.map(o => o.text)
                                                };
                                            }
                                        } catch (e) {}
                                        
                                        try {
                                            targetEl.el.click();
                                            return { 
                                                success: true, 
                                                clickedOdd: targetEl.text, 
                                                method: 'direct',
                                                lineFound: foundLineText,
                                                allOdds: uniqueOdds.map(o => o.text)
                                            };
                                        } catch (e) {}
                                    }
                                }
                            }
                            rowContainer = rowContainer.parentElement;
                        }
                    }
                    
                    return { 
                        success: false, 
                        reason: 'LINE_NOT_FOUND',
                        foundLineText: foundLineText
                    };
                }
            """, {"lineVariants": line_variants, "side": side, "sectionNames": section_names})
            
            if clicked and clicked.get('success'):
                line_found = clicked.get('lineFound', '?')
                all_odds = clicked.get('allOdds', [])
                print(f"    Linha encontrada: {line_found}")
                print(f"    Odds na linha: {all_odds}")
                print(f"    Clicou na odd {clicked.get('clickedOdd')} ({clicked.get('method')})")
                await page.wait_for_timeout(1500)
                return True
            
            # === FALLBACK: Busca alternativa ===
            print(f"    Estratégia principal falhou, tentando fallback...")
            
            for variant in line_variants:
                try:
                    line_elements = await page.query_selector_all(f"span:text-is('{variant}'), div:text-is('{variant}')")
                    
                    for line_el in line_elements:
                        try:
                            for _ in range(5):
                                parent = await line_el.evaluate_handle("el => el.parentElement")
                                parent_text = await parent.evaluate("el => el.innerText || ''")
                                
                                if 'Home' in parent_text and 'Away' in parent_text:
                                    odd_spans = await parent.evaluate_handle(
                                        "el => Array.from(el.querySelectorAll('span, div')).filter(s => /^\\d+[.,]\\d{2,3}$/.test(s.innerText.trim()))"
                                    )
                                    
                                    count = await odd_spans.evaluate("arr => arr.length")
                                    
                                    if count >= 2:
                                        idx = 0 if side == 'home' else 1
                                        
                                        result = await odd_spans.evaluate(f"""
                                            (arr) => {{
                                                const el = arr[{idx}];
                                                if (el) {{
                                                    el.scrollIntoView({{ behavior: 'instant', block: 'center' }});
                                                    try {{
                                                        el.parentElement.click();
                                                        return {{ success: true, method: 'fallback-parent' }};
                                                    }} catch {{}}
                                                    try {{
                                                        el.click();
                                                        return {{ success: true, method: 'fallback-direct' }};
                                                    }} catch {{}}
                                                }}
                                                return {{ success: false }};
                                            }}
                                        """)
                                        
                                        if result and result.get('success'):
                                            print(f"    Fallback: clicou ({result.get('method')})")
                                            await page.wait_for_timeout(1500)
                                            return True
                                    break
                                
                                line_el = parent
                        except:
                            continue
                except:
                    continue
            
            # Debug: mostra linhas disponíveis
            body_text = await page.inner_text("body")
            available_lines = set()
            for l in body_text.split('\n'):
                l_stripped = l.strip()
                if re.match(r'^[+-]?\d+([.,]\d{1,2})?$', l_stripped) and len(l_stripped) < 8:
                    available_lines.add(l_stripped)
            
            def line_sort_key(x):
                try:
                    return float(x.replace(',', '.').replace('+', ''))
                except:
                    return 0
            sorted_lines = sorted(available_lines, key=line_sort_key)
            
            print(f"    Linha não encontrada")
            print(f"    Buscando: {line_variants}")
            print(f"    Linhas AH disponíveis: {sorted_lines[:20]}")
            
            return False
            
        except Exception as e:
            logger.error(f"Erro ao clicar: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def _print_results(self):
        """Imprime resultados."""
        print("\n\n" + "=" * 70)
        print("RESULTADOS DA AUDITORIA")
        print("=" * 70)
        
        counts = {"IDENTICAL": 0, "OK": 0, "MINOR_DIFF": 0, "MAJOR_DIFF": 0, "LINE_NOT_AVAILABLE": 0}
        diffs = []
        errors = 0
        
        for r in self.audit_results:
            print(f"\n{r.match_info}")
            print(f"  AH {r.line} {r.side}")
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
        print("TAXA DE OPORTUNIDADES REAIS")
        print("=" * 70)
        
        # Oportunidades reais = odds que existem de fato (qualquer status exceto falso positivo)
        real_opportunities = counts['IDENTICAL'] + counts['OK'] + counts['MINOR_DIFF'] + counts['MAJOR_DIFF']
        false_positives = counts['LINE_NOT_AVAILABLE'] + errors
        
        if total > 0:
            real_rate = real_opportunities / total * 100
            false_rate = false_positives / total * 100
            
            print(f"\n  📊 TAXA DE SUCESSO: {real_rate:.1f}%")
            print(f"     ({real_opportunities} de {total} eventos H6 têm odds reais)")
            print(f"\n  📉 FALSOS POSITIVOS: {false_rate:.1f}%")
            print(f"     ({false_positives} de {total} eventos não existem de fato)")
            
            print(f"\n  💡 PARA ESTIMAR OPORTUNIDADES REAIS:")
            print(f"     Total H6 detectados × {real_rate:.1f}% = oportunidades apostáveis")
            
            if diffs:
                print(f"\n  📈 QUALIDADE DAS ODDS (quando existem):")
                print(f"     Diferença média WebSocket vs Betslip: {sum(diffs)/len(diffs):.3f}%")
                
                # Odds idênticas ou muito próximas
                accurate = counts['IDENTICAL'] + counts['OK']
                if real_opportunities > 0:
                    accuracy_rate = accurate / real_opportunities * 100
                    print(f"     Taxa de precisão (diff < 0.5%): {accuracy_rate:.1f}%")
        
        print("\n" + "=" * 70)
        print("CONCLUSÃO")
        print("=" * 70)
        
        if real_opportunities == 0:
            print("❓ Nenhuma auditoria bem sucedida - verificar seletores")
        elif false_positives > real_opportunities:
            print(f"⚠️ MAIORIA SÃO FALSOS POSITIVOS ({false_rate:.0f}%)")
            print("   A maioria das odds detectadas via WebSocket não existe no site.")
            print("   Pode ser: linhas extremas, mercados removidos, ou delay de dados.")
        elif counts['MAJOR_DIFF'] == 0 and counts['MINOR_DIFF'] == 0:
            print("✅ ODDS DO WEBSOCKET = BEST ODDS DO BETSLIP!")
            print("   Os dados coletados são confiáveis para apostas.")
        elif counts['MAJOR_DIFF'] == 0:
            print("✅ Odds muito próximas (diferenças < 2%)")
            print("   Provavelmente variação normal de mercado.")
        else:
            pct_major = counts['MAJOR_DIFF'] / real_opportunities * 100 if real_opportunities > 0 else 0
            print(f"⚠️ {pct_major:.0f}% com diferença > 2%")
            print("   Verificar se WebSocket captura best odds corretamente.")


async def main():
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    
    auditor = BetslipAuditor(num_audits=50)
    await auditor.run_audit()


if __name__ == "__main__":
    asyncio.run(main())
