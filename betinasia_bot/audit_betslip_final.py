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
            # Navega para página do jogo
            game_url = f"https://black.betinasia.com/sportsbook/football/{event_id}"
            print(f"    Navegando para: {game_url}")
            await self.scraper._page.goto(game_url)
            await self.scraper._page.wait_for_load_state("domcontentloaded")
            await self.scraper._page.wait_for_timeout(2000)
            
            # IMPORTANTE: Expande "Show all lines" para mostrar todas as linhas AH
            print(f"    Expandindo linhas...")
            await self._expand_all_lines()
            await self.scraper._page.wait_for_timeout(1500)
            
            # Clica na odd para abrir betslip
            print(f"    Clicando na odd {line} {side}...")
            
            # Formata a linha para o padrão do site (usa vírgula)
            line_display = line.replace(".", ",")
            
            # Tenta clicar com retry (atualizações podem causar falhas)
            clicked = False
            for attempt in range(3):
                clicked = await self._click_specific_odd(line_display, side)
                if clicked:
                    break
                print(f"    Tentativa {attempt + 1} falhou, aguardando...")
                await self.scraper._page.wait_for_timeout(1000)
            
            if not clicked:
                print(f"    ERRO: Não conseguiu clicar na odd")
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
                    status="CLICK_FAILED"
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
        
        Usa a mesma abordagem do scraper principal que funciona bem:
        1. Encontra spans com texto da odd
        2. Verifica contexto (linha correta)
        3. Clica no elemento pai
        """
        page = self.scraper._page
        
        try:
            # Formata a linha para diferentes possibilidades
            line_float = float(line.replace(",", "."))
            
            # Possíveis formatos da linha no site
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
            
            # Captura texto da página para análise
            body_text = await page.inner_text("body")
            
            # Encontra a odd alvo analisando o texto
            target_odd = None
            for variant in line_variants:
                for text_line in body_text.split('\n'):
                    # Procura linha com formato: VARIANT ... Home ... ODD ... Away ... ODD
                    if variant in text_line and 'Home' in text_line and 'Away' in text_line:
                        # Extrai odds
                        odds_found = re.findall(r'(\d+[.,]\d{2,3})', text_line)
                        if len(odds_found) >= 2:
                            home_idx = text_line.find('Home')
                            away_idx = text_line.find('Away')
                            
                            home_odd = None
                            away_odd = None
                            
                            for odd in odds_found:
                                odd_idx = text_line.find(odd)
                                if home_idx < odd_idx < away_idx and not home_odd:
                                    home_odd = odd
                                elif odd_idx > away_idx and not away_odd:
                                    away_odd = odd
                            
                            target_odd = home_odd if side == 'home' else away_odd
                            if target_odd:
                                print(f"    Odd alvo identificada: {target_odd}")
                                break
                if target_odd:
                    break
            
            if not target_odd:
                print(f"    Não encontrou odd alvo no texto da página")
                
                # Debug avançado: mostra o que tem na página
                print(f"    === DEBUG ===")
                
                # Verifica se tem Asian Handicap
                if "Asian Handicap" in body_text:
                    print(f"    Seção 'Asian Handicap' encontrada")
                    # Extrai seção AH
                    ah_start = body_text.find("Asian Handicap")
                    ah_section = body_text[ah_start:ah_start+2000]
                    print(f"    Primeiras linhas AH:")
                    for i, l in enumerate(ah_section.split('\n')[:15]):
                        if l.strip():
                            print(f"      {i}: {l.strip()[:60]}")
                else:
                    print(f"    AVISO: Seção 'Asian Handicap' NÃO encontrada!")
                
                # Mostra linhas que parecem ser handicaps (números negativos)
                print(f"\n    Linhas com handicaps negativos:")
                for l in body_text.split('\n'):
                    l_stripped = l.strip()
                    if re.match(r'^-\d', l_stripped) and len(l_stripped) < 100:
                        print(f"      {l_stripped[:70]}")
                
                # Mostra linhas que contêm as variantes
                print(f"\n    Linhas contendo variantes buscadas:")
                for variant in line_variants:
                    matches = [l.strip() for l in body_text.split('\n') if variant in l][:2]
                    for m in matches:
                        print(f"      [{variant}]: {m[:70]}...")
                
                print(f"    === FIM DEBUG ===")
                return False
            
            # === ESTRATÉGIA 1: Query selector direto + clique no pai (como no scraper) ===
            print(f"    Estratégia 1: buscando span com texto '{target_odd}'")
            
            # Busca todos os spans
            elements = await page.query_selector_all('span')
            
            # Filtra elementos com o texto da odd e contexto correto
            candidates = []
            for el in elements:
                try:
                    el_text = await el.inner_text()
                    if el_text.strip() == target_odd:
                        # Verifica contexto - precisa ter a variante da linha no pai
                        parent = await el.evaluate_handle("el => el.parentElement.parentElement.parentElement")
                        parent_text = await parent.evaluate("el => el.innerText || ''")
                        
                        for variant in line_variants:
                            if variant in parent_text and 'Home' in parent_text and 'Away' in parent_text:
                                # Determina se é home ou away pela posição
                                box = await el.bounding_box()
                                if box:
                                    candidates.append((el, box['x']))
                                break
                except:
                    continue
            
            # Ordena por posição X (home = esquerda, away = direita)
            if candidates:
                candidates.sort(key=lambda x: x[1])
                
                # Home é o primeiro (mais à esquerda), Away é o segundo
                if side == 'home' and len(candidates) >= 1:
                    el = candidates[0][0]
                elif side == 'away' and len(candidates) >= 2:
                    el = candidates[1][0]
                elif len(candidates) == 1:
                    el = candidates[0][0]  # Só tem um, usa esse
                else:
                    el = None
                
                if el:
                    print(f"    Encontrou {len(candidates)} candidatos, clicando...")
                    try:
                        await el.scroll_into_view_if_needed()
                        await page.wait_for_timeout(300)
                        
                        # Clica no elemento pai (mais confiável)
                        parent = await el.evaluate_handle("el => el.parentElement")
                        await parent.click()
                        await page.wait_for_timeout(1500)
                        return True
                    except Exception as e1:
                        print(f"    Clique no pai falhou: {e1}")
                        try:
                            # Fallback: clica direto
                            await el.click()
                            await page.wait_for_timeout(1500)
                            return True
                        except Exception as e2:
                            print(f"    Clique direto falhou: {e2}")
                            # Fallback: JavaScript
                            try:
                                await el.evaluate("el => el.parentElement.click()")
                                await page.wait_for_timeout(1500)
                                return True
                            except:
                                pass
            
            # === ESTRATÉGIA 2: Locator do Playwright com texto exato ===
            print(f"    Estratégia 2: locator com texto exato")
            
            try:
                # Busca por texto exato da odd
                locator = page.locator(f"span:text-is('{target_odd}')")
                count = await locator.count()
                
                if count > 0:
                    print(f"    Encontrou {count} elementos")
                    
                    for i in range(min(count, 5)):
                        try:
                            el = locator.nth(i)
                            await el.scroll_into_view_if_needed()
                            await page.wait_for_timeout(200)
                            
                            # Verifica se está no contexto certo
                            parent_text = await el.evaluate("el => el.parentElement.parentElement.parentElement.innerText || ''")
                            
                            is_correct_context = False
                            for variant in line_variants:
                                if variant in parent_text:
                                    is_correct_context = True
                                    break
                            
                            if is_correct_context:
                                print(f"    Clicando no elemento {i}...")
                                await el.evaluate("el => el.parentElement.click()")
                                await page.wait_for_timeout(1500)
                                return True
                        except:
                            continue
            except Exception as e:
                logger.debug(f"Estratégia 2 falhou: {e}")
            
            # === ESTRATÉGIA 3: JavaScript força bruta ===
            print(f"    Estratégia 3: JavaScript força bruta")
            
            clicked = await page.evaluate("""
                (params) => {
                    const targetOdd = params.targetOdd;
                    
                    // Encontra todos os spans
                    const spans = document.querySelectorAll('span');
                    
                    for (const span of spans) {
                        const text = (span.innerText || '').trim();
                        if (text === targetOdd) {
                            try {
                                span.scrollIntoView({block: 'center'});
                                // Tenta clicar no pai
                                span.parentElement.click();
                                return true;
                            } catch (e) {
                                try {
                                    span.click();
                                    return true;
                                } catch (e2) {}
                            }
                        }
                    }
                    
                    return false;
                }
            """, {"targetOdd": target_odd})
            
            if clicked:
                await page.wait_for_timeout(1500)
                return True
            
            print(f"    Nenhuma estratégia funcionou")
            return False
            
        except Exception as e:
            logger.error(f"Erro ao clicar: {e}")
            return False
            
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
            print(f"  WebSocket:     {r.websocket_odd:.3f}")
            
            if r.betslip_best_odd:
                print(f"  Betslip Best:  {r.betslip_best_odd:.3f}")
                print(f"  Limite:        ${r.betslip_limit:,.0f}")
                print(f"  Diferença:     {r.difference_pct:+.2f}%")
                diffs.append(abs(r.difference_pct))
                
            emoji = {
                "IDENTICAL": "✅", "OK": "✅", 
                "MINOR_DIFF": "⚠️", "MAJOR_DIFF": "❌"
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
        print(f"  ❓ Erros:                      {errors}/{total}")
        
        if diffs:
            print(f"\n  Diferença média: {sum(diffs)/len(diffs):.3f}%")
            print(f"  Diferença máxima: {max(diffs):.3f}%")
        
        print("\n" + "=" * 70)
        print("CONCLUSÃO")
        print("=" * 70)
        
        found = counts['IDENTICAL'] + counts['OK'] + counts['MINOR_DIFF'] + counts['MAJOR_DIFF']
        if found == 0:
            print("❓ Nenhuma auditoria bem sucedida - verificar seletores")
        elif counts['MAJOR_DIFF'] == 0 and counts['MINOR_DIFF'] == 0:
            print("✅ ODDS DO WEBSOCKET = BEST ODDS DO BETSLIP!")
            print("   Os dados coletados são confiáveis para apostas.")
        elif counts['MAJOR_DIFF'] == 0:
            print("✅ Odds muito próximas (diferenças < 2%)")
            print("   Provavelmente variação normal de mercado.")
        else:
            pct_major = counts['MAJOR_DIFF'] / found * 100
            print(f"⚠️ {pct_major:.0f}% com diferença > 2%")
            print("   Verificar se WebSocket captura best odds corretamente.")


async def main():
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    
    auditor = BetslipAuditor(num_audits=5)
    await auditor.run_audit()


if __name__ == "__main__":
    asyncio.run(main())
