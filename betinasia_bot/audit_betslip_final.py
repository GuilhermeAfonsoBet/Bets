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
            await self.scraper._page.goto(game_url)
            await self.scraper._page.wait_for_load_state("domcontentloaded")
            await self.scraper._page.wait_for_timeout(3000)  # Espera conteúdo carregar
            
            # Clica na odd para abrir betslip
            print(f"    Clicando na odd {line} {side}...")
            
            # Formata a linha para o padrão do site (usa vírgula)
            line_display = line.replace(".", ",")
            
            # Clica na odd específica
            clicked = await self._click_specific_odd(line_display, side)
            
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
    
    async def _click_specific_odd(self, line: str, side: str) -> bool:
        """
        Clica numa odd específica para abrir o betslip.
        
        Estrutura do BetinAsia (dos prints):
        | -0,5  | Home | 2.394 | Away | 1.700 |
        
        Usa múltiplas estratégias de clique para robustez.
        """
        page = self.scraper._page
        
        try:
            # Formata a linha para diferentes possibilidades
            line_float = float(line.replace(",", "."))
            
            # Possíveis formatos da linha no site
            line_variants = []
            if line_float == int(line_float):
                int_val = int(line_float)
                # Formatos com sinal
                if int_val > 0:
                    line_variants.extend([f"+{int_val}", f"+{int_val},0", f"+{int_val}.0"])
                elif int_val < 0:
                    line_variants.extend([str(int_val), f"{int_val},0", f"{int_val}.0"])
                else:
                    line_variants.extend(["0", "+0", "0,0", "0.0", "+0.0", "+0,0"])
            else:
                # Decimal
                line_variants.append(line.replace(".", ","))
                line_variants.append(line)
                # Com sinal explícito se positivo
                if line_float > 0 and not line.startswith("+"):
                    line_variants.append("+" + line.replace(".", ","))
                    line_variants.append("+" + line)
            
            print(f"    Procurando linha: {line_variants}")
            
            # === ESTRATÉGIA 1: Locator do Playwright ===
            for variant in line_variants:
                try:
                    # Procura span/div que contenha a linha
                    line_locator = page.locator(f"span:has-text('{variant}'), div:has-text('{variant}')").first
                    
                    if await line_locator.count() > 0:
                        # Encontrou a linha - agora encontra o container com Home/Away
                        parent = line_locator.locator("xpath=ancestor::div[contains(., 'Home') and contains(., 'Away')]").first
                        
                        if await parent.count() > 0:
                            # Encontra todos os elementos com números decimais (odds)
                            odd_elements = await parent.locator("span").all()
                            
                            for i, el in enumerate(odd_elements):
                                text = (await el.inner_text()).strip()
                                
                                # Verifica se é uma odd (formato X.XX ou X,XX)
                                if re.match(r'^\d+[.,]\d{2,3}$', text):
                                    # Determina posição relativa (home = esquerda, away = direita)
                                    box = await el.bounding_box()
                                    parent_box = await parent.bounding_box()
                                    
                                    if box and parent_box:
                                        is_left = box['x'] < (parent_box['x'] + parent_box['width'] / 2)
                                        el_side = 'home' if is_left else 'away'
                                        
                                        if el_side == side:
                                            print(f"    Encontrou odd {text} para {side} (Estratégia 1)")
                                            await el.scroll_into_view_if_needed()
                                            await page.wait_for_timeout(200)
                                            
                                            # Tenta clicar
                                            try:
                                                await el.click()
                                                await page.wait_for_timeout(1000)
                                                return True
                                            except:
                                                # Tenta clicar no pai
                                                try:
                                                    parent_el = el.locator("xpath=..")
                                                    await parent_el.click()
                                                    await page.wait_for_timeout(1000)
                                                    return True
                                                except:
                                                    pass
                except Exception as e:
                    logger.debug(f"Estratégia 1 falhou para {variant}: {e}")
                    continue
            
            # === ESTRATÉGIA 2: JavaScript mais simples ===
            print(f"    Tentando Estratégia 2 (JavaScript)")
            
            clicked = await page.evaluate("""
                (params) => {
                    const lineVariants = params.lineVariants;
                    const side = params.side;
                    
                    // Pega todo o texto da página
                    const bodyText = document.body.innerText;
                    const lines = bodyText.split('\\n');
                    
                    // Procura linha que contém a variante + Home + odd + Away + odd
                    for (let i = 0; i < lines.length; i++) {
                        const line = lines[i].trim();
                        
                        // Verifica se é uma linha de handicap
                        let hasVariant = false;
                        for (const v of lineVariants) {
                            if (line.startsWith(v) || line.includes('\\t' + v) || line.includes(' ' + v)) {
                                hasVariant = true;
                                break;
                            }
                        }
                        
                        if (!hasVariant) continue;
                        if (!line.includes('Home') || !line.includes('Away')) continue;
                        
                        // Extrai odds: procura por padrão numérico
                        const numbers = line.match(/\\d+[.,]\\d{2,3}/g);
                        if (!numbers || numbers.length < 2) continue;
                        
                        // Em geral: primeiro número após Home é odd home, primeiro após Away é odd away
                        const homeIdx = line.indexOf('Home');
                        const awayIdx = line.indexOf('Away');
                        
                        let homeOdd = null, awayOdd = null;
                        
                        for (const num of numbers) {
                            const numIdx = line.indexOf(num);
                            if (numIdx > homeIdx && numIdx < awayIdx && !homeOdd) {
                                homeOdd = num;
                            } else if (numIdx > awayIdx && !awayOdd) {
                                awayOdd = num;
                            }
                        }
                        
                        const targetOdd = side === 'home' ? homeOdd : awayOdd;
                        if (!targetOdd) continue;
                        
                        console.log('Encontrou odd alvo:', targetOdd);
                        
                        // Encontra e clica no elemento
                        const allSpans = document.querySelectorAll('span, div, button');
                        for (const el of allSpans) {
                            const elText = (el.innerText || '').trim();
                            if (elText === targetOdd) {
                                try {
                                    el.scrollIntoView({block: 'center'});
                                    el.click();
                                    return true;
                                } catch (e) {
                                    try {
                                        el.parentElement.click();
                                        return true;
                                    } catch (e2) {}
                                }
                            }
                        }
                    }
                    
                    return false;
                }
            """, {"lineVariants": line_variants, "side": side})
            
            if clicked:
                await page.wait_for_timeout(1000)
                return True
            
            # === ESTRATÉGIA 3: Query selector direto por texto de odds ===
            print(f"    Tentando Estratégia 3 (query selector)")
            
            # Pega todo texto da página para encontrar odds
            body_text = await page.inner_text("body")
            
            for variant in line_variants:
                # Procura a linha no texto
                for line_text in body_text.split('\n'):
                    if variant in line_text and 'Home' in line_text and 'Away' in line_text:
                        # Extrai odds
                        odds = re.findall(r'\d+[.,]\d{2,3}', line_text)
                        if len(odds) >= 2:
                            # Determina qual odd é qual
                            home_idx = line_text.find('Home')
                            away_idx = line_text.find('Away')
                            
                            home_odd = None
                            away_odd = None
                            
                            for odd in odds:
                                odd_idx = line_text.find(odd)
                                if odd_idx > home_idx and odd_idx < away_idx:
                                    home_odd = odd
                                elif odd_idx > away_idx:
                                    away_odd = odd
                            
                            target_odd = home_odd if side == 'home' else away_odd
                            
                            if target_odd:
                                print(f"    Odd alvo: {target_odd}")
                                
                                # Encontra elemento com esse texto
                                elements = await page.query_selector_all('span, div.odds, button')
                                
                                for el in elements:
                                    try:
                                        el_text = await el.inner_text()
                                        if el_text.strip() == target_odd:
                                            await el.scroll_into_view_if_needed()
                                            await page.wait_for_timeout(200)
                                            
                                            # Clica no pai (mais confiável)
                                            await el.evaluate("el => el.parentElement.click()")
                                            await page.wait_for_timeout(1000)
                                            return True
                                    except:
                                        continue
            
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
