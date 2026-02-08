# -*- coding: utf-8 -*-
"""
Extrator de Best Odds do Betslip BetinAsia

Estrutura do betslip (painel direito):
- Cabeçalho: "Classic | Exchange | Start Acca"
- Seleção: "Girona FC +2.0 (Asian)"
- Jogo: "Sevilla vs. Girona FC"
- Seção "All Bookies" com colunas: TOTAL | AVERAGE | BEST
  - Cada coluna: odd (ex: 1.030) e limite (ex: $1,310)
- Bookmakers individuais: bdaq, bf, etc.

Uso:
    extractor = BetslipExtractor(page)
    result = await extractor.extract_best_odd()
    print(f"Best odd: {result.best_odd}, Limite: {result.best_limit}")
"""

import re
from dataclasses import dataclass
from typing import Optional, List
from playwright.async_api import Page
from loguru import logger


@dataclass
class BookmakerOdd:
    """Odd de um bookmaker específico."""
    name: str
    total_odd: Optional[float] = None
    avg_odd: Optional[float] = None
    best_odd: Optional[float] = None
    total_limit: Optional[float] = None
    avg_limit: Optional[float] = None
    best_limit: Optional[float] = None


@dataclass
class BetslipData:
    """Dados extraídos do betslip."""
    selection: str  # Ex: "Girona FC +2.0 (Asian)"
    match: str  # Ex: "Sevilla vs. Girona FC"
    
    # Aggregated "All Bookies"
    total_odd: float
    avg_odd: float
    best_odd: float  # <-- A BEST ODD que queremos
    total_limit: float
    avg_limit: float
    best_limit: float  # <-- O limite da best odd
    
    # Individual bookmakers
    bookmakers: List[BookmakerOdd]
    
    # Raw text for debugging
    raw_text: Optional[str] = None


class BetslipExtractor:
    """Extrai dados do betslip do BetinAsia."""
    
    def __init__(self, page: Page):
        self.page = page
        
    async def extract_best_odd(self) -> Optional[BetslipData]:
        """
        Extrai dados do betslip via JavaScript estruturado.
        
        Estratégia: em vez de parsear texto (ambíguo),
        extrai diretamente do DOM usando a estrutura visual:
        - Encontra "All Bookies" 
        - Pega os 3 pares (odd, limite) na mesma linha
        """
        try:
            data = await self.page.evaluate("""
                () => {
                    // === 1. Encontra o painel do betslip ===
                    let betslipPanel = null;
                    const asides = document.querySelectorAll('aside');
                    for (const aside of asides) {
                        const text = aside.innerText || '';
                        if (text.includes('Betslip') && (text.includes('Stake') || text.includes('Place'))) {
                            betslipPanel = aside;
                            break;
                        }
                    }
                    if (!betslipPanel) return null;
                    
                    const fullText = betslipPanel.innerText || '';
                    
                    // === 2. Extrai seleção e jogo ===
                    let selection = '';
                    let match = '';
                    
                    // Extrai seleção (Asian)
                    const asianIdx = fullText.indexOf('(Asian');
                    if (asianIdx > -1) {
                        let start = fullText.lastIndexOf('\\n', asianIdx);
                        if (start === -1) start = 0; else start++;
                        let end = fullText.indexOf(')', asianIdx);
                        if (end > -1) selection = fullText.substring(start, end + 1).trim();
                    }
                    
                    // Extrai jogo (vs)
                    const vsIdx = fullText.indexOf(' vs');
                    if (vsIdx > -1) {
                        let start = fullText.lastIndexOf('\\n', vsIdx);
                        if (start === -1) start = 0; else start++;
                        let end = fullText.indexOf('\\n', vsIdx);
                        if (end === -1) end = vsIdx + 60;
                        match = fullText.substring(start, end).trim();
                    }
                    
                    // === 3. Encontra "All Bookies" e extrai dados ===
                    // Procura o elemento que contém "All Bookies"
                    let allBookiesEl = null;
                    const allEls = betslipPanel.querySelectorAll('*');
                    for (const el of allEls) {
                        const t = (el.innerText || '').trim();
                        if (t === 'All Bookies' || t === 'All bookies' || 
                            t === 'Todos Os Agentes De Apostas' || t === 'Todos os agentes') {
                            allBookiesEl = el;
                            break;
                        }
                    }
                    
                    if (!allBookiesEl) {
                        // Fallback: procura no texto
                        const abIdx = fullText.indexOf('All Bookies');
                        if (abIdx === -1) return { error: 'All Bookies not found', text: fullText.substring(0, 500) };
                    }
                    
                    // === 4. Navega para o container da linha "All Bookies" ===
                    // A estrutura é: [All Bookies] [odd1] [limit1] [odd2] [limit2] [odd3] [limit3]
                    // Procuramos o container que contém "All Bookies" e os números
                    let rowContainer = allBookiesEl ? allBookiesEl.parentElement : null;
                    let rowText = '';
                    
                    // Sobe na árvore até encontrar um container com odds
                    for (let i = 0; i < 5 && rowContainer; i++) {
                        rowText = rowContainer.innerText || '';
                        // Verifica se tem pelo menos 3 pontos decimais (indica odds)
                        let dotCount = 0;
                        let searchFrom = 0;
                        while (true) {
                            const d = rowText.indexOf('.', searchFrom);
                            if (d === -1) break;
                            dotCount++;
                            searchFrom = d + 1;
                        }
                        if (dotCount >= 3) break;
                        rowContainer = rowContainer.parentElement;
                    }
                    
                    // === 5. Extrai odds e limites da linha All Bookies ===
                    // Método: encontra todos os elementos com odds (X.XXX) e limites ($X,XXX)
                    // Odds: 3 dígitos decimais (1.030, 1.016, 1.109)
                    // Limites: precedidos por $ (podem ter vírgula como separador de milhar)
                    
                    const lines = (rowContainer ? rowContainer.innerText : fullText).split('\\n');
                    
                    // Encontra o índice da linha "All Bookies"
                    let abLineIdx = -1;
                    for (let i = 0; i < lines.length; i++) {
                        if (lines[i].trim().includes('All Bookies') || lines[i].trim().includes('All bookies')) {
                            abLineIdx = i;
                            break;
                        }
                    }
                    
                    // Pega as linhas relevantes após "All Bookies" (odds e limites)
                    // Estrutura típica no innerText (separado por \\n):
                    // "All Bookies"
                    // "1.030"      <- TOTAL odd
                    // "$1,310"     <- TOTAL limit
                    // "1.016"      <- AVERAGE odd
                    // "$840"       <- AVERAGE limit
                    // "1.109"      <- BEST odd
                    // "$329"       <- BEST limit
                    
                    let odds = [];
                    let limits = [];
                    
                    // Método robusto: percorre todas as linhas após "All Bookies"
                    // e classifica cada valor como odd ou limite
                    const startIdx = abLineIdx >= 0 ? abLineIdx : 0;
                    const searchText = lines.slice(startIdx, startIdx + 20).join('\\n');
                    
                    for (let i = startIdx + 1; i < Math.min(lines.length, startIdx + 15); i++) {
                        const line = lines[i].trim();
                        
                        // É um limite? ($XXX ou $X,XXX)
                        if (line.startsWith('$')) {
                            const numStr = line.substring(1).trim().split(',').join('');
                            const val = parseFloat(numStr);
                            if (!isNaN(val) && val > 0) {
                                limits.push(val);
                                continue;
                            }
                        }
                        
                        // É uma odd? (formato X.XXX)
                        const dotIdx = line.indexOf('.');
                        if (dotIdx > 0 && dotIdx <= 3 && line.length <= 7) {
                            const val = parseFloat(line);
                            if (!isNaN(val) && val >= 1.001 && val <= 500) {
                                odds.push(val);
                                continue;
                            }
                        }
                        
                        // Se encontrou um bookmaker, para de buscar All Bookies
                        const lineLower = line.toLowerCase();
                        const bkPrefixes = ['bdaq', 'bf', '18bet', 'mbook', 'pin88', 'sbo', 'sharp', 'sing', 'lbc', 'molly', 'isn', 'ibc', 'overtime', 'punter', '3et', '4cast'];
                        let isBk = false;
                        for (const p of bkPrefixes) { if (lineLower.startsWith(p)) { isBk = true; break; } }
                        if (isBk) break;
                    }
                    
                    // === 6. Extrai bookmakers individuais ===
                    const bookmakers = [];
                    const bkNames = ['18bet', 'bdaq', 'bf', 'lbc', 'mbook', 'overtime', 'pin88', 'punter_lo', 'sbo', 'sharp', 'sing', 'sing2', 'molly', 'isn', 'ibc', '3et', '4casters'];
                    
                    for (const bkName of bkNames) {
                        let bkIdx = -1;
                        for (let i = 0; i < lines.length; i++) {
                            if (lines[i].trim().toLowerCase().startsWith(bkName.toLowerCase())) {
                                bkIdx = i;
                                break;
                            }
                        }
                        if (bkIdx === -1) continue;
                        
                        const bkOdds = [];
                        const bkLimits = [];
                        for (let j = bkIdx + 1; j < Math.min(lines.length, bkIdx + 10); j++) {
                            const line = lines[j].trim();
                            if (line.startsWith('$')) {
                                const v = parseFloat(line.substring(1).trim().split(',').join(''));
                                if (!isNaN(v) && v > 0) bkLimits.push(v);
                                continue;
                            }
                            const dotI = line.indexOf('.');
                            if (dotI > 0 && dotI <= 3 && line.length <= 7) {
                                const v = parseFloat(line);
                                if (!isNaN(v) && v >= 1.001 && v <= 500) { bkOdds.push(v); continue; }
                            }
                            const ll = line.toLowerCase();
                            let stop = false;
                            for (const p of ['bdaq','bf','18bet','mbook','pin88','sbo','sharp','sing','lbc','molly','isn','ibc','overtime','punter','all bookies']) {
                                if (ll.startsWith(p)) { stop = true; break; }
                            }
                            if (stop) break;
                        }
                        
                        bookmakers.push({
                            name: bkName,
                            odds: bkOdds,
                            limits: bkLimits,
                        });
                    }
                    
                    return {
                        selection: selection,
                        match: match,
                        odds: odds,           // [TOTAL, AVERAGE, BEST]
                        limits: limits,       // [TOTAL_limit, AVERAGE_limit, BEST_limit]
                        bookmakers: bookmakers,
                        raw: searchText.substring(0, 500),
                    };
                }
            """)
            
            if not data:
                logger.warning("Betslip: nenhum dado extraído")
                return None
            
            if 'error' in data:
                logger.warning(f"Betslip: {data['error']}")
                return None
            
            odds = data.get('odds', [])
            limits = data.get('limits', [])
            
            logger.debug(f"Betslip extraído: odds={odds}, limits={limits}")
            
            if len(odds) < 3:
                logger.warning(f"Betslip: apenas {len(odds)} odds encontradas, esperava 3 (TOTAL/AVG/BEST)")
                # Se tem pelo menos 1 odd, usa como best
                if len(odds) >= 1:
                    best_odd = odds[-1]  # Última odd é provavelmente a BEST
                    best_limit = limits[-1] if limits else 0
                else:
                    return None
            else:
                best_odd = odds[2]  # BEST é a terceira
                best_limit = limits[2] if len(limits) >= 3 else 0
            
            total_odd = odds[0] if len(odds) >= 1 else 0
            avg_odd = odds[1] if len(odds) >= 2 else 0
            total_limit = limits[0] if len(limits) >= 1 else 0
            avg_limit = limits[1] if len(limits) >= 2 else 0
            
            # Converte bookmakers
            bk_list = []
            for bk in data.get('bookmakers', []):
                bk_odds = bk.get('odds', [])
                bk_limits = bk.get('limits', [])
                bk_list.append(BookmakerOdd(
                    name=bk['name'],
                    total_odd=bk_odds[0] if len(bk_odds) >= 1 else None,
                    avg_odd=bk_odds[1] if len(bk_odds) >= 2 else None,
                    best_odd=bk_odds[2] if len(bk_odds) >= 3 else None,
                    total_limit=bk_limits[0] if len(bk_limits) >= 1 else None,
                    avg_limit=bk_limits[1] if len(bk_limits) >= 2 else None,
                    best_limit=bk_limits[2] if len(bk_limits) >= 3 else None,
                ))
            
            return BetslipData(
                selection=data.get('selection', ''),
                match=data.get('match', ''),
                total_odd=total_odd,
                avg_odd=avg_odd,
                best_odd=best_odd,
                total_limit=total_limit,
                avg_limit=avg_limit,
                best_limit=best_limit,
                bookmakers=bk_list,
                raw_text=data.get('raw', ''),
            )
            
        except Exception as e:
            logger.error(f"Erro ao extrair betslip: {e}")
            return None
    
    async def close_betslip(self):
        """Fecha o betslip."""
        try:
            # Tenta clicar no X de fechar
            closed = await self.page.evaluate("""
                () => {
                    // Procura botão de fechar dentro do betslip aside
                    const asides = document.querySelectorAll('aside');
                    for (const aside of asides) {
                        const text = aside.innerText || '';
                        if (text.includes('Betslip') && text.includes('Stake')) {
                            // Encontra o X (botão de fechar a seleção)
                            const closeButtons = aside.querySelectorAll('button, [role="button"], span');
                            for (const btn of closeButtons) {
                                const t = (btn.innerText || '').trim();
                                if (t === '×' || t === 'X' || t === '✕') {
                                    btn.click();
                                    return true;
                                }
                            }
                            // Tenta pelo SVG/ícone de fechar
                            const svgButtons = aside.querySelectorAll('svg, [class*="close"], [class*="remove"]');
                            for (const btn of svgButtons) {
                                try { btn.click(); return true; } catch(e) {}
                                try { btn.parentElement.click(); return true; } catch(e) {}
                            }
                        }
                    }
                    return false;
                }
            """)
            
            if not closed:
                await self.page.keyboard.press('Escape')
                
        except Exception as e:
            logger.debug(f"Erro ao fechar betslip: {e}")
