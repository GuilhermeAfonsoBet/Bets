# -*- coding: utf-8 -*-
"""
Extrator de Best Odds do Betslip BetinAsia

Estrutura do betslip (painel direito):
- Título: "AC Pisa 1909 +1.5/2 (Asian)"
- Tabela com colunas: TOTAL | MÉDIA | MELHOR
- Linha "Todos Os Agentes De Apostas" contém a best odd agregada
- Abaixo de cada odd tem o limite em $
- Lista de bookmakers individuais: bdaq, bf, molly, etc.

Uso:
    extractor = BetslipExtractor(page)
    result = await extractor.extract_best_odd()
    print(f"Best odd: {result.best_odd}, Limite: {result.limit}")
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
    selection: str  # Ex: "AC Pisa 1909 +1.5/2 (Asian)"
    match: str  # Ex: "Verona vs. AC Pisa 1909"
    
    # Aggregated "Todos Os Agentes De Apostas"
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
    
    # Bookmakers conhecidos
    KNOWN_BOOKMAKERS = [
        'bdaq', 'bf', 'molly', '3et', '4casters', 
        'ibc', 'isn', 'pin', 'pinnacle', 'sbo', 
        'sing', 'mbook', 'sharp', 'isn88', 'pin88'
    ]
    
    def __init__(self, page: Page):
        self.page = page
        
    async def click_odd_to_open_betslip(
        self, 
        line: str, 
        side: str,
        market_type: str = "AH"
    ) -> bool:
        """
        Clica numa odd específica para abrir o betslip.
        
        Args:
            line: Linha do mercado (ex: "-0.5", "+1.5")
            side: "home" ou "away" (ou "over"/"under" para OU)
            market_type: "AH", "OU", ou "1X2"
            
        Returns:
            True se conseguiu abrir o betslip
        """
        try:
            # Primeiro, encontra a linha correta
            # A estrutura é: LINHA | Home | ODD | Away | ODD
            
            # Procura pelo texto da linha no DOM
            line_text = line.replace(".", ",")  # BetinAsia usa vírgula
            
            # JavaScript para encontrar e clicar na odd correta
            clicked = await self.page.evaluate(f"""
                () => {{
                    const lineText = "{line_text}";
                    const side = "{side}";
                    
                    // Encontra todas as linhas de handicap
                    const rows = document.querySelectorAll('div, tr');
                    
                    for (const row of rows) {{
                        const text = row.innerText || '';
                        
                        // Verifica se esta linha contém o handicap
                        if (text.includes(lineText) && text.includes('Home') && text.includes('Away')) {{
                            // Encontra os elementos clicáveis (odds)
                            const clickables = row.querySelectorAll('span, button, div');
                            
                            for (const el of clickables) {{
                                const elText = (el.innerText || '').trim();
                                
                                // Verifica se é uma odd (número decimal)
                                if (/^\\d+[.,]\\d{{2,3}}$/.test(elText)) {{
                                    // Determina se é home ou away baseado na posição
                                    const rect = el.getBoundingClientRect();
                                    const rowRect = row.getBoundingClientRect();
                                    const isLeftSide = rect.left < (rowRect.left + rowRect.width / 2);
                                    
                                    const elSide = isLeftSide ? 'home' : 'away';
                                    
                                    if (elSide === side) {{
                                        el.click();
                                        return true;
                                    }}
                                }}
                            }}
                        }}
                    }}
                    
                    return false;
                }}
            """)
            
            if clicked:
                await self.page.wait_for_timeout(1500)  # Espera betslip abrir
                return True
            
            # Fallback: procura pela odd específica e clica
            # Isso é menos preciso mas pode funcionar
            logger.warning(f"Método principal falhou, tentando fallback para {line} {side}")
            return False
            
        except Exception as e:
            logger.error(f"Erro ao clicar na odd: {e}")
            return False
    
    async def extract_best_odd(self) -> Optional[BetslipData]:
        """
        Extrai dados do betslip aberto.
        
        Assume que o betslip já está aberto (após clicar numa odd).
        
        Returns:
            BetslipData com best odd e limite, ou None se falhar
        """
        try:
            # Extrai texto do painel direito (betslip)
            betslip_text = await self.page.evaluate("""
                () => {
                    // Procura pelo painel do betslip
                    // Geralmente está no lado direito com classe contendo "betslip" ou "sidebar"
                    const selectors = [
                        '[class*="betslip"]',
                        '[class*="sidebar"]',
                        '[class*="panel"]',
                        'aside',
                    ];
                    
                    for (const selector of selectors) {
                        const panels = document.querySelectorAll(selector);
                        for (const panel of panels) {
                            const text = panel.innerText || '';
                            // Verifica se contém indicadores do betslip (PT ou EN)
                            if (text.includes('All Bookies') ||
                                text.includes('Todos Os Agentes') || 
                                text.includes('BEST') ||
                                text.includes('MELHOR') ||
                                text.includes('TOTAL') ||
                                text.includes('AVERAGE') ||
                                text.includes('Timeout') ||
                                text.includes('Tempo Limite')) {
                                return text;
                            }
                        }
                    }
                    
                    // Fallback: pega texto do body e procura seção do betslip
                    const bodyText = document.body.innerText;
                    const betslipStart = bodyText.indexOf('Betslip');
                    if (betslipStart > -1) {
                        return bodyText.substring(betslipStart, betslipStart + 2000);
                    }
                    
                    return bodyText;
                }
            """)
            
            if not betslip_text:
                logger.warning("Não encontrou texto do betslip")
                return None
            
            # Parse do texto do betslip
            return self._parse_betslip_text(betslip_text)
            
        except Exception as e:
            logger.error(f"Erro ao extrair betslip: {e}")
            return None
    
    def _parse_betslip_text(self, text: str) -> Optional[BetslipData]:
        """
        Parseia o texto do betslip para extrair dados.
        
        Estrutura esperada:
        - "Todos Os Agentes De Apostas" seguido de 3 valores (TOTAL, MÉDIA, MELHOR)
        - Abaixo, 3 valores de limite ($X,XXX)
        - Lista de bookmakers com suas odds
        """
        try:
            # Extrai seleção e jogo
            selection = ""
            match = ""
            
            # Procura por padrão de seleção: "Time +X.X/Y (Asian)"
            selection_match = re.search(r'([A-Za-z\s\d]+\s+[+-]?\d+[.,]?\d*/?\d*\s*\(Asian\))', text)
            if selection_match:
                selection = selection_match.group(1).strip()
            
            # Procura por padrão de jogo: "Time vs. Time"
            match_match = re.search(r'([A-Za-z\s\d]+)\s+vs\.?\s+([A-Za-z\s\d]+)', text)
            if match_match:
                match = f"{match_match.group(1).strip()} vs {match_match.group(2).strip()}"
            
            # Extrai odds agregadas (Todos Os Agentes De Apostas)
            # Padrão: número decimal seguido de mais números
            # Esperado: TOTAL MÉDIA MELHOR (3 valores)
            
            # Procura pela seção agregada (PT ou EN)
            # PT: "Todos Os Agentes De Apostas"
            # EN: "All Bookies"
            todos_idx = text.find('All Bookies')
            if todos_idx == -1:
                todos_idx = text.find('All bookies')
            if todos_idx == -1:
                todos_idx = text.find('Todos Os Agentes')
            if todos_idx == -1:
                todos_idx = text.find('Todos os Agentes')
            if todos_idx == -1:
                # Fallback: procura por padrão de 3 números decimais seguidos (TOTAL AVERAGE BEST)
                odds_pattern = r'(\d+[.,]\d{2,3})\s+(\d+[.,]\d{2,3})\s+(\d+[.,]\d{2,3})'
                match = re.search(odds_pattern, text)
                if match:
                    todos_idx = match.start() - 50  # Pega um pouco antes para contexto
                    if todos_idx < 0:
                        todos_idx = 0
            
            if todos_idx == -1:
                logger.warning("Não encontrou 'All Bookies' ou 'Todos Os Agentes De Apostas'")
                return None
            
            # Pega texto após "Todos Os Agentes"
            section = text[todos_idx:todos_idx + 500]
            
            # Extrai números decimais (odds)
            odds_pattern = r'(\d+[.,]\d{2,3})'
            odds_matches = re.findall(odds_pattern, section)
            
            # Extrai valores monetários (limites)
            limit_pattern = r'\$\s*([\d,]+)'
            limit_matches = re.findall(limit_pattern, section)
            
            if len(odds_matches) < 3:
                logger.warning(f"Encontrou apenas {len(odds_matches)} odds, esperava 3+")
                return None
            
            # Converte odds
            def parse_odd(s):
                return float(s.replace(',', '.'))
            
            def parse_limit(s):
                return float(s.replace(',', ''))
            
            total_odd = parse_odd(odds_matches[0])
            avg_odd = parse_odd(odds_matches[1])
            best_odd = parse_odd(odds_matches[2])  # <-- A BEST ODD
            
            # Limites (podem não existir ou estar em ordem diferente)
            total_limit = parse_limit(limit_matches[0]) if len(limit_matches) > 0 else 0
            avg_limit = parse_limit(limit_matches[1]) if len(limit_matches) > 1 else 0
            best_limit = parse_limit(limit_matches[2]) if len(limit_matches) > 2 else 0
            
            # Extrai bookmakers individuais
            bookmakers = self._extract_bookmakers(section)
            
            return BetslipData(
                selection=selection,
                match=match,
                total_odd=total_odd,
                avg_odd=avg_odd,
                best_odd=best_odd,
                total_limit=total_limit,
                avg_limit=avg_limit,
                best_limit=best_limit,
                bookmakers=bookmakers,
                raw_text=section[:500]
            )
            
        except Exception as e:
            logger.error(f"Erro ao parsear betslip: {e}")
            return None
    
    def _extract_bookmakers(self, text: str) -> List[BookmakerOdd]:
        """Extrai odds individuais de cada bookmaker."""
        bookmakers = []
        text_lower = text.lower()
        
        for bk_name in self.KNOWN_BOOKMAKERS:
            if bk_name in text_lower:
                # Encontra a posição do bookmaker
                idx = text_lower.find(bk_name)
                section = text[idx:idx + 200]
                
                # Extrai odds após o nome do bookmaker
                odds_pattern = r'(\d+[.,]\d{2,3})'
                odds = re.findall(odds_pattern, section)
                
                # Extrai limites
                limit_pattern = r'\$\s*([\d,]+)'
                limits = re.findall(limit_pattern, section)
                
                bk = BookmakerOdd(name=bk_name)
                
                if len(odds) >= 1:
                    bk.total_odd = float(odds[0].replace(',', '.'))
                if len(odds) >= 2:
                    bk.avg_odd = float(odds[1].replace(',', '.'))
                if len(odds) >= 3:
                    bk.best_odd = float(odds[2].replace(',', '.'))
                
                if len(limits) >= 1:
                    bk.total_limit = float(limits[0].replace(',', ''))
                if len(limits) >= 2:
                    bk.avg_limit = float(limits[1].replace(',', ''))
                if len(limits) >= 3:
                    bk.best_limit = float(limits[2].replace(',', ''))
                
                bookmakers.append(bk)
        
        return bookmakers
    
    async def close_betslip(self):
        """Fecha o betslip."""
        try:
            # Tenta clicar no X ou pressionar Escape
            close_selectors = [
                '[class*="close"]',
                'button:has-text("×")',
                'button:has-text("X")',
            ]
            
            for selector in close_selectors:
                try:
                    btn = await self.page.query_selector(selector)
                    if btn and await btn.is_visible():
                        await btn.click()
                        return
                except:
                    continue
            
            # Fallback: pressiona Escape
            await self.page.keyboard.press('Escape')
            
        except Exception as e:
            logger.debug(f"Erro ao fechar betslip: {e}")


async def test_betslip_extractor():
    """Testa o extrator de betslip."""
    from playwright.async_api import async_playwright
    
    print("=" * 60)
    print("TESTE DO EXTRATOR DE BETSLIP")
    print("=" * 60)
    
    p = await async_playwright().start()
    browser = await p.chromium.launch(headless=False)  # headless=False para ver
    context = await browser.new_context()
    page = await context.new_page()
    
    try:
        # Login
        print("\n[1] Fazendo login...")
        page.goto("https://black.betinasia.com/login")
        # ... código de login ...
        
        # Navega para um jogo
        print("\n[2] Navegando para jogo...")
        # ... 
        
        # Extrai betslip
        extractor = BetslipExtractor(page)
        
        # Clica numa odd
        clicked = await extractor.click_odd_to_open_betslip("-0.5", "home")
        
        if clicked:
            data = await extractor.extract_best_odd()
            
            if data:
                print(f"\n[3] Dados extraídos:")
                print(f"    Seleção: {data.selection}")
                print(f"    Jogo: {data.match}")
                print(f"    Best Odd: {data.best_odd}")
                print(f"    Limite: ${data.best_limit}")
                print(f"    Bookmakers: {len(data.bookmakers)}")
            else:
                print("    Falha ao extrair dados")
        else:
            print("    Falha ao abrir betslip")
            
    finally:
        await browser.close()
        await p.stop()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_betslip_extractor())
