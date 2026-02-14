# -*- coding: utf-8 -*-
"""
Verificação de Odds no Betslip

Compara as odds coletadas com as odds reais no betslip do BetinAsia
para verificar se há defasagem nos dados.
"""

import asyncio
import sys
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict
from dataclasses import dataclass
from loguru import logger
from sqlalchemy import text

sys.path.insert(0, '.')

from scraper.betinasia import BetinAsiaScraper
from storage.database import Database


@dataclass
class OddToVerify:
    """Odd a ser verificada."""
    match_id: int
    external_id: str
    home_team: str
    away_team: str
    market_type: str
    line: str
    side: str  # home/away ou over/under
    collected_odd: float
    collected_at: datetime
    event_type: str  # H6, H3B, etc


@dataclass
class VerificationResult:
    """Resultado da verificação."""
    odd_to_verify: OddToVerify
    betslip_odd: Optional[float]
    difference: Optional[float]
    difference_pct: Optional[float]
    status: str  # "ok", "different", "not_found", "error"
    message: str


class BetslipVerifier:
    """Verifica odds no betslip do BetinAsia."""
    
    def __init__(self):
        self.scraper: Optional[BetinAsiaScraper] = None
        self.db: Optional[Database] = None
        
    async def start(self):
        """Inicia o verificador."""
        # Conecta ao banco
        self.db = Database()
        await self.db.connect()
        
        # Inicia o scraper
        self.scraper = BetinAsiaScraper()
        await self.scraper.start()
        await self.scraper.login()
        
        logger.info("BetslipVerifier iniciado")
        
    async def close(self):
        """Fecha o verificador."""
        if self.scraper:
            await self.scraper.close()
        if self.db:
            await self.db.close()
            
    async def get_recent_h6_events(self, limit: int = 10) -> List[OddToVerify]:
        """Busca eventos H6 recentes para verificar."""
        odds_to_verify = []
        
        async with self.db.async_session() as session:
            result = await session.execute(text("""
                SELECT 
                    h.match_id,
                    m.external_id,
                    m.home_team,
                    m.away_team,
                    h.lagged_market_type as market_type,
                    h.lagged_line as line,
                    h.lagged_side as side,
                    h.lagged_current_odd as collected_odd,
                    h.detected_at as collected_at
                FROM h6_correlation_lag_events h
                JOIN matches m ON h.match_id = m.id
                WHERE m.kickoff_time > NOW()  -- Apenas jogos futuros
                ORDER BY h.detected_at DESC
                LIMIT :limit
            """), {"limit": limit})
            
            for row in result.fetchall():
                odds_to_verify.append(OddToVerify(
                    match_id=row[0],
                    external_id=row[1],
                    home_team=row[2],
                    away_team=row[3],
                    market_type=row[4],
                    line=str(row[5]),
                    side=row[6],
                    collected_odd=row[7],
                    collected_at=row[8],
                    event_type="H6"
                ))
                
        return odds_to_verify
    
    async def get_current_odds_from_page(self) -> Dict:
        """Extrai odds atuais da página via JavaScript."""
        # Injeta script para capturar odds do DOM
        odds_data = await self.scraper._page.evaluate("""
            () => {
                const odds = {};
                
                // Procura elementos de odds na página
                // A estrutura exata depende do HTML do BetinAsia
                document.querySelectorAll('[data-odd]').forEach(el => {
                    const oddValue = el.getAttribute('data-odd');
                    const market = el.getAttribute('data-market');
                    const line = el.getAttribute('data-line');
                    const side = el.getAttribute('data-side');
                    
                    if (oddValue && market) {
                        const key = `${market}_${line}_${side}`;
                        odds[key] = parseFloat(oddValue);
                    }
                });
                
                return odds;
            }
        """)
        return odds_data
    
    async def click_odd_and_get_betslip(self, market_type: str, line: str, side: str) -> Optional[float]:
        """
        Clica na odd e captura o valor do betslip.
        
        Returns:
            Valor da odd no betslip, ou None se não encontrar
        """
        page = self.scraper._page
        
        try:
            # Mapeia side para seletor
            side_map = {
                "home": "h",
                "away": "a",
                "over": "over",
                "under": "under"
            }
            side_code = side_map.get(side, side)
            
            # Tenta encontrar o elemento da odd
            # A estrutura exata depende do HTML do BetinAsia
            # Vamos tentar alguns seletores comuns
            
            selectors = [
                # Seletor por data attributes
                f'[data-market="{market_type}"][data-line="{line}"][data-side="{side_code}"]',
                # Seletor por classes
                f'.odd-{market_type.lower()}-{line}-{side_code}',
                # Seletor genérico de odds
                f'.bet-button[data-value*="{line}"]',
            ]
            
            odd_element = None
            for selector in selectors:
                try:
                    odd_element = await page.wait_for_selector(selector, timeout=2000)
                    if odd_element:
                        break
                except:
                    continue
            
            if not odd_element:
                logger.warning(f"Não encontrou elemento para {market_type} {line} {side}")
                return None
            
            # Clica na odd
            await odd_element.click()
            await page.wait_for_timeout(1000)  # Espera betslip abrir
            
            # Captura valor do betslip
            betslip_selectors = [
                '.betslip-odd',
                '.bet-odd-value',
                '[class*="betslip"] [class*="odd"]',
                '.selection-odds',
            ]
            
            for selector in betslip_selectors:
                try:
                    betslip_el = await page.wait_for_selector(selector, timeout=2000)
                    if betslip_el:
                        odd_text = await betslip_el.inner_text()
                        # Remove caracteres não numéricos
                        odd_value = float(''.join(c for c in odd_text if c.isdigit() or c == '.'))
                        return odd_value
                except:
                    continue
            
            logger.warning("Não encontrou valor no betslip")
            return None
            
        except Exception as e:
            logger.error(f"Erro ao clicar na odd: {e}")
            return None
        finally:
            # Fecha betslip se estiver aberto
            try:
                close_btn = await page.query_selector('.betslip-close, .close-betslip, [class*="close"]')
                if close_btn:
                    await close_btn.click()
            except:
                pass
    
    async def verify_single_odd(self, odd: OddToVerify) -> VerificationResult:
        """Verifica uma única odd."""
        try:
            # Navega para a página do jogo
            event_url = f"https://black.betinasia.com/sportsbook/football/{odd.external_id}"
            await self.scraper._page.goto(event_url)
            await self.scraper._page.wait_for_load_state("networkidle")
            await self.scraper._page.wait_for_timeout(3000)
            
            # Tenta obter a odd do betslip
            betslip_odd = await self.click_odd_and_get_betslip(
                odd.market_type, 
                odd.line, 
                odd.side
            )
            
            if betslip_odd is None:
                return VerificationResult(
                    odd_to_verify=odd,
                    betslip_odd=None,
                    difference=None,
                    difference_pct=None,
                    status="not_found",
                    message="Não foi possível encontrar a odd no site"
                )
            
            # Calcula diferença
            diff = betslip_odd - odd.collected_odd
            diff_pct = (diff / odd.collected_odd) * 100 if odd.collected_odd > 0 else 0
            
            # Determina status
            if abs(diff_pct) < 0.5:
                status = "ok"
                message = "Odds correspondem"
            elif abs(diff_pct) < 2:
                status = "minor_diff"
                message = f"Diferença pequena: {diff_pct:+.2f}%"
            else:
                status = "major_diff"
                message = f"Diferença significativa: {diff_pct:+.2f}%"
            
            return VerificationResult(
                odd_to_verify=odd,
                betslip_odd=betslip_odd,
                difference=diff,
                difference_pct=diff_pct,
                status=status,
                message=message
            )
            
        except Exception as e:
            return VerificationResult(
                odd_to_verify=odd,
                betslip_odd=None,
                difference=None,
                difference_pct=None,
                status="error",
                message=str(e)
            )
    
    async def verify_multiple(self, odds: List[OddToVerify]) -> List[VerificationResult]:
        """Verifica múltiplas odds."""
        results = []
        
        for i, odd in enumerate(odds):
            logger.info(f"Verificando {i+1}/{len(odds)}: {odd.home_team} vs {odd.away_team}")
            result = await self.verify_single_odd(odd)
            results.append(result)
            
            # Pausa entre verificações para não sobrecarregar
            await asyncio.sleep(2)
            
        return results


async def main():
    """Executa verificação."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    print("=" * 70)
    print("VERIFICAÇÃO DE ODDS NO BETSLIP")
    print("=" * 70)
    print("""
Este script verifica se as odds coletadas correspondem
às odds reais mostradas no betslip do BetinAsia.
""")
    
    verifier = BetslipVerifier()
    
    try:
        await verifier.start()
        
        # Busca eventos H6 recentes
        print("\nBuscando eventos H6 recentes para verificar...")
        odds_to_verify = await verifier.get_recent_h6_events(limit=5)
        
        if not odds_to_verify:
            print("Nenhum evento H6 encontrado para verificar")
            return
        
        print(f"Encontrados {len(odds_to_verify)} eventos para verificar")
        print("\n" + "-" * 70)
        
        # Verifica cada odd
        results = await verifier.verify_multiple(odds_to_verify)
        
        # Mostra resultados
        print("\n" + "=" * 70)
        print("RESULTADOS")
        print("=" * 70)
        
        ok_count = 0
        diff_count = 0
        error_count = 0
        
        for result in results:
            odd = result.odd_to_verify
            print(f"\n{odd.home_team} vs {odd.away_team}")
            print(f"  Mercado: {odd.market_type} {odd.line} {odd.side}")
            print(f"  Odd coletada:  {odd.collected_odd:.3f} (há {(datetime.now(timezone.utc) - odd.collected_at).seconds}s)")
            
            if result.betslip_odd:
                print(f"  Odd betslip:   {result.betslip_odd:.3f}")
                print(f"  Diferença:     {result.difference:+.3f} ({result.difference_pct:+.2f}%)")
            
            status_emoji = {
                "ok": "✅",
                "minor_diff": "⚠️",
                "major_diff": "❌",
                "not_found": "❓",
                "error": "💥"
            }
            print(f"  Status: {status_emoji.get(result.status, '?')} {result.message}")
            
            if result.status == "ok":
                ok_count += 1
            elif result.status in ["minor_diff", "major_diff"]:
                diff_count += 1
            else:
                error_count += 1
        
        # Resumo
        print("\n" + "=" * 70)
        print("RESUMO")
        print("=" * 70)
        print(f"  ✅ Odds corretas: {ok_count}/{len(results)}")
        print(f"  ⚠️ Com diferença: {diff_count}/{len(results)}")
        print(f"  ❓ Erros/não encontrado: {error_count}/{len(results)}")
        
        if diff_count > 0:
            print("""
NOTA: Diferenças podem ocorrer por:
1. Delay entre coleta e verificação (odds mudam rapidamente)
2. Diferença entre "best odds" agregado e odd de uma casa específica
3. Problemas no scraping
""")
        
    finally:
        await verifier.close()


if __name__ == "__main__":
    asyncio.run(main())
