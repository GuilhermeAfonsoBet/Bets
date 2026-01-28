# -*- coding: utf-8 -*-
"""
Scraper principal do BetinAsia.

IMPORTANTE: Este é um TEMPLATE que precisará ser adaptado
após analisar a estrutura real do site BetinAsia.

Os seletores CSS (.classe, #id) são EXEMPLOS e devem ser
substituídos pelos seletores reais após inspecionar o HTML.
"""

import asyncio
from playwright.async_api import async_playwright, Browser, Page, BrowserContext
from datetime import datetime, timezone
from typing import Optional, List
from loguru import logger
from dateutil import parser as date_parser

from .models import MatchData, AHLine, BookmakerOdds, ScrapedOpportunity
from config import settings


class BetinAsiaScraper:
    """
    Scraper assíncrono para BetinAsia.
    
    Uso:
        async with BetinAsiaScraper() as scraper:
            await scraper.login()
            matches = await scraper.scrape_league("England Premier League")
    """
    
    BASE_URL = "https://www.betinasia.com"
    
    def __init__(
        self,
        headless: bool = None,
        slow_mo: int = 0,
    ):
        """
        Inicializa o scraper.
        
        Args:
            headless: Se True, browser roda sem interface gráfica.
                      None = usa valor do settings.
            slow_mo: Milissegundos de delay entre ações (útil para debug).
        """
        self.headless = headless if headless is not None else settings.browser_headless
        self.slow_mo = slow_mo
        
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._logged_in = False
        
    async def __aenter__(self):
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        
    async def start(self):
        """Inicia o browser."""
        logger.info("Iniciando browser...")
        
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo,
        )
        
        # Cria contexto com configurações de browser real
        self._context = await self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            timezone_id="Asia/Singapore",
        )
        
        self._page = await self._context.new_page()
        
        # Timeout padrão para operações
        self._page.set_default_timeout(30000)  # 30 segundos
        
        logger.info("Browser iniciado com sucesso")
        
    async def close(self):
        """Fecha o browser."""
        logger.info("Fechando browser...")
        
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
            
        self._logged_in = False
        logger.info("Browser fechado")
        
    async def login(
        self,
        username: str = None,
        password: str = None,
    ) -> bool:
        """
        Faz login no BetinAsia.
        
        Args:
            username: Usuário (usa settings se não fornecido)
            password: Senha (usa settings se não fornecido)
            
        Returns:
            True se login bem-sucedido
        """
        username = username or settings.betinasia_username
        password = password or settings.betinasia_password
        
        logger.info("Tentando fazer login no BetinAsia...")
        
        try:
            # =====================================================
            # ATENÇÃO: Os seletores abaixo são EXEMPLOS!
            # Você precisará inspecionar o site real e ajustar.
            # =====================================================
            
            # Navega para página de login
            await self._page.goto(f"{self.BASE_URL}/Account/Login")
            
            # Aguarda campo de usuário
            # AJUSTAR SELETOR: Inspecionar o HTML real do BetinAsia
            await self._page.wait_for_selector(
                "input[name='Username'], input#Username, input[type='text']",
                timeout=15000
            )
            
            # Preenche credenciais
            # AJUSTAR SELETORES conforme o site real
            await self._page.fill("input[name='Username']", username)
            await self._page.fill("input[name='Password']", password)
            
            # Clica no botão de login
            # AJUSTAR SELETOR conforme o site real
            await self._page.click("button[type='submit'], input[type='submit']")
            
            # Aguarda redirecionamento / elemento que indica login OK
            # AJUSTAR: verificar qual elemento aparece após login
            await self._page.wait_for_url(
                f"{self.BASE_URL}/**",
                timeout=20000
            )
            
            # Verifica se realmente logou (procura elemento de usuário logado)
            # AJUSTAR conforme o site real
            # Exemplo: await self._page.wait_for_selector(".user-menu", timeout=5000)
            
            self._logged_in = True
            logger.success("Login realizado com sucesso!")
            return True
            
        except Exception as e:
            logger.error(f"Falha no login: {e}")
            
            # Tira screenshot para debug
            await self._page.screenshot(path="login_error.png")
            logger.info("Screenshot salvo em login_error.png")
            
            return False
            
    async def scrape_league(self, league_name: str) -> List[MatchData]:
        """
        Faz scrape de todas as partidas de uma liga.
        
        Args:
            league_name: Nome da liga (ex: "England Premier League")
            
        Returns:
            Lista de MatchData com odds de AH
        """
        if not self._logged_in:
            logger.warning("Não está logado. Fazendo login...")
            if not await self.login():
                return []
                
        logger.info(f"Iniciando scrape da liga: {league_name}")
        matches = []
        
        try:
            # =====================================================
            # ATENÇÃO: Este código é um TEMPLATE!
            # A estrutura real do BetinAsia pode ser diferente.
            # Você precisará inspecionar o site e ajustar.
            # =====================================================
            
            # Navega para a página de odds de futebol
            # AJUSTAR URL conforme o site real
            await self._page.goto(f"{self.BASE_URL}/Odds/Football")
            
            # Aguarda carregamento
            await self._page.wait_for_load_state("networkidle")
            
            # Encontra e clica na liga
            # AJUSTAR SELETOR conforme o site real
            league_selector = f"text={league_name}"
            
            try:
                await self._page.click(league_selector, timeout=5000)
            except:
                logger.warning(f"Liga não encontrada: {league_name}")
                return []
                
            # Aguarda carregamento das partidas
            await self._page.wait_for_timeout(2000)  # Espera 2 segundos
            
            # Encontra todas as partidas
            # AJUSTAR SELETOR conforme o site real
            match_elements = await self._page.query_selector_all(".match-row, .event-row")
            
            logger.info(f"Encontradas {len(match_elements)} partidas")
            
            for match_el in match_elements:
                try:
                    match_data = await self._parse_match(match_el, league_name)
                    if match_data:
                        matches.append(match_data)
                except Exception as e:
                    logger.warning(f"Erro ao parsear partida: {e}")
                    continue
                    
            logger.info(f"Scrape concluído: {len(matches)} partidas processadas")
            
        except Exception as e:
            logger.error(f"Erro ao fazer scrape da liga {league_name}: {e}")
            await self._page.screenshot(path=f"scrape_error_{league_name}.png")
            
        return matches
    
    async def _parse_match(self, match_el, league_name: str) -> Optional[MatchData]:
        """
        Extrai dados de uma partida.
        
        NOTA: Os seletores são EXEMPLOS e devem ser ajustados.
        """
        try:
            # =====================================================
            # AJUSTAR TODOS OS SELETORES ABAIXO
            # Inspecione o HTML real do BetinAsia
            # =====================================================
            
            # ID da partida
            match_id = await match_el.get_attribute("data-match-id")
            if not match_id:
                match_id = await match_el.get_attribute("data-event-id")
            if not match_id:
                # Gera ID único se não encontrar
                match_id = f"match_{datetime.now().timestamp()}"
            
            # Times
            home_team_el = await match_el.query_selector(".home-team, .team-home")
            away_team_el = await match_el.query_selector(".away-team, .team-away")
            
            home_team = await home_team_el.inner_text() if home_team_el else "Unknown Home"
            away_team = await away_team_el.inner_text() if away_team_el else "Unknown Away"
            
            # Limpa nomes
            home_team = home_team.strip()
            away_team = away_team.strip()
            
            # Horário de início
            kickoff_el = await match_el.query_selector(".kickoff, .match-time, .event-time")
            kickoff_str = await kickoff_el.inner_text() if kickoff_el else ""
            kickoff_time = self._parse_kickoff_time(kickoff_str)
            
            # Cria objeto da partida
            match_data = MatchData(
                match_id=str(match_id),
                league=league_name,
                home_team=home_team,
                away_team=away_team,
                kickoff_time=kickoff_time,
            )
            
            # Extrai odds de Asian Handicap
            # AJUSTAR: encontrar a seção de AH no HTML
            ah_section = await match_el.query_selector(".asian-handicap, .ah-odds, .handicap")
            
            if ah_section:
                match_data.ah_lines = await self._parse_ah_odds(ah_section)
            else:
                # Tenta parsear diretamente do elemento da partida
                match_data.ah_lines = await self._parse_ah_odds(match_el)
                
            return match_data
            
        except Exception as e:
            logger.warning(f"Erro ao parsear partida: {e}")
            return None
            
    async def _parse_ah_odds(self, container) -> dict[str, AHLine]:
        """
        Extrai todas as linhas de AH e odds.
        
        NOTA: Esta é a parte mais complexa e depende
        totalmente da estrutura do HTML do BetinAsia.
        """
        ah_lines = {}
        
        try:
            # =====================================================
            # ESTE É O CÓDIGO QUE MAIS PRECISA DE AJUSTE
            # A estrutura real pode ser completamente diferente
            # =====================================================
            
            # Encontra todas as linhas de handicap
            # AJUSTAR SELETOR
            line_rows = await container.query_selector_all(".ah-line, .handicap-row, .odds-row")
            
            for row in line_rows:
                try:
                    # Valor da linha (ex: +0.5, -0.75)
                    line_el = await row.query_selector(".line-value, .handicap-value")
                    if not line_el:
                        continue
                        
                    line_str = await line_el.inner_text()
                    line_str = line_str.strip()
                    
                    if not line_str:
                        continue
                        
                    ah_line = AHLine(line=line_str)
                    
                    # Encontra odds de cada bookmaker
                    # AJUSTAR SELETOR
                    bk_cells = await row.query_selector_all(".bookmaker-cell, .odds-cell")
                    
                    for cell in bk_cells:
                        try:
                            # Nome do bookmaker
                            bk_name = await cell.get_attribute("data-bookmaker")
                            if not bk_name:
                                bk_name = await cell.get_attribute("data-bk")
                            if not bk_name:
                                continue
                                
                            # Odds
                            home_odds_el = await cell.query_selector(".home-odds, .odds-home")
                            away_odds_el = await cell.query_selector(".away-odds, .odds-away")
                            
                            if home_odds_el and away_odds_el:
                                home_odds_text = await home_odds_el.inner_text()
                                away_odds_text = await away_odds_el.inner_text()
                                
                                home_odds = self._parse_odds(home_odds_text)
                                away_odds = self._parse_odds(away_odds_text)
                                
                                if home_odds and away_odds:
                                    ah_line.bookmaker_odds[bk_name] = BookmakerOdds(
                                        bookmaker=bk_name,
                                        home_odds=home_odds,
                                        away_odds=away_odds,
                                    )
                                    
                        except Exception as e:
                            continue
                            
                    # Só adiciona se tem pelo menos 1 bookmaker
                    if ah_line.num_bookmakers > 0:
                        ah_lines[line_str] = ah_line
                        
                except Exception as e:
                    continue
                    
        except Exception as e:
            logger.warning(f"Erro ao parsear odds AH: {e}")
            
        return ah_lines
    
    def _parse_kickoff_time(self, kickoff_str: str) -> datetime:
        """Converte string de horário para datetime."""
        try:
            # Tenta parsear automaticamente
            dt = date_parser.parse(kickoff_str)
            
            # Se não tem timezone, assume UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
                
            return dt
            
        except:
            # Fallback: retorna hora atual + 2 horas
            return datetime.now(timezone.utc)
    
    def _parse_odds(self, odds_str: str) -> Optional[float]:
        """Converte string de odds para float."""
        try:
            # Remove espaços e caracteres especiais
            odds_str = odds_str.strip()
            
            # Substitui vírgula por ponto
            odds_str = odds_str.replace(",", ".")
            
            # Remove caracteres não numéricos (exceto ponto)
            import re
            odds_str = re.sub(r"[^\d.]", "", odds_str)
            
            return float(odds_str)
            
        except:
            return None
            
    async def take_screenshot(self, filename: str = "screenshot.png"):
        """Tira screenshot da página atual (útil para debug)."""
        await self._page.screenshot(path=filename)
        logger.info(f"Screenshot salvo em {filename}")
        
    async def get_page_html(self) -> str:
        """Retorna o HTML da página atual (útil para debug)."""
        return await self._page.content()


# =====================================================
# SCRIPT DE TESTE DE CONEXÃO
# =====================================================

async def test_connection():
    """Testa a conexão com o BetinAsia."""
    print("\n" + "="*60)
    print("TESTE DE CONEXÃO COM BETINASIA")
    print("="*60 + "\n")
    
    async with BetinAsiaScraper(headless=False, slow_mo=100) as scraper:
        print("1. Browser iniciado com sucesso")
        
        # Tenta fazer login
        success = await scraper.login()
        
        if success:
            print("2. Login realizado com sucesso!")
            print("\n   PRÓXIMOS PASSOS:")
            print("   - Abra o navegador que apareceu")
            print("   - Navegue até uma página com odds")
            print("   - Pressione F12 para abrir DevTools")
            print("   - Inspecione os elementos (classes, IDs)")
            print("   - Anote os seletores CSS")
            print("\n   Pressione Enter para fechar...")
            input()
        else:
            print("2. FALHA no login")
            print("   Verifique:")
            print("   - Credenciais no arquivo .env")
            print("   - Seletores no método login()")
            print("\n   Screenshot salvo em login_error.png")
            
    print("\nTeste finalizado.")


if __name__ == "__main__":
    asyncio.run(test_connection())
