# -*- coding: utf-8 -*-
"""
Scraper do BetinAsia (BLACK).

Baseado na análise do HTML real do site em Janeiro/2026.
"""

import asyncio
import re
from playwright.async_api import async_playwright, Browser, Page, BrowserContext
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict
from loguru import logger
from dateutil import parser as date_parser

from .models import MatchData, AHLine, BookmakerOdds, ScrapedOpportunity
from config import settings


class BetinAsiaScraper:
    """
    Scraper assíncrono para BetinAsia (versão BLACK).
    
    Uso:
        async with BetinAsiaScraper() as scraper:
            await scraper.login()
            matches = await scraper.scrape_league("England Premier League")
    """
    
    # URLs do site
    BASE_URL = "https://black.betinasia.com"
    LOGIN_URL = f"{BASE_URL}/login"
    SPORTSBOOK_URL = f"{BASE_URL}/sportsbook"
    FOOTBALL_URL = f"{BASE_URL}/sportsbook/football"
    
    # Mapeamento de ligas para códigos de URL
    # Formato: /sportsbook/football/XE/{codigo}
    LEAGUE_CODES = {
        "England Premier League": "XE/1",
        "England Championship": "XE/2",
        "England FA Cup": "XE/132",
        "Germany Bundesliga": "XE/9",
        "Germany 2. Bundesliga": "XE/10",
        "Spain La Liga": "XE/11",
        "Spain Segunda": "XE/12",
        "Italy Serie A": "XE/13",
        "Italy Serie B": "XE/14",
        "France Ligue 1": "XE/15",
        "France Ligue 2": "XE/16",
        "Netherlands Eredivisie": "XE/17",
        "Portugal Primeira Liga": "XE/19",
        "Belgium Pro League": "XE/21",
        "Turkey Super Lig": "XE/26",
        "Brazil Serie A": "XE/31",
        "Argentina Primera Division": "XE/32",
        "UEFA Champions League": "XE/5",
        "UEFA Europa League": "XE/6",
    }
    
    # Bookmakers conhecidos
    KNOWN_BOOKMAKERS = [
        "3et", "bdaq", "bf", "isn", "mbook", 
        "molly", "pinn88", "pin88", "pinnacle",
        "sbo", "sharp", "sing2"
    ]
    
    def __init__(
        self,
        headless: bool = None,
        slow_mo: int = 0,
    ):
        """
        Inicializa o scraper.
        
        Args:
            headless: Se True, browser roda sem interface gráfica.
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
        
        # Contexto com configurações de browser real
        self._context = await self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="pt-BR",
            timezone_id="America/Sao_Paulo",
        )
        
        self._page = await self._context.new_page()
        self._page.set_default_timeout(30000)
        
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
            # Navega para página de login
            await self._page.goto(self.LOGIN_URL)
            await self._page.wait_for_load_state("networkidle")
            
            # Aguarda os campos de login aparecerem
            # O site usa inputs dentro de um form
            await self._page.wait_for_selector("input", timeout=15000)
            
            # Encontra os campos de input
            # Primeiro input é username, segundo é password
            inputs = await self._page.query_selector_all("input")
            
            if len(inputs) < 2:
                logger.error("Não encontrou campos de login suficientes")
                await self.take_screenshot("login_error_fields.png")
                return False
            
            # Preenche username (primeiro input de texto)
            username_input = None
            password_input = None
            
            for inp in inputs:
                input_type = await inp.get_attribute("type")
                if input_type == "password":
                    password_input = inp
                elif input_type in ["text", "email", None]:
                    if username_input is None:
                        username_input = inp
                        
            if not username_input or not password_input:
                logger.error("Não identificou campos de usuário/senha")
                await self.take_screenshot("login_error_inputs.png")
                return False
            
            # Preenche credenciais
            await username_input.fill(username)
            await self._page.wait_for_timeout(500)
            await password_input.fill(password)
            await self._page.wait_for_timeout(500)
            
            # Clica no botão de login
            # Procura por botão com texto "Iniciar Sessão" ou similar
            login_button = await self._page.query_selector(
                "button:has-text('Iniciar'), button:has-text('Login'), "
                "button:has-text('Entrar'), button[type='submit']"
            )
            
            if login_button:
                await login_button.click()
            else:
                # Tenta pressionar Enter
                await password_input.press("Enter")
                
            # Aguarda redirecionamento
            await self._page.wait_for_timeout(3000)
            
            # Verifica se logou (URL mudou ou apareceu elemento de usuário logado)
            current_url = self._page.url
            
            if "login" not in current_url.lower():
                self._logged_in = True
                logger.success("Login realizado com sucesso!")
                return True
            else:
                # Verifica se há mensagem de erro
                error_el = await self._page.query_selector(
                    ".error, .alert-danger, [class*='error']"
                )
                if error_el:
                    error_text = await error_el.inner_text()
                    logger.error(f"Erro no login: {error_text}")
                else:
                    logger.error("Login falhou - ainda na página de login")
                    
                await self.take_screenshot("login_failed.png")
                return False
                
        except Exception as e:
            logger.error(f"Exceção no login: {e}")
            await self.take_screenshot("login_exception.png")
            return False
            
    async def navigate_to_football(self) -> bool:
        """Navega para a seção de futebol."""
        try:
            await self._page.goto(self.FOOTBALL_URL)
            await self._page.wait_for_load_state("networkidle")
            await self._page.wait_for_timeout(2000)
            return True
        except Exception as e:
            logger.error(f"Erro ao navegar para futebol: {e}")
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
            # Obtém o código da liga
            league_code = self.LEAGUE_CODES.get(league_name)
            
            if not league_code:
                logger.warning(f"Liga não mapeada: {league_name}")
                # Tenta navegar pelo menu
                return await self._scrape_league_by_menu(league_name)
                
            # Navega para a página da liga
            league_url = f"{self.FOOTBALL_URL}/{league_code}"
            await self._page.goto(league_url)
            await self._page.wait_for_load_state("networkidle")
            await self._page.wait_for_timeout(2000)
            
            # Encontra os jogos na página
            matches = await self._parse_league_page(league_name)
            
            logger.info(f"Scrape concluído: {len(matches)} partidas encontradas")
            
        except Exception as e:
            logger.error(f"Erro ao fazer scrape da liga {league_name}: {e}")
            await self.take_screenshot(f"scrape_error_{league_name.replace(' ', '_')}.png")
            
        return matches
        
    async def _scrape_league_by_menu(self, league_name: str) -> List[MatchData]:
        """
        Tenta encontrar a liga pelo menu lateral.
        """
        try:
            await self.navigate_to_football()
            
            # Procura link da liga no menu
            league_link = await self._page.query_selector(f"a:has-text('{league_name}')")
            
            if league_link:
                await league_link.click()
                await self._page.wait_for_load_state("networkidle")
                await self._page.wait_for_timeout(2000)
                return await self._parse_league_page(league_name)
            else:
                logger.warning(f"Liga não encontrada no menu: {league_name}")
                return []
                
        except Exception as e:
            logger.error(f"Erro ao buscar liga por menu: {e}")
            return []
            
    async def _parse_league_page(self, league_name: str) -> List[MatchData]:
        """
        Parseia a página de uma liga e extrai os jogos.
        """
        matches = []
        
        try:
            # O site mostra jogos em linhas/cards
            # Cada jogo tem: data/hora, times, odds
            
            # Tenta encontrar elementos de jogo por diferentes seletores
            # Os jogos aparecem como linhas clicáveis
            
            # Procura por links que levam a jogos individuais
            # Padrão de URL: /sportsbook/football/XE/1/2026-01-31,22,94
            game_links = await self._page.query_selector_all(
                "a[href*='/sportsbook/football/'][href*=',']"
            )
            
            logger.info(f"Encontrados {len(game_links)} links de jogos")
            
            # Para cada jogo, extrai informações básicas da lista
            # E depois navega para a página do jogo para pegar odds detalhadas
            
            game_urls = []
            for link in game_links:
                href = await link.get_attribute("href")
                if href and "," in href:  # URLs de jogos têm vírgula
                    full_url = f"{self.BASE_URL}{href}" if href.startswith("/") else href
                    if full_url not in game_urls:
                        game_urls.append(full_url)
                        
            logger.info(f"URLs únicas de jogos: {len(game_urls)}")
            
            # Limita para não sobrecarregar (ajuste conforme necessário)
            max_games = 20
            for i, game_url in enumerate(game_urls[:max_games]):
                try:
                    match_data = await self._scrape_single_match(game_url, league_name)
                    if match_data:
                        matches.append(match_data)
                        logger.debug(f"  [{i+1}/{min(len(game_urls), max_games)}] {match_data}")
                        
                    # Delay entre jogos para não sobrecarregar
                    await self._page.wait_for_timeout(1000)
                    
                except Exception as e:
                    logger.warning(f"Erro ao processar jogo {game_url}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Erro ao parsear página da liga: {e}")
            
        return matches
        
    async def _scrape_single_match(self, match_url: str, league_name: str) -> Optional[MatchData]:
        """
        Faz scrape de um jogo individual.
        Navega para a página do jogo e extrai odds detalhadas.
        """
        try:
            # Navega para a página do jogo
            await self._page.goto(match_url)
            await self._page.wait_for_load_state("networkidle")
            await self._page.wait_for_timeout(1500)
            
            # Extrai informações da página
            page_text = await self._page.inner_text("body")
            
            # Tenta extrair times do título/header
            # O formato típico é "Time1 vs Time2" ou "Time1 - Time2"
            
            # Procura por elementos que contenham os nomes dos times
            home_team = "Unknown Home"
            away_team = "Unknown Away"
            
            # Tenta encontrar pelo padrão de URL
            # URL: /sportsbook/football/XE/1/2026-01-31,22,94
            match = re.search(r'/(\d{4}-\d{2}-\d{2}),(\d+),(\d+)', match_url)
            match_id = f"match_{match.group(2)}_{match.group(3)}" if match else f"match_{hash(match_url)}"
            
            # Tenta extrair data do URL
            if match:
                date_str = match.group(1)
                kickoff_time = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            else:
                kickoff_time = datetime.now(timezone.utc) + timedelta(hours=2)
                
            # Tenta encontrar nomes dos times na página
            # Procura por elementos grandes/destacados com nomes
            team_elements = await self._page.query_selector_all(
                "[class*='team'], [class*='Team'], h1, h2, h3"
            )
            
            team_names = []
            for el in team_elements:
                text = await el.inner_text()
                text = text.strip()
                if text and len(text) > 2 and len(text) < 50:
                    if not any(x in text.lower() for x in ['handicap', 'total', 'gol', 'over', 'under', 'odds']):
                        team_names.append(text)
                        
            # Remove duplicatas mantendo ordem
            seen = set()
            unique_teams = []
            for t in team_names:
                if t not in seen:
                    seen.add(t)
                    unique_teams.append(t)
                    
            if len(unique_teams) >= 2:
                home_team = unique_teams[0]
                away_team = unique_teams[1]
            elif len(unique_teams) == 1:
                # Tenta separar por "vs" ou "-"
                if " vs " in unique_teams[0]:
                    parts = unique_teams[0].split(" vs ")
                    home_team, away_team = parts[0].strip(), parts[1].strip()
                elif " - " in unique_teams[0]:
                    parts = unique_teams[0].split(" - ")
                    home_team, away_team = parts[0].strip(), parts[1].strip()
                    
            # Cria objeto da partida
            match_data = MatchData(
                match_id=match_id,
                league=league_name,
                home_team=home_team,
                away_team=away_team,
                kickoff_time=kickoff_time,
            )
            
            # Extrai odds de Asian Handicap
            match_data.ah_lines = await self._extract_ah_odds()
            
            return match_data
            
        except Exception as e:
            logger.warning(f"Erro ao processar jogo individual: {e}")
            return None
            
    async def _extract_ah_odds(self) -> Dict[str, AHLine]:
        """
        Extrai odds de Asian Handicap da página do jogo.
        
        A página mostra uma tabela com:
        - Linhas de handicap (-1.25, -1, -0.75, etc.)
        - Para cada linha: odds Home e Away
        - Tabela de comparação entre bookmakers
        """
        ah_lines = {}
        
        try:
            # Primeiro, tenta encontrar a seção de Handicap Asiático
            # Procura por texto "Handicap Asiático" ou similar
            
            ah_section = await self._page.query_selector(
                "text=Handicap Asiático, text=Asian Handicap, "
                "[class*='handicap'], [class*='Handicap']"
            )
            
            if not ah_section:
                logger.debug("Seção de Handicap Asiático não encontrada")
                return ah_lines
                
            # Procura por linhas de handicap
            # Formato típico: linha (-1.25), Home (odds), Away (odds)
            
            # Tenta extrair pelo padrão de texto na página
            page_text = await self._page.inner_text("body")
            
            # Regex para encontrar linhas de handicap
            # Padrão: número com sinal (ex: -1.25, +0.5, -0.75)
            # Seguido de "Home" e valor, "Away" e valor
            
            handicap_pattern = r'([+-]?\d+(?:[.,]\d+)?)\s*Home\s*(\d+[.,]\d+)\s*Away\s*(\d+[.,]\d+)'
            matches = re.findall(handicap_pattern, page_text, re.IGNORECASE)
            
            for match in matches:
                line_str = match[0].replace(",", ".")
                home_odds = float(match[1].replace(",", "."))
                away_odds = float(match[2].replace(",", "."))
                
                # Normaliza o formato da linha
                if not line_str.startswith(("+", "-")):
                    line_str = f"+{line_str}" if float(line_str) >= 0 else line_str
                    
                ah_line = AHLine(line=line_str)
                
                # Por enquanto, não temos bookmaker específico
                # Usamos "best" como placeholder
                ah_line.bookmaker_odds["best"] = BookmakerOdds(
                    bookmaker="best",
                    home_odds=home_odds,
                    away_odds=away_odds,
                )
                
                ah_lines[line_str] = ah_line
                
            # Tenta também extrair da tabela de bookmakers
            # A tabela mostra cada bookmaker com suas odds
            bookmaker_odds = await self._extract_bookmaker_table()
            
            # Mescla com as linhas encontradas
            for bk_name, bk_data in bookmaker_odds.items():
                for line_str, odds in bk_data.items():
                    if line_str not in ah_lines:
                        ah_lines[line_str] = AHLine(line=line_str)
                    ah_lines[line_str].bookmaker_odds[bk_name] = odds
                    
        except Exception as e:
            logger.warning(f"Erro ao extrair odds AH: {e}")
            
        return ah_lines
        
    async def _extract_bookmaker_table(self) -> Dict[str, Dict[str, BookmakerOdds]]:
        """
        Extrai odds da tabela de comparação entre bookmakers.
        
        A tabela mostra:
        - Coluna de bookmakers (3et, bdaq, bf, isn, mbook, molly, pinn88, sbo, sharp, sing2)
        - Colunas de odds (TOTAL, MÉDIA, MELHOR ou valores específicos)
        """
        result = {}
        
        try:
            # Procura elementos que contenham nomes de bookmakers conhecidos
            for bk_name in self.KNOWN_BOOKMAKERS:
                bk_elements = await self._page.query_selector_all(f"text={bk_name}")
                
                for bk_el in bk_elements:
                    try:
                        # Tenta encontrar odds próximas ao nome do bookmaker
                        parent = await bk_el.evaluate_handle("el => el.parentElement")
                        parent_text = await parent.inner_text()
                        
                        # Procura por números decimais (odds)
                        odds_pattern = r'(\d+[.,]\d{2,3})'
                        odds_found = re.findall(odds_pattern, parent_text)
                        
                        if len(odds_found) >= 2:
                            home_odds = float(odds_found[0].replace(",", "."))
                            away_odds = float(odds_found[1].replace(",", "."))
                            
                            if bk_name not in result:
                                result[bk_name] = {}
                                
                            # Usa "main" como linha padrão se não identificada
                            result[bk_name]["main"] = BookmakerOdds(
                                bookmaker=bk_name,
                                home_odds=home_odds,
                                away_odds=away_odds,
                            )
                            
                    except:
                        continue
                        
        except Exception as e:
            logger.debug(f"Erro ao extrair tabela de bookmakers: {e}")
            
        return result
        
    async def take_screenshot(self, filename: str = "screenshot.png"):
        """Tira screenshot da página atual."""
        try:
            await self._page.screenshot(path=filename)
            logger.info(f"Screenshot salvo: {filename}")
        except Exception as e:
            logger.warning(f"Erro ao salvar screenshot: {e}")
            
    async def get_page_html(self) -> str:
        """Retorna o HTML da página atual."""
        return await self._page.content()
        
    async def get_page_text(self) -> str:
        """Retorna o texto da página atual."""
        return await self._page.inner_text("body")


# =====================================================
# FUNÇÃO DE TESTE
# =====================================================

async def test_scraper():
    """Testa o scraper com login e scrape de uma liga."""
    print("\n" + "="*60)
    print("TESTE DO SCRAPER BETINASIA")
    print("="*60 + "\n")
    
    async with BetinAsiaScraper(headless=False, slow_mo=500) as scraper:
        print("1. Browser iniciado")
        
        # Login
        success = await scraper.login()
        
        if not success:
            print("❌ Falha no login")
            print("   Verifique as credenciais no arquivo .env")
            return
            
        print("2. ✅ Login OK")
        
        # Navega para futebol
        await scraper.navigate_to_football()
        print("3. ✅ Navegou para futebol")
        
        # Tira screenshot
        await scraper.take_screenshot("test_football.png")
        print("4. Screenshot salvo: test_football.png")
        
        # Tenta scrape da Premier League
        print("\n5. Iniciando scrape da Premier League...")
        matches = await scraper.scrape_league("England Premier League")
        
        print(f"\n   Encontradas {len(matches)} partidas:")
        for m in matches[:5]:  # Mostra apenas 5
            print(f"   - {m.home_team} vs {m.away_team}")
            for line, ah in m.ah_lines.items():
                print(f"     {line}: {ah.num_bookmakers} bookmakers")
                
        print("\n" + "="*60)
        print("Teste concluído!")
        print("="*60)
        
        # Mantém browser aberto para inspeção
        print("\nPressione Enter para fechar o browser...")
        input()


if __name__ == "__main__":
    asyncio.run(test_scraper())
