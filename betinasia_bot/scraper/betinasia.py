# -*- coding: utf-8 -*-
"""
Scraper do BetinAsia (BLACK).

Baseado na análise do HTML real do site em Janeiro/2026.
"""

import asyncio
import re
import os
import json
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
        session_file: str = "betinasia_session.json",
    ):
        """
        Inicializa o scraper.
        
        Args:
            headless: Se True, browser roda sem interface gráfica.
            slow_mo: Milissegundos de delay entre ações (útil para debug).
            session_file: Arquivo para salvar/carregar sessão (cookies).
        """
        self.headless = headless if headless is not None else settings.browser_headless
        self.slow_mo = slow_mo
        self.session_file = session_file
        
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
        """Inicia o browser e carrega sessão salva se existir."""
        logger.info("Iniciando browser...")
        
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo,
        )
        
        # Verifica se existe sessão salva
        # Não expiramos por tempo - apenas verificamos se ainda é válida ao usar
        storage_state = None
        if os.path.exists(self.session_file):
            try:
                storage_state = self.session_file
                logger.info("Carregando sessão salva...")
            except Exception as e:
                logger.warning(f"Erro ao carregar sessão: {e}")
        
        # Contexto com configurações de browser real
        context_options = {
            "viewport": {"width": 1920, "height": 1080},
            "user_agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "locale": "pt-BR",
            "timezone_id": "America/Sao_Paulo",
        }
        
        # Carrega sessão salva se existir
        if storage_state:
            context_options["storage_state"] = storage_state
            
        self._context = await self._browser.new_context(**context_options)
        
        self._page = await self._context.new_page()
        self._page.set_default_timeout(30000)
        
        logger.info("Browser iniciado com sucesso")
        
    async def save_session(self):
        """Salva a sessão atual (cookies) para reutilização."""
        try:
            await self._context.storage_state(path=self.session_file)
            logger.info(f"Sessão salva em: {self.session_file}")
        except Exception as e:
            logger.warning(f"Erro ao salvar sessão: {e}")
            
    async def is_session_valid(self) -> bool:
        """Verifica se a sessão atual ainda é válida."""
        try:
            # Navega para uma página que requer login
            await self._page.goto(self.SPORTSBOOK_URL, timeout=15000)
            await self._page.wait_for_load_state("networkidle")
            
            # Se redirecionou para login, sessão expirou
            current_url = self._page.url
            if "login" in current_url.lower():
                logger.info("Sessão expirada, necessário novo login")
                return False
                
            logger.info("Sessão válida!")
            self._logged_in = True
            return True
            
        except Exception as e:
            logger.warning(f"Erro ao verificar sessão: {e}")
            return False
        
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
        force: bool = False,
    ) -> bool:
        """
        Faz login no BetinAsia.
        
        Primeiro verifica se já existe sessão válida salva.
        Só faz login se necessário.
        
        Args:
            username: Usuário (usa settings se não fornecido)
            password: Senha (usa settings se não fornecido)
            force: Se True, força novo login mesmo com sessão válida
            
        Returns:
            True se login bem-sucedido
        """
        # Verifica se já está logado com sessão salva
        if not force and os.path.exists(self.session_file):
            if await self.is_session_valid():
                logger.info("Usando sessão existente - não precisa fazer login")
                return True
        
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
                
                # Salva a sessão para reutilização futura
                await self.save_session()
                
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
            await self._page.wait_for_timeout(3000)  # Aguarda página carregar completamente
            
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
            # Padrão de URL: /sportsbook/football/XE/1/2026-01-31,22,94?origin=sportsbook
            game_links = await self._page.query_selector_all("a")
            
            logger.info(f"Buscando links de jogos em {len(game_links)} links totais")
            
            # Para cada jogo, extrai informações básicas da lista
            # E depois navega para a página do jogo para pegar odds detalhadas
            
            game_urls = []
            for link in game_links:
                href = await link.get_attribute("href")
                # URLs de jogos têm vírgula e são de futebol
                if href and "/sportsbook/football/" in href and "," in href:
                    # Remove query string para comparação
                    base_url = href.split("?")[0]
                    full_url = f"{self.BASE_URL}{base_url}" if base_url.startswith("/") else base_url
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
            await self._page.wait_for_timeout(2000)
            
            # Extrai informações da página
            page_text = await self._page.inner_text("body")
            
            # Extrai ID da partida e data da URL
            # URL: /sportsbook/football/XE/1/2026-01-31,22,94
            url_match = re.search(r'/(\d{4}-\d{2}-\d{2}),(\d+),(\d+)', match_url)
            
            if url_match:
                date_str = url_match.group(1)
                match_id = f"match_{url_match.group(2)}_{url_match.group(3)}"
                kickoff_time = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            else:
                match_id = f"match_{hash(match_url)}"
                kickoff_time = datetime.now(timezone.utc) + timedelta(hours=2)
            
            # Extrai nomes dos times
            # Formato: "Brighton & Hove Albion FC Vs. Everton" ou similar
            home_team = "Unknown Home"
            away_team = "Unknown Away"
            
            # Divide o texto em linhas para análise
            lines = page_text.split('\n')
            
            # Procura pela linha que contém "Vs." ou "vs"
            for i, line in enumerate(lines):
                line = line.strip()
                if ' Vs. ' in line or ' vs ' in line or ' VS ' in line:
                    # Separa os times
                    if ' Vs. ' in line:
                        parts = line.split(' Vs. ')
                    elif ' vs ' in line:
                        parts = line.split(' vs ')
                    else:
                        parts = line.split(' VS ')
                    
                    if len(parts) == 2:
                        home_team = parts[0].strip()
                        away_team = parts[1].strip()
                        
                        # Limita tamanho e limpa
                        home_team = home_team[-100:] if len(home_team) > 100 else home_team
                        away_team = away_team[:100] if len(away_team) > 100 else away_team
                        
                        # Valida se parecem nomes de times
                        if len(home_team) > 2 and len(away_team) > 2:
                            if len(home_team) < 80 and len(away_team) < 80:
                                break
            
            # Se ainda não encontrou, tenta pelo título da página
            if home_team == "Unknown Home":
                # Procura padrão no título: "Time1 Vs. Time2"
                title_pattern = r'([A-Z][A-Za-z\s&\.\-\']{2,40})\s+Vs\.\s+([A-Z][A-Za-z\s&\.\-\']{2,40})'
                title_match = re.search(title_pattern, page_text[:2000])
                if title_match:
                    home_team = title_match.group(1).strip()
                    away_team = title_match.group(2).strip()
            
            # Extrai horário se disponível
            # Formato: "12:00" seguido de data
            time_pattern = r'(\d{2}:\d{2})\s*\n\s*(\d{2}/\d{2}/\d{4})'
            time_match = re.search(time_pattern, page_text)
            if time_match and url_match:
                try:
                    time_str = time_match.group(1)
                    hour, minute = map(int, time_str.split(':'))
                    kickoff_time = kickoff_time.replace(hour=hour, minute=minute)
                except:
                    pass
                    
            # Limita tamanho dos nomes (campo VARCHAR(200) no banco)
            home_team = home_team[:150] if home_team else "Unknown Home"
            away_team = away_team[:150] if away_team else "Unknown Away"
            
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
            
            if match_data.ah_lines:
                logger.debug(f"Jogo: {home_team} vs {away_team} - {len(match_data.ah_lines)} linhas AH")
            
            return match_data
            
        except Exception as e:
            logger.warning(f"Erro ao processar jogo individual: {e}")
            return None
            
    async def _extract_ah_odds(self) -> Dict[str, AHLine]:
        """
        Extrai odds de Asian Handicap da página do jogo.
        
        Formato do BetinAsia BLACK:
        - Seção "Handicap Asiático" com linhas como:
          -0,75 Home 2.205 Away 1.775
          -0,5  Home 1.923 Away 2.029
        """
        ah_lines = {}
        
        try:
            page_text = await self._page.inner_text("body")
            
            # Procura pela seção de Handicap Asiático
            if "Handicap Asiático" not in page_text and "Asian Handicap" not in page_text:
                logger.debug("Seção de Handicap Asiático não encontrada")
                return ah_lines
            
            # Padrão para extrair linhas de AH
            # Formato: -0,75 Home 2.205 Away 1.775
            # ou: -0.75 Home 2.205 Away 1.775
            ah_pattern = r'(-?\d+[,.]?\d*)\s*\n?\s*Home\s*\n?\s*(\d+[,.]\d+)\s*\n?\s*Away\s*\n?\s*(\d+[,.]\d+)'
            
            matches = re.findall(ah_pattern, page_text, re.IGNORECASE)
            
            for match in matches:
                try:
                    # Normaliza a linha (vírgula para ponto)
                    line_str = match[0].replace(",", ".").strip()
                    home_odds = float(match[1].replace(",", "."))
                    away_odds = float(match[2].replace(",", "."))
                    
                    # Ignora valores que não parecem odds válidas
                    if home_odds < 1.01 or home_odds > 50:
                        continue
                    if away_odds < 1.01 or away_odds > 50:
                        continue
                    
                    # Formata a linha com sinal
                    line_float = float(line_str)
                    if line_float > 0:
                        line_str = f"+{line_str}"
                    elif line_float == 0:
                        line_str = "0"
                        
                    ah_line = AHLine(line=line_str)
                    
                    # Salva como "best" (melhor odd agregada)
                    ah_line.bookmaker_odds["best"] = BookmakerOdds(
                        bookmaker="best",
                        home_odds=home_odds,
                        away_odds=away_odds,
                    )
                    
                    ah_lines[line_str] = ah_line
                    logger.debug(f"AH encontrado: {line_str} H:{home_odds} A:{away_odds}")
                    
                except (ValueError, IndexError) as e:
                    continue
            
            logger.info(f"Extraídas {len(ah_lines)} linhas de AH")
                    
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
