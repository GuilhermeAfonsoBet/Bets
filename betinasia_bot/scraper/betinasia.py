# -*- coding: utf-8 -*-
"""
Scraper do BetinAsia (BLACK).

Baseado na análise do HTML real do site em Janeiro/2026.
Atualizado com lógica de expansão de linhas AH e captura de bookmakers.
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

    # Navegação (Playwright)
    # Observação: `goto()` por padrão espera `load`, o que costuma estourar timeout
    # em páginas pesadas/proxy. Para estabilidade operacional, preferimos
    # `domcontentloaded` + waits explícitos.
    DEFAULT_TIMEOUT_MS = int(os.getenv("BETINASIA_DEFAULT_TIMEOUT_MS", "60000"))
    DEFAULT_NAV_TIMEOUT_MS = int(os.getenv("BETINASIA_NAV_TIMEOUT_MS", "90000"))
    DEFAULT_GOTO_WAIT_UNTIL = os.getenv("BETINASIA_GOTO_WAIT_UNTIL", "domcontentloaded")
    
    # Mapeamento de ligas para códigos de URL
    # Formato: /sportsbook/football/{codigo_pagina}
    # IMPORTANTE: Usar os códigos que aparecem na URL quando você acessa a liga no site
    LEAGUE_CODES = {
        # Inglaterra
        "England Premier League": "XE/1",
        "England Championship": "XE/2",
        "England FA Cup": "XE/132",
        # Alemanha - URL correta é DE/12 (não XB!)
        "Germany Bundesliga": "DE/12",
        "Germany 2. Bundesliga": "DE/13",
        # Espanha - URL correta é ES/16
        "Spain La Liga": "ES/16",
        "Spain Segunda": "ES/17",
        # Itália - URL correta é IT/19
        "Italy Serie A": "IT/19",
        "Italy Serie B": "IT/20",
        # França - URL correta é FR/38
        "France Ligue 1": "FR/38",
        "France Ligue 2": "FR/39",
        # Outros
        "Netherlands Eredivisie": "NL/1",
        "Portugal Primeira Liga": "PT/1",
        "Belgium Pro League": "BE/1",
        "Turkey Super Lig": "TR/160",
        "Brazil Serie A": "BR/1",
        "Argentina Primera Division": "AR/1",
        # Competições europeias
        "UEFA Champions League": "XE/5",
        "UEFA Europa League": "XE/6",
    }
    
    # Mapeamento de código de liga para códigos de URL de jogos
    # Usado para filtrar apenas jogos da liga correta
    # Baseado nos prints do site real
    LEAGUE_URL_PATTERNS = {
        "England Premier League": ["XE/1"],
        "England Championship": ["XE/2"],
        "Germany Bundesliga": ["DE/12"],
        "Germany 2. Bundesliga": ["DE/13"],
        "Spain La Liga": ["ES/16"],
        "Spain Segunda": ["ES/17"],
        "Italy Serie A": ["IT/19"],
        "Italy Serie B": ["IT/20"],
        "France Ligue 1": ["FR/38"],
        "France Ligue 2": ["FR/39"],
        "Netherlands Eredivisie": ["NL/"],
        "Portugal Primeira Liga": ["PT/"],
        "Belgium Pro League": ["BE/"],
        "Turkey Super Lig": ["TR/160"],
        "Brazil Serie A": ["BR/"],
        "Argentina Primera Division": ["AR/"],
        "UEFA Champions League": ["XE/5"],
        "UEFA Europa League": ["XE/6"],
    }
    
    # Bookmakers conhecidos (nomes podem ter 'e' no final: 3ete, pin88e)
    KNOWN_BOOKMAKERS = [
        "3et", "4casters", "bdaq", "bf", "ibc", "ipm",
        "isn", "mbook", "molly", "pin88", "pinnacle",
        "sbo", "sharp", "sing"
    ]
    
    # Filtro de stake mínimo: ignora linhas AH onde o max stake da best odd é <= este valor
    MIN_STAKE_FILTER = 20.0
    
    def __init__(
        self,
        headless: bool = None,
        slow_mo: int = 0,
        session_file: str = "betinasia_session.json",
        proxy: dict = None,
    ):
        """
        Inicializa o scraper.
        
        Args:
            headless: Se True, browser roda sem interface gráfica.
            slow_mo: Milissegundos de delay entre ações (útil para debug).
            session_file: Arquivo para salvar/carregar sessão (cookies).
            proxy: Dict com config de proxy. Ex:
                {"server": "http://brd.superproxy.io:33335",
                 "username": "brd-customer-XXX-zone-YYY",
                 "password": "ZZZ"}
        """
        self.headless = headless if headless is not None else settings.browser_headless
        self.slow_mo = slow_mo
        self.session_file = session_file
        self.proxy = proxy if proxy is not None else settings.proxy_config
        
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
        
        launch_options = {
            "headless": self.headless,
            "slow_mo": self.slow_mo,
        }
        
        # Proxy residencial (Bright Data, IPRoyal, etc)
        if self.proxy:
            launch_options["proxy"] = self.proxy
            logger.info(f"Usando proxy: {self.proxy.get('server', '?')}")
        
        self._browser = await self._playwright.chromium.launch(**launch_options)
        
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
            "ignore_https_errors": True,
        }
        
        # Carrega sessão salva se existir
        if storage_state:
            context_options["storage_state"] = storage_state
            
        self._context = await self._browser.new_context(**context_options)
        
        self._page = await self._context.new_page()
        # Timeouts mais tolerantes (proxy / sportsbook costuma ser pesado)
        self._page.set_default_timeout(self.DEFAULT_TIMEOUT_MS)
        self._page.set_default_navigation_timeout(self.DEFAULT_NAV_TIMEOUT_MS)
        
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
            await self._page.goto(
                self.SPORTSBOOK_URL,
                timeout=self.DEFAULT_NAV_TIMEOUT_MS,
                wait_until=self.DEFAULT_GOTO_WAIT_UNTIL,
            )
            await self._page.wait_for_timeout(1000)
            
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

        if not username or not password:
            logger.error(
                "Credenciais BetinAsia ausentes. Defina BETINASIA_USERNAME e BETINASIA_PASSWORD no .env "
                "ou passe username/password explicitamente."
            )
            return False
        
        logger.info("Tentando fazer login no BetinAsia...")
        
        try:
            # Navega para página de login
            await self._page.goto(
                self.LOGIN_URL,
                timeout=self.DEFAULT_NAV_TIMEOUT_MS,
                wait_until=self.DEFAULT_GOTO_WAIT_UNTIL,
            )
            
            # Aguarda os campos de login aparecerem
            # O site usa inputs dentro de um form
            await self._page.wait_for_selector("input", timeout=self.DEFAULT_TIMEOUT_MS)
            
            # Preenche APENAS o primeiro input de texto (username)
            # O site tem múltiplos inputs, precisamos ser específicos
            text_inputs = await self._page.query_selector_all("input[type='text']")
            password_input = await self._page.query_selector("input[type='password']")
            
            if not text_inputs or not password_input:
                logger.error("Não encontrou campos de login")
                await self.take_screenshot("login_error_fields.png")
                return False
            
            # Preenche credenciais
            await text_inputs[0].fill(username)  # Primeiro input de texto = username
            await self._page.wait_for_timeout(500)
            await password_input.fill(password)
            await self._page.wait_for_timeout(500)
            
            # Clica no botão de login (texto "Log In")
            login_button = await self._page.query_selector(
                "button:has-text('Log In'), button:has-text('Iniciar'), "
                "button:has-text('Login'), button:has-text('Entrar'), "
                "button[type='submit']"
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
            await self._page.goto(
                self.FOOTBALL_URL,
                timeout=self.DEFAULT_NAV_TIMEOUT_MS,
                wait_until=self.DEFAULT_GOTO_WAIT_UNTIL,
            )
            await self._page.wait_for_timeout(2000)
            return True
        except Exception as e:
            logger.error(f"Erro ao navegar para futebol: {e}")
            return False
            
    async def _expand_ah_section(self):
        """
        Expande a seção de Handicap Asiático clicando em 'Show all lines'.
        Aguarda carregamento completo e clica múltiplas vezes se necessário.
        """
        try:
            # Aguarda carregamento inicial
            await self._page.wait_for_load_state("networkidle")
            await self._page.wait_for_timeout(2000)
            
            # Clica em TODOS os botões "Show all lines" até não haver mais
            for attempt in range(5):
                # Tenta diferentes seletores
                expand_buttons = []
                
                # Seletor por texto exato
                btns1 = await self._page.query_selector_all("text='Show all lines'")
                btns2 = await self._page.query_selector_all("text='Mostrar todas as linhas'")
                btns3 = await self._page.query_selector_all("text='Show all'")
                
                # Seletor por texto parcial (mais flexível)
                btns4 = await self._page.query_selector_all("button:has-text('Show all')")
                btns5 = await self._page.query_selector_all("button:has-text('Mostrar')")
                btns6 = await self._page.query_selector_all("[role='button']:has-text('Show all')")
                
                expand_buttons = btns1 + btns2 + btns3 + btns4 + btns5 + btns6
                
                visible_buttons = []
                for button in expand_buttons:
                    try:
                        if await button.is_visible():
                            visible_buttons.append(button)
                    except:
                        continue
                
                # Remove duplicatas
                unique_buttons = []
                seen = set()
                for btn in visible_buttons:
                    try:
                        box = await btn.bounding_box()
                        if box:
                            key = (int(box['x']), int(box['y']))
                            if key not in seen:
                                seen.add(key)
                                unique_buttons.append(btn)
                    except:
                        pass
                
                if not unique_buttons:
                    logger.debug(f"Expansão completa após {attempt} tentativas")
                    break
                
                logger.debug(f"Tentativa {attempt+1}: {len(unique_buttons)} botões 'Show all' encontrados")
                
                for button in unique_buttons:
                    try:
                        await button.scroll_into_view_if_needed()
                        await button.click()
                        await self._page.wait_for_timeout(1500)
                        logger.debug("Clicou em 'Show all lines'")
                    except Exception as e:
                        logger.debug(f"Erro ao clicar: {e}")
                        continue
                            
        except Exception as e:
            logger.debug(f"Erro ao expandir seção AH: {e}")
    
    async def _expand_game_list(self):
        """
        Expande a lista de jogos fazendo scroll para carregar lazy content.
        NÃO clica em filtros "All/Todos" pois isso expande para TODAS as ligas!
        """
        try:
            # REMOVIDO: Não clicar em filtros "All/Todos" - isso expande para todas as ligas!
            # Os jogos da liga específica já são carregados quando navegamos para a página da liga.
            
            # 1. Expande grupos de data (ex: "January 30", "January 31", etc.)
            # Alguns sites agrupam jogos por data e precisam ser expandidos
            date_group_selectors = [
                "[class*='date-group'] [class*='expand']",
                "[class*='date-header']",
                "[class*='match-day']",
                "[class*='game-day']",
            ]
            
            for selector in date_group_selectors:
                try:
                    date_groups = await self._page.query_selector_all(selector)
                    for group in date_groups:
                        try:
                            if await group.is_visible():
                                await group.click()
                                await self._page.wait_for_timeout(500)
                        except:
                            continue
                except:
                    continue
            
            # 2. Tenta clicar em todos os botões "Show more" visíveis
            for attempt in range(10):
                try:
                    # Procura botões de "Mostrar mais" / "Show more"
                    show_more_buttons = await self._page.query_selector_all(
                        "text='Show more', text='Mostrar mais', text='Load more', "
                        "button:has-text('more'), button:has-text('mais'), "
                        "text='Show all', text='Ver todos'"
                    )
                    
                    clicked = False
                    for btn in show_more_buttons:
                        try:
                            if await btn.is_visible():
                                await btn.scroll_into_view_if_needed()
                                await btn.click()
                                await self._page.wait_for_timeout(1500)
                                logger.debug(f"Clicou em 'Show more' (tentativa {attempt+1})")
                                clicked = True
                        except:
                            continue
                    
                    if not clicked:
                        break
                        
                except Exception as e:
                    logger.debug(f"Erro ao expandir lista: {e}")
                    break
                    
            # 3. Faz scroll extensivo para carregar jogos via lazy loading
            # Mais agressivo para garantir que todos os jogos carreguem
            for i in range(12):
                await self._page.evaluate("window.scrollBy(0, 800)")
                await self._page.wait_for_timeout(800)
                
            # 4. Volta ao topo e aguarda
            await self._page.evaluate("window.scrollTo(0, 0)")
            await self._page.wait_for_timeout(1500)
            
            # 5. Segundo round de scroll (às vezes mais jogos carregam depois)
            for i in range(5):
                await self._page.evaluate("window.scrollBy(0, 1000)")
                await self._page.wait_for_timeout(600)
                
            await self._page.evaluate("window.scrollTo(0, 0)")
            await self._page.wait_for_timeout(1000)
            
        except Exception as e:
            logger.debug(f"Erro ao expandir lista: {e}")
            
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
            logger.info(f"Navegando para: {league_url}")
            await self._page.goto(league_url)
            await self._page.wait_for_load_state("networkidle")
            await self._page.wait_for_timeout(4000)  # Espera mais para carregar jogos
            
            # Verifica se a URL está correta
            current_url = self._page.url
            logger.info(f"URL atual após navegação: {current_url}")
            
            # Se a URL não contém o código da liga, tenta novamente
            if league_code not in current_url:
                logger.warning(f"URL incorreta! Esperado {league_code}, atual: {current_url}")
                await self._page.goto(league_url)
                await self._page.wait_for_load_state("networkidle")
                await self._page.wait_for_timeout(4000)
            
            # Tenta expandir a lista clicando em "Mostrar mais" / "Show more"
            await self._expand_game_list()
            
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
            
            # Obtém padrões de URL para filtrar apenas jogos desta liga
            url_patterns = self.LEAGUE_URL_PATTERNS.get(league_name, [])
            
            # Procura por links que levam a jogos individuais
            # Padrão de URL: /sportsbook/football/XE/1/2026-01-31,22,94?origin=sportsbook
            game_links = await self._page.query_selector_all("a")
            
            logger.info(f"Buscando links de jogos em {len(game_links)} links totais")
            
            # Para cada jogo, extrai informações básicas da lista
            # E depois navega para a página do jogo para pegar odds detalhadas
            
            game_urls = []
            game_urls_all = []  # Para debug
            game_urls_rejected = []  # URLs rejeitadas (para debug)
            
            for link in game_links:
                href = await link.get_attribute("href")
                # URLs de jogos têm vírgula e são de futebol
                if href and "/sportsbook/football/" in href and "," in href:
                    # Remove query string para comparação
                    base_url = href.split("?")[0]
                    full_url = f"{self.BASE_URL}{base_url}" if base_url.startswith("/") else base_url
                    
                    if full_url not in game_urls_all:
                        game_urls_all.append(full_url)
                    
                    # Filtra apenas jogos da liga correta
                    if url_patterns:
                        is_correct_league = any(pattern in href for pattern in url_patterns)
                        if not is_correct_league:
                            if full_url not in game_urls_rejected:
                                game_urls_rejected.append(full_url)
                            continue
                    
                    if full_url not in game_urls:
                        game_urls.append(full_url)
            
            logger.info(f"URLs únicas de jogos: {len(game_urls)} (de {len(game_urls_all)} totais, {len(game_urls_rejected)} rejeitadas)")
            
            # Debug: mostra quais padrões estão sendo usados e URLs rejeitadas
            if url_patterns:
                logger.debug(f"Padrões de filtro para {league_name}: {url_patterns}")
                
                # Mostra algumas URLs rejeitadas para debug
                if game_urls_rejected:
                    logger.debug(f"Primeiras 5 URLs rejeitadas:")
                    for url in game_urls_rejected[:5]:
                        logger.debug(f"  - {url}")
            
            # Limita para não sobrecarregar (ajuste conforme necessário)
            max_games = 20
            for i, game_url in enumerate(game_urls[:max_games]):
                try:
                    # capture_bookmakers=True para obter odds de cada casa
                    match_data = await self._scrape_single_match(
                        game_url, 
                        league_name, 
                        capture_bookmakers=True
                    )
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
        
    async def _scrape_single_match(
        self, 
        match_url: str, 
        league_name: str,
        capture_bookmakers: bool = False
    ) -> Optional[MatchData]:
        """
        Faz scrape de um jogo individual.
        Navega para a página do jogo e extrai odds detalhadas.
        
        Args:
            match_url: URL completa do jogo
            league_name: Nome da liga
            capture_bookmakers: Se True, captura odds de cada bookmaker (mais lento)
        """
        try:
            # Navega para a página do jogo
            await self._page.goto(match_url)
            await self._page.wait_for_load_state("networkidle")
            await self._page.wait_for_timeout(3000)  # Aumentado para garantir carregamento
            
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
            
            # Extrai nomes dos times usando padrão "Time1 Vs. Time2"
            home_team = "Unknown Home"
            away_team = "Unknown Away"
            
            # Procura em linhas individuais - é mais confiável
            lines = page_text.split('\n')
            for line in lines:
                line = line.strip()
                # Linha deve conter " Vs. " e ter tamanho razoável
                if ' Vs. ' in line and 10 < len(line) < 120:
                    parts = line.split(' Vs. ')
                    if len(parts) == 2:
                        potential_home = parts[0].strip()
                        potential_away = parts[1].strip()
                        
                        # Valida se parecem nomes de times (não itens de menu)
                        menu_words = ['football', 'tennis', 'live', 'top', 'basketball', 
                                     'baseball', 'hockey', 'cricket', 'rugby', 'boxing',
                                     'handball', 'volleyball', 'golf', 'snooker', 'darts']
                        
                        is_menu = any(w in potential_home.lower() for w in menu_words)
                        
                        if not is_menu and len(potential_home) > 3 and len(potential_away) > 3:
                            home_team = potential_home
                            away_team = potential_away
                            break
            
            # Extrai horário se disponível
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
            match_data.ah_lines = await self._extract_ah_odds(capture_bookmakers=capture_bookmakers)
            
            if match_data.ah_lines:
                logger.debug(f"Jogo: {home_team} vs {away_team} - {len(match_data.ah_lines)} linhas AH")
            
            return match_data
            
        except Exception as e:
            logger.warning(f"Erro ao processar jogo individual: {e}")
            return None
            
    async def _extract_ah_odds(self, capture_bookmakers: bool = False) -> Dict[str, AHLine]:
        """
        Extrai odds de Asian Handicap da página do jogo.
        
        Args:
            capture_bookmakers: Se True, clica em cada odds para capturar 
                               odds individuais de cada bookmaker.
        
        Returns:
            Dict com linhas de AH e suas odds.
        """
        ah_lines = {}
        
        try:
            # Primeiro, expande a seção de Handicap Asiático
            await self._expand_ah_section()
            await self._page.wait_for_timeout(1000)
            
            # Extrai o texto da página
            page_text = await self._page.inner_text("body")
            
            # Verifica se a seção existe
            if "Asian Handicap" not in page_text and "Handicap Asiático" not in page_text:
                logger.debug("Seção Asian Handicap não encontrada")
                return ah_lines
            
            # Encontra seção Asian Handicap (termina antes de Asian Total ou Correct Score)
            ah_section_match = re.search(
                r'Asian Handicap\n(.+?)(?=\nAsian Total|\nCorrect Score|\nShow all lines\nAsian)',
                page_text, 
                re.DOTALL
            )
            
            if not ah_section_match:
                logger.debug("Não conseguiu isolar seção Asian Handicap")
                # Tenta extrair do texto completo
                ah_text = page_text
            else:
                ah_text = ah_section_match.group(1)
            
            # Padrão: HANDICAP\nHome\nODDS\nAway\nODDS
            pattern = r'([+-]?\d+[,.]?\d*)\s*\nHome\s*\n(\d+[,.]\d+)\s*\nAway\s*\n(\d+[,.]\d+)'
            matches = re.findall(pattern, ah_text)
            
            for match in matches:
                try:
                    handicap_str = match[0].replace(",", ".")
                    handicap_value = float(handicap_str)
                    
                    # Verifica se é handicap válido (múltiplo de 0.25, entre -10 e +10)
                    if handicap_value % 0.25 != 0 or abs(handicap_value) > 10:
                        continue
                    
                    home_odds = float(match[1].replace(",", "."))
                    away_odds = float(match[2].replace(",", "."))
                    
                    # Valida odds
                    if not (1.01 <= home_odds <= 100 and 1.01 <= away_odds <= 100):
                        continue
                    
                    # Formata linha
                    if handicap_value > 0:
                        formatted_line = f"+{handicap_value:.2f}".rstrip('0').rstrip('.')
                    elif handicap_value == 0:
                        formatted_line = "0"
                    else:
                        formatted_line = f"{handicap_value:.2f}".rstrip('0').rstrip('.')
                    
                    # Evita duplicatas
                    if formatted_line in ah_lines:
                        continue
                    
                    # Cria objeto AHLine
                    ah_line = AHLine(line=formatted_line)
                    
                    # Adiciona "best" como bookmaker padrão
                    ah_line.bookmaker_odds["best"] = BookmakerOdds(
                        bookmaker="best",
                        home_odds=home_odds,
                        away_odds=away_odds,
                    )
                    
                    # Se solicitado, captura odds de cada bookmaker (COM FILTRO DE STAKE POR COMBINAÇÃO)
                    if capture_bookmakers:
                        bookmaker_odds = await self._capture_bookmaker_odds(
                            home_odds_str=match[1],
                            away_odds_str=match[2],
                            handicap=formatted_line
                        )
                        # Adiciona bookmakers que passaram no filtro de stake
                        if bookmaker_odds:
                            ah_line.bookmaker_odds.update(bookmaker_odds)
                    
                    ah_lines[formatted_line] = ah_line
                    logger.debug(f"AH: {formatted_line} H:{home_odds:.3f} A:{away_odds:.3f} ({len(ah_line.bookmaker_odds)} bks)")
                        
                except (ValueError, IndexError) as e:
                    continue
            
            logger.info(f"Extraídas {len(ah_lines)} linhas de AH")
                    
        except Exception as e:
            logger.warning(f"Erro ao extrair odds AH: {e}")
            
        return ah_lines
    
    async def _capture_bookmaker_odds(
        self, 
        home_odds_str: str, 
        away_odds_str: str,
        handicap: str = ""
    ) -> Dict[str, BookmakerOdds]:
        """
        Captura odds de bookmakers individuais clicando nas odds.
        
        Clica na odds de Home e Away para abrir o painel com todos os bookmakers.
        O parâmetro handicap ajuda a identificar o elemento correto quando há
        múltiplos elementos com a mesma odds (ex: mesma odds em AH e 1X2).
        
        FILTRO DE STAKE: Aplicado POR COMBINAÇÃO (home/away separadamente).
        Se o stake máximo de uma combinação for <= MIN_STAKE_FILTER, 
        ignora apenas essa combinação, não a linha inteira.
        """
        bookmaker_odds = {}
        home_bks_filtered = {}
        away_bks_filtered = {}
        
        try:
            # Captura odds de HOME
            home_bks = await self._click_and_extract_bookmakers(home_odds_str, "home", handicap)
            
            # FILTRO DE STAKE para HOME
            if home_bks:
                stakes = [bk.get("limit", 0) for bk in home_bks.values()]
                max_stake = max(stakes) if stakes else 0
                
                if max_stake > 0 and max_stake <= self.MIN_STAKE_FILTER:
                    logger.debug(f"AH {handicap} HOME: Stake máx ${max_stake:.0f} <= ${self.MIN_STAKE_FILTER:.0f}, ignorando combinação")
                    # Não usa os dados de HOME, mas continua para AWAY
                else:
                    home_bks_filtered = home_bks
            
            # Captura odds de AWAY
            away_bks = await self._click_and_extract_bookmakers(away_odds_str, "away", handicap)
            
            # FILTRO DE STAKE para AWAY
            if away_bks:
                stakes = [bk.get("limit", 0) for bk in away_bks.values()]
                max_stake = max(stakes) if stakes else 0
                
                if max_stake > 0 and max_stake <= self.MIN_STAKE_FILTER:
                    logger.debug(f"AH {handicap} AWAY: Stake máx ${max_stake:.0f} <= ${self.MIN_STAKE_FILTER:.0f}, ignorando combinação")
                    # Não usa os dados de AWAY
                else:
                    away_bks_filtered = away_bks
            
            # Combina os resultados (apenas combinações que passaram no filtro)
            all_bookmakers = set(home_bks_filtered.keys()) | set(away_bks_filtered.keys())
            
            for bk_name in all_bookmakers:
                home_data = home_bks_filtered.get(bk_name, {})
                away_data = away_bks_filtered.get(bk_name, {})
                
                bookmaker_odds[bk_name] = BookmakerOdds(
                    bookmaker=bk_name,
                    home_odds=home_data.get("odds", 0.0),
                    away_odds=away_data.get("odds", 0.0),
                )
                
        except Exception as e:
            logger.debug(f"Erro ao capturar bookmakers: {e}")
            
        return bookmaker_odds
    
    async def _click_and_extract_bookmakers(
        self, 
        odds_str: str, 
        side: str,
        handicap: str = ""
    ) -> Dict[str, dict]:
        """
        Clica em uma odds específica e extrai os bookmakers do painel.
        
        IMPORTANTE: O clique deve ser no elemento PAI (DIV), não no SPAN de texto.
        Quando há múltiplos elementos com a mesma odds (ex: mesma odds em AH e 1X2),
        prioriza elementos cujo contexto contenha o handicap esperado.
        """
        bookmakers = {}
        
        def normalize_handicap(h: str) -> str:
            """Normaliza handicap removendo sinais e convertendo vírgula para ponto."""
            return h.lstrip('+').lstrip('-').replace(',', '.').replace('−', '')
        
        def handicap_matches_context(handicap: str, context: str) -> bool:
            """Verifica se o handicap está presente no contexto (com variações de formato)."""
            if not handicap:
                return False
            
            # Normaliza o contexto (vírgula -> ponto)
            context_normalized = context.replace(',', '.')
            
            # Variações do handicap para buscar
            h_clean = handicap.lstrip('+')  # Remove + inicial se houver
            h_with_comma = h_clean.replace('.', ',')  # -0.5 -> -0,5
            h_no_sign = h_clean.lstrip('-')  # -0.5 -> 0.5
            h_no_sign_comma = h_no_sign.replace('.', ',')  # 0.5 -> 0,5
            
            # Verifica se alguma variação está no contexto
            # Importante: deve estar no início do contexto (ex: "-0.5home" ou "-0,5home")
            for variant in [h_clean, h_with_comma]:
                # Busca por padrão que indica início da linha AH (handicap seguido de "home")
                if f"{variant}home" in context or f"{variant}home" in context_normalized:
                    return True
            
            return False
        
        try:
            # Encontra elementos com a odds
            elements = await self._page.query_selector_all(f"text='{odds_str}'")
            
            logger.debug(f"Buscando odds '{odds_str}' ({side}, AH {handicap}): {len(elements)} elementos encontrados")
            
            # Se temos handicap e múltiplos elementos, filtra/prioriza pelo contexto
            elements_with_context = []
            for el in elements:
                try:
                    # Obtém o contexto do elemento (texto do avô)
                    context = await el.evaluate(
                        "el => (el.parentElement?.parentElement?.textContent || '').substring(0, 100).toLowerCase()"
                    )
                    elements_with_context.append((el, context))
                except:
                    elements_with_context.append((el, ""))
            
            # Se há handicap, ordena para priorizar elementos que contêm o handicap no contexto
            if handicap and len(elements_with_context) > 1:
                def context_priority(item):
                    el, context = item
                    # Prioridade 1: contexto contém o handicap exato (início da linha AH)
                    if handicap_matches_context(handicap, context):
                        return 0
                    # Prioridade 2: contexto contém "home" e "away" sem "draw" (seção AH genérica)
                    if "home" in context and "away" in context and "draw" not in context:
                        return 1
                    # Prioridade 3: outros (provavelmente 1X2 ou outro mercado)
                    return 2
                
                elements_with_context.sort(key=context_priority)
                logger.debug(f"  Elementos reordenados por contexto (handicap={handicap})")
            
            # Tenta cada elemento até encontrar um que abra o painel com bookmakers
            for i, (el, context) in enumerate(elements_with_context):
                try:
                    if await el.is_visible():
                        # Scroll agressivo: primeiro scroll normal, depois via JavaScript
                        await el.scroll_into_view_if_needed()
                        await self._page.wait_for_timeout(200)
                        
                        # Scroll adicional via JavaScript para garantir visibilidade
                        await el.evaluate("el => el.scrollIntoView({block: 'center', behavior: 'instant'})")
                        await self._page.wait_for_timeout(300)
                        
                        # Verifica o bounding box
                        box = await el.bounding_box()
                        if box and box['width'] > 20 and box['height'] > 10:
                            # Log com contexto resumido
                            context_short = context[:40].replace('\n', ' ') if context else 'N/A'
                            is_correct = handicap_matches_context(handicap, context)
                            logger.debug(f"  Elemento [{i}]: y={box['y']:.0f}, match={is_correct}, ctx='{context_short}...'")
                            
                            # Determina tempo de espera baseado no contexto
                            # Elementos da seção AH (match=True) precisam de mais tempo
                            wait_time = 1500 if is_correct else 1000
                            
                            # Tenta clicar no elemento PAI (DIV)
                            parent = await el.evaluate_handle("el => el.parentElement")
                            await parent.click()
                            await self._page.wait_for_timeout(wait_time)
                            
                            # Extrai bookmakers do texto
                            panel_text = await self._page.inner_text("body")
                            bookmakers = self._extract_bookmakers_from_text(panel_text)
                            
                            # Se não encontrou e é o elemento correto, espera mais tempo
                            if len(bookmakers) == 0 and is_correct:
                                logger.debug(f"  Elemento [{i}]: esperando mais tempo (2s)...")
                                await self._page.wait_for_timeout(2000)
                                panel_text = await self._page.inner_text("body")
                                bookmakers = self._extract_bookmakers_from_text(panel_text)
                            
                            # Se ainda não encontrou, tenta clique via JavaScript
                            if len(bookmakers) == 0:
                                logger.debug(f"  Elemento [{i}]: tentando clique via JavaScript...")
                                await el.evaluate("el => el.parentElement.click()")
                                await self._page.wait_for_timeout(1500)
                                panel_text = await self._page.inner_text("body")
                                bookmakers = self._extract_bookmakers_from_text(panel_text)
                            
                            logger.debug(f"  Bookmakers extraídos: {len(bookmakers)} - {list(bookmakers.keys())}")
                            
                            # Fecha o painel
                            await self._page.keyboard.press("Escape")
                            await self._page.wait_for_timeout(300)
                            
                            # Se encontrou bookmakers, sucesso!
                            if len(bookmakers) > 0:
                                return bookmakers
                            else:
                                logger.debug(f"  Elemento [{i}]: painel não abriu, tentando próximo...")
                        else:
                            logger.debug(f"  Elemento [{i}]: box inválido {box}")
                except Exception as e:
                    logger.debug(f"  Elemento [{i}]: erro {e}")
                    continue
            
            logger.debug(f"Nenhum elemento abriu painel de bookmakers para odds {odds_str} ({side})")
            
        except Exception as e:
            logger.debug(f"Erro ao clicar em odds {side}: {e}")
            
        return bookmakers
    
    def _extract_bookmakers_from_text(self, text: str) -> Dict[str, dict]:
        """
        Extrai bookmakers e suas odds do texto do painel.
        
        O painel mostra:
        bookmaker_name
        odds
        $limit
        """
        bookmakers = {}
        lines = text.split("\n")
        
        for i, line in enumerate(lines):
            line_clean = line.strip().lower()
            
            # Remove 'e' do final (3ete -> 3et, pin88e -> pin88)
            if line_clean.endswith("e"):
                line_clean_no_e = line_clean[:-1]
                if line_clean_no_e in self.KNOWN_BOOKMAKERS:
                    line_clean = line_clean_no_e
            
            # Verifica se é um bookmaker conhecido
            if line_clean in self.KNOWN_BOOKMAKERS:
                if i + 1 < len(lines):
                    try:
                        odds_str = lines[i + 1].strip().replace(",", ".")
                        odds = float(odds_str)
                        
                        # Próxima linha pode ser o limite
                        limit = 0.0
                        if i + 2 < len(lines):
                            limit_str = lines[i + 2].strip()
                            if limit_str.startswith("$"):
                                limit = float(limit_str.replace("$", "").replace(",", ""))
                        
                        # Evita duplicatas (pega a primeira ocorrência)
                        if line_clean not in bookmakers:
                            bookmakers[line_clean] = {
                                "odds": odds,
                                "limit": limit
                            }
                    except (ValueError, IndexError):
                        pass
        
        # Se não encontrou bookmakers pelo método estruturado,
        # tenta método alternativo buscando padrões diferentes
        if len(bookmakers) == 0:
            text_lower = text.lower()
            lines = text.split("\n")
            
            # Método 2: busca bookmaker seguido de número em qualquer formato
            import re
            for i, line in enumerate(lines):
                line_clean = line.strip().lower().rstrip("e")
                if line_clean in self.KNOWN_BOOKMAKERS:
                    # Procura odds nas próximas linhas
                    for j in range(1, 4):
                        if i + j < len(lines):
                            next_line = lines[i + j].strip()
                            # Tenta extrair número
                            match = re.match(r'^(\d+[.,]\d+)$', next_line)
                            if match:
                                try:
                                    odds = float(match.group(1).replace(",", "."))
                                    if line_clean not in bookmakers:
                                        bookmakers[line_clean] = {"odds": odds, "limit": 0.0}
                                        break
                                except:
                                    pass
        
        return bookmakers
        
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
