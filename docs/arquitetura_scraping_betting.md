# Arquitetura Completa: Sistema de Scraping + Scoring + Execução de Apostas

## Visão Geral do Sistema

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                           ARQUITETURA COMPLETA DO SISTEMA                               │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
│   │  SCRAPER    │────►│   QUEUE     │────►│  SCORING    │────►│  EXECUTOR   │          │
│   │  (Playwright)│     │  (Celery)   │     │  (FastAPI)  │     │  (Playwright)│         │
│   └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘          │
│          │                   │                   │                   │                  │
│          ▼                   ▼                   ▼                   ▼                  │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
│   │   CACHE     │     │   BROKER    │     │   MODELS    │     │   STORAGE   │          │
│   │  (Redis)    │     │  (Redis)    │     │  (.joblib)  │     │ (PostgreSQL)│          │
│   └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘          │
│                                                                        │                │
│                                                                        ▼                │
│                                                                 ┌─────────────┐         │
│                                                                 │ MONITORING  │         │
│                                                                 │ (Grafana)   │         │
│                                                                 └─────────────┘         │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. SCRAPING (Playwright + Python)

### 1.1 O que é Playwright?

Playwright é uma biblioteca de automação de browser desenvolvida pela Microsoft. É a evolução moderna do Selenium, com melhor performance e API mais limpa.

### 1.2 Por que Playwright e não Selenium?

| Aspecto | Selenium | Playwright |
|---------|----------|------------|
| **Performance** | Mais lento | 2-3x mais rápido |
| **Async nativo** | Não | Sim |
| **Auto-wait** | Manual | Automático |
| **Debugging** | Difícil | Inspector visual |
| **Manutenção** | Alta | Baixa |

### 1.3 Complexidade Técnica

```
┌─────────────────────────────────────────────────────────────────┐
│ NÍVEL DE COMPLEXIDADE: ████████░░ 8/10                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Desafios principais:                                             │
│ ├── Entender estrutura HTML/JS do BetinAsia                     │
│ ├── Lidar com carregamento dinâmico (AJAX/WebSocket)            │
│ ├── Manter sessão de login ativa                                │
│ ├── Detectar e contornar anti-bot (se existir)                  │
│ ├── Tratar mudanças no layout do site                           │
│ └── Gerenciar múltiplas abas/contextos                          │
│                                                                  │
│ Tempo estimado de desenvolvimento: 2-4 semanas                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.4 Custo

| Item | Custo |
|------|-------|
| **Playwright** | Gratuito (open source) |
| **Python** | Gratuito |
| **Servidor para rodar** | R$ 50-200/mês (VPS Linux) |

### 1.5 Estrutura de Código

```python
# scraper/betinasia_scraper.py

import asyncio
from playwright.async_api import async_playwright, Browser, Page
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class BookmakerOdds:
    """Odds de uma casa específica para uma linha de AH."""
    bookmaker: str
    home_odds: float
    away_odds: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    
@dataclass
class AHLine:
    """Uma linha de Asian Handicap com odds de múltiplas casas."""
    line: str  # Ex: "+0.5", "-0.75", "+0"
    bookmaker_odds: dict[str, BookmakerOdds] = field(default_factory=dict)
    
    @property
    def best_home_odds(self) -> tuple[str, float]:
        """Retorna (bookmaker, odds) com melhor odd para home."""
        best = max(self.bookmaker_odds.items(), key=lambda x: x[1].home_odds)
        return best[0], best[1].home_odds
    
    @property
    def best_away_odds(self) -> tuple[str, float]:
        """Retorna (bookmaker, odds) com melhor odd para away."""
        best = max(self.bookmaker_odds.items(), key=lambda x: x[1].away_odds)
        return best[0], best[1].away_odds
    
    @property
    def num_bookmakers(self) -> int:
        return len(self.bookmaker_odds)


@dataclass
class MatchData:
    """Dados completos de uma partida."""
    match_id: str
    league: str
    home_team: str
    away_team: str
    kickoff_time: datetime
    ah_lines: dict[str, AHLine] = field(default_factory=dict)
    scraped_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def minutes_to_kickoff(self) -> int:
        delta = self.kickoff_time - datetime.now(timezone.utc)
        return max(0, int(delta.total_seconds() / 60))


class BetinAsiaScraper:
    """
    Scraper assíncrono para BetinAsia.
    
    Uso:
        async with BetinAsiaScraper() as scraper:
            await scraper.login(username, password)
            matches = await scraper.scrape_league("England Premier League")
    """
    
    BASE_URL = "https://www.betinasia.com"
    
    def __init__(
        self,
        headless: bool = True,
        slow_mo: int = 0,  # ms entre ações (útil para debug)
    ):
        self.headless = headless
        self.slow_mo = slow_mo
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context = None
        self._page: Optional[Page] = None
        
    async def __aenter__(self):
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        
    async def start(self):
        """Inicia o browser."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo,
        )
        # Contexto persistente para manter cookies/sessão
        self._context = await self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        )
        self._page = await self._context.new_page()
        
    async def close(self):
        """Fecha o browser."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
            
    async def login(self, username: str, password: str) -> bool:
        """
        Faz login no BetinAsia.
        
        Returns:
            True se login bem-sucedido, False caso contrário.
        """
        try:
            await self._page.goto(f"{self.BASE_URL}/login")
            
            # Aguarda campo de usuário aparecer
            await self._page.wait_for_selector("input[name='username']", timeout=10000)
            
            # Preenche credenciais
            await self._page.fill("input[name='username']", username)
            await self._page.fill("input[name='password']", password)
            
            # Clica no botão de login
            await self._page.click("button[type='submit']")
            
            # Aguarda redirecionamento ou elemento de dashboard
            await self._page.wait_for_selector(".dashboard", timeout=15000)
            
            logger.info("Login bem-sucedido no BetinAsia")
            return True
            
        except Exception as e:
            logger.error(f"Falha no login: {e}")
            return False
            
    async def scrape_league(self, league_name: str) -> list[MatchData]:
        """
        Scrape todas as partidas de uma liga específica.
        
        Args:
            league_name: Nome da liga (ex: "England Premier League")
            
        Returns:
            Lista de MatchData com odds de AH
        """
        matches = []
        
        try:
            # Navega para a página de futebol
            await self._page.goto(f"{self.BASE_URL}/odds/football")
            
            # Expande a liga específica
            league_selector = f"[data-league='{league_name}']"
            await self._page.click(league_selector)
            
            # Aguarda carregamento das partidas
            await self._page.wait_for_selector(".match-row", timeout=10000)
            
            # Encontra todas as partidas
            match_elements = await self._page.query_selector_all(".match-row")
            
            for match_el in match_elements:
                match_data = await self._parse_match(match_el, league_name)
                if match_data:
                    matches.append(match_data)
                    
        except Exception as e:
            logger.error(f"Erro ao scrape liga {league_name}: {e}")
            
        return matches
    
    async def _parse_match(self, match_el, league_name: str) -> Optional[MatchData]:
        """
        Extrai dados de uma partida específica.
        
        NOTA: Os seletores CSS abaixo são EXEMPLOS.
        Você precisará inspecionar o HTML real do BetinAsia
        e ajustar os seletores de acordo.
        """
        try:
            # Extrai informações básicas
            match_id = await match_el.get_attribute("data-match-id")
            home_team = await match_el.query_selector(".home-team")
            away_team = await match_el.query_selector(".away-team")
            kickoff = await match_el.query_selector(".kickoff-time")
            
            home_name = await home_team.inner_text() if home_team else "Unknown"
            away_name = await away_team.inner_text() if away_team else "Unknown"
            kickoff_str = await kickoff.inner_text() if kickoff else ""
            
            # Parse do horário de kickoff
            kickoff_time = self._parse_kickoff_time(kickoff_str)
            
            # Cria objeto de dados
            match_data = MatchData(
                match_id=match_id,
                league=league_name,
                home_team=home_name,
                away_team=away_name,
                kickoff_time=kickoff_time,
            )
            
            # Extrai odds de Asian Handicap
            ah_section = await match_el.query_selector(".asian-handicap-odds")
            if ah_section:
                match_data.ah_lines = await self._parse_ah_odds(ah_section)
                
            return match_data
            
        except Exception as e:
            logger.warning(f"Erro ao parsear partida: {e}")
            return None
            
    async def _parse_ah_odds(self, ah_section) -> dict[str, AHLine]:
        """
        Extrai todas as linhas de AH e odds de cada bookmaker.
        
        Estrutura esperada (exemplo):
        ┌─────────────────────────────────────────────────┐
        │ Line    │ Pin88  │ SBO    │ Sing2  │ ISN     │
        │ +0.5    │ 1.95   │ 1.92   │ 1.93   │ 1.91    │ (home)
        │         │ 1.95   │ 1.98   │ 1.97   │ 1.99    │ (away)
        │ -0.5    │ 2.05   │ 2.02   │ 2.03   │ 2.01    │
        │         │ 1.85   │ 1.88   │ 1.87   │ 1.89    │
        └─────────────────────────────────────────────────┘
        """
        ah_lines = {}
        
        # Encontra todas as linhas de handicap
        line_rows = await ah_section.query_selector_all(".ah-line-row")
        
        for row in line_rows:
            line_value = await row.query_selector(".line-value")
            line_str = await line_value.inner_text() if line_value else None
            
            if not line_str:
                continue
                
            ah_line = AHLine(line=line_str)
            
            # Encontra odds de cada bookmaker
            bookmaker_cells = await row.query_selector_all(".bookmaker-odds")
            
            for cell in bookmaker_cells:
                bookmaker = await cell.get_attribute("data-bookmaker")
                home_odds_el = await cell.query_selector(".home-odds")
                away_odds_el = await cell.query_selector(".away-odds")
                
                if home_odds_el and away_odds_el:
                    home_odds = float(await home_odds_el.inner_text())
                    away_odds = float(await away_odds_el.inner_text())
                    
                    ah_line.bookmaker_odds[bookmaker] = BookmakerOdds(
                        bookmaker=bookmaker,
                        home_odds=home_odds,
                        away_odds=away_odds,
                    )
                    
            ah_lines[line_str] = ah_line
            
        return ah_lines
    
    def _parse_kickoff_time(self, kickoff_str: str) -> datetime:
        """
        Converte string de horário para datetime.
        Ajuste o formato de acordo com o que o BetinAsia usa.
        """
        from dateutil import parser
        try:
            return parser.parse(kickoff_str).replace(tzinfo=timezone.utc)
        except:
            return datetime.now(timezone.utc)


# Exemplo de uso
async def main():
    async with BetinAsiaScraper(headless=False, slow_mo=100) as scraper:
        # Login
        success = await scraper.login("seu_usuario", "sua_senha")
        if not success:
            return
            
        # Scrape Premier League
        matches = await scraper.scrape_league("England Premier League")
        
        for match in matches:
            print(f"\n{match.home_team} vs {match.away_team}")
            print(f"Kickoff em {match.minutes_to_kickoff} minutos")
            
            for line, ah_line in match.ah_lines.items():
                best_bk, best_odds = ah_line.best_home_odds
                print(f"  {line}: {best_odds:.2f} @ {best_bk} ({ah_line.num_bookmakers} casas)")


if __name__ == "__main__":
    asyncio.run(main())
```

### 1.6 Instalação

```bash
# Instalar Playwright
pip install playwright

# Instalar browsers (necessário rodar uma vez)
playwright install chromium

# Dependências adicionais
pip install python-dateutil
```

---

## 2. QUEUE / SCHEDULER (Celery + Redis)

### 2.1 O que é Celery?

Celery é um sistema de filas de tarefas distribuído. Permite:
- Agendar tarefas para execução futura
- Executar tarefas em paralelo
- Reintentar tarefas que falharam
- Monitorar execução

### 2.2 O que é Redis?

Redis é um banco de dados in-memory usado aqui como:
- **Message Broker**: Fila de mensagens entre componentes
- **Result Backend**: Armazena resultados das tarefas
- **Cache**: Armazena odds recentes

### 2.3 Complexidade Técnica

```
┌─────────────────────────────────────────────────────────────────┐
│ NÍVEL DE COMPLEXIDADE: ██████░░░░ 6/10                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Desafios principais:                                             │
│ ├── Configurar Celery + Redis corretamente                      │
│ ├── Definir estratégia de retry/backoff                         │
│ ├── Gerenciar concorrência (quantos workers?)                   │
│ ├── Monitorar filas (dead letter queues)                        │
│ └── Escalar horizontalmente se necessário                       │
│                                                                  │
│ Tempo estimado de configuração: 1-2 dias                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.4 Custo

| Item | Custo |
|------|-------|
| **Celery** | Gratuito (open source) |
| **Redis** | Gratuito (self-hosted) ou R$ 30-100/mês (Redis Cloud) |
| **Alternativa simples** | APScheduler (gratuito, sem necessidade de Redis) |

### 2.5 Estrutura de Código

```python
# queue/celery_config.py

from celery import Celery
from celery.schedules import crontab
import os

# Configuração do Celery
app = Celery(
    'betting_system',
    broker=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
)

# Configurações
app.conf.update(
    # Serialização
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    
    # Timezone
    timezone='UTC',
    enable_utc=True,
    
    # Retry
    task_acks_late=True,  # Só confirma task após execução
    task_reject_on_worker_lost=True,
    
    # Concorrência
    worker_prefetch_multiplier=1,  # Pega 1 task por vez
    worker_concurrency=4,  # 4 workers paralelos
    
    # Limites
    task_time_limit=300,  # 5 minutos max por task
    task_soft_time_limit=240,  # Aviso em 4 minutos
)

# Agendamento periódico (Celery Beat)
app.conf.beat_schedule = {
    # Scrape ligas Tier 1 a cada 30 segundos
    'scrape-tier1-leagues': {
        'task': 'queue.tasks.scrape_tier1',
        'schedule': 30.0,  # segundos
    },
    
    # Scrape ligas Tier 2 a cada 2 minutos
    'scrape-tier2-leagues': {
        'task': 'queue.tasks.scrape_tier2',
        'schedule': 120.0,
    },
    
    # Scrape ligas Tier 3 a cada 5 minutos
    'scrape-tier3-leagues': {
        'task': 'queue.tasks.scrape_tier3',
        'schedule': 300.0,
    },
    
    # Limpeza de cache de odds antigas (a cada hora)
    'cleanup-old-odds': {
        'task': 'queue.tasks.cleanup_old_odds',
        'schedule': 3600.0,
    },
    
    # Relatório diário às 23:59
    'daily-report': {
        'task': 'queue.tasks.generate_daily_report',
        'schedule': crontab(hour=23, minute=59),
    },
}
```

```python
# queue/tasks.py

from queue.celery_config import app
from scraper.betinasia_scraper import BetinAsiaScraper
from scoring.engine import ScoringEngine
from executor.bet_executor import BetExecutor
from storage.database import Database
from cache.redis_cache import OddsCache
import asyncio
import logging

logger = logging.getLogger(__name__)

# Configuração de ligas por tier
TIER_1_LEAGUES = [
    "England Premier League",
    "Germany Bundesliga",
    "Spain La Liga",
    "Italy Serie A",
    "France Ligue 1",
]

TIER_2_LEAGUES = [
    "England Championship",
    "Germany 2. Bundesliga",
    "Netherlands Eredivisie",
    "Portugal Primeira Liga",
]

TIER_3_LEAGUES = [
    "Belgium Pro League",
    "Turkey Super Lig",
    "Brazil Serie A",
]


def run_async(coro):
    """Helper para rodar código async em contexto sync."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    autoretry_for=(Exception,),
)
def scrape_league(self, league_name: str):
    """
    Task para scrape de uma liga específica.
    
    Fluxo:
    1. Scrape odds do BetinAsia
    2. Compara com cache (detecta mudanças)
    3. Se houve mudança significativa → dispara scoring
    4. Atualiza cache
    """
    logger.info(f"Iniciando scrape: {league_name}")
    
    async def _scrape():
        cache = OddsCache()
        db = Database()
        
        async with BetinAsiaScraper() as scraper:
            # Login (usa credenciais de env vars)
            await scraper.login(
                os.getenv('BETINASIA_USER'),
                os.getenv('BETINASIA_PASS')
            )
            
            # Scrape
            matches = await scraper.scrape_league(league_name)
            
            for match in matches:
                # Verifica se odds mudaram significativamente
                for line_str, ah_line in match.ah_lines.items():
                    cache_key = f"{match.match_id}:{line_str}"
                    cached_odds = await cache.get(cache_key)
                    
                    best_bk, best_odds = ah_line.best_home_odds
                    
                    # Detecta oportunidade: odds mudou ou é nova
                    if cached_odds is None or abs(cached_odds - best_odds) > 0.01:
                        # Dispara task de scoring
                        score_opportunity.delay(
                            match_id=match.match_id,
                            league=league_name,
                            home_team=match.home_team,
                            away_team=match.away_team,
                            ah_line=line_str,
                            best_odds=best_odds,
                            best_bookmaker=best_bk,
                            num_bookmakers=ah_line.num_bookmakers,
                            minutes_to_kickoff=match.minutes_to_kickoff,
                        )
                        
                    # Atualiza cache
                    await cache.set(cache_key, best_odds, ttl=300)  # 5 min TTL
                    
                # Persiste no banco para histórico
                await db.save_match_odds(match)
                
    run_async(_scrape())
    logger.info(f"Scrape concluído: {league_name}")


@app.task
def scrape_tier1():
    """Dispara scrape para todas as ligas Tier 1."""
    for league in TIER_1_LEAGUES:
        scrape_league.delay(league)


@app.task
def scrape_tier2():
    """Dispara scrape para todas as ligas Tier 2."""
    for league in TIER_2_LEAGUES:
        scrape_league.delay(league)


@app.task
def scrape_tier3():
    """Dispara scrape para todas as ligas Tier 3."""
    for league in TIER_3_LEAGUES:
        scrape_league.delay(league)


@app.task(
    bind=True,
    max_retries=2,
)
def score_opportunity(
    self,
    match_id: str,
    league: str,
    home_team: str,
    away_team: str,
    ah_line: str,
    best_odds: float,
    best_bookmaker: str,
    num_bookmakers: int,
    minutes_to_kickoff: int,
):
    """
    Task de scoring para uma oportunidade específica.
    
    Se o modelo aprovar (proba >= cutoff), dispara execução.
    """
    logger.info(f"Scoring: {home_team} vs {away_team} [{ah_line}]")
    
    engine = ScoringEngine()
    
    result = engine.score(
        subtipo_aposta=ah_line,
        num_casas=num_bookmakers,
        best_odds=best_odds,
        minutes_to_kickoff=minutes_to_kickoff,
        casa_aposta=best_bookmaker,
    )
    
    # Log do resultado
    logger.info(f"Scoring result: proba={result.proba:.4f}, decision={result.decision}")
    
    # Persiste decisão
    db = Database()
    asyncio.run(db.save_scoring_decision(
        match_id=match_id,
        ah_line=ah_line,
        proba=result.proba,
        decision=result.decision,
    ))
    
    # Se aprovado, executa aposta
    if result.decision:
        execute_bet.delay(
            match_id=match_id,
            league=league,
            home_team=home_team,
            away_team=away_team,
            ah_line=ah_line,
            odds=best_odds,
            bookmaker=best_bookmaker,
            proba=result.proba,
        )


@app.task(
    bind=True,
    max_retries=1,  # Apenas 1 retry para apostas
)
def execute_bet(
    self,
    match_id: str,
    league: str,
    home_team: str,
    away_team: str,
    ah_line: str,
    odds: float,
    bookmaker: str,
    proba: float,
):
    """
    Task de execução de aposta.
    
    CRÍTICO: Esta task deve ter proteções contra:
    - Dupla execução
    - Odds movidas
    - Limites de stake
    """
    logger.warning(f"EXECUTANDO APOSTA: {home_team} vs {away_team} [{ah_line}] @ {odds}")
    
    async def _execute():
        executor = BetExecutor()
        db = Database()
        
        # Verifica se já executamos esta aposta (idempotência)
        existing = await db.get_bet_by_match_line(match_id, ah_line)
        if existing:
            logger.warning(f"Aposta já executada anteriormente: {existing.id}")
            return
        
        # Executa
        result = await executor.place_bet(
            match_id=match_id,
            ah_line=ah_line,
            bookmaker=bookmaker,
            expected_odds=odds,
        )
        
        # Persiste resultado
        await db.save_bet_execution(
            match_id=match_id,
            ah_line=ah_line,
            status=result.status,
            actual_odds=result.actual_odds,
            stake=result.stake,
            confirmation_id=result.confirmation_id,
        )
        
        # Notifica via Telegram (se configurado)
        if result.status == "placed":
            notify_telegram.delay(
                f"✅ Aposta executada!\n"
                f"{home_team} vs {away_team}\n"
                f"Linha: {ah_line}\n"
                f"Odds: {result.actual_odds}\n"
                f"Stake: {result.stake}"
            )
            
    run_async(_execute())


@app.task
def notify_telegram(message: str):
    """Envia notificação via Telegram."""
    import requests
    
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if bot_token and chat_id:
        requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": message}
        )


@app.task
def cleanup_old_odds():
    """Remove odds antigas do cache e banco."""
    logger.info("Executando limpeza de odds antigas")
    # Implementar lógica de limpeza
    

@app.task
def generate_daily_report():
    """Gera relatório diário de P&L."""
    logger.info("Gerando relatório diário")
    # Implementar geração de relatório
```

### 2.6 Comandos para Executar

```bash
# Instalar dependências
pip install celery redis

# Terminal 1: Iniciar Redis (se não estiver rodando)
redis-server

# Terminal 2: Iniciar Celery Worker
celery -A queue.celery_config worker --loglevel=info

# Terminal 3: Iniciar Celery Beat (scheduler)
celery -A queue.celery_config beat --loglevel=info

# Terminal 4: Monitorar filas (opcional)
celery -A queue.celery_config flower
```

### 2.7 Alternativa Simples: APScheduler

Se você preferir algo mais simples (sem Redis):

```python
# scheduler/simple_scheduler.py

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import asyncio

scheduler = AsyncIOScheduler()

# Adiciona jobs
scheduler.add_job(
    scrape_tier1_leagues,
    trigger=IntervalTrigger(seconds=30),
    id='scrape_tier1',
    replace_existing=True,
)

scheduler.add_job(
    scrape_tier2_leagues,
    trigger=IntervalTrigger(seconds=120),
    id='scrape_tier2',
    replace_existing=True,
)

# Inicia
scheduler.start()

# Mantém rodando
asyncio.get_event_loop().run_forever()
```

---

## 3. STORAGE (PostgreSQL)

### 3.1 Por que PostgreSQL?

- **Robusto**: Banco de produção confiável
- **JSON nativo**: Pode armazenar odds complexas como JSONB
- **Queries analíticas**: Bom para relatórios de P&L
- **Gratuito**: Open source

### 3.2 Complexidade Técnica

```
┌─────────────────────────────────────────────────────────────────┐
│ NÍVEL DE COMPLEXIDADE: ████░░░░░░ 4/10                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Desafios principais:                                             │
│ ├── Definir schema adequado                                     │
│ ├── Criar índices para queries frequentes                       │
│ ├── Configurar backups                                          │
│ └── Otimizar para volume de dados                               │
│                                                                  │
│ Tempo estimado de setup: 1 dia                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Custo

| Opção | Custo |
|-------|-------|
| **Self-hosted** | Gratuito (apenas custo do servidor) |
| **Supabase** | Gratuito até 500MB, depois ~$25/mês |
| **AWS RDS** | ~$15-50/mês (t3.micro) |
| **Railway** | ~$5-20/mês |

### 3.4 Schema do Banco

```sql
-- migrations/001_initial_schema.sql

-- Tabela de partidas
CREATE TABLE matches (
    id SERIAL PRIMARY KEY,
    external_id VARCHAR(100) UNIQUE NOT NULL,  -- ID do BetinAsia
    league VARCHAR(200) NOT NULL,
    home_team VARCHAR(200) NOT NULL,
    away_team VARCHAR(200) NOT NULL,
    kickoff_time TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Índices
    INDEX idx_matches_league (league),
    INDEX idx_matches_kickoff (kickoff_time)
);

-- Tabela de odds (histórico)
CREATE TABLE odds_history (
    id SERIAL PRIMARY KEY,
    match_id INTEGER REFERENCES matches(id),
    ah_line VARCHAR(20) NOT NULL,  -- "+0.5", "-0.75", etc.
    bookmaker VARCHAR(50) NOT NULL,
    home_odds DECIMAL(5,3) NOT NULL,
    away_odds DECIMAL(5,3) NOT NULL,
    scraped_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Índices
    INDEX idx_odds_match_line (match_id, ah_line),
    INDEX idx_odds_scraped (scraped_at)
);

-- Tabela de oportunidades detectadas
CREATE TABLE opportunities (
    id SERIAL PRIMARY KEY,
    match_id INTEGER REFERENCES matches(id),
    ah_line VARCHAR(20) NOT NULL,
    best_odds DECIMAL(5,3) NOT NULL,
    best_bookmaker VARCHAR(50) NOT NULL,
    num_bookmakers INTEGER NOT NULL,
    detected_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Dados do scoring
    proba DECIMAL(6,5),
    cutoff DECIMAL(4,3),
    decision BOOLEAN,
    scored_at TIMESTAMPTZ,
    
    INDEX idx_opp_match (match_id),
    INDEX idx_opp_decision (decision)
);

-- Tabela de apostas executadas
CREATE TABLE bets (
    id SERIAL PRIMARY KEY,
    opportunity_id INTEGER REFERENCES opportunities(id),
    match_id INTEGER REFERENCES matches(id),
    ah_line VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,  -- 'home' ou 'away'
    bookmaker VARCHAR(50) NOT NULL,
    
    -- Execução
    expected_odds DECIMAL(5,3) NOT NULL,
    actual_odds DECIMAL(5,3),
    stake DECIMAL(10,2) NOT NULL,
    potential_return DECIMAL(10,2),
    
    -- Status
    status VARCHAR(20) NOT NULL,  -- 'pending', 'placed', 'rejected', 'cancelled'
    confirmation_id VARCHAR(100),
    error_message TEXT,
    
    -- Resultado
    result VARCHAR(20),  -- 'win', 'loss', 'half_win', 'half_loss', 'push', 'pending'
    profit_loss DECIMAL(10,2),
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    executed_at TIMESTAMPTZ,
    settled_at TIMESTAMPTZ,
    
    -- Prevenção de duplicatas
    UNIQUE (match_id, ah_line, side, bookmaker)
);

-- Tabela de configuração de ligas
CREATE TABLE league_config (
    id SERIAL PRIMARY KEY,
    league_name VARCHAR(200) UNIQUE NOT NULL,
    tier INTEGER NOT NULL CHECK (tier BETWEEN 1 AND 3),
    scrape_interval_seconds INTEGER NOT NULL DEFAULT 60,
    is_active BOOLEAN DEFAULT TRUE,
    priority_score DECIMAL(5,2),  -- Baseado no histórico
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Tabela de métricas diárias
CREATE TABLE daily_metrics (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    
    -- Volume
    total_opportunities INTEGER DEFAULT 0,
    total_bets INTEGER DEFAULT 0,
    
    -- P&L
    total_stake DECIMAL(12,2) DEFAULT 0,
    total_profit_loss DECIMAL(12,2) DEFAULT 0,
    roi_percent DECIMAL(6,3),
    
    -- Por resultado
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    pushes INTEGER DEFAULT 0,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Views úteis
CREATE VIEW v_pending_bets AS
SELECT 
    b.*,
    m.home_team,
    m.away_team,
    m.league,
    m.kickoff_time
FROM bets b
JOIN matches m ON b.match_id = m.id
WHERE b.result = 'pending'
ORDER BY m.kickoff_time;

CREATE VIEW v_daily_pnl AS
SELECT 
    DATE(b.settled_at) as date,
    COUNT(*) as total_bets,
    SUM(b.stake) as total_stake,
    SUM(b.profit_loss) as total_pnl,
    ROUND(SUM(b.profit_loss) / NULLIF(SUM(b.stake), 0) * 100, 2) as roi_percent
FROM bets b
WHERE b.result != 'pending'
GROUP BY DATE(b.settled_at)
ORDER BY date DESC;
```

### 3.5 Código de Acesso ao Banco

```python
# storage/database.py

import asyncpg
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
import os


@dataclass
class BetRecord:
    id: int
    match_id: int
    ah_line: str
    bookmaker: str
    stake: float
    status: str
    result: Optional[str]
    profit_loss: Optional[float]


class Database:
    """
    Classe de acesso ao banco PostgreSQL usando asyncpg.
    
    Uso:
        db = Database()
        await db.connect()
        await db.save_match(...)
        await db.close()
    """
    
    def __init__(self):
        self.pool = None
        
    async def connect(self):
        """Cria pool de conexões."""
        self.pool = await asyncpg.create_pool(
            dsn=os.getenv('DATABASE_URL', 'postgresql://localhost/betting'),
            min_size=2,
            max_size=10,
        )
        
    async def close(self):
        """Fecha pool de conexões."""
        if self.pool:
            await self.pool.close()
            
    async def save_match(self, match_data) -> int:
        """
        Salva ou atualiza uma partida.
        Retorna o ID da partida.
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO matches (external_id, league, home_team, away_team, kickoff_time)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (external_id) DO UPDATE SET
                    league = EXCLUDED.league,
                    home_team = EXCLUDED.home_team,
                    away_team = EXCLUDED.away_team,
                    kickoff_time = EXCLUDED.kickoff_time
                RETURNING id
            """, match_data.match_id, match_data.league, 
                match_data.home_team, match_data.away_team, match_data.kickoff_time)
            
            return row['id']
            
    async def save_odds(self, match_id: int, ah_line: str, bookmaker: str, 
                        home_odds: float, away_odds: float):
        """Salva registro de odds no histórico."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO odds_history (match_id, ah_line, bookmaker, home_odds, away_odds)
                VALUES ($1, $2, $3, $4, $5)
            """, match_id, ah_line, bookmaker, home_odds, away_odds)
            
    async def save_opportunity(self, match_id: int, ah_line: str, best_odds: float,
                               best_bookmaker: str, num_bookmakers: int) -> int:
        """Salva oportunidade detectada."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO opportunities (match_id, ah_line, best_odds, best_bookmaker, num_bookmakers)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id
            """, match_id, ah_line, best_odds, best_bookmaker, num_bookmakers)
            
            return row['id']
            
    async def update_opportunity_scoring(self, opp_id: int, proba: float, 
                                         cutoff: float, decision: bool):
        """Atualiza oportunidade com resultado do scoring."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE opportunities SET
                    proba = $2,
                    cutoff = $3,
                    decision = $4,
                    scored_at = NOW()
                WHERE id = $1
            """, opp_id, proba, cutoff, decision)
            
    async def save_bet(self, opportunity_id: int, match_id: int, ah_line: str,
                       side: str, bookmaker: str, expected_odds: float,
                       stake: float) -> int:
        """Salva aposta a ser executada."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO bets (
                    opportunity_id, match_id, ah_line, side, bookmaker,
                    expected_odds, stake, potential_return, status
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'pending')
                RETURNING id
            """, opportunity_id, match_id, ah_line, side, bookmaker,
                expected_odds, stake, stake * expected_odds)
            
            return row['id']
            
    async def update_bet_executed(self, bet_id: int, actual_odds: float,
                                  confirmation_id: str):
        """Atualiza aposta como executada."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE bets SET
                    status = 'placed',
                    actual_odds = $2,
                    confirmation_id = $3,
                    executed_at = NOW()
                WHERE id = $1
            """, bet_id, actual_odds, confirmation_id)
            
    async def update_bet_result(self, bet_id: int, result: str, profit_loss: float):
        """Atualiza resultado final da aposta."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE bets SET
                    result = $2,
                    profit_loss = $3,
                    settled_at = NOW()
                WHERE id = $1
            """, bet_id, result, profit_loss)
            
    async def get_bet_by_match_line(self, match_id: int, ah_line: str) -> Optional[BetRecord]:
        """Verifica se já existe aposta para esta partida/linha."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM bets
                WHERE match_id = $1 AND ah_line = $2
                ORDER BY created_at DESC
                LIMIT 1
            """, match_id, ah_line)
            
            if row:
                return BetRecord(**dict(row))
            return None
            
    async def get_pending_bets(self) -> List[BetRecord]:
        """Retorna apostas pendentes de resultado."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM v_pending_bets
            """)
            return [BetRecord(**dict(row)) for row in rows]
            
    async def get_daily_pnl(self, days: int = 30) -> List[dict]:
        """Retorna P&L diário dos últimos N dias."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM v_daily_pnl
                WHERE date >= CURRENT_DATE - $1
            """, days)
            return [dict(row) for row in rows]
```

---

## 4. CACHE (Redis)

### 4.1 Uso do Cache

O cache é crítico para:
1. **Detectar mudanças de odds** (comparar com valor anterior)
2. **Evitar scraping redundante** (se odds não mudou, não precisa processar)
3. **Rate limiting** (controlar frequência de requests)

### 4.2 Complexidade Técnica

```
┌─────────────────────────────────────────────────────────────────┐
│ NÍVEL DE COMPLEXIDADE: ███░░░░░░░ 3/10                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Desafios principais:                                             │
│ ├── Definir TTL adequado para cada tipo de dado                 │
│ ├── Serialização de objetos complexos                           │
│ └── Monitorar uso de memória                                    │
│                                                                  │
│ Tempo estimado de setup: meio dia                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Custo

| Opção | Custo |
|-------|-------|
| **Self-hosted** | Gratuito |
| **Redis Cloud** | Gratuito até 30MB, $5-30/mês depois |
| **Upstash** | Gratuito até 10K requests/dia |

### 4.4 Código do Cache

```python
# cache/redis_cache.py

import redis.asyncio as redis
import json
from datetime import datetime, timezone
from typing import Optional, Any
import os


class OddsCache:
    """
    Cache de odds usando Redis.
    
    Estrutura de chaves:
    - odds:{match_id}:{ah_line} -> última odd conhecida
    - odds:{match_id}:{ah_line}:history -> lista de odds recentes
    - rate_limit:{resource} -> contador de requests
    """
    
    DEFAULT_TTL = 300  # 5 minutos
    
    def __init__(self):
        self.redis = redis.from_url(
            os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
            decode_responses=True,
        )
        
    async def close(self):
        await self.redis.close()
        
    # ==================== ODDS ====================
    
    async def get_odds(self, match_id: str, ah_line: str) -> Optional[float]:
        """Retorna última odd conhecida para uma linha."""
        key = f"odds:{match_id}:{ah_line}"
        value = await self.redis.get(key)
        return float(value) if value else None
        
    async def set_odds(self, match_id: str, ah_line: str, odds: float, 
                       ttl: int = DEFAULT_TTL):
        """Salva odd no cache com TTL."""
        key = f"odds:{match_id}:{ah_line}"
        await self.redis.setex(key, ttl, str(odds))
        
        # Também adiciona ao histórico recente
        history_key = f"odds:{match_id}:{ah_line}:history"
        entry = json.dumps({
            "odds": odds,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        await self.redis.lpush(history_key, entry)
        await self.redis.ltrim(history_key, 0, 99)  # Mantém últimas 100 entradas
        await self.redis.expire(history_key, 3600)  # 1 hora
        
    async def get_odds_history(self, match_id: str, ah_line: str, 
                               limit: int = 10) -> list[dict]:
        """Retorna histórico recente de odds."""
        key = f"odds:{match_id}:{ah_line}:history"
        entries = await self.redis.lrange(key, 0, limit - 1)
        return [json.loads(e) for e in entries]
        
    async def odds_changed_significantly(self, match_id: str, ah_line: str,
                                         new_odds: float, threshold: float = 0.02) -> bool:
        """
        Verifica se odds mudou significativamente.
        
        Args:
            threshold: Diferença mínima para considerar mudança (default 2%)
        """
        old_odds = await self.get_odds(match_id, ah_line)
        
        if old_odds is None:
            return True  # Nova odd, considerar como mudança
            
        change_pct = abs(new_odds - old_odds) / old_odds
        return change_pct >= threshold
        
    # ==================== RATE LIMITING ====================
    
    async def check_rate_limit(self, resource: str, max_requests: int,
                               window_seconds: int) -> bool:
        """
        Verifica se pode fazer request (rate limiting).
        
        Returns:
            True se pode fazer request, False se deve esperar.
        """
        key = f"rate_limit:{resource}"
        
        current = await self.redis.incr(key)
        
        if current == 1:
            # Primeira request, define TTL
            await self.redis.expire(key, window_seconds)
            
        return current <= max_requests
        
    async def get_rate_limit_remaining(self, resource: str, 
                                       max_requests: int) -> int:
        """Retorna quantas requests ainda podem ser feitas."""
        key = f"rate_limit:{resource}"
        current = await self.redis.get(key)
        current = int(current) if current else 0
        return max(0, max_requests - current)
        
    # ==================== SESSÃO ====================
    
    async def save_session(self, session_id: str, data: dict, ttl: int = 3600):
        """Salva dados de sessão (cookies, tokens)."""
        key = f"session:{session_id}"
        await self.redis.setex(key, ttl, json.dumps(data))
        
    async def get_session(self, session_id: str) -> Optional[dict]:
        """Recupera dados de sessão."""
        key = f"session:{session_id}"
        data = await self.redis.get(key)
        return json.loads(data) if data else None
        
    # ==================== LOCKS ====================
    
    async def acquire_lock(self, resource: str, ttl: int = 30) -> bool:
        """
        Tenta adquirir lock distribuído.
        Útil para evitar execução duplicada.
        """
        key = f"lock:{resource}"
        acquired = await self.redis.set(key, "1", nx=True, ex=ttl)
        return bool(acquired)
        
    async def release_lock(self, resource: str):
        """Libera lock."""
        key = f"lock:{resource}"
        await self.redis.delete(key)
        
    # ==================== MÉTRICAS ====================
    
    async def increment_metric(self, metric: str, amount: int = 1):
        """Incrementa contador de métrica."""
        key = f"metric:{metric}:{datetime.now().strftime('%Y-%m-%d')}"
        await self.redis.incrby(key, amount)
        await self.redis.expire(key, 86400 * 7)  # 7 dias
        
    async def get_metric(self, metric: str, date: str = None) -> int:
        """Retorna valor de métrica."""
        date = date or datetime.now().strftime('%Y-%m-%d')
        key = f"metric:{metric}:{date}"
        value = await self.redis.get(key)
        return int(value) if value else 0


# Exemplo de uso
async def example_usage():
    cache = OddsCache()
    
    # Verificar se odds mudou
    match_id = "12345"
    ah_line = "+0.5"
    new_odds = 1.95
    
    if await cache.odds_changed_significantly(match_id, ah_line, new_odds):
        print("Odds mudou! Processar...")
        await cache.set_odds(match_id, ah_line, new_odds)
    else:
        print("Odds estável, ignorar.")
        
    # Rate limiting
    if await cache.check_rate_limit("betinasia_api", max_requests=60, window_seconds=60):
        print("Pode fazer request")
    else:
        print("Rate limit excedido, aguardar")
        
    # Lock para evitar dupla execução
    if await cache.acquire_lock(f"bet:{match_id}:{ah_line}"):
        try:
            print("Executando aposta...")
            # ... executa aposta ...
        finally:
            await cache.release_lock(f"bet:{match_id}:{ah_line}")
    else:
        print("Aposta já está sendo executada")
        
    await cache.close()
```

---

## 5. SCORING ENGINE (Nova Versão - Sem RebelBetting)

### 5.1 Nova Abordagem

Como você não terá mais a feature `Dif Odds RB & BIA` (diferença entre RebelBetting e BetinAsia), precisamos:

1. **Coletar novos dados** diretamente do BetinAsia
2. **Criar novas features** baseadas apenas no que o BetinAsia fornece
3. **Treinar novo modelo** de regressão logística

### 5.2 Novas Features Propostas

```python
# scoring/features.py

"""
Features para o novo modelo (sem RebelBetting).

Baseado nos dados disponíveis diretamente do BetinAsia.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List
import numpy as np


@dataclass
class ScoringFeatures:
    """Features para o modelo de scoring."""
    
    # ===== FEATURES DE MERCADO =====
    
    # Número de bookmakers oferecendo a linha
    num_bookmakers: int
    
    # Diferença percentual entre melhor odd e segunda melhor
    dif_pct_best_second: float
    
    # Diferença percentual entre melhor odd e mediana
    dif_pct_best_median: float
    
    # Diferença entre melhor odd e Pinnacle (benchmark)
    # Pinnacle é considerado o mercado mais eficiente
    dif_vs_pinnacle: float
    
    # Spread entre home e away odds (indica força da linha)
    home_away_spread: float
    
    # ===== FEATURES TEMPORAIS =====
    
    # Minutos até o início da partida
    minutes_to_kickoff: int
    
    # Dia da semana (0=segunda, 6=domingo)
    weekday: int
    
    # Turno: 0=madrugada, 1=manhã, 2=tarde, 3=noite
    turno: int
    
    # ===== FEATURES CATEGÓRICAS =====
    
    # Linha de Asian Handicap (ex: "+0.5", "-0.75")
    ah_line: str
    
    # Bookmaker com melhor odd
    best_bookmaker: str
    
    # Liga/Competição
    league: str
    
    # ===== FEATURES DERIVADAS =====
    
    # Volatilidade recente (desvio padrão das últimas N odds)
    odds_volatility: float
    
    # Tendência de movimento (odds subindo ou descendo)
    odds_trend: float  # Positivo = subindo, negativo = descendo
    
    # Concentração de liquidez (% do mercado no top 3 bookmakers)
    liquidity_concentration: float


def calculate_features(
    ah_line_data,  # AHLine do scraper
    match_data,    # MatchData do scraper
    odds_history: List[float],
    pinnacle_odds: float = None,
) -> ScoringFeatures:
    """
    Calcula todas as features a partir dos dados scrapeados.
    
    Args:
        ah_line_data: Dados da linha de AH com odds de múltiplos bookmakers
        match_data: Dados da partida
        odds_history: Histórico recente de odds (últimos 10-20 valores)
        pinnacle_odds: Odds da Pinnacle (se disponível)
    """
    
    # Extrai odds de todos os bookmakers
    all_odds = [bk.home_odds for bk in ah_line_data.bookmaker_odds.values()]
    sorted_odds = sorted(all_odds, reverse=True)
    
    # Features de mercado
    num_bookmakers = len(sorted_odds)
    best_odds = sorted_odds[0] if sorted_odds else 0
    second_best = sorted_odds[1] if len(sorted_odds) > 1 else best_odds
    median_odds = np.median(all_odds) if all_odds else 0
    
    dif_pct_best_second = (best_odds - second_best) / second_best * 100 if second_best > 0 else 0
    dif_pct_best_median = (best_odds - median_odds) / median_odds * 100 if median_odds > 0 else 0
    
    # Diferença vs Pinnacle (se disponível)
    if pinnacle_odds:
        dif_vs_pinnacle = (best_odds - pinnacle_odds) / pinnacle_odds * 100
    else:
        # Usa Pinnacle do dict se disponível
        pin_bk = ah_line_data.bookmaker_odds.get('pin88') or ah_line_data.bookmaker_odds.get('pinnacle')
        if pin_bk:
            dif_vs_pinnacle = (best_odds - pin_bk.home_odds) / pin_bk.home_odds * 100
        else:
            dif_vs_pinnacle = 0
            
    # Spread home/away
    best_home_bk, best_home_odds = ah_line_data.best_home_odds
    best_away_bk, best_away_odds = ah_line_data.best_away_odds
    home_away_spread = best_home_odds - best_away_odds
    
    # Features temporais
    now = datetime.now(timezone.utc)
    minutes_to_kickoff = match_data.minutes_to_kickoff
    weekday = now.weekday()
    hour = now.hour
    turno = 0 if hour < 6 else (1 if hour < 12 else (2 if hour < 18 else 3))
    
    # Volatilidade e tendência
    if odds_history and len(odds_history) >= 3:
        odds_volatility = np.std(odds_history)
        # Tendência: regressão linear simples
        x = np.arange(len(odds_history))
        slope, _ = np.polyfit(x, odds_history, 1)
        odds_trend = slope
    else:
        odds_volatility = 0
        odds_trend = 0
        
    # Concentração de liquidez (simplificado: top 3 vs total)
    if num_bookmakers >= 3:
        top3_avg = np.mean(sorted_odds[:3])
        total_avg = np.mean(sorted_odds)
        liquidity_concentration = top3_avg / total_avg if total_avg > 0 else 1
    else:
        liquidity_concentration = 1
        
    return ScoringFeatures(
        num_bookmakers=num_bookmakers,
        dif_pct_best_second=dif_pct_best_second,
        dif_pct_best_median=dif_pct_best_median,
        dif_vs_pinnacle=dif_vs_pinnacle,
        home_away_spread=home_away_spread,
        minutes_to_kickoff=minutes_to_kickoff,
        weekday=weekday,
        turno=turno,
        ah_line=ah_line_data.line,
        best_bookmaker=best_home_bk,
        league=match_data.league,
        odds_volatility=odds_volatility,
        odds_trend=odds_trend,
        liquidity_concentration=liquidity_concentration,
    )
```

### 5.3 Coleta de Dados para Treinamento

```python
# scoring/data_collection.py

"""
Script para coleta de dados de treinamento.

Roda por 2-4 semanas coletando:
1. Todas as oportunidades detectadas (odds, features)
2. Resultado real das partidas
3. CLV (Closing Line Value) - comparando odds do momento vs odds de fechamento

IMPORTANTE: NÃO apostar durante este período, apenas coletar.
"""

import asyncio
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import pandas as pd

from scraper.betinasia_scraper import BetinAsiaScraper
from scoring.features import calculate_features, ScoringFeatures
from storage.database import Database
from cache.redis_cache import OddsCache


@dataclass
class TrainingRecord:
    """Registro para treinamento do modelo."""
    
    # Identificação
    match_id: str
    league: str
    home_team: str
    away_team: str
    ah_line: str
    
    # Features no momento da detecção
    features: ScoringFeatures
    
    # Odds
    detection_odds: float
    detection_time: datetime
    
    # A ser preenchido depois
    closing_odds: float = None  # Odds de fechamento (última antes do jogo)
    closing_time: datetime = None
    
    # Resultado
    match_result: str = None  # "home_win", "away_win", "draw"
    bet_result: str = None    # "win", "loss", "half_win", "half_loss", "push"
    clv: float = None         # Closing Line Value
    
    
class DataCollector:
    """
    Coletor de dados para treinamento.
    
    Fluxo:
    1. Detecta oportunidades (odds divergentes)
    2. Salva features no momento da detecção
    3. Continua monitorando até o fechamento
    4. Após o jogo, registra resultado
    """
    
    def __init__(self, output_dir: str = "./training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.records: list[TrainingRecord] = []
        
    async def collect_opportunities(self, leagues: list[str], duration_hours: int = 24):
        """
        Coleta oportunidades por um período específico.
        
        Args:
            leagues: Lista de ligas para monitorar
            duration_hours: Duração da coleta em horas
        """
        cache = OddsCache()
        end_time = datetime.now(timezone.utc).timestamp() + (duration_hours * 3600)
        
        async with BetinAsiaScraper() as scraper:
            await scraper.login(
                os.getenv('BETINASIA_USER'),
                os.getenv('BETINASIA_PASS')
            )
            
            while datetime.now(timezone.utc).timestamp() < end_time:
                for league in leagues:
                    try:
                        matches = await scraper.scrape_league(league)
                        
                        for match in matches:
                            # Só considera jogos que começam em menos de 24h
                            if match.minutes_to_kickoff > 1440:
                                continue
                                
                            for line_str, ah_line in match.ah_lines.items():
                                await self._process_opportunity(
                                    match, ah_line, cache
                                )
                                
                    except Exception as e:
                        print(f"Erro ao coletar {league}: {e}")
                        
                # Aguarda antes do próximo ciclo
                await asyncio.sleep(60)  # 1 minuto entre ciclos
                
        # Salva dados coletados
        self._save_records()
        
    async def _process_opportunity(self, match, ah_line, cache):
        """Processa uma oportunidade e salva se relevante."""
        
        # Calcula features
        odds_history = await cache.get_odds_history(match.match_id, ah_line.line)
        history_values = [h['odds'] for h in odds_history]
        
        features = calculate_features(
            ah_line_data=ah_line,
            match_data=match,
            odds_history=history_values,
        )
        
        # Critério de seleção: odds divergiu do mercado
        # Ajuste estes critérios conforme sua estratégia
        should_record = (
            features.num_bookmakers >= 3 and
            features.dif_pct_best_second > 0.5  # 0.5% de divergência mínima
        )
        
        if should_record:
            best_bk, best_odds = ah_line.best_home_odds
            
            record = TrainingRecord(
                match_id=match.match_id,
                league=match.league,
                home_team=match.home_team,
                away_team=match.away_team,
                ah_line=ah_line.line,
                features=features,
                detection_odds=best_odds,
                detection_time=datetime.now(timezone.utc),
            )
            
            self.records.append(record)
            print(f"Registrado: {match.home_team} vs {match.away_team} [{ah_line.line}]")
            
        # Atualiza cache
        best_bk, best_odds = ah_line.best_home_odds
        await cache.set_odds(match.match_id, ah_line.line, best_odds)
        
    def _save_records(self):
        """Salva registros em arquivo."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"training_data_{timestamp}.json"
        
        data = []
        for record in self.records:
            record_dict = asdict(record)
            # Converte datetime para string
            record_dict['detection_time'] = record.detection_time.isoformat()
            record_dict['features'] = asdict(record.features)
            data.append(record_dict)
            
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"Salvos {len(data)} registros em {filepath}")
        
    async def update_closing_odds(self):
        """
        Atualiza odds de fechamento para registros existentes.
        Rodar pouco antes do início das partidas.
        """
        # Implementar lógica de atualização
        pass
        
    async def update_results(self):
        """
        Atualiza resultados das partidas.
        Rodar após as partidas terminarem.
        """
        # Implementar busca de resultados
        pass


# Script de coleta
async def run_data_collection():
    """
    Script principal para coleta de dados.
    
    Recomendação: rodar por 2-4 semanas para ter dados suficientes.
    """
    collector = DataCollector(output_dir="./training_data")
    
    leagues = [
        "England Premier League",
        "Germany Bundesliga",
        "Spain La Liga",
        "Italy Serie A",
        "France Ligue 1",
        "England Championship",
        "Netherlands Eredivisie",
    ]
    
    # Coleta por 24 horas (ajuste conforme necessário)
    await collector.collect_opportunities(leagues, duration_hours=24)
    

if __name__ == "__main__":
    asyncio.run(run_data_collection())
```

### 5.4 Treinamento do Novo Modelo

```python
# scoring/train_model.py

"""
Treinamento do modelo de scoring.

Entrada: dados coletados com features e resultados
Saída: modelo .joblib para produção
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, 
    classification_report, confusion_matrix
)


# Features do modelo
NUM_FEATURES = [
    'num_bookmakers',
    'dif_pct_best_second',
    'dif_pct_best_median',
    'dif_vs_pinnacle',
    'home_away_spread',
    'minutes_to_kickoff',
    'odds_volatility',
    'odds_trend',
    'liquidity_concentration',
]

CAT_FEATURES = [
    'weekday',
    'turno',
    'ah_line',
    'best_bookmaker',
    'league',
]

TARGET = 'clv_positive'  # True se CLV > 0


def load_training_data(data_dir: str) -> pd.DataFrame:
    """
    Carrega e prepara dados de treinamento.
    """
    data_path = Path(data_dir)
    all_records = []
    
    for filepath in data_path.glob("training_data_*.json"):
        with open(filepath) as f:
            records = json.load(f)
            all_records.extend(records)
            
    if not all_records:
        raise ValueError(f"Nenhum dado encontrado em {data_dir}")
        
    # Converte para DataFrame
    rows = []
    for record in all_records:
        if record.get('clv') is None:
            continue  # Pula registros sem CLV calculado
            
        row = {
            'match_id': record['match_id'],
            'detection_time': record['detection_time'],
            **record['features'],
            'clv': record['clv'],
            'clv_positive': record['clv'] > 0,
        }
        rows.append(row)
        
    df = pd.DataFrame(rows)
    
    # Ordena por tempo (importante para split temporal)
    df['detection_time'] = pd.to_datetime(df['detection_time'])
    df = df.sort_values('detection_time')
    
    print(f"Carregados {len(df)} registros para treinamento")
    print(f"Distribuição do target: {df['clv_positive'].value_counts().to_dict()}")
    
    return df


def create_pipeline() -> Pipeline:
    """
    Cria pipeline de pré-processamento + modelo.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ]), NUM_FEATURES),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ]), CAT_FEATURES),
        ]
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',  # Ajusta para desbalanceamento
            C=0.1,  # Regularização moderada
        )),
    ])
    
    return pipeline


def train_and_evaluate(df: pd.DataFrame, output_dir: str):
    """
    Treina modelo com validação temporal e salva.
    """
    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df[TARGET]
    
    # Split temporal (últimos 20% para teste)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
    
    # Cria e treina pipeline
    pipeline = create_pipeline()
    
    # Validação cruzada temporal no conjunto de treino
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=tscv, scoring='roc_auc')
    print(f"\nCV AUC scores: {cv_scores}")
    print(f"CV AUC mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Treina no conjunto de treino completo
    pipeline.fit(X_train, y_train)
    
    # Avalia no conjunto de teste
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    
    test_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nTest AUC: {test_auc:.4f}")
    print(f"\nClassification Report (Test):")
    print(classification_report(y_test, y_pred))
    
    # Análise por cutoff
    print("\nAnálise por cutoff:")
    for cutoff in [0.5, 0.55, 0.6, 0.65, 0.7]:
        y_pred_cut = (y_pred_proba >= cutoff).astype(int)
        precision = precision_score(y_test, y_pred_cut, zero_division=0)
        recall = recall_score(y_test, y_pred_cut, zero_division=0)
        n_positives = y_pred_cut.sum()
        print(f"  Cutoff {cutoff}: Precision={precision:.3f}, Recall={recall:.3f}, N={n_positives}")
    
    # Treina modelo final com todos os dados
    pipeline_final = create_pipeline()
    pipeline_final.fit(X, y)
    
    # Salva modelo
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    model_path = output_path / "model_betinasia_v1.joblib"
    joblib.dump(pipeline_final, model_path)
    print(f"\nModelo salvo em: {model_path}")
    
    # Salva metadados
    metadata = {
        'version': 'v1',
        'training_date': pd.Timestamp.now().isoformat(),
        'num_samples': len(df),
        'features_num': NUM_FEATURES,
        'features_cat': CAT_FEATURES,
        'cv_auc_mean': float(cv_scores.mean()),
        'cv_auc_std': float(cv_scores.std()),
        'test_auc': float(test_auc),
    }
    
    metadata_path = output_path / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    return pipeline_final


def main():
    # Carrega dados
    df = load_training_data("./training_data")
    
    # Treina e avalia
    model = train_and_evaluate(df, "./models")
    

if __name__ == "__main__":
    main()
```

### 5.5 Engine de Scoring para Produção

```python
# scoring/engine.py

"""
Engine de scoring para produção.

Carrega modelo treinado e faz inferência em tempo real.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import json

import numpy as np
import pandas as pd
from joblib import load


@dataclass
class ScoringResult:
    """Resultado do scoring."""
    proba: float
    decision: bool
    cutoff: float
    model_version: str
    scored_at: datetime


class ScoringEngine:
    """
    Engine de scoring para produção.
    
    Uso:
        engine = ScoringEngine("./models")
        result = engine.score(features)
    """
    
    def __init__(self, models_dir: str = "./models", cutoff: float = 0.62):
        self.models_dir = Path(models_dir)
        self.cutoff = cutoff
        self.model = None
        self.metadata = None
        self._load_model()
        
    def _load_model(self):
        """Carrega modelo e metadados."""
        model_path = self.models_dir / "model_betinasia_v1.joblib"
        metadata_path = self.models_dir / "model_metadata.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
            
        self.model = load(model_path)
        
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {'version': 'unknown'}
            
    def score(
        self,
        num_bookmakers: int,
        dif_pct_best_second: float,
        dif_pct_best_median: float,
        dif_vs_pinnacle: float,
        home_away_spread: float,
        minutes_to_kickoff: int,
        weekday: int,
        turno: int,
        ah_line: str,
        best_bookmaker: str,
        league: str,
        odds_volatility: float = 0,
        odds_trend: float = 0,
        liquidity_concentration: float = 1,
    ) -> ScoringResult:
        """
        Executa scoring para uma oportunidade.
        
        Returns:
            ScoringResult com probabilidade e decisão
        """
        # Monta DataFrame com features
        features = pd.DataFrame([{
            'num_bookmakers': num_bookmakers,
            'dif_pct_best_second': dif_pct_best_second,
            'dif_pct_best_median': dif_pct_best_median,
            'dif_vs_pinnacle': dif_vs_pinnacle,
            'home_away_spread': home_away_spread,
            'minutes_to_kickoff': minutes_to_kickoff,
            'weekday': weekday,
            'turno': turno,
            'ah_line': ah_line,
            'best_bookmaker': best_bookmaker,
            'league': league,
            'odds_volatility': odds_volatility,
            'odds_trend': odds_trend,
            'liquidity_concentration': liquidity_concentration,
        }])
        
        # Inferência
        proba = float(self.model.predict_proba(features)[0, 1])
        decision = proba >= self.cutoff
        
        return ScoringResult(
            proba=proba,
            decision=decision,
            cutoff=self.cutoff,
            model_version=self.metadata.get('version', 'unknown'),
            scored_at=datetime.now(timezone.utc),
        )
        
    def score_from_features(self, features) -> ScoringResult:
        """
        Scoring a partir de objeto ScoringFeatures.
        """
        return self.score(
            num_bookmakers=features.num_bookmakers,
            dif_pct_best_second=features.dif_pct_best_second,
            dif_pct_best_median=features.dif_pct_best_median,
            dif_vs_pinnacle=features.dif_vs_pinnacle,
            home_away_spread=features.home_away_spread,
            minutes_to_kickoff=features.minutes_to_kickoff,
            weekday=features.weekday,
            turno=features.turno,
            ah_line=features.ah_line,
            best_bookmaker=features.best_bookmaker,
            league=features.league,
            odds_volatility=features.odds_volatility,
            odds_trend=features.odds_trend,
            liquidity_concentration=features.liquidity_concentration,
        )
```

### 5.6 Complexidade do Scoring

```
┌─────────────────────────────────────────────────────────────────┐
│ NÍVEL DE COMPLEXIDADE: ███████░░░ 7/10                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Desafios principais:                                             │
│ ├── Coletar dados suficientes (2-4 semanas)                     │
│ ├── Obter resultados das partidas (web scraping adicional)      │
│ ├── Calcular CLV corretamente (closing odds)                    │
│ ├── Evitar overfitting (validação temporal)                     │
│ ├── Definir cutoff ótimo                                        │
│ └── Monitorar performance em produção (drift)                   │
│                                                                  │
│ Tempo estimado:                                                  │
│ - Coleta de dados: 2-4 semanas                                   │
│ - Treinamento + validação: 2-3 dias                              │
│ - Integração: 1-2 dias                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. MONITORING (Grafana + Prometheus)

### 6.1 O que é Prometheus?

Prometheus é um sistema de monitoramento que:
- Coleta métricas via HTTP
- Armazena em time-series database
- Permite queries com PromQL

### 6.2 O que é Grafana?

Grafana é uma plataforma de visualização que:
- Cria dashboards interativos
- Conecta a múltiplas fontes de dados
- Permite configurar alertas

### 6.3 Complexidade Técnica

```
┌─────────────────────────────────────────────────────────────────┐
│ NÍVEL DE COMPLEXIDADE: █████░░░░░ 5/10                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Desafios principais:                                             │
│ ├── Configurar Prometheus para coletar métricas                 │
│ ├── Definir métricas relevantes                                 │
│ ├── Criar dashboards informativos                               │
│ └── Configurar alertas                                          │
│                                                                  │
│ Tempo estimado de setup: 1-2 dias                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.4 Custo

| Opção | Custo |
|-------|-------|
| **Self-hosted** | Gratuito (Prometheus + Grafana são open source) |
| **Grafana Cloud** | Gratuito até 10K métricas, ~$15/mês depois |
| **Alternativa simples** | Logs estruturados (JSONL) + scripts de análise |

### 6.5 Métricas Importantes

```python
# monitoring/metrics.py

"""
Métricas para monitoramento do sistema.
"""

from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time


# ===== MÉTRICAS DE SCRAPING =====

SCRAPE_REQUESTS = Counter(
    'betting_scrape_requests_total',
    'Total de requests de scraping',
    ['league', 'status']  # status: success, error
)

SCRAPE_DURATION = Histogram(
    'betting_scrape_duration_seconds',
    'Duração do scraping por liga',
    ['league'],
    buckets=[1, 2, 5, 10, 30, 60]
)

SCRAPE_MATCHES_FOUND = Gauge(
    'betting_scrape_matches_found',
    'Número de partidas encontradas',
    ['league']
)


# ===== MÉTRICAS DE SCORING =====

SCORING_REQUESTS = Counter(
    'betting_scoring_requests_total',
    'Total de requests de scoring',
    ['decision']  # decision: approve, reject
)

SCORING_PROBA = Histogram(
    'betting_scoring_proba',
    'Distribuição das probabilidades do modelo',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
)

SCORING_DURATION = Histogram(
    'betting_scoring_duration_seconds',
    'Duração da inferência',
    buckets=[0.01, 0.05, 0.1, 0.5, 1]
)


# ===== MÉTRICAS DE EXECUÇÃO =====

BETS_TOTAL = Counter(
    'betting_bets_total',
    'Total de apostas',
    ['status', 'bookmaker']  # status: placed, rejected, error
)

BET_STAKE = Histogram(
    'betting_bet_stake',
    'Distribuição de stakes',
    buckets=[10, 25, 50, 100, 250, 500, 1000]
)


# ===== MÉTRICAS DE P&L =====

DAILY_PNL = Gauge(
    'betting_daily_pnl',
    'P&L do dia atual'
)

TOTAL_PNL = Gauge(
    'betting_total_pnl',
    'P&L total acumulado'
)

WIN_RATE = Gauge(
    'betting_win_rate',
    'Taxa de acerto (últimos 100 bets)'
)

ROI = Gauge(
    'betting_roi_percent',
    'ROI em porcentagem'
)


# ===== MÉTRICAS DE SISTEMA =====

QUEUE_SIZE = Gauge(
    'betting_queue_size',
    'Tamanho da fila de tarefas',
    ['queue_name']
)

CACHE_HIT_RATE = Gauge(
    'betting_cache_hit_rate',
    'Taxa de acerto do cache'
)


def start_metrics_server(port: int = 8000):
    """Inicia servidor de métricas Prometheus."""
    start_http_server(port)
    print(f"Metrics server started on port {port}")


# Exemplo de uso
def record_scrape(league: str, success: bool, duration: float, matches: int):
    """Registra métricas de um scrape."""
    status = 'success' if success else 'error'
    SCRAPE_REQUESTS.labels(league=league, status=status).inc()
    SCRAPE_DURATION.labels(league=league).observe(duration)
    if success:
        SCRAPE_MATCHES_FOUND.labels(league=league).set(matches)


def record_scoring(proba: float, decision: bool, duration: float):
    """Registra métricas de um scoring."""
    decision_label = 'approve' if decision else 'reject'
    SCORING_REQUESTS.labels(decision=decision_label).inc()
    SCORING_PROBA.observe(proba)
    SCORING_DURATION.observe(duration)


def record_bet(status: str, bookmaker: str, stake: float):
    """Registra métricas de uma aposta."""
    BETS_TOTAL.labels(status=status, bookmaker=bookmaker).inc()
    if status == 'placed':
        BET_STAKE.observe(stake)


def update_pnl_metrics(daily_pnl: float, total_pnl: float, 
                       win_rate: float, roi: float):
    """Atualiza métricas de P&L."""
    DAILY_PNL.set(daily_pnl)
    TOTAL_PNL.set(total_pnl)
    WIN_RATE.set(win_rate)
    ROI.set(roi)
```

### 6.6 Docker Compose para Monitoring

```yaml
# docker-compose.monitoring.yml

version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
    ports:
      - "9090:9090"
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:
```

```yaml
# monitoring/prometheus.yml

global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'betting_system'
    static_configs:
      - targets: ['host.docker.internal:8000']  # Seu app Python
```

### 6.7 Alternativa Simples: Logs Estruturados

Se Prometheus/Grafana parecer muito complexo inicialmente:

```python
# monitoring/simple_logging.py

"""
Sistema de logging estruturado simples.
Logs em JSONL que podem ser analisados depois.
"""

import json
from datetime import datetime, timezone
from pathlib import Path


class MetricsLogger:
    """Logger de métricas em JSONL."""
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
    def _log(self, event_type: str, data: dict):
        """Escreve evento no log."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **data
        }
        
        # Arquivo por dia
        date_str = datetime.now().strftime("%Y-%m-%d")
        filepath = self.log_dir / f"metrics_{date_str}.jsonl"
        
        with open(filepath, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    def log_scrape(self, league: str, success: bool, duration: float, 
                   matches_found: int, error: str = None):
        self._log("scrape", {
            "league": league,
            "success": success,
            "duration_seconds": duration,
            "matches_found": matches_found,
            "error": error,
        })
        
    def log_scoring(self, match_id: str, ah_line: str, proba: float, 
                    decision: bool, duration: float):
        self._log("scoring", {
            "match_id": match_id,
            "ah_line": ah_line,
            "proba": proba,
            "decision": decision,
            "duration_seconds": duration,
        })
        
    def log_bet(self, match_id: str, ah_line: str, bookmaker: str,
                stake: float, odds: float, status: str, error: str = None):
        self._log("bet", {
            "match_id": match_id,
            "ah_line": ah_line,
            "bookmaker": bookmaker,
            "stake": stake,
            "odds": odds,
            "status": status,
            "error": error,
        })
        
    def log_result(self, bet_id: int, result: str, profit_loss: float):
        self._log("result", {
            "bet_id": bet_id,
            "result": result,
            "profit_loss": profit_loss,
        })


# Script de análise dos logs
def analyze_logs(log_dir: str = "./logs", days: int = 7):
    """Analisa logs dos últimos N dias."""
    import pandas as pd
    from datetime import timedelta
    
    log_path = Path(log_dir)
    all_events = []
    
    for i in range(days):
        date = datetime.now() - timedelta(days=i)
        filepath = log_path / f"metrics_{date.strftime('%Y-%m-%d')}.jsonl"
        
        if filepath.exists():
            with open(filepath) as f:
                for line in f:
                    all_events.append(json.loads(line))
                    
    df = pd.DataFrame(all_events)
    
    # Análise por tipo de evento
    print("\n=== RESUMO DOS ÚLTIMOS {} DIAS ===\n".format(days))
    
    # Scraping
    scrapes = df[df['event_type'] == 'scrape']
    if len(scrapes) > 0:
        print("SCRAPING:")
        print(f"  Total de scrapes: {len(scrapes)}")
        print(f"  Taxa de sucesso: {scrapes['success'].mean():.1%}")
        print(f"  Duração média: {scrapes['duration_seconds'].mean():.2f}s")
        
    # Scoring
    scorings = df[df['event_type'] == 'scoring']
    if len(scorings) > 0:
        print("\nSCORING:")
        print(f"  Total de scorings: {len(scorings)}")
        print(f"  Aprovados: {scorings['decision'].sum()}")
        print(f"  Taxa de aprovação: {scorings['decision'].mean():.1%}")
        print(f"  Proba média: {scorings['proba'].mean():.3f}")
        
    # Apostas
    bets = df[df['event_type'] == 'bet']
    if len(bets) > 0:
        print("\nAPOSTAS:")
        print(f"  Total de apostas: {len(bets)}")
        print(f"  Stake total: R$ {bets['stake'].sum():.2f}")
        by_status = bets.groupby('status').size()
        print(f"  Por status: {by_status.to_dict()}")
        
    # Resultados
    results = df[df['event_type'] == 'result']
    if len(results) > 0:
        print("\nRESULTADOS:")
        print(f"  Total de resultados: {len(results)}")
        print(f"  P&L total: R$ {results['profit_loss'].sum():.2f}")
        by_result = results.groupby('result').size()
        print(f"  Por resultado: {by_result.to_dict()}")


if __name__ == "__main__":
    analyze_logs()
```

---

## 7. RESUMO DE CUSTOS

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        RESUMO DE CUSTOS MENSAIS                             │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  OPÇÃO 1: MÍNIMO (Self-hosted em VPS)                                       │
│  ├── VPS Linux (4GB RAM, 2 vCPU).............. R$ 50-100/mês               │
│  ├── Playwright................................ Gratuito                    │
│  ├── Celery + Redis (local)................... Gratuito                    │
│  ├── PostgreSQL (local)....................... Gratuito                    │
│  ├── Prometheus + Grafana (local)............. Gratuito                    │
│  └── TOTAL.................................... R$ 50-100/mês               │
│                                                                             │
│  OPÇÃO 2: INTERMEDIÁRIO (Cloud services)                                    │
│  ├── VPS Linux (8GB RAM, 4 vCPU).............. R$ 150-250/mês              │
│  ├── Redis Cloud (100MB)....................... R$ 30-50/mês               │
│  ├── Supabase PostgreSQL...................... Gratuito ou R$ 125/mês      │
│  ├── Grafana Cloud............................ Gratuito ou R$ 75/mês       │
│  └── TOTAL.................................... R$ 180-500/mês              │
│                                                                             │
│  OPÇÃO 3: SIMPLIFICADO (Sem Redis/Prometheus)                               │
│  ├── VPS Linux (2GB RAM, 1 vCPU).............. R$ 30-50/mês                │
│  ├── APScheduler em vez de Celery............. Gratuito                    │
│  ├── SQLite em vez de PostgreSQL.............. Gratuito                    │
│  ├── Logs JSONL em vez de Prometheus.......... Gratuito                    │
│  └── TOTAL.................................... R$ 30-50/mês                │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. COMPLEXIDADE GERAL E CRONOGRAMA SUGERIDO

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     COMPLEXIDADE POR COMPONENTE                             │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Componente          Complexidade    Tempo Setup    Dependências            │
│  ──────────────────────────────────────────────────────────────────────    │
│  Scraping            ████████░░      2-4 semanas    Entender HTML do site  │
│  Queue/Scheduler     ██████░░░░      1-2 dias       Redis (opcional)       │
│  Storage             ████░░░░░░      1 dia          PostgreSQL             │
│  Cache               ███░░░░░░░      0.5 dia        Redis                  │
│  Scoring (treino)    ███████░░░      2-4 semanas    Dados coletados        │
│  Scoring (engine)    ████░░░░░░      1 dia          Modelo treinado        │
│  Execução            ███████░░░      1-2 semanas    Scraper funcionando    │
│  Monitoring          █████░░░░░      1-2 dias       Prometheus/Grafana     │
│                                                                             │
│  TOTAL ESTIMADO: 6-10 semanas                                               │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│                        CRONOGRAMA SUGERIDO                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FASE 1: FUNDAÇÃO (Semana 1-2)                                              │
│  ├── Configurar VPS e ambiente                                              │
│  ├── Instalar PostgreSQL + Redis                                            │
│  ├── Criar estrutura de projeto                                             │
│  └── POC de login no BetinAsia                                              │
│                                                                             │
│  FASE 2: SCRAPING (Semana 2-4)                                              │
│  ├── Desenvolver scraper completo                                           │
│  ├── Testar com 2-3 ligas                                                   │
│  ├── Implementar cache de odds                                              │
│  └── Configurar scheduler básico                                            │
│                                                                             │
│  FASE 3: COLETA DE DADOS (Semana 4-8)                                       │
│  ├── Rodar sistema em modo coleta                                           │
│  ├── NÃO apostar, apenas coletar features                                   │
│  ├── Obter resultados das partidas                                          │
│  └── Calcular CLV                                                           │
│                                                                             │
│  FASE 4: MODELO (Semana 8-9)                                                │
│  ├── Treinar modelo com dados coletados                                     │
│  ├── Validar performance (AUC, calibração)                                  │
│  ├── Definir cutoff ótimo                                                   │
│  └── Integrar engine de scoring                                             │
│                                                                             │
│  FASE 5: EXECUÇÃO (Semana 9-10)                                             │
│  ├── Implementar executor de apostas                                        │
│  ├── Testar com stakes pequenos                                             │
│  ├── Configurar alertas (Telegram)                                          │
│  └── Setup de monitoring                                                    │
│                                                                             │
│  FASE 6: PRODUÇÃO (Semana 10+)                                              │
│  ├── Monitorar performance real                                             │
│  ├── Ajustar cutoff se necessário                                           │
│  ├── Expandir para mais ligas                                               │
│  └── Retreinar modelo periodicamente                                        │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. RECOMENDAÇÃO FINAL

Para começar, sugiro a **Opção 3 (Simplificado)** com possibilidade de upgrade:

1. **Scraping**: Playwright (obrigatório)
2. **Queue**: APScheduler (mais simples que Celery)
3. **Storage**: PostgreSQL ou SQLite (dependendo do volume)
4. **Cache**: Dicionário em memória ou Redis simples
5. **Monitoring**: Logs JSONL + script de análise

Conforme o sistema amadurecer e você validar que está gerando valor, upgrade para a **Opção 2**.

---

## 10. PRÓXIMOS PASSOS IMEDIATOS

1. **Criar conta em VPS** (DigitalOcean, Vultr, Hetzner, ou AWS Lightsail)
2. **Fazer engenharia reversa do BetinAsia** no browser (F12 → Network)
3. **Desenvolver POC de scraping** com 1 liga
4. **Iniciar coleta de dados** para treinamento do modelo
