# -*- coding: utf-8 -*-
"""
Configurações centralizadas do BetinAsia Bot.
Carrega variáveis do arquivo .env
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    """Configurações da aplicação carregadas do .env"""
    
    # ===========================================
    # BetinAsia
    # ===========================================
    # Observação:
    # - Estes campos são necessários para o scraper/login, mas NÃO deveriam
    #   impedir scripts puramente analíticos (DB-only) de rodarem.
    # - Por isso, usamos defaults vazios e validamos no ponto de uso (login).
    betinasia_username: str = Field(default="", env="BETINASIA_USERNAME")
    betinasia_password: str = Field(default="", env="BETINASIA_PASSWORD")
    
    # ===========================================
    # Banco de Dados
    # ===========================================
    database_url: str = Field(
        default="postgresql://betbot:password@localhost:5432/betinasia_bot",
        env="DATABASE_URL"
    )
    
    # ===========================================
    # Redis
    # ===========================================
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL"
    )
    
    # ===========================================
    # Telegram
    # ===========================================
    telegram_bot_token: Optional[str] = Field(default=None, env="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = Field(default=None, env="TELEGRAM_CHAT_ID")
    
    # ===========================================
    # Proxy Residencial (Bright Data, IPRoyal, etc)
    # ===========================================
    proxy_server: Optional[str] = Field(default=None, env="PROXY_SERVER")
    proxy_username: Optional[str] = Field(default=None, env="PROXY_USERNAME")
    proxy_password: Optional[str] = Field(default=None, env="PROXY_PASSWORD")
    
    # ===========================================
    # Scraping
    # ===========================================
    scrape_interval_tier1: int = Field(default=30, env="SCRAPE_INTERVAL_TIER1")
    scrape_interval_tier2: int = Field(default=120, env="SCRAPE_INTERVAL_TIER2")
    scrape_interval_tier3: int = Field(default=300, env="SCRAPE_INTERVAL_TIER3")
    browser_headless: bool = Field(default=True, env="BROWSER_HEADLESS")
    
    # ===========================================
    # Scoring
    # ===========================================
    models_dir: Path = Field(default=Path("./models"), env="MODELS_DIR")
    scoring_cutoff: float = Field(default=0.62, env="SCORING_CUTOFF")
    
    # ===========================================
    # Execução
    # ===========================================
    dry_run: bool = Field(default=True, env="DRY_RUN")
    base_stake: float = Field(default=10.0, env="BASE_STAKE")
    max_stake: float = Field(default=100.0, env="MAX_STAKE")

    # ===========================================
    # Auditoria / Operação (scripts)
    # ===========================================
    # Importante: alguns scripts (ex.: `audit_h3b_api.py`) leem essas configs via `os.getenv`,
    # mas como o Settings carrega o `.env`, precisamos declarar aqui para não falhar com
    # "extra_forbidden" quando as variáveis estiverem presentes no arquivo.
    audit_mode: str = Field(default="api", env="AUDIT_MODE")
    audit_ws_sample_offsets_sec: str = Field(
        default="0,3,6,9,12,15,18,21,24,27,30",
        env="AUDIT_WS_SAMPLE_OFFSETS_SEC",
    )
    
    # ===========================================
    # Logging
    # ===========================================
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Path = Field(default=Path("./logs/betinasia_bot.log"), env="LOG_FILE")
    
    # ===========================================
    # Ambiente
    # ===========================================
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    @property
    def proxy_config(self) -> dict:
        """Retorna config de proxy para Playwright, ou None se não configurado."""
        if self.proxy_server:
            config = {"server": self.proxy_server}
            if self.proxy_username:
                config["username"] = self.proxy_username
            if self.proxy_password:
                config["password"] = self.proxy_password
            return config
        return None
    
    class Config:
        # Carrega o .env do diretório raiz do projeto (betinasia_bot/.env),
        # mesmo quando o script é executado a partir de outro CWD (ex.: ~/Bets).
        env_file = str(Path(__file__).resolve().parents[1] / ".env")
        env_file_encoding = "utf-8"
        case_sensitive = False


# Instância global das configurações
settings = Settings()
