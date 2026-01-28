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
    betinasia_username: str = Field(..., env="BETINASIA_USERNAME")
    betinasia_password: str = Field(..., env="BETINASIA_PASSWORD")
    
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
    # Logging
    # ===========================================
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Path = Field(default=Path("./logs/betinasia_bot.log"), env="LOG_FILE")
    
    # ===========================================
    # Ambiente
    # ===========================================
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Instância global das configurações
settings = Settings()
