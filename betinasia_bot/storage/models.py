# -*- coding: utf-8 -*-
"""
Modelos SQLAlchemy para o banco de dados.
Define as tabelas do PostgreSQL.
"""

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, 
    ForeignKey, Text, UniqueConstraint, Index
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func
from datetime import datetime, timezone

Base = declarative_base()


class Match(Base):
    """Tabela de partidas."""
    
    __tablename__ = "matches"
    
    id = Column(Integer, primary_key=True)
    external_id = Column(String(100), unique=True, nullable=False)
    league = Column(String(200), nullable=False)
    home_team = Column(String(200), nullable=False)
    away_team = Column(String(200), nullable=False)
    kickoff_time = Column(DateTime(timezone=True), nullable=False)
    
    # Resultado (preenchido depois)
    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)
    status = Column(String(20), default="scheduled")  # scheduled, live, finished
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relacionamentos
    odds_history = relationship("OddsHistory", back_populates="match")
    opportunities = relationship("Opportunity", back_populates="match")
    bets = relationship("Bet", back_populates="match")
    
    __table_args__ = (
        Index("idx_matches_league", "league"),
        Index("idx_matches_kickoff", "kickoff_time"),
    )


class BestOddsHistory(Base):
    """Tabela de histórico de best odds - coleta rápida sem cliques."""
    
    __tablename__ = "best_odds_history"
    
    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    ah_line = Column(String(20), nullable=False)
    
    # Best odds extraídas do DOM (sem clique)
    best_home_odds = Column(Float, nullable=False)
    best_away_odds = Column(Float, nullable=False)
    
    scraped_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relacionamento
    match = relationship("Match", backref="best_odds_history")
    
    __table_args__ = (
        Index("idx_best_odds_match_line", "match_id", "ah_line"),
        Index("idx_best_odds_scraped", "scraped_at"),
    )


class OddsHistory(Base):
    """Tabela de histórico de odds - métricas consolidadas por linha de AH."""
    
    __tablename__ = "odds_history"
    
    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    ah_line = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # "home" ou "away"
    
    # Métricas principais
    best_odds = Column(Float, nullable=False)
    best_bookmaker = Column(String(50), nullable=False)
    second_best_odds = Column(Float)
    second_best_bookmaker = Column(String(50))
    median_odds = Column(Float)
    num_bookmakers = Column(Integer)
    
    # Campos legados (mantidos para compatibilidade)
    bookmaker = Column(String(50))  # Deprecated - usar best_bookmaker
    home_odds = Column(Float)  # Deprecated - usar best_odds com side="home"
    away_odds = Column(Float)  # Deprecated - usar best_odds com side="away"
    
    scraped_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relacionamento
    match = relationship("Match", back_populates="odds_history")
    
    __table_args__ = (
        Index("idx_odds_match_line", "match_id", "ah_line"),
        Index("idx_odds_scraped", "scraped_at"),
        Index("idx_odds_side", "side"),
    )


class Opportunity(Base):
    """Tabela de oportunidades detectadas."""
    
    __tablename__ = "opportunities"
    
    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    ah_line = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # home, away
    
    # Dados da detecção
    best_odds = Column(Float, nullable=False)
    best_bookmaker = Column(String(50), nullable=False)
    num_bookmakers = Column(Integer, nullable=False)
    detection_time = Column(DateTime(timezone=True), server_default=func.now())
    
    # Features
    dif_pct_best_second = Column(Float)
    dif_pct_best_median = Column(Float)
    dif_vs_pinnacle = Column(Float)
    minutes_to_kickoff = Column(Integer)
    
    # Scoring
    proba = Column(Float)
    cutoff = Column(Float)
    decision = Column(Boolean)
    scored_at = Column(DateTime(timezone=True))
    
    # Closing line
    closing_odds = Column(Float)
    closing_time = Column(DateTime(timezone=True))
    clv = Column(Float)  # (detection - closing) / closing
    clv_positive = Column(Boolean)
    
    # Resultado
    bet_result = Column(String(20))  # win, loss, half_win, half_loss, push
    profit_loss = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relacionamentos
    match = relationship("Match", back_populates="opportunities")
    bet = relationship("Bet", back_populates="opportunity", uselist=False)
    
    __table_args__ = (
        Index("idx_opp_match", "match_id"),
        Index("idx_opp_decision", "decision"),
        Index("idx_opp_detection", "detection_time"),
    )


class Bet(Base):
    """Tabela de apostas executadas."""
    
    __tablename__ = "bets"
    
    id = Column(Integer, primary_key=True)
    opportunity_id = Column(Integer, ForeignKey("opportunities.id"))
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    
    # Detalhes da aposta
    ah_line = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)
    bookmaker = Column(String(50), nullable=False)
    
    # Odds e stake
    expected_odds = Column(Float, nullable=False)
    actual_odds = Column(Float)
    stake = Column(Float, nullable=False)
    potential_return = Column(Float)
    
    # Status
    status = Column(String(20), nullable=False, default="pending")
    # pending, placed, rejected, cancelled, error
    confirmation_id = Column(String(100))
    error_message = Column(Text)
    
    # Resultado
    result = Column(String(20))  # win, loss, half_win, half_loss, push, pending
    profit_loss = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    executed_at = Column(DateTime(timezone=True))
    settled_at = Column(DateTime(timezone=True))
    
    # Relacionamentos
    match = relationship("Match", back_populates="bets")
    opportunity = relationship("Opportunity", back_populates="bet")
    
    __table_args__ = (
        UniqueConstraint("match_id", "ah_line", "side", "bookmaker", name="uq_bet_unique"),
        Index("idx_bets_status", "status"),
        Index("idx_bets_result", "result"),
    )


class LeagueConfig(Base):
    """Tabela de configuração de ligas."""
    
    __tablename__ = "league_config"
    
    id = Column(Integer, primary_key=True)
    league_name = Column(String(200), unique=True, nullable=False)
    tier = Column(Integer, nullable=False)  # 1, 2, 3
    scrape_interval = Column(Integer, default=60)  # segundos
    is_active = Column(Boolean, default=True)
    priority_score = Column(Float)  # Baseado no histórico
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class DailyMetrics(Base):
    """Tabela de métricas diárias."""
    
    __tablename__ = "daily_metrics"
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, unique=True, nullable=False)
    
    # Volume
    total_opportunities = Column(Integer, default=0)
    total_bets = Column(Integer, default=0)
    
    # P&L
    total_stake = Column(Float, default=0)
    total_profit_loss = Column(Float, default=0)
    roi_percent = Column(Float)
    
    # Por resultado
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    pushes = Column(Integer, default=0)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
