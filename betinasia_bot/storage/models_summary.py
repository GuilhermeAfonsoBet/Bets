# -*- coding: utf-8 -*-
"""
Modelo SQLAlchemy para tabela odds_summary.

Armazena resumo compactado das movimentacoes de odds.
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, 
    ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .models import Base


class OddsSummary(Base):
    """
    Tabela de resumo de odds.
    
    Um registro por jogo/mercado/linha/lado.
    Armazena estatisticas de movimentacao e resultado.
    """
    
    __tablename__ = "odds_summary"
    
    # ==========================================
    # IDENTIFICACAO
    # ==========================================
    
    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    event_id = Column(String(100), nullable=False)
    
    # Info do jogo (desnormalizado para queries rapidas)
    home_team = Column(String(200), nullable=False)
    away_team = Column(String(200), nullable=False)
    league = Column(String(200), nullable=False)
    country = Column(String(50))
    kickoff_time = Column(DateTime(timezone=True), nullable=False)
    
    # Identificacao da linha
    market_type = Column(String(10), nullable=False)  # AH, OU, 1X2
    line = Column(Float)  # -1.25, 2.5, NULL para 1X2
    side = Column(String(10), nullable=False)  # home, away, over, under, draw
    
    # ==========================================
    # ODDS DE ABERTURA
    # ==========================================
    
    opening_odds = Column(Float, nullable=False)
    opening_time = Column(DateTime(timezone=True), nullable=False)
    minutes_to_kick_at_open = Column(Integer)
    
    # ==========================================
    # ODDS DE FECHAMENTO (CLOSING LINE)
    # ==========================================
    
    closing_odds = Column(Float, nullable=False)
    closing_time = Column(DateTime(timezone=True), nullable=False)
    minutes_to_kick_at_close = Column(Integer)
    
    # ==========================================
    # ESTATISTICAS DE MOVIMENTACAO
    # ==========================================
    
    min_odds = Column(Float, nullable=False)
    max_odds = Column(Float, nullable=False)
    avg_odds = Column(Float, nullable=False)
    std_odds = Column(Float)  # Desvio padrao (volatilidade)
    num_collections = Column(Integer, nullable=False)
    
    # Metricas de movimento
    movement_pct = Column(Float)  # (closing - opening) / opening * 100
    range_pct = Column(Float)  # (max - min) / avg * 100
    direction = Column(String(10))  # up, down, stable
    
    # ==========================================
    # STEAM MOVES
    # ==========================================
    
    steam_moves_count = Column(Integer, default=0)  # Movimentos > 3%
    max_single_move_pct = Column(Float)
    avg_move_per_collection = Column(Float)
    
    # ==========================================
    # RESULTADO
    # ==========================================
    
    home_score = Column(Integer)
    away_score = Column(Integer)
    bet_result = Column(String(20))  # win, loss, half_win, half_loss, push
    profit_loss = Column(Float)  # Para stake = 1
    
    # ==========================================
    # METRICAS DE VALOR
    # ==========================================
    
    clv = Column(Float)  # opening - closing
    clv_pct = Column(Float)  # CLV percentual
    
    # ==========================================
    # METADADOS
    # ==========================================
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # ==========================================
    # RELACIONAMENTOS
    # ==========================================
    
    match = relationship("Match", backref="odds_summaries")
    
    # ==========================================
    # CONSTRAINTS E INDICES
    # ==========================================
    
    __table_args__ = (
        # Unique: um resumo por jogo/mercado/linha/lado
        UniqueConstraint(
            "match_id", "market_type", "line", "side",
            name="uq_odds_summary_unique"
        ),
        
        # Indices para queries comuns
        Index("idx_summary_match", "match_id"),
        Index("idx_summary_market_line", "market_type", "line"),
        Index("idx_summary_league", "league"),
        Index("idx_summary_kickoff", "kickoff_time"),
        Index("idx_summary_result", "bet_result"),
        Index("idx_summary_clv", "clv_pct"),
    )
    
    def __repr__(self):
        return (
            f"<OddsSummary {self.home_team} vs {self.away_team} "
            f"{self.market_type} {self.line} {self.side}>"
        )
