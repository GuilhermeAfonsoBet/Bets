# -*- coding: utf-8 -*-
"""
Modelos SQLAlchemy para eventos de hipóteses.

Armazena eventos detectados pelos monitores de hipóteses:
- H1: Precificação incorreta
- H3: Quebra de monotonicidade entre linhas adjacentes
- H3b: Reversões temporais de odds
- H6: Atrasos em odds correlacionadas
"""

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, 
    ForeignKey, Text, Index, JSON
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .models import Base


class H1PricingEvent(Base):
    """
    Eventos de precificação incorreta (H1).
    
    Detecta quando o overround de um mercado está anômalo
    (arbitragem ou mispricing significativo).
    """
    
    __tablename__ = "h1_pricing_events"
    
    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    
    # Mercado
    market_type = Column(String(20), nullable=False)  # AH, OU, 1X2
    ah_line = Column(String(20), nullable=False)
    
    # Odds observadas no momento do evento
    odd_side_a = Column(Float, nullable=False)  # home/over
    odd_side_b = Column(Float, nullable=False)  # away/under
    
    # Cálculos de precificação
    implied_prob_total = Column(Float, nullable=False)  # soma das probs implícitas
    overround = Column(Float, nullable=False)  # margem (>1 normal, <1 arb)
    fair_odd_a = Column(Float)  # odd justa calculada
    fair_odd_b = Column(Float)
    deviation_a = Column(Float)  # desvio da odd real vs justa
    deviation_b = Column(Float)
    
    # Classificação
    is_arb = Column(Boolean, default=False)  # se há arbitragem
    mispriced_side = Column(String(10))  # qual lado está mal precificado (side_a ou side_b)
    edge_estimate = Column(Float)  # estimativa de edge
    
    # === DADOS PARA ANÁLISE DE VALOR ===
    # Lado recomendado para apostar (baseado no mispricing)
    recommended_side = Column(String(10))  # "side_a" ou "side_b"
    recommended_odd = Column(Float)  # odd do lado recomendado
    
    # Closing line (preenchido após o jogo)
    closing_odd_side_a = Column(Float)
    closing_odd_side_b = Column(Float)
    closing_odd_recommended = Column(Float)
    
    # CLV (preenchido após o jogo)
    clv = Column(Float)  # recommended_odd - closing_odd_recommended
    clv_pct = Column(Float)  # CLV percentual
    
    # Resultado (preenchido após o jogo)
    bet_result = Column(String(20))  # win, loss, half_win, half_loss, push
    profit_loss = Column(Float)  # P&L para stake=1
    
    # Timestamp
    detected_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Status do jogo no momento da detecção
    is_live = Column(Boolean, default=False)  # True = in-match, False = pre-match
    
    # Relacionamento
    match = relationship("Match", backref="h1_pricing_events")
    
    __table_args__ = (
        Index("idx_h1_match", "match_id"),
        Index("idx_h1_detected", "detected_at"),
        Index("idx_h1_is_arb", "is_arb"),
        Index("idx_h1_clv", "clv_pct"),
        Index("idx_h1_result", "bet_result"),
        Index("idx_h1_is_live", "is_live"),
    )


class H3LineMonotonicityEvent(Base):
    """
    Eventos de quebra de monotonicidade entre linhas adjacentes (H3).
    
    Detecta quando a relação de preços entre linhas de AH está invertida.
    Ex: AH -0.75 pagando mais que AH -0.5 (deveria pagar menos).
    
    Estratégia: apostar na linha que está "cara demais" esperando correção.
    """
    
    __tablename__ = "h3_line_monotonicity_events"
    
    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    
    # Linhas envolvidas
    line_a = Column(String(20), nullable=False)  # linha menor (ex: -0.75)
    line_b = Column(String(20), nullable=False)  # linha maior (ex: -0.5)
    side = Column(String(10), nullable=False)  # home ou away
    
    # Odds das linhas no momento do evento
    odd_line_a = Column(Float, nullable=False)
    odd_line_b = Column(Float, nullable=False)
    
    # Métricas da anomalia
    expected_relation = Column(String(20), nullable=False)  # "a < b" ou "a > b"
    actual_relation = Column(String(20), nullable=False)
    magnitude = Column(Float, nullable=False)  # diferença absoluta
    magnitude_pct = Column(Float)  # diferença percentual
    
    # === DADOS PARA ANÁLISE DE VALOR ===
    # Linha recomendada (a que está com odd "errada" - potencial valor)
    recommended_line = Column(String(20))  # line_a ou line_b
    recommended_odd = Column(Float)  # odd da linha recomendada
    
    # Closing line (preenchido após o jogo)
    closing_odd_line_a = Column(Float)
    closing_odd_line_b = Column(Float)
    closing_odd_recommended = Column(Float)
    
    # CLV (preenchido após o jogo)
    clv = Column(Float)  # recommended_odd - closing_odd_recommended
    clv_pct = Column(Float)
    
    # Resultado (preenchido após o jogo)
    bet_result = Column(String(20))  # win, loss, half_win, half_loss, push
    profit_loss = Column(Float)  # P&L para stake=1
    
    # Timestamp
    detected_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Status do jogo no momento da detecção
    is_live = Column(Boolean, default=False)  # True = in-match, False = pre-match
    
    # Relacionamento
    match = relationship("Match", backref="h3_line_events")
    
    __table_args__ = (
        Index("idx_h3_match", "match_id"),
        Index("idx_h3_detected", "detected_at"),
        Index("idx_h3_clv", "clv_pct"),
        Index("idx_h3_result", "bet_result"),
        Index("idx_h3_is_live", "is_live"),
    )


class H3bTemporalReversalEvent(Base):
    """
    Eventos de reversão temporal de odds (H3b).
    
    Detecta quando uma odd reverte direção ao longo do tempo
    (estava subindo e começou a descer, ou vice-versa).
    
    Estratégia possível: apostar na direção da reversão (ou contra).
    """
    
    __tablename__ = "h3b_temporal_reversal_events"
    
    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    
    # Mercado
    market_type = Column(String(20), nullable=False)  # AH, OU, 1X2
    ah_line = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # home, away, over, under
    
    # Direções
    direction_before = Column(String(10), nullable=False)  # up ou down
    direction_after = Column(String(10), nullable=False)  # up ou down
    
    # Métricas
    reversal_magnitude = Column(Float, nullable=False)  # tamanho da reversão
    streak_before = Column(Integer)  # movimentos consecutivos antes
    odd_at_reversal = Column(Float, nullable=False)  # odd no momento da reversão
    odd_before = Column(Float)  # odd antes do último movimento
    
    # Contexto
    num_reversals_1h = Column(Integer)  # reversões na última hora
    oscillation_index = Column(Float)  # índice de oscilação
    
    # === DADOS PARA ANÁLISE DE VALOR ===
    # Odd e lado no momento do evento (para análise posterior)
    bet_odd = Column(Float)  # = odd_at_reversal (duplicado para clareza)
    bet_side = Column(String(10))  # = side (duplicado para clareza)
    
    # Closing line (preenchido após o jogo)
    closing_odd = Column(Float)
    
    # CLV (preenchido após o jogo)
    clv = Column(Float)  # bet_odd - closing_odd
    clv_pct = Column(Float)
    
    # Resultado (preenchido após o jogo)
    bet_result = Column(String(20))  # win, loss, half_win, half_loss, push
    profit_loss = Column(Float)  # P&L para stake=1
    
    # Timestamp
    detected_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Status do jogo no momento da detecção
    is_live = Column(Boolean, default=False)  # True = in-match, False = pre-match
    
    # Relacionamento
    match = relationship("Match", backref="h3b_reversal_events")
    
    __table_args__ = (
        Index("idx_h3b_match", "match_id"),
        Index("idx_h3b_detected", "detected_at"),
        Index("idx_h3b_market", "market_type", "ah_line"),
        Index("idx_h3b_clv", "clv_pct"),
        Index("idx_h3b_result", "bet_result"),
        Index("idx_h3b_is_live", "is_live"),
    )


class H6CorrelationLagEvent(Base):
    """
    Eventos de atraso em odds correlacionadas (H6).
    
    Detecta quando um mercado move mas mercados correlacionados
    não acompanham (lag).
    
    Estratégia: apostar no mercado atrasado na direção esperada do movimento.
    """
    
    __tablename__ = "h6_correlation_lag_events"
    
    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    
    # Mercado líder (que moveu primeiro)
    leader_market_type = Column(String(20), nullable=False)
    leader_line = Column(String(20), nullable=False)
    leader_side = Column(String(10), nullable=False)
    leader_move_direction = Column(String(10), nullable=False)  # up ou down
    leader_move_magnitude = Column(Float, nullable=False)
    leader_odd_before = Column(Float)
    leader_odd_after = Column(Float, nullable=False)
    
    # Mercado atrasado (onde apostaríamos)
    lagged_market_type = Column(String(20), nullable=False)
    lagged_line = Column(String(20), nullable=False)
    lagged_side = Column(String(10), nullable=False)
    lagged_current_odd = Column(Float, nullable=False)  # odd no momento do evento
    
    # Métricas de atraso
    lag_seconds = Column(Float, nullable=False)  # tempo de atraso
    expected_direction = Column(String(10))  # direção esperada do lag
    expected_move = Column(Float)  # movimento esperado
    correlation_coefficient = Column(Float)  # correlação esperada
    
    # === DADOS PARA ANÁLISE DE VALOR ===
    # Aposta recomendada: mercado atrasado, esperando que mova na direção do líder
    bet_market_type = Column(String(20))  # = lagged_market_type
    bet_line = Column(String(20))  # = lagged_line
    bet_side = Column(String(10))  # = lagged_side
    bet_odd = Column(Float)  # = lagged_current_odd
    
    # Closing line (preenchido após o jogo)
    closing_odd = Column(Float)
    
    # CLV (preenchido após o jogo)
    clv = Column(Float)  # bet_odd - closing_odd
    clv_pct = Column(Float)
    
    # Resultado (preenchido após o jogo)
    bet_result = Column(String(20))  # win, loss, half_win, half_loss, push
    profit_loss = Column(Float)  # P&L para stake=1
    
    # Timestamp
    detected_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Status do jogo no momento da detecção
    is_live = Column(Boolean, default=False)  # True = in-match, False = pre-match
    
    # === VERIFICAÇÃO / AUDITORIA ===
    # Permite identificar falsos positivos (odds que não existem de fato)
    verification_status = Column(String(30))  # NULL=não verificado, VERIFIED=existe, FALSE_POSITIVE=não existe
    verification_reason = Column(String(50))  # Motivo se falso positivo: LINE_NOT_AVAILABLE, GAME_NOT_FOUND, etc.
    verified_at = Column(DateTime(timezone=True))  # Quando foi verificado
    verified_odd = Column(Float)  # Odd real encontrada no betslip (se verificado)
    verified_diff_pct = Column(Float)  # Diferença % entre WebSocket e betslip
    
    # Relacionamento
    match = relationship("Match", backref="h6_lag_events")
    
    __table_args__ = (
        Index("idx_h6_match", "match_id"),
        Index("idx_h6_detected", "detected_at"),
        Index("idx_h6_lag", "lag_seconds"),
        Index("idx_h6_clv", "clv_pct"),
        Index("idx_h6_result", "bet_result"),
        Index("idx_h6_is_live", "is_live"),
        Index("idx_h6_verification", "verification_status"),
    )


class OddsMovementHistory(Base):
    """
    Histórico de movimentos de odds para análise temporal.
    
    Usado pelos detectores H3b e H6 para rastrear mudanças.
    Mais leve que BestOddsHistory, focado em detectar mudanças.
    """
    
    __tablename__ = "odds_movement_history"
    
    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    
    # Identificação do mercado
    market_type = Column(String(20), nullable=False)  # AH, OU, 1X2
    ah_line = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)
    
    # Movimento
    odd_before = Column(Float)
    odd_after = Column(Float, nullable=False)
    direction = Column(String(10))  # up, down, ou null se primeiro registro
    magnitude = Column(Float)  # diferença absoluta
    magnitude_pct = Column(Float)  # diferença percentual
    
    # Timestamp
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relacionamento
    match = relationship("Match", backref="odds_movements")
    
    __table_args__ = (
        Index("idx_movement_match_market", "match_id", "market_type", "ah_line", "side"),
        Index("idx_movement_recorded", "recorded_at"),
    )
