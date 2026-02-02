# -*- coding: utf-8 -*-
"""
Detectores de Hipóteses.

Implementa detectores que analisam odds em tempo real e identificam
eventos relevantes para as hipóteses de estratégia.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from loguru import logger

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from storage.models_hypothesis import (
    H1PricingEvent,
    H3LineMonotonicityEvent,
    H3bTemporalReversalEvent,
    H6CorrelationLagEvent,
    OddsMovementHistory,
)


# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

# H1 - Precificação
H1_DEVIATION_THRESHOLD = 0.02  # 2% de desvio para considerar mispricing
H1_ARB_THRESHOLD = 1.0  # overround < 1 = arbitragem

# H3 - Monotonicidade entre linhas
H3_MIN_LINES = 2  # mínimo de linhas para verificar

# H3b - Reversões temporais
H3B_MIN_HISTORY = 3  # mínimo de pontos para detectar reversão
H3B_WINDOW_SECONDS = 3600  # janela de 1 hora para contar reversões

# H6 - Correlação/Lag
# NOTA: Threshold deve ser > COLLECTION_INTERVAL (60s) para fazer sentido
# Com coleta a cada 60s, só conseguimos detectar lags de ~2 ciclos ou mais
H6_LAG_THRESHOLD_SECONDS = 120  # 2 minutos de atraso para considerar lag
H6_MOVEMENT_THRESHOLD = 0.005  # 0.5% de movimento para considerar significativo


# ============================================================================
# ESTRUTURAS DE DADOS
# ============================================================================

@dataclass
class OddSnapshot:
    """Snapshot de uma odd em um momento."""
    match_id: int
    market_type: str  # AH, OU, 1X2
    line: str  # ex: "-0.5", "2.5", "1X2"
    side: str  # home, away, over, under
    odd: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class MarketSnapshot:
    """Snapshot completo de um mercado (ambos os lados)."""
    match_id: int
    market_type: str
    line: str
    home_odd: float  # ou over para OU
    away_odd: float  # ou under para OU
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# H1 - DETECTOR DE PRECIFICAÇÃO
# ============================================================================

class H1PricingDetector:
    """
    Detector de precificação incorreta.
    
    Analisa pares de odds (home/away, over/under) e detecta:
    - Arbitragem (overround < 1)
    - Mispricing significativo (desvio > threshold)
    """
    
    def __init__(self):
        self.current_markets: Dict[Tuple[int, str, str], MarketSnapshot] = {}
    
    def update_market(self, snapshot: MarketSnapshot) -> Optional[H1PricingEvent]:
        """
        Atualiza mercado e verifica precificação.
        
        Returns:
            H1PricingEvent se detectar anomalia, None caso contrário.
        """
        key = (snapshot.match_id, snapshot.market_type, snapshot.line)
        self.current_markets[key] = snapshot
        
        # Verifica precificação
        return self._check_pricing(snapshot)
    
    def _check_pricing(self, snapshot: MarketSnapshot) -> Optional[H1PricingEvent]:
        """Verifica se há mispricing no mercado."""
        
        # Valida odds
        if snapshot.home_odd <= 1.0 or snapshot.away_odd <= 1.0:
            return None
        
        # Calcula probabilidades implícitas
        prob_home = 1.0 / snapshot.home_odd
        prob_away = 1.0 / snapshot.away_odd
        overround = prob_home + prob_away
        
        if overround <= 0:
            return None
        
        # Calcula odds justas
        fair_prob_home = prob_home / overround
        fair_prob_away = prob_away / overround
        fair_odd_home = 1.0 / fair_prob_home if fair_prob_home > 0 else 0
        fair_odd_away = 1.0 / fair_prob_away if fair_prob_away > 0 else 0
        
        # Calcula desvios
        deviation_home = (snapshot.home_odd - fair_odd_home) / fair_odd_home if fair_odd_home > 0 else 0
        deviation_away = (snapshot.away_odd - fair_odd_away) / fair_odd_away if fair_odd_away > 0 else 0
        
        # Detecta anomalias
        is_arb = overround < H1_ARB_THRESHOLD
        mispriced_side = None
        edge_estimate = 0.0
        
        if abs(deviation_home) > H1_DEVIATION_THRESHOLD:
            mispriced_side = "home"
            edge_estimate = deviation_home
        elif abs(deviation_away) > H1_DEVIATION_THRESHOLD:
            mispriced_side = "away"
            edge_estimate = deviation_away
        
        # Se há anomalia, cria evento
        if is_arb or mispriced_side:
            # Determina lado recomendado para apostar
            # Se side_a está com odd acima da justa (deviation > 0), apostar em side_a
            # Se side_b está com odd acima da justa (deviation > 0), apostar em side_b
            if deviation_home > deviation_away:
                recommended_side = "side_a"
                recommended_odd = snapshot.home_odd
            else:
                recommended_side = "side_b"
                recommended_odd = snapshot.away_odd
            
            return H1PricingEvent(
                match_id=snapshot.match_id,
                market_type=snapshot.market_type,
                ah_line=snapshot.line,
                odd_side_a=snapshot.home_odd,
                odd_side_b=snapshot.away_odd,
                implied_prob_total=overround,
                overround=overround,
                fair_odd_a=fair_odd_home,
                fair_odd_b=fair_odd_away,
                deviation_a=deviation_home,
                deviation_b=deviation_away,
                is_arb=is_arb,
                mispriced_side=mispriced_side,
                edge_estimate=edge_estimate,
                # Dados para análise de valor
                recommended_side=recommended_side,
                recommended_odd=recommended_odd,
            )
        
        return None


# ============================================================================
# H3 - DETECTOR DE MONOTONICIDADE ENTRE LINHAS
# ============================================================================

class H3LineMonotonicityDetector:
    """
    Detector de quebra de monotonicidade entre linhas adjacentes.
    
    Verifica se a relação de preços entre linhas de AH está correta:
    - Linha mais negativa (ex: -0.75) deve ter odd menor que linha menos negativa (-0.5)
    """
    
    def __init__(self):
        # Armazena odds por (match_id, side)
        # Valor: {line: odd}
        self.lines_by_match: Dict[Tuple[int, str], Dict[str, float]] = defaultdict(dict)
    
    def update_line(
        self, 
        match_id: int, 
        line: str, 
        side: str, 
        odd: float
    ) -> List[H3LineMonotonicityEvent]:
        """
        Atualiza uma linha e verifica monotonicidade.
        
        Returns:
            Lista de eventos de anomalia detectados.
        """
        key = (match_id, side)
        self.lines_by_match[key][line] = odd
        
        # Verifica monotonicidade entre todas as linhas
        return self._check_monotonicity(match_id, side)
    
    def _check_monotonicity(self, match_id: int, side: str) -> List[H3LineMonotonicityEvent]:
        """Verifica monotonicidade entre linhas do mesmo lado."""
        key = (match_id, side)
        lines_dict = self.lines_by_match.get(key, {})
        
        if len(lines_dict) < H3_MIN_LINES:
            return []
        
        events = []
        
        # Converte linhas para float e ordena
        try:
            sorted_lines = sorted(
                [(float(line), odd) for line, odd in lines_dict.items()],
                key=lambda x: x[0]
            )
        except ValueError:
            return []
        
        # Verifica pares adjacentes
        for i in range(len(sorted_lines) - 1):
            line_a, odd_a = sorted_lines[i]
            line_b, odd_b = sorted_lines[i + 1]
            
            # Para home: linha mais negativa deve ter odd MENOR
            # Para away: linha mais negativa deve ter odd MAIOR
            if side == "home":
                # line_a < line_b (ex: -0.75 < -0.5)
                # odd_a deveria ser < odd_b
                expected = "a < b"
                if odd_a >= odd_b:
                    actual = "a >= b"
                    magnitude = odd_a - odd_b
                    
                    # Linha A está com odd alta demais - apostar nela
                    recommended_line = str(line_a)
                    recommended_odd = odd_a
                    
                    events.append(H3LineMonotonicityEvent(
                        match_id=match_id,
                        line_a=str(line_a),
                        line_b=str(line_b),
                        side=side,
                        odd_line_a=odd_a,
                        odd_line_b=odd_b,
                        expected_relation=expected,
                        actual_relation=actual,
                        magnitude=magnitude,
                        magnitude_pct=(magnitude / odd_b * 100) if odd_b > 0 else 0,
                        # Dados para análise de valor
                        recommended_line=recommended_line,
                        recommended_odd=recommended_odd,
                    ))
            else:  # away
                # Para away: linha mais negativa = mais difícil ganhar = odd MAIOR
                expected = "a > b"
                if odd_a <= odd_b:
                    actual = "a <= b"
                    magnitude = odd_b - odd_a
                    
                    # Linha B está com odd alta demais - apostar nela
                    recommended_line = str(line_b)
                    recommended_odd = odd_b
                    
                    events.append(H3LineMonotonicityEvent(
                        match_id=match_id,
                        line_a=str(line_a),
                        line_b=str(line_b),
                        side=side,
                        odd_line_a=odd_a,
                        odd_line_b=odd_b,
                        expected_relation=expected,
                        actual_relation=actual,
                        magnitude=magnitude,
                        magnitude_pct=(magnitude / odd_a * 100) if odd_a > 0 else 0,
                        # Dados para análise de valor
                        recommended_line=recommended_line,
                        recommended_odd=recommended_odd,
                    ))
        
        return events


# ============================================================================
# H3b - DETECTOR DE REVERSÕES TEMPORAIS
# ============================================================================

class H3bTemporalReversalDetector:
    """
    Detector de reversões temporais de odds.
    
    Monitora série temporal de cada odd e detecta mudanças de direção.
    """
    
    def __init__(self):
        # Histórico de odds por (match_id, market_type, line, side)
        # Valor: [(timestamp, odd), ...]
        self.history: Dict[Tuple[int, str, str, str], List[Tuple[datetime, float]]] = defaultdict(list)
        
        # Última direção por mercado
        self.last_direction: Dict[Tuple[int, str, str, str], str] = {}
        
        # Contador de reversões por mercado
        self.reversal_counts: Dict[Tuple[int, str, str, str], int] = defaultdict(int)
    
    def update_odd(self, snapshot: OddSnapshot) -> Optional[H3bTemporalReversalEvent]:
        """
        Atualiza uma odd e verifica se houve reversão.
        
        Returns:
            H3bTemporalReversalEvent se detectar reversão, None caso contrário.
        """
        key = (snapshot.match_id, snapshot.market_type, snapshot.line, snapshot.side)
        history = self.history[key]
        now = snapshot.timestamp
        
        # Limpa histórico antigo (fora da janela)
        window_start = now - timedelta(seconds=H3B_WINDOW_SECONDS)
        self.history[key] = [(ts, odd) for ts, odd in history if ts >= window_start]
        history = self.history[key]
        
        # Se não há histórico suficiente, apenas adiciona
        if len(history) < 1:
            history.append((now, snapshot.odd))
            return None
        
        # Calcula direção do movimento atual
        last_odd = history[-1][1]
        
        if snapshot.odd > last_odd:
            current_direction = "up"
        elif snapshot.odd < last_odd:
            current_direction = "down"
        else:
            # Sem movimento
            return None
        
        # Adiciona ao histórico
        history.append((now, snapshot.odd))
        
        # Verifica se houve reversão
        event = None
        if key in self.last_direction:
            prev_direction = self.last_direction[key]
            
            if prev_direction != current_direction:
                # REVERSÃO DETECTADA!
                self.reversal_counts[key] += 1
                
                # Calcula streak antes da reversão
                streak = self._calculate_streak(history[:-1], prev_direction)
                
                # Calcula índice de oscilação
                total_moves = len(history) - 1
                oscillation_idx = self.reversal_counts[key] / total_moves if total_moves > 0 else 0
                
                event = H3bTemporalReversalEvent(
                    match_id=snapshot.match_id,
                    market_type=snapshot.market_type,
                    ah_line=snapshot.line,
                    side=snapshot.side,
                    direction_before=prev_direction,
                    direction_after=current_direction,
                    reversal_magnitude=abs(snapshot.odd - last_odd),
                    streak_before=streak,
                    odd_at_reversal=snapshot.odd,
                    odd_before=last_odd,
                    num_reversals_1h=self.reversal_counts[key],
                    oscillation_index=oscillation_idx,
                    # Dados para análise de valor
                    bet_odd=snapshot.odd,
                    bet_side=snapshot.side,
                )
        
        # Atualiza direção
        self.last_direction[key] = current_direction
        
        return event
    
    def _calculate_streak(self, history: List[Tuple[datetime, float]], direction: str) -> int:
        """Calcula quantos movimentos consecutivos na mesma direção."""
        if len(history) < 2:
            return 0
        
        streak = 0
        for i in range(len(history) - 1, 0, -1):
            curr_odd = history[i][1]
            prev_odd = history[i-1][1]
            
            move_dir = "up" if curr_odd > prev_odd else "down" if curr_odd < prev_odd else None
            
            if move_dir == direction:
                streak += 1
            else:
                break
        
        return streak


# ============================================================================
# H6 - DETECTOR DE ATRASOS EM CORRELAÇÕES
# ============================================================================

class H6CorrelationLagDetector:
    """
    Detector de atrasos em odds correlacionadas.
    
    Quando um mercado move, verifica se mercados correlacionados também moveram.
    Se não moveram dentro do threshold, detecta lag.
    """
    
    # Definição de correlações entre mercados
    # Formato: (market_type, side) -> [(corr_market, corr_side, correlation_coef), ...]
    CORRELATIONS = {
        # Asian Handicap - linhas adjacentes
        ("AH", "home"): [
            ("AH", "home", 0.90),  # outras linhas AH home
        ],
        ("AH", "away"): [
            ("AH", "away", 0.90),  # outras linhas AH away
        ],
        # Over/Under - linhas adjacentes
        ("OU", "over"): [
            ("OU", "over", 0.85),
        ],
        ("OU", "under"): [
            ("OU", "under", 0.85),
        ],
    }
    
    def __init__(self):
        # Último movimento por (match_id, market_type, line, side)
        # Valor: (timestamp, old_odd, new_odd, direction)
        self.last_moves: Dict[Tuple[int, str, str, str], Tuple[datetime, float, float, str]] = {}
        
        # Odds atuais
        self.current_odds: Dict[Tuple[int, str, str, str], OddSnapshot] = {}
    
    def update_odd(self, snapshot: OddSnapshot) -> List[H6CorrelationLagEvent]:
        """
        Atualiza uma odd e verifica atrasos em mercados correlacionados.
        
        Returns:
            Lista de eventos de lag detectados.
        """
        key = (snapshot.match_id, snapshot.market_type, snapshot.line, snapshot.side)
        now = snapshot.timestamp
        events = []
        
        # Verifica se houve movimento
        old_snap = self.current_odds.get(key)
        
        if old_snap and old_snap.odd != snapshot.odd:
            # Movimento detectado!
            direction = "up" if snapshot.odd > old_snap.odd else "down"
            magnitude = abs(snapshot.odd - old_snap.odd)
            magnitude_pct = magnitude / old_snap.odd if old_snap.odd > 0 else 0
            
            # Só considera movimentos significativos
            if magnitude_pct >= H6_MOVEMENT_THRESHOLD:
                # Registra movimento
                self.last_moves[key] = (now, old_snap.odd, snapshot.odd, direction)
                
                # Verifica mercados correlacionados
                events = self._check_correlated_markets(
                    snapshot.match_id,
                    snapshot.market_type,
                    snapshot.line,
                    snapshot.side,
                    direction,
                    magnitude,
                    now
                )
        
        # Atualiza odd atual
        self.current_odds[key] = snapshot
        
        return events
    
    def _check_correlated_markets(
        self,
        match_id: int,
        market_type: str,
        line: str,
        side: str,
        direction: str,
        magnitude: float,
        now: datetime
    ) -> List[H6CorrelationLagEvent]:
        """Verifica se mercados correlacionados estão em lag."""
        events = []
        
        # Obtém linhas adjacentes para verificar
        adjacent_lines = self._get_adjacent_lines(line)
        
        for adj_line in adjacent_lines:
            corr_key = (match_id, market_type, adj_line, side)
            
            # Verifica se o mercado correlacionado existe
            if corr_key not in self.current_odds:
                continue
            
            # Verifica se moveu recentemente
            if corr_key in self.last_moves:
                last_move = self.last_moves[corr_key]
                move_age = (now - last_move[0]).total_seconds()
                
                if move_age < H6_LAG_THRESHOLD_SECONDS:
                    # Mercado correlacionado moveu recentemente, OK
                    continue
            
            # Mercado correlacionado NÃO moveu ou está atrasado
            corr_snap = self.current_odds[corr_key]
            
            # Calcula tempo de lag
            lag_seconds = H6_LAG_THRESHOLD_SECONDS
            if corr_key in self.last_moves:
                lag_seconds = (now - self.last_moves[corr_key][0]).total_seconds()
            
            if lag_seconds >= H6_LAG_THRESHOLD_SECONDS:
                # Determina correlação esperada
                expected_direction = direction  # mesma direção para mesmo lado
                correlation_coef = 0.90  # default para linhas adjacentes
                
                events.append(H6CorrelationLagEvent(
                    match_id=match_id,
                    leader_market_type=market_type,
                    leader_line=line,
                    leader_side=side,
                    leader_move_direction=direction,
                    leader_move_magnitude=magnitude,
                    leader_odd_before=None,  # Não temos essa info aqui
                    leader_odd_after=self.current_odds[(match_id, market_type, line, side)].odd,
                    lagged_market_type=market_type,
                    lagged_line=adj_line,
                    lagged_side=side,
                    lagged_current_odd=corr_snap.odd,
                    lag_seconds=lag_seconds,
                    expected_direction=expected_direction,
                    expected_move=magnitude * correlation_coef,
                    correlation_coefficient=correlation_coef,
                    # Dados para análise de valor (apostar no mercado atrasado)
                    bet_market_type=market_type,
                    bet_line=adj_line,
                    bet_side=side,
                    bet_odd=corr_snap.odd,
                ))
        
        return events
    
    def _get_adjacent_lines(self, line: str) -> List[str]:
        """Retorna linhas adjacentes para uma linha de AH."""
        try:
            line_val = float(line)
        except ValueError:
            return []
        
        # Linhas adjacentes com delta de 0.25 e 0.5
        deltas = [-0.5, -0.25, 0.25, 0.5]
        adjacent = []
        
        for delta in deltas:
            adj_val = line_val + delta
            # Formata de volta para string
            if adj_val == int(adj_val):
                adj_str = str(int(adj_val))
            else:
                adj_str = str(adj_val)
            adjacent.append(adj_str)
        
        return adjacent


# ============================================================================
# AGREGADOR DE DETECTORES
# ============================================================================

class HypothesisDetector:
    """
    Agregador que combina todos os detectores e facilita integração.
    """
    
    def __init__(self):
        self.h1_detector = H1PricingDetector()
        self.h3_detector = H3LineMonotonicityDetector()
        self.h3b_detector = H3bTemporalReversalDetector()
        self.h6_detector = H6CorrelationLagDetector()
        
        # Contadores para logging
        self.event_counts = {
            "h1": 0,
            "h3": 0,
            "h3b": 0,
            "h6": 0,
        }
    
    def process_market_update(
        self,
        match_id: int,
        market_type: str,
        line: str,
        home_odd: float,
        away_odd: float,
        timestamp: datetime = None
    ) -> Dict[str, List]:
        """
        Processa atualização de um mercado completo (ambos os lados).
        
        Esta é a função principal para integração com o coletor.
        
        Returns:
            Dict com listas de eventos detectados por hipótese.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        results = {
            "h1_events": [],
            "h3_events": [],
            "h3b_events": [],
            "h6_events": [],
        }
        
        # Cria snapshots
        market_snap = MarketSnapshot(
            match_id=match_id,
            market_type=market_type,
            line=line,
            home_odd=home_odd,
            away_odd=away_odd,
            timestamp=timestamp
        )
        
        home_snap = OddSnapshot(
            match_id=match_id,
            market_type=market_type,
            line=line,
            side="home" if market_type in ("AH", "1X2") else "over",
            odd=home_odd,
            timestamp=timestamp
        )
        
        away_snap = OddSnapshot(
            match_id=match_id,
            market_type=market_type,
            line=line,
            side="away" if market_type in ("AH", "1X2") else "under",
            odd=away_odd,
            timestamp=timestamp
        )
        
        # H1 - Precificação
        h1_event = self.h1_detector.update_market(market_snap)
        if h1_event:
            results["h1_events"].append(h1_event)
            self.event_counts["h1"] += 1
        
        # H3 - Monotonicidade entre linhas (só para AH)
        if market_type == "AH":
            h3_events_home = self.h3_detector.update_line(
                match_id, line, "home", home_odd
            )
            h3_events_away = self.h3_detector.update_line(
                match_id, line, "away", away_odd
            )
            results["h3_events"].extend(h3_events_home)
            results["h3_events"].extend(h3_events_away)
            self.event_counts["h3"] += len(h3_events_home) + len(h3_events_away)
        
        # H3b - Reversões temporais
        h3b_event_home = self.h3b_detector.update_odd(home_snap)
        h3b_event_away = self.h3b_detector.update_odd(away_snap)
        if h3b_event_home:
            results["h3b_events"].append(h3b_event_home)
            self.event_counts["h3b"] += 1
        if h3b_event_away:
            results["h3b_events"].append(h3b_event_away)
            self.event_counts["h3b"] += 1
        
        # H6 - Correlação/Lag
        h6_events_home = self.h6_detector.update_odd(home_snap)
        h6_events_away = self.h6_detector.update_odd(away_snap)
        results["h6_events"].extend(h6_events_home)
        results["h6_events"].extend(h6_events_away)
        self.event_counts["h6"] += len(h6_events_home) + len(h6_events_away)
        
        return results
    
    def get_stats(self) -> Dict[str, int]:
        """Retorna estatísticas de eventos detectados."""
        return self.event_counts.copy()
    
    def reset_stats(self):
        """Reseta contadores."""
        for k in self.event_counts:
            self.event_counts[k] = 0


# ============================================================================
# FUNÇÃO AUXILIAR PARA SALVAR EVENTOS NO BANCO
# ============================================================================

async def save_hypothesis_events(
    session: AsyncSession,
    events: Dict[str, List]
) -> int:
    """
    Salva eventos detectados no banco de dados.
    
    Args:
        session: Sessão assíncrona do SQLAlchemy
        events: Dict retornado por HypothesisDetector.process_market_update()
        
    Returns:
        Número de eventos salvos.
    """
    count = 0
    
    for event in events.get("h1_events", []):
        session.add(event)
        count += 1
    
    for event in events.get("h3_events", []):
        session.add(event)
        count += 1
    
    for event in events.get("h3b_events", []):
        session.add(event)
        count += 1
    
    for event in events.get("h6_events", []):
        session.add(event)
        count += 1
    
    return count
