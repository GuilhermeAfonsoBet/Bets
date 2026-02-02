# -*- coding: utf-8 -*-
"""
monitoramento_hipoteses.py

Sistema de monitoramento contínuo para detectar e gravar eventos relevantes
para as hipóteses H1, H3 e H6, usando apenas BEST ODDS como fonte de dados.

Arquitetura:
- Monitora streams de best odds em tempo real
- Detecta eventos (precificação incorreta, quebras de monotonicidade, atrasos)
- Grava eventos em arquivos JSONL para posterior merge com tabela resumo
"""

from __future__ import annotations
import json
import math
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import deque
import threading
import time

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

# Diretório para salvar eventos detectados
EVENTS_DIR = Path("./eventos_hipoteses")

# Janelas de tempo para análise (em segundos)
WINDOW_MONOTONICITY = 3600      # 1 hora para análise de monotonicidade
WINDOW_CORRELATION_LAG = 300    # 5 minutos para detectar atrasos em correlacionados
WINDOW_PRICING = 60             # 1 minuto para detectar anomalias de precificação

# Thresholds
MONOTONICITY_MIN_MOVES = 3      # Mínimo de movimentos para avaliar monotonicidade
PRICING_DEVIATION_THRESHOLD = 0.02  # 2% de desvio para considerar mispricing
CORRELATION_LAG_THRESHOLD = 30  # 30 segundos de atraso para considerar "lag"


# ============================================================================
# ESTRUTURAS DE DADOS
# ============================================================================

@dataclass
class OddSnapshot:
    """Snapshot de uma best odd em um momento."""
    timestamp: datetime.datetime
    event_id: str
    market_type: str       # Ex: "AH", "OU", "ML"
    line: float            # Ex: -0.5, +1.5, 2.5
    side: str              # Ex: "home", "away", "over", "under"
    best_odd: float
    source: str = ""       # Casa de origem da best odd (se disponível)


@dataclass
class PricingEvent:
    """Evento de precificação incorreta detectado (H1)."""
    timestamp: datetime.datetime
    event_id: str
    market_type: str
    line: float
    
    # Odds observadas
    odd_side_a: float      # Ex: odd do home
    odd_side_b: float      # Ex: odd do away
    
    # Cálculos de precificação
    implied_prob_total: float    # Soma das probabilidades implícitas
    overround: float             # Margem/vig (>1 = normal, <1 = arbitragem)
    fair_odd_a: float            # Odd justa calculada
    fair_odd_b: float
    deviation_a: float           # Desvio da odd real vs justa
    deviation_b: float
    
    # Classificação
    is_arb: bool                 # Se há arbitragem
    mispriced_side: Optional[str]  # Qual lado está mal precificado
    edge_estimate: float         # Estimativa de edge


@dataclass  
class MonotonicityEvent:
    """Evento de quebra de monotonicidade detectado (H3)."""
    timestamp: datetime.datetime
    event_id: str
    market_type: str
    line: float
    side: str
    
    # Histórico de movimentos
    moves_history: List[Tuple[datetime.datetime, float]]  # [(ts, odd), ...]
    
    # Métricas de quebra
    num_reversals: int           # Número de reversões na janela
    last_reversal_magnitude: float  # Tamanho da última reversão
    direction_before: str        # "up" ou "down" antes da reversão
    direction_after: str         # "up" ou "down" depois
    
    # Contexto
    streak_before_break: int     # Movimentos consecutivos antes da quebra
    time_since_last_reversal: float  # Segundos desde reversão anterior
    oscillation_index: float     # Índice de oscilação


@dataclass
class CorrelationLagEvent:
    """Evento de atraso em odds correlacionadas detectado (H6)."""
    timestamp: datetime.datetime
    event_id: str
    
    # Mercado que moveu primeiro (líder)
    leader_market: str
    leader_line: float
    leader_move_time: datetime.datetime
    leader_move_direction: str   # "up" ou "down"
    leader_move_magnitude: float
    
    # Mercado atrasado (seguidor)
    lagged_market: str
    lagged_line: float
    lagged_current_odd: float
    
    # Métricas de atraso
    lag_seconds: float           # Tempo de atraso
    expected_move: float         # Movimento esperado no mercado correlacionado
    actual_move: float           # Movimento real (pode ser 0 se ainda não moveu)
    correlation_historical: float  # Correlação histórica entre os mercados


# ============================================================================
# H1 - DETECTOR DE PRECIFICAÇÃO INCORRETA
# ============================================================================

class PricingMonitor:
    """
    Monitora pares de odds (ex: home/away, over/under) para detectar
    precificação incorreta em tempo real.
    
    Cálculo de preço justo:
    - Probabilidade implícita = 1 / odd
    - Overround = prob_a + prob_b (normalmente > 1)
    - Fair odd = odd_atual * (prob_a + prob_b) / 1
    - Edge = (fair_odd - odd_atual) / odd_atual
    """
    
    def __init__(self, events_dir: Path = EVENTS_DIR):
        self.events_dir = events_dir
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.events_file = self.events_dir / "pricing_events.jsonl"
        
        # Buffer de odds recentes por evento/mercado
        # Chave: (event_id, market_type, line)
        # Valor: {side: OddSnapshot}
        self.current_odds: Dict[Tuple[str, str, float], Dict[str, OddSnapshot]] = {}
        
        self._lock = threading.Lock()
    
    def update_odd(self, snapshot: OddSnapshot) -> Optional[PricingEvent]:
        """
        Atualiza uma odd e verifica se há mispricing.
        Retorna PricingEvent se detectar anomalia.
        """
        key = (snapshot.event_id, snapshot.market_type, snapshot.line)
        
        with self._lock:
            if key not in self.current_odds:
                self.current_odds[key] = {}
            
            self.current_odds[key][snapshot.side] = snapshot
            
            # Tenta calcular precificação se temos ambos os lados
            return self._check_pricing(key)
    
    def _check_pricing(self, key: Tuple[str, str, float]) -> Optional[PricingEvent]:
        """Verifica precificação para um mercado completo."""
        odds_dict = self.current_odds.get(key, {})
        
        # Determina os lados esperados baseado no tipo de mercado
        market_type = key[1]
        if market_type in ("AH", "ML", "DNB"):
            sides = ("home", "away")
        elif market_type in ("OU", "Goals"):
            sides = ("over", "under")
        else:
            return None
        
        # Verifica se temos ambos os lados
        if not all(s in odds_dict for s in sides):
            return None
        
        snap_a = odds_dict[sides[0]]
        snap_b = odds_dict[sides[1]]
        
        # Verifica se odds são recentes (dentro da janela)
        now = datetime.datetime.now(datetime.timezone.utc)
        max_age = datetime.timedelta(seconds=WINDOW_PRICING)
        
        if (now - snap_a.timestamp) > max_age or (now - snap_b.timestamp) > max_age:
            return None
        
        # Calcula probabilidades implícitas
        if snap_a.best_odd <= 1 or snap_b.best_odd <= 1:
            return None  # Odds inválidas
        
        prob_a = 1.0 / snap_a.best_odd
        prob_b = 1.0 / snap_b.best_odd
        overround = prob_a + prob_b
        
        # Calcula odds justas (removendo a margem)
        if overround == 0:
            return None
        
        fair_prob_a = prob_a / overround
        fair_prob_b = prob_b / overround
        fair_odd_a = 1.0 / fair_prob_a if fair_prob_a > 0 else 0
        fair_odd_b = 1.0 / fair_prob_b if fair_prob_b > 0 else 0
        
        # Calcula desvios
        deviation_a = (snap_a.best_odd - fair_odd_a) / fair_odd_a if fair_odd_a > 0 else 0
        deviation_b = (snap_b.best_odd - fair_odd_b) / fair_odd_b if fair_odd_b > 0 else 0
        
        # Detecta anomalias
        is_arb = overround < 1.0
        mispriced_side = None
        edge_estimate = 0.0
        
        if abs(deviation_a) > PRICING_DEVIATION_THRESHOLD:
            mispriced_side = sides[0]
            edge_estimate = deviation_a
        elif abs(deviation_b) > PRICING_DEVIATION_THRESHOLD:
            mispriced_side = sides[1]
            edge_estimate = deviation_b
        
        # Se há anomalia, cria evento
        if is_arb or mispriced_side:
            event = PricingEvent(
                timestamp=now,
                event_id=key[0],
                market_type=key[1],
                line=key[2],
                odd_side_a=snap_a.best_odd,
                odd_side_b=snap_b.best_odd,
                implied_prob_total=overround,
                overround=overround,
                fair_odd_a=fair_odd_a,
                fair_odd_b=fair_odd_b,
                deviation_a=deviation_a,
                deviation_b=deviation_b,
                is_arb=is_arb,
                mispriced_side=mispriced_side,
                edge_estimate=edge_estimate,
            )
            self._save_event(event)
            return event
        
        return None
    
    def _save_event(self, event: PricingEvent) -> None:
        """Salva evento em arquivo JSONL."""
        data = asdict(event)
        # Converte datetime para ISO string
        data["timestamp"] = event.timestamp.isoformat()
        
        with open(self.events_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


# ============================================================================
# H3 - DETECTOR DE QUEBRAS DE MONOTONICIDADE
# ============================================================================

class MonotonicityMonitor:
    """
    Monitora séries de odds para detectar quebras de monotonicidade
    (reversões de direção).
    
    Definição:
    - Movimento monotônico: odds só sobem OU só descem
    - Quebra: mudança de direção (subindo -> descendo ou vice-versa)
    """
    
    def __init__(self, events_dir: Path = EVENTS_DIR):
        self.events_dir = events_dir
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.events_file = self.events_dir / "monotonicity_events.jsonl"
        
        # Histórico de odds por mercado
        # Chave: (event_id, market_type, line, side)
        # Valor: deque de (timestamp, odd)
        self.history: Dict[Tuple[str, str, float, str], deque] = {}
        
        # Última direção conhecida por mercado
        self.last_direction: Dict[Tuple[str, str, float, str], str] = {}
        
        # Contador de reversões por mercado
        self.reversal_count: Dict[Tuple[str, str, float, str], int] = {}
        
        # Timestamp da última reversão
        self.last_reversal_time: Dict[Tuple[str, str, float, str], datetime.datetime] = {}
        
        self._lock = threading.Lock()
    
    def update_odd(self, snapshot: OddSnapshot) -> Optional[MonotonicityEvent]:
        """
        Atualiza uma odd e verifica se houve quebra de monotonicidade.
        """
        key = (snapshot.event_id, snapshot.market_type, snapshot.line, snapshot.side)
        now = snapshot.timestamp
        
        with self._lock:
            # Inicializa estruturas se necessário
            if key not in self.history:
                self.history[key] = deque(maxlen=100)  # Mantém últimos 100 pontos
                self.reversal_count[key] = 0
            
            history = self.history[key]
            
            # Limpa pontos antigos (fora da janela)
            window_start = now - datetime.timedelta(seconds=WINDOW_MONOTONICITY)
            while history and history[0][0] < window_start:
                history.popleft()
            
            # Se não há histórico suficiente, apenas adiciona
            if len(history) < 1:
                history.append((now, snapshot.best_odd))
                return None
            
            # Calcula direção do movimento atual
            last_odd = history[-1][1]
            if snapshot.best_odd > last_odd:
                current_direction = "up"
            elif snapshot.best_odd < last_odd:
                current_direction = "down"
            else:
                # Sem movimento, apenas adiciona ao histórico
                history.append((now, snapshot.best_odd))
                return None
            
            # Verifica se houve reversão
            event = None
            if key in self.last_direction:
                prev_direction = self.last_direction[key]
                
                if prev_direction != current_direction:
                    # REVERSÃO DETECTADA!
                    self.reversal_count[key] += 1
                    
                    # Calcula métricas
                    streak = self._calculate_streak(history, prev_direction)
                    magnitude = abs(snapshot.best_odd - last_odd)
                    
                    time_since_last = None
                    if key in self.last_reversal_time:
                        time_since_last = (now - self.last_reversal_time[key]).total_seconds()
                    
                    # Calcula índice de oscilação
                    total_moves = len(history)
                    oscillation_idx = self.reversal_count[key] / total_moves if total_moves > 0 else 0
                    
                    event = MonotonicityEvent(
                        timestamp=now,
                        event_id=snapshot.event_id,
                        market_type=snapshot.market_type,
                        line=snapshot.line,
                        side=snapshot.side,
                        moves_history=list(history)[-10:],  # Últimos 10 pontos
                        num_reversals=self.reversal_count[key],
                        last_reversal_magnitude=magnitude,
                        direction_before=prev_direction,
                        direction_after=current_direction,
                        streak_before_break=streak,
                        time_since_last_reversal=time_since_last or 0,
                        oscillation_index=oscillation_idx,
                    )
                    
                    self.last_reversal_time[key] = now
                    self._save_event(event)
            
            # Atualiza estado
            self.last_direction[key] = current_direction
            history.append((now, snapshot.best_odd))
            
            return event
    
    def _calculate_streak(self, history: deque, direction: str) -> int:
        """Calcula quantos movimentos consecutivos na mesma direção."""
        if len(history) < 2:
            return 0
        
        streak = 0
        items = list(history)
        
        for i in range(len(items) - 1, 0, -1):
            curr_odd = items[i][1]
            prev_odd = items[i-1][1]
            
            if direction == "up" and curr_odd > prev_odd:
                streak += 1
            elif direction == "down" and curr_odd < prev_odd:
                streak += 1
            else:
                break
        
        return streak
    
    def _save_event(self, event: MonotonicityEvent) -> None:
        """Salva evento em arquivo JSONL."""
        data = asdict(event)
        data["timestamp"] = event.timestamp.isoformat()
        # Converte histórico de movimentos
        data["moves_history"] = [
            (ts.isoformat(), odd) for ts, odd in event.moves_history
        ]
        
        with open(self.events_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


# ============================================================================
# H6 - DETECTOR DE ATRASOS EM ODDS CORRELACIONADAS
# ============================================================================

class CorrelationLagMonitor:
    """
    Monitora odds de mercados correlacionados para detectar atrasos
    na movimentação.
    
    Correlações monitoradas (mesmo evento):
    - Asian Handicap linhas adjacentes (ex: -0.5 e -0.75)
    - Over/Under linhas adjacentes (ex: 2.5 e 3.0)
    - Asian Handicap home vs away (inverso)
    - Handicap vs Moneyline (quando linha = 0)
    """
    
    # Definição de mercados correlacionados
    CORRELATIONS = {
        # (market_type, line) -> lista de mercados correlacionados com coef esperado
        # coef > 0: movem na mesma direção
        # coef < 0: movem em direção oposta
    }
    
    def __init__(self, events_dir: Path = EVENTS_DIR):
        self.events_dir = events_dir
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.events_file = self.events_dir / "correlation_lag_events.jsonl"
        
        # Último movimento por mercado
        # Chave: (event_id, market_type, line, side)
        # Valor: (timestamp, old_odd, new_odd, direction)
        self.last_moves: Dict[Tuple[str, str, float, str], Tuple] = {}
        
        # Odds atuais por mercado
        self.current_odds: Dict[Tuple[str, str, float, str], OddSnapshot] = {}
        
        self._lock = threading.Lock()
    
    def update_odd(self, snapshot: OddSnapshot) -> List[CorrelationLagEvent]:
        """
        Atualiza uma odd e verifica se há atrasos em mercados correlacionados.
        Retorna lista de eventos de lag detectados.
        """
        key = (snapshot.event_id, snapshot.market_type, snapshot.line, snapshot.side)
        now = snapshot.timestamp
        events = []
        
        with self._lock:
            # Verifica se houve movimento
            old_snap = self.current_odds.get(key)
            
            if old_snap and old_snap.best_odd != snapshot.best_odd:
                # Movimento detectado!
                direction = "up" if snapshot.best_odd > old_snap.best_odd else "down"
                magnitude = abs(snapshot.best_odd - old_snap.best_odd)
                
                # Registra o movimento
                self.last_moves[key] = (now, old_snap.best_odd, snapshot.best_odd, direction)
                
                # Verifica mercados correlacionados que deveriam ter movido
                correlated = self._get_correlated_markets(snapshot.event_id, snapshot.market_type, snapshot.line, snapshot.side)
                
                for corr_market, corr_line, corr_side, expected_coef in correlated:
                    corr_key = (snapshot.event_id, corr_market, corr_line, corr_side)
                    
                    # Verifica se o mercado correlacionado moveu recentemente
                    if corr_key in self.last_moves:
                        last_move = self.last_moves[corr_key]
                        move_age = (now - last_move[0]).total_seconds()
                        
                        if move_age < CORRELATION_LAG_THRESHOLD:
                            # Mercado correlacionado moveu recentemente, OK
                            continue
                    
                    # Se chegou aqui, mercado correlacionado NÃO moveu ou está atrasado
                    if corr_key in self.current_odds:
                        corr_snap = self.current_odds[corr_key]
                        
                        # Calcula movimento esperado
                        expected_direction = direction if expected_coef > 0 else ("down" if direction == "up" else "up")
                        expected_move = magnitude * abs(expected_coef)
                        
                        # Verifica quanto tempo desde último movimento do correlacionado
                        lag_seconds = CORRELATION_LAG_THRESHOLD  # Default
                        if corr_key in self.last_moves:
                            lag_seconds = (now - self.last_moves[corr_key][0]).total_seconds()
                        
                        if lag_seconds >= CORRELATION_LAG_THRESHOLD:
                            event = CorrelationLagEvent(
                                timestamp=now,
                                event_id=snapshot.event_id,
                                leader_market=snapshot.market_type,
                                leader_line=snapshot.line,
                                leader_move_time=now,
                                leader_move_direction=direction,
                                leader_move_magnitude=magnitude,
                                lagged_market=corr_market,
                                lagged_line=corr_line,
                                lagged_current_odd=corr_snap.best_odd,
                                lag_seconds=lag_seconds,
                                expected_move=expected_move,
                                actual_move=0.0,  # Ainda não moveu
                                correlation_historical=expected_coef,
                            )
                            events.append(event)
                            self._save_event(event)
            
            # Atualiza odd atual
            self.current_odds[key] = snapshot
        
        return events
    
    def _get_correlated_markets(
        self, 
        event_id: str, 
        market_type: str, 
        line: float, 
        side: str
    ) -> List[Tuple[str, float, str, float]]:
        """
        Retorna lista de mercados correlacionados.
        Cada item: (market_type, line, side, correlation_coefficient)
        """
        correlated = []
        
        if market_type == "AH":
            # Linhas adjacentes de Asian Handicap
            for delta in [-0.25, -0.5, 0.25, 0.5]:
                adj_line = line + delta
                # Mesmo lado, correlação positiva alta
                correlated.append((market_type, adj_line, side, 0.9))
            
            # Lado oposto, correlação negativa
            opposite_side = "away" if side == "home" else "home"
            correlated.append((market_type, -line, opposite_side, -0.95))
        
        elif market_type == "OU":
            # Linhas adjacentes de Over/Under
            for delta in [-0.5, 0.5]:
                adj_line = line + delta
                correlated.append((market_type, adj_line, side, 0.85))
            
            # Lado oposto
            opposite_side = "under" if side == "over" else "over"
            correlated.append((market_type, line, opposite_side, -0.98))
        
        return correlated
    
    def _save_event(self, event: CorrelationLagEvent) -> None:
        """Salva evento em arquivo JSONL."""
        data = asdict(event)
        data["timestamp"] = event.timestamp.isoformat()
        data["leader_move_time"] = event.leader_move_time.isoformat()
        
        with open(self.events_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


# ============================================================================
# AGREGADOR DE EVENTOS
# ============================================================================

class HypothesisMonitor:
    """
    Agregador que combina os três monitores e facilita a integração
    com o sistema existente.
    """
    
    def __init__(self, events_dir: Path = EVENTS_DIR):
        self.pricing = PricingMonitor(events_dir)
        self.monotonicity = MonotonicityMonitor(events_dir)
        self.correlation = CorrelationLagMonitor(events_dir)
        self.events_dir = events_dir
    
    def process_odd_update(
        self,
        event_id: str,
        market_type: str,
        line: float,
        side: str,
        best_odd: float,
        source: str = "",
        timestamp: Optional[datetime.datetime] = None
    ) -> Dict[str, Any]:
        """
        Processa uma atualização de odd e retorna eventos detectados.
        
        Esta é a função principal para integrar com o sistema existente.
        Chame-a sempre que uma best odd for atualizada.
        """
        if timestamp is None:
            timestamp = datetime.datetime.now(datetime.timezone.utc)
        
        snapshot = OddSnapshot(
            timestamp=timestamp,
            event_id=event_id,
            market_type=market_type,
            line=line,
            side=side,
            best_odd=best_odd,
            source=source,
        )
        
        results = {
            "pricing_event": None,
            "monotonicity_event": None,
            "correlation_events": [],
        }
        
        # H1 - Precificação
        pricing_event = self.pricing.update_odd(snapshot)
        if pricing_event:
            results["pricing_event"] = asdict(pricing_event)
        
        # H3 - Monotonicidade
        mono_event = self.monotonicity.update_odd(snapshot)
        if mono_event:
            results["monotonicity_event"] = asdict(mono_event)
        
        # H6 - Correlação
        corr_events = self.correlation.update_odd(snapshot)
        if corr_events:
            results["correlation_events"] = [asdict(e) for e in corr_events]
        
        return results
    
    def get_summary_for_bet(self, event_id: str) -> Dict[str, Any]:
        """
        Retorna resumo de eventos detectados para um evento específico,
        pronto para merge com a tabela resumo de apostas.
        """
        summary = {
            "event_id": event_id,
            
            # H1 - Métricas de precificação
            "h1_pricing_events_count": 0,
            "h1_last_mispricing_detected": None,
            "h1_avg_edge": 0.0,
            "h1_had_arb": False,
            
            # H3 - Métricas de monotonicidade
            "h3_total_reversals": 0,
            "h3_last_reversal_time": None,
            "h3_avg_oscillation_index": 0.0,
            "h3_max_reversal_magnitude": 0.0,
            
            # H6 - Métricas de correlação/lag
            "h6_lag_events_count": 0,
            "h6_avg_lag_seconds": 0.0,
            "h6_max_lag_seconds": 0.0,
            "h6_markets_with_lag": [],
        }
        
        # Carrega eventos dos arquivos e agrega
        # (implementação simplificada - em produção, usar banco de dados)
        
        for events_file, prefix in [
            (self.pricing.events_file, "h1"),
            (self.monotonicity.events_file, "h3"),
            (self.correlation.events_file, "h6"),
        ]:
            if events_file.exists():
                with open(events_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            evt = json.loads(line)
                            if evt.get("event_id") == event_id:
                                self._aggregate_event(summary, evt, prefix)
                        except json.JSONDecodeError:
                            continue
        
        return summary
    
    def _aggregate_event(self, summary: Dict, event: Dict, prefix: str) -> None:
        """Agrega um evento ao resumo."""
        if prefix == "h1":
            summary["h1_pricing_events_count"] += 1
            summary["h1_last_mispricing_detected"] = event.get("timestamp")
            if event.get("is_arb"):
                summary["h1_had_arb"] = True
            # Média de edge
            edge = event.get("edge_estimate", 0)
            n = summary["h1_pricing_events_count"]
            summary["h1_avg_edge"] = (summary["h1_avg_edge"] * (n-1) + edge) / n
        
        elif prefix == "h3":
            summary["h3_total_reversals"] += 1
            summary["h3_last_reversal_time"] = event.get("timestamp")
            mag = event.get("last_reversal_magnitude", 0)
            summary["h3_max_reversal_magnitude"] = max(summary["h3_max_reversal_magnitude"], mag)
            osc = event.get("oscillation_index", 0)
            n = summary["h3_total_reversals"]
            summary["h3_avg_oscillation_index"] = (summary["h3_avg_oscillation_index"] * (n-1) + osc) / n
        
        elif prefix == "h6":
            summary["h6_lag_events_count"] += 1
            lag = event.get("lag_seconds", 0)
            summary["h6_max_lag_seconds"] = max(summary["h6_max_lag_seconds"], lag)
            n = summary["h6_lag_events_count"]
            summary["h6_avg_lag_seconds"] = (summary["h6_avg_lag_seconds"] * (n-1) + lag) / n
            
            market = event.get("lagged_market", "")
            if market and market not in summary["h6_markets_with_lag"]:
                summary["h6_markets_with_lag"].append(market)


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # Exemplo de como usar o monitor
    monitor = HypothesisMonitor()
    
    # Simula atualizações de odds
    event_id = "match_12345"
    
    print("Simulando atualizações de odds...")
    
    # Atualização 1: Home AH -0.5
    result = monitor.process_odd_update(
        event_id=event_id,
        market_type="AH",
        line=-0.5,
        side="home",
        best_odd=1.95,
    )
    print(f"Update 1: {result}")
    
    # Atualização 2: Away AH +0.5 (par para calcular pricing)
    result = monitor.process_odd_update(
        event_id=event_id,
        market_type="AH",
        line=-0.5,
        side="away",  # Este é o lado oposto
        best_odd=1.90,
    )
    print(f"Update 2: {result}")
    
    # Atualização 3: Home sobe (movimento)
    result = monitor.process_odd_update(
        event_id=event_id,
        market_type="AH",
        line=-0.5,
        side="home",
        best_odd=2.00,
    )
    print(f"Update 3: {result}")
    
    # Atualização 4: Home desce (reversão!)
    result = monitor.process_odd_update(
        event_id=event_id,
        market_type="AH",
        line=-0.5,
        side="home",
        best_odd=1.92,
    )
    print(f"Update 4 (reversão): {result}")
    
    # Obtém resumo para a tabela
    summary = monitor.get_summary_for_bet(event_id)
    print(f"\nResumo para merge com tabela:\n{json.dumps(summary, indent=2, default=str)}")
