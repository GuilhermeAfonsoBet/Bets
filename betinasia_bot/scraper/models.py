# -*- coding: utf-8 -*-
"""
Modelos de dados para o scraper.
Define as estruturas de dados extraídas do BetinAsia.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import numpy as np


@dataclass
class BookmakerOdds:
    """Odds de um bookmaker específico para uma linha de AH."""
    
    bookmaker: str
    home_odds: float
    away_odds: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __str__(self):
        return f"{self.bookmaker}: H={self.home_odds:.2f} A={self.away_odds:.2f}"


@dataclass
class AHLine:
    """Uma linha de Asian Handicap com odds de múltiplos bookmakers."""
    
    line: str  # Ex: "+0.5", "-0.75", "+0"
    bookmaker_odds: dict[str, BookmakerOdds] = field(default_factory=dict)
    
    @property
    def best_home_odds(self) -> tuple[str, float]:
        """Retorna (bookmaker, odds) com melhor odd para home."""
        if not self.bookmaker_odds:
            return ("", 0.0)
        best = max(self.bookmaker_odds.items(), key=lambda x: x[1].home_odds)
        return best[0], best[1].home_odds
    
    @property
    def best_away_odds(self) -> tuple[str, float]:
        """Retorna (bookmaker, odds) com melhor odd para away."""
        if not self.bookmaker_odds:
            return ("", 0.0)
        best = max(self.bookmaker_odds.items(), key=lambda x: x[1].away_odds)
        return best[0], best[1].away_odds
    
    @property
    def num_bookmakers(self) -> int:
        """Número de bookmakers oferecendo esta linha."""
        return len(self.bookmaker_odds)
    
    @property
    def home_odds_list(self) -> list[float]:
        """Lista de todas as odds de home."""
        return [bk.home_odds for bk in self.bookmaker_odds.values()]
    
    @property
    def away_odds_list(self) -> list[float]:
        """Lista de todas as odds de away."""
        return [bk.away_odds for bk in self.bookmaker_odds.values()]
    
    @property
    def home_odds_median(self) -> float:
        """Mediana das odds de home."""
        odds = self.home_odds_list
        return float(np.median(odds)) if odds else 0.0
    
    @property
    def away_odds_median(self) -> float:
        """Mediana das odds de away."""
        odds = self.away_odds_list
        return float(np.median(odds)) if odds else 0.0
    
    def get_dif_best_second_home(self) -> float:
        """Diferença percentual entre melhor e segunda melhor odd (home)."""
        odds = sorted(self.home_odds_list, reverse=True)
        if len(odds) < 2:
            return 0.0
        return (odds[0] - odds[1]) / odds[1] * 100
    
    def get_dif_best_median_home(self) -> float:
        """Diferença percentual entre melhor odd e mediana (home)."""
        best = self.best_home_odds[1]
        median = self.home_odds_median
        if median == 0:
            return 0.0
        return (best - median) / median * 100
    
    def get_pinnacle_odds(self, side: str = "home") -> Optional[float]:
        """Retorna odds da Pinnacle (pin88) se disponível."""
        pin = self.bookmaker_odds.get("pin88") or self.bookmaker_odds.get("pinnacle")
        if pin:
            return pin.home_odds if side == "home" else pin.away_odds
        return None
    
    def __str__(self):
        best_bk, best_odds = self.best_home_odds
        return f"AH {self.line}: {best_odds:.2f} @ {best_bk} ({self.num_bookmakers} casas)"


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
    
    # Resultado (preenchido depois)
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    status: str = "scheduled"  # scheduled, live, finished
    
    @property
    def minutes_to_kickoff(self) -> int:
        """Minutos até o início da partida."""
        delta = self.kickoff_time - datetime.now(timezone.utc)
        return max(0, int(delta.total_seconds() / 60))
    
    @property
    def is_upcoming(self) -> bool:
        """Retorna True se o jogo ainda não começou."""
        return self.minutes_to_kickoff > 0
    
    def get_main_ah_line(self) -> Optional[AHLine]:
        """
        Retorna a linha de AH principal (mais próxima de 0).
        Ex: se tem +0.5 e +1.5, retorna +0.5
        """
        if not self.ah_lines:
            return None
        
        # Ordena por valor absoluto da linha
        sorted_lines = sorted(
            self.ah_lines.items(),
            key=lambda x: abs(float(x[0].replace("+", "")))
        )
        return sorted_lines[0][1] if sorted_lines else None
    
    def __str__(self):
        return (
            f"{self.home_team} vs {self.away_team} "
            f"({self.league}) - {self.minutes_to_kickoff}min"
        )


@dataclass
class ScrapedOpportunity:
    """
    Uma oportunidade detectada durante o scraping.
    Combina dados da partida com uma linha específica de AH.
    """
    
    match: MatchData
    ah_line: AHLine
    side: str  # "home" ou "away"
    
    # Dados da oportunidade
    best_odds: float
    best_bookmaker: str
    
    # Features calculadas
    num_bookmakers: int
    dif_pct_best_second: float
    dif_pct_best_median: float
    dif_vs_pinnacle: Optional[float]
    
    # Timestamps
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict:
        """Converte para dicionário (útil para salvar em JSON/banco)."""
        return {
            "match_id": self.match.match_id,
            "league": self.match.league,
            "home_team": self.match.home_team,
            "away_team": self.match.away_team,
            "kickoff_time": self.match.kickoff_time.isoformat(),
            "ah_line": self.ah_line.line,
            "side": self.side,
            "best_odds": self.best_odds,
            "best_bookmaker": self.best_bookmaker,
            "num_bookmakers": self.num_bookmakers,
            "dif_pct_best_second": self.dif_pct_best_second,
            "dif_pct_best_median": self.dif_pct_best_median,
            "dif_vs_pinnacle": self.dif_vs_pinnacle,
            "minutes_to_kickoff": self.match.minutes_to_kickoff,
            "detected_at": self.detected_at.isoformat(),
        }
