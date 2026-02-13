# -*- coding: utf-8 -*-
"""Módulo de scraping do BetinAsia."""

from .betinasia import BetinAsiaScraper
from .models import MatchData, AHLine, BookmakerOdds
from .websocket_collector import WebSocketCollector, MatchOdds, AHOdds
from .fast_collector import FastCollector, CollectionResult

__all__ = [
    "BetinAsiaScraper", 
    "MatchData", 
    "AHLine", 
    "BookmakerOdds",
    "WebSocketCollector",
    "FastCollector",
    "CollectionResult",
    "MatchOdds",
    "AHOdds",
]
