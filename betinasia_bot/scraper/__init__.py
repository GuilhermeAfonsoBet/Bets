# -*- coding: utf-8 -*-
"""Módulo de scraping do BetinAsia."""

from .betinasia import BetinAsiaScraper
from .models import MatchData, AHLine, BookmakerOdds

__all__ = ["BetinAsiaScraper", "MatchData", "AHLine", "BookmakerOdds"]
