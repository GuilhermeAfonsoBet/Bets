# -*- coding: utf-8 -*-
"""
Módulo de detecção de hipóteses.

Este módulo implementa detectores para as hipóteses:
- H1: Precificação incorreta
- H3: Quebra de monotonicidade entre linhas adjacentes
- H3b: Reversões temporais de odds
- H6: Atrasos em odds correlacionadas
"""

from .detectors import (
    HypothesisDetector,
    H1PricingDetector,
    H3LineMonotonicityDetector,
    H3bTemporalReversalDetector,
    H6CorrelationLagDetector,
)

__all__ = [
    "HypothesisDetector",
    "H1PricingDetector",
    "H3LineMonotonicityDetector",
    "H3bTemporalReversalDetector",
    "H6CorrelationLagDetector",
]
