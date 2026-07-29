"""H3BUP CLV forward-collection config (fail-open, analytics-only)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return int(default)


@dataclass
class ClvConfig:
    enabled: bool = False
    create_obligations: bool = False
    worker_enabled: bool = False
    passive_collector_enabled: bool = False
    allow_betslip_source: bool = False
    fair_edge_enabled: bool = False
    post_5m_enabled: bool = True
    post_15m_enabled: bool = True
    closing_enabled: bool = True
    source_priority: List[str] = None
    poll_sec: float = 30.0
    batch_size: int = 100
    max_attempts: int = 10
    retry_base_sec: float = 60.0
    processing_lease_sec: float = 300.0
    post_5m_tol_before_sec: float = 60.0
    post_5m_tol_after_sec: float = 120.0
    post_15m_tol_before_sec: float = 90.0
    post_15m_tol_after_sec: float = 180.0
    closing_buffer_sec: float = 30.0
    closing_max_age_sec: float = 3600.0
    schema_version: int = 1
    collection_started_path: str = "logs/h3bup_clv_collection_started_at.txt"
    health_path: str = "logs/h3bup_clv_health.json"
    obligations_jsonl: str = "logs/h3bup_clv_obligations.jsonl"
    snapshots_jsonl: str = "logs/h3bup_clv_snapshots.jsonl"
    passive_jsonl: str = "logs/h3bup_clv_passive_snapshots.jsonl"
    passive_health_path: str = "logs/h3bup_passive_collector_health.json"

    def __post_init__(self) -> None:
        if self.source_priority is None:
            self.source_priority = ["best_odds_history", "passive_collector"]


def load_config() -> ClvConfig:
    pri = os.getenv("H3BUP_CLV_SOURCE_PRIORITY", "best_odds_history,passive_collector")
    sources = [x.strip() for x in str(pri).split(",") if x.strip()]
    return ClvConfig(
        enabled=_env_bool("H3BUP_CLV_ENABLED", False),
        create_obligations=_env_bool("H3BUP_CLV_CREATE_OBLIGATIONS", False),
        worker_enabled=_env_bool("H3BUP_CLV_WORKER_ENABLED", False),
        passive_collector_enabled=_env_bool("H3BUP_CLV_PASSIVE_COLLECTOR_ENABLED", False),
        allow_betslip_source=_env_bool("H3BUP_CLV_ALLOW_BETSLIP_SOURCE", False),
        fair_edge_enabled=_env_bool("H3BUP_CLV_FAIR_EDGE_ENABLED", False),
        post_5m_enabled=_env_bool("H3BUP_CLV_POST_5M_ENABLED", True),
        post_15m_enabled=_env_bool("H3BUP_CLV_POST_15M_ENABLED", True),
        closing_enabled=_env_bool("H3BUP_CLV_CLOSING_ENABLED", True),
        source_priority=sources or ["best_odds_history"],
        poll_sec=_env_float("H3BUP_CLV_POLL_SEC", 30.0),
        batch_size=_env_int("H3BUP_CLV_BATCH_SIZE", 100),
        max_attempts=_env_int("H3BUP_CLV_MAX_ATTEMPTS", 10),
        retry_base_sec=_env_float("H3BUP_CLV_RETRY_BASE_SEC", 60.0),
        processing_lease_sec=_env_float("H3BUP_CLV_PROCESSING_LEASE_SEC", 300.0),
        post_5m_tol_before_sec=_env_float("H3BUP_CLV_POST_5M_TOLERANCE_BEFORE_SEC", 60.0),
        post_5m_tol_after_sec=_env_float("H3BUP_CLV_POST_5M_TOLERANCE_AFTER_SEC", 120.0),
        post_15m_tol_before_sec=_env_float("H3BUP_CLV_POST_15M_TOLERANCE_BEFORE_SEC", 90.0),
        post_15m_tol_after_sec=_env_float("H3BUP_CLV_POST_15M_TOLERANCE_AFTER_SEC", 180.0),
        closing_buffer_sec=_env_float("H3BUP_CLV_CLOSING_BUFFER_SEC", 30.0),
        closing_max_age_sec=_env_float("H3BUP_CLV_CLOSING_MAX_AGE_SEC", 3600.0),
        schema_version=_env_int("H3BUP_CLV_SCHEMA_VERSION", 1),
    )


WINDOWS = ("POST_5M", "POST_15M", "CLOSING")

STATUSES = (
    "PENDING",
    "WAITING_TARGET",
    "READY",
    "PROCESSING",
    "COMPLETED",
    "RETRYABLE",
    "FAILED_FINAL",
    "SKIPPED",
    "CANCELLED",
)
