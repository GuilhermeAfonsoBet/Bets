from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ExecSide(str, Enum):
    BACK = "Back"
    LAY = "Lay"


class MarketType(str, Enum):
    AH = "AH"


class ExecStatus(str, Enum):
    DRY_OK = "DRY_OK"
    LIVE_OK = "LIVE_OK"
    STALE = "STALE"
    API_BACKOFF = "API_BACKOFF"
    CAP_BLOCKED = "CAP_BLOCKED"
    NO_SESSION = "NO_SESSION"
    RATE_LIMIT = "RATE_LIMIT"
    API_FAILED = "API_FAILED"
    BAD_REQUEST = "BAD_REQUEST"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ExecutionPolicy(BaseModel):
    """
    Campo "espelho" do que o Decision Engine decidiu.
    O executor NÃO recalcula sizing: ele mede execução/odds/tempo.
    """

    policy_version: str = "risk_sqrt_eq4_cap33_v1"

    bankroll_ref: Optional[float] = None
    bud_back_frac: Optional[float] = None
    bud_lay_frac: Optional[float] = None
    cap_signal_frac: Optional[float] = None
    risk_mode: Optional[str] = None  # fixed/signals_sqrt/signals_linear

    stake_requested: Optional[float] = None  # Back
    liability_requested: Optional[float] = None  # Lay

    match_id: Optional[int] = None
    spent_before: Optional[float] = None
    spent_after: Optional[float] = None


class ExecutionRequest(BaseModel):
    execution_id: UUID = Field(default_factory=uuid4)

    created_at: datetime = Field(..., description="Timestamp UTC do Decision Engine (quando decidiu executar).")
    audit_id: Optional[int] = None
    match_id: Optional[int] = None

    # Identidade do mercado
    event_id: str = Field(..., description="event_id usado pelo site (ex.: '2026-02-08,176,178').")
    market_type: MarketType = MarketType.AH
    side: str = Field(..., description="home/away para AH.")
    line: str = Field(..., description="Linha AH como string (ex.: '-1', '0', '+0.5').")

    exec_side: ExecSide = ExecSide.BACK
    is_live: bool = False

    # Odds observadas na decisão (para medir slippage)
    odd_at_decision: Optional[float] = None

    # Staleness
    max_late_ms: int = 8000

    policy: ExecutionPolicy = Field(default_factory=ExecutionPolicy)
    meta: Dict[str, Any] = Field(default_factory=dict)


class ExecutionTiming(BaseModel):
    queue_delay_ms: Optional[int] = None
    call_to_done_ms: Optional[int] = None
    post_ms: Optional[int] = None
    total_ms: Optional[int] = None
    # Decomposição (API/WS): tempo de espera por PMMs (parte do total_ms).
    pmm_wait_ms: Optional[int] = None


class ExecutionResult(BaseModel):
    execution_id: UUID
    status: ExecStatus
    created_at: datetime
    finished_at: datetime

    # Mercado
    audit_id: Optional[int] = None
    match_id: Optional[int] = None
    event_id: str
    market_type: MarketType
    side: str
    line: str
    exec_side: ExecSide
    is_live: bool

    # Odds capturadas no momento do ticket
    odd_at_decision: Optional[float] = None
    odd_final: Optional[float] = None
    bookie_final: Optional[str] = None
    limit_final: Optional[float] = None
    num_bk: Optional[int] = None

    delta_odds: Optional[float] = None
    delta_pct: Optional[float] = None

    timing: ExecutionTiming = Field(default_factory=ExecutionTiming)
    policy: ExecutionPolicy = Field(default_factory=ExecutionPolicy)

    # Erros/rate limit
    http_status: Optional[int] = None
    retry_after_sec: Optional[int] = None
    error: Optional[str] = None

    raw: Dict[str, Any] = Field(default_factory=dict)

