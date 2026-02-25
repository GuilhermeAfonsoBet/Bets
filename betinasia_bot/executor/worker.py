from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from loguru import logger

from scraper.betinasia import BetinAsiaScraper
from scraper.api_betslip import ApiBetslipClient, BetslipApiResult

from .contracts import ExecutionRequest, ExecutionResult, ExecStatus, ExecSide, ExecutionTiming


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _ms(dt_s: float) -> int:
    return int(round(dt_s * 1000.0))


def _extract_snapshot(side: ExecSide, api_result: Optional[BetslipApiResult]) -> Tuple[Optional[float], Optional[str], Optional[float], Optional[int], Optional[str]]:
    if not api_result:
        return None, None, None, None, "NO_API_RESULT"
    if not api_result.success:
        return None, None, None, None, api_result.error or "API_FAILED"
    if not api_result.bookmakers:
        return None, None, None, None, "NO_BOOKMAKERS"
    if side == ExecSide.LAY:
        # para Lay queremos o menor preço (odd menor = melhor para Lay)
        cands = [b for b in api_result.bookmakers if b.best_price and b.best_price > 0]
        if not cands:
            return None, None, None, None, "NO_LAY_PRICES"
        best = min(cands, key=lambda b: b.best_price)
        return float(best.best_price), str(best.bookie), float(best.max_stake), int(len(cands)), None
    # Back: maior odd
    return float(api_result.best_odd), str(api_result.best_bookie), float(api_result.best_limit), int(api_result.num_bookmakers), None


@dataclass
class ExecutorWorker:
    """
    Um worker mantém uma sessão Playwright 'quente' (login + WS) e
    executa dry-runs via ApiBetslipClient.
    """

    name: str
    football_url: str
    open_cap_window_sec: float = 300.0
    open_cap_max: int = 999999
    enable_cap: bool = True

    _scraper: Optional[BetinAsiaScraper] = None
    _api: Optional[ApiBetslipClient] = None
    _running: bool = False
    _api_backoff_until_ts: float = 0.0
    _cap_lock: asyncio.Lock = None
    _cap_open_times: Any = None  # deque[float]

    async def start(self) -> None:
        self._cap_lock = asyncio.Lock()
        from collections import deque

        self._cap_open_times = deque()
        self._scraper = BetinAsiaScraper()
        await self._scraper.start()
        ok_login = await self._scraper.login()
        if not ok_login:
            raise RuntimeError("LOGIN_FAILED")
        page = self._scraper._page
        self._api = ApiBetslipClient(page)
        self._api.setup_listener()
        await page.goto(self.football_url)
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(4000)
        self._running = True
        logger.info(f"[executor:{self.name}] started (WS warm)")

    async def close(self) -> None:
        self._running = False
        try:
            if self._scraper:
                await self._scraper.close()
        except Exception:
            pass
        self._scraper = None
        self._api = None

    async def _cap_allow(self) -> Tuple[bool, Dict[str, Any]]:
        if not self.enable_cap:
            return True, {"enabled": False}
        async with self._cap_lock:
            now = time.time()
            # GC
            while self._cap_open_times and (now - float(self._cap_open_times[0])) > float(self.open_cap_window_sec):
                self._cap_open_times.popleft()
            used = len(self._cap_open_times)
            allowed = used < int(self.open_cap_max)
            if allowed:
                self._cap_open_times.append(now)
            return allowed, {"enabled": True, "used": used, "max": int(self.open_cap_max), "window_sec": float(self.open_cap_window_sec)}

    async def execute_dryrun(self, req: ExecutionRequest, received_ts: float) -> ExecutionResult:
        assert self._api is not None
        t0 = time.time()
        finished_at = _now_utc()
        timing = ExecutionTiming(queue_delay_ms=_ms(max(0.0, t0 - float(received_ts))))

        # staleness
        try:
            late_ms = _ms(max(0.0, t0 - float(req.created_at.timestamp())))
        except Exception:
            late_ms = 0
        if late_ms > int(req.max_late_ms):
            return ExecutionResult(
                execution_id=req.execution_id,
                status=ExecStatus.STALE,
                created_at=req.created_at,
                finished_at=_now_utc(),
                audit_id=req.audit_id,
                match_id=req.match_id,
                event_id=req.event_id,
                market_type=req.market_type,
                side=req.side,
                line=req.line,
                exec_side=req.exec_side,
                is_live=bool(req.is_live),
                odd_at_decision=req.odd_at_decision,
                timing=ExecutionTiming(queue_delay_ms=timing.queue_delay_ms, call_to_done_ms=late_ms),
                policy=req.policy,
                error=f"STALE late_ms={late_ms} max_late_ms={int(req.max_late_ms)}",
            )

        # backoff
        if self._api_backoff_until_ts and time.time() < float(self._api_backoff_until_ts):
            return ExecutionResult(
                execution_id=req.execution_id,
                status=ExecStatus.API_BACKOFF,
                created_at=req.created_at,
                finished_at=_now_utc(),
                audit_id=req.audit_id,
                match_id=req.match_id,
                event_id=req.event_id,
                market_type=req.market_type,
                side=req.side,
                line=req.line,
                exec_side=req.exec_side,
                is_live=bool(req.is_live),
                odd_at_decision=req.odd_at_decision,
                timing=timing,
                policy=req.policy,
                error=f"API_BACKOFF until_ts={float(self._api_backoff_until_ts):.0f}",
            )

        allowed, cap_meta = await self._cap_allow()
        if not allowed:
            return ExecutionResult(
                execution_id=req.execution_id,
                status=ExecStatus.CAP_BLOCKED,
                created_at=req.created_at,
                finished_at=_now_utc(),
                audit_id=req.audit_id,
                match_id=req.match_id,
                event_id=req.event_id,
                market_type=req.market_type,
                side=req.side,
                line=req.line,
                exec_side=req.exec_side,
                is_live=bool(req.is_live),
                odd_at_decision=req.odd_at_decision,
                timing=timing,
                policy=req.policy,
                error="CAP_BLOCKED",
                raw={"cap": cap_meta},
            )

        # build bet_type
        betslip_type = "normal"
        if req.exec_side == ExecSide.LAY:
            bet_type = ApiBetslipClient.build_lay_bet_type(req.market_type.value, req.side, req.line)
            betslip_type = "lay"
        else:
            bet_type = ApiBetslipClient.build_bet_type(req.market_type.value, req.side, req.line)

        api_result: Optional[BetslipApiResult] = None
        try:
            api_result = await self._api.get_betslip_odds(event_id=req.event_id, bet_type=bet_type, betslip_type=betslip_type)
        except Exception as e:
            return ExecutionResult(
                execution_id=req.execution_id,
                status=ExecStatus.API_FAILED,
                created_at=req.created_at,
                finished_at=_now_utc(),
                audit_id=req.audit_id,
                match_id=req.match_id,
                event_id=req.event_id,
                market_type=req.market_type,
                side=req.side,
                line=req.line,
                exec_side=req.exec_side,
                is_live=bool(req.is_live),
                odd_at_decision=req.odd_at_decision,
                timing=timing,
                policy=req.policy,
                error=str(e),
            )

        timing.post_ms = int(getattr(api_result, "request_time_ms", 0) or 0)
        timing.total_ms = int(getattr(api_result, "total_time_ms", 0) or 0)
        timing.call_to_done_ms = _ms(max(0.0, time.time() - float(req.created_at.timestamp())))

        retry_after = int(getattr(api_result, "rate_limit_retry_after_sec", 0) or 0)
        if retry_after > 0:
            self._api_backoff_until_ts = time.time() + float(retry_after) + 5.0

        odd_final, bookie, limit_final, num_bk, snap_err = _extract_snapshot(req.exec_side, api_result)
        delta_odds = None
        delta_pct = None
        if req.odd_at_decision and odd_final and req.odd_at_decision > 0:
            delta_odds = float(odd_final) - float(req.odd_at_decision)
            delta_pct = float(delta_odds) / float(req.odd_at_decision) * 100.0

        status = ExecStatus.DRY_OK if api_result and api_result.success and odd_final else ExecStatus.API_FAILED
        err = None
        if status != ExecStatus.DRY_OK:
            err = snap_err or (api_result.error if api_result else "API_FAILED")
        if retry_after > 0:
            status = ExecStatus.RATE_LIMIT
            err = f"RATE_LIMIT retry_after={retry_after}s"

        return ExecutionResult(
            execution_id=req.execution_id,
            status=status,
            created_at=req.created_at,
            finished_at=_now_utc(),
            audit_id=req.audit_id,
            match_id=req.match_id,
            event_id=req.event_id,
            market_type=req.market_type,
            side=req.side,
            line=req.line,
            exec_side=req.exec_side,
            is_live=bool(req.is_live),
            odd_at_decision=req.odd_at_decision,
            odd_final=odd_final,
            bookie_final=bookie,
            limit_final=limit_final,
            num_bk=num_bk,
            delta_odds=delta_odds,
            delta_pct=delta_pct,
            timing=timing,
            policy=req.policy,
            http_status=int(getattr(api_result, "http_status", 0) or 0),
            retry_after_sec=retry_after or None,
            error=err,
            raw={
                "betslip_id": getattr(api_result, "betslip_id", ""),
                "cap": cap_meta,
                "bet_type": bet_type,
                "betslip_type": betslip_type,
            },
        )

