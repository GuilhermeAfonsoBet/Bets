from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from loguru import logger

from scraper.betinasia import BetinAsiaScraper
from scraper.api_betslip import ApiBetslipClient, BetslipApiResult, PlaceOrderResult, ApiGetResult

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


def _is_auth_error(api_result: Optional[BetslipApiResult]) -> bool:
    try:
        if not api_result:
            return False
        hs = int(getattr(api_result, "http_status", 0) or 0)
        if hs == 401:
            return True
        err = str(getattr(api_result, "error", "") or "").lower()
        return (
            ("auth_error" in err)
            or ("authentication credentials were not provided" in err)
            or ("http_401" in err)
            or ("no_root_session_cookie" in err)
        )
    except Exception:
        return False


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
    _betslip_cache: Any = None  # OrderedDict[str, str] | None
    _betslip_cache_max_keys: int = 0
    _op_lock: asyncio.Lock = None

    async def start(self) -> None:
        self._cap_lock = asyncio.Lock()
        self._op_lock = asyncio.Lock()
        from collections import deque, OrderedDict

        self._cap_open_times = deque()
        self._scraper = BetinAsiaScraper()
        await self._scraper.start()
        page = self._scraper._page
        self._api = ApiBetslipClient(page)
        # listener deve ser registrado ANTES de navegações que possam abrir WS
        self._api.setup_listener()

        ok_login = await self._scraper.login()
        if not ok_login:
            raise RuntimeError("LOGIN_FAILED")

        try:
            self._betslip_cache_max_keys = int(os.getenv("EXECUTOR_BETSLIP_CACHE_MAX_KEYS", "0"))
        except Exception:
            self._betslip_cache_max_keys = 0
        self._betslip_cache = OrderedDict() if self._betslip_cache_max_keys > 0 else None

        # FAST mode: reduzir esperas por PMM (trade-off: menos bookies/menos "best odd")
        if os.getenv("EXECUTOR_FAST_PMM", "1").strip() in ("1", "true", "True", "yes", "YES"):
            self._api.PMM_TIMEOUT = float(os.getenv("EXECUTOR_PMM_TIMEOUT_SEC", "0.8"))
            self._api.PMM_MIN_WAIT = float(os.getenv("EXECUTOR_PMM_MIN_WAIT_SEC", "0.0"))
            self._api.PMM_IDLE_TIMEOUT = float(os.getenv("EXECUTOR_PMM_IDLE_TIMEOUT_SEC", "0.12"))
            logger.info(
                f"[executor:{self.name}] FAST_PMM on: timeout={self._api.PMM_TIMEOUT}s "
                f"min_wait={self._api.PMM_MIN_WAIT}s idle={self._api.PMM_IDLE_TIMEOUT}s"
            )

        await page.goto(self.football_url)
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(4000)
        self._running = True
        logger.info(
            f"[executor:{self.name}] started (WS warm) betslip_cache_max_keys={self._betslip_cache_max_keys}"
        )

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

        cache_key = f"{req.event_id}|{betslip_type}|{bet_type}"
        api_result: Optional[BetslipApiResult] = None
        cache_hit = False

        def _is_too_many_open(res: Optional[BetslipApiResult]) -> bool:
            try:
                if not res:
                    return False
                if int(getattr(res, "http_status", 0) or 0) != 403:
                    return False
                return "too_many_open_betslips" in str(getattr(res, "error", "") or "")
            except Exception:
                return False

        try:
            cached_id = None
            if self._betslip_cache is not None:
                cached_id = self._betslip_cache.get(cache_key)
                if cached_id:
                    cache_hit = True
                    try:
                        self._betslip_cache.move_to_end(cache_key)
                    except Exception:
                        pass
            if cached_id:
                api_result = await self._api.refresh_betslip(cached_id)
            else:
                api_result = await self._api.get_betslip_odds(event_id=req.event_id, bet_type=bet_type, betslip_type=betslip_type)
                if self._betslip_cache is not None and api_result and api_result.success and getattr(api_result, "betslip_id", ""):
                    self._betslip_cache[cache_key] = str(api_result.betslip_id)
                    try:
                        self._betslip_cache.move_to_end(cache_key)
                    except Exception:
                        pass

            # 403 too_many_open_betslips: tenta fechar UI e re-tenta 1 vez (sem usar cache).
            if _is_too_many_open(api_result):
                try:
                    if self._betslip_cache is not None:
                        self._betslip_cache.clear()
                except Exception:
                    pass
                try:
                    await self._api.close_visible_betslip_ui()
                except Exception:
                    pass
                await asyncio.sleep(float(os.getenv("EXECUTOR_TOO_MANY_OPEN_RETRY_SLEEP_SEC", "0.6")))
                api_result = await self._api.get_betslip_odds(event_id=req.event_id, bet_type=bet_type, betslip_type=betslip_type)

            # Se der 401/auth na fase de betslip, a causa mais comum é cache de betslip
            # de uma sessão anterior. Faz relogin + força criar betslip novo (sem refresh).
            if _is_auth_error(api_result):
                try:
                    if self._betslip_cache is not None:
                        self._betslip_cache.pop(cache_key, None)
                except Exception:
                    pass
                ok = await self._relogin()
                if ok:
                    api_result = await self._api.get_betslip_odds(event_id=req.event_id, bet_type=bet_type, betslip_type=betslip_type)
                    if self._betslip_cache is not None and api_result and api_result.success and getattr(api_result, "betslip_id", ""):
                        self._betslip_cache[cache_key] = str(api_result.betslip_id)
                        try:
                            self._betslip_cache.move_to_end(cache_key)
                        except Exception:
                            pass
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
            if _is_auth_error(api_result):
                status = ExecStatus.NO_SESSION
        if retry_after > 0:
            status = ExecStatus.RATE_LIMIT
            err = f"RATE_LIMIT retry_after={retry_after}s"

        # Cleanup: no shadow/dryrun, tenta reduzir "open betslips" no servidor.
        # Em LIVE, precisamos manter o betslip aberto para o place_order().
        try:
            bid = str(getattr(api_result, "betslip_id", "") or "")
            if bid and (not bool(req.is_live)):
                if self._betslip_cache is None:
                    await self._api.close_betslip(bid)
                else:
                    # eviction LRU: fecha os expulsos
                    while self._betslip_cache_max_keys > 0 and len(self._betslip_cache) > self._betslip_cache_max_keys:
                        _, old_bid = self._betslip_cache.popitem(last=False)
                        try:
                            await self._api.close_betslip(str(old_bid))
                        except Exception:
                            pass
        except Exception:
            pass

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
                "cache_hit": bool(cache_hit),
                "cap": cap_meta,
                "bet_type": bet_type,
                "betslip_type": betslip_type,
            },
        )

    async def _relogin(self) -> bool:
        """
        Revalida login e força novo login se necessário.
        """
        try:
            assert self._scraper is not None
            try:
                ok = await asyncio.wait_for(self._scraper.is_session_valid(), timeout=float(os.getenv("EXECUTOR_RELOGIN_CHECK_TIMEOUT_SEC", "8")))
                if ok:
                    return True
            except Exception:
                pass
        except Exception:
            pass
        try:
            assert self._scraper is not None
            ok = await self._scraper.login(force=True)
            if ok:
                try:
                    page = self._scraper._page
                    await page.goto(self.football_url)
                    await page.wait_for_load_state("domcontentloaded")
                    await page.wait_for_timeout(1500)
                except Exception:
                    pass
            return bool(ok)
        except Exception:
            return False

    async def get_account_snapshot(self, *, page_size: int = 50) -> Dict[str, Any]:
        """
        Snapshot operacional (best-effort):
        - tenta descobrir saldo (endpoint variável)
        - busca últimas ordens (/v1/orders/) e resume P&L/exposição quando disponível
        """
        assert self._api is not None
        assert self._scraper is not None
        async with (self._op_lock or asyncio.Lock()):
            placer = os.getenv("BETINASIA_USERNAME", "").strip()

            bal: ApiGetResult = await self._api.get_balance_any()
            if (not bal.ok) and (
                int(bal.http_status or 0) == 401
                or "NO_ROOT_SESSION_COOKIE" in str(bal.error or "")
                or "HTTP_401" in str(bal.error or "")
            ):
                ok = await self._relogin()
                if ok:
                    bal = await self._api.get_balance_any()

            orders: Optional[ApiGetResult] = None
            if placer:
                orders = await self._api.get_orders(placer=placer, page_size=int(page_size), page=1)
                if (not orders.ok) and (
                    int(orders.http_status or 0) == 401
                    or "NO_ROOT_SESSION_COOKIE" in str(orders.error or "")
                    or "HTTP_401" in str(orders.error or "")
                ):
                    ok = await self._relogin()
                    if ok:
                        orders = await self._api.get_orders(placer=placer, page_size=int(page_size), page=1)

            def _extract_orders_list(x: Any) -> list:
                if isinstance(x, list):
                    return x
                if not isinstance(x, dict):
                    return []
                for k in ("orders", "results", "data"):
                    v = x.get(k)
                    if isinstance(v, list):
                        return v
                    if isinstance(v, dict):
                        for kk in ("orders", "results", "data"):
                            vv = v.get(kk)
                            if isinstance(vv, list):
                                return vv
                return []

            pnl = {
                "orders_ok": bool(orders.ok) if orders else False,
                "orders_http": int(orders.http_status or 0) if orders else 0,
                "orders_error": (str(orders.error)[:220] if orders and orders.error else None),
                "n": 0,
                "n_open": 0,
                "n_closed": 0,
                "pnl_realized_sum": None,
                "stake_open_sum": None,
                "stake_total_sum": None,
            }
            if orders and orders.data is not None:
                raw = orders.data
                if isinstance(raw, dict) and isinstance(raw.get("data"), dict):
                    raw = raw.get("data")
                lst = _extract_orders_list(raw)
                pnl["n"] = int(len(lst))
                pnl_real = 0.0
                pnl_have = 0
                stake_open = 0.0
                stake_tot = 0.0
                n_open = 0
                n_closed = 0
                for o in lst:
                    if not isinstance(o, dict):
                        continue
                    closed = bool(o.get("closed")) if o.get("closed") is not None else (str(o.get("status") or "").lower() in ("closed", "settled"))
                    if closed:
                        n_closed += 1
                    else:
                        n_open += 1
                    ws = o.get("want_stake")
                    if isinstance(ws, list) and len(ws) >= 2:
                        try:
                            st = float(ws[1])
                            stake_tot += st
                            if not closed:
                                stake_open += st
                        except Exception:
                            pass
                    pl = o.get("profit_loss")
                    if pl is not None:
                        try:
                            pnl_real += float(pl)
                            pnl_have += 1
                        except Exception:
                            pass
                pnl["n_open"] = n_open
                pnl["n_closed"] = n_closed
                pnl["stake_open_sum"] = float(stake_open)
                pnl["stake_total_sum"] = float(stake_tot)
                pnl["pnl_realized_sum"] = (float(pnl_real) if pnl_have > 0 else None)

            return {
                "ts": _now_utc().isoformat(),
                "placer": placer or None,
                "balance_ok": bool(bal.ok),
                "balance_http": int(bal.http_status or 0),
                "balance_error": (str(bal.error)[:220] if bal.error else None),
                "balance": bal.data,
                "pnl": pnl,
            }

    async def execute(self, req: ExecutionRequest, received_ts: float) -> ExecutionResult:
        async with (self._op_lock or asyncio.Lock()):
            return await self._execute_unlocked(req, received_ts)

    async def _execute_unlocked(self, req: ExecutionRequest, received_ts: float) -> ExecutionResult:
        """
        Implementação real do dispatcher (assume lock externo).
        """
        if not bool(req.is_live):
            return await self.execute_dryrun(req, received_ts)

        if os.getenv("EXECUTOR_ALLOW_LIVE", "0").strip() not in ("1", "true", "True", "yes", "YES"):
            r = await self.execute_dryrun(req, received_ts)
            r.error = "LIVE_DISABLED (set EXECUTOR_ALLOW_LIVE=1)"
            return r

        # 1) Obter snapshot + betslip_id (reutiliza lógica de odds)
        dry = await self.execute_dryrun(req, received_ts)
        betslip_id = str((dry.raw or {}).get("betslip_id") or "")
        if dry.status != ExecStatus.DRY_OK or not betslip_id:
            dry.error = (dry.error or "") + " | LIVE_PRECHECK_FAILED"
            return dry

        # 2) Definir stake e preço a enviar
        stake = None
        try:
            if req.policy and req.policy.stake_requested is not None:
                stake = float(req.policy.stake_requested)
        except Exception:
            stake = None

        stake_ccy = str(os.getenv("EXECUTOR_LIVE_CCY", "USD"))
        max_stake = float(os.getenv("EXECUTOR_LIVE_MAX_STAKE", "5.0"))

        price = dry.odd_final or req.odd_at_decision
        if not price or float(price) <= 1.0:
            dry.error = f"BAD_PRICE price={price}"
            return dry

        if req.exec_side == ExecSide.LAY and stake is None:
            try:
                if req.policy and req.policy.liability_requested is not None:
                    liab = float(req.policy.liability_requested)
                    if liab > 0 and float(price) > 1.0:
                        stake = liab / (float(price) - 1.0)
            except Exception:
                stake = None

        if stake is None:
            stake = float(os.getenv("EXECUTOR_LIVE_STAKE", "3.0"))

        if stake > max_stake:
            dry.error = f"LIVE_STAKE_TOO_HIGH stake={stake} max={max_stake}"
            return dry

        # 3) Place order (com 1 retry via relogin se 401)
        assert self._api is not None
        t_place0 = time.time()
        place: PlaceOrderResult = await self._api.place_order(
            betslip_id=betslip_id,
            price=float(price),
            stake_ccy=stake_ccy,
            stake=float(stake),
            exchange_mode=str(os.getenv("EXECUTOR_LIVE_EXCHANGE_MODE", "make_and_take")),
        )
        post_ms = _ms(max(0.0, time.time() - t_place0))

        if (not place.success) and (
            int(place.http_status or 0) == 401
            or "HTTP_401" in str(place.error or "")
            or "NO_ROOT_SESSION_COOKIE" in str(place.error or "")
        ):
            ok = await self._relogin()
            if ok:
                t_place1 = time.time()
                place = await self._api.place_order(
                    betslip_id=betslip_id,
                    price=float(price),
                    stake_ccy=stake_ccy,
                    stake=float(stake),
                    exchange_mode=str(os.getenv("EXECUTOR_LIVE_EXCHANGE_MODE", "make_and_take")),
                )
                post_ms = _ms(max(0.0, time.time() - t_place1))

        # 4) Montar resultado LIVE
        if place.success:
            dry.status = ExecStatus.LIVE_OK
            dry.http_status = int(place.http_status or 0) or 200
            dry.timing.post_ms = post_ms
            dry.raw = dict(dry.raw or {})
            dry.raw.update(
                {
                    "live": True,
                    "order_resp": place.response,
                    "order_http": int(place.http_status or 0),
                    "order_ms": int(place.request_time_ms or 0),
                    "order_text_prefix": (place.text_prefix or "")[:300],
                    "sent": {"stake_ccy": stake_ccy, "stake": float(stake), "price": float(price)},
                }
            )
            return dry

        dry.status = ExecStatus.API_FAILED
        dry.http_status = int(place.http_status or 0) or None
        dry.error = f"LIVE_PLACE_FAILED: {place.error or 'unknown'}"
        dry.timing.post_ms = post_ms
        dry.raw = dict(dry.raw or {})
        dry.raw.update(
            {
                "live": True,
                "order_http": int(place.http_status or 0),
                "order_ms": int(place.request_time_ms or 0),
                "order_text_prefix": (place.text_prefix or "")[:300],
                "sent": {"stake_ccy": stake_ccy, "stake": float(stake), "price": float(price)},
            }
        )
        return dry

