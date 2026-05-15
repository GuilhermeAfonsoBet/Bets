from __future__ import annotations

import asyncio
import os
import time
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, List

from loguru import logger

from scraper.betinasia import BetinAsiaScraper
from scraper.api_betslip import ApiBetslipClient, BetslipApiResult, PlaceOrderResult, ApiGetResult

from .contracts import ExecutionRequest, ExecutionResult, ExecStatus, ExecSide, ExecutionTiming


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _extract_order_id(order_resp: Any) -> Optional[str]:
    """
    Extrai um identificador estável da resposta de /v1/orders/.
    Mantém robustez: a API pode retornar `id`, `order_id`, `uuid` ou aninhado.
    """
    try:
        if not order_resp:
            return None
        if isinstance(order_resp, str):
            s = order_resp.strip()
            return s or None
        if isinstance(order_resp, dict):
            for k in ("id", "order_id", "orderId", "uuid", "uid"):
                v = order_resp.get(k)
                if v is None:
                    continue
                s = str(v).strip()
                if s:
                    return s
            # tenta 1 nível de aninhamento comum
            for k in ("data", "order", "result"):
                v = order_resp.get(k)
                if isinstance(v, dict):
                    r = _extract_order_id(v)
                    if r:
                        return r
        return None
    except Exception:
        return None


def _ms(dt_s: float) -> int:
    return int(round(dt_s * 1000.0))


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if not s:
            return None
        s = s.replace(",", ".")
        return float(s)
    except Exception:
        return None


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


def _is_no_pmms_error(api_result: Optional[BetslipApiResult]) -> bool:
    try:
        if not api_result:
            return False
        err = str(getattr(api_result, "error", "") or "")
        return _err_contains(err, "No PMMs received", "No PMMs after refresh")
    except Exception:
        return False


def _err_contains(err: Any, *needles: str) -> bool:
    try:
        s = str(err or "")
    except Exception:
        return False
    s = s.lower()
    return any(n.lower() in s for n in (needles or []) if n)


def _is_playwright_context_destroyed(err: Any) -> bool:
    # erro clássico do Playwright quando a page navega/recarrega durante evaluate/fetch
    return _err_contains(err, "execution context was destroyed", "most likely because of a navigation")


def _is_playwright_target_closed(err: Any) -> bool:
    return _err_contains(err, "target closed", "browser has been closed", "page closed", "has been closed")


def _is_login_navigation_timeout(err: Any) -> bool:
    # heurística para identificar timeouts de navegação/login
    return _err_contains(err, "timeout", "navigating to", "/login", "domcontentloaded")


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
    _account_snapshot_cache: Optional[Dict[str, Any]] = None
    _account_snapshot_cache_ts: float = 0.0  # time.monotonic()
    _auto_restart_lock: asyncio.Lock = None
    _auto_restart_times: Any = None  # deque[float]
    _fail_streak: int = 0
    # housekeeping (redução de overhead/latência em run longo)
    _hk_last_light_ts: float = 0.0
    _hk_last_ui_ts: float = 0.0
    _hk_last_cache_close_ts: float = 0.0
    _hk_last_strong_ts: float = 0.0
    _hk_overhead_local_ms: Any = None  # deque[int]
    _hk_call_minus_total_ms: Any = None  # deque[int]

    async def start(self) -> None:
        self._cap_lock = asyncio.Lock()
        self._op_lock = asyncio.Lock()
        from collections import deque, OrderedDict

        self._cap_open_times = deque()
        self._auto_restart_lock = asyncio.Lock()
        self._auto_restart_times = deque()
        self._fail_streak = 0
        # housekeeping state (best-effort; tunável via env)
        try:
            hk_n = int(float(os.getenv("EXECUTOR_HOUSEKEEP_STRONG_WINDOW_N", "50") or 50))
        except Exception:
            hk_n = 50
        hk_n = int(max(10, hk_n))
        self._hk_overhead_local_ms = deque(maxlen=hk_n)
        self._hk_call_minus_total_ms = deque(maxlen=hk_n)
        self._hk_last_light_ts = 0.0
        self._hk_last_ui_ts = 0.0
        self._hk_last_cache_close_ts = 0.0
        self._hk_last_strong_ts = 0.0
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

        # `page.goto()` por padrão espera o evento "load", que pode nunca ocorrer sob proxy/recursos pesados.
        # Para robustez operacional, usamos `domcontentloaded` e timeout configurável.
        #
        # Observação: mesmo que o goto falhe, muitas rotas (API /v1 + sessão) seguem funcionando.
        # Em produção, prefira manter `EXECUTOR_GOTO_STRICT=1` para falhar cedo, se quiser.
        wait_until = os.getenv("EXECUTOR_GOTO_WAIT_UNTIL", "domcontentloaded").strip() or "domcontentloaded"
        timeout_ms = int(float(os.getenv("EXECUTOR_GOTO_TIMEOUT_MS", "45000") or 45000))
        strict = os.getenv("EXECUTOR_GOTO_STRICT", "0").strip() in ("1", "true", "True", "yes", "YES")
        try:
            await page.goto(self.football_url, wait_until=wait_until, timeout=timeout_ms)
        except Exception as e:
            # fallback: tenta domcontentloaded explicitamente
            try:
                await page.goto(self.football_url, wait_until="domcontentloaded", timeout=timeout_ms)
            except Exception as e2:
                msg = str(e2)[:220]
                logger.warning(f"[executor:{self.name}] goto football_url failed (continuing) wait_until={wait_until} timeout_ms={timeout_ms} err={msg}")
                if strict:
                    raise
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

    def _auto_restart_enabled(self) -> bool:
        try:
            return str(os.getenv("EXECUTOR_AUTO_RESTART_ENABLE", "1") or "1").strip().lower() in ("1", "true", "yes", "y", "on")
        except Exception:
            return True

    def _auto_restart_params(self) -> Dict[str, float]:
        # Defaults conservadores para evitar loops.
        try:
            cooldown_sec = float(os.getenv("EXECUTOR_AUTO_RESTART_COOLDOWN_SEC", "120") or 120.0)
        except Exception:
            cooldown_sec = 120.0
        try:
            max_per_hour = float(os.getenv("EXECUTOR_AUTO_RESTART_MAX_PER_HOUR", "6") or 6.0)
        except Exception:
            max_per_hour = 6.0
        try:
            fail_streak = float(os.getenv("EXECUTOR_AUTO_RESTART_FAIL_STREAK", "3") or 3.0)
        except Exception:
            fail_streak = 3.0
        return {
            "cooldown_sec": float(max(0.0, cooldown_sec)),
            "max_per_hour": float(max(1.0, max_per_hour)),
            "fail_streak": float(max(1.0, fail_streak)),
        }

    async def _auto_restart_allowed(self) -> Tuple[bool, str]:
        try:
            p = self._auto_restart_params()
            cooldown = float(p["cooldown_sec"])
            max_per_hour = int(p["max_per_hour"])
        except Exception:
            cooldown = 120.0
            max_per_hour = 6
        now = time.time()
        try:
            dq = self._auto_restart_times
            if dq is None:
                return True, "no_state"
            while dq and (now - float(dq[0])) > 3600.0:
                dq.popleft()
            if dq:
                age = now - float(dq[-1])
                if cooldown > 0 and age < cooldown:
                    return False, f"cooldown age={age:.0f}s < {cooldown:.0f}s"
            if len(dq) >= int(max_per_hour):
                return False, f"rate_limit {len(dq)}/{max_per_hour} restarts na última hora"
        except Exception:
            return True, "state_err"
        return True, "ok"

    async def _restart_browser_session(self, *, reason: str) -> bool:
        """
        Reinicia Playwright/browser/sessão do worker sem depender de systemd.
        Se falhar, encerra o processo para o systemd reiniciar o serviço (mais 'limpo').
        """
        if not self._auto_restart_enabled():
            return False
        async with (self._auto_restart_lock or asyncio.Lock()):
            allowed, why = await self._auto_restart_allowed()
            if not allowed:
                try:
                    logger.warning(f"[executor:{self.name}] auto-restart bloqueado ({why}) reason={reason}")
                except Exception:
                    pass
                return False
            try:
                logger.warning(f"[executor:{self.name}] auto-restart browser/session acionado reason={reason}")
            except Exception:
                pass
            try:
                if self._auto_restart_times is not None:
                    self._auto_restart_times.append(time.time())
            except Exception:
                pass
            # caches podem referenciar betslips/sessões antigas
            try:
                if self._betslip_cache is not None:
                    self._betslip_cache.clear()
            except Exception:
                pass

            # fecha scraper atual
            try:
                if self._scraper is not None:
                    await self._scraper.close()
            except Exception:
                pass
            self._scraper = None
            self._api = None

            try:
                self._scraper = BetinAsiaScraper()
                await self._scraper.start()
                page = self._scraper._page
                self._api = ApiBetslipClient(page)
                self._api.setup_listener()
                ok_login = await self._scraper.login(force=True)
                if not ok_login:
                    raise RuntimeError("LOGIN_FAILED")
                # aquecer navegação (best-effort)
                try:
                    wait_until = os.getenv("EXECUTOR_GOTO_WAIT_UNTIL", "domcontentloaded").strip() or "domcontentloaded"
                    timeout_ms = int(float(os.getenv("EXECUTOR_GOTO_TIMEOUT_MS", "45000") or 45000))
                    await page.goto(self.football_url, wait_until=wait_until, timeout=timeout_ms)
                    await page.wait_for_timeout(1200)
                except Exception:
                    pass
                self._fail_streak = 0
                return True
            except Exception as e:
                try:
                    logger.error(f"[executor:{self.name}] auto-restart falhou: {str(e)[:220]}")
                except Exception:
                    pass
                try:
                    os._exit(22)
                except Exception:
                    raise

    def _housekeep_enabled(self) -> bool:
        try:
            return str(os.getenv("EXECUTOR_HOUSEKEEP_ENABLE", "1") or "1").strip().lower() in ("1", "true", "yes", "y", "on")
        except Exception:
            return True

    def _housekeep_params(self) -> Dict[str, Any]:
        def _b(name: str, default: str) -> bool:
            try:
                return str(os.getenv(name, default) or default).strip().lower() in ("1", "true", "yes", "y", "on")
            except Exception:
                return str(default).strip().lower() in ("1", "true", "yes", "y", "on")

        def _f(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, str(default)) or default)
            except Exception:
                return float(default)

        def _i(name: str, default: int) -> int:
            try:
                return int(float(os.getenv(name, str(default)) or default))
            except Exception:
                return int(default)

        return {
            # light
            "light_every_sec": float(max(0.0, _f("EXECUTOR_HOUSEKEEP_LIGHT_EVERY_SEC", 10.0))),
            "light_gc_enable": bool(_b("EXECUTOR_HOUSEKEEP_LIGHT_GC_ENABLE", "1")),
            "light_ui_close_enable": bool(_b("EXECUTOR_HOUSEKEEP_LIGHT_UI_CLOSE_ENABLE", "0")),
            "light_ui_every_sec": float(max(1.0, _f("EXECUTOR_HOUSEKEEP_LIGHT_UI_EVERY_SEC", 60.0))),
            "light_ui_timeout_sec": float(max(0.05, _f("EXECUTOR_HOUSEKEEP_LIGHT_UI_CLOSE_TIMEOUT_SEC", 0.35))),
            "light_close_cache_enable": bool(_b("EXECUTOR_HOUSEKEEP_LIGHT_CLOSE_CACHE_ENABLE", "0")),
            "light_close_cache_every_sec": float(max(1.0, _f("EXECUTOR_HOUSEKEEP_LIGHT_CLOSE_CACHE_EVERY_SEC", 120.0))),
            "light_close_cache_max": int(max(0, _i("EXECUTOR_HOUSEKEEP_LIGHT_CLOSE_CACHE_MAX", 6))),
            "light_close_cache_timeout_sec": float(max(0.05, _f("EXECUTOR_HOUSEKEEP_LIGHT_CLOSE_CACHE_TIMEOUT_SEC", 0.9))),
            # strong
            "strong_enable": bool(_b("EXECUTOR_HOUSEKEEP_STRONG_ENABLE", "0")),
            "strong_min_events": int(max(3, _i("EXECUTOR_HOUSEKEEP_STRONG_MIN_EVENTS", 20))),
            "strong_overhead_p50_ms": float(max(0.0, _f("EXECUTOR_HOUSEKEEP_STRONG_OVERHEAD_P50_MS", 6000.0))),
            "strong_call_minus_total_p50_ms": float(max(0.0, _f("EXECUTOR_HOUSEKEEP_STRONG_CALL_MINUS_TOTAL_P50_MS", 10000.0))),
            "strong_cooldown_sec": float(max(0.0, _f("EXECUTOR_HOUSEKEEP_STRONG_COOLDOWN_SEC", 900.0))),
        }

    def _pctl(self, xs: List[float], p: float) -> Optional[float]:
        if not xs:
            return None
        xs2 = sorted(xs)
        k = (len(xs2) - 1) * (float(p) / 100.0)
        f = int(k)
        c = min(len(xs2) - 1, f + 1)
        if f == c:
            return float(xs2[f])
        return float(xs2[f] + (k - f) * (xs2[c] - xs2[f]))

    async def _housekeep_light_maybe(self, *, allow_ui: bool, allow_close_cache: bool, reason: str) -> Dict[str, Any]:
        if not self._housekeep_enabled():
            return {"skipped": True, "why": "disabled", "reason": reason}
        if self._api is None:
            return {"skipped": True, "why": "no_api", "reason": reason}

        p = self._housekeep_params()
        now = time.time()
        if (now - float(self._hk_last_light_ts or 0.0)) < float(p["light_every_sec"]):
            return {"skipped": True, "why": "interval", "reason": reason}
        self._hk_last_light_ts = now

        out: Dict[str, Any] = {"skipped": False, "reason": reason}

        # 1) GC de caches internas do ApiBetslipClient (sync; bem barato)
        if bool(p.get("light_gc_enable")):
            t0 = time.time()
            ok = True
            try:
                self._api._gc()
            except Exception:
                ok = False
            out["gc"] = {"ok": bool(ok), "ms": _ms(max(0.0, time.time() - t0))}

        # 2) (opcional) fechar betslips cacheados e limpar cache local
        if allow_close_cache and bool(p.get("light_close_cache_enable")) and self._betslip_cache is not None:
            every = float(p.get("light_close_cache_every_sec") or 0.0)
            if every <= 0 or (now - float(self._hk_last_cache_close_ts or 0.0)) >= every:
                self._hk_last_cache_close_ts = now
                bids: List[str] = []
                try:
                    bids = [str(x) for x in list(self._betslip_cache.values()) if str(x)]
                except Exception:
                    bids = []
                uniq = list(dict.fromkeys(bids))
                mx = int(p.get("light_close_cache_max") or 0)
                if mx > 0:
                    uniq = uniq[:mx]
                n_ok = 0
                n = 0
                ms_sum = 0
                ms_max = 0
                for bid in uniq:
                    n += 1
                    t1 = time.time()
                    ok = True
                    try:
                        await asyncio.wait_for(self._api.close_betslip(bid), timeout=float(p.get("light_close_cache_timeout_sec") or 0.9))
                    except Exception:
                        ok = False
                    dt = _ms(max(0.0, time.time() - t1))
                    ms_sum += int(dt)
                    ms_max = max(int(ms_max), int(dt))
                    if ok:
                        n_ok += 1
                try:
                    self._betslip_cache.clear()
                except Exception:
                    pass
                out["close_cache"] = {"n": int(n), "n_ok": int(n_ok), "ms_sum": int(ms_sum), "ms_max": int(ms_max)}

        # 3) (opcional) fechar o betslip visível no UI (Playwright evaluate)
        if allow_ui and bool(p.get("light_ui_close_enable")):
            ui_every = float(p.get("light_ui_every_sec") or 60.0)
            if (now - float(self._hk_last_ui_ts or 0.0)) >= ui_every:
                self._hk_last_ui_ts = now
                t2 = time.time()
                ok = False
                try:
                    ok = bool(await asyncio.wait_for(self._api.close_visible_betslip_ui(), timeout=float(p.get("light_ui_timeout_sec") or 0.35)))
                except Exception:
                    ok = False
                out["close_ui"] = {"ok": bool(ok), "ms": _ms(max(0.0, time.time() - t2))}

        return out

    async def _housekeep_post_request(
        self,
        *,
        res: ExecutionResult,
        allow_ui: bool,
        allow_close_cache: bool,
        allow_strong: bool,
        overhead_local_ms: Optional[int],
        call_minus_total_ms: Optional[int],
        reason: str,
    ) -> None:
        """
        Housekeeping pós-request:
        - leve: GC / fechar UI / fechar cache (intervalado)
        - forte: restart de sessão quando overhead degrada (p50 em janela local)
        """
        try:
            res.raw = dict(res.raw or {})
        except Exception:
            return

        hk: Dict[str, Any] = {}
        try:
            hk["light"] = await self._housekeep_light_maybe(allow_ui=bool(allow_ui), allow_close_cache=bool(allow_close_cache), reason=str(reason))
        except Exception:
            hk["light"] = {"skipped": True, "why": "error", "reason": str(reason)}

        if allow_strong and self._housekeep_enabled():
            p = self._housekeep_params()
            if bool(p.get("strong_enable")):
                now = time.time()
                if (now - float(self._hk_last_strong_ts or 0.0)) >= float(p.get("strong_cooldown_sec") or 0.0):
                    try:
                        if overhead_local_ms is not None:
                            self._hk_overhead_local_ms.append(int(overhead_local_ms))
                        if call_minus_total_ms is not None:
                            self._hk_call_minus_total_ms.append(int(call_minus_total_ms))
                    except Exception:
                        pass
                    xs_ov = [float(x) for x in list(self._hk_overhead_local_ms or []) if x is not None and float(x) >= 0]
                    xs_cm = [float(x) for x in list(self._hk_call_minus_total_ms or []) if x is not None and float(x) >= 0]
                    ov_p50 = self._pctl(xs_ov, 50) if xs_ov else None
                    cm_p50 = self._pctl(xs_cm, 50) if xs_cm else None
                    hk["strong_window"] = {"n_overhead": int(len(xs_ov)), "overhead_p50_ms": ov_p50, "n_call_minus_total": int(len(xs_cm)), "call_minus_total_p50_ms": cm_p50}
                    min_n = int(p.get("strong_min_events") or 20)
                    over_thr = (ov_p50 is not None) and (float(p.get("strong_overhead_p50_ms") or 0.0) > 0) and (float(ov_p50) >= float(p["strong_overhead_p50_ms"]))
                    call_thr = (cm_p50 is not None) and (float(p.get("strong_call_minus_total_p50_ms") or 0.0) > 0) and (float(cm_p50) >= float(p["strong_call_minus_total_p50_ms"]))
                    should = (int(len(xs_ov)) >= min_n and over_thr) or (int(len(xs_cm)) >= min_n and call_thr)
                    if should:
                        self._hk_last_strong_ts = now
                        ok = await self._restart_browser_session(reason=f"HK_STRONG:{str(reason)[:160]}")
                        hk["strong"] = {"triggered": True, "restart_ok": bool(ok), "reason": str(reason)}
                        try:
                            # evita retrigger imediato em janela “ruim”
                            if self._hk_overhead_local_ms is not None:
                                self._hk_overhead_local_ms.clear()
                            if self._hk_call_minus_total_ms is not None:
                                self._hk_call_minus_total_ms.clear()
                        except Exception:
                            pass
                    else:
                        hk["strong"] = {"triggered": False}
                else:
                    hk["strong"] = {"triggered": False, "why": "cooldown"}
            else:
                hk["strong"] = {"triggered": False, "why": "disabled"}
        else:
            hk["strong"] = {"triggered": False, "why": ("skipped_live_critical" if not allow_strong else "disabled")}

        try:
            res.raw["housekeeping"] = hk
        except Exception:
            pass

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

        allow_live = os.getenv("EXECUTOR_ALLOW_LIVE", "0").strip() in ("1", "true", "True", "yes", "YES")
        will_place_live = bool(allow_live) and bool(req.is_live)

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
                # best-effort: tenta fechar betslips cacheados antes de limpar o cache
                try:
                    if self._betslip_cache is not None and self._api is not None:
                        bids = []
                        try:
                            bids = [str(x) for x in list(self._betslip_cache.values()) if str(x)]
                        except Exception:
                            bids = []
                        # limita para não travar no cleanup
                        for bid0 in list(dict.fromkeys(bids))[: int(os.getenv("EXECUTOR_TOO_MANY_OPEN_CLEANUP_MAX", "12"))]:
                            try:
                                await asyncio.wait_for(self._api.close_betslip(bid0), timeout=float(os.getenv("EXECUTOR_TOO_MANY_OPEN_CLOSE_TIMEOUT_SEC", "1.2")))
                            except Exception:
                                pass
                except Exception:
                    pass
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

            # Mitigação operacional: em "No PMMs", re-tenta 1x com timeout mais folgado.
            if _is_no_pmms_error(api_result):
                retry_enabled = str(os.getenv("EXECUTOR_NO_PMMS_RETRY_ENABLE", "1") or "1").strip().lower() in ("1", "true", "yes", "y", "on")
                if retry_enabled and self._api is not None:
                    old_timeout = float(getattr(self._api, "PMM_TIMEOUT", 0.8) or 0.8)
                    old_min_wait = float(getattr(self._api, "PMM_MIN_WAIT", 0.0) or 0.0)
                    old_idle = float(getattr(self._api, "PMM_IDLE_TIMEOUT", 0.12) or 0.12)
                    retry_mult = max(1.0, float(_safe_float(os.getenv("EXECUTOR_NO_PMMS_RETRY_TIMEOUT_MULT", "2.0")) or 2.0))
                    retry_floor = max(0.5, float(_safe_float(os.getenv("EXECUTOR_NO_PMMS_RETRY_TIMEOUT_FLOOR_SEC", "1.6")) or 1.6))
                    retry_timeout = max(old_timeout * retry_mult, retry_floor)
                    retry_min_wait = max(old_min_wait, max(0.0, float(_safe_float(os.getenv("EXECUTOR_NO_PMMS_RETRY_MIN_WAIT_SEC", "0.15")) or 0.15)))
                    retry_idle = max(old_idle, max(0.05, float(_safe_float(os.getenv("EXECUTOR_NO_PMMS_RETRY_IDLE_TIMEOUT_SEC", "0.25")) or 0.25)))
                    retry_sleep = max(0.0, float(_safe_float(os.getenv("EXECUTOR_NO_PMMS_RETRY_SLEEP_SEC", "0.2")) or 0.2))
                    try:
                        await self._api.close_visible_betslip_ui()
                    except Exception:
                        pass
                    if retry_sleep > 0:
                        await asyncio.sleep(retry_sleep)
                    try:
                        self._api.PMM_TIMEOUT = float(retry_timeout)
                        self._api.PMM_MIN_WAIT = float(retry_min_wait)
                        self._api.PMM_IDLE_TIMEOUT = float(retry_idle)
                        logger.warning(
                            f"[executor:{self.name}] NO_PMMS retry 1x with relaxed PMM "
                            f"timeout={self._api.PMM_TIMEOUT:.2f}s min_wait={self._api.PMM_MIN_WAIT:.2f}s idle={self._api.PMM_IDLE_TIMEOUT:.2f}s"
                        )
                        api_result = await self._api.get_betslip_odds(event_id=req.event_id, bet_type=bet_type, betslip_type=betslip_type)
                    finally:
                        try:
                            self._api.PMM_TIMEOUT = float(old_timeout)
                            self._api.PMM_MIN_WAIT = float(old_min_wait)
                            self._api.PMM_IDLE_TIMEOUT = float(old_idle)
                        except Exception:
                            pass
            # Se PMM continuar falhando e WS parecer "stale", força reciclagem da sessão do worker.
            if _is_no_pmms_error(api_result):
                ws_stale_ms = int(float(_safe_float(os.getenv("EXECUTOR_NO_PMMS_WS_STALE_MS", "12000")) or 12000.0))
                force_restart = str(os.getenv("EXECUTOR_NO_PMMS_FORCE_RESTART", "1") or "1").strip().lower() in ("1", "true", "yes", "y", "on")
                ws_age_ms = int(getattr(api_result, "ws_age_ms", 0) or 0)
                ws_msg_count = int(getattr(api_result, "ws_msg_count", 0) or 0)
                ws_stale = (ws_msg_count <= 0) or (ws_age_ms > 0 and ws_age_ms >= max(1000, ws_stale_ms))
                if force_restart and ws_stale:
                    await self._restart_browser_session(reason=f"NO_PMMS ws_msg_count={ws_msg_count} ws_age_ms={ws_age_ms}")
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
        try:
            pmm_wait_s = getattr(api_result, "pmm_wait_s", None)
            if pmm_wait_s is not None:
                # pmm_wait_s está em segundos; timing.* é em ms
                timing.pmm_wait_ms = _ms(max(0.0, float(pmm_wait_s)))
        except Exception:
            pass

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
        if status in (ExecStatus.API_FAILED, ExecStatus.NO_SESSION):
            self._fail_streak = int(self._fail_streak or 0) + 1
            try:
                streak_thr = int(self._auto_restart_params().get("fail_streak") or 3)
            except Exception:
                streak_thr = 3
            fatal = _is_playwright_context_destroyed(err) or _is_playwright_target_closed(err) or _is_login_navigation_timeout(err)
            if fatal or int(self._fail_streak) >= int(streak_thr):
                # Estamos sob _op_lock, então é seguro reiniciar aqui.
                await self._restart_browser_session(reason=f"{status.value}:{str(err)[:160]}")
        if retry_after > 0:
            status = ExecStatus.RATE_LIMIT
            err = f"RATE_LIMIT retry_after={retry_after}s"
        if status == ExecStatus.DRY_OK:
            self._fail_streak = 0

        # Telemetria de slippage (sem regra): para análise estatística posterior.
        slip_tel = {
            "delta_odds": float(delta_odds) if delta_odds is not None else None,
            "delta_pct": float(delta_pct) if delta_pct is not None else None,
        }

        # Métricas "pré-aposta" (aprox.) para análise (Back Pre/In):
        # - pre_submit_ms: tempo desde req.created_at até logo após capturar odd_final (antes de cleanup)
        # - slippage_pre_pct: (odd_pre_submit - odd_at_decision)/odd_at_decision
        value_sizing = None
        try:
            t_pre = time.time()
            pre_ms = _ms(max(0.0, t_pre - float(req.created_at.timestamp())))
            odd_pre = float(odd_final) if odd_final is not None else None
            slip_pre = None
            if req.odd_at_decision is not None and float(req.odd_at_decision) > 0 and odd_pre is not None:
                slip_pre = (float(odd_pre) - float(req.odd_at_decision)) / float(req.odd_at_decision) * 100.0
            value_sizing = {
                "enabled": False,
                "pre_submit_ms": (int(pre_ms) if pre_ms is not None else None),
                "odd_pre_submit": (float(odd_pre) if odd_pre is not None else None),
                "odd_at_decision": (float(req.odd_at_decision) if req.odd_at_decision is not None else None),
                "slippage_pre_pct": (float(slip_pre) if slip_pre is not None else None),
                "source": "dryrun_pre_bet",
            }
        except Exception:
            value_sizing = None

        # Cleanup: no shadow/dryrun, tenta reduzir "open betslips" no servidor.
        # Em LIVE (quando for realmente colocar ordem), precisamos manter o betslip aberto até o place_order().
        close_betslip_cleanup = None
        try:
            bid = str(getattr(api_result, "betslip_id", "") or "")
            if bid and (not bool(will_place_live)):
                # se cache está desligado: fecha sempre
                if self._betslip_cache is None:
                    t_cl = time.time()
                    ok_cl = True
                    try:
                        await self._api.close_betslip(bid)
                    except Exception:
                        ok_cl = False
                    close_betslip_cleanup = {"ok": bool(ok_cl), "ms": _ms(max(0.0, time.time() - t_cl))}
                else:
                    # eviction LRU: fecha os expulsos (vale para Pre e In)
                    while self._betslip_cache_max_keys > 0 and len(self._betslip_cache) > self._betslip_cache_max_keys:
                        _, old_bid = self._betslip_cache.popitem(last=False)
                        try:
                            await self._api.close_betslip(str(old_bid))
                        except Exception:
                            pass
        except Exception:
            pass

        res = ExecutionResult(
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
                "slippage_telemetry": slip_tel,
                "value_sizing": value_sizing,
            },
        )

        # Instrumentação: tempo no worker e overhead local vs ApiBetslipClient.
        try:
            res.raw = dict(res.raw or {})
            bd = dict(res.raw.get("timing_breakdown") or {})
            worker_ms = _ms(max(0.0, time.time() - float(t0)))
            tot_ms = int(res.timing.total_ms) if res.timing and res.timing.total_ms is not None else None
            overhead_local_ms = int(max(0, worker_ms - int(tot_ms))) if tot_ms is not None else None
            bd.update(
                {
                    "worker_ms": int(worker_ms),
                    "overhead_local_ms": overhead_local_ms,
                    "queue_delay_ms": (int(res.timing.queue_delay_ms) if res.timing and res.timing.queue_delay_ms is not None else None),
                    "total_api_ms": tot_ms,
                }
            )
            res.raw["timing_breakdown"] = bd
            if close_betslip_cleanup is not None:
                cl = dict((res.raw.get("cleanup") or {}))
                cl["close_betslip"] = close_betslip_cleanup
                res.raw["cleanup"] = cl
        except Exception:
            pass

        # Housekeeping pós-request:
        # - se for pré-checagem do LIVE, não pode fechar UI/cache nem disparar restart.
        try:
            tot_ms2 = int(res.timing.total_ms) if res.timing and res.timing.total_ms is not None else None
            worker_ms2 = _ms(max(0.0, time.time() - float(t0)))
            overhead_local_ms2 = int(max(0, worker_ms2 - int(tot_ms2))) if tot_ms2 is not None else None
            call_minus_total_ms2 = None
            try:
                # call_minus_total é “global” (created_at), mas é útil como proxy de overhead fora do ApiBetslipClient
                # quando temos total_ms disponível.
                ctd = _ms(max(0.0, time.time() - float(req.created_at.timestamp())))
                if tot_ms2 is not None:
                    call_minus_total_ms2 = int(max(0, int(ctd) - int(tot_ms2)))
            except Exception:
                call_minus_total_ms2 = None

            await self._housekeep_post_request(
                res=res,
                allow_ui=(not bool(will_place_live)),
                allow_close_cache=(not bool(will_place_live)),
                allow_strong=(not bool(will_place_live)),
                overhead_local_ms=overhead_local_ms2,
                call_minus_total_ms=call_minus_total_ms2,
                reason=("dryrun_precheck_live" if bool(will_place_live) else "dryrun"),
            )
        except Exception:
            pass

        # call_to_done no final (inclui cleanup/housekeeping)
        try:
            res.timing.call_to_done_ms = _ms(max(0.0, time.time() - float(req.created_at.timestamp())))
        except Exception:
            pass
        res.finished_at = _now_utc()
        return res

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
                    wait_until = os.getenv("EXECUTOR_GOTO_WAIT_UNTIL", "domcontentloaded").strip() or "domcontentloaded"
                    timeout_ms = int(float(os.getenv("EXECUTOR_GOTO_TIMEOUT_MS", "45000") or 45000))
                    await page.goto(self.football_url, wait_until=wait_until, timeout=timeout_ms)
                    await page.wait_for_timeout(1500)
                except Exception:
                    pass
            return bool(ok)
        except Exception:
            return False

    def _cache_account_snapshot(self, snap: Dict[str, Any]) -> None:
        try:
            if isinstance(snap, dict):
                self._account_snapshot_cache = copy.deepcopy(snap)
                self._account_snapshot_cache_ts = float(time.monotonic())
        except Exception:
            pass

    def _get_cached_account_snapshot(self) -> Optional[Dict[str, Any]]:
        try:
            if isinstance(self._account_snapshot_cache, dict):
                return copy.deepcopy(self._account_snapshot_cache)
        except Exception:
            return None
        return None

    async def get_account_snapshot_best_effort(
        self,
        *,
        page_size: int = 50,
        lock_timeout_sec: float = 0.25,
        total_timeout_sec: float = 3.0,
        cache_max_age_sec: float = 15.0,
    ) -> Dict[str, Any]:
        """
        Snapshot para uso operacional (bridge/monitoring).

        Regras:
        - Nunca deve bloquear execução por muito tempo (usa lock timeout).
        - Tenta retornar cache recente quando o worker está ocupado.
        - Aplica timeout total para evitar pendurar em chamadas externas.
        """
        now_m = float(time.monotonic())
        cached = self._get_cached_account_snapshot()
        cache_age = (now_m - float(self._account_snapshot_cache_ts or 0.0)) if cached else None

        # 1) cache fresco
        if cached is not None and cache_age is not None and cache_age <= float(cache_max_age_sec):
            cached["_account_meta"] = {
                "worker": self.name,
                "source": "cache_fresh",
                "cache_age_ms": int(round(float(cache_age) * 1000.0)),
            }
            return cached

        lock = self._op_lock or asyncio.Lock()
        acquired = False
        try:
            try:
                await asyncio.wait_for(lock.acquire(), timeout=float(lock_timeout_sec))
                acquired = True
            except asyncio.TimeoutError:
                # cache (mesmo stale) é melhor que timeout/hang
                if cached is not None and cache_age is not None:
                    cached["_account_meta"] = {
                        "worker": self.name,
                        "source": "cache_stale_lock_timeout",
                        "cache_age_ms": int(round(float(cache_age) * 1000.0)),
                        "lock_timeout_sec": float(lock_timeout_sec),
                    }
                    return cached
                return {
                    "ts": _now_utc().isoformat(),
                    "placer": os.getenv("BETINASIA_USERNAME", "").strip() or None,
                    "balance_ok": False,
                    "balance_http": 0,
                    "balance_error": "ACCOUNT_LOCK_TIMEOUT (no cache)",
                    "balance": None,
                    "pnl": {
                        "orders_ok": False,
                        "orders_http": 0,
                        "orders_error": "ACCOUNT_LOCK_TIMEOUT (no cache)",
                        "n": 0,
                        "n_open": 0,
                        "n_closed": 0,
                        "pnl_realized_sum": None,
                        "stake_open_sum": None,
                        "stake_total_sum": None,
                        "orders": None,
                    },
                    "_account_meta": {
                        "worker": self.name,
                        "source": "fallback_lock_timeout",
                        "lock_timeout_sec": float(lock_timeout_sec),
                    },
                }

            # 2) refresh live (com timeout total)
            try:
                snap = await asyncio.wait_for(self._get_account_snapshot_unlocked(page_size=int(page_size)), timeout=float(total_timeout_sec))
                if isinstance(snap, dict):
                    snap["_account_meta"] = {
                        "worker": self.name,
                        "source": "live",
                        "total_timeout_sec": float(total_timeout_sec),
                    }
                    self._cache_account_snapshot(snap)
                return snap
            except asyncio.TimeoutError:
                if cached is not None and cache_age is not None:
                    cached["_account_meta"] = {
                        "worker": self.name,
                        "source": "cache_stale_total_timeout",
                        "cache_age_ms": int(round(float(cache_age) * 1000.0)),
                        "total_timeout_sec": float(total_timeout_sec),
                    }
                    return cached
                return {
                    "ts": _now_utc().isoformat(),
                    "placer": os.getenv("BETINASIA_USERNAME", "").strip() or None,
                    "balance_ok": False,
                    "balance_http": 0,
                    "balance_error": f"ACCOUNT_TIMEOUT total_timeout_sec={float(total_timeout_sec):.2f}",
                    "balance": None,
                    "pnl": {
                        "orders_ok": False,
                        "orders_http": 0,
                        "orders_error": f"ACCOUNT_TIMEOUT total_timeout_sec={float(total_timeout_sec):.2f}",
                        "n": 0,
                        "n_open": 0,
                        "n_closed": 0,
                        "pnl_realized_sum": None,
                        "stake_open_sum": None,
                        "stake_total_sum": None,
                        "orders": None,
                    },
                    "_account_meta": {
                        "worker": self.name,
                        "source": "fallback_total_timeout",
                        "total_timeout_sec": float(total_timeout_sec),
                    },
                }
            except Exception as e:
                msg = str(e)[:220]
                if cached is not None and cache_age is not None:
                    cached["_account_meta"] = {
                        "worker": self.name,
                        "source": "cache_stale_error",
                        "cache_age_ms": int(round(float(cache_age) * 1000.0)),
                        "error": msg,
                    }
                    return cached
                return {
                    "ts": _now_utc().isoformat(),
                    "placer": os.getenv("BETINASIA_USERNAME", "").strip() or None,
                    "balance_ok": False,
                    "balance_http": 0,
                    "balance_error": f"ACCOUNT_ERROR {msg}",
                    "balance": None,
                    "pnl": {
                        "orders_ok": False,
                        "orders_http": 0,
                        "orders_error": f"ACCOUNT_ERROR {msg}",
                        "n": 0,
                        "n_open": 0,
                        "n_closed": 0,
                        "pnl_realized_sum": None,
                        "stake_open_sum": None,
                        "stake_total_sum": None,
                        "orders": None,
                    },
                    "_account_meta": {
                        "worker": self.name,
                        "source": "fallback_error",
                        "error": msg,
                    },
                }
        finally:
            if acquired:
                try:
                    lock.release()
                except Exception:
                    pass

    async def get_account_snapshot(self, *, page_size: int = 50) -> Dict[str, Any]:
        """
        Snapshot operacional (best-effort):
        - tenta descobrir saldo (endpoint variável)
        - busca últimas ordens (/v1/orders/) e resume P&L/exposição quando disponível
        """
        assert self._api is not None
        assert self._scraper is not None
        async with (self._op_lock or asyncio.Lock()):
            snap = await self._get_account_snapshot_unlocked(page_size=int(page_size))
            if isinstance(snap, dict):
                self._cache_account_snapshot(snap)
            return snap

    async def _get_account_snapshot_unlocked(self, *, page_size: int = 50) -> Dict[str, Any]:
        """
        Implementação do snapshot, assumindo que o lock operacional já foi obtido.
        Mantém o comportamento antigo do get_account_snapshot().
        """
        assert self._api is not None
        assert self._scraper is not None

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
            "orders": None,
        }
        if orders and orders.data is not None:
            raw = orders.data
            if isinstance(raw, dict) and isinstance(raw.get("data"), dict):
                raw = raw.get("data")
            lst = _extract_orders_list(raw)
            pnl["n"] = int(len(lst))
            # expõe lista (até page_size) para agregações no relatório (P&L por tipo)
            try:
                pnl["orders"] = lst[: int(page_size)]
            except Exception:
                pnl["orders"] = lst
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

        t_live0 = time.time()

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

        # ------------------------------------------------------------
        # Stake sizing (BACK) — operacionalização por slippage pré-submit:
        # - slippage_pre_pct < limite_negativo           => stake_neg (default 40)
        # - limite_negativo <= slippage_pre_pct <= limite_positivo => stake_mid (default 20)
        # - slippage_pre_pct > limite_positivo           => stake_pos (default 20, configurável)
        # Obs: medimos pre_submit_ms imediatamente antes do place_order().
        # ------------------------------------------------------------
        market_is_live = False
        try:
            mkt = req.meta.get("market") if isinstance(req.meta, dict) else None
            if isinstance(mkt, dict) and mkt.get("is_live") is not None:
                market_is_live = bool(mkt.get("is_live"))
        except Exception:
            market_is_live = False
        market_regime = "in" if bool(market_is_live) else "pre"

        # toggles / params (preferimos nomes do .env.example; mantemos compat com legados)
        def _env_bool(name: str, default: str = "0") -> bool:
            try:
                return str(os.getenv(name, default) or default).strip().lower() in ("1", "true", "yes", "y", "on")
            except Exception:
                return False

        def _env_float(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, str(default)) or default)
            except Exception:
                return float(default)

        # Sizing BACK (compatível com o toggle atual)
        # envs:
        # - EXECUTOR_BACKPRE_FAST_STAKE_ENABLE
        # - EXECUTOR_BACK_STAKE_SLIP_NEG_LIMIT_PCT / POS_LIMIT_PCT
        # - EXECUTOR_BACK_STAKE_SLIP_NEG / MID / POS
        # Gate opcional de latência (bloqueio de execução):
        # - EXECUTOR_BACK_LATENCY_GATE_ENABLE
        # - EXECUTOR_BACK_LATENCY_GATE_MAX_SEC
        sizing_enabled = _env_bool("EXECUTOR_BACKPRE_FAST_STAKE_ENABLE", "0") or _env_bool("EXECUTOR_BACK_STAKE_SIZING_ENABLE", "0")
        legacy_hi = _env_float("EXECUTOR_BACKPRE_FAST_STAKE_HI", _env_float("EXECUTOR_BACKPRE_FAST_STAKE", 20.0))
        legacy_max_slip = _env_float(
            "EXECUTOR_BACKPRE_FAST_MAX_SLIPPAGE_PCT",
            _env_float("EXECUTOR_BACKPRE_FAST_MAX_SLIP_PCT", 2.0),
        )
        slip_neg_limit_pct = _env_float("EXECUTOR_BACK_STAKE_SLIP_NEG_LIMIT_PCT", -2.0)
        slip_pos_limit_pct = _env_float("EXECUTOR_BACK_STAKE_SLIP_POS_LIMIT_PCT", legacy_max_slip)
        if slip_neg_limit_pct > slip_pos_limit_pct:
            slip_neg_limit_pct, slip_pos_limit_pct = slip_pos_limit_pct, slip_neg_limit_pct
        stake_back_neg = _env_float("EXECUTOR_BACK_STAKE_SLIP_NEG", 40.0)
        stake_back_mid = _env_float("EXECUTOR_BACK_STAKE_SLIP_MID", legacy_hi)
        stake_back_pos = _env_float("EXECUTOR_BACK_STAKE_SLIP_POS", stake_back_mid)
        latency_gate_enabled = _env_bool("EXECUTOR_BACK_LATENCY_GATE_ENABLE", "0")
        latency_gate_max_sec = max(0.0, _env_float("EXECUTOR_BACK_LATENCY_GATE_MAX_SEC", 0.0))

        # persistir métricas no JSONL para auditoria/analytics (vamos preencher os tempos
        # imediatamente antes do place_order, mas já escrevemos o "regime" aqui).
        try:
            dry.raw = dict(dry.raw or {})
            dry.raw["value_sizing"] = {
                "enabled": bool(sizing_enabled) and (req.exec_side == ExecSide.BACK),
                "market_regime": str(market_regime),
                "market_is_live": bool(market_is_live),
                "pre_submit_ms": None,
                "odd_pre_submit": (float(price) if price is not None else None),
                "odd_at_decision": (float(req.odd_at_decision) if req.odd_at_decision is not None else None),
                "slippage_pre_pct": None,
                "source": "live_stub",
            }
        except Exception:
            pass

        if req.exec_side == ExecSide.LAY and stake is None:
            try:
                if req.policy and req.policy.liability_requested is not None:
                    liab = float(req.policy.liability_requested)
                    if liab > 0 and float(price) > 1.0:
                        stake = liab / (float(price) - 1.0)
            except Exception:
                stake = None

        if stake is None:
            # Default live stake (global). Se sizing estiver habilitado, vamos sobrescrever abaixo
            # (antes de aplicar max_stake).
            stake = float(os.getenv("EXECUTOR_LIVE_STAKE", "3.0"))

        # Gate de slippage (opcional): bloqueia LIVE se odds piorarem além do limiar.
        # Caso de uso: in-match, bloquear se odds piorarem além do limiar (Lay: odds sobem; Back: odds caem).
        try:
            gate = req.meta.get("slippage_gate") if isinstance(req.meta, dict) else None
            thr_lay = None
            thr_back = None
            if isinstance(gate, dict):
                thr_lay = _safe_float(gate.get("lay_in_max_delta_pct"))
                # Para Back, o movimento adverso é odds CAINDO: bloqueia se delta_pct < -thr_back
                thr_back = _safe_float(gate.get("back_in_max_adverse_delta_pct") or gate.get("back_in_max_delta_pct"))

            if thr_lay is None:
                thr_lay = _safe_float(os.getenv("EXECUTOR_SLIPPAGE_GATE_LAY_IN_MAX_PCT", "0"))
            if thr_back is None:
                thr_back = _safe_float(os.getenv("EXECUTOR_SLIPPAGE_GATE_BACK_IN_MAX_PCT", "0"))

            fail_closed_lay = str(os.getenv("EXECUTOR_SLIPPAGE_GATE_LAY_IN_FAIL_CLOSED", "0") or "0").strip().lower() in ("1", "true", "yes", "y", "on")
            fail_closed_back = str(os.getenv("EXECUTOR_SLIPPAGE_GATE_BACK_IN_FAIL_CLOSED", "0") or "0").strip().lower() in ("1", "true", "yes", "y", "on")

            blocked = False
            block_label = None
            if bool(req.is_live) and req.exec_side == ExecSide.LAY and thr_lay is not None and float(thr_lay) > 0:
                if (dry.delta_pct is not None and float(dry.delta_pct) > float(thr_lay)) or (fail_closed_lay and dry.delta_pct is None):
                    blocked = True
                    block_label = "LAY_IN"
            if bool(req.is_live) and req.exec_side == ExecSide.BACK and thr_back is not None and float(thr_back) > 0:
                if (dry.delta_pct is not None and float(dry.delta_pct) < -float(thr_back)) or (fail_closed_back and dry.delta_pct is None):
                    blocked = True
                    block_label = "BACK_IN"

            if blocked:
                # best-effort: fecha betslip antes de sair (evita too_many_open_betslips)
                try:
                    await asyncio.wait_for(self._api.close_betslip(betslip_id), timeout=float(os.getenv("EXECUTOR_LIVE_CLOSE_TIMEOUT_SEC", "1.2")))
                except Exception:
                    pass
                dry.status = ExecStatus.CAP_BLOCKED
                dry.http_status = 200
                dry.raw = dict(dry.raw or {})

                if block_label == "BACK_IN":
                    if dry.delta_pct is None:
                        dry.error = f"SLIPPAGE_GATE_BACK_IN_MISSING_DELTA_PCT thr={float(thr_back):.2f}"
                    else:
                        dry.error = f"SLIPPAGE_GATE_BACK_IN delta_pct={float(dry.delta_pct):.2f} thr={float(thr_back):.2f}"
                    dry.raw["slippage_gate"] = {
                        "enabled": True,
                        "back_in_max_adverse_delta_pct": float(thr_back),
                        "delta_pct": (float(dry.delta_pct) if dry.delta_pct is not None else None),
                        "fail_closed": bool(fail_closed_back),
                    }
                else:
                    if dry.delta_pct is None:
                        dry.error = f"SLIPPAGE_GATE_LAY_IN_MISSING_DELTA_PCT thr={float(thr_lay):.2f}"
                    else:
                        dry.error = f"SLIPPAGE_GATE_LAY_IN delta_pct={float(dry.delta_pct):.2f} thr={float(thr_lay):.2f}"
                    dry.raw["slippage_gate"] = {
                        "enabled": True,
                        "lay_in_max_delta_pct": float(thr_lay),
                        "delta_pct": (float(dry.delta_pct) if dry.delta_pct is not None else None),
                        "fail_closed": bool(fail_closed_lay),
                    }
                return dry
        except Exception:
            pass

        # Atualiza métrica imediatamente antes da aposta (call->pre_submit) e aplica sizing para BACK.
        try:
            t_pre = time.time()
            pre_ms = _ms(max(0.0, t_pre - float(req.created_at.timestamp())))
            slip_pre = None
            if req.odd_at_decision is not None and float(req.odd_at_decision) > 0 and price is not None:
                slip_pre = (float(price) - float(req.odd_at_decision)) / float(req.odd_at_decision) * 100.0
            dry.raw = dict(dry.raw or {})
            vs = dict(dry.raw.get("value_sizing") or {})
            vs.update(
                {
                    "pre_submit_ms": (int(pre_ms) if pre_ms is not None else None),
                    "odd_pre_submit": (float(price) if price is not None else None),
                    "slippage_pre_pct": (float(slip_pre) if slip_pre is not None else None),
                    "source": "live_pre_place",
                }
            )

            if sizing_enabled and req.exec_side == ExecSide.BACK:
                slip_bucket = "na"
                if slip_pre is not None:
                    if float(slip_pre) < float(slip_neg_limit_pct):
                        slip_bucket = "lt_neg"
                        stake = float(stake_back_neg)
                    elif float(slip_pre) <= float(slip_pos_limit_pct):
                        slip_bucket = "mid"
                        stake = float(stake_back_mid)
                    else:
                        slip_bucket = "gt_pos"
                        stake = float(stake_back_pos)
                else:
                    # Falha de telemetria de slippage: não aumenta stake agressivamente.
                    slip_bucket = "na"
                    stake = float(stake_back_mid)
                pre_submit_ok = bool(
                    (not latency_gate_enabled)
                    or (latency_gate_max_sec <= 0.0)
                    or ((pre_ms is not None) and (float(pre_ms) <= float(latency_gate_max_sec) * 1000.0))
                )
                vs.update(
                    {
                        "enabled": True,
                        "eligible": bool(pre_submit_ok),
                        "rule": "stake_by_slippage: (<neg_limit=>stake_neg, [neg_limit,pos_limit]=>stake_mid, >pos_limit=>stake_pos)",
                        "params": {
                            "slip_neg_limit_pct": float(slip_neg_limit_pct),
                            "slip_pos_limit_pct": float(slip_pos_limit_pct),
                            "stake_back_neg": float(stake_back_neg),
                            "stake_back_mid": float(stake_back_mid),
                            "stake_back_pos": float(stake_back_pos),
                            "latency_gate_enabled": bool(latency_gate_enabled),
                            "latency_gate_max_sec": float(latency_gate_max_sec),
                        },
                        "eligible_latency": bool(pre_submit_ok),
                        "slip_bucket": str(slip_bucket),
                        "stake_chosen": float(stake),
                    }
                )

                # Gate opcional: em live BACK, bloqueia se pre_submit_ms exceder X segundos.
                if not pre_submit_ok:
                    try:
                        await asyncio.wait_for(self._api.close_betslip(betslip_id), timeout=float(os.getenv("EXECUTOR_LIVE_CLOSE_TIMEOUT_SEC", "1.2")))
                    except Exception:
                        pass
                    dry.status = ExecStatus.CAP_BLOCKED
                    dry.http_status = 200
                    dry.raw["value_sizing"] = vs
                    dry.raw["latency_gate"] = {
                        "enabled": bool(latency_gate_enabled),
                        "max_sec": float(latency_gate_max_sec),
                        "pre_submit_ms": (int(pre_ms) if pre_ms is not None else None),
                    }
                    dry.error = (
                        f"LATENCY_GATE_BACK pre_submit_ms={int(pre_ms) if pre_ms is not None else -1} "
                        f"max_ms={int(float(latency_gate_max_sec) * 1000.0)}"
                    )
                    return dry
            else:
                if req.exec_side != ExecSide.BACK:
                    vs.setdefault("skip_reason", "not_back")
                elif not bool(sizing_enabled):
                    vs.setdefault("skip_reason", "disabled")
            dry.raw["value_sizing"] = vs
        except Exception:
            pass

        # valida max stake após sizing
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
        order_post_ms = _ms(max(0.0, time.time() - t_place0))

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
                order_post_ms = _ms(max(0.0, time.time() - t_place1))

        # 4) Montar resultado LIVE + cleanup do betslip (evita too_many_open_betslips)
        close_betslip_live = None
        try:
            t_cl = time.time()
            ok_cl = True
            try:
                await asyncio.wait_for(self._api.close_betslip(betslip_id), timeout=float(os.getenv("EXECUTOR_LIVE_CLOSE_TIMEOUT_SEC", "1.2")))
            except Exception:
                ok_cl = False
            close_betslip_live = {"ok": bool(ok_cl), "ms": _ms(max(0.0, time.time() - t_cl))}
        except Exception:
            pass

        if place.success:
            dry.status = ExecStatus.LIVE_OK
            dry.http_status = int(place.http_status or 0) or 200
            try:
                # Mantém timing.post_ms do dryrun (betslip) e separa o POST do place_order.
                dry.timing.order_post_ms = int(order_post_ms)
            except Exception:
                pass
            try:
                dry.timing.call_to_done_ms = _ms(max(0.0, time.time() - float(req.created_at.timestamp())))
            except Exception:
                pass
            dry.raw = dict(dry.raw or {})
            oid = _extract_order_id(place.response)
            # instrumentação: cleanup e decomposição local
            try:
                cl = dict((dry.raw.get("cleanup") or {}))
                if close_betslip_live is not None:
                    cl["close_betslip_live"] = close_betslip_live
                dry.raw["cleanup"] = cl
            except Exception:
                pass
            dry.raw.update(
                {
                    "live": True,
                    "order_resp": place.response,
                    "order_id": oid,
                    "order_http": int(place.http_status or 0),
                    "order_ms": int(place.request_time_ms or 0),
                    "order_text_prefix": (place.text_prefix or "")[:300],
                    "sent": {"stake_ccy": stake_ccy, "stake": float(stake), "price": float(price)},
                }
            )
            try:
                bd = dict(dry.raw.get("timing_breakdown") or {})
                live_worker_ms = _ms(max(0.0, time.time() - float(t_live0)))
                bd["live_worker_ms"] = int(live_worker_ms)
                bd["order_post_ms"] = int(order_post_ms)
                bd["close_betslip_live_ms"] = (int(close_betslip_live.get("ms")) if isinstance(close_betslip_live, dict) and close_betslip_live.get("ms") is not None else None)
                # overhead aproximado: call_to_done - total_api - order_post
                try:
                    ctd = _ms(max(0.0, time.time() - float(req.created_at.timestamp())))
                    tot_api = int(dry.timing.total_ms) if dry.timing and dry.timing.total_ms is not None else None
                    if tot_api is not None:
                        bd["overhead_ex_order_ms"] = int(max(0, int(ctd) - int(tot_api) - int(order_post_ms)))
                except Exception:
                    pass
                dry.raw["timing_breakdown"] = bd
            except Exception:
                pass

            # Housekeeping pós-LIVE: agora é seguro (aposta já foi enviada e betslip fechado)
            try:
                ctd2 = _ms(max(0.0, time.time() - float(req.created_at.timestamp())))
                tot_api2 = int(dry.timing.total_ms) if dry.timing and dry.timing.total_ms is not None else None
                call_minus_total2 = int(max(0, int(ctd2) - int(tot_api2))) if tot_api2 is not None else None
                overhead_ex_order2 = int(max(0, int(ctd2) - int(tot_api2) - int(order_post_ms))) if tot_api2 is not None else None
                await self._housekeep_post_request(
                    res=dry,
                    allow_ui=True,
                    allow_close_cache=True,
                    allow_strong=True,
                    overhead_local_ms=overhead_ex_order2,
                    call_minus_total_ms=call_minus_total2,
                    reason="live",
                )
            except Exception:
                pass

            # call_to_done no final (inclui cleanup/housekeeping)
            try:
                dry.timing.call_to_done_ms = _ms(max(0.0, time.time() - float(req.created_at.timestamp())))
            except Exception:
                pass
            dry.finished_at = _now_utc()
            return dry

        dry.status = ExecStatus.API_FAILED
        dry.http_status = int(place.http_status or 0) or None
        dry.error = f"LIVE_PLACE_FAILED: {place.error or 'unknown'}"
        if _is_playwright_context_destroyed(dry.error) or _is_playwright_target_closed(dry.error) or _is_login_navigation_timeout(dry.error):
            await self._restart_browser_session(reason=f"LIVE_PLACE_FAILED:{str(place.error)[:160]}")
        try:
            dry.timing.order_post_ms = int(order_post_ms)
        except Exception:
            pass
        try:
            dry.timing.call_to_done_ms = _ms(max(0.0, time.time() - float(req.created_at.timestamp())))
        except Exception:
            pass
        dry.raw = dict(dry.raw or {})
        oid = _extract_order_id(place.response)
        try:
            cl = dict((dry.raw.get("cleanup") or {}))
            if close_betslip_live is not None:
                cl["close_betslip_live"] = close_betslip_live
            dry.raw["cleanup"] = cl
        except Exception:
            pass
        dry.raw.update(
            {
                "live": True,
                "order_id": oid,
                "order_http": int(place.http_status or 0),
                "order_ms": int(place.request_time_ms or 0),
                "order_text_prefix": (place.text_prefix or "")[:300],
                "sent": {"stake_ccy": stake_ccy, "stake": float(stake), "price": float(price)},
            }
        )
        try:
            bd = dict(dry.raw.get("timing_breakdown") or {})
            live_worker_ms = _ms(max(0.0, time.time() - float(t_live0)))
            bd["live_worker_ms"] = int(live_worker_ms)
            bd["order_post_ms"] = int(order_post_ms)
            dry.raw["timing_breakdown"] = bd
        except Exception:
            pass

        # Housekeeping pós-LIVE (falha): ainda é seguro após close_betslip best-effort.
        try:
            ctd2 = _ms(max(0.0, time.time() - float(req.created_at.timestamp())))
            tot_api2 = int(dry.timing.total_ms) if dry.timing and dry.timing.total_ms is not None else None
            call_minus_total2 = int(max(0, int(ctd2) - int(tot_api2))) if tot_api2 is not None else None
            overhead_ex_order2 = int(max(0, int(ctd2) - int(tot_api2) - int(order_post_ms))) if tot_api2 is not None else None
            await self._housekeep_post_request(
                res=dry,
                allow_ui=True,
                allow_close_cache=True,
                allow_strong=True,
                overhead_local_ms=overhead_ex_order2,
                call_minus_total_ms=call_minus_total2,
                reason="live_failed",
            )
        except Exception:
            pass

        # call_to_done no final (inclui cleanup/housekeeping)
        try:
            dry.timing.call_to_done_ms = _ms(max(0.0, time.time() - float(req.created_at.timestamp())))
        except Exception:
            pass
        dry.finished_at = _now_utc()
        return dry

