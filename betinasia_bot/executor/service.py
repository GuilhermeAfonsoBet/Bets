from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import UUID

from aiohttp import web
from loguru import logger

from scraper.betinasia import BetinAsiaScraper

from .contracts import ExecutionRequest, ExecutionResult, ExecStatus
from .store import ResultStore
from .worker import ExecutorWorker, _now_utc


@dataclass
class ExecutorService:
    football_url: str
    workers: int = 1
    cap_window_sec: float = 300.0
    cap_max: int = 999999
    jsonl_path: Optional[str] = None
    save_to_db: bool = False

    _queue: asyncio.Queue = None
    _worker_tasks: list = None
    _workers: list = None
    _store: ResultStore = None
    _start_task: Optional[asyncio.Task] = None
    _ready: bool = False
    _start_error: Optional[str] = None

    async def start(self):
        self._start_error = None
        self._ready = False
        self._queue = asyncio.Queue(maxsize=int(os.getenv("EXECUTOR_QUEUE_MAX", "200")))
        self._worker_tasks = []
        self._workers = []

        self._store = ResultStore(
            jsonl_path=(None if not self.jsonl_path else __import__("pathlib").Path(self.jsonl_path)),
            keep_seconds=int(os.getenv("EXECUTOR_RESULT_TTL_SEC", "3600")),
            save_to_db=bool(self.save_to_db),
        )
        await self._store.start()

        for i in range(int(self.workers)):
            w = ExecutorWorker(
                name=f"w{i+1}",
                football_url=self.football_url,
                open_cap_window_sec=float(self.cap_window_sec),
                open_cap_max=int(self.cap_max),
                enable_cap=True,
            )
            await w.start()
            self._workers.append(w)
            t = asyncio.create_task(self._run_worker_loop(w))
            self._worker_tasks.append(t)
        self._ready = True
        logger.info(f"[executor] service started workers={len(self._workers)}")

    async def close(self):
        for t in self._worker_tasks or []:
            t.cancel()
        for w in self._workers or []:
            try:
                await w.close()
            except Exception:
                pass
        if self._store:
            await self._store.close()

    async def _run_worker_loop(self, worker: ExecutorWorker):
        while True:
            item = await self._queue.get()
            try:
                req = item["req"]
                received_ts = item["received_ts"]
                res = await worker.execute(req, received_ts)
                payload = {
                    "request": req.model_dump(mode="json"),
                    "result": res.model_dump(mode="json"),
                }
                await self._store.put(req.execution_id, payload)
            except Exception as e:
                try:
                    logger.exception(f"[executor:{worker.name}] erro: {e}")
                except Exception:
                    pass
                # Evita `not_found`: sempre grava um resultado, mesmo em erro inesperado.
                try:
                    req = item.get("req")
                    if req:
                        res = ExecutionResult(
                            execution_id=req.execution_id,
                            status=ExecStatus.INTERNAL_ERROR,
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
                            policy=req.policy,
                            error=str(e)[:500],
                            raw={"where": "_run_worker_loop"},
                        )
                        payload = {
                            "request": req.model_dump(mode="json"),
                            "result": res.model_dump(mode="json"),
                        }
                        await self._store.put(req.execution_id, payload)
                except Exception:
                    pass
            finally:
                self._queue.task_done()

    async def submit(self, req: ExecutionRequest) -> Dict[str, Any]:
        received_ts = time.time()
        try:
            self._queue.put_nowait({"req": req, "received_ts": received_ts})
        except asyncio.QueueFull:
            res = ExecutionResult(
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
                policy=req.policy,
                error="QUEUE_FULL",
            )
            payload = {"request": req.model_dump(mode="json"), "result": res.model_dump(mode="json")}
            await self._store.put(req.execution_id, payload)
            return {"accepted": False, "execution_id": str(req.execution_id), "status": "QUEUE_FULL"}
        return {"accepted": True, "execution_id": str(req.execution_id)}

    async def get_result(self, execution_id: UUID) -> Optional[Dict[str, Any]]:
        return await self._store.get(execution_id)


def create_app(svc: ExecutorService) -> web.Application:
    app = web.Application(client_max_size=1_000_000)

    async def health(_req: web.Request) -> web.Response:
        return web.json_response(
            {
                "ok": True,
                "ready": bool(getattr(svc, "_ready", False)),
                "start_error": getattr(svc, "_start_error", None),
                "workers": len(svc._workers or []),
                "queue": (svc._queue.qsize() if svc._queue else 0),
            }
        )

    async def execute(req: web.Request) -> web.Response:
        if not bool(getattr(svc, "_ready", False)):
            return web.json_response({"error": "not_ready", "detail": getattr(svc, "_start_error", None)}, status=503)
        try:
            data = await req.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)
        try:
            er = ExecutionRequest.model_validate(data)
        except Exception as e:
            return web.json_response({"error": "invalid_request", "detail": str(e)}, status=400)

        out = await svc.submit(er)
        return web.json_response(out, status=202 if out.get("accepted") else 429)

    async def result(req: web.Request) -> web.Response:
        eid = req.match_info.get("execution_id")
        try:
            uid = UUID(str(eid))
        except Exception:
            return web.json_response({"error": "bad_execution_id"}, status=400)
        payload = await svc.get_result(uid)
        if not payload:
            return web.json_response({"error": "not_found"}, status=404)
        return web.json_response(payload)

    async def account(req: web.Request) -> web.Response:
        if not bool(getattr(svc, "_ready", False)):
            return web.json_response({"error": "not_ready", "detail": getattr(svc, "_start_error", None)}, status=503)
        try:
            page_size = int(req.query.get("page_size") or "50")
        except Exception:
            page_size = 50
        if not svc._workers:
            return web.json_response({"error": "no_workers"}, status=503)
        try:
            snap = await svc._workers[0].get_account_snapshot(page_size=page_size)
            return web.json_response(snap)
        except Exception as e:
            return web.json_response({"error": "account_failed", "detail": str(e)[:300]}, status=500)

    app.router.add_get("/health", health)
    app.router.add_post("/execute", execute)
    app.router.add_get("/result/{execution_id}", result)
    app.router.add_get("/account", account)

    async def on_startup(_app: web.Application):
        # Não bloquear o bind do socket: aquecer workers em background
        async def _bg_start():
            try:
                await svc.start()
            except Exception as e:
                try:
                    svc._start_error = str(e)[:300]
                except Exception:
                    pass
                raise

        svc._start_task = asyncio.create_task(_bg_start())

        def _on_done(t: asyncio.Task):
            try:
                _ = t.result()
            except Exception as e:
                # Se o start falhar, o serviço fica "not_ready" e o bridge começa a receber 503.
                # Loga explicitamente para aparecer em executor_error.log (systemd StandardError).
                try:
                    logger.error(f"[executor] start failed: {str(e)[:400]}")
                except Exception:
                    pass

        try:
            svc._start_task.add_done_callback(_on_done)
        except Exception:
            pass

        # Watchdog opcional: se o executor não ficar pronto em X segundos, reinicia via systemd.
        # Isso resolve casos de Playwright/sessão travada que deixam o /execute em 503 indefinidamente.
        try:
            deadline = float(os.getenv("EXECUTOR_STARTUP_DEADLINE_SEC", "600") or 600.0)
        except Exception:
            deadline = 600.0

        async def _watchdog():
            if deadline <= 0:
                return
            t0 = time.time()
            while True:
                await asyncio.sleep(2.0)
                if bool(getattr(svc, "_ready", False)):
                    return
                if (time.time() - t0) >= float(deadline):
                    try:
                        svc._start_error = str(svc._start_error or "startup_timeout")[:300]
                    except Exception:
                        pass
                    try:
                        logger.error(f"[executor] startup timeout after {deadline:.0f}s (ready=false). Exiting for restart.")
                    except Exception:
                        pass
                    os._exit(2)

        try:
            asyncio.create_task(_watchdog())
        except Exception:
            pass

    async def on_cleanup(_app: web.Application):
        await svc.close()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app

