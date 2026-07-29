from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import UUID

from loguru import logger
from sqlalchemy import text

from storage.database import Database

try:
    from ops.h3bup_e2e_trace import (
        emit_trace_event as _e2e_emit,
        extract_trace_id_from_meta as _e2e_from_meta,
        map_exec_status as _e2e_map_status,
    )
except Exception:  # pragma: no cover
    def _e2e_emit(*_a, **_k):
        return None
    def _e2e_from_meta(*_a, **_k):
        return None
    def _e2e_map_status(*_a, **_k):
        return "UNKNOWN"


@dataclass
class ResultStore:
    """
    Armazenamento simples:
    - memória (para GET /result)
    - JSONL (audit trail)
    - opcional: PostgreSQL (para análises OOS futuras)
    """

    jsonl_path: Optional[Path] = None
    keep_seconds: int = 3600
    save_to_db: bool = False

    _mem: Dict[str, Dict[str, Any]] = None
    _lock: asyncio.Lock = None
    _db: Optional[Database] = None

    def __post_init__(self):
        self._mem = {}
        self._lock = asyncio.Lock()

    async def start(self):
        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        if self.save_to_db:
            self._db = Database()
            await self._db.connect()
            await self._ensure_table()

    async def close(self):
        if self._db:
            await self._db.close()
            self._db = None

    async def _ensure_table(self):
        assert self._db is not None
        ddl = """
        CREATE TABLE IF NOT EXISTS execution_dryrun_attempts (
            id BIGSERIAL PRIMARY KEY,
            execution_id UUID UNIQUE NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            finished_at TIMESTAMPTZ NOT NULL,
            status TEXT NOT NULL,
            audit_id BIGINT NULL,
            match_id BIGINT NULL,
            event_id TEXT NOT NULL,
            market_type TEXT NOT NULL,
            side TEXT NOT NULL,
            line TEXT NOT NULL,
            exec_side TEXT NOT NULL,
            is_live BOOLEAN NOT NULL,
            odd_at_decision DOUBLE PRECISION NULL,
            odd_final DOUBLE PRECISION NULL,
            delta_pct DOUBLE PRECISION NULL,
            total_ms INTEGER NULL,
            post_ms INTEGER NULL,
            http_status INTEGER NULL,
            retry_after_sec INTEGER NULL,
            error TEXT NULL,
            policy_version TEXT NULL,
            request_json JSONB NULL,
            result_json JSONB NULL
        );
        """
        async with self._db.engine.begin() as conn:
            await conn.execute(text(ddl))
        logger.info("[executor] Tabela execution_dryrun_attempts pronta")

    async def put(self, execution_id: UUID, payload: Dict[str, Any]):
        now = time.time()
        _tid = None
        _mono0 = time.monotonic_ns()
        try:
            req = (payload or {}).get("request") or {}
            res = (payload or {}).get("result") or {}
            _tid = _e2e_from_meta(req.get("meta"))
            if _tid:
                _e2e_emit(
                    "H3B_RESULT_PERSIST_STARTED",
                    trace_id=_tid,
                    audit_id=req.get("audit_id") or res.get("audit_id"),
                    execution_id=str(execution_id),
                    order_id=(res.get("raw") or {}).get("order_id") if isinstance(res.get("raw"), dict) else None,
                    event_id=req.get("event_id") or res.get("event_id"),
                    market_type=req.get("market_type") or res.get("market_type"),
                    side=req.get("side") or res.get("side"),
                    line=req.get("line") or res.get("line"),
                    policy_version=((req.get("policy") or {}).get("policy_version")),
                    status=_e2e_map_status(res.get("status"), res.get("error")),
                    service="executor",
                )
        except Exception:
            pass
        async with self._lock:
            self._mem[str(execution_id)] = {"t": now, "v": payload}
            # GC simples
            dead = [k for k, it in self._mem.items() if (now - float(it.get("t", 0))) > float(self.keep_seconds)]
            for k in dead:
                self._mem.pop(k, None)

        if self.jsonl_path:
            try:
                with self.jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.warning(f"[executor] Falha ao escrever JSONL: {e}")

        if self._db:
            await self._insert_db(payload)
        try:
            if _tid:
                req = (payload or {}).get("request") or {}
                res = (payload or {}).get("result") or {}
                oid = None
                try:
                    oid = (res.get("raw") or {}).get("order_id") if isinstance(res.get("raw"), dict) else None
                except Exception:
                    oid = None
                _e2e_emit(
                    "H3B_RESULT_PERSIST_FINISHED",
                    trace_id=_tid,
                    audit_id=req.get("audit_id") or res.get("audit_id"),
                    execution_id=str(execution_id),
                    order_id=oid,
                    event_id=req.get("event_id") or res.get("event_id"),
                    market_type=req.get("market_type") or res.get("market_type"),
                    side=req.get("side") or res.get("side"),
                    line=req.get("line") or res.get("line"),
                    policy_version=((req.get("policy") or {}).get("policy_version")),
                    status=_e2e_map_status(res.get("status"), res.get("error")),
                    service="executor",
                    duration_ms=(time.monotonic_ns() - _mono0) / 1e6,
                    metadata={"result_persist_finished_ts": time.time()},
                )
        except Exception:
            pass

    async def get(self, execution_id: UUID) -> Optional[Dict[str, Any]]:
        async with self._lock:
            it = self._mem.get(str(execution_id))
            return (it or {}).get("v") if it else None

    async def _insert_db(self, payload: Dict[str, Any]):
        assert self._db is not None
        try:
            req = (payload or {}).get("request") or {}
            res = (payload or {}).get("result") or {}
            timing = (res or {}).get("timing") or {}
            policy = (req or {}).get("policy") or {}
            q = text(
                """
                INSERT INTO execution_dryrun_attempts (
                    execution_id, created_at, finished_at, status,
                    audit_id, match_id, event_id, market_type, side, line, exec_side, is_live,
                    odd_at_decision, odd_final, delta_pct, total_ms, post_ms, http_status, retry_after_sec, error,
                    policy_version, request_json, result_json
                )
                VALUES (
                    :execution_id, :created_at, :finished_at, :status,
                    :audit_id, :match_id, :event_id, :market_type, :side, :line, :exec_side, :is_live,
                    :odd_at_decision, :odd_final, :delta_pct, :total_ms, :post_ms, :http_status, :retry_after_sec, :error,
                    :policy_version, (:request_json)::jsonb, (:result_json)::jsonb
                )
                ON CONFLICT (execution_id) DO NOTHING;
                """
            )
            async with self._db.engine.begin() as conn:
                await conn.execute(
                    q,
                    {
                        "execution_id": str(res.get("execution_id") or req.get("execution_id")),
                        "created_at": res.get("created_at") or req.get("created_at"),
                        "finished_at": res.get("finished_at"),
                        "status": res.get("status"),
                        "audit_id": res.get("audit_id"),
                        "match_id": res.get("match_id"),
                        "event_id": res.get("event_id"),
                        "market_type": res.get("market_type"),
                        "side": res.get("side"),
                        "line": res.get("line"),
                        "exec_side": res.get("exec_side"),
                        "is_live": bool(res.get("is_live")),
                        "odd_at_decision": res.get("odd_at_decision"),
                        "odd_final": res.get("odd_final"),
                        "delta_pct": res.get("delta_pct"),
                        "total_ms": timing.get("total_ms"),
                        "post_ms": timing.get("post_ms"),
                        "http_status": res.get("http_status"),
                        "retry_after_sec": res.get("retry_after_sec"),
                        "error": res.get("error"),
                        "policy_version": policy.get("policy_version"),
                        "request_json": json.dumps(req, ensure_ascii=False),
                        "result_json": json.dumps(res, ensure_ascii=False),
                    },
                )
        except Exception as e:
            logger.warning(f"[executor] Falha ao inserir no DB: {e}")

