"""JSONL-first obligation/snapshot store with optional Postgres DDL."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ops.h3bup_clv_config import WINDOWS, ClvConfig, load_config


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso(dt: Any = None) -> str:
    if dt is None:
        dt = utc_now()
    if isinstance(dt, (int, float)):
        dt = datetime.fromtimestamp(float(dt), tz=timezone.utc)
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    return str(dt)


def parse_ts(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


class ClvJsonlStore:
    """Append-only JSONL + in-memory index for obligations/snapshots.

    Idempotency: unique key order_id|window|schema_version in memory+file scan on load.
    """

    def __init__(self, cfg: Optional[ClvConfig] = None) -> None:
        self.cfg = cfg or load_config()
        self._lock = threading.RLock()
        self._obligations: Dict[str, Dict[str, Any]] = {}
        self._by_order: Dict[str, List[str]] = {}
        self._snapshots: List[Dict[str, Any]] = []
        self._loaded = False

    def obligation_key(self, order_id: str, window_name: str, schema_version: int) -> str:
        return f"{order_id}|{window_name}|{int(schema_version)}"

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            path = Path(self.cfg.obligations_jsonl)
            if path.exists():
                for line in path.open("r", encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        o = json.loads(line)
                    except Exception:
                        continue
                    k = o.get("obligation_key") or self.obligation_key(
                        str(o.get("order_id")), str(o.get("window_name")), int(o.get("schema_version") or 1)
                    )
                    # last write wins
                    self._obligations[k] = o
            for k, o in self._obligations.items():
                oid = str(o.get("order_id") or "")
                self._by_order.setdefault(oid, [])
                if k not in self._by_order[oid]:
                    self._by_order[oid].append(k)
            sp = Path(self.cfg.snapshots_jsonl)
            if sp.exists():
                for line in sp.open("r", encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._snapshots.append(json.loads(line))
                    except Exception:
                        continue
            self._loaded = True

    def _append(self, path: Path, obj: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")

    def get_obligation(self, key: str) -> Optional[Dict[str, Any]]:
        self._ensure_loaded()
        with self._lock:
            return dict(self._obligations[key]) if key in self._obligations else None

    def list_obligations(self, *, status: Optional[str] = None, window: Optional[str] = None) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        with self._lock:
            out = []
            for o in self._obligations.values():
                if status and o.get("status") != status:
                    continue
                if window and o.get("window_name") != window:
                    continue
                out.append(dict(o))
            return out

    def upsert_obligation(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_loaded()
        with self._lock:
            key = obj.get("obligation_key") or self.obligation_key(
                str(obj["order_id"]), str(obj["window_name"]), int(obj.get("schema_version") or self.cfg.schema_version)
            )
            obj = dict(obj)
            obj["obligation_key"] = key
            obj["updated_at_utc"] = utc_iso()
            existing = self._obligations.get(key)
            if existing and existing.get("status") in ("COMPLETED", "SKIPPED", "FAILED_FINAL", "CANCELLED"):
                # do not regress terminal unless explicitly forced
                if not obj.get("_force"):
                    return dict(existing)
            obj.pop("_force", None)
            if not existing:
                obj.setdefault("created_at_utc", utc_iso())
                obj.setdefault("id", str(uuid.uuid4()))
                obj.setdefault("attempts", 0)
            else:
                obj.setdefault("id", existing.get("id"))
                obj.setdefault("created_at_utc", existing.get("created_at_utc"))
                if "attempts" not in obj:
                    obj["attempts"] = existing.get("attempts", 0)
            self._obligations[key] = obj
            oid = str(obj.get("order_id") or "")
            self._by_order.setdefault(oid, [])
            if key not in self._by_order[oid]:
                self._by_order[oid].append(key)
            self._append(Path(self.cfg.obligations_jsonl), obj)
            return dict(obj)

    def has_order(self, order_id: str) -> bool:
        self._ensure_loaded()
        with self._lock:
            return bool(self._by_order.get(str(order_id)))

    def append_snapshot(self, snap: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_loaded()
        with self._lock:
            snap = dict(snap)
            snap.setdefault("id", str(uuid.uuid4()))
            snap.setdefault("created_at_utc", utc_iso())
            snap.setdefault("schema_version", self.cfg.schema_version)
            self._snapshots.append(snap)
            self._append(Path(self.cfg.snapshots_jsonl), snap)
            return snap

    def snapshots(self) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        with self._lock:
            return list(self._snapshots)


_STORE: Optional[ClvJsonlStore] = None
_STORE_LOCK = threading.Lock()


def get_store(cfg: Optional[ClvConfig] = None) -> ClvJsonlStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = ClvJsonlStore(cfg or load_config())
        return _STORE


DDL_OBLIGATIONS = """
CREATE TABLE IF NOT EXISTS h3bup_clv_obligations (
  id TEXT PRIMARY KEY,
  schema_version INTEGER NOT NULL,
  obligation_key TEXT NOT NULL UNIQUE,
  order_id TEXT NOT NULL,
  execution_id TEXT,
  audit_id BIGINT,
  trace_id TEXT,
  policy_version TEXT,
  event_id TEXT,
  event_name TEXT,
  market_type TEXT,
  period TEXT,
  side TEXT,
  line TEXT,
  entry_odd DOUBLE PRECISION,
  entry_odd_source TEXT,
  odd_at_decision DOUBLE PRECISION,
  odd_final DOUBLE PRECISION,
  live_ok_ts_utc TIMESTAMPTZ,
  kickoff_ts_utc TIMESTAMPTZ,
  kickoff_source TEXT,
  kickoff_confidence TEXT,
  window_name TEXT NOT NULL,
  target_ts_utc TIMESTAMPTZ,
  status TEXT NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0,
  next_attempt_ts_utc TIMESTAMPTZ,
  last_error_code TEXT,
  last_error_message TEXT,
  created_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  completed_at_utc TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_h3bup_clv_obl_status ON h3bup_clv_obligations(status, next_attempt_ts_utc);
CREATE INDEX IF NOT EXISTS idx_h3bup_clv_obl_order ON h3bup_clv_obligations(order_id);
"""

DDL_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS h3bup_clv_snapshots (
  id TEXT PRIMARY KEY,
  schema_version INTEGER NOT NULL,
  obligation_id TEXT,
  obligation_key TEXT,
  order_id TEXT,
  execution_id TEXT,
  audit_id BIGINT,
  trace_id TEXT,
  policy_version TEXT,
  window_name TEXT,
  target_ts_utc TIMESTAMPTZ,
  snapshot_ts_utc TIMESTAMPTZ,
  snapshot_distance_sec DOUBLE PRECISION,
  kickoff_ts_utc TIMESTAMPTZ,
  source TEXT,
  source_record_id TEXT,
  event_id TEXT,
  market_type TEXT,
  period TEXT,
  side TEXT,
  line TEXT,
  entry_odd DOUBLE PRECISION,
  entry_odd_source TEXT,
  snapshot_odd DOUBLE PRECISION,
  clv_raw_decimal DOUBLE PRECISION,
  clv_raw_pct DOUBLE PRECISION,
  same_event_flag BOOLEAN,
  same_market_flag BOOLEAN,
  same_period_flag BOOLEAN,
  same_side_flag BOOLEAN,
  same_line_flag BOOLEAN,
  same_line_strict_flag BOOLEAN,
  snapshot_before_kickoff_flag BOOLEAN,
  quality_status TEXT,
  failure_reason TEXT,
  created_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_h3bup_clv_snap_order ON h3bup_clv_snapshots(order_id, window_name);
"""

DDL_PASSIVE = """
CREATE TABLE IF NOT EXISTS h3bup_clv_passive_snapshots (
  id TEXT PRIMARY KEY,
  obligation_id TEXT,
  order_id TEXT,
  event_id TEXT,
  market_type TEXT,
  period TEXT,
  side TEXT,
  line TEXT,
  observed_odd DOUBLE PRECISION,
  observed_ts_utc TIMESTAMPTZ,
  source TEXT,
  source_sequence TEXT,
  kickoff_ts_utc TIMESTAMPTZ,
  created_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_h3bup_clv_passive_key ON h3bup_clv_passive_snapshots(event_id, line, side, observed_ts_utc);
"""


async def ensure_postgres_schema(engine) -> None:
    """Best-effort DDL; never raises to caller of execution path."""
    try:
        from sqlalchemy import text

        async with engine.begin() as conn:
            for ddl in (DDL_OBLIGATIONS, DDL_SNAPSHOTS, DDL_PASSIVE):
                for stmt in ddl.split(";"):
                    s = stmt.strip()
                    if s:
                        await conn.execute(text(s))
    except Exception:
        pass
