#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auditoria H3B via API — Versão rápida (~2-3s) sem DOM

Arquitetura:
  - Monitor WS permanente (detecta H3B em todas as ligas)
  - Quando H3B detectado: POST /v1/betslips/ + escuta PMM via WS
  - Extrai best odd + limite de JSON estruturado
  - Sem browser DOM, sem page load, sem click, sem parsing de texto

Uso:
    DISPLAY=:99 python audit_h3b_api.py
    DISPLAY=:99 python audit_h3b_api.py --num-audits 20
"""

import asyncio
import argparse
import json
import os
import signal
import sys
import time
import gc
from collections import deque
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from loguru import logger
import random

sys.path.insert(0, '.')

from config import settings
from scraper.betinasia import BetinAsiaScraper
from scraper.api_betslip import ApiBetslipClient, BetslipApiResult
from hypothesis.detectors import HypothesisDetector
from sqlalchemy import text
from storage.database import Database
from storage.models_hypothesis import BetslipAuditResult

FOOTBALL_URL = "https://black.betinasia.com/sportsbook/football"
MAX_AH_LINE = 10.0  # Amplo: captura todas as linhas relevantes
WS_HEALTH_INTERVAL = 15
WS_RELOAD_INTERVAL = 120
STATS_INTERVAL = 50
RUNTIME_EVENTS_TABLE = "bot_runtime_events"


class H3bApiAudit:

    def __init__(
        self,
        num_audits: int = 0,
        direction: str = "up",
        save_to_db: bool = True,
        executor_workers: int = 4,
        temporal_workers: int = 2,
        max_queue_depth: int = 50,
        max_queue_wait_ms: int = 5000,
        mode: str = "api",
        ws_sample_offsets_sec: Optional[List[float]] = None,
        gate_drop_offset_sec: float = 5.0,
        gate_drop_ratio: float = 0.98,
        gate_rise_offset_sec: float = 5.0,
        gate_rise_ratio: float = 1.02,
        gate_open_window_sec: int = 300,
        gate_open_max: int = 3,
        gate_max_late_sec: float = 2.5,
        gate_lay_refresh: bool = False,
        gate_lay_refresh_times_sec: Optional[List[float]] = None,
        api_sides: str = "both",
    ):
        self.num_audits = num_audits
        self.direction = direction
        self.save_to_db = save_to_db
        self.executor_workers = max(1, int(executor_workers))
        self.temporal_workers = max(0, int(temporal_workers))
        self.max_queue_depth = max(0, int(max_queue_depth))
        self.max_queue_wait_ms = max(0, int(max_queue_wait_ms))

        self.scraper: Optional[BetinAsiaScraper] = None
        self.api_client: Optional[ApiBetslipClient] = None
        self.db: Optional[Database] = None

        # WS
        # Buffer WS limitado para evitar OOM (o monitor consome continuamente).
        self._ws_messages: deque[str] = deque()
        self._ws_messages_dropped: int = 0
        try:
            self._ws_messages_max = int(float(os.getenv("AUDIT_WS_BUFFER_MAX", "20000") or 20000))
        except Exception:
            self._ws_messages_max = 20000
        self._ws_msg_count: int = 0
        self._last_ws_time: float = 0
        self._start_time: float = 0
        self._events_info: Dict[str, dict] = {}
        # Estado mais recente de odds via WS por (event_id, market_type, market_period, line)
        self._ws_odds_state: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}

        # Detector
        self.detector = HypothesisDetector()

        # Stats
        # Resultados limitados (evita consumo infinito de RAM).
        self.results: deque[dict] = deque()
        try:
            self._results_max = int(float(os.getenv("AUDIT_RESULTS_MAX", "5000") or 5000))
        except Exception:
            self._results_max = 5000
        self.events_processed: int = 0
        self.h3b_detected: int = 0
        self.total_errors: int = 0
        self.consecutive_errors: int = 0
        self.running = True
        self.telemetry_file = "logs/audit_api_telemetry.jsonl"
        self.runtime_events_file = "logs/audit_runtime_events.jsonl"
        self._runtime_last_emit: Dict[str, float] = {}
        self.max_queue_depth_observed = 0
        self._queue_ref: Optional[asyncio.Queue] = None
        self.max_temporal_queue_depth_observed = 0
        self._temporal_queue_ref: Optional[asyncio.Queue] = None
        self.dropped_full_queue: int = 0
        self.dropped_stale_queue_wait: int = 0
        # Backoff global do betslip API (ex.: rate limit). Evita flood e “lock” prolongado.
        self._api_backoff_until_ts: float = 0.0
        self._relogin_lock = asyncio.Lock()
        self._last_relogin_ts: float = 0.0
        self._shutdown_lock = asyncio.Lock()
        self._tasks: List[asyncio.Task] = []

        # Deduplicação com TTL (evita set infinito).
        try:
            self._audited_ttl_sec = float(os.getenv("AUDIT_DEDUP_TTL_SEC", "7200") or 7200)
        except Exception:
            self._audited_ttl_sec = 7200.0
        self._audited_ts: Dict[str, float] = {}

        # Modos:
        # - api: comportamento atual (WS detecta + BS via API para validar)
        # - ws_only: coleta só WS (t0..t+30) e prepara dados para motor de decisão
        # - ws_vs_bs: auditoria comparativa (WS + BS no mesmo timestamp)
        # - ws_gate_lay: WS-only para gate (t0,t+5), abre ticket LAY só quando elegível
        # - ws_reversal_lay: abre ticket LAY imediatamente quando houver reversão (H3B)
        # - ws_gate_back: WS-only para gate (t0,t+5), marca oportunidade BACK quando elegível (sem abrir betslip)
        self.mode = str(mode or "api").strip().lower()
        self.ws_sample_offsets_sec = ws_sample_offsets_sec or [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]

        # Gate (H3B + queda em 5s) -> abre ticket LAY (Exchange) sob cap
        self.gate_drop_offset_sec = max(0.0, float(gate_drop_offset_sec))
        self.gate_drop_ratio = float(gate_drop_ratio)
        if self.gate_drop_ratio <= 0 or self.gate_drop_ratio >= 1.0:
            # sane default: 0.98 (= queda >2%)
            self.gate_drop_ratio = 0.98

        # Gate (H3B + subida em 5s) -> marca oportunidade BACK via WS (sem abrir betslip)
        self.gate_rise_offset_sec = max(0.0, float(gate_rise_offset_sec))
        self.gate_rise_ratio = float(gate_rise_ratio)
        if self.gate_rise_ratio <= 1.0:
            # sane default: 1.02 (= alta >=2%)
            self.gate_rise_ratio = 1.02
        self.gate_open_window_sec = max(30, int(gate_open_window_sec))
        self.gate_open_max = max(0, int(gate_open_max))
        self.gate_max_late_sec = max(0.0, float(gate_max_late_sec))
        self.gate_open_lock = asyncio.Lock()
        self.gate_open_times = deque()
        self.gate_lay_refresh = bool(gate_lay_refresh)
        self.gate_lay_refresh_times_sec = (
            [0.0, 5.0, 10.0, 15.0, 20.0]
            if (gate_lay_refresh_times_sec is None)
            else [float(x) for x in gate_lay_refresh_times_sec if x is not None and float(x) >= 0.0]
        )
        self.api_sides = (str(api_sides or "both").strip().lower() or "both")
        if self.api_sides not in ("back", "lay", "both"):
            self.api_sides = "both"

        # Contadores (observabilidade)
        self.gate_seen = 0
        self.gate_ws_missing = 0
        self.gate_not_eligible = 0
        self.gate_eligible = 0
        self.gate_blocked_cap = 0
        self.gate_blocked_backoff = 0
        self.gate_open_attempts = 0
        self.gate_open_success = 0
        self.gate_open_failed = 0

        self.gate_back_seen = 0
        self.gate_back_ws_missing = 0
        self.gate_back_not_eligible = 0
        self.gate_back_eligible = 0

        # Política financeira (insumos para análise econômica posterior)
        # Não afeta execução, apenas persistência de variáveis para analytics.
        self.finance_stake_pct_of_limit = self._parse_env_float(
            "FINANCE_STAKE_PCT_OF_LIMIT", 0.25
        )
        self.finance_stake_cap = self._parse_env_float(
            "FINANCE_STAKE_CAP", 0.0
        )
        self.finance_fx_brl = self._parse_env_float(
            "FINANCE_FX_BRL", 5.20
        )
        self.finance_base_currency = os.getenv("FINANCE_BASE_CURRENCY", "USD")

    @staticmethod
    def _rss_mib() -> Optional[float]:
        """
        RSS aproximado do processo (MiB). Usado para restart limpo antes do OOM killer.
        """
        try:
            # Linux: /proc/self/status contém "VmRSS:   123456 kB"
            with open("/proc/self/status", "r", encoding="utf-8", errors="ignore") as fh:
                for ln in fh:
                    if ln.startswith("VmRSS:"):
                        parts = ln.split()
                        if len(parts) >= 2:
                            kb = float(parts[1])
                            return kb / 1024.0
        except Exception:
            return None
        return None

    async def _ensure_runtime_events_table(self, db: Database) -> None:
        try:
            async with db.engine.begin() as conn:
                await conn.execute(
                    text(
                        f"""
                        CREATE TABLE IF NOT EXISTS {RUNTIME_EVENTS_TABLE} (
                          id SERIAL PRIMARY KEY,
                          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                          component TEXT NOT NULL,
                          kind TEXT NOT NULL,
                          level TEXT NOT NULL,
                          message TEXT NOT NULL,
                          meta JSONB
                        );
                        """
                    )
                )
                await conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{RUNTIME_EVENTS_TABLE}_created_at ON {RUNTIME_EVENTS_TABLE}(created_at);"))
                await conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{RUNTIME_EVENTS_TABLE}_kind ON {RUNTIME_EVENTS_TABLE}(kind);"))
        except Exception:
            return

    async def _emit_runtime_event(
        self,
        *,
        kind: str,
        level: str,
        message: str,
        meta: Optional[dict] = None,
        min_interval_sec: float = 60.0,
        try_db: bool = True,
    ) -> None:
        """
        Evento operacional (observabilidade): sempre grava JSONL; opcionalmente também grava no DB.
        Rate-limited por (kind, message) para evitar flood durante loops de retry.
        """
        try:
            ksig = f"{str(kind)}|{str(message)}"
            now = time.time()
            last = float(self._runtime_last_emit.get(ksig, 0.0) or 0.0)
            if last > 0 and (now - last) < float(min_interval_sec):
                return
            self._runtime_last_emit[ksig] = float(now)
        except Exception:
            pass

        payload = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "component": "audit_h3b_api",
            "kind": str(kind),
            "level": str(level),
            "message": str(message),
            "meta": meta or {},
            "mode": str(self.mode),
            "api_sides": str(self.api_sides),
            "pid": os.getpid(),
        }
        self._append_jsonl(self.runtime_events_file, payload)

        if not try_db:
            return

        # DB: tenta usar conexão já aberta; se ainda não existe (ex.: falha no login),
        # abre uma conexão curta (rate-limited acima).
        try:
            db = self.db
            close_after = False
            if db is None and self.save_to_db:
                db = Database()
                await db.connect()
                close_after = True
            if db is None:
                return
            await self._ensure_runtime_events_table(db)
            async with db.async_session() as session:
                await session.execute(
                    text(
                        f"""
                        INSERT INTO {RUNTIME_EVENTS_TABLE} (component, kind, level, message, meta)
                        VALUES (:component, :kind, :level, :message, (:meta)::jsonb)
                        """
                    ),
                    {
                        "component": "audit_h3b_api",
                        "kind": str(kind),
                        "level": str(level),
                        "message": str(message),
                        "meta": json.dumps(payload, ensure_ascii=False),
                    },
                )
                await session.commit()
            if close_after:
                try:
                    await db.close()
                except Exception:
                    pass
        except Exception:
            return

    @staticmethod
    def _parse_offsets(raw: str, *, default: List[float]) -> List[float]:
        s = (raw or "").strip()
        if not s:
            return list(default)
        out: List[float] = []
        for part in s.replace(";", ",").split(","):
            p = part.strip()
            if not p:
                continue
            try:
                out.append(float(p))
            except Exception:
                continue
        if not out:
            return list(default)
        # normaliza: >=0, ordenado, únicos, limita a 60s para segurança
        cleaned = sorted({round(x, 3) for x in out if x >= 0.0 and x <= 60.0})
        return cleaned or list(default)

    def _ws_state_key(self, event_id: str, market_type: str, market_period: str, line: str) -> Tuple[str, str, str, str]:
        return (str(event_id), str(market_type), str(market_period), str(line))

    @staticmethod
    def _ws_series_get(series: List[dict], target_s: float) -> Optional[float]:
        """
        Retorna a ws_odd do ponto com t_target_s == target_s (tolerância pequena).
        """
        try:
            tgt = float(target_s)
        except Exception:
            return None
        for p in series or []:
            try:
                if abs(float(p.get("t_target_s", -9999.0)) - tgt) <= 0.15:
                    v = p.get("ws_odd")
                    return float(v) if isinstance(v, (int, float)) and float(v) > 0 else None
            except Exception:
                continue
        return None

    async def _gate_try_acquire_open_slot(self) -> Tuple[bool, dict]:
        """
        Aplica cap de aberturas de ticket (POST /v1/betslips/) em janela deslizante.
        Retorna (ok, meta).
        """
        now = time.time()
        async with self.gate_open_lock:
            # prune
            w = float(self.gate_open_window_sec)
            while self.gate_open_times and (now - float(self.gate_open_times[0])) > w:
                try:
                    self.gate_open_times.popleft()
                except Exception:
                    break
            cnt = len(self.gate_open_times)
            if self.gate_open_max <= 0:
                return False, {"cap_enabled": False, "count_window": cnt, "window_sec": w, "max": self.gate_open_max}
            if cnt >= int(self.gate_open_max):
                return False, {"cap_enabled": True, "count_window": cnt, "window_sec": w, "max": int(self.gate_open_max)}
            self.gate_open_times.append(now)
            return True, {"cap_enabled": True, "count_window": cnt + 1, "window_sec": w, "max": int(self.gate_open_max)}

    def _ws_get_snapshot(self, key: Tuple[str, str, str, str]) -> Optional[dict]:
        snap = self._ws_odds_state.get(key)
        if not snap:
            return None
        # retorna cópia pequena (evita mutações acidentais)
        return {
            "ts": float(snap.get("ts") or 0.0),
            "side_a_name": snap.get("side_a_name"),
            "side_b_name": snap.get("side_b_name"),
            "side_a_odd": snap.get("side_a_odd"),
            "side_b_odd": snap.get("side_b_odd"),
        }

    def _ws_get_side_odd(self, key: Tuple[str, str, str, str], side: str) -> Optional[float]:
        snap = self._ws_get_snapshot(key)
        if not snap:
            return None
        side = str(side or "").strip().lower()
        if side and side == str(snap.get("side_a_name") or "").lower():
            try:
                return float(snap.get("side_a_odd"))
            except Exception:
                return None
        if side and side == str(snap.get("side_b_name") or "").lower():
            try:
                return float(snap.get("side_b_odd"))
            except Exception:
                return None
        return None

    async def shutdown(self, reason: str = ""):
        """
        Desligamento gracioso: cancela tasks, fecha browser/DB.
        Ajuda a evitar stop timeout e processos chrome órfãos no systemd.
        """
        async with self._shutdown_lock:
            if not self.running and not self._tasks:
                return
            self.running = False
            if reason:
                logger.warning(f"[SHUTDOWN] Encerrando (reason={reason})")
            else:
                logger.warning("[SHUTDOWN] Encerrando")

            tasks = list(self._tasks)
            self._tasks = []
            for t in tasks:
                try:
                    t.cancel()
                except Exception:
                    pass
            if tasks:
                try:
                    await asyncio.gather(*tasks, return_exceptions=True)
                except Exception:
                    pass

            if self.scraper:
                try:
                    await self.scraper.close()
                except Exception as e:
                    logger.warning(f"[SHUTDOWN] Falha ao fechar browser: {e}")
            if self.db:
                try:
                    await self.db.close()
                except Exception as e:
                    logger.warning(f"[SHUTDOWN] Falha ao fechar DB: {e}")

    async def _force_relogin(self, reason: str):
        """
        Força um novo login quando a API começa a retornar 401/auth_error.
        Usa lock + cooldown para evitar loop de relogin.
        """
        if not self.scraper:
            return
        now = time.time()
        async with self._relogin_lock:
            if self._last_relogin_ts and (now - self._last_relogin_ts) < 60.0:
                return
            self._last_relogin_ts = now
            try:
                logger.warning(f"[AUTH] Forçando relogin (reason={reason})")
                await self.scraper.login(force=True)
                # Reabre football para reativar WS e reduzir chance de contexto inválido
                try:
                    page = self.scraper._page
                    await page.goto(FOOTBALL_URL)
                    await page.wait_for_load_state("domcontentloaded")
                    await page.wait_for_timeout(3000)
                except Exception as e:
                    logger.warning(f"[AUTH] Falha ao reabrir football após relogin: {e}")
            except Exception as e:
                logger.error(f"[AUTH] Relogin falhou: {e}")

    @staticmethod
    def _parse_env_float(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None:
            return float(default)
        try:
            return float(raw)
        except Exception:
            return float(default)

    @staticmethod
    def _avg(values: List[float]) -> float:
        return (sum(values) / len(values)) if values else 0.0

    @staticmethod
    def _safe_num(value, default: float = 0.0) -> float:
        try:
            if value is None:
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    def _build_finance_snapshot(self, r: dict) -> Optional[dict]:
        """
        Monta insumos financeiros por auditoria.
        Objetivo: permitir análises futuras de turnover, lucro, ROI e drawdown
        com stake baseada em % do limite disponível.
        """
        # Guard rails: evita quebrar o save caso o processo esteja com versão antiga/hot-restart
        # onde os atributos financeiros ainda não existam.
        stake_pct = max(0.0, self._safe_num(getattr(self, "finance_stake_pct_of_limit", 0.25), 0.25))
        stake_cap = max(0.0, self._safe_num(getattr(self, "finance_stake_cap", 0.0), 0.0))
        fx_brl = max(0.0, self._safe_num(getattr(self, "finance_fx_brl", 5.20), 5.20))

        def _stake_from_limit(limit_value: float) -> float:
            raw_stake = max(0.0, limit_value) * stake_pct
            if stake_cap > 0:
                return min(raw_stake, stake_cap)
            return raw_stake

        bs_odd = self._safe_num(r.get("bs_odd"), 0.0)
        bs_limit = self._safe_num(r.get("bs_limit"), 0.0)
        lay_odd = self._safe_num(r.get("lay_odd"), 0.0)
        lay_limit = self._safe_num(r.get("lay_limit"), 0.0)

        back_stake = _stake_from_limit(bs_limit)
        lay_stake = _stake_from_limit(lay_limit)

        back_profit_if_win = back_stake * max(0.0, bs_odd - 1.0) if (back_stake > 0 and bs_odd > 1.0) else 0.0
        back_loss_if_lose = -back_stake if back_stake > 0 else 0.0

        lay_liability = lay_stake * max(0.0, lay_odd - 1.0) if (lay_stake > 0 and lay_odd > 1.0) else 0.0
        lay_profit_if_win = lay_stake if lay_stake > 0 else 0.0
        lay_loss_if_lose = -lay_liability if lay_liability > 0 else 0.0

        if back_stake <= 0 and lay_stake <= 0:
            return None

        finance = {
            "policy": {
                "stake_pct_of_limit": stake_pct,
                "stake_cap": stake_cap,
                "base_currency": getattr(self, "finance_base_currency", "USD"),
                "fx_brl": fx_brl,
            },
            "back": {
                "available_limit": bs_limit,
                "odd": bs_odd,
                "suggested_stake": back_stake,
                "profit_if_win": back_profit_if_win,
                "loss_if_lose": back_loss_if_lose,
            },
            "lay": {
                "available_limit": lay_limit,
                "odd": lay_odd,
                "suggested_stake": lay_stake,
                "liability_if_lose": lay_liability,
                "profit_if_win": lay_profit_if_win,
                "loss_if_lose": lay_loss_if_lose,
            },
            "brl_preview": {
                "back_suggested_stake_brl": back_stake * fx_brl,
                "back_profit_if_win_brl": back_profit_if_win * fx_brl,
                "back_loss_if_lose_brl": back_loss_if_lose * fx_brl,
                "lay_suggested_stake_brl": lay_stake * fx_brl,
                "lay_liability_if_lose_brl": lay_liability * fx_brl,
                "lay_profit_if_win_brl": lay_profit_if_win * fx_brl,
                "lay_loss_if_lose_brl": lay_loss_if_lose * fx_brl,
            },
        }
        return finance

    def _append_jsonl(self, path: str, payload: dict):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug(f"Falha ao gravar telemetria em {path}: {e}")

    def _emit_audit_telemetry(self, result: dict):
        telemetry = result.get('telemetry') or {}
        if not telemetry:
            return

        payload = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "event_id": result.get("event_id"),
            # status simples para dashboards rápidos
            "status": "OK" if result.get("success") else "FAIL",
            # status detalhado do pipeline (ex.: GATE_BLOCKED_CAP, API_RATE_LIMIT, OK, etc.)
            "status_code": result.get("status"),
            "audit_version": result.get("audit_version"),
            "market_type": result.get("market_type"),
            "line": result.get("line"),
            "side": result.get("side"),
            "is_live": result.get("is_live"),
            "ws_odd": result.get("ws_odd"),
            "bs_odd": result.get("bs_odd"),
            "lay_odd": result.get("lay_odd"),
            "diff_pct": result.get("diff_pct"),
            "error": result.get("error"),
            "telemetry": telemetry,
        }
        self._append_jsonl(self.telemetry_file, payload)

    async def start(self):
        logger.info("=" * 60)
        if self.mode in ("ws_gate_lay", "gate_lay", "gate"):
            logger.info("MONITOR H3B (WS gate t0/t+5) + abre LAY (Exchange) sob cap")
        elif self.mode in ("ws_reversal_lay", "reversal_lay"):
            logger.info("MONITOR H3B (reversal) + abre LAY imediatamente (Exchange) sob cap")
        elif self.mode in ("ws_gate_back", "gate_back"):
            logger.info("MONITOR H3B (WS gate t0/t+5) + marca BACK válido (sem abrir betslip)")
        elif self.mode in ("ws_only", "ws"):
            logger.info("MONITOR H3B (WS-only) + série temporal (t0..t+30)")
        elif self.mode in ("ws_vs_bs", "wsbs", "ws_vs_betslip"):
            logger.info("AUDITORIA H3B (WS vs BS no mesmo timestamp)")
        else:
            logger.info("AUDITORIA H3B VIA API (~2-3s)")
        logger.info(f"mode={self.mode} ws_offsets={self.ws_sample_offsets_sec}")
        if self.mode in ("ws_gate_lay", "gate_lay", "gate"):
            logger.info(
                f"gate: offset={self.gate_drop_offset_sec:.1f}s ratio={self.gate_drop_ratio:.3f} | "
                f"cap={self.gate_open_max}/{self.gate_open_window_sec}s | "
                f"lay_refresh={self.gate_lay_refresh}"
            )
        if self.mode in ("ws_reversal_lay", "reversal_lay"):
            logger.info(
                f"reversal_lay: cap={self.gate_open_max}/{self.gate_open_window_sec}s | "
                f"max_late={self.gate_max_late_sec:.1f}s | lay_refresh={self.gate_lay_refresh}"
            )
        if self.mode in ("ws_gate_back", "gate_back"):
            logger.info(
                f"gate_back: offset={self.gate_rise_offset_sec:.1f}s ratio={self.gate_rise_ratio:.3f} | "
                f"max_late={self.gate_max_late_sec:.1f}s"
            )
        logger.info("=" * 60)

        signal.signal(signal.SIGTERM, lambda s, f: setattr(self, 'running', False))
        signal.signal(signal.SIGINT, lambda s, f: setattr(self, 'running', False))

        # Browser (necessário para WS e fetch autenticado)
        self.scraper = BetinAsiaScraper()
        try:
            await self.scraper.start()
        except Exception as e:
            await self._emit_runtime_event(kind="START_BROWSER_FAIL", level="ERROR", message=str(e)[:220], meta={}, min_interval_sec=120.0)
            raise
        ok_login = False
        try:
            ok_login = await self.scraper.login()
        except Exception as e:
            await self._emit_runtime_event(kind="LOGIN_EXCEPTION", level="ERROR", message=str(e)[:220], meta={}, min_interval_sec=120.0)
            ok_login = False
        if not ok_login:
            logger.error("Login falhou. Abortando (necessário para WS + endpoints /v1/betslips/ quando usado).")
            try:
                url = self.scraper._page.url if (self.scraper and getattr(self.scraper, "_page", None)) else ""
            except Exception:
                url = ""
            await self._emit_runtime_event(
                kind="LOGIN_FAIL",
                level="ERROR",
                message="login_failed",
                meta={"url": url},
                min_interval_sec=180.0,
                try_db=True,
            )
            try:
                await self.scraper.close()
            except Exception:
                pass
            self.scraper = None
            return False
        logger.info("Login OK")

        # API client (usa o page do scraper)
        page = self.scraper._page
        self.api_client = ApiBetslipClient(page)

        # Reduz carga/memória: bloqueia media/fonts (não necessário para WS + fetch).
        try:
            async def _route_handler(route, request):
                try:
                    rt = str(getattr(request, "resource_type", "") or "")
                    if rt in ("image", "media", "font"):
                        await route.abort()
                        return
                except Exception:
                    pass
                try:
                    await route.continue_()
                except Exception:
                    try:
                        await route.fallback()
                    except Exception:
                        pass

            await page.route("**/*", _route_handler)
        except Exception:
            pass

        # WS listener único (para odds + PMM + betslip)
        def on_ws(ws):
            def on_frame(data):
                # Playwright pode entregar `str` ou `bytes` no framereceived; `str(bytes)` vira "b'...'"
                # e quebra o json.loads, causando perda de PMMs mesmo com WS ativo.
                try:
                    if isinstance(data, bytes):
                        data_str = data.decode("utf-8", errors="ignore")
                    else:
                        data_str = str(data)
                except Exception:
                    data_str = ""
                if not data_str:
                    return
                try:
                    self._ws_messages.append(data_str)
                    mx = int(self._ws_messages_max or 0)
                    if mx > 0:
                        while len(self._ws_messages) > mx:
                            self._ws_messages.popleft()
                            self._ws_messages_dropped += 1
                except Exception:
                    pass
                self._last_ws_time = time.time()
                self._ws_msg_count += 1
                
                # Também processa PMM/betslip para o API client
                try:
                    msg = json.loads(data_str)
                    if isinstance(msg, list):
                        for item in msg:
                            if isinstance(item, list) and len(item) >= 2:
                                if item[0] == 'api' and isinstance(item[1], dict):
                                    for entry in item[1].get('data', []):
                                        if isinstance(entry, list) and len(entry) >= 2:
                                            if entry[0] == 'pmm':
                                                self.api_client._handle_pmm(entry[1])
                                            elif entry[0] == 'betslip':
                                                self.api_client._handle_betslip(entry[1])
                except:
                    pass
            ws.on('framereceived', on_frame)
        page.on('websocket', on_ws)

        # Navega para football (ativa WS)
        await page.goto(FOOTBALL_URL)
        await page.wait_for_load_state("domcontentloaded")
        logger.info("Aguardando WebSocket...")
        await page.wait_for_timeout(5000)
        self._start_time = time.time()
        logger.info(f"WS: {self._ws_msg_count} msgs recebidas")
        try:
            if int(self._ws_msg_count or 0) <= 0:
                await self._emit_runtime_event(
                    kind="WS_NO_MESSAGES",
                    level="WARN",
                    message="ws_msg_count_zero_after_start",
                    meta={},
                    min_interval_sec=300.0,
                    try_db=True,
                )
        except Exception:
            pass

        # DB
        if self.save_to_db:
            try:
                self.db = Database()
                await self.db.connect()
            except Exception as e:
                await self._emit_runtime_event(kind="DB_CONNECT_FAIL", level="ERROR", message=str(e)[:220], meta={}, min_interval_sec=180.0, try_db=False)
                raise
            try:
                async with self.db.engine.begin() as conn:
                    await conn.execute(text(
                        "ALTER TABLE betslip_audit_results ADD COLUMN IF NOT EXISTS is_live BOOLEAN"))
            except:
                pass
            logger.info("Banco conectado")

        return True

    async def run(self):
        # Se o login falhar (proxy/bloqueio/captcha), não derrubamos o processo
        # para não bater StartLimit do systemd. Re-tenta com backoff.
        backoff_s = 15.0
        while self.running:
            try:
                ok = await self.start()
            except Exception as e:
                logger.error(f"Falha na inicialização: {e}")
                try:
                    await self._emit_runtime_event(
                        kind="START_EXCEPTION",
                        level="ERROR",
                        message=str(e)[:220],
                        meta={},
                        min_interval_sec=120.0,
                        try_db=True,
                    )
                except Exception:
                    pass
                ok = False
            if ok:
                break
            try:
                await self._emit_runtime_event(
                    kind="START_RETRY",
                    level="WARN",
                    message="start_failed_retrying",
                    meta={"backoff_s": float(backoff_s)},
                    min_interval_sec=120.0,
                    try_db=True,
                )
            except Exception:
                pass
            logger.warning(f"Start falhou (provável login/bloqueio). Re-tentando em {int(backoff_s)}s...")
            await asyncio.sleep(backoff_s)
            backoff_s = min(300.0, backoff_s * 1.5)
        if not self.running:
            return

        audit_queue = asyncio.Queue(maxsize=self.max_queue_depth) if self.max_queue_depth > 0 else asyncio.Queue()
        self._queue_ref = audit_queue
        temporal_queue = asyncio.Queue()
        self._temporal_queue_ref = temporal_queue

        tasks = [asyncio.create_task(self._monitor_loop(audit_queue))]
        for wid in range(1, self.executor_workers + 1):
            tasks.append(asyncio.create_task(self._executor_loop(audit_queue, worker_id=wid)))
        for twid in range(1, self.temporal_workers + 1):
            tasks.append(asyncio.create_task(self._temporal_loop(temporal_queue, worker_id=twid)))
        tasks.append(asyncio.create_task(self._maintenance_loop()))
        self._tasks = list(tasks)
        logger.info(
            f"Executores T+0 ativos: {self.executor_workers} | Temporal workers: {self.temporal_workers} | "
            f"max_queue_depth={self.max_queue_depth or 'inf'} | max_queue_wait_ms={self.max_queue_wait_ms or 'inf'}"
        )

        try:
            while self.running:
                if self.num_audits > 0 and len(self.results) >= self.num_audits:
                    break
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await self.shutdown("run_finally")

        self._print_summary()
        # fechamento já tratado em shutdown()

    # ================================================================
    # MONITOR
    # ================================================================
    async def _monitor_loop(self, queue: asyncio.Queue):
        logger.info("Monitor iniciado")
        last_seen = 0  # contador total já visto (aproximação quando usamos deque)

        while self.running:
            # Como usamos deque com trim, consumimos por pop-left.
            new: List[str] = []
            try:
                while self._ws_messages:
                    new.append(self._ws_messages.popleft())
            except Exception:
                new = []
            last_seen += len(new)

            if not new:
                await asyncio.sleep(0.05)
                continue

            for msg in new:
                try:
                    data = json.loads(msg)
                    if not isinstance(data, list):
                        continue
                    for item in data:
                        if not isinstance(item, list) or len(item) < 2:
                            continue
                        msg_type, msg_meta = item[0], item[1]
                        msg_data = item[2] if len(item) > 2 else {}

                        # Event info
                        if msg_type == 'event' and isinstance(msg_meta, list) and len(msg_meta) >= 2:
                            if msg_meta[0] == 'fb' and 'home' in msg_data:
                                eid = msg_meta[1]
                                kickoff = None
                                if 'start_ts' in msg_data:
                                    try:
                                        kickoff = datetime.fromisoformat(
                                            msg_data['start_ts'].replace('Z', '+00:00'))
                                    except: pass
                                self._events_info[eid] = {
                                    'home': msg_data.get('home', ''),
                                    'away': msg_data.get('away', ''),
                                    'league': msg_data.get('competition_name', ''),
                                    'kickoff': kickoff,
                                }

                        # Odds
                        if msg_type in ['offers_hcap', 'offers_event']:
                            if isinstance(msg_meta, list) and len(msg_meta) >= 3 and msg_meta[1] == 'fb':
                                eid = msg_meta[2]
                                if 'ah' in msg_data:
                                    self._process_odds(eid, msg_data['ah'], 'AH', 'full_time', queue)
                                if 'ahou' in msg_data:
                                    self._process_odds(eid, msg_data['ahou'], 'OU', 'full_time', queue, over_under=True)
                                if 'ah_ht' in msg_data:
                                    self._process_odds(eid, msg_data['ah_ht'], 'AH', 'half_time', queue)
                                if 'ou_ht' in msg_data:
                                    self._process_odds(eid, msg_data['ou_ht'], 'OU', 'half_time', queue, over_under=True)
                except:
                    continue

            if self.events_processed > 0 and self.events_processed % 500 == 0:
                logger.info(f"Processados: {self.events_processed} | H3B: {self.h3b_detected} | "
                            f"Auditados: {len(self.results)} | WS: {self._ws_msg_count}")

    def _process_odds(self, event_id, odds_data, market_type, period, queue, over_under=False):
        lines = []
        if isinstance(odds_data, list) and len(odds_data) >= 2:
            if isinstance(odds_data[0], (int, float)):
                lines = [odds_data]
            elif isinstance(odds_data[0], list):
                lines = odds_data

        hk = 'o' if over_under else 'h'
        ak = 'u' if over_under else 'a'

        for line_data in lines:
            if len(line_data) < 2:
                continue
            line_val = line_data[0]
            odds_list = line_data[1] if len(line_data) > 1 else []

            home_odds = away_odds = 0
            if isinstance(odds_list, list):
                for o in odds_list:
                    if isinstance(o, list) and len(o) >= 2:
                        if o[0] == hk: home_odds = float(o[1])
                        elif o[0] == ak: away_odds = float(o[1])

            if home_odds <= 0 or away_odds <= 0:
                continue
            self.events_processed += 1

            # Atualiza estado WS (para amostragem temporal WS-only e auditoria WS vs BS)
            try:
                key = self._ws_state_key(event_id, market_type, period, str(line_val))
                if over_under:
                    side_a_name, side_b_name = "over", "under"
                else:
                    side_a_name, side_b_name = "home", "away"
                self._ws_odds_state[key] = {
                    "ts": time.time(),
                    "side_a_name": side_a_name,
                    "side_b_name": side_b_name,
                    "side_a_odd": float(home_odds),
                    "side_b_odd": float(away_odds),
                }
            except Exception:
                pass

            try:
                if abs(float(line_val)) > MAX_AH_LINE:
                    continue
            except:
                pass

            # Filtra jogos acabados
            info = self._events_info.get(event_id, {})
            kickoff = info.get('kickoff')
            if kickoff:
                now = datetime.now(timezone.utc)
                if (now - kickoff).total_seconds() > 9000:
                    continue

            det = self.detector.process_market_update(
                match_id=hash(event_id) % 1000000,
                market_type=f"{market_type}{'_HT' if period == 'half_time' else ''}",
                line=str(line_val),
                home_odd=home_odds,
                away_odd=away_odds,
            )

            for h3b in det.get("h3b_events", []):
                self.h3b_detected += 1
                if self.direction != "all" and h3b.direction_after != self.direction:
                    continue

                audit_key = f"{event_id}|{market_type}|{period}|{h3b.ah_line}|{h3b.side}"
                # Dedup com TTL
                try:
                    now = time.time()
                except Exception:
                    now = 0.0
                try:
                    ts = float(self._audited_ts.get(audit_key, 0.0) or 0.0)
                    if ts > 0 and (now - ts) <= float(self._audited_ttl_sec):
                        continue
                except Exception:
                    pass

                is_live = kickoff <= datetime.now(timezone.utc) if kickoff else None

                queue_depth_at_enqueue = queue.qsize()
                if self.max_queue_depth > 0 and queue_depth_at_enqueue >= self.max_queue_depth:
                    self.dropped_full_queue += 1
                    continue
                try:
                    queue.put_nowait({
                        'event_id': event_id,
                        'audit_key': audit_key,
                        'home_team': info.get('home', '?'),
                        'away_team': info.get('away', '?'),
                        'league': info.get('league', ''),
                        'kickoff': kickoff,
                        'is_live': is_live,
                        'market_type': market_type,
                        'market_period': period,
                        'line': str(h3b.ah_line),
                        'side': h3b.side,
                        'websocket_odd': h3b.odd_at_reversal,
                        'ws_state_key': self._ws_state_key(event_id, market_type, period, str(h3b.ah_line)),
                        'direction': h3b.direction_after,
                        'detected_at': time.time(),
                        'queue_depth_at_enqueue': queue_depth_at_enqueue,
                    })
                except asyncio.QueueFull:
                    self.dropped_full_queue += 1
                    continue
                self.max_queue_depth_observed = max(self.max_queue_depth_observed, queue.qsize())
                try:
                    self._audited_ts[audit_key] = float(time.time())
                except Exception:
                    pass

                # GC dedup map
                try:
                    if self._audited_ttl_sec > 0 and len(self._audited_ts) > 200000:
                        cutoff = time.time() - float(self._audited_ttl_sec)
                        self._audited_ts = {k: v for k, v in self._audited_ts.items() if float(v) >= cutoff}
                except Exception:
                    pass

    # ================================================================
    # EXECUTOR (via API, não DOM)
    # ================================================================
    async def _executor_loop(self, queue: asyncio.Queue, worker_id: int = 1):
        logger.info(f"Executor API iniciado (worker={worker_id})")

        while self.running:
            if self.num_audits > 0 and len(self.results) >= self.num_audits:
                break
            try:
                h3b = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            h3b['dequeued_at'] = time.time()
            h3b['queue_depth_after_dequeue'] = queue.qsize()
            defer_temporal = self.save_to_db and self.temporal_workers > 0
            if self.mode in ("ws_gate_lay", "gate_lay", "gate"):
                # Gate precisa de WS(t0,t+5) inline para decidir; não pode ser deferred.
                result = await self._execute_ws_gate_lay(h3b, defer_temporal=defer_temporal)
            elif self.mode in ("ws_reversal_lay", "reversal_lay"):
                result = await self._execute_ws_reversal_lay(h3b, defer_temporal=defer_temporal)
            elif self.mode in ("ws_gate_back", "gate_back"):
                # Gate Back decide via WS(t0,t+5) (sem abrir betslip).
                result = await self._execute_ws_gate_back(h3b, defer_temporal=defer_temporal)
            elif self.mode in ("ws_only", "ws"):
                result = await self._execute_ws_only(h3b, run_ws_series=not defer_temporal)
            elif self.mode in ("ws_vs_bs", "wsbs", "ws_vs_betslip"):
                result = await self._execute_api_audit(h3b, run_temporal=not defer_temporal)
            else:
                result = await self._execute_api_audit(h3b, run_temporal=not defer_temporal)
            telemetry = result.setdefault('telemetry', {})
            telemetry['worker_id'] = worker_id
            telemetry['pipeline_total_ms_pre_db'] = int((time.time() - h3b['detected_at']) * 1000)
            telemetry['executor_total_ms_pre_db'] = int((time.time() - h3b['dequeued_at']) * 1000)
            db_t0 = time.time()
            record_id = None
            if self.save_to_db:
                record_id = await self._save_result(result)
            telemetry['db_save_ms'] = int((time.time() - db_t0) * 1000) if self.save_to_db else 0
            telemetry['pipeline_total_ms'] = int((time.time() - h3b['detected_at']) * 1000)
            telemetry['executor_total_ms'] = int((time.time() - h3b['dequeued_at']) * 1000)
            self._emit_audit_telemetry(result)
            try:
                self.results.append(result)
                mxr = int(self._results_max or 0)
                if mxr > 0:
                    while len(self.results) > mxr:
                        self.results.popleft()
            except Exception:
                pass

            temporal_refs = result.get('_temporal_refs')
            ws_refs = result.get('_ws_series_refs')
            if defer_temporal and record_id and self._temporal_queue_ref:
                job: Optional[dict] = None
                if temporal_refs:
                    job = {
                        'kind': 'betslip_temporal',
                        'record_id': record_id,
                        'event_id': result.get('event_id'),
                        'home_team': result.get('home_team'),
                        'away_team': result.get('away_team'),
                        'ws_odd': temporal_refs.get('ws_odd'),
                        'ws_state_key': temporal_refs.get('ws_state_key'),
                        'ws_side': temporal_refs.get('ws_side'),
                        'refresh_times': temporal_refs.get('refresh_times'),
                        'back_betslip_id': temporal_refs.get('back_betslip_id', ''),
                        'lay_betslip_id': temporal_refs.get('lay_betslip_id', ''),
                        'telemetry_base': dict(telemetry),
                        'queued_at': time.time(),
                    }
                elif ws_refs:
                    job = {
                        'kind': 'ws_series',
                        'record_id': record_id,
                        'event_id': result.get('event_id'),
                        'home_team': result.get('home_team'),
                        'away_team': result.get('away_team'),
                        'ws_state_key': ws_refs.get('ws_state_key'),
                        'ws_side': ws_refs.get('ws_side'),
                        'offsets_sec': ws_refs.get('offsets_sec'),
                        'telemetry_base': dict(telemetry),
                        'queued_at': time.time(),
                    }
                if job:
                    self._temporal_queue_ref.put_nowait(job)
                    self.max_temporal_queue_depth_observed = max(
                        self.max_temporal_queue_depth_observed,
                        self._temporal_queue_ref.qsize()
                    )

            # Log
            live = "LIVE" if result.get('is_live') else "PRE" if result.get('is_live') is not None else "?"
            bs_odd = result.get('bs_odd', None)
            has_bs = isinstance(bs_odd, (int, float)) and float(bs_odd) > 0
            if result.get('success') and has_bs:
                self.consecutive_errors = 0
                lay_str = ""
                if result.get('lay_odd'):
                    lay_str = f" lay={result['lay_odd']:.3f}({result.get('lay_bookie','')})"
                q_ms = telemetry.get('queue_wait_ms', 0)
                temp_ms = telemetry.get('temporal_total_ms', 0)
                temp_part = "deferred" if telemetry.get('temporal_deferred') else f"{temp_ms}ms"
                logger.info(
                    f"[OK][{live}] {result['home_team']} vs {result['away_team']} | "
                    f"{result['market_type']} {result['line']} {result['side']} | "
                    f"ws={result['ws_odd']:.3f} bs={result['bs_odd']:.3f} "
                    f"diff={result['diff_pct']:+.2f}% lim=${result['bs_limit']:,.0f} "
                    f"({result['num_bk']} bk){lay_str} | "
                    f"lag={result['total_ms']}ms q={q_ms}ms temp={temp_part} w={worker_id} | "
                    f"{len(self.results)}")
            elif result.get('success') and (not has_bs):
                self.consecutive_errors = 0
                q_ms = telemetry.get('queue_wait_ms', 0)
                total_ms = result.get('total_ms')
                ver = str(result.get("audit_version") or "")
                ws_part = "ws_series=inline" if result.get("ws_series") else "ws_series=deferred" if result.get("_ws_series_refs") else "ws_series=none"
                logger.info(
                    f"[WS][{live}] {result['home_team']} vs {result['away_team']} | "
                    f"{result['market_type']} {result['line']} {result['side']} | "
                    f"ws={result['ws_odd']:.3f} | {ws_part} ver={ver} | "
                    f"lag={total_ms}ms q={q_ms}ms w={worker_id} | {len(self.results)}"
                )
            else:
                status = result.get("status", "FAIL")
                if status == "STALE_QUEUE_WAIT":
                    q_ms = telemetry.get('queue_wait_ms', 0)
                    logger.info(
                        f"[STALE][{live}] {result['home_team']} vs {result['away_team']} | "
                        f"{result['market_type']} {result['line']} {result['side']} | "
                        f"ws={result['ws_odd']:.3f} | q={q_ms}ms | "
                        f"lag={result['total_ms']}ms | {len(self.results)}"
                    )
                else:
                    self.total_errors += 1
                    logger.warning(
                        f"[FAIL][{live}] {result['home_team']} vs {result['away_team']} | "
                        f"{result['market_type']} {result['line']} {result['side']} | "
                        f"ws={result['ws_odd']:.3f} | err={result.get('error','')} | "
                        f"lag={result['total_ms']}ms | {len(self.results)}")

            if len(self.results) % STATS_INTERVAL == 0:
                self._log_stats()

    async def _collect_temporal_series(
        self,
        ws_odd: float,
        back_betslip_id: str,
        lay_betslip_id: str,
        *,
        ws_state_key: Optional[Tuple[str, str, str, str]] = None,
        ws_side: str = "",
        refresh_times: Optional[List[float]] = None,
    ):
        refresh_times = [3, 6, 10, 15, 20] if (not refresh_times) else [float(x) for x in refresh_times if x is not None]
        back_temporal = []
        lay_temporal = []
        temporal_points = []
        temporal_refresh_durations = []
        temporal_wait_ms = 0

        def _extract_lay_snapshot(api_result: Optional[BetslipApiResult]) -> Optional[dict]:
            if not api_result or not api_result.success:
                return None
            lay_bookmakers = [b for b in api_result.bookmakers if b.best_price > 0]
            if not lay_bookmakers:
                return None
            best = min(lay_bookmakers, key=lambda b: b.best_price)
            return {
                'odd': best.best_price,
                'bookie': best.bookie,
                'limit': best.max_stake,
                'num_bk': len(lay_bookmakers),
            }

        temporal_start = time.time()
        if back_betslip_id or lay_betslip_id:
            t_start = time.time()
            for target_t in refresh_times:
                elapsed = time.time() - t_start
                wait = target_t - elapsed
                if wait > 0:
                    await asyncio.sleep(wait)
                    temporal_wait_ms += int(wait * 1000)

                labels = []
                refresh_calls = []
                if back_betslip_id:
                    labels.append("back")
                    refresh_calls.append(self.api_client.refresh_betslip(back_betslip_id))
                if lay_betslip_id:
                    labels.append("lay")
                    refresh_calls.append(self.api_client.refresh_betslip(lay_betslip_id))

                if not refresh_calls:
                    break

                refresh_t0 = time.time()
                refresh_results = await asyncio.gather(*refresh_calls, return_exceptions=True)
                refresh_ms = int((time.time() - refresh_t0) * 1000)
                temporal_refresh_durations.append(refresh_ms)
                actual_t = round(time.time() - t_start, 1)
                # Snapshot WS no momento do refresh (para comparar "mesmo timestamp")
                ws_now = ws_odd
                if ws_state_key:
                    cur = self._ws_get_side_odd(ws_state_key, ws_side)
                    if isinstance(cur, (int, float)) and float(cur) > 0:
                        ws_now = float(cur)
                point_meta = {
                    'target_s': target_t,
                    'actual_s': actual_t,
                    'refresh_ms': refresh_ms,
                    'back_ok': False,
                    'lay_ok': False,
                    'ws_odd': ws_now,
                }

                for label, ref in zip(labels, refresh_results):
                    if isinstance(ref, Exception):
                        logger.debug(f"Refresh {label} t+{target_t} falhou: {ref}")
                        continue
                    if not ref or not ref.success:
                        continue

                    if label == "back":
                        ref_diff = ((ref.best_odd - ws_now) / ws_now) * 100 if ws_now else 0
                        back_temporal.append({
                            't': actual_t,
                            'bs_odd': ref.best_odd,
                            'diff_pct': round(ref_diff, 3),
                            'bookie': ref.best_bookie,
                            'limit': ref.best_limit,
                            'num_bk': ref.num_bookmakers,
                            'ws_odd': ws_now,
                        })
                        point_meta['back_ok'] = True
                    else:
                        lay_ref = _extract_lay_snapshot(ref)
                        if lay_ref:
                            lay_diff = ((lay_ref['odd'] - ws_now) / ws_now) * 100 if ws_now else 0
                            lay_temporal.append({
                                't': actual_t,
                                'lay_odd': lay_ref['odd'],
                                'diff_pct': round(lay_diff, 3),
                                'bookie': lay_ref['bookie'],
                                'limit': lay_ref['limit'],
                                'num_bk': lay_ref['num_bk'],
                                'ws_odd': ws_now,
                            })
                            point_meta['lay_ok'] = True

                temporal_points.append(point_meta)

        temporal_total_ms = int((time.time() - temporal_start) * 1000) if (back_betslip_id or lay_betslip_id) else 0
        telemetry_patch = {
            'temporal_total_ms': temporal_total_ms,
            'temporal_wait_ms': temporal_wait_ms,
            'temporal_refresh_mean_ms': int(self._avg(temporal_refresh_durations)) if temporal_refresh_durations else 0,
            'temporal_points_back': len(back_temporal),
            'temporal_points_lay': len(lay_temporal),
            'temporal_points': temporal_points,
            'temporal_deferred': False,
        }
        # Cleanup: após finalizar refreshes, fecha betslips para evitar "too_many_open_betslips".
        try:
            close_calls = []
            if back_betslip_id:
                close_calls.append(asyncio.wait_for(self.api_client.close_betslip(back_betslip_id), timeout=1.2))
            if lay_betslip_id:
                close_calls.append(asyncio.wait_for(self.api_client.close_betslip(lay_betslip_id), timeout=1.2))
            if close_calls:
                await asyncio.gather(*close_calls, return_exceptions=True)
        except Exception:
            pass
        return back_temporal, lay_temporal, telemetry_patch

    async def _collect_ws_series(
        self,
        *,
        ws_state_key: Tuple[str, str, str, str],
        ws_side: str,
        offsets_sec: List[float],
    ) -> Tuple[List[dict], Dict[str, Any]]:
        """
        Coleta série de odds via WS em offsets específicos.
        Retorna (series, telemetry_patch).
        """
        offsets = [float(x) for x in offsets_sec if x is not None and float(x) >= 0.0]
        offsets = sorted({round(x, 3) for x in offsets})
        if not offsets:
            offsets = [0.0]

        t_start = time.time()
        series: List[dict] = []
        points_meta: List[dict] = []
        wait_ms = 0

        for target in offsets:
            elapsed = time.time() - t_start
            wait = float(target) - float(elapsed)
            if wait > 0:
                await asyncio.sleep(wait)
                wait_ms += int(wait * 1000)
            snap = self._ws_get_snapshot(ws_state_key)
            now_ts = time.time()
            actual = round(now_ts - t_start, 3)

            side_odd = self._ws_get_side_odd(ws_state_key, ws_side)
            point = {
                "t_target_s": float(target),
                "t_actual_s": float(actual),
                "ts": now_ts,
                "ws_side": ws_side,
                "ws_odd": side_odd,
                "ws_side_a_name": snap.get("side_a_name") if snap else None,
                "ws_side_b_name": snap.get("side_b_name") if snap else None,
                "ws_side_a_odd": snap.get("side_a_odd") if snap else None,
                "ws_side_b_odd": snap.get("side_b_odd") if snap else None,
                "ws_state_age_ms": int(max(0.0, (now_ts - float(snap.get("ts") or 0.0)) * 1000)) if snap else None,
                "ws_state_missing": bool(snap is None),
            }
            # série principal: sempre inclui o lado alvo + par (se houver)
            series.append(point)
            points_meta.append({"t_target_s": float(target), "t_actual_s": float(actual), "ws_ok": bool(side_odd)})

        telemetry_patch = {
            "ws_series_total_ms": int((time.time() - t_start) * 1000),
            "ws_series_wait_ms": int(wait_ms),
            "ws_series_points": points_meta,
            "ws_series_deferred": False,
        }
        return series, telemetry_patch

    async def _patch_ws_series_result(self, record_id: int, ws_series: list, telemetry: dict, meta: dict):
        if not self.db or not record_id:
            return
        patch = {"telemetry": telemetry, "ws_series": ws_series, "ws_series_meta": meta}
        async with self.db.async_session() as session:
            await session.execute(
                text(
                    """
                    UPDATE betslip_audit_results
                    SET hypothesis_details = (
                        COALESCE(hypothesis_details::jsonb, '{}'::jsonb) || CAST(:patch AS jsonb)
                    )::json
                    WHERE id = :id
                    """
                ),
                {"id": record_id, "patch": json.dumps(patch, ensure_ascii=False)},
            )
            await session.commit()

    async def _patch_temporal_result(self, record_id: int, back_temporal: list, lay_temporal: list, telemetry: dict):
        if not self.db or not record_id:
            return
        patch = {'telemetry': telemetry}
        if back_temporal:
            patch['temporal'] = back_temporal
        if lay_temporal:
            patch['lay_temporal'] = lay_temporal

        async with self.db.async_session() as session:
            await session.execute(
                text("""
                    UPDATE betslip_audit_results
                    SET hypothesis_details = (
                        COALESCE(hypothesis_details::jsonb, '{}'::jsonb) || CAST(:patch AS jsonb)
                    )::json
                    WHERE id = :id
                """),
                {"id": record_id, "patch": json.dumps(patch, ensure_ascii=False)},
            )
            await session.commit()

    async def _temporal_loop(self, queue: asyncio.Queue, worker_id: int = 1):
        logger.info(f"Temporal worker iniciado (worker={worker_id})")
        while self.running:
            try:
                job = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            try:
                kind = str(job.get("kind") or "betslip_temporal").strip().lower()
                if kind == "ws_series":
                    ws_state_key = job.get("ws_state_key")
                    ws_side = str(job.get("ws_side") or "")
                    offsets = job.get("offsets_sec") or self.ws_sample_offsets_sec
                    ws_series, telemetry_patch = await self._collect_ws_series(
                        ws_state_key=tuple(ws_state_key) if isinstance(ws_state_key, (list, tuple)) else ws_state_key,
                        ws_side=ws_side,
                        offsets_sec=[float(x) for x in offsets],
                    )
                    telemetry_final = dict(job.get("telemetry_base") or {})
                    telemetry_final.update(telemetry_patch)
                    telemetry_final["temporal_worker_id"] = worker_id
                    telemetry_final["temporal_async_latency_ms"] = int((time.time() - job.get("queued_at", time.time())) * 1000)
                    meta = {"offsets_sec": [float(x) for x in offsets], "ws_side": ws_side}
                    await self._patch_ws_series_result(
                        record_id=int(job.get("record_id") or 0),
                        ws_series=ws_series,
                        telemetry=telemetry_final,
                        meta=meta,
                    )
                    if ws_series:
                        logger.info(f"[WS_SERIES][w={worker_id}] id={job.get('record_id')} pts={len(ws_series)}")
                else:
                    back_temporal, lay_temporal, telemetry_patch = await self._collect_temporal_series(
                        ws_odd=job.get('ws_odd', 0) or 0,
                        back_betslip_id=job.get('back_betslip_id', ''),
                        lay_betslip_id=job.get('lay_betslip_id', ''),
                        ws_state_key=tuple(job.get("ws_state_key")) if isinstance(job.get("ws_state_key"), (list, tuple)) else None,
                        ws_side=str(job.get("ws_side") or ""),
                        refresh_times=job.get("refresh_times") if isinstance(job.get("refresh_times"), list) else None,
                    )
                    telemetry_final = dict(job.get('telemetry_base') or {})
                    telemetry_final.update(telemetry_patch)
                    telemetry_final['temporal_worker_id'] = worker_id
                    telemetry_final['temporal_async_latency_ms'] = int((time.time() - job.get('queued_at', time.time())) * 1000)
                    await self._patch_temporal_result(
                        record_id=job.get('record_id'),
                        back_temporal=back_temporal,
                        lay_temporal=lay_temporal,
                        telemetry=telemetry_final,
                    )

                    if back_temporal or lay_temporal:
                        logger.info(
                            f"[TEMPORAL][w={worker_id}] id={job.get('record_id')} "
                            f"back_pts={len(back_temporal)} lay_pts={len(lay_temporal)} "
                            f"ms={telemetry_patch.get('temporal_total_ms', 0)}"
                        )
            except Exception as e:
                logger.warning(f"[TEMPORAL][w={worker_id}] falha no processamento: {e}")
            finally:
                queue.task_done()

    async def _execute_api_audit(self, h3b: dict, run_temporal: bool = True) -> dict:
        detected_at = h3b['detected_at']
        execution_start = time.time()
        queue_wait_ms = int((execution_start - detected_at) * 1000)
        queue_depth_at_enqueue = h3b.get('queue_depth_at_enqueue')
        queue_depth_after_dequeue = h3b.get('queue_depth_after_dequeue')

        def _extract_lay_snapshot(api_result: Optional[BetslipApiResult]) -> Optional[dict]:
            if not api_result or not api_result.success:
                return None
            lay_bookmakers = [b for b in api_result.bookmakers if b.best_price > 0]
            if not lay_bookmakers:
                return None
            best = min(lay_bookmakers, key=lambda b: b.best_price)
            return {
                'odd': best.best_price,
                'bookie': best.bookie,
                'limit': best.max_stake,
                'num_bk': len(lay_bookmakers),
            }

        telemetry = {
            'queue_wait_ms': queue_wait_ms,
            'queue_depth_at_enqueue': queue_depth_at_enqueue,
            'queue_depth_after_dequeue': queue_depth_after_dequeue,
        }

        base = {
            'event_id': h3b['event_id'],
            'home_team': h3b['home_team'],
            'away_team': h3b['away_team'],
            'league': h3b.get('league', ''),
            'kickoff': h3b.get('kickoff'),
            'market_type': h3b['market_type'],
            'market_period': h3b.get('market_period', 'full_time'),
            'line': h3b['line'],
            'side': h3b['side'],
            'ws_odd': h3b['websocket_odd'],
            'is_live': h3b.get('is_live'),
            'direction': h3b.get('direction', 'up'),
            'detected_at': detected_at,
            'post_ms': 0,
            'pmm_ms': 0,
        }
        # Versionamento: evita misturar "api back-only" com o legado v4.0-api.
        if self.api_sides == "back":
            base["audit_version"] = "v5.2-api-back"
            # Hint para o bridge/analytics não misturar Back vs Lay.
            # (No modo API, o lado de execução é conhecido pelo serviço.)
            base["exec_side_hint"] = "Back"
        elif self.api_sides == "lay":
            base["audit_version"] = "v5.2-api-lay"
            base["exec_side_hint"] = "Lay"
        else:
            base["audit_version"] = "v4.0-api"

        # Backoff global: não chama API enquanto bloqueado (rate limit / instabilidade),
        # preserva a operação e reduz STALE por fila acumulada.
        now_ts = time.time()
        if self._api_backoff_until_ts and now_ts < float(self._api_backoff_until_ts):
            end_to_end_ms = int((time.time() - detected_at) * 1000)
            telemetry.update({
                'api_backoff': True,
                'api_backoff_until_ts': float(self._api_backoff_until_ts),
                'parallel_fetch_ms': 0,
                'temporal_total_ms': 0,
                'execution_ms': int((time.time() - execution_start) * 1000),
                'end_to_end_ms': end_to_end_ms,
                'pipeline_overhead_ms': max(0, end_to_end_ms - telemetry.get('queue_wait_ms', 0)),
            })
            base.update({
                'success': False,
                'status': 'API_BACKOFF',
                'bs_odd': 0, 'bs_limit': 0, 'num_bk': 0, 'diff_pct': 0,
                'error': f"API_BACKOFF until_ts={float(self._api_backoff_until_ts):.0f}",
                'total_ms': end_to_end_ms,
                'telemetry': telemetry,
            })
            return base

        # Drop explícito: evento ficou velho demais na fila.
        # Objetivo: preservar baixa latência e evitar gastar API em oportunidades já inválidas.
        if self.max_queue_wait_ms > 0 and queue_wait_ms > self.max_queue_wait_ms:
            self.dropped_stale_queue_wait += 1
            end_to_end_ms = int((time.time() - detected_at) * 1000)
            telemetry.update({
                'stale_dropped': True,
                'stale_reason': 'queue_wait_ms_exceeded',
                'parallel_fetch_ms': 0,
                'temporal_total_ms': 0,
                'execution_ms': int((time.time() - execution_start) * 1000),
                'end_to_end_ms': end_to_end_ms,
            })
            base.update({
                'success': False,
                'status': 'STALE_QUEUE_WAIT',
                'bs_odd': 0, 'bs_limit': 0, 'num_bk': 0, 'diff_pct': 0,
                'error': f"STALE_QUEUE_WAIT: queue_wait_ms={queue_wait_ms} > {self.max_queue_wait_ms}",
                'total_ms': end_to_end_ms,
                'telemetry': telemetry,
            })
            return base

        # Constrói bet_types (somente os necessários)
        t_build = time.time()
        back_bet_type = None
        lay_bet_type = None
        if self.api_sides in ("back", "both"):
            back_bet_type = ApiBetslipClient.build_bet_type(
                market_type=h3b['market_type'],
                side=h3b['side'],
                line=h3b['line'],
            )
        if self.api_sides in ("lay", "both"):
            lay_bet_type = ApiBetslipClient.build_lay_bet_type(
                market_type=h3b['market_type'],
                side=h3b['side'],
                line=h3b['line'],
            )
        build_bet_type_ms = int((time.time() - t_build) * 1000)

        # === T+0: BACK e/ou LAY ===
        t_parallel = time.time()
        back_result = None
        lay_result = None
        if self.api_sides == "back":
            back_result = await self.api_client.get_betslip_odds(event_id=h3b['event_id'], bet_type=str(back_bet_type))
        elif self.api_sides == "lay":
            lay_result = await self.api_client.get_betslip_odds(
                event_id=h3b['event_id'],
                bet_type=str(lay_bet_type),
                betslip_type="lay",
            )
        else:
            back_task = self.api_client.get_betslip_odds(event_id=h3b['event_id'], bet_type=str(back_bet_type))
            lay_task = self.api_client.get_betslip_odds(
                event_id=h3b['event_id'],
                bet_type=str(lay_bet_type),
                betslip_type="lay",
            )
            back_result, lay_result = await asyncio.gather(back_task, lay_task, return_exceptions=True)
        parallel_fetch_ms = int((time.time() - t_parallel) * 1000)
        
        # Trata exceções
        if isinstance(back_result, Exception):
            back_result = None
        if isinstance(lay_result, Exception):
            lay_result = None

        back_post_ms = back_result.request_time_ms if back_result else 0
        back_total_ms = back_result.total_time_ms if back_result else 0
        back_pmm_ms = max(0, back_total_ms - back_post_ms)

        lay_post_ms = lay_result.request_time_ms if lay_result else 0
        lay_total_ms = lay_result.total_time_ms if lay_result else 0
        lay_pmm_ms = max(0, lay_total_ms - lay_post_ms)

        telemetry.update({
            'build_bet_type_ms': build_bet_type_ms,
            'parallel_fetch_ms': parallel_fetch_ms,
            'back_post_ms': back_post_ms,
            'back_pmm_ms': back_pmm_ms,
            'back_total_ms': back_total_ms,
            'lay_post_ms': lay_post_ms,
            'lay_pmm_ms': lay_pmm_ms,
            'lay_total_ms': lay_total_ms,
            'back_success': bool(back_result and back_result.success),
            'lay_success': bool(lay_result and lay_result.success),
            'back_error': back_result.error if (back_result and not back_result.success and back_result.error) else '',
            'lay_error': lay_result.error if (lay_result and not lay_result.success and lay_result.error) else '',
            # Diagnóstico de WS/PMM (mais granular que apenas "No PMMs received")
            'back_pmm_count': int(getattr(back_result, "pmm_count", 0) or 0) if back_result else 0,
            'back_pmm_wait_s': float(getattr(back_result, "pmm_wait_s", 0.0) or 0.0) if back_result else 0.0,
            'back_ws_age_ms': getattr(back_result, "ws_age_ms", None) if back_result else None,
            'back_ws_msg_count': int(getattr(back_result, "ws_msg_count", 0) or 0) if back_result else 0,
            'back_betslip_id_source': str(getattr(back_result, "betslip_id_source", "") or "") if back_result else "",
            'lay_pmm_count': int(getattr(lay_result, "pmm_count", 0) or 0) if lay_result else 0,
            'lay_pmm_wait_s': float(getattr(lay_result, "pmm_wait_s", 0.0) or 0.0) if lay_result else 0.0,
            'lay_ws_age_ms': getattr(lay_result, "ws_age_ms", None) if lay_result else None,
            'lay_ws_msg_count': int(getattr(lay_result, "ws_msg_count", 0) or 0) if lay_result else 0,
            'lay_betslip_id_source': str(getattr(lay_result, "betslip_id_source", "") or "") if lay_result else "",
        })
        # Mantém compatibilidade: post_ms/pmm_ms refletem BACK quando existir, senão LAY.
        base['post_ms'] = back_post_ms if back_post_ms > 0 else lay_post_ms
        base['pmm_ms'] = back_pmm_ms if back_pmm_ms > 0 else lay_pmm_ms

        # Critério de sucesso depende do lado primário escolhido
        primary = back_result if (self.api_sides in ("back", "both")) else lay_result
        if not primary or not primary.success:
            lay_snapshot = _extract_lay_snapshot(lay_result)
            if lay_snapshot:
                base.update({
                    'lay_odd': lay_snapshot['odd'],
                    'lay_bookie': lay_snapshot['bookie'],
                    'lay_limit': lay_snapshot['limit'],
                    'lay_num_bk': lay_snapshot['num_bk'],
                })

            back_err = primary.error if primary else ('Back failed' if self.api_sides != "lay" else 'Lay failed')
            if lay_result and not lay_result.success and lay_result.error:
                back_err = f"{back_err} | lay={lay_result.error}"

            # Mitigação: quando o subcanal API (PMM/betslip) fica stale, o POST até pode responder 200,
            # mas os PMMs não chegam (No PMMs) e o ws_age_ms cresce. Isso tende a persistir até reload.
            try:
                if "No PMMs received" in str(back_err):
                    wsa = getattr(back_result, "ws_age_ms", None) if back_result else None
                    wsa = int(wsa) if isinstance(wsa, (int, float, str)) and str(wsa).strip() else None
                    # se ws_age_ms é muito alto, forçamos reload com cooldown para reativar o canal "api"
                    if wsa is not None and wsa >= int(float(os.getenv("AUDIT_API_WS_STALE_MS", "15000") or 15000)):
                        telemetry["api_ws_stale_reload"] = True
                        await self._force_reload("api_ws_stale_no_pmms")
            except Exception:
                pass

            # Se a API retorna 401/auth_error, isso é quase sempre sessão inválida/ausente.
            # Forçamos relogin com cooldown para recuperar automaticamente.
            back_http = int(getattr(back_result, "http_status", 0) or 0) if back_result else 0
            lay_http = int(getattr(lay_result, "http_status", 0) or 0) if lay_result else 0
            if (back_http == 401) or (lay_http == 401) or ("auth_error" in str(back_err)) or ("HTTP_401" in str(back_err)) or ("NO_ROOT_SESSION_COOKIE" in str(back_err)):
                telemetry["auth_401"] = True
                await self._force_relogin("HTTP_401/auth_error")

            # Rate limit: aplica backoff global para evitar “lock” por flood.
            retry_after = int(getattr(back_result, "rate_limit_retry_after_sec", 0) or 0) if back_result else 0
            if retry_after <= 0 and lay_result:
                retry_after = int(getattr(lay_result, "rate_limit_retry_after_sec", 0) or 0)
            if retry_after > 0:
                # margem de segurança
                self._api_backoff_until_ts = time.time() + float(retry_after) + 5.0
                status_code = "API_RATE_LIMIT"
            else:
                status_code = "API_FAILED"
                if "Execution context was destroyed" in str(back_err):
                    status_code = "API_CTX_DESTROYED"
                elif "Failed to fetch" in str(back_err):
                    status_code = "API_FETCH_FAILED"
                elif "No betslip_id received" in str(back_err):
                    status_code = "API_NO_BETSLIP_ID"

            end_to_end_ms = int((time.time() - detected_at) * 1000)
            telemetry['temporal_total_ms'] = 0
            telemetry['execution_ms'] = int((time.time() - execution_start) * 1000)
            telemetry['end_to_end_ms'] = end_to_end_ms
            telemetry['pipeline_overhead_ms'] = max(
                0,
                end_to_end_ms - (telemetry['queue_wait_ms'] + telemetry['parallel_fetch_ms'])
            )
            base.update({
                'success': False,
                'status': status_code,
                'bs_odd': 0, 'bs_limit': 0, 'num_bk': 0, 'diff_pct': 0,
                'error': back_err,
                'total_ms': end_to_end_ms,
                'telemetry': telemetry,
            })
            return base

        ws_odd = h3b['websocket_odd']
        if self.api_sides == "lay":
            # Em lay-only, persistimos "bs_*" como snapshot do ticket LAY (para compatibilidade com bridge)
            lay_snapshot = _extract_lay_snapshot(lay_result)
            if not lay_snapshot:
                base.update({'success': False, 'status': 'API_FAILED', 'bs_odd': 0, 'bs_limit': 0, 'num_bk': 0, 'diff_pct': 0, 'error': 'Lay snapshot missing'})
                return base
            diff = ((float(lay_snapshot['odd']) - ws_odd) / ws_odd) * 100 if ws_odd else 0
            base.update({
                'success': True,
                'status': 'OK',
                'bs_odd': float(lay_snapshot['odd']),
                'bs_bookie': lay_snapshot['bookie'],
                'bs_limit': float(lay_snapshot['limit']),
                'num_bk': int(lay_snapshot['num_bk']),
                'diff_pct': diff,
                'lay_odd': lay_snapshot['odd'],
                'lay_bookie': lay_snapshot['bookie'],
                'lay_limit': lay_snapshot['limit'],
                'lay_num_bk': lay_snapshot['num_bk'],
            })
        else:
            diff = ((back_result.best_odd - ws_odd) / ws_odd) * 100 if ws_odd else 0

            base.update({
                'success': True,
                'status': 'OK',
                'bs_odd': back_result.best_odd,
                'bs_bookie': back_result.best_bookie,
                'bs_limit': back_result.best_limit,
                'second_odd': back_result.second_odd,
                'second_bookie': back_result.second_bookie,
                'highest_limit': back_result.highest_limit,
                'highest_limit_bookie': back_result.highest_limit_bookie,
                'num_bk': back_result.num_bookmakers,
                'diff_pct': diff,
            })

            # Lay (capturado simultaneamente ao back)
            lay_snapshot = _extract_lay_snapshot(lay_result)
            if lay_snapshot:
                base['lay_odd'] = lay_snapshot['odd']
                base['lay_bookie'] = lay_snapshot['bookie']
                base['lay_limit'] = lay_snapshot['limit']
                base['lay_num_bk'] = lay_snapshot['num_bk']

        back_betslip_id = back_result.betslip_id if back_result and back_result.success else ""
        lay_betslip_id = lay_result.betslip_id if lay_result and lay_result.success else ""
        has_temporal_refs = bool(back_betslip_id or lay_betslip_id)
        if run_temporal and has_temporal_refs:
            rt = None
            if self.mode in ("ws_vs_bs", "wsbs", "ws_vs_betslip"):
                # Compara WS vs BS nos mesmos timestamps (t+offsets); t0 já foi coletado no get_betslip_odds
                rt = [x for x in (self.ws_sample_offsets_sec or []) if float(x) > 0.0]
            back_temporal, lay_temporal, telemetry_patch = await self._collect_temporal_series(
                ws_odd=ws_odd,
                back_betslip_id=back_betslip_id,
                lay_betslip_id=lay_betslip_id,
                ws_state_key=h3b.get("ws_state_key"),
                ws_side=str(h3b.get("side") or ""),
                refresh_times=rt,
            )
            telemetry.update(telemetry_patch)
            if back_temporal:
                base['temporal'] = back_temporal
                evol = " -> ".join([f"t+{t['t']:.0f}s:{t['bs_odd']:.3f}({t['diff_pct']:+.1f}%)" for t in back_temporal])
                logger.info(f"  Temporal BACK: {evol}")
            if lay_temporal:
                base['lay_temporal'] = lay_temporal
                evol_lay = " -> ".join([f"t+{t['t']:.0f}s:{t['lay_odd']:.3f}({t['diff_pct']:+.1f}%)" for t in lay_temporal])
                logger.info(f"  Temporal LAY: {evol_lay}")
        else:
            telemetry['temporal_total_ms'] = 0
            telemetry['temporal_wait_ms'] = 0
            telemetry['temporal_refresh_mean_ms'] = 0
            telemetry['temporal_points_back'] = 0
            telemetry['temporal_points_lay'] = 0
            telemetry['temporal_points'] = []
            telemetry['temporal_deferred'] = has_temporal_refs and (not run_temporal)
            if has_temporal_refs and (not run_temporal):
                # Se não há temporal workers, não adianta deferir — fecha para evitar acúmulo de betslips abertos.
                if int(getattr(self, "temporal_workers", 0) or 0) <= 0:
                    telemetry['temporal_deferred'] = False
                    try:
                        close_calls = []
                        if back_betslip_id:
                            close_calls.append(asyncio.wait_for(self.api_client.close_betslip(back_betslip_id), timeout=1.2))
                        if lay_betslip_id:
                            close_calls.append(asyncio.wait_for(self.api_client.close_betslip(lay_betslip_id), timeout=1.2))
                        if close_calls:
                            await asyncio.gather(*close_calls, return_exceptions=True)
                    except Exception:
                        pass
                else:
                    rt = None
                    if self.mode in ("ws_vs_bs", "wsbs", "ws_vs_betslip"):
                        rt = [x for x in (self.ws_sample_offsets_sec or []) if float(x) > 0.0]
                    base['_temporal_refs'] = {
                        'ws_odd': ws_odd,
                        'back_betslip_id': back_betslip_id,
                        'lay_betslip_id': lay_betslip_id,
                        'ws_state_key': h3b.get("ws_state_key"),
                        'ws_side': str(h3b.get("side") or ""),
                        'refresh_times': rt,
                    }

        end_to_end_ms = int((time.time() - detected_at) * 1000)
        telemetry['execution_ms'] = int((time.time() - execution_start) * 1000)
        telemetry['end_to_end_ms'] = end_to_end_ms
        known_ms = telemetry['queue_wait_ms'] + telemetry['parallel_fetch_ms'] + telemetry['temporal_total_ms']
        telemetry['pipeline_overhead_ms'] = max(0, end_to_end_ms - known_ms)

        base['total_ms'] = end_to_end_ms
        base['telemetry'] = telemetry
        return base

    async def _execute_ws_only(self, h3b: dict, run_ws_series: bool = True) -> dict:
        """
        Modo WS-only: não chama betslip API. Coleta série WS (t0..t+30) para motor de análise.
        """
        detected_at = h3b['detected_at']
        execution_start = time.time()
        queue_wait_ms = int((execution_start - detected_at) * 1000)
        queue_depth_at_enqueue = h3b.get('queue_depth_at_enqueue')
        queue_depth_after_dequeue = h3b.get('queue_depth_after_dequeue')

        telemetry: Dict[str, Any] = {
            'queue_wait_ms': queue_wait_ms,
            'queue_depth_at_enqueue': queue_depth_at_enqueue,
            'queue_depth_after_dequeue': queue_depth_after_dequeue,
            'ws_series_deferred': False,
        }

        base: Dict[str, Any] = {
            'event_id': h3b['event_id'],
            'home_team': h3b['home_team'],
            'away_team': h3b['away_team'],
            'league': h3b.get('league', ''),
            'kickoff': h3b.get('kickoff'),
            'market_type': h3b['market_type'],
            'market_period': h3b.get('market_period', 'full_time'),
            'line': h3b['line'],
            'side': h3b['side'],
            'ws_odd': h3b['websocket_odd'],
            'is_live': h3b.get('is_live'),
            'direction': h3b.get('direction', 'up'),
            'detected_at': detected_at,
            'post_ms': 0,
            'pmm_ms': 0,
            'audit_version': "v5.0-ws-only",
        }

        ws_state_key = h3b.get("ws_state_key")
        offsets = list(self.ws_sample_offsets_sec or [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30])
        if run_ws_series:
            ws_series, telemetry_patch = await self._collect_ws_series(
                ws_state_key=ws_state_key,
                ws_side=str(h3b.get("side") or ""),
                offsets_sec=offsets,
            )
            telemetry.update(telemetry_patch)
            base["ws_series"] = ws_series
        else:
            telemetry["ws_series_deferred"] = True
            base["_ws_series_refs"] = {
                "ws_state_key": ws_state_key,
                "ws_side": str(h3b.get("side") or ""),
                "offsets_sec": offsets,
            }

        end_to_end_ms = int((time.time() - detected_at) * 1000)
        telemetry['execution_ms'] = int((time.time() - execution_start) * 1000)
        telemetry['end_to_end_ms'] = end_to_end_ms
        telemetry['pipeline_overhead_ms'] = max(0, end_to_end_ms - telemetry.get('queue_wait_ms', 0))

        base.update({
            'success': True,
            'status': 'OK',
            'bs_odd': None,
            'bs_limit': 0,
            'num_bk': 0,
            'diff_pct': None,
            'error': '',
            'total_ms': end_to_end_ms,
            'telemetry': telemetry,
        })
        return base

    async def _execute_ws_gate_lay(self, h3b: dict, *, defer_temporal: bool = True) -> dict:
        """
        Gate (H3B UP + queda intensa em 5s):
          - coleta WS(t0) e WS(t+offset) inline
          - se WS(t+offset) < gate_drop_ratio * WS(t0): tenta abrir ticket LAY (Exchange)
          - aplica cap de aberturas por janela e respeita backoff global

        Observação: mesmo quando o cap bloqueia, salvamos um registro com status "GATE_BLOCKED"
        para medir quantos elegíveis foram perdidos por limitação.
        """
        self.gate_seen += 1
        detected_at = h3b['detected_at']
        execution_start = time.time()
        queue_wait_ms = int((execution_start - detected_at) * 1000)

        telemetry: Dict[str, Any] = {
            'queue_wait_ms': queue_wait_ms,
            'queue_depth_at_enqueue': h3b.get('queue_depth_at_enqueue'),
            'queue_depth_after_dequeue': h3b.get('queue_depth_after_dequeue'),
            'gate_mode': 'ws_gate_lay',
            'gate_drop_offset_sec': float(self.gate_drop_offset_sec),
            'gate_drop_ratio': float(self.gate_drop_ratio),
        }

        base: Dict[str, Any] = {
            'event_id': h3b['event_id'],
            'home_team': h3b['home_team'],
            'away_team': h3b['away_team'],
            'league': h3b.get('league', ''),
            'kickoff': h3b.get('kickoff'),
            'market_type': h3b['market_type'],
            'market_period': h3b.get('market_period', 'full_time'),
            'line': h3b['line'],
            'side': h3b['side'],
            'ws_odd': h3b['websocket_odd'],
            'is_live': h3b.get('is_live'),
            'direction': h3b.get('direction', 'up'),
            'detected_at': detected_at,
            'post_ms': 0,
            'pmm_ms': 0,
            'audit_version': "v5.1-ws-gate-lay",
            # evita que o bridge Back execute oportunidades Lay
            'exec_side_hint': "Lay",
        }

        ws_state_key = h3b.get("ws_state_key")
        ws_side = str(h3b.get("side") or "")

        # WS(t0): usa o valor capturado no instante da detecção (sem depender da fila)
        ws0 = None
        try:
            if isinstance(h3b.get("websocket_odd"), (int, float)) and float(h3b.get("websocket_odd")) > 0:
                ws0 = float(h3b.get("websocket_odd"))
        except Exception:
            ws0 = None

        # WS(t+offset): mede no timestamp alvo (detected_at + offset), não "5s após iniciar o worker"
        target_abs_ts = float(detected_at) + float(self.gate_drop_offset_sec)
        now0 = time.time()
        late_s = float(now0) - float(target_abs_ts)
        telemetry["gate_target_abs_ts"] = float(target_abs_ts)
        telemetry["gate_late_s_at_start"] = float(late_s)

        # Se já está tarde demais, marca stale e não tenta abrir ticket (evita distorção)
        if late_s > float(self.gate_max_late_sec):
            base['ws_gate_series'] = [
                {"t_target_s": 0.0, "t_actual_s": max(0.0, float(now0 - float(detected_at))), "ts": float(detected_at), "ws_side": ws_side, "ws_odd": ws0},
            ]
            base.update({
                'success': True,
                'status': 'GATE_STALE',
                'bs_odd': None,
                'bs_limit': 0,
                'num_bk': 0,
                'diff_pct': None,
                'error': f"LATE>{float(self.gate_max_late_sec):.1f}s",
                'total_ms': int((time.time() - detected_at) * 1000),
                'telemetry': telemetry,
            })
            return base

        wait_s = max(0.0, float(target_abs_ts) - float(now0))
        telemetry["gate_wait_s"] = float(wait_s)
        if wait_s > 0:
            await asyncio.sleep(wait_s)

        # lê WS atual do estado
        ws5 = None
        try:
            if ws_state_key and ws_side:
                cur = self._ws_get_side_odd(ws_state_key, ws_side)
                if isinstance(cur, (int, float)) and float(cur) > 0:
                    ws5 = float(cur)
        except Exception:
            ws5 = None

        telemetry['gate_ws_t0'] = ws0
        telemetry['gate_ws_t5'] = ws5

        # série do gate (para debug/auditoria)
        t_actual_5 = max(0.0, float(time.time() - float(detected_at)))
        base['ws_gate_series'] = [
            {"t_target_s": 0.0, "t_actual_s": 0.0, "ts": float(detected_at), "ws_side": ws_side, "ws_odd": ws0},
            {"t_target_s": float(self.gate_drop_offset_sec), "t_actual_s": t_actual_5, "ts": time.time(), "ws_side": ws_side, "ws_odd": ws5},
        ]

        if not ws_state_key or not ws_side:
            self.gate_ws_missing += 1
            base.update({
                'success': True,
                'status': 'GATE_WS_MISSING',
                'bs_odd': None,
                'bs_limit': 0,
                'num_bk': 0,
                'diff_pct': None,
                'error': 'WS_STATE_KEY_OR_SIDE_MISSING',
                'total_ms': int((time.time() - detected_at) * 1000),
                'telemetry': telemetry,
            })
            return base

        if not ws0 or not ws5:
            self.gate_ws_missing += 1
            base.update({
                'success': True,
                'status': 'GATE_WS_POINT_MISSING',
                'bs_odd': None,
                'bs_limit': 0,
                'num_bk': 0,
                'diff_pct': None,
                'error': 'WS_POINT_MISSING',
                'total_ms': int((time.time() - detected_at) * 1000),
                'telemetry': telemetry,
            })
            return base

        gate_ok = bool(float(ws5) < float(self.gate_drop_ratio) * float(ws0))
        telemetry['gate_eligible'] = gate_ok
        telemetry['gate_drop_ratio_obs'] = float(ws5) / float(ws0) if ws0 else None

        # Se não elegível, ainda salva para medir taxa do filtro
        if not gate_ok:
            self.gate_not_eligible += 1
            # se existir temporal worker, podemos coletar série completa WS depois (para análises)
            if defer_temporal:
                base["_ws_series_refs"] = {
                    "ws_state_key": ws_state_key,
                    "ws_side": ws_side,
                    # garante que inclui offset do gate e os offsets padrões
                    "offsets_sec": sorted({0.0, float(self.gate_drop_offset_sec), *[float(x) for x in (self.ws_sample_offsets_sec or [])]}),
                }
                telemetry["ws_series_deferred"] = True
            base.update({
                'success': True,
                'status': 'GATE_NOT_ELIGIBLE',
                'bs_odd': None,
                'bs_limit': 0,
                'num_bk': 0,
                'diff_pct': None,
                'error': '',
                'total_ms': int((time.time() - detected_at) * 1000),
                'telemetry': telemetry,
            })
            return base

        self.gate_eligible += 1

        # Backoff global (rate limit observado) -> não abre ticket
        now_ts = time.time()
        if self._api_backoff_until_ts and now_ts < float(self._api_backoff_until_ts):
            self.gate_blocked_backoff += 1
            telemetry['api_backoff'] = True
            telemetry['api_backoff_until_ts'] = float(self._api_backoff_until_ts)
            base.update({
                'success': True,
                'status': 'GATE_BLOCKED_BACKOFF',
                'bs_odd': None,
                'bs_limit': 0,
                'num_bk': 0,
                'diff_pct': None,
                'error': f"API_BACKOFF until_ts={float(self._api_backoff_until_ts):.0f}",
                'total_ms': int((time.time() - detected_at) * 1000),
                'telemetry': telemetry,
            })
            return base

        # Cap local de aberturas
        allowed, cap_meta = await self._gate_try_acquire_open_slot()
        telemetry['gate_cap'] = cap_meta
        if not allowed:
            self.gate_blocked_cap += 1
            base.update({
                'success': True,
                'status': 'GATE_BLOCKED_CAP',
                'bs_odd': None,
                'bs_limit': 0,
                'num_bk': 0,
                'diff_pct': None,
                'error': 'CAP_OPENINGS',
                'total_ms': int((time.time() - detected_at) * 1000),
                'telemetry': telemetry,
            })
            return base

        # --- abre ticket LAY (Exchange) ---
        self.gate_open_attempts += 1
        t_build = time.time()
        lay_bet_type = ApiBetslipClient.build_lay_bet_type(
            market_type=h3b['market_type'],
            side=h3b['side'],
            line=h3b['line'],
        )
        telemetry['build_bet_type_ms'] = int((time.time() - t_build) * 1000)

        t_call = time.time()
        lay_result = await self.api_client.get_betslip_odds(
            event_id=h3b['event_id'],
            bet_type=lay_bet_type,
            betslip_type="lay",
        )
        telemetry['lay_post_ms'] = int(lay_result.request_time_ms) if lay_result else 0
        telemetry['lay_total_ms'] = int(lay_result.total_time_ms) if lay_result else 0
        telemetry['parallel_fetch_ms'] = int((time.time() - t_call) * 1000)
        telemetry['lay_success'] = bool(lay_result and lay_result.success)
        telemetry['lay_error'] = lay_result.error if (lay_result and not lay_result.success and lay_result.error) else ''

        def _extract_lay_snapshot(api_result: Optional[BetslipApiResult]) -> Optional[dict]:
            if not api_result or not api_result.success:
                return None
            lay_bookmakers = [b for b in api_result.bookmakers if b.best_price > 0]
            if not lay_bookmakers:
                return None
            best = min(lay_bookmakers, key=lambda b: b.best_price)
            return {
                'odd': best.best_price,
                'bookie': best.bookie,
                'limit': best.max_stake,
                'num_bk': len(lay_bookmakers),
            }

        lay_snapshot = _extract_lay_snapshot(lay_result)
        if lay_snapshot:
            base.update({
                'lay_odd': lay_snapshot['odd'],
                'lay_bookie': lay_snapshot['bookie'],
                'lay_limit': lay_snapshot['limit'],
                'lay_num_bk': lay_snapshot['num_bk'],
            })
            # Para compatibilizar com análises e bridge:
            # - registramos a odd/limit do ticket LAY como "bs_*" (odd efetiva no betslip)
            # - diff_pct passa a ser comparável ao WS(t0): (ticket - ws0) / ws0
            try:
                base['bs_odd'] = float(lay_snapshot['odd'])
                base['bs_limit'] = float(lay_snapshot['limit'])
                base['num_bk'] = int(lay_snapshot['num_bk'])
                if ws0 and float(ws0) > 0:
                    base['diff_pct'] = (float(base['bs_odd']) - float(ws0)) / float(ws0) * 100.0
                else:
                    base['diff_pct'] = None
                # Marca explicitamente como oportunidade executável quando o gate abriu ticket com sucesso
                base['is_valid_opportunity'] = True
            except Exception:
                pass

        retry_after = int(getattr(lay_result, "rate_limit_retry_after_sec", 0) or 0) if lay_result else 0
        if retry_after > 0:
            # margem de segurança
            self._api_backoff_until_ts = time.time() + float(retry_after) + 5.0
            self.gate_open_failed += 1
            base.update({
                'success': False,
                'status': 'API_RATE_LIMIT',
                'bs_odd': None,
                'bs_limit': 0,
                'num_bk': 0,
                'diff_pct': None,
                'error': f"RATE_LIMIT retry_after={int(retry_after)}s",
                'total_ms': int((time.time() - detected_at) * 1000),
                'telemetry': telemetry,
            })
            return base

        if not lay_result or not lay_result.success:
            self.gate_open_failed += 1
            base.update({
                'success': False,
                'status': 'API_FAILED',
                'bs_odd': None,
                'bs_limit': 0,
                'num_bk': 0,
                'diff_pct': None,
                'error': lay_result.error if lay_result else 'Lay failed',
                'total_ms': int((time.time() - detected_at) * 1000),
                'telemetry': telemetry,
            })
            return base

        self.gate_open_success += 1
        lay_betslip_id = lay_result.betslip_id if lay_result and lay_result.success else ""
        # Se não vamos fazer refresh temporal, fecha o betslip já para reduzir "open betslips".
        if (not self.gate_lay_refresh) and lay_betslip_id:
            try:
                await asyncio.wait_for(self.api_client.close_betslip(lay_betslip_id), timeout=1.2)
            except Exception:
                pass
        # Opcional: coletar temporal Lay via refresh (deferred no temporal worker)
        if self.gate_lay_refresh and lay_betslip_id:
            base["_temporal_refs"] = {
                "ws_odd": ws0,
                "ws_state_key": ws_state_key,
                "ws_side": ws_side,
                "refresh_times": list(self.gate_lay_refresh_times_sec or []),
                "back_betslip_id": "",
                "lay_betslip_id": lay_betslip_id,
            }

        # Também opcional: coletar série WS completa de forma assíncrona (para reversão)
        if defer_temporal:
            base["_ws_series_refs"] = {
                "ws_state_key": ws_state_key,
                "ws_side": ws_side,
                "offsets_sec": sorted({0.0, float(self.gate_drop_offset_sec), *[float(x) for x in (self.ws_sample_offsets_sec or [])]}),
            }
            telemetry["ws_series_deferred"] = True

        end_to_end_ms = int((time.time() - detected_at) * 1000)
        telemetry['execution_ms'] = int((time.time() - execution_start) * 1000)
        telemetry['end_to_end_ms'] = end_to_end_ms
        telemetry['pipeline_overhead_ms'] = max(0, end_to_end_ms - telemetry.get('queue_wait_ms', 0))

        base.update({
            'success': True,
            'status': 'OK',
            # preserva o snapshot do ticket (se houver)
            'bs_odd': base.get('bs_odd'),
            'bs_limit': base.get('bs_limit', 0),
            'num_bk': base.get('num_bk', 0),
            'diff_pct': base.get('diff_pct'),
            'error': '',
            'total_ms': end_to_end_ms,
            'telemetry': telemetry,
        })
        return base

    async def _execute_ws_reversal_lay(self, h3b: dict, *, defer_temporal: bool = True) -> dict:
        """
        LAY por reversão (H3B):
          - usa WS(t0) do instante da reversão (odd_at_reversal) como referência
          - abre ticket LAY imediatamente (sem gate t+5)
          - aplica cap de aberturas por janela e respeita backoff global
        """
        detected_at = h3b['detected_at']
        execution_start = time.time()
        queue_wait_ms = int((execution_start - detected_at) * 1000)

        telemetry: Dict[str, Any] = {
            'queue_wait_ms': queue_wait_ms,
            'queue_depth_at_enqueue': h3b.get('queue_depth_at_enqueue'),
            'queue_depth_after_dequeue': h3b.get('queue_depth_after_dequeue'),
            'gate_mode': 'ws_reversal_lay',
        }

        base: Dict[str, Any] = {
            'event_id': h3b['event_id'],
            'home_team': h3b['home_team'],
            'away_team': h3b['away_team'],
            'league': h3b.get('league', ''),
            'kickoff': h3b.get('kickoff'),
            'market_type': h3b['market_type'],
            'market_period': h3b.get('market_period', 'full_time'),
            'line': h3b['line'],
            'side': h3b['side'],
            'ws_odd': h3b['websocket_odd'],
            'is_live': h3b.get('is_live'),
            'direction': h3b.get('direction', 'up'),
            'detected_at': detected_at,
            'post_ms': 0,
            'pmm_ms': 0,
            'audit_version': "v5.4-ws-reversal-lay",
            'exec_side_hint': "Lay",
        }

        # Protege contra fila velha demais: se atrasou muito, não abre ticket.
        late_s = float(time.time()) - float(detected_at)
        telemetry["late_s_at_start"] = float(late_s)
        if late_s > float(self.gate_max_late_sec):
            base.update({
                'success': True,
                'status': 'GATE_STALE',
                'bs_odd': None,
                'bs_limit': 0,
                'num_bk': 0,
                'diff_pct': None,
                'error': f"LATE>{float(self.gate_max_late_sec):.1f}s",
                'total_ms': int((time.time() - detected_at) * 1000),
                'telemetry': telemetry,
                'is_valid_opportunity': False,
            })
            return base

        # Backoff global (rate limit observado) -> não abre ticket
        now_ts = time.time()
        if self._api_backoff_until_ts and now_ts < float(self._api_backoff_until_ts):
            telemetry['api_backoff'] = True
            telemetry['api_backoff_until_ts'] = float(self._api_backoff_until_ts)
            base.update({
                'success': True,
                'status': 'GATE_BLOCKED_BACKOFF',
                'bs_odd': None,
                'bs_limit': 0,
                'num_bk': 0,
                'diff_pct': None,
                'error': f"API_BACKOFF until_ts={float(self._api_backoff_until_ts):.0f}",
                'total_ms': int((time.time() - detected_at) * 1000),
                'telemetry': telemetry,
                'is_valid_opportunity': False,
            })
            return base

        # Cap local de aberturas
        allowed, cap_meta = await self._gate_try_acquire_open_slot()
        telemetry['gate_cap'] = cap_meta
        if not allowed:
            base.update({
                'success': True,
                'status': 'GATE_BLOCKED_CAP',
                'bs_odd': None,
                'bs_limit': 0,
                'num_bk': 0,
                'diff_pct': None,
                'error': 'CAP_OPENINGS',
                'total_ms': int((time.time() - detected_at) * 1000),
                'telemetry': telemetry,
                'is_valid_opportunity': False,
            })
            return base

        # --- abre ticket LAY (Exchange) imediatamente ---
        self.gate_open_attempts += 1
        t_build = time.time()
        lay_bet_type = ApiBetslipClient.build_lay_bet_type(
            market_type=h3b['market_type'],
            side=h3b['side'],
            line=h3b['line'],
        )
        telemetry['build_bet_type_ms'] = int((time.time() - t_build) * 1000)

        t_call = time.time()
        lay_result = await self.api_client.get_betslip_odds(
            event_id=h3b['event_id'],
            bet_type=lay_bet_type,
            betslip_type="lay",
        )
        telemetry['lay_post_ms'] = int(lay_result.request_time_ms) if lay_result else 0
        telemetry['lay_total_ms'] = int(lay_result.total_time_ms) if lay_result else 0
        telemetry['parallel_fetch_ms'] = int((time.time() - t_call) * 1000)
        telemetry['lay_success'] = bool(lay_result and lay_result.success)
        telemetry['lay_error'] = lay_result.error if (lay_result and not lay_result.success and lay_result.error) else ''

        retry_after = int(getattr(lay_result, "rate_limit_retry_after_sec", 0) or 0) if lay_result else 0
        if retry_after > 0:
            self._api_backoff_until_ts = time.time() + float(retry_after) + 5.0
            self.gate_open_failed += 1
            base.update({
                'success': False,
                'status': 'API_RATE_LIMIT',
                'bs_odd': None,
                'bs_limit': 0,
                'num_bk': 0,
                'diff_pct': None,
                'error': f"RATE_LIMIT retry_after={int(retry_after)}s",
                'total_ms': int((time.time() - detected_at) * 1000),
                'telemetry': telemetry,
                'is_valid_opportunity': False,
            })
            return base

        if not lay_result or not lay_result.success:
            self.gate_open_failed += 1
            base.update({
                'success': False,
                'status': 'API_FAILED',
                'bs_odd': None,
                'bs_limit': 0,
                'num_bk': 0,
                'diff_pct': None,
                'error': lay_result.error if lay_result else 'Lay failed',
                'total_ms': int((time.time() - detected_at) * 1000),
                'telemetry': telemetry,
                'is_valid_opportunity': False,
            })
            return base

        # snapshot Lay (menor odd)
        lay_bookmakers = [b for b in (lay_result.bookmakers or []) if getattr(b, "best_price", 0) > 0]
        if lay_bookmakers:
            best = min(lay_bookmakers, key=lambda b: b.best_price)
            base.update({
                'lay_odd': float(best.best_price),
                'lay_bookie': str(best.bookie),
                'lay_limit': float(best.max_stake),
                'lay_num_bk': int(len(lay_bookmakers)),
            })
            try:
                base['bs_odd'] = float(best.best_price)
                base['bs_limit'] = float(best.max_stake)
                base['num_bk'] = int(len(lay_bookmakers))
            except Exception:
                pass

        self.gate_open_success += 1
        lay_betslip_id = lay_result.betslip_id if lay_result and lay_result.success else ""
        if (not self.gate_lay_refresh) and lay_betslip_id:
            try:
                await asyncio.wait_for(self.api_client.close_betslip(lay_betslip_id), timeout=1.2)
            except Exception:
                pass
        if self.gate_lay_refresh and lay_betslip_id:
            base["_temporal_refs"] = {
                "ws_odd": base.get("ws_odd"),
                "ws_state_key": h3b.get("ws_state_key"),
                "ws_side": str(h3b.get("side") or ""),
                "refresh_times": list(self.gate_lay_refresh_times_sec or []),
                "back_betslip_id": "",
                "lay_betslip_id": lay_betslip_id,
            }

        end_to_end_ms = int((time.time() - detected_at) * 1000)
        telemetry['execution_ms'] = int((time.time() - execution_start) * 1000)
        telemetry['end_to_end_ms'] = end_to_end_ms
        telemetry['pipeline_overhead_ms'] = max(0, end_to_end_ms - telemetry.get('queue_wait_ms', 0))

        base.update({
            'success': True,
            'status': 'OK',
            'diff_pct': base.get('diff_pct'),
            'error': '',
            'total_ms': end_to_end_ms,
            'telemetry': telemetry,
            'is_valid_opportunity': True,
        })
        return base

    async def _execute_ws_gate_back(self, h3b: dict, *, defer_temporal: bool = True) -> dict:
        """
        Gate Back (H3B UP + alta em 5s):
          - mede WS(t0) e WS(t+offset) no timestamp alvo (detected_at + offset)
          - se WS(t+offset) >= gate_rise_ratio * WS(t0): marca como oportunidade executável (Back)
          - NÃO abre betslip (executor fará isso no momento da execução)
        """
        self.gate_back_seen += 1
        detected_at = h3b['detected_at']
        execution_start = time.time()
        queue_wait_ms = int((execution_start - detected_at) * 1000)

        telemetry: Dict[str, Any] = {
            'queue_wait_ms': queue_wait_ms,
            'queue_depth_at_enqueue': h3b.get('queue_depth_at_enqueue'),
            'queue_depth_after_dequeue': h3b.get('queue_depth_after_dequeue'),
            'gate_mode': 'ws_gate_back',
            'gate_rise_offset_sec': float(self.gate_rise_offset_sec),
            'gate_rise_ratio': float(self.gate_rise_ratio),
        }

        base: Dict[str, Any] = {
            'event_id': h3b['event_id'],
            'home_team': h3b['home_team'],
            'away_team': h3b['away_team'],
            'league': h3b.get('league', ''),
            'kickoff': h3b.get('kickoff'),
            'market_type': h3b['market_type'],
            'market_period': h3b.get('market_period', 'full_time'),
            'line': h3b['line'],
            'side': h3b['side'],
            'ws_odd': h3b['websocket_odd'],
            'is_live': h3b.get('is_live'),
            'direction': h3b.get('direction', 'up'),
            'detected_at': detected_at,
            'post_ms': 0,
            'pmm_ms': 0,
            'audit_version': "v5.3-ws-gate-back",
            'exec_side_hint': "Back",
        }

        ws_state_key = h3b.get("ws_state_key")
        ws_side = str(h3b.get("side") or "")

        # WS(t0): valor capturado no instante da detecção
        ws0 = None
        try:
            if isinstance(h3b.get("websocket_odd"), (int, float)) and float(h3b.get("websocket_odd")) > 0:
                ws0 = float(h3b.get("websocket_odd"))
        except Exception:
            ws0 = None

        # WS(t+offset) no timestamp alvo (detected_at + offset)
        target_abs_ts = float(detected_at) + float(self.gate_rise_offset_sec)
        now0 = time.time()
        late_s = float(now0) - float(target_abs_ts)
        telemetry["gate_target_abs_ts"] = float(target_abs_ts)
        telemetry["gate_late_s_at_start"] = float(late_s)

        if late_s > float(self.gate_max_late_sec):
            base['ws_gate_series'] = [
                {"t_target_s": 0.0, "t_actual_s": max(0.0, float(now0 - float(detected_at))), "ts": float(detected_at), "ws_side": ws_side, "ws_odd": ws0},
            ]
            base.update({
                'success': True,
                'status': 'GATE_STALE',
                'bs_odd': None,
                'bs_limit': 0,
                'num_bk': 0,
                'diff_pct': None,
                'error': f"LATE>{float(self.gate_max_late_sec):.1f}s",
                'total_ms': int((time.time() - detected_at) * 1000),
                'telemetry': telemetry,
            })
            return base

        wait_s = max(0.0, float(target_abs_ts) - float(now0))
        telemetry["gate_wait_s"] = float(wait_s)
        if wait_s > 0:
            await asyncio.sleep(wait_s)

        ws5 = None
        try:
            if ws_state_key and ws_side:
                cur = self._ws_get_side_odd(ws_state_key, ws_side)
                if isinstance(cur, (int, float)) and float(cur) > 0:
                    ws5 = float(cur)
        except Exception:
            ws5 = None

        telemetry['gate_ws_t0'] = ws0
        telemetry['gate_ws_t5'] = ws5
        t_actual_5 = max(0.0, float(time.time() - float(detected_at)))
        base['ws_gate_series'] = [
            {"t_target_s": 0.0, "t_actual_s": 0.0, "ts": float(detected_at), "ws_side": ws_side, "ws_odd": ws0},
            {"t_target_s": float(self.gate_rise_offset_sec), "t_actual_s": t_actual_5, "ts": time.time(), "ws_side": ws_side, "ws_odd": ws5},
        ]

        if not ws_state_key or not ws_side:
            self.gate_back_ws_missing += 1
            base.update({
                'success': True,
                'status': 'GATE_WS_MISSING',
                'bs_odd': None,
                'bs_limit': 0,
                'num_bk': 0,
                'diff_pct': None,
                'error': 'WS_STATE_KEY_OR_SIDE_MISSING',
                'total_ms': int((time.time() - detected_at) * 1000),
                'telemetry': telemetry,
            })
            return base

        if not ws0 or not ws5:
            self.gate_back_ws_missing += 1
            base.update({
                'success': True,
                'status': 'GATE_WS_POINT_MISSING',
                'bs_odd': None,
                'bs_limit': 0,
                'num_bk': 0,
                'diff_pct': None,
                'error': 'WS_POINT_MISSING',
                'total_ms': int((time.time() - detected_at) * 1000),
                'telemetry': telemetry,
            })
            return base

        gate_ok = bool(float(ws5) >= float(self.gate_rise_ratio) * float(ws0))
        telemetry['gate_eligible'] = gate_ok
        telemetry['gate_rise_ratio_obs'] = float(ws5) / float(ws0) if ws0 else None
        telemetry['gate_rise_pct_obs'] = ((float(ws5) - float(ws0)) / float(ws0) * 100.0) if ws0 else None

        if not gate_ok:
            self.gate_back_not_eligible += 1
            if defer_temporal:
                base["_ws_series_refs"] = {
                    "ws_state_key": ws_state_key,
                    "ws_side": ws_side,
                    "offsets_sec": sorted({0.0, float(self.gate_rise_offset_sec), *[float(x) for x in (self.ws_sample_offsets_sec or [])]}),
                }
                telemetry["ws_series_deferred"] = True
            base.update({
                'success': True,
                'status': 'GATE_NOT_ELIGIBLE',
                'bs_odd': None,
                'bs_limit': 0,
                'num_bk': 0,
                'diff_pct': None,
                'error': '',
                'total_ms': int((time.time() - detected_at) * 1000),
                'telemetry': telemetry,
                'is_valid_opportunity': False,
            })
            return base

        self.gate_back_eligible += 1
        end_to_end_ms = int((time.time() - detected_at) * 1000)
        telemetry['execution_ms'] = int((time.time() - execution_start) * 1000)
        telemetry['end_to_end_ms'] = end_to_end_ms
        telemetry['pipeline_overhead_ms'] = max(0, end_to_end_ms - telemetry.get('queue_wait_ms', 0))

        base.update({
            'success': True,
            'status': 'OK',
            'bs_odd': None,
            'bs_limit': 0,
            'num_bk': 0,
            'diff_pct': None,
            'error': '',
            'total_ms': end_to_end_ms,
            'telemetry': telemetry,
            'is_valid_opportunity': True,
        })
        return base

    # ================================================================
    # SAVE
    # ================================================================
    async def _save_result(self, r: dict):
        if not self.db:
            return None
        try:
            detected_ts = r.get('detected_at')
            detected_dt = datetime.fromtimestamp(detected_ts, tz=timezone.utc) if detected_ts else None
            telemetry = r.get('telemetry') or {}

            hypothesis_details = {}
            # Replicar audit_version no JSON para facilitar diagnósticos manuais via SQL (hypothesis_details->>'audit_version')
            # sem depender da coluna `audit_version` (há pipelines/históricos em que só um dos dois foi usado).
            try:
                hypothesis_details["audit_version"] = str(r.get("audit_version") or "v4.0-api")
            except Exception:
                pass
            if r.get('direction') is not None:
                hypothesis_details['direction'] = r.get('direction')
            if r.get('lay_odd') is not None:
                hypothesis_details['lay'] = {
                    'odd': r.get('lay_odd'),
                    'bookie': r.get('lay_bookie'),
                    'limit': r.get('lay_limit'),
                    'num_bk': r.get('lay_num_bk'),
                }
            if r.get('temporal'):
                hypothesis_details['temporal'] = r.get('temporal')
            if r.get('lay_temporal'):
                hypothesis_details['lay_temporal'] = r.get('lay_temporal')
            # WS series (para ws_only / ws_gate_lay). Pode vir inline.
            if r.get('ws_series'):
                hypothesis_details['ws_series'] = r.get('ws_series')
            if r.get('ws_series_meta'):
                hypothesis_details['ws_series_meta'] = r.get('ws_series_meta')
            # Gate series/meta (t0,t+5) — usado para elegibilidade e debug
            if r.get('ws_gate_series'):
                hypothesis_details['ws_gate_series'] = r.get('ws_gate_series')
            # Hint para o bridge não misturar Back vs Lay
            if r.get('exec_side_hint'):
                hypothesis_details['exec_side_hint'] = str(r.get('exec_side_hint'))
            else:
                # Fallback: inferir pelo audit_version (útil para linhas antigas onde o hint não era preenchido).
                try:
                    ver = str(r.get("audit_version") or "")
                    if ver and ("-back" in ver or "gate-back" in ver):
                        hypothesis_details["exec_side_hint"] = "Back"
                    elif ver and ("-lay" in ver or "reversal-lay" in ver or "gate-lay" in ver):
                        hypothesis_details["exec_side_hint"] = "Lay"
                except Exception:
                    pass
            # Sempre persistir causa de falha: alguns caminhos retornam status=API_FAILED mas `error=''`.
            # Nesses casos, usamos fallbacks da telemetria (back_error/lay_error) para não deixar falhas "mudas".
            if not r.get('success'):
                err = r.get('error')
                if not err:
                    try:
                        err = (telemetry or {}).get("back_error") or (telemetry or {}).get("lay_error")
                    except Exception:
                        err = None
                if not err:
                    try:
                        st = r.get("status")
                        if st and str(st) != "API_FAILED":
                            err = f"status={st}"
                    except Exception:
                        err = None
                if err:
                    hypothesis_details['api_error'] = str(err)
            finance_snapshot = self._build_finance_snapshot(r)
            if finance_snapshot:
                hypothesis_details['finance'] = finance_snapshot
            if telemetry:
                hypothesis_details['telemetry'] = telemetry

            status = r.get("status")
            if not status:
                status = "OK" if r.get("success") else "API_FAILED"

            bs_odd = r.get("bs_odd", None)
            bs_limit = r.get("bs_limit", None)
            has_bs = isinstance(bs_odd, (int, float)) and float(bs_odd) > 0

            # Compatibilidade: em ws_gate_lay, podemos ter snapshot em lay_odd/lay_limit
            # mesmo quando bs_odd não foi preenchida por algum caminho.
            if not has_bs:
                try:
                    lay_odd = r.get("lay_odd", None)
                    lay_limit = r.get("lay_limit", None)
                    if isinstance(lay_odd, (int, float)) and float(lay_odd) > 1.0:
                        bs_odd = float(lay_odd)
                        bs_limit = float(lay_limit) if isinstance(lay_limit, (int, float)) else 0.0
                        has_bs = True
                except Exception:
                    pass
            diff_pct = r.get("diff_pct") if has_bs else None
            diff_abs = None
            try:
                ws = float(r.get("ws_odd") or 0.0)
                if has_bs and ws > 0:
                    diff_abs = float(bs_odd) - ws
                    if diff_pct is None:
                        diff_pct = diff_abs / ws * 100.0
            except Exception:
                diff_abs = None
            is_valid = r.get("is_valid_opportunity")
            if is_valid is None:
                try:
                    # "Oportunidade válida" deve significar:
                    # - existe snapshot do ticket (bs_odd)
                    # - diferença BS vs WS está dentro de um range confiável (evita parse/mismatch)
                    # - e há edge mínimo (compatível com buckets do relatório)
                    dp = float(diff_pct) if (diff_pct is not None) else None
                    is_valid = bool(
                        has_bs
                        and (dp is not None)
                        and (-10.0 <= float(dp) <= 10.0)
                        and (abs(float(dp)) >= 2.0)
                    )
                except Exception:
                    is_valid = bool(has_bs)

            # ws_gate_lay: se abriu ticket com sucesso (status OK) e temos lay_odd,
            # tratamos como oportunidade executável mesmo que diff_pct esteja ausente.
            try:
                if str(r.get("audit_version") or "") == "v5.1-ws-gate-lay" and str(status) == "OK" and has_bs:
                    is_valid = True
            except Exception:
                pass

            record = BetslipAuditResult(
                hypothesis_type="H3B",
                event_id=r['event_id'],
                sport="football",
                league=r.get('league', ''),
                home_team=r['home_team'],
                away_team=r['away_team'],
                match_info=f"{r['home_team']} vs {r['away_team']}",
                match_start_time=r.get('kickoff'),
                market_type=r['market_type'],
                market_period=r.get('market_period', 'full_time'),
                line=r['line'],
                side=r['side'],
                bet_description=f"{r['market_type']} {r['line']} {r['side']}",
                websocket_odd=r['ws_odd'],
                betslip_odd=float(bs_odd) if has_bs else None,
                difference_pct=float(diff_pct) if isinstance(diff_pct, (int, float)) else None,
                difference_absolute=float(diff_abs) if isinstance(diff_abs, (int, float)) else None,
                betslip_limit=float(bs_limit) if (has_bs and isinstance(bs_limit, (int, float))) else (r.get('bs_limit', 0) if has_bs else 0),
                status=status,
                is_valid_opportunity=bool(is_valid),
                is_live=r.get('is_live'),
                reversal_direction=r.get('direction', 'up'),
                hypothesis_detected_at=detected_dt,
                lag_detection_to_click_ms=telemetry.get('queue_wait_ms', 0) + r.get('post_ms', 0),
                lag_click_to_betslip_ms=r.get('pmm_ms', 0),
                audit_total_duration_ms=telemetry.get('pipeline_total_ms_pre_db', telemetry.get('pipeline_total_ms', r.get('total_ms', 0))),
                audit_version=str(r.get("audit_version") or "v4.0-api"),
                hypothesis_details=hypothesis_details or None,
            )
            async with self.db.async_session() as session:
                session.add(record)
                await session.commit()
                return record.id
        except Exception as e:
            logger.warning(f"Erro salvando: {e}")
        return None

    # ================================================================
    # MAINTENANCE
    # ================================================================
    async def _maintenance_loop(self):
        while self.running:
            await asyncio.sleep(WS_HEALTH_INTERVAL)

            ws_age = time.time() - self._last_ws_time if self._last_ws_time > 0 else 999
            uptime = time.time() - self._start_time
            ok_count = sum(1 for r in self.results if r.get('success'))
            queue_now = self._queue_ref.qsize() if self._queue_ref else 0
            temporal_queue_now = self._temporal_queue_ref.qsize() if self._temporal_queue_ref else 0

            logger.info(
                f"[STATS] WS: {self._ws_msg_count} msgs, {self._ws_msg_count/max(1,uptime):.1f}/s, "
                f"last {ws_age:.0f}s | "
                f"Fila T+0: now={queue_now} max={self.max_queue_depth_observed} | "
                f"Fila temporal: now={temporal_queue_now} max={self.max_temporal_queue_depth_observed} | "
                f"drops: fullq={self.dropped_full_queue} staleq={self.dropped_stale_queue_wait} | "
                f"Auditorias: {len(self.results)} (OK:{ok_count}) | "
                f"H3B: {self.h3b_detected} | Erros: {self.total_errors} | "
                f"ws_buf_drop={self._ws_messages_dropped}")

            # Mitigação OOM: monitora RSS e força restart limpo antes do killer.
            try:
                thr = float(os.getenv("AUDIT_RSS_RESTART_MIB", "1500") or 1500.0)
            except Exception:
                thr = 1500.0
            try:
                rss = self._rss_mib()
                if rss is not None and thr > 0 and float(rss) >= float(thr):
                    await self._emit_runtime_event(
                        kind="RSS_RESTART",
                        level="ERROR",
                        message=f"rss_mib={float(rss):.1f} >= thr={float(thr):.1f}",
                        meta={},
                        min_interval_sec=300.0,
                        try_db=True,
                    )
                    await self.shutdown("rss_exceeded")
                    return
            except Exception:
                pass

            # GC leve dos caches (evita crescimento indefinido)
            try:
                # audited TTL cleanup
                if self._audited_ttl_sec > 0 and self._audited_ts:
                    cutoff = time.time() - float(self._audited_ttl_sec)
                    if len(self._audited_ts) > 20000:
                        self._audited_ts = {k: v for k, v in self._audited_ts.items() if float(v) >= cutoff}
                # ws_odds_state TTL cleanup
                ttl_state = float(os.getenv("AUDIT_WS_STATE_TTL_SEC", "3600") or 3600.0)
                if ttl_state > 0 and self._ws_odds_state:
                    cutoff2 = time.time() - ttl_state
                    if len(self._ws_odds_state) > 200000:
                        self._ws_odds_state = {k: v for k, v in self._ws_odds_state.items() if float(v.get("ts") or 0.0) >= cutoff2}
                # events_info max cap (best-effort)
                mx_events = int(float(os.getenv("AUDIT_EVENTS_INFO_MAX", "50000") or 50000))
                if mx_events > 0 and len(self._events_info) > mx_events:
                    # remove arbitrary oldest-ish keys by insertion order is not guaranteed; do a cheap trim
                    for k in list(self._events_info.keys())[: max(1, len(self._events_info) - mx_events)]:
                        try:
                            self._events_info.pop(k, None)
                        except Exception:
                            pass
                gc.collect()
            except Exception:
                pass

            if ws_age > WS_RELOAD_INTERVAL:
                logger.warning("WS morto, recarregando...")
                try:
                    await self._emit_runtime_event(
                        kind="WS_STALE_RELOAD",
                        level="WARN",
                        message=f"ws_age_s={float(ws_age):.0f}",
                        meta={},
                        min_interval_sec=300.0,
                        try_db=True,
                    )
                except Exception:
                    pass
                try:
                    await self.scraper._page.reload()
                    await self.scraper._page.wait_for_load_state("domcontentloaded")
                    await asyncio.sleep(3)
                except Exception as e:
                    logger.error(f"Reload falhou: {e}")
                    try:
                        await self._emit_runtime_event(
                            kind="WS_RELOAD_FAIL",
                            level="ERROR",
                            message=str(e)[:220],
                            meta={},
                            min_interval_sec=180.0,
                            try_db=True,
                        )
                    except Exception:
                        pass
                    self.consecutive_errors += 1
                    if self.consecutive_errors >= 10:
                        logger.error("10 erros consecutivos, parando")
                        try:
                            await self._emit_runtime_event(
                                kind="FATAL_CONSECUTIVE_ERRORS",
                                level="ERROR",
                                message="consecutive_errors>=10",
                                meta={},
                                min_interval_sec=300.0,
                                try_db=True,
                            )
                        except Exception:
                            pass
                        self.running = False

    def _log_stats(self):
        ok = [r for r in self.results if r.get('success')]
        if not ok:
            return
        lags = [r['total_ms'] for r in ok]
        diffs = [r.get('diff_pct') for r in ok if isinstance(r.get('diff_pct'), (int, float))]
        queue_ms = [r.get('telemetry', {}).get('queue_wait_ms', 0) for r in ok]
        post_ms = [r.get('telemetry', {}).get('back_post_ms', r.get('post_ms', 0)) for r in ok]
        pmm_ms = [r.get('telemetry', {}).get('back_pmm_ms', r.get('pmm_ms', 0)) for r in ok]
        lay_post_ms = [r.get('telemetry', {}).get('lay_post_ms', 0) for r in ok]
        lay_pmm_ms = [r.get('telemetry', {}).get('lay_pmm_ms', 0) for r in ok]
        temporal_ms = [r.get('telemetry', {}).get('temporal_total_ms', 0) for r in ok]
        db_ms = [r.get('telemetry', {}).get('db_save_ms', 0) for r in ok]
        pipeline_ms = [r.get('telemetry', {}).get('pipeline_total_ms', r['total_ms']) for r in ok]
        qdepth_enq = [r.get('telemetry', {}).get('queue_depth_at_enqueue') for r in ok if r.get('telemetry', {}).get('queue_depth_at_enqueue') is not None]
        qdepth_deq = [r.get('telemetry', {}).get('queue_depth_after_dequeue') for r in ok if r.get('telemetry', {}).get('queue_depth_after_dequeue') is not None]
        logger.info(f"{'=' * 50}")
        logger.info(f"STATS — {len(self.results)} auditorias ({len(ok)} OK)")
        logger.info(f"  Lag: min={min(lags)}ms med={sorted(lags)[len(lags)//2]}ms avg={sum(lags)//len(lags)}ms max={max(lags)}ms")
        if diffs:
            logger.info(f"  Diff: avg={sum(diffs)/len(diffs):+.2f}% med={sorted(diffs)[len(diffs)//2]:+.2f}%")
        else:
            logger.info("  Diff: (sem BS) — modo WS-only")
        if qdepth_enq:
            logger.info(
                f"  Fila avg(itens): enq={self._avg(qdepth_enq):.2f} "
                f"deq={self._avg(qdepth_deq):.2f}"
            )
        logger.info(
            "  Etapas avg(ms): "
            f"fila={int(self._avg(queue_ms))} "
            f"post={int(self._avg(post_ms))} "
            f"pmm={int(self._avg(pmm_ms))} "
            f"lay_post={int(self._avg(lay_post_ms))} "
            f"lay_pmm={int(self._avg(lay_pmm_ms))} "
            f"temporal={int(self._avg(temporal_ms))} "
            f"db={int(self._avg(db_ms))} "
            f"pipeline={int(self._avg(pipeline_ms))}"
        )
        if self.mode in ("ws_gate_lay", "gate_lay", "gate"):
            logger.info(
                "  Gate (5s drop -> open LAY): "
                f"seen={self.gate_seen} elig={self.gate_eligible} not_elig={self.gate_not_eligible} "
                f"ws_missing={self.gate_ws_missing} "
                f"blocked_cap={self.gate_blocked_cap} blocked_backoff={self.gate_blocked_backoff} "
                f"open_attempts={self.gate_open_attempts} open_ok={self.gate_open_success} open_fail={self.gate_open_failed}"
            )
        logger.info(f"{'=' * 50}")

    def _print_summary(self):
        ok = [r for r in self.results if r.get('success')]
        fail = [r for r in self.results if not r.get('success')]
        print(f"\n{'=' * 60}")
        print(f"RESUMO — {len(self.results)} auditorias ({len(ok)} OK, {len(fail)} FAIL)")
        if ok:
            lags = [r['total_ms'] for r in ok]
            diffs = [r.get('diff_pct') for r in ok if isinstance(r.get('diff_pct'), (int, float))]
            queue_ms = [r.get('telemetry', {}).get('queue_wait_ms', 0) for r in ok]
            post_ms = [r.get('telemetry', {}).get('back_post_ms', r.get('post_ms', 0)) for r in ok]
            pmm_ms = [r.get('telemetry', {}).get('back_pmm_ms', r.get('pmm_ms', 0)) for r in ok]
            temporal_ms = [r.get('telemetry', {}).get('temporal_total_ms', 0) for r in ok]
            pipeline_ms = [r.get('telemetry', {}).get('pipeline_total_ms', r['total_ms']) for r in ok]
            qdepth_enq = [r.get('telemetry', {}).get('queue_depth_at_enqueue') for r in ok if r.get('telemetry', {}).get('queue_depth_at_enqueue') is not None]
            qdepth_deq = [r.get('telemetry', {}).get('queue_depth_after_dequeue') for r in ok if r.get('telemetry', {}).get('queue_depth_after_dequeue') is not None]
            print(f"  Lag: min={min(lags)} med={sorted(lags)[len(lags)//2]} max={max(lags)}ms")
            if diffs:
                print(f"  Diff: avg={sum(diffs)/len(diffs):+.2f}% med={sorted(diffs)[len(diffs)//2]:+.2f}%")
            else:
                print("  Diff: (sem BS) — modo WS-only")
            if qdepth_enq:
                print(f"  Fila avg(itens): enq={self._avg(qdepth_enq):.2f} deq={self._avg(qdepth_deq):.2f}")
            print(
                f"  Etapas avg(ms): fila={int(self._avg(queue_ms))} "
                f"post={int(self._avg(post_ms))} pmm={int(self._avg(pmm_ms))} "
                f"temporal={int(self._avg(temporal_ms))} pipeline={int(self._avg(pipeline_ms))}"
            )
        print(f"{'=' * 60}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-audits", type=int, default=0, help="0=infinito")
    parser.add_argument("--direction", choices=["up", "down", "all"], default="up")
    parser.add_argument("--no-db", action="store_true")
    default_mode = (os.getenv("AUDIT_MODE") or getattr(settings, "audit_mode", None) or "api").strip() or "api"
    default_offsets = os.getenv("AUDIT_WS_SAMPLE_OFFSETS_SEC") or getattr(settings, "audit_ws_sample_offsets_sec", None) or "0,3,6,9,12,15,18,21,24,27,30"
    parser.add_argument(
        "--mode",
        choices=["api", "ws_only", "ws_vs_bs", "ws_gate_lay", "ws_reversal_lay", "ws_gate_back"],
        default=default_mode,
        help="Modo: api (WS+BS), ws_only (só WS), ws_vs_bs (comparativo), ws_gate_lay (WS gate + abre LAY sob cap), ws_reversal_lay (abre LAY no reversal), ws_gate_back (WS gate + marca BACK válido).",
    )
    parser.add_argument(
        "--ws-sample-offsets-sec",
        default=str(default_offsets),
        help="Offsets (segundos) para amostragem WS (ex: '0,3,6,...,30').",
    )
    parser.add_argument(
        "--api-sides",
        default=(os.getenv("AUDIT_API_SIDES", "both") or "both").strip(),
        choices=["back", "lay", "both"],
        help="Somente para --mode api: quais lados chamar (back, lay, both). Default: both.",
    )
    # Gate: H3B UP + queda em 5s -> abre ticket LAY
    parser.add_argument("--gate-drop-offset-sec", type=float, default=float(os.getenv("GATE_DROP_OFFSET_SEC", "5")), help="Gate: offset (s) para comparar WS(t+offset) vs WS(t0). Default=5.")
    parser.add_argument("--gate-drop-ratio", type=float, default=float(os.getenv("GATE_DROP_RATIO", "0.98")), help="Gate: condição WS(t+offset) < ratio * WS(t0). Default=0.98 (queda >2%).")
    # Gate: H3B UP + alta em 5s -> oportunidade BACK via WS
    parser.add_argument("--gate-rise-offset-sec", type=float, default=float(os.getenv("GATE_RISE_OFFSET_SEC", "5")), help="Gate Back: offset (s) para comparar WS(t+offset) vs WS(t0). Default=5.")
    parser.add_argument("--gate-rise-ratio", type=float, default=float(os.getenv("GATE_RISE_RATIO", "1.02")), help="Gate Back: condição WS(t+offset) >= ratio * WS(t0). Default=1.02 (alta >=2%).")
    parser.add_argument("--gate-open-window-sec", type=int, default=int(os.getenv("GATE_OPEN_WINDOW_SEC", "300")), help="Cap: janela (s) para contar aberturas (POST /v1/betslips/). Default=300 (5 min).")
    parser.add_argument("--gate-open-max", type=int, default=int(os.getenv("GATE_OPEN_MAX", "3")), help="Cap: máximo de aberturas por janela. Default=3 por 5 min (conservador).")
    parser.add_argument("--gate-max-late-sec", type=float, default=float(os.getenv("GATE_MAX_LATE_SEC", "2.5")), help="Gate: tolerância de atraso (s). Se o worker começar >max_late após o t+offset, marca como stale e não abre ticket. Default=2.5.")
    parser.add_argument("--gate-lay-refresh", action="store_true", help="Se ligado, após abrir ticket LAY coleta lay_temporal via refresh (deferred).")
    parser.add_argument("--gate-lay-refresh-times-sec", type=str, default=os.getenv("GATE_LAY_REFRESH_TIMES_SEC", "0,5,10,15,20"), help="Tempos (s) para refresh do LAY após abrir ticket. Default=0,5,10,15,20.")
    parser.add_argument(
        "--executor-workers",
        type=int,
        default=int(os.getenv("AUDIT_EXECUTOR_WORKERS", "4")),
        help="Quantidade de workers paralelos do executor API",
    )
    parser.add_argument(
        "--temporal-workers",
        type=int,
        default=int(os.getenv("AUDIT_TEMPORAL_WORKERS", "2")),
        help="Quantidade de workers paralelos para monitoramento temporal assíncrono",
    )
    parser.add_argument(
        "--max-queue-depth",
        type=int,
        default=int(os.getenv("AUDIT_MAX_QUEUE_DEPTH", "50")),
        help="Tamanho máximo da fila T+0 (0=infinito). Acima disso, eventos são descartados.",
    )
    parser.add_argument(
        "--max-queue-wait-ms",
        type=int,
        default=int(os.getenv("AUDIT_MAX_QUEUE_WAIT_MS", "5000")),
        help="Se queue_wait_ms exceder este valor, descarta o evento sem chamar a API (0=desliga).",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr,
               format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level:<7}</level> | <level>{message}</level>",
               level="INFO",
               filter=lambda r: "H6:" not in r["message"] and "H1:" not in r["message"])
    logger.add("logs/audit_api_{time:YYYY-MM-DD}.log", rotation="00:00", retention="60 days", level="DEBUG")

    ws_offsets = H3bApiAudit._parse_offsets(
        str(args.ws_sample_offsets_sec),
        default=[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
    )

    gate_refresh_times = H3bApiAudit._parse_offsets(
        str(getattr(args, "gate_lay_refresh_times_sec", "")),
        default=[0, 5, 10, 15, 20],
    )

    audit = H3bApiAudit(
        num_audits=args.num_audits,
        direction=args.direction,
        save_to_db=not args.no_db,
        executor_workers=args.executor_workers,
        temporal_workers=args.temporal_workers,
        max_queue_depth=args.max_queue_depth,
        max_queue_wait_ms=args.max_queue_wait_ms,
        mode=str(args.mode),
        ws_sample_offsets_sec=ws_offsets,
        gate_drop_offset_sec=float(getattr(args, "gate_drop_offset_sec", 5.0)),
        gate_drop_ratio=float(getattr(args, "gate_drop_ratio", 0.98)),
        gate_rise_offset_sec=float(getattr(args, "gate_rise_offset_sec", 5.0)),
        gate_rise_ratio=float(getattr(args, "gate_rise_ratio", 1.02)),
        gate_open_window_sec=int(getattr(args, "gate_open_window_sec", 300)),
        gate_open_max=int(getattr(args, "gate_open_max", 3)),
        gate_max_late_sec=float(getattr(args, "gate_max_late_sec", 2.5)),
        gate_lay_refresh=bool(getattr(args, "gate_lay_refresh", False)),
        gate_lay_refresh_times_sec=gate_refresh_times,
        api_sides=str(getattr(args, "api_sides", "both")),
    )

    # SIGTERM/SIGINT: encerra gracioso (evita chrome órfão e stop timeout)
    try:
        loop = asyncio.get_running_loop()

        async def _on_signal(sig_name: str):
            await audit.shutdown(f"signal:{sig_name}")

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(_on_signal(s.name)))
            except NotImplementedError:
                pass
    except Exception:
        pass

    try:
        await audit.run()
    except Exception:
        logger.exception("Falha fatal no audit_h3b_api")
        try:
            await audit.shutdown("fatal_exception")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    asyncio.run(main())
