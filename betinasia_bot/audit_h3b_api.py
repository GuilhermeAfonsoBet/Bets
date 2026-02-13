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
from datetime import datetime, timezone
from typing import Optional, Dict, List
from dataclasses import dataclass
from loguru import logger

sys.path.insert(0, '.')

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


class H3bApiAudit:

    def __init__(
        self,
        num_audits: int = 0,
        direction: str = "up",
        save_to_db: bool = True,
        executor_workers: int = 4,
        temporal_workers: int = 2,
    ):
        self.num_audits = num_audits
        self.direction = direction
        self.save_to_db = save_to_db
        self.executor_workers = max(1, int(executor_workers))
        self.temporal_workers = max(0, int(temporal_workers))

        self.scraper: Optional[BetinAsiaScraper] = None
        self.api_client: Optional[ApiBetslipClient] = None
        self.db: Optional[Database] = None

        # WS
        self._ws_messages: List[str] = []
        self._ws_msg_count: int = 0
        self._last_ws_time: float = 0
        self._start_time: float = 0
        self._events_info: Dict[str, dict] = {}

        # Detector
        self.detector = HypothesisDetector()

        # Stats
        self.results: List[dict] = []
        self.events_processed: int = 0
        self.h3b_detected: int = 0
        self.total_errors: int = 0
        self.consecutive_errors: int = 0
        self.running = True
        self.telemetry_file = "logs/audit_api_telemetry.jsonl"
        self.max_queue_depth_observed = 0
        self._queue_ref: Optional[asyncio.Queue] = None
        self.max_temporal_queue_depth_observed = 0
        self._temporal_queue_ref: Optional[asyncio.Queue] = None

    @staticmethod
    def _avg(values: List[float]) -> float:
        return (sum(values) / len(values)) if values else 0.0

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
            "status": "OK" if result.get("success") else "FAIL",
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
        logger.info("AUDITORIA H3B VIA API (~2-3s)")
        logger.info("=" * 60)

        signal.signal(signal.SIGTERM, lambda s, f: setattr(self, 'running', False))
        signal.signal(signal.SIGINT, lambda s, f: setattr(self, 'running', False))

        # Browser (necessário para WS e fetch autenticado)
        self.scraper = BetinAsiaScraper()
        await self.scraper.start()
        await self.scraper.login()
        logger.info("Login OK")

        # API client (usa o page do scraper)
        page = self.scraper._page
        self.api_client = ApiBetslipClient(page)

        # WS listener único (para odds + PMM + betslip)
        def on_ws(ws):
            def on_frame(data):
                data_str = str(data)
                self._ws_messages.append(data_str)
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

        # DB
        if self.save_to_db:
            self.db = Database()
            await self.db.connect()
            try:
                async with self.db.engine.begin() as conn:
                    await conn.execute(text(
                        "ALTER TABLE betslip_audit_results ADD COLUMN IF NOT EXISTS is_live BOOLEAN"))
            except:
                pass
            logger.info("Banco conectado")

        return True

    async def run(self):
        ok = await self.start()
        if not ok:
            return

        audit_queue = asyncio.Queue()
        self._queue_ref = audit_queue
        temporal_queue = asyncio.Queue()
        self._temporal_queue_ref = temporal_queue
        audited = set()

        tasks = [asyncio.create_task(self._monitor_loop(audit_queue, audited))]
        for wid in range(1, self.executor_workers + 1):
            tasks.append(asyncio.create_task(self._executor_loop(audit_queue, worker_id=wid)))
        for twid in range(1, self.temporal_workers + 1):
            tasks.append(asyncio.create_task(self._temporal_loop(temporal_queue, worker_id=twid)))
        tasks.append(asyncio.create_task(self._maintenance_loop()))
        logger.info(f"Executores T+0 ativos: {self.executor_workers} | Temporal workers: {self.temporal_workers}")

        try:
            while self.running:
                if self.num_audits > 0 and len(self.results) >= self.num_audits:
                    break
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            for t in tasks:
                t.cancel()

        self._print_summary()
        if self.scraper:
            await self.scraper.close()
        if self.db:
            await self.db.close()

    # ================================================================
    # MONITOR
    # ================================================================
    async def _monitor_loop(self, queue: asyncio.Queue, audited: set):
        logger.info("Monitor iniciado")
        last_idx = 0

        while self.running:
            new = self._ws_messages[last_idx:]
            last_idx = len(self._ws_messages)

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
                                    self._process_odds(eid, msg_data['ah'], 'AH', 'full_time',
                                                       queue, audited)
                                if 'ahou' in msg_data:
                                    self._process_odds(eid, msg_data['ahou'], 'OU', 'full_time',
                                                       queue, audited, over_under=True)
                                if 'ah_ht' in msg_data:
                                    self._process_odds(eid, msg_data['ah_ht'], 'AH', 'half_time',
                                                       queue, audited)
                                if 'ou_ht' in msg_data:
                                    self._process_odds(eid, msg_data['ou_ht'], 'OU', 'half_time',
                                                       queue, audited, over_under=True)
                except:
                    continue

            if self.events_processed > 0 and self.events_processed % 500 == 0:
                logger.info(f"Processados: {self.events_processed} | H3B: {self.h3b_detected} | "
                            f"Auditados: {len(self.results)} | WS: {self._ws_msg_count}")

    def _process_odds(self, event_id, odds_data, market_type, period,
                      queue, audited, over_under=False):
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
                if audit_key in audited:
                    continue

                is_live = kickoff <= datetime.now(timezone.utc) if kickoff else None

                queue_depth_at_enqueue = queue.qsize()
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
                    'direction': h3b.direction_after,
                    'detected_at': time.time(),
                    'queue_depth_at_enqueue': queue_depth_at_enqueue,
                })
                self.max_queue_depth_observed = max(self.max_queue_depth_observed, queue.qsize())
                audited.add(audit_key)

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
            self.results.append(result)

            temporal_refs = result.get('_temporal_refs')
            if defer_temporal and record_id and temporal_refs and self._temporal_queue_ref:
                temporal_job = {
                    'record_id': record_id,
                    'event_id': result.get('event_id'),
                    'home_team': result.get('home_team'),
                    'away_team': result.get('away_team'),
                    'ws_odd': temporal_refs.get('ws_odd'),
                    'back_betslip_id': temporal_refs.get('back_betslip_id', ''),
                    'lay_betslip_id': temporal_refs.get('lay_betslip_id', ''),
                    'telemetry_base': dict(telemetry),
                    'queued_at': time.time(),
                }
                self._temporal_queue_ref.put_nowait(temporal_job)
                self.max_temporal_queue_depth_observed = max(
                    self.max_temporal_queue_depth_observed,
                    self._temporal_queue_ref.qsize()
                )

            # Log
            live = "LIVE" if result.get('is_live') else "PRE" if result.get('is_live') is not None else "?"
            if result.get('success'):
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
            else:
                self.total_errors += 1
                logger.warning(
                    f"[FAIL][{live}] {result['home_team']} vs {result['away_team']} | "
                    f"{result['market_type']} {result['line']} {result['side']} | "
                    f"ws={result['ws_odd']:.3f} | err={result.get('error','')} | "
                    f"lag={result['total_ms']}ms | {len(self.results)}")

            if len(self.results) % STATS_INTERVAL == 0:
                self._log_stats()

    async def _collect_temporal_series(self, ws_odd: float, back_betslip_id: str, lay_betslip_id: str):
        refresh_times = [3, 6, 10, 15, 20]
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
                point_meta = {
                    'target_s': target_t,
                    'actual_s': actual_t,
                    'refresh_ms': refresh_ms,
                    'back_ok': False,
                    'lay_ok': False,
                }

                for label, ref in zip(labels, refresh_results):
                    if isinstance(ref, Exception):
                        logger.debug(f"Refresh {label} t+{target_t} falhou: {ref}")
                        continue
                    if not ref or not ref.success:
                        continue

                    if label == "back":
                        ref_diff = ((ref.best_odd - ws_odd) / ws_odd) * 100 if ws_odd else 0
                        back_temporal.append({
                            't': actual_t,
                            'bs_odd': ref.best_odd,
                            'diff_pct': round(ref_diff, 3),
                            'bookie': ref.best_bookie,
                            'limit': ref.best_limit,
                            'num_bk': ref.num_bookmakers,
                        })
                        point_meta['back_ok'] = True
                    else:
                        lay_ref = _extract_lay_snapshot(ref)
                        if lay_ref:
                            lay_diff = ((lay_ref['odd'] - ws_odd) / ws_odd) * 100 if ws_odd else 0
                            lay_temporal.append({
                                't': actual_t,
                                'lay_odd': lay_ref['odd'],
                                'diff_pct': round(lay_diff, 3),
                                'bookie': lay_ref['bookie'],
                                'limit': lay_ref['limit'],
                                'num_bk': lay_ref['num_bk'],
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
        return back_temporal, lay_temporal, telemetry_patch

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
                back_temporal, lay_temporal, telemetry_patch = await self._collect_temporal_series(
                    ws_odd=job.get('ws_odd', 0) or 0,
                    back_betslip_id=job.get('back_betslip_id', ''),
                    lay_betslip_id=job.get('lay_betslip_id', ''),
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

        # Constrói bet_types
        t_build = time.time()
        back_bet_type = ApiBetslipClient.build_bet_type(
            market_type=h3b['market_type'],
            side=h3b['side'],
            line=h3b['line'],
        )
        lay_bet_type = ApiBetslipClient.build_lay_bet_type(
            market_type=h3b['market_type'],
            side=h3b['side'],
            line=h3b['line'],
        )
        build_bet_type_ms = int((time.time() - t_build) * 1000)

        # === T+0: BACK + LAY SIMULTÂNEOS ===
        t_parallel = time.time()
        back_task = self.api_client.get_betslip_odds(
            event_id=h3b['event_id'],
            bet_type=back_bet_type,
        )
        lay_task = self.api_client.get_betslip_odds(
            event_id=h3b['event_id'],
            bet_type=lay_bet_type,
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

        telemetry = {
            'queue_wait_ms': queue_wait_ms,
            'queue_depth_at_enqueue': queue_depth_at_enqueue,
            'queue_depth_after_dequeue': queue_depth_after_dequeue,
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
            'post_ms': back_post_ms,
            'pmm_ms': back_pmm_ms,
        }

        if not back_result or not back_result.success:
            lay_snapshot = _extract_lay_snapshot(lay_result)
            if lay_snapshot:
                base.update({
                    'lay_odd': lay_snapshot['odd'],
                    'lay_bookie': lay_snapshot['bookie'],
                    'lay_limit': lay_snapshot['limit'],
                    'lay_num_bk': lay_snapshot['num_bk'],
                })

            back_err = back_result.error if back_result else 'Back failed'
            if lay_result and not lay_result.success and lay_result.error:
                back_err = f"{back_err} | lay={lay_result.error}"

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
                'bs_odd': 0, 'bs_limit': 0, 'num_bk': 0, 'diff_pct': 0,
                'error': back_err,
                'total_ms': end_to_end_ms,
                'telemetry': telemetry,
            })
            return base

        ws_odd = h3b['websocket_odd']
        diff = ((back_result.best_odd - ws_odd) / ws_odd) * 100 if ws_odd else 0

        base.update({
            'success': True,
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
            back_temporal, lay_temporal, telemetry_patch = await self._collect_temporal_series(
                ws_odd=ws_odd,
                back_betslip_id=back_betslip_id,
                lay_betslip_id=lay_betslip_id,
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
                base['_temporal_refs'] = {
                    'ws_odd': ws_odd,
                    'back_betslip_id': back_betslip_id,
                    'lay_betslip_id': lay_betslip_id,
                }

        end_to_end_ms = int((time.time() - detected_at) * 1000)
        telemetry['execution_ms'] = int((time.time() - execution_start) * 1000)
        telemetry['end_to_end_ms'] = end_to_end_ms
        known_ms = telemetry['queue_wait_ms'] + telemetry['parallel_fetch_ms'] + telemetry['temporal_total_ms']
        telemetry['pipeline_overhead_ms'] = max(0, end_to_end_ms - known_ms)

        base['total_ms'] = end_to_end_ms
        base['telemetry'] = telemetry
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
            if (not r.get('success')) and r.get('error'):
                hypothesis_details['api_error'] = r.get('error')
            if telemetry:
                hypothesis_details['telemetry'] = telemetry

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
                betslip_odd=r.get('bs_odd') if r.get('success') else None,
                difference_pct=r.get('diff_pct') if r.get('success') else None,
                difference_absolute=(r['bs_odd'] - r['ws_odd']) if r.get('success') else None,
                betslip_limit=r.get('bs_limit', 0),
                status="OK" if r.get('success') else "API_FAILED",
                is_valid_opportunity=r.get('success', False),
                is_live=r.get('is_live'),
                reversal_direction=r.get('direction', 'up'),
                hypothesis_detected_at=detected_dt,
                lag_detection_to_click_ms=telemetry.get('queue_wait_ms', 0) + r.get('post_ms', 0),
                lag_click_to_betslip_ms=r.get('pmm_ms', 0),
                audit_total_duration_ms=telemetry.get('pipeline_total_ms_pre_db', telemetry.get('pipeline_total_ms', r.get('total_ms', 0))),
                audit_version="v4.0-api",
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
                f"Auditorias: {len(self.results)} (OK:{ok_count}) | "
                f"H3B: {self.h3b_detected} | Erros: {self.total_errors}")

            if ws_age > WS_RELOAD_INTERVAL:
                logger.warning("WS morto, recarregando...")
                try:
                    await self.scraper._page.reload()
                    await self.scraper._page.wait_for_load_state("domcontentloaded")
                    await asyncio.sleep(3)
                except Exception as e:
                    logger.error(f"Reload falhou: {e}")
                    self.consecutive_errors += 1
                    if self.consecutive_errors >= 10:
                        logger.error("10 erros consecutivos, parando")
                        self.running = False

    def _log_stats(self):
        ok = [r for r in self.results if r.get('success')]
        if not ok:
            return
        lags = [r['total_ms'] for r in ok]
        diffs = [r['diff_pct'] for r in ok]
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
        logger.info(f"  Diff: avg={sum(diffs)/len(diffs):+.2f}% med={sorted(diffs)[len(diffs)//2]:+.2f}%")
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
        logger.info(f"{'=' * 50}")

    def _print_summary(self):
        ok = [r for r in self.results if r.get('success')]
        fail = [r for r in self.results if not r.get('success')]
        print(f"\n{'=' * 60}")
        print(f"RESUMO — {len(self.results)} auditorias ({len(ok)} OK, {len(fail)} FAIL)")
        if ok:
            lags = [r['total_ms'] for r in ok]
            diffs = [r['diff_pct'] for r in ok]
            queue_ms = [r.get('telemetry', {}).get('queue_wait_ms', 0) for r in ok]
            post_ms = [r.get('telemetry', {}).get('back_post_ms', r.get('post_ms', 0)) for r in ok]
            pmm_ms = [r.get('telemetry', {}).get('back_pmm_ms', r.get('pmm_ms', 0)) for r in ok]
            temporal_ms = [r.get('telemetry', {}).get('temporal_total_ms', 0) for r in ok]
            pipeline_ms = [r.get('telemetry', {}).get('pipeline_total_ms', r['total_ms']) for r in ok]
            qdepth_enq = [r.get('telemetry', {}).get('queue_depth_at_enqueue') for r in ok if r.get('telemetry', {}).get('queue_depth_at_enqueue') is not None]
            qdepth_deq = [r.get('telemetry', {}).get('queue_depth_after_dequeue') for r in ok if r.get('telemetry', {}).get('queue_depth_after_dequeue') is not None]
            print(f"  Lag: min={min(lags)} med={sorted(lags)[len(lags)//2]} max={max(lags)}ms")
            print(f"  Diff: avg={sum(diffs)/len(diffs):+.2f}% med={sorted(diffs)[len(diffs)//2]:+.2f}%")
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
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr,
               format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level:<7}</level> | <level>{message}</level>",
               level="INFO",
               filter=lambda r: "H6:" not in r["message"] and "H1:" not in r["message"])
    logger.add("logs/audit_api_{time:YYYY-MM-DD}.log", rotation="00:00", retention="60 days", level="DEBUG")

    audit = H3bApiAudit(
        num_audits=args.num_audits,
        direction=args.direction,
        save_to_db=not args.no_db,
        executor_workers=args.executor_workers,
        temporal_workers=args.temporal_workers,
    )
    await audit.run()


if __name__ == "__main__":
    asyncio.run(main())
