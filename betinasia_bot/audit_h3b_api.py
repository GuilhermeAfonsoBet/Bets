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

    def __init__(self, num_audits: int = 0, direction: str = "up", save_to_db: bool = True):
        self.num_audits = num_audits
        self.direction = direction
        self.save_to_db = save_to_db

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
        audited = set()

        tasks = [
            asyncio.create_task(self._monitor_loop(audit_queue, audited)),
            asyncio.create_task(self._executor_loop(audit_queue)),
            asyncio.create_task(self._maintenance_loop()),
        ]

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
                })
                audited.add(audit_key)

    # ================================================================
    # EXECUTOR (via API, não DOM)
    # ================================================================
    async def _executor_loop(self, queue: asyncio.Queue):
        logger.info("Executor API iniciado")

        while self.running:
            if self.num_audits > 0 and len(self.results) >= self.num_audits:
                break
            try:
                h3b = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            result = await self._execute_api_audit(h3b)
            self.results.append(result)

            if self.save_to_db:
                await self._save_result(result)

            # Log
            live = "LIVE" if result.get('is_live') else "PRE" if result.get('is_live') is not None else "?"
            if result.get('success'):
                self.consecutive_errors = 0
                lay_str = ""
                if result.get('lay_odd'):
                    lay_str = f" lay={result['lay_odd']:.3f}({result.get('lay_bookie','')})"
                logger.info(
                    f"[OK][{live}] {result['home_team']} vs {result['away_team']} | "
                    f"{result['market_type']} {result['line']} {result['side']} | "
                    f"ws={result['ws_odd']:.3f} bs={result['bs_odd']:.3f} "
                    f"diff={result['diff_pct']:+.2f}% lim=${result['bs_limit']:,.0f} "
                    f"({result['num_bk']} bk){lay_str} | "
                    f"lag={result['total_ms']}ms | "
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

    async def _execute_api_audit(self, h3b: dict) -> dict:
        detected_at = h3b['detected_at']
        t0 = time.time()

        # Constrói bet_types
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

        # === T+0: BACK + LAY SIMULTÂNEOS ===
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
        
        # Trata exceções
        if isinstance(back_result, Exception):
            back_result = None
        if isinstance(lay_result, Exception):
            lay_result = None

        total_ms = int((time.time() - detected_at) * 1000)
        post_ms = back_result.request_time_ms if back_result else 0
        pmm_ms = (back_result.total_time_ms - post_ms) if back_result else 0

        base = {
            'event_id': h3b['event_id'],
            'home_team': h3b['home_team'],
            'away_team': h3b['away_team'],
            'league': h3b.get('league', ''),
            'market_type': h3b['market_type'],
            'market_period': h3b.get('market_period', 'full_time'),
            'line': h3b['line'],
            'side': h3b['side'],
            'ws_odd': h3b['websocket_odd'],
            'is_live': h3b.get('is_live'),
            'direction': h3b.get('direction', 'up'),
            'total_ms': total_ms,
            'post_ms': post_ms,
            'pmm_ms': pmm_ms,
        }

        if not back_result or not back_result.success:
            base.update({
                'success': False,
                'bs_odd': 0, 'bs_limit': 0, 'num_bk': 0, 'diff_pct': 0,
                'error': back_result.error if back_result else 'Back failed',
            })
            return base

        ws_odd = h3b['websocket_odd']
        diff = ((back_result.best_odd - ws_odd) / ws_odd) * 100

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
        if lay_result and lay_result.success:
            lay_odds = sorted([b.best_price for b in lay_result.bookmakers if b.best_price > 0])
            if lay_odds:
                base['lay_odd'] = lay_odds[0]
                base['lay_bookie'] = next(b.bookie for b in lay_result.bookmakers if b.best_price == lay_odds[0])
                base['lay_limit'] = next(b.max_stake for b in lay_result.bookmakers if b.best_price == lay_odds[0])
                base['lay_num_bk'] = len(lay_odds)

        # === MONITORAMENTO TEMPORAL (refresh a t+3, t+6, t+10, t+15, t+20) ===
        refresh_times = [3, 6, 10, 15, 20]
        temporal = []
        betslip_id = back_result.betslip_id

        if betslip_id and back_result.success:
            t_start = time.time()
            for target_t in refresh_times:
                # Espera até o momento certo
                elapsed = time.time() - t_start
                wait = target_t - elapsed
                if wait > 0:
                    await asyncio.sleep(wait)

                # Refresh
                try:
                    ref = await self.api_client.refresh_betslip(betslip_id)
                    actual_t = time.time() - t_start
                    if ref.success:
                        ref_diff = ((ref.best_odd - ws_odd) / ws_odd) * 100
                        temporal.append({
                            't': round(actual_t, 1),
                            'bs_odd': ref.best_odd,
                            'diff_pct': round(ref_diff, 3),
                            'bookie': ref.best_bookie,
                            'limit': ref.best_limit,
                            'num_bk': ref.num_bookmakers,
                        })
                except Exception as e:
                    logger.debug(f"Refresh t+{target_t} falhou: {e}")

        if temporal:
            base['temporal'] = temporal
            # Log evolução
            evol = " → ".join([f"t+{t['t']:.0f}s:{t['bs_odd']:.3f}({t['diff_pct']:+.1f}%)" for t in temporal])
            logger.info(f"  Temporal: {evol}")

        base['total_ms'] = int((time.time() - detected_at) * 1000)
        return base

    # ================================================================
    # SAVE
    # ================================================================
    async def _save_result(self, r: dict):
        if not self.db:
            return
        try:
            record = BetslipAuditResult(
                hypothesis_type="H3B",
                event_id=r['event_id'],
                sport="football",
                league=r.get('league', ''),
                home_team=r['home_team'],
                away_team=r['away_team'],
                match_info=f"{r['home_team']} vs {r['away_team']}",
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
                lag_detection_to_click_ms=r.get('post_ms', 0),
                lag_click_to_betslip_ms=r.get('pmm_ms', 0),
                audit_total_duration_ms=r.get('total_ms', 0),
                audit_version="v4.0-api",
            )
            async with self.db.async_session() as session:
                session.add(record)
                await session.commit()
        except Exception as e:
            logger.warning(f"Erro salvando: {e}")

    # ================================================================
    # MAINTENANCE
    # ================================================================
    async def _maintenance_loop(self):
        while self.running:
            await asyncio.sleep(WS_HEALTH_INTERVAL)

            ws_age = time.time() - self._last_ws_time if self._last_ws_time > 0 else 999
            uptime = time.time() - self._start_time
            ok_count = sum(1 for r in self.results if r.get('success'))

            logger.info(
                f"[STATS] WS: {self._ws_msg_count} msgs, {self._ws_msg_count/max(1,uptime):.1f}/s, "
                f"last {ws_age:.0f}s | "
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
        logger.info(f"{'=' * 50}")
        logger.info(f"STATS — {len(self.results)} auditorias ({len(ok)} OK)")
        logger.info(f"  Lag: min={min(lags)}ms med={sorted(lags)[len(lags)//2]}ms avg={sum(lags)//len(lags)}ms max={max(lags)}ms")
        logger.info(f"  Diff: avg={sum(diffs)/len(diffs):+.2f}% med={sorted(diffs)[len(diffs)//2]:+.2f}%")
        logger.info(f"{'=' * 50}")

    def _print_summary(self):
        ok = [r for r in self.results if r.get('success')]
        fail = [r for r in self.results if not r.get('success')]
        print(f"\n{'=' * 60}")
        print(f"RESUMO — {len(self.results)} auditorias ({len(ok)} OK, {len(fail)} FAIL)")
        if ok:
            lags = [r['total_ms'] for r in ok]
            diffs = [r['diff_pct'] for r in ok]
            print(f"  Lag: min={min(lags)} med={sorted(lags)[len(lags)//2]} max={max(lags)}ms")
            print(f"  Diff: avg={sum(diffs)/len(diffs):+.2f}% med={sorted(diffs)[len(diffs)//2]:+.2f}%")
        print(f"{'=' * 60}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-audits", type=int, default=0, help="0=infinito")
    parser.add_argument("--direction", choices=["up", "down", "all"], default="up")
    parser.add_argument("--no-db", action="store_true")
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
    )
    await audit.run()


if __name__ == "__main__":
    asyncio.run(main())
