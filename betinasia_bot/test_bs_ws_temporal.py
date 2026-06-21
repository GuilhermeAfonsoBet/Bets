#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validação temporal BS vs WS: mede evolução do betslip ao longo do tempo.

Para um jogo ativo, chama betslip API repetidamente (a cada 5-10s por 2-3 min)
e registra:
  - WS odd no momento
  - BS best odd no momento
  - Diff BS vs WS
  - Timestamp

Objetivo: entender se BS acompanha WS, com qual lag, e se convergem.

Uso:
    DISPLAY=:99 python test_bs_ws_temporal.py
    DISPLAY=:99 python test_bs_ws_temporal.py --interval 5 --duration 180
"""

import asyncio
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional
from loguru import logger

sys.path.insert(0, '.')

from scraper.betinasia import BetinAsiaScraper
from scraper.api_betslip import ApiBetslipClient

FOOTBALL_URL = "https://black.betinasia.com/sportsbook/football"


class BsWsTemporalTracker:

    def __init__(self, interval: int = 10, duration: int = 120, num_events: int = 3):
        """
        Args:
            interval: Segundos entre cada check do betslip
            duration: Duração total do tracking por evento (segundos)
            num_events: Quantos eventos acompanhar simultaneamente
        """
        self.interval = interval
        self.duration = duration
        self.num_events = num_events

        self.scraper: Optional[BetinAsiaScraper] = None
        self.api_client: Optional[ApiBetslipClient] = None
        self._ws_messages: List[str] = []
        self._ws_msg_count: int = 0
        self._last_ws_time: float = 0
        self._events_info: Dict[str, dict] = {}
        self._current_odds: Dict[str, Dict[str, float]] = {}  # event_id -> {line_side: ws_odd}

        # Resultados
        self.tracking_data: List[dict] = []

    async def start(self):
        logger.info("=" * 60)
        logger.info("VALIDACAO TEMPORAL BS vs WS")
        logger.info(f"Intervalo: {self.interval}s | Duracao: {self.duration}s | Eventos: {self.num_events}")
        logger.info("=" * 60)

        self.scraper = BetinAsiaScraper()
        await self.scraper.start()
        await self.scraper.login()

        page = self.scraper._page
        self.api_client = ApiBetslipClient(page)

        # WS listener
        def on_ws(ws):
            def on_frame(data):
                data_str = str(data)
                self._ws_messages.append(data_str)
                self._last_ws_time = time.time()
                self._ws_msg_count += 1
                # PMM listener
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

        await page.goto(FOOTBALL_URL)
        await page.wait_for_load_state("domcontentloaded")
        logger.info("Aguardando WebSocket...")
        await page.wait_for_timeout(5000)
        logger.info(f"WS: {self._ws_msg_count} msgs")

    async def find_active_events(self) -> List[dict]:
        """Encontra eventos ativos com odds AH no WebSocket."""
        events = {}

        for msg in self._ws_messages:
            try:
                data = json.loads(msg)
                if not isinstance(data, list):
                    continue
                for item in data:
                    if not isinstance(item, list) or len(item) < 2:
                        continue

                    msg_type, msg_meta = item[0], item[1]
                    msg_data = item[2] if len(item) > 2 else {}

                    if msg_type == 'event' and isinstance(msg_meta, list) and len(msg_meta) >= 2:
                        if msg_meta[0] == 'fb' and 'home' in msg_data:
                            eid = msg_meta[1]
                            events[eid] = {
                                'home': msg_data.get('home', ''),
                                'away': msg_data.get('away', ''),
                                'league': msg_data.get('competition_name', ''),
                            }

                    if msg_type in ['offers_hcap', 'offers_event']:
                        if isinstance(msg_meta, list) and len(msg_meta) >= 3 and msg_meta[1] == 'fb':
                            eid = msg_meta[2]
                            if 'ah' in msg_data and eid in events:
                                ah = msg_data['ah']
                                lines = []
                                if isinstance(ah, list) and len(ah) >= 2:
                                    if isinstance(ah[0], (int, float)):
                                        lines = [ah]
                                    elif isinstance(ah[0], list):
                                        lines = ah

                                for ld in lines:
                                    if len(ld) < 2:
                                        continue
                                    line = ld[0]
                                    odds_list = ld[1]
                                    h_odd = a_odd = 0
                                    if isinstance(odds_list, list):
                                        for o in odds_list:
                                            if isinstance(o, list) and len(o) >= 2:
                                                if o[0] == 'h':
                                                    h_odd = float(o[1])
                                                elif o[0] == 'a':
                                                    a_odd = float(o[1])

                                    if h_odd > 1.3 and h_odd < 3.0:
                                        key = f"{eid}|AH|{line}|home"
                                        self._current_odds[key] = h_odd
                                        if 'best_line' not in events.get(eid, {}):
                                            events[eid]['best_line'] = str(line)
                                            events[eid]['best_side'] = 'home'
                                            events[eid]['ws_odd'] = h_odd
            except:
                continue

        # Filtra eventos com odds
        active = [{'event_id': eid, **info} for eid, info in events.items()
                  if 'ws_odd' in info]

        # Ordena por odds mais "interessantes" (perto de 2.0)
        active.sort(key=lambda x: abs(x.get('ws_odd', 0) - 2.0))
        return active[:self.num_events * 2]  # Pega mais que precisa (alguns podem falhar)

    async def track_event(self, event: dict):
        """Acompanha um evento por 'duration' segundos, checando betslip a cada 'interval'."""
        event_id = event['event_id']
        line = event.get('best_line', '0')
        side = event.get('best_side', 'home')
        home = event.get('home', '?')
        away = event.get('away', '?')

        bet_type = ApiBetslipClient.build_bet_type('AH', side, line)

        logger.info(f"\nTRACKING: {home} vs {away} | AH {line} {side}")
        logger.info(f"  bet_type: {bet_type}")
        logger.info(f"  {'t(s)':>5} | {'WS odd':>8} | {'BS odd':>8} | {'Diff':>8} | {'BS limit':>10} | {'# BK':>4} | {'Lag':>6}")
        logger.info(f"  {'-'*60}")

        measurements = []
        start_time = time.time()
        check_num = 0

        while time.time() - start_time < self.duration:
            check_num += 1
            t_elapsed = time.time() - start_time

            # WS odd atual
            ws_key = f"{event_id}|AH|{line}|{side}"
            ws_odd = self._current_odds.get(ws_key, event.get('ws_odd', 0))

            # BS odd via API
            t0 = time.time()
            bs_result = await self.api_client.get_betslip_odds(event_id, bet_type)
            lag_ms = int((time.time() - t0) * 1000)

            if bs_result.success:
                bs_odd = bs_result.best_odd
                bs_limit = bs_result.best_limit
                num_bk = bs_result.num_bookmakers
                diff = ((bs_odd - ws_odd) / ws_odd * 100) if ws_odd > 0 else 0

                measurement = {
                    't': round(t_elapsed, 1),
                    'check': check_num,
                    'ws_odd': ws_odd,
                    'bs_odd': bs_odd,
                    'diff_pct': round(diff, 3),
                    'bs_limit': bs_limit,
                    'num_bk': num_bk,
                    'lag_ms': lag_ms,
                    'bs_bookie': bs_result.best_bookie,
                    'second_odd': bs_result.second_odd,
                    'highest_limit': bs_result.highest_limit,
                    'highest_limit_bookie': bs_result.highest_limit_bookie,
                }
                measurements.append(measurement)

                marker = " <<<" if diff > 2 else " !!!" if diff > 0 else ""
                logger.info(
                    f"  {t_elapsed:5.1f}s | {ws_odd:8.3f} | {bs_odd:8.3f} | {diff:+7.2f}% | "
                    f"${bs_limit:>9,.0f} | {num_bk:4d} | {lag_ms:5d}ms{marker}")
            else:
                logger.warning(f"  {t_elapsed:5.1f}s | {ws_odd:8.3f} | FAIL: {bs_result.error}")
                measurements.append({
                    't': round(t_elapsed, 1),
                    'check': check_num,
                    'ws_odd': ws_odd,
                    'bs_odd': None,
                    'error': bs_result.error,
                })

            # Espera intervalo
            await asyncio.sleep(self.interval)

        # Resumo do evento
        valid = [m for m in measurements if m.get('bs_odd')]
        if valid:
            ws_odds = [m['ws_odd'] for m in valid]
            bs_odds = [m['bs_odd'] for m in valid]
            diffs = [m['diff_pct'] for m in valid]

            logger.info(f"\n  RESUMO {home} vs {away} AH {line} {side}:")
            logger.info(f"    Checks: {len(valid)}/{check_num}")
            logger.info(f"    WS range: {min(ws_odds):.3f} - {max(ws_odds):.3f} (delta={max(ws_odds)-min(ws_odds):.3f})")
            logger.info(f"    BS range: {min(bs_odds):.3f} - {max(bs_odds):.3f} (delta={max(bs_odds)-min(bs_odds):.3f})")
            logger.info(f"    Diff range: {min(diffs):+.2f}% a {max(diffs):+.2f}%")
            logger.info(f"    Diff media: {sum(diffs)/len(diffs):+.2f}%")
            logger.info(f"    BS > WS: {sum(1 for d in diffs if d > 0)}/{len(diffs)} vezes ({sum(1 for d in diffs if d > 0)/len(diffs)*100:.0f}%)")
            logger.info(f"    BS > WS +2%: {sum(1 for d in diffs if d > 2)}/{len(diffs)} vezes")

            # Correlação temporal
            if len(valid) >= 3:
                ws_changes = [valid[i]['ws_odd'] - valid[i-1]['ws_odd'] for i in range(1, len(valid))]
                bs_changes = [valid[i]['bs_odd'] - valid[i-1]['bs_odd'] for i in range(1, len(valid))]
                if ws_changes and bs_changes:
                    same_direction = sum(1 for w, b in zip(ws_changes, bs_changes)
                                        if (w > 0 and b > 0) or (w < 0 and b < 0) or (w == 0 and b == 0))
                    logger.info(f"    Movem na mesma direcao: {same_direction}/{len(ws_changes)} ({same_direction/len(ws_changes)*100:.0f}%)")

        self.tracking_data.append({
            'event_id': event_id,
            'home': home,
            'away': away,
            'line': line,
            'side': side,
            'measurements': measurements,
        })

    async def run(self):
        await self.start()

        # Encontra eventos ativos
        logger.info("\nBuscando eventos ativos com AH...")
        events = await self.find_active_events()
        logger.info(f"Encontrados {len(events)} eventos com odds AH")

        if not events:
            logger.error("Nenhum evento ativo encontrado")
            await self.scraper.close()
            return

        for i, event in enumerate(events[:self.num_events]):
            logger.info(f"\n{'='*60}")
            logger.info(f"EVENTO {i+1}/{min(self.num_events, len(events))}")
            await self.track_event(event)

        # Salva dados
        with open('bs_ws_temporal.json', 'w') as f:
            json.dump(self.tracking_data, f, indent=2, default=str)
        logger.info(f"\nDados salvos em bs_ws_temporal.json")

        # Resumo geral
        logger.info(f"\n{'='*60}")
        logger.info("RESUMO GERAL")
        logger.info(f"{'='*60}")

        all_diffs = []
        for track in self.tracking_data:
            valid = [m for m in track['measurements'] if m.get('bs_odd')]
            all_diffs.extend([m['diff_pct'] for m in valid])

        if all_diffs:
            logger.info(f"  Total measurements: {len(all_diffs)}")
            logger.info(f"  Diff media geral: {sum(all_diffs)/len(all_diffs):+.2f}%")
            logger.info(f"  Diff mediana: {sorted(all_diffs)[len(all_diffs)//2]:+.2f}%")
            logger.info(f"  BS > WS: {sum(1 for d in all_diffs if d > 0)}/{len(all_diffs)} ({sum(1 for d in all_diffs if d > 0)/len(all_diffs)*100:.0f}%)")
            logger.info(f"  BS > WS +2%: {sum(1 for d in all_diffs if d > 2)}/{len(all_diffs)}")

        await self.scraper.close()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=10, help="Segundos entre checks (default: 10)")
    parser.add_argument("--duration", type=int, default=120, help="Duracao tracking por evento em segundos (default: 120)")
    parser.add_argument("--events", type=int, default=3, help="Numero de eventos a acompanhar (default: 3)")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr,
               format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level:<7}</level> | <level>{message}</level>",
               level="INFO")
    logger.add("logs/bs_ws_temporal_{time:YYYY-MM-DD}.log", rotation="00:00", retention="30 days", level="DEBUG")

    tracker = BsWsTemporalTracker(
        interval=args.interval,
        duration=args.duration,
        num_events=args.events,
    )
    await tracker.run()


if __name__ == "__main__":
    asyncio.run(main())
