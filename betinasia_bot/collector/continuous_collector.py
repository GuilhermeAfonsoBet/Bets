# -*- coding: utf-8 -*-
"""
Coletor Continuo - Roda em background no VPS.

Coleta odds das principais ligas continuamente e salva no banco de dados.
Projetado para rodar como servico systemd.

Uso:
    python -m collector.continuous_collector

Ou via systemd:
    sudo systemctl start betinasia-collector
"""

import asyncio
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from loguru import logger

from scraper.fast_collector import FastCollector, CollectionResult
from storage.database import Database
from storage.models import Match, BestOddsHistory
from storage.models_hypothesis import (
    H1PricingEvent, H3LineMonotonicityEvent,
    H3bTemporalReversalEvent, H6CorrelationLagEvent
)
from hypothesis.detectors import HypothesisDetector, save_hypothesis_events
from sqlalchemy import select, text
from config import settings


class ContinuousCollector:
    """
    Coletor continuo de odds.
    
    Roda em loop infinito, coletando odds das principais ligas
    e salvando no banco de dados.
    
    Projetado para rodar por semanas sem intervenção:
    - Auto-restart de browser periodicamente
    - Reconexão automática em caso de falha
    - Logging com rotação diária
    - Métricas de uptime e saúde
    - Limpeza periódica de memória
    """
    
    # Configuracoes
    CYCLE_TIME = 10  # segundos entre INÍCIO de cada ciclo (não após)
    MAX_CONSECUTIVE_ERRORS = 5  # erros consecutivos antes de pausa longa
    ERROR_PAUSE_SECONDS = 300  # pausa apos muitos erros (5 min)
    SESSION_REFRESH_INTERVAL = 3600  # reconecta browser a cada 1 hora
    STATS_LOG_INTERVAL = 100  # log de estatísticas a cada N ciclos
    MEMORY_CLEANUP_INTERVAL = 500  # limpeza de memória a cada N ciclos
    COLLECT_TIMEOUT_SECONDS = int(os.getenv("COLLECT_TIMEOUT_SECONDS", "120"))
    SAVE_TIMEOUT_SECONDS = int(os.getenv("SAVE_TIMEOUT_SECONDS", "90"))
    ZERO_ODDS_RECOVERY_THRESHOLD = int(os.getenv("COLLECT_ZERO_ODDS_RECOVERY_THRESHOLD", "3"))
    STARTUP_RETRY_SECONDS = int(os.getenv("COLLECT_STARTUP_RETRY_SECONDS", "30"))
    # Evita "hang silencioso" ao reiniciar browser/sessão
    BROWSER_START_TIMEOUT_SECONDS = int(os.getenv("COLLECT_BROWSER_START_TIMEOUT_SECONDS", "180"))
    # Telemetria de fase (melhor diagnóstico e evita gaps sem escrita)
    PHASE_TELEMETRY = os.getenv("COLLECT_PHASE_TELEMETRY", "1").strip() not in ("0", "false", "False", "no", "NO")
    
    def __init__(self):
        self.collector: Optional[FastCollector] = None
        self.db: Optional[Database] = None
        self.running = False
        self.consecutive_errors = 0
        self.total_collections = 0
        self.total_matches_collected = 0
        self.total_errors = 0
        self.start_time: Optional[datetime] = None
        self.last_collection_time: Optional[datetime] = None
        self.last_successful_save: Optional[datetime] = None
        self._last_browser_start: Optional[datetime] = None
        self.consecutive_zero_odds_cycles = 0
        
        # Detector de hipóteses
        self.hypothesis_detector = HypothesisDetector()
        self.total_hypothesis_events = 0
        self.telemetry_file = "logs/collector_telemetry.jsonl"
        
    async def start(self):
        """Inicia o coletor."""
        logger.info("=" * 60)
        logger.info("COLETOR CONTINUO - Iniciando...")
        logger.info("=" * 60)
        
        # Conecta ao banco
        self.db = Database()
        await self.db.connect()
        logger.info("Banco de dados conectado")
        
        # Inicia o FastCollector
        await self._start_browser()
        
        self.running = True
        self.start_time = datetime.now(timezone.utc)
        
        # Configura handlers de sinal para shutdown graceful
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._signal_handler)
            
        logger.info("Coletor continuo iniciado com sucesso")
        logger.info(f"Ciclo de coleta: {self.CYCLE_TIME}s (tempo entre início de cada ciclo)")
        logger.info(
            f"Timeout coleta: {self.COLLECT_TIMEOUT_SECONDS}s | "
            f"Timeout save: {self.SAVE_TIMEOUT_SECONDS}s | "
            f"Auto-recovery zero-odds: {self.ZERO_ODDS_RECOVERY_THRESHOLD} ciclos"
        )
        
    async def _start_browser(self):
        """Inicia ou reinicia o browser."""
        t0 = time.time()
        if self.PHASE_TELEMETRY:
            self._append_jsonl(
                self.telemetry_file,
                {
                    "ts_utc": datetime.now(timezone.utc).isoformat(),
                    "cycle": self.total_collections + 1,
                    "status": "BROWSER_START",
                },
            )
        if self.collector:
            try:
                await asyncio.wait_for(self.collector.close(), timeout=30)
            except:
                pass
                
        self.collector = FastCollector()
        await asyncio.wait_for(self.collector.start(), timeout=self.BROWSER_START_TIMEOUT_SECONDS)
        self._last_browser_start = datetime.now(timezone.utc)
        dt_ms = int((time.time() - t0) * 1000)
        logger.info(f"Browser iniciado (dt={dt_ms}ms)")
        if self.PHASE_TELEMETRY:
            self._append_jsonl(
                self.telemetry_file,
                {
                    "ts_utc": datetime.now(timezone.utc).isoformat(),
                    "cycle": self.total_collections + 1,
                    "status": "BROWSER_OK",
                    "browser_start_ms": dt_ms,
                },
            )
        
    def _signal_handler(self, signum, frame):
        """Handler para sinais de shutdown."""
        logger.info(f"Sinal {signum} recebido, iniciando shutdown...")
        self.running = False

    def _append_jsonl(self, path: str, payload: Dict[str, Any]):
        """Escreve uma linha JSON para telemetria operacional."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug(f"Falha ao gravar telemetria em {path}: {e}")
        
    async def stop(self):
        """Para o coletor."""
        logger.info("Parando coletor...")
        self.running = False
        
        if self.collector:
            await self.collector.close()
            
        if self.db:
            await self.db.close()
            
        # Estatisticas finais
        if self.start_time:
            runtime = datetime.now(timezone.utc) - self.start_time
            logger.info("=" * 60)
            logger.info("ESTATISTICAS FINAIS")
            logger.info("=" * 60)
            logger.info(f"Tempo de execucao: {runtime}")
            logger.info(f"Total de coletas: {self.total_collections}")
            logger.info(f"Total de jogos coletados: {self.total_matches_collected}")
            if self.total_collections > 0:
                avg = self.total_matches_collected / self.total_collections
                logger.info(f"Media de jogos/coleta: {avg:.1f}")
            
            # Estatísticas de hipóteses
            logger.info("--- Eventos de Hipóteses ---")
            logger.info(f"Total de eventos detectados: {self.total_hypothesis_events}")
            stats = self.hypothesis_detector.get_stats()
            logger.info(f"  H1 (Precificação): {stats['h1']}")
            logger.info(f"  H3 (Linhas adjacentes): {stats['h3']}")
            logger.info(f"  H3b (Reversões temporais): {stats['h3b']}")
            logger.info(f"  H6 (Correlação/Lag): {stats['h6']}")
                
        logger.info("Coletor parado")
        
    async def run(self):
        """Loop principal de coleta."""
        # Startup resiliente: se login/navegação falhar (proxy lento, etc),
        # não derruba o processo e evita thrash do systemd + chromes órfãos.
        while True:
            try:
                await self.start()
                break
            except Exception as e:
                logger.error(f"Falha no start do collector (retry em {self.STARTUP_RETRY_SECONDS}s): {e}")
                try:
                    await self.stop()
                except Exception:
                    pass
                await asyncio.sleep(self.STARTUP_RETRY_SECONDS)
        
        while self.running:
            try:
                cycle_start = datetime.now(timezone.utc)
                if self.PHASE_TELEMETRY:
                    self._append_jsonl(
                        self.telemetry_file,
                        {
                            "ts_utc": cycle_start.isoformat(),
                            "cycle": self.total_collections + 1,
                            "status": "CYCLE_START",
                        },
                    )
                
                # Verifica se precisa reiniciar browser
                if self._should_refresh_browser():
                    logger.info("Reiniciando browser (refresh periodico)...")
                    await self._start_browser()
                
                # Executa coleta
                cycle_metrics = await self._collect_cycle()
                
                # Reset contador de erros
                self.consecutive_errors = 0

                # Auto-recuperação quando a coleta "vive" mas sem odds úteis.
                # Evita ficar horas com serviço ativo e N=0.
                with_odds = int(cycle_metrics.get("events_with_odds", 0))
                if with_odds <= 0:
                    self.consecutive_zero_odds_cycles += 1
                    logger.warning(
                        f"Ciclo sem odds úteis ({self.consecutive_zero_odds_cycles}/"
                        f"{self.ZERO_ODDS_RECOVERY_THRESHOLD})"
                    )
                    if self.consecutive_zero_odds_cycles >= self.ZERO_ODDS_RECOVERY_THRESHOLD:
                        logger.warning("Auto-recovery: reiniciando browser por ciclos sem odds")
                        await self._start_browser()
                        self.consecutive_zero_odds_cycles = 0
                else:
                    self.consecutive_zero_odds_cycles = 0
                
                # Log periódico de estatísticas
                if self.total_collections % self.STATS_LOG_INTERVAL == 0:
                    self._log_periodic_stats()
                
                # Limpeza periódica de memória
                if self.total_collections % self.MEMORY_CLEANUP_INTERVAL == 0:
                    self._cleanup_memory()
                
                # Calcula tempo restante até próximo ciclo
                cycle_elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
                sleep_time = max(0, self.CYCLE_TIME - cycle_elapsed)
                
                # Aguarda proximo ciclo
                if self.running and sleep_time > 0:
                    logger.debug(f"Coleta levou {cycle_elapsed:.1f}s, aguardando {sleep_time:.1f}s...")
                    await asyncio.sleep(sleep_time)
                elif self.running:
                    logger.debug(f"Coleta levou {cycle_elapsed:.1f}s (> {self.CYCLE_TIME}s), próximo ciclo imediato")
                    
            except Exception as e:
                self.consecutive_errors += 1
                self.total_errors += 1
                logger.error(f"Erro na coleta ({self.consecutive_errors}/{self.MAX_CONSECUTIVE_ERRORS}): {e}")

                if "COLLECT_TIMEOUT" in str(e):
                    logger.warning("Timeout de coleta detectado, reiniciando browser imediatamente...")
                    try:
                        await self._start_browser()
                        await self._restart_db()
                        self.consecutive_errors = 0
                        await asyncio.sleep(3)
                        continue
                    except Exception as e2:
                        logger.error(f"Falha ao reiniciar browser após timeout: {e2}")

                if "SAVE_TIMEOUT" in str(e):
                    logger.warning("Timeout no SAVE detectado, reiniciando DB e browser imediatamente...")
                    try:
                        await self._restart_db()
                        await self._start_browser()
                        self.consecutive_errors = 0
                        await asyncio.sleep(3)
                        continue
                    except Exception as e2:
                        logger.error(f"Falha ao reiniciar após SAVE_TIMEOUT: {e2}")
                
                if self.consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                    logger.warning(f"Muitos erros consecutivos, pausando por {self.ERROR_PAUSE_SECONDS}s...")
                    if self.PHASE_TELEMETRY:
                        self._append_jsonl(
                            self.telemetry_file,
                            {
                                "ts_utc": datetime.now(timezone.utc).isoformat(),
                                "cycle": self.total_collections + 1,
                                "status": "PAUSE_LONG",
                                "pause_sec": int(self.ERROR_PAUSE_SECONDS),
                                "reason": "MAX_CONSECUTIVE_ERRORS",
                            },
                        )
                    await asyncio.sleep(self.ERROR_PAUSE_SECONDS)
                    
                    # Tenta reiniciar browser
                    try:
                        await self._start_browser()
                        await self._restart_db()
                        self.consecutive_errors = 0
                    except Exception as e2:
                        logger.error(f"Falha ao reiniciar browser: {e2}")
                else:
                    await asyncio.sleep(30)  # Pausa curta entre retries
                    
        await self.stop()
    
    def _log_periodic_stats(self):
        """Loga estatísticas periódicas de uptime."""
        if not self.start_time:
            return
            
        uptime = datetime.now(timezone.utc) - self.start_time
        days = uptime.days
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        
        avg_matches = (self.total_matches_collected / self.total_collections 
                      if self.total_collections > 0 else 0)
        error_rate = (self.total_errors / self.total_collections * 100 
                     if self.total_collections > 0 else 0)
        
        logger.info("=" * 50)
        logger.info("ESTATÍSTICAS PERIÓDICAS")
        logger.info(f"  Uptime: {days}d {hours}h {minutes}m")
        logger.info(f"  Ciclos: {self.total_collections}")
        logger.info(f"  Jogos coletados: {self.total_matches_collected}")
        logger.info(f"  Média jogos/ciclo: {avg_matches:.1f}")
        logger.info(f"  Erros totais: {self.total_errors} ({error_rate:.1f}%)")
        logger.info(f"  Eventos hipóteses: {self.total_hypothesis_events}")
        
        if self.last_successful_save:
            age = (datetime.now(timezone.utc) - self.last_successful_save).total_seconds()
            logger.info(f"  Último save: {age:.0f}s atrás")
        logger.info("=" * 50)
    
    def _cleanup_memory(self):
        """Limpeza periódica de memória."""
        import gc
        gc.collect()
        logger.debug("Limpeza de memória executada")
        
    def _should_refresh_browser(self) -> bool:
        """Verifica se deve reiniciar o browser."""
        if not self._last_browser_start:
            return True
            
        elapsed = (datetime.now(timezone.utc) - self._last_browser_start).total_seconds()
        return elapsed > self.SESSION_REFRESH_INTERVAL
        
    async def _collect_cycle(self):
        """Executa um ciclo de coleta."""
        cycle_start = datetime.now(timezone.utc)
        cycle_t0 = time.time()
        logger.info(f"Iniciando ciclo de coleta #{self.total_collections + 1}...")
        if self.PHASE_TELEMETRY:
            self._append_jsonl(
                self.telemetry_file,
                {
                    "ts_utc": cycle_start.isoformat(),
                    "cycle": self.total_collections + 1,
                    "status": "COLLECT_START",
                },
            )
        
        # Coleta dados
        collect_t0 = time.time()
        try:
            result = await asyncio.wait_for(
                self.collector.collect_all(),
                timeout=self.COLLECT_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            collect_ms = int((time.time() - collect_t0) * 1000)
            timeout_payload = {
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "cycle": self.total_collections + 1,
                "status": "COLLECT_TIMEOUT",
                "collect_ms": collect_ms,
                "timeout_sec": self.COLLECT_TIMEOUT_SECONDS,
                "events_discovered": 0,
                "events_with_odds": 0,
                "matches_payload": 0,
                "matches_saved": 0,
                "save_errors": 0,
            }
            self._append_jsonl(self.telemetry_file, timeout_payload)
            raise RuntimeError(f"COLLECT_TIMEOUT_{self.COLLECT_TIMEOUT_SECONDS}s")
        collect_ms = int((time.time() - collect_t0) * 1000)
        if self.PHASE_TELEMETRY:
            self._append_jsonl(
                self.telemetry_file,
                {
                    "ts_utc": datetime.now(timezone.utc).isoformat(),
                    "cycle": self.total_collections + 1,
                    "status": "COLLECT_OK",
                    "collect_ms": collect_ms,
                    "events_discovered": int(getattr(result, "total_events", 0) or 0),
                    "events_with_odds": int(getattr(result, "total_with_odds", 0) or 0),
                    "matches_payload": int(len(getattr(result, "matches", []) or [])),
                },
            )
        
        # Salva no banco
        save_t0 = time.time()
        if self.PHASE_TELEMETRY:
            self._append_jsonl(
                self.telemetry_file,
                {
                    "ts_utc": datetime.now(timezone.utc).isoformat(),
                    "cycle": self.total_collections + 1,
                    "status": "SAVE_START",
                },
            )
        try:
            save_metrics = await asyncio.wait_for(
                self._save_to_database(result),
                timeout=self.SAVE_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            save_ms = int((time.time() - save_t0) * 1000)
            timeout_payload = {
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "cycle": self.total_collections + 1,
                "status": "SAVE_TIMEOUT",
                "collect_ms": collect_ms,
                "save_ms": save_ms,
                "timeout_sec": self.SAVE_TIMEOUT_SECONDS,
                "events_discovered": int(getattr(result, "total_events", 0) or 0),
                "events_with_odds": int(getattr(result, "total_with_odds", 0) or 0),
                "matches_payload": int(len(getattr(result, "matches", []) or [])),
                "matches_saved": 0,
                "prematch_saved": 0,
                "live_saved": 0,
                "hypothesis_events_saved": 0,
                "save_errors": 1,
            }
            self._append_jsonl(self.telemetry_file, timeout_payload)
            raise RuntimeError(f"SAVE_TIMEOUT_{self.SAVE_TIMEOUT_SECONDS}s")
        save_ms = int((time.time() - save_t0) * 1000)
        saved_count = save_metrics["saved_count"]
        
        # Atualiza estatisticas
        self.total_collections += 1
        self.total_matches_collected += saved_count
        self.last_collection_time = datetime.now(timezone.utc)
        if saved_count > 0:
            self.last_successful_save = datetime.now(timezone.utc)
        
        cycle_duration = (self.last_collection_time - cycle_start).total_seconds()
        cycle_total_ms = int((time.time() - cycle_t0) * 1000)

        cycle_telemetry = {
            "ts_utc": self.last_collection_time.isoformat(),
            "cycle": self.total_collections,
            "status": "OK",
            "cycle_total_ms": cycle_total_ms,
            "collect_ms": collect_ms,
            "save_ms": save_ms,
            "collect_reported_ms": int(result.collection_time * 1000),
            "events_discovered": result.total_events,
            "events_with_odds": result.total_with_odds,
            "matches_payload": len(result.matches),
            "matches_saved": saved_count,
            "prematch_saved": save_metrics["prematch_count"],
            "live_saved": save_metrics["live_count"],
            "hypothesis_events_saved": save_metrics["hypothesis_events_count"],
            "save_errors": save_metrics["save_errors"],
        }
        self._append_jsonl(self.telemetry_file, cycle_telemetry)
        
        logger.info(
            f"Ciclo #{self.total_collections} concluido: "
            f"{saved_count} jogos salvos em {cycle_duration:.1f}s"
        )
        logger.info(
            f"Telemetria ciclo #{self.total_collections}: "
            f"collect={collect_ms}ms save={save_ms}ms total={cycle_total_ms}ms | "
            f"payload={len(result.matches)} saved={saved_count} "
            f"(pre={save_metrics['prematch_count']}, live={save_metrics['live_count']}, "
            f"erros_save={save_metrics['save_errors']})"
        )
        if result.total_with_odds <= 0:
            logger.warning(
                "Coleta retornou 0 eventos com odds; possível sessão inválida/WS inativo"
            )

        return {
            "saved_count": saved_count,
            "events_discovered": result.total_events,
            "events_with_odds": result.total_with_odds,
            "save_errors": save_metrics["save_errors"],
        }

    async def _restart_db(self):
        """Reinicia conexão com banco (útil após timeouts/hangs)."""
        try:
            if self.db:
                await self.db.close()
        except Exception:
            pass
        self.db = Database()
        await self.db.connect()
        logger.info("Banco de dados reiniciado")
    async def _save_to_database(self, result: CollectionResult) -> Dict[str, int]:
        """
        Salva resultado da coleta no banco de dados.
        
        Estrategia:
        - Jogos pré-match E in-match (com flag is_live para distinção)
        - Tabela matches: UPSERT (atualiza se existir)
        - Tabela best_odds_history: INSERT (historico)
        - Detectores de hipóteses: analisa e salva eventos com is_live flag
        """
        saved_count = 0
        live_count = 0
        prematch_count = 0
        hypothesis_events_count = 0
        save_errors = 0
        now = datetime.now(timezone.utc)
        
        async with self.db.async_session() as session:
            for match_odds in result.matches:
                # Determina se o jogo está ao vivo (in-match) ou é pre-match
                is_live = False
                if match_odds.kickoff_time:
                    is_live = match_odds.kickoff_time <= now
                
                if is_live:
                    live_count += 1
                else:
                    prematch_count += 1
                
                try:
                    # 1. Salva/atualiza match
                    existing = await session.execute(
                        select(Match).where(Match.external_id == match_odds.event_id)
                    )
                    match = existing.scalar_one_or_none()
                    
                    if match:
                        # Atualiza
                        match.league = match_odds.league
                        match.home_team = match_odds.home_team
                        match.away_team = match_odds.away_team
                        if match_odds.kickoff_time:
                            match.kickoff_time = match_odds.kickoff_time
                    else:
                        # Cria novo
                        match = Match(
                            external_id=match_odds.event_id,
                            league=match_odds.league or "Unknown",
                            home_team=match_odds.home_team or "Unknown",
                            away_team=match_odds.away_team or "Unknown",
                            kickoff_time=match_odds.kickoff_time or datetime.now(timezone.utc),
                        )
                        session.add(match)
                        await session.flush()  # Para obter o ID
                    
                    # 2. Salva odds no historico + detecta hipóteses
                    for line, odds in match_odds.ah_lines.items():
                        best_odds = BestOddsHistory(
                            match_id=match.id,
                            ah_line=str(line),
                            best_home_odds=odds.home_odds,
                            best_away_odds=odds.away_odds,
                        )
                        session.add(best_odds)
                        
                        # === DETECÇÃO DE HIPÓTESES (AH) ===
                        if odds.home_odds and odds.away_odds:
                            events = self.hypothesis_detector.process_market_update(
                                match_id=match.id,
                                market_type="AH",
                                line=str(line),
                                home_odd=odds.home_odds,
                                away_odd=odds.away_odds,
                                is_live=is_live,  # Flag para distinguir pre-match vs in-match
                            )
                            hypothesis_events_count += await save_hypothesis_events(session, events)
                    
                    # 3. Salva Over/Under (se existir) + detecta hipóteses
                    for line, odds in match_odds.over_under.items():
                        # Usa ah_line com prefixo "OU_" para diferenciar
                        best_odds = BestOddsHistory(
                            match_id=match.id,
                            ah_line=f"OU_{line}",
                            best_home_odds=odds.over_odds,  # over no campo home
                            best_away_odds=odds.under_odds,  # under no campo away
                        )
                        session.add(best_odds)
                        
                        # === DETECÇÃO DE HIPÓTESES (OU) ===
                        if odds.over_odds and odds.under_odds:
                            events = self.hypothesis_detector.process_market_update(
                                match_id=match.id,
                                market_type="OU",
                                line=str(line),
                                home_odd=odds.over_odds,
                                away_odd=odds.under_odds,
                                is_live=is_live,  # Flag para distinguir pre-match vs in-match
                            )
                            hypothesis_events_count += await save_hypothesis_events(session, events)
                    
                    # 4. Salva 1X2 (se existir) + detecta hipóteses
                    if match_odds.match_odds:
                        home_odd = match_odds.match_odds.get('h', 0)
                        away_odd = match_odds.match_odds.get('a', 0)
                        
                        best_odds = BestOddsHistory(
                            match_id=match.id,
                            ah_line="1X2",
                            best_home_odds=home_odd,
                            best_away_odds=away_odd,
                        )
                        session.add(best_odds)
                        
                        # === DETECÇÃO DE HIPÓTESES (1X2) ===
                        if home_odd and away_odd:
                            events = self.hypothesis_detector.process_market_update(
                                match_id=match.id,
                                market_type="1X2",
                                line="1X2",
                                home_odd=home_odd,
                                away_odd=away_odd,
                                is_live=is_live,  # Flag para distinguir pre-match vs in-match
                            )
                            hypothesis_events_count += await save_hypothesis_events(session, events)
                        
                        # Draw separado
                        if 'd' in match_odds.match_odds:
                            best_odds_draw = BestOddsHistory(
                                match_id=match.id,
                                ah_line="1X2_DRAW",
                                best_home_odds=match_odds.match_odds['d'],
                                best_away_odds=match_odds.match_odds['d'],
                            )
                            session.add(best_odds_draw)
                    
                    saved_count += 1
                    
                except Exception as e:
                    save_errors += 1
                    logger.warning(f"Erro ao salvar jogo {match_odds.event_id}: {e}")
                    continue
                    
            # Commit de tudo
            await session.commit()
        
        # Atualiza contador de eventos de hipóteses
        self.total_hypothesis_events += hypothesis_events_count
        if hypothesis_events_count > 0:
            logger.info(f"  → {hypothesis_events_count} eventos de hipóteses detectados")
        
        # Log de distribuição pre-match vs in-match
        if live_count > 0 or prematch_count > 0:
            logger.debug(f"  → Jogos: {prematch_count} pré-match, {live_count} in-match")

        return {
            "saved_count": saved_count,
            "live_count": live_count,
            "prematch_count": prematch_count,
            "hypothesis_events_count": hypothesis_events_count,
            "save_errors": save_errors,
        }


async def main():
    """Entry point."""
    # Configura logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "logs/collector_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="30 days",
        level="DEBUG"
    )
    
    collector = ContinuousCollector()
    await collector.run()


if __name__ == "__main__":
    asyncio.run(main())
