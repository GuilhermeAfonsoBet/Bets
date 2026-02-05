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
import signal
import sys
from datetime import datetime, timezone
from typing import Optional
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
        
        # Detector de hipóteses
        self.hypothesis_detector = HypothesisDetector()
        self.total_hypothesis_events = 0
        
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
        
    async def _start_browser(self):
        """Inicia ou reinicia o browser."""
        if self.collector:
            try:
                await self.collector.close()
            except:
                pass
                
        self.collector = FastCollector()
        await self.collector.start()
        self._last_browser_start = datetime.now(timezone.utc)
        logger.info("Browser iniciado")
        
    def _signal_handler(self, signum, frame):
        """Handler para sinais de shutdown."""
        logger.info(f"Sinal {signum} recebido, iniciando shutdown...")
        self.running = False
        
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
        await self.start()
        
        while self.running:
            try:
                cycle_start = datetime.now(timezone.utc)
                
                # Verifica se precisa reiniciar browser
                if self._should_refresh_browser():
                    logger.info("Reiniciando browser (refresh periodico)...")
                    await self._start_browser()
                
                # Executa coleta
                await self._collect_cycle()
                
                # Reset contador de erros
                self.consecutive_errors = 0
                
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
                
                if self.consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                    logger.warning(f"Muitos erros consecutivos, pausando por {self.ERROR_PAUSE_SECONDS}s...")
                    await asyncio.sleep(self.ERROR_PAUSE_SECONDS)
                    
                    # Tenta reiniciar browser
                    try:
                        await self._start_browser()
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
        logger.info(f"Iniciando ciclo de coleta #{self.total_collections + 1}...")
        
        # Coleta dados
        result = await self.collector.collect_all()
        
        # Salva no banco
        saved_count = await self._save_to_database(result)
        
        # Atualiza estatisticas
        self.total_collections += 1
        self.total_matches_collected += saved_count
        self.last_collection_time = datetime.now(timezone.utc)
        if saved_count > 0:
            self.last_successful_save = datetime.now(timezone.utc)
        
        cycle_duration = (self.last_collection_time - cycle_start).total_seconds()
        
        logger.info(
            f"Ciclo #{self.total_collections} concluido: "
            f"{saved_count} jogos salvos em {cycle_duration:.1f}s"
        )
        
    async def _save_to_database(self, result: CollectionResult) -> int:
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
            
        return saved_count


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
