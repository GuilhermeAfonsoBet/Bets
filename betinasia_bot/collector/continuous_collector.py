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
from sqlalchemy import select, text
from config import settings


class ContinuousCollector:
    """
    Coletor continuo de odds.
    
    Roda em loop infinito, coletando odds das principais ligas
    e salvando no banco de dados.
    """
    
    # Configuracoes
    COLLECTION_INTERVAL = 60  # segundos entre coletas
    MAX_CONSECUTIVE_ERRORS = 5  # erros consecutivos antes de pausa longa
    ERROR_PAUSE_SECONDS = 300  # pausa apos muitos erros (5 min)
    SESSION_REFRESH_INTERVAL = 3600  # reconecta browser a cada 1 hora
    
    def __init__(self):
        self.collector: Optional[FastCollector] = None
        self.db: Optional[Database] = None
        self.running = False
        self.consecutive_errors = 0
        self.total_collections = 0
        self.total_matches_collected = 0
        self.start_time: Optional[datetime] = None
        self.last_collection_time: Optional[datetime] = None
        self._last_browser_start: Optional[datetime] = None
        
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
        logger.info(f"Intervalo entre coletas: {self.COLLECTION_INTERVAL}s")
        
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
                
        logger.info("Coletor parado")
        
    async def run(self):
        """Loop principal de coleta."""
        await self.start()
        
        while self.running:
            try:
                # Verifica se precisa reiniciar browser
                if self._should_refresh_browser():
                    logger.info("Reiniciando browser (refresh periodico)...")
                    await self._start_browser()
                
                # Executa coleta
                await self._collect_cycle()
                
                # Reset contador de erros
                self.consecutive_errors = 0
                
                # Aguarda proximo ciclo
                if self.running:
                    logger.debug(f"Aguardando {self.COLLECTION_INTERVAL}s para proxima coleta...")
                    await asyncio.sleep(self.COLLECTION_INTERVAL)
                    
            except Exception as e:
                self.consecutive_errors += 1
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
        
        cycle_duration = (self.last_collection_time - cycle_start).total_seconds()
        
        logger.info(
            f"Ciclo #{self.total_collections} concluido: "
            f"{saved_count} jogos salvos em {cycle_duration:.1f}s"
        )
        
    async def _save_to_database(self, result: CollectionResult) -> int:
        """
        Salva resultado da coleta no banco de dados.
        
        Estrategia:
        - Tabela matches: UPSERT (atualiza se existir)
        - Tabela best_odds_history: INSERT (historico)
        """
        saved_count = 0
        
        async with self.db.async_session() as session:
            for match_odds in result.matches:
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
                    
                    # 2. Salva odds no historico
                    for line, odds in match_odds.ah_lines.items():
                        best_odds = BestOddsHistory(
                            match_id=match.id,
                            ah_line=str(line),
                            best_home_odds=odds.home_odds,
                            best_away_odds=odds.away_odds,
                        )
                        session.add(best_odds)
                    
                    # 3. Salva Over/Under (se existir)
                    for line, odds in match_odds.over_under.items():
                        # Usa ah_line com prefixo "OU_" para diferenciar
                        best_odds = BestOddsHistory(
                            match_id=match.id,
                            ah_line=f"OU_{line}",
                            best_home_odds=odds.over_odds,  # over no campo home
                            best_away_odds=odds.under_odds,  # under no campo away
                        )
                        session.add(best_odds)
                    
                    # 4. Salva 1X2 (se existir)
                    if match_odds.match_odds:
                        best_odds = BestOddsHistory(
                            match_id=match.id,
                            ah_line="1X2",
                            best_home_odds=match_odds.match_odds.get('h', 0),
                            best_away_odds=match_odds.match_odds.get('a', 0),
                        )
                        session.add(best_odds)
                        
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
