# -*- coding: utf-8 -*-
"""
Job automatico para atualizar resultados.

Roda em loop, atualizando resultados a cada X horas.
Pode rodar como servico systemd separado.

Uso:
    python -m results.auto_update_results
"""

import asyncio
import signal
from datetime import datetime, timezone, timedelta
from loguru import logger
import sys

from .update_results import update_results


class AutoResultsUpdater:
    """
    Atualizador automatico de resultados.
    
    Roda em loop, consultando a API periodicamente.
    """
    
    # Intervalo entre atualizacoes (em segundos)
    UPDATE_INTERVAL = 4 * 60 * 60  # 4 horas
    
    # Horarios ideais para atualizar (quando jogos terminam)
    # Formato: hora UTC
    PREFERRED_HOURS = [0, 4, 8, 12, 16, 20, 23]  # A cada ~4 horas
    
    def __init__(self):
        self.running = False
        self.total_updates = 0
        self.start_time = None
        
    async def start(self):
        """Inicia o atualizador."""
        logger.info("=" * 60)
        logger.info("AUTO RESULTS UPDATER - Iniciando...")
        logger.info("=" * 60)
        logger.info(f"Intervalo entre atualizacoes: {self.UPDATE_INTERVAL // 3600} horas")
        
        self.running = True
        self.start_time = datetime.now(timezone.utc)
        
        # Configura handlers de sinal
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._signal_handler)
            
    def _signal_handler(self, signum, frame):
        """Handler para sinais de shutdown."""
        logger.info(f"Sinal {signum} recebido, parando...")
        self.running = False
        
    async def run(self):
        """Loop principal."""
        await self.start()
        
        while self.running:
            try:
                # Executa atualizacao
                logger.info(f"Iniciando atualizacao #{self.total_updates + 1}...")
                
                await update_results()
                
                self.total_updates += 1
                logger.info(f"Atualizacao #{self.total_updates} concluida")
                
                # Aguarda proximo ciclo
                if self.running:
                    next_update = datetime.now(timezone.utc) + timedelta(seconds=self.UPDATE_INTERVAL)
                    logger.info(f"Proxima atualizacao: {next_update.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                    
                    # Aguarda em intervalos menores para responder a sinais
                    for _ in range(self.UPDATE_INTERVAL // 60):
                        if not self.running:
                            break
                        await asyncio.sleep(60)
                        
            except Exception as e:
                logger.error(f"Erro na atualizacao: {e}")
                # Aguarda 30 minutos antes de tentar novamente
                await asyncio.sleep(30 * 60)
                
        # Estatisticas finais
        if self.start_time:
            runtime = datetime.now(timezone.utc) - self.start_time
            logger.info("=" * 60)
            logger.info("ESTATISTICAS FINAIS")
            logger.info("=" * 60)
            logger.info(f"Tempo de execucao: {runtime}")
            logger.info(f"Total de atualizacoes: {self.total_updates}")


async def run_once():
    """Executa uma unica atualizacao (para cron)."""
    logger.info("Executando atualizacao unica de resultados...")
    await update_results()
    logger.info("Atualizacao concluida")


async def main():
    """Entry point."""
    # Configura logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "logs/results_updater_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="30 days",
        level="DEBUG"
    )
    
    # Verifica se e execucao unica ou loop
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Executa apenas uma vez")
    args = parser.parse_args()
    
    if args.once:
        await run_once()
    else:
        updater = AutoResultsUpdater()
        await updater.run()


if __name__ == "__main__":
    asyncio.run(main())
