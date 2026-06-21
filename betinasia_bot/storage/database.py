# -*- coding: utf-8 -*-
"""
Classe de acesso ao banco de dados PostgreSQL.
"""

import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, update
from datetime import datetime, timezone
from typing import Optional, List
from loguru import logger

from config import settings
from .models import Base, Match, OddsHistory, BestOddsHistory, Opportunity, Bet, LeagueConfig
from .models_hypothesis import (
    H1PricingEvent, H3LineMonotonicityEvent,
    H3bTemporalReversalEvent, H6CorrelationLagEvent, OddsMovementHistory
)


class Database:
    """
    Classe para operações no banco de dados.
    
    Uso:
        db = Database()
        await db.connect()
        await db.save_match(...)
        await db.close()
    """
    
    def __init__(self, database_url: str = None):
        """
        Inicializa a conexão com o banco.
        
        Args:
            database_url: URL do banco. Se não fornecido, usa settings.
        """
        url = database_url or settings.database_url
        
        # Converte para formato asyncpg (postgresql+asyncpg://)
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
            
        # Robustez para serviços long-running (bridge/auditors):
        # - pool_pre_ping reduz "connection is closed" em conexões recicladas
        # - pool_recycle evita conexões muito antigas (NAT/timeouts)
        try:
            pool_recycle = int(float(os.getenv("DB_POOL_RECYCLE_SEC", "900") or 900))
        except Exception:
            pool_recycle = 900
        self.engine = create_async_engine(
            url,
            echo=False,
            pool_pre_ping=True,
            pool_recycle=max(0, int(pool_recycle)),
        )
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
        
    async def connect(self):
        """Cria as tabelas se não existirem."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Conexão com banco de dados estabelecida")
        
    async def close(self):
        """Fecha a conexão."""
        await self.engine.dispose()
        logger.info("Conexão com banco de dados fechada")
        
    # ==========================================
    # MATCHES
    # ==========================================
    
    async def save_match(self, match_data) -> int:
        """
        Salva ou atualiza uma partida.
        
        Args:
            match_data: Objeto MatchData do scraper
            
        Returns:
            ID da partida no banco
        """
        async with self.async_session() as session:
            # Verifica se já existe
            result = await session.execute(
                select(Match).where(Match.external_id == match_data.match_id)
            )
            existing = result.scalar_one_or_none()
            
            if existing:
                # Atualiza
                existing.league = match_data.league
                existing.home_team = match_data.home_team
                existing.away_team = match_data.away_team
                existing.kickoff_time = match_data.kickoff_time
                match_id = existing.id
            else:
                # Cria novo
                match = Match(
                    external_id=match_data.match_id,
                    league=match_data.league,
                    home_team=match_data.home_team,
                    away_team=match_data.away_team,
                    kickoff_time=match_data.kickoff_time,
                )
                session.add(match)
                await session.flush()
                match_id = match.id
                
            await session.commit()
            return match_id
            
    async def get_match_by_external_id(self, external_id: str) -> Optional[Match]:
        """Busca partida pelo ID externo."""
        async with self.async_session() as session:
            result = await session.execute(
                select(Match).where(Match.external_id == external_id)
            )
            return result.scalar_one_or_none()
            
    async def update_match_result(
        self, 
        match_id: int, 
        home_score: int, 
        away_score: int
    ):
        """Atualiza resultado de uma partida."""
        async with self.async_session() as session:
            await session.execute(
                update(Match)
                .where(Match.id == match_id)
                .values(
                    home_score=home_score,
                    away_score=away_score,
                    status="finished"
                )
            )
            await session.commit()
            
    # ==========================================
    # ODDS HISTORY
    # ==========================================
    
    async def save_odds(
        self,
        match_id: int,
        ah_line: str,
        side: str,
        best_odds: float,
        best_bookmaker: str,
        second_best_odds: float = None,
        second_best_bookmaker: str = None,
        median_odds: float = None,
        num_bookmakers: int = None,
        # Campos legados para compatibilidade
        bookmaker: str = None,
        home_odds: float = None,
        away_odds: float = None,
    ):
        """
        Salva registro de odds no histórico.
        
        Args:
            match_id: ID da partida
            ah_line: Linha de AH (ex: "-0.5", "+1")
            side: "home" ou "away"
            best_odds: Melhor odd
            best_bookmaker: Casa com melhor odd
            second_best_odds: Segunda melhor odd
            second_best_bookmaker: Casa com segunda melhor odd
            median_odds: Mediana das odds
            num_bookmakers: Número de casas com odds
        """
        async with self.async_session() as session:
            odds = OddsHistory(
                match_id=match_id,
                ah_line=ah_line,
                side=side,
                best_odds=best_odds,
                best_bookmaker=best_bookmaker,
                second_best_odds=second_best_odds,
                second_best_bookmaker=second_best_bookmaker,
                median_odds=median_odds,
                num_bookmakers=num_bookmakers,
                # Campos legados
                bookmaker=bookmaker or best_bookmaker,
                home_odds=home_odds if home_odds else (best_odds if side == "home" else None),
                away_odds=away_odds if away_odds else (best_odds if side == "away" else None),
            )
            session.add(odds)
            await session.commit()
    
    async def save_best_odds(
        self,
        match_id: int,
        ah_line: str,
        best_home_odds: float,
        best_away_odds: float,
    ):
        """
        Salva registro de best odds (coleta rápida sem cliques).
        
        Args:
            match_id: ID da partida
            ah_line: Linha de AH (ex: "-0.5", "+1")
            best_home_odds: Melhor odd para home
            best_away_odds: Melhor odd para away
        """
        async with self.async_session() as session:
            best_odds = BestOddsHistory(
                match_id=match_id,
                ah_line=ah_line,
                best_home_odds=best_home_odds,
                best_away_odds=best_away_odds,
            )
            session.add(best_odds)
            await session.commit()
            
    # ==========================================
    # OPPORTUNITIES
    # ==========================================
    
    async def save_opportunity(
        self,
        match_id: int,
        ah_line: str,
        side: str,
        best_odds: float,
        best_bookmaker: str,
        num_bookmakers: int,
        dif_pct_best_second: float = None,
        dif_pct_best_median: float = None,
        dif_vs_pinnacle: float = None,
        minutes_to_kickoff: int = None,
    ) -> int:
        """Salva oportunidade detectada."""
        async with self.async_session() as session:
            opp = Opportunity(
                match_id=match_id,
                ah_line=ah_line,
                side=side,
                best_odds=best_odds,
                best_bookmaker=best_bookmaker,
                num_bookmakers=num_bookmakers,
                dif_pct_best_second=dif_pct_best_second,
                dif_pct_best_median=dif_pct_best_median,
                dif_vs_pinnacle=dif_vs_pinnacle,
                minutes_to_kickoff=minutes_to_kickoff,
            )
            session.add(opp)
            await session.flush()
            opp_id = opp.id
            await session.commit()
            return opp_id
            
    async def update_opportunity_scoring(
        self,
        opp_id: int,
        proba: float,
        cutoff: float,
        decision: bool,
    ):
        """Atualiza oportunidade com resultado do scoring."""
        async with self.async_session() as session:
            await session.execute(
                update(Opportunity)
                .where(Opportunity.id == opp_id)
                .values(
                    proba=proba,
                    cutoff=cutoff,
                    decision=decision,
                    scored_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()
            
    async def update_opportunity_closing(
        self,
        opp_id: int,
        closing_odds: float,
    ):
        """Atualiza oportunidade com closing line e calcula CLV."""
        async with self.async_session() as session:
            # Busca a oportunidade
            result = await session.execute(
                select(Opportunity).where(Opportunity.id == opp_id)
            )
            opp = result.scalar_one_or_none()
            
            if opp and opp.best_odds:
                clv = (opp.best_odds - closing_odds) / closing_odds
                
                await session.execute(
                    update(Opportunity)
                    .where(Opportunity.id == opp_id)
                    .values(
                        closing_odds=closing_odds,
                        closing_time=datetime.now(timezone.utc),
                        clv=clv,
                        clv_positive=clv > 0,
                    )
                )
                await session.commit()
                
    async def get_opportunities_for_training(
        self,
        min_date: datetime = None,
        max_date: datetime = None,
    ) -> List[Opportunity]:
        """Busca oportunidades para treinar o modelo."""
        async with self.async_session() as session:
            query = select(Opportunity).where(
                Opportunity.clv.isnot(None)  # Só com CLV calculado
            )
            
            if min_date:
                query = query.where(Opportunity.detection_time >= min_date)
            if max_date:
                query = query.where(Opportunity.detection_time <= max_date)
                
            result = await session.execute(query)
            return result.scalars().all()
            
    # ==========================================
    # BETS
    # ==========================================
    
    async def save_bet(
        self,
        opportunity_id: int,
        match_id: int,
        ah_line: str,
        side: str,
        bookmaker: str,
        expected_odds: float,
        stake: float,
    ) -> int:
        """Salva aposta a ser executada."""
        async with self.async_session() as session:
            bet = Bet(
                opportunity_id=opportunity_id,
                match_id=match_id,
                ah_line=ah_line,
                side=side,
                bookmaker=bookmaker,
                expected_odds=expected_odds,
                stake=stake,
                potential_return=stake * expected_odds,
                status="pending",
            )
            session.add(bet)
            await session.flush()
            bet_id = bet.id
            await session.commit()
            return bet_id
            
    async def update_bet_executed(
        self,
        bet_id: int,
        actual_odds: float,
        confirmation_id: str,
    ):
        """Atualiza aposta como executada."""
        async with self.async_session() as session:
            await session.execute(
                update(Bet)
                .where(Bet.id == bet_id)
                .values(
                    status="placed",
                    actual_odds=actual_odds,
                    confirmation_id=confirmation_id,
                    executed_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()
            
    async def update_bet_rejected(self, bet_id: int, error_message: str):
        """Marca aposta como rejeitada."""
        async with self.async_session() as session:
            await session.execute(
                update(Bet)
                .where(Bet.id == bet_id)
                .values(
                    status="rejected",
                    error_message=error_message,
                )
            )
            await session.commit()
            
    async def update_bet_result(
        self,
        bet_id: int,
        result: str,
        profit_loss: float,
    ):
        """Atualiza resultado final da aposta."""
        async with self.async_session() as session:
            await session.execute(
                update(Bet)
                .where(Bet.id == bet_id)
                .values(
                    result=result,
                    profit_loss=profit_loss,
                    settled_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()
            
    async def get_pending_bets(self) -> List[Bet]:
        """Retorna apostas pendentes de resultado."""
        async with self.async_session() as session:
            result = await session.execute(
                select(Bet).where(Bet.result == "pending")
            )
            return result.scalars().all()
            
    async def bet_exists(
        self,
        match_id: int,
        ah_line: str,
        side: str,
    ) -> bool:
        """Verifica se já existe aposta para esta combinação."""
        async with self.async_session() as session:
            result = await session.execute(
                select(Bet).where(
                    Bet.match_id == match_id,
                    Bet.ah_line == ah_line,
                    Bet.side == side,
                )
            )
            return result.scalar_one_or_none() is not None
            
    # ==========================================
    # LEAGUE CONFIG
    # ==========================================
    
    async def get_active_leagues(self, tier: int = None) -> List[LeagueConfig]:
        """Retorna ligas ativas."""
        async with self.async_session() as session:
            query = select(LeagueConfig).where(LeagueConfig.is_active == True)
            
            if tier:
                query = query.where(LeagueConfig.tier == tier)
                
            result = await session.execute(query)
            return result.scalars().all()
            
    async def add_league(
        self,
        league_name: str,
        tier: int,
        scrape_interval: int = 60,
    ):
        """Adiciona nova liga."""
        async with self.async_session() as session:
            league = LeagueConfig(
                league_name=league_name,
                tier=tier,
                scrape_interval=scrape_interval,
            )
            session.add(league)
            await session.commit()


# ==========================================
# FUNÇÃO DE INICIALIZAÇÃO
# ==========================================

async def init_db(database_url: str = None):
    """
    Inicializa o banco de dados criando todas as tabelas.
    
    Uso:
        import asyncio
        from storage.database import init_db
        asyncio.run(init_db())
    """
    db = Database(database_url)
    await db.connect()
    await db.close()
    logger.info("Banco de dados inicializado com sucesso!")
