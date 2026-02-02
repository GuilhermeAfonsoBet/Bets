# -*- coding: utf-8 -*-
"""
Atualiza eventos de hipóteses com closing line, CLV e resultado.

Após os jogos terminarem, este script:
1. Busca eventos de hipóteses sem CLV calculado
2. Busca a closing line (última odd antes do kickoff)
3. Calcula CLV
4. Calcula resultado da aposta hipotética (win/loss/etc)
5. Calcula P&L

Uso:
    python -m results.update_hypothesis_results
    python -m results.update_hypothesis_results --match-id 123
    python -m results.update_hypothesis_results --dry-run
"""

import asyncio
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
from loguru import logger
from sqlalchemy import select, and_, update

from storage.database import Database
from storage.models import Match, BestOddsHistory
from storage.models_hypothesis import (
    H1PricingEvent, H3LineMonotonicityEvent,
    H3bTemporalReversalEvent, H6CorrelationLagEvent
)


def calculate_ah_result(
    home_score: int,
    away_score: int,
    line: float,
    side: str,
    odds: float
) -> Tuple[str, float]:
    """
    Calcula resultado de aposta Asian Handicap.
    
    Returns:
        (result, profit_loss) para stake=1
    """
    if side in ("home", "side_a"):
        adjusted_diff = (home_score + line) - away_score
    else:  # away, side_b
        adjusted_diff = (away_score - line) - home_score
    
    if adjusted_diff > 0.5:
        return ("win", odds - 1)
    elif adjusted_diff < -0.5:
        return ("loss", -1.0)
    elif adjusted_diff == 0.5:
        return ("half_win", (odds - 1) / 2)
    elif adjusted_diff == -0.5:
        return ("half_loss", -0.5)
    else:
        return ("push", 0.0)


def calculate_ou_result(
    home_score: int,
    away_score: int,
    line: float,
    side: str,
    odds: float
) -> Tuple[str, float]:
    """
    Calcula resultado de aposta Over/Under.
    """
    total_goals = home_score + away_score
    
    if side in ("over", "side_a"):
        diff = total_goals - line
    else:  # under, side_b
        diff = line - total_goals
    
    if diff > 0.5:
        return ("win", odds - 1)
    elif diff < -0.5:
        return ("loss", -1.0)
    elif diff == 0.5:
        return ("half_win", (odds - 1) / 2)
    elif diff == -0.5:
        return ("half_loss", -0.5)
    else:
        return ("push", 0.0)


async def get_closing_odd(
    session,
    match: Match,
    market_type: str,
    line: str,
    side: str
) -> Optional[float]:
    """
    Busca a closing line (última odd antes do kickoff).
    """
    # Determina o campo de odd baseado no lado
    if side in ("home", "over", "side_a"):
        odd_field = BestOddsHistory.best_home_odds
    else:
        odd_field = BestOddsHistory.best_away_odds
    
    # Constrói a linha para busca
    if market_type == "OU":
        ah_line_search = f"OU_{line}"
    elif market_type == "1X2":
        ah_line_search = "1X2"
    else:
        ah_line_search = line
    
    # Busca última odd antes do kickoff
    result = await session.execute(
        select(BestOddsHistory)
        .where(
            and_(
                BestOddsHistory.match_id == match.id,
                BestOddsHistory.ah_line == ah_line_search,
                BestOddsHistory.scraped_at < match.kickoff_time
            )
        )
        .order_by(BestOddsHistory.scraped_at.desc())
        .limit(1)
    )
    
    closing = result.scalar_one_or_none()
    if closing:
        if side in ("home", "over", "side_a"):
            return closing.best_home_odds
        else:
            return closing.best_away_odds
    
    return None


async def process_h1_events(session, match: Match, dry_run: bool = False) -> int:
    """Processa eventos H1 para um jogo."""
    result = await session.execute(
        select(H1PricingEvent).where(
            and_(
                H1PricingEvent.match_id == match.id,
                H1PricingEvent.clv.is_(None)
            )
        )
    )
    events = result.scalars().all()
    
    updated = 0
    for event in events:
        if not event.recommended_odd or not event.recommended_side:
            continue
        
        # Busca closing line
        side_for_closing = "home" if event.recommended_side == "side_a" else "away"
        closing = await get_closing_odd(
            session, match, event.market_type, event.ah_line, side_for_closing
        )
        
        if not closing:
            continue
        
        # Calcula CLV
        clv = event.recommended_odd - closing
        clv_pct = (clv / closing * 100) if closing > 0 else 0
        
        # Calcula resultado
        bet_result = None
        profit_loss = None
        
        if match.home_score is not None and match.away_score is not None:
            try:
                line_value = float(event.ah_line) if event.market_type == "AH" else 0
                
                if event.market_type == "AH":
                    bet_result, profit_loss = calculate_ah_result(
                        match.home_score, match.away_score,
                        line_value, side_for_closing, event.recommended_odd
                    )
                elif event.market_type == "OU":
                    line_value = float(event.ah_line.replace("OU_", "")) if "OU_" in event.ah_line else float(event.ah_line)
                    bet_result, profit_loss = calculate_ou_result(
                        match.home_score, match.away_score,
                        line_value, side_for_closing, event.recommended_odd
                    )
            except (ValueError, TypeError):
                pass
        
        if not dry_run:
            event.closing_odd_recommended = closing
            event.clv = clv
            event.clv_pct = clv_pct
            event.bet_result = bet_result
            event.profit_loss = profit_loss
        
        updated += 1
    
    return updated


async def process_h3_events(session, match: Match, dry_run: bool = False) -> int:
    """Processa eventos H3 para um jogo."""
    result = await session.execute(
        select(H3LineMonotonicityEvent).where(
            and_(
                H3LineMonotonicityEvent.match_id == match.id,
                H3LineMonotonicityEvent.clv.is_(None)
            )
        )
    )
    events = result.scalars().all()
    
    updated = 0
    for event in events:
        if not event.recommended_odd or not event.recommended_line:
            continue
        
        # Busca closing line
        closing = await get_closing_odd(
            session, match, "AH", event.recommended_line, event.side
        )
        
        if not closing:
            continue
        
        # Calcula CLV
        clv = event.recommended_odd - closing
        clv_pct = (clv / closing * 100) if closing > 0 else 0
        
        # Calcula resultado
        bet_result = None
        profit_loss = None
        
        if match.home_score is not None and match.away_score is not None:
            try:
                line_value = float(event.recommended_line)
                bet_result, profit_loss = calculate_ah_result(
                    match.home_score, match.away_score,
                    line_value, event.side, event.recommended_odd
                )
            except (ValueError, TypeError):
                pass
        
        if not dry_run:
            event.closing_odd_recommended = closing
            event.clv = clv
            event.clv_pct = clv_pct
            event.bet_result = bet_result
            event.profit_loss = profit_loss
        
        updated += 1
    
    return updated


async def process_h3b_events(session, match: Match, dry_run: bool = False) -> int:
    """Processa eventos H3b para um jogo."""
    result = await session.execute(
        select(H3bTemporalReversalEvent).where(
            and_(
                H3bTemporalReversalEvent.match_id == match.id,
                H3bTemporalReversalEvent.clv.is_(None)
            )
        )
    )
    events = result.scalars().all()
    
    updated = 0
    for event in events:
        if not event.bet_odd:
            continue
        
        # Busca closing line
        closing = await get_closing_odd(
            session, match, event.market_type, event.ah_line, event.side
        )
        
        if not closing:
            continue
        
        # Calcula CLV
        clv = event.bet_odd - closing
        clv_pct = (clv / closing * 100) if closing > 0 else 0
        
        # Calcula resultado
        bet_result = None
        profit_loss = None
        
        if match.home_score is not None and match.away_score is not None:
            try:
                if event.market_type == "AH":
                    line_value = float(event.ah_line)
                    bet_result, profit_loss = calculate_ah_result(
                        match.home_score, match.away_score,
                        line_value, event.side, event.bet_odd
                    )
                elif event.market_type == "OU":
                    line_value = float(event.ah_line.replace("OU_", "")) if "OU_" in event.ah_line else float(event.ah_line)
                    bet_result, profit_loss = calculate_ou_result(
                        match.home_score, match.away_score,
                        line_value, event.side, event.bet_odd
                    )
            except (ValueError, TypeError):
                pass
        
        if not dry_run:
            event.closing_odd = closing
            event.clv = clv
            event.clv_pct = clv_pct
            event.bet_result = bet_result
            event.profit_loss = profit_loss
        
        updated += 1
    
    return updated


async def process_h6_events(session, match: Match, dry_run: bool = False) -> int:
    """Processa eventos H6 para um jogo."""
    result = await session.execute(
        select(H6CorrelationLagEvent).where(
            and_(
                H6CorrelationLagEvent.match_id == match.id,
                H6CorrelationLagEvent.clv.is_(None)
            )
        )
    )
    events = result.scalars().all()
    
    updated = 0
    for event in events:
        if not event.bet_odd:
            continue
        
        # Busca closing line
        closing = await get_closing_odd(
            session, match, event.bet_market_type or event.lagged_market_type,
            event.bet_line or event.lagged_line,
            event.bet_side or event.lagged_side
        )
        
        if not closing:
            continue
        
        # Calcula CLV
        clv = event.bet_odd - closing
        clv_pct = (clv / closing * 100) if closing > 0 else 0
        
        # Calcula resultado
        bet_result = None
        profit_loss = None
        
        if match.home_score is not None and match.away_score is not None:
            try:
                line_str = event.bet_line or event.lagged_line
                market_type = event.bet_market_type or event.lagged_market_type
                side = event.bet_side or event.lagged_side
                
                if market_type == "AH":
                    line_value = float(line_str)
                    bet_result, profit_loss = calculate_ah_result(
                        match.home_score, match.away_score,
                        line_value, side, event.bet_odd
                    )
                elif market_type == "OU":
                    line_value = float(line_str.replace("OU_", "")) if "OU_" in line_str else float(line_str)
                    bet_result, profit_loss = calculate_ou_result(
                        match.home_score, match.away_score,
                        line_value, side, event.bet_odd
                    )
            except (ValueError, TypeError):
                pass
        
        if not dry_run:
            event.closing_odd = closing
            event.clv = clv
            event.clv_pct = clv_pct
            event.bet_result = bet_result
            event.profit_loss = profit_loss
        
        updated += 1
    
    return updated


async def update_hypothesis_results(
    match_id: int = None,
    dry_run: bool = False
):
    """
    Atualiza eventos de hipóteses com CLV e resultado.
    """
    db = Database()
    await db.connect()
    
    try:
        print("=" * 70)
        print("ATUALIZAÇÃO DE RESULTADOS - EVENTOS DE HIPÓTESES")
        print("=" * 70)
        
        async with db.async_session() as session:
            # Busca jogos finalizados
            if match_id:
                result = await session.execute(
                    select(Match).where(Match.id == match_id)
                )
            else:
                result = await session.execute(
                    select(Match).where(
                        and_(
                            Match.status == "finished",
                            Match.home_score.isnot(None),
                            Match.away_score.isnot(None)
                        )
                    )
                )
            
            matches = result.scalars().all()
            print(f"Jogos finalizados: {len(matches)}")
            print()
            
            total_h1 = 0
            total_h3 = 0
            total_h3b = 0
            total_h6 = 0
            
            for match in matches:
                h1 = await process_h1_events(session, match, dry_run)
                h3 = await process_h3_events(session, match, dry_run)
                h3b = await process_h3b_events(session, match, dry_run)
                h6 = await process_h6_events(session, match, dry_run)
                
                if h1 + h3 + h3b + h6 > 0:
                    print(f"  {match.home_team} vs {match.away_team}: H1={h1}, H3={h3}, H3b={h3b}, H6={h6}")
                
                total_h1 += h1
                total_h3 += h3
                total_h3b += h3b
                total_h6 += h6
            
            if not dry_run:
                await session.commit()
            
            print()
            print("=" * 70)
            print("RESUMO")
            print("=" * 70)
            print(f"H1 (Precificação): {total_h1} eventos atualizados")
            print(f"H3 (Linhas adjacentes): {total_h3} eventos atualizados")
            print(f"H3b (Reversões temporais): {total_h3b} eventos atualizados")
            print(f"H6 (Correlação/Lag): {total_h6} eventos atualizados")
            print(f"TOTAL: {total_h1 + total_h3 + total_h3b + total_h6}")
            
            if dry_run:
                print("\n[DRY RUN] Nenhuma alteração foi salva.")
                
    finally:
        await db.close()


def main():
    parser = argparse.ArgumentParser(description="Atualiza eventos de hipóteses com CLV e resultado")
    parser.add_argument("--match-id", type=int, help="ID específico de jogo")
    parser.add_argument("--dry-run", action="store_true", help="Apenas mostra o que seria feito")
    
    args = parser.parse_args()
    
    asyncio.run(update_hypothesis_results(
        match_id=args.match_id,
        dry_run=args.dry_run
    ))


if __name__ == "__main__":
    main()
