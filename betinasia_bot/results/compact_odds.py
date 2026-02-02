# -*- coding: utf-8 -*-
"""
Compacta historico de odds em resumos.

Processa jogos finalizados e cria registros na tabela odds_summary.

Uso:
    python -m results.compact_odds
    python -m results.compact_odds --match-id 123
    python -m results.compact_odds --dry-run
"""

import asyncio
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
from math import sqrt
from loguru import logger
from sqlalchemy import select, and_

from storage.database import Database
from storage.models import Match, BestOddsHistory
from storage.models_summary import OddsSummary


def calculate_ah_result(
    home_score: int,
    away_score: int,
    line: float,
    side: str,
    odds: float
) -> Tuple[str, float]:
    """
    Calcula resultado de aposta Asian Handicap.
    
    Args:
        home_score: Gols do home
        away_score: Gols do away
        line: Linha de handicap (ex: -1.25)
        side: "home" ou "away"
        odds: Odds da aposta
    
    Returns:
        (result, profit_loss) para stake=1
    """
    # Calcula diferenca ajustada
    if side == "home":
        adjusted_diff = (home_score + line) - away_score
    else:  # away
        adjusted_diff = (away_score - line) - home_score
    
    # Determina resultado baseado na diferenca ajustada
    if adjusted_diff > 0.5:
        return ("win", odds - 1)
    elif adjusted_diff < -0.5:
        return ("loss", -1.0)
    elif adjusted_diff == 0.5:
        return ("half_win", (odds - 1) / 2)
    elif adjusted_diff == -0.5:
        return ("half_loss", -0.5)
    else:  # adjusted_diff == 0
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
    
    Args:
        home_score: Gols do home
        away_score: Gols do away
        line: Linha de total (ex: 2.5)
        side: "over" ou "under"
        odds: Odds da aposta
    
    Returns:
        (result, profit_loss) para stake=1
    """
    total_goals = home_score + away_score
    
    if side == "over":
        diff = total_goals - line
    else:  # under
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


def calculate_1x2_result(
    home_score: int,
    away_score: int,
    side: str,
    odds: float
) -> Tuple[str, float]:
    """
    Calcula resultado de aposta 1X2.
    
    Args:
        home_score: Gols do home
        away_score: Gols do away
        side: "home", "draw" ou "away"
        odds: Odds da aposta
    
    Returns:
        (result, profit_loss) para stake=1
    """
    if home_score > away_score:
        actual_result = "home"
    elif home_score < away_score:
        actual_result = "away"
    else:
        actual_result = "draw"
    
    if side == actual_result:
        return ("win", odds - 1)
    else:
        return ("loss", -1.0)


def calculate_statistics(odds_list: List[float]) -> Dict:
    """
    Calcula estatisticas de uma lista de odds.
    
    Returns:
        Dict com min, max, avg, std
    """
    if not odds_list:
        return {"min": 0, "max": 0, "avg": 0, "std": 0}
    
    min_val = min(odds_list)
    max_val = max(odds_list)
    avg_val = sum(odds_list) / len(odds_list)
    
    # Desvio padrao
    if len(odds_list) > 1:
        variance = sum((x - avg_val) ** 2 for x in odds_list) / len(odds_list)
        std_val = sqrt(variance)
    else:
        std_val = 0
    
    return {
        "min": min_val,
        "max": max_val,
        "avg": avg_val,
        "std": std_val
    }


def calculate_steam_moves(odds_list: List[float], threshold: float = 3.0) -> Dict:
    """
    Calcula metricas de steam moves (movimentos bruscos).
    
    Args:
        odds_list: Lista de odds em ordem cronologica
        threshold: Percentual minimo para considerar steam move
    
    Returns:
        Dict com count, max_move, avg_move
    """
    if len(odds_list) < 2:
        return {"count": 0, "max_move": 0, "avg_move": 0}
    
    moves = []
    for i in range(1, len(odds_list)):
        prev = odds_list[i - 1]
        curr = odds_list[i]
        if prev > 0:
            move_pct = abs((curr - prev) / prev) * 100
            moves.append(move_pct)
    
    if not moves:
        return {"count": 0, "max_move": 0, "avg_move": 0}
    
    return {
        "count": sum(1 for m in moves if m > threshold),
        "max_move": max(moves),
        "avg_move": sum(moves) / len(moves)
    }


async def process_match(
    session,
    match: Match,
    dry_run: bool = False
) -> int:
    """
    Processa um jogo e cria resumos de odds.
    
    Returns:
        Numero de resumos criados
    """
    logger.debug(f"Processando: {match.home_team} vs {match.away_team}")
    
    # Busca todas as odds do jogo
    result = await session.execute(
        select(BestOddsHistory)
        .where(BestOddsHistory.match_id == match.id)
        .order_by(BestOddsHistory.scraped_at)
    )
    all_odds = result.scalars().all()
    
    if not all_odds:
        logger.warning(f"  Jogo sem odds coletadas")
        return 0
    
    # Agrupa por linha
    odds_by_line: Dict[str, List[BestOddsHistory]] = {}
    for odds in all_odds:
        key = odds.ah_line
        if key not in odds_by_line:
            odds_by_line[key] = []
        odds_by_line[key].append(odds)
    
    summaries_created = 0
    
    for line_key, line_odds in odds_by_line.items():
        # Filtra apenas coletas antes do kickoff
        pre_kick_odds = [o for o in line_odds if o.scraped_at < match.kickoff_time]
        
        if len(pre_kick_odds) < 2:
            continue  # Precisa de pelo menos abertura e fechamento
        
        # Determina tipo de mercado e lado
        if line_key.startswith("OU_"):
            market_type = "OU"
            line_value = float(line_key.replace("OU_", ""))
            sides = [("over", "best_home_odds"), ("under", "best_away_odds")]
        elif line_key.startswith("1X2"):
            market_type = "1X2"
            line_value = None
            if line_key == "1X2":
                sides = [("home", "best_home_odds"), ("away", "best_away_odds")]
            else:  # 1X2_DRAW
                sides = [("draw", "best_home_odds")]
        else:
            # Asian Handicap
            market_type = "AH"
            try:
                line_value = float(line_key)
            except ValueError:
                continue
            sides = [("home", "best_home_odds"), ("away", "best_away_odds")]
        
        for side, odds_field in sides:
            # Extrai lista de odds
            odds_list = [getattr(o, odds_field) for o in pre_kick_odds if getattr(o, odds_field) > 0]
            
            if len(odds_list) < 2:
                continue
            
            # Estatisticas
            stats = calculate_statistics(odds_list)
            steam = calculate_steam_moves(odds_list)
            
            # Abertura e fechamento
            opening = pre_kick_odds[0]
            closing = pre_kick_odds[-1]
            opening_odds = getattr(opening, odds_field)
            closing_odds = getattr(closing, odds_field)
            
            if opening_odds <= 0 or closing_odds <= 0:
                continue
            
            # Movimento
            movement_pct = ((closing_odds - opening_odds) / opening_odds) * 100
            range_pct = ((stats["max"] - stats["min"]) / stats["avg"]) * 100 if stats["avg"] > 0 else 0
            
            if movement_pct > 1.0:
                direction = "up"
            elif movement_pct < -1.0:
                direction = "down"
            else:
                direction = "stable"
            
            # Tempo ate kickoff
            minutes_open = int((match.kickoff_time - opening.scraped_at).total_seconds() / 60)
            minutes_close = int((match.kickoff_time - closing.scraped_at).total_seconds() / 60)
            
            # Resultado da aposta
            bet_result = None
            profit_loss = None
            
            if match.home_score is not None and match.away_score is not None:
                if market_type == "AH":
                    bet_result, profit_loss = calculate_ah_result(
                        match.home_score, match.away_score,
                        line_value, side, closing_odds
                    )
                elif market_type == "OU":
                    bet_result, profit_loss = calculate_ou_result(
                        match.home_score, match.away_score,
                        line_value, side, closing_odds
                    )
                elif market_type == "1X2":
                    bet_result, profit_loss = calculate_1x2_result(
                        match.home_score, match.away_score,
                        side, closing_odds
                    )
            
            # CLV
            clv = opening_odds - closing_odds
            clv_pct = ((opening_odds - closing_odds) / closing_odds) * 100 if closing_odds > 0 else 0
            
            # Cria resumo
            summary = OddsSummary(
                match_id=match.id,
                event_id=match.external_id,
                home_team=match.home_team,
                away_team=match.away_team,
                league=match.league,
                country=None,  # TODO: extrair do match se disponivel
                kickoff_time=match.kickoff_time,
                
                market_type=market_type,
                line=line_value,
                side=side,
                
                opening_odds=opening_odds,
                opening_time=opening.scraped_at,
                minutes_to_kick_at_open=minutes_open,
                
                closing_odds=closing_odds,
                closing_time=closing.scraped_at,
                minutes_to_kick_at_close=minutes_close,
                
                min_odds=stats["min"],
                max_odds=stats["max"],
                avg_odds=stats["avg"],
                std_odds=stats["std"],
                num_collections=len(odds_list),
                
                movement_pct=movement_pct,
                range_pct=range_pct,
                direction=direction,
                
                steam_moves_count=steam["count"],
                max_single_move_pct=steam["max_move"],
                avg_move_per_collection=steam["avg_move"],
                
                home_score=match.home_score,
                away_score=match.away_score,
                bet_result=bet_result,
                profit_loss=profit_loss,
                
                clv=clv,
                clv_pct=clv_pct,
            )
            
            if not dry_run:
                session.add(summary)
            
            summaries_created += 1
    
    return summaries_created


async def compact_odds(
    match_id: int = None,
    dry_run: bool = False,
    force: bool = False
):
    """
    Compacta historico de odds em resumos.
    
    Args:
        match_id: ID especifico de jogo (ou None para todos pendentes)
        dry_run: Se True, apenas mostra o que seria feito
        force: Se True, reprocessa mesmo jogos ja compactados
    """
    
    db = Database()
    await db.connect()
    
    # Cria tabela se nao existir
    from storage.models_summary import OddsSummary
    async with db.engine.begin() as conn:
        await conn.run_sync(OddsSummary.__table__.create, checkfirst=True)
    
    try:
        print("=" * 70)
        print("COMPACTACAO DE ODDS")
        print("=" * 70)
        
        async with db.async_session() as session:
            # Busca jogos para processar
            if match_id:
                result = await session.execute(
                    select(Match).where(Match.id == match_id)
                )
                matches = result.scalars().all()
            else:
                # Jogos finalizados que ainda nao foram compactados
                if force:
                    result = await session.execute(
                        select(Match).where(Match.status == "finished")
                    )
                else:
                    # Exclui jogos que ja tem resumo
                    subq = select(OddsSummary.match_id).distinct()
                    result = await session.execute(
                        select(Match).where(
                            and_(
                                Match.status == "finished",
                                Match.id.notin_(subq)
                            )
                        )
                    )
                matches = result.scalars().all()
            
            print(f"Jogos a processar: {len(matches)}")
            print()
            
            if not matches:
                print("Nenhum jogo para compactar.")
                return
            
            total_summaries = 0
            
            for match in matches:
                count = await process_match(session, match, dry_run)
                total_summaries += count
                
                if count > 0:
                    print(f"  ✅ {match.home_team} vs {match.away_team}: {count} resumos")
            
            if not dry_run:
                await session.commit()
            
            print()
            print("=" * 70)
            print("RESUMO")
            print("=" * 70)
            print(f"Jogos processados: {len(matches)}")
            print(f"Resumos criados: {total_summaries}")
            
            if dry_run:
                print("\n[DRY RUN] Nenhuma alteracao foi salva.")
                
    finally:
        await db.close()


def main():
    parser = argparse.ArgumentParser(description="Compacta historico de odds")
    parser.add_argument("--match-id", type=int, help="ID especifico de jogo")
    parser.add_argument("--dry-run", action="store_true", help="Apenas mostra o que seria feito")
    parser.add_argument("--force", action="store_true", help="Reprocessa jogos ja compactados")
    
    args = parser.parse_args()
    
    asyncio.run(compact_odds(
        match_id=args.match_id,
        dry_run=args.dry_run,
        force=args.force
    ))


if __name__ == "__main__":
    main()
