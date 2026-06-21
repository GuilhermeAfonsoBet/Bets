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


async def get_hypothesis_metrics(
    session,
    match_id: int,
    market_type: str,
    line: Optional[float],
    side: str
) -> Dict:
    """
    Busca métricas de hipóteses para um mercado específico.
    
    Args:
        session: Sessão do banco
        match_id: ID do jogo
        market_type: Tipo de mercado (AH, OU, 1X2)
        line: Linha do mercado
        side: Lado (home, away, over, under)
    
    Returns:
        Dict com métricas de H1, H3, H3b, H6
    """
    metrics = {
        # H1
        "h1_pricing_events_count": 0,
        "h1_had_arb": 0,
        "h1_avg_edge": None,
        "h1_max_edge": None,
        # H3
        "h3_line_anomaly_count": 0,
        "h3_anomaly_magnitude_max": None,
        "h3_anomaly_magnitude_avg": None,
        # H3b
        "h3b_reversal_count": 0,
        "h3b_oscillation_index": None,
        "h3b_max_reversal_magnitude": None,
        "h3b_avg_reversal_magnitude": None,
        # H6
        "h6_lag_events_count": 0,
        "h6_avg_lag_seconds": None,
        "h6_max_lag_seconds": None,
    }
    
    line_str = str(line) if line is not None else market_type
    
    # ===== H1 - Precificação =====
    try:
        result = await session.execute(
            select(H1PricingEvent).where(
                and_(
                    H1PricingEvent.match_id == match_id,
                    H1PricingEvent.market_type == market_type,
                    H1PricingEvent.ah_line == line_str
                )
            )
        )
        h1_events = result.scalars().all()
        
        if h1_events:
            metrics["h1_pricing_events_count"] = len(h1_events)
            metrics["h1_had_arb"] = 1 if any(e.is_arb for e in h1_events) else 0
            edges = [e.edge_estimate for e in h1_events if e.edge_estimate]
            if edges:
                metrics["h1_avg_edge"] = sum(edges) / len(edges)
                metrics["h1_max_edge"] = max(edges)
    except Exception as e:
        logger.debug(f"Erro ao buscar H1: {e}")
    
    # ===== H3 - Monotonicidade entre linhas =====
    try:
        result = await session.execute(
            select(H3LineMonotonicityEvent).where(
                and_(
                    H3LineMonotonicityEvent.match_id == match_id,
                    H3LineMonotonicityEvent.side == side
                )
            )
        )
        h3_events = result.scalars().all()
        
        # Filtra eventos onde a linha atual está envolvida
        relevant_h3 = [
            e for e in h3_events 
            if e.line_a == line_str or e.line_b == line_str
        ]
        
        if relevant_h3:
            metrics["h3_line_anomaly_count"] = len(relevant_h3)
            magnitudes = [e.magnitude for e in relevant_h3 if e.magnitude]
            if magnitudes:
                metrics["h3_anomaly_magnitude_max"] = max(magnitudes)
                metrics["h3_anomaly_magnitude_avg"] = sum(magnitudes) / len(magnitudes)
    except Exception as e:
        logger.debug(f"Erro ao buscar H3: {e}")
    
    # ===== H3b - Reversões temporais =====
    try:
        result = await session.execute(
            select(H3bTemporalReversalEvent).where(
                and_(
                    H3bTemporalReversalEvent.match_id == match_id,
                    H3bTemporalReversalEvent.market_type == market_type,
                    H3bTemporalReversalEvent.ah_line == line_str,
                    H3bTemporalReversalEvent.side == side
                )
            )
        )
        h3b_events = result.scalars().all()
        
        if h3b_events:
            metrics["h3b_reversal_count"] = len(h3b_events)
            
            # Pega o último índice de oscilação (mais recente)
            osc_indices = [e.oscillation_index for e in h3b_events if e.oscillation_index is not None]
            if osc_indices:
                metrics["h3b_oscillation_index"] = osc_indices[-1]
            
            magnitudes = [e.reversal_magnitude for e in h3b_events if e.reversal_magnitude]
            if magnitudes:
                metrics["h3b_max_reversal_magnitude"] = max(magnitudes)
                metrics["h3b_avg_reversal_magnitude"] = sum(magnitudes) / len(magnitudes)
    except Exception as e:
        logger.debug(f"Erro ao buscar H3b: {e}")
    
    # ===== H6 - Correlação / Lag =====
    try:
        # IMPORTANTE:
        # Em alguns bancos, o schema de `h6_correlation_lag_events` pode estar "atrasado"
        # em relação ao model ORM (ex.: coluna `verification_status` não existe).
        # Um SELECT ORM referencia todas as colunas e pode ABORTAR a transação.
        #
        # Então fazemos SQL mínimo apenas com `lag_seconds` e isolamos em SAVEPOINT.
        from sqlalchemy import text

        sql = text(
            """
            SELECT lag_seconds
            FROM h6_correlation_lag_events
            WHERE match_id = :match_id
              AND (leader_line = :line_str OR lagged_line = :line_str)
            """
        )
        async with session.begin_nested():
            rows = (await session.execute(sql, {"match_id": match_id, "line_str": line_str})).fetchall()

        if rows:
            metrics["h6_lag_events_count"] = len(rows)
            lags = [float(r[0]) for r in rows if r and r[0] is not None]
            if lags:
                metrics["h6_avg_lag_seconds"] = sum(lags) / len(lags)
                metrics["h6_max_lag_seconds"] = max(lags)
    except Exception as e:
        logger.debug(f"Erro ao buscar H6: {e}")
    
    return metrics


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
            
            # Busca métricas de hipóteses
            hypothesis_metrics = await get_hypothesis_metrics(
                session, match.id, market_type, line_value, side
            )
            
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
                
                # Métricas de Hipóteses
                h1_pricing_events_count=hypothesis_metrics["h1_pricing_events_count"],
                h1_had_arb=hypothesis_metrics["h1_had_arb"],
                h1_avg_edge=hypothesis_metrics["h1_avg_edge"],
                h1_max_edge=hypothesis_metrics["h1_max_edge"],
                
                h3_line_anomaly_count=hypothesis_metrics["h3_line_anomaly_count"],
                h3_anomaly_magnitude_max=hypothesis_metrics["h3_anomaly_magnitude_max"],
                h3_anomaly_magnitude_avg=hypothesis_metrics["h3_anomaly_magnitude_avg"],
                
                h3b_reversal_count=hypothesis_metrics["h3b_reversal_count"],
                h3b_oscillation_index=hypothesis_metrics["h3b_oscillation_index"],
                h3b_max_reversal_magnitude=hypothesis_metrics["h3b_max_reversal_magnitude"],
                h3b_avg_reversal_magnitude=hypothesis_metrics["h3b_avg_reversal_magnitude"],
                
                h6_lag_events_count=hypothesis_metrics["h6_lag_events_count"],
                h6_avg_lag_seconds=hypothesis_metrics["h6_avg_lag_seconds"],
                h6_max_lag_seconds=hypothesis_metrics["h6_max_lag_seconds"],
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
                try:
                    count = await process_match(session, match, dry_run)
                    total_summaries += count

                    if count > 0:
                        print(f"  ✅ {match.home_team} vs {match.away_team}: {count} resumos")
                except Exception as e:
                    # Não deixe 1 jogo quebrar o batch inteiro (e não deixe a sessão ficar em rollback pendente)
                    try:
                        await session.rollback()
                    except Exception:
                        pass
                    logger.warning(f"Falha ao compactar match_id={getattr(match, 'id', None)}: {e}")
            
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
