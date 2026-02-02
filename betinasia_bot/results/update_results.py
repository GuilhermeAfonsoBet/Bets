# -*- coding: utf-8 -*-
"""
Atualiza resultados dos jogos no banco de dados.

Busca jogos que ja terminaram e ainda nao tem resultado,
consulta a API-Football e atualiza o banco.

Uso:
    python -m results.update_results
    
Ou com data especifica:
    python -m results.update_results --date 2026-02-01
"""

import asyncio
import argparse
from datetime import datetime, timezone, timedelta
from typing import List, Tuple
from loguru import logger
from sqlalchemy import select, update, and_

from .api_football import APIFootballClient, MatchResult
from storage.database import Database
from storage.models import Match, BestOddsHistory


# API Key (pode ser sobrescrita por variavel de ambiente)
API_KEY = "2707432f357b84409fd3212f9c1a84a5"


def normalize_team_name(name: str) -> str:
    """Normaliza nome do time para comparacao."""
    # Remove sufixos comuns
    suffixes = [" FC", " CF", " SC", " AC", " BC", " United", " City"]
    result = name
    for suffix in suffixes:
        result = result.replace(suffix, "")
    
    # Lowercase e remove espacos extras
    result = result.lower().strip()
    
    # Mapeamentos conhecidos
    mappings = {
        "man utd": "manchester united",
        "man city": "manchester city",
        "spurs": "tottenham",
        "wolves": "wolverhampton",
        "brighton": "brighton & hove albion",
        "newcastle": "newcastle united",
        "west ham": "west ham united",
        "nottm forest": "nottingham forest",
        "nott'm forest": "nottingham forest",
        "atletico madrid": "atletico de madrid",
        "atlético madrid": "atletico de madrid",
        "athletic bilbao": "athletic club",
        "real sociedad": "real sociedad san sebastian",
        "bayern munich": "bayern münchen",
        "bayern munchen": "bayern münchen",
        "rb leipzig": "rasenballsport leipzig",
        "psg": "paris saint germain",
        "paris saint-germain": "paris saint germain",
    }
    
    for k, v in mappings.items():
        if k in result:
            result = v
            break
            
    return result


def match_teams(
    betinasia_home: str, 
    betinasia_away: str,
    api_home: str,
    api_away: str
) -> bool:
    """Verifica se os times correspondem."""
    b_home = normalize_team_name(betinasia_home)
    b_away = normalize_team_name(betinasia_away)
    a_home = normalize_team_name(api_home)
    a_away = normalize_team_name(api_away)
    
    # Match exato
    if b_home == a_home and b_away == a_away:
        return True
        
    # Match parcial (um nome contem o outro)
    home_match = (b_home in a_home or a_home in b_home or 
                  any(w in a_home for w in b_home.split() if len(w) > 3))
    away_match = (b_away in a_away or a_away in b_away or
                  any(w in a_away for w in b_away.split() if len(w) > 3))
    
    return home_match and away_match


async def update_results(date: str = None, dry_run: bool = False):
    """
    Atualiza resultados dos jogos.
    
    Args:
        date: Data especifica (YYYY-MM-DD) ou None para jogos pendentes
        dry_run: Se True, apenas mostra o que seria atualizado
    """
    
    db = Database()
    await db.connect()
    
    api = APIFootballClient(API_KEY)
    await api.start()
    
    try:
        print("=" * 70)
        print("ATUALIZACAO DE RESULTADOS")
        print("=" * 70)
        
        # Verifica status da API
        status = await api.get_status()
        if status:
            print(f"API Status: {status['requests_today']}/{status['requests_limit']} requests hoje")
        print()
        
        async with db.async_session() as session:
            # Busca jogos que precisam de resultado
            now = datetime.now(timezone.utc)
            
            if date:
                # Data especifica
                target_date = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                start_of_day = target_date
                end_of_day = target_date + timedelta(days=1)
                
                query = select(Match).where(
                    and_(
                        Match.kickoff_time >= start_of_day,
                        Match.kickoff_time < end_of_day,
                        Match.status != "finished"
                    )
                )
            else:
                # Jogos que ja comecaram ha mais de 2h e nao tem resultado
                cutoff = now - timedelta(hours=2)
                query = select(Match).where(
                    and_(
                        Match.kickoff_time < cutoff,
                        Match.status != "finished"
                    )
                )
            
            result = await session.execute(query)
            matches = result.scalars().all()
            
            print(f"Jogos pendentes de resultado: {len(matches)}")
            
            if not matches:
                print("Nenhum jogo para atualizar.")
                return
            
            # Agrupa por data para minimizar requests
            matches_by_date = {}
            for match in matches:
                match_date = match.kickoff_time.strftime("%Y-%m-%d")
                if match_date not in matches_by_date:
                    matches_by_date[match_date] = []
                matches_by_date[match_date].append(match)
            
            print(f"Datas a consultar: {list(matches_by_date.keys())}")
            print()
            
            # Busca resultados para cada data
            updated = 0
            not_found = 0
            
            for match_date, date_matches in matches_by_date.items():
                print(f"\n--- {match_date} ({len(date_matches)} jogos) ---")
                
                # Busca resultados da API
                api_results = await api.get_results_by_date(match_date)
                print(f"Resultados na API: {len(api_results)}")
                
                for match in date_matches:
                    # Procura resultado correspondente
                    found_result = None
                    
                    for api_result in api_results:
                        if match_teams(
                            match.home_team, 
                            match.away_team,
                            api_result.home_team,
                            api_result.away_team
                        ):
                            found_result = api_result
                            break
                    
                    if found_result:
                        print(f"  ✅ {match.home_team} vs {match.away_team}")
                        print(f"     -> {found_result.home_score} - {found_result.away_score}")
                        
                        if not dry_run:
                            await session.execute(
                                update(Match)
                                .where(Match.id == match.id)
                                .values(
                                    home_score=found_result.home_score,
                                    away_score=found_result.away_score,
                                    status="finished"
                                )
                            )
                        updated += 1
                    else:
                        print(f"  ❌ {match.home_team} vs {match.away_team} - NAO ENCONTRADO")
                        not_found += 1
            
            if not dry_run:
                await session.commit()
            
            # Resumo
            print()
            print("=" * 70)
            print("RESUMO")
            print("=" * 70)
            print(f"Atualizados: {updated}")
            print(f"Nao encontrados: {not_found}")
            
            if dry_run:
                print("\n[DRY RUN] Nenhuma alteracao foi salva.")
                
    finally:
        await api.close()
        await db.close()


async def calculate_closing_lines():
    """
    Calcula closing lines para jogos finalizados.
    
    A closing line e a ultima odds coletada antes do kickoff.
    """
    
    db = Database()
    await db.connect()
    
    try:
        print("=" * 70)
        print("CALCULO DE CLOSING LINES")
        print("=" * 70)
        
        async with db.async_session() as session:
            # Busca jogos finalizados
            result = await session.execute(
                select(Match).where(Match.status == "finished")
            )
            finished_matches = result.scalars().all()
            
            print(f"Jogos finalizados: {len(finished_matches)}")
            
            for match in finished_matches[:5]:  # Limita para teste
                print(f"\n{match.home_team} vs {match.away_team}")
                print(f"  Kickoff: {match.kickoff_time}")
                print(f"  Resultado: {match.home_score} - {match.away_score}")
                
                # Busca ultima odds antes do kickoff (AH 0)
                result = await session.execute(
                    select(BestOddsHistory)
                    .where(
                        and_(
                            BestOddsHistory.match_id == match.id,
                            BestOddsHistory.ah_line == "0.0",
                            BestOddsHistory.scraped_at < match.kickoff_time
                        )
                    )
                    .order_by(BestOddsHistory.scraped_at.desc())
                    .limit(1)
                )
                closing = result.scalar_one_or_none()
                
                if closing:
                    print(f"  Closing Line (AH 0): H={closing.best_home_odds:.3f} A={closing.best_away_odds:.3f}")
                    print(f"  Coletada em: {closing.scraped_at}")
                else:
                    print(f"  ⚠️ Sem closing line")
                    
    finally:
        await db.close()


async def update_and_compact(date: str = None, dry_run: bool = False):
    """
    Atualiza resultados E compacta odds em uma unica execucao.
    """
    # 1. Atualiza resultados
    await update_results(date=date, dry_run=dry_run)
    
    # 2. Compacta odds dos jogos finalizados
    if not dry_run:
        print()
        from .compact_odds import compact_odds
        await compact_odds(dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser(description="Atualiza resultados dos jogos")
    parser.add_argument("--date", type=str, help="Data especifica (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="Apenas mostra o que seria atualizado")
    parser.add_argument("--closing-lines", action="store_true", help="Calcula closing lines")
    parser.add_argument("--no-compact", action="store_true", help="Nao compacta odds apos atualizar")
    
    args = parser.parse_args()
    
    if args.closing_lines:
        asyncio.run(calculate_closing_lines())
    elif args.no_compact:
        asyncio.run(update_results(date=args.date, dry_run=args.dry_run))
    else:
        asyncio.run(update_and_compact(date=args.date, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
