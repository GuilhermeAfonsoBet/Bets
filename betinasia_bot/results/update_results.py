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


import unicodedata
import re
from difflib import SequenceMatcher


def remove_accents(text: str) -> str:
    """Remove acentos de uma string."""
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def normalize_team_name(name: str) -> str:
    """
    Normaliza nome do time para comparacao.
    Muito mais agressivo para maximizar matches.
    """
    if not name:
        return ""
    
    result = name
    
    # Remove acentos
    result = remove_accents(result)
    
    # Lowercase
    result = result.lower()
    
    # Remove sufixos comuns (ordem importa - do mais específico para o mais geral)
    suffixes_to_remove = [
        # Sufixos longos primeiro
        " football club", " futebol clube", " fussball club",
        " soccer club", " sport club", " sporting club",
        " athletic club", " atletico clube",
        " reserves", " reserve", " youth", " u23", " u21", " u19", " u18",
        " women", " ladies", " feminino",
        " amateure", " amateur", " amador",
        # Sufixos médios
        " united", " city", " town", " county", " rovers", " wanderers",
        " albion", " athletic", " hotspur", " villa",
        # Sufixos curtos
        " fc", " cf", " sc", " ac", " bc", " rc", " cd", " ud", " sd",
        " fk", " sk", " nk", " bk", " if", " ff",
        " sv", " vfb", " vfl", " tsv", " tsg", " fsv", " bsc", " bsv",
        " rcd", " rsd", " afc", " cfc", " sfc",
        " (rj)", " (sp)", " (mg)", " (ba)", " (pr)", " (rs)",  # Brasil estados
        " (ksa)", " (uae)", " (jor)", " (qat)",  # Países árabes
    ]
    
    for suffix in suffixes_to_remove:
        result = result.replace(suffix, "")
    
    # Remove prefixos comuns
    prefixes_to_remove = [
        "real ", "deportivo ", "club ", "sporting ", "athletic ",
        "fc ", "cf ", "sc ", "ac ", "as ", "us ", "ss ",
        "al ", "al-",  # Times árabes
    ]
    
    for prefix in prefixes_to_remove:
        if result.startswith(prefix):
            result = result[len(prefix):]
    
    # Remove caracteres especiais, mantém apenas letras e espaços
    result = re.sub(r'[^a-z\s]', ' ', result)
    
    # Remove espaços múltiplos
    result = ' '.join(result.split())
    
    result = result.strip()
    
    return result


# Mapeamentos conhecidos (BetinAsia -> API-Football)
TEAM_MAPPINGS = {
    # Inglaterra
    "man utd": "manchester united",
    "man city": "manchester city", 
    "manchester city": "manchester city",
    "spurs": "tottenham",
    "tottenham hotspur": "tottenham",
    "wolves": "wolverhampton",
    "wolverhampton wanderers": "wolverhampton",
    "brighton": "brighton",
    "brighton hove albion": "brighton",
    "newcastle": "newcastle",
    "newcastle utd": "newcastle",
    "west ham": "west ham",
    "west ham utd": "west ham",
    "nottm forest": "nottingham forest",
    "nottingham forest": "nottingham forest",
    "nott'm forest": "nottingham forest",
    "arsenal": "arsenal",
    "chelsea": "chelsea",
    "liverpool": "liverpool",
    "everton": "everton",
    "aston villa": "aston villa",
    "leeds": "leeds",
    "leeds utd": "leeds",
    "bournemouth": "bournemouth",
    "brentford": "brentford",
    "fulham": "fulham",
    "crystal palace": "crystal palace",
    "leicester": "leicester",
    "southampton": "southampton",
    
    # Espanha
    "atletico madrid": "atletico madrid",
    "atletico de madrid": "atletico madrid",
    "atlético madrid": "atletico madrid",
    "athletic bilbao": "athletic bilbao",
    "athletic club": "athletic bilbao",
    "real sociedad": "real sociedad",
    "real betis": "real betis",
    "real madrid": "real madrid",
    "barcelona": "barcelona",
    "sevilla": "sevilla",
    "villarreal": "villarreal",
    "valencia": "valencia",
    "osasuna": "osasuna",
    "celta vigo": "celta vigo",
    "mallorca": "mallorca",
    "getafe": "getafe",
    "girona": "girona",
    "alaves": "alaves",
    "cadiz": "cadiz",
    "almeria": "almeria",
    "las palmas": "las palmas",
    "rayo vallecano": "rayo vallecano",
    "elche": "elche",
    "levante": "levante",
    "espanyol": "espanyol",
    "malaga": "malaga",
    "mirandes": "mirandes",
    "oviedo": "oviedo",
    "real oviedo": "oviedo",
    
    # Alemanha  
    "bayern munich": "bayern munich",
    "bayern munchen": "bayern munich",
    "bayern münchen": "bayern munich",
    "borussia dortmund": "borussia dortmund",
    "dortmund": "borussia dortmund",
    "rb leipzig": "rb leipzig",
    "rasenballsport leipzig": "rb leipzig",
    "bayer leverkusen": "bayer leverkusen",
    "leverkusen": "bayer leverkusen",
    "eintracht frankfurt": "eintracht frankfurt",
    "frankfurt": "eintracht frankfurt",
    "wolfsburg": "wolfsburg",
    "monchengladbach": "monchengladbach",
    "borussia monchengladbach": "monchengladbach",
    "werder bremen": "werder bremen",
    "bremen": "werder bremen",
    "hoffenheim": "hoffenheim",
    "tsg hoffenheim": "hoffenheim",
    "freiburg": "freiburg",
    "mainz": "mainz",
    "mainz 05": "mainz",
    "koln": "koln",
    "cologne": "koln",
    "augsburg": "augsburg",
    "union berlin": "union berlin",
    "hertha berlin": "hertha berlin",
    "st pauli": "st pauli",
    "fc st pauli": "st pauli",
    "hamburger": "hamburger sv",
    "hamburg": "hamburger sv",
    "hsv": "hamburger sv",
    
    # Itália
    "juventus": "juventus",
    "inter": "inter",
    "inter milan": "inter",
    "internazionale": "inter",
    "ac milan": "ac milan",
    "milan": "ac milan",
    "napoli": "napoli",
    "roma": "roma",
    "as roma": "roma",
    "lazio": "lazio",
    "atalanta": "atalanta",
    "fiorentina": "fiorentina",
    "torino": "torino",
    "bologna": "bologna",
    "udinese": "udinese",
    "sassuolo": "sassuolo",
    "verona": "verona",
    "hellas verona": "verona",
    "monza": "monza",
    "lecce": "lecce",
    "empoli": "empoli",
    "cagliari": "cagliari",
    "genoa": "genoa",
    "salernitana": "salernitana",
    "frosinone": "frosinone",
    
    # França
    "psg": "paris saint germain",
    "paris saint-germain": "paris saint germain",
    "paris sg": "paris saint germain",
    "marseille": "marseille",
    "olympique marseille": "marseille",
    "lyon": "lyon",
    "olympique lyon": "lyon",
    "monaco": "monaco",
    "lille": "lille",
    "lens": "lens",
    "rennes": "rennes",
    "nice": "nice",
    "reims": "reims",
    "montpellier": "montpellier",
    "toulouse": "toulouse",
    "strasbourg": "strasbourg",
    "nantes": "nantes",
    "brest": "brest",
    "lorient": "lorient",
    "clermont": "clermont",
    "metz": "metz",
    "le havre": "le havre",
    
    # Portugal
    "benfica": "benfica",
    "porto": "porto",
    "fc porto": "porto",
    "sporting": "sporting cp",
    "sporting cp": "sporting cp",
    "sporting lisbon": "sporting cp",
    "braga": "braga",
    "sc braga": "braga",
    "guimaraes": "guimaraes",
    "vitoria guimaraes": "guimaraes",
    "casa pia": "casa pia",
    "avs": "avs",
    "avs sad": "avs",
    
    # Holanda
    "ajax": "ajax",
    "psv": "psv",
    "psv eindhoven": "psv",
    "feyenoord": "feyenoord",
    "az alkmaar": "az alkmaar",
    "az": "az alkmaar",
    "twente": "twente",
    "fc twente": "twente",
    "utrecht": "utrecht",
    
    # Rússia
    "zenit": "zenit",
    "zenit st petersburg": "zenit",
    "fk zenit st petersburg": "zenit",
    "spartak moscow": "spartak moscow",
    "spartak moskva": "spartak moscow",
    "cska moscow": "cska moscow",
    "cska moskva": "cska moscow",
    "lokomotiv moscow": "lokomotiv moscow",
    "lokomotiv moskva": "lokomotiv moscow",
    "dinamo moscow": "dinamo moscow",
    "dinamo moskva": "dinamo moscow",
    "fk dinamo moskva": "dinamo moscow",
    
    # Turquia
    "galatasaray": "galatasaray",
    "fenerbahce": "fenerbahce",
    "fenerbahçe": "fenerbahce",
    "besiktas": "besiktas",
    "trabzonspor": "trabzonspor",
    "kocaelispor": "kocaelispor",
    
    # Áustria
    "salzburg": "salzburg",
    "red bull salzburg": "salzburg",
    "rb salzburg": "salzburg",
    "rapid wien": "rapid wien",
    "rapid vienna": "rapid wien",
    "austria wien": "austria wien",
    "austria vienna": "austria wien",
    "lask": "lask",
    "lask linz": "lask",
    "spg wels": "spg wels",
    
    # Outros
    "celtic": "celtic",
    "rangers": "rangers",
    "anderlecht": "anderlecht",
    "club brugge": "club brugge",
    "brugge": "club brugge",
}


def get_mapped_name(name: str) -> str:
    """Busca nome no mapeamento."""
    normalized = normalize_team_name(name)
    
    # Busca exata no mapeamento
    if normalized in TEAM_MAPPINGS:
        return TEAM_MAPPINGS[normalized]
    
    # Busca parcial (se o nome normalizado contém uma chave)
    for key, value in TEAM_MAPPINGS.items():
        if key in normalized or normalized in key:
            return value
    
    return normalized


def similarity_ratio(s1: str, s2: str) -> float:
    """Calcula similaridade entre duas strings (0-1)."""
    return SequenceMatcher(None, s1, s2).ratio()


def word_overlap_score(s1: str, s2: str) -> float:
    """Calcula score baseado em palavras em comum."""
    words1 = set(s1.split())
    words2 = set(s2.split())
    
    if not words1 or not words2:
        return 0.0
    
    # Palavras em comum
    common = words1 & words2
    
    # Ignora palavras muito curtas
    common = {w for w in common if len(w) > 2}
    
    if not common:
        return 0.0
    
    # Score = proporção de palavras em comum
    return len(common) / min(len(words1), len(words2))


def match_single_team(name1: str, name2: str, threshold: float = 0.6) -> bool:
    """
    Verifica se dois nomes de time correspondem.
    Usa múltiplas estratégias de matching.
    """
    # Normaliza
    n1 = normalize_team_name(name1)
    n2 = normalize_team_name(name2)
    
    # 1. Match exato após normalização
    if n1 == n2:
        return True
    
    # 2. Busca no mapeamento
    m1 = get_mapped_name(name1)
    m2 = get_mapped_name(name2)
    if m1 == m2:
        return True
    
    # 3. Um contém o outro
    if n1 in n2 or n2 in n1:
        return True
    if m1 in m2 or m2 in m1:
        return True
    
    # 4. Similaridade de string (fuzzy matching)
    if similarity_ratio(n1, n2) >= threshold:
        return True
    if similarity_ratio(m1, m2) >= threshold:
        return True
    
    # 5. Overlap de palavras significativas
    if word_overlap_score(n1, n2) >= 0.5:
        return True
    
    # 6. Primeira palavra igual (geralmente o nome principal)
    words1 = n1.split()
    words2 = n2.split()
    if words1 and words2 and len(words1[0]) > 3 and words1[0] == words2[0]:
        return True
    
    return False


def match_teams(
    betinasia_home: str, 
    betinasia_away: str,
    api_home: str,
    api_away: str
) -> bool:
    """
    Verifica se os times correspondem.
    Usa matching robusto com múltiplas estratégias.
    """
    # Match normal
    home_match = match_single_team(betinasia_home, api_home)
    away_match = match_single_team(betinasia_away, api_away)
    
    if home_match and away_match:
        return True
    
    # Tenta match invertido (caso raro de ordem trocada)
    # home_inv = match_single_team(betinasia_home, api_away)
    # away_inv = match_single_team(betinasia_away, api_home)
    # if home_inv and away_inv:
    #     return True
    
    return False


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
