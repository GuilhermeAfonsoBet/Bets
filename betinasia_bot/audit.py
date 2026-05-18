#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Auditoria - Compara dados do scraper com o site real.

Uso:
    python audit.py                    # Audita Premier League
    python audit.py "Germany Bundesliga"  # Audita liga específica
"""

import asyncio
import sys
from datetime import datetime
from scraper.betinasia import BetinAsiaScraper

async def audit_league(league_name: str = "England Premier League"):
    """Faz auditoria detalhada de uma liga."""
    
    print("=" * 70)
    print(f"AUDITORIA - {league_name}")
    print(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    
    scraper = BetinAsiaScraper()
    await scraper.start()
    
    try:
        # Login
        if not await scraper.login():
            print("ERRO: Falha no login")
            return
        
        print(f"✓ Login OK")
        print()
        
        # Scrape da liga
        matches = await scraper.scrape_league(league_name)
        
        print(f"RESUMO: {len(matches)} partidas encontradas")
        print("-" * 70)
        print()
        
        total_ah_lines = 0
        
        for i, match in enumerate(matches, 1):
            ah_count = len(match.ah_lines)
            total_ah_lines += ah_count
            
            print(f"JOGO {i}:")
            print(f"  Times: {match.home_team} vs {match.away_team}")
            print(f"  Data: {match.kickoff_time.strftime('%d/%m/%Y %H:%M') if match.kickoff_time else 'N/A'}")
            print(f"  Linhas AH: {ah_count}")
            
            if match.ah_lines:
                print(f"  Handicaps:")
                for line_str, ah_line in sorted(match.ah_lines.items(), key=lambda x: float(x[0].replace('+', ''))):
                    for bk, odds in ah_line.bookmaker_odds.items():
                        print(f"    {line_str}: Home {odds.home_odds:.3f} | Away {odds.away_odds:.3f}")
            print()
        
        print("-" * 70)
        print(f"TOTAL: {len(matches)} partidas | {total_ah_lines} linhas de AH")
        print(f"MÉDIA: {total_ah_lines/len(matches):.1f} linhas AH por partida" if matches else "N/A")
        print("=" * 70)
        
        # Salva em arquivo para referência
        filename = f"audit_{league_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"AUDITORIA - {league_name}\n")
            f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total: {len(matches)} partidas | {total_ah_lines} linhas AH\n\n")
            
            for i, match in enumerate(matches, 1):
                f.write(f"JOGO {i}: {match.home_team} vs {match.away_team}\n")
                f.write(f"  Linhas AH: {len(match.ah_lines)}\n")
                for line_str, ah_line in sorted(match.ah_lines.items(), key=lambda x: float(x[0].replace('+', ''))):
                    for bk, odds in ah_line.bookmaker_odds.items():
                        f.write(f"    {line_str}: H {odds.home_odds:.3f} | A {odds.away_odds:.3f}\n")
                f.write("\n")
        
        print(f"\nArquivo salvo: {filename}")
        
    finally:
        await scraper.close()


if __name__ == "__main__":
    league = sys.argv[1] if len(sys.argv) > 1 else "England Premier League"
    asyncio.run(audit_league(league))
