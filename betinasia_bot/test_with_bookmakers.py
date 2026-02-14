# -*- coding: utf-8 -*-
"""
Teste de scraping com captura de odds por bookmaker.
Mostra todas as métricas solicitadas.
"""

import asyncio
import sys
sys.path.insert(0, '.')

from scraper.betinasia import BetinAsiaScraper


async def test_with_bookmakers():
    print("\n" + "="*70)
    print("TESTE: SCRAPING COM CAPTURA DE BOOKMAKERS")
    print("="*70 + "\n")
    
    async with BetinAsiaScraper(headless=True) as scraper:
        # Login
        print("[1] Fazendo login...")
        success = await scraper.login(
            username="JomanaSilva",
            password="Jom1928@"
        )
        
        if not success:
            print("    ERRO: Login falhou!")
            return
        print("    OK!")
        
        # Acessa um jogo específico
        print("\n[2] Acessando jogo Brighton vs Everton...")
        
        match = await scraper._scrape_single_match(
            match_url="https://black.betinasia.com/sportsbook/football/XE/1/2026-01-31,13,33",
            league_name="England Premier League",
            capture_bookmakers=True  # ATIVA captura de bookmakers
        )
        
        if not match:
            print("    ERRO: Não conseguiu fazer scrape!")
            return
        
        print(f"    Jogo: {match.home_team} vs {match.away_team}")
        print(f"    Linhas AH: {len(match.ah_lines)}")
        
        # Mostra métricas das linhas principais
        print("\n" + "="*70)
        print("MÉTRICAS DAS LINHAS PRINCIPAIS")
        print("="*70)
        
        # Linhas principais (próximas de 0)
        main_lines = ["-0.75", "-0.5", "-0.25", "0", "+0.25", "+0.5", "+0.75"]
        
        for line_name in main_lines:
            if line_name not in match.ah_lines:
                continue
            
            ah_line = match.ah_lines[line_name]
            
            print(f"\n{'='*50}")
            print(f"AH {line_name} ({ah_line.num_bookmakers} bookmakers)")
            print(f"{'='*50}")
            
            # Métricas HOME
            home_metrics = ah_line.get_metrics_summary("home")
            print(f"\n  HOME:")
            print(f"    1. Maior odd:         {home_metrics['maior_odd']:.3f}")
            print(f"    2. Segunda maior:     {home_metrics['segunda_maior_odd']:.3f}")
            print(f"    3. Odd mediana:       {home_metrics['odd_mediana']:.3f}")
            print(f"    4. Número de casas:   {home_metrics['num_casas']}")
            print(f"    5. Casa maior odd:    {home_metrics['casa_maior_odd']}")
            print(f"    6. Casa 2ª maior:     {home_metrics['casa_segunda_maior']}")
            print(f"    - Dif% best/2nd:      {home_metrics['dif_pct_best_second']:.2f}%")
            print(f"    - Dif% best/median:   {home_metrics['dif_pct_best_median']:.2f}%")
            if home_metrics['pinnacle_odds']:
                print(f"    - Pinnacle:           {home_metrics['pinnacle_odds']:.3f}")
            
            # Métricas AWAY
            away_metrics = ah_line.get_metrics_summary("away")
            print(f"\n  AWAY:")
            print(f"    1. Maior odd:         {away_metrics['maior_odd']:.3f}")
            print(f"    2. Segunda maior:     {away_metrics['segunda_maior_odd']:.3f}")
            print(f"    3. Odd mediana:       {away_metrics['odd_mediana']:.3f}")
            print(f"    4. Número de casas:   {away_metrics['num_casas']}")
            print(f"    5. Casa maior odd:    {away_metrics['casa_maior_odd']}")
            print(f"    6. Casa 2ª maior:     {away_metrics['casa_segunda_maior']}")
            print(f"    - Dif% best/2nd:      {away_metrics['dif_pct_best_second']:.2f}%")
            print(f"    - Dif% best/median:   {away_metrics['dif_pct_best_median']:.2f}%")
            if away_metrics['pinnacle_odds']:
                print(f"    - Pinnacle:           {away_metrics['pinnacle_odds']:.3f}")
            
            # Lista todos os bookmakers
            if ah_line.num_bookmakers > 1:
                print(f"\n  Todas as odds:")
                for bk_name, bk_odds in sorted(
                    ah_line.bookmaker_odds.items(), 
                    key=lambda x: x[1].home_odds, 
                    reverse=True
                ):
                    if bk_name != "best":
                        print(f"    {bk_name:12} H={bk_odds.home_odds:.3f}  A={bk_odds.away_odds:.3f}")
        
        print("\n" + "="*70)
        print("TESTE CONCLUÍDO!")
        print("="*70)


if __name__ == "__main__":
    asyncio.run(test_with_bookmakers())
