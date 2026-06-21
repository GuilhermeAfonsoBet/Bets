"""
Script para verificar a acurácia do scraping de bookmakers.
Analisa quantos dados estão completos vs incompletos.
"""
import asyncio
import sys
sys.path.insert(0, '.')

from scraper.betinasia import BetinAsiaScraper

async def check_accuracy():
    print("\n" + "="*70)
    print("VERIFICAÇÃO DE ACURÁCIA DO SCRAPING")
    print("="*70)
    
    scraper = BetinAsiaScraper()
    await scraper.start()
    
    try:
        await scraper.login()
        
        # Scrape do jogo
        match = await scraper._scrape_single_match(
            match_url="https://black.betinasia.com/sportsbook/football/XE/1/2026-01-31,13,33",
            league_name="England Premier League",
            capture_bookmakers=True
        )
        
        if not match:
            print("ERRO: Não conseguiu fazer scraping do jogo")
            return
        
        print(f"\nJogo: {match.home_team} vs {match.away_team}")
        print(f"Linhas AH capturadas: {len(match.ah_lines)}")
        
        # Estatísticas
        total_linhas = len(match.ah_lines)
        total_bookmaker_slots = 0  # Total de slots (linha x bookmaker x lado)
        slots_preenchidos = 0
        slots_vazios = 0
        
        linhas_completas = 0
        linhas_parciais = 0
        linhas_vazias = 0
        
        problemas = []
        
        print("\n" + "-"*70)
        print("ANÁLISE POR LINHA:")
        print("-"*70)
        
        for line_name, ah_line in sorted(match.ah_lines.items(), key=lambda x: float(x[0].replace('+', ''))):
            # Conta bookmakers (excluindo "best" que é sintético)
            bks = {k: v for k, v in ah_line.bookmaker_odds.items() if k != "best"}
            num_bks = len(bks)
            
            if num_bks == 0:
                linhas_vazias += 1
                problemas.append(f"AH {line_name}: Nenhum bookmaker capturado")
                print(f"  AH {line_name:>6}: ❌ 0 bookmakers")
                continue
            
            # Conta slots preenchidos vs vazios
            home_ok = 0
            home_zero = 0
            away_ok = 0
            away_zero = 0
            
            for bk_name, bk_odds in bks.items():
                if bk_odds.home_odds > 0:
                    home_ok += 1
                else:
                    home_zero += 1
                    
                if bk_odds.away_odds > 0:
                    away_ok += 1
                else:
                    away_zero += 1
            
            total_slots = num_bks * 2  # HOME + AWAY para cada bookmaker
            filled_slots = home_ok + away_ok
            empty_slots = home_zero + away_zero
            
            total_bookmaker_slots += total_slots
            slots_preenchidos += filled_slots
            slots_vazios += empty_slots
            
            # Classifica a linha
            pct_line = (filled_slots / total_slots) * 100 if total_slots > 0 else 0
            
            if empty_slots == 0:
                linhas_completas += 1
                status = "✅"
            elif filled_slots > 0:
                linhas_parciais += 1
                status = "⚠️"
                if home_zero > 0:
                    problemas.append(f"AH {line_name}: {home_zero} bookmaker(s) sem HOME odds")
                if away_zero > 0:
                    problemas.append(f"AH {line_name}: {away_zero} bookmaker(s) sem AWAY odds")
            else:
                linhas_vazias += 1
                status = "❌"
                problemas.append(f"AH {line_name}: Todos os slots vazios")
            
            print(f"  AH {line_name:>6}: {status} {num_bks} bks, {filled_slots}/{total_slots} slots ({pct_line:.0f}%)" + 
                  (f" - H:{home_zero}❌ A:{away_zero}❌" if empty_slots > 0 else ""))
        
        # Resumo
        print("\n" + "="*70)
        print("RESUMO DE ACURÁCIA")
        print("="*70)
        
        pct_linhas_ok = (linhas_completas / total_linhas) * 100 if total_linhas > 0 else 0
        pct_linhas_parciais = (linhas_parciais / total_linhas) * 100 if total_linhas > 0 else 0
        pct_slots = (slots_preenchidos / total_bookmaker_slots) * 100 if total_bookmaker_slots > 0 else 0
        
        print(f"\n📊 LINHAS AH:")
        print(f"   Total:          {total_linhas}")
        print(f"   ✅ Completas:    {linhas_completas} ({pct_linhas_ok:.1f}%)")
        print(f"   ⚠️ Parciais:     {linhas_parciais} ({pct_linhas_parciais:.1f}%)")
        print(f"   ❌ Vazias:       {linhas_vazias}")
        
        print(f"\n📊 SLOTS (bookmaker x lado):")
        print(f"   Total:          {total_bookmaker_slots}")
        print(f"   ✅ Preenchidos:  {slots_preenchidos} ({pct_slots:.1f}%)")
        print(f"   ❌ Vazios:       {slots_vazios} ({100-pct_slots:.1f}%)")
        
        print(f"\n🎯 ACURÁCIA GERAL: {pct_slots:.1f}%")
        
        if problemas:
            print(f"\n⚠️ PROBLEMAS DETECTADOS ({len(problemas)}):")
            for p in problemas[:10]:  # Mostra até 10
                print(f"   - {p}")
            if len(problemas) > 10:
                print(f"   ... e mais {len(problemas) - 10} problemas")
        
        print("\n" + "="*70)
        
    finally:
        await scraper.close()

if __name__ == "__main__":
    asyncio.run(check_accuracy())
