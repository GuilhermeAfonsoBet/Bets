# -*- coding: utf-8 -*-
"""
Análise de Hipóteses V3 - Com correções:
1. H6: Filtrar apenas eventos onde líder moveu DOWN
2. H3B: Separar análise por tipo de reversão (UP vs DOWN)
3. H1: Lógica confirmada correta
"""

import asyncio
import sys
from sqlalchemy import text

sys.path.insert(0, '.')

from storage.database import Database


async def analyze_v3():
    db = Database()
    await db.connect()
    
    print("=" * 70)
    print("ANÁLISE DE HIPÓTESES V3 - COM CORREÇÕES")
    print("=" * 70)
    
    try:
        async with db.async_session() as session:
            
            # ============================================================
            # H1 - PRECIFICAÇÃO (lógica confirmada correta)
            # ============================================================
            print("\n" + "=" * 70)
            print("H1 - PRECIFICAÇÃO")
            print("=" * 70)
            print("""
Lógica: Detecta odds acima do fair value (deviation > 0)
Se mercado corrige para baixo → CLV positivo ✅
""")
            
            result = await session.execute(text("""
                SELECT 
                    COUNT(*) as n,
                    ROUND(AVG(clv_pct)::numeric, 3) as clv_medio,
                    ROUND(STDDEV(clv_pct)::numeric, 3) as clv_std,
                    ROUND(AVG(profit_loss)::numeric, 4) as profit_medio,
                    COUNT(CASE WHEN bet_result = 'win' THEN 1 END) as wins,
                    COUNT(CASE WHEN bet_result = 'loss' THEN 1 END) as losses,
                    ROUND(100.0 * COUNT(CASE WHEN clv_pct > 0 THEN 1 END) / NULLIF(COUNT(*), 0)::numeric, 1) as taxa_clv_pos
                FROM h1_pricing_events
                WHERE clv_pct IS NOT NULL AND clv_pct BETWEEN -50 AND 50
            """))
            row = result.fetchone()
            if row and row[0] > 0:
                print(f"""
   N = {row[0]}
   CLV médio:  {row[1]}% ± {row[2]}%
   Profit:     {row[3]}
   Win/Loss:   {row[4]}/{row[5]}
   Taxa CLV>0: {row[6]}%
""")
            
            # ============================================================
            # H3 - MONOTONICIDADE (sem alterações)
            # ============================================================
            print("\n" + "=" * 70)
            print("H3 - MONOTONICIDADE ENTRE LINHAS")
            print("=" * 70)
            
            result = await session.execute(text("""
                SELECT 
                    COUNT(*) as n,
                    ROUND(AVG(clv_pct)::numeric, 3) as clv_medio,
                    ROUND(STDDEV(clv_pct)::numeric, 3) as clv_std,
                    ROUND(AVG(profit_loss)::numeric, 4) as profit_medio,
                    COUNT(CASE WHEN bet_result = 'win' THEN 1 END) as wins,
                    COUNT(CASE WHEN bet_result = 'loss' THEN 1 END) as losses,
                    ROUND(100.0 * COUNT(CASE WHEN clv_pct > 0 THEN 1 END) / NULLIF(COUNT(*), 0)::numeric, 1) as taxa_clv_pos
                FROM h3_line_monotonicity_events
                WHERE clv_pct IS NOT NULL AND clv_pct BETWEEN -50 AND 50
            """))
            row = result.fetchone()
            if row and row[0] > 0:
                print(f"""
   N = {row[0]}
   CLV médio:  {row[1]}% ± {row[2]}%
   Profit:     {row[3]}
   Win/Loss:   {row[4]}/{row[5]}
   Taxa CLV>0: {row[6]}%
""")
            
            # ============================================================
            # H3B - REVERSÕES TEMPORAIS (SEPARADO POR DIREÇÃO)
            # ============================================================
            print("\n" + "=" * 70)
            print("H3B - REVERSÕES TEMPORAIS (SEPARADO POR DIREÇÃO)")
            print("=" * 70)
            print("""
NOVA LÓGICA:
- Reversão UP (odd subiu): odd MELHOROU → apostar NESSE lado
- Reversão DOWN (odd desceu): odd PIOROU → apostar no lado OPOSTO

Análise dos dados EXISTENTES (lógica antiga - apostava no mesmo lado sempre):
""")
            
            # Análise separada por direção - dados existentes
            result = await session.execute(text("""
                SELECT 
                    direction_after,
                    COUNT(*) as n,
                    ROUND(AVG(clv_pct)::numeric, 3) as clv_medio,
                    ROUND(STDDEV(clv_pct)::numeric, 3) as clv_std,
                    ROUND(AVG(profit_loss)::numeric, 4) as profit_medio,
                    ROUND(AVG(-profit_loss)::numeric, 4) as profit_invertido,
                    COUNT(CASE WHEN bet_result = 'win' THEN 1 END) as wins,
                    COUNT(CASE WHEN bet_result = 'loss' THEN 1 END) as losses,
                    ROUND(100.0 * COUNT(CASE WHEN clv_pct > 0 THEN 1 END) / NULLIF(COUNT(*), 0)::numeric, 1) as taxa_clv_pos
                FROM h3b_temporal_reversal_events
                WHERE clv_pct IS NOT NULL AND clv_pct BETWEEN -50 AND 50
                GROUP BY direction_after
                ORDER BY direction_after
            """))
            print("   Direção   | N   | CLV     | Profit Orig | Profit Inv | W/L     | CLV>0")
            print("   " + "-" * 70)
            for row in result.fetchall():
                melhor = "← INVERTER" if (row[5] or 0) > (row[4] or 0) else ""
                print(f"   {row[0]:8} | {row[1]:3} | {row[2]:7}% | {row[4]:11} | {row[5]:10} | {row[6]}/{row[7]:3} | {row[8]:5}% {melhor}")
            
            print("""
INTERPRETAÇÃO:
- Se "Reversão DOWN" tem profit melhor INVERTIDO → nossa correção está certa
- Se "Reversão UP" tem profit melhor ORIGINAL → a lógica para UP estava certa
""")
            
            # ============================================================
            # H6 - CORRELAÇÃO/LAG (SEPARADO POR DIREÇÃO)
            # ============================================================
            print("\n" + "=" * 70)
            print("H6 - CORRELAÇÃO/LAG (SEPARADO POR DIREÇÃO DO LÍDER)")
            print("=" * 70)
            print("""
NOVA LÓGICA:
- Líder move DOWN: atrasada está com odd MAIOR → apostar agora = CLV positivo ✅
- Líder move UP: atrasada está com odd MENOR → apostar agora = CLV negativo ❌

Análise dos dados EXISTENTES (ambas direções):
""")
            
            # Análise separada por direção do líder
            result = await session.execute(text("""
                SELECT 
                    leader_move_direction,
                    COUNT(*) as n,
                    ROUND(AVG(clv_pct)::numeric, 3) as clv_medio,
                    ROUND(STDDEV(clv_pct)::numeric, 3) as clv_std,
                    ROUND(AVG(profit_loss)::numeric, 4) as profit_medio,
                    COUNT(CASE WHEN bet_result = 'win' THEN 1 END) as wins,
                    COUNT(CASE WHEN bet_result = 'loss' THEN 1 END) as losses,
                    ROUND(100.0 * COUNT(CASE WHEN clv_pct > 0 THEN 1 END) / NULLIF(COUNT(*), 0)::numeric, 1) as taxa_clv_pos
                FROM h6_correlation_lag_events
                WHERE clv_pct IS NOT NULL AND clv_pct BETWEEN -50 AND 50
                GROUP BY leader_move_direction
                ORDER BY leader_move_direction
            """))
            print("   Dir Líder | N   | CLV     | Profit  | W/L     | CLV>0")
            print("   " + "-" * 55)
            for row in result.fetchall():
                marker = "← CORRETO (usar apenas)" if row[0] == "down" else "← IGNORAR (CLV negativo esperado)"
                print(f"   {row[0]:8} | {row[1]:3} | {row[2]:7}% | {row[4]:7} | {row[5]}/{row[6]:3} | {row[7]:5}% {marker}")
            
            print("""
INTERPRETAÇÃO:
- Se "down" tem CLV positivo e "up" tem CLV negativo → nossa correção está certa
- O detector corrigido só vai gerar eventos quando líder move DOWN
""")
            
            # ============================================================
            # RESUMO FINAL
            # ============================================================
            print("\n" + "=" * 70)
            print("RESUMO DAS CORREÇÕES")
            print("=" * 70)
            print("""
1. H1 - PRECIFICAÇÃO:
   ✅ Lógica CORRETA - detecta odds acima do fair, CLV positivo se mercado corrige

2. H3 - MONOTONICIDADE:
   ✅ Sem alterações necessárias

3. H3B - REVERSÕES TEMPORAIS:
   🔧 CORRIGIDO - agora separa:
   - Reversão UP: aposta no lado que subiu (odd melhorou)
   - Reversão DOWN: aposta no lado OPOSTO (odd piorou, oposto melhorou)

4. H6 - CORRELAÇÃO/LAG:
   🔧 CORRIGIDO - agora só gera eventos quando líder move DOWN
   - Porque atrasada ainda está com odd MAIOR
   - Apostar agora = pegar odd maior antes de descer = CLV positivo
""")
            
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(analyze_v3())
