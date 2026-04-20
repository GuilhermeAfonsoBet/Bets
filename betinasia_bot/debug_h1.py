# -*- coding: utf-8 -*-
"""
Debug do detector H1 - Verifica se os eventos de pricing estão corretos.
"""

import asyncio
import sys
from sqlalchemy import text

sys.path.insert(0, '.')

from storage.database import Database


async def analyze_h1_events():
    """Analisa os eventos H1 detectados."""
    
    db = Database()
    await db.connect()
    
    print("=" * 70)
    print("DEBUG H1 - Análise de Eventos de Pricing")
    print("=" * 70)
    
    try:
        async with db.async_session() as session:
            
            # 1. Distribuição de overround
            print("\n1. DISTRIBUIÇÃO DE OVERROUND")
            print("-" * 50)
            
            result = await session.execute(text("""
                SELECT 
                    CASE 
                        WHEN overround < 0.95 THEN '< 95% (arb forte)'
                        WHEN overround < 1.00 THEN '95-100% (arb)'
                        WHEN overround < 1.02 THEN '100-102% (normal baixo)'
                        WHEN overround < 1.05 THEN '102-105% (normal)'
                        WHEN overround < 1.10 THEN '105-110% (alto)'
                        ELSE '> 110% (muito alto)'
                    END as faixa,
                    COUNT(*) as qtd,
                    ROUND(AVG(overround)::numeric, 4) as media_overround
                FROM h1_pricing_events
                GROUP BY 1
                ORDER BY MIN(overround)
            """))
            for row in result.fetchall():
                print(f"   {row[0]}: {row[1]:,} eventos (média: {row[2]})")
            
            # 2. Quantos são arbitragem vs mispricing
            print("\n2. TIPO DE EVENTO")
            print("-" * 50)
            
            result = await session.execute(text("""
                SELECT 
                    is_arb,
                    mispriced_side,
                    COUNT(*) as qtd
                FROM h1_pricing_events
                GROUP BY is_arb, mispriced_side
                ORDER BY qtd DESC
            """))
            for row in result.fetchall():
                tipo = "ARBITRAGEM" if row[0] else f"MISPRICING ({row[1]})"
                print(f"   {tipo}: {row[2]:,}")
            
            # 3. Distribuição por mercado
            print("\n3. DISTRIBUIÇÃO POR MERCADO")
            print("-" * 50)
            
            result = await session.execute(text("""
                SELECT market_type, COUNT(*) as qtd
                FROM h1_pricing_events
                GROUP BY market_type
                ORDER BY qtd DESC
            """))
            for row in result.fetchall():
                print(f"   {row[0]}: {row[1]:,}")
            
            # 4. Exemplos de "arbitragem"
            print("\n4. EXEMPLOS DE ARBITRAGEM (overround < 1)")
            print("-" * 50)
            
            result = await session.execute(text("""
                SELECT 
                    id, match_id, market_type, ah_line,
                    odd_side_a, odd_side_b, 
                    overround,
                    ROUND((1.0/odd_side_a + 1.0/odd_side_b)::numeric, 4) as calc_overround
                FROM h1_pricing_events
                WHERE is_arb = true
                ORDER BY detected_at DESC
                LIMIT 10
            """))
            print("   ID | Market | Line | OddA | OddB | Overround | Calc")
            for row in result.fetchall():
                print(f"   {row[0]} | {row[2]} | {row[3]} | {row[4]:.2f} | {row[5]:.2f} | {row[6]:.4f} | {row[7]}")
            
            # 5. Verificar se há odds inválidas
            print("\n5. ODDS INVÁLIDAS OU ESTRANHAS")
            print("-" * 50)
            
            result = await session.execute(text("""
                SELECT 
                    COUNT(CASE WHEN odd_side_a <= 1 THEN 1 END) as odd_a_lte_1,
                    COUNT(CASE WHEN odd_side_b <= 1 THEN 1 END) as odd_b_lte_1,
                    COUNT(CASE WHEN odd_side_a > 100 THEN 1 END) as odd_a_gt_100,
                    COUNT(CASE WHEN odd_side_b > 100 THEN 1 END) as odd_b_gt_100,
                    MIN(odd_side_a) as min_odd_a,
                    MIN(odd_side_b) as min_odd_b,
                    MAX(odd_side_a) as max_odd_a,
                    MAX(odd_side_b) as max_odd_b
                FROM h1_pricing_events
            """))
            row = result.fetchone()
            print(f"   Odd A <= 1: {row[0]:,}")
            print(f"   Odd B <= 1: {row[1]:,}")
            print(f"   Odd A > 100: {row[2]:,}")
            print(f"   Odd B > 100: {row[3]:,}")
            print(f"   Range Odd A: {row[4]:.2f} - {row[6]:.2f}")
            print(f"   Range Odd B: {row[5]:.2f} - {row[7]:.2f}")
            
            # 6. Verificar cálculo de exemplo
            print("\n6. VERIFICAÇÃO DE CÁLCULO (último evento)")
            print("-" * 50)
            
            result = await session.execute(text("""
                SELECT 
                    odd_side_a, odd_side_b, 
                    overround,
                    fair_odd_a, fair_odd_b,
                    deviation_a, deviation_b,
                    edge_estimate
                FROM h1_pricing_events
                ORDER BY id DESC
                LIMIT 1
            """))
            row = result.fetchone()
            if row:
                odd_a, odd_b = row[0], row[1]
                print(f"   Odds: A={odd_a:.4f}, B={odd_b:.4f}")
                print(f"   Overround salvo: {row[2]:.4f}")
                
                # Recalcula
                prob_a = 1.0 / odd_a if odd_a > 0 else 0
                prob_b = 1.0 / odd_b if odd_b > 0 else 0
                calc_overround = prob_a + prob_b
                print(f"   Overround calculado: {calc_overround:.4f}")
                
                if calc_overround > 0:
                    fair_prob_a = prob_a / calc_overround
                    fair_prob_b = prob_b / calc_overround
                    fair_odd_a = 1.0 / fair_prob_a if fair_prob_a > 0 else 0
                    fair_odd_b = 1.0 / fair_prob_b if fair_prob_b > 0 else 0
                    print(f"   Fair odds calculadas: A={fair_odd_a:.4f}, B={fair_odd_b:.4f}")
                    print(f"   Fair odds salvas: A={row[3]:.4f}, B={row[4]:.4f}")
                    
                    dev_a = (odd_a - fair_odd_a) / fair_odd_a if fair_odd_a > 0 else 0
                    dev_b = (odd_b - fair_odd_b) / fair_odd_b if fair_odd_b > 0 else 0
                    print(f"   Desvio calculado: A={dev_a:.4f}, B={dev_b:.4f}")
                    print(f"   Desvio salvo: A={row[5]:.4f}, B={row[6]:.4f}")
            
            # 7. Mercados 1X2 (sem draw)
            print("\n7. ANÁLISE DE 1X2 (possível problema)")
            print("-" * 50)
            
            result = await session.execute(text("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN overround < 0.70 THEN 1 END) as overround_lt_70
                FROM h1_pricing_events
                WHERE market_type = '1X2'
            """))
            row = result.fetchone()
            print(f"   Total 1X2: {row[0]:,}")
            print(f"   Com overround < 70%: {row[1]:,}")
            print("\n   NOTA: 1X2 tem 3 outcomes (H/D/A), não 2!")
            print("   Calcular overround com só H+A está ERRADO para 1X2!")
            
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(analyze_h1_events())
