# -*- coding: utf-8 -*-
"""
Análise estatística dos resultados das hipóteses.
Verifica se as hipóteses geram valor (CLV positivo, ROI positivo).
"""

import asyncio
import sys
from sqlalchemy import text

sys.path.insert(0, '.')

from storage.database import Database


async def analyze_results():
    db = Database()
    await db.connect()
    
    print("=" * 70)
    print("ANÁLISE DE VALIDAÇÃO DAS HIPÓTESES")
    print("=" * 70)
    
    try:
        async with db.async_session() as session:
            
            # ============================================================
            # H1 - PRECIFICAÇÃO
            # ============================================================
            print("\n" + "=" * 70)
            print("H1 - PRECIFICAÇÃO (Mispricing/Arbitragem)")
            print("=" * 70)
            
            result = await session.execute(text("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN clv IS NOT NULL THEN 1 END) as com_clv,
                    ROUND(AVG(clv_pct)::numeric, 3) as clv_medio,
                    ROUND(AVG(CASE WHEN clv_pct > 0 THEN clv_pct END)::numeric, 3) as clv_positivo_medio,
                    COUNT(CASE WHEN clv_pct > 0 THEN 1 END) as clv_positivos,
                    COUNT(CASE WHEN bet_result = 'win' THEN 1 END) as wins,
                    COUNT(CASE WHEN bet_result = 'loss' THEN 1 END) as losses,
                    COUNT(CASE WHEN bet_result IN ('half_win', 'half_loss', 'push') THEN 1 END) as outros,
                    ROUND(SUM(profit_loss)::numeric, 2) as profit_total,
                    ROUND(AVG(profit_loss)::numeric, 4) as profit_medio
                FROM h1_pricing_events
                WHERE clv IS NOT NULL
            """))
            row = result.fetchone()
            if row and row[1] > 0:
                win_rate = row[5] / (row[5] + row[6]) * 100 if (row[5] + row[6]) > 0 else 0
                clv_hit_rate = row[4] / row[1] * 100 if row[1] > 0 else 0
                print(f"""
   Eventos com CLV: {row[1]}
   
   CLV:
   - CLV médio: {row[2]}% {'✅ POSITIVO' if row[2] and row[2] > 0 else '❌ NEGATIVO'}
   - CLV positivo médio: {row[3]}%
   - Taxa de CLV positivo: {clv_hit_rate:.1f}% ({row[4]}/{row[1]})
   
   Resultados de apostas simuladas:
   - Wins: {row[5]} | Losses: {row[6]} | Outros: {row[7]}
   - Win rate: {win_rate:.1f}%
   - Profit total: {row[8]} unidades
   - Profit médio por aposta: {row[9]} unidades
   - ROI estimado: {(row[9] or 0) * 100:.2f}%
""")
            else:
                print("\n   Nenhum evento com CLV calculado.")
            
            # Por tipo (arb vs mispricing)
            result = await session.execute(text("""
                SELECT 
                    CASE WHEN is_arb THEN 'Arbitragem' ELSE 'Mispricing' END as tipo,
                    COUNT(*) as eventos,
                    ROUND(AVG(clv_pct)::numeric, 3) as clv_medio,
                    ROUND(SUM(profit_loss)::numeric, 2) as profit,
                    COUNT(CASE WHEN bet_result = 'win' THEN 1 END) as wins,
                    COUNT(CASE WHEN bet_result = 'loss' THEN 1 END) as losses
                FROM h1_pricing_events
                WHERE clv IS NOT NULL
                GROUP BY is_arb
            """))
            rows = result.fetchall()
            if rows:
                print("   Por tipo:")
                print("   Tipo       | Eventos | CLV Médio | Profit | W-L")
                for row in rows:
                    print(f"   {row[0]:10} | {row[1]:7} | {row[2]:9}% | {row[3]:6} | {row[4]}-{row[5]}")
            
            # ============================================================
            # H3 - MONOTONICIDADE ENTRE LINHAS
            # ============================================================
            print("\n" + "=" * 70)
            print("H3 - MONOTONICIDADE ENTRE LINHAS ADJACENTES")
            print("=" * 70)
            
            result = await session.execute(text("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN clv IS NOT NULL THEN 1 END) as com_clv,
                    ROUND(AVG(clv_pct)::numeric, 3) as clv_medio,
                    COUNT(CASE WHEN clv_pct > 0 THEN 1 END) as clv_positivos,
                    COUNT(CASE WHEN bet_result = 'win' THEN 1 END) as wins,
                    COUNT(CASE WHEN bet_result = 'loss' THEN 1 END) as losses,
                    ROUND(SUM(profit_loss)::numeric, 2) as profit_total,
                    ROUND(AVG(profit_loss)::numeric, 4) as profit_medio
                FROM h3_line_monotonicity_events
                WHERE clv IS NOT NULL
            """))
            row = result.fetchone()
            if row and row[1] > 0:
                win_rate = row[4] / (row[4] + row[5]) * 100 if (row[4] + row[5]) > 0 else 0
                clv_hit_rate = row[3] / row[1] * 100 if row[1] > 0 else 0
                print(f"""
   Eventos com CLV: {row[1]}
   
   CLV:
   - CLV médio: {row[2]}% {'✅ POSITIVO' if row[2] and row[2] > 0 else '❌ NEGATIVO'}
   - Taxa de CLV positivo: {clv_hit_rate:.1f}% ({row[3]}/{row[1]})
   
   Resultados:
   - Wins: {row[4]} | Losses: {row[5]}
   - Win rate: {win_rate:.1f}%
   - Profit total: {row[6]} unidades
   - ROI estimado: {(row[7] or 0) * 100:.2f}%
""")
            else:
                print("\n   Nenhum evento com CLV calculado.")
            
            # ============================================================
            # H3B - REVERSÕES TEMPORAIS
            # ============================================================
            print("\n" + "=" * 70)
            print("H3B - REVERSÕES TEMPORAIS DE ODDS")
            print("=" * 70)
            
            result = await session.execute(text("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN clv IS NOT NULL THEN 1 END) as com_clv,
                    ROUND(AVG(clv_pct)::numeric, 3) as clv_medio,
                    COUNT(CASE WHEN clv_pct > 0 THEN 1 END) as clv_positivos,
                    COUNT(CASE WHEN bet_result = 'win' THEN 1 END) as wins,
                    COUNT(CASE WHEN bet_result = 'loss' THEN 1 END) as losses,
                    ROUND(SUM(profit_loss)::numeric, 2) as profit_total,
                    ROUND(AVG(profit_loss)::numeric, 4) as profit_medio
                FROM h3b_temporal_reversal_events
                WHERE clv IS NOT NULL
            """))
            row = result.fetchone()
            if row and row[1] > 0:
                win_rate = row[4] / (row[4] + row[5]) * 100 if (row[4] + row[5]) > 0 else 0
                clv_hit_rate = row[3] / row[1] * 100 if row[1] > 0 else 0
                print(f"""
   Eventos com CLV: {row[1]}
   
   CLV:
   - CLV médio: {row[2]}% {'✅ POSITIVO' if row[2] and row[2] > 0 else '❌ NEGATIVO'}
   - Taxa de CLV positivo: {clv_hit_rate:.1f}% ({row[3]}/{row[1]})
   
   Resultados:
   - Wins: {row[4]} | Losses: {row[5]}
   - Win rate: {win_rate:.1f}%
   - Profit total: {row[6]} unidades
   - ROI estimado: {(row[7] or 0) * 100:.2f}%
""")
            else:
                print("\n   Nenhum evento com CLV calculado.")
            
            # Por direção da reversão
            result = await session.execute(text("""
                SELECT 
                    direction_before || ' -> ' || direction_after as direcao,
                    COUNT(*) as eventos,
                    ROUND(AVG(clv_pct)::numeric, 3) as clv_medio,
                    ROUND(SUM(profit_loss)::numeric, 2) as profit
                FROM h3b_temporal_reversal_events
                WHERE clv IS NOT NULL
                GROUP BY direction_before, direction_after
            """))
            rows = result.fetchall()
            if rows:
                print("   Por direção da reversão:")
                print("   Direção     | Eventos | CLV Médio | Profit")
                for row in rows:
                    print(f"   {row[0]:12} | {row[1]:7} | {row[2]:9}% | {row[3]}")
            
            # ============================================================
            # H6 - CORRELAÇÃO/LAG
            # ============================================================
            print("\n" + "=" * 70)
            print("H6 - CORRELAÇÃO/LAG ENTRE MERCADOS")
            print("=" * 70)
            
            result = await session.execute(text("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN clv IS NOT NULL THEN 1 END) as com_clv,
                    ROUND(AVG(clv_pct)::numeric, 3) as clv_medio,
                    COUNT(CASE WHEN clv_pct > 0 THEN 1 END) as clv_positivos,
                    COUNT(CASE WHEN bet_result = 'win' THEN 1 END) as wins,
                    COUNT(CASE WHEN bet_result = 'loss' THEN 1 END) as losses,
                    ROUND(SUM(profit_loss)::numeric, 2) as profit_total,
                    ROUND(AVG(profit_loss)::numeric, 4) as profit_medio
                FROM h6_correlation_lag_events
                WHERE clv IS NOT NULL
            """))
            row = result.fetchone()
            if row and row[1] > 0:
                win_rate = row[4] / (row[4] + row[5]) * 100 if (row[4] + row[5]) > 0 else 0
                clv_hit_rate = row[3] / row[1] * 100 if row[1] > 0 else 0
                print(f"""
   Eventos com CLV: {row[1]}
   
   CLV:
   - CLV médio: {row[2]}% {'✅ POSITIVO' if row[2] and row[2] > 0 else '❌ NEGATIVO'}
   - Taxa de CLV positivo: {clv_hit_rate:.1f}% ({row[3]}/{row[1]})
   
   Resultados:
   - Wins: {row[4]} | Losses: {row[5]}
   - Win rate: {win_rate:.1f}%
   - Profit total: {row[6]} unidades
   - ROI estimado: {(row[7] or 0) * 100:.2f}%
""")
            else:
                print("\n   Nenhum evento com CLV calculado.")
            
            # ============================================================
            # RESUMO COMPARATIVO
            # ============================================================
            print("\n" + "=" * 70)
            print("RESUMO COMPARATIVO")
            print("=" * 70)
            
            result = await session.execute(text("""
                SELECT 'H1' as hip, COUNT(*) as n, ROUND(AVG(clv_pct)::numeric, 3) as clv, 
                       ROUND(SUM(profit_loss)::numeric, 2) as profit
                FROM h1_pricing_events WHERE clv IS NOT NULL
                UNION ALL
                SELECT 'H3', COUNT(*), ROUND(AVG(clv_pct)::numeric, 3), ROUND(SUM(profit_loss)::numeric, 2)
                FROM h3_line_monotonicity_events WHERE clv IS NOT NULL
                UNION ALL
                SELECT 'H3b', COUNT(*), ROUND(AVG(clv_pct)::numeric, 3), ROUND(SUM(profit_loss)::numeric, 2)
                FROM h3b_temporal_reversal_events WHERE clv IS NOT NULL
                UNION ALL
                SELECT 'H6', COUNT(*), ROUND(AVG(clv_pct)::numeric, 3), ROUND(SUM(profit_loss)::numeric, 2)
                FROM h6_correlation_lag_events WHERE clv IS NOT NULL
                ORDER BY 1
            """))
            print("\n   Hipótese | Eventos | CLV Médio | Profit Total | Status")
            print("   " + "-" * 55)
            for row in result.fetchall():
                status = "✅ Gera valor" if row[2] and row[2] > 0 else "❌ Não gera" if row[2] else "⏳ Sem dados"
                print(f"   {row[0]:8} | {row[1]:7} | {row[2] or 'N/A':9} | {row[3] or 'N/A':12} | {status}")
            
            print("""
======================================================================
INTERPRETAÇÃO
======================================================================

CLV (Closing Line Value):
- CLV > 0: Conseguimos odds melhores que o mercado de fechamento
- CLV consistentemente positivo indica edge real

Para conclusões estatísticas sólidas:
- Mínimo ~100 eventos por hipótese
- p-value < 0.05 para significância

NOTA: Esta é uma análise preliminar com dados limitados.
Aguarde mais jogos para conclusões definitivas.
""")
            
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(analyze_results())
