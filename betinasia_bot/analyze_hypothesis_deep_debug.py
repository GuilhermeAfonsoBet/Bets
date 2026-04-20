# -*- coding: utf-8 -*-
"""
Análise profunda para investigar:
1. Como CLV baseline é calculado
2. Por que H3B baseline está estranho
3. Desvios padrões de cada medida
4. Se estamos escolhendo o lado ERRADO para apostar
"""

import asyncio
import sys
from sqlalchemy import text

sys.path.insert(0, '.')

from storage.database import Database


async def deep_debug():
    db = Database()
    await db.connect()
    
    print("=" * 70)
    print("ANÁLISE PROFUNDA - DEBUG DAS HIPÓTESES")
    print("=" * 70)
    
    try:
        async with db.async_session() as session:
            
            # ============================================================
            # 1. EXPLICAÇÃO DO CLV BASELINE
            # ============================================================
            print("\n" + "=" * 70)
            print("1. COMO O CLV BASELINE É CALCULADO")
            print("=" * 70)
            print("""
CLV baseline = (média_odds_coletadas - closing_odds) / closing_odds × 100

Exemplo:
- Coletamos AH -0.5 Home 10 vezes: [2.00, 2.02, 2.05, 2.03, 2.01, ...]
- Média das odds = 2.02
- Closing (última antes kickoff) = 2.05
- CLV baseline = (2.02 - 2.05) / 2.05 = -1.46%

Se baseline > 0: odds médias são MELHORES que closing (mercado ineficiente)
Se baseline < 0: odds médias são PIORES que closing (mercado fecha no melhor preço)
""")
            
            # ============================================================
            # 2. INVESTIGAR H3B BASELINE ALTO
            # ============================================================
            print("\n" + "=" * 70)
            print("2. INVESTIGAÇÃO: POR QUE H3B TEM BASELINE DE 14.88%?")
            print("=" * 70)
            
            # Verificar distribuição do baseline por mercado
            result = await session.execute(text("""
                WITH baseline AS (
                    SELECT 
                        b.match_id,
                        b.ah_line,
                        m.kickoff_time,
                        (SELECT best_home_odds FROM best_odds_history b2 
                         WHERE b2.match_id = b.match_id AND b2.ah_line = b.ah_line
                         AND b2.scraped_at < m.kickoff_time
                         ORDER BY scraped_at DESC LIMIT 1) as closing,
                        AVG(b.best_home_odds) as avg_odds
                    FROM best_odds_history b
                    JOIN matches m ON b.match_id = m.id
                    WHERE m.status = 'finished'
                    GROUP BY b.match_id, b.ah_line, m.kickoff_time
                    HAVING COUNT(*) >= 3
                ),
                baseline_clv AS (
                    SELECT 
                        ah_line,
                        CASE WHEN closing > 0 
                             THEN (avg_odds - closing) / closing * 100 
                             ELSE NULL END as clv_baseline
                    FROM baseline WHERE closing > 0
                )
                SELECT 
                    ah_line,
                    COUNT(*) as n,
                    ROUND(AVG(clv_baseline)::numeric, 2) as clv_medio,
                    ROUND(MIN(clv_baseline)::numeric, 2) as clv_min,
                    ROUND(MAX(clv_baseline)::numeric, 2) as clv_max
                FROM baseline_clv
                GROUP BY ah_line
                ORDER BY ABS(AVG(clv_baseline)) DESC
                LIMIT 15
            """))
            print("\n   Baseline por tipo de mercado (maiores desvios):")
            print("   Mercado      | N  | CLV Médio | Min    | Max")
            for row in result.fetchall():
                print(f"   {row[0]:12} | {row[1]:2} | {row[2]:9}% | {row[3]:6}% | {row[4]:6}%")
            
            # H3B usa muitos 1X2?
            result = await session.execute(text("""
                SELECT 
                    market_type,
                    ah_line,
                    COUNT(*) as n,
                    ROUND(AVG(clv_pct)::numeric, 2) as clv_evento,
                    ROUND(MIN(clv_pct)::numeric, 2) as min_clv,
                    ROUND(MAX(clv_pct)::numeric, 2) as max_clv
                FROM h3b_temporal_reversal_events
                WHERE clv IS NOT NULL
                GROUP BY market_type, ah_line
                ORDER BY COUNT(*) DESC
                LIMIT 10
            """))
            print("\n   H3B - Distribuição por mercado:")
            print("   Market | Line      | N   | CLV Médio | Min    | Max")
            for row in result.fetchall():
                print(f"   {row[0]:6} | {row[1]:9} | {row[2]:3} | {row[3]:9}% | {row[4]:6}% | {row[5]:6}%")
            
            # ============================================================
            # 3. ANÁLISE COM DESVIOS PADRÕES
            # ============================================================
            print("\n" + "=" * 70)
            print("3. ANÁLISE COMPLETA COM DESVIOS PADRÕES")
            print("=" * 70)
            
            for hip, tabela, linha_col in [
                ("H1", "h1_pricing_events", "ah_line"),
                ("H3", "h3_line_monotonicity_events", "recommended_line"),
                ("H3B", "h3b_temporal_reversal_events", "ah_line"),
                ("H6", "h6_correlation_lag_events", "lagged_line"),
            ]:
                result = await session.execute(text(f"""
                    SELECT 
                        COUNT(*) as n,
                        ROUND(AVG(clv_pct)::numeric, 3) as clv_medio,
                        ROUND(STDDEV(clv_pct)::numeric, 3) as clv_std,
                        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY clv_pct)::numeric, 3) as clv_mediana,
                        ROUND(AVG(profit_loss)::numeric, 4) as profit_medio,
                        ROUND(STDDEV(profit_loss)::numeric, 4) as profit_std
                    FROM {tabela}
                    WHERE clv IS NOT NULL AND clv_pct BETWEEN -50 AND 50
                """))
                row = result.fetchone()
                if row and row[0] > 0:
                    print(f"""
   {hip}:
   N = {row[0]}
   CLV:    média = {row[1]}% ± {row[2]}%  |  mediana = {row[3]}%
   Profit: média = {row[4]} ± {row[5]}
""")
            
            # ============================================================
            # 4. TESTE: ESTAMOS APOSTANDO NO LADO ERRADO?
            # ============================================================
            print("\n" + "=" * 70)
            print("4. TESTE: E SE INVERTÊSSEMOS O LADO?")
            print("=" * 70)
            print("""
Se CLV adicional é NEGATIVO, talvez o detector encontre os momentos certos
mas esteja recomendando o lado ERRADO.

Vamos comparar:
- CLV do lado recomendado
- CLV do lado OPOSTO (se tivéssemos apostado no outro lado)
""")
            
            # Para H1: verificar se o lado oposto seria melhor
            result = await session.execute(text("""
                SELECT 
                    recommended_side,
                    COUNT(*) as n,
                    ROUND(AVG(clv_pct)::numeric, 3) as clv_medio,
                    ROUND(AVG(profit_loss)::numeric, 4) as profit_medio,
                    -- Desvio do lado oposto (deviation_a quando recomenda side_b, e vice-versa)
                    ROUND(AVG(CASE 
                        WHEN recommended_side = 'side_a' THEN deviation_b 
                        ELSE deviation_a 
                    END * 100)::numeric, 3) as desvio_oposto_medio
                FROM h1_pricing_events
                WHERE clv IS NOT NULL AND clv_pct BETWEEN -50 AND 50
                GROUP BY recommended_side
            """))
            print("\n   H1 - Por lado recomendado:")
            print("   Lado Rec.  | N   | CLV Médio | Profit | Desvio Oposto")
            for row in result.fetchall():
                print(f"   {row[0]:9} | {row[1]:3} | {row[2]:9}% | {row[3]:6} | {row[4]:6}%")
            
            # Para H3B: direção da reversão
            result = await session.execute(text("""
                SELECT 
                    direction_after as direcao_apostada,
                    COUNT(*) as n,
                    ROUND(AVG(clv_pct)::numeric, 3) as clv_medio,
                    ROUND(AVG(profit_loss)::numeric, 4) as profit_medio,
                    -- Se inverter: 
                    ROUND(AVG(-profit_loss)::numeric, 4) as profit_se_inverter
                FROM h3b_temporal_reversal_events
                WHERE clv IS NOT NULL AND clv_pct BETWEEN -50 AND 50
                GROUP BY direction_after
            """))
            print("\n   H3B - Por direção apostada:")
            print("   Direção    | N   | CLV    | Profit | Profit se inverter")
            for row in result.fetchall():
                print(f"   {row[0]:9} | {row[1]:3} | {row[2]:6}% | {row[3]:6} | {row[4]:6}")
            
            # ============================================================
            # 5. ANÁLISE DE INVERSÃO COMPLETA
            # ============================================================
            print("\n" + "=" * 70)
            print("5. SE INVERTÊSSEMOS TODAS AS RECOMENDAÇÕES")
            print("=" * 70)
            
            # Para apostas, inverter = trocar win por loss e vice-versa
            for hip, tabela in [
                ("H1", "h1_pricing_events"),
                ("H3", "h3_line_monotonicity_events"),
                ("H3B", "h3b_temporal_reversal_events"),
                ("H6", "h6_correlation_lag_events"),
            ]:
                result = await session.execute(text(f"""
                    SELECT 
                        COUNT(*) as n,
                        ROUND(SUM(profit_loss)::numeric, 2) as profit_original,
                        ROUND(SUM(-profit_loss)::numeric, 2) as profit_invertido,
                        COUNT(CASE WHEN bet_result = 'win' THEN 1 END) as wins_original,
                        COUNT(CASE WHEN bet_result = 'loss' THEN 1 END) as losses_original
                    FROM {tabela}
                    WHERE clv IS NOT NULL 
                      AND clv_pct BETWEEN -50 AND 50
                      AND bet_result IN ('win', 'loss')
                """))
                row = result.fetchone()
                if row and row[0] > 0:
                    melhor = "INVERTER" if (row[2] or 0) > (row[1] or 0) else "MANTER"
                    print(f"""
   {hip}:
   Apostas: {row[0]} (W:{row[3]} L:{row[4]})
   Profit original:  {row[1]} unidades
   Profit invertido: {row[2]} unidades
   → Melhor: {melhor}
""")
            
            print("""
======================================================================
CONCLUSÃO
======================================================================

Se "inverter" é melhor para alguma hipótese, significa que:
1. O detector ESTÁ identificando momentos de ineficiência
2. Mas a lógica de QUAL LADO apostar está ERRADA
3. Solução: revisar a lógica de recomendação

Se mesmo invertendo não melhora:
1. O detector provavelmente não está capturando valor real
2. Ou a amostra é muito pequena para conclusões
""")
            
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(deep_debug())
