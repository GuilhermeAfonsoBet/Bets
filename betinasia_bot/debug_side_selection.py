# -*- coding: utf-8 -*-
"""
Debug: Análise detalhada do lado escolhido em cada hipótese
e investigação do baseline alto no H6
"""

import asyncio
import sys
from sqlalchemy import text

sys.path.insert(0, '.')

from storage.database import Database


async def debug_sides():
    db = Database()
    await db.connect()
    
    print("=" * 70)
    print("DEBUG: LADO ESCOLHIDO EM CADA HIPÓTESE")
    print("=" * 70)
    
    try:
        async with db.async_session() as session:
            
            # ============================================================
            # H1 - PRECIFICAÇÃO
            # ============================================================
            print("\n" + "=" * 70)
            print("H1 - PRECIFICAÇÃO: LÓGICA DE ESCOLHA DE LADO")
            print("=" * 70)
            print("""
LÓGICA ATUAL:
- Calcula deviation_a = (fair_a - odd_a) / fair_a
- Calcula deviation_b = (fair_b - odd_b) / fair_b  
- Se deviation > 0: odd está ABAIXO do fair (valor para apostar)
- Recomenda o lado com MAIOR deviation positivo

PROBLEMA POTENCIAL:
- deviation > 0 significa odd < fair_odd
- Isso significa que a odd está "barata" - deveria ter valor
- MAS estamos calculando CLV com a odd do momento vs closing
- Se odd está abaixo do fair, pode ser que o mercado CORRIJA para cima
- Nesse caso, o CLV seria NEGATIVO (odd subiu = piorou)
""")
            
            result = await session.execute(text("""
                SELECT 
                    recommended_side,
                    COUNT(*) as n,
                    ROUND(AVG(deviation_a * 100)::numeric, 2) as dev_a_medio,
                    ROUND(AVG(deviation_b * 100)::numeric, 2) as dev_b_medio,
                    ROUND(AVG(clv_pct)::numeric, 2) as clv_medio,
                    ROUND(AVG(profit_loss)::numeric, 3) as profit_medio,
                    COUNT(CASE WHEN bet_result = 'win' THEN 1 END) as wins,
                    COUNT(CASE WHEN bet_result = 'loss' THEN 1 END) as losses
                FROM h1_pricing_events
                WHERE clv_pct IS NOT NULL AND clv_pct BETWEEN -50 AND 50
                GROUP BY recommended_side
            """))
            print("\n   Resultados por lado recomendado:")
            print("   Lado     | N   | Dev A  | Dev B  | CLV    | Profit | W/L")
            for row in result.fetchall():
                print(f"   {row[0]:8} | {row[1]:3} | {row[2]:6}% | {row[3]:6}% | {row[4]:6}% | {row[5]:6} | {row[6]}/{row[7]}")
            
            # Verificar se inverter melhora
            result = await session.execute(text("""
                SELECT 
                    COUNT(*) as n,
                    ROUND(AVG(clv_pct)::numeric, 3) as clv_original,
                    -- Se recomenda side_a, o CLV do lado oposto seria baseado em odd_b
                    -- Não temos CLV do lado oposto diretamente, mas podemos ver se profit inverso é melhor
                    ROUND(AVG(-profit_loss)::numeric, 3) as profit_invertido,
                    ROUND(AVG(profit_loss)::numeric, 3) as profit_original
                FROM h1_pricing_events
                WHERE clv_pct IS NOT NULL 
                  AND clv_pct BETWEEN -50 AND 50
                  AND bet_result IN ('win', 'loss')
            """))
            row = result.fetchone()
            if row:
                melhor = "INVERTER" if (row[2] or 0) > (row[3] or 0) else "MANTER"
                print(f"\n   Teste de inversão: Profit original={row[3]}, Profit invertido={row[2]} → {melhor}")
            
            # ============================================================
            # H3 - MONOTONICIDADE
            # ============================================================
            print("\n" + "=" * 70)
            print("H3 - MONOTONICIDADE: LÓGICA DE ESCOLHA DE LADO")
            print("=" * 70)
            print("""
LÓGICA ATUAL:
- Detecta inversão: linha A tem odd > linha B adjacente (quando deveria ser <)
- recommended_line = linha com melhor preço relativo
- side = o lado (home/away) onde a inversão foi detectada

DÚVIDA:
- Estamos apostando no lado CERTO da linha recomendada?
- Se há inversão, qual linha tem o "valor"?
""")
            
            result = await session.execute(text("""
                SELECT 
                    side,
                    COUNT(*) as n,
                    ROUND(AVG(clv_pct)::numeric, 2) as clv_medio,
                    ROUND(AVG(profit_loss)::numeric, 3) as profit_medio,
                    COUNT(CASE WHEN bet_result = 'win' THEN 1 END) as wins,
                    COUNT(CASE WHEN bet_result = 'loss' THEN 1 END) as losses
                FROM h3_line_monotonicity_events
                WHERE clv_pct IS NOT NULL AND clv_pct BETWEEN -50 AND 50
                GROUP BY side
            """))
            print("\n   Resultados por lado:")
            print("   Lado     | N   | CLV    | Profit | W/L")
            for row in result.fetchall():
                print(f"   {row[0]:8} | {row[1]:3} | {row[2]:6}% | {row[3]:6} | {row[4]}/{row[5]}")
            
            # ============================================================
            # H3B - REVERSÕES TEMPORAIS
            # ============================================================
            print("\n" + "=" * 70)
            print("H3B - REVERSÕES TEMPORAIS: LÓGICA DE ESCOLHA DE LADO")
            print("=" * 70)
            print("""
LÓGICA ATUAL:
- Detecta quando odd reverte direção (estava subindo e desceu, ou vice-versa)
- direction_before = direção ANTES da reversão
- direction_after = direção DEPOIS da reversão (nova direção)
- bet_side = lado onde a reversão aconteceu

PROBLEMA POTENCIAL:
- Se odd estava SUBINDO e DESCEU (reversão para baixo):
  → Aposta no lado que desceu? Ou no oposto?
- Se odd estava DESCENDO e SUBIU (reversão para cima):
  → A odd MELHOROU - deveria apostar NELA
  
QUESTÃO: estamos apostando NA direção da reversão ou CONTRA?
""")
            
            result = await session.execute(text("""
                SELECT 
                    direction_before,
                    direction_after,
                    bet_side,
                    COUNT(*) as n,
                    ROUND(AVG(clv_pct)::numeric, 2) as clv_medio,
                    ROUND(AVG(profit_loss)::numeric, 3) as profit_medio,
                    COUNT(CASE WHEN bet_result = 'win' THEN 1 END) as wins,
                    COUNT(CASE WHEN bet_result = 'loss' THEN 1 END) as losses
                FROM h3b_temporal_reversal_events
                WHERE clv_pct IS NOT NULL AND clv_pct BETWEEN -50 AND 50
                GROUP BY direction_before, direction_after, bet_side
                ORDER BY n DESC
            """))
            print("\n   Resultados por direção:")
            print("   Antes    | Depois   | Lado     | N   | CLV    | Profit | W/L")
            for row in result.fetchall():
                print(f"   {row[0]:8} | {row[1]:8} | {row[2]:8} | {row[3]:3} | {row[4]:6}% | {row[5]:6} | {row[6]}/{row[7]}")
            
            # Teste de inversão
            result = await session.execute(text("""
                SELECT 
                    direction_after,
                    COUNT(*) as n,
                    ROUND(AVG(profit_loss)::numeric, 3) as profit_original,
                    ROUND(AVG(-profit_loss)::numeric, 3) as profit_invertido
                FROM h3b_temporal_reversal_events
                WHERE clv_pct IS NOT NULL 
                  AND clv_pct BETWEEN -50 AND 50
                  AND bet_result IN ('win', 'loss')
                GROUP BY direction_after
            """))
            print("\n   Teste de inversão por direção:")
            for row in result.fetchall():
                melhor = "INVERTER" if (row[3] or 0) > (row[2] or 0) else "MANTER"
                print(f"   {row[0]}: Profit orig={row[2]}, invertido={row[3]} → {melhor}")
            
            # ============================================================
            # H6 - CORRELAÇÃO/LAG
            # ============================================================
            print("\n" + "=" * 70)
            print("H6 - CORRELAÇÃO/LAG: LÓGICA DE ESCOLHA DE LADO")
            print("=" * 70)
            print("""
LÓGICA ATUAL:
- Detecta quando linha LÍDER moveu mas linha ADJACENTE não acompanhou
- Aposta na linha "atrasada" (lagged_line) esperando correção
- bet_side = lado que deveria se mover

QUESTÃO: se a líder SUBIU, a atrasada deveria SUBIR também?
- Se sim, apostar AGORA (antes da correção) daria valor
- Mas estamos apostando no lado CERTO?
""")
            
            result = await session.execute(text("""
                SELECT 
                    bet_side,
                    COUNT(*) as n,
                    ROUND(AVG(clv_pct)::numeric, 2) as clv_medio,
                    ROUND(AVG(profit_loss)::numeric, 3) as profit_medio,
                    COUNT(CASE WHEN bet_result = 'win' THEN 1 END) as wins,
                    COUNT(CASE WHEN bet_result = 'loss' THEN 1 END) as losses
                FROM h6_correlation_lag_events
                WHERE clv_pct IS NOT NULL AND clv_pct BETWEEN -50 AND 50
                GROUP BY bet_side
            """))
            print("\n   Resultados por lado:")
            print("   Lado     | N   | CLV    | Profit | W/L")
            for row in result.fetchall():
                print(f"   {row[0]:8} | {row[1]:3} | {row[2]:6}% | {row[3]:6} | {row[4]}/{row[5]}")
            
            # ============================================================
            # H6 - INVESTIGAR BASELINE ALTO
            # ============================================================
            print("\n" + "=" * 70)
            print("H6 - INVESTIGAÇÃO: POR QUE BASELINE = 19%?")
            print("=" * 70)
            
            # Ver distribuição do baseline
            result = await session.execute(text("""
                WITH baseline_calc AS (
                    SELECT 
                        e.id,
                        e.match_id,
                        e.detected_at,
                        (
                            SELECT 
                                CASE WHEN closing.best_home_odds > 0 
                                THEN (snapshot.best_home_odds - closing.best_home_odds) 
                                     / closing.best_home_odds * 100
                                ELSE NULL END as clv
                            FROM best_odds_history snapshot
                            JOIN matches m ON snapshot.match_id = m.id
                            LEFT JOIN LATERAL (
                                SELECT best_home_odds 
                                FROM best_odds_history c
                                WHERE c.match_id = snapshot.match_id 
                                  AND c.ah_line = snapshot.ah_line
                                  AND c.scraped_at < m.kickoff_time
                                ORDER BY c.scraped_at DESC
                                LIMIT 1
                            ) closing ON TRUE
                            WHERE snapshot.match_id = e.match_id
                              AND snapshot.ah_line != e.lagged_line::text
                              AND ABS(EXTRACT(EPOCH FROM (snapshot.scraped_at - e.detected_at))) < 30
                              AND closing.best_home_odds > 0
                            ORDER BY ABS(
                                (snapshot.best_home_odds - closing.best_home_odds) 
                                / closing.best_home_odds * 100
                            ) DESC
                            LIMIT 1
                        ) as max_clv_outra_linha
                    FROM h6_correlation_lag_events e
                    WHERE e.clv_pct IS NOT NULL
                      AND e.clv_pct BETWEEN -50 AND 50
                )
                SELECT 
                    CASE 
                        WHEN max_clv_outra_linha < -20 THEN '< -20%'
                        WHEN max_clv_outra_linha < -10 THEN '-20% a -10%'
                        WHEN max_clv_outra_linha < 0 THEN '-10% a 0%'
                        WHEN max_clv_outra_linha < 10 THEN '0% a 10%'
                        WHEN max_clv_outra_linha < 20 THEN '10% a 20%'
                        WHEN max_clv_outra_linha < 50 THEN '20% a 50%'
                        ELSE '> 50%'
                    END as faixa,
                    COUNT(*) as n
                FROM baseline_calc
                WHERE max_clv_outra_linha IS NOT NULL
                GROUP BY 1
                ORDER BY 1
            """))
            print("\n   Distribuição dos CLVs das outras linhas (para calcular baseline):")
            print("   Faixa CLV     | N")
            for row in result.fetchall():
                print(f"   {row[0]:14} | {row[1]}")
            
            # Verificar se há outliers extremos
            result = await session.execute(text("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN ABS(clv_pct) > 30 THEN 1 END) as outliers_evento,
                    ROUND(AVG(clv_pct) FILTER (WHERE ABS(clv_pct) <= 30)::numeric, 2) as clv_sem_outliers
                FROM h6_correlation_lag_events
                WHERE clv_pct IS NOT NULL
            """))
            row = result.fetchone()
            if row:
                print(f"\n   Outliers no H6: {row[1]} de {row[0]} eventos com |CLV| > 30%")
                print(f"   CLV médio sem outliers: {row[2]}%")
            
            # ============================================================
            # RESUMO E RECOMENDAÇÕES
            # ============================================================
            print("\n" + "=" * 70)
            print("RESUMO E RECOMENDAÇÕES")
            print("=" * 70)
            print("""
1. VERIFICAR LÓGICA DE INVERSÃO:
   - Se profit_invertido > profit_original em alguma hipótese,
     a lógica de escolha de lado pode estar ERRADA
   
2. BASELINE ALTO NO H6:
   - Provavelmente há outliers extremos nas "outras linhas"
   - Mercados 1X2 com odds altas podem distorcer
   - Solução: filtrar outliers ou usar mediana no baseline

3. PRÓXIMOS PASSOS:
   - Aumentar amostra (mais dias de coleta)
   - Revisar lógica de cada detector se inversão for melhor
   - Usar mediana ao invés de média para baseline
""")
            
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(debug_sides())
