# -*- coding: utf-8 -*-
"""
Análise de Hipóteses V4 - Com CLV Adicional e separação por direção
"""

import asyncio
import sys
from sqlalchemy import text

sys.path.insert(0, '.')

from storage.database import Database


async def analyze_v4():
    db = Database()
    await db.connect()
    
    print("=" * 70)
    print("ANÁLISE DE HIPÓTESES V4 - CLV ADICIONAL + SEPARAÇÃO POR DIREÇÃO")
    print("=" * 70)
    print("""
CLV ADICIONAL = CLV do evento - CLV das outras linhas no MESMO MOMENTO
Mede se o detector encontra linhas MELHORES que as outras disponíveis.
""")
    
    try:
        async with db.async_session() as session:
            
            # ============================================================
            # H1 - PRECIFICAÇÃO
            # ============================================================
            print("\n" + "=" * 70)
            print("H1 - PRECIFICAÇÃO")
            print("=" * 70)
            
            result = await session.execute(text("""
                WITH evento_com_baseline AS (
                    SELECT 
                        e.id,
                        e.clv_pct as clv_evento,
                        (
                            SELECT AVG(
                                CASE WHEN closing.best_home_odds > 0 
                                THEN (snapshot.best_home_odds - closing.best_home_odds) 
                                     / closing.best_home_odds * 100
                                ELSE NULL END
                            )
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
                              AND snapshot.ah_line != e.ah_line
                              AND ABS(EXTRACT(EPOCH FROM (snapshot.scraped_at - e.detected_at))) < 30
                              AND closing.best_home_odds > 0
                        ) as clv_baseline
                    FROM h1_pricing_events e
                    WHERE e.clv_pct IS NOT NULL
                      AND e.clv_pct BETWEEN -50 AND 50
                )
                SELECT 
                    COUNT(*) as n,
                    ROUND(AVG(clv_evento)::numeric, 3) as clv_evento_medio,
                    ROUND(STDDEV(clv_evento)::numeric, 3) as clv_evento_std,
                    ROUND(AVG(clv_baseline)::numeric, 3) as clv_baseline_medio,
                    ROUND(STDDEV(clv_baseline)::numeric, 3) as clv_baseline_std,
                    ROUND(AVG(clv_evento - COALESCE(clv_baseline, 0))::numeric, 3) as clv_adicional,
                    ROUND(STDDEV(clv_evento - COALESCE(clv_baseline, 0))::numeric, 3) as clv_adicional_std,
                    ROUND(100.0 * COUNT(CASE WHEN clv_evento > COALESCE(clv_baseline, 0) THEN 1 END) 
                          / COUNT(*)::numeric, 1) as taxa_positivo
                FROM evento_com_baseline
                WHERE clv_baseline IS NOT NULL
            """))
            row = result.fetchone()
            if row and row[0] > 0:
                sinal = "✅ AGREGA" if row[5] and row[5] > 0 else "❌ NÃO AGREGA"
                print(f"""
   N = {row[0]}
   CLV evento:     {row[1]:>7}% ± {row[2]}%
   CLV baseline:   {row[3]:>7}% ± {row[4]}%
   CLV ADICIONAL:  {row[5]:>7}% ± {row[6]}%  {sinal}
   Taxa CLV > baseline: {row[7]}%
""")
            
            # ============================================================
            # H3 - MONOTONICIDADE
            # ============================================================
            print("\n" + "=" * 70)
            print("H3 - MONOTONICIDADE ENTRE LINHAS")
            print("=" * 70)
            
            result = await session.execute(text("""
                WITH evento_com_baseline AS (
                    SELECT 
                        e.id,
                        e.clv_pct as clv_evento,
                        (
                            SELECT AVG(
                                CASE WHEN closing.best_home_odds > 0 
                                THEN (snapshot.best_home_odds - closing.best_home_odds) 
                                     / closing.best_home_odds * 100
                                ELSE NULL END
                            )
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
                              AND snapshot.ah_line != e.recommended_line
                              AND ABS(EXTRACT(EPOCH FROM (snapshot.scraped_at - e.detected_at))) < 30
                              AND closing.best_home_odds > 0
                        ) as clv_baseline
                    FROM h3_line_monotonicity_events e
                    WHERE e.clv_pct IS NOT NULL
                      AND e.clv_pct BETWEEN -50 AND 50
                )
                SELECT 
                    COUNT(*) as n,
                    ROUND(AVG(clv_evento)::numeric, 3) as clv_evento_medio,
                    ROUND(STDDEV(clv_evento)::numeric, 3) as clv_evento_std,
                    ROUND(AVG(clv_baseline)::numeric, 3) as clv_baseline_medio,
                    ROUND(STDDEV(clv_baseline)::numeric, 3) as clv_baseline_std,
                    ROUND(AVG(clv_evento - COALESCE(clv_baseline, 0))::numeric, 3) as clv_adicional,
                    ROUND(STDDEV(clv_evento - COALESCE(clv_baseline, 0))::numeric, 3) as clv_adicional_std,
                    ROUND(100.0 * COUNT(CASE WHEN clv_evento > COALESCE(clv_baseline, 0) THEN 1 END) 
                          / COUNT(*)::numeric, 1) as taxa_positivo
                FROM evento_com_baseline
                WHERE clv_baseline IS NOT NULL
            """))
            row = result.fetchone()
            if row and row[0] > 0:
                sinal = "✅ AGREGA" if row[5] and row[5] > 0 else "❌ NÃO AGREGA"
                print(f"""
   N = {row[0]}
   CLV evento:     {row[1]:>7}% ± {row[2]}%
   CLV baseline:   {row[3]:>7}% ± {row[4]}%
   CLV ADICIONAL:  {row[5]:>7}% ± {row[6]}%  {sinal}
   Taxa CLV > baseline: {row[7]}%
""")
            
            # ============================================================
            # H3B - REVERSÕES TEMPORAIS (SEPARADO POR DIREÇÃO)
            # ============================================================
            print("\n" + "=" * 70)
            print("H3B - REVERSÕES TEMPORAIS (POR DIREÇÃO)")
            print("=" * 70)
            
            for direcao in ['up', 'down']:
                result = await session.execute(text(f"""
                    WITH evento_com_baseline AS (
                        SELECT 
                            e.id,
                            e.clv_pct as clv_evento,
                            (
                                SELECT AVG(
                                    CASE WHEN closing.best_home_odds > 0 
                                    THEN (snapshot.best_home_odds - closing.best_home_odds) 
                                         / closing.best_home_odds * 100
                                    ELSE NULL END
                                )
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
                                  AND snapshot.ah_line != e.ah_line
                                  AND ABS(EXTRACT(EPOCH FROM (snapshot.scraped_at - e.detected_at))) < 30
                                  AND closing.best_home_odds > 0
                            ) as clv_baseline
                        FROM h3b_temporal_reversal_events e
                        WHERE e.clv_pct IS NOT NULL
                          AND e.clv_pct BETWEEN -50 AND 50
                          AND e.direction_after = '{direcao}'
                    )
                    SELECT 
                        COUNT(*) as n,
                        ROUND(AVG(clv_evento)::numeric, 3) as clv_evento_medio,
                        ROUND(STDDEV(clv_evento)::numeric, 3) as clv_evento_std,
                        ROUND(AVG(clv_baseline)::numeric, 3) as clv_baseline_medio,
                        ROUND(STDDEV(clv_baseline)::numeric, 3) as clv_baseline_std,
                        ROUND(AVG(clv_evento - COALESCE(clv_baseline, 0))::numeric, 3) as clv_adicional,
                        ROUND(STDDEV(clv_evento - COALESCE(clv_baseline, 0))::numeric, 3) as clv_adicional_std,
                        ROUND(100.0 * COUNT(CASE WHEN clv_evento > COALESCE(clv_baseline, 0) THEN 1 END) 
                              / NULLIF(COUNT(*), 0)::numeric, 1) as taxa_positivo
                    FROM evento_com_baseline
                    WHERE clv_baseline IS NOT NULL
                """))
                row = result.fetchone()
                
                label = "REVERSÃO UP (odd subiu)" if direcao == 'up' else "REVERSÃO DOWN (odd desceu)"
                if row and row[0] > 0:
                    sinal = "✅ AGREGA" if row[5] and row[5] > 0 else "❌ NÃO AGREGA"
                    print(f"""
   {label}:
   N = {row[0]}
   CLV evento:     {row[1]:>7}% ± {row[2]}%
   CLV baseline:   {row[3]:>7}% ± {row[4]}%
   CLV ADICIONAL:  {row[5]:>7}% ± {row[6]}%  {sinal}
   Taxa CLV > baseline: {row[7]}%
""")
                else:
                    print(f"\n   {label}: Dados insuficientes\n")
            
            # ============================================================
            # H6 - CORRELAÇÃO/LAG (SEPARADO POR DIREÇÃO)
            # ============================================================
            print("\n" + "=" * 70)
            print("H6 - CORRELAÇÃO/LAG (POR DIREÇÃO DO LÍDER)")
            print("=" * 70)
            
            for direcao in ['down', 'up']:
                result = await session.execute(text(f"""
                    WITH evento_com_baseline AS (
                        SELECT 
                            e.id,
                            e.clv_pct as clv_evento,
                            (
                                SELECT AVG(
                                    CASE WHEN closing.best_home_odds > 0 
                                    THEN (snapshot.best_home_odds - closing.best_home_odds) 
                                         / closing.best_home_odds * 100
                                    ELSE NULL END
                                )
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
                            ) as clv_baseline
                        FROM h6_correlation_lag_events e
                        WHERE e.clv_pct IS NOT NULL
                          AND e.clv_pct BETWEEN -50 AND 50
                          AND e.leader_move_direction = '{direcao}'
                    )
                    SELECT 
                        COUNT(*) as n,
                        ROUND(AVG(clv_evento)::numeric, 3) as clv_evento_medio,
                        ROUND(STDDEV(clv_evento)::numeric, 3) as clv_evento_std,
                        ROUND(AVG(clv_baseline)::numeric, 3) as clv_baseline_medio,
                        ROUND(STDDEV(clv_baseline)::numeric, 3) as clv_baseline_std,
                        ROUND(AVG(clv_evento - COALESCE(clv_baseline, 0))::numeric, 3) as clv_adicional,
                        ROUND(STDDEV(clv_evento - COALESCE(clv_baseline, 0))::numeric, 3) as clv_adicional_std,
                        ROUND(100.0 * COUNT(CASE WHEN clv_evento > COALESCE(clv_baseline, 0) THEN 1 END) 
                              / NULLIF(COUNT(*), 0)::numeric, 1) as taxa_positivo
                    FROM evento_com_baseline
                    WHERE clv_baseline IS NOT NULL
                """))
                row = result.fetchone()
                
                if direcao == 'down':
                    label = "LÍDER DOWN (usar apenas este)"
                else:
                    label = "LÍDER UP (ignorar - lógica corrigida)"
                    
                if row and row[0] > 0:
                    sinal = "✅ AGREGA" if row[5] and row[5] > 0 else "❌ NÃO AGREGA"
                    print(f"""
   {label}:
   N = {row[0]}
   CLV evento:     {row[1]:>7}% ± {row[2]}%
   CLV baseline:   {row[3]:>7}% ± {row[4]}%
   CLV ADICIONAL:  {row[5]:>7}% ± {row[6]}%  {sinal}
   Taxa CLV > baseline: {row[7]}%
""")
                else:
                    print(f"\n   {label}: Dados insuficientes\n")
            
            # ============================================================
            # CONCLUSÃO
            # ============================================================
            print("\n" + "=" * 70)
            print("CONCLUSÃO")
            print("=" * 70)
            print("""
CLV ADICIONAL > 0: detector encontra linhas melhores que as outras disponíveis
CLV ADICIONAL ≈ 0: detector não agrega valor vs escolha aleatória
CLV ADICIONAL < 0: detector encontra linhas piores

PRÓXIMOS PASSOS:
1. Continuar coletando dados com a nova lógica
2. Reanalisar quando tivermos N > 200 para cada hipótese
3. ICs grandes indicam que precisamos de mais dados
""")
            
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(analyze_v4())
