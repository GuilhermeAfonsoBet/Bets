# -*- coding: utf-8 -*-
"""
Debug: Por que o CLV baseline está tão alto?
"""

import asyncio
import sys
from sqlalchemy import text

sys.path.insert(0, '.')

from storage.database import Database


async def debug_baseline():
    db = Database()
    await db.connect()
    
    print("=" * 70)
    print("DEBUG: ANÁLISE DO CLV BASELINE")
    print("=" * 70)
    
    try:
        async with db.async_session() as session:
            
            # ============================================================
            # 1. COMO O BASELINE ESTÁ SENDO CALCULADO
            # ============================================================
            print("\n" + "=" * 70)
            print("1. LÓGICA ATUAL DO CLV BASELINE")
            print("=" * 70)
            print("""
Para cada evento detectado:
1. Busca TODAS as outras linhas coletadas no mesmo momento (±30s)
2. Para cada linha, calcula: CLV = (odd_momento - closing) / closing * 100
3. Baseline = MÉDIA desses CLVs

PROBLEMA POTENCIAL:
- Inclui TODAS as linhas (AH, OU, 1X2, 1X2_DRAW)
- Linhas com odds muito diferentes podem ter CLVs extremos
- Outliers distorcem a média
""")
            
            # ============================================================
            # 2. DISTRIBUIÇÃO DOS CLVs DAS OUTRAS LINHAS
            # ============================================================
            print("\n" + "=" * 70)
            print("2. DISTRIBUIÇÃO DOS CLVs DAS OUTRAS LINHAS")
            print("=" * 70)
            
            result = await session.execute(text("""
                WITH outras_linhas AS (
                    SELECT 
                        e.id as evento_id,
                        snapshot.ah_line,
                        snapshot.best_home_odds as odd_momento,
                        closing.best_home_odds as odd_closing,
                        CASE WHEN closing.best_home_odds > 0 
                             THEN (snapshot.best_home_odds - closing.best_home_odds) 
                                  / closing.best_home_odds * 100
                             ELSE NULL END as clv_linha
                    FROM h1_pricing_events e
                    JOIN best_odds_history snapshot ON snapshot.match_id = e.match_id
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
                    WHERE snapshot.ah_line != e.ah_line
                      AND ABS(EXTRACT(EPOCH FROM (snapshot.scraped_at - e.detected_at))) < 30
                      AND closing.best_home_odds > 0
                      AND e.clv_pct IS NOT NULL
                    LIMIT 10000
                )
                SELECT 
                    CASE 
                        WHEN clv_linha < -50 THEN '< -50%'
                        WHEN clv_linha < -20 THEN '-50% a -20%'
                        WHEN clv_linha < -10 THEN '-20% a -10%'
                        WHEN clv_linha < 0 THEN '-10% a 0%'
                        WHEN clv_linha < 10 THEN '0% a 10%'
                        WHEN clv_linha < 20 THEN '10% a 20%'
                        WHEN clv_linha < 50 THEN '20% a 50%'
                        ELSE '> 50%'
                    END as faixa,
                    COUNT(*) as n,
                    ROUND(AVG(clv_linha)::numeric, 2) as clv_medio_faixa
                FROM outras_linhas
                WHERE clv_linha IS NOT NULL
                GROUP BY 1
                ORDER BY 1
            """))
            print("\n   Distribuição dos CLVs das outras linhas:")
            print("   Faixa         | N       | CLV médio")
            print("   " + "-" * 40)
            for row in result.fetchall():
                print(f"   {row[0]:14} | {row[1]:7} | {row[2]:8}%")
            
            # ============================================================
            # 3. OUTLIERS EXTREMOS
            # ============================================================
            print("\n" + "=" * 70)
            print("3. EXEMPLOS DE OUTLIERS EXTREMOS (CLV > 50%)")
            print("=" * 70)
            
            result = await session.execute(text("""
                WITH outras_linhas AS (
                    SELECT 
                        snapshot.ah_line,
                        snapshot.best_home_odds as odd_momento,
                        closing.best_home_odds as odd_closing,
                        CASE WHEN closing.best_home_odds > 0 
                             THEN (snapshot.best_home_odds - closing.best_home_odds) 
                                  / closing.best_home_odds * 100
                             ELSE NULL END as clv_linha
                    FROM h1_pricing_events e
                    JOIN best_odds_history snapshot ON snapshot.match_id = e.match_id
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
                    WHERE snapshot.ah_line != e.ah_line
                      AND ABS(EXTRACT(EPOCH FROM (snapshot.scraped_at - e.detected_at))) < 30
                      AND closing.best_home_odds > 0
                    LIMIT 10000
                )
                SELECT 
                    ah_line,
                    ROUND(odd_momento::numeric, 2) as odd_momento,
                    ROUND(odd_closing::numeric, 2) as odd_closing,
                    ROUND(clv_linha::numeric, 2) as clv
                FROM outras_linhas
                WHERE ABS(clv_linha) > 50
                ORDER BY ABS(clv_linha) DESC
                LIMIT 15
            """))
            print("\n   Linha        | Odd momento | Odd closing | CLV")
            print("   " + "-" * 50)
            for row in result.fetchall():
                print(f"   {row[0]:12} | {row[1]:11} | {row[2]:11} | {row[3]:8}%")
            
            # ============================================================
            # 4. MÉDIA SEM OUTLIERS
            # ============================================================
            print("\n" + "=" * 70)
            print("4. CLV BASELINE FILTRADO (sem outliers > 50%)")
            print("=" * 70)
            
            result = await session.execute(text("""
                WITH evento_com_baseline AS (
                    SELECT 
                        e.id,
                        e.clv_pct as clv_evento,
                        (
                            SELECT AVG(clv_calc)
                            FROM (
                                SELECT 
                                    CASE WHEN closing.best_home_odds > 0 
                                    THEN (snapshot.best_home_odds - closing.best_home_odds) 
                                         / closing.best_home_odds * 100
                                    ELSE NULL END as clv_calc
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
                            ) sub
                            WHERE clv_calc BETWEEN -50 AND 50  -- FILTRO DE OUTLIERS
                        ) as clv_baseline
                    FROM h1_pricing_events e
                    WHERE e.clv_pct IS NOT NULL
                      AND e.clv_pct BETWEEN -50 AND 50
                )
                SELECT 
                    'H1' as hipotese,
                    COUNT(*) as n,
                    ROUND(AVG(clv_evento)::numeric, 3) as clv_evento,
                    ROUND(AVG(clv_baseline)::numeric, 3) as clv_baseline,
                    ROUND(AVG(clv_evento - COALESCE(clv_baseline, 0))::numeric, 3) as clv_adicional,
                    ROUND(STDDEV(clv_evento - COALESCE(clv_baseline, 0))::numeric, 3) as clv_adicional_std
                FROM evento_com_baseline
                WHERE clv_baseline IS NOT NULL
            """))
            row = result.fetchone()
            if row:
                print(f"""
   H1 (com filtro de outliers no baseline):
   N = {row[1]}
   CLV evento:     {row[2]}%
   CLV baseline:   {row[3]}%
   CLV ADICIONAL:  {row[4]}% ± {row[5]}%
""")
            
            # ============================================================
            # 5. PROPOSTA: USAR MEDIANA
            # ============================================================
            print("\n" + "=" * 70)
            print("5. PROPOSTA: USAR MEDIANA EM VEZ DE MÉDIA")
            print("=" * 70)
            print("""
A MEDIANA é mais robusta a outliers que a MÉDIA.
Em vez de calcular média dos CLVs das outras linhas,
calcular a mediana.
""")
            
            result = await session.execute(text("""
                WITH outras_linhas AS (
                    SELECT 
                        e.id as evento_id,
                        CASE WHEN closing.best_home_odds > 0 
                             THEN (snapshot.best_home_odds - closing.best_home_odds) 
                                  / closing.best_home_odds * 100
                             ELSE NULL END as clv_linha
                    FROM h1_pricing_events e
                    JOIN best_odds_history snapshot ON snapshot.match_id = e.match_id
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
                    WHERE snapshot.ah_line != e.ah_line
                      AND ABS(EXTRACT(EPOCH FROM (snapshot.scraped_at - e.detected_at))) < 30
                      AND closing.best_home_odds > 0
                      AND e.clv_pct IS NOT NULL
                      AND e.clv_pct BETWEEN -50 AND 50
                ),
                baseline_por_evento AS (
                    SELECT 
                        evento_id,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY clv_linha) as mediana_baseline
                    FROM outras_linhas
                    WHERE clv_linha IS NOT NULL
                    GROUP BY evento_id
                )
                SELECT 
                    COUNT(*) as n,
                    ROUND(AVG(e.clv_pct)::numeric, 3) as clv_evento,
                    ROUND(AVG(b.mediana_baseline)::numeric, 3) as baseline_mediana,
                    ROUND(AVG(e.clv_pct - b.mediana_baseline)::numeric, 3) as clv_adicional,
                    ROUND(STDDEV(e.clv_pct - b.mediana_baseline)::numeric, 3) as clv_adicional_std
                FROM h1_pricing_events e
                JOIN baseline_por_evento b ON e.id = b.evento_id
                WHERE e.clv_pct BETWEEN -50 AND 50
            """))
            row = result.fetchone()
            if row:
                sinal = "✅ AGREGA" if row[3] and row[3] > 0 else "❌ NÃO AGREGA"
                print(f"""
   H1 (baseline = MEDIANA):
   N = {row[0]}
   CLV evento:     {row[1]}%
   CLV baseline:   {row[2]}%
   CLV ADICIONAL:  {row[3]}% ± {row[4]}%  {sinal}
""")
            
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(debug_baseline())
