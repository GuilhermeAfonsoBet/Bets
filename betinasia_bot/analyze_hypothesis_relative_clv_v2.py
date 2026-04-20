# -*- coding: utf-8 -*-
"""
Análise de CLV Relativo - Versão Corrigida

CLV adicional = CLV do evento - CLV baseline (outras linhas no MESMO MOMENTO)

Isso isola o valor específico da linha detectada vs o mercado geral
naquele instante de tempo.
"""

import asyncio
import sys
from sqlalchemy import text

sys.path.insert(0, '.')

from storage.database import Database


async def analyze_relative_clv():
    db = Database()
    await db.connect()
    
    print("=" * 70)
    print("ANÁLISE DE CLV RELATIVO - V2 (CORRIGIDA)")
    print("=" * 70)
    print("""
METODOLOGIA:
- Para cada evento detectado, calculamos o CLV
- Baseline = média do CLV de TODAS as outras linhas no MESMO MOMENTO
- CLV adicional = CLV evento - CLV baseline
- Isso mede se o detector encontra linhas MELHORES que as outras disponíveis
""")
    
    try:
        async with db.async_session() as session:
            
            # ============================================================
            # H1 - PRECIFICAÇÃO
            # ============================================================
            print("\n" + "=" * 70)
            print("H1 - PRECIFICAÇÃO")
            print("=" * 70)
            
            # Para cada evento H1, buscar CLV de outras linhas no mesmo momento
            result = await session.execute(text("""
                WITH evento_com_baseline AS (
                    SELECT 
                        e.id,
                        e.match_id,
                        e.detected_at,
                        e.ah_line as linha_evento,
                        e.clv_pct as clv_evento,
                        (
                            -- Baseline: CLV médio de outras linhas no mesmo jogo, 
                            -- coletadas no mesmo momento (±30 segundos)
                            SELECT AVG(
                                CASE WHEN closing.best_home_odds > 0 
                                THEN (snapshot.best_home_odds - closing.best_home_odds) 
                                     / closing.best_home_odds * 100
                                ELSE NULL END
                            )
                            FROM best_odds_history snapshot
                            JOIN matches m ON snapshot.match_id = m.id
                            -- Closing line para cada linha
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
                              AND snapshot.ah_line != e.ah_line  -- outras linhas
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
                sinal = "✅ AGREGA VALOR" if row[5] and row[5] > 0 else "❌ NÃO AGREGA"
                print(f"""
   Eventos analisados: {row[0]}

   CLV do evento (média):    {row[1]:>7}% ± {row[2]}%
   CLV baseline (média):     {row[3]:>7}% ± {row[4]}%
   ─────────────────────────────────────────
   CLV ADICIONAL:            {row[5]:>7}% ± {row[6]}%  {sinal}
   Taxa eventos CLV > baseline: {row[7]}%
""")
            else:
                print("   Dados insuficientes para análise")
            
            # ============================================================
            # H3 - MONOTONICIDADE ENTRE LINHAS
            # ============================================================
            print("\n" + "=" * 70)
            print("H3 - MONOTONICIDADE ENTRE LINHAS")
            print("=" * 70)
            
            result = await session.execute(text("""
                WITH evento_com_baseline AS (
                    SELECT 
                        e.id,
                        e.match_id,
                        e.detected_at,
                        e.recommended_line as linha_evento,
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
                sinal = "✅ AGREGA VALOR" if row[5] and row[5] > 0 else "❌ NÃO AGREGA"
                print(f"""
   Eventos analisados: {row[0]}

   CLV do evento (média):    {row[1]:>7}% ± {row[2]}%
   CLV baseline (média):     {row[3]:>7}% ± {row[4]}%
   ─────────────────────────────────────────
   CLV ADICIONAL:            {row[5]:>7}% ± {row[6]}%  {sinal}
   Taxa eventos CLV > baseline: {row[7]}%
""")
            else:
                print("   Dados insuficientes para análise")
            
            # ============================================================
            # H3B - REVERSÕES TEMPORAIS
            # ============================================================
            print("\n" + "=" * 70)
            print("H3B - REVERSÕES TEMPORAIS")
            print("=" * 70)
            
            result = await session.execute(text("""
                WITH evento_com_baseline AS (
                    SELECT 
                        e.id,
                        e.match_id,
                        e.detected_at,
                        e.ah_line as linha_evento,
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
                sinal = "✅ AGREGA VALOR" if row[5] and row[5] > 0 else "❌ NÃO AGREGA"
                print(f"""
   Eventos analisados: {row[0]}

   CLV do evento (média):    {row[1]:>7}% ± {row[2]}%
   CLV baseline (média):     {row[3]:>7}% ± {row[4]}%
   ─────────────────────────────────────────
   CLV ADICIONAL:            {row[5]:>7}% ± {row[6]}%  {sinal}
   Taxa eventos CLV > baseline: {row[7]}%
""")
            else:
                print("   Dados insuficientes para análise")
            
            # ============================================================
            # H6 - CORRELAÇÃO/LAG
            # ============================================================
            print("\n" + "=" * 70)
            print("H6 - CORRELAÇÃO/LAG")
            print("=" * 70)
            
            result = await session.execute(text("""
                WITH evento_com_baseline AS (
                    SELECT 
                        e.id,
                        e.match_id,
                        e.detected_at,
                        e.lagged_line::text as linha_evento,
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
                sinal = "✅ AGREGA VALOR" if row[5] and row[5] > 0 else "❌ NÃO AGREGA"
                print(f"""
   Eventos analisados: {row[0]}

   CLV do evento (média):    {row[1]:>7}% ± {row[2]}%
   CLV baseline (média):     {row[3]:>7}% ± {row[4]}%
   ─────────────────────────────────────────
   CLV ADICIONAL:            {row[5]:>7}% ± {row[6]}%  {sinal}
   Taxa eventos CLV > baseline: {row[7]}%
""")
            else:
                print("   Dados insuficientes para análise")
            
            # ============================================================
            # INTERPRETAÇÃO
            # ============================================================
            print("\n" + "=" * 70)
            print("INTERPRETAÇÃO")
            print("=" * 70)
            print("""
CLV ADICIONAL = CLV do evento - CLV das outras linhas no mesmo momento

Exemplo:
- Evento H1 às 14:30 na AH -0.5: CLV = +3%
- Outras linhas no mesmo momento: AH -1.0 CLV=+1%, OU 2.5 CLV=+2%
- Baseline = média(+1%, +2%) = +1.5%
- CLV adicional = +3% - 1.5% = +1.5%

Se CLV adicional > 0:
   → O detector encontra linhas MELHORES que as outras disponíveis
   → Escolhe o mercado certo no momento certo

Se CLV adicional ≈ 0:
   → A linha detectada não é melhor que as outras
   → Poderia apostar em qualquer linha com mesmo resultado

Se CLV adicional < 0:
   → O detector encontra linhas PIORES que as outras
   → Melhor apostar em outras linhas ou inverter a lógica
""")
            
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(analyze_relative_clv())
