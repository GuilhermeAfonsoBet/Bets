# -*- coding: utf-8 -*-
"""
Análise de Hipóteses V6 - Com Intervalos de Confiança CORRETOS
"""

import asyncio
import sys
import math
from sqlalchemy import text

sys.path.insert(0, '.')

from storage.database import Database


def calc_ci(mean: float, std: float, n: int, confidence: float = 0.90) -> tuple:
    """Calcula intervalo de confiança."""
    if n <= 1:
        return (None, None)
    
    # Z-score para IC de 90% (bicaudal)
    z = 1.645
    
    se = std / math.sqrt(n)
    margin = z * se
    
    return (mean - margin, mean + margin)


def format_result(n: int, mean: float, std: float) -> str:
    """Formata resultado com IC."""
    se = std / math.sqrt(n) if n > 0 else 0
    ci_lower, ci_upper = calc_ci(mean, std, n)
    
    # Verifica significância (IC não inclui zero)
    if ci_lower is not None and ci_upper is not None:
        if ci_lower > 0:
            sig = "✅ SIGNIFICATIVO (p<0.10)"
        elif ci_upper < 0:
            sig = "❌ SIGNIFICATIVO NEGATIVO"
        else:
            sig = "⚪ Não significativo"
    else:
        sig = "N/A"
    
    return f"""
   N = {n}
   CLV adicional = {mean:.3f}%
   Erro padrão   = {se:.3f}%
   IC 90%        = [{ci_lower:.3f}%, {ci_upper:.3f}%]
   {sig}
"""


async def analyze_v6():
    db = Database()
    await db.connect()
    
    print("=" * 70)
    print("ANÁLISE DE HIPÓTESES V6 - INTERVALOS DE CONFIANÇA CORRETOS")
    print("=" * 70)
    print("""
CORREÇÃO: Agora mostra IC = média ± Z × (σ/√n), não apenas σ

IC 90% = média ± 1.645 × erro_padrão
Se IC não inclui zero → p < 0.10 (significativo)
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
                            WHERE clv_calc BETWEEN -50 AND 50
                        ) as clv_baseline
                    FROM h1_pricing_events e
                    WHERE e.clv_pct IS NOT NULL
                      AND e.clv_pct BETWEEN -50 AND 50
                )
                SELECT 
                    COUNT(*) as n,
                    AVG(clv_evento - COALESCE(clv_baseline, 0)) as clv_adicional,
                    STDDEV(clv_evento - COALESCE(clv_baseline, 0)) as clv_adicional_std
                FROM evento_com_baseline
                WHERE clv_baseline IS NOT NULL
            """))
            row = result.fetchone()
            if row and row[0] > 0:
                print(format_result(row[0], row[1], row[2]))
            
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
                                  AND snapshot.ah_line != e.recommended_line
                                  AND ABS(EXTRACT(EPOCH FROM (snapshot.scraped_at - e.detected_at))) < 30
                                  AND closing.best_home_odds > 0
                            ) sub
                            WHERE clv_calc BETWEEN -50 AND 50
                        ) as clv_baseline
                    FROM h3_line_monotonicity_events e
                    WHERE e.clv_pct IS NOT NULL
                      AND e.clv_pct BETWEEN -50 AND 50
                )
                SELECT 
                    COUNT(*) as n,
                    AVG(clv_evento - COALESCE(clv_baseline, 0)) as clv_adicional,
                    STDDEV(clv_evento - COALESCE(clv_baseline, 0)) as clv_adicional_std
                FROM evento_com_baseline
                WHERE clv_baseline IS NOT NULL
            """))
            row = result.fetchone()
            if row and row[0] > 0:
                print(format_result(row[0], row[1], row[2]))
            
            # ============================================================
            # H3B - REVERSÕES (POR DIREÇÃO)
            # ============================================================
            print("\n" + "=" * 70)
            print("H3B - REVERSÕES TEMPORAIS")
            print("=" * 70)
            
            for direcao, label in [('up', 'REVERSÃO UP'), ('down', 'REVERSÃO DOWN')]:
                result = await session.execute(text(f"""
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
                                WHERE clv_calc BETWEEN -50 AND 50
                            ) as clv_baseline
                        FROM h3b_temporal_reversal_events e
                        WHERE e.clv_pct IS NOT NULL
                          AND e.clv_pct BETWEEN -50 AND 50
                          AND e.direction_after = '{direcao}'
                    )
                    SELECT 
                        COUNT(*) as n,
                        AVG(clv_evento - COALESCE(clv_baseline, 0)) as clv_adicional,
                        STDDEV(clv_evento - COALESCE(clv_baseline, 0)) as clv_adicional_std
                    FROM evento_com_baseline
                    WHERE clv_baseline IS NOT NULL
                """))
                row = result.fetchone()
                print(f"\n   {label}:")
                if row and row[0] > 0:
                    print(format_result(row[0], row[1], row[2]))
            
            # ============================================================
            # H6 - CORRELAÇÃO (POR DIREÇÃO)
            # ============================================================
            print("\n" + "=" * 70)
            print("H6 - CORRELAÇÃO/LAG")
            print("=" * 70)
            
            for direcao, label in [('down', 'LÍDER DOWN'), ('up', 'LÍDER UP')]:
                result = await session.execute(text(f"""
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
                                      AND snapshot.ah_line != e.lagged_line::text
                                      AND ABS(EXTRACT(EPOCH FROM (snapshot.scraped_at - e.detected_at))) < 30
                                      AND closing.best_home_odds > 0
                                ) sub
                                WHERE clv_calc BETWEEN -50 AND 50
                            ) as clv_baseline
                        FROM h6_correlation_lag_events e
                        WHERE e.clv_pct IS NOT NULL
                          AND e.clv_pct BETWEEN -50 AND 50
                          AND e.leader_move_direction = '{direcao}'
                    )
                    SELECT 
                        COUNT(*) as n,
                        AVG(clv_evento - COALESCE(clv_baseline, 0)) as clv_adicional,
                        STDDEV(clv_evento - COALESCE(clv_baseline, 0)) as clv_adicional_std
                    FROM evento_com_baseline
                    WHERE clv_baseline IS NOT NULL
                """))
                row = result.fetchone()
                print(f"\n   {label}:")
                if row and row[0] > 0:
                    print(format_result(row[0], row[1], row[2]))
            
            # ============================================================
            # RESUMO
            # ============================================================
            print("\n" + "=" * 70)
            print("RESUMO")
            print("=" * 70)
            print("""
IC 90% = média ± 1.645 × (σ / √n)

✅ SIGNIFICATIVO: IC não inclui zero → p < 0.10
⚪ Não significativo: IC inclui zero → precisa mais dados
❌ NEGATIVO: IC totalmente abaixo de zero → hipótese não tem valor
""")
            
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(analyze_v6())
