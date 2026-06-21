# -*- coding: utf-8 -*-
"""
Debug dos detectores H3 e H3B - Verifica se estão funcionando corretamente.
"""

import asyncio
import sys
from datetime import datetime, timezone, timedelta
from sqlalchemy import text

sys.path.insert(0, '.')

from storage.database import Database


async def analyze_h3_h3b():
    """Analisa os eventos H3 e H3B para verificar se estão corretos."""
    
    db = Database()
    await db.connect()
    
    print("=" * 70)
    print("DEBUG H3 e H3B - Análise de Eventos")
    print("=" * 70)
    
    try:
        async with db.async_session() as session:
            
            # ============================================================
            # H3 - MONOTONICIDADE ENTRE LINHAS
            # ============================================================
            print("\n" + "=" * 70)
            print("H3 - QUEBRA DE MONOTONICIDADE ENTRE LINHAS ADJACENTES")
            print("=" * 70)
            print("""
Conceito: Verifica se odds entre linhas adjacentes estão na ordem correta.
Exemplo: Para HOME, linha -2.0 deveria ter odd MENOR que linha -1.0
         Se -2.0 tem odd MAIOR, é uma inversão (oportunidade).
""")
            
            # 1. Distribuição por side
            print("\n1. DISTRIBUIÇÃO POR SIDE")
            print("-" * 50)
            result = await session.execute(text("""
                SELECT side, COUNT(*) as cnt
                FROM h3_line_monotonicity_events
                GROUP BY side
                ORDER BY cnt DESC
            """))
            for row in result.fetchall():
                print(f"   {row[0]}: {row[1]:,} eventos")
            
            # 2. Distribuição por magnitude
            print("\n2. DISTRIBUIÇÃO POR MAGNITUDE (diferença de odds)")
            print("-" * 50)
            result = await session.execute(text("""
                SELECT 
                    CASE 
                        WHEN magnitude < 0.05 THEN '< 0.05'
                        WHEN magnitude < 0.10 THEN '0.05 - 0.10'
                        WHEN magnitude < 0.20 THEN '0.10 - 0.20'
                        WHEN magnitude < 0.50 THEN '0.20 - 0.50'
                        ELSE '>= 0.50'
                    END as range,
                    COUNT(*) as cnt,
                    ROUND(AVG(magnitude)::numeric, 3) as avg_mag
                FROM h3_line_monotonicity_events
                GROUP BY 1
                ORDER BY MIN(magnitude)
            """))
            for row in result.fetchall():
                print(f"   {row[0]:15}: {row[1]:,} eventos (média: {row[2]})")
            
            # 3. Exemplos de eventos H3
            print("\n3. EXEMPLOS DE EVENTOS H3 (últimos 10)")
            print("-" * 50)
            result = await session.execute(text("""
                SELECT 
                    h3.id, h3.match_id, h3.line_a, h3.line_b, h3.side,
                    h3.odd_line_a, h3.odd_line_b, 
                    h3.expected_relation, h3.actual_relation,
                    h3.magnitude, h3.detected_at
                FROM h3_line_monotonicity_events h3
                ORDER BY h3.detected_at DESC
                LIMIT 10
            """))
            print("   ID | Match | LineA | LineB | Side | OddA | OddB | Expected | Actual | Mag")
            for row in result.fetchall():
                print(f"   {row[0]:4} | {row[1]:5} | {row[2]:5} | {row[3]:5} | {row[4]:4} | "
                      f"{row[5]:.2f} | {row[6]:.2f} | {row[7]:8} | {row[8]:8} | {row[9]:.3f}")
            
            # 4. Verificação de duplicatas H3
            print("\n4. VERIFICAÇÃO DE DUPLICATAS H3")
            print("-" * 50)
            result = await session.execute(text("""
                SELECT match_id, line_a, line_b, side, COUNT(*) as cnt
                FROM h3_line_monotonicity_events
                GROUP BY match_id, line_a, line_b, side
                HAVING COUNT(*) > 5
                ORDER BY cnt DESC
                LIMIT 10
            """))
            rows = result.fetchall()
            if rows:
                print("   ⚠️ Combinações com muitos eventos repetidos:")
                for row in rows:
                    print(f"   Match {row[0]}, {row[1]}->{row[2]} ({row[3]}): {row[4]} eventos")
            else:
                print("   ✅ Sem duplicatas excessivas")
            
            # ============================================================
            # H3B - REVERSÕES TEMPORAIS
            # ============================================================
            print("\n" + "=" * 70)
            print("H3B - REVERSÕES TEMPORAIS DE ODDS")
            print("=" * 70)
            print("""
Conceito: Detecta quando uma odd muda de direção.
Exemplo: Odd estava SUBINDO (1.90 -> 1.95 -> 2.00) e começa a DESCER (2.00 -> 1.95)
         A reversão pode indicar informação nova no mercado.
""")
            
            # 1. Distribuição por direção
            print("\n1. DISTRIBUIÇÃO POR DIREÇÃO DA REVERSÃO")
            print("-" * 50)
            result = await session.execute(text("""
                SELECT 
                    direction_before || ' -> ' || direction_after as reversal,
                    COUNT(*) as cnt
                FROM h3b_temporal_reversal_events
                GROUP BY direction_before, direction_after
                ORDER BY cnt DESC
            """))
            for row in result.fetchall():
                print(f"   {row[0]}: {row[1]:,} eventos")
            
            # 2. Distribuição por market_type
            print("\n2. DISTRIBUIÇÃO POR TIPO DE MERCADO")
            print("-" * 50)
            result = await session.execute(text("""
                SELECT market_type, COUNT(*) as cnt
                FROM h3b_temporal_reversal_events
                GROUP BY market_type
                ORDER BY cnt DESC
            """))
            for row in result.fetchall():
                print(f"   {row[0]}: {row[1]:,} eventos")
            
            # 3. Distribuição por magnitude da reversão
            print("\n3. DISTRIBUIÇÃO POR MAGNITUDE DA REVERSÃO")
            print("-" * 50)
            result = await session.execute(text("""
                SELECT 
                    CASE 
                        WHEN reversal_magnitude < 0.02 THEN '< 0.02'
                        WHEN reversal_magnitude < 0.05 THEN '0.02 - 0.05'
                        WHEN reversal_magnitude < 0.10 THEN '0.05 - 0.10'
                        WHEN reversal_magnitude < 0.20 THEN '0.10 - 0.20'
                        ELSE '>= 0.20'
                    END as range,
                    COUNT(*) as cnt
                FROM h3b_temporal_reversal_events
                GROUP BY 1
                ORDER BY MIN(reversal_magnitude)
            """))
            for row in result.fetchall():
                print(f"   {row[0]:15}: {row[1]:,} eventos")
            
            # 4. Exemplos de eventos H3B
            print("\n4. EXEMPLOS DE EVENTOS H3B (últimos 10)")
            print("-" * 50)
            result = await session.execute(text("""
                SELECT 
                    id, match_id, market_type, ah_line, side,
                    direction_before, direction_after,
                    odd_before, odd_at_reversal, reversal_magnitude,
                    streak_before, num_reversals_1h
                FROM h3b_temporal_reversal_events
                ORDER BY detected_at DESC
                LIMIT 10
            """))
            print("   ID | Match | Market | Line | Side | Before->After | OddBef | OddRev | Streak | #Rev1h")
            for row in result.fetchall():
                direction = f"{row[5]}->{row[6]}"
                print(f"   {row[0]:4} | {row[1]:5} | {row[2]:6} | {row[3]:5} | {row[4]:5} | "
                      f"{direction:12} | {row[7]:.2f} | {row[8]:.2f} | {row[10]:6} | {row[11]}")
            
            # 5. Verificação de eventos por jogo/mercado
            print("\n5. JOGOS COM MAIS REVERSÕES")
            print("-" * 50)
            result = await session.execute(text("""
                SELECT match_id, market_type, COUNT(*) as cnt
                FROM h3b_temporal_reversal_events
                GROUP BY match_id, market_type
                HAVING COUNT(*) > 10
                ORDER BY cnt DESC
                LIMIT 10
            """))
            rows = result.fetchall()
            if rows:
                print("   Jogos/mercados com muitas reversões:")
                for row in rows:
                    print(f"   Match {row[0]} ({row[1]}): {row[2]} reversões")
            else:
                print("   Nenhum jogo com > 10 reversões")
            
            # ============================================================
            # COMPARAÇÃO
            # ============================================================
            print("\n" + "=" * 70)
            print("COMPARAÇÃO H3 vs H3B")
            print("=" * 70)
            
            result = await session.execute(text("""
                SELECT 
                    (SELECT COUNT(*) FROM h3_line_monotonicity_events) as h3_total,
                    (SELECT COUNT(DISTINCT match_id) FROM h3_line_monotonicity_events) as h3_matches,
                    (SELECT COUNT(*) FROM h3b_temporal_reversal_events) as h3b_total,
                    (SELECT COUNT(DISTINCT match_id) FROM h3b_temporal_reversal_events) as h3b_matches
            """))
            row = result.fetchone()
            print(f"""
   H3 (Monotonicidade entre linhas):
      - Total eventos: {row[0]:,}
      - Jogos distintos: {row[1]:,}
      - Média por jogo: {row[0]/row[1]:.1f}

   H3B (Reversões temporais):
      - Total eventos: {row[2]:,}
      - Jogos distintos: {row[3]:,}
      - Média por jogo: {row[2]/row[3]:.1f}
""")
            
            # Verificar se H3 está gerando eventos repetidos para mesma inversão
            print("\n6. ANÁLISE DE REPETIÇÃO H3")
            print("-" * 50)
            result = await session.execute(text("""
                WITH event_groups AS (
                    SELECT 
                        match_id, line_a, line_b, side,
                        COUNT(*) as event_count,
                        MIN(detected_at) as first_detected,
                        MAX(detected_at) as last_detected,
                        EXTRACT(EPOCH FROM (MAX(detected_at) - MIN(detected_at))) as duration_seconds
                    FROM h3_line_monotonicity_events
                    GROUP BY match_id, line_a, line_b, side
                )
                SELECT 
                    COUNT(*) as unique_inversions,
                    SUM(event_count) as total_events,
                    ROUND(AVG(event_count)::numeric, 1) as avg_events_per_inversion,
                    ROUND(AVG(duration_seconds)::numeric, 0) as avg_duration_seconds
                FROM event_groups
            """))
            row = result.fetchone()
            print(f"""
   Inversões únicas (match+lines+side): {row[0]:,}
   Total de eventos registrados: {row[1]:,}
   Média de eventos por inversão: {row[2]}
   Duração média da inversão: {row[3]}s
   
   NOTA: Se média > 1, a mesma inversão está sendo registrada múltiplas vezes
         (cada ciclo de coleta que encontra a inversão gera novo evento)
""")
            
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(analyze_h3_h3b())
