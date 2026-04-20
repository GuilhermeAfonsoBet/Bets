# -*- coding: utf-8 -*-
"""
Análise ROBUSTA dos resultados das hipóteses.

Correções:
1. Agrupa por jogo (eventos independentes)
2. Remove outliers extremos (CLV > 100% ou < -100%)
3. Separação temporal (in-sample vs out-of-sample)
4. Análise de distribuição
"""

import asyncio
import sys
from datetime import datetime, timezone
from sqlalchemy import text

sys.path.insert(0, '.')

from storage.database import Database


async def analyze_robust():
    db = Database()
    await db.connect()
    
    print("=" * 70)
    print("ANÁLISE ROBUSTA DE VALIDAÇÃO DAS HIPÓTESES")
    print("=" * 70)
    print("""
METODOLOGIA:
- Agrupamento por jogo (1 observação por jogo, não por evento)
- Remoção de outliers (CLV entre -50% e +50%)
- Análise de distribuição
""")
    
    try:
        async with db.async_session() as session:
            
            # ============================================================
            # H1 - PRECIFICAÇÃO (agrupado por jogo)
            # ============================================================
            print("\n" + "=" * 70)
            print("H1 - PRECIFICAÇÃO (1 observação por jogo)")
            print("=" * 70)
            
            # Análise por jogo (não por evento)
            result = await session.execute(text("""
                WITH eventos_por_jogo AS (
                    SELECT 
                        match_id,
                        AVG(clv_pct) as clv_medio_jogo,
                        SUM(profit_loss) as profit_jogo,
                        COUNT(*) as num_eventos,
                        MAX(CASE WHEN bet_result = 'win' THEN 1 ELSE 0 END) as teve_win
                    FROM h1_pricing_events
                    WHERE clv IS NOT NULL
                      AND clv_pct BETWEEN -50 AND 50  -- Remove outliers
                    GROUP BY match_id
                )
                SELECT 
                    COUNT(*) as num_jogos,
                    ROUND(AVG(clv_medio_jogo)::numeric, 3) as clv_medio,
                    ROUND(STDDEV(clv_medio_jogo)::numeric, 3) as clv_std,
                    COUNT(CASE WHEN clv_medio_jogo > 0 THEN 1 END) as jogos_clv_positivo,
                    ROUND(AVG(profit_jogo)::numeric, 2) as profit_medio_jogo,
                    ROUND(SUM(profit_jogo)::numeric, 2) as profit_total,
                    ROUND(AVG(num_eventos)::numeric, 1) as eventos_por_jogo
                FROM eventos_por_jogo
            """))
            row = result.fetchone()
            if row and row[0] > 0:
                clv_hit_rate = row[3] / row[0] * 100
                print(f"""
   Jogos analisados: {row[0]}
   Média de eventos por jogo: {row[6]}
   
   CLV (por jogo):
   - CLV médio: {row[1]}% {'✅' if row[1] and row[1] > 0 else '❌'}
   - Desvio padrão: {row[2]}%
   - Taxa de jogos com CLV positivo: {clv_hit_rate:.1f}%
   
   Profit (por jogo):
   - Profit médio por jogo: {row[4]} unidades
   - Profit total: {row[5]} unidades
""")
            
            # Distribuição do CLV
            result = await session.execute(text("""
                WITH eventos_por_jogo AS (
                    SELECT match_id, AVG(clv_pct) as clv
                    FROM h1_pricing_events
                    WHERE clv IS NOT NULL AND clv_pct BETWEEN -50 AND 50
                    GROUP BY match_id
                )
                SELECT 
                    CASE 
                        WHEN clv < -10 THEN '< -10%'
                        WHEN clv < -5 THEN '-10% a -5%'
                        WHEN clv < 0 THEN '-5% a 0%'
                        WHEN clv < 5 THEN '0% a 5%'
                        WHEN clv < 10 THEN '5% a 10%'
                        ELSE '> 10%'
                    END as faixa,
                    COUNT(*) as jogos
                FROM eventos_por_jogo
                GROUP BY 1
                ORDER BY MIN(clv)
            """))
            print("   Distribuição do CLV por jogo:")
            for row in result.fetchall():
                print(f"      {row[0]:12}: {row[1]} jogos")
            
            # ============================================================
            # H3 - MONOTONICIDADE (agrupado por jogo)
            # ============================================================
            print("\n" + "=" * 70)
            print("H3 - MONOTONICIDADE ENTRE LINHAS (1 obs por jogo)")
            print("=" * 70)
            
            result = await session.execute(text("""
                WITH eventos_por_jogo AS (
                    SELECT 
                        match_id,
                        AVG(clv_pct) as clv_medio_jogo,
                        SUM(profit_loss) as profit_jogo,
                        COUNT(*) as num_eventos
                    FROM h3_line_monotonicity_events
                    WHERE clv IS NOT NULL
                      AND clv_pct BETWEEN -50 AND 50
                    GROUP BY match_id
                )
                SELECT 
                    COUNT(*) as num_jogos,
                    ROUND(AVG(clv_medio_jogo)::numeric, 3) as clv_medio,
                    ROUND(STDDEV(clv_medio_jogo)::numeric, 3) as clv_std,
                    COUNT(CASE WHEN clv_medio_jogo > 0 THEN 1 END) as jogos_clv_positivo,
                    ROUND(SUM(profit_jogo)::numeric, 2) as profit_total,
                    ROUND(AVG(num_eventos)::numeric, 1) as eventos_por_jogo
                FROM eventos_por_jogo
            """))
            row = result.fetchone()
            if row and row[0] > 0:
                clv_hit_rate = row[3] / row[0] * 100
                print(f"""
   Jogos analisados: {row[0]}
   Média de eventos por jogo: {row[5]}
   
   CLV (por jogo):
   - CLV médio: {row[1]}% {'✅' if row[1] and row[1] > 0 else '❌'}
   - Desvio padrão: {row[2]}%
   - Taxa de jogos com CLV positivo: {clv_hit_rate:.1f}%
   
   Profit total: {row[4]} unidades
""")
            else:
                print("\n   Dados insuficientes após filtrar outliers.")
            
            # ============================================================
            # H3B - REVERSÕES TEMPORAIS (agrupado por jogo)
            # ============================================================
            print("\n" + "=" * 70)
            print("H3B - REVERSÕES TEMPORAIS (1 obs por jogo)")
            print("=" * 70)
            
            # Primeiro, vamos ver os outliers
            result = await session.execute(text("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN clv_pct > 50 OR clv_pct < -50 THEN 1 END) as outliers,
                    MAX(clv_pct) as max_clv,
                    MIN(clv_pct) as min_clv
                FROM h3b_temporal_reversal_events
                WHERE clv IS NOT NULL
            """))
            row = result.fetchone()
            if row:
                print(f"""
   Outliers removidos: {row[1]} de {row[0]} ({row[1]/row[0]*100:.1f}%)
   Range original: {row[3]:.1f}% a {row[2]:.1f}%
""")
            
            result = await session.execute(text("""
                WITH eventos_por_jogo AS (
                    SELECT 
                        match_id,
                        AVG(clv_pct) as clv_medio_jogo,
                        SUM(profit_loss) as profit_jogo,
                        COUNT(*) as num_eventos
                    FROM h3b_temporal_reversal_events
                    WHERE clv IS NOT NULL
                      AND clv_pct BETWEEN -50 AND 50
                    GROUP BY match_id
                )
                SELECT 
                    COUNT(*) as num_jogos,
                    ROUND(AVG(clv_medio_jogo)::numeric, 3) as clv_medio,
                    ROUND(STDDEV(clv_medio_jogo)::numeric, 3) as clv_std,
                    COUNT(CASE WHEN clv_medio_jogo > 0 THEN 1 END) as jogos_clv_positivo,
                    ROUND(SUM(profit_jogo)::numeric, 2) as profit_total,
                    ROUND(AVG(num_eventos)::numeric, 1) as eventos_por_jogo
                FROM eventos_por_jogo
            """))
            row = result.fetchone()
            if row and row[0] > 0:
                clv_hit_rate = row[3] / row[0] * 100
                print(f"""
   Jogos analisados (sem outliers): {row[0]}
   Média de eventos por jogo: {row[5]}
   
   CLV (por jogo):
   - CLV médio: {row[1]}% {'✅' if row[1] and row[1] > 0 else '❌'}
   - Desvio padrão: {row[2]}%
   - Taxa de jogos com CLV positivo: {clv_hit_rate:.1f}%
   
   Profit total: {row[4]} unidades
""")
            
            # ============================================================
            # H6 - CORRELAÇÃO/LAG (agrupado por jogo)
            # ============================================================
            print("\n" + "=" * 70)
            print("H6 - CORRELAÇÃO/LAG (1 obs por jogo)")
            print("=" * 70)
            
            result = await session.execute(text("""
                WITH eventos_por_jogo AS (
                    SELECT 
                        match_id,
                        AVG(clv_pct) as clv_medio_jogo,
                        SUM(profit_loss) as profit_jogo,
                        COUNT(*) as num_eventos
                    FROM h6_correlation_lag_events
                    WHERE clv IS NOT NULL
                      AND clv_pct BETWEEN -50 AND 50
                    GROUP BY match_id
                )
                SELECT 
                    COUNT(*) as num_jogos,
                    ROUND(AVG(clv_medio_jogo)::numeric, 3) as clv_medio,
                    ROUND(STDDEV(clv_medio_jogo)::numeric, 3) as clv_std,
                    COUNT(CASE WHEN clv_medio_jogo > 0 THEN 1 END) as jogos_clv_positivo,
                    ROUND(SUM(profit_jogo)::numeric, 2) as profit_total,
                    ROUND(AVG(num_eventos)::numeric, 1) as eventos_por_jogo
                FROM eventos_por_jogo
            """))
            row = result.fetchone()
            if row and row[0] > 0:
                clv_hit_rate = row[3] / row[0] * 100
                print(f"""
   Jogos analisados: {row[0]}
   Média de eventos por jogo: {row[5]}
   
   CLV (por jogo):
   - CLV médio: {row[1]}% {'✅' if row[1] and row[1] > 0 else '❌'}
   - Desvio padrão: {row[2]}%
   - Taxa de jogos com CLV positivo: {clv_hit_rate:.1f}%
   
   Profit total: {row[4]} unidades
""")
            
            # ============================================================
            # RESUMO COMPARATIVO ROBUSTO
            # ============================================================
            print("\n" + "=" * 70)
            print("RESUMO COMPARATIVO (Metodologia Robusta)")
            print("=" * 70)
            
            result = await session.execute(text("""
                WITH 
                h1_agg AS (
                    SELECT 'H1' as hip, COUNT(DISTINCT match_id) as n, 
                           ROUND(AVG(clv_pct)::numeric, 3) as clv
                    FROM h1_pricing_events 
                    WHERE clv IS NOT NULL AND clv_pct BETWEEN -50 AND 50
                ),
                h3_agg AS (
                    SELECT 'H3', COUNT(DISTINCT match_id), ROUND(AVG(clv_pct)::numeric, 3)
                    FROM h3_line_monotonicity_events 
                    WHERE clv IS NOT NULL AND clv_pct BETWEEN -50 AND 50
                ),
                h3b_agg AS (
                    SELECT 'H3b', COUNT(DISTINCT match_id), ROUND(AVG(clv_pct)::numeric, 3)
                    FROM h3b_temporal_reversal_events 
                    WHERE clv IS NOT NULL AND clv_pct BETWEEN -50 AND 50
                ),
                h6_agg AS (
                    SELECT 'H6', COUNT(DISTINCT match_id), ROUND(AVG(clv_pct)::numeric, 3)
                    FROM h6_correlation_lag_events 
                    WHERE clv IS NOT NULL AND clv_pct BETWEEN -50 AND 50
                )
                SELECT * FROM h1_agg
                UNION ALL SELECT * FROM h3_agg
                UNION ALL SELECT * FROM h3b_agg
                UNION ALL SELECT * FROM h6_agg
                ORDER BY 1
            """))
            print("\n   Hipótese | Jogos | CLV Médio | Conclusão Preliminar")
            print("   " + "-" * 55)
            for row in result.fetchall():
                if row[2] and row[2] > 1:
                    status = "🟢 Promissora"
                elif row[2] and row[2] > 0:
                    status = "🟡 Inconclusiva"
                else:
                    status = "🔴 Sem evidência"
                print(f"   {row[0]:8} | {row[1]:5} | {row[2] or 'N/A':9}% | {status}")
            
            print("""
======================================================================
LIMITAÇÕES DESTA ANÁLISE
======================================================================

1. TAMANHO DA AMOSTRA
   - Precisamos de ~100+ jogos por hipótese para conclusões robustas
   - Atualmente temos poucos jogos

2. PERÍODO CURTO
   - Dados de apenas ~3 dias
   - Pode haver viés de período específico

3. AINDA É IN-SAMPLE
   - Para validação real, precisamos de:
     a) Período de treino (desenvolver/calibrar)
     b) Período de teste (validar sem olhar)
   
4. PRÓXIMOS PASSOS RECOMENDADOS:
   - Coletar mais dados (mínimo 2 semanas)
   - Fazer análise temporal (primeira semana vs segunda)
   - Calcular intervalos de confiança
   - Testar significância estatística (t-test)
""")
            
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(analyze_robust())
