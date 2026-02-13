# -*- coding: utf-8 -*-
"""
Análise de CLV Relativo/Adicional.

Compara o CLV do evento de valor com o CLV "baseline" do mesmo mercado.
Isso remove a influência do resultado do jogo e isola o valor do detector.

CLV Adicional = CLV do evento - CLV baseline do mercado
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
    print("ANÁLISE DE CLV RELATIVO (Valor Adicionado pelo Detector)")
    print("=" * 70)
    print("""
METODOLOGIA:
- CLV baseline = CLV médio de apostas "aleatórias" no mesmo mercado
- CLV adicional = CLV do evento - CLV baseline
- Isso isola o VALOR DO DETECTOR, removendo efeito do jogo/mercado

Se CLV adicional > 0: O detector encontra momentos MELHORES que o aleatório
Se CLV adicional = 0: O detector não agrega valor (poderia apostar qualquer hora)
Se CLV adicional < 0: O detector encontra momentos PIORES
""")
    
    try:
        async with db.async_session() as session:
            
            # ============================================================
            # H1 - PRECIFICAÇÃO
            # ============================================================
            print("\n" + "=" * 70)
            print("H1 - PRECIFICAÇÃO")
            print("=" * 70)
            
            # Calcula CLV baseline por mercado e compara com CLV dos eventos
            result = await session.execute(text("""
                WITH 
                -- CLV baseline: média de odds do mercado vs closing
                baseline AS (
                    SELECT 
                        b.match_id,
                        b.ah_line,
                        -- Closing line (última odd antes do kickoff)
                        (SELECT best_home_odds FROM best_odds_history b2 
                         WHERE b2.match_id = b.match_id AND b2.ah_line = b.ah_line
                         AND b2.scraped_at < m.kickoff_time
                         ORDER BY scraped_at DESC LIMIT 1) as closing_home,
                        -- Média das odds coletadas
                        AVG(b.best_home_odds) as avg_home_odds,
                        COUNT(*) as num_coletas
                    FROM best_odds_history b
                    JOIN matches m ON b.match_id = m.id
                    WHERE m.status = 'finished'
                      AND b.ah_line NOT LIKE 'OU_%'
                      AND b.ah_line NOT IN ('1X2', '1X2_DRAW')
                    GROUP BY b.match_id, b.ah_line, m.kickoff_time
                    HAVING COUNT(*) >= 3  -- Mínimo de coletas para baseline
                ),
                baseline_clv AS (
                    SELECT 
                        match_id,
                        ah_line,
                        closing_home,
                        avg_home_odds,
                        CASE WHEN closing_home > 0 
                             THEN (avg_home_odds - closing_home) / closing_home * 100 
                             ELSE NULL END as clv_baseline
                    FROM baseline
                    WHERE closing_home > 0
                ),
                -- Eventos H1 com seus CLVs
                eventos AS (
                    SELECT 
                        e.match_id,
                        e.ah_line,
                        e.clv_pct as clv_evento,
                        e.recommended_odd,
                        b.clv_baseline,
                        e.clv_pct - b.clv_baseline as clv_adicional
                    FROM h1_pricing_events e
                    JOIN baseline_clv b ON e.match_id = b.match_id AND e.ah_line = b.ah_line
                    WHERE e.clv IS NOT NULL
                      AND e.clv_pct BETWEEN -50 AND 50
                )
                SELECT 
                    COUNT(*) as n_eventos,
                    COUNT(DISTINCT match_id) as n_jogos,
                    ROUND(AVG(clv_evento)::numeric, 3) as clv_evento_medio,
                    ROUND(AVG(clv_baseline)::numeric, 3) as clv_baseline_medio,
                    ROUND(AVG(clv_adicional)::numeric, 3) as clv_adicional_medio,
                    ROUND(STDDEV(clv_adicional)::numeric, 3) as clv_adicional_std,
                    COUNT(CASE WHEN clv_adicional > 0 THEN 1 END) as eventos_valor_positivo
                FROM eventos
            """))
            row = result.fetchone()
            if row and row[0] > 0:
                taxa_pos = row[6] / row[0] * 100
                status = "✅ AGREGA VALOR" if row[4] and row[4] > 0 else "❌ NÃO AGREGA"
                print(f"""
   Eventos analisados: {row[0]} ({row[1]} jogos)
   
   CLV do evento (média): {row[2]}%
   CLV baseline (média):  {row[3]}%
   ─────────────────────────────────
   CLV ADICIONAL:         {row[4]}% {status}
   Desvio padrão:         {row[5]}%
   Taxa eventos com CLV adicional > 0: {taxa_pos:.1f}%
""")
            else:
                print("\n   Dados insuficientes para análise.")
            
            # ============================================================
            # H3 - MONOTONICIDADE
            # ============================================================
            print("\n" + "=" * 70)
            print("H3 - MONOTONICIDADE ENTRE LINHAS")
            print("=" * 70)
            
            result = await session.execute(text("""
                WITH 
                baseline AS (
                    SELECT 
                        b.match_id,
                        b.ah_line,
                        (SELECT best_home_odds FROM best_odds_history b2 
                         WHERE b2.match_id = b.match_id AND b2.ah_line = b.ah_line
                         AND b2.scraped_at < m.kickoff_time
                         ORDER BY scraped_at DESC LIMIT 1) as closing_home,
                        AVG(b.best_home_odds) as avg_home_odds
                    FROM best_odds_history b
                    JOIN matches m ON b.match_id = m.id
                    WHERE m.status = 'finished'
                      AND b.ah_line NOT LIKE 'OU_%'
                      AND b.ah_line NOT IN ('1X2', '1X2_DRAW')
                    GROUP BY b.match_id, b.ah_line, m.kickoff_time
                    HAVING COUNT(*) >= 3
                ),
                baseline_clv AS (
                    SELECT 
                        match_id, ah_line,
                        CASE WHEN closing_home > 0 
                             THEN (avg_home_odds - closing_home) / closing_home * 100 
                             ELSE NULL END as clv_baseline
                    FROM baseline WHERE closing_home > 0
                ),
                eventos AS (
                    SELECT 
                        e.match_id,
                        e.clv_pct as clv_evento,
                        b.clv_baseline,
                        e.clv_pct - b.clv_baseline as clv_adicional
                    FROM h3_line_monotonicity_events e
                    JOIN baseline_clv b ON e.match_id = b.match_id 
                        AND e.recommended_line = b.ah_line
                    WHERE e.clv IS NOT NULL AND e.clv_pct BETWEEN -50 AND 50
                )
                SELECT 
                    COUNT(*) as n,
                    ROUND(AVG(clv_evento)::numeric, 3),
                    ROUND(AVG(clv_baseline)::numeric, 3),
                    ROUND(AVG(clv_adicional)::numeric, 3),
                    COUNT(CASE WHEN clv_adicional > 0 THEN 1 END)
                FROM eventos
            """))
            row = result.fetchone()
            if row and row[0] > 0:
                taxa_pos = row[4] / row[0] * 100
                status = "✅ AGREGA VALOR" if row[3] and row[3] > 0 else "❌ NÃO AGREGA"
                print(f"""
   Eventos analisados: {row[0]}
   
   CLV do evento (média): {row[1]}%
   CLV baseline (média):  {row[2]}%
   ─────────────────────────────────
   CLV ADICIONAL:         {row[3]}% {status}
   Taxa CLV adicional > 0: {taxa_pos:.1f}%
""")
            else:
                print("\n   Dados insuficientes para análise.")
            
            # ============================================================
            # H3B - REVERSÕES (análise simplificada por mercado)
            # ============================================================
            print("\n" + "=" * 70)
            print("H3B - REVERSÕES TEMPORAIS")
            print("=" * 70)
            
            result = await session.execute(text("""
                WITH 
                baseline AS (
                    SELECT 
                        b.match_id,
                        b.ah_line,
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
                        match_id, ah_line,
                        CASE WHEN closing > 0 
                             THEN (avg_odds - closing) / closing * 100 
                             ELSE NULL END as clv_baseline
                    FROM baseline WHERE closing > 0
                ),
                eventos AS (
                    SELECT 
                        e.match_id,
                        e.clv_pct as clv_evento,
                        b.clv_baseline,
                        e.clv_pct - b.clv_baseline as clv_adicional
                    FROM h3b_temporal_reversal_events e
                    JOIN baseline_clv b ON e.match_id = b.match_id AND e.ah_line = b.ah_line
                    WHERE e.clv IS NOT NULL AND e.clv_pct BETWEEN -50 AND 50
                )
                SELECT 
                    COUNT(*) as n,
                    ROUND(AVG(clv_evento)::numeric, 3),
                    ROUND(AVG(clv_baseline)::numeric, 3),
                    ROUND(AVG(clv_adicional)::numeric, 3),
                    COUNT(CASE WHEN clv_adicional > 0 THEN 1 END)
                FROM eventos
            """))
            row = result.fetchone()
            if row and row[0] > 0:
                taxa_pos = row[4] / row[0] * 100
                status = "✅ AGREGA VALOR" if row[3] and row[3] > 0 else "❌ NÃO AGREGA"
                print(f"""
   Eventos analisados: {row[0]}
   
   CLV do evento (média): {row[1]}%
   CLV baseline (média):  {row[2]}%
   ─────────────────────────────────
   CLV ADICIONAL:         {row[3]}% {status}
   Taxa CLV adicional > 0: {taxa_pos:.1f}%
""")
            else:
                print("\n   Dados insuficientes para análise.")
            
            # ============================================================
            # H6 - CORRELAÇÃO/LAG
            # ============================================================
            print("\n" + "=" * 70)
            print("H6 - CORRELAÇÃO/LAG")
            print("=" * 70)
            
            result = await session.execute(text("""
                WITH 
                baseline AS (
                    SELECT 
                        b.match_id,
                        b.ah_line,
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
                        match_id, ah_line,
                        CASE WHEN closing > 0 
                             THEN (avg_odds - closing) / closing * 100 
                             ELSE NULL END as clv_baseline
                    FROM baseline WHERE closing > 0
                ),
                eventos AS (
                    SELECT 
                        e.match_id,
                        e.clv_pct as clv_evento,
                        b.clv_baseline,
                        e.clv_pct - b.clv_baseline as clv_adicional
                    FROM h6_correlation_lag_events e
                    JOIN baseline_clv b ON e.match_id = b.match_id AND e.lagged_line = b.ah_line
                    WHERE e.clv IS NOT NULL AND e.clv_pct BETWEEN -50 AND 50
                )
                SELECT 
                    COUNT(*) as n,
                    ROUND(AVG(clv_evento)::numeric, 3),
                    ROUND(AVG(clv_baseline)::numeric, 3),
                    ROUND(AVG(clv_adicional)::numeric, 3),
                    COUNT(CASE WHEN clv_adicional > 0 THEN 1 END)
                FROM eventos
            """))
            row = result.fetchone()
            if row and row[0] > 0:
                taxa_pos = row[4] / row[0] * 100
                status = "✅ AGREGA VALOR" if row[3] and row[3] > 0 else "❌ NÃO AGREGA"
                print(f"""
   Eventos analisados: {row[0]}
   
   CLV do evento (média): {row[1]}%
   CLV baseline (média):  {row[2]}%
   ─────────────────────────────────
   CLV ADICIONAL:         {row[3]}% {status}
   Taxa CLV adicional > 0: {taxa_pos:.1f}%
""")
            else:
                print("\n   Dados insuficientes para análise.")
            
            # ============================================================
            # INTERPRETAÇÃO
            # ============================================================
            print("\n" + "=" * 70)
            print("INTERPRETAÇÃO")
            print("=" * 70)
            print("""
CLV ADICIONAL = valor que o DETECTOR agrega além do aleatório

Se CLV adicional > 0:
   → O detector identifica MOMENTOS melhores para apostar
   → Isso é independente do resultado do jogo
   → Indica que a hipótese tem mérito

Se CLV adicional ≈ 0:
   → O detector não encontra momentos melhores
   → Apostar quando detecta = apostar aleatoriamente
   → Hipótese provavelmente não tem valor

Se CLV adicional < 0:
   → O detector encontra momentos PIORES
   → Melhor fazer o OPOSTO do que ele sugere
   → Ou revisar a lógica do detector

VANTAGEM DESTA MÉTRICA:
- Remove influência do resultado do jogo
- Remove viés da linha/mercado específico
- Isola puramente o valor do timing/detecção
""")
            
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(analyze_relative_clv())
