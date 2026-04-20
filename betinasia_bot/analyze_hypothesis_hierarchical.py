# -*- coding: utf-8 -*-
"""
Análise Hierárquica das Hipóteses.

Compara diferentes níveis de agregação:
1. Por evento (todos os eventos)
2. Por jogo (1 observação por jogo)
3. Melhor evento por jogo (estratégia conservadora)
4. Primeiro evento por jogo (sinal inicial)
"""

import asyncio
import sys
from sqlalchemy import text

sys.path.insert(0, '.')

from storage.database import Database


async def analyze_hierarchical():
    db = Database()
    await db.connect()
    
    print("=" * 70)
    print("ANÁLISE HIERÁRQUICA: EVENTO vs JOGO")
    print("=" * 70)
    print("""
PERGUNTA: A estrutura dos dados afeta as conclusões?

Níveis de análise:
1. EVENTO: Cada detecção é uma observação
2. JOGO (média): Média dos eventos por jogo
3. JOGO (melhor): Evento com maior CLV por jogo
4. JOGO (primeiro): Primeiro evento detectado por jogo
""")
    
    try:
        async with db.async_session() as session:
            
            for hipotese, tabela in [
                ("H1 - Precificação", "h1_pricing_events"),
                ("H3 - Monotonicidade", "h3_line_monotonicity_events"),
                ("H3B - Reversões", "h3b_temporal_reversal_events"),
                ("H6 - Correlação/Lag", "h6_correlation_lag_events"),
            ]:
                print("\n" + "=" * 70)
                print(f"{hipotese}")
                print("=" * 70)
                
                # 1. Por EVENTO (análise tradicional)
                result = await session.execute(text(f"""
                    SELECT 
                        COUNT(*) as n,
                        ROUND(AVG(clv_pct)::numeric, 3) as clv_medio,
                        ROUND(STDDEV(clv_pct)::numeric, 3) as clv_std,
                        COUNT(CASE WHEN clv_pct > 0 THEN 1 END) as positivos,
                        ROUND(SUM(profit_loss)::numeric, 2) as profit
                    FROM {tabela}
                    WHERE clv IS NOT NULL
                      AND clv_pct BETWEEN -50 AND 50
                """))
                row = result.fetchone()
                if row and row[0] > 0:
                    taxa_pos = row[3] / row[0] * 100
                    print(f"""
   1. POR EVENTO (n={row[0]}):
      CLV médio: {row[1]}%  |  Std: {row[2]}%  |  CLV>0: {taxa_pos:.1f}%
      Profit: {row[4]} unidades
""")
                
                # 2. Por JOGO (média)
                result = await session.execute(text(f"""
                    WITH por_jogo AS (
                        SELECT 
                            match_id,
                            AVG(clv_pct) as clv_jogo,
                            SUM(profit_loss) as profit_jogo
                        FROM {tabela}
                        WHERE clv IS NOT NULL AND clv_pct BETWEEN -50 AND 50
                        GROUP BY match_id
                    )
                    SELECT 
                        COUNT(*) as n_jogos,
                        ROUND(AVG(clv_jogo)::numeric, 3) as clv_medio,
                        ROUND(STDDEV(clv_jogo)::numeric, 3) as clv_std,
                        COUNT(CASE WHEN clv_jogo > 0 THEN 1 END) as positivos,
                        ROUND(SUM(profit_jogo)::numeric, 2) as profit
                    FROM por_jogo
                """))
                row = result.fetchone()
                if row and row[0] > 0:
                    taxa_pos = row[3] / row[0] * 100
                    print(f"""   2. POR JOGO - média (n={row[0]} jogos):
      CLV médio: {row[1]}%  |  Std: {row[2]}%  |  CLV>0: {taxa_pos:.1f}%
      Profit: {row[4]} unidades
""")
                
                # 3. Por JOGO (melhor evento)
                result = await session.execute(text(f"""
                    WITH melhor_por_jogo AS (
                        SELECT DISTINCT ON (match_id)
                            match_id,
                            clv_pct,
                            profit_loss
                        FROM {tabela}
                        WHERE clv IS NOT NULL AND clv_pct BETWEEN -50 AND 50
                        ORDER BY match_id, clv_pct DESC
                    )
                    SELECT 
                        COUNT(*) as n_jogos,
                        ROUND(AVG(clv_pct)::numeric, 3) as clv_medio,
                        ROUND(STDDEV(clv_pct)::numeric, 3) as clv_std,
                        COUNT(CASE WHEN clv_pct > 0 THEN 1 END) as positivos,
                        ROUND(SUM(profit_loss)::numeric, 2) as profit
                    FROM melhor_por_jogo
                """))
                row = result.fetchone()
                if row and row[0] > 0:
                    taxa_pos = row[3] / row[0] * 100
                    print(f"""   3. POR JOGO - melhor evento (n={row[0]} jogos):
      CLV médio: {row[1]}%  |  Std: {row[2]}%  |  CLV>0: {taxa_pos:.1f}%
      Profit: {row[4]} unidades
""")
                
                # 4. Por JOGO (primeiro evento)
                result = await session.execute(text(f"""
                    WITH primeiro_por_jogo AS (
                        SELECT DISTINCT ON (match_id)
                            match_id,
                            clv_pct,
                            profit_loss
                        FROM {tabela}
                        WHERE clv IS NOT NULL AND clv_pct BETWEEN -50 AND 50
                        ORDER BY match_id, detected_at ASC
                    )
                    SELECT 
                        COUNT(*) as n_jogos,
                        ROUND(AVG(clv_pct)::numeric, 3) as clv_medio,
                        ROUND(STDDEV(clv_pct)::numeric, 3) as clv_std,
                        COUNT(CASE WHEN clv_pct > 0 THEN 1 END) as positivos,
                        ROUND(SUM(profit_loss)::numeric, 2) as profit
                    FROM primeiro_por_jogo
                """))
                row = result.fetchone()
                if row and row[0] > 0:
                    taxa_pos = row[3] / row[0] * 100
                    print(f"""   4. POR JOGO - primeiro evento (n={row[0]} jogos):
      CLV médio: {row[1]}%  |  Std: {row[2]}%  |  CLV>0: {taxa_pos:.1f}%
      Profit: {row[4]} unidades
""")
            
            # ============================================================
            # COMPARATIVO FINAL
            # ============================================================
            print("\n" + "=" * 70)
            print("RESUMO: CONCLUSÃO MUDA COM AGREGAÇÃO?")
            print("=" * 70)
            
            comparativo = []
            for hip, tabela in [
                ("H1", "h1_pricing_events"),
                ("H3", "h3_line_monotonicity_events"),
                ("H3b", "h3b_temporal_reversal_events"),
                ("H6", "h6_correlation_lag_events"),
            ]:
                # CLV por evento
                result = await session.execute(text(f"""
                    SELECT ROUND(AVG(clv_pct)::numeric, 3)
                    FROM {tabela}
                    WHERE clv IS NOT NULL AND clv_pct BETWEEN -50 AND 50
                """))
                clv_evento = result.scalar()
                
                # CLV por jogo
                result = await session.execute(text(f"""
                    SELECT ROUND(AVG(clv_jogo)::numeric, 3)
                    FROM (
                        SELECT match_id, AVG(clv_pct) as clv_jogo
                        FROM {tabela}
                        WHERE clv IS NOT NULL AND clv_pct BETWEEN -50 AND 50
                        GROUP BY match_id
                    ) t
                """))
                clv_jogo = result.scalar()
                
                comparativo.append((hip, clv_evento, clv_jogo))
            
            print("\n   Hipótese | CLV/Evento | CLV/Jogo | Diferença | Consistente?")
            print("   " + "-" * 60)
            for hip, clv_e, clv_j in comparativo:
                diff = (clv_e or 0) - (clv_j or 0)
                # Consistente se ambos têm mesmo sinal ou ambos próximos de zero
                if clv_e and clv_j:
                    consistente = "✅" if (clv_e > 0) == (clv_j > 0) else "⚠️ DIFERE"
                else:
                    consistente = "N/A"
                print(f"   {hip:8} | {clv_e or 'N/A':10} | {clv_j or 'N/A':8} | {diff:+.3f}%   | {consistente}")
            
            print("""
======================================================================
INTERPRETAÇÃO
======================================================================

Se CLV/Evento ≈ CLV/Jogo:
   → Conclusão é robusta à agregação
   → Podemos confiar na análise por evento

Se CLV/Evento ≠ CLV/Jogo:
   → Jogos com muitos eventos estão influenciando
   → Usar análise por JOGO para decisões

RECOMENDAÇÃO:
- Para VALIDAÇÃO ESTATÍSTICA: usar agregação por jogo
- Para ESTRATÉGIA DE APOSTAS: depende de quantas apostas quer fazer
  - Conservador: 1 aposta por jogo (melhor evento)
  - Agressivo: todas as apostas em todos os eventos
""")
            
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(analyze_hierarchical())
