#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análise H3B REALISTA: CLV com odd do Betslip vs WebSocket

Mesma metodologia da v6, lado a lado:
  CLV_websocket = (odd_websocket - closing_odd) / closing_odd × 100
  CLV_betslip   = (odd_betslip   - closing_odd) / closing_odd × 100

Uso:
    python analyze_h3b_websocket_vs_betslip.py
"""

import asyncio
import sys
import math

sys.path.insert(0, '.')

from sqlalchemy import text
from storage.database import Database

Z_90 = 1.645
Z_95 = 1.960


def format_result(label: str, n, mean, std):
    """Formata resultado com IC."""
    if not n or n <= 1 or mean is None or std is None:
        print(f"\n   {label}: N={n or 0} (insuficiente)")
        return
    
    n, mean, std = int(n), float(mean), float(std)
    se = std / math.sqrt(n)
    ci90_low = mean - Z_90 * se
    ci90_high = mean + Z_90 * se
    ci95_low = mean - Z_95 * se
    ci95_high = mean + Z_95 * se
    
    if ci90_low > 0:
        sig = "✅ SIGNIFICATIVO (p<0.10)"
    elif ci90_high < 0:
        sig = "❌ SIGNIFICATIVO NEGATIVO"
    else:
        sig = "⚪ Não significativo"
    
    print(f"""
   {label}:

   N = {n}
   CLV adicional = {mean:.3f}%
   Erro padrão   = {se:.3f}%
   IC 90%        = [{ci90_low:.3f}%, {ci90_high:.3f}%]
   IC 95%        = [{ci95_low:.3f}%, {ci95_high:.3f}%]
   {sig}
""")
    
    if mean > 0 and ci90_low <= 0 and std > 0:
        n_needed = math.ceil((Z_90 * std / mean) ** 2)
        print(f"   📊 N estimado p/ significância (IC 90%): ~{n_needed}")


# ============================================================
# Query H3B WebSocket (referência v6 - idêntica)
# ============================================================
QUERY_WEBSOCKET = """
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
          AND e.direction_after = :direction
    )
    SELECT 
        COUNT(*) as n,
        AVG(clv_evento - COALESCE(clv_baseline, 0)) as clv_adicional,
        STDDEV(clv_evento - COALESCE(clv_baseline, 0)) as clv_adicional_std
    FROM evento_com_baseline
    WHERE clv_baseline IS NOT NULL
"""


# ============================================================
# Query H3B Betslip (CLV com odd real do betslip)
# ============================================================
# Normaliza formato da linha: audit salva "0","-1","3"
# mas best_odds_history salva "0.0","-1.0","3.0"
QUERY_BETSLIP = """
    WITH audit_closing AS (
        SELECT 
            a.id,
            a.betslip_odd,
            a.websocket_odd,
            a.is_live,
            a.audit_total_duration_ms,
            a.side as audit_side,
            -- Closing odd: último registro antes do kickoff
            (
                SELECT CASE WHEN a.side = 'home' THEN boh.best_home_odds
                            ELSE boh.best_away_odds END
                FROM best_odds_history boh
                WHERE boh.match_id = m.id
                  AND (
                      boh.ah_line = a.line 
                      OR boh.ah_line = a.line || '.0'
                      OR boh.ah_line = CASE 
                          WHEN a.line NOT LIKE '+%%' AND a.line NOT LIKE '-%%' 
                          THEN '+' || a.line ELSE a.line END
                      OR boh.ah_line = CASE 
                          WHEN a.line NOT LIKE '+%%' AND a.line NOT LIKE '-%%' 
                          THEN '+' || a.line || '.0' ELSE a.line || '.0' END
                  )
                  AND boh.scraped_at < m.kickoff_time
                ORDER BY boh.scraped_at DESC
                LIMIT 1
            ) as closing_odd,
            -- CLV do websocket (para comparação lado a lado)
            (
                SELECT CASE WHEN a.side = 'home' 
                       THEN (a.websocket_odd - boh2.best_home_odds) / boh2.best_home_odds * 100
                       ELSE (a.websocket_odd - boh2.best_away_odds) / boh2.best_away_odds * 100
                       END
                FROM best_odds_history boh2
                WHERE boh2.match_id = m.id
                  AND (
                      boh2.ah_line = a.line 
                      OR boh2.ah_line = a.line || '.0'
                      OR boh2.ah_line = CASE 
                          WHEN a.line NOT LIKE '+%%' AND a.line NOT LIKE '-%%' 
                          THEN '+' || a.line ELSE a.line END
                      OR boh2.ah_line = CASE 
                          WHEN a.line NOT LIKE '+%%' AND a.line NOT LIKE '-%%' 
                          THEN '+' || a.line || '.0' ELSE a.line || '.0' END
                  )
                  AND boh2.scraped_at < m.kickoff_time
                  AND boh2.best_home_odds > 0 AND boh2.best_away_odds > 0
                ORDER BY boh2.scraped_at DESC
                LIMIT 1
            ) as clv_websocket
        FROM betslip_audit_results a
        JOIN matches m ON m.external_id = a.event_id
        WHERE a.betslip_odd IS NOT NULL
          AND a.hypothesis_type = 'H3B'
          AND a.reversal_direction = :direction
          AND m.kickoff_time < NOW()
    )
    SELECT 
        COUNT(*) as n,
        AVG((betslip_odd - closing_odd) / closing_odd * 100) as clv_betslip,
        STDDEV((betslip_odd - closing_odd) / closing_odd * 100) as clv_betslip_std,
        AVG(clv_websocket) as clv_ws_medio,
        AVG(betslip_odd) as avg_betslip,
        AVG(websocket_odd) as avg_ws,
        AVG(closing_odd) as avg_closing
    FROM audit_closing
    WHERE closing_odd > 0
      AND (betslip_odd - closing_odd) / closing_odd * 100 BETWEEN -50 AND 50
"""


async def main():
    db = Database()
    await db.connect()
    
    print("=" * 70)
    print("ANÁLISE H3B REALISTA: CLV COM ODD DO BETSLIP")
    print("=" * 70)
    print("""
Metodologia IDÊNTICA à análise v6, mas com a odd REAL do betslip:

  CLV = (odd - closing_odd) / closing_odd × 100

  WebSocket: odd = odd do WebSocket no momento da detecção
  Betslip:   odd = odd real disponível no betslip (a que seria apostada)

Nota: só inclui jogos que já tiveram kickoff (closing line disponível).
""")
    
    try:
        async with db.async_session() as session:
            
            # ============================================================
            # PARTE 1: WebSocket (referência v6)
            # ============================================================
            print("=" * 70)
            print("PARTE 1: CLV WebSocket (referência v6)")
            print("=" * 70)
            
            for direcao, label in [('up', 'REVERSÃO UP'), ('down', 'REVERSÃO DOWN')]:
                result = await session.execute(
                    text(QUERY_WEBSOCKET), {"direction": direcao}
                )
                row = result.fetchone()
                if row:
                    format_result(f"WEBSOCKET - {label}", row[0], row[1], row[2])
            
            # ============================================================
            # PARTE 2: Betslip (realista)
            # ============================================================
            print("\n" + "=" * 70)
            print("PARTE 2: CLV Betslip (realista)")
            print("=" * 70)
            
            for direcao, label in [('up', 'REVERSÃO UP'), ('down', 'REVERSÃO DOWN')]:
                result = await session.execute(
                    text(QUERY_BETSLIP), {"direction": direcao}
                )
                row = result.fetchone()
                
                if row and row[0] and row[0] > 0:
                    format_result(f"BETSLIP - {label}", row[0], row[1], row[2])
                    
                    # Detalhes
                    print(f"   Detalhes:")
                    print(f"     CLV médio WebSocket (mesmos eventos): {row[3]:.3f}%" if row[3] else "")
                    print(f"     Odd média betslip: {row[4]:.3f}" if row[4] else "")
                    print(f"     Odd média websocket: {row[5]:.3f}" if row[5] else "")
                    print(f"     Closing odd média: {row[6]:.3f}" if row[6] else "")
                else:
                    print(f"\n   BETSLIP - {label}: N={row[0] if row else 0}")
                    print(f"   Sem dados. Jogos ainda não tiveram kickoff ou formato de linha sem match.")
            
            # ============================================================
            # VISÃO GERAL
            # ============================================================
            print("\n" + "=" * 70)
            print("VISÃO GERAL DOS DADOS")
            print("=" * 70)
            
            result = await session.execute(text("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN betslip_odd IS NOT NULL THEN 1 ELSE 0 END) as com_betslip,
                    SUM(CASE WHEN is_live = true THEN 1 ELSE 0 END) as in_match,
                    SUM(CASE WHEN is_live = false THEN 1 ELSE 0 END) as pre_match
                FROM betslip_audit_results
                WHERE hypothesis_type = 'H3B'
            """))
            row = result.fetchone()
            if row:
                print(f"\n  Total auditorias: {row[0]}")
                print(f"  Com betslip: {row[1]}")
                print(f"  Pre-match: {row[2] or 0}")
                print(f"  In-match: {row[3] or 0}")
            
            # Quantos jogos já tiveram kickoff?
            result = await session.execute(text("""
                SELECT 
                    COUNT(*) as total_matched,
                    SUM(CASE WHEN m.kickoff_time < NOW() THEN 1 ELSE 0 END) as kicked
                FROM betslip_audit_results a
                JOIN matches m ON m.external_id = a.event_id
                WHERE a.betslip_odd IS NOT NULL AND a.hypothesis_type = 'H3B'
            """))
            row = result.fetchone()
            if row:
                print(f"\n  Auditorias com match no banco: {row[0]}")
                print(f"  Destes, com kickoff passado: {row[1]}")
                if row[1] and row[1] < 20:
                    print(f"\n  ⚠️  Poucos jogos com closing line disponível.")
                    print(f"  Continue coletando dados - a análise fica mais robusta com o tempo.")
            
            # Verifica formato de linhas
            result = await session.execute(text("""
                SELECT a.line, COUNT(*) as n,
                       SUM(CASE WHEN EXISTS(
                           SELECT 1 FROM best_odds_history boh 
                           WHERE boh.match_id = m.id 
                             AND (boh.ah_line = a.line OR boh.ah_line = a.line || '.0')
                       ) THEN 1 ELSE 0 END) as com_match_boh
                FROM betslip_audit_results a
                JOIN matches m ON m.external_id = a.event_id
                WHERE a.betslip_odd IS NOT NULL AND m.kickoff_time < NOW()
                GROUP BY a.line
                ORDER BY n DESC
            """))
            rows = result.fetchall()
            if rows:
                print(f"\n  Linhas auditadas (com kickoff passado):")
                for row in rows:
                    print(f"    AH {row[0]}: {row[1]} auditorias, {row[2]} com closing line")
    
    finally:
        await db.close()
    
    print("\n" + "=" * 70)
    print("FIM DA ANÁLISE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
