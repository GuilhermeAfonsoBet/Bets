#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análise H3B REALISTA: CLV com odd do Betslip (execução real)

Compara:
  - CLV WebSocket: (odd_websocket - closing_odd) / closing_odd
  - CLV Betslip:   (odd_betslip   - closing_odd) / closing_odd

Se o valor sobrevive na prática, o CLV Betslip deve ser significativo
e positivo, com a mesma metodologia da análise v6.

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


def calc_ci(mean: float, std: float, n: int, z: float = 1.645) -> tuple:
    """Calcula intervalo de confiança."""
    if n <= 1:
        return (None, None)
    se = std / math.sqrt(n)
    return (mean - z * se, mean + z * se)


def format_result(label: str, n: int, mean: float, std: float):
    """Formata resultado com IC."""
    if n <= 1:
        print(f"\n   {label}: N={n} (insuficiente)")
        return
    
    se = std / math.sqrt(n)
    ci90_low, ci90_high = calc_ci(mean, std, n, Z_90)
    ci95_low, ci95_high = calc_ci(mean, std, n, Z_95)
    
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
    
    # Estimativa de N para significância (se positivo mas não significativo)
    if mean > 0 and ci90_low <= 0 and std > 0:
        n_needed = math.ceil((Z_90 * std / mean) ** 2)
        print(f"   📊 N estimado p/ significância (IC 90%): ~{n_needed}")


async def main():
    db = Database()
    await db.connect()
    
    print("=" * 70)
    print("ANÁLISE H3B REALISTA: CLV COM ODD DO BETSLIP")
    print("=" * 70)
    print("""
Metodologia IDÊNTICA à análise v6, mas com a odd REAL do betslip:

  CLV_websocket = (odd_websocket - closing_odd) / closing_odd × 100
  CLV_betslip   = (odd_betslip   - closing_odd) / closing_odd × 100

  CLV adicional = CLV do evento - CLV baseline (outras linhas)

Assim comparamos DIRETAMENTE se o valor existe quando apostamos
na odd real disponível no betslip, não na odd do WebSocket.
""")
    
    try:
        async with db.async_session() as session:
            
            # ============================================================
            # PARTE 1: Análise com ODD DO WEBSOCKET (referência)
            # Exatamente como v6, para comparação lado a lado
            # ============================================================
            print("=" * 70)
            print("PARTE 1: CLV COM ODD DO WEBSOCKET (referência v6)")
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
                if row and row[0] > 0:
                    format_result(f"WEBSOCKET - {label}", row[0], row[1], row[2])
            
            # ============================================================
            # PARTE 2: Análise com ODD DO BETSLIP (realista)
            # Substitui odd_at_reversal pela betslip_odd
            # ============================================================
            print("\n" + "=" * 70)
            print("PARTE 2: CLV COM ODD DO BETSLIP (realista)")
            print("=" * 70)
            
            for direcao, label in [('up', 'REVERSÃO UP'), ('down', 'REVERSÃO DOWN')]:
                # O CLV do betslip = (betslip_odd - closing_odd) / closing_odd × 100
                # 
                # Sabemos que:
                #   clv_pct (do evento) = (odd_at_reversal - closing_odd) / closing_odd × 100
                #   closing_odd = odd_at_reversal / (1 + clv_pct/100)
                #
                # Então:
                #   clv_betslip = (betslip_odd / closing_odd - 1) × 100
                #               = (betslip_odd × (1 + clv_pct/100) / odd_at_reversal - 1) × 100
                #
                # Onde: odd_at_reversal = e.odd_at_reversal (websocket)
                #       betslip_odd = a.betslip_odd (real)
                #       clv_pct = e.clv_pct (do evento original)
                
                result = await session.execute(text(f"""
                    WITH audit_com_clv AS (
                        SELECT 
                            a.id as audit_id,
                            a.websocket_odd,
                            a.betslip_odd,
                            a.is_live,
                            a.audit_total_duration_ms,
                            e.id as event_id,
                            e.match_id,
                            e.ah_line,
                            e.clv_pct as clv_websocket,
                            e.closing_odd,
                            e.odd_at_reversal,
                            -- CLV calculado com a odd do BETSLIP
                            CASE WHEN e.closing_odd > 0 
                                THEN (a.betslip_odd - e.closing_odd) / e.closing_odd * 100
                                ELSE NULL 
                            END as clv_betslip
                        FROM betslip_audit_results a
                        JOIN h3b_temporal_reversal_events e 
                            ON a.event_id = (
                                e.match_id::text || '_' || e.ah_line || '_' || e.side
                            )
                        WHERE a.betslip_odd IS NOT NULL
                          AND a.hypothesis_type = 'H3B'
                          AND a.reversal_direction = '{direcao}'
                          AND e.clv_pct IS NOT NULL
                          AND e.closing_odd > 0
                          AND e.direction_after = '{direcao}'
                    ),
                    -- Também calcula via fórmula derivada (para eventos sem join direto)
                    audit_derivado AS (
                        SELECT 
                            a.id as audit_id,
                            a.websocket_odd,
                            a.betslip_odd,
                            a.is_live,
                            a.audit_total_duration_ms,
                            -- Closing odd derivada: ws_odd / (1 + clv_ws/100)
                            -- Mas precisamos do clv_pct do evento original
                            -- Se não temos o join, usamos a diferença direta
                            a.difference_pct
                        FROM betslip_audit_results a
                        WHERE a.betslip_odd IS NOT NULL
                          AND a.hypothesis_type = 'H3B'
                          AND a.reversal_direction = '{direcao}'
                    )
                    SELECT 
                        COUNT(*) as n,
                        AVG(clv_betslip) as clv_medio,
                        STDDEV(clv_betslip) as clv_std
                    FROM audit_com_clv
                    WHERE clv_betslip BETWEEN -50 AND 50
                """))
                row = result.fetchone()
                
                if row and row[0] and row[0] > 0:
                    format_result(f"BETSLIP - {label} (join direto)", row[0], row[1], row[2])
                else:
                    print(f"\n   BETSLIP - {label} (join direto): N=0")
                    print(f"   Join por event_id falhou - tentando método alternativo...")
                    
                    # Método alternativo: usa event_id do audit para buscar no h3b
                    # O event_id no audit é tipo "2026-02-05,25788,26061"
                    # Precisa fazer match por match external_id, line e side
                    result2 = await session.execute(text(f"""
                        WITH audit_matched AS (
                            SELECT 
                                a.id,
                                a.websocket_odd,
                                a.betslip_odd,
                                a.is_live,
                                a.audit_total_duration_ms,
                                a.line as audit_line,
                                a.side as audit_side,
                                a.event_id as audit_event_id,
                                -- Busca closing odd do best_odds_history
                                (
                                    SELECT boh.best_home_odds
                                    FROM best_odds_history boh
                                    JOIN matches m ON boh.match_id = m.id
                                    WHERE m.external_id = a.event_id
                                      AND boh.ah_line = a.line
                                      AND boh.scraped_at < m.kickoff_time
                                    ORDER BY boh.scraped_at DESC
                                    LIMIT 1
                                ) as closing_home,
                                (
                                    SELECT boh.best_away_odds
                                    FROM best_odds_history boh
                                    JOIN matches m ON boh.match_id = m.id
                                    WHERE m.external_id = a.event_id
                                      AND boh.ah_line = a.line
                                      AND boh.scraped_at < m.kickoff_time
                                    ORDER BY boh.scraped_at DESC
                                    LIMIT 1
                                ) as closing_away
                            FROM betslip_audit_results a
                            WHERE a.betslip_odd IS NOT NULL
                              AND a.hypothesis_type = 'H3B'
                              AND a.reversal_direction = '{direcao}'
                              AND a.difference_pct BETWEEN -80 AND 80
                        )
                        SELECT 
                            COUNT(*) as n,
                            AVG(
                                CASE 
                                    WHEN audit_side = 'home' AND closing_home > 0
                                    THEN (betslip_odd - closing_home) / closing_home * 100
                                    WHEN audit_side = 'away' AND closing_away > 0
                                    THEN (betslip_odd - closing_away) / closing_away * 100
                                    ELSE NULL
                                END
                            ) as clv_betslip_medio,
                            STDDEV(
                                CASE 
                                    WHEN audit_side = 'home' AND closing_home > 0
                                    THEN (betslip_odd - closing_home) / closing_home * 100
                                    WHEN audit_side = 'away' AND closing_away > 0
                                    THEN (betslip_odd - closing_away) / closing_away * 100
                                    ELSE NULL
                                END
                            ) as clv_betslip_std
                        FROM audit_matched
                        WHERE (audit_side = 'home' AND closing_home > 0)
                           OR (audit_side = 'away' AND closing_away > 0)
                    """))
                    row2 = result2.fetchone()
                    
                    if row2 and row2[0] and row2[0] > 0:
                        format_result(f"BETSLIP - {label} (via closing line)", row2[0], row2[1], row2[2])
                    else:
                        print(f"   Método alternativo: N=0 (sem closing lines disponíveis)")
                        print(f"   Os jogos auditados provavelmente ainda não tiveram kickoff.")
            
            # ============================================================
            # PARTE 3: Análise com baseline (CLV adicional do Betslip)
            # Mesmo cálculo do v6, mas com betslip_odd
            # ============================================================
            print("\n" + "=" * 70)
            print("PARTE 3: CLV ADICIONAL DO BETSLIP (com baseline)")
            print("=" * 70)
            print("  (Mesmo método v6: CLV do evento - CLV baseline de outras linhas)")
            
            for direcao, label in [('up', 'REVERSÃO UP'), ('down', 'REVERSÃO DOWN')]:
                result = await session.execute(text(f"""
                    WITH audit_with_closing AS (
                        SELECT 
                            a.id,
                            a.betslip_odd,
                            a.websocket_odd,
                            a.is_live,
                            a.audit_total_duration_ms,
                            a.line as audit_line,
                            a.side as audit_side,
                            a.event_id as audit_event_id,
                            m.id as match_id,
                            m.kickoff_time,
                            -- Closing odd do lado correto
                            CASE 
                                WHEN a.side = 'home' THEN (
                                    SELECT boh.best_home_odds FROM best_odds_history boh
                                    WHERE boh.match_id = m.id AND boh.ah_line = a.line
                                      AND boh.scraped_at < m.kickoff_time
                                    ORDER BY boh.scraped_at DESC LIMIT 1
                                )
                                WHEN a.side = 'away' THEN (
                                    SELECT boh.best_away_odds FROM best_odds_history boh
                                    WHERE boh.match_id = m.id AND boh.ah_line = a.line
                                      AND boh.scraped_at < m.kickoff_time
                                    ORDER BY boh.scraped_at DESC LIMIT 1
                                )
                            END as closing_odd,
                            -- CLV betslip
                            CASE 
                                WHEN a.side = 'home' THEN (
                                    SELECT (a.betslip_odd - boh.best_home_odds) / boh.best_home_odds * 100
                                    FROM best_odds_history boh
                                    WHERE boh.match_id = m.id AND boh.ah_line = a.line
                                      AND boh.scraped_at < m.kickoff_time AND boh.best_home_odds > 0
                                    ORDER BY boh.scraped_at DESC LIMIT 1
                                )
                                WHEN a.side = 'away' THEN (
                                    SELECT (a.betslip_odd - boh.best_away_odds) / boh.best_away_odds * 100
                                    FROM best_odds_history boh
                                    WHERE boh.match_id = m.id AND boh.ah_line = a.line
                                      AND boh.scraped_at < m.kickoff_time AND boh.best_away_odds > 0
                                    ORDER BY boh.scraped_at DESC LIMIT 1
                                )
                            END as clv_betslip,
                            -- Baseline CLV (mesma lógica v6: média de CLV de outras linhas no mesmo momento)
                            (
                                SELECT AVG(clv_calc)
                                FROM (
                                    SELECT 
                                        CASE WHEN closing_boh.best_home_odds > 0 
                                        THEN (snapshot.best_home_odds - closing_boh.best_home_odds) 
                                             / closing_boh.best_home_odds * 100
                                        ELSE NULL END as clv_calc
                                    FROM best_odds_history snapshot
                                    LEFT JOIN LATERAL (
                                        SELECT best_home_odds 
                                        FROM best_odds_history c
                                        WHERE c.match_id = snapshot.match_id 
                                          AND c.ah_line = snapshot.ah_line
                                          AND c.scraped_at < m.kickoff_time
                                        ORDER BY c.scraped_at DESC
                                        LIMIT 1
                                    ) closing_boh ON TRUE
                                    WHERE snapshot.match_id = m.id
                                      AND snapshot.ah_line != a.line
                                      AND snapshot.scraped_at BETWEEN a.audited_at - interval '30 seconds' 
                                                                  AND a.audited_at + interval '30 seconds'
                                      AND closing_boh.best_home_odds > 0
                                ) sub
                                WHERE clv_calc BETWEEN -50 AND 50
                            ) as clv_baseline
                        FROM betslip_audit_results a
                        JOIN matches m ON m.external_id = a.event_id
                        WHERE a.betslip_odd IS NOT NULL
                          AND a.hypothesis_type = 'H3B'
                          AND a.reversal_direction = '{direcao}'
                          AND a.difference_pct BETWEEN -80 AND 80
                          AND m.kickoff_time < NOW()
                    )
                    SELECT 
                        COUNT(*) as n,
                        AVG(clv_betslip - COALESCE(clv_baseline, 0)) as clv_adicional,
                        STDDEV(clv_betslip - COALESCE(clv_baseline, 0)) as clv_adicional_std
                    FROM audit_with_closing
                    WHERE clv_betslip IS NOT NULL
                      AND clv_betslip BETWEEN -50 AND 50
                """))
                row = result.fetchone()
                if row and row[0] and row[0] > 0:
                    format_result(f"BETSLIP ADICIONAL - {label}", row[0], row[1], row[2])
                else:
                    print(f"\n   BETSLIP ADICIONAL - {label}: N={row[0] if row else 0}")
                    print(f"   Possíveis causas: jogos ainda sem kickoff, ou event_id não faz match.")
            
            # ============================================================
            # PARTE 4: Visão geral dos dados
            # ============================================================
            print("\n" + "=" * 70)
            print("VISÃO GERAL DOS DADOS DE AUDITORIA")
            print("=" * 70)
            
            result = await session.execute(text("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN betslip_odd IS NOT NULL THEN 1 ELSE 0 END) as com_betslip,
                    SUM(CASE WHEN is_live = true THEN 1 ELSE 0 END) as in_match,
                    SUM(CASE WHEN is_live = false THEN 1 ELSE 0 END) as pre_match,
                    AVG(audit_total_duration_ms) as avg_lag_ms
                FROM betslip_audit_results
                WHERE hypothesis_type = 'H3B'
            """))
            row = result.fetchone()
            if row:
                total, com_betslip, in_match, pre_match, avg_lag = row
                print(f"\n  Total auditorias: {total}")
                print(f"  Com betslip: {com_betslip} ({com_betslip/total*100:.1f}%)" if total > 0 else "")
                print(f"  Pre-match: {pre_match or 0}")
                print(f"  In-match: {in_match or 0}")
                print(f"  Lag médio: {avg_lag:.0f}ms" if avg_lag else "  Lag médio: N/A")
            
            # Contagem de jogos que já tiveram kickoff (closing line disponível)
            result = await session.execute(text("""
                SELECT COUNT(DISTINCT a.event_id)
                FROM betslip_audit_results a
                JOIN matches m ON m.external_id = a.event_id
                WHERE a.betslip_odd IS NOT NULL
                  AND m.kickoff_time < NOW()
            """))
            row = result.fetchone()
            print(f"  Jogos com kickoff passado (closing line disponível): {row[0] if row else 0}")
            
    finally:
        await db.close()
    
    print("\n" + "=" * 70)
    print("FIM DA ANÁLISE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
