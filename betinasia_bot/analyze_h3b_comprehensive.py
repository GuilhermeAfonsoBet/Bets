#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análise Abrangente H3B — CLV + ROI com múltiplas dimensões

Dimensões:
  (i)   IC CLV adicional + ROI adicional (betslip)
  (ii)  Espaços amostrais: betslip << ws, betslip < ws, betslip > ws, betslip >> ws
  (iii) Pre-match vs In-match
  (iv)  Correlações (liga, tempo antes kickoff, etc)
  (v)   Faixas de linhas AH (-2 a +2 vs fora)
  (vi)  Faixas de lag
  (vii) Revalidação WebSocket

Uso:
    python analyze_h3b_comprehensive.py
"""

import asyncio
import sys
import math
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, '.')

from sqlalchemy import text
from storage.database import Database

Z_90 = 1.645
Z_95 = 1.960


def calc_stats(values):
    """Calcula N, média, std, SE, IC90, IC95."""
    if not values or len(values) < 2:
        return None
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance)
    se = std / math.sqrt(n)
    return {
        'n': n, 'mean': mean, 'std': std, 'se': se,
        'ic90': (mean - Z_90 * se, mean + Z_90 * se),
        'ic95': (mean - Z_95 * se, mean + Z_95 * se),
        'median': sorted(values)[n // 2],
        'p25': sorted(values)[max(0, int(n * 0.25))],
        'p75': sorted(values)[min(n - 1, int(n * 0.75))],
    }


def print_stats(label, values, indent=3):
    s = calc_stats(values)
    pad = " " * indent
    if not s:
        print(f"{pad}{label}: N={len(values) if values else 0} (insuficiente)")
        return s
    
    sig = ""
    if s['ic90'][0] > 0:
        sig = "[OK] SIGNIFICATIVO POSITIVO (p<0.10)"
    elif s['ic90'][1] < 0:
        sig = "[X] SIGNIFICATIVO NEGATIVO (p<0.10)"
    else:
        sig = "[~] Nao significativo"
    
    print(f"{pad}{label}:")
    print(f"{pad}  N = {s['n']}")
    print(f"{pad}  Media = {s['mean']:+.3f}%")
    print(f"{pad}  Mediana = {s['median']:+.3f}%")
    print(f"{pad}  Erro padrao = {s['se']:.3f}%")
    print(f"{pad}  IC 90% = [{s['ic90'][0]:+.3f}%, {s['ic90'][1]:+.3f}%]")
    print(f"{pad}  IC 95% = [{s['ic95'][0]:+.3f}%, {s['ic95'][1]:+.3f}%]")
    print(f"{pad}  {sig}")
    
    if s['mean'] > 0 and s['ic90'][0] <= 0 and s['std'] > 0:
        n_needed = math.ceil((Z_90 * s['std'] / s['mean']) ** 2)
        print(f"{pad}  N estimado p/ significancia: ~{n_needed}")
    
    return s


async def main():
    db = Database()
    await db.connect()
    
    print("=" * 70)
    print("ANALISE ABRANGENTE H3B — CLV + ROI")
    print("=" * 70)
    print(f"Data: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    
    async with db.async_session() as session:
        
        # ============================================================
        # CARREGA TODOS OS DADOS DE AUDITORIA COM CLOSING LINE
        # ============================================================
        result = await session.execute(text("""
            SELECT 
                a.id,
                a.event_id,
                a.home_team,
                a.away_team,
                a.league,
                a.market_type,
                a.line,
                a.side,
                a.websocket_odd,
                a.betslip_odd,
                a.difference_pct,
                a.betslip_limit,
                a.status,
                a.is_live,
                a.audit_total_duration_ms,
                a.lag_detection_to_click_ms,
                a.lag_click_to_betslip_ms,
                a.reversal_direction,
                a.market_period,
                m.id as match_id,
                m.kickoff_time,
                m.home_score,
                m.away_score,
                m.status as match_status,
                -- Closing odd (lado correto)
                CASE 
                    WHEN a.side = 'home' THEN (
                        SELECT boh.best_home_odds FROM best_odds_history boh
                        WHERE boh.match_id = m.id 
                          AND (boh.ah_line = a.line OR boh.ah_line = a.line || '.0')
                          AND boh.scraped_at < m.kickoff_time AND boh.best_home_odds > 0
                        ORDER BY boh.scraped_at DESC LIMIT 1
                    )
                    ELSE (
                        SELECT boh.best_away_odds FROM best_odds_history boh
                        WHERE boh.match_id = m.id 
                          AND (boh.ah_line = a.line OR boh.ah_line = a.line || '.0')
                          AND boh.scraped_at < m.kickoff_time AND boh.best_away_odds > 0
                        ORDER BY boh.scraped_at DESC LIMIT 1
                    )
                END as closing_odd
            FROM betslip_audit_results a
            JOIN matches m ON m.external_id = a.event_id
            WHERE a.hypothesis_type = 'H3B'
              AND a.reversal_direction = 'up'
              AND m.kickoff_time < NOW()
        """))
        rows = result.fetchall()
    
    print(f"\nDados carregados: {len(rows)} auditorias H3B UP com match e kickoff passado")
    
    # ============================================================
    # PROCESSA E CLASSIFICA
    # ============================================================
    all_data = []
    for row in rows:
        d = {
            'id': row[0], 'event_id': row[1],
            'home_team': row[2], 'away_team': row[3],
            'league': row[4] or '', 'market_type': row[5],
            'line': row[6], 'side': row[7],
            'ws_odd': row[8], 'bs_odd': row[9],
            'diff_pct': row[10], 'limit': row[11] or 0,
            'status': row[12], 'is_live': row[13],
            'lag_total': row[14] or 0,
            'lag_click': row[15] or 0, 'lag_bs': row[16] or 0,
            'direction': row[17], 'period': row[18],
            'match_id': row[19], 'kickoff': row[20],
            'home_score': row[21], 'away_score': row[22],
            'match_status': row[23], 'closing_odd': row[24],
        }
        
        # Calcula CLV WS e CLV BS
        if d['closing_odd'] and d['closing_odd'] > 0:
            d['clv_ws'] = (d['ws_odd'] - d['closing_odd']) / d['closing_odd'] * 100
            if d['bs_odd'] and d['bs_odd'] > 0:
                d['clv_bs'] = (d['bs_odd'] - d['closing_odd']) / d['closing_odd'] * 100
            else:
                d['clv_bs'] = None
        else:
            d['clv_ws'] = None
            d['clv_bs'] = None
        
        # Calcula ROI (se temos resultado do jogo)
        d['roi_ws'] = None
        d['roi_bs'] = None
        if d['home_score'] is not None and d['away_score'] is not None:
            goal_diff = d['home_score'] - d['away_score']
            try:
                ah_line = float(d['line'].replace(',', '.'))
            except:
                ah_line = None
            
            if ah_line is not None:
                # Determina se a aposta ganhou
                # Side=home: aposta no home com handicap AH
                # Side=away: aposta no away com handicap -AH
                if d['side'] == 'home':
                    adjusted = goal_diff + ah_line
                else:
                    adjusted = -goal_diff - ah_line
                
                # Resultado: win (>0), lose (<0), push (=0), half win/lose (0.25, -0.25)
                if adjusted > 0.25:
                    result_mult = 1  # Win
                elif adjusted == 0.25:
                    result_mult = 0.5  # Half win
                elif adjusted == 0:
                    result_mult = 0  # Push
                elif adjusted == -0.25:
                    result_mult = -0.5  # Half lose
                else:
                    result_mult = -1  # Lose
                
                # ROI = (odds - 1) * mult se win, -1 * mult se lose
                if result_mult > 0:
                    if d['ws_odd']:
                        d['roi_ws'] = (d['ws_odd'] - 1) * result_mult * 100  # Em %
                    if d['bs_odd'] and d['bs_odd'] > 0:
                        d['roi_bs'] = (d['bs_odd'] - 1) * result_mult * 100
                elif result_mult < 0:
                    d['roi_ws'] = result_mult * 100  # -50% ou -100%
                    d['roi_bs'] = result_mult * 100
                else:
                    d['roi_ws'] = 0
                    d['roi_bs'] = 0
        
        # Classifica diff betslip vs ws
        if d['diff_pct'] is not None:
            if d['diff_pct'] < -10:
                d['diff_bucket'] = 'BS << WS (< -10%)'
            elif d['diff_pct'] < -2:
                d['diff_bucket'] = 'BS < WS (-10% a -2%)'
            elif d['diff_pct'] <= 2:
                d['diff_bucket'] = 'BS ~ WS (-2% a +2%)'
            elif d['diff_pct'] <= 10:
                d['diff_bucket'] = 'BS > WS (+2% a +10%)'
            else:
                d['diff_bucket'] = 'BS >> WS (> +10%)'
        else:
            d['diff_bucket'] = None
        
        # Classifica linha AH
        try:
            line_val = abs(float(d['line'].replace(',', '.')))
            if line_val <= 1:
                d['line_bucket'] = 'AH 0-1 (liquida)'
            elif line_val <= 2:
                d['line_bucket'] = 'AH 1-2 (media)'
            else:
                d['line_bucket'] = 'AH 2+ (extrema)'
        except:
            d['line_bucket'] = 'Outro'
        
        # Classifica lag
        if d['lag_total'] > 0:
            if d['lag_total'] < 10000:
                d['lag_bucket'] = 'Lag < 10s'
            elif d['lag_total'] < 20000:
                d['lag_bucket'] = 'Lag 10-20s'
            elif d['lag_total'] < 30000:
                d['lag_bucket'] = 'Lag 20-30s'
            else:
                d['lag_bucket'] = 'Lag > 30s'
        else:
            d['lag_bucket'] = 'Desconhecido'
        
        all_data.append(d)
    
    # Subconjuntos
    # Filtra apenas betslip odds com diferença razoável (-10% a +10%)
    # Registros fora deste range provavelmente têm erro de extração
    with_bs_raw = [d for d in all_data if d['bs_odd'] and d['bs_odd'] > 0]
    with_bs = [d for d in with_bs_raw if d['diff_pct'] is not None and -10 <= d['diff_pct'] <= 10]
    
    print(f"\n  FILTRO DE QUALIDADE:")
    print(f"    Betslip total (bruto): {len(with_bs_raw)}")
    print(f"    Com diff entre -10% e +10% (confiavel): {len(with_bs)}")
    print(f"    Descartados (diff fora do range): {len(with_bs_raw) - len(with_bs)}")
    with_clv_ws = [d for d in all_data if d['clv_ws'] is not None and -50 < d['clv_ws'] < 50]
    with_clv_bs = [d for d in with_bs if d['clv_bs'] is not None and -50 < d['clv_bs'] < 50]
    with_roi_ws = [d for d in all_data if d['roi_ws'] is not None]
    with_roi_bs = [d for d in with_bs if d['roi_bs'] is not None]
    
    prematch = [d for d in all_data if d['is_live'] == False]
    inmatch = [d for d in all_data if d['is_live'] == True]
    
    print(f"\nResumo:")
    print(f"  Total com match+kickoff: {len(all_data)}")
    print(f"  Com betslip: {len(with_bs)}")
    print(f"  Com CLV WS: {len(with_clv_ws)}")
    print(f"  Com CLV BS: {len(with_clv_bs)}")
    print(f"  Com ROI WS (resultado do jogo): {len(with_roi_ws)}")
    print(f"  Com ROI BS: {len(with_roi_bs)}")
    print(f"  Pre-match: {len(prematch)}")
    print(f"  In-match: {len(inmatch)}")
    
    # ============================================================
    # (vii) REVALIDACAO WEBSOCKET
    # ============================================================
    print("\n" + "=" * 70)
    print("(vii) REVALIDACAO CLV WEBSOCKET (todos os dados atuais)")
    print("=" * 70)
    
    print("\n  PRE-MATCH:")
    pm_clv = [d['clv_ws'] for d in with_clv_ws if d['is_live'] == False]
    print_stats("CLV WS Pre-Match", pm_clv)
    
    print("\n  IN-MATCH:")
    im_clv = [d['clv_ws'] for d in with_clv_ws if d['is_live'] == True]
    print_stats("CLV WS In-Match", im_clv)
    
    print("\n  TODOS:")
    print_stats("CLV WS Total", [d['clv_ws'] for d in with_clv_ws])
    
    # ============================================================
    # (i) CLV ADICIONAL + ROI ADICIONAL (BETSLIP)
    # ============================================================
    print("\n" + "=" * 70)
    print("(i) CLV E ROI COM ODD BETSLIP")
    print("=" * 70)
    
    print("\n  --- CLV BETSLIP (pre-match apenas) ---")
    pm_clv_bs = [d['clv_bs'] for d in with_clv_bs if d['is_live'] == False]
    print_stats("CLV Betslip Pre-Match", pm_clv_bs)
    
    print("\n  --- ROI BETSLIP (pre-match) ---")
    pm_roi_bs = [d['roi_bs'] for d in with_roi_bs if d['is_live'] == False]
    print_stats("ROI Betslip Pre-Match", pm_roi_bs)
    
    print("\n  --- ROI WEBSOCKET (pre-match, referencia) ---")
    pm_roi_ws = [d['roi_ws'] for d in with_roi_ws if d['is_live'] == False]
    print_stats("ROI WS Pre-Match", pm_roi_ws)
    
    print("\n  --- ROI BETSLIP (in-match) ---")
    im_roi_bs = [d['roi_bs'] for d in with_roi_bs if d['is_live'] == True]
    print_stats("ROI Betslip In-Match", im_roi_bs)
    
    print("\n  --- ROI WEBSOCKET (in-match) ---")
    im_roi_ws = [d['roi_ws'] for d in with_roi_ws if d['is_live'] == True]
    print_stats("ROI WS In-Match", im_roi_ws)
    
    # ============================================================
    # (ii) ESPACOS AMOSTRAIS: BS vs WS
    # ============================================================
    print("\n" + "=" * 70)
    print("(ii) ANALISE POR DIFERENCA BETSLIP vs WEBSOCKET")
    print("=" * 70)
    
    buckets_order = ['BS << WS (< -10%)', 'BS < WS (-10% a -2%)', 'BS ~ WS (-2% a +2%)',
                     'BS > WS (+2% a +10%)', 'BS >> WS (> +10%)']
    
    for bucket in buckets_order:
        subset = [d for d in with_bs if d['diff_bucket'] == bucket]
        if not subset:
            continue
        
        print(f"\n  === {bucket} (N={len(subset)}) ===")
        
        clv_vals = [d['clv_bs'] for d in subset if d['clv_bs'] is not None and -50 < d['clv_bs'] < 50 and d['is_live'] == False]
        if clv_vals:
            print_stats(f"CLV Betslip (pre-match)", clv_vals, indent=5)
        
        roi_vals = [d['roi_bs'] for d in subset if d['roi_bs'] is not None]
        if roi_vals:
            print_stats(f"ROI Betslip (todos)", roi_vals, indent=5)
    
    # ============================================================
    # (iii) PRE-MATCH vs IN-MATCH
    # ============================================================
    print("\n" + "=" * 70)
    print("(iii) PRE-MATCH vs IN-MATCH")
    print("=" * 70)
    
    for label, is_live_val in [("PRE-MATCH", False), ("IN-MATCH", True)]:
        subset = [d for d in with_bs if d['is_live'] == is_live_val]
        print(f"\n  === {label} (N={len(subset)}) ===")
        
        clv = [d['clv_bs'] for d in subset if d['clv_bs'] is not None and -50 < d['clv_bs'] < 50]
        roi = [d['roi_bs'] for d in subset if d['roi_bs'] is not None]
        diff = [d['diff_pct'] for d in subset if d['diff_pct'] is not None]
        
        if clv:
            print_stats(f"CLV Betslip", clv, indent=5)
        if roi:
            print_stats(f"ROI Betslip", roi, indent=5)
        if diff:
            print_stats(f"Diff BS vs WS", diff, indent=5)
    
    # ============================================================
    # (iv) CORRELACOES (liga, etc)
    # ============================================================
    print("\n" + "=" * 70)
    print("(iv) POR LIGA (top 10 com mais dados)")
    print("=" * 70)
    
    by_league = defaultdict(list)
    for d in with_bs:
        if d['league']:
            by_league[d['league']].append(d)
    
    sorted_leagues = sorted(by_league.items(), key=lambda x: -len(x[1]))[:10]
    for league, subset in sorted_leagues:
        if len(subset) < 5:
            continue
        roi_vals = [d['roi_bs'] for d in subset if d['roi_bs'] is not None]
        clv_vals = [d['clv_bs'] for d in subset if d['clv_bs'] is not None and -50 < d['clv_bs'] < 50 and d['is_live'] == False]
        
        n_roi = len(roi_vals)
        n_clv = len(clv_vals)
        avg_roi = sum(roi_vals) / len(roi_vals) if roi_vals else 0
        avg_clv = sum(clv_vals) / len(clv_vals) if clv_vals else 0
        
        print(f"\n  {league}: N={len(subset)} (ROI: {n_roi}, CLV: {n_clv})")
        if roi_vals and len(roi_vals) >= 3:
            print_stats(f"ROI", roi_vals, indent=5)
    
    # ============================================================
    # (v) FAIXAS DE LINHAS AH
    # ============================================================
    print("\n" + "=" * 70)
    print("(v) POR FAIXA DE LINHA AH")
    print("=" * 70)
    
    for bucket in ['AH 0-1 (liquida)', 'AH 1-2 (media)', 'AH 2+ (extrema)']:
        subset = [d for d in with_bs if d['line_bucket'] == bucket]
        if not subset:
            continue
        
        print(f"\n  === {bucket} (N={len(subset)}) ===")
        
        clv = [d['clv_bs'] for d in subset if d['clv_bs'] is not None and -50 < d['clv_bs'] < 50 and d['is_live'] == False]
        roi = [d['roi_bs'] for d in subset if d['roi_bs'] is not None]
        diff = [d['diff_pct'] for d in subset if d['diff_pct'] is not None]
        
        if clv:
            print_stats(f"CLV Betslip (pre-match)", clv, indent=5)
        if roi:
            print_stats(f"ROI Betslip", roi, indent=5)
        if diff:
            print_stats(f"Diff BS vs WS", diff, indent=5)
    
    # ============================================================
    # (vi) FAIXAS DE LAG
    # ============================================================
    print("\n" + "=" * 70)
    print("(vi) POR FAIXA DE LAG")
    print("=" * 70)
    
    for bucket in ['Lag < 10s', 'Lag 10-20s', 'Lag 20-30s', 'Lag > 30s']:
        subset = [d for d in with_bs if d['lag_bucket'] == bucket]
        if not subset:
            continue
        
        print(f"\n  === {bucket} (N={len(subset)}) ===")
        
        clv = [d['clv_bs'] for d in subset if d['clv_bs'] is not None and -50 < d['clv_bs'] < 50 and d['is_live'] == False]
        roi = [d['roi_bs'] for d in subset if d['roi_bs'] is not None]
        diff = [d['diff_pct'] for d in subset if d['diff_pct'] is not None]
        
        if clv:
            print_stats(f"CLV Betslip (pre-match)", clv, indent=5)
        if roi:
            print_stats(f"ROI Betslip", roi, indent=5)
        if diff:
            print_stats(f"Diff BS vs WS", diff, indent=5)
    
    # ============================================================
    # DIAGNOSTICO DE QUALIDADE DOS DADOS
    # ============================================================
    print("\n" + "=" * 70)
    print("DIAGNOSTICO DE QUALIDADE")
    print("=" * 70)
    
    # Verifica se betslip odds fazem sentido
    if with_bs:
        bs_odds = [d['bs_odd'] for d in with_bs]
        ws_odds = [d['ws_odd'] for d in with_bs]
        diffs = [d['diff_pct'] for d in with_bs if d['diff_pct'] is not None]
        
        print(f"\n  Betslip odds: min={min(bs_odds):.3f} med={sorted(bs_odds)[len(bs_odds)//2]:.3f} max={max(bs_odds):.3f}")
        print(f"  Websocket odds: min={min(ws_odds):.3f} med={sorted(ws_odds)[len(ws_odds)//2]:.3f} max={max(ws_odds):.3f}")
        if diffs:
            print(f"  Diferenca: min={min(diffs):+.1f}% med={sorted(diffs)[len(diffs)//2]:+.1f}% max={max(diffs):+.1f}%")
        
        # Contagem de diffs extremas
        extreme_neg = len([d for d in diffs if d < -30])
        extreme_pos = len([d for d in diffs if d > 30])
        print(f"  Diffs < -30%: {extreme_neg} ({extreme_neg/len(diffs)*100:.1f}%)")
        print(f"  Diffs > +30%: {extreme_pos} ({extreme_pos/len(diffs)*100:.1f}%)")
        
        # Jogos com resultado
        with_result = len([d for d in all_data if d['home_score'] is not None])
        print(f"\n  Jogos com resultado (gols): {with_result}/{len(all_data)}")
    
    await db.close()
    
    print("\n" + "=" * 70)
    print("FIM DA ANALISE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
