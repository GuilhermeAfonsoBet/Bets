#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análise H3B: Valor REALISTA (WebSocket + Betslip)

Pergunta central:
  O valor detectado no WebSocket (CLV +1.116% para H3B UP)
  SOBREVIVE quando consideramos a odd REAL do betslip?

Lógica:
  - CLV_websocket = 1.116% (medido na análise anterior)
  - Diferença betslip = (betslip_odd - websocket_odd) / websocket_odd
  - CLV_realizável ≈ CLV_websocket + diferença_betslip

Se diferença_betslip é negativa (betslip < websocket),
o lag "come" parte do valor. Se for muito negativa, come TUDO.

Uso:
    cd ~/Bets/betinasia_bot
    source ../venv/bin/activate  # ou venv/bin/activate
    python analyze_h3b_websocket_vs_betslip.py
"""

import asyncio
import sys
import math
from datetime import datetime, timezone
from collections import defaultdict

sys.path.insert(0, '.')

from sqlalchemy import select, func, text
from storage.database import Database
from storage.models_hypothesis import BetslipAuditResult


# CLV do WebSocket (da análise anterior)
CLV_WEBSOCKET_UP = 1.116  # %
CLV_WEBSOCKET_DOWN = -1.359  # %

# Z-scores para intervalos de confiança
Z_90 = 1.645
Z_95 = 1.960


def calc_stats(values: list) -> dict:
    """Calcula estatísticas descritivas e IC."""
    if not values:
        return None
    
    n = len(values)
    mean = sum(values) / n
    
    if n < 2:
        return {"n": n, "mean": mean, "std": 0, "se": 0, "ic90_low": mean, "ic90_high": mean}
    
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance)
    se = std / math.sqrt(n)  # Erro padrão
    
    ic90_low = mean - Z_90 * se
    ic90_high = mean + Z_90 * se
    
    ic95_low = mean - Z_95 * se
    ic95_high = mean + Z_95 * se
    
    # Mediana
    sorted_vals = sorted(values)
    if n % 2 == 0:
        median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    else:
        median = sorted_vals[n // 2]
    
    # Percentis
    p25 = sorted_vals[max(0, int(n * 0.25))]
    p75 = sorted_vals[min(n - 1, int(n * 0.75))]
    
    return {
        "n": n,
        "mean": mean,
        "median": median,
        "std": std,
        "se": se,
        "p25": p25,
        "p75": p75,
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "ic90_low": ic90_low,
        "ic90_high": ic90_high,
        "ic95_low": ic95_low,
        "ic95_high": ic95_high,
    }


def is_significant_90(stats: dict) -> bool:
    """Verifica se é significativo a 90% (IC não inclui zero)."""
    if not stats:
        return False
    return stats["ic90_low"] > 0 or stats["ic90_high"] < 0


def print_stats_block(title: str, stats: dict, clv_websocket: float = None):
    """Imprime um bloco formatado de estatísticas."""
    if not stats:
        print(f"\n  {title}: Sem dados suficientes")
        return
    
    n = stats["n"]
    
    print(f"\n  {title}:")
    print(f"    N = {n}")
    print(f"    Diferença média (betslip - ws) = {stats['mean']:+.3f}%")
    print(f"    Mediana                        = {stats['median']:+.3f}%")
    print(f"    Desvio padrão                  = {stats['std']:.3f}%")
    print(f"    Erro padrão                    = {stats['se']:.3f}%")
    print(f"    IC 90%  = [{stats['ic90_low']:+.3f}%, {stats['ic90_high']:+.3f}%]")
    print(f"    IC 95%  = [{stats['ic95_low']:+.3f}%, {stats['ic95_high']:+.3f}%]")
    print(f"    Range   = [{stats['min']:+.3f}%, {stats['max']:+.3f}%]")
    print(f"    P25/P75 = [{stats['p25']:+.3f}%, {stats['p75']:+.3f}%]")
    
    if clv_websocket is not None:
        realized_clv = clv_websocket + stats["mean"]
        realized_se = stats["se"]  # Aproximação (ignora erro do CLV websocket)
        realized_ic90_low = realized_clv - Z_90 * realized_se
        realized_ic90_high = realized_clv + Z_90 * realized_se
        
        print(f"\n    --- CLV REALIZÁVEL ---")
        print(f"    CLV WebSocket original     = {clv_websocket:+.3f}%")
        print(f"    Erosão média (betslip lag) = {stats['mean']:+.3f}%")
        print(f"    CLV REALIZÁVEL estimado    = {realized_clv:+.3f}%")
        print(f"    IC 90% do CLV realizável   = [{realized_ic90_low:+.3f}%, {realized_ic90_high:+.3f}%]")
        
        if realized_ic90_low > 0:
            print(f"    ✅ SIGNIFICATIVO POSITIVO - Valor SOBREVIVE na prática!")
        elif realized_ic90_high < 0:
            print(f"    ❌ SIGNIFICATIVO NEGATIVO - Lag CONSUME todo o valor")
        else:
            print(f"    ⚪ Não significativo - Precisa mais dados")
            
            # Estimativa de N necessário para significância
            if realized_clv > 0 and stats["std"] > 0:
                # n_needed tal que Z * std/sqrt(n) < realized_clv
                n_needed = math.ceil((Z_90 * stats["std"] / realized_clv) ** 2)
                print(f"    📊 N estimado p/ significância = ~{n_needed} auditorias")


async def main():
    db = Database()
    await db.connect()
    
    print("=" * 70)
    print("ANÁLISE H3B: VALOR REALISTA (WebSocket + Betslip)")
    print("=" * 70)
    print(f"""
Pergunta: O valor detectado no WebSocket sobrevive na prática?

Método:
  CLV_realizável = CLV_websocket + diferença_betslip
  
  Onde:
    CLV_websocket = {CLV_WEBSOCKET_UP:+.3f}% (H3B UP, da análise anterior)
    diferença_betslip = (betslip_odd - websocket_odd) / websocket_odd × 100

  Se diferença_betslip < 0: o lag entre detecção e execução "come" valor
  Se diferença_betslip > 0: o betslip tem odd MELHOR que o websocket
""")
    
    # === Carrega dados ===
    async with db.async_session() as session:
        result = await session.execute(
            select(BetslipAuditResult).where(
                BetslipAuditResult.hypothesis_type == "H3B"
            )
        )
        audits = result.scalars().all()
    
    if not audits:
        print("❌ Nenhum dado de auditoria encontrado!")
        print("   O script audit_h3b_betslip.py precisa rodar primeiro.")
        await db.close()
        return
    
    # === Classificação dos dados ===
    all_results = []
    successful = []  # Betslip odd extraído com sucesso
    failed = []      # Não conseguiu extrair (game not found, line not available, etc)
    
    by_status = defaultdict(list)
    by_direction = defaultdict(list)
    by_live = {"pre_match": [], "in_match": [], "unknown": []}
    by_lag_bucket = defaultdict(list)
    by_market = defaultdict(list)
    
    for a in audits:
        all_results.append(a)
        by_status[a.status].append(a)
        
        direction = a.reversal_direction or "unknown"
        by_direction[direction].append(a)
        
        if a.betslip_odd is not None and a.difference_pct is not None:
            successful.append(a)
            
            # Por pre-match / in-match
            if a.is_live is True:
                by_live["in_match"].append(a)
            elif a.is_live is False:
                by_live["pre_match"].append(a)
            else:
                by_live["unknown"].append(a)
            
            # Por bucket de lag total
            lag = a.audit_total_duration_ms or 0
            if lag < 5000:
                by_lag_bucket["< 5s"].append(a)
            elif lag < 10000:
                by_lag_bucket["5-10s"].append(a)
            elif lag < 20000:
                by_lag_bucket["10-20s"].append(a)
            elif lag < 30000:
                by_lag_bucket["20-30s"].append(a)
            else:
                by_lag_bucket["> 30s"].append(a)
            
            # Por tipo de mercado
            by_market[a.market_type or "AH"].append(a)
        else:
            failed.append(a)
    
    # === 1. VISÃO GERAL ===
    print("=" * 70)
    print("1. VISÃO GERAL DOS DADOS")
    print("=" * 70)
    print(f"\n  Total de auditorias: {len(all_results)}")
    print(f"  Com betslip extraído (sucesso): {len(successful)} ({len(successful)/len(all_results)*100:.1f}%)")
    print(f"  Sem betslip (falha): {len(failed)} ({len(failed)/len(all_results)*100:.1f}%)")
    
    print(f"\n  Por status:")
    for status, items in sorted(by_status.items(), key=lambda x: -len(x[1])):
        print(f"    {status}: {len(items)}")
    
    print(f"\n  Por direção da reversão:")
    for direction, items in sorted(by_direction.items()):
        n_success = sum(1 for a in items if a.betslip_odd is not None)
        print(f"    {direction.upper()}: {len(items)} total, {n_success} com betslip")
    
    # === 2. ANÁLISE DA DIFERENÇA BETSLIP vs WEBSOCKET ===
    print("\n" + "=" * 70)
    print("2. DIFERENÇA BETSLIP vs WEBSOCKET (todos os eventos com betslip)")
    print("=" * 70)
    
    if successful:
        diffs_all = [a.difference_pct for a in successful]
        stats_all = calc_stats(diffs_all)
        print_stats_block("TODOS OS EVENTOS", stats_all)
    
    # === 3. POR DIREÇÃO DA REVERSÃO (o mais importante) ===
    print("\n" + "=" * 70)
    print("3. POR DIREÇÃO DA REVERSÃO")
    print("=" * 70)
    
    for direction in ["up", "down"]:
        items = [a for a in successful if a.reversal_direction == direction]
        if not items:
            continue
        
        diffs = [a.difference_pct for a in items]
        stats = calc_stats(diffs)
        
        clv_ws = CLV_WEBSOCKET_UP if direction == "up" else CLV_WEBSOCKET_DOWN
        label = f"REVERSÃO {direction.upper()} (N={len(items)})"
        print_stats_block(label, stats, clv_websocket=clv_ws)
    
    # === 4. POR PRE-MATCH vs IN-MATCH ===
    print("\n" + "=" * 70)
    print("4. PRE-MATCH vs IN-MATCH")
    print("=" * 70)
    
    for label, items in [("PRE-MATCH", by_live["pre_match"]), 
                          ("IN-MATCH", by_live["in_match"]),
                          ("DESCONHECIDO", by_live["unknown"])]:
        if not items:
            continue
        
        diffs = [a.difference_pct for a in items]
        stats = calc_stats(diffs)
        print_stats_block(label, stats)
    
    # === 5. POR BUCKET DE LAG ===
    print("\n" + "=" * 70)
    print("5. IMPACTO DO LAG TIME")
    print("=" * 70)
    print("\n  Quanto mais rápido (menor lag), menor a erosão do valor?")
    
    lag_order = ["< 5s", "5-10s", "10-20s", "20-30s", "> 30s"]
    for bucket in lag_order:
        items = by_lag_bucket.get(bucket, [])
        if not items:
            continue
        
        diffs = [a.difference_pct for a in items]
        stats = calc_stats(diffs)
        print_stats_block(f"Lag {bucket} (N={len(items)})", stats)
    
    # === 6. POR TIPO DE MERCADO ===
    if len(by_market) > 1:
        print("\n" + "=" * 70)
        print("6. POR TIPO DE MERCADO")
        print("=" * 70)
        
        for market, items in sorted(by_market.items()):
            diffs = [a.difference_pct for a in items]
            stats = calc_stats(diffs)
            print_stats_block(f"{market} (N={len(items)})", stats)
    
    # === 7. ANÁLISE H3B UP COMBINADA (a que mais importa) ===
    print("\n" + "=" * 70)
    print("7. CONCLUSÃO: H3B UP - VALE A PENA NA PRÁTICA?")
    print("=" * 70)
    
    up_items = [a for a in successful if a.reversal_direction == "up"]
    
    if up_items:
        diffs_up = [a.difference_pct for a in up_items]
        stats_up = calc_stats(diffs_up)
        
        realized_clv = CLV_WEBSOCKET_UP + stats_up["mean"]
        erosion = stats_up["mean"]
        
        print(f"""
  DADOS:
    Auditorias H3B UP com betslip: {stats_up['n']}
    CLV WebSocket (análise anterior): {CLV_WEBSOCKET_UP:+.3f}%
    Erosão média pelo lag: {erosion:+.3f}%
    
  RESULTADO:
    CLV realizável estimado: {realized_clv:+.3f}%""")
        
        if realized_clv > 0:
            # Calcula N necessário para significância do CLV realizável
            if stats_up["std"] > 0:
                n_needed_90 = math.ceil((Z_90 * stats_up["std"] / realized_clv) ** 2)
                n_needed_95 = math.ceil((Z_95 * stats_up["std"] / realized_clv) ** 2)
                
                realized_se = stats_up["se"]
                realized_ic90_low = realized_clv - Z_90 * realized_se
                realized_ic90_high = realized_clv + Z_90 * realized_se
                
                print(f"    IC 90% = [{realized_ic90_low:+.3f}%, {realized_ic90_high:+.3f}%]")
                
                if realized_ic90_low > 0:
                    print(f"""
    ✅ CONCLUSÃO: VALOR SOBREVIVE NA PRÁTICA!
    
    O CLV realizável é positivo E significativo.
    A estratégia H3B UP tem valor mesmo após o lag do betslip.""")
                else:
                    print(f"""
    ⚪ CONCLUSÃO: VALOR PROMISSOR MAS INCONCLUSIVO
    
    O CLV realizável é positivo mas não significativo ainda.
    N atual: {stats_up['n']}
    N estimado para significância (IC 90%): ~{n_needed_90}
    N estimado para significância (IC 95%): ~{n_needed_95}
    
    Continue coletando dados com o audit_h3b_betslip.py.""")
        else:
            print(f"""
    ❌ CONCLUSÃO: LAG CONSOME O VALOR
    
    A erosão pelo lag ({erosion:+.3f}%) é maior que o CLV original ({CLV_WEBSOCKET_UP:+.3f}%).
    Na prática, a estratégia H3B UP NÃO tem valor.""")
        
        # Insight: qual seria o lag máximo aceitável?
        print(f"\n  INSIGHT - LAG MÁXIMO ACEITÁVEL:")
        print(f"    Para manter CLV > 0, a erosão precisa ser < {CLV_WEBSOCKET_UP:+.3f}%")
        
        for bucket in lag_order:
            items = by_lag_bucket.get(bucket, [])
            up_in_bucket = [a for a in items if a.reversal_direction == "up"]
            if up_in_bucket:
                bucket_diffs = [a.difference_pct for a in up_in_bucket]
                bucket_mean = sum(bucket_diffs) / len(bucket_diffs)
                bucket_clv = CLV_WEBSOCKET_UP + bucket_mean
                marker = "✅" if bucket_clv > 0 else "❌"
                print(f"    {marker} Lag {bucket}: erosão {bucket_mean:+.3f}% → CLV realizável {bucket_clv:+.3f}% (N={len(up_in_bucket)})")
    else:
        print("\n  ❌ Sem dados de H3B UP com betslip para análise.")
    
    # === 8. TAXA DE OPORTUNIDADES REAIS ===
    print("\n" + "=" * 70)
    print("8. TAXA DE OPORTUNIDADES REAIS")
    print("=" * 70)
    
    total = len(all_results)
    up_total = len([a for a in all_results if a.reversal_direction == "up"])
    up_with_betslip = len([a for a in successful if a.reversal_direction == "up"])
    up_ok = len([a for a in successful if a.reversal_direction == "up" and 
                 a.status in ["IDENTICAL", "OK", "MINOR_DIFF", "MAJOR_DIFF"]])
    
    if up_total > 0:
        execution_rate = up_with_betslip / up_total * 100
        print(f"""
  H3B UP:
    Total detectados: {up_total}
    Com betslip extraído: {up_with_betslip} ({execution_rate:.1f}%)
    
    Para cada 100 sinais H3B UP no WebSocket:
    → ~{int(execution_rate)} conseguem abrir betslip e verificar odd
    → ~{100 - int(execution_rate)} falham (jogo não encontrado, linha indisponível, etc)
    
    Isso significa que a taxa de execução real é {execution_rate:.1f}%.""")
    
    await db.close()
    
    print("\n" + "=" * 70)
    print("FIM DA ANÁLISE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
