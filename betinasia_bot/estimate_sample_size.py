# -*- coding: utf-8 -*-
"""
Estimativa de tamanho de amostra necessário para p-valor < 0.10
"""

import asyncio
import sys
import math
from datetime import datetime, timedelta
from sqlalchemy import text

sys.path.insert(0, '.')

from storage.database import Database


def calc_sample_size_needed(effect: float, std: float, alpha: float = 0.10, power: float = 0.80) -> int:
    """
    Calcula tamanho de amostra necessário para detectar efeito com significância.
    
    Fórmula: n = (Z_α + Z_β)² × σ² / δ²
    
    Para teste unicaudal (queremos CLV > 0):
    - Z_α = 1.28 para α = 0.10
    - Z_β = 0.84 para power = 0.80
    """
    if effect == 0:
        return float('inf')
    
    # Z-scores
    z_alpha = 1.28  # unicaudal, α = 0.10
    z_beta = 0.84   # power = 0.80
    
    n = ((z_alpha + z_beta) ** 2 * std ** 2) / (effect ** 2)
    return int(math.ceil(n))


async def estimate():
    db = Database()
    await db.connect()
    
    print("=" * 70)
    print("ESTIMATIVA DE TAMANHO DE AMOSTRA PARA P-VALOR < 0.10")
    print("=" * 70)
    print("""
Parâmetros:
- Teste unicaudal (H1: CLV adicional > 0)
- α = 0.10 (significância)
- Power = 80%
""")
    
    try:
        async with db.async_session() as session:
            
            # Calcular período de coleta
            result = await session.execute(text("""
                SELECT 
                    MIN(detected_at) as primeiro,
                    MAX(detected_at) as ultimo
                FROM h1_pricing_events
                WHERE clv_pct IS NOT NULL
            """))
            row = result.fetchone()
            if row and row[0] and row[1]:
                dias_coleta = (row[1] - row[0]).total_seconds() / 86400
                dias_coleta = max(dias_coleta, 0.5)  # mínimo meio dia
            else:
                dias_coleta = 1
            
            print(f"\nPeríodo de coleta: {dias_coleta:.1f} dias")
            
            # Dados por hipótese
            hipoteses = [
                ("H1", "h1_pricing_events", "ah_line", None, 442, 0.047, 8.927),
                ("H3", "h3_line_monotonicity_events", "recommended_line", None, 48, 0.615, 10.208),
                ("H3B UP", "h3b_temporal_reversal_events", "ah_line", "direction_after = 'up'", 273, 1.116, 15.860),
                ("H3B DOWN", "h3b_temporal_reversal_events", "ah_line", "direction_after = 'down'", 282, -1.359, 16.552),
                ("H6 DOWN", "h6_correlation_lag_events", "lagged_line", "leader_move_direction = 'down'", 388, 2.301, 11.645),
                ("H6 UP", "h6_correlation_lag_events", "lagged_line", "leader_move_direction = 'up'", 391, 3.068, 12.047),
            ]
            
            print("\n" + "=" * 70)
            print("ESTIMATIVAS POR HIPÓTESE")
            print("=" * 70)
            print("\n   Hipótese   | N atual | CLV adic | Std    | N necessário | Taxa/dia | Dias faltam")
            print("   " + "-" * 80)
            
            for nome, tabela, linha_col, filtro, n_atual, clv_adic, std in hipoteses:
                # Calcular taxa de eventos por dia
                taxa_dia = n_atual / dias_coleta
                
                # Calcular N necessário
                if clv_adic > 0:
                    n_necessario = calc_sample_size_needed(clv_adic, std)
                else:
                    # Para CLV negativo, não faz sentido calcular (hipótese não agrega)
                    n_necessario = None
                
                # Calcular dias restantes
                if n_necessario and n_necessario > n_atual:
                    n_faltam = n_necessario - n_atual
                    dias_faltam = n_faltam / taxa_dia if taxa_dia > 0 else float('inf')
                elif n_necessario and n_necessario <= n_atual:
                    dias_faltam = 0
                else:
                    dias_faltam = None
                
                # Formatação
                n_str = f"{n_necessario:,}" if n_necessario and n_necessario < 1000000 else ("N/A" if not n_necessario else ">1M")
                dias_str = f"{dias_faltam:.0f}" if dias_faltam is not None and dias_faltam < 10000 else ("N/A" if dias_faltam is None else ">10k")
                
                status = ""
                if clv_adic <= 0:
                    status = " ❌ (CLV negativo)"
                elif n_necessario and n_atual >= n_necessario:
                    status = " ✅ (já significativo!)"
                elif dias_faltam and dias_faltam < 30:
                    status = " 🔜 (em breve)"
                
                print(f"   {nome:10} | {n_atual:7} | {clv_adic:7}% | {std:6}% | {n_str:>12} | {taxa_dia:8.1f} | {dias_str:>8}{status}")
            
            print("\n" + "=" * 70)
            print("INTERPRETAÇÃO")
            print("=" * 70)
            print("""
N necessário = tamanho de amostra para detectar o efeito com p < 0.10

ATENÇÃO:
- Se CLV adicional é PEQUENO (ex: 0.047%), precisamos de MUITOS dados
- Se CLV adicional é GRANDE (ex: 2.3%), precisamos de menos dados
- Se CLV adicional é NEGATIVO, a hipótese não tem valor (N/A)

HIPÓTESES PROMISSORAS (CLV > 0 e N razoável):
- H6 DOWN: CLV adicional = 2.3%, pode ser significativo em breve
- H3B UP: CLV adicional = 1.1%, precisa de mais dados
- H3: CLV adicional = 0.6%, precisa de bastante dados
- H1: CLV adicional = 0.05%, efeito muito pequeno (inconclusivo)

RECOMENDAÇÃO:
Focar na coleta de dados para H6 DOWN e H3B UP que têm efeitos maiores.
""")
            
            # Cálculo mais detalhado para H6 DOWN
            print("\n" + "=" * 70)
            print("ANÁLISE DETALHADA: H6 DOWN (mais promissor)")
            print("=" * 70)
            
            # Verificar p-valor atual
            n = 388
            clv_adic = 2.301
            std = 11.645
            se = std / math.sqrt(n)
            t_stat = clv_adic / se
            
            # Aproximação do p-valor (usando distribuição normal)
            from scipy import stats
            p_valor = 1 - stats.norm.cdf(t_stat)
            
            print(f"""
   Dados atuais:
   N = {n}
   CLV adicional = {clv_adic}%
   Erro padrão = {se:.3f}%
   t-statistic = {t_stat:.3f}
   p-valor (unicaudal) ≈ {p_valor:.4f}
""")
            
            if p_valor < 0.10:
                print("   ✅ JÁ É SIGNIFICATIVO com p < 0.10!")
            else:
                print(f"   Precisa de mais dados para p < 0.10")
            
    except ImportError:
        print("\n   (scipy não instalado - cálculo de p-valor aproximado não disponível)")
            
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(estimate())
