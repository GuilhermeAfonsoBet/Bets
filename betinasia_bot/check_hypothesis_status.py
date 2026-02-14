# -*- coding: utf-8 -*-
"""
Verificador de Status do Sistema de Hipóteses.

Verifica:
1. Se as tabelas de hipóteses existem no banco
2. Se eventos estão sendo detectados e salvos
3. Se o coletor está configurado corretamente

Uso:
    python check_hypothesis_status.py
"""

import asyncio
import sys
from datetime import datetime, timezone, timedelta
from loguru import logger
from sqlalchemy import text, inspect

# Configura path
sys.path.insert(0, '.')

from storage.database import Database
from storage.models import Base, Match, BestOddsHistory
from storage.models_hypothesis import (
    H1PricingEvent, H3LineMonotonicityEvent,
    H3bTemporalReversalEvent, H6CorrelationLagEvent,
    OddsMovementHistory
)


async def check_tables_exist(db: Database) -> dict:
    """Verifica se as tabelas de hipóteses existem."""
    results = {}
    
    expected_tables = [
        "h1_pricing_events",
        "h3_line_monotonicity_events", 
        "h3b_temporal_reversal_events",
        "h6_correlation_lag_events",
        "odds_movement_history"
    ]
    
    async with db.engine.connect() as conn:
        # Obtém lista de tabelas existentes
        result = await conn.execute(text(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        ))
        existing_tables = [row[0] for row in result.fetchall()]
    
    for table in expected_tables:
        results[table] = table in existing_tables
    
    return results


async def count_events(db: Database) -> dict:
    """Conta eventos em cada tabela de hipóteses."""
    counts = {}
    
    async with db.async_session() as session:
        # H1
        result = await session.execute(text("SELECT COUNT(*) FROM h1_pricing_events"))
        counts["h1_pricing_events"] = result.scalar() or 0
        
        # H3
        result = await session.execute(text("SELECT COUNT(*) FROM h3_line_monotonicity_events"))
        counts["h3_line_monotonicity_events"] = result.scalar() or 0
        
        # H3b
        result = await session.execute(text("SELECT COUNT(*) FROM h3b_temporal_reversal_events"))
        counts["h3b_temporal_reversal_events"] = result.scalar() or 0
        
        # H6
        result = await session.execute(text("SELECT COUNT(*) FROM h6_correlation_lag_events"))
        counts["h6_correlation_lag_events"] = result.scalar() or 0
        
        # Movimentos
        result = await session.execute(text("SELECT COUNT(*) FROM odds_movement_history"))
        counts["odds_movement_history"] = result.scalar() or 0
    
    return counts


async def get_recent_events(db: Database, hours: int = 24) -> dict:
    """Obtém contagem de eventos recentes."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    recent = {}
    
    async with db.async_session() as session:
        # H1
        result = await session.execute(text(
            f"SELECT COUNT(*) FROM h1_pricing_events WHERE detected_at >= :cutoff"
        ), {"cutoff": cutoff})
        recent["h1"] = result.scalar() or 0
        
        # H3
        result = await session.execute(text(
            f"SELECT COUNT(*) FROM h3_line_monotonicity_events WHERE detected_at >= :cutoff"
        ), {"cutoff": cutoff})
        recent["h3"] = result.scalar() or 0
        
        # H3b
        result = await session.execute(text(
            f"SELECT COUNT(*) FROM h3b_temporal_reversal_events WHERE detected_at >= :cutoff"
        ), {"cutoff": cutoff})
        recent["h3b"] = result.scalar() or 0
        
        # H6
        result = await session.execute(text(
            f"SELECT COUNT(*) FROM h6_correlation_lag_events WHERE detected_at >= :cutoff"
        ), {"cutoff": cutoff})
        recent["h6"] = result.scalar() or 0
    
    return recent


async def get_sample_events(db: Database) -> dict:
    """Obtém exemplos de eventos recentes."""
    samples = {}
    
    async with db.async_session() as session:
        # H1 - último evento
        result = await session.execute(text("""
            SELECT id, match_id, market_type, ah_line, is_arb, edge_estimate, 
                   recommended_side, recommended_odd, detected_at
            FROM h1_pricing_events 
            ORDER BY detected_at DESC LIMIT 3
        """))
        samples["h1"] = [dict(row._mapping) for row in result.fetchall()]
        
        # H3 - último evento
        result = await session.execute(text("""
            SELECT id, match_id, line_a, line_b, side, magnitude, 
                   recommended_line, recommended_odd, detected_at
            FROM h3_line_monotonicity_events 
            ORDER BY detected_at DESC LIMIT 3
        """))
        samples["h3"] = [dict(row._mapping) for row in result.fetchall()]
        
        # H3b - último evento
        result = await session.execute(text("""
            SELECT id, match_id, market_type, ah_line, side, direction_before, 
                   direction_after, bet_odd, detected_at
            FROM h3b_temporal_reversal_events 
            ORDER BY detected_at DESC LIMIT 3
        """))
        samples["h3b"] = [dict(row._mapping) for row in result.fetchall()]
        
        # H6 - último evento
        result = await session.execute(text("""
            SELECT id, match_id, leader_line, lagged_line, lag_seconds, 
                   bet_odd, detected_at
            FROM h6_correlation_lag_events 
            ORDER BY detected_at DESC LIMIT 3
        """))
        samples["h6"] = [dict(row._mapping) for row in result.fetchall()]
    
    return samples


async def check_collector_integration(db: Database) -> dict:
    """Verifica se o coletor está chamando os detectores."""
    info = {}
    
    async with db.async_session() as session:
        # Verifica última coleta de odds
        result = await session.execute(text("""
            SELECT MAX(scraped_at) as last_scrape,
                   COUNT(*) as total_records,
                   COUNT(DISTINCT match_id) as unique_matches
            FROM best_odds_history
            WHERE scraped_at >= NOW() - INTERVAL '1 hour'
        """))
        row = result.fetchone()
        if row:
            info["last_odds_scrape"] = row[0]
            info["odds_records_1h"] = row[1]
            info["unique_matches_1h"] = row[2]
        
        # Verifica se há matches
        result = await session.execute(text("SELECT COUNT(*) FROM matches"))
        info["total_matches"] = result.scalar() or 0
        
        # Verifica matches com resultado
        result = await session.execute(text(
            "SELECT COUNT(*) FROM matches WHERE home_score IS NOT NULL"
        ))
        info["matches_with_result"] = result.scalar() or 0
    
    return info


async def check_hypothesis_detector_in_collector():
    """Verifica se o detector está importado no coletor."""
    try:
        with open("collector/continuous_collector.py", "r") as f:
            content = f.read()
        
        checks = {
            "import HypothesisDetector": "HypothesisDetector" in content,
            "hypothesis_detector attribute": "self.hypothesis_detector" in content,
            "process_market_update call": "process_market_update" in content,
            "save_hypothesis_events call": "save_hypothesis_events" in content,
        }
        return checks
    except Exception as e:
        return {"error": str(e)}


async def main():
    print("=" * 70)
    print("VERIFICAÇÃO DO SISTEMA DE HIPÓTESES")
    print("=" * 70)
    print()
    
    # 1. Verifica código do coletor
    print("1. INTEGRAÇÃO NO COLETOR")
    print("-" * 40)
    collector_checks = await check_hypothesis_detector_in_collector()
    for check, status in collector_checks.items():
        icon = "✅" if status else "❌"
        print(f"   {icon} {check}")
    print()
    
    # Conecta ao banco
    db = Database()
    try:
        await db.connect()
        print("✅ Conexão com banco de dados OK")
        print()
    except Exception as e:
        print(f"❌ Erro ao conectar ao banco: {e}")
        return
    
    try:
        # 2. Verifica se tabelas existem
        print("2. TABELAS NO BANCO DE DADOS")
        print("-" * 40)
        tables = await check_tables_exist(db)
        all_exist = True
        for table, exists in tables.items():
            icon = "✅" if exists else "❌"
            print(f"   {icon} {table}")
            if not exists:
                all_exist = False
        
        if not all_exist:
            print()
            print("⚠️  Algumas tabelas não existem!")
            print("   Execute: python -c 'from storage.database import init_db; import asyncio; asyncio.run(init_db())'")
            print()
        print()
        
        # 3. Conta eventos
        print("3. CONTAGEM DE EVENTOS (TOTAL)")
        print("-" * 40)
        try:
            counts = await count_events(db)
            total = 0
            for table, count in counts.items():
                print(f"   {table}: {count:,}")
                total += count
            print(f"   TOTAL: {total:,}")
        except Exception as e:
            print(f"   ❌ Erro ao contar eventos: {e}")
        print()
        
        # 4. Eventos recentes
        print("4. EVENTOS NAS ÚLTIMAS 24 HORAS")
        print("-" * 40)
        try:
            recent = await get_recent_events(db, hours=24)
            for hyp, count in recent.items():
                icon = "✅" if count > 0 else "⚠️"
                print(f"   {icon} {hyp.upper()}: {count} eventos")
            
            if sum(recent.values()) == 0:
                print()
                print("   ⚠️  Nenhum evento detectado nas últimas 24h!")
                print("   Verifique se o coletor está rodando corretamente.")
        except Exception as e:
            print(f"   ❌ Erro: {e}")
        print()
        
        # 5. Info do coletor
        print("5. STATUS DA COLETA")
        print("-" * 40)
        try:
            info = await check_collector_integration(db)
            print(f"   Última coleta de odds: {info.get('last_odds_scrape', 'N/A')}")
            print(f"   Registros de odds (1h): {info.get('odds_records_1h', 0):,}")
            print(f"   Jogos únicos (1h): {info.get('unique_matches_1h', 0)}")
            print(f"   Total de jogos no banco: {info.get('total_matches', 0):,}")
            print(f"   Jogos com resultado: {info.get('matches_with_result', 0):,}")
        except Exception as e:
            print(f"   ❌ Erro: {e}")
        print()
        
        # 6. Amostras de eventos
        print("6. EXEMPLOS DE EVENTOS RECENTES")
        print("-" * 40)
        try:
            samples = await get_sample_events(db)
            
            for hyp, events in samples.items():
                print(f"\n   {hyp.upper()}:")
                if events:
                    for evt in events[:2]:  # Mostra até 2
                        # Formata de forma compacta
                        evt_str = ", ".join(f"{k}={v}" for k, v in list(evt.items())[:5])
                        print(f"      - {evt_str}...")
                else:
                    print(f"      (nenhum evento)")
        except Exception as e:
            print(f"   ❌ Erro: {e}")
        print()
        
        # Resumo final
        print("=" * 70)
        print("RESUMO")
        print("=" * 70)
        
        issues = []
        if not all(collector_checks.values()):
            issues.append("Integração do detector no coletor incompleta")
        if not all_exist:
            issues.append("Tabelas de hipóteses não criadas")
        if sum(recent.values()) == 0:
            issues.append("Nenhum evento detectado nas últimas 24h")
        
        if issues:
            print("❌ PROBLEMAS ENCONTRADOS:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("✅ Sistema de hipóteses funcionando corretamente!")
        
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
