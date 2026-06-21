# -*- coding: utf-8 -*-
"""
Limpa todas as tabelas de eventos de hipóteses para recomeçar do zero.
"""

import asyncio
import sys
from sqlalchemy import text

sys.path.insert(0, '.')

from storage.database import Database


async def clear_all_events():
    """Limpa todas as tabelas de eventos de hipóteses."""
    
    db = Database()
    await db.connect()
    
    print("=" * 70)
    print("LIMPEZA DE TABELAS DE HIPÓTESES")
    print("=" * 70)
    
    tables = [
        "h1_pricing_events",
        "h3_line_monotonicity_events", 
        "h3b_temporal_reversal_events",
        "h6_correlation_lag_events",
        "odds_movement_history",
    ]
    
    try:
        async with db.async_session() as session:
            for table in tables:
                # Conta antes
                result = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count_before = result.scalar()
                
                # Limpa
                await session.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE"))
                
                print(f"   ✅ {table}: {count_before:,} eventos removidos")
            
            await session.commit()
            
        print("\n" + "=" * 70)
        print("✅ Todas as tabelas limpas com sucesso!")
        print("=" * 70)
        
    finally:
        await db.close()


if __name__ == "__main__":
    confirm = input("⚠️  Isso vai APAGAR TODOS os eventos de hipóteses. Continuar? (sim/não): ")
    if confirm.lower() == "sim":
        asyncio.run(clear_all_events())
    else:
        print("Operação cancelada.")
