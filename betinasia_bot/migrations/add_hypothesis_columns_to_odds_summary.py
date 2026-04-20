# -*- coding: utf-8 -*-
"""
Migration: Adiciona colunas de hipóteses à tabela odds_summary.
"""

import asyncio
import sys

sys.path.insert(0, '.')

from storage.database import Database
from sqlalchemy import text


async def migrate():
    """Adiciona colunas de hipóteses à tabela odds_summary."""
    
    db = Database()
    await db.connect()
    
    print("=" * 70)
    print("MIGRATION: Adicionar colunas de hipóteses ao odds_summary")
    print("=" * 70)
    
    # Colunas a adicionar
    columns = [
        # H1 - Pricing
        ("h1_pricing_events_count", "INTEGER", "0"),
        ("h1_had_arb", "INTEGER", "0"),  # Boolean como integer
        ("h1_avg_edge", "FLOAT", "NULL"),
        ("h1_max_edge", "FLOAT", "NULL"),
        
        # H3 - Line Monotonicity
        ("h3_line_anomaly_count", "INTEGER", "0"),
        ("h3_anomaly_magnitude_max", "FLOAT", "NULL"),
        ("h3_anomaly_magnitude_avg", "FLOAT", "NULL"),
        
        # H3b - Temporal Reversals
        ("h3b_reversal_count", "INTEGER", "0"),
        ("h3b_oscillation_index", "FLOAT", "NULL"),
        ("h3b_max_reversal_magnitude", "FLOAT", "NULL"),
        ("h3b_avg_reversal_magnitude", "FLOAT", "NULL"),
        
        # H6 - Correlation Lag
        ("h6_lag_events_count", "INTEGER", "0"),
        ("h6_avg_lag_seconds", "FLOAT", "NULL"),
        ("h6_max_lag_seconds", "FLOAT", "NULL"),
    ]
    
    try:
        async with db.async_session() as session:
            # Verifica se tabela existe
            result = await session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'odds_summary'
                )
            """))
            table_exists = result.scalar()
            
            if not table_exists:
                print("\n   ⚠️ Tabela odds_summary não existe. Criando...")
                # A tabela será criada pelo modelo SQLAlchemy quando necessário
                print("   Execute compact_odds para criar a tabela automaticamente.")
                return
            
            added = 0
            existed = 0
            
            for col_name, col_type, default in columns:
                # Verifica se coluna existe
                result = await session.execute(text(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.columns 
                        WHERE table_name = 'odds_summary' 
                        AND column_name = '{col_name}'
                    )
                """))
                exists = result.scalar()
                
                if exists:
                    print(f"   ⏭️ {col_name} já existe")
                    existed += 1
                else:
                    # Adiciona coluna
                    default_clause = f"DEFAULT {default}" if default != "NULL" else ""
                    await session.execute(text(f"""
                        ALTER TABLE odds_summary 
                        ADD COLUMN {col_name} {col_type} {default_clause}
                    """))
                    print(f"   ✅ {col_name} ({col_type}) adicionada")
                    added += 1
            
            await session.commit()
            
            print("\n" + "=" * 70)
            print(f"RESUMO: {added} colunas adicionadas, {existed} já existiam")
            print("=" * 70)
            
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(migrate())
