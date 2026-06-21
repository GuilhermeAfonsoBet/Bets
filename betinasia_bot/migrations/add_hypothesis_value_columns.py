# -*- coding: utf-8 -*-
"""
Migration: Adiciona colunas de análise de valor às tabelas de hipóteses.

Uso:
    python -m migrations.add_hypothesis_value_columns
    
    # Para reverter:
    python -m migrations.add_hypothesis_value_columns --rollback
"""

import asyncio
import argparse
from loguru import logger
from sqlalchemy import text

import sys
sys.path.insert(0, '.')

from storage.database import Database


# Colunas a adicionar em cada tabela
MIGRATIONS = {
    "h1_pricing_events": [
        ("recommended_side", "VARCHAR(10)"),
        ("recommended_odd", "FLOAT"),
        ("closing_odd_side_a", "FLOAT"),
        ("closing_odd_side_b", "FLOAT"),
        ("closing_odd_recommended", "FLOAT"),
        ("clv", "FLOAT"),
        ("clv_pct", "FLOAT"),
        ("bet_result", "VARCHAR(20)"),
        ("profit_loss", "FLOAT"),
    ],
    "h3_line_monotonicity_events": [
        ("recommended_line", "VARCHAR(20)"),
        ("recommended_odd", "FLOAT"),
        ("closing_odd_line_a", "FLOAT"),
        ("closing_odd_line_b", "FLOAT"),
        ("closing_odd_recommended", "FLOAT"),
        ("clv", "FLOAT"),
        ("clv_pct", "FLOAT"),
        ("bet_result", "VARCHAR(20)"),
        ("profit_loss", "FLOAT"),
    ],
    "h3b_temporal_reversal_events": [
        ("bet_odd", "FLOAT"),
        ("bet_side", "VARCHAR(10)"),
        ("closing_odd", "FLOAT"),
        ("clv", "FLOAT"),
        ("clv_pct", "FLOAT"),
        ("bet_result", "VARCHAR(20)"),
        ("profit_loss", "FLOAT"),
    ],
    "h6_correlation_lag_events": [
        ("bet_market_type", "VARCHAR(20)"),
        ("bet_line", "VARCHAR(20)"),
        ("bet_side", "VARCHAR(10)"),
        ("bet_odd", "FLOAT"),
        ("closing_odd", "FLOAT"),
        ("clv", "FLOAT"),
        ("clv_pct", "FLOAT"),
        ("bet_result", "VARCHAR(20)"),
        ("profit_loss", "FLOAT"),
    ],
}

# Índices a criar
INDEXES = [
    ("idx_h1_clv", "h1_pricing_events", "clv_pct"),
    ("idx_h1_result", "h1_pricing_events", "bet_result"),
    ("idx_h3_clv", "h3_line_monotonicity_events", "clv_pct"),
    ("idx_h3_result", "h3_line_monotonicity_events", "bet_result"),
    ("idx_h3b_clv", "h3b_temporal_reversal_events", "clv_pct"),
    ("idx_h3b_result", "h3b_temporal_reversal_events", "bet_result"),
    ("idx_h6_clv", "h6_correlation_lag_events", "clv_pct"),
    ("idx_h6_result", "h6_correlation_lag_events", "bet_result"),
]


async def check_column_exists(conn, table: str, column: str) -> bool:
    """Verifica se uma coluna existe na tabela."""
    result = await conn.execute(text("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = :table AND column_name = :column
    """), {"table": table, "column": column})
    return result.fetchone() is not None


async def check_index_exists(conn, index_name: str) -> bool:
    """Verifica se um índice existe."""
    result = await conn.execute(text("""
        SELECT indexname 
        FROM pg_indexes 
        WHERE indexname = :index_name
    """), {"index_name": index_name})
    return result.fetchone() is not None


async def run_migration():
    """Executa a migration - adiciona colunas faltantes."""
    db = Database()
    await db.connect()
    
    print("=" * 70)
    print("MIGRATION: Adicionar colunas de análise de valor")
    print("=" * 70)
    print()
    
    try:
        async with db.engine.begin() as conn:
            total_added = 0
            total_skipped = 0
            
            for table, columns in MIGRATIONS.items():
                print(f"\n📋 Tabela: {table}")
                print("-" * 40)
                
                for col_name, col_type in columns:
                    exists = await check_column_exists(conn, table, col_name)
                    
                    if exists:
                        print(f"   ⏭️  {col_name} já existe")
                        total_skipped += 1
                    else:
                        await conn.execute(text(
                            f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}"
                        ))
                        print(f"   ✅ {col_name} ({col_type}) adicionada")
                        total_added += 1
            
            # Cria índices
            print(f"\n📊 Criando índices...")
            print("-" * 40)
            
            for idx_name, table, column in INDEXES:
                exists = await check_index_exists(conn, idx_name)
                
                if exists:
                    print(f"   ⏭️  {idx_name} já existe")
                else:
                    try:
                        await conn.execute(text(
                            f"CREATE INDEX {idx_name} ON {table} ({column})"
                        ))
                        print(f"   ✅ {idx_name} criado")
                    except Exception as e:
                        print(f"   ⚠️  {idx_name}: {e}")
            
            print()
            print("=" * 70)
            print(f"RESUMO: {total_added} colunas adicionadas, {total_skipped} já existiam")
            print("=" * 70)
            
    finally:
        await db.close()


async def run_rollback():
    """Reverte a migration - remove colunas adicionadas."""
    db = Database()
    await db.connect()
    
    print("=" * 70)
    print("ROLLBACK: Remover colunas de análise de valor")
    print("=" * 70)
    print()
    
    confirm = input("⚠️  Isso vai APAGAR dados! Confirma? (digite 'SIM'): ")
    if confirm != "SIM":
        print("Cancelado.")
        return
    
    try:
        async with db.engine.begin() as conn:
            for table, columns in MIGRATIONS.items():
                print(f"\n📋 Tabela: {table}")
                
                for col_name, _ in columns:
                    exists = await check_column_exists(conn, table, col_name)
                    
                    if exists:
                        await conn.execute(text(
                            f"ALTER TABLE {table} DROP COLUMN {col_name}"
                        ))
                        print(f"   🗑️  {col_name} removida")
                    else:
                        print(f"   ⏭️  {col_name} não existe")
            
            print("\nRollback concluído!")
            
    finally:
        await db.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollback", action="store_true", help="Reverte a migration")
    args = parser.parse_args()
    
    if args.rollback:
        asyncio.run(run_rollback())
    else:
        asyncio.run(run_migration())


if __name__ == "__main__":
    main()
