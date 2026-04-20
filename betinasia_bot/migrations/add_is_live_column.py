# -*- coding: utf-8 -*-
"""
Migration: Adiciona coluna is_live às tabelas de eventos de hipóteses.

Esta coluna permite distinguir eventos detectados em jogos:
- is_live = False: jogo pre-match
- is_live = True: jogo in-match (ao vivo)

Uso:
    python -m migrations.add_is_live_column
"""

import asyncio
import sys
sys.path.insert(0, '.')

from storage.database import Database
from sqlalchemy import text
from loguru import logger


async def run_migration():
    """Executa a migration."""
    db = Database()
    await db.connect()
    
    print("=" * 70)
    print("MIGRATION: Adicionar coluna is_live às tabelas de hipóteses")
    print("=" * 70)
    
    tables = [
        "h1_pricing_events",
        "h3_line_monotonicity_events",
        "h3b_temporal_reversal_events",
        "h6_correlation_lag_events",
    ]
    
    async with db._engine.begin() as conn:
        for table in tables:
            print(f"\n📋 Tabela: {table}")
            print("-" * 40)
            
            # Verifica se coluna já existe
            check_query = text(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = '{table}' AND column_name = 'is_live'
            """)
            result = await conn.execute(check_query)
            exists = result.fetchone() is not None
            
            if exists:
                print(f"  ⏭️ Coluna is_live já existe")
            else:
                # Adiciona coluna
                alter_query = text(f"""
                    ALTER TABLE {table} 
                    ADD COLUMN is_live BOOLEAN DEFAULT FALSE
                """)
                await conn.execute(alter_query)
                print(f"  ✅ Coluna is_live adicionada")
            
            # Cria índice se não existir
            index_name = f"idx_{table.split('_')[0]}_is_live"
            check_index = text(f"""
                SELECT indexname 
                FROM pg_indexes 
                WHERE tablename = '{table}' AND indexname = '{index_name}'
            """)
            result = await conn.execute(check_index)
            index_exists = result.fetchone() is not None
            
            if index_exists:
                print(f"  ⏭️ Índice {index_name} já existe")
            else:
                try:
                    create_index = text(f"""
                        CREATE INDEX {index_name} ON {table} (is_live)
                    """)
                    await conn.execute(create_index)
                    print(f"  ✅ Índice {index_name} criado")
                except Exception as e:
                    print(f"  ⚠️ Erro ao criar índice: {e}")
    
    # Estatísticas
    print("\n" + "=" * 70)
    print("ESTATÍSTICAS")
    print("=" * 70)
    
    async with db.session() as session:
        for table in tables:
            result = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
            total = result.scalar()
            
            result = await session.execute(text(f"SELECT COUNT(*) FROM {table} WHERE is_live = TRUE"))
            live = result.scalar()
            
            result = await session.execute(text(f"SELECT COUNT(*) FROM {table} WHERE is_live = FALSE OR is_live IS NULL"))
            prematch = result.scalar()
            
            print(f"\n{table}:")
            print(f"  Total: {total}")
            print(f"  Pre-match: {prematch}")
            print(f"  In-match: {live}")
    
    await db.close()
    
    print("\n" + "=" * 70)
    print("Migration concluída!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_migration())
