# -*- coding: utf-8 -*-
"""
Migration: Adiciona colunas de verificação à tabela H6.

Permite registrar:
- Se o evento foi verificado (auditado)
- Se é um falso positivo (odd não existe de fato)
- O motivo do falso positivo
- A odd real encontrada no betslip

Uso:
    python -m migrations.add_verification_columns
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
    print("MIGRATION: Adicionar colunas de verificação à tabela H6")
    print("=" * 70)
    
    columns = [
        ("verification_status", "VARCHAR(30)"),  # NULL, VERIFIED, FALSE_POSITIVE
        ("verification_reason", "VARCHAR(50)"),  # LINE_NOT_AVAILABLE, GAME_NOT_FOUND, etc.
        ("verified_at", "TIMESTAMP WITH TIME ZONE"),
        ("verified_odd", "FLOAT"),
        ("verified_diff_pct", "FLOAT"),
    ]
    
    table = "h6_correlation_lag_events"
    
    async with db._engine.begin() as conn:
        print(f"\n📋 Tabela: {table}")
        print("-" * 40)
        
        for col_name, col_type in columns:
            # Verifica se coluna já existe
            check_query = text(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = '{table}' AND column_name = '{col_name}'
            """)
            result = await conn.execute(check_query)
            exists = result.fetchone() is not None
            
            if exists:
                print(f"  ⏭️ {col_name} já existe")
            else:
                # Adiciona coluna
                alter_query = text(f"""
                    ALTER TABLE {table} 
                    ADD COLUMN {col_name} {col_type}
                """)
                await conn.execute(alter_query)
                print(f"  ✅ {col_name} ({col_type}) adicionada")
        
        # Cria índice para verification_status
        index_name = "idx_h6_verification"
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
                    CREATE INDEX {index_name} ON {table} (verification_status)
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
        result = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
        total = result.scalar()
        
        result = await session.execute(text(f"""
            SELECT COUNT(*) FROM {table} WHERE verification_status IS NOT NULL
        """))
        verified = result.scalar()
        
        result = await session.execute(text(f"""
            SELECT COUNT(*) FROM {table} WHERE verification_status = 'FALSE_POSITIVE'
        """))
        false_positives = result.scalar()
        
        print(f"\nEventos H6:")
        print(f"  Total: {total}")
        print(f"  Verificados: {verified}")
        print(f"  Falsos positivos: {false_positives}")
        print(f"  Não verificados: {total - verified}")
    
    await db.close()
    
    print("\n" + "=" * 70)
    print("Migration concluída!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_migration())
