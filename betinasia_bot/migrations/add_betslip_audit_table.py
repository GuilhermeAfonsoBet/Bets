# -*- coding: utf-8 -*-
"""
Migration: Adicionar tabela betslip_audit_results

Cria tabela para armazenar resultados de auditoria de betslip,
permitindo análise estatística de WebSocket vs Betslip real.
"""

import asyncio
import sys
sys.path.insert(0, '.')

from storage.database import Database


async def run_migration():
    """Executa a migração."""
    db = Database()
    await db.connect()
    
    print("=" * 70)
    print("MIGRATION: Adicionar tabela betslip_audit_results")
    print("=" * 70)
    
    try:
        async with db._engine.begin() as conn:
            # Verifica se tabela já existe
            result = await conn.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'betslip_audit_results'
                )
            """)
            exists = (await result.fetchone())[0]
            
            if exists:
                print("⏭️  Tabela betslip_audit_results já existe")
            else:
                # Cria a tabela
                await conn.execute("""
                    CREATE TABLE betslip_audit_results (
                        id SERIAL PRIMARY KEY,
                        
                        -- Identificação do evento
                        hypothesis_type VARCHAR(10) NOT NULL,
                        hypothesis_event_id INTEGER,
                        event_id VARCHAR(100),
                        match_info VARCHAR(200),
                        
                        -- Mercado
                        market_type VARCHAR(20) NOT NULL,
                        line VARCHAR(20) NOT NULL,
                        side VARCHAR(10) NOT NULL,
                        
                        -- Odds comparação
                        websocket_odd FLOAT NOT NULL,
                        betslip_odd FLOAT,
                        difference_pct FLOAT,
                        
                        -- Limites
                        betslip_limit FLOAT,
                        
                        -- Status
                        status VARCHAR(30) NOT NULL,
                        is_valid_opportunity BOOLEAN DEFAULT FALSE,
                        
                        -- Contexto
                        reversal_direction VARCHAR(10),
                        lag_direction VARCHAR(10),
                        
                        -- Metadados
                        audited_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        audit_duration_ms INTEGER,
                        
                        -- Debug
                        raw_betslip_text TEXT
                    )
                """)
                print("✅ Tabela betslip_audit_results criada")
                
                # Cria índices
                indices = [
                    ("idx_audit_hypothesis", "hypothesis_type"),
                    ("idx_audit_status", "status"),
                    ("idx_audit_valid", "is_valid_opportunity"),
                    ("idx_audit_date", "audited_at"),
                    ("idx_audit_diff", "difference_pct"),
                ]
                
                for idx_name, column in indices:
                    try:
                        await conn.execute(f"""
                            CREATE INDEX {idx_name} 
                            ON betslip_audit_results ({column})
                        """)
                        print(f"✅ Índice {idx_name} criado")
                    except Exception as e:
                        if "already exists" in str(e):
                            print(f"⏭️  Índice {idx_name} já existe")
                        else:
                            print(f"❌ Erro ao criar índice {idx_name}: {e}")
        
        print("\n" + "=" * 70)
        print("MIGRATION CONCLUÍDA")
        print("=" * 70)
        
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(run_migration())
