# -*- coding: utf-8 -*-
"""
Migration: Adicionar tabela betslip_audit_results

Cria tabela para armazenar resultados de auditoria de betslip,
permitindo análise estatística de WebSocket vs Betslip real.

Versão: v1.0
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
    print("MIGRATION: Adicionar tabela betslip_audit_results (v2)")
    print("=" * 70)
    
    try:
        async with db._engine.begin() as conn:
            # Drop tabela antiga se existir (para recriar com novos campos)
            await conn.execute("DROP TABLE IF EXISTS betslip_audit_results CASCADE")
            print("🗑️  Tabela antiga removida (se existia)")
            
            # Cria a tabela com todos os campos
            await conn.execute("""
                CREATE TABLE betslip_audit_results (
                    id SERIAL PRIMARY KEY,
                    
                    -- Identificação do evento
                    hypothesis_type VARCHAR(10) NOT NULL,
                    hypothesis_event_id INTEGER,
                    event_id VARCHAR(100),
                    
                    -- Informações do jogo
                    sport VARCHAR(20) DEFAULT 'football',
                    league VARCHAR(100),
                    home_team VARCHAR(100),
                    away_team VARCHAR(100),
                    match_info VARCHAR(200),
                    match_start_time TIMESTAMP WITH TIME ZONE,
                    
                    -- Mercado/Aposta
                    market_type VARCHAR(20) NOT NULL,
                    market_period VARCHAR(20) DEFAULT 'full_time',
                    line VARCHAR(20) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    bet_description VARCHAR(100),
                    
                    -- Odds comparação
                    websocket_odd FLOAT NOT NULL,
                    betslip_odd FLOAT,
                    difference_pct FLOAT,
                    difference_absolute FLOAT,
                    
                    -- Limites
                    betslip_limit FLOAT,
                    
                    -- Status
                    status VARCHAR(30) NOT NULL,
                    is_valid_opportunity BOOLEAN DEFAULT FALSE,
                    
                    -- Contexto da hipótese
                    reversal_direction VARCHAR(10),
                    lag_direction VARCHAR(10),
                    hypothesis_details JSONB,
                    
                    -- Timing/Lag
                    hypothesis_detected_at TIMESTAMP WITH TIME ZONE,
                    audited_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    lag_detection_to_click_ms INTEGER,
                    lag_click_to_betslip_ms INTEGER,
                    audit_total_duration_ms INTEGER,
                    
                    -- Versionamento
                    audit_version VARCHAR(20) DEFAULT 'v1.0',
                    
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
                ("idx_audit_match", "home_team, away_team"),
                ("idx_audit_market", "market_type, line"),
                ("idx_audit_version", "audit_version"),
                ("idx_audit_hypothesis_event", "hypothesis_type, hypothesis_event_id"),
                ("idx_audit_lag", "audit_total_duration_ms"),
            ]
            
            for idx_name, columns in indices:
                try:
                    await conn.execute(f"""
                        CREATE INDEX {idx_name} 
                        ON betslip_audit_results ({columns})
                    """)
                    print(f"✅ Índice {idx_name} criado")
                except Exception as e:
                    print(f"⚠️  Índice {idx_name}: {e}")
        
        print("\n" + "=" * 70)
        print("CAMPOS DA TABELA")
        print("=" * 70)
        print("""
Identificação:
  - hypothesis_type: H1, H3, H3B, H6
  - hypothesis_event_id: ID para merge com tabelas de hipóteses
  - event_id: ID do WebSocket

Jogo:
  - sport, league, home_team, away_team
  - match_start_time: horário do jogo

Mercado:
  - market_type: AH, AH_HT, OU, OU_HT, 1X2
  - market_period: full_time, half_time
  - line, side, bet_description

Odds:
  - websocket_odd: odd no momento da detecção
  - betslip_odd: odd real do betslip
  - difference_pct, difference_absolute

Timing (LAG):
  - hypothesis_detected_at: quando hipótese foi detectada
  - lag_detection_to_click_ms: tempo até clicar na odd
  - lag_click_to_betslip_ms: tempo até extrair betslip
  - audit_total_duration_ms: tempo total

Versionamento:
  - audit_version: versão do script (documentar em /docs/audit_versions.md)
""")
        
        print("=" * 70)
        print("MIGRATION CONCLUÍDA")
        print("=" * 70)
        
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(run_migration())
