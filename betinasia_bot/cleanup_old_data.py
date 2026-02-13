#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Limpa dados antigos do banco de dados.

Remove registros de odds com mais de X dias (padrao: 7 dias).
Deve ser executado periodicamente via cron.

Uso:
    python cleanup_old_data.py           # Limpa dados > 7 dias
    python cleanup_old_data.py --days 14 # Limpa dados > 14 dias
    python cleanup_old_data.py --dry-run # Mostra o que seria deletado
"""

import asyncio
import argparse
from datetime import datetime, timezone, timedelta
from sqlalchemy import delete, select, func
from storage.database import Database
from storage.models import BestOddsHistory, Match
from loguru import logger


async def cleanup(days: int = 7, dry_run: bool = False):
    """
    Remove dados de odds mais antigos que X dias.
    
    Args:
        days: Numero de dias para manter
        dry_run: Se True, apenas mostra o que seria deletado
    """
    
    db = Database()
    await db.connect()
    
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    
    print("=" * 60)
    print(f"LIMPEZA DE DADOS ANTIGOS")
    print("=" * 60)
    print(f"Data de corte: {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Registros mais antigos que {days} dias serao {'LISTADOS' if dry_run else 'REMOVIDOS'}")
    print()
    
    async with db.async_session() as session:
        # Conta registros a serem deletados
        result = await session.execute(
            select(func.count(BestOddsHistory.id))
            .where(BestOddsHistory.scraped_at < cutoff_date)
        )
        old_odds_count = result.scalar()
        
        print(f"Registros de odds a remover: {old_odds_count:,}")
        
        # Conta jogos que podem ser removidos (sem odds recentes)
        # Jogos cujo kickoff ja passou e nao tem odds recentes
        result = await session.execute(
            select(func.count(Match.id))
            .where(Match.kickoff_time < cutoff_date)
            .where(Match.status == 'finished')
        )
        old_matches_count = result.scalar()
        
        print(f"Jogos finalizados > {days} dias: {old_matches_count:,}")
        
        if dry_run:
            print("\n[DRY RUN] Nenhum dado foi removido.")
        else:
            # Deleta odds antigas
            if old_odds_count > 0:
                print(f"\nDeletando {old_odds_count:,} registros de odds...")
                
                # Deleta em batches para nao sobrecarregar
                batch_size = 10000
                total_deleted = 0
                
                while True:
                    # Busca IDs para deletar
                    result = await session.execute(
                        select(BestOddsHistory.id)
                        .where(BestOddsHistory.scraped_at < cutoff_date)
                        .limit(batch_size)
                    )
                    ids_to_delete = [row[0] for row in result]
                    
                    if not ids_to_delete:
                        break
                    
                    # Deleta batch
                    await session.execute(
                        delete(BestOddsHistory)
                        .where(BestOddsHistory.id.in_(ids_to_delete))
                    )
                    await session.commit()
                    
                    total_deleted += len(ids_to_delete)
                    print(f"  Deletados: {total_deleted:,} / {old_odds_count:,}")
                
                print(f"  Concluido! {total_deleted:,} registros removidos.")
            
            print("\n✅ Limpeza concluida!")
    
    await db.close()
    
    # Estatisticas pos-limpeza
    if not dry_run:
        print("\n" + "=" * 60)
        print("ESTATISTICAS POS-LIMPEZA")
        print("=" * 60)
        
        db = Database()
        await db.connect()
        
        async with db.async_session() as session:
            result = await session.execute(select(func.count(BestOddsHistory.id)))
            total_odds = result.scalar()
            
            result = await session.execute(select(func.count(Match.id)))
            total_matches = result.scalar()
            
            print(f"Total de registros de odds: {total_odds:,}")
            print(f"Total de jogos: {total_matches:,}")
        
        await db.close()


def main():
    parser = argparse.ArgumentParser(description='Limpa dados antigos do banco')
    parser.add_argument('--days', type=int, default=7, 
                        help='Numero de dias para manter (padrao: 7)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Apenas mostra o que seria deletado')
    
    args = parser.parse_args()
    
    asyncio.run(cleanup(days=args.days, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
