# -*- coding: utf-8 -*-
"""
Salva resultados de auditoria no banco de dados.

Atualiza eventos H6 com status de verificação:
- VERIFIED: odd existe e foi confirmada
- FALSE_POSITIVE: odd não existe (linha não disponível, jogo não encontrado, etc.)

Uso:
    python save_audit_results.py <arquivo_resultados.json>
    
    Ou manualmente:
    python save_audit_results.py --match-id 123 --line "-1.0" --side "home" --status FALSE_POSITIVE --reason LINE_NOT_AVAILABLE
"""

import asyncio
import sys
import json
import argparse
from datetime import datetime, timezone
from typing import Optional

sys.path.insert(0, '.')

from storage.database import Database
from storage.models_hypothesis import H6CorrelationLagEvent
from sqlalchemy import select, and_, update


async def update_h6_verification(
    db: Database,
    match_id: int,
    line: str,
    side: str,
    status: str,  # VERIFIED ou FALSE_POSITIVE
    reason: Optional[str] = None,
    verified_odd: Optional[float] = None,
    diff_pct: Optional[float] = None,
) -> int:
    """
    Atualiza o status de verificação de eventos H6.
    
    Returns:
        Número de eventos atualizados.
    """
    async with db.session() as session:
        # Busca eventos que correspondem aos critérios
        result = await session.execute(
            select(H6CorrelationLagEvent).where(
                and_(
                    H6CorrelationLagEvent.match_id == match_id,
                    H6CorrelationLagEvent.lagged_line == line,
                    H6CorrelationLagEvent.lagged_side == side,
                    H6CorrelationLagEvent.verification_status.is_(None)  # Só atualiza não verificados
                )
            )
        )
        events = result.scalars().all()
        
        count = 0
        for event in events:
            event.verification_status = status
            event.verification_reason = reason
            event.verified_at = datetime.now(timezone.utc)
            if verified_odd:
                event.verified_odd = verified_odd
            if diff_pct:
                event.verified_diff_pct = diff_pct
            count += 1
        
        await session.commit()
        
    return count


async def process_audit_results(results_file: str):
    """Processa arquivo de resultados de auditoria."""
    db = Database()
    await db.connect()
    
    print("=" * 70)
    print("SALVANDO RESULTADOS DE AUDITORIA")
    print("=" * 70)
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        total_updated = 0
        verified = 0
        false_positives = 0
        
        for result in results:
            # Determina status baseado no resultado
            if result.get('status') in ['IDENTICAL', 'OK', 'MINOR_DIFF', 'MAJOR_DIFF']:
                status = 'VERIFIED'
                reason = None
                verified += 1
            else:
                status = 'FALSE_POSITIVE'
                reason = result.get('status', 'UNKNOWN')
                false_positives += 1
            
            # Precisa mapear event_id para match_id
            # Por enquanto, vamos pular se não tiver match_id direto
            match_id = result.get('match_id')
            if not match_id:
                print(f"  ⚠️ Sem match_id para {result.get('match_info')}")
                continue
            
            count = await update_h6_verification(
                db=db,
                match_id=match_id,
                line=result.get('line'),
                side=result.get('side'),
                status=status,
                reason=reason,
                verified_odd=result.get('betslip_best_odd'),
                diff_pct=result.get('difference_pct'),
            )
            
            total_updated += count
            print(f"  {'✅' if status == 'VERIFIED' else '❌'} {result.get('match_info')}: {status} ({count} eventos)")
        
        print("\n" + "=" * 70)
        print("RESUMO")
        print("=" * 70)
        print(f"  Verificados: {verified}")
        print(f"  Falsos positivos: {false_positives}")
        print(f"  Total eventos atualizados: {total_updated}")
        
    finally:
        await db.close()


async def manual_update(args):
    """Atualização manual de um evento."""
    db = Database()
    await db.connect()
    
    try:
        count = await update_h6_verification(
            db=db,
            match_id=args.match_id,
            line=args.line,
            side=args.side,
            status=args.status,
            reason=args.reason,
            verified_odd=args.verified_odd,
            diff_pct=args.diff_pct,
        )
        
        print(f"Atualizados {count} eventos")
        
    finally:
        await db.close()


async def show_stats():
    """Mostra estatísticas de verificação."""
    db = Database()
    await db.connect()
    
    print("=" * 70)
    print("ESTATÍSTICAS DE VERIFICAÇÃO - H6")
    print("=" * 70)
    
    try:
        async with db.session() as session:
            # Total
            result = await session.execute(
                select(H6CorrelationLagEvent)
            )
            all_events = result.scalars().all()
            
            total = len(all_events)
            verified = sum(1 for e in all_events if e.verification_status == 'VERIFIED')
            false_positive = sum(1 for e in all_events if e.verification_status == 'FALSE_POSITIVE')
            not_verified = sum(1 for e in all_events if e.verification_status is None)
            
            print(f"\nTotal de eventos H6: {total}")
            print(f"  ✅ Verificados (odd existe): {verified} ({verified/total*100:.1f}%)" if total else "")
            print(f"  ❌ Falsos positivos: {false_positive} ({false_positive/total*100:.1f}%)" if total else "")
            print(f"  ⏳ Não verificados: {not_verified} ({not_verified/total*100:.1f}%)" if total else "")
            
            # Por motivo
            if false_positive > 0:
                print(f"\nFalsos positivos por motivo:")
                reasons = {}
                for e in all_events:
                    if e.verification_status == 'FALSE_POSITIVE':
                        r = e.verification_reason or 'UNKNOWN'
                        reasons[r] = reasons.get(r, 0) + 1
                
                for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                    print(f"  {reason}: {count}")
            
            # Diferenças para verificados
            if verified > 0:
                diffs = [e.verified_diff_pct for e in all_events 
                         if e.verification_status == 'VERIFIED' and e.verified_diff_pct is not None]
                if diffs:
                    print(f"\nDiferenças WebSocket vs Betslip (verificados):")
                    print(f"  Média: {sum(diffs)/len(diffs):.3f}%")
                    print(f"  Máxima: {max(diffs):.3f}%")
                    print(f"  Mínima: {min(diffs):.3f}%")
            
    finally:
        await db.close()


def main():
    parser = argparse.ArgumentParser(description='Salva resultados de auditoria no banco')
    
    subparsers = parser.add_subparsers(dest='command')
    
    # Subcomando: processar arquivo
    file_parser = subparsers.add_parser('file', help='Processa arquivo de resultados')
    file_parser.add_argument('results_file', help='Arquivo JSON com resultados')
    
    # Subcomando: atualização manual
    manual_parser = subparsers.add_parser('manual', help='Atualização manual')
    manual_parser.add_argument('--match-id', type=int, required=True)
    manual_parser.add_argument('--line', required=True)
    manual_parser.add_argument('--side', required=True)
    manual_parser.add_argument('--status', required=True, choices=['VERIFIED', 'FALSE_POSITIVE'])
    manual_parser.add_argument('--reason', default=None)
    manual_parser.add_argument('--verified-odd', type=float, default=None)
    manual_parser.add_argument('--diff-pct', type=float, default=None)
    
    # Subcomando: estatísticas
    stats_parser = subparsers.add_parser('stats', help='Mostra estatísticas')
    
    args = parser.parse_args()
    
    if args.command == 'file':
        asyncio.run(process_audit_results(args.results_file))
    elif args.command == 'manual':
        asyncio.run(manual_update(args))
    elif args.command == 'stats':
        asyncio.run(show_stats())
    else:
        # Default: mostra estatísticas
        asyncio.run(show_stats())


if __name__ == "__main__":
    main()
