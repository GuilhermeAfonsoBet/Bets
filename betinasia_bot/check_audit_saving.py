#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verifica se o audit_h3b_betslip.py está salvando corretamente no banco.

Uso:
    python check_audit_saving.py
"""

import asyncio
import sys
from datetime import datetime, timezone, timedelta

sys.path.insert(0, '.')

from sqlalchemy import text
from storage.database import Database


async def main():
    db = Database()
    await db.connect()
    
    print("=" * 60)
    print("VERIFICAÇÃO: AUDIT H3B SALVANDO NO BANCO?")
    print(f"Data/Hora: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 60)
    
    async with db.async_session() as session:
        
        # 1. Total de registros
        r = await session.execute(text("SELECT COUNT(*) FROM betslip_audit_results"))
        total = r.scalar()
        print(f"\n1. Total de registros: {total}")
        
        # 2. Registros recuperados vs novos
        r = await session.execute(text("""
            SELECT audit_version, COUNT(*) 
            FROM betslip_audit_results 
            GROUP BY audit_version 
            ORDER BY audit_version
        """))
        print(f"\n2. Por versão:")
        for row in r:
            print(f"   {row[0]}: {row[1]}")
        
        # 3. Registros nas últimas horas
        print(f"\n3. Registros por hora (últimas 6h):")
        for h in range(6):
            start = datetime.now(timezone.utc) - timedelta(hours=h+1)
            end = datetime.now(timezone.utc) - timedelta(hours=h)
            r = await session.execute(text("""
                SELECT COUNT(*) FROM betslip_audit_results 
                WHERE audited_at >= :start AND audited_at < :end
            """), {"start": start, "end": end})
            count = r.scalar()
            label = "Última hora" if h == 0 else f"{h+1}h atrás"
            marker = "✅" if count > 0 else "⚠️"
            print(f"   {marker} {label}: {count} registros")
        
        # 4. Último registro salvo
        r = await session.execute(text("""
            SELECT audited_at, event_id, home_team, away_team, 
                   market_type, line, side, status, 
                   websocket_odd, betslip_odd, difference_pct,
                   is_live, audit_version
            FROM betslip_audit_results 
            ORDER BY audited_at DESC 
            LIMIT 1
        """))
        row = r.fetchone()
        if row:
            age = (datetime.now(timezone.utc) - row[0].replace(tzinfo=timezone.utc)).total_seconds()
            print(f"\n4. Último registro salvo:")
            print(f"   Data: {row[0]} ({age:.0f}s atrás)")
            print(f"   Jogo: {row[2]} vs {row[3]}")
            print(f"   Mercado: {row[4]} {row[5]} {row[6]}")
            print(f"   Status: {row[7]}")
            print(f"   WS odd: {row[8]}, Betslip odd: {row[9]}")
            print(f"   Diferença: {row[10]}%")
            print(f"   Is_live: {row[11]}")
            print(f"   Versão: {row[12]}")
            
            if age > 300:
                print(f"\n   ⚠️ Último registro há {age/60:.0f} minutos!")
            else:
                print(f"\n   ✅ Salvamento recente ({age:.0f}s atrás)")
        else:
            print(f"\n4. ❌ Nenhum registro encontrado!")
        
        # 5. Campos preenchidos
        r = await session.execute(text("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN betslip_odd IS NOT NULL THEN 1 ELSE 0 END) as com_betslip,
                SUM(CASE WHEN is_live IS NOT NULL THEN 1 ELSE 0 END) as com_is_live,
                SUM(CASE WHEN league IS NOT NULL AND league != '' THEN 1 ELSE 0 END) as com_league,
                SUM(CASE WHEN match_start_time IS NOT NULL THEN 1 ELSE 0 END) as com_kickoff,
                SUM(CASE WHEN audit_total_duration_ms IS NOT NULL THEN 1 ELSE 0 END) as com_lag,
                SUM(CASE WHEN betslip_limit IS NOT NULL AND betslip_limit > 0 THEN 1 ELSE 0 END) as com_limite
            FROM betslip_audit_results
            WHERE audit_version != 'v1.0-recovered'
        """))
        row = r.fetchone()
        if row and row[0] > 0:
            total_new = row[0]
            print(f"\n5. Campos preenchidos (registros NOVOS, excl. recuperados):")
            print(f"   Total: {total_new}")
            print(f"   Com betslip_odd: {row[1]} ({row[1]/total_new*100:.0f}%)")
            print(f"   Com is_live: {row[2]} ({row[2]/total_new*100:.0f}%)")
            print(f"   Com league: {row[3]} ({row[3]/total_new*100:.0f}%)")
            print(f"   Com kickoff: {row[4]} ({row[4]/total_new*100:.0f}%)")
            print(f"   Com lag time: {row[5]} ({row[5]/total_new*100:.0f}%)")
            print(f"   Com limite > 0: {row[6]} ({row[6]/total_new*100:.0f}%)")
        else:
            print(f"\n5. ❌ Nenhum registro novo (apenas recuperados)")
        
        # 6. Últimos 5 registros
        r = await session.execute(text("""
            SELECT audited_at, home_team || ' vs ' || away_team as jogo,
                   market_type || ' ' || line || ' ' || side as mercado,
                   status, websocket_odd, betslip_odd, is_live, audit_version
            FROM betslip_audit_results 
            ORDER BY audited_at DESC 
            LIMIT 5
        """))
        rows = r.fetchall()
        if rows:
            print(f"\n6. Últimos 5 registros:")
            for row in rows:
                live = "LIVE" if row[6] else "PRE" if row[6] is not None else "?"
                print(f"   [{row[7]}] {row[0]} | {row[1][:30]} | {row[2]} | {row[3]} | ws={row[4]} bs={row[5]} | {live}")
    
    await db.close()
    
    print(f"\n{'=' * 60}")
    print("FIM DA VERIFICAÇÃO")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
