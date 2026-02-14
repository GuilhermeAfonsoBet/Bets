#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recupera dados de auditoria H3B dos arquivos de log e insere no banco.

Parseia o output do audit_h3b_betslip.py nos logs para reconstruir
os registros que não foram salvos por causa do bug get_session().
"""

import asyncio
import re
import sys
import glob
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, '.')

from sqlalchemy import text
from storage.database import Database
from storage.models_hypothesis import BetslipAuditResult


def parse_audit_blocks(log_content: str) -> list:
    """Parseia blocos de auditoria do log."""
    results = []
    
    # Regex para encontrar blocos de auditoria
    # Cada bloco começa com ">>> AUDITANDO H3B"
    blocks = re.split(r'>>> AUDITANDO H3B', log_content)
    
    for block in blocks[1:]:  # Pula o primeiro (antes do primeiro >>>)
        try:
            result = parse_single_block(block)
            if result:
                results.append(result)
        except Exception as e:
            continue
    
    return results


def parse_single_block(block: str) -> dict:
    """Parseia um único bloco de auditoria."""
    r = {}
    
    # Direção e match info
    # (UP): Valencia vs Real Madrid
    # (UP) [PRE-MATCH]: Valencia vs Real Madrid
    m = re.search(r'\((\w+)\)(?:\s*\[([^\]]+)\])?\s*:\s*(.+?)(?:\n|$)', block)
    if m:
        r['reversal_direction'] = m.group(1).lower()
        live_label = m.group(2)
        r['match_info'] = m.group(3).strip()
        
        if live_label:
            r['is_live'] = live_label == "IN-MATCH"
        else:
            r['is_live'] = None
    else:
        return None
    
    # Event ID
    m = re.search(r'Event ID:\s*(.+)', block)
    if m:
        r['event_id'] = m.group(1).strip()
    
    # Mercado: AH 0 home (full_time)
    m = re.search(r'Mercado:\s*(\w+)\s+([\w.,+-]+)\s+(\w+)\s+\((\w+)\)', block)
    if m:
        r['market_type'] = m.group(1)
        r['line'] = m.group(2)
        r['side'] = m.group(3)
        r['market_period'] = m.group(4)
    
    # Odd WebSocket
    m = re.search(r'Odd WebSocket:\s*([\d.]+)', block)
    if m:
        r['websocket_odd'] = float(m.group(1))
    
    # Teams from match_info
    if 'match_info' in r and ' vs ' in r['match_info']:
        parts = r['match_info'].split(' vs ')
        r['home_team'] = parts[0].strip()
        r['away_team'] = parts[1].strip() if len(parts) > 1 else ''
    
    # Betslip Best Odd
    m = re.search(r'Betslip Best Odd:\s*([\d.]+)', block)
    if m:
        r['betslip_odd'] = float(m.group(1))
    
    # Betslip Limite
    m = re.search(r'Betslip Limite:\s*\$?([\d,.]+)', block)
    if m:
        r['betslip_limit'] = float(m.group(1).replace(',', ''))
    
    # Diferença
    m = re.search(r'Diferença:\s*([+-]?[\d.]+)%', block)
    if m:
        r['difference_pct'] = float(m.group(1))
    
    # Status
    m = re.search(r'Status:\s*(\w+)', block)
    if m:
        r['status'] = m.group(1)
    
    # Se não tem status mas tem indicadores
    if 'status' not in r:
        if 'GAME_NOT_FOUND' in block:
            r['status'] = 'GAME_NOT_FOUND'
        elif 'LINE_NOT_AVAILABLE' in block or 'LINHA NÃO DISPONÍVEL' in block:
            r['status'] = 'LINE_NOT_AVAILABLE'
        elif 'EXTRACT_FAILED' in block:
            r['status'] = 'EXTRACT_FAILED'
        elif 'ERROR' in block:
            r['status'] = 'ERROR'
        else:
            r['status'] = 'UNKNOWN'
    
    # LAG total
    m = re.search(r'TOTAL:\s*(\d+)ms', block)
    if m:
        r['audit_total_duration_ms'] = int(m.group(1))
    
    # LAG detecção → clique
    m = re.search(r'Detecção → Clique:\s*(\d+)ms', block)
    if m:
        r['lag_detection_to_click_ms'] = int(m.group(1))
    
    # LAG betslip
    m = re.search(r'Betslip abrir:\s*(\d+)ms.*?Extrair:\s*(\d+)ms', block)
    if m:
        r['lag_click_to_betslip_ms'] = int(m.group(1)) + int(m.group(2))
    
    # Precisa ter pelo menos websocket_odd e status
    if 'websocket_odd' not in r or 'status' not in r:
        return None
    
    # Calcula difference_absolute se tiver ambas as odds
    if 'betslip_odd' in r and 'websocket_odd' in r:
        r['difference_absolute'] = r['betslip_odd'] - r['websocket_odd']
        if 'difference_pct' not in r:
            r['difference_pct'] = (r['difference_absolute'] / r['websocket_odd']) * 100
    
    return r


async def main():
    print("=" * 60)
    print("RECUPERAÇÃO DE DADOS DE AUDITORIA DOS LOGS")
    print("=" * 60)
    
    # Encontra todos os arquivos de log
    log_dir = Path("logs")
    log_files = sorted(glob.glob(str(log_dir / "audit_h3b*.log")))
    
    if not log_files:
        print("Nenhum arquivo de log encontrado em logs/audit_h3b*.log")
        return
    
    print(f"\nArquivos de log encontrados: {len(log_files)}")
    for f in log_files:
        print(f"  {f}")
    
    # Parseia todos os logs
    all_results = []
    for log_file in log_files:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        results = parse_audit_blocks(content)
        print(f"\n  {log_file}: {len(results)} auditorias encontradas")
        all_results.extend(results)
    
    # Remove duplicatas (mesmo event_id + line + side)
    seen = set()
    unique_results = []
    for r in all_results:
        key = f"{r.get('event_id', '')}|{r.get('market_type', '')}|{r.get('line', '')}|{r.get('side', '')}"
        if key not in seen:
            seen.add(key)
            unique_results.append(r)
    
    print(f"\nTotal parseado: {len(all_results)}")
    print(f"Únicos (sem duplicatas): {len(unique_results)}")
    
    # Estatísticas rápidas
    with_betslip = [r for r in unique_results if 'betslip_odd' in r]
    without_betslip = [r for r in unique_results if 'betslip_odd' not in r]
    
    print(f"\n  Com betslip extraído: {len(with_betslip)}")
    print(f"  Sem betslip: {len(without_betslip)}")
    
    by_status = {}
    for r in unique_results:
        s = r.get('status', 'UNKNOWN')
        by_status[s] = by_status.get(s, 0) + 1
    print(f"\n  Por status:")
    for s, n in sorted(by_status.items(), key=lambda x: -x[1]):
        print(f"    {s}: {n}")
    
    # Insere no banco
    db = Database()
    await db.connect()
    
    # Migration: garante coluna is_live
    try:
        async with db.engine.begin() as conn:
            await conn.execute(
                text("ALTER TABLE betslip_audit_results ADD COLUMN IF NOT EXISTS is_live BOOLEAN")
            )
    except:
        pass
    
    inserted = 0
    errors = 0
    
    async with db.async_session() as session:
        for r in unique_results:
            try:
                is_valid = r.get('betslip_odd') is not None
                bet_desc = f"{r.get('market_type', '')} {r.get('line', '')} {r.get('side', '')} {r.get('market_period', '')}"
                
                record = BetslipAuditResult(
                    hypothesis_type="H3B",
                    event_id=r.get('event_id', ''),
                    sport="football",
                    home_team=r.get('home_team', ''),
                    away_team=r.get('away_team', ''),
                    match_info=r.get('match_info', ''),
                    market_type=r.get('market_type', 'AH'),
                    market_period=r.get('market_period', 'full_time'),
                    line=r.get('line', ''),
                    side=r.get('side', ''),
                    bet_description=bet_desc,
                    websocket_odd=r.get('websocket_odd', 0),
                    betslip_odd=r.get('betslip_odd'),
                    difference_pct=r.get('difference_pct'),
                    difference_absolute=r.get('difference_absolute'),
                    betslip_limit=r.get('betslip_limit'),
                    status=r.get('status', 'UNKNOWN'),
                    is_valid_opportunity=is_valid,
                    is_live=r.get('is_live'),
                    reversal_direction=r.get('reversal_direction', ''),
                    lag_detection_to_click_ms=r.get('lag_detection_to_click_ms'),
                    lag_click_to_betslip_ms=r.get('lag_click_to_betslip_ms'),
                    audit_total_duration_ms=r.get('audit_total_duration_ms'),
                    audit_version="v1.0-recovered",
                )
                session.add(record)
                inserted += 1
                
            except Exception as e:
                errors += 1
                print(f"  Erro: {e}")
        
        await session.commit()
    
    await db.close()
    
    print(f"\n{'=' * 60}")
    print(f"RESULTADO")
    print(f"{'=' * 60}")
    print(f"  Inseridos no banco: {inserted}")
    print(f"  Erros: {errors}")
    print(f"\n  Agora rode: python analyze_h3b_websocket_vs_betslip.py")


if __name__ == "__main__":
    asyncio.run(main())
