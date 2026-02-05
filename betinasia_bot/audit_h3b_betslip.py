# -*- coding: utf-8 -*-
"""
Auditoria H3B: WebSocket vs Best Odd do Betslip

Compara as odds de eventos H3B (reversão temporal UP) coletadas via WebSocket 
com a BEST ODD real exibida no painel do betslip.

Foco: Reversão UP (odd subiu = melhorou)
"""

import asyncio
import json
import sys
import re
import time
from datetime import datetime, timezone
from typing import Optional, Dict, List
from dataclasses import dataclass
from loguru import logger

sys.path.insert(0, '.')

from scraper.betinasia import BetinAsiaScraper
from scraper.betslip_extractor import BetslipExtractor, BetslipData
from hypothesis.detectors import HypothesisDetector
from storage.database import Database
from storage.models_hypothesis import BetslipAuditResult


@dataclass
class AuditResult:
    """Resultado de uma auditoria."""
    # Identificação
    timestamp: datetime
    match_info: str
    event_id: str
    
    # Jogo
    home_team: str = ""
    away_team: str = ""
    league: str = ""
    match_start_time: Optional[datetime] = None
    
    # Mercado
    market_type: str = ""
    market_period: str = "full_time"  # full_time, half_time
    line: str = ""
    side: str = ""
    
    # Odds
    websocket_odd: float = 0.0
    betslip_best_odd: Optional[float] = None
    betslip_limit: Optional[float] = None
    difference_pct: Optional[float] = None
    difference_absolute: Optional[float] = None
    
    # Status
    status: str = ""
    reversal_direction: str = ""  # "up" ou "down"
    
    # Timing/Lag GRANULAR (em milissegundos)
    hypothesis_detected_at: Optional[datetime] = None
    
    # Lag desde detecção até início da auditoria
    lag_queue_wait_ms: Optional[int] = None  # Tempo esperando na fila (se houver outros eventos antes)
    
    # Lags dentro da auditoria
    lag_find_game_ms: Optional[int] = None  # Tempo para encontrar o jogo na página
    lag_expand_lines_ms: Optional[int] = None  # Tempo para expandir linhas
    lag_click_odd_ms: Optional[int] = None  # Tempo para clicar na odd
    lag_betslip_open_ms: Optional[int] = None  # Tempo para betslip abrir
    lag_extract_data_ms: Optional[int] = None  # Tempo para extrair dados do betslip
    
    # Totais
    lag_detection_to_click_ms: Optional[int] = None  # Total: detecção → clique
    lag_click_to_betslip_ms: Optional[int] = None  # Total: clique → dados extraídos
    audit_total_duration_ms: Optional[int] = None  # Total da auditoria
    
    # Debug
    betslip_data: Optional[BetslipData] = None


class H3BAuditor:
    """Auditor que compara WebSocket vs Betslip Best Odd para eventos H3B."""
    
    FOOTBALL_URL = "https://black.betinasia.com/sportsbook/football"
    
    # Filtro de linhas extremas (volatilidade alta)
    MAX_AH_LINE = 5.0
    
    def __init__(self, num_audits: int = 50, direction_filter: str = "up", save_to_db: bool = True):
        """
        Args:
            num_audits: Número de auditorias a realizar
            direction_filter: "up" para reversão UP, "down" para DOWN, "all" para ambas
            save_to_db: Se True, salva resultados no banco de dados
        """
        self.scraper: Optional[BetinAsiaScraper] = None
        self.extractor: Optional[BetslipExtractor] = None
        self._ws_messages: List[str] = []
        self.hypothesis_detector = HypothesisDetector()
        self.num_audits = num_audits
        self.direction_filter = direction_filter
        self.save_to_db = save_to_db
        self.audit_results: List[AuditResult] = []
        self.events_processed = 0
        self.h3b_events_detected = 0
        self.db: Optional[Database] = None
        
    async def start(self):
        """Inicia o auditor."""
        self.scraper = BetinAsiaScraper()
        await self.scraper.start()
        await self.scraper.login()
        
        self.extractor = BetslipExtractor(self.scraper._page)
        self.scraper._page.on('websocket', self._on_websocket)
        
        # Conecta ao banco se necessário
        if self.save_to_db:
            self.db = Database()
            await self.db.connect()
            print("Conectado ao banco de dados")
        
        print("Auditor H3B iniciado e logado")
        
    async def close(self):
        """Fecha o auditor."""
        if self.scraper:
            await self.scraper.close()
        if self.db:
            await self.db.close()
            
    def _on_websocket(self, ws):
        """Callback quando WebSocket conecta."""
        ws.on('framereceived', lambda data: self._ws_messages.append(str(data)))
    
    async def _save_result_to_db(self, result: AuditResult, h3b: dict):
        """Salva resultado da auditoria no banco de dados."""
        if not self.db:
            return
        
        try:
            # Determina se é uma oportunidade válida (não é falso positivo)
            # Regra: conseguiu extrair odd do betslip = oportunidade existe na prática
            # Falso positivo = existe no WebSocket mas não existe no betslip real
            is_valid = result.betslip_best_odd is not None
            
            # Extrai texto bruto do betslip para debug
            raw_text = None
            if result.betslip_data:
                raw_text = result.betslip_data.raw_text
            
            # Monta descrição da aposta
            bet_description = f"{result.market_type} {result.line} {result.side} {result.market_period}"
            
            audit_record = BetslipAuditResult(
                # Identificação
                hypothesis_type="H3B",
                event_id=result.event_id,
                
                # Informações do jogo
                sport="football",
                home_team=result.home_team,
                away_team=result.away_team,
                match_info=result.match_info,
                
                # Mercado/Aposta
                market_type=result.market_type,
                market_period=result.market_period,
                line=result.line,
                side=result.side,
                bet_description=bet_description,
                
                # Odds comparação
                websocket_odd=result.websocket_odd,
                betslip_odd=result.betslip_best_odd,
                difference_pct=result.difference_pct,
                difference_absolute=result.difference_absolute,
                
                # Limites
                betslip_limit=result.betslip_limit,
                
                # Status
                status=result.status,
                is_valid_opportunity=is_valid,
                
                # Contexto da hipótese
                reversal_direction=result.reversal_direction,
                
                # Timing/Lag
                hypothesis_detected_at=result.hypothesis_detected_at,
                audited_at=result.timestamp,
                lag_detection_to_click_ms=result.lag_detection_to_click_ms,
                lag_click_to_betslip_ms=result.lag_click_to_betslip_ms,
                audit_total_duration_ms=result.audit_total_duration_ms,
                
                # Versionamento
                audit_version="v1.0",
                
                # Debug
                raw_betslip_text=raw_text
            )
            
            async with self.db.get_session() as session:
                session.add(audit_record)
                await session.commit()
                
        except Exception as e:
            logger.debug(f"Erro ao salvar auditoria no banco: {e}")

    async def run_audit(self):
        """Executa ciclos de auditoria."""
        await self.start()
        
        direction_label = {
            "up": "REVERSÃO UP (odd subiu)",
            "down": "REVERSÃO DOWN (odd desceu)",
            "all": "TODAS AS REVERSÕES"
        }.get(self.direction_filter, self.direction_filter)
        
        print("=" * 70)
        print("AUDITORIA H3B: WEBSOCKET vs BEST ODD DO BETSLIP")
        print("=" * 70)
        print(f"""
Este script compara:
- Odd de eventos H3B (reversão temporal) via WebSocket
- Best Odd do Betslip ("Todos Os Agentes De Apostas" → MELHOR)

Filtro: {direction_label}

Processo:
1. Coleta odds via WebSocket
2. Quando detecta H3B {direction_label}, navega para o jogo
3. Clica na odd para abrir betslip
4. Extrai a best odd do painel
5. Compara e reporta diferença

Vou auditar {self.num_audits} eventos.
""")
        
        audited = set()
        
        # Métricas de timing
        timing_stats = {
            'page_load': [],
            'websocket_wait': [],
            'hypothesis_detection': [],
            'total_cycle': [],
        }
        
        try:
            while len(self.audit_results) < self.num_audits:
                cycle_start = time.time()
                self._ws_messages.clear()
                
                # === ETAPA 1: Carrega página ===
                load_start = time.time()
                await self.scraper._page.goto(self.FOOTBALL_URL)
                await self.scraper._page.wait_for_load_state("domcontentloaded")
                load_time = int((time.time() - load_start) * 1000)
                timing_stats['page_load'].append(load_time)
                
                # === ETAPA 2: Espera WebSocket coletar dados ===
                # 8 segundos para garantir que WebSocket popule todos os dados
                ws_start = time.time()
                await self.scraper._page.wait_for_timeout(8000)
                ws_time = int((time.time() - ws_start) * 1000)
                timing_stats['websocket_wait'].append(ws_time)
                
                # === ETAPA 3: Detecta hipóteses ===
                detect_start = time.time()
                h3b_events = await self._find_h3b_events(audited)
                detect_time = int((time.time() - detect_start) * 1000)
                timing_stats['hypothesis_detection'].append(detect_time)
                
                cycle_time = int((time.time() - cycle_start) * 1000)
                timing_stats['total_cycle'].append(cycle_time)
                
                if h3b_events:
                    print(f"\n    → {len(h3b_events)} H3B novos para auditar")
                    print(f"    [TIMING] Ciclo: {cycle_time}ms (load:{load_time}ms + ws:{ws_time}ms + detect:{detect_time}ms)")
                
                # === ETAPA 4: Audita cada evento IMEDIATAMENTE ===
                for h3b in h3b_events:
                    if len(self.audit_results) >= self.num_audits:
                        break
                    
                    result = await self._audit_event(h3b)
                    
                    self.audit_results.append(result)
                    audited.add(h3b['audit_key'])
                    
                    # Salva no banco
                    if self.save_to_db:
                        await self._save_result_to_db(result, h3b)
                
                print(f"\rProcessados: {self.events_processed} | "
                      f"H3B: {self.h3b_events_detected} | "
                      f"Auditados: {len(self.audit_results)}/{self.num_audits}", 
                      end="", flush=True)
                
                # Pausa entre ciclos
                await asyncio.sleep(3)
                
        finally:
            await self.close()
            
        self._print_results()
        
    async def _find_h3b_events(self, already_audited: set) -> List[dict]:
        """Encontra eventos H3B nas mensagens WebSocket."""
        events = {}
        h3b_list = []
        skipped_already_audited = 0
        skipped_wrong_direction = 0
        
        for msg in self._ws_messages:
            try:
                data = json.loads(msg)
                if not isinstance(data, list):
                    continue
                    
                for item in data:
                    if not isinstance(item, list) or len(item) < 2:
                        continue
                        
                    msg_type = item[0]
                    msg_meta = item[1]
                    msg_data = item[2] if len(item) > 2 else {}
                    
                    # Captura info do evento
                    if msg_type == 'event' and isinstance(msg_meta, list) and len(msg_meta) >= 2:
                        if msg_meta[0] == 'fb' and 'home' in msg_data:
                            events[msg_meta[1]] = {
                                'home': msg_data.get('home', ''),
                                'away': msg_data.get('away', ''),
                            }
                    
                    # Processa odds de AH e OU
                    if msg_type in ['offers_hcap', 'offers_event']:
                        if isinstance(msg_meta, list) and len(msg_meta) >= 3:
                            if msg_meta[1] == 'fb':
                                event_id = msg_meta[2]
                                
                                # Processa AH
                                if 'ah' in msg_data:
                                    ah_data = msg_data['ah']
                                    lines = []
                                    
                                    if isinstance(ah_data, list) and len(ah_data) >= 2:
                                        if isinstance(ah_data[0], (int, float)):
                                            lines = [ah_data]
                                        elif isinstance(ah_data[0], list):
                                            lines = ah_data
                                            
                                    for line_data in lines:
                                        if len(line_data) < 2:
                                            continue
                                            
                                        line = line_data[0]
                                        odds_list = line_data[1] if len(line_data) > 1 else []
                                        
                                        home_odds = 0
                                        away_odds = 0
                                        
                                        if isinstance(odds_list, list):
                                            for o in odds_list:
                                                if isinstance(o, list) and len(o) >= 2:
                                                    if o[0] == 'h':
                                                        home_odds = float(o[1])
                                                    elif o[0] == 'a':
                                                        away_odds = float(o[1])
                                        
                                        if home_odds > 0 and away_odds > 0:
                                            self.events_processed += 1
                                            
                                            det = self.hypothesis_detector.process_market_update(
                                                match_id=hash(event_id) % 1000000,
                                                market_type="AH",
                                                line=str(line),
                                                home_odd=home_odds,
                                                away_odd=away_odds,
                                            )
                                            
                                            for h3b in det.get("h3b_events", []):
                                                self.h3b_events_detected += 1
                                                
                                                # Filtro por direção
                                                direction = h3b.direction_after
                                                if self.direction_filter != "all" and direction != self.direction_filter:
                                                    skipped_wrong_direction += 1
                                                    continue
                                                
                                                # Filtro de linhas extremas (|AH| > 5)
                                                try:
                                                    line_val = abs(float(h3b.ah_line))
                                                    if line_val > 5:
                                                        # print(f"    Pulando linha extrema: AH {h3b.ah_line}")
                                                        continue
                                                except:
                                                    pass
                                                
                                                # Chave única
                                                audit_key = f"{event_id}|AH|{h3b.ah_line}|{h3b.side}"
                                                
                                                if audit_key not in already_audited:
                                                    info = events.get(event_id, {})
                                                    home_team = info.get('home', '?')
                                                    away_team = info.get('away', '?')
                                                    h3b_list.append({
                                                        'event_id': event_id,
                                                        'audit_key': audit_key,
                                                        'match_info': f"{home_team} vs {away_team}",
                                                        'home_team': home_team,
                                                        'away_team': away_team,
                                                        'market_type': 'AH',
                                                        'market_period': 'full_time',
                                                        'line': str(h3b.ah_line),
                                                        'side': h3b.side,
                                                        'websocket_odd': h3b.odd_at_reversal,
                                                        'direction': direction,
                                                        'detected_at': datetime.now(timezone.utc),  # Timestamp de detecção
                                                    })
                                                else:
                                                    skipped_already_audited += 1
                                
                                # Processa OU
                                if 'ou' in msg_data:
                                    ou_data = msg_data['ou']
                                    lines = []
                                    
                                    if isinstance(ou_data, list) and len(ou_data) >= 2:
                                        if isinstance(ou_data[0], (int, float)):
                                            lines = [ou_data]
                                        elif isinstance(ou_data[0], list):
                                            lines = ou_data
                                    
                                    for line_data in lines:
                                        if len(line_data) < 2:
                                            continue
                                        
                                        line = line_data[0]
                                        odds_list = line_data[1] if len(line_data) > 1 else []
                                        
                                        over_odds = 0
                                        under_odds = 0
                                        
                                        if isinstance(odds_list, list):
                                            for o in odds_list:
                                                if isinstance(o, list) and len(o) >= 2:
                                                    if o[0] == 'o':
                                                        over_odds = float(o[1])
                                                    elif o[0] == 'u':
                                                        under_odds = float(o[1])
                                        
                                        if over_odds > 0 and under_odds > 0:
                                            self.events_processed += 1
                                            
                                            det = self.hypothesis_detector.process_market_update(
                                                match_id=hash(event_id) % 1000000,
                                                market_type="OU",
                                                line=str(line),
                                                home_odd=over_odds,
                                                away_odd=under_odds,
                                            )
                                            
                                            for h3b in det.get("h3b_events", []):
                                                self.h3b_events_detected += 1
                                                
                                                direction = h3b.direction_after
                                                if self.direction_filter != "all" and direction != self.direction_filter:
                                                    skipped_wrong_direction += 1
                                                    continue
                                                
                                                audit_key = f"{event_id}|OU|{h3b.ah_line}|{h3b.side}"
                                                
                                                if audit_key not in already_audited:
                                                    info = events.get(event_id, {})
                                                    home_team = info.get('home', '?')
                                                    away_team = info.get('away', '?')
                                                    h3b_list.append({
                                                        'event_id': event_id,
                                                        'audit_key': audit_key,
                                                        'match_info': f"{home_team} vs {away_team}",
                                                        'home_team': home_team,
                                                        'away_team': away_team,
                                                        'market_type': 'OU',
                                                        'market_period': 'full_time',
                                                        'line': str(h3b.ah_line),
                                                        'side': h3b.side,
                                                        'websocket_odd': h3b.odd_at_reversal,
                                                        'direction': direction,
                                                        'detected_at': datetime.now(timezone.utc),
                                                    })
                                                else:
                                                    skipped_already_audited += 1
                                
                                # Processa AH Half-Time
                                if 'ah_ht' in msg_data:
                                    ah_ht_data = msg_data['ah_ht']
                                    lines = []
                                    
                                    if isinstance(ah_ht_data, list) and len(ah_ht_data) >= 2:
                                        if isinstance(ah_ht_data[0], (int, float)):
                                            lines = [ah_ht_data]
                                        elif isinstance(ah_ht_data[0], list):
                                            lines = ah_ht_data
                                            
                                    for line_data in lines:
                                        if len(line_data) < 2:
                                            continue
                                            
                                        line = line_data[0]
                                        odds_list = line_data[1] if len(line_data) > 1 else []
                                        
                                        home_odds = 0
                                        away_odds = 0
                                        
                                        if isinstance(odds_list, list):
                                            for o in odds_list:
                                                if isinstance(o, list) and len(o) >= 2:
                                                    if o[0] == 'h':
                                                        home_odds = float(o[1])
                                                    elif o[0] == 'a':
                                                        away_odds = float(o[1])
                                        
                                        if home_odds > 0 and away_odds > 0:
                                            self.events_processed += 1
                                            
                                            det = self.hypothesis_detector.process_market_update(
                                                match_id=hash(event_id) % 1000000,
                                                market_type="AH_HT",
                                                line=str(line),
                                                home_odd=home_odds,
                                                away_odd=away_odds,
                                            )
                                            
                                            for h3b in det.get("h3b_events", []):
                                                self.h3b_events_detected += 1
                                                
                                                direction = h3b.direction_after
                                                if self.direction_filter != "all" and direction != self.direction_filter:
                                                    skipped_wrong_direction += 1
                                                    continue
                                                
                                                # Sem filtro de linha extrema para HT
                                                
                                                audit_key = f"{event_id}|AH_HT|{h3b.ah_line}|{h3b.side}"
                                                
                                                if audit_key not in already_audited:
                                                    info = events.get(event_id, {})
                                                    home_team = info.get('home', '?')
                                                    away_team = info.get('away', '?')
                                                    h3b_list.append({
                                                        'event_id': event_id,
                                                        'audit_key': audit_key,
                                                        'match_info': f"{home_team} vs {away_team}",
                                                        'home_team': home_team,
                                                        'away_team': away_team,
                                                        'market_type': 'AH',
                                                        'market_period': 'half_time',
                                                        'line': str(h3b.ah_line),
                                                        'side': h3b.side,
                                                        'websocket_odd': h3b.odd_at_reversal,
                                                        'direction': direction,
                                                        'detected_at': datetime.now(timezone.utc),
                                                    })
                                                else:
                                                    skipped_already_audited += 1
                                
                                # Processa OU Half-Time
                                if 'ou_ht' in msg_data:
                                    ou_ht_data = msg_data['ou_ht']
                                    lines = []
                                    
                                    if isinstance(ou_ht_data, list) and len(ou_ht_data) >= 2:
                                        if isinstance(ou_ht_data[0], (int, float)):
                                            lines = [ou_ht_data]
                                        elif isinstance(ou_ht_data[0], list):
                                            lines = ou_ht_data
                                    
                                    for line_data in lines:
                                        if len(line_data) < 2:
                                            continue
                                        
                                        line = line_data[0]
                                        odds_list = line_data[1] if len(line_data) > 1 else []
                                        
                                        over_odds = 0
                                        under_odds = 0
                                        
                                        if isinstance(odds_list, list):
                                            for o in odds_list:
                                                if isinstance(o, list) and len(o) >= 2:
                                                    if o[0] == 'o':
                                                        over_odds = float(o[1])
                                                    elif o[0] == 'u':
                                                        under_odds = float(o[1])
                                        
                                        if over_odds > 0 and under_odds > 0:
                                            self.events_processed += 1
                                            
                                            det = self.hypothesis_detector.process_market_update(
                                                match_id=hash(event_id) % 1000000,
                                                market_type="OU_HT",
                                                line=str(line),
                                                home_odd=over_odds,
                                                away_odd=under_odds,
                                            )
                                            
                                            for h3b in det.get("h3b_events", []):
                                                self.h3b_events_detected += 1
                                                
                                                direction = h3b.direction_after
                                                if self.direction_filter != "all" and direction != self.direction_filter:
                                                    skipped_wrong_direction += 1
                                                    continue
                                                
                                                audit_key = f"{event_id}|OU_HT|{h3b.ah_line}|{h3b.side}"
                                                
                                                if audit_key not in already_audited:
                                                    info = events.get(event_id, {})
                                                    home_team = info.get('home', '?')
                                                    away_team = info.get('away', '?')
                                                    h3b_list.append({
                                                        'event_id': event_id,
                                                        'audit_key': audit_key,
                                                        'match_info': f"{home_team} vs {away_team}",
                                                        'home_team': home_team,
                                                        'away_team': away_team,
                                                        'market_type': 'OU',
                                                        'market_period': 'half_time',
                                                        'line': str(h3b.ah_line),
                                                        'side': h3b.side,
                                                        'websocket_odd': h3b.odd_at_reversal,
                                                        'direction': direction,
                                                        'detected_at': datetime.now(timezone.utc),
                                                    })
                                                else:
                                                    skipped_already_audited += 1
            except:
                continue
        
        if skipped_already_audited > 0:
            print(f"\n    (Pulou {skipped_already_audited} H3B já auditados)")
        if skipped_wrong_direction > 0:
            print(f"\n    (Pulou {skipped_wrong_direction} H3B com direção diferente de '{self.direction_filter}')")
                
        return h3b_list
    
    async def _audit_event(self, h3b: dict) -> AuditResult:
        """Audita um evento abrindo o betslip com medição granular de tempo."""
        # Extrai dados do evento
        event_id = h3b['event_id']
        match_info = h3b['match_info']
        home_team = h3b.get('home_team', '')
        away_team = h3b.get('away_team', '')
        market_type = h3b['market_type']
        market_period = h3b.get('market_period', 'full_time')
        line = h3b['line']
        side = h3b['side']
        ws_odd = h3b['websocket_odd']
        direction = h3b['direction']
        detected_at = h3b.get('detected_at', datetime.now(timezone.utc))
        
        # === TIMING: Início da auditoria ===
        audit_start = time.time()
        
        # Lag desde detecção até início da auditoria (tempo na fila)
        lag_queue_wait = int((datetime.now(timezone.utc) - detected_at).total_seconds() * 1000)
        
        print(f"\n\n>>> AUDITANDO H3B ({direction.upper()}): {match_info}")
        print(f"    Event ID: {event_id}")
        print(f"    Mercado: {market_type} {line} {side} ({market_period})")
        print(f"    Odd WebSocket: {ws_odd:.3f}")
        print(f"    [LAG] Fila (detecção → início auditoria): {lag_queue_wait}ms")
        
        # Variáveis de timing granular
        lag_find_game = 0
        lag_expand_lines = 0
        lag_click_odd = 0
        lag_betslip_open = 0
        lag_extract_data = 0
        
        try:
            page = self.scraper._page
            
            # Extrai nomes dos times (fallback)
            if not home_team or not away_team:
                teams = match_info.split(' vs ')
                home_team = teams[0].strip() if len(teams) > 0 else ""
                away_team = teams[1].strip() if len(teams) > 1 else ""
            
            print(f"    Buscando jogo: '{home_team}' vs '{away_team}'")
            
            # === TIMING: Buscar jogo (via campo de busca do site) ===
            find_game_start = time.time()
            
            # Busca o jogo usando o campo de busca do site (rápido: ~3s)
            game_found = await self._find_and_click_game(home_team, away_team)
            
            lag_find_game = int((time.time() - find_game_start) * 1000)
            
            print(f"    [LAG] Buscar jogo: {lag_find_game}ms")
            
            if not game_found:
                print(f"    ERRO: Jogo não encontrado")
                audit_total = int((time.time() - audit_start) * 1000)
                print(f"    [LAG TOTAL] {audit_total}ms")
                return AuditResult(
                    timestamp=datetime.now(timezone.utc),
                    match_info=match_info,
                    event_id=event_id,
                    home_team=home_team,
                    away_team=away_team,
                    market_type=market_type,
                    market_period=market_period,
                    line=line,
                    side=side,
                    websocket_odd=ws_odd,
                    status="GAME_NOT_FOUND",
                    reversal_direction=direction,
                    hypothesis_detected_at=detected_at,
                    lag_queue_wait_ms=lag_queue_wait,
                    lag_find_game_ms=lag_find_game,
                    audit_total_duration_ms=audit_total
                )
            
            await page.wait_for_timeout(500)
            
            # === TIMING: Expandir linhas ===
            expand_start = time.time()
            print(f"    Expandindo linhas...")
            await self._expand_all_lines()
            lag_expand_lines = int((time.time() - expand_start) * 1000)
            print(f"    [LAG] Expandir linhas: {lag_expand_lines}ms")
            
            # === TIMING: Clicar na odd ===
            click_start = time.time()
            print(f"    Clicando na odd {line} {side}...")
            
            line_display = line.replace(".", ",")
            
            # Tenta clicar com retry (5 tentativas para máxima chance de sucesso)
            click_result = None
            max_attempts = 5
            
            for attempt in range(max_attempts):
                click_result = await self._click_specific_odd(line_display, side, market_type)
                
                if click_result == True:
                    break
                
                remaining = max_attempts - attempt - 1
                if remaining > 0:
                    print(f"    Tentativa {attempt + 1}/{max_attempts} falhou, aguardando...")
                    await page.wait_for_timeout(2000)
                    
                    if attempt > 0 and attempt % 2 == 0:
                        print(f"    Recarregando página...")
                        await page.reload()
                        await page.wait_for_load_state("domcontentloaded")
                        await page.wait_for_timeout(2000)
                        
                        game_found = await self._find_and_click_game(home_team, away_team)
                        if not game_found:
                            continue
                        
                        await page.wait_for_timeout(1500)
                    
                    print(f"    Re-expandindo linhas...")
                    await self._expand_all_lines()
                    await page.wait_for_timeout(1000)
            
            lag_click_odd = int((time.time() - click_start) * 1000)
            lag_detection_to_click = lag_queue_wait + lag_find_game + lag_expand_lines + lag_click_odd
            print(f"    [LAG] Clicar odd: {lag_click_odd}ms")
            
            if click_result != True:
                audit_total = int((time.time() - audit_start) * 1000)
                print(f"    LINHA NÃO DISPONÍVEL após {max_attempts} tentativas")
                print(f"    [LAG TOTAL] {audit_total}ms")
                return AuditResult(
                    timestamp=datetime.now(timezone.utc),
                    match_info=match_info,
                    event_id=event_id,
                    home_team=home_team,
                    away_team=away_team,
                    market_type=market_type,
                    market_period=market_period,
                    line=line,
                    side=side,
                    websocket_odd=ws_odd,
                    status="LINE_NOT_AVAILABLE",
                    reversal_direction=direction,
                    hypothesis_detected_at=detected_at,
                    lag_queue_wait_ms=lag_queue_wait,
                    lag_find_game_ms=lag_find_game,
                    lag_expand_lines_ms=lag_expand_lines,
                    lag_click_odd_ms=lag_click_odd,
                    lag_detection_to_click_ms=lag_detection_to_click,
                    audit_total_duration_ms=audit_total
                )
            
            print(f"    [LAG] Detecção → Clique: {lag_detection_to_click}ms")
            
            # === TIMING: Espera betslip abrir ===
            betslip_open_start = time.time()
            await page.wait_for_timeout(2000)
            lag_betslip_open = int((time.time() - betslip_open_start) * 1000)
            
            # === TIMING: Extrai dados do betslip ===
            extract_start = time.time()
            betslip_data = await self.extractor.extract_best_odd()
            lag_extract_data = int((time.time() - extract_start) * 1000)
            
            lag_click_to_betslip = lag_betslip_open + lag_extract_data
            print(f"    [LAG] Betslip abrir: {lag_betslip_open}ms, Extrair: {lag_extract_data}ms")
            
            if not betslip_data:
                audit_total = int((time.time() - audit_start) * 1000)
                print(f"    ERRO: Não conseguiu extrair dados do betslip")
                print(f"    [LAG TOTAL] {audit_total}ms")
                return AuditResult(
                    timestamp=datetime.now(timezone.utc),
                    match_info=match_info,
                    event_id=event_id,
                    home_team=home_team,
                    away_team=away_team,
                    market_type=market_type,
                    market_period=market_period,
                    line=line,
                    side=side,
                    websocket_odd=ws_odd,
                    status="EXTRACT_FAILED",
                    reversal_direction=direction,
                    hypothesis_detected_at=detected_at,
                    lag_queue_wait_ms=lag_queue_wait,
                    lag_find_game_ms=lag_find_game,
                    lag_expand_lines_ms=lag_expand_lines,
                    lag_click_odd_ms=lag_click_odd,
                    lag_betslip_open_ms=lag_betslip_open,
                    lag_extract_data_ms=lag_extract_data,
                    lag_detection_to_click_ms=lag_detection_to_click,
                    lag_click_to_betslip_ms=lag_click_to_betslip,
                    audit_total_duration_ms=audit_total
                )
            
            best_odd = betslip_data.best_odd
            best_limit = betslip_data.best_limit
            
            diff_pct = ((best_odd - ws_odd) / ws_odd) * 100
            diff_abs = best_odd - ws_odd
            
            # Classificação por magnitude da diferença
            if abs(diff_pct) < 0.1:
                status = "IDENTICAL"
            elif abs(diff_pct) < 0.5:
                status = "OK"
            elif abs(diff_pct) < 2:
                status = "MINOR_DIFF"
            else:
                status = "MAJOR_DIFF"
            
            audit_total = int((time.time() - audit_start) * 1000)
            
            print(f"    ✅ Betslip Best Odd: {best_odd:.3f}")
            print(f"    💰 Betslip Limite: ${best_limit:,.0f}")
            print(f"    📊 Diferença: {diff_pct:+.2f}%")
            print(f"    📋 Status: {status}")
            print(f"    ⏱️  [LAG BREAKDOWN]")
            print(f"       Fila: {lag_queue_wait}ms")
            print(f"       Buscar jogo: {lag_find_game}ms")
            print(f"       Expandir linhas: {lag_expand_lines}ms")
            print(f"       Clicar odd: {lag_click_odd}ms")
            print(f"       Betslip abrir: {lag_betslip_open}ms")
            print(f"       Extrair dados: {lag_extract_data}ms")
            print(f"       TOTAL: {audit_total}ms")
            
            # Fecha betslip
            await self.extractor.close_betslip()
            
            return AuditResult(
                timestamp=datetime.now(timezone.utc),
                match_info=match_info,
                event_id=event_id,
                home_team=home_team,
                away_team=away_team,
                market_type=market_type,
                market_period=market_period,
                line=line,
                side=side,
                websocket_odd=ws_odd,
                betslip_best_odd=best_odd,
                betslip_limit=best_limit,
                difference_pct=diff_pct,
                difference_absolute=diff_abs,
                status=status,
                reversal_direction=direction,
                hypothesis_detected_at=detected_at,
                lag_queue_wait_ms=lag_queue_wait,
                lag_find_game_ms=lag_find_game,
                lag_expand_lines_ms=lag_expand_lines,
                lag_click_odd_ms=lag_click_odd,
                lag_betslip_open_ms=lag_betslip_open,
                lag_extract_data_ms=lag_extract_data,
                lag_detection_to_click_ms=lag_detection_to_click,
                lag_click_to_betslip_ms=lag_click_to_betslip,
                audit_total_duration_ms=audit_total,
                betslip_data=betslip_data
            )
            
        except Exception as e:
            logger.error(f"Erro: {e}")
            audit_total = int((time.time() - audit_start) * 1000)
            return AuditResult(
                timestamp=datetime.now(timezone.utc),
                match_info=match_info,
                event_id=event_id,
                home_team=home_team,
                away_team=away_team,
                market_type=market_type,
                market_period=market_period,
                line=line,
                side=side,
                websocket_odd=ws_odd,
                status=f"ERROR: {str(e)[:30]}",
                reversal_direction=direction,
                hypothesis_detected_at=detected_at,
                audit_total_duration_ms=audit_total
            )
    
    def _get_team_aliases(self, team_name: str) -> list:
        """Retorna aliases conhecidos para um time."""
        # Mapeamento de aliases comuns
        aliases = {
            # England
            "Wolves": ["Wolverhampton", "Wolves", "Wolverhampton Wanderers"],
            "Wolverhampton": ["Wolverhampton", "Wolves", "Wolverhampton Wanderers"],
            "Man United": ["Manchester United", "Man United", "Man Utd"],
            "Manchester United": ["Manchester United", "Man United", "Man Utd"],
            "Man City": ["Manchester City", "Man City"],
            "Manchester City": ["Manchester City", "Man City"],
            "Spurs": ["Tottenham", "Spurs", "Tottenham Hotspur"],
            "Tottenham": ["Tottenham", "Spurs", "Tottenham Hotspur"],
            # Spain
            "Athletic Bilbao": ["Athletic Bilbao", "Athletic Club", "Ath Bilbao", "Athletic"],
            "Athletic Club": ["Athletic Bilbao", "Athletic Club", "Ath Bilbao", "Athletic"],
            "Atletico Madrid": ["Atletico Madrid", "Atl Madrid", "Atlético Madrid"],
            "Real Madrid": ["Real Madrid", "R Madrid"],
            "Barcelona": ["Barcelona", "FC Barcelona", "Barça"],
            # France
            "PSG": ["PSG", "Paris Saint-Germain", "Paris SG", "Paris Saint Germain"],
            "Paris Saint-Germain": ["PSG", "Paris Saint-Germain", "Paris SG"],
            "Marseille": ["Marseille", "Olympique Marseille", "OM"],
            # Germany
            "Bayern": ["Bayern Munich", "Bayern München", "FC Bayern"],
            "Dortmund": ["Borussia Dortmund", "Dortmund", "BVB"],
            "Borussia Dortmund": ["Borussia Dortmund", "Dortmund", "BVB"],
            "Leverkusen": ["Bayer Leverkusen", "Leverkusen", "B Leverkusen"],
            # Brazil
            "Flamengo": ["Flamengo", "CR Flamengo"],
            "Palmeiras": ["Palmeiras", "SE Palmeiras"],
            "Corinthians": ["Corinthians", "SC Corinthians"],
        }
        
        # Procura por aliases
        for key, values in aliases.items():
            if key.lower() in team_name.lower() or team_name.lower() in key.lower():
                return values
        
        # Se não encontrou alias, retorna variantes do nome original
        words = team_name.split()
        variants = [team_name]
        if len(words) > 1:
            variants.append(words[0])  # Primeiro nome
            variants.append(" ".join(words[:2]))  # Dois primeiros nomes
        return variants
    
    async def _find_and_click_game(self, home_team: str, away_team: str) -> bool:
        """
        Encontra e clica num jogo usando a BUSCA do site.
        
        Fluxo rápido:
        1. Clica no campo de busca (topo da página)
        2. Digita o nome de um dos times
        3. Seleciona o jogo correto nos resultados
        
        Muito mais rápido que navegar/expandir todas as ligas (~3s vs ~47s).
        """
        page = self.scraper._page
        
        # Monta lista de termos de busca (tenta home primeiro, depois away)
        search_terms = []
        
        # Usa aliases para ter variantes
        home_aliases = self._get_team_aliases(home_team)
        away_aliases = self._get_team_aliases(away_team)
        
        # Prioriza o nome mais curto/simples de cada time
        for aliases in [home_aliases, away_aliases]:
            # Ordena por tamanho (nomes mais curtos primeiro = mais genéricos = melhor busca)
            sorted_aliases = sorted(set(aliases), key=len)
            for alias in sorted_aliases:
                if alias and len(alias) >= 3 and alias not in search_terms:
                    search_terms.append(alias)
        
        for search_term in search_terms[:4]:  # Tenta até 4 variantes
            try:
                result = await self._search_and_click(search_term, home_team, away_team)
                if result:
                    return True
            except Exception as e:
                logger.debug(f"Erro na busca por '{search_term}': {e}")
                continue
        
        # Fallback: verifica se o jogo já está visível na página atual
        try:
            body_text = await page.inner_text("body")
            if ("Asian Handicap" in body_text or "Over/Under" in body_text):
                # Verifica se é o jogo certo
                home_found = any(alias in body_text for alias in home_aliases if len(alias) > 3)
                away_found = any(alias in body_text for alias in away_aliases if len(alias) > 3)
                if home_found or away_found:
                    print(f"    Jogo já visível na página atual")
                    return True
        except:
            pass
        
        print(f"    Jogo não encontrado via busca")
        return False
    
    async def _search_and_click(self, search_term: str, home_team: str, away_team: str) -> bool:
        """Usa o campo de busca do site para encontrar e navegar para o jogo."""
        page = self.scraper._page
        
        print(f"    Buscando: '{search_term}'")
        
        # === PASSO 1: Encontra e clica no campo de busca ===
        search_input = None
        
        # Tenta seletores comuns para o campo de busca
        search_selectors = [
            "input[type='text'][placeholder*='earch']",
            "input[type='text'][placeholder*='Search']",
            "input[type='search']",
            "input[placeholder*='league']",
            "input[placeholder*='game']",
            "input[autocomplete='none']",
            "input[name*='search']",
            "input[type='text']",
        ]
        
        for selector in search_selectors:
            try:
                elements = await page.query_selector_all(selector)
                for el in elements:
                    try:
                        if await el.is_visible():
                            search_input = el
                            break
                    except:
                        continue
                if search_input:
                    break
            except:
                continue
        
        if not search_input:
            # Tenta clicar em ícone/botão de busca para abrir o campo
            search_button_selectors = [
                "[class*='search']",
                "button:has-text('Search')",
                "[aria-label*='search']",
                "[aria-label*='Search']",
            ]
            for selector in search_button_selectors:
                try:
                    btn = await page.query_selector(selector)
                    if btn and await btn.is_visible():
                        await btn.click()
                        await page.wait_for_timeout(500)
                        # Tenta encontrar o input novamente
                        for s in search_selectors:
                            try:
                                search_input = await page.query_selector(s)
                                if search_input and await search_input.is_visible():
                                    break
                                search_input = None
                            except:
                                continue
                        if search_input:
                            break
                except:
                    continue
        
        if not search_input:
            print(f"    Campo de busca não encontrado")
            return False
        
        # === PASSO 2: Limpa e digita o termo de busca ===
        try:
            await search_input.click()
            await page.wait_for_timeout(300)
            
            # Limpa campo (seleciona tudo e apaga)
            await search_input.fill("")
            await page.wait_for_timeout(200)
            
            # Digita o termo de busca
            await search_input.type(search_term, delay=50)
            await page.wait_for_timeout(1500)  # Espera resultados aparecerem
            
        except Exception as e:
            logger.debug(f"Erro ao digitar no campo de busca: {e}")
            return False
        
        # === PASSO 3: Encontra e clica no resultado correto ===
        # Procura links/elementos que contenham ambos os times (ou pelo menos um)
        home_aliases = self._get_team_aliases(home_team)
        away_aliases = self._get_team_aliases(away_team)
        
        clicked = await page.evaluate("""
            (params) => {
                const homeAliases = params.homeAliases;
                const awayAliases = params.awayAliases;
                
                function textContainsTeam(text, aliases) {
                    const lower = text.toLowerCase();
                    for (const alias of aliases) {
                        if (alias.length >= 3 && lower.includes(alias.toLowerCase())) {
                            return true;
                        }
                    }
                    return false;
                }
                
                // Procura links (a) nos resultados da busca
                const links = document.querySelectorAll('a');
                let bestMatch = null;
                let bestScore = 0;
                
                for (const link of links) {
                    const text = link.innerText || '';
                    if (text.length < 5 || text.length > 200) continue;
                    
                    const hasHome = textContainsTeam(text, homeAliases);
                    const hasAway = textContainsTeam(text, awayAliases);
                    
                    // Score: 2 = ambos os times, 1 = um time + "vs" ou "v"
                    let score = 0;
                    if (hasHome && hasAway) score = 2;
                    else if ((hasHome || hasAway) && (text.includes(' vs ') || text.includes(' v '))) score = 1;
                    
                    if (score > bestScore) {
                        bestScore = score;
                        bestMatch = link;
                    }
                }
                
                if (bestMatch) {
                    bestMatch.click();
                    return { success: true, text: bestMatch.innerText.substring(0, 100), score: bestScore };
                }
                
                return { success: false };
            }
        """, {
            "homeAliases": home_aliases,
            "awayAliases": away_aliases
        })
        
        if clicked and clicked.get('success'):
            print(f"    Busca: clicou em '{clicked.get('text', '?').strip()}'")
            
            # Espera a página do jogo carregar
            await page.wait_for_load_state("domcontentloaded")
            await page.wait_for_timeout(2000)
            
            # Verifica se carregou a página do jogo (deve ter Asian Handicap ou Over/Under)
            body_text = await page.inner_text("body")
            if "Asian Handicap" in body_text or "Over/Under" in body_text or "Handicap" in body_text:
                return True
            
            # Pode precisar de mais tempo
            await page.wait_for_timeout(2000)
            body_text = await page.inner_text("body")
            if "Asian Handicap" in body_text or "Over/Under" in body_text or "Handicap" in body_text:
                return True
            
            print(f"    Página carregou mas não tem odds de Asian Handicap")
            return False
        
        # Fecha a busca (ESC)
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(300)
        
        return False
    
    async def _expand_all_lines(self):
        """Expande todas as linhas clicando em 'Show all lines' via JavaScript (rápido)."""
        page = self.scraper._page
        
        try:
            # Clica em TODOS os botões "Show all" de uma vez via JavaScript
            result = await page.evaluate("""
                () => {
                    let clicked = 0;
                    
                    // Procura todos os elementos clicáveis com texto "Show all" / "Show all lines" / "Mostrar"
                    const allElements = document.querySelectorAll('span, button, div, a, [role="button"]');
                    
                    for (const el of allElements) {
                        const text = (el.innerText || '').trim().toLowerCase();
                        
                        if ((text === 'show all lines' || text === 'show all' || 
                             text === 'mostrar todas as linhas' || text === 'mostrar') &&
                            el.offsetParent !== null) {  // Visível
                            try {
                                el.click();
                                clicked++;
                            } catch(e) {}
                        }
                    }
                    
                    return clicked;
                }
            """)
            
            if result > 0:
                # Espera única para todas as seções carregarem
                await page.wait_for_timeout(1000)
                print(f"    Expandiu {result} seções")
            else:
                print(f"    Nenhum botão 'Show all' encontrado (pode já estar expandido)")
                    
        except Exception as e:
            logger.debug(f"Erro ao expandir linhas: {e}")
    
    async def _click_specific_odd(self, line: str, side: str, market_type: str = "AH") -> bool:
        """
        Clica numa odd específica para abrir o betslip.
        
        ESTRATÉGIA ROBUSTA (baseada na estrutura DOM real):
        1. Encontra a seção correta (Handicap Asiático ou Over/Under)
        2. Busca a LINHA específica pelo valor do handicap
        3. Dentro da linha, clica no lado correto por POSIÇÃO (Home/Away, Over/Under)
        
        NÃO busca pelo valor da odd (que pode mudar entre WebSocket e DOM).
        """
        page = self.scraper._page
        
        try:
            line_float = float(line.replace(",", "."))
            
            # Gera variantes da linha (diferentes formatos possíveis)
            line_variants = []
            if line_float == int(line_float):
                int_val = int(line_float)
                if int_val > 0:
                    line_variants.extend([f"+{int_val}", f"+{int_val},0", f"+{int_val}.0", str(int_val)])
                elif int_val < 0:
                    line_variants.extend([str(int_val), f"{int_val},0", f"{int_val}.0"])
                else:
                    line_variants.extend(["0", "+0", "0,0", "0.0"])
            else:
                # Linhas com decimal
                line_comma = line.replace(".", ",")
                line_dot = line.replace(",", ".")
                line_variants.append(line_comma)
                line_variants.append(line_dot)
                if line_float > 0:
                    line_variants.append("+" + line_comma)
                    line_variants.append("+" + line_dot)
            
            print(f"    Procurando linha: {line_variants}")
            
            # Determina seção e labels (suporta PT e EN)
            if market_type == "OU":
                section_names = ["Over/Under", "Mais/Menos"]
                home_label = "Over"
                away_label = "Under"
            else:
                section_names = ["Asian Handicap", "Handicap Asiático", "Handicap"]
                home_label = "Home"
                away_label = "Away"
            
            target_label = home_label if side in ['home', 'over'] else away_label
            
            # === ESTRATÉGIA PRINCIPAL: JavaScript robusto baseado na estrutura DOM real ===
            # Estrutura: LINHA | Home | ODD_HOME | Away | ODD_AWAY
            # Os elementos clicáveis são divs ao lado dos labels Home/Away
            
            clicked = await page.evaluate("""
                (params) => {
                    const lineVariants = params.lineVariants;
                    const side = params.side;
                    const marketType = params.marketType;
                    const sectionNames = params.sectionNames;
                    const homeLabel = params.homeLabel;
                    const awayLabel = params.awayLabel;
                    
                    // Função para normalizar texto de linha
                    function normalizeLineText(text) {
                        return text.trim().replace(/\\s+/g, '').replace('.', ',');
                    }
                    
                    // Função para verificar se texto contém algum dos nomes de seção
                    function matchesSection(text) {
                        for (const name of sectionNames) {
                            if (text.includes(name)) return true;
                        }
                        return false;
                    }
                    
                    // Encontra a seção correta (Asian Handicap ou Over/Under) - suporta PT e EN
                    let sectionContainer = null;
                    const headers = document.querySelectorAll('div, span, h3, h4');
                    
                    for (const h of headers) {
                        const text = (h.innerText || '').trim();
                        // Procura por nomes de seção em PT ou EN
                        if (matchesSection(text) || text.includes('Handicap') || text.includes('Asian') || 
                            (marketType === 'OU' && (text.includes('Over') || text.includes('Under')))) {
                            // Encontra o container pai que contém todas as linhas
                            let parent = h.parentElement;
                            for (let i = 0; i < 10 && parent; i++) {
                                const parentText = parent.innerText || '';
                                // Verifica se contém múltiplas linhas de odds
                                if (parentText.includes(homeLabel) && parentText.includes(awayLabel)) {
                                    sectionContainer = parent;
                                    break;
                                }
                                parent = parent.parentElement;
                            }
                            if (sectionContainer) break;
                        }
                    }
                    
                    if (!sectionContainer) {
                        // Fallback: usa todo o body
                        sectionContainer = document.body;
                    }
                    
                    // Encontra TODOS os elementos que podem ser linhas de handicap
                    // Procura por spans/divs com texto que seja uma das variantes da linha
                    const allElements = sectionContainer.querySelectorAll('span, div');
                    
                    let foundLineElement = null;
                    let foundLineText = null;
                    
                    for (const el of allElements) {
                        const elText = (el.innerText || '').trim();
                        
                        // Verifica se este elemento é a linha que procuramos
                        // IMPORTANTE: O texto deve ser EXATAMENTE a linha, não pode conter outras coisas
                        let isLineMatch = false;
                        for (const variant of lineVariants) {
                            if (elText === variant || normalizeLineText(elText) === normalizeLineText(variant)) {
                                isLineMatch = true;
                                foundLineText = elText;
                                break;
                            }
                        }
                        
                        if (!isLineMatch) continue;
                        
                        // Verifica se este elemento é pequeno (só a linha, não um container grande)
                        if (elText.length > 10) continue;
                        
                        foundLineElement = el;
                        
                        // Encontrou a linha! Agora busca o container ROW que contém Home/Away
                        let rowContainer = el.parentElement;
                        for (let i = 0; i < 6 && rowContainer; i++) {
                            const rowText = rowContainer.innerText || '';
                            
                            // Verifica se este container tem Home e Away (ou Over e Under)
                            // E não é muito grande (deve ser só uma linha)
                            if (rowText.includes(homeLabel) && rowText.includes(awayLabel) && 
                                rowText.split('\\n').length < 15) {
                                
                                // Verifica se contém a nossa linha específica
                                let hasOurLine = false;
                                for (const variant of lineVariants) {
                                    if (rowText.includes(variant)) {
                                        hasOurLine = true;
                                        break;
                                    }
                                }
                                if (!hasOurLine) {
                                    rowContainer = rowContainer.parentElement;
                                    continue;
                                }
                                
                                // Encontra os elementos clicáveis (odds) dentro desta linha
                                // Os elementos clicáveis são divs filhos com odds numéricas
                                const clickableElements = rowContainer.querySelectorAll('div, span');
                                const oddElements = [];
                                
                                for (const child of clickableElements) {
                                    const childText = (child.innerText || '').trim();
                                    
                                    // Verifica se é uma odd (formato X.XXX ou X,XXX)
                                    // Deve ser APENAS a odd, não um container
                                    if (/^\\d+[.,]\\d{2,3}$/.test(childText) && childText.length < 10) {
                                        const rect = child.getBoundingClientRect();
                                        if (rect.width > 0 && rect.height > 0 && rect.width < 200) {
                                            oddElements.push({
                                                el: child,
                                                x: rect.x,
                                                text: childText
                                            });
                                        }
                                    }
                                }
                                
                                // Remove duplicatas (mesmo x, mesmo texto)
                                const uniqueOdds = [];
                                const seenKeys = new Set();
                                for (const odd of oddElements) {
                                    const key = Math.round(odd.x) + '|' + odd.text;
                                    if (!seenKeys.has(key)) {
                                        seenKeys.add(key);
                                        uniqueOdds.push(odd);
                                    }
                                }
                                
                                if (uniqueOdds.length >= 2) {
                                    // Ordena por posição X (esquerda para direita)
                                    uniqueOdds.sort((a, b) => a.x - b.x);
                                    
                                    // Home/Over = primeiro (esquerda), Away/Under = segundo (direita)
                                    const targetIdx = (side === 'home' || side === 'over') ? 0 : 1;
                                    const targetEl = uniqueOdds[targetIdx];
                                    
                                    if (targetEl) {
                                        // Scroll para o elemento
                                        targetEl.el.scrollIntoView({ behavior: 'instant', block: 'center' });
                                        
                                        // Tenta clicar - primeiro no pai, depois direto
                                        try {
                                            const parent = targetEl.el.parentElement;
                                            if (parent) {
                                                parent.click();
                                                return { 
                                                    success: true, 
                                                    clickedOdd: targetEl.text, 
                                                    method: 'parent',
                                                    lineFound: foundLineText,
                                                    allOdds: uniqueOdds.map(o => o.text)
                                                };
                                            }
                                        } catch (e) {}
                                        
                                        try {
                                            targetEl.el.click();
                                            return { 
                                                success: true, 
                                                clickedOdd: targetEl.text, 
                                                method: 'direct',
                                                lineFound: foundLineText,
                                                allOdds: uniqueOdds.map(o => o.text)
                                            };
                                        } catch (e) {}
                                    }
                                }
                            }
                            rowContainer = rowContainer.parentElement;
                        }
                    }
                    
                    return { 
                        success: false, 
                        reason: 'LINE_NOT_FOUND',
                        foundLineElement: foundLineElement ? true : false,
                        foundLineText: foundLineText
                    };
                }
            """, {
                "lineVariants": line_variants,
                "side": side,
                "marketType": market_type,
                "sectionNames": section_names,
                "homeLabel": home_label,
                "awayLabel": away_label
            })
            
            if clicked and clicked.get('success'):
                line_found = clicked.get('lineFound', '?')
                all_odds = clicked.get('allOdds', [])
                print(f"    Linha encontrada: {line_found}")
                print(f"    Odds na linha: {all_odds}")
                print(f"    Clicou na odd {clicked.get('clickedOdd')} ({clicked.get('method')})")
                await page.wait_for_timeout(1500)
                return True
            
            # === FALLBACK: Busca alternativa usando seletores Playwright ===
            print(f"    Estratégia principal falhou, tentando fallback...")
            
            for variant in line_variants:
                try:
                    # Busca elementos com o texto exato da linha
                    line_elements = await page.query_selector_all(f"span:text-is('{variant}'), div:text-is('{variant}')")
                    
                    for line_el in line_elements:
                        try:
                            # Navega para o container pai que tem Home/Away
                            for _ in range(5):
                                parent = await line_el.evaluate_handle("el => el.parentElement")
                                parent_text = await parent.evaluate("el => el.innerText || ''")
                                
                                if home_label in parent_text and away_label in parent_text:
                                    # Encontrou a linha! Busca as odds
                                    odd_spans = await parent.evaluate_handle(
                                        "el => Array.from(el.querySelectorAll('span, div')).filter(s => /^\\d+[.,]\\d{2,3}$/.test(s.innerText.trim()))"
                                    )
                                    
                                    count = await odd_spans.evaluate("arr => arr.length")
                                    
                                    if count >= 2:
                                        # Clica no correto baseado na posição
                                        idx = 0 if side in ['home', 'over'] else 1
                                        
                                        result = await odd_spans.evaluate(f"""
                                            (arr) => {{
                                                const el = arr[{idx}];
                                                if (el) {{
                                                    el.scrollIntoView({{ behavior: 'instant', block: 'center' }});
                                                    try {{
                                                        el.parentElement.click();
                                                        return {{ success: true, method: 'fallback-parent' }};
                                                    }} catch {{}}
                                                    try {{
                                                        el.click();
                                                        return {{ success: true, method: 'fallback-direct' }};
                                                    }} catch {{}}
                                                }}
                                                return {{ success: false }};
                                            }}
                                        """)
                                        
                                        if result and result.get('success'):
                                            print(f"    Fallback: clicou com sucesso ({result.get('method')})")
                                            await page.wait_for_timeout(1500)
                                            return True
                                    break
                                
                                line_el = parent
                        except:
                            continue
                except:
                    continue
            
            # Debug: mostra linhas disponíveis
            body_text = await page.inner_text("body")
            available_lines = set()
            for l in body_text.split('\n'):
                l_stripped = l.strip()
                if re.match(r'^[+-]?\d+([.,]\d{1,2})?$', l_stripped) and len(l_stripped) < 8:
                    available_lines.add(l_stripped)
            
            def line_sort_key(x):
                try:
                    return float(x.replace(',', '.').replace('+', ''))
                except:
                    return 0
            sorted_lines = sorted(available_lines, key=line_sort_key)
            
            print(f"    Linha não encontrada")
            print(f"    Buscando: {line_variants}")
            print(f"    Linhas AH disponíveis: {sorted_lines[:20]}")
            
            return False
            
        except Exception as e:
            logger.error(f"Erro ao clicar: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def _print_results(self):
        """Imprime resultados."""
        print("\n\n" + "=" * 70)
        print("RESULTADOS DA AUDITORIA H3B")
        print("=" * 70)
        
        counts = {"IDENTICAL": 0, "OK": 0, "MINOR_DIFF": 0, "MAJOR_DIFF": 0, "LINE_NOT_AVAILABLE": 0}
        diffs = []
        errors = 0
        
        by_direction = {"up": [], "down": []}
        
        for r in self.audit_results:
            print(f"\n{r.match_info}")
            print(f"  {r.market_type} {r.line} {r.side} (Reversão {r.reversal_direction.upper()})")
            print(f"  WebSocket:     {r.websocket_odd:.3f}")
            
            if r.betslip_best_odd:
                print(f"  Betslip Best:  {r.betslip_best_odd:.3f}")
                print(f"  Limite:        ${r.betslip_limit:,.0f}")
                print(f"  Diferença:     {r.difference_pct:+.2f}%")
                diffs.append(abs(r.difference_pct))
                
            emoji = {
                "IDENTICAL": "✅", "OK": "✅", 
                "MINOR_DIFF": "⚠️", "MAJOR_DIFF": "❌",
                "LINE_NOT_AVAILABLE": "📉"
            }.get(r.status, "❓")
            print(f"  Status:        {emoji} {r.status}")
            
            if r.status in counts:
                counts[r.status] += 1
            else:
                errors += 1
            
            by_direction[r.reversal_direction].append(r)
        
        print("\n" + "=" * 70)
        print("RESUMO")
        print("=" * 70)
        total = len(self.audit_results)
        print(f"  ✅ Idênticas/OK (diff < 0.5%): {counts['IDENTICAL'] + counts['OK']}/{total}")
        print(f"  ⚠️ Diff pequena (0.5-2%):      {counts['MINOR_DIFF']}/{total}")
        print(f"  ❌ Diff grande (>2%):          {counts['MAJOR_DIFF']}/{total}")
        print(f"  📉 Linha indisponível:         {counts['LINE_NOT_AVAILABLE']}/{total}")
        print(f"  ❓ Outros erros:               {errors}/{total}")
        
        if diffs:
            print(f"\n  Diferença média: {sum(diffs)/len(diffs):.3f}%")
            print(f"  Diferença máxima: {max(diffs):.3f}%")
        
        print("\n" + "=" * 70)
        print("TAXA DE OPORTUNIDADES REAIS - H3B")
        print("=" * 70)
        
        real_opportunities = counts['IDENTICAL'] + counts['OK'] + counts['MINOR_DIFF'] + counts['MAJOR_DIFF']
        false_positives = counts['LINE_NOT_AVAILABLE'] + errors
        
        if total > 0:
            real_rate = real_opportunities / total * 100
            false_rate = false_positives / total * 100
            
            print(f"\n  📊 TAXA DE SUCESSO: {real_rate:.1f}%")
            print(f"     ({real_opportunities} de {total} eventos H3B têm odds reais)")
            print(f"\n  📉 FALSOS POSITIVOS: {false_rate:.1f}%")
            print(f"     ({false_positives} de {total} eventos não existem de fato)")
            
            print(f"\n  💡 PARA ESTIMAR OPORTUNIDADES REAIS:")
            print(f"     Total H3B detectados × {real_rate:.1f}% = oportunidades apostáveis")
            
            if diffs:
                print(f"\n  📈 QUALIDADE DAS ODDS (quando existem):")
                print(f"     Diferença média WebSocket vs Betslip: {sum(diffs)/len(diffs):.3f}%")
                
                accurate = counts['IDENTICAL'] + counts['OK']
                if real_opportunities > 0:
                    accuracy_rate = accurate / real_opportunities * 100
                    print(f"     Taxa de precisão (diff < 0.5%): {accuracy_rate:.1f}%")
        
        # Breakdown por direção
        print("\n" + "=" * 70)
        print("BREAKDOWN POR DIREÇÃO DA REVERSÃO")
        print("=" * 70)
        
        for direction, results in by_direction.items():
            if not results:
                continue
            
            dir_total = len(results)
            dir_real = sum(1 for r in results if r.status in ["IDENTICAL", "OK", "MINOR_DIFF", "MAJOR_DIFF"])
            dir_rate = dir_real / dir_total * 100 if dir_total > 0 else 0
            
            print(f"\n  REVERSÃO {direction.upper()}:")
            print(f"    Total: {dir_total}")
            print(f"    Taxa de sucesso: {dir_rate:.1f}%")


async def run_continuous(direction_filter: str = "up", batch_size: int = 50):
    """
    Modo contínuo: roda auditoria indefinidamente, salvando resultados no banco.
    
    Projetado para rodar por dias/semanas em background no VPS.
    Após cada batch de N auditorias, loga um resumo e continua.
    Reconecta automaticamente em caso de erro.
    """
    import signal
    
    running = True
    
    def signal_handler(signum, frame):
        nonlocal running
        logger.info(f"Sinal {signum} recebido, encerrando após batch atual...")
        running = False
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    total_audits = 0
    total_batches = 0
    total_errors = 0
    start_time = datetime.now(timezone.utc)
    
    # Acumuladores globais de resultados
    global_counts = {"IDENTICAL": 0, "OK": 0, "MINOR_DIFF": 0, "MAJOR_DIFF": 0, "LINE_NOT_AVAILABLE": 0, "OTHER": 0}
    global_diffs = []
    
    logger.info("=" * 70)
    logger.info("AUDITORIA H3B - MODO CONTÍNUO")
    logger.info("=" * 70)
    logger.info(f"Direção: {direction_filter}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Salva resultados no banco de dados")
    logger.info(f"Para parar: Ctrl+C ou kill -TERM <pid>")
    logger.info("=" * 70)
    
    while running:
        total_batches += 1
        batch_start = datetime.now(timezone.utc)
        
        logger.info(f"\n--- BATCH #{total_batches} iniciando (total acumulado: {total_audits} auditorias) ---")
        
        try:
            auditor = H3BAuditor(
                num_audits=batch_size, 
                direction_filter=direction_filter, 
                save_to_db=True
            )
            await auditor.run_audit()
            
            # Acumula resultados
            for r in auditor.audit_results:
                total_audits += 1
                if r.status in global_counts:
                    global_counts[r.status] += 1
                else:
                    global_counts["OTHER"] += 1
                if r.difference_pct is not None:
                    global_diffs.append(abs(r.difference_pct))
            
            # Log resumo do batch
            batch_duration = (datetime.now(timezone.utc) - batch_start).total_seconds()
            uptime = datetime.now(timezone.utc) - start_time
            days = uptime.days
            hours, remainder = divmod(uptime.seconds, 3600)
            minutes, _ = divmod(remainder, 60)
            
            real_opps = global_counts["IDENTICAL"] + global_counts["OK"] + global_counts["MINOR_DIFF"] + global_counts["MAJOR_DIFF"]
            rate = (real_opps / total_audits * 100) if total_audits > 0 else 0
            avg_diff = (sum(global_diffs) / len(global_diffs)) if global_diffs else 0
            
            logger.info(f"\n{'=' * 60}")
            logger.info(f"RESUMO CONTÍNUO - Batch #{total_batches}")
            logger.info(f"  Uptime: {days}d {hours}h {minutes}m")
            logger.info(f"  Total auditorias: {total_audits}")
            logger.info(f"  Oportunidades reais: {real_opps} ({rate:.1f}%)")
            logger.info(f"  Diferença média: {avg_diff:.3f}%")
            logger.info(f"  Linhas indisponíveis: {global_counts['LINE_NOT_AVAILABLE']}")
            logger.info(f"  Erros reconexão: {total_errors}")
            logger.info(f"  Batch duration: {batch_duration:.0f}s")
            logger.info(f"{'=' * 60}")
            
        except KeyboardInterrupt:
            logger.info("Interrupção recebida (Ctrl+C)")
            running = False
            break
            
        except Exception as e:
            total_errors += 1
            logger.error(f"Erro no batch #{total_batches}: {e}")
            logger.info(f"Aguardando 60s antes de reconectar... (erro {total_errors})")
            
            # Pausa antes de reconectar
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                running = False
                break
        
        if running:
            # Pausa entre batches (deixa WebSocket "esfriar")
            logger.info("Pausa de 10s entre batches...")
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                running = False
                break
    
    # Resumo final
    uptime = datetime.now(timezone.utc) - start_time
    logger.info(f"\n{'=' * 70}")
    logger.info(f"ENCERRADO - Resumo final")
    logger.info(f"  Uptime total: {uptime}")
    logger.info(f"  Batches: {total_batches}")
    logger.info(f"  Total auditorias: {total_audits}")
    
    real_opps = global_counts["IDENTICAL"] + global_counts["OK"] + global_counts["MINOR_DIFF"] + global_counts["MAJOR_DIFF"]
    if total_audits > 0:
        logger.info(f"  Taxa sucesso: {real_opps / total_audits * 100:.1f}%")
    if global_diffs:
        logger.info(f"  Diferença média: {sum(global_diffs)/len(global_diffs):.3f}%")
    logger.info(f"  Erros reconexão: {total_errors}")
    logger.info(f"{'=' * 70}")


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Auditoria H3B: WebSocket vs Betslip")
    parser.add_argument("--continuous", action="store_true",
                        help="Modo contínuo: roda indefinidamente (dias/semanas)")
    parser.add_argument("--num-audits", type=int, default=50,
                        help="Número de auditorias por batch (default: 50)")
    parser.add_argument("--direction", choices=["up", "down", "all"], default="up",
                        help="Filtro de direção da reversão (default: up)")
    args = parser.parse_args()
    
    # Configura logging
    logger.remove()
    
    if args.continuous:
        # Modo contínuo: log em arquivo + stderr
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
            level="INFO"
        )
        logger.add(
            "logs/audit_h3b_{time:YYYY-MM-DD}.log",
            rotation="00:00",
            retention="60 days",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
        )
        
        await run_continuous(
            direction_filter=args.direction,
            batch_size=args.num_audits
        )
    else:
        # Modo original: N auditorias e sai
        logger.add(sys.stderr, level="WARNING")
        auditor = H3BAuditor(num_audits=args.num_audits, direction_filter=args.direction)
        await auditor.run_audit()


if __name__ == "__main__":
    asyncio.run(main())
