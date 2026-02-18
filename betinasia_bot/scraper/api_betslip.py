# -*- coding: utf-8 -*-
"""
Cliente API Betslip — Extrai odds via API REST + WebSocket (sem browser DOM).

Fluxo:
  1. POST /v1/betslips/ com event_id + bet_type
  2. Escuta mensagens PMM (Price Market Maker) no WebSocket
  3. Cada PMM contém: bookmaker, odds, limites
  4. Agrega e retorna best odd + limite

Tempo estimado: ~2-3 segundos por betslip.

Uso:
    # Dentro de um scraper com WS já conectado:
    client = ApiBetslipClient(page)
    result = await client.get_betslip_odds("2026-02-08,176,178", "for,ah,h,-1")
    print(result.best_odd, result.best_limit)
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from loguru import logger
from playwright.async_api import Page


@dataclass
class BookmakerPrice:
    """Preço de um bookmaker individual."""
    bookie: str
    best_price: float  # Melhor odd oferecida
    max_stake: float  # Limite máximo em GBP
    currency: str = "GBP"
    num_tiers: int = 1  # Quantos price tiers oferece
    all_prices: List[dict] = field(default_factory=list)  # Todos os tiers


@dataclass
class BetslipApiResult:
    """Resultado da API do betslip."""
    event_id: str
    bet_type: str
    betslip_id: str = ""
    
    # Odds agregadas
    best_odd: float = 0  # Melhor odd entre todos os bookmakers
    best_bookie: str = ""  # Bookmaker com melhor odd
    best_limit: float = 0  # Limite do bookmaker com melhor odd
    
    second_odd: float = 0  # Segunda melhor odd
    second_bookie: str = ""
    
    highest_limit: float = 0  # Maior limite (independente da odd)
    highest_limit_bookie: str = ""
    highest_limit_odd: float = 0  # Odd do bookmaker com maior limite
    
    # Todos os bookmakers
    bookmakers: List[BookmakerPrice] = field(default_factory=list)
    num_bookmakers: int = 0
    
    # Timing
    request_time_ms: int = 0  # Tempo do POST
    total_time_ms: int = 0  # Tempo total até todos PMMs recebidos
    
    # Status
    success: bool = False
    error: str = ""
    http_status: int = 0
    has_session_token: bool = False
    rate_limit_retry_after_sec: int = 0


class ApiBetslipClient:
    """
    Cliente que extrai odds do betslip via API (sem DOM).
    
    Requer uma Page do Playwright com sessão autenticada e
    WebSocket já conectado ao BetinAsia.
    """
    
    # Tempo máximo para esperar todos os PMMs
    PMM_TIMEOUT = 4.0  # segundos
    
    # Tempo mínimo antes de considerar completo (espera bookmakers lentos)
    PMM_MIN_WAIT = 1.5  # segundos
    
    # Se não receber novo PMM por este tempo, considera completo
    PMM_IDLE_TIMEOUT = 0.8  # segundos

    # Espera por betslip_id via resposta/WS
    BETSLIP_ID_TIMEOUT = 3.0  # segundos (para compatibilidade com histórico)
    
    def __init__(self, page: Page):
        self.page = page
        self._pmm_messages: Dict[str, List[dict]] = {}  # betslip_id -> [pmm_data]
        self._betslip_info: Dict[str, dict] = {}  # betslip_id -> betslip creation data
        self._listening = False
    
    def setup_listener(self):
        """Configura listener de WebSocket para PMM messages.
        Deve ser chamado UMA VEZ após o WS conectar.
        Captura tanto WS existentes quanto novos."""
        if self._listening:
            return
        
        def _attach_frame_listener(ws):
            def on_frame(data):
                try:
                    msg = json.loads(str(data))
                    if not isinstance(msg, list):
                        return
                    for item in msg:
                        if not isinstance(item, list) or len(item) < 2:
                            continue
                        if item[0] == 'api' and isinstance(item[1], dict):
                            api_data = item[1].get('data', [])
                            for entry in api_data:
                                if isinstance(entry, list) and len(entry) >= 2:
                                    if entry[0] == 'pmm':
                                        self._handle_pmm(entry[1])
                                    elif entry[0] == 'betslip':
                                        self._handle_betslip(entry[1])
                except:
                    pass
            ws.on('framereceived', on_frame)
        
        # Captura WS novos
        self.page.on('websocket', _attach_frame_listener)
        
        self._listening = True
        logger.debug("ApiBetslipClient: listener WS configurado")
    
    def _handle_betslip(self, data: dict):
        """Processa mensagem de criação de betslip."""
        betslip_id = data.get('betslip_id', '')
        if betslip_id:
            self._betslip_info[betslip_id] = data
            if betslip_id not in self._pmm_messages:
                self._pmm_messages[betslip_id] = []
    
    def _handle_pmm(self, data: dict):
        """Processa mensagem PMM (preço de bookmaker)."""
        betslip_id = data.get('betslip_id', '')
        if betslip_id:
            if betslip_id not in self._pmm_messages:
                self._pmm_messages[betslip_id] = []
            self._pmm_messages[betslip_id].append(data)
    
    async def get_betslip_odds(self, event_id: str, bet_type: str, betslip_type: str = "normal") -> BetslipApiResult:
        """
        Obtém odds do betslip via API.
        
        Args:
            event_id: ID do evento (ex: "2026-02-08,176,178")
            bet_type: Tipo de aposta. Formatos:
                - "for,ah,h,-1"  = AH -1 home
                - "for,ah,a,-1"  = AH -1 away
                - "for,ah,h,0"   = AH 0 home
                - "for,h"        = Home (match odds)
                - "for,a"        = Away (match odds)
        
        Returns:
            BetslipApiResult com odds de todos os bookmakers
        """
        result = BetslipApiResult(event_id=event_id, bet_type=bet_type)
        t0 = time.time()
        
        try:
            def _find_betslip_id(obj, depth: int = 0) -> str:
                if depth > 4:
                    return ""
                if isinstance(obj, dict):
                    for k in ("betslip_id", "betslipId", "id"):
                        v = obj.get(k)
                        if isinstance(v, str) and v.strip():
                            return v.strip()
                    for v in obj.values():
                        bid = _find_betslip_id(v, depth + 1)
                        if bid:
                            return bid
                if isinstance(obj, list):
                    for it in obj:
                        bid = _find_betslip_id(it, depth + 1)
                        if bid:
                            return bid
                return ""

            def _extract_retry_after_sec(obj) -> int:
                try:
                    if isinstance(obj, dict):
                        for k in ("retry_after", "retryAfter"):
                            if k in obj:
                                return int(float(obj[k]))
                        if "data" in obj:
                            return _extract_retry_after_sec(obj.get("data"))
                    return 0
                except Exception:
                    return 0

            # === 1. POST /v1/betslips/ via browser fetch ===
            t_post = time.time()
            
            response = await self.page.evaluate("""
                async (params) => {
                    try {
                        // Extrai session token do cookie root-session
                        const cookies = document.cookie.split(';');
                        let sessionToken = '';
                        for (const c of cookies) {
                            const [name, val] = c.trim().split('=');
                            if (name === 'root-session') {
                                sessionToken = val;
                                break;
                            }
                        }
                        
                        const resp = await fetch('/v1/betslips/', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                                'Accept': 'application/json, text/plain, */*',
                                'session': sessionToken,
                                'x-molly-client-name': 'sonic',
                                'x-molly-client-version': '2.5.34'
                            },
                            body: JSON.stringify({
                                sport: 'fb',
                                event_id: params.event_id,
                                bet_type: params.bet_type,
                                betslip_type: params.betslip_type,
                                equivalent_bets: true
                            })
                        });
                        let text = '';
                        let data = null;
                        try { text = await resp.text(); } catch(e) { text = ''; }
                        try { data = JSON.parse(text); } catch(e) { data = null; }
                        const prefix = (text || '').slice(0, 220);
                        return {ok: resp.ok, status: resp.status, data: data, text_prefix: prefix, session: sessionToken};
                    } catch(e) {
                        return {ok: false, error: e.message};
                    }
                }
            """, {"event_id": event_id, "bet_type": bet_type, "betslip_type": betslip_type})
            
            result.request_time_ms = int((time.time() - t_post) * 1000)
            result.http_status = int(response.get("status") or 0) if isinstance(response, dict) else 0
            result.has_session_token = bool(response.get("session")) if isinstance(response, dict) else False
            
            logger.info(f"POST /v1/betslips/: status={response.get('status') if response else 'none'}, "
                        f"ok={response.get('ok') if response else False}, "
                        f"time={result.request_time_ms}ms, "
                        f"data_keys={list(response.get('data', {}).keys()) if response and isinstance(response.get('data'), dict) else 'n/a'}")
            
            if not response or not isinstance(response, dict):
                result.error = "No response"
                logger.warning(f"POST /v1/betslips/ falhou: {result.error}")
                return result

            status_code = int(response.get("status") or 0)
            ok = bool(response.get("ok"))
            data = response.get("data")
            prefix = str(response.get("text_prefix") or "")

            retry_after = _extract_retry_after_sec(data)
            if retry_after > 0:
                result.rate_limit_retry_after_sec = int(retry_after)
                result.error = f"RATE_LIMIT retry_after={int(retry_after)}s"
                logger.warning(f"POST /v1/betslips/ rate limit: {result.error} | status={status_code} | prefix={prefix[:120]}")
                return result

            if (not ok) or (status_code and status_code != 200):
                # Status != 200 pode ocorrer como 401/403/429 etc. (resp.ok=False)
                result.error = response.get("error") or f"HTTP_{status_code}"
                if status_code:
                    result.error = f"HTTP_{status_code}: {result.error}"
                if prefix:
                    result.error = f"{result.error} | resp_prefix={prefix[:160]}"
                logger.warning(f"POST /v1/betslips/ falhou: {result.error}")
                return result
            
            # Extrai betslip_id da resposta
            resp_data = data or {}
            logger.debug(f"POST data: status={status_code}, resp_data keys={list(resp_data.keys()) if isinstance(resp_data, dict) else type(resp_data)}")
            betslip_id = _find_betslip_id(resp_data)
            
            if not betslip_id:
                # Betslip_id vem via WebSocket ["betslip", {betslip_id: "..."}]
                # Espera até 3 segundos pelo WS message
                wait_start = time.time()
                while time.time() - wait_start < float(self.BETSLIP_ID_TIMEOUT):
                    for bid in reversed(list(self._betslip_info.keys())):
                        info = self._betslip_info[bid]
                        if info.get('event_id') == event_id:
                            betslip_id = bid
                            break
                    if betslip_id:
                        break
                    await asyncio.sleep(0.05)
            
            if not betslip_id:
                # Último fallback: pega o betslip_id mais recente (qualquer evento)
                # Os PMMs podem já estar chegando
                if self._betslip_info:
                    betslip_id = list(self._betslip_info.keys())[-1]
                    logger.debug(f"Usando último betslip_id: {betslip_id}")
            
            if not betslip_id:
                keys = list(resp_data.keys()) if isinstance(resp_data, dict) else []
                result.error = f'No betslip_id received after {float(self.BETSLIP_ID_TIMEOUT):.0f}s (status=200 keys={keys}, has_session={result.has_session_token})'
                return result
            
            result.betslip_id = betslip_id
            
            # Inicializa lista de PMMs se não existe
            if betslip_id not in self._pmm_messages:
                self._pmm_messages[betslip_id] = []
            
            # === 2. Espera PMMs chegarem via WebSocket ===
            last_pmm_time = time.time()
            start_wait = time.time()
            last_count = 0
            
            while True:
                elapsed = time.time() - start_wait
                current_count = len(self._pmm_messages.get(betslip_id, []))
                
                # Novo PMM chegou? Reseta idle timer
                if current_count > last_count:
                    last_pmm_time = time.time()
                    last_count = current_count
                
                # Timeout total
                if elapsed > self.PMM_TIMEOUT:
                    break
                
                # Se já tem PMMs e ficou idle por PMM_IDLE_TIMEOUT
                if current_count > 0 and elapsed > self.PMM_MIN_WAIT:
                    idle = time.time() - last_pmm_time
                    if idle > self.PMM_IDLE_TIMEOUT:
                        break
                
                await asyncio.sleep(0.05)  # 50ms polling
            
            # === 3. Processa PMMs ===
            pmm_list = self._pmm_messages.get(betslip_id, [])
            
            if not pmm_list:
                result.error = f'No PMMs received (waited {time.time() - start_wait:.1f}s)'
                result.total_time_ms = int((time.time() - t0) * 1000)
                return result
            
            # Agrupa por bookmaker (pega o mais recente de cada)
            bookie_latest: Dict[str, dict] = {}
            for pmm in pmm_list:
                bookie = pmm.get('bookie', '')
                if not bookie:
                    continue
                # Chave: bookie + username (mesmo bookie pode ter múltiplas contas)
                key = f"{bookie}_{pmm.get('username', '')}"
                bookie_latest[key] = pmm
            
            # Extrai odds e limites
            for key, pmm in bookie_latest.items():
                bookie = pmm.get('bookie', '')
                status = pmm.get('status', {})
                
                if status.get('code') != 'success':
                    continue
                
                price_list = pmm.get('price_list', [])
                if not price_list:
                    continue
                
                # Primeiro tier = melhor preço
                first_tier = price_list[0]
                effective = first_tier.get('effective', {})
                price = effective.get('price', 0)
                max_stake_data = effective.get('max', [])
                
                if not price or price <= 0:
                    continue
                
                # Extrai limite (formato: ["GBP", 14717.49])
                max_stake = 0
                currency = "GBP"
                if isinstance(max_stake_data, list) and len(max_stake_data) >= 2:
                    currency = max_stake_data[0]
                    max_stake = float(max_stake_data[1])
                
                # Todos os tiers
                all_prices = []
                for tier in price_list:
                    eff = tier.get('effective', {})
                    p = eff.get('price', 0)
                    m = eff.get('max', [])
                    mx = float(m[1]) if isinstance(m, list) and len(m) >= 2 else 0
                    all_prices.append({'price': p, 'max_stake': mx})
                
                bk = BookmakerPrice(
                    bookie=bookie,
                    best_price=price,
                    max_stake=max_stake,
                    currency=currency,
                    num_tiers=len(price_list),
                    all_prices=all_prices,
                )
                result.bookmakers.append(bk)
            
            # Ordena por preço (melhor primeiro)
            result.bookmakers.sort(key=lambda b: b.best_price, reverse=True)
            result.num_bookmakers = len(result.bookmakers)
            
            if result.bookmakers:
                best = result.bookmakers[0]
                result.best_odd = best.best_price
                result.best_bookie = best.bookie
                result.best_limit = best.max_stake
                
                if len(result.bookmakers) > 1:
                    second = result.bookmakers[1]
                    result.second_odd = second.best_price
                    result.second_bookie = second.bookie
                
                # Maior limite
                by_limit = sorted(result.bookmakers, key=lambda b: b.max_stake, reverse=True)
                result.highest_limit = by_limit[0].max_stake
                result.highest_limit_bookie = by_limit[0].bookie
                result.highest_limit_odd = by_limit[0].best_price
                
                result.success = True
            
            result.total_time_ms = int((time.time() - t0) * 1000)
            
            # Cleanup
            if betslip_id in self._pmm_messages:
                del self._pmm_messages[betslip_id]
            if betslip_id in self._betslip_info:
                del self._betslip_info[betslip_id]
            
            return result
            
        except Exception as e:
            result.error = str(e)
            result.total_time_ms = int((time.time() - t0) * 1000)
            logger.error(f"ApiBetslipClient erro: {e}")
            return result
    
    async def refresh_betslip(self, betslip_id: str) -> BetslipApiResult:
        """Atualiza odds de um betslip existente sem criar novo."""
        result = BetslipApiResult(event_id="", bet_type="")
        result.betslip_id = betslip_id
        t0 = time.time()
        
        try:
            # Limpa PMMs antigos para este betslip
            self._pmm_messages[betslip_id] = []
            
            response = await self.page.evaluate("""
                async (params) => {
                    try {
                        const cookies = document.cookie.split(';');
                        let sessionToken = '';
                        for (const c of cookies) {
                            const [name, val] = c.trim().split('=');
                            if (name === 'root-session') { sessionToken = val; break; }
                        }
                        const resp = await fetch('/v1/betslips/' + params.betslip_id + '/refresh/', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                                'Accept': 'application/json',
                                'session': sessionToken,
                                'x-molly-client-name': 'sonic',
                                'x-molly-client-version': '2.5.35'
                            },
                            body: JSON.stringify({betslipId: params.betslip_id})
                        });
                        const data = await resp.json();
                        return {ok: true, status: resp.status, data: data};
                    } catch(e) { return {ok: false, error: e.message}; }
                }
            """, {"betslip_id": betslip_id})
            
            result.request_time_ms = int((time.time() - t0) * 1000)
            
            if not response or not response.get('ok'):
                result.error = response.get('error', 'Refresh failed') if response else 'No response'
                return result
            
            # Espera PMMs atualizados
            start_wait = time.time()
            last_count = 0
            while time.time() - start_wait < self.PMM_TIMEOUT:
                current = len(self._pmm_messages.get(betslip_id, []))
                if current > last_count:
                    last_count = current
                if current > 0 and time.time() - start_wait > self.PMM_MIN_WAIT:
                    break
                await asyncio.sleep(0.05)
            
            # Processa PMMs (mesma lógica de get_betslip_odds)
            pmm_list = self._pmm_messages.get(betslip_id, [])
            if not pmm_list:
                result.error = 'No PMMs after refresh'
                result.total_time_ms = int((time.time() - t0) * 1000)
                return result
            
            # Reusar lógica de processamento
            bookie_latest = {}
            for pmm in pmm_list:
                bookie = pmm.get('bookie', '')
                if bookie:
                    key = f"{bookie}_{pmm.get('username', '')}"
                    bookie_latest[key] = pmm
            
            for key, pmm in bookie_latest.items():
                if pmm.get('status', {}).get('code') != 'success':
                    continue
                price_list = pmm.get('price_list', [])
                if not price_list:
                    continue
                effective = price_list[0].get('effective', {})
                price = effective.get('price', 0)
                max_stake_data = effective.get('max', [])
                if not price or price <= 0:
                    continue
                max_stake = float(max_stake_data[1]) if isinstance(max_stake_data, list) and len(max_stake_data) >= 2 else 0
                
                result.bookmakers.append(BookmakerPrice(
                    bookie=pmm.get('bookie', ''),
                    best_price=price,
                    max_stake=max_stake,
                ))
            
            result.bookmakers.sort(key=lambda b: b.best_price, reverse=True)
            result.num_bookmakers = len(result.bookmakers)
            if result.bookmakers:
                result.best_odd = result.bookmakers[0].best_price
                result.best_bookie = result.bookmakers[0].bookie
                result.best_limit = result.bookmakers[0].max_stake
                result.success = True
            
            result.total_time_ms = int((time.time() - t0) * 1000)
            return result
            
        except Exception as e:
            result.error = str(e)
            result.total_time_ms = int((time.time() - t0) * 1000)
            return result
    
    @staticmethod
    def build_bet_type(market_type: str, side: str, line: str = None) -> str:
        """
        Constrói o bet_type para a API.
        
        Args:
            market_type: "AH", "OU", "1X2"
            side: "home", "away", "over", "under"
            line: Valor da linha (ex: "-1", "0", "2.5"). None para match odds.
        
        Returns:
            bet_type string (ex: "for,ah,h,-1")
        """
        if market_type == "AH":
            h_or_a = "h" if side == "home" else "a"
            if line is not None:
                return f"for,ah,{h_or_a},{line}"
            else:
                return f"for,{h_or_a}"
        elif market_type == "OU":
            o_or_u = "o" if side in ("over", "home") else "u"
            if line is not None:
                return f"for,ou,{o_or_u},{line}"
            else:
                return f"for,{o_or_u}"
        elif market_type == "1X2":
            if side == "home":
                return "for,h"
            elif side == "away":
                return "for,a"
            else:
                return "for,d"
        else:
            return f"for,{side[0]}"
    
    @staticmethod
    def build_lay_bet_type(market_type: str, side: str, line: str = None) -> str:
        """Constrói bet_type para LAY (exchange). Inverte o lado."""
        if market_type == "AH":
            h_or_a = "h" if side == "home" else "a"
            if line is not None:
                return f"against,ah,{h_or_a},{line}"
            else:
                return f"against,{h_or_a}"
        elif market_type == "OU":
            o_or_u = "o" if side in ("over", "home") else "u"
            if line is not None:
                return f"against,ou,{o_or_u},{line}"
            else:
                return f"against,{o_or_u}"
        else:
            return f"against,{side[0]}"
