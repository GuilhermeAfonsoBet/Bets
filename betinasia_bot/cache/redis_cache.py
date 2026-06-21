# -*- coding: utf-8 -*-
"""
Cache de odds usando Redis.

Se Redis não estiver disponível, usa cache em memória (fallback).
"""

import json
from datetime import datetime, timezone
from typing import Optional, Any, List
from loguru import logger

from config import settings


class OddsCache:
    """
    Cache de odds.
    
    Tenta usar Redis, se não disponível usa dicionário em memória.
    """
    
    DEFAULT_TTL = 300  # 5 minutos
    
    def __init__(self):
        self._redis = None
        self._memory_cache = {}  # Fallback
        self._use_redis = True
        
    async def connect(self):
        """Conecta ao Redis."""
        try:
            import redis.asyncio as redis
            
            self._redis = redis.from_url(
                settings.redis_url,
                decode_responses=True,
            )
            
            # Testa conexão
            await self._redis.ping()
            logger.info("Cache Redis conectado")
            self._use_redis = True
            
        except Exception as e:
            logger.warning(f"Redis não disponível, usando cache em memória: {e}")
            self._use_redis = False
            
    async def close(self):
        """Fecha conexão."""
        if self._redis:
            await self._redis.close()
            
    # ==========================================
    # OPERAÇÕES DE ODDS
    # ==========================================
    
    async def get_odds(self, match_id: str, ah_line: str) -> Optional[float]:
        """Retorna última odd conhecida para uma linha."""
        key = f"odds:{match_id}:{ah_line}"
        
        if self._use_redis:
            value = await self._redis.get(key)
            return float(value) if value else None
        else:
            return self._memory_cache.get(key)
            
    async def set_odds(
        self, 
        match_id: str, 
        ah_line: str, 
        odds: float, 
        ttl: int = DEFAULT_TTL
    ):
        """Salva odd no cache."""
        key = f"odds:{match_id}:{ah_line}"
        
        if self._use_redis:
            await self._redis.setex(key, ttl, str(odds))
            
            # Histórico recente
            history_key = f"odds:{match_id}:{ah_line}:history"
            entry = json.dumps({
                "odds": odds,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            await self._redis.lpush(history_key, entry)
            await self._redis.ltrim(history_key, 0, 99)
            await self._redis.expire(history_key, 3600)
        else:
            self._memory_cache[key] = odds
            
    async def odds_changed(
        self, 
        match_id: str, 
        ah_line: str, 
        new_odds: float, 
        threshold: float = 0.02
    ) -> bool:
        """
        Verifica se odds mudou significativamente.
        
        Args:
            threshold: Diferença mínima para considerar mudança (2% default)
            
        Returns:
            True se mudou, False se igual
        """
        old_odds = await self.get_odds(match_id, ah_line)
        
        if old_odds is None:
            return True  # Nova, considerar como mudança
            
        change_pct = abs(new_odds - old_odds) / old_odds
        return change_pct >= threshold
        
    async def get_odds_history(
        self, 
        match_id: str, 
        ah_line: str, 
        limit: int = 10
    ) -> List[dict]:
        """Retorna histórico recente de odds."""
        if not self._use_redis:
            return []
            
        key = f"odds:{match_id}:{ah_line}:history"
        entries = await self._redis.lrange(key, 0, limit - 1)
        return [json.loads(e) for e in entries]
        
    # ==========================================
    # RATE LIMITING
    # ==========================================
    
    async def check_rate_limit(
        self, 
        resource: str, 
        max_requests: int, 
        window_seconds: int
    ) -> bool:
        """
        Verifica rate limit.
        
        Returns:
            True se pode fazer request, False se excedeu limite
        """
        if not self._use_redis:
            return True  # Sem rate limit em memória
            
        key = f"rate_limit:{resource}"
        
        current = await self._redis.incr(key)
        
        if current == 1:
            await self._redis.expire(key, window_seconds)
            
        return current <= max_requests
        
    # ==========================================
    # LOCKS
    # ==========================================
    
    async def acquire_lock(self, resource: str, ttl: int = 30) -> bool:
        """
        Tenta adquirir lock distribuído.
        
        Útil para evitar execução duplicada de apostas.
        """
        if not self._use_redis:
            return True
            
        key = f"lock:{resource}"
        acquired = await self._redis.set(key, "1", nx=True, ex=ttl)
        return bool(acquired)
        
    async def release_lock(self, resource: str):
        """Libera lock."""
        if self._use_redis:
            key = f"lock:{resource}"
            await self._redis.delete(key)
            
    # ==========================================
    # SESSÃO
    # ==========================================
    
    async def save_session(self, session_id: str, data: dict, ttl: int = 3600):
        """Salva dados de sessão."""
        if self._use_redis:
            key = f"session:{session_id}"
            await self._redis.setex(key, ttl, json.dumps(data))
            
    async def get_session(self, session_id: str) -> Optional[dict]:
        """Recupera dados de sessão."""
        if not self._use_redis:
            return None
            
        key = f"session:{session_id}"
        data = await self._redis.get(key)
        return json.loads(data) if data else None
