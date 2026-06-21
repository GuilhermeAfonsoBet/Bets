#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testa se a infraestrutura (PostgreSQL, Redis) está funcionando.
"""

import asyncio
import sys
from pathlib import Path

# Adiciona o diretório ao path
sys.path.insert(0, str(Path(__file__).parent))

async def test_postgresql():
    """Testa conexão com PostgreSQL."""
    print("\n" + "="*60)
    print("TESTE: PostgreSQL")
    print("="*60)
    
    try:
        from config import settings
        print(f"URL: {settings.database_url[:30]}...")
        
        from storage.database import Database
        
        db = Database()
        await db.connect()
        print("✅ Conexão OK")
        print("✅ Tabelas criadas/verificadas")
        
        # Testa inserção
        from storage.models import LeagueConfig
        from sqlalchemy import select
        
        async with db.async_session() as session:
            result = await session.execute(select(LeagueConfig).limit(1))
            count = len(result.scalars().all())
            print(f"✅ Query OK (LeagueConfig: {count} registros)")
        
        await db.close()
        return True
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        return False


async def test_redis():
    """Testa conexão com Redis."""
    print("\n" + "="*60)
    print("TESTE: Redis")
    print("="*60)
    
    try:
        from config import settings
        print(f"URL: {settings.redis_url}")
        
        from cache.redis_cache import OddsCache
        
        cache = OddsCache()
        await cache.connect()
        
        if cache._use_redis:
            print("✅ Conexão Redis OK")
            
            # Testa operação
            await cache.set_odds("test_match", "test_line", 1.95)
            value = await cache.get_odds("test_match", "test_line")
            
            if value == 1.95:
                print("✅ Operações OK (set/get)")
            else:
                print(f"⚠️ Valor retornado diferente: {value}")
        else:
            print("⚠️ Redis não disponível, usando cache em memória (fallback)")
            
        await cache.close()
        return cache._use_redis
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        return False


async def test_env_file():
    """Verifica arquivo .env."""
    print("\n" + "="*60)
    print("TESTE: Arquivo .env")
    print("="*60)
    
    env_path = Path(__file__).parent / ".env"
    
    if env_path.exists():
        print("✅ Arquivo .env existe")
        
        # Verifica variáveis essenciais
        from config import settings
        
        checks = [
            ("BETINASIA_USERNAME", bool(settings.betinasia_username)),
            ("BETINASIA_PASSWORD", bool(settings.betinasia_password)),
            ("DATABASE_URL", bool(settings.database_url)),
        ]
        
        for var, ok in checks:
            if ok:
                print(f"  ✅ {var}: Configurado")
            else:
                print(f"  ❌ {var}: NÃO configurado")
                
        return all(ok for _, ok in checks)
    else:
        print("❌ Arquivo .env NÃO existe")
        print("   Execute: cp .env.example .env")
        print("   E configure as variáveis")
        return False


async def test_scraper():
    """Testa se o scraper consegue iniciar."""
    print("\n" + "="*60)
    print("TESTE: Scraper (apenas inicialização)")
    print("="*60)
    
    try:
        from scraper import BetinAsiaScraper
        
        scraper = BetinAsiaScraper()
        await scraper.start()
        print("✅ Browser iniciado")
        
        await scraper.close()
        print("✅ Browser fechado")
        return True
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        return False


async def main():
    """Executa todos os testes."""
    print("\n" + "="*60)
    print("🔧 TESTE DE INFRAESTRUTURA")
    print("="*60)
    
    results = {}
    
    # Teste .env primeiro
    results["env"] = await test_env_file()
    
    if not results["env"]:
        print("\n⚠️ Configure o arquivo .env antes de continuar")
        return
    
    # Testes de infraestrutura
    results["postgresql"] = await test_postgresql()
    results["redis"] = await test_redis()
    results["scraper"] = await test_scraper()
    
    # Resumo
    print("\n" + "="*60)
    print("📊 RESUMO")
    print("="*60)
    
    all_ok = True
    for component, ok in results.items():
        status = "✅" if ok else "❌"
        print(f"  {status} {component.upper()}")
        if not ok:
            all_ok = False
    
    if all_ok:
        print("\n🎉 Todos os componentes funcionando!")
        print("   Execute: python main.py --collect")
    else:
        print("\n⚠️ Alguns componentes precisam de atenção")
        
        if not results.get("postgresql"):
            print("\n📌 Para configurar PostgreSQL:")
            print("   sudo -u postgres psql")
            print("   CREATE USER betbot WITH PASSWORD 'sua_senha';")
            print("   CREATE DATABASE betinasia_bot OWNER betbot;")
            
        if not results.get("redis"):
            print("\n📌 Para instalar Redis:")
            print("   sudo apt install redis-server")
            print("   sudo systemctl enable redis-server")


if __name__ == "__main__":
    asyncio.run(main())
