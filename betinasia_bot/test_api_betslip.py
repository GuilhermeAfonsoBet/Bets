#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Teste da API de Betslip.

Faz login via scraper, extrai cookies, e chama a API diretamente.
"""

import asyncio
import json
import os
import httpx
import pytest
from loguru import logger
from scraper.betinasia import BetinAsiaScraper


async def test_betslip_api():
    """Testa a API de betslip diretamente."""

    # Este é um teste E2E/integrado: depende de rede, proxy, e disponibilidade do site.
    # Por padrão, mantemos desabilitado para não quebrar a suíte em ambientes sem acesso externo.
    if os.getenv("RUN_BETINASIA_E2E", "0").strip() not in ("1", "true", "True", "yes", "YES"):
        pytest.skip("E2E desabilitado (set RUN_BETINASIA_E2E=1 para rodar)")
    
    print("=" * 60)
    print("TESTE DA API DE BETSLIP")
    print("=" * 60)
    
    # 1. Usa o scraper para fazer login e obter cookies
    scraper = BetinAsiaScraper()
    await scraper.start()
    await scraper.login()
    
    # 2. Extrai cookies do browser
    cookies = await scraper._context.cookies()
    cookie_dict = {c['name']: c['value'] for c in cookies}
    
    print(f"\nCookies extraídos: {len(cookie_dict)}")
    for name in list(cookie_dict.keys())[:5]:
        print(f"  - {name}")
    
    # 3. Vai para a página de jogos para pegar um event_id válido
    await scraper._page.goto("https://black.betinasia.com/sportsbook/football/XE/1")
    await scraper._page.wait_for_timeout(3000)
    
    # Encontra um jogo
    links = await scraper._page.query_selector_all("a")
    event_id = None
    for link in links:
        href = await link.get_attribute("href")
        if href and "/sportsbook/football/" in href and "," in href:
            # Extrai event_id da URL (ex: 2026-02-01,22,94)
            parts = href.split("/")
            for part in parts:
                if "," in part and part[0].isdigit():
                    event_id = part.split("?")[0]
                    break
            if event_id:
                break
    
    print(f"\nEvent ID encontrado: {event_id}")
    
    # Fecha o browser (não precisamos mais)
    await scraper.close()
    
    if not event_id:
        print("Não foi possível encontrar um evento!")
        return
    
    # 4. Chama a API diretamente com httpx
    print("\n" + "=" * 60)
    print("CHAMANDO API DIRETAMENTE")
    print("=" * 60)
    
    async with httpx.AsyncClient(
        base_url="https://black.betinasia.com",
        cookies=cookie_dict,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Origin": "https://black.betinasia.com",
            "Referer": f"https://black.betinasia.com/sportsbook/football/XE/1/{event_id}",
        },
        timeout=30.0
    ) as client:
        
        # Testa a API de betslip
        print(f"\nPOST /v1/betslips/")
        print(f"Event ID: {event_id}")
        
        response = await client.post(
            "/v1/betslips/",
            json={
                "sport": "fb",
                "event_id": event_id,
                "bet_type": "for,d",
                "betslip_type": "normal",
                "equivalent_bets": True
            }
        )
        
        print(f"\nStatus: {response.status_code}")
        
        try:
            data = response.json()
            print(f"\nResposta JSON:")
            print(json.dumps(data, indent=2, ensure_ascii=False)[:3000])
            
            # Se sucesso, analisa a estrutura
            if response.status_code == 200:
                print("\n" + "=" * 60)
                print("ANÁLISE DA RESPOSTA")
                print("=" * 60)
                analyze_response(data)
                
        except Exception as e:
            print(f"Erro ao parsear JSON: {e}")
            print(f"Resposta raw: {response.text[:500]}")


def analyze_response(data: dict):
    """Analisa a estrutura da resposta para encontrar bookmakers."""
    
    def find_bookmakers(obj, path=""):
        """Recursivamente procura por dados de bookmakers."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                
                # Procura por campos que parecem ter odds
                if any(term in key.lower() for term in ['book', 'odds', 'price', 'line', 'stake']):
                    print(f"Campo interessante: {current_path}")
                    print(f"  Valor: {str(value)[:200]}")
                    
                find_bookmakers(value, current_path)
                
        elif isinstance(obj, list) and len(obj) > 0:
            # Analisa primeiro item da lista
            find_bookmakers(obj[0], f"{path}[0]")
    
    find_bookmakers(data)


if __name__ == "__main__":
    asyncio.run(test_betslip_api())
