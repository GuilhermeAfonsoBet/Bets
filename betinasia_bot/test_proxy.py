#!/usr/bin/env python3
"""Teste de proxy via Playwright (sem credenciais hardcoded)."""

import argparse
import asyncio
import json
import sys
from typing import Dict, Optional

from playwright.async_api import async_playwright

from config import settings


def _mask(value: Optional[str], keep: int = 2) -> str:
    """Mascara segredos para evitar vazamento em logs."""
    if not value:
        return ""
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}***{value[-keep:]}"


def _parse_proxy_line(proxy_line: str, protocol: str = "http") -> Dict[str, str]:
    """
    Parseia proxy no formato HOST:PORT:USER:PASS.
    Também aceita HOST:PORT sem autenticação.
    """
    parts = proxy_line.strip().split(":")
    if len(parts) not in (2, 4):
        raise ValueError("Formato inválido. Use HOST:PORT ou HOST:PORT:USER:PASS")

    host, port = parts[0], parts[1]
    if not host or not port:
        raise ValueError("HOST e PORT são obrigatórios")

    proxy = {"server": f"{protocol}://{host}:{port}"}
    if len(parts) == 4:
        user, password = parts[2], parts[3]
        if user:
            proxy["username"] = user
        if password:
            proxy["password"] = password
    return proxy


def _load_proxy_from_env() -> Optional[Dict[str, str]]:
    proxy = settings.proxy_config
    return proxy if proxy else None


def _build_proxy(args: argparse.Namespace) -> Dict[str, str]:
    if args.proxy_line:
        return _parse_proxy_line(args.proxy_line, protocol=args.protocol)

    proxy = _load_proxy_from_env()
    if not proxy:
        raise ValueError(
            "Proxy não configurado. Defina PROXY_SERVER/PROXY_USERNAME/PROXY_PASSWORD no .env "
            "ou use --proxy-line HOST:PORT:USER:PASS."
        )
    return proxy


async def _run_test(
    proxy: Dict[str, str],
    target_url: str,
    timeout_ms: int,
    check_betinasia: bool,
) -> int:
    browser = None
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, proxy=proxy)
            page = await browser.new_page(ignore_https_errors=True)

            await page.goto(target_url, timeout=timeout_ms)
            body = await page.inner_text("body")
            preview = body[:400].replace("\n", " ").strip()
            print(f"[OK] Acesso a {target_url}")
            print(f"[OK] Preview resposta: {preview}")

            if check_betinasia:
                check_url = "https://black.betinasia.com/sportsbook/football"
                await page.goto(check_url, timeout=timeout_ms)
                print(f"[OK] Túnel até BetinAsia: {page.url}")

            await browser.close()
            return 0
    except Exception as exc:
        print(f"[FAIL] Teste de proxy falhou: {exc}")
        return 1
    finally:
        if browser:
            try:
                await browser.close()
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Teste rápido de proxy para collector/audit (Playwright)."
    )
    parser.add_argument(
        "--proxy-line",
        default=None,
        help="Proxy no formato HOST:PORT[:USER:PASS]. Se vazio, usa .env",
    )
    parser.add_argument(
        "--protocol",
        default="http",
        choices=["http", "https"],
        help="Protocolo do proxy quando usar --proxy-line.",
    )
    parser.add_argument(
        "--target-url",
        default="https://lumtest.com/myip.json",
        help="URL para validar navegação pelo proxy.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=20000,
        help="Timeout por navegação em ms.",
    )
    parser.add_argument(
        "--check-betinasia",
        action="store_true",
        help="Além do target, testa abertura de /sportsbook/football.",
    )
    args = parser.parse_args()

    try:
        proxy = _build_proxy(args)
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2

    proxy_safe = {
        "server": proxy.get("server"),
        "username": _mask(proxy.get("username")),
        "password": _mask(proxy.get("password")),
    }
    print(f"[INFO] Proxy em uso: {json.dumps(proxy_safe, ensure_ascii=False)}")

    return asyncio.run(
        _run_test(
            proxy=proxy,
            target_url=args.target_url,
            timeout_ms=args.timeout_ms,
            check_betinasia=args.check_betinasia,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
