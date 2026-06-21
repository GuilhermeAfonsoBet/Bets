from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import aiohttp

from .contracts import ExecutionRequest, ExecSide, MarketType


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _norm_line(s: str) -> str:
    # aceita "-0,25" do PT-BR
    return (s or "").strip().replace(",", ".").replace("−", "-")


async def _post_json_http(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout) as sess:
        async with sess.post(url, json=payload) as resp:
            data = await resp.json()
            data["_http_status"] = int(resp.status)
            return data


async def _post_json_unix(sock_path: str, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=20)
    conn = aiohttp.UnixConnector(path=str(sock_path))
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as sess:
        async with sess.post(url, json=payload) as resp:
            data = await resp.json()
            data["_http_status"] = int(resp.status)
            return data


async def submit_execution(
    *,
    req: ExecutionRequest,
    unix_socket: Optional[str] = None,
    http_base: Optional[str] = None,
) -> Dict[str, Any]:
    payload = req.model_dump(mode="json")
    if unix_socket:
        return await _post_json_unix(unix_socket, "http://localhost/execute", payload)
    base = (http_base or "http://127.0.0.1:8089").rstrip("/")
    return await _post_json_http(f"{base}/execute", payload)


def main() -> int:
    p = argparse.ArgumentParser(description="Client simples para chamar o Executor (/execute).")
    p.add_argument("--event-id", required=True, help="event_id (ex.: 2026-02-28,23624,173)")
    p.add_argument("--market", default="AH", choices=["AH"], help="market_type")
    p.add_argument("--side", required=True, choices=["home", "away"], help="lado (AH)")
    p.add_argument("--line", required=True, help="linha (ex.: -0.25, +0.5, 0, ou -0,25)")
    p.add_argument("--exec-side", default="Back", choices=["Back", "Lay"])
    p.add_argument("--live", action="store_true", help="is_live=true (requer EXECUTOR_ALLOW_LIVE=1 no serviço)")
    p.add_argument("--stake", type=float, default=None, help="stake_requested (Back). Default: usa policy/env do executor.")
    p.add_argument("--odd-at-decision", type=float, default=None, help="odd observada no Decision Engine (slippage).")

    p.add_argument("--unix-socket", default=os.getenv("EXECUTOR_UNIX_SOCKET", "").strip() or None)
    p.add_argument("--http", default=os.getenv("EXECUTOR_HTTP_URL", "").strip() or None)
    args = p.parse_args()

    req = ExecutionRequest(
        created_at=_utc_now(),
        event_id=str(args.event_id),
        market_type=MarketType.AH,
        side=str(args.side),
        line=_norm_line(str(args.line)),
        exec_side=ExecSide(str(args.exec_side)),
        is_live=bool(args.live),
        odd_at_decision=float(args.odd_at_decision) if args.odd_at_decision is not None else None,
    )
    if args.stake is not None:
        req.policy.stake_requested = float(args.stake)
        req.policy.policy_version = "cli_stake_v0"

    out = asyncio.run(submit_execution(req=req, unix_socket=args.unix_socket, http_base=args.http))
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

