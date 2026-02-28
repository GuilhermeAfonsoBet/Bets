from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from aiohttp import web
from loguru import logger

from scraper.betinasia import BetinAsiaScraper
from executor.service import ExecutorService, create_app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("EXECUTOR_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("EXECUTOR_PORT", "8089")))
    parser.add_argument("--unix-socket", default=os.getenv("EXECUTOR_UNIX_SOCKET", "").strip())
    parser.add_argument("--workers", type=int, default=int(os.getenv("EXECUTOR_WORKERS", "1")))
    parser.add_argument("--cap-window-sec", type=float, default=float(os.getenv("EXECUTOR_CAP_WINDOW_SEC", "300")))
    parser.add_argument("--cap-max", type=int, default=int(os.getenv("EXECUTOR_CAP_MAX", "999999")))
    parser.add_argument("--jsonl", default=os.getenv("EXECUTOR_JSONL", "logs/executor_dryrun.jsonl"))
    parser.add_argument("--save-to-db", action="store_true", default=(os.getenv("EXECUTOR_SAVE_TO_DB", "0") in ("1", "true", "True")))
    args = parser.parse_args()

    svc = ExecutorService(
        football_url=BetinAsiaScraper.FOOTBALL_URL,
        workers=int(args.workers),
        cap_window_sec=float(args.cap_window_sec),
        cap_max=int(args.cap_max),
        jsonl_path=str(args.jsonl) if args.jsonl else None,
        save_to_db=bool(args.save_to_db),
    )
    app = create_app(svc)

    runner = web.AppRunner(app)

    async def _run():
        sock_path = None
        if args.unix_socket:
            sock_path = Path(args.unix_socket)
            sock_path.parent.mkdir(parents=True, exist_ok=True)
            # remove socket antigo ANTES do startup (startup pode demorar por login)
            try:
                if sock_path.exists():
                    sock_path.unlink()
            except Exception:
                pass

        await runner.setup()
        if sock_path is not None:
            site = web.UnixSite(runner, str(sock_path))
            await site.start()
            logger.info(f"[executor] listening unix={sock_path}")
        else:
            site = web.TCPSite(runner, host=str(args.host), port=int(args.port))
            await site.start()
            logger.info(f"[executor] listening http://{args.host}:{args.port}")

        while True:
            await asyncio.sleep(3600)

    asyncio.run(_run())


if __name__ == "__main__":
    main()

