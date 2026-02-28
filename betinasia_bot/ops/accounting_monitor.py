from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from loguru import logger

from scraper.betinasia import BetinAsiaScraper


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        s = s.replace(",", ".")
        s = re.sub(r"[^0-9.\-]", "", s)
        if s in ("", "-", ".", "-."):
            return None
        return float(s)
    except Exception:
        return None


def _parse_csv_best_effort(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"rows": 0, "cols": [], "sum_numeric_by_col": {}}
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            cols = list(reader.fieldnames or [])
            out["cols"] = cols
            sums: Dict[str, float] = {}
            n = 0
            for row in reader:
                n += 1
                for k, v in (row or {}).items():
                    if not k:
                        continue
                    fv = _safe_float(v)
                    if fv is None:
                        continue
                    sums[k] = float(sums.get(k, 0.0)) + float(fv)
            out["rows"] = n
            out["sum_numeric_by_col"] = sums
            return out
    except Exception as e:
        out["error"] = str(e)[:200]
        return out


async def _download_from_accounting_page(
    scraper: BetinAsiaScraper,
    *,
    name: str,
    url: str,
    out_dir: Path,
    timeout_ms: int = 20000,
) -> Tuple[Optional[Path], Dict[str, Any]]:
    """
    Abre uma página de accounting e tenta baixar o CSV.
    Estratégia:
    - tenta botões/links com texto 'CSV'/'Export'/'Download'
    - se não houver download, tenta detectar links com href contendo 'csv'/'export'
    """
    page = scraper._page
    assert page is not None
    meta: Dict[str, Any] = {"name": name, "url": url}

    resp = await page.goto(url, wait_until="domcontentloaded")
    try:
        meta["http_status"] = int(resp.status) if resp else None
    except Exception:
        meta["http_status"] = None
    await page.wait_for_timeout(800)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{ts}__{name}.csv"

    selectors = [
        "a:has-text('CSV')",
        "button:has-text('CSV')",
        "[role='button']:has-text('CSV')",
        "text=/^CSV$/",
        "a:has-text('Export')",
        "button:has-text('Export')",
        "[role='button']:has-text('Export')",
        "a:has-text('Download')",
        "button:has-text('Download')",
        "[role='button']:has-text('Download')",
        "a:has-text('Baixar')",
        "button:has-text('Baixar')",
        "[role='button']:has-text('Baixar')",
    ]

    for sel in selectors:
        try:
            loc = page.locator(sel)
            cnt = await loc.count()
            if cnt <= 0:
                continue
            meta["clicked_selector"] = sel
            async with page.expect_download(timeout=timeout_ms) as di:
                await loc.first.click()
            dl = await di.value
            await dl.save_as(str(out_path))
            meta["downloaded_via"] = "expect_download"
            return out_path, meta
        except Exception as e:
            meta.setdefault("attempt_errors", []).append({sel: str(e)[:160]})
            continue

    # fallback: achar links prováveis e tentar baixar via click
    try:
        hrefs = await page.evaluate(
            """
            () => {
              const out = [];
              for (const a of Array.from(document.querySelectorAll('a[href]'))) {
                const h = a.getAttribute('href') || '';
                const t = (a.innerText || '').trim();
                out.push({href: h, text: t.slice(0, 80)});
              }
              return out;
            }
            """
        )
        cand = []
        for it in (hrefs or []):
            if not isinstance(it, dict):
                continue
            h = str(it.get("href") or "")
            if not h:
                continue
            hl = h.lower()
            if ("csv" in hl) or ("export" in hl) or ("download" in hl):
                cand.append(h)
        meta["href_candidates"] = cand[:10]
        for h in cand[:5]:
            try:
                loc = page.locator(f"a[href='{h}']")
                if await loc.count() <= 0:
                    continue
                meta["clicked_href"] = h
                async with page.expect_download(timeout=timeout_ms) as di:
                    await loc.first.click()
                dl = await di.value
                await dl.save_as(str(out_path))
                meta["downloaded_via"] = "href_click"
                return out_path, meta
            except Exception as e:
                meta.setdefault("attempt_errors", []).append({f"href:{h}": str(e)[:160]})
                continue
    except Exception as e:
        meta["fallback_error"] = str(e)[:200]

    return None, meta


@dataclass
class AccountingConfig:
    poll_sec: float = 300.0
    out_dir: Path = Path("logs/accounting")
    jsonl_path: Path = Path("logs/accounting_snapshots.jsonl")
    once: bool = False


async def run_monitor(cfg: AccountingConfig) -> int:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    urls = {
        "balance": "https://black.betinasia.com/accounting/balance",
        "open_stakes": "https://black.betinasia.com/accounting/open-stakes",
    }

    scraper = BetinAsiaScraper()
    await scraper.start()
    ok = await scraper.login()
    if not ok:
        raise RuntimeError("LOGIN_FAILED")

    logger.info(f"[acct] started poll_sec={cfg.poll_sec} out_dir={cfg.out_dir} jsonl={cfg.jsonl_path}")

    while True:
        t0 = time.time()
        snap: Dict[str, Any] = {"ts": _utcnow(), "files": {}, "parsed": {}, "meta": {}}
        for name, url in urls.items():
            try:
                p, meta = await _download_from_accounting_page(scraper, name=name, url=url, out_dir=cfg.out_dir)
                snap["meta"][name] = meta
                if p:
                    snap["files"][name] = str(p)
                    snap["parsed"][name] = _parse_csv_best_effort(Path(p))
                else:
                    snap["files"][name] = None
            except Exception as e:
                snap["meta"][name] = {"error": str(e)[:300], "url": url}
                snap["files"][name] = None

        with cfg.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(snap, ensure_ascii=False) + "\n")

        dt = time.time() - t0
        logger.info(f"[acct] snapshot ok dt={dt:.1f}s files={snap['files']}")

        if cfg.once:
            break
        await asyncio.sleep(max(1.0, float(cfg.poll_sec) - dt))

    try:
        await scraper.close()
    except Exception:
        pass
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Monitor de Accounting (saldo + open stakes) via CSV do site.")
    ap.add_argument("--poll-sec", type=float, default=float(os.getenv("ACCOUNTING_POLL_SEC", "300")))
    ap.add_argument("--out-dir", default=os.getenv("ACCOUNTING_OUT_DIR", "logs/accounting"))
    ap.add_argument("--jsonl", default=os.getenv("ACCOUNTING_JSONL", "logs/accounting_snapshots.jsonl"))
    ap.add_argument("--once", action="store_true", default=(os.getenv("ACCOUNTING_ONCE", "0").strip() in ("1", "true", "True", "yes", "YES")))
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))

    cfg = AccountingConfig(
        poll_sec=float(args.poll_sec),
        out_dir=Path(str(args.out_dir)),
        jsonl_path=Path(str(args.jsonl)),
        once=bool(args.once),
    )
    asyncio.run(run_monitor(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

