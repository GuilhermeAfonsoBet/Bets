from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from loguru import logger

from scraper.betinasia import BetinAsiaScraper

from .accounting_io import atomic_write_bytes, atomic_write_json
from .accounting_status import (
    ACCOUNTING_API_FAILED,
    ACCOUNTING_AUTH_FAILED,
    ACCOUNTING_BROWSER_DEAD,
    ACCOUNTING_EMPTY_RESPONSE,
    ACCOUNTING_OK,
    ACCOUNTING_PARTIAL,
    ACCOUNTING_PARSE_FAILED,
    ACCOUNTING_RATE_LIMIT,
    ACCOUNTING_SCHEMA_CHANGED,
    ACCOUNTING_TIMEOUT,
    ACCOUNTING_UNKNOWN_FAILURE,
    ACCOUNTING_WRITE_FAILED,
    FreshnessLimits,
    classify_exception,
    classify_health,
    cycle_status,
    normalize_jsonl_path,
    sanitize_error_message,
    utcnow_iso,
    validate_csv_schema,
)


def _utcnow() -> str:
    return utcnow_iso()


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        s = s.replace(",", ".")
        import re

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
            ok_schema, schema_err = validate_csv_schema(cols)
            if not ok_schema:
                out["schema_ok"] = False
                out["schema_error"] = schema_err
                out["error"] = schema_err
                return out
            out["schema_ok"] = True
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
        out["error"] = sanitize_error_message(e)
        out["schema_ok"] = False
        return out


async def _page_alive(scraper: BetinAsiaScraper) -> bool:
    page = getattr(scraper, "_page", None)
    if page is None:
        return False
    try:
        if page.is_closed():
            return False
        await page.evaluate("1")
        return True
    except Exception:
        return False


async def recover_browser(scraper: BetinAsiaScraper, *, force_login: bool = True) -> Tuple[bool, Optional[str]]:
    """Restart Playwright browser + session using existing scraper login path."""
    try:
        try:
            await scraper.close()
        except Exception:
            pass
        await scraper.start()
        ok = await scraper.login(force=bool(force_login))
        if not ok:
            return False, ACCOUNTING_AUTH_FAILED
        if not await _page_alive(scraper):
            return False, ACCOUNTING_BROWSER_DEAD
        return True, None
    except Exception as e:
        return False, classify_exception(e)


def _customer_username() -> str:
    u = (os.getenv("BETINASIA_USERNAME") or os.getenv("ACCOUNTING_CUSTOMER_USERNAME") or "").strip()
    return u


def _api_balance_urls(username: str) -> Dict[str, str]:
    # Endpoints historically observed in successful CSV response captures.
    base = "https://black.betinasia.com"
    u = username.strip()
    return {
        "balance": f"{base}/v1/customers/{u}/balances/balance/?include_bet_details=true&layout=list&page_size=65535",
        "open_stakes": f"{base}/v1/customers/{u}/balances/stake/?include_bet_details=true&layout=list&page_size=65535",
    }


def _json_rows_to_csv_bytes(payload: Any) -> Optional[bytes]:
    """Best-effort convert API JSON list/dict into CSV bytes with required columns."""
    rows = None
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        for k in ("results", "rows", "data", "items"):
            if isinstance(payload.get(k), list):
                rows = payload.get(k)
                break
    if not isinstance(rows, list) or not rows:
        return None
    # flatten shallow keys; keep string values
    flat_rows: list[Dict[str, Any]] = []
    cols: list[str] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        fr: Dict[str, Any] = {}
        for k, v in r.items():
            key = str(k)
            # normalize common aliases
            kl = key.lower().replace("_", " ")
            if kl in ("order id", "orderid"):
                key = "order id"
            elif kl == "post date":
                key = "post date"
            elif kl == "got price":
                key = "got price"
            fr[key] = v if not isinstance(v, (dict, list)) else json.dumps(v, ensure_ascii=False)
            if key not in cols:
                cols.append(key)
        flat_rows.append(fr)
    if not flat_rows:
        return None
    # ensure required cols exist (empty if absent)
    for req in ("order id", "amount", "type", "post date"):
        if req not in cols:
            cols.insert(0, req)
    import io

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for fr in flat_rows:
        w.writerow({c: fr.get(c, "") for c in cols})
    return buf.getvalue().encode("utf-8")


async def _download_via_api(
    scraper: BetinAsiaScraper,
    *,
    name: str,
    url: str,
    out_dir: Path,
    timeout_ms: int = 60000,
) -> Tuple[Optional[Path], Dict[str, Any]]:
    """Authenticated fetch of accounting CSV/JSON using root-session (no UI click)."""
    meta: Dict[str, Any] = {"name": name, "url": url, "downloaded_via": None}
    page = scraper._page
    if page is None or page.is_closed():
        meta["error_type"] = ACCOUNTING_BROWSER_DEAD
        meta["error"] = "PAGE_CLOSED"
        return None, meta

    session_token = ""
    try:
        cookies = await scraper._context.cookies()
        for c in cookies or []:
            if c.get("name") == "root-session" and c.get("value"):
                session_token = str(c.get("value"))
                break
    except Exception as e:
        meta["error_type"] = classify_exception(e)
        meta["error"] = sanitize_error_message(e)
        return None, meta
    if not session_token:
        meta["error_type"] = ACCOUNTING_AUTH_FAILED
        meta["error"] = "NO_ROOT_SESSION"
        return None, meta

    # Ensure correct origin for relative fetch fallback
    try:
        if "betinasia.com" not in str(page.url or ""):
            await page.goto("https://black.betinasia.com/", timeout=timeout_ms, wait_until="commit")
    except Exception as e:
        meta["origin_goto_error"] = sanitize_error_message(e, limit=160)

    try:
        resp = await page.evaluate(
            """
            async (params) => {
              try {
                const r = await fetch(params.url, {
                  method: 'GET',
                  credentials: 'same-origin',
                  headers: {
                    'Accept': 'text/csv, application/json, text/plain, */*',
                    'session': params.sessionToken,
                    'x-molly-client-name': 'sonic',
                    'x-molly-client-version': '2.5.54'
                  }
                });
                const ct = (r.headers.get('content-type') || '');
                const text = await r.text();
                return {ok: r.ok, status: r.status, contentType: ct, text: text};
              } catch (e) {
                return {ok: false, status: 0, contentType: '', text: '', error: String(e && e.message ? e.message : e)};
              }
            }
            """,
            {"url": url, "sessionToken": session_token},
        )
    except Exception as e:
        meta["error_type"] = classify_exception(e)
        meta["error"] = sanitize_error_message(e)
        return None, meta

    if not isinstance(resp, dict):
        meta["error_type"] = ACCOUNTING_UNKNOWN_FAILURE
        meta["error"] = "BAD_API_RESP"
        return None, meta

    meta["http_status"] = resp.get("status")
    meta["content_type"] = str(resp.get("contentType") or "")[:120]
    status = int(resp.get("status") or 0)
    text = str(resp.get("text") or "")
    if status in (401, 403):
        meta["error_type"] = ACCOUNTING_AUTH_FAILED
        meta["error"] = f"HTTP_{status}"
        return None, meta
    if status == 429:
        meta["error_type"] = ACCOUNTING_RATE_LIMIT
        meta["error"] = "HTTP_429"
        return None, meta
    if not resp.get("ok") or status >= 400:
        meta["error_type"] = classify_exception(f"http {status} {text[:120]}")
        meta["error"] = sanitize_error_message(text[:200] or f"HTTP_{status}")
        return None, meta
    if not text.strip():
        meta["error_type"] = ACCOUNTING_EMPTY_RESPONSE
        meta["error"] = "empty body"
        return None, meta

    ct = meta["content_type"].lower()
    raw: Optional[bytes] = None
    if ("csv" in ct) or text.lstrip().lower().startswith("amount,") or ("order id" in text[:200].lower()):
        raw = text.encode("utf-8")
        meta["downloaded_via"] = "api_csv"
    else:
        try:
            payload = json.loads(text)
        except Exception:
            # maybe csv without header content-type
            if "," in text.splitlines()[0]:
                raw = text.encode("utf-8")
                meta["downloaded_via"] = "api_csv_guess"
            else:
                meta["error_type"] = ACCOUNTING_PARSE_FAILED
                meta["error"] = "unrecognized payload"
                return None, meta
        else:
            raw = _json_rows_to_csv_bytes(payload)
            meta["downloaded_via"] = "api_json_to_csv"
            if raw is None:
                meta["error_type"] = ACCOUNTING_EMPTY_RESPONSE
                meta["error"] = "json had no rows"
                return None, meta

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{ts}__{name}.csv"
    try:
        atomic_write_bytes(out_path, raw)
    except Exception as e:
        meta["error_type"] = ACCOUNTING_WRITE_FAILED
        meta["error"] = sanitize_error_message(e)
        return None, meta
    return out_path, meta


async def _download_from_accounting_page(
    scraper: BetinAsiaScraper,
    *,
    name: str,
    url: str,
    out_dir: Path,
    timeout_ms: int = 20000,
) -> Tuple[Optional[Path], Dict[str, Any]]:
    page = scraper._page
    assert page is not None
    meta: Dict[str, Any] = {"name": name, "url": url}

    if page.is_closed():
        meta["error"] = "PAGE_CLOSED"
        meta["error_type"] = ACCOUNTING_BROWSER_DEAD
        return None, meta

    try:
        resp = await page.goto(url, wait_until="commit", timeout=timeout_ms)
    except Exception as e:
        meta["error"] = sanitize_error_message(e)
        meta["error_type"] = classify_exception(e)
        return None, meta

    try:
        meta["http_status"] = int(resp.status) if resp else None
    except Exception:
        meta["http_status"] = None
    await page.wait_for_timeout(800)

    if meta.get("http_status") in (401, 403):
        meta["error_type"] = ACCOUNTING_AUTH_FAILED
        return None, meta

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
            try:
                async with page.expect_download(timeout=timeout_ms) as di:
                    await loc.first.click(force=True)
                dl = await di.value
                # download to temp then atomic rename
                tmp = out_path.with_name(out_path.name + f".{os.getpid()}.partial")
                await dl.save_as(str(tmp))
                os.replace(tmp, out_path)
                meta["downloaded_via"] = "expect_download"
                return out_path, meta
            except Exception as e1:
                meta.setdefault("attempt_errors", []).append({f"{sel}:download": sanitize_error_message(e1, limit=160)})
            try:

                def _is_csv_response(r):
                    try:
                        u = (r.url or "").lower()
                        ct = (r.headers.get("content-type", "") or "").lower()
                        cd = (r.headers.get("content-disposition", "") or "").lower()
                        return (
                            ("text/csv" in ct)
                            or ("application/csv" in ct)
                            or ("attachment" in cd and "csv" in cd)
                            or ("csv" in u and ("export" in u or "download" in u))
                            or ("/balances/" in u and "layout=list" in u)
                        )
                    except Exception:
                        return False

                async with page.expect_response(_is_csv_response, timeout=timeout_ms) as ri:
                    await loc.first.click(force=True)
                r = await ri.value
                body = await r.body()
                if not body:
                    meta["error_type"] = ACCOUNTING_EMPTY_RESPONSE
                    meta.setdefault("attempt_errors", []).append({f"{sel}:empty": "empty body"})
                    continue
                atomic_write_bytes(out_path, body)
                meta["downloaded_via"] = "expect_response_csv"
                meta["csv_resp_url"] = getattr(r, "url", None)
                meta["csv_resp_status"] = getattr(r, "status", None)
                return out_path, meta
            except Exception as e2:
                meta.setdefault("attempt_errors", []).append({f"{sel}:response": sanitize_error_message(e2, limit=160)})
        except Exception as e:
            meta.setdefault("attempt_errors", []).append({sel: sanitize_error_message(e, limit=160)})
            continue

    # fallback hrefs
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
        for it in hrefs or []:
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
                tmp = out_path.with_name(out_path.name + f".{os.getpid()}.partial")
                await dl.save_as(str(tmp))
                os.replace(tmp, out_path)
                meta["downloaded_via"] = "href_click"
                return out_path, meta
            except Exception as e:
                meta.setdefault("attempt_errors", []).append({f"href:{h}": sanitize_error_message(e, limit=160)})
                continue
    except Exception as e:
        meta["fallback_error"] = sanitize_error_message(e)

    if not meta.get("error_type"):
        # infer from attempt errors
        joined = json.dumps(meta.get("attempt_errors") or [])[:500]
        meta["error_type"] = classify_exception(joined) if joined else ACCOUNTING_EMPTY_RESPONSE
    return None, meta


@dataclass
class AccountingConfig:
    poll_sec: float = 300.0
    out_dir: Path = Path("logs/accounting")
    jsonl_path: Path = Path("logs/accounting_snapshots.jsonl")
    health_path: Path = Path("logs/accounting/accounting_health.json")
    once: bool = False
    timeout_ms: int = 20000
    max_retries: int = 2
    retry_backoff_sec: float = 2.0


def _file_age_sec(path: Optional[Path]) -> Optional[float]:
    if not path or not path.exists():
        return None
    try:
        return max(0.0, time.time() - float(path.stat().st_mtime))
    except Exception:
        return None


def _source_block(
    *,
    ok: bool,
    path: Optional[Path],
    parsed: Optional[Dict[str, Any]],
    now_ts: float,
) -> Dict[str, Any]:
    mtime = None
    age = None
    if path and path.exists():
        try:
            mtime_f = float(path.stat().st_mtime)
            mtime = datetime.fromtimestamp(mtime_f, timezone.utc).isoformat()
            age = max(0.0, now_ts - mtime_f)
        except Exception:
            pass
    return {
        "source_ok": bool(ok),
        "rows": int((parsed or {}).get("rows") or 0) if ok else 0,
        "source_ts": mtime,
        "file_path": str(path) if path else None,
        "file_mtime_utc": mtime,
        "age_sec": age,
    }


async def run_one_cycle(
    scraper: BetinAsiaScraper,
    cfg: AccountingConfig,
    *,
    consecutive_failures: int,
    limits: FreshnessLimits,
) -> Dict[str, Any]:
    run_id = str(uuid.uuid4())
    t0 = time.time()
    ui_urls = {
        "balance": "https://black.betinasia.com/accounting/balance",
        "open_stakes": "https://black.betinasia.com/accounting/open-stakes",
    }
    username = _customer_username()
    api_urls = _api_balance_urls(username) if username else {}
    prefer_api = os.getenv("ACCOUNTING_PREFER_API", "1").strip() not in ("0", "false", "False", "no", "NO")
    require_api_auth = os.getenv("ACCOUNTING_REQUIRE_API_AUTH", "1").strip() not in ("0", "false", "False", "no", "NO")
    snap: Dict[str, Any] = {
        "ts": _utcnow(),
        "run_id": run_id,
        "files": {},
        "parsed": {},
        "meta": {},
        "status": ACCOUNTING_UNKNOWN_FAILURE,
    }

    browser_ok = await _page_alive(scraper)
    session_ok = False
    proxy_ok = True
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    if not browser_ok:
        recovered, et = await recover_browser(scraper, force_login=True)
        browser_ok = recovered
        if not recovered:
            error_type = et or ACCOUNTING_BROWSER_DEAD
            error_message = "browser recover failed"

    async def _session_api_ok() -> bool:
        if not username:
            return False
        try:
            chk = await scraper._api_auth_check_orders(username=username)
            return bool(chk.get("ok")) and int(chk.get("status") or 0) == 200
        except Exception:
            return False

    if browser_ok:
        try:
            has_root = bool(await scraper._has_root_session_cookie())
        except Exception:
            has_root = False
        try:
            api_ok = await _session_api_ok() if has_root else False
        except Exception as e:
            api_ok = False
            error_type = classify_exception(e)
            error_message = sanitize_error_message(e)
        # Accounting needs root-session cookie; orders API check is best-effort.
        session_ok = bool(has_root and (api_ok or not require_api_auth or has_root))
        # Simplify: root cookie is sufficient to attempt downloads.
        session_ok = bool(has_root)
        if not session_ok:
            try:
                ok = await scraper.login(force=True)
                has_root = bool(ok) and bool(await scraper._has_root_session_cookie())
                session_ok = has_root
                if not session_ok:
                    error_type = ACCOUNTING_AUTH_FAILED
                    error_message = "LOGIN_FAILED_OR_NO_ROOT_SESSION"
            except Exception as e:
                error_type = classify_exception(e)
                error_message = sanitize_error_message(e)
        elif require_api_auth and not api_ok:
            # Soft warning only — still attempt balance/open_stakes fetches.
            error_message = error_message or "ORDERS_API_AUTH_CHECK_FAILED"

    files_ok: Dict[str, bool] = {"balance": False, "open_stakes": False}
    paths: Dict[str, Optional[Path]] = {"balance": None, "open_stakes": None}

    async def _accept_path(name: str, p: Path, meta: Dict[str, Any]) -> bool:
        nonlocal error_type, error_message
        parsed = _parse_csv_best_effort(p)
        snap["parsed"][name] = parsed
        snap["meta"][name] = meta
        if parsed.get("schema_ok") is False:
            error_type = ACCOUNTING_SCHEMA_CHANGED if "missing columns" in str(parsed.get("error") or "") else ACCOUNTING_PARSE_FAILED
            error_message = sanitize_error_message(parsed.get("error"))
            paths[name] = p
            files_ok[name] = False
            snap["files"][name] = str(p)
            return False
        paths[name] = p
        files_ok[name] = True
        snap["files"][name] = str(p)
        return True

    async def _fetch_one(name: str, ui_url: str) -> None:
        nonlocal error_type, error_message, browser_ok
        last_meta: Dict[str, Any] = {"name": name, "url": ui_url}
        for attempt in range(int(cfg.max_retries) + 1):
            if not await _page_alive(scraper):
                recovered, et = await recover_browser(scraper, force_login=True)
                browser_ok = recovered
                if not recovered:
                    last_meta = {"name": name, "url": ui_url, "error_type": et or ACCOUNTING_BROWSER_DEAD, "error": "browser dead"}
                    error_type = et or ACCOUNTING_BROWSER_DEAD
                    break
            try:
                p = None
                meta: Dict[str, Any] = {"name": name}
                if prefer_api and api_urls.get(name):
                    p, meta = await _download_via_api(
                        scraper,
                        name=name,
                        url=api_urls[name],
                        out_dir=cfg.out_dir,
                        timeout_ms=max(int(cfg.timeout_ms), 60000),
                    )
                    if p and p.exists() and p.stat().st_size > 0:
                        if await _accept_path(name, p, meta):
                            return
                        return
                    # auth failure -> force re-login once
                    if meta.get("error_type") == ACCOUNTING_AUTH_FAILED and attempt < int(cfg.max_retries):
                        await scraper.login(force=True)
                        await asyncio.sleep(float(cfg.retry_backoff_sec) * (attempt + 1))
                        continue
                # UI fallback
                p, meta = await _download_from_accounting_page(
                    scraper,
                    name=name,
                    url=ui_url,
                    out_dir=cfg.out_dir,
                    timeout_ms=max(int(cfg.timeout_ms), 60000),
                )
                last_meta = meta
                if p and p.exists() and p.stat().st_size > 0:
                    if await _accept_path(name, p, meta):
                        return
                    return
                et = meta.get("error_type") or classify_exception(meta.get("error") or ACCOUNTING_EMPTY_RESPONSE)
                if et in (ACCOUNTING_BROWSER_DEAD, ACCOUNTING_TIMEOUT, ACCOUNTING_AUTH_FAILED) and attempt < int(cfg.max_retries):
                    if et == ACCOUNTING_BROWSER_DEAD:
                        await recover_browser(scraper, force_login=True)
                    elif et == ACCOUNTING_AUTH_FAILED:
                        await scraper.login(force=True)
                    await asyncio.sleep(float(cfg.retry_backoff_sec) * (attempt + 1))
                    continue
                error_type = error_type or et
                error_message = error_message or sanitize_error_message(meta.get("error") or et)
            except Exception as e:
                last_meta = {"name": name, "url": ui_url, "error": sanitize_error_message(e), "error_type": classify_exception(e)}
                et = classify_exception(e)
                if et == ACCOUNTING_BROWSER_DEAD and attempt < int(cfg.max_retries):
                    await recover_browser(scraper, force_login=True)
                    await asyncio.sleep(float(cfg.retry_backoff_sec) * (attempt + 1))
                    continue
                error_type = error_type or et
                error_message = sanitize_error_message(e)
        snap["meta"][name] = last_meta
        snap["files"][name] = str(paths[name]) if paths[name] else None

    if browser_ok and session_ok:
        for name, url in ui_urls.items():
            await _fetch_one(name, url)
    else:
        for name, url in ui_urls.items():
            snap["files"][name] = None
            snap["meta"][name] = {
                "name": name,
                "url": url,
                "error_type": error_type or (ACCOUNTING_AUTH_FAILED if not session_ok else ACCOUNTING_BROWSER_DEAD),
                "skipped": True,
            }

    status = cycle_status(
        balance_ok=bool(files_ok["balance"]),
        open_ok=bool(files_ok["open_stakes"]),
        error_type=error_type,
    )
    # Never treat files=None as OK
    if snap["files"].get("balance") is None and snap["files"].get("open_stakes") is None and status == ACCOUNTING_OK:
        status = ACCOUNTING_EMPTY_RESPONSE

    snap["status"] = status
    snap["error_type"] = error_type
    snap["error_message"] = error_message
    snap["session_ok"] = session_ok
    snap["browser_ok"] = browser_ok
    snap["proxy_ok"] = proxy_ok

    # append jsonl (cycle ledger)
    cfg.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(snap, ensure_ascii=False) + "\n")

    now_ts = time.time()
    # For ages: prefer this cycle's files; else latest existing on disk (staleness visibility)
    bal_path = paths["balance"]
    open_path = paths["open_stakes"]
    if bal_path is None:
        cands = sorted(cfg.out_dir.glob("*__balance.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        bal_path_age = cands[0] if cands else None
    else:
        bal_path_age = bal_path
    if open_path is None:
        cands = sorted(cfg.out_dir.glob("*__open_stakes.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        open_path_age = cands[0] if cands else None
    else:
        open_path_age = open_path

    bal_age = _file_age_sec(bal_path_age)
    open_age = _file_age_sec(open_path_age)
    next_failures = 0 if status == ACCOUNTING_OK else int(consecutive_failures) + 1
    health = classify_health(
        status=status,
        balance_age_sec=bal_age,
        open_age_sec=open_age,
        consecutive_failures=next_failures,
        limits=limits,
    )

    health_doc = {
        "checked_at_utc": _utcnow(),
        "status": status,
        "health": health,
        "balance": _source_block(
            ok=bool(files_ok["balance"]),
            path=paths["balance"] or bal_path_age,
            parsed=snap["parsed"].get("balance"),
            now_ts=now_ts,
        ),
        "open_stakes": _source_block(
            ok=bool(files_ok["open_stakes"]),
            path=paths["open_stakes"] or open_path_age,
            parsed=snap["parsed"].get("open_stakes"),
            now_ts=now_ts,
        ),
        "session_ok": session_ok,
        "proxy_ok": proxy_ok,
        "api_status": status,
        "error_type": error_type,
        "error_message": error_message,
        "run_id": run_id,
        "duration_ms": int((time.time() - t0) * 1000),
        "consecutive_failures": next_failures,
        "limits": {
            "warn_stale_sec": limits.warn_stale_sec,
            "critical_stale_sec": limits.critical_stale_sec,
            "max_consecutive_failures": limits.max_consecutive_failures,
        },
    }
    # If cycle failed, ages reflect last good files — mark source_ok false when this cycle failed that source
    if not files_ok["balance"]:
        health_doc["balance"]["source_ok"] = False
    if not files_ok["open_stakes"]:
        health_doc["open_stakes"]["source_ok"] = False

    try:
        atomic_write_json(cfg.health_path, health_doc)
    except Exception as e:
        logger.error(f"[acct] health write failed: {sanitize_error_message(e)}")
        snap["status"] = ACCOUNTING_WRITE_FAILED if status == ACCOUNTING_OK else status

    dt = time.time() - t0
    level = "info" if status == ACCOUNTING_OK else ("warning" if status == ACCOUNTING_PARTIAL else "error")
    msg = f"[acct] status={status} health={health} dt={dt:.1f}s files={snap['files']} err={error_type}"
    getattr(logger, level)(msg)
    snap["_health"] = health_doc
    return snap


async def run_monitor(cfg: AccountingConfig) -> int:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.health_path.parent.mkdir(parents=True, exist_ok=True)
    limits = FreshnessLimits.from_env()

    scraper = BetinAsiaScraper()
    await scraper.start()
    ok = await scraper.login()
    if not ok:
        fail = {
            "ts": _utcnow(),
            "status": ACCOUNTING_AUTH_FAILED,
            "error": "LOGIN_FAILED",
            "error_type": ACCOUNTING_AUTH_FAILED,
            "files": {"balance": None, "open_stakes": None},
            "meta": {},
        }
        with cfg.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(fail, ensure_ascii=False) + "\n")
        health_doc = {
            "checked_at_utc": _utcnow(),
            "status": ACCOUNTING_AUTH_FAILED,
            "health": "CRITICAL",
            "balance": {"source_ok": False, "rows": 0, "source_ts": None, "file_path": None, "file_mtime_utc": None, "age_sec": None},
            "open_stakes": {"source_ok": False, "rows": 0, "source_ts": None, "file_path": None, "file_mtime_utc": None, "age_sec": None},
            "session_ok": False,
            "proxy_ok": True,
            "api_status": ACCOUNTING_AUTH_FAILED,
            "error_type": ACCOUNTING_AUTH_FAILED,
            "error_message": "LOGIN_FAILED",
            "run_id": str(uuid.uuid4()),
            "duration_ms": 0,
            "consecutive_failures": 1,
        }
        atomic_write_json(cfg.health_path, health_doc)
        raise RuntimeError("LOGIN_FAILED")

    logger.info(
        f"[acct] started poll_sec={cfg.poll_sec} out_dir={cfg.out_dir} jsonl={cfg.jsonl_path} health={cfg.health_path}"
    )

    consecutive_failures = 0
    while True:
        snap = await run_one_cycle(scraper, cfg, consecutive_failures=consecutive_failures, limits=limits)
        status = str(snap.get("status") or "")
        if status == ACCOUNTING_OK:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            # recover aggressively after repeated failures
            if consecutive_failures >= max(1, limits.max_consecutive_failures):
                await recover_browser(scraper, force_login=True)

        if cfg.once:
            break
        dt = float(snap.get("_health", {}).get("duration_ms") or 0) / 1000.0
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
    default_jsonl = normalize_jsonl_path(os.getenv("ACCOUNTING_JSONL", "logs/accounting_snapshots.jsonl"))
    ap.add_argument("--jsonl", default=default_jsonl)
    ap.add_argument(
        "--health",
        default=os.getenv("ACCOUNTING_HEALTH_JSON", "logs/accounting/accounting_health.json"),
    )
    ap.add_argument(
        "--once",
        action="store_true",
        default=(os.getenv("ACCOUNTING_ONCE", "0").strip() in ("1", "true", "True", "yes", "YES")),
    )
    ap.add_argument("--timeout-ms", type=int, default=int(os.getenv("ACCOUNTING_TIMEOUT_MS", "60000")))
    ap.add_argument("--max-retries", type=int, default=int(os.getenv("ACCOUNTING_MAX_RETRIES", "2")))
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))

    cfg = AccountingConfig(
        poll_sec=float(args.poll_sec),
        out_dir=Path(str(args.out_dir)),
        jsonl_path=Path(str(args.jsonl)),
        health_path=Path(str(args.health)),
        once=bool(args.once),
        timeout_ms=int(args.timeout_ms),
        max_retries=int(args.max_retries),
    )
    asyncio.run(run_monitor(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
