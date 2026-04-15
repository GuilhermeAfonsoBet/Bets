import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
 
 
def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        s = s.replace(",", ".")
        out = []
        for ch in s:
            if ch.isdigit() or ch in ".-":
                out.append(ch)
        s2 = "".join(out).strip()
        if s2 in ("", "-", ".", "-."):
            return None
        return float(s2)
    except Exception:
        return None
 
 
def _pct(num: Any, den: Any) -> Optional[float]:
    try:
        n = float(num)
        d = float(den)
        if d <= 0:
            return None
        return float(n / d * 100.0)
    except Exception:
        return None
 
 
def _parse_day(s: str) -> Optional[str]:
    s = str(s or "").strip()
    if not s:
        return None
    # aceita YYYY-MM-DD
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return s
    except Exception:
        return None
 
 
def _repo_roots() -> List[Path]:
    """
    Retorna possíveis roots para resolver paths relativos (repo e package).
    """
    roots: List[Path] = []
    try:
        roots.append(Path(__file__).resolve().parents[2])  # .../Bets
    except Exception:
        pass
    try:
        roots.append(Path(__file__).resolve().parents[1])  # .../Bets/betinasia_bot
    except Exception:
        pass
    try:
        roots.append(Path.cwd())
    except Exception:
        pass
    out: List[Path] = []
    seen: set[str] = set()
    for r in roots:
        try:
            rr = r.resolve()
            if str(rr) not in seen:
                out.append(rr)
                seen.add(str(rr))
        except Exception:
            continue
    return out


def _resolve_rel_path(p: Path, *, extra_roots: Optional[List[Path]] = None) -> Path:
    """
    Resolve um path potencialmente relativo usando roots plausíveis do repo/package.
    """
    try:
        if p.is_absolute():
            return p
    except Exception:
        pass
    roots = list(_repo_roots())
    for r in (extra_roots or []):
        try:
            roots.insert(0, Path(r).resolve())
        except Exception:
            continue
    for root in roots:
        cand = (root / p).resolve()
        if cand.exists():
            return cand
    # fallback: primeiro root + p
    try:
        return (roots[0] / p).resolve() if roots else p
    except Exception:
        return p


def _bucket_slip_raw_3way(slip_raw_pct: Any) -> str:
    x = _safe_float(slip_raw_pct)
    if x is None:
        return "Desconhecido"
    try:
        v = float(x)
    except Exception:
        return "Desconhecido"
    if v <= -2.0:
        return "<= -2%"
    if v <= 2.0:
        return "(-2, 2]"
    return "> 2%"
 
 
def _bucket_call_to_done_ms(lat_ms: Any) -> str:
    x = _safe_float(lat_ms)
    if x is None:
        return "Desconhecido"
    try:
        v = float(x)
    except Exception:
        return "Desconhecido"
    if v < 5000:
        return "< 5s"
    if v < 10000:
        return "5-10s"
    if v < 20000:
        return "10-20s"
    if v < 40000:
        return "20-40s"
    return "> 40s"
 
 
def _roi_weighted(pnl_sum: float, exp_sum: float) -> Optional[float]:
    try:
        if exp_sum <= 0:
            return None
        return float(pnl_sum / exp_sum * 100.0)
    except Exception:
        return None
 
 
@dataclass
class Summary:
    n_orders: int
    n_games: int
    exposure_sum: float
    pnl_sum: float
    roi_w: Optional[float]
 
 
def _summ(rows: List[dict]) -> Summary:
    evs = set()
    exp = 0.0
    pnl = 0.0
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        eid = str(r.get("event_id") or "").strip()
        if eid:
            evs.add(eid)
        try:
            pnl += float(r.get("pnl") or 0.0)
        except Exception:
            pass
        try:
            exp += float(r.get("exposure") or 0.0)
        except Exception:
            pass
    return Summary(
        n_orders=int(len(rows or [])),
        n_games=int(len(evs)),
        exposure_sum=float(exp),
        pnl_sum=float(pnl),
        roi_w=_roi_weighted(pnl, exp),
    )


def _breakdown_by_bucket(rows: List[dict], *, key: str, order: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Retorna breakdown por bucket (ex.: lat_bucket, slip_bucket):
      [{bucket, n_orders, n_games, exposure_sum, pnl_sum, roi_w_pct}]
    """
    by: Dict[str, List[dict]] = {}
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        b = str(r.get(key) or "Desconhecido")
        by.setdefault(b, []).append(r)
    buckets = list(by.keys())
    if order:
        buckets = [b for b in order if b in by] + [b for b in buckets if b not in set(order)]
    else:
        buckets = sorted(buckets)
    out: List[Dict[str, Any]] = []
    for b in buckets:
        summ = _summ(by.get(b) or [])
        out.append(
            {
                "bucket": b,
                "n_orders": int(summ.n_orders),
                "n_games": int(summ.n_games),
                "exposure_sum": float(summ.exposure_sum),
                "pnl_sum": float(summ.pnl_sum),
                "roi_w_pct": summ.roi_w,
            }
        )
    return out


def _breakdown_by_combo(rows: List[dict], *, lat_key: str, slip_key: str) -> List[Dict[str, Any]]:
    """
    2D breakdown lat×slip:
      [{lat_bucket, slip_bucket, n_orders, n_games, exposure_sum, pnl_sum, roi_w_pct}]
    """
    by: Dict[Tuple[str, str], List[dict]] = {}
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        lb = str(r.get(lat_key) or "Desconhecido")
        sb = str(r.get(slip_key) or "Desconhecido")
        by.setdefault((lb, sb), []).append(r)
    keys = sorted(by.keys(), key=lambda x: (x[0], x[1]))
    out: List[Dict[str, Any]] = []
    for (lb, sb) in keys:
        summ = _summ(by.get((lb, sb)) or [])
        out.append(
            {
                "lat_bucket": lb,
                "slip_bucket": sb,
                "n_orders": int(summ.n_orders),
                "n_games": int(summ.n_games),
                "exposure_sum": float(summ.exposure_sum),
                "pnl_sum": float(summ.pnl_sum),
                "roi_w_pct": summ.roi_w,
            }
        )
    return out
 
 
def _quantile(xs: List[float], q: float) -> Optional[float]:
    try:
        if not xs:
            return None
        xs2 = sorted(float(x) for x in xs)
        if len(xs2) == 1:
            return float(xs2[0])
        pos = q * (len(xs2) - 1)
        lo = int(pos)
        hi = min(len(xs2) - 1, lo + 1)
        w = float(pos - lo)
        return float(xs2[lo] * (1.0 - w) + xs2[hi] * w)
    except Exception:
        return None
 
 
def _bootstrap_by_game(
    *,
    by_game: Dict[str, Dict[str, float]],
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    """
    by_game[event_id] = {pnl_base, exp_base, pnl_sub, exp_sub}
    """
    evs = [e for e in (by_game or {}).keys() if str(e)]
    if not evs:
        return {"n_games": 0}
    rng = random.Random(int(seed))
    n = int(len(evs))
    roi_base = []
    roi_sub = []
    delta = []
    for _ in range(int(max(0, n_boot))):
        pnl_b = exp_b = pnl_s = exp_s = 0.0
        for _j in range(n):
            e = evs[rng.randrange(0, n)]
            rec = by_game.get(e) or {}
            pnl_b += float(rec.get("pnl_base") or 0.0)
            exp_b += float(rec.get("exp_base") or 0.0)
            pnl_s += float(rec.get("pnl_sub") or 0.0)
            exp_s += float(rec.get("exp_sub") or 0.0)
        rb = _roi_weighted(pnl_b, exp_b)
        rs = _roi_weighted(pnl_s, exp_s)
        if rb is not None:
            roi_base.append(float(rb))
        if rs is not None:
            roi_sub.append(float(rs))
        if rb is not None and rs is not None:
            delta.append(float(rs - rb))
    out: Dict[str, Any] = {
        "n_games": n,
        "n_boot": int(n_boot),
        "roi_base_ci90": {"lb": _quantile(roi_base, 0.05), "ub": _quantile(roi_base, 0.95)},
        "roi_sub_ci90": {"lb": _quantile(roi_sub, 0.05), "ub": _quantile(roi_sub, 0.95)},
        "delta_ci90": {"lb": _quantile(delta, 0.05), "ub": _quantile(delta, 0.95)},
    }
    try:
        out["p_roi_sub_gt0"] = (sum(1 for x in roi_sub if float(x) > 0.0) / float(len(roi_sub))) if roi_sub else None
    except Exception:
        out["p_roi_sub_gt0"] = None
    try:
        out["p_delta_gt0"] = (sum(1 for x in delta if float(x) > 0.0) / float(len(delta))) if delta else None
    except Exception:
        out["p_delta_gt0"] = None
    return out
 
 
def _sign_test_p(n_pos: int, n_total: int) -> Optional[float]:
    """
    Sign test bicaudal exato.
    H0: p=0.5 para "dia com delta>0".
    Retorna p-value (2-sided), com cap em 1.0.
    """
    try:
        import math
    except Exception:
        return None
    try:
        n = int(n_total)
        if n <= 0:
            return None
        k = max(0, min(n, int(n_pos)))
        # 2-sided: 2 * min( P(X<=k), P(X>=k) ).
        # Forma equivalente: use k' = min(k, n-k) e some a cauda inferior.
        k2 = min(k, n - k)
        p = 0.0
        for i in range(0, k2 + 1):
            p += math.comb(n, i) * (0.5**n)
        return float(min(1.0, 2.0 * p))
    except Exception:
        return None


async def _run(
    *,
    day_dir: Path,
    start_day: str,
    end_day: Optional[str],
    regime: str,
    lat_bucket: str,
    slip_bucket: str,
    database_url_override: Optional[str],
    balance_csv_override: Optional[str],
    n_boot: int,
    seed: int,
    limit_stake_factor: float,
    limit_stake_cap: float,
) -> Dict[str, Any]:
    # imports tardios (DB + helpers do daily)
    #
    # IMPORTANTE: não use `betinasia_bot.storage.Database` aqui.
    # Ele depende de um import top-level `config`, que quebra dependendo do cwd/PYTHONPATH no VPS.
    # Para este teste, basta um SELECT simples via SQLAlchemy usando DATABASE_URL.
    try:
        from sqlalchemy import text  # type: ignore
        from sqlalchemy.ext.asyncio import create_async_engine  # type: ignore
    except Exception as e:
        raise SystemExit(f"Dependência faltando: SQLAlchemy. Erro: {e}")

    # Evitar importar `ops.daily_full_report` aqui porque ele puxa `accounting_monitor` (scraper/Playwright),
    # o que não é necessário para este teste e pode quebrar em ambientes sem `scraper`.
    from datetime import timedelta
    import csv
    import re

    def _parse_dt_any(s: Any) -> Optional[datetime]:
        try:
            t = str(s or "").strip()
            if not t:
                return None
            if t.endswith("Z"):
                t = t[:-1] + "+00:00"
            try:
                dt = datetime.fromisoformat(t)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                pass
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(t, fmt).replace(tzinfo=timezone.utc)
                    return dt
                except Exception:
                    continue
        except Exception:
            return None
        return None

    def _to_utc_dt(x: Any) -> Optional[datetime]:
        if x is None:
            return None
        if isinstance(x, datetime):
            return x.astimezone(timezone.utc) if x.tzinfo else x.replace(tzinfo=timezone.utc)
        return _parse_dt_any(x)

    def _is_inplay_from_audit_row(a: Optional[Dict[str, Any]], *, exec_created_at_utc: datetime) -> bool:
        try:
            if not isinstance(exec_created_at_utc, datetime):
                return False
            if a and a.get("kickoff_time") is not None:
                ko = _to_utc_dt(a.get("kickoff_time"))
                if isinstance(ko, datetime):
                    return exec_created_at_utc.astimezone(timezone.utc) >= ko.astimezone(timezone.utc)
        except Exception:
            pass
        try:
            if a and a.get("is_live") is not None:
                return bool(a.get("is_live"))
        except Exception:
            pass
        return False

    def _extract_order_id_from_raw(raw: Dict[str, Any]) -> Optional[str]:
        if not isinstance(raw, dict):
            return None
        for k in ("order_id", "orderId", "bet_id", "betId", "id"):
            v = raw.get(k)
            if v is None:
                continue
            s = str(v).strip()
            if s.isdigit():
                return s
        # fallback: scan strings
        try:
            s = json.dumps(raw, ensure_ascii=False)
            m = re.search(r"\"order[_ ]?id\"\\s*:\\s*\"?(\\d+)\"?", s, flags=re.IGNORECASE)
            if m:
                return str(m.group(1))
        except Exception:
            pass
        return None

    def _parse_executor_jsonl_back_live_orders(path: Path) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        if not path.exists():
            return out
        for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            req = obj.get("request") if isinstance(obj.get("request"), dict) else {}
            res = obj.get("result") if isinstance(obj.get("result"), dict) else {}
            st = str(res.get("status") or "").strip()
            if st != "LIVE_OK":
                continue
            exec_side = str(res.get("exec_side") or req.get("exec_side") or "").strip().lower()
            if exec_side != "back":
                continue
            created = _parse_dt_any(str(res.get("created_at") or req.get("created_at") or ""))
            if created is None:
                continue
            raw = res.get("raw") if isinstance(res.get("raw"), dict) else {}
            oid = _extract_order_id_from_raw(raw)
            if not oid or not str(oid).isdigit():
                continue
            # audit_id
            audit_id = None
            try:
                v = res.get("audit_id") if res.get("audit_id") is not None else req.get("audit_id")
                audit_id = int(v) if v is not None else None
            except Exception:
                audit_id = None
            # stake (exposure)
            sent = raw.get("sent") if isinstance(raw.get("sent"), dict) else {}
            stake = _safe_float(sent.get("stake"))
            if stake is None:
                pol = res.get("policy") if isinstance(res.get("policy"), dict) else (req.get("policy") if isinstance(req.get("policy"), dict) else {})
                stake = _safe_float((pol or {}).get("stake_requested"))
            timing = res.get("timing") if isinstance(res.get("timing"), dict) else {}
            lat_ms = _safe_float(timing.get("call_to_done_ms"))
            lat_ms_i = int(lat_ms) if lat_ms is not None else None
            odd_dec = _safe_float(res.get("odd_at_decision") if res.get("odd_at_decision") is not None else req.get("odd_at_decision"))
            odd_fin = _safe_float(res.get("odd_final"))
            slip = None
            try:
                if odd_dec is not None and odd_fin is not None and float(odd_dec) != 0.0:
                    slip = (float(odd_fin) - float(odd_dec)) / float(odd_dec) * 100.0
            except Exception:
                slip = None
            rec = {
                "order_id": str(oid),
                "created_at": created.astimezone(timezone.utc),
                "slip_raw_pct": slip,
                "lat_ms": (float(lat_ms_i) if lat_ms_i is not None else None),
                "exposure": (float(stake) if stake is not None else None),
                "audit_id": (int(audit_id) if audit_id is not None else None),
            }
            prev = out.get(str(oid))
            if prev is None or rec["created_at"] >= prev.get("created_at"):
                out[str(oid)] = rec
        return out

    def _extract_audit_ids_from_exec_by_oid(exec_by_oid: Dict[str, Dict[str, Any]]) -> List[int]:
        out: set[int] = set()
        for _, ex in (exec_by_oid or {}).items():
            if not isinstance(ex, dict):
                continue
            aid = ex.get("audit_id")
            if aid is None:
                continue
            try:
                out.add(int(aid))
            except Exception:
                continue
        return sorted(list(out))

    def _load_env_file(path: Path) -> None:
        # loader minimalista .env (sem dependências)
        try:
            if not path.exists():
                return
            for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                s = ln.strip()
                if (not s) or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'").strip('"')
                if k and (k not in os.environ):
                    os.environ[k] = v
        except Exception:
            return

    def _acct_pnl_like_by_order_total_from_balance_csv(path: Path) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if not path.exists():
            return out
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            r = csv.DictReader(f)
            cols = list(r.fieldnames or [])
            if not cols:
                return out
            # colunas
            def _pick(keys: Tuple[str, ...]) -> Optional[str]:
                for k in keys:
                    for c in cols:
                        cl = c.lower()
                        if cl == k or cl.startswith(k) or k in cl:
                            return c
                return None

            pnl_col = _pick(("amount", "profit_loss", "profit", "p&l", "pnl", "net", "pl"))
            oid_col = _pick(("order_id", "order id", "order", "bet id", "bet_id", "id"))
            typ_col = _pick(("type",))
            if not pnl_col or not oid_col:
                return out
            for row in r:
                if not isinstance(row, dict):
                    continue
                oid = str(row.get(oid_col) or "").strip()
                if not oid or not oid.isdigit():
                    continue
                amt = _safe_float(row.get(pnl_col))
                if amt is None:
                    continue
                tl = str(row.get(typ_col) or "").strip().lower() if typ_col else ""
                # inclui tudo P&L-like: exclui depósitos/saques/etc.
                if any(k in tl for k in ("deposit", "withdraw", "transfer", "top", "payment", "adjust", "bonus")):
                    continue
                out[oid] = float(out.get(oid) or 0.0) + float(amt)
        return out
 
    _load_env_file(Path(os.getenv("ENV_FILE", ".env")))

    def _pick_database_url() -> str:
        # 1) override explícito
        if database_url_override:
            return str(database_url_override).strip()
        # 2) env
        u = str(os.getenv("DATABASE_URL", "") or "").strip()
        if u:
            return u
        # 3) fallback: settings do package (carrega betinasia_bot/.env por padrão)
        try:
            from betinasia_bot.config import settings  # type: ignore

            u2 = str(getattr(settings, "database_url", "") or "").strip()
            if u2:
                return u2
        except Exception:
            pass
        return ""

    database_url = _pick_database_url()
    if not database_url:
        raise SystemExit(
            "DATABASE_URL não está definido. "
            "Passe `--database-url` ou exporte DATABASE_URL, ou aponte ENV_FILE para um .env que contenha DATABASE_URL."
        )

    acct_json = day_dir / "accounting_daily_report.json"
    if not acct_json.exists():
        raise SystemExit(f"Não achei `{acct_json}`. Passe `--day-dir` correto (o day_dir impresso pelo daily).")
    acct = json.loads(acct_json.read_text(encoding="utf-8", errors="ignore") or "{}")
    # balance.csv: pode ser relativo ao repo OU ao package (dependendo do cwd em que o daily rodou).
    bal_csv_raw = str(acct.get("balance_csv") or "").strip()
    if balance_csv_override:
        bal_csv_raw = str(balance_csv_override).strip()
    bal_csv = Path(bal_csv_raw).expanduser() if bal_csv_raw else Path("")
    if not str(bal_csv):
        raise SystemExit(f"`balance_csv` ausente em `{acct_json}`. Rode o daily com accounting habilitado ou passe `--balance-csv`.")
    bal_csv = _resolve_rel_path(bal_csv, extra_roots=[day_dir, day_dir.parent, day_dir.parent.parent])
    if not bal_csv.exists():
        raise SystemExit(
            f"Não achei balance.csv em `{bal_csv_raw}` (resolvido para `{bal_csv}`) "
            f"(de `{acct_json}`). Passe `--balance-csv` apontando para o arquivo correto."
        )
 
    executor_jsonl = Path(os.getenv("EXECUTOR_JSONL", "logs/executor_live.jsonl"))
    if not executor_jsonl.is_absolute():
        # Tenta resolver relativo ao root do repo (…/Bets) e ao root do package (…/Bets/betinasia_bot).
        executor_jsonl = _resolve_rel_path(executor_jsonl, extra_roots=[day_dir, day_dir.parent, day_dir.parent.parent])
    if not executor_jsonl.exists():
        raise SystemExit(f"Não achei executor_jsonl em `{executor_jsonl}`. Sete `EXECUTOR_JSONL` (absoluto) ou rode do repo.")
 
    start_day2 = _parse_day(start_day)
    end_day2 = _parse_day(end_day) if end_day else None
    if not start_day2:
        raise SystemExit("`--start-day` inválido. Use YYYY-MM-DD.")
    if end_day and not end_day2:
        raise SystemExit("`--end-day` inválido. Use YYYY-MM-DD.")
 
    exec_by_oid = _parse_executor_jsonl_back_live_orders(executor_jsonl)
    audit_ids = _extract_audit_ids_from_exec_by_oid(exec_by_oid)

    async def _fetch_audit_rows_for_ids(*, ids: List[int]) -> Dict[int, Dict[str, Any]]:
        if not ids:
            return {}
        q = text(
            """
            SELECT
              a.id AS audit_id,
              a.event_id,
              a.is_live,
              a.audited_at,
              a.betslip_limit,
              a.hypothesis_details,
              m.kickoff_time
            FROM betslip_audit_results a
            LEFT JOIN matches m ON m.external_id = a.event_id
            WHERE a.id = ANY(:ids)
            """
        )
        out: Dict[int, Dict[str, Any]] = {}
        # chunk para evitar arrays gigantes no driver
        CH = 5000
        # normaliza url asyncpg
        url = database_url
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        eng = create_async_engine(url, echo=False, pool_pre_ping=True)
        try:
            async with eng.begin() as conn:
                for i in range(0, len(ids), CH):
                    part = ids[i : i + CH]
                    res = await conn.execute(q, {"ids": list(part)})
                    for row in res.fetchall() or []:
                        mp = dict(row._mapping)
                        try:
                            out[int(mp.get("audit_id"))] = mp
                        except Exception:
                            continue
        finally:
            try:
                await eng.dispose()
            except Exception:
                pass
        return out

    audit_by_id = await _fetch_audit_rows_for_ids(ids=audit_ids)
 
    pnl_by_oid = _acct_pnl_like_by_order_total_from_balance_csv(bal_csv)
 
    rows_base: List[dict] = []
    rows_sub: List[dict] = []
    missing_slip = 0
    missing_lat = 0
    missing_event = 0
    missing_limit = 0
    for oid, em in (exec_by_oid or {}).items():
        if not isinstance(em, dict):
            continue
        created = em.get("created_at")
        if not isinstance(created, datetime):
            continue
        day_exec = created.astimezone(timezone.utc).date().isoformat()
        if day_exec < start_day2:
            continue
        if end_day2 and day_exec > end_day2:
            continue
        pnl = pnl_by_oid.get(str(oid))
        if pnl is None:
            continue
        aid = em.get("audit_id")
        arow = None
        try:
            arow = audit_by_id.get(int(aid)) if aid is not None else None
        except Exception:
            arow = None
        is_in = bool(_is_inplay_from_audit_row(arow, exec_created_at_utc=created))
        # regime filter (Pre/In/All)
        sc = (str(regime or "") or "in").strip().lower()
        if sc not in ("in", "pre", "all"):
            sc = "in"
        if sc == "in" and (not bool(is_in)):
            continue
        if sc == "pre" and bool(is_in):
            continue
 
        event_id = None
        try:
            event_id = str((arow or {}).get("event_id") or "").strip() or None
        except Exception:
            event_id = None
        if not event_id:
            missing_event += 1
            # ainda dá para testar por order_id, mas sem cluster por jogo fica ruim
            continue
 
        lim = None
        try:
            lim = _safe_float((arow or {}).get("betslip_limit"))
        except Exception:
            lim = None
        if lim is None or float(lim) <= 0:
            missing_limit += 1
 
        lat = em.get("lat_ms")
        slip = em.get("slip_raw_pct")
        if _safe_float(lat) is None:
            missing_lat += 1
        if _safe_float(slip) is None:
            missing_slip += 1
 
        row = {
            "order_id": str(oid),
            "event_id": str(event_id),
            "created_at": created.astimezone(timezone.utc).isoformat(),
            "pnl": float(pnl),
            "exposure": _safe_float(em.get("exposure")) or 0.0,
            "lat_ms": _safe_float(lat),
            "slip_raw_pct": _safe_float(slip),
            "betslip_limit": (float(lim) if lim is not None else None),
        }
        rows_base.append(row)
 
        lab_lat = _bucket_call_to_done_ms(row.get("lat_ms"))
        lab_slip = _bucket_slip_raw_3way(row.get("slip_raw_pct"))
        if (lab_lat == str(lat_bucket)) and (lab_slip == str(slip_bucket)):
            rows_sub.append(row)
 
    summ_base = _summ(rows_base)
    summ_sub = _summ(rows_sub)
 
    # bootstrap por jogo: monta agregados por event_id
    by_game: Dict[str, Dict[str, float]] = {}
    for r in rows_base:
        eid = str(r.get("event_id") or "").strip()
        if not eid:
            continue
        rec = by_game.setdefault(eid, {"pnl_base": 0.0, "exp_base": 0.0, "pnl_sub": 0.0, "exp_sub": 0.0})
        rec["pnl_base"] += float(r.get("pnl") or 0.0)
        rec["exp_base"] += float(r.get("exposure") or 0.0)
    for r in rows_sub:
        eid = str(r.get("event_id") or "").strip()
        if not eid:
            continue
        rec = by_game.setdefault(eid, {"pnl_base": 0.0, "exp_base": 0.0, "pnl_sub": 0.0, "exp_sub": 0.0})
        rec["pnl_sub"] += float(r.get("pnl") or 0.0)
        rec["exp_sub"] += float(r.get("exposure") or 0.0)
 
    boot = _bootstrap_by_game(by_game=by_game, n_boot=int(n_boot), seed=int(seed))
 
    daily_rollup = {"base": _rollup_by_day(rows_base), "subset": _rollup_by_day(rows_sub)}

    def _stats(xs: List[float]) -> Dict[str, Any]:
        try:
            xs2 = [float(x) for x in (xs or []) if x is not None]
        except Exception:
            xs2 = []
        if not xs2:
            return {"n": 0, "mean": None, "p50": None, "p90": None, "p99": None, "min": None, "max": None}
        xs2s = sorted(xs2)
        try:
            mean = float(sum(xs2s) / float(len(xs2s)))
        except Exception:
            mean = None
        return {
            "n": int(len(xs2s)),
            "mean": mean,
            "p50": _quantile(xs2s, 0.50),
            "p90": _quantile(xs2s, 0.90),
            "p99": _quantile(xs2s, 0.99),
            "min": float(xs2s[0]),
            "max": float(xs2s[-1]),
        }

    # Estimativa de capacidade/turnover usando limit (betslip_limit) do auditor (máximo disponível no ticket)
    # Atenção: assume linearidade (pnl escala ~ linear com stake) e que "limit" é comparável à unidade de stake do executor.
    lim_factor = float(limit_stake_factor)
    lim_factor = max(0.0, lim_factor)
    lim_cap = float(limit_stake_cap)
    lim_cap = max(0.0, lim_cap)
    lims_all = [(_safe_float(r.get("betslip_limit")) if isinstance(r, dict) else None) for r in (rows_sub or [])]
    lims_pos = [float(x) for x in lims_all if x is not None and float(x) > 0.0]
    lim_mean_pos = None
    try:
        lim_mean_pos = float(sum(lims_pos) / float(len(lims_pos))) if lims_pos else None
    except Exception:
        lim_mean_pos = None

    def _stake_from_limit(limit_value: Optional[float], *, impute_mean: Optional[float]) -> Optional[float]:
        try:
            v = float(limit_value) if (limit_value is not None) else None
        except Exception:
            v = None
        if v is None or v <= 0:
            if impute_mean is None:
                return None
            v = float(impute_mean)
        st = max(0.0, v) * lim_factor
        if lim_cap > 0:
            st = min(st, lim_cap)
        return float(st)

    stake_targets_obs = []
    stake_targets_imp = []
    for x in lims_all:
        st_obs = _stake_from_limit(x, impute_mean=None)
        if st_obs is not None:
            stake_targets_obs.append(float(st_obs))
        st_imp = _stake_from_limit(x, impute_mean=lim_mean_pos)
        if st_imp is not None:
            stake_targets_imp.append(float(st_imp))

    # média por dia (dias com base >0 no período)
    days_base = [str(d.get("day") or "") for d in (daily_rollup.get("base") or []) if isinstance(d, dict) and str(d.get("day") or "")]
    n_days = int(len(days_base)) if days_base else 0
    sub_orders_total = int(len(rows_sub))
    avg_sub_orders_per_day = (float(sub_orders_total) / float(n_days)) if n_days > 0 else None

    # turnover alvo/dia (observado vs imputado)
    try:
        turnover_day_obs = (float(sum(stake_targets_obs)) / float(n_days)) if (n_days > 0 and stake_targets_obs) else None
    except Exception:
        turnover_day_obs = None
    try:
        turnover_day_imp = (float(sum(stake_targets_imp)) / float(n_days)) if (n_days > 0 and stake_targets_imp) else None
    except Exception:
        turnover_day_imp = None

    roi_sub = summ_sub.roi_w
    roi_ci = (boot or {}).get("roi_sub_ci90") if isinstance(boot, dict) else None
    roi_lb = _safe_float((roi_ci or {}).get("lb")) if isinstance(roi_ci, dict) else None
    roi_ub = _safe_float((roi_ci or {}).get("ub")) if isinstance(roi_ci, dict) else None

    def _profit(turnover: Optional[float], roi_pct: Optional[float]) -> Optional[float]:
        try:
            if turnover is None or roi_pct is None:
                return None
            return float(turnover) * float(roi_pct) / 100.0
        except Exception:
            return None

    capacity_estimate = {
        "note": "Usa betslip_limit (máximo no ticket) e aplica stake = factor*limit (com cap opcional). Assume linearidade.",
        "limit_factor": lim_factor,
        "limit_cap_abs": (lim_cap if lim_cap > 0 else None),
        "subset_days_in_range": n_days,
        "subset_orders_total": sub_orders_total,
        "subset_orders_per_day_avg": avg_sub_orders_per_day,
        "betslip_limit": {
            "n_total": int(len([x for x in lims_all if x is not None])),
            "n_pos": int(len(lims_pos)),
            "pos_pct": _pct(len(lims_pos), len([x for x in lims_all if x is not None])),
            "stats_pos": _stats(lims_pos),
        },
        "stake_target_from_limit": {
            "stats_observed": _stats(stake_targets_obs),
            "stats_imputed_missing_as_mean_pos": _stats(stake_targets_imp),
        },
        "turnover_target_per_day": {"observed": turnover_day_obs, "imputed": turnover_day_imp},
        "roi_subset_pct": roi_sub,
        "roi_subset_ci90": {"lb": roi_lb, "ub": roi_ub},
        "monthly_30d": {
            "orders": (float(avg_sub_orders_per_day) * 30.0 if avg_sub_orders_per_day is not None else None),
            "turnover": (float(turnover_day_obs) * 30.0 if turnover_day_obs is not None else None),
            "turnover_imputed": (float(turnover_day_imp) * 30.0 if turnover_day_imp is not None else None),
            "profit": _profit((float(turnover_day_obs) * 30.0 if turnover_day_obs is not None else None), roi_sub),
            "profit_ci90": {
                "lb": _profit((float(turnover_day_obs) * 30.0 if turnover_day_obs is not None else None), roi_lb),
                "ub": _profit((float(turnover_day_obs) * 30.0 if turnover_day_obs is not None else None), roi_ub),
            },
        },
        "monthly_22d": {
            "orders": (float(avg_sub_orders_per_day) * 22.0 if avg_sub_orders_per_day is not None else None),
            "turnover": (float(turnover_day_obs) * 22.0 if turnover_day_obs is not None else None),
            "turnover_imputed": (float(turnover_day_imp) * 22.0 if turnover_day_imp is not None else None),
            "profit": _profit((float(turnover_day_obs) * 22.0 if turnover_day_obs is not None else None), roi_sub),
            "profit_ci90": {
                "lb": _profit((float(turnover_day_obs) * 22.0 if turnover_day_obs is not None else None), roi_lb),
                "ub": _profit((float(turnover_day_obs) * 22.0 if turnover_day_obs is not None else None), roi_ub),
            },
        },
    }

    out = {
        "meta": {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "day_dir": str(day_dir),
            "executor_jsonl": str(executor_jsonl),
            "balance_csv": str(bal_csv),
            "start_day": start_day2,
            "end_day": end_day2,
            "regime": (sc if "sc" in locals() else str(regime or "in")),
            "subset": {"lat_bucket": str(lat_bucket), "slip_bucket": str(slip_bucket)},
            "capacity": {"limit_factor": lim_factor, "limit_cap_abs": (lim_cap if lim_cap > 0 else None)},
        },
        "daily_rollup": daily_rollup,
        "breakdowns": {
            "base_by_slip_bucket": _breakdown_by_bucket(rows_base, key="slip_bucket", order=["<= -2%", "(-2, 2]", "> 2%", "Desconhecido"]),
            "base_by_lat_bucket": _breakdown_by_bucket(rows_base, key="lat_bucket", order=["< 5s", "5-10s", "10-20s", "20-40s", "> 40s", "Desconhecido"]),
            "base_by_lat_x_slip": _breakdown_by_combo(rows_base, lat_key="lat_bucket", slip_key="slip_bucket"),
            "subset_by_slip_bucket": _breakdown_by_bucket(rows_sub, key="slip_bucket", order=["<= -2%", "(-2, 2]", "> 2%", "Desconhecido"]),
            "subset_by_lat_bucket": _breakdown_by_bucket(rows_sub, key="lat_bucket", order=["< 5s", "5-10s", "10-20s", "20-40s", "> 40s", "Desconhecido"]),
            "subset_by_lat_x_slip": _breakdown_by_combo(rows_sub, lat_key="lat_bucket", slip_key="slip_bucket"),
        },
        "capacity_estimate": capacity_estimate,
        "coverage": {
            "orders_base": summ_base.n_orders,
            "games_base": summ_base.n_games,
            "orders_subset": summ_sub.n_orders,
            "games_subset": summ_sub.n_games,
            "subset_pass_orders_pct": _pct(summ_sub.n_orders, summ_base.n_orders),
            "subset_pass_exposure_pct": _pct(summ_sub.exposure_sum, summ_base.exposure_sum),
            "missing_event_skipped": int(missing_event),
            "missing_lat_seen": int(missing_lat),
            "missing_slip_seen": int(missing_slip),
            "missing_betslip_limit_seen": int(missing_limit),
        },
        "base": {
            "pnl_sum": summ_base.pnl_sum,
            "exposure_sum": summ_base.exposure_sum,
            "roi_w_pct": summ_base.roi_w,
        },
        "subset": {
            "pnl_sum": summ_sub.pnl_sum,
            "exposure_sum": summ_sub.exposure_sum,
            "roi_w_pct": summ_sub.roi_w,
        },
        "delta": {
            "delta_pnl_sum": float(summ_sub.pnl_sum - summ_base.pnl_sum),
            "delta_roi_w_pct": (float(summ_sub.roi_w - summ_base.roi_w) if (summ_sub.roi_w is not None and summ_base.roi_w is not None) else None),
        },
        "bootstrap_by_game": boot,
    }
    return out
 
 
def _rollup_by_day(rows: List[dict]) -> List[Dict[str, Any]]:
    """
    Agrega ordens em stats por dia (created_at UTC) para evitar output gigante.
    """
    by: Dict[str, Dict[str, Any]] = {}
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        day = str(r.get("created_at") or "")[:10]
        if not _parse_day(day):
            continue
        rec = by.get(day)
        if rec is None:
            rec = {"day": day, "n_orders": 0, "pnl_sum": 0.0, "exposure_sum": 0.0, "_evs": set()}
            by[day] = rec
        rec["n_orders"] = int(rec.get("n_orders") or 0) + 1
        try:
            rec["pnl_sum"] = float(rec.get("pnl_sum") or 0.0) + float(r.get("pnl") or 0.0)
        except Exception:
            pass
        try:
            rec["exposure_sum"] = float(rec.get("exposure_sum") or 0.0) + float(r.get("exposure") or 0.0)
        except Exception:
            pass
        try:
            eid = str(r.get("event_id") or "").strip()
            if eid:
                evs = rec.get("_evs")
                # atenção: set() vazio é falsy; não use `or set()` aqui, senão perde os adds
                if not isinstance(evs, set):
                    evs = set()
                    rec["_evs"] = evs
                evs.add(eid)
        except Exception:
            pass
    out: List[Dict[str, Any]] = []
    for day in sorted(by.keys()):
        rec = by.get(day) or {}
        evs = rec.get("_evs") or set()
        pnl = float(rec.get("pnl_sum") or 0.0)
        exp = float(rec.get("exposure_sum") or 0.0)
        out.append(
            {
                "day": str(day),
                "n_orders": int(rec.get("n_orders") or 0),
                "n_games": int(len(evs)),
                "pnl_sum": float(pnl),
                "exposure_sum": float(exp),
                "roi_w_pct": _roi_weighted(pnl, exp),
            }
        )
    return out


def _walkforward_by_day(
    *,
    daily_base: List[dict],
    daily_sub: List[dict],
    start_day: str,
    end_day: Optional[str],
    train_days: int,
    min_games: int,
) -> Dict[str, Any]:
    """
    Walk-forward simples por dia (created_at UTC):
    - Para cada dia d, usa janela anterior [d-train_days, d-1] como "treino" (apenas referência; regra do subset já é fixa).
    - Avalia no dia d (OOS): ROIw(subset) vs ROIw(base), e acumula por dia.
    """
    base_by_day: Dict[str, Dict[str, Any]] = {}
    sub_by_day: Dict[str, Dict[str, Any]] = {}
    for r in daily_base or []:
        if not isinstance(r, dict):
            continue
        d = str(r.get("day") or "")[:10]
        if _parse_day(d):
            base_by_day[d] = r
    for r in daily_sub or []:
        if not isinstance(r, dict):
            continue
        d = str(r.get("day") or "")[:10]
        if _parse_day(d):
            sub_by_day[d] = r

    days = sorted({d for d in base_by_day.keys() if d >= start_day and (not end_day or d <= end_day)})
    out_rows: List[Dict[str, Any]] = []
    n_pos = 0
    n_eval = 0

    # prefix sums (para o "treino" como referência)
    def _f(x: Any) -> float:
        try:
            return float(x)
        except Exception:
            return 0.0

    base_pnl = [_f((base_by_day.get(d) or {}).get("pnl_sum")) for d in days]
    base_exp = [_f((base_by_day.get(d) or {}).get("exposure_sum")) for d in days]
    sub_pnl = [_f((sub_by_day.get(d) or {}).get("pnl_sum")) for d in days]
    sub_exp = [_f((sub_by_day.get(d) or {}).get("exposure_sum")) for d in days]
    p_base_pnl = [0.0]
    p_base_exp = [0.0]
    p_sub_pnl = [0.0]
    p_sub_exp = [0.0]
    for i in range(len(days)):
        p_base_pnl.append(p_base_pnl[-1] + base_pnl[i])
        p_base_exp.append(p_base_exp[-1] + base_exp[i])
        p_sub_pnl.append(p_sub_pnl[-1] + sub_pnl[i])
        p_sub_exp.append(p_sub_exp[-1] + sub_exp[i])

    td = int(max(1, train_days))
    mg = int(max(0, min_games))
    for idx, d in enumerate(days):
        rec_b = base_by_day.get(d) or {}
        rec_s = sub_by_day.get(d) or {}
        n_base_games = int(rec_b.get("n_games") or 0)
        skipped_reason = None
        if idx < td:
            skipped_reason = "burn_in"
        elif mg and n_base_games < mg:
            skipped_reason = "min_games"

        rb = rec_b.get("roi_w_pct")
        rs = rec_s.get("roi_w_pct") if rec_s else None
        try:
            rb = float(rb) if rb is not None else None
        except Exception:
            rb = None
        try:
            rs = float(rs) if rs is not None else None
        except Exception:
            rs = None
        delta = (float(rs - rb) if (rb is not None and rs is not None) else None)

        # treino (janela anterior)
        i0 = max(0, idx - td)
        i1 = idx
        tr_base_pnl = float(p_base_pnl[i1] - p_base_pnl[i0])
        tr_base_exp = float(p_base_exp[i1] - p_base_exp[i0])
        tr_sub_pnl = float(p_sub_pnl[i1] - p_sub_pnl[i0])
        tr_sub_exp = float(p_sub_exp[i1] - p_sub_exp[i0])
        tr_rb = _roi_weighted(tr_base_pnl, tr_base_exp)
        tr_rs = _roi_weighted(tr_sub_pnl, tr_sub_exp)
        tr_delta = (float(tr_rs - tr_rb) if (tr_rb is not None and tr_rs is not None) else None)

        evaluated = bool(skipped_reason is None and delta is not None)
        if evaluated:
            n_eval += 1
            if float(delta) > 0:
                n_pos += 1
        out_rows.append(
            {
                "day": d,
                "evaluated": evaluated,
                "skipped_reason": skipped_reason,
                "n_base_orders": int(rec_b.get("n_orders") or 0),
                "n_sub_orders": int(rec_s.get("n_orders") or 0),
                "n_base_games": n_base_games,
                "n_sub_games": int(rec_s.get("n_games") or 0),
                "roi_w_base_pct": rb,
                "roi_w_sub_pct": rs,
                "delta_roi_w_pct": delta,
                "train_window_days": int(td),
                "roi_w_train_base_pct": tr_rb,
                "roi_w_train_sub_pct": tr_rs,
                "delta_roi_w_train_pct": tr_delta,
            }
        )
    return {
        "train_days": int(train_days),
        "min_games": int(min_games),
        "days_considered": int(len(days)),
        "days_evaluated": int(n_eval),
        "days_delta_pos": int(n_pos),
        "sign_test_p": _sign_test_p(n_pos, n_eval),
        "per_day": out_rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Teste estatístico (ledger) de hipótese Back (Pre/In): subset por latência (call_to_done_ms) × slippage_raw_pct, com bootstrap por jogo."
    )
    ap.add_argument("--day-dir", required=True, help="Pasta do dia (ex.: .../20260413) gerada pelo daily_full_report.")
    ap.add_argument("--start-day", default="2026-04-04", help="Filtra por created_at UTC (YYYY-MM-DD). Default=2026-04-04.")
    ap.add_argument("--end-day", default="", help="Opcional: fim do filtro por created_at UTC (YYYY-MM-DD).")
    ap.add_argument("--regime", default="in", help="Regime a analisar: in|pre|all. Default=in.")
    ap.add_argument(
        "--database-url",
        default="",
        help="Override do DATABASE_URL (se não estiver no ENV_FILE). Ex.: postgresql://user:pass@host:5432/db",
    )
    ap.add_argument(
        "--balance-csv",
        default="",
        help="Override do caminho do balance.csv (se o path do accounting_daily_report.json estiver relativo a outro cwd).",
    )
    ap.add_argument("--lat-bucket", default="< 5s", help="Bucket de latência alvo. Default='< 5s'.")
    # argparse interpola '%' na help string; evite aspas e/ou escape.
    ap.add_argument("--slip-bucket", default="> 2%", help="Bucket de slippage alvo. Default=> 2%%.")
    ap.add_argument("--n-boot", type=int, default=2000, help="Bootstrap por jogo (recom.: 2000+).")
    ap.add_argument("--seed", type=int, default=1337, help="Seed do bootstrap.")
    ap.add_argument(
        "--limit-stake-factor",
        type=float,
        default=0.5,
        help="Para estimar capacidade: stake_alvo = factor * betslip_limit (máximo no ticket). Default=0.5.",
    )
    ap.add_argument(
        "--limit-stake-cap",
        type=float,
        default=0.0,
        help="Cap absoluto opcional do stake_alvo derivado do limit (0=sem cap).",
    )
    ap.add_argument(
        "--walkforward-by-day",
        action="store_true",
        help="Além do bootstrap por jogo, roda um walk-forward simples por dia (OOS por dia) e reporta sign-test do delta.",
    )
    ap.add_argument("--wf-train-days", type=int, default=3, help="Dias de treino (histórico mínimo) para começar o walk-forward. Default=3.")
    ap.add_argument("--wf-min-games", type=int, default=0, help="Mínimo de jogos (event_id) no dia para avaliar no walk-forward. Default=0 (off).")
    args = ap.parse_args()
 
    # normaliza end-day
    end_day = str(args.end_day or "").strip() or None
 
    import asyncio
 
    out = asyncio.run(
        _run(
            day_dir=Path(str(args.day_dir)),
            start_day=str(args.start_day),
            end_day=end_day,
            regime=str(getattr(args, "regime", "in")),
            lat_bucket=str(args.lat_bucket),
            slip_bucket=str(args.slip_bucket),
            database_url_override=(str(args.database_url).strip() or None),
            balance_csv_override=(str(args.balance_csv).strip() or None),
            n_boot=int(args.n_boot),
            seed=int(args.seed),
            limit_stake_factor=float(getattr(args, "limit_stake_factor", 0.5)),
            limit_stake_cap=float(getattr(args, "limit_stake_cap", 0.0)),
        )
    )
    if bool(getattr(args, "walkforward_by_day", False)):
        try:
            daily = out.get("daily_rollup") or {}
            daily_base = list(daily.get("base") or [])
            daily_sub = list(daily.get("subset") or [])
            wf = _walkforward_by_day(
                daily_base=daily_base,
                daily_sub=daily_sub,
                start_day=str(out.get("meta", {}).get("start_day") or str(args.start_day)),
                end_day=(str(out.get("meta", {}).get("end_day")) if out.get("meta", {}).get("end_day") else None),
                train_days=int(getattr(args, "wf_train_days", 3)),
                min_games=int(getattr(args, "wf_min_games", 0)),
            )
            out["walkforward_by_day"] = wf
        except Exception as e:
            out["walkforward_by_day_error"] = str(e)[:200]
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())

