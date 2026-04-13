import argparse
import json
import os
import random
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
 
 
async def _run(
    *,
    day_dir: Path,
    start_day: str,
    end_day: Optional[str],
    lat_bucket: str,
    slip_bucket: str,
    n_boot: int,
    seed: int,
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
 
    database_url = str(os.getenv("DATABASE_URL", "") or "").strip()
    if not database_url:
        raise SystemExit("DATABASE_URL não está definido (no ENV_FILE/.env).")

    acct_json = day_dir / "accounting_daily_report.json"
    if not acct_json.exists():
        raise SystemExit(f"Não achei `{acct_json}`. Passe `--day-dir` correto (o day_dir impresso pelo daily).")
    acct = json.loads(acct_json.read_text(encoding="utf-8", errors="ignore") or "{}")
    bal_csv = Path(str(acct.get("balance_csv") or "")).expanduser()
    if not bal_csv.exists():
        raise SystemExit(f"Não achei balance.csv em `{bal_csv}` (de `{acct_json}`).")
 
    executor_jsonl = Path(os.getenv("EXECUTOR_JSONL", "logs/executor_live.jsonl"))
    if not executor_jsonl.is_absolute():
        # Tenta resolver relativo ao root do repo (…/Bets) e ao root do package (…/Bets/betinasia_bot).
        try:
            repo_root = Path(__file__).resolve().parents[2]
        except Exception:
            repo_root = Path.cwd()
        bot_root = Path(__file__).resolve().parents[1]  # .../betinasia_bot
        cand = [(repo_root / executor_jsonl).resolve(), (bot_root / executor_jsonl).resolve(), (Path.cwd() / executor_jsonl).resolve()]
        picked = None
        for p in cand:
            if p.exists():
                picked = p
                break
        executor_jsonl = picked or cand[0]
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
        if not is_in:
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
 
    out = {
        "meta": {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "day_dir": str(day_dir),
            "executor_jsonl": str(executor_jsonl),
            "balance_csv": str(bal_csv),
            "start_day": start_day2,
            "end_day": end_day2,
            "subset": {"lat_bucket": str(lat_bucket), "slip_bucket": str(slip_bucket)},
        },
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
 
 
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Teste estatístico (ledger) de hipótese Back In: subset por latência (call_to_done_ms) × slippage_raw_pct, com bootstrap por jogo."
    )
    ap.add_argument("--day-dir", required=True, help="Pasta do dia (ex.: .../20260413) gerada pelo daily_full_report.")
    ap.add_argument("--start-day", default="2026-04-04", help="Filtra por created_at UTC (YYYY-MM-DD). Default=2026-04-04.")
    ap.add_argument("--end-day", default="", help="Opcional: fim do filtro por created_at UTC (YYYY-MM-DD).")
    ap.add_argument("--lat-bucket", default="< 5s", help="Bucket de latência alvo. Default='< 5s'.")
    # argparse interpola '%' na help string; evite aspas e/ou escape.
    ap.add_argument("--slip-bucket", default="> 2%", help="Bucket de slippage alvo. Default=> 2%%.")
    ap.add_argument("--n-boot", type=int, default=2000, help="Bootstrap por jogo (recom.: 2000+).")
    ap.add_argument("--seed", type=int, default=1337, help="Seed do bootstrap.")
    args = ap.parse_args()
 
    # normaliza end-day
    end_day = str(args.end_day or "").strip() or None
 
    import asyncio
 
    out = asyncio.run(
        _run(
            day_dir=Path(str(args.day_dir)),
            start_day=str(args.start_day),
            end_day=end_day,
            lat_bucket=str(args.lat_bucket),
            slip_bucket=str(args.slip_bucket),
            n_boot=int(args.n_boot),
            seed=int(args.seed),
        )
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())

