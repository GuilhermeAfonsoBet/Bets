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
    from storage.database import Database  # type: ignore
    from .daily_full_report import (  # type: ignore
        _acct_pnl_like_by_order_total_from_balance_csv,
        _extract_audit_ids_from_exec_by_oid,
        _fetch_audit_rows_for_ids_daily,
        _is_inplay_from_audit_row,
        _load_env_file,
        _parse_executor_jsonl_back_live_orders,
    )
 
    _load_env_file(Path(os.getenv("ENV_FILE", ".env")))
 
    acct_json = day_dir / "accounting_daily_report.json"
    if not acct_json.exists():
        raise SystemExit(f"Não achei `{acct_json}`. Passe `--day-dir` correto (o day_dir impresso pelo daily).")
    acct = json.loads(acct_json.read_text(encoding="utf-8", errors="ignore") or "{}")
    bal_csv = Path(str(acct.get("balance_csv") or "")).expanduser()
    if not bal_csv.exists():
        raise SystemExit(f"Não achei balance.csv em `{bal_csv}` (de `{acct_json}`).")
 
    executor_jsonl = Path(os.getenv("EXECUTOR_JSONL", "logs/executor_live.jsonl"))
    if not executor_jsonl.is_absolute():
        executor_jsonl = (day_dir.parent.parent / executor_jsonl).resolve()  # tenta relativo ao repo
    if not executor_jsonl.exists():
        # fallback: relativo ao repo atual
        executor_jsonl = Path(os.getenv("EXECUTOR_JSONL", "logs/executor_live.jsonl"))
    if not executor_jsonl.exists():
        raise SystemExit(f"Não achei executor_jsonl em `{executor_jsonl}`. Sete `EXECUTOR_JSONL`.")
 
    start_day2 = _parse_day(start_day)
    end_day2 = _parse_day(end_day) if end_day else None
    if not start_day2:
        raise SystemExit("`--start-day` inválido. Use YYYY-MM-DD.")
    if end_day and not end_day2:
        raise SystemExit("`--end-day` inválido. Use YYYY-MM-DD.")
 
    exec_by_oid = _parse_executor_jsonl_back_live_orders(executor_jsonl)
    audit_ids = _extract_audit_ids_from_exec_by_oid(exec_by_oid)
 
    db = Database()
    await db.connect()
    try:
        audit_by_id = await _fetch_audit_rows_for_ids_daily(db, audit_ids)
    finally:
        await db.close()
 
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

