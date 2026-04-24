from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

@dataclass
class ExecRecord:
    order_id: Optional[str]
    created_at: datetime
    audit_id: int
    event_id: Optional[str]
    line: Optional[str]
    side: Optional[str]
    odd_decision: Optional[float]
    odd_final: Optional[float]
    stake: float
    latency_ms: Optional[float]
    status: str


@dataclass
class BackObservation:
    created_at: datetime
    day_local: str
    event_id: str
    regime: str  # pre|in
    roi_pct: float
    pnl: float
    exposure: float
    latency_ms: Optional[float]
    slip_raw_pct: Optional[float]
    lat_bucket: str
    slip_bucket: str


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _parse_iso(s: Any) -> Optional[datetime]:
    try:
        t = str(s or "").strip()
        if not t:
            return None
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _sanitize_decimal_odd(odd: Optional[float]) -> Optional[float]:
    if odd is None:
        return None
    try:
        o = float(odd)
    except Exception:
        return None
    if o <= 1.0:
        return None
    if 100.0 <= o <= 3000.0:
        o2 = o / 100.0
        if 1.0 < o2 <= 30.0:
            return float(o2)
    if o > 30.0:
        return None
    return float(o)


def _repo_roots() -> List[Path]:
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


def _resolve_rel_path(p: Path) -> Path:
    try:
        if p.is_absolute():
            return p
    except Exception:
        return p
    for root in _repo_roots():
        cand = (root / p).resolve()
        if cand.exists():
            return cand
    try:
        return (_repo_roots()[0] / p).resolve()
    except Exception:
        return p


def _weighted_roi_pct(rows: Sequence[BackObservation]) -> Optional[float]:
    exp = float(sum(float(r.exposure) for r in rows))
    if exp <= 0:
        return None
    pnl = float(sum(float(r.pnl) for r in rows))
    return float(pnl / exp * 100.0)


def _summarize(rows: Sequence[BackObservation]) -> Dict[str, Any]:
    events = {r.event_id for r in rows if r.event_id}
    exp = float(sum(float(r.exposure) for r in rows))
    pnl = float(sum(float(r.pnl) for r in rows))
    return {
        "n_orders": int(len(rows)),
        "n_games": int(len(events)),
        "exposure_sum": exp,
        "pnl_sum": pnl,
        "roi_weighted_pct": (_weighted_roi_pct(rows)),
    }


def _bucket_latency(lat_ms: Optional[float]) -> str:
    if lat_ms is None:
        return "unknown"
    x = float(lat_ms)
    if x < 3000:
        return "<3s"
    if x < 6000:
        return "3-6s"
    if x < 10000:
        return "6-10s"
    if x < 20000:
        return "10-20s"
    return ">=20s"


def _bucket_slippage(slip_raw_pct: Optional[float]) -> str:
    if slip_raw_pct is None:
        return "unknown"
    x = float(slip_raw_pct)
    if x <= -2.0:
        return "<=-2%"
    if x <= 0.0:
        return "(-2,0]"
    if x <= 2.0:
        return "(0,2]"
    return ">2%"


def _breakdown_by_key(rows: Sequence[BackObservation], key: str) -> List[Dict[str, Any]]:
    groups: Dict[str, List[BackObservation]] = {}
    for r in rows:
        k = str(getattr(r, key))
        groups.setdefault(k, []).append(r)
    out: List[Dict[str, Any]] = []
    for k in sorted(groups.keys()):
        g = groups[k]
        s = _summarize(g)
        s["bucket"] = k
        out.append(s)
    return out


def _breakdown_by_combo(rows: Sequence[BackObservation]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[BackObservation]] = {}
    for r in rows:
        k = (str(r.lat_bucket), str(r.slip_bucket))
        groups.setdefault(k, []).append(r)
    out: List[Dict[str, Any]] = []
    for (lb, sb) in sorted(groups.keys(), key=lambda x: (x[0], x[1])):
        g = groups[(lb, sb)]
        s = _summarize(g)
        s["lat_bucket"] = lb
        s["slip_bucket"] = sb
        out.append(s)
    return out


def _binom_tail_prob(n: int, k: int) -> float:
    p = 0.0
    for i in range(k, n + 1):
        p += math.comb(n, i) * (0.5**n)
    return float(p)


def _sign_test_two_sided_p(n_pos: int, n_total: int) -> Optional[float]:
    if n_total <= 0:
        return None
    k = min(max(0, int(n_pos)), int(n_total))
    tail = min(_binom_tail_prob(n_total, k), _binom_tail_prob(n_total, n_total - k))
    return float(min(1.0, 2.0 * tail))


def _gap_stat(labels: Sequence[str], pnl: Sequence[float], exposure: Sequence[float], min_obs: int) -> Optional[float]:
    sums_pnl: Dict[str, float] = {}
    sums_exp: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for i, lb in enumerate(labels):
        sums_pnl[lb] = float(sums_pnl.get(lb, 0.0)) + float(pnl[i])
        sums_exp[lb] = float(sums_exp.get(lb, 0.0)) + float(exposure[i])
        counts[lb] = int(counts.get(lb, 0)) + 1
    rois: List[float] = []
    for lb in counts.keys():
        if int(counts[lb]) < int(min_obs):
            continue
        exp = float(sums_exp.get(lb, 0.0))
        if exp <= 0:
            continue
        rois.append(float(sums_pnl.get(lb, 0.0) / exp * 100.0))
    if len(rois) < 2:
        return None
    return float(max(rois) - min(rois))


def _permutation_test_gap(
    rows: Sequence[BackObservation],
    *,
    label_mode: str,
    min_obs: int,
    n_perm: int,
    seed: int,
    include_unknown: bool,
) -> Dict[str, Any]:
    labels: List[str] = []
    pnl: List[float] = []
    exposure: List[float] = []

    for r in rows:
        if label_mode == "lat":
            lb = str(r.lat_bucket)
            if not include_unknown and lb == "unknown":
                continue
        elif label_mode == "slip":
            lb = str(r.slip_bucket)
            if not include_unknown and lb == "unknown":
                continue
        elif label_mode == "combo":
            lb = f"{r.lat_bucket}__{r.slip_bucket}"
            if not include_unknown and ("unknown" in lb):
                continue
        else:
            raise ValueError(f"label_mode inválido: {label_mode}")
        labels.append(lb)
        pnl.append(float(r.pnl))
        exposure.append(float(r.exposure))

    observed = _gap_stat(labels, pnl, exposure, min_obs=int(min_obs))
    if observed is None:
        return {
            "label_mode": label_mode,
            "n_rows": int(len(labels)),
            "n_perm": int(n_perm),
            "observed_gap_pct": None,
            "p_value": None,
            "note": "sem buckets suficientes após filtros",
        }

    rng = random.Random(int(seed))
    shuffled = list(pnl)
    ge = 0
    valid_perm = 0
    for _ in range(int(max(1, n_perm))):
        rng.shuffle(shuffled)
        st = _gap_stat(labels, shuffled, exposure, int(min_obs))
        if st is None:
            continue
        valid_perm += 1
        if float(st) >= float(observed):
            ge += 1

    p_value = None
    if valid_perm > 0:
        p_value = float((ge + 1) / (valid_perm + 1))

    return {
        "label_mode": label_mode,
        "n_rows": int(len(labels)),
        "n_perm": int(n_perm),
        "valid_permutations": int(valid_perm),
        "observed_gap_pct": float(observed),
        "p_value": p_value,
    }


def _expanding_oos_combo(
    rows: Sequence[BackObservation],
    *,
    min_train_days: int,
    min_train_obs_per_combo: int,
    min_test_obs: int,
    include_unknown: bool,
) -> Dict[str, Any]:
    by_day: Dict[str, List[BackObservation]] = {}
    for r in rows:
        by_day.setdefault(r.day_local, []).append(r)
    days = sorted(by_day.keys())

    folds: List[Dict[str, Any]] = []
    deltas: List[float] = []
    for i, day in enumerate(days):
        if i < int(min_train_days):
            continue
        train_days = days[:i]
        test_rows = by_day.get(day, [])
        train_rows: List[BackObservation] = []
        for d in train_days:
            train_rows.extend(by_day.get(d, []))

        combo_train: Dict[Tuple[str, str], List[BackObservation]] = {}
        for r in train_rows:
            if (not include_unknown) and (r.lat_bucket == "unknown" or r.slip_bucket == "unknown"):
                continue
            combo_train.setdefault((r.lat_bucket, r.slip_bucket), []).append(r)

        best_combo = None
        best_roi = None
        best_n = 0
        for combo, rr in combo_train.items():
            if len(rr) < int(min_train_obs_per_combo):
                continue
            roi = _weighted_roi_pct(rr)
            if roi is None:
                continue
            if (best_roi is None) or (float(roi) > float(best_roi)) or (
                float(roi) == float(best_roi) and len(rr) > int(best_n)
            ):
                best_roi = float(roi)
                best_combo = combo
                best_n = int(len(rr))

        if best_combo is None:
            folds.append({"day": day, "evaluated": False, "skip_reason": "no_train_combo"})
            continue

        base_roi = _weighted_roi_pct(test_rows)
        sub_rows = [r for r in test_rows if (r.lat_bucket, r.slip_bucket) == best_combo]
        if len(sub_rows) < int(min_test_obs):
            folds.append(
                {
                    "day": day,
                    "evaluated": False,
                    "skip_reason": "insufficient_test_obs",
                    "best_combo": {"lat_bucket": best_combo[0], "slip_bucket": best_combo[1]},
                    "test_obs": int(len(sub_rows)),
                }
            )
            continue

        sub_roi = _weighted_roi_pct(sub_rows)
        delta = None
        if base_roi is not None and sub_roi is not None:
            delta = float(sub_roi - base_roi)
            deltas.append(float(delta))
        folds.append(
            {
                "day": day,
                "evaluated": bool(delta is not None),
                "best_combo": {"lat_bucket": best_combo[0], "slip_bucket": best_combo[1]},
                "train_obs_combo": int(best_n),
                "train_roi_combo_pct": float(best_roi) if best_roi is not None else None,
                "test_obs_base": int(len(test_rows)),
                "test_obs_combo": int(len(sub_rows)),
                "test_roi_base_pct": base_roi,
                "test_roi_combo_pct": sub_roi,
                "delta_roi_pct": delta,
            }
        )

    eval_deltas = [float(x) for x in deltas if x is not None]
    n_eval = int(len(eval_deltas))
    n_pos = int(sum(1 for x in eval_deltas if float(x) > 0.0))
    mean_delta = (float(sum(eval_deltas) / n_eval) if n_eval > 0 else None)
    median_delta = None
    if n_eval > 0:
        y = sorted(eval_deltas)
        mid = n_eval // 2
        median_delta = float(y[mid]) if (n_eval % 2 == 1) else float((y[mid - 1] + y[mid]) / 2.0)

    return {
        "days_total": int(len(days)),
        "days_evaluated": n_eval,
        "days_delta_positive": n_pos,
        "mean_delta_roi_pct": mean_delta,
        "median_delta_roi_pct": median_delta,
        "sign_test_p_value": _sign_test_two_sided_p(n_pos, n_eval),
        "folds": folds,
    }


def _to_json_compatible(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _to_json_compatible(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_compatible(v) for v in obj]
    return obj


def _mult_back_from_scores(line: Any, side: str, hs: Any, aws: Any) -> Optional[float]:
    if hs is None or aws is None:
        return None
    try:
        goal_diff = int(hs) - int(aws)
    except Exception:
        return None
    try:
        sel = (side or "").strip().lower()
        raw = str(line).strip().replace(",", ".").replace("−", "-")
        ah = float(raw)
        home_handicap = ah if (raw.startswith("+") or raw.startswith("-")) else (ah if sel == "home" else -ah)
    except Exception:
        return None
    if sel == "home":
        adjusted = goal_diff + home_handicap
    elif sel == "away":
        adjusted = -goal_diff - home_handicap
    else:
        return None
    if adjusted > 0.25:
        return 1.0
    if adjusted == 0.25:
        return 0.5
    if adjusted == 0:
        return 0.0
    if adjusted == -0.25:
        return -0.5
    return -1.0


def _roi_back_pct(odd: float, mult: float) -> float:
    if mult > 0:
        return (float(odd) - 1.0) * float(mult) * 100.0
    if mult < 0:
        return float(mult) * 100.0
    return 0.0


def _slip_raw_pct(odd_dec: Optional[float], odd_fin: Optional[float]) -> Optional[float]:
    if odd_dec is None or odd_fin is None:
        return None
    if float(odd_dec) <= 0:
        return None
    return float((float(odd_fin) - float(odd_dec)) / float(odd_dec) * 100.0)


def _infer_regime(created_at: datetime, audit_row: Dict[str, Any]) -> str:
    ko = audit_row.get("kickoff_time")
    if isinstance(ko, datetime):
        return "in" if created_at >= ko.astimezone(timezone.utc) else "pre"
    if isinstance(ko, str):
        dt = _parse_iso(ko)
        if dt is not None:
            return "in" if created_at >= dt else "pre"
    if audit_row.get("is_live") is not None:
        return "in" if bool(audit_row.get("is_live")) else "pre"
    return "pre"


def _pick_col(cols: Sequence[str], keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        kl = str(k).lower().strip()
        for c in cols:
            cl = str(c).lower().strip()
            if cl == kl or cl.startswith(kl) or kl in cl:
                return c
    return None


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
    try:
        payload = json.dumps(raw, ensure_ascii=False)
        m = re.search(r"\"order[_ ]?id\"\\s*:\\s*\"?(\\d+)\"?", payload, flags=re.IGNORECASE)
        if m:
            return str(m.group(1))
    except Exception:
        pass
    return None


def _resolve_balance_csv(args: argparse.Namespace) -> Path:
    if str(args.balance_csv or "").strip():
        p = _resolve_rel_path(Path(str(args.balance_csv)).expanduser())
        if not p.exists():
            raise SystemExit(f"--balance-csv não encontrado: {p}")
        return p

    acct_report = _resolve_rel_path(Path(str(args.accounting_report_json)).expanduser())
    if not acct_report.exists():
        raise SystemExit(
            f"Arquivo de accounting não encontrado: {acct_report}. "
            "Passe --balance-csv explicitamente ou ajuste --accounting-report-json."
        )
    try:
        obj = json.loads(acct_report.read_text(encoding="utf-8", errors="ignore") or "{}")
    except Exception as e:
        raise SystemExit(f"Falha lendo accounting report ({acct_report}): {e}")
    bal_raw = str((obj or {}).get("balance_csv") or "").strip()
    if not bal_raw:
        raise SystemExit(
            f"`balance_csv` ausente em {acct_report}. "
            "Passe --balance-csv explicitamente."
        )
    bal = _resolve_rel_path(Path(bal_raw).expanduser())
    if not bal.exists():
        raise SystemExit(f"balance_csv não encontrado: {bal} (origem: {acct_report})")
    return bal


def _read_accounting_pnl_by_order(balance_csv: Path) -> Tuple[Dict[str, float], Dict[str, Any]]:
    by_oid: Dict[str, float] = {}
    n_rows = 0
    n_rows_used = 0
    n_ignored_type = 0
    with balance_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        if not cols:
            return {}, {"rows_total": 0, "rows_used": 0, "orders": 0}
        pnl_col = _pick_col(cols, ("amount", "profit_loss", "profit", "p&l", "pnl", "net", "pl"))
        oid_col = _pick_col(cols, ("order_id", "order id", "order", "bet id", "bet_id", "id"))
        typ_col = _pick_col(cols, ("type",))
        if not pnl_col or not oid_col:
            raise SystemExit(
                f"balance_csv sem colunas necessárias (order_id/pnl). arquivo={balance_csv}"
            )
        for row in reader:
            n_rows += 1
            if not isinstance(row, dict):
                continue
            oid = str(row.get(oid_col) or "").strip()
            if not oid or not oid.isdigit():
                continue
            pnl = _safe_float(row.get(pnl_col))
            if pnl is None:
                continue
            typ = str(row.get(typ_col) or "").strip().lower() if typ_col else ""
            # Quando existe tipo, restringimos a movimentos de aposta.
            if typ and ("bet" not in typ):
                n_ignored_type += 1
                continue
            by_oid[oid] = float(by_oid.get(oid) or 0.0) + float(pnl)
            n_rows_used += 1
    meta = {
        "rows_total": int(n_rows),
        "rows_used": int(n_rows_used),
        "rows_ignored_non_bet_type": int(n_ignored_type),
        "orders": int(len(by_oid)),
    }
    return by_oid, meta


def _load_executor_rows(jsonl_path: Path, allowed_statuses: Iterable[str]) -> List[ExecRecord]:
    st_allowed = {str(x).strip().upper() for x in allowed_statuses if str(x).strip()}
    out: List[ExecRecord] = []
    if not jsonl_path.exists():
        return out
    for ln in jsonl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        req = obj.get("request") if isinstance(obj.get("request"), dict) else {}
        res = obj.get("result") if isinstance(obj.get("result"), dict) else {}
        status = str(res.get("status") or "").strip().upper()
        if status not in st_allowed:
            continue
        exec_side = str(res.get("exec_side") or req.get("exec_side") or "").strip().lower()
        if exec_side != "back":
            continue
        created_at = _parse_iso(res.get("created_at") or req.get("created_at"))
        if created_at is None:
            continue
        aid = res.get("audit_id")
        if aid is None:
            aid = req.get("audit_id")
        try:
            audit_id = int(aid)
        except Exception:
            continue

        raw = res.get("raw") if isinstance(res.get("raw"), dict) else {}
        oid = _extract_order_id_from_raw(raw)
        sent = raw.get("sent") if isinstance(raw.get("sent"), dict) else {}
        stake = _safe_float(sent.get("stake"))
        if stake is None:
            pol = res.get("policy") if isinstance(res.get("policy"), dict) else (
                req.get("policy") if isinstance(req.get("policy"), dict) else {}
            )
            stake = _safe_float((pol or {}).get("stake_requested"))
        if stake is None or float(stake) <= 0:
            stake = 1.0

        timing = res.get("timing") if isinstance(res.get("timing"), dict) else {}
        lat = _safe_float(timing.get("call_to_done_ms"))

        out.append(
            ExecRecord(
                order_id=(str(oid) if oid else None),
                created_at=created_at,
                audit_id=audit_id,
                event_id=(str(res.get("event_id") or req.get("event_id") or "").strip() or None),
                line=(str(res.get("line") or req.get("line") or "").strip() or None),
                side=(str(res.get("side") or req.get("side") or "").strip() or None),
                odd_decision=_safe_float(res.get("odd_at_decision") if res.get("odd_at_decision") is not None else req.get("odd_at_decision")),
                odd_final=_safe_float(res.get("odd_final")),
                stake=float(stake),
                latency_ms=float(lat) if lat is not None else None,
                status=status,
            )
        )
    return out


async def _fetch_audit_rows(database_url: str, audit_ids: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    if not audit_ids:
        return {}
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    url = str(database_url).strip()
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    eng = create_async_engine(url, echo=False, pool_pre_ping=True)
    q = text(
        """
        SELECT
          a.id AS audit_id,
          a.event_id,
          a.line,
          a.side,
          a.is_live,
          a.audited_at,
          m.kickoff_time,
          m.home_score,
          m.away_score
        FROM betslip_audit_results a
        LEFT JOIN matches m ON m.external_id = a.event_id
        WHERE a.id = ANY(:ids)
        """
    )
    out: Dict[int, Dict[str, Any]] = {}
    chunk = 5000
    try:
        async with eng.begin() as conn:
            for i in range(0, len(audit_ids), chunk):
                part = list(audit_ids[i : i + chunk])
                res = await conn.execute(q, {"ids": part})
                for row in res.fetchall() or []:
                    d = dict(row._mapping)
                    try:
                        out[int(d["audit_id"])] = d
                    except Exception:
                        continue
    finally:
        await eng.dispose()
    return out


def _pick_database_url(cli_database_url: str) -> str:
    if str(cli_database_url or "").strip():
        return str(cli_database_url).strip()
    env_url = str(os.getenv("DATABASE_URL", "") or "").strip()
    if env_url:
        return env_url
    raise SystemExit("DATABASE_URL ausente. Use --database-url ou exporte DATABASE_URL.")


def _parse_day(s: str) -> Optional[str]:
    try:
        x = str(s or "").strip()
        if not x:
            return None
        datetime.strptime(x, "%Y-%m-%d")
        return x
    except Exception:
        return None


async def _run(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        from zoneinfo import ZoneInfo  # type: ignore

        tz = ZoneInfo(str(args.tz))
    except Exception:
        tz = timezone.utc

    database_url = _pick_database_url(str(args.database_url or ""))
    balance_csv = _resolve_balance_csv(args)
    pnl_by_order, accounting_meta = _read_accounting_pnl_by_order(balance_csv)
    allowed_statuses = [x.strip().upper() for x in str(args.statuses).split(",") if x.strip()]
    exec_rows = _load_executor_rows(Path(str(args.executor_jsonl)), allowed_statuses=allowed_statuses)

    audit_ids = sorted({int(r.audit_id) for r in exec_rows})
    audit_map = await _fetch_audit_rows(database_url, audit_ids)

    start_day = _parse_day(str(args.start_day or ""))
    end_day = _parse_day(str(args.end_day or ""))

    obs: List[BackObservation] = []
    dropped = {
        "missing_order_id": 0,
        "missing_accounting_pnl": 0,
        "missing_audit": 0,
        "missing_event_id": 0,
        "missing_exposure": 0,
        "outside_day_range": 0,
    }

    for ex in exec_rows:
        if not ex.order_id:
            dropped["missing_order_id"] += 1
            continue
        pnl = pnl_by_order.get(str(ex.order_id))
        if pnl is None:
            dropped["missing_accounting_pnl"] += 1
            continue
        ar = audit_map.get(int(ex.audit_id))
        if not ar:
            dropped["missing_audit"] += 1
            continue
        event_id = str(ar.get("event_id") or ex.event_id or "").strip()
        if not event_id:
            dropped["missing_event_id"] += 1
            continue
        day_local = ex.created_at.astimezone(tz).date().isoformat()
        if start_day and day_local < start_day:
            dropped["outside_day_range"] += 1
            continue
        if end_day and day_local > end_day:
            dropped["outside_day_range"] += 1
            continue

        exposure = _safe_float(ex.stake)
        if exposure is None or float(exposure) <= 0:
            dropped["missing_exposure"] += 1
            continue

        regime = _infer_regime(ex.created_at, ar)
        roi = float(float(pnl) / float(exposure) * 100.0)
        slip = _slip_raw_pct(_sanitize_decimal_odd(ex.odd_decision), _sanitize_decimal_odd(ex.odd_final))
        lat_bucket = _bucket_latency(ex.latency_ms)
        slip_bucket = _bucket_slippage(slip)
        obs.append(
            BackObservation(
                created_at=ex.created_at,
                day_local=day_local,
                event_id=event_id,
                regime=regime,
                roi_pct=float(roi),
                pnl=float(pnl),
                exposure=float(exposure),
                latency_ms=(float(ex.latency_ms) if ex.latency_ms is not None else None),
                slip_raw_pct=(float(slip) if slip is not None else None),
                lat_bucket=lat_bucket,
                slip_bucket=slip_bucket,
            )
        )

    by_regime = {
        "all": list(obs),
        "pre": [r for r in obs if r.regime == "pre"],
        "in": [r for r in obs if r.regime == "in"],
    }

    summaries: Dict[str, Any] = {}
    perm_results: Dict[str, Any] = {}
    oos_results: Dict[str, Any] = {}
    for regime_key, rows in by_regime.items():
        summaries[regime_key] = {
            "base": _summarize(rows),
            "lat_buckets": _breakdown_by_key(rows, "lat_bucket"),
            "slip_buckets": _breakdown_by_key(rows, "slip_bucket"),
            "combo_buckets": _breakdown_by_combo(rows),
        }
        perm_results[regime_key] = {
            "lat_bucket": _permutation_test_gap(
                rows,
                label_mode="lat",
                min_obs=int(args.min_bucket_obs),
                n_perm=int(args.perm_n),
                seed=int(args.perm_seed),
                include_unknown=bool(args.include_unknown_buckets),
            ),
            "slip_bucket": _permutation_test_gap(
                rows,
                label_mode="slip",
                min_obs=int(args.min_bucket_obs),
                n_perm=int(args.perm_n),
                seed=int(args.perm_seed) + 101,
                include_unknown=bool(args.include_unknown_buckets),
            ),
            "lat_x_slip": _permutation_test_gap(
                rows,
                label_mode="combo",
                min_obs=int(args.min_bucket_obs),
                n_perm=int(args.perm_n),
                seed=int(args.perm_seed) + 202,
                include_unknown=bool(args.include_unknown_buckets),
            ),
        }
        oos_results[regime_key] = _expanding_oos_combo(
            rows,
            min_train_days=int(args.oos_min_train_days),
            min_train_obs_per_combo=int(args.oos_min_train_obs),
            min_test_obs=int(args.oos_min_test_obs),
            include_unknown=bool(args.include_unknown_buckets),
        )

    out: Dict[str, Any] = {
        "meta": {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "executor_jsonl": str(args.executor_jsonl),
            "balance_csv": str(balance_csv),
            "pnl_source": "accounting_real_by_order_id",
            "tz": str(args.tz),
            "statuses": allowed_statuses,
            "start_day": start_day,
            "end_day": end_day,
            "perm_n": int(args.perm_n),
            "perm_seed": int(args.perm_seed),
            "min_bucket_obs": int(args.min_bucket_obs),
            "oos_min_train_days": int(args.oos_min_train_days),
            "oos_min_train_obs": int(args.oos_min_train_obs),
            "oos_min_test_obs": int(args.oos_min_test_obs),
            "include_unknown_buckets": bool(args.include_unknown_buckets),
        },
        "coverage": {
            "jsonl_rows_back_status": int(len(exec_rows)),
            "jsonl_rows_with_order_id": int(sum(1 for x in exec_rows if x.order_id)),
            "orders_with_accounting_pnl": int(len(pnl_by_order)),
            "accounting_meta": accounting_meta,
            "audit_rows_found": int(len(audit_map)),
            "final_observations": int(len(obs)),
            "dropped": dropped,
            "final_by_regime": {k: int(len(v)) for k, v in by_regime.items()},
        },
        "summary": summaries,
        "permutation_tests": perm_results,
        "oos_expanding_combo": oos_results,
        "notes": [
            "P&L/ROI usa modo accounting-real por order_id (balance_csv).",
            "slippage usa odd executada vs odd de decisão do sinal no executor_jsonl.",
            "latência usa timing.call_to_done_ms do executor_jsonl (tempo de execução da aposta).",
            "regime Pre/In é inferido por kickoff_time vs created_at (fallback: is_live no audit).",
            "permutação mantém exposição e buckets fixos e embaralha PnL para testar dependência bucket->resultado.",
            "OOS usa seleção de melhor combo lat_bucket x slip_bucket no treino (expanding) e avalia no dia seguinte.",
        ],
    }
    return _to_json_compatible(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Teste robusto Back Pre/In: latência x slippage com permutação de buckets + OOS expanding."
    )
    ap.add_argument("--executor-jsonl", default=os.getenv("EXECUTOR_JSONL", "logs/executor_live.jsonl"))
    ap.add_argument("--database-url", default="")
    ap.add_argument("--balance-csv", default="", help="CSV do ledger/accounting para P&L real por order_id.")
    ap.add_argument(
        "--accounting-report-json",
        default=os.getenv("ACCOUNTING_DAILY_REPORT_OUT", "logs/accounting_daily_report.json"),
        help="JSON do accounting diário com campo balance_csv (usado quando --balance-csv não for passado).",
    )
    ap.add_argument("--tz", default=os.getenv("REPORT_TZ", "America/Sao_Paulo"))
    ap.add_argument("--statuses", default="LIVE_OK")
    ap.add_argument("--start-day", default="", help="YYYY-MM-DD (dia local em --tz)")
    ap.add_argument("--end-day", default="", help="YYYY-MM-DD (dia local em --tz)")
    ap.add_argument("--perm-n", type=int, default=3000)
    ap.add_argument("--perm-seed", type=int, default=1337)
    ap.add_argument("--min-bucket-obs", type=int, default=25)
    ap.add_argument("--include-unknown-buckets", action="store_true")
    ap.add_argument("--oos-min-train-days", type=int, default=20)
    ap.add_argument("--oos-min-train-obs", type=int, default=80)
    ap.add_argument("--oos-min-test-obs", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import asyncio

    rep = asyncio.run(_run(args))
    txt = json.dumps(rep, ensure_ascii=False, indent=2)
    if str(args.out or "").strip():
        out_path = Path(str(args.out))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(txt, encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
