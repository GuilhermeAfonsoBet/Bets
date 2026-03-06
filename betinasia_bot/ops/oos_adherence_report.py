from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from loguru import logger
from sqlalchemy import text

from storage.database import Database


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _parse_iso(s: str) -> Optional[datetime]:
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


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _json_safe(x: Any) -> Any:
    if isinstance(x, datetime):
        return x.isoformat()
    if isinstance(x, date):
        return x.isoformat()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_json_safe(v) for v in list(x)]
    return x


def _iter_dates(d0: date, d1: date) -> Iterable[date]:
    cur = d0
    while cur <= d1:
        yield cur
        cur += timedelta(days=1)


def _local_day_bounds_utc(*, day: date, tz_name: str) -> Tuple[datetime, datetime]:
    """
    Retorna (start_utc, end_utc) do dia local [00:00, 24:00) em tz_name.
    """
    tz = timezone.utc
    try:
        from zoneinfo import ZoneInfo  # type: ignore

        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc
    start_local = datetime.combine(day, time(0, 0, 0), tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def _norm_line(line: str) -> str:
    return (str(line or "").strip()).replace(",", ".").replace("−", "-")


def _mult_back_from_scores(line: Any, side: str, hs: Any, aws: Any) -> Optional[float]:
    """
    Multiplicador do AH para a seleção "Back" (stake=1):
    +1.0 win, +0.5 half-win, 0 push, -0.5 half-loss, -1 loss.
    """
    if hs is None or aws is None:
        return None
    try:
        goal_diff = int(hs) - int(aws)
    except Exception:
        return None
    try:
        ah_line = float(str(line).replace(",", "."))
    except Exception:
        return None
    if (side or "").strip().lower() == "home":
        adjusted = goal_diff + ah_line
    else:
        adjusted = -goal_diff - ah_line
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
        return (odd - 1.0) * mult * 100.0
    if mult < 0:
        return mult * 100.0
    return 0.0


def _roi_lay_pct_per_liability(lay_odd: float, mult_back: float) -> Optional[float]:
    """
    Convenção alinhada ao relatório b808: ROI por liability para Lay.
    - mult_back é o multiplicador da seleção (se fosse Back).
    """
    liab = max(0.0, lay_odd - 1.0)
    if liab <= 0:
        return None
    if mult_back < 0:
        return (-mult_back) / liab * 100.0
    if mult_back > 0:
        return (-mult_back) * 100.0
    return 0.0


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    try:
        if len(xs) != len(ys) or len(xs) < 3:
            return None
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        vx = sum((x - mx) ** 2 for x in xs)
        vy = sum((y - my) ** 2 for y in ys)
        if vx <= 0 or vy <= 0:
            return None
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        return float(cov / (vx**0.5) / (vy**0.5))
    except Exception:
        return None


def _slip_raw_pct(*, odd_dec: Optional[float], odd_fin: Optional[float]) -> Optional[float]:
    if odd_dec is None or odd_fin is None or float(odd_dec) <= 0:
        return None
    return float((float(odd_fin) - float(odd_dec)) / float(odd_dec) * 100.0)


def _bucketize_3way_raw(pairs: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
    """
    Bucketiza por slippage_raw_pct (com sinal), em 3 faixas:
    - <= -2%
    - (-2%, 2%]
    - > 2%
    Retorna média de ROI por bucket.
    """
    outb: List[Dict[str, Any]] = []
    if not pairs:
        return outb
    buckets = [
        ("<= -2%", lambda s: s <= -2.0),
        ("(-2, 2]", lambda s: (s > -2.0) and (s <= 2.0)),
        ("> 2%", lambda s: s > 2.0),
    ]
    for lab, fn in buckets:
        ys = [roi for (slip, roi) in pairs if fn(float(slip))]
        if not ys:
            continue
        outb.append({"bucket": lab, "n": int(len(ys)), "roi_mean": float(sum(ys) / len(ys))})
    return outb


def _slip_cost_pct(*, exec_side: str, odd_dec: Optional[float], odd_fin: Optional[float]) -> Optional[float]:
    """
    Slippage "custo" em % (sempre >=0), normalizado por lado:
    - Back: custo quando a odd caiu (piorou).
    - Lay : custo quando a odd subiu (piorou).
    """
    raw = _slip_raw_pct(odd_dec=odd_dec, odd_fin=odd_fin)
    if raw is None:
        return None
    s = str(exec_side or "").strip().lower()
    if s == "back":
        return float(max(0.0, -raw))
    if s == "lay":
        return float(max(0.0, raw))
    return float(abs(raw))


def _sanitize_decimal_odd(odd: Optional[float]) -> Optional[float]:
    """
    Odds decimais típicas (AH/futebol) raramente passam de ~20.
    Para evitar explosões de ROI por dados corrompidos (ex.: 98 ao invés de 1.98),
    descartamos odds fora de um range plausível e tentamos um rescale simples quando fizer sentido.
    """
    if odd is None:
        return None
    try:
        o = float(odd)
    except Exception:
        return None
    if not (o > 1.0):
        return None
    # heurística: se vierem odds em "centavos" (ex.: 198 => 1.98), reescala
    if o >= 100.0 and o <= 3000.0:
        o2 = o / 100.0
        if o2 > 1.0 and o2 <= 30.0:
            return float(o2)
    # odds absurdas (ex.: 98) são quase sempre bug de parsing/scrape
    if o > 30.0:
        return None
    return float(o)


@dataclass
class ExecRow:
    created_at: datetime
    execution_id: str
    status: str
    exec_side: str
    is_live: bool
    audit_id: Optional[int]
    event_id: Optional[str]
    side: Optional[str]
    line: Optional[str]
    odd_decision: Optional[float]
    odd_final: Optional[float]
    stake_sent: Optional[float]


def _parse_executor_jsonl(path: Path) -> List[ExecRow]:
    if not path.exists():
        return []
    out: List[ExecRow] = []
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
        eid = str(res.get("execution_id") or req.get("execution_id") or "").strip()
        created = _parse_iso(str(res.get("created_at") or req.get("created_at") or "")) or None
        if not st or not eid or not created:
            continue

        raw = res.get("raw") if isinstance(res.get("raw"), dict) else {}
        sent = raw.get("sent") if isinstance(raw.get("sent"), dict) else {}
        stake_sent = _safe_float(sent.get("stake"))
        if stake_sent is None:
            # fallback: muitos executores registram stake na policy (request/result) e não em raw.sent
            pol = res.get("policy") if isinstance(res.get("policy"), dict) else (req.get("policy") if isinstance(req.get("policy"), dict) else {})
            stake_sent = _safe_float((pol or {}).get("stake_requested"))
        out.append(
            ExecRow(
                created_at=created,
                execution_id=eid,
                status=st,
                exec_side=str(res.get("exec_side") or req.get("exec_side") or ""),
                is_live=bool(res.get("is_live")) if res.get("is_live") is not None else bool(req.get("is_live")),
                audit_id=(int(res.get("audit_id")) if res.get("audit_id") is not None else (int(req.get("audit_id")) if req.get("audit_id") is not None else None)),
                event_id=str(res.get("event_id") or req.get("event_id") or "") or None,
                side=str(res.get("side") or req.get("side") or "") or None,
                line=str(res.get("line") or req.get("line") or "") or None,
                odd_decision=_safe_float(res.get("odd_at_decision") if res.get("odd_at_decision") is not None else req.get("odd_at_decision")),
                odd_final=_safe_float(res.get("odd_final")),
                stake_sent=stake_sent,
            )
        )
    return out


async def _fetch_bridge_stats(db: Database, *, start_utc: datetime, end_utc: datetime) -> List[Dict[str, Any]]:
    q = text(
        """
        SELECT
          action,
          COUNT(*)::bigint AS n_rows,
          SUM(CASE WHEN execution_id IS NOT NULL THEN 1 ELSE 0 END)::bigint AS n_with_execution_id,
          SUM(CASE WHEN (meta->>'skipped') = 'true' THEN 1 ELSE 0 END)::bigint AS n_skipped,
          SUM(CASE WHEN (meta->>'reason') = 'not_active' THEN 1 ELSE 0 END)::bigint AS n_not_active,
          SUM(CASE WHEN (meta->>'reason') = 'dup_key' THEN 1 ELSE 0 END)::bigint AS n_dup_key,
          SUM(CASE WHEN (meta->>'reason') = 'min_limit' THEN 1 ELSE 0 END)::bigint AS n_min_limit,
          SUM(CASE WHEN (meta->>'accepted') = 'true' THEN 1 ELSE 0 END)::bigint AS n_accepted,
          MIN(created_at) AS first_ts,
          MAX(created_at) AS last_ts
        FROM executor_bridge_seen
        WHERE created_at >= :t0 AND created_at < :t1
        GROUP BY action
        ORDER BY n_rows DESC
        """
    )
    async with db.async_session() as session:
        r = await session.execute(q, {"t0": start_utc, "t1": end_utc})
        rows = r.fetchall() or []
        out = []
        for x in rows:
            out.append(dict(x._mapping))
        return out


async def _fetch_audit_rows_for_ids(db: Database, ids: List[int]) -> Dict[int, Dict[str, Any]]:
    if not ids:
        return {}
    q = text(
        """
        SELECT
          a.id AS audit_id,
          a.event_id,
          a.league,
          a.market_type,
          a.line,
          a.side,
          a.is_live,
          a.hypothesis_type,
          a.reversal_direction,
          a.hypothesis_details,
          a.audited_at,
          m.id AS match_id,
          m.kickoff_time,
          m.home_score,
          m.away_score,
          m.status AS match_status
        FROM betslip_audit_results a
        LEFT JOIN matches m ON m.external_id = a.event_id
        WHERE a.id = ANY(:ids)
        """
    )
    async with db.async_session() as session:
        r = await session.execute(q, {"ids": ids})
        rows = r.fetchall() or []
        out: Dict[int, Dict[str, Any]] = {}
        for x in rows:
            d = dict(x._mapping)
            try:
                if isinstance(d.get("hypothesis_details"), str):
                    d["hypothesis_details"] = json.loads(d["hypothesis_details"])
            except Exception:
                pass
            out[int(d["audit_id"])] = d
        return out


def _active_keys_by_day_from_policy(policy: Dict[str, Any], *, tz_name: str) -> Dict[str, Any]:
    """
    Cria mapa day(YYYY-MM-DD em tz_name) -> active_keys do step cujo test window cobre o dia.
    """
    steps = policy.get("steps") if isinstance(policy.get("steps"), list) else []
    out: Dict[str, Any] = {"tz": tz_name, "days": {}}
    if not steps:
        return out

    tz = timezone.utc
    try:
        from zoneinfo import ZoneInfo  # type: ignore

        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc

    for st in steps:
        if not isinstance(st, dict):
            continue
        active = st.get("active_keys") if isinstance(st.get("active_keys"), list) else None
        # Formato atual do export do WF (b808): train_days/test_days como lista de YYYY-MM-DD (UTC).
        test_days = st.get("test_days") if isinstance(st.get("test_days"), list) else None
        if test_days:
            for ds in [str(x) for x in test_days if str(x)]:
                out["days"][ds] = {
                    "active_keys": active or [],
                    "n_active_keys": len(active or []),
                    "train": st.get("train"),
                    "test": st.get("test"),
                    "train_days": st.get("train_days"),
                    "test_days": test_days,
                }
            continue

        # Fallback: formato antigo com test_window.start/end (ISO)
        test = st.get("test_window") if isinstance(st.get("test_window"), dict) else {}
        t0 = _parse_iso(str(test.get("start") or "")) if isinstance(test, dict) else None
        t1 = _parse_iso(str(test.get("end") or "")) if isinstance(test, dict) else None
        if not t0 or not t1:
            continue
        d0 = t0.astimezone(tz).date()
        d1 = t1.astimezone(tz).date()
        for d in _iter_dates(d0, d1):
            out["days"][d.isoformat()] = {
                "active_keys": active or [],
                "n_active_keys": len(active or []),
                "test_window": {"start": t0.isoformat(), "end": t1.isoformat()},
                "train_window": st.get("train_window"),
            }
    return out


async def run_report(
    *,
    policy_json: Path,
    executor_jsonl: Path,
    tz_name: str,
    days: int,
    include_today: bool,
    out_json: Optional[Path],
) -> Dict[str, Any]:
    now_utc = datetime.now(timezone.utc)
    # intervalo de dias (por timezone do report)
    tz = timezone.utc
    try:
        from zoneinfo import ZoneInfo  # type: ignore

        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc
    now_local = now_utc.astimezone(tz)
    end_day = now_local.date() if include_today else (now_local.date() - timedelta(days=1))
    start_day = end_day - timedelta(days=max(0, int(days) - 1))

    policy = _load_json(policy_json) or {}
    active_by_day = _active_keys_by_day_from_policy(policy, tz_name=tz_name)
    steps = policy.get("steps") if isinstance(policy.get("steps"), list) else []
    last_step = steps[-1] if steps and isinstance(steps[-1], dict) else None

    exec_rows = _parse_executor_jsonl(executor_jsonl)
    db = Database()
    await db.connect()

    per_day = []
    # acumulado na janela (para análise estatística)
    total_pairs_raw_back: List[Tuple[float, float]] = []
    total_pairs_raw_lay: List[Tuple[float, float]] = []
    total_pairs_cost_back: List[Tuple[float, float]] = []
    total_pairs_cost_lay: List[Tuple[float, float]] = []
    for d in _iter_dates(start_day, end_day):
        start_utc, end_utc = _local_day_bounds_utc(day=d, tz_name=tz_name)

        # bridge adherence
        bridge_stats = await _fetch_bridge_stats(db, start_utc=start_utc, end_utc=end_utc)

        # executions in that window
        xs = [e for e in exec_rows if start_utc <= e.created_at < end_utc]
        # fetch audits for ROI
        audit_ids = sorted({int(e.audit_id) for e in xs if e.audit_id is not None})
        audit_map = await _fetch_audit_rows_for_ids(db, audit_ids)

        perf = {
            "n_exec_rows": len(xs),
            "status_counts": {},
            "back": {"n": 0, "wins": 0, "losses": 0, "push": 0, "half_wins": 0, "half_losses": 0, "stake_sum": 0.0, "pnl_sum": 0.0},
            "lay": {"n": 0, "wins": 0, "losses": 0, "push": 0, "half_wins": 0, "half_losses": 0, "liability_sum": 0.0, "pnl_sum": 0.0},
            "odd_anomalies": {"back": {"n": 0, "max": None}, "lay": {"n": 0, "max": None}},
            "slippage": {
                "back": {"n": 0, "raw_pct_mean": None, "cost_pct_mean": None},
                "lay": {"n": 0, "raw_pct_mean": None, "cost_pct_mean": None},
            },
            "slippage_vs_roi": {
                "back": {"corr_cost_pct_vs_roi": None, "buckets": []},
                "lay": {"corr_cost_pct_vs_roi": None, "buckets": []},
            },
        }

        slip_raw_back: List[float] = []
        slip_cost_back: List[float] = []
        slip_raw_lay: List[float] = []
        slip_cost_lay: List[float] = []
        pairs_back: List[Tuple[float, float]] = []
        pairs_lay: List[Tuple[float, float]] = []
        pairs_raw_back: List[Tuple[float, float]] = []
        pairs_raw_lay: List[Tuple[float, float]] = []

        def _bucketize(pairs: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
            edges = [(0.0, 0.0), (0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 999.0)]
            outb = []
            for a, b in edges:
                if a == b:
                    ys = [roi for (c, roi) in pairs if c == 0.0]
                    lab = "0"
                else:
                    ys = [roi for (c, roi) in pairs if (c > a and c <= b)]
                    lab = f"({a},{b}]"
                if not ys:
                    continue
                outb.append({"bucket": lab, "n": int(len(ys)), "roi_mean": float(sum(ys) / len(ys))})
            return outb
        for e in xs:
            perf["status_counts"][e.status] = int(perf["status_counts"].get(e.status, 0)) + 1

            # só calcula "resultado" quando há placar e odd
            a = audit_map.get(int(e.audit_id)) if e.audit_id is not None else None
            if not a:
                continue
            mult = _mult_back_from_scores(a.get("line") or e.line, a.get("side") or (e.side or ""), a.get("home_score"), a.get("away_score"))
            if mult is None:
                continue
            odd = _sanitize_decimal_odd(e.odd_final if e.odd_final is not None else e.odd_decision)
            if odd is None:
                # registra anomalia por lado (quando havia algo preenchido)
                raw_odd = e.odd_final if e.odd_final is not None else e.odd_decision
                side0 = str(e.exec_side or "").strip().lower()
                if raw_odd is not None and side0 in ("back", "lay"):
                    blk = perf.get("odd_anomalies", {}).get(side0, {})
                    try:
                        blk["n"] = int(blk.get("n") or 0) + 1
                        mx = blk.get("max")
                        v = float(raw_odd)
                        blk["max"] = v if mx is None else max(float(mx), v)
                    except Exception:
                        pass
                continue

            side = str(e.exec_side or "").strip()
            if side.lower() == "back":
                perf["back"]["n"] += 1
                stake = float(e.stake_sent) if e.stake_sent is not None else 1.0
                roi = _roi_back_pct(float(odd), float(mult))
                pnl = stake * roi / 100.0
                perf["back"]["stake_sum"] += float(stake)
                perf["back"]["pnl_sum"] += float(pnl)
                if mult > 0:
                    if mult == 0.5:
                        perf["back"]["half_wins"] += 1
                    else:
                        perf["back"]["wins"] += 1
                elif mult < 0:
                    if mult == -0.5:
                        perf["back"]["half_losses"] += 1
                    else:
                        perf["back"]["losses"] += 1
                else:
                    perf["back"]["push"] += 1

                raw_pct = _slip_raw_pct(odd_dec=e.odd_decision, odd_fin=e.odd_final)
                cost_pct = _slip_cost_pct(exec_side="back", odd_dec=e.odd_decision, odd_fin=e.odd_final)
                if raw_pct is not None and cost_pct is not None:
                    slip_raw_back.append(float(raw_pct))
                    slip_cost_back.append(float(cost_pct))
                if cost_pct is not None:
                    pairs_back.append((float(cost_pct), float(roi)))
                if raw_pct is not None:
                    pairs_raw_back.append((float(raw_pct), float(roi)))
            elif side.lower() == "lay":
                perf["lay"]["n"] += 1
                stake = float(e.stake_sent) if e.stake_sent is not None else 1.0
                liab = stake * max(0.0, float(odd) - 1.0)
                roi_liab = _roi_lay_pct_per_liability(float(odd), float(mult))
                if roi_liab is None:
                    continue
                pnl = liab * float(roi_liab) / 100.0
                perf["lay"]["liability_sum"] += float(liab)
                perf["lay"]["pnl_sum"] += float(pnl)
                # outcome invertido vs Back
                if mult > 0:
                    if mult == 0.5:
                        perf["lay"]["half_losses"] += 1
                    else:
                        perf["lay"]["losses"] += 1
                elif mult < 0:
                    if mult == -0.5:
                        perf["lay"]["half_wins"] += 1
                    else:
                        perf["lay"]["wins"] += 1
                else:
                    perf["lay"]["push"] += 1

                raw_pct = _slip_raw_pct(odd_dec=e.odd_decision, odd_fin=e.odd_final)
                cost_pct = _slip_cost_pct(exec_side="lay", odd_dec=e.odd_decision, odd_fin=e.odd_final)
                if raw_pct is not None and cost_pct is not None:
                    slip_raw_lay.append(float(raw_pct))
                    slip_cost_lay.append(float(cost_pct))
                if cost_pct is not None and roi_liab is not None:
                    pairs_lay.append((float(cost_pct), float(roi_liab)))
                if raw_pct is not None and roi_liab is not None:
                    pairs_raw_lay.append((float(raw_pct), float(roi_liab)))

        # ROIs agregados
        back_roi = (float(perf["back"]["pnl_sum"]) / float(perf["back"]["stake_sum"]) * 100.0) if perf["back"]["stake_sum"] else None
        lay_roi = (float(perf["lay"]["pnl_sum"]) / float(perf["lay"]["liability_sum"]) * 100.0) if perf["lay"]["liability_sum"] else None
        perf["back"]["roi_pct"] = back_roi
        perf["lay"]["roi_pct_per_liability"] = lay_roi

        # slippage agregada
        if slip_raw_back:
            perf["slippage"]["back"]["n"] = int(len(slip_raw_back))
            perf["slippage"]["back"]["raw_pct_mean"] = float(sum(slip_raw_back) / len(slip_raw_back))
            perf["slippage"]["back"]["cost_pct_mean"] = float(sum(slip_cost_back) / len(slip_cost_back)) if slip_cost_back else None
        if slip_raw_lay:
            perf["slippage"]["lay"]["n"] = int(len(slip_raw_lay))
            perf["slippage"]["lay"]["raw_pct_mean"] = float(sum(slip_raw_lay) / len(slip_raw_lay))
            perf["slippage"]["lay"]["cost_pct_mean"] = float(sum(slip_cost_lay) / len(slip_cost_lay)) if slip_cost_lay else None

        if pairs_back:
            perf["slippage_vs_roi"]["back"]["corr_cost_pct_vs_roi"] = _pearson([c for c, _ in pairs_back], [r for _, r in pairs_back])
            perf["slippage_vs_roi"]["back"]["buckets"] = _bucketize(pairs_back)
        if pairs_lay:
            perf["slippage_vs_roi"]["lay"]["corr_cost_pct_vs_roi"] = _pearson([c for c, _ in pairs_lay], [r for _, r in pairs_lay])
            perf["slippage_vs_roi"]["lay"]["buckets"] = _bucketize(pairs_lay)

        # Buckets por slippage RAW (com sinal): <=-2%, -2..2, >2
        perf["slippage_vs_roi_raw"] = {
            "back": {"buckets": _bucketize_3way_raw(pairs_raw_back)},
            "lay": {"buckets": _bucketize_3way_raw(pairs_raw_lay)},
        }

        # acumula (janela inteira)
        if pairs_raw_back:
            total_pairs_raw_back.extend(list(pairs_raw_back))
        if pairs_raw_lay:
            total_pairs_raw_lay.extend(list(pairs_raw_lay))
        if pairs_back:
            total_pairs_cost_back.extend(list(pairs_back))
        if pairs_lay:
            total_pairs_cost_lay.extend(list(pairs_lay))

        per_day.append(
            {
                "day": d.isoformat(),
                "start_utc": start_utc.isoformat(),
                "end_utc": end_utc.isoformat(),
                "policy": active_by_day.get("days", {}).get(d.isoformat()),
                "bridge": bridge_stats,
                "execution": perf,
            }
        )

    # Carry-forward do último step para dias sem mapeamento (test_days com gaps),
    # refletindo a operação real (policy "corrente" vale até próxima atualização).
    if last_step and isinstance(last_step, dict):
        for it in per_day:
            if not isinstance(it, dict):
                continue
            if it.get("policy") is None:
                it["policy"] = {
                    "active_keys": list(last_step.get("active_keys") or []),
                    "n_active_keys": len(list(last_step.get("active_keys") or [])),
                    "train": last_step.get("train"),
                    "test": last_step.get("test"),
                    "train_days": last_step.get("train_days"),
                    "test_days": last_step.get("test_days"),
                    "carried_forward": True,
                }

    out = {
        "ts_utc": now_utc.isoformat(),
        "tz": tz_name,
        "policy_json": str(policy_json),
        "executor_jsonl": str(executor_jsonl),
        "range": {"start_day": start_day.isoformat(), "end_day": end_day.isoformat(), "days": int(days), "include_today": bool(include_today)},
        "policy_days": active_by_day,
        "per_day": per_day,
        # Estatística acumulada na janela
        "slippage_vs_roi_raw_total": {
            "back": {"buckets": _bucketize_3way_raw(total_pairs_raw_back)},
            "lay": {"buckets": _bucketize_3way_raw(total_pairs_raw_lay)},
        },
        "slippage_vs_roi_total": {
            "back": {
                "n": int(len(total_pairs_cost_back)),
                "corr_cost_pct_vs_roi": _pearson([c for c, _ in total_pairs_cost_back], [r for _, r in total_pairs_cost_back]) if total_pairs_cost_back else None,
            },
            "lay": {
                "n": int(len(total_pairs_cost_lay)),
                "corr_cost_pct_vs_roi": _pearson([c for c, _ in total_pairs_cost_lay], [r for _, r in total_pairs_cost_lay]) if total_pairs_cost_lay else None,
            },
        },
    }
    out = _json_safe(out)
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Relatório de aderência OOS: portfolio (WF policy) por dia + execução (bridge+executor) + ROI por placar.")
    ap.add_argument("--policy-json", default=os.getenv("BRIDGE_POLICY_JSON", "logs/wf_policy_current.json"))
    ap.add_argument("--executor-jsonl", default=os.getenv("EXECUTOR_JSONL", "logs/executor_live.jsonl"))
    ap.add_argument("--tz", default=os.getenv("REPORT_TZ", "America/Sao_Paulo"))
    ap.add_argument("--days", type=int, default=int(os.getenv("OOS_ADHERENCE_DAYS", "7")))
    ap.add_argument("--include-today", action="store_true", default=(os.getenv("OOS_ADHERENCE_INCLUDE_TODAY", "1").strip() not in ("0", "false", "False", "no", "NO")))
    ap.add_argument("--out", default=os.getenv("OOS_ADHERENCE_OUT", "").strip() or None)
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))

    policy = Path(str(args.policy_json))
    jsonl = Path(str(args.executor_jsonl))
    outp = Path(str(args.out)) if args.out else None

    import asyncio

    rep = asyncio.run(
        run_report(
            policy_json=policy,
            executor_jsonl=jsonl,
            tz_name=str(args.tz),
            days=int(args.days),
            include_today=bool(args.include_today),
            out_json=outp,
        )
    )
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

