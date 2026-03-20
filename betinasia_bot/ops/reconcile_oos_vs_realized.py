from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from loguru import logger
from sqlalchemy import text

from storage.database import Database


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


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        return int(s)
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


def _pick_col(cols: List[str], needles: Iterable[str]) -> Optional[str]:
    cols_map = {c.lower(): c for c in cols}
    cols_l = list(cols_map.keys())
    for n in needles:
        n = str(n).lower()
        if n in cols_map:
            return cols_map[n]
        for cl in cols_l:
            if cl.startswith(n):
                return cols_map[cl]
        for cl in cols_l:
            if n in cl:
                return cols_map[cl]
    return None


def _parse_dt_any(s: str) -> Optional[datetime]:
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
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y", "%d/%m/%Y %H:%M:%S"):
            try:
                dt = datetime.strptime(t, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except Exception:
                continue
    except Exception:
        return None
    return None


def _home_handicap_from_line_and_side(line: Any, side: str) -> Optional[float]:
    """
    Converte (line, side) para handicap do HOME (assinado).

    Convenção:
    - Se `line` vier com sinal (ex.: "-0.5"), assume que já é o handicap do HOME.
    - Se `line` vier sem sinal (ex.: "2"), assume que é a magnitude do handicap do `side`:
      - side=home => home_handicap=+2
      - side=away => home_handicap=-2
    """
    try:
        sel = (side or "").strip().lower()
        raw = str(line).strip().replace(",", ".").replace("−", "-")
        if not raw:
            return None
        ah = float(raw)
        if raw.startswith("+") or raw.startswith("-"):
            return float(ah)
        if sel == "home":
            return float(ah)
        if sel == "away":
            return float(-ah)
        return None
    except Exception:
        return None


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
    sel = (side or "").strip().lower()
    home_hcp = _home_handicap_from_line_and_side(line, sel)
    if home_hcp is None:
        return None
    if sel == "home":
        adjusted = goal_diff + home_hcp
    elif sel == "away":
        adjusted = -goal_diff - home_hcp
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
        return (odd - 1.0) * mult * 100.0
    if mult < 0:
        return mult * 100.0
    return 0.0


def _roi_lay_pct_per_liability(lay_odd: float, mult_back: float) -> Optional[float]:
    liab = max(0.0, float(lay_odd) - 1.0)
    if liab <= 0:
        return None
    if mult_back > 0:
        # seleção (Back) ganha => Lay perde 100% da liability
        return -100.0 * float(mult_back)
    if mult_back < 0:
        # seleção perde => Lay ganha stake = (-mult_back), dividido por liability
        return (float(-mult_back) / liab) * 100.0
    return 0.0


def _sanitize_decimal_odd(odd: Any) -> Optional[float]:
    o = _safe_float(odd)
    if o is None:
        return None
    if o <= 1.0:
        return None
    # heurística: odds em "centavos" (198 => 1.98)
    if 100.0 <= o <= 3000.0:
        o2 = o / 100.0
        if 1.0 < o2 <= 30.0:
            return float(o2)
    if o > 30.0:
        return None
    return float(o)


def _extract_order_id(order_resp: Any) -> Optional[str]:
    try:
        if not order_resp:
            return None
        if isinstance(order_resp, str):
            s = order_resp.strip()
            return s or None
        if isinstance(order_resp, dict):
            for k in ("id", "order_id", "orderId", "uuid", "uid"):
                v = order_resp.get(k)
                if v is None:
                    continue
                s = str(v).strip()
                if s:
                    return s
            for k in ("data", "order", "result"):
                v = order_resp.get(k)
                if isinstance(v, dict):
                    r = _extract_order_id(v)
                    if r:
                        return r
        return None
    except Exception:
        return None


def _extract_order_id_from_raw(raw: Any) -> Optional[str]:
    if not isinstance(raw, dict):
        return None
    try:
        oid = str(raw.get("order_id") or "").strip() or None
    except Exception:
        oid = None
    if oid:
        return oid
    try:
        return _extract_order_id(raw.get("order_resp"))
    except Exception:
        return None


@dataclass
class ExecOrder:
    order_id: str
    created_at: datetime
    status: str
    exec_side: str
    is_live: bool
    audit_id: Optional[int]
    match_id: Optional[int]
    side: Optional[str]
    line: Optional[str]
    odd_decision: Optional[float]
    odd_final: Optional[float]
    stake_sent: Optional[float]
    liability_req: Optional[float]


def _parse_executor_jsonl_orders(path: Path) -> Dict[str, ExecOrder]:
    if not path.exists():
        return {}
    out: Dict[str, ExecOrder] = {}
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
        if not st or st == "HEARTBEAT":
            continue
        created = _parse_iso(str(res.get("created_at") or req.get("created_at") or "")) or None
        if not created:
            continue

        raw = res.get("raw") if isinstance(res.get("raw"), dict) else {}
        oid = _extract_order_id_from_raw(raw)
        if not oid:
            continue

        sent = raw.get("sent") if isinstance(raw.get("sent"), dict) else {}
        stake_sent = _safe_float(sent.get("stake"))
        odd_final = _sanitize_decimal_odd(res.get("odd_final"))

        pol = res.get("policy") if isinstance(res.get("policy"), dict) else (req.get("policy") if isinstance(req.get("policy"), dict) else {})
        liab_req = _safe_float((pol or {}).get("liability_requested"))
        if stake_sent is None:
            if str(res.get("exec_side") or req.get("exec_side") or "").strip().lower() == "lay":
                price = _sanitize_decimal_odd(sent.get("price")) or odd_final
                if liab_req is not None and price is not None and float(price) > 1.0 and float(liab_req) > 0:
                    stake_sent = float(liab_req) / (float(price) - 1.0)
            else:
                stake_sent = _safe_float((pol or {}).get("stake_requested"))

        eo = ExecOrder(
            order_id=str(oid),
            created_at=created,
            status=st,
            exec_side=str(res.get("exec_side") or req.get("exec_side") or "").strip(),
            is_live=bool(res.get("is_live")) if res.get("is_live") is not None else bool(req.get("is_live")),
            audit_id=(_safe_int(res.get("audit_id")) if res.get("audit_id") is not None else (_safe_int(req.get("audit_id")) if req.get("audit_id") is not None else None)),
            match_id=(_safe_int(res.get("match_id")) if res.get("match_id") is not None else (_safe_int(req.get("match_id")) if req.get("match_id") is not None else None)),
            side=str(res.get("side") or req.get("side") or "").strip() or None,
            line=str(res.get("line") or req.get("line") or "").strip() or None,
            odd_decision=_sanitize_decimal_odd(res.get("odd_at_decision") if res.get("odd_at_decision") is not None else req.get("odd_at_decision")),
            odd_final=odd_final,
            stake_sent=stake_sent,
            liability_req=liab_req,
        )

        # Em caso de duplicata (retries), preferimos o mais recente
        prev = out.get(eo.order_id)
        if prev is None or eo.created_at >= prev.created_at:
            out[eo.order_id] = eo
    return out


def _read_accounting_by_order(balance_csv: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
    """
    Retorna:
    - rows_by_order: order_id -> {post_day, pnl, type, raw...} (agregado)
    - pnl_by_post_day: day -> pnl_total
    """
    rows_by_order: Dict[str, Dict[str, Any]] = {}
    pnl_by_post_day: Dict[str, float] = defaultdict(float)
    with balance_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        if not cols:
            return {}, {}
        dt_col = _pick_col(cols, ("post date", "post_date", "date", "settled", "closed", "time"))
        pnl_col = _pick_col(cols, ("amount", "profit_loss", "profit", "p&l", "pnl", "net", "pl"))
        typ_col = _pick_col(cols, ("type",))
        oid_col = _pick_col(cols, ("order_id", "order id", "order", "bet id", "bet_id", "id"))

        for row in reader:
            if not isinstance(row, dict):
                continue
            if not oid_col or not pnl_col or not dt_col:
                continue
            oid = str(row.get(oid_col) or "").strip()
            if not oid or not oid.isdigit():
                continue
            pnl = _safe_float(row.get(pnl_col))
            if pnl is None:
                continue
            dt = _parse_dt_any(str(row.get(dt_col) or ""))
            if dt is None:
                continue
            post_day = dt.date().isoformat()
            typ = str(row.get(typ_col) or "").strip() if typ_col else ""

            # para "balance ledger", pode ter vários tipos; focamos em type=bet quando existir
            if typ and typ.lower() != "bet":
                continue

            agg = rows_by_order.get(oid)
            if not agg:
                agg = {"order_id": oid, "post_day": post_day, "pnl": 0.0, "type": typ}
                rows_by_order[oid] = agg
            agg["pnl"] = float(agg.get("pnl") or 0.0) + float(pnl)
            # se houver múltiplas datas para o mesmo order_id (raro), mantemos a mais recente
            if str(post_day) > str(agg.get("post_day") or ""):
                agg["post_day"] = post_day
            pnl_by_post_day[post_day] += float(pnl)

    return rows_by_order, dict(pnl_by_post_day)


async def _fetch_audits_for_ids(db: Database, ids: List[int]) -> Dict[int, Dict[str, Any]]:
    if not ids:
        return {}
    q = text(
        """
        SELECT
          a.id AS audit_id,
          a.event_id,
          a.market_type,
          a.line,
          a.side,
          a.is_live,
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
    out: Dict[int, Dict[str, Any]] = {}
    async with db.async_session() as session:
        r = await session.execute(q, {"ids": ids})
        rows = r.fetchall() or []
        for x in rows:
            out[int(x._mapping["audit_id"])] = dict(x._mapping)
    return out


def _pnl_from_scores(*, exec_side: str, odd: float, stake: float, liability: float, mult_back: float) -> Optional[float]:
    side0 = str(exec_side or "").strip().lower()
    if side0 == "back":
        roi = _roi_back_pct(float(odd), float(mult_back))
        return float(stake) * float(roi) / 100.0
    if side0 == "lay":
        roi_liab = _roi_lay_pct_per_liability(float(odd), float(mult_back))
        if roi_liab is None:
            return None
        return float(liability) * float(roi_liab) / 100.0
    return None


def _fmt(x: Any, nd: int = 2) -> str:
    try:
        if x is None:
            return "—"
        return f"{float(x):.{int(nd)}f}"
    except Exception:
        return "—"


async def run(*, balance_csv: Path, executor_jsonl: Path, out_csv: Optional[Path], only_days: Optional[List[str]]) -> Dict[str, Any]:
    acct_by_order, _ = _read_accounting_by_order(balance_csv)
    exec_by_order = _parse_executor_jsonl_orders(executor_jsonl)

    audit_ids = sorted({int(v.audit_id) for v in exec_by_order.values() if v.audit_id is not None})
    db = Database()
    await db.connect()
    audit_map = await _fetch_audits_for_ids(db, audit_ids)

    rows_out: List[Dict[str, Any]] = []
    agg_by_exec_day: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    agg_by_post_day: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for oid, a in acct_by_order.items():
        post_day = str(a.get("post_day") or "")
        if only_days and post_day not in only_days:
            continue
        acct_pnl = float(a.get("pnl") or 0.0)
        e = exec_by_order.get(oid)
        if not e:
            rows_out.append(
                {
                    "order_id": oid,
                    "post_day": post_day,
                    "exec_day": None,
                    "acct_pnl": acct_pnl,
                    "bot": False,
                    "reason": "NO_EXECUTOR_MATCH",
                }
            )
            agg_by_post_day[post_day]["acct_pnl"] += acct_pnl
            agg_by_post_day[post_day]["acct_pnl_nonbot_or_unmatched"] += acct_pnl
            continue

        exec_day = e.created_at.date().isoformat()
        side0 = str(e.exec_side or "").strip().lower()
        stake = float(e.stake_sent) if e.stake_sent is not None else 0.0
        odd_dec = _sanitize_decimal_odd(e.odd_decision)
        odd_fin = _sanitize_decimal_odd(e.odd_final)

        # liability para Lay: preferimos a explícita; senão, inferimos por stake*(odd-1)
        liability = float(e.liability_req) if (e.liability_req is not None and float(e.liability_req) > 0) else 0.0
        if side0 == "lay" and liability <= 0 and stake > 0 and odd_fin and odd_fin > 1.0:
            liability = stake * max(0.0, odd_fin - 1.0)
        if side0 == "back":
            liability = 0.0

        aud = audit_map.get(int(e.audit_id)) if e.audit_id is not None else None
        hs = (aud or {}).get("home_score")
        aws = (aud or {}).get("away_score")
        a_side = (aud or {}).get("side") or e.side
        a_line = (aud or {}).get("line") or e.line
        mult = _mult_back_from_scores(a_line, str(a_side or ""), hs, aws)

        pnl_dec = None
        pnl_fin = None
        if mult is not None and ((side0 == "back" and stake > 0 and odd_dec) or (side0 == "lay" and (liability > 0) and odd_dec)):
            if odd_dec is not None and odd_dec > 1.0:
                pnl_dec = _pnl_from_scores(exec_side=side0, odd=float(odd_dec), stake=stake, liability=liability, mult_back=float(mult))
        if mult is not None and ((side0 == "back" and stake > 0 and odd_fin) or (side0 == "lay" and (liability > 0) and odd_fin)):
            if odd_fin is not None and odd_fin > 1.0:
                pnl_fin = _pnl_from_scores(exec_side=side0, odd=float(odd_fin), stake=stake, liability=liability, mult_back=float(mult))

        delta_slip = (float(pnl_fin) - float(pnl_dec)) if (pnl_fin is not None and pnl_dec is not None) else None
        delta_resid = (float(acct_pnl) - float(pnl_fin)) if (pnl_fin is not None) else None

        rows_out.append(
            {
                "order_id": oid,
                "post_day": post_day,
                "exec_day": exec_day,
                "acct_pnl": acct_pnl,
                "bot": True,
                "exec_status": e.status,
                "exec_side": e.exec_side,
                "is_live": bool(e.is_live),
                "audit_id": e.audit_id,
                "match_id": (aud or {}).get("match_id") or e.match_id,
                "line": a_line,
                "side": a_side,
                "home_score": hs,
                "away_score": aws,
                "mult_back": mult,
                "stake_sent": stake if stake > 0 else None,
                "liability": liability if liability > 0 else None,
                "odd_decision": odd_dec,
                "odd_final": odd_fin,
                "pnl_score_at_decision_odd": pnl_dec,
                "pnl_score_at_final_odd": pnl_fin,
                "delta_slippage_pnl": delta_slip,
                "delta_residual_acct_minus_score": delta_resid,
            }
        )

        agg_by_post_day[post_day]["acct_pnl"] += acct_pnl
        agg_by_exec_day[exec_day]["acct_pnl"] += acct_pnl
        if pnl_dec is not None:
            agg_by_exec_day[exec_day]["pnl_dec"] += float(pnl_dec)
        if pnl_fin is not None:
            agg_by_exec_day[exec_day]["pnl_fin"] += float(pnl_fin)
        if delta_slip is not None:
            agg_by_exec_day[exec_day]["delta_slip"] += float(delta_slip)
        if delta_resid is not None:
            agg_by_exec_day[exec_day]["delta_resid"] += float(delta_resid)

    # export CSV
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        cols = []
        for r in rows_out:
            for k in r.keys():
                if k not in cols:
                    cols.append(k)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows_out:
                w.writerow(r)

    # resumo (alto sinal)
    top_slip = sorted(
        [r for r in rows_out if r.get("delta_slippage_pnl") is not None],
        key=lambda r: abs(float(r.get("delta_slippage_pnl") or 0.0)),
        reverse=True,
    )[:15]
    top_resid = sorted(
        [r for r in rows_out if r.get("delta_residual_acct_minus_score") is not None],
        key=lambda r: abs(float(r.get("delta_residual_acct_minus_score") or 0.0)),
        reverse=True,
    )[:15]

    return {
        "balance_csv": str(balance_csv),
        "executor_jsonl": str(executor_jsonl),
        "out_csv": str(out_csv) if out_csv else None,
        "n_orders_acct": int(len(acct_by_order)),
        "n_orders_exec_matched": int(sum(1 for r in rows_out if r.get("bot") is True)),
        "n_orders_exec_unmatched": int(sum(1 for r in rows_out if r.get("bot") is False)),
        "by_exec_day": {k: dict(v) for k, v in sorted(agg_by_exec_day.items())},
        "by_post_day": {k: dict(v) for k, v in sorted(agg_by_post_day.items())},
        "top_delta_slippage": top_slip,
        "top_delta_residual": top_resid,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Conciliação por ordem: OOS-proxy (odd_decision) vs score (odd_final) vs accounting.")
    ap.add_argument("--balance-csv", required=True, help="CSV do accounting (balance ledger).")
    ap.add_argument("--executor-jsonl", default=os.getenv("EXECUTOR_JSONL", "logs/executor_live.jsonl"), help="executor_live.jsonl")
    ap.add_argument("--out-csv", default="", help="Export CSV (opcional).")
    ap.add_argument("--only-days", default="", help="Lista de post_day YYYY-MM-DD separada por vírgula (opcional).")
    ap.add_argument("--json", action="store_true", default=False, help="Imprime JSON resumido (além do texto).")
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))

    bal = Path(str(args.balance_csv))
    j = Path(str(args.executor_jsonl))
    out_csv = Path(str(args.out_csv)) if str(args.out_csv or "").strip() else None
    only_days = [x.strip() for x in str(args.only_days or "").split(",") if x.strip()] or None

    rep = None
    try:
        import asyncio

        rep = asyncio.run(run(balance_csv=bal, executor_jsonl=j, out_csv=out_csv, only_days=only_days))
    except Exception as e:
        print(json.dumps({"error": str(e)[:400]}, ensure_ascii=False))
        return 2

    by_exec_day = rep.get("by_exec_day") or {}
    print("\n### CONCILIAÇÃO (por exec_day; ordens BOT mapeadas)\n")
    print("exec_day    acct_pnl     pnl_dec(odd_dec)  pnl_fin(odd_fin)  delta_slip  delta_resid")
    for day, d in by_exec_day.items():
        acct = d.get("acct_pnl")
        pdec = d.get("pnl_dec")
        pfin = d.get("pnl_fin")
        dsl = d.get("delta_slip")
        dr = d.get("delta_resid")
        print(f"{day}  {_fmt(acct,2):>9}  {_fmt(pdec,2):>14}  {_fmt(pfin,2):>13}  {_fmt(dsl,2):>9}  {_fmt(dr,2):>11}")

    print("\n### TOP |delta_slippage_pnl| (score@odd_final - score@odd_decision)\n")
    for r in (rep.get("top_delta_slippage") or [])[:10]:
        print(
            "order_id=%s exec_day=%s side=%s line=%s mult=%s odd_dec=%s odd_fin=%s stake=%s liab=%s dslip=%s acct=%s"
            % (
                r.get("order_id"),
                r.get("exec_day"),
                r.get("exec_side"),
                r.get("line"),
                r.get("mult_back"),
                _fmt(r.get("odd_decision"), 3),
                _fmt(r.get("odd_final"), 3),
                _fmt(r.get("stake_sent"), 2),
                _fmt(r.get("liability"), 2),
                _fmt(r.get("delta_slippage_pnl"), 2),
                _fmt(r.get("acct_pnl"), 2),
            )
        )

    print("\n### TOP |delta_residual| (accounting - score@odd_final)\n")
    for r in (rep.get("top_delta_residual") or [])[:10]:
        print(
            "order_id=%s exec_day=%s side=%s line=%s mult=%s odd_fin=%s stake=%s liab=%s resid=%s acct=%s pnl_fin=%s"
            % (
                r.get("order_id"),
                r.get("exec_day"),
                r.get("exec_side"),
                r.get("line"),
                r.get("mult_back"),
                _fmt(r.get("odd_final"), 3),
                _fmt(r.get("stake_sent"), 2),
                _fmt(r.get("liability"), 2),
                _fmt(r.get("delta_residual_acct_minus_score"), 2),
                _fmt(r.get("acct_pnl"), 2),
                _fmt(r.get("pnl_score_at_final_odd"), 2),
            )
        )

    if args.json:
        print("\n---\n")
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

