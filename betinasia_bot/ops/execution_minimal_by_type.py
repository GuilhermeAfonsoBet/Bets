from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import aiohttp
from sqlalchemy import text

from storage.database import Database


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _load_env_file(path: Path) -> None:
    """
    Loader simples de .env para permitir rodar este módulo isoladamente (sem depender do shell exportar vars).
    - Não sobrescreve variáveis já presentes no ambiente.
    - Ignora comentários e linhas inválidas.
    """
    try:
        if not path.exists():
            return
        for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = str(ln).strip()
            if not s or s.startswith("#"):
                continue
            if "=" not in s:
                continue
            k, v = s.split("=", 1)
            k = k.strip()
            if not k:
                continue
            v = v.strip()
            if (len(v) >= 2) and ((v[0] == v[-1]) and v[0] in ("'", '"')):
                v = v[1:-1]
            if k in os.environ and str(os.environ.get(k) or "").strip():
                continue
            os.environ[k] = v
    except Exception:
        return


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


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        # formatos comuns do BetinAsia: ["USD", -10.58]
        if isinstance(x, (list, tuple)) and len(x) >= 2:
            return float(x[1])
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _dig_first_str(d: Any, keys: Iterable[str]) -> Optional[str]:
    if not isinstance(d, dict):
        return None
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    for v in d.values():
        if isinstance(v, dict):
            r = _dig_first_str(v, keys)
            if r:
                return r
    return None


def _dig_first_float(d: Any, keys: Iterable[str]) -> Optional[float]:
    if not isinstance(d, dict):
        return None
    for k in keys:
        v = d.get(k)
        fv = _safe_float(v)
        if fv is not None:
            return float(fv)
    for v in d.values():
        if isinstance(v, dict):
            r = _dig_first_float(v, keys)
            if r is not None:
                return float(r)
    return None


def _norm_line(line: str) -> str:
    return (str(line or "").strip()).replace(",", ".").replace("−", "-")


def _mult_back_from_scores(line: Any, side: str, hs: Any, aws: Any) -> Optional[float]:
    """
    Multiplicador do AH para a seleção "Back" (stake=1):
    +1.0 win, +0.5 half-win, 0 push, -0.5 half-loss, -1 loss.
    """
    try:
        if hs is None or aws is None:
            return None
        goal_diff = int(hs) - int(aws)
        ah_line = float(str(line).replace(",", "."))
        sel = str(side or "").strip().lower()
        if sel == "home":
            adjusted = goal_diff + ah_line
        elif sel == "away":
            adjusted = -goal_diff - ah_line
        else:
            return None
        if adjusted > 0.5:
            return 1.0
        if adjusted == 0.5:
            return 0.5
        if adjusted == 0.0:
            return 0.0
        if adjusted == -0.5:
            return -0.5
        if adjusted < -0.5:
            return -1.0
    except Exception:
        return None
    return None


def _sanitize_decimal_odd(odd: Any) -> Optional[float]:
    o = _safe_float(odd)
    if o is None:
        return None
    if o <= 1.0:
        return None
    # odds absurdas (ex.: 98) quase sempre bug de parsing/scrape
    if o > 30.0:
        return None
    return float(o)


def _roi_back_pct(odd: float, mult: float) -> float:
    """
    ROI % por stake para Back, dado mult do AH.
    """
    b = float(odd) - 1.0
    if mult > 0:
        return 100.0 * float(mult) * float(b)
    if mult < 0:
        return 100.0 * float(mult)
    return 0.0


def _roi_lay_pct_per_liability(lay_odd: float, mult_back: float) -> Optional[float]:
    """
    ROI% por liability para Lay, compatível com a convenção do relatório b808.
    - mult_back é o multiplicador da seleção (se fosse Back).
    """
    try:
        o = float(lay_odd)
        if o <= 1.0:
            return None
        # Se a seleção (Back) ganha, o Lay perde a liability (100%).
        if mult_back > 0:
            return -100.0 * float(mult_back)
        # Se a seleção perde, o Lay ganha o stake = liab/(o-1) => ROI/liab = 1/(o-1)
        if mult_back < 0:
            return 100.0 * (-float(mult_back)) * (1.0 / (o - 1.0))
        return 0.0
    except Exception:
        return None


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


@dataclass
class ExecLine:
    created_at: datetime
    status: str
    exec_side: str
    is_live: bool
    audit_id: Optional[int]
    odd_final: Optional[float]
    stake_sent: Optional[float]
    liability_req: Optional[float]
    order_id: Optional[str] = None


def _parse_executor_jsonl(path: Path) -> List[ExecLine]:
    if not path.exists():
        return []
    out: List[ExecLine] = []
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
        created = _parse_iso(str(res.get("created_at") or req.get("created_at") or "")) or None
        if not st or not created:
            continue
        raw = res.get("raw") if isinstance(res.get("raw"), dict) else {}
        sent = raw.get("sent") if isinstance(raw.get("sent"), dict) else {}
        stake_sent = _safe_float(sent.get("stake"))
        odd_final = _safe_float(res.get("odd_final"))
        order_id = None
        try:
            order_id = str(raw.get("order_id") or "").strip() or None
        except Exception:
            order_id = None
        if not order_id:
            try:
                order_id = _extract_order_id(raw.get("order_resp"))
            except Exception:
                order_id = None

        pol = res.get("policy") if isinstance(res.get("policy"), dict) else (req.get("policy") if isinstance(req.get("policy"), dict) else {})
        liab_req = _safe_float((pol or {}).get("liability_requested"))

        # Fallback para stake_sent (especialmente importante em DRY, que não preenche raw.sent):
        if stake_sent is None:
            if str(res.get("exec_side") or req.get("exec_side") or "").strip().lower() == "lay":
                price = _safe_float(sent.get("price")) or odd_final
                if liab_req is not None and price is not None and float(price) > 1.0 and float(liab_req) > 0:
                    stake_sent = float(liab_req) / (float(price) - 1.0)
            else:
                stake_sent = _safe_float((pol or {}).get("stake_requested"))

        out.append(
            ExecLine(
                created_at=created,
                status=st,
                exec_side=str(res.get("exec_side") or req.get("exec_side") or "").strip(),
                is_live=bool(res.get("is_live")) if res.get("is_live") is not None else bool(req.get("is_live")),
                audit_id=(
                    _safe_int(res.get("audit_id")) if res.get("audit_id") is not None else (_safe_int(req.get("audit_id")) if req.get("audit_id") is not None else None)
                ),
                odd_final=odd_final,
                stake_sent=stake_sent,
                liability_req=liab_req,
                order_id=order_id,
            )
        )
    return out


async def _fetch_audits_for_ids(db: Database, ids: List[int]) -> Dict[int, Dict[str, Any]]:
    if not ids:
        return {}
    q = text(
        """
        SELECT
          a.id AS audit_id,
          a.line,
          a.side,
          a.is_live,
          a.audited_at,
          m.home_score,
          m.away_score,
          m.status AS match_status,
          m.kickoff_time
        FROM betslip_audit_results a
        LEFT JOIN matches m ON m.external_id = a.event_id
        WHERE a.id = ANY(:ids)
        """
    )
    out: Dict[int, Dict[str, Any]] = {}
    async with db.async_session() as session:
        r = await session.execute(q, {"ids": ids})
        for x in r.fetchall() or []:
            out[int(x._mapping["audit_id"])] = dict(x._mapping)
    return out


def _audit_is_inplay(a: Dict[str, Any]) -> bool:
    try:
        if a.get("is_live") is not None:
            return bool(a.get("is_live"))
    except Exception:
        pass
    try:
        ko = a.get("kickoff_time")
        au = a.get("audited_at")
        if isinstance(ko, str):
            ko = _parse_iso(ko)
        if isinstance(au, str):
            au = _parse_iso(au)
        if isinstance(ko, datetime) and isinstance(au, datetime):
            return bool(au >= ko)
    except Exception:
        pass
    return False


def _group_key(exec_side: str, inplay: bool) -> str:
    side = "Back" if str(exec_side).strip().lower() == "back" else "Lay"
    regime = "In" if bool(inplay) else "Pre"
    return f"{side}_{regime}"


def _empty_row() -> Dict[str, Any]:
    return {
        "n_bets": 0,
        # stake (o que é enviado na ordem / “valor apostado” em Back)
        "stake_sum": 0.0,
        "stake_avg": None,
        # amount_risk = base de risco/“capital travado” por lado:
        # - Back: stake
        # - Lay : liability
        "amount_risk_sum": 0.0,
        "amount_risk_avg": None,
        "n_settled": 0,
        "n_unsettled": 0,
        "stake_sum_settled": 0.0,
        "amount_risk_sum_settled": 0.0,
        "liability_sum": 0.0,
        "liability_sum_settled": 0.0,
        "pnl_sum_settled": 0.0,
        # ROI “principal” por lado (apenas no liquidado via placar):
        # - Back: ROI por stake
        # - Lay : ROI por liability
        "roi_pct_settled": None,
        # extras (para debug/contabilidade de denom.)
        "roi_pct_settled_per_stake": None,
        "roi_pct_settled_per_liability": None,
        # P&L real (orders) não depende de placar: útil para “aderência 100%”
        "pnl_real_sum_settled": 0.0,
    }


async def compute_minimal_by_type(
    *,
    executor_jsonl: Path,
    hours: float,
    only_status: List[str],
) -> Dict[str, Any]:
    now = _utcnow()
    since = now - timedelta(hours=float(hours))

    xs = _parse_executor_jsonl(executor_jsonl)
    xs = [e for e in xs if e.created_at >= since]
    only = {s.strip() for s in (only_status or []) if str(s).strip()}
    if only:
        xs = [e for e in xs if str(e.status).strip() in only]

    audit_ids = sorted({int(e.audit_id) for e in xs if e.audit_id is not None})
    db = Database()
    await db.connect()
    try:
        audit_map = await _fetch_audits_for_ids(db, audit_ids)
    finally:
        try:
            await db.close()
        except Exception:
            pass

    by_type: Dict[str, Dict[str, Any]] = {k: _empty_row() for k in ("Back_Pre", "Back_In", "Lay_Pre", "Lay_In")}

    # Orders (P&L real) via executor /account (unix socket)
    orders_by_id: Dict[str, Dict[str, Any]] = {}
    orders_meta: Dict[str, Any] = {"enabled": False, "n_orders": 0, "error": None}
    unix_socket = os.getenv("EXECUTOR_UNIX_SOCKET", "").strip() or "/tmp/betinasia-exec.sock"
    orders_pnl = os.getenv("DAILY_EXEC_MIN_BY_TYPE_ORDERS_PNL", "1").strip() not in ("0", "false", "False", "no", "NO")
    try:
        page_size = int(os.getenv("DAILY_EXEC_MIN_BY_TYPE_ORDERS_PAGE_SIZE", "200") or 200)
    except Exception:
        page_size = 200
    if orders_pnl and unix_socket:
        try:
            conn = aiohttp.UnixConnector(path=str(unix_socket))
            async with aiohttp.ClientSession(connector=conn) as sess:
                async with sess.get(f"http://localhost/account?page_size={int(page_size)}") as resp:
                    data = await resp.json()
            pnl_blk = data.get("pnl") if isinstance(data, dict) else {}
            lst = pnl_blk.get("orders") if isinstance(pnl_blk, dict) else None
            if isinstance(lst, list):
                for o in lst:
                    if not isinstance(o, dict):
                        continue
                    oid = o.get("id") or o.get("order_id") or o.get("uuid")
                    if oid is None:
                        continue
                    orders_by_id[str(oid)] = o
                orders_meta = {"enabled": True, "n_orders": int(len(lst)), "unix_socket": str(unix_socket), "error": None}
            else:
                orders_meta = {"enabled": True, "n_orders": 0, "unix_socket": str(unix_socket), "error": "NO_ORDERS_LIST_IN_ACCOUNT_SNAPSHOT"}
        except Exception as e:
            orders_meta = {"enabled": True, "n_orders": 0, "unix_socket": str(unix_socket), "error": str(e)[:200]}

    for e in xs:
        a = audit_map.get(int(e.audit_id)) if e.audit_id is not None else None
        inplay = _audit_is_inplay(a) if isinstance(a, dict) else False
        g = _group_key(e.exec_side, bool(inplay))
        row = by_type.setdefault(g, _empty_row())

        stake = float(e.stake_sent) if e.stake_sent is not None else 0.0
        row["n_bets"] += 1
        row["stake_sum"] += float(stake)

        odd = _sanitize_decimal_odd(e.odd_final)
        is_lay = str(e.exec_side).strip().lower() == "lay"
        liab = None
        if is_lay and odd is not None:
            liab = float(stake) * max(0.0, float(odd) - 1.0)
            row["liability_sum"] += float(liab)

        # “valor apostado”/capital em risco por lado:
        # Back: stake; Lay: liability (quando possível; fallback para liability_requested)
        if not is_lay:
            row["amount_risk_sum"] += float(stake)
        else:
            if liab is not None:
                row["amount_risk_sum"] += float(liab)
            elif e.liability_req is not None:
                row["amount_risk_sum"] += float(e.liability_req)

        # settled (REAL, independente de placar): usa orders.profit_loss quando disponível
        if e.order_id and str(e.order_id) in orders_by_id:
            o = orders_by_id.get(str(e.order_id)) or {}
            # profit_loss pode vir no nível do order ou dentro de bets[]
            pl = _safe_float(o.get("profit_loss"))
            if pl is None:
                try:
                    bets = o.get("bets") if isinstance(o.get("bets"), list) else []
                    pls = [_safe_float(b.get("profit_loss")) for b in bets if isinstance(b, dict)]
                    pls = [x for x in pls if x is not None]
                    if pls:
                        pl = float(sum(pls))
                except Exception:
                    pl = None
            # consideramos “liquidada” apenas quando existe P&L (aderente ao accounting/UI)
            if pl is not None:
                row["n_settled"] += 1
                row["stake_sum_settled"] += float(stake)
                if not is_lay:
                    row["amount_risk_sum_settled"] += float(stake)
                else:
                    liab2 = None
                    if odd is not None:
                        liab2 = float(stake) * max(0.0, float(odd) - 1.0)
                    elif e.liability_req is not None:
                        liab2 = float(e.liability_req)
                    else:
                        liab2 = 0.0
                    row["liability_sum_settled"] += float(liab2)
                    row["amount_risk_sum_settled"] += float(liab2)
                row["pnl_real_sum_settled"] += float(pl)

    # post-process
    for k, r in by_type.items():
        n = int(r.get("n_bets") or 0)
        n_set = int(r.get("n_settled") or 0)
        r["n_unsettled"] = int(max(0, n - n_set))
        r["stake_avg"] = (float(r["stake_sum"]) / float(n)) if n > 0 else None
        r["amount_risk_avg"] = (float(r["amount_risk_sum"]) / float(n)) if n > 0 else None
        # ROI real por base de risco (Back: stake; Lay: liability), usando P&L real quando possível
        st_cov = float(r.get("amount_risk_sum_settled") or 0.0)
        if st_cov > 0:
            r["roi_pct_settled"] = float(r.get("pnl_real_sum_settled") or 0.0) / st_cov * 100.0

    total = _empty_row()
    for r in by_type.values():
        total["n_bets"] += int(r.get("n_bets") or 0)
        total["stake_sum"] += float(r.get("stake_sum") or 0.0)
        total["amount_risk_sum"] += float(r.get("amount_risk_sum") or 0.0)
        total["n_settled"] += int(r.get("n_settled") or 0)
        total["stake_sum_settled"] += float(r.get("stake_sum_settled") or 0.0)
        total["amount_risk_sum_settled"] += float(r.get("amount_risk_sum_settled") or 0.0)
        total["liability_sum"] += float(r.get("liability_sum") or 0.0)
        total["liability_sum_settled"] += float(r.get("liability_sum_settled") or 0.0)
        total["pnl_real_sum_settled"] += float(r.get("pnl_real_sum_settled") or 0.0)
    total["n_unsettled"] = int(max(0, int(total["n_bets"]) - int(total["n_settled"])))
    total["stake_avg"] = (float(total["stake_sum"]) / float(total["n_bets"])) if total["n_bets"] else None
    total["amount_risk_avg"] = (float(total["amount_risk_sum"]) / float(total["n_bets"])) if total["n_bets"] else None
    if float(total["amount_risk_sum_settled"]) > 0:
        total["roi_pct_settled"] = float(total["pnl_real_sum_settled"]) / float(total["amount_risk_sum_settled"]) * 100.0

    return {
        "ts_utc": now.isoformat(),
        "since_utc": since.isoformat(),
        "until_utc": now.isoformat(),
        "hours": float(hours),
        "executor_jsonl": str(executor_jsonl),
        "only_status": sorted(list(only)),
        "by_type": by_type,
        "total": total,
        "orders_pnl": orders_meta,
        "notes": [
            "Este bloco usa o executor_jsonl e classifica Pre/In por `betslip_audit_results.is_live` (com fallback kickoff_time vs audited_at).",
            "P&L/ROI aqui são calculados via `profit_loss` real do endpoint `/account` do executor (orders), sem depender de placar.",
            "Para Lay: 'valor em risco' é a liability (stake*(odd-1)); para Back: é o stake.",
            "ROI principal (roi_pct_settled) usa a base de risco por lado: Back por stake; Lay por liability.",
        ],
    }


def main() -> int:
    # garante .env (para EXECUTOR_UNIX_SOCKET, etc.)
    _load_env_file(Path(os.getenv("ENV_FILE", ".env")))

    ap = argparse.ArgumentParser(description="Métricas mínimas de execução por tipo (Back/Lay × Pre/In).")
    ap.add_argument("--executor-jsonl", default=os.getenv("EXECUTOR_JSONL", "logs/executor_live.jsonl"))
    ap.add_argument("--hours", type=float, default=float(os.getenv("DAILY_EXEC_MIN_BY_TYPE_HOURS", "24")))
    ap.add_argument("--only-status", default=os.getenv("DAILY_EXEC_MIN_BY_TYPE_ONLY_STATUS", "LIVE_OK").strip())
    ap.add_argument("--out", default=os.getenv("DAILY_EXEC_MIN_BY_TYPE_OUT", "").strip() or None)
    args = ap.parse_args()

    jsonl = Path(str(args.executor_jsonl))
    only = [s.strip() for s in str(args.only_status).split(",") if s.strip()]

    import asyncio

    rep = asyncio.run(compute_minimal_by_type(executor_jsonl=jsonl, hours=float(args.hours), only_status=only))
    if args.out:
        p = Path(str(args.out))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

