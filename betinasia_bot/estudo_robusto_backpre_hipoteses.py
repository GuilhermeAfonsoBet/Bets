#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estudo robusto da estrategia Back Pre com P&L real reconciliado.

Objetivo: testar hipoteses pre-definidas (sem otimizar retroativamente) sobre
slippage, faixa de odd, liquidez/capacidade e filtro conservador de ligas.

Unidade estatistica principal: event_id.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


WORLD_CUP_ALIASES = {
    "worldcup",
    "fifaworldcup",
    "clubworldcup",
    "fifaclubworldcup",
    "worldchampionship",
    "copadomundo",
    "mundialdeclubes",
}


def _norm(s: Any) -> str:
    return "".join(ch for ch in str(s or "").lower() if ch.isalnum())


def _pf(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip().replace(" ", "")
    if not s:
        return None
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        v = float(s)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _pdt(x: Any) -> Optional[datetime]:
    s = str(x or "").strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        d = datetime.fromisoformat(s)
    except Exception:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc)


def _fmt(x: Optional[float], nd: int = 2, pct: bool = False) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "NA"
    return f"{float(x):.{nd}f}" + ("%" if pct else "")


def _get(obj: Any, path: Sequence[str]) -> Any:
    cur = obj
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _run(cmd: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(list(cmd), capture_output=True, text=True, check=False)


def _read_env_db(path: Path) -> str:
    if not path.exists():
        return ""
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("export "):
            s = s[len("export ") :].strip()
        if s.startswith("DATABASE_URL="):
            v = s.split("=", 1)[1].strip()
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                v = v[1:-1]
            return v
    return ""


def _resolve_db(cli: str) -> str:
    if cli:
        return cli
    if os.getenv("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    for p in [Path.cwd() / "betinasia_bot/.env", Path("/home/betbot/Bets/betinasia_bot/.env")]:
        db = _read_env_db(p)
        if db:
            return db
    return ""


def _latest_balance() -> Path:
    cands = sorted(glob.glob("/home/betbot/Bets/betinasia_bot/logs/accounting/*balance*.csv"), key=os.path.getmtime)
    if not cands:
        raise RuntimeError("balance CSV nao encontrado")
    return Path(cands[-1])


def _pick(fields: Sequence[str], cands: Sequence[str]) -> Optional[str]:
    by = {_norm(c): c for c in fields}
    for c in cands:
        hit = by.get(_norm(c))
        if hit:
            return hit
    return None


def _load_balance(balance_csv: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with balance_csv.open(encoding="utf-8", errors="ignore", newline="") as f:
        rd = csv.DictReader(f)
        fields = list(rd.fieldnames or [])
        c_order = _pick(fields, ["order id", "order_id", "orderid", "bet id", "ticket id", "ticket_id"])
        c_pnl = _pick(fields, ["amount", "pnl_real", "pnl", "profit", "pl", "result", "net"])
        c_post = _pick(fields, ["post date", "post_date", "created_at", "created at", "timestamp"])
        if not c_order or not c_pnl:
            raise RuntimeError(f"balance sem order/pnl: order={c_order} pnl={c_pnl}")
        for r in rd:
            oid = str(r.get(c_order, "")).strip()
            pnl = _pf(r.get(c_pnl))
            if not oid or pnl is None:
                continue
            rec = out.setdefault(oid, {"order_id": oid, "pnl": 0.0, "post_date": ""})
            rec["pnl"] += float(pnl)
            if c_post and not rec.get("post_date"):
                rec["post_date"] = str(r.get(c_post, "")).strip()
    return out


def _extract_order_ids(obj: Any, out: set[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            kn = _norm(k)
            if kn in {"orderid", "order_id", "ticketid", "betid"}:
                sv = str(v or "").strip()
                if sv:
                    out.add(sv)
            _extract_order_ids(v, out)
    elif isinstance(obj, list):
        for x in obj:
            _extract_order_ids(x, out)


def _audit_meta(db: str, start: str, end: str) -> Dict[str, Dict[str, Any]]:
    if not db:
        return {}
    sql = f"""
    SELECT id::text,
           COALESCE(event_id::text,''),
           COALESCE(league,'')::text,
           audited_at::text,
           COALESCE(hypothesis_type,'')::text,
           COALESCE(reversal_direction,'')::text
    FROM betslip_audit_results
    WHERE audited_at >= '{start}'::timestamptz
      AND audited_at <= '{end} 23:59:59+00'::timestamptz;
    """
    p = _run(["psql", db, "-At", "-v", "ON_ERROR_STOP=1", "-c", sql])
    if p.returncode != 0:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for ln in p.stdout.splitlines():
        parts = ln.split("|")
        if len(parts) < 6:
            continue
        out[parts[0]] = {
            "event_id": parts[1],
            "league": parts[2],
            "audited_at": parts[3],
            "hypothesis_type": parts[4],
            "reversal_direction": parts[5],
        }
    return out


def _load_rows(jsonl: Path, balance_csv: Path, db: str, start: str, end: str) -> List[Dict[str, Any]]:
    bal = _load_balance(balance_csv)
    meta = _audit_meta(db, start, end)
    rows: List[Dict[str, Any]] = []
    with jsonl.open(encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if "2026-" not in ln:
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            req = obj.get("request") if isinstance(obj.get("request"), dict) else {}
            res = obj.get("result") if isinstance(obj.get("result"), dict) else {}
            if str(res.get("status") or "").upper() != "LIVE_OK":
                continue
            if str(res.get("exec_side") or req.get("exec_side") or "").lower() != "back":
                continue
            raw = res.get("raw") if isinstance(res.get("raw"), dict) else {}
            vs = raw.get("value_sizing") if isinstance(raw.get("value_sizing"), dict) else {}
            regime = str(vs.get("market_regime") or _get(req, ["meta", "market", "regime"]) or "").lower()
            market_live = vs.get("market_is_live")
            if not (regime == "pre" or market_live is False):
                continue
            created = str(res.get("created_at") or req.get("created_at") or "")
            day = created[:10]
            if day < start or day > end:
                continue
            order_ids: set[str] = set()
            _extract_order_ids(obj, order_ids)
            order_id = ""
            for oid in sorted(order_ids):
                if oid in bal:
                    order_id = oid
                    break
            if not order_id:
                continue
            pnl = float(bal[order_id]["pnl"])
            post_date = str(bal[order_id].get("post_date") or "")
            aid = str(res.get("audit_id") or req.get("audit_id") or "").strip()
            m = meta.get(aid, {})
            event_id = str(res.get("event_id") or req.get("event_id") or m.get("event_id") or aid or order_id)
            league = str(m.get("league") or _get(req, ["meta", "bridge", "league"]) or "")
            stake = _pf(vs.get("stake_chosen")) or _pf(_get(res, ["policy", "stake_requested"])) or _pf(_get(req, ["policy", "stake_requested"]))
            if stake is None or stake <= 0:
                continue
            odd = _pf(res.get("odd_at_decision") or req.get("odd_at_decision"))
            odd_final = _pf(res.get("odd_final") or vs.get("odd_pre_submit"))
            slip = _pf(vs.get("slippage_pre_pct"))
            line = _pf(res.get("line") or req.get("line"))
            limit_final = _pf(res.get("limit_final"))
            max_stake = _pf(_get(vs, ["params", "stake_pre_fast"])) or _pf(_get(res, ["policy", "stake_requested"])) or float(stake)
            pre_submit_ms = _pf(vs.get("pre_submit_ms"))
            ts = _pdt(created)
            week = f"{ts.isocalendar().year}-W{ts.isocalendar().week:02d}" if ts else "unknown"
            wc = _norm(league) in WORLD_CUP_ALIASES
            rows.append(
                {
                    "order_id": order_id,
                    "event_id": event_id,
                    "audit_id": aid,
                    "created_at": created,
                    "post_date": post_date,
                    "day": day,
                    "week": week,
                    "league": league,
                    "is_world_cup": wc,
                    "stake": float(stake),
                    "pnl": pnl,
                    "odd": odd,
                    "odd_final": odd_final,
                    "slippage_pre_pct": slip,
                    "line": line,
                    "limit_final": limit_final,
                    "max_stake": float(max_stake),
                    "capacity": limit_final if limit_final is not None else float(max_stake),
                    "pre_submit_ms": pre_submit_ms,
                    "status": "LIVE_OK",
                    "market_type": str(res.get("market_type") or req.get("market_type") or ""),
                }
            )
    rows.sort(key=lambda r: (r["created_at"], r["order_id"]))
    return rows


def _roi(rows: Sequence[Dict[str, Any]]) -> Tuple[float, float, float]:
    stake = sum(float(r["stake"]) for r in rows)
    pnl = sum(float(r["pnl"]) for r in rows)
    return (100.0 * pnl / stake if stake > 0 else float("nan"), pnl, stake)


def _events(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        k = str(r["event_id"])
        e = by.get(k)
        if e is None:
            e = {"event_id": k, "day": r["day"], "week": r["week"], "league": r["league"], "is_world_cup": r["is_world_cup"], "stake": 0.0, "pnl": 0.0, "n": 0}
            by[k] = e
        e["stake"] += float(r["stake"])
        e["pnl"] += float(r["pnl"])
        e["n"] += 1
    out = list(by.values())
    for e in out:
        e["roi"] = 100.0 * e["pnl"] / e["stake"] if e["stake"] > 0 else None
    return out


def _roi_without_topk(events: Sequence[Dict[str, Any]], k: int) -> Optional[float]:
    kept = sorted(events, key=lambda e: e["pnl"], reverse=True)[k:]
    if not kept:
        return None
    st = sum(e["stake"] for e in kept)
    return 100.0 * sum(e["pnl"] for e in kept) / st if st > 0 else None


def _drawdown(seq: Sequence[Tuple[str, float]]) -> Dict[str, Any]:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    tuw = 0
    cur_tuw = 0
    trough = 0.0
    total_pnl = sum(v for _, v in seq)
    for _, pnl in seq:
        equity += float(pnl)
        if equity >= peak:
            peak = equity
            cur_tuw = 0
        else:
            cur_tuw += 1
            tuw = max(tuw, cur_tuw)
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
            trough = equity
    return {"max_drawdown": max_dd, "time_under_water_units": tuw, "recovery_factor": (total_pnl / max_dd if max_dd > 0 else None), "ending_pnl": total_pnl, "trough_equity": trough}


def _bootstrap_events(events: Sequence[Dict[str, Any]], iters: int, seed: int) -> Dict[str, Any]:
    if not events:
        return {"ci90": [None, None], "ci95": [None, None], "prob_roi_gt_0": None}
    rng = random.Random(seed)
    vals = []
    n = len(events)
    for _ in range(iters):
        smp = [events[rng.randrange(n)] for _ in range(n)]
        st = sum(e["stake"] for e in smp)
        vals.append(100.0 * sum(e["pnl"] for e in smp) / st if st > 0 else 0.0)
    vals.sort()
    return {
        "ci90": [vals[int(0.05 * (len(vals) - 1))], vals[int(0.95 * (len(vals) - 1))]],
        "ci95": [vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]],
        "prob_roi_gt_0": sum(1 for x in vals if x > 0) / len(vals),
    }


def _perm_p(events: Sequence[Dict[str, Any]], iters: int, seed: int) -> Optional[float]:
    if not events:
        return None
    obs = 100.0 * sum(e["pnl"] for e in events) / sum(e["stake"] for e in events)
    rng = random.Random(seed)
    ge = 0
    for _ in range(iters):
        pnl = 0.0
        stake = 0.0
        for e in events:
            pnl += (1 if rng.random() >= 0.5 else -1) * e["pnl"]
            stake += e["stake"]
        stat = 100.0 * pnl / stake if stake > 0 else 0.0
        if stat >= obs:
            ge += 1
    return (ge + 1) / (iters + 1)


def _diff_test(a_ev: Sequence[Dict[str, Any]], b_ev: Sequence[Dict[str, Any]], iters: int, seed: int) -> Dict[str, Any]:
    if not a_ev or not b_ev:
        return {"diff_roi_pp": None, "ci95": [None, None], "p": None}
    def eroi(ev):
        return 100.0 * sum(e["pnl"] for e in ev) / sum(e["stake"] for e in ev)
    obs = eroi(a_ev) - eroi(b_ev)
    rng = random.Random(seed)
    vals = []
    for _ in range(iters):
        aa = [a_ev[rng.randrange(len(a_ev))] for _ in range(len(a_ev))]
        bb = [b_ev[rng.randrange(len(b_ev))] for _ in range(len(b_ev))]
        vals.append(eroi(aa) - eroi(bb))
    vals.sort()
    # permutation on event labels, one-sided A better
    all_ev = list(a_ev) + list(b_ev)
    na = len(a_ev)
    ge = 0
    for _ in range(iters):
        rng.shuffle(all_ev)
        stat = eroi(all_ev[:na]) - eroi(all_ev[na:])
        if stat >= obs:
            ge += 1
    return {"diff_roi_pp": obs, "ci95": [vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]], "p": (ge + 1) / (iters + 1)}


def _bh(pairs: List[Tuple[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    vals = [(name, p) for name, p in pairs if p is not None and not math.isnan(p)]
    vals.sort(key=lambda x: x[1])
    m = len(vals)
    out: Dict[str, Optional[float]] = {name: None for name, _ in pairs}
    prev = 1.0
    for i in range(m, 0, -1):
        name, p = vals[i - 1]
        q = min(prev, p * m / i)
        out[name] = q
        prev = q
    return out


def _weekly(rows: Sequence[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    by = defaultdict(list)
    for r in rows:
        by[r["week"]].append(r)
    rois = [_roi(v)[0] for v in by.values() if v]
    if not rois:
        return None, None
    xs = sorted(rois)
    med = xs[len(xs) // 2] if len(xs) % 2 else 0.5 * (xs[len(xs) // 2 - 1] + xs[len(xs) // 2])
    return 100.0 * sum(1 for x in rois if x > 0) / len(rois), med


def _m4(events: Sequence[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    ev = sorted(events, key=lambda e: (e["day"], e["event_id"]))
    n = len(ev)
    if not n:
        return None, None, None
    a, b = n // 3, (2 * n) // 3
    out = []
    for part in [ev[:a], ev[a:b], ev[b:]]:
        if not part:
            out.append(None)
        else:
            st = sum(e["stake"] for e in part)
            out.append(100.0 * sum(e["pnl"] for e in part) / st if st > 0 else None)
    return out[0], out[1], out[2]


def _five_ms(rows: Sequence[Dict[str, Any]], boot: Dict[str, Any], p_perm: Optional[float]) -> Dict[str, Any]:
    ev = _events(rows)
    roi, pnl, stake = _roi(rows)
    roi_no_top3 = _roi_without_topk(ev, 3)
    total_abs = sum(abs(e["pnl"]) for e in ev)
    top1_abs = 100.0 * max([abs(e["pnl"]) for e in ev] or [0]) / total_abs if total_abs > 0 else 0
    pos_ratio, med_week = _weekly(rows)
    r1, r2, r3 = _m4(ev)
    ev_bet = pnl / len(rows) if rows else None
    m1 = p_perm is not None and p_perm <= 0.10 and boot["ci90"][0] is not None and boot["ci90"][0] > 0
    m2 = roi_no_top3 is not None and roi_no_top3 > 0 and top1_abs <= 35
    m3 = pos_ratio is not None and pos_ratio >= 55 and med_week is not None and med_week > 0
    m4 = (((r1 is not None and r1 > 0) + (r2 is not None and r2 > 0) + (r3 is not None and r3 > 0)) >= 2 and (r3 is not None and r3 > 0))
    m5 = ev_bet is not None and ev_bet > 0 and roi > 0
    score = sum([m1, m2, m3, m4, m5])
    return {
        "M1": {"roi": roi, "p_perm": p_perm, "ci90_lo": boot["ci90"][0], "ci90_hi": boot["ci90"][1], "status": "OK" if m1 else "FAIL"},
        "M2": {"roi_sem_top3": roi_no_top3, "top1_abs_pct": top1_abs, "status": "OK" if m2 else "FAIL"},
        "M3": {"pos_ratio_pct": pos_ratio, "mediana_semanal_pct": med_week, "status": "OK" if m3 else "FAIL"},
        "M4": {"r1_pct": r1, "r2_pct": r2, "r3_pct": r3, "status": "OK" if m4 else "FAIL"},
        "M5": {"ev_por_aposta": ev_bet, "ev_pct": roi, "status": "OK" if m5 else "FAIL"},
        "score": score,
        "label": "robusto" if score >= 4 else ("moderado" if score >= 2 else "fragil"),
    }


def _estimate_wc_start(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_day = defaultdict(lambda: {"n": 0, "wc_n": 0, "stake": 0.0, "wc_stake": 0.0})
    for r in rows:
        d = by_day[r["day"]]
        d["n"] += 1
        d["stake"] += r["stake"]
        if r["is_world_cup"]:
            d["wc_n"] += 1
            d["wc_stake"] += r["stake"]
    days = sorted(by_day)
    out = {}
    for thr in [0.4, 0.5, 0.6]:
        hit = None
        for i, day in enumerate(days):
            win = days[max(0, i - 6) : i + 1]
            n = sum(by_day[d]["n"] for d in win)
            wn = sum(by_day[d]["wc_n"] for d in win)
            st = sum(by_day[d]["stake"] for d in win)
            wst = sum(by_day[d]["wc_stake"] for d in win)
            share_n = wn / n if n else 0
            share_st = wst / st if st else 0
            if share_n >= thr or share_st >= thr:
                hit = {"date": day, "share_n": share_n, "share_stake": share_st, "threshold": thr}
                break
        out[str(thr)] = hit
    return out


def _rule_filters(train_rows: Sequence[Dict[str, Any]]) -> set[str]:
    # Blacklist conservadora derivada apenas em treino:
    # liga com >= 8 eventos, ROI<0 e ROI sem Top-3<0.
    by = defaultdict(list)
    for r in train_rows:
        by[r["league"] or "NA"].append(r)
    bad = set()
    for league, rs in by.items():
        ev = _events(rs)
        if len(ev) < 8:
            continue
        roi = _roi(rs)[0]
        no3 = _roi_without_topk(ev, 3)
        if roi < 0 and (no3 is None or no3 < 0):
            bad.add(league)
    return bad


def _apply_rule(rows: Sequence[Dict[str, Any]], rule: str, blacklist: set[str]) -> List[Dict[str, Any]]:
    def cap_ok(r):
        c = r.get("capacity")
        return c is not None and float(c) > 100

    out = []
    for r in rows:
        odd = r.get("odd")
        slip = r.get("slippage_pre_pct")
        if rule == "R0":
            ok = True
        elif rule in {"R1", "H1"}:
            ok = slip is not None and slip < 0
        elif rule == "R2":
            ok = odd is not None and 1.9 <= odd <= 2.1
        elif rule == "R3":
            ok = cap_ok(r)
        elif rule == "R4":
            ok = (r.get("league") or "NA") not in blacklist
        elif rule == "H2":
            ok = slip is not None and slip < 0 and odd is not None and 1.9 <= odd <= 2.1
        elif rule == "H3":
            ok = slip is not None and slip < 0 and odd is not None and 1.9 <= odd <= 2.1 and cap_ok(r)
        elif rule == "H4":
            ok = slip is not None and slip < 0 and odd is not None and 1.9 <= odd <= 2.1 and cap_ok(r) and (r.get("league") or "NA") not in blacklist
        elif rule == "H5":
            ok = slip is not None and -0.5 <= slip < 0 and odd is not None and 1.9 <= odd <= 2.1 and cap_ok(r)
        else:
            ok = False
        if ok:
            out.append(r)
    return out


def _rule_stats(name: str, rows: Sequence[Dict[str, Any]], iters: int, seed: int) -> Dict[str, Any]:
    ev = _events(rows)
    roi, pnl, stake = _roi(rows)
    boot = _bootstrap_events(ev, iters, seed)
    p_perm = _perm_p(ev, iters, seed + 1)
    ev_pnls = sorted([e["pnl"] for e in ev])
    bet_pnls = sorted([r["pnl"] for r in rows])
    pos_bets = 100.0 * sum(1 for r in rows if r["pnl"] > 0) / len(rows) if rows else None
    pos_events = 100.0 * sum(1 for e in ev if e["pnl"] > 0) / len(ev) if ev else None
    top_sum = sum(e["pnl"] for e in sorted(ev, key=lambda e: e["pnl"], reverse=True)[:1])
    top3_sum = sum(e["pnl"] for e in sorted(ev, key=lambda e: e["pnl"], reverse=True)[:3])
    top1_pct = 100.0 * top_sum / pnl if pnl else None
    top3_pct = 100.0 * top3_sum / pnl if pnl else None
    dd_bet = _drawdown([(r["created_at"], r["pnl"]) for r in rows])
    dd_event = _drawdown([(e["day"], e["pnl"]) for e in sorted(ev, key=lambda e: (e["day"], e["event_id"]))])
    by_day = defaultdict(float)
    for r in rows:
        by_day[r["day"]] += r["pnl"]
    dd_day = _drawdown(sorted(by_day.items()))
    fm = _five_ms(rows, boot, p_perm)
    return {
        "name": name,
        "n_bets": len(rows),
        "n_events": len(ev),
        "stake": stake,
        "pnl": pnl,
        "roi": roi,
        "pnl_mean_bet": (sum(bet_pnls) / len(bet_pnls) if bet_pnls else None),
        "pnl_median_bet": (statistics.median(bet_pnls) if bet_pnls else None),
        "pnl_mean_event": (sum(ev_pnls) / len(ev_pnls) if ev_pnls else None),
        "pnl_median_event": (statistics.median(ev_pnls) if ev_pnls else None),
        "pct_bets_positive": pos_bets,
        "pct_events_positive": pos_events,
        "roi_without_top": {str(k): _roi_without_topk(ev, k) for k in [1, 3, 5, 10]},
        "top1_pct_of_total_pnl": top1_pct,
        "top3_pct_of_total_pnl": top3_pct,
        "drawdown_bet": dd_bet,
        "drawdown_event": dd_event,
        "drawdown_day": dd_day,
        "bootstrap": boot,
        "p_perm": p_perm,
        "five_ms": fm,
    }


def _decision(s: Dict[str, Any]) -> str:
    roi = s["roi"]
    no3 = s["roi_without_top"].get("3")
    prob = s["bootstrap"].get("prob_roi_gt_0")
    ci90_lo = s["bootstrap"].get("ci90", [None, None])[0]
    m2 = s["five_ms"]["M2"]["status"] == "OK"
    if roi <= 0 or (no3 is not None and no3 < -5) or (prob is not None and prob < 0.55):
        return "A. Rejeitada"
    if roi > 0 and (ci90_lo is None or ci90_lo < 0) and (no3 is None or no3 < 0):
        return "B. Explorável"
    if roi > 0 and (no3 is not None and no3 >= -0.5) and (prob is not None and prob > 0.70):
        return "C. Validação exploratória forte"
    if roi > 0 and (no3 is not None and no3 > 0) and ci90_lo is not None and ci90_lo >= -1:
        return "D. Validação gradual"
    if roi > 0 and m2 and ci90_lo is not None and ci90_lo > 0:
        return "E. Candidata a escala"
    return "B. Explorável"


def _odd_bucket(x: Optional[float]) -> str:
    if x is None:
        return "NA"
    if x < 1.7:
        return "<1.7"
    if x < 1.9:
        return "1.7-1.9"
    if x <= 2.1:
        return "1.9-2.1"
    if x <= 2.4:
        return "2.1-2.4"
    return ">2.4"


def _odd_analysis(rows: Sequence[Dict[str, Any]], iters: int) -> Dict[str, Any]:
    by = defaultdict(list)
    for r in rows:
        by[_odd_bucket(r.get("odd"))].append(r)
    out = {}
    for k in ["<1.7", "1.7-1.9", "1.9-2.1", "2.1-2.4", ">2.4", "NA"]:
        if by.get(k):
            out[k] = _rule_stats(f"odd_{k}", by[k], iters, 300 + len(out))
    # cortes condicionais da faixa 1.9-2.1
    bucket = [r for r in rows if _odd_bucket(r.get("odd")) == "1.9-2.1"]
    out["_conditional_1.9-2.1"] = {
        "slip_neg": _rule_stats("odd_1.9_2.1__slip_neg", [r for r in bucket if r.get("slippage_pre_pct") is not None and r["slippage_pre_pct"] < 0], iters, 401),
        "not_slip_neg": _rule_stats("odd_1.9_2.1__not_slip_neg", [r for r in bucket if r.get("slippage_pre_pct") is None or r["slippage_pre_pct"] >= 0], iters, 402),
        "world_cup": _rule_stats("odd_1.9_2.1__world_cup", [r for r in bucket if r["is_world_cup"]], iters, 403),
        "non_world_cup": _rule_stats("odd_1.9_2.1__non_world_cup", [r for r in bucket if not r["is_world_cup"]], iters, 404),
        "capacity_gt_100": _rule_stats("odd_1.9_2.1__capacity_gt_100", [r for r in bucket if r.get("capacity") is not None and r["capacity"] > 100], iters, 405),
    }
    return out


def _temporal_split(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    days = sorted({r["day"] for r in rows})
    if not days:
        return [], [], ""
    cut = days[int(len(days) * 0.6)]
    return [r for r in rows if r["day"] < cut], [r for r in rows if r["day"] >= cut], cut


def _walk_forward(rows: Sequence[Dict[str, Any]], rules: Sequence[str], blacklist: set[str], train_days: int, test_days: int) -> Dict[str, Any]:
    days = sorted({r["day"] for r in rows})
    out: Dict[str, Any] = {}
    if len(days) < train_days + test_days:
        return out
    starts = range(0, len(days) - train_days - test_days + 1, test_days)
    for rule in rules:
        vals = []
        for s in starts:
            test_set = set(days[s + train_days : s + train_days + test_days])
            te = [r for r in rows if r["day"] in test_set]
            rr = _apply_rule(te, rule, blacklist)
            if rr:
                vals.append({"start": min(test_set), "end": max(test_set), "roi": _roi(rr)[0], "n": len(rr), "pnl": _roi(rr)[1], "stake": _roi(rr)[2]})
        rois = [v["roi"] for v in vals]
        out[rule] = {
            "windows": vals,
            "n_windows": len(vals),
            "roi_mean": statistics.mean(rois) if rois else None,
            "roi_median": statistics.median(rois) if rois else None,
            "pct_windows_positive": 100.0 * sum(1 for x in rois if x > 0) / len(rois) if rois else None,
        }
    return out


def _render_md(out: Dict[str, Any], md: Path) -> None:
    lines: List[str] = []
    lines.append("# Estudo robusto Back Pre - hipoteses slippage/odd/liquidez/ligas\n")
    lines.append("## Sumario executivo\n")
    lines.append("- Estudo usa execucoes Back Pre LIVE_OK, P&L accounting real por order_id e event_id como cluster estatistico.")
    lines.append("- Conclusao conservadora: resultados positivos existem em alguns recortes, mas a tese ainda deve ser tratada como exploratoria/validacao gradual, nao como candidata a escala.")
    lines.append("- World Cup foi separado como regime proprio; nao-World Cup foi separado antes/durante regime World Cup estimado.\n")
    lines.append("## Regime World Cup estimado\n")
    for thr, rec in out["world_cup_regime"].items():
        lines.append(f"- threshold {thr}: {rec}")
    lines.append("\n## Blacklist conservadora derivada apenas no treino\n")
    lines.append(", ".join(sorted(out["train_blacklist"])) or "Nenhuma")
    lines.append("\n## Comparacao R0-R4/H1-H5\n")
    lines.append("| Regra | N apostas | N eventos | Stake | P&L | ROI | ROI sem Top-3 | CI90 | Prob ROI>0 | p_perm | Decisao |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|")
    for name in out["rule_order"]:
        s = out["rules"][name]
        lines.append(
            f"| {name} | {s['n_bets']} | {s['n_events']} | {_fmt(s['stake'],2)} | {_fmt(s['pnl'],2)} | {_fmt(s['roi'],2,True)} | "
            f"{_fmt(s['roi_without_top']['3'],2,True)} | [{_fmt(s['bootstrap']['ci90'][0],2,True)}, {_fmt(s['bootstrap']['ci90'][1],2,True)}] | "
            f"{_fmt(100*s['bootstrap']['prob_roi_gt_0'] if s['bootstrap']['prob_roi_gt_0'] is not None else None,1,True)} | {_fmt(s['p_perm'],4)} | {s['decision']} |"
        )
    lines.append("\n## Diferencas contra R0 e R1 com FDR BH\n")
    lines.append("| Regra | vs | diff ROI p.p. | CI95 diff | p | q(BH) |")
    lines.append("|---|---|---:|---|---:|---:|")
    for d in out["diff_tests"]:
        lines.append(f"| {d['rule']} | {d['vs']} | {_fmt(d['diff_roi_pp'],2)} | [{_fmt(d['ci95'][0],2)}, {_fmt(d['ci95'][1],2)}] | {_fmt(d['p'],4)} | {_fmt(d['q'],4)} |")
    lines.append("\n## 5Ms por regra\n")
    lines.append("| Regra | Score | M1 | M2 | M3 | M4 | M5 |")
    lines.append("|---|---:|---|---|---|---|---|")
    for name in out["rule_order"]:
        m = out["rules"][name]["five_ms"]
        lines.append(f"| {name} | {m['score']}/5 | {m['M1']['status']} | {m['M2']['status']} | {m['M3']['status']} | {m['M4']['status']} | {m['M5']['status']} |")
    lines.append("\n## ROI sem Top-k\n")
    lines.append("| Regra | Top-1 | Top-3 | Top-5 | Top-10 | Top1 % P&L | Top3 % P&L |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name in out["rule_order"]:
        s = out["rules"][name]
        lines.append(f"| {name} | {_fmt(s['roi_without_top']['1'],2,True)} | {_fmt(s['roi_without_top']['3'],2,True)} | {_fmt(s['roi_without_top']['5'],2,True)} | {_fmt(s['roi_without_top']['10'],2,True)} | {_fmt(s['top1_pct_of_total_pnl'],1,True)} | {_fmt(s['top3_pct_of_total_pnl'],1,True)} |")
    lines.append("\n## Validacao OOS split temporal fixo\n")
    lines.append(f"- Cut temporal treino/teste: {out['oos_split']['cut_day']}")
    lines.append("| Regra | Treino ROI | Teste N | Teste ROI | Teste P&L |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, rec in out["oos_split"]["rules"].items():
        lines.append(f"| {name} | {_fmt(rec['train_roi'],2,True)} | {rec['test_n']} | {_fmt(rec['test_roi'],2,True)} | {_fmt(rec['test_pnl'],2)} |")
    lines.append("\n## Walk-forward\n")
    lines.append("| Regra | Janelas | ROI medio | ROI mediano | % janelas positivas |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, rec in out["walk_forward"].items():
        lines.append(f"| {name} | {rec['n_windows']} | {_fmt(rec['roi_mean'],2,True)} | {_fmt(rec['roi_median'],2,True)} | {_fmt(rec['pct_windows_positive'],1,True)} |")
    lines.append("\n## Odd isolada\n")
    lines.append("| Bucket odd | N | Eventos | ROI | ROI sem Top-3 | CI90 | p_perm | Score 5Ms |")
    lines.append("|---|---:|---:|---:|---:|---|---:|---:|")
    for k, s in out["odd_analysis"].items():
        if k.startswith("_"):
            continue
        lines.append(f"| {k} | {s['n_bets']} | {s['n_events']} | {_fmt(s['roi'],2,True)} | {_fmt(s['roi_without_top']['3'],2,True)} | [{_fmt(s['bootstrap']['ci90'][0],2,True)}, {_fmt(s['bootstrap']['ci90'][1],2,True)}] | {_fmt(s['p_perm'],4)} | {s['five_ms']['score']}/5 |")
    lines.append("\n### Faixa 1.9-2.1 condicionais\n")
    lines.append("| Condicao | N | ROI | ROI sem Top-3 |")
    lines.append("|---|---:|---:|---:|")
    for k, s in out["odd_analysis"]["_conditional_1.9-2.1"].items():
        lines.append(f"| {k} | {s['n_bets']} | {_fmt(s['roi'],2,True)} | {_fmt(s['roi_without_top']['3'],2,True)} |")
    lines.append("\n## Regimes A/B/C\n")
    lines.append("| Regime | N | Eventos | ROI | P&L | Stake |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for k, s in out["regime_abc"].items():
        lines.append(f"| {k} | {s['n_bets']} | {s['n_events']} | {_fmt(s['roi'],2,True)} | {_fmt(s['pnl'],2)} | {_fmt(s['stake'],2)} |")
    lines.append("\n## Interpretacao economica\n")
    lines.append("- `slippage < 0` e filtro de protecao de preco: evita pagar pior que a odd de decisao.")
    lines.append("- Odd 1.9-2.1 e economicamente plausivel por ficar perto de 50/50; pequenas ineficiencias podem ter impacto relevante, mas precisa sobreviver a Top-k e OOS.")
    lines.append("- Liquidez/capacidade >100 busca reduzir mercados ruidosos; ainda assim pode excluir alguns eventos vencedores e deve ser validado OOS.")
    lines.append("- Ligas sao proxy de qualidade/cobertura/liquidez; blacklist foi derivada apenas em treino para reduzir overfitting.")
    lines.append("- World Cup e regime separado: nao deve ser usado automaticamente como evidencia estrutural da estrategia.\n")
    lines.append("## Conclusao conservadora\n")
    lines.append(out["conclusion"])
    md.write_text("\n".join(lines), encoding="utf-8")


def _render_pdf(md: Path, pdf: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=6.2, leading=7.2))
    styles.add(ParagraphStyle(name="BodyX", parent=styles["BodyText"], fontSize=8.4, leading=10.2))
    doc = SimpleDocTemplate(str(pdf), pagesize=landscape(A4), leftMargin=20, rightMargin=20, topMargin=20, bottomMargin=20)
    story = []
    def esc(s): return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    lines = md.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if not ln:
            story.append(Spacer(1, 4)); i += 1; continue
        if ln.startswith("# "):
            story.append(Paragraph(esc(ln[2:]), styles["Title"])); i += 1; continue
        if ln.startswith("## "):
            story.append(Paragraph(esc(ln[3:]), styles["Heading2"])); i += 1; continue
        if ln.startswith("### "):
            story.append(Paragraph(esc(ln[4:]), styles["Heading3"])); i += 1; continue
        if ln.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip()); i += 1
            parsed = [[c.strip() for c in row.strip("|").split("|")] for row in block]
            if len(parsed) >= 2:
                data = [[Paragraph(esc(c), styles["Small"]) for c in parsed[0]]]
                data += [[Paragraph(esc(c), styles["Small"]) for c in r] for r in parsed[2:]]
                tbl = Table(data, repeatRows=1)
                tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#111827")),
                    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                    ("GRID", (0,0), (-1,-1), 0.2, colors.HexColor("#d1d5db")),
                    ("VALIGN", (0,0), (-1,-1), "TOP"),
                    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f9fafb")]),
                    ("LEFTPADDING", (0,0), (-1,-1), 2),
                    ("RIGHTPADDING", (0,0), (-1,-1), 2),
                    ("TOPPADDING", (0,0), (-1,-1), 1.2),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 1.2),
                ]))
                story.append(tbl); story.append(Spacer(1, 5))
            continue
        story.append(Paragraph(esc(ln[2:] if ln.startswith("- ") else ln), styles["BodyX"]))
        i += 1
    doc.build(story)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-day", default="2026-04-19")
    ap.add_argument("--end-day", default="")
    ap.add_argument("--database-url", default="")
    ap.add_argument("--balance-csv", default="")
    ap.add_argument("--executor-jsonl", default="/home/betbot/Bets/betinasia_bot/logs/executor_live.jsonl")
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-pdf", required=True)
    args = ap.parse_args()
    end_day = args.end_day or datetime.now(timezone.utc).date().isoformat()
    db = _resolve_db(args.database_url)
    bal = Path(args.balance_csv) if args.balance_csv else _latest_balance()
    rows = _load_rows(Path(args.executor_jsonl), bal, db, args.start_day, end_day)
    if not rows:
        raise SystemExit("sem linhas reconciliadas")
    wc_reg = _estimate_wc_start(rows)
    wc_start = (wc_reg.get("0.5") or wc_reg.get("0.4") or {}).get("date") or "2026-06-01"
    train, test, cut = _temporal_split(rows)
    blacklist = _rule_filters(train)
    rules = ["R0", "R1", "R2", "R3", "R4", "H1", "H2", "H3", "H4", "H5"]
    rule_stats = {}
    for i, rule in enumerate(rules):
        rr = _apply_rule(rows, rule, blacklist)
        s = _rule_stats(rule, rr, args.iters, 1000 + i)
        s["decision"] = _decision(s)
        rule_stats[rule] = s
    diff_tests = []
    p_pairs = []
    for rule in rules:
        if rule == "R0":
            continue
        for base in ["R0", "R1"]:
            if rule == base:
                continue
            d = _diff_test(_events(_apply_rule(rows, rule, blacklist)), _events(_apply_rule(rows, base, blacklist)), args.iters, 2000 + len(diff_tests))
            d.update({"rule": rule, "vs": base})
            diff_tests.append(d)
            p_pairs.append((f"{rule}_vs_{base}", d["p"]))
    q = _bh(p_pairs)
    for d in diff_tests:
        d["q"] = q.get(f"{d['rule']}_vs_{d['vs']}")
    # OOS split
    bl_train = _rule_filters(train)
    split_rules = {}
    for rule in rules:
        tr = _apply_rule(train, rule, bl_train)
        te = _apply_rule(test, rule, bl_train)
        split_rules[rule] = {"train_roi": _roi(tr)[0], "test_n": len(te), "test_roi": _roi(te)[0], "test_pnl": _roi(te)[1], "test_stake": _roi(te)[2]}
    # Regimes A/B/C use R1 base (slippage<0) for regime read
    r1 = _apply_rule(rows, "R1", blacklist)
    regimes = {
        "A_world_cup": [r for r in r1 if r["is_world_cup"]],
        "B_non_wc_before_wc_regime": [r for r in r1 if (not r["is_world_cup"]) and r["day"] < wc_start],
        "C_non_wc_during_wc_regime": [r for r in r1 if (not r["is_world_cup"]) and r["day"] >= wc_start],
    }
    regime_stats = {k: _rule_stats(k, v, max(1000, args.iters // 2), 4000 + i) for i, (k, v) in enumerate(regimes.items())}
    out = {
        "params": {"start_day": args.start_day, "end_day": end_day, "balance_csv": str(bal), "n_rows": len(rows), "wc_start_used": wc_start},
        "world_cup_regime": wc_reg,
        "train_blacklist": sorted(blacklist),
        "rule_order": rules,
        "rules": rule_stats,
        "diff_tests": diff_tests,
        "oos_split": {"cut_day": cut, "rules": split_rules},
        "walk_forward": _walk_forward(rows, rules, blacklist, train_days=21, test_days=7),
        "odd_analysis": _odd_analysis(rows, max(1000, args.iters // 2)),
        "regime_abc": regime_stats,
        "conclusion": "Conclusao conservadora: ha sinais economicos em recortes com slippage<0, odd 1.9-2.1 e capacidade>100, mas a evidencia deve ser classificada como exploratoria/validacao gradual no maximo, dependente de OOS e concentracao. Nao ha base para recomendar aumento de exposicao ou escala operacional neste momento.",
    }
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _render_md(out, Path(args.out_md))
    _render_pdf(Path(args.out_md), Path(args.out_pdf))
    print(f"[OK] rows={len(rows)} wc_start={wc_start}")
    print(f"[OK] JSON={args.out_json}")
    print(f"[OK] MD={args.out_md}")
    print(f"[OK] PDF={args.out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
