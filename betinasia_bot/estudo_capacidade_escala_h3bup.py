#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estudo de capacidade, max_stake/liquidez e deseconomia de escala para H3BUP.

H3BUP:
- Back Pre
- slippage_pre_pct < 0
- odd tomada 1.85..2.15
- capacity/max_stake/liquidity/limit > 100
- P&L accounting real por order_id quando executada
- event_id como cluster principal

Nao recomenda stake/exposicao. Estima volume, risco e sensibilidade.
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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from estudo_robusto_backpre_hipoteses import (
    _drawdown,
    _estimate_wc_start,
    _events,
    _fmt,
    _latest_balance,
    _load_rows,
    _resolve_db,
    _roi,
    _roi_without_topk,
    _rule_filters,
)


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


def _percentile(xs: Sequence[float], q: float) -> Optional[float]:
    if not xs:
        return None
    arr = sorted(float(x) for x in xs)
    idx = max(0, min(len(arr) - 1, int(round((len(arr) - 1) * q))))
    return arr[idx]


def _get(obj: Any, path: Sequence[str]) -> Any:
    cur = obj
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _extract_order_ids(obj: Any, out: set[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if _norm(k) in {"orderid", "order_id", "ticketid", "betid"}:
                sv = str(v or "").strip()
                if sv:
                    out.add(sv)
            _extract_order_ids(v, out)
    elif isinstance(obj, list):
        for x in obj:
            _extract_order_ids(x, out)


def _read_league_file(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {ln.strip() for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip() and not ln.startswith("#")}


def _load_balance_map(balance_csv: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with balance_csv.open(encoding="utf-8", errors="ignore", newline="") as f:
        rd = csv.DictReader(f)
        fields = list(rd.fieldnames or [])
        by = {_norm(c): c for c in fields}
        c_order = next((by.get(_norm(c)) for c in ["order id", "order_id", "orderid", "bet id", "ticket id"] if by.get(_norm(c))), None)
        c_pnl = next((by.get(_norm(c)) for c in ["amount", "pnl_real", "pnl", "profit", "pl", "result", "net"] if by.get(_norm(c))), None)
        c_post = next((by.get(_norm(c)) for c in ["post date", "post_date", "created_at", "timestamp"] if by.get(_norm(c))), None)
        if not c_order or not c_pnl:
            return out
        for r in rd:
            oid = str(r.get(c_order, "")).strip()
            pnl = _pf(r.get(c_pnl))
            if not oid or pnl is None:
                continue
            rec = out.setdefault(oid, {"pnl": 0.0, "post_date": ""})
            rec["pnl"] += float(pnl)
            if c_post and not rec["post_date"]:
                rec["post_date"] = str(r.get(c_post, "")).strip()
    return out


def _h3bup_filter(r: Dict[str, Any]) -> bool:
    return (
        r.get("slippage_pre_pct") is not None
        and float(r["slippage_pre_pct"]) < 0
        and r.get("odd") is not None
        and 1.85 <= float(r["odd"]) <= 2.15
        and r.get("capacity") is not None
        and float(r["capacity"]) > 100
    )


def _cap_bucket(x: Optional[float]) -> str:
    if x is None:
        return "NA"
    if x < 250:
        return "100-250"
    if x < 500:
        return "250-500"
    if x < 1000:
        return "500-1000"
    if x < 2500:
        return "1000-2500"
    if x < 5000:
        return "2500-5000"
    return ">5000"


def _load_potential(jsonl: Path, balance_csv: Path, start: str, end: str, executed_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bal = _load_balance_map(balance_csv)
    executed_by_order = {r.get("order_id"): r for r in executed_rows if r.get("order_id")}
    out: List[Dict[str, Any]] = []
    if not jsonl.exists():
        return out
    for ln in jsonl.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "2026-" not in ln:
            continue
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        req = obj.get("request") if isinstance(obj.get("request"), dict) else {}
        res = obj.get("result") if isinstance(obj.get("result"), dict) else {}
        status = str(res.get("status") or "").upper()
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
        odd = _pf(res.get("odd_at_decision") or req.get("odd_at_decision"))
        slip = _pf(vs.get("slippage_pre_pct"))
        cap = _pf(res.get("limit_final"))
        if cap is None:
            cap = _pf(_get(raw, ["value_sizing", "params", "stake_pre_fast"]))
        line = _pf(res.get("line") or req.get("line"))
        league = str(_get(req, ["meta", "bridge", "league"]) or "")
        event_id = str(res.get("event_id") or req.get("event_id") or "")
        stake_eff = _pf(vs.get("stake_chosen")) or _pf(_get(res, ["policy", "stake_requested"])) or _pf(_get(req, ["policy", "stake_requested"]))
        order_ids: set[str] = set()
        _extract_order_ids(obj, order_ids)
        order_id = ""
        for oid in sorted(order_ids):
            if oid in bal:
                order_id = oid
                break
        pnl = bal.get(order_id, {}).get("pnl") if order_id else None
        reason = str(res.get("error") or "")
        if "SLIPPAGE" in reason.upper():
            block_reason = "slippage"
        elif "LEAGUE" in reason.upper() or "POLICY" in reason.upper():
            block_reason = "league_or_policy"
        elif "SESSION" in reason.upper() or "API" in reason.upper() or "PMM" in reason.upper() or status in {"NO_SESSION", "API_FAILED", "STALE"}:
            block_reason = "operational_or_api"
        elif "BANK" in reason.upper() or "BALANCE" in reason.upper():
            block_reason = "bankroll"
        elif status == "LIVE_OK":
            block_reason = ""
        else:
            block_reason = "other"
        rec = {
            "order_id": order_id,
            "event_id": event_id,
            "created_at": created,
            "day": day,
            "post_date": bal.get(order_id, {}).get("post_date", "") if order_id else "",
            "league": league,
            "is_world_cup": "world cup" in league.lower() or "fifa world cup" in league.lower(),
            "odd": odd,
            "slippage_pre_pct": slip,
            "capacity": cap,
            "max_stake": cap,
            "line": line,
            "stake": float(stake_eff or 0),
            "pnl": pnl,
            "status": status,
            "block_reason": block_reason,
            "executed": status == "LIVE_OK" and pnl is not None and order_id in executed_by_order,
            "policy_version": str(_get(res, ["policy", "policy_version"]) or _get(req, ["policy", "policy_version"]) or ""),
            "pre_submit_ms": _pf(vs.get("pre_submit_ms")),
        }
        if _h3bup_filter(rec):
            out.append(rec)
    # dedup: prefer executed row per order/status, else event+time.
    seen = set()
    dedup = []
    for r in out:
        key = r["order_id"] or (r["event_id"], r["created_at"], r["status"], r["block_reason"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    return dedup


def _events(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        k = str(r.get("event_id") or r.get("order_id") or r.get("created_at"))
        e = by.setdefault(k, {"event_id": k, "stake": 0.0, "pnl": 0.0, "capacity": [], "n": 0, "league": r.get("league", ""), "day": r.get("day", "")})
        e["stake"] += float(r.get("stake") or 0)
        e["pnl"] += float(r.get("pnl") or 0)
        if r.get("capacity") is not None:
            e["capacity"].append(float(r["capacity"]))
        e["n"] += 1
    out = list(by.values())
    for e in out:
        e["roi"] = 100.0 * e["pnl"] / e["stake"] if e["stake"] > 0 else None
        e["capacity_mean"] = statistics.mean(e["capacity"]) if e["capacity"] else None
    return out


def _roi(rows: Sequence[Dict[str, Any]]) -> Tuple[float, float, float]:
    st = sum(float(r.get("stake") or 0) for r in rows)
    pnl = sum(float(r.get("pnl") or 0) for r in rows if r.get("pnl") is not None)
    return (100.0 * pnl / st if st > 0 else float("nan"), pnl, st)


def _simulated_scaled_rows(executed: Sequence[Dict[str, Any]], cap: Optional[float] = None, frac: Optional[float] = None) -> List[Dict[str, Any]]:
    out = []
    for r in executed:
        max_stake = float(r.get("capacity") or r.get("max_stake") or r.get("stake") or 0)
        old_stake = float(r.get("stake") or 0)
        if max_stake <= 0 or old_stake <= 0 or r.get("pnl") is None:
            continue
        if frac is not None:
            new_stake = max_stake * float(frac)
        elif cap is not None:
            new_stake = min(max_stake, float(cap))
        else:
            new_stake = max_stake
        scale = new_stake / old_stake
        nr = dict(r)
        nr["stake"] = new_stake
        nr["pnl"] = float(r["pnl"]) * scale
        out.append(nr)
    return out


def _summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ev = _events(rows)
    roi, pnl, stake = _roi(rows)
    days = sorted({r["day"] for r in rows if r.get("day")})
    active_days = len(days)
    cal_days = 0
    if days:
        cal_days = (datetime.fromisoformat(days[-1]).date() - datetime.fromisoformat(days[0]).date()).days + 1
    return {
        "n": len(rows),
        "events": len(ev),
        "active_days": active_days,
        "calendar_days": cal_days,
        "stake": stake,
        "pnl": pnl,
        "roi": roi,
        "bets_per_active_day": len(rows) / active_days if active_days else None,
        "events_per_active_day": len(ev) / active_days if active_days else None,
        "turnover_per_active_day": stake / active_days if active_days else None,
        "monthly_turnover_active": stake / active_days * 30 if active_days else None,
        "monthly_turnover_calendar": stake / cal_days * 30 if cal_days else None,
        "roi_without_top": {str(k): _roi_without_topk(ev, k) for k in [1, 3, 5, 10]},
    }


def _roi_without_topk(events: Sequence[Dict[str, Any]], k: int) -> Optional[float]:
    kept = sorted(events, key=lambda e: e["pnl"], reverse=True)[k:]
    if not kept:
        return None
    st = sum(e["stake"] for e in kept)
    return 100.0 * sum(e["pnl"] for e in kept) / st if st > 0 else None


def _bootstrap_month(executed: Sequence[Dict[str, Any]], monthly_events: int, monthly_turnover: float, iters: int, seed: int) -> Dict[str, Any]:
    ev = _events(executed)
    if not ev or monthly_events <= 0 or monthly_turnover <= 0:
        return {}
    rng = random.Random(seed)
    vals = []
    dds = []
    for _ in range(iters):
        smp = [ev[rng.randrange(len(ev))] for __ in range(monthly_events)]
        st = sum(e["stake"] for e in smp)
        scale = monthly_turnover / st if st > 0 else 0
        pnl = sum(e["pnl"] for e in smp) * scale
        vals.append(pnl)
        eq = peak = dd = 0.0
        for e in smp:
            eq += e["pnl"] * scale
            peak = max(peak, eq)
            dd = max(dd, peak - eq)
        dds.append(dd)
    vals_s = sorted(vals)
    dds_s = sorted(dds)
    return {
        "pnl_mean": statistics.mean(vals),
        "pnl_median": statistics.median(vals),
        "p10": _percentile(vals_s, 0.10),
        "p25": _percentile(vals_s, 0.25),
        "p75": _percentile(vals_s, 0.75),
        "p90": _percentile(vals_s, 0.90),
        "p95_loss": _percentile(vals_s, 0.05),
        "prob_negative_month": sum(1 for x in vals if x < 0) / len(vals),
        "drawdown_p95": _percentile(dds_s, 0.95),
    }


def _maxstake_distribution(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    caps = [float(r["capacity"]) for r in rows if r.get("capacity") is not None]
    buckets = defaultdict(list)
    for r in rows:
        buckets[_cap_bucket(r.get("capacity"))].append(r)
    return {
        "mean": statistics.mean(caps) if caps else None,
        "median": statistics.median(caps) if caps else None,
        "p10": _percentile(caps, 0.10),
        "p25": _percentile(caps, 0.25),
        "p75": _percentile(caps, 0.75),
        "p90": _percentile(caps, 0.90),
        "p95": _percentile(caps, 0.95),
        "p99": _percentile(caps, 0.99),
        "buckets": [
            {
                "bucket": b,
                "n": len(rs),
                "events": len(_events(rs)),
                "potential_stake_total": sum(float(r.get("capacity") or 0) for r in rs),
                "executed_n": sum(1 for r in rs if r.get("executed")),
                "executed_roi": _roi([r for r in rs if r.get("executed") and r.get("pnl") is not None])[0],
                "executed_pnl": _roi([r for r in rs if r.get("executed") and r.get("pnl") is not None])[1],
            }
            for b, rs in sorted(buckets.items())
        ],
    }


def _scale_tests(executed: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    # Buckets, decile, quartile, correlations and simple OLS coefficients.
    buckets = defaultdict(list)
    for r in executed:
        buckets[_cap_bucket(r.get("capacity"))].append(r)
    bucket_stats = []
    for b, rs in sorted(buckets.items()):
        rr, pp, ss = _roi(rs)
        bucket_stats.append({"bucket": b, "n": len(rs), "events": len(_events(rs)), "roi": rr, "pnl": pp, "stake": ss})
    caps = sorted([float(r["capacity"]) for r in executed if r.get("capacity") is not None])
    def qbucket(r, qn):
        if not caps or r.get("capacity") is None:
            return "NA"
        rank = sum(1 for x in caps if x <= float(r["capacity"])) / len(caps)
        return f"Q{min(qn, max(1, int(math.ceil(rank * qn))))}"
    quart = defaultdict(list)
    dec = defaultdict(list)
    for r in executed:
        quart[qbucket(r, 4)].append(r)
        dec[qbucket(r, 10)].append(r)
    def grp(by):
        return [{"bucket": k, "n": len(v), "roi": _roi(v)[0], "pnl": _roi(v)[1], "stake": _roi(v)[2]} for k, v in sorted(by.items())]
    xs = [(float(r["capacity"]), float(r["pnl"]) / float(r["stake"]) * 100.0, r) for r in executed if r.get("capacity") and r.get("stake")]
    def corr(vals):
        if len(vals) < 3:
            return None
        x = [v[0] for v in vals]
        y = [v[1] for v in vals]
        mx, my = statistics.mean(x), statistics.mean(y)
        cov = sum((a-mx)*(b-my) for a,b in zip(x,y))
        vx = sum((a-mx)**2 for a in x)
        vy = sum((b-my)**2 for b in y)
        return cov / math.sqrt(vx*vy) if vx and vy else None
    corr_cap = corr(xs)
    corr_log = corr([(math.log(max(1e-9, x)), y, r) for x,y,r in xs])
    # Simple robust-ish regression by OLS with controls (not full clustered SE; cluster summary limitation).
    # y = b0 + b1 log(cap) + b2 odd + b3 slip + b4 wc + b5 line + b6 pre_submit_ms
    data = []
    for x,y,r in xs:
        row = [1.0, math.log(max(1e-9, x)), float(r.get("odd") or 0), float(r.get("slippage_pre_pct") or 0), 1.0 if r.get("is_world_cup") else 0.0, float(r.get("line") or 0), float(r.get("pre_submit_ms") or 0)/1000.0]
        data.append((row, y))
    beta = None
    if len(data) >= 10:
        # Normal equations with small ridge for stability, sem depender de numpy.
        n_cols = len(data[0][0])
        xtx = [[0.0 for _ in range(n_cols)] for __ in range(n_cols)]
        xty = [0.0 for _ in range(n_cols)]
        for row, y in data:
            for i in range(n_cols):
                xty[i] += row[i] * y
                for j in range(n_cols):
                    xtx[i][j] += row[i] * row[j]
        for i in range(n_cols):
            xtx[i][i] += 1e-6
        # Gaussian elimination.
        a = [xtx[i][:] + [xty[i]] for i in range(n_cols)]
        for col in range(n_cols):
            pivot = max(range(col, n_cols), key=lambda r: abs(a[r][col]))
            if abs(a[pivot][col]) < 1e-12:
                beta = None
                break
            if pivot != col:
                a[col], a[pivot] = a[pivot], a[col]
            div = a[col][col]
            for j in range(col, n_cols + 1):
                a[col][j] /= div
            for r in range(n_cols):
                if r == col:
                    continue
                factor = a[r][col]
                if factor == 0:
                    continue
                for j in range(col, n_cols + 1):
                    a[r][j] -= factor * a[col][j]
        else:
            beta = [a[i][n_cols] for i in range(n_cols)]
    return {
        "bucket_stats": bucket_stats,
        "quartiles": grp(quart),
        "deciles": grp(dec),
        "corr_capacity_roi": corr_cap,
        "corr_log_capacity_roi": corr_log,
        "ols_coefficients": None if beta is None else {
            "intercept": beta[0], "log_capacity": beta[1], "odd": beta[2], "slippage": beta[3], "world_cup": beta[4], "line": beta[5], "pre_submit_s": beta[6]
        },
        "regression_limitation": "OLS simples sem erro-padrao clustered completo; usar direcao dos coeficientes apenas como diagnostico exploratorio.",
    }


def _cap_curve(executed: Sequence[Dict[str, Any]], potential: Sequence[Dict[str, Any]], caps: Sequence[Optional[float]], iters: int) -> Dict[str, Any]:
    out = {}
    # Potential turnover uses all observed eligible opportunities; P&L uses executed scaled rows only.
    for cap in caps:
        name = "max_stake_integral" if cap is None else f"cap_{int(cap)}"
        sim_exec = _simulated_scaled_rows(executed, cap=cap)
        pot_turn = sum(min(float(r.get("capacity") or 0), float(cap)) if cap is not None else float(r.get("capacity") or 0) for r in potential)
        s = _summary(sim_exec)
        monthly_turn = (pot_turn / max(1, _active_days(potential)) * 30.0) if potential else None
        monthly_events = len(_events(potential)) / max(1, _active_days(potential)) * 30.0 if potential else None
        monthly_bets = len(potential) / max(1, _active_days(potential)) * 30.0 if potential else None
        sim = _bootstrap_month(sim_exec, max(1, int(round(monthly_events or 0))), float(monthly_turn or 0), iters, (hash(name) % 100000))
        out[name] = {"cap": cap, "executed_scaled": s, "potential_turnover_month": monthly_turn, "potential_bets_month": monthly_bets, "potential_events_month": monthly_events, "capital_avg_turnover_div3": (monthly_turn/3.0 if monthly_turn else None), "monthly_bootstrap": sim}
    return out


def _active_days(rows):
    return len({r["day"] for r in rows if r.get("day")})


def _summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ev = _events(rows)
    rr, pp, ss = _roi(rows)
    return {"n": len(rows), "events": len(ev), "stake": ss, "pnl": pp, "roi": rr, "roi_without_top": {str(k): _roi_without_topk(ev,k) for k in [1,3,5,10]}, "concentration": {str(k): _top_pct(ev,k,pp) for k in [1,3,5,10]}}


def _top_pct(ev, k, pnl):
    v = sum(e["pnl"] for e in sorted(ev, key=lambda e:e["pnl"], reverse=True)[:k])
    return 100.0*v/pnl if pnl else None


def _compare_league_filters(executed: Sequence[Dict[str, Any]], potential: Sequence[Dict[str, Any]], hist: set[str], static_bl: set[str]) -> Dict[str, Any]:
    variants = {
        "sem_filtro_ligas": lambda r: True,
        "filtro_historico": lambda r: (r.get("league") or "") in hist,
        "blacklist_estatica": lambda r: (r.get("league") or "") not in static_bl,
        "todas_ligas_reportando_roi": lambda r: True,
    }
    out = {}
    for name, fn in variants.items():
        ex = [r for r in executed if fn(r)]
        pot = [r for r in potential if fn(r)]
        out[name] = {"executed": _summary(ex), "potential": {"n": len(pot), "events": len(_events(pot)), "potential_turnover": sum(float(r.get("capacity") or 0) for r in pot)}}
    # ROI por liga
    by = defaultdict(list)
    for r in executed:
        by[r.get("league") or "NA"].append(r)
    out["roi_por_liga"] = [{"league": k, "n": len(v), "events": len(_events(v)), "roi": _roi(v)[0], "pnl": _roi(v)[1], "stake": _roi(v)[2]} for k,v in sorted(by.items(), key=lambda kv: len(kv[1]), reverse=True)]
    return out


def _render_md(out: Dict[str, Any], md: Path) -> None:
    lines = []
    lines.append("# Estudo de capacidade, max stake e deseconomia de escala - H3BUP\n")
    lines.append("## Sumario executivo\n")
    lines.append("- Objetivo: estimar capacidade e risco; nao recomendar aumento de stake/exposicao.")
    lines.append("- Base executada usa P&L real; base potencial observada usa logs e nao deve alimentar ROI quando nao houver P&L contrafactual.")
    lines.append("- H3BUP: Back Pre, slippage<0, odd 1.85-2.15, capacity>100.\n")
    b = out["bases"]
    lines.append("## Bases\n")
    lines.append("| Base | N | Eventos | Dias ativos | Turnover potencial/real | ROI realizado |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(f"| Executada | {b['executed']['n']} | {b['executed']['events']} | NA | {_fmt(b['executed']['stake'],2)} | {_fmt(b['executed']['roi'],2,True)} |")
    lines.append(f"| Elegivel observada | {b['potential']['n']} | {b['potential']['events']} | {b['potential']['active_days']} | {_fmt(b['potential']['potential_turnover'],2)} | NA |")
    lines.append(f"| Shadow/contrafactual sem P&L | {b['shadow']['n']} | {b['shadow']['events']} | {b['shadow']['active_days']} | {_fmt(b['shadow']['potential_turnover'],2)} | NA |\n")
    lines.append("## Distribuicao exata de max_stake/capacity\n")
    d = out["maxstake_distribution"]
    lines.append(f"- mean={_fmt(d['mean'],2)} median={_fmt(d['median'],2)} p75={_fmt(d['p75'],2)} p90={_fmt(d['p90'],2)} p95={_fmt(d['p95'],2)} p99={_fmt(d['p99'],2)}")
    lines.append("| Bucket | N oportunidades | Eventos | Stake potencial | Exec N | ROI executado | P&L executado |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in d["buckets"]:
        lines.append(f"| {r['bucket']} | {r['n']} | {r['events']} | {_fmt(r['potential_stake_total'],2)} | {r['executed_n']} | {_fmt(r['executed_roi'],2,True)} | {_fmt(r['executed_pnl'],2)} |")
    lines.append("\n## Curva cap aplicado\n")
    lines.append("| Cap | Apostas/mes pot. | Eventos/mes pot. | Turnover/mes pot. | Capital medio | ROI exec escalado | ROI sem Top-3 | Prob mes neg. | p95 perda | DD p95 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, c in out["cap_curve"].items():
        sim = c["monthly_bootstrap"]
        s = c["executed_scaled"]
        lines.append(f"| {name} | {_fmt(c['potential_bets_month'],1)} | {_fmt(c['potential_events_month'],1)} | {_fmt(c['potential_turnover_month'],2)} | {_fmt(c['capital_avg_turnover_div3'],2)} | {_fmt(s['roi'],2,True)} | {_fmt(s['roi_without_top']['3'],2,True)} | {_fmt(100*sim.get('prob_negative_month') if sim else None,1,True)} | {_fmt(sim.get('p95_loss') if sim else None,2)} | {_fmt(sim.get('drawdown_p95') if sim else None,2)} |")
    lines.append("\n## Deseconomia de escala\n")
    sc = out["scale_tests"]
    lines.append(f"- Corr(capacity, ROI aposta)={_fmt(sc['corr_capacity_roi'],4)}")
    lines.append(f"- Corr(log(capacity), ROI aposta)={_fmt(sc['corr_log_capacity_roi'],4)}")
    lines.append(f"- Coef OLS log_capacity={_fmt((sc.get('ols_coefficients') or {}).get('log_capacity'),4)} (diagnostico, nao inferencia causal)")
    lines.append("| Bucket | N | Eventos | ROI | P&L | Stake |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in sc["bucket_stats"]:
        lines.append(f"| {r['bucket']} | {r['n']} | {r['events']} | {_fmt(r['roi'],2,True)} | {_fmt(r['pnl'],2)} | {_fmt(r['stake'],2)} |")
    lines.append("\n### Quartis de capacity\n")
    lines.append("| Quartil | N | ROI | P&L | Stake |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in sc["quartiles"]:
        lines.append(f"| {r['bucket']} | {r['n']} | {_fmt(r['roi'],2,True)} | {_fmt(r['pnl'],2)} | {_fmt(r['stake'],2)} |")
    lines.append("\n## Com e sem filtro de ligas\n")
    lines.append("| Variante | Exec N | Exec ROI | Exec ROI sem Top3 | Pot N | Pot Eventos | Pot Turnover |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for k, v in out["league_filter_comparison"].items():
        if k == "roi_por_liga":
            continue
        lines.append(f"| {k} | {v['executed']['n']} | {_fmt(v['executed']['roi'],2,True)} | {_fmt(v['executed']['roi_without_top']['3'],2,True)} | {v['potential']['n']} | {v['potential']['events']} | {_fmt(v['potential']['potential_turnover'],2)} |")
    lines.append("\n## Regimes de calendario\n")
    lines.append("| Regime | Pot N | Exec N | ROI exec | ROI sem Top3 | MaxStake med | Turnover pot cap500 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for k, v in out["calendar_regimes"].items():
        lines.append(f"| {k} | {v['potential_n']} | {v['executed_n']} | {_fmt(v['executed_roi'],2,True)} | {_fmt(v['executed_roi_no_top3'],2,True)} | {_fmt(v['maxstake_median'],2)} | {_fmt(v['turnover_potential_cap500'],2)} |")
    lines.append("\n## Bloqueios / oportunidades perdidas\n")
    lines.append(json.dumps(out["blocked_summary"], ensure_ascii=False, indent=2))
    lines.append("\n## Respostas objetivas\n")
    for k, v in out["answers"].items():
        lines.append(f"- **{k}**: {v}")
    md.write_text("\n".join(lines), encoding="utf-8")


def _render_pdf(md: Path, pdf: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=6.1, leading=7.1))
    styles.add(ParagraphStyle(name="BodyX", parent=styles["BodyText"], fontSize=8.2, leading=10))
    doc = SimpleDocTemplate(str(pdf), pagesize=landscape(A4), leftMargin=18, rightMargin=18, topMargin=18, bottomMargin=18)
    story = []
    def esc(s): return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
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
            block=[]
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip()); i += 1
            parsed=[[c.strip() for c in row.strip("|").split("|")] for row in block]
            if len(parsed)>=2:
                data=[[Paragraph(esc(c), styles["Small"]) for c in parsed[0]]]
                data += [[Paragraph(esc(c), styles["Small"]) for c in r] for r in parsed[2:]]
                tbl=Table(data, repeatRows=1)
                tbl.setStyle(TableStyle([
                    ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#111827")),
                    ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                    ("GRID",(0,0),(-1,-1),0.2,colors.HexColor("#d1d5db")),
                    ("VALIGN",(0,0),(-1,-1),"TOP"),
                    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f9fafb")]),
                    ("LEFTPADDING",(0,0),(-1,-1),2),("RIGHTPADDING",(0,0),(-1,-1),2),
                    ("TOPPADDING",(0,0),(-1,-1),1.1),("BOTTOMPADDING",(0,0),(-1,-1),1.1),
                ]))
                story.append(tbl); story.append(Spacer(1,5))
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
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-pdf", required=True)
    args = ap.parse_args()
    end = args.end_day or datetime.now(timezone.utc).date().isoformat()
    db = _resolve_db(args.database_url)
    bal = Path(args.balance_csv) if args.balance_csv else _latest_balance()
    executed_all = _load_rows(Path(args.executor_jsonl), bal, db, args.start_day, end)
    executed = [r for r in executed_all if _h3bup_filter({"slippage_pre_pct": r.get("slippage_pre_pct"), "odd": r.get("odd"), "capacity": r.get("capacity")})]
    potential = _load_potential(Path(args.executor_jsonl), bal, args.start_day, end, executed)
    executed_orders = {r["order_id"] for r in executed}
    for r in potential:
        r["executed"] = bool(r.get("order_id") in executed_orders)
    shadow = [r for r in potential if not r.get("executed")]
    wc_reg = _estimate_wc_start(executed_all)
    wc_start = (wc_reg.get("0.5") or {}).get("date") or "2026-06-04"
    root = Path.cwd()
    hist = _read_league_file(root / "betinasia_bot/logs/approved_leagues_frozen.txt")
    # static blacklist as derived bad list from earlier study if no file; simple conservative default empty.
    static_bad = {"CONMEBOL Copa Libertadores", "FIFA World Cup", "France Ligue 1", "Japan J-League Division 1", "Spain La Liga"}
    caps = [100,250,500,750,1000,1500,2000,3000,5000,None]
    regimes = {}
    for name, pred in {
        "normal_pre_world_cup": lambda r: r["day"] < wc_start and not r.get("is_world_cup"),
        "world_cup": lambda r: r["day"] >= wc_start and r.get("is_world_cup"),
        "non_world_cup_during_world_cup": lambda r: r["day"] >= wc_start and not r.get("is_world_cup"),
        "pos_policy_ligas_aprox": lambda r: r["day"] >= "2026-06-02",
    }.items():
        pot = [r for r in potential if pred(r)]
        ex = [r for r in executed if pred(r)]
        caps500 = sum(min(float(r.get("capacity") or 0), 500.0) for r in pot)
        caps_vals = [float(r["capacity"]) for r in pot if r.get("capacity") is not None]
        regimes[name] = {"potential_n": len(pot), "potential_events": len(_events(pot)), "executed_n": len(ex), "executed_events": len(_events(ex)), "executed_roi": _roi(ex)[0], "executed_roi_no_top3": _roi_without_topk(_events(ex),3), "maxstake_mean": statistics.mean(caps_vals) if caps_vals else None, "maxstake_median": statistics.median(caps_vals) if caps_vals else None, "turnover_potential_cap500": caps500}
    out = {
        "params": {"start_day": args.start_day, "end_day": end, "balance_csv": str(bal), "executed_all_backpre": len(executed_all), "executed_h3bup": len(executed), "potential_h3bup": len(potential)},
        "world_cup_regime": wc_reg,
        "bases": {
            "executed": _summary(executed),
            "potential": {"n": len(potential), "events": len(_events(potential)), "active_days": _active_days(potential), "potential_turnover": sum(float(r.get("capacity") or 0) for r in potential)},
            "shadow": {"n": len(shadow), "events": len(_events(shadow)), "active_days": _active_days(shadow), "potential_turnover": sum(float(r.get("capacity") or 0) for r in shadow)},
        },
        "maxstake_distribution": _maxstake_distribution(potential),
        "cap_curve": _cap_curve(executed, potential, caps, args.iters),
        "scale_tests": _scale_tests(executed),
        "league_filter_comparison": _compare_league_filters(executed, potential, hist, static_bad),
        "calendar_regimes": regimes,
        "blocked_summary": {"counts": dict(Counter(r.get("block_reason") or "executed" for r in potential)), "note": "Bloqueadas sem P&L contrafactual entram em volume/capacidade, nao ROI."},
    }
    out["answers"] = {
        "Apostas/mes potencial sem filtro de ligas": "Ver coluna potential_bets_month na curva de cap; usa oportunidades H3BUP observadas sem aplicar filtro de liga.",
        "Apostas/mes por execucao real": f"{len(executed) / max(1, _active_days(executed)) * 30:.1f} por run-rate de dias ativos.",
        "Reducao por filtro de ligas": "Comparar sem_filtro_ligas vs filtro_historico; se potencial cai e ROI OOS nao melhora consistentemente, filtro muda universo mais que prova alpha.",
        "Max stake medio/mediano/p75/p90/p95": "Reportado na distribuicao max_stake.",
        "Turnover/capital por cap": "Reportado na curva cap aplicado; capital medio = turnover/3.",
        "P&L mensal esperado": "Usar bootstrap mensal por evento de cada cap; nao usar oportunidades sem P&L para ROI.",
        "Deseconomia de escala": "Avaliada por buckets/quartis/decis/correlacoes/regressao diagnostica; nao assumir causalidade.",
        "Mercados pequenos ou maiores": "Ver ROI por bucket de capacity; se ROI persiste em >1000, nao depende so de mercado pequeno.",
        "Dados faltantes": "P&L contrafactual de bloqueadas, max_stake pre-decisao para todos sinais, uptime minuto a minuto, policy_id por decisao e exposicao aberta real.",
    }
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _render_md(out, Path(args.out_md))
    _render_pdf(Path(args.out_md), Path(args.out_pdf))
    print(f"[OK] executed_h3bup={len(executed)} potential_h3bup={len(potential)} shadow={len(shadow)}")
    print(f"[OK] JSON={args.out_json}")
    print(f"[OK] MD={args.out_md}")
    print(f"[OK] PDF={args.out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
