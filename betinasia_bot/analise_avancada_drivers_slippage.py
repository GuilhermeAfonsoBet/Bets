#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analise avancada de drivers para Back Pre usando slippage do executor.

Inclui:
- Top eventos positivos/negativos e drivers
- ROI sem Top-k por evento
- M2 por subsegmento
- M1..M5 por buckets de slippage, liga e faixa de odd
- Slippage x odd x linha AH
- Bootstrap por evento no pos sem World Cup
- Rolling 30/50/100 apostas sem Top-3
- Capacidade: stake/max stake/liquidez vs ROI
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


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


def _norm(s: str) -> str:
    return "".join(ch for ch in str(s or "").strip().lower() if ch.isalnum())


def _pdt(s: Any) -> Optional[datetime]:
    x = str(s or "").strip()
    if not x:
        return None
    if x.endswith("Z"):
        x = x[:-1] + "+00:00"
    try:
        d = datetime.fromisoformat(x)
    except Exception:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc)


def _fmt(x: Optional[float], nd: int = 2, pct: bool = False) -> str:
    if x is None:
        return "NA"
    return f"{float(x):.{nd}f}" + ("%" if pct else "")


def _get_path(obj: Any, path: Sequence[str]) -> Any:
    cur = obj
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


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


def _run(cmd: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(list(cmd), capture_output=True, text=True, check=False)


def _read_database_url_from_env_file(path: Path) -> str:
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
            return v.strip()
    return ""


def _resolve_db(cli: str) -> str:
    if cli:
        return cli
    if os.getenv("DATABASE_URL"):
        return os.getenv("DATABASE_URL", "")
    for p in [Path.cwd() / "betinasia_bot/.env", Path("/home/betbot/Bets/betinasia_bot/.env")]:
        v = _read_database_url_from_env_file(p)
        if v:
            return v
    return ""


def _psql_map(db: str, start_day: str, end_day: str) -> Dict[str, Dict[str, str]]:
    if not db:
        return {}
    sql = f"""
    SELECT
      id::text,
      COALESCE(league,'')::text,
      COALESCE(event_id::text,'')::text,
      audited_at::text
    FROM betslip_audit_results
    WHERE audited_at >= '{start_day}'::timestamptz
      AND audited_at <= '{end_day} 23:59:59+00'::timestamptz
      AND hypothesis_type='H3B'
      AND reversal_direction='up';
    """
    p = _run(["psql", db, "-At", "-v", "ON_ERROR_STOP=1", "-c", sql])
    if p.returncode != 0:
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for ln in p.stdout.splitlines():
        parts = ln.split("|")
        if len(parts) < 4:
            continue
        aid, league, event_id, audited_at = parts[0], parts[1], parts[2], parts[3]
        out[aid] = {"league": league, "event_id": event_id, "audited_at": audited_at}
    return out


def _latest_balance_csv() -> Path:
    c = sorted(glob.glob("/home/betbot/Bets/betinasia_bot/logs/accounting/*balance*.csv"), key=os.path.getmtime)
    if not c:
        raise RuntimeError("balance CSV nao encontrado")
    return Path(c[-1])


def _pnl_by_order(balance_csv: Path) -> Dict[str, float]:
    out: Dict[str, float] = defaultdict(float)
    with balance_csv.open(encoding="utf-8", errors="ignore", newline="") as f:
        rd = csv.DictReader(f)
        fields = list(rd.fieldnames or [])
        by = {_norm(c): c for c in fields}
        oid_col = next((by.get(_norm(c)) for c in ["order id", "order_id", "orderid", "bet id", "ticket id", "ticket_id"] if by.get(_norm(c))), None)
        val_col = next((by.get(_norm(c)) for c in ["amount", "pnl_real", "pnl", "profit", "pl", "result", "net"] if by.get(_norm(c))), None)
        if not oid_col or not val_col:
            raise RuntimeError("balance sem colunas de order/pnl")
        for r in rd:
            oid = str(r.get(oid_col, "")).strip()
            v = _pf(r.get(val_col))
            if oid and v is not None:
                out[oid] += float(v)
    return out


def _load_rows(jsonl: Path, balance_csv: Path, db: str, start_day: str, end_day: str) -> List[Dict[str, Any]]:
    pnl_map = _pnl_by_order(balance_csv)
    audit_meta = _psql_map(db, start_day, end_day)
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
            regime = str(vs.get("market_regime") or _get_path(req, ["meta", "market", "regime"]) or "").lower()
            market_live = vs.get("market_is_live")
            if not (regime == "pre" or market_live is False):
                continue
            created = str(res.get("created_at") or req.get("created_at") or "")
            day = created[:10]
            if day < start_day or day > end_day:
                continue
            slip = _pf(vs.get("slippage_pre_pct"))
            if slip is None:
                continue
            order_ids: set[str] = set()
            _extract_order_ids(obj, order_ids)
            order_id = ""
            for oid in sorted(order_ids):
                if oid in pnl_map:
                    order_id = oid
                    break
            if not order_id:
                continue
            stake = _pf(vs.get("stake_chosen")) or _pf(_get_path(res, ["policy", "stake_requested"])) or _pf(_get_path(req, ["policy", "stake_requested"]))
            if stake is None or stake <= 0:
                continue
            aid = str(res.get("audit_id") or req.get("audit_id") or "").strip()
            meta = audit_meta.get(aid, {})
            event_id = str(res.get("event_id") or req.get("event_id") or meta.get("event_id") or "").strip()
            league = str(meta.get("league") or _get_path(req, ["meta", "bridge", "league"]) or "").strip()
            line = str(res.get("line") or req.get("line") or "").strip()
            odd_dec = _pf(res.get("odd_at_decision") or req.get("odd_at_decision"))
            odd_final = _pf(res.get("odd_final") or vs.get("odd_pre_submit"))
            limit_final = _pf(res.get("limit_final"))
            max_stake = _pf(_get_path(raw, ["value_sizing", "params", "stake_pre_fast"]))
            if max_stake is None:
                max_stake = _pf(_get_path(res, ["policy", "stake_requested"])) or stake
            pnl = pnl_map[order_id]
            ts = _pdt(created)
            rows.append(
                {
                    "audit_id": aid,
                    "event_id": event_id or aid or order_id,
                    "league": league,
                    "day": day,
                    "ts": ts,
                    "week": f"{ts.isocalendar().year}-W{ts.isocalendar().week:02d}" if ts else "unknown",
                    "order_id": order_id,
                    "line": line,
                    "line_num": _pf(line),
                    "odd_at_decision": odd_dec,
                    "odd_final": odd_final,
                    "slippage": float(slip),
                    "stake": float(stake),
                    "max_stake": float(max_stake or stake),
                    "limit_final": limit_final,
                    "pnl": float(pnl),
                    "roi": 100.0 * float(pnl) / float(stake),
                    "post_ms": _pf(_get_path(res, ["timing", "post_ms"])),
                    "pre_submit_ms": _pf(vs.get("pre_submit_ms")),
                }
            )
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
            e = {
                "event_id": k,
                "league": r.get("league", ""),
                "day": r.get("day", ""),
                "week": r.get("week", ""),
                "stake": 0.0,
                "pnl": 0.0,
                "n": 0,
                "slips": [],
                "odds": [],
                "lines": [],
                "limits": [],
            }
            by[k] = e
        e["stake"] += float(r["stake"])
        e["pnl"] += float(r["pnl"])
        e["n"] += 1
        e["slips"].append(float(r["slippage"]))
        if r.get("odd_at_decision") is not None:
            e["odds"].append(float(r["odd_at_decision"]))
        if r.get("line_num") is not None:
            e["lines"].append(float(r["line_num"]))
        if r.get("limit_final") is not None:
            e["limits"].append(float(r["limit_final"]))
    out = list(by.values())
    for e in out:
        e["roi"] = 100.0 * e["pnl"] / e["stake"] if e["stake"] else None
        e["avg_slip"] = sum(e["slips"]) / len(e["slips"]) if e["slips"] else None
        e["avg_odd"] = sum(e["odds"]) / len(e["odds"]) if e["odds"] else None
        e["avg_line"] = sum(e["lines"]) / len(e["lines"]) if e["lines"] else None
        e["avg_limit"] = sum(e["limits"]) / len(e["limits"]) if e["limits"] else None
    return out


def _roi_without_top(events: Sequence[Dict[str, Any]], k: int) -> Optional[float]:
    kept = sorted(events, key=lambda e: e["pnl"], reverse=True)[k:]
    if not kept:
        return None
    stake = sum(e["stake"] for e in kept)
    pnl = sum(e["pnl"] for e in kept)
    return 100.0 * pnl / stake if stake > 0 else None


def _bootstrap_ci(events: Sequence[Dict[str, Any]], iters: int, seed: int) -> Tuple[Optional[float], Optional[float]]:
    if not events:
        return None, None
    rng = random.Random(seed)
    vals: List[float] = []
    n = len(events)
    for _ in range(iters):
        smp = [events[rng.randrange(n)] for _ in range(n)]
        stake = sum(e["stake"] for e in smp)
        pnl = sum(e["pnl"] for e in smp)
        vals.append(100.0 * pnl / stake if stake > 0 else 0.0)
    vals.sort()
    return vals[int(0.05 * (len(vals) - 1))], vals[int(0.95 * (len(vals) - 1))]


def _perm_p(events: Sequence[Dict[str, Any]], iters: int, seed: int) -> Optional[float]:
    if not events:
        return None
    obs = 100.0 * sum(e["pnl"] for e in events) / sum(e["stake"] for e in events)
    rng = random.Random(seed + 17)
    ge = 0
    for _ in range(iters):
        pnl = 0.0
        stake = 0.0
        for e in events:
            pnl += (1 if rng.random() >= 0.5 else -1) * e["pnl"]
            stake += e["stake"]
        stat = 100.0 * pnl / stake if stake > 0 else 0
        if stat >= obs:
            ge += 1
    return (ge + 1) / (iters + 1)


def _weekly(rows: Sequence[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[r["week"]].append(r)
    rois = [_roi(v)[0] for v in by.values() if v]
    if not rois:
        return None, None
    xs = sorted(rois)
    med = xs[len(xs) // 2] if len(xs) % 2 else 0.5 * (xs[len(xs) // 2 - 1] + xs[len(xs) // 2])
    return 100.0 * sum(1 for x in rois if x > 0) / len(rois), med


def _m4(events: Sequence[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    ev = sorted(events, key=lambda e: e.get("day") or "")
    n = len(ev)
    if not n:
        return None, None, None
    a, b = n // 3, (2 * n) // 3
    out = []
    for part in [ev[:a], ev[a:b], ev[b:]]:
        if not part:
            out.append(None)
        else:
            out.append(100.0 * sum(e["pnl"] for e in part) / sum(e["stake"] for e in part))
    return out[0], out[1], out[2]


def _five_ms(rows: Sequence[Dict[str, Any]], iters: int, seed: int) -> Dict[str, Any]:
    if not rows:
        return {"n_bets": 0, "error": "empty"}
    ev = _events(rows)
    roi, pnl, stake = _roi(rows)
    ci_lo, ci_hi = _bootstrap_ci(ev, iters, seed)
    p_perm = _perm_p(ev, iters, seed)
    top_abs = None
    total_abs = sum(abs(e["pnl"]) for e in ev)
    if total_abs > 0:
        top_abs = 100.0 * max(abs(e["pnl"]) for e in ev) / total_abs
    roi_no_top3 = _roi_without_top(ev, 3)
    pos_ratio, med_week = _weekly(rows)
    r1, r2, r3 = _m4(ev)
    ev_bet = pnl / len(rows)
    m1 = p_perm is not None and p_perm <= 0.10 and ci_lo is not None and ci_lo > 0
    m2 = roi_no_top3 is not None and roi_no_top3 > 0 and top_abs is not None and top_abs <= 35
    m3 = pos_ratio is not None and pos_ratio >= 55 and med_week is not None and med_week > 0
    m4 = (((r1 is not None and r1 > 0) + (r2 is not None and r2 > 0) + (r3 is not None and r3 > 0)) >= 2 and (r3 is not None and r3 > 0))
    m5 = ev_bet > 0 and roi > 0
    score = sum([m1, m2, m3, m4, m5])
    return {
        "n_bets": len(rows),
        "n_events": len(ev),
        "roi": roi,
        "pnl": pnl,
        "stake": stake,
        "score": score,
        "label": "robusto" if score >= 4 else ("moderado" if score >= 2 else "fragil"),
        "M1": {"roi": roi, "p_perm": p_perm, "ci90_lo": ci_lo, "ci90_hi": ci_hi, "status": "OK" if m1 else "FAIL"},
        "M2": {"roi_sem_top3": roi_no_top3, "top1_abs_pct": top_abs, "status": "OK" if m2 else "FAIL"},
        "M3": {"pos_ratio_pct": pos_ratio, "mediana_semanal_pct": med_week, "status": "OK" if m3 else "FAIL"},
        "M4": {"r1_pct": r1, "r2_pct": r2, "r3_pct": r3, "status": "OK" if m4 else "FAIL"},
        "M5": {"ev_por_aposta": ev_bet, "ev_pct": roi, "status": "OK" if m5 else "FAIL"},
    }


def _bucket(value: Optional[float], cuts: Sequence[Tuple[float, float, str]]) -> str:
    if value is None:
        return "NA"
    for lo, hi, name in cuts:
        if lo <= float(value) < hi:
            return name
    return "NA"


def _line_bucket(x: Optional[float]) -> str:
    if x is None:
        return "NA"
    if x <= -3:
        return "<=-3"
    if x <= -1:
        return "[-3,-1]"
    if x < 0:
        return "(-1,0)"
    if x == 0:
        return "0"
    if x <= 1:
        return "(0,1]"
    return ">1"


def _top_events(rows: Sequence[Dict[str, Any]], n: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ev = _events(rows)
    pos = sorted(ev, key=lambda e: e["pnl"], reverse=True)[:n]
    neg = sorted(ev, key=lambda e: e["pnl"])[:n]
    return pos, neg


def _rolling_without_top3(rows: Sequence[Dict[str, Any]], windows: Sequence[int]) -> Dict[str, List[Dict[str, Any]]]:
    rows2 = sorted(rows, key=lambda r: (r.get("day") or "", str(r.get("order_id") or "")))
    out: Dict[str, List[Dict[str, Any]]] = {}
    for w in windows:
        vals = []
        if len(rows2) < w:
            out[str(w)] = vals
            continue
        for i in range(w, len(rows2) + 1):
            part = rows2[i - w : i]
            ev = _events(part)
            vals.append({"end_day": part[-1]["day"], "roi_sem_top3": _roi_without_top(ev, 3), "roi": _roi(part)[0]})
        out[str(w)] = vals
    return out


def _render_md(path: Path, out: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Analise avancada de drivers - Back Pre / slippage executor\n")
    lines.append("## Escopo\n")
    lines.append(f"- Periodo: `{out['params']['start_day']}` a `{out['params']['end_day']}`")
    lines.append("- Universo: execucoes Back Pre LIVE_OK reconciliadas com P&L.")
    lines.append("- Slippage: `result.raw.value_sizing.slippage_pre_pct` do executor.\n")
    lines.append("## ROI sem Top-k por evento\n")
    lines.append("| k removido | ROI |")
    lines.append("|---:|---:|")
    for k, v in out["roi_without_topk"].items():
        lines.append(f"| Top-{k} | {_fmt(v,2,True)} |")
    lines.append("\n## Top eventos positivos\n")
    lines.append("| Evento | Liga | Dia | N | P&L | ROI | avg slip | avg odd | avg AH | avg limit | Driver provavel |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for e in out["top_events_positive"]:
        lines.append(f"| {e['event_id']} | {e.get('league','')} | {e.get('day','')} | {e['n']} | {_fmt(e['pnl'],2)} | {_fmt(e['roi'],2,True)} | {_fmt(e.get('avg_slip'),2,True)} | {_fmt(e.get('avg_odd'),2)} | {_fmt(e.get('avg_line'),2)} | {_fmt(e.get('avg_limit'),1)} | {e['driver']} |")
    lines.append("\n## Top eventos negativos\n")
    lines.append("| Evento | Liga | Dia | N | P&L | ROI | avg slip | avg odd | avg AH | avg limit | Driver provavel |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for e in out["top_events_negative"]:
        lines.append(f"| {e['event_id']} | {e.get('league','')} | {e.get('day','')} | {e['n']} | {_fmt(e['pnl'],2)} | {_fmt(e['roi'],2,True)} | {_fmt(e.get('avg_slip'),2,True)} | {_fmt(e.get('avg_odd'),2)} | {_fmt(e.get('avg_line'),2)} | {_fmt(e.get('avg_limit'),1)} | {e['driver']} |")
    lines.append("")
    for title, data in [
        ("M2 por subsegmento", out["m2_subsegments"]),
        ("5Ms por bucket de slippage", out["five_ms_slippage_bucket"]),
        ("5Ms por liga", out["five_ms_league"]),
        ("5Ms por faixa de odd", out["five_ms_odd_bucket"]),
    ]:
        lines.append(f"## {title}\n")
        lines.append("| Segmento | N | Eventos | ROI | Score | M1 | M2 | M3 | M4 | M5 |")
        lines.append("|---|---:|---:|---:|---:|---|---|---|---|---|")
        for name, m in data:
            if m.get("error"):
                continue
            lines.append(f"| {name} | {m['n_bets']} | {m['n_events']} | {_fmt(m['roi'],2,True)} | {m['score']}/5 | {m['M1']['status']} | {m['M2']['status']} | {m['M3']['status']} | {m['M4']['status']} | {m['M5']['status']} |")
        lines.append("")
    lines.append("## Slippage x odd x linha AH\n")
    lines.append("| Slippage | Odd | Linha AH | N | ROI | P&L | Stake |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for r in out["slip_odd_line"]:
        lines.append(f"| {r['slip_bucket']} | {r['odd_bucket']} | {r['line_bucket']} | {r['n']} | {_fmt(r['roi'],2,True)} | {_fmt(r['pnl'],2)} | {_fmt(r['stake'],2)} |")
    lines.append("\n## Bootstrap por evento - pos sem World Cup\n")
    b = out["bootstrap_pos_no_wc"]
    lines.append(f"- N apostas: {b['n_bets']} | N eventos: {b['n_events']} | ROI observado: {_fmt(b['roi'],2,True)}")
    lines.append(f"- CI90 bootstrap evento: [{_fmt(b['ci90_lo'],2,True)}, {_fmt(b['ci90_hi'],2,True)}]")
    lines.append(f"- p_perm: {_fmt(b['p_perm'],4)}\n")
    lines.append("## Rolling sem Top-3\n")
    lines.append("| Janela | N pontos | min ROI sem Top-3 | mediana | max | ultimo |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for w, s in out["rolling_summary"].items():
        lines.append(f"| {w} | {s['n']} | {_fmt(s['min'],2,True)} | {_fmt(s['median'],2,True)} | {_fmt(s['max'],2,True)} | {_fmt(s['last'],2,True)} |")
    lines.append("\n## Capacidade: stake/max stake/liquidez vs ROI\n")
    for title, data in [("Stake ratio", out["capacity_stake_ratio"]), ("Liquidez", out["capacity_liquidity"])]:
        lines.append(f"### {title}\n")
        lines.append("| Bucket | N | ROI | P&L | Stake |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in data:
            lines.append(f"| {r['bucket']} | {r['n']} | {_fmt(r['roi'],2,True)} | {_fmt(r['pnl'],2)} | {_fmt(r['stake'],2)} |")
        lines.append("")
    lines.append("## Leitura executiva\n")
    lines.append("- Eventos extremos ainda importam: ROI sem Top-k deve ser usado para validar robustez antes de escalar stake.")
    lines.append("- Slippage negativo e odds/linhas interagem; a regra `slippage < 0` e prudencial, mas thresholds mais duros precisam de mais N.")
    lines.append("- Para capacidade, buckets de liquidez/stake ajudam a separar edge economico de execucao em mercados rasos.")
    path.write_text("\n".join(lines), encoding="utf-8")


def _render_pdf(md: Path, pdf: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=6.4, leading=7.5))
    styles.add(ParagraphStyle(name="BodyX", parent=styles["BodyText"], fontSize=8.5, leading=10.5))
    doc = SimpleDocTemplate(str(pdf), pagesize=landscape(A4), rightMargin=0.7 * cm, leftMargin=0.7 * cm, topMargin=0.7 * cm, bottomMargin=0.7 * cm)
    story: List[Any] = []

    def esc(x: str) -> str:
        return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    lines = md.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if not ln:
            story.append(Spacer(1, 4))
            i += 1
            continue
        if ln.startswith("# "):
            story.append(Paragraph(esc(ln[2:]), styles["Title"]))
            i += 1
            continue
        if ln.startswith("## "):
            story.append(Paragraph(esc(ln[3:]), styles["Heading2"]))
            i += 1
            continue
        if ln.startswith("### "):
            story.append(Paragraph(esc(ln[4:]), styles["Heading3"]))
            i += 1
            continue
        if ln.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            parsed = [[c.strip() for c in row.strip("|").split("|")] for row in block]
            if len(parsed) >= 2:
                header = parsed[0]
                rows = parsed[2:]
                data = [[Paragraph(esc(c), styles["Small"]) for c in header]]
                data += [[Paragraph(esc(c), styles["Small"]) for c in r] for r in rows]
                tbl = Table(data, repeatRows=1)
                tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.2, colors.HexColor("#d1d5db")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 2),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                    ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
                ]))
                story.append(tbl)
                story.append(Spacer(1, 6))
            continue
        story.append(Paragraph(esc(ln[2:] if ln.startswith("- ") else ln), styles["BodyX"]))
        i += 1
    doc.build(story)


def _driver(e: Dict[str, Any]) -> str:
    parts = []
    if e.get("avg_slip") is not None:
        parts.append("slip favoravel" if e["avg_slip"] < 0 else "slip adverso")
    if e.get("avg_odd") is not None:
        parts.append("odd alta" if e["avg_odd"] >= 2.2 else ("odd baixa" if e["avg_odd"] < 1.8 else "odd media"))
    if e.get("avg_line") is not None:
        parts.append("linha extrema" if abs(e["avg_line"]) >= 2.5 else "linha moderada")
    if e.get("avg_limit") is not None:
        parts.append("liquidez alta" if e["avg_limit"] >= 1000 else ("liquidez baixa" if e["avg_limit"] < 100 else "liquidez media"))
    return ", ".join(parts[:4])


def _group_metrics(rows: List[Dict[str, Any]], key_fn, iters: int) -> List[Tuple[str, Dict[str, Any]]]:
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[str(key_fn(r))].append(r)
    out = [(k, _five_ms(v, iters, 100 + i)) for i, (k, v) in enumerate(sorted(by.items()))]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-day", default="2026-04-19")
    ap.add_argument("--end-day", default="2026-06-27")
    ap.add_argument("--split-day", default="2026-05-25")
    ap.add_argument("--database-url", default="")
    ap.add_argument("--executor-jsonl", default="/home/betbot/Bets/betinasia_bot/logs/executor_live.jsonl")
    ap.add_argument("--balance-csv", default="")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-pdf", required=True)
    args = ap.parse_args()
    db = _resolve_db(args.database_url)
    bal = Path(args.balance_csv) if args.balance_csv else _latest_balance_csv()
    rows = _load_rows(Path(args.executor_jsonl), bal, db, args.start_day, args.end_day)
    rows_slip_neg = [r for r in rows if r["slippage"] < 0]
    ev = _events(rows_slip_neg)
    pos, neg = _top_events(rows_slip_neg, 12)
    for e in pos + neg:
        e["driver"] = _driver(e)
    roi_without = {str(k): _roi_without_top(ev, k) for k in [1, 3, 5, 10]}
    aliases = {"fifaworldcup", "worldcup", "clubworldcup", "fifaclubworldcup", "worldchampionship"}
    def wc(r): return _norm(r.get("league", "")) in aliases
    def subseg(r):
        return ("pre" if r["day"] < args.split_day else "pos") + ("_wc" if wc(r) else "_no_wc")
    slip_cuts = [(-999, -5, "<-5"), (-5, -3, "[-5,-3)"), (-3, -2, "[-3,-2)"), (-2, -1, "[-2,-1)"), (-1, -0.5, "[-1,-0.5)"), (-0.5, 0, "[-0.5,0)"), (0, 0.5, "[0,0.5)"), (0.5, 999, ">=0.5")]
    odd_cuts = [(0, 1.7, "<1.7"), (1.7, 1.9, "1.7-1.9"), (1.9, 2.1, "1.9-2.1"), (2.1, 2.4, "2.1-2.4"), (2.4, 999, ">=2.4")]
    out: Dict[str, Any] = {
        "params": vars(args),
        "n_rows_all": len(rows),
        "n_rows_slip_neg": len(rows_slip_neg),
        "roi_without_topk": roi_without,
        "top_events_positive": pos,
        "top_events_negative": neg,
        "m2_subsegments": _group_metrics(rows_slip_neg, subseg, args.iters),
        "five_ms_slippage_bucket": _group_metrics(rows, lambda r: _bucket(r["slippage"], slip_cuts), args.iters),
        "five_ms_league": [x for x in _group_metrics(rows_slip_neg, lambda r: r.get("league") or "NA", args.iters) if x[1].get("n_bets", 0) >= 10],
        "five_ms_odd_bucket": _group_metrics(rows_slip_neg, lambda r: _bucket(r.get("odd_at_decision"), odd_cuts), args.iters),
    }
    by_combo: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_combo[(
            _bucket(r["slippage"], slip_cuts),
            _bucket(r.get("odd_at_decision"), odd_cuts),
            _line_bucket(r.get("line_num")),
        )].append(r)
    combo_rows = []
    for (sb, ob, lb), arr in by_combo.items():
        if len(arr) < 5:
            continue
        rr, pp, ss = _roi(arr)
        combo_rows.append({"slip_bucket": sb, "odd_bucket": ob, "line_bucket": lb, "n": len(arr), "roi": rr, "pnl": pp, "stake": ss})
    out["slip_odd_line"] = sorted(combo_rows, key=lambda x: (x["slip_bucket"], x["odd_bucket"], x["line_bucket"]))
    pos_no_wc = [r for r in rows_slip_neg if r["day"] >= args.split_day and not wc(r)]
    ev_pos_no = _events(pos_no_wc)
    ci_lo, ci_hi = _bootstrap_ci(ev_pos_no, args.iters * 2, 555)
    out["bootstrap_pos_no_wc"] = {
        "n_bets": len(pos_no_wc),
        "n_events": len(ev_pos_no),
        "roi": _roi(pos_no_wc)[0],
        "ci90_lo": ci_lo,
        "ci90_hi": ci_hi,
        "p_perm": _perm_p(ev_pos_no, args.iters * 2, 556),
    }
    rolling = _rolling_without_top3(rows_slip_neg, [30, 50, 100])
    summary = {}
    for w, vals in rolling.items():
        xs = [v["roi_sem_top3"] for v in vals if v["roi_sem_top3"] is not None]
        if xs:
            xs2 = sorted(xs)
            summary[w] = {"n": len(xs), "min": min(xs), "median": xs2[len(xs2)//2], "max": max(xs), "last": xs[-1]}
        else:
            summary[w] = {"n": 0, "min": None, "median": None, "max": None, "last": None}
    out["rolling_summary"] = summary
    def cap_bucket(r):
        ratio = float(r["stake"]) / float(r.get("max_stake") or r["stake"])
        if ratio < 0.25: return "<25%"
        if ratio < 0.5: return "25-50%"
        if ratio < 0.9: return "50-90%"
        return ">=90%"
    def liq_bucket(r):
        x = r.get("limit_final")
        if x is None: return "NA"
        if x < 100: return "<100"
        if x < 500: return "100-500"
        if x < 1000: return "500-1000"
        return ">=1000"
    for key, fn in [("capacity_stake_ratio", cap_bucket), ("capacity_liquidity", liq_bucket)]:
        rows_out = []
        by = defaultdict(list)
        for r in rows_slip_neg:
            by[fn(r)].append(r)
        for b, arr in sorted(by.items()):
            rr, pp, ss = _roi(arr)
            rows_out.append({"bucket": b, "n": len(arr), "roi": rr, "pnl": pp, "stake": ss})
        out[key] = rows_out
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _render_md(Path(args.out_md), out)
    _render_pdf(Path(args.out_md), Path(args.out_pdf))
    print(f"[OK] rows_all={len(rows)} rows_slip_neg={len(rows_slip_neg)}")
    print(f"[OK] JSON={args.out_json}")
    print(f"[OK] MD={args.out_md}")
    print(f"[OK] PDF={args.out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
