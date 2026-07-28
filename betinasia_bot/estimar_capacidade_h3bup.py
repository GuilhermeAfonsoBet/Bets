#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimativa de tamanho de mercado, capacidade e expectativa economica para H3BUP.

Regra H3BUP:
- Back Pre
- LIVE_OK
- P&L accounting real por order_id
- slippage_pre_pct < 0
- odd 1.85..2.15
- capacidade/liquidez/limit > 100
- cluster estatistico por event_id
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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from estudo_robusto_backpre_hipoteses import (
    WORLD_CUP_ALIASES,
    _drawdown,
    _estimate_wc_start,
    _events,
    _fmt,
    _latest_balance,
    _load_rows,
    _resolve_db,
    _roi,
    _roi_without_topk,
    _walk_forward,
)


def _norm(s: Any) -> str:
    return "".join(ch for ch in str(s or "").lower() if ch.isalnum())


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


def _percentile(xs: Sequence[float], q: float) -> Optional[float]:
    if not xs:
        return None
    arr = sorted(float(x) for x in xs)
    idx = max(0, min(len(arr) - 1, int(round((len(arr) - 1) * q))))
    return arr[idx]


def _h3bup(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        odd = r.get("odd")
        slip = r.get("slippage_pre_pct")
        cap = r.get("capacity")
        if slip is None or not (float(slip) < 0):
            continue
        if odd is None or not (1.85 <= float(odd) <= 2.15):
            continue
        if cap is None or not (float(cap) > 100):
            continue
        out.append(r)
    return out


def _calendar_regimes(rows: Sequence[Dict[str, Any]], wc_start: str, wc_alt: str) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "normal_pre_world_cup": [r for r in rows if r["day"] < wc_start and not r.get("is_world_cup")],
        "world_cup": [r for r in rows if r["day"] >= wc_start and r.get("is_world_cup")],
        "non_world_cup_during_world_cup": [r for r in rows if r["day"] >= wc_start and not r.get("is_world_cup")],
        f"world_cup_alt_start_{wc_alt}": [r for r in rows if r["day"] >= wc_alt and r.get("is_world_cup")],
        f"non_wc_alt_start_{wc_alt}": [r for r in rows if r["day"] >= wc_alt and not r.get("is_world_cup")],
    }


def _active_days(rows: Sequence[Dict[str, Any]]) -> int:
    return len({r["day"] for r in rows})


def _calendar_days(rows: Sequence[Dict[str, Any]]) -> int:
    if not rows:
        return 0
    days = sorted({r["day"] for r in rows})
    d0 = datetime.fromisoformat(days[0]).date()
    d1 = datetime.fromisoformat(days[-1]).date()
    return (d1 - d0).days + 1


def _turnover_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    roi, pnl, stake = _roi(rows)
    ev = _events(rows)
    stks = [float(r["stake"]) for r in rows]
    caps = [float(r["capacity"]) for r in rows if r.get("capacity") is not None]
    active = _active_days(rows)
    cal = _calendar_days(rows)
    by_day = defaultdict(float)
    for r in rows:
        by_day[r["day"]] += float(r["stake"])
    return {
        "n_bets": len(rows),
        "n_events": len(ev),
        "calendar_days": cal,
        "active_days": active,
        "uptime_active_day_ratio": active / cal if cal else None,
        "stake": stake,
        "pnl": pnl,
        "roi": roi,
        "stake_mean": statistics.mean(stks) if stks else None,
        "stake_median": statistics.median(stks) if stks else None,
        "stake_p25": _percentile(stks, 0.25),
        "stake_p75": _percentile(stks, 0.75),
        "capacity_mean": statistics.mean(caps) if caps else None,
        "capacity_median": statistics.median(caps) if caps else None,
        "bets_per_active_day": len(rows) / active if active else None,
        "events_per_active_day": len(ev) / active if active else None,
        "turnover_per_active_day": stake / active if active else None,
        "turnover_monthly_active_runrate": (stake / active * 30.0) if active else None,
        "turnover_monthly_calendar_runrate": (stake / cal * 30.0) if cal else None,
        "pnl_monthly_active_runrate": (pnl / active * 30.0) if active else None,
        "roi_without_top": {str(k): _roi_without_topk(ev, k) for k in [1, 3, 5, 10]},
    }


def _concentration(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ev = _events(rows)
    roi, pnl, stake = _roi(rows)
    top = sorted(ev, key=lambda e: e["pnl"], reverse=True)
    by_league = defaultdict(list)
    by_day = defaultdict(list)
    by_odd = defaultdict(list)
    by_cap = defaultdict(list)
    for r in rows:
        by_league[r.get("league") or "NA"].append(r)
        by_day[r["day"]].append(r)
        odd = r.get("odd")
        if odd is None:
            ob = "NA"
        elif odd < 1.9:
            ob = "<1.9"
        elif odd <= 2.1:
            ob = "1.9-2.1"
        else:
            ob = ">2.1"
        by_odd[ob].append(r)
        cap = r.get("capacity")
        if cap is None:
            cb = "NA"
        elif cap < 250:
            cb = "100-250"
        elif cap < 500:
            cb = "250-500"
        elif cap < 1000:
            cb = "500-1000"
        else:
            cb = ">1000"
        by_cap[cb].append(r)
    def share_top(k):
        v = sum(e["pnl"] for e in top[:k])
        return 100.0 * v / pnl if pnl else None
    def group_table(by):
        out = []
        for k, rs in sorted(by.items(), key=lambda kv: len(kv[1]), reverse=True):
            rr, pp, ss = _roi(rs)
            out.append({"bucket": k, "n": len(rs), "stake": ss, "pnl": pp, "roi": rr, "share_stake_pct": 100.0 * ss / stake if stake else None})
        return out
    return {
        "top_event_pct_pnl": {str(k): share_top(k) for k in [1, 3, 5, 10]},
        "roi_without_top": {str(k): _roi_without_topk(ev, k) for k in [1, 3, 5, 10]},
        "league": group_table(by_league)[:20],
        "day": group_table(by_day)[:20],
        "odd": group_table(by_odd),
        "capacity": group_table(by_cap),
    }


def _capacity(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    buckets = defaultdict(list)
    for r in rows:
        cap = r.get("capacity")
        if cap is None:
            b = "NA"
        elif cap < 250:
            b = "100-250"
        elif cap < 500:
            b = "250-500"
        elif cap < 1000:
            b = "500-1000"
        else:
            b = ">1000"
        buckets[b].append(r)
    out = []
    for b, rs in sorted(buckets.items()):
        rr, pp, ss = _roi(rs)
        out.append({"bucket": b, "n_bets": len(rs), "n_events": len(_events(rs)), "stake": ss, "pnl": pp, "roi": rr})
    xs = [(float(r["capacity"]), float(r["roi"])) for r in rows if r.get("capacity") is not None and r.get("roi") is not None]
    corr = None
    if len(xs) > 2:
        mx = statistics.mean(x for x, _ in xs)
        my = statistics.mean(y for _, y in xs)
        cov = sum((x - mx) * (y - my) for x, y in xs)
        vx = sum((x - mx) ** 2 for x, _ in xs)
        vy = sum((y - my) ** 2 for _, y in xs)
        corr = cov / math.sqrt(vx * vy) if vx and vy else None
    return {"buckets": out, "capacity_roi_corr": corr}


def _drawdowns(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ev = _events(rows)
    by_day = defaultdict(float)
    for r in rows:
        by_day[r["day"]] += float(r["pnl"])
    return {
        "by_bet": _drawdown([(r["created_at"], r["pnl"]) for r in sorted(rows, key=lambda x: (x["created_at"], x["order_id"]))]),
        "by_event": _drawdown([(e["day"], e["pnl"]) for e in sorted(ev, key=lambda x: (x["day"], x["event_id"]))]),
        "by_day": _drawdown(sorted(by_day.items())),
    }


def _wf_median_roi(rows: Sequence[Dict[str, Any]]) -> Optional[float]:
    # Reusa janelas 21/7; regra ja filtrada, entao so mede janelas.
    days = sorted({r["day"] for r in rows})
    vals = []
    if len(days) < 28:
        return None
    for s in range(0, len(days) - 28 + 1, 7):
        test_days = set(days[s + 21 : s + 28])
        te = [r for r in rows if r["day"] in test_days]
        if te:
            vals.append(_roi(te)[0])
    return statistics.median(vals) if vals else None


def _oos_roi(rows: Sequence[Dict[str, Any]], cut: str) -> Optional[float]:
    te = [r for r in rows if r["day"] >= cut]
    return _roi(te)[0] if te else None


def _roi_scenarios(rows: Sequence[Dict[str, Any]], oos_cut: str) -> Dict[str, Any]:
    obs = _roi(rows)[0]
    no3 = _roi_without_topk(_events(rows), 3)
    oos = _oos_roi(rows, oos_cut)
    wf = _wf_median_roi(rows)
    vals = [x for x in [no3, oos, wf, obs * 0.5] if x is not None and not math.isnan(x)]
    conservative = min(vals) if vals else obs * 0.5
    base_vals = [x for x in [obs, oos, no3, wf] if x is not None and not math.isnan(x)]
    base = statistics.mean(base_vals) if base_vals else obs
    # Prob simples por bootstrap do ROI >0
    ev = _events(rows)
    prob = None
    if ev:
        rng = random.Random(123)
        rois = []
        for _ in range(3000):
            smp = [ev[rng.randrange(len(ev))] for __ in range(len(ev))]
            st = sum(e["stake"] for e in smp)
            rois.append(100.0 * sum(e["pnl"] for e in smp) / st if st else 0)
        prob = sum(1 for x in rois if x > 0) / len(rois)
    optimistic = obs * 0.75 if (no3 is not None and no3 > 0 and oos is not None and oos > 0 and prob is not None and prob > 0.70) else base
    return {"observed": obs, "roi_without_top3": no3, "oos": oos, "walk_forward_median": wf, "prob_roi_gt_0": prob, "conservative": conservative, "base": base, "optimistic_controlled": optimistic}


def _monthly_sim(rows: Sequence[Dict[str, Any]], monthly_events: int, monthly_turnover: float, roi_assumption: float, iters: int, seed: int) -> Dict[str, Any]:
    ev = _events(rows)
    if not ev or monthly_events <= 0:
        return {}
    rng = random.Random(seed)
    hist_roi = _roi(rows)[0]
    # scale event pnl by ratio between target ROI and historical ROI where possible.
    roi_scale = (roi_assumption / hist_roi) if hist_roi and not math.isnan(hist_roi) and abs(hist_roi) > 1e-9 else 1.0
    vals = []
    rois = []
    dds = []
    for _ in range(iters):
        smp = [ev[rng.randrange(len(ev))] for __ in range(monthly_events)]
        st_hist = sum(e["stake"] for e in smp)
        pnl_hist = sum(e["pnl"] for e in smp) * roi_scale
        # normaliza para turnover mensal alvo
        scale = monthly_turnover / st_hist if st_hist > 0 else 0
        pnl = pnl_hist * scale
        vals.append(pnl)
        rois.append(100.0 * pnl / monthly_turnover if monthly_turnover > 0 else 0)
        eq = 0
        peak = 0
        dd = 0
        for e in smp:
            eq += e["pnl"] * roi_scale * scale
            peak = max(peak, eq)
            dd = max(dd, peak - eq)
        dds.append(dd)
    vals_sorted = sorted(vals)
    rois_sorted = sorted(rois)
    dds_sorted = sorted(dds)
    return {
        "events_per_month": monthly_events,
        "turnover_monthly": monthly_turnover,
        "roi_assumption": roi_assumption,
        "pnl_mean": statistics.mean(vals),
        "pnl_median": statistics.median(vals),
        "p10": _percentile(vals_sorted, 0.10),
        "p25": _percentile(vals_sorted, 0.25),
        "p75": _percentile(vals_sorted, 0.75),
        "p90": _percentile(vals_sorted, 0.90),
        "prob_month_negative": sum(1 for x in vals if x < 0) / len(vals),
        "p95_loss": _percentile(vals_sorted, 0.05),
        "roi_monthly_median": statistics.median(rois),
        "drawdown_mean": statistics.mean(dds),
        "drawdown_p95": _percentile(dds_sorted, 0.95),
        "capital_at_3x_turnover": monthly_turnover / 3.0,
    }


def _volume_scenarios(rows: Sequence[Dict[str, Any]], regimes: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    base = _turnover_summary(rows)
    # oportunidades bloqueadas sao contadas separadamente; sem contrafactual de ROI.
    active_turn = base["turnover_monthly_active_runrate"] or 0
    cal_turn = base["turnover_monthly_calendar_runrate"] or 0
    active_bets = (base["bets_per_active_day"] or 0) * 30
    active_events = (base["events_per_active_day"] or 0) * 30
    # Cenários: conservador = calendário observado; base = 75% do run-rate ativo; otimista = run-rate ativo * 1.15 limitado.
    return {
        "conservative": {
            "bets_month": len(rows) / max(1, base["calendar_days"]) * 30,
            "events_month": len(_events(rows)) / max(1, base["calendar_days"]) * 30,
            "turnover_month": cal_turn,
            "stake_mean": base["stake_mean"],
            "stake_median": base["stake_median"],
        },
        "base": {
            "bets_month": active_bets * 0.75,
            "events_month": active_events * 0.75,
            "turnover_month": active_turn * 0.75,
            "stake_mean": base["stake_mean"],
            "stake_median": base["stake_median"],
        },
        "optimistic_controlled": {
            "bets_month": active_bets * 1.15,
            "events_month": active_events * 1.15,
            "turnover_month": active_turn * 1.15,
            "stake_mean": base["stake_mean"],
            "stake_median": base["stake_median"],
        },
    }


def _blocked_counts(jsonl: Path, start: str, end: str) -> Dict[str, Any]:
    counts = Counter()
    if not jsonl.exists():
        return {"available": False}
    for ln in jsonl.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "CAP_BLOCKED" not in ln and "BLOCK" not in ln and "REJECT" not in ln:
            continue
        try:
            o = json.loads(ln)
        except Exception:
            continue
        req = o.get("request") or {}
        res = o.get("result") or {}
        day = str(res.get("created_at") or req.get("created_at") or "")[:10]
        if day < start or day > end:
            continue
        err = str(res.get("error") or "")
        up = err.upper()
        if "SLIPPAGE" in up:
            key = "blocked_slippage"
        elif "LEAGUE" in up or "POLICY" in up:
            key = "blocked_league_or_policy"
        elif "BANK" in up or "BALANCE" in up:
            key = "blocked_bankroll"
        elif "SESSION" in up or "API" in up or "PMM" in up or "STALE" in up:
            key = "operational_or_api"
        else:
            key = "other_blocked"
        counts[key] += 1
    return {"available": True, "counts": dict(counts), "contrafactual_pnl_available": False}


def _render_md(out: Dict[str, Any], md: Path) -> None:
    lines = []
    lines.append("# Estimativa de mercado, capacidade e expectativa economica - H3BUP\n")
    lines.append("## Sumario executivo\n")
    lines.append("- Regra H3BUP: Back Pre LIVE_OK, P&L accounting, slippage<0, odd 1.85-2.15, capacidade>100.")
    lines.append("- Relatorio estima tamanho/capacidade/risco; nao recomenda aumento de exposicao ou stake.")
    lines.append("- World Cup e policy/filtros de liga foram tratados como regimes que distorcem volume e composicao.\n")
    lines.append("## Base e tamanho historico\n")
    b = out["base_summary"]
    lines.append(f"- N apostas H3BUP: {b['n_bets']} | N eventos: {b['n_events']} | ROI observado: {_fmt(b['roi'],2,True)} | P&L: {_fmt(b['pnl'],2)} | Stake: {_fmt(b['stake'],2)}")
    lines.append(f"- Dias corridos: {b['calendar_days']} | Dias ativos: {b['active_days']} | Uptime ativo aprox.: {_fmt(100*b['uptime_active_day_ratio'] if b['uptime_active_day_ratio'] is not None else None,1,True)}")
    lines.append(f"- Apostas/dia ativo: {_fmt(b['bets_per_active_day'],2)} | Eventos/dia ativo: {_fmt(b['events_per_active_day'],2)} | Turnover/dia ativo: {_fmt(b['turnover_per_active_day'],2)}\n")
    lines.append("## Regimes de calendario\n")
    lines.append("| Regime | N | Eventos | Dias ativos | Turnover mensal run-rate | ROI | ROI sem Top-3 | P&L mensal eq. |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, s in out["regimes"].items():
        lines.append(f"| {name} | {s['n_bets']} | {s['n_events']} | {s['active_days']} | {_fmt(s['turnover_monthly_active_runrate'],2)} | {_fmt(s['roi'],2,True)} | {_fmt(s['roi_without_top']['3'],2,True)} | {_fmt(s['pnl_monthly_active_runrate'],2)} |")
    lines.append("\n## Cenarios de volume mensal e capital teorico\n")
    lines.append("| Cenario | Apostas/mes | Eventos/mes | Turnover/mes | Capital medio (giro 3x) | ROI cons. | ROI base | ROI ot. | P&L cons. | P&L base | P&L ot. |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, s in out["volume_scenarios"].items():
        t = s["turnover_month"]
        rc = out["roi_scenarios"]["conservative"]
        rb = out["roi_scenarios"]["base"]
        ro = out["roi_scenarios"]["optimistic_controlled"]
        lines.append(f"| {name} | {_fmt(s['bets_month'],1)} | {_fmt(s['events_month'],1)} | {_fmt(t,2)} | {_fmt(t/3,2)} | {_fmt(rc,2,True)} | {_fmt(rb,2,True)} | {_fmt(ro,2,True)} | {_fmt(t*rc/100,2)} | {_fmt(t*rb/100,2)} | {_fmt(t*ro/100,2)} |")
    lines.append("\n## ROI esperado\n")
    rs = out["roi_scenarios"]
    lines.append(f"- Observado: {_fmt(rs['observed'],2,True)}")
    lines.append(f"- ROI sem Top-3: {_fmt(rs['roi_without_top3'],2,True)}")
    lines.append(f"- OOS: {_fmt(rs['oos'],2,True)}")
    lines.append(f"- Walk-forward mediano: {_fmt(rs['walk_forward_median'],2,True)}")
    lines.append(f"- Conservador: {_fmt(rs['conservative'],2,True)} | Base: {_fmt(rs['base'],2,True)} | Otimista controlado: {_fmt(rs['optimistic_controlled'],2,True)}\n")
    lines.append("## Simulacao mensal por evento\n")
    lines.append("| Cenario | ROI usado | Turnover | P&L medio | Mediana | p10 | p25 | p75 | p90 | Prob mes negativo | p95 perda | DD p95 | Capital 3x |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key, sim in out["monthly_simulations"].items():
        lines.append(f"| {key} | {_fmt(sim['roi_assumption'],2,True)} | {_fmt(sim['turnover_monthly'],2)} | {_fmt(sim['pnl_mean'],2)} | {_fmt(sim['pnl_median'],2)} | {_fmt(sim['p10'],2)} | {_fmt(sim['p25'],2)} | {_fmt(sim['p75'],2)} | {_fmt(sim['p90'],2)} | {_fmt(100*sim['prob_month_negative'],1,True)} | {_fmt(sim['p95_loss'],2)} | {_fmt(sim['drawdown_p95'],2)} | {_fmt(sim['capital_at_3x_turnover'],2)} |")
    lines.append("\n## Capacidade por liquidez/max stake\n")
    lines.append("| Bucket | N | Eventos | Stake | P&L | ROI |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in out["capacity"]["buckets"]:
        lines.append(f"| {r['bucket']} | {r['n_bets']} | {r['n_events']} | {_fmt(r['stake'],2)} | {_fmt(r['pnl'],2)} | {_fmt(r['roi'],2,True)} |")
    lines.append(f"\n- Correlacao capacidade vs ROI por aposta: {_fmt(out['capacity']['capacity_roi_corr'],4)}\n")
    lines.append("## Concentracao\n")
    c = out["concentration"]
    lines.append("| Top-k | % do P&L | ROI sem Top-k |")
    lines.append("|---|---:|---:|")
    for k in ["1", "3", "5", "10"]:
        lines.append(f"| Top-{k} | {_fmt(c['top_event_pct_pnl'][k],1,True)} | {_fmt(c['roi_without_top'][k],2,True)} |")
    lines.append("\n### Concentração por liga\n")
    lines.append("| Liga | N | Stake share | ROI | P&L |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in c["league"][:12]:
        lines.append(f"| {r['bucket']} | {r['n']} | {_fmt(r['share_stake_pct'],1,True)} | {_fmt(r['roi'],2,True)} | {_fmt(r['pnl'],2)} |")
    lines.append("\n## Drawdown e capital de risco\n")
    dd = out["drawdowns"]
    lines.append(f"- Max DD por aposta: {_fmt(dd['by_bet']['max_drawdown'],2)} | recovery factor: {_fmt(dd['by_bet']['recovery_factor'],2)}")
    lines.append(f"- Max DD por evento: {_fmt(dd['by_event']['max_drawdown'],2)} | recovery factor: {_fmt(dd['by_event']['recovery_factor'],2)}")
    lines.append(f"- Max DD por dia: {_fmt(dd['by_day']['max_drawdown'],2)} | recovery factor: {_fmt(dd['by_day']['recovery_factor'],2)}")
    lines.append("\n## Oportunidades bloqueadas / perdidas\n")
    lines.append(json.dumps(out["blocked_counts"], ensure_ascii=False, indent=2))
    lines.append("\n## Respostas objetivas\n")
    for k, v in out["answers"].items():
        lines.append(f"- **{k}**: {v}")
    md.write_text("\n".join(lines), encoding="utf-8")


def _render_pdf(md: Path, pdf: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import landscape, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=6.1, leading=7.1))
    styles.add(ParagraphStyle(name="BodyX", parent=styles["BodyText"], fontSize=8.2, leading=10))
    doc = SimpleDocTemplate(str(pdf), pagesize=landscape(A4), leftMargin=18, rightMargin=18, topMargin=18, bottomMargin=18)
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
                    ("TOPPADDING", (0,0), (-1,-1), 1.1),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 1.1),
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
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-pdf", required=True)
    args = ap.parse_args()
    end = args.end_day or datetime.now(timezone.utc).date().isoformat()
    db = _resolve_db(args.database_url)
    bal = Path(args.balance_csv) if args.balance_csv else _latest_balance()
    all_rows = _load_rows(Path(args.executor_jsonl), bal, db, args.start_day, end)
    rows = _h3bup(all_rows)
    if not rows:
        raise SystemExit("sem linhas H3BUP")
    wc_reg = _estimate_wc_start(all_rows)
    wc_start = (wc_reg.get("0.5") or {}).get("date") or "2026-06-04"
    regimes = {k: _h3bup(v) for k, v in _calendar_regimes(all_rows, wc_start, "2026-06-05").items()}
    reg_summ = {k: _turnover_summary(v) for k, v in regimes.items()}
    base = _turnover_summary(rows)
    roi_s = _roi_scenarios(rows, "2026-06-01")
    vol = _volume_scenarios(rows, regimes)
    sims = {}
    for scen, v in vol.items():
        events_month = max(1, int(round(v["events_month"] or 0)))
        t = float(v["turnover_month"] or 0)
        for roi_name in ["conservative", "base", "optimistic_controlled"]:
            sims[f"{scen}_{roi_name}"] = _monthly_sim(rows, events_month, t, float(roi_s[roi_name]), args.iters, hash(scen + roi_name) % 100000)
    out = {
        "params": {"start_day": args.start_day, "end_day": end, "balance_csv": str(bal), "n_all_backpre": len(all_rows), "n_h3bup": len(rows), "wc_start": wc_start},
        "world_cup_regime": wc_reg,
        "base_summary": base,
        "regimes": reg_summ,
        "volume_scenarios": vol,
        "roi_scenarios": roi_s,
        "monthly_simulations": sims,
        "capacity": _capacity(rows),
        "concentration": _concentration(rows),
        "drawdowns": _drawdowns(rows),
        "blocked_counts": _blocked_counts(Path(args.executor_jsonl), args.start_day, end),
    }
    out["answers"] = {
        "Tamanho mensal provavel": "Usar cenarios conservador/base/otimista do relatorio; a estimativa estrutural deve privilegiar mercado normal sem World Cup.",
        "Historico subestima volume por uptime baixo?": "Provavelmente sim parcialmente; run-rate por dia ativo e maior que run-rate calendario, mas oportunidades bloqueadas nao tem P&L contrafactual.",
        "World Cup distorce volume normal?": "Sim. O regime World Cup inicia em torno de 2026-06-04 e muda composicao de ligas/eventos.",
        "Turnover esperado": "Apresentado por cenario; capital medio teorico = turnover/3 dado giro mensal de 3x.",
        "ROI com haircut": "Usar ROI conservador/base/otimista controlado; conservador incorpora sem Top-3, OOS, WF e haircut de 50%.",
        "Capital teorico": "Capital medio = turnover/3; buffers de drawdown e p95 simulado reportados como risco, nao recomendacao operacional.",
        "Escalabilidade": "Limitada por concentracao e buckets de capacidade; precisa validar se ROI persiste em capacidade alta e fora de poucos eventos.",
        "Dependencia Top eventos": "Medida por ROI sem Top-k e share de P&L dos Top eventos; se deteriorar, marcar fragilidade.",
        "Dados faltantes": "P&L contrafactual de oportunidades bloqueadas, uptime real por minuto, oportunidades elegiveis nao executadas e exposicao aberta real por data.",
    }
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _render_md(out, Path(args.out_md))
    _render_pdf(Path(args.out_md), Path(args.out_pdf))
    print(f"[OK] all_backpre={len(all_rows)} h3bup={len(rows)} wc_start={wc_start}")
    print(f"[OK] JSON={args.out_json}")
    print(f"[OK] MD={args.out_md}")
    print(f"[OK] PDF={args.out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
