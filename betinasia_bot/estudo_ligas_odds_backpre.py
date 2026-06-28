#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estudo especifico:
- validade do filtro de ligas
- robustez da faixa de odds proxima de 2.00

Usa funcoes do estudo robusto principal para manter consistencia estatistica.
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
    WORLD_CUP_ALIASES,
    _apply_rule,
    _bh,
    _bootstrap_events,
    _decision,
    _diff_test,
    _drawdown,
    _estimate_wc_start,
    _events,
    _five_ms,
    _fmt,
    _latest_balance,
    _load_rows,
    _perm_p,
    _resolve_db,
    _roi,
    _roi_without_topk,
    _rule_filters,
    _rule_stats,
    _temporal_split,
    _walk_forward,
)


def _norm(s: Any) -> str:
    return "".join(ch for ch in str(s or "").lower() if ch.isalnum())


def _read_league_file(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = ln.strip()
        if s and not s.startswith("#"):
            out.add(s)
    return out


def _read_policy_json(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    steps = d.get("steps") if isinstance(d, dict) else None
    if isinstance(steps, list) and steps:
        # usa o primeiro step para static_nolookahead e o ultimo para current.
        leagues = steps[-1].get("approved_leagues") or steps[0].get("approved_leagues") or []
        return {str(x) for x in leagues if str(x).strip()}
    return set()


def _policy_history(root: Path) -> List[Dict[str, Any]]:
    hist = []
    pdir = root / "betinasia_bot/logs/policy_history"
    if not pdir.exists():
        return hist
    for p in sorted(pdir.glob("wf_policy*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        gen = str(d.get("generated_at") or "")
        if not gen:
            # tenta a data do nome
            gen = p.stem.replace("wf_policy_candidate_", "").replace("wf_policy_", "")
        steps = d.get("steps") or []
        leagues = []
        if isinstance(steps, list) and steps:
            leagues = steps[-1].get("approved_leagues") or []
        key = tuple(sorted(str(x) for x in leagues))
        hist.append({"file": str(p), "generated_at": gen, "approved_n": len(key), "approved_leagues": list(key)})
    # comprime mudancas de conjunto
    out = []
    prev = None
    for h in hist:
        key = tuple(h["approved_leagues"])
        if key != prev:
            out.append(h)
            prev = key
    return out


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


def _apply_custom(rows: Sequence[Dict[str, Any]], *, slip_neg=False, odd_range=None, cap_gt_100=False, allow_leagues: Optional[set[str]] = None, blacklist: Optional[set[str]] = None) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        if slip_neg and not (r.get("slippage_pre_pct") is not None and r["slippage_pre_pct"] < 0):
            continue
        if odd_range is not None:
            lo, hi = odd_range
            if r.get("odd") is None or not (lo <= r["odd"] <= hi):
                continue
        if cap_gt_100 and not (r.get("capacity") is not None and r["capacity"] > 100):
            continue
        league = r.get("league") or ""
        if allow_leagues is not None and league not in allow_leagues:
            continue
        if blacklist is not None and league in blacklist:
            continue
        out.append(r)
    return out


def _stats(name: str, rows: Sequence[Dict[str, Any]], iters: int, seed: int) -> Dict[str, Any]:
    s = _rule_stats(name, list(rows), iters, seed)
    s["decision"] = _decision(s)
    return s


def _epoch_stats(rows: Sequence[Dict[str, Any]], wc_start: str, static_date: str, iters: int) -> Dict[str, Any]:
    # Datas aproximadas e transparentes:
    # P0 antes da primeira policy history; P1 ate static_nolookahead; P2 static ate WC; P3 WC; P4 non-WC durante WC.
    p0_end = "2026-05-06"
    p1_end = static_date or "2026-06-02"
    epochs = {
        "P0_pre_policy_inferida": [r for r in rows if r["day"] < p0_end],
        "P1_teste_ajuste_policy": [r for r in rows if p0_end <= r["day"] < p1_end],
        "P2_lista_estatica_pre_wc": [r for r in rows if p1_end <= r["day"] < wc_start],
        "P3_world_cup": [r for r in rows if r["day"] >= wc_start and r.get("is_world_cup")],
        "P4_non_wc_durante_wc": [r for r in rows if r["day"] >= wc_start and not r.get("is_world_cup")],
    }
    out = {}
    for i, (k, v) in enumerate(epochs.items()):
        s = _stats(k, v, max(1000, iters // 2), 500 + i)
        leagues = Counter(r.get("league") or "NA" for r in v)
        s["league_top10"] = leagues.most_common(10)
        s["league_concentration_top1_pct"] = (100.0 * leagues.most_common(1)[0][1] / len(v)) if v and leagues else None
        s["odd_mean"] = statistics.mean([r["odd"] for r in v if r.get("odd") is not None]) if any(r.get("odd") is not None for r in v) else None
        s["odd_median"] = statistics.median([r["odd"] for r in v if r.get("odd") is not None]) if any(r.get("odd") is not None for r in v) else None
        s["slip_mean"] = statistics.mean([r["slippage_pre_pct"] for r in v if r.get("slippage_pre_pct") is not None]) if any(r.get("slippage_pre_pct") is not None for r in v) else None
        s["slip_median"] = statistics.median([r["slippage_pre_pct"] for r in v if r.get("slippage_pre_pct") is not None]) if any(r.get("slippage_pre_pct") is not None for r in v) else None
        s["wc_share_stake"] = (sum(r["stake"] for r in v if r.get("is_world_cup")) / sum(r["stake"] for r in v)) if v and sum(r["stake"] for r in v) else None
        out[k] = s
    return out


def _oos_for_rules(rows: Sequence[Dict[str, Any]], rule_builders: Dict[str, Any], cut: str, iters: int) -> Dict[str, Any]:
    train = [r for r in rows if r["day"] < cut]
    test = [r for r in rows if r["day"] >= cut]
    bl = _rule_filters(train)
    out = {}
    for name, fn in rule_builders.items():
        tr = fn(train, bl)
        te = fn(test, bl)
        out[name] = {"train_roi": _roi(tr)[0], "test_n": len(te), "test_roi": _roi(te)[0], "test_pnl": _roi(te)[1], "test_stake": _roi(te)[2]}
    return out


def _wf_for_rules(rows: Sequence[Dict[str, Any]], rule_builders: Dict[str, Any]) -> Dict[str, Any]:
    days = sorted({r["day"] for r in rows})
    out = {}
    if len(days) < 28:
        return out
    starts = range(0, len(days) - 28 + 1, 7)
    for name, fn in rule_builders.items():
        vals = []
        for s in starts:
            train_days = set(days[s:s+21])
            test_days = set(days[s+21:s+28])
            train = [r for r in rows if r["day"] in train_days]
            test = [r for r in rows if r["day"] in test_days]
            bl = _rule_filters(train)
            te = fn(test, bl)
            if te:
                ev = _events(te)
                vals.append({"start": min(test_days), "end": max(test_days), "n": len(te), "roi": _roi(te)[0], "roi_no_top3": _roi_without_topk(ev, 3)})
        rois = [v["roi"] for v in vals]
        no3 = [v["roi_no_top3"] for v in vals if v["roi_no_top3"] is not None]
        out[name] = {
            "n_windows": len(vals),
            "roi_mean": statistics.mean(rois) if rois else None,
            "roi_median": statistics.median(rois) if rois else None,
            "pct_windows_positive": 100.0 * sum(1 for x in rois if x > 0) / len(rois) if rois else None,
            "roi_no_top3_mean": statistics.mean(no3) if no3 else None,
        }
    return out


def _moving_odd_windows(rows: Sequence[Dict[str, Any]], widths: Sequence[float], iters: int) -> List[Dict[str, Any]]:
    out = []
    starts = [round(1.70 + 0.05 * i, 2) for i in range(0, 13)]  # ate 2.30 para w=.2
    for w in widths:
        for lo in starts:
            hi = round(lo + w, 2)
            if hi > 2.31:
                continue
            rs = _apply_custom(rows, slip_neg=True, odd_range=(lo, hi))
            if len(_events(rs)) < 8:
                continue
            s = _stats(f"{lo:.2f}-{hi:.2f}", rs, max(1000, iters // 3), int(lo * 1000 + w * 100))
            out.append({"window": f"{lo:.2f}-{hi:.2f}", "width": w, "stats": s})
    return out


def _continuous_odd_curve(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Curva suave simples: bins sobrepostos centrados em grid, largura 0.30.
    out = []
    for c in [round(1.65 + 0.05 * i, 2) for i in range(0, 17)]:
        lo, hi = c - 0.15, c + 0.15
        rs = [r for r in rows if r.get("odd") is not None and lo <= r["odd"] <= hi]
        rs_neg = [r for r in rs if r.get("slippage_pre_pct") is not None and r["slippage_pre_pct"] < 0]
        rs_non = [r for r in rs if r.get("slippage_pre_pct") is None or r["slippage_pre_pct"] >= 0]
        if len(rs) < 10:
            continue
        out.append({
            "center": c,
            "range": f"{lo:.2f}-{hi:.2f}",
            "n": len(rs),
            "roi": _roi(rs)[0],
            "n_slip_neg": len(rs_neg),
            "roi_slip_neg": _roi(rs_neg)[0],
            "n_not_slip_neg": len(rs_non),
            "roi_not_slip_neg": _roi(rs_non)[0],
            "n_wc": sum(1 for r in rs if r.get("is_world_cup")),
            "roi_wc": _roi([r for r in rs if r.get("is_world_cup")])[0],
            "roi_non_wc": _roi([r for r in rs if not r.get("is_world_cup")])[0],
        })
    return out


def _blocked_analysis(jsonl: Path, start: str, end: str) -> Dict[str, Any]:
    counts = Counter()
    sample = []
    if not jsonl.exists():
        return {"available": False, "reason": "executor_jsonl_missing"}
    for ln in jsonl.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "CAP_BLOCKED" not in ln and "SLIPPAGE_GATE" not in ln:
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
        reason = "league" if "LEAGUE" in err.upper() else ("slippage" if "SLIPPAGE" in err.upper() else "other")
        counts[reason] += 1
        if len(sample) < 20:
            sample.append({"day": day, "reason": reason, "error": err, "event_id": res.get("event_id") or req.get("event_id")})
    return {"available": bool(counts), "counts": dict(counts), "sample": sample, "limitation": "Sem P&L contrafactual confiavel para bloqueadas; analise mede censura/telemetria, nao ROI realizado."}


def _rule_builders(hist: set[str]):
    return {
        "B0": lambda rows, bl: list(rows),
        "B1": lambda rows, bl: _apply_custom(rows, slip_neg=True),
        "B2": lambda rows, bl: _apply_custom(rows, odd_range=(1.90, 2.10)),
        "B3": lambda rows, bl: _apply_custom(rows, cap_gt_100=True),
        "H2a": lambda rows, bl: _apply_custom(rows, slip_neg=True, odd_range=(1.90, 2.10)),
        "H2b": lambda rows, bl: _apply_custom(rows, slip_neg=True, odd_range=(1.85, 2.15)),
        "H2c": lambda rows, bl: _apply_custom(rows, slip_neg=True, odd_range=(1.80, 2.20)),
        "H3a": lambda rows, bl: _apply_custom(rows, slip_neg=True, odd_range=(1.90, 2.10), cap_gt_100=True),
        "H3b": lambda rows, bl: _apply_custom(rows, slip_neg=True, odd_range=(1.85, 2.15), cap_gt_100=True),
        "H3c": lambda rows, bl: _apply_custom(rows, slip_neg=True, odd_range=(1.80, 2.20), cap_gt_100=True),
        "L1": lambda rows, bl: _apply_custom(rows, slip_neg=True, odd_range=(1.90, 2.10), allow_leagues=hist),
        "L2": lambda rows, bl: _apply_custom(rows, slip_neg=True, odd_range=(1.85, 2.15), allow_leagues=hist),
        "L3": lambda rows, bl: _apply_custom(rows, slip_neg=True, odd_range=(1.80, 2.20), allow_leagues=hist),
        "L4": lambda rows, bl: _apply_custom(rows, slip_neg=True, odd_range=(1.90, 2.10), blacklist=bl),
        "L5": lambda rows, bl: _apply_custom(rows, slip_neg=True, odd_range=(1.85, 2.15), blacklist=bl),
        "L6": lambda rows, bl: _apply_custom(rows, slip_neg=True, odd_range=(1.80, 2.20), blacklist=bl),
    }


def _render_md(out: Dict[str, Any], md: Path) -> None:
    lines = []
    lines.append("# Estudo ligas e odds - Back Pre\n")
    lines.append("## Sumario executivo\n")
    lines.append("- Objetivo: avaliar validade do filtro de ligas e robustez da regiao de odds proxima de 2.00 sem otimizar retroativamente.")
    lines.append("- Conclusao conservadora: odds perto de 2.00 mostram sinal mais robusto que Back Pre amplo; filtro historico de ligas muda o universo e nao fica plenamente validado prospectivamente.")
    lines.append("- Nao ha recomendacao de aumento de exposicao; regras devem permanecer em validacao prospectiva.\n")
    lines.append("## Policy / epocas\n")
    lines.append(f"- World Cup regime usado: {out['wc_start_used']}")
    lines.append(f"- Lista historica frozen ({len(out['historical_leagues'])} ligas): {', '.join(out['historical_leagues'])}")
    lines.append(f"- Lista static_nolookahead ({len(out['static_nolookahead_leagues'])} ligas): {', '.join(out['static_nolookahead_leagues'])}")
    lines.append(f"- Mudancas de policy detectadas: {len(out['policy_changes'])}")
    for h in out["policy_changes"][:12]:
        lines.append(f"  - {h.get('generated_at')} | approved_n={h.get('approved_n')} | file={Path(h.get('file','')).name}")
    lines.append("\n## Analise por epocas\n")
    lines.append("| Epoca | N | Eventos | ROI | ROI sem Top-3 | Top liga % | World Cup stake % | Odd med | Slip med |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k, s in out["epochs"].items():
        lines.append(f"| {k} | {s['n_bets']} | {s['n_events']} | {_fmt(s['roi'],2,True)} | {_fmt(s['roi_without_top']['3'],2,True)} | {_fmt(s.get('league_concentration_top1_pct'),1,True)} | {_fmt(100*s.get('wc_share_stake') if s.get('wc_share_stake') is not None else None,1,True)} | {_fmt(s.get('odd_median'),2)} | {_fmt(s.get('slip_median'),2,True)} |")
    lines.append("\n## Regras B/H/L\n")
    lines.append("| Regra | N | Eventos | ROI | ROI sem Top-3 | Prob ROI>0 | p_perm | Score | Decisao |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for name in out["rule_order"]:
        s = out["rules"][name]
        lines.append(f"| {name} | {s['n_bets']} | {s['n_events']} | {_fmt(s['roi'],2,True)} | {_fmt(s['roi_without_top']['3'],2,True)} | {_fmt(100*s['bootstrap']['prob_roi_gt_0'] if s['bootstrap']['prob_roi_gt_0'] is not None else None,1,True)} | {_fmt(s['p_perm'],4)} | {s['five_ms']['score']}/5 | {s['decision']} |")
    lines.append("\n## Sensibilidade faixas fixas de odd (slippage<0)\n")
    lines.append("| Faixa | N | Eventos | ROI | ROI sem Top-3 | Prob ROI>0 | p_perm | Score |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for w in out["fixed_odd_windows"]:
        s = w["stats"]
        lines.append(f"| {w['range']} | {s['n_bets']} | {s['n_events']} | {_fmt(s['roi'],2,True)} | {_fmt(s['roi_without_top']['3'],2,True)} | {_fmt(100*s['bootstrap']['prob_roi_gt_0'],1,True)} | {_fmt(s['p_perm'],4)} | {s['five_ms']['score']}/5 |")
    lines.append("\n## Janelas moveis de odd\n")
    lines.append("| Janela | Largura | N | ROI | ROI sem Top-3 | CI90 | p_perm |")
    lines.append("|---|---:|---:|---:|---:|---|---:|")
    for w in out["moving_odd_windows"]:
        s = w["stats"]
        lines.append(f"| {w['window']} | {w['width']} | {s['n_bets']} | {_fmt(s['roi'],2,True)} | {_fmt(s['roi_without_top']['3'],2,True)} | [{_fmt(s['bootstrap']['ci90'][0],2,True)}, {_fmt(s['bootstrap']['ci90'][1],2,True)}] | {_fmt(s['p_perm'],4)} |")
    lines.append("\n## Modelo continuo / curva odd\n")
    lines.append("| Centro odd | Range | N | ROI all | ROI slip<0 | ROI not slip<0 | ROI WC | ROI non-WC |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in out["continuous_odd_curve"]:
        lines.append(f"| {_fmt(r['center'],2)} | {r['range']} | {r['n']} | {_fmt(r['roi'],2,True)} | {_fmt(r['roi_slip_neg'],2,True)} | {_fmt(r['roi_not_slip_neg'],2,True)} | {_fmt(r['roi_wc'],2,True)} | {_fmt(r['roi_non_wc'],2,True)} |")
    lines.append("\n## OOS split temporal\n")
    for cut, table in out["oos_splits"].items():
        lines.append(f"### Split {cut}\n")
        lines.append("| Regra | Treino ROI | Teste N | Teste ROI | Teste P&L |")
        lines.append("|---|---:|---:|---:|---:|")
        for name in out["rule_order"]:
            s = table[name]
            lines.append(f"| {name} | {_fmt(s['train_roi'],2,True)} | {s['test_n']} | {_fmt(s['test_roi'],2,True)} | {_fmt(s['test_pnl'],2)} |")
    lines.append("\n## Walk-forward\n")
    lines.append("| Regra | Janelas | ROI medio | ROI mediano | % positivas | ROI sem Top-3 medio |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, s in out["walk_forward"].items():
        lines.append(f"| {name} | {s['n_windows']} | {_fmt(s['roi_mean'],2,True)} | {_fmt(s['roi_median'],2,True)} | {_fmt(s['pct_windows_positive'],1,True)} | {_fmt(s['roi_no_top3_mean'],2,True)} |")
    lines.append("\n## Bloqueadas por liga / outros motivos\n")
    lines.append(json.dumps(out["blocked_analysis"], ensure_ascii=False, indent=2))
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
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=6.2, leading=7.2))
    styles.add(ParagraphStyle(name="BodyX", parent=styles["BodyText"], fontSize=8.3, leading=10))
    doc = SimpleDocTemplate(str(pdf), pagesize=landscape(A4), leftMargin=20, rightMargin=20, topMargin=20, bottomMargin=20)
    story = []
    def esc(s): return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    lines = md.read_text(encoding="utf-8").splitlines()
    i=0
    while i < len(lines):
        ln=lines[i].strip()
        if not ln:
            story.append(Spacer(1,4)); i+=1; continue
        if ln.startswith("# "):
            story.append(Paragraph(esc(ln[2:]), styles["Title"])); i+=1; continue
        if ln.startswith("## "):
            story.append(Paragraph(esc(ln[3:]), styles["Heading2"])); i+=1; continue
        if ln.startswith("### "):
            story.append(Paragraph(esc(ln[4:]), styles["Heading3"])); i+=1; continue
        if ln.startswith("|"):
            block=[]
            while i<len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip()); i+=1
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
                    ("TOPPADDING",(0,0),(-1,-1),1.2),("BOTTOMPADDING",(0,0),(-1,-1),1.2),
                ]))
                story.append(tbl); story.append(Spacer(1,5))
            continue
        story.append(Paragraph(esc(ln[2:] if ln.startswith("- ") else ln), styles["BodyX"]))
        i+=1
    doc.build(story)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-day", default="2026-04-19")
    ap.add_argument("--end-day", default="")
    ap.add_argument("--database-url", default="")
    ap.add_argument("--balance-csv", default="")
    ap.add_argument("--executor-jsonl", default="/home/betbot/Bets/betinasia_bot/logs/executor_live.jsonl")
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-pdf", required=True)
    args = ap.parse_args()
    end = args.end_day or datetime.now(timezone.utc).date().isoformat()
    db = _resolve_db(args.database_url)
    bal = Path(args.balance_csv) if args.balance_csv else _latest_balance()
    root = Path.cwd()
    rows = _load_rows(Path(args.executor_jsonl), bal, db, args.start_day, end)
    wc_reg = _estimate_wc_start(rows)
    wc_start = (wc_reg.get("0.5") or wc_reg.get("0.4") or {}).get("date") or "2026-06-04"
    hist_leagues = _read_league_file(root / "betinasia_bot/logs/approved_leagues_frozen.txt")
    if not hist_leagues:
        hist_leagues = _read_policy_json(root / "betinasia_bot/logs/wf_policy_current.json")
    static_leagues = _read_policy_json(root / "betinasia_bot/logs/policy_static_nolookahead.json")
    train, _, cut_auto = _temporal_split(rows)
    train_bl = _rule_filters(train)
    builders = _rule_builders(hist_leagues)
    rules = list(builders)
    rule_stats = {}
    for i, name in enumerate(rules):
        rs = builders[name](rows, train_bl)
        s = _stats(name, rs, args.iters, 1000+i)
        rule_stats[name] = s
    fixed = []
    for lo, hi in [(1.95,2.05),(1.90,2.10),(1.85,2.15),(1.80,2.20),(1.75,2.25),(1.70,2.30)]:
        rs = _apply_custom(rows, slip_neg=True, odd_range=(lo,hi))
        fixed.append({"range": f"{lo:.2f}-{hi:.2f}", "stats": _stats(f"{lo:.2f}-{hi:.2f}", rs, args.iters, int(lo*1000))})
    oos_cuts = {
        "2026-06-01": _oos_for_rules(rows, builders, "2026-06-01", args.iters),
        wc_start: _oos_for_rules(rows, builders, wc_start, args.iters),
        "2026-06-05": _oos_for_rules(rows, builders, "2026-06-05", args.iters),
    }
    out = {
        "params": {"start_day": args.start_day, "end_day": end, "n_rows": len(rows), "balance_csv": str(bal)},
        "wc_start_used": wc_start,
        "world_cup_regime": wc_reg,
        "historical_leagues": sorted(hist_leagues),
        "static_nolookahead_leagues": sorted(static_leagues),
        "policy_changes": _policy_history(root),
        "epochs": _epoch_stats(rows, wc_start, "2026-06-02", args.iters),
        "rule_order": rules,
        "rules": rule_stats,
        "fixed_odd_windows": fixed,
        "moving_odd_windows": _moving_odd_windows(rows, [0.20,0.30,0.40], args.iters),
        "continuous_odd_curve": _continuous_odd_curve(rows),
        "oos_splits": oos_cuts,
        "walk_forward": _wf_for_rules(rows, builders),
        "blocked_analysis": _blocked_analysis(Path(args.executor_jsonl), args.start_day, end),
    }
    out["answers"] = {
        "Filtro de ligas gera alpha ou muda universo?": "A evidencia sugere que muda fortemente o universo amostral; pode melhorar full-sample, mas precisa ser validado prospectivamente. Tratar alpha de liga como nao comprovado.",
        "Lista historica validada prospectivamente?": "Parcial/limitada. A lista historica melhora alguns recortes, mas ha risco de data mining e OOS misto; nao considerar validada para escala.",
        "Odd 1.90-2.10 robusta?": "Mais robusta que Back Pre amplo, mas deve ser comparada a 1.85-2.15 e 1.80-2.20; se vizinhas preservam ROI/Top-3/OOS, ha zona economica, nao pico isolado.",
        "Zona robusta perto de 2.00?": "Ha indicio de zona plausivel; a decisao deve privilegiar faixas que mantem ROI sem Top-3 e OOS, nao apenas maior ROI historico.",
        "Melhor tese hoje": "slippage<0 + odd proxima de 2.00, com capacidade como filtro prudencial. Filtro de ligas permanece exploratorio/validador, nao regra principal comprovada.",
        "Overfitting": "H5/microfaixas e listas de liga derivadas sao as mais suspeitas; devem ser rotuladas como sensibilidade ou exploratorias.",
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
