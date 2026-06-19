#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calcula M1..M5 por segmento:
- pre_com_world_cup
- pre_sem_world_cup
- pos_com_world_cup
- pos_sem_world_cup

Entrada: CSV por aposta (com stake + pnl), tipicamente:
- /tmp/projecao_por_aposta_enriquecido.csv
- /tmp/base_5ms_real_ate_*.csv

Saidas:
- JSON estruturado
- Markdown executivo
- PDF opcional (reportlab)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class Row:
    event_id: str
    league: str
    ts: Optional[datetime]
    week: str
    stake: float
    pnl: float


def _pf(x: Any) -> Optional[float]:
    s = str(x or "").strip().replace(" ", "")
    if not s:
        return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
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


def _norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _iso_week(dt: Optional[datetime]) -> str:
    if dt is None:
        return "week:unknown"
    y, w, _ = dt.isocalendar()
    return f"{y}-W{w:02d}"


def _pick_col(fields: Sequence[str], cands: Sequence[str]) -> Optional[str]:
    by_l = {f.lower(): f for f in fields}
    by_n = {re.sub(r"[^a-z0-9]+", "", f.lower()): f for f in fields}
    for c in cands:
        c_l = c.lower()
        if c_l in by_l:
            return by_l[c_l]
        c_n = re.sub(r"[^a-z0-9]+", "", c_l)
        if c_n in by_n:
            return by_n[c_n]
    return None


def _weighted_roi(rows: Sequence[Row]) -> Optional[float]:
    st = sum(r.stake for r in rows)
    if st <= 0:
        return None
    return 100.0 * (sum(r.pnl for r in rows) / st)


def _aggregate_events(rows: Sequence[Row]) -> List[Dict[str, Any]]:
    by_evt: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        rec = by_evt.get(r.event_id)
        if rec is None:
            rec = {
                "event_id": r.event_id,
                "league": r.league,
                "week": r.week,
                "ts": r.ts,
                "stake": 0.0,
                "pnl": 0.0,
            }
            by_evt[r.event_id] = rec
        rec["stake"] += r.stake
        rec["pnl"] += r.pnl
        if rec["ts"] is None and r.ts is not None:
            rec["ts"] = r.ts
    out = list(by_evt.values())
    for e in out:
        e["roi_pct"] = 100.0 * e["pnl"] / e["stake"] if e["stake"] > 0 else None
    return out


def _bootstrap_ci90(events: Sequence[Dict[str, Any]], iters: int, seed: int) -> Tuple[Optional[float], Optional[float]]:
    if not events:
        return None, None
    if len(events) == 1:
        roi = 100.0 * events[0]["pnl"] / events[0]["stake"] if events[0]["stake"] > 0 else None
        return roi, roi
    rng = random.Random(seed)
    n = len(events)
    vals: List[float] = []
    for _ in range(max(1, int(iters))):
        smp = [events[rng.randrange(n)] for _ in range(n)]
        st = sum(x["stake"] for x in smp)
        pnl = sum(x["pnl"] for x in smp)
        vals.append(100.0 * pnl / st if st > 0 else 0.0)
    vals.sort()
    lo = vals[int(round((len(vals) - 1) * 0.05))]
    hi = vals[int(round((len(vals) - 1) * 0.95))]
    return lo, hi


def _perm_p(events: Sequence[Dict[str, Any]], iters: int, seed: int) -> Optional[float]:
    if not events:
        return None
    st_obs = sum(e["stake"] for e in events)
    if st_obs <= 0:
        return None
    obs = 100.0 * sum(e["pnl"] for e in events) / st_obs
    strata: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for e in events:
        strata[(e["week"], e["league"])].append(e)
    rng = random.Random(seed + 17)
    ge = 0
    it = max(1, int(iters))
    for _ in range(it):
        st = 0.0
        pnl = 0.0
        for grp in strata.values():
            for e in grp:
                sign = -1.0 if rng.random() < 0.5 else 1.0
                st += e["stake"]
                pnl += sign * e["pnl"]
        stat = 100.0 * pnl / st if st > 0 else 0.0
        if stat >= obs:
            ge += 1
    return (ge + 1) / (it + 1)


def _top1_abs_pct(events: Sequence[Dict[str, Any]]) -> Optional[float]:
    if not events:
        return None
    total = sum(abs(e["pnl"]) for e in events)
    if total <= 0:
        return 0.0
    mx = max(abs(e["pnl"]) for e in events)
    return 100.0 * mx / total


def _roi_sem_topk(events: Sequence[Dict[str, Any]], k: int) -> Optional[float]:
    if not events:
        return None
    kept = sorted(events, key=lambda x: x["pnl"], reverse=True)[max(0, int(k)) :]
    if not kept:
        return None
    st = sum(e["stake"] for e in kept)
    if st <= 0:
        return None
    return 100.0 * sum(e["pnl"] for e in kept) / st


def _weekly_stats(events: Sequence[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    by_week: Dict[str, Dict[str, float]] = defaultdict(lambda: {"pnl": 0.0, "stake": 0.0})
    for e in events:
        by_week[e["week"]]["pnl"] += e["pnl"]
        by_week[e["week"]]["stake"] += e["stake"]
    rois: List[float] = []
    for v in by_week.values():
        if v["stake"] > 0:
            rois.append(100.0 * v["pnl"] / v["stake"])
    if not rois:
        return None, None
    pos = sum(1 for x in rois if x > 0)
    ratio = 100.0 * pos / len(rois)
    xs = sorted(rois)
    n = len(xs)
    med = xs[n // 2] if n % 2 == 1 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
    return ratio, med


def _m4_blocks(events: Sequence[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not events:
        return None, None, None
    ev = sorted(events, key=lambda e: (e["ts"] is None, e["ts"]))
    n = len(ev)
    a = n // 3
    b = (2 * n) // 3
    blks = [ev[:a], ev[a:b], ev[b:]]
    rois: List[Optional[float]] = []
    for blk in blks:
        if not blk:
            rois.append(None)
            continue
        st = sum(x["stake"] for x in blk)
        rois.append(100.0 * sum(x["pnl"] for x in blk) / st if st > 0 else None)
    return rois[0], rois[1], rois[2]


def _m5(rows: Sequence[Row]) -> Tuple[Optional[float], Optional[float]]:
    if not rows:
        return None, None
    ev = sum(r.pnl for r in rows) / len(rows)
    wins = [r.pnl for r in rows if r.pnl > 0]
    losses = [r.pnl for r in rows if r.pnl < 0]
    if not wins or not losses:
        return ev, None
    avg_win = sum(wins) / len(wins)
    avg_loss = abs(sum(losses) / len(losses))
    payoff = (avg_win / avg_loss) if avg_loss > 0 else None
    return ev, payoff


def _fmt(x: Optional[float], nd: int = 2, pct: bool = False) -> str:
    if x is None:
        return "NA"
    return f"{x:.{nd}f}" + ("%" if pct else "")


def _analyze_segment(name: str, rows: Sequence[Row], boot: int, perm: int, seed: int) -> Dict[str, Any]:
    if not rows:
        return {"name": name, "error": "segmento_vazio"}
    events = _aggregate_events(rows)
    roi = _weighted_roi(rows)
    ci90_lo, ci90_hi = _bootstrap_ci90(events, boot, seed)
    p_perm = _perm_p(events, perm, seed)
    roi_sem_top3 = _roi_sem_topk(events, 3)
    top1_abs = _top1_abs_pct(events)
    pos_ratio, med_week = _weekly_stats(events)
    r1, r2, r3 = _m4_blocks(events)
    ev_aposta, payoff_ratio = _m5(rows)

    m1 = (p_perm is not None and p_perm <= 0.10 and ci90_lo is not None and ci90_lo > 0)
    m2 = (roi_sem_top3 is not None and roi_sem_top3 > 0 and top1_abs is not None and top1_abs <= 35)
    m3 = (pos_ratio is not None and pos_ratio >= 55 and med_week is not None and med_week > 0)
    m4 = (((r1 is not None and r1 > 0) + (r2 is not None and r2 > 0) + (r3 is not None and r3 > 0)) >= 2 and (r3 is not None and r3 > 0))
    m5 = (ev_aposta is not None and ev_aposta > 0 and payoff_ratio is not None and payoff_ratio >= 1.8)

    score = sum([m1, m2, m3, m4, m5])
    label = "robusto" if score >= 4 else ("moderado" if score >= 2 else "fragil")

    return {
        "name": name,
        "core": {"n_bets": len(rows), "n_events": len(events), "roi_pct": roi},
        "M1": {"p_perm": p_perm, "ci90_lo": ci90_lo, "ci90_hi": ci90_hi, "status": "OK" if m1 else "FAIL"},
        "M2": {"roi_sem_top3": roi_sem_top3, "top1_abs_pct": top1_abs, "status": "OK" if m2 else "FAIL"},
        "M3": {"pos_ratio_pct": pos_ratio, "mediana_semanal_pct": med_week, "status": "OK" if m3 else "FAIL"},
        "M4": {"r1_pct": r1, "r2_pct": r2, "r3_pct": r3, "status": "OK" if m4 else "FAIL"},
        "M5": {"ev_por_aposta": ev_aposta, "payoff_ratio": payoff_ratio, "status": "OK" if m5 else "FAIL"},
        "score_5ms": score,
        "label": label,
    }


def _render_pdf(md_path: Path, pdf_path: Path, skip_if_missing: bool) -> bool:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except Exception:
        if skip_if_missing:
            print("[WARN] reportlab ausente; PDF nao gerado.")
            return False
        raise

    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    _, h = A4
    x, y = 36, h - 36
    for ln in md_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if y < 36:
            c.showPage()
            y = h - 36
        c.drawString(x, y, ln[:140])
        y -= 13
    c.save()
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Calcula M1..M5 por segmento com/sem World Cup e pre/pos.")
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--split-date", default="2026-05-25T00:00:00+00:00")
    ap.add_argument("--bootstrap-iters", type=int, default=10000)
    ap.add_argument("--perm-iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--world-cup-aliases",
        default="FIFA World Cup,World Cup,Copa do Mundo,FIFA Club World Cup,Club World Cup,Mundial de Clubes",
    )
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-pdf", default="")
    ap.add_argument("--skip-pdf-if-missing", type=int, default=1)
    args = ap.parse_args()

    src = Path(args.input_csv)
    if not src.exists():
        raise SystemExit(f"CSV nao encontrado: {src}")

    split = _pdt(args.split_date)
    if split is None:
        raise SystemExit(f"split-date invalido: {args.split_date}")

    aliases = {_norm_text(x) for x in str(args.world_cup_aliases).split(",") if str(x).strip()}

    with src.open(newline="", encoding="utf-8", errors="ignore") as f:
        rd = csv.DictReader(f)
        fields = list(rd.fieldnames or [])

        c_event = _pick_col(fields, ["event_id", "match_id", "fixture_id", "game_id", "id", "order_id", "audit_id"])
        c_league = _pick_col(fields, ["league", "league_name", "competition", "tournament"])
        c_ts = _pick_col(fields, ["audited_at", "timestamp", "created_at", "executed_at", "updated_at"])
        c_stake = _pick_col(fields, ["stake", "stake_real", "exposure", "exposure_real"])
        c_pnl = _pick_col(fields, ["pnl_real", "pnl", "profit", "pl", "result"])
        c_side = _pick_col(fields, ["side", "exec_side", "direction"])
        c_regime = _pick_col(fields, ["regime", "market_regime", "phase"])
        c_live = _pick_col(fields, ["is_live", "live"])
        c_slip = _pick_col(fields, ["slippage_pre_pct", "slippage", "slippage_raw_pct"])

        if not (c_event and c_ts and c_stake and c_pnl):
            raise SystemExit(
                f"Colunas minimas ausentes. event={c_event} ts={c_ts} stake={c_stake} pnl={c_pnl}"
            )

        rows: List[Row] = []
        idx = 0
        for r in rd:
            idx += 1
            st = _pf(r.get(c_stake))
            pnl = _pf(r.get(c_pnl))
            if st is None or st <= 0 or pnl is None:
                continue
            ts = _pdt(r.get(c_ts))
            if ts is None:
                continue

            if c_side:
                sv = str(r.get(c_side, "")).strip().lower()
                if sv and sv not in {"back", "b", "home", "away", "h", "a"}:
                    continue
            if c_regime:
                rv = str(r.get(c_regime, "")).strip().lower()
                if rv and ("pre" not in rv and rv not in {"prematch", "pre_match"}):
                    continue
            elif c_live:
                lv = str(r.get(c_live, "")).strip().lower()
                if lv in {"1", "true", "t", "yes", "y", "sim", "s"}:
                    continue

            if c_slip:
                sl = _pf(r.get(c_slip))
                if sl is None or not (sl < 0):
                    continue

            eid = str(r.get(c_event, "")).strip() or f"row_{idx}"
            lg = str(r.get(c_league, "")).strip() if c_league else ""
            rows.append(Row(eid, lg, ts, _iso_week(ts), st, pnl))

    pre_wc: List[Row] = []
    pre_no: List[Row] = []
    pos_wc: List[Row] = []
    pos_no: List[Row] = []
    for r in rows:
        is_wc = _norm_text(r.league) in aliases
        if r.ts is not None and r.ts < split:
            (pre_wc if is_wc else pre_no).append(r)
        else:
            (pos_wc if is_wc else pos_no).append(r)

    segs = [
        _analyze_segment("pre_com_world_cup", pre_wc, args.bootstrap_iters, args.perm_iters, args.seed + 101),
        _analyze_segment("pre_sem_world_cup", pre_no, args.bootstrap_iters, args.perm_iters, args.seed + 102),
        _analyze_segment("pos_com_world_cup", pos_wc, args.bootstrap_iters, args.perm_iters, args.seed + 103),
        _analyze_segment("pos_sem_world_cup", pos_no, args.bootstrap_iters, args.perm_iters, args.seed + 104),
    ]

    out = {
        "input_csv": str(src),
        "split_date": split.isoformat(),
        "aliases_world_cup": sorted(aliases),
        "segments": segs,
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("# Ms por segmento (pre/pos x com/sem World Cup)\n")
    lines.append(f"- input: `{src}`")
    lines.append(f"- split_date: `{split.isoformat()}`")
    lines.append("")
    for s in segs:
        lines.append(f"## {s['name']}")
        if s.get("error"):
            lines.append(f"- erro: {s['error']}\n")
            continue
        c = s["core"]
        lines.append(f"- n_bets: {c['n_bets']} | n_events: {c['n_events']} | ROI: {_fmt(c['roi_pct'], 2, True)}")
        lines.append(f"- score_5ms: {s['score_5ms']}/5 | label: **{str(s['label']).upper()}**\n")
        lines.append("| M | Metrica | Valor | Status |")
        lines.append("|---|---|---:|---|")
        lines.append(f"| M1 | p_perm / ci90_lo | {_fmt(s['M1']['p_perm'], 4)} / {_fmt(s['M1']['ci90_lo'], 2, True)} | {s['M1']['status']} |")
        lines.append(f"| M2 | ROI sem Top-3 / top1_abs | {_fmt(s['M2']['roi_sem_top3'], 2, True)} / {_fmt(s['M2']['top1_abs_pct'], 2, True)} | {s['M2']['status']} |")
        lines.append(f"| M3 | pos_ratio / mediana semanal | {_fmt(s['M3']['pos_ratio_pct'], 2, True)} / {_fmt(s['M3']['mediana_semanal_pct'], 2, True)} | {s['M3']['status']} |")
        lines.append(f"| M4 | r1 / r2 / r3 | {_fmt(s['M4']['r1_pct'], 2, True)} / {_fmt(s['M4']['r2_pct'], 2, True)} / {_fmt(s['M4']['r3_pct'], 2, True)} | {s['M4']['status']} |")
        lines.append(f"| M5 | EV/aposta / payoff_ratio | {_fmt(s['M5']['ev_por_aposta'], 4)} / {_fmt(s['M5']['payoff_ratio'], 4)} | {s['M5']['status']} |\n")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] JSON: {out_json}")
    print(f"[OK] MD  : {out_md}")

    if str(args.out_pdf or "").strip():
        out_pdf = Path(args.out_pdf)
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        generated = _render_pdf(out_md, out_pdf, bool(int(args.skip_pdf_if_missing)))
        if generated:
            print(f"[OK] PDF : {out_pdf}")

    for s in segs:
        if s.get("error"):
            print(f"{s['name']} -> {s['error']}")
        else:
            print(f"{s['name']} | n={s['core']['n_bets']} | score={s['score_5ms']} | label={s['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

