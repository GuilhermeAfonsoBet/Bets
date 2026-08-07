#!/usr/bin/env python3
"""Temporal slice analysis for H3BUP CLV + ROI (read-only)."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def fnum(x: Any) -> Optional[float]:
    try:
        if x in (None, "", "None"):
            return None
        return float(x)
    except Exception:
        return None


def fbool(x: Any) -> bool:
    return str(x).strip().lower() in ("1", "true", "yes")


def pct(x: Optional[float], d: int = 1) -> str:
    return "—" if x is None else f"{100 * x:.{d}f}%"


def money(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{x:+.2f}" if abs(x) > 1e-12 else "0.00"


def fmt_clv(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.2f}%"


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


WINDOWS = [
    ("POST_5M", "clv_post_5m", "clv_post_5m_valid_strict"),
    ("POST_15M", "clv_post_15m", "clv_post_15m_valid_strict"),
    ("CLOSING", "clv_closing", "clv_closing_valid_strict"),
]

STAKE2_START = datetime(2026, 8, 1, 12, 7, 0, tzinfo=timezone.utc)


def metrics(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
    settled = [r for r in rs if r.get("settlement_status") == "SETTLED_DECIDED"]
    voids = [r for r in rs if r.get("settlement_status") == "VOID_PUSH"]
    opens = [r for r in rs if r.get("settlement_status") == "OPEN"]
    missing = [r for r in rs if r.get("settlement_status") == "MISSING"]
    resolved = settled + voids
    pnl = sum(r["_pnl"] for r in settled)
    stake_res = sum(r["_stake"] for r in resolved)
    stake_pl = sum(r["_stake"] for r in rs)
    wr = (sum(1 for r in settled if r["_pnl"] > 0) / len(settled)) if settled else None
    clv: Dict[str, Any] = {}
    for w, field, validf in WINDOWS:
        vals = [
            fnum(r.get(field))
            for r in rs
            if fbool(r.get(validf)) and fnum(r.get(field)) is not None
        ]
        vals = [v for v in vals if v is not None]
        clv[w] = {
            "n": len(vals),
            "coverage": (len(vals) / len(rs) if rs else None),
            "mean": (statistics.mean(vals) if vals else None),
            "median": (statistics.median(vals) if vals else None),
            "p_positive": (sum(1 for v in vals if v > 0) / len(vals) if vals else None),
        }
    # bootstrap ROI event-cluster (light)
    roi_boot = None
    if settled:
        rng = random.Random(20260807 + len(settled))
        by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in settled:
            by[str(r.get("event_id"))].append(r)
        keys = list(by)
        vals_b = []
        for _ in range(3000):
            sp = ss = 0.0
            for k in rng.choices(keys, k=len(keys)):
                for r in by[k]:
                    sp += r["_pnl"]
                    ss += r["_stake"]
            if ss:
                vals_b.append(sp / ss)
        vals_b.sort()
        roi_boot = {
            "mean": statistics.mean(vals_b),
            "ci95": [vals_b[int(0.025 * len(vals_b))], vals_b[int(0.975 * len(vals_b)) - 1]],
            "p_pos": sum(1 for v in vals_b if v > 0) / len(vals_b),
        }
    def _boot_clv(field: str, validf: str, seed: int) -> Optional[Dict[str, Any]]:
        vals = [
            fnum(r.get(field))
            for r in rs
            if fbool(r.get(validf)) and fnum(r.get(field)) is not None
        ]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        rng = random.Random(seed + len(vals))
        out = [statistics.mean(rng.choices(vals, k=len(vals))) for _ in range(3000)]
        out.sort()
        return {
            "mean": statistics.mean(out),
            "ci95": [out[int(0.025 * len(out))], out[int(0.975 * len(out)) - 1]],
            "p_pos": sum(1 for v in out if v > 0) / len(out),
        }

    clv5_boot = _boot_clv("clv_post_5m", "clv_post_5m_valid_strict", 77)
    clvC_boot = _boot_clv("clv_closing", "clv_closing_valid_strict", 91)
    return {
        "n": len(rs),
        "events": len({r.get("event_id") for r in rs}),
        "settled": len(settled),
        "void": len(voids),
        "open": len(opens),
        "missing": len(missing),
        "friendly_n": sum(1 for r in rs if r.get("friendly_class") == "FRIENDLY"),
        "nf_n": sum(1 for r in rs if r.get("friendly_class") == "NON_FRIENDLY"),
        "stake_placed": stake_pl,
        "stake_resolved": stake_res,
        "pnl": pnl,
        "roi": (pnl / stake_res if stake_res else None),
        "wr": wr,
        "clv": clv,
        "roi_boot": roi_boot,
        "clv5_boot": clv5_boot,
        "clvC_boot": clvC_boot,
    }


def main() -> int:
    random.seed(20260807)
    run = "a27c1dc4ab52"
    freeze = Path(f"/workspace/betinasia_bot/logs/h3bup_friendly_analysis/20260807/{run}")
    out = Path(f"/workspace/betinasia_bot/logs/h3bup_strategy_results/20260807/temporal_{run}")
    docs = Path("/workspace/betinasia_bot/docs")
    out.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader((freeze / f"h3bup_friendly_order_level_{run}.csv").open()))
    for r in rows:
        r["_ts"] = parse_ts(r["created_at_utc"])
        r["_pnl"] = fnum(r.get("pnl")) or 0.0
        r["_stake"] = fnum(r.get("stake")) or 0.0
        s = fnum(r.get("stake"))
        r["_era"] = "stake10" if s and abs(s - 10) < 1e-9 else ("stake2" if s and abs(s - 2) < 1e-9 else "other")

    cutoff = max(r["_ts"] for r in rows)
    # use freeze cutoff as reference end-of-day for rolling windows
    asof = datetime(2026, 8, 7, 14, 25, 6, tzinfo=timezone.utc)

    slices: List[Tuple[str, str, List[Dict[str, Any]]]] = []

    # 1) Full
    slices.append(("full", "2026-07-28 → 2026-08-07 (tudo)", rows))

    # 2) Stake eras
    pre = [r for r in rows if r["_ts"] < STAKE2_START]
    post = [r for r in rows if r["_ts"] >= STAKE2_START]
    slices.append(("era_stake10", f"Antes stake2 (< {STAKE2_START.isoformat()})", pre))
    slices.append(("era_stake2", f"Desde stake2 (≥ {STAKE2_START.isoformat()})", post))

    # 3) Rolling windows ending at asof
    for days in (1, 2, 3, 5, 7):
        start = asof - timedelta(days=days)
        rs = [r for r in rows if r["_ts"] > start]
        slices.append((f"last_{days}d", f"Últimos {days}d (após {start.date().isoformat()})", rs))

    # 4) Calendar weeks (Mon-Sun UTC)
    by_week: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        iso = r["_ts"].isocalendar()
        key = f"{iso.year}-W{iso.week:02d}"
        by_week[key].append(r)
    for wk in sorted(by_week):
        rs = by_week[wk]
        lo = min(x["_ts"] for x in rs).date().isoformat()
        hi = max(x["_ts"] for x in rs).date().isoformat()
        slices.append((f"week_{wk}", f"Semana {wk} ({lo}→{hi})", rs))

    # 5) Early vs late within stake2
    if post:
        mid = STAKE2_START + (asof - STAKE2_START) / 2
        s2_early = [r for r in post if r["_ts"] < mid]
        s2_late = [r for r in post if r["_ts"] >= mid]
        slices.append(("stake2_early", f"stake2 1ª metade (< {mid.date().isoformat()})", s2_early))
        slices.append(("stake2_late", f"stake2 2ª metade (≥ {mid.date().isoformat()})", s2_late))

    # 6) Exclude capacity-fix first 3 days
    after_warm = [r for r in rows if r["_ts"] >= datetime(2026, 7, 31, tzinfo=timezone.utc)]
    slices.append(("from_0731", "Desde 2026-07-31 (excl. warm-up 28–30)", after_warm))

    # 7) Day-by-day
    by_day: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_day[r["_ts"].date().isoformat()].append(r)
    for d in sorted(by_day):
        slices.append((f"day_{d}", f"Dia {d}", by_day[d]))

    # compute
    results = []
    for sid, label, rs in slices:
        m = metrics(rs)
        results.append({"id": sid, "label": label, **m})

    # cumulative by day (equity of settled only, ordered)
    settled_all = sorted(
        [r for r in rows if r.get("settlement_status") == "SETTLED_DECIDED"],
        key=lambda r: r["_ts"],
    )
    cum = 0.0
    stake_cum = 0.0
    equity = []
    for r in settled_all:
        cum += r["_pnl"]
        stake_cum += r["_stake"]
        equity.append(
            {
                "ts": r["created_at_utc"],
                "day": r["_ts"].date().isoformat(),
                "era": r["_era"],
                "cum_pnl": cum,
                "cum_stake": stake_cum,
                "cum_roi": (cum / stake_cum if stake_cum else None),
                "order_id": r.get("order_id"),
            }
        )

    # rolling 7d ROI/CLV by end-day
    rolling = []
    days_sorted = sorted(by_day)
    for i, d in enumerate(days_sorted):
        end = datetime.fromisoformat(d).replace(tzinfo=timezone.utc) + timedelta(days=1)
        start = end - timedelta(days=7)
        rs = [r for r in rows if start < r["_ts"] <= end]
        m = metrics(rs)
        rolling.append(
            {
                "end_day": d,
                "n": m["n"],
                "roi": m["roi"],
                "pnl": m["pnl"],
                "clv5_mean": m["clv"]["POST_5M"]["mean"],
                "clv5_n": m["clv"]["POST_5M"]["n"],
                "clv5_pos": m["clv"]["POST_5M"]["p_positive"],
                "clvC_mean": m["clv"]["CLOSING"]["mean"],
                "clvC_n": m["clv"]["CLOSING"]["n"],
                "clvC_pos": m["clv"]["CLOSING"]["p_positive"],
                "wr": m["wr"],
            }
        )

    bundle = {
        "meta": {
            "freeze_run": run,
            "asof_utc": asof.isoformat(),
            "stake2_start_utc": STAKE2_START.isoformat(),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "n_total": len(rows),
            "clv_windows": ["POST_5M", "POST_15M", "CLOSING"],
        },
        "slices": [
            {
                "id": r["id"],
                "label": r["label"],
                "n": r["n"],
                "settled": r["settled"],
                "open": r["open"],
                "pnl": r["pnl"],
                "roi": r["roi"],
                "wr": r["wr"],
                "clv5": r["clv"]["POST_5M"],
                "clv15": r["clv"]["POST_15M"],
                "clvC": r["clv"]["CLOSING"],
                "roi_boot": r["roi_boot"],
                "clv5_boot": r["clv5_boot"],
                "clvC_boot": r["clvC_boot"],
                "friendly_n": r["friendly_n"],
                "nf_n": r["nf_n"],
                "stake_placed": r["stake_placed"],
                "stake_resolved": r["stake_resolved"],
            }
            for r in results
        ],
        "rolling_7d": rolling,
        "equity_tail": equity[-20:],
    }
    (out / f"h3bup_temporal_bundle_{run}.json").write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")

    flat = []
    for r in results:
        flat.append(
            {
                "id": r["id"],
                "label": r["label"],
                "n": r["n"],
                "settled": r["settled"],
                "open": r["open"],
                "void": r["void"],
                "friendly_n": r["friendly_n"],
                "nf_n": r["nf_n"],
                "stake_placed": r["stake_placed"],
                "stake_resolved": r["stake_resolved"],
                "pnl": r["pnl"],
                "roi": r["roi"],
                "wr": r["wr"],
                "clv5_n": r["clv"]["POST_5M"]["n"],
                "clv5_mean": r["clv"]["POST_5M"]["mean"],
                "clv5_pos": r["clv"]["POST_5M"]["p_positive"],
                "clv5_cov": r["clv"]["POST_5M"]["coverage"],
                "clvC_n": r["clv"]["CLOSING"]["n"],
                "clvC_mean": r["clv"]["CLOSING"]["mean"],
                "clvC_pos": r["clv"]["CLOSING"]["p_positive"],
                "clvC_cov": r["clv"]["CLOSING"]["coverage"],
                "roi_ci95_lo": (r["roi_boot"]["ci95"][0] if r["roi_boot"] else None),
                "roi_ci95_hi": (r["roi_boot"]["ci95"][1] if r["roi_boot"] else None),
                "roi_p_pos": (r["roi_boot"]["p_pos"] if r["roi_boot"] else None),
                "clv5_ci95_lo": (r["clv5_boot"]["ci95"][0] if r["clv5_boot"] else None),
                "clv5_ci95_hi": (r["clv5_boot"]["ci95"][1] if r["clv5_boot"] else None),
                "clvC_ci95_lo": (r["clvC_boot"]["ci95"][0] if r["clvC_boot"] else None),
                "clvC_ci95_hi": (r["clvC_boot"]["ci95"][1] if r["clvC_boot"] else None),
            }
        )
    with (out / f"h3bup_temporal_slices_{run}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)
    with (out / f"h3bup_temporal_rolling7d_{run}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rolling[0].keys()))
        w.writeheader()
        w.writerows(rolling)
    with (out / f"h3bup_temporal_equity_{run}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(equity[0].keys()))
        w.writeheader()
        w.writerows(equity)

    def row_by_id(sid: str) -> Dict[str, Any]:
        return next(r for r in results if r["id"] == sid)

    full = row_by_id("full")
    s10 = row_by_id("era_stake10")
    s2 = row_by_id("era_stake2")
    l3 = row_by_id("last_3d")
    l5 = row_by_id("last_5d")
    l7 = row_by_id("last_7d")
    s2e = row_by_id("stake2_early")
    s2l = row_by_id("stake2_late")
    l1 = row_by_id("last_1d")

    def d_roi(r: Dict[str, Any]) -> str:
        if r["roi"] is None or full["roi"] is None:
            return "—"
        return f"{(r['roi'] - full['roi']) * 100:+.1f} pp"

    def d_clv(r: Dict[str, Any], window: str) -> str:
        a_ = r["clv"][window]["mean"]
        b = full["clv"][window]["mean"]
        if a_ is None or b is None:
            return "—"
        return f"{a_ - b:+.2f}"

    def clv_cell(r: Dict[str, Any], window: str) -> str:
        c = r["clv"][window]
        return f"{fmt_clv(c['mean'])} (n={c['n']}, pos={pct(c['p_positive'])})"

    lines: List[str] = []
    a = lines.append
    a("# H3BUP_vNext — Recortes temporais (CLV 5m + Closing + ROI)\n")
    a(f"- **Freeze:** `{run}` · as-of `2026-08-07T14:25:06Z`")
    a(f"- **N total:** {len(rows)} · stake2 desde `{STAKE2_START.isoformat()}`")
    a("- **Foco:** ROI + **CLV POST_5M** + **CLV CLOSING** (VALID_STRICT)\n")
    a("> Read-only. ROI = P&L settled / stake resolved (void no denominador).\n")

    a("## 1) Recortes principais\n")
    a("| Recorte | N | Settled/Open | P&L | ROI | IC95 ROI | CLV5 mean (n/%pos) | CLV Closing mean (n/%pos) | WR |")
    a("|---|---:|---:|---:|---:|---|---:|---:|---:|")
    main_ids = [
        "full",
        "era_stake10",
        "era_stake2",
        "stake2_early",
        "stake2_late",
        "last_7d",
        "last_5d",
        "last_3d",
        "last_2d",
        "last_1d",
        "from_0731",
        "week_2026-W31",
        "week_2026-W32",
    ]
    for sid in main_ids:
        try:
            r = row_by_id(sid)
        except StopIteration:
            continue
        ci = "—"
        if r["roi_boot"]:
            ci = f"[{pct(r['roi_boot']['ci95'][0])}, {pct(r['roi_boot']['ci95'][1])}]"
        c5 = clv_cell(r, "POST_5M")
        cc = clv_cell(r, "CLOSING")
        a(
            f"| {r['label']} | {r['n']} | {r['settled']}/{r['open']} | {money(r['pnl'])} | "
            f"**{pct(r['roi'])}** | {ci} | {c5} | {cc} | {pct(r['wr'])} |"
        )
    a("")

    a("## 2) A estratégia melhorou nos últimos dias?\n")
    a("| Comparação | N | ROI | Δ ROI vs full | CLV5 | Δ CLV5 | CLV Closing | Δ Closing |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, r in [
        ("Full período", full),
        ("Era stake10", s10),
        ("Era stake2", s2),
        ("stake2 1ª metade", s2e),
        ("stake2 2ª metade", s2l),
        ("Últimos 7d", l7),
        ("Últimos 5d", l5),
        ("Últimos 3d", l3),
        ("Últimos 1d", l1),
    ]:
        a(
            f"| {label} | {r['n']} | {pct(r['roi'])} | {d_roi(r)} | "
            f"{fmt_clv(r['clv']['POST_5M']['mean'])} | {d_clv(r, 'POST_5M')} | "
            f"{fmt_clv(r['clv']['CLOSING']['mean'])} | {d_clv(r, 'CLOSING')} |"
        )
    a("")
    a("### Leitura objetiva\n")
    a(
        f"1. **ROI recente:** últimos 3d = **{pct(l3['roi'])}** vs full **{pct(full['roi'])}** "
        f"vs stake10 **{pct(s10['roi'])}**."
    )
    a(
        f"2. **stake2 late vs early:** ROI {pct(s2l['roi'])} vs {pct(s2e['roi'])} "
        f"(P&L late {money(s2l['pnl'])}, early {money(s2e['pnl'])})."
    )
    a(
        f"3. **CLV POST_5M:** últimos 3d {fmt_clv(l3['clv']['POST_5M']['mean'])} "
        f"(n={l3['clv']['POST_5M']['n']}, pos={pct(l3['clv']['POST_5M']['p_positive'])}) "
        f"vs full {fmt_clv(full['clv']['POST_5M']['mean'])}."
    )
    a(
        f"4. **CLV CLOSING:** últimos 3d {fmt_clv(l3['clv']['CLOSING']['mean'])} "
        f"(n={l3['clv']['CLOSING']['n']}, pos={pct(l3['clv']['CLOSING']['p_positive'])}) "
        f"vs full {fmt_clv(full['clv']['CLOSING']['mean'])} "
        f"(Δ {d_clv(l3, 'CLOSING')} pp)."
    )
    clv5_worse = (
        l3["clv"]["POST_5M"]["mean"] is not None
        and full["clv"]["POST_5M"]["mean"] is not None
        and l3["clv"]["POST_5M"]["mean"] <= full["clv"]["POST_5M"]["mean"]
    )
    clvC_worse = (
        l3["clv"]["CLOSING"]["mean"] is not None
        and full["clv"]["CLOSING"]["mean"] is not None
        and l3["clv"]["CLOSING"]["mean"] <= full["clv"]["CLOSING"]["mean"]
    )
    if clv5_worse and clvC_worse:
        a("5. **CLV 5m e Closing** nos últimos 3d estão **iguais ou piores** que o full → ROI recente **não** é confirmado por preço.")
    elif clv5_worse or clvC_worse:
        a("5. Um dos CLVs (5m ou Closing) não confirma a melhoria de ROI; ler as duas janelas em conjunto.")
    else:
        a("5. CLV 5m e Closing recentes melhoram vs full — evidência preliminar de preço também melhor.")
    a("6. N dos recortes curtos é pequeno — IC95 largos; não fechar decisão operacional só com 1–3 dias.\n")

    a("## 3) Dia a dia (ROI + CLV5 + Closing)\n")
    a("| Dia | N | F/NF | Era | P&L | ROI | CLV5 (n/%pos) | CLV Closing (n/%pos) | WR |")
    a("|---|---:|---:|---|---:|---:|---:|---:|---:|")
    for r in results:
        if not r["id"].startswith("day_"):
            continue
        eras: Dict[str, int] = {}
        day = r["id"].replace("day_", "")
        for x in by_day[day]:
            eras[x["_era"]] = eras.get(x["_era"], 0) + 1
        era_dom = max(eras, key=eras.get) if eras else "—"
        a(
            f"| {day} | {r['n']} | {r['friendly_n']}/{r['nf_n']} | {era_dom} | "
            f"{money(r['pnl'])} | {pct(r['roi'])} | {clv_cell(r, 'POST_5M')} | "
            f"{clv_cell(r, 'CLOSING')} | {pct(r['wr'])} |"
        )
    a("")

    a("## 4) Rolling 7 dias (ROI + CLV5 + Closing)\n")
    a("| Fim (UTC) | N | ROI | P&L | CLV5 mean (n/%pos) | CLV Closing mean (n/%pos) | WR |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    for r in rolling:
        c5 = f"{fmt_clv(r['clv5_mean'])} ({r['clv5_n']}/{pct(r['clv5_pos'])})"
        cc = f"{fmt_clv(r['clvC_mean'])} ({r['clvC_n']}/{pct(r['clvC_pos'])})"
        a(
            f"| {r['end_day']} | {r['n']} | {pct(r['roi'])} | {money(r['pnl'])} | "
            f"{c5} | {cc} | {pct(r['wr'])} |"
        )
    a("")
    a("Se ROI rolling sobe mas **CLV5 e/ou Closing** continuam ≤0, a melhoria é **frágil**.\n")

    a("## 5) Semanas ISO\n")
    a("| Semana | N | ROI | CLV5 (n/%pos) | CLV Closing (n/%pos) | P&L |")
    a("|---|---:|---:|---:|---:|---:|")
    for r in results:
        if not r["id"].startswith("week_"):
            continue
        a(
            f"| {r['label']} | {r['n']} | {pct(r['roi'])} | {clv_cell(r, 'POST_5M')} | "
            f"{clv_cell(r, 'CLOSING')} | {money(r['pnl'])} |"
        )
    a("")

    a("## 6) Bootstrap CLV Closing (recortes-chave)\n")
    a("| Recorte | N valid Closing | Mean | IC95 | P(CLV>0) |")
    a("|---|---:|---:|---|---:|")
    for label, r in [
        ("Full", full),
        ("stake10", s10),
        ("stake2", s2),
        ("Últimos 7d", l7),
        ("Últimos 5d", l5),
        ("Últimos 3d", l3),
    ]:
        b = r.get("clvC_boot")
        n_v = r["clv"]["CLOSING"]["n"]
        if not b:
            a(f"| {label} | {n_v} | — | — | — |")
            continue
        a(
            f"| {label} | {n_v} | {b['mean']:.2f}% | "
            f"[{b['ci95'][0]:.2f}%, {b['ci95'][1]:.2f}%] | {pct(b['p_pos'])} |"
        )
    a("")

    improved_roi = l3["roi"] is not None and full["roi"] is not None and l3["roi"] > full["roi"] + 0.02
    improved_clv5 = (
        l3["clv"]["POST_5M"]["mean"] is not None
        and full["clv"]["POST_5M"]["mean"] is not None
        and l3["clv"]["POST_5M"]["mean"] > full["clv"]["POST_5M"]["mean"] + 0.5
    )
    improved_clvC = (
        l3["clv"]["CLOSING"]["mean"] is not None
        and full["clv"]["CLOSING"]["mean"] is not None
        and l3["clv"]["CLOSING"]["mean"] > full["clv"]["CLOSING"]["mean"] + 0.5
    )
    improved_clv = improved_clv5 and improved_clvC
    if improved_roi and improved_clv:
        verdict = "RECENT_IMPROVEMENT_ROI_AND_CLV_PRELIMINARY"
    elif improved_roi and (improved_clv5 or improved_clvC):
        verdict = "RECENT_ROI_UP_CLV_MIXED"
    elif improved_roi and not improved_clv5 and not improved_clvC:
        verdict = "RECENT_ROI_UP_BUT_CLV_NOT_CONFIRMED"
    elif not improved_roi and improved_clv:
        verdict = "RECENT_CLV_UP_BUT_ROI_LAGS"
    else:
        verdict = "NO_CLEAR_RECENT_IMPROVEMENT"

    a("## 7) Veredicto do recorte\n")
    a(f"**`{verdict}`**\n")
    a(f"- Últimos 3d ROI={pct(l3['roi'])} (n={l3['n']}, settled={l3['settled']})")
    a(
        f"- Últimos 3d CLV5={fmt_clv(l3['clv']['POST_5M']['mean'])} "
        f"(n={l3['clv']['POST_5M']['n']}, pos={pct(l3['clv']['POST_5M']['p_positive'])})"
    )
    a(
        f"- Últimos 3d CLV Closing={fmt_clv(l3['clv']['CLOSING']['mean'])} "
        f"(n={l3['clv']['CLOSING']['n']}, pos={pct(l3['clv']['CLOSING']['p_positive'])})"
    )
    a(
        f"- Full: ROI={pct(full['roi'])} · CLV5={fmt_clv(full['clv']['POST_5M']['mean'])} · "
        f"Closing={fmt_clv(full['clv']['CLOSING']['mean'])}"
    )
    a(f"- stake2 ROI={pct(s2['roi'])} · stake10 ROI={pct(s10['roi'])}\n")

    a("## 8) Artefactos\n")
    a(f"- `logs/h3bup_strategy_results/20260807/temporal_{run}/`")
    a(f"- Freeze base: `logs/h3bup_friendly_analysis/20260807/{run}/`\n")

    text = "\n".join(lines) + "\n"
    (out / f"h3bup_temporal_report_{run}.md").write_text(text, encoding="utf-8")
    (docs / "h3bup_temporal_slices_20260807.md").write_text(text, encoding="utf-8")

    exec_sum = f"""# Executive — Recortes temporais H3BUP (CLV5 + Closing + ROI)

- **Veredicto:** `{verdict}`
- Full: ROI {pct(full['roi'])} · CLV5 {fmt_clv(full['clv']['POST_5M']['mean'])} · **Closing {fmt_clv(full['clv']['CLOSING']['mean'])}**
- stake10 → stake2: ROI {pct(s10['roi'])} → {pct(s2['roi'])} · Closing {fmt_clv(s10['clv']['CLOSING']['mean'])} → {fmt_clv(s2['clv']['CLOSING']['mean'])}
- Últimos 7d: ROI {pct(l7['roi'])} · CLV5 {fmt_clv(l7['clv']['POST_5M']['mean'])} · Closing {fmt_clv(l7['clv']['CLOSING']['mean'])}
- Últimos 3d: ROI {pct(l3['roi'])} · CLV5 {fmt_clv(l3['clv']['POST_5M']['mean'])} · **Closing {fmt_clv(l3['clv']['CLOSING']['mean'])}** (n={l3['n']})
- stake2 late vs early: ROI {pct(s2l['roi'])} vs {pct(s2e['roi'])} · Closing {fmt_clv(s2l['clv']['CLOSING']['mean'])} vs {fmt_clv(s2e['clv']['CLOSING']['mean'])}

ROI recente pode parecer melhor; **CLV Closing** precisa confirmar. N curto ⇒ preliminar.
"""
    (out / f"h3bup_temporal_executive_{run}.md").write_text(exec_sum, encoding="utf-8")
    (docs / "h3bup_temporal_slices_executive_20260807.md").write_text(exec_sum, encoding="utf-8")

    print("VERDICT", verdict)
    print("full roi/clv5/close", full["roi"], full["clv"]["POST_5M"]["mean"], full["clv"]["CLOSING"]["mean"])
    print("last3 roi/clv5/close", l3["roi"], l3["clv"]["POST_5M"]["mean"], l3["clv"]["CLOSING"]["mean"])
    print("WROTE", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
