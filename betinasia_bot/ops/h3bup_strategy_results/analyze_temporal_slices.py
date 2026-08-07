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
    # CLV5 boot
    clv5_vals = [
        fnum(r.get("clv_post_5m"))
        for r in rs
        if fbool(r.get("clv_post_5m_valid_strict")) and fnum(r.get("clv_post_5m")) is not None
    ]
    clv5_vals = [v for v in clv5_vals if v is not None]
    clv5_boot = None
    if clv5_vals:
        rng = random.Random(77 + len(clv5_vals))
        out = [statistics.mean(rng.choices(clv5_vals, k=len(clv5_vals))) for _ in range(3000)]
        out.sort()
        clv5_boot = {
            "mean": statistics.mean(out),
            "ci95": [out[int(0.025 * len(out))], out[int(0.975 * len(out)) - 1]],
            "p_pos": sum(1 for v in out if v > 0) / len(out),
        }
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
                "clvC_mean": m["clv"]["CLOSING"]["mean"],
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
                "clvC": r["clv"]["CLOSING"],
                "roi_boot": r["roi_boot"],
                "clv5_boot": r["clv5_boot"],
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

    # CSV main slices (exclude day_* from primary table? include all in one csv)
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
                "roi_ci95_lo": (r["roi_boot"]["ci95"][0] if r["roi_boot"] else None),
                "roi_ci95_hi": (r["roi_boot"]["ci95"][1] if r["roi_boot"] else None),
                "roi_p_pos": (r["roi_boot"]["p_pos"] if r["roi_boot"] else None),
                "clv5_ci95_lo": (r["clv5_boot"]["ci95"][0] if r["clv5_boot"] else None),
                "clv5_ci95_hi": (r["clv5_boot"]["ci95"][1] if r["clv5_boot"] else None),
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

    # Report
    lines: List[str] = []
    a = lines.append
    a("# H3BUP_vNext — Recortes temporais (CLV + ROI)\n")
    a(f"- **Freeze:** `{run}` · as-of `2026-08-07T14:25:06Z`")
    a(f"- **N total:** {len(rows)} · stake2 desde `{STAKE2_START.isoformat()}`")
    a("- **Foco:** comparar se os últimos dias melhoraram em **ROI e CLV**\n")
    a("> Read-only. CLV = VALID_STRICT. ROI = P&L settled / stake resolved.\n")

    a("## 1) Recortes principais\n")
    a("| Recorte | N | Settled/Open | P&L | ROI | IC95 ROI | CLV5 mean (n) | CLV Close | WR |")
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
        c5 = f"{fmt_clv(r['clv']['POST_5M']['mean'])} ({r['clv']['POST_5M']['n']})"
        a(
            f"| {r['label']} | {r['n']} | {r['settled']}/{r['open']} | {money(r['pnl'])} | "
            f"**{pct(r['roi'])}** | {ci} | {c5} | {fmt_clv(r['clv']['CLOSING']['mean'])} | {pct(r['wr'])} |"
        )
    a("")

    # Highlight comparison
    full = row_by_id("full")
    s10 = row_by_id("era_stake10")
    s2 = row_by_id("era_stake2")
    l3 = row_by_id("last_3d")
    l7 = row_by_id("last_7d")
    s2e = row_by_id("stake2_early")
    s2l = row_by_id("stake2_late")

    a("## 2) A estratégia melhorou nos últimos dias?\n")
    a("| Comparação | ROI | Δ vs full | CLV5 | Δ CLV5 vs full |")
    a("|---|---:|---:|---:|---:|")

    def delta(a_: Optional[float], b: Optional[float]) -> str:
        if a_ is None or b is None:
            return "—"
        return f"{(a_ - b) * 100:+.1f} pp"

    for label, r in [
        ("Full período", full),
        ("Era stake10", s10),
        ("Era stake2", s2),
        ("stake2 1ª metade", s2e),
        ("stake2 2ª metade", s2l),
        ("Últimos 7d", l7),
        ("Últimos 3d", l3),
        ("Últimos 1d", row_by_id("last_1d")),
    ]:
        a(
            f"| {label} | {pct(r['roi'])} | {delta(r['roi'], full['roi'])} | "
            f"{fmt_clv(r['clv']['POST_5M']['mean'])} | "
            f"{delta((r['clv']['POST_5M']['mean'] or 0)/100 if r['clv']['POST_5M']['mean'] is not None else None, (full['clv']['POST_5M']['mean'] or 0)/100 if full['clv']['POST_5M']['mean'] is not None else None)} |"
        )
    # Fix delta for CLV - they're already in percent units (e.g. -1.35 meaning -1.35%), not fractions
    # Actually in metrics clv mean is like -1.35 (percent points already from clv_raw_pct).
    # delta() multiplies by 100 assuming fraction - WRONG for CLV.
    # I'll fix the comparison table below by rewriting CLV delta properly in a cleaner section.

    a("")
    a("### Leitura objetiva\n")
    a(
        f"1. **ROI recente:** últimos 3d = **{pct(l3['roi'])}** vs full **{pct(full['roi'])}** "
        f"vs stake10 **{pct(s10['roi'])}**."
    )
    a(
        f"2. **stake2 late vs early:** {pct(s2l['roi'])} vs {pct(s2e['roi'])} "
        f"(P&L late {money(s2l['pnl'])}, early {money(s2e['pnl'])})."
    )
    a(
        f"3. **CLV5 recente:** últimos 3d {fmt_clv(l3['clv']['POST_5M']['mean'])} (n={l3['clv']['POST_5M']['n']}) "
        f"vs full {fmt_clv(full['clv']['POST_5M']['mean'])} — "
        + (
            "CLV também melhorou (menos negativo / positivo)."
            if (l3["clv"]["POST_5M"]["mean"] is not None and full["clv"]["POST_5M"]["mean"] is not None and l3["clv"]["POST_5M"]["mean"] > full["clv"]["POST_5M"]["mean"])
            else "CLV **não** confirma melhoria de preço (igual ou pior)."
        )
    )
    a("4. Melhoria só em ROI sem CLV positivo = possível **sorte de settlement / sample curto**, não edge estrutural.")
    a("5. N dos recortes curtos é pequeno — IC95 largos; não fechar decisão operacional só com 1–3 dias.\n")

    a("## 3) Dia a dia\n")
    a("| Dia | N | F/NF | Era dom. | P&L | ROI | CLV5 (n) | CLV Close (n) | WR |")
    a("|---|---:|---:|---|---:|---:|---:|---:|---:|")
    for r in results:
        if not r["id"].startswith("day_"):
            continue
        eras = {}
        day = r["id"].replace("day_", "")
        for x in by_day[day]:
            eras[x["_era"]] = eras.get(x["_era"], 0) + 1
        era_dom = max(eras, key=eras.get) if eras else "—"
        c5 = f"{fmt_clv(r['clv']['POST_5M']['mean'])} ({r['clv']['POST_5M']['n']})"
        cc = f"{fmt_clv(r['clv']['CLOSING']['mean'])} ({r['clv']['CLOSING']['n']})"
        a(
            f"| {day} | {r['n']} | {r['friendly_n']}/{r['nf_n']} | {era_dom} | "
            f"{money(r['pnl'])} | {pct(r['roi'])} | {c5} | {cc} | {pct(r['wr'])} |"
        )
    a("")

    a("## 4) Rolling 7 dias (série)\n")
    a("| Fim (UTC) | N | ROI | P&L | CLV5 mean (n) | CLV Close | WR |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    for r in rolling:
        c5 = f"{fmt_clv(r['clv5_mean'])} ({r['clv5_n']})"
        a(
            f"| {r['end_day']} | {r['n']} | {pct(r['roi'])} | {money(r['pnl'])} | "
            f"{c5} | {fmt_clv(r['clvC_mean'])} | {pct(r['wr'])} |"
        )
    a("")
    a("Se a curva rolling 7d de ROI sobe mas CLV5 rolling continua ≤0, a melhoria é **frágil**.\n")

    a("## 5) Semanas ISO\n")
    a("| Semana | N | ROI | CLV5 | CLV Close | P&L |")
    a("|---|---:|---:|---:|---:|---:|")
    for r in results:
        if not r["id"].startswith("week_"):
            continue
        a(
            f"| {r['label']} | {r['n']} | {pct(r['roi'])} | {fmt_clv(r['clv']['POST_5M']['mean'])} | "
            f"{fmt_clv(r['clv']['CLOSING']['mean'])} | {money(r['pnl'])} |"
        )
    a("")

    # Verdict banner
    improved_roi = (l3["roi"] is not None and full["roi"] is not None and l3["roi"] > full["roi"] + 0.02)
    improved_clv = (
        l3["clv"]["POST_5M"]["mean"] is not None
        and full["clv"]["POST_5M"]["mean"] is not None
        and l3["clv"]["POST_5M"]["mean"] > full["clv"]["POST_5M"]["mean"] + 0.5
    )
    if improved_roi and improved_clv:
        verdict = "RECENT_IMPROVEMENT_ROI_AND_CLV_PRELIMINARY"
    elif improved_roi and not improved_clv:
        verdict = "RECENT_ROI_UP_BUT_CLV_NOT_CONFIRMED"
    elif not improved_roi and improved_clv:
        verdict = "RECENT_CLV_UP_BUT_ROI_LAGS"
    else:
        verdict = "NO_CLEAR_RECENT_IMPROVEMENT"

    a("## 6) Veredicto do recorte\n")
    a(f"**`{verdict}`**\n")
    a(f"- Últimos 3d ROI={pct(l3['roi'])} (n={l3['n']}, settled={l3['settled']})")
    a(f"- Últimos 3d CLV5={fmt_clv(l3['clv']['POST_5M']['mean'])} (n_valid={l3['clv']['POST_5M']['n']})")
    a(f"- Full ROI={pct(full['roi'])} · Full CLV5={fmt_clv(full['clv']['POST_5M']['mean'])}")
    a(f"- stake2 ROI={pct(s2['roi'])} · stake10 ROI={pct(s10['roi'])}\n")

    a("## 7) Artefactos\n")
    a(f"- `logs/h3bup_strategy_results/20260807/temporal_{run}/`")
    a(f"- Freeze base: `logs/h3bup_friendly_analysis/20260807/{run}/`\n")

    text = "\n".join(lines) + "\n"
    # Fix the broken CLV delta table section - replace section 2 comparison table with clean one
    # Actually rebuild section 2 comparison more cleanly by rewriting file content for that part
    text_lines = text.splitlines()
    out_lines: List[str] = []
    skip = False
    for i, line in enumerate(text_lines):
        if line.startswith("## 2) A estratégia melhorou"):
            out_lines.append(line)
            out_lines.append("")
            out_lines.append("| Comparação | N | ROI | Δ ROI vs full | CLV5 | Δ CLV5 vs full (pp) |")
            out_lines.append("|---|---:|---:|---:|---:|---:|")
            for label, r in [
                ("Full período", full),
                ("Era stake10", s10),
                ("Era stake2", s2),
                ("stake2 1ª metade", s2e),
                ("stake2 2ª metade", s2l),
                ("Últimos 7d", l7),
                ("Últimos 3d", l3),
                ("Últimos 1d", row_by_id("last_1d")),
            ]:
                d_roi = "—"
                if r["roi"] is not None and full["roi"] is not None:
                    d_roi = f"{(r['roi'] - full['roi']) * 100:+.1f} pp"
                d_clv = "—"
                if r["clv"]["POST_5M"]["mean"] is not None and full["clv"]["POST_5M"]["mean"] is not None:
                    d_clv = f"{r['clv']['POST_5M']['mean'] - full['clv']['POST_5M']['mean']:+.2f}"
                out_lines.append(
                    f"| {label} | {r['n']} | {pct(r['roi'])} | {d_roi} | "
                    f"{fmt_clv(r['clv']['POST_5M']['mean'])} | {d_clv} |"
                )
            skip = True
            continue
        if skip:
            if line.startswith("### Leitura objetiva"):
                skip = False
                out_lines.append("")
                out_lines.append(line)
            continue
        out_lines.append(line)
    text = "\n".join(out_lines) + "\n"

    (out / f"h3bup_temporal_report_{run}.md").write_text(text, encoding="utf-8")
    (docs / "h3bup_temporal_slices_20260807.md").write_text(text, encoding="utf-8")

    exec_sum = f"""# Executive — Recortes temporais H3BUP (2026-08-07)

- **Veredicto:** `{verdict}`
- Full ROI {pct(full['roi'])} · CLV5 {fmt_clv(full['clv']['POST_5M']['mean'])}
- stake10 ROI {pct(s10['roi'])} → stake2 ROI {pct(s2['roi'])}
- Últimos 7d ROI {pct(l7['roi'])} · CLV5 {fmt_clv(l7['clv']['POST_5M']['mean'])}
- Últimos 3d ROI {pct(l3['roi'])} · CLV5 {fmt_clv(l3['clv']['POST_5M']['mean'])} (n={l3['n']})
- stake2 late ROI {pct(s2l['roi'])} vs early {pct(s2e['roi'])}

ROI recente pode parecer melhor; validar sempre contra CLV. N curto ⇒ preliminar.
"""
    (out / f"h3bup_temporal_executive_{run}.md").write_text(exec_sum, encoding="utf-8")
    (docs / "h3bup_temporal_slices_executive_20260807.md").write_text(exec_sum, encoding="utf-8")

    print("VERDICT", verdict)
    print("full", full["roi"], full["clv"]["POST_5M"]["mean"])
    print("last3", l3["roi"], l3["clv"]["POST_5M"]["mean"], "n", l3["n"])
    print("last7", l7["roi"], l7["clv"]["POST_5M"]["mean"])
    print("s2", s2["roi"], "s10", s10["roi"])
    print("s2e", s2e["roi"], "s2l", s2l["roi"])
    print("WROTE", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
