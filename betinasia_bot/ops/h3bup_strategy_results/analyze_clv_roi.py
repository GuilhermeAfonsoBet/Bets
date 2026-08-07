#!/usr/bin/env python3
"""Deep CLV + ROI analysis on a Friendly freeze (read-only)."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def era_of(r: Dict[str, Any]) -> str:
    s = fnum(r.get("stake"))
    if s is None:
        return "NA"
    if abs(s - 10) < 1e-9:
        return "stake10"
    if abs(s - 2) < 1e-9:
        return "stake2"
    return f"other_{s}"


WINDOWS = [
    ("POST_5M", "clv_post_5m", "clv_post_5m_valid_strict"),
    ("POST_15M", "clv_post_15m", "clv_post_15m_valid_strict"),
    ("CLOSING", "clv_closing", "clv_closing_valid_strict"),
]


def subset_metrics(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
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
    return {
        "n": len(rs),
        "events": len({r.get("event_id") for r in rs}),
        "settled": len(settled),
        "void": len(voids),
        "open": len(opens),
        "missing": len(missing),
        "stake_placed": stake_pl,
        "stake_resolved": stake_res,
        "pnl": pnl,
        "roi": (pnl / stake_res if stake_res else None),
        "wr": wr,
        "clv": clv,
    }


def boot_mean(vals: List[float], n: int = 5000, seed: int = 1) -> Optional[Dict[str, Any]]:
    if not vals:
        return None
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.append(statistics.mean(rng.choices(vals, k=len(vals))))
    out.sort()
    return {
        "mean": statistics.mean(out),
        "ci95": [out[int(0.025 * len(out))], out[int(0.975 * len(out)) - 1]],
        "p_pos": sum(1 for v in out if v > 0) / len(out),
    }


def boot_roi(settled: List[Dict[str, Any]], n: int = 5000, seed: int = 2) -> Optional[Dict[str, Any]]:
    if not settled:
        return None
    rng = random.Random(seed)
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in settled:
        by[str(r.get("event_id"))].append(r)
    keys = list(by)
    vals = []
    for _ in range(n):
        sp = ss = 0.0
        for k in rng.choices(keys, k=len(keys)):
            for r in by[k]:
                sp += r["_pnl"]
                ss += r["_stake"]
        if ss:
            vals.append(sp / ss)
    vals.sort()
    return {
        "mean": statistics.mean(vals),
        "ci95": [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]],
        "p_pos": sum(1 for v in vals if v > 0) / len(vals),
    }


def clv_roi_joint(rs: List[Dict[str, Any]], field: str, validf: str) -> Dict[str, Any]:
    bins_def = [
        ("CLV < -3%", None, -3.0),
        ("[-3%, -1%)", -3.0, -1.0),
        ("[-1%, 0%)", -1.0, 0.0),
        ("[0%, +1%)", 0.0, 1.0),
        ("CLV ≥ +1%", 1.0, None),
    ]
    pts = []
    for r in rs:
        if r.get("settlement_status") != "SETTLED_DECIDED":
            continue
        if not fbool(r.get(validf)):
            continue
        c = fnum(r.get(field))
        if c is None or r["_stake"] <= 0:
            continue
        pts.append((c, r["_pnl"] / r["_stake"], r))
    corr = None
    if len(pts) >= 5:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
        corr = (num / den) if den else None
    same = sum(1 for c, roi, _ in pts if (c > 0 and roi > 0) or (c < 0 and roi < 0))
    out_bins = []
    for name, lo, hi in bins_def:
        sub = [p for p in pts if (lo is None or p[0] >= lo) and (hi is None or p[0] < hi)]
        if not sub:
            out_bins.append({"bin": name, "n": 0, "clv_mean": None, "roi": None, "wr": None})
            continue
        pnl = sum(p[2]["_pnl"] for p in sub)
        stake = sum(p[2]["_stake"] for p in sub)
        out_bins.append(
            {
                "bin": name,
                "n": len(sub),
                "clv_mean": statistics.mean(p[0] for p in sub),
                "roi": (pnl / stake if stake else None),
                "wr": sum(1 for p in sub if p[2]["_pnl"] > 0) / len(sub),
            }
        )
    return {
        "n": len(pts),
        "corr_pearson": corr,
        "sign_concordance": (same / len(pts) if pts else None),
        "bins": out_bins,
    }


def main() -> int:
    random.seed(20260807)
    run = "a27c1dc4ab52"
    freeze = Path(f"/workspace/betinasia_bot/logs/h3bup_friendly_analysis/20260807/{run}")
    out = Path(f"/workspace/betinasia_bot/logs/h3bup_strategy_results/20260807/clvroi_{run}")
    docs = Path("/workspace/betinasia_bot/docs")
    out.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader((freeze / f"h3bup_friendly_order_level_{run}.csv").open()))
    for r in rows:
        r["_era"] = era_of(r)
        r["_pnl"] = fnum(r.get("pnl")) or 0.0
        r["_stake"] = fnum(r.get("stake")) or 0.0

    cohorts = {
        "ALL": rows,
        "FRIENDLY": [r for r in rows if r["friendly_class"] == "FRIENDLY"],
        "NON_FRIENDLY": [r for r in rows if r["friendly_class"] == "NON_FRIENDLY"],
        "stake10": [r for r in rows if r["_era"] == "stake10"],
        "stake2": [r for r in rows if r["_era"] == "stake2"],
        "stake10_FRIENDLY": [r for r in rows if r["_era"] == "stake10" and r["friendly_class"] == "FRIENDLY"],
        "stake10_NON_FRIENDLY": [r for r in rows if r["_era"] == "stake10" and r["friendly_class"] == "NON_FRIENDLY"],
        "stake2_FRIENDLY": [r for r in rows if r["_era"] == "stake2" and r["friendly_class"] == "FRIENDLY"],
        "stake2_NON_FRIENDLY": [r for r in rows if r["_era"] == "stake2" and r["friendly_class"] == "NON_FRIENDLY"],
    }
    metrics = {k: subset_metrics(v) for k, v in cohorts.items()}

    roi_boots = {}
    for k in ["ALL", "FRIENDLY", "NON_FRIENDLY", "stake10", "stake2"]:
        settled = [r for r in cohorts[k] if r.get("settlement_status") == "SETTLED_DECIDED"]
        roi_boots[k] = boot_roi(settled)

    clv_boots: Dict[str, Dict[str, Any]] = {}
    for k in ["ALL", "FRIENDLY", "NON_FRIENDLY", "stake10", "stake2"]:
        clv_boots[k] = {}
        for w, field, validf in WINDOWS:
            vals = [
                fnum(r.get(field))
                for r in cohorts[k]
                if fbool(r.get(validf)) and fnum(r.get(field)) is not None
            ]
            vals = [v for v in vals if v is not None]
            clv_boots[k][w] = boot_mean(vals, seed=abs(hash(k + w)) % 100000)

    joints = {w: clv_roi_joint(rows, field, validf) for w, field, validf in WINDOWS}

    by_day: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_day[str(r["created_at_utc"])[:10]].append(r)
    daily = []
    for d in sorted(by_day):
        m = subset_metrics(by_day[d])
        daily.append(
            {
                "day": d,
                "n": m["n"],
                "stake_era": dict(Counter(r["_era"] for r in by_day[d])),
                "pnl": m["pnl"],
                "roi": m["roi"],
                "clv5_mean": m["clv"]["POST_5M"]["mean"],
                "clv5_n": m["clv"]["POST_5M"]["n"],
                "clvC_mean": m["clv"]["CLOSING"]["mean"],
                "clvC_n": m["clv"]["CLOSING"]["n"],
                "friendly_n": sum(1 for r in by_day[d] if r["friendly_class"] == "FRIENDLY"),
                "nf_n": sum(1 for r in by_day[d] if r["friendly_class"] == "NON_FRIENDLY"),
            }
        )

    bundle = {
        "meta": {
            "freeze_run": run,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "cutoff_utc": "2026-08-07T14:25:06+00:00",
            "n": len(rows),
            "focus": ["CLV_VALID_STRICT", "ROI_resolved", "stake_era"],
        },
        "metrics": metrics,
        "roi_boots_event_cluster": roi_boots,
        "clv_boots": clv_boots,
        "clv_roi_joints": joints,
        "daily": daily,
    }
    (out / f"h3bup_clv_roi_bundle_{run}.json").write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")

    # cohort CSV
    perf_flat = []
    for k, m in metrics.items():
        perf_flat.append(
            {
                "cohort": k,
                "n": m["n"],
                "settled": m["settled"],
                "open": m["open"],
                "void": m["void"],
                "stake_placed": m["stake_placed"],
                "stake_resolved": m["stake_resolved"],
                "pnl": m["pnl"],
                "roi": m["roi"],
                "wr": m["wr"],
                "clv5_n": m["clv"]["POST_5M"]["n"],
                "clv5_mean": m["clv"]["POST_5M"]["mean"],
                "clv5_med": m["clv"]["POST_5M"]["median"],
                "clv5_pos": m["clv"]["POST_5M"]["p_positive"],
                "clv5_cov": m["clv"]["POST_5M"]["coverage"],
                "clv15_mean": m["clv"]["POST_15M"]["mean"],
                "clvC_n": m["clv"]["CLOSING"]["n"],
                "clvC_mean": m["clv"]["CLOSING"]["mean"],
                "clvC_med": m["clv"]["CLOSING"]["median"],
                "clvC_pos": m["clv"]["CLOSING"]["p_positive"],
                "clvC_cov": m["clv"]["CLOSING"]["coverage"],
            }
        )
    with (out / f"h3bup_clv_roi_cohorts_{run}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(perf_flat[0].keys()))
        w.writeheader()
        w.writerows(perf_flat)

    joint_flat = []
    for wname, j in joints.items():
        for b in j["bins"]:
            joint_flat.append(
                {
                    "window": wname,
                    "corr": j["corr_pearson"],
                    "sign_concordance": j["sign_concordance"],
                    "n_joint": j["n"],
                    **b,
                }
            )
    with (out / f"h3bup_clv_roi_joint_bins_{run}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(joint_flat[0].keys()))
        w.writeheader()
        w.writerows(joint_flat)

    m_all, m_f, m_nf = metrics["ALL"], metrics["FRIENDLY"], metrics["NON_FRIENDLY"]
    m10, m2 = metrics["stake10"], metrics["stake2"]
    status = "CLV_NEGATIVE_ROI_SLIGHTLY_NEGATIVE_INSUFFICIENT_N"

    lines: List[str] = []
    a = lines.append
    a("# H3BUP_vNext — Análise CLV + ROI (até 2026-08-07)\n")
    a(f"- **Status:** `{status}`")
    a(f"- **Freeze:** `{run}` · cutoff `2026-08-07T14:25:06Z`")
    a(f"- **Universo:** LIVE_OK Back Pre `H3BUP_vNext_20260629` · N={len(rows)} (F={m_f['n']}, NF={m_nf['n']})")
    a(f"- **Eras:** stake10={m10['n']} · stake2={m2['n']}")
    a("- **Friendly pack status:** `NO_CLEAR_FRIENDLY_DIFFERENCE`\n")
    a("> Read-only. CLV oficial = **VALID_STRICT** apenas. ROI = P&L settled / stake resolved (void no denominador).\n")

    a("## 1) Veredicto dual (CLV e ROI)\n")
    a("| Métrica | All | stake10 | stake2 | Friendly | Non-Friendly |")
    a("|---|---:|---:|---:|---:|---:|")
    a(f"| LIVE_OK | {m_all['n']} | {m10['n']} | {m2['n']} | {m_f['n']} | {m_nf['n']} |")
    a(f"| Settled / Open | {m_all['settled']}/{m_all['open']} | {m10['settled']}/{m10['open']} | {m2['settled']}/{m2['open']} | {m_f['settled']}/{m_f['open']} | {m_nf['settled']}/{m_nf['open']} |")
    a(f"| Stake placed | {m_all['stake_placed']:.0f} | {m10['stake_placed']:.0f} | {m2['stake_placed']:.0f} | {m_f['stake_placed']:.0f} | {m_nf['stake_placed']:.0f} |")
    a(f"| P&L resolved | {money(m_all['pnl'])} | {money(m10['pnl'])} | {money(m2['pnl'])} | {money(m_f['pnl'])} | {money(m_nf['pnl'])} |")
    a(f"| **ROI resolved** | **{pct(m_all['roi'])}** | {pct(m10['roi'])} | {pct(m2['roi'])} | {pct(m_f['roi'])} | {pct(m_nf['roi'])} |")
    a(f"| WR settled | {pct(m_all['wr'])} | {pct(m10['wr'])} | {pct(m2['wr'])} | {pct(m_f['wr'])} | {pct(m_nf['wr'])} |")
    a(f"| CLV POST_5M mean | {fmt_clv(m_all['clv']['POST_5M']['mean'])} | {fmt_clv(m10['clv']['POST_5M']['mean'])} | {fmt_clv(m2['clv']['POST_5M']['mean'])} | {fmt_clv(m_f['clv']['POST_5M']['mean'])} | {fmt_clv(m_nf['clv']['POST_5M']['mean'])} |")
    a(f"| CLV POST_5M cov | {pct(m_all['clv']['POST_5M']['coverage'])} | {pct(m10['clv']['POST_5M']['coverage'])} | {pct(m2['clv']['POST_5M']['coverage'])} | {pct(m_f['clv']['POST_5M']['coverage'])} | {pct(m_nf['clv']['POST_5M']['coverage'])} |")
    a(f"| CLV CLOSING mean | {fmt_clv(m_all['clv']['CLOSING']['mean'])} | {fmt_clv(m10['clv']['CLOSING']['mean'])} | {fmt_clv(m2['clv']['CLOSING']['mean'])} | {fmt_clv(m_f['clv']['CLOSING']['mean'])} | {fmt_clv(m_nf['clv']['CLOSING']['mean'])} |")
    a(f"| CLV CLOSING %pos | {pct(m_all['clv']['CLOSING']['p_positive'])} | {pct(m10['clv']['CLOSING']['p_positive'])} | {pct(m2['clv']['CLOSING']['p_positive'])} | {pct(m_f['clv']['CLOSING']['p_positive'])} | {pct(m_nf['clv']['CLOSING']['p_positive'])} |")
    a("")
    a("**Leitura:**")
    a(f"1. **ROI** agregado **{pct(m_all['roi'])}** (P&L {money(m_all['pnl'])} / stake resolved {m_all['stake_resolved']:.0f}) — negativo, mas **melhor** que o freeze 01/Ago (−9.9%).")
    a("2. **CLV** é consistentemente **negativo** em POST_5M / POST_15M / CLOSING (médias < 0; % positivo baixo). Sinal mais estável: **sem evidência de edge de preço**.")
    a(f"3. Era **stake2** (ROI {pct(m2['roi'])}) vs **stake10** (ROI {pct(m10['roi'])}): stake10 continua pior; stake2 perto de flat/ligeiramente negativo.")
    a(f"4. Friendly vs NF em ROI quase iguais ({pct(m_f['roi'])} vs {pct(m_nf['roi'])}). Em CLV, Friendly está **pior** (POST_5M {fmt_clv(m_f['clv']['POST_5M']['mean'])} vs {fmt_clv(m_nf['clv']['POST_5M']['mean'])}), com coverage mais baixa.")
    a("")

    a("## 2) Inferência (bootstrap)\n")
    a("### ROI (event-cluster, settled)\n")
    a("| Cohort | Média | IC95 | P(ROI>0) |")
    a("|---|---:|---|---:|")
    for k in ["ALL", "stake10", "stake2", "FRIENDLY", "NON_FRIENDLY"]:
        b = roi_boots[k]
        assert b is not None
        a(f"| {k} | {pct(b['mean'])} | [{pct(b['ci95'][0])}, {pct(b['ci95'][1])}] | {pct(b['p_pos'])} |")
    a("")
    a("### CLV mean VALID_STRICT\n")
    a("| Cohort | Janela | N | Média | IC95 | P(CLV>0) |")
    a("|---|---|---:|---:|---|---:|")
    for k in ["ALL", "stake10", "stake2", "FRIENDLY", "NON_FRIENDLY"]:
        for w, _, _ in WINDOWS:
            b = clv_boots[k][w]
            if not b:
                continue
            a(
                f"| {k} | {w} | {metrics[k]['clv'][w]['n']} | {b['mean']:.2f}% | "
                f"[{b['ci95'][0]:.2f}%, {b['ci95'][1]:.2f}%] | {pct(b['p_pos'])} |"
            )
    a("")
    a("Para CLV All com N≥~40, os IC95 ficam **maioritariamente abaixo de zero** — CLV negativo é mais crível que o ROI negativo (cujo IC ainda cruza zero em vários cohorts).\n")

    a("## 3) CLV × ROI realizado (joint)\n")
    a("Só ordens **settled** com CLV VALID_STRICT.\n")
    for w, j in joints.items():
        corr = f"{j['corr_pearson']:.3f}" if j["corr_pearson"] is not None else "—"
        a(f"### {w} (N={j['n']}, corr={corr}, sign-concordance={pct(j['sign_concordance'])})\n")
        a("| Bucket CLV | N | CLV médio | ROI realizado | WR |")
        a("|---|---:|---:|---:|---:|")
        for b in j["bins"]:
            a(f"| {b['bin']} | {b['n']} | {fmt_clv(b['clv_mean'])} | {pct(b['roi'])} | {pct(b['wr'])} |")
        a("")
    a("Correlação CLV↔ROI é fraca/mista (N CLV limitado). A **massa de CLV < 0** continua o warning principal de mispricing.\n")

    a("## 4) Cobertura CLV\n")
    a("| Cohort | POST_5M | POST_15M | CLOSING |")
    a("|---|---:|---:|---:|")
    for label in ["ALL", "FRIENDLY", "NON_FRIENDLY", "stake10", "stake2"]:
        m = metrics[label]
        a(
            f"| {label} | {pct(m['clv']['POST_5M']['coverage'])} (n={m['clv']['POST_5M']['n']}) | "
            f"{pct(m['clv']['POST_15M']['coverage'])} (n={m['clv']['POST_15M']['n']}) | "
            f"{pct(m['clv']['CLOSING']['coverage'])} (n={m['clv']['CLOSING']['n']}) |"
        )
    a("")
    a("Friendly tem coverage ~2× menor que Non-Friendly em POST_5M — comparar médias F vs NF com cautela.\n")

    a("## 5) Evolução diária (ROI + CLV)\n")
    a("| Dia | N | F/NF | Era | P&L | ROI | CLV5 mean (n) | CLV Close mean (n) |")
    a("|---|---:|---:|---|---:|---:|---:|---:|")
    for d in daily:
        eras = ",".join(f"{k}:{v}" for k, v in sorted(d["stake_era"].items()))
        c5 = f"{d['clv5_mean']:.2f}% ({d['clv5_n']})" if d["clv5_mean"] is not None else f"— ({d['clv5_n']})"
        cc = f"{d['clvC_mean']:.2f}% ({d['clvC_n']})" if d["clvC_mean"] is not None else f"— ({d['clvC_n']})"
        a(
            f"| {d['day']} | {d['n']} | {d['friendly_n']}/{d['nf_n']} | {eras} | "
            f"{money(d['pnl'])} | {pct(d['roi'])} | {c5} | {cc} |"
        )
    a("")

    a("## 6) Cohorts cruzados stake × Friendly\n")
    a("| Cohort | N | ROI | CLV5 mean | CLV Close mean |")
    a("|---|---:|---:|---:|---:|")
    for k in ["stake10_FRIENDLY", "stake10_NON_FRIENDLY", "stake2_FRIENDLY", "stake2_NON_FRIENDLY"]:
        m = metrics[k]
        a(
            f"| {k} | {m['n']} | {pct(m['roi'])} | "
            f"{fmt_clv(m['clv']['POST_5M']['mean'])} | {fmt_clv(m['clv']['CLOSING']['mean'])} |"
        )
    a("")

    a("## 7) Conclusões (diagnóstico)\n")
    a("1. **CLV primeiro:** mercado move-se contra a posição após o fill. Sem CLV positivo sustentado, ROI positivo seria ruído.")
    a("2. **ROI melhorou vs 01/Ago** (−9.9% → −2.9%) com maturidade + stake2; **não** prova edge.")
    a("3. **Stake2** mitiga $ PnL, mas CLV também negativo — não resolve mispricing.")
    a("4. **Friendly filter** não se justifica: ROI F≈NF; CLV Friendly pior (coverage menor).")
    a("5. Prioridade: subir coverage CLV (esp. Friendly/POST_5M) e reavaliar com N VALID_STRICT ≥100/janela.\n")

    a("## 8) Artefactos\n")
    a(f"- Freeze: `logs/h3bup_friendly_analysis/20260807/{run}/`")
    a(f"- Bundle CLV+ROI: `logs/h3bup_strategy_results/20260807/clvroi_{run}/`\n")

    text = "\n".join(lines) + "\n"
    (out / f"h3bup_clv_roi_report_{run}.md").write_text(text, encoding="utf-8")
    (docs / "h3bup_clv_roi_analysis_20260807.md").write_text(text, encoding="utf-8")

    ci = roi_boots["ALL"]["ci95"]
    exec_sum = f"""# Executive — H3BUP CLV + ROI (2026-08-07)

- **Status:** `{status}`
- **Freeze:** `{run}` · N={len(rows)} (stake10={m10['n']}, stake2={m2['n']})
- **ROI resolved:** {pct(m_all['roi'])} (P&L {money(m_all['pnl'])}) · IC95 [{pct(ci[0])}, {pct(ci[1])}]
- **CLV POST_5M:** mean {fmt_clv(m_all['clv']['POST_5M']['mean'])} · pos {pct(m_all['clv']['POST_5M']['p_positive'])} · cov {pct(m_all['clv']['POST_5M']['coverage'])}
- **CLV CLOSING:** mean {fmt_clv(m_all['clv']['CLOSING']['mean'])} · pos {pct(m_all['clv']['CLOSING']['p_positive'])} · cov {pct(m_all['clv']['CLOSING']['coverage'])}
- **stake2 ROI:** {pct(m2['roi'])} | **stake10 ROI:** {pct(m10['roi'])}
- **Friendly vs NF ROI:** {pct(m_f['roi'])} vs {pct(m_nf['roi'])} (sem diferença clara)
- **Mensagem:** CLV negativo é o sinal dominante; ROI ligeiramente negativo e ainda inconclusivo no IC.

Read-only.
"""
    (out / f"h3bup_clv_roi_executive_{run}.md").write_text(exec_sum, encoding="utf-8")
    (docs / "h3bup_clv_roi_executive_20260807.md").write_text(exec_sum, encoding="utf-8")

    print("ALL roi", m_all["roi"], "pnl", m_all["pnl"])
    print("stake10", m10["roi"], "stake2", m2["roi"])
    print("CLV5", m_all["clv"]["POST_5M"])
    print("CLVC", m_all["clv"]["CLOSING"])
    print("WROTE", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
