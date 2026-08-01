"""Performance tables, temporal, league, bookmaker, concentration, scenarios."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .settlement import performance_block, sample_gate

CLASSES = ("FRIENDLY", "NON_FRIENDLY", "UNCLASSIFIED", "CONFLICT")


def _parse_day(created: str) -> str:
    if not created:
        return ""
    return str(created)[:10]


def group_by_class(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    g: Dict[str, List[Dict[str, Any]]] = {c: [] for c in CLASSES}
    for r in rows:
        cls = str(r.get("friendly_class") or "UNCLASSIFIED")
        if cls not in g:
            cls = "UNCLASSIFIED"
        g[cls].append(r)
    return g


def performance_summary_table(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups = group_by_class(rows)
    blocks = {c: performance_block(groups[c]) for c in CLASSES}
    total = performance_block(list(rows))
    # reshape to metric x class
    metrics = [
        ("LIVE_OK", "n_live_ok"),
        ("eventos_unicos", "n_events"),
        ("stake_placed", "stake_placed"),
        ("open", "n_open"),
        ("settled_decided", "n_settled_decided"),
        ("void_push", "n_void_push"),
        ("missing", "n_missing"),
        ("stake_resolved", "stake_resolved_total"),
        ("pnl_resolved", "pnl_resolved"),
        ("roi_resolved", "roi_resolved"),
        ("roi_ex_void", "roi_decided_ex_void"),
        ("accounting_coverage", "accounting_coverage"),
        ("maturity", "maturity"),
        ("sample_gate", None),
    ]
    out = []
    for label, key in metrics:
        row = {"metric": label}
        for c in CLASSES:
            if key is None:
                row[c] = sample_gate(int(blocks[c].get("n_resolved") or 0))
            else:
                row[c] = blocks[c].get(key)
        if key is None:
            row["TOTAL"] = sample_gate(int(total.get("n_resolved") or 0))
        else:
            row["TOTAL"] = total.get(key)
        out.append(row)
    return out


def daily_performance(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_day: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        day = _parse_day(str(r.get("created_at_utc") or ""))
        cls = str(r.get("friendly_class") or "UNCLASSIFIED")
        by_day[day][cls].append(r)
    out = []
    for day in sorted(by_day.keys()):
        f = performance_block(by_day[day].get("FRIENDLY", []))
        nf = performance_block(by_day[day].get("NON_FRIENDLY", []))
        out.append(
            {
                "day_utc": day,
                "friendly_n": f["n_live_ok"],
                "friendly_pnl": f["pnl_resolved"],
                "friendly_roi": f["roi_resolved"],
                "non_friendly_n": nf["n_live_ok"],
                "non_friendly_pnl": nf["pnl_resolved"],
                "non_friendly_roi": nf["roi_resolved"],
            }
        )
    return out


def cumulative_series(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ordered = sorted(rows, key=lambda r: str(r.get("created_at_utc") or ""))
    groups = {"FRIENDLY": [], "NON_FRIENDLY": []}
    cum = {"FRIENDLY": [], "NON_FRIENDLY": []}
    for r in ordered:
        cls = str(r.get("friendly_class") or "")
        if cls not in groups:
            continue
        groups[cls].append(r)
        b = performance_block(groups[cls])
        cum[cls].append(
            {
                "created_at_utc": r.get("created_at_utc"),
                "n": b["n_live_ok"],
                "pnl_cum": b["pnl_resolved"],
                "roi_cum": b["roi_resolved"],
            }
        )

    def slice_perf(lst: List[Dict[str, Any]], start_frac: float, end_frac: float) -> Dict[str, Any]:
        n = len(lst)
        if n == 0:
            return performance_block([])
        a = int(n * start_frac)
        b = max(a + 1, int(n * end_frac))
        return performance_block(lst[a:b])

    stability = {}
    for cls, lst in groups.items():
        stability[cls] = {
            "first_30pct": slice_perf(lst, 0.0, 0.3),
            "middle_40pct": slice_perf(lst, 0.3, 0.7),
            "last_30pct": slice_perf(lst, 0.7, 1.0),
            "first_half": slice_perf(lst, 0.0, 0.5),
            "second_half": slice_perf(lst, 0.5, 1.0),
        }
    return {"cumulative": cum, "stability_slices": stability}


def _pct(xs: List[float], p: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    idx = int(round(p * (len(s) - 1)))
    return s[max(0, min(len(s) - 1, idx))]


def execution_summary(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups = group_by_class(rows)
    f = groups["FRIENDLY"]
    nf = groups["NON_FRIENDLY"]

    def col(lst: List[Dict[str, Any]], key: str) -> List[float]:
        out = []
        for r in lst:
            if r.get(key) is not None:
                try:
                    out.append(float(r[key]))
                except Exception:
                    pass
        return out

    metrics = [
        ("odd_mediana", "odd_at_decision", "median"),
        ("slippage_mediana", "slippage_pre_pct", "median"),
        ("pre_submit_p50", "pre_submit_ms", "p50"),
        ("pre_submit_p95", "pre_submit_ms", "p95"),
        ("place_p50", "place_duration_ms", "p50"),
        ("capacity_mediana", "capacity_final", "median"),
    ]
    out = []
    for label, key, how in metrics:
        fv = col(f, key)
        nv = col(nf, key)
        if how == "median":
            fval = median(fv) if fv else None
            nval = median(nv) if nv else None
        elif how == "p50":
            fval = _pct(fv, 0.50)
            nval = _pct(nv, 0.50)
        else:
            fval = _pct(fv, 0.95)
            nval = _pct(nv, 0.95)
        delta = None if fval is None or nval is None else fval - nval
        out.append({"metric": label, "FRIENDLY": fval, "NON_FRIENDLY": nval, "delta": delta})
    return out


def league_breakdown(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for cls, lst in group_by_class(rows).items():
        by_lg: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in lst:
            lg = str(r.get("league_name") or r.get("competition_name") or "(missing)")
            by_lg[lg].append(r)
        for lg, rs in by_lg.items():
            b = performance_block(rs)
            clv_vals = [
                float(r["clv_closing"])
                for r in rs
                if r.get("clv_closing_valid_strict") and r.get("clv_closing") is not None
            ]
            out.append(
                {
                    "class": cls,
                    "league": lg,
                    "n": b["n_live_ok"],
                    "stake": b["stake_placed"],
                    "pnl": b["pnl_resolved"],
                    "roi": b["roi_resolved"],
                    "clv_closing_median": (median(clv_vals) if clv_vals else None),
                    "clv_coverage": (len(clv_vals) / b["n_live_ok"]) if b["n_live_ok"] else None,
                }
            )
    out.sort(key=lambda r: (-abs(float(r["pnl"] or 0)), -int(r["n"] or 0)))
    return out


def bookmaker_breakdown(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for cls, lst in group_by_class(rows).items():
        by_bk: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in lst:
            by_bk[str(r.get("bookmaker") or "(missing)")].append(r)
        for bk, rs in by_bk.items():
            b = performance_block(rs)
            out.append(
                {
                    "class": cls,
                    "bookmaker": bk,
                    "n": b["n_live_ok"],
                    "stake": b["stake_placed"],
                    "pnl": b["pnl_resolved"],
                    "roi": b["roi_resolved"],
                }
            )
    out.sort(key=lambda r: (-int(r["n"] or 0),))
    return out


def concentration_report(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for cls, lst in group_by_class(rows).items():
        if not lst:
            continue
        total_stake = sum(float(r["stake"]) for r in lst if r.get("stake") is not None) or 0.0
        # by event
        by_ev: Dict[str, Dict[str, float]] = defaultdict(lambda: {"stake": 0.0, "pnl": 0.0, "n": 0})
        for r in lst:
            ev = str(r.get("event_id") or r.get("event_name") or "unknown")
            by_ev[ev]["n"] += 1
            if r.get("stake") is not None:
                by_ev[ev]["stake"] += float(r["stake"])
            if r.get("pnl") is not None:
                by_ev[ev]["pnl"] += float(r["pnl"])
        ev_sorted = sorted(by_ev.items(), key=lambda kv: -kv[1]["stake"])
        pnl_pos = sorted(by_ev.items(), key=lambda kv: -kv[1]["pnl"])
        pnl_neg = sorted(by_ev.items(), key=lambda kv: kv[1]["pnl"])

        def share_stake(kvs, k):
            s = sum(v["stake"] for _, v in kvs[:k])
            return (s / total_stake) if total_stake else None

        by_bk = Counter(str(r.get("bookmaker") or "(missing)") for r in lst)
        top_bk = by_bk.most_common(5)
        out.append(
            {
                "class": cls,
                "top1_event_stake_share": share_stake(ev_sorted, 1),
                "top3_event_stake_share": share_stake(ev_sorted, 3),
                "top5_event_stake_share": share_stake(ev_sorted, 5),
                "top1_event_id": ev_sorted[0][0] if ev_sorted else None,
                "top1_bookmaker": top_bk[0][0] if top_bk else None,
                "top1_bookmaker_share": (top_bk[0][1] / len(lst)) if top_bk and lst else None,
                "top3_bookmakers": ",".join(b for b, _ in top_bk[:3]),
                "top1_gain_event": pnl_pos[0][0] if pnl_pos else None,
                "top1_gain_pnl": pnl_pos[0][1]["pnl"] if pnl_pos else None,
                "top1_loss_event": pnl_neg[0][0] if pnl_neg else None,
                "top1_loss_pnl": pnl_neg[0][1]["pnl"] if pnl_neg else None,
            }
        )
    return out


def leave_one_league_out(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Result total / without top1 / top3 / top5 leagues by |pnl| within each class and overall."""
    out = []
    lg_rows = league_breakdown(rows)
    # overall by |pnl|
    overall_lg = defaultdict(lambda: {"pnl": 0.0, "n": 0, "stake": 0.0})
    for r in rows:
        lg = str(r.get("league_name") or r.get("competition_name") or "(missing)")
        if r.get("pnl") is not None:
            overall_lg[lg]["pnl"] += float(r["pnl"])
        overall_lg[lg]["n"] += 1
        if r.get("stake") is not None:
            overall_lg[lg]["stake"] += float(r["stake"])
    ranked = sorted(overall_lg.keys(), key=lambda k: -abs(overall_lg[k]["pnl"]))

    def perf_excluding(exclude: set) -> Dict[str, Any]:
        subset = [
            r
            for r in rows
            if str(r.get("league_name") or r.get("competition_name") or "(missing)") not in exclude
        ]
        return performance_block(subset)

    base = performance_block(list(rows))
    out.append({"scenario": "total", "excluded_leagues": "", **{k: base[k] for k in ("n_live_ok", "stake_placed", "pnl_resolved", "roi_resolved")}})
    for k, label in ((1, "without_top1_league"), (3, "without_top3_leagues"), (5, "without_top5_leagues")):
        excl = set(ranked[:k])
        b = perf_excluding(excl)
        out.append(
            {
                "scenario": label,
                "excluded_leagues": "|".join(ranked[:k]),
                "n_live_ok": b["n_live_ok"],
                "stake_placed": b["stake_placed"],
                "pnl_resolved": b["pnl_resolved"],
                "roi_resolved": b["roi_resolved"],
            }
        )
    return out


def counterfactual_scenarios(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Analytical only — not operational recommendations."""
    g = group_by_class(rows)
    scenarios = {
        "A_H3BUP_completa": list(rows),
        "B_apenas_Friendly": g["FRIENDLY"],
        "C_apenas_non_Friendly": g["NON_FRIENDLY"],
        "D_non_Friendly_plus_Unclassified": g["NON_FRIENDLY"] + g["UNCLASSIFIED"],
        "E_confirmed_classes_only": g["FRIENDLY"] + g["NON_FRIENDLY"],
    }
    out = []
    for name, lst in scenarios.items():
        b = performance_block(lst)
        clv = [
            float(r["clv_closing"])
            for r in lst
            if r.get("clv_closing_valid_strict") and r.get("clv_closing") is not None
        ]
        out.append(
            {
                "scenario": name,
                "n": b["n_live_ok"],
                "stake": b["stake_placed"],
                "pnl": b["pnl_resolved"],
                "roi": b["roi_resolved"],
                "open": b["n_open"],
                "maturity": b["maturity"],
                "clv_closing_median": (median(clv) if clv else None),
                "clv_coverage": (len(clv) / b["n_live_ok"]) if b["n_live_ok"] else None,
                "disclaimer": (
                    "Cenários históricos não representam resultado out-of-sample e não devem ser "
                    "interpretados como recomendação operacional."
                ),
            }
        )
    return out
