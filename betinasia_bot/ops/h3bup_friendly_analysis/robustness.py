"""Robustness / leave-one-out diagnostics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence

from .settlement import performance_block


def _pnl_of(r: Dict[str, Any]) -> float:
    return float(r["pnl"]) if r.get("pnl") is not None else 0.0


def robustness_suite(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    f0 = [r for r in rows if r.get("friendly_class") == "FRIENDLY"]
    nf0 = [r for r in rows if r.get("friendly_class") == "NON_FRIENDLY"]
    base_delta = None
    rf = performance_block(f0).get("roi_resolved")
    rnf = performance_block(nf0).get("roi_resolved")
    if rf is not None and rnf is not None:
        base_delta = rf - rnf

    out: List[Dict[str, Any]] = []

    def add(name: str, subset: List[Dict[str, Any]]) -> None:
        bf = performance_block([r for r in subset if r.get("friendly_class") == "FRIENDLY"])
        bnf = performance_block([r for r in subset if r.get("friendly_class") == "NON_FRIENDLY"])
        bt = performance_block(subset)
        delta = None
        if bf.get("roi_resolved") is not None and bnf.get("roi_resolved") is not None:
            delta = bf["roi_resolved"] - bnf["roi_resolved"]
        conclusion_same = None
        if base_delta is not None and delta is not None:
            conclusion_same = (base_delta >= 0) == (delta >= 0)
        out.append(
            {
                "scenario": name,
                "n": bt["n_live_ok"],
                "pnl_total": bt["pnl_resolved"],
                "roi_total": bt["roi_resolved"],
                "pnl_friendly": bf["pnl_resolved"],
                "roi_friendly": bf["roi_resolved"],
                "pnl_non_friendly": bnf["pnl_resolved"],
                "roi_non_friendly": bnf["roi_resolved"],
                "delta_roi_f_minus_nf": delta,
                "sign_vs_base_unchanged": conclusion_same,
            }
        )

    add("baseline", list(rows))

    settled = [r for r in rows if r.get("settlement_status") == "SETTLED_DECIDED"]
    by_gain = sorted(settled, key=_pnl_of, reverse=True)
    by_loss = sorted(settled, key=_pnl_of)

    def drop_ids(ids):
        return [r for r in rows if r.get("order_id") not in ids]

    for k, label in ((1, "remove_top1_gain"), (3, "remove_top3_gains"), (5, "remove_top5_gains")):
        ids = {r.get("order_id") for r in by_gain[:k]}
        add(label, drop_ids(ids))
    for k, label in ((1, "remove_top1_loss"), (3, "remove_top3_losses")):
        ids = {r.get("order_id") for r in by_loss[:k]}
        add(label, drop_ids(ids))

    by_lg: Dict[str, float] = defaultdict(float)
    for r in settled:
        lg = str(r.get("league_name") or r.get("competition_name") or "(missing)")
        by_lg[lg] += _pnl_of(r)
    ranked_lg = sorted(by_lg.keys(), key=lambda k: -abs(by_lg[k]))
    for k, label in ((1, "remove_top1_league"), (3, "remove_top3_leagues")):
        excl = set(ranked_lg[:k])
        subset = [
            r
            for r in rows
            if str(r.get("league_name") or r.get("competition_name") or "(missing)") not in excl
        ]
        add(label, subset)

    by_bk: Dict[str, int] = defaultdict(int)
    for r in rows:
        by_bk[str(r.get("bookmaker") or "(missing)")] += 1
    if by_bk:
        dom = max(by_bk.items(), key=lambda kv: kv[1])[0]
        add("remove_dominant_bookmaker", [r for r in rows if str(r.get("bookmaker") or "(missing)") != dom])

    add(
        "only_accounting_reconciled",
        [r for r in rows if r.get("settlement_status") in {"SETTLED_DECIDED", "VOID_PUSH"}],
    )
    add("only_clv_closing_valid_strict", [r for r in rows if r.get("clv_closing_valid_strict")])
    add("only_pre_submit_available", [r for r in rows if r.get("pre_submit_ms") is not None])
    add("only_valid_event_id", [r for r in rows if r.get("event_id")])
    return out
