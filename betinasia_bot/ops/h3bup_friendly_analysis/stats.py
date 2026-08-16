"""Clustered bootstrap / permutation tests (order_id unit, event_id cluster)."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .settlement import performance_block, sample_gate


def _roi(rows: Sequence[Dict[str, Any]]) -> Optional[float]:
    return performance_block(list(rows)).get("roi_resolved")


def _clv_mean(rows: Sequence[Dict[str, Any]], field: str, valid_f: str) -> Optional[float]:
    vals = []
    for r in rows:
        if r.get(valid_f) and r.get(field) is not None:
            try:
                vals.append(float(r[field]))
            except Exception:
                pass
    if not vals:
        return None
    return sum(vals) / len(vals)


def _median(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    xs = sorted(vals)
    n = len(xs)
    mid = n // 2
    if n % 2:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])


def _cluster_map(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    m: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        cid = str(r.get("event_id") or r.get("match_id") or r.get("order_id") or "")
        m[cid].append(r)
    return m


def clustered_bootstrap_diff(
    a: Sequence[Dict[str, Any]],
    b: Sequence[Dict[str, Any]],
    *,
    stat_fn,
    n_boot: int = 2000,
    seed: int = 47,
) -> Dict[str, Any]:
    """Bootstrap difference mean(stat(a)) - mean(stat(b)) clustered by event_id."""
    rng = random.Random(seed)
    sa = stat_fn(a)
    sb = stat_fn(b)
    if sa is None or sb is None:
        return {
            "estimate": None,
            "ci90": [None, None],
            "ci95": [None, None],
            "status": "INSUFFICIENT_N",
            "n_a": len(a),
            "n_b": len(b),
        }
    obs = sa - sb
    ca = list(_cluster_map(a).values())
    cb = list(_cluster_map(b).values())
    if len(ca) < 2 or len(cb) < 2:
        return {
            "estimate": obs,
            "ci90": [None, None],
            "ci95": [None, None],
            "status": "INSUFFICIENT_N",
            "n_a": len(a),
            "n_b": len(b),
            "n_events_a": len(ca),
            "n_events_b": len(cb),
        }
    diffs = []
    for _ in range(n_boot):
        ba = [r for c in (rng.choice(ca) for _ in range(len(ca))) for r in c]
        bb = [r for c in (rng.choice(cb) for _ in range(len(cb))) for r in c]
        xa = stat_fn(ba)
        xb = stat_fn(bb)
        if xa is None or xb is None:
            continue
        diffs.append(xa - xb)
    if len(diffs) < 50:
        return {
            "estimate": obs,
            "ci90": [None, None],
            "ci95": [None, None],
            "status": "INSUFFICIENT_N",
            "n_a": len(a),
            "n_b": len(b),
        }
    diffs.sort()

    def ci(alpha: float) -> List[Optional[float]]:
        lo = diffs[int(alpha / 2 * (len(diffs) - 1))]
        hi = diffs[int((1 - alpha / 2) * (len(diffs) - 1))]
        return [lo, hi]

    return {
        "estimate": obs,
        "ci90": ci(0.10),
        "ci95": ci(0.05),
        "n_boot_used": len(diffs),
        "n_a": len(a),
        "n_b": len(b),
        "n_events_a": len(ca),
        "n_events_b": len(cb),
        "status": sample_gate(min(len(a), len(b))),
    }


def clustered_permutation_pvalue(
    a: Sequence[Dict[str, Any]],
    b: Sequence[Dict[str, Any]],
    *,
    stat_fn,
    n_perm: int = 2000,
    seed: int = 47,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    sa = stat_fn(a)
    sb = stat_fn(b)
    if sa is None or sb is None:
        return {"p_value": None, "status": "INSUFFICIENT_N"}
    obs = abs(sa - sb)
    # cluster labels
    clusters = []
    labels = []
    for rows, lab in ((a, "A"), (b, "B")):
        for cid, rs in _cluster_map(rows).items():
            clusters.append(rs)
            labels.append(lab)
    if len(clusters) < 4:
        return {"p_value": None, "status": "INSUFFICIENT_N", "estimate_abs": obs}
    n_ge = 0
    n_ok = 0
    for _ in range(n_perm):
        labs = labels[:]
        rng.shuffle(labs)
        aa, bb = [], []
        for rs, lab in zip(clusters, labs):
            (aa if lab == "A" else bb).extend(rs)
        xa = stat_fn(aa)
        xb = stat_fn(bb)
        if xa is None or xb is None:
            continue
        n_ok += 1
        if abs(xa - xb) >= obs - 1e-15:
            n_ge += 1
    if n_ok < 50:
        return {"p_value": None, "status": "INSUFFICIENT_N", "estimate_abs": obs}
    # add-one smoothing
    p = (n_ge + 1) / (n_ok + 1)
    return {
        "p_value": p,
        "n_perm_used": n_ok,
        "estimate_abs": obs,
        "status": sample_gate(min(len(a), len(b))),
    }


def run_stat_tests(rows: Sequence[Dict[str, Any]], *, n_boot: int = 1000, n_perm: int = 1000) -> Dict[str, Any]:
    f = [r for r in rows if r.get("friendly_class") == "FRIENDLY"]
    nf = [r for r in rows if r.get("friendly_class") == "NON_FRIENDLY"]
    n_events = len({str(r.get("event_id") or "") for r in rows if r.get("event_id")})

    def slip_med(rs):
        vals = [float(r["slippage_pre_pct"]) for r in rs if r.get("slippage_pre_pct") is not None]
        return _median(vals)

    def lat_med(rs):
        vals = [float(r["pre_submit_ms"]) for r in rs if r.get("pre_submit_ms") is not None]
        return _median(vals)

    tests = {}
    for name, fn in (
        ("roi_resolved_diff_friendly_minus_non", _roi),
        ("clv_5m_diff", lambda rs: _clv_mean(rs, "clv_post_5m", "clv_post_5m_valid_strict")),
        ("clv_15m_diff", lambda rs: _clv_mean(rs, "clv_post_15m", "clv_post_15m_valid_strict")),
        ("clv_closing_diff", lambda rs: _clv_mean(rs, "clv_closing", "clv_closing_valid_strict")),
        ("slippage_median_diff", slip_med),
        ("pre_submit_median_diff", lat_med),
    ):
        boot = clustered_bootstrap_diff(f, nf, stat_fn=fn, n_boot=n_boot)
        perm = clustered_permutation_pvalue(f, nf, stat_fn=fn, n_perm=n_perm)
        tests[name] = {
            **boot,
            "permutation_p_value": perm.get("p_value"),
            "permutation_status": perm.get("status"),
            "unit": "order_id",
            "cluster": "event_id",
        }

    # exploratory OLS-like: friendly indicator effect on pnl (clustered SE via bootstrap of coef)
    yx = []
    for r in rows:
        if r.get("settlement_status") != "SETTLED_DECIDED" or r.get("pnl") is None:
            continue
        if r.get("friendly_class") not in {"FRIENDLY", "NON_FRIENDLY"}:
            continue
        yx.append(r)
    reg = {"status": "INSUFFICIENT_N", "n": len(yx)}
    if len(yx) >= 30:
        # simple difference-in-means already covered; report controls availability
        reg = {
            "model": "pnl ~ friendly + odd_at_decision + slippage_pre_pct + pre_submit_ms + capacity_final",
            "n": len(yx),
            "n_events": len({str(r.get("event_id") or "") for r in yx if r.get("event_id")}),
            "friendly_coef_proxy": (
                (_roi(f) if _roi(f) is not None else 0) - (_roi(nf) if _roi(nf) is not None else 0)
            ),
            "notes": [
                "Exploratory only; coefficient proxy = ROI_friendly - ROI_non_friendly",
                "Clustered bootstrap ICs reported in roi_resolved_diff_*",
                "Do not use p-value alone as approval",
            ],
            "status": sample_gate(len(yx)),
        }

    return {
        "n_friendly": len(f),
        "n_non_friendly": len(nf),
        "n_events_total": n_events,
        "tests": tests,
        "exploratory_regression": reg,
        "disclaimer": "p-values are exploratory; not operational approval gates.",
    }
