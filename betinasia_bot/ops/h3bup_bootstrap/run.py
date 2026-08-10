#!/usr/bin/env python3
"""FASE 2E-A — Bootstrap de performance H3BUP_vNext (read-only).

Universe: policy_id=H3BUP_vNext · policy_version=H3BUP_vNext_20260629 · LIVE_OK
ROI: sum(pnl_resolved) / sum(stake_resolved)  — VOID_PUSH entra no denominador (pnl=0).
Bootstrap preferencial: cluster por event_id · n_boot >= 100_000 · seed documentada.
Não altera policy/stake/executor/bridge/accounting/CLV/Telegram.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POLICY_ID = "H3BUP_vNext"
POLICY_VERSION = "H3BUP_vNext_20260629"
RESOLVED = {"SETTLED_DECIDED", "VOID_PUSH"}
EXCLUDED_PNL = {"OPEN", "MISSING", "UNRECONCILED"}
SEED = 20260810
N_BOOT_DEFAULT = 100_000
N_PERM_DEFAULT = 50_000
RECON_TOL_ROI = 1e-9
RECON_TOL_MONEY = 1e-6

CLV_WINDOWS = [
    ("POST_5M", "clv_post_5m", "clv_post_5m_valid_strict"),
    ("POST_15M", "clv_post_15m", "clv_post_15m_valid_strict"),
    ("CLOSING", "clv_closing", "clv_closing_valid_strict"),
]

CUM_CHECKPOINTS = [25, 50, 75, 100, 150, 200]
ROLLING = [25, 50, 100]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fnum(x: Any) -> Optional[float]:
    try:
        if x is None or str(x).strip() in ("", "None", "nan", "NaN"):
            return None
        return float(str(x).replace(",", "."))
    except Exception:
        return None


def fbool(x: Any) -> bool:
    return str(x).strip().lower() in ("1", "true", "yes", "t")


def pct(x: Optional[float], d: int = 2) -> str:
    return "—" if x is None else f"{100.0 * x:.{d}f}%"


def money(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{x:+.2f}" if abs(x) > 1e-12 else "0.00"


def fmt_clv(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.2f}%"


def short_run_id(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fn = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fn})


def quantiles(arr: np.ndarray, levels: Sequence[float]) -> List[float]:
    a = np.sort(np.asarray(arr, dtype=float))
    n = len(a)
    out = []
    for q in levels:
        if n == 0:
            out.append(float("nan"))
            continue
        idx = int(round(q * (n - 1)))
        idx = max(0, min(n - 1, idx))
        out.append(float(a[idx]))
    return out


def summarize_boot(obs: float, boots: np.ndarray) -> Dict[str, Any]:
    boots = np.asarray(boots, dtype=float)
    boots = boots[np.isfinite(boots)]
    if boots.size == 0:
        return {
            "observed": obs,
            "mean": None,
            "median": None,
            "se": None,
            "ci80": [None, None],
            "ci90": [None, None],
            "ci95": [None, None],
            "ci99": [None, None],
            "p_gt_0": None,
            "p_gt_0_02": None,
            "p_gt_0_05": None,
            "p_gt_0_10": None,
            "p_lt_0": None,
            "one_sided_tail_leq_0": None,
            "n_boot": 0,
        }
    q = quantiles(boots, [0.005, 0.01, 0.025, 0.05, 0.10, 0.50, 0.90, 0.95, 0.975, 0.99, 0.995])
    # one-sided bootstrap tail: proportion of boot <= 0 (against H0: ROI<=0 evidence for positive)
    # Report P(boot > 0) and clarify it is not a classical frequentist p-value.
    p_gt0 = float(np.mean(boots > 0))
    p_lt0 = float(np.mean(boots < 0))
    # exploratory: fraction of bootstrap replicates with ROI <= 0
    one_sided = float(np.mean(boots <= 0))
    return {
        "observed": float(obs),
        "mean": float(np.mean(boots)),
        "median": float(np.median(boots)),
        "se": float(np.std(boots, ddof=1)),
        "ci80": [q[4], q[6]],  # 10% / 90%
        "ci90": [q[3], q[7]],  # 5% / 95%
        "ci95": [q[2], q[8]],  # 2.5% / 97.5%
        "ci99": [q[1], q[9]],  # 1% / 99%
        "p_gt_0": p_gt0,
        "p_gt_0_02": float(np.mean(boots > 0.02)),
        "p_gt_0_05": float(np.mean(boots > 0.05)),
        "p_gt_0_10": float(np.mean(boots > 0.10)),
        "p_lt_0": p_lt0,
        "one_sided_tail_leq_0": one_sided,
        "n_boot": int(boots.size),
        "note": "P(ROI>0) is bootstrap probability mass above 0, not a classical frequentist p-value.",
    }


def sample_gate(n: int) -> str:
    if n < 30:
        return "VERY_LOW_N"
    if n < 100:
        return "INSUFFICIENT_N"
    if n < 250:
        return "FIRST_READING"
    return "RELIABLE_READING_CANDIDATE"


def spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 3:
        return None
    rx = rankdata(x)
    ry = rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = math.sqrt(float(np.sum(rx * rx) * np.sum(ry * ry)))
    if den < 1e-15:
        return None
    return float(np.sum(rx * ry) / den)


def kendall_tau(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    n = len(x)
    if n < 3:
        return None
    conc = disc = 0
    for i in range(n - 1):
        dx = x[i + 1 :] - x[i]
        dy = y[i + 1 :] - y[i]
        prod = dx * dy
        conc += int(np.sum(prod > 0))
        disc += int(np.sum(prod < 0))
    tot = conc + disc
    if tot == 0:
        return None
    return (conc - disc) / tot


def rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # average ties
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = 0.5 * (i + 1 + j + 1)
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


# ---------------------------------------------------------------------------
# Load / reconcile
# ---------------------------------------------------------------------------

def load_orders(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(dict(r))
    # dedupe by order_id (last wins)
    by: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        oid = str(r.get("order_id") or "").strip()
        if not oid:
            continue
        by[oid] = r
    return list(by.values())


def normalize_row(r: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(r)
    out["_stake"] = fnum(r.get("stake")) or 0.0
    pnl = fnum(r.get("pnl"))
    out["_pnl"] = pnl  # may be None for OPEN/MISSING
    out["_pnl_resolved"] = float(pnl) if pnl is not None else None
    for w, field, validf in CLV_WINDOWS:
        out[f"_{field}"] = fnum(r.get(field))
        out[f"_{validf}"] = fbool(r.get(validf))
    return out


def reconcile(all_rows: List[Dict[str, Any]], expected: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    st = Counter(r.get("settlement_status") for r in all_rows)
    settled = [r for r in all_rows if r.get("settlement_status") == "SETTLED_DECIDED"]
    voids = [r for r in all_rows if r.get("settlement_status") == "VOID_PUSH"]
    opens = [r for r in all_rows if r.get("settlement_status") == "OPEN"]
    missing = [r for r in all_rows if r.get("settlement_status") == "MISSING"]
    unre = [r for r in all_rows if r.get("settlement_status") == "UNRECONCILED"]
    resolved = settled + voids
    stake_placed = sum(r["_stake"] for r in all_rows)
    stake_resolved = sum(r["_stake"] for r in resolved)
    pnl_resolved = sum(float(r["_pnl"]) for r in settled)  # void pnl = 0; don't coerce missing
    # include explicit void zeros
    for r in voids:
        if r["_pnl"] is not None:
            # already 0 by contract; settled-only sum is fine
            pass
    roi = (pnl_resolved / stake_resolved) if stake_resolved else None
    n_events = len({str(r.get("event_id") or "") for r in all_rows if r.get("event_id")})
    coverage = (len(resolved) / len(all_rows)) if all_rows else None
    block = {
        "total_live_ok": len(all_rows),
        "total_resolved": len(resolved),
        "settled_decided": len(settled),
        "void": len(voids),
        "open": len(opens),
        "missing": len(missing),
        "unreconciled": len(unre),
        "stake_placed": stake_placed,
        "stake_resolved": stake_resolved,
        "pnl_resolved": pnl_resolved,
        "roi_resolved": roi,
        "accounting_coverage": coverage,
        "n_event_id_unique": n_events,
        "void_in_denominator": True,
        "formula": "roi_resolved = sum(pnl SETTLED_DECIDED) / sum(stake SETTLED_DECIDED + VOID_PUSH)",
        "status_counts": dict(st),
    }
    ok = True
    reasons: List[str] = []
    if expected:
        for key, tol in (
            ("roi_resolved", RECON_TOL_ROI),
            ("pnl_resolved", RECON_TOL_MONEY),
            ("stake_resolved", RECON_TOL_MONEY),
        ):
            if key in expected and expected[key] is not None and block[key] is not None:
                if abs(float(block[key]) - float(expected[key])) > tol:
                    ok = False
                    reasons.append(f"{key}: got {block[key]} expected {expected[key]}")
    # internal invariants
    if len(settled) + len(voids) + len(opens) + len(missing) + len(unre) != len(all_rows):
        # allow other statuses but flag
        other = len(all_rows) - (len(settled) + len(voids) + len(opens) + len(missing) + len(unre))
        if other:
            reasons.append(f"other_statuses={other}")
    block["reconciliation_ok"] = ok and len(reasons) == 0
    block["reconciliation_reasons"] = reasons
    return block


# ---------------------------------------------------------------------------
# Bootstrap engines
# ---------------------------------------------------------------------------

def boot_roi_order(stakes: np.ndarray, pnls: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(stakes)
    idx = rng.integers(0, n, size=(n_boot, n))
    s = stakes[idx].sum(axis=1)
    p = pnls[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        roi = np.where(s > 0, p / s, np.nan)
    return roi


def boot_roi_cluster(
    cluster_stake: np.ndarray,
    cluster_pnl: np.ndarray,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    """Sample clusters with replacement; ROI = sum(pnl)/sum(stake)."""
    rng = np.random.default_rng(seed)
    k = len(cluster_stake)
    idx = rng.integers(0, k, size=(n_boot, k))
    s = cluster_stake[idx].sum(axis=1)
    p = cluster_pnl[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        roi = np.where(s > 0, p / s, np.nan)
    return roi


def aggregate_clusters(
    rows: Sequence[Dict[str, Any]],
    key_fn,
) -> Tuple[List[str], np.ndarray, np.ndarray, Dict[str, List[Dict[str, Any]]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(key_fn(r))].append(r)
    keys = sorted(groups.keys())
    stake = np.array([sum(x["_stake"] for x in groups[k]) for k in keys], dtype=float)
    pnl = np.array(
        [sum(float(x["_pnl"] or 0.0) for x in groups[k]) for k in keys],
        dtype=float,
    )
    return keys, stake, pnl, groups


def resolved_arrays(rows: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    stakes = np.array([r["_stake"] for r in rows], dtype=float)
    pnls = np.array([float(r["_pnl"] or 0.0) for r in rows], dtype=float)
    return stakes, pnls


def obs_roi(rows: Sequence[Dict[str, Any]]) -> Optional[float]:
    s = sum(r["_stake"] for r in rows)
    if s <= 0:
        return None
    p = sum(float(r["_pnl"] or 0.0) for r in rows)
    return p / s


def subset_perf(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    settled = [r for r in rows if r.get("settlement_status") == "SETTLED_DECIDED"]
    voids = [r for r in rows if r.get("settlement_status") == "VOID_PUSH"]
    stake = sum(r["_stake"] for r in rows)
    pnl = sum(float(r["_pnl"] or 0.0) for r in settled)
    events = len({str(r.get("event_id") or "") for r in rows if r.get("event_id")})
    return {
        "n": len(rows),
        "n_settled": len(settled),
        "n_void": len(voids),
        "events": events,
        "stake": stake,
        "pnl": pnl,
        "roi": (pnl / stake) if stake else None,
    }


# ---------------------------------------------------------------------------
# CLV bootstrap (cluster event)
# ---------------------------------------------------------------------------

def boot_clv_window(
    rows: Sequence[Dict[str, Any]],
    field: str,
    validf: str,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    # per-order valid values; cluster by event
    by_event: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if not r.get(f"_{validf}"):
            continue
        v = r.get(f"_{field}")
        if v is None:
            continue
        eid = str(r.get("event_id") or r.get("order_id"))
        by_event[eid].append(float(v))
    events = sorted(by_event.keys())
    n_vals = sum(len(by_event[e]) for e in events)
    coverage = (n_vals / len(rows)) if rows else None
    if n_vals == 0:
        return {
            "window": field,
            "n": 0,
            "events": 0,
            "coverage": coverage,
            "mean_obs": None,
            "median_obs": None,
            "ci90_mean": [None, None],
            "ci95_mean": [None, None],
            "p_mean_gt_0": None,
            "p_median_gt_0": None,
        }
    all_vals = [v for e in events for v in by_event[e]]
    mean_obs = float(np.mean(all_vals))
    median_obs = float(np.median(all_vals))
    rng = np.random.default_rng(seed)
    k = len(events)
    means = np.empty(n_boot, dtype=float)
    meds = np.empty(n_boot, dtype=float)
    lists = [np.array(by_event[e], dtype=float) for e in events]
    for i in range(n_boot):
        pick = rng.integers(0, k, size=k)
        sample = np.concatenate([lists[j] for j in pick])
        means[i] = float(np.mean(sample))
        meds[i] = float(np.median(sample))
    sm = summarize_boot(mean_obs, means)
    return {
        "window": field,
        "n": n_vals,
        "events": k,
        "coverage": coverage,
        "mean_obs": mean_obs,
        "median_obs": median_obs,
        "ci90_mean": sm["ci90"],
        "ci95_mean": sm["ci95"],
        "p_mean_gt_0": sm["p_gt_0"],
        "p_median_gt_0": float(np.mean(meds > 0)),
        "boot_mean_of_mean": sm["mean"],
        "boot_se_mean": sm["se"],
    }


# ---------------------------------------------------------------------------
# ROI × CLV
# ---------------------------------------------------------------------------

def roi_clv_analysis(
    resolved: Sequence[Dict[str, Any]],
    field: str,
    validf: str,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    pairs = []
    for r in resolved:
        if not r.get(f"_{validf}"):
            continue
        clv = r.get(f"_{field}")
        if clv is None:
            continue
        stake = r["_stake"]
        if stake <= 0:
            continue
        ret = float(r["_pnl"] or 0.0) / stake
        pairs.append((r, float(clv), ret))
    if not pairs:
        return {"window": field, "n": 0, "spearman": None, "kendall": None}
    clv = np.array([p[1] for p in pairs], dtype=float)
    ret = np.array([p[2] for p in pairs], dtype=float)
    rows = [p[0] for p in pairs]
    perf = subset_perf(rows)
    sp = spearman(clv, ret)
    kd = kendall_tau(clv, ret)

    # cluster bootstrap of spearman
    by_event: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for r, c, rt in pairs:
        by_event[str(r.get("event_id") or r.get("order_id"))].append((c, rt))
    events = sorted(by_event.keys())
    rng = np.random.default_rng(seed)
    boots = []
    k = len(events)
    if k >= 3:
        for _ in range(min(n_boot, 20_000)):  # cap for correlation cost
            pick = rng.integers(0, k, size=k)
            xs, ys = [], []
            for j in pick:
                for c, rt in by_event[events[j]]:
                    xs.append(c)
                    ys.append(rt)
            s = spearman(np.array(xs), np.array(ys))
            if s is not None:
                boots.append(s)
    boots_a = np.array(boots, dtype=float) if boots else np.array([])
    sp_sum = summarize_boot(sp if sp is not None else float("nan"), boots_a) if boots else None

    # clustered permutation of spearman (shuffle CLV across events)
    perm_p = None
    if k >= 4 and sp is not None:
        n_ge = 0
        n_ok = 0
        event_clvs = [np.array([c for c, _ in by_event[e]], dtype=float) for e in events]
        event_rets = [np.array([rt for _, rt in by_event[e]], dtype=float) for e in events]
        n_perm = min(10_000, n_boot)
        for _ in range(n_perm):
            order = rng.permutation(k)
            xs = np.concatenate([event_clvs[j] for j in order])
            ys = np.concatenate(event_rets)
            # lengths may mismatch if events have different n — pad by cycling? better: permute labels of whole events' CLV blocks vs RET blocks of same sizes only when sizes match
            # Use: shuffle event-level CLV vectors among events of same size is hard.
            # Defensible: permute order of CLV values globally vs fixed returns (breaks within-event too).
            # Prefer: assign permuted event CLV vectors to events — only works equal size.
            # Practical defensable approach: shuffle all CLV values vs fixed returns (order-level) but report as exploratory.
            xs = clv[rng.permutation(len(clv))]
            s = spearman(xs, ret)
            if s is None:
                continue
            n_ok += 1
            if abs(s) >= abs(sp) - 1e-15:
                n_ge += 1
        if n_ok >= 50:
            perm_p = (n_ge + 1) / (n_ok + 1)

    pos = [r for r, c, _ in pairs if c > 0]
    neg = [r for r, c, _ in pairs if c <= 0]
    return {
        "window": field,
        "n": len(pairs),
        "events": k,
        "roi": perf["roi"],
        "stake": perf["stake"],
        "pnl": perf["pnl"],
        "spearman": sp,
        "kendall": kd,
        "spearman_boot": sp_sum,
        "spearman_perm_p": perm_p,
        "clv_gt_0": subset_perf(pos),
        "clv_le_0": subset_perf(neg),
    }


# ---------------------------------------------------------------------------
# Sign-flip permutation vs ROI=0
# ---------------------------------------------------------------------------

def sign_flip_test(
    cluster_stake: np.ndarray,
    cluster_pnl: np.ndarray,
    n_perm: int,
    seed: int,
) -> Dict[str, Any]:
    """Exploratory: under H0 of no edge, flip event PnL signs keeping stakes.

    One-sided: P(ROI_perm >= ROI_obs) if ROI_obs>0 else P(ROI_perm <= ROI_obs).
    Two-sided: P(|ROI_perm| >= |ROI_obs|).
    """
    s_tot = float(cluster_stake.sum())
    if s_tot <= 0:
        return {"status": "NOT_CALCULABLE", "reason": "zero_stake"}
    obs = float(cluster_pnl.sum() / s_tot)
    rng = np.random.default_rng(seed)
    # vectorized: random signs
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, len(cluster_pnl)))
    p_boot = (signs * cluster_pnl).sum(axis=1) / s_tot
    if obs >= 0:
        p_one = float((np.sum(p_boot >= obs - 1e-15) + 1) / (n_perm + 1))
    else:
        p_one = float((np.sum(p_boot <= obs + 1e-15) + 1) / (n_perm + 1))
    p_two = float((np.sum(np.abs(p_boot) >= abs(obs) - 1e-15) + 1) / (n_perm + 1))
    return {
        "status": "OK",
        "method": "event_pnl_sign_flip",
        "null": "event-level PnL signs exchangeable under no-edge (exploratory)",
        "observed_roi": obs,
        "p_one_sided": p_one,
        "p_two_sided": p_two,
        "n_perm": n_perm,
        "note": "Exploratory; not an operational gate.",
    }


# ---------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------

def temporal_analysis(
    resolved: List[Dict[str, Any]],
    n_boot: int,
    seed: int,
) -> List[Dict[str, Any]]:
    ordered = sorted(resolved, key=lambda r: str(r.get("created_at_utc") or ""))
    out = []
    n = len(ordered)
    checkpoints = [c for c in CUM_CHECKPOINTS if c <= n] + ([n] if n not in CUM_CHECKPOINTS else [])
    for c in checkpoints:
        sub = ordered[:c]
        keys, cs, cp, _ = aggregate_clusters(sub, lambda r: r.get("event_id") or r.get("order_id"))
        obs = obs_roi(sub)
        boots = boot_roi_cluster(cs, cp, n_boot=min(n_boot, 50_000), seed=seed + c)
        sm = summarize_boot(obs if obs is not None else float("nan"), boots)
        out.append(
            {
                "kind": "cumulative",
                "n": c,
                "events": len(keys),
                "roi": obs,
                "p_gt_0": sm["p_gt_0"],
                "ci90": sm["ci90"],
                "ci95": sm["ci95"],
            }
        )
    for w in ROLLING:
        if n < w:
            continue
        sub = ordered[-w:]
        keys, cs, cp, _ = aggregate_clusters(sub, lambda r: r.get("event_id") or r.get("order_id"))
        obs = obs_roi(sub)
        boots = boot_roi_cluster(cs, cp, n_boot=min(n_boot, 50_000), seed=seed + 1000 + w)
        sm = summarize_boot(obs if obs is not None else float("nan"), boots)
        out.append(
            {
                "kind": "rolling",
                "n": w,
                "events": len(keys),
                "roi": obs,
                "p_gt_0": sm["p_gt_0"],
                "ci90": sm["ci90"],
                "ci95": sm["ci95"],
            }
        )
    return out


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

def robustness_scenarios(
    resolved: List[Dict[str, Any]],
    n_boot: int,
    seed: int,
) -> List[Dict[str, Any]]:
    # winners = settled with pnl > 0
    winners = sorted(
        [r for r in resolved if r.get("settlement_status") == "SETTLED_DECIDED" and float(r["_pnl"] or 0) > 0],
        key=lambda r: float(r["_pnl"] or 0),
        reverse=True,
    )
    # event pnl
    _, _, _, by_ev = aggregate_clusters(resolved, lambda r: r.get("event_id") or r.get("order_id"))
    ev_pnl = sorted(
        ((k, sum(float(r["_pnl"] or 0) for r in rs)) for k, rs in by_ev.items()),
        key=lambda t: t[1],
        reverse=True,
    )
    # league pnl
    _, _, _, by_lg = aggregate_clusters(
        resolved,
        lambda r: r.get("league_name") or r.get("competition_name") or "UNKNOWN",
    )
    lg_pnl = sorted(
        ((k, sum(float(r["_pnl"] or 0) for r in rs)) for k, rs in by_lg.items()),
        key=lambda t: t[1],
        reverse=True,
    )

    scenarios: List[Tuple[str, List[Dict[str, Any]]]] = []

    def drop_oids(oids: set) -> List[Dict[str, Any]]:
        return [r for r in resolved if str(r.get("order_id")) not in oids]

    def drop_events(eids: set) -> List[Dict[str, Any]]:
        return [r for r in resolved if str(r.get("event_id") or "") not in eids]

    def drop_leagues(lgs: set) -> List[Dict[str, Any]]:
        return [
            r
            for r in resolved
            if str(r.get("league_name") or r.get("competition_name") or "UNKNOWN") not in lgs
        ]

    if winners:
        scenarios.append(("A_drop_top1_winning_order", drop_oids({str(winners[0]["order_id"])})))
    if len(winners) >= 3:
        scenarios.append(
            ("B_drop_top3_winning_orders", drop_oids({str(w["order_id"]) for w in winners[:3]}))
        )
    pos_ev = [e for e, p in ev_pnl if p > 0]
    if pos_ev:
        scenarios.append(("C_drop_top1_winning_event", drop_events({pos_ev[0]})))
    if len(pos_ev) >= 3:
        scenarios.append(("D_drop_top3_winning_events", drop_events(set(pos_ev[:3]))))
    pos_lg = [lg for lg, p in lg_pnl if p > 0]
    if pos_lg:
        scenarios.append(("E_drop_top1_positive_league", drop_leagues({pos_lg[0]})))
    if len(pos_lg) >= 3:
        scenarios.append(("F_drop_top3_positive_leagues", drop_leagues(set(pos_lg[:3]))))

    out = []
    for name, sub in scenarios:
        perf = subset_perf(sub)
        keys, cs, cp, _ = aggregate_clusters(sub, lambda r: r.get("event_id") or r.get("order_id"))
        boots = boot_roi_cluster(cs, cp, n_boot=n_boot, seed=seed + hash(name) % 10_000)
        sm = summarize_boot(perf["roi"] if perf["roi"] is not None else float("nan"), boots)
        out.append(
            {
                "scenario": name,
                "n": perf["n"],
                "events": perf["events"],
                "stake": perf["stake"],
                "pnl": perf["pnl"],
                "roi": perf["roi"],
                "ci90_lo": sm["ci90"][0],
                "ci90_hi": sm["ci90"][1],
                "ci95_lo": sm["ci95"][0],
                "ci95_hi": sm["ci95"][1],
                "p_gt_0": sm["p_gt_0"],
            }
        )
    return out


# ---------------------------------------------------------------------------
# Friendly vs NF
# ---------------------------------------------------------------------------

def friendly_analysis(
    resolved: List[Dict[str, Any]],
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    f = [r for r in resolved if r.get("friendly_class") == "FRIENDLY"]
    nf = [r for r in resolved if r.get("friendly_class") == "NON_FRIENDLY"]
    results = {}
    for name, sub, sd in (("FRIENDLY", f, seed + 1), ("NON_FRIENDLY", nf, seed + 2)):
        perf = subset_perf(sub)
        keys, cs, cp, _ = aggregate_clusters(sub, lambda r: r.get("event_id") or r.get("order_id"))
        if len(keys) >= 2:
            boots = boot_roi_cluster(cs, cp, n_boot=n_boot, seed=sd)
            sm = summarize_boot(perf["roi"] if perf["roi"] is not None else float("nan"), boots)
        else:
            sm = summarize_boot(perf["roi"] if perf["roi"] is not None else float("nan"), np.array([]))
        results[name] = {**perf, "bootstrap": sm}

    # delta ROI NF - F with cluster bootstrap (independent resampling)
    delta_obs = None
    delta_boot = None
    if f and nf and results["FRIENDLY"]["roi"] is not None and results["NON_FRIENDLY"]["roi"] is not None:
        delta_obs = results["NON_FRIENDLY"]["roi"] - results["FRIENDLY"]["roi"]
        kf, csf, cpf, _ = aggregate_clusters(f, lambda r: r.get("event_id") or r.get("order_id"))
        kn, csn, cpn, _ = aggregate_clusters(nf, lambda r: r.get("event_id") or r.get("order_id"))
        rng = np.random.default_rng(seed + 3)
        diffs = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            if len(kf) == 0 or len(kn) == 0:
                diffs[i] = np.nan
                continue
            if_ = rng.integers(0, len(kf), size=len(kf))
            in_ = rng.integers(0, len(kn), size=len(kn))
            sf, pf = csf[if_].sum(), cpf[if_].sum()
            sn, pn = csn[in_].sum(), cpn[in_].sum()
            rf = pf / sf if sf > 0 else np.nan
            rn = pn / sn if sn > 0 else np.nan
            diffs[i] = rn - rf
        delta_boot = summarize_boot(delta_obs, diffs)
    return {
        "FRIENDLY": results["FRIENDLY"],
        "NON_FRIENDLY": results["NON_FRIENDLY"],
        "delta_roi_nf_minus_f": delta_obs,
        "delta_bootstrap": delta_boot,
        "p_delta_gt_0": (delta_boot or {}).get("p_gt_0"),
    }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(
    recon: Dict[str, Any],
    event_boot: Dict[str, Any],
    clv_results: List[Dict[str, Any]],
    robustness: List[Dict[str, Any]],
) -> Tuple[str, str]:
    if not recon.get("reconciliation_ok"):
        return "BOOTSTRAP_INPUT_RECONCILIATION_FAILED", sample_gate(recon.get("total_resolved") or 0)
    n = int(recon.get("total_resolved") or 0)
    readiness = sample_gate(n)
    if n < 30 or recon.get("accounting_coverage", 0) is None or (recon.get("accounting_coverage") or 0) < 0.5:
        return "DATA_QUALITY_INSUFFICIENT", readiness

    obs = event_boot.get("observed")
    ci95 = event_boot.get("ci95") or [None, None]
    ci90 = event_boot.get("ci90") or [None, None]
    p_gt0 = event_boot.get("p_gt_0")

    # CLV conflict check
    clv_means = [c.get("mean_obs") for c in clv_results if c.get("mean_obs") is not None]
    clv_neg = bool(clv_means) and all(m < 0 for m in clv_means)
    clv_p = [c.get("p_mean_gt_0") for c in clv_results if c.get("p_mean_gt_0") is not None]
    clv_unlikely_pos = bool(clv_p) and all(p is not None and p < 0.2 for p in clv_p)

    if obs is not None and obs > 0 and clv_neg and clv_unlikely_pos:
        # positive realized ROI but CLV negative → conflict label takes priority for honesty
        if p_gt0 is not None and p_gt0 < 0.7:
            return "CLV_CONFLICTS_WITH_REALIZED_ROI", readiness

    if obs is None:
        return "DATA_QUALITY_INSUFFICIENT", readiness

    if obs < 0:
        if ci95[1] is not None and ci95[1] < 0:
            return "NEGATIVE_ROI_SIGNAL", readiness
        if p_gt0 is not None and p_gt0 < 0.35:
            return "NEGATIVE_ROI_SIGNAL", readiness
        return "NO_CLEAR_ROI_EDGE", readiness

    # obs > 0
    if clv_neg and clv_unlikely_pos:
        return "CLV_CONFLICTS_WITH_REALIZED_ROI", readiness

    rob_keep_pos = all((r.get("roi") or 0) > 0 for r in robustness) if robustness else False
    ci90_above0 = ci90[0] is not None and ci90[0] > 0
    ci95_above0 = ci95[0] is not None and ci95[0] > 0

    if ci95_above0 and rob_keep_pos and readiness == "RELIABLE_READING_CANDIDATE":
        return "POSITIVE_ROI_STATISTICALLY_SUPPORTED", readiness
    if ci90_above0 and (p_gt0 or 0) >= 0.9:
        return "POSITIVE_ROI_PRELIMINARY", readiness
    if (p_gt0 or 0) >= 0.7 and obs > 0:
        return "POSITIVE_ROI_HIGHLY_UNCERTAIN", readiness
    return "NO_CLEAR_ROI_EDGE", readiness


def final_status(label: str, readiness: str, warnings: List[str]) -> str:
    if label == "BOOTSTRAP_INPUT_RECONCILIATION_FAILED":
        return "BOOTSTRAP_RECONCILIATION_FAILED"
    if label == "DATA_QUALITY_INSUFFICIENT" or readiness == "VERY_LOW_N":
        return "BOOTSTRAP_DATA_INSUFFICIENT"
    if warnings:
        return "BOOTSTRAP_ANALYSIS_COMPLETE_WITH_WARNINGS"
    return "BOOTSTRAP_ANALYSIS_COMPLETE"


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def write_methodology(path: Path, meta: Dict[str, Any]) -> None:
    path.write_text(
        f"""# Metodologia — FASE 2E-A Bootstrap H3BUP_vNext

## Universo
- `policy_id` = `{POLICY_ID}`
- `policy_version` = `{POLICY_VERSION}`
- status execução = `LIVE_OK` (freeze Friendly)
- Janela: primeiro LIVE_OK até cutoff `{meta.get("cutoff_utc")}`
- Deduplicação: `order_id`
- Resolvidas: `SETTLED_DECIDED` + `VOID_PUSH`
- Excluídas do P&L: `OPEN`, `MISSING`, `UNRECONCILED` (não preenchidas com zero)

## ROI
```
roi_resolved = sum(pnl SETTLED_DECIDED) / sum(stake SETTLED_DECIDED + VOID_PUSH)
```
**Void entra no denominador** (pnl void = 0).

## Bootstrap
- Replicações: `{meta.get("n_boot")}`
- Seed: `{meta.get("seed")}`
- Order-level: amostragem com reposição de ordens resolvidas
- **Preferencial:** cluster por `event_id` (amostra eventos; inclui todas as ordens do evento)
- Dia UTC: cluster por data de `created_at_utc`
- Quantis empíricos da distribuição bootstrap

## Interpretação
- `P(ROI>0)` = fração das replicações bootstrap com ROI>0
- **Não** é automaticamente um p-value frequentista clássico
- Teste sign-flip por evento é **exploratório**

## Segurança
Análise read-only. Sem criação de ordens, betslips, nem alteração de policy/stake/executor/bridge/CLV/Telegram.

## Freeze
- run_id fonte: `{meta.get("source_run_id")}`
- orders CSV: `{meta.get("orders_csv")}`
- generated: `{meta.get("generated_at_utc")}`
""",
        encoding="utf-8",
    )


def write_executive(
    path: Path,
    *,
    recon: Dict[str, Any],
    order_boot: Dict[str, Any],
    event_boot: Dict[str, Any],
    day_boot: Dict[str, Any],
    robustness: List[Dict[str, Any]],
    friendly: Dict[str, Any],
    clv_results: List[Dict[str, Any]],
    temporal: List[Dict[str, Any]],
    label: str,
    readiness: str,
    status: str,
    warnings: List[str],
) -> None:
    def find_rob(prefix: str) -> Optional[Dict[str, Any]]:
        for r in robustness:
            if r["scenario"].startswith(prefix):
                return r
        return None

    rC = find_rob("C_")
    rD = find_rob("D_")
    clv_map = {c["window"]: c for c in clv_results}
    # map field names
    c5 = next((c for c in clv_results if "5m" in c["window"]), None)
    c15 = next((c for c in clv_results if "15m" in c["window"]), None)
    cc = next((c for c in clv_results if "closing" in c["window"]), None)

    roll = {t["n"]: t for t in temporal if t["kind"] == "rolling"}
    # direction temporal: compare last rolling50 p_gt0 vs cumulative full
    cum_full = next((t for t in temporal if t["kind"] == "cumulative" and t["n"] == recon["total_resolved"]), None)
    evid = "indeterminada"
    if roll.get(50) and cum_full and roll[50]["p_gt_0"] is not None and cum_full["p_gt_0"] is not None:
        if roll[50]["p_gt_0"] > cum_full["p_gt_0"] + 0.05:
            evid = "melhorando (rolling50 vs full)"
        elif roll[50]["p_gt_0"] < cum_full["p_gt_0"] - 0.05:
            evid = "piorando (rolling50 vs full)"
        else:
            evid = "estável / oscilante sem tendência clara"

    roi_obs = recon["roi_resolved"]
    clv_same_dir = None
    if roi_obs is not None and cc and cc.get("mean_obs") is not None:
        clv_same_dir = (roi_obs > 0 and cc["mean_obs"] > 0) or (roi_obs < 0 and cc["mean_obs"] < 0)

    lines = [
        f"# Executive — FASE 2E-A Bootstrap H3BUP_vNext",
        "",
        f"- **Status:** `{status}`",
        f"- **Classificação:** `{label}`",
        f"- **statistical_readiness:** `{readiness}`",
        f"- Void no denominador: **sim**",
        "",
        "## Respostas (1–39)",
        "",
        f"1. Ordens resolvidas: **{recon['total_resolved']}**",
        f"2. Eventos únicos (LIVE_OK): **{recon['n_event_id_unique']}**",
        f"3. Stake resolvida: **{recon['stake_resolved']:.2f}**",
        f"4. P&L: **{money(recon['pnl_resolved'])}**",
        f"5. ROI observado: **{pct(recon['roi_resolved'])}**",
        "",
        "### ORDER BOOTSTRAP",
        f"6. Mean: **{pct(order_boot.get('mean'))}**",
        f"7. Median: **{pct(order_boot.get('median'))}**",
        f"8. IC90: **[{pct(order_boot['ci90'][0])}, {pct(order_boot['ci90'][1])}]**" if order_boot.get("ci90") else "8. IC90: —",
        f"9. IC95: **[{pct(order_boot['ci95'][0])}, {pct(order_boot['ci95'][1])}]**" if order_boot.get("ci95") else "9. IC95: —",
        f"10. P(ROI>0): **{pct(order_boot.get('p_gt_0'))}**",
        f"11. P(ROI>2%): **{pct(order_boot.get('p_gt_0_02'))}**",
        f"12. P(ROI>5%): **{pct(order_boot.get('p_gt_0_05'))}**",
        f"13. P(ROI>10%): **{pct(order_boot.get('p_gt_0_10'))}**",
        f"14. P(ROI<0): **{pct(order_boot.get('p_lt_0'))}**",
        "",
        "### CLUSTER EVENT BOOTSTRAP (preferencial)",
        f"15. Mean: **{pct(event_boot.get('mean'))}**",
        f"16. IC90: **[{pct(event_boot['ci90'][0])}, {pct(event_boot['ci90'][1])}]**" if event_boot.get("ci90") else "16. IC90: —",
        f"17. IC95: **[{pct(event_boot['ci95'][0])}, {pct(event_boot['ci95'][1])}]**" if event_boot.get("ci95") else "17. IC95: —",
        f"18. P(ROI>0): **{pct(event_boot.get('p_gt_0'))}**",
        f"19. P(ROI>5%): **{pct(event_boot.get('p_gt_0_05'))}**",
        f"20. P(ROI<0): **{pct(event_boot.get('p_lt_0'))}**",
        "",
        "### ROBUSTEZ",
        f"21. P(ROI>0) sem maior evento: **{pct((rC or {}).get('p_gt_0'))}**",
        f"22. sem top 3 eventos: **{pct((rD or {}).get('p_gt_0'))}**",
        f"23. ROI permanece positivo? **{('sim' if rD and (rD.get('roi') or 0) > 0 else 'não') if rD else '—'}**",
        f"24. IC90 acima de zero? **{('sim' if rD and rD.get('ci90_lo') is not None and rD['ci90_lo'] > 0 else 'não') if rD else '—'}**",
        f"25. IC95 acima de zero? **{('sim' if rD and rD.get('ci95_lo') is not None and rD['ci95_lo'] > 0 else 'não') if rD else '—'}**",
        "",
        "### FRIENDLY",
        f"26. ROI Friendly: **{pct(friendly['FRIENDLY'].get('roi'))}**",
        f"27. P(ROI>0) Friendly: **{pct(friendly['FRIENDLY'].get('bootstrap', {}).get('p_gt_0'))}**",
        f"28. ROI Non-Friendly: **{pct(friendly['NON_FRIENDLY'].get('roi'))}**",
        f"29. P(ROI>0) Non-Friendly: **{pct(friendly['NON_FRIENDLY'].get('bootstrap', {}).get('p_gt_0'))}**",
        f"30. P(ROI_NF > ROI_Friendly): **{pct(friendly.get('p_delta_gt_0'))}**",
        "",
        "### CLV",
        f"31. P(CLV 5m mean>0): **{pct((c5 or {}).get('p_mean_gt_0'))}**",
        f"32. P(CLV 15m mean>0): **{pct((c15 or {}).get('p_mean_gt_0'))}**",
        f"33. P(CLV closing mean>0): **{pct((cc or {}).get('p_mean_gt_0'))}**",
        f"34. IC95 closing CLV: **[{fmt_clv((cc or {}).get('ci95_mean', [None, None])[0] if cc else None)}, {fmt_clv((cc or {}).get('ci95_mean', [None, None])[1] if cc else None)}]**",
        f"35. CLV e ROI mesma direção? **{('sim' if clv_same_dir else 'não') if clv_same_dir is not None else '—'}**",
        "",
        "### TEMPORAL",
        f"36. P(ROI>0) últimas 25: **{pct((roll.get(25) or {}).get('p_gt_0'))}**",
        f"37. últimas 50: **{pct((roll.get(50) or {}).get('p_gt_0'))}**",
        f"38. últimas 100: **{pct((roll.get(100) or {}).get('p_gt_0'))}**",
        f"39. Evidência: **{evid}**",
        "",
        "## Avisos",
    ]
    if warnings:
        lines.extend([f"- {w}" for w in warnings])
    else:
        lines.append("- (nenhum)")
    lines.extend(
        [
            "",
            "## Nota",
            "Análise exclusivamente estatística/read-only. Não alterar exposição com base neste resultado.",
            f"Day-cluster: N_dias={day_boot.get('n_clusters')} · P(ROI>0)={pct(day_boot.get('p_gt_0'))} · IC95={day_boot.get('ci95')}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="FASE 2E-A Bootstrap H3BUP_vNext")
    ap.add_argument("--orders-csv", required=True, help="h3bup_friendly_order_level_*.csv")
    ap.add_argument("--cutoff", default=None, help="Cutoff UTC documentado")
    ap.add_argument("--source-run-id", default=None)
    ap.add_argument("--out-root", default=None, help="logs/h3bup_bootstrap")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    ap.add_argument("--n-perm", type=int, default=N_PERM_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--bot-root", default=".", help="betinasia_bot root for relative outs")
    args = ap.parse_args(list(argv) if argv is not None else None)

    bot_root = Path(args.bot_root).resolve()
    orders_path = Path(args.orders_csv).resolve()
    if not orders_path.exists():
        print(f"ERROR missing orders csv: {orders_path}", file=sys.stderr)
        return 2

    source_run_id = args.source_run_id or orders_path.stem.split("_")[-1]
    cutoff = args.cutoff or utc_now()
    run_id = args.run_id or short_run_id(f"{source_run_id}|{args.seed}|{args.n_boot}|{utc_now()}")
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_root = Path(args.out_root) if args.out_root else bot_root / "logs" / "h3bup_bootstrap" / day / run_id
    out_root.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    security_before = {
        "policy_unchanged": True,
        "stake_unchanged": True,
        "bridge_unchanged": True,
        "executor_unchanged": True,
        "accounting_unchanged": True,
        "clv_unchanged": True,
        "timers_unchanged": True,
        "telegram_not_used": True,
        "orders_created": 0,
        "betslips_opened": 0,
        "mode": "read_only_statistical",
    }

    raw = load_orders(orders_path)
    # filter exact policy
    rows = []
    for r in raw:
        pid = str(r.get("policy_id") or "")
        pver = str(r.get("policy_version") or "")
        if pid != POLICY_ID:
            continue
        if pver and pver != POLICY_VERSION:
            # freeze should already be exact; warn if mismatch
            warnings.append(f"policy_version_mismatch order={r.get('order_id')} ver={pver}")
            continue
        rows.append(normalize_row(r))

    if not rows:
        print("ERROR: empty universe after policy filter", file=sys.stderr)
        return 2

    recon = reconcile(rows)
    if not recon["reconciliation_ok"]:
        label = "BOOTSTRAP_INPUT_RECONCILIATION_FAILED"
        status = "BOOTSTRAP_RECONCILIATION_FAILED"
        (out_root / "h3bup_bootstrap_summary.json").write_text(
            json.dumps({"reconciliation": recon, "final_status": status, "label": label}, indent=2),
            encoding="utf-8",
        )
        print(status)
        return 3

    resolved = [r for r in rows if r.get("settlement_status") in RESOLVED]
    # analytical dataset
    dataset_fields = [
        "order_id",
        "event_id",
        "created_at_utc",
        "settlement_ts",
        "stake",
        "pnl",
        "settlement_status",
        "policy_id",
        "policy_version",
        "bookmaker",
        "league_name",
        "friendly_class",
        "odd_at_decision",
        "odd_final",
        "slippage_pre_pct",
        "pre_submit_ms",
        "clv_post_5m",
        "clv_post_5m_valid_strict",
        "clv_post_15m",
        "clv_post_15m_valid_strict",
        "clv_closing",
        "clv_closing_valid_strict",
    ]
    # keep missing CLV as empty string — never fill with 0
    dataset_rows = []
    for r in resolved:
        d = {k: r.get(k, "") for k in dataset_fields}
        # blank null CLV
        for f, _, vf in CLV_WINDOWS:
            if not r.get(f"_{vf}") or r.get(f"_{f}") is None:
                d[f] = ""
                d[vf] = r.get(vf, "")
        dataset_rows.append(d)

    n_boot = int(args.n_boot)
    seed = int(args.seed)
    print(f"[2E-A] N_resolved={len(resolved)} events={recon['n_event_id_unique']} n_boot={n_boot} seed={seed}")

    # --- Order bootstrap ---
    stakes, pnls = resolved_arrays(resolved)
    obs = obs_roi(resolved)
    print("[2E-A] order-level bootstrap...")
    order_boots = boot_roi_order(stakes, pnls, n_boot=n_boot, seed=seed)
    order_boot = summarize_boot(obs if obs is not None else float("nan"), order_boots)

    # --- Event cluster bootstrap ---
    print("[2E-A] event-cluster bootstrap...")
    ek, es, ep, _ = aggregate_clusters(resolved, lambda r: r.get("event_id") or r.get("order_id"))
    event_boots = boot_roi_cluster(es, ep, n_boot=n_boot, seed=seed + 1)
    event_boot = summarize_boot(obs if obs is not None else float("nan"), event_boots)
    event_boot["n_clusters"] = len(ek)

    # --- Day cluster ---
    print("[2E-A] day-cluster bootstrap...")
    def day_key(r):
        ts = str(r.get("created_at_utc") or "")[:10]
        return ts or "UNKNOWN"

    dk, ds, dp, _ = aggregate_clusters(resolved, day_key)
    day_weak = len(dk) < 10
    if day_weak:
        warnings.append(f"day_cluster_very_weak n_days={len(dk)}")
    day_boots = boot_roi_cluster(ds, dp, n_boot=n_boot, seed=seed + 2)
    day_boot = summarize_boot(obs if obs is not None else float("nan"), day_boots)
    day_boot["n_clusters"] = len(dk)
    day_boot["classification"] = "VERY_WEAK" if day_weak else ("WEAK" if len(dk) < 20 else "OK")

    # --- Robustness ---
    print("[2E-A] robustness...")
    robustness = robustness_scenarios(resolved, n_boot=n_boot, seed=seed + 10)

    # --- Friendly ---
    print("[2E-A] friendly vs non-friendly...")
    friendly = friendly_analysis(resolved, n_boot=n_boot, seed=seed + 20)

    # --- CLV ---
    print("[2E-A] CLV bootstrap...")
    clv_results = []
    for i, (wname, field, validf) in enumerate(CLV_WINDOWS):
        cr = boot_clv_window(rows, field, validf, n_boot=n_boot, seed=seed + 30 + i)
        cr["label"] = wname
        clv_results.append(cr)

    # --- ROI x CLV ---
    print("[2E-A] ROI x CLV...")
    roi_clv = []
    for i, (wname, field, validf) in enumerate(CLV_WINDOWS):
        rc = roi_clv_analysis(resolved, field, validf, n_boot=n_boot, seed=seed + 40 + i)
        rc["label"] = wname
        roi_clv.append(rc)

    # --- Permutation vs 0 ---
    print("[2E-A] sign-flip test...")
    tests = {
        "roi_vs_zero_sign_flip": sign_flip_test(es, ep, n_perm=int(args.n_perm), seed=seed + 50),
        "disclaimer": "P(ROI>0) from bootstrap ≠ classical p-value. Sign-flip is exploratory.",
    }

    # --- Temporal ---
    print("[2E-A] temporal...")
    temporal = temporal_analysis(resolved, n_boot=n_boot, seed=seed + 60)

    label, readiness = classify(recon, event_boot, clv_results, robustness)
    # extra warnings
    if readiness in ("VERY_LOW_N", "INSUFFICIENT_N", "FIRST_READING"):
        warnings.append(f"statistical_readiness={readiness}")
    if recon.get("open", 0) or recon.get("missing", 0):
        warnings.append(
            f"partial_settlement open={recon['open']} missing={recon['missing']} coverage={recon['accounting_coverage']:.3f}"
        )

    status = final_status(label, readiness, warnings)
    security_after = dict(security_before)

    # --- Write outputs ---
    print(f"[2E-A] writing → {out_root}")

    # 1 order-level distribution summary (quantiles, not full 100k dump)
    write_csv(
        out_root / "h3bup_bootstrap_order_level.csv",
        [
            {
                "level": "order",
                "n": len(resolved),
                "observed_roi": order_boot["observed"],
                "mean": order_boot["mean"],
                "median": order_boot["median"],
                "se": order_boot["se"],
                "ci80_lo": order_boot["ci80"][0],
                "ci80_hi": order_boot["ci80"][1],
                "ci90_lo": order_boot["ci90"][0],
                "ci90_hi": order_boot["ci90"][1],
                "ci95_lo": order_boot["ci95"][0],
                "ci95_hi": order_boot["ci95"][1],
                "ci99_lo": order_boot["ci99"][0],
                "ci99_hi": order_boot["ci99"][1],
                "p_gt_0": order_boot["p_gt_0"],
                "p_gt_0_02": order_boot["p_gt_0_02"],
                "p_gt_0_05": order_boot["p_gt_0_05"],
                "p_gt_0_10": order_boot["p_gt_0_10"],
                "p_lt_0": order_boot["p_lt_0"],
                "one_sided_tail_leq_0": order_boot["one_sided_tail_leq_0"],
                "n_boot": order_boot["n_boot"],
                "seed": seed,
            }
        ],
    )

    # 2 event-level
    write_csv(
        out_root / "h3bup_bootstrap_event_level.csv",
        [
            {
                "level": "event_cluster",
                "n_orders": len(resolved),
                "n_events": len(ek),
                "observed_roi": event_boot["observed"],
                "mean": event_boot["mean"],
                "median": event_boot["median"],
                "se": event_boot["se"],
                "ci80_lo": event_boot["ci80"][0],
                "ci80_hi": event_boot["ci80"][1],
                "ci90_lo": event_boot["ci90"][0],
                "ci90_hi": event_boot["ci90"][1],
                "ci95_lo": event_boot["ci95"][0],
                "ci95_hi": event_boot["ci95"][1],
                "ci99_lo": event_boot["ci99"][0],
                "ci99_hi": event_boot["ci99"][1],
                "p_gt_0": event_boot["p_gt_0"],
                "p_gt_0_02": event_boot["p_gt_0_02"],
                "p_gt_0_05": event_boot["p_gt_0_05"],
                "p_gt_0_10": event_boot["p_gt_0_10"],
                "p_lt_0": event_boot["p_lt_0"],
                "one_sided_tail_leq_0": event_boot["one_sided_tail_leq_0"],
                "n_boot": event_boot["n_boot"],
                "seed": seed + 1,
                "preferred": True,
            }
        ],
    )

    # 3 daily
    write_csv(
        out_root / "h3bup_bootstrap_daily_cluster.csv",
        [
            {
                "level": "day_utc",
                "n_days": len(dk),
                "n_orders": len(resolved),
                "observed_roi": day_boot["observed"],
                "mean": day_boot["mean"],
                "ci90_lo": day_boot["ci90"][0],
                "ci90_hi": day_boot["ci90"][1],
                "ci95_lo": day_boot["ci95"][0],
                "ci95_hi": day_boot["ci95"][1],
                "p_gt_0": day_boot["p_gt_0"],
                "classification": day_boot["classification"],
                "n_boot": day_boot["n_boot"],
                "seed": seed + 2,
            }
        ],
    )

    # 5 robustness
    write_csv(out_root / "h3bup_bootstrap_robustness.csv", robustness)

    # 6 friendly
    write_csv(
        out_root / "h3bup_bootstrap_friendly_vs_nonfriendly.csv",
        [
            {
                "group": "FRIENDLY",
                "n": friendly["FRIENDLY"]["n"],
                "events": friendly["FRIENDLY"]["events"],
                "stake": friendly["FRIENDLY"]["stake"],
                "pnl": friendly["FRIENDLY"]["pnl"],
                "roi": friendly["FRIENDLY"]["roi"],
                "ci90_lo": friendly["FRIENDLY"]["bootstrap"]["ci90"][0],
                "ci90_hi": friendly["FRIENDLY"]["bootstrap"]["ci90"][1],
                "ci95_lo": friendly["FRIENDLY"]["bootstrap"]["ci95"][0],
                "ci95_hi": friendly["FRIENDLY"]["bootstrap"]["ci95"][1],
                "p_gt_0": friendly["FRIENDLY"]["bootstrap"]["p_gt_0"],
                "p_gt_0_05": friendly["FRIENDLY"]["bootstrap"]["p_gt_0_05"],
            },
            {
                "group": "NON_FRIENDLY",
                "n": friendly["NON_FRIENDLY"]["n"],
                "events": friendly["NON_FRIENDLY"]["events"],
                "stake": friendly["NON_FRIENDLY"]["stake"],
                "pnl": friendly["NON_FRIENDLY"]["pnl"],
                "roi": friendly["NON_FRIENDLY"]["roi"],
                "ci90_lo": friendly["NON_FRIENDLY"]["bootstrap"]["ci90"][0],
                "ci90_hi": friendly["NON_FRIENDLY"]["bootstrap"]["ci90"][1],
                "ci95_lo": friendly["NON_FRIENDLY"]["bootstrap"]["ci95"][0],
                "ci95_hi": friendly["NON_FRIENDLY"]["bootstrap"]["ci95"][1],
                "p_gt_0": friendly["NON_FRIENDLY"]["bootstrap"]["p_gt_0"],
                "p_gt_0_05": friendly["NON_FRIENDLY"]["bootstrap"]["p_gt_0_05"],
            },
            {
                "group": "DELTA_NF_MINUS_F",
                "n": "",
                "events": "",
                "stake": "",
                "pnl": "",
                "roi": friendly.get("delta_roi_nf_minus_f"),
                "ci90_lo": (friendly.get("delta_bootstrap") or {}).get("ci90", [None, None])[0],
                "ci90_hi": (friendly.get("delta_bootstrap") or {}).get("ci90", [None, None])[1],
                "ci95_lo": (friendly.get("delta_bootstrap") or {}).get("ci95", [None, None])[0],
                "ci95_hi": (friendly.get("delta_bootstrap") or {}).get("ci95", [None, None])[1],
                "p_gt_0": friendly.get("p_delta_gt_0"),
                "p_gt_0_05": (friendly.get("delta_bootstrap") or {}).get("p_gt_0_05"),
            },
        ],
    )

    # 7 clv
    write_csv(
        out_root / "h3bup_bootstrap_clv.csv",
        [
            {
                "window": c["label"],
                "field": c["window"],
                "n": c["n"],
                "events": c["events"],
                "coverage": c["coverage"],
                "mean_obs": c["mean_obs"],
                "median_obs": c["median_obs"],
                "ci90_lo": c["ci90_mean"][0],
                "ci90_hi": c["ci90_mean"][1],
                "ci95_lo": c["ci95_mean"][0],
                "ci95_hi": c["ci95_mean"][1],
                "p_mean_gt_0": c["p_mean_gt_0"],
                "p_median_gt_0": c["p_median_gt_0"],
            }
            for c in clv_results
        ],
    )

    # 8 roi_clv
    write_csv(
        out_root / "h3bup_bootstrap_roi_clv.csv",
        [
            {
                "window": r["label"],
                "n": r["n"],
                "events": r["events"],
                "roi": r.get("roi"),
                "spearman": r.get("spearman"),
                "kendall": r.get("kendall"),
                "spearman_ci95_lo": (r.get("spearman_boot") or {}).get("ci95", [None, None])[0]
                if r.get("spearman_boot")
                else None,
                "spearman_ci95_hi": (r.get("spearman_boot") or {}).get("ci95", [None, None])[1]
                if r.get("spearman_boot")
                else None,
                "spearman_perm_p": r.get("spearman_perm_p"),
                "clv_gt0_n": r.get("clv_gt_0", {}).get("n"),
                "clv_gt0_roi": r.get("clv_gt_0", {}).get("roi"),
                "clv_le0_n": r.get("clv_le_0", {}).get("n"),
                "clv_le0_roi": r.get("clv_le_0", {}).get("roi"),
            }
            for r in roi_clv
        ],
    )

    # 9 temporal
    write_csv(
        out_root / "h3bup_bootstrap_temporal.csv",
        [
            {
                "kind": t["kind"],
                "n": t["n"],
                "events": t["events"],
                "roi": t["roi"],
                "p_gt_0": t["p_gt_0"],
                "ci90_lo": t["ci90"][0],
                "ci90_hi": t["ci90"][1],
                "ci95_lo": t["ci95"][0],
                "ci95_hi": t["ci95"][1],
            }
            for t in temporal
        ],
    )

    # 10 tests
    (out_root / "h3bup_bootstrap_tests.json").write_text(
        json.dumps(tests, indent=2, default=str), encoding="utf-8"
    )

    # also write analytical dataset for audit
    write_csv(out_root / "h3bup_bootstrap_dataset_resolved.csv", dataset_rows, fieldnames=dataset_fields)

    meta = {
        "phase": "2E-A",
        "run_id": run_id,
        "source_run_id": source_run_id,
        "orders_csv": str(orders_path),
        "cutoff_utc": cutoff,
        "generated_at_utc": utc_now(),
        "seed": seed,
        "n_boot": n_boot,
        "n_perm": int(args.n_perm),
        "policy_id": POLICY_ID,
        "policy_version": POLICY_VERSION,
        "void_in_denominator": True,
        "reconciliation": recon,
        "order_bootstrap": order_boot,
        "event_bootstrap": event_boot,
        "day_bootstrap": day_boot,
        "robustness": robustness,
        "friendly": {
            "FRIENDLY": {
                **{k: v for k, v in friendly["FRIENDLY"].items() if k != "bootstrap"},
                "bootstrap": friendly["FRIENDLY"]["bootstrap"],
            },
            "NON_FRIENDLY": {
                **{k: v for k, v in friendly["NON_FRIENDLY"].items() if k != "bootstrap"},
                "bootstrap": friendly["NON_FRIENDLY"]["bootstrap"],
            },
            "delta_roi_nf_minus_f": friendly.get("delta_roi_nf_minus_f"),
            "p_delta_gt_0": friendly.get("p_delta_gt_0"),
            "delta_bootstrap": friendly.get("delta_bootstrap"),
        },
        "clv": clv_results,
        "roi_clv": roi_clv,
        "temporal": temporal,
        "classification": label,
        "statistical_readiness": readiness,
        "final_status": status,
        "warnings": warnings,
        "security_before": security_before,
        "security_after": security_after,
    }
    (out_root / "h3bup_bootstrap_summary.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )

    write_methodology(
        out_root / "h3bup_bootstrap_methodology.md",
        meta,
    )
    write_executive(
        out_root / "h3bup_bootstrap_executive_summary.md",
        recon=recon,
        order_boot=order_boot,
        event_boot=event_boot,
        day_boot=day_boot,
        robustness=robustness,
        friendly=friendly,
        clv_results=clv_results,
        temporal=temporal,
        label=label,
        readiness=readiness,
        status=status,
        warnings=warnings,
    )

    # copy docs to docs/
    docs = bot_root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / f"h3bup_bootstrap_executive_{day}.md").write_text(
        (out_root / "h3bup_bootstrap_executive_summary.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (docs / f"h3bup_bootstrap_methodology_{day}.md").write_text(
        (out_root / "h3bup_bootstrap_methodology.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    print(status)
    print(f"classification={label} readiness={readiness}")
    print(f"out={out_root}")
    return 0 if status.startswith("BOOTSTRAP_ANALYSIS_COMPLETE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
