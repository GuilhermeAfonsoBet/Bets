#!/usr/bin/env python3
"""Offline deep-dive on H3BUP Friendly freeze → strategy results artefacts.

Usage (from betinasia_bot/):
  python -m ops.h3bup_strategy_results.analyze_freeze \\
    --freeze logs/h3bup_friendly_analysis/20260801/78c9f53d95df \\
    --out-dir logs/h3bup_strategy_results/20260807
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import defaultdict
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


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def pct(x: Optional[float], digits: int = 1) -> str:
    if x is None:
        return "—"
    return f"{100 * x:.{digits}f}%"


def money(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{x:+.2f}" if x != 0 else "0.00"


def boot_roi(settled: List[Dict[str, Any]], *, cluster: bool, n: int = 10000, seed: int = 20260807) -> Dict[str, Any]:
    rng = random.Random(seed + (1 if cluster else 0))
    vals: List[float] = []
    if cluster:
        by_e: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in settled:
            by_e[str(r.get("event_id"))].append(r)
        keys = list(by_e.keys())
        for _ in range(n):
            sp = ss = 0.0
            for k in rng.choices(keys, k=len(keys)):
                for r in by_e[k]:
                    sp += fnum(r.get("pnl")) or 0.0
                    ss += fnum(r.get("stake")) or 0.0
            if ss:
                vals.append(sp / ss)
    else:
        pairs = [(fnum(r.get("pnl")) or 0.0, fnum(r.get("stake")) or 0.0) for r in settled]
        for _ in range(n):
            samp = rng.choices(pairs, k=len(pairs))
            sp = sum(p for p, _ in samp)
            ss = sum(s for _, s in samp)
            if ss:
                vals.append(sp / ss)
    vals.sort()
    return {
        "mean": statistics.mean(vals),
        "ci90": [vals[int(0.05 * len(vals))], vals[int(0.95 * len(vals)) - 1]],
        "ci95": [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]],
        "p_positive": sum(1 for v in vals if v > 0) / len(vals),
        "n": len(vals),
    }


def class_metrics(rows: List[Dict[str, Any]], cls: Optional[str] = None) -> Dict[str, Any]:
    sub = rows if cls is None else [r for r in rows if r.get("friendly_class") == cls]
    s = [r for r in sub if r.get("settlement_status") == "SETTLED_DECIDED"]
    v = [r for r in sub if r.get("settlement_status") == "VOID_PUSH"]
    o = [r for r in sub if r.get("settlement_status") == "OPEN"]
    p = sum(fnum(r.get("pnl")) or 0.0 for r in s)
    sr = sum(fnum(r.get("stake")) or 0.0 for r in s + v)
    sp = sum(fnum(r.get("stake")) or 0.0 for r in sub)
    wins = [fnum(r.get("pnl")) for r in s if (fnum(r.get("pnl")) or 0) > 0]
    losses = [fnum(r.get("pnl")) for r in s if (fnum(r.get("pnl")) or 0) < 0]
    wins_f = [x for x in wins if x is not None]
    losses_f = [x for x in losses if x is not None]
    return {
        "n": len(sub),
        "events": len({r.get("event_id") for r in sub}),
        "settled": len(s),
        "void": len(v),
        "open": len(o),
        "stake_placed": sp,
        "stake_resolved": sr,
        "pnl": p,
        "roi": (p / sr if sr else None),
        "wr": (sum(1 for r in s if (fnum(r.get("pnl")) or 0) > 0) / len(s) if s else None),
        "avg_win": statistics.mean(wins_f) if wins_f else None,
        "avg_loss": statistics.mean(losses_f) if losses_f else None,
        "accounting_coverage": ((len(s) + len(v)) / len(sub) if sub else None),
    }


def analyze(freeze_dir: Path, out_dir: Path, run_id: str) -> Path:
    rows = list(csv.DictReader((freeze_dir / [p.name for p in freeze_dir.glob("h3bup_friendly_order_level_*.csv")][0]).open()))
    settled = [r for r in rows if r.get("settlement_status") == "SETTLED_DECIDED"]
    voids = [r for r in rows if r.get("settlement_status") == "VOID_PUSH"]
    opens = [r for r in rows if r.get("settlement_status") == "OPEN"]
    resolved = settled + voids
    pnl = sum(fnum(r.get("pnl")) or 0.0 for r in settled)
    stake_res = sum(fnum(r.get("stake")) or 0.0 for r in resolved)
    roi = pnl / stake_res if stake_res else None
    boot_e = boot_roi(settled, cluster=True)
    boot_o = boot_roi(settled, cluster=False)
    out = out_dir / run_id
    out.mkdir(parents=True, exist_ok=True)
    bundle = {
        "meta": {
            "run_id": run_id,
            "freeze_dir": str(freeze_dir),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "n": len(rows),
            "n_settled": len(settled),
            "n_open": len(opens),
            "pnl_resolved": pnl,
            "roi_resolved": roi,
        },
        "by_class": {
            "TOTAL": class_metrics(rows),
            "FRIENDLY": class_metrics(rows, "FRIENDLY"),
            "NON_FRIENDLY": class_metrics(rows, "NON_FRIENDLY"),
        },
        "boot_roi_order": boot_o,
        "boot_roi_event_cluster": boot_e,
    }
    (out / f"h3bup_strategy_results_bundle_{run_id}.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--run-id", default="strategy_offline")
    args = ap.parse_args(argv)
    out = analyze(args.freeze, args.out_dir, args.run_id)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
