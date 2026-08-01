#!/usr/bin/env python3
"""H3BUP Friendly vs Non-Friendly historical analysis runner (read-only).

Freeze classification (checksum) BEFORE joining P&L / settlement / CLV.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ops.h3bup_friendly_analysis import FRIENDLY_CLASSIFICATION_VERSION
from ops.h3bup_friendly_analysis.classification import (
    build_classification_mapping,
    write_freeze_artifacts,
)
from ops.h3bup_friendly_analysis.clv_join import attach_clv, clv_summary_for_group, load_clv_by_order
from ops.h3bup_friendly_analysis.data_quality import build_alerts, data_quality_report
from ops.h3bup_friendly_analysis.metrics import (
    bookmaker_breakdown,
    concentration_report,
    counterfactual_scenarios,
    cumulative_series,
    daily_performance,
    execution_summary,
    group_by_class,
    leave_one_league_out,
    league_breakdown,
    performance_summary_table,
)
from ops.h3bup_friendly_analysis.report import (
    build_executive_summary,
    build_full_report,
    classify_final_status,
    write_simple_pdf,
)
from ops.h3bup_friendly_analysis.robustness import robustness_suite
from ops.h3bup_friendly_analysis.security import compare_checksums, snapshot_checksums
from ops.h3bup_friendly_analysis.settlement import attach_settlement, load_open_oids, load_pnl_by_order
from ops.h3bup_friendly_analysis.stats import run_stat_tests
from ops.h3bup_friendly_analysis.enrich import enrich_orders, load_league_map_csv, try_sql_league_map
from ops.h3bup_friendly_analysis.universe import (
    load_primary_h3bup_universe,
    load_secondary_historical_comparable,
    parse_dt,
)


ORDER_FIELDS = [
    "order_id",
    "execution_id",
    "audit_id",
    "trace_id",
    "event_id",
    "event_name",
    "league_id",
    "league_name",
    "competition_id",
    "competition_name",
    "country",
    "created_at_utc",
    "kickoff_utc",
    "policy_id",
    "policy_version",
    "side",
    "period",
    "line",
    "odd_at_decision",
    "odd_final",
    "stake",
    "capacity_final",
    "slippage_pre_pct",
    "pre_submit_ms",
    "call_to_done_ms",
    "place_duration_ms",
    "bookmaker",
    "settlement_status",
    "settlement_ts",
    "pnl",
    "friendly_class",
    "friendly_source",
    "friendly_rule_id",
    "friendly_confidence",
    "clv_post_5m",
    "clv_post_5m_valid_strict",
    "clv_post_15m",
    "clv_post_15m_valid_strict",
    "clv_closing",
    "clv_closing_valid_strict",
    "clv_failure_reason",
]


def _write_csv(path: Path, rows: List[Dict[str, Any]], fields: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fields:
        keys = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fields = keys or ["_empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def _latest_glob(dir_path: Path, pattern: str) -> List[Path]:
    if not dir_path.exists():
        return []
    return sorted(dir_path.glob(pattern), key=lambda p: p.stat().st_mtime)


def run_analysis(
    *,
    root: Path,
    cutoff: Optional[datetime] = None,
    run_id: Optional[str] = None,
    out_base: Optional[Path] = None,
    n_boot: int = 500,
    n_perm: int = 500,
    include_secondary: bool = True,
) -> Dict[str, Any]:
    root = root.resolve()
    cutoff = cutoff or datetime.now(timezone.utc)
    run_id = run_id or uuid.uuid4().hex[:12]
    day = cutoff.strftime("%Y%m%d")
    out_dir = out_base or (root / "logs" / "h3bup_friendly_analysis" / day / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_lines: List[str] = []

    def log(msg: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
        log_lines.append(line)
        print(line)

    log("START friendly analysis read-only")
    before = snapshot_checksums(root)

    executor = root / "logs" / "executor_live.jsonl"
    if not executor.exists():
        # alternate path used on some hosts
        alt = root / "logs" / "executor" / "executor_live.jsonl"
        if alt.exists():
            executor = alt

    balance_paths = _latest_glob(root / "logs" / "accounting", "*__balance.csv")[-5:]
    open_paths = _latest_glob(root / "logs" / "accounting", "*__open_stakes.csv")[-5:]
    clv_snaps = root / "logs" / "h3bup_clv_snapshots.jsonl"

    # 1) Universe (identity only)
    primary, primary_meta = load_primary_h3bup_universe(executor, cutoff=cutoff)
    log(f"primary universe n={len(primary)}")

    # Optional league enrichment (still pre-PnL)
    league_map_path = root / "logs" / "h3bup_friendly_league_map.csv"
    league_map = load_league_map_csv(league_map_path)
    sql_map = try_sql_league_map()
    league_map.update(sql_map)
    if league_map:
        primary = enrich_orders(primary, league_map=league_map)
        log(f"league map keys={len(league_map)}")

    # 2) FREEZE classification BEFORE settlement/CLV/PnL join
    # Strip any accidental performance keys
    identity_orders = []
    for o in primary:
        clean = {k: v for k, v in o.items() if k not in {"pnl", "settlement_status", "clv_post_5m", "clv_closing"}}
        identity_orders.append(clean)
    mapping = build_classification_mapping(identity_orders)
    freeze = write_freeze_artifacts(out_dir, mapping, run_id=run_id)
    checksum = freeze["sha256"]
    log(f"classification frozen sha256={checksum}")

    class_by_oid = {r["order_id"]: r for r in mapping if r.get("order_id")}

    # 3) Join settlement + CLV AFTER freeze
    pnl_map = load_pnl_by_order(balance_paths)
    open_oids = load_open_oids(open_paths)
    settled_rows = attach_settlement(identity_orders, pnl_by_oid=pnl_map, open_oids=open_oids)
    clv_map = load_clv_by_order(clv_snaps) if clv_snaps.exists() else {}
    joined = attach_clv(settled_rows, clv_map)

    # attach classification from frozen mapping only
    order_rows = []
    for r in joined:
        oid = str(r.get("order_id") or "")
        m = class_by_oid.get(oid) or {}
        # fallback for empty order_id: match by event+execution in mapping without oid
        row = dict(r)
        row["friendly_class"] = m.get("friendly_class") or "UNCLASSIFIED"
        row["friendly_source"] = m.get("friendly_source") or "none"
        row["friendly_rule_id"] = m.get("friendly_rule_id") or "R_UNCLASSIFIED"
        row["friendly_confidence"] = m.get("friendly_confidence") or "none"
        row["settlement_ts"] = None
        order_rows.append(row)

    # For fallback rows without order_id, classify inline from frozen mapping by rebuilding
    # Ensure all mapping applied: if oid empty, find mapping row by event_id
    by_event_map = {}
    for m in mapping:
        if not m.get("order_id") and m.get("event_id"):
            by_event_map[str(m["event_id"])] = m
    for row in order_rows:
        if not row.get("order_id") and row.get("event_id") in by_event_map:
            m = by_event_map[str(row["event_id"])]
            row["friendly_class"] = m["friendly_class"]
            row["friendly_source"] = m["friendly_source"]
            row["friendly_rule_id"] = m["friendly_rule_id"]
            row["friendly_confidence"] = m["friendly_confidence"]

    # Metrics
    perf_summary = performance_summary_table(order_rows)
    daily = daily_performance(order_rows)
    cum = cumulative_series(order_rows)
    exec_sum = execution_summary(order_rows)
    leagues = league_breakdown(order_rows)
    books = bookmaker_breakdown(order_rows)
    conc = concentration_report(order_rows)
    lolo = leave_one_league_out(order_rows)
    scenarios = counterfactual_scenarios(order_rows)
    robust = robustness_suite(order_rows)
    groups = group_by_class(order_rows)
    clv_sum = []
    for gname, gl in groups.items():
        clv_sum.extend(clv_summary_for_group(gl, group=gname))
    stats = run_stat_tests(order_rows, n_boot=n_boot, n_perm=n_perm)
    dq = data_quality_report(order_rows, mapping_rows=mapping)
    alerts = build_alerts(order_rows, dq=dq, concentration=conc)

    secondary_rows, secondary_meta = ([], {"skipped": True})
    if include_secondary:
        secondary_rows, secondary_meta = load_secondary_historical_comparable(executor, cutoff=cutoff)

    after = snapshot_checksums(root)
    security = compare_checksums(before, after)
    security["telegram_used"] = False
    security["orders_created"] = False
    security["betslip_opened"] = False
    security["env_changed"] = False

    bundle: Dict[str, Any] = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cutoff_utc": cutoff.isoformat(),
        "friendly_classification_version": FRIENDLY_CLASSIFICATION_VERSION,
        "classification_checksum": checksum,
        "primary_meta": primary_meta,
        "secondary_meta": secondary_meta,
        "order_rows": order_rows,
        "performance_summary": perf_summary,
        "daily_performance": daily,
        "cumulative": cum,
        "execution_summary": exec_sum,
        "league_breakdown": leagues,
        "bookmaker_breakdown": books,
        "concentration": conc,
        "leave_one_league_out": lolo,
        "scenarios": scenarios,
        "robustness": robust,
        "clv_summary": clv_sum,
        "stat_tests": stats,
        "data_quality": dq,
        "alerts": alerts,
        "security": security,
        "security_before": before,
        "security_after": after,
    }
    bundle["final_status"] = classify_final_status(bundle)

    # Write outputs
    _write_csv(out_dir / f"h3bup_friendly_order_level_{run_id}.csv", order_rows, ORDER_FIELDS)
    _write_csv(out_dir / f"h3bup_friendly_performance_summary_{run_id}.csv", perf_summary)
    _write_csv(out_dir / f"h3bup_friendly_daily_performance_{run_id}.csv", daily)
    _write_csv(out_dir / f"h3bup_friendly_clv_summary_{run_id}.csv", clv_sum)
    _write_csv(out_dir / f"h3bup_friendly_execution_summary_{run_id}.csv", exec_sum)
    _write_csv(out_dir / f"h3bup_friendly_league_breakdown_{run_id}.csv", leagues)
    _write_csv(out_dir / f"h3bup_friendly_bookmaker_breakdown_{run_id}.csv", books)
    _write_csv(out_dir / f"h3bup_friendly_concentration_{run_id}.csv", list(conc))
    _write_csv(out_dir / f"h3bup_friendly_robustness_{run_id}.csv", robust)
    _write_csv(out_dir / f"h3bup_friendly_alerts_{run_id}.csv", alerts)
    _write_csv(out_dir / f"h3bup_friendly_lolo_{run_id}.csv", lolo)
    _write_csv(out_dir / f"h3bup_friendly_scenarios_{run_id}.csv", scenarios)
    if secondary_rows:
        _write_csv(
            out_dir / f"h3bup_friendly_secondary_historical_{run_id}.csv",
            secondary_rows,
            ORDER_FIELDS[:20],
        )

    (out_dir / f"h3bup_friendly_stat_tests_{run_id}.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    (out_dir / f"h3bup_friendly_data_quality_{run_id}.json").write_text(
        json.dumps(dq, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    manifest = {
        "run_id": run_id,
        "root": str(root),
        "executor_jsonl": str(executor),
        "executor_exists": executor.exists(),
        "balance_paths": [str(p) for p in balance_paths],
        "open_paths": [str(p) for p in open_paths],
        "clv_snapshots": str(clv_snaps),
        "clv_exists": clv_snaps.exists(),
        "cutoff_utc": cutoff.isoformat(),
        "outputs_dir": str(out_dir),
        "read_only": True,
        "operational_changes": False,
    }
    (out_dir / f"h3bup_friendly_source_manifest_{run_id}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    exec_md = build_executive_summary(bundle)
    full_md = build_full_report(bundle)
    (out_dir / f"h3bup_friendly_executive_summary_{run_id}.md").write_text(exec_md, encoding="utf-8")
    # also docs paths relative to root
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    design = docs / f"h3bup_friendly_analysis_design_{day}.md"
    report = docs / f"h3bup_friendly_analysis_report_{day}.md"
    if not design.exists():
        design.write_text(
            "# Design — H3BUP Friendly vs Non-Friendly\n\n"
            f"- classification_version: `{FRIENDLY_CLASSIFICATION_VERSION}`\n"
            "- freeze-before-results: mapping checksum before P&L/CLV join\n"
            "- primary universe: H3BUP_vNext_20260629 LIVE_OK Back Pre\n"
            "- secondary: HISTORICAL_COMPARABLE_BACK_PRE (diagnostic only)\n"
            "- no operational changes\n",
            encoding="utf-8",
        )
    report.write_text(full_md, encoding="utf-8")
    # copy report into out_dir too
    (out_dir / f"h3bup_friendly_analysis_report_{day}.md").write_text(full_md, encoding="utf-8")

    pdf_path = out_dir / f"H3BUP_FRIENDLY_VS_NON_FRIENDLY_{day}_{run_id}.pdf"
    write_simple_pdf(pdf_path, f"H3BUP Friendly vs Non-Friendly {day}", exec_md)

    log(f"final_status={bundle['final_status']}")
    log(f"outputs={out_dir}")
    (out_dir / f"h3bup_friendly_execution_log_{run_id}.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    # slim bundle for JSON (without full order_rows duplication if huge)
    slim = {k: v for k, v in bundle.items() if k != "order_rows"}
    slim["n_order_rows"] = len(order_rows)
    (out_dir / f"h3bup_friendly_bundle_{run_id}.json").write_text(
        json.dumps(slim, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    return bundle


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="H3BUP Friendly analysis (read-only)")
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--cutoff", type=str, default="", help="ISO UTC cutoff")
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--out-base", type=Path, default=None)
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--n-perm", type=int, default=500)
    p.add_argument("--no-secondary", action="store_true")
    args = p.parse_args(argv)
    cutoff = parse_dt(args.cutoff) if args.cutoff else datetime.now(timezone.utc)
    try:
        bundle = run_analysis(
            root=args.root,
            cutoff=cutoff,
            run_id=args.run_id or None,
            out_base=args.out_base,
            n_boot=args.n_boot,
            n_perm=args.n_perm,
            include_secondary=not args.no_secondary,
        )
        print(json.dumps({"ok": True, "final_status": bundle.get("final_status"), "n": len(bundle.get("order_rows") or [])}))
        return 0
    except Exception as e:
        traceback.print_exc()
        print(json.dumps({"ok": False, "error": str(e)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
