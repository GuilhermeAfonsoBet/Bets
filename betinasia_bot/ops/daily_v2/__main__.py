"""CLI entrypoint for H3BUP Daily V2 (shadow / publish-gated)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import traceback
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .canonical import build_snapshot
from .compare_v1 import compare_snapshots, write_comparison_csv
from .contracts import catalog_v2
from .io_atomic import atomic_write_json, atomic_write_text, promote_last_known_good, update_latest_symlink
from .render import render_markdown
from .time_windows import resolve_window


def _env_bool(k: str, default: str = "0") -> bool:
    return str(os.getenv(k, default)).strip().lower() in {"1", "true", "yes", "on"}


def _parse_day(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    return date.fromisoformat(s)


def run(argv: Optional[list] = None) -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser(description="H3BUP Daily V2 (reporting only)")
    ap.add_argument("--root", default=os.getenv("H3BUP_DAILY_V2_ROOT", "."))
    ap.add_argument("--out-dir", default=os.getenv("H3BUP_DAILY_V2_OUT_DIR", "logs/daily_v2"))
    ap.add_argument("--report-type", default="DAILY_CLOSED", choices=["DAILY_CLOSED", "INTRADAY"])
    ap.add_argument("--report-date", default=None, help="YYYY-MM-DD UTC cohort date")
    ap.add_argument("--require-h3bup", action="store_true", default=True)
    ap.add_argument("--no-require-h3bup", action="store_true")
    ap.add_argument("--v1-md", default=None, help="Optional V1 report_daily.md for comparison")
    ap.add_argument("--publish", action="store_true", default=False)
    args = ap.parse_args(argv)

    enabled = _env_bool("H3BUP_DAILY_V2_ENABLED", "1")
    publish = bool(args.publish) or _env_bool("H3BUP_DAILY_V2_PUBLISH", "0")
    compare = _env_bool("H3BUP_DAILY_V2_COMPARE_V1", "1")
    fail_open = _env_bool("H3BUP_DAILY_V2_FAIL_OPEN", "1")

    if not enabled:
        print(json.dumps({"skipped": True, "reason": "H3BUP_DAILY_V2_ENABLED=0"}))
        return 0

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    require_h3bup = not bool(args.no_require_h3bup)
    day = _parse_day(args.report_date)
    perf: Dict[str, Any] = {"sections": {}}

    try:
        t_build = time.time()
        win = resolve_window(report_type=args.report_type, report_date=day)
        snap = build_snapshot(root=root, window=win, require_h3bup=require_h3bup)
        perf["sections"]["build_s"] = time.time() - t_build

        day_s = snap["report_date_utc"].replace("-", "")
        run_id = snap["run_id"]
        snap_path = out_dir / f"h3bup_daily_snapshot_{day_s}_{run_id}.json"
        md_path = out_dir / f"h3bup_daily_report_{day_s}_{run_id}.md"
        health_path = out_dir / f"h3bup_daily_health_{day_s}_{run_id}.json"
        exc_path = out_dir / f"h3bup_daily_exceptions_{day_s}_{run_id}.csv"

        atomic_write_json(snap_path, snap)
        md = render_markdown(snap)
        atomic_write_text(md_path, md)

        health = {
            "report_health": snap.get("report_health"),
            "operations_health": snap.get("operations_health"),
            "data_quality": snap.get("data_quality"),
            "statistical_readiness": snap.get("statistical_readiness"),
            "source_manifest": {
                k: {"status": v.get("status"), "cutoff": v.get("source_cutoff_utc"), "age_seconds": v.get("age_seconds")}
                for k, v in (snap.get("source_manifest") or {}).items()
            },
            "generated_at_utc": snap.get("generated_at_utc"),
            "run_id": run_id,
            "publish": False,
        }
        atomic_write_json(health_path, health)

        # exceptions csv
        with exc_path.open("w", encoding="utf-8", newline="") as f:
            fields = ["alert_id", "severity", "status", "evidence", "affected_metrics"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for e in snap.get("exceptions") or []:
                w.writerow(
                    {
                        "alert_id": e.get("alert_id"),
                        "severity": e.get("severity"),
                        "status": e.get("status"),
                        "evidence": json.dumps(e.get("evidence"), ensure_ascii=False),
                        "affected_metrics": ",".join(e.get("affected_metrics") or []),
                    }
                )

        # LKG only if report not CRITICAL/FAILED
        rh = (snap.get("report_health") or {}).get("status")
        if rh not in {"FAILED"}:
            promote_last_known_good(out_dir / "lkg", snap_path, md_path)

        # shadow latest pointers (not official publish)
        update_latest_symlink(out_dir / "latest_snapshot.json", snap_path)
        update_latest_symlink(out_dir / "latest_report.md", md_path)

        cmp_path = None
        if compare:
            v1_md = None
            if args.v1_md and Path(args.v1_md).exists():
                v1_md = Path(args.v1_md).read_text(encoding="utf-8", errors="replace")
            rows = compare_snapshots(v2=snap, v1_md=v1_md)
            cmp_path = root / "logs" / f"h3bup_daily_v1_vs_v2_{day_s}.csv"
            write_comparison_csv(cmp_path, rows)

        # catalog export
        cat_path = root / "logs" / f"h3bup_daily_metric_catalog_v2_{day_s}.csv"
        cat = catalog_v2()
        if cat:
            with cat_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(cat[0].keys()))
                w.writeheader()
                for r in cat:
                    w.writerow(r)

        published = False
        publish_path = None
        if publish:
            # Controlled publish: copy into versioned published/ path WITHOUT deleting V1
            pub = out_dir / "published" / day_s
            pub.mkdir(parents=True, exist_ok=True)
            publish_path = pub / f"report_{run_id}.md"
            atomic_write_text(publish_path, md)
            atomic_write_json(pub / f"snapshot_{run_id}.json", snap)
            published = True
            health["publish"] = True
            atomic_write_json(health_path, health)

        perf.update(
            {
                "total_s": time.time() - t0,
                "snapshot_bytes": snap_path.stat().st_size,
                "report_bytes": md_path.stat().st_size,
                "published": published,
                "publish_path": str(publish_path) if publish_path else None,
            }
        )
        perf_path = root / "logs" / f"h3bup_daily_v2_performance_{day_s}.json"
        atomic_write_json(perf_path, perf)

        print(
            json.dumps(
                {
                    "ok": True,
                    "snapshot": str(snap_path),
                    "report": str(md_path),
                    "health": str(health_path),
                    "exceptions": str(exc_path),
                    "compare": str(cmp_path) if cmp_path else None,
                    "published": published,
                    "report_health": rh,
                    "live_ok": ((snap.get("execution_funnel") or {}).get("live_ok") or {}).get("value"),
                    "elapsed_s": perf["total_s"],
                },
                ensure_ascii=False,
            )
        )
        return 0
    except Exception as e:
        err = {"ok": False, "error": str(e)[:400], "trace": traceback.format_exc()[-2000:]}
        try:
            atomic_write_json(out_dir / "h3bup_daily_v2_last_error.json", err)
        except Exception:
            pass
        print(json.dumps(err, ensure_ascii=False))
        return 0 if fail_open else 1


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
