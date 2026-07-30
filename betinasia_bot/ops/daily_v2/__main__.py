"""CLI entrypoint for H3BUP Daily V2 (shadow + optional Telegram PREVIEW).

Flags (distinct semantics):
  H3BUP_DAILY_V2_ENABLED          — execute + write artifacts
  H3BUP_DAILY_V2_TELEGRAM_PREVIEW — send PREVIEW to Telegram (not official)
  H3BUP_DAILY_V2_OFFICIAL         — must stay 0 in this phase
  H3BUP_DAILY_V2_COMPARE_V1       — write V1×V2 comparison
  H3BUP_DAILY_V2_FAIL_OPEN        — never fail hard against ops

Legacy H3BUP_DAILY_V2_PUBLISH is treated as OFFICIAL only (not preview).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .canonical import build_snapshot
from .compare_v1 import compare_snapshots, write_comparison_csv
from .contracts import catalog_v2
from .cutoff import find_v1_report_md, resolve_parity_cutoffs
from .io_atomic import atomic_write_json, atomic_write_text, promote_last_known_good, update_latest_symlink
from .pdf_preview import pdf_contains_preview_label, render_preview_pdf
from .preview_labels import PREVIEW_BANNER, preview_pdf_filename, validate_preview_artifacts
from .render import render_markdown
from .telegram_preview import maybe_send_telegram_preview
from .time_windows import resolve_window


def _env_bool(k: str, default: str = "0") -> bool:
    return str(os.getenv(k, default)).strip().lower() in {"1", "true", "yes", "on"}


def _parse_day(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    return date.fromisoformat(s)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            try:
                s.flush()
            except Exception:
                pass
        return len(data)

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def run(argv: Optional[list] = None) -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser(description="H3BUP Daily V2 (shadow / telegram preview)")
    ap.add_argument("--root", default=os.getenv("H3BUP_DAILY_V2_ROOT", "."))
    ap.add_argument("--out-dir", default=os.getenv("H3BUP_DAILY_V2_OUT_DIR", "logs/daily_v2"))
    ap.add_argument("--report-type", default="DAILY_CLOSED", choices=["DAILY_CLOSED", "INTRADAY"])
    ap.add_argument("--report-date", default=None, help="YYYY-MM-DD UTC cohort date")
    ap.add_argument("--require-h3bup", action="store_true", default=True)
    ap.add_argument("--no-require-h3bup", action="store_true")
    ap.add_argument("--v1-md", default=None, help="Optional V1 report_daily.md for comparison")
    ap.add_argument("--telegram-preview", action="store_true", default=False)
    ap.add_argument("--no-telegram-preview", action="store_true", default=False)
    # legacy alias kept for compatibility; maps to OFFICIAL only
    ap.add_argument("--publish", action="store_true", default=False, help="legacy=OFFICIAL (forbidden in preview phase)")
    args = ap.parse_args(argv)

    enabled = _env_bool("H3BUP_DAILY_V2_ENABLED", "1")
    telegram_preview = _env_bool("H3BUP_DAILY_V2_TELEGRAM_PREVIEW", "0")
    if args.telegram_preview:
        telegram_preview = True
    if args.no_telegram_preview:
        telegram_preview = False
    official = _env_bool("H3BUP_DAILY_V2_OFFICIAL", "0") or bool(args.publish) or _env_bool(
        "H3BUP_DAILY_V2_PUBLISH", "0"
    )
    compare = _env_bool("H3BUP_DAILY_V2_COMPARE_V1", "1")
    fail_open = _env_bool("H3BUP_DAILY_V2_FAIL_OPEN", "1")

    if not enabled:
        print(json.dumps({"skipped": True, "reason": "H3BUP_DAILY_V2_ENABLED=0"}))
        return 0

    # Hard guard: this intervention must not make V2 official
    if official:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "H3BUP_DAILY_V2_OFFICIAL/PUBLISH=1 refused; V1 must remain official",
                    "telegram_preview": False,
                }
            )
        )
        return 0 if fail_open else 2

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    require_h3bup = not bool(args.no_require_h3bup)
    day = _parse_day(args.report_date)
    perf: Dict[str, Any] = {"sections": {}}
    log_fh = None
    old_stdout, old_stderr = sys.stdout, sys.stderr

    try:
        t_build = time.time()
        win = resolve_window(report_type=args.report_type, report_date=day)
        snap = build_snapshot(
            root=root, window=win, require_h3bup=require_h3bup, out_dir=out_dir
        )
        perf["sections"]["build_s"] = time.time() - t_build

        day_s = snap["report_date_utc"].replace("-", "")
        run_id = snap["run_id"]

        # Run log (immutable per run_id)
        log_path = out_dir / f"h3bup_daily_v2_run_{day_s}_{run_id}.log"
        log_fh = log_path.open("w", encoding="utf-8")
        sys.stdout = _Tee(old_stdout, log_fh)
        sys.stderr = _Tee(old_stderr, log_fh)

        # Parity cutoffs vs V1
        v1_path = Path(args.v1_md) if args.v1_md else find_v1_report_md(root, snap["report_date_utc"])
        parity = resolve_parity_cutoffs(
            root=root,
            report_date_utc=snap["report_date_utc"],
            v2_generated_at=datetime.fromisoformat(str(snap["generated_at_utc"]).replace("Z", "+00:00")),
            v1_md_path=v1_path,
        )
        # Attach parity metadata (without huge md dump in snapshot file)
        v1_md_text = parity.pop("v1_md_text", None)
        snap["parity"] = {k: v for k, v in parity.items()}
        cut = dict(snap.get("cutoffs") or {})
        cut["v1_report_cutoff_utc"] = parity.get("v1_report_cutoff_utc")
        cut["v2_comparison_cutoff_utc"] = parity.get("v2_comparison_cutoff_utc")
        cut["parity_status"] = parity.get("parity_status")
        snap["cutoffs"] = cut
        # Re-derive parity alerts into exceptions if needed
        from .health_model import derive_alerts

        parity_alerts = derive_alerts(
            health={
                "report_health": snap.get("report_health"),
                "operations_health": snap.get("operations_health"),
                "data_quality": snap.get("data_quality"),
                "statistical_readiness": snap.get("statistical_readiness"),
                "config": snap.get("config") or {},
            },
            settlement=snap.get("settlement") or {},
            clv=snap.get("clv") or {},
            latency=snap.get("latency") or {},
            parity_status=parity.get("parity_status"),
            now_iso=str(snap.get("generated_at_utc")),
        )
        existing = {e.get("alert_id") for e in (snap.get("exceptions") or [])}
        for al in parity_alerts:
            if al.get("alert_id") not in existing:
                snap.setdefault("exceptions", []).append(al)
                existing.add(al.get("alert_id"))
        snap["shadow"] = {
            "official": False,
            "telegram_preview_enabled": bool(telegram_preview),
            "label": PREVIEW_BANNER,
        }

        snap_path = out_dir / f"h3bup_daily_snapshot_{day_s}_{run_id}.json"
        md_path = out_dir / f"h3bup_daily_report_{day_s}_{run_id}.md"
        health_path = out_dir / f"h3bup_daily_health_{day_s}_{run_id}.json"
        exc_path = out_dir / f"h3bup_daily_exceptions_{day_s}_{run_id}.csv"
        cmp_path = out_dir / f"h3bup_daily_v1_vs_v2_{day_s}_{run_id}.csv"
        pdf_path = out_dir / preview_pdf_filename(day_s, run_id)

        atomic_write_json(snap_path, snap)
        md = render_markdown(snap)
        atomic_write_text(md_path, md)

        # PDF preview
        t_pdf = time.time()
        try:
            render_preview_pdf(md, pdf_path, root=root)
            pdf_ok = pdf_path.exists() and pdf_path.stat().st_size > 64
            pdf_labeled = pdf_contains_preview_label(pdf_path) if pdf_ok else False
        except Exception as e:
            pdf_ok = False
            pdf_labeled = False
            print(json.dumps({"pdf_error": str(e)[:240]}))
        perf["sections"]["pdf_s"] = time.time() - t_pdf

        health = {
            "report_health": snap.get("report_health"),
            "operations_health": snap.get("operations_health"),
            "data_quality": snap.get("data_quality"),
            "statistical_readiness": snap.get("statistical_readiness"),
            "config": snap.get("config"),
            "parity": snap.get("parity"),
            "cutoffs": snap.get("cutoffs"),
            "source_manifest": {
                k: {"status": v.get("status"), "cutoff": v.get("source_cutoff_utc"), "age_seconds": v.get("age_seconds")}
                for k, v in (snap.get("source_manifest") or {}).items()
            },
            "generated_at_utc": snap.get("generated_at_utc"),
            "run_id": run_id,
            "official": False,
            "telegram_preview_enabled": bool(telegram_preview),
            "pdf": str(pdf_path) if pdf_ok else None,
            "pdf_preview_labeled": bool(pdf_labeled),
        }
        atomic_write_json(health_path, health)

        with exc_path.open("w", encoding="utf-8", newline="") as f:
            fields = [
                "alert_id",
                "severity",
                "status",
                "first_seen_utc",
                "last_seen_utc",
                "message",
                "affected_metrics",
                "evidence",
                "resolution_hint",
            ]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for e in snap.get("exceptions") or []:
                w.writerow(
                    {
                        "alert_id": e.get("alert_id"),
                        "severity": e.get("severity"),
                        "status": e.get("status"),
                        "first_seen_utc": e.get("first_seen_utc"),
                        "last_seen_utc": e.get("last_seen_utc"),
                        "message": e.get("message"),
                        "affected_metrics": ",".join(e.get("affected_metrics") or []),
                        "evidence": json.dumps(e.get("evidence"), ensure_ascii=False),
                        "resolution_hint": e.get("resolution_hint"),
                    }
                )

        rh = (snap.get("report_health") or {}).get("status")
        if rh not in {"FAILED"}:
            promote_last_known_good(out_dir / "lkg", snap_path, md_path)

        # Shadow pointers ONLY under daily_v2/ — never V1 paths
        update_latest_symlink(out_dir / "latest_snapshot.json", snap_path)
        update_latest_symlink(out_dir / "latest_report.md", md_path)
        if pdf_ok:
            update_latest_symlink(out_dir / "latest_preview.pdf", pdf_path)

        parity_status = (snap.get("parity") or {}).get("parity_status") or "PARITY_COMPARISON_UNAVAILABLE"
        if compare:
            try:
                rows = compare_snapshots(v2=snap, v1_md=v1_md_text)
                write_comparison_csv(cmp_path, rows)
                # also mirror under logs/ for convenience (non-official)
                mirror = root / "logs" / f"h3bup_daily_v1_vs_v2_{day_s}.csv"
                write_comparison_csv(mirror, rows)
            except Exception as e:
                parity_status = "PARITY_COMPARISON_UNAVAILABLE"
                atomic_write_text(cmp_path.with_suffix(".error.txt"), str(e)[:500])

        cat_path = out_dir / f"h3bup_daily_metric_catalog_v2_{day_s}_{run_id}.csv"
        try:
            cat = catalog_v2()
            if cat:
                with cat_path.open("w", encoding="utf-8", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(cat[0].keys()))
                    w.writeheader()
                    for r in cat:
                        w.writerow(r)
                # best-effort mirror under logs/ (non-fatal if permission denied)
                try:
                    mirror = root / "logs" / f"h3bup_daily_metric_catalog_v2_{day_s}.csv"
                    with mirror.open("w", encoding="utf-8", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(cat[0].keys()))
                        w.writeheader()
                        for r in cat:
                            w.writerow(r)
                except Exception as e_mir:
                    print(json.dumps({"catalog_mirror_warn": str(e_mir)[:160]}))
        except Exception as e_cat:
            print(json.dumps({"catalog_warn": str(e_cat)[:160]}))

        # Telegram PREVIEW (after all artifacts)
        tg = {"telegram_status": "SKIPPED"}
        label_ok, label_reason = validate_preview_artifacts(md=md, pdf_name=pdf_path.name)
        if telegram_preview:
            if not pdf_ok:
                tg = {
                    "run_id": run_id,
                    "telegram_status": "FAILED",
                    "error": "pdf_not_ready",
                }
                atomic_write_json(out_dir / f"h3bup_daily_v2_telegram_preview_{day_s}_{run_id}.json", tg)
            elif not label_ok or not pdf_labeled:
                tg = {
                    "run_id": run_id,
                    "telegram_status": "PREVIEW_LABEL_VALIDATION_FAILED",
                    "error": label_reason if not label_ok else "pdf_bytes_missing_preview_label",
                }
                atomic_write_json(out_dir / f"h3bup_daily_v2_telegram_preview_{day_s}_{run_id}.json", tg)
            else:
                tg = maybe_send_telegram_preview(
                    root=root,
                    out_dir=out_dir,
                    day_s=day_s,
                    run_id=run_id,
                    snap=snap,
                    md_text=md,
                    pdf_path=pdf_path,
                    parity_status=parity_status,
                    telegram_preview_enabled=True,
                    official=False,
                )

        perf.update(
            {
                "total_s": time.time() - t0,
                "snapshot_bytes": snap_path.stat().st_size,
                "report_bytes": md_path.stat().st_size,
                "pdf_bytes": pdf_path.stat().st_size if pdf_ok else 0,
                "official": False,
                "telegram_preview": bool(telegram_preview),
                "telegram_status": tg.get("telegram_status"),
                "parity_status": parity_status,
            }
        )
        perf_path = out_dir / f"h3bup_daily_v2_performance_{day_s}_{run_id}.json"
        atomic_write_json(perf_path, perf)
        try:
            atomic_write_json(root / "logs" / f"h3bup_daily_v2_performance_{day_s}.json", perf)
        except Exception:
            pass

        out = {
            "ok": True,
            "snapshot": str(snap_path),
            "report": str(md_path),
            "pdf": str(pdf_path) if pdf_ok else None,
            "health": str(health_path),
            "exceptions": str(exc_path),
            "compare": str(cmp_path) if cmp_path.exists() else None,
            "run_log": str(log_path),
            "telegram_evidence": str(out_dir / f"h3bup_daily_v2_telegram_preview_{day_s}_{run_id}.json"),
            "official": False,
            "telegram_preview": bool(telegram_preview),
            "telegram_status": tg.get("telegram_status"),
            "telegram_message_id": tg.get("telegram_message_id"),
            "parity_status": parity_status,
            "report_health": rh,
            "live_ok": ((snap.get("execution_funnel") or {}).get("live_ok") or {}).get("value"),
            "elapsed_s": perf["total_s"],
            "label": PREVIEW_BANNER,
        }
        print(json.dumps(out, ensure_ascii=False))
        return 0
    except Exception as e:
        err = {"ok": False, "error": str(e)[:400], "trace": traceback.format_exc()[-2000:]}
        try:
            atomic_write_json(out_dir / "h3bup_daily_v2_last_error.json", err)
        except Exception:
            pass
        print(json.dumps(err, ensure_ascii=False))
        return 0 if fail_open else 1
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        if log_fh is not None:
            try:
                log_fh.close()
            except Exception:
                pass


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
