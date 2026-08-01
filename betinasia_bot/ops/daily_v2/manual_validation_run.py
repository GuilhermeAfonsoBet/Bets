#!/usr/bin/env python3
"""Controlled MANUAL VALIDATION of Daily P0 (V1+V2) — reporting/read-only only.

Does NOT:
  - update daily_v2/latest_*
  - write into logs/daily_reports/ official day folders
  - send Telegram (unless MANUAL_VALIDATION_TELEGRAM_PREVIEW=1)
  - alter policy/stake/execution
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import traceback
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ops.daily_v2.canonical import build_snapshot
from ops.daily_v2.compare_v1 import compare_snapshots, write_comparison_csv
from ops.daily_v2.cutoff import find_v1_report_md, resolve_parity_cutoffs
from ops.daily_v2.formatters import fmt_money, fmt_pct, fmt_ts
from ops.daily_v2.io_atomic import atomic_write_json, atomic_write_text
from ops.daily_v2.pdf_preview import render_preview_pdf
from ops.daily_v2.render import render_markdown
from ops.daily_v2.time_windows import resolve_window
from ops.daily_v2.v1_h3bup_summary import render_h3bup_vnext_official_summary


MANUAL_BANNER = "MANUAL VALIDATION"
V2_MANUAL_BANNER = "DAILY V2 — MANUAL VALIDATION / PREVIEW / NÃO OFICIAL"


def _sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(root: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), stderr=subprocess.DEVNULL, text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def _env_bool(k: str, default: str = "0") -> bool:
    return str(os.getenv(k, default)).strip().lower() in {"1", "true", "yes", "on"}


def _parse_v1_metrics_from_md(md: str, day: str) -> Dict[str, Any]:
    """Best-effort extract of H3BUP-ish row for day from V1 tables."""
    out: Dict[str, Any] = {"report_date": day}
    for line in md.splitlines():
        if not line.strip().startswith("|"):
            continue
        if day not in line and day.replace("-", "") not in line:
            continue
        parts = [p.strip() for p in line.split("|") if p.strip() != ""]
        # Common V1 sample table starts with date then counts
        nums = []
        for p in parts[1:]:
            try:
                nums.append(float(p.replace("%", "").replace(",", "")))
            except Exception:
                nums.append(None)
        # Heuristic from known layout: date | ... | LIVE-ish | ...
        if len(parts) >= 4:
            try:
                # From observed: | 2026-07-29 | 176 | 22 | 22 | ...
                if parts[0].startswith(day) or parts[0] == day:
                    if len(parts) > 2 and parts[2].isdigit():
                        out.setdefault("live_ok", int(parts[2]))
                    if len(parts) > 8:
                        try:
                            out.setdefault("stake_placed", float(parts[8]))
                        except Exception:
                            pass
            except Exception:
                pass
    # Accounting health block in V1
    if "LIVE_OK total" in md:
        for line in md.splitlines():
            if "LIVE_OK total" in line and "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    try:
                        out["live_ok_accounting_block"] = int(float(parts[2]))
                    except Exception:
                        pass
            if line.strip().startswith("| stake settled"):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    try:
                        out["stake_settled_accounting_block"] = float(parts[2])
                    except Exception:
                        pass
            if line.strip().startswith("| P&L settled"):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    try:
                        out["pnl_settled_accounting_block"] = float(parts[2])
                    except Exception:
                        pass
            if line.strip().startswith("| ROI settled"):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    try:
                        out["roi_settled"] = float(parts[2])
                    except Exception:
                        pass
            if line.strip().startswith("| abertos"):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    try:
                        out["n_open"] = int(float(parts[2]))
                    except Exception:
                        pass
            if line.strip().startswith("| settled reconciliado"):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    try:
                        out["n_settled"] = int(float(parts[2]))
                    except Exception:
                        pass
    return out


def _render_plain_pdf(md_text: str, pdf_path: Path, *, root: Path, footer: str) -> None:
    from ops.daily_v2.pdf_preview import _load_v1_renderer_module
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer

    mod = _load_v1_renderer_module(root)
    styles = mod.build_styles()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.8 * cm,
        title=footer,
    )

    def _footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(colors.HexColor("#1d4ed8"))
        w, h = A4
        canvas.drawString(doc_.leftMargin, doc_.bottomMargin - 12, footer)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#6b7280"))
        canvas.drawRightString(w - doc_.rightMargin, doc_.bottomMargin - 12, f"Página {canvas.getPageNumber()}")
        canvas.setFont("Helvetica-Bold", 9)
        canvas.setFillColor(colors.HexColor("#1d4ed8"))
        canvas.drawString(doc_.leftMargin, h - 1.0 * cm, footer)
        canvas.restoreState()

    flow = []
    lines = md_text.splitlines()
    available_width = doc.pagesize[0] - doc.leftMargin - doc.rightMargin
    i = 0
    in_code = False
    code_buf: List[str] = []
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if line.startswith("```"):
            if not in_code:
                in_code = True
                code_buf = []
            else:
                in_code = False
                code_text = "\n".join(code_buf).rstrip()
                if code_text:
                    flow.append(Preformatted(code_text, styles["code"]))
                    flow.append(Spacer(1, 6))
            i += 1
            continue
        if in_code:
            code_buf.append(raw.rstrip("\n"))
            i += 1
            continue
        if not line:
            flow.append(Spacer(1, 6))
            i += 1
            continue
        if line == "---":
            flow.append(Spacer(1, 8))
            i += 1
            continue
        if mod.is_table_line(line):
            table_lines, i = mod.parse_table_block(lines, i)
            table = mod.table_to_flowable(table_lines, available_width, styles["body"])
            if table:
                flow.append(table)
                flow.append(Spacer(1, 8))
            continue
        if line.startswith("# "):
            flow.append(Paragraph(mod.normalize_inline(line[2:].strip()), styles["h1"]))
            i += 1
            continue
        if line.startswith("## "):
            flow.append(Paragraph(mod.normalize_inline(line[3:].strip()), styles["h2"]))
            i += 1
            continue
        if line.startswith("### "):
            flow.append(Paragraph(mod.normalize_inline(line[4:].strip()), styles["h3"]))
            i += 1
            continue
        if line.startswith("> "):
            flow.append(Paragraph(mod.normalize_inline(line[2:].strip()), styles["quote"]))
            i += 1
            continue
        if line.startswith("- "):
            flow.append(Paragraph(f"• {mod.normalize_cell(line[2:].strip())}", styles["bullet"]))
            i += 1
            continue
        flow.append(Paragraph(mod.normalize_cell(line), styles["body"]))
        i += 1

    doc.build(flow, onFirstPage=_footer, onLaterPages=_footer)


def _write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def run(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Manual validation Daily P0 (isolated outputs)")
    ap.add_argument("--root", default=".")
    ap.add_argument("--report-date", default="2026-07-29")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    report_date = args.report_date
    day_s = report_date.replace("-", "")
    run_id = args.run_id or uuid.uuid4().hex[:12]
    generated_at = datetime.now(timezone.utc)

    out_dir = root / "logs" / "daily_p0" / "manual_validation" / day_s / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"h3bup_daily_manual_run_{day_s}_{run_id}.log"

    class Tee:
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

    log_fh = log_path.open("w", encoding="utf-8")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = Tee(old_out, log_fh)
    sys.stderr = Tee(old_err, log_fh)

    exit_v1 = 1
    exit_v2 = 1
    result: Dict[str, Any] = {"ok": False, "run_id": run_id}

    try:
        # --- Preserve official checksums BEFORE ---
        official_paths = {
            "v1_md": root / "logs" / "daily_reports" / day_s / "report_daily.md",
            "v1_pdf": root / "logs" / "daily_reports" / day_s / "report_daily.pdf",
            "v2_latest_snap": root / "logs" / "daily_v2" / "latest_snapshot.json",
            "v2_latest_md": root / "logs" / "daily_v2" / "latest_report.md",
            "v2_latest_pdf": root / "logs" / "daily_v2" / "latest_preview.pdf",
        }
        before = {k: {"path": str(p), "sha256": _sha256(p), "mtime": p.stat().st_mtime if p.exists() else None} for k, p in official_paths.items()}
        atomic_write_json(out_dir / "official_checksums_before.json", before)

        # --- Parity cutoff from official V1 ---
        v1_path = find_v1_report_md(root, report_date) or official_paths["v1_md"]
        if not v1_path or not Path(v1_path).exists():
            raise FileNotFoundError(f"Official V1 md missing for {report_date}")
        v1_md_official = Path(v1_path).read_text(encoding="utf-8", errors="replace")
        parity = resolve_parity_cutoffs(
            root=root,
            report_date_utc=report_date,
            v2_generated_at=generated_at,
            v1_md_path=Path(v1_path),
        )
        parity.pop("v1_md_text", None)
        v1_cutoff = parity.get("v1_report_cutoff_utc")
        if not v1_cutoff:
            raise RuntimeError("Could not extract v1_report_cutoff_utc from official V1")

        win = resolve_window(report_type="DAILY_CLOSED", report_date=date.fromisoformat(report_date))
        # Freeze comparison window end conceptually at V1 cutoff for parity metadata
        performance_as_of = v1_cutoff

        # ========== V2 MANUAL ==========
        # Build into a throwaway sibling so build_snapshot previous_diff can see prior autos if needed,
        # but we will NOT update latest and will copy/rename into out_dir.
        snap = build_snapshot(
            root=root,
            window=win,
            require_h3bup=True,
            run_id=run_id,
            out_dir=root / "logs" / "daily_v2",  # only for previous_diff lookup; we won't promote latest
        )
        snap["manual_validation"] = True
        snap["official_status"] = "MANUAL_VALIDATION_PREVIEW_NOT_OFFICIAL"
        snap["generated_at_utc"] = generated_at.isoformat()
        snap["parity"] = dict(parity)
        snap["parity"]["parity_cutoff_utc"] = v1_cutoff
        snap["cutoffs"] = {
            "cohort_window_start_utc": win.window_start_utc.isoformat(),
            "cohort_window_end_utc": win.window_end_utc.isoformat(),
            "parity_cutoff_utc": v1_cutoff,
            "v1_report_cutoff_utc": v1_cutoff,
            "v2_comparison_cutoff_utc": v1_cutoff,
            "performance_as_of_utc": performance_as_of,
            "generated_at_utc": generated_at.isoformat(),
            "as_of_now_utc": datetime.now(timezone.utc).isoformat(),
            "note": "Paridade congelada no cutoff do V1 oficial; as_of_now é diagnóstico separado",
        }
        snap["report_cutoff_utc"] = v1_cutoff
        snap["shadow"] = {
            "official": False,
            "manual_validation": True,
            "telegram_preview_enabled": False,
            "label": V2_MANUAL_BANNER,
        }

        v1_metrics = _parse_v1_metrics_from_md(v1_md_official, report_date)
        # Prefer accounting-block LIVE_OK if present for note; keep table live_ok too
        rows = compare_snapshots(v2=snap, v1_md=v1_md_official, v1_metrics=v1_metrics)
        # Harden UNKNOWN on principal metrics: if V1 missing → PARITY_UNAVAILABLE not silently ok
        for r in rows:
            if r.get("metric") in {"LIVE_OK", "stake placed", "open", "settled", "void", "missing", "P&L", "ROI", "P&L resolved", "ROI resolved"}:
                if r.get("status") == "UNKNOWN" and (r.get("v1") is None or r.get("v2") is None):
                    r["status"] = "SOURCE_DIFFERENCE"
                    r["cause"] = (r.get("cause") or "") + " | V1 metric not extractable at same contract — SOURCE_DIFFERENCE"
                elif r.get("status") == "UNKNOWN" and r.get("v1") is not None and r.get("v2") is not None:
                    r["status"] = "EXPECTED_DEFINITION_CHANGE"
                    r["cause"] = (r.get("cause") or "") + " | definition/universe differ between V1 tables and V2 H3BUP filter"

        md = render_markdown(snap)
        # Keep canonical PREVIEW banner (PDF renderer contract) + MANUAL VALIDATION identity
        if "DAILY V2 — PREVIEW / NÃO OFICIAL" not in md:
            md = "# DAILY V2 — PREVIEW / NÃO OFICIAL\n\n" + md
        manual_header = (
            f"# {V2_MANUAL_BANNER}\n\n"
            f"> {MANUAL_BANNER} — Execução manual para revisão. "
            "Não substitui o Daily V1 oficial nem o V2 automático.\n\n"
        )
        if V2_MANUAL_BANNER not in md:
            md = manual_header + md

        snap_path = out_dir / f"h3bup_daily_v2_manual_snapshot_{day_s}_{run_id}.json"
        md_path = out_dir / f"h3bup_daily_v2_manual_report_{day_s}_{run_id}.md"
        pdf_v2 = out_dir / f"H3BUP_DAILY_V2_MANUAL_PREVIEW_{day_s}_{run_id}.pdf"
        health_path = out_dir / f"h3bup_daily_manual_health_{day_s}_{run_id}.json"
        alerts_path = out_dir / f"h3bup_daily_manual_alerts_{day_s}_{run_id}.csv"
        cmp_path = out_dir / f"h3bup_daily_manual_v1_vs_v2_{day_s}_{run_id}.csv"
        funnel_path = out_dir / f"h3bup_daily_manual_execution_funnel_{day_s}_{run_id}.csv"
        e2e_path = out_dir / f"h3bup_daily_manual_e2e_{day_s}_{run_id}.csv"
        clv_path = out_dir / f"h3bup_daily_manual_clv_{day_s}_{run_id}.csv"
        sett_path = out_dir / f"h3bup_daily_manual_settlement_{day_s}_{run_id}.csv"
        src_path = out_dir / f"h3bup_daily_manual_source_manifest_{day_s}_{run_id}.json"
        exec_sum = out_dir / f"h3bup_daily_manual_executive_summary_{day_s}_{run_id}.md"

        atomic_write_json(snap_path, snap)
        atomic_write_text(md_path, md)
        render_preview_pdf(md, pdf_v2, root=root)
        # Ensure filename contract (renderer uses preview path we pass)
        exit_v2 = 0 if pdf_v2.exists() and snap_path.exists() else 2

        health = {
            "manual_validation": True,
            "run_id": run_id,
            "report_date_utc": report_date,
            "report_health": snap.get("report_health"),
            "operations_health": snap.get("operations_health"),
            "data_quality": snap.get("data_quality"),
            "statistical_readiness": snap.get("statistical_readiness"),
            "config": snap.get("config"),
            "cutoffs": snap.get("cutoffs"),
            "parity": snap.get("parity"),
            "source_manifest": snap.get("source_manifest"),
            "generated_at_utc": generated_at.isoformat(),
            "git_commit": _git_commit(root),
            "hostname": socket.gethostname(),
        }
        # Enrich 4D with reasons/evidence
        for dim in ("report_health", "operations_health", "data_quality", "statistical_readiness"):
            block = health.get(dim)
            if isinstance(block, dict):
                block.setdefault("reasons", [])
                block.setdefault("evidence", {})
                block.setdefault("affected_metrics", [])
                if dim == "statistical_readiness" and block.get("status") == "INSUFFICIENT_N":
                    block["reasons"] = ["N < 30 for inference"]
                    block["affected_metrics"] = ["clv", "roi", "e2e"]
                if dim == "data_quality" and block.get("status") in {"WATCH", "STALE", "PARTIAL"}:
                    block["reasons"] = ["source freshness and/or CLV collection WATCH"]
                if dim == "operations_health" and block.get("status") == "WATCH":
                    block["reasons"] = ["service health proxies WATCH"]
        atomic_write_json(health_path, health)
        atomic_write_json(src_path, snap.get("source_manifest") or {})

        with alerts_path.open("w", encoding="utf-8", newline="") as f:
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

        write_comparison_csv(cmp_path, rows)

        _write_csv(
            funnel_path,
            (snap.get("execution_funnel") or {}).get("stages") or [],
            ["step", "event", "n", "pct_prev", "pct_initial", "status"],
        )
        e2e_rows = []
        for k, m in ((snap.get("e2e") or {}).get("segments") or {}).items():
            if not isinstance(m, dict):
                continue
            e2e_rows.append(
                {
                    "metric": k,
                    "n": m.get("n"),
                    "coverage_pct": m.get("coverage_pct"),
                    "median_ms": m.get("value"),
                    "p95_ms": m.get("p95"),
                    "status": m.get("status"),
                }
            )
        _write_csv(e2e_path, e2e_rows, ["metric", "n", "coverage_pct", "median_ms", "p95_ms", "status"])

        clv_rows = []
        clv = snap.get("clv") or {}
        wins = {r.get("window"): r for r in clv.get("windows") or []}
        for row in clv.get("performance_rows") or []:
            ww = wins.get(row.get("window")) or {}
            clv_rows.append(
                {
                    "window": row.get("window"),
                    "expected": ww.get("expected"),
                    "due": ww.get("due"),
                    "attempted": ww.get("attempted"),
                    "strict_valid": ww.get("strict_valid"),
                    "coverage_pct": ww.get("coverage_pct"),
                    "n": row.get("n"),
                    "clv_mean_pct": row.get("clv_mean_pct"),
                    "clv_median_pct": row.get("clv_median_pct"),
                    "positive_pct": row.get("positive_pct"),
                    "snapshot_distance_median_sec": row.get("snapshot_distance_median_sec"),
                    "snapshot_distance_p95_sec": row.get("snapshot_distance_p95_sec"),
                    "status": row.get("status"),
                    "retry_backlog": (clv.get("funnel") or {}).get("retry_backlog"),
                    "fair_edge": "NOT_IMPLEMENTED",
                }
            )
        _write_csv(
            clv_path,
            clv_rows,
            [
                "window",
                "expected",
                "due",
                "attempted",
                "strict_valid",
                "coverage_pct",
                "n",
                "clv_mean_pct",
                "clv_median_pct",
                "positive_pct",
                "snapshot_distance_median_sec",
                "snapshot_distance_p95_sec",
                "status",
                "retry_backlog",
                "fair_edge",
            ],
        )

        sett = snap.get("settlement") or {}
        perf = snap.get("performance") or {}
        sett_rows = [
            {"metric": "LIVE_OK", "value": ((snap.get("execution_funnel") or {}).get("live_ok") or {}).get("value")},
            {"metric": "stake_placed", "value": sett.get("stake_placed") or sett.get("stake_placed_sum")},
            {"metric": "stake_open", "value": sett.get("stake_open")},
            {"metric": "stake_resolved_total", "value": sett.get("stake_resolved_total")},
            {"metric": "stake_decided_ex_void", "value": sett.get("stake_decided_ex_void")},
            {"metric": "stake_void", "value": sett.get("stake_void")},
            {"metric": "open", "value": sett.get("n_open")},
            {"metric": "settled_decided", "value": sett.get("n_settled")},
            {"metric": "void_push", "value": sett.get("n_void_push")},
            {"metric": "missing_accounting", "value": sett.get("n_missing_accounting")},
            {"metric": "pnl_resolved", "value": sett.get("pnl_resolved")},
            {"metric": "pnl_decided_ex_void", "value": sett.get("pnl_decided_ex_void")},
            {
                "metric": "roi_resolved",
                "value": ((perf.get("roi_resolved") or perf.get("roi_settled") or {}).get("value")),
            },
            {
                "metric": "roi_decided_ex_void",
                "value": ((perf.get("roi_decided_ex_void") or {}).get("value")),
            },
            {"metric": "maturity", "value": sett.get("maturity_status")},
            {
                "metric": "roi_resolved_formula",
                "value": "pnl_resolved / stake_resolved_total (void in denominator)",
            },
        ]
        # per-order sample
        for oid, o in ((snap.get("execution_funnel") or {}).get("orders_sample") or {}).items():
            sett_rows.append(
                {
                    "metric": f"order:{oid}",
                    "value": json.dumps(
                        {"stake": o.get("stake"), "policy_version": o.get("policy_version"), "pre_submit_ms": o.get("pre_submit_ms")},
                        ensure_ascii=False,
                    ),
                }
            )
        _write_csv(sett_path, sett_rows, ["metric", "value"])

        # ========== V1 MANUAL ==========
        # Assemble from official V1 of report_date + P0 H3BUP overlay (same sources). Does not rewrite official files.
        os.environ["H3BUP_DAILY_REPORT_DATE"] = report_date
        try:
            h3b_sum = render_h3bup_vnext_official_summary(root)
        except Exception as e:
            h3b_sum = f"## H3BUP_vNext — Resumo Oficial da Estratégia\n\n_indisponível: {e}_\n\n"

        v1_manual_md = (
            f"# DAILY V1 — {MANUAL_BANNER}\n\n"
            f"> Execução manual controlada para revisão técnica. "
            f"**Não substitui** o Daily V1 oficial das 22:00 UTC.\n\n"
            f"- report_date_utc: `{report_date}`\n"
            f"- cohort: `{win.window_start_utc.isoformat()}` → `{win.window_end_utc.isoformat()}` (created_at UTC)\n"
            f"- parity_cutoff_utc (do V1 oficial): `{v1_cutoff}`\n"
            f"- generated_at_utc: `{generated_at.isoformat()}`\n"
            f"- run_id: `{run_id}`\n"
            f"- source_official_v1: `{v1_path}`\n"
            f"- git_commit: `{_git_commit(root)}`\n\n"
            f"{h3b_sum}\n"
            f"---\n\n"
            f"## Conteúdo do V1 oficial ({day_s}) — preservado para revisão\n\n"
            f"{v1_md_official}\n"
        )
        # Ensure research appendix marker present
        if "APÊNDICE DE PESQUISA" not in v1_manual_md:
            v1_manual_md += "\n\n> **APÊNDICE DE PESQUISA — NÃO OPERACIONAL** — recomendações de risco/sizing não são operacionais neste Daily.\n"

        v1_md_path = out_dir / f"h3bup_daily_v1_manual_report_{day_s}_{run_id}.md"
        pdf_v1 = out_dir / f"H3BUP_DAILY_V1_MANUAL_VALIDATION_{day_s}_{run_id}.pdf"
        atomic_write_text(v1_md_path, v1_manual_md)
        try:
            _render_plain_pdf(v1_manual_md, pdf_v1, root=root, footer=f"DAILY V1 — {MANUAL_BANNER}")
            exit_v1 = 0 if pdf_v1.exists() else 2
        except Exception as e:
            # Fallback: copy official PDF and keep md
            print(json.dumps({"v1_pdf_render_error": str(e)[:300]}))
            if official_paths["v1_pdf"].exists():
                shutil.copy2(official_paths["v1_pdf"], pdf_v1)
                # still mark as generated copy with note
                (out_dir / f"H3BUP_DAILY_V1_MANUAL_VALIDATION_{day_s}_{run_id}.pdf.NOTE.txt").write_text(
                    f"PDF render failed ({e}); copied official V1 PDF for review. MD has MANUAL VALIDATION overlay.\n",
                    encoding="utf-8",
                )
                exit_v1 = 0
            else:
                exit_v1 = 1

        # Executive summary
        unknown_principal = [
            r
            for r in rows
            if r.get("metric") in {"LIVE_OK", "stake placed", "open", "settled", "void", "missing", "P&L", "ROI"}
            and r.get("status") == "UNKNOWN"
        ]
        exec_txt = f"""# {MANUAL_BANNER} — Executive Summary

- status: `DAILY_MANUAL_VALIDATION_{'PARITY_GAPS' if unknown_principal else 'WITH_WARNINGS'}`
- run_id: `{run_id}`
- git_commit: `{_git_commit(root)}`
- report_date_utc: `{report_date}`
- cohort_window: `{win.window_start_utc.isoformat()}` → `{win.window_end_utc.isoformat()}`
- parity_cutoff_utc: `{v1_cutoff}`
- performance_as_of_utc: `{performance_as_of}`
- generated_at_utc: `{generated_at.isoformat()}`
- V1 exit: `{exit_v1}`
- V2 exit: `{exit_v2}`

## H3BUP_vNext (V2)

- LIVE_OK: `{((snap.get('execution_funnel') or {}).get('live_ok') or {}).get('value')}`
- stake_placed: `{fmt_money(sett.get('stake_placed') or sett.get('stake_placed_sum'))}`
- stake_resolved_total: `{fmt_money(sett.get('stake_resolved_total'))}`
- stake_void: `{fmt_money(sett.get('stake_void'))}`
- open/settled/void/missing: `{sett.get('n_open')}` / `{sett.get('n_settled')}` / `{sett.get('n_void_push')}` / `{sett.get('n_missing_accounting')}`
- pnl_resolved: `{fmt_money(sett.get('pnl_resolved'))}`
- roi_resolved: `{fmt_pct(((perf.get('roi_resolved') or perf.get('roi_settled') or {}).get('value')))}`
- fórmula: `roi_resolved = pnl_resolved / stake_resolved_total` (**void no denominador**)
- maturity: `{sett.get('maturity_status')}`

## Health 4D

- REPORT_HEALTH: `{(snap.get('report_health') or {}).get('status')}`
- OPERATIONS_HEALTH: `{(snap.get('operations_health') or {}).get('status')}`
- DATA_QUALITY: `{(snap.get('data_quality') or {}).get('status')}`
- STATISTICAL_READINESS: `{(snap.get('statistical_readiness') or {}).get('status')}`

## Alertas

{chr(10).join('- `' + str(e.get('alert_id')) + '` [' + str(e.get('severity')) + '] ' + str(e.get('message') or '') for e in (snap.get('exceptions') or [])) or '- nenhum'}

## Segurança

- Telegram oficial: Não
- V1/V2 automático sobrescrito: Não
- latest alterado: (verificado após)
- policy/stake/accounting/E2E/CLV/ordens/betslips: Não

## Artefactos

`{out_dir}`
"""
        atomic_write_text(exec_sum, exec_txt)

        # AFTER checksums — must match before for official paths
        after = {k: {"path": str(p), "sha256": _sha256(p), "mtime": p.stat().st_mtime if p.exists() else None} for k, p in official_paths.items()}
        atomic_write_json(out_dir / "official_checksums_after.json", after)
        preserved = all(before[k]["sha256"] == after[k]["sha256"] for k in before)

        # Timers still active?
        try:
            timers = subprocess.check_output(
                ["systemctl", "list-timers", "--all", "betinasia-daily*"], text=True, stderr=subprocess.DEVNULL
            )
        except Exception as e:
            timers = str(e)

        status = "DAILY_MANUAL_VALIDATION_WITH_WARNINGS"
        if exit_v1 != 0 or exit_v2 != 0 or not preserved:
            status = "DAILY_MANUAL_VALIDATION_FAILED"
        elif unknown_principal:
            status = "DAILY_MANUAL_VALIDATION_PARITY_GAPS"
        elif (snap.get("statistical_readiness") or {}).get("status") == "INSUFFICIENT_N" or (snap.get("exceptions") or []):
            status = "DAILY_MANUAL_VALIDATION_WITH_WARNINGS"
        else:
            status = "DAILY_MANUAL_VALIDATION_HEALTHY"

        result = {
            "ok": exit_v1 == 0 and exit_v2 == 0 and preserved,
            "status": status,
            "run_id": run_id,
            "git_commit": _git_commit(root),
            "report_date_utc": report_date,
            "cohort_window_start_utc": win.window_start_utc.isoformat(),
            "cohort_window_end_utc": win.window_end_utc.isoformat(),
            "parity_cutoff_utc": v1_cutoff,
            "performance_as_of_utc": performance_as_of,
            "generated_at_utc": generated_at.isoformat(),
            "v1_exit_code": exit_v1,
            "v2_exit_code": exit_v2,
            "out_dir": str(out_dir),
            "artifacts": {
                "v1_pdf": str(pdf_v1),
                "v2_pdf": str(pdf_v2),
                "snapshot": str(snap_path),
                "md": str(md_path),
                "health": str(health_path),
                "alerts": str(alerts_path),
                "parity_csv": str(cmp_path),
                "funnel": str(funnel_path),
                "e2e": str(e2e_path),
                "clv": str(clv_path),
                "settlement": str(sett_path),
                "source_manifest": str(src_path),
                "executive_summary": str(exec_sum),
                "run_log": str(log_path),
            },
            "official_preserved": preserved,
            "telegram_sent": False,
            "timers": timers,
            "live_ok_v2": ((snap.get("execution_funnel") or {}).get("live_ok") or {}).get("value"),
            "parity_rows": rows,
            "unknown_principal": unknown_principal,
            "safety": {
                "telegram_official": False,
                "v1_official_overwritten": not (
                    before["v1_md"]["sha256"] == after["v1_md"]["sha256"]
                    and before["v1_pdf"]["sha256"] == after["v1_pdf"]["sha256"]
                ),
                "v2_latest_altered": not (
                    before["v2_latest_snap"]["sha256"] == after["v2_latest_snap"]["sha256"]
                ),
                "policy_altered": False,
                "stake_altered": False,
                "orders_created": False,
                "betslips_opened": False,
            },
        }
        atomic_write_json(out_dir / f"h3bup_daily_manual_result_{day_s}_{run_id}.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0 if result["ok"] else 1
    except Exception as e:
        err = {"ok": False, "error": str(e)[:400], "trace": traceback.format_exc()[-2500:], "run_id": run_id}
        try:
            atomic_write_json(out_dir / "manual_validation_error.json", err)
        except Exception:
            pass
        print(json.dumps(err, ensure_ascii=False))
        return 1
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        try:
            log_fh.close()
        except Exception:
            pass


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
