#!/usr/bin/env python3
"""Controlled parity-hardening run — isolated outputs, no Telegram/latest overwrite."""

from __future__ import annotations

import argparse
import csv
import json
import socket
import subprocess
import sys
import traceback
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ops.daily_v2.cutoff import find_v1_report_md, resolve_parity_cutoffs
from ops.daily_v2.formatters import fmt_money, fmt_pct
from ops.daily_v2.io_atomic import atomic_write_json, atomic_write_text
from ops.daily_v2.parity_hardening import PARITY_CUTOFF_20260729, build_parity_hardening_bundle
from ops.daily_v2.pdf_preview import render_preview_pdf
from ops.daily_v2.time_windows import resolve_window


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> Optional[str]:
    import hashlib

    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def _render_markdown(bundle: Dict[str, Any], *, run_id: str) -> str:
    u = bundle["universe"]
    pv = bundle["parity_view"]
    mv = bundle["matured_view"]
    pp = pv["performance"]
    mp = mv["performance"]
    lines: List[str] = []
    a = lines.append
    a("# DAILY V2 — PARITY HARDENING / PREVIEW / NÃO OFICIAL\n\n")
    a("> MANUAL / reporting-only. Não substitui V1 oficial nem V2 automático.\n\n")
    a("## 0) Manifesto\n")
    a(f"- status: `{bundle.get('status')}`\n")
    a(f"- report_date_utc: `{bundle.get('report_date_utc')}`\n")
    a(f"- cohort_window: `{bundle.get('cohort_window_start_utc')}` → `{bundle.get('cohort_window_end_utc')}`\n")
    a(f"- parity_as_of_utc: `{bundle.get('parity_as_of_utc')}`\n")
    a(f"- matured_as_of_utc: `{bundle.get('matured_as_of_utc')}`\n")
    a(f"- generated_at_utc: `{datetime.now(timezone.utc).isoformat()}`\n")
    a(f"- run_id: `{run_id}`\n")
    a(f"- historical_asof_status: `{bundle.get('historical_asof_status')}`\n\n")

    a("## 1) Health / order-set\n")
    a(f"- V1 parity hash: `{u['v1_parity']['hash']}`\n")
    a(f"- V2 parity hash: `{u['v2_parity']['hash']}`\n")
    a(f"- order_set_match (parity): `{u['diff_parity_vs_v1']['order_set_match']}`\n")
    a(f"- only_in_v2 (full day): `{u['diff_full_day_vs_v1']['only_in_v2']}`\n")
    a(f"- only_in_v1: `{u['diff_full_day_vs_v1']['only_in_v1']}`\n\n")

    a("## 2) Resumo executivo (matured)\n")
    a(f"- LIVE_OK full day: `{u['v2_full_day']['count']}` stake `{fmt_money(u['v2_full_day']['stake_placed'])}`\n")
    a(f"- open/settled/void/missing: `{mv['counts']['open']}` / `{mv['counts']['settled']}` / `{mv['counts']['void']}` / `{mv['counts']['missing']}`\n")
    a(f"- stake_resolved: `{fmt_money(mp.get('stake_resolved_total'))}`\n")
    a(f"- pnl_resolved: `{fmt_money(mp.get('pnl_resolved'))}`\n")
    a(f"- roi_resolved: `{fmt_pct((mp.get('roi_resolved') or {}).get('value'))}`\n")
    a(f"- roi_decided_ex_void: `{fmt_pct((mp.get('roi_decided_ex_void') or {}).get('value'))}`\n")
    a("- void no denominador de roi_resolved: **sim**\n\n")

    a("## 3) Paridade com Daily V1 — visão congelada\n")
    a("| Métrica | V1 | V2 parity | Delta | Status |\n|---|---:|---:|---:|---|\n")
    v1_n = u["v1_parity"]["count"]
    v2p_n = u["v2_parity"]["count"]
    a(f"| LIVE_OK | {v1_n} | {v2p_n} | {v2p_n-v1_n} | `{'MATCH' if v1_n==v2p_n else 'DIFF'}` |\n")
    a(f"| order_id set hash | `{u['v1_parity']['hash'][:12]}…` | `{u['v2_parity']['hash'][:12]}…` | — | `{'MATCH' if u['diff_parity_vs_v1']['order_set_match'] else 'DIFF'}` |\n")
    a(f"| stake placed | {fmt_money(u['v1_parity']['stake_placed'])} | {fmt_money(u['v2_parity']['stake_placed'])} | — | `{'MATCH' if abs(u['v1_parity']['stake_placed']-u['v2_parity']['stake_placed'])<1e-9 else 'DIFF'}` |\n")
    a(f"| open as of | — | {pv['counts']['open']} | — | PARITY_AS_OF |\n")
    a(f"| settled as of | — | {pv['counts']['settled']} | — | PARITY_AS_OF |\n")
    a(f"| void as of | — | {pv['counts']['void']} | — | PARITY_AS_OF |\n")
    a(f"| missing as of | — | {pv['counts']['missing']} | — | PARITY_AS_OF |\n")
    a(f"| stake resolved as of | — | {fmt_money(pp.get('stake_resolved_total'))} | — | PARITY_AS_OF |\n")
    a(f"| P&L as of | — | {fmt_money(pp.get('pnl_resolved'))} | — | PARITY_AS_OF |\n")
    a(f"| ROI resolved as of | — | {fmt_pct((pp.get('roi_resolved') or {}).get('value'))} | — | PARITY_AS_OF |\n\n")

    a("## 4) Atualização de maturity da coorte\n")
    a(f"> {mv['warning']}\n\n")
    a("| Métrica | Parity as of | Matured as of | Delta |\n|---|---:|---:|---:|\n")
    for key, label in [
        ("open", "open"),
        ("settled", "settled"),
        ("void", "void"),
        ("missing", "missing"),
    ]:
        left, right = pv["counts"][key], mv["counts"][key]
        lines.append(f"| {label} | {left} | {right} | {right-left} |\n")
    a(f"| stake resolved | {fmt_money(pp.get('stake_resolved_total'))} | {fmt_money(mp.get('stake_resolved_total'))} | — |\n")
    a(f"| P&L resolved | {fmt_money(pp.get('pnl_resolved'))} | {fmt_money(mp.get('pnl_resolved'))} | — |\n")
    a(f"| ROI resolved | {fmt_pct((pp.get('roi_resolved') or {}).get('value'))} | {fmt_pct((mp.get('roi_resolved') or {}).get('value'))} | — |\n\n")

    a("## 5) Divergências explicadas\n")
    a("| ID | Métrica | Root cause | Classificação | Blocker |\n|---|---|---|---|---|\n")
    for r in bundle.get("root_causes") or []:
        a(f"| {r.get('ID')} | {r.get('metric')} | {r.get('root_cause')} | `{r.get('classification')}` | {r.get('publication_blocker')} |\n")
    a("\n")
    for d in u.get("divergent_orders") or []:
        a(f"- order `{d.get('order_id')}` created `{d.get('created_at_utc')}` stake `{d.get('stake')}` → `{d.get('classification')}`: {d.get('root_cause')}\n")
    a("\n")

    a("## 6) Metodologia as-of\n")
    a("- cohort: created_at UTC half-open day\n")
    a("- parity universe: H3BUP_vNext Back LIVE_OK with created_at <= parity_as_of\n")
    a("- matured universe: full closed day; settlement from latest accounting\n")
    a("- roi_resolved = pnl_resolved / stake_resolved_total (void **entra** no denominador)\n")
    a("- fair_edge: NOT_IMPLEMENTED\n")
    a("---\n**PREVIEW / NÃO OFICIAL**\n")
    return "".join(lines)


def run(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--report-date", default="2026-07-29")
    ap.add_argument("--parity-as-of", default=PARITY_CUTOFF_20260729)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    report_date = args.report_date
    day_s = report_date.replace("-", "")
    run_id = args.run_id or uuid.uuid4().hex[:12]
    generated_at = datetime.now(timezone.utc)
    parity_as_of = datetime.fromisoformat(args.parity_as_of.replace("Z", "+00:00"))
    matured_as_of = generated_at

    out_dir = root / "logs" / "daily_p0" / "parity_hardening" / day_s / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "h3bup_parity_execution.log"

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

    try:
        official = {
            "v1_md": root / "logs" / "daily_reports" / day_s / "report_daily.md",
            "v1_pdf": root / "logs" / "daily_reports" / day_s / "report_daily.pdf",
            "v2_latest": root / "logs" / "daily_v2" / "latest_snapshot.json",
        }
        before = {k: _sha256(p) for k, p in official.items()}

        # Confirm cutoff from official V1
        v1_path = find_v1_report_md(root, report_date) or official["v1_md"]
        parity_meta = resolve_parity_cutoffs(
            root=root,
            report_date_utc=report_date,
            v2_generated_at=generated_at,
            v1_md_path=Path(v1_path) if v1_path else None,
        )
        extracted = parity_meta.get("v1_report_cutoff_utc")
        if extracted:
            parity_as_of = datetime.fromisoformat(str(extracted).replace("Z", "+00:00"))

        win = resolve_window(report_type="DAILY_CLOSED", report_date=date.fromisoformat(report_date))
        v1_frozen = {"n_open": 9, "n_settled": 3, "live_ok_health": 12, "live_ok_day_table": 22, "stake_day_table": 220.0}

        bundle = build_parity_hardening_bundle(
            root=root,
            window=win,
            parity_as_of=parity_as_of,
            matured_as_of=matured_as_of,
            v1_frozen_metrics=v1_frozen,
        )
        bundle["generated_at_utc"] = generated_at.isoformat()
        bundle["run_id"] = run_id
        bundle["git_commit"] = _git_commit(root)
        bundle["hostname"] = socket.gethostname()
        bundle["official_status"] = "PREVIEW_NOT_OFFICIAL"
        bundle["manual_validation"] = False
        bundle["parity_hardening"] = True

        u = bundle["universe"]
        # CSVs
        v1_rows = [{"order_id": oid} for oid in u["v1_parity"]["order_ids"]]
        v2_rows = [{"order_id": oid} for oid in u["v2_full_day"]["order_ids"]]
        _write_csv(out_dir / "h3bup_v1_order_ids.csv", v1_rows, ["order_id"])
        _write_csv(out_dir / "h3bup_v2_order_ids.csv", v2_rows, ["order_id"])
        _write_csv(
            out_dir / "h3bup_order_id_set_diff.csv",
            u.get("divergent_orders") or [],
            [
                "order_id",
                "created_at_utc",
                "policy_version",
                "stake",
                "side",
                "status",
                "audit_id",
                "V1_included",
                "V2_full_day_included",
                "V2_parity_included",
                "filtro_V1",
                "filtro_V2",
                "classification",
                "root_cause",
                "after_parity_cutoff",
            ],
        )
        atomic_write_json(
            out_dir / "h3bup_order_id_set_hashes.json",
            {
                "v1_order_set_hash": u["v1_parity"]["hash"],
                "v2_parity_order_set_hash": u["v2_parity"]["hash"],
                "v2_full_day_order_set_hash": u["v2_full_day"]["hash"],
                "order_set_match_parity": u["diff_parity_vs_v1"]["order_set_match"],
                "only_in_v1_count": u["diff_full_day_vs_v1"]["only_in_v1_count"],
                "only_in_v2_count": u["diff_full_day_vs_v1"]["only_in_v2_count"],
                "only_in_v2": u["diff_full_day_vs_v1"]["only_in_v2"],
            },
        )
        _write_csv(out_dir / "h3bup_parity_asof_orders.csv", bundle["parity_view"]["orders"], [
            "order_id", "created_at_utc", "policy_version", "stake", "status_as_of", "pnl_as_of", "audit_id"
        ])
        _write_csv(out_dir / "h3bup_matured_asof_orders.csv", bundle["matured_view"]["orders"], [
            "order_id", "created_at_utc", "policy_version", "stake", "status_as_of", "pnl_as_of", "audit_id"
        ])

        # parity comparison table
        parity_rows = [
            {"metric": "LIVE_OK", "v1": u["v1_parity"]["count"], "v2_parity": u["v2_parity"]["count"], "v2_full": u["v2_full_day"]["count"],
             "status": "MATCH" if u["diff_parity_vs_v1"]["order_set_match"] else "DIFF",
             "cause": "parity filters created_at<=parity_as_of"},
            {"metric": "stake_placed", "v1": u["v1_parity"]["stake_placed"], "v2_parity": u["v2_parity"]["stake_placed"], "v2_full": u["v2_full_day"]["stake_placed"],
             "status": "MATCH" if abs(u["v1_parity"]["stake_placed"] - u["v2_parity"]["stake_placed"]) < 1e-9 else "DIFF",
             "cause": "same as LIVE_OK set"},
            {"metric": "open", "v1": v1_frozen.get("n_open"), "v2_parity": bundle["parity_view"]["counts"]["open"], "v2_full": bundle["matured_view"]["counts"]["open"],
             "status": "AS_OF_MATURITY_DIFFERENCE", "cause": "V1 health subset + maturity"},
            {"metric": "settled", "v1": v1_frozen.get("n_settled"), "v2_parity": bundle["parity_view"]["counts"]["settled"], "v2_full": bundle["matured_view"]["counts"]["settled"],
             "status": "AS_OF_MATURITY_DIFFERENCE", "cause": "settlements after cutoff"},
            {"metric": "order_set_hash", "v1": u["v1_parity"]["hash"], "v2_parity": u["v2_parity"]["hash"], "v2_full": u["v2_full_day"]["hash"],
             "status": "MATCH" if u["diff_parity_vs_v1"]["order_set_match"] else "DIFF", "cause": "SHA256 sorted order_ids"},
        ]
        _write_csv(out_dir / "h3bup_parity_v1_vs_v2.csv", parity_rows, ["metric", "v1", "v2_parity", "v2_full", "status", "cause"])

        mat_delta = []
        for k in ("open", "settled", "void", "missing"):
            a = bundle["parity_view"]["counts"][k]
            b = bundle["matured_view"]["counts"][k]
            mat_delta.append({"metric": k, "parity_as_of": a, "matured_as_of": b, "delta": b - a})
        pp, mp = bundle["parity_view"]["performance"], bundle["matured_view"]["performance"]
        mat_delta.extend([
            {"metric": "stake_resolved", "parity_as_of": pp.get("stake_resolved_total"), "matured_as_of": mp.get("stake_resolved_total"), "delta": None},
            {"metric": "pnl_resolved", "parity_as_of": pp.get("pnl_resolved"), "matured_as_of": mp.get("pnl_resolved"), "delta": None},
            {"metric": "roi_resolved", "parity_as_of": (pp.get("roi_resolved") or {}).get("value"), "matured_as_of": (mp.get("roi_resolved") or {}).get("value"), "delta": None},
        ])
        _write_csv(out_dir / "h3bup_maturity_delta.csv", mat_delta, ["metric", "parity_as_of", "matured_as_of", "delta"])
        _write_csv(out_dir / "h3bup_parity_root_causes.csv", bundle["root_causes"], [
            "ID", "metric", "symptom", "affected_order_ids", "root_cause", "correct_behaviour", "classification", "patch_applied", "publication_blocker", "owner"
        ])

        health = {
            "status": bundle["status"],
            "report_health": {"status": "HEALTHY", "reasons": ["artifacts written atomically to isolated dir"]},
            "operations_health": {"status": "WATCH", "reasons": ["unchanged ops; reporting only"]},
            "data_quality": {"status": "WATCH" if bundle["historical_asof_status"] != "AVAILABLE" else "HEALTHY"},
            "statistical_readiness": {"status": "INSUFFICIENT_N"},
            "order_set_match_parity": u["diff_parity_vs_v1"]["order_set_match"],
            "unknown_divergent": bundle.get("unknown_divergent"),
            "cutoffs": {
                "parity_as_of_utc": bundle["parity_as_of_utc"],
                "matured_as_of_utc": bundle["matured_as_of_utc"],
                "cohort_window_start_utc": bundle["cohort_window_start_utc"],
                "cohort_window_end_utc": bundle["cohort_window_end_utc"],
            },
        }
        atomic_write_json(out_dir / "h3bup_parity_health.json", health)
        alerts = []
        if not u["diff_parity_vs_v1"]["order_set_match"]:
            alerts.append({"alert_id": "PARITY_ORDER_SET_MISMATCH", "severity": "CRITICAL", "message": "parity hashes differ"})
        if u["diff_full_day_vs_v1"]["only_in_v2_count"]:
            alerts.append({
                "alert_id": "POST_CUTOFF_LIVE_OK_IN_FULL_DAY",
                "severity": "INFO",
                "message": f"full-day includes {u['diff_full_day_vs_v1']['only_in_v2_count']} post-cutoff LIVE_OK",
                "evidence": u["diff_full_day_vs_v1"]["only_in_v2"],
            })
        alerts.append({"alert_id": "AS_OF_MATURITY_DELTA", "severity": "INFO", "message": "matured settlements differ from parity as-of"})
        _write_csv(out_dir / "h3bup_parity_alerts.csv", alerts, ["alert_id", "severity", "message", "evidence"])

        snap_path = out_dir / f"h3bup_daily_v2_parity_snapshot_{day_s}_{run_id}.json"
        atomic_write_json(snap_path, bundle)
        md = _render_markdown(bundle, run_id=run_id)
        # keep PREVIEW substring for pdf renderer contract
        if "PREVIEW / NÃO OFICIAL" not in md:
            md = "# DAILY V2 — PREVIEW / NÃO OFICIAL\n\n" + md
        md_path = out_dir / f"h3bup_daily_v2_parity_report_{day_s}_{run_id}.md"
        atomic_write_text(md_path, md)
        pdf_path = out_dir / f"H3BUP_DAILY_V2_PARITY_HARDENING_PREVIEW_{day_s}_{run_id}.pdf"
        render_preview_pdf(md, pdf_path, root=root)

        after = {k: _sha256(p) for k, p in official.items()}
        preserved = before == after

        exec_sum = f"""# Parity Hardening — Executive Summary

**Status:** `{bundle['status']}`

## Universo
- only_in_v2 (full day): `{u['diff_full_day_vs_v1']['only_in_v2']}`
- classifications: EXPECTED_SCOPE_DIFFERENCE (post-cutoff LIVE_OK)
- parity hashes match: `{u['diff_parity_vs_v1']['order_set_match']}`
- V1/V2 parity LIVE_OK: `{u['v1_parity']['count']}` / `{u['v2_parity']['count']}`
- V2 full-day LIVE_OK: `{u['v2_full_day']['count']}`

## As-of
- parity_as_of: `{bundle['parity_as_of_utc']}`
- matured_as_of: `{bundle['matured_as_of_utc']}`
- parity open/settled/void/missing: `{bundle['parity_view']['counts']}`
- matured open/settled/void/missing: `{bundle['matured_view']['counts']}`

## Performance
- parity stake_resolved / pnl / roi: `{pp.get('stake_resolved_total')}` / `{pp.get('pnl_resolved')}` / `{(pp.get('roi_resolved') or {}).get('value')}`
- matured stake_resolved / pnl / roi: `{mp.get('stake_resolved_total')}` / `{mp.get('pnl_resolved')}` / `{(mp.get('roi_resolved') or {}).get('value')}`
- void in roi_resolved denominator: **yes**

## Segurança
Official preserved: `{preserved}` · Telegram: Não · V2 official: Não · policy/stake/orders: Não

## Dir
`{out_dir}`
"""
        atomic_write_text(out_dir / "h3bup_parity_executive_summary.md", exec_sum)
        result = {
            "ok": bool(preserved and not bundle.get("unknown_divergent") and u["diff_parity_vs_v1"]["order_set_match"]),
            "status": bundle["status"],
            "run_id": run_id,
            "out_dir": str(out_dir),
            "parity_as_of_utc": bundle["parity_as_of_utc"],
            "matured_as_of_utc": bundle["matured_as_of_utc"],
            "only_in_v2": u["diff_full_day_vs_v1"]["only_in_v2"],
            "order_set_match_parity": u["diff_parity_vs_v1"]["order_set_match"],
            "official_preserved": preserved,
            "pdf": str(pdf_path),
        }
        atomic_write_json(out_dir / "h3bup_parity_result.json", result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["ok"] else 1
    except Exception as e:
        err = {"ok": False, "error": str(e)[:400], "trace": traceback.format_exc()[-2000:]}
        print(json.dumps(err, ensure_ascii=False))
        atomic_write_json(out_dir / "error.json", err)
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
