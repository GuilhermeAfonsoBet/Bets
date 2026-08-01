"""V1 P0 helper: isolated H3BUP_vNext official summary (read-only reporting)."""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .clv_section import build_clv_section
from .formatters import fmt_money, fmt_pct, fmt_ts
from .friendly_section import build_friendly_section, render_friendly_markdown
from .performance import compute_settlement_and_performance
from .time_windows import resolve_window
from .universes import load_executor_orders, load_open_order_ids, load_pnl_by_order_from_balance_csv


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def render_h3bup_vnext_official_summary(root: Path | None = None) -> str:
    root = Path(root or os.getcwd())
    # Prefer previous UTC day for closed daily (same as V1 default when run at 22:00)
    day = date.today()
    # If run after midnight UTC for "yesterday" report, env may set REPORT_DATE
    env_day = os.getenv("H3BUP_DAILY_REPORT_DATE") or os.getenv("DAILY_REPORT_DATE")
    if env_day:
        try:
            day = date.fromisoformat(env_day[:10])
        except Exception:
            pass
    else:
        # closed cohort: typically previous calendar day when generating at 22:00 same day
        # Keep today if mid-day smoke; V1 usually sets its own day — use yesterday if hour>=22 else today-ish
        now = datetime.now(timezone.utc)
        if now.hour >= 22:
            from datetime import timedelta

            day = (now - timedelta(days=0)).date()  # same UTC day folder for evening run
        day = now.date()

    win = resolve_window(report_type="DAILY_CLOSED", report_date=day)
    exec_path = root / os.getenv("EXECUTOR_JSONL", "logs/executor_live.jsonl")
    orders = load_executor_orders(exec_path, window=win, require_h3bup=True)

    acct_dir = root / os.getenv("ACCOUNTING_OUT_DIR", "logs/accounting")
    bal = None
    opens = None
    if acct_dir.exists():
        bals = sorted(acct_dir.glob("*__balance.csv")) + sorted(acct_dir.glob("*balance*.csv"))
        ops = sorted(acct_dir.glob("*__open_stakes.csv")) + sorted(acct_dir.glob("*open_stakes*.csv"))
        bal = bals[-1] if bals else None
        opens = ops[-1] if ops else None
    pnl = load_pnl_by_order_from_balance_csv(bal) if bal else {}
    open_oids = load_open_order_ids(opens) if opens else set()
    acct_h = _load_json(root / os.getenv("ACCOUNTING_HEALTH_JSON", "logs/accounting/accounting_health.json"))
    perf = compute_settlement_and_performance(
        orders=orders,
        pnl_by_oid=pnl,
        open_oids=open_oids,
        accounting_health_status=str(acct_h.get("status") or "NOT_AVAILABLE"),
    )
    clv = build_clv_section(root)
    e2e_health_st = "AVAILABLE" if (root / os.getenv("H3BUP_E2E_TRACE_PATH", "logs/h3bup_e2e_trace.jsonl")).exists() else "MISSING"
    policy_version = os.getenv("H3BUP_POLICY_VERSION", "H3BUP_vNext_20260629")

    n = len(orders)
    cov = None
    if n:
        resolved = int(perf.get("n_settled") or 0) + int(perf.get("n_void_push") or 0)
        cov = 100.0 * resolved / n

    lines: List[str] = []
    a = lines.append
    a("## H3BUP_vNext — Resumo Oficial da Estratégia\n\n")
    a(
        "> Os valores de banca, P&L semanal/mensal da conta e estudos históricos "
        "não representam necessariamente a performance da H3BUP_vNext.\n\n"
    )
    a("| Métrica | Valor | Status |\n|---|---|---|\n")
    a(f"| policy_version | `{policy_version}` | CURRENT |\n")
    a(f"| LIVE_OK da coorte | {n} | AVAILABLE |\n")
    a(f"| stake placed | {fmt_money(perf.get('stake_placed'))} | AVAILABLE |\n")
    a(f"| open | {perf.get('n_open')} | AVAILABLE |\n")
    a(f"| settled decided | {perf.get('n_settled')} | AVAILABLE |\n")
    a(f"| void/push | {perf.get('n_void_push')} | AVAILABLE |\n")
    a(f"| missing accounting | {perf.get('n_missing_accounting')} | {'WATCH' if perf.get('n_missing_accounting') else 'OK'} |\n")
    a(f"| stake resolved total | {fmt_money(perf.get('stake_resolved_total'))} | AVAILABLE |\n")
    a(f"| stake decided ex-void | {fmt_money(perf.get('stake_decided_ex_void'))} | AVAILABLE |\n")
    a(f"| stake void | {fmt_money(perf.get('stake_void'))} | AVAILABLE |\n")
    a(f"| P&L resolved | {fmt_money(perf.get('pnl_resolved'))} | AVAILABLE |\n")
    roi_r = perf.get("roi_resolved") or {}
    a(f"| ROI resolved | {fmt_pct(roi_r.get('value')) if isinstance(roi_r, dict) else '—'} | `{roi_r.get('status') if isinstance(roi_r, dict) else '—'}` |\n")
    roi_x = perf.get("roi_decided_ex_void") or {}
    a(f"| ROI decided ex-void | {fmt_pct(roi_x.get('value')) if isinstance(roi_x, dict) else '—'} | `{roi_x.get('status') if isinstance(roi_x, dict) else '—'}` |\n")
    a(f"| accounting coverage | {('—' if cov is None else f'{cov:.1f}%')} | AVAILABLE |\n")
    a(f"| maturity | `{perf.get('maturity_status')}` | AVAILABLE |\n")
    a(f"| CLV collection status | `{clv.get('collection_status')}` | `{clv.get('collection_status')}` |\n")
    a(f"| E2E status | `{e2e_health_st}` | `{e2e_health_st}` |\n")
    a(f"| data quality | ver Accounting/CLV/E2E | SEPARATED |\n")
    a(f"| statistical readiness | `{'INSUFFICIENT_N' if n < 30 else 'AVAILABLE'}` | `{'INSUFFICIENT_N' if n < 30 else 'AVAILABLE'}` |\n")
    a("\n")

    # CLV performance table (same contract as V2)
    a("### CLV forward (VALID_STRICT) — H3BUP_vNext\n\n")
    a("| Janela | N | CLV médio | Mediana | Positivo % | Status |\n|---|---:|---:|---:|---:|---|\n")
    for row in clv.get("performance_rows") or []:
        a(
            f"| {row.get('window')} | {row.get('n')} | "
            f"{fmt_pct(row.get('clv_mean_pct'), already_percent=True)} | "
            f"{fmt_pct(row.get('clv_median_pct'), already_percent=True)} | "
            f"{fmt_pct(row.get('positive_pct'), already_percent=True)} | `{row.get('status')}` |\n"
        )
    a("\n")
    a("| Janela | Expected | Due | Attempted | Strict valid | Coverage |\n|---|---:|---:|---:|---:|---:|\n")
    for row in clv.get("windows") or []:
        covw = row.get("coverage_pct")
        a(
            f"| {row.get('window')} | {row.get('expected')} | {row.get('due')} | {row.get('attempted')} | "
            f"{row.get('strict_valid')} | {('—' if covw is None else f'{covw:.1f}%')} |\n"
        )
    a("\n")
    a(f"- fair edge: `NOT_IMPLEMENTED`\n")
    a(f"- coorte: `{fmt_ts(win.window_start_utc.isoformat())}` → `{fmt_ts(win.window_end_utc.isoformat())}` (created_at UTC)\n\n")

    try:
        friendly = build_friendly_section(
            root=root,
            orders=orders,
            pnl_by_oid=pnl,
            open_oids=open_oids,
            accounting_health_status=str(acct_h.get("status") or "NOT_AVAILABLE"),
        )
        a(render_friendly_markdown(friendly))
    except Exception as e:
        a(f"### Friendly vs Non-Friendly (diagnóstico / shadow)\n\n> indisponível: `{type(e).__name__}`\n\n")

    a("---\n\n")
    a("### Separação visual de universos\n\n")
    a("1. **H3BUP_vNext** — estratégia oficial deste resumo\n")
    a("2. **CONTA TOTAL** — banca/P&L da conta (abaixo)\n")
    a("3. **POLICIES LEGADAS** — fora deste resumo\n")
    a("4. **ESTUDOS HISTÓRICOS / CONTRAFACTUAIS** — apêndice de pesquisa (não operacional)\n\n")
    return "".join(lines)
