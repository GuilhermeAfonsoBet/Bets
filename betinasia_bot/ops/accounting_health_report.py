"""Daily markdown snippet for Accounting Health — H3BUP (no side effects)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_health(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"status": "ACCOUNTING_UNKNOWN_FAILURE", "health": "CRITICAL", "error_type": "HEALTH_FILE_MISSING"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"status": "ACCOUNTING_PARSE_FAILED", "health": "CRITICAL", "error_type": str(e)[:120]}


def render_accounting_health_h3bup_section(
    *,
    health: Dict[str, Any],
    reconcile_summary: Optional[Dict[str, Any]] = None,
) -> str:
    s = reconcile_summary or {}
    c = s.get("counts") or {}
    rows = [
        ("status", f"{health.get('status')} / {health.get('health')}"),
        ("último sucesso UTC", health.get("checked_at_utc")),
        ("balance age", (health.get("balance") or {}).get("age_sec")),
        ("open_stakes age", (health.get("open_stakes") or {}).get("age_sec")),
        ("falhas consecutivas", health.get("consecutive_failures")),
        ("última falha", health.get("error_type") or health.get("error_message")),
        ("LIVE_OK total", s.get("n_live_ok", "—")),
        ("settled reconciliado", s.get("n_settled_confirmed", "—")),
        ("não iniciados", c.get("EVENT_NOT_STARTED", "—")),
        ("abertos", (c.get("OPEN_NOT_SETTLED", 0) or 0) + (c.get("EVENT_IN_PROGRESS", 0) or 0) if c else "—"),
        ("missing accounting", c.get("SETTLED_MISSING_ACCOUNTING", "—")),
        ("coverage accounting", s.get("accounting_coverage", "—")),
        ("stake settled", s.get("stake_settled", "—")),
        ("P&L settled", s.get("pnl_settled", "—")),
        ("ROI settled", s.get("roi_settled", "—")),
    ]
    lines = ["## Accounting Health — H3BUP", "", "| Métrica | Valor |", "|---|---|"]
    for k, v in rows:
        lines.append(f"| {k} | {v} |")
    lines.append("")
    if s.get("disclaimer_low_n") or s.get("disclaimer_low_coverage") or health.get("health") != "HEALTHY":
        lines.append(
            "_Disclaimer: ROI settled é parcial (N baixo e/ou coverage/health insuficientes); não é ROI total da estratégia._"
        )
        lines.append("")
    return "\n".join(lines)
