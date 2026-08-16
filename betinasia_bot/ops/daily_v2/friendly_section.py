"""Daily V2/V1 Friendly vs Non-Friendly breakdown (reporting / shadow only).

Does NOT filter execution. Classification version FRIENDLY_CLASS_V1_20260731.
UNCLASSIFIED is never coerced to NON_FRIENDLY.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ops.h3bup_friendly_analysis import FRIENDLY_CLASSIFICATION_VERSION
from ops.h3bup_friendly_analysis.classification import classify_entity
from ops.h3bup_friendly_analysis.enrich import enrich_orders, load_league_map_csv, try_sql_league_map

from .performance import compute_settlement_and_performance
from .statuses import metric_envelope

CLASSES = ("FRIENDLY", "NON_FRIENDLY", "UNCLASSIFIED", "CONFLICT")


def _roi_value(perf: Dict[str, Any]) -> Optional[float]:
    roi = perf.get("roi_resolved") or perf.get("roi_settled") or {}
    if isinstance(roi, dict):
        return roi.get("value")
    return None


def _class_row(label: str, orders: Dict[str, Dict[str, Any]], perf: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "class": label,
        "n_live_ok": len(orders),
        "n_open": perf.get("n_open"),
        "n_settled": perf.get("n_settled"),
        "n_void_push": perf.get("n_void_push"),
        "n_missing": perf.get("n_missing_accounting"),
        "stake_placed": perf.get("stake_placed") if perf.get("stake_placed") is not None else perf.get("stake_placed_sum"),
        "stake_resolved": perf.get("stake_resolved_total") if perf.get("stake_resolved_total") is not None else perf.get("stake_settled_sum"),
        "pnl_resolved": perf.get("pnl_resolved") if perf.get("pnl_resolved") is not None else perf.get("pnl_settled_sum"),
        "roi_resolved": _roi_value(perf),
        "roi_status": (perf.get("roi_resolved") or {}).get("status") if isinstance(perf.get("roi_resolved"), dict) else None,
        "maturity": perf.get("maturity_status"),
        "accounting_coverage_pct": (
            None
            if not orders
            else 100.0
            * (
                int(perf.get("n_settled") or 0)
                + int(perf.get("n_void_push") or 0)
                + int(perf.get("n_open") or 0)
            )
            / len(orders)
            # note: open counted in coverage of accounting presence differently;
            # prefer settled+void over live for "resolved coverage"
        ),
        "resolved_coverage_pct": (
            None
            if not orders
            else 100.0 * (int(perf.get("n_settled") or 0) + int(perf.get("n_void_push") or 0)) / len(orders)
        ),
    }


def classify_daily_orders(
    orders: Dict[str, Dict[str, Any]],
    *,
    root: Path,
) -> Dict[str, Dict[str, Any]]:
    """Return orders dict enriched with friendly_class (+ league fields when found)."""
    root = Path(root)
    rows = []
    for oid, o in orders.items():
        rows.append(
            {
                "order_id": oid,
                "event_id": o.get("event_id") or "",
                "audit_id": o.get("audit_id") or "",
                "league_name": o.get("league_name") or o.get("league") or "",
                "competition_name": o.get("competition_name") or o.get("competition") or "",
                "competition_type": o.get("competition_type") or "",
                "league_type": o.get("league_type") or "",
                "event_name": o.get("event_name") or "",
                "is_friendly": o.get("is_friendly"),
            }
        )
    league_map = load_league_map_csv(root / "logs" / "h3bup_friendly_league_map.csv")
    try:
        league_map.update(try_sql_league_map())
    except Exception:
        pass
    if league_map:
        rows = enrich_orders(rows, league_map=league_map)

    out: Dict[str, Dict[str, Any]] = {}
    for base, row in zip(orders.values(), rows):
        oid = str(row.get("order_id") or base.get("order_id") or "")
        merged = dict(base)
        for k in ("league_name", "competition_name", "competition_type", "event_name", "event_id"):
            if row.get(k) and not merged.get(k):
                merged[k] = row.get(k)
        cls = classify_entity(
            structured_flag=row.get("is_friendly"),
            competition_type=row.get("competition_type") or merged.get("competition_type"),
            league_type=row.get("league_type") or merged.get("league_type"),
            league_name=row.get("league_name") or merged.get("league_name"),
            competition_name=row.get("competition_name") or merged.get("competition_name"),
            tournament_name=row.get("tournament_name"),
            event_name=row.get("event_name") or merged.get("event_name"),
        )
        merged["friendly_class"] = cls.friendly_class
        merged["friendly_source"] = cls.friendly_source
        merged["friendly_rule_id"] = cls.friendly_rule_id
        out[oid] = merged
    return out


def build_friendly_section(
    *,
    root: Path,
    orders: Dict[str, Dict[str, Any]],
    pnl_by_oid: Dict[str, float],
    open_oids: Set[str],
    accounting_health_status: Optional[str] = None,
) -> Dict[str, Any]:
    """Shadow/diagnostic Friendly breakdown for the Daily H3BUP cohort."""
    classified = classify_daily_orders(orders, root=root)
    by_class: Dict[str, Dict[str, Dict[str, Any]]] = {c: {} for c in CLASSES}
    for oid, o in classified.items():
        cls = str(o.get("friendly_class") or "UNCLASSIFIED")
        if cls not in by_class:
            cls = "UNCLASSIFIED"
        by_class[cls][oid] = o

    rows: List[Dict[str, Any]] = []
    by_class_perf: Dict[str, Any] = {}
    for c in CLASSES:
        perf = compute_settlement_and_performance(
            orders=by_class[c],
            pnl_by_oid=pnl_by_oid,
            open_oids=open_oids,
            accounting_health_status=accounting_health_status,
        )
        by_class_perf[c] = perf
        rows.append(_class_row(c, by_class[c], perf))

    n = len(classified)
    n_classified = len(by_class["FRIENDLY"]) + len(by_class["NON_FRIENDLY"])
    coverage = (100.0 * n_classified / n) if n else None
    n_uncl = len(by_class["UNCLASSIFIED"])
    n_conflict = len(by_class["CONFLICT"])

    status = "AVAILABLE"
    if n == 0:
        status = "MISSING"
    elif coverage is not None and coverage < 50.0:
        status = "WATCH"
    elif n < 30:
        status = "INSUFFICIENT_N"

    return {
        "label": "Friendly vs Non-Friendly (diagnóstico / shadow)",
        "official_filter": False,
        "classification_version": FRIENDLY_CLASSIFICATION_VERSION,
        "status": status,
        "n_orders": n,
        "classification_coverage_pct": coverage,
        "n_friendly": len(by_class["FRIENDLY"]),
        "n_non_friendly": len(by_class["NON_FRIENDLY"]),
        "n_unclassified": n_uncl,
        "n_conflict": n_conflict,
        "rows": rows,
        "by_class_performance": {
            c: {
                "n": len(by_class[c]),
                "roi_resolved": by_class_perf[c].get("roi_resolved"),
                "maturity": by_class_perf[c].get("maturity_status"),
            }
            for c in CLASSES
        },
        "notes": [
            "Reporting-only — não altera policy, stake nem filtros de execução.",
            "UNCLASSIFIED não é tratado como NON_FRIENDLY.",
            f"classification_version={FRIENDLY_CLASSIFICATION_VERSION}",
            "roi_resolved inclui void no denominador (contrato oficial).",
        ],
        "metric_envelope": metric_envelope(
            value=coverage,
            unit="percent",
            n=n_classified,
            denominator=n,
            status=status,
            metric_version="v2.0",
            source="executor+audit_league_map+FRIENDLY_CLASS_V1",
            notes=["shadow diagnostic"],
        ),
    }


def render_friendly_markdown(section: Dict[str, Any]) -> str:
    """Markdown fragment for V1/V2 reports."""
    from .formatters import fmt_money, fmt_pct

    lines: List[str] = []
    a = lines.append
    a(f"### {section.get('label') or 'Friendly vs Non-Friendly'}\n\n")
    a(
        f"> Diagnóstico shadow · `classification_version={section.get('classification_version')}` · "
        f"**não é filtro operacional** · status=`{section.get('status')}`\n\n"
    )
    a(
        "- coverage classificação: "
        + (
            "—"
            if section.get("classification_coverage_pct") is None
            else f"{section.get('classification_coverage_pct'):.1f}%"
        )
        + f" (F={section.get('n_friendly')} · NF={section.get('n_non_friendly')} · "
        + f"U={section.get('n_unclassified')} · C={section.get('n_conflict')})\n\n"
    )
    a(
        "| Classe | N | Open | Settled | Void | Stake placed | Stake resolved | P&L resolved | ROI resolved | Maturity |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"
    )
    for r in section.get("rows") or []:
        a(
            f"| {r.get('class')} | {r.get('n_live_ok')} | {r.get('n_open')} | {r.get('n_settled')} | "
            f"{r.get('n_void_push')} | {fmt_money(r.get('stake_placed'))} | {fmt_money(r.get('stake_resolved'))} | "
            f"{fmt_money(r.get('pnl_resolved'))} | {fmt_pct(r.get('roi_resolved'))} | `{r.get('maturity')}` |\n"
        )
    a("\n")
    a("> `roi_resolved = pnl_resolved / stake_resolved_total` (void no denominador). "
      "Comparar classes só com maturity/coverage visíveis.\n\n")
    return "".join(lines)
