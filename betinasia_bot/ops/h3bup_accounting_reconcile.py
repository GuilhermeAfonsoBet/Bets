"""H3BUP order-level accounting reconciliation (read-only)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .accounting_status import order_id_key


CUTOFF_DEFAULT = "2026-07-28T13:19:39+00:00"
POLICY_SUBSTR = "H3BUP_vNext"


def _parse_ts(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or str(x).strip() == "":
            return None
        return float(str(x).replace(",", "."))
    except Exception:
        return None


def load_balance_pnl_by_order(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate balance.csv by order id.
    amount is treated as ledger cash movement / P&L-like (as in existing accounting_report).
    """
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            oid = order_id_key(row.get("order id") or row.get("order_id") or row.get("orderid"))
            if not oid:
                continue
            amt = _safe_float(row.get("amount"))
            typ = str(row.get("type") or "").strip().lower()
            note = str(row.get("note") or "")
            got = _safe_float(row.get("got price") or row.get("got_price"))
            post = str(row.get("post date") or row.get("post_date") or "")
            status = str(row.get("status") or "")
            rec = out.setdefault(
                oid,
                {
                    "order_id": oid,
                    "amount_sum": 0.0,
                    "n_rows": 0,
                    "types": [],
                    "notes": [],
                    "got_prices": [],
                    "post_dates": [],
                    "statuses": [],
                },
            )
            rec["n_rows"] += 1
            if amt is not None:
                rec["amount_sum"] = float(rec["amount_sum"]) + float(amt)
            if typ:
                rec["types"].append(typ)
            if note:
                rec["notes"].append(note[:200])
            if got is not None:
                rec["got_prices"].append(got)
            if post:
                rec["post_dates"].append(post)
            if status:
                rec["statuses"].append(status)
    return out


def load_open_order_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            oid = order_id_key(row.get("order id") or row.get("order_id"))
            if oid:
                ids.add(oid)
    return ids


def iter_live_ok_from_jsonl(
    path: Path,
    *,
    cutoff: datetime,
    policy_substr: str = POLICY_SUBSTR,
) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            result = o.get("result") or {}
            req = o.get("request") or {}
            status = str(result.get("status") or o.get("status") or "")
            if status != "LIVE_OK":
                continue
            pol_obj = result.get("policy") if isinstance(result.get("policy"), dict) else {}
            pol = str(
                o.get("policy_version")
                or req.get("policy_version")
                or pol_obj.get("policy_version")
                or (o.get("shadow") or {}).get("policy_version")
                or ""
            )
            if policy_substr not in pol:
                continue
            finished = _parse_ts(result.get("finished_at") or o.get("finished_at") or o.get("ts"))
            created = _parse_ts(req.get("created_at") or o.get("created_at"))
            ts = finished or created
            if ts is None or ts < cutoff:
                continue
            raw = result.get("raw") if isinstance(result.get("raw"), dict) else {}
            order_resp = raw.get("order_resp") if isinstance(raw.get("order_resp"), dict) else {}
            order_data = order_resp.get("data") if isinstance(order_resp.get("data"), dict) else {}
            sent = o.get("sent") or {}
            dry = o.get("dry") or req.get("dry") or {}
            if not isinstance(dry, dict):
                dry = {}
            # value_sizing / enrichment fields often nested in request or result
            vs = result.get("value_sizing") if isinstance(result.get("value_sizing"), dict) else {}
            yield {
                "raw": o,
                "order_id": order_id_key(
                    o.get("order_id")
                    or result.get("order_id")
                    or order_data.get("order_id")
                    or sent.get("order_id")
                    or (o.get("response") or {}).get("order_id")
                ),
                "execution_id": str(result.get("execution_id") or req.get("execution_id") or o.get("execution_id") or ""),
                "audit_id": str(result.get("audit_id") or req.get("audit_id") or (o.get("shadow") or {}).get("audit_id") or ""),
                "event_id": str(result.get("event_id") or req.get("event_id") or ""),
                "event_name": str(result.get("event_name") or req.get("event_name") or req.get("match_name") or ""),
                "market": str(result.get("market_type") or req.get("market_type") or req.get("market") or "AH"),
                "side": str(result.get("side") or req.get("side") or ""),
                "line": result.get("line") if result.get("line") is not None else req.get("line"),
                "odd_at_decision": result.get("odd_at_decision") or req.get("odd_at_decision"),
                "odd_final": result.get("odd_final") or sent.get("odd") or dry.get("odd_final"),
                "stake": _safe_float(
                    (result.get("sent") or {}).get("stake")
                    if isinstance(result.get("sent"), dict)
                    else None
                    or sent.get("stake")
                    or req.get("stake")
                    or pol_obj.get("stake_requested")
                    or 10
                ),
                "live_ok_ts": (finished or ts).isoformat(),
                "kickoff_ts": str(result.get("kickoff") or req.get("kickoff") or req.get("event_date") or ""),
                "policy_version": pol,
            }


def classify_row(
    *,
    order_id: str,
    kickoff_ts: str,
    now: datetime,
    balance: Optional[Dict[str, Any]],
    in_open: bool,
    snapshot_ts: datetime,
    prev_snapshot_ts: Optional[datetime],
) -> Tuple[str, str]:
    if not order_id:
        return "ORDER_ID_JOIN_FAILURE", "missing_order_id"
    ko = _parse_ts(kickoff_ts)
    if ko and ko > now:
        return "EVENT_NOT_STARTED", "kickoff_in_future"
    # heuristic event end = kickoff + 2.5h
    if ko and (ko + timedelta(hours=2, minutes=30)) > now and not balance:
        if in_open:
            return "EVENT_IN_PROGRESS", "open_stakes_and_within_match_window"
        return "EVENT_IN_PROGRESS", "within_match_window_no_settlement"
    if in_open and not balance:
        return "OPEN_NOT_SETTLED", "present_in_open_stakes"
    if balance:
        amt = float(balance.get("amount_sum") or 0.0)
        notes = " ".join(balance.get("notes") or []).lower()
        types = " ".join(balance.get("types") or []).lower()
        if abs(amt) < 1e-9 or "void" in notes or "push" in notes or "void" in types:
            return "VOID_OR_PUSH", "amount_zero_or_void_note"
        if int(balance.get("n_rows") or 0) > 1 and abs(amt) > 0:
            # multiple ledger rows may be normal (partials / adjustments)
            if "partial" in notes:
                return "PARTIAL_SETTLEMENT", "partial_note"
        if prev_snapshot_ts and any(_parse_ts(p) and _parse_ts(p) > prev_snapshot_ts for p in (balance.get("post_dates") or [])):
            return "SETTLED_AFTER_PREVIOUS_SNAPSHOT", "post_date_after_prev_snapshot"
        return "SETTLED_ACCOUNTING_OK", "found_in_balance"
    if ko and (ko + timedelta(hours=2, minutes=30)) < now:
        return "SETTLED_MISSING_ACCOUNTING", "event_ended_not_in_balance"
    return "UNKNOWN_SETTLEMENT_STATE", "no_balance_no_open_unclear_event"


def reconcile(
    *,
    live_rows: List[Dict[str, Any]],
    balance_path: Path,
    open_path: Optional[Path],
    snapshot_ts: datetime,
    prev_snapshot_ts: Optional[datetime] = None,
    now: Optional[datetime] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    now = now or datetime.now(timezone.utc)
    bal = load_balance_pnl_by_order(balance_path)
    open_ids = load_open_order_ids(open_path) if open_path and open_path.exists() else set()

    # dedupe by order_id (keep latest live_ok_ts)
    by_oid: Dict[str, Dict[str, Any]] = {}
    missing_oid = []
    for r in live_rows:
        oid = order_id_key(r.get("order_id"))
        if not oid:
            missing_oid.append(r)
            continue
        prev = by_oid.get(oid)
        if not prev or str(r.get("live_ok_ts") or "") >= str(prev.get("live_ok_ts") or ""):
            by_oid[oid] = r

    out_rows: List[Dict[str, Any]] = []
    for oid, r in sorted(by_oid.items(), key=lambda kv: str(kv[1].get("live_ok_ts") or "")):
        b = bal.get(oid)
        status, reason = classify_row(
            order_id=oid,
            kickoff_ts=str(r.get("kickoff_ts") or ""),
            now=now,
            balance=b,
            in_open=oid in open_ids,
            snapshot_ts=snapshot_ts,
            prev_snapshot_ts=prev_snapshot_ts,
        )
        out_rows.append(
            {
                "order_id": oid,
                "execution_id": r.get("execution_id"),
                "audit_id": r.get("audit_id"),
                "event_id": r.get("event_id"),
                "event_name": r.get("event_name"),
                "market": r.get("market"),
                "side": r.get("side"),
                "line": r.get("line"),
                "odd_at_decision": r.get("odd_at_decision"),
                "odd_final": r.get("odd_final"),
                "stake": r.get("stake"),
                "live_ok_ts": r.get("live_ok_ts"),
                "kickoff_ts": r.get("kickoff_ts"),
                "accounting_snapshot_ts": snapshot_ts.isoformat(),
                "accounting_status": "found" if b else ("open" if oid in open_ids else "missing"),
                "accounting_amount": (None if not b else b.get("amount_sum")),
                "matched_stake": r.get("stake"),
                "got_price": (None if not b else (b.get("got_prices") or [None])[-1]),
                "settlement_ts": (None if not b else (b.get("post_dates") or [None])[-1]),
                "reconciliation_status": status,
                "reconciliation_reason": reason,
                "balance_n_rows": (0 if not b else b.get("n_rows")),
            }
        )
    for r in missing_oid:
        out_rows.append(
            {
                **{k: r.get(k) for k in (
                    "execution_id","audit_id","event_id","event_name","market","side","line",
                    "odd_at_decision","odd_final","stake","live_ok_ts","kickoff_ts"
                )},
                "order_id": "",
                "accounting_snapshot_ts": snapshot_ts.isoformat(),
                "accounting_status": "missing_order_id",
                "accounting_amount": None,
                "matched_stake": r.get("stake"),
                "got_price": None,
                "settlement_ts": None,
                "reconciliation_status": "ORDER_ID_JOIN_FAILURE",
                "reconciliation_reason": "missing_order_id",
                "balance_n_rows": 0,
            }
        )

    settled_statuses = {"SETTLED_ACCOUNTING_OK", "VOID_OR_PUSH", "PARTIAL_SETTLEMENT", "SETTLED_AFTER_PREVIOUS_SNAPSHOT"}
    settled = [x for x in out_rows if x["reconciliation_status"] in settled_statuses]
    # ROI only on confirmed settled with numeric pnl and stake; void/push included with pnl=0
    pnl_sum = 0.0
    stake_sum = 0.0
    n_pnl = 0
    for x in settled:
        if x["reconciliation_status"] == "VOID_OR_PUSH":
            stake = _safe_float(x.get("stake")) or 0.0
            stake_sum += stake
            n_pnl += 1
            continue
        amt = _safe_float(x.get("accounting_amount"))
        stake = _safe_float(x.get("stake"))
        if amt is None or stake is None or stake <= 0:
            continue
        pnl_sum += amt
        stake_sum += stake
        n_pnl += 1

    counts = Counter(x["reconciliation_status"] for x in out_rows)
    summary = {
        "n_live_ok": len(out_rows),
        "counts": dict(counts),
        "n_settled_confirmed": len(settled),
        "stake_settled": stake_sum,
        "pnl_settled": pnl_sum,
        "roi_settled": (pnl_sum / stake_sum) if stake_sum > 0 else None,
        "accounting_coverage": (sum(1 for x in out_rows if x.get("accounting_status") == "found") / len(out_rows)) if out_rows else 0.0,
        "disclaimer_low_n": len(settled) < 30,
        "disclaimer_low_coverage": (
            (sum(1 for x in out_rows if x.get("accounting_status") == "found") / len(out_rows)) < 0.95 if out_rows else True
        ),
    }
    funnel = [
        {"stage": "LIVE_OK", "n": len(out_rows)},
        {"stage": "EVENT_NOT_STARTED", "n": counts.get("EVENT_NOT_STARTED", 0)},
        {"stage": "EVENT_IN_PROGRESS", "n": counts.get("EVENT_IN_PROGRESS", 0)},
        {"stage": "OPEN_NOT_SETTLED", "n": counts.get("OPEN_NOT_SETTLED", 0)},
        {"stage": "SETTLED_FOUND", "n": sum(counts.get(s, 0) for s in settled_statuses)},
        {"stage": "SETTLED_MISSING_ACCOUNTING", "n": counts.get("SETTLED_MISSING_ACCOUNTING", 0)},
        {"stage": "ORDER_ID_JOIN_FAILURE", "n": counts.get("ORDER_ID_JOIN_FAILURE", 0)},
        {"stage": "UNKNOWN", "n": counts.get("UNKNOWN_SETTLEMENT_STATE", 0)},
    ]
    summary["funnel"] = funnel
    return out_rows, summary


def render_daily_section(summary: Dict[str, Any], health: Dict[str, Any]) -> str:
    c = summary.get("counts") or {}
    lines = [
        "## Accounting Health — H3BUP",
        "",
        "| Métrica | Valor |",
        "|---|---|",
        f"| status | {health.get('status')} / {health.get('health')} |",
        f"| último sucesso UTC | {health.get('checked_at_utc')} |",
        f"| balance age | {(health.get('balance') or {}).get('age_sec')} |",
        f"| open_stakes age | {(health.get('open_stakes') or {}).get('age_sec')} |",
        f"| falhas consecutivas | {health.get('consecutive_failures')} |",
        f"| última falha | {health.get('error_type')} |",
        f"| LIVE_OK total | {summary.get('n_live_ok')} |",
        f"| settled reconciliado | {summary.get('n_settled_confirmed')} |",
        f"| não iniciados | {c.get('EVENT_NOT_STARTED', 0)} |",
        f"| abertos | {c.get('OPEN_NOT_SETTLED', 0) + c.get('EVENT_IN_PROGRESS', 0)} |",
        f"| missing accounting | {c.get('SETTLED_MISSING_ACCOUNTING', 0)} |",
        f"| coverage accounting | {summary.get('accounting_coverage')} |",
        f"| stake settled | {summary.get('stake_settled')} |",
        f"| P&L settled | {summary.get('pnl_settled')} |",
        f"| ROI settled | {summary.get('roi_settled')} |",
        "",
    ]
    if summary.get("disclaimer_low_n") or summary.get("disclaimer_low_coverage") or health.get("health") != "HEALTHY":
        lines.append("_Disclaimer: ROI settled é parcial (N baixo e/ou coverage/health insuficientes); não é ROI total da estratégia._")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--executor-jsonl", default=os.getenv("EXECUTOR_LIVE_JSONL", "logs/executor_live.jsonl"))
    ap.add_argument("--balance-csv", required=False)
    ap.add_argument("--open-csv", required=False)
    ap.add_argument("--out-csv", default="logs/h3bup_settlement_reconciliation_fresh_20260729.csv")
    ap.add_argument("--out-funnel", default="logs/h3bup_accounting_funnel_20260729.csv")
    ap.add_argument("--cutoff", default=CUTOFF_DEFAULT)
    ap.add_argument("--health-json", default="logs/accounting/accounting_health.json")
    args = ap.parse_args()

    out_dir = Path(os.getenv("ACCOUNTING_OUT_DIR", "logs/accounting"))
    bal = Path(args.balance_csv) if args.balance_csv else sorted(out_dir.glob("*__balance.csv"), key=lambda p: p.stat().st_mtime, reverse=True)[0]
    opn = Path(args.open_csv) if args.open_csv else (
        sorted(out_dir.glob("*__open_stakes.csv"), key=lambda p: p.stat().st_mtime, reverse=True)[0]
        if list(out_dir.glob("*__open_stakes.csv")) else None
    )
    cutoff = _parse_ts(args.cutoff) or datetime.fromisoformat(CUTOFF_DEFAULT)
    live = list(iter_live_ok_from_jsonl(Path(args.executor_jsonl), cutoff=cutoff))
    snap_ts = datetime.fromtimestamp(bal.stat().st_mtime, timezone.utc)
    rows, summary = reconcile(live_rows=live, balance_path=bal, open_path=opn, snapshot_ts=snap_ts)
    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with outp.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    fun = Path(args.out_funnel)
    with fun.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "n"])
        w.writeheader()
        for r in summary.get("funnel") or []:
            w.writerow(r)
    health = {}
    hp = Path(args.health_json)
    if hp.exists():
        health = json.loads(hp.read_text(encoding="utf-8"))
    print(json.dumps({"summary": summary, "section": render_daily_section(summary, health)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
