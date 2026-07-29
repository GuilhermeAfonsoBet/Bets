"""Operational universes and deduplication for Daily V2.

Aligned with V1 `_parse_executor_jsonl_back_live_orders` field layout:
  status in result.status; exec_side Back/back; order_id in raw.order_resp.data;
  pre_submit_ms in raw.value_sizing; stake in raw.sent.stake or policy.stake_requested.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

from . import DAILY_FAST_LE_6S_MS, STUDY_FAST_LT_4S_MS
from .time_windows import ReportWindow, ensure_utc, in_half_open, parse_dt


H3BUP_POLICY_NEEDLE = "H3BUP_vNext"


def _extract_order_id_from_raw(raw: Any) -> Optional[str]:
    if not isinstance(raw, dict):
        return None
    oid = str(raw.get("order_id") or "").strip() or None
    if oid:
        return oid
    resp = raw.get("order_resp")
    if isinstance(resp, dict):
        for k in ("id", "order_id", "orderId", "uuid", "uid"):
            v = resp.get(k)
            if v is not None and str(v).strip():
                return str(v).strip()
        for k in ("data", "order", "result"):
            v = resp.get(k)
            if isinstance(v, dict):
                for kk in ("id", "order_id", "orderId", "uuid", "uid"):
                    vv = v.get(kk)
                    if vv is not None and str(vv).strip():
                        return str(vv).strip()
    if isinstance(resp, str) and resp.strip():
        return resp.strip()
    return None


def _policy_version_of(obj: dict, req: dict, res: dict) -> str:
    for src in (
        obj.get("policy_version"),
        (req.get("policy") or {}).get("policy_version") if isinstance(req.get("policy"), dict) else None,
        (res.get("policy") or {}).get("policy_version") if isinstance(res.get("policy"), dict) else None,
        req.get("policy_version"),
        res.get("policy_version"),
        (obj.get("meta") or {}).get("policy_version") if isinstance(obj.get("meta"), dict) else None,
        (req.get("meta") or {}).get("policy_version") if isinstance(req.get("meta"), dict) else None,
    ):
        if src:
            return str(src)
    return ""


def _is_heartbeat(obj: dict, req: dict, res: dict) -> bool:
    st = str(res.get("status") or obj.get("status") or "")
    if st in {"LIVE_OK", "DRY_OK", "CAP_BLOCKED", "API_FAILED", "LIVE_PRECHECK_FAILED", "LIVE_PLACE_FAILED"}:
        return False
    blob = json.dumps({"o": obj.get("note"), "r": req.get("meta"), "h": obj.get("heartbeat")}, default=str)
    return "heartbeat" in blob.lower()


def iter_jsonl(path: Path, *, max_lines: Optional[int] = None) -> Iterable[dict]:
    if not path.exists():
        return
    n = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj
                n += 1
                if max_lines is not None and n >= max_lines:
                    break


def load_executor_orders(
    path: Path,
    *,
    window: ReportWindow,
    require_h3bup: bool = True,
    max_lines: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Return deduped LIVE_OK Back orders keyed by order_id within window."""
    out: Dict[str, Dict[str, Any]] = {}
    for obj in iter_jsonl(path, max_lines=max_lines):
        req = obj.get("request") if isinstance(obj.get("request"), dict) else {}
        res = obj.get("result") if isinstance(obj.get("result"), dict) else {}
        if _is_heartbeat(obj, req, res):
            continue
        st = str(res.get("status") or obj.get("status") or "").strip()
        if st != "LIVE_OK":
            continue
        exec_side = str(res.get("exec_side") or req.get("exec_side") or "").strip().lower()
        if exec_side != "back":
            continue
        created = parse_dt(res.get("created_at") or req.get("created_at") or obj.get("created_at"))
        if created is None:
            continue
        if not in_half_open(created, window.window_start_utc, window.window_end_utc):
            continue
        raw = res.get("raw") if isinstance(res.get("raw"), dict) else {}
        oid = _extract_order_id_from_raw(raw)
        if not oid:
            continue
        policy_version = _policy_version_of(obj, req, res)
        is_h3bup = H3BUP_POLICY_NEEDLE in policy_version
        if require_h3bup and policy_version and not is_h3bup:
            continue
        if require_h3bup and not policy_version:
            # Unknown policy: exclude from H3BUP-strict universe
            continue

        sent = raw.get("sent") if isinstance(raw.get("sent"), dict) else {}
        stake = None
        try:
            if sent.get("stake") is not None:
                stake = float(sent.get("stake"))
        except Exception:
            stake = None
        if stake is None:
            pol = res.get("policy") if isinstance(res.get("policy"), dict) else (
                req.get("policy") if isinstance(req.get("policy"), dict) else {}
            )
            try:
                if pol and pol.get("stake_requested") is not None:
                    stake = float(pol.get("stake_requested"))
            except Exception:
                pass

        vs = raw.get("value_sizing") if isinstance(raw.get("value_sizing"), dict) else {}
        pre_submit_ms = None
        try:
            if vs.get("pre_submit_ms") is not None:
                pre_submit_ms = int(float(vs.get("pre_submit_ms")))
        except Exception:
            pre_submit_ms = None

        audit_id = res.get("audit_id") if res.get("audit_id") is not None else req.get("audit_id")
        rec = {
            "order_id": str(oid),
            "status": st,
            "created_at": ensure_utc(created).isoformat(),
            "created_at_dt": ensure_utc(created),
            "stake": stake,
            "pre_submit_ms": pre_submit_ms,
            "policy_version": policy_version,
            "exec_side": exec_side,
            "audit_id": audit_id,
            "slippage_pre_pct": vs.get("slippage_pre_pct"),
            "is_h3bup": is_h3bup,
        }
        prev = out.get(str(oid))
        if prev is None or ensure_utc(created) >= prev["created_at_dt"]:
            out[str(oid)] = rec
    return out


def classify_fast_buckets(orders: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    daily_fast = []
    study_fast = []
    na = []
    slow_daily = []
    for oid, o in orders.items():
        ms = o.get("pre_submit_ms")
        if ms is None:
            na.append(oid)
            continue
        if int(ms) <= DAILY_FAST_LE_6S_MS:
            daily_fast.append(oid)
        else:
            slow_daily.append(oid)
        if int(ms) < STUDY_FAST_LT_4S_MS:
            study_fast.append(oid)
    n_known = len(daily_fast) + len(slow_daily)
    return {
        "DAILY_FAST_LE_6S": {
            "n": len(daily_fast),
            "order_ids": daily_fast,
            "threshold_ms": DAILY_FAST_LE_6S_MS,
            "op": "<=",
        },
        "STUDY_FAST_LT_4S": {
            "n": len(study_fast),
            "order_ids": study_fast,
            "threshold_ms": STUDY_FAST_LT_4S_MS,
            "op": "<",
            "label": "exploratory_only",
        },
        "DAILY_SLOW_GT_6S": {"n": len(slow_daily), "order_ids": slow_daily},
        "PRE_SUBMIT_MS_NA": {"n": len(na), "order_ids": na},
        "n_with_pre_submit_ms": n_known,
    }


def load_open_order_ids(path: Path) -> Set[str]:
    if not path or not Path(path).exists():
        return set()
    out: Set[str] = set()
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if Path(path).suffix.lower() == ".json":
        try:
            data = json.loads(text)
            if isinstance(data, list):
                for x in data:
                    if isinstance(x, dict) and x.get("order_id") is not None:
                        out.add(str(x["order_id"]))
        except Exception:
            return out
        return out
    lines = text.splitlines()
    if not lines:
        return out
    headers = [h.strip().strip('"') for h in lines[0].split(",")]
    idx = None
    for i, h in enumerate(headers):
        if h.lower().replace(" ", "_") in {"order_id", "orderid", "id"}:
            idx = i
            break
    if idx is None:
        return out
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) > idx and parts[idx].strip():
            out.add(parts[idx].strip().strip('"'))
    return out


def load_pnl_by_order_from_balance_csv(path: Path) -> Dict[str, float]:
    """Lifetime lifetime P&L-like sum per order_id from accounting balance CSV.

    Uses post date only as row timestamp metadata — NOT for cohort day.
    """
    p = Path(path)
    if not p.exists():
        return {}
    out: Dict[str, float] = {}
    with p.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}
        cols = {c.lower(): c for c in reader.fieldnames}

        def pick(*names):
            for n in names:
                if n in cols:
                    return cols[n]
            for k, orig in cols.items():
                for n in names:
                    if n in k:
                        return orig
            return None

        oid_c = pick("order_id", "order id", "orderid")
        amt_c = pick("amount", "pnl", "profit", "value")
        type_c = pick("type", "transaction type", "tx_type", "description")
        if not oid_c or not amt_c:
            return {}
        exclude_types = {"deposit", "withdraw", "withdrawal", "transfer", "bonus"}
        for row in reader:
            oid = str(row.get(oid_c) or "").strip()
            if not oid:
                continue
            if type_c:
                t = str(row.get(type_c) or "").strip().lower()
                if any(x in t for x in exclude_types):
                    continue
            try:
                amt = float(str(row.get(amt_c) or "0").replace(",", ""))
            except Exception:
                continue
            out[oid] = float(out.get(oid, 0.0) + amt)
    return out
