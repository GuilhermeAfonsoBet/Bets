"""Primary H3BUP_vNext exact universe + optional historical comparable appendix."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import POLICY_START_UTC, POLICY_VERSION_EXACT

try:
    from ops.accounting_status import order_id_key
except Exception:  # pragma: no cover
    def order_id_key(x: Any) -> str:  # type: ignore
        return str(x or "").strip()


def parse_dt(raw: Any) -> Optional[datetime]:
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


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or str(x).strip() == "":
            return None
        return float(str(x).replace(",", "."))
    except Exception:
        return None


def extract_order_id(obj: dict, req: dict, res: dict) -> Optional[str]:
    raw = res.get("raw") if isinstance(res.get("raw"), dict) else {}
    if not isinstance(raw, dict):
        raw = {}
    order_resp = raw.get("order_resp") if isinstance(raw.get("order_resp"), dict) else {}
    order_data = order_resp.get("data") if isinstance(order_resp.get("data"), dict) else {}
    sent = raw.get("sent") if isinstance(raw.get("sent"), dict) else {}
    candidates = [
        obj.get("order_id"),
        res.get("order_id"),
        req.get("order_id"),
        order_data.get("order_id"),
        order_data.get("id"),
        order_resp.get("order_id"),
        order_resp.get("id"),
        sent.get("order_id"),
        raw.get("order_id"),
    ]
    for c in candidates:
        oid = order_id_key(c)
        if oid:
            return oid
    return None


def policy_version_of(obj: dict, req: dict, res: dict) -> str:
    pol = res.get("policy") if isinstance(res.get("policy"), dict) else {}
    req_pol = req.get("policy") if isinstance(req.get("policy"), dict) else {}
    for src in (
        obj.get("policy_version"),
        req.get("policy_version"),
        res.get("policy_version"),
        pol.get("policy_version"),
        req_pol.get("policy_version"),
        (obj.get("shadow") or {}).get("policy_version") if isinstance(obj.get("shadow"), dict) else None,
    ):
        if src:
            return str(src)
    return ""


def policy_id_of(obj: dict, req: dict, res: dict, policy_version: str) -> str:
    for src in (
        obj.get("policy_id"),
        req.get("policy_id"),
        res.get("policy_id"),
        (res.get("policy") or {}).get("policy_id") if isinstance(res.get("policy"), dict) else None,
        (req.get("policy") or {}).get("policy_id") if isinstance(req.get("policy"), dict) else None,
    ):
        if src:
            return str(src)
    if "H3BUP_vNext" in policy_version:
        return "H3BUP_vNext"
    return ""


def is_heartbeat(obj: dict, req: dict, res: dict) -> bool:
    st = str(res.get("status") or obj.get("status") or "")
    if st in {"LIVE_OK", "DRY_OK", "CAP_BLOCKED", "API_FAILED", "LIVE_PRECHECK_FAILED", "LIVE_PLACE_FAILED"}:
        return False
    blob = json.dumps({"o": obj.get("note"), "r": req.get("meta"), "h": obj.get("heartbeat")}, default=str)
    return "heartbeat" in blob.lower()


def dig_league_fields(obj: dict, req: dict, res: dict) -> Dict[str, Any]:
    """Collect competition identity fields from common nests (no P&L)."""
    nests = [obj, req, res]
    for n in (req, res, obj):
        if isinstance(n.get("meta"), dict):
            nests.append(n["meta"])
        if isinstance(n.get("shadow"), dict):
            nests.append(n["shadow"])
        if isinstance(n.get("raw"), dict):
            nests.append(n["raw"])
            if isinstance(n["raw"].get("enrichment"), dict):
                nests.append(n["raw"]["enrichment"])

    def first(*keys: str) -> Any:
        for nest in nests:
            if not isinstance(nest, dict):
                continue
            for k in keys:
                v = nest.get(k)
                if v not in (None, ""):
                    return v
        return None

    return {
        "league_id": first("league_id", "leagueId"),
        "league_name": first("league_name", "league", "League"),
        "competition_id": first("competition_id", "competitionId"),
        "competition_name": first("competition_name", "competition", "Competition"),
        "competition_type": first("competition_type", "comp_type", "competitionType"),
        "league_type": first("league_type", "leagueType"),
        "is_friendly": first("is_friendly", "friendly_flag", "competition_is_friendly"),
        "tournament_name": first("tournament_name", "tournament"),
        "country": first("country", "competition_country", "league_country"),
        "event_name": first("event_name", "match_name", "eventName"),
        "event_id": first("event_id", "eventId"),
        "match_id": first("match_id", "matchId"),
    }


def iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if isinstance(o, dict):
                yield o


def _timing_ms(res: dict, raw: dict, key: str) -> Optional[float]:
    timing = res.get("timing") if isinstance(res.get("timing"), dict) else {}
    vs = raw.get("value_sizing") if isinstance(raw.get("value_sizing"), dict) else {}
    for src in (timing, vs, res, raw):
        if isinstance(src, dict) and src.get(key) is not None:
            try:
                return float(src.get(key))
            except Exception:
                pass
    return None


def parse_live_record(obj: dict) -> Optional[Dict[str, Any]]:
    req = obj.get("request") if isinstance(obj.get("request"), dict) else {}
    res = obj.get("result") if isinstance(obj.get("result"), dict) else {}
    if is_heartbeat(obj, req, res):
        return None
    status = str(res.get("status") or obj.get("status") or "").strip()
    raw = res.get("raw") if isinstance(res.get("raw"), dict) else {}
    sent = raw.get("sent") if isinstance(raw.get("sent"), dict) else {}
    vs = raw.get("value_sizing") if isinstance(raw.get("value_sizing"), dict) else {}
    pol = res.get("policy") if isinstance(res.get("policy"), dict) else (
        req.get("policy") if isinstance(req.get("policy"), dict) else {}
    )
    pv = policy_version_of(obj, req, res)
    pid = policy_id_of(obj, req, res, pv)
    created = parse_dt(res.get("created_at") or req.get("created_at") or obj.get("created_at"))
    finished = parse_dt(res.get("finished_at") or obj.get("finished_at"))
    league = dig_league_fields(obj, req, res)
    oid = extract_order_id(obj, req, res)
    stake = safe_float(sent.get("stake"))
    if stake is None:
        stake = safe_float(pol.get("stake_requested") if isinstance(pol, dict) else None)
    exec_side = str(res.get("exec_side") or req.get("exec_side") or "").strip().lower()
    side = str(res.get("side") or req.get("side") or "").strip()
    # IMPORTANT: executor `is_live` means LIVE betting mode (vs dry), NOT in-play.
    # Period/regime must come from market_regime / period / regime / market_period.
    period = "Pre"
    vs = raw.get("value_sizing") if isinstance(raw.get("value_sizing"), dict) else {}
    for p in (
        res.get("period"),
        req.get("period"),
        res.get("regime"),
        req.get("regime"),
        res.get("market_regime"),
        req.get("market_regime"),
        vs.get("market_regime"),
        res.get("market_period"),
        req.get("market_period"),
    ):
        if p is None:
            continue
        ps = str(p).strip().lower()
        if ps in {"pre", "prematch", "pre-match", "full_time", "ft"}:
            period = "Pre"
        elif ps in {"in", "inplay", "in-play", "in_play", "live_match"}:
            period = "In"
    # H3BUP_vNext is an operational Back-Pre policy; keep Pre unless explicit in-play regime.
    if "H3BUP_vNext" in pv and period != "In":
        period = "Pre"

    return {
        "order_id": oid or "",
        "execution_id": str(res.get("execution_id") or req.get("execution_id") or obj.get("execution_id") or ""),
        "audit_id": str(res.get("audit_id") or req.get("audit_id") or ""),
        "trace_id": str(res.get("trace_id") or req.get("trace_id") or obj.get("trace_id") or ""),
        "status": status,
        "policy_id": pid,
        "policy_version": pv,
        "side": side,
        "exec_side": exec_side,
        "period": period,
        "line": res.get("line") if res.get("line") is not None else req.get("line"),
        "odd_at_decision": safe_float(res.get("odd_at_decision") if res.get("odd_at_decision") is not None else req.get("odd_at_decision")),
        "odd_final": safe_float(res.get("odd_final") if res.get("odd_final") is not None else sent.get("odd")),
        "stake": stake,
        "capacity_final": safe_float(res.get("limit_final") if res.get("limit_final") is not None else res.get("capacity_final")),
        "slippage_pre_pct": safe_float(vs.get("slippage_pre_pct")),
        "pre_submit_ms": safe_float(vs.get("pre_submit_ms")),
        "call_to_done_ms": _timing_ms(res, raw, "call_to_done_ms"),
        "place_duration_ms": _timing_ms(res, raw, "place_duration_ms"),
        "bookmaker": str(res.get("bookie_final") or sent.get("bookmaker") or res.get("bookmaker") or ""),
        "created_at_utc": created.isoformat() if created else "",
        "created_at_dt": created,
        "live_ok_ts": (finished or created).isoformat() if (finished or created) else "",
        "kickoff_utc": str(res.get("kickoff") or req.get("kickoff") or req.get("event_date") or ""),
        **league,
        "_fallback_key": None,
    }


def load_primary_h3bup_universe(
    executor_jsonl: Path,
    *,
    cutoff: datetime,
    policy_version_exact: str = POLICY_VERSION_EXACT,
    policy_start: Optional[datetime] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """H3BUP_vNext exact: LIVE_OK, Back, Pre, exact policy_version, since start.

    Dedup by order_id. Fallback identity only when order_id missing:
    execution_id + live_ok_ts + event_id — never mixed with valid order_id rows.
    """
    policy_start = policy_start or parse_dt(POLICY_START_UTC)
    assert policy_start is not None

    with_oid: Dict[str, Dict[str, Any]] = {}
    fallback: Dict[str, Dict[str, Any]] = {}
    excluded = {
        "not_live_ok": 0,
        "legacy_policy": 0,
        "wrong_policy_version": 0,
        "not_back": 0,
        "not_pre": 0,
        "before_start": 0,
        "after_cutoff": 0,
        "stake_20_legacy": 0,
        "dry_ok": 0,
        "heartbeat": 0,
        "duplicates_collapsed": 0,
    }

    for obj in iter_jsonl(executor_jsonl) or []:
        rec = parse_live_record(obj)
        if rec is None:
            excluded["heartbeat"] += 1
            continue
        st = rec["status"]
        if st == "DRY_OK":
            excluded["dry_ok"] += 1
            continue
        if st != "LIVE_OK":
            excluded["not_live_ok"] += 1
            continue
        pv = rec["policy_version"]
        if "bridge_h3b_live_v0" in pv or (pv and "H3BUP_vNext" not in pv):
            excluded["legacy_policy"] += 1
            continue
        if pv != policy_version_exact:
            excluded["wrong_policy_version"] += 1
            continue
        if rec["exec_side"] != "back":
            excluded["not_back"] += 1
            continue
        if rec["period"] != "Pre":
            excluded["not_pre"] += 1
            continue
        created = rec.get("created_at_dt")
        if created is None:
            excluded["before_start"] += 1
            continue
        if created < policy_start:
            excluded["before_start"] += 1
            continue
        if created > cutoff:
            excluded["after_cutoff"] += 1
            continue
        # Exclude legacy stake 20
        if rec.get("stake") is not None and abs(float(rec["stake"]) - 20.0) < 1e-9:
            excluded["stake_20_legacy"] += 1
            continue

        oid = order_id_key(rec.get("order_id"))
        if oid:
            prev = with_oid.get(oid)
            if prev is None:
                with_oid[oid] = rec
            else:
                excluded["duplicates_collapsed"] += 1
                if str(rec.get("live_ok_ts") or "") >= str(prev.get("live_ok_ts") or ""):
                    with_oid[oid] = rec
            continue

        # Fallback identity — only when order_id missing
        fb = f"{rec.get('execution_id')}|{rec.get('live_ok_ts')}|{rec.get('event_id')}"
        if not rec.get("execution_id") or not rec.get("event_id"):
            continue
        rec["order_id"] = ""
        rec["_fallback_key"] = fb
        rec["_identity_mode"] = "fallback_execution_event"
        prev = fallback.get(fb)
        if prev is None or str(rec.get("live_ok_ts") or "") >= str(prev.get("live_ok_ts") or ""):
            fallback[fb] = rec

    rows = list(with_oid.values())
    # Attach synthetic order_id placeholder for fallback? Keep separate list but include in universe
    # with empty order_id and fallback key documented — brief says fallback only when no order_id
    for r in fallback.values():
        r["_identity_mode"] = "fallback_execution_event"
        rows.append(r)

    for r in rows:
        r.setdefault("_identity_mode", "order_id")
        # drop non-serializable
        r.pop("created_at_dt", None)

    rows.sort(key=lambda r: (str(r.get("created_at_utc") or ""), str(r.get("order_id") or "")))
    meta = {
        "universe": "H3BUP_vNext_exact",
        "policy_id": "H3BUP_vNext",
        "policy_version": policy_version_exact,
        "n": len(rows),
        "n_with_order_id": sum(1 for r in rows if r.get("order_id")),
        "n_fallback_identity": sum(1 for r in rows if not r.get("order_id")),
        "excluded": excluded,
        "cutoff_utc": cutoff.isoformat(),
        "policy_start_utc": policy_start.isoformat(),
    }
    return rows, meta


def load_secondary_historical_comparable(
    executor_jsonl: Path,
    *,
    cutoff: datetime,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """HISTORICAL_COMPARABLE_BACK_PRE — diagnostic only; never merge into primary ROI."""
    rows: List[Dict[str, Any]] = []
    seen: Dict[str, Dict[str, Any]] = {}
    for obj in iter_jsonl(executor_jsonl) or []:
        rec = parse_live_record(obj)
        if rec is None:
            continue
        if rec["status"] != "LIVE_OK":
            continue
        if rec["exec_side"] != "back":
            continue
        if rec["period"] != "Pre":
            continue
        if not rec.get("order_id"):
            continue
        if rec.get("stake") is None or rec.get("odd_at_decision") is None:
            continue
        league_ok = bool(rec.get("league_name") or rec.get("competition_name"))
        if not league_ok:
            continue
        created = parse_dt(rec.get("created_at_utc"))
        if created is None or created > cutoff:
            continue
        oid = order_id_key(rec["order_id"])
        prev = seen.get(oid)
        if prev is None or str(rec.get("live_ok_ts") or "") >= str(prev.get("live_ok_ts") or ""):
            seen[oid] = rec
    rows = list(seen.values())
    for r in rows:
        r.pop("created_at_dt", None)
        r["universe"] = "HISTORICAL_COMPARABLE_BACK_PRE"
    meta = {
        "universe": "HISTORICAL_COMPARABLE_BACK_PRE",
        "diagnostic_only": True,
        "n": len(rows),
        "note": "Never consolidate with H3BUP_vNext exact ROI.",
    }
    return rows, meta
