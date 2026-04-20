from __future__ import annotations

import argparse
import json
import os
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _parse_iso(s: Any) -> Optional[datetime]:
    try:
        t = str(s or "").strip()
        if not t:
            return None
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        return datetime.fromisoformat(t)
    except Exception:
        return None


@dataclass
class Row:
    execution_id: str
    created_at: Optional[str]
    order_id: Optional[str]
    status: str
    exec_side: str

    market_regime: Optional[str]
    market_is_live: Optional[bool]

    stake_sent: Optional[float]
    pre_submit_ms: Optional[int]
    rule: Optional[str]
    eligible: Optional[bool]
    stake_chosen: Optional[float]
    skip_reason: Optional[str]

    call_to_done_ms: Optional[int]
    post_ms: Optional[int]
    total_api_ms: Optional[int]
    pmm_wait_ms: Optional[int]
    queue_delay_ms: Optional[int]


def _tail_lines(path: Path, last: int) -> List[str]:
    dq: deque[str] = deque(maxlen=max(1, int(last)))
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                dq.append(ln)
    return list(dq)


def _extract_market(req: Dict[str, Any]) -> Tuple[Optional[str], Optional[bool]]:
    meta = req.get("meta") if isinstance(req.get("meta"), dict) else {}
    mkt = meta.get("market") if isinstance(meta.get("market"), dict) else None
    if not isinstance(mkt, dict):
        return None, None
    reg = mkt.get("regime")
    isl = mkt.get("is_live")
    regime = str(reg).strip().lower() if reg is not None else None
    is_live = bool(isl) if isl is not None else None
    if regime not in ("pre", "in"):
        regime = ("in" if is_live else "pre") if is_live is not None else None
    return regime, is_live


def _extract_order_id(raw: Dict[str, Any]) -> Optional[str]:
    if not isinstance(raw, dict):
        return None
    for k in ("order_id", "orderId", "bet_id", "betId", "id"):
        v = raw.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s and s.isdigit():
            return s
    return None


def _expected_stake_for_back(
    *,
    market_regime: Optional[str],
    pre_submit_ms: Optional[int],
    pre_fast_max_ms: int,
    stake_hi: float,
    stake_lo: float,
) -> Optional[float]:
    if market_regime not in ("pre", "in"):
        return None
    is_pre = (market_regime == "pre")
    ok_time = (pre_submit_ms is not None) and (int(pre_submit_ms) <= int(pre_fast_max_ms))
    return float(stake_hi if (is_pre and ok_time) else stake_lo)


def main() -> int:
    ap = argparse.ArgumentParser(description="Auditoria de staking (Back Pre fast=12; senão=1.50) no executor_jsonl.")
    ap.add_argument("--jsonl", default=os.getenv("EXECUTOR_JSONL", "logs/executor_live.jsonl"))
    ap.add_argument("--last", type=int, default=20000, help="Ler os últimos N registros do JSONL.")
    ap.add_argument("--pre-fast-max-ms", type=int, default=5000)
    ap.add_argument("--stake-hi", type=float, default=12.0)
    ap.add_argument("--stake-lo", type=float, default=1.5)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--max-print", type=int, default=40)
    args = ap.parse_args()

    path = Path(str(args.jsonl))
    if not path.exists():
        print(json.dumps({"error": "jsonl_not_found", "path": str(path)}, ensure_ascii=False))
        return 2

    lines = _tail_lines(path, int(args.last))
    rows: List[Row] = []

    for ln in lines:
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        req = obj.get("request") if isinstance(obj.get("request"), dict) else {}
        res = obj.get("result") if isinstance(obj.get("result"), dict) else {}
        status = str(res.get("status") or "")
        if status != "LIVE_OK":
            continue

        exec_side = str(res.get("exec_side") or req.get("exec_side") or "")
        if exec_side.strip().lower() != "back":
            continue

        raw = res.get("raw") if isinstance(res.get("raw"), dict) else {}
        sent = raw.get("sent") if isinstance(raw.get("sent"), dict) else {}
        vs = raw.get("value_sizing") if isinstance(raw.get("value_sizing"), dict) else {}
        timing = res.get("timing") if isinstance(res.get("timing"), dict) else {}

        stake_sent = _safe_float(sent.get("stake"))
        pre_submit_ms = _safe_int(vs.get("pre_submit_ms"))
        rule = str(vs.get("rule") or "") or None
        eligible = (bool(vs.get("eligible")) if vs.get("eligible") is not None else None)
        stake_chosen = _safe_float(vs.get("stake_chosen"))
        skip_reason = str(vs.get("skip_reason") or "") or None

        market_regime, market_is_live = _extract_market(req)
        created_at = str(res.get("created_at") or req.get("created_at") or "")

        rows.append(
            Row(
                execution_id=str(res.get("execution_id") or req.get("execution_id") or ""),
                created_at=created_at or None,
                order_id=_extract_order_id(raw),
                status=status,
                exec_side=exec_side,
                market_regime=market_regime,
                market_is_live=market_is_live,
                stake_sent=stake_sent,
                pre_submit_ms=pre_submit_ms,
                rule=rule,
                eligible=eligible,
                stake_chosen=stake_chosen,
                skip_reason=skip_reason,
                call_to_done_ms=_safe_int(timing.get("call_to_done_ms")),
                post_ms=_safe_int(timing.get("post_ms")),
                total_api_ms=_safe_int(timing.get("total_ms")),
                pmm_wait_ms=_safe_int(timing.get("pmm_wait_ms")),
                queue_delay_ms=_safe_int(timing.get("queue_delay_ms")),
            )
        )

    stake_counts = Counter([r.stake_sent for r in rows])
    exp_counts = Counter([_expected_stake_for_back(market_regime=r.market_regime, pre_submit_ms=r.pre_submit_ms, pre_fast_max_ms=int(args.pre_fast_max_ms), stake_hi=float(args.stake_hi), stake_lo=float(args.stake_lo)) for r in rows])

    mismatches: List[Dict[str, Any]] = []
    hi_wrong: List[Dict[str, Any]] = []
    missed_hi: List[Dict[str, Any]] = []

    for r in rows:
        exp = _expected_stake_for_back(
            market_regime=r.market_regime,
            pre_submit_ms=r.pre_submit_ms,
            pre_fast_max_ms=int(args.pre_fast_max_ms),
            stake_hi=float(args.stake_hi),
            stake_lo=float(args.stake_lo),
        )
        if exp is None or r.stake_sent is None:
            continue
        if abs(float(r.stake_sent) - float(exp)) > float(args.eps):
            mismatches.append(
                {
                    "execution_id": r.execution_id,
                    "created_at": r.created_at,
                    "order_id": r.order_id,
                    "stake_sent": r.stake_sent,
                    "expected_stake": exp,
                    "market_regime": r.market_regime,
                    "pre_submit_ms": r.pre_submit_ms,
                    "rule": r.rule,
                    "stake_chosen": r.stake_chosen,
                    "skip_reason": r.skip_reason,
                    "call_to_done_ms": r.call_to_done_ms,
                    "post_ms": r.post_ms,
                    "total_api_ms": r.total_api_ms,
                    "pmm_wait_ms": r.pmm_wait_ms,
                    "queue_delay_ms": r.queue_delay_ms,
                }
            )

        # stake_hi deveria implicar pre&fast
        if abs(float(r.stake_sent) - float(args.stake_hi)) <= float(args.eps):
            ok = (r.market_regime == "pre") and (r.pre_submit_ms is not None) and (int(r.pre_submit_ms) <= int(args.pre_fast_max_ms))
            if not ok:
                hi_wrong.append(
                    {
                        "execution_id": r.execution_id,
                        "created_at": r.created_at,
                        "order_id": r.order_id,
                        "stake_sent": r.stake_sent,
                        "market_regime": r.market_regime,
                        "pre_submit_ms": r.pre_submit_ms,
                        "rule": r.rule,
                        "stake_chosen": r.stake_chosen,
                    }
                )

        # deveria ser stake_hi mas foi stake_lo
        if (r.market_regime == "pre") and (r.pre_submit_ms is not None) and (int(r.pre_submit_ms) <= int(args.pre_fast_max_ms)):
            if abs(float(r.stake_sent) - float(args.stake_hi)) > float(args.eps):
                missed_hi.append(
                    {
                        "execution_id": r.execution_id,
                        "created_at": r.created_at,
                        "order_id": r.order_id,
                        "stake_sent": r.stake_sent,
                        "market_regime": r.market_regime,
                        "pre_submit_ms": r.pre_submit_ms,
                        "rule": r.rule,
                        "stake_chosen": r.stake_chosen,
                        "skip_reason": r.skip_reason,
                    }
                )

    out = {
        "path": str(path),
        "n_scanned": len(lines),
        "n_live_ok_back": len(rows),
        "stake_sent_counts": {str(k): v for k, v in stake_counts.items()},
        "expected_stake_counts": {str(k): v for k, v in exp_counts.items()},
        "params": {
            "pre_fast_max_ms": int(args.pre_fast_max_ms),
            "stake_hi": float(args.stake_hi),
            "stake_lo": float(args.stake_lo),
        },
        "mismatch_n": len(mismatches),
        "stake_hi_wrong_n": len(hi_wrong),
        "missed_hi_n": len(missed_hi),
        "mismatch_sample": mismatches[: int(args.max_print)],
        "stake_hi_wrong_sample": hi_wrong[: int(args.max_print)],
        "missed_hi_sample": missed_hi[: int(args.max_print)],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

