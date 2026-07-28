#!/usr/bin/env python3
"""Verificação operacional + performance H3BUP pós-religação (VPS).

Uso na VPS:
  python3 ops/verify_h3bup_post_restore.py \\
    --since 2026-07-20T15:16:00Z \\
    --jsonl /home/betbot/Bets/betinasia_bot/logs/executor_live.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _pf(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _parse_ts(s: str) -> Optional[datetime]:
    if not s:
        return None
    s = str(s).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


@dataclass
class Row:
    ts: str
    status: str
    policy: str
    stake_req: Optional[float]
    stake_sent: Optional[float]
    odd_dec: Optional[float]
    odd_final: Optional[float]
    slip: Optional[float]
    err: str
    event_id: str
    side: str
    line: str
    exec_side: str
    market_is_live: Optional[bool]


def load_rows(path: Path, since: datetime) -> List[Row]:
    out: List[Row] = []
    with path.open() as f:
        for line in f:
            try:
                o = json.loads(line)
            except Exception:
                continue
            req = o.get("request") or {}
            r = o.get("result") or {}
            st = str(r.get("status") or "")
            if st == "HEARTBEAT":
                continue
            ts = str(req.get("created_at") or r.get("created_at") or "")
            dt = _parse_ts(ts)
            if dt is None or dt < since:
                continue
            raw = r.get("raw") or {}
            vs = raw.get("value_sizing") or {}
            pol = ((req.get("policy") or {}).get("policy_version")) or ""
            meta_m = ((req.get("meta") or {}).get("market") or {})
            out.append(
                Row(
                    ts=ts,
                    status=st,
                    policy=str(pol),
                    stake_req=_pf((req.get("policy") or {}).get("stake_requested")),
                    stake_sent=_pf((raw.get("sent") or {}).get("stake")),
                    odd_dec=_pf(req.get("odd_at_decision")),
                    odd_final=_pf(r.get("odd_final") or (raw.get("sent") or {}).get("price")),
                    slip=_pf(vs.get("slippage_pre_pct")),
                    err=str(r.get("error") or "")[:160],
                    event_id=str(req.get("event_id") or ""),
                    side=str(req.get("side") or ""),
                    line=str(req.get("line") or ""),
                    exec_side=str(req.get("exec_side") or ""),
                    market_is_live=(
                        bool(meta_m.get("is_live"))
                        if isinstance(meta_m, dict) and "is_live" in meta_m
                        else None
                    ),
                )
            )
    return out


def svc(unit: str) -> Tuple[str, str]:
    a = subprocess.run(["systemctl", "is-active", unit], capture_output=True, text=True)
    e = subprocess.run(["systemctl", "is-enabled", unit], capture_output=True, text=True)
    return a.stdout.strip() or a.stderr.strip(), e.stdout.strip() or e.stderr.strip()


def unit_env(unit: str, keys: Iterable[str]) -> Dict[str, str]:
    pid = subprocess.check_output(
        ["systemctl", "show", "-p", "MainPID", "--value", unit], text=True
    ).strip()
    if not pid or pid == "0":
        return {}
    raw = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    env = {}
    for item in raw:
        if b"=" not in item:
            continue
        k, v = item.decode(errors="ignore").split("=", 1)
        env[k] = v
    return {k: env.get(k, "") for k in keys}


def code_markers(root: Path) -> Dict[str, bool]:
    w = (root / "executor" / "worker.py").read_text(errors="ignore")
    b = (root / "ops" / "executor_bridge_audit.py").read_text(errors="ignore")
    return {
        "worker_non_h3bup_reject": "non_h3bup_backpre_rejected" in w,
        "worker_hard_stake_10": "hard_stake_cap_h3bup_only_10" in w,
        "worker_h3bup_gate": "H3BUP_VNEXT_GATE" in w,
        "worker_force_stake_10": "H3BUP_vNext_force_stake_10" in w,
        "bridge_h3bup_live_only": "h3bup_vnext_live_submit_required" in b,
        "bridge_version_const": "H3BUP_vNext_20260629" in b,
    }


def analyze(rows: List[Row]) -> Dict[str, Any]:
    by_status = Counter(r.status for r in rows)
    by_policy = Counter(r.policy for r in rows)
    live = [r for r in rows if r.status == "LIVE_OK"]
    blocked = [r for r in rows if r.status == "CAP_BLOCKED"]
    non_h3 = [r for r in rows if r.policy and "H3BUP_vNext" not in r.policy]
    sent_bad = [
        r
        for r in live
        if r.stake_sent is not None and abs(float(r.stake_sent) - 10.0) > 1e-6
    ]
    req_bad = [
        r
        for r in rows
        if r.stake_req is not None and abs(float(r.stake_req) - 10.0) > 1e-6
    ]
    odd_out = [
        r
        for r in live
        if r.odd_dec is None or not (1.85 <= float(r.odd_dec) <= 2.15)
    ]
    err_c = Counter()
    for r in blocked:
        err_c[r.err.split("|")[0][:80] if r.err else "(empty)"] += 1

    # Performance proxy from execution log (no settlement): volume / fill rate
    stake_sum = sum(float(r.stake_sent or 0) for r in live)
    by_day = defaultdict(lambda: Counter())
    for r in rows:
        day = r.ts[:10]
        by_day[day][r.status] += 1
        if r.status == "LIVE_OK":
            by_day[day]["stake_sum"] += float(r.stake_sent or 0)

    return {
        "n": len(rows),
        "by_status": dict(by_status),
        "by_policy": dict(by_policy),
        "live_ok": len(live),
        "cap_blocked": len(blocked),
        "non_h3bup_requests": len(non_h3),
        "live_stake_not_10": [
            {"ts": r.ts, "sent": r.stake_sent, "pol": r.policy, "odd": r.odd_dec}
            for r in sent_bad[:20]
        ],
        "req_stake_not_10": [
            {"ts": r.ts, "sr": r.stake_req, "st": r.status, "pol": r.policy}
            for r in req_bad[:20]
        ],
        "live_odd_outside_band": [
            {"ts": r.ts, "odd": r.odd_dec, "sent": r.stake_sent}
            for r in odd_out[:20]
        ],
        "block_reasons": err_c.most_common(15),
        "live_stake_sum": stake_sum,
        "live_avg_odd": (
            sum(float(r.odd_final or r.odd_dec or 0) for r in live) / len(live)
            if live
            else None
        ),
        "by_day": {d: dict(c) for d, c in sorted(by_day.items())},
        "last_10_live": [
            {
                "ts": r.ts,
                "pol": r.policy,
                "sent": r.stake_sent,
                "odd": r.odd_final or r.odd_dec,
                "slip": r.slip,
                "event": r.event_id,
                "side": r.side,
                "line": r.line,
            }
            for r in live[-10:]
        ],
        "last_15_any": [
            {
                "ts": r.ts,
                "st": r.status,
                "pol": r.policy,
                "sr": r.stake_req,
                "sent": r.stake_sent,
                "odd": r.odd_dec,
                "err": r.err,
            }
            for r in rows[-15:]
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-07-20T15:16:00+00:00")
    ap.add_argument(
        "--jsonl",
        default="/home/betbot/Bets/betinasia_bot/logs/executor_live.jsonl",
    )
    ap.add_argument("--root", default="/home/betbot/Bets/betinasia_bot")
    args = ap.parse_args()

    since = _parse_ts(args.since)
    assert since is not None
    root = Path(args.root)
    jsonl = Path(args.jsonl)

    print("=== OPERACIONAL ===")
    for u in [
        "betinasia-executor",
        "betinasia-executor-bridge-back",
        "betinasia-executor-bridge-lay",
        "betinasia-executor-bridge-dt",
    ]:
        a, e = svc(u)
        print(f"{u}: active={a} enabled={e}")

    keys = [
        "BRIDGE_MODE",
        "BRIDGE_STAKE",
        "BRIDGE_HYPOTHESIS",
        "BRIDGE_PREMATCH_ONLY",
        "EXECUTOR_LIVE_STAKE",
        "EXECUTOR_LIVE_MAX_STAKE",
        "EXECUTOR_BACKPRE_FAST_STAKE_ENABLE",
        "EXECUTOR_BACKPRE_FAST_STAKE_HI",
    ]
    print("executor env:", unit_env("betinasia-executor", keys))
    print("bridge-back env:", unit_env("betinasia-executor-bridge-back", keys))
    print("code markers:", code_markers(root))

    risk = root / "logs" / "bridge_risk_params.json"
    if risk.exists():
        print("risk_params:", risk.read_text()[:500])

    print("\n=== PERFORMANCE / FLUXO since", args.since, "===")
    rows = load_rows(jsonl, since)
    rep = analyze(rows)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
