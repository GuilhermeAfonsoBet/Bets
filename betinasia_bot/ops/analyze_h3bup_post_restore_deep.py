#!/usr/bin/env python3
"""Deep post-restore H3BUP ops+performance analysis."""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def pf(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(str(x).replace(",", "").replace("$", ""))
    except Exception:
        return None


def main() -> int:
    root = Path("/home/betbot/Bets/betinasia_bot")
    acc = root / "logs" / "accounting"
    opens = sorted(acc.glob("*open_stakes*.csv"), key=lambda p: p.stat().st_mtime)[-1]
    bals = sorted(acc.glob("*balance*.csv"), key=lambda p: p.stat().st_mtime)[-1]
    print("open", opens.name)
    print("bal", bals.name)

    with opens.open(newline="", encoding="utf-8", errors="ignore") as f:
        orows = list(csv.DictReader(f))
    print("open_n", len(orows))
    if orows:
        print("open_cols", list(orows[0].keys()))
        stake_keys = [k for k in orows[0] if "stake" in k.lower() or "matched" in k.lower() or "size" in k.lower()]
        print("stake_keys", stake_keys)
        stakes = []
        for r in orows:
            for k in stake_keys or list(r.keys()):
                v = pf(r.get(k))
                if v is not None and v > 0:
                    stakes.append((k, v, r))
                    break
        print("open_stake_sum", sum(v for _, v, _ in stakes), "n", len(stakes))
        print("open_stake_dist", Counter(round(v, 2) for _, v, _ in stakes).most_common(20))
        for _, v, r in stakes[:25]:
            print(
                {
                    "stake": v,
                    **{k: r.get(k) for k in list(r.keys())[:10]},
                }
            )

    with bals.open(newline="", encoding="utf-8", errors="ignore") as f:
        brows = list(csv.DictReader(f))
    print("bal_n", len(brows), "cols", list(brows[0].keys()) if brows else None)
    for r in brows[-3:]:
        print({k: r.get(k) for k in list(r.keys())[:20]})

    cut = "2026-07-20T18:47:00"
    restore = "2026-07-20T15:16:00"
    p = root / "logs" / "executor_live.jsonl"

    legacy_live = []
    h3_live = []
    post_non_h3 = []
    caps = []
    day_status = defaultdict(Counter)
    block_combo = Counter()

    with p.open() as f:
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
            ts = str(req.get("created_at") or "")
            if ts < restore:
                continue
            pol = str((req.get("policy") or {}).get("policy_version") or "")
            raw = r.get("raw") or {}
            br = ((req.get("meta") or {}).get("bridge") or {})
            vs = raw.get("value_sizing") or {}
            day_status[ts[:10]][st] += 1
            if st == "LIVE_OK":
                info = None
                data = (raw.get("order_resp") or {}).get("data")
                if isinstance(data, dict):
                    info = data.get("event_info")
                row = {
                    "ts": ts,
                    "pol": pol,
                    "sr": (req.get("policy") or {}).get("stake_requested"),
                    "sent": (raw.get("sent") or {}).get("stake"),
                    "odd_dec": req.get("odd_at_decision"),
                    "odd_final": r.get("odd_final") or (raw.get("sent") or {}).get("price"),
                    "slip": vs.get("slippage_pre_pct"),
                    "event": req.get("event_id"),
                    "side": req.get("side"),
                    "line": req.get("line"),
                    "teams": info,
                    "limit": br.get("betslip_limit"),
                }
                if "H3BUP" in pol:
                    h3_live.append(row)
                else:
                    legacy_live.append(row)
            if ts >= cut and pol and "H3BUP" not in pol:
                post_non_h3.append(
                    {
                        "ts": ts,
                        "st": st,
                        "pol": pol,
                        "sr": (req.get("policy") or {}).get("stake_requested"),
                        "err": str(r.get("error") or "")[:100],
                    }
                )
            if st == "CAP_BLOCKED" and "H3BUP" in pol and ts >= cut:
                caps.append(pf(br.get("betslip_limit")))
                err = str(r.get("error") or "")
                block_combo[err[:120]] += 1

    print("\n=== LIVE SUMMARY ===")
    print("legacy_live", len(legacy_live), "stake_sum", sum(pf(x["sent"]) or 0 for x in legacy_live))
    print("h3_live", len(h3_live), "stake_sum", sum(pf(x["sent"]) or 0 for x in h3_live))
    print("post_fix_non_h3", len(post_non_h3))
    if post_non_h3:
        print("post_non_h3 sample", post_non_h3[:10])

    print("\n=== LEGACY LIVE DETAIL ===")
    for x in legacy_live:
        t = x.get("teams") or {}
        print(
            x["ts"],
            "sent",
            x["sent"],
            "odd",
            x["odd_dec"],
            "->",
            x["odd_final"],
            "slip",
            x["slip"],
            "side/line",
            x["side"],
            x["line"],
            t.get("event_name") if isinstance(t, dict) else None,
            t.get("competition_name") if isinstance(t, dict) else None,
        )

    print("\n=== CAPACITY on H3BUP CAP_BLOCKED post-fix ===")
    print("n", len(caps), "null", sum(1 for c in caps if c is None))
    buckets = Counter()
    for c in caps:
        if c is None:
            buckets["null"] += 1
        elif c <= 0:
            buckets["<=0"] += 1
        elif c <= 10:
            buckets["0-10"] += 1
        elif c <= 50:
            buckets["10-50"] += 1
        elif c <= 100:
            buckets["50-100"] += 1
        else:
            buckets[">100"] += 1
    print(dict(buckets))
    print("top block reasons", block_combo.most_common(8))

    print("\n=== BY DAY ===")
    for d, c in sorted(day_status.items()):
        print(d, dict(c))

    # try settle PnL via balance history keywords matching event names
    print("\n=== TRY MATCH SETTLED ROWS IN BALANCE FOR LEGACY EVENTS ===")
    names = []
    for x in legacy_live:
        t = x.get("teams") or {}
        if isinstance(t, dict) and t.get("event_name"):
            names.append(str(t["event_name"]))
    print("event_names", names)
    # scan balance csv for those team tokens
    hits = []
    with bals.open(newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        text_cols = [c for c in cols if c]
        for r in reader:
            blob = " ".join(str(r.get(c) or "") for c in text_cols).lower()
            for n in names:
                token = n.split(" vs")[0].split(" vs.")[0].lower()
                if token and token[:12] in blob:
                    hits.append(r)
                    break
    print("balance_hits", len(hits))
    for r in hits[:40]:
        print({k: r.get(k) for k in list(r.keys())[:18]})

    # Proxy / session health quick
    print("\n=== SESSION / PROXY QUICK ===")
    auth = root / "logs" / "auth_guard_state.json"
    if auth.exists():
        print("auth_guard_state", auth.read_text()[:500])
    # last executor errors
    errp = root / "logs" / "executor_error.log"
    if errp.exists():
        print("executor_error_tail:")
        print(errp.read_text(errors="ignore")[-1200:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
