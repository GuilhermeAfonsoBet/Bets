from __future__ import annotations

import argparse
import json
import os
import statistics
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


def _pctl(xs: List[float], p: float) -> Optional[float]:
    if not xs:
        return None
    xs2 = sorted(xs)
    k = (len(xs2) - 1) * (p / 100.0)
    f = int(k)
    c = min(len(xs2) - 1, f + 1)
    if f == c:
        return float(xs2[f])
    return float(xs2[f] + (k - f) * (xs2[c] - xs2[f]))


def _parse_iso(s: str) -> Optional[datetime]:
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
class Agg:
    n: int
    p50: Optional[float]
    p90: Optional[float]
    p99: Optional[float]
    mean: Optional[float]


def _agg(xs: List[float]) -> Agg:
    if not xs:
        return Agg(0, None, None, None, None)
    return Agg(
        n=len(xs),
        p50=_pctl(xs, 50),
        p90=_pctl(xs, 90),
        p99=_pctl(xs, 99),
        mean=float(statistics.fmean(xs)),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="KPIs de execução (lag/slippage/status) a partir do JSONL do executor.")
    ap.add_argument("--jsonl", default=os.getenv("EXECUTOR_JSONL", "logs/executor_live.jsonl"))
    ap.add_argument("--last", type=int, default=int(os.getenv("EXEC_KPI_LAST", "5000")), help="Ler apenas os últimos N registros (0=all).")
    args = ap.parse_args()

    path = Path(str(args.jsonl))
    if not path.exists():
        print(json.dumps({"error": "jsonl_not_found", "path": str(path)}, ensure_ascii=False))
        return 2

    # read lines (best-effort; para N pequeno não é caro)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if int(args.last) > 0 and len(lines) > int(args.last):
        lines = lines[-int(args.last) :]

    status_counts: Dict[str, int] = {}
    t_call: List[float] = []
    t_post: List[float] = []
    t_queue: List[float] = []
    slip_abs: List[float] = []
    slip_pct: List[float] = []
    first_ts: Optional[str] = None
    last_ts: Optional[str] = None

    for ln in lines:
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        res = obj.get("result") if isinstance(obj, dict) else None
        req = obj.get("request") if isinstance(obj, dict) else None
        if not isinstance(res, dict):
            continue

        st = str(res.get("status") or "")
        status_counts[st] = int(status_counts.get(st, 0)) + 1

        created_at = _parse_iso(str(res.get("created_at") or "")) or _parse_iso(str(req.get("created_at") or "")) if isinstance(req, dict) else None
        finished_at = _parse_iso(str(res.get("finished_at") or ""))
        if created_at and (first_ts is None or created_at.isoformat() < first_ts):
            first_ts = created_at.isoformat()
        if finished_at:
            last_ts = finished_at.isoformat()

        timing = res.get("timing") or {}
        q = _safe_int(timing.get("queue_delay_ms"))
        c = _safe_int(timing.get("call_to_done_ms"))
        p = _safe_int(timing.get("post_ms"))
        if q is not None:
            t_queue.append(float(q))
        if c is not None and c > 0:
            t_call.append(float(c))
        if p is not None and p > 0:
            t_post.append(float(p))

        odd_dec = _safe_float(res.get("odd_at_decision"))
        odd_fin = _safe_float(res.get("odd_final"))
        if odd_dec and odd_fin and odd_dec > 0:
            da = float(odd_fin) - float(odd_dec)
            dp = (da / float(odd_dec)) * 100.0
            slip_abs.append(da)
            slip_pct.append(dp)

    out = {
        "path": str(path),
        "n_lines": len(lines),
        "first_ts": first_ts,
        "last_ts": last_ts,
        "status_counts": dict(sorted(status_counts.items(), key=lambda x: x[1], reverse=True)),
        "timing_ms": {
            "queue_delay": _agg(t_queue).__dict__,
            "call_to_done": _agg(t_call).__dict__,
            "post": _agg(t_post).__dict__,
        },
        "slippage": {
            "abs": _agg(slip_abs).__dict__,
            "pct": _agg(slip_pct).__dict__,
        },
        "notes": [
            "slippage usa apenas linhas com odd_at_decision e odd_final presentes",
            "ROI/Lucro real depende de settlement; aqui só KPIs de execução",
        ],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

