from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


def compute_kpis_from_lines(
    lines: List[str],
    *,
    path: str = "",
    only_status: Optional[Iterable[str]] = None,
    exclude_status: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    only_set = set([str(s).strip() for s in (only_status or []) if str(s).strip()])
    excl_set = set([str(s).strip() for s in (exclude_status or []) if str(s).strip()])

    status_counts: Dict[str, int] = {}
    t_call: List[float] = []
    t_post: List[float] = []
    t_queue: List[float] = []
    t_total: List[float] = []
    t_pmm_wait: List[float] = []
    t_total_minus_post: List[float] = []
    t_call_minus_total: List[float] = []
    slip_abs: List[float] = []
    slip_pct: List[float] = []
    slip_by_side: Dict[str, Dict[str, List[float]]] = {}
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
        if st == "HEARTBEAT":
            continue
        if only_set and st not in only_set:
            continue
        if excl_set and st in excl_set:
            continue

        status_counts[st] = int(status_counts.get(st, 0)) + 1

        created_at = _parse_iso(str(res.get("created_at") or ""))
        if created_at is None and isinstance(req, dict):
            created_at = _parse_iso(str(req.get("created_at") or ""))
        finished_at = _parse_iso(str(res.get("finished_at") or ""))
        if created_at and (first_ts is None or created_at.isoformat() < first_ts):
            first_ts = created_at.isoformat()
        if finished_at:
            last_ts = finished_at.isoformat()

        timing = res.get("timing") or {}
        q = _safe_int(timing.get("queue_delay_ms"))
        c = _safe_int(timing.get("call_to_done_ms"))
        p = _safe_int(timing.get("post_ms"))
        tot = _safe_int(timing.get("total_ms"))
        pmm = _safe_int(timing.get("pmm_wait_ms"))
        if q is not None:
            t_queue.append(float(q))
        if c is not None and c > 0:
            t_call.append(float(c))
        if p is not None and p > 0:
            t_post.append(float(p))
        if tot is not None and tot > 0:
            t_total.append(float(tot))
        if pmm is not None and pmm >= 0:
            t_pmm_wait.append(float(pmm))
        if tot is not None and p is not None and tot > 0 and p > 0:
            t_total_minus_post.append(float(max(0.0, float(tot) - float(p))))
        if c is not None and tot is not None and c > 0 and tot > 0:
            t_call_minus_total.append(float(max(0.0, float(c) - float(tot))))

        odd_dec = _safe_float(res.get("odd_at_decision"))
        if odd_dec is None and isinstance(req, dict):
            odd_dec = _safe_float(req.get("odd_at_decision"))
        odd_fin = _safe_float(res.get("odd_final"))
        if odd_dec and odd_fin and odd_dec > 0:
            da = float(odd_fin) - float(odd_dec)
            dp = (da / float(odd_dec)) * 100.0
            slip_abs.append(da)
            slip_pct.append(dp)

            side = str(res.get("exec_side") or (req.get("exec_side") if isinstance(req, dict) else "") or "").strip()
            side_norm = side.capitalize() if side else "NA"
            slot = slip_by_side.setdefault(side_norm, {"raw_abs": [], "raw_pct": [], "cost_abs": [], "cost_pct": []})
            slot["raw_abs"].append(float(da))
            slot["raw_pct"].append(float(dp))

            # "cost" = movimento adverso ao operador (>=0):
            # - Back: odds caíram (da<0 / dp<0) é pior
            # - Lay: odds subiram (da>0 / dp>0) é pior
            if side_norm.lower() == "back":
                slot["cost_abs"].append(float(max(0.0, -da)))
                slot["cost_pct"].append(float(max(0.0, -dp)))
            elif side_norm.lower() == "lay":
                slot["cost_abs"].append(float(max(0.0, da)))
                slot["cost_pct"].append(float(max(0.0, dp)))
            else:
                # desconhecido: usa abs como "custo" neutro
                slot["cost_abs"].append(float(abs(da)))
                slot["cost_pct"].append(float(abs(dp)))

    return {
        "path": str(path),
        "n_lines": len(lines),
        "first_ts": first_ts,
        "last_ts": last_ts,
        "filter": {"only_status": sorted(list(only_set)), "exclude_status": sorted(list(excl_set))},
        "status_counts": dict(sorted(status_counts.items(), key=lambda x: x[1], reverse=True)),
        "timing_ms": {
            "queue_delay": _agg(t_queue).__dict__,
            "call_to_done": _agg(t_call).__dict__,
            "post": _agg(t_post).__dict__,
            "total_api": _agg(t_total).__dict__,
            "pmm_wait": _agg(t_pmm_wait).__dict__,
            "total_minus_post": _agg(t_total_minus_post).__dict__,
            "call_minus_total": _agg(t_call_minus_total).__dict__,
        },
        "slippage": {
            "abs": _agg(slip_abs).__dict__,
            "pct": _agg(slip_pct).__dict__,
        },
        "slippage_by_side": {
            k: {
                "raw_abs": _agg(v.get("raw_abs") or []).__dict__,
                "raw_pct": _agg(v.get("raw_pct") or []).__dict__,
                "cost_abs": _agg(v.get("cost_abs") or []).__dict__,
                "cost_pct": _agg(v.get("cost_pct") or []).__dict__,
            }
            for k, v in sorted(slip_by_side.items(), key=lambda x: x[0])
        },
        "notes": [
            "slippage usa apenas linhas com odd_at_decision e odd_final presentes",
            "slippage.abs/pct são (odd_final - odd_at_decision); sinal não é comparável entre Back e Lay",
            "slippage_by_side.cost_* sempre representa movimento adverso (>=0) por lado",
            "timing.total_api é timing.total_ms do ApiBetslipClient (POST + espera PMMs); pmm_wait usa timing.pmm_wait_ms quando disponível",
            "total_minus_post ≈ espera de PMMs/WS (proxy); call_minus_total ≈ overhead fora do ApiBetslipClient (fila/scheduler/outros)",
            "ROI/Lucro real depende de settlement; aqui só KPIs de execução",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="KPIs de execução (lag/slippage/status) a partir do JSONL do executor.")
    ap.add_argument("--jsonl", default=os.getenv("EXECUTOR_JSONL", "logs/executor_live.jsonl"))
    ap.add_argument("--last", type=int, default=int(os.getenv("EXEC_KPI_LAST", "5000")), help="Ler apenas os últimos N registros (0=all).")
    ap.add_argument("--only-status", default=os.getenv("EXEC_KPI_ONLY_STATUS", "").strip(), help="CSV de status para incluir (ex.: LIVE_OK,DRY_OK).")
    ap.add_argument("--exclude-status", default=os.getenv("EXEC_KPI_EXCLUDE_STATUS", "").strip(), help="CSV de status para excluir.")
    args = ap.parse_args()

    path = Path(str(args.jsonl))
    if not path.exists():
        print(json.dumps({"error": "jsonl_not_found", "path": str(path)}, ensure_ascii=False))
        return 2

    # read lines (best-effort; para N pequeno não é caro)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if int(args.last) > 0 and len(lines) > int(args.last):
        lines = lines[-int(args.last) :]
    only = [s.strip() for s in str(args.only_status).split(",") if s.strip()]
    excl = [s.strip() for s in str(args.exclude_status).split(",") if s.strip()]
    out = compute_kpis_from_lines(lines, path=str(path), only_status=only, exclude_status=excl)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

