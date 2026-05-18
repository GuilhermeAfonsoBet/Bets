from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _safe_float_or_none(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _normalize_order_id(x: Any) -> Optional[str]:
    s = str(x or "").strip()
    if not s:
        return None
    if s.isdigit():
        return s
    # Casos comuns em CSV/API: "12345.0", "#12345", "order=12345"
    try:
        f = float(s.replace(",", "."))
        if f >= 0 and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
    except Exception:
        pass
    m = re.search(r"(\d{4,})", s)
    if m:
        return str(m.group(1))
    return None


def _parse_dt_any(s: str) -> Optional[datetime]:
    t = str(s or "").strip()
    if not t:
        return None
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _extract_order_id_from_raw(raw: Any) -> Optional[str]:
    try:
        if isinstance(raw, dict):
            oid = _normalize_order_id(raw.get("order_id"))
            if oid:
                return oid
            sent = raw.get("sent")
            if isinstance(sent, dict):
                for k in ("id", "order_id", "orderId", "uuid", "uid"):
                    oid = _normalize_order_id(sent.get(k))
                    if oid:
                        return oid
            data = raw.get("data")
            if isinstance(data, dict):
                for k in ("id", "order_id", "orderId", "uuid", "uid"):
                    oid = _normalize_order_id(data.get(k))
                    if oid:
                        return oid
        return None
    except Exception:
        return None


@dataclass
class ExecRow:
    order_id: str
    created_at: datetime
    lat_ms: float
    stake: float


def _load_exec_rows(exec_jsonl: Path, *, since_utc: datetime) -> Dict[str, ExecRow]:
    out: Dict[str, ExecRow] = {}
    if not exec_jsonl.exists():
        return out
    for ln in exec_jsonl.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        req = obj.get("request") if isinstance(obj.get("request"), dict) else {}
        res = obj.get("result") if isinstance(obj.get("result"), dict) else {}
        if str(res.get("status") or "").strip() != "LIVE_OK":
            continue
        side = str(res.get("exec_side") or req.get("exec_side") or "").strip().lower()
        if side != "back":
            continue
        created = _parse_dt_any(str(res.get("created_at") or req.get("created_at") or ""))
        if created is None or created < since_utc:
            continue
        raw = res.get("raw") if isinstance(res.get("raw"), dict) else {}
        oid = _extract_order_id_from_raw(raw)
        if not oid:
            continue
        timing = res.get("timing") if isinstance(res.get("timing"), dict) else {}
        lat = _safe_float_or_none(timing.get("call_to_done_ms"))
        if lat is None:
            continue
        sent = raw.get("sent") if isinstance(raw.get("sent"), dict) else {}
        stake = _safe_float_or_none(sent.get("stake"))
        if stake is None:
            pol = res.get("policy") if isinstance(res.get("policy"), dict) else (
                req.get("policy") if isinstance(req.get("policy"), dict) else {}
            )
            stake = _safe_float_or_none((pol or {}).get("stake_requested"))
        if stake is None or stake <= 0:
            continue
        rec = ExecRow(order_id=str(oid), created_at=created, lat_ms=float(lat), stake=float(stake))
        prev = out.get(str(oid))
        if prev is None or rec.created_at >= prev.created_at:
            out[str(oid)] = rec
    return out


def _load_pnl_by_order(balance_csv: Path, *, since_day: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not balance_csv.exists():
        return out
    with balance_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        r = csv.DictReader(f)
        cols = list(r.fieldnames or [])
        if not cols:
            return out

        def _pick(keys: List[str]) -> Optional[str]:
            for k in keys:
                for c in cols:
                    cl = c.lower()
                    if cl == k or cl.startswith(k) or k in cl:
                        return c
            return None

        dt_col = _pick(["post date", "post_date", "date", "settled", "closed", "time"])
        oid_col = _pick(["order_id", "order id", "order", "bet id", "bet_id", "id"])
        pnl_col = _pick(["amount", "profit_loss", "profit", "p&l", "pnl", "net", "pl"])
        if not dt_col or not oid_col or not pnl_col:
            return out

        for row in r:
            if not isinstance(row, dict):
                continue
            oid = _normalize_order_id(row.get(oid_col))
            if not oid:
                continue
            dt = _parse_dt_any(str(row.get(dt_col) or ""))
            if dt is None or str(dt.date().isoformat()) < str(since_day):
                continue
            pnl = _safe_float_or_none(row.get(pnl_col))
            if pnl is None:
                continue
            out[oid] = float(out.get(oid) or 0.0) + float(pnl)
    return out


def _summarize(rows: List[Dict[str, float]]) -> Dict[str, Optional[float]]:
    n = len(rows)
    if n == 0:
        return {"n": 0, "stake_sum": 0.0, "roi_mean_pct": None, "roi_weighted_pct": None}
    stake_sum = float(sum(float(r.get("stake") or 0.0) for r in rows))
    roi_vals = [float(r.get("roi") or 0.0) for r in rows]
    roi_mean = float(sum(roi_vals) / float(n))
    pnl_sum = float(sum(float(r.get("pnl") or 0.0) for r in rows))
    roi_weighted = (float(pnl_sum) / float(stake_sum) * 100.0) if stake_sum > 0 else None
    return {
        "n": int(n),
        "stake_sum": float(stake_sum),
        "roi_mean_pct": float(roi_mean),
        "roi_weighted_pct": (float(roi_weighted) if roi_weighted is not None else None),
    }


def _bootstrap_delta_mean(
    a: List[float],
    b: List[float],
    *,
    n_boot: int = 5000,
    seed: int = 123,
) -> Dict[str, Optional[float]]:
    if len(a) < 5 or len(b) < 5:
        return {"delta_mean": None, "ci90_lb": None, "ci90_ub": None, "ci95_lb": None, "ci95_ub": None}
    rnd = random.Random(int(seed))
    na = len(a)
    nb = len(b)
    deltas: List[float] = []
    for _ in range(max(300, int(n_boot))):
        sa = 0.0
        sb = 0.0
        for _i in range(na):
            sa += float(a[rnd.randrange(0, na)])
        for _j in range(nb):
            sb += float(b[rnd.randrange(0, nb)])
        deltas.append((sa / float(na)) - (sb / float(nb)))
    deltas.sort()
    n = len(deltas)
    i05 = int(round(0.05 * (n - 1)))
    i95 = int(round(0.95 * (n - 1)))
    i025 = int(round(0.025 * (n - 1)))
    i975 = int(round(0.975 * (n - 1)))
    i50 = int(round(0.50 * (n - 1)))
    return {
        "delta_mean": float(deltas[i50]),
        "ci90_lb": float(deltas[i05]),
        "ci90_ub": float(deltas[i95]),
        "ci95_lb": float(deltas[i025]),
        "ci95_ub": float(deltas[i975]),
    }


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{x:.2f}%"


def _fmt_num(x: Optional[float], nd: int = 2) -> str:
    if x is None:
        return "—"
    return f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="ROI médio por faixas de tempo de execução Back LIVE_OK (accounting por order_id).")
    ap.add_argument("--exec-jsonl", default="logs/executor_live.jsonl")
    ap.add_argument("--balance-csv", default="logs/accounting/latest__balance.csv")
    ap.add_argument("--since-day", default="2026-04-20")
    ap.add_argument("--out", default="logs/backpre_exec_time_roi_analysis.md")
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    since_utc = _parse_dt_any(f"{args.since_day}T00:00:00+00:00")
    if since_utc is None:
        raise SystemExit(f"since_day inválido: {args.since_day}")

    exec_rows = _load_exec_rows(Path(str(args.exec_jsonl)), since_utc=since_utc)
    pnl_by_oid = _load_pnl_by_order(Path(str(args.balance_csv)), since_day=str(args.since_day))

    rows: List[Dict[str, float]] = []
    n_join = 0
    for oid, em in exec_rows.items():
        pnl = _safe_float_or_none(pnl_by_oid.get(str(oid)))
        if pnl is None:
            continue
        n_join += 1
        stake = float(em.stake)
        if stake <= 0:
            continue
        roi = float(pnl) / float(stake) * 100.0
        rows.append({"lat_ms": float(em.lat_ms), "stake": float(stake), "pnl": float(pnl), "roi": float(roi)})

    g_lt4 = [r for r in rows if float(r["lat_ms"]) < 4000.0]
    g_lt6 = [r for r in rows if float(r["lat_ms"]) < 6000.0]
    g_gt6 = [r for r in rows if float(r["lat_ms"]) > 6000.0]

    s_lt4 = _summarize(g_lt4)
    s_lt6 = _summarize(g_lt6)
    s_gt6 = _summarize(g_gt6)

    d_lt4_vs_gt6 = _bootstrap_delta_mean([float(r["roi"]) for r in g_lt4], [float(r["roi"]) for r in g_gt6], n_boot=int(args.n_boot), seed=123)
    d_lt6_vs_gt6 = _bootstrap_delta_mean([float(r["roi"]) for r in g_lt6], [float(r["roi"]) for r in g_gt6], n_boot=int(args.n_boot), seed=223)
    d_lt4_vs_lt6 = _bootstrap_delta_mean([float(r["roi"]) for r in g_lt4], [float(r["roi"]) for r in g_lt6], n_boot=int(args.n_boot), seed=323)

    md: List[str] = []
    md.append("# ROI por tempo de execução (Back LIVE_OK; accounting por order_id)\n\n")
    md.append(f"- since_day (UTC): `{args.since_day}`\n")
    md.append(f"- source exec_jsonl: `{args.exec_jsonl}`\n")
    md.append(f"- source balance_csv: `{args.balance_csv}`\n")
    md.append(f"- n_exec_orders (Back LIVE_OK): `{len(exec_rows)}`\n")
    md.append(f"- n_orders no balance (após filtro de data): `{len(pnl_by_oid)}`\n")
    md.append(f"- n_orders com join exato: `{n_join}`\n")
    md.append(f"- n_total com join por order_id: `{len(rows)}`\n\n")
    if n_join == 0:
        ex_keys = list(exec_rows.keys())[:5]
        ac_keys = list(pnl_by_oid.keys())[:5]
        md.append("- _Diagnóstico_: join zerado. Verifique se `order_id` do balance CSV corresponde ao `raw.sent.id`/`raw.order_id` do executor.\n")
        md.append(f"- Exemplos order_id executor: `{ex_keys}`\n")
        md.append(f"- Exemplos order_id balance: `{ac_keys}`\n\n")

    md.append("| Grupo | n | stake_sum | ROI mean | ROIw |\n")
    md.append("|---|---:|---:|---:|---:|\n")
    md.append(f"| < 4s | {int(s_lt4.get('n') or 0)} | {_fmt_num(s_lt4.get('stake_sum'))} | {_fmt_pct(s_lt4.get('roi_mean_pct'))} | {_fmt_pct(s_lt4.get('roi_weighted_pct'))} |\n")
    md.append(f"| < 6s | {int(s_lt6.get('n') or 0)} | {_fmt_num(s_lt6.get('stake_sum'))} | {_fmt_pct(s_lt6.get('roi_mean_pct'))} | {_fmt_pct(s_lt6.get('roi_weighted_pct'))} |\n")
    md.append(f"| > 6s | {int(s_gt6.get('n') or 0)} | {_fmt_num(s_gt6.get('stake_sum'))} | {_fmt_pct(s_gt6.get('roi_mean_pct'))} | {_fmt_pct(s_gt6.get('roi_weighted_pct'))} |\n\n")

    def _delta_line(label: str, d: Dict[str, Optional[float]]) -> str:
        return (
            f"- {label}: ΔROI mean=`{_fmt_pct(d.get('delta_mean'))}` "
            f"| IC90=`{_fmt_pct(d.get('ci90_lb'))} .. {_fmt_pct(d.get('ci90_ub'))}` "
            f"| IC95=`{_fmt_pct(d.get('ci95_lb'))} .. {_fmt_pct(d.get('ci95_ub'))}`\n"
        )

    md.append("## Diferença estatística (bootstrap, delta de ROI mean)\n\n")
    md.append(_delta_line("<4s - >6s", d_lt4_vs_gt6))
    md.append(_delta_line("<6s - >6s", d_lt6_vs_gt6))
    md.append(_delta_line("<4s - <6s", d_lt4_vs_lt6))

    out_path = Path(str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(md), encoding="utf-8")
    print("".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
