from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _parse_dt_any(value: Any) -> Optional[datetime]:
    s = str(value or "").strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _normalize_order_id(value: Any) -> Optional[str]:
    s = str(value or "").strip()
    if not s:
        return None
    if s.isdigit():
        return s
    try:
        f = float(s.replace(",", "."))
        if f >= 0 and abs(f - round(f)) <= 1e-9:
            return str(int(round(f)))
    except Exception:
        pass
    m = re.search(r"(\d{4,})", s)
    if m:
        return str(m.group(1))
    return None


def _extract_order_id(raw: Any) -> Optional[str]:
    if not isinstance(raw, dict):
        return None
    for key in ("order_id", "orderId", "id", "uuid", "uid"):
        oid = _normalize_order_id(raw.get(key))
        if oid:
            return oid
    sent = raw.get("sent")
    if isinstance(sent, dict):
        for key in ("order_id", "orderId", "id", "uuid", "uid"):
            oid = _normalize_order_id(sent.get(key))
            if oid:
                return oid
    data = raw.get("data")
    if isinstance(data, dict):
        for key in ("order_id", "orderId", "id", "uuid", "uid"):
            oid = _normalize_order_id(data.get(key))
            if oid:
                return oid
    return None


@dataclass
class ExecSample:
    order_id: str
    created_at: datetime
    stake: float
    roi: Optional[float] = None
    pnl: Optional[float] = None
    e2e_ms: Optional[float] = None
    d2s_ms: Optional[float] = None
    s2d_ms: Optional[float] = None
    regime: Optional[str] = None


def _read_exec_samples(exec_jsonl: Path, since_utc: datetime, regime_filter: str = "all") -> Dict[str, ExecSample]:
    out: Dict[str, ExecSample] = {}
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
        raw = res.get("raw") if isinstance(res.get("raw"), dict) else {}
        vs = raw.get("value_sizing") if isinstance(raw.get("value_sizing"), dict) else {}
        meta = req.get("meta") if isinstance(req.get("meta"), dict) else {}
        market = meta.get("market") if isinstance(meta.get("market"), dict) else {}
        regime = str(vs.get("market_regime") or market.get("regime") or "").strip().lower()
        if regime not in ("pre", "in"):
            regime = "in" if bool(market.get("is_live")) else "pre"
        if str(regime_filter or "all").strip().lower() in ("pre", "in"):
            if regime != str(regime_filter).strip().lower():
                continue
        created = _parse_dt_any(res.get("created_at") or req.get("created_at"))
        if created is None or created < since_utc:
            continue
        oid = _extract_order_id(raw)
        if not oid:
            continue
        timing = res.get("timing") if isinstance(res.get("timing"), dict) else {}
        e2e = _safe_float(timing.get("e2e_total_ms"))
        d2s = _safe_float(timing.get("detect_to_submit_ms"))
        s2d = _safe_float(timing.get("call_to_done_ms"))
        if e2e is None:
            if d2s is not None and s2d is not None:
                e2e = float(d2s + s2d)
            else:
                continue
        sent = raw.get("sent") if isinstance(raw.get("sent"), dict) else {}
        stake = _safe_float(sent.get("stake"))
        if stake is None:
            pol = res.get("policy") if isinstance(res.get("policy"), dict) else {}
            stake = _safe_float(pol.get("stake_requested"))
        if stake is None or stake <= 0:
            continue
        rec = ExecSample(
            order_id=str(oid),
            created_at=created,
            stake=float(stake),
            e2e_ms=float(e2e),
            d2s_ms=float(d2s) if d2s is not None else None,
            s2d_ms=float(s2d) if s2d is not None else None,
            regime=str(regime),
        )
        old = out.get(rec.order_id)
        if old is None or rec.created_at >= old.created_at:
            out[rec.order_id] = rec
    return out


def _read_balance_pnl(balance_csv: Path, since_day: str) -> Dict[str, float]:
    return _read_balance_pnl_many([balance_csv], since_day=since_day)


def _pick_balance_files(*, balance_csv: str, balance_dir: str) -> List[Path]:
    raw_csv = str(balance_csv or "").strip()
    if raw_csv:
        p = Path(raw_csv)
        return [p] if p.exists() else []

    out: List[Path] = []
    root = Path(str(balance_dir or "logs/accounting"))
    if root.exists():
        out.extend(sorted(root.glob("*__balance.csv")))
        # fallback comum em alguns ambientes
        latest = root / "latest__balance.csv"
        if latest.exists() and latest not in out:
            out.append(latest)
    return out


def _read_balance_pnl_many(balance_csvs: Iterable[Path], since_day: str) -> Dict[str, float]:
    """
    Lê vários balance.csv e deduplica linhas repetidas entre snapshots por:
      (order_id, timestamp, pnl)
    Isso evita double-count quando o mesmo extrato diário é reexportado.
    """
    out: Dict[str, float] = {}
    seen_rows: set[Tuple[str, str, float]] = set()

    for balance_csv in balance_csvs:
        if not balance_csv.exists():
            continue
        with balance_csv.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
            reader = csv.DictReader(fh)
            cols = [str(c or "") for c in (reader.fieldnames or [])]
            if not cols:
                continue

            def pick(*needles: str) -> Optional[str]:
                for needle in needles:
                    n = needle.lower()
                    for col in cols:
                        c = col.lower()
                        if c == n or c.startswith(n) or n in c:
                            return col
                return None

            dt_col = pick("post date", "post_date", "date", "settled", "closed", "time")
            oid_col = pick("order_id", "order id", "order", "bet id", "bet_id", "id")
            pnl_col = pick("amount", "profit_loss", "profit", "p&l", "pnl", "net", "pl")
            if not dt_col or not oid_col or not pnl_col:
                continue

            for row in reader:
                if not isinstance(row, dict):
                    continue
                oid = _normalize_order_id(row.get(oid_col))
                if not oid:
                    continue
                dt = _parse_dt_any(row.get(dt_col))
                if dt is None or dt.date().isoformat() < since_day:
                    continue
                pnl = _safe_float(row.get(pnl_col))
                if pnl is None:
                    continue
                sig = (str(oid), dt.isoformat(), float(pnl))
                if sig in seen_rows:
                    continue
                seen_rows.add(sig)
                out[oid] = float(out.get(oid) or 0.0) + float(pnl)
    return out


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def _quantile(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None
    ys = sorted(xs)
    idx = max(0, min(len(ys) - 1, int(round((len(ys) - 1) * float(q)))))
    return float(ys[idx])


def _bucket_rows(rows: List[ExecSample], field: str, cuts: List[float]) -> List[Tuple[str, List[ExecSample]]]:
    out: List[Tuple[str, List[ExecSample]]] = []
    if len(cuts) < 2:
        return out
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        label = f"[{int(lo)},{int(hi)})"
        bucket = []
        for r in rows:
            v = getattr(r, field)
            if v is None:
                continue
            if float(lo) <= float(v) < float(hi):
                bucket.append(r)
        out.append((label, bucket))
    return out


def _weighted_roi_pct(rows: Iterable[ExecSample]) -> Optional[float]:
    stake = 0.0
    pnl = 0.0
    for r in rows:
        if r.pnl is None:
            continue
        stake += float(r.stake or 0.0)
        pnl += float(r.pnl or 0.0)
    if stake <= 0:
        return None
    return (pnl / stake) * 100.0


def _solve_linear_system(a: List[List[float]], b: List[float]) -> Optional[List[float]]:
    n = len(a)
    if n == 0 or any(len(row) != n for row in a) or len(b) != n:
        return None
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            return None
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
        pivot_val = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= pivot_val
        for i in range(n):
            if i == col:
                continue
            factor = m[i][col]
            if abs(factor) < 1e-15:
                continue
            for j in range(col, n + 1):
                m[i][j] -= factor * m[col][j]
    return [m[i][n] for i in range(n)]


def _ols(rows: List[ExecSample]) -> Optional[Dict[str, float]]:
    # y = roi_pct; X = [1, e2e_s, d2s_s, s2d_s]
    mat: List[List[float]] = []
    vec: List[float] = []
    for r in rows:
        if r.roi is None or r.e2e_ms is None or r.d2s_ms is None or r.s2d_ms is None:
            continue
        mat.append([1.0, r.e2e_ms / 1000.0, r.d2s_ms / 1000.0, r.s2d_ms / 1000.0])
        vec.append(float(r.roi))
    n = len(mat)
    p = 4
    if n < max(20, p + 2):
        return None

    xtx = [[0.0 for _ in range(p)] for _ in range(p)]
    xty = [0.0 for _ in range(p)]
    for x, y in zip(mat, vec):
        for i in range(p):
            xty[i] += x[i] * y
            for j in range(p):
                xtx[i][j] += x[i] * x[j]
    beta = _solve_linear_system(xtx, xty)
    if not beta:
        return None
    return {
        "intercept": float(beta[0]),
        "e2e_per_sec_roi_pp": float(beta[1]),
        "d2s_per_sec_roi_pp": float(beta[2]),
        "s2d_per_sec_roi_pp": float(beta[3]),
    }


def _fmt(x: Optional[float], nd: int = 3) -> str:
    if x is None:
        return "—"
    return f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Atribuição de ROI por componentes de latência (e2e, detect_to_submit, submit_to_done)."
    )
    ap.add_argument("--exec-jsonl", default="logs/executor_live.jsonl")
    ap.add_argument("--balance-csv", default="", help="Opcional: caminho único para balance.csv.")
    ap.add_argument("--balance-dir", default="logs/accounting", help="Diretório para buscar *__balance.csv.")
    ap.add_argument("--since-day", default=(datetime.now(timezone.utc) - timedelta(days=14)).date().isoformat())
    ap.add_argument("--regime", default="all", choices=["pre", "in", "all"], help="Filtra regime de mercado no executor.")
    ap.add_argument("--out-md", default="logs/e2e_roi_attribution.md")
    ap.add_argument("--out-json", default="logs/e2e_roi_attribution.json")
    args = ap.parse_args()

    since_utc = _parse_dt_any(f"{args.since_day}T00:00:00+00:00")
    if since_utc is None:
        raise SystemExit(f"since_day inválido: {args.since_day}")

    exec_rows = _read_exec_samples(
        Path(str(args.exec_jsonl)),
        since_utc=since_utc,
        regime_filter=str(getattr(args, "regime", "all")),
    )
    balance_files = _pick_balance_files(balance_csv=str(args.balance_csv), balance_dir=str(args.balance_dir))
    pnl_by_oid = _read_balance_pnl_many(balance_files, since_day=str(args.since_day))

    rows: List[ExecSample] = []
    for oid, sample in exec_rows.items():
        pnl = pnl_by_oid.get(oid)
        if pnl is None:
            continue
        sample.pnl = float(pnl)
        sample.roi = (float(sample.pnl) / float(sample.stake) * 100.0) if sample.stake > 0 else None
        rows.append(sample)

    rows.sort(key=lambda r: r.created_at)
    n = len(rows)
    if n == 0:
        report = {
            "since_day": str(args.since_day),
            "regime": str(args.regime),
            "balance_files_n": len(balance_files),
            "n_exec_live_ok": len(exec_rows),
            "n_joined": 0,
            "note": "Sem join entre executor_live.jsonl e balance.csv para o período.",
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        Path(args.out_md).write_text(
            "# E2E ROI attribution\n\nSem dados suficientes (join=0). Verifique order_id/stake em executor e balance.csv.\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False))
        return 0

    e2e_pairs = [(float(r.e2e_ms), float(r.roi)) for r in rows if r.e2e_ms is not None and r.roi is not None]
    d2s_pairs = [(float(r.d2s_ms), float(r.roi)) for r in rows if r.d2s_ms is not None and r.roi is not None]
    s2d_pairs = [(float(r.s2d_ms), float(r.roi)) for r in rows if r.s2d_ms is not None and r.roi is not None]
    e2e_vals = [x for x, _ in e2e_pairs]
    d2s_vals = [x for x, _ in d2s_pairs]
    s2d_vals = [x for x, _ in s2d_pairs]

    corr_e2e = _pearson([x for x, _ in e2e_pairs], [y for _, y in e2e_pairs])
    corr_d2s = _pearson([x for x, _ in d2s_pairs], [y for _, y in d2s_pairs])
    corr_s2d = _pearson([x for x, _ in s2d_pairs], [y for _, y in s2d_pairs])

    e2e_q = [_quantile(e2e_vals, q) for q in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    e2e_cuts = [float(x) for x in e2e_q if x is not None]
    if len(e2e_cuts) >= 2 and e2e_cuts[-1] == e2e_cuts[0]:
        e2e_cuts = [e2e_cuts[0], e2e_cuts[-1] + 1.0]
    elif len(e2e_cuts) >= 2:
        e2e_cuts[-1] += 1.0
    buckets = _bucket_rows(rows, "e2e_ms", e2e_cuts) if len(e2e_cuts) >= 2 else []

    ols = _ols(rows)

    by_bucket = []
    for label, group in buckets:
        by_bucket.append(
            {
                "bucket": label,
                "n": len(group),
                "roi_weighted_pct": _weighted_roi_pct(group),
                "e2e_p50_ms": _quantile([float(x.e2e_ms) for x in group if x.e2e_ms is not None], 0.5),
                "d2s_p50_ms": _quantile([float(x.d2s_ms) for x in group if x.d2s_ms is not None], 0.5),
                "s2d_p50_ms": _quantile([float(x.s2d_ms) for x in group if x.s2d_ms is not None], 0.5),
            }
        )

    result = {
        "since_day": str(args.since_day),
        "regime": str(args.regime),
        "balance_files_n": len(balance_files),
        "n_exec_live_ok": len(exec_rows),
        "n_joined": n,
        "roi_weighted_pct": _weighted_roi_pct(rows),
        "corr_roi_vs_e2e": corr_e2e,
        "corr_roi_vs_d2s": corr_d2s,
        "corr_roi_vs_s2d": corr_s2d,
        "ols_roi_pct": ols,
        "by_e2e_bucket": by_bucket,
    }

    md: List[str] = []
    md.append("# Atribuição ROI × latência (E2E)\n")
    md.append(f"- since_day: `{args.since_day}`")
    md.append(f"- regime: `{args.regime}`")
    md.append(f"- balance_files_n: **{len(balance_files)}**")
    md.append(f"- LIVE_OK(back) no executor: **{len(exec_rows)}**")
    md.append(f"- join executor×balance: **{n}**")
    md.append(f"- ROI ponderado (join): **{_fmt(result['roi_weighted_pct'], 2)}%**\n")
    md.append("## Correlações brutas (não-causais)")
    md.append(f"- corr(ROI, e2e_ms): `{_fmt(corr_e2e, 4)}`")
    md.append(f"- corr(ROI, detect_to_submit_ms): `{_fmt(corr_d2s, 4)}`")
    md.append(f"- corr(ROI, submit_to_done_ms): `{_fmt(corr_s2d, 4)}`\n")
    md.append("## Buckets por E2E")
    md.append("| bucket_e2e_ms | n | roi_weighted_pct | e2e_p50 | d2s_p50 | s2d_p50 |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for b in by_bucket:
        md.append(
            f"| {b['bucket']} | {b['n']} | {_fmt(b['roi_weighted_pct'], 2)} | "
            f"{_fmt(b['e2e_p50_ms'], 0)} | {_fmt(b['d2s_p50_ms'], 0)} | {_fmt(b['s2d_p50_ms'], 0)} |"
        )
    md.append("")
    md.append("## OLS simples (ROI percentual)")
    md.append("Modelo: `ROI% = b0 + b1*e2e_s + b2*d2s_s + b3*s2d_s`")
    if ols:
        md.append(f"- b1 (e2e por +1s): `{_fmt(ols['e2e_per_sec_roi_pp'], 4)} pp`")
        md.append(f"- b2 (d2s por +1s): `{_fmt(ols['d2s_per_sec_roi_pp'], 4)} pp`")
        md.append(f"- b3 (s2d por +1s): `{_fmt(ols['s2d_per_sec_roi_pp'], 4)} pp`")
        md.append(f"- impacto estimado de +2s no executor (s2d): `{_fmt(2.0 * ols['s2d_per_sec_roi_pp'], 4)} pp`")
    else:
        md.append("- amostra insuficiente para OLS robusto.")
    md.append("")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.out_md).write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
