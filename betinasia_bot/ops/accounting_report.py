from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_date(dt: datetime, tz: timezone) -> date:
    return dt.astimezone(tz).date()


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        s = s.replace(",", ".")
        s = re.sub(r"[^0-9.\-]", "", s)
        if s in ("", "-", ".", "-."):
            return None
        return float(s)
    except Exception:
        return None


def _parse_dt_any(s: str) -> Optional[datetime]:
    try:
        t = str(s or "").strip()
        if not t:
            return None
        # ISO
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(t)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass
        # common formats
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y", "%d/%m/%Y %H:%M:%S"):
            try:
                dt = datetime.strptime(t, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except Exception:
                continue
    except Exception:
        return None
    return None


def _pick_col(cols: List[str], needles: Iterable[str]) -> Optional[str]:
    """
    Seleciona uma coluna por heurística, mas preferindo:
    1) match exato (case-insensitive)
    2) prefix (ex.: 'post date' vs 'post date (utc)')
    3) contains (fallback)
    """
    cols_map = {c.lower(): c for c in cols}
    cols_l = list(cols_map.keys())
    for n in needles:
        n = str(n).lower()
        if n in cols_map:
            return cols_map[n]
        for cl in cols_l:
            if cl.startswith(n):
                return cols_map[cl]
        for cl in cols_l:
            if n in cl:
                return cols_map[cl]
    return None


@dataclass
class Report:
    tz: timezone
    rows: int
    kind: str
    dt_col: Optional[str]
    pnl_col: Optional[str]
    pnl_by_day: Dict[str, float]
    pnl_by_month: Dict[str, float]
    # extras (balance ledger)
    balance_current: Optional[float] = None
    type_col: Optional[str] = None
    by_type: Optional[Dict[str, float]] = None
    pnl_by_day_filtered: Optional[Dict[str, float]] = None
    pnl_by_month_filtered: Optional[Dict[str, float]] = None


def compute_pnl_report(csv_path: Path, *, tz: timezone) -> Report:
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        cols_l = [c.lower() for c in cols]
        is_balance_ledger = ("amount" in cols_l) and ("balance" in cols_l) and (("post date" in cols_l) or ("post_date" in cols_l))

        kind = "generic"
        dt_col = _pick_col(cols, ("post date", "post_date", "settled", "settle", "closed", "date", "time", "placement"))
        pnl_col = _pick_col(cols, ("profit_loss", "profit", "p&l", "pnl", "net", "pl"))
        type_col = _pick_col(cols, ("type",))
        balance_col = _pick_col(cols, ("balance",))

        if is_balance_ledger:
            kind = "balance_ledger"
            dt_col = _pick_col(cols, ("post date", "post_date", "date", "time"))
            pnl_col = _pick_col(cols, ("amount",)) or pnl_col

        pnl_by_day: Dict[str, float] = defaultdict(float)
        pnl_by_month: Dict[str, float] = defaultdict(float)
        pnl_by_day_f: Dict[str, float] = defaultdict(float)
        pnl_by_month_f: Dict[str, float] = defaultdict(float)
        by_type: Dict[str, float] = defaultdict(float)
        balance_current: Optional[float] = None
        n = 0
        for row in reader:
            n += 1
            if not isinstance(row, dict):
                continue
            pnl = _safe_float(row.get(pnl_col)) if pnl_col else None
            if pnl is None:
                continue
            dt = _parse_dt_any(str(row.get(dt_col) or "")) if dt_col else None
            if dt is None:
                # sem data: ignora do agregado temporal
                continue
            d = _to_date(dt, tz)
            dayk = d.isoformat()
            monthk = f"{d.year:04d}-{d.month:02d}"
            pnl_by_day[dayk] += float(pnl)
            pnl_by_month[monthk] += float(pnl)

            if kind == "balance_ledger":
                typ = str(row.get(type_col) or "").strip() if type_col else ""
                if typ:
                    by_type[typ] += float(pnl)
                if balance_col:
                    b = _safe_float(row.get(balance_col))
                    if b is not None:
                        balance_current = float(b)

                # filtro “P&L-like”: exclui movimentações que não são resultado de aposta
                tl = typ.lower()
                exclude = any(k in tl for k in ("deposit", "withdraw", "transfer", "top", "payment", "adjust", "bonus"))
                if not exclude:
                    pnl_by_day_f[dayk] += float(pnl)
                    pnl_by_month_f[monthk] += float(pnl)

        return Report(
            tz=tz,
            rows=int(n),
            kind=str(kind),
            dt_col=dt_col,
            pnl_col=pnl_col,
            pnl_by_day=dict(pnl_by_day),
            pnl_by_month=dict(pnl_by_month),
            balance_current=balance_current,
            type_col=type_col,
            by_type=dict(by_type) if by_type else None,
            pnl_by_day_filtered=(dict(pnl_by_day_f) if pnl_by_day_f else None),
            pnl_by_month_filtered=(dict(pnl_by_month_f) if pnl_by_month_f else None),
        )


def _week_start(d: date) -> date:
    # semana começando na segunda
    return d - timedelta(days=int(d.weekday()))


def main() -> int:
    ap = argparse.ArgumentParser(description="Resumo de P&L por dia/semana/mês a partir de CSV do accounting.")
    ap.add_argument("--csv", required=True, help="Path do CSV (export do accounting).")
    ap.add_argument("--tz", default=os.getenv("REPORT_TZ", "America/Sao_Paulo"), help="Timezone para agregações.")
    args = ap.parse_args()

    # Python stdlib não tem tz database sem zoneinfo (py3.9+ tem).
    tz = timezone(timedelta(hours=-3))
    try:
        from zoneinfo import ZoneInfo  # type: ignore

        tz = ZoneInfo(str(args.tz))
    except Exception:
        pass

    p = Path(str(args.csv))
    rep = compute_pnl_report(p, tz=tz)

    today = _to_date(_utcnow(), tz)
    week0 = _week_start(today)
    monthk = f"{today.year:04d}-{today.month:02d}"

    pnl_today = float(rep.pnl_by_day.get(today.isoformat(), 0.0))
    pnl_week = float(sum(v for k, v in rep.pnl_by_day.items() if week0.isoformat() <= k <= today.isoformat()))
    pnl_month = float(rep.pnl_by_month.get(monthk, 0.0))

    # meses anteriores fechados
    closed = {k: float(v) for k, v in sorted(rep.pnl_by_month.items()) if k < monthk}

    pnl_today_f = None
    pnl_week_f = None
    pnl_month_f = None
    closed_f = None
    if rep.pnl_by_day_filtered and rep.pnl_by_month_filtered:
        pnl_today_f = float(rep.pnl_by_day_filtered.get(today.isoformat(), 0.0))
        pnl_week_f = float(sum(v for k, v in rep.pnl_by_day_filtered.items() if week0.isoformat() <= k <= today.isoformat()))
        pnl_month_f = float(rep.pnl_by_month_filtered.get(monthk, 0.0))
        closed_f = {k: float(v) for k, v in sorted(rep.pnl_by_month_filtered.items()) if k < monthk}

    out = {
        "csv": str(p),
        "rows": rep.rows,
        "kind": rep.kind,
        "dt_col": rep.dt_col,
        "pnl_col": rep.pnl_col,
        "tz": str(args.tz),
        "pnl_today": pnl_today,
        "pnl_week": pnl_week,
        "pnl_month": pnl_month,
        "closed_months": closed,
        "balance_current": rep.balance_current,
        "type_col": rep.type_col,
        "by_type": rep.by_type,
        "pnl_filtered_today": pnl_today_f,
        "pnl_filtered_week": pnl_week_f,
        "pnl_filtered_month": pnl_month_f,
        "closed_months_filtered": closed_f,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

