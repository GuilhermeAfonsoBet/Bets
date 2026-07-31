from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from .accounting_monitor import AccountingConfig, run_monitor
from .accounting_report import compute_pnl_report
from .accounting_status import normalize_jsonl_path


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _latest_balance_csv(out_dir: Path) -> Optional[Path]:
    cands = sorted(out_dir.glob("*__balance.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


@dataclass
class DailyCfg:
    out_dir: Path
    jsonl: Path
    tz_name: str = "America/Sao_Paulo"
    report_out: Path = Path("logs/accounting_daily_report.json")
    print_json: bool = True


async def run_daily(cfg: DailyCfg) -> Dict[str, Any]:
    # 1) snapshot CSVs (once)
    mon = AccountingConfig(out_dir=cfg.out_dir, jsonl_path=cfg.jsonl, once=True)
    await run_monitor(mon)

    # 2) escolher último balance
    bal = _latest_balance_csv(cfg.out_dir)
    if not bal:
        return {"ts": _utcnow(), "error": "NO_BALANCE_CSV", "out_dir": str(cfg.out_dir)}

    # 3) calcular P&L + saldo
    tz = timezone.utc
    try:
        from zoneinfo import ZoneInfo  # type: ignore

        tz = ZoneInfo(cfg.tz_name)
    except Exception:
        # fallback BRT
        tz = timezone.utc

    rep = compute_pnl_report(bal, tz=tz)

    now = datetime.now(timezone.utc).astimezone(tz)
    today = now.date().isoformat()
    monthk = f"{now.year:04d}-{now.month:02d}"

    def _week_start(d):
        # segunda
        import datetime as _dt

        return d - _dt.timedelta(days=int(d.weekday()))

    week0 = _week_start(now.date()).isoformat()

    pnl_today = float((rep.pnl_by_day or {}).get(today, 0.0))
    pnl_week = float(sum(v for k, v in (rep.pnl_by_day or {}).items() if week0 <= k <= today))
    pnl_month = float((rep.pnl_by_month or {}).get(monthk, 0.0))
    closed = {k: float(v) for k, v in sorted((rep.pnl_by_month or {}).items()) if k < monthk}

    pnl_today_f = None
    pnl_week_f = None
    pnl_month_f = None
    closed_f = None
    if rep.pnl_by_day_filtered and rep.pnl_by_month_filtered:
        pnl_today_f = float(rep.pnl_by_day_filtered.get(today, 0.0))
        pnl_week_f = float(sum(v for k, v in rep.pnl_by_day_filtered.items() if week0 <= k <= today))
        pnl_month_f = float(rep.pnl_by_month_filtered.get(monthk, 0.0))
        closed_f = {k: float(v) for k, v in sorted(rep.pnl_by_month_filtered.items()) if k < monthk}

    # Para uso em relatórios operacionais (ex.: daily_full_report),
    # expomos uma janela curta por dia (best-effort) para evitar P&L "falso"
    # quando não há cobertura de ROI via placares/audit.
    try:
        from datetime import timedelta as _td

        def _recent_days_map(src: Optional[Dict[str, float]], *, ndays: int) -> Dict[str, float]:
            if not src:
                return {}
            outm: Dict[str, float] = {}
            d0 = now.date() - _td(days=max(0, int(ndays) - 1))
            d = d0
            while d <= now.date():
                k = d.isoformat()
                if k in src:
                    outm[k] = float(src.get(k) or 0.0)
                d += _td(days=1)
            return outm

        pnl_by_day_recent = _recent_days_map(rep.pnl_by_day, ndays=21)
        pnl_by_day_filtered_recent = _recent_days_map(rep.pnl_by_day_filtered, ndays=21) if rep.pnl_by_day_filtered else {}
    except Exception:
        pnl_by_day_recent = {}
        pnl_by_day_filtered_recent = {}

    out: Dict[str, Any] = {
        "ts": _utcnow(),
        "tz": cfg.tz_name,
        "balance_csv": str(bal),
        "kind": rep.kind,
        "balance_current": rep.balance_current,
        "pnl_today": pnl_today,
        "pnl_week": pnl_week,
        "pnl_month": pnl_month,
        "closed_months": closed,
        "pnl_filtered_today": pnl_today_f,
        "pnl_filtered_week": pnl_week_f,
        "pnl_filtered_month": pnl_month_f,
        "closed_months_filtered": closed_f,
        "pnl_by_day_recent": pnl_by_day_recent,
        "pnl_by_day_filtered_recent": pnl_by_day_filtered_recent,
        "by_type": rep.by_type,
        "rows": rep.rows,
        "dt_col": rep.dt_col,
        "pnl_col": rep.pnl_col,
        "type_col": rep.type_col,
    }
    cfg.report_out.parent.mkdir(parents=True, exist_ok=True)
    cfg.report_out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Baixa CSV (balance/open-stakes) e gera resumo de saldo/P&L.")
    ap.add_argument("--out-dir", default=os.getenv("ACCOUNTING_OUT_DIR", "logs/accounting"))
    ap.add_argument("--jsonl", default=normalize_jsonl_path(os.getenv("ACCOUNTING_JSONL", "logs/accounting_snapshots.jsonl")))
    ap.add_argument("--tz", default=os.getenv("REPORT_TZ", "America/Sao_Paulo"))
    ap.add_argument("--report-out", default=os.getenv("ACCOUNTING_DAILY_REPORT_OUT", "logs/accounting_daily_report.json"))
    ap.add_argument("--no-print", action="store_true", default=False)
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))

    cfg = DailyCfg(
        out_dir=Path(str(args.out_dir)),
        jsonl=Path(str(args.jsonl)),
        tz_name=str(args.tz),
        report_out=Path(str(args.report_out)),
        print_json=(not bool(args.no_print)),
    )
    import asyncio
    try:
        out = asyncio.run(run_daily(cfg))
    except Exception as e:
        err = {"ts": _utcnow(), "error": str(e)[:300]}
        print(json.dumps(err, ensure_ascii=False))
        return 2

    if cfg.print_json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

