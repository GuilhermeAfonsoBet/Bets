"""Resolve V1 report cutoff for V2 parity comparison."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .time_windows import ensure_utc, parse_dt


_GENERATED_RE = re.compile(
    r"Gerado em \(UTC\):\s*`([^`]+)`",
    re.IGNORECASE,
)
_DAY_RE = re.compile(
    r"Dia do relatório \(UTC\):\s*`(\d{8})`",
    re.IGNORECASE,
)


def find_v1_report_md(root: Path, report_date_utc: str) -> Optional[Path]:
    """Locate V1 report_daily.md for a cohort/generation day.

    Tries logs/daily_reports/{YYYYMMDD}/report_daily.md first (immutable day folder).
    """
    root = Path(root)
    day_compact = report_date_utc.replace("-", "")
    candidates = [
        root / "logs" / "daily_reports" / day_compact / "report_daily.md",
        root / "logs" / "daily_reports_smoke_2r" / day_compact / "report_daily.md",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def parse_v1_cutoff_from_md(md: str) -> Tuple[Optional[datetime], Optional[str]]:
    """Return (generated_at_utc, day_folder_yyyymmdd) from V1 markdown header."""
    gen = None
    m = _GENERATED_RE.search(md or "")
    if m:
        gen = parse_dt(m.group(1))
    day = None
    d = _DAY_RE.search(md or "")
    if d:
        day = d.group(1)
    return gen, day


def resolve_parity_cutoffs(
    *,
    root: Path,
    report_date_utc: str,
    v2_generated_at: datetime,
    v1_md_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build cutoff metadata.

    Parity rule: v2_comparison_cutoff_utc = v1_report_cutoff_utc when V1 is available.
    """
    path = v1_md_path or find_v1_report_md(root, report_date_utc)
    v1_cutoff = None
    v1_day = None
    parity_status = "PARITY_COMPARISON_UNAVAILABLE"
    v1_md = None
    if path and Path(path).exists():
        v1_md = Path(path).read_text(encoding="utf-8", errors="replace")
        v1_cutoff, v1_day = parse_v1_cutoff_from_md(v1_md)
        if v1_cutoff is not None:
            parity_status = "CUTOFF_ALIGNED"
        else:
            parity_status = "V1_CUTOFF_PARSE_FAILED"

    v2_gen = ensure_utc(v2_generated_at)
    comparison = ensure_utc(v1_cutoff) if v1_cutoff is not None else None

    return {
        "v1_report_path": str(path) if path else None,
        "v1_report_day_folder": v1_day,
        "v1_report_cutoff_utc": v1_cutoff.isoformat() if v1_cutoff else None,
        "v2_generated_at_utc": v2_gen.isoformat(),
        "v2_comparison_cutoff_utc": comparison.isoformat() if comparison else None,
        "parity_status": parity_status,
        "cutoffs_equal": bool(
            comparison is not None and v1_cutoff is not None and ensure_utc(v1_cutoff) == comparison
        ),
        "v1_md_text": v1_md,
        "note": (
            "Comparação de paridade usa o cutoff lógico do V1. "
            "Eventos após v1_report_cutoff_utc até v2_generated_at_utc são apenas diagnóstico intraday "
            "e não entram na paridade."
        ),
    }
