#!/usr/bin/env python3
"""
Gera heatmap tabular (semana x bucket de E2E) para ROI realizado.

Fluxo suportado:
1) CSV unico ja "merged" com colunas de P&L/stake/e2e_total_ms.
2) Dois CSVs: base de P&L/stake + CSV de E2E por audit_id.

Saidas:
- CSV long (uma linha por semana x bucket)
- CSV pivot (uma linha por semana, colunas de buckets)
- Markdown com tabela estilo heatmap textual

Observacao:
- O script usa ROI realizado como `pnl / turnover`.
- Por padrao, aplica o recorte operacional back + pre + slippage_pre_pct < 0
  quando essas colunas existirem (modo leniente para bases pre-filtradas).
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


AUDIT_ID_COLS = ("audit_id", "id", "bet_id", "opportunity_id")
TS_COLS = (
    "ts_utc",
    "audited_utc",
    "audited_at",
    "audited_at_utc",
    "timestamp_utc",
    "timestamp",
    "ts",
    "date_utc",
    "datetime_utc",
    "event_ts",
    "event_ts_utc",
    "created_at",
    "updated_at",
)
E2E_COLS = (
    "e2e_total_ms",
    "e2e_ms",
    "lag_total_ms",
    "lag_e2e_ms",
    "detect_to_submit_ms",
    "executor_submit_to_done_ms",
    "call_to_done_ms",
)
STAKE_COLS = (
    "stake_exec",
    "stake_real",
    "stake",
    "turnover_unit",
    "turnover",
    "turn",
    "stake_usd",
    "amount",
)
PNL_COLS = (
    "pnl_real",
    "pnl_real_usd",
    "pnl_ledger",
    "pnl_usd",
    "pnl",
    "profit_real",
    "result_real",
    "result",
    "daily_unit_pnl",
    "profit",
)
SIDE_COLS = ("side", "side_lc", "bet_side", "selection_side")
REGIME_COLS = ("market_regime", "market_period", "market_period_lc", "regime")
SLIP_COLS = ("slippage_pre_pct", "slip_pre_pct", "slippage_pct_pre")

TRUE_VALUES = {"1", "true", "t", "yes", "y", "sim"}
FALSE_VALUES = {"0", "false", "f", "no", "n", "nao", "não"}


@dataclass
class CellAgg:
    n: int = 0
    turnover: float = 0.0
    pnl: float = 0.0
    e2e_vals: List[float] = field(default_factory=list)

    def add(self, stake: float, pnl: float, e2e_ms: float) -> None:
        self.n += 1
        self.turnover += stake
        self.pnl += pnl
        self.e2e_vals.append(e2e_ms)

    @property
    def roi_pct(self) -> Optional[float]:
        if self.turnover <= 0:
            return None
        return 100.0 * self.pnl / self.turnover


def _first_nonempty(row: Dict[str, str], candidates: Sequence[str]) -> Optional[str]:
    for key in candidates:
        if key in row:
            raw = row.get(key)
            if raw is None:
                continue
            txt = str(raw).strip()
            if txt != "":
                return txt
    return None


def _to_float(raw: Optional[str]) -> Optional[float]:
    if raw is None:
        return None
    txt = str(raw).strip().replace(",", ".")
    if txt == "":
        return None
    try:
        return float(txt)
    except ValueError:
        return None


def _to_int(raw: Optional[str]) -> Optional[int]:
    if raw is None:
        return None
    txt = str(raw).strip()
    if txt == "":
        return None
    try:
        return int(txt)
    except ValueError:
        try:
            return int(float(txt.replace(",", ".")))
        except ValueError:
            return None


def _to_dt(raw: Optional[str]) -> Optional[datetime]:
    if raw is None:
        return None
    txt = str(raw).strip()
    if txt == "":
        return None
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(txt)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_bool(raw: Optional[str]) -> Optional[bool]:
    if raw is None:
        return None
    txt = str(raw).strip().lower()
    if txt in TRUE_VALUES:
        return True
    if txt in FALSE_VALUES:
        return False
    return None


def _is_back(row: Dict[str, str]) -> Optional[bool]:
    side = _first_nonempty(row, SIDE_COLS)
    if side is None:
        return None
    s = side.strip().lower()
    if s == "back":
        return True
    if s == "lay":
        return False
    return None


def _is_pre(row: Dict[str, str]) -> Optional[bool]:
    regime = _first_nonempty(row, REGIME_COLS)
    if regime is not None:
        r = regime.strip().lower()
        if r in {"pre", "prematch", "pre_match", "pre-match"}:
            return True
        if r in {"live", "in", "inplay", "in_play", "in-match", "in_match"}:
            return False
    is_live = _to_bool(row.get("is_live"))
    if is_live is None:
        return None
    return not is_live


def _is_slipneg(row: Dict[str, str]) -> Optional[bool]:
    slip = _to_float(_first_nonempty(row, SLIP_COLS))
    if slip is None:
        return None
    return slip < 0


def _parse_edges(raw: str) -> List[float]:
    vals = []
    for part in str(raw).split(","):
        txt = part.strip()
        if not txt:
            continue
        v = _to_float(txt)
        if v is None:
            raise ValueError(f"bucket invalido: '{txt}'")
        vals.append(v)
    uniq = sorted(set(vals))
    if len(uniq) < 2:
        raise ValueError("forneca ao menos dois limites em --buckets-ms")
    return uniq


def _sec_label(ms: float) -> str:
    sec = ms / 1000.0
    if abs(sec - int(sec)) < 1e-9:
        return str(int(sec))
    return f"{sec:.3f}".rstrip("0").rstrip(".")


def _build_bucket_labels(edges_ms: Sequence[float]) -> List[str]:
    labels: List[str] = [f"<{_sec_label(edges_ms[0])}s"]
    for i in range(len(edges_ms) - 1):
        labels.append(f"{_sec_label(edges_ms[i])}-{_sec_label(edges_ms[i + 1])}s")
    labels.append(f">{_sec_label(edges_ms[-1])}s")
    return labels


def _bucket_of(e2e_ms: float, edges_ms: Sequence[float], labels: Sequence[str]) -> str:
    if e2e_ms < edges_ms[0]:
        return labels[0]
    for i in range(len(edges_ms) - 1):
        if edges_ms[i] <= e2e_ms < edges_ms[i + 1]:
            return labels[i + 1]
    return labels[-1]


def _iso_week(dt: datetime) -> Tuple[int, int]:
    iso = dt.isocalendar()
    return int(iso.year), int(iso.week)


def _week_key(iso_year: int, iso_week: int) -> str:
    return f"{iso_year}-W{iso_week:02d}"


def _pct(vals: Sequence[float], q: float) -> Optional[float]:
    if not vals:
        return None
    arr = sorted(float(v) for v in vals)
    if len(arr) == 1:
        return arr[0]
    pos = (len(arr) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(arr) - 1)
    frac = pos - lo
    return arr[lo] * (1.0 - frac) + arr[hi] * frac


def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v:.2f}%"


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        return [dict(r) for r in rd]


def _nonempty_count(rows: Sequence[Dict[str, str]], col: str) -> int:
    n = 0
    for row in rows:
        raw = row.get(col)
        if raw is None:
            continue
        if str(raw).strip() != "":
            n += 1
    return n


def _best_col_report(rows: Sequence[Dict[str, str]], candidates: Sequence[str], label: str) -> str:
    hits: List[Tuple[int, str]] = []
    for c in candidates:
        k = _nonempty_count(rows, c)
        if k > 0:
            hits.append((k, c))
    if not hits:
        return f"{label}: nenhum candidato com dados"
    hits.sort(reverse=True)
    top = ", ".join(f"{c}={k}" for k, c in hits[:5])
    return f"{label}: {top}"


def _load_e2e_map(e2e_csv: Path) -> Dict[int, Tuple[float, Optional[datetime]]]:
    rows = _read_csv(e2e_csv)
    by_aid: Dict[int, Tuple[float, Optional[datetime]]] = {}
    for row in rows:
        aid = _to_int(_first_nonempty(row, AUDIT_ID_COLS))
        e2e = _to_float(_first_nonempty(row, E2E_COLS))
        if aid is None or e2e is None or e2e <= 0:
            continue
        ts = _to_dt(_first_nonempty(row, TS_COLS))
        prev = by_aid.get(aid)
        if prev is None:
            by_aid[aid] = (e2e, ts)
            continue
        _, prev_ts = prev
        if prev_ts is None and ts is not None:
            by_aid[aid] = (e2e, ts)
        elif prev_ts is not None and ts is not None and ts > prev_ts:
            by_aid[aid] = (e2e, ts)
    return by_aid


def _resolve_row_fields(
    row: Dict[str, str],
    e2e_map: Optional[Dict[int, Tuple[float, Optional[datetime]]]],
) -> Tuple[
    Optional[Tuple[int, datetime, float, float, float, Optional[bool], Optional[bool], Optional[bool]]],
    Optional[str],
]:
    aid = _to_int(_first_nonempty(row, AUDIT_ID_COLS))
    if aid is None:
        return None, "missing_audit_id"

    e2e: Optional[float] = None
    ts: Optional[datetime] = _to_dt(_first_nonempty(row, TS_COLS))
    if e2e_map is not None:
        mapped = e2e_map.get(aid)
        if mapped is not None:
            e2e = mapped[0]
            if ts is None and mapped[1] is not None:
                ts = mapped[1]
    if e2e is None:
        e2e = _to_float(_first_nonempty(row, E2E_COLS))

    stake = _to_float(_first_nonempty(row, STAKE_COLS))
    pnl = _to_float(_first_nonempty(row, PNL_COLS))

    if ts is None:
        return None, "missing_timestamp"
    if e2e is None:
        return None, "missing_e2e"
    if stake is None:
        return None, "missing_stake"
    if pnl is None:
        return None, "missing_pnl"
    if e2e <= 0:
        return None, "invalid_e2e_nonpositive"
    if stake <= 0:
        return None, "invalid_stake_nonpositive"

    return (
        (
            aid,
            ts,
            e2e,
            stake,
            pnl,
            _is_back(row),
            _is_pre(row),
            _is_slipneg(row),
        ),
        None,
    )


def _write_long_csv(
    out_path: Path,
    week_order: Sequence[Tuple[int, int]],
    labels: Sequence[str],
    agg: Dict[Tuple[int, int, str], CellAgg],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(
            [
                "iso_week",
                "bucket",
                "n",
                "turnover",
                "pnl",
                "roi_pct",
                "e2e_p50_ms",
                "e2e_p90_ms",
            ]
        )
        for yw in week_order:
            for b in labels:
                cell = agg.get((yw[0], yw[1], b), CellAgg())
                wr.writerow(
                    [
                        _week_key(yw[0], yw[1]),
                        b,
                        cell.n,
                        f"{cell.turnover:.6f}",
                        f"{cell.pnl:.6f}",
                        "" if cell.roi_pct is None else f"{cell.roi_pct:.6f}",
                        "" if not cell.e2e_vals else f"{(_pct(cell.e2e_vals, 50) or 0.0):.3f}",
                        "" if not cell.e2e_vals else f"{(_pct(cell.e2e_vals, 90) or 0.0):.3f}",
                    ]
                )


def _write_pivot_csv(
    out_path: Path,
    week_order: Sequence[Tuple[int, int]],
    labels: Sequence[str],
    agg: Dict[Tuple[int, int, str], CellAgg],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        header = ["iso_week"]
        for b in labels:
            header.extend([f"{b}_roi_pct", f"{b}_n"])
        header.extend(["total_roi_pct", "total_n", "total_turnover", "total_pnl"])
        wr.writerow(header)

        for yw in week_order:
            row = [_week_key(yw[0], yw[1])]
            total = CellAgg()
            for b in labels:
                cell = agg.get((yw[0], yw[1], b), CellAgg())
                row.append("" if cell.roi_pct is None else f"{cell.roi_pct:.6f}")
                row.append(cell.n)
                total.n += cell.n
                total.turnover += cell.turnover
                total.pnl += cell.pnl
            row.extend(
                [
                    "" if total.roi_pct is None else f"{total.roi_pct:.6f}",
                    total.n,
                    f"{total.turnover:.6f}",
                    f"{total.pnl:.6f}",
                ]
            )
            wr.writerow(row)


def _write_markdown(
    out_path: Path,
    week_order: Sequence[Tuple[int, int]],
    labels: Sequence[str],
    agg: Dict[Tuple[int, int, str], CellAgg],
    meta_lines: Sequence[str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Heatmap tabular: ROI por semana x bucket de E2E\n")
    lines.extend([f"- {m}\n" for m in meta_lines])
    lines.append("\n")

    header = "| Semana | " + " | ".join(labels) + " | Total |\n"
    sep = "|---|" + "|".join(["---:" for _ in labels]) + "|---:|\n"
    lines.append(header)
    lines.append(sep)

    grand_total = CellAgg()
    for yw in week_order:
        wk = _week_key(yw[0], yw[1])
        row_cells: List[str] = []
        total = CellAgg()
        for b in labels:
            cell = agg.get((yw[0], yw[1], b), CellAgg())
            if cell.n == 0:
                row_cells.append("—")
            else:
                row_cells.append(f"{_fmt_pct(cell.roi_pct)} (n={cell.n})")
            total.n += cell.n
            total.turnover += cell.turnover
            total.pnl += cell.pnl
        total_txt = "—" if total.n == 0 else f"{_fmt_pct(total.roi_pct)} (n={total.n})"
        lines.append(f"| {wk} | " + " | ".join(row_cells) + f" | {total_txt} |\n")

        grand_total.n += total.n
        grand_total.turnover += total.turnover
        grand_total.pnl += total.pnl

    grand_row = []
    for b in labels:
        bt = CellAgg()
        for yw in week_order:
            cell = agg.get((yw[0], yw[1], b), CellAgg())
            bt.n += cell.n
            bt.turnover += cell.turnover
            bt.pnl += cell.pnl
        grand_row.append("—" if bt.n == 0 else f"{_fmt_pct(bt.roi_pct)} (n={bt.n})")
    all_txt = "—" if grand_total.n == 0 else f"{_fmt_pct(grand_total.roi_pct)} (n={grand_total.n})"
    lines.append(f"| TOTAL | " + " | ".join(grand_row) + f" | {all_txt} |\n")

    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Heatmap semanal de ROI por bucket de E2E")
    parser.add_argument("--pnl-csv", required=True, help="CSV base com stake/pnl (e audit_id)")
    parser.add_argument("--e2e-csv", default="", help="CSV opcional com e2e_total_ms por audit_id")
    parser.add_argument("--out-long", required=True, help="CSV de saida (formato long)")
    parser.add_argument("--out-pivot", required=True, help="CSV de saida (formato pivot)")
    parser.add_argument("--out-md", required=True, help="Markdown de saida (tabela heatmap)")
    parser.add_argument("--start-utc", default="", help="Filtro inicial UTC (ISO-8601)")
    parser.add_argument("--end-utc", default="", help="Filtro final UTC (ISO-8601)")
    parser.add_argument("--min-e2e-ms", type=float, default=7000.0, help="Filtro minimo de e2e_total_ms")
    parser.add_argument("--max-e2e-ms", type=float, default=0.0, help="Filtro maximo de e2e_total_ms (0 desliga)")
    parser.add_argument(
        "--buckets-ms",
        default="7000,10000,15000,20000,30000",
        help="Limites internos dos buckets em ms (CSV crescente)",
    )
    parser.add_argument(
        "--require-back-pre-slipneg",
        type=int,
        default=1,
        help="1 aplica filtro back+pre+slippage<0 quando campos existirem; 0 desliga.",
    )
    parser.add_argument(
        "--strict-segment-fields",
        type=int,
        default=0,
        help="1 exige que side/regime/slippage existam para incluir linha; 0 aceita base pre-filtrada.",
    )
    args = parser.parse_args()

    pnl_path = Path(args.pnl_csv)
    if not pnl_path.exists():
        raise SystemExit(f"arquivo inexistente: {pnl_path}")

    e2e_map: Optional[Dict[int, Tuple[float, Optional[datetime]]]] = None
    if str(args.e2e_csv).strip():
        e2e_path = Path(args.e2e_csv)
        if not e2e_path.exists():
            raise SystemExit(f"arquivo inexistente: {e2e_path}")
        e2e_map = _load_e2e_map(e2e_path)

    start_dt = _to_dt(args.start_utc) if str(args.start_utc).strip() else None
    end_dt = _to_dt(args.end_utc) if str(args.end_utc).strip() else None
    edges = _parse_edges(args.buckets_ms)
    labels = _build_bucket_labels(edges)

    rows = _read_csv(pnl_path)
    agg: Dict[Tuple[int, int, str], CellAgg] = defaultdict(CellAgg)
    kept = 0
    dropped_missing = 0
    dropped_time = 0
    dropped_e2e = 0
    dropped_segment = 0
    dropped_reasons: Counter[str] = Counter()
    seen_weeks: set[Tuple[int, int]] = set()
    aid_in_input: set[int] = set()

    for row in rows:
        aid_raw = _to_int(_first_nonempty(row, AUDIT_ID_COLS))
        if aid_raw is not None:
            aid_in_input.add(aid_raw)

        parsed, reason = _resolve_row_fields(row, e2e_map)
        if parsed is None:
            dropped_missing += 1
            if reason:
                dropped_reasons[reason] += 1
            continue
        _, ts, e2e, stake, pnl, is_back, is_pre, is_slipneg = parsed

        if start_dt is not None and ts < start_dt:
            dropped_time += 1
            continue
        if end_dt is not None and ts >= end_dt:
            dropped_time += 1
            continue
        if e2e < float(args.min_e2e_ms):
            dropped_e2e += 1
            continue
        if float(args.max_e2e_ms) > 0 and e2e > float(args.max_e2e_ms):
            dropped_e2e += 1
            continue

        if int(args.require_back_pre_slipneg) == 1:
            if is_back is False or is_pre is False or is_slipneg is False:
                dropped_segment += 1
                continue
            if int(args.strict_segment_fields) == 1 and (
                is_back is None or is_pre is None or is_slipneg is None
            ):
                dropped_segment += 1
                continue

        yw = _iso_week(ts)
        bucket = _bucket_of(e2e, edges, labels)
        agg[(yw[0], yw[1], bucket)].add(stake=stake, pnl=pnl, e2e_ms=e2e)
        seen_weeks.add(yw)
        kept += 1

    week_order = sorted(seen_weeks)
    if not week_order:
        header_cols = sorted({k for r in rows for k in r.keys()})
        e2e_map_n = len(e2e_map) if e2e_map is not None else 0
        e2e_map_cov = 0
        if e2e_map is not None and aid_in_input:
            e2e_map_cov = sum(1 for aid in aid_in_input if aid in e2e_map)

        diag = [
            "nenhuma linha apos filtros.",
            f"rows_in={len(rows)} dropped_missing={dropped_missing} dropped_time={dropped_time} dropped_e2e={dropped_e2e} dropped_segment={dropped_segment}",
            f"top_missing_reasons={dict(dropped_reasons.most_common(8))}",
            _best_col_report(rows, AUDIT_ID_COLS, "audit_id_candidates"),
            _best_col_report(rows, TS_COLS, "timestamp_candidates"),
            _best_col_report(rows, STAKE_COLS, "stake_candidates"),
            _best_col_report(rows, PNL_COLS, "pnl_candidates"),
            _best_col_report(rows, SIDE_COLS, "side_candidates"),
            _best_col_report(rows, REGIME_COLS, "regime_candidates"),
            _best_col_report(rows, SLIP_COLS, "slippage_candidates"),
            f"e2e_map_rows_valid={e2e_map_n} e2e_map_coverage_on_input_aids={e2e_map_cov}/{len(aid_in_input) if aid_in_input else 0}",
            _best_col_report(rows, E2E_COLS, "e2e_candidates_no_map"),
            f"headers_detectados={header_cols}",
            "Dica: teste com --require-back-pre-slipneg 0 para validar cobertura base.",
        ]
        raise SystemExit("\n".join(diag))

    out_long = Path(args.out_long)
    out_pivot = Path(args.out_pivot)
    out_md = Path(args.out_md)
    _write_long_csv(out_long, week_order, labels, agg)
    _write_pivot_csv(out_pivot, week_order, labels, agg)

    meta = [
        f"linhas_entrada={len(rows)}",
        f"linhas_mantidas={kept}",
        f"drop_missing_fields={dropped_missing}",
        f"drop_time_window={dropped_time}",
        f"drop_e2e_filter={dropped_e2e}",
        f"drop_segment_filter={dropped_segment}",
        f"start_utc={start_dt.isoformat() if start_dt else 'None'}",
        f"end_utc={end_dt.isoformat() if end_dt else 'None'}",
        f"min_e2e_ms={float(args.min_e2e_ms)}",
        f"max_e2e_ms={float(args.max_e2e_ms)}",
        f"buckets_ms={','.join(str(int(x)) if abs(x-int(x)) < 1e-9 else str(x) for x in edges)}",
    ]
    _write_markdown(out_md, week_order, labels, agg, meta)

    print("OK")
    print(f"rows_in={len(rows)} kept={kept}")
    print(
        f"dropped_missing={dropped_missing} dropped_time={dropped_time} "
        f"dropped_e2e={dropped_e2e} dropped_segment={dropped_segment}"
    )
    print(f"out_long={out_long}")
    print(f"out_pivot={out_pivot}")
    print(f"out_md={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
