#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analise robusta da estrategia Back Pre com slippage < 0.

Entrada principal: CSV por aposta (idealmente com P&L real do ledger por order_id).

Metodos implementados:
- ROI ponderado por stake
- Bootstrap por cluster de evento (IC90/IC95)
- Permutacao estratificada (semana x liga), unilateral H1: ROI > 0
- Concentracao de P&L por evento (Top-k, HHI, Gini)
- Robustez a concentracao (leave-top-k-out)
- Estabilidade semanal
- Decisao automatica (robusto/moderado/fragil)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _parse_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    s = s.replace(" ", "")
    # suporta "1.234,56" e "1234.56"
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        x = float(s)
    except Exception:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _parse_bool(v: Any) -> Optional[bool]:
    if v is None:
        return None
    s = str(v).strip().lower()
    if not s:
        return None
    if s in {"1", "true", "t", "yes", "y", "sim", "s"}:
        return True
    if s in {"0", "false", "f", "no", "n", "nao", "não"}:
        return False
    return None


def _parse_dt(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    s = str(v).strip()
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


def _iso_week(dt: Optional[datetime]) -> str:
    if dt is None:
        return "week:unknown"
    y, w, _ = dt.isocalendar()
    return f"{y}-W{w:02d}"


def _pick_col(fieldnames: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lookup = {f.lower(): f for f in fieldnames}
    lookup_norm = { _norm_colname(f): f for f in fieldnames }
    for c in candidates:
        f = lookup.get(c.lower())
        if f:
            return f
        fn = lookup_norm.get(_norm_colname(c))
        if fn:
            return fn
    return None


def _norm_colname(s: str) -> str:
    out = []
    for ch in str(s).strip().lower():
        if ch.isalnum():
            out.append(ch)
    return "".join(out)


def _sniff_csv(path: Path) -> Tuple[csv.Dialect, List[str]]:
    sample = path.read_text(encoding="utf-8", errors="ignore")[:8192]
    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return dialect, lines


@dataclass
class BetRow:
    event_id: str
    league: str
    week: str
    stake: float
    pnl: float


def load_rows_from_csv(
    path: Path,
    *,
    enforce_slip_neg: bool = True,
    assume_back_pre_when_missing: bool = True,
    col_overrides: Optional[Dict[str, str]] = None,
    allow_unit_stake_fallback: bool = False,
) -> Tuple[List[BetRow], Dict[str, int], Dict[str, Optional[str]]]:
    dialect, lines = _sniff_csv(path)
    reader = csv.DictReader(lines, dialect=dialect)
    fields = list(reader.fieldnames or [])
    if not fields:
        raise RuntimeError(f"CSV sem cabecalho: {path}")

    ov = col_overrides or {}

    def _pick(name: str, candidates: Sequence[str]) -> Optional[str]:
        manual = str(ov.get(name, "") or "").strip()
        if manual:
            f_lookup = {f.lower(): f for f in fields}
            by_norm = {_norm_colname(f): f for f in fields}
            if manual.lower() in f_lookup:
                return f_lookup[manual.lower()]
            mm = by_norm.get(_norm_colname(manual))
            if mm:
                return mm
            raise RuntimeError(
                f"Coluna manual '{manual}' ({name}) nao encontrada no CSV. "
                f"Headers: {fields}"
            )
        return _pick_col(fields, candidates)

    col_event = _pick("event", ["event_id", "match_id", "game_id", "fixture_id", "event", "eventid", "matchid"])
    col_league = _pick("league", ["league", "league_name", "competition", "tournament", "liga"])
    col_ts = _pick("timestamp", ["audited_at", "timestamp", "ts", "created_at", "post_date_utc", "date_utc", "datetime"])
    col_stake = _pick(
        "stake",
        [
            "stake",
            "exposure",
            "stake_usd",
            "stake_total",
            "amount",
            "stake_liquidado",
            "stake_liq",
            "apostado_back",
            "valor_em_risco",
            "exposicao",
            "exposição",
        ],
    )
    col_pnl = _pick(
        "pnl",
        [
            "pnl",
            "pnl_acct",
            "pnl_real",
            "profit",
            "pl",
            "result",
            "lucro",
            "p&l",
            "pnl_total",
        ],
    )
    col_roi = _pick("roi", ["roi", "roi_pct", "roiw", "roi%", "roi_w"])
    col_slip = _pick("slippage", ["slippage_pre_pct", "slippage_raw_pct", "slippage", "slippagepct", "slippage_pre"])
    col_side = _pick("side", ["side", "exec_side", "direction_side", "lado"])
    col_regime = _pick("regime", ["regime", "market_regime", "market_period", "phase", "pre_in"])
    col_is_live = _pick("is_live", ["is_live", "live", "ao_vivo"])

    if col_event is None:
        col_event = _pick_col(fields, ["order_id", "audit_id", "id"])
    if col_event is None:
        raise RuntimeError("Nao encontrei coluna de event_id/order_id/audit_id no CSV.")
    if col_stake is None and not allow_unit_stake_fallback:
        raise RuntimeError(
            "Nao encontrei coluna de stake/exposure no CSV. "
            "Use --stake-col <nome_coluna> ou --allow-unit-stake-fallback 1."
        )
    if col_pnl is None and col_roi is None:
        raise RuntimeError("Nao encontrei coluna de pnl (ou roi) no CSV.")

    dropped = defaultdict(int)
    rows: List[BetRow] = []

    for raw in reader:
        event_id = str(raw.get(col_event, "")).strip()
        if not event_id:
            dropped["missing_event_id"] += 1
            continue

        # Filtro de estrategia: Back + Pre
        side_ok = True
        regime_ok = True
        if col_side:
            side_v = str(raw.get(col_side, "")).strip().lower()
            side_ok = side_v in {"back", "home", "away", "h", "a", "b"}
        if col_regime:
            reg_v = str(raw.get(col_regime, "")).strip().lower()
            regime_ok = ("pre" in reg_v) or (reg_v in {"prematch", "pre_match"})
        elif col_is_live:
            is_live = _parse_bool(raw.get(col_is_live))
            regime_ok = (is_live is False)
        elif not assume_back_pre_when_missing:
            regime_ok = False

        if not side_ok or not regime_ok:
            dropped["outside_back_pre"] += 1
            continue

        if enforce_slip_neg:
            if not col_slip:
                dropped["missing_slippage_column"] += 1
                continue
            slip = _parse_float(raw.get(col_slip))
            if slip is None:
                dropped["missing_slippage_value"] += 1
                continue
            if not (slip < 0):
                dropped["slippage_not_negative"] += 1
                continue

        pnl = _parse_float(raw.get(col_pnl)) if col_pnl else None
        roi = _parse_float(raw.get(col_roi)) if col_roi else None

        stake = _parse_float(raw.get(col_stake)) if col_stake else None
        if stake is None and allow_unit_stake_fallback:
            # fallback conservador: unidade por aposta
            stake = 1.0
            dropped["unit_stake_fallback"] += 1
        if stake is None or stake <= 0:
            # tenta derivar stake de pnl/roi quando possivel
            if pnl is not None and roi is not None and abs(roi) > 1e-12:
                stake = abs(100.0 * pnl / roi)
                dropped["stake_derived_from_pnl_roi"] += 1
            else:
                dropped["invalid_stake"] += 1
                continue

        if pnl is None and roi is not None:
            pnl = (roi / 100.0) * stake
        if pnl is None:
            dropped["missing_pnl_and_roi"] += 1
            continue

        dt = _parse_dt(raw.get(col_ts)) if col_ts else None
        week = _iso_week(dt)
        league = str(raw.get(col_league, "")).strip() if col_league else ""
        if not league:
            league = "league:unknown"

        rows.append(BetRow(event_id=event_id, league=league, week=week, stake=stake, pnl=pnl))

    meta_cols = {
        "event_id": col_event,
        "league": col_league,
        "timestamp": col_ts,
        "stake": col_stake,
        "pnl": col_pnl,
        "roi": col_roi,
        "slippage": col_slip,
        "side": col_side,
        "regime": col_regime or col_is_live,
    }
    return rows, dict(dropped), meta_cols


def weighted_roi_pct(rows: Iterable[BetRow]) -> Optional[float]:
    rows = list(rows)
    st = sum(r.stake for r in rows)
    if st <= 0:
        return None
    return 100.0 * (sum(r.pnl for r in rows) / st)


def aggregate_by_event(rows: Sequence[BetRow]) -> List[Dict[str, Any]]:
    by_evt: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        rec = by_evt.get(r.event_id)
        if rec is None:
            rec = {
                "event_id": r.event_id,
                "league": r.league,
                "week": r.week,
                "stake": 0.0,
                "pnl": 0.0,
            }
            by_evt[r.event_id] = rec
        rec["stake"] += r.stake
        rec["pnl"] += r.pnl
    out = list(by_evt.values())
    for rec in out:
        rec["roi_pct"] = 100.0 * rec["pnl"] / rec["stake"] if rec["stake"] > 0 else None
    return out


def bootstrap_cluster_roi(
    events: Sequence[Dict[str, Any]],
    *,
    n_boot: int,
    seed: int,
) -> Dict[str, Optional[float]]:
    if not events:
        return {"mean": None, "ci90_lo": None, "ci90_hi": None, "ci95_lo": None, "ci95_hi": None}
    if len(events) == 1:
        roi = 100.0 * (events[0]["pnl"] / events[0]["stake"]) if events[0]["stake"] > 0 else None
        return {"mean": roi, "ci90_lo": None, "ci90_hi": None, "ci95_lo": None, "ci95_hi": None}

    rng = random.Random(seed)
    vals = []
    n = len(events)
    for _ in range(max(1, int(n_boot))):
        sample = [events[rng.randrange(n)] for _ in range(n)]
        st = sum(e["stake"] for e in sample)
        pnl = sum(e["pnl"] for e in sample)
        vals.append(100.0 * pnl / st if st > 0 else 0.0)
    vals.sort()

    def q(p: float) -> float:
        idx = int(round((len(vals) - 1) * p))
        return vals[max(0, min(len(vals) - 1, idx))]

    return {
        "mean": sum(vals) / len(vals),
        "ci90_lo": q(0.05),
        "ci90_hi": q(0.95),
        "ci95_lo": q(0.025),
        "ci95_hi": q(0.975),
    }


def permutation_pvalue_stratified(
    events: Sequence[Dict[str, Any]],
    *,
    n_perm: int,
    seed: int,
) -> Optional[float]:
    if not events:
        return None
    obs = weighted_roi_pct(BetRow(event_id=e["event_id"], league=e["league"], week=e["week"], stake=e["stake"], pnl=e["pnl"]) for e in events)
    if obs is None:
        return None

    strata: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for e in events:
        strata[(e["week"], e["league"])].append(e)

    # Sign-flip por evento dentro de cada estrato.
    # Mantem estrutura de stake/composicao e testa H0 de ausencia de edge.
    rng = random.Random(seed + 17)
    ge = 0
    total = max(1, int(n_perm))
    for _ in range(total):
        pnl_tot = 0.0
        st_tot = 0.0
        for group in strata.values():
            for e in group:
                sign = -1.0 if rng.random() < 0.5 else 1.0
                pnl_tot += sign * e["pnl"]
                st_tot += e["stake"]
        stat = 100.0 * pnl_tot / st_tot if st_tot > 0 else 0.0
        if stat >= obs:
            ge += 1
    return (ge + 1) / (total + 1)


def topk_concentration(events: Sequence[Dict[str, Any]], k: int) -> Dict[str, Optional[float]]:
    if not events:
        return {"share_total_pnl": None, "share_abs_pnl": None}
    sorted_by_pnl = sorted(events, key=lambda x: x["pnl"], reverse=True)
    top = sorted_by_pnl[: max(0, int(k))]
    total_pnl = sum(e["pnl"] for e in events)
    total_abs = sum(abs(e["pnl"]) for e in events)
    top_pnl = sum(e["pnl"] for e in top)
    top_abs = sum(abs(e["pnl"]) for e in top)
    share_total = (top_pnl / total_pnl) if abs(total_pnl) > 1e-12 else None
    share_abs = (top_abs / total_abs) if total_abs > 1e-12 else None
    return {"share_total_pnl": share_total, "share_abs_pnl": share_abs}


def hhi_abs_pnl(events: Sequence[Dict[str, Any]]) -> Optional[float]:
    total_abs = sum(abs(e["pnl"]) for e in events)
    if total_abs <= 0:
        return None
    return sum((abs(e["pnl"]) / total_abs) ** 2 for e in events)


def gini_abs_pnl(events: Sequence[Dict[str, Any]]) -> Optional[float]:
    xs = sorted(abs(e["pnl"]) for e in events)
    n = len(xs)
    if n == 0:
        return None
    s = sum(xs)
    if s <= 0:
        return 0.0
    weighted = sum((i + 1) * x for i, x in enumerate(xs))
    return (2.0 * weighted) / (n * s) - (n + 1) / n


def leave_topk_out_roi(events: Sequence[Dict[str, Any]], k: int) -> Optional[float]:
    if not events:
        return None
    kept = sorted(events, key=lambda x: x["pnl"], reverse=True)[max(0, int(k)) :]
    if not kept:
        return None
    st = sum(e["stake"] for e in kept)
    if st <= 0:
        return None
    return 100.0 * (sum(e["pnl"] for e in kept) / st)


def weekly_stability(events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_week: Dict[str, Dict[str, float]] = defaultdict(lambda: {"pnl": 0.0, "stake": 0.0})
    for e in events:
        by_week[e["week"]]["pnl"] += e["pnl"]
        by_week[e["week"]]["stake"] += e["stake"]
    rows = []
    for w, v in sorted(by_week.items()):
        roi = 100.0 * v["pnl"] / v["stake"] if v["stake"] > 0 else None
        rows.append({"week": w, "pnl": v["pnl"], "stake": v["stake"], "roi_pct": roi})
    n = len(rows)
    pos = sum(1 for r in rows if (r["roi_pct"] is not None and r["roi_pct"] > 0))
    return {
        "n_weeks": n,
        "n_positive_weeks": pos,
        "positive_week_ratio": (pos / n) if n > 0 else None,
        "weekly_rows": rows,
    }


def classify_result(
    *,
    ci95_lo: Optional[float],
    p_perm: Optional[float],
    roi_leave_top5: Optional[float],
    top1_abs_share: Optional[float],
    positive_week_ratio: Optional[float],
) -> Tuple[str, List[str]]:
    checks = []
    checks.append(("ci95_lo_gt_0", ci95_lo is not None and ci95_lo > 0))
    checks.append(("perm_p_lt_0_05", p_perm is not None and p_perm < 0.05))
    checks.append(("leave_top5_roi_gt_0", roi_leave_top5 is not None and roi_leave_top5 > 0))
    checks.append(("top1_abs_share_le_0_35", top1_abs_share is not None and top1_abs_share <= 0.35))
    checks.append(("positive_week_ratio_ge_0_55", positive_week_ratio is not None and positive_week_ratio >= 0.55))

    ok = [name for name, passed in checks if passed]
    score = len(ok)
    if score >= 4:
        label = "robusto"
    elif score >= 2:
        label = "moderado"
    else:
        label = "fragil"
    return label, ok


def _pct(v: Optional[float], nd: int = 2) -> str:
    if v is None:
        return "NA"
    return f"{v:.{nd}f}%"


def build_markdown(report: Dict[str, Any]) -> str:
    c = report["core"]
    b = report["bootstrap"]
    conc = report["concentration"]
    wk = report["weekly"]
    dec = report["decision"]

    lines = []
    lines.append("# Avaliacao robusta - Back Pre com slippage < 0\n")
    lines.append("## Resumo executivo\n")
    lines.append(f"- ROI observado (ponderado por stake): **{_pct(c['roi_pct'])}**\n")
    lines.append(
        f"- IC95 bootstrap por evento: **[{_pct(b['ci95_lo'])}, {_pct(b['ci95_hi'])}]**\n"
    )
    lines.append(f"- p-valor permutacao estratificada (unilateral): **{report['permutation_pvalue']}**\n")
    lines.append(f"- Classificacao final: **{dec['label'].upper()}**\n")
    lines.append("")
    lines.append("## Cobertura\n")
    lines.append(f"- N apostas filtradas: {c['n_bets']}\n")
    lines.append(f"- N eventos (clusters): {c['n_events']}\n")
    lines.append(f"- Stake total: {c['stake_total']:.2f}\n")
    lines.append(f"- P&L total: {c['pnl_total']:.2f}\n")
    lines.append("")
    lines.append("## Confianca estatistica\n")
    lines.append(f"- IC90: [{_pct(b['ci90_lo'])}, {_pct(b['ci90_hi'])}]\n")
    lines.append(f"- IC95: [{_pct(b['ci95_lo'])}, {_pct(b['ci95_hi'])}]\n")
    lines.append("")
    lines.append("## Concentracao de P&L por evento\n")
    lines.append(f"- Top1 share (absoluto): {_pct(conc['top1_abs_share'] * 100 if conc['top1_abs_share'] is not None else None)}\n")
    lines.append(f"- Top5 share (absoluto): {_pct(conc['top5_abs_share'] * 100 if conc['top5_abs_share'] is not None else None)}\n")
    lines.append(f"- HHI (abs P&L): {conc['hhi_abs_pnl']}\n")
    lines.append(f"- Gini (abs P&L): {conc['gini_abs_pnl']}\n")
    lines.append("")
    lines.append("## Robustez (retirando melhores eventos)\n")
    for k, v in report["leave_topk_roi_pct"].items():
        lines.append(f"- ROI sem Top-{k}: {_pct(v)}\n")
    lines.append("")
    lines.append("## Estabilidade temporal\n")
    lines.append(f"- Semanas com dados: {wk['n_weeks']}\n")
    lines.append(f"- Semanas positivas: {wk['n_positive_weeks']}\n")
    lines.append(f"- Proporcao semanas positivas: {wk['positive_week_ratio']}\n")
    lines.append("")
    lines.append("## Regras acionadas na decisao\n")
    lines.append(f"- Checks aprovados: {', '.join(dec['checks_passed']) if dec['checks_passed'] else 'nenhum'}\n")
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analise robusta do ROI Back Pre slippage<0")
    parser.add_argument("--input-csv", required=True, help="CSV por aposta com stake e pnl (ou roi)")
    parser.add_argument("--out-json", default="betinasia_bot/docs/roi_backpre_slipneg_robusto.json")
    parser.add_argument("--out-md", default="betinasia_bot/docs/roi_backpre_slipneg_robusto.md")
    parser.add_argument("--bootstrap-iters", type=int, default=10000)
    parser.add_argument("--perm-iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--no-enforce-slip-neg", action="store_true", help="Nao aplicar filtro slip<0")
    parser.add_argument(
        "--assume-back-pre-when-missing",
        type=int,
        default=1,
        help="Se side/regime ausentes, assume que CSV ja esta no recorte Back Pre (default 1).",
    )
    parser.add_argument("--topk", default="1,3,5,10", help="Lista de k para leave-top-k e concentracao")
    parser.add_argument("--event-col", default="", help="Override manual da coluna de evento (event_id/order_id/audit_id).")
    parser.add_argument("--league-col", default="", help="Override manual da coluna de liga.")
    parser.add_argument("--timestamp-col", default="", help="Override manual da coluna de timestamp.")
    parser.add_argument("--stake-col", default="", help="Override manual da coluna de stake/exposure.")
    parser.add_argument("--pnl-col", default="", help="Override manual da coluna de pnl.")
    parser.add_argument("--roi-col", default="", help="Override manual da coluna de roi.")
    parser.add_argument("--slippage-col", default="", help="Override manual da coluna de slippage.")
    parser.add_argument("--side-col", default="", help="Override manual da coluna de side.")
    parser.add_argument("--regime-col", default="", help="Override manual da coluna de regime.")
    parser.add_argument(
        "--allow-unit-stake-fallback",
        type=int,
        default=0,
        help="Se 1, usa stake=1 quando coluna de stake estiver ausente/invalida.",
    )
    parser.add_argument(
        "--print-columns-only",
        type=int,
        default=0,
        help="Se 1, imprime colunas do CSV e sai (debug).",
    )
    args = parser.parse_args()

    csv_path = Path(args.input_csv)
    if not csv_path.exists():
        raise RuntimeError(f"CSV nao encontrado: {csv_path}")

    if int(args.print_columns_only) == 1:
        _, lines = _sniff_csv(csv_path)
        rd = csv.DictReader(lines)
        print(rd.fieldnames or [])
        return 0

    overrides = {
        "event": args.event_col,
        "league": args.league_col,
        "timestamp": args.timestamp_col,
        "stake": args.stake_col,
        "pnl": args.pnl_col,
        "roi": args.roi_col,
        "slippage": args.slippage_col,
        "side": args.side_col,
        "regime": args.regime_col,
    }

    rows, dropped, cols = load_rows_from_csv(
        csv_path,
        enforce_slip_neg=(not args.no_enforce_slip_neg),
        assume_back_pre_when_missing=bool(args.assume_back_pre_when_missing),
        col_overrides=overrides,
        allow_unit_stake_fallback=bool(args.allow_unit_stake_fallback),
    )
    if not rows:
        raise RuntimeError(f"Nenhuma linha valida apos filtros. dropped={dropped}")

    events = aggregate_by_event(rows)
    roi = weighted_roi_pct(rows)
    boot = bootstrap_cluster_roi(events, n_boot=args.bootstrap_iters, seed=args.seed)
    p_perm = permutation_pvalue_stratified(events, n_perm=args.perm_iters, seed=args.seed)

    ks = []
    for tok in str(args.topk).split(","):
        tok = tok.strip()
        if tok.isdigit():
            ks.append(int(tok))
    ks = sorted(set(k for k in ks if k > 0))
    if not ks:
        ks = [1, 3, 5, 10]

    conc_top = {k: topk_concentration(events, k) for k in ks}
    leave_top = {k: leave_topk_out_roi(events, k) for k in ks}
    wk = weekly_stability(events)

    dec_label, dec_checks = classify_result(
        ci95_lo=boot["ci95_lo"],
        p_perm=p_perm,
        roi_leave_top5=leave_top.get(5),
        top1_abs_share=(conc_top.get(1, {}).get("share_abs_pnl")),
        positive_week_ratio=wk.get("positive_week_ratio"),
    )

    report: Dict[str, Any] = {
        "input": {
            "csv": str(csv_path),
            "detected_columns": cols,
            "dropped_reasons": dropped,
        },
        "core": {
            "n_bets": len(rows),
            "n_events": len(events),
            "stake_total": sum(r.stake for r in rows),
            "pnl_total": sum(r.pnl for r in rows),
            "roi_pct": roi,
        },
        "bootstrap": boot,
        "permutation_pvalue": p_perm,
        "concentration": {
            "top1_total_share": conc_top.get(1, {}).get("share_total_pnl"),
            "top1_abs_share": conc_top.get(1, {}).get("share_abs_pnl"),
            "top5_total_share": conc_top.get(5, {}).get("share_total_pnl"),
            "top5_abs_share": conc_top.get(5, {}).get("share_abs_pnl"),
            "hhi_abs_pnl": hhi_abs_pnl(events),
            "gini_abs_pnl": gini_abs_pnl(events),
        },
        "leave_topk_roi_pct": leave_top,
        "weekly": wk,
        "decision": {
            "label": dec_label,
            "checks_passed": dec_checks,
        },
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(build_markdown(report), encoding="utf-8")

    print(f"[OK] Analise concluida.")
    print(f"[OK] JSON: {out_json}")
    print(f"[OK] MD:   {out_md}")
    print(f"[OK] ROI observado: {_pct(report['core']['roi_pct'])}")
    print(f"[OK] CI95 bootstrap: [{_pct(report['bootstrap']['ci95_lo'])}, {_pct(report['bootstrap']['ci95_hi'])}]")
    print(f"[OK] p-valor permutacao: {report['permutation_pvalue']}")
    print(f"[OK] Classificacao: {report['decision']['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

