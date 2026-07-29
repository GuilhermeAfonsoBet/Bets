#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analise avancada de distribuicao/risco para CSV por aposta.

Cobertura principal:
1) Distribuicao por aposta (media/mediana/quantis/win-rate)
2) Distribuicao por evento (inclui concentracao e ROI sem top-k)
3) Curvas acumuladas (aposta/dia/evento) + drawdown/TUW/Ulcer
4) Rolling windows (20/50/100 apostas, 7/14/28 dias)
5) Concentracao (top-1/3/5 apostas e eventos)
6) Distribuicao de perdas
7) Buckets por odds
8) EV esperado vs realizado (quando odd justa disponivel)
9) Slippage granular (decis/combinacoes)
10) Latencia como explicativa
11) Bootstrap (aposta/evento/dia)
12) Dashboard consolidado
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class Bet:
    row_id: int
    ts: datetime
    day: str
    hour: int
    event_id: str
    league: str
    stake: float
    pnl: float
    roi_pct: float
    slippage: Optional[float]
    latency_sec: Optional[float]
    odd_taken: Optional[float]
    odd_fair: Optional[float]
    ev_pct: Optional[float]


def _pf(v: Any) -> Optional[float]:
    s = str(v or "").strip().replace(" ", "")
    if not s:
        return None
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


def _pdt(v: Any) -> Optional[datetime]:
    s = str(v or "").strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        d = datetime.fromisoformat(s)
    except Exception:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc)


def _pick(fields: Sequence[str], cands: Sequence[str]) -> Optional[str]:
    by_l = {f.lower(): f for f in fields}
    by_n = {"".join(ch for ch in f.lower() if ch.isalnum()): f for f in fields}
    for c in cands:
        cl = c.lower()
        if cl in by_l:
            return by_l[cl]
        cn = "".join(ch for ch in cl if ch.isalnum())
        if cn in by_n:
            return by_n[cn]
    return None


def _quantile(xs: Sequence[float], q: float) -> Optional[float]:
    if not xs:
        return None
    ys = sorted(xs)
    n = len(ys)
    if n == 1:
        return ys[0]
    p = (n - 1) * q
    i = int(math.floor(p))
    j = int(math.ceil(p))
    if i == j:
        return ys[i]
    w = p - i
    return ys[i] * (1.0 - w) + ys[j] * w


def _mean(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    return sum(xs) / len(xs)


def _median(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    ys = sorted(xs)
    n = len(ys)
    return ys[n // 2] if n % 2 == 1 else 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def _weighted_roi_pct(pnls: Sequence[float], stakes: Sequence[float]) -> Optional[float]:
    st = sum(stakes)
    if st <= 0:
        return None
    return 100.0 * (sum(pnls) / st)


def _fmt(x: Optional[float], nd: int = 2, pct: bool = False) -> str:
    if x is None:
        return "NA"
    return f"{x:.{nd}f}" + ("%" if pct else "")


def _parse_edges(raw: str) -> List[float]:
    vals: List[float] = []
    for p in str(raw or "").split(","):
        x = _pf(p)
        if x is not None:
            vals.append(x)
    return sorted(set(vals))


def _bucket(v: Optional[float], edges: Sequence[float]) -> str:
    if v is None:
        return "NA"
    if not edges:
        return "all"
    if v < edges[0]:
        return f"<{edges[0]:.2f}"
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        if a <= v < b:
            return f"[{a:.2f},{b:.2f})"
    return f">={edges[-1]:.2f}"


def _load_bets(
    csv_path: Path,
    *,
    enforce_slip_neg: bool,
    filter_back_pre: bool,
) -> Tuple[List[Bet], Dict[str, str], Dict[str, int]]:
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        rd = csv.DictReader(f)
        fields = list(rd.fieldnames or [])
        if not fields:
            raise RuntimeError(f"CSV sem cabecalho: {csv_path}")

        col_event = _pick(fields, ["event_id", "match_id", "fixture_id", "game_id", "order_id", "audit_id", "id"])
        col_ts = _pick(fields, ["audited_at", "executed_at", "created_at", "timestamp", "post date", "post_date"])
        col_league = _pick(fields, ["league", "league_name", "competition", "tournament"])
        col_stake = _pick(fields, ["stake", "stake_real", "exposure", "exposure_real", "stake_liq"])
        col_pnl = _pick(fields, ["pnl_real", "pnl", "profit", "pl", "result", "amount"])
        col_roi = _pick(fields, ["roi_pct", "roi", "roi_real_pct"])
        col_slip = _pick(fields, ["slippage_pre_pct", "slippage_raw_pct", "slippage", "diff_pct"])
        col_side = _pick(fields, ["side", "exec_side", "direction"])
        col_regime = _pick(fields, ["regime", "market_regime", "phase"])
        col_is_live = _pick(fields, ["is_live", "live"])
        col_latency = _pick(fields, ["latency_s", "latency_sec", "latency", "lat_s", "decision_latency_s"])
        col_odd = _pick(fields, ["got price", "got_price", "price", "odd", "odds", "matched price"])
        col_odd_fair = _pick(fields, ["odd_justa", "fair_odd", "closing_odd", "odd_fair", "odd_justa_pre"])

        if not col_event or not col_ts or not col_stake:
            raise RuntimeError(
                f"Colunas minimas ausentes. event={col_event}, ts={col_ts}, stake={col_stake}"
            )
        if not col_pnl and not col_roi:
            raise RuntimeError("CSV sem pnl e sem roi.")

        mapping = {
            "event_id": col_event,
            "timestamp": col_ts,
            "league": col_league or "",
            "stake": col_stake,
            "pnl": col_pnl or "",
            "roi": col_roi or "",
            "slippage": col_slip or "",
            "side": col_side or "",
            "regime": col_regime or col_is_live or "",
            "latency": col_latency or "",
            "odd_taken": col_odd or "",
            "odd_fair": col_odd_fair or "",
        }

        dropped: Dict[str, int] = defaultdict(int)
        bets: List[Bet] = []
        i = 0
        for row in rd:
            i += 1
            event_id = str(row.get(col_event, "")).strip() or f"row_{i}"
            dt = _pdt(row.get(col_ts))
            if dt is None:
                dropped["invalid_ts"] += 1
                continue
            stake = _pf(row.get(col_stake))
            if stake is None or stake <= 0:
                dropped["invalid_stake"] += 1
                continue

            if filter_back_pre:
                if col_side:
                    side_v = str(row.get(col_side, "")).strip().lower()
                    if side_v and side_v not in {"back", "b", "home", "away", "h", "a"}:
                        dropped["outside_back_side"] += 1
                        continue
                if col_regime:
                    reg_v = str(row.get(col_regime, "")).strip().lower()
                    if reg_v and ("pre" not in reg_v and reg_v not in {"prematch", "pre_match"}):
                        dropped["outside_pre_regime"] += 1
                        continue
                elif col_is_live:
                    lv = str(row.get(col_is_live, "")).strip().lower()
                    if lv in {"1", "true", "t", "yes", "sim", "s"}:
                        dropped["outside_pre_regime"] += 1
                        continue

            slip = _pf(row.get(col_slip)) if col_slip else None
            if enforce_slip_neg:
                if slip is None:
                    dropped["missing_slippage"] += 1
                    continue
                if not (slip < 0):
                    dropped["slippage_not_negative"] += 1
                    continue

            pnl = _pf(row.get(col_pnl)) if col_pnl else None
            roi_pct = _pf(row.get(col_roi)) if col_roi else None
            if pnl is None and roi_pct is not None:
                pnl = stake * (roi_pct / 100.0)
            if roi_pct is None and pnl is not None:
                roi_pct = 100.0 * pnl / stake
            if pnl is None or roi_pct is None:
                dropped["missing_pnl_roi"] += 1
                continue

            odd_taken = _pf(row.get(col_odd)) if col_odd else None
            odd_fair = _pf(row.get(col_odd_fair)) if col_odd_fair else None
            ev_pct = None
            if odd_taken and odd_fair and odd_taken > 0 and odd_fair > 0:
                ev_pct = 100.0 * (odd_taken / odd_fair - 1.0)

            bets.append(
                Bet(
                    row_id=i,
                    ts=dt,
                    day=dt.date().isoformat(),
                    hour=int(dt.hour),
                    event_id=event_id,
                    league=str(row.get(col_league, "")).strip() if col_league else "",
                    stake=stake,
                    pnl=pnl,
                    roi_pct=roi_pct,
                    slippage=slip,
                    latency_sec=_pf(row.get(col_latency)) if col_latency else None,
                    odd_taken=odd_taken,
                    odd_fair=odd_fair,
                    ev_pct=ev_pct,
                )
            )

    bets.sort(key=lambda b: (b.ts, b.row_id))
    return bets, mapping, dict(dropped)


def _aggregate_events(bets: Sequence[Bet]) -> List[Dict[str, Any]]:
    by_event: Dict[str, Dict[str, Any]] = {}
    for b in bets:
        rec = by_event.get(b.event_id)
        if rec is None:
            rec = {
                "event_id": b.event_id,
                "stake": 0.0,
                "pnl": 0.0,
                "n_bets": 0,
                "first_ts": b.ts,
                "max_exposure": 0.0,
            }
            by_event[b.event_id] = rec
        rec["stake"] += b.stake
        rec["pnl"] += b.pnl
        rec["n_bets"] += 1
        rec["max_exposure"] = max(rec["max_exposure"], b.stake)
        if b.ts < rec["first_ts"]:
            rec["first_ts"] = b.ts
    out = list(by_event.values())
    for e in out:
        e["roi_pct"] = 100.0 * e["pnl"] / e["stake"] if e["stake"] > 0 else None
    return out


def _top_contrib(values: Sequence[float], total: float, k: int) -> Optional[float]:
    if not values or abs(total) < 1e-12:
        return None
    xs = sorted(values, reverse=True)[: max(0, int(k))]
    return 100.0 * (sum(xs) / total)


def _roi_without_top_k(records: Sequence[Dict[str, Any]], k: int) -> Optional[float]:
    if not records:
        return None
    kept = sorted(records, key=lambda x: x["pnl"], reverse=True)[max(0, int(k)) :]
    if not kept:
        return None
    st = sum(r["stake"] for r in kept)
    if st <= 0:
        return None
    return 100.0 * sum(r["pnl"] for r in kept) / st


def _roi_without_top_k_bets(bets: Sequence[Bet], k: int) -> Optional[float]:
    if not bets:
        return None
    kept = sorted(bets, key=lambda b: b.pnl, reverse=True)[max(0, int(k)) :]
    if not kept:
        return None
    return _weighted_roi_pct([b.pnl for b in kept], [b.stake for b in kept])


def _accumulate(xs: Sequence[float]) -> List[float]:
    out: List[float] = []
    cur = 0.0
    for x in xs:
        cur += x
        out.append(cur)
    return out


def _drawdown_metrics(pnl_path: Sequence[float]) -> Dict[str, Optional[float]]:
    if not pnl_path:
        return {
            "max_drawdown_abs": None,
            "max_drawdown_pct_on_peak": None,
            "time_under_water_steps": None,
            "recovery_factor": None,
            "ulcer_index": None,
        }
    eq = _accumulate(pnl_path)
    peak = eq[0]
    peak_idx = 0
    max_dd = 0.0
    max_dd_pct = 0.0
    max_tuw = 0
    cur_underwater_start: Optional[int] = None
    dd_sq: List[float] = []
    for i, v in enumerate(eq):
        if v >= peak:
            peak = v
            peak_idx = i
            if cur_underwater_start is not None:
                max_tuw = max(max_tuw, i - cur_underwater_start)
                cur_underwater_start = None
        else:
            if cur_underwater_start is None:
                cur_underwater_start = i
            dd = peak - v
            max_dd = max(max_dd, dd)
            if abs(peak) > 1e-12:
                max_dd_pct = max(max_dd_pct, 100.0 * dd / abs(peak))
            dd_pct = 0.0 if abs(peak) < 1e-12 else 100.0 * dd / abs(peak)
            dd_sq.append(dd_pct * dd_pct)
    if cur_underwater_start is not None:
        max_tuw = max(max_tuw, len(eq) - 1 - cur_underwater_start)
    total_pnl = sum(pnl_path)
    rf = (total_pnl / max_dd) if max_dd > 1e-12 else None
    ulcer = math.sqrt(sum(dd_sq) / len(dd_sq)) if dd_sq else 0.0
    return {
        "max_drawdown_abs": max_dd,
        "max_drawdown_pct_on_peak": max_dd_pct,
        "time_under_water_steps": float(max_tuw),
        "recovery_factor": rf,
        "ulcer_index": ulcer,
    }


def _longest_negative_streak(xs: Sequence[float]) -> int:
    best = 0
    cur = 0
    for x in xs:
        if x < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _rolling_bets(bets: Sequence[Bet], windows: Sequence[int]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for w in windows:
        if w <= 0:
            continue
        stats = []
        for i in range(0, len(bets) - w + 1):
            sl = bets[i : i + w]
            pnls = [b.pnl for b in sl]
            stakes = [b.stake for b in sl]
            by_evt = defaultdict(lambda: {"stake": 0.0, "pnl": 0.0})
            for b in sl:
                by_evt[b.event_id]["stake"] += b.stake
                by_evt[b.event_id]["pnl"] += b.pnl
            evs = [{"stake": v["stake"], "pnl": v["pnl"]} for v in by_evt.values()]
            stats.append(
                {
                    "roi_pct": _weighted_roi_pct(pnls, stakes),
                    "pnl": sum(pnls),
                    "median_pnl": _median(pnls),
                    "n_events": len(evs),
                    "roi_sem_top1_event": _roi_without_top_k(evs, 1),
                    "roi_sem_top3_event": _roi_without_top_k(evs, 3),
                    "max_dd_abs": _drawdown_metrics(pnls)["max_drawdown_abs"],
                }
            )
        rois = [s["roi_pct"] for s in stats if s["roi_pct"] is not None]
        dds = [s["max_dd_abs"] for s in stats if s["max_dd_abs"] is not None]
        out[str(w)] = {
            "n_windows": len(stats),
            "pct_janelas_roi_positivo": (100.0 * sum(1 for r in rois if r > 0) / len(rois)) if rois else None,
            "roi_medio_janela_pct": _mean(rois),
            "roi_mediano_janela_pct": _median(rois),
            "max_drawdown_medio_janela": _mean(dds),
            "roi_sem_top1_event_medio_pct": _mean([s["roi_sem_top1_event"] for s in stats if s["roi_sem_top1_event"] is not None]),
            "roi_sem_top3_event_medio_pct": _mean([s["roi_sem_top3_event"] for s in stats if s["roi_sem_top3_event"] is not None]),
            "n_events_medio_por_janela": _mean([float(s["n_events"]) for s in stats]),
        }
    return out


def _rolling_days(bets: Sequence[Bet], windows: Sequence[int]) -> Dict[str, Any]:
    by_day: Dict[str, Dict[str, float]] = defaultdict(lambda: {"pnl": 0.0, "stake": 0.0})
    for b in bets:
        by_day[b.day]["pnl"] += b.pnl
        by_day[b.day]["stake"] += b.stake
    days = sorted(by_day.keys())
    out: Dict[str, Any] = {}
    for w in windows:
        stats = []
        for i in range(0, len(days) - w + 1):
            dsl = days[i : i + w]
            pnls = [by_day[d]["pnl"] for d in dsl]
            stakes = [by_day[d]["stake"] for d in dsl]
            stats.append(
                {
                    "roi_pct": _weighted_roi_pct(pnls, stakes),
                    "pnl": sum(pnls),
                    "median_pnl": _median(pnls),
                    "max_dd_abs": _drawdown_metrics(pnls)["max_drawdown_abs"],
                }
            )
        rois = [s["roi_pct"] for s in stats if s["roi_pct"] is not None]
        out[str(w)] = {
            "n_windows": len(stats),
            "pct_janelas_roi_positivo": (100.0 * sum(1 for r in rois if r > 0) / len(rois)) if rois else None,
            "roi_medio_janela_pct": _mean(rois),
            "roi_mediano_janela_pct": _median(rois),
            "max_drawdown_medio_janela": _mean([s["max_dd_abs"] for s in stats if s["max_dd_abs"] is not None]),
        }
    return out


def _bootstrap_unit(
    unit_values: Sequence[float],
    unit_stakes: Sequence[float],
    *,
    n_boot: int,
    seed: int,
    dd_threshold_abs: float,
) -> Dict[str, Optional[float]]:
    n = len(unit_values)
    if n == 0:
        return {
            "roi_mean_pct": None,
            "roi_ci90_lo_pct": None,
            "roi_ci90_hi_pct": None,
            "roi_ci95_lo_pct": None,
            "roi_ci95_hi_pct": None,
            "prob_roi_gt_0_pct": None,
            "prob_dd_gt_threshold_pct": None,
            "monthly_p10_pnl": None,
            "monthly_p90_pnl": None,
        }
    rng = random.Random(seed)
    rois = []
    mdd = []
    monthly = []
    horizon = min(30, max(1, n))
    for _ in range(max(1, int(n_boot))):
        idx = [rng.randrange(n) for _ in range(n)]
        pnl = [unit_values[i] for i in idx]
        st = [unit_stakes[i] for i in idx]
        roi = _weighted_roi_pct(pnl, st)
        if roi is not None:
            rois.append(roi)
        dd = _drawdown_metrics(pnl)["max_drawdown_abs"]
        if dd is not None:
            mdd.append(dd)
        idx_m = [rng.randrange(n) for _ in range(horizon)]
        monthly.append(sum(unit_values[i] for i in idx_m))
    rois_s = sorted(rois)
    mdd_s = sorted(mdd)
    monthly_s = sorted(monthly)

    def q(xs: Sequence[float], p: float) -> Optional[float]:
        return _quantile(xs, p) if xs else None

    return {
        "roi_mean_pct": _mean(rois_s),
        "roi_ci90_lo_pct": q(rois_s, 0.05),
        "roi_ci90_hi_pct": q(rois_s, 0.95),
        "roi_ci95_lo_pct": q(rois_s, 0.025),
        "roi_ci95_hi_pct": q(rois_s, 0.975),
        "prob_roi_gt_0_pct": (100.0 * sum(1 for x in rois_s if x > 0) / len(rois_s)) if rois_s else None,
        "prob_dd_gt_threshold_pct": (100.0 * sum(1 for x in mdd_s if x > dd_threshold_abs) / len(mdd_s)) if mdd_s else None,
        "monthly_p10_pnl": q(monthly_s, 0.10),
        "monthly_p90_pnl": q(monthly_s, 0.90),
    }


def _pairs_roi_table(rows: Iterable[Tuple[str, Optional[float], float, float]]) -> List[Dict[str, Any]]:
    agg: Dict[str, Dict[str, float]] = defaultdict(lambda: {"pnl": 0.0, "stake": 0.0, "n": 0.0})
    for key, _, pnl, stake in rows:
        agg[key]["pnl"] += pnl
        agg[key]["stake"] += stake
        agg[key]["n"] += 1.0
    out = []
    for k, v in sorted(agg.items(), key=lambda kv: kv[0]):
        out.append(
            {
                "segmento": k,
                "n": int(v["n"]),
                "stake": v["stake"],
                "pnl": v["pnl"],
                "roi_pct": (100.0 * v["pnl"] / v["stake"]) if v["stake"] > 0 else None,
            }
        )
    return out


def analyze(
    bets: Sequence[Bet],
    *,
    odds_edges: Sequence[float],
    slip_edges: Sequence[float],
    latency_edges: Sequence[float],
    loss_thresholds: Sequence[float],
    boot_iters: int,
    boot_seed: int,
    dd_threshold_abs: float,
) -> Dict[str, Any]:
    if not bets:
        return {"error": "sem_linhas_apos_filtros"}

    pnls = [b.pnl for b in bets]
    stakes = [b.stake for b in bets]
    rois = [b.roi_pct for b in bets]
    total_pnl = sum(pnls)
    total_stake = sum(stakes)
    roi_total = _weighted_roi_pct(pnls, stakes)

    # 1) Distribuicao por aposta
    wins = [b.pnl for b in bets if b.pnl > 0]
    losses = [b.pnl for b in bets if b.pnl < 0]
    aposta_dist = {
        "n_bets": len(bets),
        "stake_total": total_stake,
        "pnl_total": total_pnl,
        "roi_total_pct": roi_total,
        "pnl_medio": _mean(pnls),
        "pnl_mediano": _median(pnls),
        "roi_medio_aposta_pct": _mean(rois),
        "roi_mediano_aposta_pct": _median(rois),
        "pnl_p10": _quantile(pnls, 0.10),
        "pnl_p25": _quantile(pnls, 0.25),
        "pnl_p75": _quantile(pnls, 0.75),
        "pnl_p90": _quantile(pnls, 0.90),
        "pct_apostas_positivas": 100.0 * len(wins) / len(bets),
        "pnl_medio_vencedoras": _mean(wins),
        "pnl_medio_perdedoras": _mean(losses),
    }

    # 2) Evento
    events = _aggregate_events(bets)
    ev_pnls = [e["pnl"] for e in events]
    ev_stakes = [e["stake"] for e in events]
    evento_dist = {
        "n_events": len(events),
        "pnl_medio_evento": _mean(ev_pnls),
        "pnl_mediano_evento": _median(ev_pnls),
        "roi_total_evento_pct": _weighted_roi_pct(ev_pnls, ev_stakes),
        "apostas_por_evento_media": _mean([float(e["n_bets"]) for e in events]),
        "apostas_por_evento_mediana": _median([float(e["n_bets"]) for e in events]),
        "exposicao_max_evento": max((e["max_exposure"] for e in events), default=None),
        "top3_eventos_contrib_pct": _top_contrib([e["pnl"] for e in events], total_pnl, 3),
        "roi_sem_top3_eventos_pct": _roi_without_top_k(events, 3),
    }

    # 3) Curvas
    by_day = defaultdict(lambda: {"pnl": 0.0, "stake": 0.0})
    for b in bets:
        by_day[b.day]["pnl"] += b.pnl
        by_day[b.day]["stake"] += b.stake
    day_keys = sorted(by_day.keys())
    day_pnls = [by_day[d]["pnl"] for d in day_keys]
    event_sorted = sorted(events, key=lambda e: e["first_ts"])
    curve = {
        "acumulado_por_aposta": _drawdown_metrics(pnls),
        "acumulado_por_dia": _drawdown_metrics(day_pnls),
        "acumulado_por_evento": _drawdown_metrics([e["pnl"] for e in event_sorted]),
        "maior_seq_perdas_apostas": _longest_negative_streak(pnls),
        "maior_seq_dias_negativos": _longest_negative_streak(day_pnls),
    }

    # 4) Rolling
    rolling = {
        "bets": _rolling_bets(bets, [20, 50, 100]),
        "days": _rolling_days(bets, [7, 14, 28]),
    }

    # 5) Concentracao
    conc = {
        "top1_aposta_contrib_pct": _top_contrib(pnls, total_pnl, 1),
        "top3_apostas_contrib_pct": _top_contrib(pnls, total_pnl, 3),
        "top5_apostas_contrib_pct": _top_contrib(pnls, total_pnl, 5),
        "top1_evento_contrib_pct": _top_contrib(ev_pnls, total_pnl, 1),
        "top3_eventos_contrib_pct": _top_contrib(ev_pnls, total_pnl, 3),
        "top5_eventos_contrib_pct": _top_contrib(ev_pnls, total_pnl, 5),
        "roi_sem_top3_apostas_pct": _roi_without_top_k_bets(bets, 3),
        "roi_sem_top3_eventos_pct": _roi_without_top_k(events, 3),
        "roi_sem_top5_apostas_pct": _roi_without_top_k_bets(bets, 5),
        "roi_sem_top5_eventos_pct": _roi_without_top_k(events, 5),
    }

    # 6) Perdas
    loss_abs = sorted([abs(x) for x in losses])
    neg_events = [e for e in events if e["pnl"] < 0]
    neg_days = [d for d in day_keys if by_day[d]["pnl"] < 0]
    losses_table = {}
    for thr in loss_thresholds:
        losses_table[f"days_loss_gt_{thr:g}"] = sum(1 for d in day_keys if by_day[d]["pnl"] < -abs(thr))
    sorted_loss_bets = sorted([b for b in bets if b.pnl < 0], key=lambda b: abs(b.pnl), reverse=True)
    total_loss_abs = sum(abs(b.pnl) for b in sorted_loss_bets)
    cum = 0.0
    k80_b = 0
    for b in sorted_loss_bets:
        cum += abs(b.pnl)
        k80_b += 1
        if total_loss_abs > 0 and cum / total_loss_abs >= 0.80:
            break
    sorted_loss_events = sorted(neg_events, key=lambda e: abs(e["pnl"]), reverse=True)
    total_ev_loss_abs = sum(abs(e["pnl"]) for e in sorted_loss_events)
    cum2 = 0.0
    k80_e = 0
    for e in sorted_loss_events:
        cum2 += abs(e["pnl"])
        k80_e += 1
        if total_ev_loss_abs > 0 and cum2 / total_ev_loss_abs >= 0.80:
            break
    loss_dist = {
        "n_apostas_negativas": len(losses),
        "media_perdas": _mean(losses),
        "mediana_perdas": _median(losses),
        "pior_perda": min(losses) if losses else None,
        "p95_perda_abs": _quantile(loss_abs, 0.95),
        "p99_perda_abs": _quantile(loss_abs, 0.99),
        "perda_media_por_evento_negativo": _mean([e["pnl"] for e in neg_events]),
        "perda_maxima_por_evento": min([e["pnl"] for e in events], default=None),
        "n_dias_negativos": len(neg_days),
        "responsaveis_80_perdas_apostas_n": k80_b if total_loss_abs > 0 else 0,
        "responsaveis_80_perdas_eventos_n": k80_e if total_ev_loss_abs > 0 else 0,
        "threshold_days": losses_table,
    }

    # 7) Buckets odds
    odds_rows = []
    for b in bets:
        key = _bucket(b.odd_taken, odds_edges)
        odds_rows.append((key, b.ev_pct, b.pnl, b.stake, b.roi_pct, b.odd_fair))
    by_odds = defaultdict(
        lambda: {
            "n": 0,
            "stake": 0.0,
            "pnl": 0.0,
            "roi_aposta": [],
            "pnl_aposta": [],
            "win": 0,
            "odd_fair_vals": [],
            "ev_stake": 0.0,
            "ev_num": 0.0,
        }
    )
    for key, ev_pct, pnl, stake, roi_pct, odd_fair in odds_rows:
        rec = by_odds[key]
        rec["n"] += 1
        rec["stake"] += stake
        rec["pnl"] += pnl
        rec["roi_aposta"].append(roi_pct)
        rec["pnl_aposta"].append(pnl)
        if pnl > 0:
            rec["win"] += 1
        if odd_fair and odd_fair > 0:
            rec["odd_fair_vals"].append(odd_fair)
        if ev_pct is not None:
            rec["ev_stake"] += stake * (ev_pct / 100.0)
            rec["ev_num"] += stake
    odds_table = []
    for k in sorted(by_odds.keys()):
        r = by_odds[k]
        win_rate = 100.0 * r["win"] / r["n"] if r["n"] else None
        fair_implied = None
        if r["odd_fair_vals"]:
            fair = _mean(r["odd_fair_vals"])
            fair_implied = (100.0 / fair) if fair and fair > 0 else None
        ev_pct_bucket = (100.0 * r["ev_stake"] / r["ev_num"]) if r["ev_num"] > 0 else None
        odds_table.append(
            {
                "bucket": k,
                "n": r["n"],
                "stake": r["stake"],
                "pnl": r["pnl"],
                "roi_realizado_pct": (100.0 * r["pnl"] / r["stake"]) if r["stake"] > 0 else None,
                "pnl_medio": (r["pnl"] / r["n"]) if r["n"] else None,
                "pnl_mediano": _median(r["pnl_aposta"]),
                "roi_mediano_aposta_pct": _median(r["roi_aposta"]),
                "win_rate_realizado_pct": win_rate,
                "win_rate_implicito_odd_justa_pct": fair_implied,
                "ev_esperado_pct_ponderado": ev_pct_bucket,
                "pnl_esperado": r["ev_stake"],
            }
        )

    # 8) EV esperado vs realizado
    ev_bets = [b for b in bets if b.ev_pct is not None]
    ev_summary = None
    if ev_bets:
        exp_total = sum(b.stake * (float(b.ev_pct) / 100.0) for b in ev_bets if b.ev_pct is not None)
        real_total = sum(b.pnl for b in ev_bets)
        stake_total_ev = sum(b.stake for b in ev_bets)
        ev_summary = {
            "n_bets_com_ev": len(ev_bets),
            "stake_coberta_ev": stake_total_ev,
            "ev_medio_ponderado_pct": (100.0 * exp_total / stake_total_ev) if stake_total_ev > 0 else None,
            "ev_mediano_pct": _median([float(b.ev_pct) for b in ev_bets if b.ev_pct is not None]),
            "pct_stake_ev_positivo": (
                100.0 * sum(b.stake for b in ev_bets if b.ev_pct is not None and b.ev_pct > 0) / stake_total_ev
                if stake_total_ev > 0
                else None
            ),
            "pnl_esperado_total": exp_total,
            "pnl_realizado_total": real_total,
            "capture_ratio": (real_total / exp_total) if abs(exp_total) > 1e-12 else None,
        }

    # 9) Slippage granular
    slip_vals = [b.slippage for b in bets if b.slippage is not None]
    slip_rois = [b.roi_pct for b in bets if b.slippage is not None]
    corr = None
    if len(slip_vals) > 2:
        try:
            corr = statistics.correlation(slip_vals, slip_rois)
        except Exception:
            corr = None
    slip_sorted = sorted([b for b in bets if b.slippage is not None], key=lambda x: float(x.slippage))
    dec = []
    if slip_sorted:
        for d in range(10):
            a = int(d * len(slip_sorted) / 10)
            b = int((d + 1) * len(slip_sorted) / 10)
            sl = slip_sorted[a:b]
            if not sl:
                continue
            dec.append(
                {
                    "decil": d + 1,
                    "slip_min": sl[0].slippage,
                    "slip_max": sl[-1].slippage,
                    "roi_pct": _weighted_roi_pct([x.pnl for x in sl], [x.stake for x in sl]),
                    "n": len(sl),
                }
            )
    slip_combo_odds = _pairs_roi_table(
        (
            f"slip={_bucket(b.slippage, slip_edges)}|odds={_bucket(b.odd_taken, odds_edges)}",
            b.slippage,
            b.pnl,
            b.stake,
        )
        for b in bets
    )
    slip_combo_lat = _pairs_roi_table(
        (
            f"slip={_bucket(b.slippage, slip_edges)}|lat={_bucket(b.latency_sec, latency_edges)}",
            b.slippage,
            b.pnl,
            b.stake,
        )
        for b in bets
    )
    slip_analysis = {
        "corr_slippage_vs_roi": corr,
        "decil_slippage": dec,
        "slippage_medio_vencedoras": _mean([float(b.slippage) for b in bets if b.slippage is not None and b.pnl > 0]),
        "slippage_medio_perdedoras": _mean([float(b.slippage) for b in bets if b.slippage is not None and b.pnl < 0]),
        "roi_por_slippage_odds": slip_combo_odds,
        "roi_por_slippage_latencia": slip_combo_lat,
    }

    # 10) Latencia explicativa
    lat_bucket_tbl = _pairs_roi_table(
        (f"lat={_bucket(b.latency_sec, latency_edges)}", b.latency_sec, b.pnl, b.stake) for b in bets
    )
    lat_slip = _pairs_roi_table(
        (f"lat={_bucket(b.latency_sec, latency_edges)}|slip={_bucket(b.slippage, slip_edges)}", b.latency_sec, b.pnl, b.stake)
        for b in bets
    )
    lat_odds = _pairs_roi_table(
        (f"lat={_bucket(b.latency_sec, latency_edges)}|odds={_bucket(b.odd_taken, odds_edges)}", b.latency_sec, b.pnl, b.stake)
        for b in bets
    )
    hour_edges = [0.0, 6.0, 12.0, 18.0]
    lat_hour = _pairs_roi_table(
        (f"lat={_bucket(b.latency_sec, latency_edges)}|hora={_bucket(float(b.hour), hour_edges)}", b.latency_sec, b.pnl, b.stake)
        for b in bets
    )
    lat_sorted = sorted([b for b in bets if b.latency_sec is not None], key=lambda x: float(x.latency_sec), reverse=True)
    lat_outliers = [
        {"row_id": b.row_id, "ts": b.ts.isoformat(), "latency_sec": b.latency_sec, "pnl": b.pnl, "event_id": b.event_id}
        for b in lat_sorted[:10]
    ]
    latency_analysis = {
        "roi_por_bucket_latencia": lat_bucket_tbl,
        "roi_latencia_slippage": lat_slip,
        "roi_latencia_odds": lat_odds,
        "roi_latencia_horario": lat_hour,
        "outliers_latencia": lat_outliers,
    }

    # 11) Bootstrap
    by_event_units = _aggregate_events(bets)
    by_day_units = [{"pnl": by_day[d]["pnl"], "stake": by_day[d]["stake"]} for d in day_keys]
    boot = {
        "por_aposta": _bootstrap_unit(pnls, stakes, n_boot=boot_iters, seed=boot_seed + 1, dd_threshold_abs=dd_threshold_abs),
        "por_evento": _bootstrap_unit(
            [x["pnl"] for x in by_event_units],
            [x["stake"] for x in by_event_units],
            n_boot=boot_iters,
            seed=boot_seed + 2,
            dd_threshold_abs=dd_threshold_abs,
        ),
        "por_dia": _bootstrap_unit(
            [x["pnl"] for x in by_day_units],
            [x["stake"] for x in by_day_units],
            n_boot=boot_iters,
            seed=boot_seed + 3,
            dd_threshold_abs=dd_threshold_abs,
        ),
    }

    # 12) Dashboard
    dashboard = {
        "resultado": {
            "pnl_total": total_pnl,
            "roi_total_pct": roi_total,
            "pnl_medio_aposta": aposta_dist["pnl_medio"],
            "pnl_mediano_aposta": aposta_dist["pnl_mediano"],
            "roi_mediano_aposta_pct": aposta_dist["roi_mediano_aposta_pct"],
        },
        "distribuicao": {
            "p10_pnl": aposta_dist["pnl_p10"],
            "p25_pnl": aposta_dist["pnl_p25"],
            "p75_pnl": aposta_dist["pnl_p75"],
            "p90_pnl": aposta_dist["pnl_p90"],
            "pior_aposta": min(pnls) if pnls else None,
            "melhor_aposta": max(pnls) if pnls else None,
        },
        "curva": curve["acumulado_por_aposta"],
        "concentracao": {
            "top3_eventos_contrib_pct": conc["top3_eventos_contrib_pct"],
            "roi_sem_top3_eventos_pct": conc["roi_sem_top3_eventos_pct"],
            "roi_sem_top3_apostas_pct": conc["roi_sem_top3_apostas_pct"],
        },
        "ev": ev_summary or {"status": "ev_nao_disponivel"},
        "rolling_50_100": {
            "rolling50": rolling["bets"].get("50"),
            "rolling100": rolling["bets"].get("100"),
        },
        "bootstrap_evento": boot["por_evento"],
    }

    return {
        "meta": {
            "n_bets": len(bets),
            "n_events": len(events),
            "n_days": len(day_keys),
            "first_ts": bets[0].ts.isoformat(),
            "last_ts": bets[-1].ts.isoformat(),
        },
        "dist_aposta": aposta_dist,
        "dist_evento": evento_dist,
        "curvas": curve,
        "rolling": rolling,
        "concentracao": conc,
        "perdas": loss_dist,
        "odds_buckets": odds_table,
        "ev_esperado_vs_realizado": ev_summary,
        "slippage": slip_analysis,
        "latencia": latency_analysis,
        "bootstrap": boot,
        "dashboard": dashboard,
    }


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---" for _ in headers]) + "|")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return out


def render_markdown(
    result: Dict[str, Any],
    *,
    input_csv: Path,
    mapping: Dict[str, str],
    dropped: Dict[str, int],
) -> str:
    if result.get("error"):
        return f"# Analise avancada CSV\n\nErro: {result['error']}\n"

    d = result
    lines: List[str] = []
    lines.append("# Analise avancada da estrategia (CSV)\n")
    lines.append(f"- input_csv: `{input_csv}`")
    lines.append(f"- periodo: `{d['meta']['first_ts']}` -> `{d['meta']['last_ts']}`")
    lines.append(f"- n_bets: **{d['meta']['n_bets']}** | n_events: **{d['meta']['n_events']}** | n_days: **{d['meta']['n_days']}**\n")
    lines.append("## Mapeamento de colunas")
    for k, v in mapping.items():
        lines.append(f"- {k}: `{v}`")
    if dropped:
        lines.append("\n## Linhas descartadas")
        for k, v in sorted(dropped.items(), key=lambda kv: kv[0]):
            lines.append(f"- {k}: {v}")

    a = d["dist_aposta"]
    lines.append("\n## 1) Distribuicao por aposta")
    lines.extend(
        _md_table(
            ["Metrica", "Valor"],
            [
                ["P&L total", _fmt(a["pnl_total"], 2)],
                ["ROI total", _fmt(a["roi_total_pct"], 2, True)],
                ["P&L medio/aposta", _fmt(a["pnl_medio"], 4)],
                ["P&L mediano/aposta", _fmt(a["pnl_mediano"], 4)],
                ["ROI medio/aposta", _fmt(a["roi_medio_aposta_pct"], 2, True)],
                ["ROI mediano/aposta", _fmt(a["roi_mediano_aposta_pct"], 2, True)],
                ["P10/P25/P75/P90 P&L", f"{_fmt(a['pnl_p10'],2)} / {_fmt(a['pnl_p25'],2)} / {_fmt(a['pnl_p75'],2)} / {_fmt(a['pnl_p90'],2)}"],
                ["% apostas positivas", _fmt(a["pct_apostas_positivas"], 2, True)],
                ["P&L medio vencedoras", _fmt(a["pnl_medio_vencedoras"], 4)],
                ["P&L medio perdedoras", _fmt(a["pnl_medio_perdedoras"], 4)],
            ],
        )
    )

    e = d["dist_evento"]
    lines.append("\n## 2) Distribuicao por evento")
    lines.extend(
        _md_table(
            ["Metrica", "Valor"],
            [
                ["P&L medio por evento", _fmt(e["pnl_medio_evento"], 4)],
                ["P&L mediano por evento", _fmt(e["pnl_mediano_evento"], 4)],
                ["ROI total por evento", _fmt(e["roi_total_evento_pct"], 2, True)],
                ["Apostas por evento (media/mediana)", f"{_fmt(e['apostas_por_evento_media'],2)} / {_fmt(e['apostas_por_evento_mediana'],2)}"],
                ["Exposicao maxima por evento", _fmt(e["exposicao_max_evento"], 2)],
                ["Top-3 eventos contribuicao", _fmt(e["top3_eventos_contrib_pct"], 2, True)],
                ["ROI sem Top-3 eventos", _fmt(e["roi_sem_top3_eventos_pct"], 2, True)],
            ],
        )
    )

    c = d["curvas"]
    lines.append("\n## 3) Curvas acumuladas e risco")
    lines.extend(
        _md_table(
            ["Curva", "MaxDD", "MaxDD%", "TUW", "Recovery", "Ulcer"],
            [
                [
                    "Aposta",
                    _fmt(c["acumulado_por_aposta"]["max_drawdown_abs"], 2),
                    _fmt(c["acumulado_por_aposta"]["max_drawdown_pct_on_peak"], 2, True),
                    _fmt(c["acumulado_por_aposta"]["time_under_water_steps"], 0),
                    _fmt(c["acumulado_por_aposta"]["recovery_factor"], 3),
                    _fmt(c["acumulado_por_aposta"]["ulcer_index"], 3),
                ],
                [
                    "Dia",
                    _fmt(c["acumulado_por_dia"]["max_drawdown_abs"], 2),
                    _fmt(c["acumulado_por_dia"]["max_drawdown_pct_on_peak"], 2, True),
                    _fmt(c["acumulado_por_dia"]["time_under_water_steps"], 0),
                    _fmt(c["acumulado_por_dia"]["recovery_factor"], 3),
                    _fmt(c["acumulado_por_dia"]["ulcer_index"], 3),
                ],
                [
                    "Evento",
                    _fmt(c["acumulado_por_evento"]["max_drawdown_abs"], 2),
                    _fmt(c["acumulado_por_evento"]["max_drawdown_pct_on_peak"], 2, True),
                    _fmt(c["acumulado_por_evento"]["time_under_water_steps"], 0),
                    _fmt(c["acumulado_por_evento"]["recovery_factor"], 3),
                    _fmt(c["acumulado_por_evento"]["ulcer_index"], 3),
                ],
            ],
        )
    )
    lines.append(f"- Maior sequencia de perdas (apostas): **{int(c['maior_seq_perdas_apostas'])}**")
    lines.append(f"- Maior sequencia de dias negativos: **{int(c['maior_seq_dias_negativos'])}**")

    roll = d["rolling"]
    lines.append("\n## 4) Rolling windows (resumo)")
    rb50 = roll["bets"].get("50", {}) or {}
    rb100 = roll["bets"].get("100", {}) or {}
    rd14 = roll["days"].get("14", {}) or {}
    lines.extend(
        _md_table(
            ["Janela", "n", "% ROI>0", "ROI medio", "ROI mediano", "ROI sem Top-3 evento"],
            [
                [
                    "50 apostas",
                    str(rb50.get("n_windows", 0)),
                    _fmt(rb50.get("pct_janelas_roi_positivo"), 2, True),
                    _fmt(rb50.get("roi_medio_janela_pct"), 2, True),
                    _fmt(rb50.get("roi_mediano_janela_pct"), 2, True),
                    _fmt(rb50.get("roi_sem_top3_event_medio_pct"), 2, True),
                ],
                [
                    "100 apostas",
                    str(rb100.get("n_windows", 0)),
                    _fmt(rb100.get("pct_janelas_roi_positivo"), 2, True),
                    _fmt(rb100.get("roi_medio_janela_pct"), 2, True),
                    _fmt(rb100.get("roi_mediano_janela_pct"), 2, True),
                    _fmt(rb100.get("roi_sem_top3_event_medio_pct"), 2, True),
                ],
                [
                    "14 dias",
                    str(rd14.get("n_windows", 0)),
                    _fmt(rd14.get("pct_janelas_roi_positivo"), 2, True),
                    _fmt(rd14.get("roi_medio_janela_pct"), 2, True),
                    _fmt(rd14.get("roi_mediano_janela_pct"), 2, True),
                    "NA",
                ],
            ],
        )
    )

    conc = d["concentracao"]
    lines.append("\n## 5) Concentracao")
    lines.extend(
        _md_table(
            ["Corte", "% P&L total", "ROI residual"],
            [
                ["Top 1 aposta", _fmt(conc["top1_aposta_contrib_pct"], 2, True), "-"],
                ["Top 3 apostas", _fmt(conc["top3_apostas_contrib_pct"], 2, True), _fmt(conc["roi_sem_top3_apostas_pct"], 2, True)],
                ["Top 5 apostas", _fmt(conc["top5_apostas_contrib_pct"], 2, True), _fmt(conc["roi_sem_top5_apostas_pct"], 2, True)],
                ["Top 1 evento", _fmt(conc["top1_evento_contrib_pct"], 2, True), "-"],
                ["Top 3 eventos", _fmt(conc["top3_eventos_contrib_pct"], 2, True), _fmt(conc["roi_sem_top3_eventos_pct"], 2, True)],
                ["Top 5 eventos", _fmt(conc["top5_eventos_contrib_pct"], 2, True), _fmt(conc["roi_sem_top5_eventos_pct"], 2, True)],
            ],
        )
    )

    p = d["perdas"]
    lines.append("\n## 6) Distribuicao de perdas")
    lines.extend(
        _md_table(
            ["Metrica", "Valor"],
            [
                ["n apostas negativas", str(p["n_apostas_negativas"])],
                ["media perdas", _fmt(p["media_perdas"], 4)],
                ["mediana perdas", _fmt(p["mediana_perdas"], 4)],
                ["pior perda", _fmt(p["pior_perda"], 2)],
                ["p95/p99 perda abs", f"{_fmt(p['p95_perda_abs'],2)} / {_fmt(p['p99_perda_abs'],2)}"],
                ["perda media por evento negativo", _fmt(p["perda_media_por_evento_negativo"], 3)],
                ["perda maxima por evento", _fmt(p["perda_maxima_por_evento"], 2)],
                ["n dias negativos", str(p["n_dias_negativos"])],
                ["n apostas p/ 80% perdas", str(p["responsaveis_80_perdas_apostas_n"])],
                ["n eventos p/ 80% perdas", str(p["responsaveis_80_perdas_eventos_n"])],
            ],
        )
    )

    lines.append("\n## 7) Buckets por odds")
    odds_rows = []
    for r in d["odds_buckets"]:
        odds_rows.append(
            [
                str(r["bucket"]),
                str(r["n"]),
                _fmt(r["stake"], 1),
                _fmt(r["roi_realizado_pct"], 2, True),
                _fmt(r["pnl_medio"], 3),
                _fmt(r["pnl_mediano"], 3),
                _fmt(r["roi_mediano_aposta_pct"], 2, True),
                _fmt(r["win_rate_realizado_pct"], 2, True),
                _fmt(r["ev_esperado_pct_ponderado"], 2, True),
            ]
        )
    lines.extend(_md_table(["Bucket", "n", "Stake", "ROI", "P&L medio", "P&L mediano", "ROI mediana", "Win rate", "EV%"], odds_rows))

    lines.append("\n## 8) EV esperado vs realizado")
    ev = d["ev_esperado_vs_realizado"]
    if not ev:
        lines.append("- EV nao disponivel (faltam colunas de odd tomada/justa).")
    else:
        lines.extend(
            _md_table(
                ["Metrica", "Valor"],
                [
                    ["n bets com EV", str(ev["n_bets_com_ev"])],
                    ["EV medio ponderado", _fmt(ev["ev_medio_ponderado_pct"], 2, True)],
                    ["EV mediano", _fmt(ev["ev_mediano_pct"], 2, True)],
                    ["% stake EV+", _fmt(ev["pct_stake_ev_positivo"], 2, True)],
                    ["P&L esperado total", _fmt(ev["pnl_esperado_total"], 2)],
                    ["P&L realizado total", _fmt(ev["pnl_realizado_total"], 2)],
                    ["Capture ratio", _fmt(ev["capture_ratio"], 3)],
                ],
            )
        )

    slip = d["slippage"]
    lines.append("\n## 9) Slippage (granular)")
    lines.extend(
        _md_table(
            ["Metrica", "Valor"],
            [
                ["Correlacao slippage vs ROI", _fmt(slip["corr_slippage_vs_roi"], 4)],
                ["Slippage medio vencedoras", _fmt(slip["slippage_medio_vencedoras"], 4)],
                ["Slippage medio perdedoras", _fmt(slip["slippage_medio_perdedoras"], 4)],
                ["Decis de slippage", str(len(slip["decil_slippage"]))],
            ],
        )
    )

    lat = d["latencia"]
    lines.append("\n## 10) Latencia (resumo)")
    lines.append(f"- segmentos latencia: {len(lat['roi_por_bucket_latencia'])}")
    lines.append(f"- segmentos latencia+slippage: {len(lat['roi_latencia_slippage'])}")
    lines.append(f"- segmentos latencia+odds: {len(lat['roi_latencia_odds'])}")
    lines.append(f"- outliers de latencia (top): {len(lat['outliers_latencia'])}")

    lines.append("\n## 11) Bootstrap (resumo)")
    bs = d["bootstrap"]
    bs_rows = []
    for k in ["por_aposta", "por_evento", "por_dia"]:
        b = bs[k]
        bs_rows.append(
            [
                k,
                _fmt(b["roi_mean_pct"], 2, True),
                f"{_fmt(b['roi_ci90_lo_pct'],2,True)} / {_fmt(b['roi_ci90_hi_pct'],2,True)}",
                _fmt(b["prob_roi_gt_0_pct"], 2, True),
                _fmt(b["prob_dd_gt_threshold_pct"], 2, True),
                f"{_fmt(b['monthly_p10_pnl'],2)} / {_fmt(b['monthly_p90_pnl'],2)}",
            ]
        )
    lines.extend(
        _md_table(
            ["Bootstrap", "ROI medio", "IC90 ROI", "P(ROI>0)", "P(DD>X)", "Mensal p10/p90"],
            bs_rows,
        )
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Analise avancada do CSV da estrategia")
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-md", default="")
    ap.add_argument("--enforce-slip-neg", type=int, default=1)
    ap.add_argument("--filter-back-pre", type=int, default=1)
    ap.add_argument("--odds-buckets", default="1.20,1.50,1.80,2.20,3.00")
    ap.add_argument("--slippage-buckets", default="-1.00,-0.25,0.00,0.25")
    ap.add_argument("--latency-buckets", default="4,6,8")
    ap.add_argument("--loss-thresholds", default="100,300,500")
    ap.add_argument("--bootstrap-iters", type=int, default=3000)
    ap.add_argument("--bootstrap-seed", type=int, default=1337)
    ap.add_argument("--dd-threshold-abs", type=float, default=300.0)
    args = ap.parse_args()

    csv_path = Path(args.input_csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV nao encontrado: {csv_path}")

    bets, mapping, dropped = _load_bets(
        csv_path,
        enforce_slip_neg=bool(int(args.enforce_slip_neg)),
        filter_back_pre=bool(int(args.filter_back_pre)),
    )
    res = analyze(
        bets,
        odds_edges=_parse_edges(args.odds_buckets),
        slip_edges=_parse_edges(args.slippage_buckets),
        latency_edges=_parse_edges(args.latency_buckets),
        loss_thresholds=_parse_edges(args.loss_thresholds),
        boot_iters=int(args.bootstrap_iters),
        boot_seed=int(args.bootstrap_seed),
        dd_threshold_abs=float(args.dd_threshold_abs),
    )

    payload = {
        "input_csv": str(csv_path),
        "mapping": mapping,
        "dropped": dropped,
        "result": res,
    }

    out_json = Path(args.out_json) if str(args.out_json).strip() else Path(
        f"/tmp/analise_csv_avancada_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_md = Path(args.out_md) if str(args.out_md).strip() else out_json.with_suffix(".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(
        render_markdown(res, input_csv=csv_path, mapping=mapping, dropped=dropped),
        encoding="utf-8",
    )
    print(f"[OK] JSON: {out_json}")
    print(f"[OK] MD  : {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

