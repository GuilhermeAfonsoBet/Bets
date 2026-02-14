#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera relatório estatístico ROBUSTO (cluster por jogo) no estilo do:
  docs/analise_h3b_resultados_v4_somente.md

Foco: H3B (reversal_direction), comparando modelos por audit_version.

Robustez adicionada:
- métricas reportadas com N_eventos e N_jogos
- IC via bootstrap por cluster (jogo/match_id), reduzindo viés por correlação intra-jogo

Uso:
  # (requer .env com DATABASE_URL; BETINASIA_USERNAME/PASSWORD podem ser dummy p/ Settings)
  python analyze_contexto_operacao_b808_robust_report.py --out docs/analise_contexto_operacao_b808_robusta.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sqlalchemy import Integer, bindparam, text

# Mantém padrão dos scripts existentes
import sys

sys.path.insert(0, ".")

from storage.database import Database


Z_90 = 1.645
Z_95 = 1.960


@dataclass(frozen=True)
class MetricSummary:
    n_events: int
    n_matches: int
    mean_event: Optional[float]
    ci90_event: Optional[Tuple[float, float]]
    mean_cluster: Optional[float]
    ci90_cluster: Optional[Tuple[float, float]]
    median_event: Optional[float]
    p25_event: Optional[float]
    p75_event: Optional[float]
    hit_rate_event: Optional[float]  # % valores > 0 (exclui zeros)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return None
        return xf
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        xi = int(x)
        return xi
    except Exception:
        return None


def _as_dict(x: Any) -> Optional[dict]:
    """
    Normaliza hypothesis_details: pode vir como dict (JSON/JSONB) ou string JSON.
    """
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    if isinstance(x, (str, bytes)):
        try:
            return json.loads(x)
        except Exception:
            return None
    return None


def _get_path(d: Optional[dict], path: Sequence[str]) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _pctl(xs: Sequence[float], q: float) -> Optional[float]:
    if not xs:
        return None
    return float(np.percentile(np.asarray(xs, dtype=float), q))


def _es_tail(xs: Sequence[float], q: float = 95.0) -> Optional[float]:
    """
    Expected Shortfall (média da cauda acima do percentil q).
    Usado para risco (ex.: ES95 de liability).
    """
    if not xs:
        return None
    arr = np.asarray(xs, dtype=float)
    thr = np.percentile(arr, q)
    tail = arr[arr >= thr]
    return float(np.mean(tail)) if tail.size else None


def _mean(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _std(xs: Sequence[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    m = _mean(xs)
    assert m is not None
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return float(math.sqrt(var))


def _normal_ci(mean: Optional[float], std: Optional[float], n: int, z: float) -> Optional[Tuple[float, float]]:
    if mean is None or std is None or n < 2:
        return None
    se = std / math.sqrt(n)
    return (float(mean - z * se), float(mean + z * se))


def _percentiles(xs: Sequence[float], ps: Sequence[float]) -> List[Optional[float]]:
    if not xs:
        return [None for _ in ps]
    arr = np.asarray(xs, dtype=float)
    return [float(np.percentile(arr, p)) for p in ps]


def cluster_bootstrap_ci(
    values_by_match: Dict[int, List[float]],
    n_boot: int = 4000,
    alpha: float = 0.10,
    seed: int = 1337,
) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
    """
    Bootstrap por cluster (match_id).
    Estimador: média dos MEANS por match (cada jogo pesa 1).
    """
    match_ids = list(values_by_match.keys())
    if not match_ids:
        return None, None

    # mean por match
    per_match_means: Dict[int, float] = {}
    for mid, vals in values_by_match.items():
        if not vals:
            continue
        per_match_means[mid] = float(sum(vals) / len(vals))

    match_ids = list(per_match_means.keys())
    if len(match_ids) < 2:
        m = per_match_means[match_ids[0]] if match_ids else None
        return m, None

    rng = random.Random(seed)
    boot = []
    for _ in range(int(n_boot)):
        sample = [per_match_means[rng.choice(match_ids)] for _ in range(len(match_ids))]
        boot.append(float(sum(sample) / len(sample)))

    boot_arr = np.asarray(boot, dtype=float)
    mean_hat = float(np.mean(boot_arr))
    lo = float(np.quantile(boot_arr, alpha / 2))
    hi = float(np.quantile(boot_arr, 1 - alpha / 2))
    return mean_hat, (lo, hi)


def summarize_metric(
    values: Sequence[float],
    match_ids: Sequence[int],
    clip_low: Optional[float] = None,
    clip_high: Optional[float] = None,
) -> MetricSummary:
    # filtro + sanity
    v = []
    mids = []
    for x, mid in zip(values, match_ids):
        xf = _safe_float(x)
        if xf is None:
            continue
        if clip_low is not None and xf < clip_low:
            continue
        if clip_high is not None and xf > clip_high:
            continue
        v.append(float(xf))
        mids.append(int(mid))

    n_events = len(v)
    n_matches = len(set(mids))

    mean_event = _mean(v)
    std_event = _std(v)
    ci90_event = _normal_ci(mean_event, std_event, n_events, Z_90)

    median_event, p25_event, p75_event = _percentiles(v, [50, 25, 75])

    # hit rate (exclui zeros, como no log)
    pos = sum(1 for x in v if x > 0)
    neg = sum(1 for x in v if x < 0)
    hit_rate_event = (pos / (pos + neg) * 100.0) if (pos + neg) > 0 else None

    by_match: Dict[int, List[float]] = {}
    for x, mid in zip(v, mids):
        by_match.setdefault(mid, []).append(x)

    mean_cluster, ci90_cluster = cluster_bootstrap_ci(by_match, n_boot=4000, alpha=0.10)

    return MetricSummary(
        n_events=n_events,
        n_matches=n_matches,
        mean_event=mean_event,
        ci90_event=ci90_event,
        mean_cluster=mean_cluster,
        ci90_cluster=ci90_cluster,
        median_event=median_event,
        p25_event=p25_event,
        p75_event=p75_event,
        hit_rate_event=hit_rate_event,
    )


def _sig_label(ci: Optional[Tuple[float, float]]) -> str:
    if not ci:
        return "N/A"
    if ci[0] > 0:
        return "sig. positivo"
    if ci[1] < 0:
        return "sig. negativo"
    return "NS"


def _fmt_pct(x: Optional[float], digits: int = 3) -> str:
    if x is None:
        return "—"
    return f"{x:+.{digits}f}%"


def _fmt_num(x: Optional[float], digits: int = 1) -> str:
    if x is None:
        return "—"
    return f"{x:.{digits}f}"


def _fmt_ci(ci: Optional[Tuple[float, float]], digits: int = 3, suffix: str = "%") -> str:
    if not ci:
        return "—"
    return f"[{ci[0]:+.{digits}f}{suffix}, {ci[1]:+.{digits}f}{suffix}]"


def _redact_db_url(db_url: str) -> str:
    """
    Remove senha de uma URL tipo postgresql://user:pass@host:port/db.
    """
    try:
        sp = urlsplit(db_url)
        if "@" not in (sp.netloc or ""):
            return db_url
        creds, host = sp.netloc.rsplit("@", 1)
        if ":" in creds:
            user, _ = creds.split(":", 1)
            netloc = f"{user}:***@{host}"
        else:
            netloc = f"{creds}@{host}"
        return urlunsplit((sp.scheme, netloc, sp.path, sp.query, sp.fragment))
    except Exception:
        return "<database_url_redacted>"


def classify_model(audit_version: str) -> str:
    v = (audit_version or "").strip()
    if v == "v4.0-api":
        return "API (2-4s)"
    if v in ("v1.0", "v1.0-recovered"):
        return "DOM (15-30s)"
    return f"Outro ({v})" if v else "Outro"


def diff_bucket(diff_pct: Optional[float]) -> Optional[str]:
    if diff_pct is None:
        return None
    if diff_pct < -10:
        return "BS < WS (-10% a -2%)"  # fora do confiável, mas mantemos label macro
    if diff_pct < -2:
        return "BS < WS (-10% a -2%)"
    if diff_pct <= 2:
        return "BS ~ WS (-2% a +2%)"
    if diff_pct <= 10:
        return "BS > WS (+2% a +10%)"
    return "BS > WS (+2% a +10%)"


def line_bucket(line_str: str) -> str:
    try:
        x = abs(float(str(line_str).replace(",", ".")))
    except Exception:
        return "Outro"
    if x <= 1:
        return "AH 0-1 (líquida)"
    if x <= 2:
        return "AH 1-2 (média)"
    return "AH 2+ (extrema)"


def lag_bucket(ms: Optional[int]) -> str:
    if not ms or ms <= 0:
        return "Desconhecido"
    if ms < 10000:
        return "< 10s"
    if ms < 20000:
        return "10-20s"
    if ms < 30000:
        return "20-30s"
    return "> 30s"


async def fetch_h3b_audit_rows(
    db: Database,
    direction: str,
    versions: List[str],
    lookback_days: Optional[int] = None,
) -> List[Tuple[Any, ...]]:
    """
    Traz a base principal:
    - betslip_audit_results (H3B, direction) + match (kickoff passado)
    - closing_odd por best_odds_history (último antes do kickoff, linha+lado)
    """
    q = (
        text(
        """
        SELECT
            a.id,
            a.event_id,
            a.home_team,
            a.away_team,
            a.league,
            a.market_type,
            a.line,
            a.side,
            a.websocket_odd,
            a.betslip_odd,
            a.difference_pct,
            a.betslip_limit,
            a.status,
            a.is_live,
            a.audit_total_duration_ms,
            a.lag_detection_to_click_ms,
            a.lag_click_to_betslip_ms,
            a.reversal_direction,
            a.market_period,
            a.audit_version,
            a.audited_at,
            a.hypothesis_details,
            m.id AS match_id,
            m.kickoff_time,
            m.home_score,
            m.away_score,
            m.status AS match_status,
            CASE
                WHEN a.side = 'home' THEN (
                    SELECT boh.best_home_odds
                    FROM best_odds_history boh
                    WHERE boh.match_id = m.id
                      AND (boh.ah_line = a.line OR boh.ah_line = a.line || '.0'
                           OR boh.ah_line = CASE
                                WHEN a.line NOT LIKE '+%' AND a.line NOT LIKE '-%' THEN '+' || a.line ELSE a.line END
                           OR boh.ah_line = CASE
                                WHEN a.line NOT LIKE '+%' AND a.line NOT LIKE '-%' THEN '+' || a.line || '.0' ELSE a.line || '.0' END
                      )
                      AND boh.scraped_at < m.kickoff_time
                      AND boh.best_home_odds > 0
                    ORDER BY boh.scraped_at DESC
                    LIMIT 1
                )
                ELSE (
                    SELECT boh.best_away_odds
                    FROM best_odds_history boh
                    WHERE boh.match_id = m.id
                      AND (boh.ah_line = a.line OR boh.ah_line = a.line || '.0'
                           OR boh.ah_line = CASE
                                WHEN a.line NOT LIKE '+%' AND a.line NOT LIKE '-%' THEN '+' || a.line ELSE a.line END
                           OR boh.ah_line = CASE
                                WHEN a.line NOT LIKE '+%' AND a.line NOT LIKE '-%' THEN '+' || a.line || '.0' ELSE a.line || '.0' END
                      )
                      AND boh.scraped_at < m.kickoff_time
                      AND boh.best_away_odds > 0
                    ORDER BY boh.scraped_at DESC
                    LIMIT 1
                )
            END AS closing_odd
        FROM betslip_audit_results a
        JOIN matches m ON m.external_id = a.event_id
        WHERE a.hypothesis_type = 'H3B'
          AND a.reversal_direction = :direction
          AND m.kickoff_time < NOW()
          AND a.audit_version = ANY(:versions)
          AND (
            :lookback_days IS NULL
            OR a.audited_at >= NOW() - make_interval(days => :lookback_days)
          )
        """
        )
        # Evita AmbiguousParameterError no asyncpg (tipa explicitamente).
        .bindparams(bindparam("lookback_days", type_=Integer))
    )
    async with db.async_session() as session:
        res = await session.execute(q, {"direction": direction, "versions": versions, "lookback_days": lookback_days})
        return list(res.fetchall())


def compute_roi_pct(line: str, side: str, ws_odd: Optional[float], bs_odd: Optional[float], hs: Any, aws: Any) -> Tuple[Optional[float], Optional[float]]:
    """
    ROI em % para stake=1 (compatível com analyze_h3b_comprehensive.py).
    """
    if hs is None or aws is None:
        return None, None

    try:
        goal_diff = int(hs) - int(aws)
    except Exception:
        return None, None

    try:
        ah_line = float(str(line).replace(",", "."))
    except Exception:
        return None, None

    if (side or "").strip() == "home":
        adjusted = goal_diff + ah_line
    else:
        adjusted = -goal_diff - ah_line

    # win/loss/push/half
    if adjusted > 0.25:
        mult = 1.0
    elif adjusted == 0.25:
        mult = 0.5
    elif adjusted == 0:
        mult = 0.0
    elif adjusted == -0.25:
        mult = -0.5
    else:
        mult = -1.0

    roi_ws = None
    roi_bs = None

    if mult > 0:
        if ws_odd and ws_odd > 0:
            roi_ws = (ws_odd - 1.0) * mult * 100.0
        if bs_odd and bs_odd > 0:
            roi_bs = (bs_odd - 1.0) * mult * 100.0
    elif mult < 0:
        roi_ws = mult * 100.0
        roi_bs = mult * 100.0
    else:
        roi_ws = 0.0
        roi_bs = 0.0

    return roi_ws, roi_bs


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", default="up", choices=["up", "down"], help="Direção H3B (default: up)")
    parser.add_argument(
        "--versions",
        default="v4.0-api,v1.0,v1.0-recovered",
        help="Lista de audit_version separada por vírgula (default: v4.0-api,v1.0,v1.0-recovered)",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Override do DATABASE_URL (útil se o .env não estiver sendo carregado ou para apontar outro host)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        help="Se definido, filtra auditorias por janela móvel (a.audited_at >= NOW() - N dias).",
    )
    parser.add_argument("--out", required=True, help="Caminho do markdown de saída (relativo a betinasia_bot/)")
    parser.add_argument(
        "--pdf",
        default=None,
        help="Se definido, renderiza o markdown para PDF (requer reportlab). Ex.: docs/relatorio.pdf",
    )
    parser.add_argument("--back-diff-min", type=float, default=2.0, help="Corte de edge Back (default: 2.0)")
    parser.add_argument("--lay-diff-max", type=float, default=-2.0, help="Corte de edge Lay (default: -2.0)")
    parser.add_argument("--stake-pct-of-limit", type=float, default=0.25, help="Stake fallback (% do limite), default 0.25")
    parser.add_argument("--stake-cap", type=float, default=0.0, help="Cap opcional para stake fallback (0=sem cap)")
    parser.add_argument("--seed", type=int, default=1337, help="Seed do bootstrap")
    args = parser.parse_args()

    # seed global para reprodutibilidade do bootstrap
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    versions = [v.strip() for v in str(args.versions).split(",") if v.strip()]
    out_path = Path(args.out)

    db = Database(database_url=args.database_url) if args.database_url else Database()
    try:
        await db.connect()
    except Exception as e:
        # Ajuda diagnóstico do lado do usuário (sem vazar senha)
        try:
            from config.settings import settings

            effective = args.database_url or settings.database_url
            print("\n[ERRO] Falha ao conectar no banco.")
            print(f"- DATABASE_URL efetivo: {_redact_db_url(str(effective))}")
            print(f"- Erro: {e}\n")
        except Exception:
            pass
        raise

    try:
        print(
            f"[INFO] Carregando dados H3B (direction={args.direction}, versions={versions}, lookback_days={args.lookback_days})...",
            flush=True,
        )
        t0 = time.perf_counter()
        rows = await fetch_h3b_audit_rows(
            db,
            direction=str(args.direction),
            versions=versions,
            lookback_days=args.lookback_days,
        )
        dt = time.perf_counter() - t0
        print(f"[INFO] Linhas carregadas: {len(rows)} (tempo: {dt:.1f}s)", flush=True)

        # dataset em dicts (compatível com análise original)
        all_data: List[Dict[str, Any]] = []
        for r in rows:
            d: Dict[str, Any] = {
                "id": r[0],
                "event_id": r[1],
                "home_team": r[2],
                "away_team": r[3],
                "league": r[4] or "",
                "market_type": r[5],
                "line": r[6],
                "side": r[7],
                "ws_odd": r[8],
                "bs_odd": r[9],
                "diff_pct": r[10],
                "limit": r[11] or 0.0,
                "status": r[12],
                "is_live": r[13],
                "lag_total": r[14] or 0,
                "lag_click": r[15] or 0,
                "lag_bs": r[16] or 0,
                "direction": r[17],
                "period": r[18],
                "version": r[19],
                "audited_at": r[20],
                "hypothesis_details": _as_dict(r[21]),
                "match_id": int(r[22]),
                "kickoff": r[23],
                "home_score": r[24],
                "away_score": r[25],
                "match_status": r[26],
                "closing_odd": r[27],
            }

            # CLV (bruto)
            if d["closing_odd"] and d["closing_odd"] > 0:
                d["clv_ws"] = (d["ws_odd"] - d["closing_odd"]) / d["closing_odd"] * 100.0 if d["ws_odd"] else None
                if d["bs_odd"] and d["bs_odd"] > 0:
                    d["clv_bs"] = (d["bs_odd"] - d["closing_odd"]) / d["closing_odd"] * 100.0
                else:
                    d["clv_bs"] = None
            else:
                d["clv_ws"] = None
                d["clv_bs"] = None

            # ROI
            roi_ws, roi_bs = compute_roi_pct(
                line=str(d["line"]),
                side=str(d["side"]),
                ws_odd=_safe_float(d["ws_odd"]),
                bs_odd=_safe_float(d["bs_odd"]),
                hs=d["home_score"],
                aws=d["away_score"],
            )
            d["roi_ws"] = roi_ws
            d["roi_bs"] = roi_bs

            # Buckets auxiliares
            d["model"] = classify_model(str(d.get("version", "")))
            d["diff_bucket"] = diff_bucket(_safe_float(d.get("diff_pct")))
            d["line_bucket"] = line_bucket(str(d.get("line", "")))
            d["lag_bucket"] = lag_bucket(int(d.get("lag_total") or 0))

            all_data.append(d)

        # Filtro qualidade betslip (igual ao script: -10 a +10)
        with_bs_raw = [d for d in all_data if d.get("bs_odd") and d["bs_odd"] > 0]
        with_bs = [d for d in with_bs_raw if d.get("diff_pct") is not None and -10 <= float(d["diff_pct"]) <= 10]

        unique_matches_all = len(set(d["match_id"] for d in all_data))
        unique_matches_bs = len(set(d["match_id"] for d in with_bs))
        avg_obs_per_match = (len(all_data) / unique_matches_all) if unique_matches_all else 0.0

        # Cobertura de resultados (diagnóstico de ROI)
        matches_with_scores = len(
            {d["match_id"] for d in all_data if d.get("home_score") is not None and d.get("away_score") is not None}
        )
        matches_finished_flag = len({d["match_id"] for d in all_data if str(d.get("match_status", "")).lower() == "finished"})

        # Coortes operacionais e coberturas (sem depender de ROI/closing)
        back_cut = float(args.back_diff_min)
        lay_cut = float(args.lay_diff_max)

        ok_bs = [d for d in with_bs if str(d.get("status", "")).upper() == "OK" and d.get("diff_pct") is not None]
        back_edge = [d for d in ok_bs if float(d["diff_pct"]) >= back_cut]
        lay_edge = [d for d in ok_bs if float(d["diff_pct"]) <= lay_cut]
        back_edge_ids = {int(d["id"]) for d in back_edge if d.get("id") is not None}
        lay_edge_ids = {int(d["id"]) for d in lay_edge if d.get("id") is not None}

        def _is_nonempty_array(x: Any) -> bool:
            return isinstance(x, list) and len(x) > 0

        n_temporal = sum(
            1
            for d in ok_bs
            if _is_nonempty_array(_get_path(d.get("hypothesis_details") or {}, ["temporal"]))
        )
        n_lay_temporal = sum(
            1
            for d in ok_bs
            if _is_nonempty_array(_get_path(d.get("hypothesis_details") or {}, ["lay_temporal"]))
        )
        n_finance = sum(
            1
            for d in ok_bs
            if isinstance(_get_path(d.get("hypothesis_details") or {}, ["finance"]), dict)
        )

        # CLV adicional (baseline por jogo: média das outras observações WS do jogo)
        by_match: Dict[int, List[Dict[str, Any]]] = {}
        for d in all_data:
            if d.get("clv_ws") is not None:
                by_match.setdefault(d["match_id"], []).append(d)
        for mid, entries in by_match.items():
            if len(entries) < 2:
                for e in entries:
                    e["clv_baseline"] = None
                    e["clv_ws_adicional"] = e.get("clv_ws")
                    e["clv_bs_adicional"] = e.get("clv_bs")
                continue
            for e in entries:
                others = [x["clv_ws"] for x in entries if x["id"] != e["id"] and x.get("clv_ws") is not None]
                if others:
                    e["clv_baseline"] = float(sum(others) / len(others))
                    e["clv_ws_adicional"] = float(e["clv_ws"] - e["clv_baseline"])
                    e["clv_bs_adicional"] = float(e["clv_bs"] - e["clv_baseline"]) if e.get("clv_bs") is not None else None
                else:
                    e["clv_baseline"] = None
                    e["clv_ws_adicional"] = e.get("clv_ws")
                    e["clv_bs_adicional"] = e.get("clv_bs")

        # Helpers para métricas por modelo
        def subset_model(model_name: str) -> List[Dict[str, Any]]:
            return [d for d in all_data if d.get("model") == model_name]

        def subset_bs_model(model_name: str) -> List[Dict[str, Any]]:
            return [d for d in with_bs if d.get("model") == model_name]

        models = ["API (2-4s)", "DOM (15-30s)"]

        # --- relatório markdown ---
        now_utc = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
        lines: List[str] = []
        lines.append("# Análise Estatística Robusta — Contexto Operação (b808)\n")
        lines.append(f"**Data da execução:** {now_utc}  \n")
        lines.append("**Escopo:** H3B (auditoria), com comparação por `audit_version` e inferência robusta por jogo (cluster bootstrap).  \n")
        lines.append("**Nota:** a robustez aqui significa que intervalos de confiança consideram correlação intra-jogo (múltiplas auditorias por partida).\n")
        lines.append("---\n")

        # 1) Contexto
        lines.append("## 1) Contexto do corte (b808)\n")
        lines.append("| Indicador | Valor |\n|---|---:|\n")
        lines.append(f"| Auditorias H3B `{args.direction.upper()}` (match + kickoff passado) | {len(all_data)} |\n")
        lines.append(f"| Betslip bruto | {len(with_bs_raw)} |\n")
        lines.append(f"| Betslip confiável (diff -10% a +10%) | {len(with_bs)} |\n")
        lines.append(f"| Descartados no filtro de qualidade | {len(with_bs_raw) - len(with_bs)} |\n")
        lines.append(f"| Jogos únicos (geral) | {unique_matches_all} |\n")
        lines.append(f"| Média de observações por jogo | {avg_obs_per_match:.1f} |\n")
        lines.append(f"| Jogos únicos com betslip confiável | {unique_matches_bs} |\n")
        lines.append("\n---\n")

        # 2) Base comparativa
        lines.append("## 2) Base comparativa: API vs DOM\n")
        lines.append("| Métrica | API (v4.0-api) | DOM (v1.0) |\n|---|---:|---:|\n")
        for m in models:
            pass
        # total obs
        api_all = subset_model("API (2-4s)")
        dom_all = subset_model("DOM (15-30s)")
        api_bs = subset_bs_model("API (2-4s)")
        dom_bs = subset_bs_model("DOM (15-30s)")
        api_lag = _mean([float(d["lag_total"]) for d in api_all if d.get("lag_total")])
        dom_lag = _mean([float(d["lag_total"]) for d in dom_all if d.get("lag_total")])
        api_clv_pm_n = len([d for d in api_bs if d.get("clv_bs") is not None and d.get("is_live") is False and -50 < float(d["clv_bs"]) < 50])
        dom_clv_pm_n = len([d for d in dom_bs if d.get("clv_bs") is not None and d.get("is_live") is False and -50 < float(d["clv_bs"]) < 50])
        api_roi_n = len([d for d in api_bs if d.get("roi_bs") is not None])
        dom_roi_n = len([d for d in dom_bs if d.get("roi_bs") is not None])
        lines.append(f"| Total de observações | {len(api_all)} | {len(dom_all)} |\n")
        lines.append(f"| Com betslip confiável | {len(api_bs)} | {len(dom_bs)} |\n")
        lines.append(f"| Com CLV pre-match (betslip) | {api_clv_pm_n} | {dom_clv_pm_n} |\n")
        lines.append(f"| Com ROI (betslip) | {api_roi_n} | {dom_roi_n} |\n")
        lines.append(f"| Lag médio observado (fim-a-fim) | {_fmt_num(api_lag, 0)} ms | {_fmt_num(dom_lag, 0)} ms |\n")
        lines.append("\n---\n")

        # 2.1) Pre vs in
        lines.append("### 2.1 Cobertura temporal (pre-match vs in-match)\n")
        pm = [d for d in all_data if d.get("is_live") is False]
        im = [d for d in all_data if d.get("is_live") is True]
        lines.append("| Métrica | Pre-match | In-match | Observação |\n|---|---:|---:|---|\n")
        lines.append(f"| Observações totais com classificação temporal | {len(pm)} | {len(im)} | Contagem bruta do corte |\n")
        lines.append(
            f"| ROI Betslip | {len([d for d in with_bs if d.get('is_live') is False and d.get('roi_bs') is not None])} | {len([d for d in with_bs if d.get('is_live') is True and d.get('roi_bs') is not None])} | Amostra com resultado do jogo |\n"
        )
        lines.append(
            f"| ROI WebSocket | {len([d for d in all_data if d.get('is_live') is False and d.get('roi_ws') is not None])} | {len([d for d in all_data if d.get('is_live') is True and d.get('roi_ws') is not None])} | Referência de mercado |\n"
        )
        lines.append(
            f"| CLV Betslip (informativo) | {len([d for d in with_bs if d.get('is_live') is False and d.get('clv_bs') is not None and -50 < float(d['clv_bs']) < 50])} | {len([d for d in with_bs if d.get('is_live') is True and d.get('clv_bs') is not None and -50 < float(d['clv_bs']) < 50])} | Decisão prioriza CLV pre-match |\n"
        )
        lines.append("\n---\n")

        # 2.2) Performance por regime (pre vs in)
        lines.append("### 2.2 Performance por regime (pre-match vs in-match)\n")
        lines.append("| Regime | N auditorias | N betslip conf. | N OK | Back edge | Lay edge | Diff médio (OK) |\n|---|---:|---:|---:|---:|---:|---:|\n")
        for label, is_live_val in [("PRE_MATCH", False), ("IN_MATCH", True)]:
            sub_all = [d for d in all_data if d.get("is_live") is is_live_val]
            sub_bs = [d for d in with_bs if d.get("is_live") is is_live_val]
            sub_ok = [d for d in ok_bs if d.get("is_live") is is_live_val]
            sub_back = [d for d in back_edge if d.get("is_live") is is_live_val]
            sub_lay = [d for d in lay_edge if d.get("is_live") is is_live_val]
            diff_mean = _mean([float(d["diff_pct"]) for d in sub_ok]) if sub_ok else None
            lines.append(
                f"| {label} | {len(sub_all)} | {len(sub_bs)} | {len(sub_ok)} | {len(sub_back)} | {len(sub_lay)} | {_fmt_pct(diff_mean)} |\n"
            )
        lines.append("\n---\n")

        # 3) CLV pre-match (robusto)
        lines.append("## 3) CLV pre-match (núcleo)\n")
        lines.append("### 3.1 CLV com odd do Betslip (execução real)\n")
        lines.append(
            "| Métrica | API | DOM |\n|---|---:|---:|\n"
        )

        def model_metric(model_name: str, key: str, prematch_only: bool = True) -> MetricSummary:
            subset = subset_bs_model(model_name)  # apenas betslip confiável
            if prematch_only:
                subset = [d for d in subset if d.get("is_live") is False]
            vals = [d.get(key) for d in subset]
            mids = [d.get("match_id") for d in subset]
            return summarize_metric(vals, mids, clip_low=-50, clip_high=50)

        api_clv = model_metric("API (2-4s)", "clv_bs", prematch_only=True)
        dom_clv = model_metric("DOM (15-30s)", "clv_bs", prematch_only=True)
        api_clv_ad = model_metric("API (2-4s)", "clv_bs_adicional", prematch_only=True)
        dom_clv_ad = model_metric("DOM (15-30s)", "clv_bs_adicional", prematch_only=True)

        lines.append(
            f"| CLV Bruto BS Pre-Match | {_fmt_pct(api_clv.mean_event)} ({_sig_label(api_clv.ci90_cluster)}, N={api_clv.n_events}, jogos={api_clv.n_matches}) | {_fmt_pct(dom_clv.mean_event)} ({_sig_label(dom_clv.ci90_cluster)}, N={dom_clv.n_events}, jogos={dom_clv.n_matches}) |\n"
        )
        lines.append(
            f"| CLV Adicional BS Pre-Match | {_fmt_pct(api_clv_ad.mean_event)} ({_sig_label(api_clv_ad.ci90_cluster)}, N={api_clv_ad.n_events}, jogos={api_clv_ad.n_matches}) | {_fmt_pct(dom_clv_ad.mean_event)} ({_sig_label(dom_clv_ad.ci90_cluster)}, N={dom_clv_ad.n_events}, jogos={dom_clv_ad.n_matches}) |\n"
        )
        lines.append(
            f"| Taxa de CLV > 0 (bruto) | {_fmt_num(api_clv.hit_rate_event, 1)}% | {_fmt_num(dom_clv.hit_rate_event, 1)}% |\n"
        )
        lines.append(
            f"| Taxa de CLV > 0 (adicional) | {_fmt_num(api_clv_ad.hit_rate_event, 1)}% | {_fmt_num(dom_clv_ad.hit_rate_event, 1)}% |\n"
        )
        lines.append("\nNotas de robustez (IC 90% por jogo):  \n")
        lines.append(f"- API CLV bruto (cluster): média {_fmt_pct(api_clv.mean_cluster)}; IC90 {_fmt_ci(api_clv.ci90_cluster)}  \n")
        lines.append(f"- DOM CLV bruto (cluster): média {_fmt_pct(dom_clv.mean_cluster)}; IC90 {_fmt_ci(dom_clv.ci90_cluster)}  \n")
        lines.append("\n---\n")

        # 4) ROI por modelo
        lines.append("## 4) ROI por modelo\n")
        lines.append("| Métrica | API | DOM |\n|---|---:|---:|\n")

        api_roi = model_metric("API (2-4s)", "roi_bs", prematch_only=False)
        dom_roi = model_metric("DOM (15-30s)", "roi_bs", prematch_only=False)
        api_roi_ws = summarize_metric(
            [d.get("roi_ws") for d in api_all],
            [d.get("match_id") for d in api_all],
            clip_low=-100,
            clip_high=500,  # ROI pode ser alto; limitamos só para evitar infinito
        )
        dom_roi_ws = summarize_metric(
            [d.get("roi_ws") for d in dom_all],
            [d.get("match_id") for d in dom_all],
            clip_low=-100,
            clip_high=500,
        )

        lines.append(
            f"| ROI Betslip | {_fmt_pct(api_roi.mean_event)} ({_sig_label(api_roi.ci90_cluster)}, N={api_roi.n_events}) | {_fmt_pct(dom_roi.mean_event)} ({_sig_label(dom_roi.ci90_cluster)}, N={dom_roi.n_events}) |\n"
        )
        lines.append(
            f"| ROI WebSocket | {_fmt_pct(api_roi_ws.mean_event)} ({_sig_label(api_roi_ws.ci90_cluster)}, N={api_roi_ws.n_events}) | {_fmt_pct(dom_roi_ws.mean_event)} ({_sig_label(dom_roi_ws.ci90_cluster)}, N={dom_roi_ws.n_events}) |\n"
        )
        lines.append(
            f"| Win rate ROI Betslip | {_fmt_num(api_roi.hit_rate_event, 1)}% | {_fmt_num(dom_roi.hit_rate_event, 1)}% |\n"
        )
        lines.append(
            f"| Win rate ROI WS | {_fmt_num(api_roi_ws.hit_rate_event, 1)}% | {_fmt_num(dom_roi_ws.hit_rate_event, 1)}% |\n"
        )
        lines.append("\n---\n")

        # 5) Diferença de preço BS vs WS
        lines.append("## 5) Diferença de preço BS vs WS\n")
        lines.append("| Métrica | API | DOM |\n|---|---:|---:|\n")

        def model_diff(model_name: str) -> MetricSummary:
            subset = subset_bs_model(model_name)
            vals = [d.get("diff_pct") for d in subset]
            mids = [d.get("match_id") for d in subset]
            return summarize_metric(vals, mids, clip_low=-50, clip_high=50)

        api_diff = model_diff("API (2-4s)")
        dom_diff = model_diff("DOM (15-30s)")
        api_bs_gt_ws = len([d for d in api_bs if d.get("diff_pct") is not None and float(d["diff_pct"]) > 0])
        dom_bs_gt_ws = len([d for d in dom_bs if d.get("diff_pct") is not None and float(d["diff_pct"]) > 0])
        api_bs_gt_ws_2 = len([d for d in api_bs if d.get("diff_pct") is not None and float(d["diff_pct"]) > 2])
        dom_bs_gt_ws_2 = len([d for d in dom_bs if d.get("diff_pct") is not None and float(d["diff_pct"]) > 2])

        lines.append(
            f"| Diff BS vs WS (média) | {_fmt_pct(api_diff.mean_event)} ({_sig_label(api_diff.ci90_cluster)}, N={api_diff.n_events}) | {_fmt_pct(dom_diff.mean_event)} ({_sig_label(dom_diff.ci90_cluster)}, N={dom_diff.n_events}) |\n"
        )
        lines.append(f"| BS > WS | {_fmt_num(api_bs_gt_ws / len(api_bs) * 100 if api_bs else None, 1)}% ({api_bs_gt_ws}/{len(api_bs)}) | {_fmt_num(dom_bs_gt_ws / len(dom_bs) * 100 if dom_bs else None, 1)}% ({dom_bs_gt_ws}/{len(dom_bs)}) |\n")
        lines.append(f"| BS > WS +2% | {_fmt_num(api_bs_gt_ws_2 / len(api_bs) * 100 if api_bs else None, 1)}% ({api_bs_gt_ws_2}/{len(api_bs)}) | {_fmt_num(dom_bs_gt_ws_2 / len(dom_bs) * 100 if dom_bs else None, 1)}% ({dom_bs_gt_ws_2}/{len(dom_bs)}) |\n")
        lines.append("\n---\n")

        # 6) Combinações (buckets / linha / lag)
        lines.append("## 6) Combinações de valor\n")

        # 6.1 buckets
        bucket_clv_pm: Dict[str, MetricSummary] = {}
        lines.append("### 6.1 Buckets por diferença BS vs WS\n")
        lines.append("| Bucket | N bucket | CLV BS PM (média) | IC90 (cluster) | N CLV PM | Jogos CLV PM | ROI BS (todos) | IC90 (cluster) |\n|---|---:|---:|---|---:|---:|---:|---|\n")
        for bucket in ["BS < WS (-10% a -2%)", "BS ~ WS (-2% a +2%)", "BS > WS (+2% a +10%)"]:
            subset = [d for d in with_bs if d.get("diff_bucket") == bucket]
            clv_pm = summarize_metric(
                [d.get("clv_bs") for d in subset if d.get("is_live") is False],
                [d.get("match_id") for d in subset if d.get("is_live") is False],
                clip_low=-50,
                clip_high=50,
            )
            bucket_clv_pm[bucket] = clv_pm
            roi_all = summarize_metric(
                [d.get("roi_bs") for d in subset],
                [d.get("match_id") for d in subset],
                clip_low=-100,
                clip_high=500,
            )
            lines.append(
                f"| {bucket} | {len(subset)} | {_fmt_pct(clv_pm.mean_event)} | {_fmt_ci(clv_pm.ci90_cluster)} | {clv_pm.n_events} | {clv_pm.n_matches} | {_fmt_pct(roi_all.mean_event)} | {_fmt_ci(roi_all.ci90_cluster)} |\n"
            )
        lines.append("\n---\n")

        # 6.2 linha AH
        lines.append("### 6.2 Combinação por faixa de linha AH\n")
        lines.append("| Faixa AH | CLV BS PM (média) | IC90 (cluster) | ROI BS (todos) | IC90 (cluster) | Diff BS vs WS (média) |\n|---|---:|---|---:|---|---:|\n")
        for lb in ["AH 0-1 (líquida)", "AH 1-2 (média)", "AH 2+ (extrema)"]:
            subset = [d for d in with_bs if d.get("line_bucket") == lb]
            clv_pm = summarize_metric(
                [d.get("clv_bs") for d in subset if d.get("is_live") is False],
                [d.get("match_id") for d in subset if d.get("is_live") is False],
                clip_low=-50,
                clip_high=50,
            )
            roi_all = summarize_metric(
                [d.get("roi_bs") for d in subset],
                [d.get("match_id") for d in subset],
                clip_low=-100,
                clip_high=500,
            )
            diff_all = summarize_metric(
                [d.get("diff_pct") for d in subset],
                [d.get("match_id") for d in subset],
                clip_low=-50,
                clip_high=50,
            )
            lines.append(
                f"| {lb} | {_fmt_pct(clv_pm.mean_event)} | {_fmt_ci(clv_pm.ci90_cluster)} | {_fmt_pct(roi_all.mean_event)} | {_fmt_ci(roi_all.ci90_cluster)} | {_fmt_pct(diff_all.mean_event)} |\n"
            )
        lines.append("\n---\n")

        # 6.3 lag
        lag_clv_pm: Dict[str, MetricSummary] = {}
        lines.append("### 6.3 Combinação por faixa de lag\n")
        lines.append("| Faixa de lag | CLV BS PM (média) | IC90 (cluster) | N CLV PM | Jogos CLV PM | ROI BS (todos) | IC90 (cluster) | Diff BS vs WS (média) |\n|---|---:|---|---:|---:|---:|---|---:|\n")
        for lag in ["< 10s", "10-20s", "20-30s", "> 30s"]:
            subset = [d for d in with_bs if d.get("lag_bucket") == lag]
            clv_pm = summarize_metric(
                [d.get("clv_bs") for d in subset if d.get("is_live") is False],
                [d.get("match_id") for d in subset if d.get("is_live") is False],
                clip_low=-50,
                clip_high=50,
            )
            lag_clv_pm[lag] = clv_pm
            roi_all = summarize_metric(
                [d.get("roi_bs") for d in subset],
                [d.get("match_id") for d in subset],
                clip_low=-100,
                clip_high=500,
            )
            diff_all = summarize_metric(
                [d.get("diff_pct") for d in subset],
                [d.get("match_id") for d in subset],
                clip_low=-50,
                clip_high=50,
            )
            lines.append(
                f"| {lag} | {_fmt_pct(clv_pm.mean_event)} | {_fmt_ci(clv_pm.ci90_cluster)} | {clv_pm.n_events} | {clv_pm.n_matches} | {_fmt_pct(roi_all.mean_event)} | {_fmt_ci(roi_all.ci90_cluster)} | {_fmt_pct(diff_all.mean_event)} |\n"
            )
        lines.append("\n---\n")

        # ============================================================
        # Insere sumário executivo no topo (após header)
        # ============================================================
        summary_lines: List[str] = []
        summary_lines.append("## 0) Sumário executivo (leitura rápida)\n")
        summary_lines.append(f"- **Recorte**: direction=`{args.direction}`, lookback_days=`{args.lookback_days}`, versions=`{','.join(versions)}`.\n")
        summary_lines.append(
            f"- **Amostra**: {len(all_data)} auditorias (jogos únicos={unique_matches_all}, média={avg_obs_per_match:.1f} obs/jogo); betslip confiável={len(with_bs)}.\n"
        )
        summary_lines.append(
            f"- **Coortes (status=OK, betslip confiável)**: Back (diff>={back_cut:.1f}%): **{len(back_edge)}**; Lay (diff<={lay_cut:.1f}%): **{len(lay_edge)}**.\n"
        )
        summary_lines.append(
            f"- **Coberturas em `hypothesis_details` (OK)**: temporal(back)={n_temporal}/{len(ok_bs)}; lay_temporal={n_lay_temporal}/{len(ok_bs)}; finance={n_finance}/{len(ok_bs)}.\n"
        )
        summary_lines.append(
            f"- **Cobertura de placar (ROI)**: jogos com placar={matches_with_scores}/{unique_matches_all} (status finished={matches_finished_flag}).\n"
        )
        if len(dom_all) == 0:
            summary_lines.append("- **DOM**: sem dados no recorte atual (N=0), então não há comparação API vs DOM aqui.\n")
        # CLV foco
        summary_lines.append(
            f"- **CLV pre-match (Betslip, API)**: média robusta por jogo {_fmt_pct(api_clv.mean_cluster)} (IC90 {_fmt_ci(api_clv.ci90_cluster)}), "
            f"com N={api_clv.n_events} eventos (jogos={api_clv.n_matches}).\n"
        )
        # Buckets
        b_neg = bucket_clv_pm.get("BS < WS (-10% a -2%)")
        b_neu = bucket_clv_pm.get("BS ~ WS (-2% a +2%)")
        b_pos = bucket_clv_pm.get("BS > WS (+2% a +10%)")
        if b_neg and b_neu and b_pos:
            summary_lines.append(
                "- **Padrão por bucket (CLV PM)**: "
                f"`BS < WS` {_fmt_pct(b_neg.mean_event)} ({_sig_label(b_neg.ci90_cluster)}), "
                f"`BS ~ WS` {_fmt_pct(b_neu.mean_event)} ({_sig_label(b_neu.ci90_cluster)}), "
                f"`BS > WS` {_fmt_pct(b_pos.mean_event)} ({_sig_label(b_pos.ci90_cluster)}).\n"
            )
        # ROI coverage
        if api_roi.n_events == 0 and dom_roi.n_events == 0:
            summary_lines.append(
                "- **ROI**: sem cobertura no recorte (N=0). Isso normalmente acontece quando os placares ainda não foram sincronizados para esses jogos.\n"
            )
        summary_lines.append(
            "- **Leitura recomendada**: use este relatório para validar **qualidade de execução/CLV**. Para concluir sobre **ROI**, rode após atualizar resultados "
            "e/ou use uma janela com jogos já liquidados.\n"
        )
        summary_lines.append("\n---\n")

        # Header atual tem 5 linhas: título, data, escopo, nota, '---'
        # Inserimos o sumário logo depois disso.
        lines[5:5] = summary_lines

        # ============================================================
        # 7) Estimativa financeira (proxy) + risco
        # ============================================================
        lines.append("## 7) Estimativa financeira (proxy) e risco\n")
        lines.append("Este bloco usa `hypothesis_details.finance` quando existe; se não existir, usa stake fallback = `stake_pct_of_limit × limit`.\n\n")

        stake_pct = max(0.0, float(args.stake_pct_of_limit))
        stake_cap = max(0.0, float(args.stake_cap))

        def stake_from_limit(limit_value: float) -> float:
            s = max(0.0, limit_value) * stake_pct
            if stake_cap > 0:
                s = min(s, stake_cap)
            return s

        def finance_for_row(d: dict) -> Tuple[float, float, float, float]:
            """
            Retorna (back_stake, back_profit_if_win, lay_stake, lay_liability).
            """
            h = d.get("hypothesis_details") or {}
            fin = h.get("finance") if isinstance(h, dict) else None
            bs = bp = ls = ll = None
            if isinstance(fin, dict):
                bs = _safe_float(_get_path(fin, ["back", "suggested_stake"]))
                bp = _safe_float(_get_path(fin, ["back", "profit_if_win"]))
                ls = _safe_float(_get_path(fin, ["lay", "suggested_stake"]))
                ll = _safe_float(_get_path(fin, ["lay", "liability_if_lose"]))

            if bs is None:
                bs = stake_from_limit(float(d.get("limit") or 0.0))
            if bp is None:
                odd = _safe_float(d.get("bs_odd")) or 0.0
                bp = bs * max(0.0, odd - 1.0) if (bs and odd > 1.0) else 0.0

            if ls is None:
                # tenta usar lay.available_limit
                lay_lim = _safe_float(_get_path(h, ["lay", "available_limit"]))
                ls = stake_from_limit(float(lay_lim if lay_lim is not None else (d.get("limit") or 0.0)))
            lay_odd = _safe_float(_get_path(h, ["lay", "odd"])) or 0.0
            if lay_odd <= 0:
                lay_odd = _safe_float(d.get("bs_odd")) or 0.0  # proxy
            if ll is None:
                ll = ls * max(0.0, lay_odd - 1.0) if (ls and lay_odd > 1.0) else 0.0

            return float(bs), float(bp), float(ls), float(ll)

        back_stakes = []
        back_profit_if_win = []
        lay_stakes = []
        lay_liability = []
        for d in ok_bs:
            bs, bp, ls, ll = finance_for_row(d)
            if int(d["id"]) in back_edge_ids:
                back_stakes.append(bs)
                back_profit_if_win.append(bp)
            if int(d["id"]) in lay_edge_ids:
                lay_stakes.append(ls)
                lay_liability.append(ll)

        lines.append("### 7.1 Back (BS >> WS)\n")
        lines.append("| Métrica | Valor |\n|---|---:|\n")
        lines.append(f"| Corte (diff_pct) | >= {back_cut:.1f}% |\n")
        lines.append(f"| N eventos | {len(back_edge)} |\n")
        lines.append(f"| Stake total (estimado) | {_fmt_num(sum(back_stakes) if back_stakes else None, 2)} |\n")
        lines.append(f"| Stake médio | {_fmt_num(_mean(back_stakes), 2)} |\n")
        lines.append(f"| Profit_if_win total (estimado) | {_fmt_num(sum(back_profit_if_win) if back_profit_if_win else None, 2)} |\n")
        lines.append(f"| Profit_if_win médio | {_fmt_num(_mean(back_profit_if_win), 2)} |\n")

        lines.append("\n### 7.2 Lay (BS << WS) — risco de cauda\n")
        lines.append("| Métrica | Valor |\n|---|---:|\n")
        lines.append(f"| Corte (diff_pct) | <= {lay_cut:.1f}% |\n")
        lines.append(f"| N eventos | {len(lay_edge)} |\n")
        lines.append(f"| Stake total (estimado) | {_fmt_num(sum(lay_stakes) if lay_stakes else None, 2)} |\n")
        lines.append(f"| Liability total (estimada) | {_fmt_num(sum(lay_liability) if lay_liability else None, 2)} |\n")
        lines.append(f"| Liability média | {_fmt_num(_mean(lay_liability), 2)} |\n")
        lines.append(f"| Liability p95 | {_fmt_num(_pctl(lay_liability, 95), 2)} |\n")
        lines.append(f"| Liability p99 | {_fmt_num(_pctl(lay_liability, 99), 2)} |\n")
        lines.append(f"| ES95 (liability) | {_fmt_num(_es_tail(lay_liability, 95), 2)} |\n")
        lines.append(f"| Liability max | {_fmt_num(max(lay_liability) if lay_liability else None, 2)} |\n")
        lines.append("\n---\n")

        # ============================================================
        # 8) Evolução temporal (T+0 -> último ponto)
        # ============================================================
        def _last_point(arr: Any) -> Tuple[Optional[float], Optional[float], int]:
            """
            Retorna (diff_last, t_last_s, n_pts) a partir de um array de pontos.
            """
            if not isinstance(arr, list) or not arr:
                return None, None, 0
            n_pts = len(arr)
            diff_last = None
            t_last = None
            for e in reversed(arr):
                if isinstance(e, dict):
                    if diff_last is None:
                        diff_last = _safe_float(e.get("diff_pct"))
                    if t_last is None:
                        t_last = _safe_float(e.get("t"))
                if diff_last is not None and t_last is not None:
                    break
            return diff_last, t_last, n_pts

        def _agg_temporal(rows_in: List[dict], mode: str) -> Dict[str, dict]:
            """
            mode='back' usa difference_pct como t0 e h.temporal como série.
            mode='lay'  usa lay odd do h.lay como t0 e h.lay_temporal como série.
            """
            out: Dict[str, dict] = {}
            for regime, is_live_val in [("PRE_MATCH", False), ("IN_MATCH", True)]:
                sub = [d for d in rows_in if d.get("is_live") is is_live_val]
                t0s = []
                lasts = []
                deltas = []
                npts = []
                tlasts = []
                for d in sub:
                    h = d.get("hypothesis_details") or {}
                    if mode == "back":
                        arr = h.get("temporal")
                        diff_t0 = _safe_float(d.get("diff_pct"))
                    else:
                        arr = h.get("lay_temporal")
                        ws_odd = _safe_float(d.get("ws_odd"))
                        lay_odd = _safe_float(_get_path(h, ["lay", "odd"]))
                        diff_t0 = ((lay_odd - ws_odd) / ws_odd * 100.0) if (ws_odd and lay_odd) else None
                    diff_last, t_last_s, n = _last_point(arr)
                    if diff_t0 is None or diff_last is None:
                        continue
                    t0s.append(float(diff_t0))
                    lasts.append(float(diff_last))
                    deltas.append(float(diff_last - diff_t0))
                    npts.append(n)
                    if t_last_s is not None:
                        tlasts.append(float(t_last_s))
                if not deltas:
                    out[regime] = {"n": 0}
                    continue
                m = float(np.mean(deltas))
                sd = float(np.std(deltas, ddof=1)) if len(deltas) >= 2 else None
                se = (sd / math.sqrt(len(deltas))) if (sd is not None and len(deltas) >= 2) else None
                ci95 = (m - 1.96 * se, m + 1.96 * se) if se is not None else None

                # retenção/perda/ganho de edge
                if mode == "back":
                    retain = sum(1 for a, b in zip(t0s, lasts) if a >= back_cut and b >= back_cut) / len(deltas)
                    loss = sum(1 for a, b in zip(t0s, lasts) if a >= back_cut and b < back_cut) / len(deltas)
                    gain = None
                else:
                    retain = sum(1 for a, b in zip(t0s, lasts) if a <= lay_cut and b <= lay_cut) / len(deltas)
                    loss = sum(1 for a, b in zip(t0s, lasts) if a <= lay_cut and b > lay_cut) / len(deltas)
                    gain = sum(1 for a, b in zip(t0s, lasts) if a > lay_cut and b <= lay_cut) / len(deltas)

                out[regime] = {
                    "n": len(deltas),
                    "diff_t0_avg": float(np.mean(t0s)),
                    "diff_tlast_avg": float(np.mean(lasts)),
                    "delta_avg": m,
                    "delta_ci95": ci95,
                    "retain": retain,
                    "loss": loss,
                    "gain": gain,
                    "avg_points": float(np.mean(npts)) if npts else None,
                    "t_last_avg_s": float(np.mean(tlasts)) if tlasts else None,
                }
            return out

        lines.append("## 8) Evolução temporal (T+0 -> último ponto)\n")
        lines.append("Este bloco mede se o edge (diff_pct) **retém** ou **se perde** após alguns segundos/minutos (quando há `hypothesis_details.temporal`/`lay_temporal`).\n\n")

        back_temp = _agg_temporal(ok_bs, mode="back")
        lay_temp = _agg_temporal(ok_bs, mode="lay")

        lines.append("### 8.1 Back temporal\n")
        lines.append("| Regime | N | diff_t0_avg % | diff_tlast_avg % | delta_avg % | IC95 delta | retenção edge % | perda edge % | pts médios | t_last médio (s) |\n|---|---:|---:|---:|---:|---|---:|---:|---:|---:|\n")
        for regime in ["PRE_MATCH", "IN_MATCH"]:
            s = back_temp.get(regime, {"n": 0})
            if s.get("n", 0) <= 0:
                lines.append(f"| {regime} | 0 | — | — | — | — | — | — | — | — |\n")
                continue
            ci = s.get("delta_ci95")
            lines.append(
                f"| {regime} | {s['n']} | {_fmt_pct(s['diff_t0_avg'],2)} | {_fmt_pct(s['diff_tlast_avg'],2)} | {_fmt_pct(s['delta_avg'],2)} | {_fmt_ci(ci,2)} | {s['retain']*100:.1f} | {s['loss']*100:.1f} | {_fmt_num(s.get('avg_points'),2)} | {_fmt_num(s.get('t_last_avg_s'),1)} |\n"
            )
        lines.append("\n### 8.2 Lay temporal\n")
        lines.append("| Regime | N | diff_t0_avg % | diff_tlast_avg % | delta_avg % | IC95 delta | retenção edge % | perda edge % | ganho edge % | pts médios | t_last médio (s) |\n|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|\n")
        for regime in ["PRE_MATCH", "IN_MATCH"]:
            s = lay_temp.get(regime, {"n": 0})
            if s.get("n", 0) <= 0:
                lines.append(f"| {regime} | 0 | — | — | — | — | — | — | — | — | — |\n")
                continue
            ci = s.get("delta_ci95")
            gain = (s.get("gain") * 100.0) if s.get("gain") is not None else None
            lines.append(
                f"| {regime} | {s['n']} | {_fmt_pct(s['diff_t0_avg'],2)} | {_fmt_pct(s['diff_tlast_avg'],2)} | {_fmt_pct(s['delta_avg'],2)} | {_fmt_ci(ci,2)} | {s['retain']*100:.1f} | {s['loss']*100:.1f} | {_fmt_num(gain,1)} | {_fmt_num(s.get('avg_points'),2)} | {_fmt_num(s.get('t_last_avg_s'),1)} |\n"
            )
        lines.append("\n---\n")

        # ============================================================
        # 9) Combinações de valor (regime x linha x lag)
        # ============================================================
        lines.append("## 9) Combinações de valor (regime × linha AH × lag)\n")
        lines.append("Tabela rankeada por volume (N). Métrica: diff_pct (OK, betslip confiável).\n\n")

        def top_combos(rows_in: List[dict], max_rows: int = 12) -> List[Tuple[Tuple[str, str, str], MetricSummary]]:
            groups: Dict[Tuple[str, str, str], Tuple[List[float], List[int]]] = {}
            for d in rows_in:
                regime = "IN_MATCH" if d.get("is_live") is True else "PRE_MATCH"
                key = (regime, str(d.get("line_bucket")), str(d.get("lag_bucket")))
                groups.setdefault(key, ([], []))
                groups[key][0].append(float(d["diff_pct"]))
                groups[key][1].append(int(d["match_id"]))
            summaries = []
            for k, (vals, mids) in groups.items():
                s = summarize_metric(vals, mids, clip_low=-50, clip_high=50)
                summaries.append((k, s))
            summaries.sort(key=lambda x: x[1].n_events, reverse=True)
            return summaries[:max_rows]

        lines.append("### 9.1 Back combos (diff_pct >= corte)\n")
        back_combo_rows = [d for d in back_edge if d.get("diff_pct") is not None]
        lines.append("| Regime | Linha | Lag | N | Jogos | Mean diff % | IC90 cluster |\n|---|---|---|---:|---:|---:|---|\n")
        for (reg, lb, lag), s in top_combos(back_combo_rows, 12):
            lines.append(f"| {reg} | {lb} | {lag} | {s.n_events} | {s.n_matches} | {_fmt_pct(s.mean_event,2)} | {_fmt_ci(s.ci90_cluster,2)} |\n")

        lines.append("\n### 9.2 Lay combos (diff_pct <= corte) + risco\n")
        lay_combo_rows = [d for d in lay_edge if d.get("diff_pct") is not None]
        lines.append("| Regime | Linha | Lag | N | Jogos | Mean diff % | IC90 cluster | Liability p95 |\n|---|---|---|---:|---:|---:|---|---:|\n")
        # liability por grupo (usando finance_for_row)
        lay_liab_by_group: Dict[Tuple[str, str, str], List[float]] = {}
        for d in lay_combo_rows:
            reg = "IN_MATCH" if d.get("is_live") is True else "PRE_MATCH"
            key = (reg, str(d.get("line_bucket")), str(d.get("lag_bucket")))
            _, _, _, ll = finance_for_row(d)
            lay_liab_by_group.setdefault(key, []).append(float(ll))
        for (reg, lb, lag), s in top_combos(lay_combo_rows, 12):
            p95 = _pctl(lay_liab_by_group.get((reg, lb, lag), []), 95)
            lines.append(f"| {reg} | {lb} | {lag} | {s.n_events} | {s.n_matches} | {_fmt_pct(s.mean_event,2)} | {_fmt_ci(s.ci90_cluster,2)} | {_fmt_num(p95,2)} |\n")
        lines.append("\n---\n")

        # ============================================================
        # 10) Diagnóstico de ROI / atualização de resultados
        # ============================================================
        lines.append("## 10) Diagnóstico: por que o ROI pode estar zerado\n")
        lines.append("ROI aqui é calculado por placar do jogo (`matches.home_score/away_score`). Se os placares não estiverem preenchidos no banco, a cobertura de ROI será 0.\n\n")
        lines.append("| Indicador | Valor |\n|---|---:|\n")
        lines.append(f"| Jogos únicos no recorte | {unique_matches_all} |\n")
        lines.append(f"| Jogos com placar disponível (home_score/away_score não nulos) | {matches_with_scores} |\n")
        lines.append(f"| Jogos com status='finished' no banco | {matches_finished_flag} |\n")
        lines.append("\nSe `placar disponível` estiver 0, isso geralmente indica que o job de resultados não rodou (ou está sem credenciais/chave da API).  \n")
        lines.append("Sugestão (rodar no servidor): `cd betinasia_bot && python3 -m results.auto_update_results --once` (ou configure o serviço para rodar em loop).\n")
        lines.append("\n---\n")

        # ============================================================
        # 11) Conclusões, riscos e pontos em aberto
        # ============================================================
        lines.append("## 11) Conclusões, riscos e pontos em aberto\n")
        lines.append("- **Execução (CLV)**: use as seções 3/6 para validar qualidade de execução (especialmente pre-match).\n")
        lines.append("- **Lay**: não pode ser decidido por média. Use p95/p99/ES95 de liability (seção 7.2) e combos com risco (seção 9.2).\n")
        lines.append("- **Temporal**: se a cobertura `temporal/lay_temporal` for baixa, a inferência de retenção de edge fica limitada (seção 8).\n")
        lines.append("- **ROI/resultado**: se placares não estão no banco, qualquer conclusão de lucro realizado fica em aberto (seção 10).\n")
        lines.append("\n---\n")

        # ============================================================
        # 12) Como reproduzir
        # ============================================================
        lines.append("## 12) Como reproduzir\n")
        lines.append("1. Configure `betinasia_bot/.env` com `DATABASE_URL`.  \n")
        lines.append("2. (Opcional) Atualize resultados para ter ROI: `cd betinasia_bot && python3 -m results.auto_update_results --once`.  \n")
        lines.append("3. Execute:\n\n")
        lines.append("```bash\n")
        lines.append("python3 betinasia_bot/analyze_contexto_operacao_b808_robust_report.py \\\n")
        lines.append("  --direction up \\\n")
        lines.append("  --versions v4.0-api,v1.0,v1.0-recovered \\\n")
        lines.append("  --lookback-days 14 \\\n")
        lines.append("  --out betinasia_bot/docs/analise_contexto_operacao_b808_robusta.md \\\n")
        lines.append("  --pdf betinasia_bot/docs/analise_contexto_operacao_b808_robusta.pdf\n")
        lines.append("```\n")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("".join(lines), encoding="utf-8")

        print(f"Relatório gerado em: {out_path}")

        if args.pdf:
            pdf_path = Path(str(args.pdf))
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            renderer = Path(__file__).resolve().parent / "docs" / "render_markdown_to_pdf.py"
            cmd = [sys.executable, str(renderer), str(out_path), str(pdf_path)]
            try:
                subprocess.run(cmd, check=True)
                print(f"PDF gerado em: {pdf_path}")
            except FileNotFoundError:
                print(f"[WARN] Não achei o renderizador de PDF em: {renderer}")
            except subprocess.CalledProcessError as e:
                print(f"[WARN] Falha ao gerar PDF (instale 'reportlab'): {e}")
        return 0

    finally:
        await db.close()


if __name__ == "__main__":
    import asyncio

    raise SystemExit(asyncio.run(main()))

