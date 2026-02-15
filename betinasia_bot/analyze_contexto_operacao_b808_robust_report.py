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


def exec_time_bucket(ms: Optional[float]) -> str:
    """Buckets operacionais por tempo total (ms)."""
    if ms is None or ms <= 0:
        return "Desconhecido"
    if ms < 5000:
        return "< 5s"
    if ms < 10000:
        return "5-10s"
    if ms < 20000:
        return "10-20s"
    if ms < 40000:
        return "20-40s"
    return "> 40s"


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
    # OBS importante de performance:
    # A versão antiga calculava `closing_odd` via subquery correlacionada em `best_odds_history`
    # para CADA auditoria. Isso explode o tempo para janelas maiores (ex.: 14 dias).
    #
    # Aqui buscamos a base "audit + match" primeiro (rápido) e calculamos `closing_odd` em batch depois.
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
                a.hypothesis_detected_at,
                a.audited_at,
                a.reversal_direction,
                a.market_period,
                a.audit_version,
                a.hypothesis_details,
                m.id AS match_id,
                m.kickoff_time,
                m.home_score,
                m.away_score,
                m.status AS match_status
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
    parser.add_argument(
        "--exec-bucket",
        default=None,
        help="Opcional. Filtra o relatório para um (ou mais) regimes operacionais por tempo total. "
        "Ex.: \"< 5s\" ou \"< 5s,5-10s\".",
    )
    parser.add_argument("--out", required=True, help="Caminho do markdown de saída (relativo a betinasia_bot/)")
    parser.add_argument(
        "--pdf",
        default=None,
        help="Se definido, renderiza o markdown para PDF (requer reportlab). Ex.: docs/relatorio.pdf",
    )
    parser.add_argument("--back-diff-min", type=float, default=2.0, help="Corte de edge Back (default: 2.0)")
    parser.add_argument("--lay-diff-max", type=float, default=-2.0, help="Corte de edge Lay (default: -2.0)")
    # OBS: argparse usa interpolação estilo `%` na help string; por isso `%` precisa ser escapado como `%%`.
    parser.add_argument("--stake-pct-of-limit", type=float, default=0.25, help="Stake fallback (%% do limite), default 0.25")
    parser.add_argument("--stake-cap", type=float, default=0.0, help="Cap opcional para stake fallback (0=sem cap)")
    parser.add_argument(
        "--git-commit",
        action="store_true",
        help="Se definido, adiciona o .md/.pdf gerados ao git e cria um commit local (não faz push).",
    )
    parser.add_argument(
        "--git-push",
        action="store_true",
        help="Se definido junto com --git-commit, faz push após o commit (usa remote/branch atuais).",
    )
    parser.add_argument(
        "--git-message",
        default=None,
        help="Mensagem do commit (opcional). Se omitida, usa uma mensagem padrão.",
    )
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
        # OBS: `closing_odd` é calculado em batch após carregar as auditorias.
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
                # timing/lag (em ms)
                "audit_total_ms": r[14],
                "lag_det_to_click_ms": r[15],
                "lag_click_to_betslip_ms": r[16],
                "hypothesis_detected_at": r[17],
                "audited_at": r[18],
                "direction": r[19],
                "period": r[20],
                "version": r[21],
                "hypothesis_details": _as_dict(r[22]),
                "match_id": int(r[23]),
                "kickoff": r[24],
                "home_score": r[25],
                "away_score": r[26],
                "match_status": r[27],
                "closing_odd": None,
            }

            # Lag fim-a-fim (proxy) e overhead:
            # - fim-a-fim = detecção->click + click->betslip
            # - overhead = duração_total - fim-a-fim (sugere fila/retries/esperas adicionais)
            det_ms = _safe_float(d.get("lag_det_to_click_ms"))
            bs_ms = _safe_float(d.get("lag_click_to_betslip_ms"))
            total_ms = _safe_float(d.get("audit_total_ms"))
            if det_ms is not None and det_ms > 0 and bs_ms is not None and bs_ms > 0:
                d["lag_e2e_ms"] = float(det_ms + bs_ms)
            else:
                d["lag_e2e_ms"] = None
            if total_ms is not None and total_ms > 0 and d.get("lag_e2e_ms") is not None:
                d["lag_overhead_ms"] = float(total_ms - float(d["lag_e2e_ms"]))
            else:
                d["lag_overhead_ms"] = None

            # Lag "total" em parede (se timestamps existem): audited_at - detected_at
            d["lag_wall_ms"] = None
            try:
                det_at = d.get("hypothesis_detected_at")
                aud_at = d.get("audited_at")
                if det_at and aud_at:
                    d["lag_wall_ms"] = float((aud_at - det_at).total_seconds() * 1000.0)
            except Exception:
                d["lag_wall_ms"] = None

            # Lag total operacional: preferimos wall_ms; fallback = audit_total_ms
            d["lag_total_ms"] = _safe_float(d.get("lag_wall_ms")) or _safe_float(d.get("audit_total_ms")) or None

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
            d["lag_bucket"] = lag_bucket(int(d.get("lag_total_ms") or 0))
            d["exec_bucket"] = exec_time_bucket(_safe_float(d.get("lag_total_ms")))

            all_data.append(d)

        # ------------------------------------------------------------
        # Closing odds em batch (melhora muito performance em janelas longas)
        # ------------------------------------------------------------
        from sqlalchemy.dialects.postgresql import ARRAY

        def _line_variants(line: Any) -> List[str]:
            s = str(line).strip().replace(",", ".")
            out = {s}
            try:
                f = float(s)
            except Exception:
                return list(out)
            # formatações comuns
            out.add(f"{f:.1f}")
            if f > 0:
                out.add(f"+{f:.1f}")
                if float(int(f)) == f:
                    out.add(f"+{int(f)}")
                    out.add(str(int(f)))
            else:
                if float(int(f)) == f:
                    out.add(str(int(f)))
            # remove/insere + e .0
            if s.startswith("+"):
                out.add(s[1:])
            if "." not in s:
                out.add(s + ".0")
                if not s.startswith(("+", "-")):
                    out.add("+" + s)
                    out.add("+" + s + ".0")
            return list(out)

        async def _fetch_closing_by_match_line(db_in: Database, match_ids: List[int]) -> Dict[Tuple[int, str], Tuple[Optional[float], Optional[float]]]:
            if not match_ids:
                return {}
            q_clo = (
                text(
                    """
                    SELECT match_id, ah_line, best_home_odds, best_away_odds
                    FROM (
                      SELECT
                        boh.match_id,
                        boh.ah_line,
                        boh.best_home_odds,
                        boh.best_away_odds,
                        row_number() OVER (PARTITION BY boh.match_id, boh.ah_line ORDER BY boh.scraped_at DESC) AS rn
                      FROM best_odds_history boh
                      JOIN matches m ON m.id = boh.match_id
                      WHERE boh.match_id = ANY(:match_ids)
                        AND boh.scraped_at < m.kickoff_time
                        AND (boh.best_home_odds > 0 OR boh.best_away_odds > 0)
                    ) t
                    WHERE rn = 1
                    """
                )
                .bindparams(bindparam("match_ids", type_=ARRAY(Integer)))
            )
            async with db_in.async_session() as session:
                res = await session.execute(q_clo, {"match_ids": match_ids})
                rows2 = list(res.fetchall())
            out: Dict[Tuple[int, str], Tuple[Optional[float], Optional[float]]] = {}
            for mid, ah_line, bho, bao in rows2:
                out[(int(mid), str(ah_line))] = (_safe_float(bho), _safe_float(bao))
            return out

        # busca closing odds apenas para matches do recorte
        match_ids_all = sorted({int(d["match_id"]) for d in all_data})
        closing_map = await _fetch_closing_by_match_line(db, match_ids_all)

        # aplica closing_odd e calcula CLV (somente AH)
        for d in all_data:
            closing = None
            try:
                if str(d.get("market_type")) != "AH":
                    closing = None
                else:
                    mid = int(d["match_id"])
                    side = str(d.get("side") or "")
                    for lv in _line_variants(d.get("line")):
                        key = (mid, str(lv))
                        if key not in closing_map:
                            continue
                        bho, bao = closing_map[key]
                        if side == "home" and bho and bho > 0:
                            closing = float(bho)
                            break
                        if side != "home" and bao and bao > 0:
                            closing = float(bao)
                            break
            except Exception:
                closing = None
            d["closing_odd"] = closing

            # CLV (bruto) depende do closing_odd
            if closing and closing > 0:
                ws = _safe_float(d.get("ws_odd"))
                bs = _safe_float(d.get("bs_odd"))
                d["clv_ws"] = (ws - closing) / closing * 100.0 if ws else None
                d["clv_bs"] = (bs - closing) / closing * 100.0 if bs and bs > 0 else None
            else:
                d["clv_ws"] = None
                d["clv_bs"] = None

        # Filtro opcional por regime de execução (tempo total) — aplicado ANTES de qualquer métrica,
        # para não misturar regimes em CLV adicional, buckets, finanças, etc.
        if args.exec_bucket:
            wanted = {x.strip() for x in str(args.exec_bucket).split(",") if x.strip()}
            all_data = [d for d in all_data if str(d.get("exec_bucket")) in wanted]

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
        title = "Análise Estatística Robusta — Contexto Operação (b808)"
        if args.exec_bucket:
            title += f" — Regime(s): {args.exec_bucket}"
        lines.append(f"# {title}\n")
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
        # Tempo total observado (preferindo wall_ms quando disponível)
        api_lag_total = _mean([float(d["lag_total_ms"]) for d in api_all if d.get("lag_total_ms") is not None])
        dom_lag_total = _mean([float(d["lag_total_ms"]) for d in dom_all if d.get("lag_total_ms") is not None])
        # Decomposição instrumentada (não inclui tudo, mas ajuda a identificar gargalo)
        api_lag_e2e = _mean([float(d["lag_e2e_ms"]) for d in api_all if d.get("lag_e2e_ms") is not None])
        dom_lag_e2e = _mean([float(d["lag_e2e_ms"]) for d in dom_all if d.get("lag_e2e_ms") is not None])
        api_clv_pm_n = len([d for d in api_bs if d.get("clv_bs") is not None and d.get("is_live") is False and -50 < float(d["clv_bs"]) < 50])
        dom_clv_pm_n = len([d for d in dom_bs if d.get("clv_bs") is not None and d.get("is_live") is False and -50 < float(d["clv_bs"]) < 50])
        api_roi_n = len([d for d in api_bs if d.get("roi_bs") is not None])
        dom_roi_n = len([d for d in dom_bs if d.get("roi_bs") is not None])
        lines.append(f"| Total de observações | {len(api_all)} | {len(dom_all)} |\n")
        lines.append(f"| Com betslip confiável | {len(api_bs)} | {len(dom_bs)} |\n")
        lines.append(f"| Com CLV pre-match (betslip) | {api_clv_pm_n} | {dom_clv_pm_n} |\n")
        lines.append(f"| Com ROI (betslip) | {api_roi_n} | {dom_roi_n} |\n")
        lines.append(f"| Tempo total observado (detecção→betslip, wall/total) | {_fmt_num(api_lag_total, 0)} ms | {_fmt_num(dom_lag_total, 0)} ms |\n")
        lines.append(f"| Tempo instrumentado (detecção→clique→betslip) | {_fmt_num(api_lag_e2e, 0)} ms | {_fmt_num(dom_lag_e2e, 0)} ms |\n")
        lines.append("\n---\n")

        # 2.0b) Decomposição de latência (foco: fila/overhead)
        lines.append("### 2.0b Decomposição do tempo (detecção→clique→betslip vs. overhead)\n")
        lines.append(
            "Interpretação: `lag_e2e` é o **tempo fim-a-fim** (detecção→clique + clique→betslip). "
            "`overhead` = `audit_total` − `lag_e2e` (proxy de fila/retries/esperas fora das 2 etapas instrumentadas).\n\n"
        )

        def _stage_stats(rows_in: List[Dict[str, Any]], key: str) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
            vals: List[float] = []
            for d in rows_in:
                v = _safe_float(d.get(key))
                if v is None:
                    continue
                vals.append(float(v))
            # remove zeros e negativos para tempos, exceto overhead (pode ser <0 por inconsistência de medição)
            if key != "lag_overhead_ms":
                vals = [v for v in vals if v > 0]
            if not vals:
                return None, None, None, 0
            return float(np.mean(vals)), float(np.median(vals)), float(np.quantile(vals, 0.95)), len(vals)

        lines.append("| Modelo | Métrica | mean (ms) | p50 (ms) | p95 (ms) | N |\n|---|---|---:|---:|---:|---:|\n")
        for model_name, rows_in in [("API (2-4s)", api_all), ("DOM (15-30s)", dom_all)]:
            for label, key in [
                ("lag_det→click", "lag_det_to_click_ms"),
                ("lag_click→betslip", "lag_click_to_betslip_ms"),
                ("lag_e2e (soma)", "lag_e2e_ms"),
                ("audit_total (duração)", "audit_total_ms"),
                ("overhead (total - e2e)", "lag_overhead_ms"),
            ]:
                mu, p50, p95, n = _stage_stats(rows_in, key)
                lines.append(f"| {model_name} | {label} | {_fmt_num(mu,0)} | {_fmt_num(p50,0)} | {_fmt_num(p95,0)} | {n} |\n")
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
            f"| CLV (apenas pre-match) | {len([d for d in with_bs if d.get('is_live') is False and d.get('clv_bs') is not None and -50 < float(d['clv_bs']) < 50])} | — | CLV vs closing pré-jogo não é interpretável in-match |\n"
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

        if not args.exec_bucket:
            # 2.3) Regimes operacionais por bucket de tempo total (lag_total_ms)
            lines.append("### 2.3 Regimes operacionais por tempo total (bucket)\n")
            lines.append(
                "Objetivo: separar a amostra em **regimes de execução** (tempo total) e medir performance em cada regime. "
                "Use isso para detectar **fila/saturação**: regimes lentos tendem a ter edge menor e pior ROI.\n\n"
            )
            lines.append("| Bucket (tempo total) | N OK | Jogos | lag_total mean (ms) | lag_total p95 (ms) | overhead p95 (ms) | Back edge | Lay edge | CLV PM (mean; IC90) | ROI (mean; IC90) |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for b in ["< 5s", "5-10s", "10-20s", "20-40s", "> 40s", "Desconhecido"]:
                sub = [d for d in ok_bs if str(d.get("exec_bucket")) == b]
                if not sub:
                    lines.append(f"| {b} | 0 | 0 | — | — | — | 0 | 0 | — | — |\n")
                    continue
                lags = [float(x) for x in [(_safe_float(d.get("lag_total_ms"))) for d in sub] if x is not None and x > 0]
                over = [float(x) for x in [(_safe_float(d.get("lag_overhead_ms"))) for d in sub] if x is not None]
                lag_mean = float(np.mean(lags)) if lags else None
                lag_p95 = float(np.quantile(lags, 0.95)) if lags else None
                ov_p95 = float(np.quantile(over, 0.95)) if over else None

                n_matches = len({int(d["match_id"]) for d in sub})
                n_back = sum(1 for d in sub if int(d["id"]) in back_edge_ids)
                n_lay = sum(1 for d in sub if int(d["id"]) in lay_edge_ids)

                clv_pm = summarize_metric(
                    [d.get("clv_bs") for d in sub if d.get("is_live") is False],
                    [d.get("match_id") for d in sub if d.get("is_live") is False],
                    clip_low=-50,
                    clip_high=50,
                )
                roi_all = summarize_metric(
                    [d.get("roi_bs") for d in sub],
                    [d.get("match_id") for d in sub],
                    clip_low=-100,
                    clip_high=500,
                )
                clv_txt = f"{_fmt_pct(clv_pm.mean_cluster,2)} {_fmt_ci(clv_pm.ci90_cluster,2)}" if clv_pm.n_events else "—"
                roi_txt = f"{_fmt_pct(roi_all.mean_cluster,2)} {_fmt_ci(roi_all.ci90_cluster,2)}" if roi_all.n_events else "—"
                lines.append(
                    f"| {b} | {len(sub)} | {n_matches} | {_fmt_num(lag_mean,0)} | {_fmt_num(lag_p95,0)} | {_fmt_num(ov_p95,0)} | {n_back} | {n_lay} | {clv_txt} | {roi_txt} |\n"
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
            # IMPORTANTE:
            # - CLV é % vs closing e pode ser sanity-clipped (evita lixo/parse errado).
            # - ROI pode assumir -100% em loss e >50% em wins (principalmente odds altas),
            #   então NÃO pode herdar o clip de CLV, senão distorce N e win rate.
            if "clv" in str(key).lower():
                return summarize_metric(vals, mids, clip_low=-50, clip_high=50)
            if "roi" in str(key).lower():
                return summarize_metric(vals, mids, clip_low=-100, clip_high=500)
            return summarize_metric(vals, mids)

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
        lines.append("\nNotas de robustez (IC 90% por jogo):  \n")
        lines.append(f"- API ROI betslip (cluster): média {_fmt_pct(api_roi.mean_cluster)}; IC90 {_fmt_ci(api_roi.ci90_cluster)}  \n")
        lines.append(f"- API ROI WS (cluster): média {_fmt_pct(api_roi_ws.mean_cluster)}; IC90 {_fmt_ci(api_roi_ws.ci90_cluster)}  \n")
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

        # ROI realizado (quando houver placar)
        back_realized = []
        back_realized_stakes = []
        back_realized_roi = []
        back_realized_mids = []
        for d in back_edge:
            roi = _safe_float(d.get("roi_bs"))
            if roi is None:
                continue
            bs, _, _, _ = finance_for_row(d)
            back_realized.append(float(bs) * float(roi) / 100.0)
            back_realized_stakes.append(float(bs))
            back_realized_roi.append(float(roi))
            back_realized_mids.append(int(d.get("match_id")))
        if back_realized:
            roi_weighted = (sum(back_realized) / sum(back_realized_stakes) * 100.0) if sum(back_realized_stakes) > 0 else None
            lines.append(f"| N com ROI realizado | {len(back_realized)} |\n")
            lines.append(f"| P&L realizado total (estimado) | {_fmt_num(sum(back_realized), 2)} |\n")
            lines.append(f"| ROI realizado (ponderado por stake) | {_fmt_num(roi_weighted, 2)}% |\n")

            roi_sum = summarize_metric(back_realized_roi, back_realized_mids, clip_low=-100, clip_high=500)
            lines.append(f"| ROI realizado (robusto por jogo, mean; IC90) | {_fmt_pct(roi_sum.mean_cluster,2)} {_fmt_ci(roi_sum.ci90_cluster,2)} |\n")

            # IC90 também para ROI ponderado por stake (agrega por jogo e bootstrap)
            by_match_w: Dict[int, Tuple[float, float]] = {}
            for d in back_edge:
                roi = _safe_float(d.get("roi_bs"))
                if roi is None:
                    continue
                mid = int(d.get("match_id"))
                bs, _, _, _ = finance_for_row(d)
                pnl = float(bs) * float(roi) / 100.0
                s, w = by_match_w.get(mid, (0.0, 0.0))
                by_match_w[mid] = (s + pnl, w + float(bs))
            per_match_roi = {mid: (pnl / w * 100.0) for mid, (pnl, w) in by_match_w.items() if w > 0}
            mean_w, ci_w = cluster_bootstrap_ci({mid: [val] for mid, val in per_match_roi.items()}, n_boot=4000, alpha=0.10)
            lines.append(f"| ROI ponderado por stake (robusto por jogo, mean; IC90) | {_fmt_pct(mean_w,2)} {_fmt_ci(ci_w,2)} |\n")
        else:
            lines.append("| N com ROI realizado | 0 (placares ausentes no recorte) |\n")
        lines.append(
            "\nObservação: a Seção 4 reporta ROI médio no **recorte completo** (betslip confiável). "
            "Aqui (7.1) é apenas a coorte **Back edge** (diff>=corte) e o ROI também é mostrado ponderado por stake; "
            "por isso sinais podem divergir.\n"
        )

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
        lines.append(f"| Proxy de banca (>= p99 liability) | {_fmt_num(_pctl(lay_liability, 99), 2)} |\n")

        # ROI/P&L realizado no Lay (quando houver placar)
        def _mult_back_from_row(d: dict) -> Optional[float]:
            hs = d.get("home_score")
            aws = d.get("away_score")
            if hs is None or aws is None:
                return None
            try:
                goal_diff = int(hs) - int(aws)
            except Exception:
                return None
            try:
                ah_line = float(str(d.get("line", "")).replace(",", "."))
            except Exception:
                return None
            side = (str(d.get("side") or "")).strip()
            if side == "home":
                adjusted = goal_diff + ah_line
            else:
                adjusted = -goal_diff - ah_line
            if adjusted > 0.25:
                return 1.0
            if adjusted == 0.25:
                return 0.5
            if adjusted == 0:
                return 0.0
            if adjusted == -0.25:
                return -0.5
            return -1.0

        lay_realized_pnl = []
        lay_realized_liab = []
        lay_realized_stake = []
        lay_realized_roi_liab = []
        lay_realized_roi_stake = []
        lay_realized_mids = []
        for d in lay_edge:
            mult = _mult_back_from_row(d)
            if mult is None:
                continue
            h = d.get("hypothesis_details") or {}
            lay_odd = _safe_float(_get_path(h, ["lay", "odd"]))
            if lay_odd is None or lay_odd <= 0:
                lay_odd = _safe_float(d.get("bs_odd"))
            if lay_odd is None or lay_odd <= 1.0:
                continue
            _, _, ls, ll = finance_for_row(d)
            ls = float(ls)
            ll = float(ll)
            if ls <= 0 or ll <= 0:
                continue
            # ROI por stake e por liability
            roi_stake = (-mult) * 100.0 if mult < 0 else (-mult) * (lay_odd - 1.0) * 100.0 if mult > 0 else 0.0
            roi_stake = float(roi_stake)
            roi_liab = (-mult) / (lay_odd - 1.0) * 100.0 if mult < 0 else (-mult) * 100.0 if mult > 0 else 0.0
            roi_liab = float(roi_liab)
            pnl = ls * roi_stake / 100.0  # equivale a ll * roi_liab/100

            lay_realized_pnl.append(pnl)
            lay_realized_liab.append(ll)
            lay_realized_stake.append(ls)
            lay_realized_roi_liab.append(roi_liab)
            lay_realized_roi_stake.append(roi_stake)
            lay_realized_mids.append(int(d.get("match_id")))

        if lay_realized_pnl:
            roi_liab_weighted = (sum(lay_realized_pnl) / sum(lay_realized_liab) * 100.0) if sum(lay_realized_liab) > 0 else None
            roi_stake_weighted = (sum(lay_realized_pnl) / sum(lay_realized_stake) * 100.0) if sum(lay_realized_stake) > 0 else None
            lines.append(f"| N com ROI realizado | {len(lay_realized_pnl)} |\n")
            lines.append(f"| P&L realizado total (estimado) | {_fmt_num(sum(lay_realized_pnl), 2)} |\n")
            lines.append(f"| ROI realizado (ponderado por liability) | {_fmt_num(roi_liab_weighted, 2)}% |\n")
            lines.append(f"| ROI realizado (ponderado por stake) | {_fmt_num(roi_stake_weighted, 2)}% |\n")

            # robusto por jogo (não ponderado)
            roi_liab_sum = summarize_metric(lay_realized_roi_liab, lay_realized_mids, clip_low=-200, clip_high=5000)
            lines.append(f"| ROI/liability (robusto por jogo, mean; IC90) | {_fmt_pct(roi_liab_sum.mean_cluster,2)} {_fmt_ci(roi_liab_sum.ci90_cluster,2)} |\n")

            # robusto por jogo (ponderado por liability)
            by_match_l: Dict[int, Tuple[float, float]] = {}
            for d in lay_edge:
                mult = _mult_back_from_row(d)
                if mult is None:
                    continue
                h = d.get("hypothesis_details") or {}
                lay_odd = _safe_float(_get_path(h, ["lay", "odd"]))
                if lay_odd is None or lay_odd <= 0:
                    lay_odd = _safe_float(d.get("bs_odd"))
                if lay_odd is None or lay_odd <= 1.0:
                    continue
                _, _, ls, ll = finance_for_row(d)
                ls = float(ls)
                ll = float(ll)
                if ls <= 0 or ll <= 0:
                    continue
                roi_stake = (-mult) * 100.0 if mult < 0 else (-mult) * (lay_odd - 1.0) * 100.0 if mult > 0 else 0.0
                pnl = ls * float(roi_stake) / 100.0
                mid = int(d.get("match_id"))
                s, w = by_match_l.get(mid, (0.0, 0.0))
                by_match_l[mid] = (s + pnl, w + ll)
            per_match_roi_l = {mid: (pnl / w * 100.0) for mid, (pnl, w) in by_match_l.items() if w > 0}
            mean_lw, ci_lw = cluster_bootstrap_ci({mid: [val] for mid, val in per_match_roi_l.items()}, n_boot=4000, alpha=0.10)
            lines.append(f"| ROI/liability ponderado (robusto por jogo, mean; IC90) | {_fmt_pct(mean_lw,2)} {_fmt_ci(ci_lw,2)} |\n")
        else:
            lines.append("| N com ROI realizado | 0 (placares ausentes no recorte) |\n")
        lines.append("\n---\n")

        # ============================================================
        # 8) Curva temporal (pico, reversão e melhor timing)
        # ============================================================
        EPS_DIFF_STABLE = 0.20   # variação < 0.20pp conta como estável (ruído)
        EPS_REV = 0.50           # reversão = queda/subida >= 0.50pp vs pico/vale

        def _outcome_mult(line: str, side: str, hs: Any, aws: Any) -> Optional[float]:
            """Multiplicador do resultado para back (1, 0.5, 0, -0.5, -1)."""
            if hs is None or aws is None:
                return None
            try:
                goal_diff = int(hs) - int(aws)
            except Exception:
                return None
            try:
                ah_line = float(str(line).replace(",", "."))
            except Exception:
                return None
            if (side or "").strip() == "home":
                adjusted = goal_diff + ah_line
            else:
                adjusted = -goal_diff - ah_line
            if adjusted > 0.25:
                return 1.0
            if adjusted == 0.25:
                return 0.5
            if adjusted == 0:
                return 0.0
            if adjusted == -0.25:
                return -0.5
            return -1.0

        def _roi_back_pct(odd: float, mult: float) -> float:
            if mult > 0:
                return (odd - 1.0) * mult * 100.0
            if mult < 0:
                return mult * 100.0
            return 0.0

        def _roi_lay_pct_per_stake(lay_odd: float, mult_back: float) -> float:
            # stake=1 (ganho máx = +1), perda = -(odd-1) quando o back vence
            if mult_back > 0:
                return -mult_back * max(0.0, lay_odd - 1.0) * 100.0
            if mult_back < 0:
                return (-mult_back) * 100.0
            return 0.0

        def _roi_lay_pct_per_liability(lay_odd: float, mult_back: float) -> Optional[float]:
            liab = max(0.0, lay_odd - 1.0)
            if liab <= 0:
                return None
            if mult_back > 0:
                return -mult_back * 100.0
            if mult_back < 0:
                # lucro por stake = -mult_back (ex.: 1 ou 0.5), divide pela liability
                return ((-mult_back) / liab) * 100.0
            return 0.0

        def _build_back_series(d: dict) -> List[dict]:
            h = d.get("hypothesis_details") or {}
            series = []
            # T0 (auditoria)
            t0_diff = _safe_float(d.get("diff_pct"))
            t0_odd = _safe_float(d.get("bs_odd"))
            if t0_diff is not None and t0_odd is not None:
                series.append({"t": 0.0, "diff_pct": t0_diff, "odd": t0_odd})
            arr = h.get("temporal")
            if isinstance(arr, list):
                for e in arr:
                    if not isinstance(e, dict):
                        continue
                    t = _safe_float(e.get("t"))
                    diff = _safe_float(e.get("diff_pct"))
                    odd = _safe_float(e.get("bs_odd"))
                    if t is None or diff is None or odd is None:
                        continue
                    series.append({"t": float(t), "diff_pct": float(diff), "odd": float(odd)})
            series.sort(key=lambda x: x["t"])
            return series

        def _build_lay_series(d: dict) -> List[dict]:
            h = d.get("hypothesis_details") or {}
            series = []
            ws_odd = _safe_float(d.get("ws_odd"))
            lay0 = _safe_float(_get_path(h, ["lay", "odd"]))
            if ws_odd and lay0:
                series.append(
                    {"t": 0.0, "diff_pct": ((lay0 - ws_odd) / ws_odd) * 100.0, "odd": lay0}
                )
            arr = h.get("lay_temporal")
            if isinstance(arr, list):
                for e in arr:
                    if not isinstance(e, dict):
                        continue
                    t = _safe_float(e.get("t"))
                    diff = _safe_float(e.get("diff_pct"))
                    odd = _safe_float(e.get("lay_odd"))
                    if t is None or diff is None or odd is None:
                        continue
                    series.append({"t": float(t), "diff_pct": float(diff), "odd": float(odd)})
            series.sort(key=lambda x: x["t"])
            return series

        def _analyze_peak(series: List[dict], mode: str) -> dict:
            """
            mode='back': pico = max(diff_pct)
            mode='lay' : vale = min(diff_pct) (mais negativo)
            """
            if not series:
                return {"n": 0}
            diffs = [p["diff_pct"] for p in series]
            if mode == "back":
                idx_ext = int(np.argmax(diffs))
            else:
                idx_ext = int(np.argmin(diffs))
            ext = series[idx_ext]
            last = series[-1]
            # monotonicidade (para "indefinidamente")
            mono = True
            for a, b in zip(diffs, diffs[1:]):
                if mode == "back" and (b < a - EPS_DIFF_STABLE):
                    mono = False
                    break
                if mode == "lay" and (b > a + EPS_DIFF_STABLE):
                    mono = False
                    break
            # reversão: qualquer ponto após o extremo que "volta" >= EPS_REV
            after = diffs[idx_ext + 1 :] if idx_ext + 1 < len(diffs) else []
            had_rev = False
            t_rev = None
            if after:
                if mode == "back":
                    threshold = ext["diff_pct"] - EPS_REV
                    for p in series[idx_ext + 1 :]:
                        if p["diff_pct"] <= threshold:
                            had_rev = True
                            t_rev = p["t"]
                            break
                else:
                    threshold = ext["diff_pct"] + EPS_REV
                    for p in series[idx_ext + 1 :]:
                        if p["diff_pct"] >= threshold:
                            had_rev = True
                            t_rev = p["t"]
                            break
            ext_at_end = abs(last["diff_pct"] - ext["diff_pct"]) <= EPS_DIFF_STABLE
            return {
                "n": len(series),
                "t_ext": float(ext["t"]),
                "diff_ext": float(ext["diff_pct"]),
                "odd_ext": float(ext["odd"]),
                "t_last": float(last["t"]),
                "diff_last": float(last["diff_pct"]),
                "odd_last": float(last["odd"]),
                "monotonic": bool(mono),
                "ext_at_end": bool(ext_at_end),
                "had_reversal": bool(had_rev),
                "t_reversal": float(t_rev) if t_rev is not None else None,
            }

        def _clv_pct_from_odd(odd: Optional[float], closing_odd: Any) -> Optional[float]:
            odd = _safe_float(odd)
            clo = _safe_float(closing_odd)
            if odd is None or clo is None or clo <= 0:
                return None
            return (odd - clo) / clo * 100.0

        def _clv_pct_lay_from_odd(lay_odd: Optional[float], closing_odd: Any) -> Optional[float]:
            """
            Para Lay, o sinal é invertido: é "bom" quando você LAYA barato e o closing sobe.
            Definição: (closing - entry) / closing * 100
            """
            lay_odd = _safe_float(lay_odd)
            clo = _safe_float(closing_odd)
            if lay_odd is None or clo is None or clo <= 0:
                return None
            return (clo - lay_odd) / clo * 100.0

        def _summarize_timing(rows_in: List[dict], mode: str) -> Dict[str, Any]:
            stats = []
            for d in rows_in:
                if mode == "back":
                    s = _build_back_series(d)
                else:
                    s = _build_lay_series(d)
                a = _analyze_peak(s, mode=mode)
                if a.get("n", 0) == 0:
                    continue
                a["is_live"] = d.get("is_live")
                a["match_id"] = d.get("match_id")
                a["closing_odd"] = d.get("closing_odd")
                a["line"] = d.get("line")
                a["side"] = d.get("side")
                a["hs"] = d.get("home_score")
                a["as"] = d.get("away_score")
                stats.append(a)
            return {"rows": stats}

        def _agg_by_regime(stats_rows: List[dict]) -> Dict[str, dict]:
            out: Dict[str, dict] = {}
            for regime, is_live_val in [("PRE_MATCH", False), ("IN_MATCH", True)]:
                sub = [r for r in stats_rows if r.get("is_live") is is_live_val]
                if not sub:
                    out[regime] = {"n": 0}
                    continue
                t_ext = [r["t_ext"] for r in sub if r.get("t_ext") is not None]
                t_rev = [r["t_reversal"] for r in sub if r.get("t_reversal") is not None]
                t_rev_delay = [
                    float(r["t_reversal"]) - float(r["t_ext"])
                    for r in sub
                    if r.get("t_reversal") is not None and r.get("t_ext") is not None
                ]
                # Partição 100% (categorias exclusivas)
                end_no_rev = sum(1 for r in sub if r.get("ext_at_end") and not r.get("had_reversal"))
                end_with_rev = sum(1 for r in sub if r.get("ext_at_end") and r.get("had_reversal"))
                not_end_with_rev = sum(1 for r in sub if (not r.get("ext_at_end")) and r.get("had_reversal"))
                not_end_no_rev = sum(1 for r in sub if (not r.get("ext_at_end")) and not r.get("had_reversal"))
                out[regime] = {
                    "n": len(sub),
                    "t_ext_avg": float(np.mean(t_ext)) if t_ext else None,
                    "t_ext_p50": float(np.median(t_ext)) if t_ext else None,
                    "pct_monotonic": 100.0 * sum(1 for r in sub if r.get("monotonic")) / len(sub),
                    # "melhora até o fim" = monotônico + extremo no fim (proxy do "sobe/desce indef.")
                    "pct_improve_to_end": 100.0 * sum(1 for r in sub if r.get("monotonic") and r.get("ext_at_end")) / len(sub),
                    "pct_reversal": 100.0 * sum(1 for r in sub if r.get("had_reversal")) / len(sub),
                    "t_rev_avg": float(np.mean(t_rev)) if t_rev else None,
                    "t_rev_p50": float(np.median(t_rev)) if t_rev else None,
                    "dt_rev_avg": float(np.mean(t_rev_delay)) if t_rev_delay else None,
                    "dt_rev_p50": float(np.median(t_rev_delay)) if t_rev_delay else None,
                    # partição 100% (exclusiva)
                    "pct_end_no_rev": 100.0 * end_no_rev / len(sub),
                    "pct_end_with_rev": 100.0 * end_with_rev / len(sub),
                    "pct_not_end_with_rev": 100.0 * not_end_with_rev / len(sub),
                    "pct_not_end_no_rev": 100.0 * not_end_no_rev / len(sub),
                }
            return out

        def _curve_table(rows_in: List[dict], mode: str) -> List[Tuple[str, int, float, float, Optional[float], Optional[float]]]:
            """
            Retorna linhas por tempo: (t_label, n, mean_diff, mean_odd, mean_clv, mean_roi)
            """
            times = [0, 3, 6, 10, 15, 20]
            buckets: Dict[int, List[Tuple[float, float, Optional[float], Optional[float]]]] = {t: [] for t in times}
            for d in rows_in:
                series = _build_back_series(d) if mode == "back" else _build_lay_series(d)
                if not series:
                    continue
                # outcome
                mult = _outcome_mult(str(d.get("line", "")), str(d.get("side", "")), d.get("home_score"), d.get("away_score"))
                for p in series:
                    # bin pelo target mais próximo
                    t = float(p["t"])
                    tgt = min(times, key=lambda x: abs(x - t))
                    diff = float(p["diff_pct"])
                    odd = float(p["odd"])
                    # CLV só faz sentido pre-match (closing_odd é pré-jogo).
                    if d.get("is_live") is True:
                        clv = None
                    else:
                        if mode == "back":
                            clv = _clv_pct_from_odd(odd, d.get("closing_odd"))
                        else:
                            clv = _clv_pct_lay_from_odd(odd, d.get("closing_odd"))
                    roi = None
                    if mult is not None:
                        if mode == "back":
                            roi = _roi_back_pct(odd, mult)
                        else:
                            roi = _roi_lay_pct_per_liability(odd, mult)
                    buckets[tgt].append((diff, odd, clv, roi))
            out = []
            for t in times:
                pts = buckets[t]
                if not pts:
                    continue
                diffs = [x[0] for x in pts]
                odds = [x[1] for x in pts]
                clvs = [x[2] for x in pts if x[2] is not None]
                rois = [x[3] for x in pts if x[3] is not None]
                out.append((f"t+{t}s", len(pts), float(np.mean(diffs)), float(np.mean(odds)), float(np.mean(clvs)) if clvs else None, float(np.mean(rois)) if rois else None))
            return out

        lines.append("## 8) Curva temporal (pico, reversão e melhor timing)\n")
        lines.append("Esta seção usa `hypothesis_details.temporal` (Back) e `hypothesis_details.lay_temporal` (Lay), coletados em pontos discretos (t≈0,3,6,10,15,20s).\n\n")
        lines.append(
            "O objetivo é responder: **tempo até o pico/vale**, **% que segue melhorando até t_max**, **% com reversão** e **impacto esperado por timing**. "
            "**CLV é reportado somente pre-match** (closing pré-jogo).\n\n"
        )

        # BACK
        back_stats = _summarize_timing(ok_bs, mode="back")["rows"]
        back_agg = _agg_by_regime(back_stats)
        lines.append("### 8.1 Back (pico em diff_pct)\n")
        lines.append(
            f"Definições: **pico** = `max(diff_pct)`. **reversão** = após o pico, `diff_pct` cair pelo menos {EPS_REV:.2f} p.p. "
            f"(para Lay, subir {EPS_REV:.2f} p.p. após o vale). `t_reversão` é o 1º tempo que cruza esse limiar (logo, não é o pico).\n\n"
        )
        lines.append("| Regime | N | t_pico médio (s) | t_pico p50 (s) | % melhora até fim | % com reversão | t_reversão médio (s) | Δt (rev - pico) médio (s) |\n|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for regime in ["PRE_MATCH", "IN_MATCH"]:
            s = back_agg.get(regime, {"n": 0})
            if s.get("n", 0) == 0:
                lines.append(f"| {regime} | 0 | — | — | — | — | — | — |\n")
                continue
            lines.append(
                f"| {regime} | {s['n']} | {_fmt_num(s.get('t_ext_avg'),1)} | {_fmt_num(s.get('t_ext_p50'),1)} | {_fmt_num(s.get('pct_improve_to_end'),1)}% | {_fmt_num(s.get('pct_reversal'),1)}% | {_fmt_num(s.get('t_rev_avg'),1)} | {_fmt_num(s.get('dt_rev_avg'),1)} |\n"
            )

        lines.append("\n**Partição 100% (Back)**: categorias exclusivas (somam 100% por regime).\n\n")
        lines.append("| Regime | % pico no fim, sem reversão | % pico no fim, com reversão (recuperou) | % pico antes, com reversão | % pico antes, sem reversão relevante |\n|---|---:|---:|---:|---:|\n")
        for regime in ["PRE_MATCH", "IN_MATCH"]:
            s = back_agg.get(regime, {"n": 0})
            if s.get("n", 0) == 0:
                lines.append(f"| {regime} | — | — | — | — |\n")
                continue
            lines.append(
                f"| {regime} | {_fmt_num(s.get('pct_end_no_rev'),1)}% | {_fmt_num(s.get('pct_end_with_rev'),1)}% | {_fmt_num(s.get('pct_not_end_with_rev'),1)}% | {_fmt_num(s.get('pct_not_end_no_rev'),1)}% |\n"
            )

        lines.append("\n**Curva média (Back)**: média de `diff_pct` e `odd` por tempo. `CLV` é reportado **somente pre-match** (closing pré-jogo). `ROI` (quando aparece) é o **ROI realizado** se a aposta fosse feita naquele ponto.\n\n")
        lines.append("| Tempo | N pts | diff_pct médio | odd média | CLV médio (pre-match) | ROI médio (se houver) |\n|---|---:|---:|---:|---:|---:|\n")
        for t_label, n, md, mo, mclv, mroi in _curve_table(ok_bs, mode="back"):
            lines.append(f"| {t_label} | {n} | {_fmt_pct(md,2)} | {_fmt_num(mo,3)} | {_fmt_pct(mclv,2)} | {_fmt_num(mroi,2)} |\n")

        def _entry_metrics_back(d: dict) -> Optional[dict]:
            series = _build_back_series(d)
            if not series:
                return None
            a = _analyze_peak(series, mode="back")
            if a.get("n", 0) <= 0:
                return None
            closing = d.get("closing_odd")
            # define pontos
            p0 = series[0]  # t=0 incluído quando houver
            plast = series[-1]
            # ROI por ponto (se houver placar)
            mult = _outcome_mult(str(d.get("line", "")), str(d.get("side", "")), d.get("home_score"), d.get("away_score"))
            roi0 = _roi_back_pct(p0["odd"], mult) if mult is not None else None
            roipeak = _roi_back_pct(a["odd_ext"], mult) if mult is not None else None
            roilast = _roi_back_pct(plast["odd"], mult) if mult is not None else None
            # CLV só faz sentido pre-match (closing_odd é pré-jogo).
            clv0 = _clv_pct_from_odd(p0["odd"], closing) if d.get("is_live") is False else None
            clve = _clv_pct_from_odd(a["odd_ext"], closing) if d.get("is_live") is False else None
            clvl = _clv_pct_from_odd(plast["odd"], closing) if d.get("is_live") is False else None
            return {
                "match_id": int(d.get("match_id")),
                "is_live": d.get("is_live"),
                "had_reversal": bool(a.get("had_reversal")),
                "ext_at_end": bool(a.get("ext_at_end")),
                "monotonic": bool(a.get("monotonic")),
                "t_ext": a.get("t_ext"),
                "odd_t0": float(p0["odd"]),
                "odd_ext": float(a["odd_ext"]),
                "odd_last": float(plast["odd"]),
                "closing_odd": _safe_float(closing),
                "clv_t0": clv0,
                "clv_ext": clve,
                "clv_last": clvl,
                "roi_t0": roi0,
                "roi_ext": roipeak,
                "roi_last": roilast,
            }

        def _entry_metrics_lay(d: dict) -> Optional[dict]:
            series = _build_lay_series(d)
            if not series:
                return None
            a = _analyze_peak(series, mode="lay")
            if a.get("n", 0) <= 0:
                return None
            closing = d.get("closing_odd")
            p0 = series[0]
            plast = series[-1]
            mult = _outcome_mult(str(d.get("line", "")), str(d.get("side", "")), d.get("home_score"), d.get("away_score"))
            roi0 = _roi_lay_pct_per_liability(p0["odd"], mult) if mult is not None else None
            roival = _roi_lay_pct_per_liability(a["odd_ext"], mult) if mult is not None else None
            roilast = _roi_lay_pct_per_liability(plast["odd"], mult) if mult is not None else None
            # CLV só faz sentido pre-match (closing_odd é pré-jogo).
            clv0 = _clv_pct_lay_from_odd(p0["odd"], closing) if d.get("is_live") is False else None
            clve = _clv_pct_lay_from_odd(a["odd_ext"], closing) if d.get("is_live") is False else None
            clvl = _clv_pct_lay_from_odd(plast["odd"], closing) if d.get("is_live") is False else None
            return {
                "match_id": int(d.get("match_id")),
                "is_live": d.get("is_live"),
                "had_reversal": bool(a.get("had_reversal")),
                "ext_at_end": bool(a.get("ext_at_end")),
                "monotonic": bool(a.get("monotonic")),
                "t_ext": a.get("t_ext"),
                "clv_t0": clv0,
                "clv_ext": clve,
                "clv_last": clvl,
                "roi_t0": roi0,
                "roi_ext": roival,
                "roi_last": roilast,
            }

        def _summarize_entry(rows: List[dict], key: str, *, clip_low: Optional[float] = None, clip_high: Optional[float] = None) -> MetricSummary:
            vals = [r.get(key) for r in rows]
            mids = [r.get("match_id") for r in rows]
            return summarize_metric(vals, mids, clip_low=clip_low, clip_high=clip_high)

        # 8.1b) impacto: t0 vs pico vs último, com/sem reversão
        back_entries = [em for d in ok_bs for em in [_entry_metrics_back(d)] if em is not None]
        lines.append("\n### 8.1b Back — impacto por timing (t0 vs pico vs último) e reversão\n")
        lines.append(
            "Leitura: se a estratégia é **entrar no pico**, compare CLV/ROI no pico vs t0 e veja diferença entre **casos com reversão** vs **sem reversão**. "
            "Aqui reportamos **média robusta por jogo + IC90**. Observação: **CLV só é calculado pre-match**.\n\n"
        )

        # CLV (pre-match) com IC
        lines.append("**CLV (somente pre-match) — média robusta por jogo (IC90)**\n\n")
        lines.append("| Subcoorte | N total | N CLV (PM) | CLV t0 (mean; IC90) | CLV pico (mean; IC90) | CLV último (mean; IC90) |\n|---|---:|---:|---:|---:|---:|\n")
        for label, filt in [("SEM_REVERSAO", [r for r in back_entries if not r["had_reversal"]]), ("COM_REVERSAO", [r for r in back_entries if r["had_reversal"]])]:
            pm = [r for r in filt if r.get("is_live") is False]
            s0 = _summarize_entry(pm, "clv_t0", clip_low=-50, clip_high=50)
            se = _summarize_entry(pm, "clv_ext", clip_low=-50, clip_high=50)
            sl = _summarize_entry(pm, "clv_last", clip_low=-50, clip_high=50)
            lines.append(
                f"| {label} | {len(filt)} | {s0.n_events} | {_fmt_pct(s0.mean_cluster,2)} {_fmt_ci(s0.ci90_cluster,2)} | {_fmt_pct(se.mean_cluster,2)} {_fmt_ci(se.ci90_cluster,2)} | {_fmt_pct(sl.mean_cluster,2)} {_fmt_ci(sl.ci90_cluster,2)} |\n"
            )

        # ROI (back) com IC
        lines.append("\n**ROI (stake=1) — média robusta por jogo (IC90)**\n\n")
        lines.append("| Subcoorte | N total | N ROI | ROI t0 (mean; IC90) | ROI pico (mean; IC90) | ROI último (mean; IC90) |\n|---|---:|---:|---:|---:|---:|\n")
        for label, filt in [("SEM_REVERSAO", [r for r in back_entries if not r["had_reversal"]]), ("COM_REVERSAO", [r for r in back_entries if r["had_reversal"]])]:
            s0 = _summarize_entry(filt, "roi_t0", clip_low=-100, clip_high=500)
            se = _summarize_entry(filt, "roi_ext", clip_low=-100, clip_high=500)
            sl = _summarize_entry(filt, "roi_last", clip_low=-100, clip_high=500)
            lines.append(
                f"| {label} | {len(filt)} | {s0.n_events} | {_fmt_pct(s0.mean_cluster,2)} {_fmt_ci(s0.ci90_cluster,2)} | {_fmt_pct(se.mean_cluster,2)} {_fmt_ci(se.ci90_cluster,2)} | {_fmt_pct(sl.mean_cluster,2)} {_fmt_ci(sl.ci90_cluster,2)} |\n"
            )

        lines.append(
            "\nNota interpretativa: a subcoorte **COM_REVERSAO** pode ter CLV maior no **pico** porque, por definição, ela inclui casos em que o edge atingiu um extremo mais alto antes de reverter. "
            "Para entender a hipótese 'sem reversão deveria ser maior', compare também a categoria '**pico no fim, sem reversão**' na partição 100% (tabela 8.1).\n"
        )
        lines.append("\n**Decomposição (pre-match): entry_odd vs closing_odd (IC90)**\n\n")
        lines.append("| Subcoorte | N CLV (PM) | entry_odd t0 (mean; IC90) | entry_odd pico (mean; IC90) | closing_odd (mean; IC90) |\n|---|---:|---:|---:|---:|\n")
        for label, filt in [
            ("SEM_REVERSAO", [r for r in back_entries if (not r.get("had_reversal")) and r.get("is_live") is False]),
            ("COM_REVERSAO", [r for r in back_entries if r.get("had_reversal") and r.get("is_live") is False]),
        ]:
            s_t0 = summarize_metric([r.get("odd_t0") for r in filt], [r.get("match_id") for r in filt], clip_low=1.01, clip_high=200)
            s_ext = summarize_metric([r.get("odd_ext") for r in filt], [r.get("match_id") for r in filt], clip_low=1.01, clip_high=200)
            s_clo = summarize_metric([r.get("closing_odd") for r in filt], [r.get("match_id") for r in filt], clip_low=1.01, clip_high=200)
            lines.append(
                f"| {label} | {s_clo.n_events} | {_fmt_num(s_t0.mean_cluster,3)} {_fmt_ci(s_t0.ci90_cluster,3,suffix='')} | {_fmt_num(s_ext.mean_cluster,3)} {_fmt_ci(s_ext.ci90_cluster,3,suffix='')} | {_fmt_num(s_clo.mean_cluster,3)} {_fmt_ci(s_clo.ci90_cluster,3,suffix='')} |\n"
            )

        # LAY
        lay_stats = _summarize_timing(ok_bs, mode="lay")["rows"]
        lay_agg = _agg_by_regime(lay_stats)
        lines.append("\n### 8.2 Lay (vale em diff_pct)\n")
        lines.append("| Regime | N | t_vale médio (s) | t_vale p50 (s) | % melhora até fim | % com reversão | t_reversão médio (s) | Δt (rev - vale) médio (s) |\n|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for regime in ["PRE_MATCH", "IN_MATCH"]:
            s = lay_agg.get(regime, {"n": 0})
            if s.get("n", 0) == 0:
                lines.append(f"| {regime} | 0 | — | — | — | — | — | — |\n")
                continue
            lines.append(
                f"| {regime} | {s['n']} | {_fmt_num(s.get('t_ext_avg'),1)} | {_fmt_num(s.get('t_ext_p50'),1)} | {_fmt_num(s.get('pct_improve_to_end'),1)}% | {_fmt_num(s.get('pct_reversal'),1)}% | {_fmt_num(s.get('t_rev_avg'),1)} | {_fmt_num(s.get('dt_rev_avg'),1)} |\n"
            )

        lines.append("\n**Partição 100% (Lay)**: categorias exclusivas (somam 100% por regime).\n\n")
        lines.append("| Regime | % vale no fim, sem reversão | % vale no fim, com reversão (recuperou) | % vale antes, com reversão | % vale antes, sem reversão relevante |\n|---|---:|---:|---:|---:|\n")
        for regime in ["PRE_MATCH", "IN_MATCH"]:
            s = lay_agg.get(regime, {"n": 0})
            if s.get("n", 0) == 0:
                lines.append(f"| {regime} | — | — | — | — |\n")
                continue
            lines.append(
                f"| {regime} | {_fmt_num(s.get('pct_end_no_rev'),1)}% | {_fmt_num(s.get('pct_end_with_rev'),1)}% | {_fmt_num(s.get('pct_not_end_with_rev'),1)}% | {_fmt_num(s.get('pct_not_end_no_rev'),1)}% |\n"
            )

        lines.append("\n**Curva média (Lay)**: usa `odd=lay_odd`. `CLV` é reportado **somente pre-match** (closing pré-jogo). O ROI mostrado aqui é **ROI por liability** (não por stake), porque Lay é governado por risco.\n\n")
        lines.append("| Tempo | N pts | diff_pct médio | lay_odd média | CLV médio (pre-match) | ROI/liability médio (se houver) |\n|---|---:|---:|---:|---:|---:|\n")
        for t_label, n, md, mo, mclv, mroi in _curve_table(ok_bs, mode="lay"):
            lines.append(f"| {t_label} | {n} | {_fmt_pct(md,2)} | {_fmt_num(mo,3)} | {_fmt_pct(mclv,2)} | {_fmt_num(mroi,2)} |\n")

        lay_entries = [em for d in ok_bs for em in [_entry_metrics_lay(d)] if em is not None]
        lines.append("\n### 8.2b Lay — impacto por timing (t0 vs vale vs último) e reversão\n")
        lines.append(
            "Leitura: se a estratégia é **entrar no vale (odd mais baixa)**, compare CLV/ROI no vale vs t0 e veja diferença entre **casos com reversão** vs **sem reversão**. "
            "Aqui reportamos **média robusta por jogo + IC90**. Observação: **CLV só é calculado pre-match**.\n\n"
        )

        lines.append("**CLV (somente pre-match) — média robusta por jogo (IC90)**\n\n")
        lines.append("| Subcoorte | N total | N CLV (PM) | CLV t0 (mean; IC90) | CLV vale (mean; IC90) | CLV último (mean; IC90) |\n|---|---:|---:|---:|---:|---:|\n")
        for label, filt in [("SEM_REVERSAO", [r for r in lay_entries if not r["had_reversal"]]), ("COM_REVERSAO", [r for r in lay_entries if r["had_reversal"]])]:
            pm = [r for r in filt if r.get("is_live") is False]
            s0 = _summarize_entry(pm, "clv_t0", clip_low=-50, clip_high=50)
            se = _summarize_entry(pm, "clv_ext", clip_low=-50, clip_high=50)
            sl = _summarize_entry(pm, "clv_last", clip_low=-50, clip_high=50)
            lines.append(
                f"| {label} | {len(filt)} | {s0.n_events} | {_fmt_pct(s0.mean_cluster,2)} {_fmt_ci(s0.ci90_cluster,2)} | {_fmt_pct(se.mean_cluster,2)} {_fmt_ci(se.ci90_cluster,2)} | {_fmt_pct(sl.mean_cluster,2)} {_fmt_ci(sl.ci90_cluster,2)} |\n"
            )

        lines.append("\n**ROI/liability — média robusta por jogo (IC90)**\n\n")
        lines.append("| Subcoorte | N total | N ROI | ROI/liab t0 (mean; IC90) | ROI/liab vale (mean; IC90) | ROI/liab último (mean; IC90) |\n|---|---:|---:|---:|---:|---:|\n")
        for label, filt in [("SEM_REVERSAO", [r for r in lay_entries if not r["had_reversal"]]), ("COM_REVERSAO", [r for r in lay_entries if r["had_reversal"]])]:
            s0 = _summarize_entry(filt, "roi_t0", clip_low=-200, clip_high=5000)
            se = _summarize_entry(filt, "roi_ext", clip_low=-200, clip_high=5000)
            sl = _summarize_entry(filt, "roi_last", clip_low=-200, clip_high=5000)
            lines.append(
                f"| {label} | {len(filt)} | {s0.n_events} | {_fmt_pct(s0.mean_cluster,2)} {_fmt_ci(s0.ci90_cluster,2)} | {_fmt_pct(se.mean_cluster,2)} {_fmt_ci(se.ci90_cluster,2)} | {_fmt_pct(sl.mean_cluster,2)} {_fmt_ci(sl.ci90_cluster,2)} |\n"
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
        # distribuição por data de kickoff (para explicar janela da API free)
        match_meta: Dict[int, Dict[str, Any]] = {}
        for d in all_data:
            mid = int(d.get("match_id"))
            # No dataset usamos a chave "kickoff" (datetime do match).
            kickoff = d.get("kickoff") or d.get("kickoff_time")
            has_score = d.get("home_score") is not None and d.get("away_score") is not None
            if mid not in match_meta:
                match_meta[mid] = {"kickoff": kickoff, "has_score": bool(has_score)}
            else:
                if has_score:
                    match_meta[mid]["has_score"] = True
        kickoff_dates = [m.get("kickoff") for m in match_meta.values() if m.get("kickoff") is not None]
        kickoff_min = min(kickoff_dates) if kickoff_dates else None
        kickoff_max = max(kickoff_dates) if kickoff_dates else None
        by_date: Dict[str, Dict[str, int]] = {}
        for m in match_meta.values():
            ko = m.get("kickoff")
            if not ko:
                continue
            ds = ko.astimezone(timezone.utc).strftime("%Y-%m-%d")
            by_date.setdefault(ds, {"matches": 0, "with_score": 0})
            by_date[ds]["matches"] += 1
            by_date[ds]["with_score"] += 1 if m.get("has_score") else 0

        lines.append("\n### 10.1 Distribuição por data de kickoff (explica janela da API)\n")
        def _fmt_ko(dt: Any) -> str:
            try:
                return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            except Exception:
                return str(dt)

        lines.append(f"- Kickoff (UTC) no recorte: **{_fmt_ko(kickoff_min) if kickoff_min else '—'}** até **{_fmt_ko(kickoff_max) if kickoff_max else '—'}**.\n\n")
        lines.append("| Kickoff date (UTC) | Jogos | Com placar | Cobertura |\n|---|---:|---:|---:|\n")
        for ds in sorted(by_date.keys(), reverse=True)[:14]:
            m = by_date[ds]
            cov = 100.0 * m["with_score"] / m["matches"] if m["matches"] else 0.0
            lines.append(f"| {ds} | {m['matches']} | {m['with_score']} | {_fmt_num(cov,1)}% |\n")

        lines.append(
            "\n**Leitura**: se seu recorte inclui muitos jogos com kickoff antigo, a API-Football **free** pode não retornar fixtures dessa data "
            "(limitação por janela recente). Nesse cenário, mesmo com o job rodando, `placar disponível` ficará baixo para jogos fora da janela.\n\n"
        )
        lines.append("Se `placar disponível` estiver 0 (mesmo para datas recentes), isso geralmente indica que o job de resultados não rodou ou está sem chave válida.  \n")
        lines.append("Sugestão (rodar no servidor): `cd betinasia_bot && python3 -m results.auto_update_results --once` (ou configure o serviço para rodar em loop).\n")
        lines.append("\n---\n")

        # ============================================================
        # 11) Conclusões, riscos e pontos em aberto
        # ============================================================
        lines.append("## 11) Conclusões, riscos e pontos em aberto\n")
        lines.append("- **Execução (CLV)**: use as seções 3/6 para validar qualidade de execução (especialmente pre-match). Se CLV cluster ficar robustamente positivo, há evidência de boa entrada; se ficar negativo, há erosão estrutural.\n")
        lines.append("- **Pre-match vs in-match**: valide que o comportamento de edge/diff e ROI (quando houver placar) difere entre regimes (seção 2.2). Não é seguro misturar regimes para decisão.\n")
        lines.append("- **Lay**: não pode ser decidido por média. Governança tem que usar p95/p99/ES95 de liability (seção 7.2) e combos com risco (seção 9.2). Se p99/ES95 forem altos, a estratégia precisa limite de exposição por janela.\n")
        lines.append("- **Temporal (retenção de edge)**: se a cobertura `temporal/lay_temporal` for baixa, a inferência de retenção fica limitada (seção 8). Quando há cobertura, delta e retenção indicam se o edge “some” rápido.\n")
        lines.append("- **ROI/resultado realizado**: sem placares no banco, ROI fica 0/ausente e a conclusão financeira final não é possível (seção 10). Primeiro garanta o job de resultados.\n")
        lines.append("- **Pontos em aberto típicos**: (i) trazer DOM para a mesma janela, (ii) garantir atualização de placares, (iii) aumentar cobertura temporal e finance no `hypothesis_details`, (iv) definir política de banca para Lay.\n")
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

        pdf_path = None
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

        if args.git_commit:
            # Tenta versionar os artefatos gerados para facilitar download via GitHub.
            try:
                # Descobre root do repo
                repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
                msg = (
                    args.git_message
                    or f"Adiciona relatório b808 ({args.direction}, {','.join(versions)}, lookback={args.lookback_days})"
                )
                paths = [str(out_path)]
                if pdf_path and pdf_path.exists():
                    paths.append(str(pdf_path))
                subprocess.run(["git", "add", "--"] + paths, check=True, cwd=repo_root)
                subprocess.run(["git", "commit", "-m", msg], check=True, cwd=repo_root)
                print(f"[INFO] Artefatos commitados no git: {', '.join(paths)}")
                if args.git_push:
                    subprocess.run(["git", "push"], check=True, cwd=repo_root)
                    print("[INFO] Push concluído.")
            except subprocess.CalledProcessError as e:
                print(f"[WARN] Falha ao commitar/pushar artefatos: {e}")
            except Exception as e:
                print(f"[WARN] Git não disponível ou não é repositório: {e}")
        return 0

    finally:
        await db.close()


if __name__ == "__main__":
    import asyncio

    raise SystemExit(asyncio.run(main()))

