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
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from collections import Counter

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


def cluster_bootstrap_quantile(
    values_by_match: Dict[int, List[float]],
    q: float,
    *,
    n_boot: int = 4000,
    seed: int = 1337,
) -> Optional[float]:
    """
    Quantil bootstrap por cluster (match_id) do estimador "média dos means por jogo".
    Útil para decisões one-sided (ex.: p10/p30/p90).
    """
    try:
        qf = float(q)
    except Exception:
        return None
    qf = min(1.0, max(0.0, qf))
    match_ids = list(values_by_match.keys())
    if not match_ids:
        return None

    per_match_means: Dict[int, float] = {}
    for mid, vals in values_by_match.items():
        if not vals:
            continue
        per_match_means[mid] = float(sum(vals) / len(vals))
    match_ids = list(per_match_means.keys())
    if not match_ids:
        return None
    if len(match_ids) < 2:
        return per_match_means[match_ids[0]]

    rng = random.Random(int(seed))
    boot = []
    for _ in range(int(n_boot)):
        sample = [per_match_means[rng.choice(match_ids)] for _ in range(len(match_ids))]
        boot.append(float(sum(sample) / len(sample)))
    boot_arr = np.asarray(boot, dtype=float)
    return float(np.quantile(boot_arr, qf))


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
    # Auditoria WS-only (sem betslip): série temporal inteiramente via WebSocket.
    # Mantemos como modelo distinto para permitir comparações no relatório.
    if "ws-only" in v.lower() or v.lower().startswith("v5.") and "ws" in v.lower():
        return "WS-only (t0..t+N)"
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


def line_abs(line_str: Any) -> Optional[float]:
    try:
        if line_str is None:
            return None
        x = abs(float(str(line_str).replace(",", ".")))
        if math.isnan(x) or math.isinf(x):
            return None
        return float(x)
    except Exception:
        return None


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
        default="v4.0-api,v5.0-ws-only,v5.1-ws-gate-lay",
        help="Lista de audit_version separada por vírgula (default: v4.0-api,v5.0-ws-only,v5.1-ws-gate-lay)",
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
    parser.add_argument(
        "--only-oos",
        action="store_true",
        help="Gera somente a seção OOS walk-forward (mais rápido). Ignora in-sample e blocos pesados.",
    )
    parser.add_argument("--out", required=True, help="Caminho do markdown de saída (relativo a betinasia_bot/)")
    parser.add_argument(
        "--pdf",
        default=None,
        help="Se definido, renderiza o markdown para PDF (requer reportlab). Ex.: docs/relatorio.pdf",
    )
    parser.add_argument(
        "--exclude-audited-days",
        default=os.getenv("EXCLUDE_AUDITED_DAYS", ""),
        help="CSV de dias UTC (YYYY-MM-DD) a excluir do recorte antes de todas as métricas "
        "(útil para dias com falha operacional que parecem 'sem apostas'). Ex.: 2026-02-17,2026-02-18",
    )
    parser.add_argument(
        "--no-auto-exclude-days",
        action="store_true",
        help="Desliga exclusões automáticas de dias sem dados e dias WS-only sem Lay (qualidade operacional).",
    )
    parser.add_argument("--back-diff-min", type=float, default=2.0, help="Corte de edge Back (default: 2.0)")
    parser.add_argument("--lay-diff-max", type=float, default=-2.0, help="Corte de edge Lay (default: -2.0)")
    # OBS: argparse usa interpolação estilo `%` na help string; por isso `%` precisa ser escapado como `%%`.
    parser.add_argument("--stake-pct-of-limit", type=float, default=0.25, help="Stake fallback (%% do limite), default 0.25")
    parser.add_argument("--stake-cap", type=float, default=0.0, help="Cap opcional para stake fallback (0=sem cap)")
    parser.add_argument(
        "--fx-usdbrl",
        type=float,
        default=5.20,
        help="Taxa de conversão para reportar valores em R$ (default: 5.20).",
    )
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
    # Sizing / oportunidade (Kelly + capacidade)
    parser.add_argument(
        "--kelly-fractions",
        default="0.10,0.25,0.50,1.00",
        help="Frações de Kelly a reportar (CSV). Default: 0.10,0.25,0.50,1.00",
    )
    parser.add_argument(
        "--kelly-bankroll",
        type=float,
        default=None,
        help="Se definido (>0), usa esta banca (em USD/unidade do relatório) como escala do Kelly (em vez de p99 proxy).",
    )
    parser.add_argument(
        "--kelly-back-cap-frac",
        type=float,
        default=float(os.getenv("KELLY_BACK_CAP_FRAC", "0.02")),
        help="Cap por aposta Back como fração da banca de referência do Kelly. Default: 0.02 (2%%).",
    )
    parser.add_argument(
        "--kelly-lay-cap-frac",
        type=float,
        default=float(os.getenv("KELLY_LAY_CAP_FRAC", "0.01")),
        help="Cap por aposta Lay (liability) como fração da banca de referência do Kelly. Default: 0.01 (1%%).",
    )
    parser.add_argument(
        "--max-stake-pct-of-limit",
        type=float,
        default=float(os.getenv("MAX_STAKE_PCT_OF_LIMIT", "1.0")),
        help="Cap por evento via limit (stake máximo como %% do limit). Default: 1.0 (100%%).",
    )
    parser.add_argument(
        "--max-stake-cap",
        type=float,
        default=float(os.getenv("MAX_STAKE_CAP", "0.0")),
        help="Cap absoluto adicional para stake máximo por evento (0=sem cap).",
    )
    parser.add_argument(
        "--walkforward",
        action="store_true",
        help="Habilita estudo OOS rolling-forward (walk-forward) por dia, com seleção de estratégias por IC90/p90.",
    )
    parser.add_argument(
        "--report-mode",
        choices=["full", "oos_first"],
        default=os.getenv("REPORT_MODE", "full").strip() or "full",
        help="Formato do documento. 'full' mantém o relatório completo; 'oos_first' move o bloco OOS para o topo (leitura OOS-first).",
    )
    parser.add_argument("--wf-train-days", type=int, default=int(os.getenv("WF_TRAIN_DAYS", "2")), help="Dias de treino por passo (default 2).")
    parser.add_argument("--wf-test-days", type=int, default=int(os.getenv("WF_TEST_DAYS", "1")), help="Dias de teste OOS por passo (default 1).")
    parser.add_argument(
        "--wf-step-days",
        type=int,
        default=int(os.getenv("WF_STEP_DAYS", "1")),
        help="Avanço do walk-forward em dias (default 1). Para evitar janelas de teste sobrepostas, use `wf_step_days = wf_test_days`.",
    )
    parser.add_argument(
        "--wf-start-date",
        default=os.getenv("WF_START_DATE", "").strip(),
        help="Se definido (YYYY-MM-DD), força o calendário do walk-forward a iniciar a partir desta data (UTC). "
        "Se omitido, o script usa o primeiro dia observado dentro do recorte (`--lookback-days`).",
    )
    parser.add_argument(
        "--wf-bankroll-grid",
        default=os.getenv("WF_BANKROLL_GRID", "").strip(),
        help="CSV de bancas para sensibilidade no OOS mantendo budgets/caps (ex.: '1659,5000,10000,25000'). "
        "Se vazio e `--kelly-bankroll` estiver setado, usa o default: 10k,20k,30k,50k,100k.",
    )
    parser.add_argument(
        "--wf-flat-stake-back",
        type=float,
        default=float(os.getenv("WF_FLAT_STAKE_BACK", "1.0")),
        help="Stake constante quando o scheme do Back for FLAT (em unidades monetárias do relatório). Default=1.0.",
    )
    parser.add_argument(
        "--wf-flat-liab-lay",
        type=float,
        default=float(os.getenv("WF_FLAT_LIAB_LAY", "1.0")),
        help="Liability constante quando o scheme do Lay for FLAT (em unidades monetárias do relatório). Default=1.0.",
    )
    parser.add_argument(
        "--wf-ws-proxy-offset-sec",
        type=float,
        default=float(os.getenv("WF_WS_PROXY_OFFSET_SEC", "5.0")),
        help="Quando houver `ws_series` (ws-only), usa a odd WS neste offset (s) como proxy de BS para entrar no OOS. Default=5.0.",
    )
    parser.add_argument(
        "--wf-ws-proxy-max-gap-sec",
        type=float,
        default=float(os.getenv("WF_WS_PROXY_MAX_GAP_SEC", "2.5")),
        help="Tolerância máxima (s) entre o offset alvo e o ponto WS observado para aceitar o proxy. Default=2.5.",
    )
    parser.add_argument(
        "--wf-exclude-exec-buckets",
        default=os.getenv("WF_EXCLUDE_EXEC_BUCKETS", "").strip(),
        help="CSV de exec_bucket a excluir SOMENTE no OOS (walk-forward), aplicado a Back e Lay. "
        "Ex.: '10-20s' para banir execução ruim sem filtrar o relatório inteiro. "
        "Se você quer banir só Back (recomendado p/ 10-20s), use --wf-exclude-exec-buckets-back.",
    )
    parser.add_argument(
        "--wf-exclude-exec-buckets-back",
        default=os.getenv("WF_EXCLUDE_EXEC_BUCKETS_BACK", "").strip(),
        help="CSV de exec_bucket a excluir SOMENTE no OOS para eventos Back. Ex.: '10-20s'.",
    )
    parser.add_argument(
        "--wf-exclude-exec-buckets-lay",
        default=os.getenv("WF_EXCLUDE_EXEC_BUCKETS_LAY", "").strip(),
        help="CSV de exec_bucket a excluir SOMENTE no OOS para eventos Lay.",
    )
    parser.add_argument(
        "--wf-shrinkage",
        action="store_true",
        default=(os.getenv("WF_SHRINKAGE", "0").strip() in ("1", "true", "True", "yes", "YES")),
        help="Ativa shrinkage (empirical Bayes) para estabilizar ROI do treino por combinação no WF (reduz liga/desliga por ruído).",
    )
    parser.add_argument(
        "--wf-train-mode",
        default=os.getenv("WF_TRAIN_MODE", "rolling"),
        choices=["rolling", "expanding"],
        help="Modo de janela de treino no walk-forward: rolling (últimos N dias) ou expanding (tudo até t-1). Default: rolling.",
    )
    parser.add_argument(
        "--wf-min-matches",
        type=int,
        default=int(os.getenv("WF_MIN_MATCHES", "20")),
        help="Mínimo de jogos por combinação para ser elegível (default 20). Use 0 para desabilitar o mínimo.",
    )
    parser.add_argument(
        "--wf-key-by-league",
        action="store_true",
        default=(os.getenv("WF_KEY_BY_LEAGUE", "0").strip() in ("1", "true", "True", "yes", "YES")),
        help="Se definido, inclui `league` na chave de combinação do OOS (combinação×liga). Isso incorpora liga no modelo de seleção/ativação.",
    )
    parser.add_argument(
        "--wf-key-by-league-scope",
        choices=["pre", "all"],
        default=os.getenv("WF_KEY_BY_LEAGUE_SCOPE", "pre").strip() or "pre",
        help="Escopo de `--wf-key-by-league`: 'pre' inclui liga somente no pre-match (recomendado, menos ruído); "
        "'all' inclui liga também no in-match.",
    )
    parser.add_argument(
        "--wf-liquidity-mode",
        choices=["none", "gate_p50", "gate_p75", "gate_min"],
        default=os.getenv("WF_LIQUIDITY_MODE", "none").strip() or "none",
        help="Tratamento de liquidez (proxy via limit) no OOS. "
        "'gate_p50'/'gate_p75' filtram eventos abaixo do limiar (percentil) estimado na janela de treino (sem lookahead). "
        "'gate_min' usa um mínimo absoluto. Default: none.",
    )
    parser.add_argument(
        "--wf-liquidity-scope",
        choices=["pre", "all"],
        default=os.getenv("WF_LIQUIDITY_SCOPE", "pre").strip() or "pre",
        help="Escopo do filtro de liquidez: 'pre' aplica só no pre-match; 'all' aplica também no in-match. Default: pre.",
    )
    parser.add_argument(
        "--wf-liquidity-min-limit",
        type=float,
        default=float(os.getenv("WF_LIQUIDITY_MIN_LIMIT", "0")),
        help="Mínimo absoluto de limit (USD) para `--wf-liquidity-mode gate_min`. Default 0 (desliga).",
    )
    parser.add_argument(
        "--wf-scheme-pre",
        default=os.getenv("WF_SCHEME_PRE", "KELLY_0.25"),
        help="Scheme de sizing para OOS pre-match (default KELLY_0.25). Ex.: PROXY, FLAT, KELLY_0.10",
    )
    parser.add_argument(
        "--wf-scheme-in",
        default=os.getenv("WF_SCHEME_IN", "FLAT"),
        help="Scheme de sizing para OOS in-match (default FLAT). Ex.: FLAT, PROXY, ROI_TRAIN",
    )
    parser.add_argument(
        "--wf-expand-missing-roi",
        action="store_true",
        default=(os.getenv("WF_EXPAND_MISSING_ROI", "1").strip() not in ("0", "false", "False", "no", "NO")),
        help="Expande lucro observado (com ROI) para população elegível via scaling por exposição/turnover (default=on).",
    )
    parser.add_argument(
        "--wf-match-budget",
        action="store_true",
        default=(os.getenv("WF_MATCH_BUDGET", "1").strip() not in ("0", "false", "False", "no", "NO")),
        help="Inclui simulação de governança por jogo (budget por match_id) na seção OOS (default=on).",
    )
    parser.add_argument("--wf-budget-back-frac", type=float, default=float(os.getenv("WF_BUDGET_BACK_FRAC", "0.01")), help="Budget Back por jogo (fração da banca ref).")
    parser.add_argument("--wf-budget-lay-frac", type=float, default=float(os.getenv("WF_BUDGET_LAY_FRAC", "0.005")), help="Budget Lay por jogo (fração da banca ref, em liability).")
    parser.add_argument("--wf-budget-cap-signal-frac", type=float, default=float(os.getenv("WF_BUDGET_CAP_SIGNAL_FRAC", "0.33")), help="Cap por sinal como fração do budget do jogo.")
    parser.add_argument(
        "--wf-budget-risk-mode",
        default=os.getenv("WF_BUDGET_RISK_MODE", "fixed").strip() or "fixed",
        choices=["fixed", "signals_sqrt", "signals_linear"],
        help="Modo de budget por match_id no OOS: fixed (constante), ou adaptativo por concentração de sinais observada (signals_sqrt/signals_linear).",
    )
    parser.add_argument(
        "--wf-ah-max-abs-line",
        type=float,
        default=float(os.getenv("WF_AH_MAX_ABS_LINE", "0")),
        help="Política OOS por linha AH (proxy de liquidez): se >0, filtra eventos com |line| > este valor "
        "(ex.: 2.0 para excluir AH 2+). Default 0 (sem filtro).",
    )
    parser.add_argument(
        "--wf-ah-scope",
        choices=["pre", "all"],
        default=os.getenv("WF_AH_SCOPE", "all").strip() or "all",
        help="Escopo do filtro por linha AH no OOS: 'pre' aplica só no pre-match; 'all' aplica também no in-match. Default: all.",
    )
    parser.add_argument(
        "--wf-export-policy-json",
        default=os.getenv("WF_EXPORT_POLICY_JSON", "").strip(),
        help="Se definido (path), exporta um JSON com os steps do walk-forward (inclui active_keys/diag) para uso operacional (Decision Engine).",
    )
    args = parser.parse_args()

    # seed global para reprodutibilidade do bootstrap
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    versions = [v.strip() for v in str(args.versions).split(",") if v.strip()]
    out_path = Path(args.out)
    # Robustez: `steps` pode ser referenciado em blocos de transparência do OOS.
    # Inicializamos aqui para evitar `UnboundLocalError` em variações de parâmetros/fluxo.
    steps: List[dict] = []
    active_counts: Dict[str, int] = {}
    # Robustez adicional: séries OOS podem ser referenciadas em seções 12.1+.
    # Em alguns caminhos de execução (ex.: filtros/early-exit), a atribuição poderia não ocorrer.
    daily_turn: Dict[str, float] = {}
    daily_turn_pre: Dict[str, float] = {}
    daily_turn_in: Dict[str, float] = {}
    daily_pnl_obs: Dict[str, float] = {}
    daily_pnl_obs_pre: Dict[str, float] = {}
    daily_pnl_obs_in: Dict[str, float] = {}
    daily_pnl_exp: Dict[str, float] = {}
    daily_pnl_exp_pre: Dict[str, float] = {}
    daily_pnl_exp_in: Dict[str, float] = {}
    oos_back_stakes_all: List[float] = []
    oos_lay_liab_all: List[float] = []
    oos_jobs: List[Tuple[datetime, datetime, float]] = []

    def _liq_p99_from_jobs(jobs: List[Tuple[datetime, datetime, float]]) -> Optional[float]:
        """
        Banca de liquidez (p99) a partir de exposições simultâneas.
        Definido no escopo externo para não depender de blocos condicionais do walk-forward.
        """
        if not jobs:
            return None
        grid_min = int(os.getenv("LIQUIDITY_GRID_MINUTES", "5"))
        buf_pct = float(os.getenv("LIQUIDITY_BANK_BUFFER_PCT", "10"))
        step = max(1, grid_min)
        t_min = min(j[0] for j in jobs)
        t_max = max(j[1] for j in jobs)
        t = t_min
        vals: List[float] = []
        while t <= t_max:
            s = 0.0
            for a, b, exp in jobs:
                if a <= t <= b:
                    s += float(exp)
            vals.append(float(s))
            t = t + timedelta(minutes=step)
        if not vals:
            return None
        p99 = float(np.quantile(vals, 0.99))
        return float(p99) * (1.0 + max(0.0, float(buf_pct)) / 100.0)

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

            # Overhead operacional: total_observado - e2e_instrumentado
            # (usa lag_total_ms, não audit_total_ms; audit_total pode ter janela diferente em alguns regimes/versões)
            if d.get("lag_total_ms") is not None and d.get("lag_e2e_ms") is not None:
                d["lag_overhead_ms"] = float(d["lag_total_ms"]) - float(d["lag_e2e_ms"])
            elif total_ms is not None and total_ms > 0 and d.get("lag_e2e_ms") is not None:
                # fallback
                d["lag_overhead_ms"] = float(total_ms - float(d["lag_e2e_ms"]))
            else:
                d["lag_overhead_ms"] = None

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
        # Exclusão de dias (audited_at UTC) para evitar distorções
        # ------------------------------------------------------------
        def _day_utc(ts: Any) -> Optional[str]:
            if isinstance(ts, datetime):
                try:
                    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d")
                except Exception:
                    return ts.strftime("%Y-%m-%d")
            return None

        # Diagnóstico: calendário bruto observado (antes de exclusões)
        raw_days_obs = sorted({d for d in (_day_utc(x.get("audited_at")) for x in all_data) if d})
        raw_days_missing: List[str] = []
        if raw_days_obs:
            try:
                d0 = datetime.fromisoformat(raw_days_obs[0]).date()
                d1 = datetime.fromisoformat(raw_days_obs[-1]).date()
                cur = d0
                raw_set = set(raw_days_obs)
                while cur <= d1:
                    s = cur.isoformat()
                    if s not in raw_set:
                        raw_days_missing.append(s)
                    cur = cur + timedelta(days=1)
            except Exception:
                raw_days_missing = []

        # Auto-exclusão: dias com falha operacional / regime WS-only sem Lay
        auto_excluded_ws_only_no_lay: List[str] = []
        auto_excluded_unusable_no_bs_ws_lay: List[str] = []
        if not bool(getattr(args, "no_auto_exclude_days", False)):
            def _has_ws_series(dd: dict) -> bool:
                try:
                    h = dd.get("hypothesis_details") or {}
                    ws_series = _get_path(h, ["ws_series"])
                    if isinstance(ws_series, list) and len(ws_series) > 0:
                        return True
                    # ws_gate_lay grava a série curta t0/t+5 em ws_gate_series
                    ws_gate = _get_path(h, ["ws_gate_series"])
                    if isinstance(ws_gate, list) and len(ws_gate) > 0:
                        return True
                    return False
                except Exception:
                    return False

            def _has_lay_any(dd: dict) -> bool:
                try:
                    h = dd.get("hypothesis_details") or {}
                    lay_odd = _safe_float(_get_path(h, ["lay", "odd"]))
                    if lay_odd is not None and float(lay_odd) > 1.0:
                        return True
                    lay_temporal = _get_path(h, ["lay_temporal"])
                    if isinstance(lay_temporal, list) and len(lay_temporal) > 0:
                        return True
                except Exception:
                    return False
                return False

            day_flags: Dict[str, Dict[str, bool]] = {}
            for dd in all_data:
                day = _day_utc(dd.get("audited_at"))
                if not day:
                    continue
                f = day_flags.setdefault(day, {"has_ws": False, "has_bs": False, "has_lay": False})
                if _has_ws_series(dd):
                    f["has_ws"] = True
                if _safe_float(dd.get("bs_odd")) is not None and float(dd.get("bs_odd") or 0) > 0:
                    f["has_bs"] = True
                if _has_lay_any(dd):
                    f["has_lay"] = True

            # Dia "WS-only sem Lay": há ws_series, mas não há bs_odd e não há qualquer Lay.
            auto_excluded_ws_only_no_lay = sorted(
                [day for day, f in day_flags.items() if f.get("has_ws") and (not f.get("has_bs")) and (not f.get("has_lay"))]
            )
            # Dia "inútil" (falha operacional / bloqueio / mudança de schema):
            # existe volume de registros, mas não há BS, nem WS series/gate, nem Lay em nenhum registro do dia.
            auto_excluded_unusable_no_bs_ws_lay = sorted(
                [day for day, f in day_flags.items() if (not f.get("has_ws")) and (not f.get("has_bs")) and (not f.get("has_lay"))]
            )

        # Exclusão manual via CLI/env
        exclude_days_manual = {x.strip() for x in str(getattr(args, "exclude_audited_days", "") or "").split(",") if x.strip()}

        # Conjunto final de dias a excluir do dataset (quando existem registros)
        exclude_days_all = set(exclude_days_manual) | set(auto_excluded_ws_only_no_lay) | set(auto_excluded_unusable_no_bs_ws_lay)
        excluded_days_summary = {
            "manual": sorted(exclude_days_manual),
            "auto_ws_only_no_lay": list(auto_excluded_ws_only_no_lay),
            "auto_unusable_no_bs_ws_lay": list(auto_excluded_unusable_no_bs_ws_lay),
            "missing_no_data": list(raw_days_missing),
        }

        if exclude_days_all:
            before_n = len(all_data)
            all_data = [d for d in all_data if (_day_utc(d.get("audited_at")) not in exclude_days_all)]
            after_n = len(all_data)
            if before_n != after_n:
                print(
                    f"[INFO] Excluídos {before_n - after_n} registros por dias UTC (manual={len(exclude_days_manual)}, "
                    f"auto_ws_only_no_lay={len(auto_excluded_ws_only_no_lay)}, "
                    f"auto_unusable_no_bs_ws_lay={len(auto_excluded_unusable_no_bs_ws_lay)})."
                )

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

        # Diagnóstico de cobertura do closing e distribuição de mercado (após filtros de recorte)
        mt_counts = Counter(str(d.get("market_type") or "NA") for d in all_data)
        ah_rows = [d for d in all_data if str(d.get("market_type")) == "AH"]
        ah_unique_matches = len({int(d["match_id"]) for d in ah_rows}) if ah_rows else 0
        ah_rows_with_closing = [
            d for d in ah_rows if d.get("closing_odd") is not None and float(d["closing_odd"]) > 0
        ]
        ah_unique_matches_with_closing = (
            len({int(d["match_id"]) for d in ah_rows_with_closing}) if ah_rows_with_closing else 0
        )
        ah_closing_coverage_pct = (
            100.0 * ah_unique_matches_with_closing / ah_unique_matches if ah_unique_matches else None
        )

        # Filtro qualidade betslip (igual ao script: -10 a +10)
        with_bs_raw = [d for d in all_data if d.get("bs_odd") and d["bs_odd"] > 0]
        with_bs = [d for d in with_bs_raw if d.get("diff_pct") is not None and -10 <= float(d["diff_pct"]) <= 10]

        unique_matches_all = len(set(d["match_id"] for d in all_data))
        unique_matches_bs = len(set(d["match_id"] for d in with_bs))
        avg_obs_per_match = (len(all_data) / unique_matches_all) if unique_matches_all else 0.0

        # Janela efetiva do recorte (audited_at) — ajuda a validar lookback e explicar N "parecido"
        audited_ts = [d.get("audited_at") for d in all_data if isinstance(d.get("audited_at"), datetime)]
        audited_min = min(audited_ts) if audited_ts else None
        audited_max = max(audited_ts) if audited_ts else None
        audited_unique_days = len({t.date() for t in audited_ts}) if audited_ts else 0
        audited_span_days = None
        if audited_min and audited_max:
            try:
                audited_span_days = max(0.0, float((audited_max - audited_min).total_seconds()) / 86400.0)
            except Exception:
                audited_span_days = None

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

        ok_any = [d for d in all_data if str(d.get("status", "")).upper() == "OK"]
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
        n_ws_series = sum(
            1
            for d in ok_any
            if _is_nonempty_array(_get_path(d.get("hypothesis_details") or {}, ["ws_series"]))
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
        lines.append(
            "| Distribuição por market_type | "
            + ", ".join([f"{k}={mt_counts.get(k,0)}" for k in ["AH", "OU", "1X2", "NA"] if mt_counts.get(k, 0)])
            + " |\n"
        )
        lines.append(f"| Jogos únicos (AH) no recorte | {ah_unique_matches} |\n")
        lines.append(f"| Jogos únicos (AH) com closing_odd disponível | {ah_unique_matches_with_closing} |\n")
        lines.append(f"| Cobertura closing_odd (AH) | {_fmt_num(ah_closing_coverage_pct,1)}% |\n")
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

        # 2.0a) Glossário (métricas-chave)
        lines.append("### 2.0a Glossário de métricas (definições operacionais)\n")
        lines.append(
            "Este glossário existe para eliminar ambiguidades entre **tempo total**, **tempos instrumentados** e **overhead**.\n\n"
        )
        lines.append("- **`hypothesis_detected_at`**: timestamp (UTC) de detecção do evento que gerou a auditoria.\n")
        lines.append("- **`audited_at`**: timestamp (UTC) em que a auditoria foi concluída/persistida.\n")
        lines.append("- **`lag_total_ms` (tempo total observado / wall)**: proxy de tempo “de parede” do pipeline do evento até o betslip; quando disponível usa wall time (ex.: `audited_at - detected_at`).\n")
        lines.append("- **`lag_det_to_click_ms` (detecção→clique)**: tempo até o robô executar o clique/ação de betslip.\n")
        lines.append("- **`lag_click_to_betslip_ms` (clique→betslip)**: tempo até carregar/obter o payload do betslip após o clique.\n")
        lines.append("- **`lag_e2e_ms` (tempo instrumentado)**: `lag_det_to_click_ms + lag_click_to_betslip_ms`.\n")
        lines.append("- **`audit_total_ms` (duração da auditoria)**: duração instrumentada do ciclo de auditoria (pode diferir de `lag_total_ms` se houver esperas fora do escopo instrumentado).\n")
        lines.append("- **`lag_overhead_ms` (overhead)**: `lag_total_ms - lag_e2e_ms`; agrega espera fora das duas etapas instrumentadas (ex.: fila, retries, pausas, latência externa).\n")
        lines.append(
            "- **`diff_pct` (BS vs WS)**: diferença percentual entre a odd do **betslip no momento da execução** (BS) e a odd do **WebSocket no momento da detecção** (WS): "
            "`(BS - WS) / WS * 100`. Importante: **BS e WS são medidos em instantes diferentes**, então este número mede principalmente "
            "**drift durante a execução + slippage/atualização** (e não “mispricing contemporâneo”).\n"
        )
        lines.append("- **Betslip confiável**: filtro de qualidade `diff_pct ∈ [-10%, +10%]` para reduzir casos de mismatch/parse incorreto.\n")
        lines.append("\n---\n")

        # 2.0b) Decomposição de latência (foco: fila/overhead)
        lines.append("### 2.0b Decomposição do tempo (detecção→clique→betslip vs. overhead)\n")
        lines.append(
            "Interpretação: `lag_e2e` é o **tempo fim-a-fim** (detecção→clique + clique→betslip). "
            "`overhead` = `lag_total` − `lag_e2e` (proxy de fila/retries/esperas fora das 2 etapas instrumentadas).\n\n"
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

        # Diagnóstico de cauda (% acima de thresholds) e consistência do overhead
        def _pct_gt(rows_in: List[Dict[str, Any]], key: str, thr_ms: float) -> Optional[float]:
            vals = [_safe_float(d.get(key)) for d in rows_in]
            vals = [float(v) for v in vals if v is not None]
            if not vals:
                return None
            return 100.0 * sum(1 for v in vals if v > thr_ms) / len(vals)

        def _pct_lt(rows_in: List[Dict[str, Any]], key: str, thr_ms: float) -> Optional[float]:
            vals = [_safe_float(d.get(key)) for d in rows_in]
            vals = [float(v) for v in vals if v is not None]
            if not vals:
                return None
            return 100.0 * sum(1 for v in vals if v < thr_ms) / len(vals)

        lines.append("\n**Diagnóstico de cauda (percentual acima do limiar)**\n\n")
        lines.append("| Modelo | % det→click > 5s | % det→click > 20s | % total > 10s | % total > 40s | % overhead < 0 |\n|---|---:|---:|---:|---:|---:|\n")
        for model_name, rows_in in [("API (2-4s)", api_all), ("DOM (15-30s)", dom_all)]:
            p1 = _pct_gt(rows_in, "lag_det_to_click_ms", 5000)
            p2 = _pct_gt(rows_in, "lag_det_to_click_ms", 20000)
            p3 = _pct_gt(rows_in, "lag_total_ms", 10000)
            p4 = _pct_gt(rows_in, "lag_total_ms", 40000)
            p5 = _pct_lt(rows_in, "lag_overhead_ms", 0)
            lines.append(f"| {model_name} | {_fmt_num(p1,1)}% | {_fmt_num(p2,1)}% | {_fmt_num(p3,1)}% | {_fmt_num(p4,1)}% | {_fmt_num(p5,1)}% |\n")
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

        # 2.2c Quebra por liga (top por volume)
        lines.append("### 2.2c Quebra por liga (top por volume)\n")
        lines.append(
            "Objetivo: detectar não-uniformidade do edge por **liga**. "
            "Reporta volume, cobertura de closing (para CLV) e métricas robustas por jogo.\n\n"
        )
        ok_conf = [d for d in with_bs if str(d.get("status", "")).upper() == "OK"]
        if ok_conf:
            league_cnt = Counter(str(d.get("league") or "—") for d in ok_conf)
            top_leagues = [lg for lg, _ in league_cnt.most_common(12)]
            lines.append("| Liga | N OK (conf.) | Jogos | Closing cov (jogos PM) | CLV PM (mean; IC90) | ROI (mean; IC90) | Back edge | Lay edge |\n")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for lg in top_leagues:
                sub = [d for d in ok_conf if str(d.get("league") or "—") == lg]
                mids = {int(d.get("match_id")) for d in sub if d.get("match_id") is not None}
                pm_sub = [d for d in sub if d.get("is_live") is False]
                clv_sum = summarize_metric(
                    [d.get("clv_bs") for d in pm_sub],
                    [d.get("match_id") for d in pm_sub],
                    clip_low=-50,
                    clip_high=50,
                )
                roi_sum = summarize_metric(
                    [d.get("roi_bs") for d in sub],
                    [d.get("match_id") for d in sub],
                    clip_low=-100,
                    clip_high=500,
                )
                pm_mids = {int(d.get("match_id")) for d in pm_sub if d.get("match_id") is not None}
                pm_mids_closing = {
                    int(d.get("match_id"))
                    for d in pm_sub
                    if d.get("match_id") is not None and d.get("closing_odd") is not None and float(d.get("closing_odd") or 0.0) > 0
                }
                cov = (100.0 * len(pm_mids_closing) / len(pm_mids)) if pm_mids else None
                be = sum(1 for d in sub if d.get("diff_pct") is not None and float(d.get("diff_pct")) >= float(back_cut))
                le = sum(1 for d in sub if d.get("diff_pct") is not None and float(d.get("diff_pct")) <= float(lay_cut))
                clv_txt = f"{_fmt_pct(clv_sum.mean_cluster,2)} {_fmt_ci(clv_sum.ci90_cluster,2)}" if clv_sum.n_events else "—"
                roi_txt = f"{_fmt_pct(roi_sum.mean_cluster,2)} {_fmt_ci(roi_sum.ci90_cluster,2)}" if roi_sum.n_events else "—"
                lines.append(f"| {lg} | {len(sub)} | {len(mids)} | {_fmt_num(cov,1)}% | {clv_txt} | {roi_txt} | {be} | {le} |\n")
            lines.append("\n---\n")
        else:
            lines.append("_Sem dados OK+conf suficientes para quebrar por liga._\n\n---\n")

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

            # 2.3b) Mesmo bucket, mas separando coortes por delta BS vs WS (diff buckets)
            lines.append("### 2.3b Regimes por tempo total — separando coortes por `diff_pct` (BS vs WS)\n")
            lines.append(
                "Nesta tabela, separamos duas coortes operacionais por **delta de execução**: "
                "`BS > WS` (diff_pct >= +2%) e `BS < WS` (diff_pct <= -2%). "
                "Isso **não** é (por si só) “Back vs Lay”; é um recorte por **melhora/piora do preço** entre detecção (WS) e execução (BS). "
                "CLV é reportado apenas em pre‑match.\n\n"
            )
            lines.append("| Bucket | N OK | N (BS>WS +2%) | N (BS<WS -2%) | CLV PM (BS>WS) | CLV PM (BS<WS) | ROI (BS>WS) | ROI (BS<WS) |\n|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for b in ["< 5s", "5-10s", "10-20s", "20-40s", "> 40s", "Desconhecido"]:
                sub = [d for d in ok_bs if str(d.get("exec_bucket")) == b]
                if not sub:
                    lines.append(f"| {b} | 0 | 0 | 0 | — | — | — | — |\n")
                    continue

                sub_back = [d for d in sub if int(d["id"]) in back_edge_ids]
                sub_lay = [d for d in sub if int(d["id"]) in lay_edge_ids]

                clv_back = summarize_metric(
                    [d.get("clv_bs") for d in sub_back if d.get("is_live") is False],
                    [d.get("match_id") for d in sub_back if d.get("is_live") is False],
                    clip_low=-50,
                    clip_high=50,
                )
                clv_lay = summarize_metric(
                    [d.get("clv_bs") for d in sub_lay if d.get("is_live") is False],
                    [d.get("match_id") for d in sub_lay if d.get("is_live") is False],
                    clip_low=-50,
                    clip_high=50,
                )
                roi_back = summarize_metric(
                    [d.get("roi_bs") for d in sub_back],
                    [d.get("match_id") for d in sub_back],
                    clip_low=-100,
                    clip_high=500,
                )
                roi_lay = summarize_metric(
                    [d.get("roi_bs") for d in sub_lay],
                    [d.get("match_id") for d in sub_lay],
                    clip_low=-100,
                    clip_high=500,
                )

                clv_back_txt = f"{_fmt_pct(clv_back.mean_cluster,2)} {_fmt_ci(clv_back.ci90_cluster,2)}" if clv_back.n_events else "—"
                clv_lay_txt = f"{_fmt_pct(clv_lay.mean_cluster,2)} {_fmt_ci(clv_lay.ci90_cluster,2)}" if clv_lay.n_events else "—"
                roi_back_txt = f"{_fmt_pct(roi_back.mean_cluster,2)} {_fmt_ci(roi_back.ci90_cluster,2)}" if roi_back.n_events else "—"
                roi_lay_txt = f"{_fmt_pct(roi_lay.mean_cluster,2)} {_fmt_ci(roi_lay.ci90_cluster,2)}" if roi_lay.n_events else "—"
                lines.append(
                    f"| {b} | {len(sub)} | {len(sub_back)} | {len(sub_lay)} | {clv_back_txt} | {clv_lay_txt} | {roi_back_txt} | {roi_lay_txt} |\n"
                )
            lines.append("\n---\n")

            # 2.3c) Estabilidade temporal (time-dependent) — por dia
            lines.append("### 2.3c Estabilidade temporal (por dia, `audited_at`)\n")
            lines.append(
                "Objetivo: checar se o regime de edge/execução é **time‑dependent**. "
                "Se houver dias com comportamento distinto (ex.: collector intermitente, horários/ligas), isso aparecerá aqui.\n\n"
            )
            # Foco na amostra executável (OK + betslip confiável). Por padrão mostramos API; se DOM existir, aparece em outra linha.
            lines.append(
                "| Dia (UTC) | Modelo | N OK | Jogos | % BS>WS +2% | % BS<WS -2% | lag_total p50 (ms) | CLV PM (BS>WS) | CLV PM (BS<WS) |\n"
                "|---|---|---:|---:|---:|---:|---:|---:|---:|\n"
            )

            def _day_key(dt: Any) -> Optional[str]:
                if not isinstance(dt, datetime):
                    return None
                try:
                    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")
                except Exception:
                    return None

            day_keys = sorted({k for k in (_day_key(d.get("audited_at")) for d in ok_bs) if k})
            for ds in day_keys[-14:]:  # último ~14 dias no máximo
                for model_name in ["API (2-4s)", "DOM (15-30s)"]:
                    sub = [d for d in ok_bs if str(d.get("model")) == model_name and _day_key(d.get("audited_at")) == ds]
                    if not sub:
                        continue
                    sub_back = [d for d in sub if int(d["id"]) in back_edge_ids]
                    sub_lay = [d for d in sub if int(d["id"]) in lay_edge_ids]
                    share_back = (100.0 * len(sub_back) / len(sub)) if sub else None
                    share_lay = (100.0 * len(sub_lay) / len(sub)) if sub else None
                    n_matches = len({int(d["match_id"]) for d in sub})
                    lags = [float(v) for v in (_safe_float(d.get("lag_total_ms")) for d in sub) if v is not None and float(v) > 0]
                    lag_p50 = float(np.median(lags)) if lags else None

                    clv_back = summarize_metric(
                        [d.get("clv_bs") for d in sub_back if d.get("is_live") is False],
                        [d.get("match_id") for d in sub_back if d.get("is_live") is False],
                        clip_low=-50,
                        clip_high=50,
                    )
                    clv_lay = summarize_metric(
                        [d.get("clv_bs") for d in sub_lay if d.get("is_live") is False],
                        [d.get("match_id") for d in sub_lay if d.get("is_live") is False],
                        clip_low=-50,
                        clip_high=50,
                    )
                    clv_back_txt = f"{_fmt_pct(clv_back.mean_cluster,2)}" if clv_back.n_events else "—"
                    clv_lay_txt = f"{_fmt_pct(clv_lay.mean_cluster,2)}" if clv_lay.n_events else "—"

                    lines.append(
                        f"| {ds} | {model_name} | {len(sub)} | {n_matches} | {_fmt_num(share_back,1)}% | {_fmt_num(share_lay,1)}% | "
                        f"{_fmt_num(lag_p50,0)} | {clv_back_txt} | {clv_lay_txt} |\n"
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

        # 4.1) Validade do CLV: CLV vs ROI (pre-match)
        lines.append("## 4.1) Validade do CLV: relação CLV × ROI (pre-match)\n")
        lines.append(
            "Objetivo: avaliar se **CLV** (vs closing) é um bom proxy de **ROI realizado** (por placar), ao menos no regime **pre‑match**.\n\n"
        )
        lines.append(
            "Regras do recorte desta seção:\n\n"
            "- Apenas `status=OK` com betslip confiável (diff ∈ [-10%, +10%])\n"
            "- Apenas `PRE_MATCH` (`is_live=False`)\n"
            "- Exige **closing_odd** (para CLV) e **placar** (para ROI)\n\n"
        )

        clv_roi_rows = [
            d
            for d in ok_bs
            if d.get("is_live") is False
            and d.get("clv_bs") is not None
            and d.get("roi_bs") is not None
            and _safe_float(d.get("closing_odd")) is not None
        ]
        bym_clv: Dict[int, List[float]] = {}
        bym_roi: Dict[int, List[float]] = {}
        for d in clv_roi_rows:
            mid = int(d.get("match_id"))
            cv = _safe_float(d.get("clv_bs"))
            rv = _safe_float(d.get("roi_bs"))
            if cv is None or rv is None:
                continue
            if not (-50.0 <= float(cv) <= 50.0):
                continue
            if not (-100.0 <= float(rv) <= 500.0):
                continue
            bym_clv.setdefault(mid, []).append(float(cv))
            bym_roi.setdefault(mid, []).append(float(rv))

        # per-match means (cada jogo pesa 1)
        match_ids = sorted(set(bym_clv.keys()) & set(bym_roi.keys()))
        xs = [float(sum(bym_clv[mid]) / len(bym_clv[mid])) for mid in match_ids if bym_clv.get(mid)]
        ys = [float(sum(bym_roi[mid]) / len(bym_roi[mid])) for mid in match_ids if bym_roi.get(mid)]

        def _pearson(x: List[float], y: List[float]) -> Optional[float]:
            if len(x) < 3 or len(y) < 3:
                return None
            try:
                xa = np.asarray(x, dtype=float)
                ya = np.asarray(y, dtype=float)
                if float(np.std(xa)) <= 1e-12 or float(np.std(ya)) <= 1e-12:
                    return None
                return float(np.corrcoef(xa, ya)[0, 1])
            except Exception:
                return None

        def _rankdata(a: np.ndarray) -> np.ndarray:
            # ranks simples (sem empates perfeitos; empates recebem ranks consecutivos)
            order = np.argsort(a, kind="mergesort")
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, len(a) + 1, dtype=float)
            return ranks

        def _spearman(x: List[float], y: List[float]) -> Optional[float]:
            if len(x) < 3 or len(y) < 3:
                return None
            try:
                xa = np.asarray(x, dtype=float)
                ya = np.asarray(y, dtype=float)
                rx = _rankdata(xa)
                ry = _rankdata(ya)
                return _pearson(rx.tolist(), ry.tolist())
            except Exception:
                return None

        rho_p = _pearson(xs, ys)
        rho_s = _spearman(xs, ys)
        n_matches_clv_roi = len(match_ids)
        n_events_clv_roi = len(clv_roi_rows)

        # matriz de sinais (por jogo)
        pos_pos = sum(1 for x, y in zip(xs, ys) if x > 0 and y > 0)
        pos_neg = sum(1 for x, y in zip(xs, ys) if x > 0 and y <= 0)
        neg_pos = sum(1 for x, y in zip(xs, ys) if x <= 0 and y > 0)
        neg_neg = sum(1 for x, y in zip(xs, ys) if x <= 0 and y <= 0)

        lines.append("### 4.1a Estatística global (por jogo)\n")
        lines.append("| Métrica | Valor |\n|---|---:|\n")
        lines.append(f"| Jogos com CLV+ROI | {n_matches_clv_roi} |\n")
        lines.append(f"| Eventos (auditorias) usados | {n_events_clv_roi} |\n")
        lines.append(f"| Correlação Pearson (mean por jogo) | {_fmt_num(rho_p, 3)} |\n")
        lines.append(f"| Correlação Spearman (mean por jogo) | {_fmt_num(rho_s, 3)} |\n")
        lines.append("\n")

        lines.append("### 4.1b Concordância de sinal (CLV vs ROI)\n")
        lines.append("| CLV (jogo) | ROI (jogo) | Jogos |\n|---|---|---:|\n")
        lines.append(f"| > 0 | > 0 | {pos_pos} |\n")
        lines.append(f"| > 0 | ≤ 0 | {pos_neg} |\n")
        lines.append(f"| ≤ 0 | > 0 | {neg_pos} |\n")
        lines.append(f"| ≤ 0 | ≤ 0 | {neg_neg} |\n")
        lines.append("\n")
        lines.append(
            "Leitura: CLV e ROI podem divergir por **variância do resultado** (ROI) e por **missingness** (jogos sem closing/sem placar). "
            "A correlação acima é um diagnóstico de “alinhamento”, não causalidade.\n\n"
        )

        # bucket por quantis de CLV (por jogo)
        lines.append("### 4.1c ROI por bucket de CLV (quintis; por jogo)\n")
        if n_matches_clv_roi >= 10:
            clv_means = np.asarray(xs, dtype=float)
            q_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            cuts = [float(np.quantile(clv_means, q)) for q in q_edges]
            # garante cortes estritamente crescentes para evitar buckets vazios por empate
            cuts = [cuts[0]] + [max(cuts[i], cuts[i - 1] + 1e-12) for i in range(1, len(cuts))]

            def _bucket_idx(v: float) -> int:
                for i in range(5):
                    if v < cuts[i + 1]:
                        return i
                return 4

            buckets: List[List[int]] = [[] for _ in range(5)]
            for mid in match_ids:
                mclv = float(sum(bym_clv[mid]) / len(bym_clv[mid]))
                buckets[_bucket_idx(mclv)].append(int(mid))

            lines.append("| Bucket (CLV por jogo) | Jogos | CLV mean (IC90) | ROI mean (IC90) | Win rate ROI |\n|---|---:|---:|---:|---:|\n")
            for i, mids in enumerate(buckets):
                if not mids:
                    continue
                b_clv = {mid: bym_clv[mid] for mid in mids if mid in bym_clv}
                b_roi = {mid: bym_roi[mid] for mid in mids if mid in bym_roi}
                clv_mean, clv_ci = cluster_bootstrap_ci(b_clv, n_boot=2000, alpha=0.10, seed=int(args.seed))
                roi_mean, roi_ci = cluster_bootstrap_ci(b_roi, n_boot=2000, alpha=0.10, seed=int(args.seed))
                # win rate por jogo (média ROI do jogo >0)
                wr = None
                try:
                    roi_game = [float(sum(bym_roi[mid]) / len(bym_roi[mid])) for mid in mids if bym_roi.get(mid)]
                    wr = 100.0 * sum(1 for v in roi_game if v > 0) / len(roi_game) if roi_game else None
                except Exception:
                    wr = None
                label = f"Q{i+1} ({_fmt_pct(cuts[i],2)}→{_fmt_pct(cuts[i+1],2)})"
                lines.append(
                    f"| {label} | {len(mids)} | {_fmt_pct(clv_mean)} {_fmt_ci(clv_ci)} | {_fmt_pct(roi_mean)} {_fmt_ci(roi_ci)} | {_fmt_num(wr,1)}% |\n"
                )
            lines.append("\n")
        else:
            lines.append("Amostra insuficiente (jogos com CLV+ROI < 10) para buckets estáveis.\n\n")

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
        lines.append(
            "Nota de leitura:\n"
            "- `BS > WS (+2% a +10%)`: preço no betslip ficou **melhor** do que o WS observado na detecção (delta positivo entre instantes).\n"
            "- `BS < WS (-10% a -2%)`: preço no betslip ficou **pior** do que o WS observado na detecção (delta negativo entre instantes).\n"
            "- O ROI abaixo é calculado **dentro do bucket** (não mistura buckets), mas ainda é sensível à cobertura de placar.\n\n"
        )
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
        if audited_min and audited_max:
            span_label = f"{audited_span_days:.1f}d" if audited_span_days is not None else "—"
            summary_lines.append(
                f"- **Janela efetiva (audited_at)**: {audited_min.strftime('%d/%m %H:%M')} → {audited_max.strftime('%d/%m %H:%M')} UTC "
                f"(span≈{span_label}; dias com dados={audited_unique_days}).\n"
            )
            if args.lookback_days and audited_span_days is not None and audited_span_days < (0.70 * float(args.lookback_days)):
                summary_lines.append(
                    f"- **Alerta**: lookback_days={args.lookback_days}, mas a janela efetiva observada foi menor (span≈{span_label}). "
                    "Isso costuma indicar falta de auditorias antigas para essas `audit_version` (ou recorte por regime/qualidade).\n"
                )
        # Dias excluídos/missing (qualidade operacional)
        try:
            man = list((excluded_days_summary or {}).get("manual") or [])
            auto_nl = list((excluded_days_summary or {}).get("auto_ws_only_no_lay") or [])
            auto_unusable = list((excluded_days_summary or {}).get("auto_unusable_no_bs_ws_lay") or [])
            miss = list((excluded_days_summary or {}).get("missing_no_data") or [])
            if man or auto_nl or auto_unusable or miss:
                def _fmt_days(xs: List[str], max_show: int = 8) -> str:
                    xs = [str(x) for x in xs if str(x)]
                    if not xs:
                        return "—"
                    if len(xs) <= max_show:
                        return ", ".join(xs)
                    return ", ".join(xs[:max_show]) + f" ... (+{len(xs) - max_show})"

                summary_lines.append(
                    "- **Dias excluídos / missing** (UTC, não tratados como 0): "
                    f"manual={len(man)} [{_fmt_days(man)}]; "
                    f"auto(ws-only sem Lay)={len(auto_nl)} [{_fmt_days(auto_nl)}]; "
                    f"auto(sem BS/WS/Lay)={len(auto_unusable)} [{_fmt_days(auto_unusable)}]; "
                    f"missing(sem dados)={len(miss)} [{_fmt_days(miss)}].\n"
                )
        except Exception:
            pass
        summary_lines.append(
            f"- **Coortes (status=OK, betslip confiável)**: `BS>WS` (diff>={back_cut:.1f}%): **{len(back_edge)}**; `BS<WS` (diff<={lay_cut:.1f}%): **{len(lay_edge)}**.\n"
        )
        summary_lines.append(
            f"- **Coberturas em `hypothesis_details` (OK)**: temporal(BS)={n_temporal}/{len(ok_bs)}; lay_temporal(BS)={n_lay_temporal}/{len(ok_bs)}; "
            f"ws_series(WS)={n_ws_series}/{len(ok_any)}; finance={n_finance}/{len(ok_bs)}.\n"
        )
        summary_lines.append(
            f"- **Cobertura de placar (ROI)**: jogos com placar={matches_with_scores}/{unique_matches_all} (status finished={matches_finished_flag}).\n"
        )
        if ah_unique_matches:
            summary_lines.append(
                f"- **Cobertura de closing_odd (AH)**: jogos com closing={ah_unique_matches_with_closing}/{ah_unique_matches} "
                f"({_fmt_num(ah_closing_coverage_pct,1)}%). CLV pre‑match depende disso.\n"
            )
            if ah_closing_coverage_pct is not None and ah_closing_coverage_pct < 80.0:
                summary_lines.append(
                    "- **Alerta**: cobertura de closing_odd está baixa. Isso tende a reduzir N de CLV e pode enviesar a leitura "
                    "(os jogos “sem odds” ficam fora do CLV). Priorize estabilizar o collector e/ou condicionar análises a jogos com closing.\n"
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

        lines.append("**Política de stake (proxy)**\n\n")
        lines.append("| Parâmetro | Valor |\n|---|---:|\n")
        lines.append(f"| stake_pct_of_limit | {stake_pct:.2f} |\n")
        lines.append(f"| stake_cap | {stake_cap:.2f} |\n")
        lines.append(f"| Cobertura finance (OK, betslip conf.) | {n_finance}/{len(ok_bs)} |\n")
        lines.append("\n")

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

        # ------------------------------------------------------------
        # Política inteligente de sizing (para P&L/ROI "realizado" no in-sample)
        # ------------------------------------------------------------
        # Reusa o mesmo par de schemes do WF, para consistência entre seções.
        ins_scheme_pre = str(getattr(args, "wf_scheme_pre", "KELLY_0.25") or "KELLY_0.25").strip()
        ins_scheme_in = str(getattr(args, "wf_scheme_in", "FLAT") or "FLAT").strip()
        ins_bank_ref = _safe_float(getattr(args, "kelly_bankroll", None))
        if ins_bank_ref is None or float(ins_bank_ref) <= 0:
            # fallback: não ideal, mas evita dividir por zero; em produção o daily sempre passa kelly_bankroll
            ins_bank_ref = 10000.0
        # caps conservadores (alinhados ao bloco 9.3)
        INS_BACK_CAP_FRAC = max(0.0, float(getattr(args, "kelly_back_cap_frac", 0.02)))
        INS_LAY_CAP_FRAC = max(0.0, float(getattr(args, "kelly_lay_cap_frac", 0.01)))

        def _ins_scheme_for_row(d: dict) -> str:
            return ins_scheme_in if d.get("is_live") is True else ins_scheme_pre

        def _kelly_back_frac_simple(entry_odd: Any, closing_odd: Any) -> Optional[float]:
            o = _safe_float(entry_odd)
            c = _safe_float(closing_odd)
            if o is None or c is None or o <= 1.0 or c <= 1.0:
                return None
            p = 1.0 / c
            b = o - 1.0
            ev = (o * p) - 1.0
            f = ev / b
            return float(f)

        def _kelly_lay_liab_frac_simple(entry_lay_odd: Any, closing_odd: Any) -> Optional[float]:
            o = _safe_float(entry_lay_odd)
            c = _safe_float(closing_odd)
            if o is None or c is None or o <= 1.0 or c <= 1.0:
                return None
            p = 1.0 / c
            f = 1.0 - (p * o)
            return float(f)

        def _ins_sizing_back(d: dict) -> Optional[float]:
            sc = _ins_scheme_for_row(d)
            if sc == "FLAT":
                return float(os.getenv("WF_FLAT_STAKE_BACK", "1.0"))
            if sc == "PROXY":
                bs, _, _, _ = finance_for_row(d)
                return float(bs)
            if sc.startswith("KELLY"):
                if d.get("is_live") is True:
                    return None
                try:
                    frac = float(sc.split("_")[1])
                except Exception:
                    return None
                f0 = _kelly_back_frac_simple(d.get("bs_odd"), d.get("closing_odd"))
                if f0 is None:
                    # fallback: sem closing_odd não dá Kelly; cai para PROXY/FLAT para não “zerar” policy.
                    bs, _, _, _ = finance_for_row(d)
                    if bs is not None and float(bs) > 0:
                        return float(bs)
                    return float(os.getenv("WF_FLAT_STAKE_BACK", "1.0"))
                f = max(0.0, float(f0)) * float(frac)
                cap = INS_BACK_CAP_FRAC * float(ins_bank_ref)
                return float(min(f * float(ins_bank_ref), cap))
            return None

        def _ins_sizing_lay_liab(d: dict) -> Optional[Tuple[float, float]]:
            sc = _ins_scheme_for_row(d)
            h = d.get("hypothesis_details") or {}
            lay_odd = _safe_float(_get_path(h, ["lay", "odd"]))
            if lay_odd is None or lay_odd <= 0:
                lay_odd = _safe_float(d.get("bs_odd"))
            if lay_odd is None or lay_odd <= 1.0:
                return None
            if sc == "FLAT":
                return (float(os.getenv("WF_FLAT_LIAB_LAY", "1.0")), float(lay_odd))
            if sc == "PROXY":
                _, _, _, ll = finance_for_row(d)
                return (float(ll), float(lay_odd))
            if sc.startswith("KELLY"):
                if d.get("is_live") is True:
                    return None
                try:
                    frac = float(sc.split("_")[1])
                except Exception:
                    return None
                f0 = _kelly_lay_liab_frac_simple(lay_odd, d.get("closing_odd"))
                if f0 is None:
                    # fallback: sem closing_odd não dá Kelly; cai para PROXY/FLAT.
                    _, _, _, ll = finance_for_row(d)
                    if ll is not None and float(ll) > 0:
                        return (float(ll), float(lay_odd))
                    return (float(os.getenv("WF_FLAT_LIAB_LAY", "1.0")), float(lay_odd))
                f = max(0.0, float(f0)) * float(frac)
                cap = INS_LAY_CAP_FRAC * float(ins_bank_ref)
                return (float(min(f * float(ins_bank_ref), cap)), float(lay_odd))
            return None

        def _roi_lay_pct_per_liability_policy(lay_odd: float, mult_back: float) -> Optional[float]:
            """
            ROI por liability para Lay (mesma convenção do relatório), usada aqui no bloco 7.x
            antes do helper equivalente ser definido mais abaixo no arquivo.
            """
            liab = max(0.0, float(lay_odd) - 1.0)
            if liab <= 0:
                return None
            if float(mult_back) < 0:
                return (-float(mult_back)) / liab * 100.0
            if float(mult_back) > 0:
                return (-float(mult_back)) * 100.0
            return 0.0

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
        back_finance_cov = sum(
            1 for d in back_edge if isinstance(_get_path(d.get("hypothesis_details") or {}, ["finance"]), dict)
        )
        lines.append(f"| Cobertura finance (na coorte) | {back_finance_cov}/{len(back_edge)} |\n")
        lines.append(f"| Stake total (estimado) | {_fmt_num(sum(back_stakes) if back_stakes else None, 2)} |\n")
        lines.append(f"| Stake médio | {_fmt_num(_mean(back_stakes), 2)} |\n")
        lines.append(f"| Profit_if_win total (estimado) | {_fmt_num(sum(back_profit_if_win) if back_profit_if_win else None, 2)} |\n")
        lines.append(f"| Profit_if_win médio | {_fmt_num(_mean(back_profit_if_win), 2)} |\n")

        # ROI realizado (quando houver placar)
        back_realized = []
        back_realized_stakes = []
        back_realized_roi = []
        back_realized_mids = []
        back_realized_policy = []
        back_realized_policy_stakes = []
        for d in back_edge:
            roi = _safe_float(d.get("roi_bs"))
            if roi is None:
                continue
            bs, _, _, _ = finance_for_row(d)
            back_realized.append(float(bs) * float(roi) / 100.0)
            back_realized_stakes.append(float(bs))
            back_realized_roi.append(float(roi))
            back_realized_mids.append(int(d.get("match_id")))
            st_pol = _ins_sizing_back(d)
            if st_pol is not None and float(st_pol) > 0:
                back_realized_policy.append(float(st_pol) * float(roi) / 100.0)
                back_realized_policy_stakes.append(float(st_pol))
        if back_realized:
            roi_weighted = (sum(back_realized) / sum(back_realized_stakes) * 100.0) if sum(back_realized_stakes) > 0 else None
            lines.append(f"| N com ROI realizado | {len(back_realized)} |\n")
            # Política inteligente (principal)
            if back_realized_policy and sum(back_realized_policy_stakes) > 0:
                roi_pol = (sum(back_realized_policy) / sum(back_realized_policy_stakes) * 100.0)
                lines.append(f"| P&L realizado total (**policy sizing**: Pre={ins_scheme_pre}, In={ins_scheme_in}, bank_ref={_fmt_num(ins_bank_ref,0)}) | {_fmt_num(sum(back_realized_policy), 2)} |\n")
                lines.append(f"| ROI realizado (**policy**, ponderado por stake) | {_fmt_num(roi_pol, 2)}% |\n")
            else:
                lines.append(f"| P&L realizado total (**policy sizing**) | — |\n")
                lines.append(f"| ROI realizado (**policy**, ponderado por stake) | — |\n")
            # Proxy/legado (comparativo)
            lines.append(f"| P&L realizado total (proxy finance/limit) | {_fmt_num(sum(back_realized), 2)} |\n")
            lines.append(f"| ROI realizado (proxy, ponderado por stake) | {_fmt_num(roi_weighted, 2)}% |\n")

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
            "\n**Como ler as 3 linhas de ROI (Back)**\n\n"
            "- **ROI realizado (ponderado por stake)**: \u03a3P&L / \u03a3stake (pode ser dominado por stakes grandes).\n"
            "- **ROI realizado (robusto por jogo)**: média por jogo (cada jogo pesa parecido; reduz dominância de outliers).\n"
            "- **ROI ponderado por stake (robusto por jogo)**: calcula ROI ponderado dentro do jogo e depois faz média/IC por jogo.\n"
        )
        lines.append(
            "\nObservação: a Seção 4 reporta ROI médio no **recorte completo** (betslip confiável). "
            "Aqui (7.1) é apenas a coorte **Back edge** (diff>=corte) e o ROI também é mostrado ponderado por stake; "
            "por isso sinais podem divergir.\n"
        )

        lines.append("\n### 7.2 Lay (BS << WS) — risco de cauda\n")
        lines.append("| Métrica | Valor |\n|---|---:|\n")
        lines.append(f"| Corte (diff_pct) | <= {lay_cut:.1f}% |\n")
        lines.append(f"| N eventos | {len(lay_edge)} |\n")
        lay_finance_cov = sum(
            1 for d in lay_edge if isinstance(_get_path(d.get("hypothesis_details") or {}, ["finance"]), dict)
        )
        lines.append(f"| Cobertura finance (na coorte) | {lay_finance_cov}/{len(lay_edge)} |\n")
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
        lay_realized_policy_pnl = []
        lay_realized_policy_liab = []
        lay_realized_policy_stake = []
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
            sized_pol = _ins_sizing_lay_liab(d)
            if sized_pol and sized_pol[0] is not None and float(sized_pol[0]) > 0:
                liab_pol, odd_pol = sized_pol
                roi_liab_pol = _roi_lay_pct_per_liability_policy(float(odd_pol), float(mult))
                if roi_liab_pol is not None:
                    pnl_pol = float(liab_pol) * float(roi_liab_pol) / 100.0
                    lay_realized_policy_pnl.append(float(pnl_pol))
                    lay_realized_policy_liab.append(float(liab_pol))
                    st_eq = float(liab_pol) / max(1e-9, (float(odd_pol) - 1.0))
                    lay_realized_policy_stake.append(float(st_eq))

        if lay_realized_pnl:
            roi_liab_weighted = (sum(lay_realized_pnl) / sum(lay_realized_liab) * 100.0) if sum(lay_realized_liab) > 0 else None
            roi_stake_weighted = (sum(lay_realized_pnl) / sum(lay_realized_stake) * 100.0) if sum(lay_realized_stake) > 0 else None
            lines.append(f"| N com ROI realizado | {len(lay_realized_pnl)} |\n")
            # Política inteligente (principal)
            if lay_realized_policy_pnl and sum(lay_realized_policy_liab) > 0:
                roi_liab_pol = (sum(lay_realized_policy_pnl) / sum(lay_realized_policy_liab) * 100.0)
                roi_st_pol = (sum(lay_realized_policy_pnl) / sum(lay_realized_policy_stake) * 100.0) if sum(lay_realized_policy_stake) > 0 else None
                lines.append(f"| P&L realizado total (**policy sizing**: Pre={ins_scheme_pre}, In={ins_scheme_in}, bank_ref={_fmt_num(ins_bank_ref,0)}) | {_fmt_num(sum(lay_realized_policy_pnl), 2)} |\n")
                lines.append(f"| ROI realizado (**policy**, ponderado por liability) | {_fmt_num(roi_liab_pol, 2)}% |\n")
                lines.append(f"| ROI realizado (**policy**, ponderado por stake eq.) | {_fmt_num(roi_st_pol, 2)}% |\n")
            else:
                lines.append(f"| P&L realizado total (**policy sizing**) | — |\n")
                lines.append(f"| ROI realizado (**policy**, ponderado por liability) | — |\n")
                lines.append(f"| ROI realizado (**policy**, ponderado por stake eq.) | — |\n")
            # Proxy/legado (comparativo)
            lines.append(f"| P&L realizado total (proxy finance/limit) | {_fmt_num(sum(lay_realized_pnl), 2)} |\n")
            lines.append(f"| ROI realizado (proxy, ponderado por liability) | {_fmt_num(roi_liab_weighted, 2)}% |\n")
            lines.append(f"| ROI realizado (proxy, ponderado por stake) | {_fmt_num(roi_stake_weighted, 2)}% |\n")

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

        # ============================================================
        # 7.3) Projeção mensal (30 dias fixo)
        # ============================================================
        lines.append("\n### 7.3 Projeção mensal (30 dias fixo) — turnover, lucro e banca\n")
        lines.append(
            "Premissas:\n"
            "- **Mês = 30 dias fixo**.\n"
            "- Turnover = soma de stakes (Back e Lay). Para Lay, também reportamos exposição por **liability**.\n"
            "- **Banca por risco (unitária)** = p99/ES95(exposição por aposta).\n"
            "- **Banca por liquidez (turnover)** = p95/p99 de capital **simultaneamente travado** (stake/liability em aberto até a liquidação), "
            "capturando distribuição desigual de entradas ao longo do tempo.\n"
            "- Projeção de lucro usa duas visões: (i) **lucro/dia observado** e (ii) **ROI observado × turnover projetado**.\n\n"
        )

        def _days_for_rate() -> float:
            # IMPORTANT: usar dias com dados (após exclusões) — não use lookback_days,
            # pois dias "zerados" por falha operacional NÃO devem entrar como 0.
            if audited_unique_days and int(audited_unique_days) > 0:
                return float(audited_unique_days)
            if audited_span_days is not None and audited_span_days > 0:
                return float(audited_span_days)
            return 1.0

        window_days = _days_for_rate()
        horizon_days = 30.0

        def _proj_turnover(total_turnover: float) -> float:
            return (float(total_turnover) / max(1e-9, window_days)) * horizon_days

        def _roi_pct(pnl: float, turnover: float) -> Optional[float]:
            if turnover <= 0:
                return None
            return (float(pnl) / float(turnover)) * 100.0

        def _proj_profit_from_roi(turnover_30d: float, roi_pct: Optional[float]) -> Optional[float]:
            if roi_pct is None:
                return None
            return float(turnover_30d) * float(roi_pct) / 100.0

        # Back (realizado)
        back_turnover_total = float(sum(back_stakes) if back_stakes else 0.0)
        back_turnover_realized = float(sum(back_realized_stakes) if back_realized_stakes else 0.0)
        back_pnl_realized = float(sum(back_realized) if back_realized else 0.0)
        back_roi_realized = _roi_pct(back_pnl_realized, back_turnover_realized) if back_realized else None

        # Lay (realizado)
        lay_turnover_total = float(sum(lay_stakes) if lay_stakes else 0.0)
        lay_liab_total = float(sum(lay_liability) if lay_liability else 0.0)
        lay_turnover_realized = float(sum(lay_realized_stake) if lay_realized_stake else 0.0)
        lay_liab_realized = float(sum(lay_realized_liab) if lay_realized_liab else 0.0)
        lay_pnl_realized = float(sum(lay_realized_pnl) if lay_realized_pnl else 0.0)
        lay_roi_realized_stake = _roi_pct(lay_pnl_realized, lay_turnover_realized) if lay_realized_pnl else None
        lay_roi_realized_liab = _roi_pct(lay_pnl_realized, lay_liab_realized) if lay_realized_pnl else None

        # Bancas (exposição unitária)
        back_bank_conservative = _pctl(back_stakes, 99) if back_stakes else None
        back_bank_aggressive = _es_tail(back_stakes, 95) if back_stakes else None
        lay_bank_conservative = _pctl(lay_liability, 99) if lay_liability else None
        lay_bank_aggressive = _es_tail(lay_liability, 95) if lay_liability else None

        # Projeções 30d
        back_turnover_30d = _proj_turnover(back_turnover_total)
        lay_turnover_30d = _proj_turnover(lay_turnover_total)
        lay_liab_30d = _proj_turnover(lay_liab_total)

        back_profit_30d_direct = (back_pnl_realized / max(1e-9, window_days)) * horizon_days if back_realized else None
        lay_profit_30d_direct = (lay_pnl_realized / max(1e-9, window_days)) * horizon_days if lay_realized_pnl else None
        back_profit_30d_roi = _proj_profit_from_roi(back_turnover_30d, back_roi_realized) if back_realized else None
        lay_profit_30d_roi = _proj_profit_from_roi(lay_turnover_30d, lay_roi_realized_stake) if lay_realized_pnl else None

        total_turnover_30d = back_turnover_30d + lay_turnover_30d
        total_profit_30d_direct = None
        if back_profit_30d_direct is not None or lay_profit_30d_direct is not None:
            total_profit_30d_direct = float(back_profit_30d_direct or 0.0) + float(lay_profit_30d_direct or 0.0)
        total_profit_30d_roi = None
        if back_profit_30d_roi is not None or lay_profit_30d_roi is not None:
            total_profit_30d_roi = float(back_profit_30d_roi or 0.0) + float(lay_profit_30d_roi or 0.0)

        total_bank_conservative = None
        if back_bank_conservative is not None or lay_bank_conservative is not None:
            total_bank_conservative = float(back_bank_conservative or 0.0) + float(lay_bank_conservative or 0.0)
        total_bank_aggressive = None
        if back_bank_aggressive is not None or lay_bank_aggressive is not None:
            total_bank_aggressive = float(back_bank_aggressive or 0.0) + float(lay_bank_aggressive or 0.0)

        # ------------------------------------------------------------
        # Banca por liquidez (capital travado ao longo do tempo)
        # ------------------------------------------------------------
        LIQUIDITY_SETTLE_BUFFER_HOURS = float(os.getenv("LIQUIDITY_SETTLE_BUFFER_HOURS", "2.25"))
        # Aproximação de tempo até liquidação: duração média do jogo + buffer operacional.
        # Isso reduz subestimação de liquidez (e “giro de banca” irreal).
        LIQUIDITY_MATCH_DURATION_HOURS = float(os.getenv("LIQUIDITY_MATCH_DURATION_HOURS", "2.0"))
        LIQUIDITY_GRID_MINUTES = int(os.getenv("LIQUIDITY_GRID_MINUTES", "5"))
        LIQUIDITY_BANK_BUFFER_PCT = float(os.getenv("LIQUIDITY_BANK_BUFFER_PCT", "10"))  # margem extra

        def _exposure_jobs(rows_in: List[dict], mode: str) -> List[Tuple[datetime, datetime, float]]:
            """
            mode='back': exposure=stake
            mode='lay' : exposure=liability
            """
            jobs: List[Tuple[datetime, datetime, float]] = []
            for d in rows_in:
                t0 = d.get("audited_at") or d.get("hypothesis_detected_at")
                if not isinstance(t0, datetime):
                    continue
                ko = d.get("kickoff") or d.get("kickoff_time")
                if isinstance(ko, datetime):
                    t1 = ko + timedelta(hours=(LIQUIDITY_MATCH_DURATION_HOURS + LIQUIDITY_SETTLE_BUFFER_HOURS))
                else:
                    t1 = t0 + timedelta(hours=(LIQUIDITY_MATCH_DURATION_HOURS + LIQUIDITY_SETTLE_BUFFER_HOURS))
                if not isinstance(t1, datetime) or t1 <= t0:
                    t1 = t0 + timedelta(hours=(LIQUIDITY_MATCH_DURATION_HOURS + LIQUIDITY_SETTLE_BUFFER_HOURS))

                bs, _, _, ll = finance_for_row(d)
                exp = float(bs) if mode == "back" else float(ll)
                if exp <= 0:
                    continue
                jobs.append((t0, t1, exp))
            return jobs

        def _liquidity_pctl(jobs: List[Tuple[datetime, datetime, float]]) -> Dict[str, Optional[float]]:
            if not jobs:
                return {"mean": None, "p50": None, "p95": None, "p99": None, "max": None, "n_grid": 0}
            t_min = min(t0 for t0, _, _ in jobs)
            t_max = max(t1 for _, t1, _ in jobs)
            if t_max <= t_min:
                return {"mean": None, "p50": None, "p95": None, "p99": None, "max": None, "n_grid": 0}

            step = max(1, int(LIQUIDITY_GRID_MINUTES))
            # grid alinhado por minutos (para estabilidade)
            t = t_min
            vals: List[float] = []
            while t <= t_max:
                s = 0.0
                for a, b, exp in jobs:
                    if a <= t < b:
                        s += float(exp)
                vals.append(float(s))
                t = t + timedelta(minutes=step)
            if not vals:
                return {"mean": None, "p50": None, "p95": None, "p99": None, "max": None, "n_grid": 0}
            return {
                "mean": float(np.mean(vals)),
                "p50": float(np.quantile(vals, 0.50)),
                "p95": float(np.quantile(vals, 0.95)),
                "p99": float(np.quantile(vals, 0.99)),
                "max": float(max(vals)),
                "n_grid": int(len(vals)),
            }

        back_jobs = _exposure_jobs([d for d in back_edge], mode="back")
        lay_jobs = _exposure_jobs([d for d in lay_edge], mode="lay")
        total_jobs = back_jobs + lay_jobs

        back_liq = _liquidity_pctl(back_jobs)
        lay_liq = _liquidity_pctl(lay_jobs)
        total_liq = _liquidity_pctl(total_jobs)

        def _with_buffer(x: Optional[float]) -> Optional[float]:
            if x is None:
                return None
            return float(x) * (1.0 + max(0.0, LIQUIDITY_BANK_BUFFER_PCT) / 100.0)

        total_bank_liq_p99 = _with_buffer(total_liq.get("p99"))
        total_bank_eff_conservative = None
        if total_bank_conservative is not None or total_bank_liq_p99 is not None:
            total_bank_eff_conservative = max(float(total_bank_conservative or 0.0), float(total_bank_liq_p99 or 0.0))

        def _roi_on_bank(profit_30d: Optional[float], bank: Optional[float]) -> Optional[float]:
            if profit_30d is None or bank is None or bank <= 0:
                return None
            return (float(profit_30d) / float(bank)) * 100.0

        lines.append("| Bloco | Janela(d) | Turnover 30d | Lucro 30d (direto) | Lucro 30d (ROI×turnover) |\n|---|---:|---:|---:|---:|\n")
        lines.append(
            f"| Back | {window_days:.1f} | {_fmt_num(back_turnover_30d, 2)} | {_fmt_num(back_profit_30d_direct, 2)} | {_fmt_num(back_profit_30d_roi, 2)} |\n"
        )
        lines.append(
            f"| Lay (stake) | {window_days:.1f} | {_fmt_num(lay_turnover_30d, 2)} | {_fmt_num(lay_profit_30d_direct, 2)} | {_fmt_num(lay_profit_30d_roi, 2)} |\n"
        )
        lines.append(
            f"| Total (Back+Lay) | {window_days:.1f} | {_fmt_num(total_turnover_30d, 2)} | {_fmt_num(total_profit_30d_direct, 2)} | {_fmt_num(total_profit_30d_roi, 2)} |\n"
        )
        lines.append("\n")

        lines.append("**Banca por risco (exposição unitária)**\n\n")
        lines.append("| Bloco | Banca conservadora (p99) | Banca agressiva (ES95) | ROI/banca 30d (direto) | ROI/banca 30d (ROI×turnover) |\n|---|---:|---:|---:|---:|\n")
        lines.append(
            f"| Back (stake) | {_fmt_num(back_bank_conservative, 2)} | {_fmt_num(back_bank_aggressive, 2)} | "
            f"{_fmt_num(_roi_on_bank(back_profit_30d_direct, back_bank_conservative), 2)}% | "
            f"{_fmt_num(_roi_on_bank(back_profit_30d_roi, back_bank_conservative), 2)}% |\n"
        )
        lines.append(
            f"| Lay (liability) | {_fmt_num(lay_bank_conservative, 2)} | {_fmt_num(lay_bank_aggressive, 2)} | "
            f"{_fmt_num(_roi_on_bank(lay_profit_30d_direct, lay_bank_conservative), 2)}% | "
            f"{_fmt_num(_roi_on_bank(lay_profit_30d_roi, lay_bank_conservative), 2)}% |\n"
        )
        lines.append(
            f"| Total (soma) | {_fmt_num(total_bank_conservative, 2)} | {_fmt_num(total_bank_aggressive, 2)} | "
            f"{_fmt_num(_roi_on_bank(total_profit_30d_direct, total_bank_conservative), 2)}% | "
            f"{_fmt_num(_roi_on_bank(total_profit_30d_roi, total_bank_conservative), 2)}% |\n"
        )
        lines.append("\n")

        lines.append("**Banca por liquidez (capital simultaneamente travado)**\n\n")
        lines.append(
            f"Definição operacional: cada aposta trava capital de `audited_at` até `kickoff + {LIQUIDITY_MATCH_DURATION_HOURS:.2f}h + {LIQUIDITY_SETTLE_BUFFER_HOURS:.2f}h` "
            f"(grid={LIQUIDITY_GRID_MINUTES}min). A banca recomendada aplica buffer de +{LIQUIDITY_BANK_BUFFER_PCT:.0f}%.\n\n"
        )
        lines.append("| Bloco | Liquidez mean | Liquidez p95 | Liquidez p99 | Liquidez max | Banca liq p99 (+buffer) |\n|---|---:|---:|---:|---:|---:|\n")
        lines.append(
            f"| Back (stake) | {_fmt_num(back_liq.get('mean'), 2)} | {_fmt_num(back_liq.get('p95'), 2)} | {_fmt_num(back_liq.get('p99'), 2)} | {_fmt_num(back_liq.get('max'), 2)} | {_fmt_num(_with_buffer(back_liq.get('p99')), 2)} |\n"
        )
        lines.append(
            f"| Lay (liability) | {_fmt_num(lay_liq.get('mean'), 2)} | {_fmt_num(lay_liq.get('p95'), 2)} | {_fmt_num(lay_liq.get('p99'), 2)} | {_fmt_num(lay_liq.get('max'), 2)} | {_fmt_num(_with_buffer(lay_liq.get('p99')), 2)} |\n"
        )
        lines.append(
            f"| Total (Back+Lay) | {_fmt_num(total_liq.get('mean'), 2)} | {_fmt_num(total_liq.get('p95'), 2)} | {_fmt_num(total_liq.get('p99'), 2)} | {_fmt_num(total_liq.get('max'), 2)} | {_fmt_num(total_bank_liq_p99, 2)} |\n"
        )
        lines.append("\n")

        lines.append("**Banca recomendada (conservadora)**\n\n")
        lines.append("| Métrica | Valor |\n|---|---:|\n")
        lines.append(f"| Banca por risco (p99 unitário, soma) | {_fmt_num(total_bank_conservative, 2)} |\n")
        lines.append(f"| Banca por liquidez (p99 simultâneo + buffer) | {_fmt_num(total_bank_liq_p99, 2)} |\n")
        lines.append(f"| Banca efetiva (max das duas) | {_fmt_num(total_bank_eff_conservative, 2)} |\n")
        lines.append(f"| ROI/banca 30d (direto, banca efetiva) | {_fmt_num(_roi_on_bank(total_profit_30d_direct, total_bank_eff_conservative), 2)}% |\n")
        lines.append(f"| ROI/banca 30d (ROI×turnover, banca efetiva) | {_fmt_num(_roi_on_bank(total_profit_30d_roi, total_bank_eff_conservative), 2)}% |\n")

        lines.append("**Diagnóstico de cobertura (placar/ROI)**\n\n")
        lines.append("| Bloco | Turnover total | Turnover com ROI | Cobertura turnover |\n|---|---:|---:|---:|\n")
        lines.append(
            f"| Back | {_fmt_num(back_turnover_total, 2)} | {_fmt_num(back_turnover_realized, 2)} | "
            f"{_fmt_num((back_turnover_realized / back_turnover_total * 100.0) if back_turnover_total > 0 else None, 2)}% |\n"
        )
        lines.append(
            f"| Lay | {_fmt_num(lay_turnover_total, 2)} | {_fmt_num(lay_turnover_realized, 2)} | "
            f"{_fmt_num((lay_turnover_realized / lay_turnover_total * 100.0) if lay_turnover_total > 0 else None, 2)}% |\n"
        )
        lines.append("\n")
        lines.append(
            f"Notas (Lay): exposição 30d por liability (não é turnover) = {_fmt_num(lay_liab_30d, 2)}; "
            f"ROI realizado por liability (ponderado) = {_fmt_num(lay_roi_realized_liab, 2)}%.\n"
        )

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

        def _extract_ws_series(d: dict) -> List[dict]:
            """
            Normaliza `hypothesis_details.ws_series` (WS-only) para pontos {t, odd}.
            Observação: os pontos podem vir com `t_target_s` e `t_actual_s`; preferimos `t_actual_s`.
            """
            h = d.get("hypothesis_details") or {}
            arr = h.get("ws_series") if isinstance(h, dict) else None
            if not isinstance(arr, list):
                return []
            out: List[dict] = []
            for e in arr:
                if not isinstance(e, dict):
                    continue
                t = _safe_float(e.get("t_actual_s"))
                if t is None:
                    t = _safe_float(e.get("t_target_s"))
                odd = _safe_float(e.get("ws_odd"))
                if t is None or odd is None or odd <= 0:
                    continue
                out.append({"t": float(t), "odd": float(odd)})
            out.sort(key=lambda x: x["t"])
            return out

        def _build_back_series(d: dict) -> List[dict]:
            """
            Série temporal do BACK (entrada) para análise de pico/reversão.

            Compatibilidade:
            - Antigo: `hypothesis_details.temporal` (pontos com BS refresh).
            - Novo: `hypothesis_details.ws_series` (pontos inteiramente via WS).

            Para manter comparabilidade, definimos `diff_pct` sempre vs `ws_odd` do t0:
              diff_pct(t) = (odd_t - ws_t0) / ws_t0 * 100
            onde odd_t é BS(t) quando existir; caso contrário WS(t).
            """
            h = d.get("hypothesis_details") or {}
            ws0 = _safe_float(d.get("ws_odd"))
            bs0 = _safe_float(d.get("bs_odd"))
            ws_series = _extract_ws_series(d)
            # fallback se ws_odd estiver ausente
            if (ws0 is None or ws0 <= 0) and ws_series:
                ws0 = _safe_float(ws_series[0].get("odd"))
            if ws0 is None or ws0 <= 0:
                return []

            # t0 odd: preferimos BS (execução), senão WS (mercado)
            t0_odd = bs0 if (bs0 is not None and bs0 > 0) else ws0
            series: List[dict] = [{"t": 0.0, "odd": float(t0_odd), "diff_pct": float((t0_odd - ws0) / ws0 * 100.0)}]

            arr = h.get("temporal") if isinstance(h, dict) else None
            if isinstance(arr, list) and len(arr) > 0:
                # temporal via BS
                for e in arr:
                    if not isinstance(e, dict):
                        continue
                    t = _safe_float(e.get("t"))
                    # Pós-mudança BS->WS alguns pipelines podem salvar `ws_odd` no array temporal.
                    odd = _safe_float(e.get("bs_odd"))
                    if odd is None:
                        odd = _safe_float(e.get("ws_odd"))
                    if t is None or odd is None or odd <= 0:
                        continue
                    if float(t) <= 0.0005:
                        continue
                    series.append({"t": float(t), "odd": float(odd), "diff_pct": float((odd - ws0) / ws0 * 100.0)})
            else:
                # temporal via WS
                for p in ws_series:
                    t = _safe_float(p.get("t"))
                    odd = _safe_float(p.get("odd"))
                    if t is None or odd is None or odd <= 0:
                        continue
                    if float(t) <= 0.0005:
                        continue
                    series.append({"t": float(t), "odd": float(odd), "diff_pct": float((odd - ws0) / ws0 * 100.0)})

            series.sort(key=lambda x: x["t"])
            return series

        def _build_lay_series(d: dict) -> List[dict]:
            """
            Série temporal do LAY.

            Preferimos dados reais do lay via betslip (`lay`/`lay_temporal`). Se não houver
            `lay_temporal` mas existir `ws_series`, usamos WS como proxy APENAS para dinâmica temporal
            (mantendo `diff_pct` vs ws_t0).
            """
            h = d.get("hypothesis_details") or {}
            ws0 = _safe_float(d.get("ws_odd"))
            lay0 = _safe_float(_get_path(h, ["lay", "odd"])) if isinstance(h, dict) else None
            ws_series = _extract_ws_series(d)

            # base de comparação: ws_t0 (quando disponível)
            if (ws0 is None or ws0 <= 0) and ws_series:
                ws0 = _safe_float(ws_series[0].get("odd"))
            if ws0 is None or ws0 <= 0:
                return []

            series: List[dict] = []
            if lay0 is not None and lay0 > 0:
                series.append({"t": 0.0, "odd": float(lay0), "diff_pct": float((lay0 - ws0) / ws0 * 100.0)})
            else:
                # Sem lay0: não inventamos série de lay, a menos que haja dados explícitos.
                return []

            arr = h.get("lay_temporal") if isinstance(h, dict) else None
            if isinstance(arr, list) and len(arr) > 0:
                for e in arr:
                    if not isinstance(e, dict):
                        continue
                    t = _safe_float(e.get("t"))
                    odd = _safe_float(e.get("lay_odd"))
                    if odd is None:
                        odd = _safe_float(e.get("ws_odd"))
                    if t is None or odd is None or odd <= 0:
                        continue
                    if float(t) <= 0.0005:
                        continue
                    series.append({"t": float(t), "odd": float(odd), "diff_pct": float((odd - ws0) / ws0 * 100.0)})
            else:
                # Se não há lay_temporal, mas há ws_series, podemos ao menos medir dinâmica temporal
                # (proxy do lay) — útil quando o pipeline mudou de BS->WS para t>0.
                for p in ws_series:
                    t = _safe_float(p.get("t"))
                    odd = _safe_float(p.get("odd"))
                    if t is None or odd is None or odd <= 0:
                        continue
                    if float(t) <= 0.0005:
                        continue
                    series.append({"t": float(t), "odd": float(odd), "diff_pct": float((odd - ws0) / ws0 * 100.0)})

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
            odd_rev = None
            diff_rev = None
            if after:
                if mode == "back":
                    threshold = ext["diff_pct"] - EPS_REV
                    for p in series[idx_ext + 1 :]:
                        if p["diff_pct"] <= threshold:
                            had_rev = True
                            t_rev = p["t"]
                            odd_rev = p.get("odd")
                            diff_rev = p.get("diff_pct")
                            break
                else:
                    threshold = ext["diff_pct"] + EPS_REV
                    for p in series[idx_ext + 1 :]:
                        if p["diff_pct"] >= threshold:
                            had_rev = True
                            t_rev = p["t"]
                            odd_rev = p.get("odd")
                            diff_rev = p.get("diff_pct")
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
                "odd_reversal": float(odd_rev) if odd_rev is not None else None,
                "diff_reversal": float(diff_rev) if diff_rev is not None else None,
            }

        def _clv_pct_from_odd(odd: Optional[float], closing_odd: Any) -> Optional[float]:
            odd = _safe_float(odd)
            clo = _safe_float(closing_odd)
            if odd is None or clo is None or clo <= 0:
                return None
            return (odd - clo) / clo * 100.0

        def _clv_pct_lay_from_odd(lay_odd: Optional[float], closing_odd: Any) -> Optional[float]:
            """
            CLV "raw" para Lay, compatível com a convenção geral (entry - closing) / closing.

            Interpretação:
            - Lay "bom" tende a ser NEGATIVO (você entrou barato e o closing subiu).
            - Se você preferir uma convenção positiva para Lay, use `clv_conv = -clv_raw`.
            """
            lay_odd = _safe_float(lay_odd)
            clo = _safe_float(closing_odd)
            if lay_odd is None or clo is None or clo <= 0:
                return None
            return (lay_odd - clo) / clo * 100.0

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

        # Base para timing: inclui auditorias OK mesmo sem betslip (WS-only),
        # desde que exista série temporal suficiente (>=2 pontos).
        ok_timing_back = [d for d in ok_any if len(_build_back_series(d)) >= 2]
        ok_timing_lay = [d for d in ok_any if len(_build_lay_series(d)) >= 2]

        lines.append("## 8) Curva temporal (pico, reversão e melhor timing)\n")
        lines.append(
            "Esta seção usa séries temporais coletadas em pontos discretos (t≈0,3,6,10,15,20s). Fontes possíveis:\n\n"
            "- **BS-temporal (legado)**: `hypothesis_details.temporal` (Back) e `hypothesis_details.lay_temporal` (Lay)\n"
            "- **WS-temporal (novo)**: `hypothesis_details.ws_series` (todos os t’s via WebSocket)\n\n"
            "Para manter comparabilidade, nesta seção `diff_pct(t)` é sempre calculado contra o **WS do t0** (`ws_odd`): `(odd_t - ws_t0)/ws_t0*100`.\n\n"
        )
        lines.append(
            "O objetivo é responder: **tempo até o pico/vale**, **% que segue melhorando até t_max**, **% com reversão** e **impacto esperado por timing**. "
            "**CLV é reportado somente pre-match** (closing pré-jogo).\n\n"
        )

        # BACK
        back_stats = _summarize_timing(ok_timing_back, mode="back")["rows"]
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
        for t_label, n, md, mo, mclv, mroi in _curve_table(ok_timing_back, mode="back"):
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
            # Entrada operacional (Back):
            # - se há BS (execução), usamos BS como odd de entrada
            # - senão (WS-only), usamos WS em t+offset como proxy de entrada (por padrão 5s)
            ws0 = _safe_float(d.get("ws_odd"))
            if (ws0 is None or ws0 <= 0) and len(series) > 0:
                # fallback: ws0 como o primeiro ponto observado
                ws0 = _safe_float(series[0].get("odd"))
            tplus = _safe_float(os.getenv("WS_GATE_TPLUS_SEC", "5.0")) or 5.0
            bs_exec = _safe_float(d.get("bs_odd"))
            entry_odd = None
            entry_t = None
            entry_src = None
            if bs_exec is not None and float(bs_exec) > 0:
                entry_odd = float(bs_exec)
                entry_t = 0.0
                entry_src = "BS"
            else:
                # procura ponto mais próximo (>=tplus); se não houver, usa o último ponto
                cand = [p for p in series if _safe_float(p.get("t")) is not None and float(p.get("t")) >= float(tplus) and _safe_float(p.get("odd"))]
                if cand:
                    # pega o mais perto do alvo
                    cand.sort(key=lambda p: abs(float(p.get("t")) - float(tplus)))
                    entry_odd = float(cand[0]["odd"])
                    entry_t = float(cand[0]["t"])
                    entry_src = f"WS@t+{entry_t:.1f}s"
                else:
                    entry_odd = float(plast["odd"])
                    entry_t = float(plast.get("t") or 0.0)
                    entry_src = f"WS@t+{entry_t:.1f}s"
            diff_entry = None
            if ws0 is not None and float(ws0) > 0 and entry_odd is not None and float(entry_odd) > 0:
                diff_entry = (float(entry_odd) - float(ws0)) / float(ws0) * 100.0
            # ROI por ponto (se houver placar)
            mult = _outcome_mult(str(d.get("line", "")), str(d.get("side", "")), d.get("home_score"), d.get("away_score"))
            roi0 = _roi_back_pct(p0["odd"], mult) if mult is not None else None
            roipeak = _roi_back_pct(a["odd_ext"], mult) if mult is not None else None
            roilast = _roi_back_pct(plast["odd"], mult) if mult is not None else None
            roi_entry = _roi_back_pct(float(entry_odd), mult) if (mult is not None and entry_odd is not None) else None
            # CLV só faz sentido pre-match (closing_odd é pré-jogo).
            clv0 = _clv_pct_from_odd(p0["odd"], closing) if d.get("is_live") is False else None
            clve = _clv_pct_from_odd(a["odd_ext"], closing) if d.get("is_live") is False else None
            clvl = _clv_pct_from_odd(plast["odd"], closing) if d.get("is_live") is False else None
            clv_entry = _clv_pct_from_odd(float(entry_odd), closing) if (d.get("is_live") is False and entry_odd is not None) else None
            return {
                "audit_id": int(d.get("id")) if d.get("id") is not None else None,
                "match_id": int(d.get("match_id")),
                "is_live": d.get("is_live"),
                "audited_at": d.get("audited_at"),
                "had_reversal": bool(a.get("had_reversal")),
                "ext_at_end": bool(a.get("ext_at_end")),
                "monotonic": bool(a.get("monotonic")),
                "t_ext": a.get("t_ext"),
                "odd_t0": float(p0["odd"]),
                "odd_ext": float(a["odd_ext"]),
                "odd_last": float(plast["odd"]),
                "odd_entry": float(entry_odd) if entry_odd is not None else None,
                "t_entry": float(entry_t) if entry_t is not None else None,
                "entry_source": str(entry_src or ""),
                "diff_entry": diff_entry,
                "closing_odd": _safe_float(closing),
                "clv_t0": clv0,
                "clv_ext": clve,
                "clv_last": clvl,
                "clv_entry": clv_entry,
                "roi_t0": roi0,
                "roi_ext": roipeak,
                "roi_last": roilast,
                "roi_entry": roi_entry,
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
            odd_rev = _safe_float(a.get("odd_reversal"))
            roirev = _roi_lay_pct_per_liability(odd_rev, mult) if (mult is not None and odd_rev is not None) else None
            # CLV só faz sentido pre-match (closing_odd é pré-jogo).
            clv0 = _clv_pct_lay_from_odd(p0["odd"], closing) if d.get("is_live") is False else None
            clve = _clv_pct_lay_from_odd(a["odd_ext"], closing) if d.get("is_live") is False else None
            clvl = _clv_pct_lay_from_odd(plast["odd"], closing) if d.get("is_live") is False else None
            # Convenção unificada de CLV para Lay (8.3/OOS):
            # clv_conv = -(entry - closing)/closing = (closing - entry)/closing
            # Portanto, Lay "bom" tende a ser POSITIVO.
            clv0_conv = (-float(clv0)) if clv0 is not None else None
            clve_conv = (-float(clve)) if clve is not None else None
            clvl_conv = (-float(clvl)) if clvl is not None else None

            # Política de entrada Lay (para 8.3 e OOS):
            # - se há reversão: entrar **logo após a reversão** (odd_reversal)
            # - se não há reversão: entrar no **último ponto** (t_last, ~t+20s)
            has_rev = bool(a.get("had_reversal")) and (odd_rev is not None)
            odd_entry = float(odd_rev) if has_rev else float(plast["odd"])
            roi_entry = roirev if has_rev else roilast
            clv_entry = _clv_pct_lay_from_odd(odd_entry, closing) if d.get("is_live") is False else None
            clv_entry_conv = (-float(clv_entry)) if clv_entry is not None else None
            # diff no ponto de entrada vs ws_t0 (para coerência com OOS)
            ws0 = _safe_float(d.get("ws_odd"))
            if (ws0 is None or ws0 <= 0):
                ws_series = _extract_ws_series(d)
                if ws_series:
                    ws0 = _safe_float(ws_series[0].get("odd"))
            diff_entry = None
            if ws0 is not None and float(ws0) > 0 and odd_entry is not None and float(odd_entry) > 0:
                diff_entry = (float(odd_entry) - float(ws0)) / float(ws0) * 100.0
            return {
                "audit_id": int(d.get("id")) if d.get("id") is not None else None,
                "match_id": int(d.get("match_id")),
                "is_live": d.get("is_live"),
                "audited_at": d.get("audited_at"),
                "had_reversal": bool(a.get("had_reversal")),
                "ext_at_end": bool(a.get("ext_at_end")),
                "monotonic": bool(a.get("monotonic")),
                "t_ext": a.get("t_ext"),
                "clv_t0": clv0,
                "clv_ext": clve,
                "clv_last": clvl,
                "clv_conv_t0": clv0_conv,
                "clv_conv_ext": clve_conv,
                "clv_conv_last": clvl_conv,
                "roi_t0": roi0,
                "roi_ext": roival,
                "roi_last": roilast,
                "odd_reversal": odd_rev,
                "roi_rev": roirev,
                "odd_entry": odd_entry,
                "roi_entry": roi_entry,
                "clv_entry": clv_entry,
                "clv_conv_entry": clv_entry_conv,
                "diff_entry": diff_entry,
            }

        def _summarize_entry(rows: List[dict], key: str, *, clip_low: Optional[float] = None, clip_high: Optional[float] = None) -> MetricSummary:
            vals = [r.get(key) for r in rows]
            mids = [r.get("match_id") for r in rows]
            return summarize_metric(vals, mids, clip_low=clip_low, clip_high=clip_high)

        # 8.1b) impacto: t0 vs pico vs último, com/sem reversão
        back_entries = [em for d in ok_timing_back for em in [_entry_metrics_back(d)] if em is not None]
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
        lay_stats = _summarize_timing(ok_timing_lay, mode="lay")["rows"]
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
        for t_label, n, md, mo, mclv, mroi in _curve_table(ok_timing_lay, mode="lay"):
            lines.append(f"| {t_label} | {n} | {_fmt_pct(md,2)} | {_fmt_num(mo,3)} | {_fmt_pct(mclv,2)} | {_fmt_num(mroi,2)} |\n")

        lay_entries = [em for d in ok_timing_lay for em in [_entry_metrics_lay(d)] if em is not None]
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
        # 8.3) Resumo das 8 combinações (Side × Pre/In × Reversal)
        # ============================================================
        lines.append("### 8.3 Resumo de estratégias — combinações (Side × Pre/In × Reversal)\n")
        lines.append(
            "Esta tabela resume as combinações possíveis. Observação importante:\n\n"
            "- **Back**: a estratégia é **entrar rápido em `t0`**, então **não faz sentido separar por Reversal(Sim/Não)** (agregamos como `Any`).\n"
            "- **Lay**: entrada **após reversão** quando ela existe (`odd_reversal`), senão no **último ponto** (~t+20s).\n"
            "- **CLV** aqui é **somente pre‑match** (closing pré‑jogo). Para **Lay**, usamos a convenção unificada `clv_conv = -(entry - closing)/closing`, logo **Lay “bom” tende a CLV_CONV > 0**.\n"
            "- **ROI** é calculado no **ponto de entrada da estratégia** (se houver placar). Para Lay, ROI é **por liability**.\n"
            "- **IC90** é bootstrap por jogo (cluster em `match_id`). Para critério “p30” usamos o quantil bootstrap **p30** do estimador.\n\n"
        )

        by_audit_id: Dict[int, dict] = {int(d.get("id")): d for d in ok_any if d.get("id") is not None}

        def _combo_rows(entries: List[dict], *, is_live: bool, had_rev: bool) -> List[dict]:
            return [r for r in entries if r.get("is_live") is is_live and bool(r.get("had_reversal")) is bool(had_rev)]

        def _summ_ci(rows: List[dict], key: str, clip_low: float, clip_high: float) -> Tuple[Optional[float], Optional[Tuple[float, float]], Optional[float], Optional[float], int, int]:
            vals = [r.get(key) for r in rows]
            mids = [r.get("match_id") for r in rows]
            s = summarize_metric(vals, mids, clip_low=clip_low, clip_high=clip_high)
            bym: Dict[int, List[float]] = {}
            for v, mid in zip(vals, mids):
                vf = _safe_float(v)
                if vf is None:
                    continue
                if clip_low is not None and vf < clip_low:
                    continue
                if clip_high is not None and vf > clip_high:
                    continue
                bym.setdefault(int(mid), []).append(float(vf))
            q10 = cluster_bootstrap_quantile(bym, 0.10, n_boot=2000, seed=int(args.seed))
            q30 = cluster_bootstrap_quantile(bym, 0.30, n_boot=2000, seed=int(args.seed))
            return s.mean_cluster, s.ci90_cluster, q10, q30, s.n_events, s.n_matches

        lines.append("| Side | Pre/In | Reversal | N | Jogos | CLV t0 (mean; IC90) | ROI (mean; IC90) | ROI p30 | Ativa? (critério) |\n")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---|\n")
        for side, entries in [("Back", back_entries), ("Lay", lay_entries)]:
            for is_live_val, regime_label in [(False, "Pre"), (True, "In")]:
                rev_iters = [(None, "Any")] if side == "Back" else [(True, "Yes"), (False, "No")]
                for had_rev_val, rev_label in rev_iters:
                    if had_rev_val is None:
                        filt = [r for r in entries if r.get("is_live") is is_live_val]
                    else:
                        filt = _combo_rows(entries, is_live=is_live_val, had_rev=had_rev_val)
                    # CLV só pre-match
                    clv_mean = clv_ci = clv_q10 = clv_q30 = None
                    n_clv = m_clv = 0
                    if is_live_val is False:
                        clv_key = "clv_t0" if side == "Back" else "clv_conv_entry"
                        clv_mean, clv_ci, clv_q10, clv_q30, n_clv, m_clv = _summ_ci(
                            filt,
                            clv_key,
                            clip_low=-50,
                            clip_high=50,
                        )
                    # ROI (t0). Back: ROI/stake. Lay: ROI/liability (já calculado assim no entry_metrics)
                    roi_clip_hi = 500.0 if side == "Back" else 5000.0
                    roi_key = "roi_t0" if side == "Back" else "roi_entry"
                    roi_mean, roi_ci, roi_q10, roi_q30, n_roi, m_roi = _summ_ci(
                        filt,
                        roi_key,
                        clip_low=-200.0,
                        clip_high=roi_clip_hi,
                    )
                    n_total = len(filt)
                    m_total = len(set(int(r.get("match_id")) for r in filt if r.get("match_id") is not None))

                    # Critérios alinhados ao OOS (walk-forward):
                    # - Se ROI for sig<0 (IC90 ub<0): bloqueia
                    # - Se ROI for sig>0 (IC90 lb>0): ativa
                    # - Se ROI>0 mas não sig:
                    #   - Pre: ativa se CLV>0 (não precisa ser sig)  [Lay usa CLV_CONV]
                    #   - In : ativa se ROI>0
                    active = None
                    crit = ""
                    roi_sig_pos = bool(roi_ci and float(roi_ci[0]) > 0) if roi_ci else False
                    roi_sig_neg = bool(roi_ci and float(roi_ci[1]) < 0) if roi_ci else False
                    roi_pos = bool(roi_mean is not None and float(roi_mean) > 0)
                    clv_pos = bool(clv_mean is not None and float(clv_mean) > 0) if regime_label == "Pre" else False
                    if roi_sig_neg:
                        active = False
                        crit = "ROI sig<0 bloqueia"
                    elif roi_sig_pos:
                        active = True
                        crit = "ROI sig>0"
                    else:
                        if regime_label == "Pre":
                            active = bool(roi_pos and clv_pos)
                            crit = "ROI>0 (NS) AND CLV>0"
                        else:
                            active = bool(roi_pos)
                            crit = "ROI>0"
                    active_lbl = ("sim" if active else "não") if active is not None else "—"

                    clv_str = f"{_fmt_pct(clv_mean,2)} {_fmt_ci(clv_ci,2)}" if (is_live_val is False) else "—"
                    roi_str = f"{_fmt_pct(roi_mean,2)} {_fmt_ci(roi_ci,2)}"
                    lines.append(
                        f"| {side} | {regime_label} | {rev_label} | {n_total} | {m_total} | {clv_str} | {roi_str} | {_fmt_pct(roi_q30,2)} | {active_lbl} ({crit}) |\n"
                    )

        lines.append(
            "\nNotas:\n"
            "- Se você quiser operar **cada combinação como uma estratégia separada**, o sizing (Kelly/caps) deve ser estimado e governado separadamente por estratégia.\n"
            "- Esta tabela é **in-sample** na janela (`--lookback-days`). A seção OOS (walk-forward) pode ser habilitada com `--walkforward`.\n\n"
        )

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
        # 9.3) Stake sizing (teoria + calibração empírica)
        # ============================================================
        lines.append("### 9.3 Stake sizing — teoria mínima + calibração empírica\n")
        lines.append(
            "Objetivo: explicar por que **ROI por aposta** pode divergir de **ROI ponderado por stake/liability**, "
            "e propor uma política de staking que seja (i) coerente com edge/CLV e (ii) controlada por risco (p99/ES).\n\n"
        )

        lines.append("**Teoria (resumo prático)**\n\n")
        lines.append("- **Flat stake**: cada aposta pesa igual. Boa baseline para checar se o sizing atual está piorando resultado.\n")
        lines.append("- **Proporcional ao limite**: útil operacionalmente (capacidade), mas **não é** sizing por edge.\n")
        lines.append("- **Kelly fracionado**: sizing por edge. Para Back, \\(f^* \\propto \\frac{EV}{odds-1}\\). Para Lay, o sizing natural é por **liability**.\n")
        lines.append("- **Governança de risco**: impor **cap por aposta** (ex.: 1–2% da banca) e olhar p95/p99/ES95 de exposição.\n\n")

        lines.append("**Como o Kelly está sendo calculado aqui (detalhado, com premissas)**\n\n")
        lines.append(
            "Como ainda não temos um modelo explícito de probabilidade \\(p\\) por aposta, usamos um proxy padrão: "
            "**o closing pré‑jogo como melhor estimativa de preço justo**. A partir disso inferimos \\(p\\) e aplicamos Kelly como aproximação.\n\n"
        )
        lines.append("Premissas e entradas:\n\n")
        lines.append("- **Entrada (Back)**: `entry_odd = bs_odd` (odd do betslip no momento de execução).\n")
        lines.append("- **Entrada (Lay)**: `entry_lay_odd = hypothesis_details.lay.odd` (fallback: `bs_odd`).\n")
        lines.append("- **Preço justo (pre‑match)**: `closing_odd` (closing line). Inferimos \\(p \\approx 1/closing\\_odd\\).\n")
        lines.append("- **Aplicabilidade**: para `is_live=True` (in‑match), **não usamos** `closing_odd` como benchmark de CLV/Kelly.\n\n")
        lines.append("Fórmulas (Back):\n\n")
        lines.append("- Odds decimais \\(O\\); retorno líquido \\(b = O-1\\).\n")
        lines.append("- \\(p \\approx 1/closing\\_odd\\).\n")
        lines.append("- Valor esperado por unidade de stake: \\(EV = O\\cdot p - 1\\).\n")
        lines.append("- Kelly cheio (fração de banca em **stake**): \\(f^* = \\frac{EV}{b} = \\frac{O\\cdot p - 1}{O-1}\\).\n")
        lines.append("- No relatório: \\(f = \\max(0,f^*)\\cdot \\text{frac}\\) com `frac` em {0.10, 0.25, 0.50, 1.00}.\n\n")
        lines.append("Fórmulas (Lay):\n\n")
        lines.append("- Para Lay, o “capital em risco” natural é a **liability** \\(L\\) (perda máxima), não o stake.\n")
        lines.append("- Usamos \\(p \\approx 1/closing\\_odd\\) e \\(o = entry\\_lay\\_odd\\).\n")
        lines.append("- Kelly em termos de **liability** (proxy): \\(f^*_{liab} = 1 - p\\cdot o\\).\n")
        lines.append("- No relatório: \\(f_{liab} = \\max(0,f^*_{liab})\\cdot \\text{frac}\\).\n")
        lines.append("- Conversão para stake (apenas para turnover): \\(stake = L/(o-1)\\).\n\n")
        lines.append("Derivação rápida (por que \\(f^*_{liab}=1-p\\cdot o\\)):\n\n")
        lines.append(
            "- Defina \\(W\\) como banca e escolha alocar \\(L=f\\cdot W\\) como **liability**.\n"
            "- Se o evento acontece (prob. \\(p\\)), você perde \\(L\\): \\(W' = W-L = W(1-f)\\).\n"
            "- Se o evento não acontece (prob. \\(1-p\\)), você ganha o **stake** do Lay, que é \\(S=L/(o-1)\\): "
            "\\(W' = W+S = W\\left(1+\\frac{f}{o-1}\\right)\\).\n"
            "- Kelly maximiza \\(p\\log(1-f) + (1-p)\\log\\left(1+\\frac{f}{o-1}\\right)\\). "
            "Derivando e igualando a zero, obtém-se \\(f^* = 1 - p\\cdot o\\).\n\n"
        )
        lines.append("Parâmetros de escala (proxy de banca) e caps:\n\n")
        lines.append("- Por padrão: `back_bank_ref = p99(stake)` e `lay_bank_ref = p99(liability)` observados no sizing **PROXY** da janela.\n")
        lines.append("- Opcional: com `--kelly-bankroll`, usamos `bank_ref = bankroll` para simular capacidade com banca explícita.\n")
        lines.append("- `stake_back = min(f * back_bank_ref, cap_back, cap_evento_limit)`.\n")
        lines.append("- `liab_lay = min(f_liab * lay_bank_ref, cap_lay, cap_evento_limit)`.\n")
        _bk = max(0.0, float(getattr(args, "kelly_back_cap_frac", 0.02)))
        _lk = max(0.0, float(getattr(args, "kelly_lay_cap_frac", 0.01)))
        _ms = max(0.0, float(getattr(args, "max_stake_pct_of_limit", 1.0)))
        _msc = max(0.0, float(getattr(args, "max_stake_cap", 0.0)))
        lines.append(
            f"- Caps atuais (guardrail): `cap_back = {_bk:.1%} * ref`, `cap_lay = {_lk:.1%} * ref`. "
            f"Cap por evento: `max_stake = {_ms:.0%} * limit`"
            + (f" e `max_stake_cap={_msc:.2f}`" if _msc > 0 else "")
            + ".\n"
        )
        lines.append(
            "- **Implicação importante**: se o cap estiver frequentemente ativo, aumentar `frac` (ex.: >0,25×Kelly) "
            "**não aumenta** tamanho real — a curva satura.\n\n"
        )
        lines.append(
            "Limitações: comissão/vigorish não modelados; correlação entre apostas ignorada; closing como preço justo é aproximação; "
            "e o `bank_ref` é uma escala interna (proxy) baseada em limits observados.\n\n"
        )

        # --- calibração: stake vs ROI/CLV ---
        def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
            if len(xs) < 3 or len(ys) < 3:
                return None
            try:
                x = np.array(xs, dtype=float)
                y = np.array(ys, dtype=float)
                if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
                    return None
                return float(np.corrcoef(x, y)[0, 1])
            except Exception:
                return None

        back_pairs = []
        lay_pairs = []
        for d in ok_bs:
            roi = _safe_float(d.get("roi_bs"))
            if roi is None:
                continue
            bs, _, _, ll = finance_for_row(d)
            clv = _safe_float(d.get("clv_bs"))
            if int(d["id"]) in back_edge_ids:
                back_pairs.append((float(bs), float(roi), float(clv) if clv is not None else None))
            if int(d["id"]) in lay_edge_ids:
                lay_pairs.append((float(ll), float(roi), float(clv) if clv is not None else None))

        def _corr_block(pairs: List[Tuple[float, float, Optional[float]]], label: str):
            xs_roi = [p[0] for p in pairs]
            ys_roi = [p[1] for p in pairs]
            xs_clv = [p[0] for p in pairs if p[2] is not None]
            ys_clv = [p[2] for p in pairs if p[2] is not None]
            c_roi = _pearson(xs_roi, ys_roi)
            c_clv = _pearson(xs_clv, ys_clv)
            lines.append(f"- **{label}**: corr(exposição, ROI)={_fmt_num(c_roi,3)}; corr(exposição, CLV)={_fmt_num(c_clv,3)} (onde CLV existe).\n")

        lines.append("**Diagnóstico: exposição vs performance (correlação de Pearson; indicativo, não causal)**\n\n")
        _corr_block(back_pairs, "Back (stake)")
        _corr_block(lay_pairs, "Lay (liability)")
        lines.append("\n")

        # --- sizing rules / backtest ---
        def _kelly_back_frac(entry_odd: Any, closing_odd: Any) -> Optional[float]:
            o = _safe_float(entry_odd)
            c = _safe_float(closing_odd)
            if o is None or c is None or o <= 1.0 or c <= 1.0:
                return None
            p = 1.0 / c
            b = o - 1.0
            ev = (o * p) - 1.0  # esperado por stake=1
            f = ev / b
            return float(f)

        def _kelly_lay_liab_frac(entry_lay_odd: Any, closing_odd: Any) -> Optional[float]:
            """
            Fração ótima de banca em termos de **liability** para Lay:
            f* = 1 - p*o, com p≈1/closing_odd e o=entry_lay_odd.
            """
            o = _safe_float(entry_lay_odd)
            c = _safe_float(closing_odd)
            if o is None or c is None or o <= 1.0 or c <= 1.0:
                return None
            p = 1.0 / c
            f = 1.0 - (p * o)
            return float(f)

        def _pnl_back(roi_pct: Optional[float], stake: float) -> Optional[float]:
            if roi_pct is None:
                return None
            return float(stake) * float(roi_pct) / 100.0

        def _pnl_lay_from_liab(liab: float, lay_odd: float, mult_back: Optional[float]) -> Optional[float]:
            if mult_back is None:
                return None
            roi_liab = _roi_lay_pct_per_liability(float(lay_odd), float(mult_back))
            if roi_liab is None:
                return None
            return float(liab) * float(roi_liab) / 100.0

        def _max_drawdown(series: List[float]) -> float:
            eq = 0.0
            peak = 0.0
            max_dd = 0.0
            for x in series:
                eq += float(x)
                peak = max(peak, eq)
                max_dd = min(max_dd, eq - peak)
            return float(-max_dd)

        def _bootstrap_dd(daily_pnls: List[float], horizon_days: int = 30, n_boot: int = 2000) -> Tuple[Optional[float], Optional[float]]:
            if not daily_pnls:
                return (None, None)
            dd = []
            for _ in range(int(n_boot)):
                samp = [float(random.choice(daily_pnls)) for _ in range(int(horizon_days))]
                dd.append(_max_drawdown(samp))
            return (float(np.mean(dd)), float(np.quantile(dd, 0.95)))

        # Window days para projeções (mesma lógica do 7.3, mas com fallback seguro)
        window_days = None
        try:
            # IMPORTANT: usar dias com dados (após exclusões), não lookback_days.
            window_days = float(audited_unique_days) if audited_unique_days else None
        except Exception:
            window_days = None
        if audited_span_days is not None:
            window_days = max(float(audited_span_days), float(window_days or 0.0))
        window_days = float(window_days or 1.0)
        horizon_days = 30.0

        def _proj_30(x: Optional[float]) -> Optional[float]:
            if x is None:
                return None
            return float(x) * (horizon_days / window_days)

        # Bancas de referência (proxy): p99 exposição observada no sizing proxy atual
        back_bank_ref_proxy = float(_pctl(back_stakes, 99) or 0.0)
        lay_bank_ref_proxy = float(_pctl(lay_liability, 99) or 0.0)
        # Escala do Kelly: por padrão usa p99 proxy; opcionalmente usa bankroll explícito.
        kelly_bankroll = _safe_float(getattr(args, "kelly_bankroll", None))
        use_bankroll = (kelly_bankroll is not None) and (float(kelly_bankroll) > 0)
        back_bank_ref = float(kelly_bankroll) if use_bankroll else float(back_bank_ref_proxy)
        lay_bank_ref = float(kelly_bankroll) if use_bankroll else float(lay_bank_ref_proxy)

        # caps fracionários (guardrail) — configuráveis
        BACK_CAP_FRAC = max(0.0, float(getattr(args, "kelly_back_cap_frac", 0.02)))
        LAY_CAP_FRAC = max(0.0, float(getattr(args, "kelly_lay_cap_frac", 0.01)))

        # cap por evento via limit (capacidade operacional)
        MAX_STAKE_PCT_OF_LIMIT = max(0.0, float(getattr(args, "max_stake_pct_of_limit", 1.0)))
        MAX_STAKE_CAP = max(0.0, float(getattr(args, "max_stake_cap", 0.0)))

        def _max_stake_from_limit(limit_value: float) -> float:
            # IMPORTANTE (robustez do turnover):
            # Em alguns períodos/versões o `limit` pode vir como 0.0 (não medido/indisponível),
            # o que não deve "matar" o sizing no OOS (especialmente em KELLY, que já é capado por fração da banca).
            try:
                lim = float(limit_value)
            except Exception:
                lim = 0.0
            if lim <= 0:
                # se houver cap absoluto, usa ele; senão retorna um teto alto para não limitar
                return float(MAX_STAKE_CAP) if float(MAX_STAKE_CAP) > 0 else float(1e18)
            s = float(lim) * float(MAX_STAKE_PCT_OF_LIMIT)
            if MAX_STAKE_CAP > 0:
                s = min(s, float(MAX_STAKE_CAP))
            return float(s)

        def _max_back_stake_event(d: dict) -> float:
            return _max_stake_from_limit(float(d.get("limit") or 0.0))

        def _max_lay_stake_event(d: dict) -> float:
            h = d.get("hypothesis_details") or {}
            lay_lim = _safe_float(_get_path(h, ["lay", "available_limit"]))
            if lay_lim is None:
                lay_lim = _safe_float(d.get("limit"))
            return _max_stake_from_limit(float(lay_lim or 0.0))

        def _sizing_back(d: dict, scheme: str, *, bank_ref: float) -> Optional[float]:
            """
            Retorna stake (em unidade monetária proxy).
            """
            if scheme == "FLAT":
                return 1.0
            if scheme == "PROXY":
                bs, _, _, _ = finance_for_row(d)
                return float(bs)
            if scheme.startswith("KELLY"):
                # Kelly/closing só é interpretável pre-match
                if d.get("is_live") is True:
                    return None
                # KELLY_xx: xx = fração (0.10, 0.25, 0.50)
                frac = float(scheme.split("_")[1])
                f = _kelly_back_frac(d.get("bs_odd"), d.get("closing_odd"))
                if f is None:
                    return None
                f = max(0.0, f) * frac
                cap = BACK_CAP_FRAC * max(1e-9, float(bank_ref))
                st = min(f * float(bank_ref), cap)
                # cap adicional por evento (limit)
                st = min(float(st), float(_max_back_stake_event(d)))
                return float(st)
            return None

        def _sizing_lay_liab(d: dict, scheme: str, *, bank_ref: float) -> Optional[Tuple[float, float]]:
            """
            Retorna (liability, lay_odd) em unidade monetária proxy.
            Para Lay, o sizing governado por liability.
            """
            h = d.get("hypothesis_details") or {}
            lay_odd = _safe_float(_get_path(h, ["lay", "odd"]))
            if lay_odd is None:
                lay_odd = _safe_float(d.get("bs_odd"))
            if lay_odd is None or lay_odd <= 1.0:
                return None
            if scheme == "FLAT":
                return (1.0, float(lay_odd))
            if scheme == "PROXY":
                _, _, _, ll = finance_for_row(d)
                return (float(ll), float(lay_odd))
            if scheme.startswith("KELLY"):
                # Kelly/closing só é interpretável pre-match
                if d.get("is_live") is True:
                    return None
                frac = float(scheme.split("_")[1])
                f = _kelly_lay_liab_frac(lay_odd, d.get("closing_odd"))
                if f is None:
                    return None
                f = max(0.0, f) * frac
                cap = LAY_CAP_FRAC * max(1e-9, float(bank_ref))
                liab = min(f * float(bank_ref), cap)
                # cap adicional por evento (limit): converte stake max -> liab max
                max_st = _max_lay_stake_event(d)
                liab = min(float(liab), float(max_st) * max(0.0, float(lay_odd) - 1.0))
                return (float(liab), float(lay_odd))
            return None

        # Frações a reportar (configurável por CLI)
        try:
            frac_list = [float(x.strip()) for x in str(getattr(args, "kelly_fractions", "")).split(",") if x.strip()]
        except Exception:
            frac_list = []
        frac_list = [f for f in frac_list if f > 0]
        if not frac_list:
            frac_list = [0.10, 0.25, 0.50, 1.00]

        schemes = ["FLAT", "PROXY"] + [f"KELLY_{f:.2f}" for f in frac_list]

        def _eval_back(rows_in: List[dict], scheme: str, *, bank_ref: float) -> Dict[str, Any]:
            pnls = []
            stakes = []
            mids = []
            by_day = {}
            for d in rows_in:
                roi = _safe_float(d.get("roi_bs"))
                if roi is None:
                    continue
                st = _sizing_back(d, scheme, bank_ref=float(bank_ref))
                if st is None or st <= 0:
                    continue
                pnl = _pnl_back(roi, st)
                if pnl is None:
                    continue
                pnls.append(float(pnl))
                stakes.append(float(st))
                mids.append(int(d.get("match_id")))
                ko = d.get("kickoff") or d.get("audited_at")
                if ko:
                    day = ko.astimezone(timezone.utc).strftime("%Y-%m-%d")
                    by_day[day] = by_day.get(day, 0.0) + float(pnl)
            turnover = float(sum(stakes)) if stakes else 0.0
            profit = float(sum(pnls)) if pnls else 0.0
            roi_turn = (profit / turnover * 100.0) if turnover > 0 else None
            dd_mean, dd_p95 = _bootstrap_dd(list(by_day.values()), horizon_days=30, n_boot=1200)
            return {
                "n": len(pnls),
                "turnover": turnover,
                "profit": profit,
                "roi_turn": roi_turn,
                "p99_stake": _pctl(stakes, 99),
                "es95_stake": _es_tail(stakes, 95),
                "dd_mean": dd_mean,
                "dd_p95": dd_p95,
            }

        def _eval_lay(rows_in: List[dict], scheme: str, *, bank_ref: float) -> Dict[str, Any]:
            pnls = []
            liabs = []
            stakes = []
            by_day = {}
            for d in rows_in:
                mult = _outcome_mult(str(d.get("line", "")), str(d.get("side", "")), d.get("home_score"), d.get("away_score"))
                if mult is None:
                    continue
                sized = _sizing_lay_liab(d, scheme, bank_ref=float(bank_ref))
                if not sized:
                    continue
                liab, lay_odd = sized
                if liab is None or liab <= 0:
                    continue
                pnl = _pnl_lay_from_liab(float(liab), float(lay_odd), mult)
                if pnl is None:
                    continue
                pnls.append(float(pnl))
                liabs.append(float(liab))
                st = float(liab) / max(1e-9, (float(lay_odd) - 1.0))
                stakes.append(st)
                ko = d.get("kickoff") or d.get("audited_at")
                if ko:
                    day = ko.astimezone(timezone.utc).strftime("%Y-%m-%d")
                    by_day[day] = by_day.get(day, 0.0) + float(pnl)
            turnover = float(sum(stakes)) if stakes else 0.0
            profit = float(sum(pnls)) if pnls else 0.0
            roi_turn = (profit / turnover * 100.0) if turnover > 0 else None
            roi_liab = (profit / float(sum(liabs)) * 100.0) if sum(liabs) > 0 else None
            dd_mean, dd_p95 = _bootstrap_dd(list(by_day.values()), horizon_days=30, n_boot=1200)
            return {
                "n": len(pnls),
                "turnover_stake": turnover,
                "exposure_liab": float(sum(liabs)) if liabs else 0.0,
                "profit": profit,
                "roi_turn": roi_turn,
                "roi_liab": roi_liab,
                "p99_liab": _pctl(liabs, 99),
                "es95_liab": _es_tail(liabs, 95),
                "dd_mean": dd_mean,
                "dd_p95": dd_p95,
            }

        # Baselines: coortes (com placar)
        back_finished = [d for d in back_edge if d.get("roi_bs") is not None]
        lay_finished = [d for d in lay_edge if d.get("home_score") is not None and d.get("away_score") is not None]

        lines.append("**Backtest de sizing (apenas eventos com placar; valores em unidade monetária *proxy*)**\n\n")
        lines.append("| Lado | Scheme | N | Turnover (janela) | Lucro (janela) | ROI/turnover | p99 exposição | ES95 exposição | DD 30d (média) | DD 30d (p95) |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for sc in schemes:
            eb = _eval_back(back_finished, sc, bank_ref=float(back_bank_ref))
            el = _eval_lay(lay_finished, sc, bank_ref=float(lay_bank_ref))
            lines.append(
                f"| Back | {sc} | {eb['n']} | {_fmt_num(eb['turnover'],2)} | {_fmt_num(eb['profit'],2)} | {_fmt_num(eb['roi_turn'],2)}% | "
                f"{_fmt_num(eb['p99_stake'],2)} | {_fmt_num(eb['es95_stake'],2)} | {_fmt_num(eb['dd_mean'],2)} | {_fmt_num(eb['dd_p95'],2)} |\n"
            )
            lines.append(
                f"| Lay | {sc} | {el['n']} | {_fmt_num(el['turnover_stake'],2)} | {_fmt_num(el['profit'],2)} | {_fmt_num(el['roi_turn'],2)}% | "
                f"{_fmt_num(el['p99_liab'],2)} | {_fmt_num(el['es95_liab'],2)} | {_fmt_num(el['dd_mean'],2)} | {_fmt_num(el['dd_p95'],2)} |\n"
            )

        # Requisito (relatório): repetir o backtest para 4 bancas explícitas.
        # Observação: FLAT/PROXY não dependem de banca; o impacto aparece principalmente nos KELLY_*.
        kelly_default = "KELLY_0.25" if "KELLY_0.25" in schemes else next((s for s in schemes if s.startswith("KELLY_")), None)
        schemes_4bank = [s for s in ["FLAT", "PROXY", kelly_default] if s]
        if kelly_default:
            lines.append("\n**Backtest de sizing por banca (10k/50k/100k/500k; foco em FLAT/PROXY/KELLY)**\n\n")
            for bank_ref_use in [10000, 50000, 100000, 500000]:
                lines.append(f"**Banca (ref) = {bank_ref_use:,}**\n\n".replace(",", "."))
                lines.append("| Lado | Scheme | N | Turnover (janela) | Lucro (janela) | ROI/turnover | p99 exposição | ES95 exposição | DD 30d (média) | DD 30d (p95) |\n")
                lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
                for sc in schemes_4bank:
                    eb = _eval_back(back_finished, sc, bank_ref=float(bank_ref_use))
                    el = _eval_lay(lay_finished, sc, bank_ref=float(bank_ref_use))
                    lines.append(
                        f"| Back | {sc} | {eb['n']} | {_fmt_num(eb['turnover'],2)} | {_fmt_num(eb['profit'],2)} | {_fmt_num(eb['roi_turn'],2)}% | "
                        f"{_fmt_num(eb['p99_stake'],2)} | {_fmt_num(eb['es95_stake'],2)} | {_fmt_num(eb['dd_mean'],2)} | {_fmt_num(eb['dd_p95'],2)} |\n"
                    )
                    lines.append(
                        f"| Lay | {sc} | {el['n']} | {_fmt_num(el['turnover_stake'],2)} | {_fmt_num(el['profit'],2)} | {_fmt_num(el['roi_turn'],2)}% | "
                        f"{_fmt_num(el['p99_liab'],2)} | {_fmt_num(el['es95_liab'],2)} | {_fmt_num(el['dd_mean'],2)} | {_fmt_num(el['dd_p95'],2)} |\n"
                    )
                lines.append("\n")
        lines.append(
            "\nLeitura:\n"
            "- Se `PROXY` piora ROI/turnover vs `FLAT`, isso indica que a política de stake atual está concentrando exposição em pontos com pior performance.\n"
            "- `KELLY_0.25` tende a ser um bom compromisso quando o edge é estimado por CLV, mas requer **caps** e só é aplicável quando há `closing_odd` (pre‑match).\n"
            "- Em Lay, é comum observar ROI alto por **liability**, mas sizing menor em **stake**: isso é uma decisão deliberada de governança de risco (liability tem cauda pior).\n"
            "- DD é estimado por bootstrap i.i.d de dias (aproximação). Para uma curva mais fiel, use bootstrap por dia com blocos maiores.\n\n"
        )

        # 9.3b) Sizing separado por estratégia (8 combinações)
        lines.append("### 9.3b Stake sizing por estratégia (8 combinações)\n")
        lines.append(
            "Abaixo repetimos o backtest de sizing **separado** por cada combinação `Side × Pre/In × Reversal`. "
            "Isso responde diretamente sua necessidade: **se várias combinações tiverem valor, o Kelly/caps deve ser calibrado por estratégia**.\n\n"
            "Observações:\n"
            "- Kelly é calculado **somente pre-match** (depende de `closing_odd`). Em combinações `In`, reportamos apenas `FLAT` e `PROXY`.\n"
            "- ROI do Lay é por **liability**; turnover é mostrado em stake equivalente.\n\n"
        )
        lines.append("| Side | Pre/In | Reversal | Scheme | N (placar) | Turnover | Lucro | ROI/turnover | p99 exp | DD30 p95 |\n|---|---|---|---|---:|---:|---:|---:|---:|---:|\n")

        # Edge sets coerentes com a entrada operacional (entry_odd) + filtro AH (quando habilitado),
        # para evitar incoerências entre esta tabela e a “política sugerida”.
        ah_max_93 = _safe_float(getattr(args, "wf_ah_max_abs_line", 0.0)) or 0.0
        ah_scope_93 = str(getattr(args, "wf_ah_scope", "all") or "all").strip().lower()
        def _ah_ok_row_93(d0: dict) -> bool:
            if ah_max_93 <= 0:
                return True
            abs_line = line_abs(d0.get("line"))
            if abs_line is None:
                return False
            if ah_scope_93 == "pre":
                # aplica só no pre-match
                if d0.get("is_live") is True:
                    return True
            return bool(float(abs_line) <= float(ah_max_93) + 1e-12)

        back_edge_ids_93 = set()
        for r in back_entries:
            aid = r.get("audit_id")
            if aid is None:
                continue
            d0 = by_audit_id.get(int(aid))
            if not d0 or str(d0.get("status", "")).upper() != "OK":
                continue
            if not _ah_ok_row_93(d0):
                continue
            diff = _safe_float(r.get("diff_entry"))
            if diff is None:
                continue
            if not (-10.0 <= float(diff) <= 10.0):
                continue
            if float(diff) >= float(back_cut):
                back_edge_ids_93.add(int(aid))

        lay_edge_ids_93 = set()
        for r in lay_entries:
            aid = r.get("audit_id")
            if aid is None:
                continue
            d0 = by_audit_id.get(int(aid))
            if not d0 or str(d0.get("status", "")).upper() != "OK":
                continue
            if not _ah_ok_row_93(d0):
                continue
            diff = _safe_float(r.get("diff_entry"))
            if diff is None:
                continue
            if not (-10.0 <= float(diff) <= 10.0):
                continue
            if float(diff) <= float(lay_cut):
                lay_edge_ids_93.add(int(aid))

        def _rows_for_combo(side: str, is_live_val: bool, had_rev_val: bool) -> List[dict]:
            # usa entries para filtrar e volta para o d original (necessário para sizing)
            ent = back_entries if side == "Back" else lay_entries
            flt = [r for r in ent if r.get("is_live") is is_live_val and bool(r.get("had_reversal")) is bool(had_rev_val)]
            out = []
            for r in flt:
                aid = r.get("audit_id")
                if aid is None:
                    continue
                d0 = by_audit_id.get(int(aid))
                if d0:
                    # filtra por coorte de edge (Back/Lay) para evitar misturar lados
                    if side == "Back" and int(aid) not in back_edge_ids_93:
                        continue
                    if side == "Lay" and int(aid) not in lay_edge_ids_93:
                        continue
                    out.append(d0)
            return out

        for side in ["Back", "Lay"]:
            for is_live_val, regime_label in [(False, "Pre"), (True, "In")]:
                for had_rev_val, rev_label in [(True, "Yes"), (False, "No")]:
                    rows_combo = _rows_for_combo(side, is_live_val, had_rev_val)
                    # apenas eventos com placar, como no backtest geral
                    if side == "Back":
                        rows_fin = [d for d in rows_combo if d.get("roi_bs") is not None]
                        for sc in (["FLAT", "PROXY"] + ([f"KELLY_{f:.2f}" for f in frac_list] if (is_live_val is False) else [])):
                            eb = _eval_back(rows_fin, sc, bank_ref=float(back_bank_ref))
                            lines.append(
                                f"| {side} | {regime_label} | {rev_label} | {sc} | {eb['n']} | {_fmt_num(eb['turnover'],2)} | {_fmt_num(eb['profit'],2)} | {_fmt_num(eb['roi_turn'],2)}% | {_fmt_num(eb['p99_stake'],2)} | {_fmt_num(eb['dd_p95'],2)} |\n"
                            )
                    else:
                        rows_fin = [d for d in rows_combo if d.get("home_score") is not None and d.get("away_score") is not None]
                        for sc in (["FLAT", "PROXY"] + ([f"KELLY_{f:.2f}" for f in frac_list] if (is_live_val is False) else [])):
                            el = _eval_lay(rows_fin, sc, bank_ref=float(lay_bank_ref))
                            lines.append(
                                f"| {side} | {regime_label} | {rev_label} | {sc} | {el['n']} | {_fmt_num(el['turnover_stake'],2)} | {_fmt_num(el['profit'],2)} | {_fmt_num(el['roi_turn'],2)}% | {_fmt_num(el['p99_liab'],2)} | {_fmt_num(el['dd_p95'],2)} |\n"
                            )

        # ============================================================
        # 9.4) Estratégias candidatas (com sizing recomendado)
        # ============================================================
        lines.append("### 9.4 Estratégias candidatas (combinações 8.3 + sizing recomendado)\n")
        lines.append(
            "Esta seção foi atualizada para refletir as **combinações** que você está analisando (Back/Lay × Pre/In × Reversal). "
            "Ela não assume mais apenas `BackFast` e `LayReversal`.\n\n"
            "**Política de entrada**:\n"
            "- Back: `t0`.\n"
            "- Lay: **após reversão** quando existir; senão no **último ponto** (~t+20s).\n\n"
            "**Política de sizing sugerida** (padrão):\n"
            "- Pre‑match: `KELLY_0.25` (com caps e cap por evento).\n"
            "- In‑match: `FLAT` ou `PROXY` capado, até existir um benchmark live (Kelly live não é confiável sem referência).\n\n"
        )

        # Estratégias candidatas = 8 combinações; aqui apenas listamos métricas de entrada (CLV/ROI) e N
        lines.append("| Side | Pre/In | Reversal | N (janela) | Jogos | CLV (entry; IC90) | ROI (entry; IC90) | ROI p30 | Observação |\n")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---|\n")
        for side, entries in [("Back", back_entries), ("Lay", lay_entries)]:
            for is_live_val, regime_label in [(False, "Pre"), (True, "In")]:
                for had_rev_val, rev_label in [(True, "Yes"), (False, "No")]:
                    filt = [r for r in entries if r.get("is_live") is is_live_val and bool(r.get("had_reversal")) is bool(had_rev_val)]
                    m_total = len(set(int(r.get("match_id")) for r in filt if r.get("match_id") is not None))
                    # CLV: pre-match only
                    clv_mean = clv_ci = None
                    if is_live_val is False:
                        clv_key = "clv_t0" if side == "Back" else "clv_conv_entry"
                        s_clv = summarize_metric([r.get(clv_key) for r in filt], [r.get("match_id") for r in filt], clip_low=-50, clip_high=50)
                        clv_mean, clv_ci = s_clv.mean_cluster, s_clv.ci90_cluster
                    # ROI: entry policy
                    roi_key = "roi_t0" if side == "Back" else "roi_entry"
                    s_roi = summarize_metric([r.get(roi_key) for r in filt], [r.get("match_id") for r in filt], clip_low=-200, clip_high=(500.0 if side == "Back" else 5000.0))
                    bym = {}
                    for v, mid in zip([r.get(roi_key) for r in filt], [r.get("match_id") for r in filt]):
                        vf = _safe_float(v)
                        if vf is None:
                            continue
                        bym.setdefault(int(mid), []).append(float(vf))
                    roi_p30 = cluster_bootstrap_quantile(bym, 0.30, n_boot=2000, seed=int(args.seed))
                    obs = "pre: Kelly OK" if regime_label == "Pre" else "in: use FLAT/PROXY"
                    lines.append(
                        f"| {side} | {regime_label} | {rev_label} | {len(filt)} | {m_total} | "
                        f"{_fmt_pct(clv_mean,2)} {_fmt_ci(clv_ci,2)} | {_fmt_pct(s_roi.mean_cluster,2)} {_fmt_ci(s_roi.ci90_cluster,2)} | {_fmt_pct(roi_p30,2)} | {obs} |\n"
                    )

        FX = float(args.fx_usdbrl or 5.20)

        def _liq_bank_from_sized(rows_b_all: List[dict], rows_l_all: List[dict], scheme: str, *, bank_ref_use: float) -> Optional[float]:
            """Banca por liquidez: p99 do capital simultaneamente travado (com buffer)."""
            # Reusa premissas de liquidez do 7.3 (defaults/env)
            settle_h = float(os.getenv("LIQUIDITY_SETTLE_BUFFER_HOURS", "2.25"))
            dur_h = float(os.getenv("LIQUIDITY_MATCH_DURATION_HOURS", "2.0"))
            grid_min = int(os.getenv("LIQUIDITY_GRID_MINUTES", "5"))
            buf_pct = float(os.getenv("LIQUIDITY_BANK_BUFFER_PCT", "10"))

            jobs: List[Tuple[datetime, datetime, float]] = []

            for d in rows_b_all:
                t0 = d.get("audited_at") or d.get("hypothesis_detected_at")
                if not isinstance(t0, datetime):
                    continue
                ko = d.get("kickoff") or d.get("kickoff_time")
                t1 = (ko + timedelta(hours=(dur_h + settle_h))) if isinstance(ko, datetime) else (t0 + timedelta(hours=(dur_h + settle_h)))
                if t1 <= t0:
                    t1 = t0 + timedelta(hours=(dur_h + settle_h))
                st = _sizing_back(d, scheme, bank_ref=float(bank_ref_use))
                if st is None or float(st) <= 0:
                    continue
                jobs.append((t0, t1, float(st)))

            for d in rows_l_all:
                t0 = d.get("audited_at") or d.get("hypothesis_detected_at")
                if not isinstance(t0, datetime):
                    continue
                ko = d.get("kickoff") or d.get("kickoff_time")
                t1 = (ko + timedelta(hours=(dur_h + settle_h))) if isinstance(ko, datetime) else (t0 + timedelta(hours=(dur_h + settle_h)))
                if t1 <= t0:
                    t1 = t0 + timedelta(hours=(dur_h + settle_h))
                sized = _sizing_lay_liab(d, scheme, bank_ref=float(bank_ref_use))
                if not sized:
                    continue
                liab, _ = sized
                if liab is None or float(liab) <= 0:
                    continue
                jobs.append((t0, t1, float(liab)))

            if not jobs:
                return None
            t_min = min(a for a, _, _ in jobs)
            t_max = max(b for _, b, _ in jobs)
            if t_max <= t_min:
                return None
            step = max(1, int(grid_min))
            vals: List[float] = []
            t = t_min
            while t <= t_max:
                s = 0.0
                for a, b, exp in jobs:
                    if a <= t < b:
                        s += float(exp)
                vals.append(float(s))
                t = t + timedelta(minutes=step)
            if not vals:
                return None
            p99 = float(np.quantile(vals, 0.99))
            return float(p99) * (1.0 + max(0.0, float(buf_pct)) / 100.0)

        def _summ_strategy(name: str, rows_b: List[dict], rows_l: List[dict], scheme: str):
            eb = _eval_back([d for d in rows_b if d.get("roi_bs") is not None], scheme, bank_ref=float(back_bank_ref))
            el = _eval_lay([d for d in rows_l if d.get("home_score") is not None and d.get("away_score") is not None], scheme, bank_ref=float(lay_bank_ref))

            back_n = int(eb["n"] or 0)
            lay_n = int(el["n"] or 0)
            back_n_30d = int(round((back_n / window_days) * horizon_days)) if window_days > 0 else None
            lay_n_30d = int(round((lay_n / window_days) * horizon_days)) if window_days > 0 else None

            turn_win = float(eb["turnover"] + el["turnover_stake"])
            prof_win = float(eb["profit"] + el["profit"])
            roi_turn_win = (prof_win / turn_win * 100.0) if turn_win > 0 else None
            lay_roi_liab_win = el.get("roi_liab")
            lay_roi_turn_win = el.get("roi_turn")

            turn_30d = _proj_30(turn_win)
            profit_30d = _proj_30(prof_win)

            # stake/liability médios na janela (proxy)
            stake_avg_back = (float(eb["turnover"]) / back_n) if back_n > 0 else None
            stake_avg_lay = (float(el["turnover_stake"]) / lay_n) if lay_n > 0 else None
            liab_avg_lay = (float(el["exposure_liab"]) / lay_n) if lay_n > 0 else None

            # Banca por risco (unitária) deve vir do sizing (não só do subconjunto com ROI).
            risk_back = []
            for d in rows_b:
                st = _sizing_back(d, scheme, bank_ref=float(back_bank_ref))
                if st is not None and float(st) > 0:
                    risk_back.append(float(st))
            risk_lay = []
            for d in rows_l:
                sized = _sizing_lay_liab(d, scheme, bank_ref=float(lay_bank_ref))
                if sized and sized[0] is not None and float(sized[0]) > 0:
                    risk_lay.append(float(sized[0]))
            bank_back_p99 = _pctl(risk_back, 99) if risk_back else None
            bank_lay_p99 = _pctl(risk_lay, 99) if risk_lay else None
            bank_risk_total = None
            if bank_back_p99 is not None or bank_lay_p99 is not None:
                bank_risk_total = float(bank_back_p99 or 0.0) + float(bank_lay_p99 or 0.0)

            bank_ref_liq = float(kelly_bankroll) if use_bankroll else float(max(float(back_bank_ref), float(lay_bank_ref)))
            bank_liq_total = _liq_bank_from_sized(rows_b, rows_l, scheme, bank_ref_use=bank_ref_liq)
            bank_eff = None
            if bank_risk_total is not None or bank_liq_total is not None:
                bank_eff = max(float(bank_risk_total or 0.0), float(bank_liq_total or 0.0))

            roi_bank_30d = (float(profit_30d) / float(bank_eff) * 100.0) if (profit_30d is not None and bank_eff and bank_eff > 0) else None

            return {
                "name": name,
                "scheme": scheme,
                # N: sempre na janela; também reportamos projeção simples 30d
                "back_n": back_n,
                "lay_n": lay_n,
                "back_n_30d": back_n_30d,
                "lay_n_30d": lay_n_30d,
                # stake sizing médio (janela)
                "stake_avg_back": stake_avg_back,
                "stake_avg_lay": stake_avg_lay,
                "liab_avg_lay": liab_avg_lay,
                # janela
                "turn_win": turn_win,
                "prof_win": prof_win,
                "roi_turn_win": roi_turn_win,
                "lay_roi_liab_win": lay_roi_liab_win,
                "lay_roi_turn_win": lay_roi_turn_win,
                # 30d
                "turn_30d": turn_30d,
                "profit_30d": profit_30d,
                "bank_risk_p99": bank_risk_total,
                "bank_liq_p99": bank_liq_total,
                "bank_eff": bank_eff,
                "roi_bank_30d": roi_bank_30d,
                # risco
                "back_bank_p99": bank_back_p99,
                "lay_bank_p99": bank_lay_p99,
                "dd_p95": max([x for x in [eb["dd_p95"], el["dd_p95"]] if x is not None], default=None),
                # BRL
                "turn_30d_brl": (float(turn_30d) * FX) if turn_30d is not None else None,
                "profit_30d_brl": (float(profit_30d) * FX) if profit_30d is not None else None,
                "bank_eff_brl": (float(bank_eff) * FX) if bank_eff is not None else None,
            }

        lines.append("**Tabela política de sizing sugerida — resumo executivo (30d)**\n\n")
        lines.append("| Estratégia | Scheme | Turnover 30d | Lucro 30d | Banca rec. (max) | ROI/banca 30d | DD 30d p95 |\n")
        lines.append("|---|---|---:|---:|---:|---:|---:|\n")
        # Consolida estratégia pre-match como união das combinações ativas (8.3) — somente PRE (Kelly depende de closing).
        by_audit_id_94: Dict[int, dict] = {int(d.get("id")): d for d in ok_bs if d.get("id") is not None}

        def _combo_active_83(side: str, had_rev: bool) -> bool:
            # Aplica os mesmos critérios da tabela 8.3 (PRE apenas)
            ent = back_entries if side == "Back" else lay_entries
            filt = [r for r in ent if r.get("is_live") is False and bool(r.get("had_reversal")) is bool(had_rev)]
            if not filt:
                return False
            # Filtro AH (alinha com OOS quando configurado)
            ah_max = _safe_float(getattr(args, "wf_ah_max_abs_line", 0.0)) or 0.0
            ah_scope = str(getattr(args, "wf_ah_scope", "all") or "all").strip().lower()
            if ah_max > 0:
                # Para Back/Lay pre-match, aplicamos sempre no PRE; para IN só se scope=all (aqui é PRE apenas).
                # As entradas não carregam a linha; então filtramos via lookup do audit.
                filt2 = []
                for r in filt:
                    aid = r.get("audit_id")
                    d0 = by_audit_id_94.get(int(aid)) if (aid is not None) else None
                    if not d0:
                        continue
                    abs_line = line_abs(d0.get("line"))
                    if abs_line is None:
                        continue
                    if float(abs_line) <= float(ah_max) + 1e-12:
                        filt2.append(r)
                filt = filt2
                if not filt:
                    return False
            # CLV
            clv_key = "clv_entry" if side == "Back" else "clv_conv_entry"
            s_clv = summarize_metric([r.get(clv_key) for r in filt], [r.get("match_id") for r in filt], clip_low=-50, clip_high=50)
            # ROI (entry)
            roi_key = "roi_entry" if side == "Back" else "roi_entry"
            s_roi = summarize_metric([r.get(roi_key) for r in filt], [r.get("match_id") for r in filt], clip_low=-200, clip_high=(500.0 if side == "Back" else 5000.0))
            if side == "Back":
                clv_ok = bool(s_clv.ci90_cluster and float(s_clv.ci90_cluster[0]) > 0)
                roi_ok = bool(s_roi.mean_cluster is not None and float(s_roi.mean_cluster) > 0)
                return bool(clv_ok and roi_ok)
            # Lay: CLV_CONV bom tende a ser POSITIVO nesta convenção (closing - entry)/closing
            clv_ok = bool(s_clv.ci90_cluster and float(s_clv.ci90_cluster[0]) > 0)
            # “sig a p30” ≈ p30 > 0
            bym = {}
            for v, mid in zip([r.get(roi_key) for r in filt], [r.get("match_id") for r in filt]):
                vf = _safe_float(v)
                if vf is None:
                    continue
                bym.setdefault(int(mid), []).append(float(vf))
            roi_p30 = cluster_bootstrap_quantile(bym, 0.30, n_boot=2000, seed=int(args.seed))
            roi_ok = bool(roi_p30 is not None and float(roi_p30) > 0)
            return bool(clv_ok and roi_ok)

        active_back_aids = set()
        for had_rev in (True, False):
            if _combo_active_83("Back", had_rev):
                active_back_aids.update(int(r.get("audit_id")) for r in back_entries if r.get("is_live") is False and bool(r.get("had_reversal")) is bool(had_rev) and r.get("audit_id") is not None)
        active_lay_aids = set()
        for had_rev in (True, False):
            if _combo_active_83("Lay", had_rev):
                active_lay_aids.update(int(r.get("audit_id")) for r in lay_entries if r.get("is_live") is False and bool(r.get("had_reversal")) is bool(had_rev) and r.get("audit_id") is not None)

        rows_back_active = [by_audit_id_94[aid] for aid in active_back_aids if aid in by_audit_id_94 and aid in back_edge_ids]
        rows_lay_active = [by_audit_id_94[aid] for aid in active_lay_aids if aid in by_audit_id_94 and aid in lay_edge_ids]

        base_schemes_94 = ["FLAT"] + [f"KELLY_{f:.2f}" for f in frac_list if abs(f - 0.25) < 1e-9]
        if len(base_schemes_94) == 1:
            base_schemes_94.append("KELLY_0.25")
        for sc in base_schemes_94:
            s1 = _summ_strategy("Ativas (PRE, critérios 8.3)", rows_back_active, rows_lay_active, sc)
            lines.append(
                f"| {s1['name']} | {sc} | {_fmt_num(s1['turn_30d'],2)} | {_fmt_num(s1['profit_30d'],2)} | "
                f"{_fmt_num(s1['bank_eff'],2)} | {_fmt_num(s1['roi_bank_30d'],2)}% | {_fmt_num(s1['dd_p95'],2)} |\n"
            )
        lines.append("\n**Tabela política de sizing sugerida — detalhe (volume/sizing/risco)**\n\n")
        lines.append("| Estratégia | Scheme | N Back | N Lay | N Back 30d | N Lay 30d | Stake méd Back | Stake méd Lay | Liab méd Lay |\n")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for sc in base_schemes_94:
            s1 = _summ_strategy("Ativas (PRE, critérios 8.3)", rows_back_active, rows_lay_active, sc)
            lines.append(
                f"| {s1['name']} | {sc} | {s1['back_n']} | {s1['lay_n']} | {s1['back_n_30d']} | {s1['lay_n_30d']} | "
                f"{_fmt_num(s1['stake_avg_back'],2)} | {_fmt_num(s1['stake_avg_lay'],2)} | {_fmt_num(s1['liab_avg_lay'],2)} |\n"
            )
        lines.append("\n")
        lines.append("| Estratégia | Scheme | ROI/turnover (janela) | ROI Lay/liab (janela) | Banca risco p99 | Banca liq p99 | Banca rec. (max) |\n")
        lines.append("|---|---|---:|---:|---:|---:|---:|\n")
        for sc in base_schemes_94:
            s1 = _summ_strategy("Ativas (PRE, critérios 8.3)", rows_back_active, rows_lay_active, sc)
            lines.append(
                f"| {s1['name']} | {sc} | {_fmt_num(s1['roi_turn_win'],2)}% | {_fmt_num(s1['lay_roi_liab_win'],2)}% | "
                f"{_fmt_num(s1['bank_risk_p99'],2)} | {_fmt_num(s1['bank_liq_p99'],2)} | {_fmt_num(s1['bank_eff'],2)} |\n"
            )
        lines.append("\n")
        lines.append(
            "\nNotas:\n"
            "- **N Back/Lay** na tabela é **na janela observada**. As colunas `N 30d (proj.)` são uma **escala linear** por dias observados.\n"
            "- Esta linha usa a união das **combinações pre‑match ativas** sob os critérios da 8.3 (é um resumo, não substitui a tabela 8.3).\n"
            "- `Stake médio` e `Liability média` são **proxies** na unidade monetária do seu limit/finance; valores em **R$** usam fx `--fx-usdbrl`.\n"
            "- `Banca recomendada` = max( banca por risco p99 (unitária, soma) ; banca por liquidez p99 (+buffer) ).\n"
            "- **Por que Lay pode ter stake médio menor mesmo com ROI maior**: (i) Lay é governado por **liability** (risco) e usamos cap mais conservador; "
            "(ii) stake em Lay é derivado de `liability/(odd-1)` e depende do nível médio de odds; (iii) Kelly depende do **edge vs closing** (gap entry→closing), "
            "não do ROI observado isoladamente.\n"
            "- **Por que não incluímos Back in‑match aqui**: Kelly/CLV usam `closing_odd` pré‑jogo; in‑match requer benchmark diferente (ex.: referência por minuto/VWAP) e "
            "governança operacional específica. A seção 2.2 já compara PRE_MATCH vs IN_MATCH.\n"
        )

        # ------------------------------------------------------------
        # 9.4b) Curva de capacidade (fração de Kelly) — tamanho potencial
        # ------------------------------------------------------------
        lines.append("\n### 9.4b Curva de capacidade — frações de Kelly e tamanho potencial\n")
        lines.append(
            "Esta tabela é um exercício para estimar **capacidade** (turnover/lucro/risco) ao variar a fração de Kelly. "
            "Ela deixa explícito quando o sizing satura por **cap por aposta**.\n\n"
        )

        def _cap_rate_back(rows_b_all: List[dict], scheme: str) -> Optional[float]:
            if not str(scheme).startswith("KELLY"):
                return None
            try:
                frac = float(str(scheme).split("_")[1])
            except Exception:
                return None
            cap = BACK_CAP_FRAC * max(1e-9, float(back_bank_ref))
            hits = 0
            n = 0
            for d in rows_b_all:
                f0 = _kelly_back_frac(d.get("bs_odd"), d.get("closing_odd"))
                if f0 is None:
                    continue
                f = max(0.0, float(f0)) * float(frac)
                raw = f * float(back_bank_ref)
                if raw > cap + 1e-12:
                    hits += 1
                n += 1
            return (100.0 * hits / n) if n > 0 else None

        def _cap_rate_lay(rows_l_all: List[dict], scheme: str) -> Optional[float]:
            if not str(scheme).startswith("KELLY"):
                return None
            try:
                frac = float(str(scheme).split("_")[1])
            except Exception:
                return None
            cap = LAY_CAP_FRAC * max(1e-9, float(lay_bank_ref))
            hits = 0
            n = 0
            for d in rows_l_all:
                h = d.get("hypothesis_details") or {}
                lay_odd = _safe_float(_get_path(h, ["lay", "odd"])) or _safe_float(d.get("bs_odd"))
                if lay_odd is None:
                    continue
                f0 = _kelly_lay_liab_frac(lay_odd, d.get("closing_odd"))
                if f0 is None:
                    continue
                f = max(0.0, float(f0)) * float(frac)
                raw = f * float(lay_bank_ref)
                if raw > cap + 1e-12:
                    hits += 1
                n += 1
            return (100.0 * hits / n) if n > 0 else None

        def _hit_rates_back(rows_b_all: List[dict], scheme: str) -> Tuple[Optional[float], Optional[float]]:
            """Retorna (cap_hit_pct, limit_hit_pct) para Back em Kelly."""
            if not str(scheme).startswith("KELLY"):
                return (None, None)
            try:
                frac = float(str(scheme).split("_")[1])
            except Exception:
                return (None, None)
            cap = BACK_CAP_FRAC * max(1e-9, float(back_bank_ref))
            cap_hits = 0
            lim_hits = 0
            n = 0
            for d in rows_b_all:
                f0 = _kelly_back_frac(d.get("bs_odd"), d.get("closing_odd"))
                if f0 is None:
                    continue
                f = max(0.0, float(f0)) * float(frac)
                raw = f * float(back_bank_ref)
                st1 = min(raw, cap)
                lim = float(_max_back_stake_event(d))
                st2 = min(st1, lim)
                if raw > cap + 1e-12:
                    cap_hits += 1
                if st1 > lim + 1e-12:
                    lim_hits += 1
                n += 1
            return ((100.0 * cap_hits / n) if n > 0 else None, (100.0 * lim_hits / n) if n > 0 else None)

        def _hit_rates_lay(rows_l_all: List[dict], scheme: str) -> Tuple[Optional[float], Optional[float]]:
            """Retorna (cap_hit_pct, limit_hit_pct) para Lay em Kelly (liability)."""
            if not str(scheme).startswith("KELLY"):
                return (None, None)
            try:
                frac = float(str(scheme).split("_")[1])
            except Exception:
                return (None, None)
            cap = LAY_CAP_FRAC * max(1e-9, float(lay_bank_ref))
            cap_hits = 0
            lim_hits = 0
            n = 0
            for d in rows_l_all:
                h = d.get("hypothesis_details") or {}
                lay_odd = _safe_float(_get_path(h, ["lay", "odd"])) or _safe_float(d.get("bs_odd"))
                if lay_odd is None:
                    continue
                f0 = _kelly_lay_liab_frac(lay_odd, d.get("closing_odd"))
                if f0 is None:
                    continue
                f = max(0.0, float(f0)) * float(frac)
                raw = f * float(lay_bank_ref)
                liab1 = min(raw, cap)
                max_st = float(_max_lay_stake_event(d))
                lim = float(max_st) * max(0.0, float(lay_odd) - 1.0)
                liab2 = min(liab1, lim)
                if raw > cap + 1e-12:
                    cap_hits += 1
                if liab1 > lim + 1e-12:
                    lim_hits += 1
                n += 1
            return ((100.0 * cap_hits / n) if n > 0 else None, (100.0 * lim_hits / n) if n > 0 else None)

        lines.append(
            f"**Escala Kelly usada nesta curva**: {'BANKROLL' if use_bankroll else 'P99_PROXY'} | "
            f"ref_back={_fmt_num(back_bank_ref,2)} ref_lay={_fmt_num(lay_bank_ref,2)} | "
            f"cap_back={BACK_CAP_FRAC:.1%} cap_lay={LAY_CAP_FRAC:.1%} | max_stake_event={MAX_STAKE_PCT_OF_LIMIT:.0%}*limit\n\n"
        )
        lines.append(
            "| Strategy | Scheme | cap_hit Back (%) | limit_hit Back (%) | cap_hit Lay (%) | limit_hit Lay (%) | "
            "Turnover 30d (proj.) | Lucro 30d (proj.) | ROI/banca 30d | DD 30d p95 |\n"
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        for sc in [f"KELLY_{f:.2f}" for f in frac_list]:
            s = _summ_strategy("Ativas (PRE, critérios 8.3)", rows_back_active, rows_lay_active, sc)
            crb, lrb = _hit_rates_back(rows_back_active, sc)
            crl, lrl = _hit_rates_lay(rows_lay_active, sc)
            lines.append(
                f"| {s['name']} | {sc} | {_fmt_num(crb,1)}% | {_fmt_num(lrb,1)}% | {_fmt_num(crl,1)}% | {_fmt_num(lrl,1)}% | "
                f"{_fmt_num(s['turn_30d'],2)} | {_fmt_num(s['profit_30d'],2)} | {_fmt_num(s['roi_bank_30d'],2)}% | {_fmt_num(s['dd_p95'],2)} |\n"
            )
        lines.append(
            "\nLeitura rápida:\n"
            "- Se `cap_hit` estiver alto, a alavancagem adicional (ex.: 0,50× ou 1,00×) não vira turnover — o cap está “travando” o tamanho.\n"
            "- Se `limit_hit` estiver alto, você está batendo no **limit operacional** (capacidade da conta/mercado) — aumentar banca ou frac não aumenta turnover.\n"
            "- Se `cap_hit` estiver baixo, a curva deve escalar quase linearmente com `frac` (até bater em limites reais/operacionais).\n"
            "- Lay normalmente satura antes por ter cap mais conservador (liability) e por ter risco de cauda mais assimétrico.\n\n"
        )

        # ------------------------------------------------------------
        # 9.4c) Diagnóstico in-match (Back): por que não entra na estratégia
        # ------------------------------------------------------------
        lines.append("### 9.4c Diagnóstico: Back in‑match (por que não está na estratégia candidata)\n")
        lines.append(
            "Aqui reportamos Back in‑match **apenas como diagnóstico**. "
            "Não incluímos in‑match na estratégia candidata porque (i) `closing_odd` pré‑jogo não é benchmark in‑match e "
            "(ii) o sizing Kelly acima depende desse benchmark.\n\n"
        )
        strat_back_im = [d for d in back_edge if str(d.get("exec_bucket")) == "< 5s" and d.get("is_live") is True]
        eb_flat_im = _eval_back([d for d in strat_back_im if d.get("roi_bs") is not None], "FLAT", bank_ref=float(back_bank_ref))
        eb_proxy_im = _eval_back([d for d in strat_back_im if d.get("roi_bs") is not None], "PROXY", bank_ref=float(back_bank_ref))
        lines.append("| Regime | Scheme | N | Turnover (janela) | Lucro (janela) | ROI/turnover |\n|---|---|---:|---:|---:|---:|\n")
        lines.append(f"| IN_MATCH BackFast (<5s) | FLAT | {eb_flat_im['n']} | {_fmt_num(eb_flat_im['turnover'],2)} | {_fmt_num(eb_flat_im['profit'],2)} | {_fmt_num(eb_flat_im['roi_turn'],2)}% |\n")
        lines.append(f"| IN_MATCH BackFast (<5s) | PROXY | {eb_proxy_im['n']} | {_fmt_num(eb_proxy_im['turnover'],2)} | {_fmt_num(eb_proxy_im['profit'],2)} | {_fmt_num(eb_proxy_im['roi_turn'],2)}% |\n")
        lines.append(
            "\nPróximo passo (se quiser operar in‑match): definir um benchmark de preço justo in‑match (ex.: referência por minuto, VWAP, ou odds externas) "
            "e calibrar sizing/risco específico do live.\n\n"
        )

        # ============================================================
        # 12) OOS rolling-forward (walk-forward)
        # ============================================================
        wf_train_mode_hdr = str(getattr(args, "wf_train_mode", "rolling") or "rolling").strip().lower()
        if wf_train_mode_hdr not in ("rolling", "expanding"):
            wf_train_mode_hdr = "rolling"
        wf_mode_label = "expanding window" if wf_train_mode_hdr == "expanding" else "rolling window"
        lines.append(f"## 12) OOS walk-forward ({wf_mode_label}): seleção e validação\n")
        lines.append(
            "Até aqui o relatório é **in-sample** (na janela `--lookback-days`). "
            "Este bloco (opcional) faz um walk-forward por dia:\n\n"
            f"- **Train mode**: `{wf_train_mode_hdr}`.\n"
            "- Em cada passo, usamos uma janela de treino para **selecionar** combinações (Side×Pre/In×Reversal) com evidência de valor.\n"
            "  - `rolling`: usa os **últimos** `wf_train_days`.\n"
            "  - `expanding`: usa **todos os dias anteriores** (com `wf_train_days` só definindo quando o teste começa).\n"
            "- No(s) dia(s) seguinte(s) (`wf_test_days`), medimos o resultado OOS nas combinações ativas.\n\n"
            "**Evidência de valor (por combinação, no treino)** (atualizado para dar mais peso a ROI):\n"
            "- Elegibilidade (todas): `N_ROI >= wf_min_matches` (jogos com ROI na janela de treino). Se `wf_min_matches=0`, o mínimo fica desabilitado.\n"
            "- **Regra de bloqueio**: se `ROI` for **significativamente negativo** (IC90 inteiro < 0), **não ativa**.\n"
            "- Se `ROI` for **significativamente positivo** (IC90 inteiro > 0), **ativa**.\n"
            "- Caso `ROI` seja **>0 mas não sig.**:\n"
            "  - Pre-match: ativa apenas se `CLV > 0` (não precisa ser sig.)\n"
            "  - In-match: ativa se `ROI > 0` (não precisa ser sig.; CLV não é aplicável)\n\n"
            f"- **Step do WF**: `wf_step_days={int(getattr(args,'wf_step_days',1))}`. Se `wf_test_days>1` e `wf_step_days=1`, os test windows ficam **sobrepostos**; nesse caso, "
            "os lucros/prejuízos por linha não são somáveis. Para não sobrepor: use `--wf-step-days` igual a `--wf-test-days`.\n\n"
            "Isso aproxima o fluxo operacional que você descreveu (seleciona no passo atual e mede no(s) próximo(s) dia(s)).\n\n"
        )

        if not bool(getattr(args, "walkforward", False)):
            lines.append(
                "Walk-forward **desligado**. Para habilitar: rode com `--walkforward` "
                "(e ajuste `--wf-train-days/--wf-test-days/--wf-min-matches` se quiser).\n\n"
            )
        else:
            wf_train = int(max(1, getattr(args, "wf_train_days", 2)))
            wf_test = int(max(1, getattr(args, "wf_test_days", 1)))
            wf_step = int(max(1, getattr(args, "wf_step_days", 1)))
            # `wf_min_matches=0` desabilita a elegibilidade por volume no OOS.
            wf_min_m = int(max(0, getattr(args, "wf_min_matches", 20)))
            wf_train_mode = str(getattr(args, "wf_train_mode", "rolling") or "rolling").strip().lower()
            if wf_train_mode not in ("rolling", "expanding"):
                wf_train_mode = "rolling"
            wf_scheme_pre = str(getattr(args, "wf_scheme_pre", "KELLY_0.25") or "KELLY_0.25").strip()
            wf_scheme_in = str(getattr(args, "wf_scheme_in", "FLAT") or "FLAT").strip()
            wf_expand = bool(getattr(args, "wf_expand_missing_roi", True))
            bud_back_frac = max(0.0, float(getattr(args, "wf_budget_back_frac", 0.01)))
            bud_lay_frac = max(0.0, float(getattr(args, "wf_budget_lay_frac", 0.005)))
            bud_cap_sig_frac = max(0.0, float(getattr(args, "wf_budget_cap_signal_frac", 0.33)))
            bud_risk_mode = str(getattr(args, "wf_budget_risk_mode", "fixed") or "fixed").strip()
            wf_flat_stake_back = max(0.0, float(getattr(args, "wf_flat_stake_back", 1.0)))
            wf_flat_liab_lay = max(0.0, float(getattr(args, "wf_flat_liab_lay", 1.0)))
            wf_ws_proxy_offset = max(0.0, float(getattr(args, "wf_ws_proxy_offset_sec", 5.0)))
            wf_ws_proxy_max_gap = max(0.0, float(getattr(args, "wf_ws_proxy_max_gap_sec", 2.5)))
            wf_excl_exec_all = {x.strip() for x in str(getattr(args, "wf_exclude_exec_buckets", "") or "").split(",") if x.strip()}
            wf_excl_exec_back = {x.strip() for x in str(getattr(args, "wf_exclude_exec_buckets_back", "") or "").split(",") if x.strip()}
            wf_excl_exec_lay = {x.strip() for x in str(getattr(args, "wf_exclude_exec_buckets_lay", "") or "").split(",") if x.strip()}
            wf_excl_exec_back = set(wf_excl_exec_back) | set(wf_excl_exec_all)
            wf_excl_exec_lay = set(wf_excl_exec_lay) | set(wf_excl_exec_all)
            wf_use_shrink = bool(getattr(args, "wf_shrinkage", False))
            wf_liq_mode = str(getattr(args, "wf_liquidity_mode", "none") or "none").strip().lower()
            if wf_liq_mode not in ("none", "gate_p50", "gate_p75", "gate_min"):
                wf_liq_mode = "none"
            wf_liq_scope = str(getattr(args, "wf_liquidity_scope", "pre") or "pre").strip().lower()
            if wf_liq_scope not in ("pre", "all"):
                wf_liq_scope = "pre"
            wf_liq_min = float(max(0.0, _safe_float(getattr(args, "wf_liquidity_min_limit", 0.0)) or 0.0))
            wf_ah_max_abs = float(max(0.0, _safe_float(getattr(args, "wf_ah_max_abs_line", 0.0)) or 0.0))
            wf_ah_scope = str(getattr(args, "wf_ah_scope", "all") or "all").strip().lower()
            if wf_ah_scope not in ("pre", "all"):
                wf_ah_scope = "all"

            def _liq_ok(ev: dict, *, thr: Optional[float]) -> bool:
                if wf_liq_mode == "none":
                    return True
                if wf_liq_scope == "pre" and str(ev.get("regime")) != "Pre":
                    return True
                lim = _safe_float(ev.get("liq_limit"))
                if lim is None or not math.isfinite(float(lim)) or float(lim) <= 0:
                    return False
                if wf_liq_mode == "gate_min":
                    return float(lim) >= float(wf_liq_min)
                if thr is None or not math.isfinite(float(thr)) or float(thr) <= 0:
                    # sem limiar estimável => não filtra
                    return True
                return float(lim) >= float(thr)

            def _ah_ok(ev: dict) -> bool:
                if wf_ah_max_abs <= 0:
                    return True
                if wf_ah_scope == "pre" and str(ev.get("regime")) != "Pre":
                    return True
                a = _safe_float(ev.get("ah_abs"))
                if a is None or not math.isfinite(float(a)):
                    return False
                return float(a) <= float(wf_ah_max_abs)

            # Para scheme in-match "ROI_TRAIN": usamos ROI médio no treino por combinação como proxy de EV.
            # Mapeamento preenchido por step (train window) e usado no loop de sizing do test.
            roi_train_by_key: Dict[str, float] = {}

            # index para recuperar campos de sizing/liquidez (inclui ws-only)
            audit_by_id: Dict[int, dict] = {int(d.get("id")): d for d in ok_any if d.get("id") is not None}

            # dataset de entradas com timestamp
            combo_events: List[dict] = []
            if wf_excl_exec_back or wf_excl_exec_lay:
                lines.append(
                    f"**Filtro operacional (OOS)**: excluindo exec_bucket apenas no walk-forward "
                    f"(Back={sorted(wf_excl_exec_back) if wf_excl_exec_back else '—'}; "
                    f"Lay={sorted(wf_excl_exec_lay) if wf_excl_exec_lay else '—'}).\n\n"
                )
            if wf_liq_mode != "none":
                lines.append(
                    f"**Política de liquidez (OOS)**: mode=`{wf_liq_mode}`, scope=`{wf_liq_scope}`"
                    + (f", min_limit={_fmt_num(wf_liq_min,2)}" if wf_liq_mode == "gate_min" else "")
                    + ".\n\n"
                )
            if wf_ah_max_abs and float(wf_ah_max_abs) > 0:
                lines.append(
                    f"**Política por linha AH (OOS)**: max_abs_line={_fmt_num(wf_ah_max_abs,2)} (scope=`{wf_ah_scope}`).\n\n"
                )
            def _ws_proxy_odd(d0: dict) -> Optional[Tuple[float, float]]:
                """
                Retorna (odd, t_s) do ponto de entrada via WS para o proxy da execução.

                Compatibilidade (migração BS -> WS):
                - Preferimos `hypothesis_details.ws_series` quando existir.
                - Se `ws_series` não existir, mas o pipeline salvou pontos WS em `hypothesis_details.temporal`,
                  aceitamos `ws_odd` (fallback: `bs_odd`) com `t`.

                Regra de seleção do ponto:
                - Preferimos o primeiro ponto com t>=offset (simula "entrada após lag"),
                  desde que |t-offset| <= max_gap; caso contrário usamos o ponto mais próximo
                  (desde que também esteja dentro de max_gap).
                """
                h = d0.get("hypothesis_details") or {}
                pts: List[Tuple[float, float]] = []

                # 1) ws_series (formato padrão)
                arr = h.get("ws_series") if isinstance(h, dict) else None
                if isinstance(arr, list) and arr:
                    for e in arr:
                        if not isinstance(e, dict):
                            continue
                        t = _safe_float(e.get("t_actual_s"))
                        if t is None:
                            t = _safe_float(e.get("t_target_s"))
                        odd = _safe_float(e.get("ws_odd"))
                        if t is None or odd is None or odd <= 0:
                            continue
                        pts.append((float(t), float(odd)))

                # 2) temporal (fallback pós-mudança: alguns pipelines salvam ws_odd aqui)
                if not pts:
                    arr2 = h.get("temporal") if isinstance(h, dict) else None
                    if isinstance(arr2, list) and arr2:
                        for e in arr2:
                            if not isinstance(e, dict):
                                continue
                            t = _safe_float(e.get("t"))
                            odd = _safe_float(e.get("ws_odd"))
                            if odd is None:
                                odd = _safe_float(e.get("bs_odd"))
                            if t is None or odd is None or odd <= 0:
                                continue
                            if float(t) <= 0.0005:
                                continue
                            pts.append((float(t), float(odd)))

                if not pts:
                    return None
                pts.sort(key=lambda x: x[0])

                # preferir t>=offset (primeiro)
                candidates = [(t, o) for (t, o) in pts if t >= float(wf_ws_proxy_offset)]
                best = None
                if candidates:
                    t, o = candidates[0]
                    if abs(float(t) - float(wf_ws_proxy_offset)) <= float(wf_ws_proxy_max_gap):
                        best = (o, t)
                if best is None:
                    # ponto mais próximo
                    t, o = min(pts, key=lambda x: abs(float(x[0]) - float(wf_ws_proxy_offset)))
                    if abs(float(t) - float(wf_ws_proxy_offset)) <= float(wf_ws_proxy_max_gap):
                        best = (o, t)
                return best

            # --- Back: inclui v4.0-api (BS real) e v5.0-ws-only (proxy WS@t+offset) ---
            def _is_live_eff(d: dict, *, ts: Optional[datetime] = None) -> bool:
                """
                Robustez: quando a instrumentação muda (ex.: BS -> WS), o campo `is_live` pode vir inconsistente.
                Se `kickoff_time`/`kickoff` estiver disponível, inferimos `is_live` por timestamp (`audited_at >= kickoff`).
                """
                try:
                    ko = d.get("kickoff") or d.get("kickoff_time")
                    if ts is not None and isinstance(ko, datetime):
                        return bool(ts >= ko)
                except Exception:
                    pass
                try:
                    if d.get("is_live") is True:
                        return True
                    if d.get("is_live") is False:
                        return False
                except Exception:
                    pass
                return False

            for d0 in all_data:
                if str(d0.get("status", "")).upper() != "OK":
                    continue
                aid = d0.get("id")
                if aid is None:
                    continue
                if wf_excl_exec_back and str(d0.get("exec_bucket")) in wf_excl_exec_back:
                    continue
                ts = d0.get("audited_at")
                if not isinstance(ts, datetime):
                    continue
                is_live_eff = _is_live_eff(d0, ts=ts)
                ws0 = _safe_float(d0.get("ws_odd"))
                if ws0 is None or ws0 <= 0:
                    continue
                # entry odd: BS real se existir; senão WS proxy
                bs = _safe_float(d0.get("bs_odd"))
                src = "BS"
                t_proxy = None
                entry = None
                if bs is not None and bs > 0:
                    entry = float(bs)
                else:
                    ref = _ws_proxy_odd(d0)
                    if not ref:
                        continue
                    entry, t_proxy = float(ref[0]), float(ref[1])
                    src = f"WS@t+{t_proxy:.1f}s"
                if entry is None or entry <= 0:
                    continue
                diff = ((float(entry) - float(ws0)) / float(ws0)) * 100.0 if ws0 else None
                if diff is None:
                    continue
                # filtro de qualidade (mesmo range do BS confiável)
                if not (-10.0 <= float(diff) <= 10.0):
                    continue
                # edge Back
                if float(diff) < float(back_cut):
                    continue
                # ROI no ponto de entrada (stake=1)
                _, roi_entry = compute_roi_pct(
                    line=str(d0.get("line", "")),
                    side=str(d0.get("side", "")),
                    ws_odd=None,
                    bs_odd=float(entry),
                    hs=d0.get("home_score"),
                    aws=d0.get("away_score"),
                )
                # CLV pre-match
                clv_entry = None
                if not is_live_eff:
                    clo = _safe_float(d0.get("closing_odd"))
                    if clo is not None and clo > 0:
                        clv_entry = (float(entry) - float(clo)) / float(clo) * 100.0
                combo_events.append(
                    {
                        "day": ts.astimezone(timezone.utc).strftime("%Y-%m-%d"),
                        "audit_id": int(aid),
                        "side": "Back",
                        "regime": "Pre" if (not is_live_eff) else "In",
                        "reversal": "Any",
                        "match_id": int(d0.get("match_id")),
                        "league": str(d0.get("league") or ""),
                        "ah_abs": line_abs(d0.get("line")),
                        "liq_limit": (_safe_float(d0.get("limit")) if (_safe_float(d0.get("limit")) or 0.0) > 0 else None),
                        "clv_back": clv_entry,
                        "clv_lay_conv": None,
                        "roi": roi_entry,
                        "entry_odd": float(entry),
                        "entry_source": src,
                        "diff_pct": float(diff),
                    }
                )

            # --- Lay: entrada pós-reversal (ou último ponto). Alinha com operação WS (não depende de betslip). ---
            for r in lay_entries:
                aid = r.get("audit_id")
                if aid is None:
                    continue
                d0 = audit_by_id.get(int(aid))
                if d0 and wf_excl_exec_lay and str(d0.get("exec_bucket")) in wf_excl_exec_lay:
                    continue
                ts = r.get("audited_at")
                if not isinstance(ts, datetime):
                    continue
                is_live_eff = _is_live_eff(d0 or {}, ts=ts) if isinstance(d0, dict) else bool(r.get("is_live") is True)
                ws0 = _safe_float((d0 or {}).get("ws_odd"))
                if ws0 is None or ws0 <= 0:
                    continue
                entry = _safe_float(r.get("odd_entry"))
                src = str(r.get("entry_policy") or "WS_ENTRY")
                if entry is None or entry <= 0:
                    # fallback: se odd_entry não vier, tentar WS proxy no offset (melhor que nada)
                    ref = _ws_proxy_odd(d0 or {})
                    if not ref:
                        continue
                    entry = float(ref[0])
                    src = f"WS@t+{float(ref[1]):.1f}s"
                diff = ((float(entry) - float(ws0)) / float(ws0)) * 100.0 if ws0 else None
                if diff is None:
                    continue
                if not (-10.0 <= float(diff) <= 10.0):
                    continue
                # edge Lay (mesmo corte antigo, mas agora usando WS-entry vs WS0)
                if float(diff) > float(lay_cut):
                    continue
                combo_events.append(
                    {
                        "day": ts.astimezone(timezone.utc).strftime("%Y-%m-%d"),
                        "audit_id": int(aid),
                        "side": "Lay",
                        "regime": "Pre" if (not is_live_eff) else "In",
                        "reversal": "Yes" if bool(r.get("had_reversal")) else "No",
                        "match_id": int(r.get("match_id")),
                        "league": str((d0 or {}).get("league") or ""),
                        "ah_abs": line_abs((d0 or {}).get("line")),
                        "liq_limit": (
                            _safe_float(_get_path((d0 or {}).get("hypothesis_details") or {}, ["lay", "available_limit"]))
                            if (_safe_float(_get_path((d0 or {}).get("hypothesis_details") or {}, ["lay", "available_limit"])) or 0.0) > 0
                            else (_safe_float((d0 or {}).get("limit")) if (_safe_float((d0 or {}).get("limit")) or 0.0) > 0 else None)
                        ),
                        "clv_back": None,
                        "clv_lay_conv": r.get("clv_conv_entry") if (not is_live_eff) else None,
                        "roi": r.get("roi_entry"),
                        # IMPORTANT: use a odd efetivamente usada na elegibilidade/edge; não o campo cru (pode vir None)
                        "entry_odd": float(entry),
                        "entry_source": src,
                        "diff_pct": float(diff),
                    }
                )

            # Calendário do walk-forward: usar os dias com dados carregados no recorte (mesmo que não haja eventos OK/edge),
            # para não “sumir” dias recentes (ex.: 17/18) na tabela OOS.
            def _day_utc_from_ts(ts: Any) -> Optional[str]:
                if isinstance(ts, datetime):
                    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d")
                return None

            days_loaded = sorted({d for d in (_day_utc_from_ts(r.get("audited_at")) for r in all_data) if d})
            days_ok = sorted({d for d in (_day_utc_from_ts(r.get("audited_at")) for r in ok_any) if d})
            days_combo = sorted({e["day"] for e in combo_events})
            days = days_loaded if days_loaded else days_combo
            # Ajuste do início do calendário:
            # - Se o usuário definir `--wf-start-date`, respeita.
            # - Caso contrário, começa no 1º dia observado no recorte (modular ao `--lookback-days`).
            wf_start = str(getattr(args, "wf_start_date", "") or "").strip()
            if not wf_start:
                if days_loaded:
                    wf_start = min(days_loaded)
                elif days_combo:
                    wf_start = min(days_combo)
            if wf_start:
                days = [d for d in days if str(d) >= wf_start]
            # Diagnóstico de cobertura (explica por que OOS tem N bem menor)
            uniq_matches_total = len({int(e["match_id"]) for e in combo_events})
            uniq_matches_roi = len({int(e["match_id"]) for e in combo_events if e.get("roi") is not None})
            uniq_matches_clv = len(
                {int(e["match_id"]) for e in combo_events if (e.get("clv_back") is not None or e.get("clv_lay_conv") is not None)}
            )
            lines.append("### 12.0 Diagnóstico de cobertura OOS (por que N cai)\n")
            lines.append("| Filtro | Jogos únicos |\n|---|---:|\n")
            lines.append(f"| Combinações elegíveis (edge + timing + t0) | {uniq_matches_total} |\n")
            lines.append(f"| Com ROI disponível (precisa de placar) | {uniq_matches_roi} |\n")
            lines.append(f"| Com CLV disponível (pre-match + closing) | {uniq_matches_clv} |\n")
            lines.append("\n**Calendário do walk-forward (dias únicos)**\n\n")
            lines.append("| Tipo | Dias |\n|---|---:|\n")
            lines.append(f"| Dias com dados carregados (audited_at) | {len(days_loaded)} |\n")
            lines.append(f"| Dias com eventos OK (qualquer versão, incl. ws-only) | {len(days_ok)} |\n")
            lines.append(f"| Dias com eventos elegíveis p/ WF (edge) | {len(days_combo)} |\n")
            lines.append(f"| Dias usados no walk-forward | {len(days)} |\n")

            # Diagnóstico por dia: ajuda a distinguir "dia vazio por falta de oportunidade" vs "dia vazio por falha de qualidade/execução"
            day_all_cnt = Counter(_day_utc_from_ts(d.get("audited_at")) for d in all_data if _day_utc_from_ts(d.get("audited_at")))
            day_bs_raw_cnt = Counter(_day_utc_from_ts(d.get("audited_at")) for d in with_bs_raw if _day_utc_from_ts(d.get("audited_at")))
            day_bs_conf_cnt = Counter(_day_utc_from_ts(d.get("audited_at")) for d in with_bs if _day_utc_from_ts(d.get("audited_at")))
            day_ok_cnt = Counter(_day_utc_from_ts(d.get("audited_at")) for d in ok_any if _day_utc_from_ts(d.get("audited_at")))
            day_back_edge_cnt = Counter(e.get("day") for e in combo_events if e.get("side") == "Back" and e.get("day"))
            day_lay_edge_cnt = Counter(e.get("day") for e in combo_events if e.get("side") == "Lay" and e.get("day"))
            day_pre_edge_cnt = Counter(e.get("day") for e in combo_events if e.get("regime") == "Pre" and e.get("day"))
            day_in_edge_cnt = Counter(e.get("day") for e in combo_events if e.get("regime") != "Pre" and e.get("day"))

            day_non_ok_top: Dict[str, str] = {}
            tmp: Dict[str, Counter] = {}
            for d in with_bs:
                day = _day_utc_from_ts(d.get("audited_at"))
                if not day:
                    continue
                st = str(d.get("status", "") or "").upper()
                if st == "OK":
                    continue
                tmp.setdefault(day, Counter())
                tmp[day][st or "NA"] += 1
            for day, c in tmp.items():
                day_non_ok_top[day] = str(c.most_common(1)[0][0]) if c else "—"

            lines.append("\n**Diagnóstico por dia (audited_at): betslip vs qualidade vs edge**\n\n")
            lines.append(
                "| Dia | Auditorias carregadas | Betslip bruto | Betslip conf. | OK (conf.) | Edge Back/Lay | Edge Pre/In | %OK/conf. | Status não-OK dominante |\n"
                "|---|---:|---:|---:|---:|---:|---:|---:|---|\n"
            )
            for day in days_loaded:
                a = int(day_all_cnt.get(day, 0))
                br = int(day_bs_raw_cnt.get(day, 0))
                conf = int(day_bs_conf_cnt.get(day, 0))
                ok = int(day_ok_cnt.get(day, 0))
                eb = int(day_back_edge_cnt.get(day, 0))
                el = int(day_lay_edge_cnt.get(day, 0))
                ep = int(day_pre_edge_cnt.get(day, 0))
                ei = int(day_in_edge_cnt.get(day, 0))
                ok_rate = (100.0 * ok / conf) if conf > 0 else None
                lines.append(
                    f"| {day} | {a} | {br} | {conf} | {ok} | {eb}/{el} | {ep}/{ei} | {_fmt_num(ok_rate,1)}% | {day_non_ok_top.get(day,'—')} |\n"
                )
            lines.append(
                "\nLeitura:\n"
                "- Se `Auditorias carregadas > 0` mas `Betslip conf.` ≈ 0, geralmente houve **mismatch/parse** (diff fora de [-10,+10]) ou ausência de betslip.\n"
                "- Se `Betslip conf. > 0` mas `OK (conf.) = 0`, o robô coletou betslip, mas os eventos falharam por **status != OK** (ver coluna de status).\n"
                "- Dias com `OK (conf.) = 0` **não devem ser tratados como “0 oportunidade”** sem investigar o operacional.\n\n"
            )
            lines.append(
                "\nLeitura: o walk-forward mede OOS principalmente por **ROI**, então ele encolhe quando a cobertura de placar é baixa. "
                "Além disso, a métrica é agregada por **jogo único** (cluster), então você verá números menores que o N de eventos.\n\n"
            )
            # Transparência: parâmetros efetivos do WF (evita confusão quando env vars/CLI mudam defaults)
            try:
                lines.append(
                    f"**Parâmetros efetivos (WF)**: dias_unicos={len(days)} | wf_train_days={wf_train} | wf_test_days={wf_test} | wf_step_days={wf_step} | only_oos={'ON' if bool(getattr(args,'only_oos',False)) else 'OFF'}\n\n"
                )
            except Exception:
                pass

            # Modo `--only-oos`: se a janela for curta, preferimos AJUSTAR o treino para caber no recorte (ao invés de abortar o WF)
            # e, se ainda assim não couber, encerramos cedo para não cair em seções 12.1+.
            if bool(getattr(args, "only_oos", False)) and (len(days) > 0) and (len(days) < (wf_train + wf_test)):
                try:
                    wf_train_old = int(wf_train)
                    # garante pelo menos 1 passo: precisamos de i=wf_train <= len(days)-wf_test
                    wf_train = int(max(1, min(int(wf_train), int(max(1, len(days) - wf_test)))))
                    if wf_train != wf_train_old:
                        lines.append(
                            f"[INFO] only-oos: ajustando `wf_train_days` de {wf_train_old} para {wf_train} para caber em dias_unicos={len(days)}.\n\n"
                        )
                except Exception:
                    pass
            if len(days) < (wf_train + wf_test):
                lines.append(
                    f"[WARN] Janela curta para walk-forward: dias únicos={len(days)}; precisa >= {wf_train + wf_test}.\n\n"
                )
                if bool(getattr(args, "only_oos", False)):
                    try:
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        out_path.write_text("".join(lines), encoding="utf-8")
                        print(f"Relatório gerado em: {out_path}")
                    except Exception as e:
                        print(f"[WARN] Falha ao escrever saída only-oos (janela curta): {e}")
                    return 0
            else:
                def _key(e: dict) -> str:
                    """
                    Chave de combinação para OOS.
                    - Back: não depende de reversão (estratégia é "entrar rápido"), então agregamos Yes/No como 'Any'.
                    - Lay: mantém Yes/No porque a política de entrada depende de reversão (reversal -> entra após; senão entra no último ponto).
                    """
                    side = str(e.get("side"))
                    regime = str(e.get("regime"))
                    rev = str(e.get("reversal"))
                    if side == "Back":
                        rev = "Any"
                    base = f"{side}_{regime}_{rev}"
                    if bool(getattr(args, "wf_key_by_league", False)):
                        scope = str(getattr(args, "wf_key_by_league_scope", "pre") or "pre").strip().lower()
                        if scope not in ("pre", "all"):
                            scope = "pre"
                        # default recomendado: liga influencia mais o pre-match; evita ruído no in-match.
                        if scope == "pre" and regime != "Pre":
                            return base
                        lg = str(e.get("league") or "—").strip() or "—"
                        # normaliza para evitar quebrar tabelas/markdown
                        lg = lg.replace("|", "/").replace("\n", " ").replace("\r", " ").strip()
                        # limita tamanho para manter legibilidade
                        if len(lg) > 48:
                            lg = lg[:48].rstrip() + "…"
                        return f"{base}__{lg}"
                    return base

                def _ev_ts_utc(ev: dict) -> float:
                    """
                    Timestamp para ordenar eventos dentro do match.
                    Usamos `audited_at` do registro audit original (via audit_id).
                    """
                    try:
                        aid = int(ev.get("audit_id"))
                    except Exception:
                        return 0.0
                    d0 = audit_by_id.get(aid)
                    ts = (d0 or {}).get("audited_at")
                    if isinstance(ts, datetime):
                        try:
                            return float(ts.astimezone(timezone.utc).timestamp())
                        except Exception:
                            return float(ts.timestamp())
                    return 0.0

                def _dedup_match_key(events: List[dict]) -> List[dict]:
                    """
                    IMPORTANT (robustez do turnover):
                    Em alguns regimes coletamos múltiplas auditorias por jogo (ex.: BS temporal),
                    enquanto em outros há ~1 evento por jogo (ex.: WS gate t+5).
                    Se somarmos em nível de evento, o turnover vira uma métrica de instrumentação,
                    e não de oportunidade.

                    Portanto, deduplicamos por (match_id, key) e mantemos o evento mais cedo no tempo
                    (proxy do “primeiro ponto executável”).
                    """
                    if not events:
                        return []
                    best: Dict[Tuple[int, str], dict] = {}
                    best_ts: Dict[Tuple[int, str], float] = {}
                    for ev in events:
                        try:
                            mid = int(ev.get("match_id"))
                        except Exception:
                            continue
                        kk = str(_key(ev))
                        k = (mid, kk)
                        t = _ev_ts_utc(ev)
                        if k not in best or t < float(best_ts.get(k, 9e18)):
                            best[k] = ev
                            best_ts[k] = float(t)
                    # retorna em ordem temporal (melhor para budget por match)
                    out = list(best.values())
                    out.sort(key=_ev_ts_utc)
                    return out

                def _bym(sub: List[dict], key: str) -> Dict[int, List[float]]:
                    bym: Dict[int, List[float]] = {}
                    for e in sub:
                        v = _safe_float(e.get(key))
                        if v is None:
                            continue
                        bym.setdefault(int(e["match_id"]), []).append(float(v))
                    return bym

                def _q(bym: Dict[int, List[float]], q: float) -> Optional[float]:
                    return cluster_bootstrap_quantile(bym, q, n_boot=2000, seed=int(args.seed))

                def _mean_ci90(bym: Dict[int, List[float]]) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
                    return cluster_bootstrap_ci(bym, n_boot=2000, alpha=0.10, seed=int(args.seed))

                def _se_from_ci90(ci: Optional[Tuple[float, float]]) -> Optional[float]:
                    if not ci:
                        return None
                    try:
                        lb, ub = float(ci[0]), float(ci[1])
                    except Exception:
                        return None
                    if not (math.isfinite(lb) and math.isfinite(ub)):
                        return None
                    # CI90 ~= mean ± 1.645*SE (aprox Normal)
                    return max(1e-9, (ub - lb) / (2.0 * 1.645))

                def _shrink_means_empirical_bayes(means_se: Dict[str, Tuple[Optional[float], Optional[float]]]) -> Dict[str, Optional[float]]:
                    """
                    Empirical Bayes (Normal-Normal) shrinkage por step:
                    - cada combinação k tem (mean_k, se_k)
                    - prior mean = média global dos means_k (ponderada por 1/se^2 quando possível)
                    - prior variance tau^2 estimada via método dos momentos
                    Retorna mean_shrunk_k. Se dados insuficientes, retorna mean_k.
                    """
                    items = [(k, v[0], v[1]) for k, v in (means_se or {}).items() if v and v[0] is not None]
                    if len(items) < 2:
                        return {k: (m if m is not None else None) for k, (m, _) in (means_se or {}).items()}

                    # pesos = 1/se^2 (fallback 1)
                    ws = []
                    ms = []
                    for _, m, se in items:
                        w = 1.0
                        if se is not None and float(se) > 0 and math.isfinite(float(se)):
                            w = 1.0 / (float(se) ** 2)
                        ws.append(float(w))
                        ms.append(float(m))
                    wsum = sum(ws) if sum(ws) > 0 else float(len(ws))
                    mu0 = sum(w * m for w, m in zip(ws, ms)) / wsum

                    # variância entre-means observada
                    mean_unw = sum(ms) / float(len(ms))
                    s2 = sum((m - mean_unw) ** 2 for m in ms) / max(1.0, float(len(ms) - 1))
                    # variância média do erro
                    vbar = 0.0
                    n_v = 0
                    for _, _, se in items:
                        if se is None:
                            continue
                        v = float(se) ** 2
                        if not math.isfinite(v):
                            continue
                        vbar += v
                        n_v += 1
                    vbar = (vbar / float(n_v)) if n_v > 0 else 0.0
                    tau2 = max(0.0, float(s2) - float(vbar))

                    out: Dict[str, Optional[float]] = {}
                    for k, (m, se) in (means_se or {}).items():
                        if m is None:
                            out[str(k)] = None
                            continue
                        if tau2 <= 0:
                            out[str(k)] = float(m)
                            continue
                        if se is None or float(se) <= 0 or not math.isfinite(float(se)):
                            out[str(k)] = float(m)
                            continue
                        w = float(tau2) / (float(tau2) + float(se) ** 2)
                        out[str(k)] = float(w) * float(m) + float(1.0 - w) * float(mu0)
                    return out

                steps = []
                active_counts: Dict[str, int] = {}
                # séries para estimativa 30d (OOS)
                daily_turn = {}  # day -> turnover stake eq (todas elegíveis)
                daily_turn_pre = {}
                daily_turn_in = {}
                daily_pnl_obs = {}  # day -> pnl observado (somente ROI disponível)
                daily_pnl_obs_pre = {}
                daily_pnl_obs_in = {}
                daily_pnl_exp = {}  # day -> pnl expandido (se wf_expand)
                daily_pnl_exp_pre = {}
                daily_pnl_exp_in = {}
                # exposições para banca/liquidez (todas elegíveis)
                oos_back_stakes_all: List[float] = []
                oos_lay_liab_all: List[float] = []
                oos_jobs: List[Tuple[datetime, datetime, float]] = []  # (t0, t1, exposure)

                def _scheme_for_event(ev: dict) -> str:
                    sc = wf_scheme_pre if ev.get("regime") == "Pre" else wf_scheme_in
                    # Guardrail: Kelly depende de closing_odd pré‑jogo e não é aplicável no in‑match.
                    # Se o usuário configurar wf_scheme_in=KELLY_*, caímos para FLAT para não “sumir” turnover/lucro.
                    try:
                        if str(ev.get("regime")) != "Pre" and str(sc).upper().startswith("KELLY"):
                            return "FLAT"
                    except Exception:
                        pass
                    return sc

                def _sizing_for_event(
                    ev: dict,
                    *,
                    roi_train_map: Optional[Dict[str, float]] = None,
                ) -> Tuple[Optional[float], Optional[float], Optional[str]]:
                    """
                    Retorna (stake_turnover_equivalente, exposure_risk).
                    - Back: stake=exposure=stake
                    - Lay: stake_turnover = liab/(odd-1), exposure_risk=liab
                    """
                    aid = int(ev.get("audit_id"))
                    d0 = audit_by_id.get(aid)
                    if not d0:
                        return (None, None, "NO_AUDIT_ROW")
                    sc = _scheme_for_event(ev)
                    if ev.get("side") == "Back":
                        st = None
                        if sc == "FLAT":
                            st = float(wf_flat_stake_back)
                        elif sc == "PROXY":
                            st = _sizing_back(d0, "PROXY", bank_ref=float(back_bank_ref))
                            # se o limit vier 0/None, o PROXY pode virar 0; não queremos “sumir” do turnover
                            if st is None or float(st) <= 0:
                                st = float(wf_flat_stake_back)
                        elif str(sc).startswith("KELLY"):
                            # Kelly no WF deve usar a odd de entrada do evento (pode vir de WS proxy em ws-only)
                            if d0.get("is_live") is True:
                                # Kelly não é aplicável in-match; fallback para FLAT/PROXY.
                                st_fb = _sizing_back(d0, "PROXY", bank_ref=float(back_bank_ref))
                                if st_fb is None or float(st_fb) <= 0:
                                    st_fb = float(wf_flat_stake_back)
                                st = float(st_fb)
                                st = min(float(st), float(_max_back_stake_event(d0)))
                                if st is None or float(st) <= 0:
                                    return (None, None, "BACK_ST_NONPOS_IN_FB")
                                return (float(st), float(st), None)
                            try:
                                frac = float(str(sc).split("_")[1])
                            except Exception:
                                return (None, None, "BACK_BAD_SCHEME")
                            entry_odd = _safe_float(ev.get("entry_odd"))
                            if entry_odd is None:
                                entry_odd = _safe_float(d0.get("bs_odd"))
                            f0 = _kelly_back_frac(entry_odd, d0.get("closing_odd"))
                            if f0 is None:
                                # fallback: quando não há closing_odd (ou cálculo falha), não podemos aplicar Kelly.
                                # Para não “sumir” turnover/lucro em buckets recentes, caímos para PROXY/FLAT.
                                st_fb = _sizing_back(d0, "PROXY", bank_ref=float(back_bank_ref))
                                if st_fb is None or float(st_fb) <= 0:
                                    st_fb = float(wf_flat_stake_back)
                                st = float(st_fb)
                                st = min(float(st), float(_max_back_stake_event(d0)))
                                if st is None or float(st) <= 0:
                                    return (None, None, "BACK_ST_NONPOS_FB")
                                return (float(st), float(st), "BACK_KELLY_FALLBACK_NO_CLOSING")
                            f = max(0.0, float(f0)) * float(frac)
                            bank_ref_budget_local = float(max(float(back_bank_ref or 0.0), float(lay_bank_ref or 0.0), 1.0))
                            cap = BACK_CAP_FRAC * max(1e-9, bank_ref_budget_local)
                            st = min(f * bank_ref_budget_local, cap)
                            st = min(float(st), float(_max_back_stake_event(d0)))
                        elif str(sc).upper() == "ROI_TRAIN":
                            k = _key(ev)
                            roi_hat = _safe_float(((roi_train_map or roi_train_by_key) or {}).get(k))
                            if roi_hat is None:
                                return (None, None, "BACK_ROI_TRAIN_MISSING")
                            # proxy de Kelly: f ~= EV (assumindo odds típicas ~2.0). Caps + limit controlam.
                            f = max(0.0, float(roi_hat)) / 100.0
                            # escala pela mesma referência usada no budget (coerente com governança)
                            bank_ref_budget_local = float(max(float(back_bank_ref or 0.0), float(lay_bank_ref or 0.0), 1.0))
                            cap = BACK_CAP_FRAC * max(1e-9, bank_ref_budget_local)
                            st = min(f * bank_ref_budget_local, cap)
                            st = min(float(st), float(_max_back_stake_event(d0)))
                        if st is None or float(st) <= 0:
                            return (None, None, "BACK_ST_NONPOS")
                        return (float(st), float(st), None)
                    # Lay
                    # Usa odd de entrada (reversão/último) para sizing coerente com a estratégia
                    lay_odd = _safe_float(ev.get("entry_odd"))
                    if lay_odd is None:
                        h = d0.get("hypothesis_details") or {}
                        lay_odd = _safe_float(_get_path(h, ["lay", "odd"])) or _safe_float(d0.get("bs_odd"))
                    if lay_odd is None or float(lay_odd) <= 1.0:
                        return (None, None, "LAY_ODD_MISSING_OR_LE1")
                    if sc == "FLAT":
                        liab = float(wf_flat_liab_lay)
                    elif sc == "PROXY":
                        # replica finance_for_row, mas com odd de entrada
                        h = d0.get("hypothesis_details") or {}
                        lay_lim = _safe_float(_get_path(h, ["lay", "available_limit"]))
                        if lay_lim is None:
                            lay_lim = _safe_float(d0.get("limit"))
                        if lay_lim is None or float(lay_lim or 0.0) <= 0:
                            # limit indisponível/zerado: usa fallback FLAT para não zerar turnover
                            liab = float(wf_flat_liab_lay)
                        else:
                            st = stake_from_limit(float(lay_lim or 0.0))
                            if st is None or float(st) <= 0:
                                liab = float(wf_flat_liab_lay)
                            else:
                                liab = float(st) * max(0.0, float(lay_odd) - 1.0)
                    elif sc.startswith("KELLY"):
                        # Kelly não faz sentido in-match (closing_odd é pré‑jogo); ainda assim não “mata” turnover.
                        if d0.get("is_live") is True:
                            # fallback para PROXY/FLAT
                            h = d0.get("hypothesis_details") or {}
                            lay_lim = _safe_float(_get_path(h, ["lay", "available_limit"]))
                            if lay_lim is None:
                                lay_lim = _safe_float(d0.get("limit"))
                            st_fb = stake_from_limit(float(lay_lim or 0.0))
                            if st_fb is None or float(st_fb) <= 0:
                                liab = float(wf_flat_liab_lay)
                            else:
                                liab = float(st_fb) * max(0.0, float(lay_odd) - 1.0)
                            cap = LAY_CAP_FRAC * max(1e-9, float(lay_bank_ref))
                            liab = min(float(liab), float(cap))
                            max_st = _max_lay_stake_event(d0)
                            liab = min(float(liab), float(max_st) * max(0.0, float(lay_odd) - 1.0))
                            if liab is None or float(liab) <= 0:
                                return (None, None, "LAY_LIAB_NONPOS_IN_FB")
                            stake_eq = float(liab) / max(1e-9, (float(lay_odd) - 1.0))
                            return (float(stake_eq), float(liab), None)
                        frac = float(sc.split("_")[1])
                        f0 = _kelly_lay_liab_frac(lay_odd, d0.get("closing_odd"))
                        if f0 is None:
                            # fallback: sem closing_odd não dá para Kelly. Usa PROXY/FLAT para não distorcer turnover.
                            h = d0.get("hypothesis_details") or {}
                            lay_lim = _safe_float(_get_path(h, ["lay", "available_limit"]))
                            if lay_lim is None:
                                lay_lim = _safe_float(d0.get("limit"))
                            st_fb = stake_from_limit(float(lay_lim or 0.0))
                            if st_fb is None or float(st_fb) <= 0:
                                liab = float(wf_flat_liab_lay)
                            else:
                                liab = float(st_fb) * max(0.0, float(lay_odd) - 1.0)
                            cap = LAY_CAP_FRAC * max(1e-9, float(lay_bank_ref))
                            liab = min(float(liab), float(cap))
                            # cap por evento (limit)
                            max_st = _max_lay_stake_event(d0)
                            liab = min(float(liab), float(max_st) * max(0.0, float(lay_odd) - 1.0))
                            if liab is None or float(liab) <= 0:
                                return (None, None, "LAY_LIAB_NONPOS_FB")
                            stake_eq = float(liab) / max(1e-9, (float(lay_odd) - 1.0))
                            return (float(stake_eq), float(liab), "LAY_KELLY_FALLBACK_NO_CLOSING")
                        f = max(0.0, float(f0)) * float(frac)
                        cap = LAY_CAP_FRAC * max(1e-9, float(lay_bank_ref))
                        liab = min(f * float(lay_bank_ref), cap)
                    else:
                        if str(sc).upper() == "ROI_TRAIN":
                            k = _key(ev)
                            roi_hat = _safe_float(((roi_train_map or roi_train_by_key) or {}).get(k))
                            if roi_hat is None:
                                return (None, None, "LAY_ROI_TRAIN_MISSING")
                            f = max(0.0, float(roi_hat)) / 100.0
                            bank_ref_budget_local = float(max(float(back_bank_ref or 0.0), float(lay_bank_ref or 0.0), 1.0))
                            cap = LAY_CAP_FRAC * max(1e-9, bank_ref_budget_local)
                            liab = min(f * bank_ref_budget_local, cap)
                        else:
                            return (None, None, "LAY_UNSUPPORTED_SCHEME")
                    # cap por evento (limit): converte stake max -> liab max
                    max_st = _max_lay_stake_event(d0)
                    liab = min(float(liab), float(max_st) * max(0.0, float(lay_odd) - 1.0))
                    if liab is None or float(liab) <= 0 or lay_odd is None or float(lay_odd) <= 1.0:
                        return (None, None, "LAY_LIAB_NONPOS")
                    stake_eq = float(liab) / max(1e-9, (float(lay_odd) - 1.0))
                    return (float(stake_eq), float(liab), None)

                def _append_job(ev: dict, exposure: float):
                    """Para banca de liquidez: capital simultaneamente travado."""
                    aid = int(ev.get("audit_id"))
                    d0 = audit_by_id.get(aid)
                    if not d0:
                        return
                    t0 = d0.get("audited_at") or d0.get("hypothesis_detected_at")
                    if not isinstance(t0, datetime):
                        return
                    settle_h = float(os.getenv("LIQUIDITY_SETTLE_BUFFER_HOURS", "2.25"))
                    dur_h = float(os.getenv("LIQUIDITY_MATCH_DURATION_HOURS", "2.0"))
                    ko = d0.get("kickoff") or d0.get("kickoff_time")
                    t1 = (ko + timedelta(hours=(dur_h + settle_h))) if isinstance(ko, datetime) else (t0 + timedelta(hours=(dur_h + settle_h)))
                    if t1 <= t0:
                        t1 = t0 + timedelta(hours=(dur_h + settle_h))
                    oos_jobs.append((t0, t1, float(exposure)))

                def _liq_p99_from_jobs(jobs: List[Tuple[datetime, datetime, float]]) -> Optional[float]:
                    if not jobs:
                        return None
                    grid_min = int(os.getenv("LIQUIDITY_GRID_MINUTES", "5"))
                    buf_pct = float(os.getenv("LIQUIDITY_BANK_BUFFER_PCT", "10"))
                    step = max(1, grid_min)
                    t_min = min(j[0] for j in jobs)
                    t_max = max(j[1] for j in jobs)
                    t = t_min
                    vals = []
                    while t <= t_max:
                        s = 0.0
                        for a, b, exp in jobs:
                            if a <= t <= b:
                                s += float(exp)
                        vals.append(float(s))
                        t = t + timedelta(minutes=step)
                    if not vals:
                        return None
                    p99 = float(np.quantile(vals, 0.99))
                    return float(p99) * (1.0 + max(0.0, float(buf_pct)) / 100.0)
                # Walk-forward por dia:
                # - `wf_step=1` (default) cria janelas de teste deslizantes; se `wf_test>1`, haverá sobreposição.
                # - Para janelas de teste não sobrepostas (resultados somáveis por passo), use `wf_step=wf_test`.
                for i in range(wf_train, len(days) - wf_test + 1, wf_step):
                    train_days = set(days[:i]) if wf_train_mode == "expanding" else set(days[i - wf_train : i])
                    test_days = set(days[i : i + wf_test])

                    # Garante que dias de teste apareçam no acumulado, mesmo com turnover/lucro zero
                    # (importante para refletir “dias sem operação” no OOS e não encurtar artificialmente o horizonte).
                    for dday in test_days:
                        daily_turn.setdefault(dday, 0.0)
                        daily_turn_pre.setdefault(dday, 0.0)
                        daily_turn_in.setdefault(dday, 0.0)
                        daily_pnl_obs.setdefault(dday, 0.0)
                        daily_pnl_exp.setdefault(dday, 0.0)
                        daily_pnl_obs_pre.setdefault(dday, 0.0)
                        daily_pnl_obs_in.setdefault(dday, 0.0)
                        daily_pnl_exp_pre.setdefault(dday, 0.0)
                        daily_pnl_exp_in.setdefault(dday, 0.0)

                    # Aplica políticas operacionais determinísticas também no treino/teste (sem lookahead):
                    # - filtro por linha AH (|line|)
                    # Observação: gates baseados em percentil de limit (p50/p75) são aplicados no teste com limiar do treino.
                    train_raw = [e for e in combo_events if e["day"] in train_days and _ah_ok(e)]
                    test_raw = [e for e in combo_events if e["day"] in test_days and e.get("roi") is not None and _ah_ok(e)]
                    # Dedup por jogo×chave para estabilizar turnover/ROI contra diferenças de instrumentação
                    train = _dedup_match_key(train_raw)
                    test = _dedup_match_key(test_raw)

                    # thresholds de liquidez estimados na janela de treino (sem lookahead)
                    thr_p50_pre = thr_p75_pre = None
                    thr_p50_all = thr_p75_all = None
                    liq_train_pre: List[float] = []
                    liq_train_all: List[float] = []
                    for e in train:
                        x = _safe_float(e.get("liq_limit"))
                        if x is None or not math.isfinite(float(x)) or float(x) <= 0:
                            continue
                        liq_train_all.append(float(x))
                        if str(e.get("regime")) == "Pre":
                            liq_train_pre.append(float(x))
                    if liq_train_pre:
                        thr_p50_pre = float(np.quantile(liq_train_pre, 0.50))
                        thr_p75_pre = float(np.quantile(liq_train_pre, 0.75))
                    if liq_train_all:
                        thr_p50_all = float(np.quantile(liq_train_all, 0.50))
                        thr_p75_all = float(np.quantile(liq_train_all, 0.75))

                    thr_use = None
                    if wf_liq_mode == "gate_p50":
                        thr_use = thr_p50_pre if wf_liq_scope == "pre" else thr_p50_all
                    elif wf_liq_mode == "gate_p75":
                        thr_use = thr_p75_pre if wf_liq_scope == "pre" else thr_p75_all
                    elif wf_liq_mode == "gate_min":
                        thr_use = float(wf_liq_min)

                    # seleção por combinação
                    active: List[str] = []
                    diag = {}
                    wf_key_by_league = bool(getattr(args, "wf_key_by_league", False))
                    # pré-calcula shrinkage de ROI por combinação neste step
                    shrink_map: Dict[str, Optional[float]] = {}
                    if wf_use_shrink:
                        means_se: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
                        for kk in sorted({_key(e) for e in train}):
                            subk = [e for e in train if _key(e) == kk]
                            bym_roi_k = _bym([e for e in subk if e.get("roi") is not None], "roi")
                            if not bym_roi_k:
                                means_se[kk] = (None, None)
                                continue
                            m_k, ci_k = _mean_ci90(bym_roi_k)
                            se_k = _se_from_ci90(ci_k)
                            means_se[kk] = (m_k, se_k)
                        shrink_map = _shrink_means_empirical_bayes(means_se)
                    for k in sorted({_key(e) for e in train}):
                        sub = [e for e in train if _key(e) == k]
                        # aplica critérios OOS (mais peso em ROI; CLV é gate apenas quando ROI é >0 mas não-sig em pre-match)
                        # parse key
                        parts = k.split("_")
                        side = parts[0]
                        regime = parts[1]
                        ok_sel = False
                        reason = ""

                        def _ci_sig_pos(ci: Optional[Tuple[float, float]]) -> bool:
                            return bool(ci and float(ci[0]) > 0)

                        def _ci_sig_neg(ci: Optional[Tuple[float, float]]) -> bool:
                            return bool(ci and float(ci[1]) < 0)

                        # ROI por jogo (cluster)
                        bym_roi = _bym([e for e in sub if e.get("roi") is not None], "roi")
                        roi_mean, roi_ci = _mean_ci90(bym_roi) if bym_roi else (None, None)
                        roi_mean_eff = roi_mean
                        if wf_use_shrink:
                            roi_sh = _safe_float((shrink_map or {}).get(str(k)))
                            if roi_sh is not None:
                                roi_mean_eff = float(roi_sh)
                        roi_pos = bool(roi_mean_eff is not None and float(roi_mean_eff) > 0)
                        roi_sig_pos = _ci_sig_pos(roi_ci)
                        roi_sig_neg = _ci_sig_neg(roi_ci)
                        eligible = bool(len(bym_roi) >= wf_min_m)

                        if side == "Back" and regime == "Pre":
                            bym_clv = _bym(sub, "clv_back")
                            clv_mean, clv_ci = _mean_ci90(bym_clv) if bym_clv else (None, None)
                            clv_pos = bool(clv_mean is not None and float(clv_mean) > 0)
                            clv_avail = bool(clv_mean is not None)
                            if not eligible:
                                ok_sel = False
                                reason = f"BackPre: N_ROI<{wf_min_m} (N={len(bym_roi)})"
                            elif roi_sig_neg:
                                ok_sel = False
                                reason = "BackPre: ROI sig<0 (bloqueia)"
                            elif roi_sig_pos:
                                ok_sel = True
                                reason = "BackPre: ROI sig>0"
                            elif roi_pos and ((not clv_avail) or clv_pos):
                                ok_sel = True
                                reason = "BackPre: ROI>0 (NS) AND (CLV ausente OU CLV>0)"
                            else:
                                ok_sel = False
                                reason = f"BackPre: ROI>0={roi_pos}, CLV>0={clv_pos} (CLV ausente={not clv_avail})"
                            diag[k] = {
                                "ok": ok_sel,
                                "reason": reason,
                                "train_matches_total": len({e['match_id'] for e in sub}),
                                "train_matches_clv": len(bym_clv),
                                "clv_q10": _q(bym_clv, 0.10) if bym_clv else None,
                                "clv_mean": clv_mean,
                                "clv_ci90": clv_ci,
                                "clv_available": clv_avail,
                                "train_matches_roi": len(bym_roi),
                                "roi_mean": roi_mean,
                                "roi_mean_eff": roi_mean_eff,
                                "roi_ci90": roi_ci,
                                "roi_sig_neg": roi_sig_neg,
                                "roi_sig_pos": roi_sig_pos,
                                "roi_q30": _q(bym_roi, 0.30) if bym_roi else None,
                            }
                        elif side == "Back" and regime == "In":
                            if not eligible:
                                ok_sel = False
                                reason = f"BackIn: N_ROI<{wf_min_m} (N={len(bym_roi)})"
                            elif roi_sig_neg:
                                ok_sel = False
                                reason = "BackIn: ROI sig<0 (bloqueia)"
                            elif roi_sig_pos:
                                ok_sel = True
                                reason = "BackIn: ROI sig>0"
                            else:
                                ok_sel = bool(roi_pos)
                                reason = f"BackIn: ROI>0={roi_pos}"
                            diag[k] = {
                                "ok": ok_sel,
                                "reason": reason,
                                "train_matches_total": len({e['match_id'] for e in sub}),
                                "train_matches_roi": len(bym_roi),
                                "roi_mean": roi_mean,
                                "roi_mean_eff": roi_mean_eff,
                                "roi_ci90": roi_ci,
                                "roi_sig_neg": roi_sig_neg,
                                "roi_sig_pos": roi_sig_pos,
                                "roi_q30": _q(bym_roi, 0.30) if bym_roi else None,
                            }
                        elif side == "Lay" and regime == "Pre":
                            bym_clv = _bym(sub, "clv_lay_conv")
                            # `clv_lay_conv` é a convenção com sinal "unificado":
                            # clv_conv = -(entry - closing) / closing  => Lay "bom" tende a ser POSITIVO.
                            clv_mean, clv_ci = _mean_ci90(bym_clv) if bym_clv else (None, None)
                            clv_pos = bool(clv_mean is not None and float(clv_mean) > 0)
                            clv_avail = bool(clv_mean is not None)
                            if not eligible:
                                ok_sel = False
                                reason = f"LayPre: N_ROI<{wf_min_m} (N={len(bym_roi)})"
                            elif roi_sig_neg:
                                ok_sel = False
                                reason = "LayPre: ROI sig<0 (bloqueia)"
                            elif roi_sig_pos:
                                ok_sel = True
                                reason = "LayPre: ROI sig>0"
                            elif roi_pos and ((not clv_avail) or clv_pos):
                                ok_sel = True
                                reason = "LayPre: ROI>0 (NS) AND (CLV_CONV ausente OU CLV_CONV>0)"
                            else:
                                ok_sel = False
                                reason = f"LayPre: ROI>0={roi_pos}, CLV_CONV>0={clv_pos} (CLV ausente={not clv_avail})"
                            diag[k] = {
                                "ok": ok_sel,
                                "reason": reason,
                                "train_matches_total": len({e['match_id'] for e in sub}),
                                "train_matches_clv": len(bym_clv),
                                "clv_q10": _q(bym_clv, 0.10) if bym_clv else None,
                                "clv_mean": clv_mean,
                                "clv_ci90": clv_ci,
                                "clv_available": clv_avail,
                                "train_matches_roi": len(bym_roi),
                                "roi_mean": roi_mean,
                                "roi_mean_eff": roi_mean_eff,
                                "roi_ci90": roi_ci,
                                "roi_sig_neg": roi_sig_neg,
                                "roi_sig_pos": roi_sig_pos,
                                "roi_q30": _q(bym_roi, 0.30) if bym_roi else None,
                            }
                        else:
                            if not eligible:
                                ok_sel = False
                                reason = f"In: N_ROI<{wf_min_m} (N={len(bym_roi)})"
                            elif roi_sig_neg:
                                ok_sel = False
                                reason = "In: ROI sig<0 (bloqueia)"
                            elif roi_sig_pos:
                                ok_sel = True
                                reason = "In: ROI sig>0"
                            else:
                                ok_sel = bool(roi_pos)
                                reason = f"In: ROI>0={roi_pos}"
                            diag[k] = {
                                "ok": ok_sel,
                                "reason": reason,
                                "train_matches_total": len({e['match_id'] for e in sub}),
                                "train_matches_roi": len(bym_roi),
                                "roi_mean": roi_mean,
                                "roi_mean_eff": roi_mean_eff,
                                "roi_ci90": roi_ci,
                                "roi_sig_neg": roi_sig_neg,
                                "roi_sig_pos": roi_sig_pos,
                                "roi_q30": _q(bym_roi, 0.30) if bym_roi else None,
                            }
                        if ok_sel:
                            active.append(k)
                            active_counts[k] = active_counts.get(k, 0) + 1

                    # Atualiza mapa de ROI do treino para sizing ROI_TRAIN (apenas combos ativos).
                    roi_train_by_key = {}
                    for k in active:
                        d = (diag.get(k) or {})
                        rm = _safe_float(d.get("roi_mean"))
                        if rm is None:
                            continue
                        roi_train_by_key[str(k)] = float(rm)

                    # avaliação OOS: ROI agregado nas combinações ativas
                    test_active = [e for e in test if _key(e) in set(active) and _liq_ok(e, thr=thr_use) and _ah_ok(e)]
                    bym_roi: Dict[int, List[float]] = {}
                    for e in test_active:
                        bym_roi.setdefault(int(e["match_id"]), []).append(float(e["roi"]))
                    oos_mean, oos_ci = cluster_bootstrap_ci(bym_roi, n_boot=2000, alpha=0.10, seed=int(args.seed))

                    # sizing + P&L no período de teste
                    # Conjunto elegível no teste (inclui sem ROI para turnover)
                    test_elig_raw = [e for e in combo_events if e["day"] in test_days and _key(e) in set(active) and _liq_ok(e, thr=thr_use) and _ah_ok(e)]
                    test_elig = _dedup_match_key(test_elig_raw)
                    back_st_all = back_st_roi = 0.0
                    lay_liab_all = lay_liab_roi = 0.0
                    turn_all = turn_roi = 0.0
                    pnl_obs = 0.0
                    # breakdown Pre/In (com budget efetivamente aplicado)
                    turn_pre = turn_in = 0.0
                    pnl_obs_pre = pnl_obs_in = 0.0
                    back_pre_all = back_pre_roi = 0.0
                    lay_pre_all = lay_pre_roi = 0.0
                    back_in_all = back_in_roi = 0.0
                    lay_in_all = lay_in_roi = 0.0
                    # contagens Pre/In (para diagnosticar quedas bruscas de turnover)
                    n_ev_elig_pre = sum(1 for ev in test_elig if str(ev.get("regime")) == "Pre")
                    n_ev_elig_in = sum(1 for ev in test_elig if str(ev.get("regime")) != "Pre")
                    n_ev_sized_pre = 0
                    n_ev_sized_in = 0
                    n_ev_after_budget_pre = 0
                    n_ev_after_budget_in = 0
                    # budget por jogo (padrão)
                    bank_ref_budget = _safe_float(getattr(args, "kelly_bankroll", None))
                    if bank_ref_budget is None or float(bank_ref_budget) <= 0:
                        bank_ref_budget = max(float(back_bank_ref or 0.0), float(lay_bank_ref or 0.0), 1.0)
                    bud_back = float(bud_back_frac) * float(bank_ref_budget)
                    bud_lay = float(bud_lay_frac) * float(bank_ref_budget)
                    # cap por sinal é fração do budget do jogo (no modo fixed). Em modos adaptativos, recalculamos por match.
                    cap_sig_back = float(bud_cap_sig_frac) * float(bud_back)
                    cap_sig_lay = float(bud_cap_sig_frac) * float(bud_lay)
                    spent_back: Dict[int, float] = {}
                    spent_lay: Dict[int, float] = {}
                    n_ev_elig = len(test_elig)
                    n_ev_sized = 0
                    n_ev_after_budget = 0

                    # ordena por tempo para consumir budget de forma realista
                    def _ts_ev(ev: dict) -> float:
                        d0 = audit_by_id.get(int(ev.get("audit_id")))
                        ts = d0.get("audited_at") if d0 else None
                        if isinstance(ts, datetime):
                            return ts.timestamp()
                        return 0.0
                    test_elig.sort(key=_ts_ev)
                    # concentração observada (risk-adaptive): usa a própria quantidade de sinais do match na janela de teste
                    cnt_back = Counter(int(ev.get("match_id")) for ev in test_elig if ev.get("side") == "Back")
                    cnt_lay = Counter(int(ev.get("match_id")) for ev in test_elig if ev.get("side") != "Back")
                    # Diagnóstico: quando N elig existe mas N sized colapsa, queremos saber "por quê" por regime/side
                    fail_sizing_pre: Counter[str] = Counter()
                    fail_sizing_in: Counter[str] = Counter()
                    fail_sizing_side_pre: Counter[str] = Counter()
                    fail_sizing_side_in: Counter[str] = Counter()
                    for ev in test_elig:
                        res_sz = _sizing_for_event(ev)
                        if isinstance(res_sz, tuple) and len(res_sz) == 2:
                            st_eq, exp = res_sz
                            why = None
                        else:
                            st_eq, exp, why = res_sz
                        if st_eq is None or exp is None:
                            w = str(why or "UNKNOWN")
                            if str(ev.get("regime")) == "Pre":
                                fail_sizing_pre[w] += 1
                                fail_sizing_side_pre[str(ev.get("side") or "UNK")] += 1
                            else:
                                fail_sizing_in[w] += 1
                                fail_sizing_side_in[str(ev.get("side") or "UNK")] += 1
                            continue
                        n_ev_sized += 1
                        if str(ev.get("regime")) == "Pre":
                            n_ev_sized_pre += 1
                        else:
                            n_ev_sized_in += 1
                        # aplica budget por jogo como padrão
                        mid = int(ev.get("match_id"))
                        if ev.get("side") == "Back":
                            # budget por match (fixo ou adaptativo por concentração)
                            bud_m = float(bud_back)
                            if bud_risk_mode == "signals_sqrt":
                                bud_m = float(bud_back) / max(1.0, math.sqrt(float(cnt_back.get(mid, 1))))
                            elif bud_risk_mode == "signals_linear":
                                bud_m = float(bud_back) / max(1.0, float(cnt_back.get(mid, 1)))
                            cap_sig_m = float(bud_cap_sig_frac) * float(bud_m)
                            rem = max(0.0, float(bud_m) - float(spent_back.get(mid, 0.0)))
                            if rem <= 0:
                                continue
                            exp_use = min(float(exp), float(rem), float(cap_sig_m))
                            if exp_use <= 0:
                                continue
                            # proporcionalmente reduz stake_eq
                            ratio = exp_use / max(1e-9, float(exp))
                            exp = exp_use
                            st_eq = float(st_eq) * float(ratio)
                            spent_back[mid] = float(spent_back.get(mid, 0.0)) + float(exp_use)
                        else:
                            bud_m = float(bud_lay)
                            if bud_risk_mode == "signals_sqrt":
                                bud_m = float(bud_lay) / max(1.0, math.sqrt(float(cnt_lay.get(mid, 1))))
                            elif bud_risk_mode == "signals_linear":
                                bud_m = float(bud_lay) / max(1.0, float(cnt_lay.get(mid, 1)))
                            cap_sig_m = float(bud_cap_sig_frac) * float(bud_m)
                            rem = max(0.0, float(bud_m) - float(spent_lay.get(mid, 0.0)))
                            if rem <= 0:
                                continue
                            exp_use = min(float(exp), float(rem), float(cap_sig_m))
                            if exp_use <= 0:
                                continue
                            ratio = exp_use / max(1e-9, float(exp))
                            exp = exp_use
                            st_eq = float(st_eq) * float(ratio)
                            spent_lay[mid] = float(spent_lay.get(mid, 0.0)) + float(exp_use)
                        n_ev_after_budget += 1
                        if str(ev.get("regime")) == "Pre":
                            n_ev_after_budget_pre += 1
                        else:
                            n_ev_after_budget_in += 1
                        turn_all += float(st_eq)
                        if ev.get("side") == "Back":
                            back_st_all += float(exp)
                        else:
                            lay_liab_all += float(exp)
                        _append_job(ev, float(exp))
                        # breakdown Pre/In (turnover + exposição)
                        if ev.get("regime") == "Pre":
                            turn_pre += float(st_eq)
                            if ev.get("side") == "Back":
                                back_pre_all += float(exp)
                            else:
                                lay_pre_all += float(exp)
                        else:
                            turn_in += float(st_eq)
                            if ev.get("side") == "Back":
                                back_in_all += float(exp)
                            else:
                                lay_in_all += float(exp)
                        # lucro observado se ROI existe
                        if ev.get("roi") is None:
                            continue
                        turn_roi += float(st_eq)
                        roi_pct = float(ev.get("roi"))
                        if ev.get("side") == "Back":
                            back_st_roi += float(exp)
                            pnl_obs += float(exp) * roi_pct / 100.0
                        else:
                            lay_liab_roi += float(exp)
                            pnl_obs += float(exp) * roi_pct / 100.0
                        # breakdown Pre/In (lucro + exposições com ROI)
                        if ev.get("regime") == "Pre":
                            pnl_obs_pre += float(exp) * roi_pct / 100.0
                            if ev.get("side") == "Back":
                                back_pre_roi += float(exp)
                            else:
                                lay_pre_roi += float(exp)
                        else:
                            pnl_obs_in += float(exp) * roi_pct / 100.0
                            if ev.get("side") == "Back":
                                back_in_roi += float(exp)
                            else:
                                lay_in_roi += float(exp)

                    pnl_exp = pnl_obs
                    pnl_exp_pre = pnl_obs_pre
                    pnl_exp_in = pnl_obs_in
                    if wf_expand:
                        # expande separadamente por tipo de exposição (stake Back, liability Lay)
                        scale_back = (back_st_all / back_st_roi) if back_st_roi > 0 else 1.0
                        scale_lay = (lay_liab_all / lay_liab_roi) if lay_liab_roi > 0 else 1.0
                        # aproxima: separa pnl por lado no loop acima? como não guardamos, re-estima com base em shares
                        # fallback: escala global por turnover quando não há como separar
                        if back_st_roi > 0 and lay_liab_roi > 0:
                            # estima split por peso de exposição observada
                            w_back = back_st_roi / (back_st_roi + lay_liab_roi)
                            w_lay = 1.0 - w_back
                            pnl_exp = float(pnl_obs) * (w_back * scale_back + w_lay * scale_lay)
                        elif back_st_roi > 0:
                            pnl_exp = float(pnl_obs) * float(scale_back)
                        elif lay_liab_roi > 0:
                            pnl_exp = float(pnl_obs) * float(scale_lay)
                        else:
                            pnl_exp = float(pnl_obs)
                        # expande por regime (Pre/In) com exposições observadas (já com budget aplicado)
                        if (back_pre_roi + lay_pre_roi) > 0:
                            wbp = back_pre_roi / max(1e-9, (back_pre_roi + lay_pre_roi))
                            wlp = 1.0 - wbp
                            pnl_exp_pre = float(pnl_obs_pre) * (
                                wbp * ((back_pre_all / back_pre_roi) if back_pre_roi > 0 else 1.0)
                                + wlp * ((lay_pre_all / lay_pre_roi) if lay_pre_roi > 0 else 1.0)
                            )
                        if (back_in_roi + lay_in_roi) > 0:
                            wbi = back_in_roi / max(1e-9, (back_in_roi + lay_in_roi))
                            wli = 1.0 - wbi
                            pnl_exp_in = float(pnl_obs_in) * (
                                wbi * ((back_in_all / back_in_roi) if back_in_roi > 0 else 1.0)
                                + wli * ((lay_in_all / lay_in_roi) if lay_in_roi > 0 else 1.0)
                            )

                    # acumula séries diárias (test pode ter vários dias)
                    for dday in test_days:
                        daily_turn[dday] = daily_turn.get(dday, 0.0) + float(turn_all) / max(1, len(test_days))
                        daily_pnl_obs[dday] = daily_pnl_obs.get(dday, 0.0) + float(pnl_obs) / max(1, len(test_days))
                        daily_pnl_exp[dday] = daily_pnl_exp.get(dday, 0.0) + float(pnl_exp) / max(1, len(test_days))
                        daily_turn_pre[dday] = daily_turn_pre.get(dday, 0.0) + float(turn_pre) / max(1, len(test_days))
                        daily_turn_in[dday] = daily_turn_in.get(dday, 0.0) + float(turn_in) / max(1, len(test_days))
                        daily_pnl_obs_pre[dday] = daily_pnl_obs_pre.get(dday, 0.0) + float(pnl_obs_pre) / max(1, len(test_days))
                        daily_pnl_obs_in[dday] = daily_pnl_obs_in.get(dday, 0.0) + float(pnl_obs_in) / max(1, len(test_days))
                        daily_pnl_exp_pre[dday] = daily_pnl_exp_pre.get(dday, 0.0) + float(pnl_exp_pre) / max(1, len(test_days))
                        daily_pnl_exp_in[dday] = daily_pnl_exp_in.get(dday, 0.0) + float(pnl_exp_in) / max(1, len(test_days))

                    if back_st_all > 0:
                        oos_back_stakes_all.append(float(back_st_all))
                    if lay_liab_all > 0:
                        oos_lay_liab_all.append(float(lay_liab_all))

                    # número de combinações "base" (sem a liga, quando wf_key_by_league=1)
                    active_base = set()
                    for k in active:
                        kb = str(k).split("__", 1)[0] if wf_key_by_league else str(k)
                        active_base.add(kb)

                    steps.append(
                        {
                            "train": f"{min(train_days)}→{max(train_days)}",
                            "test": f"{min(test_days)}→{max(test_days)}",
                            "train_days": sorted(train_days),
                            "test_days": sorted(test_days),
                            "active_keys": list(active),
                            "active_n": len(active),
                            "active_n_base": len(active_base),
                            "active_keys_base": sorted(list(active_base)),
                            "oos_matches": len(bym_roi),
                            "oos_mean": oos_mean,
                            "oos_ci": oos_ci,
                            "turn_all": turn_all,
                            "turn_pre": turn_pre,
                            "turn_in": turn_in,
                            "n_ev_elig": int(n_ev_elig),
                            "n_ev_sized": int(n_ev_sized),
                            "n_ev_after_budget": int(n_ev_after_budget),
                            "n_ev_elig_pre": int(n_ev_elig_pre),
                            "n_ev_elig_in": int(n_ev_elig_in),
                            "n_ev_sized_pre": int(n_ev_sized_pre),
                            "n_ev_sized_in": int(n_ev_sized_in),
                            "n_ev_after_budget_pre": int(n_ev_after_budget_pre),
                            "n_ev_after_budget_in": int(n_ev_after_budget_in),
                            "fail_sizing_pre": dict(fail_sizing_pre),
                            "fail_sizing_in": dict(fail_sizing_in),
                            "fail_sizing_side_pre": dict(fail_sizing_side_pre),
                            "fail_sizing_side_in": dict(fail_sizing_side_in),
                            "pnl_obs": pnl_obs,
                            "pnl_exp": pnl_exp,
                            "diag": diag,
                            "liq_mode": wf_liq_mode,
                            "liq_scope": wf_liq_scope,
                            "liq_thr_p50_pre": thr_p50_pre,
                            "liq_thr_p75_pre": thr_p75_pre,
                            "liq_thr_p50_all": thr_p50_all,
                            "liq_thr_p75_all": thr_p75_all,
                            "liq_thr_use": thr_use,
                        }
                    )

                wf_export = str(getattr(args, "wf_export_policy_json", "") or "").strip()
                if wf_export:
                    try:
                        outp = Path(wf_export)
                        outp.parent.mkdir(parents=True, exist_ok=True)
                        payload = {
                            "generated_at": datetime.now(timezone.utc).isoformat(),
                            "report_out": str(out_path),
                            "lookback_days": (int(args.lookback_days) if args.lookback_days is not None else None),
                            "versions": versions,
                            "walkforward": True,
                            "wf": {
                                "train_days": int(getattr(args, "wf_train_days", 2)),
                                "test_days": int(getattr(args, "wf_test_days", 1)),
                                "step_days": int(getattr(args, "wf_step_days", 1)),
                                "min_matches": int(getattr(args, "wf_min_matches", 0)),
                                "key_by_league": bool(getattr(args, "wf_key_by_league", False)),
                                "key_by_league_scope": str(getattr(args, "wf_key_by_league_scope", "pre") or "pre"),
                                "liquidity_mode": str(getattr(args, "wf_liquidity_mode", "none") or "none"),
                                "liquidity_scope": str(getattr(args, "wf_liquidity_scope", "pre") or "pre"),
                                "liquidity_min_limit": float(getattr(args, "wf_liquidity_min_limit", 0.0) or 0.0),
                                "ah_max_abs_line": float(getattr(args, "wf_ah_max_abs_line", 0.0) or 0.0),
                                "ah_scope": str(getattr(args, "wf_ah_scope", "all") or "all"),
                                "scheme_pre": str(getattr(args, "wf_scheme_pre", "") or ""),
                                "scheme_in": str(getattr(args, "wf_scheme_in", "") or ""),
                                "match_budget": bool(getattr(args, "wf_match_budget", False)),
                                "budget_back_frac": float(getattr(args, "wf_budget_back_frac", 0.0) or 0.0),
                                "budget_lay_frac": float(getattr(args, "wf_budget_lay_frac", 0.0) or 0.0),
                                "budget_cap_signal_frac": float(getattr(args, "wf_budget_cap_signal_frac", 0.0) or 0.0),
                                "budget_risk_mode": str(getattr(args, "wf_budget_risk_mode", "fixed") or "fixed"),
                                "shrinkage": bool(getattr(args, "wf_shrinkage", False)),
                                "exclude_exec_buckets": str(getattr(args, "wf_exclude_exec_buckets", "") or ""),
                                "exclude_exec_buckets_back": str(getattr(args, "wf_exclude_exec_buckets_back", "") or ""),
                                "exclude_exec_buckets_lay": str(getattr(args, "wf_exclude_exec_buckets_lay", "") or ""),
                            },
                            "steps": steps,
                            "active_counts": active_counts,
                        }
                        outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                        lines.append(f"\n[INFO] WF policy exportado: `{wf_export}`\n\n")
                        try:
                            print(f"[INFO] WF policy exportado: {wf_export}", flush=True)
                        except Exception:
                            pass
                    except Exception as e:
                        lines.append(f"\n[WARN] Falha ao exportar WF policy JSON ({wf_export}): {e}\n\n")

                wf_key_by_league2 = bool(getattr(args, "wf_key_by_league", False))
                if wf_key_by_league2:
                    lines.append(
                        "| Train window | Test window | #ativas (keys) | #ativas (comb) | Jogos OOS | ROI OOS (mean; IC90) | Turnover (teste) | Turnover Pre | Turnover In | Lucro (estratégia, budget) |\n"
                        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
                    )
                else:
                    lines.append("| Train window | Test window | #ativas | Jogos OOS | ROI OOS (mean; IC90) | Turnover (teste) | Turnover Pre | Turnover In | Lucro (estratégia, budget) |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
                for s in steps[:20]:
                    if wf_key_by_league2:
                        lines.append(
                            f"| {s['train']} | {s['test']} | {s['active_n']} | {int(s.get('active_n_base') or 0)} | {s['oos_matches']} | {_fmt_pct(s['oos_mean'],2)} {_fmt_ci(s['oos_ci'],2)} | "
                            f"{_fmt_num(s.get('turn_all'),2)} | {_fmt_num(s.get('turn_pre'),2)} | {_fmt_num(s.get('turn_in'),2)} | {_fmt_num(s.get('pnl_exp'),2)} |\n"
                        )
                    else:
                        lines.append(
                            f"| {s['train']} | {s['test']} | {s['active_n']} | {s['oos_matches']} | {_fmt_pct(s['oos_mean'],2)} {_fmt_ci(s['oos_ci'],2)} | "
                            f"{_fmt_num(s.get('turn_all'),2)} | {_fmt_num(s.get('turn_pre'),2)} | {_fmt_num(s.get('turn_in'),2)} | {_fmt_num(s.get('pnl_exp'),2)} |\n"
                        )
                # Transparência do turnover (diagnóstico de quedas bruscas)
                lines.append(
                    "\n**Diagnóstico do turnover (por step)**\n\n"
                    "| Test window | N elig (Pre/In) | N sized (Pre/In) | N após budget (Pre/In) | Turnover Pre | Turnover In | Turnover total |\n"
                    "|---|---:|---:|---:|---:|---:|---:|\n"
                )
                for s in steps[:20]:
                    lines.append(
                        f"| {s['test']} | {int(s.get('n_ev_elig_pre') or 0)}/{int(s.get('n_ev_elig_in') or 0)} | {int(s.get('n_ev_sized_pre') or 0)}/{int(s.get('n_ev_sized_in') or 0)} | {int(s.get('n_ev_after_budget_pre') or 0)}/{int(s.get('n_ev_after_budget_in') or 0)} | "
                        f"{_fmt_num(s.get('turn_pre'),2)} | {_fmt_num(s.get('turn_in'),2)} | {_fmt_num(s.get('turn_all'),2)} |\n"
                    )
                lines.append("\n")

                # Diagnóstico causal: por que eventos elegíveis não viram sized (especialmente no Pre)
                lines.append("\n**Falhas de sizing (top motivos; steps recentes)**\n\n")
                lines.append("| Test window | Fail sizing Pre (top) | Fail sizing In (top) |\n|---|---|---|\n")
                for s in steps[-12:]:
                    fp = s.get("fail_sizing_pre") or {}
                    fi = s.get("fail_sizing_in") or {}
                    top_pre = ", ".join([f"{k}×{v}" for k, v in sorted(fp.items(), key=lambda x: x[1], reverse=True)[:3]]) or "—"
                    top_in = ", ".join([f"{k}×{v}" for k, v in sorted(fi.items(), key=lambda x: x[1], reverse=True)[:3]]) or "—"
                    lines.append(f"| {s['test']} | {top_pre} | {top_in} |\n")
                lines.append("\n")

                # Modo rápido: para diagnosticar o colapso do turnover (ex.: 22-02), não precisamos das projeções 12.1/12.2+.
                # Aqui encerramos o relatório após as tabelas essenciais e escrevemos somente o bloco OOS.
                if bool(getattr(args, "only_oos", False)):
                    try:
                        def _extract_oos_block(doc_lines: List[str]) -> List[str]:
                            i0 = None
                            for ii, ln in enumerate(doc_lines):
                                if str(ln).startswith("## 12) OOS walk-forward"):
                                    i0 = ii
                                    break
                            if i0 is None:
                                return doc_lines
                            j0 = len(doc_lines)
                            for jj in range(i0 + 1, len(doc_lines)):
                                ln = str(doc_lines[jj] or "")
                                if ln.startswith("## ") and (not ln.startswith("## 12) OOS walk-forward")):
                                    j0 = jj
                                    break
                            hdr = [
                                "## OOS (extraído) — modo `--only-oos`\n\n",
                                "_Este arquivo contém **apenas** o bloco de walk-forward, para inspeção rápida (turnover, N elig/sized, e falhas de sizing)._ \n\n",
                            ]
                            return hdr + doc_lines[i0:j0]

                        out_lines = _extract_oos_block(lines)
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        out_path.write_text("".join(out_lines), encoding="utf-8")
                        print(f"Relatório gerado em: {out_path}")
                    except Exception as e:
                        print(f"[WARN] Falha ao escrever saída only-oos: {e}")
                    return 0
                if wf_key_by_league2:
                    scope = str(getattr(args, "wf_key_by_league_scope", "pre") or "pre").strip().lower()
                    lines.append(
                        f"\n_Neste modo, '#ativas (keys)' conta chaves por liga quando aplicável (scope='{scope}'); "
                        f"'#ativas (comb)' agrega ignorando liga._\n\n"
                    )
                if len(steps) > 20:
                    lines.append(f"\n*(mostrando apenas 20 passos; total passos={len(steps)})*\n\n")

                lines.append("\n**Frequência de ativação por combinação (quantas janelas ela entrou como ativa)**\n\n")
                lines.append("| Combinação | #steps ativa |\n|---|---:|\n")
                for k, c in sorted(active_counts.items(), key=lambda x: x[1], reverse=True):
                    lines.append(f"| {k} | {c} |\n")

                # Transparência: métricas por combinação no treino (por janela)
                lines.append("\n### 12.A Transparência da seleção: métricas por combinação no treino\n")
                lines.append(
                    "Para cada janela de treino, mostramos as métricas usadas para decidir se cada combinação ficou **ativa** ou não. "
                    "Isso ajuda a entender, por exemplo, por que nenhuma Lay entrou em algumas janelas (geralmente N insuficiente, ROI sig<0, ou ROI<=0 com CLV<=0 no pre‑match).\n\n"
                )
            if wf_min_m > 0:
                lines.append(f"**Regra de elegibilidade (todas as combinações):** exige `N_ROI >= wf_min_matches` (aqui: {wf_min_m}).\n\n")
            else:
                lines.append("**Regra de elegibilidade (todas as combinações):** `wf_min_matches=0` ⇒ mínimo de N **desligado**.\n\n")
                wf_key_by_league = bool(getattr(args, "wf_key_by_league", False))
                combos_all = [
                    "Back_Pre_Any", "Back_In_Any",
                    "Lay_Pre_Yes", "Lay_Pre_No", "Lay_In_Yes", "Lay_In_No",
                ] if not wf_key_by_league else []
                for st in steps[:10]:
                    lines.append(f"**Train {st['train']} → Test {st['test']}**\n\n")
                    hdr_key = "Chave (combinação×liga)" if wf_key_by_league else "Combinação"
                    if wf_use_shrink:
                        lines.append(
                            f"| {hdr_key} | Ativa? | Jogos treino (tot/CLV/ROI) | CLV mean (IC90) | ROI mean (IC90) | ROI mean (shrunk) | ROI q30 | Motivo |\n"
                            "|---|---|---:|---:|---:|---:|---:|---|\n"
                        )
                    else:
                        lines.append(
                            f"| {hdr_key} | Ativa? | Jogos treino (tot/CLV/ROI) | CLV mean (IC90) | ROI mean (IC90) | ROI q30 | Motivo |\n"
                            "|---|---|---:|---:|---:|---:|---|\n"
                        )
                    diag = st.get("diag") or {}
                    keys_tbl = combos_all
                    if wf_key_by_league:
                        keys_tbl = sorted(
                            list(diag.keys()),
                            key=lambda kk: int((diag.get(kk) or {}).get("train_matches_total") or 0),
                            reverse=True,
                        )[:20]
                    for k in keys_tbl:
                        d = diag.get(k) or {}
                        ok = bool(d.get("ok"))
                        tm = int(d.get("train_matches_total") or 0)
                        tclv = d.get("train_matches_clv")
                        troi = d.get("train_matches_roi")
                        tclv_s = str(int(tclv)) if tclv is not None else "—"
                        troi_s = str(int(troi)) if troi is not None else "—"
                        clv_mean = _safe_float(d.get("clv_mean"))
                        clv_ci = d.get("clv_ci90")
                        clv_val = "—"
                        if clv_mean is not None:
                            clv_val = f"{_fmt_pct(clv_mean,2)} {_fmt_ci(clv_ci,2)}"
                        roi_mean = _safe_float(d.get("roi_mean"))
                        roi_ci = d.get("roi_ci90")
                        roi_val = "—"
                        if roi_mean is not None:
                            roi_val = f"{_fmt_pct(roi_mean,2)} {_fmt_ci(roi_ci,2)}"
                        roi_mean_eff = _safe_float(d.get("roi_mean_eff"))
                        roi_eff_val = "—"
                        if roi_mean_eff is not None:
                            roi_eff_val = f"{_fmt_pct(roi_mean_eff,2)}"
                        roi_q30 = _safe_float(d.get("roi_q30"))
                        if wf_use_shrink:
                            lines.append(
                                f"| {k} | {'SIM' if ok else 'NÃO'} | {tm} / {tclv_s} / {troi_s} | {clv_val} | {roi_val} | {roi_eff_val} | {_fmt_pct(roi_q30,2)} | {str(d.get('reason') or '')} |\n"
                            )
                        else:
                            lines.append(
                                f"| {k} | {'SIM' if ok else 'NÃO'} | {tm} / {tclv_s} / {troi_s} | {clv_val} | {roi_val} | {_fmt_pct(roi_q30,2)} | {str(d.get('reason') or '')} |\n"
                            )
                    lines.append("\n")

                lines.append(
                    "\nNotas importantes:\n"
                    "- Se `Jogos OOS` for baixo em muitos passos, você ainda não tem volume suficiente para decisões por combinação. "
                    "Nesse cenário faz sentido **Bayes hierárquico (partial pooling)** para estabilizar estimativas.\n"
                    "- **Lucro (estratégia, budget)** acima já incorpora a política de risco por jogo (match budget) e é a métrica principal.\n"
                    "- O walk-forward usa ROI no **ponto de entrada**: Back em `t0`; Lay em `t_reversal` quando existir, senão `t_last` (~t+20s).\n"
                    "- Para Lay pre-match, o CLV usado na seleção é `clv_conv = -(entry-closing)/closing`, ou seja **Lay “bom” tende a ser positivo**.\n"
                    "- Para pre-match, também é útil monitorar CLV OOS (menos dependente de resultados), mas CLV mede qualidade de entrada, não P&L.\n\n"
                )
                lines.append(
                    "**O que significa 'Bayes hierárquico / partial pooling' aqui?**\n\n"
                    "Quando você tem poucas partidas por combinação na janela de treino/teste, o estimador (ex.: ROI p30) fica muito ruidoso e pode alternar sinal por acaso. "
                    "O Bayes hierárquico modela cada combinação como um desvio de um **efeito global** (ex.: ROI médio global do live) e aplica **shrinkage**: "
                    "combinações com pouco N são puxadas para o global; combinações com muito N “ganham identidade própria”.\n\n"
                    "Na prática isso reduz falsos positivos/negativos no rolling e torna a seleção mais estável quando o volume ainda é baixo.\n\n"
                )

                # ------------------------------------------------------------
                # 12.1 Estimativa do tamanho da oportunidade (30 dias) — OOS
                # ------------------------------------------------------------
                lines.append("### 12.1 Estimativa 30 dias (OOS): turnover, lucro, banca, ROI/banca e drawdown\n")
                lines.append(
                    "Esta estimativa usa o walk-forward acima como **simulador OOS**. "
                    "O lucro pode ser reportado em duas versões:\n\n"
                    "- **obs.**: apenas jogos com ROI (placar) disponível.\n"
                    "- **exp.**: expande o lucro para a população elegível usando scaling por exposição/turnover (assume missing-at-random condicional à estratégia).\n\n"
                )
                lines.append(
                    f"**Padrão de risco**: P&L aqui já é calculado com **budget por jogo (match_id)** consumido ao longo do tempo "
                    f"(Back={bud_back_frac:.2%} da banca ref; Lay={bud_lay_frac:.2%} em liability; cap por sinal={bud_cap_sig_frac:.0%} do budget; "
                    f"mode={bud_risk_mode}).\n\n"
                )
                lines.append(
                    f"**Sizing FLAT (quando aplicável no WF)**: Back stake={_fmt_num(wf_flat_stake_back,2)} | Lay liability={_fmt_num(wf_flat_liab_lay,2)}.\n\n"
                )

                oos_days = sorted(daily_turn.keys())
                n_oos_days = len(oos_days) if oos_days else 0
                # Dias "operacionais": houve ao menos 1 evento OK+conf naquele dia (evita tratar downtime como 0 oportunidade)
                try:
                    oos_days_ok = [d for d in oos_days if int(day_ok_cnt.get(d, 0)) > 0]
                except Exception:
                    oos_days_ok = []
                n_oos_days_ok = len(oos_days_ok)
                turn_sum = float(sum(daily_turn.values())) if daily_turn else 0.0
                pnl_obs_sum = float(sum(daily_pnl_obs.values())) if daily_pnl_obs else 0.0
                pnl_exp_sum = float(sum(daily_pnl_exp.values())) if daily_pnl_exp else 0.0
                pnl_obs_pre_sum = float(sum(daily_pnl_obs_pre.values())) if daily_pnl_obs_pre else 0.0
                pnl_obs_in_sum = float(sum(daily_pnl_obs_in.values())) if daily_pnl_obs_in else 0.0
                pnl_exp_pre_sum = float(sum(daily_pnl_exp_pre.values())) if daily_pnl_exp_pre else 0.0
                pnl_exp_in_sum = float(sum(daily_pnl_exp_in.values())) if daily_pnl_exp_in else 0.0
                turn_pre_sum = float(sum(daily_turn_pre.values())) if daily_turn_pre else 0.0
                turn_in_sum = float(sum(daily_turn_in.values())) if daily_turn_in else 0.0
                horizon = 30.0
                scale = (horizon / float(n_oos_days)) if n_oos_days > 0 else None
                scale_ok = (horizon / float(n_oos_days_ok)) if n_oos_days_ok > 0 else None

                # Projeção "calendário": inclui dias sem OK como 0 (boa para refletir downtime).
                turn_30d = float(turn_sum) * float(scale) if scale is not None else None
                profit_obs_30d = float(pnl_obs_sum) * float(scale) if scale is not None else None
                profit_exp_30d = float(pnl_exp_sum) * float(scale) if scale is not None else None

                # Projeção "condicional a dia OK": exclui dias sem OK do denominador (boa para não deturpar edge quando operacional falhou).
                turn_30d_ok = float(turn_sum) * float(scale_ok) if scale_ok is not None else None
                profit_obs_30d_ok = float(pnl_obs_sum) * float(scale_ok) if scale_ok is not None else None
                profit_exp_30d_ok = float(pnl_exp_sum) * float(scale_ok) if scale_ok is not None else None
                profit_obs_pre_30d = float(pnl_obs_pre_sum) * float(scale) if scale is not None else None
                profit_obs_in_30d = float(pnl_obs_in_sum) * float(scale) if scale is not None else None
                profit_exp_pre_30d = float(pnl_exp_pre_sum) * float(scale) if scale is not None else None
                profit_exp_in_30d = float(pnl_exp_in_sum) * float(scale) if scale is not None else None
                turn_pre_30d = float(turn_pre_sum) * float(scale) if scale is not None else None
                turn_in_30d = float(turn_in_sum) * float(scale) if scale is not None else None

                # banca por risco (unitária) e por liquidez (simultânea)
                bank_back_p99 = _pctl(oos_back_stakes_all, 99) if oos_back_stakes_all else None
                bank_lay_p99 = _pctl(oos_lay_liab_all, 99) if oos_lay_liab_all else None
                bank_risk = (float(bank_back_p99 or 0.0) + float(bank_lay_p99 or 0.0)) if (bank_back_p99 is not None or bank_lay_p99 is not None) else None
                bank_liq = _liq_p99_from_jobs(oos_jobs)
                bank_eff = None
                if bank_risk is not None or bank_liq is not None:
                    bank_eff = max(float(bank_risk or 0.0), float(bank_liq or 0.0))

                # drawdown p95 via bootstrap de dias OOS (obs/exp)
                dd_obs_mean, dd_obs_p95 = _bootstrap_dd(list(daily_pnl_obs.values()), horizon_days=30, n_boot=2000)
                dd_exp_mean, dd_exp_p95 = _bootstrap_dd(list(daily_pnl_exp.values()), horizon_days=30, n_boot=2000)
                dd_obs_mean_ok, dd_obs_p95_ok = _bootstrap_dd([daily_pnl_obs.get(d, 0.0) for d in oos_days_ok], horizon_days=30, n_boot=2000)
                dd_exp_mean_ok, dd_exp_p95_ok = _bootstrap_dd([daily_pnl_exp.get(d, 0.0) for d in oos_days_ok], horizon_days=30, n_boot=2000)

                roi_bank_obs = (float(profit_obs_30d) / float(bank_eff) * 100.0) if (profit_obs_30d is not None and bank_eff and bank_eff > 0) else None
                roi_bank_exp = (float(profit_exp_30d) / float(bank_eff) * 100.0) if (profit_exp_30d is not None and bank_eff and bank_eff > 0) else None
                roi_bank_obs_ok = (float(profit_obs_30d_ok) / float(bank_eff) * 100.0) if (profit_obs_30d_ok is not None and bank_eff and bank_eff > 0) else None
                roi_bank_exp_ok = (float(profit_exp_30d_ok) / float(bank_eff) * 100.0) if (profit_exp_30d_ok is not None and bank_eff and bank_eff > 0) else None

                lines.append("| Premissa | Valor |\n|---|---:|\n")
                lines.append(f"| Train mode (OOS) | `{wf_train_mode}` |\n")
                lines.append(f"| Scheme pre-match (OOS) | `{wf_scheme_pre}` |\n")
                lines.append(f"| Scheme in-match (OOS) | `{wf_scheme_in}` |\n")
                lines.append(f"| Expansão missing ROI | {'ON' if wf_expand else 'OFF'} |\n")
                lines.append(f"| Dias OOS (calendário de teste) | {n_oos_days} |\n")
                lines.append(f"| Dias OOS com OK (>=1 evento OK/conf) | {n_oos_days_ok} |\n")
                lines.append(f"| Turnover 30d (proj., calendário) | {_fmt_num(turn_30d,2)} |\n")
                lines.append(f"| Turnover 30d (proj., cond OK) | {_fmt_num(turn_30d_ok,2)} |\n")
                lines.append(f"| Turnover 30d (Pre/In) | {_fmt_num(turn_pre_30d,2)} / {_fmt_num(turn_in_30d,2)} |\n")
                lines.append(f"| Lucro 30d (obs., calendário) | {_fmt_num(profit_obs_30d,2)} |\n")
                lines.append(f"| Lucro 30d (obs., cond OK) | {_fmt_num(profit_obs_30d_ok,2)} |\n")
                lines.append(f"| Lucro 30d (obs.) Pre/In | {_fmt_num(profit_obs_pre_30d,2)} / {_fmt_num(profit_obs_in_30d,2)} |\n")
                lines.append(f"| Lucro 30d (exp., calendário) | {_fmt_num(profit_exp_30d,2)} |\n")
                lines.append(f"| Lucro 30d (exp., cond OK) | {_fmt_num(profit_exp_30d_ok,2)} |\n")
                lines.append(f"| Lucro 30d (exp.) Pre/In | {_fmt_num(profit_exp_pre_30d,2)} / {_fmt_num(profit_exp_in_30d,2)} |\n")
                lines.append(f"| Banca risco p99 (Back+Lay) | {_fmt_num(bank_risk,2)} |\n")
                lines.append(f"| Banca liquidez p99 (+buf) | {_fmt_num(bank_liq,2)} |\n")
                lines.append(f"| Banca recomendada (max) | {_fmt_num(bank_eff,2)} |\n")
                lines.append(f"| ROI/banca 30d (obs., calendário) | {_fmt_num(roi_bank_obs,2)}% |\n")
                lines.append(f"| ROI/banca 30d (obs., cond OK) | {_fmt_num(roi_bank_obs_ok,2)}% |\n")
                lines.append(f"| ROI/banca 30d (exp., calendário) | {_fmt_num(roi_bank_exp,2)}% |\n")
                lines.append(f"| ROI/banca 30d (exp., cond OK) | {_fmt_num(roi_bank_exp_ok,2)}% |\n")
                lines.append(f"| DD 30d p95 (obs., calendário) | {_fmt_num(dd_obs_p95,2)} |\n")
                lines.append(f"| DD 30d p95 (obs., cond OK) | {_fmt_num(dd_obs_p95_ok,2)} |\n")
                lines.append(f"| DD 30d p95 (exp., calendário) | {_fmt_num(dd_exp_p95,2)} |\n")
                lines.append(f"| DD 30d p95 (exp., cond OK) | {_fmt_num(dd_exp_p95_ok,2)} |\n")
                lines.append("\n")

                # Diagnóstico rápido: de onde vem o P&L (Pre vs In)
                roi_turn_pre = (float(profit_exp_pre_30d) / float(turn_pre_30d) * 100.0) if (profit_exp_pre_30d is not None and turn_pre_30d and float(turn_pre_30d) > 0) else None
                roi_turn_in = (float(profit_exp_in_30d) / float(turn_in_30d) * 100.0) if (profit_exp_in_30d is not None and turn_in_30d and float(turn_in_30d) > 0) else None
                lines.append("**Ablation (OOS): operar só Pre vs só In (com o MESMO budget/sizing)**\n\n")
                lines.append("| Universo | Turnover 30d | Lucro 30d (exp.) | ROI/turnover 30d (exp.) |\n")
                lines.append("|---|---:|---:|---:|\n")
                lines.append(f"| Só Pre | {_fmt_num(turn_pre_30d,2)} | {_fmt_num(profit_exp_pre_30d,2)} | {_fmt_num(roi_turn_pre,2)}% |\n")
                lines.append(f"| Só In | {_fmt_num(turn_in_30d,2)} | {_fmt_num(profit_exp_in_30d,2)} | {_fmt_num(roi_turn_in,2)}% |\n")
                lines.append("\n")

                # ------------------------------------------------------------
                # 12.2 Governança por jogo (budget por match_id) — sensibilidade
                # ------------------------------------------------------------
                if bool(getattr(args, "wf_match_budget", True)):
                    lines.append("### 12.2 Governança de exposição por jogo (budget por `match_id`) — sensibilidade\n")
                    lines.append(
                        "Objetivo: evitar concentração quando um mesmo jogo gera muitos sinais. "
                        "Simulamos um orçamento de exposição por jogo, consumido ao longo do tempo.\n\n"
                        "- **Back** consome budget em **stake**.\n"
                        "- **Lay** consome budget em **liability**.\n"
                        "- Também aplicamos um cap por sinal como fração do budget do jogo (para não gastar tudo no 1º sinal).\n\n"
                        "Importante: o budget é parametrizado como fração de uma referência de banca. Aqui usamos:\n"
                        "- se `--kelly-bankroll` estiver setado: essa banca explícita;\n"
                        "- senão: a `Banca recomendada (max)` estimada na 12.1 (baseline).\n\n"
                    )

                    bank_ref_budget = _safe_float(getattr(args, "kelly_bankroll", None))
                    if bank_ref_budget is None or bank_ref_budget <= 0:
                        bank_ref_budget = bank_eff
                    bank_ref_budget = float(bank_ref_budget or 1.0)

                    def _liq_ok_policy(ev: dict, st: dict) -> bool:
                        # política adicional: linha AH (proxy de liquidez por mercado)
                        if not _ah_ok(ev):
                            return False
                        if wf_liq_mode == "none":
                            return True
                        if wf_liq_scope == "pre" and str(ev.get("regime")) != "Pre":
                            return True
                        lim = _safe_float(ev.get("liq_limit"))
                        if lim is None or float(lim) <= 0 or not math.isfinite(float(lim)):
                            return False
                        if wf_liq_mode == "gate_min":
                            return float(lim) >= float(wf_liq_min)
                        thr = _safe_float(st.get("liq_thr_use"))
                        if thr is None or float(thr) <= 0 or not math.isfinite(float(thr)):
                            return True
                        return float(lim) >= float(thr)

                    def _simulate_with_budget(
                        *,
                        bud_back_frac: float,
                        bud_lay_frac: float,
                        cap_signal_frac: float,
                        risk_mode: str = "fixed",
                    ) -> Dict[str, Any]:
                        bud_back = float(bud_back_frac) * float(bank_ref_budget)
                        bud_lay = float(bud_lay_frac) * float(bank_ref_budget)
                        cap_sig_back = float(cap_signal_frac) * float(bud_back)
                        cap_sig_lay = float(cap_signal_frac) * float(bud_lay)

                        dturn: Dict[str, float] = {}
                        dpnl_obs: Dict[str, float] = {}
                        dpnl_exp: Dict[str, float] = {}
                        back_exps: List[float] = []
                        lay_exps: List[float] = []
                        jobs: List[Tuple[datetime, datetime, float]] = []

                        for st in steps:
                            test_days = set(st.get("test_days") or [])
                            active_keys = set(st.get("active_keys") or [])
                            if not test_days or not active_keys:
                                continue

                            # elegíveis no teste (inclui sem ROI para turnover)
                            test_elig = [e for e in combo_events if e["day"] in test_days and _key(e) in active_keys and _liq_ok_policy(e, st)]
                            # ROI do treino por combinação (necessário se wf_scheme_* = ROI_TRAIN)
                            roi_map = {
                                k: _safe_float((st.get("diag") or {}).get(k, {}).get("roi_mean"))
                                for k in active_keys
                            }
                            # ordena por tempo para consumir budget de forma realista
                            def _ts_ev(ev: dict) -> float:
                                d0 = audit_by_id.get(int(ev.get("audit_id")))
                                ts = d0.get("audited_at") if d0 else None
                                if isinstance(ts, datetime):
                                    return ts.timestamp()
                                return 0.0
                            test_elig.sort(key=_ts_ev)
                            cnt_back = Counter(int(ev.get("match_id")) for ev in test_elig if ev.get("side") == "Back")
                            cnt_lay = Counter(int(ev.get("match_id")) for ev in test_elig if ev.get("side") != "Back")

                            spent_back: Dict[int, float] = {}
                            spent_lay: Dict[int, float] = {}

                            back_all = back_roi = 0.0
                            lay_all = lay_roi = 0.0
                            pnl_obs = 0.0
                            turn_all = 0.0

                            for ev in test_elig:
                                res_sz = _sizing_for_event(ev, roi_train_map=roi_map)
                                if isinstance(res_sz, tuple) and len(res_sz) == 2:
                                    st_eq, exp = res_sz
                                else:
                                    st_eq, exp, _why = res_sz
                                if st_eq is None or exp is None:
                                    continue
                                mid = int(ev.get("match_id"))
                                if ev.get("side") == "Back":
                                    bud_m = float(bud_back)
                                    if risk_mode == "signals_sqrt":
                                        bud_m = float(bud_back) / max(1.0, math.sqrt(float(cnt_back.get(mid, 1))))
                                    elif risk_mode == "signals_linear":
                                        bud_m = float(bud_back) / max(1.0, float(cnt_back.get(mid, 1)))
                                    cap_sig_m = float(cap_signal_frac) * float(bud_m)
                                    rem = max(0.0, float(bud_m) - float(spent_back.get(mid, 0.0)))
                                    if rem <= 0:
                                        continue
                                    exp_use = min(float(exp), float(rem), float(cap_sig_m))
                                    if exp_use <= 0:
                                        continue
                                    spent_back[mid] = float(spent_back.get(mid, 0.0)) + float(exp_use)
                                    ratio = float(exp_use) / max(1e-9, float(exp))
                                    exp = float(exp_use)
                                    st_eq = float(st_eq) * float(ratio)
                                    turn_all += st_eq
                                    back_all += exp
                                    _append_job(ev, exp)
                                    if ev.get("roi") is not None:
                                        pnl_obs += exp * float(ev.get("roi")) / 100.0
                                        back_roi += exp
                                else:
                                    bud_m = float(bud_lay)
                                    if risk_mode == "signals_sqrt":
                                        bud_m = float(bud_lay) / max(1.0, math.sqrt(float(cnt_lay.get(mid, 1))))
                                    elif risk_mode == "signals_linear":
                                        bud_m = float(bud_lay) / max(1.0, float(cnt_lay.get(mid, 1)))
                                    cap_sig_m = float(cap_signal_frac) * float(bud_m)
                                    rem = max(0.0, float(bud_m) - float(spent_lay.get(mid, 0.0)))
                                    if rem <= 0:
                                        continue
                                    exp_use = min(float(exp), float(rem), float(cap_sig_m))
                                    if exp_use <= 0:
                                        continue
                                    spent_lay[mid] = float(spent_lay.get(mid, 0.0)) + float(exp_use)
                                    ratio = float(exp_use) / max(1e-9, float(exp))
                                    exp = float(exp_use)  # liability
                                    st_eq = float(st_eq) * float(ratio)
                                    turn_all += st_eq
                                    lay_all += exp
                                    _append_job(ev, exp)
                                    if ev.get("roi") is not None:
                                        pnl_obs += exp * float(ev.get("roi")) / 100.0
                                        lay_roi += exp

                            pnl_exp = pnl_obs
                            if wf_expand:
                                scale_back = (back_all / back_roi) if back_roi > 0 else 1.0
                                scale_lay = (lay_all / lay_roi) if lay_roi > 0 else 1.0
                                if back_roi > 0 and lay_roi > 0:
                                    w_back = back_roi / (back_roi + lay_roi)
                                    w_lay = 1.0 - w_back
                                    pnl_exp = float(pnl_obs) * (w_back * scale_back + w_lay * scale_lay)
                                elif back_roi > 0:
                                    pnl_exp = float(pnl_obs) * float(scale_back)
                                elif lay_roi > 0:
                                    pnl_exp = float(pnl_obs) * float(scale_lay)

                            for dday in test_days:
                                dturn[dday] = dturn.get(dday, 0.0) + float(turn_all) / max(1, len(test_days))
                                dpnl_obs[dday] = dpnl_obs.get(dday, 0.0) + float(pnl_obs) / max(1, len(test_days))
                                dpnl_exp[dday] = dpnl_exp.get(dday, 0.0) + float(pnl_exp) / max(1, len(test_days))

                            if back_all > 0:
                                back_exps.append(float(back_all))
                            if lay_all > 0:
                                lay_exps.append(float(lay_all))
                        # projeção 30d
                        oos_days2 = sorted(dturn.keys())
                        n_days2 = len(oos_days2) if oos_days2 else 0
                        scale2 = (30.0 / float(n_days2)) if n_days2 > 0 else None
                        turn_30 = float(sum(dturn.values())) * float(scale2) if scale2 is not None else None
                        prof_obs_30 = float(sum(dpnl_obs.values())) * float(scale2) if scale2 is not None else None
                        prof_exp_30 = float(sum(dpnl_exp.values())) * float(scale2) if scale2 is not None else None
                        # banca
                        bank_back = _pctl(back_exps, 99) if back_exps else None
                        bank_lay = _pctl(lay_exps, 99) if lay_exps else None
                        bank_risk2 = (float(bank_back or 0.0) + float(bank_lay or 0.0)) if (bank_back is not None or bank_lay is not None) else None
                        bank_liq2 = _liq_p99_from_jobs(jobs)
                        bank_eff2 = None
                        if bank_risk2 is not None or bank_liq2 is not None:
                            bank_eff2 = max(float(bank_risk2 or 0.0), float(bank_liq2 or 0.0))
                        roi_bank2 = (float(prof_exp_30) / float(bank_eff2) * 100.0) if (prof_exp_30 is not None and bank_eff2 and bank_eff2 > 0) else None
                        dd_mean2, dd_p952 = _bootstrap_dd(list(dpnl_exp.values()), horizon_days=30, n_boot=2000)
                        return {
                            "turn_30d": turn_30,
                            "profit_30d_exp": prof_exp_30,
                            "bank_eff": bank_eff2,
                            "roi_bank_30d": roi_bank2,
                            "dd_p95": dd_p952,
                            "days": n_days2,
                        }

                    scenarios = [
                        ("BUDGET_0.50%/0.25% cap25%", 0.005, 0.0025, 0.25),
                        ("BUDGET_1.00%/0.50% cap33%", 0.010, 0.0050, 0.33),
                        ("BUDGET_2.00%/1.00% cap50%", 0.020, 0.0100, 0.50),
                        ("BUDGET_3.00%/1.50% cap33%", 0.030, 0.0150, 0.33),
                        ("BUDGET_4.00%/2.00% cap33%", 0.040, 0.0200, 0.33),
                        ("BUDGET_3.00%/1.50% cap50%", 0.030, 0.0150, 0.50),
                        ("BUDGET_4.00%/2.00% cap50%", 0.040, 0.0200, 0.50),
                        # Sensibilidade: Lay budget = Back budget (não penaliza Lay por default)
                        ("BUDGET_EQ_0.50%/0.50% cap25%", 0.005, 0.0050, 0.25),
                        ("BUDGET_EQ_1.00%/1.00% cap33%", 0.010, 0.0100, 0.33),
                        ("BUDGET_EQ_2.00%/2.00% cap50%", 0.020, 0.0200, 0.50),
                        ("BUDGET_EQ_3.00%/3.00% cap33%", 0.030, 0.0300, 0.33),
                        ("BUDGET_EQ_4.00%/4.00% cap33%", 0.040, 0.0400, 0.33),
                        ("BUDGET_EQ_3.00%/3.00% cap50%", 0.030, 0.0300, 0.50),
                        ("BUDGET_EQ_4.00%/4.00% cap50%", 0.040, 0.0400, 0.50),
                    ]
                    lines.append(
                        f"Referência de banca p/ budget: {_fmt_num(bank_ref_budget,2)} | "
                        f"budgets por jogo aplicados em stake (Back) e liability (Lay).\n\n"
                    )
                    lines.append("| Cenário | Turnover 30d | Lucro 30d (exp.) | Banca rec. (max) | ROI/banca 30d (exp.) | DD 30d p95 (exp.) |\n")
                    lines.append("|---|---:|---:|---:|---:|---:|\n")
                    # baseline = sem budget por jogo (12.1)
                    lines.append(
                        f"| BASELINE (sem budget) | {_fmt_num(turn_30d,2)} | {_fmt_num(profit_exp_30d,2)} | {_fmt_num(bank_eff,2)} | {_fmt_num(roi_bank_exp,2)}% | {_fmt_num(dd_exp_p95,2)} |\n"
                    )
                    for name, bbf, blf, csf in scenarios:
                        r = _simulate_with_budget(bud_back_frac=bbf, bud_lay_frac=blf, cap_signal_frac=csf, risk_mode="fixed")
                        lines.append(
                            f"| {name} | {_fmt_num(r.get('turn_30d'),2)} | {_fmt_num(r.get('profit_30d_exp'),2)} | {_fmt_num(r.get('bank_eff'),2)} | {_fmt_num(r.get('roi_bank_30d'),2)}% | {_fmt_num(r.get('dd_p95'),2)} |\n"
                        )
                    lines.append("\n**Risk-adaptive (signals_sqrt): sensibilidade variando budgets/caps**\n\n")
                    lines.append("| Cenário (risk) | Turnover 30d | Lucro 30d (exp.) | Banca rec. (max) | ROI/banca 30d (exp.) | DD 30d p95 (exp.) |\n")
                    lines.append("|---|---:|---:|---:|---:|---:|\n")
                    for name, bbf, blf, csf in scenarios:
                        r_ra = _simulate_with_budget(bud_back_frac=bbf, bud_lay_frac=blf, cap_signal_frac=csf, risk_mode="signals_sqrt")
                        lines.append(
                            f"| RISK(signals_sqrt) {name} | {_fmt_num(r_ra.get('turn_30d'),2)} | {_fmt_num(r_ra.get('profit_30d_exp'),2)} | "
                            f"{_fmt_num(r_ra.get('bank_eff'),2)} | {_fmt_num(r_ra.get('roi_bank_30d'),2)}% | {_fmt_num(r_ra.get('dd_p95'),2)} |\n"
                        )
                    lines.append(
                        "\nLeitura:\n"
                        "- Se a curva com budget melhora muito (menos negativo ou mais positivo) com pouca perda de turnover, o problema era **concentração por jogo**.\n"
                        "- Se tudo continuar negativo, o problema é **edge OOS** (principalmente in‑match) e budget só reduz a escala da perda.\n\n"
                    )

                    # ------------------------------------------------------------
                    # 12.2b Sensibilidade por banca (mantendo budgets/caps)
                    # ------------------------------------------------------------
                    raw_grid = str(getattr(args, "wf_bankroll_grid", "") or "").strip()
                    grid: List[float] = []
                    # Default solicitado: se usuário setou `--kelly-bankroll` mas não passou grid,
                    # roda uma sensibilidade padrão de banca (OOS).
                    if (not raw_grid) and (_safe_float(getattr(args, "kelly_bankroll", None)) or 0.0) > 0:
                        raw_grid = "10000,20000,30000,50000,100000"
                    if raw_grid:
                        for part in raw_grid.replace(";", ",").split(","):
                            p = part.strip()
                            if not p:
                                continue
                            try:
                                v = float(p)
                                if v > 0:
                                    grid.append(v)
                            except Exception:
                                continue
                    # sempre inclui o baseline da seção 12.2 (referência atual do budget)
                    if bank_ref_budget and float(bank_ref_budget) > 0:
                        grid.append(float(bank_ref_budget))
                    grid = sorted({round(float(x), 6) for x in grid if x and float(x) > 0})

                    def _sizing_for_event_bankroll(
                        ev: dict,
                        bank_ref: float,
                        *,
                        roi_train_map: Optional[Dict[str, float]] = None,
                    ) -> Tuple[Optional[float], Optional[float]]:
                        """
                        Sizing do evento no OOS para uma banca explícita `bank_ref`.
                        Mantém os mesmos caps (BACK_CAP_FRAC/LAY_CAP_FRAC) e caps por evento (limit).
                        """
                        aid = int(ev.get("audit_id"))
                        d0 = audit_by_id.get(aid)
                        if not d0:
                            return (None, None)
                        sc = _scheme_for_event(ev)

                        if ev.get("side") == "Back":
                            if sc == "FLAT":
                                st = float(wf_flat_stake_back)
                            elif sc == "PROXY":
                                st = _sizing_back(d0, "PROXY", bank_ref=float(back_bank_ref))
                            elif str(sc).upper() == "ROI_TRAIN":
                                k = _key(ev)
                                roi_hat = _safe_float((roi_train_map or {}).get(k))
                                if roi_hat is None:
                                    return (None, None)
                                f = max(0.0, float(roi_hat)) / 100.0
                                cap = BACK_CAP_FRAC * max(1e-9, float(bank_ref))
                                st = min(f * float(bank_ref), cap)
                                st = min(float(st), float(_max_back_stake_event(d0)))
                            elif str(sc).startswith("KELLY"):
                                # Kelly só pre-match
                                if d0.get("is_live") is True:
                                    return (None, None)
                                try:
                                    frac = float(str(sc).split("_")[1])
                                except Exception:
                                    return (None, None)
                                entry_odd = _safe_float(ev.get("entry_odd"))
                                if entry_odd is None:
                                    entry_odd = _safe_float(d0.get("bs_odd"))
                                f0 = _kelly_back_frac(entry_odd, d0.get("closing_odd"))
                                if f0 is None:
                                    return (None, None)
                                f = max(0.0, float(f0)) * float(frac)
                                cap = BACK_CAP_FRAC * max(1e-9, float(bank_ref))
                                st = min(f * float(bank_ref), cap)
                                st = min(float(st), float(_max_back_stake_event(d0)))
                            else:
                                return (None, None)
                            if st is None or float(st) <= 0:
                                return (None, None)
                            return (float(st), float(st))

                        # Lay: exposure = liability; turnover = stake equivalente
                        lay_odd = _safe_float(ev.get("entry_odd"))
                        if lay_odd is None:
                            h = d0.get("hypothesis_details") or {}
                            lay_odd = _safe_float(_get_path(h, ["lay", "odd"])) or _safe_float(d0.get("bs_odd"))
                        if lay_odd is None or float(lay_odd) <= 1.0:
                            return (None, None)

                        if sc == "FLAT":
                            liab = float(wf_flat_liab_lay)
                        elif sc == "PROXY":
                            sized = _sizing_lay_liab(d0, "PROXY", bank_ref=float(lay_bank_ref))
                            if not sized:
                                return (None, None)
                            liab = float(sized[0])
                        elif str(sc).upper() == "ROI_TRAIN":
                            k = _key(ev)
                            roi_hat = _safe_float((roi_train_map or {}).get(k))
                            if roi_hat is None:
                                return (None, None)
                            f = max(0.0, float(roi_hat)) / 100.0
                            cap = LAY_CAP_FRAC * max(1e-9, float(bank_ref))
                            liab = min(f * float(bank_ref), cap)
                        elif str(sc).startswith("KELLY"):
                            if d0.get("is_live") is True:
                                return (None, None)
                            try:
                                frac = float(str(sc).split("_")[1])
                            except Exception:
                                return (None, None)
                            f0 = _kelly_lay_liab_frac(float(lay_odd), d0.get("closing_odd"))
                            if f0 is None:
                                return (None, None)
                            f = max(0.0, float(f0)) * float(frac)
                            cap = LAY_CAP_FRAC * max(1e-9, float(bank_ref))
                            liab = min(f * float(bank_ref), cap)
                        else:
                            return (None, None)

                        # cap por evento via limit (stake max -> liab max)
                        max_st = _max_lay_stake_event(d0)
                        liab = min(float(liab), float(max_st) * max(0.0, float(lay_odd) - 1.0))
                        if liab is None or float(liab) <= 0:
                            return (None, None)
                        stake_eq = float(liab) / max(1e-9, (float(lay_odd) - 1.0))
                        return (float(stake_eq), float(liab))

                    def _simulate_with_bankroll(
                        *,
                        bank_ref: float,
                        bud_back_frac_use: Optional[float] = None,
                        bud_lay_frac_use: Optional[float] = None,
                        cap_signal_frac_use: Optional[float] = None,
                        risk_mode: str = "fixed",
                    ) -> Dict[str, Any]:
                        """
                        Simula todo o OOS (nos steps já selecionados) variando somente a referência de banca.
                        Mantém budgets/caps (frações) constantes.
                        """
                        bbf = float(bud_back_frac if bud_back_frac_use is None else bud_back_frac_use)
                        blf = float(bud_lay_frac if bud_lay_frac_use is None else bud_lay_frac_use)
                        csf = float(bud_cap_sig_frac if cap_signal_frac_use is None else cap_signal_frac_use)
                        bud_back = float(bbf) * float(bank_ref)
                        bud_lay = float(blf) * float(bank_ref)

                        dturn: Dict[str, float] = {}
                        dpnl_exp: Dict[str, float] = {}
                        back_exps: List[float] = []
                        lay_exps: List[float] = []
                        jobs: List[Tuple[datetime, datetime, float]] = []

                        for st in steps:
                            test_days = set(st.get("test_days") or [])
                            active_keys = set(st.get("active_keys") or [])
                            if not test_days or not active_keys:
                                continue

                            test_elig = [e for e in combo_events if e["day"] in test_days and _key(e) in active_keys and _liq_ok_policy(e, st)]
                            if not test_elig:
                                continue

                            # ordena por tempo para consumir budget (por jogo) de forma realista
                            def _ts_ev(ev: dict) -> float:
                                d0 = audit_by_id.get(int(ev.get("audit_id")))
                                ts = d0.get("audited_at") if d0 else None
                                if isinstance(ts, datetime):
                                    return ts.timestamp()
                                return 0.0

                            test_elig.sort(key=_ts_ev)
                            cnt_back = Counter(int(ev.get("match_id")) for ev in test_elig if ev.get("side") == "Back")
                            cnt_lay = Counter(int(ev.get("match_id")) for ev in test_elig if ev.get("side") != "Back")

                            turn_all = 0.0
                            pnl_obs = 0.0
                            back_all = back_roi = 0.0
                            lay_all = lay_roi = 0.0
                            spent_back: Dict[int, float] = {}
                            spent_lay: Dict[int, float] = {}

                            for ev in test_elig:
                                st_eq, exp = _sizing_for_event_bankroll(
                                    ev,
                                    float(bank_ref),
                                    roi_train_map={k: _safe_float((st.get("diag") or {}).get(k, {}).get("roi_mean")) for k in active_keys},
                                )
                                if st_eq is None or exp is None:
                                    continue

                                mid = int(ev.get("match_id"))
                                if ev.get("side") == "Back":
                                    bud_m = float(bud_back)
                                    if risk_mode == "signals_sqrt":
                                        bud_m = float(bud_back) / max(1.0, math.sqrt(float(cnt_back.get(mid, 1))))
                                    elif risk_mode == "signals_linear":
                                        bud_m = float(bud_back) / max(1.0, float(cnt_back.get(mid, 1)))
                                    cap_sig_m = float(csf) * float(bud_m)
                                    rem = max(0.0, float(bud_m) - float(spent_back.get(mid, 0.0)))
                                    if rem <= 0:
                                        continue
                                    exp_use = min(float(exp), float(rem), float(cap_sig_m))
                                    if exp_use <= 0:
                                        continue
                                    ratio = exp_use / max(1e-9, float(exp))
                                    exp = exp_use
                                    st_eq = float(st_eq) * float(ratio)
                                    spent_back[mid] = float(spent_back.get(mid, 0.0)) + float(exp_use)
                                    back_all += float(exp)
                                    if ev.get("roi") is not None:
                                        back_roi += float(exp)
                                else:
                                    bud_m = float(bud_lay)
                                    if risk_mode == "signals_sqrt":
                                        bud_m = float(bud_lay) / max(1.0, math.sqrt(float(cnt_lay.get(mid, 1))))
                                    elif risk_mode == "signals_linear":
                                        bud_m = float(bud_lay) / max(1.0, float(cnt_lay.get(mid, 1)))
                                    cap_sig_m = float(csf) * float(bud_m)
                                    rem = max(0.0, float(bud_m) - float(spent_lay.get(mid, 0.0)))
                                    if rem <= 0:
                                        continue
                                    exp_use = min(float(exp), float(rem), float(cap_sig_m))
                                    if exp_use <= 0:
                                        continue
                                    ratio = exp_use / max(1e-9, float(exp))
                                    exp = exp_use
                                    st_eq = float(st_eq) * float(ratio)
                                    spent_lay[mid] = float(spent_lay.get(mid, 0.0)) + float(exp_use)
                                    lay_all += float(exp)
                                    if ev.get("roi") is not None:
                                        lay_roi += float(exp)

                                turn_all += float(st_eq)
                                _append_job(ev, float(exp))

                                if ev.get("roi") is None:
                                    continue
                                pnl_obs += float(exp) * float(ev.get("roi")) / 100.0

                            # expansão missing ROI (mesma lógica do WF principal)
                            pnl_exp = float(pnl_obs)
                            if wf_expand:
                                scale_back = (back_all / back_roi) if back_roi > 0 else 1.0
                                scale_lay = (lay_all / lay_roi) if lay_roi > 0 else 1.0
                                if back_roi > 0 and lay_roi > 0:
                                    w_back = back_roi / max(1e-9, (back_roi + lay_roi))
                                    w_lay = 1.0 - w_back
                                    pnl_exp = float(pnl_obs) * (w_back * scale_back + w_lay * scale_lay)
                                elif back_roi > 0:
                                    pnl_exp = float(pnl_obs) * float(scale_back)
                                elif lay_roi > 0:
                                    pnl_exp = float(pnl_obs) * float(scale_lay)

                            # acumula por dia (divide por qtd de dias do step, como no principal)
                            for dday in test_days:
                                dturn[dday] = dturn.get(dday, 0.0) + float(turn_all) / max(1, len(test_days))
                                dpnl_exp[dday] = dpnl_exp.get(dday, 0.0) + float(pnl_exp) / max(1, len(test_days))

                            if back_all > 0:
                                back_exps.append(float(back_all))
                            if lay_all > 0:
                                lay_exps.append(float(lay_all))

                        oos_days2 = sorted(dturn.keys())
                        n_days2 = len(oos_days2) if oos_days2 else 0
                        horizon = 30.0
                        scale = (horizon / float(n_days2)) if n_days2 > 0 else None
                        turn_sum = float(sum(dturn.values())) if dturn else 0.0
                        pnl_exp_sum = float(sum(dpnl_exp.values())) if dpnl_exp else 0.0
                        turn_30 = float(turn_sum) * float(scale) if scale is not None else None
                        prof_exp_30 = float(pnl_exp_sum) * float(scale) if scale is not None else None

                        bank_back_p99 = _pctl(back_exps, 99) if back_exps else None
                        bank_lay_p99 = _pctl(lay_exps, 99) if lay_exps else None
                        bank_risk = (float(bank_back_p99 or 0.0) + float(bank_lay_p99 or 0.0)) if (bank_back_p99 is not None or bank_lay_p99 is not None) else None
                        bank_liq = _liq_p99_from_jobs(jobs)
                        bank_eff2 = max(float(bank_risk or 0.0), float(bank_liq or 0.0)) if (bank_risk is not None or bank_liq is not None) else None

                        _, dd_p95 = _bootstrap_dd(list(dpnl_exp.values()), horizon_days=30, n_boot=2000)
                        roi_bank2 = (float(prof_exp_30) / float(bank_eff2) * 100.0) if (prof_exp_30 is not None and bank_eff2 and bank_eff2 > 0) else None
                        return {
                            "bank_ref": float(bank_ref),
                            "turn_30d": turn_30,
                            "profit_30d_exp": prof_exp_30,
                            "bank_eff": bank_eff2,
                            "roi_bank_30d": roi_bank2,
                            "dd_p95": dd_p95,
                            "days": n_days2,
                        }

                    if grid and len(grid) >= 2:
                        lines.append("### 12.2b Sensibilidade por banca (mantendo budgets/caps e seleção)\n")
                        lines.append(
                            "Aqui variamos a **banca de referência** usada tanto para o **sizing (Kelly/caps)** quanto para o **budget por jogo** "
                            f"(frações fixas: Back={bud_back_frac:.2%}, Lay={bud_lay_frac:.2%}, cap_sinal={bud_cap_sig_frac:.0%}).\n\n"
                        )
                        lines.append(
                            "**Definições (importante):**\n"
                            "- `Banca (ref)`: parâmetro de simulação que escala **budgets/caps** (e a fração de Kelly quando aplicável).\n"
                            "- `Banca rec. (max)`: capital **recomendado** para suportar a operação no cenário simulado, calculado como:\n"
                            "  `max( banca_risco_p99 ; banca_liq_p99 )`.\n"
                            "- `ROI/banca`: por conservadorismo, usamos `Lucro 30d / Banca rec. (max)` (não `Lucro/Banca(ref)`), pois é o retorno sobre o capital que\n"
                            "  de fato precisa estar alocado para rodar a estratégia sem estourar risco/liquidez.\n\n"
                        )
                        lines.append("| Banca (ref) | Turnover 30d | Lucro 30d (exp.) | Banca rec. (max) | ROI/banca 30d (exp.) | DD 30d p95 (exp.) |\n")
                        lines.append("|---:|---:|---:|---:|---:|---:|\n")
                        for b in grid:
                            r = _simulate_with_bankroll(bank_ref=float(b))
                            lines.append(
                                f"| {_fmt_num(r.get('bank_ref'),2)} | {_fmt_num(r.get('turn_30d'),2)} | {_fmt_num(r.get('profit_30d_exp'),2)} | "
                                f"{_fmt_num(r.get('bank_eff'),2)} | {_fmt_num(r.get('roi_bank_30d'),2)}% | {_fmt_num(r.get('dd_p95'),2)} |\n"
                            )
                        lines.append("\n")

                        # cenário solicitado: Risk(signals_sqrt) + EQ 4%/4% cap50%
                        lines.append("### 12.2c Sensibilidade por banca — RISK(signals_sqrt) + BUDGET_EQ_4.00%/4.00% cap50%\n")
                        lines.append(
                            "Aqui repetimos a sensibilidade por banca usando **risk_mode=signals_sqrt** e budgets **EQ** "
                            "(Back=4%, Lay=4%) com **cap por sinal=50%** do budget do jogo.\n\n"
                        )
                        lines.append("| Banca (ref) | Turnover 30d | Lucro 30d (exp.) | Banca rec. (max) | ROI/banca 30d (exp.) | DD 30d p95 (exp.) |\n")
                        lines.append("|---:|---:|---:|---:|---:|---:|\n")
                        for b in grid:
                            r = _simulate_with_bankroll(
                                bank_ref=float(b),
                                bud_back_frac_use=0.04,
                                bud_lay_frac_use=0.04,
                                cap_signal_frac_use=0.50,
                                risk_mode="signals_sqrt",
                            )
                            lines.append(
                                f"| {_fmt_num(r.get('bank_ref'),2)} | {_fmt_num(r.get('turn_30d'),2)} | {_fmt_num(r.get('profit_30d_exp'),2)} | "
                                f"{_fmt_num(r.get('bank_eff'),2)} | {_fmt_num(r.get('roi_bank_30d'),2)}% | {_fmt_num(r.get('dd_p95'),2)} |\n"
                            )
                        lines.append("\n")

                        # sensibilidade adicional: Risk(signals_sqrt) + EQ 2%/2% cap33%
                        lines.append("### 12.2d Sensibilidade por banca — RISK(signals_sqrt) + BUDGET_EQ_2.00%/2.00% cap33%\n")
                        lines.append(
                            "Aqui repetimos a sensibilidade por banca usando **risk_mode=signals_sqrt** e budgets **EQ** "
                            "(Back=2%, Lay=2%) com **cap por sinal=33%** do budget do jogo.\n\n"
                        )
                        lines.append("| Banca (ref) | Turnover 30d | Lucro 30d (exp.) | Banca rec. (max) | ROI/banca 30d (exp.) | DD 30d p95 (exp.) |\n")
                        lines.append("|---:|---:|---:|---:|---:|---:|\n")
                        for b in grid:
                            r = _simulate_with_bankroll(
                                bank_ref=float(b),
                                bud_back_frac_use=0.02,
                                bud_lay_frac_use=0.02,
                                cap_signal_frac_use=0.33,
                                risk_mode="signals_sqrt",
                            )
                            lines.append(
                                f"| {_fmt_num(r.get('bank_ref'),2)} | {_fmt_num(r.get('turn_30d'),2)} | {_fmt_num(r.get('profit_30d_exp'),2)} | "
                                f"{_fmt_num(r.get('bank_eff'),2)} | {_fmt_num(r.get('roi_bank_30d'),2)}% | {_fmt_num(r.get('dd_p95'),2)} |\n"
                            )
                        lines.append("\n")

                    # ------------------------------------------------------------
                    # 12.3 Liquidez por linha AH — sensibilidade OOS e política
                    # ------------------------------------------------------------
                    lines.append("### 12.3 Linha AH (0–1, 1–2, 2+) no OOS — sensibilidade e política\n")
                    lines.append(
                        "Interpretação operacional (proxy de liquidez do **mercado AH**): linhas extremas (ex.: **AH 2+**) tendem a ser menos líquidas e "
                        "podem sofrer mais com slippage/execução. Aqui testamos políticas de filtro por `|line|` no OOS.\n\n"
                        "Buckets usados: **0–1**, **1–2**, **2+** (por `abs(line)`).\n\n"
                    )

                    def _ah_bucket(abs_line: Optional[float]) -> Optional[str]:
                        if abs_line is None or not math.isfinite(float(abs_line)):
                            return None
                        x = float(abs_line)
                        if x <= 1.0:
                            return "AH 0-1"
                        if x <= 2.0:
                            return "AH 1-2"
                        return "AH 2+"

                    def _simulate_oos_filtered(*, filt_fn) -> Dict[str, Any]:
                        dturn: Dict[str, float] = {}
                        dpnl: Dict[str, float] = {}
                        for st in steps:
                            test_days = set(st.get("test_days") or [])
                            active_keys = set(st.get("active_keys") or [])
                            if not test_days or not active_keys:
                                continue
                            test_elig = [e for e in combo_events if e["day"] in test_days and _key(e) in active_keys and filt_fn(e, st)]
                            if not test_elig:
                                continue
                            # budget/sizing padrão (mesmo da 12.1)
                            bank_ref_budget = _safe_float(getattr(args, "kelly_bankroll", None))
                            if bank_ref_budget is None or float(bank_ref_budget) <= 0:
                                bank_ref_budget = bank_eff
                            bank_ref_budget = float(bank_ref_budget or 1.0)
                            bud_back = float(bud_back_frac) * float(bank_ref_budget)
                            bud_lay = float(bud_lay_frac) * float(bank_ref_budget)
                            spent_back: Dict[int, float] = {}
                            spent_lay: Dict[int, float] = {}
                            # ordem temporal
                            def _ts_ev(ev: dict) -> float:
                                d0 = audit_by_id.get(int(ev.get("audit_id")))
                                ts = d0.get("audited_at") if d0 else None
                                if isinstance(ts, datetime):
                                    return ts.timestamp()
                                return 0.0
                            test_elig.sort(key=_ts_ev)
                            cnt_back = Counter(int(ev.get("match_id")) for ev in test_elig if ev.get("side") == "Back")
                            cnt_lay = Counter(int(ev.get("match_id")) for ev in test_elig if ev.get("side") != "Back")
                            turn_all = 0.0
                            pnl_obs = 0.0
                            back_all = back_roi = 0.0
                            lay_all = lay_roi = 0.0
                            for ev in test_elig:
                                res_sz = _sizing_for_event(ev, roi_train_map={k: _safe_float((st.get("diag") or {}).get(k, {}).get("roi_mean")) for k in active_keys})
                                if isinstance(res_sz, tuple) and len(res_sz) == 2:
                                    st_eq, exp = res_sz
                                else:
                                    st_eq, exp, _why = res_sz
                                if st_eq is None or exp is None:
                                    continue
                                mid = int(ev.get("match_id"))
                                if ev.get("side") == "Back":
                                    bud_m = float(bud_back)
                                    if bud_risk_mode == "signals_sqrt":
                                        bud_m = float(bud_back) / max(1.0, math.sqrt(float(cnt_back.get(mid, 1))))
                                    elif bud_risk_mode == "signals_linear":
                                        bud_m = float(bud_back) / max(1.0, float(cnt_back.get(mid, 1)))
                                    cap_sig_m = float(bud_cap_sig_frac) * float(bud_m)
                                    rem = max(0.0, float(bud_m) - float(spent_back.get(mid, 0.0)))
                                    if rem <= 0:
                                        continue
                                    exp_use = min(float(exp), float(rem), float(cap_sig_m))
                                    if exp_use <= 0:
                                        continue
                                    ratio = exp_use / max(1e-9, float(exp))
                                    exp = exp_use
                                    st_eq = float(st_eq) * float(ratio)
                                    spent_back[mid] = float(spent_back.get(mid, 0.0)) + float(exp_use)
                                    back_all += float(exp)
                                    if ev.get("roi") is not None:
                                        back_roi += float(exp)
                                else:
                                    bud_m = float(bud_lay)
                                    if bud_risk_mode == "signals_sqrt":
                                        bud_m = float(bud_lay) / max(1.0, math.sqrt(float(cnt_lay.get(mid, 1))))
                                    elif bud_risk_mode == "signals_linear":
                                        bud_m = float(bud_lay) / max(1.0, float(cnt_lay.get(mid, 1)))
                                    cap_sig_m = float(bud_cap_sig_frac) * float(bud_m)
                                    rem = max(0.0, float(bud_m) - float(spent_lay.get(mid, 0.0)))
                                    if rem <= 0:
                                        continue
                                    exp_use = min(float(exp), float(rem), float(cap_sig_m))
                                    if exp_use <= 0:
                                        continue
                                    ratio = exp_use / max(1e-9, float(exp))
                                    exp = exp_use
                                    st_eq = float(st_eq) * float(ratio)
                                    spent_lay[mid] = float(spent_lay.get(mid, 0.0)) + float(exp_use)
                                    lay_all += float(exp)
                                    if ev.get("roi") is not None:
                                        lay_roi += float(exp)
                                turn_all += float(st_eq)
                                if ev.get("roi") is None:
                                    continue
                                pnl_obs += float(exp) * float(ev.get("roi")) / 100.0
                            pnl_exp = float(pnl_obs)
                            if wf_expand:
                                scale_back = (back_all / back_roi) if back_roi > 0 else 1.0
                                scale_lay = (lay_all / lay_roi) if lay_roi > 0 else 1.0
                                if back_roi > 0 and lay_roi > 0:
                                    w_back = back_roi / max(1e-9, (back_roi + lay_roi))
                                    w_lay = 1.0 - w_back
                                    pnl_exp = float(pnl_obs) * (w_back * scale_back + w_lay * scale_lay)
                                elif back_roi > 0:
                                    pnl_exp = float(pnl_obs) * float(scale_back)
                                elif lay_roi > 0:
                                    pnl_exp = float(pnl_obs) * float(scale_lay)
                            for dday in test_days:
                                dturn[dday] = dturn.get(dday, 0.0) + float(turn_all) / max(1, len(test_days))
                                dpnl[dday] = dpnl.get(dday, 0.0) + float(pnl_exp) / max(1, len(test_days))
                        oos_days2 = sorted(dturn.keys())
                        n_days2 = len(oos_days2) if oos_days2 else 0
                        scale = (30.0 / float(n_days2)) if n_days2 > 0 else None
                        turn_30 = float(sum(dturn.values())) * float(scale) if scale is not None else None
                        prof_30 = float(sum(dpnl.values())) * float(scale) if scale is not None else None
                        _, dd_p95 = _bootstrap_dd(list(dpnl.values()), horizon_days=30, n_boot=2000)
                        roi_turn = (float(prof_30) / float(turn_30) * 100.0) if (prof_30 is not None and turn_30 and float(turn_30) > 0) else None
                        return {"turn_30d": turn_30, "profit_30d_exp": prof_30, "roi_turn": roi_turn, "dd_p95": dd_p95}

                    def _filt_gate(max_abs: float, scope: str):
                        scope = str(scope or "all").strip().lower()
                        if scope not in ("pre", "all"):
                            scope = "all"
                        def _f(ev: dict, st: dict) -> bool:
                            if scope == "pre" and str(ev.get("regime")) != "Pre":
                                return True
                            a = _safe_float(ev.get("ah_abs"))
                            if a is None or not math.isfinite(float(a)):
                                return False
                            return float(a) <= float(max_abs)
                        return _f

                    def _filt_only_bucket(bucket: str):
                        def _f(ev: dict, st: dict) -> bool:
                            a = _safe_float(ev.get("ah_abs"))
                            return _ah_bucket(a) == bucket
                        return _f

                    lines.append("| Cenário | Scope | Turnover 30d | Lucro 30d (exp.) | ROI/turnover 30d | DD 30d p95 |\n")
                    lines.append("|---|---|---:|---:|---:|---:|\n")
                    r0 = _simulate_oos_filtered(filt_fn=lambda ev, st: True)
                    lines.append(f"| BASELINE (sem filtro) | — | {_fmt_num(r0.get('turn_30d'),2)} | {_fmt_num(r0.get('profit_30d_exp'),2)} | {_fmt_num(r0.get('roi_turn'),2)}% | {_fmt_num(r0.get('dd_p95'),2)} |\n")
                    for scope in ("pre", "all"):
                        r2 = _simulate_oos_filtered(filt_fn=_filt_gate(2.0, scope))
                        lines.append(f"| GATE abs<=2.0 | {scope} | {_fmt_num(r2.get('turn_30d'),2)} | {_fmt_num(r2.get('profit_30d_exp'),2)} | {_fmt_num(r2.get('roi_turn'),2)}% | {_fmt_num(r2.get('dd_p95'),2)} |\n")
                    for scope in ("pre", "all"):
                        r1 = _simulate_oos_filtered(filt_fn=_filt_gate(1.0, scope))
                        lines.append(f"| GATE abs<=1.0 | {scope} | {_fmt_num(r1.get('turn_30d'),2)} | {_fmt_num(r1.get('profit_30d_exp'),2)} | {_fmt_num(r1.get('roi_turn'),2)}% | {_fmt_num(r1.get('dd_p95'),2)} |\n")
                    lines.append("\n**Ablation (diagnóstico)**: operar apenas em um bucket de linha (budget reinicia por step; serve como diagnóstico, não como decomposição exata do baseline).\n\n")
                    lines.append("| Bucket | Turnover 30d | Lucro 30d (exp.) | ROI/turnover 30d |\n")
                    lines.append("|---|---:|---:|---:|\n")
                    for bkt in ("AH 0-1", "AH 1-2", "AH 2+"):
                        r = _simulate_oos_filtered(filt_fn=_filt_only_bucket(bkt))
                        lines.append(f"| {bkt} | {_fmt_num(r.get('turn_30d'),2)} | {_fmt_num(r.get('profit_30d_exp'),2)} | {_fmt_num(r.get('roi_turn'),2)}% |\n")
                    lines.append("\n**Política sugerida (OOS)**: se `AH 2+` degradar ROI/turnover, começar com `--wf-ah-max-abs-line 2 --wf-ah-scope all` (ou `pre`).\n\n")

                    # ------------------------------------------------------------
                    # 12.4 Liquidez por limit (betslip_limit) — sensibilidade OOS
                    # ------------------------------------------------------------
                    lines.append("### 12.4 Liquidez por limit (betslip_limit) no OOS — sensibilidade (opcional)\n")
                    lines.append(
                        "Este bloco é **opcional** e usa o proxy `betslip_limit`/`lay.available_limit` (capacidade por aposta) como outra visão de liquidez.\n\n"
                    )

                    def _simulate_liq_sensitivity(*, mode: str, scope: str) -> Dict[str, Any]:
                        mode = str(mode or "none").strip().lower()
                        scope = str(scope or "pre").strip().lower()
                        if mode not in ("none", "gate_p50", "gate_p75"):
                            mode = "none"
                        if scope not in ("pre", "all"):
                            scope = "pre"
                        dturn: Dict[str, float] = {}
                        dpnl: Dict[str, float] = {}
                        thrs: List[float] = []
                        for st in steps:
                            test_days = set(st.get("test_days") or [])
                            active_keys = set(st.get("active_keys") or [])
                            if not test_days or not active_keys:
                                continue
                            thr = None
                            if mode == "gate_p50":
                                thr = _safe_float(st.get("liq_thr_p50_pre" if scope == "pre" else "liq_thr_p50_all"))
                            elif mode == "gate_p75":
                                thr = _safe_float(st.get("liq_thr_p75_pre" if scope == "pre" else "liq_thr_p75_all"))
                            if thr is not None and float(thr) > 0:
                                thrs.append(float(thr))
                            # eventos no teste
                            test_elig = [e for e in combo_events if e["day"] in test_days and _key(e) in active_keys]
                            if mode != "none":
                                def _ok(ev: dict) -> bool:
                                    if scope == "pre" and str(ev.get("regime")) != "Pre":
                                        return True
                                    lim = _safe_float(ev.get("liq_limit"))
                                    if lim is None or float(lim) <= 0 or not math.isfinite(float(lim)):
                                        return False
                                    if thr is None or float(thr) <= 0 or not math.isfinite(float(thr)):
                                        return True
                                    return float(lim) >= float(thr)
                                test_elig = [e for e in test_elig if _ok(e)]
                            if not test_elig:
                                continue
                            # budget/sizing padrão (mesmo da 12.1)
                            bank_ref_budget = _safe_float(getattr(args, "kelly_bankroll", None))
                            if bank_ref_budget is None or float(bank_ref_budget) <= 0:
                                bank_ref_budget = bank_eff
                            bank_ref_budget = float(bank_ref_budget or 1.0)
                            bud_back = float(bud_back_frac) * float(bank_ref_budget)
                            bud_lay = float(bud_lay_frac) * float(bank_ref_budget)
                            spent_back: Dict[int, float] = {}
                            spent_lay: Dict[int, float] = {}
                            # ordem temporal
                            def _ts_ev(ev: dict) -> float:
                                d0 = audit_by_id.get(int(ev.get("audit_id")))
                                ts = d0.get("audited_at") if d0 else None
                                if isinstance(ts, datetime):
                                    return ts.timestamp()
                                return 0.0
                            test_elig.sort(key=_ts_ev)
                            cnt_back = Counter(int(ev.get("match_id")) for ev in test_elig if ev.get("side") == "Back")
                            cnt_lay = Counter(int(ev.get("match_id")) for ev in test_elig if ev.get("side") != "Back")
                            turn_all = 0.0
                            pnl_obs = 0.0
                            back_all = back_roi = 0.0
                            lay_all = lay_roi = 0.0
                            for ev in test_elig:
                                res_sz = _sizing_for_event(ev, roi_train_map={k: _safe_float((st.get("diag") or {}).get(k, {}).get("roi_mean")) for k in active_keys})
                                if isinstance(res_sz, tuple) and len(res_sz) == 2:
                                    st_eq, exp = res_sz
                                else:
                                    st_eq, exp, _why = res_sz
                                if st_eq is None or exp is None:
                                    continue
                                mid = int(ev.get("match_id"))
                                if ev.get("side") == "Back":
                                    bud_m = float(bud_back)
                                    if bud_risk_mode == "signals_sqrt":
                                        bud_m = float(bud_back) / max(1.0, math.sqrt(float(cnt_back.get(mid, 1))))
                                    elif bud_risk_mode == "signals_linear":
                                        bud_m = float(bud_back) / max(1.0, float(cnt_back.get(mid, 1)))
                                    cap_sig_m = float(bud_cap_sig_frac) * float(bud_m)
                                    rem = max(0.0, float(bud_m) - float(spent_back.get(mid, 0.0)))
                                    if rem <= 0:
                                        continue
                                    exp_use = min(float(exp), float(rem), float(cap_sig_m))
                                    if exp_use <= 0:
                                        continue
                                    ratio = exp_use / max(1e-9, float(exp))
                                    exp = exp_use
                                    st_eq = float(st_eq) * float(ratio)
                                    spent_back[mid] = float(spent_back.get(mid, 0.0)) + float(exp_use)
                                    back_all += float(exp)
                                    if ev.get("roi") is not None:
                                        back_roi += float(exp)
                                else:
                                    bud_m = float(bud_lay)
                                    if bud_risk_mode == "signals_sqrt":
                                        bud_m = float(bud_lay) / max(1.0, math.sqrt(float(cnt_lay.get(mid, 1))))
                                    elif bud_risk_mode == "signals_linear":
                                        bud_m = float(bud_lay) / max(1.0, float(cnt_lay.get(mid, 1)))
                                    cap_sig_m = float(bud_cap_sig_frac) * float(bud_m)
                                    rem = max(0.0, float(bud_m) - float(spent_lay.get(mid, 0.0)))
                                    if rem <= 0:
                                        continue
                                    exp_use = min(float(exp), float(rem), float(cap_sig_m))
                                    if exp_use <= 0:
                                        continue
                                    ratio = exp_use / max(1e-9, float(exp))
                                    exp = exp_use
                                    st_eq = float(st_eq) * float(ratio)
                                    spent_lay[mid] = float(spent_lay.get(mid, 0.0)) + float(exp_use)
                                    lay_all += float(exp)
                                    if ev.get("roi") is not None:
                                        lay_roi += float(exp)
                                turn_all += float(st_eq)
                                if ev.get("roi") is None:
                                    continue
                                pnl_obs += float(exp) * float(ev.get("roi")) / 100.0
                            pnl_exp = float(pnl_obs)
                            if wf_expand:
                                scale_back = (back_all / back_roi) if back_roi > 0 else 1.0
                                scale_lay = (lay_all / lay_roi) if lay_roi > 0 else 1.0
                                if back_roi > 0 and lay_roi > 0:
                                    w_back = back_roi / max(1e-9, (back_roi + lay_roi))
                                    w_lay = 1.0 - w_back
                                    pnl_exp = float(pnl_obs) * (w_back * scale_back + w_lay * scale_lay)
                                elif back_roi > 0:
                                    pnl_exp = float(pnl_obs) * float(scale_back)
                                elif lay_roi > 0:
                                    pnl_exp = float(pnl_obs) * float(scale_lay)
                            for dday in test_days:
                                dturn[dday] = dturn.get(dday, 0.0) + float(turn_all) / max(1, len(test_days))
                                dpnl[dday] = dpnl.get(dday, 0.0) + float(pnl_exp) / max(1, len(test_days))
                        oos_days2 = sorted(dturn.keys())
                        n_days2 = len(oos_days2) if oos_days2 else 0
                        scale = (30.0 / float(n_days2)) if n_days2 > 0 else None
                        turn_30 = float(sum(dturn.values())) * float(scale) if scale is not None else None
                        prof_30 = float(sum(dpnl.values())) * float(scale) if scale is not None else None
                        _, dd_p95 = _bootstrap_dd(list(dpnl.values()), horizon_days=30, n_boot=2000)
                        roi_turn = (float(prof_30) / float(turn_30) * 100.0) if (prof_30 is not None and turn_30 and float(turn_30) > 0) else None
                        thr_med = float(np.median(thrs)) if thrs else None
                        return {"turn_30d": turn_30, "profit_30d_exp": prof_30, "roi_turn": roi_turn, "dd_p95": dd_p95, "thr_med": thr_med}

                    lines.append("| Cenário | Scope | limiar (mediana, treino) | Turnover 30d | Lucro 30d (exp.) | ROI/turnover 30d | DD 30d p95 |\n")
                    lines.append("|---|---|---:|---:|---:|---:|---:|\n")
                    r0 = _simulate_liq_sensitivity(mode="none", scope="all")
                    lines.append(f"| BASELINE | — | — | {_fmt_num(r0.get('turn_30d'),2)} | {_fmt_num(r0.get('profit_30d_exp'),2)} | {_fmt_num(r0.get('roi_turn'),2)}% | {_fmt_num(r0.get('dd_p95'),2)} |\n")
                    for md, sc in [("gate_p50", "pre"), ("gate_p75", "pre"), ("gate_p50", "all")]:
                        r = _simulate_liq_sensitivity(mode=md, scope=sc)
                        lines.append(f"| LIQ_{md.upper()} | {sc} | {_fmt_num(r.get('thr_med'),2)} | {_fmt_num(r.get('turn_30d'),2)} | {_fmt_num(r.get('profit_30d_exp'),2)} | {_fmt_num(r.get('roi_turn'),2)}% | {_fmt_num(r.get('dd_p95'),2)} |\n")
                    lines.append("\nPolítica sugerida (opcional): `--wf-liquidity-mode gate_p50 --wf-liquidity-scope pre`.\n\n")

                    # Breakdown de volume por combinação (para explicar queda de turnover/lucro)
                    lines.append("**Volume e stake médio por combinação (janela OOS, com budget padrão)**\n\n")
                    lines.append(
                        "| Combinação | #steps ativa | Jogos OOS (uniques) | N eventos OOS | Stake eq. médio | Observação |\n"
                        "|---|---:|---:|---:|---:|---|\n"
                    )
                    # coleta universo OOS usado nos steps
                    oos_keys = set(active_counts.keys())
                    # estimativa rápida em cima dos próprios steps: soma por test_days
                    combo_stats: Dict[str, Dict[str, float]] = {}
                    for st in steps:
                        tdays = set(st.get("test_days") or [])
                        akeys = set(st.get("active_keys") or [])
                        for k in akeys:
                            combo_stats.setdefault(k, {"events": 0.0, "turn": 0.0, "matches": 0.0})
                        # percorre eventos elegíveis do teste
                        test_elig = [e for e in combo_events if e["day"] in tdays and _key(e) in akeys]
                        # sem aplicar budget de novo; usamos sizing base para estimar stake médio (ordem correta é budget, mas aqui é diagnóstico)
                        seen_matches: Dict[str, set] = {}
                        for ev in test_elig:
                            k = _key(ev)
                            res_sz = _sizing_for_event(ev)
                            if isinstance(res_sz, tuple) and len(res_sz) == 2:
                                st_eq, exp = res_sz
                            else:
                                st_eq, exp, _why = res_sz
                            if st_eq is None or exp is None:
                                continue
                            combo_stats.setdefault(k, {"events": 0.0, "turn": 0.0, "matches": 0.0})
                            combo_stats[k]["events"] += 1.0
                            combo_stats[k]["turn"] += float(st_eq)
                            seen_matches.setdefault(k, set()).add(int(ev.get("match_id")))
                        for k, ms in seen_matches.items():
                            combo_stats.setdefault(k, {"events": 0.0, "turn": 0.0, "matches": 0.0})
                            combo_stats[k]["matches"] += float(len(ms))

                    for k, c in sorted(active_counts.items(), key=lambda x: x[1], reverse=True):
                        st = combo_stats.get(k, {})
                        events = int(st.get("events") or 0)
                        matches = int(st.get("matches") or 0)
                        turn = float(st.get("turn") or 0.0)
                        stake_avg = (turn / events) if events > 0 else None
                        note = "budget reduz concentração por jogo" if True else ""
                        lines.append(f"| {k} | {c} | {matches} | {events} | {_fmt_num(stake_avg,2)} | {note} |\n")

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
        lines.append("## 11) Conclusões (visão de investidor), riscos e próximos passos\n")
        lines.append(
            "Esta seção é escrita como se um investidor externo estivesse avaliando a tese: **há edge replicável? o sistema executa? "
            "o risco é governável? a mensuração é confiável?**\n\n"
        )

        lines.append("### 11.1 O que já está forte (e por quê)\n")
        lines.append(
            "- **Evidência de execução (CLV pre‑match)**: CLV robusto por jogo positivo é um dos melhores sinais de edge/execução em janela curta. "
            "Diferente de ROI, CLV não depende de amostra grande de jogos liquidados; ele mede **qualidade de entrada**.\n"
        )
        lines.append(
            "- **Controle de latência por regime**: o relatório já separa regimes de execução por tempo total (2.3/2.3b). "
            "Isso permite uma regra objetiva de operação (ex.: só operar `exec_bucket < 5s`).\n"
        )
        lines.append(
            "- **Separação Back vs Lay**: Back e Lay têm perfis de risco diferentes. Lay deve ser governado por **liability** (p95/p99/ES), "
            "e isso já aparece como métrica de banca e risco.\n\n"
        )

        lines.append("### 11.2 O que ainda está frágil (e impede captação hoje)\n")
        lines.append(
            "- **ROI ainda não é prova**: mesmo quando ROI aparece, a incerteza por jogo pode ser grande e a cobertura de placar pode ser incompleta. "
            "Para captação, um investidor vai pedir **histórico maior**, **pipeline de resultados estável** e **métrica de drawdown** bem definida.\n"
        )
        lines.append(
            "- **Risco de viés por falhas de coleta**: quando o collector fica “active” mas não coleta odds, você perde janelas do mercado de forma não aleatória. "
            "Isso impacta a extrapolação para execução.\n"
        )
        lines.append(
            "- **Stake sizing ainda é proxy**: parte do sizing usa limit/finance como aproximação. Para captação, é necessário um sizing governado por risco "
            "e consistente com edge (ex.: Kelly fracionado + caps), com auditoria clara.\n\n"
        )

        lines.append("### 11.3 Avaliação das 2 estratégias candidatas (como um investidor leria)\n")
        lines.append(
            "Você propôs duas teses operacionais coerentes com o mecanismo observado:\n"
            "1) **BackFast**: operar Back edge apenas quando a execução foi rápida (`< 5s`) e pre‑match.\n"
            "2) **LayReversal**: operar Lay edge apenas quando há reversão e entrar próximo do vale (t_ext curto).\n\n"
        )
        lines.append(
            "O relatório quantifica isso na **Seção 9.4** com (i) N na janela, (ii) projeção 30d, "
            "(iii) stake/liability médio, (iv) banca p99 e ROI/banca mensal, e (v) drawdown p95.\n\n"
        )
        lines.append(
            "**Como um investidor decide**: ele vai priorizar uma estratégia com\n"
            "- sinal de edge (CLV) consistente,\n"
            "- execução estável (latência controlada),\n"
            "- sizing governado por risco (caps + banca p99/ES),\n"
            "- e um perfil de drawdown aceitável no horizonte de caixa.\n\n"
        )

        lines.append("### 11.4 Stake sizing: recomendação inicial para produção (sem overfitting)\n")
        lines.append(
            "- Use **baseline FLAT** como controle (para detectar se o sizing está degradando performance).\n"
            "- Para Back, use **Kelly fracionado** (ex.: `KELLY_0.25`) apenas quando houver `closing_odd` (pre‑match), com **cap** por aposta (ex.: 2% da banca p99).\n"
            "- Para Lay, faça sizing por **liability**, com cap mais conservador (ex.: 1% da banca p99) e monitoramento de cauda (p95/p99/ES95).\n\n"
        )
        lines.append(
            "A Seção 9.3 compara `FLAT` vs `PROXY` vs `KELLY` (fracionado) no subconjunto com placar, "
            "e reporta risco (p99/ES) e drawdown 30d via bootstrap.\n\n"
        )

        lines.append("### 11.5 Status para captação (checkpoint objetivo)\n")
        lines.append(
            "Se você estivesse captando hoje, um investidor institucional provavelmente pediria:\n"
            "- **(A)** 30–90 dias de execução estável com SLO de coleta (collector), auditoria e resultados.\n"
            "- **(B)** KPIs: CLV pre‑match por jogo estável; latência por bucket; taxa de falhas; cobertura de placar.\n"
            "- **(C)** Política de risco: banca por p99/ES, caps por aposta, limites por janela e mecanismos de stop.\n"
            "- **(D)** Demonstração de P&L com sizing definido (não só proxy) e drawdown observado/estimado.\n\n"
        )
        lines.append(
            "Minha leitura: **a tese de edge/execução parece promissora pelo CLV**, mas o projeto ainda está em fase de "
            "**consolidação operacional/medição** para uma captação “grande”. Um caminho pragmático é:\n"
            "- validar BackFast com sizing conservador e risco baixo,\n"
            "- validar LayReversal com governança de liability,\n"
            "- e só então ampliar banca.\n"
        )

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

        # ============================================================
        # Pós-processamento do documento (ex.: OOS-first)
        # ============================================================
        report_mode = str(getattr(args, "report_mode", "full") or "full").strip().lower()
        if report_mode == "oos_first":
            def _reorder_oos_first(doc_lines: List[str]) -> List[str]:
                def _find_in(lines_list: List[str], prefix: str, start: int = 0) -> Optional[int]:
                    for ii in range(start, len(lines_list)):
                        if str(lines_list[ii] or "").startswith(prefix):
                            return ii
                    return None

                i_oos = _find_in(doc_lines, "## 12) OOS walk-forward", 0)
                if i_oos is None:
                    return doc_lines
                j_oos = _find_in(doc_lines, "## 10) Diagnóstico", i_oos + 1)
                if j_oos is None:
                    j_oos = len(doc_lines)
                oos_block = doc_lines[i_oos:j_oos]
                rest = doc_lines[:i_oos] + doc_lines[j_oos:]

                # transforma headings para leitura OOS-first
                oos2: List[str] = []
                for ln in oos_block:
                    if "Até aqui o relatório é **in-sample**" in ln:
                        ln = (
                            "Este relatório é **OOS-first**: começamos pelo walk-forward (OOS) e deixamos as análises in-sample/diagnósticos no apêndice.\n\n"
                        )
                    if ln.startswith("## 12) OOS walk-forward"):
                        ln = ln.replace("## 12)", "## 1)", 1)
                    if ln.startswith("### 12."):
                        ln = ln.replace("### 12.", "### 1.", 1)
                    if ln.startswith("### 12.A"):
                        ln = ln.replace("### 12.A", "### 1.A", 1)
                    oos2.append(ln)

                ins = _find_in(rest, "## 1) ", 0)
                if ins is None:
                    ins = 0

                appendix_hdr = [
                    "\n---\n\n",
                    "## Apêndice — Diagnósticos e in-sample\n\n",
                    "_Nota: as seções abaixo mantêm a numeração original do relatório completo._\n\n",
                ]
                return rest[:ins] + oos2 + appendix_hdr + rest[ins:]

            lines = _reorder_oos_first(lines)

        # ============================================================
        # Modo rápido de inspeção: somente OOS (sem PDF)
        # ============================================================
        if bool(getattr(args, "only_oos", False)):
            try:
                def _find_oos_block(doc_lines: List[str]) -> List[str]:
                    i0 = None
                    for ii, ln in enumerate(doc_lines):
                        if "OOS walk-forward" in str(ln):
                            # pega o heading mais alto encontrado
                            if str(ln).startswith("## "):
                                i0 = ii
                                break
                    if i0 is None:
                        return doc_lines
                    j0 = len(doc_lines)
                    for jj in range(i0 + 1, len(doc_lines)):
                        ln = str(doc_lines[jj] or "")
                        if ln.startswith("## ") and ("OOS walk-forward" not in ln):
                            j0 = jj
                            break
                    hdr = [
                        "## OOS (extraído) — modo `--only-oos`\n\n",
                        "_Este arquivo contém **apenas** o bloco de walk-forward, para inspeção rápida (turnover, N elig/sized, e falhas de sizing)._ \n\n",
                    ]
                    return hdr + doc_lines[i0:j0]

                lines = _find_oos_block(lines)
                # não faz sentido PDF quando só queremos o bloco OOS
                args.pdf = None
            except Exception:
                pass

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
                repo_root_str = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
                repo_root = Path(repo_root_str).resolve()
                msg = (
                    args.git_message
                    or f"Adiciona relatório b808 ({args.direction}, {','.join(versions)}, lookback={args.lookback_days})"
                )

                def _as_git_path(p: Path) -> str:
                    """
                    Converte um Path (possivelmente relativo ao CWD atual) em path válido para `git add`
                    rodando em `repo_root`. Isso evita erro de pathspec quando o script é executado a partir
                    de um subdiretório e `--out/--pdf` foram passados como paths relativos.
                    """
                    abs_p = p.expanduser().resolve()
                    try:
                        return str(abs_p.relative_to(repo_root))
                    except Exception:
                        # fallback: path absoluto (git aceita se estiver dentro do worktree)
                        return str(abs_p)

                paths: List[str] = [_as_git_path(out_path)]
                if pdf_path and pdf_path.exists():
                    paths.append(_as_git_path(pdf_path))

                subprocess.run(["git", "add", "--"] + paths, check=True, cwd=str(repo_root))
                subprocess.run(["git", "commit", "-m", msg], check=True, cwd=str(repo_root))
                print(f"[INFO] Artefatos commitados no git: {', '.join(paths)}")
                if args.git_push:
                    subprocess.run(["git", "push"], check=True, cwd=str(repo_root))
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

