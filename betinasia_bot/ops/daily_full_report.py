from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import statistics
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import requests
from loguru import logger

from .accounting_daily_report import DailyCfg as AcctDailyCfg, run_daily as run_acct_daily
from .accounting_report import compute_pnl_report
from .execution_kpis import compute_kpis_from_lines


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _load_env_file(path: Path) -> None:
    """
    Carrega variáveis de um arquivo .env simples (KEY=VALUE), sem sobrescrever env já definido.
    Ajuda quando rodando manualmente fora do systemd (que usa EnvironmentFile=...).
    """
    try:
        if not path.exists():
            return
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            if not k or k in os.environ:
                continue
            os.environ[k] = v.strip()
    except Exception:
        return


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _pick_col(cols: list[str], needles: tuple[str, ...] | list[str]) -> Optional[str]:
    """
    Seleciona uma coluna por heurística, preferindo:
    1) match exato (case-insensitive)
    2) prefix match
    3) contains (fallback)
    """
    try:
        cols = list(cols or [])
        cols_map = {str(c).lower(): str(c) for c in cols if str(c)}
        cols_l = list(cols_map.keys())
        for n0 in (needles or []):
            n = str(n0).lower()
            if not n:
                continue
            if n in cols_map:
                return cols_map[n]
            for cl in cols_l:
                if cl.startswith(n):
                    return cols_map[cl]
            for cl in cols_l:
                if n in cl:
                    return cols_map[cl]
    except Exception:
        return None
    return None


def _to_utc_dt(x: Any) -> Optional[datetime]:
    try:
        if x is None:
            return None
        if isinstance(x, datetime):
            dt = x
        else:
            dt = _parse_dt_any(str(x))
        if not isinstance(dt, datetime):
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _is_inplay_from_audit_row(a: Optional[Dict[str, Any]], *, exec_created_at_utc: datetime) -> bool:
    """
    Determina Pre/In para uma execução usando:
    1) kickoff_time (preferencial): exec_created_at >= kickoff => In
    2) audit.is_live quando não-NULL
    3) fallback: False (Pre)
    """
    if not isinstance(a, dict) or not a:
        return False
    try:
        ko = _to_utc_dt(a.get("kickoff_time"))
        if isinstance(ko, datetime):
            return bool(exec_created_at_utc >= ko)
    except Exception:
        pass
    try:
        if a.get("is_live") is not None:
            return bool(a.get("is_live"))
    except Exception:
        pass
    return False


def _fmt_pct(x: Any, nd: int = 2) -> str:
    try:
        if x is None:
            return "—"
        return f"{float(x):.{nd}f}%"
    except Exception:
        return "—"


def _fmt_num(x: Any, nd: int = 2) -> str:
    try:
        if x is None:
            return "—"
        # separador de milhar para leitura operacional (ex.: 1,000,000.00)
        return f"{float(x):,.{nd}f}"
    except Exception:
        return "—"


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        s = s.replace(",", ".")
        # mantém apenas dígitos/sinal/ponto
        out = []
        for ch in s:
            if ch.isdigit() or ch in ".-":
                out.append(ch)
        s2 = "".join(out).strip()
        if s2 in ("", "-", ".", "-."):
            return None
        return float(s2)
    except Exception:
        return None


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _is_truthy(x: Any) -> bool:
    try:
        return str(x or "").strip().lower() in ("1", "true", "yes", "y", "on")
    except Exception:
        return False


def _policy_compatibility_check(
    active_keys: Any,
    *,
    bridge_exec_side: str,
    bridge_prematch_only: bool,
    min_pre_keys: int,
) -> Dict[str, Any]:
    keys = [str(k) for k in (active_keys or []) if str(k).strip()]
    side = str(bridge_exec_side or "back").strip().lower()
    if side not in ("back", "lay", "both"):
        side = "back"
    prematch = bool(bridge_prematch_only)
    min_need = max(1, int(min_pre_keys))

    def _count(prefix: str) -> int:
        return sum(1 for k in keys if str(k).startswith(prefix))

    cnt_back_pre = _count("Back_Pre_")
    cnt_back_in = _count("Back_In_")
    cnt_lay_pre = _count("Lay_Pre_")
    cnt_lay_in = _count("Lay_In_")

    checks: List[Tuple[bool, str]] = []
    if prematch:
        if side in ("back", "both"):
            checks.append((cnt_back_pre >= min_need, f"Back_Pre_>={min_need} (atual={cnt_back_pre})"))
        if side in ("lay", "both"):
            checks.append((cnt_lay_pre >= min_need, f"Lay_Pre_>={min_need} (atual={cnt_lay_pre})"))
    else:
        if side in ("back", "both"):
            checks.append(((cnt_back_pre + cnt_back_in) >= 1, f"Back_(Pre/In)>=1 (atual={cnt_back_pre + cnt_back_in})"))
        if side in ("lay", "both"):
            checks.append(((cnt_lay_pre + cnt_lay_in) >= 1, f"Lay_(Pre/In)>=1 (atual={cnt_lay_pre + cnt_lay_in})"))

    ok = bool(checks) and all(c[0] for c in checks)
    failed = [msg for cond, msg in checks if not cond]
    reason = "ok" if ok else ("; ".join(failed) if failed else "no_checks")
    return {
        "ok": bool(ok),
        "reason": str(reason),
        "bridge_exec_side": str(side),
        "bridge_prematch_only": bool(prematch),
        "min_pre_keys": int(min_need),
        "n_active_keys": int(len(keys)),
        "n_back_pre": int(cnt_back_pre),
        "n_back_in": int(cnt_back_in),
        "n_lay_pre": int(cnt_lay_pre),
        "n_lay_in": int(cnt_lay_in),
        "checks": [{"ok": bool(cond), "rule": str(msg)} for cond, msg in checks],
    }


def _parse_dt_any(s: Any) -> Optional[datetime]:
    """
    Best-effort parse para campos como 'post date' do CSV de accounting.
    Retorna datetime timezone-aware em UTC quando possível.
    """
    try:
        t = str(s or "").strip()
        if not t:
            return None
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        # ISO (com ou sem offset)
        try:
            dt = datetime.fromisoformat(t)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass
        # formatos comuns
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(t, fmt).replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                continue
    except Exception:
        return None
    return None


async def _fetch_audit_rows_for_ids_daily(db, ids: list[int]) -> Dict[int, Dict[str, Any]]:
    """
    Busca metadados do audit necessários para classificar Pre/In corretamente no report diário.
    Usa `kickoff_time` (matches) e `betslip_audit_results.is_live` (quando presente).
    """
    if not ids:
        return {}
    try:
        from sqlalchemy import text  # type: ignore
    except Exception:
        return {}
    q = text(
        """
        SELECT
          a.id AS audit_id,
          a.event_id,
          a.is_live,
          a.audited_at,
          a.hypothesis_detected_at,
          a.lag_detection_to_click_ms,
          a.lag_click_to_betslip_ms,
          a.audit_total_duration_ms,
          a.hypothesis_details,
          m.kickoff_time
        FROM betslip_audit_results a
        LEFT JOIN matches m ON m.external_id = a.event_id
        WHERE a.id = ANY(:ids)
        """
    )
    out: Dict[int, Dict[str, Any]] = {}
    try:
        async with db.async_session() as session:
            r = await session.execute(q, {"ids": list(ids)})
            for x in r.fetchall() or []:
                out[int(x._mapping["audit_id"])] = dict(x._mapping)
    except Exception:
        return {}
    return out


def _bucket_min_to_kickoff(exec_created_at_utc: datetime, kickoff_time: Optional[datetime]) -> Optional[int]:
    """
    Retorna minutos desde kickoff (>=0) quando exec >= kickoff; ou minutos até kickoff (<0) se exec < kickoff.
    """
    try:
        if not isinstance(exec_created_at_utc, datetime) or not isinstance(kickoff_time, datetime):
            return None
        dt = exec_created_at_utc.astimezone(timezone.utc)
        ko = kickoff_time.astimezone(timezone.utc)
        return int(round((dt - ko).total_seconds() / 60.0))
    except Exception:
        return None


def _bucket_label_min_since_kickoff(mins: Optional[int]) -> str:
    """
    mins: minutos desde kickoff (pode ser <0 se pre-match).
    Buckets focados em in-play para análise de timing (0-5, 5-15, 15-30, 30-60, >60).
    """
    if mins is None:
        return "Desconhecido"
    try:
        m = int(mins)
    except Exception:
        return "Desconhecido"
    if m < 0:
        return "Pre (<0)"
    if m <= 5:
        return "0-5m"
    if m <= 15:
        return "5-15m"
    if m <= 30:
        return "15-30m"
    if m <= 60:
        return "30-60m"
    return ">60m"


def _bucket_label_call_to_done_ms(lat_ms: Any) -> str:
    """
    Buckets de latência de efetivação (tempo total) usando `call_to_done_ms` do executor.
    """
    x = _safe_float(lat_ms)
    if x is None:
        return "Desconhecido"
    try:
        v = float(x)
    except Exception:
        return "Desconhecido"
    if v < 5000:
        return "< 5s"
    if v < 10000:
        return "5-10s"
    if v < 20000:
        return "10-20s"
    if v < 40000:
        return "20-40s"
    return "> 40s"


def _summarize_rows_pnl_exp(rows: list[dict]) -> dict:
    """
    rows: [{pnl, exposure, event_id?}] — agrega P&L, exposição, ROIw e contagem de jogos únicos (event_id).
    """
    try:
        n = int(len(rows or []))
    except Exception:
        n = 0
    pnl = 0.0
    exp = 0.0
    evs: set[str] = set()
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        try:
            if r.get("pnl") is not None:
                pnl += float(r.get("pnl") or 0.0)
        except Exception:
            pass
        try:
            if r.get("exposure") is not None:
                exp += float(r.get("exposure") or 0.0)
        except Exception:
            pass
        try:
            eid = str(r.get("event_id") or "").strip()
            if eid:
                evs.add(eid)
        except Exception:
            pass
    roiw = (pnl / exp * 100.0) if exp > 0 else None
    return {"n_orders": n, "n_events": int(len(evs)), "pnl_sum": float(pnl), "exposure_sum": float(exp), "roi_weighted": roiw}


def _pct(num: Any, den: Any) -> Optional[float]:
    try:
        n = float(num)
        d = float(den)
        if d <= 0:
            return None
        return float(n / d * 100.0)
    except Exception:
        return None


def _acct_pnl_per_event_from_balance_csv(
    path: Path,
    *,
    days_utc: Optional[set[str]] = None,
    only_type_bet: bool = True,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Agrupa `amount` do balance.csv por dia (UTC) e por jogo (event info event id).
    Retorna:
      day -> event_id -> {pnl_sum, stake_est_sum, n_rows, event_name}

    stake_est_sum é uma proxy de turnover: soma de (-amount) quando amount<0.
    """
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if not isinstance(row, dict):
                continue
            if only_type_bet:
                typ = str(row.get("type") or "").strip().lower()
                if typ and typ != "bet":
                    continue
            dt = _parse_dt_any(row.get("post date") or row.get("post_date") or row.get("date") or "")
            if dt is None:
                continue
            day = dt.date().isoformat()
            if days_utc is not None and day not in days_utc:
                continue
            amt = _safe_float(row.get("amount"))
            if amt is None:
                continue
            ev_id = str(row.get("event info event id") or "").strip()
            if not ev_id:
                ev_id = "__NO_EVENT_ID__"
            ev_name = str(row.get("event info event name") or "").strip()
            blk = out.setdefault(day, {}).setdefault(ev_id, {"pnl_sum": 0.0, "stake_est_sum": 0.0, "n_rows": 0, "event_name": ev_name})
            blk["pnl_sum"] = float(blk.get("pnl_sum") or 0.0) + float(amt)
            blk["n_rows"] = int(blk.get("n_rows") or 0) + 1
            if ev_name and (not str(blk.get("event_name") or "").strip()):
                blk["event_name"] = ev_name
            try:
                if float(amt) < 0:
                    blk["stake_est_sum"] = float(blk.get("stake_est_sum") or 0.0) + float(-amt)
            except Exception:
                pass
    return out


def _summarize_event_pnls(ev_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    ev_map: event_id -> {pnl_sum, stake_est_sum, ...}
    """
    pnls = []
    stakes = []
    for _, v in (ev_map or {}).items():
        try:
            pnls.append(float(v.get("pnl_sum") or 0.0))
        except Exception:
            pass
        try:
            stakes.append(float(v.get("stake_est_sum") or 0.0))
        except Exception:
            pass
    pnls_sorted = sorted(pnls)
    stakes_sorted = sorted(stakes)
    out: Dict[str, Any] = {
        "events_n": int(len(ev_map or {})),
        "pnl_sum": float(sum(pnls)) if pnls else 0.0,
        "stake_est_sum": float(sum(stakes)) if stakes else 0.0,
        "pnl_median": (float(statistics.median(pnls_sorted)) if pnls_sorted else None),
    }
    # Stake médio/jogo (proxy): soma de (-amount) por jogo / #jogos
    try:
        n_ev = int(out.get("events_n") or 0)
        st_sum = float(out.get("stake_est_sum") or 0.0)
        out["stake_mean_per_game"] = (st_sum / float(n_ev)) if n_ev > 0 else None
    except Exception:
        out["stake_mean_per_game"] = None
    # ROI mediana (proxy): (P&L mediana/jogo) / (stake médio/jogo)
    try:
        pm = out.get("pnl_median")
        sm = out.get("stake_mean_per_game")
        out["roi_median_pct"] = (float(pm) / float(sm) * 100.0) if (pm is not None and sm is not None and float(sm) > 0) else None
    except Exception:
        out["roi_median_pct"] = None
    try:
        if len(pnls_sorted) >= 10:
            out["pnl_p10"] = float(statistics.quantiles(pnls_sorted, n=10, method="inclusive")[0])
            out["pnl_p90"] = float(statistics.quantiles(pnls_sorted, n=10, method="inclusive")[8])
        else:
            out["pnl_p10"] = None
            out["pnl_p90"] = None
    except Exception:
        out["pnl_p10"] = None
        out["pnl_p90"] = None
    try:
        abs_sum = float(sum(abs(x) for x in pnls)) if pnls else 0.0
        out["pnl_conc_max_abs_share"] = (float(max(abs(x) for x in pnls)) / abs_sum) if abs_sum > 0 else None
    except Exception:
        out["pnl_conc_max_abs_share"] = None
    try:
        abs_st_sum = float(sum(abs(x) for x in stakes)) if stakes else 0.0
        out["stake_conc_max_share"] = (float(max(abs(x) for x in stakes)) / abs_st_sum) if abs_st_sum > 0 else None
    except Exception:
        out["stake_conc_max_share"] = None
    return out


def _event_tail_risk_metrics(ev_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Métricas simples de risco de cauda por jogo (event_id), usando P&L agregado por jogo.
    Retorna:
      - p5_pnl_per_game: percentil 5% do P&L por jogo
      - cvar5_pnl_per_game: média dos piores 5% jogos (ES/CVaR proxy)
      - worst_game_pnl: pior jogo
    """
    out: Dict[str, Any] = {
        "games_n": 0,
        "p5_pnl_per_game": None,
        "cvar5_pnl_per_game": None,
        "worst_game_pnl": None,
    }
    try:
        pnls = [float(v.get("pnl_sum") or 0.0) for v in (ev_map or {}).values() if isinstance(v, dict)]
    except Exception:
        pnls = []
    if not pnls:
        return out
    xs = sorted(pnls)
    n = len(xs)
    out["games_n"] = int(n)
    try:
        q_idx = int(max(0, min(n - 1, math.floor((n - 1) * 0.05))))
        qv = float(xs[q_idx])
        out["p5_pnl_per_game"] = qv
        tail = [x for x in xs if x <= qv]
        if not tail:
            tail = [xs[0]]
        out["cvar5_pnl_per_game"] = float(sum(tail) / float(len(tail)))
        out["worst_game_pnl"] = float(xs[0])
    except Exception:
        pass
    return out


def _top_event_exposures(ev_map: Dict[str, Dict[str, Any]], *, top_n: int = 5) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    rows: List[Tuple[str, float, float, str]] = []
    for ev_id, rec in (ev_map or {}).items():
        if not isinstance(rec, dict):
            continue
        try:
            st = float(rec.get("stake_est_sum") or 0.0)
            pnl = float(rec.get("pnl_sum") or 0.0)
        except Exception:
            continue
        ev_name = str(rec.get("event_name") or "").strip()
        rows.append((str(ev_id), st, pnl, ev_name))
    rows.sort(key=lambda x: abs(float(x[1])), reverse=True)
    total_abs = float(sum(abs(float(x[1])) for x in rows)) if rows else 0.0
    for ev_id, st, pnl, ev_name in rows[: max(1, int(top_n))]:
        share = (abs(float(st)) / total_abs * 100.0) if total_abs > 0 else None
        out.append(
            {
                "event_id": ev_id,
                "event_name": ev_name,
                "stake_est_sum": float(st),
                "pnl_sum": float(pnl),
                "share_pct": share,
            }
        )
    return out


def _order_tail_risk_by_bucket(rows: List[Dict[str, Any]], *, top_n: int = 6) -> Dict[str, Any]:
    """
    Risco de cauda por bucket operacional (lado×regime), usando ordens com accounting.
    rows: itens com {side, regime, pnl, exposure}.
    """
    out: Dict[str, Any] = {"by_bucket": [], "top_event_exposure": []}
    if not rows:
        return out

    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if not isinstance(r, dict):
            continue
        side = str(r.get("side") or "NA")
        regime = str(r.get("regime") or "NA")
        key = f"{side}_{regime}"
        by[key].append(r)

    # métrica por bucket (ordens)
    metrics: List[Dict[str, Any]] = []
    for key, sub in by.items():
        pnls: List[float] = []
        exps: List[float] = []
        for r in sub:
            try:
                pnl = float(r.get("pnl") or 0.0)
                exp = float(r.get("exposure") or 0.0)
            except Exception:
                continue
            pnls.append(pnl)
            exps.append(exp)
        if not pnls:
            continue
        xs = sorted(pnls)
        n = len(xs)
        q_idx = int(max(0, min(n - 1, math.floor((n - 1) * 0.05))))
        qv = float(xs[q_idx])
        tail = [x for x in xs if x <= qv] or [xs[0]]
        cvar5 = float(sum(tail) / float(len(tail)))
        worst = float(xs[0])
        exp_sum = float(sum(exps)) if exps else 0.0
        metrics.append(
            {
                "bucket": key,
                "n_orders": int(n),
                "exp_sum": exp_sum,
                "p5": qv,
                "cvar5": cvar5,
                "worst": worst,
            }
        )
    metrics.sort(key=lambda d: float(d.get("cvar5") or 0.0))
    out["by_bucket"] = metrics

    # top jogos por exposição, por bucket (proxy de concentração em score-state)
    top_rows: List[Dict[str, Any]] = []
    for key, sub in by.items():
        ev_agg: Dict[str, Dict[str, Any]] = {}
        for r in sub:
            ev_id = str(r.get("event_id") or "").strip()
            if not ev_id:
                continue
            try:
                st = float(r.get("exposure") or 0.0)
                pnl = float(r.get("pnl") or 0.0)
            except Exception:
                continue
            blk = ev_agg.setdefault(ev_id, {"event_id": ev_id, "stake_est_sum": 0.0, "pnl_sum": 0.0, "event_name": str(r.get("event_name") or "")})
            blk["stake_est_sum"] = float(blk.get("stake_est_sum") or 0.0) + float(st)
            blk["pnl_sum"] = float(blk.get("pnl_sum") or 0.0) + float(pnl)
        tops = _top_event_exposures(ev_agg, top_n=max(1, int(top_n)))
        for t in tops:
            top_rows.append({"bucket": key, **t})
    out["top_event_exposure"] = top_rows
    return out


def _safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    """
    Conversão resiliente para int.

    - Sem default explícito: mantém retorno Optional[int] (None em falha).
    - Com default explícito: retorna o fallback informado em falha/None.
    """
    try:
        if x is None:
            return default
        return int(str(x).strip())
    except Exception:
        return default


def _extract_order_id_from_raw(raw: Any) -> Optional[str]:
    """
    Best-effort: extrai order_id do `result.raw` do executor_jsonl.
    """
    if not isinstance(raw, dict):
        return None
    try:
        oid = str(raw.get("order_id") or "").strip() or None
    except Exception:
        oid = None
    if oid:
        return oid
    # fallback: tenta achar em order_resp
    try:
        resp = raw.get("order_resp")
        if isinstance(resp, dict):
            for k in ("id", "order_id", "orderId", "uuid", "uid"):
                v = resp.get(k)
                if v is None:
                    continue
                s = str(v).strip()
                if s:
                    return s
            for k in ("data", "order", "result"):
                v = resp.get(k)
                if isinstance(v, dict):
                    for kk in ("id", "order_id", "orderId", "uuid", "uid"):
                        vv = v.get(kk)
                        if vv is None:
                            continue
                        s = str(vv).strip()
                        if s:
                            return s
        if isinstance(resp, str):
            s = resp.strip()
            return s or None
    except Exception:
        return None
    return None


def _slip_raw_pct(*, odd_dec: Optional[float], odd_fin: Optional[float]) -> Optional[float]:
    try:
        if odd_dec is None or odd_fin is None or float(odd_dec) <= 0:
            return None
        return float((float(odd_fin) - float(odd_dec)) / float(odd_dec) * 100.0)
    except Exception:
        return None


def _safe_int_or_none(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(float(str(x).strip()))
    except Exception:
        return None


def _safe_float_or_none(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        s = s.replace(",", ".")
        return float(s)
    except Exception:
        return None


def _approx_eq(a: Any, b: float, *, eps: float = 1e-6) -> bool:
    try:
        if a is None:
            return False
        return abs(float(a) - float(b)) <= float(eps)
    except Exception:
        return False


def _stake_bucket(stake: Any, *, hi_min: float, lo_ref: float = 1.5) -> str:
    """
    Bucket simples para acompanhamento operacional do sizing:
    - "HI" para stake > hi_min
    - "LO" para stake≈lo_ref
    - "other" para valores diferentes/ausentes
    """
    if stake is None:
        return "NA"
    try:
        if float(stake) > float(hi_min):
            return "HI"
    except Exception:
        return "NA"
    if _approx_eq(stake, float(lo_ref), eps=0.02):
        return "LO"
    return "other"


def _in_range(x: Any, lo: float, hi: float) -> bool:
    try:
        v = float(x)
        return float(lo) <= v <= float(hi)
    except Exception:
        return False


def _slip_bucket_3(slip_pct: Any) -> str:
    s = _safe_float_or_none(slip_pct)
    if s is None:
        return "NA"
    if s <= -2.0:
        return "<= -2%"
    if s <= 2.0:
        return "(-2, 2]"
    return "> 2%"


def _bootstrap_ci_mean(xs: List[float], *, ci: float = 0.90, n_boot: int = 2000, seed: int = 0) -> Optional[Tuple[float, float]]:
    """
    IC bootstrap para a média (percentil). Retorna (lb, ub).
    xs: amostras (ex.: ROI por ordem).
    """
    try:
        xs2 = [float(x) for x in (xs or [])]
    except Exception:
        xs2 = []
    if len(xs2) < 5:
        return None
    n = len(xs2)
    nb = int(max(100, int(n_boot)))
    rnd = random.Random(int(seed))
    means: List[float] = []
    for _ in range(nb):
        s = 0.0
        for _j in range(n):
            s += xs2[rnd.randrange(0, n)]
        means.append(s / float(n))
    means.sort()
    alpha = float(max(0.0, min(1.0, 1.0 - float(ci))))
    lo = int(round((alpha / 2.0) * (len(means) - 1)))
    hi = int(round((1.0 - alpha / 2.0) * (len(means) - 1)))
    lo = max(0, min(len(means) - 1, lo))
    hi = max(0, min(len(means) - 1, hi))
    return float(means[lo]), float(means[hi])


def _latest_open_stakes_csv(out_dir: Path) -> Optional[Path]:
    try:
        cands = sorted(out_dir.glob("*__open_stakes.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        return cands[0] if cands else None
    except Exception:
        return None


def _open_order_ids_from_open_stakes_csv(path: Path) -> Optional[set[str]]:
    """
    Retorna set(order_id) em aberto (ainda não liquidadas) a partir do CSV open_stakes.
    Best-effort: se não conseguir achar coluna de order_id, retorna None.
    """
    if not path or not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            r = csv.DictReader(f)
            cols = list(r.fieldnames or [])
            if not cols:
                return None
            oid_col = _pick_col(cols, ("order_id", "order id", "order", "bet id", "bet_id", "id"))
            if not oid_col:
                return None
            out: set[str] = set()
            for row in r:
                if not isinstance(row, dict):
                    continue
                oid = str(row.get(oid_col) or "").strip()
                if not oid:
                    continue
                # geralmente é numérico; se não for, ainda assim guardamos o string (para joins futuros)
                out.add(oid)
            return out
    except Exception:
        return None


def _agg_pnl_exposure(rows: list[dict]) -> Dict[str, Any]:
    """
    Agrega {pnl, exposure} e retorna ROIw=(∑pnl)/(∑exposure)*100.
    """
    try:
        n = int(len(rows or []))
        exp = float(sum(float(r.get("exposure") or 0.0) for r in (rows or []) if r.get("exposure") is not None))
        pnl = float(sum(float(r.get("pnl") or 0.0) for r in (rows or []) if r.get("pnl") is not None))
        roi_w = (float(pnl) / float(exp) * 100.0) if exp > 0 else None
        return {"n": n, "exposure_sum": exp, "pnl_sum": pnl, "roi_weighted": roi_w}
    except Exception:
        return {"n": int(len(rows or [])), "exposure_sum": 0.0, "pnl_sum": 0.0, "roi_weighted": None}


def _counterfactual_filters_back(
    rows: list[dict],
    *,
    slip_raw_pct_max: float = 2.0,
    lat_ms_max: int = 6000,
    slip_missing_pass: bool = True,
    lat_missing_fail_closed: bool = True,
) -> Dict[str, Any]:
    """
    Contrafactual operacional aplicado a rows com {pnl, exposure, slip_raw_pct, lat_ms}.
    """
    base = _agg_pnl_exposure(rows)

    def _pass_slip(r: Dict[str, Any]) -> bool:
        s = r.get("slip_raw_pct")
        if s is None:
            return bool(slip_missing_pass)
        try:
            return float(s) <= float(slip_raw_pct_max)
        except Exception:
            return bool(slip_missing_pass)

    def _pass_lat(r: Dict[str, Any]) -> bool:
        t = r.get("lat_ms")
        if t is None:
            return (not bool(lat_missing_fail_closed))
        try:
            return float(t) <= float(lat_ms_max)
        except Exception:
            return (not bool(lat_missing_fail_closed))

    after_slip_rows = [r for r in rows if _pass_slip(r)]
    after_lat_rows = [r for r in rows if _pass_lat(r)]
    after_both_rows = [r for r in rows if _pass_slip(r) and _pass_lat(r)]
    after_slip = _agg_pnl_exposure(after_slip_rows)
    after_lat = _agg_pnl_exposure(after_lat_rows)
    after_both = _agg_pnl_exposure(after_both_rows)

    def _delta(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        try:
            dpnl = float(b.get("pnl_sum") or 0.0) - float(a.get("pnl_sum") or 0.0)
        except Exception:
            dpnl = None
        try:
            droi = None
            if a.get("roi_weighted") is not None and b.get("roi_weighted") is not None:
                droi = float(b.get("roi_weighted")) - float(a.get("roi_weighted"))
        except Exception:
            droi = None
        try:
            pass_n = (float(b.get("n") or 0) / float(a.get("n") or 0) * 100.0) if float(a.get("n") or 0) > 0 else None
        except Exception:
            pass_n = None
        try:
            pass_exp = (float(b.get("exposure_sum") or 0.0) / float(a.get("exposure_sum") or 0.0) * 100.0) if float(a.get("exposure_sum") or 0.0) > 0 else None
        except Exception:
            pass_exp = None
        return {"delta_pnl_sum": dpnl, "delta_roi_weighted": droi, "pass_n_pct": pass_n, "pass_exposure_pct": pass_exp}

    out: Dict[str, Any] = {
        "rule": {
            "slip_raw_pct_max": float(slip_raw_pct_max),
            "lat_ms_max": int(lat_ms_max),
            "slip_missing_pass": bool(slip_missing_pass),
            "lat_missing_fail_closed": bool(lat_missing_fail_closed),
        },
        "base": base,
        "after_slip": after_slip,
        "after_lat": after_lat,
        "after_both": after_both,
        "effect": {
            "slip": _delta(base, after_slip),
            "lat": _delta(base, after_lat),
            "both": _delta(base, after_both),
        },
    }
    return out


def _parse_executor_jsonl_back_live_orders(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Lê executor_jsonl e retorna order_id -> métricas para Back LIVE_OK:
      {created_at, slip_raw_pct, lat_ms, exposure, audit_id, is_live_mode, pre_submit_ms, slippage_pre_pct, market_regime}
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        req = obj.get("request") if isinstance(obj.get("request"), dict) else {}
        res = obj.get("result") if isinstance(obj.get("result"), dict) else {}
        st = str(res.get("status") or "").strip()
        if st != "LIVE_OK":
            continue
        exec_side = str(res.get("exec_side") or req.get("exec_side") or "").strip().lower()
        if exec_side != "back":
            continue
        created = _parse_dt_any(str(res.get("created_at") or req.get("created_at") or ""))
        if created is None:
            # fallback: tenta ISO parser já existente na base via fromisoformat (em _parse_dt_any)
            continue
        raw = res.get("raw") if isinstance(res.get("raw"), dict) else {}
        oid = _extract_order_id_from_raw(raw)
        if not oid or not str(oid).isdigit():
            continue
        audit_id = None
        try:
            audit_id = _safe_int(res.get("audit_id")) if res.get("audit_id") is not None else (_safe_int(req.get("audit_id")) if req.get("audit_id") is not None else None)
        except Exception:
            audit_id = None
        sent = raw.get("sent") if isinstance(raw.get("sent"), dict) else {}
        stake = _safe_float(sent.get("stake"))
        if stake is None:
            pol = res.get("policy") if isinstance(res.get("policy"), dict) else (req.get("policy") if isinstance(req.get("policy"), dict) else {})
            stake = _safe_float((pol or {}).get("stake_requested"))
        timing = res.get("timing") if isinstance(res.get("timing"), dict) else {}
        lat_ms = _safe_float(timing.get("call_to_done_ms"))
        lat_ms_i = int(lat_ms) if lat_ms is not None else None
        odd_dec = _safe_float(res.get("odd_at_decision") if res.get("odd_at_decision") is not None else req.get("odd_at_decision"))
        odd_fin = _safe_float(res.get("odd_final"))
        slip = _slip_raw_pct(odd_dec=odd_dec, odd_fin=odd_fin)
        vs = raw.get("value_sizing") if isinstance(raw.get("value_sizing"), dict) else {}
        pre_submit_ms = _safe_int_or_none(vs.get("pre_submit_ms"))
        slip_pre = _safe_float_or_none(vs.get("slippage_pre_pct"))
        mreg = str(vs.get("market_regime") or "").strip() or None
        # IMPORTANTE: no executor, `is_live` significa "modo LIVE (apostar de verdade)", não "in-play".
        is_live_mode = res.get("is_live") if res.get("is_live") is not None else req.get("is_live")
        rec = {
            "order_id": str(oid),
            "created_at": created.astimezone(timezone.utc),
            "slip_raw_pct": slip,
            "lat_ms": (float(lat_ms_i) if lat_ms_i is not None else None),
            "exposure": (float(stake) if stake is not None else None),
            "audit_id": (int(audit_id) if audit_id is not None else None),
            "is_live_mode": bool(is_live_mode) if is_live_mode is not None else False,
            "pre_submit_ms": (int(pre_submit_ms) if pre_submit_ms is not None else None),
            "slippage_pre_pct": (float(slip_pre) if slip_pre is not None else None),
            "market_regime": mreg,
        }
        prev = out.get(str(oid))
        if prev is None or rec["created_at"] >= prev.get("created_at"):
            out[str(oid)] = rec
    return out


def _append_backpre_fast_slow_sections(
    out_lines: list[str],
    *,
    exec_by_oid_back: Dict[str, Dict[str, Any]],
    acct_pnl_by_oid_total: Dict[str, float],
    audit_by_id: Optional[Dict[int, Dict[str, Any]]],
    open_order_ids: Optional[set[str]] = None,
) -> None:
    """
    Seções para acompanhar a tese Back Pre fast (pre_submit_ms<=5s) e o sizing (stake HI vs LO):
    - contagem por grupo e por stake_bucket
    - P&L/ROIw (accounting ledger por order_id)
    - slippage_pre_pct por grupo
    - robustez: IC90/IC95 (bootstrap) para média do ROI por ordem
    """
    if not exec_by_oid_back:
        return
    try:
        thr_ms_post = int(float(os.getenv("EXECUTOR_BACKPRE_FAST_MAX_PRE_SUBMIT_MS", "5000") or 5000))
    except Exception:
        thr_ms_post = 5000
    try:
        thr_ms_old = int(float(os.getenv("DAILY_BACKPRE_FAST_OLD_MAX_PRE_SUBMIT_MS", "6000") or 6000))
    except Exception:
        thr_ms_old = 6000
    try:
        n_boot = int(float(os.getenv("DAILY_BACKPRE_FAST_BOOTSTRAP_N", "2000") or 2000))
    except Exception:
        n_boot = 2000
    try:
        min_n = int(float(os.getenv("DAILY_BACKPRE_FAST_MIN_ORDERS", "25") or 25))
    except Exception:
        min_n = 25

    # “início operacional” da tese (quando stake=HI foi habilitado em produção).
    # Se vazio, usa tudo (comportamento antigo).
    thesis_start_day = str(os.getenv("DAILY_BACKPRE_FAST_THESIS_START_DAY", "") or "").strip()
    stake_hi = _safe_float_or_none(os.getenv("EXECUTOR_BACKPRE_FAST_STAKE_HI", "20") or 20.0) or 20.0
    stake_lo = _safe_float_or_none(os.getenv("EXECUTOR_BACKPRE_FAST_STAKE_LO", "1.5") or 1.5) or 1.5
    thesis_hi_min = _safe_float_or_none(os.getenv("DAILY_BACKPRE_FAST_HI_MIN", "5.0") or 5.0) or 5.0
    thesis_hi_max_raw = str(os.getenv("DAILY_BACKPRE_FAST_HI_MAX", "") or "").strip()
    thesis_hi_max = _safe_float_or_none(thesis_hi_max_raw) if thesis_hi_max_raw else None
    if thesis_hi_max is not None and float(thesis_hi_max) <= float(thesis_hi_min):
        thesis_hi_max = None
    old_hi_min = _safe_float_or_none(os.getenv("DAILY_BACKPRE_FAST_OLD_HI_MIN", "5.0") or 5.0) or 5.0
    old_hi_max = _safe_float_or_none(os.getenv("DAILY_BACKPRE_FAST_OLD_HI_MAX", "14.0") or 14.0) or 14.0
    if float(old_hi_max) < float(old_hi_min):
        old_hi_max = float(old_hi_min)
    has_transition = bool(thesis_start_day)
    try:
        old_period_end = (
            (date.fromisoformat(thesis_start_day) - timedelta(days=1)).isoformat() if has_transition else None
        )
    except Exception:
        old_period_end = None
    old_period_label = (f"até {old_period_end}" if old_period_end else f"< {thesis_start_day}") if has_transition else ""
    post_period_label = f"desde {thesis_start_day}" if has_transition else "janela analisada"
    old_hi_label = f"stake em [{_fmt_num(old_hi_min,2)}, {_fmt_num(old_hi_max,2)}]"
    post_hi_label = (
        f"stake > {_fmt_num(thesis_hi_min,2)}"
        if thesis_hi_max is None
        else f"stake em ({_fmt_num(thesis_hi_min,2)}, {_fmt_num(thesis_hi_max,2)}]"
    )
    fast_dyn_key = "Back Pre fast (critério dinâmico por período)"
    slow_dyn_key = "Back Pre slow (critério dinâmico por período)"
    if has_transition:
        fast_old_diag_key = f"Back Pre fast ({old_period_label}; pre_submit_ms<= {thr_ms_old}ms)"
        fast_post_diag_key = f"Back Pre fast ({post_period_label}; pre_submit_ms<= {thr_ms_post}ms)"
        slow_old_diag_key = f"Back Pre slow ({old_period_label}; pre_submit_ms> {thr_ms_old}ms)"
        slow_post_diag_key = f"Back Pre slow ({post_period_label}; pre_submit_ms> {thr_ms_post}ms)"
        thesis_old_key = f"Back Pre fast ({old_period_label}; {old_hi_label}; pre_submit_ms<= {thr_ms_old}ms)"
    else:
        fast_old_diag_key = None
        fast_post_diag_key = f"Back Pre fast (pre_submit_ms<= {thr_ms_post}ms)"
        slow_old_diag_key = None
        slow_post_diag_key = f"Back Pre slow (pre_submit_ms> {thr_ms_post}ms)"
        thesis_old_key = None
    thesis_post_key = f"Back Pre fast ({post_period_label}; {post_hi_label}; pre_submit_ms<= {thr_ms_post}ms)"
    aux_low_old_key = (
        f"Back Pre slow ({old_period_label}; stake < {_fmt_num(old_hi_min,2)}; pre_submit_ms> {thr_ms_old}ms)"
        if has_transition
        else None
    )
    aux_low_post_key = (
        f"Back Pre slow ({post_period_label}; stake < {_fmt_num(thesis_hi_min,2)}; pre_submit_ms> {thr_ms_post}ms)"
    )

    # rows com accounting (pnl) + stake (exposure)
    groups_all: Dict[str, List[Dict[str, Any]]] = defaultdict(list)  # diagnóstico (todos stakes)
    groups_thesis: Dict[str, List[Dict[str, Any]]] = defaultdict(list)  # tese (fast + stake HI)
    groups_aux_low: Dict[str, List[Dict[str, Any]]] = defaultdict(list)  # slow + stake abaixo do limiar
    for oid, em in (exec_by_oid_back or {}).items():
        if not isinstance(em, dict):
            continue
        created = em.get("created_at")
        if not isinstance(created, datetime):
            continue
        exp = _safe_float_or_none(em.get("exposure"))
        if exp is None or exp <= 0:
            continue
        pnl = acct_pnl_by_oid_total.get(str(oid))
        if pnl is None:
            continue

        # Pre/In: preferir audit (kickoff/is_live); fallback: market_regime quando existir
        is_in = None
        try:
            aid = em.get("audit_id")
            arow = (audit_by_id or {}).get(int(aid)) if (aid is not None and audit_by_id is not None) else None
            if isinstance(arow, dict):
                is_in = bool(_is_inplay_from_audit_row(arow, exec_created_at_utc=created))
        except Exception:
            is_in = None
        if is_in is None:
            try:
                mreg = str(em.get("market_regime") or "").strip().lower()
                if mreg in ("pre", "in"):
                    is_in = bool(mreg == "in")
            except Exception:
                is_in = None

        pre_submit_ms = _safe_int_or_none(em.get("pre_submit_ms"))
        slip_pre = _safe_float_or_none(em.get("slippage_pre_pct"))
        created_day = str(created.date().isoformat())
        is_post = bool(has_transition and created_day >= thesis_start_day)
        if is_post:
            fast_thr = int(thr_ms_post)
            hi_min = float(thesis_hi_min)
            hi_max = (float(thesis_hi_max) if thesis_hi_max is not None else None)
            hi_inclusive_min = False
            fast_reg_key = fast_post_diag_key
            slow_reg_key = slow_post_diag_key
            thesis_key = thesis_post_key
            low_key = aux_low_post_key
        else:
            fast_thr = int(thr_ms_old)
            hi_min = float(old_hi_min)
            hi_max = float(old_hi_max)
            hi_inclusive_min = True
            fast_reg_key = fast_old_diag_key
            slow_reg_key = slow_old_diag_key
            thesis_key = thesis_old_key
            low_key = aux_low_old_key
        is_hi = ((float(exp) >= hi_min) if hi_inclusive_min else (float(exp) > hi_min)) and (
            hi_max is None or float(exp) <= float(hi_max)
        )
        if is_hi:
            stake_b = "HI"
        elif _approx_eq(exp, float(stake_lo), eps=0.02):
            stake_b = "LO"
        else:
            stake_b = "other"
        roi_i = float(pnl) / float(exp) * 100.0
        row = {
            "order_id": str(oid),
            "pnl": float(pnl),
            "exposure": float(exp),
            "roi": float(roi_i),
            "stake_bucket": stake_b,
            "pre_submit_ms": pre_submit_ms,
            "slippage_pre_pct": slip_pre,
            "created_day": created_day,
        }

        if bool(is_in):
            groups_all["Back In"].append(row)
        else:
            # Pre
            if pre_submit_ms is None:
                groups_all["Back Pre (pre_submit_ms NA)"].append(row)
            elif int(pre_submit_ms) <= int(fast_thr):
                groups_all[fast_dyn_key].append(row)
                if fast_reg_key:
                    groups_all[fast_reg_key].append(row)
                if is_hi and thesis_key:
                    groups_thesis[thesis_key].append(row)
            else:
                groups_all[slow_dyn_key].append(row)
                if slow_reg_key:
                    groups_all[slow_reg_key].append(row)
                if low_key and float(exp) < float(thesis_hi_min if is_post else old_hi_min):
                    groups_aux_low[low_key].append(row)

    if not groups_all and not groups_thesis:
        return

    # -------------------------
    # A) Performance da tese (stake=HI) com métricas “liquidadas”
    # -------------------------
    out_lines.append("**Tese: Back Pre fast (pós-início; elegível HI) — performance (accounting; order_id)**\n\n")
    if has_transition:
        out_lines.append(
            f"- Critério antigo (`{old_period_label}`): `{old_hi_label}` e `pre_submit_ms<= {int(thr_ms_old)}ms`.\n"
            f"- Critério atual (`{post_period_label}`): `{post_hi_label}` e `pre_submit_ms<= {int(thr_ms_post)}ms`.\n"
            f"- Stake HI configurado no executor (`EXECUTOR_BACKPRE_FAST_STAKE_HI`): `{_fmt_num(stake_hi,2)}`.\n\n"
        )
    else:
        out_lines.append(
            f"- Critério aplicado: `{post_hi_label}` e `pre_submit_ms<= {int(thr_ms_post)}ms`.\n"
            f"- Stake HI configurado no executor (`EXECUTOR_BACKPRE_FAST_STAKE_HI`): `{_fmt_num(stake_hi,2)}`.\n\n"
        )
    out_lines.append("| Grupo | n_ordens | n_liquidadas | n_abertas | Stake_liquidado (∑) | P&L_liquidado (∑acct) | ROIw_liquidado |\n")
    out_lines.append("|---|---:|---:|---:|---:|---:|---:|\n")

    def _split_settled(rows: List[Dict[str, Any]]) -> Tuple[Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]:
        # Sem open_stakes.csv não sabemos o que está liquidado vs aberto.
        if open_order_ids is None:
            return None, None
        settled = []
        open_rows = []
        for r in rows or []:
            oid0 = str(r.get("order_id") or "").strip()
            if oid0 and oid0 in open_order_ids:
                open_rows.append(r)
            else:
                settled.append(r)
        return settled, open_rows

    thesis_order: List[str] = []
    if thesis_old_key:
        thesis_order.append(thesis_old_key)
    thesis_order.append(thesis_post_key)
    for g in sorted((groups_thesis or {}).keys(), key=lambda x: (thesis_order.index(x) if x in thesis_order else 999, x)):
        rows = groups_thesis.get(g) or []
        settled_rows, open_rows = _split_settled(rows)
        if settled_rows is None or open_rows is None:
            out_lines.append(f"| {g} | {len(rows)} | — | — | — | — | — |\n")
        else:
            summ_set = _summarize_rows_pnl_exp(settled_rows)
            out_lines.append(
                f"| {g} | {len(rows)} | {len(settled_rows)} | {len(open_rows)} | {_fmt_num(summ_set.get('exposure_sum'),2)} | {_fmt_num(summ_set.get('pnl_sum'),2)} | {_fmt_pct(summ_set.get('roi_weighted'))} |\n"
            )
    out_lines.append("\n")
    if open_order_ids is None:
        out_lines.append("_Nota: `n_liquidadas`/`ROIw_liquidado` requer `open_stakes.csv` (accounting). Sem isso, este bloco fica como `—`._\n\n")

    # Performance auxiliar: slow/fallback abaixo do limiar HI (ex.: latência acima do limiar do período)
    out_lines.append("**Back Pre slow (pós-início; stake < limiar HI) — performance auxiliar (accounting; order_id)**\n\n")
    if has_transition:
        out_lines.append(
            f"- Critério antigo (`{old_period_label}`): `pre_submit_ms> {int(thr_ms_old)}ms` e `stake < {_fmt_num(old_hi_min,2)}`.\n"
            f"- Critério atual (`{post_period_label}`): `pre_submit_ms> {int(thr_ms_post)}ms` e `stake < {_fmt_num(thesis_hi_min,2)}`.\n\n"
        )
    else:
        out_lines.append(
            f"- Critério aplicado: `pre_submit_ms> {int(thr_ms_post)}ms` e `stake < {_fmt_num(thesis_hi_min,2)}`.\n\n"
        )
    out_lines.append("| Grupo | n_ordens | n_liquidadas | n_abertas | Stake_liquidado (∑) | P&L_liquidado (∑acct) | ROIw_liquidado |\n")
    out_lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    aux_order: List[str] = []
    if aux_low_old_key:
        aux_order.append(aux_low_old_key)
    aux_order.append(aux_low_post_key)
    for g in sorted((groups_aux_low or {}).keys(), key=lambda x: (aux_order.index(x) if x in aux_order else 999, x)):
        rows = groups_aux_low.get(g) or []
        settled_rows, open_rows = _split_settled(rows)
        if settled_rows is None or open_rows is None:
            out_lines.append(f"| {g} | {len(rows)} | — | — | — | — | — |\n")
        else:
            summ_set = _summarize_rows_pnl_exp(settled_rows)
            out_lines.append(
                f"| {g} | {len(rows)} | {len(settled_rows)} | {len(open_rows)} | {_fmt_num(summ_set.get('exposure_sum'),2)} | {_fmt_num(summ_set.get('pnl_sum'),2)} | {_fmt_pct(summ_set.get('roi_weighted'))} |\n"
            )
    out_lines.append("\n")

    def _cnt_stake(rows: List[Dict[str, Any]]) -> Dict[str, int]:
        c: Dict[str, int] = {"HI": 0, "LO": 0, "other": 0}
        for r in rows or []:
            sb = str(r.get("stake_bucket") or "")
            if sb == "HI":
                c["HI"] += 1
            elif sb == "LO":
                c["LO"] += 1
            else:
                c["other"] += 1
        return c

    # -------------------------
    # B) Compliance: fast/slow/NA (todos stakes) + contagem por stake bucket
    # -------------------------
    out_lines.append("**Tese Back Pre fast — compliance (pós-início; distribuição de stake e pre_submit_ms)**\n\n")
    out_lines.append(
        f"| Grupo | n_ordens | stake=HI (critério por período) | stake≈{_fmt_num(stake_lo,2)} | stake=other/NA |\n"
    )
    out_lines.append("|---|---:|---:|---:|---:|\n")
    order_diag = [fast_dyn_key, slow_dyn_key]
    if fast_old_diag_key:
        order_diag.append(fast_old_diag_key)
    if fast_post_diag_key:
        order_diag.append(fast_post_diag_key)
    if slow_old_diag_key:
        order_diag.append(slow_old_diag_key)
    if slow_post_diag_key:
        order_diag.append(slow_post_diag_key)
    order_diag.extend(["Back Pre (pre_submit_ms NA)", "Back In"])
    for g in order_diag:
        rows = groups_all.get(g) or []
        if not rows:
            continue
        stc = _cnt_stake(rows)
        out_lines.append(f"| {g} | {len(rows)} | {int(stc.get('HI') or 0)} | {int(stc.get('LO') or 0)} | {int(stc.get('other') or 0)} |\n")
    out_lines.append("\n")

    # Slippage_pre_pct por grupo (3 buckets)
    out_lines.append("**Tese: Back Pre fast — slippage_pre_pct (bucket 3-way; accounting por order_id)**\n\n")
    out_lines.append("| Grupo | n_ordens | slippage_pre_pct mean | slippage_pre_pct mediana | <= -2% | (-2,2] | > 2% | NA |\n")
    out_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for g in order_diag:
        rows = groups_all.get(g) or []
        if not rows:
            continue
        slips = [float(r.get("slippage_pre_pct")) for r in rows if r.get("slippage_pre_pct") is not None]
        mean_s = float(statistics.fmean(slips)) if slips else None
        med_s = float(statistics.median(slips)) if slips else None
        bc = {"<= -2%": 0, "(-2, 2]": 0, "> 2%": 0, "NA": 0}
        for r in rows:
            b = _slip_bucket_3(r.get("slippage_pre_pct"))
            bc[b] = int(bc.get(b) or 0) + 1
        out_lines.append(
            f"| {g} | {len(rows)} | {_fmt_pct(mean_s)} | {_fmt_pct(med_s)} | {bc.get('<= -2%')} | {bc.get('(-2, 2]')} | {bc.get('> 2%')} | {bc.get('NA')} |\n"
        )
    out_lines.append("\n")

    # Robustez: delta fast-slow
    out_lines.append("**Tese: Back Pre fast vs slow — diferença de ROI mean (por ordem)**\n\n")
    try:
        fast = groups_all.get(fast_dyn_key) or []
        slow = groups_all.get(slow_dyn_key) or []
        fast_roi = [float(r.get("roi")) for r in fast if r.get("roi") is not None]
        slow_roi = [float(r.get("roi")) for r in slow if r.get("roi") is not None]
        out_lines.append(
            f"- Critério dinâmico por período (pré: `<= {int(thr_ms_old)}ms`; pós: `<= {int(thr_ms_post)}ms`).\n"
        )
        out_lines.append(f"- Amostra líquida: fast=`{len(fast_roi)}` | slow=`{len(slow_roi)}` | min_n=`{min_n}`.\n")
        if len(fast_roi) >= min_n and len(slow_roi) >= min_n:
            # delta por bootstrap: resample separadamente e computa mean(fast)-mean(slow)
            rnd = random.Random(123)
            deltas: List[float] = []
            nf = len(fast_roi)
            ns = len(slow_roi)
            for _ in range(int(max(300, n_boot))):
                mf = 0.0
                ms = 0.0
                for _j in range(nf):
                    mf += float(fast_roi[rnd.randrange(0, nf)])
                for _j in range(ns):
                    ms += float(slow_roi[rnd.randrange(0, ns)])
                deltas.append((mf / float(nf)) - (ms / float(ns)))
            deltas.sort()
            lo90 = deltas[int(round(0.05 * (len(deltas) - 1)))]
            hi90 = deltas[int(round(0.95 * (len(deltas) - 1)))]
            lo95 = deltas[int(round(0.025 * (len(deltas) - 1)))]
            hi95 = deltas[int(round(0.975 * (len(deltas) - 1)))]
            out_lines.append(f"- Delta (fast − slow) IC90 bootstrap: `{_fmt_pct(lo90)} .. {_fmt_pct(hi90)}`.\n")
            out_lines.append(f"- Delta (fast − slow) IC95 bootstrap: `{_fmt_pct(lo95)} .. {_fmt_pct(hi95)}`.\n\n")
        else:
            out_lines.append(
                f"- _N insuficiente para inferência bootstrap (fast={len(fast_roi)}, slow={len(slow_roi)}, min_n={min_n})._\n\n"
            )
    except Exception as e:
        out_lines.append(
            f"- _Erro ao montar seção fast vs slow: `{type(e).__name__}: {str(e)[:180]}`._\n\n"
        )


def _acct_amount_by_order_day_from_balance_csv(
    path: Path,
    *,
    days_utc: Optional[set[str]] = None,
    only_type_bet: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Retorna: order_id -> day(UTC) -> amount_sum.
    Necessário para contrafactual exato no accounting ledger (post date UTC).
    """
    out: Dict[str, Dict[str, float]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        r = csv.DictReader(f)
        cols = list(r.fieldnames or [])
        if not cols:
            return out
        cols_l = [c.lower() for c in cols]
        # heurística similar a ops.reconcile_oos_vs_realized
        dt_col = None
        for k in ("post date", "post_date", "date", "settled", "closed", "time"):
            for c in cols:
                if c.lower() == k or c.lower().startswith(k) or k in c.lower():
                    dt_col = c
                    break
            if dt_col:
                break
        pnl_col = None
        for k in ("amount", "profit_loss", "profit", "p&l", "pnl", "net", "pl"):
            for c in cols:
                if c.lower() == k or c.lower().startswith(k) or k in c.lower():
                    pnl_col = c
                    break
            if pnl_col:
                break
        typ_col = None
        for c in cols:
            if c.lower() == "type" or c.lower().startswith("type"):
                typ_col = c
                break
        oid_col = None
        for k in ("order_id", "order id", "order", "bet id", "bet_id", "id"):
            for c in cols:
                cl = c.lower()
                if cl == k or cl.startswith(k) or k in cl:
                    oid_col = c
                    break
            if oid_col:
                break
        for row in r:
            if not isinstance(row, dict):
                continue
            if not oid_col or not pnl_col or not dt_col:
                continue
            if only_type_bet and typ_col:
                typ = str(row.get(typ_col) or "").strip().lower()
                if typ and typ != "bet":
                    continue
            oid = str(row.get(oid_col) or "").strip()
            if not oid or not oid.isdigit():
                continue
            dt = _parse_dt_any(str(row.get(dt_col) or ""))
            if dt is None:
                continue
            day = dt.date().isoformat()
            if days_utc is not None and day not in days_utc:
                continue
            amt = _safe_float(row.get(pnl_col))
            if amt is None:
                continue
            blk = out.setdefault(oid, {})
            blk[day] = float(blk.get(day) or 0.0) + float(amt)
    return out


def _acct_amount_by_order_day_by_type_from_balance_csv(
    path: Path,
    *,
    days_utc: Optional[set[str]] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Retorna: order_id -> day(UTC) -> type_lower -> amount_sum.

    Útil para:
    - métricas de void/refund/cancel (quando aparecem como `type` ≠ bet)
    - P&L por ordem mais fiel (incluindo ajustes relacionados à ordem, se existirem)

    Observação: o CSV pode NÃO ter coluna `order_id` ou `type`. Nesses casos retorna {}.
    """
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    if not path.exists():
        return out
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            r = csv.DictReader(f)
            cols = list(r.fieldnames or [])
            if not cols:
                return out
            dt_col = None
            for k in ("post date", "post_date", "date", "settled", "closed", "time"):
                for c in cols:
                    cl = c.lower()
                    if cl == k or cl.startswith(k) or k in cl:
                        dt_col = c
                        break
                if dt_col:
                    break
            pnl_col = None
            for k in ("amount", "profit_loss", "profit", "p&l", "pnl", "net", "pl"):
                for c in cols:
                    cl = c.lower()
                    if cl == k or cl.startswith(k) or k in cl:
                        pnl_col = c
                        break
                if pnl_col:
                    break
            typ_col = None
            for c in cols:
                cl = c.lower()
                if cl == "type" or cl.startswith("type"):
                    typ_col = c
                    break
            oid_col = None
            for k in ("order_id", "order id", "order", "bet id", "bet_id", "id"):
                for c in cols:
                    cl = c.lower()
                    if cl == k or cl.startswith(k) or k in cl:
                        oid_col = c
                        break
                if oid_col:
                    break
            if not oid_col or not pnl_col or not dt_col:
                return out
            for row in r:
                if not isinstance(row, dict):
                    continue
                oid = str(row.get(oid_col) or "").strip()
                if not oid or not oid.isdigit():
                    continue
                dt = _parse_dt_any(str(row.get(dt_col) or ""))
                if dt is None:
                    continue
                day = dt.date().isoformat()
                if days_utc is not None and day not in days_utc:
                    continue
                amt = _safe_float(row.get(pnl_col))
                if amt is None:
                    continue
                typ = str(row.get(typ_col) or "").strip().lower() if typ_col else ""
                if not typ:
                    typ = "unknown"
                blk = out.setdefault(oid, {}).setdefault(day, {})
                blk[typ] = float(blk.get(typ) or 0.0) + float(amt)
    except Exception:
        return {}
    return out


def _summarize_accounting_types(
    acct_type_sums: Dict[str, float],
    *,
    exclude_pred=None,
) -> Dict[str, Any]:
    """
    Recebe mapa: type_lower -> amount_sum e produz resumo focado em void/refund/cancel.
    Heurísticas:
    - void/refund/push/cancel normalmente aparecem como movimentos "não resultado" (muitas vezes 0 ou reversões).
    - aqui apenas agregamos; interpretação é contextual e depende do provider.
    """
    try:
        items = []
        for k, v in (acct_type_sums or {}).items():
            try:
                kk = str(k)
                if exclude_pred is not None:
                    try:
                        if bool(exclude_pred(str(kk).lower())):
                            continue
                    except Exception:
                        pass
                items.append((kk, float(v)))
            except Exception:
                continue
        items.sort(key=lambda x: abs(float(x[1])), reverse=True)
    except Exception:
        items = []

    def _sum_if(pred) -> float:
        s = 0.0
        for k, v in items:
            try:
                if pred(str(k).lower()):
                    s += float(v)
            except Exception:
                continue
        return float(s)

    void_sum = _sum_if(lambda t: ("void" in t) or ("push" in t))
    # Alguns providers usam `voided`/`refunded`
    refund_sum = _sum_if(lambda t: ("refund" in t) or ("refunded" in t))
    # "Cancel" pode aparecer como "cancelled"/"canceled" e variações
    cancel_sum = _sum_if(lambda t: ("cancel" in t) or ("canceled" in t) or ("cancelled" in t))
    bet_sum = _sum_if(lambda t: t == "bet" or t.startswith("bet"))
    other_sum = float(sum(float(v) for _, v in items)) - float(void_sum + refund_sum + cancel_sum + bet_sum)

    top = [{"type": k, "amount_sum": v} for k, v in items[:10]]
    return {
        "bet_sum": bet_sum,
        "void_push_sum": void_sum,
        "refund_sum": refund_sum,
        "cancel_sum": cancel_sum,
        "other_sum": other_sum,
        "top_types": top,
    }


def _acct_amount_by_day_type_from_balance_csv(
    path: Path,
    *,
    days_utc: Optional[set[str]] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Retorna: day(UTC) -> type_lower -> {amount_sum, n_rows}
    Para diagnóstico de void/refund/cancel e para P&L por post date UTC consistente com o daily (aderência usa UTC).
    """
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if not path.exists():
        return out
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            r = csv.DictReader(f)
            cols = list(r.fieldnames or [])
            if not cols:
                return out
            dt_col = None
            for k in ("post date", "post_date", "date", "settled", "closed", "time"):
                for c in cols:
                    cl = c.lower()
                    if cl == k or cl.startswith(k) or k in cl:
                        dt_col = c
                        break
                if dt_col:
                    break
            pnl_col = None
            for k in ("amount", "profit_loss", "profit", "p&l", "pnl", "net", "pl"):
                for c in cols:
                    cl = c.lower()
                    if cl == k or cl.startswith(k) or k in cl:
                        pnl_col = c
                        break
                if pnl_col:
                    break
            typ_col = None
            for c in cols:
                cl = c.lower()
                if cl == "type" or cl.startswith("type"):
                    typ_col = c
                    break
            if not dt_col or not pnl_col:
                return out
            for row in r:
                if not isinstance(row, dict):
                    continue
                dt = _parse_dt_any(str(row.get(dt_col) or ""))
                if dt is None:
                    continue
                day = dt.date().isoformat()
                if days_utc is not None and day not in days_utc:
                    continue
                amt = _safe_float(row.get(pnl_col))
                if amt is None:
                    continue
                typ = str(row.get(typ_col) or "").strip().lower() if typ_col else ""
                if not typ:
                    typ = "unknown"
                blk = out.setdefault(day, {}).setdefault(typ, {"amount_sum": 0.0, "n_rows": 0})
                blk["amount_sum"] = float(blk.get("amount_sum") or 0.0) + float(amt)
                blk["n_rows"] = int(blk.get("n_rows") or 0) + 1
    except Exception:
        return {}
    return out


def _split_back_acct_pnl_pre_in_by_order_id(
    *,
    exec_by_oid: Dict[str, Dict[str, Any]],
    acct_by_oid_day_typ: Dict[str, Dict[str, Dict[str, float]]],
    day_utc: str,
    audit_by_id: Optional[Dict[int, Dict[str, Any]]] = None,
    include_types: Optional[set[str]] = None,
) -> Dict[str, Any]:
    """
    Particiona o P&L do accounting por dia (UTC, post date) em Back Pre vs Back In usando join por order_id.

    - exec_by_oid: order_id -> {created_at, exposure, slip_raw_pct, lat_ms, audit_id, is_live_mode, ...}
      (hoje vem de `_parse_executor_jsonl_back_live_orders`, portanto apenas Back+LIVE_OK)
    - audit_by_id: audit_id -> {kickoff_time, is_live, audited_at, ...} para classificar Pre/In corretamente.
    - acct_by_oid_day_typ: order_id -> day -> type_lower -> amount_sum (de balance.csv)
    - include_types: quais `type` do ledger entram no P&L da ordem. Default = {"bet"} para manter semântica antiga.

    Retorna:
      {
        pnl_pre, pnl_in, pnl_total, n_pre, n_in, n_total,
        coverage_n_pct, missing_orders_n, types_included, types_excluded_top
      }

    Observação importante:
    - Isso é “exato” apenas no sentido de ledger por order_id.
      Se void/refund for registrado como type≠bet, você pode incluir esses types via include_types.
    """
    # Se include_types=None: inclui todos os `type` “P&L-like” e exclui depósitos/saques/transferências/etc.
    # Se include_types=set(...): usa allowlist exata.
    dayk = str(day_utc or "").strip()
    if not dayk:
        return {
            "pnl_pre": None,
            "pnl_in": None,
            "pnl_total": None,
            "n_pre": 0,
            "n_in": 0,
            "n_total": 0,
            "coverage_n_pct": None,
            "missing_orders_n": 0,
            "types_included": (sorted(list(include_types)) if include_types is not None else ["__PNL_LIKE__"]),
            "types_excluded_top": [],
        }

    pnl_pre = 0.0
    pnl_in = 0.0
    pnl_tot = 0.0
    n_pre = 0
    n_in = 0
    n_tot = 0
    missing = 0
    excluded_types_sum: Dict[str, float] = defaultdict(float)

    for oid, ex in (exec_by_oid or {}).items():
        if not isinstance(ex, dict):
            continue
        amt_day = acct_by_oid_day_typ.get(str(oid)) if isinstance(acct_by_oid_day_typ.get(str(oid)), dict) else None
        if not amt_day or dayk not in amt_day:
            continue
        typ_map = amt_day.get(dayk) if isinstance(amt_day.get(dayk), dict) else {}
        if not typ_map:
            missing += 1
            continue

        # soma tipos incluídos; também registramos “outros/excluídos” para diagnóstico
        s_incl = 0.0
        got_any = False
        for typ, amt in (typ_map or {}).items():
            tl = str(typ or "").strip().lower() or "unknown"
            try:
                v = float(amt)
            except Exception:
                continue
            got_any = True
            if include_types is not None:
                if tl in include_types:
                    s_incl += float(v)
                else:
                    excluded_types_sum[tl] += float(v)
            else:
                # filtro “P&L-like” consistente com accounting_report.compute_pnl_report()
                excl = any(k in tl for k in ("deposit", "withdraw", "transfer", "top", "payment", "adjust", "bonus"))
                if excl:
                    excluded_types_sum[tl] += float(v)
                else:
                    s_incl += float(v)
        if not got_any:
            missing += 1
            continue

        # IMPORTANTE: `is_live_mode` do executor significa "modo LIVE", não "in-play".
        # Para Pre/In (prematch/inplay), usamos audit.kickoff_time quando disponível; senão audit.is_live quando não-NULL.
        try:
            aid = ex.get("audit_id")
            arow = (audit_by_id or {}).get(int(aid)) if (aid is not None and audit_by_id is not None) else None
            is_in = bool(_is_inplay_from_audit_row(arow, exec_created_at_utc=ex.get("created_at")))
        except Exception:
            is_in = False
        if is_in:
            pnl_in += float(s_incl)
            n_in += 1
        else:
            pnl_pre += float(s_incl)
            n_pre += 1
        pnl_tot += float(s_incl)
        n_tot += 1

    # cobertura: % de ordens do exec_by_oid que aparecem no ledger no dia
    try:
        denom = int(len(exec_by_oid or {}))
        cov = (float(n_tot) / float(denom) * 100.0) if denom > 0 else None
    except Exception:
        cov = None

    # top tipos excluídos por |impacto|
    try:
        top_ex = sorted([(k, float(v)) for k, v in excluded_types_sum.items()], key=lambda x: abs(float(x[1])), reverse=True)[:8]
        top_ex = [{"type": k, "amount_sum": v} for k, v in top_ex]
    except Exception:
        top_ex = []

    return {
        "pnl_pre": pnl_pre,
        "pnl_in": pnl_in,
        "pnl_total": pnl_tot,
        "n_pre": int(n_pre),
        "n_in": int(n_in),
        "n_total": int(n_tot),
        "coverage_n_pct": cov,
        "missing_orders_n": int(missing),
        "types_included": (sorted(list(include_types)) if include_types is not None else ["__PNL_LIKE__"]),
        "types_excluded_top": top_ex,
    }


def _extract_audit_ids_from_exec_by_oid(exec_by_oid: Dict[str, Dict[str, Any]]) -> list[int]:
    out: set[int] = set()
    for _, ex in (exec_by_oid or {}).items():
        if not isinstance(ex, dict):
            continue
        aid = ex.get("audit_id")
        if aid is None:
            continue
        try:
            out.add(int(aid))
        except Exception:
            continue
    return sorted(list(out))


def _pick_last_day_with_slippage_vs_roi_raw(per_day: list[dict]) -> Optional[dict]:
    """
    O bloco slippage×ROI depende de ROI (placar disponível). Em dias recentes pode estar vazio.
    Pegamos o último dia que tenha pelo menos 1 bucket (Back ou Lay).
    """
    try:
        for it in reversed(per_day or []):
            if not isinstance(it, dict):
                continue
            ex = it.get("execution") if isinstance(it.get("execution"), dict) else {}
            rawblk = ex.get("slippage_vs_roi_raw") if isinstance(ex.get("slippage_vs_roi_raw"), dict) else {}
            b = rawblk.get("back") if isinstance(rawblk.get("back"), dict) else {}
            l = rawblk.get("lay") if isinstance(rawblk.get("lay"), dict) else {}
            bb = b.get("buckets") if isinstance(b.get("buckets"), list) else []
            lb = l.get("buckets") if isinstance(l.get("buckets"), list) else []
            if bb or lb:
                return it
    except Exception:
        return None
    return None


def _bucketize_latency_call_to_done_ms_accounting(rows: list[dict]) -> list[dict]:
    """
    rows: {pnl, exposure, lat_ms}
    Retorna buckets com ROIw = sum(pnl)/sum(exposure)*100.
    """
    if not rows:
        return []

    def _lab(lat_ms: Optional[float]) -> str:
        if lat_ms is None:
            return "Desconhecido"
        x = float(lat_ms)
        if x < 5000:
            return "< 5s"
        if x < 10000:
            return "5-10s"
        if x < 20000:
            return "10-20s"
        if x < 40000:
            return "20-40s"
        return "> 40s"

    order = ["< 5s", "5-10s", "10-20s", "20-40s", "> 40s", "Desconhecido"]
    out = []
    for lab in order:
        sub = [r for r in rows if _lab(_safe_float(r.get("lat_ms"))) == lab]
        if not sub:
            continue
        agg = _agg_pnl_exposure(sub)
        out.append(
            {
                "bucket": lab,
                "n": int(agg.get("n") or 0),
                "exposure_sum": agg.get("exposure_sum"),
                "pnl_sum": agg.get("pnl_sum"),
                "roi_weighted": agg.get("roi_weighted"),
            }
        )
    return out


def _bucketize_slip_raw_3way_accounting(rows: list[dict]) -> list[dict]:
    """
    rows: {pnl, exposure, slip_raw_pct}
    Buckets com sinal: <=-2%, (-2,2], >2%, e "Desconhecido" quando slip_raw_pct=None.
    """
    if not rows:
        return []

    def _lab(slip: Optional[float]) -> str:
        if slip is None:
            return "Desconhecido"
        x = float(slip)
        if x <= -2.0:
            return "<= -2%"
        if x <= 2.0:
            return "(-2, 2]"
        return "> 2%"

    order = ["<= -2%", "(-2, 2]", "> 2%", "Desconhecido"]
    out = []
    for lab in order:
        sub = [r for r in rows if _lab(_safe_float(r.get("slip_raw_pct"))) == lab]
        if not sub:
            continue
        agg = _agg_pnl_exposure(sub)
        out.append(
            {
                "bucket": lab,
                "n": int(agg.get("n") or 0),
                "exposure_sum": agg.get("exposure_sum"),
                "pnl_sum": agg.get("pnl_sum"),
                "roi_weighted": agg.get("roi_weighted"),
            }
        )
    return out


def _slip_raw_3bucket_rows(buckets: list[dict]) -> list[dict]:
    """
    Normaliza para sempre retornar 3 buckets: <=-2%, (-2,2], >2%.
    buckets pode vir incompleto (quando N=0 em algum bucket).
    """
    want = ["<= -2%", "(-2, 2]", "> 2%"]
    by = {}
    for b in buckets or []:
        if isinstance(b, dict) and str(b.get("bucket") or ""):
            by[str(b.get("bucket"))] = b
    out = []
    for lab in want:
        it = by.get(lab) or {}
        out.append(
            {
                "bucket": lab,
                "n": int(it.get("n") or 0),
                "roi_mean": it.get("roi_mean"),
                "roi_se": it.get("roi_se"),
                "roi_ci95": it.get("roi_ci95"),
                "odd_median": it.get("odd_median"),
                "exposure_median": it.get("exposure_median"),
                "exposure_sum": it.get("exposure_sum"),
                "roi_weighted": it.get("roi_weighted"),
            }
        )
    return out


def _fmt_roi_mean_se_ci_pct(row: dict) -> str:
    """
    Formata ROI (mean) com SE e IC95% (quando disponíveis).
    """
    try:
        mean = row.get("roi_mean")
        if mean is None:
            return "—"
        se = row.get("roi_se")
        ci = row.get("roi_ci95") if isinstance(row.get("roi_ci95"), dict) else None
        if se is None and not ci:
            return _fmt_pct(mean)
        se_s = _fmt_pct(se) if se is not None else "—"
        if ci and (ci.get("lb") is not None) and (ci.get("ub") is not None):
            base = f"{_fmt_pct(mean)} (SE {se_s}) [{_fmt_pct(ci.get('lb'))}, {_fmt_pct(ci.get('ub'))}]"
        else:
            base = f"{_fmt_pct(mean)} (SE {se_s})"
        # ROI ponderado por exposição (quando disponível)
        if row.get("roi_weighted") is not None:
            base += f" | ROIw {_fmt_pct(row.get('roi_weighted'))}"
        return base
    except Exception:
        return "—"


def _acct_pnl_by_order_total_from_balance_csv(
    path: Path,
    *,
    only_type_bet: bool = True,
) -> Dict[str, float]:
    """
    Retorna: order_id -> pnl_total (soma de amount) no balance.csv.
    Usado para auditoria de ROI por ordem (accounting).
    """
    out: Dict[str, float] = {}
    if not path.exists():
        return out
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            r = csv.DictReader(f)
            cols = list(r.fieldnames or [])
            if not cols:
                return out
            dt_col = _pick_col(cols, ("post date", "post_date", "date", "settled", "closed", "time"))
            pnl_col = _pick_col(cols, ("amount", "profit_loss", "profit", "p&l", "pnl", "net", "pl"))
            typ_col = _pick_col(cols, ("type",))
            oid_col = _pick_col(cols, ("order_id", "order id", "order", "bet id", "bet_id", "id"))
            if not oid_col or not pnl_col:
                return out
            for row in r:
                if not isinstance(row, dict):
                    continue
                if only_type_bet and typ_col:
                    typ = str(row.get(typ_col) or "").strip().lower()
                    if typ and typ != "bet":
                        continue
                oid = str(row.get(oid_col) or "").strip()
                if not oid or not oid.isdigit():
                    continue
                pnl = _safe_float(row.get(pnl_col))
                if pnl is None:
                    continue
                # dt_col não é usado no total, mas mantemos a checagem para evitar formatos estranhos
                if dt_col:
                    _ = _parse_dt_any(str(row.get(dt_col) or ""))
                out[oid] = float(out.get(oid) or 0.0) + float(pnl)
    except Exception:
        return out
    return out


def _acct_pnl_like_by_order_total_from_balance_csv(path: Path) -> Dict[str, float]:
    """
    Retorna: order_id -> pnl_total (soma de amount) no balance.csv, incluindo todos os tipos "P&L-like".
    Exclui depósitos/saques/transferências/top-ups/pagamentos/ajustes/bonus.

    Usado para:
    - atribuição por **dia de execução** (created_at UTC no executor_jsonl)
    - detecção robusta de "void/push-like" (pnl≈0 por ordem) independente do `type`
    """
    out: Dict[str, float] = {}
    if not path.exists():
        return out
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            r = csv.DictReader(f)
            cols = list(r.fieldnames or [])
            if not cols:
                return out
            dt_col = _pick_col(cols, ("post date", "post_date", "date", "settled", "closed", "time"))
            pnl_col = _pick_col(cols, ("amount", "profit_loss", "profit", "p&l", "pnl", "net", "pl"))
            typ_col = _pick_col(cols, ("type",))
            oid_col = _pick_col(cols, ("order_id", "order id", "order", "bet id", "bet_id", "id"))
            if not oid_col or not pnl_col:
                return out

            def _excl_type(tl: str) -> bool:
                t = str(tl or "").strip().lower()
                return any(k in t for k in ("deposit", "withdraw", "transfer", "top", "payment", "adjust", "bonus"))

            for row in r:
                if not isinstance(row, dict):
                    continue
                oid = str(row.get(oid_col) or "").strip()
                if not oid or not oid.isdigit():
                    continue
                pnl = _safe_float(row.get(pnl_col))
                if pnl is None:
                    continue
                if typ_col:
                    tl = str(row.get(typ_col) or "").strip().lower()
                    if tl and _excl_type(tl):
                        continue
                # dt_col não é usado no total; mantemos parse para evitar formatos estranhos
                if dt_col:
                    _ = _parse_dt_any(str(row.get(dt_col) or ""))
                out[oid] = float(out.get(oid) or 0.0) + float(pnl)
    except Exception:
        return out
    return out


def _exec_day_split_back_pre_in_from_order_pnls(
    *,
    exec_by_oid_back: Dict[str, Dict[str, Any]],
    acct_pnl_by_oid_total: Dict[str, float],
    day_exec_utc: str,
    audit_by_id: Optional[Dict[int, Dict[str, Any]]] = None,
    pnl_zero_eps: float = 1e-9,
) -> Dict[str, Any]:
    """
    Agrega P&L do accounting por **dia de execução** (created_at UTC do executor_jsonl), split Pre vs In.
    - exec_by_oid_back: retorna apenas Back LIVE_OK, com `created_at` e `audit_id`
      (atenção: o flag `is_live` do executor é "modo LIVE", não "in-play")
    - acct_pnl_by_oid_total: P&L total por ordem no ledger (tipos P&L-like)

    Também retorna contagem de ordens "void/push-like" (|pnl|<=eps) como diagnóstico.
    """
    dayk = str(day_exec_utc or "").strip()
    if not dayk:
        return {
            "pnl_pre": None,
            "pnl_in": None,
            "pnl_total": None,
            "n_pre": 0,
            "n_in": 0,
            "n_total": 0,
            "coverage_n_pct": None,
            "n_pnl_zero": 0,
            "pnl_zero_eps": float(pnl_zero_eps),
        }

    pnl_pre = pnl_in = pnl_tot = 0.0
    exp_pre = exp_in = exp_tot = 0.0
    n_pre = n_in = n_tot = 0
    n_exec_day = 0
    n_zero = 0

    for oid, em in (exec_by_oid_back or {}).items():
        if not isinstance(em, dict):
            continue
        dt = em.get("created_at")
        if not isinstance(dt, datetime):
            continue
        d = dt.astimezone(timezone.utc).date().isoformat()
        if d != dayk:
            continue
        n_exec_day += 1
        if str(oid) not in acct_pnl_by_oid_total:
            continue
        try:
            pnl = float(acct_pnl_by_oid_total.get(str(oid)) or 0.0)
        except Exception:
            continue
        try:
            exp = float(em.get("exposure") or 0.0) if em.get("exposure") is not None else 0.0
        except Exception:
            exp = 0.0
        if abs(float(pnl)) <= float(pnl_zero_eps):
            n_zero += 1
        try:
            aid = em.get("audit_id")
            arow = (audit_by_id or {}).get(int(aid)) if (aid is not None and audit_by_id is not None) else None
            is_in = bool(_is_inplay_from_audit_row(arow, exec_created_at_utc=dt))
        except Exception:
            is_in = False
        if bool(is_in):
            pnl_in += pnl
            exp_in += exp
            n_in += 1
        else:
            pnl_pre += pnl
            exp_pre += exp
            n_pre += 1
        pnl_tot += pnl
        exp_tot += exp
        n_tot += 1

    try:
        denom = int(n_exec_day)
        cov = (float(n_tot) / float(denom) * 100.0) if denom > 0 else None
    except Exception:
        cov = None

    return {
        "pnl_pre": pnl_pre,
        "pnl_in": pnl_in,
        "pnl_total": pnl_tot,
        "exp_pre": exp_pre,
        "exp_in": exp_in,
        "exp_total": exp_tot,
        "n_pre": int(n_pre),
        "n_in": int(n_in),
        "n_total": int(n_tot),
        "n_exec_day": int(n_exec_day),
        "coverage_n_pct": cov,
        "n_pnl_zero": int(n_zero),
        "pnl_zero_eps": float(pnl_zero_eps),
    }


def _counterfactual_rows_for_exec_day(
    *,
    exec_by_oid_back: Dict[str, Dict[str, Any]],
    acct_pnl_by_oid_total: Dict[str, float],
    day_exec_utc: str,
    only_inplay: bool,
    audit_by_id: Optional[Dict[int, Dict[str, Any]]] = None,
    pnl_zero_eps: float = 1e-9,
) -> Dict[str, Any]:
    """
    Monta rows [{pnl, exposure, slip_raw_pct, lat_ms}] para aplicar contrafactual, bucketizando por **dia de execução**.
    Retorna também métricas de cobertura e void-like.
    """
    dayk = str(day_exec_utc or "").strip()
    rows0: list[dict] = []
    n_exec_day = 0
    n_with_acct = 0
    n_zero = 0
    for oid, em in (exec_by_oid_back or {}).items():
        if not isinstance(em, dict):
            continue
        dt = em.get("created_at")
        if not isinstance(dt, datetime):
            continue
        d = dt.astimezone(timezone.utc).date().isoformat()
        if d != dayk:
            continue
        n_exec_day += 1
        if only_inplay:
            try:
                aid = em.get("audit_id")
                arow = (audit_by_id or {}).get(int(aid)) if (aid is not None and audit_by_id is not None) else None
                is_in = bool(_is_inplay_from_audit_row(arow, exec_created_at_utc=dt))
            except Exception:
                is_in = False
            if not bool(is_in):
                continue
        if str(oid) not in acct_pnl_by_oid_total:
            continue
        try:
            pnl = float(acct_pnl_by_oid_total.get(str(oid)) or 0.0)
        except Exception:
            continue
        if abs(float(pnl)) <= float(pnl_zero_eps):
            n_zero += 1
        n_with_acct += 1
        rows0.append(
            {
                "pnl": float(pnl),
                "exposure": em.get("exposure"),
                "slip_raw_pct": em.get("slip_raw_pct"),
                "lat_ms": em.get("lat_ms"),
            }
        )
    cov = (float(n_with_acct) / float(n_exec_day) * 100.0) if n_exec_day > 0 else None
    return {
        "rows": rows0,
        "n_exec_day": int(n_exec_day),
        "n_with_acct": int(n_with_acct),
        "coverage_n_pct": cov,
        "n_pnl_zero": int(n_zero),
        "pnl_zero_eps": float(pnl_zero_eps),
    }


def _fmt_ctx_suffix(row: dict) -> str:
    """
    Sufixo opcional com contexto para interpretar ROIs extremos:
    odd_median e exposure_median (stake para Back; liability para Lay).
    """
    try:
        om = row.get("odd_median")
        em = row.get("exposure_median")
        if om is None and em is None:
            return ""
        s = []
        if om is not None:
            s.append(f"odd~{_fmt_num(om,2)}")
        if em is not None:
            s.append(f"exp~{_fmt_num(em,2)}")
        return " (" + ", ".join(s) + ")"
    except Exception:
        return ""


def _append_slippage_vs_roi_raw_section(
    out_lines: list[str],
    *,
    adh_slip: Optional[Dict[str, Any]],
    title: str,
    combo_top_limit: int = 2,
) -> None:
    """
    Renderiza o bloco "Slippage × ROI (raw, com sinal)" preservando as mesmas tabelas:
      - buckets 3-way (Back + Lay)
      - Lay bounded por stake
      - Contrafactual (placar): filtro de slippage
      - Diagnóstico AH (linha)
      - Slippage × ROI por combinação (top N por volume; acumulado)
    """
    try:
        if not isinstance(adh_slip, dict) or not adh_slip:
            return

        # slippage x ROI (3 buckets raw com sinal) — acumulado na janela (não só um dia)
        raw_total: Dict[str, Any] = {}
        try:
            raw_total = (
                adh_slip.get("slippage_vs_roi_raw_total_ctx")
                if isinstance(adh_slip.get("slippage_vs_roi_raw_total_ctx"), dict)
                else (adh_slip.get("slippage_vs_roi_raw_total") if isinstance(adh_slip.get("slippage_vs_roi_raw_total"), dict) else {})
            )
        except Exception:
            raw_total = {}
        if not isinstance(raw_total, dict) or not raw_total:
            return

        try:
            # Para slippage×ROI, respeitamos o range semântico (pós-fix) quando disponível.
            rg = adh_slip.get("slippage_range", None) if isinstance(adh_slip, dict) else None
            if not isinstance(rg, dict) or not rg:
                rg = adh_slip.get("range", {}) if isinstance(adh_slip, dict) else {}
            span = rg.get("span_days") if isinstance(rg, dict) else None
            out_lines.append(f"**{title} (range: `{rg.get('start_day')}` → `{rg.get('end_day')}`; span_days=`{int(span or 0)}`)**\n\n")
        except Exception:
            out_lines.append(f"**{title}**\n\n")

        for side_key, subtitle in (("back", "Back (ROI por stake)"), ("lay", "Lay (ROI por liability)")):
            b = raw_total.get(side_key) if isinstance(raw_total.get(side_key), dict) else {}
            buckets0 = b.get("buckets") if isinstance(b.get("buckets"), list) else []
            buckets = _slip_raw_3bucket_rows(buckets0)
            if not any(int(r.get("n") or 0) > 0 for r in buckets):
                continue
            out_lines.append(f"- **{subtitle}**\n\n")
            out_lines.append("| Bucket slippage_raw_pct | n | ROI mean (SE; IC95) |\n|---|---:|---:|\n")
            for row in buckets:
                out_lines.append(
                    f"| {row.get('bucket')} | {int(row.get('n') or 0)} | {_fmt_roi_mean_se_ci_pct(row)}{_fmt_ctx_suffix(row)} |\n"
                )
            out_lines.append("\n")

        # Back: separar Pre vs In (hipótese: latência/slippage pesa mais em in‑match)
        try:
            by_reg = adh_slip.get("slippage_vs_roi_raw_total_ctx_by_regime") if isinstance(adh_slip, dict) else None
            if isinstance(by_reg, dict) and (isinstance(by_reg.get("back_pre"), dict) or isinstance(by_reg.get("back_in"), dict)):
                for key, sub in (("back_pre", "Back Pre (ROI por stake)"), ("back_in", "Back In (ROI por stake)")):
                    blk = by_reg.get(key) if isinstance(by_reg.get(key), dict) else {}
                    buckets0 = blk.get("buckets") if isinstance(blk.get("buckets"), list) else []
                    buckets = _slip_raw_3bucket_rows(buckets0)
                    if not any(int(r.get("n") or 0) > 0 for r in buckets):
                        continue
                    out_lines.append(f"- **{sub}**\n\n")
                    out_lines.append("| Bucket slippage_raw_pct | n | ROI mean (SE; IC95) |\n|---|---:|---:|\n")
                    for row in buckets:
                        out_lines.append(f"| {row.get('bucket')} | {int(row.get('n') or 0)} | {_fmt_roi_mean_se_ci_pct(row)}{_fmt_ctx_suffix(row)} |\n")
                    out_lines.append("\n")
        except Exception:
            pass

        out_lines.append(
            "- Nota: `ROIw` é o **ROI ponderado por exposição** (peso=stake no Back; peso=liability no Lay). "
            "Em prática, dentro de um bucket, `ROIw ≈ (∑P&L)/(∑exposição)`; já o `ROI mean` é a média simples por linha/sinal.\n\n"
        )
        out_lines.append(
            "- Nota importante (reconciliação): as tabelas **Slippage × ROI** usam **somente execuções cobertas por ROI via placar** (precisa audit+placar+odd). "
            "Isso é um subconjunto e pode ter viés (ex.: jogos ainda não liquidaram, falta de odds finais, etc.). "
            "Já o **accounting ledger** inclui todo o resultado financeiro (incluindo void/refund/cancel quando existirem) por `post date`.\n\n"
        )

        # Lay também em ROI por stake (bounded; sanity-check)
        lay_stake_blk = (
            adh_slip.get("slippage_vs_roi_raw_total_ctx_lay_stake")
            if (isinstance(adh_slip, dict) and isinstance(adh_slip.get("slippage_vs_roi_raw_total_ctx_lay_stake"), dict))
            else {}
        )
        b2 = lay_stake_blk.get("lay") if isinstance(lay_stake_blk.get("lay"), dict) else {}
        buckets02 = b2.get("buckets") if isinstance(b2.get("buckets"), list) else []
        buckets2 = _slip_raw_3bucket_rows(buckets02)
        if any(int(r.get("n") or 0) > 0 for r in buckets2):
            out_lines.append("- **Lay (ROI por stake; bounded)**\n\n")
            out_lines.append("| Bucket slippage_raw_pct | n | ROI mean (SE; IC95) |\n|---|---:|---:|\n")
            for row in buckets2:
                out_lines.append(f"| {row.get('bucket')} | {int(row.get('n') or 0)} | {_fmt_roi_mean_se_ci_pct(row)}{_fmt_ctx_suffix(row)} |\n")
            out_lines.append("\n")

        # Contrafactual: filtro de slippage (placar)
        try:
            cf = adh_slip.get("slippage_filter_counterfactual") if isinstance(adh_slip, dict) else None
            if isinstance(cf, dict) and isinstance(cf.get("rule"), dict):
                b = cf.get("back") if isinstance(cf.get("back"), dict) else {}
                l = cf.get("lay") if isinstance(cf.get("lay"), dict) else {}
                if (int(b.get("n") or 0) + int(l.get("n") or 0)) > 0:
                    out_lines.append("**Contrafactual (placar): aplicar filtro de slippage**\n\n")
                    out_lines.append("- Regra: **Back** pula `slippage_raw_pct <= -2%`; **Lay** pula `slippage_raw_pct > 2%`.\n")
                    out_lines.append("- Observação: usa somente execuções com ROI via placar; não é o P&L do accounting.\n\n")
                    out_lines.append("| Lado | n (base) | P&L (base) | Exposição (base) | n (após filtro) | P&L (após) | Exposição (após) |\n")
                    out_lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
                    out_lines.append(
                        f"| Back | {int(b.get('n') or 0)} | {_fmt_num(b.get('pnl'),2)} | {_fmt_num(b.get('stake'),2)} | {int(b.get('n_filtered') or 0)} | {_fmt_num(b.get('pnl_filtered'),2)} | {_fmt_num(b.get('stake_filtered'),2)} |\n"
                    )
                    out_lines.append(
                        f"| Lay (liab) | {int(l.get('n') or 0)} | {_fmt_num(l.get('pnl'),2)} | {_fmt_num(l.get('liability'),2)} | {int(l.get('n_filtered') or 0)} | {_fmt_num(l.get('pnl_filtered'),2)} | {_fmt_num(l.get('liability_filtered'),2)} |\n"
                    )
                    try:
                        pnl0 = float(b.get("pnl") or 0.0) + float(l.get("pnl") or 0.0)
                        pnl1 = float(b.get("pnl_filtered") or 0.0) + float(l.get("pnl_filtered") or 0.0)
                        out_lines.append(f"| **Total** | — | {_fmt_num(pnl0,2)} | — | — | {_fmt_num(pnl1,2)} | — |\n")
                    except Exception:
                        pass
                    out_lines.append("\n")
        except Exception:
            pass

        # Diagnóstico AH (linha) observado na execução
        try:
            ah = adh_slip.get("observed_ah_line_abs") if isinstance(adh_slip, dict) else None
            if isinstance(ah, dict):
                thr = ah.get("threshold")
                scope = ah.get("scope")
                allx = ah.get("all_exec") if isinstance(ah.get("all_exec"), dict) else {}
                covx = ah.get("cov_placar") if isinstance(ah.get("cov_placar"), dict) else {}
                if int(allx.get("n") or 0) > 0:
                    out_lines.append("**Diagnóstico AH (linha) observado na execução**\n\n")
                    out_lines.append(f"- Policy: `ah_max_abs_line={thr}` | `ah_scope={scope}`\n")
                    out_lines.append(
                        f"- Execuções (todas): `n={int(allx.get('n') or 0)}` | `max|line|={_fmt_num(allx.get('max_abs_line'),2)}` | `n_over={int(allx.get('n_over') or 0)}`\n"
                    )
                    out_lines.append(
                        f"- Execuções com placar/ROI: `n={int(covx.get('n') or 0)}` | `max|line|={_fmt_num(covx.get('max_abs_line'),2)}` | `n_over={int(covx.get('n_over') or 0)}`\n\n"
                    )
        except Exception:
            pass

        # Por combinação (top por volume)
        rows = adh_slip.get("slippage_vs_roi_raw_by_combo_top") if (isinstance(adh_slip, dict) and isinstance(adh_slip.get("slippage_vs_roi_raw_by_combo_top"), list)) else []
        if rows:
            try:
                back_rows = [r for r in rows if isinstance(r, dict) and str(r.get("side")) == "Back"]
                lay_rows = [r for r in rows if isinstance(r, dict) and str(r.get("side")) == "Lay"]

                def _print_combo_block(title2: str, xs: list[dict], limit: int) -> None:
                    if not xs:
                        return
                    out_lines.append(f"**Slippage × ROI por combinação (top {min(limit, len(xs))} por volume; acumulado)**\n\n")
                    out_lines.append(f"- **{title2}**\n\n")
                    out_lines.append("| Combinação | n | ROI<=-2% | n | ROI(-2..2] | n | ROI>2% | n | corr(slip_raw,ROI) |\n")
                    out_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
                    for r in xs[:limit]:
                        comb = str(r.get("comb") or "")
                        n = int(r.get("n") or 0)
                        corr = r.get("corr_raw_pct_vs_roi")
                        bmap = {str(b.get("bucket")): b for b in (r.get("buckets") or []) if isinstance(b, dict)}

                        def _bn(lab: str) -> tuple[int, Any]:
                            bb = bmap.get(lab) or {}
                            return int(bb.get("n") or 0), bb

                        n1, roi1 = _bn("<= -2%")
                        n2, roi2 = _bn("(-2, 2]")
                        n3, roi3 = _bn("> 2%")
                        out_lines.append(
                            f"| {comb} | {n} | {_fmt_roi_mean_se_ci_pct(roi1)} | {n1} | {_fmt_roi_mean_se_ci_pct(roi2)} | {n2} | {_fmt_roi_mean_se_ci_pct(roi3)} | {n3} | {_fmt_num(corr,2)} |\n"
                        )
                    out_lines.append("\n")

                _print_combo_block("Back", back_rows, int(combo_top_limit))
                _print_combo_block("Lay", lay_rows, int(combo_top_limit))
            except Exception:
                pass
    except Exception:
        return


def _append_latency_vs_roi_back_pre_in_section(out_lines: list[str], *, adh_slip: Optional[Dict[str, Any]], title: str) -> None:
    """
    Renderiza "Latência × ROI (Back Pre/In)" preservando o padrão de tabelas:
      - buckets com n + ROI mean (SE; IC95) + ROIw + contexto (odd/exposure)
    """
    try:
        if not isinstance(adh_slip, dict) or not adh_slip:
            return
        blk = adh_slip.get("latency_vs_roi_call_to_done_ms") if isinstance(adh_slip.get("latency_vs_roi_call_to_done_ms"), dict) else None
        if not isinstance(blk, dict) or not blk:
            return

        out_lines.append(f"**{title}**\n\n")
        note = str(blk.get("note") or "").strip()
        if note:
            out_lines.append(f"- {note}\n\n")

        def _print(subkey: str, subtitle: str) -> None:
            sub = blk.get(subkey) if isinstance(blk.get(subkey), dict) else {}
            buckets0 = sub.get("buckets") if isinstance(sub.get("buckets"), list) else []
            if not any(isinstance(r, dict) and int(r.get("n") or 0) > 0 for r in buckets0):
                return
            out_lines.append(f"- **{subtitle}**\n\n")
            out_lines.append("| Bucket call_to_done_ms | n | ROI mean (SE; IC95) |\n|---|---:|---:|\n")
            for r in buckets0:
                if not isinstance(r, dict):
                    continue
                out_lines.append(
                    f"| {r.get('bucket')} | {int(r.get('n') or 0)} | {_fmt_roi_mean_se_ci_pct(r)}{_fmt_ctx_suffix(r)} |\n"
                )
            out_lines.append("\n")

        _print("back_pre", "Back Pre (ROI por stake)")
        _print("back_in", "Back In (ROI por stake)")
    except Exception:
        return


def _append_slippage_vs_latency_back_pre_in_section(out_lines: list[str], *, adh_slip: Optional[Dict[str, Any]], title: str) -> None:
    """
    Renderiza "Slippage × Latência (Back Pre/In)" usando o agregado exportado por `oos_adherence_report`.
    Mantém o padrão de buckets e inclui estatísticas de slippage_raw_pct e ROIw por bucket.
    """
    try:
        if not isinstance(adh_slip, dict) or not adh_slip:
            return
        blk = adh_slip.get("slippage_vs_latency_call_to_done_ms") if isinstance(adh_slip.get("slippage_vs_latency_call_to_done_ms"), dict) else None
        if not isinstance(blk, dict) or not blk:
            return
        out_lines.append(f"**{title}**\n\n")
        note = str(blk.get("note") or "").strip()
        if note:
            out_lines.append(f"- {note}\n\n")

        def _print(subkey: str, subtitle: str) -> None:
            sub = blk.get(subkey) if isinstance(blk.get(subkey), dict) else {}
            buckets0 = sub.get("buckets") if isinstance(sub.get("buckets"), list) else []
            if not any(isinstance(r, dict) and int(r.get("n") or 0) > 0 for r in buckets0):
                return
            out_lines.append(f"- **{subtitle}**\n\n")
            out_lines.append("| Bucket call_to_done_ms | n | Slippage_raw mean (SE; IC95) | Slippage_raw mediana | Slippage_raw_w (por exp.) | ROIw (por exp.) |\n")
            out_lines.append("|---|---:|---:|---:|---:|---:|\n")
            for r in buckets0:
                if not isinstance(r, dict):
                    continue
                ci = r.get("slip_raw_ci95") if isinstance(r.get("slip_raw_ci95"), dict) else None
                se = r.get("slip_raw_se")
                mean = r.get("slip_raw_mean")
                if mean is None:
                    mean_s = "—"
                else:
                    se_s = _fmt_pct(se) if se is not None else "—"
                    if ci and (ci.get("lb") is not None) and (ci.get("ub") is not None):
                        mean_s = f"{_fmt_pct(mean)} (SE {se_s}) [{_fmt_pct(ci.get('lb'))}, {_fmt_pct(ci.get('ub'))}]"
                    else:
                        mean_s = f"{_fmt_pct(mean)} (SE {se_s})"
                out_lines.append(
                    f"| {r.get('bucket')} | {int(r.get('n') or 0)} | {mean_s} | {_fmt_pct(r.get('slip_raw_median'))} | {_fmt_pct(r.get('slip_raw_weighted'))} | {_fmt_pct(r.get('roi_weighted'))} |\n"
                )
            out_lines.append("\n")

        _print("back_pre", "Back Pre (slippage_raw_pct por stake)")
        _print("back_in", "Back In (slippage_raw_pct por stake)")
    except Exception:
        return


def _demote_h2_to_h3(md: str) -> str:
    # Usado para "embrulhar" o bloco in-sample sem reescrever o conteúdo.
    out = []
    for ln in (md or "").splitlines(True):
        if ln.startswith("## "):
            out.append("### " + ln[3:])
        else:
            out.append(ln)
    return "".join(out)


def _split_base_into_insample_and_oos(md: str) -> tuple[str, str]:
    """
    O relatório robusto pode escrever o bloco OOS no topo-nível como:
      - '## 12) OOS walk-forward ...' (modo "full")
      - '## 1) OOS walk-forward ...'  (modo "oos_first")
    Tudo antes disso é o bloco in-sample.
    """
    txt = md or ""
    keys = ["## 12) OOS walk-forward", "## 1) OOS walk-forward", "## 2) OOS walk-forward"]
    hits = [(txt.find(k), k) for k in keys if txt.find(k) >= 0]
    if not hits:
        # fallback: não encontrou; trata tudo como in-sample
        return txt, ""
    i, _ = sorted(hits, key=lambda x: x[0])[0]
    return txt[:i], txt[i:]


def _extract_md_block(md: str, *, start: str, until_any: list[str]) -> str:
    """
    Extrai um trecho de markdown começando em `start` até antes do primeiro marcador em `until_any`.
    Best-effort: se não achar `start`, retorna "".
    """
    txt = md or ""
    i = txt.find(start)
    if i < 0:
        return ""
    j = None
    for u in until_any:
        k = txt.find(u, i + len(start))
        if k >= 0:
            j = k if j is None else min(j, k)
    return txt[i : (j if j is not None else len(txt))].strip() + "\n"


def _extract_md_table(md: str, *, header_startswith: str) -> tuple[str, list[list[str]]]:
    """
    Extrai uma tabela markdown cujo header começa com `header_startswith` (linha iniciando com '| ...').
    Retorna (table_md, rows) onde rows são as linhas de dados já separadas em colunas (sem pipes).
    """
    txt = md or ""
    lines = txt.splitlines()
    i = None
    for idx, ln in enumerate(lines):
        if ln.strip().startswith(header_startswith):
            i = idx
            break
    if i is None:
        return "", []
    # coletar até a primeira linha vazia após começar
    out_lines = []
    rows: list[list[str]] = []
    for ln in lines[i:]:
        if not ln.strip():
            break
        if not ln.strip().startswith("|"):
            break
        out_lines.append(ln)
        # data row (skip separator)
        if ln.strip().startswith("|---"):
            continue
        cols = [c.strip() for c in ln.strip().strip("|").split("|")]
        # pula header
        if cols and cols[0].lower().startswith("train window"):
            continue
        rows.append(cols)
    return "\n".join(out_lines).strip() + "\n", rows


def _md_table_header_cols(table_md: str) -> list[str]:
    """
    Retorna as colunas do header da tabela (linha 1) sem pipes.
    """
    try:
        for ln in (table_md or "").splitlines():
            s = ln.strip()
            if not s.startswith("|"):
                continue
            if s.startswith("|---"):
                continue
            cols = [c.strip() for c in s.strip().strip("|").split("|")]
            return cols
    except Exception:
        return []


def _parse_md_number(x: Any) -> Optional[float]:
    """
    Parser robusto para números vindos de Markdown (OOS / tabelas no PDF):
    - aceita en-US (1,234.56) e pt-BR (1.234,56)
    - preserva decimais
    - aceita percentuais ("49.54%")
    """
    try:
        t = str(x or "").strip().replace("−", "-")
        if not t:
            return None
        t = t.replace("%", "").strip()
        t = t.replace(" ", "")
        if "." in t and "," in t:
            # decide separador decimal pelo último
            if t.rfind(".") > t.rfind(","):
                t = t.replace(",", "")
            else:
                t = t.replace(".", "").replace(",", ".")
        else:
            if "," in t and "." not in t:
                t = t.replace(",", ".")
        return float(t)
    except Exception:
        return None
    return None

def _tail_lines(path: Path, n: int) -> list[str]:
    try:
        xs = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return xs[-n:] if n > 0 else xs
    except Exception:
        return []

def _week_start_iso(day_iso: str) -> Optional[str]:
    try:
        from datetime import date as _date, timedelta as _td

        d = _date.fromisoformat(str(day_iso))
        ws = d - _td(days=int(d.weekday()))
        return ws.isoformat()
    except Exception:
        return None


def _month_key(day_iso: str) -> Optional[str]:
    try:
        from datetime import date as _date

        d = _date.fromisoformat(str(day_iso))
        return f"{d.year:04d}-{d.month:02d}"
    except Exception:
        return None


def _agg_by_week(pnls_by_day: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for d, v in pnls_by_day.items():
        ws = _week_start_iso(d)
        if not ws:
            continue
        out[ws] = float(out.get(ws, 0.0)) + float(v or 0.0)
    return dict(sorted(out.items()))


def _agg_by_month(pnls_by_day: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for d, v in pnls_by_day.items():
        mk = _month_key(d)
        if not mk:
            continue
        out[mk] = float(out.get(mk, 0.0)) + float(v or 0.0)
    return dict(sorted(out.items()))


def _max_drawdown(pnls_by_day: Dict[str, float]) -> Dict[str, Any]:
    """
    Max drawdown em unidade monetária, usando curva de equity = cumsum(P&L diário).
    """
    days = sorted([d for d in pnls_by_day.keys() if str(d)])
    eq = 0.0
    peak = 0.0
    mdd = 0.0
    mdd_from = None
    mdd_to = None
    peak_day = None
    for d in days:
        eq += float(pnls_by_day.get(d) or 0.0)
        if eq >= peak:
            peak = eq
            peak_day = d
        dd = peak - eq
        if dd > mdd:
            mdd = dd
            mdd_from = peak_day
            mdd_to = d
    return {"mdd": float(mdd), "from_day": mdd_from, "to_day": mdd_to}


def _sharpe_annualized(pnls_by_day: Dict[str, float], *, bankroll_ref: float) -> Optional[float]:
    """
    Sharpe anualizado (sqrt(252)) usando retornos diários r = pnl / bankroll_ref.
    """
    try:
        br = float(bankroll_ref)
        if br <= 0:
            return None
        rs = [float(v) / br for _, v in sorted(pnls_by_day.items())]
        if len(rs) < 5:
            return None
        import statistics
        import math

        mu = statistics.fmean(rs)
        sd = statistics.pstdev(rs)
        if sd <= 0:
            return None
        return float((mu / sd) * math.sqrt(252.0))
    except Exception:
        return None

def _read_jsonl_last(path: Path, last: int) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if last > 0 and len(lines) > last:
        return lines[-last:]
    return lines


def _parse_iso_dt(s: str) -> Optional[datetime]:
    try:
        t = str(s or "").strip()
        if not t:
            return None
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _executor_gaps_summary(lines: list[str]) -> Dict[str, Any]:
    """
    Sumário simples de "downtime" por gaps no JSONL do executor.
    Interpretação: gaps grandes sugerem paradas/restarts ou ausência de tráfego.
    """
    ts = []
    for ln in lines or []:
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        res = obj.get("result") if isinstance(obj.get("result"), dict) else {}
        req = obj.get("request") if isinstance(obj.get("request"), dict) else {}
        dt = _parse_iso_dt(str(res.get("created_at") or req.get("created_at") or ""))
        if dt:
            ts.append(dt)
    ts.sort()
    if len(ts) < 2:
        return {"n": len(ts), "max_gap_s": None, "gaps_gt_300s": 0, "gaps_gt_900s": 0}
    gaps = [(ts[i] - ts[i - 1]).total_seconds() for i in range(1, len(ts))]
    return {
        "n": int(len(ts)),
        "first_ts": ts[0].isoformat(),
        "last_ts": ts[-1].isoformat(),
        "max_gap_s": float(max(gaps)) if gaps else None,
        "gaps_gt_300s": int(sum(1 for g in gaps if g > 300.0)),
        "gaps_gt_900s": int(sum(1 for g in gaps if g > 900.0)),
    }


def _filter_executor_jsonl_lines_window(lines: list[str], *, since_utc: datetime, until_utc: Optional[datetime] = None) -> list[str]:
    """
    Filtra linhas do executor_jsonl por timestamp (created_at/finished_at) para aproximar "últimas 24h".
    Observação: JSONL não é heartbeat; se o pipeline ficou sem tráfego, a janela pode retornar N baixo.
    """
    until = until_utc or datetime.now(timezone.utc)
    out: list[str] = []
    for ln in lines or []:
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        res = obj.get("result") if isinstance(obj.get("result"), dict) else {}
        req = obj.get("request") if isinstance(obj.get("request"), dict) else {}
        dt = _parse_iso_dt(str(res.get("created_at") or res.get("finished_at") or req.get("created_at") or ""))
        if not dt:
            continue
        if since_utc <= dt <= until:
            out.append(ln)
    return out


def _executor_post_accept_failures_24h(lines_24h: list[str]) -> Dict[str, Any]:
    """
    Diagnóstico pós-accepted no executor (janela 24h, a partir do JSONL):
    - accepted: `result.status in {LIVE_OK, API_FAILED, NO_SESSION, RATE_LIMIT, CAP_BLOCKED}`
      (i.e., requisições que passaram da etapa de enfileiramento e geraram resultado de execução)
    - separa por fase:
      - precheck_fail: erro antes do place_order (sinalizado por LIVE_PRECHECK_FAILED)
      - place_fail: erro no place_order (LIVE_PLACE_FAILED)
    """
    out: Dict[str, Any] = {
        "accepted_n": 0,
        "live_ok_n": 0,
        "accepted_fail_n": 0,
        "precheck_fail_n": 0,
        "place_fail_n": 0,
        "api_failed_n": 0,
        "no_session_n": 0,
        "rate_limit_n": 0,
        "cap_blocked_n": 0,
        "no_pmms_n": 0,
        "ctx_destroyed_n": 0,
        "auth_401_n": 0,
        "ws_stale_n": 0,
        "precheck_pmm_wait_ms_p50": None,
        "precheck_pmm_wait_ms_p90": None,
        "precheck_ws_age_ms_p50": None,
        "precheck_ws_age_ms_p90": None,
        "top_errors": [],
    }
    try:
        accepted_status = {"LIVE_OK", "API_FAILED", "NO_SESSION", "RATE_LIMIT", "CAP_BLOCKED"}
        err_counts: Dict[str, int] = defaultdict(int)
        pmm_wait_vals: list[float] = []
        ws_age_vals: list[float] = []

        for ln in lines_24h or []:
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            res = obj.get("result") if isinstance(obj.get("result"), dict) else {}
            if not isinstance(res, dict):
                continue
            st = str(res.get("status") or "").upper().strip()
            if st == "HEARTBEAT" or st not in accepted_status:
                continue

            out["accepted_n"] = int(out["accepted_n"]) + 1
            if st == "LIVE_OK":
                out["live_ok_n"] = int(out["live_ok_n"]) + 1
                continue

            out["accepted_fail_n"] = int(out["accepted_fail_n"]) + 1
            if st == "API_FAILED":
                out["api_failed_n"] = int(out["api_failed_n"]) + 1
            elif st == "NO_SESSION":
                out["no_session_n"] = int(out["no_session_n"]) + 1
            elif st == "RATE_LIMIT":
                out["rate_limit_n"] = int(out["rate_limit_n"]) + 1
            elif st == "CAP_BLOCKED":
                out["cap_blocked_n"] = int(out["cap_blocked_n"]) + 1

            err = str(res.get("error") or "").strip()
            err_low = err.lower()
            if err:
                err_counts[err[:180]] += 1

            raw = res.get("raw") if isinstance(res.get("raw"), dict) else {}
            if "live_precheck_failed" in err_low:
                out["precheck_fail_n"] = int(out["precheck_fail_n"]) + 1
                try:
                    t = raw.get("timing_breakdown") if isinstance(raw.get("timing_breakdown"), dict) else {}
                    pmmw = _safe_float(t.get("pmm_wait_ms"))
                    if pmmw is not None:
                        pmm_wait_vals.append(float(pmmw))
                except Exception:
                    pass
                try:
                    ws_age = _safe_float(raw.get("ws_age_ms"))
                    if ws_age is not None:
                        ws_age_vals.append(float(ws_age))
                except Exception:
                    pass
            if "live_place_failed" in err_low:
                out["place_fail_n"] = int(out["place_fail_n"]) + 1

            if "no pmms received" in err_low:
                out["no_pmms_n"] = int(out["no_pmms_n"]) + 1
            if ("execution context was destroyed" in err_low) or ("target closed" in err_low):
                out["ctx_destroyed_n"] = int(out["ctx_destroyed_n"]) + 1
            if ("http_401" in err_low) or ("auth_error" in err_low) or ("no_root_session_cookie" in err_low):
                out["auth_401_n"] = int(out["auth_401_n"]) + 1
            if "ws_age_ms=" in err_low or "ws stale" in err_low:
                out["ws_stale_n"] = int(out["ws_stale_n"]) + 1

        # percentis simples (numpy-free)
        def _pct(xs: list[float], p: float) -> Optional[float]:
            if not xs:
                return None
            ys = sorted(float(x) for x in xs)
            if len(ys) == 1:
                return float(ys[0])
            k = int(round((len(ys) - 1) * float(p)))
            k = max(0, min(len(ys) - 1, k))
            return float(ys[k])

        out["precheck_pmm_wait_ms_p50"] = _pct(pmm_wait_vals, 0.50)
        out["precheck_pmm_wait_ms_p90"] = _pct(pmm_wait_vals, 0.90)
        out["precheck_ws_age_ms_p50"] = _pct(ws_age_vals, 0.50)
        out["precheck_ws_age_ms_p90"] = _pct(ws_age_vals, 0.90)

        tops = sorted(err_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
        out["top_errors"] = [{"error": k, "n": int(v)} for k, v in tops]
    except Exception:
        return out
    return out


def _extract_audit_ids_from_exec_lines(lines: list[str]) -> list[int]:
    out: set[int] = set()
    for ln in lines or []:
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        req = obj.get("request") if isinstance(obj.get("request"), dict) else {}
        res = obj.get("result") if isinstance(obj.get("result"), dict) else {}
        aid = _safe_int_or_none(
            res.get("audit_id") if res.get("audit_id") is not None else req.get("audit_id")
        )
        if aid is not None and int(aid) > 0:
            out.add(int(aid))
    return sorted(out)


def _ms_stats(xs: list[float]) -> Dict[str, Any]:
    if not xs:
        return {"n": 0, "p50": None, "p90": None, "p99": None, "mean": None}
    ys = sorted(float(x) for x in xs if x is not None)
    if not ys:
        return {"n": 0, "p50": None, "p90": None, "p99": None, "mean": None}

    def _q(p: float) -> Optional[float]:
        if not ys:
            return None
        if len(ys) == 1:
            return float(ys[0])
        k = int(round((len(ys) - 1) * float(p)))
        k = max(0, min(len(ys) - 1, k))
        return float(ys[k])

    return {
        "n": int(len(ys)),
        "p50": _q(0.50),
        "p90": _q(0.90),
        "p99": _q(0.99),
        "mean": float(sum(ys) / float(len(ys))),
    }


def _executor_e2e_latency_24h(lines_24h: list[str], audit_by_id: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Latência ponta a ponta em 24h:
      WS detectado (hypothesis_detected_at) -> executor finished_at.

    Breakdown principal:
      1) detect_to_submit_ms   : detect -> request.created_at (bridge submit)
      2) audit_total_ms        : detect -> fim do audit (quando disponível no DB)
      3) bridge_wait_ms        : (detect->submit) - audit_total_ms
      4) executor_submit_to_done_ms : submit -> finished_at (call_to_done efetivo)
      5) e2e_total_ms          : detect -> finished_at
    """
    out: Dict[str, Any] = {
        "n_jsonl_24h": int(len(lines_24h or [])),
        "n_with_audit_id": 0,
        "n_with_detected_at": 0,
        "n_e2e_all": 0,
        "n_e2e_success": 0,
        "ok_statuses": ["LIVE_OK", "DRY_OK"],
        "all": {},
        "success": {},
    }
    metrics_all: Dict[str, list[float]] = defaultdict(list)
    metrics_ok: Dict[str, list[float]] = defaultdict(list)
    ok_status = {"LIVE_OK", "DRY_OK"}

    def _audit_telemetry(row: Dict[str, Any]) -> Dict[str, Any]:
        try:
            raw = row.get("hypothesis_details")
            if isinstance(raw, str):
                raw = json.loads(raw)
            if not isinstance(raw, dict):
                return {}
            t = raw.get("telemetry")
            return t if isinstance(t, dict) else {}
        except Exception:
            return {}

    for ln in lines_24h or []:
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        req = obj.get("request") if isinstance(obj.get("request"), dict) else {}
        res = obj.get("result") if isinstance(obj.get("result"), dict) else {}
        if not isinstance(res, dict):
            continue
        st = str(res.get("status") or "").upper().strip()
        if st == "HEARTBEAT":
            continue

        aid = _safe_int_or_none(
            res.get("audit_id") if res.get("audit_id") is not None else req.get("audit_id")
        )
        if aid is None or int(aid) <= 0:
            continue
        out["n_with_audit_id"] = int(out["n_with_audit_id"]) + 1

        a = audit_by_id.get(int(aid)) if isinstance(audit_by_id, dict) else None
        if not isinstance(a, dict):
            continue

        det = _parse_iso_dt_best(a.get("hypothesis_detected_at"))
        if not isinstance(det, datetime):
            continue
        out["n_with_detected_at"] = int(out["n_with_detected_at"]) + 1

        req_created = _parse_iso_dt_best(req.get("created_at") or res.get("created_at"))
        fin = _parse_iso_dt_best(res.get("finished_at") or res.get("created_at"))
        if not isinstance(req_created, datetime) or not isinstance(fin, datetime):
            continue
        if fin < det:
            continue

        detect_to_submit_ms = max(0.0, (req_created - det).total_seconds() * 1000.0)
        submit_to_done_ms = _safe_float(
            ((res.get("timing") or {}).get("call_to_done_ms"))
            if isinstance(res.get("timing"), dict)
            else None
        )
        if submit_to_done_ms is None:
            submit_to_done_ms = max(0.0, (fin - req_created).total_seconds() * 1000.0)
        e2e_total_ms = max(0.0, (fin - det).total_seconds() * 1000.0)

        audit_total_ms = _safe_float(a.get("audit_total_duration_ms"))
        audit_det_click_ms = _safe_float(a.get("lag_detection_to_click_ms"))
        audit_click_bs_ms = _safe_float(a.get("lag_click_to_betslip_ms"))
        tele = _audit_telemetry(a)
        gate_wait_s = _safe_float(tele.get("gate_wait_s"))
        gate_wait_ms = (float(gate_wait_s) * 1000.0) if gate_wait_s is not None else None
        bridge_wait_ms = (
            max(0.0, float(detect_to_submit_ms) - float(audit_total_ms))
            if audit_total_ms is not None
            else None
        )

        t = res.get("timing") if isinstance(res.get("timing"), dict) else {}
        queue_delay_ms = _safe_float(t.get("queue_delay_ms")) if isinstance(t, dict) else None
        post_ms = _safe_float(t.get("post_ms")) if isinstance(t, dict) else None
        total_api_ms = _safe_float(t.get("total_ms")) if isinstance(t, dict) else None

        vals = {
            "e2e_total_ms": e2e_total_ms,
            "detect_to_submit_ms": detect_to_submit_ms,
            "audit_total_ms": audit_total_ms,
            "audit_detect_to_click_ms": audit_det_click_ms,
            "audit_click_to_betslip_ms": audit_click_bs_ms,
            "audit_queue_wait_ms": _safe_float(tele.get("queue_wait_ms")),
            "audit_parallel_fetch_ms": _safe_float(tele.get("parallel_fetch_ms")),
            "audit_temporal_total_ms": _safe_float(tele.get("temporal_total_ms")),
            "audit_execution_ms": _safe_float(tele.get("execution_ms")),
            "audit_pipeline_overhead_ms": _safe_float(tele.get("pipeline_overhead_ms")),
            "audit_db_save_ms": _safe_float(tele.get("db_save_ms")),
            "audit_gate_wait_ms": gate_wait_ms,
            "bridge_wait_ms": bridge_wait_ms,
            "executor_submit_to_done_ms": submit_to_done_ms,
            "executor_queue_delay_ms": queue_delay_ms,
            "executor_post_ms": post_ms,
            "executor_total_api_ms": total_api_ms,
        }
        out["n_e2e_all"] = int(out["n_e2e_all"]) + 1
        for k, v in vals.items():
            if v is not None:
                metrics_all[k].append(float(v))
        if st in ok_status:
            out["n_e2e_success"] = int(out["n_e2e_success"]) + 1
            for k, v in vals.items():
                if v is not None:
                    metrics_ok[k].append(float(v))

    keys = [
        "e2e_total_ms",
        "detect_to_submit_ms",
        "audit_total_ms",
        "audit_detect_to_click_ms",
        "audit_click_to_betslip_ms",
        "audit_queue_wait_ms",
        "audit_parallel_fetch_ms",
        "audit_temporal_total_ms",
        "audit_execution_ms",
        "audit_pipeline_overhead_ms",
        "audit_db_save_ms",
        "audit_gate_wait_ms",
        "bridge_wait_ms",
        "executor_submit_to_done_ms",
        "executor_queue_delay_ms",
        "executor_post_ms",
        "executor_total_api_ms",
    ]
    out["all"] = {k: _ms_stats(metrics_all.get(k, [])) for k in keys}
    out["success"] = {k: _ms_stats(metrics_ok.get(k, [])) for k in keys}
    return out


def _executor_gaps_summary_window(lines: list[str], *, since_utc: datetime, until_utc: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Mesmo sumário de gaps, mas focado em uma janela (ex.: últimas 24h).

    Observação: como o JSONL é escrito apenas quando há requisição/resposta, não é um heartbeat.
    Então isso mede "silêncio" do pipeline (executor sem tráfego, audit/bridge parados, ou executor down).
    Para aproximar "tempo em silêncio", somamos (gap - 900s) para gaps>15min (proxy de downtime acima do limiar).
    """
    until = until_utc or datetime.now(timezone.utc)
    ts_all: list[datetime] = []
    for ln in lines or []:
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        res = obj.get("result") if isinstance(obj.get("result"), dict) else {}
        req = obj.get("request") if isinstance(obj.get("request"), dict) else {}
        dt = _parse_iso_dt(str(res.get("created_at") or req.get("created_at") or ""))
        if dt:
            ts_all.append(dt)
    ts_all.sort()
    if not ts_all:
        return {
            "since_utc": since_utc.isoformat(),
            "until_utc": until.isoformat(),
            "n": 0,
            "first_ts": None,
            "last_ts": None,
            "max_gap_s": None,
            "gaps_gt_300s": 0,
            "gaps_gt_900s": 0,
            "silence_over_15m_s": 0.0,
            "silence_over_15m_pct": None,
        }
    # inclui 1 ponto anterior ao since (se existir) para captar gap cruzando a borda da janela
    prev = None
    for dt in reversed(ts_all):
        if dt < since_utc:
            prev = dt
            break
    tsw = [dt for dt in ts_all if since_utc <= dt <= until]
    if prev:
        tsw = [prev] + tsw
    tsw.sort()
    if len(tsw) < 2:
        return {
            "since_utc": since_utc.isoformat(),
            "until_utc": until.isoformat(),
            "n": int(len(tsw)),
            "first_ts": tsw[0].isoformat() if tsw else None,
            "last_ts": tsw[-1].isoformat() if tsw else None,
            "max_gap_s": None,
            "gaps_gt_300s": 0,
            "gaps_gt_900s": 0,
            "silence_over_15m_s": 0.0,
            "silence_over_15m_pct": None,
        }
    gaps = [(tsw[i] - tsw[i - 1]).total_seconds() for i in range(1, len(tsw))]
    over_15 = [g for g in gaps if g > 900.0]
    silence_over = float(sum((g - 900.0) for g in over_15)) if over_15 else 0.0
    win_s = max(1.0, (until - since_utc).total_seconds())
    return {
        "since_utc": since_utc.isoformat(),
        "until_utc": until.isoformat(),
        "n": int(len(tsw)),
        "first_ts": tsw[0].isoformat(),
        "last_ts": tsw[-1].isoformat(),
        "max_gap_s": float(max(gaps)) if gaps else None,
        "gaps_gt_300s": int(sum(1 for g in gaps if g > 300.0)),
        "gaps_gt_900s": int(sum(1 for g in gaps if g > 900.0)),
        "silence_over_15m_s": float(silence_over),
        "silence_over_15m_pct": float(silence_over / win_s * 100.0) if win_s > 0 else None,
    }


def _mem_available_mib() -> Optional[float]:
    try:
        p = Path("/proc/meminfo")
        if not p.exists():
            return None
        mem_av = None
        for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            if ln.startswith("MemAvailable:"):
                parts = ln.split()
                if len(parts) >= 2:
                    mem_av = float(parts[1])  # kB
                    break
        if mem_av is None:
            return None
        return float(mem_av / 1024.0)
    except Exception:
        return None


def _vcpu_count() -> Optional[int]:
    try:
        n = os.cpu_count()
        if n is None:
            return None
        n2 = int(n)
        return n2 if n2 > 0 else None
    except Exception:
        return None


def _safe_div(a: Any, b: Any) -> Optional[float]:
    try:
        aa = float(a)
        bb = float(b)
        if bb == 0:
            return None
        return float(aa / bb)
    except Exception:
        return None


def _load_wf_policy_last_step(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        d = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(d, dict):
            return None
        steps = d.get("steps") if isinstance(d.get("steps"), list) else []
        last = steps[-1] if steps and isinstance(steps[-1], dict) else None
        return last if isinstance(last, dict) else None
    except Exception:
        return None


def _pick_prev_policy_file(policy_dir: Path, *, cur_day: str) -> Optional[Path]:
    try:
        if not policy_dir.exists():
            return None
        xs = sorted([p for p in policy_dir.glob("wf_policy_*.json") if p.is_file()])
        if not xs:
            return None
        def _day_from_name(p: Path) -> Optional[str]:
            # aceita wf_policy_YYYYMMDD.json ou wf_policy_YYYYMMDD_HHMMSS.json
            s = p.name.replace("wf_policy_", "").replace(".json", "")
            s = s.split("_", 1)[0]
            if len(s) == 8 and s.isdigit():
                return s
            return None

        # preferir o snapshot canônico do dia (sem sufixo), se existir
        by_day: Dict[str, Dict[str, Optional[Path]]] = {}
        for p in xs:
            d = _day_from_name(p)
            if not d:
                continue
            slot = by_day.setdefault(d, {"canonical": None, "fallback": None})
            if p.name == f"wf_policy_{d}.json":
                slot["canonical"] = p
            else:
                # fallback: guarda o "maior" lexicográfico do dia (normalmente o mais recente)
                cur = slot.get("fallback")
                if cur is None or p.name > cur.name:
                    slot["fallback"] = p

        prev_day = None
        for d in sorted(by_day.keys()):
            if str(d) < str(cur_day):
                prev_day = d
        if not prev_day:
            return None
        slot = by_day.get(prev_day) or {}
        return slot.get("canonical") or slot.get("fallback")
    except Exception:
        return None


def _parse_iso_dt_best(s: Any) -> Optional[datetime]:
    try:
        if s is None:
            return None
        t = str(s).strip()
        if not t:
            return None
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def _env_bool(k: str, default: str = "0") -> bool:
    v = str(os.getenv(k, default) or "").strip()
    return v in ("1", "true", "True", "yes", "YES", "on", "ON")


def _env_float(k: str, default: str) -> float:
    try:
        return float(os.getenv(k, default))
    except Exception:
        return float(default)


def _count_err_substr(audit_rep: Optional[Dict[str, Any]], needle: str) -> int:
    """
    Conta ocorrências em audit_status_kpis.error_rows cujo api_error contém `needle` (case-insensitive).
    """
    try:
        if not isinstance(audit_rep, dict):
            return 0
        xs = audit_rep.get("error_rows") or []
        if not isinstance(xs, list) or not needle:
            return 0
        nd = str(needle).lower()
        tot = 0
        for it in xs:
            if not isinstance(it, dict):
                continue
            err = str(it.get("api_error") or "").lower()
            if nd in err:
                tot += int(it.get("n") or 0)
        return int(tot)
    except Exception:
        return 0


def _sum_status(audit_rep: Optional[Dict[str, Any]], status: str) -> int:
    try:
        if not isinstance(audit_rep, dict):
            return 0
        tot = 0
        for v in audit_rep.get("by_version") or []:
            if not isinstance(v, dict):
                continue
            sc = v.get("status_counts") if isinstance(v.get("status_counts"), dict) else {}
            tot += int(sc.get(status) or 0)
        return int(tot)
    except Exception:
        return 0


def _sum_total(audit_rep: Optional[Dict[str, Any]]) -> int:
    try:
        if not isinstance(audit_rep, dict):
            return 0
        tot = 0
        for v in audit_rep.get("by_version") or []:
            if isinstance(v, dict):
                tot += int(v.get("total") or 0)
        return int(tot)
    except Exception:
        return 0


def _sum_ok_valid(audit_rep: Optional[Dict[str, Any]]) -> int:
    try:
        if not isinstance(audit_rep, dict):
            return 0
        tot = 0
        for v in audit_rep.get("by_version") or []:
            if isinstance(v, dict):
                tot += int(v.get("ok_valid") or 0)
        return int(tot)
    except Exception:
        return 0


def _sum_ok(audit_rep: Optional[Dict[str, Any]]) -> int:
    try:
        if not isinstance(audit_rep, dict):
            return 0
        tot = 0
        for v in audit_rep.get("by_version") or []:
            if isinstance(v, dict):
                sc = v.get("status_counts") if isinstance(v.get("status_counts"), dict) else {}
                tot += int(sc.get("OK") or 0)
        return int(tot)
    except Exception:
        return 0


def _fmt_status(ok: Optional[bool]) -> str:
    if ok is None:
        return "—"
    return "OK" if ok else "FAIL"


def _telegram_send_document(
    token: str,
    chat_id: str,
    *,
    file_path: Path,
    caption: str,
) -> Tuple[bool, Optional[int], str]:
    url = f"https://api.telegram.org/bot{token}/sendDocument"
    try:
        with file_path.open("rb") as f:
            files = {"document": (file_path.name, f, "application/pdf")}
            data = {"chat_id": chat_id, "caption": caption[:900]}
            r = requests.post(url, data=data, files=files, timeout=60)
            if r.ok:
                return True, int(r.status_code), ""
            return False, int(r.status_code), str(r.text or "")[:500]
    except Exception as e:
        return False, None, str(e)[:240]


def _telegram_send_message(token: str, chat_id: str, text_msg: str) -> Tuple[bool, Optional[int], str]:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": chat_id, "text": str(text_msg)[:3900]}, timeout=30)
        if r.ok:
            return True, int(r.status_code), ""
        return False, int(r.status_code), str(r.text or "")[:500]
    except Exception as e:
        return False, None, str(e)[:240]


@dataclass
class DailyReportCfg:
    out_dir: Path = Path("logs/daily_reports")
    report_tz: str = "America/Sao_Paulo"
    # Alinhar com o relatório “v38” por default
    # Default atualizado: inclui Back API moderno (v5.2) e mantém Lay gate (v5.1).
    # Isso evita OOS truncar (ex.: parar em 03/04) quando as versões antigas não têm histórico recente.
    versions: str = os.getenv("DAILY_OOS_VERSIONS", "v4.0-api,v5.2-api-back,v5.1-ws-gate-lay")
    hypothesis_type: str = os.getenv("DAILY_OOS_HYPOTHESIS_TYPE", "H3B")
    direction: str = os.getenv("DAILY_OOS_DIRECTION", "up")
    # Alinha com o relatório “atual” (ex.: 21d) se o usuário não setar nada.
    lookback_days: str = os.getenv("DAILY_OOS_LOOKBACK_DAYS", "21")
    no_auto_exclude_days: bool = (os.getenv("DAILY_NO_AUTO_EXCLUDE_DAYS", "0").strip() in ("1", "true", "True", "yes", "YES"))
    report_mode: str = os.getenv("DAILY_REPORT_MODE", "oos_first")
    wf_policy_current: Path = Path(os.getenv("DAILY_WF_POLICY_CURRENT", "logs/wf_policy_current.json"))
    wf_policy_history_dir: Path = Path(os.getenv("DAILY_WF_POLICY_HISTORY_DIR", "logs/policy_history"))
    wf_policy_history_jsonl: Path = Path(os.getenv("DAILY_WF_POLICY_HISTORY_JSONL", "logs/wf_policy_history.jsonl"))
    publish_policy_current: bool = (os.getenv("DAILY_WF_PUBLISH_CURRENT", "1").strip() in ("1", "true", "True", "yes", "YES"))
    policy_compat_guard_enable: bool = (_is_truthy(os.getenv("DAILY_WF_COMPAT_GUARD_ENABLE", "1")))
    policy_compat_fail_closed: bool = (_is_truthy(os.getenv("DAILY_WF_COMPAT_FAIL_CLOSED", "1")))
    policy_compat_min_pre_keys: int = max(1, _safe_int(os.getenv("DAILY_WF_COMPAT_MIN_PRE_KEYS", "1"), 1))
    policy_compat_bridge_exec_side: str = os.getenv("DAILY_WF_COMPAT_BRIDGE_EXEC_SIDE", os.getenv("BRIDGE_EXEC_SIDE", "Back"))
    policy_compat_prematch_only: bool = (_is_truthy(os.getenv("DAILY_WF_COMPAT_PREMATCH_ONLY", os.getenv("BRIDGE_PREMATCH_ONLY", "1"))))
    # Walk-forward knobs (para casar com versões como leaguePre / AHgatePre / expanding)
    wf_train_mode: str = os.getenv("DAILY_WF_TRAIN_MODE", "expanding")
    wf_train_days: str = os.getenv("DAILY_WF_TRAIN_DAYS", "2")
    wf_test_days: str = os.getenv("DAILY_WF_TEST_DAYS", "2")
    wf_step_days: str = os.getenv("DAILY_WF_STEP_DAYS", "2")
    wf_key_by_league: bool = (os.getenv("DAILY_WF_KEY_BY_LEAGUE", "1").strip() in ("1", "true", "True", "yes", "YES"))
    wf_key_by_league_scope: str = os.getenv("DAILY_WF_KEY_BY_LEAGUE_SCOPE", "pre")
    # Estatística exploratória no OOS (deve ficar OFF no daily 19h)
    wf_experimental_stats: bool = (os.getenv("DAILY_WF_EXPERIMENTAL_STATS", "0").strip() in ("1", "true", "True", "yes", "YES"))
    wf_ah_max_abs_line: str = os.getenv("DAILY_WF_AH_MAX_ABS_LINE", "2.0")
    wf_ah_scope: str = os.getenv("DAILY_WF_AH_SCOPE", "pre")
    wf_liquidity_mode: str = os.getenv("DAILY_WF_LIQUIDITY_MODE", "none")
    wf_liquidity_scope: str = os.getenv("DAILY_WF_LIQUIDITY_SCOPE", "pre")
    wf_min_matches: str = os.getenv("DAILY_WF_MIN_MATCHES", "0")
    wf_pre_activation_mode: str = os.getenv("DAILY_WF_PRE_ACTIVATION_MODE", "roi_clv").strip()
    wf_roi_min_activate: str = os.getenv("DAILY_WF_ROI_MIN_ACTIVATE", "0").strip()
    wf_shrinkage: bool = (os.getenv("DAILY_WF_SHRINKAGE", "1").strip() in ("1", "true", "True", "yes", "YES"))
    wf_exclude_exec_buckets_back: str = os.getenv("DAILY_WF_EXCLUDE_EXEC_BUCKETS_BACK", "10-20s")
    wf_exclude_exec_buckets_lay: str = os.getenv("DAILY_WF_EXCLUDE_EXEC_BUCKETS_LAY", "")
    wf_backpre_slip_max: str = os.getenv("DAILY_WF_BACKPRE_SLIP_MAX", "").strip()
    wf_backpre_slip_field: str = os.getenv("DAILY_WF_BACKPRE_SLIP_FIELD", "diff_pct").strip()
    wf_backpre_fast_max_lag_ms: str = os.getenv("DAILY_WF_BACKPRE_FAST_MAX_LAG_MS", "").strip()
    # Sizing no WF (útil para simular in-match governado por budget/caps, sem trocar policy do robô)
    wf_scheme_pre: str = os.getenv("DAILY_WF_SCHEME_PRE", "").strip()
    wf_scheme_in: str = os.getenv("DAILY_WF_SCHEME_IN", "").strip()
    wf_flat_stake_back: str = os.getenv("DAILY_WF_FLAT_STAKE_BACK", "").strip()
    # Importante: o default do analyzer é 1.0; para sensibilidade de banca (Lay in-match FLAT),
    # isso pode "saturar" lucro/turnover. Por default operacional, usamos 50.0 (override via env).
    wf_flat_liab_lay: str = os.getenv("DAILY_WF_FLAT_LIAB_LAY", "50").strip()
    # Budget por match_id no WF (permite rodar manual com EQ 4%/4% cap33% sem mexer no agendado das 19h)
    wf_budget_back_frac: str = os.getenv("DAILY_WF_BUDGET_BACK_FRAC", "").strip()
    wf_budget_lay_frac: str = os.getenv("DAILY_WF_BUDGET_LAY_FRAC", "").strip()
    wf_budget_cap_signal_frac: str = os.getenv("DAILY_WF_BUDGET_CAP_SIGNAL_FRAC", "").strip()
    wf_budget_risk_mode: str = os.getenv("DAILY_WF_BUDGET_RISK_MODE", "").strip()
    # Estudo rápido: sweep de caps absolutos (stake médio) no OOS, para curva lucro×cap (1D + grid 2D).
    wf_sweep_stakes: bool = (os.getenv("DAILY_WF_SWEEP_STAKES", "0").strip() in ("1", "true", "True", "yes", "YES"))
    wf_sweep_back_caps: str = os.getenv("DAILY_WF_SWEEP_BACK_CAPS", "").strip()
    wf_sweep_lay_caps: str = os.getenv("DAILY_WF_SWEEP_LAY_CAPS", "").strip()
    wf_sweep_grid_in: bool = (os.getenv("DAILY_WF_SWEEP_GRID_IN", "1").strip() in ("1", "true", "True", "yes", "YES"))
    # Escala de banca/sizing (manter “10k etc.”)
    kelly_bankroll: str = os.getenv("DAILY_KELLY_BANKROLL", "10000")
    # Grid default para sempre gerar sensibilidade (pequeno o bastante para ser barato).
    wf_bankroll_grid: str = os.getenv("DAILY_WF_BANKROLL_GRID", "10000,50000,100000,500000,1000000,1500000,3000000,5000000").strip()
    executor_jsonl: Path = Path(os.getenv("EXECUTOR_JSONL", "logs/executor_live.jsonl"))
    exec_kpi_last: int = int(os.getenv("DAILY_EXEC_KPI_LAST", "50000"))
    send_telegram: bool = (os.getenv("DAILY_REPORT_TELEGRAM", "1").strip() not in ("0", "false", "False", "no", "NO"))
    skip_accounting: bool = (os.getenv("DAILY_SKIP_ACCOUNTING", "0").strip() in ("1", "true", "True", "yes", "YES"))
    skip_oos: bool = (os.getenv("DAILY_SKIP_OOS", "0").strip() in ("1", "true", "True", "yes", "YES"))

    def __post_init__(self) -> None:
        # Releitura de env em runtime (importante quando rodando manualmente e carregando .env em main()).
        self.versions = os.getenv("DAILY_OOS_VERSIONS", self.versions)
        self.hypothesis_type = os.getenv("DAILY_OOS_HYPOTHESIS_TYPE", self.hypothesis_type)
        self.direction = os.getenv("DAILY_OOS_DIRECTION", self.direction)
        self.lookback_days = os.getenv("DAILY_OOS_LOOKBACK_DAYS", self.lookback_days)
        self.report_mode = os.getenv("DAILY_REPORT_MODE", self.report_mode)
        self.wf_policy_current = Path(os.getenv("DAILY_WF_POLICY_CURRENT", str(self.wf_policy_current)))
        self.wf_policy_history_dir = Path(os.getenv("DAILY_WF_POLICY_HISTORY_DIR", str(self.wf_policy_history_dir)))
        self.wf_policy_history_jsonl = Path(os.getenv("DAILY_WF_POLICY_HISTORY_JSONL", str(self.wf_policy_history_jsonl)))
        self.publish_policy_current = (os.getenv("DAILY_WF_PUBLISH_CURRENT", "1" if self.publish_policy_current else "0").strip() in ("1", "true", "True", "yes", "YES"))
        self.policy_compat_guard_enable = _is_truthy(os.getenv("DAILY_WF_COMPAT_GUARD_ENABLE", "1" if self.policy_compat_guard_enable else "0"))
        self.policy_compat_fail_closed = _is_truthy(os.getenv("DAILY_WF_COMPAT_FAIL_CLOSED", "1" if self.policy_compat_fail_closed else "0"))
        self.policy_compat_min_pre_keys = max(1, _safe_int(os.getenv("DAILY_WF_COMPAT_MIN_PRE_KEYS", str(self.policy_compat_min_pre_keys or 1)), 1))
        self.policy_compat_bridge_exec_side = os.getenv("DAILY_WF_COMPAT_BRIDGE_EXEC_SIDE", os.getenv("BRIDGE_EXEC_SIDE", self.policy_compat_bridge_exec_side))
        self.policy_compat_prematch_only = _is_truthy(os.getenv("DAILY_WF_COMPAT_PREMATCH_ONLY", os.getenv("BRIDGE_PREMATCH_ONLY", "1")))
        self.executor_jsonl = Path(os.getenv("EXECUTOR_JSONL", str(self.executor_jsonl)))
        self.wf_exclude_exec_buckets_back = os.getenv("DAILY_WF_EXCLUDE_EXEC_BUCKETS_BACK", self.wf_exclude_exec_buckets_back)
        self.wf_exclude_exec_buckets_lay = os.getenv("DAILY_WF_EXCLUDE_EXEC_BUCKETS_LAY", self.wf_exclude_exec_buckets_lay)
        self.wf_pre_activation_mode = os.getenv("DAILY_WF_PRE_ACTIVATION_MODE", self.wf_pre_activation_mode).strip()
        self.wf_roi_min_activate = os.getenv("DAILY_WF_ROI_MIN_ACTIVATE", self.wf_roi_min_activate).strip()
        self.wf_backpre_slip_max = os.getenv("DAILY_WF_BACKPRE_SLIP_MAX", self.wf_backpre_slip_max).strip()
        self.wf_backpre_slip_field = os.getenv("DAILY_WF_BACKPRE_SLIP_FIELD", self.wf_backpre_slip_field).strip()
        self.wf_backpre_fast_max_lag_ms = os.getenv("DAILY_WF_BACKPRE_FAST_MAX_LAG_MS", self.wf_backpre_fast_max_lag_ms).strip()
        self.wf_scheme_pre = os.getenv("DAILY_WF_SCHEME_PRE", self.wf_scheme_pre).strip()
        self.wf_scheme_in = os.getenv("DAILY_WF_SCHEME_IN", self.wf_scheme_in).strip()
        self.wf_flat_stake_back = os.getenv("DAILY_WF_FLAT_STAKE_BACK", self.wf_flat_stake_back).strip()
        self.wf_flat_liab_lay = os.getenv("DAILY_WF_FLAT_LIAB_LAY", self.wf_flat_liab_lay).strip()
        self.wf_budget_back_frac = os.getenv("DAILY_WF_BUDGET_BACK_FRAC", self.wf_budget_back_frac).strip()
        self.wf_budget_lay_frac = os.getenv("DAILY_WF_BUDGET_LAY_FRAC", self.wf_budget_lay_frac).strip()
        self.wf_budget_cap_signal_frac = os.getenv("DAILY_WF_BUDGET_CAP_SIGNAL_FRAC", self.wf_budget_cap_signal_frac).strip()
        self.wf_budget_risk_mode = os.getenv("DAILY_WF_BUDGET_RISK_MODE", self.wf_budget_risk_mode).strip()
        self.wf_sweep_stakes = (os.getenv("DAILY_WF_SWEEP_STAKES", "1" if self.wf_sweep_stakes else "0").strip() in ("1", "true", "True", "yes", "YES"))
        self.wf_sweep_back_caps = os.getenv("DAILY_WF_SWEEP_BACK_CAPS", self.wf_sweep_back_caps).strip()
        self.wf_sweep_lay_caps = os.getenv("DAILY_WF_SWEEP_LAY_CAPS", self.wf_sweep_lay_caps).strip()
        self.wf_sweep_grid_in = (os.getenv("DAILY_WF_SWEEP_GRID_IN", "1" if self.wf_sweep_grid_in else "0").strip() in ("1", "true", "True", "yes", "YES"))
        try:
            self.exec_kpi_last = int(os.getenv("DAILY_EXEC_KPI_LAST", str(self.exec_kpi_last)))
        except Exception:
            pass
        self.skip_accounting = (os.getenv("DAILY_SKIP_ACCOUNTING", "1" if self.skip_accounting else "0").strip() in ("1", "true", "True", "yes", "YES"))
        self.skip_oos = (os.getenv("DAILY_SKIP_OOS", "1" if self.skip_oos else "0").strip() in ("1", "true", "True", "yes", "YES"))


async def run_daily_full(cfg: DailyReportCfg) -> Dict[str, Any]:
    ts = _utcnow()
    day = ts.astimezone(timezone.utc).strftime("%Y%m%d")
    day_dir = cfg.out_dir / day
    day_dir.mkdir(parents=True, exist_ok=True)

    # 1) Accounting snapshot + report
    acct_out = day_dir / "accounting_daily_report.json"
    acct: Dict[str, Any] = {}
    if cfg.skip_accounting:
        acct = {"ts": ts.isoformat(), "skipped": True, "error": "ACCOUNTING_SKIPPED (DAILY_SKIP_ACCOUNTING=1)"}
        try:
            acct_out.write_text(json.dumps(acct, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    else:
        try:
            acct = await run_acct_daily(
                AcctDailyCfg(
                    out_dir=Path(os.getenv("ACCOUNTING_OUT_DIR", "logs/accounting")),
                    jsonl=Path(os.getenv("ACCOUNTING_JSONL", "logs/accounting_snapshots.jsonl")),
                    tz_name=str(os.getenv("REPORT_TZ", cfg.report_tz)),
                    report_out=acct_out,
                    print_json=False,
                )
            )
        except Exception as e:
            # Não aborta o daily: ainda queremos OOS + KPIs + aderência mesmo sem login no accounting.
            acct = {"ts": ts.isoformat(), "error": f"ACCOUNTING_FAILED: {str(e)[:200]}"}
            try:
                acct_out.write_text(json.dumps(acct, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

    # 2) Execution KPIs (all + success-only)
    exec_lines = []
    if cfg.executor_jsonl.exists():
        exec_lines = _read_jsonl_last(cfg.executor_jsonl, int(cfg.exec_kpi_last))
    kpi_all = compute_kpis_from_lines(exec_lines, path=str(cfg.executor_jsonl))
    kpi_ok = compute_kpis_from_lines(exec_lines, path=str(cfg.executor_jsonl), only_status=["LIVE_OK", "DRY_OK"])
    (day_dir / "execution_kpis_all.json").write_text(json.dumps(kpi_all, ensure_ascii=False, indent=2), encoding="utf-8")
    (day_dir / "execution_kpis_ok.json").write_text(json.dumps(kpi_ok, ensure_ascii=False, indent=2), encoding="utf-8")

    # recorte 24h (para prontidão LIVE): gaps/latência devem ser comparáveis a thresholds (≤8 gaps, p90≤8s, etc.)
    exec_lines_24h: list[str] = []
    try:
        since24 = _utcnow() - timedelta(hours=24.0)
        exec_lines_24h = _filter_executor_jsonl_lines_window(exec_lines, since_utc=since24)
    except Exception:
        exec_lines_24h = []
    kpi_ok_24h = compute_kpis_from_lines(exec_lines_24h, path=str(cfg.executor_jsonl), only_status=["LIVE_OK", "DRY_OK"])
    (day_dir / "execution_kpis_ok_24h.json").write_text(json.dumps(kpi_ok_24h, ensure_ascii=False, indent=2), encoding="utf-8")
    exec_post_24h = _executor_post_accept_failures_24h(exec_lines_24h)
    (day_dir / "execution_post_accept_24h.json").write_text(
        json.dumps(exec_post_24h, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    exec_e2e_24h: Dict[str, Any] = {"error": None}
    try:
        audit_ids_24h = _extract_audit_ids_from_exec_lines(exec_lines_24h)
        if audit_ids_24h:
            from storage.database import Database  # local import para manter daily operável sem DB

            db = Database()
            await db.connect()
            try:
                audit_by_id_24h = await _fetch_audit_rows_for_ids_daily(db, audit_ids_24h)
            finally:
                await db.close()
            exec_e2e_24h = _executor_e2e_latency_24h(exec_lines_24h, audit_by_id_24h)
            exec_e2e_24h["audit_ids_24h"] = int(len(audit_ids_24h))
            exec_e2e_24h["audit_rows_found_24h"] = int(len(audit_by_id_24h))
        else:
            exec_e2e_24h = _executor_e2e_latency_24h(exec_lines_24h, {})
    except Exception as e:
        exec_e2e_24h = {"error": str(e)[:240], "n_jsonl_24h": int(len(exec_lines_24h or []))}
    (day_dir / "execution_latency_e2e_24h.json").write_text(
        json.dumps(exec_e2e_24h, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # atividade recente (ajuda a diagnosticar "hoje não teve aposta" sem depender do DB)
    exec_activity: Dict[str, Any] = {"last_live_ok_ts": None, "live_ok_1h": 0, "live_ok_6h": 0, "live_ok_24h": 0}
    try:
        nowu = _utcnow()
        cut1 = nowu - timedelta(hours=1.0)
        cut6 = nowu - timedelta(hours=6.0)
        cut24 = nowu - timedelta(hours=24.0)
        last_live = None
        c1 = c6 = c24 = 0
        for ln in exec_lines:
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            res = obj.get("result") if isinstance(obj, dict) else None
            req = obj.get("request") if isinstance(obj, dict) else None
            if not isinstance(res, dict):
                continue
            st = str(res.get("status") or "")
            if st == "HEARTBEAT":
                continue
            if st != "LIVE_OK":
                continue
            ts0 = _parse_iso_dt_best(str(res.get("finished_at") or res.get("created_at") or (req.get("created_at") if isinstance(req, dict) else "") or ""))
            if not ts0:
                continue
            if last_live is None or ts0 > last_live:
                last_live = ts0
            if ts0 >= cut24:
                c24 += 1
            if ts0 >= cut6:
                c6 += 1
            if ts0 >= cut1:
                c1 += 1
        exec_activity = {
            "last_live_ok_ts": last_live.isoformat() if last_live else None,
            "live_ok_1h": int(c1),
            "live_ok_6h": int(c6),
            "live_ok_24h": int(c24),
        }
    except Exception:
        exec_activity = {"last_live_ok_ts": None, "live_ok_1h": 0, "live_ok_6h": 0, "live_ok_24h": 0}

    # 3) Rodar OOS (walk-forward) e exportar policy
    base_md = day_dir / "report_base.md"
    # Histórico de policy:
    # - `wf_policy_YYYYMMDD.json` é o snapshot canônico do dia (não deve ser sobrescrito em reruns).
    # - reruns escrevem `wf_policy_YYYYMMDD_HHMMSS.json` para evitar “revisar o passado”.
    policy_canon = cfg.wf_policy_history_dir / f"wf_policy_{day}.json"
    policy_hist = policy_canon
    if policy_canon.exists():
        policy_hist = cfg.wf_policy_history_dir / f"wf_policy_{day}_{ts.strftime('%H%M%S')}.json"
    bank_sens_json = day_dir / "wf_bank_sensitivity.json"
    cfg.wf_policy_history_dir.mkdir(parents=True, exist_ok=True)

    args = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent / "analyze_contexto_operacao_b808_robust_report.py"),
        "--hypothesis-type",
        str(cfg.hypothesis_type),
        "--direction",
        str(cfg.direction),
        "--versions",
        str(cfg.versions),
        "--out",
        str(base_md),
        "--report-mode",
        str(cfg.report_mode),
        "--walkforward",
        "--wf-export-policy-json",
        str(policy_hist),
        "--wf-export-bank-sensitivity-json",
        str(bank_sens_json),
    ]
    if bool(cfg.no_auto_exclude_days):
        args += ["--no-auto-exclude-days"]
    if str(cfg.lookback_days).strip():
        args += ["--lookback-days", str(cfg.lookback_days).strip()]
    if str(cfg.kelly_bankroll).strip():
        args += ["--kelly-bankroll", str(cfg.kelly_bankroll).strip()]
    if str(cfg.wf_bankroll_grid).strip():
        args += ["--wf-bankroll-grid", str(cfg.wf_bankroll_grid).strip()]
    if str(cfg.wf_train_mode).strip():
        args += ["--wf-train-mode", str(cfg.wf_train_mode).strip()]
    if str(cfg.wf_train_days).strip():
        args += ["--wf-train-days", str(cfg.wf_train_days).strip()]
    if str(cfg.wf_test_days).strip():
        args += ["--wf-test-days", str(cfg.wf_test_days).strip()]
    if str(cfg.wf_step_days).strip():
        args += ["--wf-step-days", str(cfg.wf_step_days).strip()]
    if bool(cfg.wf_key_by_league):
        args += ["--wf-key-by-league"]
        if str(cfg.wf_key_by_league_scope).strip():
            args += ["--wf-key-by-league-scope", str(cfg.wf_key_by_league_scope).strip()]
        if bool(cfg.wf_experimental_stats):
            args += ["--wf-experimental-stats"]
    if str(cfg.wf_ah_max_abs_line).strip():
        args += ["--wf-ah-max-abs-line", str(cfg.wf_ah_max_abs_line).strip()]
        if str(cfg.wf_ah_scope).strip():
            args += ["--wf-ah-scope", str(cfg.wf_ah_scope).strip()]
    if str(cfg.wf_liquidity_mode).strip():
        args += ["--wf-liquidity-mode", str(cfg.wf_liquidity_mode).strip()]
        if str(cfg.wf_liquidity_scope).strip():
            args += ["--wf-liquidity-scope", str(cfg.wf_liquidity_scope).strip()]
    if str(cfg.wf_min_matches).strip():
        args += ["--wf-min-matches", str(cfg.wf_min_matches).strip()]
    if str(cfg.wf_pre_activation_mode).strip():
        args += ["--wf-pre-activation-mode", str(cfg.wf_pre_activation_mode).strip()]
    if str(cfg.wf_roi_min_activate).strip():
        args += ["--wf-roi-min-activate", str(cfg.wf_roi_min_activate).strip()]
    if bool(cfg.wf_shrinkage):
        args += ["--wf-shrinkage"]
    if str(cfg.wf_exclude_exec_buckets_back).strip():
        args += ["--wf-exclude-exec-buckets-back", str(cfg.wf_exclude_exec_buckets_back).strip()]
    if str(cfg.wf_exclude_exec_buckets_lay).strip():
        args += ["--wf-exclude-exec-buckets-lay", str(cfg.wf_exclude_exec_buckets_lay).strip()]
    # Restrição por lado/regime para alinhar policy com o modo operacional do bridge.
    # Defaults defensivos:
    # - sides: herda DAILY_WF_SIDES; se vazio, deriva de BRIDGE_EXEC_SIDE.
    # - regimes: herda DAILY_WF_REGIMES; se vazio e BRIDGE_PREMATCH_ONLY=1, usa "pre".
    try:
        wf_sides = str(os.getenv("DAILY_WF_SIDES", "") or "").strip().lower()
        if not wf_sides:
            bside = str(os.getenv("BRIDGE_EXEC_SIDE", "Back") or "Back").strip().lower()
            if bside in ("back", "lay", "both"):
                wf_sides = bside
            else:
                wf_sides = "back"
        if wf_sides:
            args += ["--wf-sides", wf_sides]
    except Exception:
        pass
    try:
        wf_regimes = str(os.getenv("DAILY_WF_REGIMES", "") or "").strip().lower()
        if not wf_regimes:
            wf_regimes = "pre" if _is_truthy(os.getenv("BRIDGE_PREMATCH_ONLY", "1")) else "both"
        if wf_regimes:
            args += ["--wf-regimes", wf_regimes]
    except Exception:
        pass
    try:
        if str(cfg.wf_backpre_slip_max).strip():
            args += ["--wf-backpre-slip-max", str(cfg.wf_backpre_slip_max).strip()]
        if str(cfg.wf_backpre_slip_field).strip():
            args += ["--wf-backpre-slip-field", str(cfg.wf_backpre_slip_field).strip()]
        if str(cfg.wf_backpre_fast_max_lag_ms).strip():
            args += ["--wf-backpre-fast-max-lag-ms", str(cfg.wf_backpre_fast_max_lag_ms).strip()]
    except Exception:
        pass

    # Overrides opcionais (rodagem manual): sizing/budget do WF
    if str(cfg.wf_scheme_pre).strip():
        args += ["--wf-scheme-pre", str(cfg.wf_scheme_pre).strip()]
    if str(cfg.wf_scheme_in).strip():
        args += ["--wf-scheme-in", str(cfg.wf_scheme_in).strip()]
    if str(cfg.wf_flat_stake_back).strip():
        args += ["--wf-flat-stake-back", str(cfg.wf_flat_stake_back).strip()]
    if str(cfg.wf_flat_liab_lay).strip():
        args += ["--wf-flat-liab-lay", str(cfg.wf_flat_liab_lay).strip()]
    if str(cfg.wf_budget_back_frac).strip():
        args += ["--wf-budget-back-frac", str(cfg.wf_budget_back_frac).strip()]
    if str(cfg.wf_budget_lay_frac).strip():
        args += ["--wf-budget-lay-frac", str(cfg.wf_budget_lay_frac).strip()]
    if str(cfg.wf_budget_cap_signal_frac).strip():
        args += ["--wf-budget-cap-signal-frac", str(cfg.wf_budget_cap_signal_frac).strip()]
    if str(cfg.wf_budget_risk_mode).strip():
        args += ["--wf-budget-risk-mode", str(cfg.wf_budget_risk_mode).strip()]

    # Sweep de caps absolutos no OOS (nova seção no PDF)
    if bool(cfg.wf_sweep_stakes):
        args += ["--wf-sweep-stakes"]
        if str(cfg.wf_sweep_back_caps).strip():
            args += ["--wf-sweep-back-caps", str(cfg.wf_sweep_back_caps).strip()]
        if str(cfg.wf_sweep_lay_caps).strip():
            args += ["--wf-sweep-lay-caps", str(cfg.wf_sweep_lay_caps).strip()]
        if bool(cfg.wf_sweep_grid_in):
            args += ["--wf-sweep-grid-in"]

    oos_run = {"skipped": False, "ok": True, "returncode": 0, "error": None, "log": str(day_dir / "oos_run.log")}
    if cfg.skip_oos:
        oos_run = {"skipped": True, "ok": False, "returncode": None, "error": "OOS_SKIPPED (DAILY_SKIP_OOS=1)", "log": None}
    else:
        try:
            log_path = Path(str(oos_run["log"]))
            proc = subprocess.run(args, check=False, cwd=str(Path(__file__).resolve().parent.parent), capture_output=True, text=True)
            oos_run["returncode"] = int(proc.returncode)
            if proc.returncode != 0:
                oos_run["ok"] = False
                oos_run["error"] = f"OOS_FAILED: returncode={proc.returncode}"
            # sempre grava log (stdout+stderr) para debug no VPS
            try:
                log_path.write_text((proc.stdout or "") + "\n\n--- STDERR ---\n\n" + (proc.stderr or ""), encoding="utf-8")
            except Exception:
                pass
        except Exception as e:
            oos_run["ok"] = False
            oos_run["error"] = f"OOS_EXCEPTION: {str(e)[:200]}"

    policy_publish_info: Dict[str, Any] = {
        "enabled": bool(cfg.publish_policy_current),
        "guard_enabled": bool(cfg.policy_compat_guard_enable),
        "fail_closed": bool(cfg.policy_compat_fail_closed),
        "published": False,
        "reason": "skipped",
        "compatibility": None,
        "candidate_path": str(policy_hist),
        "effective_path": str(cfg.wf_policy_current),
    }
    active_keys = None
    active_keys_base = None
    policy_wf: Optional[Dict[str, Any]] = None
    policy_last_step: Optional[Dict[str, Any]] = None
    candidate_active_keys = None

    # Atualiza policy_current (atomic replace) e registra histórico (jsonl) apenas se o OOS rodou com sucesso
    if (not cfg.skip_oos) and bool(oos_run.get("ok")) and policy_hist.exists():
        pol_candidate: Optional[Dict[str, Any]] = None
        candidate_last: Optional[Dict[str, Any]] = None
        try:
            pol_candidate = json.loads(policy_hist.read_text(encoding="utf-8"))
            csteps = pol_candidate.get("steps") if isinstance(pol_candidate, dict) else []
            candidate_last = csteps[-1] if isinstance(csteps, list) and csteps else {}
            if isinstance(candidate_last, dict):
                candidate_active_keys = candidate_last.get("active_keys")
                policy_publish_info["candidate_active_keys_n"] = int(len(list(candidate_active_keys or [])))
        except Exception as e:
            policy_publish_info["reason"] = f"candidate_policy_parse_failed: {str(e)[:120]}"

        # Preenche o snapshot canônico do dia (best-effort) apenas se ainda não existir.
        # Isso evita que re-runs manuais sobrescrevam o arquivo `wf_policy_YYYYMMDD.json`.
        try:
            if policy_canon and (not policy_canon.exists()):
                tmpc = policy_canon.with_suffix(".tmp")
                tmpc.write_text(policy_hist.read_text(encoding="utf-8"), encoding="utf-8")
                tmpc.replace(policy_canon)
        except Exception:
            pass

        can_publish = bool(cfg.publish_policy_current)
        if can_publish and isinstance(candidate_last, dict):
            compat = _policy_compatibility_check(
                candidate_last.get("active_keys"),
                bridge_exec_side=str(cfg.policy_compat_bridge_exec_side or "Back"),
                bridge_prematch_only=bool(cfg.policy_compat_prematch_only),
                min_pre_keys=int(cfg.policy_compat_min_pre_keys),
            )
            policy_publish_info["compatibility"] = compat
            if bool(cfg.policy_compat_guard_enable) and (not bool(compat.get("ok"))):
                if bool(cfg.policy_compat_fail_closed):
                    can_publish = False
                    policy_publish_info["reason"] = f"blocked_by_compat_guard: {compat.get('reason')}"
                else:
                    policy_publish_info["reason"] = f"compat_guard_warn_fail_open: {compat.get('reason')}"

        if can_publish and bool(cfg.publish_policy_current):
            try:
                cfg.wf_policy_current.parent.mkdir(parents=True, exist_ok=True)
                tmp = cfg.wf_policy_current.with_suffix(".tmp")
                tmp.write_text(policy_hist.read_text(encoding="utf-8"), encoding="utf-8")
                tmp.replace(cfg.wf_policy_current)
                policy_publish_info["published"] = True
                policy_publish_info["reason"] = "published"
            except Exception as e:
                policy_publish_info["published"] = False
                policy_publish_info["reason"] = f"publish_failed: {str(e)[:120]}"
        elif bool(cfg.publish_policy_current) and str(policy_publish_info.get("reason") or "") == "skipped":
            policy_publish_info["reason"] = "publish_disabled_or_guard_block"

        # Política efetiva para relatório: a publicada no current (quando publish foi bloqueado/falhou),
        # senão a policy candidata recém-gerada.
        pol_effective = pol_candidate if bool(policy_publish_info.get("published")) else _read_json(cfg.wf_policy_current)
        if not isinstance(pol_effective, dict):
            pol_effective = pol_candidate if isinstance(pol_candidate, dict) else None
        if isinstance(pol_effective, dict):
            esteps = pol_effective.get("steps") if isinstance(pol_effective, dict) else []
            elast = esteps[-1] if isinstance(esteps, list) and esteps else {}
            if isinstance(elast, dict):
                active_keys = elast.get("active_keys")
                active_keys_base = elast.get("active_keys_base")
                policy_last_step = elast
            if isinstance(pol_effective.get("wf"), dict):
                policy_wf = pol_effective.get("wf")

        try:
            rec = {
                "ts": ts.isoformat(),
                "policy_path": str(policy_hist),
                "policy_current": str(cfg.wf_policy_current),
                "active_keys": active_keys,
                "active_keys_base": active_keys_base,
                "published_to_current": bool(policy_publish_info.get("published")),
                "publish_reason": str(policy_publish_info.get("reason") or ""),
                "candidate_active_keys": candidate_active_keys,
            }
            cfg.wf_policy_history_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with cfg.wf_policy_history_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # 4) Relatórios auxiliares para seção 0/1 e apêndices (99.x)
    # Adherence: (a) curto para tabelas diárias; (b) longo/acumulado para slippage/combos/contrafactuais
    adh_short_json = day_dir / "oos_adherence_short.json"
    adh_long_json = day_dir / "oos_adherence_long.json"
    exec_min_json = day_dir / "execution_minimal_by_type_24h.json"
    adh_short: Optional[Dict[str, Any]] = None
    adh_long: Optional[Dict[str, Any]] = None
    exec_min: Optional[Dict[str, Any]] = None
    try:
        slip_cf_start_day = str(os.getenv("OOS_ADHERENCE_SLIP_CF_START_DAY", "") or "").strip() or None
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ops.oos_adherence_report",
                "--policy-json",
                str(cfg.wf_policy_current),
                "--executor-jsonl",
                str(cfg.executor_jsonl),
                "--tz",
                "UTC",
                "--days",
                str(os.getenv("DAILY_ADHERENCE_DAYS_TABLE", os.getenv("DAILY_ADHERENCE_DAYS", "7"))),
                *(
                    ["--slippage-cf-start-day", slip_cf_start_day]
                    if slip_cf_start_day
                    else []
                ),
                "--out",
                str(adh_short_json),
            ],
            check=False,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        adh_short = _read_json(adh_short_json)
    except Exception:
        adh_short = None
    try:
        slip_cf_start_day = str(os.getenv("OOS_ADHERENCE_SLIP_CF_START_DAY", "") or "").strip() or None
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ops.oos_adherence_report",
                "--policy-json",
                str(cfg.wf_policy_current),
                "--executor-jsonl",
                str(cfg.executor_jsonl),
                "--tz",
                "UTC",
                "--days",
                str(os.getenv("DAILY_ADHERENCE_DAYS_SLIPPAGE", "0")),
                "--no-per-day",
                *(
                    ["--slippage-cf-start-day", slip_cf_start_day]
                    if slip_cf_start_day
                    else []
                ),
                "--out",
                str(adh_long_json),
            ],
            check=False,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        adh_long = _read_json(adh_long_json)
    except Exception:
        adh_long = None

    # Execução: métricas mínimas por tipo (Back/Lay × Pre/In) — janela curta (horas)
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ops.execution_minimal_by_type",
                "--executor-jsonl",
                str(cfg.executor_jsonl),
                "--hours",
                str(os.getenv("DAILY_EXEC_MIN_BY_TYPE_HOURS", "24")),
                "--only-status",
                str(os.getenv("DAILY_EXEC_MIN_BY_TYPE_ONLY_STATUS", "LIVE_OK")),
                "--out",
                str(exec_min_json),
            ],
            check=False,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        exec_min = _read_json(exec_min_json)
    except Exception:
        exec_min = None

    audit_json = day_dir / "audit_status_kpis.json"
    audit_rep: Optional[Dict[str, Any]] = None
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ops.audit_status_kpis",
                "--hours",
                str(os.getenv("DAILY_AUDIT_KPI_HOURS", "24")),
                "--direction",
                str(cfg.direction),
                "--out",
                str(audit_json),
            ],
            check=False,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        audit_rep = _read_json(audit_json)
    except Exception:
        audit_rep = None

    # 5) Montagem do relatório final:
    # Ordem pedida: 0 (Resumo) -> 1 (Resultados reais) -> 2 (OOS) -> 3 (In-sample) -> 99 (apêndices operacionais)
    base_txt = ""
    if base_md.exists():
        try:
            base_txt = base_md.read_text(encoding="utf-8")
        except Exception:
            base_txt = ""
    insample_txt, oos_txt = _split_base_into_insample_and_oos(base_txt)
    oos_as_annex = (os.getenv("DAILY_OOS_AS_ANNEX", "1").strip() not in ("0", "false", "False", "no", "NO"))
    if oos_txt and (not oos_as_annex):
        oos_txt = (
            oos_txt.replace("## 12) OOS walk-forward", "## 2) OOS walk-forward")
            .replace("## 1) OOS walk-forward", "## 2) OOS walk-forward")
        )

    # Accounting: série por dia/mês a partir do CSV (quando disponível)
    acct_series = None
    try:
        bal_csv = Path(str(acct.get("balance_csv") or "")).expanduser()
        if bal_csv.exists():
            tz = timezone.utc
            try:
                from zoneinfo import ZoneInfo  # type: ignore

                tz = ZoneInfo(str(os.getenv("REPORT_TZ", cfg.report_tz)))
            except Exception:
                tz = timezone.utc
            acct_series = compute_pnl_report(bal_csv, tz=tz)
    except Exception:
        acct_series = None

    # --- Seção 0: Resumo / conclusões (executivo) ---
    s0 = []
    s0.append("## 0) Resumo e conclusões (executivo)\n\n")
    # BEGIN H3BUP_VNEXT_OFFICIAL_SUMMARY_P0
    try:
        from .daily_v2.v1_h3bup_summary import render_h3bup_vnext_official_summary

        s0.append(render_h3bup_vnext_official_summary(Path(".")))
        s0.append("\n")
    except Exception as _e_h3b_sum:
        s0.append("## H3BUP_vNext — Resumo Oficial da Estratégia\n\n")
        s0.append(f"_indisponível (fail-open): {str(_e_h3b_sum)[:160]}_\n\n")
        s0.append(
            "> Os valores de banca, P&L semanal/mensal da conta e estudos históricos "
            "não representam necessariamente a performance da H3BUP_vNext.\n\n"
        )
    # END H3BUP_VNEXT_OFFICIAL_SUMMARY_P0

    s0.append("### CONTA TOTAL (não confundir com H3BUP_vNext)\n\n")
    if cfg.skip_oos or (isinstance(oos_run, dict) and not bool(oos_run.get("ok"))):
        s0.append("**Status do OOS (walk-forward)**\n\n")
        if cfg.skip_oos:
            s0.append("- **OOS**: **SKIPPED** (`DAILY_SKIP_OOS=1`).\n\n")
        else:
            s0.append(f"- **OOS**: **FAILED** — `{oos_run.get('error')}`\n")
            if oos_run.get("log"):
                s0.append(f"- Log: `{oos_run.get('log')}`\n\n")
    else:
        if bool(cfg.publish_policy_current):
            if bool(policy_publish_info.get("published")):
                s0.append(
                    "- **Policy publish**: `OK` "
                    f"(candidate `{policy_publish_info.get('candidate_active_keys_n', '—')}` keys → `{cfg.wf_policy_current}`).\n"
                )
            else:
                s0.append(
                    "- **Policy publish**: `BLOQUEADO` "
                    f"(`{policy_publish_info.get('reason')}`) — mantendo policy anterior em `{cfg.wf_policy_current}`.\n"
                )

    # performance “real” (accounting) quando houver — CONTA TOTAL
    if isinstance(acct, dict) and not acct.get("error"):
        s0.append(f"- **Banca real (saldo atual) — CONTA TOTAL**: `{acct.get('balance_current')}`\n")
        s0.append(
            f"- **P&L conta (hoje / semana / mês) — CONTA TOTAL**: "
            f"`{acct.get('pnl_today')} / {acct.get('pnl_week')} / {acct.get('pnl_month')}`\n"
        )
    else:
        s0.append("- **Accounting (CONTA TOTAL)**: indisponível (ver apêndice 99.1)\n")

    # BEGIN H3BUP_ACCOUNTING_HEALTH_SECTION
    try:
        from .accounting_health_report import load_health, render_accounting_health_h3bup_section
        _hpath = Path(os.getenv("ACCOUNTING_HEALTH_JSON", "logs/accounting/accounting_health.json"))
        _health = load_health(_hpath)
        _sum = {}
        try:
            _sum_path = Path(os.getenv("H3BUP_ACCOUNTING_SUMMARY_JSON", "logs/h3bup_accounting_summary_latest.json"))
            if _sum_path.exists():
                import json as _json
                _sum = _json.loads(_sum_path.read_text(encoding="utf-8"))
        except Exception:
            _sum = {}
        s0.append(render_accounting_health_h3bup_section(health=_health, reconcile_summary=_sum))
        s0.append("\n")
    except Exception as _e_acc_health:
        s0.append("## Accounting Health — H3BUP\n\n")
        s0.append(f"_indisponível: {str(_e_acc_health)[:160]}_\n\n")
    # END H3BUP_ACCOUNTING_HEALTH_SECTION

    # BEGIN H3BUP_E2E_LATENCY_SECTION
    try:
        from pathlib import Path as _Path
        import os as _os
        from .analyze_h3bup_e2e_latency import load_events, group_traces, analyze_trace, summarize, render_daily_section
        _tpath = _Path(_os.getenv("H3BUP_E2E_TRACE_PATH", "logs/h3bup_e2e_trace.jsonl"))
        _evs = load_events(_tpath) if _tpath.exists() else []
        _trs = group_traces(_evs)
        _rows = [analyze_trace(tid, evs) for tid, evs in _trs.items()]
        _summary, _by_st, _cov = summarize(_rows)
        _health = {
            "enabled": bool(_tpath.exists()),
            "schema_version": 1,
            "trace_events_dropped": 0,
            "clock_skew": sum(1 for r in _rows if r.get("clock_skew_suspected")),
            "ordering_violations": sum(1 for r in _rows if r.get("ordering_violations")),
        }
        s0.append(render_daily_section(
            _summary, _cov, health=_health,
            n_traces=len(_rows),
            n_live=sum(1 for r in _rows if r.get("status") == "LIVE_OK"),
        ))
        s0.append("\n")
    except Exception as _e_e2e:
        s0.append("## H3BUP End-to-End Latency\n\n")
        s0.append(f"_indisponível (fail-open): {str(_e_e2e)[:160]}_\n\n")
    # END H3BUP_E2E_LATENCY_SECTION

    # BEGIN H3BUP_CLV_FORWARD_SECTION
    try:
        import json as _json
        from pathlib import Path as _Path
        import os as _os
        _hp = _Path(_os.getenv("H3BUP_CLV_HEALTH_PATH", "logs/h3bup_clv_health.json"))
        _h = _json.loads(_hp.read_text(encoding="utf-8")) if _hp.exists() else {"status": "WATCH", "enabled": False}
        s0.append("## H3BUP CLV Forward Collection\n\n")
        s0.append("| Métrica | Valor |\n|---|---|\n")
        for k in [
            ("collection status", _h.get("status")),
            ("collection started at", _h.get("collection_started_at_utc")),
            ("source priority", ",".join(_h.get("source_priority") or [])),
            ("passive collector status", _h.get("collector_status")),
            ("LIVE_OK após activação", _h.get("live_ok_after_activation")),
            ("obligations esperadas", _h.get("obligations_expected")),
            ("obligations criadas", _h.get("obligations_created")),
            ("POST_5M strict válidas", _h.get("post_5m_valid_strict")),
            ("POST_15M strict válidas", _h.get("post_15m_valid_strict")),
            ("CLOSING strict válidas", _h.get("closing_valid_strict")),
            ("source missing", _h.get("source_missing")),
            ("line mismatch", _h.get("line_mismatch")),
            ("kickoff missing", _h.get("kickoff_missing")),
            ("retry backlog", _h.get("retry_backlog")),
            ("status estatístico", ("INSUFFICIENT_N" if int(_h.get("live_ok_after_activation") or 0) < 30 else "OK")),
        ]:
            s0.append(f"| {k[0]} | {k[1]} |\n")
        s0.append("\n")
    except Exception as _e_clv:
        s0.append("## H3BUP CLV Forward Collection\n\n")
        s0.append(f"_indisponível (fail-open): {str(_e_clv)[:160]}_\n\n")
    # END H3BUP_CLV_FORWARD_SECTION

    # lucro "esperado operacional" aproximado: aplica a regra do gate de slippage no subconjunto com placar (contrafactual)
    try:
        per_day = (adh_short or {}).get("per_day") if isinstance(adh_short, dict) else None
        if isinstance(per_day, list) and per_day:
            base = 0.0
            filt = 0.0
            n = 0
            for it in per_day:
                if not isinstance(it, dict):
                    continue
                cf = it.get("slippage_filter_counterfactual")
                if not isinstance(cf, dict):
                    continue
                b = cf.get("back") if isinstance(cf.get("back"), dict) else {}
                l = cf.get("lay") if isinstance(cf.get("lay"), dict) else {}
                try:
                    base += float(b.get("pnl") or 0.0) + float(l.get("pnl") or 0.0)
                    filt += float(b.get("pnl_filtered") or 0.0) + float(l.get("pnl_filtered") or 0.0)
                    n += int(b.get("n") or 0) + int(l.get("n") or 0)
                except Exception:
                    continue
            if n > 0:
                s0.append(
                    f"- **Lucro esperado (com gate de slippage; exec c/ placar)**: `{_fmt_num(filt,2)}` "
                    f"(base `{_fmt_num(base,2)}`, Δ `{_fmt_num(filt-base,2)}`)\n"
                )
    except Exception:
        pass

    # risco/estabilidade operacional (últimas 24h) via audit_status_kpis
    top_errs = []
    try:
        for it in (audit_rep or {}).get("error_rows") or []:
            if not isinstance(it, dict):
                continue
            n = int(it.get("n") or 0)
            if n <= 0:
                continue
            top_errs.append((n, str(it.get("audit_version") or ""), str(it.get("status") or ""), str(it.get("api_error") or "")))
        top_errs.sort(key=lambda x: x[0], reverse=True)
    except Exception:
        top_errs = []
    if top_errs:
        s0.append("\n**Principais causas de perda de throughput (24h)**\n\n")
        for n, ver, st, err in top_errs[:6]:
            err2 = (err[:160] + "…") if len(err) > 160 else err
            s0.append(f"- `{ver}`: **{st} ×{n}** — `{err2}`\n")

    # conversão (audit) e saúde do executor (gaps)
    try:
        if isinstance(audit_rep, dict) and isinstance(audit_rep.get("by_version"), list):
            tot = ok = valid = 0
            for v in audit_rep.get("by_version") or []:
                if not isinstance(v, dict):
                    continue
                sc = v.get("status_counts") if isinstance(v.get("status_counts"), dict) else {}
                tot += int(v.get("total") or 0)
                ok += int(sc.get("OK") or 0)
                valid += int(v.get("ok_valid") or 0)
            if tot > 0:
                s0.append("\n**Conversão (últimas 24h; auditoria DB)**\n\n")
                s0.append(f"- OK/total: **{ok}/{tot}** ({(ok/tot)*100.0:.1f}%)\n")
                s0.append(f"- OK_valid/total: **{valid}/{tot}** ({(valid/tot)*100.0:.1f}%)\n")
    except Exception:
        pass
    try:
        gaps = _executor_gaps_summary(exec_lines)
        if gaps.get("n") and gaps.get("max_gap_s") is not None:
            s0.append("\n**Saúde do executor (amostra lida do JSONL; não é 24h)**\n\n")
            s0.append(f"- Janela: `{gaps.get('first_ts')}` → `{gaps.get('last_ts')}` (n={gaps.get('n')})\n")
            s0.append(
                f"- Maior gap: `{_fmt_num(gaps.get('max_gap_s'),1)}s` | gaps>5min: `{gaps.get('gaps_gt_300s')}`\n"
            )
    except Exception:
        pass

    # gaps em janela fixa (24h) para prontidão LIVE
    gaps24 = None
    try:
        since24 = _utcnow() - timedelta(hours=24.0)
        gaps24 = _executor_gaps_summary_window(exec_lines, since_utc=since24)
        if isinstance(gaps24, dict) and gaps24.get("n"):
            s0.append("\n**Saúde do executor (últimas 24h; proxy por gaps no JSONL)**\n\n")
            s0.append(f"- Janela: `{gaps24.get('since_utc')}` → `{gaps24.get('until_utc')}` (n={gaps24.get('n')})\n")
            s0.append(
                f"- Maior gap: `{_fmt_num(gaps24.get('max_gap_s'),1)}s` | gaps>15min: `{gaps24.get('gaps_gt_900s')}` | "
                f"silêncio>15min (est.): `{_fmt_num(gaps24.get('silence_over_15m_s'),0)}s` ({_fmt_num(gaps24.get('silence_over_15m_pct'),2)}%)\n"
            )
    except Exception:
        gaps24 = None

    # snapshot simples de memória (ajuda a explicar latência/timeouts)
    try:
        mav = _mem_available_mib()
        if mav is not None:
            s0.append("\n**Recursos da VPS (snapshot)**\n\n")
            s0.append(f"- MemAvailable: `{_fmt_num(mav,0)} MiB`\n")
            try:
                vc = _vcpu_count()
                if vc is not None:
                    s0.append(f"- vCPUs (os.cpu_count): `{int(vc)}`\n")
            except Exception:
                pass
    except Exception:
        pass

    # Se o JSONL está stale, a seção "Execução por dia" vai aparecer zerada mesmo que o bridge/audit estejam rodando.
    try:
        exec_last_ts = _parse_iso_dt_best((gaps or {}).get("last_ts"))
        if exec_last_ts:
            age_h = (datetime.now(timezone.utc) - exec_last_ts).total_seconds() / 3600.0
            thr_h = float(os.getenv("DAILY_EXECUTOR_JSONL_STALE_HOURS", "6.0"))
            if age_h > thr_h:
                s0.append(
                    f"\n**Alerta: executor_jsonl possivelmente desatualizado**\n\n"
                    f"- Último registro no `executor_jsonl`: `{exec_last_ts.isoformat()}` (idade ≈ `{_fmt_num(age_h,1)}h`, limiar `{_fmt_num(thr_h,1)}h`).\n"
                    "- Isso explica dias com `Exec rows=0` mesmo com auditoria DB (funil) mostrando volume.\n\n"
                )
    except Exception:
        pass

    # atividade recente (LIVE_OK): diagnostica rapidamente "hoje não teve aposta"
    try:
        last_live_ok = exec_activity.get("last_live_ok_ts") if isinstance(exec_activity, dict) else None
        s0.append("\n**Atividade recente (executor)**\n\n")
        s0.append(
            f"- Último `LIVE_OK`: `{last_live_ok or '—'}` | "
            f"`LIVE_OK` (1h/6h/24h): `{int(exec_activity.get('live_ok_1h') or 0)}/{int(exec_activity.get('live_ok_6h') or 0)}/{int(exec_activity.get('live_ok_24h') or 0)}`\n"
        )
        if int(exec_activity.get("live_ok_6h") or 0) == 0:
            s0.append("- Se isso persistir com auditoria OK no DB, suspeite de sessão/PMM/timeout ou bridge travado (ver checklist abaixo).\n")
        s0.append("\n")
    except Exception:
        pass

    # diagnóstico explícito pós-accepted (executor): separa pré-place vs place
    try:
        pa = exec_post_24h if isinstance(exec_post_24h, dict) else {}
        acc_n = int(pa.get("accepted_n") or 0)
        if acc_n > 0:
            ok_n = int(pa.get("live_ok_n") or 0)
            fail_n = int(pa.get("accepted_fail_n") or 0)
            s0.append("**Falhas pós-accepted (executor, 24h)**\n\n")
            s0.append("| Métrica | Valor |\n|---|---:|\n")
            s0.append(f"| accepted | {acc_n} |\n")
            s0.append(f"| LIVE_OK | {ok_n} ({_fmt_num((100.0*ok_n/acc_n) if acc_n else None,1)}%) |\n")
            s0.append(f"| accepted sem LIVE_OK | {fail_n} ({_fmt_num((100.0*fail_n/acc_n) if acc_n else None,1)}%) |\n")
            s0.append(f"| precheck fail (`LIVE_PRECHECK_FAILED`) | {int(pa.get('precheck_fail_n') or 0)} |\n")
            s0.append(f"| place fail (`LIVE_PLACE_FAILED`) | {int(pa.get('place_fail_n') or 0)} |\n")
            s0.append(f"| API_FAILED | {int(pa.get('api_failed_n') or 0)} |\n")
            s0.append(f"| NO_SESSION | {int(pa.get('no_session_n') or 0)} |\n")
            s0.append(f"| RATE_LIMIT | {int(pa.get('rate_limit_n') or 0)} |\n")
            s0.append(f"| CAP_BLOCKED | {int(pa.get('cap_blocked_n') or 0)} |\n")
            s0.append(f"| No PMMs received | {int(pa.get('no_pmms_n') or 0)} |\n")
            s0.append(f"| Execution context destroyed/target closed | {int(pa.get('ctx_destroyed_n') or 0)} |\n")
            s0.append(f"| Auth 401 / NO_ROOT_SESSION_COOKIE | {int(pa.get('auth_401_n') or 0)} |\n")
            s0.append(
                f"| p50/p90 `pmm_wait_ms` (precheck fail) | {_fmt_num(pa.get('precheck_pmm_wait_ms_p50'),0)} / {_fmt_num(pa.get('precheck_pmm_wait_ms_p90'),0)} |\n"
            )
            s0.append(
                f"| p50/p90 `ws_age_ms` (precheck fail) | {_fmt_num(pa.get('precheck_ws_age_ms_p50'),0)} / {_fmt_num(pa.get('precheck_ws_age_ms_p90'),0)} |\n"
            )
            s0.append("\n")
            tops = pa.get("top_errors") if isinstance(pa.get("top_errors"), list) else []
            if tops:
                s0.append("- Top erros pós-accepted:\n")
                for it in tops[:6]:
                    if not isinstance(it, dict):
                        continue
                    err = str(it.get("error") or "").strip()
                    n = int(it.get("n") or 0)
                    if err:
                        err = (err[:180] + "…") if len(err) > 180 else err
                        s0.append(f"  - ×{n}: `{err}`\n")
                s0.append("\n")
    except Exception:
        pass

    # Latência ponta a ponta (WS detectado -> executor_done) já no bloco inicial.
    try:
        e2e = exec_e2e_24h if isinstance(exec_e2e_24h, dict) else {}
        grp_ok = e2e.get("success") if isinstance(e2e.get("success"), dict) else {}
        n_ok = int(e2e.get("n_e2e_success") or 0)
        n_all = int(e2e.get("n_e2e_all") or 0)
        n_aid = int(e2e.get("n_with_audit_id") or 0)
        n_det = int(e2e.get("n_with_detected_at") or 0)
        if n_all > 0:
            s0.append("**Latência ponta a ponta (24h; WS → executor_done)**\n\n")
            s0.append(
                f"- Cobertura: `n_jsonl_24h={int(e2e.get('n_jsonl_24h') or 0)}`, "
                f"`com_audit_id={n_aid}`, `com_hypothesis_detected_at={n_det}`, "
                f"`e2e_all={n_all}`, `e2e_success={n_ok}`.\n"
            )
            s0.append("| Etapa | p50 | p90 | p99 | mean |\n|---|---:|---:|---:|---:|\n")
            rows = [
                ("e2e_total", "e2e_total_ms"),
                ("detect_to_submit", "detect_to_submit_ms"),
                ("audit_total", "audit_total_ms"),
                ("audit_detect_to_click", "audit_detect_to_click_ms"),
                ("audit_click_to_betslip", "audit_click_to_betslip_ms"),
                ("audit_queue_wait", "audit_queue_wait_ms"),
                ("audit_parallel_fetch", "audit_parallel_fetch_ms"),
                ("audit_temporal_total", "audit_temporal_total_ms"),
                ("audit_execution", "audit_execution_ms"),
                ("audit_pipeline_overhead", "audit_pipeline_overhead_ms"),
                ("audit_db_save", "audit_db_save_ms"),
                ("audit_gate_wait", "audit_gate_wait_ms"),
                ("bridge_wait", "bridge_wait_ms"),
                ("executor_submit_to_done", "executor_submit_to_done_ms"),
                ("executor_queue_delay", "executor_queue_delay_ms"),
                ("executor_post", "executor_post_ms"),
                ("executor_total_api", "executor_total_api_ms"),
            ]
            for label, key in rows:
                a = grp_ok.get(key) if isinstance(grp_ok.get(key), dict) else {}
                s0.append(
                    f"| {label} | {_fmt_num(a.get('p50'),0)} | {_fmt_num(a.get('p90'),0)} | {_fmt_num(a.get('p99'),0)} | {_fmt_num(a.get('mean'),0)} |\n"
                )
            s0.append("\n")
    except Exception:
        pass

    # ------------------------------------------------------------
    # Prontidão para LIVE (go/no-go) — checklist objetivo
    # ------------------------------------------------------------
    s0.append("\n**Prontidão para LIVE (go/no-go)**\n\n")
    allow_live = _env_bool("EXECUTOR_ALLOW_LIVE", "0")
    # thresholds (configuráveis por env; defaults conservadores)
    thr_ok_valid_pct = _env_float("DAILY_LIVE_MIN_OK_VALID_PCT", "5.0")
    thr_api_failed_pct = _env_float("DAILY_LIVE_MAX_API_FAILED_PCT", "20.0")
    thr_stale_pct = _env_float("DAILY_LIVE_MAX_STALE_QUEUE_PCT", "10.0")
    thr_gaps_15m = int(_env_float("DAILY_LIVE_MAX_GAPS_15MIN", "8"))
    thr_p90_ms = int(_env_float("DAILY_LIVE_MAX_CALL_TO_DONE_P90_MS", "8000"))
    thr_open_betslips = int(_env_float("DAILY_LIVE_MAX_TOO_MANY_OPEN_BETSLIPS", "0"))
    thr_no_pmms = int(_env_float("DAILY_LIVE_MAX_NO_PMMS", "0"))

    tot24 = _sum_total(audit_rep)
    ok24 = _sum_ok(audit_rep)
    okv24 = _sum_ok_valid(audit_rep)
    api_failed24 = _sum_status(audit_rep, "API_FAILED")
    stale24 = _sum_status(audit_rep, "STALE_QUEUE_WAIT")
    err_open = _count_err_substr(audit_rep, "too_many_open_betslips")
    err_pmms = _count_err_substr(audit_rep, "no pmms received")
    pmm_consults_24h = None
    no_pmms_24h = None
    no_pmms_rate_24h = None
    pmm_ws_diag = None
    try:
        blk = (audit_rep or {}).get("pmm") if isinstance((audit_rep or {}).get("pmm"), dict) else {}
        tot = blk.get("total") if isinstance(blk.get("total"), dict) else {}
        pmm_consults_24h = int(tot.get("pmm_consults")) if tot.get("pmm_consults") is not None else None
        no_pmms_24h = int(tot.get("no_pmms")) if tot.get("no_pmms") is not None else None
        no_pmms_rate_24h = _safe_float(tot.get("no_pmms_rate_pct"))
        pmm_ws_diag = blk.get("ws_diag") if isinstance(blk.get("ws_diag"), dict) else None
    except Exception:
        pmm_consults_24h = None
        no_pmms_24h = None
        no_pmms_rate_24h = None
        pmm_ws_diag = None

    ok_valid_pct = (100.0 * okv24 / tot24) if tot24 > 0 else None
    api_failed_pct = (100.0 * api_failed24 / tot24) if tot24 > 0 else None
    stale_pct = (100.0 * stale24 / tot24) if tot24 > 0 else None

    # latência p90 (somente sucessos) — recorte 24h (do JSONL) para ser comparável ao checklist de LIVE
    p90_call = None
    p50_call = None
    p90_call_24h = None
    p50_call_24h = None
    n_succ_24h = None
    p50_queue_24h = None
    p90_queue_24h = None
    p50_queue_all = None
    try:
        blk = ((kpi_ok.get("timing_ms") or {}).get("call_to_done") or {})
        p90_call = int(blk.get("p90") or 0) or None
        p50_call = int(blk.get("p50") or 0) or None
    except Exception:
        p90_call = None
        p50_call = None
    try:
        blkq = ((kpi_ok.get("timing_ms") or {}).get("queue_delay") or {})
        p50_queue_all = int(blkq.get("p50") or 0) or None
    except Exception:
        p50_queue_all = None
    try:
        blk24 = ((kpi_ok_24h.get("timing_ms") or {}).get("call_to_done") or {}) if isinstance(kpi_ok_24h, dict) else {}
        p90_call_24h = int(blk24.get("p90") or 0) or None
        p50_call_24h = int(blk24.get("p50") or 0) or None
        n_succ_24h = int(blk24.get("n") or 0) if blk24.get("n") is not None else None
    except Exception:
        p90_call_24h = None
        p50_call_24h = None
        n_succ_24h = None
    try:
        blkq24 = ((kpi_ok_24h.get("timing_ms") or {}).get("queue_delay") or {}) if isinstance(kpi_ok_24h, dict) else {}
        p50_queue_24h = int(blkq24.get("p50") or 0) or None
        p90_queue_24h = int(blkq24.get("p90") or 0) or None
    except Exception:
        p50_queue_24h = None
        p90_queue_24h = None

    gaps15 = None
    try:
        # força janela 24h; se não houver amostra, deixa None (não compara com threshold).
        src = gaps24 if isinstance(gaps24, dict) else None
        gaps15 = int(src.get("gaps_gt_900s")) if isinstance(src, dict) and src.get("gaps_gt_900s") is not None else None
    except Exception:
        gaps15 = None

    # checks
    chk_allow = bool(allow_live)
    chk_okv = (ok_valid_pct is not None and float(ok_valid_pct) >= float(thr_ok_valid_pct))
    chk_api = (api_failed_pct is not None and float(api_failed_pct) <= float(thr_api_failed_pct))
    chk_stale = (stale_pct is not None and float(stale_pct) <= float(thr_stale_pct))
    chk_gap = (gaps15 is None) or (int(gaps15) <= int(thr_gaps_15m))
    chk_p90 = (p90_call_24h is None) or (int(p90_call_24h) <= int(thr_p90_ms))
    chk_open = int(err_open) <= int(thr_open_betslips)
    chk_pmms = int(err_pmms) <= int(thr_no_pmms)

    s0.append("| Critério | Atual | Alvo | Status |\n|---|---:|---:|---|\n")
    s0.append(f"| Live liberado (`EXECUTOR_ALLOW_LIVE`) | `{allow_live}` | `True` | **{_fmt_status(chk_allow)}** |\n")
    s0.append(f"| OK_valid/total (24h, DB) | {_fmt_num(ok_valid_pct,1)}% | ≥{_fmt_num(thr_ok_valid_pct,1)}% | **{_fmt_status(chk_okv)}** |\n")
    s0.append(f"| API_FAILED/total (24h, DB) | {_fmt_num(api_failed_pct,1)}% | ≤{_fmt_num(thr_api_failed_pct,1)}% | **{_fmt_status(chk_api)}** |\n")
    s0.append(f"| STALE_QUEUE_WAIT/total (24h, DB) | {_fmt_num(stale_pct,1)}% | ≤{_fmt_num(thr_stale_pct,1)}% | **{_fmt_status(chk_stale)}** |\n")
    if pmm_consults_24h is not None and no_pmms_24h is not None and no_pmms_rate_24h is not None:
        s0.append(
            f"| `No PMMs received` (24h, DB) | {int(no_pmms_24h)} / {int(pmm_consults_24h)} ({_fmt_num(no_pmms_rate_24h,2)}%) | ≤{int(thr_no_pmms)} (abs) | **{_fmt_status(chk_pmms)}** |\n"
        )
    else:
        s0.append(f"| `No PMMs received` (24h, DB) | {int(err_pmms)} | ≤{int(thr_no_pmms)} | **{_fmt_status(chk_pmms)}** |\n")
        s0.append("| `No PMMs` / `PMM-consults` (24h, DB) | — | — | — |\n")
    s0.append(f"| `too_many_open_betslips` (24h, DB) | {int(err_open)} | ≤{int(thr_open_betslips)} | **{_fmt_status(chk_open)}** |\n")
    s0.append(
        f"| Latência p90 `call_to_done_ms` (24h; sucessos) | {_fmt_num(p90_call_24h,0)}ms | ≤{int(thr_p90_ms)}ms | **{_fmt_status(chk_p90)}** |\n"
    )
    s0.append(f"| Latência p50 `call_to_done_ms` (24h; sucessos) | {_fmt_num(p50_call_24h,0)}ms | — | — |\n")
    s0.append(f"| n sucessos no JSONL (24h) | {n_succ_24h if n_succ_24h is not None else '—'} | — | — |\n")
    s0.append(f"| Gaps >15min no executor_jsonl (24h; proxy) | {gaps15 if gaps15 is not None else '—'} | ≤{int(thr_gaps_15m)} | **{_fmt_status(chk_gap)}** |\n")
    s0.append("\n")

    # Diagnóstico WS para "No PMMs received": ajuda a distinguir "timeout curto" vs "WS morto".
    try:
        if isinstance(pmm_ws_diag, dict) and int(pmm_ws_diag.get("no_pmms_total") or 0) > 0:
            s0.append("**Diagnóstico de WebSocket (quando ocorre `No PMMs received`)**\n\n")
            thr_ms = int(pmm_ws_diag.get("ws_stale_ms_thr") or 0)
            n0 = int(pmm_ws_diag.get("no_pmms_total") or 0)
            nst = int(pmm_ws_diag.get("no_pmms_ws_stale") or 0)
            p50 = pmm_ws_diag.get("no_pmms_ws_age_ms_median")
            p90 = pmm_ws_diag.get("no_pmms_ws_age_ms_p90")
            mx = pmm_ws_diag.get("no_pmms_ws_age_ms_max")
            s0.append("| Métrica | Valor |\n|---|---:|\n")
            s0.append(f"| `No PMMs` total (24h) | {n0} |\n")
            s0.append(f"| `No PMMs` com WS stale (ws_age_ms≥{thr_ms} ou NULL) | {nst} ({_fmt_num((100.0*nst/n0) if n0 else None,2)}%) |\n")
            s0.append(f"| ws_age_ms p50 / p90 / max | {_fmt_num(p50,0)} / {_fmt_num(p90,0)} / {_fmt_num(mx,0)} |\n")
            s0.append("\n")
    except Exception:
        pass

    # Diagnóstico curto de causa (latência): quando p50 sobe, quase sempre é fila (queue_delay_ms) e/ou timeout de PMM/relógio.
    try:
        if p50_queue_24h is not None and int(p50_queue_24h) > 500:
            s0.append(
                f"**Diagnóstico (latência)**: p50 `queue_delay_ms` (24h) = `{int(p50_queue_24h)}ms` (p90 `{_fmt_num(p90_queue_24h,0)}ms`)"
            )
            if p50_queue_all is not None:
                s0.append(f" vs baseline `{_fmt_num(p50_queue_all,0)}ms`.\n")
            else:
                s0.append(".\n")
            s0.append(
                "- Interpretação: há backlog na fila do executor (workers/concurrency insuficiente ou bursts). Mitigação típica: aumentar `EXECUTOR_WORKERS` e/ou reduzir bursts no bridge.\n\n"
            )
    except Exception:
        pass

    # Decomposição extra (call_to_done = queue_delay + post + overhead): ajuda a distinguir "fila" vs "POST lento" (UI/anti-bot)
    try:
        blk_post_24 = ((kpi_ok_24h.get("timing_ms") or {}).get("post") or {}) if isinstance(kpi_ok_24h, dict) else {}
        p50_post_24h = int(blk_post_24.get("p50") or 0) or None
        p90_post_24h = int(blk_post_24.get("p90") or 0) or None
        if p50_call_24h is not None and p50_queue_24h is not None and p50_post_24h is not None:
            p50_over = int(max(0, int(p50_call_24h) - int(p50_queue_24h) - int(p50_post_24h)))
            s0.append("**Decomposição (latência; p50, 24h; sucessos)**\n\n")
            s0.append("| Componente | p50 |\n|---|---:|\n")
            s0.append(f"| call_to_done_ms | {_fmt_num(p50_call_24h,0)}ms |\n")
            s0.append(f"| queue_delay_ms | {_fmt_num(p50_queue_24h,0)}ms |\n")
            s0.append(f"| post_ms | {_fmt_num(p50_post_24h,0)}ms |\n")
            s0.append(f"| overhead (aprox.) | {_fmt_num(p50_over,0)}ms |\n")
            s0.append("\n")
            # Aciona recomendações específicas quando post_ms é o gargalo
            if int(p50_post_24h) >= 2000:
                s0.append(
                    "- Leitura: `post_ms` alto sugere lentidão no passo de POST/submit (UI/JS/anti-bot/latência de página), não fila. Mitigação típica: estabilizar sessão, reduzir re-login, otimizar fluxo do scraper e revisar timeout/retries.\n\n"
                )
    except Exception:
        pass

    hard_fails = []
    if not chk_allow:
        hard_fails.append("LIVE bloqueado (`EXECUTOR_ALLOW_LIVE=0`)")
    if ok_valid_pct is None or not chk_okv:
        hard_fails.append("conversão `OK_valid/total` baixa")
    if api_failed_pct is None or not chk_api:
        hard_fails.append("taxa de `API_FAILED` alta")
    if stale_pct is None or not chk_stale:
        hard_fails.append("taxa de `STALE_QUEUE_WAIT` alta")
    if not chk_pmms:
        hard_fails.append("erros `No PMMs received` presentes")
    if not chk_open:
        hard_fails.append("erros `too_many_open_betslips` presentes")

    verdict = "APTO (com cautela)" if not hard_fails else "NÃO APTO"
    s0.append(f"**Veredito operacional (observação — sem alteração de risco)**: **{verdict}**\n\n")
    if hard_fails:
        s0.append("**Observação / Evidência / Status / Limitação**\n\n")
        for x in hard_fails[:8]:
            s0.append(f"- Observação: {x}\n")
            s0.append("  - Evidência: KPIs das últimas 24h no audit/executor\n")
            s0.append("  - Status: WATCH\n")
            s0.append("  - Limitação: este Daily **não recomenda** alteração de stake, exposição, filtro ou threshold\n")
        s0.append("\n")
        s0.append(
            "> **APÊNDICE DE PESQUISA — NÃO OPERACIONAL** (antigas 'recomendações' de desbloqueio LIVE "
            "foram removidas do núcleo; não alterar policy/stake a partir deste relatório).\n\n"
        )

    # leitura executiva: observação sem recomendações de risco
    s0.append(
        "\n**Observações operacionais (sem sizing / sem alteração de stake)**\n\n"
        "- Formato: Observação · Evidência · Status · Limitação.\n"
        "- Conversão / API_FAILED / STALE: monitorar KPIs; **não** alterar thresholds neste Daily.\n"
        "- Cap / open betslips: evidência no audit; **não** alterar caps a partir deste relatório.\n"
        "- Slippage / ROI por bucket: diagnóstico; **não** afirmar edge nem mudar policy.\n"
        "- Estudos Kelly/sizing/contrafactuais: ver **APÊNDICE DE PESQUISA — NÃO OPERACIONAL**.\n\n"
    )

    # ------------------------------------------------------------
    # Carteira (active_keys): delta vs policy anterior + marginais OOS por key
    # ------------------------------------------------------------
    try:
        cur_step = policy_last_step if isinstance(policy_last_step, dict) else None
        cur_keys = set(cur_step.get("active_keys") or []) if cur_step else set()
        prev_pol = _pick_prev_policy_file(cfg.wf_policy_history_dir, cur_day=str(day))
        prev_step = _load_wf_policy_last_step(prev_pol) if prev_pol else None
        prev_keys = set(prev_step.get("active_keys") or []) if prev_step else set()
        if cur_keys and prev_keys:
            entered = sorted(list(cur_keys - prev_keys))
            exited = sorted(list(prev_keys - cur_keys))
            s0.append("\n**Carteira (policy current): keys que entraram/sairam vs policy anterior**\n\n")
            s0.append(f"- Policy anterior: `{prev_pol}`\n")
            s0.append(f"- Δ keys: `{len(prev_keys)}` → `{len(cur_keys)}` (entraram `{len(entered)}`, saíram `{len(exited)}`)\n\n")
            if entered:
                s0.append("- **Entraram**:\n")
                for k in entered[:40]:
                    s0.append(f"  - `{k}`\n")
                if len(entered) > 40:
                    s0.append(f"  - … (+{len(entered)-40})\n")
                s0.append("\n")
            if exited:
                s0.append("- **Saíram**:\n")
                for k in exited[:40]:
                    s0.append(f"  - `{k}`\n")
                if len(exited) > 40:
                    s0.append(f"  - … (+{len(exited)-40})\n")
                s0.append("\n")

            # Marginal por key (OOS): extraído do texto OOS atual (base_md)
            if oos_txt and isinstance(oos_txt, str):
                tbl_md, rows = _extract_md_table(oos_txt, header_startswith="| Combinação (key) | Turnover 30d |")
                hdr = _md_table_header_cols(tbl_md)
                if rows and hdr:
                    idx = {c: i for i, c in enumerate(hdr)}

                    def _f(x: str) -> Optional[float]:
                        """
                        Parse numérico robusto vindo de Markdown (OOS):
                        - Aceita formatos en-US (1,234.56) e pt-BR (1.234,56)
                        - Preserva decimais (não faz replace('.') indiscriminado)
                        - Aceita percentuais ("49.54%")
                        """
                        try:
                            t = str(x or "").strip().replace("−", "-")
                            if not t:
                                return None
                            t = t.replace("%", "").strip()
                            # remove espaços e símbolos comuns
                            t = t.replace(" ", "")
                            # Se contém ambos '.' e ',', inferimos qual é o separador decimal pelo último.
                            if "." in t and "," in t:
                                if t.rfind(".") > t.rfind(","):
                                    # en-US: ',' milhar, '.' decimal
                                    t = t.replace(",", "")
                                else:
                                    # pt-BR: '.' milhar, ',' decimal
                                    t = t.replace(".", "").replace(",", ".")
                            else:
                                # Apenas ',' => assume decimal pt-BR
                                if "," in t and "." not in t:
                                    t = t.replace(",", ".")
                                # Apenas '.' => já é decimal en-US (mantém)
                            return float(t)
                        except Exception:
                            return None

                    mp: Dict[str, Dict[str, Any]] = {}
                    for cols in rows:
                        if not cols or len(cols) < len(hdr):
                            continue
                        k0 = str(cols[idx.get("Combinação (key)", 0)]).strip()
                        if not k0 or k0.lower().startswith("combina"):
                            continue
                        mp[k0] = {
                            "turn_30d": _f(cols[idx.get("Turnover 30d", 1)]) if "Turnover 30d" in idx else None,
                            "share_turn_pct": _f(cols[idx.get("Share turnover", 2)]) if "Share turnover" in idx else None,
                            "profit_30d": _f(cols[idx.get("Lucro 30d (exp.)", 3)]) if "Lucro 30d (exp.)" in idx else None,
                            "share_profit_pct": _f(cols[idx.get("Share lucro", 4)]) if "Share lucro" in idx else None,
                            "roi_turn_pct": _f(cols[idx.get("ROI/turnover 30d", 5)]) if "ROI/turnover 30d" in idx else None,
                        }

                    if mp and (entered or exited):
                        s0.append("**OOS marginal por key (30d exp.) — entraram/sairam**\n\n")
                        s0.append("| Key | Status | Turnover 30d | Share turn | Lucro 30d (exp.) | Share lucro | ROI/turn |\n")
                        s0.append("|---|---|---:|---:|---:|---:|---:|\n")
                        for k in entered:
                            v = mp.get(k) or {}
                            s0.append(
                                f"| `{k}` | entrou | {_fmt_num(v.get('turn_30d'),2)} | {_fmt_num(v.get('share_turn_pct'),2)}% | {_fmt_num(v.get('profit_30d'),2)} | {_fmt_num(v.get('share_profit_pct'),2)}% | {_fmt_num(v.get('roi_turn_pct'),2)}% |\n"
                            )
                        for k in exited:
                            v = mp.get(k) or {}
                            s0.append(
                                f"| `{k}` | saiu | {_fmt_num(v.get('turn_30d'),2)} | {_fmt_num(v.get('share_turn_pct'),2)}% | {_fmt_num(v.get('profit_30d'),2)} | {_fmt_num(v.get('share_profit_pct'),2)}% | {_fmt_num(v.get('roi_turn_pct'),2)}% |\n"
                            )
                        s0.append("\n")

                        # shares top para contexto
                        try:
                            xs = []
                            for k, v in mp.items():
                                st = _f(str(v.get("share_turn_pct") or "")) if isinstance(v, dict) else None
                                sp = _f(str(v.get("share_profit_pct") or "")) if isinstance(v, dict) else None
                                xs.append((k, st, sp))
                            top_turn = sorted([x for x in xs if x[1] is not None], key=lambda x: float(x[1]), reverse=True)[:10]
                            top_prof = sorted([x for x in xs if x[2] is not None], key=lambda x: float(x[2]), reverse=True)[:10]
                            if top_turn:
                                s0.append("- **Top 10 por share de turnover (OOS 30d exp.)**: " + ", ".join([f"`{k}`({float(st):.2f}%)" for k, st, _ in top_turn]) + "\n")
                            if top_prof:
                                s0.append("- **Top 10 por share de lucro (OOS 30d exp.)**: " + ", ".join([f"`{k}`({float(sp):.2f}%)" for k, _, sp in top_prof]) + "\n")
                            s0.append("\n")
                        except Exception:
                            pass

            # Proxy do efeito na sensibilidade de banca: Δ da tabela 12.2b vs report_base do dia anterior (se existir)
            try:
                prev_day = (ts - timedelta(days=1)).astimezone(timezone.utc).strftime("%Y%m%d")
                prev_base = cfg.out_dir / prev_day / "report_base.md"
                if prev_base.exists() and oos_txt:
                    prev_txt = prev_base.read_text(encoding="utf-8", errors="ignore")
                    cur_blk = _extract_md_block(oos_txt, start="### 12.2b Sensibilidade por banca", until_any=["### 12.2c", "### 12.2d", "### 12.3", "## 10)", "## 11)"])
                    prev_blk = _extract_md_block(prev_txt, start="### 12.2b Sensibilidade por banca", until_any=["### 12.2c", "### 12.2d", "### 12.3", "## 10)", "## 11)"])

                    def _parse_sens(bl: str) -> Dict[float, Dict[str, Any]]:
                        out = {}
                        for ln in (bl or "").splitlines():
                            if not ln.startswith("|") or ln.strip().startswith("|---"):
                                continue
                            cols = [c.strip() for c in ln.strip().strip("|").split("|")]
                            if len(cols) < 6 or cols[0].lower().startswith("banca"):
                                continue
                            def _f2(s: str) -> Optional[float]:
                                # reusa o parser robusto acima (mesma semântica)
                                return _f(s)
                            bank = _f2(cols[0])
                            turn = _f2(cols[1])
                            prof = _f2(cols[2])
                            roi = _f2(cols[4])
                            if bank is None:
                                continue
                            out[float(bank)] = {"turn": turn, "profit": prof, "roi": roi}
                        return out

                    curm = _parse_sens(cur_blk)
                    prevm = _parse_sens(prev_blk)
                    inter = sorted(set(curm.keys()) & set(prevm.keys()))
                    if inter:
                        s0.append("**Efeito na sensibilidade de banca (proxy): Δ 12.2b vs dia anterior**\n\n")
                        s0.append(f"- Base anterior: `{prev_base}`\n\n")
                        s0.append("| Banca(ref) | Δ Turnover 30d | Δ Lucro 30d (exp.) | Δ ROI/banca 30d |\n")
                        s0.append("|---:|---:|---:|---:|\n")
                        for b in inter[:12]:
                            s0.append(
                                f"| {int(b)} | {_fmt_num((curm[b].get('turn') or 0.0)-(prevm[b].get('turn') or 0.0),2)} | "
                                f"{_fmt_num((curm[b].get('profit') or 0.0)-(prevm[b].get('profit') or 0.0),2)} | "
                                f"{_fmt_num((curm[b].get('roi') or 0.0)-(prevm[b].get('roi') or 0.0),2)}% |\n"
                            )
                        s0.append("\n")
            except Exception:
                pass
    except Exception:
        pass

    # --- Seção 1: Resultados reais (shadow/live) ---
    s1 = []
    s1.append("## 1) Resultados reais (shadow/live)\n\n")

    # KPIs por recortes (diário/semana/mês) — quando há série
    if acct_series is not None:
        # preferir P&L filtrado (exclui depósitos/saques) quando existir
        pnls = acct_series.pnl_by_day_filtered or acct_series.pnl_by_day
        # semana corrente (por dia)
        try:
            now = _utcnow().astimezone(timezone.utc)
            # usa o maior dia presente como "today" do dataset para evitar mismatch tz
            days_sorted = sorted(pnls.keys())
            today = days_sorted[-1] if days_sorted else now.date().isoformat()
            ws = _week_start_iso(today) or today
            cur_week_days = [d for d in days_sorted if ws <= d <= today]
        except Exception:
            cur_week_days = []
            today = None
            ws = None

        s1.append("**P&L real por dia (semana corrente)**\n\n")
        s1.append("| Dia | P&L |\n|---|---:|\n")
        for d in (cur_week_days or [])[-14:]:
            s1.append(f"| {d} | {_fmt_num(pnls.get(d), 2)} |\n")
        s1.append("\n")

        # Transparência: regras efetivas de seleção e sizing (operacional)
        try:
            s1.append("**Regras efetivas (seleção + sizing) — aplicadas na execução**\n\n")

            # Policy WF (do daily) — define universo/combos e filtros de mercado
            wf = policy_wf if isinstance(policy_wf, dict) else {}
            if wf:
                s1.append("| Policy (WF) | Valor |\n|---|---|\n")
                for k in [
                    "train_mode",
                    "train_days",
                    "test_days",
                    "step_days",
                    "min_matches",
                    "key_by_league",
                    "key_by_league_scope",
                    "ah_max_abs_line",
                    "ah_scope",
                    "liquidity_mode",
                    "liquidity_scope",
                    "liquidity_min_limit",
                ]:
                    if k in wf:
                        s1.append(f"| {k} | `{wf.get(k)}` |\n")
                s1.append("\n")
                if str(wf.get("train_mode") or "").strip().lower() == "expanding":
                    s1.append(
                        "_Nota (WF expanding): `train_days/test_days/step_days` são parâmetros do calendário do walk-forward. "
                        "O **intervalo real** do treino/teste do step vigente é o que aparece logo abaixo em `train=... | test=...`._\n\n"
                    )
            if isinstance(policy_last_step, dict):
                s1.append(f"- Último step (janelas): `train={policy_last_step.get('train')}` | `test={policy_last_step.get('test')}`\n")
                s1.append(f"- Ativas: `keys={len(list(policy_last_step.get('active_keys') or []))}` | `base={len(list(policy_last_step.get('active_keys_base') or []))}`\n\n")

            # Risk params (manual) — governa budget/caps por jogo e sizing base
            rp_path = os.getenv("BRIDGE_RISK_PARAMS_JSON", "").strip()
            rp = _read_json(Path(rp_path)) if rp_path else None
            if isinstance(rp, dict) and rp:
                s1.append("| Risk params (manual) | Valor |\n|---|---|\n")
                for k in [
                    "budget_back_frac",
                    "budget_lay_frac",
                    "cap_signal_frac",
                    "cap_event_back_frac",
                    "cap_event_lay_frac",
                    "stake_pct_of_limit",
                    "stake_cap_abs",
                ]:
                    if k in rp:
                        s1.append(f"| {k} | `{rp.get(k)}` |\n")
                s1.append("\n")

            # Bridge/executor: modo e fontes principais
            s1.append("| Runtime | Valor |\n|---|---|\n")
            for k in [
                "EXECUTOR_ALLOW_LIVE",
                "BRIDGE_USE_WF_BUDGET",
                "BRIDGE_ENFORCE_WF_FILTERS",
                "BRIDGE_WF_RISK_MODE_OVERRIDE",
                "BRIDGE_BANKROLL_REF",
                "BRIDGE_BANKROLL_JSON",
                "BRIDGE_POLICY_JSON",
                "BRIDGE_RISK_PARAMS_JSON",
            ]:
                v = os.getenv(k, "")
                if v:
                    s1.append(f"| {k} | `{v}` |\n")
            s1.append("\n")

            s1.append(
                "_Nota: filtro **AH** é por **|linha|** (ex.: `ah_max_abs_line=2.0` significa |line|≤2.0), não por odds; odds médias >2 podem ocorrer mesmo com AH válido._\n\n"
            )
        except Exception:
            pass

        # drawdown e sharpe (curto, usando a própria janela da série)
        dd = _max_drawdown({k: float(v) for k, v in pnls.items()})
        dd_w = _max_drawdown(_agg_by_week({k: float(v) for k, v in pnls.items()}))
        dd_m = _max_drawdown(_agg_by_month({k: float(v) for k, v in pnls.items()}))
        br_real = None
        try:
            br_real = float(acct.get("balance_current")) if isinstance(acct, dict) and acct.get("balance_current") is not None else None
        except Exception:
            br_real = None
        br_theo = None
        try:
            br_theo = float(cfg.kelly_bankroll) if str(cfg.kelly_bankroll).strip() else None
        except Exception:
            br_theo = None

        s1.append("**Risco/consistência (a partir do P&L diário)**\n\n")
        s1.append("| Métrica | Valor |\n|---|---:|\n")
        s1.append(f"| Max drawdown (diário, monetário) | {_fmt_num(dd.get('mdd'), 2)} |\n")
        s1.append(f"| Max drawdown (semanal, monetário) | {_fmt_num(dd_w.get('mdd'), 2)} |\n")
        s1.append(f"| Max drawdown (mensal, monetário) | {_fmt_num(dd_m.get('mdd'), 2)} |\n")
        if dd.get("from_day") and dd.get("to_day"):
            s1.append(f"| Janela do DD | {dd.get('from_day')} → {dd.get('to_day')} |\n")
        if br_real:
            sh = _sharpe_annualized(pnls, bankroll_ref=float(br_real))
            s1.append(f"| Sharpe anualizado (vs banca real) | {_fmt_num(sh, 2)} |\n")
            # ROI por banca (recortes simples)
            try:
                pnl_week = float(acct.get("pnl_filtered_week") if acct.get("pnl_filtered_week") is not None else acct.get("pnl_week") or 0.0)
                pnl_month = float(acct.get("pnl_filtered_month") if acct.get("pnl_filtered_month") is not None else acct.get("pnl_month") or 0.0)
                s1.append(f"| ROI/banca real (semana) | {_fmt_num((pnl_week/float(br_real))*100.0, 2)}% |\n")
                s1.append(f"| ROI/banca real (mês) | {_fmt_num((pnl_month/float(br_real))*100.0, 2)}% |\n")
            except Exception:
                pass
        if br_theo:
            sh2 = _sharpe_annualized(pnls, bankroll_ref=float(br_theo))
            s1.append(f"| Sharpe anualizado (vs banca teórica) | {_fmt_num(sh2, 2)} |\n")
            try:
                pnl_week = float(acct.get("pnl_filtered_week") if acct.get("pnl_filtered_week") is not None else acct.get("pnl_week") or 0.0)
                pnl_month = float(acct.get("pnl_filtered_month") if acct.get("pnl_filtered_month") is not None else acct.get("pnl_month") or 0.0)
                s1.append(f"| ROI/banca teórica (semana; ref={_fmt_num(br_theo,0)}) | {_fmt_num((pnl_week/float(br_theo))*100.0, 2)}% |\n")
                s1.append(f"| ROI/banca teórica (mês; ref={_fmt_num(br_theo,0)}) | {_fmt_num((pnl_month/float(br_theo))*100.0, 2)}% |\n")
            except Exception:
                pass
        s1.append("\n")

        # semanas fechadas do mês corrente (visão executiva)
        try:
            weeks = _agg_by_week({k: float(v) for k, v in pnls.items()})
            # identifica mês corrente pelo último dia do dataset
            days_sorted = sorted(pnls.keys())
            if days_sorted:
                mk_cur = _month_key(days_sorted[-1])
                # semanas cujo week_start está no mesmo mês e não é a semana corrente
                ws_cur = _week_start_iso(days_sorted[-1])
                rows = [(ws, val) for ws, val in weeks.items() if _month_key(ws) == mk_cur and ws != ws_cur]
                if rows:
                    s1.append("**Semanas anteriores fechadas (mês corrente)**\n\n")
                    s1.append("| Semana (start) | P&L |\n|---|---:|\n")
                    for ws, val in rows[-6:]:
                        s1.append(f"| {ws} | {_fmt_num(val, 2)} |\n")
                    s1.append("\n")
        except Exception:
            pass
    else:
        s1.append("_Sem série de accounting disponível para métricas diárias/Sharpe/DD (ver 99.1)._ \n\n")

    adh_day = adh_short if isinstance(adh_short, dict) else None
    adh_slip = adh_long if isinstance(adh_long, dict) else (adh_day if isinstance(adh_day, dict) else None)

    # execução (contagens + stake médio) via aderência (janela curta)
    if isinstance(exec_min, dict) and isinstance(exec_min.get("by_type"), dict):
        try:
            s1.append("**Execução — métricas mínimas por tipo (Back/Lay × Pre/In; janela curta)**\n\n")
            s1.append(
                "| Tipo | #ordens | #eventos_jsonl | #linhas_api | #jogos | Valor em risco ($) | Ticket médio ($/ordem) | Stake total ($) | #liq | #pend | P&L (liq, $) | ROI% (liq) |\n"
            )
            s1.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            order = ["Back_Pre", "Back_In", "Lay_Pre", "Lay_In"]
            for k in order:
                r = exec_min.get("by_type", {}).get(k) if isinstance(exec_min.get("by_type"), dict) else None
                if not isinstance(r, dict):
                    continue
                s1.append(
                    f"| {k.replace('_', ' ')} | {int(r.get('n_orders') or 0)} | {int(r.get('n_bets') or 0)} | {int(r.get('n_bet_lines_api') or 0)} | {int(r.get('n_matches') or 0)} | "
                    f"{_fmt_num(r.get('amount_risk_sum'), 2)} | {_fmt_num(r.get('amount_risk_avg_per_order'), 2)} | "
                    f"{_fmt_num(r.get('stake_sum'), 2)} | {int(r.get('n_settled') or 0)} | {int(r.get('n_unsettled') or 0)} | {_fmt_num(r.get('pnl_real_sum_settled') or r.get('pnl_sum_settled'), 2)} | "
                    f"{_fmt_pct(r.get('roi_pct_settled'))} |\n"
                )
            tot = exec_min.get("total") if isinstance(exec_min.get("total"), dict) else {}
            if isinstance(tot, dict) and tot:
                s1.append(
                    f"| **TOTAL** | **{int(tot.get('n_orders') or 0)}** | **{int(tot.get('n_bets') or 0)}** | **{int(tot.get('n_bet_lines_api') or 0)}** | **{int(tot.get('n_matches') or 0)}** | "
                    f"**{_fmt_num(tot.get('amount_risk_sum'), 2)}** | **{_fmt_num(tot.get('amount_risk_avg_per_order'), 2)}** | "
                    f"**{_fmt_num(tot.get('stake_sum'), 2)}** | **{int(tot.get('n_settled') or 0)}** | **{int(tot.get('n_unsettled') or 0)}** | **{_fmt_num(tot.get('pnl_real_sum_settled') or tot.get('pnl_sum_settled'), 2)}** | "
                    f"**{_fmt_pct(tot.get('roi_pct_settled'))}** |\n"
                )
            s1.append("\n")
        except Exception:
            pass
    else:
        # Ajuda a explicar o caso "apostado mas ROI/resultado vazio": aqui usa filtro por status.
        try:
            only_st = str(os.getenv("DAILY_EXEC_MIN_BY_TYPE_ONLY_STATUS", "LIVE_OK"))
            s1.append(
                f"_Execução — métricas mínimas por tipo: sem dados no recorte (provável filtro `DAILY_EXEC_MIN_BY_TYPE_ONLY_STATUS={only_st}`)._\n\n"
            )
        except Exception:
            pass

    if isinstance(adh_day, dict) and isinstance(adh_day.get("per_day"), list) and adh_day.get("per_day"):
        s1.append("**Execução (últimos dias; executor_jsonl + placares quando disponíveis)**\n\n")
        # Preparação (best-effort) para:
        # - P&L total por dia consistente com "post date UTC" (ledger)
        # - split Back Pre/In via join por order_id (Back LIVE_OK)
        try:
            days_utc_exec = {str(it.get("day") or "") for it in (adh_day.get("per_day") or []) if isinstance(it, dict) and str(it.get("day") or "")}
        except Exception:
            days_utc_exec = set()
        try:
            bal_csv = Path(str(acct.get("balance_csv") or "")).expanduser() if isinstance(acct, dict) else None
        except Exception:
            bal_csv = None

        acct_day_typ: Dict[str, Dict[str, Dict[str, Any]]] = {}
        acct_by_oid_day_typ: Dict[str, Dict[str, Dict[str, float]]] = {}
        acct_pnl_by_oid_total: Dict[str, float] = {}
        exec_by_oid_back: Dict[str, Dict[str, Any]] = {}
        audit_by_id: Dict[int, Dict[str, Any]] = {}
        if bal_csv and days_utc_exec and bal_csv.exists():
            try:
                acct_day_typ = _acct_amount_by_day_type_from_balance_csv(bal_csv, days_utc=set(days_utc_exec))
            except Exception:
                acct_day_typ = {}
            try:
                acct_by_oid_day_typ = _acct_amount_by_order_day_by_type_from_balance_csv(bal_csv, days_utc=set(days_utc_exec))
            except Exception:
                acct_by_oid_day_typ = {}
        # P&L por ordem (order_id) acumulado no ledger (inclui tipos P&L-like; exclui dep/saque/etc.)
        if bal_csv and bal_csv.exists():
            try:
                acct_pnl_by_oid_total = _acct_pnl_like_by_order_total_from_balance_csv(bal_csv)
            except Exception:
                acct_pnl_by_oid_total = {}
        if cfg.executor_jsonl.exists():
            try:
                exec_by_oid_back = _parse_executor_jsonl_back_live_orders(Path(str(cfg.executor_jsonl)))
            except Exception:
                exec_by_oid_back = {}
        # Para separar Back Pre vs Back In corretamente, precisamos de kickoff_time/is_live do audit (DB).
        # Sem isso, `ExecutionRequest.is_live` do executor NÃO é in-play (é "modo LIVE").
        try:
            if exec_by_oid_back:
                from storage.database import Database  # local import para não exigir DB em modo "report-only"

                audit_ids = _extract_audit_ids_from_exec_by_oid(exec_by_oid_back)
                if audit_ids:
                    db = Database()
                    await db.connect()
                    try:
                        audit_by_id = await _fetch_audit_rows_for_ids_daily(db, audit_ids)
                    finally:
                        try:
                            await db.close()
                        except Exception:
                            pass
        except Exception:
            audit_by_id = {}

        def _acct_is_excluded_type(tl: str) -> bool:
            t = str(tl or "").strip().lower()
            return any(k in t for k in ("deposit", "withdraw", "transfer", "top", "payment", "adjust", "bonus"))

        s1.append(
            "| Dia | Exec rows | Sucessos | LIVE_OK | DRY_OK | API_FAILED | N Back | N Lay | Apostado Back ($) | Apostado Lay stake ($) | Apostado Lay liab ($) | "
            "P&L total (acct; post date UTC) | ROI/$ (acct) | P&L Back (acct; oid join) | P&L Back Pre (acct; oid) | P&L Back In (acct; oid) | Δ (acct_total - acct_back_oid) | Cobertura oids% (Back) | "
            "P&L (placar) | ROI/$ (placar) | P&L Back | ROI Back | P&L Lay | ROI Lay/liab | ROI Lay/stake |\n"
        )
        s1.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for it in adh_day.get("per_day") or []:
            if not isinstance(it, dict):
                continue
            ex = it.get("execution") if isinstance(it.get("execution"), dict) else {}
            sc = ex.get("status_counts") if isinstance(ex.get("status_counts"), dict) else {}
            back = ex.get("back") if isinstance(ex.get("back"), dict) else {}
            lay = ex.get("lay") if isinstance(ex.get("lay"), dict) else {}
            pnl_back = back.get("pnl_sum")
            pnl_lay = lay.get("pnl_sum")
            pnl_total_placar = (float(pnl_back or 0.0) + float(pnl_lay or 0.0)) if (pnl_back is not None or pnl_lay is not None) else None
            st_back = back.get("stake_sum")
            st_lay = lay.get("stake_sum")
            liab_lay = lay.get("liability_sum")
            # ROI/$ só faz sentido na mesma base do P&L (placar), então usa denominadores "cobertos"
            st_back_cov = back.get("stake_sum_cov")
            st_lay_cov = lay.get("stake_sum_cov")
            st_total_cov = (float(st_back_cov or 0.0) + float(st_lay_cov or 0.0)) if (st_back_cov is not None or st_lay_cov is not None) else None
            roi_dol = (float(pnl_total_placar) / float(st_total_cov) * 100.0) if (pnl_total_placar is not None and st_total_cov and float(st_total_cov) > 0) else None

            # P&L real (accounting) por dia:
            # 1) Preferir ledger por post date UTC (para casar com dayk do adherence em UTC)
            # 2) Fallback: mapa do accounting_daily_report (pode estar em outro TZ)
            dayk = str(it.get("day") or "")
            pnl_acct = None
            try:
                if dayk and isinstance(acct_day_typ, dict) and dayk in acct_day_typ:
                    blk = acct_day_typ.get(dayk) if isinstance(acct_day_typ.get(dayk), dict) else {}
                    if blk:
                        pnl_acct = float(
                            sum(float(v.get("amount_sum") or 0.0) for k, v in blk.items() if isinstance(v, dict) and (not _acct_is_excluded_type(str(k))))
                        )
            except Exception:
                pnl_acct = None
            if pnl_acct is None:
                try:
                    if isinstance(acct, dict):
                        mp = acct.get("pnl_by_day_filtered_recent") if isinstance(acct.get("pnl_by_day_filtered_recent"), dict) else (
                            acct.get("pnl_by_day_recent") if isinstance(acct.get("pnl_by_day_recent"), dict) else {}
                        )
                        if isinstance(mp, dict) and dayk in mp:
                            pnl_acct = float(mp.get(dayk) or 0.0)
                except Exception:
                    pnl_acct = None

            # Accounting ROI/$: usa stake_total do Back como denominador (operacional), porque o balance.csv não expõe stake/liability por lado.
            roi_acct_dol = None
            try:
                if pnl_acct is not None and st_back is not None and float(st_back) > 0:
                    roi_acct_dol = float(pnl_acct) / float(st_back) * 100.0
            except Exception:
                roi_acct_dol = None

            # Split Back Pre/In (accounting) por join em order_id (Back LIVE_OK).
            # Isso deixa de “alocar” o P&L total do dia e passa a medir Back-only por regime.
            pnl_acct_back_oid = None
            pnl_acct_back_pre = None
            pnl_acct_back_in = None
            pnl_acct_back_delta = None
            cov_oid_pct = None
            try:
                if dayk and exec_by_oid_back and acct_by_oid_day_typ:
                    sp = _split_back_acct_pnl_pre_in_by_order_id(
                        exec_by_oid=exec_by_oid_back,
                        acct_by_oid_day_typ=acct_by_oid_day_typ,
                        day_utc=dayk,
                        audit_by_id=(audit_by_id or None),
                        include_types=None,  # P&L-like: inclui void/refund/cancel quando existirem; exclui depósitos/saques/etc.
                    )
                    if isinstance(sp, dict) and int(sp.get("n_total") or 0) > 0:
                        pnl_acct_back_oid = sp.get("pnl_total")
                        pnl_acct_back_pre = sp.get("pnl_pre")
                        pnl_acct_back_in = sp.get("pnl_in")
                        cov_oid_pct = sp.get("coverage_n_pct")
                        try:
                            if pnl_acct is not None and pnl_acct_back_oid is not None:
                                pnl_acct_back_delta = float(pnl_acct) - float(pnl_acct_back_oid)
                        except Exception:
                            pnl_acct_back_delta = None
            except Exception:
                pnl_acct_back_oid = None
                pnl_acct_back_pre = None
                pnl_acct_back_in = None
                pnl_acct_back_delta = None
                cov_oid_pct = None
            s1.append(
                f"| {it.get('day')} | {int(ex.get('n_exec_rows') or 0)} | {int(ex.get('n_exec_success') or 0)} | {int(sc.get('LIVE_OK') or 0)} | {int(sc.get('DRY_OK') or 0)} | "
                f"{int(sc.get('API_FAILED') or 0)} | {int(back.get('n_success') or 0)} | {int(lay.get('n_success') or 0)} | "
                f"{_fmt_num(st_back,2)} | {_fmt_num(st_lay,2)} | {_fmt_num(liab_lay,2)} | "
                f"{_fmt_num(pnl_acct,2)} | {_fmt_pct(roi_acct_dol)} | {_fmt_num(pnl_acct_back_oid,2)} | {_fmt_num(pnl_acct_back_pre,2)} | {_fmt_num(pnl_acct_back_in,2)} | {_fmt_num(pnl_acct_back_delta,2)} | {_fmt_pct(cov_oid_pct,1)} | "
                f"{_fmt_num(pnl_total_placar,2)} | {_fmt_pct(roi_dol)} | {_fmt_num(pnl_back,2)} | {_fmt_pct(back.get('roi_pct'))} | "
                f"{_fmt_num(pnl_lay,2)} | {_fmt_pct(lay.get('roi_pct_per_liability'))} | {_fmt_pct(ex.get('lay_roi_pct_per_stake'))} |\n"
            )
        s1.append("\n")

        s1.append(
            "_Nota: `P&L total (acct)` é calculado por **post date UTC** diretamente do `balance.csv` quando disponível (exclui depósitos/saques/etc.). "
            "`P&L Back Pre/In (acct; order_id)` é **Back-only** via join `order_id` (ledger ↔ executor_jsonl) e inclui tipos P&L-like (ex.: void/refund) quando existirem. "
            "Se o CSV não tiver `order_id`, esses campos podem ficar vazios._\n\n"
        )

        # Accounting por dia de execução (created_at UTC): atribui P&L total por ordem (ledger) ao dia em que a ordem foi enviada.
        # Isso reduz a confusão "post date != exec date" e permite auditoria operacional realista.
        try:
            if bal_csv and bal_csv.exists() and exec_by_oid_back:
                if acct_pnl_by_oid_total:
                    s1.append("**Accounting (por order_id): P&L por dia de execução (created_at UTC; Back Pre/In)**\n\n")
                    s1.append(
                        "| Dia (exec UTC) | P&L Back Pre | P&L Back In | P&L Total | ROIw Total | Cobertura oids% (no dia) | #ordens c/ P&L≈0 (void-like) |\n"
                    )
                    s1.append("|---|---:|---:|---:|---:|---:|---:|\n")
                    for dayk2 in sorted(list(days_utc_exec))[-10:]:
                        sp2 = _exec_day_split_back_pre_in_from_order_pnls(
                            exec_by_oid_back=exec_by_oid_back,
                            acct_pnl_by_oid_total=acct_pnl_by_oid_total,
                            day_exec_utc=str(dayk2),
                            audit_by_id=(audit_by_id or None),
                            pnl_zero_eps=float(os.getenv("DAILY_VOID_LIKE_PNL_EPS", "1e-9") or 1e-9),
                        )
                        if not isinstance(sp2, dict) or int(sp2.get("n_exec_day") or 0) <= 0:
                            continue
                        roiw = None
                        try:
                            exp = float(sp2.get("exp_total") or 0.0)
                            pnl = float(sp2.get("pnl_total") or 0.0)
                            roiw = (pnl / exp * 100.0) if exp > 0 else None
                        except Exception:
                            roiw = None
                        s1.append(
                            f"| {dayk2} | {_fmt_num(sp2.get('pnl_pre'),2)} | {_fmt_num(sp2.get('pnl_in'),2)} | {_fmt_num(sp2.get('pnl_total'),2)} | "
                            f"{_fmt_pct(roiw)} | {_fmt_pct(sp2.get('coverage_n_pct'),1)} | {int(sp2.get('n_pnl_zero') or 0)} |\n"
                        )
                    s1.append("\n")
        except Exception:
            pass

        # Back In × tempo no jogo (min desde kickoff; robustez): usa P&L do ledger por order_id (realizado) e timing do audit.
        try:
            if acct_pnl_by_oid_total and exec_by_oid_back and audit_by_id:
                # universo: execuções Back In cujas created_at caem nos dias mostrados em Execução
                order_rows_by_bucket: Dict[str, list[dict]] = defaultdict(list)
                n_in_total = 0
                n_in_with_acct = 0
                for oid, em in (exec_by_oid_back or {}).items():
                    if not isinstance(em, dict):
                        continue
                    created = em.get("created_at")
                    if not isinstance(created, datetime):
                        continue
                    if days_utc_exec and str(created.date().isoformat()) not in days_utc_exec:
                        continue
                    # in-play?
                    try:
                        aid = em.get("audit_id")
                        arow = (audit_by_id or {}).get(int(aid)) if (aid is not None and audit_by_id is not None) else None
                        is_in = bool(_is_inplay_from_audit_row(arow, exec_created_at_utc=created))
                    except Exception:
                        is_in = False
                        arow = None
                    if not bool(is_in):
                        continue
                    n_in_total += 1
                    pnl = acct_pnl_by_oid_total.get(str(oid))
                    if pnl is None:
                        continue
                    n_in_with_acct += 1
                    ko = None
                    ev_id = None
                    try:
                        if isinstance(arow, dict):
                            ko = _to_utc_dt(arow.get("kickoff_time"))
                            ev_id = str(arow.get("event_id") or "").strip() or None
                    except Exception:
                        ko = None
                        ev_id = None
                    mins = _bucket_min_to_kickoff(created, ko) if ko else None
                    lab = _bucket_label_min_since_kickoff(mins)
                    order_rows_by_bucket[lab].append(
                        {
                            "pnl": float(pnl),
                            "exposure": em.get("exposure"),
                            "event_id": ev_id,
                        }
                    )
                if order_rows_by_bucket:
                    s1.append("**Back In (acct; order_id): P&L × tempo no jogo (min desde kickoff; robusto)**\n\n")
                    s1.append(
                        "_Definição: `min_since_kickoff = created_at_utc − kickoff_time_utc` (minutos). "
                        "Classificação **Back In** usa `kickoff_time` quando disponível; senão `betslip_audit_results.is_live` (quando não-NULL). "
                        "Isto mede **tempo dentro do jogo**, não a latência para efetivar a aposta. "
                        "`ROIw = (∑P&L ledger)/(∑stake do executor)`._\n\n"
                    )
                    covp = _pct(n_in_with_acct, n_in_total)
                    s1.append(f"- Universo: Back In nos dias exibidos: n_orders=`{n_in_total}`; com ledger por `order_id`: `{n_in_with_acct}` ({_fmt_pct(covp,1)}).\n\n")
                    s1.append("| Bucket min_since_kickoff | n_ordens | n_jogos | Exposição (∑stake) | P&L (∑acct) | ROIw |\n")
                    s1.append("|---|---:|---:|---:|---:|---:|\n")
                    bucket_order = ["0-5m", "5-15m", "15-30m", "30-60m", ">60m", "Pre (<0)", "Desconhecido"]
                    for b in bucket_order:
                        summ = _summarize_rows_pnl_exp(order_rows_by_bucket.get(b) or [])
                        if int(summ.get("n_orders") or 0) <= 0:
                            continue
                        s1.append(
                            f"| {b} | {int(summ.get('n_orders') or 0)} | {int(summ.get('n_events') or 0)} | "
                            f"{_fmt_num(summ.get('exposure_sum'),2)} | {_fmt_num(summ.get('pnl_sum'),2)} | {_fmt_pct(summ.get('roi_weighted'))} |\n"
                        )
                    s1.append("\n")
        except Exception:
            pass

        # Back In × latência de efetivação (call_to_done_ms; robustez): P&L do ledger por order_id vs tempo total de execução.
        try:
            if acct_pnl_by_oid_total and exec_by_oid_back and audit_by_id:
                order_rows_by_bucket_lat: Dict[str, list[dict]] = defaultdict(list)
                n_in_total = 0
                n_in_with_acct = 0
                for oid, em in (exec_by_oid_back or {}).items():
                    if not isinstance(em, dict):
                        continue
                    created = em.get("created_at")
                    if not isinstance(created, datetime):
                        continue
                    if days_utc_exec and str(created.date().isoformat()) not in days_utc_exec:
                        continue

                    # in-play?
                    try:
                        aid = em.get("audit_id")
                        arow = (audit_by_id or {}).get(int(aid)) if (aid is not None and audit_by_id is not None) else None
                        is_in = bool(_is_inplay_from_audit_row(arow, exec_created_at_utc=created))
                    except Exception:
                        is_in = False
                        arow = None
                    if not bool(is_in):
                        continue

                    n_in_total += 1
                    pnl = acct_pnl_by_oid_total.get(str(oid))
                    if pnl is None:
                        continue
                    n_in_with_acct += 1

                    ev_id = None
                    try:
                        if isinstance(arow, dict):
                            ev_id = str(arow.get("event_id") or "").strip() or None
                    except Exception:
                        ev_id = None

                    lab = _bucket_label_call_to_done_ms(em.get("lat_ms"))
                    order_rows_by_bucket_lat[lab].append(
                        {
                            "pnl": float(pnl),
                            "exposure": em.get("exposure"),
                            "event_id": ev_id,
                        }
                    )

                if order_rows_by_bucket_lat:
                    s1.append("**Back In (acct; order_id): P&L × latência de efetivação (call_to_done_ms; robusto)**\n\n")
                    s1.append(
                        "_Definição: `call_to_done_ms` vem do `executor_jsonl` (tempo total do request até finalizar). "
                        "Isto mede **tempo para efetivar a aposta** (latência), não o minuto do jogo. "
                        "`ROIw = (∑P&L ledger)/(∑stake do executor)`._\n\n"
                    )
                    covp = _pct(n_in_with_acct, n_in_total)
                    s1.append(f"- Universo: Back In nos dias exibidos: n_orders=`{n_in_total}`; com ledger por `order_id`: `{n_in_with_acct}` ({_fmt_pct(covp,1)}).\n\n")
                    s1.append("| Bucket call_to_done_ms | n_ordens | n_jogos | Exposição (∑stake) | P&L (∑acct) | ROIw |\n")
                    s1.append("|---|---:|---:|---:|---:|---:|\n")
                    bucket_order = ["< 5s", "5-10s", "10-20s", "20-40s", "> 40s", "Desconhecido"]
                    for b in bucket_order:
                        summ = _summarize_rows_pnl_exp(order_rows_by_bucket_lat.get(b) or [])
                        if int(summ.get("n_orders") or 0) <= 0:
                            continue
                        s1.append(
                            f"| {b} | {int(summ.get('n_orders') or 0)} | {int(summ.get('n_events') or 0)} | "
                            f"{_fmt_num(summ.get('exposure_sum'),2)} | {_fmt_num(summ.get('pnl_sum'),2)} | {_fmt_pct(summ.get('roi_weighted'))} |\n"
                        )
                    s1.append("\n")
        except Exception:
            pass

        # Slippage × ROI (accounting; por order_id): janela móvel + acumulada (pós-início).
        try:
            if acct_pnl_by_oid_total and exec_by_oid_back:
                rows_all: list[dict] = []
                rows_pre: list[dict] = []
                rows_in: list[dict] = []
                rows_all_post: list[dict] = []
                rows_pre_post: list[dict] = []
                rows_in_post: list[dict] = []
                rows_all_since_post: list[dict] = []
                rows_pre_since_post: list[dict] = []
                rows_in_since_post: list[dict] = []
                post_start = str(os.getenv("DAILY_SLIPPAGE_POST_START_DAY", "2026-04-04") or "").strip()
                win_min = None
                win_max = None
                win_days = None
                try:
                    if days_utc_exec:
                        d0 = sorted(list(days_utc_exec))
                        if d0:
                            win_min = d0[0]
                            win_max = d0[-1]
                            try:
                                win_days = (datetime.fromisoformat(win_max).date() - datetime.fromisoformat(win_min).date()).days + 1
                            except Exception:
                                win_days = len(d0)
                except Exception:
                    win_min = None
                    win_max = None
                    win_days = None
                for oid, em in (exec_by_oid_back or {}).items():
                    if not isinstance(em, dict):
                        continue
                    created = em.get("created_at")
                    if not isinstance(created, datetime):
                        continue
                    pnl = acct_pnl_by_oid_total.get(str(oid))
                    if pnl is None:
                        continue
                    row = {"pnl": float(pnl), "exposure": em.get("exposure"), "slip_raw_pct": em.get("slip_raw_pct")}
                    created_day = str(created.date().isoformat())
                    rows_all.append(row)
                    # classifica Pre/In via audit quando possível
                    try:
                        aid = em.get("audit_id")
                        arow = (audit_by_id or {}).get(int(aid)) if (aid is not None and audit_by_id is not None) else None
                        is_in = bool(_is_inplay_from_audit_row(arow, exec_created_at_utc=created))
                    except Exception:
                        is_in = False
                    (rows_in if is_in else rows_pre).append(row)
                    # janela móvel (days_utc_exec)
                    if (not days_utc_exec) or (created_day in days_utc_exec):
                        rows_all_post.append(row)
                        (rows_in_post if is_in else rows_pre_post).append(row)
                    # corte pós-início (por created_at UTC)
                    try:
                        if post_start and created_day >= post_start:
                            rows_all_since_post.append(row)
                            (rows_in_since_post if is_in else rows_pre_since_post).append(row)
                    except Exception:
                        pass

                def _render(rows: list[dict], *, title: str) -> None:
                    buckets = _bucketize_slip_raw_3way_accounting(rows)
                    if not any(int(b.get("n") or 0) > 0 for b in buckets):
                        return
                    s1.append(f"**{title}**\n\n")
                    s1.append("| Bucket slippage_raw_pct | n | Exposição (∑stake) | P&L (∑acct) | ROIw |\n")
                    s1.append("|---|---:|---:|---:|---:|\n")
                    for b in buckets:
                        s1.append(
                            f"| {b.get('bucket')} | {int(b.get('n') or 0)} | {_fmt_num(b.get('exposure_sum'),2)} | {_fmt_num(b.get('pnl_sum'),2)} | {_fmt_pct(b.get('roi_weighted'))} |\n"
                        )
                    s1.append("\n")

                # Janela móvel explícita
                win_lbl = f"{win_min}..{win_max}" if (win_min and win_max) else "janela móvel"
                if win_days is not None and win_days > 0:
                    win_lbl = f"{win_lbl} ({int(win_days)} dias)"
                _render(rows_all_post, title=f"Slippage × ROI (accounting; order_id) — Back (janela móvel: {win_lbl})")
                _render(rows_pre_post, title=f"Slippage × ROI (accounting; order_id) — Back Pre (janela móvel: {win_lbl})")
                _render(rows_in_post, title=f"Slippage × ROI (accounting; order_id) — Back In (janela móvel: {win_lbl})")

                # Acumulado fixo desde post_start (independente da janela móvel)
                if rows_all_since_post:
                    _render(rows_all_since_post, title=f"Slippage × ROI (accounting; order_id) — Back (acumulado pós-início >= {post_start})")
                    _render(rows_pre_since_post, title=f"Slippage × ROI (accounting; order_id) — Back Pre (acumulado pós-início >= {post_start})")
                    _render(rows_in_since_post, title=f"Slippage × ROI (accounting; order_id) — Back In (acumulado pós-início >= {post_start})")
        except Exception:
            pass

        # Tese Back Pre fast (pre_submit_ms) — P&L/ROI por order_id e robustez (bootstrap)
        # Usa: exec_by_oid_back (executor_jsonl) + acct_pnl_by_oid_total (ledger) + audit_by_id (classificar Pre/In)
        try:
            if acct_pnl_by_oid_total and exec_by_oid_back:
                # Para n_liquidadas / ROIw_liquidado precisamos do open_stakes.csv (quando disponível)
                open_csv = None
                open_oids = None
                try:
                    out_dir = Path(os.getenv("ACCOUNTING_OUT_DIR", "logs/accounting"))
                    open_csv = _latest_open_stakes_csv(out_dir)
                    if open_csv is not None:
                        open_oids = _open_order_ids_from_open_stakes_csv(open_csv)
                except Exception:
                    open_oids = None
                _append_backpre_fast_slow_sections(
                    s1,
                    exec_by_oid_back=exec_by_oid_back,
                    acct_pnl_by_oid_total=acct_pnl_by_oid_total,
                    audit_by_id=(audit_by_id or None),
                    open_order_ids=open_oids,
                )
        except Exception:
            pass

        # Diagnóstico explícito de void/refund/cancel por dia (UTC), quando há ledger.
        try:
            if isinstance(acct_day_typ, dict) and acct_day_typ:
                s1.append("**Accounting ledger: Voids/Refunds/Cancels por dia (post date UTC)**\n\n")
                s1.append("| Dia | Bet (∑amount) | Void/Push (∑amount) | Refund (∑amount) | Cancel (∑amount) | Excluídos (dep/saque/etc.) | Top types (|amt|) |\n")
                s1.append("|---|---:|---:|---:|---:|---:|---|\n")
                for dayk in [str(it.get("day") or "") for it in (adh_day.get("per_day") or []) if isinstance(it, dict)]:
                    if not dayk or dayk not in acct_day_typ:
                        continue
                    blk = acct_day_typ.get(dayk) if isinstance(acct_day_typ.get(dayk), dict) else {}
                    if not blk:
                        continue
                    sums: Dict[str, float] = {}
                    excl: Dict[str, float] = {}
                    for typ, rec in blk.items():
                        if not isinstance(rec, dict):
                            continue
                        try:
                            amt = float(rec.get("amount_sum") or 0.0)
                        except Exception:
                            continue
                        tl = str(typ or "").strip().lower() or "unknown"
                        if _acct_is_excluded_type(tl):
                            excl[tl] = float(excl.get(tl) or 0.0) + float(amt)
                        else:
                            sums[tl] = float(sums.get(tl) or 0.0) + float(amt)
                    summ = _summarize_accounting_types(sums)
                    top = ", ".join([f"`{x.get('type')}`({_fmt_num(x.get('amount_sum'),2)})" for x in (summ.get("top_types") or []) if isinstance(x, dict)])
                    s1.append(
                        f"| {dayk} | {_fmt_num(summ.get('bet_sum'),2)} | {_fmt_num(summ.get('void_push_sum'),2)} | {_fmt_num(summ.get('refund_sum'),2)} | "
                        f"{_fmt_num(summ.get('cancel_sum'),2)} | {_fmt_num(sum(excl.values()) if excl else 0.0,2)} | {top or '—'} |\n"
                    )
                s1.append("\n")
        except Exception:
            pass

        # Cobertura de placar entre execuções bem-sucedidas (por n, stake e por jogo/event_id).
        # Isso é crucial para interpretar gaps entre P&L (acct) e P&L/ROI (placar),
        # porque as métricas por placar só usam o subconjunto coberto (n_cov / stake_sum_cov).
        try:
            s1.append("**Cobertura de placar (somente entre execuções bem-sucedidas)**\n\n")
            s1.append(
                "| Dia | Back n_cov/n_success | Back stake_cov/stake | Back jogos_cov/jogos_success | Lay n_cov/n_success | Lay stake_cov/stake | Lay jogos_cov/jogos_success |\n"
            )
            s1.append("|---|---:|---:|---:|---:|---:|---:|\n")
            for it in adh_day.get("per_day") or []:
                if not isinstance(it, dict):
                    continue
                ex = it.get("execution") if isinstance(it.get("execution"), dict) else {}
                back = ex.get("back") if isinstance(ex.get("back"), dict) else {}
                lay = ex.get("lay") if isinstance(ex.get("lay"), dict) else {}

                ns_b = int(back.get("n_success") or 0)
                nc_b = int(back.get("n_cov") or 0)
                st_b = float(back.get("stake_sum") or 0.0)
                stc_b = float(back.get("stake_sum_cov") or 0.0)
                covn_b = back.get("cov_pct_n")
                covs_b = back.get("cov_pct_stake")
                if covn_b is None:
                    covn_b = _pct(nc_b, ns_b)
                if covs_b is None:
                    covs_b = _pct(stc_b, st_b)

                evs_b = int(back.get("events_success_n") or 0)
                evc_b = int(back.get("events_cov_n") or 0)
                evp_b = back.get("events_cov_pct")
                if evp_b is None:
                    evp_b = _pct(evc_b, evs_b)

                ns_l = int(lay.get("n_success") or 0)
                nc_l = int(lay.get("n_cov") or 0)
                st_l = float(lay.get("stake_sum") or 0.0)
                stc_l = float(lay.get("stake_sum_cov") or 0.0)
                covn_l = lay.get("cov_pct_n")
                covs_l = lay.get("cov_pct_stake")
                if covn_l is None:
                    covn_l = _pct(nc_l, ns_l)
                if covs_l is None:
                    covs_l = _pct(stc_l, st_l)

                evs_l = int(lay.get("events_success_n") or 0)
                evc_l = int(lay.get("events_cov_n") or 0)
                evp_l = lay.get("events_cov_pct")
                if evp_l is None:
                    evp_l = _pct(evc_l, evs_l)

                def _fmt_ratio(n1: int, n0: int, pct: Optional[float]) -> str:
                    if n0 <= 0:
                        return "—"
                    p = f"{float(pct):.1f}%" if pct is not None else "—"
                    return f"{n1}/{n0} ({p})"

                def _fmt_ratio_num(x1: float, x0: float, pct: Optional[float]) -> str:
                    if x0 <= 0:
                        return "—"
                    p = f"{float(pct):.1f}%" if pct is not None else "—"
                    return f"{_fmt_num(x1,2)}/{_fmt_num(x0,2)} ({p})"

                s1.append(
                    f"| {it.get('day')} | {_fmt_ratio(nc_b, ns_b, covn_b)} | {_fmt_ratio_num(stc_b, st_b, covs_b)} | {_fmt_ratio(evc_b, evs_b, evp_b)} | "
                    f"{_fmt_ratio(nc_l, ns_l, covn_l)} | {_fmt_ratio_num(stc_l, st_l, covs_l)} | {_fmt_ratio(evc_l, evs_l, evp_l)} |\n"
                )
            s1.append("\n")
        except Exception:
            pass

        # Contrafactual operacional (placar): efeito de filtros simples
        # - remover slippage_raw_pct > +2% (beneficia se preços “subiram” demais vs decisão)
        # - manter call_to_done_ms <= 6s (proxy de execução rápida)
        # Observação: aplica-se SOMENTE ao subconjunto com ROI via placar (cobertura).
        try:
            # Versão 1 (Accounting ledger, por order_id): exata em termos de movimentos no balance.csv
            # (quando houver coluna de order_id). A interpretação é: "se eu tivesse aplicado esses filtros,
            # eu não teria colocado estas ordens; portanto, seus lançamentos no ledger não existiriam".
            if bal_csv and bal_csv.exists():
                try:
                    exec_by_oid = _parse_executor_jsonl_back_live_orders(Path(str(cfg.executor_jsonl)))
                except Exception:
                    exec_by_oid = {}
                try:
                    days_utc_cf = {str(it.get("day") or "") for it in (adh_day.get("per_day") or []) if isinstance(it, dict) and str(it.get("day") or "")}
                except Exception:
                    days_utc_cf = set()
                # Preferir mapa com `type` para capturar void/refund/cancel quando existirem como tipos separados.
                acct_by_oid_day_typ = _acct_amount_by_order_day_by_type_from_balance_csv(bal_csv, days_utc=set(days_utc_cf)) if days_utc_cf else {}
                # Fallback legado: apenas type=bet (quando não há coluna type/order_id adequada).
                acct_by_oid_day = _acct_amount_by_order_day_from_balance_csv(bal_csv, days_utc=set(days_utc_cf), only_type_bet=True) if days_utc_cf else {}

                def _acct_type_excl(tl: str) -> bool:
                    t = str(tl or "").strip().lower()
                    return any(k in t for k in ("deposit", "withdraw", "transfer", "top", "payment", "adjust", "bonus"))

                if exec_by_oid and (acct_by_oid_day_typ or acct_by_oid_day):
                    s1.append("**Contrafactual (accounting ledger; por order_id): filtros operacionais (Back)**\n\n")
                    s1.append(
                        "_Nota: P&L aqui vem do ledger por `order_id`. Quando o CSV expõe `type`, incluímos todos os tipos **P&L-like** (exclui dep/saque/transfer/etc.), "
                        "para capturar void/refund/cancel se existirem. Caso contrário, cai no legado `type=bet`._\n\n"
                    )
                    s1.append(
                        "| Dia (post date UTC) | Base P&L (acct) | Base ROIw | Após slippage_raw_pct<=+2%: P&L | ROIw | Após lat<=6s: P&L | ROIw | Após ambos: P&L | ROIw | Cobertura (orders com acct no dia) |\n"
                    )
                    s1.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
                    for it in adh_day.get("per_day") or []:
                        if not isinstance(it, dict):
                            continue
                        dayk = str(it.get("day") or "")
                        if not dayk:
                            continue
                        rows0 = []
                        n_with_acct = 0
                        for oid, em in (exec_by_oid or {}).items():
                            if not isinstance(em, dict):
                                continue
                            pnl_amt = None
                            # caminho preferencial: type-aware
                            if isinstance(acct_by_oid_day_typ, dict) and str(oid) in acct_by_oid_day_typ:
                                dmap = acct_by_oid_day_typ.get(str(oid)) if isinstance(acct_by_oid_day_typ.get(str(oid)), dict) else None
                                tmap = dmap.get(dayk) if isinstance(dmap, dict) and isinstance(dmap.get(dayk), dict) else None
                                if isinstance(tmap, dict) and tmap:
                                    try:
                                        pnl_amt = float(sum(float(v or 0.0) for k, v in tmap.items() if not _acct_type_excl(str(k))))
                                    except Exception:
                                        pnl_amt = None
                            # fallback legado: bet-only
                            if pnl_amt is None and isinstance(acct_by_oid_day, dict) and str(oid) in acct_by_oid_day:
                                amt_by_day = acct_by_oid_day.get(str(oid)) if isinstance(acct_by_oid_day.get(str(oid)), dict) else None
                                if isinstance(amt_by_day, dict) and dayk in amt_by_day:
                                    try:
                                        pnl_amt = float(amt_by_day.get(dayk) or 0.0)
                                    except Exception:
                                        pnl_amt = None
                            if pnl_amt is None:
                                continue
                            n_with_acct += 1
                            rows0.append(
                                {
                                    "pnl": float(pnl_amt),
                                    "exposure": em.get("exposure"),
                                    "slip_raw_pct": em.get("slip_raw_pct"),
                                    "lat_ms": em.get("lat_ms"),
                                }
                            )
                        if not rows0:
                            continue
                        cf_acct = _counterfactual_filters_back(
                            rows0,
                            slip_raw_pct_max=float(os.getenv("CF_SLIP_RAW_PCT_MAX", "2.0") or 2.0),
                            lat_ms_max=int(float(os.getenv("CF_LAT_CALL_TO_DONE_MS_MAX", "6000") or 6000)),
                            slip_missing_pass=True,
                            lat_missing_fail_closed=True,
                        )
                        base = cf_acct.get("base") if isinstance(cf_acct.get("base"), dict) else {}
                        a_slip = cf_acct.get("after_slip") if isinstance(cf_acct.get("after_slip"), dict) else {}
                        a_lat = cf_acct.get("after_lat") if isinstance(cf_acct.get("after_lat"), dict) else {}
                        a_both = cf_acct.get("after_both") if isinstance(cf_acct.get("after_both"), dict) else {}
                        s1.append(
                            f"| {dayk} | {_fmt_num(base.get('pnl_sum'),2)} | {_fmt_pct(base.get('roi_weighted'))} | "
                            f"{_fmt_num(a_slip.get('pnl_sum'),2)} | {_fmt_pct(a_slip.get('roi_weighted'))} | "
                            f"{_fmt_num(a_lat.get('pnl_sum'),2)} | {_fmt_pct(a_lat.get('roi_weighted'))} | "
                            f"{_fmt_num(a_both.get('pnl_sum'),2)} | {_fmt_pct(a_both.get('roi_weighted'))} | "
                            f"{int(n_with_acct)} |\n"
                        )
                    s1.append("\n")

                    # Versão 1b: somente Back In (exclui Back Pre)
                    try:
                        s1.append("**Contrafactual (accounting ledger; por order_id): filtros operacionais (Back In somente)**\n\n")
                        s1.append(
                            "| Dia (post date UTC) | Base P&L (acct) | Base ROIw | Após slippage_raw_pct<=+2%: P&L | ROIw | Após lat<=6s: P&L | ROIw | Após ambos: P&L | ROIw | Cobertura (orders Back In com acct no dia) |\n"
                        )
                        s1.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
                        for it in adh_day.get("per_day") or []:
                            if not isinstance(it, dict):
                                continue
                            dayk = str(it.get("day") or "")
                            if not dayk:
                                continue
                            rows0 = []
                            n_with_acct = 0
                            for oid, em in (exec_by_oid or {}).items():
                                if not isinstance(em, dict):
                                    continue
                                # filtro Back In (in-play): usa kickoff_time/is_live do audit (DB), não o flag `is_live` do executor.
                                try:
                                    aid = em.get("audit_id")
                                    arow = (audit_by_id or {}).get(int(aid)) if (aid is not None and audit_by_id is not None) else None
                                    is_in = bool(_is_inplay_from_audit_row(arow, exec_created_at_utc=em.get("created_at")))
                                except Exception:
                                    is_in = False
                                if not bool(is_in):
                                    continue
                                pnl_amt = None
                                if isinstance(acct_by_oid_day_typ, dict) and str(oid) in acct_by_oid_day_typ:
                                    dmap = acct_by_oid_day_typ.get(str(oid)) if isinstance(acct_by_oid_day_typ.get(str(oid)), dict) else None
                                    tmap = dmap.get(dayk) if isinstance(dmap, dict) and isinstance(dmap.get(dayk), dict) else None
                                    if isinstance(tmap, dict) and tmap:
                                        try:
                                            pnl_amt = float(sum(float(v or 0.0) for k, v in tmap.items() if not _acct_type_excl(str(k))))
                                        except Exception:
                                            pnl_amt = None
                                if pnl_amt is None and isinstance(acct_by_oid_day, dict) and str(oid) in acct_by_oid_day:
                                    amt_by_day = acct_by_oid_day.get(str(oid)) if isinstance(acct_by_oid_day.get(str(oid)), dict) else None
                                    if isinstance(amt_by_day, dict) and dayk in amt_by_day:
                                        try:
                                            pnl_amt = float(amt_by_day.get(dayk) or 0.0)
                                        except Exception:
                                            pnl_amt = None
                                if pnl_amt is None:
                                    continue
                                n_with_acct += 1
                                rows0.append(
                                    {
                                        "pnl": float(pnl_amt),
                                        "exposure": em.get("exposure"),
                                        "slip_raw_pct": em.get("slip_raw_pct"),
                                        "lat_ms": em.get("lat_ms"),
                                    }
                                )
                            if not rows0:
                                continue
                            cf_acct = _counterfactual_filters_back(
                                rows0,
                                slip_raw_pct_max=float(os.getenv("CF_SLIP_RAW_PCT_MAX", "2.0") or 2.0),
                                lat_ms_max=int(float(os.getenv("CF_LAT_CALL_TO_DONE_MS_MAX", "6000") or 6000)),
                                slip_missing_pass=True,
                                lat_missing_fail_closed=True,
                            )
                            base = cf_acct.get("base") if isinstance(cf_acct.get("base"), dict) else {}
                            a_slip = cf_acct.get("after_slip") if isinstance(cf_acct.get("after_slip"), dict) else {}
                            a_lat = cf_acct.get("after_lat") if isinstance(cf_acct.get("after_lat"), dict) else {}
                            a_both = cf_acct.get("after_both") if isinstance(cf_acct.get("after_both"), dict) else {}
                            s1.append(
                                f"| {dayk} | {_fmt_num(base.get('pnl_sum'),2)} | {_fmt_pct(base.get('roi_weighted'))} | "
                                f"{_fmt_num(a_slip.get('pnl_sum'),2)} | {_fmt_pct(a_slip.get('roi_weighted'))} | "
                                f"{_fmt_num(a_lat.get('pnl_sum'),2)} | {_fmt_pct(a_lat.get('roi_weighted'))} | "
                                f"{_fmt_num(a_both.get('pnl_sum'),2)} | {_fmt_pct(a_both.get('roi_weighted'))} | "
                                f"{int(n_with_acct)} |\n"
                            )
                        s1.append("\n")
                    except Exception:
                        pass

                    # Versão 1c: somente Back Pre (exclui Back In)
                    try:
                        s1.append("**Contrafactual (accounting ledger; por order_id): filtros operacionais (Back Pre somente)**\n\n")
                        s1.append(
                            "_Nota: `Base P&L` usa todas as ordens Back Pre com `order_id` no ledger daquele dia. "
                            "O filtro contrafactual usa `slippage_raw_pct` (pós-execução, `odd_final` vs `odd_at_decision`). "
                            "Se o gate operacional `slippage_raw_pct<=+2%` já estiver efetivamente aplicado no runtime, `Base` e `Após slippage<=+2%` "
                            "tendem a coincidir; divergências sugerem ordens fora do gate e/ou diferença entre métricas de slippage usadas no runtime vs relatório._"
                            "\n\n"
                        )
                        s1.append(
                            "| Dia (post date UTC) | Base P&L (acct) | Base ROIw | Após slippage_raw_pct<=+2%: P&L | ROIw | Após lat<=6s: P&L | ROIw | Após ambos: P&L | ROIw | Cobertura (orders Back Pre com acct no dia) |\n"
                        )
                        s1.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
                        for it in adh_day.get("per_day") or []:
                            if not isinstance(it, dict):
                                continue
                            dayk = str(it.get("day") or "")
                            if not dayk:
                                continue
                            rows0 = []
                            n_with_acct = 0
                            for oid, em in (exec_by_oid or {}).items():
                                if not isinstance(em, dict):
                                    continue
                                # filtro Back Pre: usa kickoff_time/is_live do audit (DB) quando possível; fallback market_regime
                                is_in = None
                                try:
                                    aid = em.get("audit_id")
                                    arow = (audit_by_id or {}).get(int(aid)) if (aid is not None and audit_by_id is not None) else None
                                    if isinstance(arow, dict):
                                        is_in = bool(_is_inplay_from_audit_row(arow, exec_created_at_utc=em.get("created_at")))
                                except Exception:
                                    is_in = None
                                if is_in is None:
                                    try:
                                        mreg = str(em.get("market_regime") or "").strip().lower()
                                        if mreg in ("pre", "in"):
                                            is_in = bool(mreg == "in")
                                    except Exception:
                                        is_in = None
                                if bool(is_in):
                                    continue

                                pnl_amt = None
                                if isinstance(acct_by_oid_day_typ, dict) and str(oid) in acct_by_oid_day_typ:
                                    dmap = acct_by_oid_day_typ.get(str(oid)) if isinstance(acct_by_oid_day_typ.get(str(oid)), dict) else None
                                    tmap = dmap.get(dayk) if isinstance(dmap, dict) and isinstance(dmap.get(dayk), dict) else None
                                    if isinstance(tmap, dict) and tmap:
                                        try:
                                            pnl_amt = float(sum(float(v or 0.0) for k, v in tmap.items() if not _acct_type_excl(str(k))))
                                        except Exception:
                                            pnl_amt = None
                                if pnl_amt is None and isinstance(acct_by_oid_day, dict) and str(oid) in acct_by_oid_day:
                                    amt_by_day = acct_by_oid_day.get(str(oid)) if isinstance(acct_by_oid_day.get(str(oid)), dict) else None
                                    if isinstance(amt_by_day, dict) and dayk in amt_by_day:
                                        try:
                                            pnl_amt = float(amt_by_day.get(dayk) or 0.0)
                                        except Exception:
                                            pnl_amt = None
                                if pnl_amt is None:
                                    continue
                                n_with_acct += 1
                                rows0.append(
                                    {
                                        "pnl": float(pnl_amt),
                                        "exposure": em.get("exposure"),
                                        "slip_raw_pct": em.get("slip_raw_pct"),
                                        "lat_ms": em.get("lat_ms"),
                                    }
                                )
                            if not rows0:
                                continue
                            cf_acct = _counterfactual_filters_back(
                                rows0,
                                slip_raw_pct_max=float(os.getenv("CF_SLIP_RAW_PCT_MAX", "2.0") or 2.0),
                                lat_ms_max=int(float(os.getenv("CF_LAT_CALL_TO_DONE_MS_MAX", "6000") or 6000)),
                                slip_missing_pass=True,
                                lat_missing_fail_closed=True,
                            )
                            base = cf_acct.get("base") if isinstance(cf_acct.get("base"), dict) else {}
                            a_slip = cf_acct.get("after_slip") if isinstance(cf_acct.get("after_slip"), dict) else {}
                            a_lat = cf_acct.get("after_lat") if isinstance(cf_acct.get("after_lat"), dict) else {}
                            a_both = cf_acct.get("after_both") if isinstance(cf_acct.get("after_both"), dict) else {}
                            s1.append(
                                f"| {dayk} | {_fmt_num(base.get('pnl_sum'),2)} | {_fmt_pct(base.get('roi_weighted'))} | "
                                f"{_fmt_num(a_slip.get('pnl_sum'),2)} | {_fmt_pct(a_slip.get('roi_weighted'))} | "
                                f"{_fmt_num(a_lat.get('pnl_sum'),2)} | {_fmt_pct(a_lat.get('roi_weighted'))} | "
                                f"{_fmt_num(a_both.get('pnl_sum'),2)} | {_fmt_pct(a_both.get('roi_weighted'))} | "
                                f"{int(n_with_acct)} |\n"
                            )
                        s1.append("\n")
                    except Exception:
                        pass

            # Versão 2 (placar; cobertura): mantém como diagnóstico, mas NÃO é accounting.
            s1.append("**Contrafactual (placar; somente cobertos por ROI): filtros operacionais (Back)**\n\n")
            s1.append(
                "| Dia | Base P&L | Base ROIw | Após slippage_raw_pct<=+2%: P&L | ROIw | Após lat<=6s: P&L | ROIw | Após ambos: P&L | ROIw |\n"
            )
            s1.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for it in adh_day.get("per_day") or []:
                if not isinstance(it, dict):
                    continue
                ex = it.get("execution") if isinstance(it.get("execution"), dict) else {}
                back = ex.get("back") if isinstance(ex.get("back"), dict) else {}
                cf = back.get("filters_counterfactual") if isinstance(back.get("filters_counterfactual"), dict) else {}
                base = cf.get("base") if isinstance(cf.get("base"), dict) else {}
                a_slip = cf.get("after_slip") if isinstance(cf.get("after_slip"), dict) else {}
                a_lat = cf.get("after_lat") if isinstance(cf.get("after_lat"), dict) else {}
                a_both = cf.get("after_both") if isinstance(cf.get("after_both"), dict) else {}
                if not base:
                    continue
                s1.append(
                    f"| {it.get('day')} | {_fmt_num(base.get('pnl_sum'),2)} | {_fmt_pct(base.get('roi_weighted'))} | "
                    f"{_fmt_num(a_slip.get('pnl_sum'),2)} | {_fmt_pct(a_slip.get('roi_weighted'))} | "
                    f"{_fmt_num(a_lat.get('pnl_sum'),2)} | {_fmt_pct(a_lat.get('roi_weighted'))} | "
                    f"{_fmt_num(a_both.get('pnl_sum'),2)} | {_fmt_pct(a_both.get('roi_weighted'))} |\n"
                )
            s1.append("\n")
        except Exception:
            pass

        # Accounting: P&L por jogo (event_id) e comparação coberto vs não-coberto.
        # Usa o `balance_csv` baixado no começo do daily.
        try:
            bal_csv = Path(str(acct.get("balance_csv") or "")).expanduser() if isinstance(acct, dict) else None
        except Exception:
            bal_csv = None
        try:
            days_utc = {str(it.get("day") or "") for it in (adh_day.get("per_day") or []) if isinstance(it, dict) and str(it.get("day") or "")}
        except Exception:
            days_utc = set()
        by_day_event = None
        if bal_csv and days_utc and bal_csv.exists():
            try:
                by_day_event = _acct_pnl_per_event_from_balance_csv(bal_csv, days_utc=set(days_utc), only_type_bet=True)
            except Exception:
                by_day_event = None
        if isinstance(by_day_event, dict) and by_day_event:
            try:
                s1.append("**Accounting: distribuição de P&L por jogo (event_id; por post date UTC)**\n\n")
                s1.append(
                    "| Dia | #jogos | P&L total (acct; bets) | P&L mediana/jogo | Stake médio/jogo (proxy) | ROI mediana (P&L mediana / stake médio) | P10 | P90 | Concentração P&L (max |abs| / soma |abs|) | Turnover proxy (∑-amount) | Concentração turnover (max share) |\n"
                )
                s1.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
                for it in adh_day.get("per_day") or []:
                    if not isinstance(it, dict):
                        continue
                    dayk = str(it.get("day") or "")
                    ev_map = by_day_event.get(dayk) if isinstance(by_day_event.get(dayk), dict) else {}
                    summ = _summarize_event_pnls(ev_map)
                    s1.append(
                        f"| {dayk} | {int(summ.get('events_n') or 0)} | {_fmt_num(summ.get('pnl_sum'),2)} | {_fmt_num(summ.get('pnl_median'),2)} | "
                        f"{_fmt_num(summ.get('stake_mean_per_game'),2)} | {_fmt_pct(summ.get('roi_median_pct'))} | "
                        f"{_fmt_num(summ.get('pnl_p10'),2)} | {_fmt_num(summ.get('pnl_p90'),2)} | {_fmt_pct((float(summ.get('pnl_conc_max_abs_share'))*100.0) if summ.get('pnl_conc_max_abs_share') is not None else None,2)} | "
                        f"{_fmt_num(summ.get('stake_est_sum'),2)} | {_fmt_pct((float(summ.get('stake_conc_max_share'))*100.0) if summ.get('stake_conc_max_share') is not None else None,2)} |\n"
                    )
                s1.append("\n")
                s1.append("**Risco de cauda por jogo (event_id; accounting)**\n\n")
                s1.append(
                    "| Dia | #jogos | P5 P&L/jogo | CVaR5 P&L/jogo (média piores 5%) | Pior jogo |\n"
                )
                s1.append("|---|---:|---:|---:|---:|\n")
                for it in adh_day.get("per_day") or []:
                    if not isinstance(it, dict):
                        continue
                    dayk = str(it.get("day") or "")
                    ev_map = by_day_event.get(dayk) if isinstance(by_day_event.get(dayk), dict) else {}
                    tail = _event_tail_risk_metrics(ev_map)
                    s1.append(
                        f"| {dayk} | {int(tail.get('games_n') or 0)} | {_fmt_num(tail.get('p5_pnl_per_game'),2)} | {_fmt_num(tail.get('cvar5_pnl_per_game'),2)} | {_fmt_num(tail.get('worst_game_pnl'),2)} |\n"
                    )
                s1.append("\n")
                s1.append("**Top jogos por exposição (proxy) — concentração operacional**\n\n")
                s1.append("| Dia | event_id | event_name | Exposição proxy (∑-amount) | Share da exposição do dia | P&L por jogo |\n")
                s1.append("|---|---|---|---:|---:|---:|\n")
                for it in adh_day.get("per_day") or []:
                    if not isinstance(it, dict):
                        continue
                    dayk = str(it.get("day") or "")
                    ev_map = by_day_event.get(dayk) if isinstance(by_day_event.get(dayk), dict) else {}
                    tops = _top_event_exposures(ev_map, top_n=5)
                    for t in tops:
                        s1.append(
                            f"| {dayk} | {t.get('event_id')} | {str(t.get('event_name') or '')[:48]} | {_fmt_num(t.get('stake_est_sum'),2)} | {_fmt_pct(t.get('share_pct'),2)} | {_fmt_num(t.get('pnl_sum'),2)} |\n"
                        )
                s1.append("\n")
            except Exception:
                pass

            # Coberto vs não-coberto (placar) no accounting, por jogo/event_id.
            # Observação: isso depende de conciliar "dia UTC" e de o balance lançar movimentos no mesmo dia.
            try:
                s1.append("**Accounting: coberto vs não-coberto (placar), por jogo/event_id (mesmo dia UTC)**\n\n")
                s1.append(
                    "| Dia | Back jogos_success | Back jogos_cov | Back jogos_uncov | P&L acct cov | P&L acct uncov | Turnover proxy cov | Turnover proxy uncov |\n"
                )
                s1.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
                for it in adh_day.get("per_day") or []:
                    if not isinstance(it, dict):
                        continue
                    ex = it.get("execution") if isinstance(it.get("execution"), dict) else {}
                    back = ex.get("back") if isinstance(ex.get("back"), dict) else {}
                    dayk = str(it.get("day") or "")
                    ev_map = by_day_event.get(dayk) if isinstance(by_day_event.get(dayk), dict) else {}
                    if not dayk or not ev_map:
                        continue

                    ev_success = set(str(x).strip() for x in (back.get("event_ids_success") or []) if str(x).strip())
                    ev_cov = set(str(x).strip() for x in (back.get("event_ids_cov") or []) if str(x).strip())
                    if not ev_success:
                        continue
                    ev_uncov = set(ev_success) - set(ev_cov)

                    pnl_cov = 0.0
                    pnl_uncov = 0.0
                    st_cov = 0.0
                    st_uncov = 0.0
                    for ev_id, rec in (ev_map or {}).items():
                        if not isinstance(rec, dict):
                            continue
                        try:
                            pnl = float(rec.get("pnl_sum") or 0.0)
                        except Exception:
                            pnl = 0.0
                        try:
                            st = float(rec.get("stake_est_sum") or 0.0)
                        except Exception:
                            st = 0.0
                        if ev_id in ev_cov:
                            pnl_cov += pnl
                            st_cov += st
                        if ev_id in ev_uncov:
                            pnl_uncov += pnl
                            st_uncov += st

                    s1.append(
                        f"| {dayk} | {len(ev_success)} | {len(ev_cov)} | {len(ev_uncov)} | {_fmt_num(pnl_cov,2)} | {_fmt_num(pnl_uncov,2)} | {_fmt_num(st_cov,2)} | {_fmt_num(st_uncov,2)} |\n"
                    )
                s1.append("\n")
            except Exception:
                pass

        # Quebra por tipo (Back/Lay × Pre/In) no P&L por placar (cobertura). Ajuda a explicar dias OOS/placar positivos vs accounting negativo.
        try:
            s1.append("**Quebra (placar): Back/Lay × Pre/In (somente cobertos por ROI)**\n\n")
            s1.append("| Dia | P&L Back Pre | ROI Back Pre | P&L Back In | ROI Back In | P&L Lay Pre | ROI Lay Pre/liab | P&L Lay In | ROI Lay In/liab |\n")
            s1.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for it in adh_day.get("per_day") or []:
                if not isinstance(it, dict):
                    continue
                ex = it.get("execution") if isinstance(it.get("execution"), dict) else {}
                bt = ex.get("pnl_placar_by_type") if isinstance(ex.get("pnl_placar_by_type"), dict) else {}
                def _p(k: str) -> float:
                    try:
                        return float((bt.get(k) or {}).get("pnl") or 0.0) if isinstance(bt.get(k), dict) else 0.0
                    except Exception:
                        return 0.0
                def _e(k: str) -> float:
                    try:
                        return float((bt.get(k) or {}).get("exposure") or 0.0) if isinstance(bt.get(k), dict) else 0.0
                    except Exception:
                        return 0.0
                pbp, ebp = _p("Back_Pre"), _e("Back_Pre")
                pbi, ebi = _p("Back_In"), _e("Back_In")
                plp, elp = _p("Lay_Pre"), _e("Lay_Pre")
                pli, eli = _p("Lay_In"), _e("Lay_In")
                r_bp = (pbp / ebp * 100.0) if ebp > 0 else None
                r_bi = (pbi / ebi * 100.0) if ebi > 0 else None
                r_lp = (plp / elp * 100.0) if elp > 0 else None
                r_li = (pli / eli * 100.0) if eli > 0 else None
                s1.append(
                    f"| {it.get('day')} | {_fmt_num(pbp,2)} | {_fmt_pct(r_bp)} | {_fmt_num(pbi,2)} | {_fmt_pct(r_bi)} | "
                    f"{_fmt_num(plp,2)} | {_fmt_pct(r_lp)} | {_fmt_num(pli,2)} | {_fmt_pct(r_li)} |\n"
                )
            s1.append("\n")
            # Risco de cauda por bucket operacional (ordens com accounting)
            try:
                recs: List[Dict[str, Any]] = []
                for it in adh_day.get("per_day") or []:
                    if not isinstance(it, dict):
                        continue
                    dayk = str(it.get("day") or "")
                    ex = it.get("execution") if isinstance(it.get("execution"), dict) else {}
                    bt = ex.get("pnl_placar_by_type") if isinstance(ex.get("pnl_placar_by_type"), dict) else {}
                    # usamos o bloco por tipo para construir uma proxy de ordens por bucket
                    # (cada linha do bucket agrega ordens cobertas).
                    for k, side, regime in (
                        ("Back_Pre", "Back", "Pre"),
                        ("Back_In", "Back", "In"),
                        ("Lay_Pre", "Lay", "Pre"),
                        ("Lay_In", "Lay", "In"),
                    ):
                        d = bt.get(k) if isinstance(bt.get(k), dict) else {}
                        if not d:
                            continue
                        pnl = _safe_float(d.get("pnl"))
                        exp = _safe_float(d.get("exposure"))
                        n = _safe_int(d.get("n"))
                        if pnl is None or exp is None or n is None or int(n) <= 0:
                            continue
                        # replica n registros sintéticos para permitir quantis/cauda por bucket;
                        # cada item recebe média por ordem (aproximação operacional).
                        pnlo = float(pnl) / float(max(1, int(n)))
                        expo = float(exp) / float(max(1, int(n)))
                        for _ in range(int(max(1, int(n)))):
                            recs.append(
                                {
                                    "day": dayk,
                                    "side": side,
                                    "regime": regime,
                                    "pnl": pnlo,
                                    "exposure": expo,
                                }
                            )
                tail_bucket = _order_tail_risk_by_bucket(recs, top_n=5)
                byb = tail_bucket.get("by_bucket") if isinstance(tail_bucket.get("by_bucket"), list) else []
                if byb:
                    s1.append("**Risco de cauda por bucket operacional (proxy por ordem coberta)**\n\n")
                    s1.append("| Bucket | n_ordens | Exposição (∑) | P5 P&L/ordem | CVaR5 P&L/ordem | Pior ordem |\n")
                    s1.append("|---|---:|---:|---:|---:|---:|\n")
                    for r in byb:
                        if not isinstance(r, dict):
                            continue
                        s1.append(
                            f"| {r.get('bucket')} | {int(r.get('n_orders') or 0)} | {_fmt_num(r.get('exp_sum'),2)} | {_fmt_num(r.get('p5'),2)} | {_fmt_num(r.get('cvar5'),2)} | {_fmt_num(r.get('worst'),2)} |\n"
                        )
                    s1.append("\n")
            except Exception:
                pass
        except Exception:
            pass

        # slippage x ROI (3 buckets raw com sinal) — acumulado na janela (não só um dia)
        # 2.2) Latência × ROI (Back Pre/In): só depende do oos_adherence_long (slip)
        try:
            _append_latency_vs_roi_back_pre_in_section(
                s1,
                adh_slip=adh_slip,
                title="Latência × ROI (Back Pre/In) — acumulado (call_to_done_ms)",
            )
        except Exception:
            pass

        # 2.3) Slippage × Latência (Back Pre/In): acumulado na janela
        try:
            _append_slippage_vs_latency_back_pre_in_section(
                s1,
                adh_slip=adh_slip,
                title="Slippage × Latência (Back Pre/In) — acumulado (call_to_done_ms)",
            )
        except Exception:
            pass

        # 2.4) Slippage × ROI (raw, com sinal): acumulado na janela
        _append_slippage_vs_roi_raw_section(
            s1,
            adh_slip=adh_slip,
            title="Slippage × ROI por bucket (raw, com sinal) — acumulado",
            combo_top_limit=2,
        )

        # 2.5) Slippage pós-início (>= 2026-04-04) — mantém as mesmas tabelas, mas com corte.
        try:
            post_start = str(os.getenv("DAILY_SLIPPAGE_POST_START_DAY", "2026-04-04") or "").strip()
            if post_start:
                adh_post_json = day_dir / "oos_adherence_long_post_start.json"
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "ops.oos_adherence_report",
                        "--policy-json",
                        str(cfg.wf_policy_current),
                        "--executor-jsonl",
                        str(cfg.executor_jsonl),
                        "--tz",
                        "UTC",
                        "--days",
                        "0",
                        "--no-per-day",
                        "--slippage-cf-start-day",
                        post_start,
                        "--out",
                        str(adh_post_json),
                    ],
                    check=False,
                    cwd=str(Path(__file__).resolve().parent.parent),
                )
                adh_post = _read_json(adh_post_json)
                try:
                    _append_slippage_vs_latency_back_pre_in_section(
                        s1,
                        adh_slip=adh_post,
                        title=f"Slippage × Latência (Back Pre/In) — pós-início (>= {post_start})",
                    )
                except Exception:
                    pass
                _append_slippage_vs_roi_raw_section(
                    s1,
                    adh_slip=adh_post,
                    title=f"Slippage × ROI por bucket (raw, com sinal) — pós-início (>= {post_start})",
                    combo_top_limit=2,
                )
        except Exception:
            pass
        # fallback: se nada foi renderizado, pega o último dia com buckets (melhor do que vazio)
        try:
            if (not isinstance(adh_slip, dict) or not adh_slip) and isinstance(adh_day, dict):
                last = _pick_last_day_with_slippage_vs_roi_raw(list(adh_day.get("per_day") or []))
                if isinstance(last, dict):
                    ex = last.get("execution") if isinstance(last.get("execution"), dict) else {}
                    rawblk = ex.get("slippage_vs_roi_raw") if isinstance(ex.get("slippage_vs_roi_raw"), dict) else {}
                    if rawblk:
                        s1.append(f"**Slippage × ROI por bucket (raw, com sinal) — exemplo do dia `{last.get('day')}`**\n\n")
                        for side_key, title in (("back", "Back (ROI por stake)"), ("lay", "Lay (ROI por liability)")):
                            b = rawblk.get(side_key) if isinstance(rawblk.get(side_key), dict) else {}
                            buckets0 = b.get("buckets") if isinstance(b.get("buckets"), list) else []
                            buckets = _slip_raw_3bucket_rows(buckets0)
                            if not any(int(r.get("n") or 0) > 0 for r in buckets):
                                continue
                            s1.append(f"- **{title}**\n\n")
                            s1.append("| Bucket slippage_raw_pct | n | ROI mean (SE; IC95) |\n|---|---:|---:|\n")
                            for row in buckets:
                                s1.append(f"| {row.get('bucket')} | {int(row.get('n') or 0)} | {_fmt_roi_mean_se_ci_pct(row)} |\n")
                            s1.append("\n")
        except Exception:
            pass

    # Funil (24h) por auditoria: total → OK/valid → erros principais
    if isinstance(audit_rep, dict) and isinstance(audit_rep.get("by_version"), list) and audit_rep.get("by_version"):
        s1.append("**Funil de oportunidades (últimas 24h; auditoria DB)**\n\n")
        s1.append("| audit_version | total | OK | OK_valid | GATE_NOT_ELIGIBLE | API_FAILED | STALE_QUEUE_WAIT |\n")
        s1.append("|---|---:|---:|---:|---:|---:|---:|\n")
        for v in audit_rep.get("by_version") or []:
            if not isinstance(v, dict):
                continue
            sc = v.get("status_counts") if isinstance(v.get("status_counts"), dict) else {}
            s1.append(
                f"| {v.get('audit_version')} | {int(v.get('total') or 0)} | {int(sc.get('OK') or 0)} | {int(v.get('ok_valid') or 0)} | "
                f"{int(sc.get('GATE_NOT_ELIGIBLE') or 0)} | {int(sc.get('API_FAILED') or 0)} | {int(sc.get('STALE_QUEUE_WAIT') or 0)} |\n"
            )
        s1.append("\n")

        # motivos top (api_error) agregados
        errs = []
        try:
            for it in (audit_rep.get("error_rows") or []):
                if not isinstance(it, dict):
                    continue
                n = int(it.get("n") or 0)
                if n <= 0:
                    continue
                errs.append((n, str(it.get("audit_version") or ""), str(it.get("status") or ""), str(it.get("api_error") or "")))
            errs.sort(key=lambda x: x[0], reverse=True)
        except Exception:
            errs = []
        if errs:
            s1.append("**Motivos principais de não-execução / falha (top)**\n\n")
            for n, ver, st, err in errs[:8]:
                err2 = (err[:180] + "…") if len(err) > 180 else err
                s1.append(f"- `{ver}`: {st} ×{n} — `{err2}`\n")
            s1.append("\n")

        s1.append("> **APÊNDICE DE PESQUISA — NÃO OPERACIONAL**\n\n")
        s1.append("**Oportunidades identificadas (pesquisa — sem alteração de stake/threshold)**\n\n")
        s1.append(
            "- Observação: PMM/timeout / betslips / fila podem limitar conversão.\n"
            "- Evidência: KPIs de audit/executor nas últimas 24h.\n"
            "- Status: WATCH / diagnóstico.\n"
            "- Limitação: este Daily **não** recomenda mudança de policy, stake ou caps.\n\n"
        )

    # Sensibilidade de banca (reusa tabela OOS existente)
    if oos_txt:
        # OOS pode estar numerado como 12.x (full) ou 1.x (oos_first)
        sens = _extract_md_block(
            oos_txt,
            start="### 12.2b Sensibilidade por banca",
            until_any=["### 12.2c Sensibilidade por banca", "### 12.3 ", "### 1.2c Sensibilidade por banca", "### 1.3 "],
        )
        if not sens.strip():
            sens = _extract_md_block(
                oos_txt,
                start="### 1.2b Sensibilidade por banca",
                until_any=["### 1.2c Sensibilidade por banca", "### 1.3 ", "### 12.2c Sensibilidade por banca", "### 12.3 "],
            )
        if sens.strip():
            s1.append("> **APÊNDICE DE PESQUISA — NÃO OPERACIONAL** (Kelly/sizing/sensibilidade)\n\n")
            s1.append("**Estudo de sensibilidade (banca × lucro)**\n\n")
            s1.append(
                "_A tabela abaixo é reaproveitada do bloco OOS (mesmo layout). Ela responde “até onde a operação escala” antes de bater em caps/limites._\n\n"
            )
            s1.append(sens + "\n")

        # Diagnóstico: por que turnover/jogos/lucro OOS podem cair nos steps recentes
        try:
            tbl_md, rows = _extract_md_table(oos_txt, header_startswith="| Train window")
            if tbl_md.strip() and rows:
                s1.append("**OOS recente: escala (turnover/jogos/lucro) por step**\n\n")
                s1.append(
                    "_Leitura: se `#ativas (keys)` e `Jogos OOS` caem, a causa típica é calendário/fragmentação (por liga) + filtros (AH/exec_bucket) + cobertura de placar. "
                    "Se jogos não caem, mas turnover cai, o gargalo tende a ser budget/caps (governança) e sizing._\n\n"
                )
                # mostrar a tabela original e destacar os últimos 4 steps (no topo executivo)
                s1.append(tbl_md + "\n")
                last4 = rows[-4:]
                hdr = _md_table_header_cols(tbl_md)
                hmap = {str(c).strip(): i for i, c in enumerate(hdr)}
                # índices robustos (compatível com tabela antiga e nova)
                ix_games = hmap.get("Jogos OOS")
                ix_turn = hmap.get("Turnover (teste)")
                ix_pnl = hmap.get("Lucro (estratégia, budget)")
                ix_turn_pre = hmap.get("Turnover Pre")
                ix_turn_in = hmap.get("Turnover In")
                def _g(cols, ix):
                    try:
                        if ix is None:
                            return ""
                        return cols[int(ix)]
                    except Exception:
                        return ""
                # heurística: comparar último vs mediana
                try:
                    games = []
                    turns = []
                    turns_pre = []
                    turns_in = []
                    profs = []
                    for r in rows:
                        try:
                            games.append(float(_g(r, ix_games)) if _g(r, ix_games) else 0.0)
                        except Exception:
                            pass
                        try:
                            turns.append(float(str(_g(r, ix_turn)).replace(",", ".")))
                        except Exception:
                            pass
                        try:
                            if _g(r, ix_turn_pre):
                                turns_pre.append(float(str(_g(r, ix_turn_pre)).replace(",", ".")))
                        except Exception:
                            pass
                        try:
                            if _g(r, ix_turn_in):
                                turns_in.append(float(str(_g(r, ix_turn_in)).replace(",", ".")))
                        except Exception:
                            pass
                        try:
                            profs.append(float(str(_g(r, ix_pnl)).replace(",", ".")))
                        except Exception:
                            pass
                    if games and turns:
                        import statistics
                        med_g = statistics.median(games)
                        med_t = statistics.median(turns)
                        s1.append("**Diagnóstico rápido (último step vs mediana histórica do WF)**\n\n")
                        # último row
                        lr = rows[-1]
                        g_last = float(_g(lr, ix_games) or 0.0)
                        t_last = None
                        try:
                            t_last = float(str(_g(lr, ix_turn)).replace(",", "."))
                        except Exception:
                            t_last = None
                        s1.append(f"- Jogos OOS (último): `{_fmt_num(g_last,0)}` vs mediana `{_fmt_num(med_g,0)}`\n")
                        if t_last is not None:
                            s1.append(f"- Turnover teste (último): `{_fmt_num(t_last,2)}` vs mediana `{_fmt_num(med_t,2)}`\n")
                        # Pre/In (se houver)
                        try:
                            if ix_turn_pre is not None and turns_pre:
                                med_tp = statistics.median(turns_pre)
                                tp_last = float(str(_g(lr, ix_turn_pre)).replace(",", ".")) if _g(lr, ix_turn_pre) else None
                                if tp_last is not None:
                                    s1.append(f"- Turnover Pre (último): `{_fmt_num(tp_last,2)}` vs mediana `{_fmt_num(med_tp,2)}`\n")
                            if ix_turn_in is not None and turns_in:
                                med_ti = statistics.median(turns_in)
                                ti_last = float(str(_g(lr, ix_turn_in)).replace(",", ".")) if _g(lr, ix_turn_in) else None
                                if ti_last is not None:
                                    s1.append(f"- Turnover In (último): `{_fmt_num(ti_last,2)}` vs mediana `{_fmt_num(med_ti,2)}`\n")
                        except Exception:
                            pass
                        s1.append(
                            "- Se a queda é em **Jogos OOS**: problema é **volume/cobertura** (placar, calendário, fragmentação por liga, filtros como AH/exec_bucket).\n"
                            "- Se Jogos OOS está ok mas turnover cai: **governança/sizing** (budgets/caps) está limitando escala.\n\n"
                        )
                except Exception:
                    pass
        except Exception:
            pass

    # Histórico recente de policy (parâmetros “passados” do portfólio ativo)
    try:
        hist_lines = _tail_lines(cfg.wf_policy_history_jsonl, 12)
        recs = []
        for ln in hist_lines:
            try:
                recs.append(json.loads(ln))
            except Exception:
                continue
        recs = [r for r in recs if isinstance(r, dict)]
        if recs:
            s1.append("**Portfólio OOS: vigente vs histórico recente**\n\n")
            s1.append("| ts | n_active_keys |\n|---|---:|\n")
            for r in recs[-8:]:
                nkeys = None
                try:
                    ak = r.get("active_keys")
                    nkeys = len(ak) if isinstance(ak, list) else None
                except Exception:
                    nkeys = None
                s1.append(f"| {r.get('ts')} | {nkeys if nkeys is not None else '—'} |\n")
            s1.append("\n")
    except Exception:
        pass

    # parâmetros de negócio e técnicos: manter 99.6 como fonte, mas resumir aqui
    s1.append("**Parâmetros vigentes (visão executiva)**\n\n")
    # decomposição de active_keys (negócio)
    try:
        if isinstance(active_keys, list) and active_keys:
            def _cnt(prefix: str) -> int:
                return sum(1 for k in active_keys if str(k).startswith(prefix))
            by_league = sum(1 for k in active_keys if "__" in str(k))
            s1.append("| Dimensão | Valor |\n|---|---:|\n")
            s1.append(f"| active_keys (total) | {len(active_keys)} |\n")
            s1.append(f"| chaves por liga (suFIXO `__<League>`) | {by_league} |\n")
            s1.append(f"| Back_Pre | {_cnt('Back_Pre_')} |\n")
            s1.append(f"| Back_In | {_cnt('Back_In_')} |\n")
            s1.append(f"| Lay_Pre_Yes | {_cnt('Lay_Pre_Yes')} |\n")
            s1.append(f"| Lay_Pre_No | {_cnt('Lay_Pre_No')} |\n")
            s1.append(f"| Lay_In_Yes | {_cnt('Lay_In_Yes')} |\n")
            s1.append(f"| Lay_In_No | {_cnt('Lay_In_No')} |\n")
            s1.append("\n")
    except Exception:
        pass
    s1.append(
        "- **Combinações ativas (OOS)**: ver `99.3` (active_keys) e o bloco `2) OOS`.\n"
        "- **Stake sizing operacional (real)**: hoje é **FLAT** via `BRIDGE_STAKE` (ver `99.3` e `99.6`).\n"
        "- **Parâmetros técnicos efetivos** (executor/audit/bridge): ver `99.6 Filtros ativos`.\n\n"
    )

    # Critérios (OOS e real) + clareza do filtro de AH
    # Preferir policy_current (fonte da verdade operacional) quando disponível.
    wf_key_by_league = bool(policy_wf.get("key_by_league")) if isinstance(policy_wf, dict) else (
        str(os.getenv("DAILY_WF_KEY_BY_LEAGUE", "1")).strip() not in ("0", "false", "False", "no", "NO")
    )
    wf_key_scope = str(policy_wf.get("key_by_league_scope") or "") if isinstance(policy_wf, dict) else str(os.getenv("DAILY_WF_KEY_BY_LEAGUE_SCOPE", str(cfg.wf_key_by_league_scope)) or "")
    wf_key_scope = wf_key_scope.strip() or "pre"
    try:
        wf_ah = float(policy_wf.get("ah_max_abs_line")) if isinstance(policy_wf, dict) and policy_wf.get("ah_max_abs_line") is not None else float(os.getenv("DAILY_WF_AH_MAX_ABS_LINE", str(cfg.wf_ah_max_abs_line)) or 0.0)
    except Exception:
        wf_ah = 0.0
    wf_ah_scope = str(policy_wf.get("ah_scope") or "") if isinstance(policy_wf, dict) else str(os.getenv("DAILY_WF_AH_SCOPE", str(cfg.wf_ah_scope)) or "")
    wf_ah_scope = wf_ah_scope.strip() or "pre"
    wf_min_matches = str(policy_wf.get("min_matches") or "") if isinstance(policy_wf, dict) else str(os.getenv("DAILY_WF_MIN_MATCHES", str(cfg.wf_min_matches)) or "")
    wf_min_matches = wf_min_matches.strip() or "0"
    s1.append("**Critérios de seleção (OOS) e critérios do real (bridge/executor)**\n\n")
    s1.append(
        "- **OOS (walk-forward)** decide o portfólio `active_keys`.\n"
        f"  - **Chave por liga**: `{wf_key_by_league}` (scope=`{wf_key_scope}`) ⇒ em pre-match a chave pode virar `...__<League>`.\n"
        f"  - **Filtro de AH ativo?**: `{wf_ah > 0}` (max_abs_line=`{_fmt_num(wf_ah,2)}`; scope=`{wf_ah_scope}`) ⇒ remove eventos com `abs(line)` acima do limiar.\n"
        f"  - **Mínimo de jogos no treino**: `wf_min_matches={wf_min_matches}` (0 = desligado).\n"
        "  - **Regra de decisão (por combinação, no treino)**:\n"
        "    - Se `ROI` for **significativamente negativo** (IC90 inteiro < 0): **bloqueia**.\n"
        "    - Se `ROI` for **significativamente positivo** (IC90 inteiro > 0): **ativa**.\n"
        "    - Se `ROI` > 0 mas **não significativo**:\n"
        "      - **Pre-match**: ativa apenas se **CLV > 0** (CLV não precisa ser sig.).\n"
        "      - **In-match**: ativa se **ROI > 0** (CLV não se aplica).\n"
        "  - Operacionalmente, o OOS também pode excluir buckets de execução (ex.: `wf_exclude_exec_buckets_back`).\n"
        "- **Real (shadow/live)**:\n"
        "  - O bridge só envia oportunidades cuja chave esteja em `active_keys` (policy current).\n"
        "  - `DRY_OK` = **shadow** (não apostou); `LIVE_OK` = **efetivo** (apostou).\n\n"
    )
    # Transparência do step vigente (train/test) usado para gerar o policy current.
    if isinstance(policy_last_step, dict):
        try:
            s1.append("**Policy current: janela de treino/teste (do último step exportado)**\n\n")
            s1.append(
                f"- Train window: `{policy_last_step.get('train')}` | Test window: `{policy_last_step.get('test')}` | "
                f"train_days={len(policy_last_step.get('train_days') or [])} | test_days={len(policy_last_step.get('test_days') or [])}\n\n"
            )
        except Exception:
            pass
    # explicitar shadow vs live na janela
    try:
        live_ok = int((kpi_all.get("status_counts") or {}).get("LIVE_OK") or 0)
        dry_ok = int((kpi_all.get("status_counts") or {}).get("DRY_OK") or 0)
        s1.append("**Este período está rodando shadow ou efetivo?**\n\n")
        if live_ok > 0 and dry_ok > 0:
            s1.append(f"- Misturado: `LIVE_OK={live_ok}` e `DRY_OK={dry_ok}`.\n\n")
        elif live_ok > 0:
            s1.append(f"- Predominantemente **efetivo**: `LIVE_OK={live_ok}` (e `DRY_OK={dry_ok}`).\n\n")
        else:
            s1.append(f"- Predominantemente **shadow**: `DRY_OK={dry_ok}` (e `LIVE_OK={live_ok}`).\n\n")
    except Exception:
        pass

    # aspectos técnicos (latência/gaps/restarts proxy)
    try:
        gaps = _executor_gaps_summary(exec_lines)
        s1.append("**Aspectos técnicos (latência/estabilidade)**\n\n")
        s1.append("- Latência detalhada: ver `99.2` (p50/p90/p99 por etapa).\n")
        if gaps.get("max_gap_s") is not None:
            s1.append(
                f"- Gaps no `executor_jsonl` (proxy de downtime/restart/sem tráfego): max `{_fmt_num(gaps.get('max_gap_s'),1)}s`, "
                f"gaps>5min `{gaps.get('gaps_gt_300s')}`, gaps>15min `{gaps.get('gaps_gt_900s')}`.\n\n"
            )
        else:
            s1.append("- Gaps no `executor_jsonl`: amostra insuficiente.\n\n")
    except Exception:
        pass

    # 6) Combinar markdown (base reordenado + blocos operacionais 99.x)
    extra = []
    extra.append("\n\n## 99) Operacional — saldo, P&L e execução\n\n")
    extra.append("### 99.1 Accounting (saldo + P&L)\n\n")
    extra.append(f"- Arquivo: `{acct_out}`\n")
    if acct.get("error"):
        extra.append(f"- **Erro**: **{acct.get('error')}**\n")
    extra.append(f"- Saldo atual: **{acct.get('balance_current')}**\n")
    extra.append(f"- P&L hoje/semana/mês: **{acct.get('pnl_today')} / {acct.get('pnl_week')} / {acct.get('pnl_month')}**\n")
    extra.append("\nMeses fechados:\n\n")
    extra.append("| Mês | P&L |\n|---|---:|\n")
    for k, v in (acct.get("closed_months") or {}).items():
        extra.append(f"| {k} | {v} |\n")

    extra.append("\n### 99.2 Execução (KPIs)\n\n")
    extra.append(f"- Fonte: `{cfg.executor_jsonl}`\n")
    extra.append("- Nota: métricas abaixo vêm do JSONL; se ele estiver **stale** ou incompleto, podem divergir do volume “24h, DB”.\n\n")

    # Status table
    extra.append("**Status (all)**\n\n")
    extra.append("| Status | N |\n|---|---:|\n")
    for k, v in (kpi_all.get("status_counts") or {}).items():
        extra.append(f"| {k} | {int(v)} |\n")
    extra.append("\n")

    def _timing_table(title: str, obj: Dict[str, Any]) -> str:
        def _row(name: str, a: dict) -> str:
            return (
                f"| {name} | {a.get('n')} | {a.get('p50')} | {a.get('p90')} | {a.get('p99')} | {a.get('mean')} |\n"
            )

        s = []
        s.append(f"**{title}**\n\n")
        s.append("| Métrica | n | p50 | p90 | p99 | mean |\n|---|---:|---:|---:|---:|---:|\n")
        for nm in ("queue_delay", "call_to_done", "post"):
            a = ((obj.get(nm) or {}) if isinstance(obj, dict) else {})
            s.append(_row(nm, a if isinstance(a, dict) else {}))
        s.append("\n")
        return "".join(s)

    timing_ok = (kpi_ok.get("timing_ms") or {}) if isinstance(kpi_ok, dict) else {}
    extra.append(_timing_table("Latência (somente LIVE_OK/DRY_OK) — ms", timing_ok))

    # Recorte 24h (consistente com o checklist de prontidão LIVE)
    try:
        timing_ok24 = (kpi_ok_24h.get("timing_ms") or {}) if isinstance(kpi_ok_24h, dict) else {}
        extra.append(_timing_table("Latência (últimas 24h; somente LIVE_OK/DRY_OK) — ms", timing_ok24))
    except Exception:
        pass
    slip_ok = (kpi_ok.get("slippage") or {}) if isinstance(kpi_ok, dict) else {}
    extra.append("**Slippage (somente LIVE_OK/DRY_OK, quando houver odd_at_decision)**\n\n")
    extra.append(
        "- Definição: `slippage = odd_final - odd_at_decision` (em odds decimais) e `slippage_pct = slippage/odd_at_decision`.\n"
        "- Interpretação depende do lado:\n"
        "  - **Back**: slippage_pct **negativo** = piorou (odd caiu); **positivo** = melhorou.\n"
        "  - **Lay**: slippage_pct **positivo** = piorou (odd subiu); **negativo** = melhorou.\n\n"
    )
    extra.append("| Tipo | n | p50 | p90 | p99 | mean |\n|---|---:|---:|---:|---:|---:|\n")
    for nm in ("abs", "pct"):
        a = (slip_ok.get(nm) or {}) if isinstance(slip_ok, dict) else {}
        extra.append(f"| {nm} | {a.get('n')} | {a.get('p50')} | {a.get('p90')} | {a.get('p99')} | {a.get('mean')} |\n")
    extra.append("\n")

    # Slippage por lado (Back vs Lay)
    slip_by_side = (kpi_ok.get("slippage_by_side") or {}) if isinstance(kpi_ok, dict) else {}
    if isinstance(slip_by_side, dict) and slip_by_side:
        extra.append("**Slippage por lado (Back vs Lay)**\n\n")
        extra.append("| Lado | Métrica | n | p50 | p90 | p99 | mean |\n|---|---|---:|---:|---:|---:|---:|\n")
        for side, obj in slip_by_side.items():
            if not isinstance(obj, dict):
                continue
            for nm, label in (
                ("raw_pct", "slippage_pct (raw)"),
                ("cost_pct", "slippage_pct (custo, >=0)"),
            ):
                a = obj.get(nm) if isinstance(obj.get(nm), dict) else {}
                extra.append(
                    f"| {side} | {label} | {a.get('n')} | {a.get('p50')} | {a.get('p90')} | {a.get('p99')} | {a.get('mean')} |\n"
                )
        extra.append("\n")
    extra.append(
        "_Nota: o p90/p99 de `call_to_done_ms` explode quando inclui `NO_SESSION/API_FAILED` (timeouts/relogin). "
        "Por isso reportamos também o recorte apenas de sucessos._\n\n"
    )

    if active_keys:
        extra.append("\n### 99.3 Regras OOS ativas (último step)\n\n")
        extra.append(f"- active_keys: {len(active_keys) if isinstance(active_keys, list) else '—'}\n")
        extra.append("```json\n" + json.dumps(active_keys, ensure_ascii=False, indent=2) + "\n```\n\n")
        extra.append("**Como ler `active_keys` (regra de aprovação)**\n\n")
        extra.append(
            "- `active_keys` é o **portfólio aprovado** pelo walk-forward (OOS) no **último step** exportado.\n"
            "- O bridge (`ops.executor_bridge_audit`) só envia para o executor oportunidades cuja **chave operacional** (combinação) esteja ativa.\n"
            "- Mapeamento de chaves (simplificado):\n"
            "  - **Back**: `Back_Pre_Any` (pre) ou `Back_In_Any` (in). Se o walk-forward estiver com `key_by_league`, a chave pode ter sufixo `__<League>`.\n"
            "  - **Lay**: `Lay_Pre_Yes/No` (pre) ou `Lay_In_Yes/No` (in). Para **H3B**, `Yes` indica que o sinal envolve reversão (por definição da hipótese).\n\n"
        )
        extra.append("**Regras de execução atuais (stake sizing)**\n\n")
        extra.append(
            "- No operacional (shadow/live), o tamanho enviado pelo bridge é **FLAT** via `BRIDGE_STAKE`.\n"
            "- Em **Back**: stake padrão = `BRIDGE_STAKE`.\n"
            "  - **Exceção (sizing no executor)**: se `EXECUTOR_BACKPRE_FAST_STAKE_ENABLE=1`, o executor pode sobrescrever stake em Back:\n"
            "    - `slippage_pre_pct < EXECUTOR_BACK_STAKE_SLIP_NEG_LIMIT_PCT` ⇒ `EXECUTOR_BACK_STAKE_SLIP_NEG` (ex.: 40)\n"
            "    - `EXECUTOR_BACK_STAKE_SLIP_NEG_LIMIT_PCT <= slippage_pre_pct <= EXECUTOR_BACK_STAKE_SLIP_POS_LIMIT_PCT` ⇒ `EXECUTOR_BACK_STAKE_SLIP_MID` (ex.: 20)\n"
            "    - `slippage_pre_pct > EXECUTOR_BACK_STAKE_SLIP_POS_LIMIT_PCT` ⇒ `EXECUTOR_BACK_STAKE_SLIP_POS` (default pode ser 20)\n"
            "    - opcional: gate de latência com `EXECUTOR_BACK_LATENCY_GATE_ENABLE=1` e `EXECUTOR_BACK_LATENCY_GATE_MAX_SEC`.\n"
            "- Em **Lay**: o executor recebe stake, mas o risco relevante é a **liability**, aproximadamente `liability ≈ stake × (odd - 1)`.\n"
            "- Importante: o Kelly/caps que aparece no relatório OOS é **simulação/diagnóstico** do walk-forward; ele não está sendo aplicado no executor/bridge neste momento.\n\n"
        )
        extra.append("**Lay em `ws_gate_lay`: é só pós-reversão?**\n\n")
        extra.append(
            "- Sim: o audit `v5.1-ws-gate-lay` só abre ticket Lay quando passa pelo gate de **queda** (ex.: >2% em 5s): `WS(t+offset) < ratio × WS(t0)`.\n"
            "- Isso significa que, mesmo em shadow, a amostra Lay desse audit representa apenas casos em que houve a movimentação (reversão/queda) definida pela estratégia.\n\n"
        )

    # 99.6 Config efetiva (filtros ativos) — executor/audit/bridge/OOS
    extra.append("\n### 99.6 Filtros ativos (config efetiva)\n\n")
    extra.append(
        "_Nota: esta seção reflete as variáveis carregadas pelo `daily_full_report` (via `.env`). "
        "Services do systemd podem ter overrides (`Environment=`) que não aparecem aqui; use `systemctl show` para confirmar no VPS._\n\n"
    )
    def _env(k: str, default: str = "") -> str:
        v = os.getenv(k, default)
        return str(v) if v is not None else ""

    extra.append("**Executor**\n\n")
    extra.append("| chave | valor |\n|---|---|\n")
    for k in [
        "EXECUTOR_ALLOW_LIVE",
        "EXECUTOR_WORKERS",
        "EXECUTOR_QUEUE_MAX",
        "EXECUTOR_CAP_WINDOW_SEC",
        "EXECUTOR_CAP_MAX",
        "EXECUTOR_BACKPRE_FAST_STAKE_ENABLE",
        "EXECUTOR_BACK_STAKE_SIZING_ENABLE",
        "EXECUTOR_BACK_STAKE_SLIP_NEG_LIMIT_PCT",
        "EXECUTOR_BACK_STAKE_SLIP_POS_LIMIT_PCT",
        "EXECUTOR_BACK_STAKE_SLIP_NEG",
        "EXECUTOR_BACK_STAKE_SLIP_MID",
        "EXECUTOR_BACK_STAKE_SLIP_POS",
        "EXECUTOR_BACK_LATENCY_GATE_ENABLE",
        "EXECUTOR_BACK_LATENCY_GATE_MAX_SEC",
        "EXECUTOR_FAST_PMM",
        "EXECUTOR_PMM_TIMEOUT_SEC",
        "EXECUTOR_PMM_MIN_WAIT_SEC",
        "EXECUTOR_PMM_IDLE_TIMEOUT_SEC",
        "EXECUTOR_BETSLIP_CACHE_MAX_KEYS",
    ]:
        extra.append(f"| {k} | `{_env(k)}` |\n")
    extra.append("\n")

    extra.append(f"**Audit {str(cfg.hypothesis_type).upper()}**\n\n")
    extra.append("| chave | valor |\n|---|---|\n")
    for k in [
        "AUDIT_MODE",
        "AUDIT_API_SIDES",
        "AUDIT_EXECUTOR_WORKERS",
        "AUDIT_TEMPORAL_WORKERS",
        "AUDIT_MAX_QUEUE_DEPTH",
        "AUDIT_MAX_QUEUE_WAIT_MS",
        "WS_SAMPLE_OFFSETS_SEC",
        "GATE_DROP_OFFSET_SEC",
        "GATE_DROP_RATIO",
        "GATE_RISE_OFFSET_SEC",
        "GATE_RISE_RATIO",
        "GATE_OPEN_WINDOW_SEC",
        "GATE_OPEN_MAX",
        "GATE_MAX_LATE_SEC",
        "GATE_LAY_REFRESH_TIMES_SEC",
    ]:
        extra.append(f"| {k} | `{_env(k)}` |\n")
    extra.append("\n")
    extra.append(
        "_Nota (Back vs Lay): o `AUDIT_MODE` acima costuma refletir o serviço principal (ex.: `ws_gate_lay`). "
        "Em operação real, o **Back** pode vir de um serviço separado (ex.: `betinasia-audit-api-back`, `audit_version=v5.2-api-back`) "
        "ou de uma variante `ws_gate_back` (dependendo do deploy). Para confirmar o que rodou nas últimas 24h, veja `99.5 Auditoria (DB)`._\n\n"
    )

    # Interpretação operacional (audit/entrada) para reduzir ambiguidade
    extra.append("**Interpretação operacional (timing de entrada)**\n\n")
    extra.append("| Item | Regra efetiva |\n|---|---|\n")
    extra.append("| Back (mais cedo possível) | Depende do executor: `EXECUTOR_FAST_PMM`, `EXECUTOR_PMM_MIN_WAIT_SEC`, `EXECUTOR_PMM_TIMEOUT_SEC` (ver tabela Executor). |\n")
    extra.append(
        "| Lay (reversão vs fim) | Depende do `AUDIT_MODE`/audit_version: `ws_gate_lay` abre Lay só quando o gate em `t+GATE_DROP_OFFSET_SEC` passa; "
        "`ws_reversal_lay` tende a entrar no pós-reversal; `ws_only` usa a série WS (offsets até o último ponto, tipicamente 30s). |\n"
    )
    extra.append("\n")

    extra.append("**Bridge**\n\n")
    extra.append("| chave | valor |\n|---|---|\n")
    for k in [
        "BRIDGE_MODE",
        "BRIDGE_EXEC_SIDE",
        "BRIDGE_STAKE",
        "BRIDGE_POLL_SEC",
        "BRIDGE_LOOKBACK_SEC",
        "BRIDGE_MAX_PER_CYCLE",
        "BRIDGE_PREMATCH_ONLY",
        "BRIDGE_POLICY_JSON",
        "BRIDGE_POLICY_RELOAD_SEC",
        "BRIDGE_POLICY_USE_BASE",
        "BRIDGE_MIN_LIMIT",
    ]:
        extra.append(f"| {k} | `{_env(k)}` |\n")
    extra.append("\n")

    extra.append("**OOS / Walk-forward (daily)**\n\n")
    extra.append("| chave | valor |\n|---|---|\n")
    for k in [
        "DAILY_OOS_DIRECTION",
        "DAILY_OOS_VERSIONS",
        "DAILY_OOS_LOOKBACK_DAYS",
        "DAILY_WF_TRAIN_MODE",
        "DAILY_WF_TRAIN_DAYS",
        "DAILY_WF_TEST_DAYS",
        "DAILY_WF_STEP_DAYS",
        "DAILY_WF_SIDES",
        "DAILY_WF_REGIMES",
        "DAILY_WF_BACKPRE_SLIP_MAX",
        "DAILY_WF_BACKPRE_SLIP_FIELD",
        "DAILY_WF_BACKPRE_FAST_MAX_LAG_MS",
        "DAILY_WF_KEY_BY_LEAGUE",
        "DAILY_WF_KEY_BY_LEAGUE_SCOPE",
        "DAILY_WF_AH_MAX_ABS_LINE",
        "DAILY_WF_AH_SCOPE",
        "DAILY_WF_EXCLUDE_EXEC_BUCKETS_BACK",
    ]:
        extra.append(f"| {k} | `{_env(k)}` |\n")
    extra.append("\n")

    # 99.4 Aderência OOS (policy por dia × execução)
    try:
        if isinstance(adh_day, dict) and isinstance(adh_day.get("per_day"), list) and adh_day.get("per_day"):
            extra.append("\n### 99.4 Aderência OOS (portfolio por dia × execução)\n\n")
            extra.append(f"- Arquivo (curto): `{adh_short_json}`\n")
            if adh_long_json:
                extra.append(f"- Arquivo (acumulado/slippage): `{adh_long_json}`\n")
            extra.append(f"- Policy current: `{cfg.wf_policy_current}`\n\n")

            extra.append("**Resumo (últimos dias)**\n\n")
            extra.append("| Dia | Ativas (keys) | Bridge rows | Skipped(not_active) | Exec rows | LIVE_OK | DRY_OK | Back bloqueadas (slip<=-2%; cov) | Lay bloqueadas (slip>2%; cov) | ΔP&L cf (placar; cov) | P&L Back | ROI Back | P&L Lay | ROI Lay/liab | P&L total |\n")
            extra.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for it in adh_day.get("per_day") or []:
                if not isinstance(it, dict):
                    continue
                pol = it.get("policy") if isinstance(it.get("policy"), dict) else {}
                nkeys = pol.get("n_active_keys")
                bridge_rows = 0
                skipped_na = 0
                for b in (it.get("bridge") or []):
                    if isinstance(b, dict):
                        bridge_rows += int(b.get("n_rows") or 0)
                        skipped_na += int(b.get("n_not_active") or 0)
                ex = it.get("execution") if isinstance(it.get("execution"), dict) else {}
                sc = ex.get("status_counts") if isinstance(ex.get("status_counts"), dict) else {}
                live_ok = int(sc.get("LIVE_OK") or 0)
                dry_ok = int(sc.get("DRY_OK") or 0)
                back = ex.get("back") if isinstance(ex.get("back"), dict) else {}
                lay = ex.get("lay") if isinstance(ex.get("lay"), dict) else {}
                pnl_b = float(back.get("pnl_sum") or 0.0)
                pnl_l = float(lay.get("pnl_sum") or 0.0)
                # contrafactual slippage gate (por dia; somente cobertos por placar+odd)
                cf = it.get("slippage_filter_counterfactual") if isinstance(it.get("slippage_filter_counterfactual"), dict) else {}
                cfb = cf.get("back") if isinstance(cf.get("back"), dict) else {}
                cfl = cf.get("lay") if isinstance(cf.get("lay"), dict) else {}
                nblock_back = None
                nblock_lay = None
                dpnl_cf = None
                try:
                    nblock_back = int(cfb.get("n") or 0) - int(cfb.get("n_filtered") or 0)
                    nblock_lay = int(cfl.get("n") or 0) - int(cfl.get("n_filtered") or 0)
                    pnl_cf = float(cfb.get("pnl_filtered") or 0.0) + float(cfl.get("pnl_filtered") or 0.0)
                    pnl_base = float(cfb.get("pnl") or 0.0) + float(cfl.get("pnl") or 0.0)
                    dpnl_cf = float(pnl_cf - pnl_base)
                except Exception:
                    nblock_back = None
                    nblock_lay = None
                    dpnl_cf = None
                extra.append(
                    f"| {it.get('day')} | {nkeys if nkeys is not None else '—'} | {bridge_rows} | {skipped_na} | "
                    f"{int(ex.get('n_exec_rows') or 0)} | {live_ok} | {dry_ok} | {nblock_back if nblock_back is not None else '—'} | {nblock_lay if nblock_lay is not None else '—'} | {_fmt_num(dpnl_cf,2)} | {_fmt_num(pnl_b,2)} | {_fmt_pct(back.get('roi_pct'))} | "
                    f"{_fmt_num(pnl_l,2)} | {_fmt_pct(lay.get('roi_pct_per_liability'))} | {_fmt_num(pnl_b + pnl_l,2)} |\n"
                )
            extra.append("\n")

            # Potencial (30d) pela banca que maximiza lucro na sensibilidade OOS (se disponível)
            try:
                sens = _extract_md_block(
                    oos_txt,
                    start="### 12.2b Sensibilidade por banca",
                    until_any=["### 12.2c Sensibilidade por banca", "### 12.3 ", "### 1.2c Sensibilidade por banca", "### 1.3 "],
                )
                if not sens.strip():
                    sens = _extract_md_block(
                        oos_txt,
                        start="### 1.2b Sensibilidade por banca",
                        until_any=["### 1.2c Sensibilidade por banca", "### 1.3 ", "### 12.2c Sensibilidade por banca", "### 12.3 "],
                    )
                best = None
                if sens:
                    for ln in sens.splitlines():
                        if not ln.startswith("|") or ln.strip().startswith("|---"):
                            continue
                        cols = [c.strip() for c in ln.strip().strip("|").split("|")]
                        if len(cols) < 6 or cols[0].lower().startswith("banca"):
                            continue
                        bank_ref = _parse_md_number(cols[0])
                        turn_30 = _parse_md_number(cols[1])
                        prof_30 = _parse_md_number(cols[2])
                        bank_eff = _parse_md_number(cols[3])
                        roi_bank = _parse_md_number(cols[4])
                        dd_p95 = _parse_md_number(cols[5])
                        if prof_30 is None:
                            continue
                        if best is None or float(prof_30) > float(best["profit_30d_exp"]):
                            best = {
                                "bank_ref": bank_ref,
                                "turn_30d": turn_30,
                                "profit_30d_exp": prof_30,
                                "bank_eff": bank_eff,
                                "roi_bank_30d": roi_bank,
                                "dd_p95": dd_p95,
                            }
                if best:
                    # share observado (últimos dias) para decompor back/lay como estimativa
                    pnl_b = pnl_l = 0.0
                    for it in adh_day.get("per_day") or []:
                        ex = it.get("execution") if isinstance(it.get("execution"), dict) else {}
                        back = ex.get("back") if isinstance(ex.get("back"), dict) else {}
                        lay = ex.get("lay") if isinstance(ex.get("lay"), dict) else {}
                        pnl_b += float(back.get("pnl_sum") or 0.0)
                        pnl_l += float(lay.get("pnl_sum") or 0.0)
                    tot = pnl_b + pnl_l
                    w_b = (pnl_b / tot) if tot else 0.5
                    w_l = 1.0 - w_b
                    extra.append("**Potencial de lucro (30d) pela banca ótima (sensibilidade OOS)**\n\n")
                    extra.append(f"- Banca ref (grid): `{_fmt_num(best.get('bank_ref'),2)}` | banca rec. (max): `{_fmt_num(best.get('bank_eff'),2)}`\n")
                    extra.append(f"- Lucro 30d (exp.): `{_fmt_num(best.get('profit_30d_exp'),2)}` | ROI/banca 30d (exp.): `{_fmt_num(best.get('roi_bank_30d'),2)}%` | DD p95 (30d): `{_fmt_num(best.get('dd_p95'),2)}`\n")
                    extra.append(
                        f"- Decomposição *estimada* por lado (proporcional ao P&L observado na janela): total `{_fmt_num(best.get('profit_30d_exp'),2)}` → "
                        f"Back `{_fmt_num(float(best.get('profit_30d_exp'))*w_b,2)}` | Lay `{_fmt_num(float(best.get('profit_30d_exp'))*w_l,2)}`\n\n"
                    )
            except Exception:
                pass

            # Slippage × ROI (com sinal): acumulado na janela (para análise estatística)
            raw_total = adh_slip.get("slippage_vs_roi_raw_total") if (isinstance(adh_slip, dict) and isinstance(adh_slip.get("slippage_vs_roi_raw_total"), dict)) else {}
            if isinstance(raw_total, dict) and raw_total:
                try:
                    rg = adh_slip.get("slippage_range", None) if isinstance(adh_slip, dict) else None
                    if not isinstance(rg, dict) or not rg:
                        rg = adh_slip.get("range", {}) if isinstance(adh_slip, dict) else {}
                    span = rg.get("span_days") if isinstance(rg, dict) else None
                    slip_cut = (adh_slip.get("slippage_start_day_local") if isinstance(adh_slip, dict) else None) or None
                    extra.append(
                        f"**Slippage × ROI (raw, com sinal; 3 buckets) — acumulado (range: `{rg.get('start_day')}` → `{rg.get('end_day')}`; span_days=`{int(span or 0)}`; cut=`{slip_cut}`)**\n\n"
                    )
                except Exception:
                    extra.append("**Slippage × ROI (raw, com sinal; 3 buckets) — acumulado**\n\n")
                for side_key, title in (("back", "Back (ROI por stake)"), ("lay", "Lay (ROI por liability)")):
                    blk = raw_total.get(side_key) if isinstance(raw_total.get(side_key), dict) else {}
                    buckets0 = blk.get("buckets") if isinstance(blk.get("buckets"), list) else []
                    buckets = _slip_raw_3bucket_rows(buckets0)
                    if not any(int(r.get("n") or 0) > 0 for r in buckets):
                        continue
                    extra.append(f"- **{title}**\n\n")
                    extra.append("| Bucket slippage_raw_pct | n | ROI mean |\n|---|---:|---:|\n")
                    for b in buckets:
                        extra.append(f"| {b.get('bucket')} | {int(b.get('n') or 0)} | {_fmt_pct(b.get('roi_mean'))} |\n")
                    extra.append("\n")
            else:
                extra.append(
                    "_Slippage × ROI (por bucket) indisponível na janela: precisa de execuções com odd (decision/final) **e** placar (ROI) no DB._\n\n"
                )
    except Exception:
        pass

    # 99.5 Auditoria (DB): motivos de no-OK por versão + qualidade dos OK
    try:
        rep = audit_rep
        if isinstance(rep, dict) and isinstance(rep.get("by_version"), list) and rep.get("by_version"):
            extra.append("\n### 99.5 Auditoria (DB) — motivos de no-OK (por versão)\n\n")
            extra.append(f"- Arquivo: `{audit_json}`\n")
            extra.append(f"- Janela: últimas **{rep.get('hours')}h** (desde `{rep.get('since_utc')}`)\n\n")
            extra.append("**Definições (colunas)**\n\n")
            extra.append(
                "- **OK**: `status='OK'` no `betslip_audit_results` (a auditoria concluiu com sucesso).\n"
                "- **OK com betslip_odd**: subset de OK em que `betslip_odd` está preenchido (houve snapshot do ticket/odds).\n"
                "- **OK valid**: subset de OK em que `is_valid_opportunity=true` (passou o critério operacional de “oportunidade executável”).\n"
                "  - Na prática, o `is_valid_opportunity` tende a cair quando `difference_pct` está fora do range aceito (edge muito pequeno <2% ou mismatch >10%) ou quando campos essenciais do ticket estão ausentes.\n\n"
            )
            extra.append("**Glossário rápido (`audit_version`)**\n\n")
            extra.append("| padrão | significado |\n|---|---|\n")
            extra.append("| `v5.2-api-back` | Back via API (serviço back-only); tende a abrir betslip e medir limites/odds. |\n")
            extra.append("| `v5.1-ws-gate-lay` | Lay via WS gate (queda em 5s); só abre ticket quando o gate passa. |\n")
            extra.append("| `v5.4-ws-reversal-lay` | Lay no pós-reversal; volume baixo pode ser “evento raro” (depende de reversões). |\n")
            extra.append("| `v5.3-ws-gate-back` | Back via WS gate; se `OK` é baixo, costuma indicar gate muito restritivo, parse/click falhando, ou credenciais/sessão instável. |\n")
            extra.append("| `v4.*` / `v1.*` | versões antigas/legadas do pipeline (API/WS), úteis para comparação histórica. |\n")
            extra.append("\n")
            extra.append("| audit_version | total | OK | OK com betslip_odd | OK valid | top no-OK |\n")
            extra.append("|---|---:|---:|---:|---:|---|\n")
            for v in rep.get("by_version") or []:
                if not isinstance(v, dict):
                    continue
                sc = v.get("status_counts") if isinstance(v.get("status_counts"), dict) else {}
                total = int(v.get("total") or 0)
                nok = 0
                try:
                    nok = int(sc.get("OK") or 0)
                except Exception:
                    nok = 0
                # top no-OK
                pairs = []
                for k, cnt in sc.items():
                    if str(k) == "OK":
                        continue
                    try:
                        pairs.append((str(k), int(cnt)))
                    except Exception:
                        continue
                pairs.sort(key=lambda x: x[1], reverse=True)
                top = ", ".join([f"{k}={c}" for k, c in pairs[:4]]) if pairs else "—"
                extra.append(
                    f"| {v.get('audit_version')} | {total} | {nok} | {int(v.get('ok_with_bs') or 0)} | {int(v.get('ok_valid') or 0)} | {top} |\n"
                )
            extra.append("\n")

            # Diagnóstico (OK): por que OK_with_bs >> OK_valid?
            extra.append("**Diagnóstico dos OK (por versão): buckets de |difference_pct|**\n\n")
            extra.append(
                "_Leitura: `OK valid` tende a ser aproximadamente o bucket `2% ≤ |difference_pct| ≤ 10%` (dependendo da regra vigente)._\n\n"
            )
            extra.append("| audit_version | OK diff nulo | OK |diff|<2% | OK 2–10% | OK |diff|>10% |\n")
            extra.append("|---|---:|---:|---:|---:|\n")
            for v in rep.get("by_version") or []:
                if not isinstance(v, dict):
                    continue
                extra.append(
                    f"| {v.get('audit_version')} | {int(v.get('ok_diff_null') or 0)} | {int(v.get('ok_absdiff_lt2') or 0)} | {int(v.get('ok_absdiff_2_10') or 0)} | {int(v.get('ok_absdiff_gt10') or 0)} |\n"
                )
            extra.append("\n")

            # Top api_error (quando houver) para explicar API_FAILED/NO_SESSION/etc.
            tev = rep.get("top_errors_by_version") if isinstance(rep.get("top_errors_by_version"), dict) else {}
            if tev:
                extra.append("**Top erros (api_error) por versão**\n\n")
                for ver, xs in tev.items():
                    if not isinstance(xs, list) or not xs:
                        continue
                    extra.append(f"- `{ver}`:\n")
                    for it in xs[:5]:
                        if not isinstance(it, dict):
                            continue
                        st = str(it.get("status") or "NA")
                        err = str(it.get("api_error") or "").strip()
                        n = int(it.get("n") or 0)
                        if err:
                            err = (err[:180] + "…") if len(err) > 180 else err
                            extra.append(f"  - {st} ×{n}: `{err}`\n")
                    extra.append("\n")
    except Exception:
        pass

    combined_md = day_dir / "report_daily.md"
    insample_wrapped = ""
    if insample_txt.strip():
        insample_wrapped = "## 3) In-sample (detalhe)\n\n" + _demote_h2_to_h3(insample_txt.strip() + "\n")

    oos_annex = ""
    if oos_as_annex and oos_txt.strip():
        oos_annex = "## Anexo A) OOS walk-forward (Seção 12)\n\n" + _demote_h2_to_h3(oos_txt.strip() + "\n")

    # Ajuste adicional (capacidade): sensibilidade por banca com efeito do gate de slippage.
    # Usa (a) curvas base exportadas pelo OOS (wf_bank_sensitivity.json) e (b) contrafactual observado (execuções com placar).
    try:
        sens = _read_json(bank_sens_json) if "bank_sens_json" in locals() else None
        cf_src = None
        if isinstance(adh_long, dict) and isinstance(adh_long.get("slippage_filter_counterfactual"), dict):
            cf_src = adh_long.get("slippage_filter_counterfactual")
        elif isinstance(adh_short, dict) and isinstance(adh_short.get("slippage_filter_counterfactual"), dict):
            cf_src = adh_short.get("slippage_filter_counterfactual")
        if isinstance(cf_src, dict):
            # Sempre imprime um bloco diagnóstico, mesmo se o JSON de sensibilidade estiver ausente (para não “sumir” no PDF).
            sens_ok = bool(isinstance(sens, dict) and isinstance(sens.get("scenarios"), dict) and sens.get("scenarios"))
            block = []
            block.append("\n### Ajuste operacional: Sensibilidade por banca com gate de slippage (contrafactual)\n\n")
            block.append(
                "_Leitura: aplica a regra `Back: pula slippage_raw_pct<=-2%` e `Lay: pula slippage_raw_pct>2%` "
                "como um ajuste de capacidade, usando a evidência contrafactual nas execuções cobertas por placar. "
                "O ajuste é um **proxy**: usa exposição observada (Back=stake, Lay=liability) para estimar redução de N/turnover e mudança de ROI._\n\n"
            )
            try:
                block.append(
                    f"- Fonte OOS (curvas por banca): `{str(bank_sens_json)}` (existe={('sim' if (bank_sens_json.exists() if 'bank_sens_json' in locals() else False) else 'não')}; "
                    f"sens_ok={('sim' if sens_ok else 'não')}).\n\n"
                )
                if isinstance(sens, dict) and isinstance(sens.get("warn"), str) and sens.get("warn"):
                    block.append(f"- Aviso do export: `{sens.get('warn')}`.\n\n")
            except Exception:
                pass

            if not sens_ok:
                block.append(
                    "_Aviso: não foi possível aplicar o ajuste na sensibilidade por banca porque o export `wf_bank_sensitivity.json` está ausente/vazio/ilegível. "
                    "Isso não afeta o OOS em si; apenas impede esta tabela ajustada. "
                    "Se persistir, verifique se o daily está rodando a versão mais recente do `analyze_contexto_operacao_b808_robust_report.py` com "
                    "`--wf-export-bank-sensitivity-json` habilitado._\n\n"
                )
            else:
                sens = sens or {}
                back = cf_src.get("back") if isinstance(cf_src.get("back"), dict) else {}
                lay = cf_src.get("lay") if isinstance(cf_src.get("lay"), dict) else {}
                # "exposição" do contrafactual (apenas cobertura com placar+odd).
                # Back: stake. Lay: preferir liability (sempre existe no contrafactual); se stake existir (versões novas),
                # podemos usar stake, mas não dependemos dele.
                back_exp_base = float(back.get("stake") or 0.0)
                back_exp_filt = float(back.get("stake_filtered") or 0.0)
                lay_exp_base = (
                    float(lay.get("liability") or 0.0)
                    if lay.get("liability") is not None
                    else float(lay.get("stake") or 0.0)
                )
                lay_exp_filt = (
                    float(lay.get("liability_filtered") or 0.0)
                    if lay.get("liability_filtered") is not None
                    else float(lay.get("stake_filtered") or 0.0)
                )
                exp_base = float(back_exp_base) + float(lay_exp_base)
                exp_filt = float(back_exp_filt) + float(lay_exp_filt)
                pnl_base = float(back.get("pnl") or 0.0) + float(lay.get("pnl") or 0.0)
                pnl_filt = float(back.get("pnl_filtered") or 0.0) + float(lay.get("pnl_filtered") or 0.0)
                n_base = int(back.get("n") or 0) + int(lay.get("n") or 0)
                n_filt = int(back.get("n_filtered") or 0) + int(lay.get("n_filtered") or 0)
                exp_factor = _safe_div(exp_filt, exp_base)
                n_factor = _safe_div(n_filt, n_base)
                roi_base = _safe_div(pnl_base, exp_base)
                roi_filt = _safe_div(pnl_filt, exp_filt)
                roi_factor = _safe_div(roi_filt, roi_base) if (roi_base is not None and roi_filt is not None and roi_base != 0) else None
                profit_factor = _safe_div(pnl_filt, pnl_base) if pnl_base != 0 else None

                block.append(
                    f"- Fatores (da janela contrafactual): pass_exposição≈`{_fmt_num((exp_factor*100.0) if exp_factor is not None else None,2)}%`, "
                    f"pass_N≈`{_fmt_num((n_factor*100.0) if n_factor is not None else None,2)}%`, "
                    f"ROI_mult≈`{_fmt_num(roi_factor,3)}` , lucro_mult≈`{_fmt_num(profit_factor,3)}`.\n\n"
                )
                if exp_factor is None or profit_factor is None:
                    block.append(
                        f"_Aviso: ajuste não pôde ser aplicado (exp_base={_fmt_num(exp_base,2)}, exp_filt={_fmt_num(exp_filt,2)}, pnl_base={_fmt_num(pnl_base,2)})._\n\n"
                    )
                else:
                    scen = sens.get("scenarios") if isinstance(sens.get("scenarios"), dict) else {}
                    name_map = {
                        "12.2b_base": "1.2b (base)",
                        "12.2c_eq4_signals_sqrt": "1.2c (EQ 4%/4% cap50%, signals_sqrt)",
                        "12.2e_eq4_fixed": "1.2e (EQ 4%/4% cap50%, fixed)",
                        "12.2d_eq2_signals_sqrt": "1.2d (EQ 2%/2% cap33%, signals_sqrt)",
                    }
                    for name, payload in scen.items():
                        rows = payload.get("rows") if isinstance(payload, dict) else None
                        if not isinstance(rows, list) or not rows:
                            continue
                        ttl = name_map.get(str(name), str(name))
                        block.append(f"**{ttl} — com gate de slippage (ajuste proxy)**\n\n")
                        block.append("| Banca (ref) | Turnover 30d (adj, proxy) | Lucro 30d (adj) | ROI/banca 30d (adj) | n_after_budget (adj) |\n")
                        block.append("|---:|---:|---:|---:|---:|\n")
                        for r in rows:
                            if not isinstance(r, dict):
                                continue
                            br = r.get("bank_ref")
                            t0 = _safe_float(r.get("turn_30d"))
                            p0 = _safe_float(r.get("profit_30d_exp"))
                            beff = _safe_float(r.get("bank_eff"))
                            n0 = None
                            try:
                                h = r.get("limit_hits") if isinstance(r.get("limit_hits"), dict) else {}
                                n0 = int(h.get("n_after_budget")) if h.get("n_after_budget") is not None else None
                            except Exception:
                                n0 = None
                            # turnover: proxy pelo pass_exposição (principalmente para Lay, onde exposição é liability)
                            t1 = (float(t0) * float(exp_factor)) if t0 is not None else None
                            p1 = (float(p0) * float(profit_factor)) if p0 is not None else None
                            roi_bank = (float(p1) / float(beff) * 100.0) if (p1 is not None and beff is not None and beff > 0) else None
                            n1 = int(round(float(n0) * float(n_factor))) if (n0 is not None and n_factor is not None) else None
                            block.append(
                                f"| {_fmt_num(br,2)} | {_fmt_num(t1,2)} | {_fmt_num(p1,2)} | {_fmt_num(roi_bank,2)}% | {n1 if n1 is not None else '—'} |\n"
                            )
                        block.append("\n")

            add_txt = "".join(block)
            if oos_as_annex and oos_annex:
                oos_annex = oos_annex + _demote_h2_to_h3(add_txt)
            else:
                extra.append("\n\n## Anexo B) Ajuste operacional (slippage gate × capacidade)\n\n")
                extra.append(add_txt)
    except Exception:
        pass
    # Cabeçalho com o dia relativo ao report para facilitar leitura/auditoria entre dias.
    report_header = (
        "# Daily Report BetinAsia\n\n"
        f"- Dia do relatório (UTC): `{day}`\n"
        f"- Gerado em (UTC): `{ts.isoformat()}`\n\n"
    )
    combined_core = report_header + "".join(s0) + "".join(s1) + insample_wrapped
    combined_md.write_text(combined_core + "".join(extra) + oos_annex, encoding="utf-8")

    # 5) PDF
    pdf = day_dir / "report_daily.pdf"
    renderer = Path(__file__).resolve().parent.parent / "docs" / "render_markdown_to_pdf.py"
    subprocess.run([sys.executable, str(renderer), str(combined_md), str(pdf)], check=True)

    out = {
        "ts": ts.isoformat(),
        "day_dir": str(day_dir),
        "pdf": str(pdf),
        "pdf_size_mb": round(float(pdf.stat().st_size) / (1024.0 * 1024.0), 2) if pdf.exists() else None,
        "policy_current": str(cfg.wf_policy_current),
        "policy_publish": dict(policy_publish_info or {}),
    }

    # 6) Telegram
    if cfg.send_telegram:
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        if token and chat_id and pdf.exists():
            retries = max(1, int(float(os.getenv("DAILY_TELEGRAM_RETRIES", "2") or 2)))
            retry_sleep_sec = max(0.0, float(os.getenv("DAILY_TELEGRAM_RETRY_SLEEP_SEC", "3.0") or 3.0))

            ok = False
            last_status: Optional[int] = None
            last_err = ""
            for i in range(retries):
                ok, st, err = _telegram_send_document(
                    token,
                    chat_id,
                    file_path=pdf,
                    caption=f"Relatório diário BetinAsia ({day})",
                )
                last_status = st
                last_err = err
                if ok:
                    break
                if i < (retries - 1) and retry_sleep_sec > 0:
                    time.sleep(retry_sleep_sec)

            out["telegram_sent"] = bool(ok)
            out["telegram_attempts"] = int(retries)
            out["telegram_http_status"] = int(last_status) if last_status is not None else None
            out["telegram_error"] = str(last_err or "")[:500] if not ok else ""

            if not ok:
                logger.warning(
                    "Daily report Telegram send failed: "
                    f"status={last_status} err={str(last_err or '')[:220]} pdf={pdf} size_mb={out.get('pdf_size_mb')}"
                )
                # fallback: tenta ao menos avisar no chat que o envio do PDF falhou.
                _telegram_send_message(
                    token,
                    chat_id,
                    (
                        f"[daily_full_report] Falha ao enviar PDF ({day}). "
                        f"status={last_status or '-'} err={str(last_err or '')[:220]}"
                    ),
                )
        else:
            out["telegram_sent"] = False
            out["telegram_error"] = "missing_token_or_chat_or_pdf"

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Relatório diário completo: OOS + execution KPIs + accounting + PDF + Telegram.")
    ap.add_argument("--out-dir", default=os.getenv("DAILY_REPORT_OUT_DIR", "logs/daily_reports"))
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))

    # Se rodando manualmente, garante que .env seja carregado antes do cfg.
    _load_env_file(Path(os.getenv("ENV_FILE", ".env")))

    cfg = DailyReportCfg(out_dir=Path(str(args.out_dir)))
    import asyncio

    out = asyncio.run(run_daily_full(cfg))
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

