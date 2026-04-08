from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from loguru import logger
from sqlalchemy import text

from storage.database import Database


def _add_float(d: Dict[str, float], k: str, v: Optional[float]) -> None:
    try:
        if not k:
            return
        if v is None:
            return
        d[k] = float(d.get(k, 0.0)) + float(v)
    except Exception:
        return


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _parse_iso(s: str) -> Optional[datetime]:
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


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _json_safe(x: Any) -> Any:
    if isinstance(x, datetime):
        return x.isoformat()
    if isinstance(x, date):
        return x.isoformat()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_json_safe(v) for v in list(x)]
    return x


def _iter_dates(d0: date, d1: date) -> Iterable[date]:
    cur = d0
    while cur <= d1:
        yield cur
        cur += timedelta(days=1)


def _local_day_bounds_utc(*, day: date, tz_name: str) -> Tuple[datetime, datetime]:
    """
    Retorna (start_utc, end_utc) do dia local [00:00, 24:00) em tz_name.
    """
    tz = timezone.utc
    try:
        from zoneinfo import ZoneInfo  # type: ignore

        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc
    start_local = datetime.combine(day, time(0, 0, 0), tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def _norm_line(line: str) -> str:
    return (str(line or "").strip()).replace(",", ".").replace("−", "-")


def _mult_back_from_scores(line: Any, side: str, hs: Any, aws: Any) -> Optional[float]:
    """
    Multiplicador do AH para a seleção "Back" (stake=1):
    +1.0 win, +0.5 half-win, 0 push, -0.5 half-loss, -1 loss.
    """
    if hs is None or aws is None:
        return None
    try:
        goal_diff = int(hs) - int(aws)
    except Exception:
        return None
    try:
        sel = (side or "").strip().lower()
        raw = str(line).strip().replace(",", ".").replace("−", "-")
        ah = float(raw)
        # Convenção: quando `line` vem sem sinal (ex.: "2"), interpretamos como magnitude
        # do handicap do `side` (home:+line, away:+line). Convertendo para handicap do HOME:
        # - side=home => home_handicap=+line
        # - side=away => home_handicap=-line
        # Se vier com sinal (ex.: "-0.5"), tratamos como handicap do HOME já assinado.
        home_handicap = ah if (raw.startswith("+") or raw.startswith("-")) else (ah if sel == "home" else -ah)
    except Exception:
        return None
    if sel == "home":
        adjusted = goal_diff + home_handicap
    elif sel == "away":
        adjusted = -goal_diff - home_handicap
    else:
        return None
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


def _roi_lay_pct_per_liability(lay_odd: float, mult_back: float) -> Optional[float]:
    """
    Convenção alinhada ao relatório b808: ROI por liability para Lay.
    - mult_back é o multiplicador da seleção (se fosse Back).
    """
    liab = max(0.0, lay_odd - 1.0)
    if liab <= 0:
        return None
    if mult_back < 0:
        return (-mult_back) / liab * 100.0
    if mult_back > 0:
        return (-mult_back) * 100.0
    return 0.0


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    try:
        if len(xs) != len(ys) or len(xs) < 3:
            return None
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        vx = sum((x - mx) ** 2 for x in xs)
        vy = sum((y - my) ** 2 for y in ys)
        if vx <= 0 or vy <= 0:
            return None
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        return float(cov / (vx**0.5) / (vy**0.5))
    except Exception:
        return None


def _slip_raw_pct(*, odd_dec: Optional[float], odd_fin: Optional[float]) -> Optional[float]:
    if odd_dec is None or odd_fin is None or float(odd_dec) <= 0:
        return None
    return float((float(odd_fin) - float(odd_dec)) / float(odd_dec) * 100.0)


def _mean_se_ci95(ys: List[float]) -> Dict[str, Any]:
    """
    Estatística clássica (SEM + IC95% normal approx).
    Não assume independência perfeita (o ideal seria cluster por match), mas é um
    baseline útil para decisão e para comparar buckets.
    """
    try:
        n = int(len(ys))
        if n <= 0:
            return {"n": 0, "mean": None, "sd": None, "se": None, "ci95": None}
        mu = float(sum(float(x) for x in ys) / float(n))
        if n < 2:
            return {"n": n, "mean": mu, "sd": None, "se": None, "ci95": None}
        # sample sd
        var = float(sum((float(x) - mu) ** 2 for x in ys) / float(n - 1))
        sd = float(var ** 0.5)
        se = float(sd / (float(n) ** 0.5)) if sd > 0 else 0.0
        z = 1.96
        ci = {"lb": float(mu - z * se), "ub": float(mu + z * se)}
        return {"n": n, "mean": mu, "sd": sd, "se": se, "ci95": ci}
    except Exception:
        return {"n": int(len(ys)), "mean": None, "sd": None, "se": None, "ci95": None}


def _median(xs: List[float]) -> Optional[float]:
    try:
        if not xs:
            return None
        ys = sorted(float(x) for x in xs)
        n = len(ys)
        if n % 2 == 1:
            return float(ys[n // 2])
        return float((ys[n // 2 - 1] + ys[n // 2]) / 2.0)
    except Exception:
        return None


def _bucketize_latency_call_to_done_ms_with_context(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Bucketiza por call_to_done_ms (latência) e retorna estatísticas de ROI por bucket,
    incluindo ROI ponderado por exposição (stake) e contexto (odd/exposure median).
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
    out: List[Dict[str, Any]] = []
    for lab in order:
        sub = [r for r in rows if _lab(_safe_float(r.get("lat_ms"))) == lab]
        rois = [float(r.get("roi")) for r in sub if r.get("roi") is not None]
        if not rois:
            continue
        st = _mean_se_ci95(rois)
        odds = [float(r.get("odd")) for r in sub if r.get("odd") is not None]
        exps = [float(r.get("exposure")) for r in sub if r.get("exposure") is not None]
        exp_sum = float(sum(exps)) if exps else 0.0
        roi_w = None
        try:
            if exp_sum > 0:
                roi_w = float(
                    sum(float(r.get("roi")) * float(r.get("exposure")) for r in sub if r.get("roi") is not None and r.get("exposure") is not None)
                    / exp_sum
                )
        except Exception:
            roi_w = None
        out.append(
            {
                "bucket": lab,
                "n": int(st.get("n") or 0),
                "roi_mean": st.get("mean"),
                "roi_sd": st.get("sd"),
                "roi_se": st.get("se"),
                "roi_ci95": st.get("ci95"),
                "odd_median": _median(odds),
                "exposure_median": _median(exps),
                "exposure_sum": exp_sum,
                "roi_weighted": roi_w,
            }
        )
    return out


def _bucketize_slip_raw_pct_vs_latency_with_context(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Bucketiza por call_to_done_ms (latência) e retorna estatísticas de slippage_raw_pct por bucket,
    junto com ROI ponderado por exposição (ROIw) para ajudar a ligar execução (latência/slippage) a resultado.
    Espera rows no formato:
      {lat_ms, slip_raw_pct, roi, odd, exposure}
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
    out: List[Dict[str, Any]] = []
    for lab in order:
        sub = [r for r in rows if _lab(_safe_float(r.get("lat_ms"))) == lab]
        slips = [float(r.get("slip_raw_pct")) for r in sub if r.get("slip_raw_pct") is not None]
        if not slips:
            continue
        st = _mean_se_ci95(slips)
        odds = [float(r.get("odd")) for r in sub if r.get("odd") is not None]
        exps = [float(r.get("exposure")) for r in sub if r.get("exposure") is not None]
        exp_sum = float(sum(exps)) if exps else 0.0
        slip_w = None
        roi_w = None
        try:
            if exp_sum > 0:
                slip_w = float(
                    sum(float(r.get("slip_raw_pct")) * float(r.get("exposure")) for r in sub if r.get("slip_raw_pct") is not None and r.get("exposure") is not None)
                    / exp_sum
                )
                roi_w = float(
                    sum(float(r.get("roi")) * float(r.get("exposure")) for r in sub if r.get("roi") is not None and r.get("exposure") is not None) / exp_sum
                )
        except Exception:
            slip_w = None
            roi_w = None
        out.append(
            {
                "bucket": lab,
                "n": int(st.get("n") or 0),
                "slip_raw_mean": st.get("mean"),
                "slip_raw_sd": st.get("sd"),
                "slip_raw_se": st.get("se"),
                "slip_raw_ci95": st.get("ci95"),
                "slip_raw_median": _median(slips),
                "odd_median": _median(odds),
                "exposure_median": _median(exps),
                "exposure_sum": exp_sum,
                "slip_raw_weighted": slip_w,
                "roi_weighted": roi_w,
            }
        )
    return out


def _bucketize_3way_raw(pairs: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
    """
    Bucketiza por slippage_raw_pct (com sinal), em 3 faixas:
    - <= -2%
    - (-2%, 2%]
    - > 2%
    Retorna média de ROI por bucket.
    """
    outb: List[Dict[str, Any]] = []
    if not pairs:
        return outb
    buckets = [
        ("<= -2%", lambda s: s <= -2.0),
        ("(-2, 2]", lambda s: (s > -2.0) and (s <= 2.0)),
        ("> 2%", lambda s: s > 2.0),
    ]
    for lab, fn in buckets:
        ys = [roi for (slip, roi) in pairs if fn(float(slip))]
        if not ys:
            continue
        st = _mean_se_ci95([float(x) for x in ys])
        outb.append(
            {
                "bucket": lab,
                "n": int(st.get("n") or 0),
                "roi_mean": st.get("mean"),
                "roi_sd": st.get("sd"),
                "roi_se": st.get("se"),
                "roi_ci95": st.get("ci95"),
            }
        )
    return outb


def _bucketize_3way_raw_with_context(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    rows: [{slip_raw_pct, roi, odd, exposure}]
    Além de mean/SE/IC, retorna median(odd) e median(exposure) para interpretar ROIs extremos.
    """
    outb: List[Dict[str, Any]] = []
    if not rows:
        return outb
    buckets = [
        ("<= -2%", lambda s: s <= -2.0),
        ("(-2, 2]", lambda s: (s > -2.0) and (s <= 2.0)),
        ("> 2%", lambda s: s > 2.0),
    ]
    for lab, fn in buckets:
        sub = [r for r in rows if (r.get("slip_raw_pct") is not None) and fn(float(r.get("slip_raw_pct")))]
        if not sub:
            continue
        rois = [float(r.get("roi")) for r in sub if r.get("roi") is not None]
        if not rois:
            continue
        st = _mean_se_ci95(rois)
        odds = [float(r.get("odd")) for r in sub if r.get("odd") is not None]
        exps = [float(r.get("exposure")) for r in sub if r.get("exposure") is not None]
        exp_sum = float(sum(exps)) if exps else 0.0
        # ROI ponderado por exposição (mais robusto quando liabilities são minúsculas)
        roi_w = None
        try:
            if exp_sum > 0:
                roi_w = float(sum(float(r.get("roi")) * float(r.get("exposure")) for r in sub if r.get("roi") is not None and r.get("exposure") is not None) / exp_sum)
        except Exception:
            roi_w = None
        outb.append(
            {
                "bucket": lab,
                "n": int(st.get("n") or 0),
                "roi_mean": st.get("mean"),
                "roi_sd": st.get("sd"),
                "roi_se": st.get("se"),
                "roi_ci95": st.get("ci95"),
                "odd_median": _median(odds),
                "exposure_median": _median(exps),
                "exposure_sum": exp_sum,
                "roi_weighted": roi_w,
            }
        )
    return outb


def _slip_regime_from_audit(a: Dict[str, Any], *, exec_created_at: datetime, exec_is_live: bool) -> str:
    """
    Regime Pre/In para slippage×ROI (placar).
    Preferimos inferência por kickoff_time quando disponível; caso contrário, fallback para is_live do executor.
    """
    try:
        ko = a.get("kickoff_time")
        if isinstance(ko, datetime):
            return "In" if exec_created_at >= ko else "Pre"
    except Exception:
        pass
    try:
        return "In" if bool(exec_is_live) else "Pre"
    except Exception:
        return "Pre"


def _agg_pnl_exposure(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Agrega linhas no formato:
      {pnl: float, exposure: float}
    Retorna ROI ponderado por exposição (ROIw) = (∑pnl)/(∑exposure)*100.
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
    rows: List[Dict[str, Any]],
    *,
    slip_raw_pct_max: float = 2.0,
    lat_ms_max: int = 6000,
    slip_missing_pass: bool = True,
    lat_missing_fail_closed: bool = True,
) -> Dict[str, Any]:
    """
    Contrafactual operacional (placar) para Back, aplicado SOMENTE às execuções cobertas por ROI:
    - filtro de slippage (raw com sinal): remove slippage_raw_pct > +slip_raw_pct_max
    - filtro de latência (call_to_done_ms): mantém lat_ms <= lat_ms_max
    Retorna estatísticas base e após filtros (separados e combinados), com deltas.
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
    # contadores de missing para debug
    try:
        out["missing"] = {
            "slip_raw_pct": int(sum(1 for r in rows if r.get("slip_raw_pct") is None)),
            "lat_ms": int(sum(1 for r in rows if r.get("lat_ms") is None)),
        }
    except Exception:
        pass
    return out


def _combo_regime_from_audit(a: Dict[str, Any], *, exec_created_at: datetime) -> str:
    """
    Robustez: se kickoff_time existir, inferimos Pre/In por timestamp (created_at >= kickoff => In).
    Caso contrário, usa a.is_live.
    """
    try:
        ko = a.get("kickoff_time")
        if isinstance(ko, datetime):
            return "In" if exec_created_at >= ko else "Pre"
    except Exception:
        pass
    try:
        return "In" if bool(a.get("is_live") is True) else "Pre"
    except Exception:
        return "Pre"


def _combo_rev_yes_no(a: Dict[str, Any]) -> str:
    """
    Para Lay, usa had_reversal quando disponível; fallback para reversal_direction.
    """
    details = a.get("hypothesis_details")
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except Exception:
            details = None
    try:
        if isinstance(details, dict) and "had_reversal" in details:
            return "Yes" if bool(details.get("had_reversal")) else "No"
    except Exception:
        pass
    try:
        return "Yes" if bool(str(a.get("reversal_direction") or "").strip()) else "No"
    except Exception:
        return "No"


def _slip_cost_pct(*, exec_side: str, odd_dec: Optional[float], odd_fin: Optional[float]) -> Optional[float]:
    """
    Slippage "custo" em % (sempre >=0), normalizado por lado:
    - Back: custo quando a odd caiu (piorou).
    - Lay : custo quando a odd subiu (piorou).
    """
    raw = _slip_raw_pct(odd_dec=odd_dec, odd_fin=odd_fin)
    if raw is None:
        return None
    s = str(exec_side or "").strip().lower()
    if s == "back":
        return float(max(0.0, -raw))
    if s == "lay":
        return float(max(0.0, raw))
    return float(abs(raw))


def _sanitize_decimal_odd(odd: Optional[float]) -> Optional[float]:
    """
    Odds decimais típicas (AH/futebol) raramente passam de ~20.
    Para evitar explosões de ROI por dados corrompidos (ex.: 98 ao invés de 1.98),
    descartamos odds fora de um range plausível e tentamos um rescale simples quando fizer sentido.
    """
    if odd is None:
        return None
    try:
        o = float(odd)
    except Exception:
        return None
    if not (o > 1.0):
        return None
    # heurística: se vierem odds em "centavos" (ex.: 198 => 1.98), reescala
    if o >= 100.0 and o <= 3000.0:
        o2 = o / 100.0
        if o2 > 1.0 and o2 <= 30.0:
            return float(o2)
    # odds absurdas (ex.: 98) são quase sempre bug de parsing/scrape
    if o > 30.0:
        return None
    return float(o)


@dataclass
class ExecRow:
    created_at: datetime
    execution_id: str
    status: str
    exec_side: str
    is_live: bool
    audit_id: Optional[int]
    event_id: Optional[str]
    side: Optional[str]
    line: Optional[str]
    odd_decision: Optional[float]
    odd_final: Optional[float]
    stake_sent: Optional[float]
    # latência (ms) — quando o executor fornece timing no JSONL
    queue_delay_ms: Optional[int] = None
    call_to_done_ms: Optional[int] = None
    post_ms: Optional[int] = None
    betslip_id: Optional[str] = None


def _parse_executor_jsonl(path: Path) -> List[ExecRow]:
    if not path.exists():
        return []
    out: List[ExecRow] = []
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
        eid = str(res.get("execution_id") or req.get("execution_id") or "").strip()
        created = _parse_iso(str(res.get("created_at") or req.get("created_at") or "")) or None
        if not st or not eid or not created:
            continue

        raw = res.get("raw") if isinstance(res.get("raw"), dict) else {}
        betslip_id = None
        try:
            # vários executores colocam o id do slip em raw
            betslip_id = str(raw.get("betslip_id") or "").strip() or None
        except Exception:
            betslip_id = None
        sent = raw.get("sent") if isinstance(raw.get("sent"), dict) else {}
        stake_sent = _safe_float(sent.get("stake"))
        if stake_sent is None:
            # fallback: muitos executores registram stake na policy (request/result) e não em raw.sent
            pol = res.get("policy") if isinstance(res.get("policy"), dict) else (req.get("policy") if isinstance(req.get("policy"), dict) else {})
            stake_sent = _safe_float((pol or {}).get("stake_requested"))

        timing = res.get("timing") if isinstance(res.get("timing"), dict) else {}
        qd = _safe_float(timing.get("queue_delay_ms"))
        cd = _safe_float(timing.get("call_to_done_ms"))
        pm = _safe_float(timing.get("post_ms"))
        queue_delay_ms = int(qd) if qd is not None else None
        call_to_done_ms = int(cd) if cd is not None else None
        post_ms = int(pm) if pm is not None else None
        out.append(
            ExecRow(
                created_at=created,
                execution_id=eid,
                status=st,
                exec_side=str(res.get("exec_side") or req.get("exec_side") or ""),
                is_live=bool(res.get("is_live")) if res.get("is_live") is not None else bool(req.get("is_live")),
                audit_id=(int(res.get("audit_id")) if res.get("audit_id") is not None else (int(req.get("audit_id")) if req.get("audit_id") is not None else None)),
                event_id=str(res.get("event_id") or req.get("event_id") or "") or None,
                side=str(res.get("side") or req.get("side") or "") or None,
                line=str(res.get("line") or req.get("line") or "") or None,
                odd_decision=_safe_float(res.get("odd_at_decision") if res.get("odd_at_decision") is not None else req.get("odd_at_decision")),
                odd_final=_safe_float(res.get("odd_final")),
                stake_sent=stake_sent,
                queue_delay_ms=queue_delay_ms,
                call_to_done_ms=call_to_done_ms,
                post_ms=post_ms,
                betslip_id=betslip_id,
            )
        )
    return out


def _uniq_sorted_str(xs: Iterable[Any]) -> List[str]:
    out = []
    try:
        s = {str(x).strip() for x in xs if str(x).strip()}
        out = sorted(s)
    except Exception:
        out = []
    return out


async def _fetch_bridge_stats(db: Database, *, start_utc: datetime, end_utc: datetime) -> List[Dict[str, Any]]:
    q = text(
        """
        SELECT
          action,
          COUNT(*)::bigint AS n_rows,
          SUM(CASE WHEN execution_id IS NOT NULL THEN 1 ELSE 0 END)::bigint AS n_with_execution_id,
          SUM(CASE WHEN (meta->>'skipped') = 'true' THEN 1 ELSE 0 END)::bigint AS n_skipped,
          SUM(CASE WHEN (meta->>'reason') = 'not_active' THEN 1 ELSE 0 END)::bigint AS n_not_active,
          SUM(CASE WHEN (meta->>'reason') = 'dup_key' THEN 1 ELSE 0 END)::bigint AS n_dup_key,
          SUM(CASE WHEN (meta->>'reason') = 'min_limit' THEN 1 ELSE 0 END)::bigint AS n_min_limit,
          SUM(CASE WHEN (meta->>'accepted') = 'true' THEN 1 ELSE 0 END)::bigint AS n_accepted,
          MIN(created_at) AS first_ts,
          MAX(created_at) AS last_ts
        FROM executor_bridge_seen
        WHERE created_at >= :t0 AND created_at < :t1
        GROUP BY action
        ORDER BY n_rows DESC
        """
    )
    async with db.async_session() as session:
        r = await session.execute(q, {"t0": start_utc, "t1": end_utc})
        rows = r.fetchall() or []
        out = []
        for x in rows:
            out.append(dict(x._mapping))
        return out


async def _fetch_audit_rows_for_ids(db: Database, ids: List[int]) -> Dict[int, Dict[str, Any]]:
    if not ids:
        return {}
    q = text(
        """
        SELECT
          a.id AS audit_id,
          a.event_id,
          a.league,
          a.market_type,
          a.line,
          a.side,
          a.is_live,
          a.hypothesis_type,
          a.reversal_direction,
          a.hypothesis_details,
          a.audited_at,
          m.id AS match_id,
          m.kickoff_time,
          m.home_score,
          m.away_score,
          m.status AS match_status
        FROM betslip_audit_results a
        LEFT JOIN matches m ON m.external_id = a.event_id
        WHERE a.id = ANY(:ids)
        """
    )
    async with db.async_session() as session:
        r = await session.execute(q, {"ids": ids})
        rows = r.fetchall() or []
        out: Dict[int, Dict[str, Any]] = {}
        for x in rows:
            d = dict(x._mapping)
            try:
                if isinstance(d.get("hypothesis_details"), str):
                    d["hypothesis_details"] = json.loads(d["hypothesis_details"])
            except Exception:
                pass
            out[int(d["audit_id"])] = d
        return out


def _active_keys_by_day_from_policy(policy: Dict[str, Any], *, tz_name: str) -> Dict[str, Any]:
    """
    Cria mapa day(YYYY-MM-DD em tz_name) -> active_keys do step cujo test window cobre o dia.
    """
    steps = policy.get("steps") if isinstance(policy.get("steps"), list) else []
    out: Dict[str, Any] = {"tz": tz_name, "days": {}}
    if not steps:
        return out

    tz = timezone.utc
    try:
        from zoneinfo import ZoneInfo  # type: ignore

        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc

    for st in steps:
        if not isinstance(st, dict):
            continue
        active = st.get("active_keys") if isinstance(st.get("active_keys"), list) else None
        # Formato atual do export do WF (b808): train_days/test_days como lista de YYYY-MM-DD (UTC).
        test_days = st.get("test_days") if isinstance(st.get("test_days"), list) else None
        if test_days:
            for ds in [str(x) for x in test_days if str(x)]:
                out["days"][ds] = {
                    "active_keys": active or [],
                    "n_active_keys": len(active or []),
                    "train": st.get("train"),
                    "test": st.get("test"),
                    "train_days": st.get("train_days"),
                    "test_days": test_days,
                }
            continue

        # Fallback: formato antigo com test_window.start/end (ISO)
        test = st.get("test_window") if isinstance(st.get("test_window"), dict) else {}
        t0 = _parse_iso(str(test.get("start") or "")) if isinstance(test, dict) else None
        t1 = _parse_iso(str(test.get("end") or "")) if isinstance(test, dict) else None
        if not t0 or not t1:
            continue
        d0 = t0.astimezone(tz).date()
        d1 = t1.astimezone(tz).date()
        for d in _iter_dates(d0, d1):
            out["days"][d.isoformat()] = {
                "active_keys": active or [],
                "n_active_keys": len(active or []),
                "test_window": {"start": t0.isoformat(), "end": t1.isoformat()},
                "train_window": st.get("train_window"),
            }
    return out


async def run_report(
    *,
    policy_json: Path,
    executor_jsonl: Path,
    tz_name: str,
    days: int,
    include_today: bool,
    no_per_day: bool = False,
    slip_cf_start_day: Optional[str] = None,
    out_json: Optional[Path] = None,
) -> Dict[str, Any]:
    now_utc = datetime.now(timezone.utc)
    # intervalo de dias (por timezone do report)
    tz = timezone.utc
    try:
        from zoneinfo import ZoneInfo  # type: ignore

        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc
    now_local = now_utc.astimezone(tz)
    end_day = now_local.date() if include_today else (now_local.date() - timedelta(days=1))

    # Corte opcional (pós-fix) para o contrafactual de slippage (placar).
    # Interpreta `slip_cf_start_day` como dia LOCAL em `tz_name` (YYYY-MM-DD).
    #
    # Importante: apesar do nome histórico (cf = contrafactual), o corte é usado para
    # TODAS as tabelas de slippage×ROI (raw/cost, totals, ctx e por combo) quando definido,
    # para evitar que “pré-fix” contamine as estatísticas após correções de cálculo.
    cf_start_day: Optional[date] = None
    if slip_cf_start_day:
        try:
            cf_start_day = date.fromisoformat(str(slip_cf_start_day).strip())
        except Exception:
            cf_start_day = None

    def _cf_allowed(e: Any) -> bool:
        if cf_start_day is None:
            return True
        try:
            return e.created_at.astimezone(tz).date() >= cf_start_day  # type: ignore[attr-defined]
        except Exception:
            return True

    # Alias semântico: o mesmo corte vale para slippage×ROI (não só contrafactual).
    def _slip_allowed_day(d: date) -> bool:
        if cf_start_day is None:
            return True
        try:
            return d >= cf_start_day
        except Exception:
            return True

    policy = _load_json(policy_json) or {}
    active_by_day = _active_keys_by_day_from_policy(policy, tz_name=tz_name)
    steps = policy.get("steps") if isinstance(policy.get("steps"), list) else []
    last_step = steps[-1] if steps and isinstance(steps[-1], dict) else None

    exec_rows = _parse_executor_jsonl(executor_jsonl)

    # days<=0 => período completo (desde o primeiro evento presente no jsonl)
    if int(days) <= 0:
        if exec_rows:
            try:
                start_day = min(e.created_at.astimezone(tz).date() for e in exec_rows)
            except Exception:
                start_day = end_day
        else:
            start_day = end_day
    else:
        start_day = end_day - timedelta(days=max(0, int(days) - 1))

    # Range efetivo para análises de slippage×ROI quando há corte pós-fix.
    # Mantemos `range` como janela base (usada para iterar per_day), mas expomos um
    # range semântico para slippage (que pode ser diferente quando `slip_cf_start_day` é definido).
    slip_start_day = start_day
    if cf_start_day is not None:
        try:
            slip_start_day = max(start_day, cf_start_day)
        except Exception:
            slip_start_day = start_day
    try:
        base_span_days = int((end_day - start_day).days) + 1
    except Exception:
        base_span_days = None
    try:
        slip_span_days = int((end_day - slip_start_day).days) + 1
    except Exception:
        slip_span_days = None
    db = Database()
    await db.connect()

    per_day = []
    # acumulado na janela (para análise estatística)
    total_pairs_raw_back: List[Tuple[float, float]] = []
    total_pairs_raw_lay: List[Tuple[float, float]] = []
    total_pairs_cost_back: List[Tuple[float, float]] = []
    total_pairs_cost_lay: List[Tuple[float, float]] = []
    # acumulado por combinação (top N por volume)
    comb_pairs_raw: Dict[str, List[Tuple[float, float]]] = {}
    comb_meta: Dict[str, Dict[str, Any]] = {}
    # contexto para interpretação (odd/exposure)
    total_rows_ctx_back: List[Dict[str, Any]] = []
    total_rows_ctx_back_pre: List[Dict[str, Any]] = []
    total_rows_ctx_back_in: List[Dict[str, Any]] = []
    total_rows_ctx_lay: List[Dict[str, Any]] = []
    total_rows_ctx_lay_stake: List[Dict[str, Any]] = []
    # contrafactual: filtro de slippage (apenas execuções com ROI via placar)
    # Observação operacional: por padrão aplicamos o gate do Lay apenas no PRE (evita interferir em Lay in‑match, ex.: Lay_In_Yes),
    # mas isso é configurável por env.
    slip_gate_back_scope = str(os.getenv("SLIPPAGE_GATE_BACK_SCOPE", "all") or "all").strip().lower()  # all|pre|in
    slip_gate_lay_scope = str(os.getenv("SLIPPAGE_GATE_LAY_SCOPE", "pre") or "pre").strip().lower()  # all|pre|in
    if slip_gate_back_scope not in ("all", "pre", "in"):
        slip_gate_back_scope = "all"
    if slip_gate_lay_scope not in ("all", "pre", "in"):
        slip_gate_lay_scope = "pre"

    cf = {
        "rule": {
            "back_skip_raw_pct_le": -2.0,
            "lay_skip_raw_pct_gt": 2.0,
            "back_scope": slip_gate_back_scope,
            "lay_scope": slip_gate_lay_scope,
            "start_day_local": cf_start_day.isoformat() if cf_start_day else None,
        },
        "note": "Contrafactual baseado apenas nas execuções com ROI via placar (audit+scores+odd). Não é P&L accounting.",
        "back": {"n": 0, "pnl": 0.0, "stake": 0.0, "n_filtered": 0, "pnl_filtered": 0.0, "stake_filtered": 0.0},
        # Lay: além de liability, guardamos stake para permitir ajuste de turnover (capacidade) em tabelas de sensibilidade
        "lay": {
            "n": 0,
            "pnl": 0.0,
            "stake": 0.0,
            "liability": 0.0,
            "n_filtered": 0,
            "pnl_filtered": 0.0,
            "stake_filtered": 0.0,
            "liability_filtered": 0.0,
        },
    }

    def _cf_init() -> Dict[str, Any]:
        return {
            "rule": dict(cf.get("rule") or {}),
            "note": str(cf.get("note") or ""),
            "back": {"n": 0, "pnl": 0.0, "stake": 0.0, "n_filtered": 0, "pnl_filtered": 0.0, "stake_filtered": 0.0},
            "lay": {
                "n": 0,
                "pnl": 0.0,
                "stake": 0.0,
                "liability": 0.0,
                "n_filtered": 0,
                "pnl_filtered": 0.0,
                "stake_filtered": 0.0,
                "liability_filtered": 0.0,
            },
        }

    # Diagnóstico AH (linha): o filtro do WF é por |line|, não por odds.
    wf_cfg = policy.get("wf") if isinstance(policy.get("wf"), dict) else {}
    ah_thr = None
    try:
        ah_thr = float(wf_cfg.get("ah_max_abs_line")) if wf_cfg.get("ah_max_abs_line") is not None else None
    except Exception:
        ah_thr = None
    ah_scope = str(wf_cfg.get("ah_scope") or "pre").strip().lower()  # pre|all|in
    ah_obs = {
        "threshold": ah_thr,
        "scope": ah_scope,
        "all_exec": {"n": 0, "n_over": 0, "max_abs_line": None},
        "cov_placar": {"n": 0, "n_over": 0, "max_abs_line": None},
    }

    def _ah_apply_for_regime(regime: str) -> bool:
        r = str(regime).strip().lower()
        if ah_scope == "all":
            return True
        if ah_scope == "in":
            return r == "in"
        return r == "pre"

    def _ah_track(bucket: str, *, line: Optional[str], regime: str) -> None:
        if not line:
            return
        try:
            x = abs(float(str(line).strip()))
        except Exception:
            return
        blk = ah_obs.get(bucket) if isinstance(ah_obs.get(bucket), dict) else None
        if not blk:
            return
        blk["n"] = int(blk.get("n") or 0) + 1
        try:
            mx = blk.get("max_abs_line")
            blk["max_abs_line"] = float(x) if mx is None else max(float(mx), float(x))
        except Exception:
            pass
        if ah_thr is not None and _ah_apply_for_regime(regime) and float(x) > float(ah_thr):
            blk["n_over"] = int(blk.get("n_over") or 0) + 1
    # Modo "totais apenas": evita loop por dia e reduz queries.
    if bool(no_per_day):
        start_utc0, _ = _local_day_bounds_utc(day=start_day, tz_name=tz_name)
        _, end_utc0 = _local_day_bounds_utc(day=end_day, tz_name=tz_name)
        xs = [e for e in exec_rows if start_utc0 <= e.created_at < end_utc0]
        audit_ids = sorted({int(e.audit_id) for e in xs if e.audit_id is not None})
        audit_map = await _fetch_audit_rows_for_ids(db, audit_ids)

        # Latência × ROI (Back Pre/In): acumulado na janela (somente execuções com ROI via placar)
        lat_rows_back_pre: List[Dict[str, Any]] = []
        lat_rows_back_in: List[Dict[str, Any]] = []
        # Slippage × Latência (Back Pre/In): acumulado (somente execuções com ROI via placar)
        slip_lat_rows_back_pre: List[Dict[str, Any]] = []
        slip_lat_rows_back_in: List[Dict[str, Any]] = []

        for e in xs:
            # diagnóstico AH em todas as execuções (com ou sem placar)
            try:
                _ah_track("all_exec", line=e.line, regime=("In" if bool(e.is_live) else "Pre"))
            except Exception:
                pass
            st_norm = str(e.status or "").strip().upper()
            is_success = st_norm in ("DRY_OK", "LIVE_OK")
            a = audit_map.get(int(e.audit_id)) if e.audit_id is not None else None
            if not a:
                continue
            # P&L/ROI por placar só faz sentido para execuções bem‑sucedidas (LIVE_OK/DRY_OK).
            # Caso contrário, vira "what-if" e tende a ficar sistematicamente acima/abaixo do accounting.
            if not is_success:
                continue
            try:
                _ah_track("cov_placar", line=str(a.get("line") or e.line or ""), regime=("In" if bool(e.is_live) else "Pre"))
            except Exception:
                pass
            mult = _mult_back_from_scores(a.get("line") or e.line, a.get("side") or (e.side or ""), a.get("home_score"), a.get("away_score"))
            if mult is None:
                continue
            # ROI/placar deve refletir a odd executada. Sem odd_final, não há como medir
            # slippage nem ROI realizado com consistência, então tratamos como não-coberto.
            odd = _sanitize_decimal_odd(e.odd_final)
            if odd is None:
                continue
            side = str(e.exec_side or "").strip().lower()
            raw_pct = _slip_raw_pct(odd_dec=e.odd_decision, odd_fin=e.odd_final)
            cost_pct = _slip_cost_pct(exec_side=side, odd_dec=e.odd_decision, odd_fin=e.odd_final)

            if side == "back":
                stake = float(e.stake_sent) if e.stake_sent is not None else 1.0
                roi = _roi_back_pct(float(odd), float(mult))
                pnl = stake * roi / 100.0
                # Latência × ROI (Back Pre/In)
                try:
                    regime = _combo_regime_from_audit(a, exec_created_at=e.created_at)
                except Exception:
                    regime = ("In" if bool(e.is_live) else "Pre")
                lat_row = {
                    "lat_ms": (float(e.call_to_done_ms) if e.call_to_done_ms is not None else None),
                    "roi": float(roi),
                    "odd": float(odd),
                    "exposure": float(stake),
                }
                # Respeita corte pós-fix também no modo "totais apenas"
                if _cf_allowed(e):
                    if str(regime).strip().lower() == "in":
                        lat_rows_back_in.append(lat_row)
                    else:
                        lat_rows_back_pre.append(lat_row)
                    if cost_pct is not None:
                        total_pairs_cost_back.append((float(cost_pct), float(roi)))
                    if raw_pct is not None:
                        total_pairs_raw_back.append((float(raw_pct), float(roi)))
                        total_rows_ctx_back.append({"slip_raw_pct": float(raw_pct), "roi": float(roi), "odd": float(odd), "exposure": float(stake)})
                        try:
                            slip_reg = _slip_regime_from_audit(a, exec_created_at=e.created_at, exec_is_live=bool(e.is_live))
                            if str(slip_reg).strip().lower() == "in":
                                total_rows_ctx_back_in.append({"slip_raw_pct": float(raw_pct), "roi": float(roi), "odd": float(odd), "exposure": float(stake)})
                            else:
                                total_rows_ctx_back_pre.append({"slip_raw_pct": float(raw_pct), "roi": float(roi), "odd": float(odd), "exposure": float(stake)})
                        except Exception:
                            pass
                        try:
                            slip_lat_row = {
                                "lat_ms": (float(e.call_to_done_ms) if e.call_to_done_ms is not None else None),
                                "slip_raw_pct": float(raw_pct),
                                "roi": float(roi),
                                "odd": float(odd),
                                "exposure": float(stake),
                            }
                            if str(regime).strip().lower() == "in":
                                slip_lat_rows_back_in.append(slip_lat_row)
                            else:
                                slip_lat_rows_back_pre.append(slip_lat_row)
                        except Exception:
                            pass
                    try:
                        regime = _combo_regime_from_audit(a, exec_created_at=e.created_at)
                        comb = f"Back_{regime}_Any"
                        comb_pairs_raw.setdefault(comb, []).append((float(raw_pct), float(roi)))
                        comb_meta.setdefault(comb, {"side": "Back", "regime": regime, "reversal": "Any", "league": None})
                    except Exception:
                        pass
                try:
                    regime = _combo_regime_from_audit(a, exec_created_at=e.created_at)
                except Exception:
                    regime = ("In" if bool(e.is_live) else "Pre")
                # contrafactual (acumulado): aplica corte pós-fix se configurado
                if _cf_allowed(e):
                    cf["back"]["n"] += 1
                    cf["back"]["pnl"] += float(pnl)
                    cf["back"]["stake"] += float(stake)
                    scope = str((cf.get("rule") or {}).get("back_scope") or "all").strip().lower()
                    apply_gate = True
                    if scope in ("pre", "in"):
                        apply_gate = (str(regime).strip().lower() == ("pre" if scope == "pre" else "in"))
                    skip = bool(apply_gate and raw_pct is not None and float(raw_pct) <= float(cf["rule"]["back_skip_raw_pct_le"]))
                    if not skip:
                        cf["back"]["n_filtered"] += 1
                        cf["back"]["pnl_filtered"] += float(pnl)
                        cf["back"]["stake_filtered"] += float(stake)
            elif side == "lay":
                stake = float(e.stake_sent) if e.stake_sent is not None else 1.0
                liab = stake * max(0.0, float(odd) - 1.0)
                roi_liab = _roi_lay_pct_per_liability(float(odd), float(mult))
                if roi_liab is None:
                    continue
                pnl = liab * float(roi_liab) / 100.0
                # Respeita corte pós-fix também no modo "totais apenas"
                if _cf_allowed(e):
                    if cost_pct is not None:
                        total_pairs_cost_lay.append((float(cost_pct), float(roi_liab)))
                    if raw_pct is not None:
                        total_pairs_raw_lay.append((float(raw_pct), float(roi_liab)))
                        total_rows_ctx_lay.append({"slip_raw_pct": float(raw_pct), "roi": float(roi_liab), "odd": float(odd), "exposure": float(liab)})
                    try:
                        roi_st = (float(pnl) / float(stake) * 100.0) if stake > 0 else None
                    except Exception:
                        roi_st = None
                    if roi_st is not None:
                        total_rows_ctx_lay_stake.append({"slip_raw_pct": float(raw_pct), "roi": float(roi_st), "odd": float(odd), "exposure": float(stake)})
                    try:
                        regime = _combo_regime_from_audit(a, exec_created_at=e.created_at)
                        rev = _combo_rev_yes_no(a)
                        comb = f"Lay_{regime}_{rev}"
                        comb_pairs_raw.setdefault(comb, []).append((float(raw_pct), float(roi_liab)))
                        comb_meta.setdefault(comb, {"side": "Lay", "regime": regime, "reversal": rev, "league": None})
                    except Exception:
                        pass
                try:
                    regime = _combo_regime_from_audit(a, exec_created_at=e.created_at)
                except Exception:
                    regime = ("In" if bool(e.is_live) else "Pre")
                # contrafactual (acumulado): aplica corte pós-fix se configurado
                if _cf_allowed(e):
                    cf["lay"]["n"] += 1
                    cf["lay"]["pnl"] += float(pnl)
                    cf["lay"]["stake"] += float(stake)
                    cf["lay"]["liability"] += float(liab)
                    scope = str((cf.get("rule") or {}).get("lay_scope") or "pre").strip().lower()
                    apply_gate = True
                    if scope in ("pre", "in"):
                        apply_gate = (str(regime).strip().lower() == ("pre" if scope == "pre" else "in"))
                    skip = bool(apply_gate and raw_pct is not None and float(raw_pct) > float(cf["rule"]["lay_skip_raw_pct_gt"]))
                    if not skip:
                        cf["lay"]["n_filtered"] += 1
                        cf["lay"]["pnl_filtered"] += float(pnl)
                        cf["lay"]["stake_filtered"] += float(stake)
                        cf["lay"]["liability_filtered"] += float(liab)

        # carry-forward do último step se existir (mesma lógica do modo per_day)
        if last_step and isinstance(last_step, dict):
            # sem per_day, não há o que preencher aqui
            pass

        # build output (per_day vazio)
        out = {
            "ts_utc": now_utc.isoformat(),
            "tz": tz_name,
            "policy_json": str(policy_json),
            "executor_jsonl": str(executor_jsonl),
            "range": {"start_day": start_day.isoformat(), "end_day": end_day.isoformat(), "days": int(days), "include_today": bool(include_today)},
            "slippage_range": {
                "start_day": slip_start_day.isoformat(),
                "end_day": end_day.isoformat(),
                "span_days": slip_span_days,
                "cut_start_day_local": cf_start_day.isoformat() if cf_start_day else None,
            },
            "policy_days": active_by_day,
            "per_day": [],
            "observed_ah_line_abs": ah_obs,
            "slippage_filter_counterfactual": cf,
            "slippage_start_day_local": cf_start_day.isoformat() if cf_start_day else None,
            "latency_vs_roi_call_to_done_ms": {
                "note": "Latência por execução vem de result.timing.call_to_done_ms no executor_jsonl. ROI/placar usa somente odd executada (odd_final). Se odd_final estiver ausente, a execução não entra no subconjunto coberto.",
                "back_pre": {"buckets": _bucketize_latency_call_to_done_ms_with_context(lat_rows_back_pre)},
                "back_in": {"buckets": _bucketize_latency_call_to_done_ms_with_context(lat_rows_back_in)},
            },
            "slippage_vs_latency_call_to_done_ms": {
                "note": "Slippage_raw_pct vs latência usa execuções com ROI via placar e odd_final presente; slippage_raw_pct=(odd_final-odd_at_decision)/odd_at_decision.",
                "back_pre": {"buckets": _bucketize_slip_raw_pct_vs_latency_with_context(slip_lat_rows_back_pre)},
                "back_in": {"buckets": _bucketize_slip_raw_pct_vs_latency_with_context(slip_lat_rows_back_in)},
            },
            "slippage_vs_roi_raw_total": {
                "back": {"buckets": _bucketize_3way_raw(total_pairs_raw_back)},
                "lay": {"buckets": _bucketize_3way_raw(total_pairs_raw_lay)},
            },
            "slippage_vs_roi_raw_total_ctx": {
                "back": {"buckets": _bucketize_3way_raw_with_context(total_rows_ctx_back)},
                "lay": {"buckets": _bucketize_3way_raw_with_context(total_rows_ctx_lay)},
            },
            "slippage_vs_roi_raw_total_ctx_by_regime": {
                "back_pre": {"buckets": _bucketize_3way_raw_with_context(total_rows_ctx_back_pre)},
                "back_in": {"buckets": _bucketize_3way_raw_with_context(total_rows_ctx_back_in)},
            },
            "slippage_vs_roi_raw_total_ctx_lay_stake": {
                "lay": {"buckets": _bucketize_3way_raw_with_context(total_rows_ctx_lay_stake)},
            },
            "slippage_vs_roi_total": {
                "back": {
                    "n": int(len(total_pairs_cost_back)),
                    "corr_cost_pct_vs_roi": _pearson([c for c, _ in total_pairs_cost_back], [r for _, r in total_pairs_cost_back]) if total_pairs_cost_back else None,
                },
                "lay": {
                    "n": int(len(total_pairs_cost_lay)),
                    "corr_cost_pct_vs_roi": _pearson([c for c, _ in total_pairs_cost_lay], [r for _, r in total_pairs_cost_lay]) if total_pairs_cost_lay else None,
                },
            },
        }
        # combos (já sem liga)
        try:
            rows = []
            for comb, pairs in (comb_pairs_raw or {}).items():
                if not pairs:
                    continue
                meta = comb_meta.get(comb) or {}
                buckets = _bucketize_3way_raw(pairs)
                corr = _pearson([s for s, _ in pairs], [r for _, r in pairs]) if len(pairs) >= 5 else None
                rows.append(
                    {
                        "comb": comb,
                        "n": int(len(pairs)),
                        "side": meta.get("side"),
                        "regime": meta.get("regime"),
                        "reversal": meta.get("reversal"),
                        "league": None,
                        "corr_raw_pct_vs_roi": corr,
                        "buckets": buckets,
                    }
                )
            rows.sort(key=lambda r: int(r.get("n") or 0), reverse=True)
            out["slippage_vs_roi_raw_by_combo_top"] = rows[:40]
        except Exception:
            pass
        out = _json_safe(out)
        if out_json:
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    for d in _iter_dates(start_day, end_day):
        start_utc, end_utc = _local_day_bounds_utc(day=d, tz_name=tz_name)
        cf_day = _cf_init()

        # bridge adherence
        bridge_stats = await _fetch_bridge_stats(db, start_utc=start_utc, end_utc=end_utc)

        # executions in that window
        xs = [e for e in exec_rows if start_utc <= e.created_at < end_utc]
        # diagnóstico AH (all exec)
        for e in xs:
            try:
                _ah_track("all_exec", line=e.line, regime=("In" if bool(e.is_live) else "Pre"))
            except Exception:
                continue
        # fetch audits for ROI
        audit_ids = sorted({int(e.audit_id) for e in xs if e.audit_id is not None})
        audit_map = await _fetch_audit_rows_for_ids(db, audit_ids)

        perf = {
            "n_exec_rows": len(xs),
            "n_exec_success": 0,
            "status_counts": {},
            # P&L por placar (somente cobertura com ROI) quebrado por Back/Lay × Pre/In.
            # Back: exposição=stake; Lay: exposição=liability.
            "pnl_placar_by_type": {
                "Back_Pre": {"n": 0, "exposure": 0.0, "pnl": 0.0},
                "Back_In": {"n": 0, "exposure": 0.0, "pnl": 0.0},
                "Lay_Pre": {"n": 0, "exposure": 0.0, "pnl": 0.0},
                "Lay_In": {"n": 0, "exposure": 0.0, "pnl": 0.0},
            },
            "back": {
                "n_success": 0,
                "n_cov": 0,
                "wins": 0,
                "losses": 0,
                "push": 0,
                "half_wins": 0,
                "half_losses": 0,
                # stake enviado total (sucessos)
                "stake_sum": 0.0,
                # stake enviado por regime (Pre/In) — útil para alocar P&L do accounting por regime (proxy)
                "stake_sum_pre": 0.0,
                "stake_sum_in": 0.0,
                # stake com cobertura (placar+odd)
                "stake_sum_cov": 0.0,
                "pnl_sum": 0.0,
                "roi_pct": None,
                # cobertura (placar) entre sucessos: % por n e por stake
                "cov_pct_n": None,
                "cov_pct_stake": None,
                "n_uncov": 0,
                "stake_sum_uncov": 0.0,
                # cobertura por jogo (event_id) — útil para conciliar com accounting
                "events_success_n": 0,
                "events_cov_n": 0,
                "events_cov_pct": None,
                "event_ids_success": [],
                "event_ids_cov": [],
            },
            "lay": {
                "n_success": 0,
                "n_cov": 0,
                "wins": 0,
                "losses": 0,
                "push": 0,
                "half_wins": 0,
                "half_losses": 0,
                # stake enviado (unidade natural do executor)
                "stake_sum": 0.0,
                # exposição de risco (liability = stake*(odd-1))
                "liability_sum": 0.0,
                "stake_sum_cov": 0.0,
                "liability_sum_cov": 0.0,
                "pnl_sum": 0.0,
                "roi_pct_per_liability": None,
                # cobertura (placar) entre sucessos
                "cov_pct_n": None,
                "cov_pct_stake": None,
                "n_uncov": 0,
                "stake_sum_uncov": 0.0,
                "liability_sum_uncov": 0.0,
                # cobertura por jogo (event_id)
                "events_success_n": 0,
                "events_cov_n": 0,
                "events_cov_pct": None,
                "event_ids_success": [],
                "event_ids_cov": [],
            },
            "odd_anomalies": {"back": {"n": 0, "max": None}, "lay": {"n": 0, "max": None}},
            "slippage": {
                "back": {"n": 0, "raw_pct_mean": None, "cost_pct_mean": None},
                "lay": {"n": 0, "raw_pct_mean": None, "cost_pct_mean": None},
            },
            "slippage_vs_roi": {
                "back": {"corr_cost_pct_vs_roi": None, "buckets": []},
                "lay": {"corr_cost_pct_vs_roi": None, "buckets": []},
            },
            # ROI por stake (Lay) é limitado e evita explosões quando odd≈1.
            "lay_roi_pct_per_stake": None,
        }

        # sets por jogo (event_id), para conciliação com accounting
        ev_success_back: set[str] = set()
        ev_cov_back: set[str] = set()
        ev_success_lay: set[str] = set()
        ev_cov_lay: set[str] = set()

        slip_raw_back: List[float] = []
        slip_cost_back: List[float] = []
        slip_raw_lay: List[float] = []
        slip_cost_lay: List[float] = []
        pairs_back: List[Tuple[float, float]] = []
        pairs_lay: List[Tuple[float, float]] = []
        pairs_raw_back: List[Tuple[float, float]] = []
        pairs_raw_lay: List[Tuple[float, float]] = []

        # Linhas cobertas (placar) para contrafactuais operacionais (Back)
        cov_back_rows: List[Dict[str, Any]] = []
        cov_back_rows_pre: List[Dict[str, Any]] = []
        cov_back_rows_in: List[Dict[str, Any]] = []

        def _bucketize(pairs: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
            edges = [(0.0, 0.0), (0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 999.0)]
            outb = []
            for a, b in edges:
                if a == b:
                    ys = [roi for (c, roi) in pairs if c == 0.0]
                    lab = "0"
                else:
                    ys = [roi for (c, roi) in pairs if (c > a and c <= b)]
                    lab = f"({a},{b}]"
                if not ys:
                    continue
                outb.append({"bucket": lab, "n": int(len(ys)), "roi_mean": float(sum(ys) / len(ys))})
            return outb
        for e in xs:
            perf["status_counts"][e.status] = int(perf["status_counts"].get(e.status, 0)) + 1

            st_norm = str(e.status or "").strip().upper()
            is_success = st_norm in ("DRY_OK", "LIVE_OK")
            side0 = str(e.exec_side or "").strip().lower()
            if is_success:
                perf["n_exec_success"] = int(perf.get("n_exec_success") or 0) + 1
                stake0 = float(e.stake_sent) if e.stake_sent is not None else 1.0
                if side0 == "back":
                    perf["back"]["n_success"] += 1
                    perf["back"]["stake_sum"] += float(stake0)
                    # split Pre/In (best-effort): usa kickoff_time do audit quando disponível
                    try:
                        reg = None
                        a0 = audit_map.get(int(e.audit_id)) if e.audit_id is not None else None
                        if isinstance(a0, dict) and a0:
                            reg = _slip_regime_from_audit(a0, exec_created_at=e.created_at, exec_is_live=bool(e.is_live))
                        if reg is None:
                            reg = ("In" if bool(e.is_live) else "Pre")
                        if str(reg).strip().lower() == "in":
                            perf["back"]["stake_sum_in"] += float(stake0)
                        else:
                            perf["back"]["stake_sum_pre"] += float(stake0)
                    except Exception:
                        # não falha o relatório por ausência de audit/kickoff
                        pass
                    try:
                        if e.event_id:
                            ev_success_back.add(str(e.event_id).strip())
                    except Exception:
                        pass
                elif side0 == "lay":
                    perf["lay"]["n_success"] += 1
                    perf["lay"]["stake_sum"] += float(stake0)
                    try:
                        if e.event_id:
                            ev_success_lay.add(str(e.event_id).strip())
                    except Exception:
                        pass
                    # liability (exposição) é medida na odd executada; sem odd_final não estimamos.
                    odd0 = _sanitize_decimal_odd(e.odd_final)
                    if odd0 is not None:
                        perf["lay"]["liability_sum"] += float(stake0) * max(0.0, float(odd0) - 1.0)

            # só calcula "resultado" quando há placar e odd
            if not is_success:
                continue
            a = audit_map.get(int(e.audit_id)) if e.audit_id is not None else None
            if not a:
                continue
            try:
                _ah_track("cov_placar", line=str(a.get("line") or e.line or ""), regime=("In" if bool(e.is_live) else "Pre"))
            except Exception:
                pass
            mult = _mult_back_from_scores(a.get("line") or e.line, a.get("side") or (e.side or ""), a.get("home_score"), a.get("away_score"))
            if mult is None:
                continue
            # ROI/placar deve usar odd executada; sem odd_final consideramos não-coberto.
            odd = _sanitize_decimal_odd(e.odd_final)
            if odd is None:
                # registra anomalia por lado (quando havia algo preenchido)
                raw_odd = e.odd_final
                if raw_odd is not None and side0 in ("back", "lay"):
                    blk = perf.get("odd_anomalies", {}).get(side0, {})
                    try:
                        blk["n"] = int(blk.get("n") or 0) + 1
                        mx = blk.get("max")
                        v = float(raw_odd)
                        blk["max"] = v if mx is None else max(float(mx), v)
                    except Exception:
                        pass
                continue

            side = str(e.exec_side or "").strip().lower()
            if side == "back":
                perf["back"]["n_cov"] += 1
                stake = float(e.stake_sent) if e.stake_sent is not None else 1.0
                roi = _roi_back_pct(float(odd), float(mult))
                pnl = stake * roi / 100.0
                # Regime Pre/In: usado tanto para split de P&L quanto para contrafactuais por regime.
                try:
                    reg = _combo_regime_from_audit(a, exec_created_at=e.created_at)
                except Exception:
                    reg = ("In" if bool(e.is_live) else "Pre")
                # Para contrafactual operacional: precisa ROI (placar), exposição e metadados de execução
                try:
                    lat_ms = int(e.call_to_done_ms) if e.call_to_done_ms is not None else None
                except Exception:
                    lat_ms = None
                raw_pct = _slip_raw_pct(odd_dec=e.odd_decision, odd_fin=e.odd_final)
                try:
                    cov_row = {
                        "roi": float(roi),
                        "pnl": float(pnl),
                        "exposure": float(stake),
                        "slip_raw_pct": (float(raw_pct) if raw_pct is not None else None),
                        "lat_ms": (float(lat_ms) if lat_ms is not None else None),
                    }
                    cov_back_rows.append(cov_row)
                    if str(reg).strip().lower() == "in":
                        cov_back_rows_in.append(cov_row)
                    else:
                        cov_back_rows_pre.append(cov_row)
                except Exception:
                    pass
                try:
                    ev_id = str(a.get("event_id") or e.event_id or "").strip()
                    if ev_id:
                        ev_cov_back.add(ev_id)
                except Exception:
                    pass
                # quebra por Pre/In (para conciliar com OOS e com contagens por jogo)
                try:
                    k2 = f"Back_{reg}"
                    blk = perf.get("pnl_placar_by_type", {}).get(k2) if isinstance(perf.get("pnl_placar_by_type"), dict) else None
                    if isinstance(blk, dict):
                        blk["n"] = int(blk.get("n") or 0) + 1
                        blk["exposure"] = float(blk.get("exposure") or 0.0) + float(stake)
                        blk["pnl"] = float(blk.get("pnl") or 0.0) + float(pnl)
                except Exception:
                    pass
                perf["back"]["stake_sum_cov"] += float(stake)
                perf["back"]["pnl_sum"] += float(pnl)
                if mult > 0:
                    if mult == 0.5:
                        perf["back"]["half_wins"] += 1
                    else:
                        perf["back"]["wins"] += 1
                elif mult < 0:
                    if mult == -0.5:
                        perf["back"]["half_losses"] += 1
                    else:
                        perf["back"]["losses"] += 1
                else:
                    perf["back"]["push"] += 1

                # raw_pct já calculado acima (reusa)
                cost_pct = _slip_cost_pct(exec_side="back", odd_dec=e.odd_decision, odd_fin=e.odd_final)
                if raw_pct is not None and cost_pct is not None:
                    slip_raw_back.append(float(raw_pct))
                    slip_cost_back.append(float(cost_pct))
                if cost_pct is not None:
                    pairs_back.append((float(cost_pct), float(roi)))
                if raw_pct is not None:
                    pairs_raw_back.append((float(raw_pct), float(roi)))
                    total_rows_ctx_back.append({"slip_raw_pct": float(raw_pct), "roi": float(roi), "odd": float(odd), "exposure": float(stake)})
                    try:
                        slip_reg = _slip_regime_from_audit(a, exec_created_at=e.created_at, exec_is_live=bool(e.is_live))
                        if str(slip_reg).strip().lower() == "in":
                            total_rows_ctx_back_in.append({"slip_raw_pct": float(raw_pct), "roi": float(roi), "odd": float(odd), "exposure": float(stake)})
                        else:
                            total_rows_ctx_back_pre.append({"slip_raw_pct": float(raw_pct), "roi": float(roi), "odd": float(odd), "exposure": float(stake)})
                    except Exception:
                        pass
                    # por combinação (sem quebra por liga): Back/Lay × Pre/In × Yes/No/Any
                    try:
                        regime = _combo_regime_from_audit(a, exec_created_at=e.created_at)
                        comb = f"Back_{regime}_Any"
                        comb_pairs_raw.setdefault(comb, []).append((float(raw_pct), float(roi)))
                        comb_meta.setdefault(comb, {"side": "Back", "regime": regime, "reversal": "Any", "league": None})
                    except Exception:
                        pass
                # contrafactual slippage filter (Back): pula raw<=-2%
                try:
                    if _cf_allowed(e):
                        cf["back"]["n"] += 1
                        cf["back"]["pnl"] += float(pnl)
                        cf["back"]["stake"] += float(stake)
                        scope = str((cf.get("rule") or {}).get("back_scope") or "all").strip().lower()
                        apply_gate = True
                        if scope in ("pre", "in"):
                            apply_gate = (str(reg).strip().lower() == ("pre" if scope == "pre" else "in"))
                        skip = bool(apply_gate and raw_pct is not None and float(raw_pct) <= float(cf["rule"]["back_skip_raw_pct_le"]))
                        if not skip:
                            cf["back"]["n_filtered"] += 1
                            cf["back"]["pnl_filtered"] += float(pnl)
                            cf["back"]["stake_filtered"] += float(stake)
                except Exception:
                    pass
                try:
                    if _cf_allowed(e):
                        cf_day["back"]["n"] += 1
                        cf_day["back"]["pnl"] += float(pnl)
                        cf_day["back"]["stake"] += float(stake)
                        scope = str((cf_day.get("rule") or {}).get("back_scope") or "all").strip().lower()
                        apply_gate = True
                        if scope in ("pre", "in"):
                            apply_gate = (str(reg).strip().lower() == ("pre" if scope == "pre" else "in"))
                        skip = bool(apply_gate and raw_pct is not None and float(raw_pct) <= float(cf_day["rule"]["back_skip_raw_pct_le"]))
                        if not skip:
                            cf_day["back"]["n_filtered"] += 1
                            cf_day["back"]["pnl_filtered"] += float(pnl)
                            cf_day["back"]["stake_filtered"] += float(stake)
                except Exception:
                    pass
            elif side == "lay":
                perf["lay"]["n_cov"] += 1
                stake = float(e.stake_sent) if e.stake_sent is not None else 1.0
                liab = stake * max(0.0, float(odd) - 1.0)
                perf["lay"]["stake_sum_cov"] += float(stake)
                roi_liab = _roi_lay_pct_per_liability(float(odd), float(mult))
                if roi_liab is None:
                    continue
                pnl = liab * float(roi_liab) / 100.0
                try:
                    ev_id = str(a.get("event_id") or e.event_id or "").strip()
                    if ev_id:
                        ev_cov_lay.add(ev_id)
                except Exception:
                    pass
                try:
                    reg = _combo_regime_from_audit(a, exec_created_at=e.created_at)
                except Exception:
                    reg = ("In" if bool(e.is_live) else "Pre")
                try:
                    k2 = f"Lay_{reg}"
                    blk = perf.get("pnl_placar_by_type", {}).get(k2) if isinstance(perf.get("pnl_placar_by_type"), dict) else None
                    if isinstance(blk, dict):
                        blk["n"] = int(blk.get("n") or 0) + 1
                        blk["exposure"] = float(blk.get("exposure") or 0.0) + float(liab)
                        blk["pnl"] = float(blk.get("pnl") or 0.0) + float(pnl)
                except Exception:
                    pass
                perf["lay"]["liability_sum_cov"] += float(liab)
                perf["lay"]["pnl_sum"] += float(pnl)
                # outcome invertido vs Back
                if mult > 0:
                    if mult == 0.5:
                        perf["lay"]["half_losses"] += 1
                    else:
                        perf["lay"]["losses"] += 1
                elif mult < 0:
                    if mult == -0.5:
                        perf["lay"]["half_wins"] += 1
                    else:
                        perf["lay"]["wins"] += 1
                else:
                    perf["lay"]["push"] += 1

                raw_pct = _slip_raw_pct(odd_dec=e.odd_decision, odd_fin=e.odd_final)
                cost_pct = _slip_cost_pct(exec_side="lay", odd_dec=e.odd_decision, odd_fin=e.odd_final)
                if raw_pct is not None and cost_pct is not None:
                    slip_raw_lay.append(float(raw_pct))
                    slip_cost_lay.append(float(cost_pct))
                if cost_pct is not None and roi_liab is not None:
                    pairs_lay.append((float(cost_pct), float(roi_liab)))
                if raw_pct is not None and roi_liab is not None:
                    pairs_raw_lay.append((float(raw_pct), float(roi_liab)))
                    total_rows_ctx_lay.append({"slip_raw_pct": float(raw_pct), "roi": float(roi_liab), "odd": float(odd), "exposure": float(liab)})
                    # ROI por stake (bounded): pnl / stake
                    try:
                        roi_st = (float(pnl) / float(stake) * 100.0) if stake > 0 else None
                    except Exception:
                        roi_st = None
                    if roi_st is not None:
                        # reusa o mesmo bucketizador (slip_raw_pct vs ROI), mas com exposure=stake
                        total_rows_ctx_lay_stake.append({"slip_raw_pct": float(raw_pct), "roi": float(roi_st), "odd": float(odd), "exposure": float(stake)})
                    try:
                        regime = _combo_regime_from_audit(a, exec_created_at=e.created_at)
                        rev = _combo_rev_yes_no(a)
                        comb = f"Lay_{regime}_{rev}"
                        comb_pairs_raw.setdefault(comb, []).append((float(raw_pct), float(roi_liab)))
                        comb_meta.setdefault(comb, {"side": "Lay", "regime": regime, "reversal": rev, "league": None})
                    except Exception:
                        pass
                # contrafactual slippage filter (Lay): pula raw>2%
                try:
                    if _cf_allowed(e):
                        cf["lay"]["n"] += 1
                        cf["lay"]["pnl"] += float(pnl)
                        cf["lay"]["stake"] += float(stake)
                        cf["lay"]["liability"] += float(liab)
                        scope = str((cf.get("rule") or {}).get("lay_scope") or "pre").strip().lower()
                        apply_gate = True
                        if scope in ("pre", "in"):
                            apply_gate = (str(reg).strip().lower() == ("pre" if scope == "pre" else "in"))
                        skip = bool(apply_gate and raw_pct is not None and float(raw_pct) > float(cf["rule"]["lay_skip_raw_pct_gt"]))
                        if not skip:
                            cf["lay"]["n_filtered"] += 1
                            cf["lay"]["pnl_filtered"] += float(pnl)
                            cf["lay"]["stake_filtered"] += float(stake)
                            cf["lay"]["liability_filtered"] += float(liab)
                except Exception:
                    pass
                try:
                    if _cf_allowed(e):
                        cf_day["lay"]["n"] += 1
                        cf_day["lay"]["pnl"] += float(pnl)
                        cf_day["lay"]["stake"] += float(stake)
                        cf_day["lay"]["liability"] += float(liab)
                        scope = str((cf_day.get("rule") or {}).get("lay_scope") or "pre").strip().lower()
                        apply_gate = True
                        if scope in ("pre", "in"):
                            apply_gate = (str(reg).strip().lower() == ("pre" if scope == "pre" else "in"))
                        skip = bool(apply_gate and raw_pct is not None and float(raw_pct) > float(cf_day["rule"]["lay_skip_raw_pct_gt"]))
                        if not skip:
                            cf_day["lay"]["n_filtered"] += 1
                            cf_day["lay"]["pnl_filtered"] += float(pnl)
                            cf_day["lay"]["stake_filtered"] += float(stake)
                            cf_day["lay"]["liability_filtered"] += float(liab)
                except Exception:
                    pass

        # ROIs agregados
        back_roi = (float(perf["back"]["pnl_sum"]) / float(perf["back"]["stake_sum_cov"]) * 100.0) if perf["back"]["stake_sum_cov"] else None
        lay_roi = (float(perf["lay"]["pnl_sum"]) / float(perf["lay"]["liability_sum_cov"]) * 100.0) if perf["lay"]["liability_sum_cov"] else None
        perf["back"]["roi_pct"] = back_roi
        perf["lay"]["roi_pct_per_liability"] = lay_roi
        perf["lay_roi_pct_per_stake"] = (float(perf["lay"]["pnl_sum"]) / float(perf["lay"]["stake_sum_cov"]) * 100.0) if perf["lay"]["stake_sum_cov"] else None

        # Contrafactual operacional (placar): filtros de slippage/latência no subconjunto coberto
        try:
            perf["back"]["filters_counterfactual"] = _counterfactual_filters_back(
                cov_back_rows,
                slip_raw_pct_max=float(os.getenv("CF_SLIP_RAW_PCT_MAX", "2.0") or 2.0),
                lat_ms_max=int(float(os.getenv("CF_LAT_CALL_TO_DONE_MS_MAX", "6000") or 6000)),
                slip_missing_pass=True,
                lat_missing_fail_closed=True,
            )
            perf["back"]["filters_counterfactual_by_regime"] = {
                "pre": _counterfactual_filters_back(
                    cov_back_rows_pre,
                    slip_raw_pct_max=float(os.getenv("CF_SLIP_RAW_PCT_MAX", "2.0") or 2.0),
                    lat_ms_max=int(float(os.getenv("CF_LAT_CALL_TO_DONE_MS_MAX", "6000") or 6000)),
                    slip_missing_pass=True,
                    lat_missing_fail_closed=True,
                ),
                "in": _counterfactual_filters_back(
                    cov_back_rows_in,
                    slip_raw_pct_max=float(os.getenv("CF_SLIP_RAW_PCT_MAX", "2.0") or 2.0),
                    lat_ms_max=int(float(os.getenv("CF_LAT_CALL_TO_DONE_MS_MAX", "6000") or 6000)),
                    slip_missing_pass=True,
                    lat_missing_fail_closed=True,
                ),
            }
        except Exception:
            pass

        # cobertura (placar) entre sucessos — Back/Lay
        try:
            ns = int(perf.get("back", {}).get("n_success") or 0)
            nc = int(perf.get("back", {}).get("n_cov") or 0)
            st = float(perf.get("back", {}).get("stake_sum") or 0.0)
            stc = float(perf.get("back", {}).get("stake_sum_cov") or 0.0)
            perf["back"]["n_uncov"] = int(max(0, ns - nc))
            perf["back"]["stake_sum_uncov"] = float(max(0.0, st - stc))
            perf["back"]["cov_pct_n"] = (float(nc) / float(ns) * 100.0) if ns > 0 else None
            perf["back"]["cov_pct_stake"] = (float(stc) / float(st) * 100.0) if st > 0 else None
        except Exception:
            pass
        try:
            ns = int(perf.get("lay", {}).get("n_success") or 0)
            nc = int(perf.get("lay", {}).get("n_cov") or 0)
            st = float(perf.get("lay", {}).get("stake_sum") or 0.0)
            stc = float(perf.get("lay", {}).get("stake_sum_cov") or 0.0)
            li = float(perf.get("lay", {}).get("liability_sum") or 0.0)
            lic = float(perf.get("lay", {}).get("liability_sum_cov") or 0.0)
            perf["lay"]["n_uncov"] = int(max(0, ns - nc))
            perf["lay"]["stake_sum_uncov"] = float(max(0.0, st - stc))
            perf["lay"]["liability_sum_uncov"] = float(max(0.0, li - lic))
            perf["lay"]["cov_pct_n"] = (float(nc) / float(ns) * 100.0) if ns > 0 else None
            perf["lay"]["cov_pct_stake"] = (float(stc) / float(st) * 100.0) if st > 0 else None
        except Exception:
            pass

        # cobertura por jogo (event_id) — útil para conciliar com accounting por jogo
        try:
            perf["back"]["events_success_n"] = int(len(ev_success_back))
            perf["back"]["events_cov_n"] = int(len(ev_cov_back))
            perf["back"]["events_cov_pct"] = (float(len(ev_cov_back)) / float(len(ev_success_back)) * 100.0) if ev_success_back else None
            perf["back"]["event_ids_success"] = _uniq_sorted_str(ev_success_back)
            perf["back"]["event_ids_cov"] = _uniq_sorted_str(ev_cov_back)
        except Exception:
            pass
        try:
            perf["lay"]["events_success_n"] = int(len(ev_success_lay))
            perf["lay"]["events_cov_n"] = int(len(ev_cov_lay))
            perf["lay"]["events_cov_pct"] = (float(len(ev_cov_lay)) / float(len(ev_success_lay)) * 100.0) if ev_success_lay else None
            perf["lay"]["event_ids_success"] = _uniq_sorted_str(ev_success_lay)
            perf["lay"]["event_ids_cov"] = _uniq_sorted_str(ev_cov_lay)
        except Exception:
            pass

        # slippage agregada
        if slip_raw_back:
            perf["slippage"]["back"]["n"] = int(len(slip_raw_back))
            perf["slippage"]["back"]["raw_pct_mean"] = float(sum(slip_raw_back) / len(slip_raw_back))
            perf["slippage"]["back"]["cost_pct_mean"] = float(sum(slip_cost_back) / len(slip_cost_back)) if slip_cost_back else None
        if slip_raw_lay:
            perf["slippage"]["lay"]["n"] = int(len(slip_raw_lay))
            perf["slippage"]["lay"]["raw_pct_mean"] = float(sum(slip_raw_lay) / len(slip_raw_lay))
            perf["slippage"]["lay"]["cost_pct_mean"] = float(sum(slip_cost_lay) / len(slip_cost_lay)) if slip_cost_lay else None

        if pairs_back:
            perf["slippage_vs_roi"]["back"]["corr_cost_pct_vs_roi"] = _pearson([c for c, _ in pairs_back], [r for _, r in pairs_back])
            perf["slippage_vs_roi"]["back"]["buckets"] = _bucketize(pairs_back)
        if pairs_lay:
            perf["slippage_vs_roi"]["lay"]["corr_cost_pct_vs_roi"] = _pearson([c for c, _ in pairs_lay], [r for _, r in pairs_lay])
            perf["slippage_vs_roi"]["lay"]["buckets"] = _bucketize(pairs_lay)

        # Buckets por slippage RAW (com sinal): <=-2%, -2..2, >2
        perf["slippage_vs_roi_raw"] = {
            "back": {"buckets": _bucketize_3way_raw(pairs_raw_back)},
            "lay": {"buckets": _bucketize_3way_raw(pairs_raw_lay)},
        }

        # acumula (janela inteira) — com corte pós-fix se configurado
        if _slip_allowed_day(d):
            if pairs_raw_back:
                total_pairs_raw_back.extend(list(pairs_raw_back))
            if pairs_raw_lay:
                total_pairs_raw_lay.extend(list(pairs_raw_lay))
            if pairs_back:
                total_pairs_cost_back.extend(list(pairs_back))
            if pairs_lay:
                total_pairs_cost_lay.extend(list(pairs_lay))

        per_day.append(
            {
                "day": d.isoformat(),
                "start_utc": start_utc.isoformat(),
                "end_utc": end_utc.isoformat(),
                "policy": active_by_day.get("days", {}).get(d.isoformat()),
                "bridge": bridge_stats,
                "execution": perf,
                "slippage_filter_counterfactual": cf_day,
            }
        )

    # Carry-forward do último step para dias sem mapeamento (test_days com gaps),
    # refletindo a operação real (policy "corrente" vale até próxima atualização).
    if last_step and isinstance(last_step, dict):
        for it in per_day:
            if not isinstance(it, dict):
                continue
            if it.get("policy") is None:
                it["policy"] = {
                    "active_keys": list(last_step.get("active_keys") or []),
                    "n_active_keys": len(list(last_step.get("active_keys") or [])),
                    "train": last_step.get("train"),
                    "test": last_step.get("test"),
                    "train_days": last_step.get("train_days"),
                    "test_days": last_step.get("test_days"),
                    "carried_forward": True,
                }

    out = {
        "ts_utc": now_utc.isoformat(),
        "tz": tz_name,
        "policy_json": str(policy_json),
        "executor_jsonl": str(executor_jsonl),
        "range": {
            "start_day": start_day.isoformat(),
            "end_day": end_day.isoformat(),
            "days": int(days),
            "include_today": bool(include_today),
            # Clarifica a diferença entre "days (arg)" e a duração real do span.
            "span_days": base_span_days,
        },
        "slippage_range": {
            "start_day": slip_start_day.isoformat(),
            "end_day": end_day.isoformat(),
            "span_days": slip_span_days,
            "cut_start_day_local": cf_start_day.isoformat() if cf_start_day else None,
        },
        "policy_days": active_by_day,
        "per_day": per_day,
        "observed_ah_line_abs": ah_obs,
        "slippage_filter_counterfactual": cf,
        "slippage_start_day_local": cf_start_day.isoformat() if cf_start_day else None,
        # Estatística acumulada na janela
        "slippage_vs_roi_raw_total": {
            "back": {"buckets": _bucketize_3way_raw(total_pairs_raw_back)},
            "lay": {"buckets": _bucketize_3way_raw(total_pairs_raw_lay)},
        },
        "slippage_vs_roi_raw_total_ctx": {
            "back": {"buckets": _bucketize_3way_raw_with_context(total_rows_ctx_back)},
            "lay": {"buckets": _bucketize_3way_raw_with_context(total_rows_ctx_lay)},
        },
        "slippage_vs_roi_raw_total_ctx_by_regime": {
            "back_pre": {"buckets": _bucketize_3way_raw_with_context(total_rows_ctx_back_pre)},
            "back_in": {"buckets": _bucketize_3way_raw_with_context(total_rows_ctx_back_in)},
        },
        # Lay também em ROI por stake (bounded; útil para sanity-check de retornos)
        "slippage_vs_roi_raw_total_ctx_lay_stake": {
            "lay": {"buckets": _bucketize_3way_raw_with_context(total_rows_ctx_lay_stake)},
        },
        "slippage_vs_roi_total": {
            "back": {
                "n": int(len(total_pairs_cost_back)),
                "corr_cost_pct_vs_roi": _pearson([c for c, _ in total_pairs_cost_back], [r for _, r in total_pairs_cost_back]) if total_pairs_cost_back else None,
            },
            "lay": {
                "n": int(len(total_pairs_cost_lay)),
                "corr_cost_pct_vs_roi": _pearson([c for c, _ in total_pairs_cost_lay], [r for _, r in total_pairs_cost_lay]) if total_pairs_cost_lay else None,
            },
        },
    }
    # Top combinações por volume (slippage_raw_pct vs ROI)
    try:
        rows = []
        for comb, pairs in (comb_pairs_raw or {}).items():
            if not pairs:
                continue
            meta = comb_meta.get(comb) or {}
            buckets = _bucketize_3way_raw(pairs)
            corr = _pearson([s for s, _ in pairs], [r for _, r in pairs]) if len(pairs) >= 5 else None
            rows.append(
                {
                    "comb": comb,
                    "n": int(len(pairs)),
                    "side": meta.get("side"),
                    "regime": meta.get("regime"),
                    "reversal": meta.get("reversal"),
                    "league": meta.get("league"),
                    "corr_raw_pct_vs_roi": corr,
                    "buckets": buckets,
                }
            )
        rows.sort(key=lambda r: int(r.get("n") or 0), reverse=True)
        out["slippage_vs_roi_raw_by_combo_top"] = rows[:40]
    except Exception:
        pass
    out = _json_safe(out)
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Relatório de aderência OOS: portfolio (WF policy) por dia + execução (bridge+executor) + ROI por placar.")
    ap.add_argument("--policy-json", default=os.getenv("BRIDGE_POLICY_JSON", "logs/wf_policy_current.json"))
    ap.add_argument("--executor-jsonl", default=os.getenv("EXECUTOR_JSONL", "logs/executor_live.jsonl"))
    ap.add_argument("--tz", default=os.getenv("REPORT_TZ", "America/Sao_Paulo"))
    ap.add_argument("--days", type=int, default=int(os.getenv("OOS_ADHERENCE_DAYS", "7")))
    ap.add_argument("--include-today", action="store_true", default=(os.getenv("OOS_ADHERENCE_INCLUDE_TODAY", "1").strip() not in ("0", "false", "False", "no", "NO")))
    ap.add_argument("--no-per-day", action="store_true", default=(os.getenv("OOS_ADHERENCE_NO_PER_DAY", "0").strip() in ("1", "true", "True", "yes", "YES")))
    ap.add_argument(
        "--slippage-cf-start-day",
        default=os.getenv("OOS_ADHERENCE_SLIP_CF_START_DAY", "").strip() or None,
        help="(opcional) Restringe o contrafactual de slippage (placar) para execuções a partir deste dia local (YYYY-MM-DD).",
    )
    ap.add_argument("--out", default=os.getenv("OOS_ADHERENCE_OUT", "").strip() or None)
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))

    policy = Path(str(args.policy_json))
    jsonl = Path(str(args.executor_jsonl))
    outp = Path(str(args.out)) if args.out else None

    import asyncio

    rep = asyncio.run(
        run_report(
            policy_json=policy,
            executor_jsonl=jsonl,
            tz_name=str(args.tz),
            days=int(args.days),
            include_today=bool(args.include_today),
            no_per_day=bool(args.no_per_day),
            slip_cf_start_day=str(args.slippage_cf_start_day).strip() if args.slippage_cf_start_day else None,
            out_json=outp,
        )
    )
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

