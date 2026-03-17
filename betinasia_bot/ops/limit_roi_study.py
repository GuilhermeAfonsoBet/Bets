from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import String, bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY

from storage.database import Database


def _load_env_file(path: Path) -> None:
    """
    Loader simples de .env para permitir rodar este módulo isoladamente (sem depender do shell exportar vars).
    - Não sobrescreve variáveis já presentes no ambiente.
    - Ignora comentários e linhas inválidas.
    """
    try:
        if not path.exists():
            return
        for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = str(ln).strip()
            if not s or s.startswith("#"):
                continue
            if "=" not in s:
                continue
            k, v = s.split("=", 1)
            k = k.strip()
            if not k:
                continue
            v = v.strip()
            if (len(v) >= 2) and ((v[0] == v[-1]) and v[0] in ("'", '"')):
                v = v[1:-1]
            if k in os.environ and str(os.environ.get(k) or "").strip():
                continue
            os.environ[k] = v
    except Exception:
        return


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _to_utc(ts: Any) -> Optional[datetime]:
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    return None


def _get_path(d: Any, path: List[Any]) -> Any:
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _extract_ws_series(hypothesis_details: dict) -> List[dict]:
    """
    Normaliza `hypothesis_details.ws_series` (WS-only) para pontos {t, odd}.
    Observação: os pontos podem vir com `t_target_s` e `t_actual_s`; preferimos `t_actual_s`.
    """
    try:
        arr = hypothesis_details.get("ws_series")
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
    except Exception:
        return []


def _build_lay_series(*, ws0: float, hypothesis_details: dict, lay0: Optional[float]) -> List[dict]:
    """
    Série temporal do Lay (proxy) para achar vale e reversão.
    Preferimos `lay_temporal` quando existir; caso contrário usamos `ws_series`.
    """
    if ws0 <= 0:
        return []
    ws_series = _extract_ws_series(hypothesis_details)
    series: List[dict] = []

    if lay0 is not None and lay0 > 0:
        series.append({"t": 0.0, "odd": float(lay0), "diff_pct": float((float(lay0) - ws0) / ws0 * 100.0)})
    elif ws_series:
        o0 = _safe_float(ws_series[0].get("odd"))
        if o0 is None or o0 <= 0:
            return []
        series.append({"t": 0.0, "odd": float(o0), "diff_pct": float((float(o0) - ws0) / ws0 * 100.0)})
    else:
        return []

    arr = hypothesis_details.get("lay_temporal")
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
            series.append({"t": float(t), "odd": float(odd), "diff_pct": float((float(odd) - ws0) / ws0 * 100.0)})
    else:
        for p in ws_series:
            t = _safe_float(p.get("t"))
            odd = _safe_float(p.get("odd"))
            if t is None or odd is None or odd <= 0:
                continue
            if float(t) <= 0.0005:
                continue
            series.append({"t": float(t), "odd": float(odd), "diff_pct": float((float(odd) - ws0) / ws0 * 100.0)})

    series.sort(key=lambda x: float(x.get("t") or 0.0))
    return series


def _analyze_vale_reversao(series: List[dict], *, eps_rev: float = 0.5) -> dict:
    """
    Para Lay: vale = min(diff_pct). Reversão = após o vale, diff_pct subir >= eps_rev p.p.
    """
    if not series:
        return {"n": 0}
    diffs = [float(p.get("diff_pct") or 0.0) for p in series]
    idx_ext = min(range(len(diffs)), key=lambda i: diffs[i])
    ext = series[idx_ext]
    last = series[-1]
    had_rev = False
    t_rev = None
    odd_rev = None
    diff_rev = None
    if idx_ext + 1 < len(series):
        threshold = float(ext.get("diff_pct") or 0.0) + float(eps_rev)
        for p in series[idx_ext + 1 :]:
            try:
                if float(p.get("diff_pct") or 0.0) >= threshold:
                    had_rev = True
                    t_rev = float(p.get("t") or 0.0)
                    odd_rev = _safe_float(p.get("odd"))
                    diff_rev = float(p.get("diff_pct") or 0.0)
                    break
            except Exception:
                continue
    return {
        "n": len(series),
        "t_ext": float(ext.get("t") or 0.0),
        "diff_ext": float(ext.get("diff_pct") or 0.0),
        "odd_ext": float(ext.get("odd") or 0.0),
        "t_last": float(last.get("t") or 0.0),
        "diff_last": float(last.get("diff_pct") or 0.0),
        "odd_last": float(last.get("odd") or 0.0),
        "had_reversal": bool(had_rev),
        "t_reversal": t_rev,
        "odd_reversal": odd_rev,
        "diff_reversal": diff_rev,
    }


def _is_live_eff(is_live: Any, *, audited_at: Optional[datetime], kickoff_time: Optional[datetime]) -> Optional[bool]:
    try:
        if is_live is True:
            return True
        if is_live is False:
            return False
    except Exception:
        pass
    if audited_at is not None and kickoff_time is not None:
        try:
            return bool(audited_at >= kickoff_time)
        except Exception:
            return None
    return None


def _ah_outcome_mult(*, line: Any, side: Any, home_score: Any, away_score: Any) -> Optional[float]:
    """
    Retorna o multiplicador do resultado para a seleção "Back" do lado/linha informados.
    Convenção: +1 win, +0.5 half-win, 0 push, -0.5 half-loss, -1 loss.
    Mesma convenção usada no relatório robusto.
    """
    if home_score is None or away_score is None:
        return None
    try:
        gd = int(home_score) - int(away_score)
    except Exception:
        return None
    try:
        ah_line = float(str(line).replace(",", "."))
    except Exception:
        return None
    try:
        if str(side or "").strip() == "home":
            adjusted = float(gd) + float(ah_line)
        else:
            adjusted = -float(gd) - float(ah_line)
    except Exception:
        return None

    if adjusted > 0.25:
        return 1.0
    if adjusted == 0.25:
        return 0.5
    if adjusted == 0.0:
        return 0.0
    if adjusted == -0.25:
        return -0.5
    return -1.0


def _roi_lay_pct_per_liability(*, lay_odd: Any, mult_back: Optional[float]) -> Optional[float]:
    """
    ROI (%) por *liability* para Lay.
    - Se o Back ganha (mult>0): Lay perde mult*liability => ROI = -mult*100
    - Se o Back perde (mult<0): Lay ganha (-mult)*stake, com stake = liability/(odd-1)
      => ROI = (-mult)/(odd-1)*100
    """
    if mult_back is None:
        return None
    o = _safe_float(lay_odd)
    if o is None or o <= 1.0:
        return None
    if float(mult_back) > 0:
        return -float(mult_back) * 100.0
    if float(mult_back) < 0:
        return (-float(mult_back)) / max(1e-9, (float(o) - 1.0)) * 100.0
    return 0.0


def _pct(x: Optional[float], nd: int = 2) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.{nd}f}%"
    except Exception:
        return "—"


def _num(x: Optional[float], nd: int = 2) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):,.{nd}f}"
    except Exception:
        return "—"


def _parse_csv_floats(s: str) -> List[float]:
    out: List[float] = []
    if not s:
        return out
    for it in str(s).split(","):
        t = str(it).strip()
        if not t:
            continue
        try:
            out.append(float(t))
        except Exception:
            continue
    # dedup preserve order
    seen = set()
    out2: List[float] = []
    for x in out:
        k = float(x)
        if k in seen:
            continue
        seen.add(k)
        out2.append(float(k))
    return out2


@dataclass(frozen=True)
class Row:
    audited_at_utc: datetime
    is_live_eff: Optional[bool]
    line: Any
    side: Any
    home_score: Any
    away_score: Any
    betslip_odd: Optional[float]
    websocket_odd: Optional[float]
    betslip_limit: Optional[float]
    lay_available_limit: Optional[float]
    lay_odd_hint: Optional[float]
    hypothesis_details: Dict[str, Any]

    def entry_odd(self, *, mode: str, wf_lay_end_sec: float, wf_lay_end_max_gap_sec: float, eps_rev: float) -> Optional[float]:
        """
        mode:
        - 'betslip': usa betslip_odd (rápido; útil para sanity-check)
        - 'lay_policy': replica a regra do relatório robusto: se há reversão, entra após reversão; senão entra ~t+end_sec (ou último ponto)
        """
        if str(mode) == "betslip":
            for v in (self.betslip_odd, self.lay_odd_hint, self.websocket_odd):
                if v is not None and float(v) > 1e-9:
                    return float(v)
            return None

        # lay_policy (mais fiel ao Sweep do PDF)
        ws0 = self.websocket_odd
        if ws0 is None or float(ws0) <= 0:
            ws_series = _extract_ws_series(self.hypothesis_details or {})
            if ws_series:
                ws0 = _safe_float(ws_series[0].get("odd"))
        if ws0 is None or float(ws0) <= 0:
            return None

        h = self.hypothesis_details or {}
        lay0 = _safe_float(_get_path(h, ["lay", "odd"]))
        series = _build_lay_series(ws0=float(ws0), hypothesis_details=h, lay0=lay0)
        if not series:
            return None
        a = _analyze_vale_reversao(series, eps_rev=float(eps_rev))
        if int(a.get("n") or 0) <= 0:
            return None

        odd_rev = _safe_float(a.get("odd_reversal"))
        if bool(a.get("had_reversal")) and odd_rev is not None and float(odd_rev) > 0:
            return float(odd_rev)

        # sem reversão: pega ponto mais próximo de t=end_sec (se dentro do gap); senão último
        target = float(wf_lay_end_sec)
        max_gap = float(wf_lay_end_max_gap_sec)
        p_end = series[-1]
        try:
            p0 = min(series, key=lambda p: abs(float(p.get("t") or 0.0) - target))
            if abs(float(p0.get("t") or 0.0) - target) <= max_gap:
                p_end = p0
        except Exception:
            p_end = series[-1]
        odd_end = _safe_float(p_end.get("odd"))
        if odd_end is None or float(odd_end) <= 0:
            return None
        return float(odd_end)

    def limit_stake(self) -> Optional[float]:
        # Para Lay, preferimos `available_limit` (stake max); senão usa betslip_limit.
        for v in (self.lay_available_limit, self.betslip_limit):
            if v is not None and float(v) > 0:
                return float(v)
        return None


@dataclass(frozen=True)
class Event:
    day_utc: str  # YYYY-MM-DD
    limit_stake: float
    odd_entry: float
    roi_liab_pct: float
    diff_entry_pct: float


async def _fetch_rows(
    *,
    db: Database,
    since_utc: datetime,
    until_utc: datetime,
    direction: Optional[str],
    only_status: Sequence[str],
) -> List[Row]:
    only = [str(s).upper() for s in (only_status or []) if str(s).strip()]
    if not only:
        only = ["OK"]
    # Nota (asyncpg): `(:direction IS NULL OR a.reversal_direction = :direction)` pode gerar
    # AmbiguousParameterError se `reversal_direction` for enum/domínio. Para robustez, comparamos
    # via cast do lado da coluna.
    q = text(
        """
        SELECT
            a.audited_at,
            a.is_live,
            a.status,
            a.reversal_direction,
            a.market_type,
            a.line,
            a.side,
            a.websocket_odd,
            a.betslip_odd,
            a.betslip_limit,
            a.hypothesis_details,
            m.kickoff_time,
            m.home_score,
            m.away_score
        FROM betslip_audit_results a
        JOIN matches m ON m.external_id = a.event_id
        WHERE a.hypothesis_type = 'H3B'
          AND upper(a.market_type) = 'AH'
          AND a.audited_at >= :since_utc
          AND a.audited_at < :until_utc
          AND upper(a.status) = ANY(:only_status)
          AND (:direction IS NULL OR a.reversal_direction::text = :direction)
        """
    ).bindparams(
        bindparam("since_utc"),
        bindparam("until_utc"),
        bindparam("direction", type_=String()),
        bindparam("only_status", type_=ARRAY(String())),
    )
    out: List[Row] = []
    async with db.async_session() as session:
        res = await session.execute(
            q,
            {
                "since_utc": since_utc,
                "until_utc": until_utc,
                "direction": direction,
                "only_status": only,
            },
        )
        for r in res.fetchall():
            audited_at = _to_utc(getattr(r, "audited_at", None) or (r[0] if len(r) > 0 else None))
            if audited_at is None:
                continue
            is_live = getattr(r, "is_live", None)
            kickoff = _to_utc(getattr(r, "kickoff_time", None))
            hs = getattr(r, "home_score", None)
            aws = getattr(r, "away_score", None)
            h = getattr(r, "hypothesis_details", None)
            if not isinstance(h, dict):
                h = {}
            lay = h.get("lay") if isinstance(h.get("lay"), dict) else {}
            out.append(
                Row(
                    audited_at_utc=audited_at,
                    is_live_eff=_is_live_eff(is_live, audited_at=audited_at, kickoff_time=kickoff),
                    line=getattr(r, "line", None),
                    side=getattr(r, "side", None),
                    home_score=hs,
                    away_score=aws,
                    betslip_odd=_safe_float(getattr(r, "betslip_odd", None)),
                    websocket_odd=_safe_float(getattr(r, "websocket_odd", None)),
                    betslip_limit=_safe_float(getattr(r, "betslip_limit", None)),
                    lay_available_limit=_safe_float(lay.get("available_limit")),
                    lay_odd_hint=_safe_float(lay.get("odd")),
                    hypothesis_details=h,
                )
            )
    return out


def _quantile_bins(xs: List[float], *, ntiles: int) -> List[Tuple[float, float]]:
    if not xs:
        return []
    ys = sorted(float(x) for x in xs if x is not None and float(x) > 0)
    if not ys:
        return []
    out: List[Tuple[float, float]] = []
    n = len(ys)
    for k in range(ntiles):
        i0 = int(round(k * n / ntiles))
        i1 = int(round((k + 1) * n / ntiles))
        i0 = max(0, min(n - 1, i0))
        i1 = max(i0 + 1, min(n, i1))
        lo = ys[i0]
        hi = ys[i1 - 1]
        out.append((float(lo), float(hi)))
    return out


def _assign_bin(x: float, bins: List[Tuple[float, float]]) -> Optional[int]:
    for i, (lo, hi) in enumerate(bins):
        if float(lo) <= float(x) <= float(hi):
            return i
    return None


def _assign_bin_clamped(x: float, bins: List[Tuple[float, float]]) -> Optional[int]:
    if not bins:
        return None
    idx = _assign_bin(x, bins)
    if idx is not None:
        return idx
    # se cair fora (buracos por quantis com empates), clampa para o bin mais próximo
    try:
        if float(x) < float(bins[0][0]):
            return 0
        if float(x) > float(bins[-1][1]):
            return len(bins) - 1
    except Exception:
        return None
    best_i = None
    best_d = None
    for i, (lo, hi) in enumerate(bins):
        try:
            if float(x) < float(lo):
                d = float(lo) - float(x)
            elif float(x) > float(hi):
                d = float(x) - float(hi)
            else:
                d = 0.0
        except Exception:
            continue
        if best_d is None or d < best_d:
            best_d = d
            best_i = i
    return best_i


def _profit_for_cap(rows: Iterable[Tuple[float, float, float]], cap_liab: float) -> float:
    """
    rows: (limit_stake, odd_entry, roi_liab_pct)
    """
    pnl = 0.0
    for lim_st, odd, roi in rows:
        liab_lim = float(lim_st) * max(0.0, float(odd) - 1.0)
        exp = min(float(cap_liab), float(liab_lim))
        pnl += float(exp) * float(roi) / 100.0
    return float(pnl)


def _liab_used_for_cap(rows: Iterable[Tuple[float, float, float]], cap_liab: float) -> float:
    used = 0.0
    for lim_st, odd, _roi in rows:
        liab_lim = float(lim_st) * max(0.0, float(odd) - 1.0)
        used += min(float(cap_liab), float(liab_lim))
    return float(used)


def _pnl_used_for_cap(events: Iterable[Event], *, cap_liab: float, cap_by_bin: Optional[Dict[int, float]] = None, bins: Optional[List[Tuple[float, float]]] = None) -> Tuple[float, float]:
    pnl = 0.0
    used = 0.0
    for e in events:
        try:
            liab_lim = float(e.limit_stake) * max(0.0, float(e.odd_entry) - 1.0)
        except Exception:
            continue
        cap_eff = float(cap_liab)
        if cap_by_bin is not None and bins is not None:
            bi = _assign_bin_clamped(float(e.limit_stake), bins)
            if bi is not None and int(bi) in cap_by_bin:
                cap_eff = min(float(cap_eff), float(cap_by_bin[int(bi)]))
        exp = min(float(cap_eff), float(liab_lim)) if cap_eff > 0 else 0.0
        used += float(exp)
        pnl += float(exp) * float(e.roi_liab_pct) / 100.0
    return float(pnl), float(used)


def _run_walk_forward(events: List[Event], *, since: datetime, until: datetime, args: argparse.Namespace) -> None:
    """
    Walk-forward simples por dias (sem lookahead):
    - bins de limit e decisão de cap/drop são aprendidos no TREINO e aplicados no TESTE.
    - objetivo: maximizar P&L absoluto no teste sob guardrails de ROI/liab mínimo no bin.
    """
    from datetime import date

    def _parse_day(s: str) -> Optional[date]:
        try:
            y, m, d = str(s).split("-", 2)
            return date(int(y), int(m), int(d))
        except Exception:
            return None

    by_day: Dict[str, List[Event]] = {}
    for e in events:
        by_day.setdefault(str(e.day_utc), []).append(e)

    ds = sorted({_parse_day(d) for d in by_day.keys() if _parse_day(d) is not None})
    if not ds:
        print("[wf] Sem dias válidos.")
        return
    d0 = min(ds)
    d1 = max(ds)
    # calendário contínuo (evita “pular” dias sem eventos)
    cal: List[date] = []
    cur = d0
    while cur <= d1:
        cal.append(cur)
        cur = cur + timedelta(days=1)

    train_days = int(getattr(args, "wf_train_days", 14) or 14)
    test_days = int(getattr(args, "wf_test_days", 7) or 7)
    step_days = int(getattr(args, "wf_step_days", test_days) or test_days)
    wf_ntiles = int(getattr(args, "wf_limit_ntiles", 0) or 0)
    if wf_ntiles <= 0:
        wf_ntiles = int(getattr(args, "limit_ntiles", 10) or 10)
    base_cap = float(getattr(args, "wf_base_cap", 50.0) or 50.0)
    max_cap = float(getattr(args, "wf_max_cap", 100.0) or 100.0)
    min_roi_base = float(getattr(args, "wf_min_roi_base_cap_pct", 0.0) or 0.0)
    min_roi_marg = float(getattr(args, "wf_min_roi_marg_pct", 5.0) or 5.0)
    min_n_bin = int(getattr(args, "wf_min_n_bin", 8) or 8)

    # indices de início do teste
    starts = []
    for i in range(len(cal)):
        t0 = cal[i]
        t1 = t0 + timedelta(days=test_days - 1)
        tr0 = t0 - timedelta(days=train_days)
        tr1 = t0 - timedelta(days=1)
        if tr0 < d0:
            continue
        if t1 > d1:
            break
        starts.append(i)
    if not starts:
        print("[wf] Janela insuficiente para walk-forward (verifique lookback/train/test).")
        return

    print("\n### Walk-forward (treino→teste) — política por bins de limit\n")
    print(f"- Treino: {train_days}d | Teste: {test_days}d | Step: {step_days}d")
    print(f"- Bins (limit_ntiles no treino): {wf_ntiles}")
    print(f"- Caps: base={base_cap:.2f} max={max_cap:.2f}")
    print(f"- Guardrails: ROI@base >= {min_roi_base:.2f}% ; ROI_marg(base→max) >= {min_roi_marg:.2f}% ; min_n_bin={min_n_bin}\n")

    hdr = (
        "| step | train (UTC) | test (UTC) | n_train | n_test | pnl_policy | pnl_cap50 | pnl_cap100 | ROI_policy | ROI_cap50 | ROI_cap100 | drop_bins | expand_bins |\n"
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    print(hdr)

    # agregados
    agg = {"p_pol": 0.0, "u_pol": 0.0, "p50": 0.0, "u50": 0.0, "p100": 0.0, "u100": 0.0, "steps": 0}

    step_idx = 0
    i = 0
    while i < len(cal):
        if i not in starts:
            i += 1
            continue
        t0 = cal[i]
        tr0 = t0 - timedelta(days=train_days)
        tr1 = t0 - timedelta(days=1)
        te0 = t0
        te1 = t0 + timedelta(days=test_days - 1)

        # coleta eventos
        train_events: List[Event] = []
        test_events: List[Event] = []
        cur = tr0
        while cur <= tr1:
            train_events.extend(by_day.get(cur.isoformat(), []))
            cur = cur + timedelta(days=1)
        cur = te0
        while cur <= te1:
            test_events.extend(by_day.get(cur.isoformat(), []))
            cur = cur + timedelta(days=1)

        if not train_events or not test_events:
            i += step_days
            continue

        # bins aprendidos no treino
        train_lims = [float(e.limit_stake) for e in train_events if float(e.limit_stake) > 0]
        bins = _quantile_bins(train_lims, ntiles=int(wf_ntiles))
        cap_by_bin: Dict[int, float] = {}
        drop_bins = 0
        expand_bins = 0
        if bins:
            # alerta de parametrização: se min_n_bin é alto demais, a policy tende a não fazer nada (cap_base em tudo)
            try:
                approx = float(len(train_events)) / max(1.0, float(len(bins)))
                if float(min_n_bin) > approx + 1e-9:
                    print(
                        f"| {step_idx+1} | {tr0.isoformat()}→{tr1.isoformat()} | {te0.isoformat()}→{te1.isoformat()} | {len(train_events)} | {len(test_events)} | "
                        f"— | — | — | — | — | — | — | — |"
                    )
                    print(
                        f"[wf][warn] step {step_idx+1}: n_train={len(train_events)} com {len(bins)} bins ⇒ ~{approx:.1f}/bin < min_n_bin={min_n_bin}. "
                        f"Considere `--wf-limit-ntiles 4|5` e/ou `--wf-min-n-bin 2|3`.\n"
                    )
            except Exception:
                pass
            for bi, (lo, hi) in enumerate(bins):
                sub = [e for e in train_events if (float(lo) <= float(e.limit_stake) <= float(hi))]
                if len(sub) < min_n_bin:
                    cap_by_bin[int(bi)] = float(base_cap)
                    continue
                p_base, u_base = _pnl_used_for_cap(sub, cap_liab=float(base_cap))
                roi_base = (p_base / u_base * 100.0) if u_base > 1e-12 else None
                p_max, u_max = _pnl_used_for_cap(sub, cap_liab=float(max_cap))
                roi_marg = ((p_max - p_base) / (u_max - u_base) * 100.0) if (u_max - u_base) > 1e-12 else None

                if roi_base is not None and float(roi_base) < float(min_roi_base):
                    cap_by_bin[int(bi)] = 0.0
                    drop_bins += 1
                elif roi_marg is not None and float(roi_marg) >= float(min_roi_marg):
                    cap_by_bin[int(bi)] = float(max_cap)
                    expand_bins += 1
                else:
                    cap_by_bin[int(bi)] = float(base_cap)

        # avaliação no teste
        pnl_pol, used_pol = _pnl_used_for_cap(test_events, cap_liab=float(max_cap), cap_by_bin=cap_by_bin, bins=bins)
        pnl_50, used_50 = _pnl_used_for_cap(test_events, cap_liab=float(base_cap))
        pnl_100, used_100 = _pnl_used_for_cap(test_events, cap_liab=float(max_cap))

        roi_pol = (pnl_pol / used_pol * 100.0) if used_pol > 1e-12 else None
        roi50 = (pnl_50 / used_50 * 100.0) if used_50 > 1e-12 else None
        roi100 = (pnl_100 / used_100 * 100.0) if used_100 > 1e-12 else None

        print(
            f"| {step_idx+1} | {tr0.isoformat()}→{tr1.isoformat()} | {te0.isoformat()}→{te1.isoformat()} | {len(train_events)} | {len(test_events)} | "
            f"{_num(pnl_pol,2)} | {_num(pnl_50,2)} | {_num(pnl_100,2)} | {_pct(roi_pol,2)} | {_pct(roi50,2)} | {_pct(roi100,2)} | {drop_bins} | {expand_bins} |"
        )

        agg["p_pol"] += float(pnl_pol)
        agg["u_pol"] += float(used_pol)
        agg["p50"] += float(pnl_50)
        agg["u50"] += float(used_50)
        agg["p100"] += float(pnl_100)
        agg["u100"] += float(used_100)
        agg["steps"] += 1

        step_idx += 1
        i += step_days

    if agg["steps"] <= 0:
        print("\n[wf] Nenhum step válido (train/test sem eventos).")
        return

    roi_pol = (agg["p_pol"] / agg["u_pol"] * 100.0) if agg["u_pol"] > 1e-12 else None
    roi50 = (agg["p50"] / agg["u50"] * 100.0) if agg["u50"] > 1e-12 else None
    roi100 = (agg["p100"] / agg["u100"] * 100.0) if agg["u100"] > 1e-12 else None
    print("\n**Resumo WF (agregado nos testes)**")
    print("| cenário | P&L | liab_usada | ROI/liab |")
    print("|---|---:|---:|---:|")
    print(f"| policy (bins) | {_num(agg['p_pol'],2)} | {_num(agg['u_pol'],2)} | {_pct(roi_pol,2)} |")
    print(f"| baseline cap{_num(base_cap,0)} | {_num(agg['p50'],2)} | {_num(agg['u50'],2)} | {_pct(roi50,2)} |")
    print(f"| baseline cap{_num(max_cap,0)} | {_num(agg['p100'],2)} | {_num(agg['u100'],2)} | {_pct(roi100,2)} |")
    print()


async def _run(args: argparse.Namespace) -> int:
    _load_env_file(Path(os.getenv("ENV_FILE", ".env")))

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=float(args.lookback_days))

    db = Database(database_url=(args.database_url or None))
    await db.connect()
    try:
        rows = await _fetch_rows(
            db=db,
            since_utc=since,
            until_utc=now,
            direction=(str(args.direction).strip() if args.direction is not None else None),
            only_status=[s.strip() for s in str(args.only_status or "").split(",") if s.strip()],
        )
    finally:
        try:
            await db.close()
        except Exception:
            pass

    # Filtro: Lay + In (ou conforme args)
    keep: List[Event] = []
    for r in rows:
        if args.regime != "all":
            want_live = True if args.regime == "in" else False
            if r.is_live_eff is None:
                continue
            if bool(r.is_live_eff) is not bool(want_live):
                continue

        odd = r.entry_odd(
            mode=str(args.entry_mode),
            wf_lay_end_sec=float(args.wf_lay_end_sec),
            wf_lay_end_max_gap_sec=float(args.wf_lay_end_max_gap_sec),
            eps_rev=float(args.eps_rev),
        )
        lim = r.limit_stake()
        if odd is None or lim is None:
            continue
        if float(odd) <= 1.0 or float(lim) <= 0:
            continue

        ws0 = r.websocket_odd
        diff = None
        if ws0 is not None and float(ws0) > 0:
            diff = (float(odd) - float(ws0)) / float(ws0) * 100.0

        # filtro de qualidade (mesmo range do relatório)
        if diff is None or not (float(args.diff_min) <= float(diff) <= float(args.diff_max)):
            continue

        # edge Lay: entrada "melhor" tende a ter BS<WS (diff negativo)
        if float(diff) > float(args.lay_diff_max):
            continue

        mult = _ah_outcome_mult(line=r.line, side=r.side, home_score=r.home_score, away_score=r.away_score)
        roi_liab = _roi_lay_pct_per_liability(lay_odd=odd, mult_back=mult)
        if roi_liab is None:
            continue
        keep.append(
            Event(
                day_utc=r.audited_at_utc.date().isoformat(),
                limit_stake=float(lim),
                odd_entry=float(odd),
                roi_liab_pct=float(roi_liab),
                diff_entry_pct=float(diff),
            )
        )

    if not keep:
        print("[study] Sem eventos após filtros (verifique lookback/status/direction).")
        return 2

    caps = _parse_csv_floats(str(args.caps_liab or ""))
    if not caps:
        caps = [0.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0]

    # bins por limit (stake) — por padrão em decis
    lims = [x.limit_stake for x in keep if x.limit_stake > 0]
    bins = _quantile_bins(lims, ntiles=int(args.limit_ntiles))

    print("\n### Estudo rápido: Lay — ROI vs limit (proxy) (DB)\n")
    print(f"- Janela UTC: {since.isoformat()} → {now.isoformat()}")
    print(f"- Filtros: regime={args.regime} direction={args.direction or '—'} status={args.only_status or 'OK'} diff∈[{args.diff_min},{args.diff_max}] lay_diff_max={args.lay_diff_max}")
    print(f"- Eventos usados: {len(keep)} (com placar, odd e limit)\n")

    # curva global (efeito do cap)
    base_rows = [(e.limit_stake, e.odd_entry, e.roi_liab_pct) for e in keep]
    print("**Curva global (lucro total na janela; cap em *liability*)**")
    print("| cap_liab | lucro | ROI/liab (pnl / liab_usado) |")
    print("|---:|---:|---:|")
    cap_stats: List[Tuple[float, float, float]] = []  # (cap, pnl, used)
    for cap in caps:
        if float(cap) <= 0:
            # cap 0 => lucro 0
            print(f"| {_num(cap,2)} | {_num(0.0,2)} | — |")
            cap_stats.append((float(cap), 0.0, 0.0))
            continue
        pnl = _profit_for_cap(base_rows, float(cap))
        liab_used = _liab_used_for_cap(base_rows, float(cap))
        roi_w = (float(pnl) / float(liab_used) * 100.0) if liab_used > 0 else None
        print(f"| {_num(cap,2)} | {_num(pnl,2)} | {_pct(roi_w,2)} |")
        cap_stats.append((float(cap), float(pnl), float(liab_used)))
    print()

    # ROI marginal (incremental) por aumento do cap
    if len(cap_stats) >= 2:
        print("**ROI marginal por aumento do cap (incremental)**")
        print("| de cap | para cap | Δlucro | Δliab_usada | ROI_marg (Δpnl/Δliab) |")
        print("|---:|---:|---:|---:|---:|")
        for (c0, p0, u0), (c1, p1, u1) in zip(cap_stats, cap_stats[1:]):
            dp = float(p1) - float(p0)
            du = float(u1) - float(u0)
            r = (dp / du * 100.0) if du > 1e-12 else None
            print(f"| {_num(c0,2)} | {_num(c1,2)} | {_num(dp,2)} | {_num(du,2)} | {_pct(r,2)} |")
        print()

    # por bin de limit: média de ROI e contribuição marginal (cap alto pesa bins altos)
    if bins:
        # helper para mediana
        def _median(xs: List[float]) -> Optional[float]:
            if not xs:
                return None
            ys = sorted(xs)
            m = len(ys) // 2
            if len(ys) % 2 == 1:
                return float(ys[m])
            return 0.5 * (float(ys[m - 1]) + float(ys[m]))

        # 1) tabela principal por bin
        bin_stats: List[Dict[str, Any]] = []
        for i, (lo, hi) in enumerate(bins):
            sub = [t for t in keep if (t.limit_stake is not None and float(lo) <= float(t.limit_stake) <= float(hi))]
            if not sub:
                continue
            rois = [t.roi_liab_pct for t in sub]
            mean_roi = float(sum(rois) / len(rois)) if rois else None
            med_roi = _median(rois)
            rows3 = [(t.limit_stake, t.odd_entry, t.roi_liab_pct) for t in sub]
            p30 = _profit_for_cap(rows3, 30.0)
            p50 = _profit_for_cap(rows3, 50.0)
            p100 = _profit_for_cap(rows3, 100.0)
            u30 = _liab_used_for_cap(rows3, 30.0)
            u50 = _liab_used_for_cap(rows3, 50.0)
            u100 = _liab_used_for_cap(rows3, 100.0)
            roi30 = (p30 / u30 * 100.0) if u30 > 1e-12 else None
            roi50 = (p50 / u50 * 100.0) if u50 > 1e-12 else None
            roi100 = (p100 / u100 * 100.0) if u100 > 1e-12 else None
            r3050 = ((p50 - p30) / (u50 - u30) * 100.0) if (u50 - u30) > 1e-12 else None
            r50100 = ((p100 - p50) / (u100 - u50) * 100.0) if (u100 - u50) > 1e-12 else None
            bin_stats.append(
                {
                    "bin": int(i + 1),
                    "lo": float(lo),
                    "hi": float(hi),
                    "n": int(len(sub)),
                    "roi_mean": mean_roi,
                    "roi_p50": med_roi,
                    "p30": p30,
                    "p50": p50,
                    "p100": p100,
                    "roi30": roi30,
                    "roi50": roi50,
                    "roi100": roi100,
                    "r3050": r3050,
                    "r50100": r50100,
                }
            )

        print(f"**Por faixa de limit (stake) — ntiles={int(args.limit_ntiles)}**")
        print("| bin | limit_stake (lo-hi) | N | ROI/liab mean | ROI/liab p50 | ROI@cap30 | ROI@cap50 | ROI@cap100 | lucro@cap30 | lucro@cap50 | lucro@cap100 |")
        print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for s in bin_stats:
            print(
                f"| {s['bin']} | {_num(s['lo'],2)}–{_num(s['hi'],2)} | {s['n']} | {_pct(s.get('roi_mean'),2)} | {_pct(s.get('roi_p50'),2)} | "
                f"{_pct(s.get('roi30'),2)} | {_pct(s.get('roi50'),2)} | {_pct(s.get('roi100'),2)} | "
                f"{_num(s.get('p30'),2)} | {_num(s.get('p50'),2)} | {_num(s.get('p100'),2)} |"
            )
        print()

        # 2) marginais por bin (separado, sem poluir a tabela acima)
        print("**ROI marginal por bin (cap30→50 e cap50→100)**")
        print("| bin | limit_stake (lo-hi) | N | ROI_marg_30_50 | ROI_marg_50_100 |")
        print("|---:|---:|---:|---:|---:|")
        for s in bin_stats:
            print(
                f"| {s['bin']} | {_num(s['lo'],2)}–{_num(s['hi'],2)} | {s['n']} | {_pct(s.get('r3050'),2)} | {_pct(s.get('r50100'),2)} |"
            )
        print()

    print("Leitura recomendada:")
    print("- Se o lucro **cai** quando o cap aumenta, isso sugere que a massa de eventos que ainda não está saturada pelo `event_limit` (i.e., limit alto) tem **ROI médio pior/negativo**.")
    print("- Para confirmar (sem proxy), rode este estudo também com `entry_odd` mais fiel (pós-reversal) — hoje este script usa um modo rápido (betslip_odd).")

    if bool(getattr(args, "wf", False)):
        _run_walk_forward(keep, since=since, until=now, args=args)

    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Estudo rápido: Lay ROI vs limit (via DB), sem gerar PDF.")
    p.add_argument("--database-url", default=None, help="Override do DATABASE_URL (opcional).")
    p.add_argument("--lookback-days", type=float, default=float(os.getenv("STUDY_LOOKBACK_DAYS", "30")), help="Janela (dias) para puxar auditorias.")
    p.add_argument("--direction", default=os.getenv("STUDY_DIRECTION", "up"), help="reversal_direction (ex.: up). Use '' para não filtrar.")
    p.add_argument("--only-status", default=os.getenv("STUDY_ONLY_STATUS", "OK"), help="CSV de status (ex.: OK,GATE_NOT_ELIGIBLE). Default OK.")
    p.add_argument("--regime", choices=["in", "pre", "all"], default=os.getenv("STUDY_REGIME", "in"), help="Filtra Pre/In (via is_live/kickoff).")
    p.add_argument(
        "--entry-mode",
        choices=["betslip", "lay_policy"],
        default=os.getenv("STUDY_ENTRY_MODE", "betslip").strip() or "betslip",
        help="Como escolher odd de entrada do Lay. 'lay_policy' replica a regra do Sweep (pós-reversal / ~t+end_sec).",
    )
    p.add_argument("--wf-lay-end-sec", type=float, default=float(os.getenv("WF_LAY_END_SEC", "30.0")), help="(entry-mode=lay_policy) alvo t+end (segundos).")
    p.add_argument(
        "--wf-lay-end-max-gap-sec",
        type=float,
        default=float(os.getenv("WF_LAY_END_MAX_GAP_SEC", "12.0")),
        help="(entry-mode=lay_policy) tolerância máxima vs t+end (segundos).",
    )
    p.add_argument("--eps-rev", type=float, default=float(os.getenv("STUDY_EPS_REV", "0.5")), help="(entry-mode=lay_policy) threshold de reversão em p.p. de diff.")
    p.add_argument("--diff-min", type=float, default=float(os.getenv("STUDY_DIFF_MIN", "-10.0")), help="Filtro de qualidade diff (min).")
    p.add_argument("--diff-max", type=float, default=float(os.getenv("STUDY_DIFF_MAX", "10.0")), help="Filtro de qualidade diff (max).")
    p.add_argument("--lay-diff-max", type=float, default=float(os.getenv("STUDY_LAY_DIFF_MAX", "-2.0")), help="Corte de edge Lay (diff <= este valor).")
    p.add_argument("--caps-liab", default=os.getenv("STUDY_CAPS_LIAB", "0,10,20,30,50,75,100"), help="CSV de caps (liability) para a curva global.")
    p.add_argument("--limit-ntiles", type=int, default=int(os.getenv("STUDY_LIMIT_NTILES", "10")), help="Número de faixas (quantis) de limit para sumarizar.")
    p.add_argument("--wf", action="store_true", default=(os.getenv("STUDY_WF", "0").strip() in ("1", "true", "True", "yes", "YES")), help="Roda walk-forward (treino→teste) e compara política por bins vs baselines.")
    p.add_argument("--wf-train-days", type=int, default=int(os.getenv("STUDY_WF_TRAIN_DAYS", "14")), help="Dias de treino por step (WF).")
    p.add_argument("--wf-test-days", type=int, default=int(os.getenv("STUDY_WF_TEST_DAYS", "7")), help="Dias de teste por step (WF).")
    p.add_argument("--wf-step-days", type=int, default=int(os.getenv("STUDY_WF_STEP_DAYS", "7")), help="Avanço do step em dias (WF).")
    p.add_argument("--wf-limit-ntiles", type=int, default=int(os.getenv("STUDY_WF_LIMIT_NTILES", "0")), help="Número de bins de limit usados no treino (WF). 0 = usa --limit-ntiles.")
    p.add_argument("--wf-base-cap", type=float, default=float(os.getenv("STUDY_WF_BASE_CAP", "50")), help="Cap base (liability) usado como 'mínimo' por bin (WF).")
    p.add_argument("--wf-max-cap", type=float, default=float(os.getenv("STUDY_WF_MAX_CAP", "100")), help="Cap máximo (liability) permitido quando o bin passa no guardrail marginal (WF).")
    p.add_argument("--wf-min-roi-base-cap-pct", type=float, default=float(os.getenv("STUDY_WF_MIN_ROI_BASE_CAP_PCT", "0")), help="Guardrail: ROI/liab mínimo no bin @cap_base para manter o bin (WF).")
    p.add_argument("--wf-min-roi-marg-pct", type=float, default=float(os.getenv("STUDY_WF_MIN_ROI_MARG_PCT", "5")), help="Guardrail: ROI marginal mínimo (base→max) para permitir expandir cap no bin (WF).")
    p.add_argument("--wf-min-n-bin", type=int, default=int(os.getenv("STUDY_WF_MIN_N_BIN", "8")), help="Mínimo de eventos no treino por bin para decidir (senão mantém cap_base).")
    args = p.parse_args()
    if isinstance(args.direction, str) and not args.direction.strip():
        args.direction = None
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())

