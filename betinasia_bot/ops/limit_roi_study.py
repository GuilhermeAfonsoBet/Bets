from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import bindparam, text

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

    def entry_odd(self) -> Optional[float]:
        # Modo rápido: assume entrada ~ betslip_odd (quando existe). Fallbacks ajudam a não "matar" a amostra.
        for v in (self.betslip_odd, self.lay_odd_hint, self.websocket_odd):
            if v is not None and float(v) > 1e-9:
                return float(v)
        return None

    def limit_stake(self) -> Optional[float]:
        # Para Lay, preferimos `available_limit` (stake max); senão usa betslip_limit.
        for v in (self.lay_available_limit, self.betslip_limit):
            if v is not None and float(v) > 0:
                return float(v)
        return None


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
          AND (:direction IS NULL OR a.reversal_direction = :direction)
        """
    ).bindparams(
        bindparam("since_utc"),
        bindparam("until_utc"),
        bindparam("direction"),
        bindparam("only_status"),
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
    keep: List[Tuple[float, float, float, float]] = []
    # (limit_stake, odd_entry, roi_liab_pct, diff_entry_pct)
    for r in rows:
        if args.regime != "all":
            want_live = True if args.regime == "in" else False
            if r.is_live_eff is None:
                continue
            if bool(r.is_live_eff) is not bool(want_live):
                continue

        odd = r.entry_odd()
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
        keep.append((float(lim), float(odd), float(roi_liab), float(diff)))

    if not keep:
        print("[study] Sem eventos após filtros (verifique lookback/status/direction).")
        return 2

    caps = _parse_csv_floats(str(args.caps_liab or ""))
    if not caps:
        caps = [0.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0]

    # bins por limit (stake) — por padrão em decis
    lims = [x[0] for x in keep if x[0] > 0]
    bins = _quantile_bins(lims, ntiles=int(args.limit_ntiles))

    print("\n### Estudo rápido: Lay — ROI vs limit (proxy) (DB)\n")
    print(f"- Janela UTC: {since.isoformat()} → {now.isoformat()}")
    print(f"- Filtros: regime={args.regime} direction={args.direction or '—'} status={args.only_status or 'OK'} diff∈[{args.diff_min},{args.diff_max}] lay_diff_max={args.lay_diff_max}")
    print(f"- Eventos usados: {len(keep)} (com placar, odd e limit)\n")

    # curva global (efeito do cap)
    base_rows = [(lim, odd, roi) for (lim, odd, roi, _diff) in keep]
    print("**Curva global (lucro total na janela; cap em *liability*)**")
    print("| cap_liab | lucro | ROI/liab (pnl / liab_usado) |")
    print("|---:|---:|---:|")
    for cap in caps:
        if float(cap) <= 0:
            # cap 0 => lucro 0
            print(f"| {_num(cap,2)} | {_num(0.0,2)} | — |")
            continue
        pnl = _profit_for_cap(base_rows, float(cap))
        liab_used = 0.0
        for lim, odd, roi in base_rows:
            liab_lim = float(lim) * max(0.0, float(odd) - 1.0)
            liab_used += min(float(cap), float(liab_lim))
        roi_w = (float(pnl) / float(liab_used) * 100.0) if liab_used > 0 else None
        print(f"| {_num(cap,2)} | {_num(pnl,2)} | {_pct(roi_w,2)} |")
    print()

    # por bin de limit: média de ROI e contribuição marginal (cap alto pesa bins altos)
    if bins:
        print(f"**Por faixa de limit (stake) — ntiles={int(args.limit_ntiles)}**")
        print("| bin | limit_stake (lo-hi) | N | ROI/liab mean | ROI/liab p50 | lucro@cap30 | lucro@cap50 | lucro@cap100 |")
        print("|---:|---:|---:|---:|---:|---:|---:|---:|")

        # helper para mediana
        def _median(xs: List[float]) -> Optional[float]:
            if not xs:
                return None
            ys = sorted(xs)
            m = len(ys) // 2
            if len(ys) % 2 == 1:
                return float(ys[m])
            return 0.5 * (float(ys[m - 1]) + float(ys[m]))

        for i, (lo, hi) in enumerate(bins):
            sub = [t for t in keep if (t[0] is not None and float(lo) <= float(t[0]) <= float(hi))]
            rois = [t[2] for t in sub]
            if not sub:
                continue
            mean_roi = float(sum(rois) / len(rois)) if rois else None
            med_roi = _median(rois)
            rows3 = [(t[0], t[1], t[2]) for t in sub]
            p30 = _profit_for_cap(rows3, 30.0)
            p50 = _profit_for_cap(rows3, 50.0)
            p100 = _profit_for_cap(rows3, 100.0)
            print(
                f"| {i+1} | {_num(lo,2)}–{_num(hi,2)} | {len(sub)} | {_pct(mean_roi,2)} | {_pct(med_roi,2)} | {_num(p30,2)} | {_num(p50,2)} | {_num(p100,2)} |"
            )
        print()

    print("Leitura recomendada:")
    print("- Se o lucro **cai** quando o cap aumenta, isso sugere que a massa de eventos que ainda não está saturada pelo `event_limit` (i.e., limit alto) tem **ROI médio pior/negativo**.")
    print("- Para confirmar (sem proxy), rode este estudo também com `entry_odd` mais fiel (pós-reversal) — hoje este script usa um modo rápido (betslip_odd).")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Estudo rápido: Lay ROI vs limit (via DB), sem gerar PDF.")
    p.add_argument("--database-url", default=None, help="Override do DATABASE_URL (opcional).")
    p.add_argument("--lookback-days", type=float, default=float(os.getenv("STUDY_LOOKBACK_DAYS", "30")), help="Janela (dias) para puxar auditorias.")
    p.add_argument("--direction", default=os.getenv("STUDY_DIRECTION", "up"), help="reversal_direction (ex.: up). Use '' para não filtrar.")
    p.add_argument("--only-status", default=os.getenv("STUDY_ONLY_STATUS", "OK"), help="CSV de status (ex.: OK,GATE_NOT_ELIGIBLE). Default OK.")
    p.add_argument("--regime", choices=["in", "pre", "all"], default=os.getenv("STUDY_REGIME", "in"), help="Filtra Pre/In (via is_live/kickoff).")
    p.add_argument("--diff-min", type=float, default=float(os.getenv("STUDY_DIFF_MIN", "-10.0")), help="Filtro de qualidade diff (min).")
    p.add_argument("--diff-max", type=float, default=float(os.getenv("STUDY_DIFF_MAX", "10.0")), help="Filtro de qualidade diff (max).")
    p.add_argument("--lay-diff-max", type=float, default=float(os.getenv("STUDY_LAY_DIFF_MAX", "-2.0")), help="Corte de edge Lay (diff <= este valor).")
    p.add_argument("--caps-liab", default=os.getenv("STUDY_CAPS_LIAB", "0,10,20,30,50,75,100"), help="CSV de caps (liability) para a curva global.")
    p.add_argument("--limit-ntiles", type=int, default=int(os.getenv("STUDY_LIMIT_NTILES", "10")), help="Número de faixas (quantis) de limit para sumarizar.")
    args = p.parse_args()
    if isinstance(args.direction, str) and not args.direction.strip():
        args.direction = None
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())

