import argparse
import asyncio
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import Integer, String, bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY

from storage.database import Database

# Helpers mínimos (evita importar o relatório inteiro; isso também reduz overhead)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        f = float(x)
        if f != f:  # NaN
            return None
        return f
    except Exception:
        return None


def _get_path(obj: Any, path: List[Any]) -> Any:
    cur = obj
    for p in path:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(p)
        elif isinstance(cur, list) and isinstance(p, int) and 0 <= p < len(cur):
            cur = cur[p]
        else:
            return None
    return cur


def _as_dict(x: Any) -> Optional[dict]:
    # `hypothesis_details` já vem como dict via SQLAlchemy, mas mantém fallback simples
    return x if isinstance(x, dict) else None


@dataclass(frozen=True)
class DayAgg:
    ok_total: int = 0
    ok_pre: int = 0
    ok_in: int = 0
    back_edge_pre: int = 0
    back_edge_in: int = 0
    lay_edge_pre: int = 0
    lay_edge_in: int = 0


def _day_utc(ts: Any) -> Optional[str]:
    if isinstance(ts, datetime):
        try:
            return ts.astimezone(timezone.utc).strftime("%Y-%m-%d")
        except Exception:
            return ts.strftime("%Y-%m-%d")
    return None


def _is_live_eff(*, audited_at: Optional[datetime], kickoff: Optional[datetime], is_live_flag: Any) -> bool:
    try:
        if audited_at is not None and kickoff is not None:
            return bool(audited_at >= kickoff)
    except Exception:
        pass
    try:
        if is_live_flag is True:
            return True
        if is_live_flag is False:
            return False
    except Exception:
        pass
    return False


def _extract_ws_series(d0: dict) -> List[dict]:
    h = d0.get("hypothesis_details") or {}
    ws0 = _safe_float(d0.get("ws_odd"))
    if ws0 is None or ws0 <= 0:
        return []
    series: List[dict] = []
    arr = _get_path(h, ["ws_series"])
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
            if float(t) <= 0.0005:
                continue
            series.append({"t": float(t), "odd": float(odd)})
    # fallback (alguns pipelines gravam série curta)
    if not series:
        arr2 = _get_path(h, ["ws_gate_series"])
        if isinstance(arr2, list) and arr2:
            for e in arr2:
                if not isinstance(e, dict):
                    continue
                t = _safe_float(e.get("t"))
                odd = _safe_float(e.get("ws_odd"))
                if t is None or odd is None or odd <= 0:
                    continue
                if float(t) <= 0.0005:
                    continue
                series.append({"t": float(t), "odd": float(odd)})
    # fallback (temporal)
    if not series:
        arr3 = _get_path(h, ["temporal"])
        if isinstance(arr3, list) and arr3:
            for e in arr3:
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
                series.append({"t": float(t), "odd": float(odd)})
    series.sort(key=lambda x: x["t"])
    return series


def _ws_proxy_odd(d0: dict, *, offset_s: float, max_gap_s: float) -> Tuple[Optional[float], Optional[float], str]:
    """
    Retorna (odd_proxy, t_proxy, reason). reason vazio quando ok.
    """
    pts = _extract_ws_series(d0)
    if not pts:
        return None, None, "NO_WS_SERIES"
    # preferir primeiro t>=offset
    cands = [(p["t"], p["odd"]) for p in pts if float(p["t"]) >= float(offset_s)]
    best = None
    if cands:
        t, o = cands[0]
        if abs(float(t) - float(offset_s)) <= float(max_gap_s):
            best = (float(o), float(t))
    if best is None:
        # ponto mais próximo
        t, o = min(((float(p["t"]), float(p["odd"])) for p in pts), key=lambda x: abs(float(x[0]) - float(offset_s)))
        if abs(float(t) - float(offset_s)) <= float(max_gap_s):
            best = (float(o), float(t))
    if best is None:
        return None, None, "WS_PROXY_GAP"
    return float(best[0]), float(best[1]), ""


def _lay_entry_odd_simple(d0: dict) -> Tuple[Optional[float], str]:
    """
    Diagnóstico rápido: tenta achar uma odd de Lay.
    Não replica a política completa pós-reversal; a ideia aqui é apenas medir cobertura/dados.
    """
    h = d0.get("hypothesis_details") or {}
    lay0 = _safe_float(_get_path(h, ["lay", "odd"]))
    if lay0 is not None:
        return float(lay0), ""
    arr = _get_path(h, ["lay_temporal"])
    if isinstance(arr, list) and arr:
        # pega o último ponto com lay_odd/ws_odd
        for e in reversed(arr):
            if not isinstance(e, dict):
                continue
            odd = _safe_float(e.get("lay_odd"))
            if odd is None:
                odd = _safe_float(e.get("ws_odd"))
            if odd is not None:
                return float(odd), "LAY_FROM_TEMPORAL"
    return None, "LAY_ODD_MISSING"


async def run() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--direction", default="up", choices=["up", "down"])
    p.add_argument("--versions", default="v4.0-api,v5.0-ws-only,v5.1-ws-gate-lay")
    p.add_argument("--lookback-days", type=int, default=14)
    p.add_argument("--database-url", default=None)
    p.add_argument("--back-diff-min", type=float, default=2.0)
    p.add_argument("--lay-diff-max", type=float, default=-2.0)
    p.add_argument("--ws-proxy-offset-sec", type=float, default=5.0)
    p.add_argument("--ws-proxy-max-gap-sec", type=float, default=2.5)
    args = p.parse_args()

    versions = [v.strip() for v in str(args.versions).split(",") if v.strip()]
    db = Database(database_url=args.database_url) if args.database_url else Database()
    await db.connect()
    try:
        # Query rápida (sem JOIN matches): para diagnóstico de cobertura/insumos, basta o audit.
        q = (
            text(
                """
                SELECT
                    a.id,
                    a.audited_at,
                    a.status,
                    a.is_live,
                    a.audit_version,
                    a.websocket_odd,
                    a.betslip_odd,
                    a.difference_pct,
                    a.hypothesis_details
                FROM betslip_audit_results a
                WHERE a.hypothesis_type = 'H3B'
                  AND a.reversal_direction = :direction
                  AND a.audit_version = ANY(:versions)
                  AND (
                    :lookback_days IS NULL
                    OR a.audited_at >= NOW() - make_interval(days => :lookback_days)
                  )
                """
            )
            .bindparams(bindparam("lookback_days", type_=Integer))
            # audit_version é texto (varchar) no banco
            .bindparams(bindparam("versions", type_=ARRAY(String), expanding=False))
        )
        # Observação: `versions` é text[] no DB; aqui passamos como list e confiamos no driver. Se falhar,
        # o usuário consegue passar 1 versão por vez.
        async with db.async_session() as session:
            res = await session.execute(q, {"direction": str(args.direction), "versions": versions, "lookback_days": int(args.lookback_days)})
            rows = list(res.fetchall())

        all_data: List[Dict[str, Any]] = []
        for rid, audited_at, status, is_live, ver, ws_odd, bs_odd, diff_pct, hyp in rows:
            all_data.append(
                {
                    "id": rid,
                    "audited_at": audited_at,
                    "status": status,
                    "is_live": is_live,
                    "version": ver,
                    "ws_odd": ws_odd,
                    "bs_odd": bs_odd,
                    "diff_pct": diff_pct,
                    "hypothesis_details": _as_dict(hyp) or {},
                }
            )

        day_agg: Dict[str, DayAgg] = {}
        # motivos: (day, side, regime) -> counter
        reasons: Dict[Tuple[str, str, str], Counter[str]] = defaultdict(Counter)

        for d0 in all_data:
            if str(d0.get("status", "")).upper() != "OK":
                continue
            day = _day_utc(d0.get("audited_at")) or "NA"
            audited_at = d0.get("audited_at") if isinstance(d0.get("audited_at"), datetime) else None
            # Sem JOIN matches, usamos `is_live` do audit como proxy (suficiente para o diagnóstico de colapso).
            is_live = bool(d0.get("is_live") is True)
            regime = "In" if is_live else "Pre"

            cur = day_agg.get(day, DayAgg())
            cur = DayAgg(
                ok_total=cur.ok_total + 1,
                ok_pre=cur.ok_pre + (0 if is_live else 1),
                ok_in=cur.ok_in + (1 if is_live else 0),
                back_edge_pre=cur.back_edge_pre,
                back_edge_in=cur.back_edge_in,
                lay_edge_pre=cur.lay_edge_pre,
                lay_edge_in=cur.lay_edge_in,
            )

            ws0 = _safe_float(d0.get("ws_odd"))
            if ws0 is None or ws0 <= 0:
                reasons[(day, "Back", regime)]["NO_WS0"] += 1
                reasons[(day, "Lay", regime)]["NO_WS0"] += 1
                day_agg[day] = cur
                continue

            # BACK: entrada = BS se existir; senão WS proxy
            bs = _safe_float(d0.get("bs_odd"))
            entry = None
            src = None
            if bs is not None and bs > 0:
                entry = float(bs)
                src = "BS"
            else:
                entry, t_proxy, src_reason = _ws_proxy_odd(d0, offset_s=float(args.ws_proxy_offset_sec), max_gap_s=float(args.ws_proxy_max_gap_sec))
                if entry is None:
                    reasons[(day, "Back", regime)][src_reason] += 1
                    day_agg[day] = cur
                    # segue para Lay
                    pass
                else:
                    src = "WS_PROXY"
            if entry is not None and src is not None:
                diff = (float(entry) - float(ws0)) / float(ws0) * 100.0
                if not (-10.0 <= float(diff) <= 10.0):
                    reasons[(day, "Back", regime)]["DIFF_OOR"] += 1
                elif float(diff) < float(args.back_diff_min):
                    reasons[(day, "Back", regime)]["DIFF_BELOW_CUT"] += 1
                else:
                    reasons[(day, "Back", regime)]["EDGE_OK"] += 1
                    if is_live:
                        cur = DayAgg(**{**cur.__dict__, "back_edge_in": cur.back_edge_in + 1})
                    else:
                        cur = DayAgg(**{**cur.__dict__, "back_edge_pre": cur.back_edge_pre + 1})
                # visibilidade: fonte de entrada
                reasons[(day, "Back", regime)][f"SRC_{src}"] += 1
                day_agg[day] = cur

            # LAY: diagnóstico simples (cobertura)
            lay_entry, lay_reason = _lay_entry_odd_simple(d0)
            if lay_entry is None:
                reasons[(day, "Lay", regime)][lay_reason] += 1
                day_agg[day] = cur
                continue
            if float(lay_entry) <= 1.0:
                reasons[(day, "Lay", regime)]["LAY_ODD_LE1"] += 1
                day_agg[day] = cur
                continue
            diff_lay = (float(lay_entry) - float(ws0)) / float(ws0) * 100.0
            if not (-10.0 <= float(diff_lay) <= 10.0):
                reasons[(day, "Lay", regime)]["DIFF_OOR"] += 1
            elif float(diff_lay) > float(args.lay_diff_max):
                reasons[(day, "Lay", regime)]["DIFF_ABOVE_CUT"] += 1
            else:
                reasons[(day, "Lay", regime)]["EDGE_OK"] += 1
                if is_live:
                    cur = DayAgg(**{**cur.__dict__, "lay_edge_in": cur.lay_edge_in + 1})
                else:
                    cur = DayAgg(**{**cur.__dict__, "lay_edge_pre": cur.lay_edge_pre + 1})
            day_agg[day] = cur

        days = sorted(day_agg.keys())
        print("\n### Diagnóstico rápido de INSUMOS para OOS (por dia; OK-only)\n")
        print("| Dia | OK total | OK Pre | OK In | Back edge Pre | Back edge In | Lay edge Pre | Lay edge In |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for day in days:
            a = day_agg[day]
            print(f"| {day} | {a.ok_total} | {a.ok_pre} | {a.ok_in} | {a.back_edge_pre} | {a.back_edge_in} | {a.lay_edge_pre} | {a.lay_edge_in} |")

        def _print_reasons(side: str):
            print(f"\n### Motivos (top) — {side}\n")
            print("| Dia | Regime | Top motivos |")
            print("|---|---|---|")
            for day in days:
                for regime in ("Pre", "In"):
                    c = reasons.get((day, side, regime), Counter())
                    top = ", ".join([f"{k}×{v}" for k, v in c.most_common(5)]) if c else "—"
                    print(f"| {day} | {regime} | {top} |")

        _print_reasons("Back")
        _print_reasons("Lay")
        return 0
    finally:
        await db.close()


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()

