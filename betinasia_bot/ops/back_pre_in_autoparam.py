from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


@dataclass
class BucketScheme:
    name: str
    lat_edges_ms: List[float]  # n bins: [-inf,e1], (e1,e2], ..., (ek,+inf)
    slip_edges_pct: List[float]
    include_unknown: bool


def _lat_bucket(v: Optional[float], edges: Sequence[float], include_unknown: bool) -> Optional[str]:
    if v is None:
        return "unknown" if include_unknown else None
    x = float(v)
    if not edges:
        return "all"
    prev = None
    for e in edges:
        if x <= float(e):
            if prev is None:
                return f"<= {float(e):g}ms"
            return f"({float(prev):g},{float(e):g}]ms"
        prev = float(e)
    return f"> {float(edges[-1]):g}ms"


def _slip_bucket(v: Optional[float], edges: Sequence[float], include_unknown: bool) -> Optional[str]:
    if v is None:
        return "unknown" if include_unknown else None
    x = float(v)
    if not edges:
        return "all"
    prev = None
    for e in edges:
        if x <= float(e):
            if prev is None:
                return f"<= {float(e):g}%"
            return f"({float(prev):g},{float(e):g}]%"
        prev = float(e)
    return f"> {float(edges[-1]):g}%"


def _weighted_roi(rows: Sequence[Dict[str, Any]]) -> Optional[float]:
    exp = float(sum(float(r.get("exposure") or 0.0) for r in rows))
    if exp <= 0:
        return None
    pnl = float(sum(float(r.get("pnl") or 0.0) for r in rows))
    return float(pnl / exp * 100.0)


def _build_rows_from_source(data: Dict[str, Any], regime: str) -> List[Dict[str, Any]]:
    # Tenta localizar fonte de linhas detalhadas se existir no futuro.
    # Neste momento usamos a malha de combo_buckets já agregada do script robusto.
    # Para autoparam real, exigimos arquivo com observações detalhadas.
    detailed = data.get("observations")
    if isinstance(detailed, list):
        out = []
        rg = str(regime).strip().lower()
        for r in detailed:
            if not isinstance(r, dict):
                continue
            if str(r.get("regime") or "").strip().lower() != rg:
                continue
            out.append(r)
        return out
    return []


def _score_candidate(
    *,
    perm_combo_p: Optional[float],
    oos_mean_delta: Optional[float],
    oos_sign_p: Optional[float],
    oos_days: Optional[int],
    min_days_eval: int,
) -> float:
    score = 0.0
    if perm_combo_p is not None:
        score += max(0.0, 1.0 - min(1.0, float(perm_combo_p))) * 40.0
    if oos_mean_delta is not None:
        score += max(-2.0, min(2.0, float(oos_mean_delta))) * 15.0
    if oos_sign_p is not None:
        score += max(0.0, 1.0 - min(1.0, float(oos_sign_p))) * 30.0
    if oos_days is not None and int(oos_days) >= int(min_days_eval):
        score += 15.0
    return float(score)


def _rank_from_precomputed(
    data: Dict[str, Any],
    *,
    min_days_eval: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    perm = data.get("permutation_tests") if isinstance(data.get("permutation_tests"), dict) else {}
    oos = data.get("oos_expanding_combo") if isinstance(data.get("oos_expanding_combo"), dict) else {}
    summ = data.get("summary") if isinstance(data.get("summary"), dict) else {}

    for regime in ("pre", "in"):
        pblk = perm.get(regime) if isinstance(perm, dict) else {}
        oblk = oos.get(regime) if isinstance(oos, dict) else {}
        sblk = summ.get(regime) if isinstance(summ, dict) else {}
        p_combo = _safe_float(((pblk or {}).get("lat_x_slip") or {}).get("p_value")) if isinstance(pblk, dict) else None
        mdelta = _safe_float((oblk or {}).get("mean_delta_roi_pct")) if isinstance(oblk, dict) else None
        sp = _safe_float((oblk or {}).get("sign_test_p_value")) if isinstance(oblk, dict) else None
        ndays = _safe_int((oblk or {}).get("days_evaluated")) if isinstance(oblk, dict) else None
        score = _score_candidate(
            perm_combo_p=p_combo,
            oos_mean_delta=mdelta,
            oos_sign_p=sp,
            oos_days=ndays,
            min_days_eval=min_days_eval,
        )
        # melhor combo observado no summary atual
        best_combo = None
        combos = (sblk or {}).get("combo_buckets") if isinstance(sblk, dict) else None
        if isinstance(combos, list):
            for c in combos:
                if not isinstance(c, dict):
                    continue
                roi = _safe_float(c.get("roi_weighted_pct"))
                n = _safe_int(c.get("n_orders")) or 0
                if roi is None:
                    continue
                if best_combo is None:
                    best_combo = c
                    continue
                broi = _safe_float(best_combo.get("roi_weighted_pct"))
                bn = _safe_int(best_combo.get("n_orders")) or 0
                if broi is None or float(roi) > float(broi) or (float(roi) == float(broi) and int(n) > int(bn)):
                    best_combo = c

        out[regime] = {
            "score": score,
            "perm_combo_p": p_combo,
            "oos_mean_delta_roi_pct": mdelta,
            "oos_sign_test_p": sp,
            "oos_days_evaluated": ndays,
            "best_combo_current": (
                {
                    "lat_bucket": best_combo.get("lat_bucket"),
                    "slip_bucket": best_combo.get("slip_bucket"),
                    "n_orders": best_combo.get("n_orders"),
                    "roi_weighted_pct": best_combo.get("roi_weighted_pct"),
                }
                if isinstance(best_combo, dict)
                else None
            ),
        }
    return out


def _build_schemes() -> List[BucketScheme]:
    return [
        BucketScheme("coarse", lat_edges_ms=[5000, 10000, 20000], slip_edges_pct=[-2, 0, 2], include_unknown=False),
        BucketScheme("medium", lat_edges_ms=[3000, 6000, 10000, 20000], slip_edges_pct=[-2, -0.5, 0.5, 2], include_unknown=False),
        BucketScheme("fine", lat_edges_ms=[2000, 4000, 6000, 10000, 15000, 25000], slip_edges_pct=[-3, -2, -1, 0, 1, 2, 3], include_unknown=False),
        BucketScheme("medium+unknown", lat_edges_ms=[3000, 6000, 10000, 20000], slip_edges_pct=[-2, -0.5, 0.5, 2], include_unknown=True),
    ]


def run(args: argparse.Namespace) -> Dict[str, Any]:
    src = Path(str(args.input_json))
    data = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("JSON de entrada inválido")

    # Ranking base (com os testes já calculados no pipeline robusto).
    ranked = _rank_from_precomputed(data, min_days_eval=int(args.min_days_eval))

    # Se houver observações detalhadas no futuro, podemos realmente recomputar por esquema.
    # Por enquanto, registramos os esquemas candidatos para operacionalização.
    schemes = _build_schemes()
    regime_recos: Dict[str, Any] = {}
    for regime in ("pre", "in"):
        rr = ranked.get(regime) if isinstance(ranked, dict) else {}
        score = _safe_float((rr or {}).get("score")) if isinstance(rr, dict) else None
        best = (rr or {}).get("best_combo_current") if isinstance(rr, dict) else None
        if score is None:
            level = "insuficiente"
        elif score >= 55:
            level = "forte"
        elif score >= 35:
            level = "moderada"
        else:
            level = "fraca"

        decision = "nao_aplicar_filtro_duro"
        if level == "forte" and isinstance(best, dict):
            decision = "aplicar_combo_vencedor"
        elif level == "moderada":
            decision = "aplicar_shadow_monitoring"

        regime_recos[regime] = {
            "robustness_level": level,
            "decision": decision,
            "score": score,
            "best_combo_current": best if isinstance(best, dict) else None,
            "candidate_schemes": [
                {
                    "name": s.name,
                    "lat_edges_ms": s.lat_edges_ms,
                    "slip_edges_pct": s.slip_edges_pct,
                    "include_unknown": s.include_unknown,
                }
                for s in schemes
            ],
            "evidence": {
                "perm_combo_p": (rr or {}).get("perm_combo_p") if isinstance(rr, dict) else None,
                "oos_mean_delta_roi_pct": (rr or {}).get("oos_mean_delta_roi_pct") if isinstance(rr, dict) else None,
                "oos_sign_test_p": (rr or {}).get("oos_sign_test_p") if isinstance(rr, dict) else None,
                "oos_days_evaluated": (rr or {}).get("oos_days_evaluated") if isinstance(rr, dict) else None,
            },
        }

    out = {
        "meta": {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "input_json": str(src),
            "min_days_eval": int(args.min_days_eval),
            "mode": "autoparam_rank",
            "note": "Ranking com base em evidência OOS/permutação já calculada; esquemas listados para iteração operacional.",
        },
        "autoparam": regime_recos,
    }
    return out


def _to_markdown(rep: Dict[str, Any]) -> str:
    auto = rep.get("autoparam") if isinstance(rep.get("autoparam"), dict) else {}
    ts = (rep.get("meta") or {}).get("ts_utc") if isinstance(rep.get("meta"), dict) else None
    lines = [f"# Back Pre/In Autoparam ({ts})", ""]
    for rg in ("pre", "in"):
        blk = auto.get(rg) if isinstance(auto, dict) else {}
        lines.append(f"## {rg.upper()}")
        lines.append(f"- robustez: **{blk.get('robustness_level')}**")
        lines.append(f"- decisão: **{blk.get('decision')}**")
        lines.append(f"- score: {blk.get('score')}")
        ev = blk.get("evidence") if isinstance(blk.get("evidence"), dict) else {}
        lines.append(
            "- evidência: "
            f"perm_combo_p={ev.get('perm_combo_p')}, "
            f"oos_mean_delta={ev.get('oos_mean_delta_roi_pct')}, "
            f"oos_sign_p={ev.get('oos_sign_test_p')}, "
            f"oos_days={ev.get('oos_days_evaluated')}"
        )
        bc = blk.get("best_combo_current") if isinstance(blk.get("best_combo_current"), dict) else None
        if bc:
            lines.append(
                f"- combo atual: lat={bc.get('lat_bucket')} × slip={bc.get('slip_bucket')} "
                f"(n={bc.get('n_orders')}, ROIw={bc.get('roi_weighted_pct')})"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Autoparam Back Pre/In: rankeia robustez e sugere esquemas de buckets lat/slip."
    )
    ap.add_argument("--input-json", required=True, help="Saída do back_pre_in_oos_permutation.py")
    ap.add_argument("--out-json", default="", help="Opcional: salvar JSON de autoparam")
    ap.add_argument("--out-md", default="", help="Opcional: salvar markdown de autoparam")
    ap.add_argument("--min-days-eval", type=int, default=10)
    args = ap.parse_args()

    rep = run(args)
    txt = json.dumps(rep, ensure_ascii=False, indent=2)
    if str(args.out_json or "").strip():
        p = Path(str(args.out_json))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, encoding="utf-8")
    if str(args.out_md or "").strip():
        p = Path(str(args.out_md))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_to_markdown(rep), encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
