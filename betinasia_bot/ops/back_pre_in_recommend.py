from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


def _load_json(path: Path) -> Dict[str, Any]:
    txt = path.read_text(encoding="utf-8")
    obj = json.loads(txt)
    if not isinstance(obj, dict):
        raise ValueError("JSON raiz inválido: esperado objeto")
    return obj


def _get_perm_p(perm_block: Dict[str, Any], key: str) -> Optional[float]:
    if not isinstance(perm_block, dict):
        return None
    x = perm_block.get(key)
    if not isinstance(x, dict):
        return None
    return _safe_float(x.get("p_value"))


def _is_strong_signal(
    *,
    p_lat: Optional[float],
    p_slip: Optional[float],
    p_combo: Optional[float],
    mean_delta: Optional[float],
    sign_p: Optional[float],
    days_eval: Optional[int],
    min_days_eval: int,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    ok_days = days_eval is not None and int(days_eval) >= int(min_days_eval)
    if not ok_days:
        reasons.append(f"dias OOS avaliados insuficientes ({days_eval} < {min_days_eval})")

    cond_perm = False
    if p_combo is not None and p_combo <= 0.10:
        cond_perm = True
        reasons.append(f"permutação combo significativa (p={p_combo:.4f})")
    else:
        aux = []
        if p_lat is not None and p_lat <= 0.10:
            aux.append(f"lat p={p_lat:.4f}")
        if p_slip is not None and p_slip <= 0.10:
            aux.append(f"slip p={p_slip:.4f}")
        if len(aux) >= 2:
            cond_perm = True
            reasons.append("permutação lat+slip significativa (" + ", ".join(aux) + ")")

    cond_oos = False
    if mean_delta is not None and sign_p is not None:
        if float(mean_delta) > 0 and float(sign_p) <= 0.10:
            cond_oos = True
            reasons.append(f"OOS positivo (mean_delta={mean_delta:.3f}pp, sign_p={sign_p:.4f})")
        else:
            reasons.append(f"OOS fraco/negativo (mean_delta={mean_delta}, sign_p={sign_p})")
    else:
        reasons.append("métrica OOS incompleta")

    strong = bool(ok_days and cond_perm and cond_oos)
    return strong, reasons


def _pick_top_combo(summary_regime: Dict[str, Any], *, min_obs: int) -> Optional[Dict[str, Any]]:
    if not isinstance(summary_regime, dict):
        return None
    combos = summary_regime.get("combo_buckets")
    if not isinstance(combos, list):
        return None
    best: Optional[Dict[str, Any]] = None
    for c in combos:
        if not isinstance(c, dict):
            continue
        n = _safe_int(c.get("n_orders")) or 0
        roi = _safe_float(c.get("roi_weighted_pct"))
        if n < int(min_obs) or roi is None:
            continue
        if best is None:
            best = c
            continue
        best_roi = _safe_float(best.get("roi_weighted_pct"))
        best_n = _safe_int(best.get("n_orders")) or 0
        if best_roi is None or float(roi) > float(best_roi) or (float(roi) == float(best_roi) and n > best_n):
            best = c
    return best


def _fmt(v: Optional[float], nd: int = 3) -> str:
    if v is None:
        return "n/a"
    return f"{float(v):.{nd}f}"


def _build_regime_recommendation(
    regime: str,
    *,
    summary_regime: Dict[str, Any],
    perm_regime: Dict[str, Any],
    oos_regime: Dict[str, Any],
    min_days_eval: int,
    min_combo_obs: int,
) -> Dict[str, Any]:
    p_lat = _get_perm_p(perm_regime, "lat_bucket")
    p_slip = _get_perm_p(perm_regime, "slip_bucket")
    p_combo = _get_perm_p(perm_regime, "lat_x_slip")
    mean_delta = _safe_float(oos_regime.get("mean_delta_roi_pct")) if isinstance(oos_regime, dict) else None
    sign_p = _safe_float(oos_regime.get("sign_test_p_value")) if isinstance(oos_regime, dict) else None
    days_eval = _safe_int(oos_regime.get("days_evaluated")) if isinstance(oos_regime, dict) else None

    strong, reasons = _is_strong_signal(
        p_lat=p_lat,
        p_slip=p_slip,
        p_combo=p_combo,
        mean_delta=mean_delta,
        sign_p=sign_p,
        days_eval=days_eval,
        min_days_eval=min_days_eval,
    )

    top_combo = _pick_top_combo(summary_regime, min_obs=min_combo_obs)
    top_lat = str(top_combo.get("lat_bucket")) if isinstance(top_combo, dict) else None
    top_slip = str(top_combo.get("slip_bucket")) if isinstance(top_combo, dict) else None
    top_n = _safe_int(top_combo.get("n_orders")) if isinstance(top_combo, dict) else None
    top_roi = _safe_float(top_combo.get("roi_weighted_pct")) if isinstance(top_combo, dict) else None

    action = "manter_sem_filtro"
    if strong and top_combo is not None:
        action = "aplicar_filtro_combo"
    elif not strong:
        action = "nao_aplicar_filtro_duro"

    cmd_hint = None
    if action == "aplicar_filtro_combo" and top_lat and top_slip:
        # Mapeamento simples para thresholds operacionais (bridge/risk policy).
        cmd_hint = {
            "lat_bucket": top_lat,
            "slip_bucket": top_slip,
            "suggestion": "converter bucket vencedor em gate operacional (risk_params/policy) e revalidar em OOS semanal",
        }

    base = summary_regime.get("base") if isinstance(summary_regime, dict) else {}
    return {
        "regime": regime,
        "base_n_orders": _safe_int((base or {}).get("n_orders")),
        "base_roi_weighted_pct": _safe_float((base or {}).get("roi_weighted_pct")),
        "signal_strength": "forte" if strong else "fraco",
        "decision": action,
        "evidence": {
            "perm_p_lat": p_lat,
            "perm_p_slip": p_slip,
            "perm_p_lat_x_slip": p_combo,
            "oos_days_evaluated": days_eval,
            "oos_mean_delta_roi_pct": mean_delta,
            "oos_sign_test_p": sign_p,
            "reasons": reasons,
        },
        "best_combo_candidate": (
            {
                "lat_bucket": top_lat,
                "slip_bucket": top_slip,
                "n_orders": top_n,
                "roi_weighted_pct": top_roi,
            }
            if top_combo is not None
            else None
        ),
        "operational_hint": cmd_hint,
    }


def _markdown_summary(report: Dict[str, Any], rec_pre: Dict[str, Any], rec_in: Dict[str, Any]) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    lines: List[str] = []
    lines.append(f"# Resumo executivo Back Pre/In ({ts})")
    lines.append("")
    lines.append("## Decisão")
    lines.append(
        f"- **Pre**: {rec_pre.get('decision')} | sinal: {rec_pre.get('signal_strength')} | ROI base: {_fmt(_safe_float(rec_pre.get('base_roi_weighted_pct')))}%"
    )
    lines.append(
        f"- **In**: {rec_in.get('decision')} | sinal: {rec_in.get('signal_strength')} | ROI base: {_fmt(_safe_float(rec_in.get('base_roi_weighted_pct')))}%"
    )
    lines.append("")
    lines.append("## Evidências-chave")
    for tag, rec in (("Pre", rec_pre), ("In", rec_in)):
        ev = rec.get("evidence") if isinstance(rec.get("evidence"), dict) else {}
        lines.append(f"### {tag}")
        lines.append(
            f"- perm(lat/slip/combo): {_fmt(_safe_float(ev.get('perm_p_lat')),4)} / {_fmt(_safe_float(ev.get('perm_p_slip')),4)} / {_fmt(_safe_float(ev.get('perm_p_lat_x_slip')),4)}"
        )
        lines.append(
            f"- OOS: days={ev.get('oos_days_evaluated')} | mean_delta={_fmt(_safe_float(ev.get('oos_mean_delta_roi_pct')))}pp | sign_p={_fmt(_safe_float(ev.get('oos_sign_test_p')),4)}"
        )
        bc = rec.get("best_combo_candidate") if isinstance(rec.get("best_combo_candidate"), dict) else None
        if bc:
            lines.append(
                f"- melhor combo candidato: lat={bc.get('lat_bucket')} × slip={bc.get('slip_bucket')} | n={bc.get('n_orders')} | ROIw={_fmt(_safe_float(bc.get('roi_weighted_pct')))}%"
            )
        else:
            lines.append("- melhor combo candidato: n/a")
    lines.append("")
    lines.append("## Próximo passo operacional")
    lines.append("- Se **Pre** forte: aplicar gate do combo vencedor somente em Pre.")
    lines.append("- Se **In** fraco: manter sem gate duro e seguir monitorando semanalmente.")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> Dict[str, Any]:
    src = Path(str(args.input_json))
    data = _load_json(src)

    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    perm = data.get("permutation_tests") if isinstance(data.get("permutation_tests"), dict) else {}
    oos = data.get("oos_expanding_combo") if isinstance(data.get("oos_expanding_combo"), dict) else {}

    rec_pre = _build_regime_recommendation(
        "pre",
        summary_regime=(summary.get("pre") if isinstance(summary, dict) else {}),
        perm_regime=(perm.get("pre") if isinstance(perm, dict) else {}),
        oos_regime=(oos.get("pre") if isinstance(oos, dict) else {}),
        min_days_eval=int(args.min_days_eval),
        min_combo_obs=int(args.min_combo_obs),
    )
    rec_in = _build_regime_recommendation(
        "in",
        summary_regime=(summary.get("in") if isinstance(summary, dict) else {}),
        perm_regime=(perm.get("in") if isinstance(perm, dict) else {}),
        oos_regime=(oos.get("in") if isinstance(oos, dict) else {}),
        min_days_eval=int(args.min_days_eval),
        min_combo_obs=int(args.min_combo_obs),
    )

    out = {
        "meta": {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "input_json": str(src),
            "min_days_eval": int(args.min_days_eval),
            "min_combo_obs": int(args.min_combo_obs),
        },
        "recommendations": {"pre": rec_pre, "in": rec_in},
    }

    md = _markdown_summary(out, rec_pre, rec_in)
    out["markdown_summary"] = md
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Resumo executivo de Back Pre/In com recomendação operacional a partir do JSON OOS/permutação."
    )
    ap.add_argument("--input-json", required=True, help="Saída do back_pre_in_oos_permutation.py")
    ap.add_argument("--out-json", default="", help="Opcional: caminho para salvar JSON de recomendação")
    ap.add_argument("--out-md", default="", help="Opcional: caminho para salvar resumo markdown")
    ap.add_argument("--min-days-eval", type=int, default=10)
    ap.add_argument("--min-combo-obs", type=int, default=40)
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
        p.write_text(str(rep.get("markdown_summary") or ""), encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
