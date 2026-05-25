from __future__ import annotations

import argparse
import json
import os
import subprocess
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


def _fmt_num(x: Any, nd: int = 2) -> str:
    v = _safe_float(x)
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}"


def _fmt_pct(x: Any, nd: int = 2) -> str:
    v = _safe_float(x)
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}%"


def _fmt_ci(ci_obj: Any, nd: int = 2) -> str:
    if not isinstance(ci_obj, dict):
        return "n/a"
    lb = _safe_float(ci_obj.get("lb"))
    ub = _safe_float(ci_obj.get("ub"))
    if lb is None or ub is None:
        return "n/a"
    return f"[{lb:.{nd}f}, {ub:.{nd}f}]"


def _load_json(path: Path) -> Dict[str, Any]:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    obj = json.loads(txt or "{}")
    return obj if isinstance(obj, dict) else {}


def _pick_combo_tests(data: Dict[str, Any], regime: str) -> List[Dict[str, Any]]:
    pt = data.get("permutation_tests") if isinstance(data.get("permutation_tests"), dict) else {}
    rg = pt.get(regime) if isinstance(pt.get(regime), dict) else {}
    cvs = rg.get("combo_vs_rest_tests") if isinstance(rg.get("combo_vs_rest_tests"), dict) else {}
    tests = cvs.get("tests") if isinstance(cvs.get("tests"), list) else []
    out: List[Dict[str, Any]] = []
    for t in tests:
        if isinstance(t, dict):
            out.append(t)
    return out


def _pick_combos_summary(data: Dict[str, Any], regime: str) -> List[Dict[str, Any]]:
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    rg = summary.get(regime) if isinstance(summary.get(regime), dict) else {}
    combos = rg.get("combo_buckets") if isinstance(rg.get("combo_buckets"), list) else []
    out: List[Dict[str, Any]] = []
    for c in combos:
        if isinstance(c, dict):
            out.append(c)
    return out


def _build_combo_rows(data: Dict[str, Any], regime: str) -> List[Dict[str, Any]]:
    combos = _pick_combos_summary(data, regime)
    tests = _pick_combo_tests(data, regime)
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for t in tests:
        k = (str(t.get("lat_bucket") or ""), str(t.get("slip_bucket") or ""))
        by_key[k] = t

    rows: List[Dict[str, Any]] = []
    for c in combos:
        lat = str(c.get("lat_bucket") or "")
        slip = str(c.get("slip_bucket") or "")
        test = by_key.get((lat, slip), {})
        rows.append(
            {
                "lat_bucket": lat,
                "slip_bucket": slip,
                "n_orders": int(c.get("n_orders") or 0),
                "roi_weighted_pct": _safe_float(c.get("roi_weighted_pct")),
                "roi_ci95_bootstrap": c.get("roi_ci95_bootstrap"),
                "delta_roi_pct": _safe_float(test.get("delta_roi_pct")),
                "delta_roi_ci95_bootstrap": test.get("delta_roi_ci95_bootstrap"),
                "p_value": _safe_float(test.get("p_value")),
                "q_value_bh": _safe_float(test.get("q_value_bh")),
                "significant_fdr": bool(test.get("significant_fdr")) if isinstance(test, dict) and ("significant_fdr" in test) else False,
            }
        )

    rows.sort(
        key=lambda r: (
            0 if (r.get("q_value_bh") is not None) else 1,
            1.0 if r.get("q_value_bh") is None else float(r.get("q_value_bh")),
            1.0 if r.get("p_value") is None else float(r.get("p_value")),
            -abs(float(r.get("delta_roi_pct") or 0.0)),
        )
    )
    return rows


def _choose_top_positive(rows: List[Dict[str, Any]], n_top: int) -> List[Dict[str, Any]]:
    cand = [r for r in rows if (_safe_float(r.get("delta_roi_pct")) or 0.0) > 0 and int(r.get("n_orders") or 0) >= 80]
    cand.sort(
        key=lambda r: (
            1.0 if r.get("p_value") is None else float(r.get("p_value")),
            -(float(r.get("delta_roi_pct") or 0.0)),
            -(int(r.get("n_orders") or 0)),
        )
    )
    return cand[: max(0, int(n_top))]


def _base_block(data: Dict[str, Any], regime: str) -> Dict[str, Any]:
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    rg = summary.get(regime) if isinstance(summary.get(regime), dict) else {}
    base = rg.get("base") if isinstance(rg.get("base"), dict) else {}
    return base


def _perm_block(data: Dict[str, Any], regime: str) -> Dict[str, Any]:
    perm = data.get("permutation_tests") if isinstance(data.get("permutation_tests"), dict) else {}
    return perm.get(regime) if isinstance(perm.get(regime), dict) else {}


def _oos_block(data: Dict[str, Any], regime: str) -> Dict[str, Any]:
    oos = data.get("oos_expanding_combo") if isinstance(data.get("oos_expanding_combo"), dict) else {}
    return oos.get(regime) if isinstance(oos.get(regime), dict) else {}


def _conclusion_md(data: Dict[str, Any]) -> str:
    base_all = _base_block(data, "all")
    base_pre = _base_block(data, "pre")
    base_in = _base_block(data, "in")
    oos_pre = _oos_block(data, "pre")
    oos_in = _oos_block(data, "in")
    oos_all = _oos_block(data, "all")

    rows_all = _build_combo_rows(data, "all")
    rows_pre = _build_combo_rows(data, "pre")
    rows_in = _build_combo_rows(data, "in")

    top_pos_in = _choose_top_positive(rows_in, 5)
    top_pos_pre = _choose_top_positive(rows_pre, 5)

    sig_neg_in = [
        r
        for r in rows_in
        if bool(r.get("significant_fdr")) and (_safe_float(r.get("delta_roi_pct")) or 0.0) < 0
    ]

    lines: List[str] = []
    lines.append("## Conclusão robusta (executiva)\n")
    lines.append("- O sistema Back no período analisado tem expectativa negativa em **ALL/IN**, com IC95 do ROI base abaixo de zero.")
    lines.append(
        f"  - ALL: ROI {_fmt_pct(base_all.get('roi_weighted_pct'),3)} | IC95 {_fmt_ci(base_all.get('roi_ci95_bootstrap'),3)}"
    )
    lines.append(
        f"  - IN: ROI {_fmt_pct(base_in.get('roi_weighted_pct'),3)} | IC95 {_fmt_ci(base_in.get('roi_ci95_bootstrap'),3)}"
    )
    lines.append(
        f"  - PRE: ROI {_fmt_pct(base_pre.get('roi_weighted_pct'),3)} | IC95 {_fmt_ci(base_pre.get('roi_ci95_bootstrap'),3)}"
    )
    lines.append("- Não há evidência robusta de combo positivo estável após correção de múltiplos testes (BH/FDR), exceto sinal negativo robusto em IN.")
    if sig_neg_in:
        b = sig_neg_in[0]
        lines.append(
            f"  - Sinal negativo robusto IN: `{b['lat_bucket']} × {b['slip_bucket']}` | delta={_fmt_pct(b.get('delta_roi_pct'),2)} | "
            f"IC95={_fmt_ci(b.get('delta_roi_ci95_bootstrap'),2)} | q={_fmt_num(b.get('q_value_bh'),4)}"
        )
    lines.append("- OOS ainda curto para decisão ofensiva (seleção de combos positivos):")
    lines.append(
        f"  - ALL: days_evaluated={int(oos_all.get('days_evaluated') or 0)} | mean_delta={_fmt_num(oos_all.get('mean_delta_roi_pct'),3)} pp | sign_p={_fmt_num(oos_all.get('sign_test_p_value'),4)}"
    )
    lines.append(
        f"  - IN: days_evaluated={int(oos_in.get('days_evaluated') or 0)} | mean_delta={_fmt_num(oos_in.get('mean_delta_roi_pct'),3)} pp | sign_p={_fmt_num(oos_in.get('sign_test_p_value'),4)}"
    )
    lines.append(
        f"  - PRE: days_evaluated={int(oos_pre.get('days_evaluated') or 0)} | mean_delta={_fmt_num(oos_pre.get('mean_delta_roi_pct'),3)} pp | sign_p={_fmt_num(oos_pre.get('sign_test_p_value'),4)}"
    )
    lines.append("\n### Combos positivos (critério relaxado, exploratório)\n")
    lines.append("- Critério: delta>0, n>=80, priorização por menor p-value.")

    if top_pos_in:
        lines.append("- **IN (top candidatos):**")
        for r in top_pos_in:
            lines.append(
                f"  - `{r['lat_bucket']} × {r['slip_bucket']}` | n={r['n_orders']} | ROI={_fmt_pct(r.get('roi_weighted_pct'),2)} | "
                f"delta={_fmt_pct(r.get('delta_roi_pct'),2)} | IC95(delta)={_fmt_ci(r.get('delta_roi_ci95_bootstrap'),2)} | "
                f"p={_fmt_num(r.get('p_value'),4)} | q={_fmt_num(r.get('q_value_bh'),4)}"
            )
    else:
        lines.append("- **IN:** sem candidatos positivos pelo critério relaxado.")

    if top_pos_pre:
        lines.append("- **PRE (top candidatos):**")
        for r in top_pos_pre:
            lines.append(
                f"  - `{r['lat_bucket']} × {r['slip_bucket']}` | n={r['n_orders']} | ROI={_fmt_pct(r.get('roi_weighted_pct'),2)} | "
                f"delta={_fmt_pct(r.get('delta_roi_pct'),2)} | IC95(delta)={_fmt_ci(r.get('delta_roi_ci95_bootstrap'),2)} | "
                f"p={_fmt_num(r.get('p_value'),4)} | q={_fmt_num(r.get('q_value_bh'),4)}"
            )
    else:
        lines.append("- **PRE:** sem candidatos positivos robustos (amostra curta e OOS insuficiente).")

    lines.append("\n### Recomendação operacional\n")
    lines.append("- Não aplicar gate positivo duro neste estágio.")
    lines.append("- Aplicar hardening defensivo em combos com sinal negativo robusto (IN).")
    lines.append("- Reavaliar com janela maior (60-90 dias) e alvo de >=20 dias OOS avaliados.")
    lines.append("")
    return "\n".join(lines)


def _combo_table_md(regime: str, rows: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append(f"## Tabela completa de combinações — {regime.upper()}\n")
    lines.append(
        "| lat_bucket | slip_bucket | n_orders | roi_weighted_pct | roi_ci95_bootstrap | delta_vs_rest_pct | delta_ci95_bootstrap | p_value | q_value_bh | significant_fdr |"
    )
    lines.append(
        "|---|---:|---:|---:|---|---:|---|---:|---:|---:|"
    )
    for r in rows:
        lines.append(
            "| "
            + f"{r.get('lat_bucket') or ''} | "
            + f"{r.get('slip_bucket') or ''} | "
            + f"{int(r.get('n_orders') or 0)} | "
            + f"{_fmt_num(r.get('roi_weighted_pct'),4)} | "
            + f"{_fmt_ci(r.get('roi_ci95_bootstrap'),4)} | "
            + f"{_fmt_num(r.get('delta_roi_pct'),4)} | "
            + f"{_fmt_ci(r.get('delta_roi_ci95_bootstrap'),4)} | "
            + f"{_fmt_num(r.get('p_value'),4)} | "
            + f"{_fmt_num(r.get('q_value_bh'),4)} | "
            + f"{'true' if bool(r.get('significant_fdr')) else 'false'} |"
        )
    lines.append("")
    return "\n".join(lines)


def _build_markdown(data: Dict[str, Any], src_json: Path) -> str:
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    coverage = data.get("coverage") if isinstance(data.get("coverage"), dict) else {}
    ts = datetime.now(timezone.utc).isoformat()

    rows_all = _build_combo_rows(data, "all")
    rows_pre = _build_combo_rows(data, "pre")
    rows_in = _build_combo_rows(data, "in")

    parts: List[str] = []
    parts.append(f"# Relatório robusto Back Pre/In — Latência × Slippage ({ts})\n")
    parts.append("## Contexto e cobertura\n")
    parts.append(f"- Fonte JSON: `{src_json}`")
    parts.append(f"- PnL source: `{meta.get('pnl_source')}`")
    parts.append(f"- Intervalo: `{meta.get('start_day')}` até `{meta.get('end_day')}`")
    parts.append(f"- Observações finais: `{coverage.get('final_observations')}`")
    parts.append(
        f"- Regimes: ALL=`{(coverage.get('final_by_regime') or {}).get('all')}`, "
        f"PRE=`{(coverage.get('final_by_regime') or {}).get('pre')}`, "
        f"IN=`{(coverage.get('final_by_regime') or {}).get('in')}`"
    )
    parts.append(
        f"- Robustez: perm_n=`{meta.get('perm_n')}`, bootstrap_n=`{meta.get('bootstrap_n')}`, "
        f"fdr_alpha=`{meta.get('fdr_alpha')}`"
    )
    parts.append("")

    parts.append(_conclusion_md(data))
    parts.append(_combo_table_md("all", rows_all))
    parts.append(_combo_table_md("pre", rows_pre))
    parts.append(_combo_table_md("in", rows_in))
    return "\n".join(parts).strip() + "\n"


def run(args: argparse.Namespace) -> Dict[str, Any]:
    src = Path(str(args.input_json)).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"Arquivo não encontrado: {src}")
    data = _load_json(src)

    out_md = Path(str(args.out_md)).expanduser().resolve()
    out_pdf = Path(str(args.out_pdf)).expanduser().resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    md = _build_markdown(data, src)
    out_md.write_text(md, encoding="utf-8")

    render = Path(str(args.render_script)).expanduser().resolve()
    if not render.exists():
        raise SystemExit(f"Render script não encontrado: {render}")

    py = str(args.python_bin or "").strip()
    if not py:
        # fallback simples
        py = "python3"
    subprocess.run([py, str(render), str(out_md), str(out_pdf)], check=True)

    return {
        "input_json": str(src),
        "output_md": str(out_md),
        "output_pdf": str(out_pdf),
        "rows": {
            "all": len(_build_combo_rows(data, "all")),
            "pre": len(_build_combo_rows(data, "pre")),
            "in": len(_build_combo_rows(data, "in")),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Gera PDF robusto Back Pre/In com conclusão + tabela completa de combinações."
    )
    ap.add_argument("--input-json", required=True, help="Saída do back_pre_in_oos_permutation.py")
    ap.add_argument("--out-md", default="logs/back_pre_in_robust_report.md")
    ap.add_argument("--out-pdf", default="logs/back_pre_in_robust_report.pdf")
    ap.add_argument("--python-bin", default=os.getenv("PYTHON_BIN", ""))
    ap.add_argument(
        "--render-script",
        default="docs/render_markdown_to_pdf.py",
        help="Script que renderiza markdown->pdf (default: docs/render_markdown_to_pdf.py)",
    )
    args = ap.parse_args()
    rep = run(args)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

