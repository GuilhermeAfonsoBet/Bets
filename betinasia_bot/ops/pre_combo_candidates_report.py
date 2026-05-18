#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Relatório estatístico para combos candidatos em Back PRE.

Objetivo:
- executar o mesmo framework do ops.backin_value_test para 3 subsets candidatos:
  A: < 5s & (-2, 2]
  B: < 5s & <= -2%
  C: 10-20s & (-2, 2]
- consolidar métricas:
  ROI subset, delta vs baseline PRE, bootstrap por jogo,
  CI90/CI95 do delta, p(ROI subset > 0), p(delta > 0),
  teste de permutação por jogo e correção múltipla (Holm/Bonferroni).

Saídas:
- combo_A.json, combo_B.json, combo_C.json (output bruto por combo)
- pre_combo_candidates_summary.json
- pre_combo_candidates_summary.md
- pre_combo_candidates_summary.pdf (opcional)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from ops import backin_value_test as bvt
except Exception:
    # Compatibilidade ao rodar como arquivo direto:
    #   python ops/pre_combo_candidates_report.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ops import backin_value_test as bvt


@dataclass(frozen=True)
class Combo:
    name: str
    lat_bucket: str
    slip_bucket: str


DEFAULT_COMBOS: List[Combo] = [
    Combo(name="A", lat_bucket="< 5s", slip_bucket="(-2, 2]"),
    Combo(name="B", lat_bucket="< 5s", slip_bucket="<= -2%"),
    Combo(name="C", lat_bucket="10-20s", slip_bucket="(-2, 2]"),
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _quantile(xs: List[float], q: float) -> Optional[float]:
    try:
        if not xs:
            return None
        xs2 = sorted(float(x) for x in xs)
        if len(xs2) == 1:
            return float(xs2[0])
        pos = q * (len(xs2) - 1)
        lo = int(pos)
        hi = min(len(xs2) - 1, lo + 1)
        w = float(pos - lo)
        return float(xs2[lo] * (1.0 - w) + xs2[hi] * w)
    except Exception:
        return None


def _roi_weighted(pnl_sum: float, exp_sum: float) -> Optional[float]:
    try:
        if exp_sum <= 0:
            return None
        return float(pnl_sum / exp_sum * 100.0)
    except Exception:
        return None


def _bootstrap_distributions(*, by_game: Dict[str, Dict[str, float]], n_boot: int, seed: int) -> Tuple[List[float], List[float]]:
    evs = [e for e in (by_game or {}).keys() if str(e)]
    if not evs:
        return [], []
    rng = random.Random(int(seed))
    n = int(len(evs))
    roi_sub: List[float] = []
    delta: List[float] = []
    for _ in range(int(max(0, n_boot))):
        pnl_b = exp_b = pnl_s = exp_s = 0.0
        for _j in range(n):
            rec = by_game.get(evs[rng.randrange(0, n)]) or {}
            pnl_b += float(rec.get("pnl_base") or 0.0)
            exp_b += float(rec.get("exp_base") or 0.0)
            pnl_s += float(rec.get("pnl_sub") or 0.0)
            exp_s += float(rec.get("exp_sub") or 0.0)
        rb = _roi_weighted(pnl_b, exp_b)
        rs = _roi_weighted(pnl_s, exp_s)
        if rs is not None:
            roi_sub.append(float(rs))
        if rb is not None and rs is not None:
            delta.append(float(rs - rb))
    return roi_sub, delta


def _perm_signflip_delta_gt0(*, by_game: Dict[str, Dict[str, float]], n_perm: int, seed: int) -> Dict[str, Any]:
    deltas: List[float] = []
    for rec in (by_game or {}).values():
        rb = _roi_weighted(float(rec.get("pnl_base") or 0.0), float(rec.get("exp_base") or 0.0))
        rs = _roi_weighted(float(rec.get("pnl_sub") or 0.0), float(rec.get("exp_sub") or 0.0))
        if rb is None or rs is None:
            continue
        deltas.append(float(rs - rb))
    if not deltas:
        return {
            "n_games": 0,
            "obs_mean_delta": None,
            "p_one_sided_gt0": None,
            "p_two_sided": None,
            "n_perm": int(n_perm),
        }
    obs = float(sum(deltas) / float(len(deltas)))
    rng = random.Random(int(seed))
    ge = 0
    ge_abs = 0
    for _ in range(int(max(0, n_perm))):
        stat = float(sum((d if rng.random() < 0.5 else -d) for d in deltas) / float(len(deltas)))
        if stat >= obs:
            ge += 1
        if abs(stat) >= abs(obs):
            ge_abs += 1
    return {
        "n_games": int(len(deltas)),
        "obs_mean_delta": float(obs),
        "p_one_sided_gt0": float((ge + 1) / float(int(n_perm) + 1)),
        "p_two_sided": float((ge_abs + 1) / float(int(n_perm) + 1)),
        "n_perm": int(n_perm),
    }


def _discover_day_dir(day_dir_arg: str) -> Path:
    if str(day_dir_arg or "").strip():
        p = Path(str(day_dir_arg)).expanduser().resolve()
        if not p.exists():
            raise SystemExit(f"--day-dir não existe: {p}")
        return p

    roots = [Path("logs/daily_reports"), Path("logs")]
    candidates: List[Path] = []
    if roots[0].exists():
        for d in sorted(roots[0].glob("*"), key=lambda x: x.stat().st_mtime, reverse=True):
            if not d.is_dir():
                continue
            if (d / "accounting_daily_report.json").exists():
                candidates.append(d.resolve())
    if roots[1].exists() and (roots[1] / "accounting_daily_report.json").exists():
        candidates.append(roots[1].resolve())
    if not candidates:
        raise SystemExit("Nenhum day_dir válido encontrado (faltando accounting_daily_report.json). Use --day-dir.")
    return candidates[0]


def _format_pct(x: Any) -> str:
    v = _safe_float(x)
    return "null" if v is None else f"{v:.4f}"


def _md_row(rec: Dict[str, Any]) -> str:
    return (
        f"| {rec.get('combo')} | {rec.get('lat_bucket')} | {rec.get('slip_bucket')} | "
        f"{_format_pct(rec.get('roi_subset_pct'))} | {_format_pct(rec.get('delta_vs_pre_baseline_pct'))} | "
        f"{json.dumps(rec.get('delta_ci90'), ensure_ascii=False)} | {json.dumps(rec.get('delta_ci95'), ensure_ascii=False)} | "
        f"{_format_pct(rec.get('p_roi_subset_gt0'))} | {_format_pct(rec.get('p_delta_gt0_boot'))} | "
        f"{_format_pct(rec.get('p_delta_gt0_perm'))} | {_format_pct(rec.get('p_delta_gt0_perm_holm'))} | "
        f"{_format_pct(rec.get('p_delta_gt0_perm_bonf'))} |"
    )


def _render_pdf(md_path: Path, pdf_path: Path, render_script: Path) -> Tuple[bool, str]:
    if not render_script.exists():
        return False, f"render script ausente: {render_script}"
    try:
        p = subprocess.run(
            [sys.executable, str(render_script), str(md_path), str(pdf_path)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if p.returncode == 0:
            return True, (p.stdout or "").strip()
        return False, (p.stdout or "").strip()
    except Exception as e:
        return False, str(e)


def main() -> int:
    ap = argparse.ArgumentParser(description="Gera estudo PRE para combos candidatos (A/B/C) com bootstrap, permutação e correção múltipla.")
    ap.add_argument("--day-dir", default="", help="Pasta com accounting_daily_report.json (opcional; autodiscover se vazio).")
    ap.add_argument("--start-day", default="2026-04-04", help="Data inicial UTC (YYYY-MM-DD).")
    ap.add_argument("--end-day", default="", help="Data final UTC opcional (YYYY-MM-DD).")
    ap.add_argument("--out-dir", default="", help="Pasta de saída (default: logs/daily_reports/<ts>_pre_combo_candidates).")
    ap.add_argument("--n-boot", type=int, default=5000, help="N bootstrap por jogo.")
    ap.add_argument("--n-perm", type=int, default=5000, help="N permutações por jogo.")
    ap.add_argument("--seed", type=int, default=1337, help="Seed base.")
    ap.add_argument("--emit-pdf", action="store_true", help="Gerar PDF (usa docs/render_markdown_to_pdf.py).")
    ap.add_argument("--render-script", default="docs/render_markdown_to_pdf.py", help="Script de render Markdown->PDF.")
    args = ap.parse_args()

    day_dir = _discover_day_dir(str(args.day_dir or ""))
    out_dir = Path(str(args.out_dir or "")).expanduser()
    if not str(out_dir):
        out_dir = Path("logs/daily_reports") / f"{_utcnow().strftime('%Y%m%d_%H%M%S')}_pre_combo_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_boot = int(max(100, int(args.n_boot)))
    n_perm = int(max(100, int(args.n_perm)))
    seed_base = int(args.seed)
    end_day = str(args.end_day or "").strip() or None

    orig_boot = bvt._bootstrap_by_game

    def patched_boot(*, by_game: Dict[str, Dict[str, float]], n_boot: int, seed: int) -> Dict[str, Any]:
        out = orig_boot(by_game=by_game, n_boot=int(n_boot), seed=int(seed))
        roi_sub_dist, delta_dist = _bootstrap_distributions(by_game=by_game, n_boot=int(n_boot), seed=int(seed))
        out["roi_sub_ci95"] = {"lb": _quantile(roi_sub_dist, 0.025), "ub": _quantile(roi_sub_dist, 0.975)}
        out["delta_ci95"] = {"lb": _quantile(delta_dist, 0.025), "ub": _quantile(delta_dist, 0.975)}
        out["permutation_by_game"] = _perm_signflip_delta_gt0(by_game=by_game, n_perm=int(n_perm), seed=int(seed) + 911)
        return out

    bvt._bootstrap_by_game = patched_boot
    try:
        rows: List[Dict[str, Any]] = []
        pvals_perm: List[float] = []

        for idx, combo in enumerate(DEFAULT_COMBOS):
            out = asyncio.run(
                bvt._run(
                    day_dir=day_dir,
                    start_day=str(args.start_day),
                    end_day=end_day,
                    regime="pre",
                    lat_bucket=str(combo.lat_bucket),
                    slip_bucket=str(combo.slip_bucket),
                    lat_min_ms=None,
                    lat_max_ms=None,
                    pre_submit_min_ms=None,
                    pre_submit_max_ms=None,
                    slip_pre_min_pct=None,
                    slip_pre_max_pct=None,
                    database_url_override=None,
                    balance_csv_override=None,
                    n_boot=int(n_boot),
                    seed=int(seed_base + idx),
                    limit_stake_factor=0.5,
                    limit_stake_cap=0.0,
                )
            )
            (out_dir / f"combo_{combo.name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

            boot = out.get("bootstrap_by_game") if isinstance(out, dict) else {}
            boot = boot if isinstance(boot, dict) else {}
            perm = boot.get("permutation_by_game") if isinstance(boot.get("permutation_by_game"), dict) else {}

            rec = {
                "combo": combo.name,
                "lat_bucket": combo.lat_bucket,
                "slip_bucket": combo.slip_bucket,
                "roi_subset_pct": ((out.get("subset") or {}).get("roi_w_pct") if isinstance(out, dict) else None),
                "delta_vs_pre_baseline_pct": ((out.get("delta") or {}).get("delta_roi_w_pct") if isinstance(out, dict) else None),
                "delta_ci90": boot.get("delta_ci90"),
                "delta_ci95": boot.get("delta_ci95"),
                "p_roi_subset_gt0": boot.get("p_roi_sub_gt0"),
                "p_delta_gt0_boot": boot.get("p_delta_gt0"),
                "p_delta_gt0_perm": perm.get("p_one_sided_gt0") if isinstance(perm, dict) else None,
                "n_games_boot": boot.get("n_games"),
                "orders_subset": ((out.get("coverage") or {}).get("orders_subset") if isinstance(out, dict) else None),
            }
            rows.append(rec)
            pvals_perm.append(float(rec["p_delta_gt0_perm"]) if _safe_float(rec["p_delta_gt0_perm"]) is not None else 1.0)

        # Correção múltipla
        m = int(len(pvals_perm))
        bonf = [min(1.0, p * float(m)) for p in pvals_perm]
        order = sorted(range(m), key=lambda i: pvals_perm[i])
        holm: List[Optional[float]] = [None] * m
        prev = 0.0
        for rank, idx in enumerate(order):
            adj = min(1.0, (float(m - rank) * float(pvals_perm[idx])))
            adj = max(float(adj), float(prev))
            holm[idx] = float(adj)
            prev = float(adj)
        for i in range(m):
            rows[i]["p_delta_gt0_perm_bonf"] = bonf[i]
            rows[i]["p_delta_gt0_perm_holm"] = holm[i]

        summary = {
            "meta": {
                "ts_utc": _utcnow().isoformat(),
                "day_dir": str(day_dir),
                "start_day": str(args.start_day),
                "end_day": end_day,
                "regime": "pre",
                "n_boot": int(n_boot),
                "n_perm": int(n_perm),
                "seed": int(seed_base),
            },
            "results": rows,
        }
        summary_json = out_dir / "pre_combo_candidates_summary.json"
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        md_lines: List[str] = []
        md_lines.append("# Back PRE - Combos candidatos (A/B/C)")
        md_lines.append("")
        md_lines.append("- Framework: `ops.backin_value_test` + extensões estatísticas (CI95, permutação, Holm/Bonferroni).")
        md_lines.append(f"- day_dir: `{day_dir}`")
        md_lines.append(f"- start_day: `{args.start_day}`")
        if end_day:
            md_lines.append(f"- end_day: `{end_day}`")
        md_lines.append(f"- n_boot: {n_boot}")
        md_lines.append(f"- n_perm: {n_perm}")
        md_lines.append("")
        md_lines.append("| Combo | Lat bucket | Slip bucket | ROI subset % | Delta vs baseline PRE % | CI90 delta | CI95 delta | p(ROI>0) | p(delta>0) boot | p(delta>0) perm | Holm | Bonferroni |")
        md_lines.append("|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|")
        for rec in rows:
            md_lines.append(_md_row(rec))
        md_lines.append("")
        md_lines.append("## JSON consolidado")
        md_lines.append("```json")
        md_lines.append(json.dumps(summary, ensure_ascii=False, indent=2))
        md_lines.append("```")
        md_lines.append("")

        summary_md = out_dir / "pre_combo_candidates_summary.md"
        summary_md.write_text("\n".join(md_lines), encoding="utf-8")

        out_payload: Dict[str, Any] = {
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
        }

        if bool(args.emit_pdf):
            summary_pdf = out_dir / "pre_combo_candidates_summary.pdf"
            ok_pdf, msg_pdf = _render_pdf(summary_md, summary_pdf, Path(str(args.render_script)))
            out_payload["summary_pdf"] = str(summary_pdf) if ok_pdf else None
            out_payload["pdf_status"] = "ok" if ok_pdf else "error"
            out_payload["pdf_log"] = msg_pdf

        print(json.dumps(out_payload, ensure_ascii=False, indent=2))
        return 0
    finally:
        bvt._bootstrap_by_game = orig_boot


if __name__ == "__main__":
    raise SystemExit(main())

