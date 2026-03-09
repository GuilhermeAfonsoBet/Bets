#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera um comparativo OOS (walk-forward) para múltiplas bancas, em um único PDF,
usando exatamente os mesmos critérios/filtros de um "preset" do OOS.

Motivação: facilitar conciliação entre relatórios e evitar PDFs rasterizados.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional


def _parse_csv_floats(s: str) -> List[float]:
    out: List[float] = []
    for x in (s or "").split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    return out


def _preset_args(name: str) -> List[str]:
    """
    Presets de critérios OOS.

    v38_oos_AHgatePre_leaguePre:
    - AH gate: abs(line) <= 2 apenas no Pre
    - liga entra na chave somente no Pre (combinação×liga)
    - exclui exec_bucket Back 10-20s apenas no WF
    - WF 2d treino / 2d teste / step 2d (como nos relatórios antigos)
    """
    name = (name or "").strip()
    if name in ("v38_oos_AHgatePre_leaguePre", "AHgatePre_leaguePre"):
        return [
            "--wf-ah-max-abs-line",
            "2",
            "--wf-ah-scope",
            "pre",
            "--wf-key-by-league",
            "--wf-key-by-league-scope",
            "pre",
            "--wf-exclude-exec-buckets-back",
            "10-20s",
            "--wf-train-days",
            "2",
            "--wf-test-days",
            "2",
            "--wf-step-days",
            "2",
            "--wf-min-matches",
            "0",
        ]
    raise SystemExit(f"Preset desconhecido: {name!r}")


def _run_analyzer(
    *,
    repo_root: Path,
    out_md: Path,
    direction: str,
    versions: str,
    lookback_days: int,
    end_utc: Optional[str],
    bankroll: float,
    preset: str,
) -> None:
    script = repo_root / "betinasia_bot" / "analyze_contexto_operacao_b808_robust_report.py"
    cmd = [
        str(repo_root / "betinasia_bot" / "venv" / "bin" / "python"),
        str(script),
        "--direction",
        direction,
        "--versions",
        versions,
        "--lookback-days",
        str(int(lookback_days)),
        "--walkforward",
        "--wf-train-mode",
        "expanding",
        "--kelly-bankroll",
        str(float(bankroll)),
        "--only-oos",
        "--out",
        str(out_md),
    ]
    if end_utc:
        cmd += ["--end-utc", str(end_utc)]
    cmd += _preset_args(preset)
    subprocess.run(cmd, check=True, cwd=str(repo_root / "betinasia_bot"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--direction", default="up", choices=["up", "down"])
    p.add_argument(
        "--versions",
        default="v4.0-api,v5.0-ws-only,v5.1-ws-gate-lay",
        help="CSV de audit_version (igual ao relatório base).",
    )
    p.add_argument("--lookback-days", type=int, default=21)
    p.add_argument(
        "--end-utc",
        default="",
        help="Fim do recorte (ISO-8601 ou YYYY-MM-DD). Use para conciliar com um relatório antigo.",
    )
    p.add_argument(
        "--banks",
        default="10000,50000,100000",
        help="CSV de bancas para rodar (ex.: 10000,50000,100000).",
    )
    p.add_argument(
        "--preset",
        default="v38_oos_AHgatePre_leaguePre",
        help="Preset de critérios OOS.",
    )
    p.add_argument("--out-md", default="/tmp/oos_3banks.md")
    p.add_argument("--out-pdf", default="/tmp/oos_3banks.pdf")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    banks = _parse_csv_floats(args.banks)
    if not banks:
        raise SystemExit("Nenhuma banca em --banks.")

    out_md = Path(args.out_md).expanduser().resolve()
    out_pdf = Path(args.out_pdf).expanduser().resolve()

    end_utc = (args.end_utc or "").strip() or None

    parts: List[str] = []
    parts.append("# OOS — Comparativo por banca (3 banks)\n\n")
    parts.append(f"- **Preset**: `{args.preset}`\n")
    parts.append(f"- **Recorte**: direction=`{args.direction}`, versions=`{args.versions}`, lookback_days=`{args.lookback_days}`, end_utc=`{end_utc or 'agora'}`\n\n")
    parts.append("---\n\n")

    with tempfile.TemporaryDirectory(prefix="oos_3banks_") as td:
        td_path = Path(td)
        for b in banks:
            one_md = td_path / f"oos_{int(b)}.md"
            _run_analyzer(
                repo_root=repo_root,
                out_md=one_md,
                direction=args.direction,
                versions=args.versions,
                lookback_days=int(args.lookback_days),
                end_utc=end_utc,
                bankroll=float(b),
                preset=args.preset,
            )
            parts.append(f"## Banca (ref) = {float(b):.2f}\n\n")
            parts.append(one_md.read_text(encoding="utf-8"))
            parts.append("\n\n---\n\n")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("".join(parts), encoding="utf-8")

    # Render PDF (texto selecionável)
    render = repo_root / "betinasia_bot" / "docs" / "render_markdown_to_pdf.py"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(repo_root / "betinasia_bot" / "venv" / "bin" / "python"),
            str(render),
            str(out_md),
            str(out_pdf),
        ],
        check=True,
        cwd=str(repo_root / "betinasia_bot"),
    )

    print(str(out_md))
    print(str(out_pdf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

