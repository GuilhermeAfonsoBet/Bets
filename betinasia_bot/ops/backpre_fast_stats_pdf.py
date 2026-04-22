from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _load_env_file(path: Path) -> None:
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Gera PDF: estatísticas da tese Back Pre fast (pre_submit_ms).")
    ap.add_argument("--out", default=os.getenv("BACKPRE_FAST_STATS_PDF_OUT", "docs/backpre_fast_stats.pdf"))
    ap.add_argument("--env-file", default=os.getenv("ENV_FILE", ".env"))
    args = ap.parse_args()

    _load_env_file(Path(str(args.env_file)))

    ts = _utcnow().strftime("%Y-%m-%d %H:%M UTC")
    start_day = str(os.getenv("DAILY_BACKPRE_FAST_THESIS_START_DAY", "") or "").strip() or "—"
    thr_ms = str(os.getenv("EXECUTOR_BACKPRE_FAST_MAX_PRE_SUBMIT_MS", "5000") or "5000").strip()
    stake_hi = str(os.getenv("EXECUTOR_BACKPRE_FAST_STAKE_HI", "12") or "12").strip()
    stake_lo = str(os.getenv("EXECUTOR_BACKPRE_FAST_STAKE_LO", "1.50") or "1.50").strip()

    md = f"""# Estatísticas — Tese Back Pre fast (pre_submit_ms)

Gerado em: **{ts}**

## Definição da tese (operacional)
- Universo: **Back Pre** (pre-match), apostas efetivas (`LIVE_OK`).
- “Fast”: `pre_submit_ms <= {thr_ms}ms`.
- Sizing operacional (quando habilitado): fast ⇒ **stake={stake_hi}**, demais Back ⇒ **stake={stake_lo}**.
- Início operacional (recorte recomendado): `{start_day}` (UTC).  
  (Use a data em que você ligou `EXECUTOR_BACKPRE_FAST_STAKE_ENABLE=1` em produção.)

## Evidências no relatório diário (PDF)
As evidências quantitativas ficam no `daily_full_report` nas seções:
- **Latência × ROI (Back Pre/In)** (proxy: `call_to_done_ms` buckets, usando subset com placar/ROI).
- **Slippage × Latência** (Back Pre) e **Slippage × ROI** (accounting por `order_id`).
- **Tese: Back Pre fast (pós-início; stake=HI)**:
  - `ROIw_liquidado = (∑P&L_liquidado)/(∑stake_liquidado)` por grupo
  - `n_liquidadas` vs `n_abertas` (quando `open_stakes.csv` está disponível)
  - compliance de stake (12 vs 1.5 vs other/NA) e distribuição de `pre_submit_ms`

## Interpretação (como decidir)
- Para afirmar robustez, olhe **IC90/IC95** (bootstrap) e principalmente a estabilidade do sinal ao longo do tempo.
- Se `n_liquidadas` for pequeno, **não** usar ROIw do dia como veredito; esperar maturação (settlement).

## Observações
Este PDF é um “sumário metodológico” para acompanhar a tese. Os números do dia devem ser lidos no PDF diário.

## Como anexar evidências dos testes (PowerShell)
Cole abaixo (ou mantenha versionado no repo) os outputs/prints dos comandos que vocês rodaram no PowerShell, por exemplo:

```text
<cole aqui os resultados dos testes estatísticos (ICs/p-valores/ablation) que motivaram a tese>
```
"""

    out_pdf = Path(str(args.out)).expanduser()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    tmp_md = out_pdf.with_suffix(".md")
    tmp_md.write_text(md, encoding="utf-8")

    renderer = Path(__file__).resolve().parent.parent / "docs" / "render_markdown_to_pdf.py"
    subprocess.run([sys.executable, str(renderer), str(tmp_md), str(out_pdf)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

