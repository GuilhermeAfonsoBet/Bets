#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera um relatório objetivo para validar o efeito de Região como gating (modo block-bad).

Entradas:
- Baseline OOS semanal: oos_walkforward_global_bayes_roll12_robust_p10_p70_weekly.csv
- Região-gating (block-bad) OOS semanal: oos_walkforward_region_gating_exantepred_blockbad_weekly.csv
- Lista de bloqueios por semana/segmento: oos_walkforward_region_gating_exantepred_blockbad_blocked_regions.csv

Saídas:
- compare_region_gating_blockbad_vs_baseline_weekly.csv
- region_gating_blockbad_blocked_regions_summary.csv
- region_gating_blockbad_report.md
"""

from __future__ import annotations

from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd


BASE_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")
MODE = "global_bayes_roll12_robust_p10_p70"

BASELINE_WEEKLY = BASE_DIR / f"oos_walkforward_{MODE}_weekly.csv"
GATING_WEEKLY = BASE_DIR / "oos_walkforward_region_gating_exantepred_blockbad_weekly.csv"
BLOCKED = BASE_DIR / "oos_walkforward_region_gating_exantepred_blockbad_blocked_regions.csv"

OUT_WEEKLY = BASE_DIR / "compare_region_gating_blockbad_vs_baseline_weekly.csv"
OUT_BLOCK_SUM = BASE_DIR / "region_gating_blockbad_blocked_regions_summary.csv"
OUT_MD = BASE_DIR / "region_gating_blockbad_report.md"


def _fmt_money(x: float) -> str:
    return f"{x:,.2f}"


def _fmt_pct(x: float) -> str:
    return f"{x*100.0:+.2f}%"


def main() -> int:
    for p in [BASELINE_WEEKLY, GATING_WEEKLY]:
        if not p.exists():
            raise FileNotFoundError(str(p))

    b = pd.read_csv(BASELINE_WEEKLY)
    g = pd.read_csv(GATING_WEEKLY)
    b["week"] = b["week"].astype(str)
    g["week"] = g["week"].astype(str)

    m = b.merge(g, on="week", suffixes=("_base", "_gate"))
    # deltas
    m["delta_pnl"] = pd.to_numeric(m["profit_cap2_usd_gate"], errors="coerce") - pd.to_numeric(m["profit_cap2_usd_base"], errors="coerce")
    m["delta_stake"] = pd.to_numeric(m["stake_usd_gate"], errors="coerce") - pd.to_numeric(m["stake_usd_base"], errors="coerce")
    m["delta_n_bets"] = pd.to_numeric(m["n_bets_gate"], errors="coerce") - pd.to_numeric(m["n_bets_base"], errors="coerce")
    m["roi_base"] = pd.to_numeric(m["roi_on_stake_cap2_base"], errors="coerce")
    m["roi_gate"] = pd.to_numeric(m["roi_on_stake_cap2_gate"], errors="coerce")

    # impacto percentual no PnL (baseline)
    pnl_base = pd.to_numeric(m["profit_cap2_usd_base"], errors="coerce").to_numpy(float)
    m["delta_pnl_over_pnl_base"] = m["delta_pnl"] / pnl_base
    m.loc[~np.isfinite(pnl_base) | (np.abs(pnl_base) < 1e-9), "delta_pnl_over_pnl_base"] = np.nan

    OUT_WEEKLY.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(OUT_WEEKLY, index=False)

    # blocked regions summary
    if BLOCKED.exists():
        blk = pd.read_csv(BLOCKED)
        blk["week"] = blk["week"].astype(str)
        blk["rule_key"] = blk["rule_key"].astype(str)
        blk["blocked_regions"] = blk["blocked_regions"].astype(str).fillna("")

        reg_ctr = Counter()
        for s in blk["blocked_regions"].tolist():
            for r in str(s).split(","):
                r = r.strip()
                if r:
                    reg_ctr[r] += 1

        by_rule = (
            blk.assign(n_regions=blk["blocked_regions"].apply(lambda s: len([r for r in str(s).split(",") if r.strip()])))
            .groupby("rule_key", as_index=False)
            .agg(
                weeks_with_blocks=("week", "nunique"),
                block_events=("week", "size"),
                mean_regions_blocked=("n_regions", "mean"),
                max_regions_blocked=("n_regions", "max"),
            )
            .sort_values(["weeks_with_blocks", "block_events"], ascending=False)
        )

        by_week = (
            blk.assign(n_regions=blk["blocked_regions"].apply(lambda s: len([r for r in str(s).split(",") if r.strip()])))
            .groupby("week", as_index=False)
            .agg(
                n_rules_with_blocks=("rule_key", "nunique"),
                block_events=("rule_key", "size"),
                mean_regions_blocked=("n_regions", "mean"),
                max_regions_blocked=("n_regions", "max"),
            )
            .sort_values("week")
        )

        reg_rows = [{"region": k, "block_count": int(v)} for k, v in reg_ctr.most_common()]
        by_region = pd.DataFrame(reg_rows)

        # store as single CSV with sections (wide)
        # We write three CSVs into one file with headers separating blocks.
        # For compatibility, we write as a single table by adding a 'section' column.
        out = []
        if not by_rule.empty:
            t = by_rule.copy()
            t.insert(0, "section", "by_rule_key")
            out.append(t)
        if not by_week.empty:
            t = by_week.copy()
            t.insert(0, "section", "by_week")
            out.append(t)
        if not by_region.empty:
            t = by_region.copy()
            t.insert(0, "section", "by_region")
            out.append(t)
        out_df = pd.concat(out, axis=0, ignore_index=True) if out else pd.DataFrame()
        out_df.to_csv(OUT_BLOCK_SUM, index=False)
    else:
        OUT_BLOCK_SUM.write_text("section,info\nmissing,blocked_regions_file_not_found\n", encoding="utf-8")

    # markdown report
    stake_b = float(pd.to_numeric(m["stake_usd_base"], errors="coerce").sum())
    pnl_b = float(pd.to_numeric(m["profit_cap2_usd_base"], errors="coerce").sum())
    stake_g = float(pd.to_numeric(m["stake_usd_gate"], errors="coerce").sum())
    pnl_g = float(pd.to_numeric(m["profit_cap2_usd_gate"], errors="coerce").sum())

    weeks_active_b = int((pd.to_numeric(m["stake_usd_base"], errors="coerce") > 0).sum())
    weeks_active_g = int((pd.to_numeric(m["stake_usd_gate"], errors="coerce") > 0).sum())
    changed_weeks = int((pd.to_numeric(m["stake_usd_gate"], errors="coerce") != pd.to_numeric(m["stake_usd_base"], errors="coerce")).sum())

    mm = m[(pd.to_numeric(m["stake_usd_base"], errors="coerce") > 0) | (pd.to_numeric(m["stake_usd_gate"], errors="coerce") > 0)].copy()
    improved = int((mm["delta_pnl"] > 1e-12).sum())
    worse = int((mm["delta_pnl"] < -1e-12).sum())
    equal = int((np.abs(mm["delta_pnl"]) <= 1e-12).sum())

    best = m.sort_values("delta_pnl", ascending=False).head(6)
    worst = m.sort_values("delta_pnl", ascending=True).head(6)

    lines = []
    lines.append("## Região como gating (block-bad) — Validação numérica\n")
    lines.append(f"- Baseline: PnL={_fmt_money(pnl_b)} | Stake={_fmt_money(stake_b)} | ROI/$={pnl_b/stake_b:.5f} | semanas ativas={weeks_active_b}\n")
    lines.append(f"- Gating:   PnL={_fmt_money(pnl_g)} | Stake={_fmt_money(stake_g)} | ROI/$={pnl_g/stake_g:.5f} | semanas ativas={weeks_active_g}\n")
    lines.append(f"- ΔPnL={_fmt_money(pnl_g-pnl_b)} | ΔStake={_fmt_money(stake_g-stake_b)} | semanas com mudança={changed_weeks}/{len(m)}\n")
    lines.append(f"- Entre semanas com stake: melhorou={improved} | piorou={worse} | igual={equal}\n")

    lines.append("\n### Maiores melhorias (por semana)\n")
    for _, r in best.iterrows():
        lines.append(
            f"- {r['week']}: ΔPnL={_fmt_money(float(r['delta_pnl']))} | "
            f"PnL base={_fmt_money(float(r['profit_cap2_usd_base']))} → gate={_fmt_money(float(r['profit_cap2_usd_gate']))} | "
            f"Stake base={_fmt_money(float(r['stake_usd_base']))} → gate={_fmt_money(float(r['stake_usd_gate']))}\n"
        )
    lines.append("\n### Maiores pioras (por semana)\n")
    for _, r in worst.iterrows():
        lines.append(
            f"- {r['week']}: ΔPnL={_fmt_money(float(r['delta_pnl']))} | "
            f"PnL base={_fmt_money(float(r['profit_cap2_usd_base']))} → gate={_fmt_money(float(r['profit_cap2_usd_gate']))} | "
            f"Stake base={_fmt_money(float(r['stake_usd_base']))} → gate={_fmt_money(float(r['stake_usd_gate']))}\n"
        )

    lines.append("\n### Artefatos\n")
    lines.append(f"- Compare semanal: `{OUT_WEEKLY}`\n")
    lines.append(f"- Resumo bloqueios: `{OUT_BLOCK_SUM}`\n")
    lines.append(f"- Weekly gating: `{GATING_WEEKLY}`\n")

    OUT_MD.write_text("".join(lines), encoding="utf-8")

    print(str(OUT_WEEKLY))
    print(str(OUT_BLOCK_SUM))
    print(str(OUT_MD))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

