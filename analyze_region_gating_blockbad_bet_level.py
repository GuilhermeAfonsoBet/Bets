#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análise bet-a-bet: por que o stake total caiu pouco mesmo com Europa Ocidental bloqueada várias vezes?

Reconstrói as apostas selecionadas pelo baseline (regras OOS) e marca quais seriam bloqueadas
pelo gating block-bad (por semana e rule_key). Agrega stake removido por região/semana/segmento.

Saídas:
- analysis_proba_raw/pro_portfolio_all/region_gating_blockbad_bet_level_selected.csv
- analysis_proba_raw/pro_portfolio_all/region_gating_blockbad_bet_level_removed_by_region.csv
- analysis_proba_raw/pro_portfolio_all/region_gating_blockbad_bet_level_removed_by_week.csv
- analysis_proba_raw/pro_portfolio_all/region_gating_blockbad_bet_level_removed_by_rule.csv
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")
MODE = "global_bayes_roll12_robust_p10_p70"

SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")
RULES = BASE_DIR / f"oos_walkforward_{MODE}_selected_rules.csv"
REGION = BASE_DIR / "region_exante_pred.csv"
BLOCKED = BASE_DIR / "oos_walkforward_region_gating_exantepred_blockbad_blocked_regions.csv"

OUT_SEL = BASE_DIR / "region_gating_blockbad_bet_level_selected.csv"
OUT_R = BASE_DIR / "region_gating_blockbad_bet_level_removed_by_region.csv"
OUT_W = BASE_DIR / "region_gating_blockbad_bet_level_removed_by_week.csv"
OUT_K = BASE_DIR / "region_gating_blockbad_bet_level_removed_by_rule.csv"


def main() -> int:
    for p in [SCORED, RULES, REGION, BLOCKED]:
        if not p.exists():
            raise FileNotFoundError(str(p))

    df = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])
    df["week"] = pd.to_datetime(df["BIA_ApostaUTC"]).dt.to_period("W-SUN").astype(str)
    df["date"] = pd.to_datetime(df["BIA_ApostaUTC"]).dt.date.astype(str)
    df["house_cap"] = pd.to_numeric(df["house_cap"], errors="coerce").astype(float)
    df["house_cap"] = df["house_cap"].where(np.isfinite(df["house_cap"]) & (df["house_cap"] > 0), np.inf)
    df["roi_cap2"] = pd.to_numeric(df["roi_calc"], errors="coerce").clip(upper=2.0).to_numpy(float)

    reg = pd.read_csv(REGION, usecols=["ID Aposta", "region_pred"])
    reg = reg.rename(columns={"region_pred": "region"})
    df = df.merge(reg, how="left", on="ID Aposta")
    df["region"] = df["region"].astype("string").fillna("desconhecida").astype(str)

    rules = pd.read_csv(RULES)
    # block map: (week, rule_key) -> set(regions)
    blk = pd.read_csv(BLOCKED)
    blk["week"] = blk["week"].astype(str)
    blk["rule_key"] = blk["rule_key"].astype(str)
    blk["blocked_regions"] = blk["blocked_regions"].astype(str).fillna("")
    block_map = {}
    for _, r in blk.iterrows():
        key = (str(r["week"]), str(r["rule_key"]))
        s = set([x.strip() for x in str(r["blocked_regions"]).split(",") if x.strip()])
        block_map[key] = s

    rows = []
    for w_test, rw in rules.groupby("test_week", as_index=False):
        w = str(w_test)
        dfw = df[df["week"] == w].copy()
        if dfw.empty:
            continue
        alpha = float(rw["alpha_effective"].iloc[0]) if "alpha_effective" in rw.columns and np.isfinite(float(rw["alpha_effective"].iloc[0])) else float(rw["alpha_global"].iloc[0])
        for _, r in rw.iterrows():
            if str(r.get("status")) != "ok":
                continue
            stake_frac = float(r.get("stake_frac", 0.0))
            if stake_frac <= 0:
                continue
            bt = str(r["bet_type"])
            dow = str(r["dow_pt"])
            sc = str(r["score_col"])
            cutoff = float(r["cutoff"])
            rk = str(r.get("rule_key", f"{bt}|{dow}"))

            x = dfw[(dfw["bet_type"] == bt) & (dfw["dow_pt"] == dow)].copy()
            if x.empty or sc not in x.columns:
                continue
            score = pd.to_numeric(x[sc], errors="coerce").to_numpy(float)
            roi2 = x["roi_cap2"].to_numpy(float)
            m = np.isfinite(score) & (score >= cutoff) & np.isfinite(roi2)
            if not np.any(m):
                continue
            x = x.iloc[np.where(m)[0]].copy()

            stake0 = 2300.0 * stake_frac * float(alpha)
            x["stake_eff"] = np.minimum(stake0, x["house_cap"].to_numpy(float))
            x["profit_cap2"] = x["stake_eff"].to_numpy(float) * x["roi_cap2"].to_numpy(float)
            x["rule_key"] = rk
            x["alpha_effective"] = float(alpha)

            blocked = block_map.get((w, rk), set())
            x["is_blocked"] = x["region"].astype(str).isin(list(blocked)).astype(int)
            x["blocked_regions_list"] = ",".join(sorted(blocked)) if blocked else ""
            rows.append(
                x[
                    [
                        "ID Aposta",
                        "BIA_ApostaUTC",
                        "week",
                        "bet_type",
                        "dow_pt",
                        "rule_key",
                        "region",
                        "stake_eff",
                        "profit_cap2",
                        "is_blocked",
                        "blocked_regions_list",
                    ]
                ]
            )

    sel = pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()
    OUT_SEL.parent.mkdir(parents=True, exist_ok=True)
    sel.to_csv(OUT_SEL, index=False)

    if sel.empty:
        print("sem apostas selecionadas")
        return 0

    removed = sel[sel["is_blocked"] == 1].copy()
    removed["stake_eff"] = pd.to_numeric(removed["stake_eff"], errors="coerce").astype(float)
    removed["profit_cap2"] = pd.to_numeric(removed["profit_cap2"], errors="coerce").astype(float)

    # aggregates
    by_region = removed.groupby("region", as_index=False).agg(
        stake_removed=("stake_eff", "sum"),
        pnl_removed=("profit_cap2", "sum"),
        n_bets=("ID Aposta", "size"),
        n_weeks=("week", "nunique"),
    ).sort_values("stake_removed", ascending=False)
    by_week = removed.groupby("week", as_index=False).agg(
        stake_removed=("stake_eff", "sum"),
        pnl_removed=("profit_cap2", "sum"),
        n_bets=("ID Aposta", "size"),
        n_rules=("rule_key", "nunique"),
    ).sort_values("week")
    by_rule = removed.groupby("rule_key", as_index=False).agg(
        stake_removed=("stake_eff", "sum"),
        pnl_removed=("profit_cap2", "sum"),
        n_bets=("ID Aposta", "size"),
        n_weeks=("week", "nunique"),
    ).sort_values("stake_removed", ascending=False)

    by_region.to_csv(OUT_R, index=False)
    by_week.to_csv(OUT_W, index=False)
    by_rule.to_csv(OUT_K, index=False)

    # quick print
    stake_total = float(pd.to_numeric(sel["stake_eff"], errors="coerce").sum())
    stake_removed = float(by_week["stake_removed"].sum()) if not by_week.empty else 0.0
    print("stake_total_selected", stake_total, "stake_removed_total", stake_removed, "pct_removed", (stake_removed / stake_total * 100.0) if stake_total > 0 else 0.0)
    print("top removed regions:")
    print(by_region.head(10).to_string(index=False))
    print(str(OUT_SEL))
    print(str(OUT_R))
    print(str(OUT_W))
    print(str(OUT_K))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

