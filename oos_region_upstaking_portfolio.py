#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimento OOS: upstaking por Região (ex-ante) para o portfólio global_bayes (p10_p70).

Ideia:
- Mantém as regras (cutoff/stake_frac/score_col) do baseline.
- Para cada semana de teste, estima no TREINO (últimas 12 semanas) o ROI_cap2 médio por região
  *dentro de cada segmento (DoW x FT/FH) e acima do cutoff*.
- Converte isso em um multiplicador de stake por região (conservador, com clipping).
- Recalcula alpha_global (coarse) para respeitar constraints globais no treino.
- Aplica na semana teste: mesma regra, mas com stake ajustado por região.

Opção:
- Pode combinar com o gating block-bad (bloquear regiões claramente ruins) antes do upstaking.

Saídas:
- oos_walkforward_region_upstake_exantepred_weekly.csv
- oos_walkforward_region_upstake_exantepred_summary.csv
- oos_walkforward_region_upstake_exantepred_multipliers.csv

- oos_walkforward_region_blockbad_upstake_exantepred_weekly.csv
- oos_walkforward_region_blockbad_upstake_exantepred_summary.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

import evaluate_oos_walkforward_strategy as wf


OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")
MODE = "global_bayes_roll12_robust_p10_p70"
SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")
REGION_PRED = OUT_DIR / "region_exante_pred.csv"  # determinístico via EventName

WF_RULES = OUT_DIR / f"oos_walkforward_{MODE}_selected_rules.csv"
WF_WEEKLY = OUT_DIR / f"oos_walkforward_{MODE}_weekly.csv"

TRAIN_WINDOW_WEEKS = 12
MIN_N_PER_REGION = 25

# Upstaking: m = clip(1 + GAMMA*(mu_r - mu_all), [M_MIN, M_MAX])
GAMMA = 0.70
M_MIN = 0.80
M_MAX = 1.20

# Gating block-bad (mesma lógica do outro script)
BLOCK_BAD_REGIONS = True
BAD_MEAN_ROI_TH = -0.02
BLOCK_MIN_N = 20


def _apply_isotonic_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Garante colunas calibradas do score atual (qui/sexdom)."""
    import json

    df = df.copy()
    if "proba_cal_segqui" not in df.columns:
        df["proba_cal_segqui"] = np.nan
    if "proba_cal_sexdom" not in df.columns:
        df["proba_cal_sexdom"] = np.nan
    if "proba_raw_segqui" in df.columns and wf.CALIB_SEGQUI.exists():
        calib = json.loads(wf.CALIB_SEGQUI.read_text(encoding="utf-8"))
        x = np.asarray(calib.get("isotonic", {}).get("x", []), dtype=float)
        y = np.asarray(calib.get("isotonic", {}).get("y", []), dtype=float)
        p_raw = pd.to_numeric(df["proba_raw_segqui"], errors="coerce").to_numpy(dtype=float)
        df["proba_cal_segqui"] = wf._apply_isotonic_vec(p_raw, x=x, y=y, floor=wf.CALIB_FLOOR)
    if "proba_raw_sexdom" in df.columns and wf.CALIB_SEXDOM.exists():
        calib = json.loads(wf.CALIB_SEXDOM.read_text(encoding="utf-8"))
        x = np.asarray(calib.get("isotonic", {}).get("x", []), dtype=float)
        y = np.asarray(calib.get("isotonic", {}).get("y", []), dtype=float)
        p_raw = pd.to_numeric(df["proba_raw_sexdom"], errors="coerce").to_numpy(dtype=float)
        df["proba_cal_sexdom"] = wf._apply_isotonic_vec(p_raw, x=x, y=y, floor=wf.CALIB_FLOOR)
    return df


def _merge_region(df: pd.DataFrame) -> pd.DataFrame:
    if not REGION_PRED.exists():
        raise FileNotFoundError(str(REGION_PRED))
    r = pd.read_csv(REGION_PRED, usecols=["ID Aposta", "region_pred"])
    r = r.rename(columns={"region_pred": "region_evt"})
    out = df.merge(r, how="left", on="ID Aposta")
    out["region_evt"] = out["region_evt"].astype("string").fillna("desconhecida").astype(str)
    return out


def _stake0(stake_frac: float, alpha: float) -> float:
    return float(wf.BANKROLL) * float(stake_frac) * float(alpha)


def _apply_rules_with_mult(
    df_any: pd.DataFrame,
    rules_df: pd.DataFrame,
    alpha: float,
    mult_by_rule_region: Dict[Tuple[str, str], float],
    block_by_rule: Dict[str, Set[str]] | None = None,
) -> pd.DataFrame:
    """
    Aplica regras baseline com alpha e multiplicadores por região.
    mult_by_rule_region key: (rule_key, region) -> multiplier
    """
    rows = []
    for _, rr in rules_df.iterrows():
        if str(rr.get("status")) != "ok":
            continue
        stake_frac = float(rr.get("stake_frac", 0.0))
        if stake_frac <= 0:
            continue
        bt = str(rr["bet_type"])
        dow = str(rr["dow_pt"])
        sc = str(rr["score_col"])
        cutoff = float(rr["cutoff"])
        rk = str(rr.get("rule_key", f"{bt}|{dow}"))

        x = df_any[(df_any["bet_type"] == bt) & (df_any["dow_pt"] == dow)].copy()
        if x.empty or sc not in x.columns:
            continue
        score = pd.to_numeric(x[sc], errors="coerce").to_numpy(float)
        roi2 = x["roi_cap2"].to_numpy(float)
        cap = x["house_cap"].to_numpy(float)
        reg = x["region_evt"].astype(str).to_numpy()
        m = np.isfinite(score) & (score >= cutoff) & np.isfinite(roi2) & np.isfinite(cap) & (cap > 0)
        if block_by_rule is not None and rk in block_by_rule:
            block = block_by_rule[rk]
            if block:
                m = m & (~np.isin(reg, list(block)))
        if not np.any(m):
            continue
        x = x.iloc[np.where(m)[0]].copy()

        # multiplicador por região (default=1)
        mult = np.array([float(mult_by_rule_region.get((rk, r), 1.0)) for r in x["region_evt"].astype(str).tolist()], dtype=float)
        mult = np.clip(mult, M_MIN, M_MAX)

        stake0 = _stake0(stake_frac, alpha)
        stake_eff = np.minimum(stake0 * mult, pd.to_numeric(x["house_cap"], errors="coerce").to_numpy(float))
        x["stake_eff"] = stake_eff
        x["profit_cap2"] = x["stake_eff"].to_numpy(float) * x["roi_cap2"].to_numpy(float)
        x["rule_key"] = rk
        rows.append(x[["date", "week", "stake_eff", "profit_cap2", "rule_key", "region_evt"]])

    if not rows:
        return df_any.iloc[:0].copy()
    return pd.concat(rows, axis=0, ignore_index=True)


def _constraints_ok(bets: pd.DataFrame) -> bool:
    if bets.empty:
        return True
    stake_day = bets.groupby("date")["stake_eff"].sum().to_numpy(float)
    pnl_day = bets.groupby("date")["profit_cap2"].sum().to_numpy(float)
    p80_exp = float(np.quantile(stake_day, wf.DAILY_EXPOSURE_Q)) if stake_day.size else 0.0
    daily_var = float(np.quantile(pnl_day, wf.DAILY_VAR_Q)) if pnl_day.size else 0.0
    p_dd = float((pnl_day <= (-wf.MAX_DAILY_DRAWDOWN_FRAC * wf.BANKROLL)).mean()) if pnl_day.size else 0.0
    if p80_exp > wf.MAX_DAILY_EXPOSURE_FRAC_Q * wf.BANKROLL:
        return False
    if daily_var < -wf.MAX_DAILY_DRAWDOWN_FRAC * wf.BANKROLL:
        return False
    if p_dd > wf.MAX_P_DAILY_DD:
        return False
    return True


def _find_alpha_coarse(
    df_train: pd.DataFrame,
    rules_df: pd.DataFrame,
    mult_by_rule_region: Dict[Tuple[str, str], float],
    block_by_rule: Dict[str, Set[str]] | None,
) -> float:
    for a in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
        bets = _apply_rules_with_mult(df_train, rules_df, alpha=a, mult_by_rule_region=mult_by_rule_region, block_by_rule=block_by_rule)
        if _constraints_ok(bets):
            return float(a)
    return 0.0


def _compute_block_by_rule(df_train: pd.DataFrame, rules_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Block-bad regions in train, per rule_key."""
    out: Dict[str, Set[str]] = {}
    for _, rr in rules_df.iterrows():
        if str(rr.get("status")) != "ok":
            continue
        stake_frac = float(rr.get("stake_frac", 0.0))
        if stake_frac <= 0:
            continue
        bt = str(rr["bet_type"])
        dow = str(rr["dow_pt"])
        sc = str(rr["score_col"])
        cutoff = float(rr["cutoff"])
        rk = str(rr.get("rule_key", f"{bt}|{dow}"))

        x = df_train[(df_train["bet_type"] == bt) & (df_train["dow_pt"] == dow)].copy()
        if x.empty or sc not in x.columns:
            continue
        score = pd.to_numeric(x[sc], errors="coerce").to_numpy(float)
        roi2 = x["roi_cap2"].to_numpy(float)
        reg = x["region_evt"].astype(str).to_numpy()
        m = np.isfinite(score) & (score >= cutoff) & np.isfinite(roi2)
        if not np.any(m):
            continue
        xt = pd.DataFrame({"region": reg[m], "roi": roi2[m]})
        by = xt.groupby("region", as_index=False).agg(n=("roi", "size"), mean_roi=("roi", "mean"))
        block = set(by.loc[(by["n"] >= int(BLOCK_MIN_N)) & (by["mean_roi"] <= float(BAD_MEAN_ROI_TH)), "region"].astype(str).tolist())
        if block:
            out[rk] = block
    return out


def _compute_multipliers(df_train: pd.DataFrame, rules_df: pd.DataFrame, block_by_rule: Dict[str, Set[str]] | None) -> Tuple[Dict[Tuple[str, str], float], pd.DataFrame]:
    """
    Estima multiplicadores por (rule_key, region) no treino.
    - Usa apenas apostas que passariam no cutoff.
    - Se block_by_rule presente, exclui regiões bloqueadas (para não upstakar o que seria removido).
    """
    rows = []
    mult: Dict[Tuple[str, str], float] = {}

    for _, rr in rules_df.iterrows():
        if str(rr.get("status")) != "ok":
            continue
        stake_frac = float(rr.get("stake_frac", 0.0))
        if stake_frac <= 0:
            continue
        bt = str(rr["bet_type"])
        dow = str(rr["dow_pt"])
        sc = str(rr["score_col"])
        cutoff = float(rr["cutoff"])
        rk = str(rr.get("rule_key", f"{bt}|{dow}"))

        x = df_train[(df_train["bet_type"] == bt) & (df_train["dow_pt"] == dow)].copy()
        if x.empty or sc not in x.columns:
            continue
        score = pd.to_numeric(x[sc], errors="coerce").to_numpy(float)
        roi2 = x["roi_cap2"].to_numpy(float)
        reg = x["region_evt"].astype(str).to_numpy()
        m = np.isfinite(score) & (score >= cutoff) & np.isfinite(roi2)
        if not np.any(m):
            continue
        if block_by_rule is not None and rk in block_by_rule and block_by_rule[rk]:
            m = m & (~np.isin(reg, list(block_by_rule[rk])))
        if not np.any(m):
            continue

        # referência do segmento
        mu_all = float(np.mean(roi2[m]))

        xt = pd.DataFrame({"region": reg[m], "roi": roi2[m]})
        by = xt.groupby("region", as_index=False).agg(n=("roi", "size"), mean_roi=("roi", "mean"))
        for _, br in by.iterrows():
            region = str(br["region"])
            n = int(br["n"])
            mu_r = float(br["mean_roi"])
            # mínimo de evidência
            if n < int(MIN_N_PER_REGION):
                mr = 1.0
            else:
                mr = float(np.clip(1.0 + float(GAMMA) * float(mu_r - mu_all), M_MIN, M_MAX))
            mult[(rk, region)] = mr
            rows.append({"rule_key": rk, "bet_type": bt, "dow_pt": dow, "region": region, "n": n, "mean_roi": mu_r, "mu_all": mu_all, "multiplier": mr})

    return mult, pd.DataFrame(rows)


def _run_variant(name: str, use_blockbad: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])
    df["week"] = wf.week_key(df["BIA_ApostaUTC"]).astype(str)
    df["date"] = wf.date_key(df["BIA_ApostaUTC"])
    df["house_cap"] = df["house_cap"].apply(wf.safe_cap)
    df["roi_raw"] = pd.to_numeric(df["roi_calc"], errors="coerce").astype(float)
    df["roi_cap2"] = np.minimum(df["roi_raw"].to_numpy(dtype=float), 2.0)
    df = _apply_isotonic_cols(df)
    df = _merge_region(df)

    base_rules = pd.read_csv(WF_RULES)
    weeks = sorted(df["week"].unique().tolist())

    weekly_rows = []
    mult_rows = []
    blocked_rows = []

    for w_test, rw in base_rules.groupby("test_week", as_index=False):
        w_test = str(w_test)
        if w_test not in weeks:
            continue
        i = weeks.index(w_test)
        train_weeks = weeks[max(0, i - TRAIN_WINDOW_WEEKS) : i]
        df_train = df[df["week"].isin(train_weeks)].copy()
        df_test = df[df["week"] == w_test].copy()
        if df_test.empty:
            continue

        # bloqueios (se habilitado)
        block_by_rule = _compute_block_by_rule(df_train, rw) if use_blockbad and BLOCK_BAD_REGIONS else None
        if block_by_rule:
            for rk, s in block_by_rule.items():
                blocked_rows.append({"week": w_test, "rule_key": rk, "blocked_regions": ",".join(sorted(s))})

        # multiplicadores por região
        mult_by, mult_df = _compute_multipliers(df_train, rw, block_by_rule=block_by_rule)
        if not mult_df.empty:
            mult_df = mult_df.copy()
            mult_df["week"] = w_test
            mult_rows.append(mult_df)

        # alpha (coarse) respeitando constraints com upstaking
        alpha = _find_alpha_coarse(df_train, rw, mult_by_rule_region=mult_by, block_by_rule=block_by_rule)

        bets = _apply_rules_with_mult(df_test, rw, alpha=alpha, mult_by_rule_region=mult_by, block_by_rule=block_by_rule)
        stake_sum = float(bets["stake_eff"].sum()) if len(bets) else 0.0
        pnl_sum = float(bets["profit_cap2"].sum()) if len(bets) else 0.0
        n_bets = int(len(bets))
        weekly_rows.append({"week": w_test, "alpha_effective": float(alpha), "n_bets": n_bets, "stake_usd": stake_sum, "profit_cap2_usd": pnl_sum, "roi_on_stake_cap2": float(pnl_sum / stake_sum) if stake_sum > 0 else float("nan")})

    wk = pd.DataFrame(weekly_rows).sort_values("week")
    stake = float(wk["stake_usd"].sum())
    pnl = float(wk["profit_cap2_usd"].sum())
    summ = pd.DataFrame([{"name": name, "profit_cap2_total": pnl, "stake_total": stake, "roi_total_cap2": (pnl / stake) if stake > 0 else float("nan"), "weeks": int(len(wk)), "weeks_with_stake": int((wk["stake_usd"] > 0).sum())}])
    mult_out = pd.concat(mult_rows, axis=0, ignore_index=True) if mult_rows else pd.DataFrame()
    blk_out = pd.DataFrame(blocked_rows)
    return wk, summ, mult_out, blk_out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not WF_RULES.exists() or not WF_WEEKLY.exists():
        raise FileNotFoundError("Arquivos do walk-forward base não encontrados.")

    # upstake only
    wk1, s1, mult1, blk1 = _run_variant("region_upstake", use_blockbad=False)
    wk1.to_csv(OUT_DIR / "oos_walkforward_region_upstake_exantepred_weekly.csv", index=False)
    s1.to_csv(OUT_DIR / "oos_walkforward_region_upstake_exantepred_summary.csv", index=False)
    mult1.to_csv(OUT_DIR / "oos_walkforward_region_upstake_exantepred_multipliers.csv", index=False)

    # blockbad + upstake
    wk2, s2, mult2, blk2 = _run_variant("region_blockbad_upstake", use_blockbad=True)
    wk2.to_csv(OUT_DIR / "oos_walkforward_region_blockbad_upstake_exantepred_weekly.csv", index=False)
    s2.to_csv(OUT_DIR / "oos_walkforward_region_blockbad_upstake_exantepred_summary.csv", index=False)
    mult2.to_csv(OUT_DIR / "oos_walkforward_region_blockbad_upstake_exantepred_multipliers.csv", index=False)
    blk2.to_csv(OUT_DIR / "oos_walkforward_region_blockbad_upstake_exantepred_blocked_regions.csv", index=False)

    # quick comparison vs baseline and blockbad-only
    base = pd.read_csv(WF_WEEKLY)
    bb = pd.read_csv(OUT_DIR / "oos_walkforward_region_gating_exantepred_blockbad_weekly.csv") if (OUT_DIR / "oos_walkforward_region_gating_exantepred_blockbad_weekly.csv").exists() else pd.DataFrame()
    base_pnl = float(pd.to_numeric(base["profit_cap2_usd"], errors="coerce").sum())
    base_stake = float(pd.to_numeric(base["stake_usd"], errors="coerce").sum())
    print("baseline", base_pnl, base_stake, base_pnl / base_stake if base_stake > 0 else float("nan"))
    print("upstake_only", float(wk1["profit_cap2_usd"].sum()), float(wk1["stake_usd"].sum()))
    print("blockbad+upstake", float(wk2["profit_cap2_usd"].sum()), float(wk2["stake_usd"].sum()))
    if not bb.empty:
        print("blockbad_only", float(pd.to_numeric(bb["profit_cap2_usd"], errors="coerce").sum()), float(pd.to_numeric(bb["stake_usd"], errors="coerce").sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

