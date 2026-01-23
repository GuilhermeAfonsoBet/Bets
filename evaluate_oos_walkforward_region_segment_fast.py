#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimento FAST (exploratório): otimização walk-forward incluindo Região como dimensão do portfólio.

Objetivo:
- Comparar, apenas nas últimas N semanas (6–8 por default), o baseline (DoW×Tipo) vs
  o modelo com segmentação (DoW×Tipo×Região), usando score atual por dia.

Premissas:
- Região é uma coluna ex-ante inferida (predita offline) em `region_exante_pred.csv`:
  - usamos `region_pred` quando `region_pred_pmax >= pmax_min`
  - caso contrário, tratamos como "desconhecida"

Acelerações (modo fast):
- Treino com janela fixa (roll) de TRAIN_WINDOW_WEEKS (default 12) em vez de expanding.
- Reduz o custo do Bayes e simplifica grids de cutoff/stake.
- Aumenta mínimos de evidência para reduzir segmentos inviáveis.
- Seleção de alpha_global por grid (coarse) em vez de busca binária.

Saídas:
- /workspace/analysis_proba_raw/pro_portfolio_all/oos_walkforward_regionseg_fast_summary.csv
- /workspace/analysis_proba_raw/pro_portfolio_all/oos_walkforward_regionseg_fast_weekly.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import evaluate_oos_walkforward_strategy as wf


OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")
SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")
REGION_PRED = OUT_DIR / "region_exante_pred.csv"


TRAIN_WINDOW_WEEKS = 12


@dataclass(frozen=True)
class RuleR:
    bet_type: str
    dow: str
    region: str
    score_col: str
    cutoff: float
    stake_frac: float
    status: str


def _ensure_calibrated_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "proba_cal_segqui" not in df.columns:
        df["proba_cal_segqui"] = np.nan
    if "proba_cal_sexdom" not in df.columns:
        df["proba_cal_sexdom"] = np.nan

    try:
        if "proba_raw_segqui" in df.columns and wf.CALIB_SEGQUI.exists():
            calib = json.loads(wf.CALIB_SEGQUI.read_text(encoding="utf-8"))
            x = np.asarray(calib.get("isotonic", {}).get("x", []), dtype=float)
            y = np.asarray(calib.get("isotonic", {}).get("y", []), dtype=float)
            p_raw = pd.to_numeric(df["proba_raw_segqui"], errors="coerce").to_numpy(dtype=float)
            df["proba_cal_segqui"] = wf._apply_isotonic_vec(p_raw, x=x, y=y, floor=wf.CALIB_FLOOR)
    except Exception:
        pass

    try:
        if "proba_raw_sexdom" in df.columns and wf.CALIB_SEXDOM.exists():
            calib = json.loads(wf.CALIB_SEXDOM.read_text(encoding="utf-8"))
            x = np.asarray(calib.get("isotonic", {}).get("x", []), dtype=float)
            y = np.asarray(calib.get("isotonic", {}).get("y", []), dtype=float)
            p_raw = pd.to_numeric(df["proba_raw_sexdom"], errors="coerce").to_numpy(dtype=float)
            df["proba_cal_sexdom"] = wf._apply_isotonic_vec(p_raw, x=x, y=y, floor=wf.CALIB_FLOOR)
    except Exception:
        pass

    return df


def _merge_region(df: pd.DataFrame, pmax_min: float) -> pd.DataFrame:
    df = df.copy()
    if not REGION_PRED.exists():
        raise FileNotFoundError(f"Arquivo ausente: {REGION_PRED}")
    r = pd.read_csv(REGION_PRED, usecols=["ID Aposta", "region_pred", "region_pred_pmax"])
    r["region_pred_pmax"] = pd.to_numeric(r["region_pred_pmax"], errors="coerce").astype(float)
    r["region_pred"] = r["region_pred"].astype("string").fillna("desconhecida").astype(str)
    r.loc[~np.isfinite(r["region_pred_pmax"].to_numpy(float)) | (r["region_pred_pmax"].to_numpy(float) < float(pmax_min)), "region_pred"] = "desconhecida"
    r = r.rename(columns={"region_pred": "region_evt"})
    out = df.merge(r[["ID Aposta", "region_evt"]], how="left", on="ID Aposta")
    out["region_evt"] = out["region_evt"].astype("string").fillna("desconhecida").astype(str)
    return out


def _apply_rules_region(df_any: pd.DataFrame, rules: Dict[str, RuleR], alpha: float) -> pd.DataFrame:
    rows = []
    for rule in rules.values():
        if rule.stake_frac <= 0:
            continue
        stake0 = wf.BANKROLL * float(rule.stake_frac) * float(alpha)
        if stake0 <= 0:
            continue
        x = df_any[(df_any["dow_pt"] == rule.dow) & (df_any["bet_type"] == rule.bet_type) & (df_any["region_evt"] == rule.region)].copy()
        if x.empty:
            continue
        score = pd.to_numeric(x[rule.score_col], errors="coerce").to_numpy(dtype=float)
        roi2 = x["roi_cap2"].to_numpy(dtype=float)
        m = np.isfinite(score) & (score >= float(rule.cutoff)) & np.isfinite(roi2)
        if not np.any(m):
            continue
        x = x.iloc[np.where(m)[0]].copy()
        x["stake_eff"] = np.minimum(stake0, x["house_cap"].to_numpy(dtype=float))
        x["profit_cap2"] = x["stake_eff"].to_numpy(dtype=float) * x["roi_cap2"].to_numpy(dtype=float)
        x["rule_key"] = f"{rule.bet_type}|{rule.dow}|{rule.region}"
        rows.append(x[["date", "week", "stake_eff", "profit_cap2", "rule_key"]])
    if not rows:
        return df_any.iloc[:0].copy()
    return pd.concat(rows, axis=0, ignore_index=True)


def _portfolio_constraints_ok_region(df_train: pd.DataFrame, rules: Dict[str, RuleR], alpha: float) -> bool:
    bets = _apply_rules_region(df_train, rules, alpha=alpha)
    if bets.empty:
        return True
    stake_day = bets.groupby("date")["stake_eff"].sum().to_numpy(dtype=float)
    pnl_day = bets.groupby("date")["profit_cap2"].sum().to_numpy(dtype=float)
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


def _find_alpha_fast(df_train: pd.DataFrame, rules: Dict[str, RuleR]) -> float:
    # grid coarse (rápido). assume monotonicidade aproximada.
    for a in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
        if _portfolio_constraints_ok_region(df_train, rules, alpha=a):
            return float(a)
    return 0.0


def _run_fast(
    df: pd.DataFrame,
    weeks: List[str],
    test_weeks: List[str],
    with_region: bool,
    bayes_n: int,
    cutoffs: np.ndarray,
    stake_fracs: np.ndarray,
    min_selected: int,
    min_bets_per_bin: int,
    min_seg_train_weeks: int,
) -> pd.DataFrame:
    # patch globals do wf (modo exploratório)
    wf.BAYES_N = int(bayes_n)
    wf.CUTOFFS = np.asarray(cutoffs, dtype=float)
    wf.STAKE_FRACS = np.asarray(stake_fracs, dtype=float)
    wf.MIN_SELECTED_BETS = int(min_selected)
    wf.MIN_BETS_PER_BIN = int(min_bets_per_bin)
    wf.MIN_SEG_TRAIN_WEEKS = int(min_seg_train_weeks)

    weekly_rows = []
    prev_rules: Dict[str, wf.Rule] = {}

    for w_test in test_weeks:
        i = weeks.index(w_test)
        train_weeks = weeks[max(0, i - TRAIN_WINDOW_WEEKS) : i]
        df_train = df[df["week"].astype(str).isin(train_weeks)].copy()
        df_test = df[df["week"].astype(str) == w_test].copy()
        if df_test.empty:
            continue

        if with_region:
            regions = sorted(df_train["region_evt"].astype(str).unique().tolist())
        else:
            regions = ["__all__"]

        rules_r: Dict[str, RuleR] = {}
        for bet_type in ("FT", "FH"):
            for dow in wf.WEEKDAY_PT:
                sc = wf.segment_score_col(dow)
                for reg in regions:
                    if with_region:
                        x = df_train[(df_train["dow_pt"] == dow) & (df_train["bet_type"] == bet_type) & (df_train["region_evt"] == reg)].copy()
                        key_prev = f"{bet_type}|{dow}|{reg}"
                    else:
                        x = df_train[(df_train["dow_pt"] == dow) & (df_train["bet_type"] == bet_type)].copy()
                        key_prev = f"{bet_type}|{dow}"

                    if x.empty or sc not in x.columns:
                        rule = RuleR(bet_type=bet_type, dow=dow, region=reg, score_col=sc, cutoff=1.0, stake_frac=0.0, status="no_data")
                    else:
                        # usa wf.optimize_segment_train (mesma lógica), com prev_rule opcional
                        prev = prev_rules.get(key_prev)
                        wr = wf.optimize_segment_train(x, sc, bayes_select=True, prev_rule=prev, roi_bias_adj=0.0)
                        rule = RuleR(bet_type=bet_type, dow=dow, region=reg, score_col=sc, cutoff=float(wr.cutoff), stake_frac=float(wr.stake_frac), status=str(wr.status))
                    rules_r[key_prev] = rule

        # alpha global (fast grid) respeitando as constraints globais
        if with_region:
            alpha = _find_alpha_fast(df_train, rules_r)
            bets = _apply_rules_region(df_test, rules_r, alpha=alpha)
        else:
            # baseline usa implementação do wf (mais simples e rápida)
            rules_base: Dict[str, wf.Rule] = {}
            for k, rr in rules_r.items():
                if rr.region != "__all__":
                    continue
                rules_base[k] = wf.Rule(bet_type=rr.bet_type, dow=rr.dow, score_col=rr.score_col, cutoff=rr.cutoff, stake_frac=rr.stake_frac, status=rr.status)
            alpha, _, _ = wf.find_alpha_global(df_train, rules_base)
            bets = wf.apply_rules_on_df(df_test, rules_base, alpha=float(alpha))

        stake_sum = float(bets["stake_eff"].sum()) if len(bets) else 0.0
        pnl_sum = float(bets["profit_cap2"].sum()) if len(bets) else 0.0
        n_bets = int(len(bets))
        weekly_rows.append(
            {
                "week": w_test,
                "alpha_global": float(alpha),
                "n_bets": n_bets,
                "stake_usd": stake_sum,
                "profit_cap2_usd": pnl_sum,
                "roi_on_stake_cap2": float(pnl_sum / stake_sum) if stake_sum > 0 else float("nan"),
            }
        )

        # atualizar prev_rules apenas para baseline (para histerese interna do wf) ou por compat
        if with_region:
            # criamos placeholder de wf.Rule para manter memória de prev_rule (cutoff/stake)
            prev_rules = {k: wf.Rule(bet_type=r.bet_type, dow=r.dow, score_col=r.score_col, cutoff=r.cutoff, stake_frac=r.stake_frac, status=r.status) for k, r in rules_r.items()}
        else:
            prev_rules = {k: wf.Rule(bet_type=r.bet_type, dow=r.dow, score_col=r.score_col, cutoff=r.cutoff, stake_frac=r.stake_frac, status=r.status) for k, r in rules_r.items()}

    return pd.DataFrame(weekly_rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--last-weeks", type=int, default=8, help="Rodar apenas as últimas N semanas (default: 8).")
    ap.add_argument("--train-window-weeks", type=int, default=12, help="Janela de treino (semanas) para o modo fast (default: 12).")
    ap.add_argument("--pmax-min", type=float, default=0.70, help="Limiar de confiança pmax para usar região (default: 0.70).")
    ap.add_argument("--bayes-n", type=int, default=2000, help="N de amostras Bayes (Dirichlet) no modo fast.")
    ap.add_argument("--min-selected-bets", type=int, default=10, help="Mínimo de apostas selecionadas por candidato (fast).")
    ap.add_argument("--min-bets-per-bin", type=int, default=40, help="Mínimo de apostas por bin (stability) (fast).")
    ap.add_argument("--min-seg-train-weeks", type=int, default=6, help="Mínimo de semanas no treino do segmento (fast).")
    ap.add_argument("--out-suffix", type=str, default="", help="Sufixo opcional para não sobrescrever saídas (ex.: _p75_strict).")
    args = ap.parse_args()

    global TRAIN_WINDOW_WEEKS
    TRAIN_WINDOW_WEEKS = int(args.train_window_weeks)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])
    df["house_cap"] = df["house_cap"].apply(wf.safe_cap)
    df["week"] = wf.week_key(df["BIA_ApostaUTC"]).astype(str)
    df["date"] = wf.date_key(df["BIA_ApostaUTC"])
    df["roi_raw"] = pd.to_numeric(df["roi_calc"], errors="coerce").astype(float)
    df["roi_cap2"] = np.minimum(df["roi_raw"].to_numpy(dtype=float), 2.0)
    df["roi_cap1"] = np.minimum(df["roi_raw"].to_numpy(dtype=float), 1.0)
    df = _ensure_calibrated_cols(df)
    df = _merge_region(df, pmax_min=float(args.pmax_min))

    weeks = sorted(df["week"].astype(str).unique().tolist())
    if len(weeks) < (wf.MIN_GLOBAL_TRAIN_WEEKS + 3):
        raise SystemExit(f"Poucas semanas no dataset: {len(weeks)}")

    last_n = int(max(1, args.last_weeks))
    test_weeks = weeks[-last_n:]
    # garantir que cada semana de teste tem pelo menos 1 janela de treino
    test_weeks = [w for w in test_weeks if weeks.index(w) >= 1]

    # grids simplificados (modo exploratório)
    cutoffs = np.round(np.arange(0.10, 0.91, 0.10), 2)
    stake_fracs = np.array([0.01, 0.02, 0.03, 0.04], dtype=float)

    wk_base = _run_fast(
        df,
        weeks=weeks,
        test_weeks=test_weeks,
        with_region=False,
        bayes_n=int(args.bayes_n),
        cutoffs=cutoffs,
        stake_fracs=stake_fracs,
        min_selected=int(args.min_selected_bets),
        min_bets_per_bin=int(args.min_bets_per_bin),
        min_seg_train_weeks=int(args.min_seg_train_weeks),
    )
    wk_reg = _run_fast(
        df,
        weeks=weeks,
        test_weeks=test_weeks,
        with_region=True,
        bayes_n=int(args.bayes_n),
        cutoffs=cutoffs,
        stake_fracs=stake_fracs,
        min_selected=int(args.min_selected_bets),
        min_bets_per_bin=int(args.min_bets_per_bin),
        min_seg_train_weeks=int(args.min_seg_train_weeks),
    )

    def summarize(name: str, wk: pd.DataFrame) -> Dict[str, float]:
        stake = float(pd.to_numeric(wk["stake_usd"], errors="coerce").sum()) if not wk.empty else 0.0
        pnl = float(pd.to_numeric(wk["profit_cap2_usd"], errors="coerce").sum()) if not wk.empty else 0.0
        return {
            "name": name,
            "weeks_total": int(len(wk)),
            "weeks_with_stake": int((pd.to_numeric(wk["stake_usd"], errors="coerce") > 0).sum()) if not wk.empty else 0,
            "stake_total": stake,
            "profit_cap2_total": pnl,
            "roi_total_cap2": float(pnl / stake) if stake > 0 else float("nan"),
        }

    summ = pd.DataFrame([summarize("baseline_fast", wk_base), summarize("regionseg_fast", wk_reg)])
    suffix = str(args.out_suffix).strip()
    out_sum = OUT_DIR / f"oos_walkforward_regionseg_fast{suffix}_summary.csv"
    out_wk = OUT_DIR / f"oos_walkforward_regionseg_fast{suffix}_weekly.csv"
    # salvar semanal com indicador
    wb = wk_base.copy()
    wb["variant"] = "baseline_fast"
    wr = wk_reg.copy()
    wr["variant"] = "regionseg_fast"
    pd.concat([wb, wr], axis=0, ignore_index=True).to_csv(out_wk, index=False)
    # registrar config no summary
    summ["last_weeks"] = int(last_n)
    summ["train_window_weeks"] = int(TRAIN_WINDOW_WEEKS)
    summ["pmax_min"] = float(args.pmax_min)
    summ["bayes_n"] = int(args.bayes_n)
    summ["min_selected_bets"] = int(args.min_selected_bets)
    summ["min_bets_per_bin"] = int(args.min_bets_per_bin)
    summ["min_seg_train_weeks"] = int(args.min_seg_train_weeks)
    summ.to_csv(out_sum, index=False)

    print(str(out_sum))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

