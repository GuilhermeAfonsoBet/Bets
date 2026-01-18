#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste OOS: pós-filtro de combinações usando ROI "debiased" (apenas para análise).

Ideia:
- Mantém as regras θ_t do global_bayes (cutoff/stake_frac/alpha) geradas pelo WF.
- Para cada semana t, estima (apenas com histórico passado) o ROI esperado realizado por combinação:
    ROI_debiased ≈ ROI_pred_train + E[error_roi], onde error_roi = ROI_real - ROI_pred_train.
- Se ROI_debiased <= 0, desligamos a combinação na semana t (stake=0).

Obs:
- Isso NÃO é re-otimização (não muda cutoffs/stakes), é só um gating heurístico.
- O objetivo é avaliar se remover combinações "estruturalmente negativas" melhora o OOS.

Saídas:
- analysis_proba_raw/pro_portfolio_all/oos_postfilter_debiased_roi_<MODE>_weekly.csv
- analysis_proba_raw/pro_portfolio_all/oos_postfilter_debiased_roi_<MODE>_summary.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import json
import numpy as np
import pandas as pd


OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")
SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")
# Modo do WF para usar como base (regras θ_t). Ex.: \"global_bayes_roll12_robust\".
MODE = "global_bayes_roll12_robust"
WF_RULES = OUT_DIR / f"oos_walkforward_{MODE}_selected_rules.csv"
WF_WEEKLY = OUT_DIR / f"oos_walkforward_{MODE}_weekly.csv"

CALIB_SEXDOM = Path("/workspace/clv_calib_SexDom.json")
CALIB_FLOOR_SEXDOM = 0.005
CALIB_SEGQUI = Path("/workspace/clv_calib_SegQui.json")
CALIB_FLOOR_SEGQUI = 0.005

BANKROLL = 2300.0
SEED = 7
MIN_TOTAL_OBS_FOR_SHRINK = 8


def _apply_isotonic_vec(p: np.ndarray, x: np.ndarray, y: np.ndarray, floor: float | None) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if x.size and y.size and x.size == y.size:
        out = np.interp(p, x, y, left=float(y[0]), right=float(y[-1]))
    else:
        out = p.copy()
    if floor is not None:
        out = np.maximum(out, float(floor))
    return np.clip(out, 0.0, 1.0)


def _empirical_bayes_shrink(means: np.ndarray, se2: np.ndarray) -> Tuple[float, float, np.ndarray]:
    m = np.asarray(means, dtype=float)
    v = np.asarray(se2, dtype=float)
    ok = np.isfinite(m) & np.isfinite(v) & (v > 0)
    m = m[ok]
    v = v[ok]
    if m.size == 0:
        return float("nan"), 0.0, np.array([], dtype=float)
    w = 1.0 / v
    mu0 = float(np.sum(w * m) / np.sum(w))
    var_m = float(np.var(m, ddof=1)) if m.size > 1 else 0.0
    tau2 = float(max(0.0, var_m - float(np.mean(v))))
    if tau2 <= 1e-12:
        post = np.full(m.size, mu0, dtype=float)
    else:
        post = (m / v + mu0 / tau2) / (1.0 / v + 1.0 / tau2)
    return mu0, tau2, post


def _apply_single_rule(df: pd.DataFrame, bet_type: str, dow: str, score_col: str, cutoff: float, stake_frac: float, alpha: float) -> pd.DataFrame:
    if stake_frac <= 0:
        return df.iloc[:0].copy()
    x = df[(df["bet_type"] == bet_type) & (df["dow_pt"] == dow)].copy()
    if x.empty:
        return x
    score = pd.to_numeric(x[score_col], errors="coerce").to_numpy(dtype=float)
    roi2 = x["roi_cap2"].to_numpy(dtype=float)
    m = np.isfinite(score) & (score >= float(cutoff)) & np.isfinite(roi2)
    if not np.any(m):
        return x.iloc[:0].copy()
    x = x.iloc[np.where(m)[0]].copy()
    stake0 = BANKROLL * float(stake_frac) * float(alpha)
    x["stake_eff"] = np.minimum(stake0, x["house_cap"].to_numpy(dtype=float))
    x["profit_cap2"] = x["stake_eff"].to_numpy(dtype=float) * x["roi_cap2"].to_numpy(dtype=float)
    x["rule_key"] = f"{bet_type}|{dow}"
    return x[["week", "stake_eff", "profit_cap2", "rule_key"]]


def _roi_pred_weekly_mean(df_train: pd.DataFrame, rule_row: pd.Series, train_weeks: List[str]) -> float:
    if not train_weeks:
        return 0.0
    bets = _apply_single_rule(
        df_train,
        bet_type=str(rule_row["bet_type"]),
        dow=str(rule_row["dow_pt"]),
        score_col=str(rule_row["score_col"]),
        cutoff=float(rule_row["cutoff"]),
        stake_frac=float(rule_row["stake_frac"]),
        alpha=float(rule_row["alpha_global"]),
    )
    if bets.empty:
        return 0.0
    g = bets.groupby("week", as_index=False).agg(stake=("stake_eff", "sum"), pnl=("profit_cap2", "sum"))
    gm = g.set_index("week").reindex(train_weeks, fill_value=0.0)
    stake = gm["stake"].to_numpy(dtype=float)
    pnl = gm["pnl"].to_numpy(dtype=float)
    roi_w = np.zeros_like(stake, dtype=float)
    np.divide(pnl, stake, out=roi_w, where=(stake > 0))
    return float(np.mean(roi_w))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rules = pd.read_csv(WF_RULES)
    wf_week = pd.read_csv(WF_WEEKLY)

    df = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])
    if "roi_calc" not in df.columns:
        raise KeyError("Coluna roi_calc ausente. Regerar scored_dedup_proba_raw_all.csv antes de rodar este script.")
    df["roi_raw"] = pd.to_numeric(df["roi_calc"], errors="coerce").astype(float)
    df["roi_cap2"] = np.minimum(df["roi_raw"].to_numpy(dtype=float), 2.0)
    df["house_cap"] = pd.to_numeric(df["house_cap"], errors="coerce").astype(float)
    df["week"] = pd.to_datetime(df["BIA_ApostaUTC"]).dt.to_period("W-SUN").astype(str)

    # garantir colunas calibradas se regras referenciam proba_cal_*
    if "proba_cal_sexdom" not in df.columns:
        df["proba_cal_sexdom"] = np.nan
    if "proba_raw_sexdom" in df.columns and CALIB_SEXDOM.exists():
        try:
            calib = json.loads(CALIB_SEXDOM.read_text(encoding="utf-8"))
            x = np.asarray(calib.get("isotonic", {}).get("x", []), dtype=float)
            y = np.asarray(calib.get("isotonic", {}).get("y", []), dtype=float)
            p_raw = pd.to_numeric(df["proba_raw_sexdom"], errors="coerce").to_numpy(dtype=float)
            df["proba_cal_sexdom"] = _apply_isotonic_vec(p_raw, x=x, y=y, floor=CALIB_FLOOR_SEXDOM)
        except Exception:
            pass

    if "proba_cal_segqui" not in df.columns:
        df["proba_cal_segqui"] = np.nan
    if "proba_raw_segqui" in df.columns and CALIB_SEGQUI.exists():
        try:
            calib = json.loads(CALIB_SEGQUI.read_text(encoding="utf-8"))
            x = np.asarray(calib.get("isotonic", {}).get("x", []), dtype=float)
            y = np.asarray(calib.get("isotonic", {}).get("y", []), dtype=float)
            p_raw = pd.to_numeric(df["proba_raw_segqui"], errors="coerce").to_numpy(dtype=float)
            df["proba_cal_segqui"] = _apply_isotonic_vec(p_raw, x=x, y=y, floor=CALIB_FLOOR_SEGQUI)
        except Exception:
            pass

    weeks_sorted = sorted(df["week"].unique().tolist())
    week_to_i = {w: i for i, w in enumerate(weeks_sorted)}
    rng = np.random.default_rng(SEED)

    # histórico de erros por regra (somente passado)
    err_hist: Dict[str, List[float]] = {}

    out_rows = []
    gate_rows = []

    for w_test in wf_week["week"].astype(str).tolist():
        if w_test not in week_to_i:
            continue
        i = week_to_i[w_test]
        train_weeks = weeks_sorted[:i]
        df_train = df[df["week"].isin(train_weeks)].copy()
        df_test = df[df["week"] == w_test].copy()

        rw = rules[rules["test_week"].astype(str) == w_test].copy()
        if rw.empty:
            continue
        alpha = float(rw["alpha_global"].iloc[0]) if "alpha_global" in rw.columns else 1.0

        # shrinkage bias estimate using only past errors
        all_obs = np.array([e for v in err_hist.values() for e in v if np.isfinite(float(e))], dtype=float)
        bias_map: Dict[str, float] = {}
        mu0 = float("nan")
        tau2 = 0.0
        if int(all_obs.size) >= MIN_TOTAL_OBS_FOR_SHRINK:
            global_var = float(np.var(all_obs, ddof=1)) if all_obs.size > 1 else 0.0
            keys = []
            means = []
            se2 = []
            for k, vals in err_hist.items():
                a = np.asarray(vals, dtype=float)
                a = a[np.isfinite(a)]
                if a.size == 0:
                    continue
                v = float(np.var(a, ddof=1)) if a.size > 1 else global_var
                v = float(v if v > 1e-12 else 1e-4)
                keys.append(k)
                means.append(float(np.mean(a)))
                se2.append(float(v / a.size))
            if len(keys) >= 2:
                mu0, tau2, post = _empirical_bayes_shrink(np.asarray(means), np.asarray(se2))
                for k, pm in zip(keys, post):
                    bias_map[k] = float(pm)

        # aplicar gating na semana teste
        pnl_sum = 0.0
        stake_sum = 0.0
        enabled = 0
        disabled = 0

        for _, r in rw.iterrows():
            if str(r.get("status")) != "ok" or float(r.get("stake_frac", 0.0)) <= 0:
                continue
            rk = str(r.get("rule_key", f"{r['bet_type']}|{r['dow_pt']}"))
            roi_pred = _roi_pred_weekly_mean(df_train, r, train_weeks=train_weeks)
            bias_est = float(bias_map.get(rk, 0.0)) if np.isfinite(mu0) else 0.0
            roi_debiased = float(roi_pred + bias_est)
            keep = bool(roi_debiased > 0)
            gate_rows.append(
                {
                    "week": w_test,
                    "rule_key": rk,
                    "roi_pred_train": roi_pred,
                    "bias_est": bias_est,
                    "roi_debiased": roi_debiased,
                    "keep": int(keep),
                    "shrink_mu0": mu0,
                    "shrink_tau2": tau2,
                }
            )
            if not keep:
                disabled += 1
                continue
            enabled += 1
            bets_test = _apply_single_rule(df_test, str(r["bet_type"]), str(r["dow_pt"]), str(r["score_col"]), float(r["cutoff"]), float(r["stake_frac"]), alpha=float(alpha))
            pnl_sum += float(bets_test["profit_cap2"].sum()) if not bets_test.empty else 0.0
            stake_sum += float(bets_test["stake_eff"].sum()) if not bets_test.empty else 0.0

        out_rows.append(
            {
                "week": w_test,
                "alpha_global": float(alpha),
                "n_enabled_rules": int(enabled),
                "n_disabled_rules": int(disabled),
                "stake_usd": float(stake_sum),
                "profit_cap2_usd": float(pnl_sum),
                "roi_on_stake_cap2": float(pnl_sum / stake_sum) if stake_sum > 0 else float("nan"),
            }
        )

        # atualizar histórico de erro com base nas regras originais (mesmo que desabilitadas no gating)
        for _, r in rw.iterrows():
            if str(r.get("status")) != "ok" or float(r.get("stake_frac", 0.0)) <= 0:
                continue
            rk = str(r.get("rule_key", f"{r['bet_type']}|{r['dow_pt']}"))
            roi_pred = _roi_pred_weekly_mean(df_train, r, train_weeks=train_weeks)
            bets_test = _apply_single_rule(df_test, str(r["bet_type"]), str(r["dow_pt"]), str(r["score_col"]), float(r["cutoff"]), float(r["stake_frac"]), alpha=float(alpha))
            stake_t = float(bets_test["stake_eff"].sum()) if not bets_test.empty else 0.0
            pnl_t = float(bets_test["profit_cap2"].sum()) if not bets_test.empty else 0.0
            if stake_t <= 0:
                continue
            roi_real = float(pnl_t / stake_t)
            err_roi = float(roi_real - roi_pred)
            err_hist.setdefault(rk, []).append(err_roi)

    out = pd.DataFrame(out_rows)
    out_path = OUT_DIR / f"oos_postfilter_debiased_roi_{MODE}_weekly.csv"
    out.to_csv(out_path, index=False)

    gates = pd.DataFrame(gate_rows)
    gates.to_csv(OUT_DIR / f"oos_postfilter_debiased_roi_{MODE}_gating.csv", index=False)

    # comparação vs OOS original
    base = wf_week[["week", "stake_usd", "profit_cap2_usd"]].copy()
    base = base.rename(columns={"stake_usd": "stake_base", "profit_cap2_usd": "pnl_base"})
    comp = base.merge(out[["week", "stake_usd", "profit_cap2_usd"]].rename(columns={"stake_usd": "stake_gate", "profit_cap2_usd": "pnl_gate"}), on="week", how="left")
    comp["delta_pnl"] = comp["pnl_gate"] - comp["pnl_base"]
    comp.to_csv(OUT_DIR / f"oos_postfilter_debiased_roi_{MODE}_comparison.csv", index=False)

    def stats(x: np.ndarray) -> Tuple[float, float, float]:
        x = np.asarray(x, dtype=float)
        m = float(np.mean(x))
        s = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
        sh = float((m * 52.0) / (s * np.sqrt(52.0))) if s > 0 else float("nan")
        return m, s, sh

    m0, s0, sh0 = stats(comp["pnl_base"].to_numpy(dtype=float))
    m1, s1, sh1 = stats(comp["pnl_gate"].to_numpy(dtype=float))

    lines = []
    lines.append("## OOS pós-filtro por ROI debiased (gating)\n\n")
    lines.append("Heurística: em cada semana, desligar combinações com ROI_debiased<=0 (estimado apenas com passado).\n\n")
    lines.append(f"- Baseline mean/sem={m0:.1f}, std={s0:.1f}, Sharpe_ann={sh0:.3f}\n")
    lines.append(f"- Gating   mean/sem={m1:.1f}, std={s1:.1f}, Sharpe_ann={sh1:.3f}\n\n")
    lines.append("Arquivos:\n")
    lines.append(f"- `analysis_proba_raw/pro_portfolio_all/oos_postfilter_debiased_roi_{MODE}_weekly.csv`\n")
    lines.append(f"- `analysis_proba_raw/pro_portfolio_all/oos_postfilter_debiased_roi_{MODE}_gating.csv`\n")
    lines.append(f"- `analysis_proba_raw/pro_portfolio_all/oos_postfilter_debiased_roi_{MODE}_comparison.csv`\n")
    (OUT_DIR / f"oos_postfilter_debiased_roi_{MODE}_summary.md").write_text("".join(lines), encoding="utf-8")

    print(str(out_path))
    print(str(OUT_DIR / f"oos_postfilter_debiased_roi_{MODE}_summary.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

