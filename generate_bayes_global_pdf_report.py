#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera PDF consolidado (mesa profissional) do portfólio escolhido: global_bayes.

Entrada (gerados previamente no projeto):
- portfolio_pro_all.json (portfólio fixo in-sample)
- oos_walkforward_global_bayes_weekly.csv
- oos_walkforward_global_bayes_daily.csv
- oos_walkforward_global_bayes_selected_rules.csv
- global_bayes_current_week_rules.csv
- portfolio_refined_global_bayes_full_comparison.csv

Saída:
- analysis_proba_raw/pro_portfolio_all/Relatorio_BayesGlobal_Mesa_Profissional_<YYYY-MM-DD>.pdf
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")
SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")

PORT_FIXED = OUT_DIR / "portfolio_pro_all.json"
WF_WEEKLY = OUT_DIR / "oos_walkforward_global_bayes_weekly.csv"
WF_DAILY = OUT_DIR / "oos_walkforward_global_bayes_daily.csv"
WF_RULES = OUT_DIR / "oos_walkforward_global_bayes_selected_rules.csv"
CUR_RULES = OUT_DIR / "global_bayes_current_week_rules.csv"
COMPARISON = OUT_DIR / "portfolio_refined_global_bayes_full_comparison.csv"

BANKROLL = 2300.0
# mesmos limites do otimizador
MAX_DAILY_EXPOSURE_FRAC_Q = 0.70
MAX_DAILY_DRAWDOWN_FRAC = 0.25
MAX_P_DAILY_DD = 0.10


def quantiles(a: np.ndarray, qs: List[float]) -> Dict[str, float]:
    x = np.asarray(a, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"q{int(q*100):02d}": float("nan") for q in qs}
    out = {}
    v = np.quantile(x, qs)
    for q, val in zip(qs, v):
        out[f"q{int(q*100):02d}"] = float(val)
    return out


def compute_weekly_stats(w: np.ndarray) -> Dict[str, float]:
    x = np.asarray(w, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    m = float(x.mean())
    s = float(x.std(ddof=1)) if x.size > 1 else 0.0
    med = float(np.median(x))
    pneg = float((x < 0).mean())
    ppos = float((x > 0).mean())
    sk = float(np.mean(((x - m) / s) ** 3)) if s > 0 else float("nan")
    q = quantiles(x, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    # sharpe (weekly) and annualized
    sharpe_w = float(m / s) if s > 0 else float("nan")
    sharpe_ann = float((m * 52.0) / (s * math.sqrt(52.0))) if s > 0 else float("nan")
    return {
        "n": int(x.size),
        "mean": m,
        "std": s,
        "median": med,
        "pneg": pneg,
        "ppos": ppos,
        "skew": sk,
        "sharpe_week": sharpe_w,
        "sharpe_annual": sharpe_ann,
        **q,
    }


def jaccard_instability(rules_df: pd.DataFrame) -> Dict[str, float]:
    r = rules_df.copy()
    r["active"] = (r["status"] == "ok") & (r["stake_frac"] > 0)
    weeks = r["test_week"].dropna().unique().tolist()
    weeks = list(weeks)
    if len(weeks) < 2:
        return {"jaccard_mean": float("nan")}
    # preserve original order in file (it is chronological in our outputs)
    # compute consecutive jaccards
    j = []
    for a, b in zip(weeks[:-1], weeks[1:]):
        A = set(r[(r["test_week"] == a) & (r["active"])]["rule_key"].astype(str).tolist())
        B = set(r[(r["test_week"] == b) & (r["active"])]["rule_key"].astype(str).tolist())
        inter = len(A & B)
        uni = len(A | B)
        j.append(inter / uni if uni else 1.0)
    arr = np.asarray(j, dtype=float)
    return {
        "n_pairs": int(arr.size),
        "jaccard_mean": float(arr.mean()),
        "jaccard_p10": float(np.quantile(arr, 0.10)),
        "jaccard_p50": float(np.quantile(arr, 0.50)),
        "jaccard_p90": float(np.quantile(arr, 0.90)),
    }


def rule_change_stats(rules_df: pd.DataFrame) -> pd.DataFrame:
    r = rules_df.copy()
    r["active"] = (r["status"] == "ok") & (r["stake_frac"] > 0)
    out = []
    for rk, g in r.sort_values(["rule_key", "test_week"]).groupby("rule_key"):
        ga = g[g["active"]].copy()
        if ga.shape[0] < 2:
            out.append({"rule_key": rk, "n_active_weeks": int(ga.shape[0]), "cutoff_change_rate": float("nan"), "stake_change_rate": float("nan"), "cutoff_std": float(ga["cutoff"].std(ddof=1)) if ga.shape[0] > 1 else 0.0, "stake_std": float(ga["stake_frac"].std(ddof=1)) if ga.shape[0] > 1 else 0.0})
            continue
        cut = ga["cutoff"].to_numpy(dtype=float)
        st = ga["stake_frac"].to_numpy(dtype=float)
        out.append(
            {
                "rule_key": rk,
                "n_active_weeks": int(ga.shape[0]),
                "cutoff_change_rate": float(np.mean(cut[1:] != cut[:-1])),
                "stake_change_rate": float(np.mean(st[1:] != st[:-1])),
                "cutoff_std": float(ga["cutoff"].std(ddof=1)) if ga.shape[0] > 1 else 0.0,
                "stake_std": float(ga["stake_frac"].std(ddof=1)) if ga.shape[0] > 1 else 0.0,
            }
        )
    return pd.DataFrame(out).sort_values(["n_active_weeks", "cutoff_change_rate"], ascending=[False, False])


def main() -> int:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

    today = date.today().isoformat()
    pdf_path = OUT_DIR / f"Relatorio_BayesGlobal_Mesa_Profissional_{today}.pdf"

    # load artifacts
    wf_week = pd.read_csv(WF_WEEKLY)
    wf_daily = pd.read_csv(WF_DAILY)
    wf_rules = pd.read_csv(WF_RULES)
    cur = pd.read_csv(CUR_RULES) if CUR_RULES.exists() else pd.DataFrame()
    comp = pd.read_csv(COMPARISON) if COMPARISON.exists() else pd.DataFrame()

    # core metrics for chosen policy (WF 16 weeks)
    w = wf_week["profit_cap2_usd"].to_numpy(dtype=float)
    wstats = compute_weekly_stats(w)
    stake_tot = float(wf_week["stake_usd"].sum())
    profit_tot = float(wf_week["profit_cap2_usd"].sum())
    roi_on_stake = float(profit_tot / stake_tot) if stake_tot > 0 else float("nan")
    mean_week = float(wstats.get("mean", float("nan")))
    roi_bank_week = float(mean_week / BANKROLL) if np.isfinite(mean_week) else float("nan")
    exp_month = float(mean_week * 4.33) if np.isfinite(mean_week) else float("nan")
    exp_year = float(mean_week * 52.0) if np.isfinite(mean_week) else float("nan")

    # daily risk in OOS (WF sample)
    ds = wf_daily["stake_usd"].to_numpy(dtype=float) if not wf_daily.empty else np.array([])
    dp = wf_daily["profit_cap2_usd"].to_numpy(dtype=float) if not wf_daily.empty else np.array([])
    p80_exp = float(np.quantile(ds, 0.80)) if ds.size else float("nan")
    var10 = float(np.quantile(dp, 0.10)) if dp.size else float("nan")
    p_dd = float(np.mean(dp <= (-0.25 * BANKROLL))) if dp.size else float("nan")

    # alpha summary
    alpha = wf_week["alpha_global"].to_numpy(dtype=float)
    alpha_mean = float(np.mean(alpha))
    alpha_p10 = float(np.quantile(alpha, 0.10))
    alpha_p50 = float(np.quantile(alpha, 0.50))
    alpha_p90 = float(np.quantile(alpha, 0.90))
    p_alpha_lt1 = float(np.mean(alpha < 0.999))

    # stability of rules
    jac = jaccard_instability(wf_rules)
    rc = rule_change_stats(wf_rules)

    # -------------------------
    # Extra: auditoria por semana
    # -------------------------
    r = wf_rules.copy()
    r["active"] = (r["status"] == "ok") & (r["stake_frac"] > 0)
    weekly_audit_rows = []
    for wk, g in r.groupby("test_week", sort=False):
        ga = g[g["active"]].copy()
        segs = ga["rule_key"].astype(str).tolist()
        segs_sorted = sorted(segs)
        weekly_audit_rows.append(
            {
                "week": str(wk),
                "alpha": float(ga["alpha_global"].iloc[0]) if (not ga.empty and "alpha_global" in ga.columns) else float("nan"),
                "n_active": int(len(segs_sorted)),
                "segments": ", ".join(segs_sorted),
            }
        )
    weekly_audit = pd.DataFrame(weekly_audit_rows)

    # -------------------------
    # Extra: operação no "máximo" (banca não limita stake)
    # - Mantém as mesmas regras por semana, mas usa stake_eff_max = house_cap.
    # - Calcula banca necessária para que stake0 = banca*stake_frac >= house_cap em (p95 e max).
    # -------------------------
    df_all = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])
    df_all["roi_raw"] = pd.to_numeric(df_all["ROI Real"], errors="coerce").astype(float)
    df_all["roi_cap2"] = np.minimum(df_all["roi_raw"].to_numpy(dtype=float), 2.0)
    df_all["house_cap"] = pd.to_numeric(df_all["house_cap"], errors="coerce").astype(float)
    df_all["week"] = pd.to_datetime(df_all["BIA_ApostaUTC"]).dt.to_period("W-SUN").astype(str)

    # only weeks evaluated in WF
    wf_weeks = set(wf_week["week"].astype(str).tolist())
    df_oos_wf = df_all[df_all["week"].astype(str).isin(wf_weeks)].copy()

    # build rules map per week
    max_rows = []
    bank_reqs = []
    for wk, g in r.groupby("test_week", sort=False):
        wk = str(wk)
        df_w = df_oos_wf[df_oos_wf["week"].astype(str) == wk].copy()
        if df_w.empty:
            continue
        for _, row in g.iterrows():
            if not bool(row["active"]):
                continue
            rk = str(row["rule_key"])
            bt = str(row["bet_type"])
            dow = str(row["dow_pt"])
            score_col = str(row["score_col"])
            cutoff = float(row["cutoff"])
            stake_frac = float(row["stake_frac"])

            x = df_w[(df_w["bet_type"] == bt) & (df_w["dow_pt"] == dow)].copy()
            if x.empty:
                continue
            score = pd.to_numeric(x[score_col], errors="coerce").to_numpy(dtype=float)
            roi2 = x["roi_cap2"].to_numpy(dtype=float)
            cap = x["house_cap"].to_numpy(dtype=float)
            m = np.isfinite(score) & (score >= cutoff) & np.isfinite(roi2) & np.isfinite(cap) & (cap > 0)
            if not np.any(m):
                continue
            cap_sel = cap[m]
            roi_sel = roi2[m]
            # stake atual (com banca=2300 e alpha da semana)
            alpha_used = float(row["alpha_global"]) if "alpha_global" in row and np.isfinite(float(row["alpha_global"])) else 1.0
            stake0 = BANKROLL * stake_frac * alpha_used
            stake_eff_cur = np.minimum(stake0, cap_sel)
            profit_cur = stake_eff_cur * roi_sel
            # stake máximo (banca não limita): stake_eff_max = cap
            stake_eff_max = cap_sel
            profit_max = stake_eff_max * roi_sel

            max_rows.append({"week": wk, "rule_key": rk, "stake_cur": float(stake_eff_cur.sum()), "profit_cur": float(profit_cur.sum()), "stake_max": float(stake_eff_max.sum()), "profit_max": float(profit_max.sum())})

            # banca necessária para não limitar (alpha=1): banca >= house_cap / stake_frac
            # (ignorando α, pois em banca grande α tende a 1 e o limitante vira o cap)
            bank_reqs.extend((cap_sel / max(stake_frac, 1e-9)).tolist())

    max_df = pd.DataFrame(max_rows)
    if not max_df.empty:
        weekly_max = max_df.groupby("week", as_index=False).agg(stake_cur=("stake_cur", "sum"), profit_cur=("profit_cur", "sum"), stake_max=("stake_max", "sum"), profit_max=("profit_max", "sum"))
        weekly_max["roi_cur"] = np.where(weekly_max["stake_cur"] > 0, weekly_max["profit_cur"] / weekly_max["stake_cur"], np.nan)
        weekly_max["roi_max"] = np.where(weekly_max["stake_max"] > 0, weekly_max["profit_max"] / weekly_max["stake_max"], np.nan)
        pnl_max = weekly_max["profit_max"].to_numpy(dtype=float)
        pnl_cur = weekly_max["profit_cur"].to_numpy(dtype=float)
        max_week_stats = compute_weekly_stats(pnl_max)
    else:
        weekly_max = pd.DataFrame()
        max_week_stats = {"n": 0}

    if bank_reqs:
        bank_reqs = np.asarray(bank_reqs, dtype=float)
        bank_p95 = float(np.quantile(bank_reqs, 0.95))
        bank_max = float(np.max(bank_reqs))
    else:
        bank_p95 = float("nan")
        bank_max = float("nan")

    # build document
    styles = getSampleStyleSheet()
    # helper: wrapped paragraph in table cells
    def P(txt: str) -> Paragraph:
        return Paragraph(str(txt).replace("\n", "<br/>"), styles["BodyText"])

    story: List = []

    story.append(Paragraph(f"<b>Relatório — Portfólio Bayes Global (mesa profissional)</b>", styles["Title"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(f"Data: <b>{today}</b>", styles["Normal"]))
    story.append(Paragraph(f"Workspace: <b>/workspace</b>", styles["Normal"]))
    story.append(Spacer(1, 0.4 * cm))

    # 1) description / methods
    story.append(Paragraph("<b>1. O que foi feito (método e objetivo)</b>", styles["Heading2"]))
    story.append(
        Paragraph(
            "Construímos e validamos uma estratégia de apostas por portfólio de combinações (DoW × tipo FT/FH × score ≥ cutoff × stake). "
            "O score é <b>proba_raw</b> (alinhado ao processo operacional). "
            "O objetivo foi maximizar retorno com robustez e limitar risco, com avaliação OOS via walk-forward semanal e constraints de risco diário.",
            styles["BodyText"],
        )
    )
    story.append(
        Paragraph(
            "O modelo escolhido é o <b>global_bayes</b>: a cada semana ele re-otimiza as regras por segmento usando somente o histórico disponível "
            "(expanding window), aplica um fator global <b>α</b> para respeitar risco do portfólio agregado no treino daquele passo, "
            "e então é avaliado na semana seguinte (OOS).",
            styles["BodyText"],
        )
    )

    # 6) parameters
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("<b>6. Parâmetros e constraints usados</b>", styles["Heading2"]))
    params_tbl = [
        ["Item", "Valor"],
        ["Banca (BANKROLL)", f"USD {BANKROLL:,.0f}"],
        ["Stake grid", "1%..7% (passo 1%)"],
        ["Cutoff grid", "0.05..0.95 (passo 0.02)"],
        ["Cap de ROI (stress)", "cap2 = min(ROI Real, 2.0)"],
        ["Exposição diária", "p80(soma stakes/dia) ≤ 70% da banca"],
        ["Risco diário", "VaR10% do PnL diário ≥ -25% da banca e P(loss≥25%) ≤ 10%"],
        ["Sharpe mínimo (cap2)", "Sharpe semanal ≥ 0.10"],
        ["Score-bin stability", "bins=5; exigir ≥4 bins com lucro médio positivo (com relaxamentos se <5 bins)"],
        ["Walk-forward", "semanal, expanding window; mínimo global 10 semanas; mínimo por segmento 6 semanas"],
        ["Seleção Bayes (por candidato)", "Bayesian bootstrap semanal; exigir P(mean>0) ≥ 80%; objetivo = p05(mean) − 0.001·p95(exposição diária)"],
        ["α global", "busca binária em [0,1] para satisfazer constraints globais no treino do passo"],
    ]
    # wrap long strings
    params_tbl_wrapped = [[params_tbl[0][0], params_tbl[0][1]]]
    for k, v in params_tbl[1:]:
        params_tbl_wrapped.append([P(k), P(v)])

    t = Table(params_tbl_wrapped, colWidths=[6.2 * cm, 10.8 * cm], repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t)

    story.append(PageBreak())

    # 2) chosen portfolio details (current week)
    story.append(Paragraph("<b>2. Portfólio otimizado escolhido (Bayes Global)</b>", styles["Heading2"]))
    story.append(
        Paragraph(
            "Importante: o Bayes Global é uma <b>política dinâmica</b> (as regras mudam ao longo do tempo). "
            "Abaixo estão as regras da semana mais recente disponível no dataset (portfólio 'atual' segundo o walk-forward).",
            styles["BodyText"],
        )
    )
    if cur.empty:
        story.append(Paragraph("Arquivo de regras atuais não encontrado.", styles["BodyText"]))
    else:
        cur2 = cur[["bet_type", "dow_pt", "score_col", "cutoff", "stake_frac", "alpha_global"]].copy()
        cur2 = cur2.sort_values(["bet_type", "dow_pt"])
        # ---------------------------------------------------------
        # Métricas por combinação (estimadas no histórico disponível
        # antes da semana atual, com stake_eff = min(banca*stake*α, cap))
        # ---------------------------------------------------------
        # Para refletir a política dinâmica, usamos o histórico anterior
        # à semana corrente (última semana do WF).
        def week_start(week_str: str) -> pd.Timestamp:
            return pd.to_datetime(str(week_str).split("/")[0])

        last_week = str(cur["test_week"].iloc[0]) if "test_week" in cur.columns and not cur.empty else str(wf_rules["test_week"].iloc[-1])
        last_start = week_start(last_week)

        df_all = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])
        df_all["roi_raw"] = pd.to_numeric(df_all["ROI Real"], errors="coerce").astype(float)
        df_all["roi_cap2"] = np.minimum(df_all["roi_raw"].to_numpy(dtype=float), 2.0)
        df_all["house_cap"] = pd.to_numeric(df_all["house_cap"], errors="coerce").astype(float)
        df_all["week"] = pd.to_datetime(df_all["BIA_ApostaUTC"]).dt.to_period("W-SUN").astype(str)
        df_all["week_start"] = pd.to_datetime(df_all["week"].str.split("/").str[0])
        df_train_cur = df_all[df_all["week_start"] < last_start].copy()
        weeks_all_train = sorted(df_train_cur["week"].unique().tolist())

        def per_combo_metrics(bt: str, dow: str, score_col: str, cutoff: float, stake_frac: float, alpha: float) -> Dict[str, float]:
            x = df_train_cur[(df_train_cur["bet_type"] == bt) & (df_train_cur["dow_pt"] == dow)].copy()
            if x.empty:
                return {"mean_week_profit": float("nan"), "std_week_profit": float("nan"), "roi_on_stake": float("nan"), "mean_week_stake": float("nan"), "mean_week_bets": float("nan"), "pneg_week": float("nan")}
            score = pd.to_numeric(x[score_col], errors="coerce").to_numpy(dtype=float)
            roi2 = x["roi_cap2"].to_numpy(dtype=float)
            cap = x["house_cap"].to_numpy(dtype=float)
            wk = x["week"].to_numpy()
            m = np.isfinite(score) & (score >= cutoff) & np.isfinite(roi2) & np.isfinite(cap) & (cap > 0)
            if not np.any(m):
                # sem apostas selecionadas: semanas zeradas
                w = np.zeros(len(weeks_all_train), dtype=float)
                s = np.zeros(len(weeks_all_train), dtype=float)
                b = np.zeros(len(weeks_all_train), dtype=float)
            else:
                stake0 = BANKROLL * stake_frac * alpha
                stake_eff = np.minimum(stake0, cap[m])
                profit = stake_eff * roi2[m]
                g = pd.DataFrame({"week": wk[m], "stake": stake_eff, "profit": profit}).groupby("week", as_index=False).agg(stake=("stake", "sum"), profit=("profit", "sum"), bets=("profit", "size"))
                g = g.set_index("week").reindex(weeks_all_train, fill_value=0.0)
                w = g["profit"].to_numpy(dtype=float)
                s = g["stake"].to_numpy(dtype=float)
                b = g["bets"].to_numpy(dtype=float)
            mean_w = float(np.mean(w))
            std_w = float(np.std(w, ddof=1)) if w.size > 1 else 0.0
            pneg = float(np.mean(w < 0))
            roi = float(np.sum(w) / np.sum(s)) if float(np.sum(s)) > 0 else float("nan")
            return {
                "mean_week_profit": mean_w,
                "std_week_profit": std_w,
                "pneg_week": pneg,
                "roi_on_stake": roi,
                "mean_week_stake": float(np.mean(s)),
                "mean_week_bets": float(np.mean(b)),
            }

        # tabela com métricas
        rows = [[
            P("<b>Tipo</b>"),
            P("<b>Dia</b>"),
            P("<b>Score</b>"),
            P("<b>Cutoff</b>"),
            P("<b>Stake</b>"),
            P("<b>α</b>"),
            P("<b>Mean PnL<br/>sem (USD)</b>"),
            P("<b>Std PnL<br/>sem (USD)</b>"),
            P("<b>P(sem&lt;0)</b>"),
            P("<b>ROI/$</b>"),
            P("<b>Stake/sem<br/>(USD)</b>"),
            P("<b>Apostas/sem</b>"),
        ]]

        for _, rr in cur2.iterrows():
            bt = str(rr["bet_type"])
            dow = str(rr["dow_pt"])
            sc = str(rr["score_col"])
            cut = float(rr["cutoff"])
            frac = float(rr["stake_frac"])
            a = float(rr["alpha_global"])
            met = per_combo_metrics(bt, dow, sc, cut, frac, a)
            rows.append(
                [
                    P(bt),
                    P(dow),
                    P(sc),
                    P(f"{cut:.2f}"),
                    P(f"{frac*100:.1f}%"),
                    P(f"{a:.3f}"),
                    P(f"{met['mean_week_profit']:.0f}"),
                    P(f"{met['std_week_profit']:.0f}"),
                    P(f"{met['pneg_week']*100:.1f}%"),
                    P(f"{met['roi_on_stake']:.3f}" if np.isfinite(met["roi_on_stake"]) else "nan"),
                    P(f"{met['mean_week_stake']:.0f}"),
                    P(f"{met['mean_week_bets']:.1f}"),
                ]
            )

        tt = Table(
            rows,
            colWidths=[1.0 * cm, 2.5 * cm, 3.6 * cm, 1.2 * cm, 1.2 * cm, 0.9 * cm, 1.6 * cm, 1.6 * cm, 1.2 * cm, 1.0 * cm, 1.5 * cm, 1.2 * cm],
            repeatRows=1,
        )
        tt.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(tt)

    story.append(Spacer(1, 0.4 * cm))
    story.append(
        Paragraph(
            "A execução operacional por aposta usa: <b>stake_eff = min(banca × stake_frac × α, house_cap)</b>. "
            "O parâmetro α é global por semana (controle de risco agregado).",
            styles["BodyText"],
        )
    )

    # 3) metrics
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("<b>3. Métricas estatísticas e de negócio</b>", styles["Heading2"]))
    story.append(Paragraph("<b>3.1 Métricas do Bayes Global (walk-forward OOS, cap2)</b>", styles["Heading3"]))

    m_tbl = [
        ["Métrica", "Valor"],
        ["Semanas avaliadas (WF OOS)", f"{wstats.get('n', 0)}"],
        ["Lucro total (WF)", f"USD {profit_tot:,.1f}"],
        ["Stake total (turnover, WF)", f"USD {stake_tot:,.1f}"],
        ["ROI por $ apostado (WF)", f"{roi_on_stake:.4f}"],
        ["Lucro médio semanal", f"USD {wstats.get('mean', float('nan')):,.1f}"],
        ["Mediana semanal", f"USD {wstats.get('median', float('nan')):,.1f}"],
        ["Std semanal", f"USD {wstats.get('std', float('nan')):,.1f}"],
        ["P(semana<0)", f"{wstats.get('pneg', float('nan'))*100:.1f}%"],
        ["Assimetria (skewness)", f"{wstats.get('skew', float('nan')):.3f}"],
        ["Sharpe semanal (cap2)", f"{wstats.get('sharpe_week', float('nan')):.3f}"],
        ["Sharpe anualizado (cap2)", f"{wstats.get('sharpe_annual', float('nan')):.3f}"],
        ["ROI banca (médio por semana)", f"{roi_bank_week*100:.2f}%"],
        ["Lucro esperado mensal (≈4,33 sem)", f"USD {exp_month:,.0f}"],
        ["Lucro esperado anual (≈52 sem)", f"USD {exp_year:,.0f}"],
        ["Risco diário: p80(stake/dia)", f"USD {p80_exp:,.0f} (limite USD {MAX_DAILY_EXPOSURE_FRAC_Q*BANKROLL:,.0f})"],
        ["Risco diário: VaR10%(PnL dia)", f"USD {var10:,.1f} (limite ≥ USD {-MAX_DAILY_DRAWDOWN_FRAC*BANKROLL:,.0f})"],
        ["Risco diário: P(PnL dia ≤ -25% banca)", f"{p_dd*100:.1f}% (limite ≤ {MAX_P_DAILY_DD*100:.0f}%)"],
        ["α global: média / p10 / p50 / p90", f"{alpha_mean:.3f} / {alpha_p10:.3f} / {alpha_p50:.3f} / {alpha_p90:.3f}"],
        ["α global: P(α<1)", f"{p_alpha_lt1*100:.1f}%"],
    ]
    t2 = Table(m_tbl, colWidths=[7.5 * cm, 9.5 * cm])
    t2.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(t2)

    # weekly quantiles section
    story.append(Spacer(1, 0.3 * cm))
    story.append(
        Paragraph(
            f"Quantis do PnL semanal (USD): p05={wstats.get('q05', float('nan')):,.0f}, "
            f"p10={wstats.get('q10', float('nan')):,.0f}, p25={wstats.get('q25', float('nan')):,.0f}, "
            f"p50={wstats.get('q50', float('nan')):,.0f}, p75={wstats.get('q75', float('nan')):,.0f}, "
            f"p90={wstats.get('q90', float('nan')):,.0f}, p95={wstats.get('q95', float('nan')):,.0f}.",
            styles["BodyText"],
        )
    )

    # comparison table vs fixed in-sample
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("<b>3.2 Comparação com o portfólio fixo in-sample</b>", styles["Heading3"]))
    if comp.empty:
        story.append(Paragraph("Arquivo de comparação não encontrado.", styles["BodyText"]))
    else:
        show = comp.copy()
        # choose a few rows
        cols = ["name", "weeks", "bets", "stake", "profit", "roi_on_stake", "mean_week", "std_week", "pneg_week", "p80_day_stake", "var10_day_pnl"]
        show = show[cols].copy()
        rows = [[
            P("<b>Cenário</b>"),
            P("<b>Sem</b>"),
            P("<b>Apostas</b>"),
            P("<b>Stake</b>"),
            P("<b>Lucro</b>"),
            P("<b>ROI/$</b>"),
            P("<b>Mean<br/>sem</b>"),
            P("<b>Std<br/>sem</b>"),
            P("<b>P(sem&lt;0)</b>"),
            P("<b>p80<br/>stake/dia</b>"),
            P("<b>VaR10%<br/>dia</b>"),
        ]]
        for _, r in show.iterrows():
            rows.append(
                [
                    P(str(r["name"])),
                    str(int(r["weeks"])),
                    str(int(r["bets"])),
                    f"{float(r['stake']):,.0f}",
                    f"{float(r['profit']):,.0f}",
                    f"{float(r['roi_on_stake']):.4f}" if np.isfinite(float(r["roi_on_stake"])) else "nan",
                    f"{float(r['mean_week']):,.0f}",
                    f"{float(r['std_week']):,.0f}",
                    f"{float(r['pneg_week'])*100:.1f}%" if np.isfinite(float(r["pneg_week"])) else "nan",
                    f"{float(r['p80_day_stake']):,.0f}" if np.isfinite(float(r["p80_day_stake"])) else "nan",
                    f"{float(r['var10_day_pnl']):,.0f}" if np.isfinite(float(r["var10_day_pnl"])) else "nan",
                ]
            )
        tt2 = Table(
            rows,
            colWidths=[3.9 * cm, 1.0 * cm, 1.4 * cm, 1.35 * cm, 1.35 * cm, 1.1 * cm, 1.1 * cm, 1.1 * cm, 1.2 * cm, 1.45 * cm, 1.3 * cm],
            rowHeights=[1.1 * cm] + [None] * (len(rows) - 1),
            repeatRows=1,
        )
        tt2.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTSIZE", (0, 0), (-1, -1), 6),
                    ("LEADING", (0, 0), (-1, 0), 7),
                    ("ALIGN", (1, 0), (-1, 0), "CENTER"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(tt2)

    # 3.3 maximum operation
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("<b>3.3 Operação no máximo (banca não limita stake)</b>", styles["Heading3"]))
    story.append(
        Paragraph(
            "Aqui calculamos um cenário hipotético em que a banca é grande o suficiente para que o stake por aposta não seja limitado pela banca, "
            "ou seja, <b>stake_eff_max = house_cap</b> (o limitante passa a ser apenas o cap da casa). "
            "Mantivemos as mesmas regras (cutoff/stake_frac) escolhidas semanalmente no walk-forward para não misturar mudanças de estratégia com sizing.",
            styles["BodyText"],
        )
    )
    story.append(
        Paragraph(
            f"Banca necessária para que a banca não limite a maioria das apostas: "
            f"<b>p95(house_cap / stake_frac) ≈ USD {bank_p95:,.0f}</b>. "
            f"Para não limitar nenhuma aposta na amostra: <b>max ≈ USD {bank_max:,.0f}</b>.",
            styles["BodyText"],
        )
    )
    if max_week_stats.get("n", 0) > 0:
        # ROI on bank using bank_p95 (as reference)
        mean_week_max = float(max_week_stats["mean"])
        roi_bank_week_max = (mean_week_max / bank_p95) if np.isfinite(bank_p95) and bank_p95 > 0 else float("nan")
        m_tbl2 = [
            ["Métrica", "Valor"],
            ["Semanas (amostra WF)", f"{int(max_week_stats['n'])}"],
            ["Stake total (máx)", f"USD {float(weekly_max['stake_max'].sum()):,.0f}"],
            ["Lucro total (máx)", f"USD {float(weekly_max['profit_max'].sum()):,.0f}"],
            ["ROI por $ apostado (máx)", f"{float(weekly_max['profit_max'].sum()/weekly_max['stake_max'].sum()):.4f}"],
            ["Lucro médio semanal (máx)", f"USD {mean_week_max:,.1f}"],
            ["Mediana semanal (máx)", f"USD {float(max_week_stats['median']):,.1f}"],
            ["Std semanal (máx)", f"USD {float(max_week_stats['std']):,.1f}"],
            ["Sharpe anualizado (máx)", f"{float(max_week_stats['sharpe_annual']):.3f}"],
            ["ROI banca/sem (usando banca p95)", f"{roi_bank_week_max*100:.2f}%"],
        ]
        tmax = Table(m_tbl2, colWidths=[7.5 * cm, 9.5 * cm])
        tmax.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(tmax)
    else:
        story.append(Paragraph("Não foi possível calcular o cenário de operação máxima (faltam dados/alinhamento de amostra).", styles["BodyText"]))

    story.append(PageBreak())

    # 5) stability
    story.append(Paragraph("<b>5. Estabilidade (regras ao longo do tempo)</b>", styles["Heading2"]))
    story.append(
        Paragraph(
            "Para medir estabilidade/instabilidade, analisamos como o conjunto de segmentos ativos e os parâmetros (cutoff/stake) mudam entre semanas.",
            styles["BodyText"],
        )
    )
    story.append(Paragraph("<b>5.1 Similaridade Jaccard (segmentos ativos, semana a semana)</b>", styles["Heading3"]))
    story.append(
        Paragraph(
            f"Jaccard médio={jac.get('jaccard_mean', float('nan')):.3f} "
            f"(p10={jac.get('jaccard_p10', float('nan')):.3f}, p50={jac.get('jaccard_p50', float('nan')):.3f}, p90={jac.get('jaccard_p90', float('nan')):.3f}).",
            styles["BodyText"],
        )
    )
    story.append(
        Paragraph(
            "Interpretação: Jaccard próximo de 1 indica que o conjunto de segmentos ativos muda pouco; valores mais baixos indicam maior rotatividade. "
            "Não existe um 'padrão universal de mercado' para esse número (depende do quão adaptativo é o processo e do regime), "
            "mas valores ~0,8 sugerem estabilidade moderada/alta, com algumas semanas de mudança forte quando o p10 é bem menor.",
            styles["BodyText"],
        )
    )

    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("<b>5.2 Volatilidade de parâmetros (cutoff/stake) por segmento</b>", styles["Heading3"]))
    if rc.empty:
        story.append(Paragraph("Sem dados suficientes para estatísticas de mudança.", styles["BodyText"]))
    else:
        # incluir todas as combinações (tabela completa, 14 segmentos)
        top_unstable = rc.sort_values(["n_active_weeks", "cutoff_std"], ascending=[False, False]).copy()
        rows = [["Segmento", "Semanas ativas", "Cutoff std", "Stake std", "Taxa mudança cutoff", "Taxa mudança stake"]]
        for _, r in top_unstable.iterrows():
            rows.append(
                [
                    str(r["rule_key"]),
                    str(int(r["n_active_weeks"])),
                    f"{float(r['cutoff_std']):.3f}",
                    f"{float(r['stake_std'])*100:.2f}%",
                    f"{float(r['cutoff_change_rate']):.3f}" if np.isfinite(float(r["cutoff_change_rate"])) else "nan",
                    f"{float(r['stake_change_rate']):.3f}" if np.isfinite(float(r["stake_change_rate"])) else "nan",
                ]
            )
        tt3 = Table(rows, colWidths=[3.2 * cm, 2.0 * cm, 2.0 * cm, 1.8 * cm, 3.6 * cm, 3.6 * cm], repeatRows=1)
        tt3.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(tt3)

    # extra page: full audit of active segments by week
    story.append(PageBreak())
    story.append(Paragraph("<b>Anexo A — Auditoria: segmentos ativos por semana (walk-forward)</b>", styles["Heading2"]))
    story.append(
        Paragraph(
            "Tabela para auditoria operacional: em cada semana do walk-forward, quais segmentos ficaram ativos (status=ok e stake>0), "
            "o α global da semana e a lista de segmentos.",
            styles["BodyText"],
        )
    )
    if weekly_audit.empty:
        story.append(Paragraph("Sem dados para auditoria semanal.", styles["BodyText"]))
    else:
        rows = [["Semana", "α", "N ativos", "Segmentos ativos"]]
        for _, rr in weekly_audit.iterrows():
            rows.append(
                [
                    P(rr["week"]),
                    P(f"{float(rr['alpha']):.3f}" if np.isfinite(float(rr["alpha"])) else "nan"),
                    P(str(int(rr["n_active"]))),
                    P(rr["segments"]),
                ]
            )
        ta = Table(rows, colWidths=[3.6 * cm, 1.0 * cm, 1.4 * cm, 11.0 * cm], repeatRows=1)
        ta.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(ta)

    # 4) improvements / suggestions
    story.append(PageBreak())
    story.append(Paragraph("<b>4 & 8. Melhorias sugeridas e evoluções necessárias</b>", styles["Heading2"]))
    story.append(
        Paragraph(
            "Sugestões práticas para aumentar robustez e previsibilidade operacional:",
            styles["BodyText"],
        )
    )
    bullets = [
        "Aumentar o horizonte OOS (mais semanas) e monitorar por regime; hoje o OOS real (2026) é curto.",
        "Adicionar penalidade explícita por instabilidade (custo por mudança de cutoff/stake) e/ou restringir o grid por segmento.",
        "Adicionar uma camada de seleção de segmentos com partial pooling (já implementado) como 'prior' para ligar/desligar segmentos.",
        "Testes de qualidade de ROI/payout: manter cap1/cap2 como stress tests e investigar outliers.",
        "Aprimorar o controle de risco global no OOS (hoje ele é imposto no treino do passo; deve ser monitorado na execução real).",
    ]
    for b in bullets:
        story.append(Paragraph(f"• {b}", styles["BodyText"]))

    # 7) investor-style decision
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("<b>7. Análise de investimento (se fosse meu dinheiro)</b>", styles["Heading2"]))
    story.append(
        Paragraph(
            "Com os dados atuais, eu trataria este portfólio como um ativo com evidência inicial de edge, mas ainda com incerteza material: "
            "o IC95% do PnL semanal no walk-forward ainda cruza zero e o período OOS real (2026) é curto. "
            "Eu investiria apenas com sizing conservador (fração pequena da banca alvo), operação monitorada (especialmente risco diário) e reotimização semanal, "
            "até acumular mais semanas OOS para confirmar estabilidade.",
            styles["BodyText"],
        )
    )

    # 9) data note
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("<b>9. Data e reprodutibilidade</b>", styles["Heading2"]))
    story.append(
        Paragraph(
            f"Relatório gerado em {today}. Fontes principais: `{WF_WEEKLY.name}`, `{WF_DAILY.name}`, `{WF_RULES.name}`, `{CUR_RULES.name}`, `{COMPARISON.name}`.",
            styles["BodyText"],
        )
    )

    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=1.2 * cm, leftMargin=1.2 * cm, topMargin=1.2 * cm, bottomMargin=1.2 * cm)
    doc.build(story)
    print(str(pdf_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

