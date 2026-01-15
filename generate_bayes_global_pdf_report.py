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

    # build document
    styles = getSampleStyleSheet()
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
        ["Seleção Bayes (por candidato)", "Bayesian bootstrap semanal; exigir P(mean>0) ≥ 80%; objetivo = p05(mean) − 0.001·p95_exposição"],
        ["α global", "busca binária em [0,1] para satisfazer constraints globais no treino do passo"],
    ]
    t = Table(params_tbl, colWidths=[6.5 * cm, 10.5 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
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
        rows = [["Tipo", "Dia", "Score", "Cutoff", "Stake", "α"]]
        for _, r in cur2.iterrows():
            rows.append(
                [
                    str(r["bet_type"]),
                    str(r["dow_pt"]),
                    str(r["score_col"]),
                    f"{float(r['cutoff']):.2f}",
                    f"{float(r['stake_frac'])*100:.1f}%",
                    f"{float(r['alpha_global']):.3f}",
                ]
            )
        tt = Table(rows, colWidths=[1.3 * cm, 3.1 * cm, 4.8 * cm, 1.6 * cm, 1.6 * cm, 1.2 * cm])
        tt.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
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
        rows = [["Cenário", "Sem", "Apostas", "Stake", "Lucro", "ROI/$", "Mean/sem", "Std/sem", "P(sem<0)", "p80 stake/dia", "VaR10% dia"]]
        for _, r in show.iterrows():
            rows.append(
                [
                    str(r["name"]),
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
        tt2 = Table(rows, colWidths=[4.2 * cm, 1.0 * cm, 1.2 * cm, 1.4 * cm, 1.4 * cm, 1.2 * cm, 1.4 * cm, 1.2 * cm, 1.3 * cm, 1.5 * cm, 1.2 * cm])
        tt2.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(tt2)

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
        top_unstable = rc.sort_values("cutoff_std", ascending=False).head(8)
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
        tt3 = Table(rows, colWidths=[3.2 * cm, 2.2 * cm, 2.1 * cm, 2.0 * cm, 3.3 * cm, 3.3 * cm])
        tt3.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(tt3)

    # 4) improvements / suggestions
    story.append(Spacer(1, 0.4 * cm))
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

