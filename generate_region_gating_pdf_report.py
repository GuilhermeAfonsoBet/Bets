#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera um PDF detalhado do estudo: Região como gating (modo conservador block-bad).

Fonte de verdade:
- Baseline OOS (modelo global_bayes_roll12_robust_p10_p70): oos_walkforward_*_weekly.csv
- Gating OOS (block-bad): oos_walkforward_region_gating_exantepred_blockbad_weekly.csv + _summary.csv
- Comparativo semanal: compare_region_gating_blockbad_vs_baseline_weekly.csv
- Bloqueios: oos_walkforward_region_gating_exantepred_blockbad_blocked_regions.csv e/ou resumo

Saída:
- analysis_proba_raw/pro_portfolio_all/Relatorio_RegiaoGating_BlockBad_<YYYY-MM-DD>.pdf
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")
MODE = "global_bayes_roll12_robust_p10_p70"

BASELINE_WEEKLY = OUT_DIR / f"oos_walkforward_{MODE}_weekly.csv"
GATING_WEEKLY = OUT_DIR / "oos_walkforward_region_gating_exantepred_blockbad_weekly.csv"
GATING_SUMMARY = OUT_DIR / "oos_walkforward_region_gating_exantepred_blockbad_summary.csv"

COMPARE_WEEKLY = OUT_DIR / "compare_region_gating_blockbad_vs_baseline_weekly.csv"
BLOCKED = OUT_DIR / "oos_walkforward_region_gating_exantepred_blockbad_blocked_regions.csv"
BLOCKED_SUMMARY = OUT_DIR / "region_gating_blockbad_blocked_regions_summary.csv"

REGION_PRED = OUT_DIR / "region_exante_pred.csv"
RULES = OUT_DIR / f"oos_walkforward_{MODE}_selected_rules.csv"
SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")


def _fmt_money(x: float) -> str:
    return f"{x:,.2f}"


def _fmt_pct(x: float) -> str:
    return f"{x*100.0:+.2f}%"


def _bootstrap_mean_ci(x: np.ndarray, n_boot: int = 20000, seed: int = 7) -> Tuple[float, float, float, float]:
    """
    Retorna: (mean_obs, ci95_lo, ci95_hi, P_boot(mean>0))
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    n = int(a.size)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    m = a[idx].mean(axis=1)
    lo = float(np.quantile(m, 0.025))
    hi = float(np.quantile(m, 0.975))
    return float(a.mean()), lo, hi, float(np.mean(m > 0))


def _signflip_perm_p_mean_gt0(x: np.ndarray, n_perm: int = 50000, seed: int = 7) -> float:
    """
    Sign-flip permutation test (H0: média=0 e simetria) — p-value (um lado) para mean>0.
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    mu_obs = float(np.mean(a))
    signs = rng.choice(np.array([-1.0, 1.0], dtype=float), size=(int(n_perm), int(a.size)))
    mu_perm = np.mean(signs * a[None, :], axis=1)
    return float((1.0 + np.sum(mu_perm >= mu_obs)) / (1.0 + float(n_perm)))


def _binom_sf_geq(k: int, n: int, p: float) -> float:
    """P(X>=k) para Binomial(n,p) — exato via soma (n pequeno: 14)."""
    if n <= 0:
        return float("nan")
    from math import comb

    out = 0.0
    for i in range(int(k), int(n) + 1):
        out += comb(int(n), int(i)) * (p**i) * ((1 - p) ** (n - i))
    return float(out)


def _simulate_blockbad(
    df_scored: pd.DataFrame,
    rules: pd.DataFrame,
    bad_mean_roi_th: float,
    min_n_per_region: int,
) -> pd.DataFrame:
    """
    Reproduz (aprox) o gating block-bad para uma grade de parâmetros,
    usando a mesma regra OOS (sem reotimizar cutoffs/stakes).
    """
    df = df_scored.copy()
    df["week"] = pd.to_datetime(df["BIA_ApostaUTC"]).dt.to_period("W-SUN").astype(str)
    df["date"] = pd.to_datetime(df["BIA_ApostaUTC"]).dt.date.astype(str)
    df["roi_cap2"] = pd.to_numeric(df["roi_calc"], errors="coerce").clip(upper=2.0).to_numpy(float)
    df["house_cap"] = pd.to_numeric(df["house_cap"], errors="coerce").to_numpy(float)
    df["house_cap"] = np.where(np.isfinite(df["house_cap"]) & (df["house_cap"] > 0), df["house_cap"], np.inf)

    # region
    reg = pd.read_csv(REGION_PRED, usecols=["ID Aposta", "region_pred"])
    reg = reg.rename(columns={"region_pred": "region_evt"})
    df = df.merge(reg, how="left", on="ID Aposta")
    df["region_evt"] = df["region_evt"].astype("string").fillna("desconhecida").astype(str)

    weeks = sorted(df["week"].unique().tolist())
    out_rows = []

    for w_test, rw in rules.groupby("test_week", as_index=False):
        w_test = str(w_test)
        if w_test not in weeks:
            continue
        i = weeks.index(w_test)
        train_weeks = weeks[max(0, i - 12) : i]  # 12 como no script principal
        df_train = df[df["week"].isin(train_weeks)].copy()
        df_test = df[df["week"] == w_test].copy()
        if df_test.empty:
            continue

        # bloqueios por rule_key
        block_by_rule: Dict[str, set] = {}
        for _, rr in rw.iterrows():
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
            regv = x["region_evt"].astype(str).to_numpy()
            m = np.isfinite(score) & (score >= cutoff) & np.isfinite(roi2)
            if not np.any(m):
                continue
            xt = pd.DataFrame({"region": regv[m], "roi": roi2[m]})
            by = xt.groupby("region", as_index=False).agg(n=("roi", "size"), mean_roi=("roi", "mean"))
            block = set(by.loc[(by["n"] >= int(min_n_per_region)) & (by["mean_roi"] <= float(bad_mean_roi_th)), "region"].astype(str).tolist())
            if block:
                block_by_rule[rk] = block

        # aplicar regras + bloqueios
        bets = []
        alpha = float(rw["alpha_effective"].iloc[0]) if "alpha_effective" in rw.columns and np.isfinite(float(rw["alpha_effective"].iloc[0])) else float(rw["alpha_global"].iloc[0])
        for _, rr in rw.iterrows():
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

            x = df_test[(df_test["bet_type"] == bt) & (df_test["dow_pt"] == dow)].copy()
            if x.empty or sc not in x.columns:
                continue
            score = pd.to_numeric(x[sc], errors="coerce").to_numpy(float)
            roi2 = x["roi_cap2"].to_numpy(float)
            cap = x["house_cap"].to_numpy(float)
            regv = x["region_evt"].astype(str).to_numpy()
            m = np.isfinite(score) & (score >= cutoff) & np.isfinite(roi2) & np.isfinite(cap) & (cap > 0)
            if rk in block_by_rule:
                m = m & (~np.isin(regv, list(block_by_rule[rk])))
            if not np.any(m):
                continue
            stake0 = 2300.0 * stake_frac * float(alpha)
            stake_eff = np.minimum(stake0, cap[m])
            profit = stake_eff * roi2[m]
            bets.append(pd.DataFrame({"stake_eff": stake_eff, "profit_cap2": profit}))

        if bets:
            bb = pd.concat(bets, axis=0, ignore_index=True)
            stake_sum = float(bb["stake_eff"].sum())
            pnl_sum = float(bb["profit_cap2"].sum())
            n_bets = int(len(bb))
        else:
            stake_sum = 0.0
            pnl_sum = 0.0
            n_bets = 0

        out_rows.append({"week": w_test, "stake_usd": stake_sum, "profit_cap2_usd": pnl_sum, "n_bets": n_bets})

    return pd.DataFrame(out_rows).sort_values("week")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not BASELINE_WEEKLY.exists() or not GATING_WEEKLY.exists() or not COMPARE_WEEKLY.exists():
        raise FileNotFoundError("Artefatos obrigatórios ausentes (baseline/gating/compare).")

    base = pd.read_csv(BASELINE_WEEKLY)
    gate = pd.read_csv(GATING_WEEKLY)
    cmp = pd.read_csv(COMPARE_WEEKLY)

    # estatística em semanas com stake (baseline ou gating)
    cmp["stake_usd_base"] = pd.to_numeric(cmp["stake_usd_base"], errors="coerce")
    cmp["stake_usd_gate"] = pd.to_numeric(cmp["stake_usd_gate"], errors="coerce")
    cmp["delta_pnl"] = pd.to_numeric(cmp["delta_pnl"], errors="coerce")
    active = (cmp["stake_usd_base"] > 0) | (cmp["stake_usd_gate"] > 0)
    d = cmp.loc[active, "delta_pnl"].to_numpy(float)

    mean_obs, ci_lo, ci_hi, pboot = _bootstrap_mean_ci(d, n_boot=30000, seed=7)
    p_perm = _signflip_perm_p_mean_gt0(d, n_perm=50000, seed=7)
    n = int(np.isfinite(d).sum())
    k_pos = int(np.sum(d[np.isfinite(d)] > 0))
    p_sign = _binom_sf_geq(k_pos, n, 0.5)

    # robustez: pequena varredura de parâmetros do gating
    df_sc = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])
    rules = pd.read_csv(RULES)
    grid = []
    for bad_th in [-0.05, -0.02, 0.0]:
        for min_n in [20, 30]:
            wk = _simulate_blockbad(df_sc, rules, bad_mean_roi_th=float(bad_th), min_n_per_region=int(min_n))
            if wk.empty:
                continue
            stake = float(pd.to_numeric(wk["stake_usd"], errors="coerce").sum())
            pnl = float(pd.to_numeric(wk["profit_cap2_usd"], errors="coerce").sum())
            grid.append({"bad_mean_roi_th": bad_th, "min_n_per_region": min_n, "stake_total": stake, "pnl_total": pnl, "roi": (pnl / stake) if stake > 0 else float("nan")})
    grid_df = pd.DataFrame(grid).sort_values(["bad_mean_roi_th", "min_n_per_region"])

    # blocked summary
    blocked_df = pd.read_csv(BLOCKED) if BLOCKED.exists() else pd.DataFrame()

    # PDF
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.lineplots import LinePlot

    styles = getSampleStyleSheet()

    out_pdf = OUT_DIR / f"Relatorio_RegiaoGating_BlockBad_{date.today().isoformat()}.pdf"
    doc = SimpleDocTemplate(str(out_pdf), pagesize=A4, leftMargin=1.4 * cm, rightMargin=1.4 * cm, topMargin=1.2 * cm, bottomMargin=1.2 * cm)

    def P(txt: str):
        return Paragraph(txt, styles["BodyText"])

    story: List = []
    story.append(Paragraph("<b>Relatório — Região como gating (block-bad)</b>", styles["Title"]))
    story.append(Paragraph(f"<b>Data</b>: {date.today().isoformat()}", styles["BodyText"]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("<b>1) Objetivo</b>", styles["Heading2"]))
    story.append(P("Avaliar se a inclusão de <b>Região do evento</b> como <b>gating</b> (filtro) melhora a estratégia OOS, e se a melhora pode ser explicada apenas por acaso."))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("<b>2) Construção de Região (ex-ante, determinística)</b>", styles["Heading2"]))
    story.append(P("A região foi construída de forma <b>determinística</b> a partir de <b>`RebelBetting.EventName`</b> (Excel original), extraindo o país (prefixo) e mapeando para continente/região."))
    story.append(P("Isso evita dependência de campos ex-post (ex.: BetinAsia) e dá cobertura alta para o gating."))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("<b>3) Mecânica do gating (block-bad)</b>", styles["Heading2"]))
    story.append(P("O gating é aplicado <b>após</b> o critério principal (score ≥ cutoff) do portfólio. Ele <b>não altera o score</b> nem os cutoffs."))
    story.append(P("Para cada semana de teste e para cada segmento (DoW×FT/FH), calculamos no TREINO (últimas 12 semanas) a performance por região e <b>bloqueamos</b> apenas regiões com evidência de ROI ruim:"))
    story.append(P(f"- condição de bloqueio: <b>n ≥ {20}</b> e <b>mean(ROI_cap2) ≤ {-0.02}</b>"))
    story.append(P("Na semana de teste, apostas que passariam pelo cutoff são filtradas removendo apenas as regiões bloqueadas."))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("<b>4) Resultados OOS (baseline vs gating)</b>", styles["Heading2"]))
    summ = pd.read_csv(GATING_SUMMARY) if GATING_SUMMARY.exists() else pd.DataFrame()
    if not summ.empty:
        rows = [[P("<b>Variante</b>"), P("<b>PnL cap2</b>"), P("<b>Stake</b>"), P("<b>ROI/$</b>"), P("<b>Semanas ativas</b>")]]
        for _, r in summ.iterrows():
            rows.append([P(str(r["name"])), P(_fmt_money(float(r["profit_cap2_total"]))), P(_fmt_money(float(r["stake_total"]))), P(f"{float(r['roi_total_cap2']):.5f}"), P(str(int(r["weeks_with_stake"])))])
        t = Table(rows, colWidths=[3.2 * cm, 3.0 * cm, 3.0 * cm, 2.2 * cm, 2.5 * cm], repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
        story.append(t)
        story.append(Spacer(1, 0.2 * cm))

    # cumulative delta plot
    cc = cmp.copy()
    cc["week_i"] = np.arange(len(cc))
    cc["cum_delta"] = pd.to_numeric(cc["delta_pnl"], errors="coerce").fillna(0.0).cumsum()
    pts = [(float(i), float(v)) for i, v in zip(cc["week_i"], cc["cum_delta"])]
    dplot = Drawing(16 * cm, 5 * cm)
    lp = LinePlot()
    lp.x = 0
    lp.y = 0
    lp.height = 5 * cm
    lp.width = 16 * cm
    lp.data = [pts]
    lp.lines[0].strokeColor = colors.darkblue
    lp.joinedLines = 1
    lp.xValueAxis.valueMin = 0
    lp.xValueAxis.valueMax = max(1, len(cc) - 1)
    lp.yValueAxis.valueMin = float(min(cc["cum_delta"].min(), 0.0))
    lp.yValueAxis.valueMax = float(max(cc["cum_delta"].max(), 0.0))
    dplot.add(lp)
    story.append(P("<b>Cumulativo do ΔPnL (gating − baseline) ao longo das semanas</b>"))
    story.append(dplot)

    story.append(Spacer(1, 0.2 * cm))
    story.append(P("Nota: semanas sem stake aparecem com ΔPnL=0 e não afetam o cumulativo."))

    story.append(PageBreak())

    story.append(Paragraph("<b>5) Testes estatísticos — a melhora pode ser acaso?</b>", styles["Heading2"]))
    story.append(P("Analisamos o vetor de diferenças semanais de PnL (ΔPnL = gating − baseline) nas semanas com stake em pelo menos uma das variantes."))
    story.append(P(f"n semanas (ativas): <b>{n}</b>; semanas com ΔPnL>0: <b>{k_pos}</b>."))
    rows = [
        [P("<b>Teste</b>"), P("<b>O que testa</b>"), P("<b>Resultado</b>")],
        [P("Bootstrap da média (semanas)"), P("IC95% da média de ΔPnL e P(ΔPnL_mean>0)"), P(f"mean={_fmt_money(mean_obs)}; IC95%=[{_fmt_money(ci_lo)}, {_fmt_money(ci_hi)}]; P_boot(mean>0)={pboot:.3f}")],
        [P("Permutação sign-flip (um lado)"), P("H0: média=0 com simetria (robusto)"), P(f"p-value={p_perm:.4f}")],
        [P("Sign test (um lado)"), P("H0: P(ΔPnL>0)=0.5 (sem assumir normalidade)"), P(f"p-value={p_sign:.4f} (X≥{k_pos} em n={n})")],
    ]
    t = Table(rows, colWidths=[4.0 * cm, 7.0 * cm, 5.0 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(t)
    story.append(Spacer(1, 0.2 * cm))
    story.append(P("<b>Interpretação</b>: p-values baixos sugerem que a melhora média semanal não é facilmente explicada por acaso sob H0. Ainda assim, semanas não são perfeitamente i.i.d.; trate como evidência, não prova absoluta."))

    story.append(Spacer(1, 0.25 * cm))
    story.append(Paragraph("<b>6) Robustez (sensibilidade a parâmetros do gating)</b>", styles["Heading2"]))
    story.append(P("Varremos poucos valores para verificar se o ganho depende de um único número."))
    if not grid_df.empty:
        rows = [[P("<b>bad_mean_roi_th</b>"), P("<b>min_n</b>"), P("<b>Stake</b>"), P("<b>PnL</b>"), P("<b>ROI/$</b>")]]
        for _, r in grid_df.iterrows():
            rows.append([P(f"{float(r['bad_mean_roi_th']):+.2f}"), P(str(int(r["min_n_per_region"]))), P(_fmt_money(float(r["stake_total"]))), P(_fmt_money(float(r["pnl_total"]))), P(f"{float(r['roi']):.5f}")])
        t = Table(rows, colWidths=[3.2 * cm, 2.0 * cm, 3.0 * cm, 3.0 * cm, 2.6 * cm], repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
        story.append(t)
    else:
        story.append(P("Sem dados para varredura."))

    story.append(PageBreak())

    story.append(Paragraph("<b>7) Diagnóstico de bloqueios</b>", styles["Heading2"]))
    if BLOCKED_SUMMARY.exists():
        bs = pd.read_csv(BLOCKED_SUMMARY)
        # by_region top
        br = bs[bs["section"] == "by_region"].copy() if "section" in bs.columns else pd.DataFrame()
        if not br.empty:
            br = br.dropna(subset=["region", "block_count"]).copy()
            br["block_count"] = pd.to_numeric(br["block_count"], errors="coerce").astype(float)
            br = br.sort_values("block_count", ascending=False).head(12)
            rows = [[P("<b>Região</b>"), P("<b># bloqueios</b>")]]
            for _, r in br.iterrows():
                rows.append([P(str(r["region"])), P(str(int(r["block_count"])))])
            t = Table(rows, colWidths=[6.0 * cm, 3.0 * cm], repeatRows=1)
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
            story.append(P("<b>Regiões mais bloqueadas (frequência)</b>"))
            story.append(t)
            story.append(Spacer(1, 0.2 * cm))

        # by_rule_key top
        bk = bs[bs["section"] == "by_rule_key"].copy() if "section" in bs.columns else pd.DataFrame()
        if not bk.empty:
            bk = bk.dropna(subset=["rule_key", "weeks_with_blocks"]).copy()
            bk["weeks_with_blocks"] = pd.to_numeric(bk["weeks_with_blocks"], errors="coerce").astype(float)
            bk = bk.sort_values("weeks_with_blocks", ascending=False).head(12)
            rows = [[P("<b>Segmento</b>"), P("<b>Semanas com bloqueio</b>"), P("<b>Eventos de bloqueio</b>")]]
            for _, r in bk.iterrows():
                rows.append([P(str(r["rule_key"])), P(str(int(r["weeks_with_blocks"]))), P(str(int(float(r.get("block_events", 0.0)))))])
            t = Table(rows, colWidths=[6.0 * cm, 3.0 * cm, 3.0 * cm], repeatRows=1)
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
            story.append(P("<b>Segmentos com mais bloqueios</b>"))
            story.append(t)

    else:
        story.append(P("Arquivo de resumo de bloqueios não encontrado."))

    story.append(Spacer(1, 0.25 * cm))
    story.append(Paragraph("<b>8) Recomendação operacional</b>", styles["Heading2"]))
    story.append(P("Para produção, recomenda-se:"))
    story.append(P("- Calcular região <b>deterministicamente</b> a partir de `EventName` (mesma lógica do estudo)."))
    story.append(P("- Usar gating <b>conservador</b> (block-bad), evitando allow-list agressiva que pode zerar semanas inteiras."))
    story.append(P("- Monitorar métricas semanalmente: ΔPnL, ΔStake, regiões bloqueadas por segmento, e refazer os testes de estabilidade conforme o histórico aumenta."))

    doc.build(story)
    print(str(out_pdf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

