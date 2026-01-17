#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera um PDF curto de análise de uma semana específica.

Default: semana 2026-01-12 .. 2026-01-18 (W-SUN), conforme pedido.

Entrada:
- analysis_proba_raw/pro_portfolio_all/oos_walkforward_global_bayes_weekly.csv
- analysis_proba_raw/pro_portfolio_all/oos_walkforward_global_bayes_daily.csv
- analysis_proba_raw/pro_portfolio_all/oos_walkforward_global_bayes_selected_rules.csv

Saída:
- analysis_proba_raw/pro_portfolio_all/Analise_Semana_<YYYY-MM-DD>_<YYYY-MM-DD>.pdf
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")
WF_WEEKLY = OUT_DIR / "oos_walkforward_global_bayes_weekly.csv"
WF_DAILY = OUT_DIR / "oos_walkforward_global_bayes_daily.csv"
WF_RULES = OUT_DIR / "oos_walkforward_global_bayes_selected_rules.csv"
FC_WEEK = OUT_DIR / "forecast_calibration_global_bayes.csv"
FC_ONLINE = OUT_DIR / "forecast_calibration_global_bayes_online_bias.csv"
SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")
BANKROLL = 2300.0


@dataclass(frozen=True)
class WeekSpec:
    start: str  # YYYY-MM-DD
    end: str    # YYYY-MM-DD

    @property
    def key(self) -> str:
        return f"{self.start}/{self.end}"


def fmt_money(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x:,.2f}"


def fmt_pct(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{100.0 * x:.2f}%"


def tbl(data: List[List[object]], col_widths=None) -> Table:
    t = Table(data, colWidths=col_widths)
    t.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(0.98, 0.98, 0.98)]),
            ]
        )
    )
    return t


def _apply_rules(
    df: pd.DataFrame, rules_week: pd.DataFrame, alpha: float, use_cap2: bool
) -> pd.DataFrame:
    """
    Aplica as regras da semana em um dataframe (treino ou teste).
    Retorna bets selecionadas com stake_eff, profit e score.
    """
    rows = []
    for _, r in rules_week.iterrows():
        if str(r.get("status")) != "ok":
            continue
        stake_frac = float(r.get("stake_frac", 0.0))
        if stake_frac <= 0:
            continue
        bt = str(r["bet_type"])
        dow = str(r["dow_pt"])
        score_col = str(r["score_col"])
        cutoff = float(r["cutoff"])

        x = df[(df["bet_type"] == bt) & (df["dow_pt"] == dow)].copy()
        if x.empty or score_col not in x.columns:
            continue
        score = pd.to_numeric(x[score_col], errors="coerce").to_numpy(dtype=float)
        roi_raw = pd.to_numeric(x["ROI Real"], errors="coerce").to_numpy(dtype=float)
        roi_use = np.minimum(roi_raw, 2.0) if use_cap2 else roi_raw
        cap = pd.to_numeric(x["house_cap"], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(score) & (score >= cutoff) & np.isfinite(roi_use) & np.isfinite(cap)
        if not np.any(m):
            continue
        x = x.iloc[np.where(m)[0]].copy()
        stake0 = BANKROLL * stake_frac * float(alpha)
        x["stake_eff"] = np.minimum(stake0, pd.to_numeric(x["house_cap"], errors="coerce").to_numpy(dtype=float))
        x["roi_use"] = roi_use[m]
        x["profit"] = x["stake_eff"].to_numpy(dtype=float) * x["roi_use"].to_numpy(dtype=float)
        x["score_used"] = pd.to_numeric(x[score_col], errors="coerce")
        x["rule_key"] = f"{bt}|{dow}"
        rows.append(x[["week", "stake_eff", "profit", "rule_key", "score_used", "roi_use"]])

    return pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame(
        columns=["week", "stake_eff", "profit", "rule_key", "score_used", "roi_use"]
    )


def _weekly_agg(bets: pd.DataFrame) -> Tuple[float, float, int]:
    if bets.empty:
        return 0.0, 0.0, 0
    stake = float(pd.to_numeric(bets["stake_eff"], errors="coerce").sum())
    pnl = float(pd.to_numeric(bets["profit"], errors="coerce").sum())
    n = int(len(bets))
    return pnl, stake, n


def _train_weekly_distribution(
    df_train: pd.DataFrame, rules_week: pd.DataFrame, alpha: float, use_cap2: bool, train_weeks: List[str]
) -> Dict[str, float]:
    bets = _apply_rules(df_train, rules_week=rules_week, alpha=alpha, use_cap2=use_cap2)
    if bets.empty:
        return {"n_weeks": int(len(train_weeks)), "mean_pnl": float("nan")}
    g = bets.groupby("week", as_index=False).agg(
        stake=("stake_eff", "sum"), pnl=("profit", "sum"), n_bets=("profit", "size")
    )
    gm = g.set_index("week").reindex(train_weeks, fill_value=0.0)
    pnl = gm["pnl"].to_numpy(dtype=float)
    stake = gm["stake"].to_numpy(dtype=float)
    nb = gm["n_bets"].to_numpy(dtype=float)
    roi = np.zeros_like(stake, dtype=float)
    np.divide(pnl, stake, out=roi, where=(stake > 0))
    return {
        "n_weeks": int(len(train_weeks)),
        "mean_pnl": float(np.mean(pnl)),
        "p10_pnl": float(np.quantile(pnl, 0.10)),
        "p50_pnl": float(np.quantile(pnl, 0.50)),
        "p90_pnl": float(np.quantile(pnl, 0.90)),
        "mean_stake": float(np.mean(stake)),
        "p10_stake": float(np.quantile(stake, 0.10)),
        "p50_stake": float(np.quantile(stake, 0.50)),
        "p90_stake": float(np.quantile(stake, 0.90)),
        "mean_n_bets": float(np.mean(nb)),
        "p10_n_bets": float(np.quantile(nb, 0.10)),
        "p50_n_bets": float(np.quantile(nb, 0.50)),
        "p90_n_bets": float(np.quantile(nb, 0.90)),
        "mean_roi": float(np.mean(roi)),
        "p10_roi": float(np.quantile(roi, 0.10)),
        "p50_roi": float(np.quantile(roi, 0.50)),
        "p90_roi": float(np.quantile(roi, 0.90)),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ws = WeekSpec(start="2026-01-12", end="2026-01-18")
    out_pdf = OUT_DIR / f"Analise_Semana_{ws.start}_{ws.end}.pdf"

    weekly = pd.read_csv(WF_WEEKLY)
    daily = pd.read_csv(WF_DAILY, parse_dates=["date"])
    rules = pd.read_csv(WF_RULES)
    fc = pd.read_csv(FC_WEEK) if FC_WEEK.exists() else pd.DataFrame()
    fco = pd.read_csv(FC_ONLINE) if FC_ONLINE.exists() else pd.DataFrame()
    df_all = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])
    df_all["week"] = pd.to_datetime(df_all["BIA_ApostaUTC"]).dt.to_period("W-SUN").astype(str)

    w = weekly[weekly["week"].astype(str) == ws.key].copy()
    if w.empty:
        raise SystemExit(f"Semana não encontrada em {WF_WEEKLY}: {ws.key}")
    w0 = w.iloc[0]
    fcw = fc[fc["week"].astype(str) == ws.key].copy() if (not fc.empty and "week" in fc.columns) else pd.DataFrame()
    fco_w = fco[fco["week"].astype(str) == ws.key].copy() if (not fco.empty and "week" in fco.columns) else pd.DataFrame()

    # daily breakdown for this week
    d = daily[daily["week"].astype(str) == ws.key].copy()
    if not d.empty:
        d = d.sort_values("date")
        d["roi"] = 0.0
        s = pd.to_numeric(d["stake_usd"], errors="coerce").to_numpy(float)
        p = pd.to_numeric(d["profit_cap2_usd"], errors="coerce").to_numpy(float)
        np.divide(p, s, out=d["roi"].to_numpy(), where=(s > 0))
        d["cum_profit"] = pd.to_numeric(d["profit_cap2_usd"], errors="coerce").cumsum()

    # active rules
    r = rules[rules["test_week"].astype(str) == ws.key].copy()
    r["stake_frac"] = pd.to_numeric(r["stake_frac"], errors="coerce")
    r_act = r[(r["status"].astype(str) == "ok") & (r["stake_frac"] > 0)].copy()
    r_act = r_act.sort_values(["bet_type", "dow_pt"])

    alpha = float(w0.get("alpha_global", 1.0))
    # treino = semanas anteriores à semana alvo (W-SUN)
    weeks_sorted = sorted(df_all["week"].unique().tolist())
    if ws.key not in weeks_sorted:
        raise SystemExit(f"Semana {ws.key} não encontrada no dataset {SCORED}")
    i_w = weeks_sorted.index(ws.key)
    train_weeks = weeks_sorted[:i_w]
    df_train = df_all[df_all["week"].isin(train_weeks)].copy()
    df_test = df_all[df_all["week"] == ws.key].copy()

    # realizado (teórico OOS) da semana, usando ROI Real (sem cap) e também cap2 como referência
    bets_test_raw = _apply_rules(df_test, rules_week=r_act, alpha=alpha, use_cap2=False)
    bets_test_cap2 = _apply_rules(df_test, rules_week=r_act, alpha=alpha, use_cap2=True)
    pnl_raw, stake_raw, n_bets_raw = _weekly_agg(bets_test_raw)
    pnl_cap2, stake_cap2, n_bets_cap2 = _weekly_agg(bets_test_cap2)

    # distribuição “esperada” baseada em histórico (aplicando as regras da semana no treino)
    dist_cap2 = _train_weekly_distribution(df_train, r_act, alpha=alpha, use_cap2=True, train_weeks=train_weeks) if train_weeks else {}
    dist_raw = _train_weekly_distribution(df_train, r_act, alpha=alpha, use_cap2=False, train_weeks=train_weeks) if train_weeks else {}

    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Análise da Semana (12–18/01/2026)", styles["Title"]))
    story.append(Paragraph(f"Semana (W-SUN): <b>{ws.key}</b>", styles["Normal"]))
    story.append(Paragraph(f"Gerado em: <b>{date.today().isoformat()}</b>", styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Resumo da performance (OOS global_bayes, cap2)", styles["Heading2"]))
    summary = [
        ["Métrica", "Valor"],
        ["alpha_global", f"{alpha:.3f}"],
        ["n_bets", f"{int(w0['n_bets'])}"],
        ["stake_usd", fmt_money(float(w0["stake_usd"]))],
        ["profit_cap2_usd", fmt_money(float(w0["profit_cap2_usd"]))],
        ["ROI on stake (cap2)", fmt_pct(float(w0["roi_on_stake_cap2"]))],
    ]
    story.append(tbl(summary, col_widths=[220, 260]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Previsto vs realizado (OOS teórico)", styles["Heading2"]))
    if fcw.empty:
        story.append(Paragraph("Sem dados de forecast para essa semana (arquivo de calibração ausente).", styles["Normal"]))
    else:
        r0 = fcw.iloc[0]
        realized = float(r0.get("pnl_theoretical", float("nan")))
        pred_mean = float(r0.get("pred_mean", float("nan")))
        pred_p10 = float(r0.get("pred_p10", float("nan")))
        pred_p50 = float(r0.get("pred_p50", float("nan")))
        pred_p90 = float(r0.get("pred_p90", float("nan")))
        online = float("nan")
        if not fco_w.empty and "pred_mean_bias_adj_online" in fco_w.columns:
            online = float(fco_w.iloc[0].get("pred_mean_bias_adj_online", float("nan")))
        tdata = [
            ["Métrica", "Valor"],
            ["Realizado OOS (PnL teórico, cap2)", fmt_money(realized)],
            ["Realizado OOS (PnL teórico, ROI Real sem cap)", fmt_money(pnl_raw)],
            ["Realizado: stake / n_bets (sem cap)", f"{fmt_money(stake_raw)} / {n_bets_raw}"],
            ["Previsto (média, sem correção)", fmt_money(pred_mean)],
            ["Previsto (p10 / p50 / p90)", f"{fmt_money(pred_p10)} / {fmt_money(pred_p50)} / {fmt_money(pred_p90)}"],
            ["Previsto (média, bias-adjusted on-line)", fmt_money(online)],
            ["Δ Realizado(cap2) - Previsto (média)", fmt_money(realized - pred_mean) if np.isfinite(realized) and np.isfinite(pred_mean) else "—"],
            ["Δ Realizado(cap2) - Previsto (média on-line)", fmt_money(realized - online) if np.isfinite(realized) and np.isfinite(online) else "—"],
        ]
        story.append(tbl(tdata, col_widths=[220, 260]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Esperado vs realizado (histórico do treino aplicando as regras da semana)", styles["Heading2"]))
    if not dist_cap2:
        story.append(Paragraph("Sem histórico suficiente de treino para estimar distribuição esperada.", styles["Normal"]))
    else:
        # comparação usando cap2 (para consistência com o WF) e sem cap (apenas leitura)
        exp_tbl = [
            ["Métrica", "Cap2 (consistente com WF)", "Sem cap (ROI Real)"],
            ["PnL/sem (mean | p10/p50/p90)", f"{fmt_money(dist_cap2['mean_pnl'])} | {fmt_money(dist_cap2['p10_pnl'])}/{fmt_money(dist_cap2['p50_pnl'])}/{fmt_money(dist_cap2['p90_pnl'])}", f"{fmt_money(dist_raw.get('mean_pnl', float('nan')))} | {fmt_money(dist_raw.get('p10_pnl', float('nan')) )}/{fmt_money(dist_raw.get('p50_pnl', float('nan')) )}/{fmt_money(dist_raw.get('p90_pnl', float('nan')) )}"],
            ["Stake/sem (mean | p10/p50/p90)", f"{fmt_money(dist_cap2['mean_stake'])} | {fmt_money(dist_cap2['p10_stake'])}/{fmt_money(dist_cap2['p50_stake'])}/{fmt_money(dist_cap2['p90_stake'])}", f"{fmt_money(dist_raw.get('mean_stake', float('nan')))} | {fmt_money(dist_raw.get('p10_stake', float('nan')) )}/{fmt_money(dist_raw.get('p50_stake', float('nan')) )}/{fmt_money(dist_raw.get('p90_stake', float('nan')) )}"],
            [
                "n_bets/sem (mean | p10/p50/p90)",
                f"{dist_cap2['mean_n_bets']:.1f} | {dist_cap2['p10_n_bets']:.1f}/{dist_cap2['p50_n_bets']:.1f}/{dist_cap2['p90_n_bets']:.1f}",
                f"{dist_raw.get('mean_n_bets', float('nan')):.1f} | {dist_raw.get('p10_n_bets', float('nan')):.1f}/{dist_raw.get('p50_n_bets', float('nan')):.1f}/{dist_raw.get('p90_n_bets', float('nan')):.1f}",
            ],
            [
                "ROI on stake (mean | p10/p50/p90)",
                f"{dist_cap2['mean_roi']:.4f} | {dist_cap2['p10_roi']:.4f}/{dist_cap2['p50_roi']:.4f}/{dist_cap2['p90_roi']:.4f}",
                f"{dist_raw.get('mean_roi', float('nan')):.4f} | {dist_raw.get('p10_roi', float('nan')):.4f}/{dist_raw.get('p50_roi', float('nan')):.4f}/{dist_raw.get('p90_roi', float('nan')):.4f}",
            ],
        ]
        story.append(tbl(exp_tbl, col_widths=[180, 160, 140]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Score vs ROI (apenas apostas selecionadas na semana)", styles["Heading2"]))
    if bets_test_raw.empty:
        story.append(Paragraph("Nenhuma aposta selecionada na semana para análise score↔ROI.", styles["Normal"]))
    else:
        sc = pd.to_numeric(bets_test_raw["score_used"], errors="coerce").to_numpy(float)
        roi = pd.to_numeric(bets_test_raw["roi_use"], errors="coerce").to_numpy(float)
        ok = np.isfinite(sc) & np.isfinite(roi)
        if ok.sum() < 5:
            story.append(Paragraph("Amostra pequena demais para estatística estável.", styles["Normal"]))
        else:
            corr = float(np.corrcoef(sc[ok], roi[ok])[0, 1])
            # bins de score (quintis)
            q = np.quantile(sc[ok], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            rows_bin = [["Bin score", "n", "ROI médio", "PnL médio (USD)"]]
            stake = pd.to_numeric(bets_test_raw["stake_eff"], errors="coerce").to_numpy(float)[ok]
            pnl = stake * roi[ok]
            for i in range(5):
                lo, hi = q[i], q[i + 1]
                m = (sc[ok] >= lo) & (sc[ok] <= hi) if i == 4 else ((sc[ok] >= lo) & (sc[ok] < hi))
                if not np.any(m):
                    continue
                rows_bin.append(
                    [
                        f"[{lo:.3f},{hi:.3f}]",
                        str(int(m.sum())),
                        f"{float(np.mean(roi[ok][m])):.4f}",
                        fmt_money(float(np.mean(pnl[m]))),
                    ]
                )
            story.append(Paragraph(f"Correlação (score, ROI): <b>{corr:.3f}</b>", styles["Normal"]))
            story.append(tbl(rows_bin, col_widths=[140, 50, 90, 130]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Contexto de múltiplas semanas (variância vs possível regime)", styles["Heading2"]))
    # junta: realizado (WF_WEEKLY) e previsto (forecast csv) para últimas semanas até a alvo
    w_wf = weekly.copy()
    w_wf["week"] = w_wf["week"].astype(str)
    fc2 = fc.copy()
    if not fc2.empty and "week" in fc2.columns:
        fc2["week"] = fc2["week"].astype(str)
    fco2 = fco.copy()
    if not fco2.empty and "week" in fco2.columns:
        fco2["week"] = fco2["week"].astype(str)

    all_weeks = w_wf["week"].astype(str).tolist()
    if ws.key in all_weeks:
        i0 = all_weeks.index(ws.key)
        w_sel = all_weeks[max(0, i0 - 7) : i0 + 1]  # 8 semanas até a semana alvo
    else:
        w_sel = []

    if not w_sel:
        story.append(Paragraph("Semanas insuficientes para contexto.", styles["Normal"]))
    else:
        rows_ctx = [["week", "real cap2", "pred_mean", "p10..p90", "inside?", "stake", "n_bets", "α"]]
        for wk in w_sel:
            rw = w_wf[w_wf["week"] == wk]
            if rw.empty:
                continue
            rr = rw.iloc[0]
            real_cap2 = float(rr.get("profit_cap2_usd", float("nan")))
            stake = float(rr.get("stake_usd", float("nan")))
            nb = int(rr.get("n_bets", 0)) if str(rr.get("n_bets", "")).strip() != "" else 0
            alpha_w = float(rr.get("alpha_global", float("nan")))

            pred_mean = p10 = p90 = float("nan")
            inside = "—"
            fr = fc2[fc2["week"] == wk] if (not fc2.empty and "week" in fc2.columns) else pd.DataFrame()
            if not fr.empty:
                pred_mean = float(fr.iloc[0].get("pred_mean", float("nan")))
                p10 = float(fr.iloc[0].get("pred_p10", float("nan")))
                p90 = float(fr.iloc[0].get("pred_p90", float("nan")))
                if np.isfinite(real_cap2) and np.isfinite(p10) and np.isfinite(p90):
                    inside = "sim" if (real_cap2 >= p10 and real_cap2 <= p90) else "não"
            rows_ctx.append(
                [
                    wk,
                    fmt_money(real_cap2),
                    fmt_money(pred_mean),
                    f"{fmt_money(p10)}..{fmt_money(p90)}",
                    inside,
                    fmt_money(stake),
                    str(nb),
                    (f"{alpha_w:.3f}" if np.isfinite(alpha_w) else "—"),
                ]
            )
        story.append(
            Paragraph(
                "Leitura: se o realizado cap2 cair consistentemente fora do intervalo previsto (p10..p90) por várias semanas, "
                "isso sugere possível quebra de regime; se ocorrer pontualmente, é compatível com variância esperada.",
                styles["Normal"],
            )
        )
        story.append(tbl(rows_ctx, col_widths=[90, 70, 70, 90, 45, 55, 45, 35]))

    story.append(Paragraph("Quebra diária (OOS)", styles["Heading2"]))
    if d.empty:
        story.append(Paragraph("Sem dias registrados nessa semana (stake=0).", styles["Normal"]))
    else:
        dd = [
            ["date", "stake_usd", "profit_cap2_usd", "ROI dia", "cum_profit"],
        ]
        for _, row in d.iterrows():
            dd.append(
                [
                    row["date"].date().isoformat(),
                    fmt_money(float(row["stake_usd"])),
                    fmt_money(float(row["profit_cap2_usd"])),
                    fmt_pct(float(row["roi"])),
                    fmt_money(float(row["cum_profit"])),
                ]
            )
        story.append(tbl(dd, col_widths=[90, 100, 110, 80, 100]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Segmentos ativos (regras aplicadas nessa semana)", styles["Heading2"]))
    if r_act.empty:
        story.append(Paragraph("Nenhum segmento ativo (stake_frac=0 ou status!=ok).", styles["Normal"]))
    else:
        rr = [["bet_type", "dow_pt", "score_col", "cutoff", "stake_frac", "status"]]
        for _, row in r_act.iterrows():
            rr.append(
                [
                    str(row["bet_type"]),
                    str(row["dow_pt"]),
                    str(row["score_col"]),
                    f"{float(row['cutoff']):.3f}",
                    fmt_pct(float(row["stake_frac"])),
                    str(row["status"]),
                ]
            )
        story.append(tbl(rr, col_widths=[55, 95, 155, 55, 70, 60]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Notas", styles["Heading2"]))
    story.append(
        Paragraph(
            "<b>Sobre α (alpha_global)</b>: é um fator (0..1) aplicado a todos os stakes na semana, "
            "calculado no treino de cada passo para satisfazer constraints de risco globais (exposição diária e drawdown). "
            "Na prática, o stake efetivo por aposta é <b>min(banca × stake_frac × α, house_cap)</b>.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "<b>Interpretação (por que pode divergir do previsto)</b>: mesmo quando o scoring está correto, "
            "o PnL semanal tem alta variância e depende de: (i) poucas observações na semana, (ii) distribuição de odds/ROI "
            "capada (cap2), (iii) mudança de regime (mercado), (iv) mudança do conjunto de segmentos ativos (política dinâmica), "
            "e (v) α reduzir/exacerbar exposição conforme restrições globais.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "<b>Aplicabilidade do modelo</b>: este relatório avalia a política como geradora de decisões. "
            "A aderência operacional deve ser checada via `bet_id` (payload + log JSONL + score gravado). "
            "Se a aderência for 100%, divergências entre previsto e realizado passam a ser atribuídas ao risco/variância do mercado "
            "e à incerteza estatística (não a erro de pipeline).",
            styles["Normal"],
        )
    )

    doc = SimpleDocTemplate(str(out_pdf), pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    doc.build(story)
    print(str(out_pdf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

