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
MODE = "global_bayes_roll12_robust_p10_p70"
WF_WEEKLY = OUT_DIR / f"oos_walkforward_{MODE}_weekly.csv"
WF_DAILY = OUT_DIR / f"oos_walkforward_{MODE}_daily.csv"
WF_RULES = OUT_DIR / f"oos_walkforward_{MODE}_selected_rules.csv"
FC_WEEK = OUT_DIR / f"forecast_calibration_{MODE}.csv"
FC_ONLINE = OUT_DIR / "forecast_calibration_global_bayes_online_bias.csv"
SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")
BANKROLL_BASE = 2300.0
# banca “stake máximo (p95)” conforme relatório 17/01/2026
BANKROLL_MAX_P95 = 63_205.0
CALIB_FLOOR = 0.005
CALIB_SEGQUI = Path("/workspace/clv_calib_SegQui.json")
CALIB_SEXDOM = Path("/workspace/clv_calib_SexDom.json")


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


def _apply_isotonic_from_json(p: np.ndarray, json_path: Path, floor: float) -> np.ndarray:
    """
    Aplica uma calibração isotonic (salva como {isotonic: {x:[], y:[]}}) via interpolação monotônica.
    """
    if not json_path.exists():
        return np.clip(p.astype(float), floor, 1.0 - floor)
    import json

    obj = json.loads(json_path.read_text(encoding="utf-8"))
    x = np.asarray(obj["isotonic"]["x"], dtype=float)
    y = np.asarray(obj["isotonic"]["y"], dtype=float)
    pp = p.astype(float)
    out = np.interp(pp, x, y, left=y[0], right=y[-1])
    return np.clip(out, floor, 1.0 - floor)


def _roi_from_result_and_odds(result: object, odds: float) -> float:
    if not np.isfinite(odds) or odds <= 1e-9:
        return float("nan")
    s = str(result).strip().lower()
    if s == "win":
        return float(odds - 1.0)
    if s in {"lose", "loss"}:
        return -1.0
    if s in {"halfwin", "half win"}:
        return float((odds - 1.0) / 2.0)
    if s in {"halfloss", "halflose", "half loss", "half lose"}:
        return -0.5
    if s in {"push", "void", "refund", "cancelled", "canceled"}:
        return 0.0
    return float("nan")

def _select_bets(
    df: pd.DataFrame, rules_week: pd.DataFrame, alpha: float, bankroll: float
) -> pd.DataFrame:
    """
    Aplica as regras da semana em um dataframe (treino ou teste).
    Retorna bets selecionadas com stake_eff, ROI (raw e cap2) e PnL (raw e cap2).
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
        cap = pd.to_numeric(x["house_cap"], errors="coerce").to_numpy(dtype=float)
        # Importante: a seleção NÃO depende de cap2 vs sem cap (apenas score e dados válidos).
        # cap2 só altera o PnL (truncando ROI Real em 2.0).
        # Obs: cap pode ser +inf (sem limite). Isso é válido e NÃO deve excluir a aposta.
        m = np.isfinite(score) & (score >= cutoff) & (~np.isnan(cap)) & (cap > 0)
        if not np.any(m):
            continue
        x = x.iloc[np.where(m)[0]].copy()
        stake0 = float(bankroll) * stake_frac * float(alpha)
        x["stake_eff"] = np.minimum(stake0, pd.to_numeric(x["house_cap"], errors="coerce").to_numpy(dtype=float))
        x["score_used"] = pd.to_numeric(x[score_col], errors="coerce")
        # ROI da planilha (diagnóstico)
        x["roi_sheet"] = pd.to_numeric(x.get("ROI Real"), errors="coerce")
        # ROI calculado via odds + resultado (mais confiável para \"sem cap\")
        def _col(name: str) -> pd.Series:
            return x[name] if name in x.columns else pd.Series(np.nan, index=x.index)

        odds_series = pd.to_numeric(_col("Odd Aposta Realizada"), errors="coerce")
        for nm in ["BetinAsia.got price", "BetinAsia.Odds", "RebelBetting.Odds", "Odd Indicada no RB"]:
            odds_series = odds_series.combine_first(pd.to_numeric(_col(nm), errors="coerce"))
        odds = odds_series.to_numpy(dtype=float)
        res = x.get("RebelBetting.Result")
        roi_calc = np.array([_roi_from_result_and_odds(r0, o0) for r0, o0 in zip(res, odds)], dtype=float)
        # manter apenas apostas com ROI calculável (para não divergir da seleção OOS, que requer ROI finito)
        keep = np.isfinite(roi_calc)
        if not np.any(keep):
            continue
        x = x.iloc[np.where(keep)[0]].copy()
        roi_calc = roi_calc[keep]
        x["roi_calc"] = roi_calc
        x["roi_calc_cap2"] = np.minimum(roi_calc, 2.0)
        x["profit_calc"] = x["stake_eff"].to_numpy(dtype=float) * x["roi_calc"].to_numpy(dtype=float)
        x["profit_calc_cap2"] = x["stake_eff"].to_numpy(dtype=float) * x["roi_calc_cap2"].to_numpy(dtype=float)
        x["rule_key"] = f"{bt}|{dow}"
        rows.append(
            x[
                [
                    "week",
                    "date",
                    "stake_eff",
                    "rule_key",
                    "score_used",
                    "roi_sheet",
                    "roi_calc",
                    "roi_calc_cap2",
                    "profit_calc",
                    "profit_calc_cap2",
                ]
            ]
        )

    return pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame(
        columns=[
            "week",
            "date",
            "stake_eff",
            "rule_key",
            "score_used",
            "roi_sheet",
            "roi_calc",
            "roi_calc_cap2",
            "profit_calc",
            "profit_calc_cap2",
        ]
    )


def _weekly_agg(bets: pd.DataFrame, profit_col: str) -> Tuple[float, float, int, float]:
    if bets.empty:
        return 0.0, 0.0, 0, 0.0
    stake = float(pd.to_numeric(bets["stake_eff"], errors="coerce").sum())
    pnl = float(pd.to_numeric(bets[profit_col], errors="coerce").sum())
    n = int(len(bets))
    roi = (pnl / stake) if stake > 0 else 0.0
    return pnl, stake, n, roi


def _weekly_dist_from_bets(
    bets: pd.DataFrame, train_weeks: List[str], profit_col: str
) -> Dict[str, float]:
    """
    Constrói a distribuição semanal (incluindo semanas com 0 apostas) a partir de um DF de bets já selecionadas.
    """
    if bets.empty:
        return {"n_weeks": int(len(train_weeks)), "mean_pnl": float("nan")}
    g = bets.groupby("week", as_index=False).agg(
        stake=("stake_eff", "sum"),
        pnl=(profit_col, "sum"),
        n_bets=(profit_col, "size"),
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


def _train_weekly_distribution_multi(
    df_train: pd.DataFrame, rules_week: pd.DataFrame, alpha: float, train_weeks: List[str], bankroll: float
) -> Tuple[Dict[str, float], Dict[str, float]]:
    bets = _select_bets(df_train, rules_week=rules_week, alpha=alpha, bankroll=bankroll)
    return (
        _weekly_dist_from_bets(bets, train_weeks=train_weeks, profit_col="profit_calc_cap2"),
        _weekly_dist_from_bets(bets, train_weeks=train_weeks, profit_col="profit_calc"),
    )


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
    df_all["date"] = pd.to_datetime(df_all["BIA_ApostaUTC"]).dt.floor("D")
    # criar colunas calibradas (proba_cal_*) para aplicar as regras do WF em qui/sex/sáb/dom
    if "proba_cal_segqui" not in df_all.columns:
        if "proba_raw_segqui" in df_all.columns:
            pr = pd.to_numeric(df_all["proba_raw_segqui"], errors="coerce").to_numpy(dtype=float)
            df_all["proba_cal_segqui"] = _apply_isotonic_from_json(pr, CALIB_SEGQUI, floor=CALIB_FLOOR)
        else:
            df_all["proba_cal_segqui"] = np.nan
    if "proba_cal_sexdom" not in df_all.columns:
        if "proba_raw_sexdom" in df_all.columns:
            pr = pd.to_numeric(df_all["proba_raw_sexdom"], errors="coerce").to_numpy(dtype=float)
            df_all["proba_cal_sexdom"] = _apply_isotonic_from_json(pr, CALIB_SEXDOM, floor=CALIB_FLOOR)
        else:
            df_all["proba_cal_sexdom"] = np.nan

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

    # realizado (teórico OOS) da semana: mesma seleção; cap2 NÃO remove apostas (stake/n_bets iguais).
    bets_test = _select_bets(df_test, rules_week=r_act, alpha=alpha, bankroll=BANKROLL_BASE)
    pnl_calc, stake_sel, n_bets_sel, roi_calc = _weekly_agg(bets_test, profit_col="profit_calc")
    pnl_calc_cap2, _, _, roi_calc_cap2 = _weekly_agg(bets_test, profit_col="profit_calc_cap2")
    # sanity: stake e n_bets não devem variar entre cap2 e sem cap
    stake_sel = float(stake_sel)
    n_bets_sel = int(n_bets_sel)

    # distribuição “esperada” baseada em histórico (aplicando as regras da semana no treino)
    dist_cap2, dist_raw = _train_weekly_distribution_multi(df_train, r_act, alpha=alpha, train_weeks=train_weeks, bankroll=BANKROLL_BASE) if train_weeks else ({}, {})

    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Análise da Semana (12–18/01/2026)", styles["Title"]))
    story.append(Paragraph(f"Semana (W-SUN): <b>{ws.key}</b>", styles["Normal"]))
    story.append(Paragraph(f"Gerado em: <b>{date.today().isoformat()}</b>", styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Resumo da performance (OOS global_bayes)", styles["Heading2"]))
    cap2_official_pnl = float(w0.get("profit_cap2_usd", float("nan")))
    cap2_official_stake = float(w0.get("stake_usd", float("nan")))
    cap2_official_n = int(w0.get("n_bets", 0))
    cap2_official_roi = float(w0.get("roi_on_stake_cap2", float("nan")))
    summary = [
        ["Métrica", "Valor"],
        ["alpha_global", f"{alpha:.3f}"],
        ["Realizado (cap2, oficial WF): n_bets / stake / PnL / ROI", f"{cap2_official_n} / {fmt_money(cap2_official_stake)} / {fmt_money(cap2_official_pnl)} / {fmt_pct(cap2_official_roi)}"],
        ["Realizado (sem cap, via odds+resultado): n_bets / stake / PnL / ROI", f"{n_bets_sel} / {fmt_money(stake_sel)} / {fmt_money(pnl_calc)} / {fmt_pct(roi_calc)}"],
    ]
    story.append(tbl(summary, col_widths=[220, 260]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Cap2: conceito e impacto na semana", styles["Heading2"]))
    story.append(
        Paragraph(
            "<b>Definição (cap2)</b>: para reduzir a influência de poucos eventos com ROI muito alto (cauda pesada), "
            "o estudo usa uma versão capada do ROI: <b>ROI_cap2 = min(ROI_Real, 2.0)</b>. "
            "<b>Isso não remove apostas</b> (stake e n_bets ficam iguais). Ele só altera o PnL/ROI calculados "
            "para análise de risco/robustez (long-run).",
            styles["Normal"],
        )
    )
    if not bets_test.empty:
        # Diferença cap2 vs sem cap usando ROI calculado (odds+resultado)
        delta = pd.to_numeric(bets_test["profit_calc"], errors="coerce") - pd.to_numeric(bets_test["profit_calc_cap2"], errors="coerce")
        m = pd.to_numeric(bets_test["roi_calc"], errors="coerce") > 2.0
        n_aff = int(np.sum(m.to_numpy(dtype=bool)))
        stake_aff = float(pd.to_numeric(bets_test.loc[m, "stake_eff"], errors="coerce").sum()) if n_aff > 0 else 0.0
        delta_sum = float(pd.to_numeric(delta, errors="coerce").sum()) if np.isfinite(delta).any() else 0.0

        # Diagnóstico: ROI da planilha pode conter outliers incompatíveis com odds/resultados
        roi_sheet = pd.to_numeric(bets_test["roi_sheet"], errors="coerce")
        stake0 = pd.to_numeric(bets_test["stake_eff"], errors="coerce")
        pnl_sheet = float((stake0 * roi_sheet).sum()) if np.isfinite(roi_sheet.to_numpy(float)).any() else float("nan")
        pnl_sheet_cap2 = float((stake0 * np.minimum(roi_sheet.to_numpy(float), 2.0)).sum()) if np.isfinite(roi_sheet.to_numpy(float)).any() else float("nan")
        n_sheet_gt2 = int(np.sum((roi_sheet.to_numpy(float) > 2.0) & np.isfinite(roi_sheet.to_numpy(float))))
        story.append(
            Paragraph(
                f"<b>Nesta semana</b>:<br/>"
                f"- Cap2 (oficial WF): <b>{fmt_money(cap2_official_pnl)}</b><br/>"
                f"- Sem cap (via odds+resultado): <b>{fmt_money(pnl_calc)}</b><br/>"
                f"- Diferença cap2 vs sem cap (via odds+resultado): <b>{fmt_money(delta_sum)}</b> "
                f"(apostas com ROI&gt;2: <b>{n_aff}</b>, stake nessas apostas: <b>{fmt_money(stake_aff)}</b>).<br/>"
                f"<br/><b>Diagnóstico (ROI Real da planilha)</b>: stake×ROI_Real gera sem cap = <b>{fmt_money(pnl_sheet)}</b> "
                f"e cap2 = <b>{fmt_money(pnl_sheet_cap2)}</b> (n apostas com ROI_Real&gt;2: <b>{n_sheet_gt2}</b>). "
                "Se aparecerem ROIs muito acima do compatível com as odds, isso indica dado/coluna incorreta para uso como ROI.",
                styles["Normal"],
            )
        )
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
            ["Realizado OOS (PnL teórico no arquivo, cap2)", fmt_money(realized)],
            ["Realizado (cap2, oficial WF): PnL / stake / n_bets / ROI", f"{fmt_money(cap2_official_pnl)} / {fmt_money(cap2_official_stake)} / {cap2_official_n} / {fmt_pct(cap2_official_roi)}"],
            ["Realizado (sem cap, via odds+resultado): PnL / stake / n_bets / ROI", f"{fmt_money(pnl_calc)} / {fmt_money(stake_sel)} / {n_bets_sel} / {fmt_pct(roi_calc)}"],
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
        exp_tbl = [
            ["Variante", "Realizado", "Esperado (mean)", "Esperado (p10/p50/p90)"],
            [
                "cap2",
                f"PnL {fmt_money(cap2_official_pnl)} | stake {fmt_money(cap2_official_stake)} | n {cap2_official_n} | ROI {fmt_pct(cap2_official_roi)}",
                f"PnL {fmt_money(dist_cap2['mean_pnl'])} | stake {fmt_money(dist_cap2['mean_stake'])} | n {dist_cap2['mean_n_bets']:.1f} | ROI {dist_cap2['mean_roi']:.4f}",
                f"PnL {fmt_money(dist_cap2['p10_pnl'])}/{fmt_money(dist_cap2['p50_pnl'])}/{fmt_money(dist_cap2['p90_pnl'])} | "
                f"stake {fmt_money(dist_cap2['p10_stake'])}/{fmt_money(dist_cap2['p50_stake'])}/{fmt_money(dist_cap2['p90_stake'])} | "
                f"n {dist_cap2['p10_n_bets']:.1f}/{dist_cap2['p50_n_bets']:.1f}/{dist_cap2['p90_n_bets']:.1f} | "
                f"ROI {dist_cap2['p10_roi']:.4f}/{dist_cap2['p50_roi']:.4f}/{dist_cap2['p90_roi']:.4f}",
            ],
            [
                "sem cap",
                f"PnL {fmt_money(pnl_calc)} | stake {fmt_money(stake_sel)} | n {n_bets_sel} | ROI {fmt_pct(roi_calc)}",
                f"PnL {fmt_money(dist_raw['mean_pnl'])} | stake {fmt_money(dist_raw['mean_stake'])} | n {dist_raw['mean_n_bets']:.1f} | ROI {dist_raw['mean_roi']:.4f}",
                f"PnL {fmt_money(dist_raw['p10_pnl'])}/{fmt_money(dist_raw['p50_pnl'])}/{fmt_money(dist_raw['p90_pnl'])} | "
                f"stake {fmt_money(dist_raw['p10_stake'])}/{fmt_money(dist_raw['p50_stake'])}/{fmt_money(dist_raw['p90_stake'])} | "
                f"n {dist_raw['p10_n_bets']:.1f}/{dist_raw['p50_n_bets']:.1f}/{dist_raw['p90_n_bets']:.1f} | "
                f"ROI {dist_raw['p10_roi']:.4f}/{dist_raw['p50_roi']:.4f}/{dist_raw['p90_roi']:.4f}",
            ],
        ]
        texp = Table(exp_tbl, colWidths=[50, 170, 135, 125], repeatRows=1)
        texp.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("LEADING", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(texp)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Score vs ROI (apenas apostas selecionadas na semana)", styles["Heading2"]))
    if bets_test.empty:
        story.append(Paragraph("Nenhuma aposta selecionada na semana para análise score↔ROI.", styles["Normal"]))
    else:
        sc = pd.to_numeric(bets_test["score_used"], errors="coerce").to_numpy(float)
        roi = pd.to_numeric(bets_test["roi_calc"], errors="coerce").to_numpy(float)
        ok = np.isfinite(sc) & np.isfinite(roi)
        if ok.sum() < 5:
            story.append(Paragraph("Amostra pequena demais para estatística estável.", styles["Normal"]))
        else:
            corr = float(np.corrcoef(sc[ok], roi[ok])[0, 1])
            # bins de score (quintis)
            q = np.quantile(sc[ok], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            rows_bin = [["Bin score", "n", "ROI médio", "PnL médio (USD)"]]
            stake = pd.to_numeric(bets_test["stake_eff"], errors="coerce").to_numpy(float)[ok]
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
        # tabela compacta (colunas menores + quebra de linha em week/p10..p90)
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
                    wk.replace("/", "/\n"),
                    fmt_money(real_cap2),
                    fmt_money(pred_mean),
                    f"{fmt_money(p10)}..\n{fmt_money(p90)}",
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
        tctx = Table(rows_ctx, colWidths=[80, 60, 60, 70, 40, 55, 40, 28], repeatRows=1)
        tctx.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("LEADING", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(tctx)

    story.append(Paragraph("Quebra diária (OOS)", styles["Heading2"]))
    if d.empty:
        story.append(Paragraph("Sem dias registrados nessa semana (stake=0).", styles["Normal"]))
    else:
        # cap2 diário oficial do WF + sem cap diário (via odds+resultado) reconstruído dos bets selecionados
        d_cap2 = d[["date", "stake_usd", "profit_cap2_usd"]].copy()
        if not bets_test.empty:
            d_raw = bets_test.groupby("date", as_index=False).agg(
                stake_usd=("stake_eff", "sum"),
                profit_raw_usd=("profit_calc", "sum"),
            )
        else:
            d_raw = pd.DataFrame(columns=["date", "stake_usd", "profit_raw_usd"])

        d2 = pd.merge(d_cap2, d_raw, on=["date"], how="outer", suffixes=("_cap2", "_raw")).sort_values("date")
        # usar stake do WF (cap2) como referência; se faltar, cair no stake do raw
        if "stake_usd_cap2" in d2.columns and "stake_usd_raw" in d2.columns:
            d2["stake_usd"] = pd.to_numeric(d2["stake_usd_cap2"], errors="coerce").fillna(pd.to_numeric(d2["stake_usd_raw"], errors="coerce"))
        elif "stake_usd_cap2" in d2.columns:
            d2["stake_usd"] = pd.to_numeric(d2["stake_usd_cap2"], errors="coerce")
        elif "stake_usd_raw" in d2.columns:
            d2["stake_usd"] = pd.to_numeric(d2["stake_usd_raw"], errors="coerce")
        else:
            d2["stake_usd"] = 0.0
        d2["profit_cap2_usd"] = pd.to_numeric(d2["profit_cap2_usd"], errors="coerce").fillna(0.0)
        d2["profit_raw_usd"] = pd.to_numeric(d2["profit_raw_usd"], errors="coerce").fillna(0.0)
        d2["roi_cap2"] = 0.0
        d2["roi_raw"] = 0.0
        s = pd.to_numeric(d2["stake_usd"], errors="coerce").to_numpy(float)
        pc2 = pd.to_numeric(d2["profit_cap2_usd"], errors="coerce").to_numpy(float)
        pr = pd.to_numeric(d2["profit_raw_usd"], errors="coerce").to_numpy(float)
        np.divide(pc2, s, out=d2["roi_cap2"].to_numpy(), where=(s > 0))
        np.divide(pr, s, out=d2["roi_raw"].to_numpy(), where=(s > 0))

        dd = [
            ["date", "stake", "PnL cap2", "PnL sem cap", "ROI cap2", "ROI sem cap"],
        ]
        for _, row in d2.iterrows():
            dd.append(
                [
                    row["date"].date().isoformat(),
                    fmt_money(float(row["stake_usd"])),
                    fmt_money(float(row["profit_cap2_usd"])),
                    fmt_money(float(row["profit_raw_usd"])),
                    fmt_pct(float(row["roi_cap2"])),
                    fmt_pct(float(row["roi_raw"])),
                ]
            )
        story.append(tbl(dd, col_widths=[80, 70, 70, 70, 70, 70]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Cenário: banca 63.205 (stake máximo p95 do relatório 17/01)", styles["Heading2"]))
    bets_test_max = _select_bets(df_test, rules_week=r_act, alpha=alpha, bankroll=BANKROLL_MAX_P95)
    pnl_calc_m, stake_m, n_m, roi_calc_m = _weekly_agg(bets_test_max, profit_col="profit_calc")
    pnl_calc_cap2_m, _, _, roi_calc_cap2_m = _weekly_agg(bets_test_max, profit_col="profit_calc_cap2")
    dist_cap2_m, dist_raw_m = _train_weekly_distribution_multi(df_train, r_act, alpha=alpha, train_weeks=train_weeks, bankroll=BANKROLL_MAX_P95) if train_weeks else ({}, {})
    scen = [
        ["Métrica", "Valor"],
        ["Realizado (sem cap, via odds+resultado): PnL / stake / n_bets / ROI", f"{fmt_money(pnl_calc_m)} / {fmt_money(stake_m)} / {n_m} / {fmt_pct(roi_calc_m)}"],
        ["Realizado (cap2, via odds+resultado): PnL / stake / n_bets / ROI", f"{fmt_money(pnl_calc_cap2_m)} / {fmt_money(stake_m)} / {n_m} / {fmt_pct(roi_calc_cap2_m)}"],
    ]
    if dist_cap2_m:
        scen.append(["Esperado cap2 (mean PnL / stake / n_bets / ROI)", f"{fmt_money(dist_cap2_m['mean_pnl'])} / {fmt_money(dist_cap2_m['mean_stake'])} / {dist_cap2_m['mean_n_bets']:.1f} / {dist_cap2_m['mean_roi']:.4f}"])
        scen.append(["Esperado sem cap (mean PnL / stake / n_bets / ROI)", f"{fmt_money(dist_raw_m['mean_pnl'])} / {fmt_money(dist_raw_m['mean_stake'])} / {dist_raw_m['mean_n_bets']:.1f} / {dist_raw_m['mean_roi']:.4f}"])
    story.append(tbl(scen, col_widths=[220, 260]))
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

