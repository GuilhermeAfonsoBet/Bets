#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera um PDF explicando as métricas de qualidade de previsão usadas na calibração:
- Bias, MAE, RMSE
- Coverage (80%/90%)
- PIT (Probability Integral Transform)
- CRPS (Continuous Ranked Probability Score)

Inclui também um resumo numérico a partir de:
  analysis_proba_raw/pro_portfolio_all/forecast_calibration_global_bayes.csv

Saída:
  analysis_proba_raw/pro_portfolio_all/Relatorio_Calibracao_Forecast_PnL_<YYYY-MM-DD>.pdf
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")
CALIB = OUT_DIR / "forecast_calibration_global_bayes.csv"
BEFORE_AFTER = OUT_DIR / "before_after_global_comparison.csv"
CALIB_ONLINE = OUT_DIR / "forecast_calibration_global_bayes_online_bias.csv"


def _fmt_usd(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"USD {x:,.1f}"


def _fmt_pct(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{100.0 * x:.1f}%"


def main() -> int:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

    styles = getSampleStyleSheet()
    style_h = styles["Heading2"]
    style_p = styles["BodyText"]
    style_p.leading = 12

    today = date.today().isoformat()
    pdf_path = OUT_DIR / f"Relatorio_Calibracao_Forecast_PnL_{today}.pdf"

    if not CALIB.exists():
        raise SystemExit(f"Arquivo não encontrado: {CALIB}")

    df = pd.read_csv(CALIB)
    n_folds = int(df.shape[0])

    err = df["error"].to_numpy(dtype=float)  # y - pred_mean
    bias = float(np.mean(err))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))

    cov80 = float(np.mean((df["pnl_theoretical"] >= df["pred_p10"]) & (df["pnl_theoretical"] <= df["pred_p90"])))
    cov90 = float(np.mean((df["pnl_theoretical"] >= df["pred_p05"]) & (df["pnl_theoretical"] <= df["pred_p95"])))
    pit_mean = float(np.mean(df["pit"].to_numpy(dtype=float)))
    crps_mean = float(np.mean(df["crps"].to_numpy(dtype=float)))

    # decomposição (médias)
    m_stake = float(df["error_stake_component"].mean()) if "error_stake_component" in df.columns else float("nan")
    m_roi = float(df["error_roi_component"].mean()) if "error_roi_component" in df.columns else float("nan")
    m_inter = float(df["error_interaction"].mean()) if "error_interaction" in df.columns else float("nan")
    m_cov = float(df["pred_cov_term"].mean()) if "pred_cov_term" in df.columns else float("nan")

    # comparativo antes vs depois (se existir)
    ba = None
    if BEFORE_AFTER.exists():
        try:
            ba = pd.read_csv(BEFORE_AFTER)
        except Exception:
            ba = None

    # on-line bias adjusted (se existir)
    online = None
    if CALIB_ONLINE.exists():
        try:
            online = pd.read_csv(CALIB_ONLINE)
        except Exception:
            online = None

    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm)
    story = []

    story.append(Paragraph("Relatório — Calibração de Forecast do Modelo (PnL)", styles["Title"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"Data: <b>{today}</b>", style_p))
    story.append(Paragraph(f"Fonte numérica: <b>{CALIB.name}</b> (folds: <b>{n_folds}</b>)", style_p))
    story.append(Spacer(1, 12))

    story.append(Paragraph("1. O que estamos medindo", style_h))
    story.append(
        Paragraph(
            "Para cada semana do walk-forward, o modelo escolhe regras \\(\\theta_t\\) usando apenas o passado. "
            "Em seguida medimos se a <b>distribuição prevista</b> para o PnL semanal está coerente com o "
            "<b>PnL teórico realizado</b> (aplicando \\(\\theta_t\\) na semana teste).",
            style_p,
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "<b>Importante</b>: isso é uma calibração “teórico vs teórico”. Ela mede qualidade do modelo como "
            "gerador de decisão+previsão no dataset, mas não captura fricções de execução real (atraso, liquidez, limites dinâmicos, etc.).",
            style_p,
        )
    )
    story.append(Spacer(1, 12))

    story.append(Paragraph("2. Métricas de erro (pontuais)", style_h))
    story.append(Paragraph("<b>Erro</b>: \\(e_t = y_t - \\hat{\\mu}_t\\), onde \\(y_t\\) é o PnL teórico realizado e \\(\\hat{\\mu}_t\\) é a média prevista.", style_p))
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Bias (viés)</b>: média do erro. Se Bias < 0, o modelo está <b>otimista</b> (prevendo PnL maior do que realiza).", style_p))
    story.append(Paragraph("<b>MAE</b>: erro absoluto médio. Medida robusta de “tamanho típico” do erro.", style_p))
    story.append(Paragraph("<b>RMSE</b>: raiz do erro quadrático médio. Penaliza mais os erros grandes (sensível a outliers).", style_p))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Resumo numérico (PnL semanal)", styles["Heading3"]))
    t = Table(
        [
            ["Bias", "MAE", "RMSE"],
            [_fmt_usd(bias), _fmt_usd(mae), _fmt_usd(rmse)],
        ],
        colWidths=[5.0 * cm, 5.0 * cm, 5.0 * cm],
    )
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 14))

    story.append(Paragraph("3. Métricas probabilísticas (calibração)", style_h))
    story.append(
        Paragraph(
            "<b>Coverage</b>: frequência com que o realizado cai dentro do intervalo previsto. "
            "Ex.: Coverage 80% mede quantas vezes \\(y_t\\) cai entre p10 e p90 do preditivo. "
            "Se o modelo for bem calibrado, a coverage observada deve ficar próxima do nível nominal (80%/90%).",
            style_p,
        )
    )
    story.append(
        Paragraph(
            "<b>PIT</b> (Probability Integral Transform): para cada semana, calcula \\(\\text{PIT}_t = F_t(y_t)\\), "
            "ou seja, a posição do realizado dentro da CDF prevista. Se a distribuição prevista for correta, "
            "os PITs ao longo do tempo se comportam como Uniforme(0,1) (média ~0,5). "
            "PIT muito baixo indica que o realizado veio pior do que o previsto; PIT muito alto indica o oposto.",
            style_p,
        )
    )
    story.append(
        Paragraph(
            "<b>CRPS</b> (Continuous Ranked Probability Score): mede a distância entre a distribuição prevista "
            "e o realizado. Generaliza o MAE para previsões probabilísticas. <b>Menor é melhor</b>.",
            style_p,
        )
    )
    story.append(Spacer(1, 10))

    t2 = Table(
        [
            ["Coverage 80% (p10..p90)", "Coverage 90% (p05..p95)", "PIT (média)", "CRPS (médio)"],
            [_fmt_pct(cov80), _fmt_pct(cov90), f"{pit_mean:.3f}" if np.isfinite(pit_mean) else "—", _fmt_usd(crps_mean)],
        ],
        colWidths=[4.2 * cm, 4.2 * cm, 3.0 * cm, 3.6 * cm],
    )
    t2.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    story.append(t2)
    story.append(Spacer(1, 14))

    story.append(PageBreak())
    story.append(Paragraph("4. Por que o PnL previsto pode “inflar” (Stake vs ROI)", style_h))
    story.append(
        Paragraph(
            "Como \\(\\text{PnL} = \\text{Stake} \\times \\text{ROI}\\), um forecast pode parecer otimista por dois caminhos:\n"
            "<br/>- o modelo projeta <b>stake</b> alto (mais volume do que acontece)\n"
            "<br/>- o modelo projeta <b>ROI</b> alto (retorno por dólar piora)\n"
            "<br/><br/>"
            "No nosso caso, o preditivo é gerado reamostrando semanas passadas (bootstrap Bayesiano), "
            "então o modelo aprende uma distribuição conjunta (Stake, ROI) implícita.",
            style_p,
        )
    )
    story.append(Spacer(1, 10))
    story.append(Paragraph("Decomposição (médias)", styles["Heading3"]))
    story.append(
        Paragraph(
            "Uma decomposição útil para o erro semanal \\(y-\\hat{P}\\) é:\n"
            "<br/>\\(y-\\hat{P} = (S-\\hat{S})\\hat{R} + \\hat{S}(R-\\hat{R}) + (S-\\hat{S})(R-\\hat{R}) - \\text{cov}\\),\n"
            "<br/>onde \\(\\text{cov}=E[S\\cdot R]-E[S]E[R]\\) captura a dependência entre stake e ROI no preditivo.",
            style_p,
        )
    )
    story.append(Spacer(1, 8))
    t3 = Table(
        [
            ["Componente stake", "Componente ROI", "Interação", "cov (dependência)"],
            [_fmt_usd(m_stake), _fmt_usd(m_roi), _fmt_usd(m_inter), _fmt_usd(m_cov)],
        ],
        colWidths=[4.0 * cm, 4.0 * cm, 4.0 * cm, 4.0 * cm],
    )
    t3.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    story.append(t3)
    story.append(Spacer(1, 12))

    story.append(Paragraph("5. Como usar essas métricas para melhorar o modelo", style_h))
    story.append(
        Paragraph(
            "<b>1) Correção de viés (bias correction)</b>: manter a distribuição prevista, mas ajustar a média "
            "por um viés estimado out-of-sample (ex.: subtrair o Bias médio ou um Bias móvel).",
            style_p,
        )
    )
    story.append(
        Paragraph(
            "<b>2) Recalibração de incerteza</b>: se a coverage está abaixo do nominal, os intervalos estão estreitos. "
            "Dá para corrigir escalando a dispersão (ex.: inflar desvio) ou recalibrando quantis com base em PIT.",
            style_p,
        )
    )
    story.append(
        Paragraph(
            "<b>3) Modelar resíduos</b>: criar um modelo simples para \\(e_t\\) como função do contexto (nº de apostas, "
            "stake, alpha, nº de segmentos ativos, etc.) e ajustar forecast condicionalmente.",
            style_p,
        )
    )
    story.append(
        Paragraph(
            "<b>4) Atualizar o PDF principal</b>: usar Bias/coverage para reportar um “cenário calibrado” conservador "
            "(lucro esperado ajustado) além do cenário base.",
            style_p,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("6. Avaliação e conclusão (o que os números significam na prática)", style_h))
    story.append(
        Paragraph(
            "Estas métricas respondem duas perguntas: (i) o modelo é <b>otimista/pessimista</b> na média? "
            "(ii) o modelo está <b>calibrado</b> na incerteza (intervalos/quantis)?",
            style_p,
        )
    )
    story.append(Spacer(1, 8))
    pred_mean_mean = float(np.mean(df["pred_mean"].to_numpy(dtype=float))) if "pred_mean" in df.columns else float("nan")
    pred_mean_cal = float(pred_mean_mean + bias) if np.isfinite(pred_mean_mean) and np.isfinite(bias) else float("nan")
    story.append(
        Paragraph(
            f"No estado atual, o modelo apresenta <b>Bias = {_fmt_usd(bias)}</b>. "
            "Bias negativo significa que, em média, o realizado ficou abaixo do previsto (otimismo). "
            f"A média prevista (E[pred_mean]) é {_fmt_usd(pred_mean_mean)} e, ao aplicar correção de viés "
            f"(E[pred_mean]+Bias), cai para {_fmt_usd(pred_mean_cal)}. "
            f"O erro típico (MAE) é {_fmt_usd(mae)} e o RMSE é {_fmt_usd(rmse)}: "
            "isso indica que a incerteza semanal é grande e que a previsão pontual sozinha não é suficiente para tomada de decisão.",
            style_p,
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            f"A calibração probabilística também está fraca: Coverage 80% = {_fmt_pct(cov80)} (ideal ~80%) "
            f"e Coverage 90% = {_fmt_pct(cov90)} (ideal ~90%). "
            "Isso sugere que os intervalos previstos estão <b>estreitos</b> (subestimando risco). "
            f"PIT médio = {pit_mean:.3f} (ideal ~0,5) reforça a assimetria para resultados piores que o previsto.",
            style_p,
        )
    )
    story.append(Spacer(1, 10))

    story.append(Paragraph("Impacto na estratégia", styles["Heading3"]))
    story.append(
        Paragraph(
            "Quando Bias é negativo e Coverage fica muito abaixo do nominal, o efeito prático é:\n"
            "<br/>- projeções de lucro tendem a estar infladas\n"
            "<br/>- intervalos/quantis (p10/p90 etc.) subestimam perdas\n"
            "<br/>- decisões que dependem de “p05/p10 positivo” ou de VaR estimado podem ficar excessivamente confiantes.",
            style_p,
        )
    )
    story.append(Spacer(1, 10))

    if online is not None and (not online.empty) and {"diff_real_minus_model_raw", "diff_real_minus_model_bias_adj", "abs_diff_raw", "abs_diff_bias_adj"}.issubset(set(online.columns)):
        story.append(Paragraph("Bias-adjusted on-line (walk-forward)", styles["Heading3"]))
        e0 = online["diff_real_minus_model_raw"].to_numpy(dtype=float)
        e1 = online["diff_real_minus_model_bias_adj"].to_numpy(dtype=float)
        mae0 = float(np.mean(np.abs(e0)))
        mae1 = float(np.mean(np.abs(e1)))
        b0 = float(np.mean(e0))
        b1 = float(np.mean(e1))
        story.append(
            Paragraph(
                "O ajuste de bias operacional deve ser <b>on-line</b>: para cada semana, estimamos o bias usando apenas semanas anteriores "
                "(ex.: média móvel) e aplicamos na previsão da semana corrente. Isso evita correção ex-post que “fecha a conta” no período completo.",
                style_p,
            )
        )
        t_on = Table(
            [
                ["Métrica", "Modelo cru", "Bias-adjusted on-line"],
                ["Bias (média de y - pred)", _fmt_usd(b0), _fmt_usd(b1)],
                ["MAE", _fmt_usd(mae0), _fmt_usd(mae1)],
            ],
            colWidths=[6.0 * cm, 5.0 * cm, 5.0 * cm],
        )
        t_on.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            )
        )
        story.append(Spacer(1, 6))
        story.append(t_on)

    if ba is not None and (not ba.empty) and {"scenario", "mean_week", "std_week", "sharpe_annual", "roi_on_stake", "fc_bias", "fc_mae", "fc_cov80"}.issubset(set(ba.columns)):
        story.append(Paragraph("Antes vs depois (com correções)", styles["Heading3"]))
        rows = [["cenário", "mean/sem", "std/sem", "Sharpe ann", "ROI/$", "Bias forecast", "MAE", "Cov80"]]
        for _, r in ba.iterrows():
            rows.append(
                [
                    str(r["scenario"]),
                    f"{float(r['mean_week']):,.1f}",
                    f"{float(r['std_week']):,.1f}",
                    f"{float(r['sharpe_annual']):.3f}" if np.isfinite(float(r["sharpe_annual"])) else "—",
                    f"{float(r['roi_on_stake']):.4f}" if np.isfinite(float(r["roi_on_stake"])) else "—",
                    f"{float(r['fc_bias']):,.1f}" if np.isfinite(float(r["fc_bias"])) else "—",
                    f"{float(r['fc_mae']):,.1f}" if np.isfinite(float(r["fc_mae"])) else "—",
                    _fmt_pct(float(r["fc_cov80"])) if np.isfinite(float(r["fc_cov80"])) else "—",
                ]
            )
        tt = Table(rows, colWidths=[2.0 * cm, 2.2 * cm, 2.2 * cm, 2.0 * cm, 1.6 * cm, 2.3 * cm, 1.8 * cm, 1.6 * cm])
        tt.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                ]
            )
        )
        story.append(tt)
        story.append(Spacer(1, 8))
        story.append(
            Paragraph(
                "Leitura: a calibração pode ser usada de duas formas.\n"
                "<br/>- <b>Somente no reporte</b>: ajusta expectativas (ex.: média prevista corrigida por Bias), sem alterar o OOS realizado.\n"
                "<br/>- <b>Dentro do otimizador</b>: muda seleção/stakes/cutoffs e, portanto, pode alterar (e até degradar) o OOS realizado.\n"
                "<br/><br/>Na prática, com amostra curta, é comum que usar correção por combinação diretamente no otimizador gere over-penalização. "
                "Por isso, nesta versão, recomendamos usar calibração por combinação como <b>monitoramento/reporte</b> (com shrinkage) "
                "e tratar o problema de risco principalmente via <b>recalibração de dispersão</b> (coverage/PIT).",
                style_p,
            )
        )

    story.append(Spacer(1, 10))
    story.append(Paragraph("Recomendação objetiva", styles["Heading3"]))
    story.append(
        Paragraph(
            "Eu recomendo manter duas visões em paralelo:\n"
            "<br/>- <b>Visão de decisão</b>: usar a correção por combinação de forma conservadora e com limites (shrinkage + mínimo de observações).\n"
            "<br/>- <b>Visão de reporte</b>: sempre reportar também um cenário “ajustado por Bias” e uma nota explícita sobre Coverage/PIT.\n"
            "<br/><br/>"
            "Se Coverage continuar muito abaixo do nominal, a próxima evolução deve ser uma <b>recalibração de dispersão</b> (widening dos quantis/intervalos), "
            "porque hoje o problema não é só a média (Bias), mas principalmente a subestimação do risco.",
            style_p,
        )
    )

    doc.build(story)
    print(str(pdf_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

