#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera um PDF curto com a memória do trabalho feito neste chat/projeto:
- mudanças em scoring (stdout/logs/auditoria)
- mudanças em relatórios (Relatorio_BayesGlobal_Mesa_Profissional)
- mudanças no walk-forward/experimentos (bankroll/cap_bin)
- estado atual e próximos passos

Saída:
  /workspace/docs/project_context/Trabalhos_Chat_<YYYY-MM-DD>.pdf
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors


OUT_DIR = Path("/workspace/docs/project_context")


def P(s: str, styles):
    return Paragraph(s, styles["BodyText"])


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    out = OUT_DIR / f"Trabalhos_Chat_{today}.pdf"

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Memória do trabalho (chat)</b>", styles["Title"]))
    story.append(Paragraph(f"<b>Data</b>: {today}", styles["BodyText"]))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("<b>1) Objetivo e premissas</b>", styles["Heading2"]))
    story.append(
        P(
            "Construir e manter uma plataforma/estratégia de apostas com <b>scoring determinístico e auditável</b>, "
            "e um portfólio walk-forward robusto (Bayes Global). O requisito operacional é que o score gravado "
            "(Excel → <b>ApostaLive.rf_prob</b>) seja reprodutível e rastreável por <b>bet_id</b>.",
            styles,
        )
    )

    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("<b>2) Scoring: principais correções e contrato operacional</b>", styles["Heading2"]))
    story.append(
        P(
            "<b>Weekdays (Seg–Qua)</b>: `score_logit_weekdays_cli.py` usa modelos por dia e devolve `proba,decision` no stdout. "
            "Foi restaurada a compatibilidade com PAD: por padrão o stdout tem <b>2 linhas</b> (header + última linha do payload). "
            "Para auditoria existe `--stdout-all-rows`.",
            styles,
        )
    )
    story.append(
        P(
            "<b>Weekend (Sex–Dom)</b>: `score_logit_by_dow_cli.py` aplica calibração isotônica (SexDom) e loga `bet_id`.",
            styles,
        )
    )
    story.append(
        P(
            "<b>Logs JSONL</b>: os CLIs registram `bet_id`, `ts`, `model_path`, `payload_hash`, `proba`, `decision` para auditoria determinística.",
            styles,
        )
    )

    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("<b>3) Auditoria de consistência (Excel ↔ logs)</b>", styles["Heading2"]))
    story.append(
        P(
            "Com `ResumoApostas_PBI_final_20.01.2026.xlsx` e `scoring_weekdays.jsonl` atualizados, foi feito join por `bet_id` "
            "e verificado match por arredondamento a 6 casas.",
            styles,
        )
    )
    t = Table(
        [
            ["Data", "Dia", "N", "Match6", "MAE"],
            ["2026-01-19", "segunda-feira", "25", "100%", "0.0"],
            ["2026-01-20", "terça-feira", "35", "100%", "0.0"],
        ],
        colWidths=[3.0 * cm, 4.0 * cm, 1.2 * cm, 2.0 * cm, 2.0 * cm],
    )
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(t)
    story.append(
        P(
            "<b>Conclusão</b>: quando o Excel está atualizado e os logs têm `bet_id`, a aderência é mensurável e pode ser exigida como 100%.",
            styles,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("<b>4) Relatórios</b>", styles["Heading2"]))
    story.append(
        P(
            "O `Relatorio_BayesGlobal_Mesa_Profissional` foi atualizado para incluir: "
            "<b>curva PnL vs banca</b> (inclui pico), <b>testes formais</b> de edge na banca 2,3k, "
            "diagnóstico do cenário máximo (variância vs modelo) e teste formal ROI vs stake máximo.",
            styles,
        )
    )

    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("<b>5) Walk-forward / estratégia</b>", styles["Heading2"]))
    story.append(
        P(
            "A estratégia atual (p10_p70) otimiza por segmento (DoW×FT/FH) em walk-forward usando seleção Bayesiana "
            "com gating P(μ&gt;0)≥70% e objetivo conservador (p10 do posterior).",
            styles,
        )
    )
    story.append(
        P(
            "Foram adicionados recursos para experimentos auditáveis: `--bankroll` e `--out-suffix` no walk-forward, "
            "e um modo experimental `cap_bin` (stake máximo como feature de seleção por bins de house_cap).",
            styles,
        )
    )

    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("<b>6) Estado atual e próximos passos</b>", styles["Heading2"]))
    story.append(
        P(
            "Próximos passos recomendados: (i) manter rotina diária de auditoria Excel↔logs por `bet_id`, "
            "(ii) ao aparecerem novas inconsistências (Ter/Qua), isolar por payload+log e versionar a correção, "
            "(iii) avançar no estudo de função-objetivo (p10 vs p50/híbridos) e, se desejado, políticas diferentes por faixa de banca.",
            styles,
        )
    )

    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    doc.build(story)
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

