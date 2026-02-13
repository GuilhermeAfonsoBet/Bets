#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera PDF visual de arquitetura:
- Modelo antigo (v4 original)
- Modelo novo (fase 1 com pools separados)
- Impactos observados
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


PAGE_W, PAGE_H = A4
MARGIN = 16 * mm


def draw_header(c: canvas.Canvas, title: str, subtitle: str = "") -> None:
    c.setFillColor(colors.HexColor("#0f172a"))
    c.setFont("Helvetica-Bold", 16)
    c.drawString(MARGIN, PAGE_H - MARGIN, title)
    if subtitle:
        c.setFillColor(colors.HexColor("#334155"))
        c.setFont("Helvetica", 10)
        c.drawString(MARGIN, PAGE_H - MARGIN - 14, subtitle)
    c.setStrokeColor(colors.HexColor("#cbd5e1"))
    c.line(MARGIN, PAGE_H - MARGIN - 20, PAGE_W - MARGIN, PAGE_H - MARGIN - 20)


def draw_box(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: list[str],
    fill: str = "#f8fafc",
    stroke: str = "#94a3b8",
    title_color: str = "#0f172a",
) -> None:
    c.setFillColor(colors.HexColor(fill))
    c.setStrokeColor(colors.HexColor(stroke))
    c.roundRect(x, y, w, h, 6, stroke=1, fill=1)

    c.setFillColor(colors.HexColor(title_color))
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x + 8, y + h - 15, title)

    c.setFillColor(colors.HexColor("#334155"))
    c.setFont("Helvetica", 8.5)
    ty = y + h - 29
    for ln in lines:
        c.drawString(x + 8, ty, f"- {ln}")
        ty -= 11
        if ty < y + 7:
            break


def draw_arrow(
    c: canvas.Canvas,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    label: str | None = None,
    color: str = "#64748b",
) -> None:
    c.setStrokeColor(colors.HexColor(color))
    c.setFillColor(colors.HexColor(color))
    c.setLineWidth(1.2)
    c.line(x1, y1, x2, y2)

    # ponta da seta (triangulo simples)
    size = 4
    if abs(x2 - x1) >= abs(y2 - y1):
        if x2 >= x1:
            tri = [(x2, y2), (x2 - size * 2, y2 + size), (x2 - size * 2, y2 - size)]
        else:
            tri = [(x2, y2), (x2 + size * 2, y2 + size), (x2 + size * 2, y2 - size)]
    else:
        if y2 >= y1:
            tri = [(x2, y2), (x2 - size, y2 - size * 2), (x2 + size, y2 - size * 2)]
        else:
            tri = [(x2, y2), (x2 - size, y2 + size * 2), (x2 + size, y2 + size * 2)]
    p = c.beginPath()
    p.moveTo(*tri[0])
    p.lineTo(*tri[1])
    p.lineTo(*tri[2])
    p.close()
    c.drawPath(p, stroke=0, fill=1)

    if label:
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.HexColor("#475569"))
        lx = (x1 + x2) / 2 - 22
        ly = (y1 + y2) / 2 + 6
        c.drawString(lx, ly, label)


def draw_paragraph(c: canvas.Canvas, x: float, y: float, text: str, max_width: float, leading: float = 12) -> float:
    c.setFillColor(colors.HexColor("#1e293b"))
    c.setFont("Helvetica", 9.5)
    words = text.split()
    line = []
    cur_y = y
    for w in words:
        candidate = " ".join(line + [w])
        if c.stringWidth(candidate, "Helvetica", 9.5) <= max_width:
            line.append(w)
        else:
            c.drawString(x, cur_y, " ".join(line))
            cur_y -= leading
            line = [w]
    if line:
        c.drawString(x, cur_y, " ".join(line))
        cur_y -= leading
    return cur_y


def page_old_architecture(c: canvas.Canvas) -> None:
    draw_header(
        c,
        "Arquitetura antiga (v4 original)",
        "Fluxo principal antes da separacao T+0 x temporal",
    )

    top_y = PAGE_H - MARGIN - 90
    box_w = 78 * mm
    box_h = 34 * mm

    draw_box(
        c, MARGIN, top_y, box_w, box_h,
        "Detector H3B (WS)",
        ["Lendo odds em tempo real", "Detecta reversao", "Enfileira evento"],
        fill="#e0f2fe", stroke="#7dd3fc"
    )
    draw_box(
        c, MARGIN + 92 * mm, top_y, 42 * mm, box_h,
        "Fila unica",
        ["Eventos aguardam", "Ordem FIFO"],
        fill="#fef3c7", stroke="#fcd34d"
    )
    draw_box(
        c, MARGIN + 140 * mm, top_y, 45 * mm, box_h,
        "Worker",
        ["1 worker (ou poucos)", "Sem separacao de etapa"],
        fill="#fee2e2", stroke="#fca5a5"
    )

    draw_arrow(c, MARGIN + box_w, top_y + 17 * mm, MARGIN + 92 * mm, top_y + 17 * mm)
    draw_arrow(c, MARGIN + 134 * mm, top_y + 17 * mm, MARGIN + 140 * mm, top_y + 17 * mm)

    y2 = top_y - 58 * mm
    draw_box(
        c, MARGIN + 20 * mm, y2, 70 * mm, 46 * mm,
        "Etapa T+0",
        ["Back/Lay API", "Captura odd imediata", "Rapida (segundos)"],
        fill="#dcfce7", stroke="#86efac"
    )
    draw_box(
        c, MARGIN + 98 * mm, y2, 70 * mm, 46 * mm,
        "Etapa temporal",
        ["Refresh t+3/6/10/15/20", "Acompanha curva", "Lenta (~20s+)"],
        fill="#ede9fe", stroke="#c4b5fd"
    )
    draw_arrow(c, MARGIN + 162 * mm, top_y, MARGIN + 162 * mm, y2 + 46 * mm, "processa")
    draw_arrow(c, MARGIN + 90 * mm, y2 + 23 * mm, MARGIN + 98 * mm, y2 + 23 * mm, "depois")

    y3 = y2 - 44 * mm
    draw_box(
        c, MARGIN + 52 * mm, y3, 95 * mm, 28 * mm,
        "Banco de dados",
        ["Grava resultado ao final", "Fila cresce quando worker fica ocupado"],
        fill="#f1f5f9", stroke="#94a3b8"
    )
    draw_arrow(c, MARGIN + 133 * mm, y2, MARGIN + 133 * mm, y3 + 28 * mm)

    ytxt = y3 - 22
    ytxt = draw_paragraph(
        c,
        MARGIN,
        ytxt,
        "Problema principal: o mesmo worker fazia a parte rapida (T+0) e a parte longa (temporal). "
        "Com rajadas de sinais, o worker ficava ocupado por muito tempo e a fila aumentava.",
        PAGE_W - 2 * MARGIN,
    )
    draw_paragraph(
        c,
        MARGIN,
        ytxt,
        "Consequencia observada antes da mudanca: fila media relevante e espera alta para novos eventos.",
        PAGE_W - 2 * MARGIN,
    )


def page_new_architecture(c: canvas.Canvas) -> None:
    draw_header(
        c,
        "Arquitetura nova (Fase 1)",
        "Separacao de esteiras: pool T+0 (4) e pool temporal (2)",
    )

    y = PAGE_H - MARGIN - 88
    draw_box(
        c, MARGIN, y, 68 * mm, 32 * mm,
        "Detector H3B (WS)",
        ["Detecta sinal", "Envia para fila T+0"],
        fill="#e0f2fe", stroke="#7dd3fc"
    )
    draw_box(
        c, MARGIN + 74 * mm, y, 38 * mm, 32 * mm,
        "Fila T+0",
        ["Critica", "Prioridade alta"],
        fill="#fef3c7", stroke="#fcd34d"
    )
    draw_box(
        c, MARGIN + 118 * mm, y, 68 * mm, 32 * mm,
        "Pool T+0 (4 workers)",
        ["So captura imediata", "worker_id 1..4"],
        fill="#dcfce7", stroke="#86efac"
    )
    draw_arrow(c, MARGIN + 68 * mm, y + 16 * mm, MARGIN + 74 * mm, y + 16 * mm)
    draw_arrow(c, MARGIN + 112 * mm, y + 16 * mm, MARGIN + 118 * mm, y + 16 * mm)

    y2 = y - 52 * mm
    draw_box(
        c, MARGIN + 12 * mm, y2, 72 * mm, 36 * mm,
        "Banco: registro inicial",
        ["Resultado T+0 rapido", "telemetria de fila", "temporal_deferred=true"],
        fill="#f1f5f9", stroke="#94a3b8"
    )
    draw_box(
        c, MARGIN + 92 * mm, y2 + 2 * mm, 34 * mm, 30 * mm,
        "Fila temporal",
        ["jobs de curva"],
        fill="#fef3c7", stroke="#fcd34d"
    )
    draw_box(
        c, MARGIN + 132 * mm, y2, 54 * mm, 36 * mm,
        "Pool temporal (2)",
        ["t+3/6/10/15/20", "temporal_worker_id"],
        fill="#ede9fe", stroke="#c4b5fd"
    )
    draw_arrow(c, MARGIN + 152 * mm, y, MARGIN + 152 * mm, y2 + 36 * mm, "enqueue")
    draw_arrow(c, MARGIN + 126 * mm, y2 + 18 * mm, MARGIN + 132 * mm, y2 + 18 * mm)

    y3 = y2 - 40 * mm
    draw_box(
        c, MARGIN + 46 * mm, y3, 112 * mm, 28 * mm,
        "Banco: patch temporal",
        ["Atualiza hypothesis_details com temporal, lay_temporal e telemetria final"],
        fill="#f1f5f9", stroke="#94a3b8"
    )
    draw_arrow(c, MARGIN + 158 * mm, y2, MARGIN + 158 * mm, y3 + 28 * mm)

    ytxt = y3 - 20
    ytxt = draw_paragraph(
        c,
        MARGIN,
        ytxt,
        "Resultado pratico: T+0 deixa de esperar o temporal. A fila critica cai, e o temporal segue em paralelo.",
        PAGE_W - 2 * MARGIN,
    )
    draw_paragraph(
        c,
        MARGIN,
        ytxt,
        "Logs confirmam workers separados e banco mostra worker_id/temporal_worker_id.",
        PAGE_W - 2 * MARGIN,
    )


def page_impacts(c: canvas.Canvas) -> None:
    draw_header(
        c,
        "Impactos observados e implicacoes",
        "Comparativo antes vs depois da Fase 1 (dados reais da operacao)",
    )

    x = MARGIN
    y = PAGE_H - MARGIN - 80
    col_w = [70 * mm, 48 * mm, 48 * mm]
    row_h = 11 * mm
    rows = [
        ("Indicador", "Antes", "Depois (Fase 1)"),
        ("Queue depth enq (media)", "6.93", "0.04"),
        ("Queue depth enq (p90)", "14", "proximo de 0-1"),
        ("Queue wait (media)", "42.956 ms", "0.3 ms"),
        ("Queue wait (p90)", "alto", "1.0 ms"),
        ("Queue wait (p95)", "alto", "2.3 ms"),
        ("Cobertura telemetria", "parcial", "100% na janela"),
        ("Back/Lay temporal", "assimetrico", "quase paridade"),
    ]

    for r_idx, row in enumerate(rows):
        yy = y - r_idx * row_h
        for c_idx, val in enumerate(row):
            xx = x + sum(col_w[:c_idx])
            fill = "#e2e8f0" if r_idx == 0 else ("#f8fafc" if r_idx % 2 == 0 else "#ffffff")
            c.setFillColor(colors.HexColor(fill))
            c.setStrokeColor(colors.HexColor("#cbd5e1"))
            c.rect(xx, yy - row_h, col_w[c_idx], row_h, stroke=1, fill=1)
            c.setFillColor(colors.HexColor("#0f172a"))
            c.setFont("Helvetica-Bold" if r_idx == 0 else "Helvetica", 8.5)
            c.drawString(xx + 4, yy - row_h + 4, str(val))

    ytxt = y - len(rows) * row_h - 18
    ytxt = draw_paragraph(
        c,
        MARGIN,
        ytxt,
        "Leitura simples: a fila do caminho critico caiu de forma drastica apos separar os pools. "
        "Agora o gargalo principal deixa de ser T+0 e passa a ser monitoramento temporal em picos.",
        PAGE_W - 2 * MARGIN,
    )
    ytxt = draw_paragraph(
        c,
        MARGIN,
        ytxt,
        "Impactos operacionais de aumentar workers: mais CPU, mais memoria, mais chamadas no proxy e possivel aumento de timeout se passar do ponto ideal.",
        PAGE_W - 2 * MARGIN,
    )

    draw_box(
        c, MARGIN, ytxt - 40, 84 * mm, 34 * mm,
        "Recomendacao atual",
        ["Manter 4 workers T+0", "Manter 2 workers temporal", "Reavaliar com janela 12-24h"],
        fill="#dcfce7", stroke="#86efac"
    )
    draw_box(
        c, MARGIN + 92 * mm, ytxt - 40, 94 * mm, 34 * mm,
        "Gatilhos para proxima mudanca",
        ["Se p95 fila T+0 > 1s: subir T+0", "Se temporal acumular: subir temporal", "Sempre monitorar CPU/RAM/proxy"],
        fill="#fef3c7", stroke="#fcd34d"
    )


def page_roadmap(c: canvas.Canvas) -> None:
    draw_header(
        c,
        "Proposta de evolucao da arquitetura (proxima versao)",
        "Roadmap simples para estabilidade + latencia baixa",
    )
    y = PAGE_H - MARGIN - 72
    items = [
        ("Fase 1 (ja ativa)", "Separar T+0 e temporal; pools 4/2; telemetria de fila"),
        ("Fase 2", "Autoajuste de workers por meta (p95 fila T+0 < 1s)"),
        ("Fase 3", "Descartar evento stale no T+0 (evita processar fila velha)"),
        ("Fase 4", "Dashboard com p50/p90/p95/p99 e custo de proxy por worker"),
    ]
    for i, (title, desc) in enumerate(items, start=1):
        yy = y - (i - 1) * 34 * mm
        draw_box(
            c, MARGIN, yy - 24 * mm, 186 * mm, 22 * mm,
            f"{i}. {title}",
            [desc],
            fill="#f8fafc",
            stroke="#cbd5e1",
        )

    note_y = y - len(items) * 34 * mm + 8
    draw_paragraph(
        c,
        MARGIN,
        note_y,
        "Mensagem final: seu objetivo de fila < 1s no caminho critico e viavel. "
        "A separacao de esteiras foi o passo estrutural mais importante e ja mostrou ganho real.",
        PAGE_W - 2 * MARGIN,
    )


def generate_pdf(output_path: Path) -> None:
    c = canvas.Canvas(str(output_path), pagesize=A4)
    c.setTitle("Arquitetura antiga vs nova - BetinAsia")
    c.setAuthor("BetinAsia Operacao")

    page_old_architecture(c)
    c.showPage()
    page_new_architecture(c)
    c.showPage()
    page_impacts(c)
    c.showPage()
    page_roadmap(c)

    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#64748b"))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    c.drawRightString(PAGE_W - MARGIN, 10 * mm, f"Gerado em {stamp}")
    c.save()


def main() -> None:
    out = Path("/workspace/betinasia_bot/docs/arquitetura_v4_antiga_vs_nova_fluxos_2026-02-13.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    generate_pdf(out)
    print(f"PDF gerado: {out}")


if __name__ == "__main__":
    main()
