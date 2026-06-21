#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera uma versão "executiva/dissertativa" de um relatório .md já existente.

Motivação:
- O relatório base é ótimo para leitura estatística e auditoria.
- Para decisões, queremos um prefácio curto com interpretação, limites e próximos passos,
  preservando o relatório original como apêndice.

Uso:
  python3 betinasia_bot/docs/enrich_report_for_pdf.py \
    --input-md analise_b808_2d__full.md \
    --output-md analise_b808_2d__full__executivo.md
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple


def _rx(text: str, pattern: str) -> Optional[re.Match]:
    return re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)


def _grab(text: str, pattern: str, group: int = 1, default: str = "N/A") -> str:
    m = _rx(text, pattern)
    if not m:
        return default
    try:
        return (m.group(group) or "").strip() or default
    except Exception:
        return default


def _grab_float_ci(text: str, pattern: str) -> Tuple[str, str]:
    """
    Extrai algo do tipo: +2.063% (IC90 [+1.502%, +2.616%])
    Retorna (mean, ci) como strings já formatadas.
    """
    m = _rx(text, pattern)
    if not m:
        return ("N/A", "N/A")
    mean = (m.group(1) or "").strip()
    ci = (m.group(2) or "").strip()
    return (mean, ci)


def build_preface(md: str, *, report_name: str) -> str:
    exec_date = _grab(md, r"^\*\*Data da execução:\*\*\s*(.+?)\s*$")
    recorte = _grab(md, r"^\-\s+\*\*Recorte\*\*:\s*(.+?)\s*$")
    amostra = _grab(md, r"^\-\s+\*\*Amostra\*\*:\s*(.+?)\s*$")
    janela = _grab(md, r"^\-\s+\*\*Janela efetiva.*?\*\*:\s*(.+?)\s*$")
    coortes = _grab(md, r"^\-\s+\*\*Coortes.*?\*\*:\s*(.+?)\s*$")
    placar = _grab(md, r"^\-\s+\*\*Cobertura de placar.*?\*\*:\s*(.+?)\s*$")
    dom = _grab(md, r"^\-\s+\*\*DOM\*\*:\s*(.+?)\s*$")

    clv_mean, clv_ci = _grab_float_ci(
        md,
        r"CLV pre\-match.*?média robusta por jogo\s*([+\-]?\d+(?:\.\d+)?)%\s*\(IC90\s*\[([^\]]+)\]\)",
    )

    diff_mean = _grab(md, r"^\|\s*Diff BS vs WS \(média\)\s*\|\s*([^|]+)\|", default="N/A")

    # Performance (p50/p95) do bloco 2.0b
    # Esperado (colunas): mean | p50 | p95 | N
    def _grab_row(metric_label: str) -> Tuple[str, str, str, str]:
        m = _rx(
            md,
            rf"^\|\s*API\s*\(2-4s\)\s*\|\s*{re.escape(metric_label)}\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*$",
        )
        if not m:
            return ("N/A", "N/A", "N/A", "N/A")
        return (m.group(1).strip(), m.group(2).strip(), m.group(3).strip(), m.group(4).strip())

    det_mean, det_p50, det_p95, _det_n = _grab_row("lag_det→click")
    bs_mean, bs_p50, bs_p95, _bs_n = _grab_row("lag_click→betslip")
    e2e_mean, e2e_p50, e2e_p95, e2e_n = _grab_row("lag_e2e (soma)")
    tot_mean, tot_p50, tot_p95, tot_n = _grab_row("audit_total (duração)")
    ov_mean, ov_p50, ov_p95, ov_n = _grab_row("overhead (total - e2e)")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""# {report_name} — Versão Executiva (com interpretação)
**Gerado em:** {now}  
**Fonte:** relatório estatístico base (apêndice), extraído do repositório e enriquecido com interpretação.

---
## 0) Leitura executiva (o que importa e como decidir)

Este documento foi escrito para responder a uma pergunta prática:
**o robô está capturando valor de preço de forma consistente, com execução suficientemente rápida para que o sinal (CLV) seja preservado, e com qualidade de dado adequada para extrapolar para uma fase de execução?**

### 0.1 O que o recorte mostra (em linguagem de decisão)
- **Recorte**: {recorte}
- **Amostra**: {amostra}
- **Janela efetiva**: {janela}
- **Coortes operacionais**: {coortes}
- **Cobertura de placar/ROI**: {placar}
- **DOM**: {dom}

O sinal mais importante neste recorte é **CLV pre‑match positivo e robusto**:
- **CLV (pre‑match, por jogo)**: **{clv_mean}%** com **IC90 [{clv_ci}]**.

Interpretação: em média, quando a auditoria aponta edge e a execução ocorre no pre‑match, o preço efetivamente obtido no betslip tende a “ganhar do mercado” (closing) em magnitude material, com confiança estatística **por jogo** (robustez contra correlação intra‑jogo).

### 0.2 O que eu não concluiria ainda (evitar erro caro)
- **ROI** ainda tem **incerteza grande** neste recorte. Mesmo quando a média parece positiva/negativa em alguma coorte, os intervalos por jogo são largos e a cobertura de jogos finalizados nem sempre é completa. Para ROI, trate este PDF como *diagnóstico*, não como validação final.
- **In‑match**: CLV “clássico” (vs closing pré‑jogo) **não é interpretável** in‑match. Use in‑match para outras perguntas (latência, integridade do pipeline), mas não para concluir edge pre‑match.
- **DOM ausente (N=0)**: não há comparação API vs DOM neste recorte; qualquer conclusão “API melhor que DOM” aqui seria especulação.

### 0.3 Diagnóstico operacional (execução)
Do ponto de vista de execução (detecção→betslip), em linguagem direta:
- **detecção→clique**: p50≈{det_p50}ms, p95≈{det_p95}ms
- **clique→betslip**: p50≈{bs_p50}ms, p95≈{bs_p95}ms
- **tempo instrumentado (detecção→clique→betslip)**: p50≈{e2e_p50}ms, p95≈{e2e_p95}ms (N={e2e_n})
- **tempo total observado (detecção→betslip, “wall/total”)**: p50≈{tot_p50}ms, p95≈{tot_p95}ms (N={tot_n})
- **overhead (total − instrumentado)**: p50≈{ov_p50}ms, p95≈{ov_p95}ms (N={ov_n})  
  *Interpretação operacional*: overhead agrega espera fora das duas etapas instrumentadas (ex.: fila, retries, pausas e latências externas).

Tradução: o pipeline API está “rápido o suficiente” na mediana, mas ainda existe cauda (p95) e regimes lentos. A lição operacional é simples:
**o edge é perecível**; portanto, regimes lentos devem ser tratados como *degradação de qualidade* e não como “só mais devagar”.

### 0.3b Glossário rápido (para não gerar ambiguidade)
- **Tempo total observado**: tempo “de parede” do pipeline completo de auditoria (o que o operador sente), incluindo tudo que não está explicitamente instrumentado.
- **Tempo instrumentado**: soma das etapas instrumentadas **detecção→clique** + **clique→betslip**.
- **overhead**: diferença entre total observado e instrumentado; é um proxy para **fila/esperas/retries**.
- **p50 / p95**: percentis. p95 deve ser ≥ p50; se aparecer invertido, é sinal de erro de parsing/relato.

### 0.4 Sinal de consistência: BS vs WS
O relatório base mede também a diferença média entre o preço no betslip (BS) e o preço via WS (WS):
- **Diff BS vs WS (média)**: {diff_mean}

Em termos de governança de execução, esse indicador funciona como “termômetro”:
quando o betslip sistematicamente piora vs WS (ou vice‑versa), você está medindo fricções reais (latência, proteção de stake, limites, redirecionamento, sessão ruim, etc.). Isso é valioso porque independe do resultado do jogo.

---
## 0.5 Recomendação prática (como usar este relatório na operação)
Se o objetivo é entrar na fase de execução sem contaminar conclusões:
- **Use CLV como KPI primário** de qualidade/edge (especialmente em janelas curtas).
- **Use ROI como KPI secundário** e somente com janela maior e cobertura de resultados consistente.
- **Aplique quality gate para “closing stale”**: se a última odd pré‑kickoff estiver muito distante do kickoff, não há closing confiável; nesse caso, **CLV deve virar `NULL`** e o evento fica fora de estatísticas de CLV.
- **Defina regimes aceitáveis de execução** (ex.: privilegiar `lag_total < 10s` e monitorar quando cai em 5–10s ou pior).

---
## 0.6 Principais riscos de viés (para você confiar no que está medindo)
- **Viés de observação por falhas do collector**: quando o collector fica “active” mas não coleta odds, você perde janelas de mercado e cria amostra não‑aleatória.
- **Viés por filtro de betslip confiável**: necessário, mas altera a população (você está analisando a parte do universo onde BS≈WS dentro de um range).
- **Cobertura parcial de placares**: ROI e métricas de P&L ficam sub‑amostradas (e podem parecer “pior/melhor” por acaso).

---
## Apêndice — Relatório estatístico base (sem alterações)
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-md", required=True)
    p.add_argument("--output-md", required=True)
    p.add_argument("--report-name", default="Análise Estatística Robusta — Contexto Operação (b808)")
    args = p.parse_args()

    in_path = Path(args.input_md)
    out_path = Path(args.output_md)
    md = in_path.read_text(encoding="utf-8")

    preface = build_preface(md, report_name=args.report_name)
    out_path.write_text(preface + "\n\n" + md, encoding="utf-8")
    print(f"OK: escrito {out_path} (len={out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

