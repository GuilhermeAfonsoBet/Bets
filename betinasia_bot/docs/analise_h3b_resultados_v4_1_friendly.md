# Análise H3B v4.1 — Relatório Executivo

**Data de atualização:** 12/02/2026  
**Objetivo:** apresentar os resultados de H3B em formato de decisão, com foco em leitura humana e implicações práticas.

---

## Leitura em 60 segundos

- A base analisada é robusta em volume (**3975 auditorias**), mas ainda com dependência intra-jogo (11,4 observações por jogo).
- No **pre-match**, o sinal H3B continua fraco:
  - **CLV adicional WS:** -0,356% (significativo negativo)
  - **CLV adicional BS:** -1,794% (significativo negativo)
- Em **ROI**, não há consistência estatística no betslip:
  - **ROI BS pre-match:** -2,877% (não significativo)
  - **ROI BS in-match:** +0,464% (não significativo)
- O modelo **API** segue claramente melhor que **DOM** em qualidade de execução e erosão de preço.
- Conclusão operacional: **não há edge robusto no agregado atual**; o foco deve ser aumentar N no recorte API e melhorar seleção de entradas.

---

## 1) Escopo e qualidade da amostra

### Volumes principais

| Indicador | Valor |
|---|---:|
| Auditorias H3B UP com match + kickoff passado | 3975 |
| Betslip bruto | 2013 |
| Betslip confiável (diff -10% a +10%) | 892 |
| Descartados no filtro de qualidade | 1121 |
| Jogos únicos (geral) | 349 |
| Média de observações por jogo | 11,4 |
| Jogos únicos com betslip | 155 |

### Cobertura por métrica

| Métrica | N |
|---|---:|
| CLV WS bruto | 1656 |
| CLV WS adicional | 1657 |
| CLV BS bruto | 126 |
| CLV BS adicional | 125 |
| ROI WS | 3270 |
| ROI BS | 785 |
| Recorte pre-match | 1568 |
| Recorte in-match | 1652 |

---

## 2) Resultado central de CLV (o que mais importa para edge)

### WebSocket (referência)

| Métrica | N | Média | Status |
|---|---:|---:|---|
| CLV bruto WS pre-match | 1051 | +0,172% | Significativo positivo |
| CLV adicional WS pre-match | 1051 | -0,356% | Significativo negativo |

Leitura: o CLV bruto é positivo, mas o CLV adicional (ajustado por baseline) é negativo.  
Na prática, o sinal não mostra valor incremental consistente sobre o mercado.

### Betslip (execução real)

| Métrica | N | Média | Status |
|---|---:|---:|---|
| CLV bruto BS pre-match | 80 | -0,839% | Significativo negativo |
| CLV adicional BS pre-match | 80 | -1,794% | Significativo negativo |

Leitura: ao entrar no preço real de execução, o resultado piora.

---

## 3) ROI por momento de entrada

| Métrica | N | Média | Mediana | Status |
|---|---:|---:|---:|---|
| ROI BS pre-match | 379 | -2,877% | 0,0% | Não significativo |
| ROI BS in-match | 403 | +0,464% | 0,0% | Não significativo |
| ROI WS pre-match | 1430 | -4,140% | 0,0% | Significativo negativo |
| ROI WS in-match | 1250 | +2,882% | 0,0% | Não significativo |

Leitura: o ROI permanece instável e sem significância no betslip.

---

## 4) API vs DOM (comparação prática)

| Indicador | API (2-4s) | DOM (15-30s) |
|---|---:|---:|
| Total observações | 1053 | 2843 |
| Com betslip | 763 | 129 |
| Lag médio | 11.522 ms | 15.170 ms |
| CLV bruto BS pre-match | +0,839% (NS) | -2,144% (sig. neg.) |
| CLV adicional BS pre-match | +0,211% (NS) | -3,354% (sig. neg.) |
| Diff BS vs WS | +0,185% (NS) | -2,931% (sig. neg.) |
| BS > WS | 38,0% | 8,5% |
| BS > WS +2% | 18,5% | 7,0% |

Leitura: API preserva melhor preço e mantém chance de edge; DOM destrói edge de forma recorrente.

---

## 5) Onde está o potencial (e onde evitar)

### Potencial
- Buckets com **BS > WS** continuam sendo os mais promissores para CLV.
- No recorte de **API**, há sinais de melhora estrutural versus DOM.

### Cautela
- Em **AH 0-1 (linhas líquidas)**, CLV BS pre-match segue negativo e significativo.
- Em muitos recortes, a mediana de ROI fica em 0%, indicando baixa robustez operacional.

---

## 6) Faixas de linha AH (resumo rápido)

| Faixa AH | CLV BS PM | ROI BS | Leitura |
|---|---:|---:|---|
| 0-1 (líquida) | -2,551% (sig. neg.) | -1,349% (NS) | Faixa mais crítica hoje |
| 1-2 (média) | +0,882% (NS) | +8,983% (NS) | Promissora, sem confirmação |
| 2+ (extrema) | +1,661% (sig. pos.) | -4,041% (NS) | Ruído/volatilidade alta |

---

## 7) Qualidade dos dados (sanidade)

- Betslip odds: min **1,029** | med **1,939** | max **23,775**
- WebSocket odds: min **1,068** | med **1,943** | max **24,725**
- Diff percentual pós-filtro: min **-10,0%** | med **-0,2%** | max **+10,0%**
- Outliers extremos (>30% ou <-30%): **0**
- Jogos com placar disponível: **3270/3975**

---

## 8) Conclusão de decisão

1. O sinal H3B UP **não apresenta edge robusto no agregado atual**.
2. O canal **API** é claramente superior ao DOM e deve ser o padrão operacional.
3. O próximo ciclo deve focar em:
   - aumentar N de CLV no recorte API pre-match;
   - reduzir ainda mais erosão de execução;
   - usar filtros de seleção (ex.: BS > WS) para concentrar entradas.

---

## 9) Próximos passos recomendados (curto prazo)

1. Rodar coleta adicional para elevar amostra de CLV BS pre-match (API).
2. Recalcular intervalos com ajuste por correlação intra-jogo (cluster por match_id).
3. Publicar v4.2 com:
   - seção executiva (decisão)
   - anexo técnico completo (estatística detalhada).

