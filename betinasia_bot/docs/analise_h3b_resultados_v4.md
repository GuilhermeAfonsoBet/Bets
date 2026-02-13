# Analise H3B — Resultados v4: API vs DOM

**Data:** 11 de Fevereiro de 2026
**Versao:** 4.0
**Status:** Analise com separacao por modelo de execucao

---

## 1. Resumo Executivo

A velocidade de execucao e o fator determinante da estrategia H3B UP. Com o modelo API (2-4s de lag), o CLV e **positivo** (+0.591% adicional). Com o modelo DOM (15-30s de lag), o CLV e **negativo** (-3.424% adicional). A diferenca e inteiramente explicada pela erosao temporal da odd.

| Metrica | API (2-4s) | DOM (15-30s) | Diferenca |
|---------|-----------|-------------|-----------|
| CLV Adicional BS | +0.591% | -3.424% | +4.015% |
| CLV Bruto BS | +0.235% | -2.244% | +2.479% |
| Win Rate CLV Adicional | 64.3% | 27.3% | +37pp |
| Diff BS vs WS | -0.128% | -2.965% | +2.837% |
| BS > WS (% dos casos) | 36.3% | 7.8% | +28.5pp |

**Conclusao:** H3B UP tem potencial de edge com execucao rapida (2-4s). Precisamos de mais ~17 observacoes com closing line para confirmar significancia estatistica (temos 14, estimamos ~31 necessarios).

---

## 2. Dados e Amostras

| Metrica | Total | API (2-4s) | DOM (15-30s) |
|---------|-------|-----------|-------------|
| Observacoes | 3.672 | 770 | 2.825 |
| Com betslip | 1.746 | 587 | 128 |
| Filtrado (-10%/+10%) | 715 | 587 | 128 |
| Com CLV (pre-match) | 104 | 14 | 44 |
| Com ROI (resultado) | 559 | 455 | 104 |
| Jogos unicos | 323 | ~100 | ~200 |

**Nota sobre qualidade:** O modelo API tem 96% de taxa de extracao (587/770) vs ~5% no DOM. O filtro de qualidade (-10%/+10%) quase nao descarta dados API (diff mediana -0.053%) enquanto descarta 86% dos dados DOM.

---

## 3. CLV por Modelo de Execucao

### 3.1. API (2-4s) — Edge Positivo (N pequeno)

| Metrica | Valor |
|---------|-------|
| CLV Bruto BS Pre-Match | +0.235% |
| CLV Adicional BS Pre-Match | **+0.591%** |
| IC 90% (Adicional) | [-0.279%, +1.461%] |
| Significancia | Nao significativo (N=14) |
| N estimado p/ significancia | **~31** |
| Win Rate CLV Adicional | 64.3% (9/5/0) |

O CLV adicional de +0.591% indica que a odd do betslip, quando capturada em 2-4s, ainda esta acima do baseline do mercado. A velocidade preserva o edge que o lag destroi.

### 3.2. DOM (15-30s) — Edge Negativo (confirmado)

| Metrica | Valor |
|---------|-------|
| CLV Bruto BS Pre-Match | -2.244% |
| CLV Adicional BS Pre-Match | **-3.424%** |
| IC 90% (Adicional) | [-4.846%, -2.003%] |
| Significancia | SIGNIFICATIVO NEGATIVO |
| Win Rate CLV Adicional | 27.3% (12/32/0) |

Com 15-30s de lag, a odd do betslip ja deteriorou alem do baseline. Confirmado com N=44.

---

## 4. Erosao Temporal (Diff BS vs WS)

| Modelo | Diff Media | Diff Mediana | BS > WS | BS > WS +2% |
|--------|-----------|-------------|---------|------------|
| API (2-4s) | -0.128% | -0.053% | 36.3% | 14.0% |
| DOM (15-30s) | -2.965% | -2.773% | 7.8% | 7.0% |

**Padrao claro:** Com 2-4s de lag, a erosao e quase zero (-0.13%). Com 15-30s, e quase 3%. Cada segundo adicional de lag custa ~0.15% de erosao.

### 4.1. Por Faixa de Lag

| Lag | Diff BS vs WS | CLV BS (PM) | N |
|-----|--------------|------------|---|
| < 10s | -0.160% (NS) | +0.103% (NS) | 582/17 |
| 10-20s | -2.437% (neg) | -2.355% (neg) | 102/38 |
| 20-30s | -3.940% (neg) | -4.679% (neg) | 23/2 |
| > 30s | -2.801% (neg) | N=1 | 8/1 |

A transicao de lag < 10s (quase zero erosao) para lag 10-20s (2.4% erosao) e abrupta. O limiar critico esta em torno de 10 segundos.

---

## 5. ROI Real por Modelo

| Metrica | API (2-4s) | DOM (15-30s) |
|---------|-----------|-------------|
| N | 455 | 104 |
| ROI Media | -1.60% | +7.92% |
| ROI Mediana | 0.000% | 0.000% |
| Win Rate | 51.1% | 61.9% |
| Significancia | NS | NS |

**Nota:** O ROI do DOM (+7.92%) e inflado por outliers em linhas extremas (mesmo problema da v3). O ROI do API (-1.60%) e mais confiavel mas com N grande que inclui todas as linhas.

### 5.1. ROI Pre-Match vs In-Match

| Recorte | Media | Mediana | Win Rate | N |
|---------|-------|---------|----------|---|
| Pre-Match BS | +2.43% | 0.00% | 53.0% | 245 |
| In-Match BS | -1.54% | 0.00% | 51.5% | 311 |
| Pre-Match WS | -3.55% | 0.00% | 48.6% | 1.249 |
| In-Match WS | +2.35% | 0.00% | 48.7% | 1.074 |

Nenhum ROI e significativo. Medianas sao todas zero.

---

## 6. Combinacoes de Valor

### 6.1. BS > WS (+2% a +10%)

| Metrica | Valor |
|---------|-------|
| N total | 91 |
| CLV BS Pre-Match | **+3.760%** (sig positivo, N=6) |
| ROI | +5.10% (NS, N=75) |
| Win Rate ROI | 53.0% |
| Win Rate CLV | 83.3% |

Quando betslip > websocket por 2-10%, o CLV e significativamente positivo. Porem N pequeno para CLV.

### 6.2. AH 1-2 (Media Liquidez)

| Metrica | Valor |
|---------|-------|
| N total | 91 |
| ROI | **+19.44%** (sig positivo, N=57) |
| Win Rate | 63.4% |
| Diff BS vs WS | +0.08% (NS) |

Linhas de media liquidez (AH 1-2) mostram ROI significativamente positivo. Possivel ineficiencia de precificacao nesta faixa.

### 6.3. AH 2+ com CLV (Extremas)

| Metrica | Valor |
|---------|-------|
| CLV BS Pre-Match | **+1.562%** (sig positivo, N=11) |
| ROI | -3.56% (NS, N=226) |
| Win Rate CLV | 63.6% |
| Win Rate ROI | 50.0% |

CLV positivo mas ROI negativo — inconsistencia que sugere que CLV nao e bom preditor de ROI em linhas extremas.

---

## 7. Combinacoes com Valor Negativo

### 7.1. AH 0-1 (Linhas Liquidas)

| Metrica | Valor |
|---------|-------|
| CLV BS Pre-Match | **-2.551%** (sig negativo, N=46) |
| ROI | -0.76% (NS, N=276) |
| Win Rate CLV | 20.0% |
| Win Rate ROI | 52.2% |
| Diff BS vs WS | -1.21% (sig negativo) |

Linhas mais liquidas (AH 0-1) tem CLV consistentemente negativo. A erosao e maior nesta faixa (-1.21% diff). Mercados mais eficientes corrigem mais rapido.

### 7.2. Lag > 10s (DOM)

Qualquer lag acima de 10s resulta em CLV negativo:
- 10-20s: CLV -2.36%
- 20-30s: CLV -4.68%
- Padrao monotonicamente decrescente

---

## 8. Tendencias Demonstradas

### 8.1. Velocidade e o Fator Determinante

A correlacao entre lag e CLV e clara e monotonica. O grafico implicito:

| Lag | CLV BS | Tendencia |
|-----|--------|-----------|
| 2-4s (API) | +0.24% | Positivo |
| < 10s | +0.10% | Neutro |
| 10-20s | -2.36% | Negativo |
| 20-30s | -4.68% | Muito negativo |

Cada segundo de lag apos os primeiros 5s custa ~0.15-0.25% de CLV. O ponto de breakeven esta em torno de 5-8 segundos.

### 8.2. Qualidade dos Dados API vs DOM

| Metrica | API | DOM |
|---------|-----|-----|
| Taxa extracao | 96% | ~5% |
| Diff mediana | -0.05% | -2.77% |
| Diffs extremas | 0% | 86% |
| Confiabilidade | Alta | Baixa |

Os dados API sao dramaticamente superiores em qualidade.

### 8.3. BS > WS como Sinal

No modelo API, 36.3% dos casos tem BS > WS (betslip melhor que websocket). Isto sugere que o websocket nem sempre reflete a melhor odd disponivel — o betslip agrega mais fontes e pode oferecer precos melhores.

### 8.4. CLV Websocket Declinante

O CLV bruto WS pre-match caiu de +1.116% (estudo v6 original, N=273) para +0.146% (N=1.005). Com mais dados, o edge bruto converge para proximo de zero. O CLV adicional e negativo (-0.355%), confirmando que H3B UP nao tem valor incremental no websocket puro.

---

## 9. Proximos Passos Criticos

### 9.1. Acumular Dados API com Closing Line (PRIORIDADE 1)

Temos N=14 para CLV adicional BS pre-match na API. Precisamos de ~31. Estimativa: **2-3 dias** ao ritmo atual. Este e o dado que valida ou invalida a estrategia.

### 9.2. Dados Lay (Em Coleta)

O sistema v5.0 ja captura odds de Lay simultaneamente ao Back. Quando tivermos dados suficientes, analisaremos:
- Spread Back-Lay por momento
- Lay como estrategia alternativa em combinacoes com CLV negativo

### 9.3. Monitoramento Temporal (Em Coleta)

O sistema v5.0 faz refresh a cada 3-20s apos deteccao. Dados de evolucao temporal permitem:
- Identificar momento otimo de execucao
- Medir velocidade de convergencia BS vs WS
- Estimar janela de oportunidade

### 9.4. Correcao de Erros Clusterizados

16.6 observacoes por jogo (eram 11.4, piorou). N efetivo ~323 jogos, nao 3.672 observacoes. ICs estao subestimados. Correcao pendente.

---

## 10. Conclusao

A analise v4 transforma a perspectiva da estrategia H3B UP:

**Antes (v3, DOM):** "Sem edge executavel. CLV -3.4%, ROI mediano zero."

**Agora (v4, API):** "Edge potencialmente positivo com execucao rapida. CLV +0.59%, pendente validacao com mais dados."

A velocidade de execucao (2-4s vs 15-30s) e a variavel que separa lucro de prejuizo. O investimento em infraestrutura rapida (API, proxy, WebSocket persistente) esta se mostrando justificado.

**Nivel de confianca na estrategia: 50-60%** (era 20% na v3). Pendente confirmacao com N~31 para CLV adicional.
