# Relatorio Executivo Estatistico H3B v6

Data de emissao: 2026-02-14  
Escopo: inferencia estatistica de retorno (CLV e ROI), com foco em decisoes de alocacao.

---

## 1) Base estatistica e metodo

### 1.1 Fontes utilizadas

1. Base inferencial de retorno com jogos liquidados:
- arquivo: `04_h3b_comprehensive.log`
- referencia temporal da extração: 2026-02-12 20:33 UTC
- contem CLV e ROI com erro padrao por combinacao.

2. Base operacional e risco Lay (mais recente):
- saida de `hypothesis_performance_robust.sh` (2026-02-14 10:27 UTC)
- janela: 14 dias
- versao: `v4.0-api`.

### 1.2 Estatistica reportada

- media
- erro padrao (SE)
- intervalo de confianca de 95%: `IC95 = media +- 1.96 * SE`
- estatistica sinal/ruido: `t ~= media / SE`

Regra de leitura:
- robusto positivo: IC95 inteiro acima de zero
- robusto negativo: IC95 inteiro abaixo de zero
- inconclusivo: IC95 cruza zero

---

## 2) Resumo executivo de decisao

1. No agregado atual, **ROI ainda inconclusivo** nas principais combinacoes (IC95 cruza zero).  
2. Existe sinal estatistico em CLV para algumas faixas:
   - negativo em `BS < WS (-10% a -2%)`
   - positivo em `BS > WS (+2% a +10%)`, porem com amostra pequena de CLV (`N=9`).  
3. Em faixa de linha, `AH 0-1` apresenta CLV robustamente negativo.  
4. Em latencia, faixa `10-20s` apresenta CLV robustamente negativo.  
5. Para Lay, o risco de cauda e alto e a base de retorno liquidado por combinacao ainda e insuficiente para inferencia robusta.

---

## 3) Retorno por combinacao BS vs WS (principal para selecao)

### 3.1 CLV e ROI com IC95 por bucket

| Bucket BS vs WS | N (CLV) | CLV medio (%) | IC95 CLV (%) | t CLV | N (ROI) | ROI medio (%) | IC95 ROI (%) | t ROI | Leitura |
|---|---:|---:|---|---:|---:|---:|---|---:|---|
| BS < WS (-10 a -2) | 21 | -4.639 | [-6.240, -3.038] | -5.68 | 159 | +3.852 | [-8.527, +16.231] | +0.61 | CLV negativo robusto; ROI inconclusivo |
| BS ~ WS (-2 a +2) | 57 | -0.300 | [-1.088, +0.488] | -0.75 | 488 | -4.185 | [-11.982, +3.612] | -1.05 | Sem evidencia robusta de retorno |
| BS > WS (+2 a +10) | 9 | +3.661 | [+0.925, +6.397] | +2.62 | 140 | +3.464 | [-12.737, +19.665] | +0.42 | CLV positivo, mas N baixo e ROI inconclusivo |

Leitura de decisao:
- bucket `BS > WS (+2,+10)` e o unico com CLV positivo robusto nesta leitura;
- nao ha confirmacao estatistica de ROI em nenhum bucket.

---

## 4) Retorno por faixa de linha AH

| Faixa AH | N (CLV) | CLV medio (%) | IC95 CLV (%) | t CLV | N (ROI) | ROI medio (%) | IC95 ROI (%) | t ROI | Leitura |
|---|---:|---:|---|---:|---:|---:|---|---:|---|
| AH 0-1 | 46 | -2.551 | [-3.588, -1.514] | -4.82 | 337 | -1.349 | [-9.426, +6.728] | -0.33 | CLV negativo robusto; evitar como faixa base |
| AH 1-2 | 8 | +0.882 | [-1.384, +3.148] | +0.76 | 103 | +8.983 | [-8.824, +26.790] | +0.99 | Promissora, mas amostra pequena em CLV |
| AH 2+ | 33 | +0.871 | [-0.519, +2.261] | +1.23 | 347 | -4.079 | [-14.355, +6.197] | -0.78 | Sem robustez de retorno |

---

## 5) Retorno por faixa de latencia

| Faixa de lag | N (CLV) | CLV medio (%) | IC95 CLV (%) | t CLV | N (ROI) | ROI medio (%) | IC95 ROI (%) | t ROI | Leitura |
|---|---:|---:|---|---:|---:|---:|---|---:|---|
| < 10s | 38 | -0.020 | [-0.914, +0.874] | -0.04 | 560 | -1.938 | [-9.433, +5.557] | -0.51 | Neutro/inconclusivo |
| 10-20s | 39 | -2.237 | [-3.595, -0.879] | -3.23 | 87 | +5.202 | [-8.518, +18.922] | +0.74 | Erosao de CLV robusta |
| 20-30s | 6 | +1.268 | [-4.583, +7.119] | +0.42 | 72 | -14.518 | [-34.939, +5.903] | -1.39 | Alta incerteza |
| > 30s | 4 | -0.286 | [-3.765, +3.193] | -0.16 | 68 | +10.782 | [-11.325, +32.889] | +0.96 | Alta incerteza |

Leitura:
- a degradacao mais consistente de CLV aparece na janela `10-20s`.

---

## 6) Comparativo de modelo (retorno)

| Modelo | N CLV adicional | CLV adicional medio (%) | IC95 CLV adicional (%) | t | N ROI | ROI medio (%) | IC95 ROI (%) | t | Leitura |
|---|---:|---:|---|---:|---:|---:|---|---:|---|
| API (2-4s, rotulo legado) | 42 | +0.023 | [-1.098, +1.144] | +0.04 | 682 | -2.446 | [-9.302, +4.410] | -0.70 | Inconclusivo em retorno |
| DOM (15-30s) | 45 | -3.354 | [-5.014, -1.694] | -3.96 | 105 | +6.893 | [-5.171, +18.957] | +1.12 | CLV adicional robustamente negativo |

Leitura:
- em qualidade de execucao (CLV adicional), API domina DOM;
- em ROI, ambos ainda sem confirmacao robusta.

---

## 7) Lay: inferencia de retorno e risco de cauda

### 7.1 Evidencia de risco (janela 14d, v4.0-api)

| Medida de risco Lay | Valor |
|---|---:|
| N Lay (coorte) | 105 |
| Liability p95 | 413.92 |
| Liability p99 | 3561.36 |
| Liability max | 4386.23 |
| ES95 (single liability) | 2131.85 |

### 7.2 Cobertura de retorno Lay por combinacao

Base consolidada disponivel para Lay por bucket mostra cobertura de CLV muito baixa em varios blocos, com buckets sem massa inferencial suficiente para IC robusto.

Conclusao estatistica para Lay:
- risco de cauda esta bem caracterizado;
- retorno (CLV/ROI) por combinacao ainda sem base suficiente para inferencia robusta de escala.

---

## 8) Diagnostico da anomalia "CLV medio 150% em H3B"

No resumo agregado de uma das consultas, apareceu:
- `h3b_temporal_reversal_events`: `clv_mean_pct = +150.5592%` com cobertura de CLV de 3.3%.

Interpretacao tecnica:
- essa media agregada esta sensivel a outliers/extremos e nao deve ser usada isoladamente para decisao de stake;
- quando se usa leitura robusta por combinacao com controle de faixa (ex.: CLV em banda operacional), os valores voltam para ordens plausiveis.

Referencia robusta observada na base inferencial:
- CLV Bruto BS Pre-Match: media `-0.938%`, IC95 `[-1.796%, -0.080%]`
- CLV Adicional BS Pre-Match: media `-1.724%`, IC95 `[-2.794%, -0.654%]`

---

## 9) Matriz final de decisao

| Bloco | Status estatistico | Decisao |
|---|---|---|
| BS < WS (-10,-2) | CLV negativo robusto | Evitar como entrada padrao |
| BS ~ WS (-2,+2) | CLV e ROI inconclusivos | Nao priorizar |
| BS > WS (+2,+10) | CLV positivo robusto, ROI inconclusivo | Priorizar em teste controlado, sem escala agressiva |
| AH 0-1 | CLV negativo robusto | Penalizar/filtrar |
| Lag 10-20s | CLV negativo robusto | Reduzir prioridade operacional |
| Lay (retorno) | base insuficiente por combinacao | Nao escalar por retorno ate aumentar N liquidado |
| Lay (risco) | cauda elevada (p95/p99/ES95) | Operar apenas com limite estrito de exposicao |

---

## 10) Requisitos minimos para escalar stake com rigor

1. CLV positivo com IC95 acima de zero em combinacoes elegiveis e N adequado.  
2. ROI com IC95 acima de zero no mesmo recorte (hoje ainda nao atendido).  
3. Para Lay, alem de CLV/ROI, controle de cauda por limite de responsabilidade por aposta e por janela temporal.

