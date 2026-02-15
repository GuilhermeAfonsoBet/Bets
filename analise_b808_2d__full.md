# Análise Estatística Robusta — Contexto Operação (b808)
**Data da execução:** 15/02/2026 01:04 UTC  
**Escopo:** H3B (auditoria), com comparação por `audit_version` e inferência robusta por jogo (cluster bootstrap).  
**Nota:** a robustez aqui significa que intervalos de confiança consideram correlação intra-jogo (múltiplas auditorias por partida).
---
## 0) Sumário executivo (leitura rápida)
- **Recorte**: direction=`up`, lookback_days=`2`, versions=`v4.0-api,v1.0,v1.0-recovered`.
- **Amostra**: 1882 auditorias (jogos únicos=160, média=11.8 obs/jogo); betslip confiável=914.
- **Coortes (status=OK, betslip confiável)**: Back (diff>=2.0%): **323**; Lay (diff<=-2.0%): **119**.
- **Coberturas em `hypothesis_details` (OK)**: temporal(back)=696/914; lay_temporal=654/914; finance=516/914.
- **Cobertura de placar (ROI)**: jogos com placar=130/160 (status finished=130).
- **DOM**: sem dados no recorte atual (N=0), então não há comparação API vs DOM aqui.
- **CLV pre-match (Betslip, API)**: média robusta por jogo +1.977% (IC90 [+1.303%, +2.616%]), com N=429 eventos (jogos=86).
- **Padrão por bucket (CLV PM)**: `BS < WS` -3.854% (sig. negativo), `BS ~ WS` -0.177% (NS), `BS > WS` +6.975% (sig. positivo).
- **Leitura recomendada**: use este relatório para validar **qualidade de execução/CLV**. Para concluir sobre **ROI**, rode após atualizar resultados e/ou use uma janela com jogos já liquidados.

---
## 1) Contexto do corte (b808)
| Indicador | Valor |
|---|---:|
| Auditorias H3B `UP` (match + kickoff passado) | 1882 |
| Betslip bruto | 1519 |
| Betslip confiável (diff -10% a +10%) | 914 |
| Descartados no filtro de qualidade | 605 |
| Jogos únicos (geral) | 160 |
| Média de observações por jogo | 11.8 |
| Jogos únicos com betslip confiável | 153 |

---
## 2) Base comparativa: API vs DOM
| Métrica | API (v4.0-api) | DOM (v1.0) |
|---|---:|---:|
| Total de observações | 1882 | 0 |
| Com betslip confiável | 914 | 0 |
| Com CLV pre-match (betslip) | 429 | 0 |
| Com ROI (betslip) | 782 | 0 |
| Tempo total observado (detecção→betslip, wall/total) | 15367 ms | — ms |
| Tempo instrumentado (detecção→clique→betslip) | 11658 ms | — ms |

---
### 2.0b Decomposição do tempo (detecção→clique→betslip vs. overhead)
Interpretação: `lag_e2e` é o **tempo fim-a-fim** (detecção→clique + clique→betslip). `overhead` = `audit_total` − `lag_e2e` (proxy de fila/retries/esperas fora das 2 etapas instrumentadas).

| Modelo | Métrica | mean (ms) | p50 (ms) | p95 (ms) | N |
|---|---|---:|---:|---:|---:|
| API (2-4s) | lag_det→click | 9129 | 878 | 21150 | 1882 |
| API (2-4s) | lag_click→betslip | 2675 | 2184 | 4561 | 1878 |
| API (2-4s) | lag_e2e (soma) | 11658 | 3670 | 21441 | 1878 |
| API (2-4s) | audit_total (duração) | 15363 | 4786 | 34402 | 1882 |
| API (2-4s) | overhead (total - e2e) | 3572 | 242 | 22766 | 1878 |
| DOM (15-30s) | lag_det→click | — | — | — | 0 |
| DOM (15-30s) | lag_click→betslip | — | — | — | 0 |
| DOM (15-30s) | lag_e2e (soma) | — | — | — | 0 |
| DOM (15-30s) | audit_total (duração) | — | — | — | 0 |
| DOM (15-30s) | overhead (total - e2e) | — | — | — | 0 |

**Diagnóstico de cauda (percentual acima do limiar)**

| Modelo | % det→click > 5s | % det→click > 20s | % total > 10s | % total > 40s | % overhead < 0 |
|---|---:|---:|---:|---:|---:|
| API (2-4s) | 8.0% | 5.2% | 16.1% | 4.3% | 0.0% |
| DOM (15-30s) | —% | —% | —% | —% | —% |

---
### 2.1 Cobertura temporal (pre-match vs in-match)
| Métrica | Pre-match | In-match | Observação |
|---|---:|---:|---|
| Observações totais com classificação temporal | 713 | 1169 | Contagem bruta do corte |
| ROI Betslip | 436 | 346 | Amostra com resultado do jogo |
| ROI WebSocket | 607 | 991 | Referência de mercado |
| CLV (apenas pre-match) | 429 | — | CLV vs closing pré-jogo não é interpretável in-match |

---
### 2.2 Performance por regime (pre-match vs in-match)
| Regime | N auditorias | N betslip conf. | N OK | Back edge | Lay edge | Diff médio (OK) |
|---|---:|---:|---:|---:|---:|---:|
| PRE_MATCH | 713 | 496 | 496 | 190 | 36 | +2.165% |
| IN_MATCH | 1169 | 418 | 418 | 133 | 83 | +0.993% |

---
### 2.3 Regimes operacionais por tempo total (bucket)
Objetivo: separar a amostra em **regimes de execução** (tempo total) e medir performance em cada regime. Use isso para detectar **fila/saturação**: regimes lentos tendem a ter edge menor e pior ROI.

| Bucket (tempo total) | N OK | Jogos | lag_total mean (ms) | lag_total p95 (ms) | overhead p95 (ms) | Back edge | Lay edge | CLV PM (mean; IC90) | ROI (mean; IC90) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| < 5s | 599 | 135 | 3675 | 4891 | 2514 | 193 | 70 | +2.25% [+1.62%, +2.86%] | +7.76% [+0.14%, +15.70%] |
| 5-10s | 147 | 89 | 5900 | 7914 | 3303 | 61 | 29 | +1.48% [+0.29%, +2.66%] | -11.65% [-27.09%, +3.50%] |
| 10-20s | 11 | 10 | 12942 | 18007 | 3252 | 4 | 0 | +1.47% [-0.09%, +3.05%] | +20.84% [-23.22%, +64.46%] |
| 20-40s | 116 | 65 | 27471 | 34092 | 29429 | 45 | 16 | +1.88% [+0.59%, +3.20%] | +18.76% [+2.90%, +34.94%] |
| > 40s | 41 | 31 | 184629 | 526665 | 30079 | 20 | 4 | +2.82% [+1.06%, +4.52%] | -21.38% [-47.56%, +5.42%] |
| Desconhecido | 0 | 0 | — | — | — | 0 | 0 | — | — |

---
## 3) CLV pre-match (núcleo)
### 3.1 CLV com odd do Betslip (execução real)
| Métrica | API | DOM |
|---|---:|---:|
| CLV Bruto BS Pre-Match | +2.383% (sig. positivo, N=429, jogos=86) | — (N/A, N=0, jogos=0) |
| CLV Adicional BS Pre-Match | +1.452% (NS, N=429, jogos=86) | — (N/A, N=0, jogos=0) |
| Taxa de CLV > 0 (bruto) | 60.8% | —% |
| Taxa de CLV > 0 (adicional) | 60.8% | —% |

Notas de robustez (IC 90% por jogo):  
- API CLV bruto (cluster): média +1.977%; IC90 [+1.303%, +2.616%]  
- DOM CLV bruto (cluster): média —; IC90 —  

---
## 4) ROI por modelo
| Métrica | API | DOM |
|---|---:|---:|
| ROI Betslip | +2.702% (NS, N=782) | — (N/A, N=0) |
| ROI WebSocket | -0.603% (NS, N=1592) | — (N/A, N=0) |
| Win rate ROI Betslip | 51.7% | —% |
| Win rate ROI WS | 50.0% | —% |

Notas de robustez (IC 90% por jogo):  
- API ROI betslip (cluster): média +3.692%; IC90 [-3.160%, +10.680%]  
- API ROI WS (cluster): média -1.324%; IC90 [-4.427%, +1.739%]  

---
## 5) Diferença de preço BS vs WS
| Métrica | API | DOM |
|---|---:|---:|
| Diff BS vs WS (média) | +1.629% (sig. positivo, N=914) | — (N/A, N=0) |
| BS > WS | 46.5% (425/914) | —% (0/0) |
| BS > WS +2% | 35.3% (323/914) | —% (0/0) |

---
## 6) Combinações de valor
### 6.1 Buckets por diferença BS vs WS
| Bucket | N bucket | CLV BS PM (média) | IC90 (cluster) | N CLV PM | Jogos CLV PM | ROI BS (todos) | IC90 (cluster) |
|---|---:|---:|---|---:|---:|---:|---|
| BS < WS (-10% a -2%) | 119 | -3.854% | [-5.565%, -3.073%] | 30 | 24 | -0.710% | [-17.487%, +13.054%] |
| BS ~ WS (-2% a +2%) | 472 | -0.177% | [-0.632%, +0.363%] | 230 | 79 | +2.775% | [-8.365%, +8.897%] |
| BS > WS (+2% a +10%) | 323 | +6.975% | [+6.532%, +7.595%] | 169 | 62 | +3.883% | [-1.948%, +22.408%] |

---
### 6.2 Combinação por faixa de linha AH
| Faixa AH | CLV BS PM (média) | IC90 (cluster) | ROI BS (todos) | IC90 (cluster) | Diff BS vs WS (média) |
|---|---:|---|---:|---|---:|
| AH 0-1 (líquida) | +2.658% | [+1.631%, +3.346%] | +5.444% | [+3.486%, +24.642%] | +1.783% |
| AH 1-2 (média) | +2.656% | [+1.240%, +4.153%] | +4.987% | [-14.207%, +17.747%] | +2.241% |
| AH 2+ (extrema) | +1.852% | [+0.186%, +2.519%] | -0.001% | [-9.932%, +12.109%] | +1.328% |

---
### 6.3 Combinação por faixa de lag
| Faixa de lag | CLV BS PM (média) | IC90 (cluster) | N CLV PM | Jogos CLV PM | ROI BS (todos) | IC90 (cluster) | Diff BS vs WS (média) |
|---|---:|---|---:|---:|---:|---|---:|
| < 10s | +2.465% | [+1.512%, +2.778%] | 331 | 82 | +3.003% | [-3.762%, +10.043%] | +1.541% |
| 10-20s | +1.493% | [-0.093%, +3.047%] | 4 | 4 | +18.820% | [-23.222%, +64.456%] | +1.011% |
| 20-30s | +1.839% | [+0.480%, +2.909%] | 63 | 45 | +5.289% | [-4.824%, +30.966%] | +1.934% |
| > 30s | +2.728% | [+1.287%, +4.466%] | 31 | 23 | -9.333% | [-35.300%, +10.984%] | +2.318% |

---
## 7) Estimativa financeira (proxy) e risco
Este bloco usa `hypothesis_details.finance` quando existe; se não existir, usa stake fallback = `stake_pct_of_limit × limit`.

### 7.1 Back (BS >> WS)
| Métrica | Valor |
|---|---:|
| Corte (diff_pct) | >= 2.0% |
| N eventos | 323 |
| Stake total (estimado) | 110531.39 |
| Stake médio | 342.20 |
| Profit_if_win total (estimado) | 116232.43 |
| Profit_if_win médio | 359.85 |
| N com ROI realizado | 279 |
| P&L realizado total (estimado) | -14150.04 |
| ROI realizado (ponderado por stake) | -13.49% |
| ROI realizado (robusto por jogo, mean; IC90) | +10.15% [-1.95%, +22.41%] |
| ROI ponderado por stake (robusto por jogo, mean; IC90) | +3.39% [-10.07%, +16.95%] |

Observação: a Seção 4 reporta ROI médio no **recorte completo** (betslip confiável). Aqui (7.1) é apenas a coorte **Back edge** (diff>=corte) e o ROI também é mostrado ponderado por stake; por isso sinais podem divergir.

### 7.2 Lay (BS << WS) — risco de cauda
| Métrica | Valor |
|---|---:|
| Corte (diff_pct) | <= -2.0% |
| N eventos | 119 |
| Stake total (estimado) | 12660.00 |
| Liability total (estimada) | 12013.44 |
| Liability média | 100.95 |
| Liability p95 | 442.96 |
| Liability p99 | 969.76 |
| ES95 (liability) | 775.92 |
| Liability max | 1199.86 |
| Proxy de banca (>= p99 liability) | 969.76 |
| N com ROI realizado | 73 |
| P&L realizado total (estimado) | -820.85 |
| ROI realizado (ponderado por liability) | -7.60% |
| ROI realizado (ponderado por stake) | -7.19% |
| ROI/liability (robusto por jogo, mean; IC90) | +21.74% [+1.46%, +42.65%] |
| ROI/liability ponderado (robusto por jogo, mean; IC90) | +16.68% [-4.59%, +38.79%] |

---
## 8) Curva temporal (pico, reversão e melhor timing)
Esta seção usa `hypothesis_details.temporal` (Back) e `hypothesis_details.lay_temporal` (Lay), coletados em pontos discretos (t≈0,3,6,10,15,20s).

O objetivo é responder: **tempo até o pico/vale**, **% que segue melhorando até t_max**, **% com reversão** e **impacto esperado por timing**. **CLV é reportado somente pre-match** (closing pré-jogo).

### 8.1 Back (pico em diff_pct)
Definições: **pico** = `max(diff_pct)`. **reversão** = após o pico, `diff_pct` cair pelo menos 0.50 p.p. (para Lay, subir 0.50 p.p. após o vale). `t_reversão` é o 1º tempo que cruza esse limiar (logo, não é o pico).

| Regime | N | t_pico médio (s) | t_pico p50 (s) | % melhora até fim | % com reversão | t_reversão médio (s) | Δt (rev - pico) médio (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRE_MATCH | 496 | 4.5 | 5.2 | 73.6% | 20.0% | 12.3 | 7.1 |
| IN_MATCH | 418 | 4.3 | 0.0 | 66.0% | 26.1% | 13.1 | 7.9 |

**Partição 100% (Back)**: categorias exclusivas (somam 100% por regime).

| Regime | % pico no fim, sem reversão | % pico no fim, com reversão (recuperou) | % pico antes, com reversão | % pico antes, sem reversão relevante |
|---|---:|---:|---:|---:|
| PRE_MATCH | 77.4% | 7.7% | 12.3% | 2.6% |
| IN_MATCH | 71.1% | 4.1% | 22.0% | 2.9% |

**Curva média (Back)**: média de `diff_pct` e `odd` por tempo. `CLV` é reportado **somente pre-match** (closing pré-jogo). `ROI` (quando aparece) é o **ROI realizado** se a aposta fosse feita naquele ponto.

| Tempo | N pts | diff_pct médio | odd média | CLV médio (pre-match) | ROI médio (se houver) |
|---|---:|---:|---:|---:|---:|
| t+0s | 914 | +1.63% | 1.991 | +2.38% | 2.70 |
| t+6s | 683 | +2.21% | 1.999 | +2.72% | 2.20 |
| t+10s | 1099 | +2.95% | 2.016 | +2.76% | 3.80 |
| t+15s | 687 | +2.90% | 2.017 | +2.80% | -0.14 |
| t+20s | 949 | +3.52% | 2.018 | +2.84% | 2.96 |

### 8.1b Back — impacto por timing (t0 vs pico vs último) e reversão
Leitura: se a estratégia é **entrar no pico**, compare CLV/ROI no pico vs t0 e veja diferença entre **casos com reversão** vs **sem reversão**. Aqui reportamos **média robusta por jogo + IC90**. Observação: **CLV só é calculado pre-match**.

**CLV (somente pre-match) — média robusta por jogo (IC90)**

| Subcoorte | N total | N CLV (PM) | CLV t0 (mean; IC90) | CLV pico (mean; IC90) | CLV último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 706 | 342 | +1.69% [+1.06%, +2.32%] | +2.00% [+1.38%, +2.61%] | +1.99% [+1.37%, +2.60%] |
| COM_REVERSAO | 208 | 87 | +3.82% [+2.73%, +4.87%] | +5.48% [+4.16%, +6.77%] | +4.48% [+3.18%, +5.77%] |

**ROI (stake=1) — média robusta por jogo (IC90)**

| Subcoorte | N total | N ROI | ROI t0 (mean; IC90) | ROI pico (mean; IC90) | ROI último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 706 | 605 | +3.79% [-4.00%, +11.50%] | +3.49% [-4.38%, +11.34%] | +3.48% [-4.39%, +11.33%] |
| COM_REVERSAO | 208 | 177 | +2.82% [-11.36%, +17.04%] | +6.83% [-8.15%, +21.67%] | +2.84% [-11.34%, +17.07%] |

Nota interpretativa: a subcoorte **COM_REVERSAO** pode ter CLV maior no **pico** porque, por definição, ela inclui casos em que o edge atingiu um extremo mais alto antes de reverter. Para entender a hipótese 'sem reversão deveria ser maior', compare também a categoria '**pico no fim, sem reversão**' na partição 100% (tabela 8.1).

**Decomposição (pre-match): entry_odd vs closing_odd (IC90)**

| Subcoorte | N CLV (PM) | entry_odd t0 (mean; IC90) | entry_odd pico (mean; IC90) | closing_odd (mean; IC90) |
|---|---:|---:|---:|---:|
| SEM_REVERSAO | 342 | 1.984 [+1.968, +2.000] | 1.994 [+1.978, +2.010] | 1.969 [+1.959, +1.979] |
| COM_REVERSAO | 87 | 2.029 [+2.005, +2.052] | 2.064 [+2.036, +2.092] | 1.965 [+1.949, +1.980] |

### 8.2 Lay (vale em diff_pct)
| Regime | N | t_vale médio (s) | t_vale p50 (s) | % melhora até fim | % com reversão | t_reversão médio (s) | Δt (rev - vale) médio (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRE_MATCH | 437 | 4.6 | 5.2 | 65.7% | 26.1% | 13.1 | 7.1 |
| IN_MATCH | 303 | 5.4 | 0.0 | 51.5% | 37.6% | 13.5 | 8.4 |

**Partição 100% (Lay)**: categorias exclusivas (somam 100% por regime).

| Regime | % vale no fim, sem reversão | % vale no fim, com reversão (recuperou) | % vale antes, com reversão | % vale antes, sem reversão relevante |
|---|---:|---:|---:|---:|
| PRE_MATCH | 69.3% | 10.5% | 15.6% | 4.6% |
| IN_MATCH | 58.1% | 4.6% | 33.0% | 4.3% |

**Curva média (Lay)**: usa `odd=lay_odd`. `CLV` é reportado **somente pre-match** (closing pré-jogo). O ROI mostrado aqui é **ROI por liability** (não por stake), porque Lay é governado por risco.

| Tempo | N pts | diff_pct médio | lay_odd média | CLV médio (pre-match) | ROI/liability médio (se houver) |
|---|---:|---:|---:|---:|---:|
| t+0s | 740 | +0.91% | 1.967 | -0.14% | 7.16 |
| t+6s | 645 | +1.31% | 1.974 | +0.02% | 8.25 |
| t+10s | 1025 | +0.30% | 1.960 | +0.33% | 10.72 |
| t+15s | 652 | +1.61% | 1.980 | -0.27% | 13.96 |
| t+20s | 902 | +2.78% | 2.003 | +0.04% | 5.01 |

### 8.2b Lay — impacto por timing (t0 vs vale vs último) e reversão
Leitura: se a estratégia é **entrar no vale (odd mais baixa)**, compare CLV/ROI no vale vs t0 e veja diferença entre **casos com reversão** vs **sem reversão**. Aqui reportamos **média robusta por jogo + IC90**. Observação: **CLV só é calculado pre-match**.

**CLV (somente pre-match) — média robusta por jogo (IC90)**

| Subcoorte | N total | N CLV (PM) | CLV t0 (mean; IC90) | CLV vale (mean; IC90) | CLV último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 512 | 274 | -0.79% [-1.61%, +0.06%] | -0.34% [-1.14%, +0.48%] | -0.37% [-1.17%, +0.45%] |
| COM_REVERSAO | 228 | 97 | +0.86% [-0.44%, +2.12%] | +2.51% [+1.31%, +3.71%] | +1.21% [-0.01%, +2.39%] |

**ROI/liability — média robusta por jogo (IC90)**

| Subcoorte | N total | N ROI | ROI/liab t0 (mean; IC90) | ROI/liab vale (mean; IC90) | ROI/liab último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 512 | 444 | +2.08% [-9.84%, +15.04%] | +2.53% [-9.53%, +15.51%] | +2.46% [-9.57%, +15.37%] |
| COM_REVERSAO | 228 | 191 | +26.84% [+4.95%, +52.70%] | +41.19% [+8.52%, +83.37%] | +19.06% [+1.97%, +36.71%] |

---
## 9) Combinações de valor (regime × linha AH × lag)
Tabela rankeada por volume (N). Métrica: diff_pct (OK, betslip confiável).

### 9.1 Back combos (diff_pct >= corte)
| Regime | Linha | Lag | N | Jogos | Mean diff % | IC90 cluster |
|---|---|---|---:|---:|---:|---|
| IN_MATCH | AH 2+ (extrema) | < 10s | 69 | 47 | +6.10% | [+5.58%, +6.70%] |
| PRE_MATCH | AH 0-1 (líquida) | < 10s | 67 | 28 | +6.96% | [+6.85%, +7.62%] |
| PRE_MATCH | AH 2+ (extrema) | < 10s | 48 | 24 | +6.91% | [+6.58%, +7.47%] |
| PRE_MATCH | AH 1-2 (média) | < 10s | 35 | 18 | +6.94% | [+6.63%, +7.62%] |
| IN_MATCH | AH 0-1 (líquida) | < 10s | 21 | 16 | +7.05% | [+6.05%, +7.79%] |
| IN_MATCH | AH 1-2 (média) | < 10s | 14 | 14 | +6.56% | [+5.45%, +7.64%] |
| PRE_MATCH | AH 0-1 (líquida) | 20-30s | 12 | 10 | +7.40% | [+6.56%, +8.08%] |
| IN_MATCH | AH 2+ (extrema) | > 30s | 10 | 7 | +7.14% | [+4.59%, +8.32%] |
| PRE_MATCH | AH 2+ (extrema) | 20-30s | 7 | 5 | +7.24% | [+6.33%, +8.65%] |
| PRE_MATCH | AH 2+ (extrema) | > 30s | 6 | 5 | +6.24% | [+4.88%, +7.99%] |
| PRE_MATCH | AH 0-1 (líquida) | > 30s | 6 | 5 | +7.02% | [+6.04%, +7.90%] |
| IN_MATCH | AH 0-1 (líquida) | > 30s | 6 | 4 | +6.91% | [+5.89%, +7.94%] |

### 9.2 Lay combos (diff_pct <= corte) + risco
| Regime | Linha | Lag | N | Jogos | Mean diff % | IC90 cluster | Liability p95 |
|---|---|---|---:|---:|---:|---|---:|
| IN_MATCH | AH 2+ (extrema) | < 10s | 40 | 33 | -4.97% | [-5.73%, -4.52%] | 290.66 |
| IN_MATCH | AH 0-1 (líquida) | < 10s | 23 | 19 | -5.26% | [-6.07%, -4.65%] | 191.72 |
| PRE_MATCH | AH 0-1 (líquida) | < 10s | 13 | 11 | -4.86% | [-6.47%, -4.10%] | 70.95 |
| PRE_MATCH | AH 2+ (extrema) | < 10s | 11 | 11 | -4.53% | [-5.54%, -3.57%] | 1022.80 |
| IN_MATCH | AH 1-2 (média) | < 10s | 6 | 6 | -3.43% | [-4.00%, -3.01%] | 867.76 |
| PRE_MATCH | AH 1-2 (média) | < 10s | 6 | 5 | -4.45% | [-6.25%, -3.07%] | 81.80 |
| IN_MATCH | AH 2+ (extrema) | > 30s | 5 | 5 | -4.39% | [-6.60%, -2.94%] | 182.86 |
| IN_MATCH | AH 0-1 (líquida) | 20-30s | 3 | 3 | -3.18% | [-3.50%, -2.85%] | 175.10 |
| IN_MATCH | AH 0-1 (líquida) | > 30s | 3 | 3 | -6.34% | [-8.04%, -4.63%] | 214.41 |
| IN_MATCH | AH 2+ (extrema) | 20-30s | 3 | 3 | -5.17% | [-5.79%, -4.54%] | 273.15 |
| PRE_MATCH | AH 0-1 (líquida) | > 30s | 2 | 2 | -3.51% | [-3.67%, -3.35%] | 133.35 |
| PRE_MATCH | AH 2+ (extrema) | 20-30s | 2 | 2 | -4.59% | [-6.26%, -2.92%] | 80.99 |

---
## 10) Diagnóstico: por que o ROI pode estar zerado
ROI aqui é calculado por placar do jogo (`matches.home_score/away_score`). Se os placares não estiverem preenchidos no banco, a cobertura de ROI será 0.

| Indicador | Valor |
|---|---:|
| Jogos únicos no recorte | 160 |
| Jogos com placar disponível (home_score/away_score não nulos) | 130 |
| Jogos com status='finished' no banco | 130 |

### 10.1 Distribuição por data de kickoff (explica janela da API)
- Kickoff (UTC) no recorte: **2026-02-13 00:30 UTC** até **2026-02-14 23:30 UTC**.

| Kickoff date (UTC) | Jogos | Com placar | Cobertura |
|---|---:|---:|---:|
| 2026-02-14 | 110 | 97 | 88.2% |
| 2026-02-13 | 50 | 33 | 66.0% |

**Leitura**: se seu recorte inclui muitos jogos com kickoff antigo, a API-Football **free** pode não retornar fixtures dessa data (limitação por janela recente). Nesse cenário, mesmo com o job rodando, `placar disponível` ficará baixo para jogos fora da janela.

Se `placar disponível` estiver 0 (mesmo para datas recentes), isso geralmente indica que o job de resultados não rodou ou está sem chave válida.  
Sugestão (rodar no servidor): `cd betinasia_bot && python3 -m results.auto_update_results --once` (ou configure o serviço para rodar em loop).

---
## 11) Conclusões, riscos e pontos em aberto
- **Execução (CLV)**: use as seções 3/6 para validar qualidade de execução (especialmente pre-match). Se CLV cluster ficar robustamente positivo, há evidência de boa entrada; se ficar negativo, há erosão estrutural.
- **Pre-match vs in-match**: valide que o comportamento de edge/diff e ROI (quando houver placar) difere entre regimes (seção 2.2). Não é seguro misturar regimes para decisão.
- **Lay**: não pode ser decidido por média. Governança tem que usar p95/p99/ES95 de liability (seção 7.2) e combos com risco (seção 9.2). Se p99/ES95 forem altos, a estratégia precisa limite de exposição por janela.
- **Temporal (retenção de edge)**: se a cobertura `temporal/lay_temporal` for baixa, a inferência de retenção fica limitada (seção 8). Quando há cobertura, delta e retenção indicam se o edge “some” rápido.
- **ROI/resultado realizado**: sem placares no banco, ROI fica 0/ausente e a conclusão financeira final não é possível (seção 10). Primeiro garanta o job de resultados.
- **Pontos em aberto típicos**: (i) trazer DOM para a mesma janela, (ii) garantir atualização de placares, (iii) aumentar cobertura temporal e finance no `hypothesis_details`, (iv) definir política de banca para Lay.

---
## 12) Como reproduzir
1. Configure `betinasia_bot/.env` com `DATABASE_URL`.  
2. (Opcional) Atualize resultados para ter ROI: `cd betinasia_bot && python3 -m results.auto_update_results --once`.  
3. Execute:

```bash
python3 betinasia_bot/analyze_contexto_operacao_b808_robust_report.py \
  --direction up \
  --versions v4.0-api,v1.0,v1.0-recovered \
  --lookback-days 14 \
  --out betinasia_bot/docs/analise_contexto_operacao_b808_robusta.md \
  --pdf betinasia_bot/docs/analise_contexto_operacao_b808_robusta.pdf
```
