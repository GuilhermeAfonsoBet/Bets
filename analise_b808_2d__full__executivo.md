# Análise Estatística Robusta — Contexto Operação (b808) — Versão Executiva (com interpretação)
**Gerado em:** 2026-02-15 15:30 UTC  
**Fonte:** relatório estatístico base (apêndice), extraído do repositório e enriquecido com interpretação.

---
## 0) Leitura executiva (o que importa e como decidir)

Este documento foi escrito para responder a uma pergunta prática:
**o robô está capturando valor de preço de forma consistente, com execução suficientemente rápida para que o sinal (CLV) seja preservado, e com qualidade de dado adequada para extrapolar para uma fase de execução?**

### 0.1 O que o recorte mostra (em linguagem de decisão)
- **Recorte**: direction=`up`, lookback_days=`2`, versions=`v4.0-api,v1.0,v1.0-recovered`.
- **Amostra**: 1837 auditorias (jogos únicos=162, média=11.3 obs/jogo); betslip confiável=890.
- **Janela efetiva**: 13/02 15:26 → 15/02 15:18 UTC (span≈2.0d; dias com dados=3).
- **Coortes operacionais**: Back (diff>=2.0%): **308**; Lay (diff<=-2.0%): **116**.
- **Cobertura de placar/ROI**: jogos com placar=127/162 (status finished=127).
- **DOM**: sem dados no recorte atual (N=0), então não há comparação API vs DOM aqui.

O sinal mais importante neste recorte é **CLV pre‑match positivo e robusto**:
- **CLV (pre‑match, por jogo)**: **+2.063%** com **IC90 [+1.502%, +2.616%]**.

Interpretação: em média, quando a auditoria aponta edge e a execução ocorre no pre‑match, o preço efetivamente obtido no betslip tende a “ganhar do mercado” (closing) em magnitude material, com confiança estatística **por jogo** (robustez contra correlação intra‑jogo).

### 0.2 O que eu não concluiria ainda (evitar erro caro)
- **ROI** ainda tem **incerteza grande** neste recorte. Mesmo quando a média parece positiva/negativa em alguma coorte, os intervalos por jogo são largos e a cobertura de jogos finalizados nem sempre é completa. Para ROI, trate este PDF como *diagnóstico*, não como validação final.
- **In‑match**: CLV “clássico” (vs closing pré‑jogo) **não é interpretável** in‑match. Use in‑match para outras perguntas (latência, integridade do pipeline), mas não para concluir edge pre‑match.
- **DOM ausente (N=0)**: não há comparação API vs DOM neste recorte; qualquer conclusão “API melhor que DOM” aqui seria especulação.

### 0.3 Diagnóstico operacional (execução)
Do ponto de vista de execução (detecção→betslip):
- **lag_e2e**: p50≈7695ms, p95≈1832ms
- **overhead**: p50≈2547ms (proxy de fila/retries fora das etapas instrumentadas)

Tradução: o pipeline API está “rápido o suficiente” na mediana, mas ainda existe cauda (p95) e regimes lentos. A lição operacional é simples:
**o edge é perecível**; portanto, regimes lentos devem ser tratados como *degradação de qualidade* e não como “só mais devagar”.

### 0.4 Sinal de consistência: BS vs WS
O relatório base mede também a diferença média entre o preço no betslip (BS) e o preço via WS (WS):
- **Diff BS vs WS (média)**: +1.514% (sig. positivo, N=890)

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


# Análise Estatística Robusta — Contexto Operação (b808)
**Data da execução:** 15/02/2026 15:24 UTC  
**Escopo:** H3B (auditoria), com comparação por `audit_version` e inferência robusta por jogo (cluster bootstrap).  
**Nota:** a robustez aqui significa que intervalos de confiança consideram correlação intra-jogo (múltiplas auditorias por partida).
---
## 0) Sumário executivo (leitura rápida)
- **Recorte**: direction=`up`, lookback_days=`2`, versions=`v4.0-api,v1.0,v1.0-recovered`.
- **Amostra**: 1837 auditorias (jogos únicos=162, média=11.3 obs/jogo); betslip confiável=890.
- **Janela efetiva (audited_at)**: 13/02 15:26 → 15/02 15:18 UTC (span≈2.0d; dias com dados=3).
- **Coortes (status=OK, betslip confiável)**: Back (diff>=2.0%): **308**; Lay (diff<=-2.0%): **116**.
- **Coberturas em `hypothesis_details` (OK)**: temporal(back)=661/890; lay_temporal=630/890; finance=678/890.
- **Cobertura de placar (ROI)**: jogos com placar=127/162 (status finished=127).
- **DOM**: sem dados no recorte atual (N=0), então não há comparação API vs DOM aqui.
- **CLV pre-match (Betslip, API)**: média robusta por jogo +2.063% (IC90 [+1.502%, +2.616%]), com N=378 eventos (jogos=92).
- **Padrão por bucket (CLV PM)**: `BS < WS` -4.367% (sig. negativo), `BS ~ WS` +0.094% (NS), `BS > WS` +6.790% (sig. positivo).
- **Leitura recomendada**: use este relatório para validar **qualidade de execução/CLV**. Para concluir sobre **ROI**, rode após atualizar resultados e/ou use uma janela com jogos já liquidados.

---
## 1) Contexto do corte (b808)
| Indicador | Valor |
|---|---:|
| Auditorias H3B `UP` (match + kickoff passado) | 1837 |
| Betslip bruto | 1461 |
| Betslip confiável (diff -10% a +10%) | 890 |
| Descartados no filtro de qualidade | 571 |
| Jogos únicos (geral) | 162 |
| Média de observações por jogo | 11.3 |
| Jogos únicos com betslip confiável | 156 |

---
## 2) Base comparativa: API vs DOM
| Métrica | API (v4.0-api) | DOM (v1.0) |
|---|---:|---:|
| Total de observações | 1837 | 0 |
| Com betslip confiável | 890 | 0 |
| Com CLV pre-match (betslip) | 378 | 0 |
| Com ROI (betslip) | 678 | 0 |
| Tempo total observado (detecção→betslip, wall/total) | 5312 ms | — ms |
| Tempo instrumentado (detecção→clique→betslip) | 4425 ms | — ms |

---
### 2.0b Decomposição do tempo (detecção→clique→betslip vs. overhead)
Interpretação: `lag_e2e` é o **tempo fim-a-fim** (detecção→clique + clique→betslip). `overhead` = `lag_total` − `lag_e2e` (proxy de fila/retries/esperas fora das 2 etapas instrumentadas).

| Modelo | Métrica | mean (ms) | p50 (ms) | p95 (ms) | N |
|---|---|---:|---:|---:|---:|
| API (2-4s) | lag_det→click | 1882 | 840 | 2946 | 1837 |
| API (2-4s) | lag_click→betslip | 2721 | 2282 | 4573 | 1832 |
| API (2-4s) | lag_e2e (soma) | 4425 | 3582 | 7695 | 1832 |
| API (2-4s) | audit_total (duração) | 5307 | 4739 | 8223 | 1837 |
| API (2-4s) | overhead (total - e2e) | 704 | 76 | 2547 | 1832 |
| DOM (15-30s) | lag_det→click | — | — | — | 0 |
| DOM (15-30s) | lag_click→betslip | — | — | — | 0 |
| DOM (15-30s) | lag_e2e (soma) | — | — | — | 0 |
| DOM (15-30s) | audit_total (duração) | — | — | — | 0 |
| DOM (15-30s) | overhead (total - e2e) | — | — | — | 0 |

**Diagnóstico de cauda (percentual acima do limiar)**

| Modelo | % det→click > 5s | % det→click > 20s | % total > 10s | % total > 40s | % overhead < 0 |
|---|---:|---:|---:|---:|---:|
| API (2-4s) | 3.8% | 1.5% | 3.8% | 0.5% | 0.0% |
| DOM (15-30s) | —% | —% | —% | —% | —% |

---
### 2.1 Cobertura temporal (pre-match vs in-match)
| Métrica | Pre-match | In-match | Observação |
|---|---:|---:|---|
| Observações totais com classificação temporal | 650 | 1187 | Contagem bruta do corte |
| ROI Betslip | 351 | 327 | Amostra com resultado do jogo |
| ROI WebSocket | 489 | 949 | Referência de mercado |
| CLV (apenas pre-match) | 378 | — | CLV vs closing pré-jogo não é interpretável in-match |

---
### 2.2 Performance por regime (pre-match vs in-match)
| Regime | N auditorias | N betslip conf. | N OK | Back edge | Lay edge | Diff médio (OK) |
|---|---:|---:|---:|---:|---:|---:|
| PRE_MATCH | 650 | 457 | 457 | 177 | 38 | +2.090% |
| IN_MATCH | 1187 | 433 | 433 | 131 | 78 | +0.906% |

---
### 2.3 Regimes operacionais por tempo total (bucket)
Objetivo: separar a amostra em **regimes de execução** (tempo total) e medir performance em cada regime. Use isso para detectar **fila/saturação**: regimes lentos tendem a ter edge menor e pior ROI.

| Bucket (tempo total) | N OK | Jogos | lag_total mean (ms) | lag_total p95 (ms) | overhead p95 (ms) | Back edge | Lay edge | CLV PM (mean; IC90) | ROI (mean; IC90) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| < 5s | 698 | 151 | 3662 | 4874 | 2515 | 228 | 79 | +2.25% [+1.70%, +2.80%] | +5.79% [-1.72%, +13.49%] |
| 5-10s | 169 | 100 | 5880 | 7971 | 3091 | 67 | 36 | +1.45% [+0.33%, +2.57%] | -10.24% [-24.92%, +4.53%] |
| 10-20s | 13 | 12 | 13204 | 18429 | 5197 | 5 | 1 | +1.47% [-0.09%, +3.05%] | +34.18% [-4.22%, +71.44%] |
| 20-40s | 8 | 8 | 27867 | 37289 | 2533 | 7 | 0 | +3.92% [+0.80%, +7.04%] | +0.25% [-57.14%, +58.19%] |
| > 40s | 2 | 2 | 54752 | 57130 | 2395 | 1 | 0 | -0.25% — | -100.00% — |
| Desconhecido | 0 | 0 | — | — | — | 0 | 0 | — | — |

---
## 3) CLV pre-match (núcleo)
### 3.1 CLV com odd do Betslip (execução real)
| Métrica | API | DOM |
|---|---:|---:|
| CLV Bruto BS Pre-Match | +2.379% (sig. positivo, N=378, jogos=92) | — (N/A, N=0, jogos=0) |
| CLV Adicional BS Pre-Match | +1.956% (sig. positivo, N=374, jogos=91) | — (N/A, N=0, jogos=0) |
| Taxa de CLV > 0 (bruto) | 62.4% | —% |
| Taxa de CLV > 0 (adicional) | 60.2% | —% |

Notas de robustez (IC 90% por jogo):  
- API CLV bruto (cluster): média +2.063%; IC90 [+1.502%, +2.616%]  
- DOM CLV bruto (cluster): média —; IC90 —  

---
## 4) ROI por modelo
| Métrica | API | DOM |
|---|---:|---:|
| ROI Betslip | +2.844% (NS, N=678) | — (N/A, N=0) |
| ROI WebSocket | -0.659% (NS, N=1430) | — (N/A, N=0) |
| Win rate ROI Betslip | 51.7% | —% |
| Win rate ROI WS | 49.9% | —% |

Notas de robustez (IC 90% por jogo):  
- API ROI betslip (cluster): média +4.214%; IC90 [-2.576%, +11.061%]  
- API ROI WS (cluster): média -2.056%; IC90 [-4.866%, +0.641%]  

---
## 5) Diferença de preço BS vs WS
| Métrica | API | DOM |
|---|---:|---:|
| Diff BS vs WS (média) | +1.514% (sig. positivo, N=890) | — (N/A, N=0) |
| BS > WS | 47.4% (422/890) | —% (0/0) |
| BS > WS +2% | 34.6% (308/890) | —% (0/0) |

---
## 6) Combinações de valor
### 6.1 Buckets por diferença BS vs WS
| Bucket | N bucket | CLV BS PM (média) | IC90 (cluster) | N CLV PM | Jogos CLV PM | ROI BS (todos) | IC90 (cluster) |
|---|---:|---:|---|---:|---:|---:|---|
| BS < WS (-10% a -2%) | 116 | -4.367% | [-5.781%, -3.439%] | 30 | 25 | +6.781% | [-8.455%, +23.626%] |
| BS ~ WS (-2% a +2%) | 466 | +0.094% | [-0.258%, +0.584%] | 199 | 84 | +1.423% | [-9.142%, +7.901%] |
| BS > WS (+2% a +10%) | 308 | +6.790% | [+6.331%, +7.340%] | 149 | 67 | +3.455% | [-5.275%, +21.124%] |

---
### 6.2 Combinação por faixa de linha AH
| Faixa AH | CLV BS PM (média) | IC90 (cluster) | ROI BS (todos) | IC90 (cluster) | Diff BS vs WS (média) |
|---|---:|---|---:|---|---:|
| AH 0-1 (líquida) | +2.681% | [+1.664%, +3.103%] | +6.126% | [+4.189%, +26.205%] | +1.727% |
| AH 1-2 (média) | +2.403% | [+1.222%, +3.915%] | +0.703% | [-16.131%, +15.580%] | +2.265% |
| AH 2+ (extrema) | +1.892% | [+0.881%, +2.887%] | +1.304% | [-6.915%, +15.948%] | +1.102% |

---
### 6.3 Combinação por faixa de lag
| Faixa de lag | CLV BS PM (média) | IC90 (cluster) | N CLV PM | Jogos CLV PM | ROI BS (todos) | IC90 (cluster) | Diff BS vs WS (média) |
|---|---:|---|---:|---:|---:|---|---:|
| < 10s | +2.384% | [+1.543%, +2.647%] | 370 | 92 | +2.503% | [-5.196%, +9.164%] | +1.481% |
| 10-20s | +1.493% | [-0.093%, +3.047%] | 4 | 4 | +31.358% | [-4.218%, +71.437%] | +1.164% |
| 20-30s | +3.920% | [+0.797%, +7.043%] | 3 | 3 | +40.960% | [-20.460%, +100.500%] | +5.426% |
| > 30s | -0.246% | — | 1 | 1 | -100.000% | [-100.000%, -100.000%] | +3.885% |

---
## 7) Estimativa financeira (proxy) e risco
Este bloco usa `hypothesis_details.finance` quando existe; se não existir, usa stake fallback = `stake_pct_of_limit × limit`.

**Política de stake (proxy)**

| Parâmetro | Valor |
|---|---:|
| stake_pct_of_limit | 0.25 |
| stake_cap | 0.00 |
| Cobertura finance (OK, betslip conf.) | 678/890 |

### 7.1 Back (BS >> WS)
| Métrica | Valor |
|---|---:|
| Corte (diff_pct) | >= 2.0% |
| N eventos | 308 |
| Cobertura finance (na coorte) | 235/308 |
| Stake total (estimado) | 118542.72 |
| Stake médio | 384.88 |
| Profit_if_win total (estimado) | 125210.98 |
| Profit_if_win médio | 406.53 |
| N com ROI realizado | 242 |
| P&L realizado total (estimado) | -17806.32 |
| ROI realizado (ponderado por stake) | -16.81% |
| ROI realizado (robusto por jogo, mean; IC90) | +7.88% [-5.28%, +21.12%] |
| ROI ponderado por stake (robusto por jogo, mean; IC90) | -0.02% [-14.58%, +14.02%] |

**Como ler as 3 linhas de ROI (Back)**

- **ROI realizado (ponderado por stake)**: ΣP&L / Σstake (pode ser dominado por stakes grandes).
- **ROI realizado (robusto por jogo)**: média por jogo (cada jogo pesa parecido; reduz dominância de outliers).
- **ROI ponderado por stake (robusto por jogo)**: calcula ROI ponderado dentro do jogo e depois faz média/IC por jogo.

Observação: a Seção 4 reporta ROI médio no **recorte completo** (betslip confiável). Aqui (7.1) é apenas a coorte **Back edge** (diff>=corte) e o ROI também é mostrado ponderado por stake; por isso sinais podem divergir.

### 7.2 Lay (BS << WS) — risco de cauda
| Métrica | Valor |
|---|---:|
| Corte (diff_pct) | <= -2.0% |
| N eventos | 116 |
| Cobertura finance (na coorte) | 97/116 |
| Stake total (estimado) | 13170.36 |
| Liability total (estimada) | 12366.62 |
| Liability média | 106.61 |
| Liability p95 | 553.64 |
| Liability p99 | 1169.43 |
| ES95 (liability) | 1048.31 |
| Liability max | 2089.57 |
| Proxy de banca (>= p99 liability) | 1169.43 |
| N com ROI realizado | 54 |
| P&L realizado total (estimado) | -846.83 |
| ROI realizado (ponderado por liability) | -10.04% |
| ROI realizado (ponderado por stake) | -9.39% |
| ROI/liability (robusto por jogo, mean; IC90) | +11.36% [-11.98%, +34.96%] |
| ROI/liability ponderado (robusto por jogo, mean; IC90) | +5.22% [-18.96%, +29.93%] |

### 7.3 Projeção mensal (30 dias fixo) — turnover, lucro e banca
Premissas:
- **Mês = 30 dias fixo**.
- Turnover = soma de stakes (Back e Lay). Para Lay, também reportamos exposição por **liability**.
- **Banca conservadora** = p99(exposição unitária). **Banca agressiva** = ES95(exposição unitária).
- Projeção de lucro usa duas visões: (i) **lucro/dia observado** e (ii) **ROI observado × turnover projetado**.

| Bloco | Janela(d) | Turnover 30d | Lucro 30d (direto) | Lucro 30d (ROI×turnover) |
|---|---:|---:|---:|---:|
| Back | 2.0 | 1778140.83 | -267094.76 | -298872.44 |
| Lay (stake) | 2.0 | 197555.37 | -12702.40 | -18542.30 |
| Total (Back+Lay) | 2.0 | 1975696.21 | -279797.16 | -317414.74 |

**Risco/Banca (exposição unitária)**

| Bloco | Banca conservadora (p99) | Banca agressiva (ES95) | ROI/banca 30d (direto) | ROI/banca 30d (ROI×turnover) |
|---|---:|---:|---:|---:|
| Back (stake) | 4393.27 | 4228.14 | -6079.64% | -6802.96% |
| Lay (liability) | 1169.43 | 1048.31 | -1086.21% | -1585.59% |
| Total (soma) | 5562.70 | 5276.45 | -5029.88% | -5706.13% |

**Diagnóstico de cobertura (placar/ROI)**

| Bloco | Turnover total | Turnover com ROI | Cobertura turnover |
|---|---:|---:|---:|
| Back | 118542.72 | 105938.64 | 89.37% |
| Lay | 13170.36 | 9022.36 | 68.51% |

Notas (Lay): exposição 30d por liability (não é turnover) = 185499.25; ROI realizado por liability (ponderado) = -10.04%.

---
## 8) Curva temporal (pico, reversão e melhor timing)
Esta seção usa `hypothesis_details.temporal` (Back) e `hypothesis_details.lay_temporal` (Lay), coletados em pontos discretos (t≈0,3,6,10,15,20s).

O objetivo é responder: **tempo até o pico/vale**, **% que segue melhorando até t_max**, **% com reversão** e **impacto esperado por timing**. **CLV é reportado somente pre-match** (closing pré-jogo).

### 8.1 Back (pico em diff_pct)
Definições: **pico** = `max(diff_pct)`. **reversão** = após o pico, `diff_pct` cair pelo menos 0.50 p.p. (para Lay, subir 0.50 p.p. após o vale). `t_reversão` é o 1º tempo que cruza esse limiar (logo, não é o pico).

| Regime | N | t_pico médio (s) | t_pico p50 (s) | % melhora até fim | % com reversão | t_reversão médio (s) | Δt (rev - pico) médio (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRE_MATCH | 457 | 4.8 | 5.2 | 72.2% | 19.7% | 12.8 | 7.6 |
| IN_MATCH | 433 | 4.8 | 0.0 | 65.8% | 25.9% | 13.8 | 8.0 |

**Partição 100% (Back)**: categorias exclusivas (somam 100% por regime).

| Regime | % pico no fim, sem reversão | % pico no fim, com reversão (recuperou) | % pico antes, com reversão | % pico antes, sem reversão relevante |
|---|---:|---:|---:|---:|
| PRE_MATCH | 77.5% | 7.9% | 11.8% | 2.8% |
| IN_MATCH | 72.1% | 3.2% | 22.6% | 2.1% |

**Curva média (Back)**: média de `diff_pct` e `odd` por tempo. `CLV` é reportado **somente pre-match** (closing pré-jogo). `ROI` (quando aparece) é o **ROI realizado** se a aposta fosse feita naquele ponto.

| Tempo | N pts | diff_pct médio | odd média | CLV médio (pre-match) | ROI médio (se houver) |
|---|---:|---:|---:|---:|---:|
| t+0s | 890 | +1.51% | 2.009 | +2.38% | 2.84 |
| t+6s | 647 | +2.24% | 2.016 | +2.89% | 1.94 |
| t+10s | 1044 | +3.03% | 2.036 | +3.02% | 4.32 |
| t+15s | 634 | +3.35% | 2.041 | +2.91% | 2.02 |
| t+20s | 926 | +2.40% | 2.041 | +3.14% | -0.84 |

### 8.1b Back — impacto por timing (t0 vs pico vs último) e reversão
Leitura: se a estratégia é **entrar no pico**, compare CLV/ROI no pico vs t0 e veja diferença entre **casos com reversão** vs **sem reversão**. Aqui reportamos **média robusta por jogo + IC90**. Observação: **CLV só é calculado pre-match**.

**CLV (somente pre-match) — média robusta por jogo (IC90)**

| Subcoorte | N total | N CLV (PM) | CLV t0 (mean; IC90) | CLV pico (mean; IC90) | CLV último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 688 | 304 | +1.93% [+1.38%, +2.46%] | +2.26% [+1.74%, +2.78%] | +2.25% [+1.73%, +2.77%] |
| COM_REVERSAO | 202 | 74 | +3.55% [+2.47%, +4.63%] | +5.22% [+3.93%, +6.48%] | +4.29% [+3.01%, +5.59%] |

**ROI (stake=1) — média robusta por jogo (IC90)**

| Subcoorte | N total | N ROI | ROI t0 (mean; IC90) | ROI pico (mean; IC90) | ROI último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 688 | 535 | +4.26% [-3.78%, +12.30%] | +4.58% [-3.50%, +12.65%] | +4.56% [-3.51%, +12.64%] |
| COM_REVERSAO | 202 | 143 | +4.50% [-10.50%, +19.62%] | +8.95% [-6.68%, +25.03%] | +4.68% [-10.39%, +20.02%] |

Nota interpretativa: a subcoorte **COM_REVERSAO** pode ter CLV maior no **pico** porque, por definição, ela inclui casos em que o edge atingiu um extremo mais alto antes de reverter. Para entender a hipótese 'sem reversão deveria ser maior', compare também a categoria '**pico no fim, sem reversão**' na partição 100% (tabela 8.1).

**Decomposição (pre-match): entry_odd vs closing_odd (IC90)**

| Subcoorte | N CLV (PM) | entry_odd t0 (mean; IC90) | entry_odd pico (mean; IC90) | closing_odd (mean; IC90) |
|---|---:|---:|---:|---:|
| SEM_REVERSAO | 304 | 1.990 [+1.976, +2.004] | 2.000 [+1.986, +2.014] | 1.966 [+1.956, +1.976] |
| COM_REVERSAO | 74 | 2.033 [+2.010, +2.058] | 2.069 [+2.043, +2.096] | 1.965 [+1.949, +1.981] |

### 8.2 Lay (vale em diff_pct)
| Regime | N | t_vale médio (s) | t_vale p50 (s) | % melhora até fim | % com reversão | t_reversão médio (s) | Δt (rev - vale) médio (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRE_MATCH | 405 | 4.8 | 5.3 | 66.4% | 25.2% | 13.4 | 7.5 |
| IN_MATCH | 317 | 5.6 | 0.0 | 49.8% | 39.7% | 13.4 | 8.6 |

**Partição 100% (Lay)**: categorias exclusivas (somam 100% por regime).

| Regime | % vale no fim, sem reversão | % vale no fim, com reversão (recuperou) | % vale antes, com reversão | % vale antes, sem reversão relevante |
|---|---:|---:|---:|---:|
| PRE_MATCH | 69.9% | 8.9% | 16.3% | 4.9% |
| IN_MATCH | 56.8% | 3.8% | 36.0% | 3.5% |

**Curva média (Lay)**: usa `odd=lay_odd`. `CLV` é reportado **somente pre-match** (closing pré-jogo). O ROI mostrado aqui é **ROI por liability** (não por stake), porque Lay é governado por risco.

| Tempo | N pts | diff_pct médio | lay_odd média | CLV médio (pre-match) | ROI/liability médio (se houver) |
|---|---:|---:|---:|---:|---:|
| t+0s | 722 | +0.97% | 2.009 | -0.44% | 17.85 |
| t+6s | 618 | +1.39% | 2.014 | -0.23% | 60.44 |
| t+10s | 982 | +0.81% | 2.010 | -0.11% | 57.90 |
| t+15s | 612 | +1.44% | 2.021 | -0.54% | 32.75 |
| t+20s | 893 | +1.60% | 2.034 | -0.38% | 27.51 |

### 8.2b Lay — impacto por timing (t0 vs vale vs último) e reversão
Leitura: se a estratégia é **entrar no vale (odd mais baixa)**, compare CLV/ROI no vale vs t0 e veja diferença entre **casos com reversão** vs **sem reversão**. Aqui reportamos **média robusta por jogo + IC90**. Observação: **CLV só é calculado pre-match**.

**CLV (somente pre-match) — média robusta por jogo (IC90)**

| Subcoorte | N total | N CLV (PM) | CLV t0 (mean; IC90) | CLV vale (mean; IC90) | CLV último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 494 | 246 | -1.13% [-1.87%, -0.36%] | -0.67% [-1.38%, +0.07%] | -0.69% [-1.40%, +0.04%] |
| COM_REVERSAO | 228 | 81 | +0.55% [-0.51%, +1.59%] | +2.01% [+0.85%, +3.19%] | +0.65% [-0.59%, +1.83%] |

**ROI/liability — média robusta por jogo (IC90)**

| Subcoorte | N total | N ROI | ROI/liab t0 (mean; IC90) | ROI/liab vale (mean; IC90) | ROI/liab último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 494 | 376 | +0.82% [-9.88%, +11.31%] | +1.17% [-9.50%, +11.70%] | +1.14% [-9.52%, +11.66%] |
| COM_REVERSAO | 228 | 164 | +14.05% [-3.34%, +32.02%] | +19.93% [-0.10%, +41.36%] | +13.25% [-4.51%, +31.86%] |

---
## 9) Combinações de valor (regime × linha AH × lag)
Tabela rankeada por volume (N). Métrica: diff_pct (OK, betslip confiável).

### 9.1 Back combos (diff_pct >= corte)
| Regime | Linha | Lag | N | Jogos | Mean diff % | IC90 cluster |
|---|---|---|---:|---:|---:|---|
| PRE_MATCH | AH 0-1 (líquida) | < 10s | 84 | 34 | +6.84% | [+6.75%, +7.44%] |
| IN_MATCH | AH 2+ (extrema) | < 10s | 77 | 52 | +6.13% | [+5.61%, +6.67%] |
| PRE_MATCH | AH 2+ (extrema) | < 10s | 47 | 28 | +6.77% | [+6.32%, +7.26%] |
| PRE_MATCH | AH 1-2 (média) | < 10s | 42 | 20 | +6.60% | [+6.45%, +7.39%] |
| IN_MATCH | AH 0-1 (líquida) | < 10s | 26 | 22 | +6.83% | [+5.99%, +7.47%] |
| IN_MATCH | AH 1-2 (média) | < 10s | 19 | 18 | +6.71% | [+5.69%, +7.45%] |
| IN_MATCH | AH 2+ (extrema) | > 30s | 3 | 3 | +5.26% | [+3.14%, +7.38%] |
| PRE_MATCH | AH 0-1 (líquida) | 20-30s | 2 | 2 | +6.91% | [+5.24%, +8.57%] |
| IN_MATCH | AH 2+ (extrema) | 10-20s | 2 | 2 | +5.93% | [+2.03%, +9.83%] |
| IN_MATCH | AH 0-1 (líquida) | 20-30s | 2 | 2 | +6.20% | [+3.28%, +9.12%] |
| PRE_MATCH | AH 0-1 (líquida) | 10-20s | 1 | 1 | +2.20% | — |
| PRE_MATCH | AH 1-2 (média) | 10-20s | 1 | 1 | +4.52% | — |

### 9.2 Lay combos (diff_pct <= corte) + risco
| Regime | Linha | Lag | N | Jogos | Mean diff % | IC90 cluster | Liability p95 |
|---|---|---|---:|---:|---:|---|---:|
| IN_MATCH | AH 2+ (extrema) | < 10s | 46 | 36 | -5.18% | [-5.86%, -4.70%] | 379.53 |
| IN_MATCH | AH 0-1 (líquida) | < 10s | 26 | 22 | -5.32% | [-6.14%, -4.85%] | 213.72 |
| PRE_MATCH | AH 0-1 (líquida) | < 10s | 17 | 15 | -5.24% | [-6.67%, -4.50%] | 76.41 |
| PRE_MATCH | AH 2+ (extrema) | < 10s | 13 | 13 | -4.58% | [-5.48%, -3.68%] | 987.39 |
| PRE_MATCH | AH 1-2 (média) | < 10s | 8 | 5 | -5.38% | [-6.88%, -4.02%] | 79.04 |
| IN_MATCH | AH 1-2 (média) | < 10s | 5 | 5 | -3.52% | [-4.20%, -3.01%] | 893.60 |
| IN_MATCH | AH 2+ (extrema) | 10-20s | 1 | 1 | -5.81% | — | 84.09 |

---
## 10) Diagnóstico: por que o ROI pode estar zerado
ROI aqui é calculado por placar do jogo (`matches.home_score/away_score`). Se os placares não estiverem preenchidos no banco, a cobertura de ROI será 0.

| Indicador | Valor |
|---|---:|
| Jogos únicos no recorte | 162 |
| Jogos com placar disponível (home_score/away_score não nulos) | 127 |
| Jogos com status='finished' no banco | 127 |

### 10.1 Distribuição por data de kickoff (explica janela da API)
- Kickoff (UTC) no recorte: **2026-02-13 14:00 UTC** até **2026-02-15 15:15 UTC**.

| Kickoff date (UTC) | Jogos | Com placar | Cobertura |
|---|---:|---:|---:|
| 2026-02-15 | 20 | 3 | 15.0% |
| 2026-02-14 | 110 | 101 | 91.8% |
| 2026-02-13 | 32 | 23 | 71.9% |

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
