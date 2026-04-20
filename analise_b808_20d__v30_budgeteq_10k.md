# Análise Estatística Robusta — Contexto Operação (b808)
**Data da execução:** 23/02/2026 22:26 UTC  
**Escopo:** H3B (auditoria), com comparação por `audit_version` e inferência robusta por jogo (cluster bootstrap).  
**Nota:** a robustez aqui significa que intervalos de confiança consideram correlação intra-jogo (múltiplas auditorias por partida).
---
## 0) Sumário executivo (leitura rápida)
- **Recorte**: direction=`up`, lookback_days=`20`, versions=`v4.0-api,v5.0-ws-only,v5.1-ws-gate-lay`.
- **Amostra**: 7623 auditorias (jogos únicos=684, média=11.1 obs/jogo); betslip confiável=3770.
- **Janela efetiva (audited_at)**: 08/02 20:15 → 23/02 21:47 UTC (span≈15.1d; dias com dados=12).
- **Dias excluídos / missing** (UTC, não tratados como 0): manual=0 [—]; auto(ws-only sem Lay)=2 [2026-02-20, 2026-02-21]; auto(sem BS/WS/Lay)=2 [2026-02-17, 2026-02-18]; missing(sem dados)=0 [—].
- **Coortes (status=OK, betslip confiável)**: `BS>WS` (diff>=2.0%): **909**; `BS<WS` (diff<=-2.0%): **412**.
- **Coberturas em `hypothesis_details` (OK)**: temporal(BS)=2241/3770; lay_temporal(BS)=2079/3770; ws_series(WS)=1181/6377; finance=2001/3770.
- **Cobertura de placar (ROI)**: jogos com placar=606/684 (status finished=606).
- **Cobertura de closing_odd (AH)**: jogos com closing=356/684 (52.0%). CLV pre‑match depende disso.
- **Alerta**: cobertura de closing_odd está baixa. Isso tende a reduzir N de CLV e pode enviesar a leitura (os jogos “sem odds” ficam fora do CLV). Priorize estabilizar o collector e/ou condicionar análises a jogos com closing.
- **DOM**: sem dados no recorte atual (N=0), então não há comparação API vs DOM aqui.
- **CLV pre-match (Betslip, API)**: média robusta por jogo +0.925% (IC90 [+0.622%, +1.228%]), com N=2001 eventos (jogos=287).
- **Padrão por bucket (CLV PM)**: `BS < WS` -2.978% (sig. negativo), `BS ~ WS` -0.463% (sig. negativo), `BS > WS` +6.193% (sig. positivo).
- **Leitura recomendada**: use este relatório para validar **qualidade de execução/CLV**. Para concluir sobre **ROI**, rode após atualizar resultados e/ou use uma janela com jogos já liquidados.

---
## 1) Contexto do corte (b808)
| Indicador | Valor |
|---|---:|
| Auditorias H3B `UP` (match + kickoff passado) | 7623 |
| Betslip bruto | 5196 |
| Betslip confiável (diff -10% a +10%) | 3770 |
| Descartados no filtro de qualidade | 1426 |
| Jogos únicos (geral) | 684 |
| Média de observações por jogo | 11.1 |
| Jogos únicos com betslip confiável | 549 |
| Distribuição por market_type | AH=7623 |
| Jogos únicos (AH) no recorte | 684 |
| Jogos únicos (AH) com closing_odd disponível | 356 |
| Cobertura closing_odd (AH) | 52.0% |

---
## 2) Base comparativa: API vs DOM
| Métrica | API (v4.0-api) | DOM (v1.0) |
|---|---:|---:|
| Total de observações | 6079 | 0 |
| Com betslip confiável | 3770 | 0 |
| Com CLV pre-match (betslip) | 2001 | 0 |
| Com ROI (betslip) | 3437 | 0 |
| Tempo total observado (detecção→betslip, wall/total) | 12487 ms | — ms |
| Tempo instrumentado (detecção→clique→betslip) | 6796 ms | — ms |

---
### 2.0a Glossário de métricas (definições operacionais)
Este glossário existe para eliminar ambiguidades entre **tempo total**, **tempos instrumentados** e **overhead**.

- **`hypothesis_detected_at`**: timestamp (UTC) de detecção do evento que gerou a auditoria.
- **`audited_at`**: timestamp (UTC) em que a auditoria foi concluída/persistida.
- **`lag_total_ms` (tempo total observado / wall)**: proxy de tempo “de parede” do pipeline do evento até o betslip; quando disponível usa wall time (ex.: `audited_at - detected_at`).
- **`lag_det_to_click_ms` (detecção→clique)**: tempo até o robô executar o clique/ação de betslip.
- **`lag_click_to_betslip_ms` (clique→betslip)**: tempo até carregar/obter o payload do betslip após o clique.
- **`lag_e2e_ms` (tempo instrumentado)**: `lag_det_to_click_ms + lag_click_to_betslip_ms`.
- **`audit_total_ms` (duração da auditoria)**: duração instrumentada do ciclo de auditoria (pode diferir de `lag_total_ms` se houver esperas fora do escopo instrumentado).
- **`lag_overhead_ms` (overhead)**: `lag_total_ms - lag_e2e_ms`; agrega espera fora das duas etapas instrumentadas (ex.: fila, retries, pausas, latência externa).
- **`diff_pct` (BS vs WS)**: diferença percentual entre a odd do **betslip no momento da execução** (BS) e a odd do **WebSocket no momento da detecção** (WS): `(BS - WS) / WS * 100`. Importante: **BS e WS são medidos em instantes diferentes**, então este número mede principalmente **drift durante a execução + slippage/atualização** (e não “mispricing contemporâneo”).
- **Betslip confiável**: filtro de qualidade `diff_pct ∈ [-10%, +10%]` para reduzir casos de mismatch/parse incorreto.

---
### 2.0b Decomposição do tempo (detecção→clique→betslip vs. overhead)
Interpretação: `lag_e2e` é o **tempo fim-a-fim** (detecção→clique + clique→betslip). `overhead` = `lag_total` − `lag_e2e` (proxy de fila/retries/esperas fora das 2 etapas instrumentadas).

| Modelo | Métrica | mean (ms) | p50 (ms) | p95 (ms) | N |
|---|---|---:|---:|---:|---:|
| API (2-4s) | lag_det→click | 4230 | 779 | 4417 | 6077 |
| API (2-4s) | lag_click→betslip | 2534 | 2079 | 4065 | 5812 |
| API (2-4s) | lag_e2e (soma) | 6796 | 3200 | 7393 | 5812 |
| API (2-4s) | audit_total (duração) | 12488 | 4247 | 39422 | 6077 |
| API (2-4s) | overhead (total - e2e) | 6009 | 8 | 25148 | 5812 |
| DOM (15-30s) | lag_det→click | — | — | — | 0 |
| DOM (15-30s) | lag_click→betslip | — | — | — | 0 |
| DOM (15-30s) | lag_e2e (soma) | — | — | — | 0 |
| DOM (15-30s) | audit_total (duração) | — | — | — | 0 |
| DOM (15-30s) | overhead (total - e2e) | — | — | — | 0 |

**Diagnóstico de cauda (percentual acima do limiar)**

| Modelo | % det→click > 5s | % det→click > 20s | % total > 10s | % total > 40s | % overhead < 0 |
|---|---:|---:|---:|---:|---:|
| API (2-4s) | 4.6% | 2.3% | 14.7% | 4.9% | 0.0% |
| DOM (15-30s) | —% | —% | —% | —% | —% |

---
### 2.1 Cobertura temporal (pre-match vs in-match)
| Métrica | Pre-match | In-match | Observação |
|---|---:|---:|---|
| Observações totais com classificação temporal | 3872 | 3751 | Contagem bruta do corte |
| ROI Betslip | 2370 | 1067 | Amostra com resultado do jogo |
| ROI WebSocket | 3649 | 3337 | Referência de mercado |
| CLV (apenas pre-match) | 2001 | — | CLV vs closing pré-jogo não é interpretável in-match |

---
### 2.2 Performance por regime (pre-match vs in-match)
| Regime | N auditorias | N betslip conf. | N OK | Back edge | Lay edge | Diff médio (OK) |
|---|---:|---:|---:|---:|---:|---:|
| PRE_MATCH | 3872 | 2523 | 2523 | 589 | 180 | +1.055% |
| IN_MATCH | 3751 | 1247 | 1247 | 320 | 232 | +0.590% |

---
### 2.2c Quebra por liga (top por volume)
Objetivo: detectar não-uniformidade do edge por **liga**. Reporta volume, cobertura de closing (para CLV) e métricas robustas por jogo.

| Liga | N OK (conf.) | Jogos | Closing cov (jogos PM) | CLV PM (mean; IC90) | ROI (mean; IC90) | Back edge | Lay edge |
|---|---:|---:|---:|---:|---:|---:|---:|
| Italy Serie A | 307 | 22 | 100.0% | +1.25% [+0.79%, +1.69%] | -1.64% [-6.73%, +3.90%] | 90 | 25 |
| Spain La Liga | 291 | 22 | 100.0% | +0.96% [+0.20%, +1.67%] | +5.12% [-5.76%, +17.03%] | 76 | 20 |
| Germany Bundesliga | 249 | 18 | 100.0% | +0.74% [+0.15%, +1.28%] | +7.44% [-0.80%, +15.48%] | 54 | 21 |
| Club Friendly | 239 | 59 | 25.0% | +0.58% [-3.41%, +4.41%] | +0.25% [-17.17%, +18.85%] | 61 | 44 |
| France Ligue 1 | 234 | 18 | 100.0% | +0.29% [-0.49%, +1.10%] | +9.01% [-3.22%, +20.73%] | 66 | 15 |
| England Football League Championship | 229 | 25 | 84.0% | +0.46% [-0.19%, +1.08%] | -1.15% [-16.25%, +13.33%] | 59 | 13 |
| England Premier League | 212 | 21 | 85.7% | +0.49% [-0.22%, +1.21%] | -2.35% [-12.80%, +7.93%] | 38 | 34 |
| England National League | 189 | 29 | 57.1% | +1.04% [-0.85%, +3.04%] | +9.68% [-3.95%, +23.33%] | 43 | 29 |
| England League 1 | 174 | 27 | 88.9% | +0.80% [-0.32%, +1.96%] | -0.39% [-17.01%, +15.96%] | 49 | 20 |
| England League 2 | 170 | 27 | 85.2% | +0.42% [-0.51%, +1.36%] | -2.24% [-20.97%, +16.03%] | 48 | 15 |
| UEFA Europa League | 166 | 8 | 100.0% | -0.15% [-0.59%, +0.29%] | +0.60% [-13.71%, +17.75%] | 16 | 4 |
| Scotland Premier League | 138 | 13 | 53.8% | +2.45% [+1.58%, +3.39%] | -12.86% [-31.28%, +2.56%] | 40 | 14 |

---
### 2.3 Regimes operacionais por tempo total (bucket)
Objetivo: separar a amostra em **regimes de execução** (tempo total) e medir performance em cada regime. Use isso para detectar **fila/saturação**: regimes lentos tendem a ter edge menor e pior ROI.

| Bucket (tempo total) | N OK | Jogos | lag_total mean (ms) | lag_total p95 (ms) | overhead p95 (ms) | Back edge | Lay edge | CLV PM (mean; IC90) | ROI (mean; IC90) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| < 5s | 2821 | 507 | 3305 | 4778 | 1974 | 594 | 267 | +0.88% [+0.59%, +1.17%] | +2.24% [-1.92%, +6.40%] |
| 5-10s | 404 | 254 | 6230 | 8769 | 4468 | 124 | 64 | +0.96% [+0.33%, +1.63%] | +1.13% [-7.76%, +9.84%] |
| 10-20s | 61 | 51 | 14022 | 19117 | 16313 | 16 | 15 | -0.89% [-2.39%, +0.56%] | -18.81% [-38.61%, +0.80%] |
| 20-40s | 314 | 150 | 27116 | 34762 | 30902 | 105 | 40 | +2.04% [+1.21%, +2.87%] | +4.17% [-6.30%, +14.82%] |
| > 40s | 170 | 99 | 137561 | 465726 | 329045 | 70 | 26 | +1.55% [+0.63%, +2.50%] | -5.72% [-18.77%, +7.87%] |
| Desconhecido | 0 | 0 | — | — | — | 0 | 0 | — | — |

---
### 2.3b Regimes por tempo total — separando coortes por `diff_pct` (BS vs WS)
Nesta tabela, separamos duas coortes operacionais por **delta de execução**: `BS > WS` (diff_pct >= +2%) e `BS < WS` (diff_pct <= -2%). Isso **não** é (por si só) “Back vs Lay”; é um recorte por **melhora/piora do preço** entre detecção (WS) e execução (BS). CLV é reportado apenas em pre‑match.

| Bucket | N OK | N (BS>WS +2%) | N (BS<WS -2%) | CLV PM (BS>WS) | CLV PM (BS<WS) | ROI (BS>WS) | ROI (BS<WS) |
|---|---:|---:|---:|---:|---:|---:|---:|
| < 5s | 2821 | 594 | 267 | +6.39% [+5.89%, +6.88%] | -2.86% [-3.77%, -1.91%] | +1.91% [-6.57%, +10.28%] | -2.33% [-12.24%, +7.99%] |
| 5-10s | 404 | 124 | 64 | +5.46% [+4.73%, +6.16%] | -3.91% [-5.40%, -2.33%] | -3.93% [-19.35%, +11.78%] | -7.36% [-24.63%, +10.76%] |
| 10-20s | 61 | 16 | 15 | +3.46% [+0.34%, +6.60%] | -4.34% [-7.03%, -1.47%] | -19.08% [-59.33%, +21.46%] | -35.87% [-75.00%, +3.31%] |
| 20-40s | 314 | 105 | 40 | +7.19% [+6.20%, +8.15%] | -3.26% [-5.15%, -1.25%] | +5.10% [-11.27%, +22.04%] | +6.91% [-18.40%, +32.24%] |
| > 40s | 170 | 70 | 26 | +4.89% [+3.32%, +6.40%] | -1.73% [-3.63%, +0.28%] | -16.24% [-37.32%, +4.74%] | -7.05% [-38.60%, +24.14%] |
| Desconhecido | 0 | 0 | 0 | — | — | — | — |

---
### 2.3c Estabilidade temporal (por dia, `audited_at`)
Objetivo: checar se o regime de edge/execução é **time‑dependent**. Se houver dias com comportamento distinto (ex.: collector intermitente, horários/ligas), isso aparecerá aqui.

| Dia (UTC) | Modelo | N OK | Jogos | % BS>WS +2% | % BS<WS -2% | lag_total p50 (ms) | CLV PM (BS>WS) | CLV PM (BS<WS) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-02-08 | API (2-4s) | 55 | 37 | 7.3% | 5.5% | 2813 | +5.17% | +2.40% |
| 2026-02-09 | API (2-4s) | 225 | 117 | 11.6% | 14.7% | 4067 | +3.72% | -3.26% |
| 2026-02-10 | API (2-4s) | 673 | 188 | 12.2% | 12.8% | 3592 | +2.82% | -1.98% |
| 2026-02-11 | API (2-4s) | 247 | 88 | 30.8% | 16.2% | 25019 | +7.38% | -3.12% |
| 2026-02-12 | API (2-4s) | 59 | 47 | 35.6% | 8.5% | 26877 | +5.64% | -4.34% |
| 2026-02-13 | API (2-4s) | 605 | 153 | 35.9% | 10.7% | 4839 | +6.83% | -2.22% |
| 2026-02-14 | API (2-4s) | 539 | 164 | 35.6% | 14.7% | 4205 | +6.19% | -5.29% |
| 2026-02-15 | API (2-4s) | 489 | 131 | 31.3% | 11.5% | 3699 | +5.97% | -1.78% |
| 2026-02-16 | API (2-4s) | 329 | 114 | 38.0% | 5.8% | 3469 | +6.38% | -4.35% |
| 2026-02-19 | API (2-4s) | 549 | 126 | 2.4% | 4.7% | 2337 | +1.31% | -2.35% |

---
## 3) CLV pre-match (núcleo)
### 3.1 CLV com odd do Betslip (execução real)
| Métrica | API | DOM |
|---|---:|---:|
| CLV Bruto BS Pre-Match | +1.027% (sig. positivo, N=2001, jogos=287) | — (N/A, N=0, jogos=0) |
| CLV Adicional BS Pre-Match | +0.876% (sig. positivo, N=2001, jogos=287) | — (N/A, N=0, jogos=0) |
| Taxa de CLV > 0 (bruto) | 52.0% | —% |
| Taxa de CLV > 0 (adicional) | 54.1% | —% |

Notas de robustez (IC 90% por jogo):  
- API CLV bruto (cluster): média +0.925%; IC90 [+0.622%, +1.228%]  
- DOM CLV bruto (cluster): média —; IC90 —  

---
## 4) ROI por modelo
| Métrica | API | DOM |
|---|---:|---:|
| ROI Betslip | +0.328% (NS, N=3433) | — (N/A, N=0) |
| ROI WebSocket | -0.346% (NS, N=5495) | — (N/A, N=0) |
| Win rate ROI Betslip | 50.7% | —% |
| Win rate ROI WS | 50.4% | —% |

Notas de robustez (IC 90% por jogo):  
- API ROI betslip (cluster): média +0.141%; IC90 [-3.348%, +3.714%]  
- API ROI WS (cluster): média -1.954%; IC90 [-4.667%, +0.672%]  

---
## 4.1) Validade do CLV: relação CLV × ROI (pre-match)
Objetivo: avaliar se **CLV** (vs closing) é um bom proxy de **ROI realizado** (por placar), ao menos no regime **pre‑match**.

Regras do recorte desta seção:

- Apenas `status=OK` com betslip confiável (diff ∈ [-10%, +10%])
- Apenas `PRE_MATCH` (`is_live=False`)
- Exige **closing_odd** (para CLV) e **placar** (para ROI)

### 4.1a Estatística global (por jogo)
| Métrica | Valor |
|---|---:|
| Jogos com CLV+ROI | 273 |
| Eventos (auditorias) usados | 1924 |
| Correlação Pearson (mean por jogo) | 0.077 |
| Correlação Spearman (mean por jogo) | 0.072 |

### 4.1b Concordância de sinal (CLV vs ROI)
| CLV (jogo) | ROI (jogo) | Jogos |
|---|---|---:|
| > 0 | > 0 | 73 |
| > 0 | ≤ 0 | 91 |
| ≤ 0 | > 0 | 41 |
| ≤ 0 | ≤ 0 | 68 |

Leitura: CLV e ROI podem divergir por **variância do resultado** (ROI) e por **missingness** (jogos sem closing/sem placar). A correlação acima é um diagnóstico de “alinhamento”, não causalidade.

### 4.1c ROI por bucket de CLV (quintis; por jogo)
| Bucket (CLV por jogo) | Jogos | CLV mean (IC90) | ROI mean (IC90) | Win rate ROI |
|---|---:|---:|---:|---:|
| Q1 (-9.61%→-1.13%) | 55 | -2.905% [-3.352%, -2.492%] | -10.536% [-22.740%, +1.264%] | 36.4% |
| Q2 (-1.13%→+0.00%) | 54 | -0.402% [-0.478%, -0.324%] | -1.183% [-15.632%, +13.015%] | 38.9% |
| Q3 (+0.00%→+1.30%) | 55 | +0.597% [+0.510%, +0.684%] | +1.329% [-7.837%, +10.278%] | 47.3% |
| Q4 (+1.30%→+2.80%) | 54 | +1.959% [+1.860%, +2.062%] | +3.063% [-6.019%, +12.009%] | 42.6% |
| Q5 (+2.80%→+14.04%) | 55 | +5.247% [+4.732%, +5.792%] | +0.505% [-12.654%, +13.364%] | 43.6% |


---
## 5) Diferença de preço BS vs WS
| Métrica | API | DOM |
|---|---:|---:|
| Diff BS vs WS (média) | +0.901% (sig. positivo, N=3770) | — (N/A, N=0) |
| BS > WS | 40.3% (1521/3770) | —% (0/0) |
| BS > WS +2% | 24.1% (909/3770) | —% (0/0) |

---
## 6) Combinações de valor
### 6.1 Buckets por diferença BS vs WS
Nota de leitura:
- `BS > WS (+2% a +10%)`: preço no betslip ficou **melhor** do que o WS observado na detecção (delta positivo entre instantes).
- `BS < WS (-10% a -2%)`: preço no betslip ficou **pior** do que o WS observado na detecção (delta negativo entre instantes).
- O ROI abaixo é calculado **dentro do bucket** (não mistura buckets), mas ainda é sensível à cobertura de placar.

| Bucket | N bucket | CLV BS PM (média) | IC90 (cluster) | N CLV PM | Jogos CLV PM | ROI BS (todos) | IC90 (cluster) |
|---|---:|---:|---|---:|---:|---:|---|
| BS < WS (-10% a -2%) | 412 | -2.978% | [-3.789%, -2.437%] | 140 | 100 | -2.325% | [-12.938%, +3.362%] |
| BS ~ WS (-2% a +2%) | 2449 | -0.463% | [-0.617%, -0.102%] | 1360 | 264 | +0.638% | [-5.580%, +2.805%] |
| BS > WS (+2% a +10%) | 909 | +6.193% | [+5.935%, +6.746%] | 501 | 167 | +0.688% | [-3.046%, +11.224%] |

---
### 6.2 Combinação por faixa de linha AH
| Faixa AH | CLV BS PM (média) | IC90 (cluster) | ROI BS (todos) | IC90 (cluster) | Diff BS vs WS (média) |
|---|---:|---|---:|---|---:|
| AH 0-1 (líquida) | +1.044% | [+0.853%, +1.737%] | -0.285% | [-1.326%, +8.454%] | +0.969% |
| AH 1-2 (média) | +1.052% | [+0.224%, +1.626%] | +2.757% | [-3.332%, +12.962%] | +1.310% |
| AH 2+ (extrema) | +0.978% | [+0.159%, +1.093%] | -0.263% | [-5.538%, +5.634%] | +0.639% |

---
### 6.3 Combinação por faixa de lag
| Faixa de lag | CLV BS PM (média) | IC90 (cluster) | N CLV PM | Jogos CLV PM | ROI BS (todos) | IC90 (cluster) | Diff BS vs WS (média) |
|---|---:|---|---:|---:|---:|---|---:|
| < 10s | +0.945% | [+0.558%, +1.136%] | 1683 | 280 | +0.798% | [-2.082%, +5.519%] | +0.805% |
| 10-20s | -0.672% | [-2.387%, +0.560%] | 31 | 26 | -17.654% | [-38.609%, +0.798%] | +0.339% |
| 20-30s | +2.090% | [+1.323%, +3.082%] | 165 | 94 | +0.283% | [-7.804%, +14.407%] | +1.590% |
| > 30s | +1.154% | [+0.547%, +2.222%] | 122 | 73 | -1.504% | [-13.409%, +9.557%] | +1.637% |

---
## 7) Estimativa financeira (proxy) e risco
Este bloco usa `hypothesis_details.finance` quando existe; se não existir, usa stake fallback = `stake_pct_of_limit × limit`.

**Política de stake (proxy)**

| Parâmetro | Valor |
|---|---:|
| stake_pct_of_limit | 0.25 |
| stake_cap | 0.00 |
| Cobertura finance (OK, betslip conf.) | 2001/3770 |

### 7.1 Back (BS >> WS)
| Métrica | Valor |
|---|---:|
| Corte (diff_pct) | >= 2.0% |
| N eventos | 909 |
| Cobertura finance (na coorte) | 517/909 |
| Stake total (estimado) | 245001.88 |
| Stake médio | 269.53 |
| Profit_if_win total (estimado) | 259335.66 |
| Profit_if_win médio | 285.30 |
| N com ROI realizado | 835 |
| P&L realizado total (estimado) | -53513.93 |
| ROI realizado (ponderado por stake) | -22.44% |
| ROI realizado (robusto por jogo, mean; IC90) | +3.98% [-3.05%, +11.22%] |
| ROI ponderado por stake (robusto por jogo, mean; IC90) | +0.86% [-7.58%, +9.72%] |

**Como ler as 3 linhas de ROI (Back)**

- **ROI realizado (ponderado por stake)**: ΣP&L / Σstake (pode ser dominado por stakes grandes).
- **ROI realizado (robusto por jogo)**: média por jogo (cada jogo pesa parecido; reduz dominância de outliers).
- **ROI ponderado por stake (robusto por jogo)**: calcula ROI ponderado dentro do jogo e depois faz média/IC por jogo.

Observação: a Seção 4 reporta ROI médio no **recorte completo** (betslip confiável). Aqui (7.1) é apenas a coorte **Back edge** (diff>=corte) e o ROI também é mostrado ponderado por stake; por isso sinais podem divergir.

### 7.2 Lay (BS << WS) — risco de cauda
| Métrica | Valor |
|---|---:|
| Corte (diff_pct) | <= -2.0% |
| N eventos | 412 |
| Cobertura finance (na coorte) | 188/412 |
| Stake total (estimado) | 65606.04 |
| Liability total (estimada) | 59170.39 |
| Liability média | 143.62 |
| Liability p95 | 553.14 |
| Liability p99 | 1862.29 |
| ES95 (liability) | 1363.02 |
| Liability max | 4395.35 |
| Proxy de banca (>= p99 liability) | 1862.29 |
| N com ROI realizado | 317 |
| P&L realizado total (estimado) | -6120.34 |
| ROI realizado (ponderado por liability) | -10.91% |
| ROI realizado (ponderado por stake) | -9.83% |
| ROI/liability (robusto por jogo, mean; IC90) | +10.73% [+0.57%, +20.51%] |
| ROI/liability ponderado (robusto por jogo, mean; IC90) | +10.65% [-0.07%, +20.98%] |

### 7.3 Projeção mensal (30 dias fixo) — turnover, lucro e banca
Premissas:
- **Mês = 30 dias fixo**.
- Turnover = soma de stakes (Back e Lay). Para Lay, também reportamos exposição por **liability**.
- **Banca por risco (unitária)** = p99/ES95(exposição por aposta).
- **Banca por liquidez (turnover)** = p95/p99 de capital **simultaneamente travado** (stake/liability em aberto até a liquidação), capturando distribuição desigual de entradas ao longo do tempo.
- Projeção de lucro usa duas visões: (i) **lucro/dia observado** e (ii) **ROI observado × turnover projetado**.

| Bloco | Janela(d) | Turnover 30d | Lucro 30d (direto) | Lucro 30d (ROI×turnover) |
|---|---:|---:|---:|---:|
| Back | 12.0 | 612504.71 | -133784.83 | -137426.25 |
| Lay (stake) | 12.0 | 164015.11 | -15300.85 | -16118.64 |
| Total (Back+Lay) | 12.0 | 776519.81 | -149085.68 | -153544.89 |

**Banca por risco (exposição unitária)**

| Bloco | Banca conservadora (p99) | Banca agressiva (ES95) | ROI/banca 30d (direto) | ROI/banca 30d (ROI×turnover) |
|---|---:|---:|---:|---:|
| Back (stake) | 4393.89 | 3348.68 | -3044.79% | -3127.67% |
| Lay (liability) | 1862.29 | 1363.02 | -821.62% | -865.53% |
| Total (soma) | 6256.18 | 4711.70 | -2383.02% | -2454.29% |

**Banca por liquidez (capital simultaneamente travado)**

Definição operacional: cada aposta trava capital de `audited_at` até `kickoff + 2.00h + 2.25h` (grid=5min). A banca recomendada aplica buffer de +10%.

| Bloco | Liquidez mean | Liquidez p95 | Liquidez p99 | Liquidez max | Banca liq p99 (+buffer) |
|---|---:|---:|---:|---:|---:|
| Back (stake) | 12155.95 | 48371.23 | 79713.03 | 92960.76 | 87684.33 |
| Lay (liability) | 4078.37 | 9322.68 | 12694.59 | 14312.45 | 13964.05 |
| Total (Back+Lay) | 16221.76 | 54734.56 | 89604.95 | 103549.09 | 98565.45 |

**Banca recomendada (conservadora)**

| Métrica | Valor |
|---|---:|
| Banca por risco (p99 unitário, soma) | 6256.18 |
| Banca por liquidez (p99 simultâneo + buffer) | 98565.45 |
| Banca efetiva (max das duas) | 98565.45 |
| ROI/banca 30d (direto, banca efetiva) | -151.26% |
| ROI/banca 30d (ROI×turnover, banca efetiva) | -155.78% |
**Diagnóstico de cobertura (placar/ROI)**

| Bloco | Turnover total | Turnover com ROI | Cobertura turnover |
|---|---:|---:|---:|
| Back | 245001.88 | 238510.00 | 97.35% |
| Lay | 65606.04 | 62277.47 | 94.93% |

Notas (Lay): exposição 30d por liability (não é turnover) = 147925.96; ROI realizado por liability (ponderado) = -10.91%.

---
## 8) Curva temporal (pico, reversão e melhor timing)
Esta seção usa séries temporais coletadas em pontos discretos (t≈0,3,6,10,15,20s). Fontes possíveis:

- **BS-temporal (legado)**: `hypothesis_details.temporal` (Back) e `hypothesis_details.lay_temporal` (Lay)
- **WS-temporal (novo)**: `hypothesis_details.ws_series` (todos os t’s via WebSocket)

Para manter comparabilidade, nesta seção `diff_pct(t)` é sempre calculado contra o **WS do t0** (`ws_odd`): `(odd_t - ws_t0)/ws_t0*100`.

O objetivo é responder: **tempo até o pico/vale**, **% que segue melhorando até t_max**, **% com reversão** e **impacto esperado por timing**. **CLV é reportado somente pre-match** (closing pré-jogo).

### 8.1 Back (pico em diff_pct)
Definições: **pico** = `max(diff_pct)`. **reversão** = após o pico, `diff_pct` cair pelo menos 0.50 p.p. (para Lay, subir 0.50 p.p. após o vale). `t_reversão` é o 1º tempo que cruza esse limiar (logo, não é o pico).

| Regime | N | t_pico médio (s) | t_pico p50 (s) | % melhora até fim | % com reversão | t_reversão médio (s) | Δt (rev - pico) médio (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRE_MATCH | 2328 | 2.5 | 0.0 | 76.2% | 16.5% | 11.8 | 8.4 |
| IN_MATCH | 1853 | 7.1 | 0.0 | 42.9% | 44.4% | 12.9 | 8.6 |

**Partição 100% (Back)**: categorias exclusivas (somam 100% por regime).

| Regime | % pico no fim, sem reversão | % pico no fim, com reversão (recuperou) | % pico antes, com reversão | % pico antes, sem reversão relevante |
|---|---:|---:|---:|---:|
| PRE_MATCH | 79.1% | 5.8% | 10.7% | 4.4% |
| IN_MATCH | 50.8% | 5.0% | 39.4% | 4.8% |

**Curva média (Back)**: média de `diff_pct` e `odd` por tempo. `CLV` é reportado **somente pre-match** (closing pré-jogo). `ROI` (quando aparece) é o **ROI realizado** se a aposta fosse feita naquele ponto.

| Tempo | N pts | diff_pct médio | odd média | CLV médio (pre-match) | ROI médio (se houver) |
|---|---:|---:|---:|---:|---:|
| t+0s | 4181 | +13.18% | 2.240 | +3.48% | 16.27 |
| t+3s | 1181 | +0.00% | 2.047 | +0.16% | 2.93 |
| t+6s | 4596 | +11.79% | 2.212 | +3.01% | 14.75 |
| t+10s | 6869 | +16.47% | 2.287 | +3.77% | 16.18 |
| t+15s | 4155 | +13.10% | 2.242 | +3.62% | 14.03 |
| t+20s | 9781 | +7.70% | 2.184 | +2.53% | 13.34 |

### 8.1b Back — impacto por timing (t0 vs pico vs último) e reversão
Leitura: se a estratégia é **entrar no pico**, compare CLV/ROI no pico vs t0 e veja diferença entre **casos com reversão** vs **sem reversão**. Aqui reportamos **média robusta por jogo + IC90**. Observação: **CLV só é calculado pre-match**.

**CLV (somente pre-match) — média robusta por jogo (IC90)**

| Subcoorte | N total | N CLV (PM) | CLV t0 (mean; IC90) | CLV pico (mean; IC90) | CLV último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 2974 | 1741 | +3.44% [+2.95%, +3.95%] | +3.74% [+3.24%, +4.24%] | +3.72% [+3.22%, +4.23%] |
| COM_REVERSAO | 1207 | 329 | +4.22% [+3.51%, +4.95%] | +5.26% [+4.53%, +6.00%] | +4.03% [+3.34%, +4.73%] |

**ROI (stake=1) — média robusta por jogo (IC90)**

| Subcoorte | N total | N ROI | ROI t0 (mean; IC90) | ROI pico (mean; IC90) | ROI último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 2974 | 2728 | +1.55% [-3.04%, +6.26%] | +1.22% [-3.43%, +5.72%] | +1.19% [-3.46%, +5.69%] |
| COM_REVERSAO | 1207 | 1067 | +5.36% [-1.27%, +12.16%] | +7.55% [+0.79%, +14.52%] | +4.11% [-2.25%, +10.62%] |

Nota interpretativa: a subcoorte **COM_REVERSAO** pode ter CLV maior no **pico** porque, por definição, ela inclui casos em que o edge atingiu um extremo mais alto antes de reverter. Para entender a hipótese 'sem reversão deveria ser maior', compare também a categoria '**pico no fim, sem reversão**' na partição 100% (tabela 8.1).

**Decomposição (pre-match): entry_odd vs closing_odd (IC90)**

| Subcoorte | N CLV (PM) | entry_odd t0 (mean; IC90) | entry_odd pico (mean; IC90) | closing_odd (mean; IC90) |
|---|---:|---:|---:|---:|
| SEM_REVERSAO | 1747 | 2.018 [+2.006, +2.030] | 2.025 [+2.013, +2.037] | 1.959 [+1.953, +1.965] |
| COM_REVERSAO | 329 | 2.048 [+2.033, +2.063] | 2.071 [+2.056, +2.087] | 1.964 [+1.955, +1.974] |

### 8.2 Lay (vale em diff_pct)
| Regime | N | t_vale médio (s) | t_vale p50 (s) | % melhora até fim | % com reversão | t_reversão médio (s) | Δt (rev - vale) médio (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRE_MATCH | 1738 | 2.5 | 0.0 | 66.7% | 25.3% | 11.3 | 7.5 |
| IN_MATCH | 1031 | 6.5 | 0.0 | 38.2% | 49.8% | 13.2 | 8.5 |

**Partição 100% (Lay)**: categorias exclusivas (somam 100% por regime).

| Regime | % vale no fim, sem reversão | % vale no fim, com reversão (recuperou) | % vale antes, com reversão | % vale antes, sem reversão relevante |
|---|---:|---:|---:|---:|
| PRE_MATCH | 70.3% | 11.6% | 13.8% | 4.4% |
| IN_MATCH | 46.2% | 7.4% | 42.4% | 4.1% |

**Curva média (Lay)**: usa `odd=lay_odd`. `CLV` é reportado **somente pre-match** (closing pré-jogo). O ROI mostrado aqui é **ROI por liability** (não por stake), porque Lay é governado por risco.

| Tempo | N pts | diff_pct médio | lay_odd média | CLV médio (pre-match) | ROI/liability médio (se houver) |
|---|---:|---:|---:|---:|---:|
| t+0s | 2769 | +16.10% | 2.287 | +1.86% | 4.87 |
| t+3s | 13 | -4.00% | 1.804 | — | 297.73 |
| t+6s | 3187 | +14.88% | 2.264 | +2.14% | 13.31 |
| t+10s | 4101 | +15.43% | 2.283 | +1.36% | 15.67 |
| t+15s | 2761 | +12.12% | 2.216 | +2.11% | 8.05 |
| t+20s | 3681 | +17.63% | 2.353 | +1.50% | 13.45 |

### 8.2b Lay — impacto por timing (t0 vs vale vs último) e reversão
Leitura: se a estratégia é **entrar no vale (odd mais baixa)**, compare CLV/ROI no vale vs t0 e veja diferença entre **casos com reversão** vs **sem reversão**. Aqui reportamos **média robusta por jogo + IC90**. Observação: **CLV só é calculado pre-match**.

**CLV (somente pre-match) — média robusta por jogo (IC90)**

| Subcoorte | N total | N CLV (PM) | CLV t0 (mean; IC90) | CLV vale (mean; IC90) | CLV último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 1816 | 1158 | +2.83% [+2.31%, +3.36%] | +2.42% [+1.91%, +2.94%] | +2.44% [+1.93%, +2.95%] |
| COM_REVERSAO | 953 | 395 | +3.16% [+2.36%, +4.02%] | +1.80% [+1.09%, +2.53%] | +2.74% [+2.03%, +3.48%] |

**ROI/liability — média robusta por jogo (IC90)**

| Subcoorte | N total | N ROI | ROI/liab t0 (mean; IC90) | ROI/liab vale (mean; IC90) | ROI/liab último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 1816 | 1634 | +3.24% [-7.17%, +16.77%] | +8.66% [-4.28%, +24.79%] | +8.62% [-4.31%, +24.74%] |
| COM_REVERSAO | 953 | 833 | +8.96% [+0.90%, +16.90%] | +15.58% [+5.36%, +25.93%] | +8.83% [+0.73%, +17.07%] |

---
### 8.3 Resumo de estratégias — combinações (Side × Pre/In × Reversal)
Esta tabela resume as combinações possíveis. Observação importante:

- **Back**: a estratégia é **entrar rápido em `t0`**, então **não faz sentido separar por Reversal(Sim/Não)** (agregamos como `Any`).
- **Lay**: entrada **após reversão** quando ela existe (`odd_reversal`), senão no **último ponto** (~t+20s).
- **CLV** aqui é **somente pre‑match** (closing pré‑jogo). Para **Lay**, usamos a convenção unificada `clv_conv = -(entry - closing)/closing`, logo **Lay “bom” tende a CLV_CONV > 0**.
- **ROI** é calculado no **ponto de entrada da estratégia** (se houver placar). Para Lay, ROI é **por liability**.
- **IC90** é bootstrap por jogo (cluster em `match_id`). Para critério “p30” usamos o quantil bootstrap **p30** do estimador.

| Side | Pre/In | Reversal | N | Jogos | CLV t0 (mean; IC90) | ROI (mean; IC90) | ROI p30 | Ativa? (critério) |
|---|---|---|---:|---:|---:|---:|---:|---|
| Back | Pre | Any | 2328 | 378 | +3.82% [+3.32%, +4.33%] | +2.58% [-2.45%, +7.55%] | +1.00% | sim (CLV p90>0 AND ROI>0) |
| Back | In | Any | 1853 | 327 | — | +1.24% [-3.48%, +6.07%] | -0.17% | não (ROI p30>0) |
| Lay | Pre | Yes | 440 | 207 | -3.92% [-4.70%, -3.15%] | -0.71% [-10.01%, +8.68%] | -3.62% | não (CLV_CONV p90>0 AND ROI p30>0) |
| Lay | Pre | No | 1298 | 302 | -2.44% [-2.95%, -1.93%] | -5.22% [-11.08%, +0.82%] | -7.21% | não (CLV_CONV p90>0 AND ROI p30>0) |
| Lay | In | Yes | 513 | 172 | — | +21.14% [+7.23%, +36.62%] | +16.06% | sim (ROI p30>0) |
| Lay | In | No | 518 | 189 | — | +25.54% [-3.31%, +60.08%] | +14.19% | sim (ROI p30>0) |

Notas:
- Se você quiser operar **cada combinação como uma estratégia separada**, o sizing (Kelly/caps) deve ser estimado e governado separadamente por estratégia.
- Esta tabela é **in-sample** na janela (`--lookback-days`). A seção OOS (walk-forward) pode ser habilitada com `--walkforward`.

## 9) Combinações de valor (regime × linha AH × lag)
Tabela rankeada por volume (N). Métrica: diff_pct (OK, betslip confiável).

### 9.1 Back combos (diff_pct >= corte)
| Regime | Linha | Lag | N | Jogos | Mean diff % | IC90 cluster |
|---|---|---|---:|---:|---:|---|
| PRE_MATCH | AH 0-1 (líquida) | < 10s | 216 | 92 | +6.74% | [+6.50%, +7.09%] |
| IN_MATCH | AH 2+ (extrema) | < 10s | 156 | 115 | +5.86% | [+5.52%, +6.27%] |
| PRE_MATCH | AH 2+ (extrema) | < 10s | 126 | 62 | +6.32% | [+5.60%, +6.46%] |
| PRE_MATCH | AH 1-2 (média) | < 10s | 125 | 52 | +6.14% | [+5.65%, +6.46%] |
| IN_MATCH | AH 0-1 (líquida) | < 10s | 59 | 51 | +6.33% | [+5.71%, +6.78%] |
| IN_MATCH | AH 1-2 (média) | < 10s | 36 | 35 | +6.38% | [+5.60%, +7.04%] |
| PRE_MATCH | AH 0-1 (líquida) | 20-30s | 29 | 25 | +7.05% | [+6.30%, +7.47%] |
| PRE_MATCH | AH 0-1 (líquida) | > 30s | 25 | 23 | +6.60% | [+6.00%, +7.40%] |
| IN_MATCH | AH 2+ (extrema) | > 30s | 22 | 16 | +6.53% | [+5.19%, +7.41%] |
| PRE_MATCH | AH 2+ (extrema) | 20-30s | 18 | 13 | +7.23% | [+6.41%, +7.93%] |
| PRE_MATCH | AH 1-2 (média) | 20-30s | 16 | 12 | +6.30% | [+5.52%, +7.23%] |
| PRE_MATCH | AH 1-2 (média) | > 30s | 14 | 13 | +5.73% | [+4.92%, +6.90%] |

### 9.2 Lay combos (diff_pct <= corte) + risco
| Regime | Linha | Lag | N | Jogos | Mean diff % | IC90 cluster | Liability p95 |
|---|---|---|---:|---:|---:|---|---:|
| IN_MATCH | AH 2+ (extrema) | < 10s | 121 | 98 | -5.00% | [-5.38%, -4.61%] | 430.34 |
| PRE_MATCH | AH 0-1 (líquida) | < 10s | 55 | 43 | -4.73% | [-5.47%, -4.29%] | 1051.93 |
| PRE_MATCH | AH 2+ (extrema) | < 10s | 55 | 48 | -4.65% | [-5.13%, -4.21%] | 561.98 |
| IN_MATCH | AH 0-1 (líquida) | < 10s | 49 | 42 | -5.07% | [-5.67%, -4.65%] | 572.86 |
| PRE_MATCH | AH 1-2 (média) | < 10s | 26 | 21 | -4.57% | [-5.54%, -3.68%] | 106.17 |
| IN_MATCH | AH 1-2 (média) | < 10s | 25 | 23 | -4.78% | [-5.53%, -4.07%] | 673.14 |
| IN_MATCH | AH 2+ (extrema) | > 30s | 15 | 13 | -4.60% | [-5.65%, -3.72%] | 1438.89 |
| PRE_MATCH | AH 0-1 (líquida) | > 30s | 8 | 8 | -3.76% | [-4.54%, -3.02%] | 123.01 |
| IN_MATCH | AH 2+ (extrema) | 20-30s | 8 | 8 | -3.92% | [-5.03%, -2.84%] | 527.66 |
| PRE_MATCH | AH 1-2 (média) | > 30s | 7 | 6 | -2.96% | [-3.89%, -2.30%] | 183.66 |
| PRE_MATCH | AH 0-1 (líquida) | 10-20s | 6 | 6 | -4.76% | [-6.61%, -3.15%] | 91.77 |
| PRE_MATCH | AH 2+ (extrema) | 20-30s | 6 | 6 | -4.06% | [-5.44%, -2.70%] | 261.62 |

---
### 9.3 Stake sizing — teoria mínima + calibração empírica
Objetivo: explicar por que **ROI por aposta** pode divergir de **ROI ponderado por stake/liability**, e propor uma política de staking que seja (i) coerente com edge/CLV e (ii) controlada por risco (p99/ES).

**Teoria (resumo prático)**

- **Flat stake**: cada aposta pesa igual. Boa baseline para checar se o sizing atual está piorando resultado.
- **Proporcional ao limite**: útil operacionalmente (capacidade), mas **não é** sizing por edge.
- **Kelly fracionado**: sizing por edge. Para Back, \(f^* \propto \frac{EV}{odds-1}\). Para Lay, o sizing natural é por **liability**.
- **Governança de risco**: impor **cap por aposta** (ex.: 1–2% da banca) e olhar p95/p99/ES95 de exposição.

**Como o Kelly está sendo calculado aqui (detalhado, com premissas)**

Como ainda não temos um modelo explícito de probabilidade \(p\) por aposta, usamos um proxy padrão: **o closing pré‑jogo como melhor estimativa de preço justo**. A partir disso inferimos \(p\) e aplicamos Kelly como aproximação.

Premissas e entradas:

- **Entrada (Back)**: `entry_odd = bs_odd` (odd do betslip no momento de execução).
- **Entrada (Lay)**: `entry_lay_odd = hypothesis_details.lay.odd` (fallback: `bs_odd`).
- **Preço justo (pre‑match)**: `closing_odd` (closing line). Inferimos \(p \approx 1/closing\_odd\).
- **Aplicabilidade**: para `is_live=True` (in‑match), **não usamos** `closing_odd` como benchmark de CLV/Kelly.

Fórmulas (Back):

- Odds decimais \(O\); retorno líquido \(b = O-1\).
- \(p \approx 1/closing\_odd\).
- Valor esperado por unidade de stake: \(EV = O\cdot p - 1\).
- Kelly cheio (fração de banca em **stake**): \(f^* = \frac{EV}{b} = \frac{O\cdot p - 1}{O-1}\).
- No relatório: \(f = \max(0,f^*)\cdot \text{frac}\) com `frac` em {0.10, 0.25, 0.50, 1.00}.

Fórmulas (Lay):

- Para Lay, o “capital em risco” natural é a **liability** \(L\) (perda máxima), não o stake.
- Usamos \(p \approx 1/closing\_odd\) e \(o = entry\_lay\_odd\).
- Kelly em termos de **liability** (proxy): \(f^*_{liab} = 1 - p\cdot o\).
- No relatório: \(f_{liab} = \max(0,f^*_{liab})\cdot \text{frac}\).
- Conversão para stake (apenas para turnover): \(stake = L/(o-1)\).

Derivação rápida (por que \(f^*_{liab}=1-p\cdot o\)):

- Defina \(W\) como banca e escolha alocar \(L=f\cdot W\) como **liability**.
- Se o evento acontece (prob. \(p\)), você perde \(L\): \(W' = W-L = W(1-f)\).
- Se o evento não acontece (prob. \(1-p\)), você ganha o **stake** do Lay, que é \(S=L/(o-1)\): \(W' = W+S = W\left(1+\frac{f}{o-1}\right)\).
- Kelly maximiza \(p\log(1-f) + (1-p)\log\left(1+\frac{f}{o-1}\right)\). Derivando e igualando a zero, obtém-se \(f^* = 1 - p\cdot o\).

Parâmetros de escala (proxy de banca) e caps:

- Por padrão: `back_bank_ref = p99(stake)` e `lay_bank_ref = p99(liability)` observados no sizing **PROXY** da janela.
- Opcional: com `--kelly-bankroll`, usamos `bank_ref = bankroll` para simular capacidade com banca explícita.
- `stake_back = min(f * back_bank_ref, cap_back, cap_evento_limit)`.
- `liab_lay = min(f_liab * lay_bank_ref, cap_lay, cap_evento_limit)`.
- Caps atuais (guardrail): `cap_back = 2.0% * ref`, `cap_lay = 1.0% * ref`. Cap por evento: `max_stake = 100% * limit`.
- **Implicação importante**: se o cap estiver frequentemente ativo, aumentar `frac` (ex.: >0,25×Kelly) **não aumenta** tamanho real — a curva satura.

Limitações: comissão/vigorish não modelados; correlação entre apostas ignorada; closing como preço justo é aproximação; e o `bank_ref` é uma escala interna (proxy) baseada em limits observados.

**Diagnóstico: exposição vs performance (correlação de Pearson; indicativo, não causal)**

- **Back (stake)**: corr(exposição, ROI)=-0.079; corr(exposição, CLV)=0.015 (onde CLV existe).
- **Lay (liability)**: corr(exposição, ROI)=0.049; corr(exposição, CLV)=0.043 (onde CLV existe).

**Backtest de sizing (apenas eventos com placar; valores em unidade monetária *proxy*)**

| Lado | Scheme | N | Turnover (janela) | Lucro (janela) | ROI/turnover | p99 exposição | ES95 exposição | DD 30d (média) | DD 30d (p95) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Back | FLAT | 835 | 835.00 | 12.23 | 1.47% | 1.00 | 1.00 | 18.75 | 36.16 |
| Lay | FLAT | 373 | 444.44 | 10.65 | 2.40% | 1.00 | 1.00 | 9.66 | 19.24 |
| Back | PROXY | 835 | 238510.00 | -53513.93 | -22.44% | 4394.27 | 3564.06 | 102740.09 | 155907.44 |
| Lay | PROXY | 317 | 62277.47 | -6120.34 | -9.83% | 2061.06 | 1612.25 | 16871.41 | 28396.29 |
| Back | KELLY_0.10 | 436 | 22078.40 | 36.12 | 0.16% | 134.66 | 118.32 | 1811.95 | 3487.05 |
| Lay | KELLY_0.10 | 96 | 4524.36 | 739.40 | 16.34% | 100.00 | 100.00 | 35.64 | 62.97 |
| Back | KELLY_0.25 | 436 | 42474.17 | -436.40 | -1.03% | 200.00 | 200.00 | 3594.94 | 6723.95 |
| Lay | KELLY_0.25 | 96 | 6257.74 | 1130.88 | 18.07% | 100.00 | 100.00 | 98.25 | 168.98 |
| Back | KELLY_0.50 | 436 | 51569.32 | -1905.46 | -3.69% | 200.00 | 200.00 | 6181.02 | 10837.02 |
| Lay | KELLY_0.50 | 96 | 7074.23 | 1185.14 | 16.75% | 100.00 | 100.00 | 111.70 | 199.69 |
| Back | KELLY_1.00 | 436 | 53696.58 | -2143.52 | -3.99% | 200.00 | 200.00 | 6734.91 | 11682.79 |
| Lay | KELLY_1.00 | 96 | 7561.08 | 1350.30 | 17.86% | 100.00 | 100.00 | 123.85 | 221.14 |

Leitura:
- Se `PROXY` piora ROI/turnover vs `FLAT`, isso indica que a política de stake atual está concentrando exposição em pontos com pior performance.
- `KELLY_0.25` tende a ser um bom compromisso quando o edge é estimado por CLV, mas requer **caps** e só é aplicável quando há `closing_odd` (pre‑match).
- Em Lay, é comum observar ROI alto por **liability**, mas sizing menor em **stake**: isso é uma decisão deliberada de governança de risco (liability tem cauda pior).
- DD é estimado por bootstrap i.i.d de dias (aproximação). Para uma curva mais fiel, use bootstrap por dia com blocos maiores.

### 9.3b Stake sizing por estratégia (8 combinações)
Abaixo repetimos o backtest de sizing **separado** por cada combinação `Side × Pre/In × Reversal`. Isso responde diretamente sua necessidade: **se várias combinações tiverem valor, o Kelly/caps deve ser calibrado por estratégia**.

Observações:
- Kelly é calculado **somente pre-match** (depende de `closing_odd`). Em combinações `In`, reportamos apenas `FLAT` e `PROXY`.
- ROI do Lay é por **liability**; turnover é mostrado em stake equivalente.

| Side | Pre/In | Reversal | Scheme | N (placar) | Turnover | Lucro | ROI/turnover | p99 exp | DD30 p95 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Back | Pre | Yes | FLAT | 107 | 107.00 | -3.85 | -3.60% | 1.00 | 47.69 |
| Back | Pre | Yes | PROXY | 107 | 32365.05 | -9700.30 | -29.97% | 4391.80 | 55009.01 |
| Back | Pre | Yes | KELLY_0.10 | 92 | 5241.88 | -307.50 | -5.87% | 105.05 | 2431.40 |
| Back | Pre | Yes | KELLY_0.25 | 92 | 10676.22 | -469.67 | -4.40% | 200.00 | 4100.49 |
| Back | Pre | Yes | KELLY_0.50 | 92 | 12952.86 | -695.37 | -5.37% | 200.00 | 5343.04 |
| Back | Pre | Yes | KELLY_1.00 | 92 | 13278.26 | -806.52 | -6.07% | 200.00 | 5583.23 |
| Back | Pre | No | FLAT | 313 | 313.00 | 0.81 | 0.26% | 1.00 | 50.93 |
| Back | Pre | No | PROXY | 313 | 118871.49 | -21646.75 | -18.21% | 4414.02 | 104096.04 |
| Back | Pre | No | KELLY_0.10 | 267 | 13133.23 | -121.85 | -0.93% | 131.41 | 2904.90 |
| Back | Pre | No | KELLY_0.25 | 267 | 24839.28 | -1028.03 | -4.14% | 200.00 | 7013.94 |
| Back | Pre | No | KELLY_0.50 | 267 | 30451.96 | -2115.71 | -6.95% | 200.00 | 10903.91 |
| Back | Pre | No | KELLY_1.00 | 267 | 31708.85 | -1924.39 | -6.07% | 200.00 | 10886.08 |
| Back | In | Yes | FLAT | 61 | 61.00 | -3.48 | -5.70% | 1.00 | 41.52 |
| Back | In | Yes | PROXY | 61 | 12310.03 | -4788.07 | -38.90% | 2280.55 | 38200.42 |
| Back | In | No | FLAT | 59 | 59.00 | 13.58 | 23.03% | 1.00 | 0.00 |
| Back | In | No | PROXY | 59 | 12716.15 | -4078.71 | -32.08% | 1395.93 | 33875.17 |
| Lay | Pre | Yes | FLAT | 19 | 19.84 | -0.66 | -3.35% | 1.00 | 20.52 |
| Lay | Pre | Yes | PROXY | 19 | 7349.71 | -4833.23 | -65.76% | 3697.04 | 32442.19 |
| Lay | Pre | Yes | KELLY_0.10 | 10 | 353.78 | -51.42 | -14.54% | 84.87 | 792.19 |
| Lay | Pre | Yes | KELLY_0.25 | 10 | 567.75 | -28.56 | -5.03% | 100.00 | 786.54 |
| Lay | Pre | Yes | KELLY_0.50 | 10 | 673.43 | -40.17 | -5.97% | 100.00 | 811.86 |
| Lay | Pre | Yes | KELLY_1.00 | 10 | 673.43 | -40.17 | -5.97% | 100.00 | 813.62 |
| Lay | Pre | No | FLAT | 55 | 61.98 | -4.72 | -7.61% | 1.00 | 30.29 |
| Lay | Pre | No | PROXY | 55 | 14254.51 | -7098.28 | -49.80% | 3223.73 | 31061.76 |
| Lay | Pre | No | KELLY_0.10 | 30 | 1647.21 | -105.93 | -6.43% | 100.00 | 1029.64 |
| Lay | Pre | No | KELLY_0.25 | 30 | 2131.42 | 37.91 | 1.78% | 100.00 | 864.29 |
| Lay | Pre | No | KELLY_0.50 | 30 | 2268.31 | -5.14 | -0.23% | 100.00 | 1001.12 |
| Lay | Pre | No | KELLY_1.00 | 30 | 2277.41 | -13.12 | -0.58% | 100.00 | 980.23 |
| Lay | In | Yes | FLAT | 31 | 36.32 | 4.39 | 12.10% | 1.00 | 4.00 |
| Lay | In | Yes | PROXY | 31 | 6871.71 | 2439.81 | 35.51% | 1638.92 | 2450.93 |
| Lay | In | No | FLAT | 46 | 70.99 | 0.95 | 1.35% | 1.00 | 19.79 |
| Lay | In | No | PROXY | 46 | 9275.51 | -2462.19 | -26.55% | 1045.83 | 19711.59 |
### 9.4 Estratégias candidatas (combinações 8.3 + sizing recomendado)
Esta seção foi atualizada para refletir as **combinações** que você está analisando (Back/Lay × Pre/In × Reversal). Ela não assume mais apenas `BackFast` e `LayReversal`.

**Política de entrada**:
- Back: `t0`.
- Lay: **após reversão** quando existir; senão no **último ponto** (~t+20s).

**Política de sizing sugerida** (padrão):
- Pre‑match: `KELLY_0.25` (com caps e cap por evento).
- In‑match: `FLAT` ou `PROXY` capado, até existir um benchmark live (Kelly live não é confiável sem referência).

| Side | Pre/In | Reversal | N (janela) | Jogos | CLV (entry; IC90) | ROI (entry; IC90) | ROI p30 | Observação |
|---|---|---|---:|---:|---:|---:|---:|---|
| Back | Pre | Yes | 384 | 210 | +4.22% [+3.51%, +4.95%] | +4.97% [-4.68%, +14.52%] | +1.93% | pre: Kelly OK |
| Back | Pre | No | 1944 | 355 | +3.44% [+2.95%, +3.95%] | +3.58% [-1.80%, +8.86%] | +2.27% | pre: Kelly OK |
| Back | In | Yes | 823 | 269 | — — | +6.53% [-1.70%, +15.12%] | +18.67% | in: use FLAT/PROXY |
| Back | In | No | 1030 | 288 | — — | -1.43% [-8.44%, +5.59%] | +32.80% | in: use FLAT/PROXY |
| Lay | Pre | Yes | 440 | 207 | -3.92% [-4.70%, -3.15%] | -0.71% [-10.01%, +8.68%] | -3.62% | pre: Kelly OK |
| Lay | Pre | No | 1298 | 302 | -2.44% [-2.95%, -1.93%] | -5.22% [-11.08%, +0.82%] | -7.21% | pre: Kelly OK |
| Lay | In | Yes | 513 | 172 | — — | +21.14% [+7.23%, +36.62%] | +16.06% | in: use FLAT/PROXY |
| Lay | In | No | 518 | 189 | — — | +25.54% [-3.31%, +60.08%] | +38.41% | in: use FLAT/PROXY |
| Estratégia | Scheme | N Back (janela) | N Lay (janela) | N Back 30d (proj.) | N Lay 30d (proj.) | Stake médio Back | Stake médio Lay | Liability média Lay | Turnover 30d (proj.) | Lucro 30d (proj.) | ROI/turnover (janela) | ROI Lay/liability (janela) | ROI Lay/turnover (janela) | Banca p99 (Back) | Banca p99 (Lay) | Banca risco p99 (soma) | Banca liquidez p99 (+buf) | Banca recomendada (max) | ROI/banca 30d | Turnover 30d (R$) | Lucro 30d (R$) | Banca rec. (R$) | DD 30d p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ativas (PRE, critérios 8.3) | FLAT | 420 | 0 | 836 | 0 | 1.00 | — | — | 836.47 | -6.07 | -0.73% | —% | —% | 1.00 | — | 1.00 | 235.06 | 235.06 | -2.58% | 4349.66 | -31.56 | 1222.31 | 63.27 |
| Ativas (PRE, critérios 8.3) | KELLY_0.25 | 359 | 0 | 715 | 0 | 98.93 | — | — | 70732.81 | -2982.82 | -4.22% | —% | —% | 200.00 | — | 200.00 | 20212.50 | 20212.50 | -14.76% | 367810.61 | -15510.68 | 105104.98 | 8446.00 |

Notas:
- **N Back/Lay** na tabela é **na janela observada**. As colunas `N 30d (proj.)` são uma **escala linear** por dias observados.
- Esta linha usa a união das **combinações pre‑match ativas** sob os critérios da 8.3 (é um resumo, não substitui a tabela 8.3).
- `Stake médio` e `Liability média` são **proxies** na unidade monetária do seu limit/finance; valores em **R$** usam fx `--fx-usdbrl`.
- `Banca recomendada` = max( banca por risco p99 (unitária, soma) ; banca por liquidez p99 (+buffer) ).
- **Por que Lay pode ter stake médio menor mesmo com ROI maior**: (i) Lay é governado por **liability** (risco) e usamos cap mais conservador; (ii) stake em Lay é derivado de `liability/(odd-1)` e depende do nível médio de odds; (iii) Kelly depende do **edge vs closing** (gap entry→closing), não do ROI observado isoladamente.
- **Por que não incluímos Back in‑match aqui**: Kelly/CLV usam `closing_odd` pré‑jogo; in‑match requer benchmark diferente (ex.: referência por minuto/VWAP) e governança operacional específica. A seção 2.2 já compara PRE_MATCH vs IN_MATCH.

### 9.4b Curva de capacidade — frações de Kelly e tamanho potencial
Esta tabela é um exercício para estimar **capacidade** (turnover/lucro/risco) ao variar a fração de Kelly. Ela deixa explícito quando o sizing satura por **cap por aposta**.

**Escala Kelly usada nesta curva**: BANKROLL | ref_back=10000.00 ref_lay=10000.00 | cap_back=2.0% cap_lay=1.0% | max_stake_event=100%*limit

| Strategy | Scheme | cap_hit Back (%) | limit_hit Back (%) | cap_hit Lay (%) | limit_hit Lay (%) | Turnover 30d (proj.) | Lucro 30d (proj.) | ROI/banca 30d | DD 30d p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ativas (PRE, critérios 8.3) | KELLY_0.10 | 0.0% | 29.4% | —% | —% | 36595.94 | -855.09 | -8.29% | 3708.84 |
| Ativas (PRE, critérios 8.3) | KELLY_0.25 | 22.5% | 44.7% | —% | —% | 70732.81 | -2982.82 | -14.76% | 8601.95 |
| Ativas (PRE, critérios 8.3) | KELLY_0.50 | 77.5% | 46.9% | —% | —% | 86445.20 | -5598.55 | -22.56% | 12858.80 |
| Ativas (PRE, critérios 8.3) | KELLY_1.00 | 86.9% | 48.1% | —% | —% | 89596.51 | -5438.90 | -21.30% | 12782.79 |

Leitura rápida:
- Se `cap_hit` estiver alto, a alavancagem adicional (ex.: 0,50× ou 1,00×) não vira turnover — o cap está “travando” o tamanho.
- Se `limit_hit` estiver alto, você está batendo no **limit operacional** (capacidade da conta/mercado) — aumentar banca ou frac não aumenta turnover.
- Se `cap_hit` estiver baixo, a curva deve escalar quase linearmente com `frac` (até bater em limites reais/operacionais).
- Lay normalmente satura antes por ter cap mais conservador (liability) e por ter risco de cauda mais assimétrico.

### 9.4c Diagnóstico: Back in‑match (por que não está na estratégia candidata)
Aqui reportamos Back in‑match **apenas como diagnóstico**. Não incluímos in‑match na estratégia candidata porque (i) `closing_odd` pré‑jogo não é benchmark in‑match e (ii) o sizing Kelly acima depende desse benchmark.

| Regime | Scheme | N | Turnover (janela) | Lucro (janela) | ROI/turnover |
|---|---|---:|---:|---:|---:|
| IN_MATCH BackFast (<5s) | FLAT | 171 | 171.00 | 13.46 | 7.87% |
| IN_MATCH BackFast (<5s) | PROXY | 171 | 32612.75 | -2731.00 | -8.37% |

Próximo passo (se quiser operar in‑match): definir um benchmark de preço justo in‑match (ex.: referência por minuto, VWAP, ou odds externas) e calibrar sizing/risco específico do live.

## 12) OOS walk-forward (expanding window): seleção e validação
Até aqui o relatório é **in-sample** (na janela `--lookback-days`). Este bloco (opcional) faz um walk-forward por dia:

- **Train mode**: `expanding`.
- Em cada passo, usamos uma janela de treino para **selecionar** combinações (Side×Pre/In×Reversal) com evidência de valor.
  - `rolling`: usa os **últimos** `wf_train_days`.
  - `expanding`: usa **todos os dias anteriores** (com `wf_train_days` só definindo quando o teste começa).
- No(s) dia(s) seguinte(s) (`wf_test_days`), medimos o resultado OOS nas combinações ativas.

**Evidência de valor (por combinação, no treino)** (atualizado para dar mais peso a ROI):
- Elegibilidade (todas): `N_ROI >= wf_min_matches` (jogos com ROI na janela de treino). Se `wf_min_matches=0`, o mínimo fica desabilitado.
- **Regra de bloqueio**: se `ROI` for **significativamente negativo** (IC90 inteiro < 0), **não ativa**.
- Se `ROI` for **significativamente positivo** (IC90 inteiro > 0), **ativa**.
- Caso `ROI` seja **>0 mas não sig.**:
  - Pre-match: ativa apenas se `CLV > 0` (não precisa ser sig.)
  - In-match: ativa se `ROI > 0` (não precisa ser sig.; CLV não é aplicável)

- **Step do WF**: `wf_step_days=2`. Se `wf_test_days>1` e `wf_step_days=1`, os test windows ficam **sobrepostos**; nesse caso, os lucros/prejuízos por linha não são somáveis. Para não sobrepor: use `--wf-step-days` igual a `--wf-test-days`.

Isso aproxima o fluxo operacional que você descreveu (seleciona no passo atual e mede no(s) próximo(s) dia(s)).

**Filtro operacional (OOS)**: excluindo exec_bucket apenas no walk-forward (Back=['10-20s']; Lay=—).

### 12.0 Diagnóstico de cobertura OOS (por que N cai)
| Filtro | Jogos únicos |
|---|---:|
| Combinações elegíveis (edge + timing + t0) | 388 |
| Com ROI disponível (precisa de placar) | 351 |
| Com CLV disponível (pre-match + closing) | 184 |

**Calendário do walk-forward (dias únicos)**

| Tipo | Dias |
|---|---:|
| Dias com dados carregados (audited_at) | 12 |
| Dias com eventos OK (qualquer versão, incl. ws-only) | 12 |
| Dias com eventos elegíveis p/ WF (edge) | 11 |
| Dias usados no walk-forward | 12 |

**Diagnóstico por dia (audited_at): betslip vs qualidade vs edge**

| Dia | Auditorias carregadas | Betslip bruto | Betslip conf. | OK (conf.) | Edge Back/Lay | %OK/conf. | Status não-OK dominante |
|---|---:|---:|---:|---:|---:|---:|---|
| 2026-02-08 | 86 | 57 | 55 | 57 | 4/0 | 103.6% | — |
| 2026-02-09 | 244 | 244 | 225 | 244 | 25/0 | 108.4% | — |
| 2026-02-10 | 821 | 810 | 673 | 810 | 76/0 | 120.4% | — |
| 2026-02-11 | 389 | 356 | 247 | 356 | 76/0 | 144.1% | — |
| 2026-02-12 | 105 | 96 | 59 | 96 | 21/0 | 162.7% | — |
| 2026-02-13 | 1025 | 902 | 605 | 902 | 217/56 | 149.1% | — |
| 2026-02-14 | 1219 | 940 | 539 | 940 | 186/32 | 174.4% | — |
| 2026-02-15 | 882 | 761 | 489 | 761 | 151/39 | 155.6% | — |
| 2026-02-16 | 662 | 479 | 329 | 479 | 124/14 | 145.6% | — |
| 2026-02-19 | 946 | 551 | 549 | 851 | 19/26 | 155.0% | — |
| 2026-02-22 | 989 | 0 | 0 | 873 | 33/0 | —% | — |
| 2026-02-23 | 255 | 0 | 0 | 8 | 0/0 | —% | — |

Leitura:
- Se `Auditorias carregadas > 0` mas `Betslip conf.` ≈ 0, geralmente houve **mismatch/parse** (diff fora de [-10,+10]) ou ausência de betslip.
- Se `Betslip conf. > 0` mas `OK (conf.) = 0`, o robô coletou betslip, mas os eventos falharam por **status != OK** (ver coluna de status).
- Dias com `OK (conf.) = 0` **não devem ser tratados como “0 oportunidade”** sem investigar o operacional.


Leitura: o walk-forward mede OOS principalmente por **ROI**, então ele encolhe quando a cobertura de placar é baixa. Além disso, a métrica é agregada por **jogo único** (cluster), então você verá números menores que o N de eventos.

| Train window | Test window | #ativas | Jogos OOS | ROI OOS (mean; IC90) | Turnover (teste) | Lucro (estratégia, budget) |
|---|---|---:|---:|---:|---:|---:|
| 2026-02-08→2026-02-09 | 2026-02-10→2026-02-11 | 1 | 56 | -5.15% [-23.59%, +13.52%] | 686.99 | 14.95 |
| 2026-02-08→2026-02-11 | 2026-02-12→2026-02-13 | 2 | 91 | -6.14% [-18.48%, +6.72%] | 6011.27 | -759.86 |
| 2026-02-08→2026-02-13 | 2026-02-14→2026-02-15 | 6 | 178 | +14.93% [+4.99%, +25.02%] | 9601.70 | 830.76 |
| 2026-02-08→2026-02-15 | 2026-02-16→2026-02-19 | 6 | 92 | -2.87% [-21.41%, +17.00%] | 4002.52 | 231.75 |
| 2026-02-08→2026-02-19 | 2026-02-22→2026-02-23 | 3 | 23 | +42.44% [-19.58%, +133.60%] | 1023.00 | 255.75 |

**Frequência de ativação por combinação (quantas janelas ela entrou como ativa)**

| Combinação | #steps ativa |
|---|---:|
| Back_Pre_Any | 4 |
| Back_In_Any | 4 |
| Lay_In_No | 3 |
| Lay_In_Yes | 3 |
| Lay_Pre_No | 2 |
| Lay_Pre_Yes | 2 |

### 12.A Transparência da seleção: métricas por combinação no treino
Para cada janela de treino, mostramos as métricas usadas para decidir se cada combinação ficou **ativa** ou não. Isso ajuda a entender, por exemplo, por que nenhuma Lay entrou em algumas janelas (geralmente N insuficiente, ROI sig<0, ou ROI<=0 com CLV<=0 no pre‑match).

**Regra de elegibilidade (todas as combinações):** `wf_min_matches=0` ⇒ mínimo de N **desligado**.

**Train 2026-02-08→2026-02-09 → Test 2026-02-10→2026-02-11**

| Combinação | Ativa? | Jogos treino (tot/CLV/ROI) | CLV mean (IC90) | ROI mean (IC90) | ROI mean (shrunk) | ROI q30 | Motivo |
|---|---|---:|---:|---:|---:|---:|---|
| Back_Pre_Any | SIM | 23 / 14 / 21 | +3.91% [+1.35%, +6.35%] | +25.39% [-8.42%, +57.70%] | +25.39% | +15.17% | BackPre: ROI>0 (NS) AND CLV>0 |
| Back_In_Any | NÃO | 5 / — / 5 | — | -18.14% [-100.00%, +86.48%] | -18.14% | -46.00% | BackIn: ROI>0=False |
| Lay_Pre_Yes | NÃO | 0 / — / — | — | — | — | — |  |
| Lay_Pre_No | NÃO | 0 / — / — | — | — | — | — |  |
| Lay_In_Yes | NÃO | 0 / — / — | — | — | — | — |  |
| Lay_In_No | NÃO | 0 / — / — | — | — | — | — |  |

**Train 2026-02-08→2026-02-11 → Test 2026-02-12→2026-02-13**

| Combinação | Ativa? | Jogos treino (tot/CLV/ROI) | CLV mean (IC90) | ROI mean (IC90) | ROI mean (shrunk) | ROI q30 | Motivo |
|---|---|---:|---:|---:|---:|---:|---|
| Back_Pre_Any | SIM | 74 / 39 / 70 | +4.96% [+3.57%, +6.37%] | +3.77% [-12.69%, +20.43%] | +3.77% | -1.54% | BackPre: ROI>0 (NS) AND CLV>0 |
| Back_In_Any | SIM | 53 / — / 49 | — | +11.43% [-7.38%, +31.30%] | +11.43% | +5.25% | BackIn: ROI>0=True |
| Lay_Pre_Yes | NÃO | 0 / — / — | — | — | — | — |  |
| Lay_Pre_No | NÃO | 0 / — / — | — | — | — | — |  |
| Lay_In_Yes | NÃO | 0 / — / — | — | — | — | — |  |
| Lay_In_No | NÃO | 0 / — / — | — | — | — | — |  |

**Train 2026-02-08→2026-02-13 → Test 2026-02-14→2026-02-15**

| Combinação | Ativa? | Jogos treino (tot/CLV/ROI) | CLV mean (IC90) | ROI mean (IC90) | ROI mean (shrunk) | ROI q30 | Motivo |
|---|---|---:|---:|---:|---:|---:|---|
| Back_Pre_Any | SIM | 133 / 99 / 123 | +6.37% [+5.76%, +6.96%] | +0.93% [-9.35%, +11.52%] | +2.23% | -2.51% | BackPre: ROI>0 (NS) AND CLV>0 |
| Back_In_Any | SIM | 85 / — / 69 | — | +5.29% [-10.02%, +21.20%] | +4.64% | +0.10% | BackIn: ROI>0=True |
| Lay_Pre_Yes | SIM | 6 / 5 / 6 | +6.83% [+3.77%, +10.04%] | -19.72% [-67.83%, +45.35%] | +3.03% | -35.66% | LayPre: ROI>0 (NS) AND CLV_CONV>0 |
| Lay_Pre_No | SIM | 17 / 17 / 16 | +3.18% [+0.52%, +5.61%] | -20.68% [-60.17%, +20.81%] | +1.93% | -33.45% | LayPre: ROI>0 (NS) AND CLV_CONV>0 |
| Lay_In_Yes | SIM | 9 / — / 6 | — | +35.45% [+5.01%, +67.84%] | +8.68% | +22.01% | In: ROI sig>0 |
| Lay_In_No | SIM | 14 / — / 13 | — | +24.26% [-10.21%, +59.38%] | +6.61% | +12.54% | In: ROI>0=True |

**Train 2026-02-08→2026-02-15 → Test 2026-02-16→2026-02-19**

| Combinação | Ativa? | Jogos treino (tot/CLV/ROI) | CLV mean (IC90) | ROI mean (IC90) | ROI mean (shrunk) | ROI q30 | Motivo |
|---|---|---:|---:|---:|---:|---:|---|
| Back_Pre_Any | SIM | 188 / 141 / 174 | +6.29% [+5.83%, +6.76%] | +8.22% [-0.64%, +17.48%] | +8.22% | +5.09% | BackPre: ROI>0 (NS) AND CLV>0 |
| Back_In_Any | SIM | 175 / — / 155 | — | +7.51% [-3.75%, +19.14%] | +7.51% | +3.84% | BackIn: ROI>0=True |
| Lay_Pre_Yes | SIM | 13 / 10 / 13 | +4.09% [+0.64%, +7.13%] | +9.94% [-31.01%, +50.28%] | +9.94% | -2.85% | LayPre: ROI>0 (NS) AND CLV_CONV>0 |
| Lay_Pre_No | SIM | 37 / 30 / 34 | +4.14% [+2.34%, +5.90%] | +8.70% [-19.18%, +37.11%] | +8.70% | +0.02% | LayPre: ROI>0 (NS) AND CLV_CONV>0 |
| Lay_In_Yes | SIM | 25 / — / 19 | — | +18.05% [-17.98%, +53.71%] | +18.05% | +6.62% | In: ROI>0=True |
| Lay_In_No | SIM | 32 / — / 30 | — | +4.60% [-24.97%, +33.72%] | +4.60% | -4.19% | In: ROI>0=True |

**Train 2026-02-08→2026-02-19 → Test 2026-02-22→2026-02-23**

| Combinação | Ativa? | Jogos treino (tot/CLV/ROI) | CLV mean (IC90) | ROI mean (IC90) | ROI mean (shrunk) | ROI q30 | Motivo |
|---|---|---:|---:|---:|---:|---:|---|
| Back_Pre_Any | NÃO | 214 / 165 / 200 | +6.34% [+5.92%, +6.75%] | -0.01% [-8.63%, +8.41%] | -0.01% | -2.73% | BackPre: ROI>0=False, CLV>0=True |
| Back_In_Any | SIM | 197 / — / 171 | — | +12.57% [-0.27%, +25.90%] | +12.57% | +8.21% | BackIn: ROI>0=True |
| Lay_Pre_Yes | NÃO | 18 / 15 / 18 | +0.99% [-2.49%, +4.22%] | -3.32% [-37.25%, +30.60%] | -3.32% | -14.63% | LayPre: ROI>0=False, CLV_CONV>0=True |
| Lay_Pre_No | NÃO | 52 / 43 / 48 | +3.10% [+1.31%, +4.91%] | -8.49% [-30.90%, +14.31%] | -8.49% | -15.57% | LayPre: ROI>0=False, CLV_CONV>0=True |
| Lay_In_Yes | SIM | 29 / — / 22 | — | +22.67% [-10.23%, +54.82%] | +22.67% | +12.44% | In: ROI>0=True |
| Lay_In_No | SIM | 39 / — / 36 | — | +11.29% [-14.84%, +38.31%] | +11.29% | +2.87% | In: ROI>0=True |


Notas importantes:
- Se `Jogos OOS` for baixo em muitos passos, você ainda não tem volume suficiente para decisões por combinação. Nesse cenário faz sentido **Bayes hierárquico (partial pooling)** para estabilizar estimativas.
- **Lucro (estratégia, budget)** acima já incorpora a política de risco por jogo (match budget) e é a métrica principal.
- O walk-forward usa ROI no **ponto de entrada**: Back em `t0`; Lay em `t_reversal` quando existir, senão `t_last` (~t+20s).
- Para Lay pre-match, o CLV usado na seleção é `clv_conv = -(entry-closing)/closing`, ou seja **Lay “bom” tende a ser positivo**.
- Para pre-match, também é útil monitorar CLV OOS (menos dependente de resultados), mas CLV mede qualidade de entrada, não P&L.

**O que significa 'Bayes hierárquico / partial pooling' aqui?**

Quando você tem poucas partidas por combinação na janela de treino/teste, o estimador (ex.: ROI p30) fica muito ruidoso e pode alternar sinal por acaso. O Bayes hierárquico modela cada combinação como um desvio de um **efeito global** (ex.: ROI médio global do live) e aplica **shrinkage**: combinações com pouco N são puxadas para o global; combinações com muito N “ganham identidade própria”.

Na prática isso reduz falsos positivos/negativos no rolling e torna a seleção mais estável quando o volume ainda é baixo.

### 12.1 Estimativa 30 dias (OOS): turnover, lucro, banca, ROI/banca e drawdown
Esta estimativa usa o walk-forward acima como **simulador OOS**. O lucro pode ser reportado em duas versões:

- **obs.**: apenas jogos com ROI (placar) disponível.
- **exp.**: expande o lucro para a população elegível usando scaling por exposição/turnover (assume missing-at-random condicional à estratégia).

**Padrão de risco**: P&L aqui já é calculado com **budget por jogo (match_id)** consumido ao longo do tempo (Back=1.00% da banca ref; Lay=0.50% em liability; cap por sinal=33% do budget; mode=fixed).

**Sizing FLAT (quando aplicável no WF)**: Back stake=80.00 | Lay liability=80.00.

| Premissa | Valor |
|---|---:|
| Train mode (OOS) | `expanding` |
| Scheme pre-match (OOS) | `KELLY_0.25` |
| Scheme in-match (OOS) | `FLAT` |
| Expansão missing ROI | ON |
| Dias OOS (calendário de teste) | 10 |
| Dias OOS com OK (>=1 evento OK/conf) | 10 |
| Turnover 30d (proj., calendário) | 63976.45 |
| Turnover 30d (proj., cond OK) | 63976.45 |
| Turnover 30d (Pre/In) | 128991.29 / 79508.22 |
| Lucro 30d (obs., calendário) | 1921.09 |
| Lucro 30d (obs., cond OK) | 1921.09 |
| Lucro 30d (obs.) Pre/In | -2678.63 / 4874.10 |
| Lucro 30d (exp., calendário) | 1720.02 |
| Lucro 30d (exp., cond OK) | 1720.02 |
| Lucro 30d (exp.) Pre/In | -2771.88 / 5476.87 |
| Banca risco p99 (Back+Lay) | 8988.48 |
| Banca liquidez p99 (+buf) | 9127.25 |
| Banca recomendada (max) | 9127.25 |
| ROI/banca 30d (obs., calendário) | 21.05% |
| ROI/banca 30d (obs., cond OK) | 21.05% |
| ROI/banca 30d (exp., calendário) | 18.84% |
| ROI/banca 30d (exp., cond OK) | 18.84% |
| DD 30d p95 (obs., calendário) | 1573.69 |
| DD 30d p95 (obs., cond OK) | 1491.19 |
| DD 30d p95 (exp., calendário) | 1904.46 |
| DD 30d p95 (exp., cond OK) | 1899.70 |

### 12.2 Governança de exposição por jogo (budget por `match_id`) — sensibilidade
Objetivo: evitar concentração quando um mesmo jogo gera muitos sinais. Simulamos um orçamento de exposição por jogo, consumido ao longo do tempo.

- **Back** consome budget em **stake**.
- **Lay** consome budget em **liability**.
- Também aplicamos um cap por sinal como fração do budget do jogo (para não gastar tudo no 1º sinal).

Importante: o budget é parametrizado como fração de uma referência de banca. Aqui usamos:
- se `--kelly-bankroll` estiver setado: essa banca explícita;
- senão: a `Banca recomendada (max)` estimada na 12.1 (baseline).

Referência de banca p/ budget: 10000.00 | budgets por jogo aplicados em stake (Back) e liability (Lay).

| Cenário | Turnover 30d | Lucro 30d (exp.) | Banca rec. (max) | ROI/banca 30d (exp.) | DD 30d p95 (exp.) |
|---|---:|---:|---:|---:|---:|
| BASELINE (sem budget) | 63976.45 | 1720.02 | 9127.25 | 18.84% | 1904.46 |
| BUDGET_0.50%/0.25% cap25% | 16793.78 | 64.53 | 2127.69 | 3.03% | 754.69 |
| BUDGET_1.00%/0.50% cap33% | 37806.88 | -285.18 | 4713.83 | -6.05% | 2322.29 |
| BUDGET_2.00%/1.00% cap50% | 82216.91 | -2984.22 | 10352.70 | -28.83% | 7024.18 |
| BUDGET_3.00%/1.50% cap33% | 89002.38 | -2600.90 | 11382.62 | -22.85% | 6555.31 |
| BUDGET_4.00%/2.00% cap33% | 107115.83 | -3733.91 | 13864.76 | -26.93% | 7892.91 |
| BUDGET_3.00%/1.50% cap50% | 106345.11 | -3152.37 | 13512.57 | -23.33% | 7739.93 |
| BUDGET_4.00%/2.00% cap50% | 120008.19 | -2789.43 | 15398.92 | -18.11% | 6813.40 |
| BUDGET_EQ_0.50%/0.50% cap25% | 17248.33 | 145.92 | 2224.46 | 6.56% | 791.33 |
| BUDGET_EQ_1.00%/1.00% cap33% | 38843.67 | -145.33 | 4938.25 | -2.94% | 2290.56 |
| BUDGET_EQ_2.00%/2.00% cap50% | 83667.99 | -2844.99 | 10684.44 | -26.63% | 6980.63 |
| BUDGET_EQ_3.00%/3.00% cap33% | 90464.42 | -2462.95 | 11715.85 | -21.02% | 6633.70 |
| BUDGET_EQ_4.00%/4.00% cap33% | 107777.13 | -3612.77 | 14031.64 | -25.75% | 7994.91 |
| BUDGET_EQ_3.00%/3.00% cap50% | 106764.54 | -3099.56 | 13620.86 | -22.76% | 7591.37 |
| BUDGET_EQ_4.00%/4.00% cap50% | 120008.19 | -2789.43 | 15398.92 | -18.11% | 6924.94 |
| RISK(signals_sqrt) @ 1.00%/0.50% cap33% | 27443.88 | -36.27 | 3319.94 | -1.09% | 1414.24 |

Leitura:
- Se a curva com budget melhora muito (menos negativo ou mais positivo) com pouca perda de turnover, o problema era **concentração por jogo**.
- Se tudo continuar negativo, o problema é **edge OOS** (principalmente in‑match) e budget só reduz a escala da perda.

**Volume e stake médio por combinação (janela OOS, com budget padrão)**

| Combinação | #steps ativa | Jogos OOS (uniques) | N eventos OOS | Stake eq. médio | Observação |
|---|---:|---:|---:|---:|---|
| Back_Pre_Any | 4 | 245 | 432 | 96.11 | budget reduz concentração por jogo |
| Back_In_Any | 4 | 167 | 264 | 80.00 | budget reduz concentração por jogo |
| Lay_In_No | 3 | 25 | 30 | 115.47 | budget reduz concentração por jogo |
| Lay_In_Yes | 3 | 20 | 26 | 73.79 | budget reduz concentração por jogo |
| Lay_Pre_No | 2 | 19 | 19 | 67.29 | budget reduz concentração por jogo |
| Lay_Pre_Yes | 2 | 5 | 5 | 40.15 | budget reduz concentração por jogo |
## 10) Diagnóstico: por que o ROI pode estar zerado
ROI aqui é calculado por placar do jogo (`matches.home_score/away_score`). Se os placares não estiverem preenchidos no banco, a cobertura de ROI será 0.

| Indicador | Valor |
|---|---:|
| Jogos únicos no recorte | 684 |
| Jogos com placar disponível (home_score/away_score não nulos) | 606 |
| Jogos com status='finished' no banco | 606 |

### 10.1 Distribuição por data de kickoff (explica janela da API)
- Kickoff (UTC) no recorte: **2026-02-08 15:00 UTC** até **2026-02-23 20:15 UTC**.

| Kickoff date (UTC) | Jogos | Com placar | Cobertura |
|---|---:|---:|---:|
| 2026-02-23 | 21 | 16 | 76.2% |
| 2026-02-22 | 95 | 93 | 97.9% |
| 2026-02-21 | 79 | 78 | 98.7% |
| 2026-02-20 | 17 | 16 | 94.1% |
| 2026-02-19 | 36 | 32 | 88.9% |
| 2026-02-18 | 12 | 9 | 75.0% |
| 2026-02-17 | 40 | 40 | 100.0% |
| 2026-02-16 | 24 | 19 | 79.2% |
| 2026-02-15 | 77 | 72 | 93.5% |
| 2026-02-14 | 118 | 108 | 91.5% |
| 2026-02-13 | 51 | 34 | 66.7% |
| 2026-02-12 | 5 | 2 | 40.0% |
| 2026-02-11 | 41 | 30 | 73.2% |
| 2026-02-10 | 55 | 45 | 81.8% |

**Leitura**: se seu recorte inclui muitos jogos com kickoff antigo, a API-Football **free** pode não retornar fixtures dessa data (limitação por janela recente). Nesse cenário, mesmo com o job rodando, `placar disponível` ficará baixo para jogos fora da janela.

Se `placar disponível` estiver 0 (mesmo para datas recentes), isso geralmente indica que o job de resultados não rodou ou está sem chave válida.  
Sugestão (rodar no servidor): `cd betinasia_bot && python3 -m results.auto_update_results --once` (ou configure o serviço para rodar em loop).

---
## 11) Conclusões (visão de investidor), riscos e próximos passos
Esta seção é escrita como se um investidor externo estivesse avaliando a tese: **há edge replicável? o sistema executa? o risco é governável? a mensuração é confiável?**

### 11.1 O que já está forte (e por quê)
- **Evidência de execução (CLV pre‑match)**: CLV robusto por jogo positivo é um dos melhores sinais de edge/execução em janela curta. Diferente de ROI, CLV não depende de amostra grande de jogos liquidados; ele mede **qualidade de entrada**.
- **Controle de latência por regime**: o relatório já separa regimes de execução por tempo total (2.3/2.3b). Isso permite uma regra objetiva de operação (ex.: só operar `exec_bucket < 5s`).
- **Separação Back vs Lay**: Back e Lay têm perfis de risco diferentes. Lay deve ser governado por **liability** (p95/p99/ES), e isso já aparece como métrica de banca e risco.

### 11.2 O que ainda está frágil (e impede captação hoje)
- **ROI ainda não é prova**: mesmo quando ROI aparece, a incerteza por jogo pode ser grande e a cobertura de placar pode ser incompleta. Para captação, um investidor vai pedir **histórico maior**, **pipeline de resultados estável** e **métrica de drawdown** bem definida.
- **Risco de viés por falhas de coleta**: quando o collector fica “active” mas não coleta odds, você perde janelas do mercado de forma não aleatória. Isso impacta a extrapolação para execução.
- **Stake sizing ainda é proxy**: parte do sizing usa limit/finance como aproximação. Para captação, é necessário um sizing governado por risco e consistente com edge (ex.: Kelly fracionado + caps), com auditoria clara.

### 11.3 Avaliação das 2 estratégias candidatas (como um investidor leria)
Você propôs duas teses operacionais coerentes com o mecanismo observado:
1) **BackFast**: operar Back edge apenas quando a execução foi rápida (`< 5s`) e pre‑match.
2) **LayReversal**: operar Lay edge apenas quando há reversão e entrar próximo do vale (t_ext curto).

O relatório quantifica isso na **Seção 9.4** com (i) N na janela, (ii) projeção 30d, (iii) stake/liability médio, (iv) banca p99 e ROI/banca mensal, e (v) drawdown p95.

**Como um investidor decide**: ele vai priorizar uma estratégia com
- sinal de edge (CLV) consistente,
- execução estável (latência controlada),
- sizing governado por risco (caps + banca p99/ES),
- e um perfil de drawdown aceitável no horizonte de caixa.

### 11.4 Stake sizing: recomendação inicial para produção (sem overfitting)
- Use **baseline FLAT** como controle (para detectar se o sizing está degradando performance).
- Para Back, use **Kelly fracionado** (ex.: `KELLY_0.25`) apenas quando houver `closing_odd` (pre‑match), com **cap** por aposta (ex.: 2% da banca p99).
- Para Lay, faça sizing por **liability**, com cap mais conservador (ex.: 1% da banca p99) e monitoramento de cauda (p95/p99/ES95).

A Seção 9.3 compara `FLAT` vs `PROXY` vs `KELLY` (fracionado) no subconjunto com placar, e reporta risco (p99/ES) e drawdown 30d via bootstrap.

### 11.5 Status para captação (checkpoint objetivo)
Se você estivesse captando hoje, um investidor institucional provavelmente pediria:
- **(A)** 30–90 dias de execução estável com SLO de coleta (collector), auditoria e resultados.
- **(B)** KPIs: CLV pre‑match por jogo estável; latência por bucket; taxa de falhas; cobertura de placar.
- **(C)** Política de risco: banca por p99/ES, caps por aposta, limites por janela e mecanismos de stop.
- **(D)** Demonstração de P&L com sizing definido (não só proxy) e drawdown observado/estimado.

Minha leitura: **a tese de edge/execução parece promissora pelo CLV**, mas o projeto ainda está em fase de **consolidação operacional/medição** para uma captação “grande”. Um caminho pragmático é:
- validar BackFast com sizing conservador e risco baixo,
- validar LayReversal com governança de liability,
- e só então ampliar banca.

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
