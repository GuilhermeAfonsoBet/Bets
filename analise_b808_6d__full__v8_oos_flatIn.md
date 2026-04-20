# Análise Estatística Robusta — Contexto Operação (b808)
**Data da execução:** 17/02/2026 13:11 UTC  
**Escopo:** H3B (auditoria), com comparação por `audit_version` e inferência robusta por jogo (cluster bootstrap).  
**Nota:** a robustez aqui significa que intervalos de confiança consideram correlação intra-jogo (múltiplas auditorias por partida).
---
## 0) Sumário executivo (leitura rápida)
- **Recorte**: direction=`up`, lookback_days=`6`, versions=`v4.0-api,v1.0,v1.0-recovered`.
- **Amostra**: 3430 auditorias (jogos únicos=299, média=11.5 obs/jogo); betslip confiável=1736.
- **Janela efetiva (audited_at)**: 11/02 13:14 → 17/02 09:51 UTC (span≈5.9d; dias com dados=7).
- **Coortes (status=OK, betslip confiável)**: `BS>WS` (diff>=2.0%): **605**; `BS<WS` (diff<=-2.0%): **220**.
- **Coberturas em `hypothesis_details` (OK)**: temporal(back)=1288/1736; lay_temporal=1168/1736; finance=1058/1736.
- **Cobertura de placar (ROI)**: jogos com placar=252/299 (status finished=252).
- **Cobertura de closing_odd (AH)**: jogos com closing=150/299 (50.2%). CLV pre‑match depende disso.
- **Alerta**: cobertura de closing_odd está baixa. Isso tende a reduzir N de CLV e pode enviesar a leitura (os jogos “sem odds” ficam fora do CLV). Priorize estabilizar o collector e/ou condicionar análises a jogos com closing.
- **DOM**: sem dados no recorte atual (N=0), então não há comparação API vs DOM aqui.
- **CLV pre-match (Betslip, API)**: média robusta por jogo +1.959% (IC90 [+1.424%, +2.484%]), com N=747 eventos (jogos=123).
- **Padrão por bucket (CLV PM)**: `BS < WS` -3.974% (sig. negativo), `BS ~ WS` -0.116% (NS), `BS > WS` +6.647% (sig. positivo).
- **Leitura recomendada**: use este relatório para validar **qualidade de execução/CLV**. Para concluir sobre **ROI**, rode após atualizar resultados e/ou use uma janela com jogos já liquidados.

---
## 1) Contexto do corte (b808)
| Indicador | Valor |
|---|---:|
| Auditorias H3B `UP` (match + kickoff passado) | 3430 |
| Betslip bruto | 2811 |
| Betslip confiável (diff -10% a +10%) | 1736 |
| Descartados no filtro de qualidade | 1075 |
| Jogos únicos (geral) | 299 |
| Média de observações por jogo | 11.5 |
| Jogos únicos com betslip confiável | 286 |
| Distribuição por market_type | AH=3430 |
| Jogos únicos (AH) no recorte | 299 |
| Jogos únicos (AH) com closing_odd disponível | 150 |
| Cobertura closing_odd (AH) | 50.2% |

---
## 2) Base comparativa: API vs DOM
| Métrica | API (v4.0-api) | DOM (v1.0) |
|---|---:|---:|
| Total de observações | 3430 | 0 |
| Com betslip confiável | 1736 | 0 |
| Com CLV pre-match (betslip) | 747 | 0 |
| Com ROI (betslip) | 1533 | 0 |
| Tempo total observado (detecção→betslip, wall/total) | 16596 ms | — ms |
| Tempo instrumentado (detecção→clique→betslip) | 8573 ms | — ms |

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
| API (2-4s) | lag_det→click | 5987 | 824 | 7557 | 3430 |
| API (2-4s) | lag_click→betslip | 2602 | 2140 | 4298 | 3350 |
| API (2-4s) | lag_e2e (soma) | 8573 | 3422 | 9898 | 3350 |
| API (2-4s) | audit_total (duração) | 16591 | 4765 | 57699 | 3430 |
| API (2-4s) | overhead (total - e2e) | 8187 | 306 | 33851 | 3350 |
| DOM (15-30s) | lag_det→click | — | — | — | 0 |
| DOM (15-30s) | lag_click→betslip | — | — | — | 0 |
| DOM (15-30s) | lag_e2e (soma) | — | — | — | 0 |
| DOM (15-30s) | audit_total (duração) | — | — | — | 0 |
| DOM (15-30s) | overhead (total - e2e) | — | — | — | 0 |

**Diagnóstico de cauda (percentual acima do limiar)**

| Modelo | % det→click > 5s | % det→click > 20s | % total > 10s | % total > 40s | % overhead < 0 |
|---|---:|---:|---:|---:|---:|
| API (2-4s) | 5.9% | 3.4% | 21.0% | 7.2% | 0.0% |
| DOM (15-30s) | —% | —% | —% | —% | —% |

---
### 2.1 Cobertura temporal (pre-match vs in-match)
| Métrica | Pre-match | In-match | Observação |
|---|---:|---:|---|
| Observações totais com classificação temporal | 1289 | 2141 | Contagem bruta do corte |
| ROI Betslip | 814 | 719 | Amostra com resultado do jogo |
| ROI WebSocket | 1159 | 1857 | Referência de mercado |
| CLV (apenas pre-match) | 747 | — | CLV vs closing pré-jogo não é interpretável in-match |

---
### 2.2 Performance por regime (pre-match vs in-match)
| Regime | N auditorias | N betslip conf. | N OK | Back edge | Lay edge | Diff médio (OK) |
|---|---:|---:|---:|---:|---:|---:|
| PRE_MATCH | 1289 | 896 | 896 | 344 | 64 | +2.113% |
| IN_MATCH | 2141 | 840 | 840 | 261 | 156 | +0.980% |

---
### 2.3 Regimes operacionais por tempo total (bucket)
Objetivo: separar a amostra em **regimes de execução** (tempo total) e medir performance em cada regime. Use isso para detectar **fila/saturação**: regimes lentos tendem a ter edge menor e pior ROI.

| Bucket (tempo total) | N OK | Jogos | lag_total mean (ms) | lag_total p95 (ms) | overhead p95 (ms) | Back edge | Lay edge | CLV PM (mean; IC90) | ROI (mean; IC90) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| < 5s | 1117 | 239 | 3585 | 4850 | 2507 | 361 | 130 | +2.22% [+1.72%, +2.72%] | +3.26% [-3.05%, +9.62%] |
| 5-10s | 210 | 130 | 5921 | 8399 | 2966 | 84 | 42 | +1.72% [+0.70%, +2.70%] | -4.10% [-16.69%, +8.60%] |
| 10-20s | 17 | 16 | 13456 | 18180 | 4295 | 8 | 2 | +1.22% [-2.21%, +4.44%] | +17.36% [-14.29%, +49.40%] |
| 20-40s | 268 | 119 | 27292 | 34410 | 30761 | 93 | 31 | +2.03% [+1.09%, +3.00%] | +10.08% [-1.28%, +21.48%] |
| > 40s | 124 | 74 | 137452 | 433092 | 221376 | 59 | 15 | +2.25% [+1.11%, +3.35%] | -13.75% [-29.42%, +1.70%] |
| Desconhecido | 0 | 0 | — | — | — | 0 | 0 | — | — |

---
### 2.3b Regimes por tempo total — separando coortes por `diff_pct` (BS vs WS)
Nesta tabela, separamos duas coortes operacionais por **delta de execução**: `BS > WS` (diff_pct >= +2%) e `BS < WS` (diff_pct <= -2%). Isso **não** é (por si só) “Back vs Lay”; é um recorte por **melhora/piora do preço** entre detecção (WS) e execução (BS). CLV é reportado apenas em pre‑match.

| Bucket | N OK | N (BS>WS +2%) | N (BS<WS -2%) | CLV PM (BS>WS) | CLV PM (BS<WS) | ROI (BS>WS) | ROI (BS<WS) |
|---|---:|---:|---:|---:|---:|---:|---:|
| < 5s | 1117 | 361 | 130 | +6.90% [+6.42%, +7.38%] | -3.74% [-4.92%, -2.54%] | +1.75% [-9.01%, +12.55%] | +4.44% [-11.11%, +20.37%] |
| 5-10s | 210 | 84 | 42 | +5.78% [+4.98%, +6.52%] | -6.16% [-7.68%, -4.56%] | -9.94% [-29.10%, +9.07%] | -13.90% [-34.07%, +6.88%] |
| 10-20s | 17 | 8 | 2 | +4.89% [+2.51%, +7.32%] | -6.98% — | +0.99% [-49.88%, +52.37%] | +40.04% [+0.00%, +79.90%] |
| 20-40s | 268 | 93 | 31 | +7.79% [+6.78%, +8.80%] | -2.03% [-4.39%, +0.55%] | +11.30% [-6.29%, +28.98%] | +6.33% [-24.85%, +35.90%] |
| > 40s | 124 | 59 | 15 | +5.06% [+3.27%, +6.89%] | -7.29% — | -12.41% [-35.19%, +10.41%] | -11.35% [-55.50%, +31.89%] |
| Desconhecido | 0 | 0 | 0 | — | — | — | — |

---
### 2.3c Estabilidade temporal (por dia, `audited_at`)
Objetivo: checar se o regime de edge/execução é **time‑dependent**. Se houver dias com comportamento distinto (ex.: collector intermitente, horários/ligas), isso aparecerá aqui.

| Dia (UTC) | Modelo | N OK | Jogos | % BS>WS +2% | % BS<WS -2% | lag_total p50 (ms) | CLV PM (BS>WS) | CLV PM (BS<WS) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-02-11 | API (2-4s) | 156 | 55 | 39.1% | 17.3% | 27244 | +7.02% | -4.08% |
| 2026-02-12 | API (2-4s) | 50 | 40 | 36.0% | 8.0% | 26843 | +5.67% | -5.50% |
| 2026-02-13 | API (2-4s) | 563 | 132 | 36.2% | 10.3% | 4821 | +6.78% | -3.45% |
| 2026-02-14 | API (2-4s) | 488 | 126 | 35.2% | 15.0% | 4203 | +6.63% | -5.24% |
| 2026-02-15 | API (2-4s) | 380 | 74 | 30.5% | 12.4% | 3561 | +6.17% | -2.58% |
| 2026-02-16 | API (2-4s) | 99 | 30 | 34.3% | 11.1% | 3314 | +6.42% | — |

---
## 3) CLV pre-match (núcleo)
### 3.1 CLV com odd do Betslip (execução real)
| Métrica | API | DOM |
|---|---:|---:|
| CLV Bruto BS Pre-Match | +2.282% (sig. positivo, N=747, jogos=123) | — (N/A, N=0, jogos=0) |
| CLV Adicional BS Pre-Match | +1.548% (sig. positivo, N=747, jogos=123) | — (N/A, N=0, jogos=0) |
| Taxa de CLV > 0 (bruto) | 61.0% | —% |
| Taxa de CLV > 0 (adicional) | 60.2% | —% |

Notas de robustez (IC 90% por jogo):  
- API CLV bruto (cluster): média +1.959%; IC90 [+1.424%, +2.484%]  
- DOM CLV bruto (cluster): média —; IC90 —  

---
## 4) ROI por modelo
| Métrica | API | DOM |
|---|---:|---:|
| ROI Betslip | +1.144% (NS, N=1531) | — (N/A, N=0) |
| ROI WebSocket | -0.405% (NS, N=3000) | — (N/A, N=0) |
| Win rate ROI Betslip | 50.7% | —% |
| Win rate ROI WS | 50.1% | —% |

Notas de robustez (IC 90% por jogo):  
- API ROI betslip (cluster): média +1.158%; IC90 [-3.951%, +6.205%]  
- API ROI WS (cluster): média -2.127%; IC90 [-4.455%, +0.177%]  

---
## 5) Diferença de preço BS vs WS
| Métrica | API | DOM |
|---|---:|---:|
| Diff BS vs WS (média) | +1.565% (sig. positivo, N=1736) | — (N/A, N=0) |
| BS > WS | 46.6% (809/1736) | —% (0/0) |
| BS > WS +2% | 34.9% (605/1736) | —% (0/0) |

---
## 6) Combinações de valor
### 6.1 Buckets por diferença BS vs WS
Nota de leitura:
- `BS > WS (+2% a +10%)`: preço no betslip ficou **melhor** do que o WS observado na detecção (delta positivo entre instantes).
- `BS < WS (-10% a -2%)`: preço no betslip ficou **pior** do que o WS observado na detecção (delta negativo entre instantes).
- O ROI abaixo é calculado **dentro do bucket** (não mistura buckets), mas ainda é sensível à cobertura de placar.

| Bucket | N bucket | CLV BS PM (média) | IC90 (cluster) | N CLV PM | Jogos CLV PM | ROI BS (todos) | IC90 (cluster) |
|---|---:|---:|---|---:|---:|---:|---|
| BS < WS (-10% a -2%) | 220 | -3.974% | [-5.072%, -3.182%] | 51 | 39 | +4.468% | [-11.908%, +11.113%] |
| BS ~ WS (-2% a +2%) | 911 | -0.116% | [-0.612%, +0.265%] | 402 | 109 | +0.029% | [-9.810%, +3.208%] |
| BS > WS (+2% a +10%) | 605 | +6.647% | [+6.334%, +7.204%] | 294 | 91 | +1.615% | [-2.538%, +14.594%] |

---
### 6.2 Combinação por faixa de linha AH
| Faixa AH | CLV BS PM (média) | IC90 (cluster) | ROI BS (todos) | IC90 (cluster) | Diff BS vs WS (média) |
|---|---:|---|---:|---|---:|
| AH 0-1 (líquida) | +2.462% | [+1.676%, +3.200%] | +0.833% | [+0.641%, +15.353%] | +1.807% |
| AH 1-2 (média) | +2.032% | [+1.017%, +3.534%] | +3.592% | [-10.999%, +15.867%] | +2.169% |
| AH 2+ (extrema) | +2.200% | [+0.590%, +2.424%] | +0.518% | [-8.663%, +6.763%] | +1.178% |

---
### 6.3 Combinação por faixa de lag
| Faixa de lag | CLV BS PM (média) | IC90 (cluster) | N CLV PM | Jogos CLV PM | ROI BS (todos) | IC90 (cluster) | Diff BS vs WS (média) |
|---|---:|---|---:|---:|---:|---|---:|
| < 10s | +2.376% | [+1.611%, +2.625%] | 534 | 113 | +1.170% | [-4.277%, +8.099%] | +1.473% |
| 10-20s | +1.285% | [-2.212%, +4.435%] | 6 | 6 | +16.300% | [-14.288%, +49.400%] | +1.638% |
| 20-30s | +2.109% | [+1.141%, +3.119%] | 132 | 71 | +3.652% | [-4.853%, +18.811%] | +1.781% |
| > 30s | +1.998% | [+1.116%, +3.199%] | 75 | 49 | -4.227% | [-18.292%, +8.322%] | +1.992% |

---
## 7) Estimativa financeira (proxy) e risco
Este bloco usa `hypothesis_details.finance` quando existe; se não existir, usa stake fallback = `stake_pct_of_limit × limit`.

**Política de stake (proxy)**

| Parâmetro | Valor |
|---|---:|
| stake_pct_of_limit | 0.25 |
| stake_cap | 0.00 |
| Cobertura finance (OK, betslip conf.) | 1058/1736 |

### 7.1 Back (BS >> WS)
| Métrica | Valor |
|---|---:|
| Corte (diff_pct) | >= 2.0% |
| N eventos | 605 |
| Cobertura finance (na coorte) | 355/605 |
| Stake total (estimado) | 213454.35 |
| Stake médio | 352.82 |
| Profit_if_win total (estimado) | 229316.64 |
| Profit_if_win médio | 379.04 |
| N com ROI realizado | 540 |
| P&L realizado total (estimado) | -43545.02 |
| ROI realizado (ponderado por stake) | -20.99% |
| ROI realizado (robusto por jogo, mean; IC90) | +5.86% [-2.54%, +14.59%] |
| ROI ponderado por stake (robusto por jogo, mean; IC90) | +5.67% [-5.25%, +17.11%] |

**Como ler as 3 linhas de ROI (Back)**

- **ROI realizado (ponderado por stake)**: ΣP&L / Σstake (pode ser dominado por stakes grandes).
- **ROI realizado (robusto por jogo)**: média por jogo (cada jogo pesa parecido; reduz dominância de outliers).
- **ROI ponderado por stake (robusto por jogo)**: calcula ROI ponderado dentro do jogo e depois faz média/IC por jogo.

Observação: a Seção 4 reporta ROI médio no **recorte completo** (betslip confiável). Aqui (7.1) é apenas a coorte **Back edge** (diff>=corte) e o ROI também é mostrado ponderado por stake; por isso sinais podem divergir.

### 7.2 Lay (BS << WS) — risco de cauda
| Métrica | Valor |
|---|---:|
| Corte (diff_pct) | <= -2.0% |
| N eventos | 220 |
| Cobertura finance (na coorte) | 138/220 |
| Stake total (estimado) | 40997.50 |
| Liability total (estimada) | 35849.09 |
| Liability média | 162.95 |
| Liability p95 | 572.91 |
| Liability p99 | 2055.72 |
| ES95 (liability) | 1762.94 |
| Liability max | 4395.35 |
| Proxy de banca (>= p99 liability) | 2055.72 |
| N com ROI realizado | 142 |
| P&L realizado total (estimado) | -3174.20 |
| ROI realizado (ponderado por liability) | -9.46% |
| ROI realizado (ponderado por stake) | -8.24% |
| ROI/liability (robusto por jogo, mean; IC90) | +11.82% [-3.37%, +27.07%] |
| ROI/liability ponderado (robusto por jogo, mean; IC90) | +12.95% [-3.17%, +28.90%] |

### 7.3 Projeção mensal (30 dias fixo) — turnover, lucro e banca
Premissas:
- **Mês = 30 dias fixo**.
- Turnover = soma de stakes (Back e Lay). Para Lay, também reportamos exposição por **liability**.
- **Banca por risco (unitária)** = p99/ES95(exposição por aposta).
- **Banca por liquidez (turnover)** = p95/p99 de capital **simultaneamente travado** (stake/liability em aberto até a liquidação), capturando distribuição desigual de entradas ao longo do tempo.
- Projeção de lucro usa duas visões: (i) **lucro/dia observado** e (ii) **ROI observado × turnover projetado**.

| Bloco | Janela(d) | Turnover 30d | Lucro 30d (direto) | Lucro 30d (ROI×turnover) |
|---|---:|---:|---:|---:|
| Back | 6.0 | 1067271.74 | -217725.09 | -223973.23 |
| Lay (stake) | 6.0 | 204987.48 | -15871.02 | -16898.82 |
| Total (Back+Lay) | 6.0 | 1272259.22 | -233596.12 | -240872.04 |

**Banca por risco (exposição unitária)**

| Bloco | Banca conservadora (p99) | Banca agressiva (ES95) | ROI/banca 30d (direto) | ROI/banca 30d (ROI×turnover) |
|---|---:|---:|---:|---:|
| Back (stake) | 4399.26 | 4132.71 | -4949.13% | -5091.16% |
| Lay (liability) | 2055.72 | 1762.94 | -772.04% | -822.04% |
| Total (soma) | 6454.97 | 5895.65 | -3618.86% | -3731.57% |

**Banca por liquidez (capital simultaneamente travado)**

Definição operacional: cada aposta trava capital de `audited_at` até `kickoff + 2.25h` (grid=5min). A banca recomendada aplica buffer de +10%.

| Bloco | Liquidez mean | Liquidez p95 | Liquidez p99 | Liquidez max | Banca liq p99 (+buffer) |
|---|---:|---:|---:|---:|---:|
| Back (stake) | 18431.19 | 51493.70 | 74135.84 | 80021.28 | 81549.42 |
| Lay (liability) | 2429.40 | 6135.87 | 8732.10 | 8957.34 | 9605.31 |
| Total (Back+Lay) | 20758.19 | 51793.15 | 80301.83 | 85324.76 | 88332.01 |

**Banca recomendada (conservadora)**

| Métrica | Valor |
|---|---:|
| Banca por risco (p99 unitário, soma) | 6454.97 |
| Banca por liquidez (p99 simultâneo + buffer) | 88332.01 |
| Banca efetiva (max das duas) | 88332.01 |
| ROI/banca 30d (direto, banca efetiva) | -264.45% |
| ROI/banca 30d (ROI×turnover, banca efetiva) | -272.69% |
**Diagnóstico de cobertura (placar/ROI)**

| Bloco | Turnover total | Turnover com ROI | Cobertura turnover |
|---|---:|---:|---:|
| Back | 213454.35 | 207499.66 | 97.21% |
| Lay | 40997.50 | 38504.01 | 93.92% |

Notas (Lay): exposição 30d por liability (não é turnover) = 179245.45; ROI realizado por liability (ponderado) = -9.46%.

---
## 8) Curva temporal (pico, reversão e melhor timing)
Esta seção usa `hypothesis_details.temporal` (Back) e `hypothesis_details.lay_temporal` (Lay), coletados em pontos discretos (t≈0,3,6,10,15,20s).

O objetivo é responder: **tempo até o pico/vale**, **% que segue melhorando até t_max**, **% com reversão** e **impacto esperado por timing**. **CLV é reportado somente pre-match** (closing pré-jogo).

### 8.1 Back (pico em diff_pct)
Definições: **pico** = `max(diff_pct)`. **reversão** = após o pico, `diff_pct` cair pelo menos 0.50 p.p. (para Lay, subir 0.50 p.p. após o vale). `t_reversão` é o 1º tempo que cruza esse limiar (logo, não é o pico).

| Regime | N | t_pico médio (s) | t_pico p50 (s) | % melhora até fim | % com reversão | t_reversão médio (s) | Δt (rev - pico) médio (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRE_MATCH | 896 | 4.2 | 5.2 | 77.3% | 15.8% | 12.3 | 7.3 |
| IN_MATCH | 840 | 5.0 | 0.0 | 63.1% | 28.9% | 13.5 | 8.1 |

**Partição 100% (Back)**: categorias exclusivas (somam 100% por regime).

| Regime | % pico no fim, sem reversão | % pico no fim, com reversão (recuperou) | % pico antes, com reversão | % pico antes, sem reversão relevante |
|---|---:|---:|---:|---:|
| PRE_MATCH | 81.1% | 6.5% | 9.4% | 3.0% |
| IN_MATCH | 68.6% | 4.5% | 24.4% | 2.5% |

**Curva média (Back)**: média de `diff_pct` e `odd` por tempo. `CLV` é reportado **somente pre-match** (closing pré-jogo). `ROI` (quando aparece) é o **ROI realizado** se a aposta fosse feita naquele ponto.

| Tempo | N pts | diff_pct médio | odd média | CLV médio (pre-match) | ROI médio (se houver) |
|---|---:|---:|---:|---:|---:|
| t+0s | 1736 | +1.57% | 2.013 | +2.28% | 1.92 |
| t+6s | 1263 | +2.23% | 2.026 | +2.70% | 1.58 |
| t+10s | 2038 | +2.60% | 2.032 | +2.74% | 3.12 |
| t+15s | 1271 | +2.76% | 2.052 | +2.79% | 1.46 |
| t+20s | 1757 | +3.34% | 2.047 | +2.79% | 1.38 |

### 8.1b Back — impacto por timing (t0 vs pico vs último) e reversão
Leitura: se a estratégia é **entrar no pico**, compare CLV/ROI no pico vs t0 e veja diferença entre **casos com reversão** vs **sem reversão**. Aqui reportamos **média robusta por jogo + IC90**. Observação: **CLV só é calculado pre-match**.

**CLV (somente pre-match) — média robusta por jogo (IC90)**

| Subcoorte | N total | N CLV (PM) | CLV t0 (mean; IC90) | CLV pico (mean; IC90) | CLV último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 1351 | 626 | +1.73% [+1.20%, +2.26%] | +2.01% [+1.50%, +2.52%] | +2.00% [+1.49%, +2.51%] |
| COM_REVERSAO | 385 | 121 | +3.88% [+2.96%, +4.78%] | +5.42% [+4.38%, +6.46%] | +4.47% [+3.40%, +5.55%] |

**ROI (stake=1) — média robusta por jogo (IC90)**

| Subcoorte | N total | N ROI | ROI t0 (mean; IC90) | ROI pico (mean; IC90) | ROI último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 1351 | 1200 | +0.59% [-5.39%, +6.63%] | +0.71% [-5.30%, +6.85%] | +0.69% [-5.32%, +6.82%] |
| COM_REVERSAO | 385 | 331 | +4.06% [-7.04%, +15.64%] | +7.42% [-4.00%, +19.44%] | +4.17% [-6.86%, +15.50%] |

Nota interpretativa: a subcoorte **COM_REVERSAO** pode ter CLV maior no **pico** porque, por definição, ela inclui casos em que o edge atingiu um extremo mais alto antes de reverter. Para entender a hipótese 'sem reversão deveria ser maior', compare também a categoria '**pico no fim, sem reversão**' na partição 100% (tabela 8.1).

**Decomposição (pre-match): entry_odd vs closing_odd (IC90)**

| Subcoorte | N CLV (PM) | entry_odd t0 (mean; IC90) | entry_odd pico (mean; IC90) | closing_odd (mean; IC90) |
|---|---:|---:|---:|---:|
| SEM_REVERSAO | 626 | 1.983 [+1.971, +1.995] | 1.990 [+1.978, +2.002] | 1.969 [+1.961, +1.978] |
| COM_REVERSAO | 121 | 2.035 [+2.014, +2.055] | 2.066 [+2.043, +2.089] | 1.965 [+1.950, +1.980] |

### 8.2 Lay (vale em diff_pct)
| Regime | N | t_vale médio (s) | t_vale p50 (s) | % melhora até fim | % com reversão | t_reversão médio (s) | Δt (rev - vale) médio (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRE_MATCH | 786 | 4.0 | 5.2 | 71.6% | 19.6% | 12.6 | 7.0 |
| IN_MATCH | 676 | 5.5 | 0.0 | 51.3% | 38.9% | 13.5 | 7.9 |

**Partição 100% (Lay)**: categorias exclusivas (somam 100% por regime).

| Regime | % vale no fim, sem reversão | % vale no fim, com reversão (recuperou) | % vale antes, com reversão | % vale antes, sem reversão relevante |
|---|---:|---:|---:|---:|
| PRE_MATCH | 75.4% | 7.8% | 11.8% | 5.0% |
| IN_MATCH | 57.5% | 5.8% | 33.1% | 3.6% |

**Curva média (Lay)**: usa `odd=lay_odd`. `CLV` é reportado **somente pre-match** (closing pré-jogo). O ROI mostrado aqui é **ROI por liability** (não por stake), porque Lay é governado por risco.

| Tempo | N pts | diff_pct médio | lay_odd média | CLV médio (pre-match) | ROI/liability médio (se houver) |
|---|---:|---:|---:|---:|---:|
| t+0s | 1462 | +0.59% | 1.998 | +0.18% | 13.13 |
| t+6s | 1147 | +0.93% | 2.007 | +0.26% | 31.53 |
| t+10s | 1837 | +0.31% | 1.995 | +0.40% | 31.25 |
| t+15s | 1153 | +1.39% | 2.038 | +0.09% | 20.12 |
| t+20s | 1615 | +1.78% | 2.024 | +0.25% | 14.32 |

### 8.2b Lay — impacto por timing (t0 vs vale vs último) e reversão
Leitura: se a estratégia é **entrar no vale (odd mais baixa)**, compare CLV/ROI no vale vs t0 e veja diferença entre **casos com reversão** vs **sem reversão**. Aqui reportamos **média robusta por jogo + IC90**. Observação: **CLV só é calculado pre-match**.

**CLV (somente pre-match) — média robusta por jogo (IC90)**

| Subcoorte | N total | N CLV (PM) | CLV t0 (mean; IC90) | CLV vale (mean; IC90) | CLV último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 1045 | 523 | -0.28% [-0.93%, +0.39%] | +0.05% [-0.59%, +0.71%] | +0.03% [-0.62%, +0.69%] |
| COM_REVERSAO | 417 | 128 | +0.43% [-0.66%, +1.54%] | +1.80% [+0.77%, +2.87%] | +0.63% [-0.43%, +1.70%] |

**ROI/liability — média robusta por jogo (IC90)**

| Subcoorte | N total | N ROI | ROI/liab t0 (mean; IC90) | ROI/liab vale (mean; IC90) | ROI/liab último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 1045 | 940 | +8.08% [-0.70%, +17.12%] | +8.75% [-0.20%, +17.83%] | +8.70% [-0.23%, +17.75%] |
| COM_REVERSAO | 417 | 360 | +18.10% [+3.03%, +34.72%] | +27.08% [+6.78%, +51.88%] | +13.65% [+1.03%, +27.35%] |

---
### 8.3 Resumo de estratégias — 8 combinações (Side × Pre/In × Reversal)
Esta tabela resume as **8 combinações** possíveis: `Back/Lay × Pre/In × Reversal(Sim/Não)`.

- **CLV** aqui é medido em `t0` e **somente pre‑match** (closing pré‑jogo).
- Para **Lay**, o CLV nesta tabela usa a convenção única **(entry - closing)/closing**; logo, **Lay “bom” tende a CLV < 0**.
- **ROI** aqui é em `t0` (se houver placar). Para Lay, o ROI é **por liability**.
- **IC90** é bootstrap por jogo (cluster em `match_id`). Para critério “p30” usamos o quantil bootstrap **p30** do estimador.

| Side | Pre/In | Reversal | N | Jogos | CLV t0 (mean; IC90) | ROI t0 (mean; IC90) | ROI p30 | Ativa? (critério) |
|---|---|---|---:|---:|---:|---:|---:|---|
| Back | Pre | Yes | 142 | 85 | +3.88% [+2.96%, +4.78%] | +3.72% [-10.29%, +17.61%] | -1.24% | sim (CLV p90>0 AND ROI>0) |
| Back | Pre | No | 754 | 163 | +1.73% [+1.20%, +2.26%] | -4.96% [-12.85%, +2.75%] | -7.43% | não (CLV p90>0 AND ROI>0) |
| Back | In | Yes | 243 | 122 | — | +1.86% [-12.61%, +17.04%] | -2.67% | não (ROI p30>0) |
| Back | In | No | 597 | 215 | — | +2.57% [-5.42%, +10.51%] | -0.03% | não (ROI p30>0) |
| Lay | Pre | Yes | 154 | 83 | -0.43% [-1.54%, +0.66%] | -1.08% [-16.09%, +13.90%] | -5.78% | não (CLV p90<0 AND ROI p30>0) |
| Lay | Pre | No | 632 | 156 | +0.28% [-0.39%, +0.93%] | +5.36% [-3.14%, +13.88%] | +2.70% | não (CLV p90<0 AND ROI p30>0) |
| Lay | In | Yes | 263 | 138 | — | +32.04% [+11.04%, +55.32%] | +24.90% | sim (ROI p30>0) |
| Lay | In | No | 413 | 181 | — | +9.44% [-3.39%, +22.69%] | +5.04% | sim (ROI p30>0) |

Notas:
- Se você quiser operar **cada combinação como uma estratégia separada**, o sizing (Kelly/caps) deve ser estimado e governado separadamente por estratégia.
- Esta tabela é **in-sample** na janela (`--lookback-days`). A seção OOS (walk-forward) pode ser habilitada com `--walkforward`.

## 9) Combinações de valor (regime × linha AH × lag)
Tabela rankeada por volume (N). Métrica: diff_pct (OK, betslip confiável).

### 9.1 Back combos (diff_pct >= corte)
| Regime | Linha | Lag | N | Jogos | Mean diff % | IC90 cluster |
|---|---|---|---:|---:|---:|---|
| PRE_MATCH | AH 0-1 (líquida) | < 10s | 125 | 45 | +6.60% | [+6.53%, +7.15%] |
| IN_MATCH | AH 2+ (extrema) | < 10s | 119 | 89 | +6.17% | [+5.73%, +6.61%] |
| PRE_MATCH | AH 2+ (extrema) | < 10s | 75 | 36 | +6.94% | [+6.56%, +7.42%] |
| PRE_MATCH | AH 1-2 (média) | < 10s | 53 | 23 | +6.68% | [+6.57%, +7.42%] |
| IN_MATCH | AH 0-1 (líquida) | < 10s | 45 | 37 | +6.66% | [+6.06%, +7.19%] |
| IN_MATCH | AH 1-2 (média) | < 10s | 28 | 27 | +7.01% | [+6.16%, +7.66%] |
| PRE_MATCH | AH 0-1 (líquida) | 20-30s | 24 | 20 | +7.39% | [+6.76%, +7.81%] |
| IN_MATCH | AH 2+ (extrema) | > 30s | 22 | 16 | +6.53% | [+5.17%, +7.41%] |
| PRE_MATCH | AH 0-1 (líquida) | > 30s | 17 | 16 | +6.79% | [+6.02%, +7.51%] |
| PRE_MATCH | AH 2+ (extrema) | 20-30s | 16 | 11 | +7.03% | [+6.04%, +7.76%] |
| PRE_MATCH | AH 1-2 (média) | 20-30s | 13 | 10 | +5.90% | [+5.12%, +6.84%] |
| IN_MATCH | AH 2+ (extrema) | 20-30s | 12 | 10 | +6.66% | [+5.62%, +7.45%] |

### 9.2 Lay combos (diff_pct <= corte) + risco
| Regime | Linha | Lag | N | Jogos | Mean diff % | IC90 cluster | Liability p95 |
|---|---|---|---:|---:|---:|---|---:|
| IN_MATCH | AH 2+ (extrema) | < 10s | 73 | 60 | -4.99% | [-5.54%, -4.58%] | 308.36 |
| IN_MATCH | AH 0-1 (líquida) | < 10s | 34 | 28 | -5.06% | [-5.83%, -4.59%] | 454.08 |
| PRE_MATCH | AH 0-1 (líquida) | < 10s | 21 | 18 | -4.85% | [-6.18%, -4.24%] | 4386.45 |
| PRE_MATCH | AH 2+ (extrema) | < 10s | 18 | 18 | -4.99% | [-5.78%, -4.17%] | 898.86 |
| IN_MATCH | AH 2+ (extrema) | > 30s | 15 | 13 | -4.60% | [-5.66%, -3.72%] | 1438.89 |
| IN_MATCH | AH 1-2 (média) | < 10s | 14 | 13 | -4.31% | [-5.16%, -3.62%] | 660.98 |
| PRE_MATCH | AH 1-2 (média) | < 10s | 12 | 8 | -5.08% | [-6.61%, -3.85%] | 104.45 |
| IN_MATCH | AH 2+ (extrema) | 20-30s | 8 | 8 | -3.92% | [-5.03%, -2.84%] | 527.66 |
| IN_MATCH | AH 0-1 (líquida) | > 30s | 6 | 6 | -6.37% | [-7.74%, -5.16%] | 323.18 |
| IN_MATCH | AH 0-1 (líquida) | 20-30s | 4 | 4 | -3.48% | [-4.04%, -2.94%] | 170.81 |
| PRE_MATCH | AH 2+ (extrema) | > 30s | 3 | 3 | -4.42% | [-6.13%, -2.71%] | 91.55 |
| PRE_MATCH | AH 2+ (extrema) | 20-30s | 3 | 3 | -4.00% | [-5.14%, -2.86%] | 295.04 |

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

- **Back (stake)**: corr(exposição, ROI)=-0.090; corr(exposição, CLV)=0.004 (onde CLV existe).
- **Lay (liability)**: corr(exposição, ROI)=0.015; corr(exposição, CLV)=0.066 (onde CLV existe).

**Backtest de sizing (apenas eventos com placar; valores em unidade monetária *proxy*)**

| Lado | Scheme | N | Turnover (janela) | Lucro (janela) | ROI/turnover | p99 exposição | ES95 exposição | DD 30d (média) | DD 30d (p95) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Back | FLAT | 540 | 540.00 | 15.20 | 2.82% | 1.00 | 1.00 | 7.34 | 13.58 |
| Lay | FLAT | 192 | 245.34 | -7.33 | -2.99% | 1.00 | 1.00 | 54.08 | 85.63 |
| Back | PROXY | 540 | 207499.66 | -43545.02 | -20.99% | 4409.56 | 4419.69 | 220516.53 | 288358.94 |
| Lay | PROXY | 142 | 38504.01 | -3174.20 | -8.24% | 3444.73 | 2162.72 | 28478.69 | 51320.20 |
| Back | KELLY_0.10 | 264 | 6796.47 | 127.23 | 1.87% | 56.18 | 50.69 | 685.83 | 1346.31 |
| Lay | KELLY_0.10 | 40 | 524.11 | 57.45 | 10.96% | 20.56 | 20.56 | 33.69 | 60.95 |
| Back | KELLY_0.25 | 264 | 14199.88 | -47.39 | -0.33% | 87.99 | 87.99 | 1775.00 | 3349.67 |
| Lay | KELLY_0.25 | 40 | 765.02 | 107.81 | 14.09% | 20.56 | 20.56 | 33.80 | 59.00 |
| Back | KELLY_0.50 | 264 | 17621.76 | -204.35 | -1.16% | 87.99 | 87.99 | 2120.64 | 4043.18 |
| Lay | KELLY_0.50 | 40 | 839.04 | 104.89 | 12.50% | 20.56 | 20.56 | 24.16 | 42.73 |
| Back | KELLY_1.00 | 264 | 18248.18 | -164.19 | -0.90% | 87.99 | 87.99 | 2067.84 | 3838.44 |
| Lay | KELLY_1.00 | 40 | 854.25 | 113.99 | 13.34% | 20.56 | 20.56 | 15.00 | 27.46 |

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
| Back | Pre | Yes | FLAT | 71 | 71.00 | 4.83 | 6.81% | 1.00 | 24.73 |
| Back | Pre | Yes | PROXY | 71 | 28265.04 | -8668.12 | -30.67% | 5420.32 | 98956.88 |
| Back | Pre | Yes | KELLY_0.10 | 61 | 1691.64 | 60.56 | 3.58% | 44.86 | 480.73 |
| Back | Pre | Yes | KELLY_0.25 | 61 | 3768.80 | 86.69 | 2.30% | 87.99 | 1212.85 |
| Back | Pre | Yes | KELLY_0.50 | 61 | 4591.00 | 123.92 | 2.70% | 87.99 | 1410.33 |
| Back | Pre | Yes | KELLY_1.00 | 61 | 4696.42 | 112.17 | 2.39% | 87.99 | 1627.59 |
| Back | Pre | No | FLAT | 248 | 248.00 | -5.31 | -2.14% | 1.00 | 73.64 |
| Back | Pre | No | PROXY | 248 | 131387.15 | -24031.39 | -18.29% | 4491.02 | 197022.52 |
| Back | Pre | No | KELLY_0.10 | 203 | 5104.83 | 66.66 | 1.31% | 57.65 | 897.83 |
| Back | Pre | No | KELLY_0.25 | 203 | 10431.08 | -134.08 | -1.29% | 87.99 | 2473.30 |
| Back | Pre | No | KELLY_0.50 | 203 | 13030.77 | -328.27 | -2.52% | 87.99 | 3197.52 |
| Back | Pre | No | KELLY_1.00 | 203 | 13551.76 | -276.36 | -2.04% | 87.99 | 3028.11 |
| Back | In | Yes | FLAT | 58 | 58.00 | -4.52 | -7.80% | 1.00 | 49.33 |
| Back | In | Yes | PROXY | 58 | 9973.72 | -3099.84 | -31.08% | 1457.62 | 33967.78 |
| Back | In | No | FLAT | 163 | 163.00 | 20.21 | 12.40% | 1.00 | 0.00 |
| Back | In | No | PROXY | 163 | 37873.75 | -7745.68 | -20.45% | 2341.42 | 63284.14 |
| Lay | Pre | Yes | FLAT | 10 | 11.25 | 0.32 | 2.89% | 1.00 | 12.15 |
| Lay | Pre | Yes | PROXY | 10 | 6007.01 | -3757.61 | -62.55% | 4040.94 | 47247.53 |
| Lay | Pre | Yes | KELLY_0.10 | 7 | 85.55 | 4.53 | 5.30% | 20.28 | 164.45 |
| Lay | Pre | Yes | KELLY_0.25 | 7 | 131.16 | 16.16 | 12.32% | 20.56 | 137.71 |
| Lay | Pre | Yes | KELLY_0.50 | 7 | 161.41 | 22.29 | 13.81% | 20.56 | 142.24 |
| Lay | Pre | Yes | KELLY_1.00 | 7 | 163.71 | 24.60 | 15.02% | 20.56 | 136.30 |
| Lay | Pre | No | FLAT | 31 | 35.03 | 1.56 | 4.46% | 1.00 | 4.05 |
| Lay | Pre | No | PROXY | 31 | 10352.02 | -3715.30 | -35.89% | 3436.70 | 37870.17 |
| Lay | Pre | No | KELLY_0.10 | 22 | 323.03 | 18.22 | 5.64% | 20.56 | 35.61 |
| Lay | Pre | No | KELLY_0.25 | 22 | 456.41 | 42.01 | 9.20% | 20.56 | 2.31 |
| Lay | Pre | No | KELLY_0.50 | 22 | 484.77 | 34.78 | 7.17% | 20.56 | 14.14 |
| Lay | Pre | No | KELLY_1.00 | 22 | 484.77 | 34.78 | 7.17% | 20.56 | 14.14 |
| Lay | In | Yes | FLAT | 26 | 31.49 | 3.39 | 10.77% | 1.00 | 5.04 |
| Lay | In | Yes | PROXY | 26 | 5812.58 | 1881.87 | 32.38% | 1713.54 | 3036.37 |
| Lay | In | No | FLAT | 64 | 104.80 | -6.32 | -6.03% | 1.00 | 66.07 |
| Lay | In | No | PROXY | 64 | 14160.59 | 429.76 | 3.03% | 1391.25 | 13255.27 |
### 9.4 Estratégias candidatas (filtros + sizing recomendado)
Aqui consolidamos duas hipóteses operacionais do usuário e reportamos resultados esperados no recorte (e projeção simples para 30 dias). **Recomendação de sizing**: comece com `KELLY_0.25` + caps, e compare contra `FLAT` como baseline.

| Estratégia | Scheme | N Back (janela) | N Lay (janela) | N Back 30d (proj.) | N Lay 30d (proj.) | Stake médio Back | Stake médio Lay | Liability média Lay | Turnover 30d (proj.) | Lucro 30d (proj.) | ROI/turnover (janela) | ROI Lay/liability (janela) | ROI Lay/turnover (janela) | Banca p99 (Back) | Banca p99 (Lay) | Banca risco p99 (soma) | Banca liquidez p99 (+buf) | Banca recomendada (max) | ROI/banca 30d | Turnover 30d (R$) | Lucro 30d (R$) | Banca rec. (R$) | DD 30d p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BackFast (<5s, PM) + LayReversal (t_ext<=3s) | FLAT | 197 | 22 | 985 | 110 | 1.00 | 1.10 | 1.00 | 1105.68 | 41.86 | 3.79% | 2.97% | 2.71% | 1.00 | 1.00 | 2.00 | 158.47 | 158.47 | 26.42% | 5749.53 | 217.69 | 824.02 | 41.21 |
| BackFast (<5s, PM) + LayReversal (t_ext<=3s) | KELLY_0.25 | 167 | 18 | 835 | 90 | 55.83 | 17.99 | 15.79 | 48238.74 | 1373.86 | 2.85% | 2.63% | 2.31% | 87.99 | 20.56 | 108.54 | 6969.30 | 6969.30 | 19.71% | 250841.44 | 7144.07 | 36240.34 | 2028.67 |

Notas:
- **N Back/Lay** na tabela é **na janela observada**. As colunas `N 30d (proj.)` são uma **escala linear** por dias observados.
- `BackFast` implementa sua hipótese **Back só quando execução <5s** e pre‑match.
- `LayReversal` usa o diagnóstico da seção 8.2b: **subcoorte com reversão** e vale cedo (t_ext<=3s) como proxy de “logo após a reversão”.
- `Stake médio` e `Liability média` são **proxies** na unidade monetária do seu limit/finance; valores em **R$** usam fx `--fx-usdbrl`.
- `Banca recomendada` = max( banca por risco p99 (unitária, soma) ; banca por liquidez p99 (+buffer) ).
- **Por que Lay pode ter stake médio menor mesmo com ROI maior**: (i) Lay é governado por **liability** (risco) e usamos cap mais conservador; (ii) stake em Lay é derivado de `liability/(odd-1)` e depende do nível médio de odds; (iii) Kelly depende do **edge vs closing** (gap entry→closing), não do ROI observado isoladamente.
- **Por que não incluímos Back in‑match aqui**: Kelly/CLV usam `closing_odd` pré‑jogo; in‑match requer benchmark diferente (ex.: referência por minuto/VWAP) e governança operacional específica. A seção 2.2 já compara PRE_MATCH vs IN_MATCH.

### 9.4b Curva de capacidade — frações de Kelly e tamanho potencial
Esta tabela é um exercício para estimar **capacidade** (turnover/lucro/risco) ao variar a fração de Kelly. Ela deixa explícito quando o sizing satura por **cap por aposta**.

**Escala Kelly usada nesta curva**: P99_PROXY | ref_back=4399.26 ref_lay=2055.72 | cap_back=2.0% cap_lay=1.0% | max_stake_event=100%*limit

| Strategy | Scheme | cap_hit Back (%) | limit_hit Back (%) | cap_hit Lay (%) | limit_hit Lay (%) | Turnover 30d (proj.) | Lucro 30d (proj.) | ROI/banca 30d | DD 30d p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BackFast+LayReversal | KELLY_0.10 | 0.0% | 13.1% | 4.8% | 4.8% | 22547.38 | 775.04 | 23.63% | 903.65 |
| BackFast+LayReversal | KELLY_0.25 | 19.7% | 25.1% | 57.1% | 4.8% | 48238.74 | 1373.86 | 19.71% | 2034.65 |
| BackFast+LayReversal | KELLY_0.50 | 83.1% | 31.1% | 85.7% | 4.8% | 60579.68 | 1311.71 | 15.09% | 1484.66 |
| BackFast+LayReversal | KELLY_1.00 | 93.4% | 31.7% | 100.0% | 4.8% | 62471.72 | 1809.47 | 20.19% | 1381.80 |

Leitura rápida:
- Se `cap_hit` estiver alto, a alavancagem adicional (ex.: 0,50× ou 1,00×) não vira turnover — o cap está “travando” o tamanho.
- Se `limit_hit` estiver alto, você está batendo no **limit operacional** (capacidade da conta/mercado) — aumentar banca ou frac não aumenta turnover.
- Se `cap_hit` estiver baixo, a curva deve escalar quase linearmente com `frac` (até bater em limites reais/operacionais).
- Lay normalmente satura antes por ter cap mais conservador (liability) e por ter risco de cauda mais assimétrico.

### 9.4c Diagnóstico: Back in‑match (por que não está na estratégia candidata)
Aqui reportamos Back in‑match **apenas como diagnóstico**. Não incluímos in‑match na estratégia candidata porque (i) `closing_odd` pré‑jogo não é benchmark in‑match e (ii) o sizing Kelly acima depende desse benchmark.

| Regime | Scheme | N | Turnover (janela) | Lucro (janela) | ROI/turnover |
|---|---|---:|---:|---:|---:|
| IN_MATCH BackFast (<5s) | FLAT | 125 | 125.00 | 13.96 | 11.17% |
| IN_MATCH BackFast (<5s) | PROXY | 125 | 24593.24 | -2591.29 | -10.54% |

Próximo passo (se quiser operar in‑match): definir um benchmark de preço justo in‑match (ex.: referência por minuto, VWAP, ou odds externas) e calibrar sizing/risco específico do live.

## 12) OOS rolling-forward (walk-forward): seleção e validação
Até aqui o relatório é **in-sample** (na janela `--lookback-days`). Este bloco (opcional) faz um walk-forward simples por dia:

- Em cada passo, usamos os últimos `wf_train_days` para **selecionar** combinações com evidência de valor (IC90 lb>0).
- No(s) dia(s) seguinte(s) (`wf_test_days`), medimos o resultado OOS nas combinações ativas.

Isso aproxima o fluxo operacional que você descreveu (seleciona no rolling atual e mede no próximo rolling).

### 12.0 Diagnóstico de cobertura OOS (por que N cai)
| Filtro | Jogos únicos |
|---|---:|
| Combinações elegíveis (edge + timing + t0) | 231 |
| Com ROI disponível (precisa de placar) | 201 |
| Com CLV disponível (pre-match + closing) | 100 |

Leitura: o walk-forward mede OOS principalmente por **ROI**, então ele encolhe quando a cobertura de placar é baixa. Além disso, a métrica é agregada por **jogo único** (cluster), então você verá números menores que o N de eventos.

| Train window | Test window | #ativas | Jogos OOS | ROI OOS (mean; IC90) | Turnover (teste) | Lucro obs. | Lucro exp. |
|---|---|---:|---:|---:|---:|---:|---:|
| 2026-02-11→2026-02-13 | 2026-02-14→2026-02-14 | 1 | 44 | +13.53% [-10.44%, +39.86%] | 65.00 | 1.16 | 1.16 |
| 2026-02-12→2026-02-14 | 2026-02-15→2026-02-15 | 3 | 40 | +3.05% [-16.95%, +24.15%] | 1923.14 | -179.59 | -180.06 |

**Frequência de ativação por combinação (quantas janelas ela entrou como ativa)**

| Combinação | #steps ativa |
|---|---:|
| Back_In_No | 2 |
| Back_Pre_No | 1 |
| Back_Pre_Yes | 1 |

Notas importantes:
- Se `Jogos OOS` for baixo em muitos passos, você ainda não tem volume suficiente para decisões por combinação. Nesse cenário faz sentido **Bayes hierárquico (partial pooling)** para estabilizar estimativas.
- Este walk-forward usa ROI em t0 como avaliação OOS. Para pre-match, você também pode avaliar por CLV OOS (menos dependente de resultados), mas isso mede qualidade de entrada, não P&L.

### 12.1 Estimativa 30 dias (OOS): turnover, lucro, banca, ROI/banca e drawdown
Esta estimativa usa o walk-forward acima como **simulador OOS**. O lucro pode ser reportado em duas versões:

- **obs.**: apenas jogos com ROI (placar) disponível.
- **exp.**: expande o lucro para a população elegível usando scaling por exposição/turnover (assume missing-at-random condicional à estratégia).

| Premissa | Valor |
|---|---:|
| Scheme pre-match (OOS) | `KELLY_0.25` |
| Scheme in-match (OOS) | `FLAT` |
| Expansão missing ROI | ON |
| Dias OOS usados | 2 |
| Turnover 30d (proj.) | 29822.12 |
| Lucro 30d (obs.) | -2676.55 |
| Lucro 30d (exp.) | -2683.57 |
| Banca risco p99 (Back+Lay) | 1904.56 |
| Banca liquidez p99 (+buf) | 1680.70 |
| Banca recomendada (max) | 1904.56 |
| ROI/banca 30d (obs.) | -140.53% |
| ROI/banca 30d (exp.) | -140.90% |
| DD 30d p95 (obs.) | 3401.86 |
| DD 30d p95 (exp.) | 3411.92 |

## 10) Diagnóstico: por que o ROI pode estar zerado
ROI aqui é calculado por placar do jogo (`matches.home_score/away_score`). Se os placares não estiverem preenchidos no banco, a cobertura de ROI será 0.

| Indicador | Valor |
|---|---:|
| Jogos únicos no recorte | 299 |
| Jogos com placar disponível (home_score/away_score não nulos) | 252 |
| Jogos com status='finished' no banco | 252 |

### 10.1 Distribuição por data de kickoff (explica janela da API)
- Kickoff (UTC) no recorte: **2026-02-11 15:00 UTC** até **2026-02-17 08:00 UTC**.

| Kickoff date (UTC) | Jogos | Com placar | Cobertura |
|---|---:|---:|---:|
| 2026-02-17 | 2 | 1 | 50.0% |
| 2026-02-16 | 24 | 18 | 75.0% |
| 2026-02-15 | 75 | 70 | 93.3% |
| 2026-02-14 | 114 | 104 | 91.2% |
| 2026-02-13 | 50 | 33 | 66.0% |
| 2026-02-12 | 5 | 2 | 40.0% |
| 2026-02-11 | 29 | 24 | 82.8% |

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
