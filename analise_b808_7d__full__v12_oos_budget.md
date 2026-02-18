# Análise Estatística Robusta — Contexto Operação (b808)
**Data da execução:** 18/02/2026 20:50 UTC  
**Escopo:** H3B (auditoria), com comparação por `audit_version` e inferência robusta por jogo (cluster bootstrap).  
**Nota:** a robustez aqui significa que intervalos de confiança consideram correlação intra-jogo (múltiplas auditorias por partida).
---
## 0) Sumário executivo (leitura rápida)
- **Recorte**: direction=`up`, lookback_days=`7`, versions=`v4.0-api,v1.0,v1.0-recovered`.
- **Amostra**: 4009 auditorias (jogos únicos=333, média=12.0 obs/jogo); betslip confiável=1813.
- **Janela efetiva (audited_at)**: 11/02 20:51 → 18/02 20:39 UTC (span≈7.0d; dias com dados=8).
- **Coortes (status=OK, betslip confiável)**: `BS>WS` (diff>=2.0%): **643**; `BS<WS` (diff<=-2.0%): **207**.
- **Coberturas em `hypothesis_details` (OK)**: temporal(back)=1501/1813; lay_temporal=1358/1813; finance=1269/1813.
- **Cobertura de placar (ROI)**: jogos com placar=284/333 (status finished=284).
- **Cobertura de closing_odd (AH)**: jogos com closing=197/333 (59.2%). CLV pre‑match depende disso.
- **Alerta**: cobertura de closing_odd está baixa. Isso tende a reduzir N de CLV e pode enviesar a leitura (os jogos “sem odds” ficam fora do CLV). Priorize estabilizar o collector e/ou condicionar análises a jogos com closing.
- **DOM**: sem dados no recorte atual (N=0), então não há comparação API vs DOM aqui.
- **CLV pre-match (Betslip, API)**: média robusta por jogo +2.126% (IC90 [+1.656%, +2.600%]), com N=939 eventos (jogos=158).
- **Padrão por bucket (CLV PM)**: `BS < WS` -3.920% (sig. negativo), `BS ~ WS` -0.123% (NS), `BS > WS` +6.532% (sig. positivo).
- **Leitura recomendada**: use este relatório para validar **qualidade de execução/CLV**. Para concluir sobre **ROI**, rode após atualizar resultados e/ou use uma janela com jogos já liquidados.

---
## 1) Contexto do corte (b808)
| Indicador | Valor |
|---|---:|
| Auditorias H3B `UP` (match + kickoff passado) | 4009 |
| Betslip bruto | 2875 |
| Betslip confiável (diff -10% a +10%) | 1813 |
| Descartados no filtro de qualidade | 1062 |
| Jogos únicos (geral) | 333 |
| Média de observações por jogo | 12.0 |
| Jogos únicos com betslip confiável | 299 |
| Distribuição por market_type | AH=4009 |
| Jogos únicos (AH) no recorte | 333 |
| Jogos únicos (AH) com closing_odd disponível | 197 |
| Cobertura closing_odd (AH) | 59.2% |

---
## 2) Base comparativa: API vs DOM
| Métrica | API (v4.0-api) | DOM (v1.0) |
|---|---:|---:|
| Total de observações | 4009 | 0 |
| Com betslip confiável | 1813 | 0 |
| Com CLV pre-match (betslip) | 939 | 0 |
| Com ROI (betslip) | 1570 | 0 |
| Tempo total observado (detecção→betslip, wall/total) | 12434 ms | — ms |
| Tempo instrumentado (detecção→clique→betslip) | 8800 ms | — ms |

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
| API (2-4s) | lag_det→click | 5581 | 797 | 7170 | 4007 |
| API (2-4s) | lag_click→betslip | 2626 | 2168 | 4505 | 3416 |
| API (2-4s) | lag_e2e (soma) | 8800 | 3460 | 11077 | 3414 |
| API (2-4s) | audit_total (duração) | 12430 | 4551 | 31416 | 4009 |
| API (2-4s) | overhead (total - e2e) | 4889 | 204 | 22834 | 3414 |
| DOM (15-30s) | lag_det→click | — | — | — | 0 |
| DOM (15-30s) | lag_click→betslip | — | — | — | 0 |
| DOM (15-30s) | lag_e2e (soma) | — | — | — | 0 |
| DOM (15-30s) | audit_total (duração) | — | — | — | 0 |
| DOM (15-30s) | overhead (total - e2e) | — | — | — | 0 |

**Diagnóstico de cauda (percentual acima do limiar)**

| Modelo | % det→click > 5s | % det→click > 20s | % total > 10s | % total > 40s | % overhead < 0 |
|---|---:|---:|---:|---:|---:|
| API (2-4s) | 5.7% | 3.3% | 13.0% | 3.6% | 0.0% |
| DOM (15-30s) | —% | —% | —% | —% | —% |

---
### 2.1 Cobertura temporal (pre-match vs in-match)
| Métrica | Pre-match | In-match | Observação |
|---|---:|---:|---|
| Observações totais com classificação temporal | 1858 | 2151 | Contagem bruta do corte |
| ROI Betslip | 925 | 645 | Amostra com resultado do jogo |
| ROI WebSocket | 1464 | 1828 | Referência de mercado |
| CLV (apenas pre-match) | 939 | — | CLV vs closing pré-jogo não é interpretável in-match |

---
### 2.2 Performance por regime (pre-match vs in-match)
| Regime | N auditorias | N betslip conf. | N OK | Back edge | Lay edge | Diff médio (OK) |
|---|---:|---:|---:|---:|---:|---:|
| PRE_MATCH | 1858 | 1048 | 1048 | 415 | 69 | +2.230% |
| IN_MATCH | 2151 | 765 | 765 | 228 | 138 | +0.924% |

---
### 2.3 Regimes operacionais por tempo total (bucket)
Objetivo: separar a amostra em **regimes de execução** (tempo total) e medir performance em cada regime. Use isso para detectar **fila/saturação**: regimes lentos tendem a ter edge menor e pior ROI.

| Bucket (tempo total) | N OK | Jogos | lag_total mean (ms) | lag_total p95 (ms) | overhead p95 (ms) | Back edge | Lay edge | CLV PM (mean; IC90) | ROI (mean; IC90) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| < 5s | 1287 | 270 | 3597 | 4851 | 2506 | 433 | 132 | +2.34% [+1.92%, +2.76%] | +4.47% [-1.41%, +10.36%] |
| 5-10s | 234 | 148 | 5887 | 8212 | 2824 | 98 | 45 | +1.80% [+0.88%, +2.73%] | -2.71% [-14.64%, +9.28%] |
| 10-20s | 18 | 17 | 13668 | 18117 | 4070 | 9 | 2 | +1.17% [-1.58%, +3.89%] | +10.43% [-23.26%, +43.02%] |
| 20-40s | 196 | 101 | 27171 | 33595 | 29908 | 67 | 21 | +2.34% [+1.39%, +3.32%] | +7.10% [-6.28%, +20.38%] |
| > 40s | 78 | 57 | 184409 | 505379 | 387084 | 36 | 7 | +2.54% [+1.37%, +3.68%] | -11.38% [-31.71%, +8.91%] |
| Desconhecido | 0 | 0 | — | — | — | 0 | 0 | — | — |

---
### 2.3b Regimes por tempo total — separando coortes por `diff_pct` (BS vs WS)
Nesta tabela, separamos duas coortes operacionais por **delta de execução**: `BS > WS` (diff_pct >= +2%) e `BS < WS` (diff_pct <= -2%). Isso **não** é (por si só) “Back vs Lay”; é um recorte por **melhora/piora do preço** entre detecção (WS) e execução (BS). CLV é reportado apenas em pre‑match.

| Bucket | N OK | N (BS>WS +2%) | N (BS<WS -2%) | CLV PM (BS>WS) | CLV PM (BS<WS) | ROI (BS>WS) | ROI (BS<WS) |
|---|---:|---:|---:|---:|---:|---:|---:|
| < 5s | 1287 | 433 | 132 | +6.92% [+6.48%, +7.34%] | -3.72% [-4.89%, -2.53%] | +4.41% [-5.62%, +14.60%] | +1.04% [-13.64%, +16.06%] |
| 5-10s | 234 | 98 | 45 | +5.55% [+4.71%, +6.34%] | -5.91% [-7.17%, -4.54%] | -5.65% [-23.36%, +12.05%] | -8.67% [-29.11%, +12.24%] |
| 10-20s | 18 | 9 | 2 | +3.86% [+1.34%, +6.62%] | -6.98% — | -10.16% [-55.47%, +36.27%] | +40.04% [+0.00%, +79.90%] |
| 20-40s | 196 | 67 | 21 | +7.43% [+6.38%, +8.48%] | -1.06% [-3.91%, +2.13%] | +2.78% [-17.64%, +23.11%] | -11.33% [-49.36%, +28.71%] |
| > 40s | 78 | 36 | 7 | +5.38% [+3.68%, +7.04%] | -4.80% [-7.29%, -2.29%] | -23.83% [-53.30%, +7.72%] | -49.64% [-100.00%, +0.00%] |
| Desconhecido | 0 | 0 | 0 | — | — | — | — |

---
### 2.3c Estabilidade temporal (por dia, `audited_at`)
Objetivo: checar se o regime de edge/execução é **time‑dependent**. Se houver dias com comportamento distinto (ex.: collector intermitente, horários/ligas), isso aparecerá aqui.

| Dia (UTC) | Modelo | N OK | Jogos | % BS>WS +2% | % BS<WS -2% | lag_total p50 (ms) | CLV PM (BS>WS) | CLV PM (BS<WS) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-02-11 | API (2-4s) | 4 | 3 | 50.0% | 25.0% | 46520 | — | — |
| 2026-02-12 | API (2-4s) | 55 | 44 | 36.4% | 9.1% | 26856 | +5.58% | -4.34% |
| 2026-02-13 | API (2-4s) | 578 | 140 | 36.2% | 10.4% | 4822 | +6.96% | -3.22% |
| 2026-02-14 | API (2-4s) | 506 | 140 | 35.0% | 15.2% | 4207 | +6.25% | -5.29% |
| 2026-02-15 | API (2-4s) | 444 | 102 | 32.0% | 11.5% | 3691 | +6.40% | -2.66% |
| 2026-02-16 | API (2-4s) | 226 | 66 | 41.2% | 5.8% | 3445 | +6.43% | -6.79% |

---
## 3) CLV pre-match (núcleo)
### 3.1 CLV com odd do Betslip (execução real)
| Métrica | API | DOM |
|---|---:|---:|
| CLV Bruto BS Pre-Match | +2.339% (sig. positivo, N=939, jogos=158) | — (N/A, N=0, jogos=0) |
| CLV Adicional BS Pre-Match | +1.754% (sig. positivo, N=939, jogos=158) | — (N/A, N=0, jogos=0) |
| Taxa de CLV > 0 (bruto) | 61.2% | —% |
| Taxa de CLV > 0 (adicional) | 61.9% | —% |

Notas de robustez (IC 90% por jogo):  
- API CLV bruto (cluster): média +2.126%; IC90 [+1.656%, +2.600%]  
- DOM CLV bruto (cluster): média —; IC90 —  

---
## 4) ROI por modelo
| Métrica | API | DOM |
|---|---:|---:|
| ROI Betslip | +1.500% (NS, N=1568) | — (N/A, N=0) |
| ROI WebSocket | -0.657% (NS, N=3277) | — (N/A, N=0) |
| Win rate ROI Betslip | 50.8% | —% |
| Win rate ROI WS | 50.1% | —% |

Notas de robustez (IC 90% por jogo):  
- API ROI betslip (cluster): média +2.021%; IC90 [-3.228%, +7.251%]  
- API ROI WS (cluster): média -2.649%; IC90 [-5.545%, +0.367%]  

---
## 5) Diferença de preço BS vs WS
| Métrica | API | DOM |
|---|---:|---:|
| Diff BS vs WS (média) | +1.679% (sig. positivo, N=1813) | — (N/A, N=0) |
| BS > WS | 47.7% (865/1813) | —% (0/0) |
| BS > WS +2% | 35.5% (643/1813) | —% (0/0) |

---
## 6) Combinações de valor
### 6.1 Buckets por diferença BS vs WS
Nota de leitura:
- `BS > WS (+2% a +10%)`: preço no betslip ficou **melhor** do que o WS observado na detecção (delta positivo entre instantes).
- `BS < WS (-10% a -2%)`: preço no betslip ficou **pior** do que o WS observado na detecção (delta negativo entre instantes).
- O ROI abaixo é calculado **dentro do bucket** (não mistura buckets), mas ainda é sensível à cobertura de placar.

| Bucket | N bucket | CLV BS PM (média) | IC90 (cluster) | N CLV PM | Jogos CLV PM | ROI BS (todos) | IC90 (cluster) |
|---|---:|---:|---|---:|---:|---:|---|
| BS < WS (-10% a -2%) | 207 | -3.920% | [-4.884%, -3.048%] | 59 | 46 | +0.790% | [-16.075%, +7.786%] |
| BS ~ WS (-2% a +2%) | 963 | -0.123% | [-0.535%, +0.193%] | 499 | 143 | +1.110% | [-9.361%, +3.624%] |
| BS > WS (+2% a +10%) | 643 | +6.532% | [+6.382%, +7.155%] | 381 | 119 | +2.307% | [-2.603%, +15.329%] |

---
### 6.2 Combinação por faixa de linha AH
| Faixa AH | CLV BS PM (média) | IC90 (cluster) | ROI BS (todos) | IC90 (cluster) | Diff BS vs WS (média) |
|---|---:|---|---:|---|---:|
| AH 0-1 (líquida) | +2.577% | [+2.128%, +3.483%] | +1.694% | [+1.556%, +16.410%] | +1.880% |
| AH 1-2 (média) | +2.455% | [+1.264%, +3.429%] | +0.667% | [-13.637%, +12.443%] | +2.381% |
| AH 2+ (extrema) | +1.850% | [+0.336%, +1.970%] | +1.665% | [-7.747%, +8.920%] | +1.222% |

---
### 6.3 Combinação por faixa de lag
| Faixa de lag | CLV BS PM (média) | IC90 (cluster) | N CLV PM | Jogos CLV PM | ROI BS (todos) | IC90 (cluster) | Diff BS vs WS (média) |
|---|---:|---|---:|---:|---:|---|---:|
| < 10s | +2.381% | [+1.714%, +2.566%] | 738 | 151 | +1.677% | [-2.524%, +9.052%] | +1.649% |
| 10-20s | +1.220% | [-1.579%, +3.893%] | 7 | 7 | +9.839% | [-23.261%, +43.022%] | +1.793% |
| 20-30s | +2.267% | [+1.370%, +3.283%] | 123 | 72 | +5.139% | [-7.774%, +20.212%] | +1.801% |
| > 30s | +2.131% | [+1.404%, +3.539%] | 71 | 48 | -9.149% | [-26.924%, +7.369%] | +1.890% |

---
## 7) Estimativa financeira (proxy) e risco
Este bloco usa `hypothesis_details.finance` quando existe; se não existir, usa stake fallback = `stake_pct_of_limit × limit`.

**Política de stake (proxy)**

| Parâmetro | Valor |
|---|---:|
| stake_pct_of_limit | 0.25 |
| stake_cap | 0.00 |
| Cobertura finance (OK, betslip conf.) | 1269/1813 |

### 7.1 Back (BS >> WS)
| Métrica | Valor |
|---|---:|
| Corte (diff_pct) | >= 2.0% |
| N eventos | 643 |
| Cobertura finance (na coorte) | 446/643 |
| Stake total (estimado) | 201429.30 |
| Stake médio | 313.26 |
| Profit_if_win total (estimado) | 216171.33 |
| Profit_if_win médio | 336.19 |
| N com ROI realizado | 558 |
| P&L realizado total (estimado) | -34200.36 |
| ROI realizado (ponderado por stake) | -17.72% |
| ROI realizado (robusto por jogo, mean; IC90) | +6.39% [-2.60%, +15.33%] |
| ROI ponderado por stake (robusto por jogo, mean; IC90) | +6.10% [-4.97%, +17.89%] |

**Como ler as 3 linhas de ROI (Back)**

- **ROI realizado (ponderado por stake)**: ΣP&L / Σstake (pode ser dominado por stakes grandes).
- **ROI realizado (robusto por jogo)**: média por jogo (cada jogo pesa parecido; reduz dominância de outliers).
- **ROI ponderado por stake (robusto por jogo)**: calcula ROI ponderado dentro do jogo e depois faz média/IC por jogo.

Observação: a Seção 4 reporta ROI médio no **recorte completo** (betslip confiável). Aqui (7.1) é apenas a coorte **Back edge** (diff>=corte) e o ROI também é mostrado ponderado por stake; por isso sinais podem divergir.

### 7.2 Lay (BS << WS) — risco de cauda
| Métrica | Valor |
|---|---:|
| Corte (diff_pct) | <= -2.0% |
| N eventos | 207 |
| Cobertura finance (na coorte) | 148/207 |
| Stake total (estimado) | 34519.40 |
| Liability total (estimada) | 30427.54 |
| Liability média | 146.99 |
| Liability p95 | 527.22 |
| Liability p99 | 2036.19 |
| ES95 (liability) | 1568.72 |
| Liability max | 4395.35 |
| Proxy de banca (>= p99 liability) | 2036.19 |
| N com ROI realizado | 125 |
| P&L realizado total (estimado) | -7730.98 |
| ROI realizado (ponderado por liability) | -27.38% |
| ROI realizado (ponderado por stake) | -24.05% |
| ROI/liability (robusto por jogo, mean; IC90) | +17.29% [+0.90%, +33.24%] |
| ROI/liability ponderado (robusto por jogo, mean; IC90) | +15.53% [-1.00%, +32.15%] |

### 7.3 Projeção mensal (30 dias fixo) — turnover, lucro e banca
Premissas:
- **Mês = 30 dias fixo**.
- Turnover = soma de stakes (Back e Lay). Para Lay, também reportamos exposição por **liability**.
- **Banca por risco (unitária)** = p99/ES95(exposição por aposta).
- **Banca por liquidez (turnover)** = p95/p99 de capital **simultaneamente travado** (stake/liability em aberto até a liquidação), capturando distribuição desigual de entradas ao longo do tempo.
- Projeção de lucro usa duas visões: (i) **lucro/dia observado** e (ii) **ROI observado × turnover projetado**.

| Bloco | Janela(d) | Turnover 30d | Lucro 30d (direto) | Lucro 30d (ROI×turnover) |
|---|---:|---:|---:|---:|
| Back | 7.0 | 863268.44 | -146572.96 | -152982.85 |
| Lay (stake) | 7.0 | 147940.26 | -33132.76 | -35578.51 |
| Total (Back+Lay) | 7.0 | 1011208.70 | -179705.71 | -188561.36 |

**Banca por risco (exposição unitária)**

| Bloco | Banca conservadora (p99) | Banca agressiva (ES95) | ROI/banca 30d (direto) | ROI/banca 30d (ROI×turnover) |
|---|---:|---:|---:|---:|
| Back (stake) | 4394.23 | 3831.80 | -3335.58% | -3481.45% |
| Lay (liability) | 2036.19 | 1568.72 | -1627.20% | -1747.31% |
| Total (soma) | 6430.41 | 5400.52 | -2794.62% | -2932.34% |

**Banca por liquidez (capital simultaneamente travado)**

Definição operacional: cada aposta trava capital de `audited_at` até `kickoff + 2.00h + 2.25h` (grid=5min). A banca recomendada aplica buffer de +10%.

| Bloco | Liquidez mean | Liquidez p95 | Liquidez p99 | Liquidez max | Banca liq p99 (+buffer) |
|---|---:|---:|---:|---:|---:|
| Back (stake) | 17685.31 | 57199.40 | 82921.74 | 88455.79 | 91213.91 |
| Lay (liability) | 2293.54 | 7295.91 | 9702.60 | 11832.35 | 10672.86 |
| Total (Back+Lay) | 19969.51 | 61605.01 | 89284.53 | 95803.56 | 98212.98 |

**Banca recomendada (conservadora)**

| Métrica | Valor |
|---|---:|
| Banca por risco (p99 unitário, soma) | 6430.41 |
| Banca por liquidez (p99 simultâneo + buffer) | 98212.98 |
| Banca efetiva (max das duas) | 98212.98 |
| ROI/banca 30d (direto, banca efetiva) | -182.98% |
| ROI/banca 30d (ROI×turnover, banca efetiva) | -191.99% |
**Diagnóstico de cobertura (placar/ROI)**

| Bloco | Turnover total | Turnover com ROI | Cobertura turnover |
|---|---:|---:|---:|
| Back | 201429.30 | 192989.53 | 95.81% |
| Lay | 34519.40 | 32146.45 | 93.13% |

Notas (Lay): exposição 30d por liability (não é turnover) = 130403.76; ROI realizado por liability (ponderado) = -27.38%.

---
## 8) Curva temporal (pico, reversão e melhor timing)
Esta seção usa `hypothesis_details.temporal` (Back) e `hypothesis_details.lay_temporal` (Lay), coletados em pontos discretos (t≈0,3,6,10,15,20s).

O objetivo é responder: **tempo até o pico/vale**, **% que segue melhorando até t_max**, **% com reversão** e **impacto esperado por timing**. **CLV é reportado somente pre-match** (closing pré-jogo).

### 8.1 Back (pico em diff_pct)
Definições: **pico** = `max(diff_pct)`. **reversão** = após o pico, `diff_pct` cair pelo menos 0.50 p.p. (para Lay, subir 0.50 p.p. após o vale). `t_reversão` é o 1º tempo que cruza esse limiar (logo, não é o pico).

| Regime | N | t_pico médio (s) | t_pico p50 (s) | % melhora até fim | % com reversão | t_reversão médio (s) | Δt (rev - pico) médio (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRE_MATCH | 1048 | 4.5 | 5.2 | 76.2% | 17.0% | 12.2 | 7.1 |
| IN_MATCH | 765 | 5.5 | 0.0 | 59.5% | 31.8% | 13.5 | 8.1 |

**Partição 100% (Back)**: categorias exclusivas (somam 100% por regime).

| Regime | % pico no fim, sem reversão | % pico no fim, com reversão (recuperou) | % pico antes, com reversão | % pico antes, sem reversão relevante |
|---|---:|---:|---:|---:|
| PRE_MATCH | 80.0% | 7.4% | 9.5% | 3.1% |
| IN_MATCH | 65.5% | 5.0% | 26.8% | 2.7% |

**Curva média (Back)**: média de `diff_pct` e `odd` por tempo. `CLV` é reportado **somente pre-match** (closing pré-jogo). `ROI` (quando aparece) é o **ROI realizado** se a aposta fosse feita naquele ponto.

| Tempo | N pts | diff_pct médio | odd média | CLV médio (pre-match) | ROI médio (se houver) |
|---|---:|---:|---:|---:|---:|
| t+0s | 1813 | +1.68% | 2.017 | +2.34% | 2.26 |
| t+6s | 1473 | +2.28% | 2.024 | +2.65% | 1.52 |
| t+10s | 2393 | +2.62% | 2.029 | +2.69% | 2.93 |
| t+15s | 1487 | +2.75% | 2.047 | +2.78% | 1.98 |
| t+20s | 2030 | +3.25% | 2.044 | +2.76% | 0.96 |

### 8.1b Back — impacto por timing (t0 vs pico vs último) e reversão
Leitura: se a estratégia é **entrar no pico**, compare CLV/ROI no pico vs t0 e veja diferença entre **casos com reversão** vs **sem reversão**. Aqui reportamos **média robusta por jogo + IC90**. Observação: **CLV só é calculado pre-match**.

**CLV (somente pre-match) — média robusta por jogo (IC90)**

| Subcoorte | N total | N CLV (PM) | CLV t0 (mean; IC90) | CLV pico (mean; IC90) | CLV último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 1392 | 782 | +1.76% [+1.33%, +2.20%] | +2.04% [+1.60%, +2.46%] | +2.02% [+1.59%, +2.45%] |
| COM_REVERSAO | 421 | 157 | +4.03% [+3.22%, +4.85%] | +5.48% [+4.57%, +6.39%] | +4.62% [+3.68%, +5.55%] |

**ROI (stake=1) — média robusta por jogo (IC90)**

| Subcoorte | N total | N ROI | ROI t0 (mean; IC90) | ROI pico (mean; IC90) | ROI último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 1392 | 1210 | +1.90% [-4.26%, +8.15%] | +2.07% [-4.07%, +8.36%] | +2.05% [-4.10%, +8.34%] |
| COM_REVERSAO | 421 | 358 | +0.12% [-10.40%, +10.97%] | +3.18% [-7.81%, +14.34%] | +0.25% [-10.24%, +11.05%] |

Nota interpretativa: a subcoorte **COM_REVERSAO** pode ter CLV maior no **pico** porque, por definição, ela inclui casos em que o edge atingiu um extremo mais alto antes de reverter. Para entender a hipótese 'sem reversão deveria ser maior', compare também a categoria '**pico no fim, sem reversão**' na partição 100% (tabela 8.1).

**Decomposição (pre-match): entry_odd vs closing_odd (IC90)**

| Subcoorte | N CLV (PM) | entry_odd t0 (mean; IC90) | entry_odd pico (mean; IC90) | closing_odd (mean; IC90) |
|---|---:|---:|---:|---:|
| SEM_REVERSAO | 782 | 1.983 [+1.972, +1.994] | 1.991 [+1.979, +2.002] | 1.964 [+1.955, +1.972] |
| COM_REVERSAO | 157 | 2.035 [+2.017, +2.054] | 2.064 [+2.044, +2.085] | 1.961 [+1.948, +1.974] |

### 8.2 Lay (vale em diff_pct)
| Regime | N | t_vale médio (s) | t_vale p50 (s) | % melhora até fim | % com reversão | t_reversão médio (s) | Δt (rev - vale) médio (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRE_MATCH | 928 | 4.3 | 5.2 | 69.7% | 21.4% | 12.6 | 7.0 |
| IN_MATCH | 608 | 6.1 | 5.3 | 45.9% | 43.3% | 13.5 | 7.9 |

**Partição 100% (Lay)**: categorias exclusivas (somam 100% por regime).

| Regime | % vale no fim, sem reversão | % vale no fim, com reversão (recuperou) | % vale antes, com reversão | % vale antes, sem reversão relevante |
|---|---:|---:|---:|---:|
| PRE_MATCH | 73.3% | 8.0% | 13.5% | 5.3% |
| IN_MATCH | 52.8% | 6.4% | 36.8% | 3.9% |

**Curva média (Lay)**: usa `odd=lay_odd`. `CLV` é reportado **somente pre-match** (closing pré-jogo). O ROI mostrado aqui é **ROI por liability** (não por stake), porque Lay é governado por risco.

| Tempo | N pts | diff_pct médio | lay_odd média | CLV médio (pre-match) | ROI/liability médio (se houver) |
|---|---:|---:|---:|---:|---:|
| t+0s | 1536 | +0.68% | 2.000 | +0.04% | 10.69 |
| t+6s | 1334 | +0.84% | 2.002 | +0.18% | 27.31 |
| t+10s | 2154 | +0.31% | 1.991 | +0.29% | 26.85 |
| t+15s | 1345 | +1.23% | 2.028 | +0.04% | 16.67 |
| t+20s | 1858 | +1.57% | 2.018 | +0.17% | 12.84 |

### 8.2b Lay — impacto por timing (t0 vs vale vs último) e reversão
Leitura: se a estratégia é **entrar no vale (odd mais baixa)**, compare CLV/ROI no vale vs t0 e veja diferença entre **casos com reversão** vs **sem reversão**. Aqui reportamos **média robusta por jogo + IC90**. Observação: **CLV só é calculado pre-match**.

**CLV (somente pre-match) — média robusta por jogo (IC90)**

| Subcoorte | N total | N CLV (PM) | CLV t0 (mean; IC90) | CLV vale (mean; IC90) | CLV último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 1074 | 648 | -0.43% [-1.03%, +0.19%] | -0.09% [-0.68%, +0.52%] | -0.11% [-0.71%, +0.49%] |
| COM_REVERSAO | 462 | 173 | +0.15% [-0.76%, +1.07%] | +1.46% [+0.53%, +2.37%] | +0.26% [-0.65%, +1.18%] |

**ROI/liability — média robusta por jogo (IC90)**

| Subcoorte | N total | N ROI | ROI/liab t0 (mean; IC90) | ROI/liab vale (mean; IC90) | ROI/liab último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 1074 | 937 | +5.79% [-2.57%, +14.75%] | +6.51% [-2.00%, +15.60%] | +6.46% [-2.02%, +15.52%] |
| COM_REVERSAO | 462 | 396 | +18.39% [+4.64%, +33.88%] | +26.65% [+8.23%, +49.16%] | +14.35% [+2.29%, +26.55%] |

---
### 8.3 Resumo de estratégias — 8 combinações (Side × Pre/In × Reversal)
Esta tabela resume as **8 combinações** possíveis: `Back/Lay × Pre/In × Reversal(Sim/Não)`.

- **Back**: entrada em `t0`.
- **Lay**: entrada **após reversão** quando ela existe (`odd_reversal`), senão no **último ponto** (~t+20s).
- **CLV** aqui é **somente pre‑match** (closing pré‑jogo). Para **Lay**, reportamos CLV na convenção única **(entry - closing)/closing**; logo, **Lay “bom” tende a CLV < 0**.
- **ROI** é calculado no **ponto de entrada da estratégia** (se houver placar). Para Lay, ROI é **por liability**.
- **IC90** é bootstrap por jogo (cluster em `match_id`). Para critério “p30” usamos o quantil bootstrap **p30** do estimador.

| Side | Pre/In | Reversal | N | Jogos | CLV t0 (mean; IC90) | ROI t0 (mean; IC90) | ROI p30 | Ativa? (critério) |
|---|---|---|---:|---:|---:|---:|---:|---|
| Back | Pre | Yes | 178 | 109 | +4.03% [+3.22%, +4.85%] | -3.25% [-16.32%, +9.90%] | -7.60% | não (CLV p90>0 AND ROI>0) |
| Back | Pre | No | 870 | 176 | +1.76% [+1.33%, +2.20%] | -2.19% [-10.08%, +5.59%] | -4.67% | não (CLV p90>0 AND ROI>0) |
| Back | In | Yes | 243 | 122 | — | +1.86% [-12.61%, +17.04%] | -2.67% | não (ROI p30>0) |
| Back | In | No | 522 | 196 | — | +3.57% [-5.22%, +12.66%] | +0.77% | sim (ROI p30>0) |
| Lay | Pre | Yes | 199 | 105 | -0.15% [-1.07%, +0.76%] | +3.51% [-9.69%, +17.11%] | -0.66% | não (CLV p90<0 AND ROI p30>0) |
| Lay | Pre | No | 729 | 172 | +0.43% [-0.19%, +1.03%] | +0.61% [-8.04%, +8.91%] | -2.09% | não (CLV p90<0 AND ROI p30>0) |
| Lay | In | Yes | 263 | 138 | — | +24.72% [+6.27%, +45.25%] | +18.20% | sim (ROI p30>0) |
| Lay | In | No | 345 | 165 | — | +10.01% [-4.20%, +24.86%] | +5.35% | sim (ROI p30>0) |

Notas:
- Se você quiser operar **cada combinação como uma estratégia separada**, o sizing (Kelly/caps) deve ser estimado e governado separadamente por estratégia.
- Esta tabela é **in-sample** na janela (`--lookback-days`). A seção OOS (walk-forward) pode ser habilitada com `--walkforward`.

## 9) Combinações de valor (regime × linha AH × lag)
Tabela rankeada por volume (N). Métrica: diff_pct (OK, betslip confiável).

### 9.1 Back combos (diff_pct >= corte)
| Regime | Linha | Lag | N | Jogos | Mean diff % | IC90 cluster |
|---|---|---|---:|---:|---:|---|
| PRE_MATCH | AH 0-1 (líquida) | < 10s | 161 | 62 | +6.81% | [+6.83%, +7.42%] |
| IN_MATCH | AH 2+ (extrema) | < 10s | 119 | 89 | +6.17% | [+5.73%, +6.61%] |
| PRE_MATCH | AH 2+ (extrema) | < 10s | 89 | 40 | +6.81% | [+6.42%, +7.22%] |
| PRE_MATCH | AH 1-2 (média) | < 10s | 89 | 31 | +6.57% | [+6.66%, +7.46%] |
| IN_MATCH | AH 0-1 (líquida) | < 10s | 45 | 37 | +6.66% | [+6.06%, +7.19%] |
| IN_MATCH | AH 1-2 (média) | < 10s | 28 | 27 | +7.01% | [+6.16%, +7.66%] |
| PRE_MATCH | AH 0-1 (líquida) | 20-30s | 21 | 18 | +7.01% | [+6.21%, +7.56%] |
| PRE_MATCH | AH 0-1 (líquida) | > 30s | 12 | 11 | +6.12% | [+5.33%, +6.72%] |
| PRE_MATCH | AH 1-2 (média) | 20-30s | 12 | 10 | +6.03% | [+5.15%, +7.07%] |
| IN_MATCH | AH 2+ (extrema) | > 30s | 11 | 8 | +7.27% | [+5.01%, +8.34%] |
| PRE_MATCH | AH 1-2 (média) | > 30s | 10 | 9 | +5.95% | [+5.11%, +7.34%] |
| PRE_MATCH | AH 2+ (extrema) | 20-30s | 10 | 7 | +7.16% | [+6.44%, +8.44%] |

### 9.2 Lay combos (diff_pct <= corte) + risco
| Regime | Linha | Lag | N | Jogos | Mean diff % | IC90 cluster | Liability p95 |
|---|---|---|---:|---:|---:|---|---:|
| IN_MATCH | AH 2+ (extrema) | < 10s | 72 | 59 | -4.92% | [-5.44%, -4.50%] | 310.77 |
| IN_MATCH | AH 0-1 (líquida) | < 10s | 34 | 28 | -5.06% | [-5.83%, -4.59%] | 454.08 |
| PRE_MATCH | AH 0-1 (líquida) | < 10s | 24 | 20 | -4.82% | [-6.00%, -4.20%] | 3755.43 |
| PRE_MATCH | AH 2+ (extrema) | < 10s | 20 | 19 | -4.86% | [-5.53%, -4.04%] | 863.45 |
| PRE_MATCH | AH 1-2 (média) | < 10s | 14 | 10 | -5.09% | [-6.49%, -3.91%] | 103.88 |
| IN_MATCH | AH 1-2 (média) | < 10s | 13 | 12 | -4.23% | [-5.14%, -3.48%] | 686.83 |
| IN_MATCH | AH 2+ (extrema) | > 30s | 6 | 6 | -4.92% | [-6.68%, -3.25%] | 226.61 |
| IN_MATCH | AH 0-1 (líquida) | > 30s | 4 | 4 | -5.99% | [-7.82%, -4.47%] | 210.93 |
| IN_MATCH | AH 2+ (extrema) | 20-30s | 4 | 4 | -4.40% | [-5.69%, -3.05%] | 268.18 |
| PRE_MATCH | AH 2+ (extrema) | 20-30s | 3 | 3 | -3.90% | [-5.14%, -2.65%] | 79.09 |
| PRE_MATCH | AH 0-1 (líquida) | > 30s | 3 | 3 | -3.39% | [-3.56%, -3.21%] | 132.82 |
| IN_MATCH | AH 0-1 (líquida) | 20-30s | 3 | 3 | -3.18% | [-3.50%, -2.85%] | 175.10 |

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

- **Back (stake)**: corr(exposição, ROI)=-0.075; corr(exposição, CLV)=0.008 (onde CLV existe).
- **Lay (liability)**: corr(exposição, ROI)=0.077; corr(exposição, CLV)=0.061 (onde CLV existe).

**Backtest de sizing (apenas eventos com placar; valores em unidade monetária *proxy*)**

| Lado | Scheme | N | Turnover (janela) | Lucro (janela) | ROI/turnover | p99 exposição | ES95 exposição | DD 30d (média) | DD 30d (p95) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Back | FLAT | 558 | 558.00 | 19.35 | 3.47% | 1.00 | 1.00 | 6.64 | 12.26 |
| Lay | FLAT | 176 | 215.86 | 0.66 | 0.30% | 1.00 | 1.00 | 22.31 | 41.25 |
| Back | PROXY | 558 | 192989.53 | -34200.36 | -17.72% | 4396.60 | 4212.57 | 150487.56 | 219842.52 |
| Lay | PROXY | 125 | 32146.45 | -7730.98 | -24.05% | 3835.20 | 2142.82 | 38765.42 | 52863.82 |
| Back | KELLY_0.10 | 318 | 8251.65 | 380.55 | 4.61% | 56.97 | 51.27 | 586.61 | 1064.41 |
| Lay | KELLY_0.10 | 44 | 582.71 | 105.17 | 18.05% | 20.36 | 20.36 | 22.18 | 36.98 |
| Back | KELLY_0.25 | 318 | 16861.93 | 509.28 | 3.02% | 87.88 | 87.88 | 1303.29 | 2554.43 |
| Lay | KELLY_0.25 | 44 | 837.52 | 170.70 | 20.38% | 20.36 | 20.36 | 21.94 | 35.99 |
| Back | KELLY_0.50 | 318 | 20567.83 | 443.31 | 2.16% | 87.88 | 87.88 | 1339.38 | 2659.24 |
| Lay | KELLY_0.50 | 44 | 922.81 | 158.86 | 17.21% | 20.36 | 20.36 | 12.32 | 20.03 |
| Back | KELLY_1.00 | 318 | 21244.04 | 434.13 | 2.04% | 87.88 | 87.88 | 1396.85 | 2685.65 |
| Lay | KELLY_1.00 | 44 | 937.88 | 167.87 | 17.90% | 20.36 | 20.36 | 0.00 | 0.00 |

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
| Back | Pre | Yes | FLAT | 91 | 91.00 | -0.33 | -0.36% | 1.00 | 52.59 |
| Back | Pre | Yes | PROXY | 91 | 30945.32 | -8836.63 | -28.56% | 4734.71 | 82345.34 |
| Back | Pre | Yes | KELLY_0.10 | 80 | 2251.96 | -93.66 | -4.16% | 47.96 | 1489.81 |
| Back | Pre | Yes | KELLY_0.25 | 80 | 4798.29 | -78.98 | -1.65% | 87.88 | 2039.22 |
| Back | Pre | Yes | KELLY_0.50 | 80 | 5766.76 | 17.81 | 0.31% | 87.88 | 1709.57 |
| Back | Pre | Yes | KELLY_1.00 | 80 | 5893.44 | -14.73 | -0.25% | 87.88 | 2127.36 |
| Back | Pre | No | FLAT | 278 | 278.00 | 10.61 | 3.82% | 1.00 | 21.97 |
| Back | Pre | No | PROXY | 278 | 122098.36 | -16089.93 | -13.18% | 4433.26 | 148471.39 |
| Back | Pre | No | KELLY_0.10 | 238 | 5999.69 | 474.21 | 7.90% | 56.97 | 763.86 |
| Back | Pre | No | KELLY_0.25 | 238 | 12063.64 | 588.26 | 4.88% | 87.88 | 1672.26 |
| Back | Pre | No | KELLY_0.50 | 238 | 14801.07 | 425.50 | 2.87% | 87.88 | 1932.56 |
| Back | Pre | No | KELLY_1.00 | 238 | 15350.60 | 448.86 | 2.92% | 87.88 | 1879.55 |
| Back | In | Yes | FLAT | 58 | 58.00 | -4.52 | -7.80% | 1.00 | 48.71 |
| Back | In | Yes | PROXY | 58 | 9973.72 | -3099.84 | -31.08% | 1457.62 | 35311.89 |
| Back | In | No | FLAT | 131 | 131.00 | 13.59 | 10.38% | 1.00 | 0.00 |
| Back | In | No | PROXY | 131 | 29972.12 | -6173.97 | -20.60% | 2069.67 | 54731.01 |
| Lay | Pre | Yes | FLAT | 13 | 14.38 | 3.45 | 24.02% | 1.00 | 7.00 |
| Lay | Pre | Yes | PROXY | 13 | 6138.31 | -3626.31 | -59.08% | 3925.76 | 39313.10 |
| Lay | Pre | Yes | KELLY_0.10 | 9 | 96.55 | 16.30 | 16.88% | 20.00 | 115.25 |
| Lay | Pre | Yes | KELLY_0.25 | 9 | 148.95 | 35.04 | 23.53% | 20.36 | 97.22 |
| Lay | Pre | Yes | KELLY_0.50 | 9 | 179.21 | 41.42 | 23.11% | 20.36 | 98.05 |
| Lay | Pre | Yes | KELLY_1.00 | 9 | 181.49 | 43.70 | 24.08% | 20.36 | 95.58 |
| Lay | Pre | No | FLAT | 32 | 36.66 | 2.74 | 7.49% | 1.00 | 3.24 |
| Lay | Pre | No | PROXY | 32 | 10392.85 | -3622.01 | -34.85% | 3404.75 | 36547.52 |
| Lay | Pre | No | KELLY_0.10 | 24 | 356.71 | 26.47 | 7.42% | 20.36 | 46.67 |
| Lay | Pre | No | KELLY_0.25 | 24 | 497.03 | 55.08 | 11.08% | 20.36 | 28.40 |
| Lay | Pre | No | KELLY_0.50 | 24 | 536.78 | 38.67 | 7.20% | 20.36 | 78.08 |
| Lay | Pre | No | KELLY_1.00 | 24 | 536.78 | 38.67 | 7.20% | 20.36 | 75.65 |
| Lay | In | Yes | FLAT | 26 | 31.49 | 3.39 | 10.77% | 1.00 | 5.04 |
| Lay | In | Yes | PROXY | 26 | 5812.58 | 1881.87 | 32.38% | 1713.54 | 3036.37 |
| Lay | In | No | FLAT | 50 | 77.28 | -1.08 | -1.39% | 1.00 | 25.25 |
| Lay | In | No | PROXY | 50 | 9256.93 | -2770.63 | -29.93% | 1042.27 | 19824.91 |
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
| Back | Pre | Yes | 178 | 109 | +4.03% [+3.22%, +4.85%] | -3.25% [-16.32%, +9.90%] | -7.60% | pre: Kelly OK |
| Back | Pre | No | 870 | 176 | +1.76% [+1.33%, +2.20%] | -2.19% [-10.08%, +5.59%] | -4.67% | pre: Kelly OK |
| Back | In | Yes | 243 | 122 | — — | +1.86% [-12.61%, +17.04%] | -2.67% | in: use FLAT/PROXY |
| Back | In | No | 522 | 196 | — — | +3.57% [-5.22%, +12.66%] | +4.25% | in: use FLAT/PROXY |
| Lay | Pre | Yes | 199 | 105 | +0.74% [-0.18%, +1.64%] | +3.51% [-9.69%, +17.11%] | -0.66% | pre: Kelly OK |
| Lay | Pre | No | 729 | 172 | +0.11% [-0.49%, +0.71%] | +0.61% [-8.04%, +8.91%] | -2.09% | pre: Kelly OK |
| Lay | In | Yes | 263 | 138 | — — | +24.72% [+6.27%, +45.25%] | +18.20% | in: use FLAT/PROXY |
| Lay | In | No | 345 | 165 | — — | +10.01% [-4.20%, +24.86%] | +19.76% | in: use FLAT/PROXY |
| Estratégia | Scheme | N Back (janela) | N Lay (janela) | N Back 30d (proj.) | N Lay 30d (proj.) | Stake médio Back | Stake médio Lay | Liability média Lay | Turnover 30d (proj.) | Lucro 30d (proj.) | ROI/turnover (janela) | ROI Lay/liability (janela) | ROI Lay/turnover (janela) | Banca p99 (Back) | Banca p99 (Lay) | Banca risco p99 (soma) | Banca liquidez p99 (+buf) | Banca recomendada (max) | ROI/banca 30d | Turnover 30d (R$) | Lucro 30d (R$) | Banca rec. (R$) | DD 30d p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ativas (PRE, critérios 8.3) | FLAT | 0 | 0 | 0 | 0 | — | — | — | 0.00 | 0.00 | —% | —% | —% | — | — | — | — | — | —% | 0.00 | 0.00 | — | — |
| Ativas (PRE, critérios 8.3) | KELLY_0.25 | 0 | 0 | 0 | 0 | — | — | — | 0.00 | 0.00 | —% | —% | —% | — | — | — | — | — | —% | 0.00 | 0.00 | — | — |

Notas:
- **N Back/Lay** na tabela é **na janela observada**. As colunas `N 30d (proj.)` são uma **escala linear** por dias observados.
- Esta linha usa a união das **combinações pre‑match ativas** sob os critérios da 8.3 (é um resumo, não substitui a tabela 8.3).
- `Stake médio` e `Liability média` são **proxies** na unidade monetária do seu limit/finance; valores em **R$** usam fx `--fx-usdbrl`.
- `Banca recomendada` = max( banca por risco p99 (unitária, soma) ; banca por liquidez p99 (+buffer) ).
- **Por que Lay pode ter stake médio menor mesmo com ROI maior**: (i) Lay é governado por **liability** (risco) e usamos cap mais conservador; (ii) stake em Lay é derivado de `liability/(odd-1)` e depende do nível médio de odds; (iii) Kelly depende do **edge vs closing** (gap entry→closing), não do ROI observado isoladamente.
- **Por que não incluímos Back in‑match aqui**: Kelly/CLV usam `closing_odd` pré‑jogo; in‑match requer benchmark diferente (ex.: referência por minuto/VWAP) e governança operacional específica. A seção 2.2 já compara PRE_MATCH vs IN_MATCH.

### 9.4b Curva de capacidade — frações de Kelly e tamanho potencial
Esta tabela é um exercício para estimar **capacidade** (turnover/lucro/risco) ao variar a fração de Kelly. Ela deixa explícito quando o sizing satura por **cap por aposta**.

**Escala Kelly usada nesta curva**: P99_PROXY | ref_back=4394.23 ref_lay=2036.19 | cap_back=2.0% cap_lay=1.0% | max_stake_event=100%*limit

| Strategy | Scheme | cap_hit Back (%) | limit_hit Back (%) | cap_hit Lay (%) | limit_hit Lay (%) | Turnover 30d (proj.) | Lucro 30d (proj.) | ROI/banca 30d | DD 30d p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ativas (PRE, critérios 8.3) | KELLY_0.10 | —% | —% | —% | —% | 0.00 | 0.00 | —% | — |
| Ativas (PRE, critérios 8.3) | KELLY_0.25 | —% | —% | —% | —% | 0.00 | 0.00 | —% | — |
| Ativas (PRE, critérios 8.3) | KELLY_0.50 | —% | —% | —% | —% | 0.00 | 0.00 | —% | — |
| Ativas (PRE, critérios 8.3) | KELLY_1.00 | —% | —% | —% | —% | 0.00 | 0.00 | —% | — |

Leitura rápida:
- Se `cap_hit` estiver alto, a alavancagem adicional (ex.: 0,50× ou 1,00×) não vira turnover — o cap está “travando” o tamanho.
- Se `limit_hit` estiver alto, você está batendo no **limit operacional** (capacidade da conta/mercado) — aumentar banca ou frac não aumenta turnover.
- Se `cap_hit` estiver baixo, a curva deve escalar quase linearmente com `frac` (até bater em limites reais/operacionais).
- Lay normalmente satura antes por ter cap mais conservador (liability) e por ter risco de cauda mais assimétrico.

### 9.4c Diagnóstico: Back in‑match (por que não está na estratégia candidata)
Aqui reportamos Back in‑match **apenas como diagnóstico**. Não incluímos in‑match na estratégia candidata porque (i) `closing_odd` pré‑jogo não é benchmark in‑match e (ii) o sizing Kelly acima depende desse benchmark.

| Regime | Scheme | N | Turnover (janela) | Lucro (janela) | ROI/turnover |
|---|---|---:|---:|---:|---:|
| IN_MATCH BackFast (<5s) | FLAT | 126 | 126.00 | 16.17 | 12.83% |
| IN_MATCH BackFast (<5s) | PROXY | 126 | 24639.00 | -2490.17 | -10.11% |

Próximo passo (se quiser operar in‑match): definir um benchmark de preço justo in‑match (ex.: referência por minuto, VWAP, ou odds externas) e calibrar sizing/risco específico do live.

## 12) OOS rolling-forward (walk-forward): seleção e validação
Até aqui o relatório é **in-sample** (na janela `--lookback-days`). Este bloco (opcional) faz um walk-forward simples por dia:

- Em cada passo, usamos os últimos `wf_train_days` para **selecionar** combinações (das 8: `Side×Pre/In×Reversal`) com evidência de valor.
- No(s) dia(s) seguinte(s) (`wf_test_days`), medimos o resultado OOS nas combinações ativas.

**Evidência de valor (por combinação, no treino)** segue seus critérios:
- Back/Pre: CLV p90>0 (IC90 lb>0) e ROI>0 (não precisa ser sig.)
- Back/In: ROI p30>0
- Lay/Pre: CLV p90<0 (IC90 ub<0) e ROI p30>0
- Lay/In: ROI p30>0

Isso aproxima o fluxo operacional que você descreveu (seleciona no rolling atual e mede no próximo rolling).

### 12.0 Diagnóstico de cobertura OOS (por que N cai)
| Filtro | Jogos únicos |
|---|---:|
| Combinações elegíveis (edge + timing + t0) | 243 |
| Com ROI disponível (precisa de placar) | 208 |
| Com CLV disponível (pre-match + closing) | 129 |

Leitura: o walk-forward mede OOS principalmente por **ROI**, então ele encolhe quando a cobertura de placar é baixa. Além disso, a métrica é agregada por **jogo único** (cluster), então você verá números menores que o N de eventos.

| Train window | Test window | #ativas | Jogos OOS | ROI OOS (mean; IC90) | Turnover (teste) | Lucro (estratégia, budget) |
|---|---|---:|---:|---:|---:|---:|
| 2026-02-11→2026-02-13 | 2026-02-14→2026-02-14 | 1 | 51 | +12.77% [-6.68%, +31.65%] | 659.89 | 54.55 |
| 2026-02-12→2026-02-14 | 2026-02-15→2026-02-15 | 3 | 56 | +15.77% [-3.07%, +34.20%] | 772.24 | 168.90 |

**Frequência de ativação por combinação (quantas janelas ela entrou como ativa)**

| Combinação | #steps ativa |
|---|---:|
| Back_Pre_No | 2 |
| Back_In_No | 1 |
| Back_Pre_Yes | 1 |

### 12.A Transparência da seleção: métricas por combinação no treino
Para cada janela de treino, mostramos as métricas usadas para decidir se cada combinação ficou **ativa** ou não. Isso ajuda a entender, por exemplo, por que nenhuma Lay entrou em algumas janelas (geralmente N insuficiente ou ROI p30 <= 0).

**Train 2026-02-11→2026-02-13 → Test 2026-02-14→2026-02-14**

| Combinação | Ativa? | Jogos treino (tot/CLV/ROI) | CLV q10/q90 ou CI | ROI q30 | Motivo |
|---|---|---:|---:|---:|---|
| Back_Pre_Yes | NÃO | 23 / 23 / 22 | q10=6.68 | CI90_lb=6.44 | -9.22% | BackPre: clv_q10>0=False, roi_mean>0=True |
| Back_Pre_No | SIM | 72 / 72 / 64 | q10=6.45 | CI90_lb=6.33 | -3.86% | BackPre: clv_q10>0=True, roi_mean>0=True |
| Back_In_Yes | NÃO | 15 / — / 8 | — | -58.82% | BackIn: roi_q30>0 |
| Back_In_No | NÃO | 30 / — / 19 | — | +2.48% | BackIn: roi_q30>0 |
| Lay_Pre_Yes | NÃO | 5 / 4 / 5 | q90=-3.00 | CI90_ub=-2.37 | -22.79% | LayPre: clv_q90<0=False, roi_q30>0=False |
| Lay_Pre_No | NÃO | 15 / 15 / 12 | q90=-3.31 | CI90_ub=-2.75 | -19.38% | LayPre: clv_q90<0=False, roi_q30>0=False |
| Lay_In_Yes | NÃO | 9 / — / 6 | — | +22.01% | In: roi_q30>0 |
| Lay_In_No | NÃO | 16 / — / 14 | — | +2.34% | In: roi_q30>0 |

**Train 2026-02-12→2026-02-14 → Test 2026-02-15→2026-02-15**

| Combinação | Ativa? | Jogos treino (tot/CLV/ROI) | CLV q10/q90 ou CI | ROI q30 | Motivo |
|---|---|---:|---:|---:|---|
| Back_Pre_Yes | SIM | 44 / 39 / 41 | q10=6.71 | CI90_lb=6.58 | -2.48% | BackPre: clv_q10>0=True, roi_mean>0=True |
| Back_Pre_No | SIM | 93 / 85 / 82 | q10=6.13 | CI90_lb=5.97 | +5.30% | BackPre: clv_q10>0=True, roi_mean>0=True |
| Back_In_Yes | NÃO | 26 / — / 19 | — | -38.23% | BackIn: roi_q30>0 |
| Back_In_No | SIM | 72 / — / 61 | — | +6.09% | BackIn: roi_q30>0 |
| Lay_Pre_Yes | NÃO | 8 / 6 / 8 | q90=-3.76 | CI90_ub=-3.24 | -0.87% | LayPre: clv_q90<0=False, roi_q30>0=False |
| Lay_Pre_No | NÃO | 29 / 23 / 23 | q90=-4.26 | CI90_ub=-3.91 | +8.97% | LayPre: clv_q90<0=False, roi_q30>0=False |
| Lay_In_Yes | NÃO | 11 / — / 7 | — | +2.44% | In: roi_q30>0 |
| Lay_In_No | NÃO | 31 / — / 29 | — | +1.03% | In: roi_q30>0 |


Notas importantes:
- Se `Jogos OOS` for baixo em muitos passos, você ainda não tem volume suficiente para decisões por combinação. Nesse cenário faz sentido **Bayes hierárquico (partial pooling)** para estabilizar estimativas.
- **Lucro (estratégia, budget)** acima já incorpora a política de risco por jogo (match budget) e é a métrica principal.
- O walk-forward usa ROI no **ponto de entrada**: Back em `t0`; Lay em `t_reversal` quando existir, senão `t_last` (~t+20s).
- Para pre-match, também é útil monitorar CLV OOS (menos dependente de resultados), mas CLV mede qualidade de entrada, não P&L.

### 12.1 Estimativa 30 dias (OOS): turnover, lucro, banca, ROI/banca e drawdown
Esta estimativa usa o walk-forward acima como **simulador OOS**. O lucro pode ser reportado em duas versões:

- **obs.**: apenas jogos com ROI (placar) disponível.
- **exp.**: expande o lucro para a população elegível usando scaling por exposição/turnover (assume missing-at-random condicional à estratégia).

**Padrão de risco**: P&L aqui já é calculado com **budget por jogo (match_id)** consumido ao longo do tempo (Back=1.00% da banca ref; Lay=0.50% em liability; cap por sinal=33% do budget).

| Premissa | Valor |
|---|---:|
| Scheme pre-match (OOS) | `KELLY_0.25` |
| Scheme in-match (OOS) | `FLAT` |
| Expansão missing ROI | ON |
| Dias OOS usados | 2 |
| Turnover 30d (proj.) | 21481.98 |
| Turnover 30d (Pre/In) | 80339.22 / 540.00 |
| Lucro 30d (obs.) | 3067.47 |
| Lucro 30d (obs.) Pre/In | 7635.63 / 70.80 |
| Lucro 30d (exp.) | 3351.78 |
| Lucro 30d (exp.) Pre/In | 8090.89 / 82.22 |
| Banca risco p99 (Back+Lay) | 771.12 |
| Banca liquidez p99 (+buf) | 718.95 |
| Banca recomendada (max) | 771.12 |
| ROI/banca 30d (obs.) | 397.80% |
| ROI/banca 30d (exp.) | 434.67% |
| DD 30d p95 (obs.) | 0.00 |
| DD 30d p95 (exp.) | 0.00 |

### 12.2 Governança de exposição por jogo (budget por `match_id`) — sensibilidade
Objetivo: evitar concentração quando um mesmo jogo gera muitos sinais. Simulamos um orçamento de exposição por jogo, consumido ao longo do tempo.

- **Back** consome budget em **stake**.
- **Lay** consome budget em **liability**.
- Também aplicamos um cap por sinal como fração do budget do jogo (para não gastar tudo no 1º sinal).

Importante: o budget é parametrizado como fração de uma referência de banca. Aqui usamos:
- se `--kelly-bankroll` estiver setado: essa banca explícita;
- senão: a `Banca recomendada (max)` estimada na 12.1 (baseline).

Referência de banca p/ budget: 771.12 | budgets por jogo aplicados em stake (Back) e liability (Lay).

| Cenário | Turnover 30d | Lucro 30d (exp.) | Banca rec. (max) | ROI/banca 30d (exp.) | DD 30d p95 (exp.) |
|---|---:|---:|---:|---:|---:|
| BASELINE (sem budget) | 21481.98 | 3351.78 | 771.12 | 434.67% | 0.00 |
| BUDGET_0.50%/0.25% cap25% | 1951.88 | 304.94 | 84.43 | 361.18% | 0.00 |
| BUDGET_1.00%/0.50% cap33% | 4291.16 | 677.45 | 166.01 | 408.08% | 0.00 |
| BUDGET_2.00%/1.00% cap50% | 11466.42 | 1728.32 | 401.61 | 430.35% | 0.00 |

Leitura:
- Se a curva com budget melhora muito (menos negativo ou mais positivo) com pouca perda de turnover, o problema era **concentração por jogo**.
- Se tudo continuar negativo, o problema é **edge OOS** (principalmente in‑match) e budget só reduz a escala da perda.

**Volume e stake médio por combinação (janela OOS, com budget padrão)**

| Combinação | #steps ativa | Jogos OOS (uniques) | N eventos OOS | Stake eq. médio | Observação |
|---|---:|---:|---:|---:|---|
| Back_Pre_No | 2 | 73 | 93 | 51.27 | budget reduz concentração por jogo |
| Back_In_No | 1 | 27 | 36 | 1.00 | budget reduz concentração por jogo |
| Back_Pre_Yes | 1 | 8 | 9 | 65.32 | budget reduz concentração por jogo |
## 10) Diagnóstico: por que o ROI pode estar zerado
ROI aqui é calculado por placar do jogo (`matches.home_score/away_score`). Se os placares não estiverem preenchidos no banco, a cobertura de ROI será 0.

| Indicador | Valor |
|---|---:|
| Jogos únicos no recorte | 333 |
| Jogos com placar disponível (home_score/away_score não nulos) | 284 |
| Jogos com status='finished' no banco | 284 |

### 10.1 Distribuição por data de kickoff (explica janela da API)
- Kickoff (UTC) no recorte: **2026-02-11 15:00 UTC** até **2026-02-18 20:00 UTC**.

| Kickoff date (UTC) | Jogos | Com placar | Cobertura |
|---|---:|---:|---:|
| 2026-02-18 | 14 | 5 | 35.7% |
| 2026-02-17 | 46 | 46 | 100.0% |
| 2026-02-16 | 24 | 19 | 79.2% |
| 2026-02-15 | 75 | 70 | 93.3% |
| 2026-02-14 | 114 | 104 | 91.2% |
| 2026-02-13 | 50 | 33 | 66.0% |
| 2026-02-12 | 5 | 2 | 40.0% |
| 2026-02-11 | 5 | 5 | 100.0% |

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
