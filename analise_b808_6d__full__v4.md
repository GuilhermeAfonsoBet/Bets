# Análise Estatística Robusta — Contexto Operação (b808)
**Data da execução:** 17/02/2026 11:24 UTC  
**Escopo:** H3B (auditoria), com comparação por `audit_version` e inferência robusta por jogo (cluster bootstrap).  
**Nota:** a robustez aqui significa que intervalos de confiança consideram correlação intra-jogo (múltiplas auditorias por partida).
---
## 0) Sumário executivo (leitura rápida)
- **Recorte**: direction=`up`, lookback_days=`6`, versions=`v4.0-api,v1.0,v1.0-recovered`.
- **Amostra**: 3471 auditorias (jogos únicos=307, média=11.3 obs/jogo); betslip confiável=1768.
- **Janela efetiva (audited_at)**: 11/02 11:30 → 17/02 09:51 UTC (span≈5.9d; dias com dados=7).
- **Coortes (status=OK, betslip confiável)**: `BS>WS` (diff>=2.0%): **613**; `BS<WS` (diff<=-2.0%): **226**.
- **Coberturas em `hypothesis_details` (OK)**: temporal(back)=1288/1768; lay_temporal=1168/1768; finance=1058/1768.
- **Cobertura de placar (ROI)**: jogos com placar=255/307 (status finished=255).
- **Cobertura de closing_odd (AH)**: jogos com closing=150/307 (48.9%). CLV pre‑match depende disso.
- **Alerta**: cobertura de closing_odd está baixa. Isso tende a reduzir N de CLV e pode enviesar a leitura (os jogos “sem odds” ficam fora do CLV). Priorize estabilizar o collector e/ou condicionar análises a jogos com closing.
- **DOM**: sem dados no recorte atual (N=0), então não há comparação API vs DOM aqui.
- **CLV pre-match (Betslip, API)**: média robusta por jogo +1.956% (IC90 [+1.456%, +2.483%]), com N=748 eventos (jogos=123).
- **Padrão por bucket (CLV PM)**: `BS < WS` -3.974% (sig. negativo), `BS ~ WS` -0.118% (NS), `BS > WS` +6.647% (sig. positivo).
- **Leitura recomendada**: use este relatório para validar **qualidade de execução/CLV**. Para concluir sobre **ROI**, rode após atualizar resultados e/ou use uma janela com jogos já liquidados.

---
## 1) Contexto do corte (b808)
| Indicador | Valor |
|---|---:|
| Auditorias H3B `UP` (match + kickoff passado) | 3471 |
| Betslip bruto | 2852 |
| Betslip confiável (diff -10% a +10%) | 1768 |
| Descartados no filtro de qualidade | 1084 |
| Jogos únicos (geral) | 307 |
| Média de observações por jogo | 11.3 |
| Jogos únicos com betslip confiável | 293 |
| Distribuição por market_type | AH=3471 |
| Jogos únicos (AH) no recorte | 307 |
| Jogos únicos (AH) com closing_odd disponível | 150 |
| Cobertura closing_odd (AH) | 48.9% |

---
## 2) Base comparativa: API vs DOM
| Métrica | API (v4.0-api) | DOM (v1.0) |
|---|---:|---:|
| Total de observações | 3471 | 0 |
| Com betslip confiável | 1768 | 0 |
| Com CLV pre-match (betslip) | 748 | 0 |
| Com ROI (betslip) | 1550 | 0 |
| Tempo total observado (detecção→betslip, wall/total) | 16449 ms | — ms |
| Tempo instrumentado (detecção→clique→betslip) | 8499 ms | — ms |

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
| API (2-4s) | lag_det→click | 5926 | 821 | 7489 | 3471 |
| API (2-4s) | lag_click→betslip | 2592 | 2128 | 4254 | 3391 |
| API (2-4s) | lag_e2e (soma) | 8499 | 3398 | 9718 | 3391 |
| API (2-4s) | audit_total (duração) | 16445 | 4763 | 56782 | 3471 |
| API (2-4s) | overhead (total - e2e) | 8109 | 283 | 33193 | 3391 |
| DOM (15-30s) | lag_det→click | — | — | — | 0 |
| DOM (15-30s) | lag_click→betslip | — | — | — | 0 |
| DOM (15-30s) | lag_e2e (soma) | — | — | — | 0 |
| DOM (15-30s) | audit_total (duração) | — | — | — | 0 |
| DOM (15-30s) | overhead (total - e2e) | — | — | — | 0 |

**Diagnóstico de cauda (percentual acima do limiar)**

| Modelo | % det→click > 5s | % det→click > 20s | % total > 10s | % total > 40s | % overhead < 0 |
|---|---:|---:|---:|---:|---:|
| API (2-4s) | 5.8% | 3.3% | 20.9% | 7.1% | 0.0% |
| DOM (15-30s) | —% | —% | —% | —% | —% |

---
### 2.1 Cobertura temporal (pre-match vs in-match)
| Métrica | Pre-match | In-match | Observação |
|---|---:|---:|---|
| Observações totais com classificação temporal | 1311 | 2160 | Contagem bruta do corte |
| ROI Betslip | 822 | 728 | Amostra com resultado do jogo |
| ROI WebSocket | 1170 | 1870 | Referência de mercado |
| CLV (apenas pre-match) | 748 | — | CLV vs closing pré-jogo não é interpretável in-match |

---
### 2.2 Performance por regime (pre-match vs in-match)
| Regime | N auditorias | N betslip conf. | N OK | Back edge | Lay edge | Diff médio (OK) |
|---|---:|---:|---:|---:|---:|---:|
| PRE_MATCH | 1311 | 915 | 915 | 348 | 68 | +2.075% |
| IN_MATCH | 2160 | 853 | 853 | 265 | 158 | +0.989% |

---
### 2.3 Regimes operacionais por tempo total (bucket)
Objetivo: separar a amostra em **regimes de execução** (tempo total) e medir performance em cada regime. Use isso para detectar **fila/saturação**: regimes lentos tendem a ter edge menor e pior ROI.

| Bucket (tempo total) | N OK | Jogos | lag_total mean (ms) | lag_total p95 (ms) | overhead p95 (ms) | Back edge | Lay edge | CLV PM (mean; IC90) | ROI (mean; IC90) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| < 5s | 1142 | 244 | 3561 | 4850 | 2507 | 369 | 132 | +2.22% [+1.73%, +2.71%] | +3.86% [-2.14%, +10.03%] |
| 5-10s | 213 | 131 | 5949 | 8505 | 3167 | 84 | 43 | +1.72% [+0.70%, +2.70%] | -2.88% [-15.44%, +9.35%] |
| 10-20s | 21 | 19 | 13829 | 19179 | 14060 | 8 | 5 | +1.22% [-2.21%, +4.44%] | +17.36% [-14.29%, +49.40%] |
| 20-40s | 268 | 119 | 27292 | 34410 | 30761 | 93 | 31 | +2.03% [+1.09%, +3.00%] | +10.08% [-1.28%, +21.48%] |
| > 40s | 124 | 74 | 137452 | 433092 | 221376 | 59 | 15 | +2.25% [+1.11%, +3.35%] | -13.75% [-29.42%, +1.70%] |
| Desconhecido | 0 | 0 | — | — | — | 0 | 0 | — | — |

---
### 2.3b Regimes por tempo total — separando coortes por `diff_pct` (BS vs WS)
Nesta tabela, separamos duas coortes operacionais por **delta de execução**: `BS > WS` (diff_pct >= +2%) e `BS < WS` (diff_pct <= -2%). Isso **não** é (por si só) “Back vs Lay”; é um recorte por **melhora/piora do preço** entre detecção (WS) e execução (BS). CLV é reportado apenas em pre‑match.

| Bucket | N OK | N (BS>WS +2%) | N (BS<WS -2%) | CLV PM (BS>WS) | CLV PM (BS<WS) | ROI (BS>WS) | ROI (BS<WS) |
|---|---:|---:|---:|---:|---:|---:|---:|
| < 5s | 1142 | 369 | 132 | +6.90% [+6.42%, +7.38%] | -3.74% [-4.92%, -2.54%] | +3.19% [-7.78%, +13.84%] | +3.27% [-12.09%, +18.53%] |
| 5-10s | 213 | 84 | 43 | +5.78% [+4.98%, +6.52%] | -6.16% [-7.68%, -4.56%] | -9.94% [-29.10%, +9.07%] | -12.40% [-32.46%, +8.21%] |
| 10-20s | 21 | 8 | 5 | +4.89% [+2.51%, +7.32%] | -6.98% — | +0.99% [-49.88%, +52.37%] | +40.04% [+0.00%, +79.90%] |
| 20-40s | 268 | 93 | 31 | +7.79% [+6.78%, +8.80%] | -2.03% [-4.39%, +0.55%] | +11.30% [-6.29%, +28.98%] | +6.33% [-24.85%, +35.90%] |
| > 40s | 124 | 59 | 15 | +5.06% [+3.27%, +6.89%] | -7.29% — | -12.41% [-35.19%, +10.41%] | -11.35% [-55.50%, +31.89%] |
| Desconhecido | 0 | 0 | 0 | — | — | — | — |

---
### 2.3c Estabilidade temporal (por dia, `audited_at`)
Objetivo: checar se o regime de edge/execução é **time‑dependent**. Se houver dias com comportamento distinto (ex.: collector intermitente, horários/ligas), isso aparecerá aqui.

| Dia (UTC) | Modelo | N OK | Jogos | % BS>WS +2% | % BS<WS -2% | lag_total p50 (ms) | CLV PM (BS>WS) | CLV PM (BS<WS) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-02-11 | API (2-4s) | 188 | 63 | 36.7% | 17.6% | 26349 | +7.02% | -4.08% |
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
| CLV Bruto BS Pre-Match | +2.278% (sig. positivo, N=748, jogos=123) | — (N/A, N=0, jogos=0) |
| CLV Adicional BS Pre-Match | +1.545% (sig. positivo, N=748, jogos=123) | — (N/A, N=0, jogos=0) |
| Taxa de CLV > 0 (bruto) | 60.9% | —% |
| Taxa de CLV > 0 (adicional) | 60.3% | —% |

Notas de robustez (IC 90% por jogo):  
- API CLV bruto (cluster): média +1.956%; IC90 [+1.456%, +2.483%]  
- DOM CLV bruto (cluster): média —; IC90 —  

---
## 4) ROI por modelo
| Métrica | API | DOM |
|---|---:|---:|
| ROI Betslip | +1.377% (NS, N=1548) | — (N/A, N=0) |
| ROI WebSocket | -0.354% (NS, N=3024) | — (N/A, N=0) |
| Win rate ROI Betslip | 50.8% | —% |
| Win rate ROI WS | 50.2% | —% |

Notas de robustez (IC 90% por jogo):  
- API ROI betslip (cluster): média +1.830%; IC90 [-3.067%, +6.869%]  
- API ROI WS (cluster): média -2.046%; IC90 [-4.357%, +0.289%]  

---
## 5) Diferença de preço BS vs WS
| Métrica | API | DOM |
|---|---:|---:|
| Diff BS vs WS (média) | +1.551% (sig. positivo, N=1768) | — (N/A, N=0) |
| BS > WS | 46.3% (819/1768) | —% (0/0) |
| BS > WS +2% | 34.7% (613/1768) | —% (0/0) |

---
## 6) Combinações de valor
### 6.1 Buckets por diferença BS vs WS
Nota de leitura:
- `BS > WS (+2% a +10%)`: preço no betslip ficou **melhor** do que o WS observado na detecção (delta positivo entre instantes).
- `BS < WS (-10% a -2%)`: preço no betslip ficou **pior** do que o WS observado na detecção (delta negativo entre instantes).
- O ROI abaixo é calculado **dentro do bucket** (não mistura buckets), mas ainda é sensível à cobertura de placar.

| Bucket | N bucket | CLV BS PM (média) | IC90 (cluster) | N CLV PM | Jogos CLV PM | ROI BS (todos) | IC90 (cluster) |
|---|---:|---:|---|---:|---:|---:|---|
| BS < WS (-10% a -2%) | 226 | -3.974% | [-5.072%, -3.182%] | 51 | 39 | +3.906% | [-12.396%, +10.053%] |
| BS ~ WS (-2% a +2%) | 929 | -0.118% | [-0.599%, +0.276%] | 403 | 109 | +0.301% | [-9.200%, +3.686%] |
| BS > WS (+2% a +10%) | 613 | +6.647% | [+6.334%, +7.204%] | 294 | 91 | +2.074% | [-1.565%, +15.505%] |

---
### 6.2 Combinação por faixa de linha AH
| Faixa AH | CLV BS PM (média) | IC90 (cluster) | ROI BS (todos) | IC90 (cluster) | Diff BS vs WS (média) |
|---|---:|---|---:|---|---:|
| AH 0-1 (líquida) | +2.462% | [+1.676%, +3.200%] | +1.305% | [+1.616%, +16.676%] | +1.798% |
| AH 1-2 (média) | +2.032% | [+1.017%, +3.534%] | +2.786% | [-11.777%, +14.379%] | +2.118% |
| AH 2+ (extrema) | +2.187% | [+0.554%, +2.411%] | +0.938% | [-7.945%, +7.036%] | +1.179% |

---
### 6.3 Combinação por faixa de lag
| Faixa de lag | CLV BS PM (média) | IC90 (cluster) | N CLV PM | Jogos CLV PM | ROI BS (todos) | IC90 (cluster) | Diff BS vs WS (média) |
|---|---:|---|---:|---:|---:|---|---:|
| < 10s | +2.370% | [+1.607%, +2.607%] | 535 | 113 | +1.474% | [-3.396%, +8.554%] | +1.469% |
| 10-20s | +1.285% | [-2.212%, +4.435%] | 6 | 6 | +16.300% | [-14.288%, +49.400%] | +0.899% |
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
| Cobertura finance (OK, betslip conf.) | 1058/1768 |

### 7.1 Back (BS >> WS)
| Métrica | Valor |
|---|---:|
| Corte (diff_pct) | >= 2.0% |
| N eventos | 613 |
| Cobertura finance (na coorte) | 355/613 |
| Stake total (estimado) | 214776.44 |
| Stake médio | 350.37 |
| Profit_if_win total (estimado) | 230904.25 |
| Profit_if_win médio | 376.68 |
| N com ROI realizado | 546 |
| P&L realizado total (estimado) | -42381.49 |
| ROI realizado (ponderado por stake) | -20.32% |
| ROI realizado (robusto por jogo, mean; IC90) | +6.78% [-1.57%, +15.50%] |
| ROI ponderado por stake (robusto por jogo, mean; IC90) | +6.78% [-4.36%, +18.36%] |

**Como ler as 3 linhas de ROI (Back)**

- **ROI realizado (ponderado por stake)**: ΣP&L / Σstake (pode ser dominado por stakes grandes).
- **ROI realizado (robusto por jogo)**: média por jogo (cada jogo pesa parecido; reduz dominância de outliers).
- **ROI ponderado por stake (robusto por jogo)**: calcula ROI ponderado dentro do jogo e depois faz média/IC por jogo.

Observação: a Seção 4 reporta ROI médio no **recorte completo** (betslip confiável). Aqui (7.1) é apenas a coorte **Back edge** (diff>=corte) e o ROI também é mostrado ponderado por stake; por isso sinais podem divergir.

### 7.2 Lay (BS << WS) — risco de cauda
| Métrica | Valor |
|---|---:|
| Corte (diff_pct) | <= -2.0% |
| N eventos | 226 |
| Cobertura finance (na coorte) | 138/226 |
| Stake total (estimado) | 41446.64 |
| Liability total (estimada) | 36260.11 |
| Liability média | 160.44 |
| Liability p95 | 568.12 |
| Liability p99 | 2045.02 |
| ES95 (liability) | 1663.71 |
| Liability max | 4395.35 |
| Proxy de banca (>= p99 liability) | 2045.02 |
| N com ROI realizado | 144 |
| P&L realizado total (estimado) | -3084.58 |
| ROI realizado (ponderado por liability) | -9.17% |
| ROI realizado (ponderado por stake) | -7.99% |
| ROI/liability (robusto por jogo, mean; IC90) | +12.84% [-2.30%, +28.22%] |
| ROI/liability ponderado (robusto por jogo, mean; IC90) | +14.22% [-1.70%, +29.85%] |

### 7.3 Projeção mensal (30 dias fixo) — turnover, lucro e banca
Premissas:
- **Mês = 30 dias fixo**.
- Turnover = soma de stakes (Back e Lay). Para Lay, também reportamos exposição por **liability**.
- **Banca por risco (unitária)** = p99/ES95(exposição por aposta).
- **Banca por liquidez (turnover)** = p95/p99 de capital **simultaneamente travado** (stake/liability em aberto até a liquidação), capturando distribuição desigual de entradas ao longo do tempo.
- Projeção de lucro usa duas visões: (i) **lucro/dia observado** e (ii) **ROI observado × turnover projetado**.

| Bloco | Janela(d) | Turnover 30d | Lucro 30d (direto) | Lucro 30d (ROI×turnover) |
|---|---:|---:|---:|---:|
| Back | 6.0 | 1073882.21 | -211907.44 | -218217.12 |
| Lay (stake) | 6.0 | 207233.22 | -15422.88 | -16556.02 |
| Total (Back+Lay) | 6.0 | 1281115.42 | -227330.32 | -234773.14 |

**Banca por risco (exposição unitária)**

| Bloco | Banca conservadora (p99) | Banca agressiva (ES95) | ROI/banca 30d (direto) | ROI/banca 30d (ROI×turnover) |
|---|---:|---:|---:|---:|
| Back (stake) | 4398.85 | 4132.71 | -4817.33% | -4960.77% |
| Lay (liability) | 2045.02 | 1663.71 | -754.17% | -809.58% |
| Total (soma) | 6443.88 | 5796.43 | -3527.85% | -3643.35% |

**Banca por liquidez (capital simultaneamente travado)**

Definição operacional: cada aposta trava capital de `audited_at` até `kickoff + 2.25h` (grid=5min). A banca recomendada aplica buffer de +10%.

| Bloco | Liquidez mean | Liquidez p95 | Liquidez p99 | Liquidez max | Banca liq p99 (+buffer) |
|---|---:|---:|---:|---:|---:|
| Back (stake) | 18160.43 | 49021.01 | 74135.76 | 79598.19 | 81549.34 |
| Lay (liability) | 2430.97 | 6139.73 | 8803.49 | 9058.33 | 9683.84 |
| Total (Back+Lay) | 20573.16 | 52492.16 | 80288.43 | 85737.92 | 88317.28 |

**Banca recomendada (conservadora)**

| Métrica | Valor |
|---|---:|
| Banca por risco (p99 unitário, soma) | 6443.88 |
| Banca por liquidez (p99 simultâneo + buffer) | 88317.28 |
| Banca efetiva (max das duas) | 88317.28 |
| ROI/banca 30d (direto, banca efetiva) | -257.40% |
| ROI/banca 30d (ROI×turnover, banca efetiva) | -265.83% |
**Diagnóstico de cobertura (placar/ROI)**

| Bloco | Turnover total | Turnover com ROI | Cobertura turnover |
|---|---:|---:|---:|
| Back | 214776.44 | 208566.24 | 97.11% |
| Lay | 41446.64 | 38609.92 | 93.16% |

Notas (Lay): exposição 30d por liability (não é turnover) = 181300.57; ROI realizado por liability (ponderado) = -9.17%.

---
## 8) Curva temporal (pico, reversão e melhor timing)
Esta seção usa `hypothesis_details.temporal` (Back) e `hypothesis_details.lay_temporal` (Lay), coletados em pontos discretos (t≈0,3,6,10,15,20s).

O objetivo é responder: **tempo até o pico/vale**, **% que segue melhorando até t_max**, **% com reversão** e **impacto esperado por timing**. **CLV é reportado somente pre-match** (closing pré-jogo).

### 8.1 Back (pico em diff_pct)
Definições: **pico** = `max(diff_pct)`. **reversão** = após o pico, `diff_pct` cair pelo menos 0.50 p.p. (para Lay, subir 0.50 p.p. após o vale). `t_reversão` é o 1º tempo que cruza esse limiar (logo, não é o pico).

| Regime | N | t_pico médio (s) | t_pico p50 (s) | % melhora até fim | % com reversão | t_reversão médio (s) | Δt (rev - pico) médio (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRE_MATCH | 915 | 4.1 | 5.1 | 77.8% | 15.5% | 12.3 | 7.3 |
| IN_MATCH | 853 | 4.9 | 0.0 | 63.7% | 28.5% | 13.5 | 8.1 |

**Partição 100% (Back)**: categorias exclusivas (somam 100% por regime).

| Regime | % pico no fim, sem reversão | % pico no fim, com reversão (recuperou) | % pico antes, com reversão | % pico antes, sem reversão relevante |
|---|---:|---:|---:|---:|
| PRE_MATCH | 81.5% | 6.3% | 9.2% | 3.0% |
| IN_MATCH | 69.1% | 4.5% | 24.0% | 2.5% |

**Curva média (Back)**: média de `diff_pct` e `odd` por tempo. `CLV` é reportado **somente pre-match** (closing pré-jogo). `ROI` (quando aparece) é o **ROI realizado** se a aposta fosse feita naquele ponto.

| Tempo | N pts | diff_pct médio | odd média | CLV médio (pre-match) | ROI médio (se houver) |
|---|---:|---:|---:|---:|---:|
| t+0s | 1768 | +1.55% | 2.012 | +2.28% | 2.14 |
| t+6s | 1263 | +2.23% | 2.026 | +2.70% | 1.58 |
| t+10s | 2038 | +2.60% | 2.032 | +2.74% | 3.12 |
| t+15s | 1271 | +2.76% | 2.052 | +2.79% | 1.46 |
| t+20s | 1757 | +3.34% | 2.047 | +2.79% | 1.38 |

### 8.1b Back — impacto por timing (t0 vs pico vs último) e reversão
Leitura: se a estratégia é **entrar no pico**, compare CLV/ROI no pico vs t0 e veja diferença entre **casos com reversão** vs **sem reversão**. Aqui reportamos **média robusta por jogo + IC90**. Observação: **CLV só é calculado pre-match**.

**CLV (somente pre-match) — média robusta por jogo (IC90)**

| Subcoorte | N total | N CLV (PM) | CLV t0 (mean; IC90) | CLV pico (mean; IC90) | CLV último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 1383 | 627 | +1.73% [+1.20%, +2.25%] | +2.00% [+1.48%, +2.51%] | +1.99% [+1.47%, +2.50%] |
| COM_REVERSAO | 385 | 121 | +3.88% [+2.96%, +4.78%] | +5.42% [+4.38%, +6.46%] | +4.47% [+3.40%, +5.55%] |

**ROI (stake=1) — média robusta por jogo (IC90)**

| Subcoorte | N total | N ROI | ROI t0 (mean; IC90) | ROI pico (mean; IC90) | ROI último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 1383 | 1217 | +1.37% [-4.50%, +7.12%] | +1.49% [-4.44%, +7.36%] | +1.47% [-4.45%, +7.34%] |
| COM_REVERSAO | 385 | 331 | +4.06% [-7.04%, +15.64%] | +7.42% [-4.00%, +19.44%] | +4.17% [-6.86%, +15.50%] |

Nota interpretativa: a subcoorte **COM_REVERSAO** pode ter CLV maior no **pico** porque, por definição, ela inclui casos em que o edge atingiu um extremo mais alto antes de reverter. Para entender a hipótese 'sem reversão deveria ser maior', compare também a categoria '**pico no fim, sem reversão**' na partição 100% (tabela 8.1).

**Decomposição (pre-match): entry_odd vs closing_odd (IC90)**

| Subcoorte | N CLV (PM) | entry_odd t0 (mean; IC90) | entry_odd pico (mean; IC90) | closing_odd (mean; IC90) |
|---|---:|---:|---:|---:|
| SEM_REVERSAO | 627 | 1.979 [+1.967, +1.990] | 1.986 [+1.973, +1.998] | 1.969 [+1.961, +1.978] |
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
## 9) Combinações de valor (regime × linha AH × lag)
Tabela rankeada por volume (N). Métrica: diff_pct (OK, betslip confiável).

### 9.1 Back combos (diff_pct >= corte)
| Regime | Linha | Lag | N | Jogos | Mean diff % | IC90 cluster |
|---|---|---|---:|---:|---:|---|
| PRE_MATCH | AH 0-1 (líquida) | < 10s | 125 | 45 | +6.60% | [+6.53%, +7.15%] |
| IN_MATCH | AH 2+ (extrema) | < 10s | 121 | 91 | +6.22% | [+5.82%, +6.69%] |
| PRE_MATCH | AH 2+ (extrema) | < 10s | 78 | 37 | +6.92% | [+6.52%, +7.34%] |
| PRE_MATCH | AH 1-2 (média) | < 10s | 54 | 24 | +6.64% | [+6.45%, +7.34%] |
| IN_MATCH | AH 0-1 (líquida) | < 10s | 47 | 39 | +6.52% | [+5.87%, +7.00%] |
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
| IN_MATCH | AH 2+ (extrema) | < 10s | 74 | 61 | -4.95% | [-5.51%, -4.53%] | 305.95 |
| IN_MATCH | AH 0-1 (líquida) | < 10s | 34 | 28 | -5.06% | [-5.83%, -4.59%] | 454.08 |
| PRE_MATCH | AH 0-1 (líquida) | < 10s | 22 | 19 | -4.75% | [-5.98%, -4.13%] | 4176.11 |
| PRE_MATCH | AH 2+ (extrema) | < 10s | 18 | 18 | -4.99% | [-5.78%, -4.17%] | 898.86 |
| IN_MATCH | AH 1-2 (média) | < 10s | 15 | 14 | -4.21% | [-5.01%, -3.51%] | 635.14 |
| IN_MATCH | AH 2+ (extrema) | > 30s | 15 | 13 | -4.60% | [-5.66%, -3.72%] | 1438.89 |
| PRE_MATCH | AH 1-2 (média) | < 10s | 12 | 8 | -5.08% | [-6.61%, -3.85%] | 104.45 |
| IN_MATCH | AH 2+ (extrema) | 20-30s | 8 | 8 | -3.92% | [-5.03%, -2.84%] | 527.66 |
| IN_MATCH | AH 0-1 (líquida) | > 30s | 6 | 6 | -6.37% | [-7.74%, -5.16%] | 323.18 |
| IN_MATCH | AH 0-1 (líquida) | 20-30s | 4 | 4 | -3.48% | [-4.04%, -2.94%] | 170.81 |
| PRE_MATCH | AH 1-2 (média) | 10-20s | 3 | 3 | -4.16% | [-5.75%, -2.58%] | 51.05 |
| PRE_MATCH | AH 2+ (extrema) | > 30s | 3 | 3 | -4.42% | [-6.13%, -2.71%] | 91.55 |

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

- **Back (stake)**: corr(exposição, ROI)=-0.089; corr(exposição, CLV)=0.004 (onde CLV existe).
- **Lay (liability)**: corr(exposição, ROI)=0.016; corr(exposição, CLV)=0.066 (onde CLV existe).

**Backtest de sizing (apenas eventos com placar; valores em unidade monetária *proxy*)**

| Lado | Scheme | N | Turnover (janela) | Lucro (janela) | ROI/turnover | p99 exposição | ES95 exposição | DD 30d (média) | DD 30d (p95) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Back | FLAT | 546 | 546.00 | 17.80 | 3.26% | 1.00 | 1.00 | 7.28 | 13.40 |
| Lay | FLAT | 194 | 247.84 | -6.02 | -2.43% | 1.00 | 1.00 | 47.54 | 77.21 |
| Back | PROXY | 546 | 208566.24 | -42381.49 | -20.32% | 4408.56 | 4354.70 | 212923.26 | 283145.61 |
| Lay | PROXY | 144 | 38609.92 | -3084.58 | -7.99% | 3398.79 | 2162.72 | 28282.41 | 51140.94 |
| Back | KELLY_0.10 | 274 | 7094.88 | 141.51 | 1.99% | 57.78 | 52.76 | 641.88 | 1252.46 |
| Lay | KELLY_0.10 | 46 | 570.84 | 39.77 | 6.97% | 20.45 | 20.45 | 33.63 | 60.06 |
| Back | KELLY_0.25 | 274 | 14715.02 | 54.55 | 0.37% | 87.98 | 87.98 | 1551.96 | 2897.15 |
| Lay | KELLY_0.25 | 46 | 864.73 | 78.40 | 9.07% | 20.45 | 20.45 | 30.71 | 53.31 |
| Back | KELLY_0.50 | 274 | 18253.67 | -57.27 | -0.31% | 87.98 | 87.98 | 1648.60 | 3162.32 |
| Lay | KELLY_0.50 | 46 | 969.09 | 83.39 | 8.60% | 20.45 | 20.45 | 34.41 | 60.33 |
| Back | KELLY_1.00 | 274 | 18967.52 | -27.24 | -0.14% | 87.98 | 87.98 | 1752.97 | 3302.02 |
| Lay | KELLY_1.00 | 46 | 990.84 | 99.07 | 10.00% | 20.45 | 20.45 | 33.20 | 57.37 |

Leitura:
- Se `PROXY` piora ROI/turnover vs `FLAT`, isso indica que a política de stake atual está concentrando exposição em pontos com pior performance.
- `KELLY_0.25` tende a ser um bom compromisso quando o edge é estimado por CLV, mas requer **caps** e só é aplicável quando há `closing_odd` (pre‑match).
- Em Lay, é comum observar ROI alto por **liability**, mas sizing menor em **stake**: isso é uma decisão deliberada de governança de risco (liability tem cauda pior).
- DD é estimado por bootstrap i.i.d de dias (aproximação). Para uma curva mais fiel, use bootstrap por dia com blocos maiores.

### 9.4 Estratégias candidatas (filtros + sizing recomendado)
Aqui consolidamos duas hipóteses operacionais do usuário e reportamos resultados esperados no recorte (e projeção simples para 30 dias). **Recomendação de sizing**: comece com `KELLY_0.25` + caps, e compare contra `FLAT` como baseline.

| Estratégia | Scheme | N Back (janela) | N Lay (janela) | N Back 30d (proj.) | N Lay 30d (proj.) | Stake médio Back | Stake médio Lay | Liability média Lay | Turnover 30d (proj.) | Lucro 30d (proj.) | ROI/turnover (janela) | ROI Lay/liability (janela) | ROI Lay/turnover (janela) | Banca p99 (Back) | Banca p99 (Lay) | Banca risco p99 (soma) | Banca liquidez p99 (+buf) | Banca recomendada (max) | ROI/banca 30d | Turnover 30d (R$) | Lucro 30d (R$) | Banca rec. (R$) | DD 30d p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BackFast (<5s, PM) + LayReversal (t_ext<=3s) | FLAT | 200 | 22 | 1000 | 110 | 1.00 | 1.10 | 1.00 | 1120.68 | 36.76 | 3.28% | 2.97% | 2.71% | 1.00 | 1.00 | 2.00 | 157.68 | 157.68 | 23.32% | 5827.53 | 191.17 | 819.96 | 45.39 |
| BackFast (<5s, PM) + LayReversal (t_ext<=3s) | KELLY_0.25 | 167 | 18 | 835 | 90 | 55.83 | 17.90 | 15.71 | 48226.78 | 1373.66 | 2.85% | 2.64% | 2.31% | 87.98 | 20.45 | 108.43 | 6967.79 | 6967.79 | 19.71% | 250779.26 | 7143.05 | 36232.51 | 2038.01 |

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

**Escala Kelly usada nesta curva**: P99_PROXY | ref_back=4398.85 ref_lay=2045.02 | cap_back=2.0% cap_lay=1.0% | max_stake_event=100%*limit

| Strategy | Scheme | cap_hit Back (%) | limit_hit Back (%) | cap_hit Lay (%) | limit_hit Lay (%) | Turnover 30d (proj.) | Lucro 30d (proj.) | ROI/banca 30d | DD 30d p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BackFast+LayReversal | KELLY_0.10 | 0.0% | 13.1% | 4.8% | 4.8% | 22540.61 | 775.27 | 23.65% | 875.78 |
| BackFast+LayReversal | KELLY_0.25 | 19.7% | 25.1% | 57.1% | 4.8% | 48226.78 | 1373.66 | 19.71% | 1877.12 |
| BackFast+LayReversal | KELLY_0.50 | 83.1% | 31.1% | 85.7% | 4.8% | 60565.45 | 1311.59 | 15.09% | 1417.98 |
| BackFast+LayReversal | KELLY_1.00 | 93.4% | 31.7% | 100.0% | 4.8% | 62456.96 | 1809.04 | 20.19% | 1356.68 |

Leitura rápida:
- Se `cap_hit` estiver alto, a alavancagem adicional (ex.: 0,50× ou 1,00×) não vira turnover — o cap está “travando” o tamanho.
- Se `limit_hit` estiver alto, você está batendo no **limit operacional** (capacidade da conta/mercado) — aumentar banca ou frac não aumenta turnover.
- Se `cap_hit` estiver baixo, a curva deve escalar quase linearmente com `frac` (até bater em limites reais/operacionais).
- Lay normalmente satura antes por ter cap mais conservador (liability) e por ter risco de cauda mais assimétrico.

### 9.4c Diagnóstico: Back in‑match (por que não está na estratégia candidata)
Aqui reportamos Back in‑match **apenas como diagnóstico**. Não incluímos in‑match na estratégia candidata porque (i) `closing_odd` pré‑jogo não é benchmark in‑match e (ii) o sizing Kelly acima depende desse benchmark.

| Regime | Scheme | N | Turnover (janela) | Lucro (janela) | ROI/turnover |
|---|---|---:|---:|---:|---:|
| IN_MATCH BackFast (<5s) | FLAT | 128 | 128.00 | 17.58 | 13.73% |
| IN_MATCH BackFast (<5s) | PROXY | 128 | 25515.59 | -1394.06 | -5.46% |

Próximo passo (se quiser operar in‑match): definir um benchmark de preço justo in‑match (ex.: referência por minuto, VWAP, ou odds externas) e calibrar sizing/risco específico do live.

## 10) Diagnóstico: por que o ROI pode estar zerado
ROI aqui é calculado por placar do jogo (`matches.home_score/away_score`). Se os placares não estiverem preenchidos no banco, a cobertura de ROI será 0.

| Indicador | Valor |
|---|---:|
| Jogos únicos no recorte | 307 |
| Jogos com placar disponível (home_score/away_score não nulos) | 255 |
| Jogos com status='finished' no banco | 255 |

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
| 2026-02-11 | 37 | 27 | 73.0% |

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
