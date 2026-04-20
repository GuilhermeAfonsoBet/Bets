# Análise Estatística Robusta — Contexto Operação (b808)
**Data da execução:** 22/02/2026 11:52 UTC  
**Escopo:** H3B (auditoria), com comparação por `audit_version` e inferência robusta por jogo (cluster bootstrap).  
**Nota:** a robustez aqui significa que intervalos de confiança consideram correlação intra-jogo (múltiplas auditorias por partida).
---
## 0) Sumário executivo (leitura rápida)
- **Recorte**: direction=`up`, lookback_days=`9`, versions=`v4.0-api`.
- **Amostra**: 4049 auditorias (jogos únicos=420, média=9.6 obs/jogo); betslip confiável=2267.
- **Janela efetiva (audited_at)**: 13/02 12:01 → 19/02 17:39 UTC (span≈6.2d; dias com dados=5).
- **Alerta**: lookback_days=9, mas a janela efetiva observada foi menor (span≈6.2d). Isso costuma indicar falta de auditorias antigas para essas `audit_version` (ou recorte por regime/qualidade).
- **Coortes (status=OK, betslip confiável)**: `BS>WS` (diff>=2.0%): **648**; `BS<WS` (diff<=-2.0%): **221**.
- **Coberturas em `hypothesis_details` (OK)**: temporal(back)=1952/2267; lay_temporal=1855/2267; finance=1877/2267.
- **Cobertura de placar (ROI)**: jogos com placar=376/420 (status finished=376).
- **Cobertura de closing_odd (AH)**: jogos com closing=281/420 (66.9%). CLV pre‑match depende disso.
- **Alerta**: cobertura de closing_odd está baixa. Isso tende a reduzir N de CLV e pode enviesar a leitura (os jogos “sem odds” ficam fora do CLV). Priorize estabilizar o collector e/ou condicionar análises a jogos com closing.
- **DOM**: sem dados no recorte atual (N=0), então não há comparação API vs DOM aqui.
- **CLV pre-match (Betslip, API)**: média robusta por jogo +1.418% (IC90 [+1.089%, +1.738%]), com N=1263 eventos (jogos=243).
- **Padrão por bucket (CLV PM)**: `BS < WS` -3.528% (sig. negativo), `BS ~ WS` -0.184% (NS), `BS > WS` +6.457% (sig. positivo).
- **Leitura recomendada**: use este relatório para validar **qualidade de execução/CLV**. Para concluir sobre **ROI**, rode após atualizar resultados e/ou use uma janela com jogos já liquidados.

---
## 1) Contexto do corte (b808)
| Indicador | Valor |
|---|---:|
| Auditorias H3B `UP` (match + kickoff passado) | 4049 |
| Betslip bruto | 3303 |
| Betslip confiável (diff -10% a +10%) | 2267 |
| Descartados no filtro de qualidade | 1036 |
| Jogos únicos (geral) | 420 |
| Média de observações por jogo | 9.6 |
| Jogos únicos com betslip confiável | 398 |
| Distribuição por market_type | AH=4049 |
| Jogos únicos (AH) no recorte | 420 |
| Jogos únicos (AH) com closing_odd disponível | 281 |
| Cobertura closing_odd (AH) | 66.9% |

---
## 2) Base comparativa: API vs DOM
| Métrica | API (v4.0-api) | DOM (v1.0) |
|---|---:|---:|
| Total de observações | 4049 | 0 |
| Com betslip confiável | 2267 | 0 |
| Com CLV pre-match (betslip) | 1263 | 0 |
| Com ROI (betslip) | 2031 | 0 |
| Tempo total observado (detecção→betslip, wall/total) | 9548 ms | — ms |
| Tempo instrumentado (detecção→clique→betslip) | 8270 ms | — ms |

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
| API (2-4s) | lag_det→click | 5575 | 780 | 5868 | 4049 |
| API (2-4s) | lag_click→betslip | 2609 | 2146 | 4397 | 3846 |
| API (2-4s) | lag_e2e (soma) | 8270 | 3402 | 8494 | 3846 |
| API (2-4s) | audit_total (duração) | 9544 | 4264 | 21338 | 4049 |
| API (2-4s) | overhead (total - e2e) | 1464 | 41 | 2914 | 3846 |
| DOM (15-30s) | lag_det→click | — | — | — | 0 |
| DOM (15-30s) | lag_click→betslip | — | — | — | 0 |
| DOM (15-30s) | lag_e2e (soma) | — | — | — | 0 |
| DOM (15-30s) | audit_total (duração) | — | — | — | 0 |
| DOM (15-30s) | overhead (total - e2e) | — | — | — | 0 |

**Diagnóstico de cauda (percentual acima do limiar)**

| Modelo | % det→click > 5s | % det→click > 20s | % total > 10s | % total > 40s | % overhead < 0 |
|---|---:|---:|---:|---:|---:|
| API (2-4s) | 5.4% | 2.8% | 6.5% | 2.2% | 0.0% |
| DOM (15-30s) | —% | —% | —% | —% | —% |

---
### 2.1 Cobertura temporal (pre-match vs in-match)
| Métrica | Pre-match | In-match | Observação |
|---|---:|---:|---|
| Observações totais com classificação temporal | 2002 | 2047 | Contagem bruta do corte |
| ROI Betslip | 1321 | 710 | Amostra com resultado do jogo |
| ROI WebSocket | 1858 | 1755 | Referência de mercado |
| CLV (apenas pre-match) | 1263 | — | CLV vs closing pré-jogo não é interpretável in-match |

---
### 2.2 Performance por regime (pre-match vs in-match)
| Regime | N auditorias | N betslip conf. | N OK | Back edge | Lay edge | Diff médio (OK) |
|---|---:|---:|---:|---:|---:|---:|
| PRE_MATCH | 2002 | 1415 | 1415 | 419 | 79 | +1.595% |
| IN_MATCH | 2047 | 852 | 852 | 229 | 142 | +0.808% |

---
### 2.3 Regimes operacionais por tempo total (bucket)
Objetivo: separar a amostra em **regimes de execução** (tempo total) e medir performance em cada regime. Use isso para detectar **fila/saturação**: regimes lentos tendem a ter edge menor e pior ROI.

| Bucket (tempo total) | N OK | Jogos | lag_total mean (ms) | lag_total p95 (ms) | overhead p95 (ms) | Back edge | Lay edge | CLV PM (mean; IC90) | ROI (mean; IC90) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| < 5s | 1860 | 375 | 3378 | 4816 | 2479 | 477 | 160 | +1.50% [+1.18%, +1.83%] | +4.31% [-0.73%, +9.28%] |
| 5-10s | 274 | 177 | 5934 | 8378 | 2702 | 110 | 47 | +1.71% [+0.95%, +2.45%] | -5.07% [-15.74%, +5.52%] |
| 10-20s | 19 | 18 | 13480 | 18055 | 3845 | 9 | 2 | +1.15% [-1.31%, +3.55%] | +14.50% [-17.82%, +45.29%] |
| 20-40s | 66 | 42 | 27426 | 34115 | 29505 | 29 | 9 | +2.91% [+1.01%, +4.78%] | +10.88% [-11.42%, +34.14%] |
| > 40s | 48 | 39 | 207638 | 546379 | 30061 | 23 | 3 | +1.88% [+0.63%, +3.18%] | -19.01% [-42.28%, +4.42%] |
| Desconhecido | 0 | 0 | — | — | — | 0 | 0 | — | — |

---
### 2.3b Regimes por tempo total — separando coortes por `diff_pct` (BS vs WS)
Nesta tabela, separamos duas coortes operacionais por **delta de execução**: `BS > WS` (diff_pct >= +2%) e `BS < WS` (diff_pct <= -2%). Isso **não** é (por si só) “Back vs Lay”; é um recorte por **melhora/piora do preço** entre detecção (WS) e execução (BS). CLV é reportado apenas em pre‑match.

| Bucket | N OK | N (BS>WS +2%) | N (BS<WS -2%) | CLV PM (BS>WS) | CLV PM (BS<WS) | ROI (BS>WS) | ROI (BS<WS) |
|---|---:|---:|---:|---:|---:|---:|---:|
| < 5s | 1860 | 477 | 160 | +6.80% [+6.35%, +7.26%] | -2.81% [-3.89%, -1.72%] | +1.31% [-8.07%, +10.68%] | -0.05% [-13.86%, +13.72%] |
| 5-10s | 274 | 110 | 47 | +5.64% [+4.84%, +6.37%] | -5.46% [-6.79%, -3.99%] | -6.58% [-23.32%, +10.42%] | -10.83% [-31.47%, +9.70%] |
| 10-20s | 19 | 9 | 2 | +3.92% [+1.34%, +6.75%] | -6.98% — | -9.86% [-55.47%, +37.18%] | +40.04% [+0.00%, +79.90%] |
| 20-40s | 66 | 29 | 9 | +7.11% [+5.33%, +9.04%] | -7.89% — | +10.45% [-20.60%, +41.31%] | +28.14% [-36.60%, +93.83%] |
| > 40s | 48 | 23 | 3 | +5.33% [+3.36%, +7.28%] | — | -77.07% [-92.65%, -54.54%] | -50.11% [-100.00%, +0.00%] |
| Desconhecido | 0 | 0 | 0 | — | — | — | — |

---
### 2.3c Estabilidade temporal (por dia, `audited_at`)
Objetivo: checar se o regime de edge/execução é **time‑dependent**. Se houver dias com comportamento distinto (ex.: collector intermitente, horários/ligas), isso aparecerá aqui.

| Dia (UTC) | Modelo | N OK | Jogos | % BS>WS +2% | % BS<WS -2% | lag_total p50 (ms) | CLV PM (BS>WS) | CLV PM (BS<WS) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-02-13 | API (2-4s) | 485 | 131 | 37.7% | 10.3% | 4384 | +6.78% | -3.48% |
| 2026-02-14 | API (2-4s) | 530 | 158 | 35.3% | 14.7% | 4200 | +6.41% | -5.42% |
| 2026-02-15 | API (2-4s) | 475 | 123 | 31.4% | 11.6% | 3686 | +5.96% | -1.78% |
| 2026-02-16 | API (2-4s) | 291 | 94 | 40.5% | 4.5% | 3535 | +6.45% | -6.79% |
| 2026-02-19 | API (2-4s) | 486 | 104 | 2.3% | 5.1% | 2244 | +1.02% | -2.31% |

---
## 3) CLV pre-match (núcleo)
### 3.1 CLV com odd do Betslip (execução real)
| Métrica | API | DOM |
|---|---:|---:|
| CLV Bruto BS Pre-Match | +1.653% (sig. positivo, N=1263, jogos=243) | — (N/A, N=0, jogos=0) |
| CLV Adicional BS Pre-Match | +1.377% (sig. positivo, N=1260, jogos=243) | — (N/A, N=0, jogos=0) |
| Taxa de CLV > 0 (bruto) | 57.1% | —% |
| Taxa de CLV > 0 (adicional) | 58.2% | —% |

Notas de robustez (IC 90% por jogo):  
- API CLV bruto (cluster): média +1.418%; IC90 [+1.089%, +1.738%]  
- DOM CLV bruto (cluster): média —; IC90 —  

---
## 4) ROI por modelo
| Métrica | API | DOM |
|---|---:|---:|
| ROI Betslip | +0.879% (NS, N=2027) | — (N/A, N=0) |
| ROI WebSocket | -0.414% (NS, N=3596) | — (N/A, N=0) |
| Win rate ROI Betslip | 50.6% | —% |
| Win rate ROI WS | 50.3% | —% |

Notas de robustez (IC 90% por jogo):  
- API ROI betslip (cluster): média +2.415%; IC90 [-2.355%, +7.079%]  
- API ROI WS (cluster): média -1.378%; IC90 [-4.754%, +2.014%]  

---
## 5) Diferença de preço BS vs WS
| Métrica | API | DOM |
|---|---:|---:|
| Diff BS vs WS (média) | +1.299% (sig. positivo, N=2267) | — (N/A, N=0) |
| BS > WS | 43.1% (978/2267) | —% (0/0) |
| BS > WS +2% | 28.6% (648/2267) | —% (0/0) |

---
## 6) Combinações de valor
### 6.1 Buckets por diferença BS vs WS
Nota de leitura:
- `BS > WS (+2% a +10%)`: preço no betslip ficou **melhor** do que o WS observado na detecção (delta positivo entre instantes).
- `BS < WS (-10% a -2%)`: preço no betslip ficou **pior** do que o WS observado na detecção (delta negativo entre instantes).
- O ROI abaixo é calculado **dentro do bucket** (não mistura buckets), mas ainda é sensível à cobertura de placar.

| Bucket | N bucket | CLV BS PM (média) | IC90 (cluster) | N CLV PM | Jogos CLV PM | ROI BS (todos) | IC90 (cluster) |
|---|---:|---:|---|---:|---:|---:|---|
| BS < WS (-10% a -2%) | 221 | -3.528% | [-4.376%, -2.455%] | 65 | 53 | +2.016% | [-14.896%, +8.073%] |
| BS ~ WS (-2% a +2%) | 1398 | -0.184% | [-0.416%, +0.105%] | 816 | 224 | +1.043% | [-5.294%, +5.791%] |
| BS > WS (+2% a +10%) | 648 | +6.457% | [+6.201%, +6.989%] | 382 | 140 | +0.146% | [-5.354%, +11.299%] |

---
### 6.2 Combinação por faixa de linha AH
| Faixa AH | CLV BS PM (média) | IC90 (cluster) | ROI BS (todos) | IC90 (cluster) | Diff BS vs WS (média) |
|---|---:|---|---:|---|---:|
| AH 0-1 (líquida) | +1.801% | [+1.314%, +2.270%] | -1.173% | [-2.186%, +10.982%] | +1.465% |
| AH 1-2 (média) | +1.689% | [+0.733%, +2.266%] | +5.284% | [-6.751%, +13.580%] | +1.708% |
| AH 2+ (extrema) | +1.363% | [+0.401%, +1.629%] | +0.663% | [-3.894%, +10.687%] | +0.951% |

---
### 6.3 Combinação por faixa de lag
| Faixa de lag | CLV BS PM (média) | IC90 (cluster) | N CLV PM | Jogos CLV PM | ROI BS (todos) | IC90 (cluster) | Diff BS vs WS (média) |
|---|---:|---|---:|---:|---:|---|---:|
| < 10s | +1.619% | [+1.097%, +1.745%] | 1198 | 239 | +0.933% | [-2.225%, +7.802%] | +1.239% |
| 10-20s | +1.122% | [-1.307%, +3.553%] | 8 | 8 | +13.805% | [-17.823%, +45.289%] | +1.721% |
| 20-30s | +2.620% | [+0.863%, +4.664%] | 23 | 22 | +10.526% | [-10.993%, +35.853%] | +2.079% |
| > 30s | +2.314% | [+0.915%, +3.593%] | 34 | 30 | -14.709% | [-40.214%, +4.096%] | +2.600% |

---
## 7) Estimativa financeira (proxy) e risco
Este bloco usa `hypothesis_details.finance` quando existe; se não existir, usa stake fallback = `stake_pct_of_limit × limit`.

**Política de stake (proxy)**

| Parâmetro | Valor |
|---|---:|
| stake_pct_of_limit | 0.25 |
| stake_cap | 0.00 |
| Cobertura finance (OK, betslip conf.) | 1877/2267 |

### 7.1 Back (BS >> WS)
| Métrica | Valor |
|---|---:|
| Corte (diff_pct) | >= 2.0% |
| N eventos | 648 |
| Cobertura finance (na coorte) | 499/648 |
| Stake total (estimado) | 201167.66 |
| Stake médio | 310.44 |
| Profit_if_win total (estimado) | 214183.77 |
| Profit_if_win médio | 330.53 |
| N com ROI realizado | 586 |
| P&L realizado total (estimado) | -39005.47 |
| ROI realizado (ponderado por stake) | -19.94% |
| ROI realizado (robusto por jogo, mean; IC90) | +3.04% [-5.35%, +11.30%] |
| ROI ponderado por stake (robusto por jogo, mean; IC90) | +0.90% [-9.57%, +11.70%] |

**Como ler as 3 linhas de ROI (Back)**

- **ROI realizado (ponderado por stake)**: ΣP&L / Σstake (pode ser dominado por stakes grandes).
- **ROI realizado (robusto por jogo)**: média por jogo (cada jogo pesa parecido; reduz dominância de outliers).
- **ROI ponderado por stake (robusto por jogo)**: calcula ROI ponderado dentro do jogo e depois faz média/IC por jogo.

Observação: a Seção 4 reporta ROI médio no **recorte completo** (betslip confiável). Aqui (7.1) é apenas a coorte **Back edge** (diff>=corte) e o ROI também é mostrado ponderado por stake; por isso sinais podem divergir.

### 7.2 Lay (BS << WS) — risco de cauda
| Métrica | Valor |
|---|---:|
| Corte (diff_pct) | <= -2.0% |
| N eventos | 221 |
| Cobertura finance (na coorte) | 179/221 |
| Stake total (estimado) | 39023.47 |
| Liability total (estimada) | 35253.11 |
| Liability média | 159.52 |
| Liability p95 | 557.59 |
| Liability p99 | 2198.46 |
| ES95 (liability) | 1627.34 |
| Liability max | 4395.35 |
| Proxy de banca (>= p99 liability) | 2198.46 |
| N com ROI realizado | 141 |
| P&L realizado total (estimado) | -10276.73 |
| ROI realizado (ponderado por liability) | -30.66% |
| ROI realizado (ponderado por stake) | -27.59% |
| ROI/liability (robusto por jogo, mean; IC90) | +17.46% [+2.24%, +32.66%] |
| ROI/liability ponderado (robusto por jogo, mean; IC90) | +16.51% [+0.71%, +32.43%] |

### 7.3 Projeção mensal (30 dias fixo) — turnover, lucro e banca
Premissas:
- **Mês = 30 dias fixo**.
- Turnover = soma de stakes (Back e Lay). Para Lay, também reportamos exposição por **liability**.
- **Banca por risco (unitária)** = p99/ES95(exposição por aposta).
- **Banca por liquidez (turnover)** = p95/p99 de capital **simultaneamente travado** (stake/liability em aberto até a liquidação), capturando distribuição desigual de entradas ao longo do tempo.
- Projeção de lucro usa duas visões: (i) **lucro/dia observado** e (ii) **ROI observado × turnover projetado**.

| Bloco | Janela(d) | Turnover 30d | Lucro 30d (direto) | Lucro 30d (ROI×turnover) |
|---|---:|---:|---:|---:|
| Back | 9.0 | 670558.86 | -130018.22 | -133682.02 |
| Lay (stake) | 9.0 | 130078.22 | -34255.76 | -35893.07 |
| Total (Back+Lay) | 9.0 | 800637.09 | -164273.98 | -169575.09 |

**Banca por risco (exposição unitária)**

| Bloco | Banca conservadora (p99) | Banca agressiva (ES95) | ROI/banca 30d (direto) | ROI/banca 30d (ROI×turnover) |
|---|---:|---:|---:|---:|
| Back (stake) | 4394.20 | 3850.68 | -2958.86% | -3042.24% |
| Lay (liability) | 2198.46 | 1627.34 | -1558.17% | -1632.65% |
| Total (soma) | 6592.66 | 5478.03 | -2491.77% | -2572.18% |

**Banca por liquidez (capital simultaneamente travado)**

Definição operacional: cada aposta trava capital de `audited_at` até `kickoff + 2.00h + 2.25h` (grid=5min). A banca recomendada aplica buffer de +10%.

| Bloco | Liquidez mean | Liquidez p95 | Liquidez p99 | Liquidez max | Banca liq p99 (+buffer) |
|---|---:|---:|---:|---:|---:|
| Back (stake) | 14364.81 | 44655.20 | 76162.43 | 85932.63 | 83778.67 |
| Lay (liability) | 2275.79 | 6347.27 | 9424.51 | 11627.51 | 10366.96 |
| Total (Back+Lay) | 16478.19 | 49503.05 | 82978.75 | 92653.13 | 91276.63 |

**Banca recomendada (conservadora)**

| Métrica | Valor |
|---|---:|
| Banca por risco (p99 unitário, soma) | 6592.66 |
| Banca por liquidez (p99 simultâneo + buffer) | 91276.63 |
| Banca efetiva (max das duas) | 91276.63 |
| ROI/banca 30d (direto, banca efetiva) | -179.97% |
| ROI/banca 30d (ROI×turnover, banca efetiva) | -185.78% |
**Diagnóstico de cobertura (placar/ROI)**

| Bloco | Turnover total | Turnover com ROI | Cobertura turnover |
|---|---:|---:|---:|
| Back | 201167.66 | 195654.30 | 97.26% |
| Lay | 39023.47 | 37243.36 | 95.44% |

Notas (Lay): exposição 30d por liability (não é turnover) = 117510.38; ROI realizado por liability (ponderado) = -30.66%.

---
## 8) Curva temporal (pico, reversão e melhor timing)
Esta seção usa `hypothesis_details.temporal` (Back) e `hypothesis_details.lay_temporal` (Lay), coletados em pontos discretos (t≈0,3,6,10,15,20s).

O objetivo é responder: **tempo até o pico/vale**, **% que segue melhorando até t_max**, **% com reversão** e **impacto esperado por timing**. **CLV é reportado somente pre-match** (closing pré-jogo).

### 8.1 Back (pico em diff_pct)
Definições: **pico** = `max(diff_pct)`. **reversão** = após o pico, `diff_pct` cair pelo menos 0.50 p.p. (para Lay, subir 0.50 p.p. após o vale). `t_reversão` é o 1º tempo que cruza esse limiar (logo, não é o pico).

| Regime | N | t_pico médio (s) | t_pico p50 (s) | % melhora até fim | % com reversão | t_reversão médio (s) | Δt (rev - pico) médio (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRE_MATCH | 1415 | 4.2 | 4.8 | 76.7% | 15.8% | 12.4 | 7.3 |
| IN_MATCH | 852 | 5.5 | 0.0 | 58.5% | 32.6% | 13.1 | 8.1 |

**Partição 100% (Back)**: categorias exclusivas (somam 100% por regime).

| Regime | % pico no fim, sem reversão | % pico no fim, com reversão (recuperou) | % pico antes, com reversão | % pico antes, sem reversão relevante |
|---|---:|---:|---:|---:|
| PRE_MATCH | 80.8% | 6.9% | 8.9% | 3.4% |
| IN_MATCH | 64.7% | 5.4% | 27.2% | 2.7% |

**Curva média (Back)**: média de `diff_pct` e `odd` por tempo. `CLV` é reportado **somente pre-match** (closing pré-jogo). `ROI` (quando aparece) é o **ROI realizado** se a aposta fosse feita naquele ponto.

| Tempo | N pts | diff_pct médio | odd média | CLV médio (pre-match) | ROI médio (se houver) |
|---|---:|---:|---:|---:|---:|
| t+0s | 2267 | +1.30% | 2.013 | +1.65% | 2.36 |
| t+6s | 2320 | +1.25% | 2.009 | +1.43% | 2.29 |
| t+10s | 2878 | +2.24% | 2.029 | +2.07% | 2.94 |
| t+15s | 1929 | +1.98% | 2.034 | +1.86% | 3.31 |
| t+20s | 2490 | +2.09% | 2.027 | +2.08% | -0.08 |

### 8.1b Back — impacto por timing (t0 vs pico vs último) e reversão
Leitura: se a estratégia é **entrar no pico**, compare CLV/ROI no pico vs t0 e veja diferença entre **casos com reversão** vs **sem reversão**. Aqui reportamos **média robusta por jogo + IC90**. Observação: **CLV só é calculado pre-match**.

**CLV (somente pre-match) — média robusta por jogo (IC90)**

| Subcoorte | N total | N CLV (PM) | CLV t0 (mean; IC90) | CLV pico (mean; IC90) | CLV último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 1765 | 1073 | +1.25% [+0.92%, +1.57%] | +1.55% [+1.20%, +1.90%] | +1.54% [+1.19%, +1.89%] |
| COM_REVERSAO | 502 | 190 | +2.96% [+2.28%, +3.62%] | +4.23% [+3.48%, +4.99%] | +3.43% [+2.67%, +4.21%] |

**ROI (stake=1) — média robusta por jogo (IC90)**

| Subcoorte | N total | N ROI | ROI t0 (mean; IC90) | ROI pico (mean; IC90) | ROI último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 1765 | 1587 | +2.32% [-3.06%, +7.56%] | +2.77% [-2.63%, +8.04%] | +2.76% [-2.65%, +8.02%] |
| COM_REVERSAO | 502 | 440 | +2.27% [-6.85%, +11.36%] | +4.85% [-4.61%, +14.25%] | +2.37% [-6.64%, +11.47%] |

Nota interpretativa: a subcoorte **COM_REVERSAO** pode ter CLV maior no **pico** porque, por definição, ela inclui casos em que o edge atingiu um extremo mais alto antes de reverter. Para entender a hipótese 'sem reversão deveria ser maior', compare também a categoria '**pico no fim, sem reversão**' na partição 100% (tabela 8.1).

**Decomposição (pre-match): entry_odd vs closing_odd (IC90)**

| Subcoorte | N CLV (PM) | entry_odd t0 (mean; IC90) | entry_odd pico (mean; IC90) | closing_odd (mean; IC90) |
|---|---:|---:|---:|---:|
| SEM_REVERSAO | 1073 | 1.977 [+1.969, +1.985] | 1.984 [+1.976, +1.993] | 1.962 [+1.955, +1.969] |
| COM_REVERSAO | 190 | 2.019 [+2.003, +2.034] | 2.046 [+2.029, +2.062] | 1.967 [+1.955, +1.979] |

### 8.2 Lay (vale em diff_pct)
| Regime | N | t_vale médio (s) | t_vale p50 (s) | % melhora até fim | % com reversão | t_reversão médio (s) | Δt (rev - vale) médio (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRE_MATCH | 1289 | 4.4 | 5.2 | 66.2% | 23.9% | 12.3 | 6.9 |
| IN_MATCH | 697 | 6.4 | 5.3 | 45.2% | 43.6% | 13.3 | 7.7 |

**Partição 100% (Lay)**: categorias exclusivas (somam 100% por regime).

| Regime | % vale no fim, sem reversão | % vale no fim, com reversão (recuperou) | % vale antes, com reversão | % vale antes, sem reversão relevante |
|---|---:|---:|---:|---:|
| PRE_MATCH | 71.0% | 10.0% | 13.9% | 5.1% |
| IN_MATCH | 52.9% | 6.9% | 36.7% | 3.4% |

**Curva média (Lay)**: usa `odd=lay_odd`. `CLV` é reportado **somente pre-match** (closing pré-jogo). O ROI mostrado aqui é **ROI por liability** (não por stake), porque Lay é governado por risco.

| Tempo | N pts | diff_pct médio | lay_odd média | CLV médio (pre-match) | ROI/liability médio (se houver) |
|---|---:|---:|---:|---:|---:|
| t+0s | 1986 | +2.35% | 2.045 | -1.25% | 6.13 |
| t+6s | 2228 | +3.33% | 2.061 | -1.83% | 13.17 |
| t+10s | 2718 | +1.63% | 2.031 | -0.78% | 17.87 |
| t+15s | 1838 | +2.92% | 2.071 | -1.75% | 7.94 |
| t+20s | 2374 | +2.29% | 2.041 | -0.88% | 9.48 |

### 8.2b Lay — impacto por timing (t0 vs vale vs último) e reversão
Leitura: se a estratégia é **entrar no vale (odd mais baixa)**, compare CLV/ROI no vale vs t0 e veja diferença entre **casos com reversão** vs **sem reversão**. Aqui reportamos **média robusta por jogo + IC90**. Observação: **CLV só é calculado pre-match**.

**CLV (somente pre-match) — média robusta por jogo (IC90)**

| Subcoorte | N total | N CLV (PM) | CLV t0 (mean; IC90) | CLV vale (mean; IC90) | CLV último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 1374 | 862 | -1.73% [-2.25%, -1.21%] | -1.43% [-1.94%, -0.92%] | -1.46% [-1.97%, -0.94%] |
| COM_REVERSAO | 612 | 277 | -2.38% [-3.32%, -1.47%] | -1.10% [-1.97%, -0.28%] | -2.16% [-3.01%, -1.36%] |

**ROI/liability — média robusta por jogo (IC90)**

| Subcoorte | N total | N ROI | ROI/liab t0 (mean; IC90) | ROI/liab vale (mean; IC90) | ROI/liab último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 1374 | 1238 | -2.69% [-9.08%, +4.11%] | -2.12% [-8.65%, +4.80%] | -2.14% [-8.67%, +4.78%] |
| COM_REVERSAO | 612 | 543 | +13.13% [+3.49%, +22.72%] | +16.29% [+5.80%, +26.95%] | +12.82% [+2.89%, +22.75%] |

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
| Back | Pre | Any | 1415 | 282 | +1.42% [+1.09%, +1.74%] | -1.42% [-7.28%, +4.52%] | -3.18% | não (CLV p90>0 AND ROI>0) |
| Back | In | Any | 852 | 230 | — | +5.94% [-0.91%, +12.62%] | +3.89% | sim (ROI p30>0) |
| Lay | Pre | Yes | 308 | 156 | +3.21% [+2.32%, +4.13%] | +7.82% [-3.20%, +18.70%] | +4.39% | sim (CLV_CONV p90>0 AND ROI p30>0) |
| Lay | Pre | No | 981 | 247 | +1.46% [+0.94%, +1.97%] | -5.83% [-12.64%, +1.38%] | -8.01% | não (CLV_CONV p90>0 AND ROI p30>0) |
| Lay | In | Yes | 304 | 149 | — | +22.64% [+5.57%, +41.10%] | +16.89% | sim (ROI p30>0) |
| Lay | In | No | 393 | 175 | — | +3.50% [-8.66%, +15.62%] | -0.27% | não (ROI p30>0) |

Notas:
- Se você quiser operar **cada combinação como uma estratégia separada**, o sizing (Kelly/caps) deve ser estimado e governado separadamente por estratégia.
- Esta tabela é **in-sample** na janela (`--lookback-days`). A seção OOS (walk-forward) pode ser habilitada com `--walkforward`.

## 9) Combinações de valor (regime × linha AH × lag)
Tabela rankeada por volume (N). Métrica: diff_pct (OK, betslip confiável).

### 9.1 Back combos (diff_pct >= corte)
| Regime | Linha | Lag | N | Jogos | Mean diff % | IC90 cluster |
|---|---|---|---:|---:|---:|---|
| PRE_MATCH | AH 0-1 (líquida) | < 10s | 186 | 74 | +6.88% | [+6.82%, +7.37%] |
| IN_MATCH | AH 2+ (extrema) | < 10s | 122 | 92 | +6.14% | [+5.73%, +6.57%] |
| PRE_MATCH | AH 1-2 (média) | < 10s | 107 | 42 | +6.37% | [+6.18%, +6.99%] |
| PRE_MATCH | AH 2+ (extrema) | < 10s | 96 | 45 | +6.69% | [+6.23%, +7.01%] |
| IN_MATCH | AH 0-1 (líquida) | < 10s | 46 | 38 | +6.62% | [+6.00%, +7.14%] |
| IN_MATCH | AH 1-2 (média) | < 10s | 30 | 29 | +6.74% | [+5.86%, +7.37%] |
| IN_MATCH | AH 2+ (extrema) | > 30s | 9 | 6 | +6.87% | [+3.93%, +8.00%] |
| PRE_MATCH | AH 0-1 (líquida) | 20-30s | 8 | 8 | +6.52% | [+5.58%, +7.43%] |
| PRE_MATCH | AH 0-1 (líquida) | > 30s | 6 | 5 | +6.05% | [+4.73%, +6.99%] |
| IN_MATCH | AH 0-1 (líquida) | > 30s | 6 | 4 | +6.91% | [+5.89%, +7.94%] |
| IN_MATCH | AH 2+ (extrema) | 20-30s | 6 | 5 | +6.62% | [+4.61%, +8.04%] |
| PRE_MATCH | AH 2+ (extrema) | > 30s | 5 | 4 | +5.60% | [+4.20%, +7.19%] |

### 9.2 Lay combos (diff_pct <= corte) + risco
| Regime | Linha | Lag | N | Jogos | Mean diff % | IC90 cluster | Liability p95 |
|---|---|---|---:|---:|---:|---|---:|
| IN_MATCH | AH 2+ (extrema) | < 10s | 79 | 65 | -4.89% | [-5.36%, -4.52%] | 293.88 |
| IN_MATCH | AH 0-1 (líquida) | < 10s | 36 | 30 | -4.99% | [-5.73%, -4.52%] | 560.12 |
| PRE_MATCH | AH 2+ (extrema) | < 10s | 31 | 28 | -4.64% | [-5.26%, -4.06%] | 709.00 |
| PRE_MATCH | AH 0-1 (líquida) | < 10s | 29 | 24 | -4.56% | [-5.56%, -3.99%] | 3522.14 |
| PRE_MATCH | AH 1-2 (média) | < 10s | 17 | 13 | -5.33% | [-6.47%, -4.45%] | 103.03 |
| IN_MATCH | AH 1-2 (média) | < 10s | 15 | 13 | -4.45% | [-5.21%, -3.62%] | 635.14 |
| IN_MATCH | AH 2+ (extrema) | > 30s | 4 | 4 | -3.21% | [-3.80%, -2.72%] | 183.03 |
| IN_MATCH | AH 2+ (extrema) | 20-30s | 3 | 3 | -5.17% | [-5.79%, -4.54%] | 273.15 |
| IN_MATCH | AH 0-1 (líquida) | 20-30s | 2 | 2 | -3.37% | [-3.76%, -2.99%] | 179.40 |
| IN_MATCH | AH 0-1 (líquida) | > 30s | 2 | 2 | -7.52% | [-9.10%, -5.93%] | 220.20 |
| IN_MATCH | AH 2+ (extrema) | 10-20s | 1 | 1 | -5.81% | — | 84.09 |
| PRE_MATCH | AH 0-1 (líquida) | 20-30s | 1 | 1 | -5.84% | — | 0.00 |

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

- **Back (stake)**: corr(exposição, ROI)=-0.073; corr(exposição, CLV)=0.003 (onde CLV existe).
- **Lay (liability)**: corr(exposição, ROI)=0.085; corr(exposição, CLV)=0.064 (onde CLV existe).

**Backtest de sizing (apenas eventos com placar; valores em unidade monetária *proxy*)**

| Lado | Scheme | N | Turnover (janela) | Lucro (janela) | ROI/turnover | p99 exposição | ES95 exposição | DD 30d (média) | DD 30d (p95) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Back | FLAT | 586 | 586.00 | 7.36 | 1.26% | 1.00 | 1.00 | 20.43 | 40.25 |
| Lay | FLAT | 197 | 235.36 | -3.67 | -1.56% | 1.00 | 1.00 | 25.68 | 47.79 |
| Back | PROXY | 586 | 195654.30 | -39005.47 | -19.94% | 4395.20 | 4060.25 | 130836.77 | 197564.71 |
| Lay | PROXY | 141 | 37243.36 | -10276.73 | -27.59% | 3522.14 | 2153.18 | 34360.85 | 47956.43 |
| Back | KELLY_0.10 | 338 | 8652.41 | 214.77 | 2.48% | 57.84 | 53.64 | 746.98 | 1445.87 |
| Lay | KELLY_0.10 | 44 | 625.95 | 82.25 | 13.14% | 21.98 | 21.98 | 64.02 | 122.22 |
| Back | KELLY_0.25 | 338 | 17709.42 | 225.50 | 1.27% | 87.88 | 87.88 | 1467.69 | 2719.00 |
| Lay | KELLY_0.25 | 44 | 905.48 | 123.37 | 13.63% | 21.98 | 21.98 | 91.20 | 169.74 |
| Back | KELLY_0.50 | 338 | 21940.01 | -148.37 | -0.68% | 87.88 | 87.88 | 2232.17 | 4263.05 |
| Lay | KELLY_0.50 | 44 | 988.97 | 101.99 | 10.31% | 21.98 | 21.98 | 92.70 | 167.96 |
| Back | KELLY_1.00 | 338 | 22779.85 | -302.61 | -1.33% | 87.88 | 87.88 | 2520.91 | 4782.71 |
| Lay | KELLY_1.00 | 44 | 994.66 | 101.15 | 10.17% | 21.98 | 21.98 | 94.19 | 173.99 |

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
| Back | Pre | Yes | FLAT | 95 | 95.00 | 2.04 | 2.15% | 1.00 | 29.30 |
| Back | Pre | Yes | PROXY | 95 | 31262.59 | -9308.05 | -29.77% | 4597.59 | 54732.84 |
| Back | Pre | Yes | KELLY_0.10 | 81 | 2245.98 | 40.04 | 1.78% | 44.81 | 657.45 |
| Back | Pre | Yes | KELLY_0.25 | 81 | 4833.23 | 133.80 | 2.77% | 87.88 | 1188.81 |
| Back | Pre | Yes | KELLY_0.50 | 81 | 5877.62 | 218.39 | 3.72% | 87.88 | 1181.59 |
| Back | Pre | Yes | KELLY_1.00 | 81 | 6023.65 | 133.59 | 2.22% | 87.88 | 1525.65 |
| Back | Pre | No | FLAT | 302 | 302.00 | -6.84 | -2.27% | 1.00 | 80.33 |
| Back | Pre | No | PROXY | 302 | 122504.39 | -18030.42 | -14.72% | 4415.85 | 102533.46 |
| Back | Pre | No | KELLY_0.10 | 257 | 6406.43 | 174.73 | 2.73% | 58.81 | 1508.69 |
| Back | Pre | No | KELLY_0.25 | 257 | 12876.18 | 91.71 | 0.71% | 87.88 | 2821.25 |
| Back | Pre | No | KELLY_0.50 | 257 | 16062.39 | -366.76 | -2.28% | 87.88 | 4940.26 |
| Back | Pre | No | KELLY_1.00 | 257 | 16756.20 | -436.19 | -2.60% | 87.88 | 5455.91 |
| Back | In | Yes | FLAT | 59 | 59.00 | -5.52 | -9.36% | 1.00 | 56.07 |
| Back | In | Yes | PROXY | 59 | 12171.62 | -5297.74 | -43.53% | 2292.06 | 47729.57 |
| Back | In | No | FLAT | 130 | 130.00 | 17.68 | 13.60% | 1.00 | 0.00 |
| Back | In | No | PROXY | 130 | 29715.69 | -6369.26 | -21.43% | 2073.45 | 56200.16 |
| Lay | Pre | Yes | FLAT | 14 | 14.41 | 3.34 | 23.15% | 1.00 | 7.73 |
| Lay | Pre | Yes | PROXY | 14 | 6256.92 | -3739.78 | -59.77% | 3887.37 | 39522.15 |
| Lay | Pre | Yes | KELLY_0.10 | 7 | 57.56 | 39.01 | 67.78% | 16.51 | 0.00 |
| Lay | Pre | Yes | KELLY_0.25 | 7 | 103.23 | 56.87 | 55.09% | 18.28 | 0.00 |
| Lay | Pre | Yes | KELLY_0.50 | 7 | 135.58 | 63.43 | 46.78% | 21.98 | 0.00 |
| Lay | Pre | Yes | KELLY_1.00 | 7 | 138.04 | 65.89 | 47.73% | 21.98 | 0.00 |
| Lay | Pre | No | FLAT | 44 | 48.89 | 0.00 | 0.00% | 1.00 | 15.87 |
| Lay | Pre | No | PROXY | 44 | 14058.52 | -7012.80 | -49.88% | 3462.39 | 35838.58 |
| Lay | Pre | No | KELLY_0.10 | 24 | 380.24 | 9.57 | 2.52% | 21.98 | 140.63 |
| Lay | Pre | No | KELLY_0.25 | 24 | 534.26 | 38.22 | 7.15% | 21.98 | 150.76 |
| Lay | Pre | No | KELLY_0.50 | 24 | 576.30 | 19.62 | 3.41% | 21.98 | 225.69 |
| Lay | Pre | No | KELLY_1.00 | 24 | 576.30 | 19.62 | 3.41% | 21.98 | 218.46 |
| Lay | In | Yes | FLAT | 30 | 35.38 | 5.39 | 15.24% | 1.00 | 4.00 |
| Lay | In | Yes | PROXY | 30 | 6686.10 | 2636.75 | 39.44% | 1653.95 | 2341.75 |
| Lay | In | No | FLAT | 52 | 78.09 | 0.34 | 0.43% | 1.00 | 20.30 |
| Lay | In | No | PROXY | 52 | 10009.93 | -2392.78 | -23.90% | 1040.50 | 17624.25 |
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
| Back | Pre | Yes | 224 | 150 | +2.96% [+2.28%, +3.62%] | +1.30% [-10.05%, +12.84%] | -2.39% | pre: Kelly OK |
| Back | Pre | No | 1191 | 266 | +1.25% [+0.92%, +1.57%] | -1.04% [-7.72%, +5.50%] | -3.04% | pre: Kelly OK |
| Back | In | Yes | 278 | 132 | — — | +1.24% [-11.81%, +14.98%] | -0.05% | in: use FLAT/PROXY |
| Back | In | No | 574 | 208 | — — | +4.96% [-3.34%, +13.38%] | +5.61% | in: use FLAT/PROXY |
| Lay | Pre | Yes | 308 | 156 | +3.21% [+2.32%, +4.13%] | +7.82% [-3.20%, +18.70%] | +4.39% | pre: Kelly OK |
| Lay | Pre | No | 981 | 247 | +1.46% [+0.94%, +1.97%] | -5.83% [-12.64%, +1.38%] | -8.01% | pre: Kelly OK |
| Lay | In | Yes | 304 | 149 | — — | +22.64% [+5.57%, +41.10%] | +16.89% | in: use FLAT/PROXY |
| Lay | In | No | 393 | 175 | — — | +3.50% [-8.66%, +15.62%] | +9.81% | in: use FLAT/PROXY |
| Estratégia | Scheme | N Back (janela) | N Lay (janela) | N Back 30d (proj.) | N Lay 30d (proj.) | Stake médio Back | Stake médio Lay | Liability média Lay | Turnover 30d (proj.) | Lucro 30d (proj.) | ROI/turnover (janela) | ROI Lay/liability (janela) | ROI Lay/turnover (janela) | Banca p99 (Back) | Banca p99 (Lay) | Banca risco p99 (soma) | Banca liquidez p99 (+buf) | Banca recomendada (max) | ROI/banca 30d | Turnover 30d (R$) | Lucro 30d (R$) | Banca rec. (R$) | DD 30d p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ativas (PRE, critérios 8.3) | FLAT | 95 | 0 | 317 | 0 | 1.00 | — | — | 316.67 | 6.80 | 2.15% | —% | —% | 1.00 | — | 1.00 | 45.90 | 45.90 | 14.81% | 1646.67 | 35.34 | 238.70 | 28.72 |
| Ativas (PRE, critérios 8.3) | KELLY_0.25 | 81 | 0 | 270 | 0 | 59.67 | — | — | 16110.78 | 445.99 | 2.77% | —% | —% | 87.88 | — | 87.88 | 2411.57 | 2411.57 | 18.49% | 83776.04 | 2319.13 | 12540.14 | 1198.22 |

Notas:
- **N Back/Lay** na tabela é **na janela observada**. As colunas `N 30d (proj.)` são uma **escala linear** por dias observados.
- Esta linha usa a união das **combinações pre‑match ativas** sob os critérios da 8.3 (é um resumo, não substitui a tabela 8.3).
- `Stake médio` e `Liability média` são **proxies** na unidade monetária do seu limit/finance; valores em **R$** usam fx `--fx-usdbrl`.
- `Banca recomendada` = max( banca por risco p99 (unitária, soma) ; banca por liquidez p99 (+buffer) ).
- **Por que Lay pode ter stake médio menor mesmo com ROI maior**: (i) Lay é governado por **liability** (risco) e usamos cap mais conservador; (ii) stake em Lay é derivado de `liability/(odd-1)` e depende do nível médio de odds; (iii) Kelly depende do **edge vs closing** (gap entry→closing), não do ROI observado isoladamente.
- **Por que não incluímos Back in‑match aqui**: Kelly/CLV usam `closing_odd` pré‑jogo; in‑match requer benchmark diferente (ex.: referência por minuto/VWAP) e governança operacional específica. A seção 2.2 já compara PRE_MATCH vs IN_MATCH.

### 9.4b Curva de capacidade — frações de Kelly e tamanho potencial
Esta tabela é um exercício para estimar **capacidade** (turnover/lucro/risco) ao variar a fração de Kelly. Ela deixa explícito quando o sizing satura por **cap por aposta**.

**Escala Kelly usada nesta curva**: P99_PROXY | ref_back=4394.20 ref_lay=2198.46 | cap_back=2.0% cap_lay=1.0% | max_stake_event=100%*limit

| Strategy | Scheme | cap_hit Back (%) | limit_hit Back (%) | cap_hit Lay (%) | limit_hit Lay (%) | Turnover 30d (proj.) | Lucro 30d (proj.) | ROI/banca 30d | DD 30d p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ativas (PRE, critérios 8.3) | KELLY_0.10 | 0.0% | 11.2% | —% | —% | 7486.59 | 133.46 | 12.35% | 683.18 |
| Ativas (PRE, critérios 8.3) | KELLY_0.25 | 24.7% | 27.0% | —% | —% | 16110.78 | 445.99 | 18.49% | 1186.73 |
| Ativas (PRE, critérios 8.3) | KELLY_0.50 | 85.4% | 30.3% | —% | —% | 19592.08 | 727.97 | 24.17% | 1186.58 |
| Ativas (PRE, critérios 8.3) | KELLY_1.00 | 94.4% | 30.3% | —% | —% | 20078.84 | 445.29 | 14.32% | 1471.39 |

Leitura rápida:
- Se `cap_hit` estiver alto, a alavancagem adicional (ex.: 0,50× ou 1,00×) não vira turnover — o cap está “travando” o tamanho.
- Se `limit_hit` estiver alto, você está batendo no **limit operacional** (capacidade da conta/mercado) — aumentar banca ou frac não aumenta turnover.
- Se `cap_hit` estiver baixo, a curva deve escalar quase linearmente com `frac` (até bater em limites reais/operacionais).
- Lay normalmente satura antes por ter cap mais conservador (liability) e por ter risco de cauda mais assimétrico.

### 9.4c Diagnóstico: Back in‑match (por que não está na estratégia candidata)
Aqui reportamos Back in‑match **apenas como diagnóstico**. Não incluímos in‑match na estratégia candidata porque (i) `closing_odd` pré‑jogo não é benchmark in‑match e (ii) o sizing Kelly acima depende desse benchmark.

| Regime | Scheme | N | Turnover (janela) | Lucro (janela) | ROI/turnover |
|---|---|---:|---:|---:|---:|
| IN_MATCH BackFast (<5s) | FLAT | 129 | 129.00 | 18.59 | 14.41% |
| IN_MATCH BackFast (<5s) | PROXY | 129 | 24800.33 | -2639.58 | -10.64% |

Próximo passo (se quiser operar in‑match): definir um benchmark de preço justo in‑match (ex.: referência por minuto, VWAP, ou odds externas) e calibrar sizing/risco específico do live.

## 12) OOS walk-forward (expanding window): seleção e validação
Até aqui o relatório é **in-sample** (na janela `--lookback-days`). Este bloco (opcional) faz um walk-forward por dia:

- **Train mode**: `expanding`.
- Em cada passo, usamos uma janela de treino para **selecionar** combinações (Side×Pre/In×Reversal) com evidência de valor.
  - `rolling`: usa os **últimos** `wf_train_days`.
  - `expanding`: usa **todos os dias anteriores** (com `wf_train_days` só definindo quando o teste começa).
- No(s) dia(s) seguinte(s) (`wf_test_days`), medimos o resultado OOS nas combinações ativas.

**Evidência de valor (por combinação, no treino)** segue seus critérios (com elegibilidade por volume):
- Elegibilidade: `N_ROI >= wf_min_matches` (jogos com ROI na janela de treino).
- Back/Pre: CLV p90>0 (IC90 lb>0) e ROI>0 (não precisa ser sig.)
- Back/In: ROI p30>0
- Lay/Pre: CLV_CONV p90>0 (IC90 lb>0) e ROI p30>0
- Lay/In: ROI p30>0

Isso aproxima o fluxo operacional que você descreveu (seleciona no passo atual e mede no(s) próximo(s) dia(s)).

### 12.0 Diagnóstico de cobertura OOS (por que N cai)
| Filtro | Jogos únicos |
|---|---:|
| Combinações elegíveis (edge + timing + t0) | 276 |
| Com ROI disponível (precisa de placar) | 247 |
| Com CLV disponível (pre-match + closing) | 158 |

**Calendário do walk-forward (dias únicos)**

| Tipo | Dias |
|---|---:|
| Dias com dados carregados (audited_at) | 5 |
| Dias com eventos OK/betslip conf. | 5 |
| Dias com eventos elegíveis p/ WF (edge) | 5 |
| Dias usados no walk-forward | 5 |

**Diagnóstico por dia (audited_at): betslip vs qualidade vs edge**

| Dia | Auditorias carregadas | Betslip bruto | Betslip conf. | OK (conf.) | Edge Back/Lay | %OK/conf. | Status não-OK dominante |
|---|---:|---:|---:|---:|---:|---:|---|
| 2026-02-13 | 839 | 730 | 485 | 485 | 183/50 | 100.0% | — |
| 2026-02-14 | 1206 | 928 | 530 | 530 | 187/78 | 100.0% | — |
| 2026-02-15 | 855 | 735 | 475 | 475 | 149/55 | 100.0% | — |
| 2026-02-16 | 589 | 422 | 291 | 291 | 118/13 | 100.0% | — |
| 2026-02-19 | 560 | 488 | 486 | 486 | 11/25 | 100.0% | — |

Leitura:
- Se `Auditorias carregadas > 0` mas `Betslip conf.` ≈ 0, geralmente houve **mismatch/parse** (diff fora de [-10,+10]) ou ausência de betslip.
- Se `Betslip conf. > 0` mas `OK (conf.) = 0`, o robô coletou betslip, mas os eventos falharam por **status != OK** (ver coluna de status).
- Dias com `OK (conf.) = 0` **não devem ser tratados como “0 oportunidade”** sem investigar o operacional.


Leitura: o walk-forward mede OOS principalmente por **ROI**, então ele encolhe quando a cobertura de placar é baixa. Além disso, a métrica é agregada por **jogo único** (cluster), então você verá números menores que o N de eventos.

| Train window | Test window | #ativas | Jogos OOS | ROI OOS (mean; IC90) | Turnover (teste) | Lucro (estratégia, budget) |
|---|---|---:|---:|---:|---:|---:|
| 2026-02-13→2026-02-15 | 2026-02-16→2026-02-16 | 2 | 56 | -1.08% [-24.70%, +24.47%] | 1159.86 | -76.15 |
| 2026-02-13→2026-02-16 | 2026-02-19→2026-02-19 | 1 | 2 | +64.80% [+6.27%, +123.10%] | 6.00 | 2.13 |

**Frequência de ativação por combinação (quantas janelas ela entrou como ativa)**

| Combinação | #steps ativa |
|---|---:|
| Back_In_Any | 2 |
| Back_Pre_Any | 1 |

### 12.A Transparência da seleção: métricas por combinação no treino
Para cada janela de treino, mostramos as métricas usadas para decidir se cada combinação ficou **ativa** ou não. Isso ajuda a entender, por exemplo, por que nenhuma Lay entrou em algumas janelas (geralmente N insuficiente ou ROI p30 <= 0).

**Regra de elegibilidade (todas as combinações):** exige `N_ROI >= wf_min_matches` (aqui: 20).

**Train 2026-02-13→2026-02-15 → Test 2026-02-16→2026-02-16**

| Combinação | Ativa? | Jogos treino (tot/CLV/ROI) | CLV q10/q90 ou CI | ROI q30 | Motivo |
|---|---|---:|---:|---:|---|
| Back_Pre_Any | SIM | 133 / 120 / 122 | q10=6.17 | CI90_lb=6.08 | +4.37% | BackPre: clv_q10>0=True, roi_mean>0=True |
| Back_In_Any | SIM | 120 / — / 105 | — | +2.97% | BackIn: roi_q30>0 AND N_ROI>=min (N=105/20) |
| Lay_Pre_Yes | NÃO | 10 / 7 / 10 | q10=-4.24 | CI90_lb=-5.00 | +19.31% | LayPre: clv_conv_q10>0=False, roi_q30>0=False |
| Lay_Pre_No | NÃO | 32 / 26 / 29 | q10=-6.60 | CI90_lb=-6.96 | +11.92% | LayPre: clv_conv_q10>0=False, roi_q30>0=True |
| Lay_In_Yes | NÃO | 25 / — / 19 | — | +6.62% | In: roi_q30>0 AND N_ROI>=min (N=19/20) |
| Lay_In_No | NÃO | 37 / — / 35 | — | -3.74% | In: roi_q30>0 AND N_ROI>=min (N=35/20) |

**Train 2026-02-13→2026-02-16 → Test 2026-02-19→2026-02-19**

| Combinação | Ativa? | Jogos treino (tot/CLV/ROI) | CLV q10/q90 ou CI | ROI q30 | Motivo |
|---|---|---:|---:|---:|---|
| Back_Pre_Any | NÃO | 152 / 139 / 141 | q10=6.32 | CI90_lb=6.22 | -4.58% | BackPre: clv_q10>0=True, roi_mean>0=False |
| Back_In_Any | SIM | 133 / — / 115 | — | +8.36% | BackIn: roi_q30>0 AND N_ROI>=min (N=115/20) |
| Lay_Pre_Yes | NÃO | 11 / 8 / 11 | q10=-4.89 | CI90_lb=-5.40 | +27.79% | LayPre: clv_conv_q10>0=False, roi_q30>0=False |
| Lay_Pre_No | NÃO | 33 / 27 / 30 | q10=-6.71 | CI90_lb=-7.05 | +6.40% | LayPre: clv_conv_q10>0=False, roi_q30>0=True |
| Lay_In_Yes | NÃO | 26 / — / 19 | — | +6.62% | In: roi_q30>0 AND N_ROI>=min (N=19/20) |
| Lay_In_No | NÃO | 39 / — / 36 | — | -6.84% | In: roi_q30>0 AND N_ROI>=min (N=36/20) |


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

**Padrão de risco**: P&L aqui já é calculado com **budget por jogo (match_id)** consumido ao longo do tempo (Back=1.00% da banca ref; Lay=0.50% em liability; cap por sinal=33% do budget).

| Premissa | Valor |
|---|---:|
| Train mode (OOS) | `expanding` |
| Scheme pre-match (OOS) | `KELLY_0.25` |
| Scheme in-match (OOS) | `FLAT` |
| Expansão missing ROI | ON |
| Dias OOS (calendário de teste) | 2 |
| Dias OOS com OK (>=1 evento OK/conf) | 2 |
| Turnover 30d (proj., calendário) | 17487.89 |
| Turnover 30d (proj., cond OK) | 17487.89 |
| Turnover 30d (Pre/In) | 66242.79 / 375.00 |
| Lucro 30d (obs., calendário) | -1101.75 |
| Lucro 30d (obs., cond OK) | -1101.75 |
| Lucro 30d (obs.) Pre/In | 1979.07 / 144.71 |
| Lucro 30d (exp., calendário) | -1110.31 |
| Lucro 30d (exp., cond OK) | -1110.31 |
| Lucro 30d (exp.) Pre/In | 1987.97 / 199.43 |
| Banca risco p99 (Back+Lay) | 1148.32 |
| Banca liquidez p99 (+buf) | 1215.85 |
| Banca recomendada (max) | 1215.85 |
| ROI/banca 30d (obs., calendário) | -90.62% |
| ROI/banca 30d (obs., cond OK) | -90.62% |
| ROI/banca 30d (exp., calendário) | -91.32% |
| ROI/banca 30d (exp., cond OK) | -91.32% |
| DD 30d p95 (obs., calendário) | 1483.19 |
| DD 30d p95 (obs., cond OK) | 1483.19 |
| DD 30d p95 (exp., calendário) | 1501.70 |
| DD 30d p95 (exp., cond OK) | 1429.81 |

### 12.2 Governança de exposição por jogo (budget por `match_id`) — sensibilidade
Objetivo: evitar concentração quando um mesmo jogo gera muitos sinais. Simulamos um orçamento de exposição por jogo, consumido ao longo do tempo.

- **Back** consome budget em **stake**.
- **Lay** consome budget em **liability**.
- Também aplicamos um cap por sinal como fração do budget do jogo (para não gastar tudo no 1º sinal).

Importante: o budget é parametrizado como fração de uma referência de banca. Aqui usamos:
- se `--kelly-bankroll` estiver setado: essa banca explícita;
- senão: a `Banca recomendada (max)` estimada na 12.1 (baseline).

Referência de banca p/ budget: 1215.85 | budgets por jogo aplicados em stake (Back) e liability (Lay).

| Cenário | Turnover 30d | Lucro 30d (exp.) | Banca rec. (max) | ROI/banca 30d (exp.) | DD 30d p95 (exp.) |
|---|---:|---:|---:|---:|---:|
| BASELINE (sem budget) | 17487.89 | -1110.31 | 1215.85 | -91.32% | 1501.70 |
| BUDGET_0.50%/0.25% cap25% | 2289.96 | -43.40 | 145.26 | -29.87% | 79.15 |
| BUDGET_1.00%/0.50% cap33% | 5197.06 | -252.59 | 337.13 | -74.92% | 343.36 |
| BUDGET_2.00%/1.00% cap50% | 12641.97 | -648.88 | 828.49 | -78.32% | 845.33 |
| BUDGET_3.00%/1.50% cap33% | 14674.51 | -973.72 | 962.64 | -101.15% | 1319.58 |
| BUDGET_4.00%/2.00% cap33% | 19237.00 | -1177.06 | 1263.76 | -93.14% | 1514.46 |
| BUDGET_3.00%/1.50% cap50% | 18506.15 | -801.47 | 1215.53 | -65.94% | 1036.59 |
| BUDGET_4.00%/2.00% cap50% | 23836.61 | -791.84 | 1567.34 | -50.52% | 1026.41 |

Leitura:
- Se a curva com budget melhora muito (menos negativo ou mais positivo) com pouca perda de turnover, o problema era **concentração por jogo**.
- Se tudo continuar negativo, o problema é **edge OOS** (principalmente in‑match) e budget só reduz a escala da perda.

**Volume e stake médio por combinação (janela OOS, com budget padrão)**

| Combinação | #steps ativa | Jogos OOS (uniques) | N eventos OOS | Stake eq. médio | Observação |
|---|---:|---:|---:|---:|---|
| Back_In_Any | 2 | 18 | 25 | 1.00 | budget reduz concentração por jogo |
| Back_Pre_Any | 1 | 43 | 87 | 50.76 | budget reduz concentração por jogo |
## 10) Diagnóstico: por que o ROI pode estar zerado
ROI aqui é calculado por placar do jogo (`matches.home_score/away_score`). Se os placares não estiverem preenchidos no banco, a cobertura de ROI será 0.

| Indicador | Valor |
|---|---:|
| Jogos únicos no recorte | 420 |
| Jogos com placar disponível (home_score/away_score não nulos) | 376 |
| Jogos com status='finished' no banco | 376 |

### 10.1 Distribuição por data de kickoff (explica janela da API)
- Kickoff (UTC) no recorte: **2026-02-13 10:45 UTC** até **2026-02-22 11:30 UTC**.

| Kickoff date (UTC) | Jogos | Com placar | Cobertura |
|---|---:|---:|---:|
| 2026-02-22 | 2 | 1 | 50.0% |
| 2026-02-21 | 70 | 69 | 98.6% |
| 2026-02-20 | 15 | 14 | 93.3% |
| 2026-02-19 | 27 | 24 | 88.9% |
| 2026-02-18 | 11 | 9 | 81.8% |
| 2026-02-17 | 39 | 39 | 100.0% |
| 2026-02-16 | 24 | 19 | 79.2% |
| 2026-02-15 | 74 | 69 | 93.2% |
| 2026-02-14 | 111 | 102 | 91.9% |
| 2026-02-13 | 47 | 30 | 63.8% |

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
