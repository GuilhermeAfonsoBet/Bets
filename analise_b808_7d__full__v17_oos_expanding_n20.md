# Análise Estatística Robusta — Contexto Operação (b808)
**Data da execução:** 18/02/2026 22:54 UTC  
**Escopo:** H3B (auditoria), com comparação por `audit_version` e inferência robusta por jogo (cluster bootstrap).  
**Nota:** a robustez aqui significa que intervalos de confiança consideram correlação intra-jogo (múltiplas auditorias por partida).
---
## 0) Sumário executivo (leitura rápida)
- **Recorte**: direction=`up`, lookback_days=`7`, versions=`v4.0-api,v1.0,v1.0-recovered`.
- **Amostra**: 4032 auditorias (jogos únicos=329, média=12.3 obs/jogo); betslip confiável=1810.
- **Janela efetiva (audited_at)**: 12/02 18:29 → 18/02 21:55 UTC (span≈6.1d; dias com dados=7).
- **Coortes (status=OK, betslip confiável)**: `BS>WS` (diff>=2.0%): **641**; `BS<WS` (diff<=-2.0%): **206**.
- **Coberturas em `hypothesis_details` (OK)**: temporal(back)=1502/1810; lay_temporal=1359/1810; finance=1269/1810.
- **Cobertura de placar (ROI)**: jogos com placar=288/329 (status finished=288).
- **Cobertura de closing_odd (AH)**: jogos com closing=198/329 (60.2%). CLV pre‑match depende disso.
- **Alerta**: cobertura de closing_odd está baixa. Isso tende a reduzir N de CLV e pode enviesar a leitura (os jogos “sem odds” ficam fora do CLV). Priorize estabilizar o collector e/ou condicionar análises a jogos com closing.
- **DOM**: sem dados no recorte atual (N=0), então não há comparação API vs DOM aqui.
- **CLV pre-match (Betslip, API)**: média robusta por jogo +2.115% (IC90 [+1.650%, +2.580%]), com N=939 eventos (jogos=159).
- **Padrão por bucket (CLV PM)**: `BS < WS` -3.920% (sig. negativo), `BS ~ WS` -0.132% (NS), `BS > WS` +6.532% (sig. positivo).
- **Leitura recomendada**: use este relatório para validar **qualidade de execução/CLV**. Para concluir sobre **ROI**, rode após atualizar resultados e/ou use uma janela com jogos já liquidados.

---
## 1) Contexto do corte (b808)
| Indicador | Valor |
|---|---:|
| Auditorias H3B `UP` (match + kickoff passado) | 4032 |
| Betslip bruto | 2868 |
| Betslip confiável (diff -10% a +10%) | 1810 |
| Descartados no filtro de qualidade | 1058 |
| Jogos únicos (geral) | 329 |
| Média de observações por jogo | 12.3 |
| Jogos únicos com betslip confiável | 298 |
| Distribuição por market_type | AH=4032 |
| Jogos únicos (AH) no recorte | 329 |
| Jogos únicos (AH) com closing_odd disponível | 198 |
| Cobertura closing_odd (AH) | 60.2% |

---
## 2) Base comparativa: API vs DOM
| Métrica | API (v4.0-api) | DOM (v1.0) |
|---|---:|---:|
| Total de observações | 4032 | 0 |
| Com betslip confiável | 1810 | 0 |
| Com CLV pre-match (betslip) | 939 | 0 |
| Com ROI (betslip) | 1620 | 0 |
| Tempo total observado (detecção→betslip, wall/total) | 12296 ms | — ms |
| Tempo instrumentado (detecção→clique→betslip) | 8821 ms | — ms |

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
| API (2-4s) | lag_det→click | 5559 | 796 | 7152 | 4030 |
| API (2-4s) | lag_click→betslip | 2627 | 2170 | 4509 | 3408 |
| API (2-4s) | lag_e2e (soma) | 8821 | 3462 | 11182 | 3406 |
| API (2-4s) | audit_total (duração) | 12291 | 4506 | 31133 | 4032 |
| API (2-4s) | overhead (total - e2e) | 4785 | 204 | 22806 | 3406 |
| DOM (15-30s) | lag_det→click | — | — | — | 0 |
| DOM (15-30s) | lag_click→betslip | — | — | — | 0 |
| DOM (15-30s) | lag_e2e (soma) | — | — | — | 0 |
| DOM (15-30s) | audit_total (duração) | — | — | — | 0 |
| DOM (15-30s) | overhead (total - e2e) | — | — | — | 0 |

**Diagnóstico de cauda (percentual acima do limiar)**

| Modelo | % det→click > 5s | % det→click > 20s | % total > 10s | % total > 40s | % overhead < 0 |
|---|---:|---:|---:|---:|---:|
| API (2-4s) | 5.7% | 3.3% | 12.8% | 3.5% | 0.0% |
| DOM (15-30s) | —% | —% | —% | —% | —% |

---
### 2.1 Cobertura temporal (pre-match vs in-match)
| Métrica | Pre-match | In-match | Observação |
|---|---:|---:|---|
| Observações totais com classificação temporal | 1859 | 2173 | Contagem bruta do corte |
| ROI Betslip | 978 | 642 | Amostra com resultado do jogo |
| ROI WebSocket | 1742 | 1896 | Referência de mercado |
| CLV (apenas pre-match) | 939 | — | CLV vs closing pré-jogo não é interpretável in-match |

---
### 2.2 Performance por regime (pre-match vs in-match)
| Regime | N auditorias | N betslip conf. | N OK | Back edge | Lay edge | Diff médio (OK) |
|---|---:|---:|---:|---:|---:|---:|
| PRE_MATCH | 1859 | 1048 | 1048 | 415 | 69 | +2.230% |
| IN_MATCH | 2173 | 762 | 762 | 226 | 137 | +0.926% |

---
### 2.3 Regimes operacionais por tempo total (bucket)
Objetivo: separar a amostra em **regimes de execução** (tempo total) e medir performance em cada regime. Use isso para detectar **fila/saturação**: regimes lentos tendem a ter edge menor e pior ROI.

| Bucket (tempo total) | N OK | Jogos | lag_total mean (ms) | lag_total p95 (ms) | overhead p95 (ms) | Back edge | Lay edge | CLV PM (mean; IC90) | ROI (mean; IC90) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| < 5s | 1287 | 270 | 3597 | 4851 | 2506 | 433 | 132 | +2.34% [+1.92%, +2.76%] | +3.88% [-1.99%, +9.90%] |
| 5-10s | 234 | 148 | 5887 | 8212 | 2824 | 98 | 45 | +1.80% [+0.88%, +2.73%] | -4.09% [-15.96%, +8.08%] |
| 10-20s | 18 | 17 | 13668 | 18117 | 4070 | 9 | 2 | +1.17% [-1.58%, +3.89%] | +10.43% [-23.26%, +43.02%] |
| 20-40s | 195 | 101 | 27127 | 33419 | 29826 | 67 | 20 | +2.36% [+1.39%, +3.33%] | +3.97% [-9.31%, +17.80%] |
| > 40s | 76 | 55 | 187614 | 507883 | 389950 | 34 | 7 | +2.54% [+1.37%, +3.68%] | -9.62% [-28.92%, +10.07%] |
| Desconhecido | 0 | 0 | — | — | — | 0 | 0 | — | — |

---
### 2.3b Regimes por tempo total — separando coortes por `diff_pct` (BS vs WS)
Nesta tabela, separamos duas coortes operacionais por **delta de execução**: `BS > WS` (diff_pct >= +2%) e `BS < WS` (diff_pct <= -2%). Isso **não** é (por si só) “Back vs Lay”; é um recorte por **melhora/piora do preço** entre detecção (WS) e execução (BS). CLV é reportado apenas em pre‑match.

| Bucket | N OK | N (BS>WS +2%) | N (BS<WS -2%) | CLV PM (BS>WS) | CLV PM (BS<WS) | ROI (BS>WS) | ROI (BS<WS) |
|---|---:|---:|---:|---:|---:|---:|---:|
| < 5s | 1287 | 433 | 132 | +6.92% [+6.48%, +7.34%] | -3.72% [-4.89%, -2.53%] | +2.47% [-7.35%, +12.54%] | +4.18% [-10.67%, +19.15%] |
| 5-10s | 234 | 98 | 45 | +5.55% [+4.71%, +6.34%] | -5.91% [-7.17%, -4.54%] | -8.14% [-25.87%, +9.39%] | -11.12% [-31.39%, +9.19%] |
| 10-20s | 18 | 9 | 2 | +3.86% [+1.34%, +6.62%] | -6.98% — | -10.16% [-55.47%, +36.27%] | +40.04% [+0.00%, +79.90%] |
| 20-40s | 195 | 67 | 20 | +7.43% [+6.38%, +8.48%] | -1.06% [-3.91%, +2.13%] | +0.10% [-19.89%, +20.68%] | -27.75% [-64.76%, +10.82%] |
| > 40s | 76 | 34 | 7 | +5.38% [+3.68%, +7.04%] | -4.80% [-7.29%, -2.29%] | -26.12% [-53.99%, +2.81%] | -22.27% [-80.00%, +35.02%] |
| Desconhecido | 0 | 0 | 0 | — | — | — | — |

---
### 2.3c Estabilidade temporal (por dia, `audited_at`)
Objetivo: checar se o regime de edge/execução é **time‑dependent**. Se houver dias com comportamento distinto (ex.: collector intermitente, horários/ligas), isso aparecerá aqui.

| Dia (UTC) | Modelo | N OK | Jogos | % BS>WS +2% | % BS<WS -2% | lag_total p50 (ms) | CLV PM (BS>WS) | CLV PM (BS<WS) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-02-12 | API (2-4s) | 55 | 44 | 36.4% | 9.1% | 26856 | +5.58% | -4.34% |
| 2026-02-13 | API (2-4s) | 579 | 141 | 36.1% | 10.4% | 4823 | +6.96% | -3.22% |
| 2026-02-14 | API (2-4s) | 506 | 140 | 35.0% | 15.2% | 4207 | +6.25% | -5.29% |
| 2026-02-15 | API (2-4s) | 444 | 102 | 32.0% | 11.5% | 3691 | +6.40% | -2.66% |
| 2026-02-16 | API (2-4s) | 226 | 66 | 41.2% | 5.8% | 3445 | +6.43% | -6.79% |

---
## 3) CLV pre-match (núcleo)
### 3.1 CLV com odd do Betslip (execução real)
| Métrica | API | DOM |
|---|---:|---:|
| CLV Bruto BS Pre-Match | +2.334% (sig. positivo, N=939, jogos=159) | — (N/A, N=0, jogos=0) |
| CLV Adicional BS Pre-Match | +1.761% (sig. positivo, N=939, jogos=159) | — (N/A, N=0, jogos=0) |
| Taxa de CLV > 0 (bruto) | 61.1% | —% |
| Taxa de CLV > 0 (adicional) | 61.9% | —% |

Notas de robustez (IC 90% por jogo):  
- API CLV bruto (cluster): média +2.115%; IC90 [+1.650%, +2.580%]  
- DOM CLV bruto (cluster): média —; IC90 —  

---
## 4) ROI por modelo
| Métrica | API | DOM |
|---|---:|---:|
| ROI Betslip | +0.622% (NS, N=1618) | — (N/A, N=0) |
| ROI WebSocket | -1.271% (sig. negativo, N=3623) | — (N/A, N=0) |
| Win rate ROI Betslip | 50.3% | —% |
| Win rate ROI WS | 49.7% | —% |

Notas de robustez (IC 90% por jogo):  
- API ROI betslip (cluster): média +0.737%; IC90 [-4.373%, +5.710%]  
- API ROI WS (cluster): média -3.178%; IC90 [-6.065%, -0.270%]  

---
## 5) Diferença de preço BS vs WS
| Métrica | API | DOM |
|---|---:|---:|
| Diff BS vs WS (média) | +1.681% (sig. positivo, N=1810) | — (N/A, N=0) |
| BS > WS | 47.7% (863/1810) | —% (0/0) |
| BS > WS +2% | 35.4% (641/1810) | —% (0/0) |

---
## 6) Combinações de valor
### 6.1 Buckets por diferença BS vs WS
Nota de leitura:
- `BS > WS (+2% a +10%)`: preço no betslip ficou **melhor** do que o WS observado na detecção (delta positivo entre instantes).
- `BS < WS (-10% a -2%)`: preço no betslip ficou **pior** do que o WS observado na detecção (delta negativo entre instantes).
- O ROI abaixo é calculado **dentro do bucket** (não mistura buckets), mas ainda é sensível à cobertura de placar.

| Bucket | N bucket | CLV BS PM (média) | IC90 (cluster) | N CLV PM | Jogos CLV PM | ROI BS (todos) | IC90 (cluster) |
|---|---:|---:|---|---:|---:|---:|---|
| BS < WS (-10% a -2%) | 206 | -3.920% | [-4.884%, -3.048%] | 59 | 46 | +1.271% | [-15.212%, +8.004%] |
| BS ~ WS (-2% a +2%) | 963 | -0.132% | [-0.529%, +0.181%] | 499 | 144 | +0.507% | [-9.886%, +3.090%] |
| BS > WS (+2% a +10%) | 641 | +6.532% | [+6.382%, +7.155%] | 381 | 119 | +0.589% | [-4.780%, +12.952%] |

---
### 6.2 Combinação por faixa de linha AH
| Faixa AH | CLV BS PM (média) | IC90 (cluster) | ROI BS (todos) | IC90 (cluster) | Diff BS vs WS (média) |
|---|---:|---|---:|---|---:|
| AH 0-1 (líquida) | +2.567% | [+2.118%, +3.441%] | +1.146% | [-0.226%, +14.728%] | +1.886% |
| AH 1-2 (média) | +2.455% | [+1.264%, +3.429%] | +0.897% | [-13.153%, +12.694%] | +2.381% |
| AH 2+ (extrema) | +1.850% | [+0.336%, +1.970%] | +0.049% | [-8.663%, +7.701%] | +1.222% |

---
### 6.3 Combinação por faixa de lag
| Faixa de lag | CLV BS PM (média) | IC90 (cluster) | N CLV PM | Jogos CLV PM | ROI BS (todos) | IC90 (cluster) | Diff BS vs WS (média) |
|---|---:|---|---:|---:|---:|---|---:|
| < 10s | +2.381% | [+1.714%, +2.566%] | 738 | 151 | +0.940% | [-3.175%, +8.298%] | +1.649% |
| 10-20s | +1.220% | [-1.579%, +3.893%] | 7 | 7 | +9.839% | [-23.261%, +43.022%] | +1.793% |
| 20-30s | +2.230% | [+1.406%, +3.324%] | 123 | 73 | +2.299% | [-9.360%, +17.351%] | +1.803% |
| > 30s | +2.131% | [+1.404%, +3.539%] | 71 | 48 | -9.163% | [-26.722%, +7.038%] | +1.930% |

---
## 7) Estimativa financeira (proxy) e risco
Este bloco usa `hypothesis_details.finance` quando existe; se não existir, usa stake fallback = `stake_pct_of_limit × limit`.

**Política de stake (proxy)**

| Parâmetro | Valor |
|---|---:|
| stake_pct_of_limit | 0.25 |
| stake_cap | 0.00 |
| Cobertura finance (OK, betslip conf.) | 1269/1810 |

### 7.1 Back (BS >> WS)
| Métrica | Valor |
|---|---:|
| Corte (diff_pct) | >= 2.0% |
| N eventos | 641 |
| Cobertura finance (na coorte) | 446/641 |
| Stake total (estimado) | 201226.53 |
| Stake médio | 313.93 |
| Profit_if_win total (estimado) | 215902.02 |
| Profit_if_win médio | 336.82 |
| N com ROI realizado | 579 |
| P&L realizado total (estimado) | -35789.57 |
| ROI realizado (ponderado por stake) | -18.31% |
| ROI realizado (robusto por jogo, mean; IC90) | +3.99% [-4.78%, +12.95%] |
| ROI ponderado por stake (robusto por jogo, mean; IC90) | +3.91% [-6.74%, +15.29%] |

**Como ler as 3 linhas de ROI (Back)**

- **ROI realizado (ponderado por stake)**: ΣP&L / Σstake (pode ser dominado por stakes grandes).
- **ROI realizado (robusto por jogo)**: média por jogo (cada jogo pesa parecido; reduz dominância de outliers).
- **ROI ponderado por stake (robusto por jogo)**: calcula ROI ponderado dentro do jogo e depois faz média/IC por jogo.

Observação: a Seção 4 reporta ROI médio no **recorte completo** (betslip confiável). Aqui (7.1) é apenas a coorte **Back edge** (diff>=corte) e o ROI também é mostrado ponderado por stake; por isso sinais podem divergir.

### 7.2 Lay (BS << WS) — risco de cauda
| Métrica | Valor |
|---|---:|
| Corte (diff_pct) | <= -2.0% |
| N eventos | 206 |
| Cobertura finance (na coorte) | 148/206 |
| Stake total (estimado) | 34400.01 |
| Liability total (estimada) | 30301.95 |
| Liability média | 147.10 |
| Liability p95 | 530.59 |
| Liability p99 | 2045.09 |
| ES95 (liability) | 1568.72 |
| Liability max | 4395.35 |
| Proxy de banca (>= p99 liability) | 2045.09 |
| N com ROI realizado | 128 |
| P&L realizado total (estimado) | -7751.03 |
| ROI realizado (ponderado por liability) | -27.33% |
| ROI realizado (ponderado por stake) | -23.99% |
| ROI/liability (robusto por jogo, mean; IC90) | +18.28% [+1.95%, +33.95%] |
| ROI/liability ponderado (robusto por jogo, mean; IC90) | +16.49% [-0.14%, +32.98%] |

### 7.3 Projeção mensal (30 dias fixo) — turnover, lucro e banca
Premissas:
- **Mês = 30 dias fixo**.
- Turnover = soma de stakes (Back e Lay). Para Lay, também reportamos exposição por **liability**.
- **Banca por risco (unitária)** = p99/ES95(exposição por aposta).
- **Banca por liquidez (turnover)** = p95/p99 de capital **simultaneamente travado** (stake/liability em aberto até a liquidação), capturando distribuição desigual de entradas ao longo do tempo.
- Projeção de lucro usa duas visões: (i) **lucro/dia observado** e (ii) **ROI observado × turnover projetado**.

| Bloco | Janela(d) | Turnover 30d | Lucro 30d (direto) | Lucro 30d (ROI×turnover) |
|---|---:|---:|---:|---:|
| Back | 7.0 | 862399.41 | -153383.87 | -157918.60 |
| Lay (stake) | 7.0 | 147428.62 | -33218.70 | -35374.15 |
| Total (Back+Lay) | 7.0 | 1009828.03 | -186602.57 | -193292.76 |

**Banca por risco (exposição unitária)**

| Bloco | Banca conservadora (p99) | Banca agressiva (ES95) | ROI/banca 30d (direto) | ROI/banca 30d (ROI×turnover) |
|---|---:|---:|---:|---:|
| Back (stake) | 4394.24 | 3831.80 | -3490.57% | -3593.77% |
| Lay (liability) | 2045.09 | 1568.72 | -1624.32% | -1729.72% |
| Total (soma) | 6439.32 | 5400.52 | -2897.86% | -3001.76% |

**Banca por liquidez (capital simultaneamente travado)**

Definição operacional: cada aposta trava capital de `audited_at` até `kickoff + 2.00h + 2.25h` (grid=5min). A banca recomendada aplica buffer de +10%.

| Bloco | Liquidez mean | Liquidez p95 | Liquidez p99 | Liquidez max | Banca liq p99 (+buffer) |
|---|---:|---:|---:|---:|---:|
| Back (stake) | 20278.34 | 58516.19 | 85387.92 | 88455.79 | 93926.71 |
| Lay (liability) | 2625.43 | 7385.66 | 9708.84 | 11832.35 | 10679.73 |
| Total (Back+Lay) | 22890.91 | 63251.04 | 91619.40 | 95803.56 | 100781.34 |

**Banca recomendada (conservadora)**

| Métrica | Valor |
|---|---:|
| Banca por risco (p99 unitário, soma) | 6439.32 |
| Banca por liquidez (p99 simultâneo + buffer) | 100781.34 |
| Banca efetiva (max das duas) | 100781.34 |
| ROI/banca 30d (direto, banca efetiva) | -185.16% |
| ROI/banca 30d (ROI×turnover, banca efetiva) | -191.79% |
**Diagnóstico de cobertura (placar/ROI)**

| Bloco | Turnover total | Turnover com ROI | Cobertura turnover |
|---|---:|---:|---:|
| Back | 201226.53 | 195448.19 | 97.13% |
| Lay | 34400.01 | 32303.91 | 93.91% |

Notas (Lay): exposição 30d por liability (não é turnover) = 129865.51; ROI realizado por liability (ponderado) = -27.33%.

---
## 8) Curva temporal (pico, reversão e melhor timing)
Esta seção usa `hypothesis_details.temporal` (Back) e `hypothesis_details.lay_temporal` (Lay), coletados em pontos discretos (t≈0,3,6,10,15,20s).

O objetivo é responder: **tempo até o pico/vale**, **% que segue melhorando até t_max**, **% com reversão** e **impacto esperado por timing**. **CLV é reportado somente pre-match** (closing pré-jogo).

### 8.1 Back (pico em diff_pct)
Definições: **pico** = `max(diff_pct)`. **reversão** = após o pico, `diff_pct` cair pelo menos 0.50 p.p. (para Lay, subir 0.50 p.p. após o vale). `t_reversão` é o 1º tempo que cruza esse limiar (logo, não é o pico).

| Regime | N | t_pico médio (s) | t_pico p50 (s) | % melhora até fim | % com reversão | t_reversão médio (s) | Δt (rev - pico) médio (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRE_MATCH | 1048 | 4.5 | 5.2 | 76.2% | 17.0% | 12.2 | 7.1 |
| IN_MATCH | 762 | 5.5 | 0.0 | 59.3% | 31.9% | 13.5 | 8.1 |

**Partição 100% (Back)**: categorias exclusivas (somam 100% por regime).

| Regime | % pico no fim, sem reversão | % pico no fim, com reversão (recuperou) | % pico antes, com reversão | % pico antes, sem reversão relevante |
|---|---:|---:|---:|---:|
| PRE_MATCH | 80.0% | 7.4% | 9.5% | 3.1% |
| IN_MATCH | 65.4% | 5.0% | 26.9% | 2.8% |

**Curva média (Back)**: média de `diff_pct` e `odd` por tempo. `CLV` é reportado **somente pre-match** (closing pré-jogo). `ROI` (quando aparece) é o **ROI realizado** se a aposta fosse feita naquele ponto.

| Tempo | N pts | diff_pct médio | odd média | CLV médio (pre-match) | ROI médio (se houver) |
|---|---:|---:|---:|---:|---:|
| t+0s | 1810 | +1.68% | 2.016 | +2.33% | 1.35 |
| t+6s | 1474 | +2.28% | 2.024 | +2.64% | 0.59 |
| t+10s | 2395 | +2.61% | 2.029 | +2.69% | 1.74 |
| t+15s | 1488 | +2.75% | 2.047 | +2.78% | 1.61 |
| t+20s | 2031 | +3.25% | 2.043 | +2.76% | 0.23 |

### 8.1b Back — impacto por timing (t0 vs pico vs último) e reversão
Leitura: se a estratégia é **entrar no pico**, compare CLV/ROI no pico vs t0 e veja diferença entre **casos com reversão** vs **sem reversão**. Aqui reportamos **média robusta por jogo + IC90**. Observação: **CLV só é calculado pre-match**.

**CLV (somente pre-match) — média robusta por jogo (IC90)**

| Subcoorte | N total | N CLV (PM) | CLV t0 (mean; IC90) | CLV pico (mean; IC90) | CLV último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 1389 | 782 | +1.76% [+1.33%, +2.19%] | +2.03% [+1.60%, +2.45%] | +2.02% [+1.59%, +2.44%] |
| COM_REVERSAO | 421 | 157 | +4.03% [+3.22%, +4.85%] | +5.48% [+4.57%, +6.39%] | +4.62% [+3.68%, +5.55%] |

**ROI (stake=1) — média robusta por jogo (IC90)**

| Subcoorte | N total | N ROI | ROI t0 (mean; IC90) | ROI pico (mean; IC90) | ROI último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 1389 | 1251 | +0.49% [-5.58%, +6.69%] | +0.66% [-5.47%, +6.92%] | +0.64% [-5.48%, +6.90%] |
| COM_REVERSAO | 421 | 367 | +0.28% [-9.89%, +10.88%] | +3.27% [-7.26%, +14.26%] | +0.42% [-9.86%, +10.94%] |

Nota interpretativa: a subcoorte **COM_REVERSAO** pode ter CLV maior no **pico** porque, por definição, ela inclui casos em que o edge atingiu um extremo mais alto antes de reverter. Para entender a hipótese 'sem reversão deveria ser maior', compare também a categoria '**pico no fim, sem reversão**' na partição 100% (tabela 8.1).

**Decomposição (pre-match): entry_odd vs closing_odd (IC90)**

| Subcoorte | N CLV (PM) | entry_odd t0 (mean; IC90) | entry_odd pico (mean; IC90) | closing_odd (mean; IC90) |
|---|---:|---:|---:|---:|
| SEM_REVERSAO | 782 | 1.982 [+1.971, +1.993] | 1.989 [+1.978, +2.001] | 1.962 [+1.953, +1.971] |
| COM_REVERSAO | 157 | 2.035 [+2.017, +2.054] | 2.064 [+2.044, +2.085] | 1.961 [+1.948, +1.974] |

### 8.2 Lay (vale em diff_pct)
| Regime | N | t_vale médio (s) | t_vale p50 (s) | % melhora até fim | % com reversão | t_reversão médio (s) | Δt (rev - vale) médio (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRE_MATCH | 928 | 4.3 | 5.2 | 69.7% | 21.4% | 12.6 | 7.0 |
| IN_MATCH | 606 | 6.2 | 5.3 | 45.7% | 43.4% | 13.5 | 7.9 |

**Partição 100% (Lay)**: categorias exclusivas (somam 100% por regime).

| Regime | % vale no fim, sem reversão | % vale no fim, com reversão (recuperou) | % vale antes, com reversão | % vale antes, sem reversão relevante |
|---|---:|---:|---:|---:|
| PRE_MATCH | 73.3% | 8.0% | 13.5% | 5.3% |
| IN_MATCH | 52.6% | 6.4% | 37.0% | 4.0% |

**Curva média (Lay)**: usa `odd=lay_odd`. `CLV` é reportado **somente pre-match** (closing pré-jogo). O ROI mostrado aqui é **ROI por liability** (não por stake), porque Lay é governado por risco.

| Tempo | N pts | diff_pct médio | lay_odd média | CLV médio (pre-match) | ROI/liability médio (se houver) |
|---|---:|---:|---:|---:|---:|
| t+0s | 1534 | +0.67% | 2.000 | +0.05% | 11.23 |
| t+6s | 1335 | +0.84% | 2.002 | +0.18% | 27.30 |
| t+10s | 2156 | +0.31% | 1.990 | +0.29% | 26.94 |
| t+15s | 1346 | +1.23% | 2.028 | +0.04% | 16.73 |
| t+20s | 1859 | +1.57% | 2.018 | +0.17% | 13.04 |

### 8.2b Lay — impacto por timing (t0 vs vale vs último) e reversão
Leitura: se a estratégia é **entrar no vale (odd mais baixa)**, compare CLV/ROI no vale vs t0 e veja diferença entre **casos com reversão** vs **sem reversão**. Aqui reportamos **média robusta por jogo + IC90**. Observação: **CLV só é calculado pre-match**.

**CLV (somente pre-match) — média robusta por jogo (IC90)**

| Subcoorte | N total | N CLV (PM) | CLV t0 (mean; IC90) | CLV vale (mean; IC90) | CLV último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 1072 | 648 | -0.41% [-1.02%, +0.20%] | -0.07% [-0.68%, +0.52%] | -0.09% [-0.70%, +0.50%] |
| COM_REVERSAO | 462 | 173 | +0.15% [-0.76%, +1.07%] | +1.46% [+0.53%, +2.37%] | +0.26% [-0.65%, +1.18%] |

**ROI/liability — média robusta por jogo (IC90)**

| Subcoorte | N total | N ROI | ROI/liab t0 (mean; IC90) | ROI/liab vale (mean; IC90) | ROI/liab último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 1072 | 969 | +6.77% [-1.77%, +15.46%] | +7.49% [-1.07%, +16.26%] | +7.44% [-1.10%, +16.21%] |
| COM_REVERSAO | 462 | 406 | +17.45% [+4.13%, +32.36%] | +25.47% [+7.58%, +46.96%] | +13.59% [+2.20%, +25.65%] |

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
| Back | Pre | Any | 1048 | 190 | +2.12% [+1.65%, +2.58%] | -4.97% [-11.57%, +1.62%] | -7.04% | não (CLV p90>0 AND ROI>0) |
| Back | In | Any | 762 | 217 | — | +5.04% [-2.01%, +12.01%] | +2.67% | sim (ROI p30>0) |
| Lay | Pre | Yes | 199 | 105 | +0.74% [-0.18%, +1.64%] | +2.46% [-10.80%, +15.67%] | -2.20% | não (CLV_CONV p90>0 AND ROI p30>0) |
| Lay | Pre | No | 729 | 173 | +0.09% [-0.50%, +0.70%] | +2.35% [-5.42%, +10.40%] | -0.22% | não (CLV_CONV p90>0 AND ROI p30>0) |
| Lay | In | Yes | 263 | 138 | — | +24.72% [+6.27%, +45.25%] | +18.20% | sim (ROI p30>0) |
| Lay | In | No | 343 | 164 | — | +10.22% [-4.32%, +25.20%] | +5.23% | sim (ROI p30>0) |

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
| IN_MATCH | AH 2+ (extrema) | 20-30s | 4 | 4 | -4.40% | [-5.69%, -3.05%] | 268.18 |
| PRE_MATCH | AH 2+ (extrema) | 20-30s | 3 | 3 | -3.90% | [-5.14%, -2.65%] | 79.09 |
| PRE_MATCH | AH 0-1 (líquida) | > 30s | 3 | 3 | -3.39% | [-3.56%, -3.21%] | 132.82 |
| IN_MATCH | AH 0-1 (líquida) | 20-30s | 3 | 3 | -3.18% | [-3.50%, -2.85%] | 175.10 |
| IN_MATCH | AH 0-1 (líquida) | > 30s | 3 | 3 | -6.34% | [-8.04%, -4.63%] | 214.41 |

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

- **Back (stake)**: corr(exposição, ROI)=-0.070; corr(exposição, CLV)=0.009 (onde CLV existe).
- **Lay (liability)**: corr(exposição, ROI)=0.075; corr(exposição, CLV)=0.063 (onde CLV existe).

**Backtest de sizing (apenas eventos com placar; valores em unidade monetária *proxy*)**

| Lado | Scheme | N | Turnover (janela) | Lucro (janela) | ROI/turnover | p99 exposição | ES95 exposição | DD 30d (média) | DD 30d (p95) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Back | FLAT | 579 | 579.00 | 9.90 | 1.71% | 1.00 | 1.00 | 23.24 | 43.87 |
| Lay | FLAT | 181 | 221.59 | 0.20 | 0.09% | 1.00 | 1.00 | 23.57 | 43.52 |
| Back | PROXY | 579 | 195448.19 | -35789.57 | -18.31% | 4395.55 | 4129.56 | 154464.19 | 227927.34 |
| Lay | PROXY | 128 | 32303.91 | -7751.03 | -23.99% | 3766.29 | 2142.82 | 38855.01 | 53155.74 |
| Back | KELLY_0.10 | 339 | 8707.89 | 281.24 | 3.23% | 56.97 | 51.27 | 694.56 | 1257.26 |
| Lay | KELLY_0.10 | 48 | 633.38 | 94.42 | 14.91% | 20.45 | 20.45 | 33.64 | 58.25 |
| Back | KELLY_0.25 | 339 | 17834.88 | 269.41 | 1.51% | 87.88 | 87.88 | 1642.82 | 3117.59 |
| Lay | KELLY_0.25 | 48 | 925.33 | 144.58 | 15.63% | 20.45 | 20.45 | 58.42 | 107.18 |
| Back | KELLY_0.50 | 339 | 21946.66 | -91.91 | -0.42% | 87.88 | 87.88 | 2563.22 | 4879.75 |
| Lay | KELLY_0.50 | 48 | 1019.90 | 124.43 | 12.20% | 20.45 | 20.45 | 70.66 | 126.54 |
| Back | KELLY_1.00 | 339 | 22772.34 | -250.54 | -1.10% | 87.88 | 87.88 | 3237.40 | 6149.30 |
| Lay | KELLY_1.00 | 48 | 1035.03 | 133.49 | 12.90% | 20.45 | 20.45 | 63.71 | 110.57 |

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
| Back | Pre | Yes | FLAT | 95 | 95.00 | 2.21 | 2.32% | 1.00 | 37.67 |
| Back | Pre | Yes | PROXY | 95 | 31226.53 | -8818.22 | -28.24% | 4597.59 | 71962.54 |
| Back | Pre | Yes | KELLY_0.10 | 84 | 2340.70 | -16.44 | -0.70% | 47.36 | 1156.56 |
| Back | Pre | Yes | KELLY_0.25 | 84 | 5020.14 | 114.08 | 2.27% | 87.88 | 1465.62 |
| Back | Pre | Yes | KELLY_0.50 | 84 | 6070.75 | 239.26 | 3.94% | 87.88 | 1159.77 |
| Back | Pre | Yes | KELLY_1.00 | 84 | 6222.30 | 181.85 | 2.92% | 87.88 | 1480.42 |
| Back | Pre | No | FLAT | 297 | 297.00 | -1.04 | -0.35% | 1.00 | 84.42 |
| Back | Pre | No | PROXY | 297 | 124478.59 | -17436.68 | -14.01% | 4419.01 | 136697.72 |
| Back | Pre | No | KELLY_0.10 | 255 | 6367.19 | 297.68 | 4.68% | 56.97 | 1389.57 |
| Back | Pre | No | KELLY_0.25 | 255 | 12814.74 | 155.33 | 1.21% | 87.88 | 3520.56 |
| Back | Pre | No | KELLY_0.50 | 255 | 15875.91 | -331.17 | -2.09% | 87.88 | 5918.10 |
| Back | Pre | No | KELLY_1.00 | 255 | 16550.03 | -432.39 | -2.61% | 87.88 | 6929.61 |
| Back | In | Yes | FLAT | 58 | 58.00 | -4.52 | -7.80% | 1.00 | 49.64 |
| Back | In | Yes | PROXY | 58 | 9973.72 | -3099.84 | -31.08% | 1457.62 | 34113.77 |
| Back | In | No | FLAT | 129 | 129.00 | 13.26 | 10.28% | 1.00 | 0.00 |
| Back | In | No | PROXY | 129 | 29769.35 | -6434.83 | -21.62% | 2077.23 | 67907.05 |
| Lay | Pre | Yes | FLAT | 13 | 14.38 | 3.45 | 24.02% | 1.00 | 6.86 |
| Lay | Pre | Yes | PROXY | 13 | 6138.31 | -3626.31 | -59.08% | 3925.76 | 39497.31 |
| Lay | Pre | Yes | KELLY_0.10 | 9 | 96.94 | 16.34 | 16.85% | 20.09 | 114.58 |
| Lay | Pre | Yes | KELLY_0.25 | 9 | 149.48 | 35.07 | 23.46% | 20.45 | 100.47 |
| Lay | Pre | Yes | KELLY_0.50 | 9 | 179.82 | 41.42 | 23.04% | 20.45 | 99.80 |
| Lay | Pre | Yes | KELLY_1.00 | 9 | 182.11 | 43.71 | 24.00% | 20.45 | 98.47 |
| Lay | Pre | No | FLAT | 36 | 41.16 | 3.28 | 7.98% | 1.00 | 2.52 |
| Lay | Pre | No | PROXY | 36 | 10669.70 | -3767.66 | -35.31% | 3276.93 | 32024.81 |
| Lay | Pre | No | KELLY_0.10 | 26 | 380.73 | 38.85 | 10.20% | 20.45 | 37.69 |
| Lay | Pre | No | KELLY_0.25 | 26 | 538.72 | 69.48 | 12.90% | 20.45 | 21.85 |
| Lay | Pre | No | KELLY_0.50 | 26 | 587.61 | 44.79 | 7.62% | 20.45 | 68.32 |
| Lay | Pre | No | KELLY_1.00 | 26 | 587.61 | 44.79 | 7.62% | 20.45 | 65.27 |
| Lay | In | Yes | FLAT | 26 | 31.49 | 3.39 | 10.77% | 1.00 | 5.04 |
| Lay | In | Yes | PROXY | 26 | 5812.58 | 1881.87 | 32.38% | 1713.54 | 3145.55 |
| Lay | In | No | FLAT | 49 | 76.33 | -0.08 | -0.10% | 1.00 | 24.75 |
| Lay | In | No | PROXY | 49 | 9137.55 | -2645.04 | -28.95% | 1043.16 | 22337.04 |
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
| Back | Pre | Yes | 178 | 109 | +4.03% [+3.22%, +4.85%] | -2.81% [-15.82%, +10.03%] | -6.96% | pre: Kelly OK |
| Back | Pre | No | 870 | 177 | +1.76% [+1.33%, +2.19%] | -3.49% [-10.61%, +3.83%] | -5.78% | pre: Kelly OK |
| Back | In | Yes | 243 | 122 | — — | +1.86% [-12.61%, +17.04%] | -2.67% | in: use FLAT/PROXY |
| Back | In | No | 519 | 194 | — — | +2.72% [-5.91%, +11.26%] | +3.31% | in: use FLAT/PROXY |
| Lay | Pre | Yes | 199 | 105 | +0.74% [-0.18%, +1.64%] | +2.46% [-10.80%, +15.67%] | -2.20% | pre: Kelly OK |
| Lay | Pre | No | 729 | 173 | +0.09% [-0.50%, +0.70%] | +2.35% [-5.42%, +10.40%] | -0.22% | pre: Kelly OK |
| Lay | In | Yes | 263 | 138 | — — | +24.72% [+6.27%, +45.25%] | +18.20% | in: use FLAT/PROXY |
| Lay | In | No | 343 | 164 | — — | +10.22% [-4.32%, +25.20%] | +19.81% | in: use FLAT/PROXY |
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

**Escala Kelly usada nesta curva**: P99_PROXY | ref_back=4394.24 ref_lay=2045.09 | cap_back=2.0% cap_lay=1.0% | max_stake_event=100%*limit

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
| Combinações elegíveis (edge + timing + t0) | 241 |
| Com ROI disponível (precisa de placar) | 214 |
| Com CLV disponível (pre-match + closing) | 129 |

**Calendário do walk-forward (dias únicos)**

| Tipo | Dias |
|---|---:|
| Dias com dados carregados (audited_at) | 7 |
| Dias com eventos OK/betslip conf. | 5 |
| Dias com eventos elegíveis p/ WF (edge) | 5 |
| Dias usados no walk-forward | 7 |

**Diagnóstico por dia (audited_at): betslip vs qualidade vs edge**

| Dia | Auditorias carregadas | Betslip bruto | Betslip conf. | OK (conf.) | Edge Back/Lay | %OK/conf. | Status não-OK dominante |
|---|---:|---:|---:|---:|---:|---:|---|
| 2026-02-12 | 96 | 88 | 55 | 55 | 20/5 | 100.0% | — |
| 2026-02-13 | 989 | 869 | 579 | 579 | 209/60 | 100.0% | — |
| 2026-02-14 | 1167 | 893 | 506 | 506 | 177/77 | 100.0% | — |
| 2026-02-15 | 796 | 682 | 444 | 444 | 142/51 | 100.0% | — |
| 2026-02-16 | 454 | 336 | 226 | 226 | 93/13 | 100.0% | — |
| 2026-02-17 | 264 | 0 | 0 | 0 | 0/0 | —% | — |
| 2026-02-18 | 266 | 0 | 0 | 0 | 0/0 | —% | — |

Leitura:
- Se `Auditorias carregadas > 0` mas `Betslip conf.` ≈ 0, geralmente houve **mismatch/parse** (diff fora de [-10,+10]) ou ausência de betslip.
- Se `Betslip conf. > 0` mas `OK (conf.) = 0`, o robô coletou betslip, mas os eventos falharam por **status != OK** (ver coluna de status).
- Dias com `OK (conf.) = 0` **não devem ser tratados como “0 oportunidade”** sem investigar o operacional.


Leitura: o walk-forward mede OOS principalmente por **ROI**, então ele encolhe quando a cobertura de placar é baixa. Além disso, a métrica é agregada por **jogo único** (cluster), então você verá números menores que o N de eventos.

| Train window | Test window | #ativas | Jogos OOS | ROI OOS (mean; IC90) | Turnover (teste) | Lucro (estratégia, budget) |
|---|---|---:|---:|---:|---:|---:|
| 2026-02-12→2026-02-14 | 2026-02-15→2026-02-15 | 2 | 48 | +11.53% [-8.69%, +31.86%] | 769.69 | 138.94 |
| 2026-02-12→2026-02-15 | 2026-02-16→2026-02-16 | 3 | 42 | -0.74% [-29.99%, +35.06%] | 865.51 | -100.54 |
| 2026-02-12→2026-02-16 | 2026-02-17→2026-02-17 | 2 | 0 | — — | 0.00 | 0.00 |
| 2026-02-12→2026-02-17 | 2026-02-18→2026-02-18 | 2 | 0 | — — | 0.00 | 0.00 |

**Frequência de ativação por combinação (quantas janelas ela entrou como ativa)**

| Combinação | #steps ativa |
|---|---:|
| Back_Pre_Any | 4 |
| Back_In_Any | 3 |
| Lay_In_No | 2 |

### 12.A Transparência da seleção: métricas por combinação no treino
Para cada janela de treino, mostramos as métricas usadas para decidir se cada combinação ficou **ativa** ou não. Isso ajuda a entender, por exemplo, por que nenhuma Lay entrou em algumas janelas (geralmente N insuficiente ou ROI p30 <= 0).

**Regra de elegibilidade (todas as combinações):** exige `N_ROI >= wf_min_matches` (aqui: 20).

**Train 2026-02-12→2026-02-14 → Test 2026-02-15→2026-02-15**

| Combinação | Ativa? | Jogos treino (tot/CLV/ROI) | CLV q10/q90 ou CI | ROI q30 | Motivo |
|---|---|---:|---:|---:|---|
| Back_Pre_Any | SIM | 99 / 91 / 90 | q10=6.19 | CI90_lb=6.05 | +5.99% | BackPre: clv_q10>0=True, roi_mean>0=True |
| Back_In_Any | NÃO | 81 / — / 69 | — | -2.72% | BackIn: roi_q30>0 AND N_ROI>=min (N=69/20) |
| Lay_Pre_Yes | NÃO | 8 / 6 / 8 | q10=-8.56 | CI90_lb=-9.29 | -0.87% | LayPre: clv_conv_q10>0=False, roi_q30>0=False |
| Lay_Pre_No | NÃO | 29 / 23 / 26 | q10=-7.07 | CI90_lb=-7.57 | +9.13% | LayPre: clv_conv_q10>0=False, roi_q30>0=True |
| Lay_In_Yes | NÃO | 11 / — / 7 | — | +2.44% | In: roi_q30>0 AND N_ROI>=min (N=7/20) |
| Lay_In_No | SIM | 31 / — / 29 | — | +1.03% | In: roi_q30>0 AND N_ROI>=min (N=29/20) |

**Train 2026-02-12→2026-02-15 → Test 2026-02-16→2026-02-16**

| Combinação | Ativa? | Jogos treino (tot/CLV/ROI) | CLV q10/q90 ou CI | ROI q30 | Motivo |
|---|---|---:|---:|---:|---|
| Back_Pre_Any | SIM | 121 / 108 / 111 | q10=6.42 | CI90_lb=6.32 | +7.11% | BackPre: clv_q10>0=True, roi_mean>0=True |
| Back_In_Any | SIM | 123 / — / 107 | — | +1.11% | BackIn: roi_q30>0 AND N_ROI>=min (N=107/20) |
| Lay_Pre_Yes | NÃO | 12 / 9 / 12 | q10=-6.08 | CI90_lb=-6.88 | +6.65% | LayPre: clv_conv_q10>0=False, roi_q30>0=False |
| Lay_Pre_No | NÃO | 33 / 27 / 30 | q10=-6.69 | CI90_lb=-7.11 | +7.96% | LayPre: clv_conv_q10>0=False, roi_q30>0=True |
| Lay_In_Yes | NÃO | 25 / — / 19 | — | +6.49% | In: roi_q30>0 AND N_ROI>=min (N=19/20) |
| Lay_In_No | SIM | 41 / — / 38 | — | +2.04% | In: roi_q30>0 AND N_ROI>=min (N=38/20) |

**Train 2026-02-12→2026-02-16 → Test 2026-02-17→2026-02-17**

| Combinação | Ativa? | Jogos treino (tot/CLV/ROI) | CLV q10/q90 ou CI | ROI q30 | Motivo |
|---|---|---:|---:|---:|---|
| Back_Pre_Any | SIM | 132 / 119 / 122 | q10=6.46 | CI90_lb=6.39 | -0.53% | BackPre: clv_q10>0=True, roi_mean>0=True |
| Back_In_Any | SIM | 136 / — / 117 | — | +6.17% | BackIn: roi_q30>0 AND N_ROI>=min (N=117/20) |
| Lay_Pre_Yes | NÃO | 13 / 10 / 13 | q10=-6.31 | CI90_lb=-6.92 | +14.17% | LayPre: clv_conv_q10>0=False, roi_q30>0=False |
| Lay_Pre_No | NÃO | 34 / 28 / 31 | q10=-6.80 | CI90_lb=-7.09 | +4.67% | LayPre: clv_conv_q10>0=False, roi_q30>0=True |
| Lay_In_Yes | NÃO | 26 / — / 19 | — | +6.49% | In: roi_q30>0 AND N_ROI>=min (N=19/20) |
| Lay_In_No | NÃO | 43 / — / 39 | — | -0.11% | In: roi_q30>0 AND N_ROI>=min (N=39/20) |

**Train 2026-02-12→2026-02-17 → Test 2026-02-18→2026-02-18**

| Combinação | Ativa? | Jogos treino (tot/CLV/ROI) | CLV q10/q90 ou CI | ROI q30 | Motivo |
|---|---|---:|---:|---:|---|
| Back_Pre_Any | SIM | 132 / 119 / 122 | q10=6.46 | CI90_lb=6.39 | -0.53% | BackPre: clv_q10>0=True, roi_mean>0=True |
| Back_In_Any | SIM | 136 / — / 117 | — | +6.17% | BackIn: roi_q30>0 AND N_ROI>=min (N=117/20) |
| Lay_Pre_Yes | NÃO | 13 / 10 / 13 | q10=-6.31 | CI90_lb=-6.92 | +14.17% | LayPre: clv_conv_q10>0=False, roi_q30>0=False |
| Lay_Pre_No | NÃO | 34 / 28 / 31 | q10=-6.80 | CI90_lb=-7.09 | +4.67% | LayPre: clv_conv_q10>0=False, roi_q30>0=True |
| Lay_In_Yes | NÃO | 26 / — / 19 | — | +6.49% | In: roi_q30>0 AND N_ROI>=min (N=19/20) |
| Lay_In_No | NÃO | 43 / — / 39 | — | -0.11% | In: roi_q30>0 AND N_ROI>=min (N=39/20) |


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
| Dias OOS (calendário de teste) | 4 |
| Dias OOS com OK (>=1 evento OK/conf) | 2 |
| Turnover 30d (proj., calendário) | 12264.00 |
| Turnover 30d (proj., cond OK) | 24528.01 |
| Turnover 30d (Pre/In) | 46942.55 / 420.68 |
| Lucro 30d (obs., calendário) | 292.74 |
| Lucro 30d (obs., cond OK) | 585.49 |
| Lucro 30d (obs.) Pre/In | 1336.34 / 48.95 |
| Lucro 30d (exp., calendário) | 288.04 |
| Lucro 30d (exp., cond OK) | 576.07 |
| Lucro 30d (exp.) Pre/In | 1336.34 / 66.18 |
| Banca risco p99 (Back+Lay) | 869.03 |
| Banca liquidez p99 (+buf) | 1374.55 |
| Banca recomendada (max) | 1374.55 |
| ROI/banca 30d (obs., calendário) | 21.30% |
| ROI/banca 30d (obs., cond OK) | 42.59% |
| ROI/banca 30d (exp., calendário) | 20.96% |
| ROI/banca 30d (exp., cond OK) | 41.91% |
| DD 30d p95 (obs., calendário) | 742.26 |
| DD 30d p95 (obs., cond OK) | 997.24 |
| DD 30d p95 (exp., calendário) | 727.49 |
| DD 30d p95 (exp., cond OK) | 976.75 |

### 12.2 Governança de exposição por jogo (budget por `match_id`) — sensibilidade
Objetivo: evitar concentração quando um mesmo jogo gera muitos sinais. Simulamos um orçamento de exposição por jogo, consumido ao longo do tempo.

- **Back** consome budget em **stake**.
- **Lay** consome budget em **liability**.
- Também aplicamos um cap por sinal como fração do budget do jogo (para não gastar tudo no 1º sinal).

Importante: o budget é parametrizado como fração de uma referência de banca. Aqui usamos:
- se `--kelly-bankroll` estiver setado: essa banca explícita;
- senão: a `Banca recomendada (max)` estimada na 12.1 (baseline).

Referência de banca p/ budget: 1374.55 | budgets por jogo aplicados em stake (Back) e liability (Lay).

| Cenário | Turnover 30d | Lucro 30d (exp.) | Banca rec. (max) | ROI/banca 30d (exp.) | DD 30d p95 (exp.) |
|---|---:|---:|---:|---:|---:|
| BASELINE (sem budget) | 12264.00 | 288.04 | 1374.55 | 20.96% | 727.49 |
| BUDGET_0.50%/0.25% cap25% | 1874.20 | 38.20 | 135.43 | 28.20% | 81.72 |
| BUDGET_1.00%/0.50% cap33% | 4199.61 | 118.86 | 296.61 | 40.07% | 216.83 |
| BUDGET_2.00%/1.00% cap50% | 10206.83 | 269.36 | 688.40 | 39.13% | 672.94 |
| BUDGET_3.00%/1.50% cap33% | 11557.32 | 267.76 | 818.98 | 32.69% | 691.38 |
| BUDGET_4.00%/2.00% cap33% | 15104.44 | 401.03 | 1069.25 | 37.51% | 823.71 |
| BUDGET_3.00%/1.50% cap50% | 14777.93 | 519.69 | 992.49 | 52.36% | 852.84 |
| BUDGET_4.00%/2.00% cap50% | 18862.79 | 881.10 | 1263.84 | 69.72% | 835.16 |

Leitura:
- Se a curva com budget melhora muito (menos negativo ou mais positivo) com pouca perda de turnover, o problema era **concentração por jogo**.
- Se tudo continuar negativo, o problema é **edge OOS** (principalmente in‑match) e budget só reduz a escala da perda.

**Volume e stake médio por combinação (janela OOS, com budget padrão)**

| Combinação | #steps ativa | Jogos OOS (uniques) | N eventos OOS | Stake eq. médio | Observação |
|---|---:|---:|---:|---:|---|
| Back_Pre_Any | 4 | 65 | 121 | 51.73 | budget reduz concentração por jogo |
| Back_In_Any | 3 | 14 | 19 | 1.00 | budget reduz concentração por jogo |
| Lay_In_No | 2 | 12 | 15 | 2.47 | budget reduz concentração por jogo |
## 10) Diagnóstico: por que o ROI pode estar zerado
ROI aqui é calculado por placar do jogo (`matches.home_score/away_score`). Se os placares não estiverem preenchidos no banco, a cobertura de ROI será 0.

| Indicador | Valor |
|---|---:|
| Jogos únicos no recorte | 329 |
| Jogos com placar disponível (home_score/away_score não nulos) | 288 |
| Jogos com status='finished' no banco | 288 |

### 10.1 Distribuição por data de kickoff (explica janela da API)
- Kickoff (UTC) no recorte: **2026-02-12 15:00 UTC** até **2026-02-18 22:30 UTC**.

| Kickoff date (UTC) | Jogos | Com placar | Cobertura |
|---|---:|---:|---:|
| 2026-02-18 | 15 | 14 | 93.3% |
| 2026-02-17 | 46 | 46 | 100.0% |
| 2026-02-16 | 24 | 19 | 79.2% |
| 2026-02-15 | 75 | 70 | 93.3% |
| 2026-02-14 | 114 | 104 | 91.2% |
| 2026-02-13 | 50 | 33 | 66.0% |
| 2026-02-12 | 5 | 2 | 40.0% |

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
