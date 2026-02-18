# Análise Estatística Robusta — Contexto Operação (b808)
**Data da execução:** 18/02/2026 13:13 UTC  
**Escopo:** H3B (auditoria), com comparação por `audit_version` e inferência robusta por jogo (cluster bootstrap).  
**Nota:** a robustez aqui significa que intervalos de confiança consideram correlação intra-jogo (múltiplas auditorias por partida).
---
## 0) Sumário executivo (leitura rápida)
- **Recorte**: direction=`up`, lookback_days=`7`, versions=`v4.0-api,v1.0,v1.0-recovered`.
- **Amostra**: 3909 auditorias (jogos únicos=343, média=11.4 obs/jogo); betslip confiável=1911.
- **Janela efetiva (audited_at)**: 11/02 13:14 → 17/02 21:33 UTC (span≈6.3d; dias com dados=7).
- **Coortes (status=OK, betslip confiável)**: `BS>WS` (diff>=2.0%): **679**; `BS<WS` (diff<=-2.0%): **227**.
- **Coberturas em `hypothesis_details` (OK)**: temporal(back)=1450/1911; lay_temporal=1317/1911; finance=1222/1911.
- **Cobertura de placar (ROI)**: jogos com placar=298/343 (status finished=298).
- **Cobertura de closing_odd (AH)**: jogos com closing=192/343 (56.0%). CLV pre‑match depende disso.
- **Alerta**: cobertura de closing_odd está baixa. Isso tende a reduzir N de CLV e pode enviesar a leitura (os jogos “sem odds” ficam fora do CLV). Priorize estabilizar o collector e/ou condicionar análises a jogos com closing.
- **DOM**: sem dados no recorte atual (N=0), então não há comparação API vs DOM aqui.
- **CLV pre-match (Betslip, API)**: média robusta por jogo +2.131% (IC90 [+1.636%, +2.630%]), com N=920 eventos (jogos=154).
- **Padrão por bucket (CLV PM)**: `BS < WS` -3.928% (sig. negativo), `BS ~ WS` -0.165% (NS), `BS > WS` +6.648% (sig. positivo).
- **Leitura recomendada**: use este relatório para validar **qualidade de execução/CLV**. Para concluir sobre **ROI**, rode após atualizar resultados e/ou use uma janela com jogos já liquidados.

---
## 1) Contexto do corte (b808)
| Indicador | Valor |
|---|---:|
| Auditorias H3B `UP` (match + kickoff passado) | 3909 |
| Betslip bruto | 3054 |
| Betslip confiável (diff -10% a +10%) | 1911 |
| Descartados no filtro de qualidade | 1143 |
| Jogos únicos (geral) | 343 |
| Média de observações por jogo | 11.4 |
| Jogos únicos com betslip confiável | 317 |
| Distribuição por market_type | AH=3909 |
| Jogos únicos (AH) no recorte | 343 |
| Jogos únicos (AH) com closing_odd disponível | 192 |
| Cobertura closing_odd (AH) | 56.0% |

---
## 2) Base comparativa: API vs DOM
| Métrica | API (v4.0-api) | DOM (v1.0) |
|---|---:|---:|
| Total de observações | 3909 | 0 |
| Com betslip confiável | 1911 | 0 |
| Com CLV pre-match (betslip) | 920 | 0 |
| Com ROI (betslip) | 1710 | 0 |
| Tempo total observado (detecção→betslip, wall/total) | 15363 ms | — ms |
| Tempo instrumentado (detecção→clique→betslip) | 8343 ms | — ms |

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
| API (2-4s) | lag_det→click | 5507 | 808 | 6417 | 3909 |
| API (2-4s) | lag_click→betslip | 2612 | 2150 | 4380 | 3618 |
| API (2-4s) | lag_e2e (soma) | 8343 | 3422 | 9450 | 3618 |
| API (2-4s) | audit_total (duração) | 15359 | 4743 | 52591 | 3909 |
| API (2-4s) | overhead (total - e2e) | 7790 | 258 | 30822 | 3618 |
| DOM (15-30s) | lag_det→click | — | — | — | 0 |
| DOM (15-30s) | lag_click→betslip | — | — | — | 0 |
| DOM (15-30s) | lag_e2e (soma) | — | — | — | 0 |
| DOM (15-30s) | audit_total (duração) | — | — | — | 0 |
| DOM (15-30s) | overhead (total - e2e) | — | — | — | 0 |

**Diagnóstico de cauda (percentual acima do limiar)**

| Modelo | % det→click > 5s | % det→click > 20s | % total > 10s | % total > 40s | % overhead < 0 |
|---|---:|---:|---:|---:|---:|
| API (2-4s) | 5.4% | 3.1% | 19.0% | 6.5% | 0.0% |
| DOM (15-30s) | —% | —% | —% | —% | —% |

---
### 2.1 Cobertura temporal (pre-match vs in-match)
| Métrica | Pre-match | In-match | Observação |
|---|---:|---:|---|
| Observações totais com classificação temporal | 1682 | 2227 | Contagem bruta do corte |
| ROI Betslip | 990 | 720 | Amostra com resultado do jogo |
| ROI WebSocket | 1553 | 1950 | Referência de mercado |
| CLV (apenas pre-match) | 920 | — | CLV vs closing pré-jogo não é interpretável in-match |

---
### 2.2 Performance por regime (pre-match vs in-match)
| Regime | N auditorias | N betslip conf. | N OK | Back edge | Lay edge | Diff médio (OK) |
|---|---:|---:|---:|---:|---:|---:|
| PRE_MATCH | 1682 | 1071 | 1071 | 418 | 71 | +2.219% |
| IN_MATCH | 2227 | 840 | 840 | 261 | 156 | +0.980% |

---
### 2.3 Regimes operacionais por tempo total (bucket)
Objetivo: separar a amostra em **regimes de execução** (tempo total) e medir performance em cada regime. Use isso para detectar **fila/saturação**: regimes lentos tendem a ter edge menor e pior ROI.

| Bucket (tempo total) | N OK | Jogos | lag_total mean (ms) | lag_total p95 (ms) | overhead p95 (ms) | Back edge | Lay edge | CLV PM (mean; IC90) | ROI (mean; IC90) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| < 5s | 1262 | 268 | 3582 | 4843 | 2504 | 420 | 134 | +2.34% [+1.90%, +2.78%] | +3.27% [-2.68%, +9.30%] |
| 5-10s | 230 | 145 | 5892 | 8252 | 2915 | 95 | 45 | +1.78% [+0.84%, +2.73%] | -3.33% [-15.10%, +8.72%] |
| 10-20s | 18 | 17 | 13668 | 18117 | 4070 | 9 | 2 | +1.17% [-1.58%, +3.89%] | +10.43% [-23.26%, +43.02%] |
| 20-40s | 274 | 124 | 27257 | 34384 | 30679 | 95 | 31 | +2.27% [+1.36%, +3.27%] | +9.52% [-1.58%, +20.69%] |
| > 40s | 127 | 77 | 139062 | 439121 | 246483 | 60 | 15 | +2.43% [+1.33%, +3.57%] | -11.26% [-26.85%, +4.84%] |
| Desconhecido | 0 | 0 | — | — | — | 0 | 0 | — | — |

---
### 2.3b Regimes por tempo total — separando coortes por `diff_pct` (BS vs WS)
Nesta tabela, separamos duas coortes operacionais por **delta de execução**: `BS > WS` (diff_pct >= +2%) e `BS < WS` (diff_pct <= -2%). Isso **não** é (por si só) “Back vs Lay”; é um recorte por **melhora/piora do preço** entre detecção (WS) e execução (BS). CLV é reportado apenas em pre‑match.

| Bucket | N OK | N (BS>WS +2%) | N (BS<WS -2%) | CLV PM (BS>WS) | CLV PM (BS<WS) | ROI (BS>WS) | ROI (BS<WS) |
|---|---:|---:|---:|---:|---:|---:|---:|
| < 5s | 1262 | 420 | 134 | +7.04% [+6.60%, +7.46%] | -3.58% [-4.78%, -2.36%] | +3.56% [-6.42%, +13.95%] | -0.47% [-15.89%, +14.43%] |
| 5-10s | 230 | 95 | 45 | +5.65% [+4.74%, +6.49%] | -5.91% [-7.17%, -4.54%] | -5.65% [-23.36%, +12.05%] | -11.41% [-31.05%, +8.92%] |
| 10-20s | 18 | 9 | 2 | +3.86% [+1.34%, +6.62%] | -6.98% — | -10.16% [-55.47%, +36.27%] | +40.04% [+0.00%, +79.90%] |
| 20-40s | 274 | 95 | 31 | +7.77% [+6.73%, +8.80%] | -2.03% [-4.39%, +0.55%] | +7.62% [-9.74%, +25.38%] | +6.33% [-24.85%, +35.90%] |
| > 40s | 127 | 60 | 15 | +5.31% [+3.52%, +7.08%] | -7.29% — | -14.78% [-36.89%, +8.24%] | -11.35% [-55.50%, +31.89%] |
| Desconhecido | 0 | 0 | 0 | — | — | — | — |

---
### 2.3c Estabilidade temporal (por dia, `audited_at`)
Objetivo: checar se o regime de edge/execução é **time‑dependent**. Se houver dias com comportamento distinto (ex.: collector intermitente, horários/ligas), isso aparecerá aqui.

| Dia (UTC) | Modelo | N OK | Jogos | % BS>WS +2% | % BS<WS -2% | lag_total p50 (ms) | CLV PM (BS>WS) | CLV PM (BS<WS) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-02-11 | API (2-4s) | 157 | 56 | 38.9% | 17.2% | 27149 | +7.02% | -4.08% |
| 2026-02-12 | API (2-4s) | 51 | 41 | 35.3% | 7.8% | 26856 | +5.67% | -5.50% |
| 2026-02-13 | API (2-4s) | 574 | 137 | 36.2% | 10.3% | 4821 | +6.98% | -3.29% |
| 2026-02-14 | API (2-4s) | 500 | 136 | 35.4% | 15.0% | 4207 | +6.25% | -5.29% |
| 2026-02-15 | API (2-4s) | 432 | 97 | 31.7% | 11.3% | 3691 | +6.36% | -2.04% |
| 2026-02-16 | API (2-4s) | 197 | 58 | 39.6% | 6.6% | 3469 | +7.08% | -6.79% |

---
## 3) CLV pre-match (núcleo)
### 3.1 CLV com odd do Betslip (execução real)
| Métrica | API | DOM |
|---|---:|---:|
| CLV Bruto BS Pre-Match | +2.323% (sig. positivo, N=920, jogos=154) | — (N/A, N=0, jogos=0) |
| CLV Adicional BS Pre-Match | +1.798% (sig. positivo, N=920, jogos=154) | — (N/A, N=0, jogos=0) |
| Taxa de CLV > 0 (bruto) | 60.4% | —% |
| Taxa de CLV > 0 (adicional) | 61.8% | —% |

Notas de robustez (IC 90% por jogo):  
- API CLV bruto (cluster): média +2.131%; IC90 [+1.636%, +2.630%]  
- DOM CLV bruto (cluster): média —; IC90 —  

---
## 4) ROI por modelo
| Métrica | API | DOM |
|---|---:|---:|
| ROI Betslip | +1.235% (NS, N=1708) | — (N/A, N=0) |
| ROI WebSocket | -0.715% (sig. negativo, N=3487) | — (N/A, N=0) |
| Win rate ROI Betslip | 50.6% | —% |
| Win rate ROI WS | 50.0% | —% |

Notas de robustez (IC 90% por jogo):  
- API ROI betslip (cluster): média +0.944%; IC90 [-3.958%, +5.808%]  
- API ROI WS (cluster): média -2.905%; IC90 [-5.827%, -0.088%]  

---
## 5) Diferença de preço BS vs WS
| Métrica | API | DOM |
|---|---:|---:|
| Diff BS vs WS (média) | +1.674% (sig. positivo, N=1911) | — (N/A, N=0) |
| BS > WS | 47.5% (908/1911) | —% (0/0) |
| BS > WS +2% | 35.5% (679/1911) | —% (0/0) |

---
## 6) Combinações de valor
### 6.1 Buckets por diferença BS vs WS
Nota de leitura:
- `BS > WS (+2% a +10%)`: preço no betslip ficou **melhor** do que o WS observado na detecção (delta positivo entre instantes).
- `BS < WS (-10% a -2%)`: preço no betslip ficou **pior** do que o WS observado na detecção (delta negativo entre instantes).
- O ROI abaixo é calculado **dentro do bucket** (não mistura buckets), mas ainda é sensível à cobertura de placar.

| Bucket | N bucket | CLV BS PM (média) | IC90 (cluster) | N CLV PM | Jogos CLV PM | ROI BS (todos) | IC90 (cluster) |
|---|---:|---:|---|---:|---:|---:|---|
| BS < WS (-10% a -2%) | 227 | -3.928% | [-4.914%, -3.049%] | 58 | 45 | +2.579% | [-14.129%, +8.956%] |
| BS ~ WS (-2% a +2%) | 1005 | -0.165% | [-0.612%, +0.121%] | 494 | 135 | -0.029% | [-10.275%, +2.486%] |
| BS > WS (+2% a +10%) | 679 | +6.648% | [+6.467%, +7.288%] | 368 | 115 | +2.642% | [-1.926%, +14.536%] |

---
### 6.2 Combinação por faixa de linha AH
| Faixa AH | CLV BS PM (média) | IC90 (cluster) | ROI BS (todos) | IC90 (cluster) | Diff BS vs WS (média) |
|---|---:|---|---:|---|---:|
| AH 0-1 (líquida) | +2.557% | [+2.153%, +3.563%] | +0.771% | [-0.655%, +13.346%] | +1.907% |
| AH 1-2 (média) | +2.287% | [+1.169%, +3.401%] | +1.448% | [-11.698%, +12.875%] | +2.388% |
| AH 2+ (extrema) | +1.959% | [+0.245%, +1.943%] | +1.548% | [-7.882%, +8.125%] | +1.202% |

---
### 6.3 Combinação por faixa de lag
| Faixa de lag | CLV BS PM (média) | IC90 (cluster) | N CLV PM | Jogos CLV PM | ROI BS (todos) | IC90 (cluster) | Diff BS vs WS (média) |
|---|---:|---|---:|---:|---:|---|---:|
| < 10s | +2.371% | [+1.694%, +2.601%] | 697 | 143 | +1.280% | [-3.775%, +7.959%] | +1.618% |
| 10-20s | +1.220% | [-1.579%, +3.893%] | 7 | 7 | +9.839% | [-23.261%, +43.022%] | +1.793% |
| 20-30s | +2.261% | [+1.390%, +3.301%] | 137 | 75 | +4.045% | [-4.434%, +19.704%] | +1.792% |
| > 30s | +2.105% | [+1.294%, +3.328%] | 79 | 53 | -3.999% | [-17.621%, +8.932%] | +1.990% |

---
## 7) Estimativa financeira (proxy) e risco
Este bloco usa `hypothesis_details.finance` quando existe; se não existir, usa stake fallback = `stake_pct_of_limit × limit`.

**Política de stake (proxy)**

| Parâmetro | Valor |
|---|---:|
| stake_pct_of_limit | 0.25 |
| stake_cap | 0.00 |
| Cobertura finance (OK, betslip conf.) | 1222/1911 |

### 7.1 Back (BS >> WS)
| Métrica | Valor |
|---|---:|
| Corte (diff_pct) | >= 2.0% |
| N eventos | 679 |
| Cobertura finance (na coorte) | 426/679 |
| Stake total (estimado) | 217839.75 |
| Stake médio | 320.82 |
| Profit_if_win total (estimado) | 233838.37 |
| Profit_if_win médio | 344.39 |
| N com ROI realizado | 615 |
| P&L realizado total (estimado) | -43547.38 |
| ROI realizado (ponderado por stake) | -20.55% |
| ROI realizado (robusto por jogo, mean; IC90) | +6.37% [-1.93%, +14.54%] |
| ROI ponderado por stake (robusto por jogo, mean; IC90) | +5.29% [-4.89%, +15.99%] |

**Como ler as 3 linhas de ROI (Back)**

- **ROI realizado (ponderado por stake)**: ΣP&L / Σstake (pode ser dominado por stakes grandes).
- **ROI realizado (robusto por jogo)**: média por jogo (cada jogo pesa parecido; reduz dominância de outliers).
- **ROI ponderado por stake (robusto por jogo)**: calcula ROI ponderado dentro do jogo e depois faz média/IC por jogo.

Observação: a Seção 4 reporta ROI médio no **recorte completo** (betslip confiável). Aqui (7.1) é apenas a coorte **Back edge** (diff>=corte) e o ROI também é mostrado ponderado por stake; por isso sinais podem divergir.

### 7.2 Lay (BS << WS) — risco de cauda
| Métrica | Valor |
|---|---:|
| Corte (diff_pct) | <= -2.0% |
| N eventos | 227 |
| Cobertura finance (na coorte) | 144/227 |
| Stake total (estimado) | 41223.79 |
| Liability total (estimada) | 36047.62 |
| Liability média | 158.80 |
| Liability p95 | 567.29 |
| Liability p99 | 2043.24 |
| ES95 (liability) | 1663.71 |
| Liability max | 4395.35 |
| Proxy de banca (>= p99 liability) | 2043.24 |
| N com ROI realizado | 148 |
| P&L realizado total (estimado) | -3001.17 |
| ROI realizado (ponderado por liability) | -8.89% |
| ROI realizado (ponderado por stake) | -7.75% |
| ROI/liability (robusto por jogo, mean; IC90) | +12.31% [-2.18%, +27.49%] |
| ROI/liability ponderado (robusto por jogo, mean; IC90) | +13.34% [-1.89%, +29.31%] |

### 7.3 Projeção mensal (30 dias fixo) — turnover, lucro e banca
Premissas:
- **Mês = 30 dias fixo**.
- Turnover = soma de stakes (Back e Lay). Para Lay, também reportamos exposição por **liability**.
- **Banca por risco (unitária)** = p99/ES95(exposição por aposta).
- **Banca por liquidez (turnover)** = p95/p99 de capital **simultaneamente travado** (stake/liability em aberto até a liquidação), capturando distribuição desigual de entradas ao longo do tempo.
- Projeção de lucro usa duas visões: (i) **lucro/dia observado** e (ii) **ROI observado × turnover projetado**.

| Bloco | Janela(d) | Turnover 30d | Lucro 30d (direto) | Lucro 30d (ROI×turnover) |
|---|---:|---:|---:|---:|
| Back | 7.0 | 933598.92 | -186631.63 | -191835.19 |
| Lay (stake) | 7.0 | 176673.38 | -12862.16 | -13690.24 |
| Total (Back+Lay) | 7.0 | 1110272.30 | -199493.80 | -205525.43 |

**Banca por risco (exposição unitária)**

| Bloco | Banca conservadora (p99) | Banca agressiva (ES95) | ROI/banca 30d (direto) | ROI/banca 30d (ROI×turnover) |
|---|---:|---:|---:|---:|
| Back (stake) | 4395.55 | 3928.47 | -4245.93% | -4364.31% |
| Lay (liability) | 2043.24 | 1663.71 | -629.50% | -670.03% |
| Total (soma) | 6438.79 | 5592.18 | -3098.31% | -3191.99% |

**Banca por liquidez (capital simultaneamente travado)**

Definição operacional: cada aposta trava capital de `audited_at` até `kickoff + 2.00h + 2.25h` (grid=5min). A banca recomendada aplica buffer de +10%.

| Bloco | Liquidez mean | Liquidez p95 | Liquidez p99 | Liquidez max | Banca liq p99 (+buffer) |
|---|---:|---:|---:|---:|---:|
| Back (stake) | 19187.88 | 58273.98 | 85182.88 | 88615.66 | 93701.16 |
| Lay (liability) | 2554.12 | 7176.48 | 9501.44 | 11582.29 | 10451.59 |
| Total (Back+Lay) | 21642.67 | 63020.85 | 90081.02 | 95513.70 | 99089.13 |

**Banca recomendada (conservadora)**

| Métrica | Valor |
|---|---:|
| Banca por risco (p99 unitário, soma) | 6438.79 |
| Banca por liquidez (p99 simultâneo + buffer) | 99089.13 |
| Banca efetiva (max das duas) | 99089.13 |
| ROI/banca 30d (direto, banca efetiva) | -201.33% |
| ROI/banca 30d (ROI×turnover, banca efetiva) | -207.41% |
**Diagnóstico de cobertura (placar/ROI)**

| Bloco | Turnover total | Turnover com ROI | Cobertura turnover |
|---|---:|---:|---:|
| Back | 217839.75 | 211930.82 | 97.29% |
| Lay | 41223.79 | 38730.31 | 93.95% |

Notas (Lay): exposição 30d por liability (não é turnover) = 154489.79; ROI realizado por liability (ponderado) = -8.89%.

---
## 8) Curva temporal (pico, reversão e melhor timing)
Esta seção usa `hypothesis_details.temporal` (Back) e `hypothesis_details.lay_temporal` (Lay), coletados em pontos discretos (t≈0,3,6,10,15,20s).

O objetivo é responder: **tempo até o pico/vale**, **% que segue melhorando até t_max**, **% com reversão** e **impacto esperado por timing**. **CLV é reportado somente pre-match** (closing pré-jogo).

### 8.1 Back (pico em diff_pct)
Definições: **pico** = `max(diff_pct)`. **reversão** = após o pico, `diff_pct` cair pelo menos 0.50 p.p. (para Lay, subir 0.50 p.p. após o vale). `t_reversão` é o 1º tempo que cruza esse limiar (logo, não é o pico).

| Regime | N | t_pico médio (s) | t_pico p50 (s) | % melhora até fim | % com reversão | t_reversão médio (s) | Δt (rev - pico) médio (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRE_MATCH | 1071 | 4.2 | 5.2 | 77.9% | 15.8% | 12.2 | 7.2 |
| IN_MATCH | 840 | 5.0 | 0.0 | 63.1% | 28.9% | 13.5 | 8.1 |

**Partição 100% (Back)**: categorias exclusivas (somam 100% por regime).

| Regime | % pico no fim, sem reversão | % pico no fim, com reversão (recuperou) | % pico antes, com reversão | % pico antes, sem reversão relevante |
|---|---:|---:|---:|---:|
| PRE_MATCH | 81.3% | 6.6% | 9.2% | 2.9% |
| IN_MATCH | 68.6% | 4.5% | 24.4% | 2.5% |

**Curva média (Back)**: média de `diff_pct` e `odd` por tempo. `CLV` é reportado **somente pre-match** (closing pré-jogo). `ROI` (quando aparece) é o **ROI realizado** se a aposta fosse feita naquele ponto.

| Tempo | N pts | diff_pct médio | odd média | CLV médio (pre-match) | ROI médio (se houver) |
|---|---:|---:|---:|---:|---:|
| t+0s | 1911 | +1.67% | 2.014 | +2.32% | 1.93 |
| t+6s | 1423 | +2.30% | 2.026 | +2.65% | 1.44 |
| t+10s | 2310 | +2.64% | 2.031 | +2.70% | 2.84 |
| t+15s | 1431 | +2.78% | 2.049 | +2.78% | 1.91 |
| t+20s | 1964 | +3.29% | 2.045 | +2.74% | 0.91 |

### 8.1b Back — impacto por timing (t0 vs pico vs último) e reversão
Leitura: se a estratégia é **entrar no pico**, compare CLV/ROI no pico vs t0 e veja diferença entre **casos com reversão** vs **sem reversão**. Aqui reportamos **média robusta por jogo + IC90**. Observação: **CLV só é calculado pre-match**.

**CLV (somente pre-match) — média robusta por jogo (IC90)**

| Subcoorte | N total | N CLV (PM) | CLV t0 (mean; IC90) | CLV pico (mean; IC90) | CLV último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 1499 | 772 | +1.78% [+1.32%, +2.24%] | +2.05% [+1.59%, +2.49%] | +2.04% [+1.58%, +2.48%] |
| COM_REVERSAO | 412 | 148 | +4.14% [+3.27%, +5.02%] | +5.60% [+4.62%, +6.56%] | +4.69% [+3.71%, +5.68%] |

**ROI (stake=1) — média robusta por jogo (IC90)**

| Subcoorte | N total | N ROI | ROI t0 (mean; IC90) | ROI pico (mean; IC90) | ROI último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 1499 | 1350 | +0.89% [-4.83%, +6.53%] | +1.04% [-4.71%, +6.82%] | +1.03% [-4.72%, +6.79%] |
| COM_REVERSAO | 412 | 358 | +0.12% [-10.40%, +10.97%] | +3.18% [-7.81%, +14.34%] | +0.25% [-10.24%, +11.05%] |

Nota interpretativa: a subcoorte **COM_REVERSAO** pode ter CLV maior no **pico** porque, por definição, ela inclui casos em que o edge atingiu um extremo mais alto antes de reverter. Para entender a hipótese 'sem reversão deveria ser maior', compare também a categoria '**pico no fim, sem reversão**' na partição 100% (tabela 8.1).

**Decomposição (pre-match): entry_odd vs closing_odd (IC90)**

| Subcoorte | N CLV (PM) | entry_odd t0 (mean; IC90) | entry_odd pico (mean; IC90) | closing_odd (mean; IC90) |
|---|---:|---:|---:|---:|
| SEM_REVERSAO | 772 | 1.984 [+1.974, +1.995] | 1.991 [+1.980, +2.002] | 1.966 [+1.958, +1.974] |
| COM_REVERSAO | 148 | 2.037 [+2.018, +2.057] | 2.066 [+2.045, +2.088] | 1.962 [+1.949, +1.976] |

### 8.2 Lay (vale em diff_pct)
| Regime | N | t_vale médio (s) | t_vale p50 (s) | % melhora até fim | % com reversão | t_reversão médio (s) | Δt (rev - vale) médio (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRE_MATCH | 943 | 4.0 | 5.2 | 71.8% | 20.0% | 12.6 | 7.0 |
| IN_MATCH | 676 | 5.5 | 0.0 | 51.3% | 38.9% | 13.5 | 7.9 |

**Partição 100% (Lay)**: categorias exclusivas (somam 100% por regime).

| Regime | % vale no fim, sem reversão | % vale no fim, com reversão (recuperou) | % vale antes, com reversão | % vale antes, sem reversão relevante |
|---|---:|---:|---:|---:|
| PRE_MATCH | 75.2% | 7.4% | 12.6% | 4.8% |
| IN_MATCH | 57.5% | 5.8% | 33.1% | 3.6% |

**Curva média (Lay)**: usa `odd=lay_odd`. `CLV` é reportado **somente pre-match** (closing pré-jogo). O ROI mostrado aqui é **ROI por liability** (não por stake), porque Lay é governado por risco.

| Tempo | N pts | diff_pct médio | lay_odd média | CLV médio (pre-match) | ROI/liability médio (se houver) |
|---|---:|---:|---:|---:|---:|
| t+0s | 1619 | +0.58% | 1.996 | +0.08% | 11.38 |
| t+6s | 1294 | +0.86% | 2.004 | +0.20% | 27.42 |
| t+10s | 2088 | +0.31% | 1.992 | +0.32% | 26.99 |
| t+15s | 1297 | +1.29% | 2.031 | +0.03% | 16.77 |
| t+20s | 1807 | +1.61% | 2.019 | +0.21% | 12.91 |

### 8.2b Lay — impacto por timing (t0 vs vale vs último) e reversão
Leitura: se a estratégia é **entrar no vale (odd mais baixa)**, compare CLV/ROI no vale vs t0 e veja diferença entre **casos com reversão** vs **sem reversão**. Aqui reportamos **média robusta por jogo + IC90**. Observação: **CLV só é calculado pre-match**.

**CLV (somente pre-match) — média robusta por jogo (IC90)**

| Subcoorte | N total | N CLV (PM) | CLV t0 (mean; IC90) | CLV vale (mean; IC90) | CLV último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 1167 | 644 | -0.49% [-1.13%, +0.15%] | -0.17% [-0.80%, +0.47%] | -0.19% [-0.82%, +0.45%] |
| COM_REVERSAO | 452 | 163 | +0.25% [-0.72%, +1.19%] | +1.60% [+0.62%, +2.55%] | +0.40% [-0.58%, +1.34%] |

**ROI/liability — média robusta por jogo (IC90)**

| Subcoorte | N total | N ROI | ROI/liab t0 (mean; IC90) | ROI/liab vale (mean; IC90) | ROI/liab último (mean; IC90) |
|---|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 1167 | 1063 | +6.94% [-1.31%, +15.64%] | +7.58% [-0.78%, +16.38%] | +7.53% [-0.82%, +16.31%] |
| COM_REVERSAO | 452 | 396 | +18.39% [+4.64%, +33.88%] | +26.65% [+8.23%, +49.16%] | +14.35% [+2.29%, +26.55%] |

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
| Back | Pre | Yes | 169 | 103 | +4.14% [+3.27%, +5.02%] | -3.25% [-16.32%, +9.90%] | -7.60% | não (CLV p90>0 AND ROI>0) |
| Back | Pre | No | 902 | 192 | +1.78% [+1.32%, +2.24%] | -4.26% [-11.61%, +2.91%] | -6.44% | não (CLV p90>0 AND ROI>0) |
| Back | In | Yes | 243 | 122 | — | +1.86% [-12.61%, +17.04%] | -2.67% | não (ROI p30>0) |
| Back | In | No | 597 | 215 | — | +3.75% [-4.27%, +11.91%] | +1.16% | sim (ROI p30>0) |
| Lay | Pre | Yes | 189 | 99 | -0.25% [-1.19%, +0.72%] | +3.51% [-9.69%, +17.11%] | -0.66% | não (CLV p90<0 AND ROI p30>0) |
| Lay | Pre | No | 754 | 183 | +0.49% [-0.15%, +1.13%] | +3.78% [-3.82%, +11.48%] | +1.43% | não (CLV p90<0 AND ROI p30>0) |
| Lay | In | Yes | 263 | 138 | — | +24.72% [+6.27%, +45.25%] | +18.20% | sim (ROI p30>0) |
| Lay | In | No | 413 | 181 | — | +10.59% [-2.35%, +23.86%] | +6.15% | sim (ROI p30>0) |

Notas:
- Se você quiser operar **cada combinação como uma estratégia separada**, o sizing (Kelly/caps) deve ser estimado e governado separadamente por estratégia.
- Esta tabela é **in-sample** na janela (`--lookback-days`). A seção OOS (walk-forward) pode ser habilitada com `--walkforward`.

## 9) Combinações de valor (regime × linha AH × lag)
Tabela rankeada por volume (N). Métrica: diff_pct (OK, betslip confiável).

### 9.1 Back combos (diff_pct >= corte)
| Regime | Linha | Lag | N | Jogos | Mean diff % | IC90 cluster |
|---|---|---|---:|---:|---:|---|
| PRE_MATCH | AH 0-1 (líquida) | < 10s | 160 | 60 | +6.81% | [+6.82%, +7.42%] |
| IN_MATCH | AH 2+ (extrema) | < 10s | 119 | 89 | +6.17% | [+5.73%, +6.61%] |
| PRE_MATCH | AH 2+ (extrema) | < 10s | 85 | 39 | +6.99% | [+6.61%, +7.40%] |
| PRE_MATCH | AH 1-2 (média) | < 10s | 78 | 29 | +6.71% | [+6.69%, +7.44%] |
| IN_MATCH | AH 0-1 (líquida) | < 10s | 45 | 37 | +6.66% | [+6.06%, +7.19%] |
| IN_MATCH | AH 1-2 (média) | < 10s | 28 | 27 | +7.01% | [+6.16%, +7.66%] |
| PRE_MATCH | AH 0-1 (líquida) | 20-30s | 25 | 21 | +7.20% | [+6.43%, +7.64%] |
| IN_MATCH | AH 2+ (extrema) | > 30s | 22 | 16 | +6.53% | [+5.17%, +7.41%] |
| PRE_MATCH | AH 0-1 (líquida) | > 30s | 17 | 16 | +6.79% | [+6.02%, +7.51%] |
| PRE_MATCH | AH 2+ (extrema) | 20-30s | 16 | 11 | +7.03% | [+6.04%, +7.76%] |
| PRE_MATCH | AH 1-2 (média) | 20-30s | 14 | 11 | +6.15% | [+5.36%, +7.22%] |
| IN_MATCH | AH 2+ (extrema) | 20-30s | 12 | 10 | +6.66% | [+5.62%, +7.45%] |

### 9.2 Lay combos (diff_pct <= corte) + risco
| Regime | Linha | Lag | N | Jogos | Mean diff % | IC90 cluster | Liability p95 |
|---|---|---|---:|---:|---:|---|---:|
| IN_MATCH | AH 2+ (extrema) | < 10s | 73 | 60 | -4.99% | [-5.54%, -4.58%] | 308.36 |
| IN_MATCH | AH 0-1 (líquida) | < 10s | 34 | 28 | -5.06% | [-5.83%, -4.59%] | 454.08 |
| PRE_MATCH | AH 2+ (extrema) | < 10s | 22 | 21 | -4.91% | [-5.60%, -4.10%] | 832.07 |
| PRE_MATCH | AH 0-1 (líquida) | < 10s | 22 | 19 | -4.74% | [-6.00%, -4.13%] | 4176.11 |
| IN_MATCH | AH 2+ (extrema) | > 30s | 15 | 13 | -4.60% | [-5.66%, -3.72%] | 1438.89 |
| IN_MATCH | AH 1-2 (média) | < 10s | 14 | 13 | -4.31% | [-5.16%, -3.62%] | 660.98 |
| PRE_MATCH | AH 1-2 (média) | < 10s | 14 | 10 | -5.09% | [-6.49%, -3.91%] | 103.88 |
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

- **Back (stake)**: corr(exposição, ROI)=-0.086; corr(exposição, CLV)=-0.001 (onde CLV existe).
- **Lay (liability)**: corr(exposição, ROI)=0.020; corr(exposição, CLV)=0.063 (onde CLV existe).

**Backtest de sizing (apenas eventos com placar; valores em unidade monetária *proxy*)**

| Lado | Scheme | N | Turnover (janela) | Lucro (janela) | ROI/turnover | p99 exposição | ES95 exposição | DD 30d (média) | DD 30d (p95) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Back | FLAT | 615 | 615.00 | 22.72 | 3.69% | 1.00 | 1.00 | 6.15 | 11.12 |
| Lay | FLAT | 199 | 253.48 | -3.85 | -1.52% | 1.00 | 1.00 | 36.83 | 65.23 |
| Back | PROXY | 615 | 211930.82 | -43547.38 | -20.55% | 4398.75 | 4136.53 | 189374.88 | 260967.14 |
| Lay | PROXY | 148 | 38730.31 | -3001.17 | -7.75% | 3306.91 | 2162.72 | 23572.92 | 41909.53 |
| Back | KELLY_0.10 | 326 | 8540.42 | 440.21 | 5.15% | 57.49 | 52.52 | 496.48 | 909.12 |
| Lay | KELLY_0.10 | 46 | 605.70 | 85.43 | 14.10% | 20.43 | 20.43 | 27.41 | 47.89 |
| Back | KELLY_0.25 | 326 | 17364.43 | 499.39 | 2.88% | 87.91 | 87.91 | 1182.66 | 2247.03 |
| Lay | KELLY_0.25 | 46 | 869.63 | 143.15 | 16.46% | 20.43 | 20.43 | 28.54 | 50.36 |
| Back | KELLY_0.50 | 326 | 21165.61 | 363.03 | 1.72% | 87.91 | 87.91 | 1257.85 | 2497.30 |
| Lay | KELLY_0.50 | 46 | 955.17 | 131.22 | 13.74% | 20.43 | 20.43 | 20.58 | 35.45 |
| Back | KELLY_1.00 | 326 | 21856.34 | 339.52 | 1.55% | 87.91 | 87.91 | 1405.57 | 2763.11 |
| Lay | KELLY_1.00 | 46 | 970.29 | 140.27 | 14.46% | 20.43 | 20.43 | 13.07 | 22.96 |

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
| Back | Pre | Yes | FLAT | 91 | 91.00 | -0.33 | -0.36% | 1.00 | 50.63 |
| Back | Pre | Yes | PROXY | 91 | 30945.32 | -8836.63 | -28.56% | 4734.71 | 84562.33 |
| Back | Pre | Yes | KELLY_0.10 | 80 | 2252.58 | -93.70 | -4.16% | 47.98 | 1563.44 |
| Back | Pre | Yes | KELLY_0.25 | 80 | 4799.50 | -79.02 | -1.65% | 87.91 | 2031.80 |
| Back | Pre | Yes | KELLY_0.50 | 80 | 5768.20 | 17.81 | 0.31% | 87.91 | 1728.84 |
| Back | Pre | Yes | KELLY_1.00 | 80 | 5894.92 | -14.73 | -0.25% | 87.91 | 2130.98 |
| Back | Pre | No | FLAT | 302 | 302.00 | 5.16 | 1.71% | 1.00 | 45.83 |
| Back | Pre | No | PROXY | 302 | 133092.26 | -23966.37 | -18.01% | 4490.27 | 170896.62 |
| Back | Pre | No | KELLY_0.10 | 246 | 6287.84 | 533.91 | 8.49% | 57.36 | 567.49 |
| Back | Pre | No | KELLY_0.25 | 246 | 12564.93 | 578.41 | 4.60% | 87.91 | 1451.96 |
| Back | Pre | No | KELLY_0.50 | 246 | 15397.41 | 345.22 | 2.24% | 87.91 | 1864.79 |
| Back | Pre | No | KELLY_1.00 | 246 | 15961.42 | 354.26 | 2.22% | 87.91 | 1843.24 |
| Back | In | Yes | FLAT | 58 | 58.00 | -4.52 | -7.80% | 1.00 | 49.71 |
| Back | In | Yes | PROXY | 58 | 9973.72 | -3099.84 | -31.08% | 1457.62 | 34130.06 |
| Back | In | No | FLAT | 164 | 164.00 | 22.42 | 13.67% | 1.00 | 0.00 |
| Back | In | No | PROXY | 164 | 37919.51 | -7644.55 | -20.16% | 2337.25 | 62255.72 |
| Lay | Pre | Yes | FLAT | 13 | 14.38 | 3.45 | 24.02% | 1.00 | 7.00 |
| Lay | Pre | Yes | PROXY | 13 | 6138.31 | -3626.31 | -59.08% | 3925.76 | 40750.69 |
| Lay | Pre | Yes | KELLY_0.10 | 9 | 96.86 | 16.33 | 16.86% | 20.07 | 117.69 |
| Lay | Pre | Yes | KELLY_0.25 | 9 | 149.37 | 35.06 | 23.47% | 20.43 | 102.16 |
| Lay | Pre | Yes | KELLY_0.50 | 9 | 179.69 | 41.42 | 23.05% | 20.43 | 102.16 |
| Lay | Pre | Yes | KELLY_1.00 | 9 | 181.98 | 43.71 | 24.02% | 20.43 | 99.69 |
| Lay | Pre | No | FLAT | 34 | 38.88 | 0.74 | 1.91% | 1.00 | 6.81 |
| Lay | Pre | No | PROXY | 34 | 10447.02 | -3673.57 | -35.16% | 3340.84 | 31418.67 |
| Lay | Pre | No | KELLY_0.10 | 25 | 372.20 | 12.80 | 3.44% | 20.43 | 46.37 |
| Lay | Pre | No | KELLY_0.25 | 25 | 519.91 | 34.88 | 6.71% | 20.43 | 28.25 |
| Lay | Pre | No | KELLY_0.50 | 25 | 559.80 | 18.41 | 3.29% | 20.43 | 84.31 |
| Lay | Pre | No | KELLY_1.00 | 25 | 559.80 | 18.41 | 3.29% | 20.43 | 84.54 |
| Lay | In | Yes | FLAT | 26 | 31.49 | 3.39 | 10.77% | 1.00 | 5.04 |
| Lay | In | Yes | PROXY | 26 | 5812.58 | 1881.87 | 32.38% | 1713.54 | 3363.92 |
| Lay | In | No | FLAT | 64 | 104.80 | -6.32 | -6.03% | 1.00 | 67.68 |
| Lay | In | No | PROXY | 64 | 14160.59 | 429.76 | 3.03% | 1391.25 | 13640.67 |
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
| Back | Pre | Yes | 169 | 103 | +4.14% [+3.27%, +5.02%] | -3.25% [-16.32%, +9.90%] | -7.60% | pre: Kelly OK |
| Back | Pre | No | 902 | 192 | +1.78% [+1.32%, +2.24%] | -4.26% [-11.61%, +2.91%] | -6.44% | pre: Kelly OK |
| Back | In | Yes | 243 | 122 | — — | +1.86% [-12.61%, +17.04%] | -2.67% | in: use FLAT/PROXY |
| Back | In | No | 597 | 215 | — — | +3.75% [-4.27%, +11.91%] | +4.41% | in: use FLAT/PROXY |
| Lay | Pre | Yes | 189 | 99 | +0.63% [-0.31%, +1.60%] | +3.51% [-9.69%, +17.11%] | -0.66% | pre: Kelly OK |
| Lay | Pre | No | 754 | 183 | +0.19% [-0.45%, +0.82%] | +3.78% [-3.82%, +11.48%] | +1.43% | pre: Kelly OK |
| Lay | In | Yes | 263 | 138 | — — | +24.72% [+6.27%, +45.25%] | +18.20% | in: use FLAT/PROXY |
| Lay | In | No | 413 | 181 | — — | +10.59% [-2.35%, +23.86%] | +17.23% | in: use FLAT/PROXY |
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

**Escala Kelly usada nesta curva**: P99_PROXY | ref_back=4395.55 ref_lay=2043.24 | cap_back=2.0% cap_lay=1.0% | max_stake_event=100%*limit

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
| Combinações elegíveis (edge + timing + t0) | 257 |
| Com ROI disponível (precisa de placar) | 228 |
| Com CLV disponível (pre-match + closing) | 126 |

Leitura: o walk-forward mede OOS principalmente por **ROI**, então ele encolhe quando a cobertura de placar é baixa. Além disso, a métrica é agregada por **jogo único** (cluster), então você verá números menores que o N de eventos.

| Train window | Test window | #ativas | Jogos OOS | ROI OOS (mean; IC90) | Turnover (teste) | Lucro (estratégia, budget) |
|---|---|---:|---:|---:|---:|---:|
| 2026-02-11→2026-02-13 | 2026-02-14→2026-02-14 | 1 | 44 | +13.53% [-10.44%, +39.86%] | 715.84 | -52.58 |
| 2026-02-12→2026-02-14 | 2026-02-15→2026-02-15 | 3 | 56 | +15.77% [-3.07%, +34.20%] | 1126.64 | 256.67 |

**Frequência de ativação por combinação (quantas janelas ela entrou como ativa)**

| Combinação | #steps ativa |
|---|---:|
| Back_In_No | 2 |
| Back_Pre_No | 1 |
| Back_Pre_Yes | 1 |

### 12.A Transparência da seleção: métricas por combinação no treino
Para cada janela de treino, mostramos as métricas usadas para decidir se cada combinação ficou **ativa** ou não. Isso ajuda a entender, por exemplo, por que nenhuma Lay entrou em algumas janelas (geralmente N insuficiente ou ROI p30 <= 0).

**Train 2026-02-11→2026-02-13 → Test 2026-02-14→2026-02-14**

| Combinação | Ativa? | Jogos treino (tot/CLV/ROI) | CLV q10/q90 ou CI | ROI q30 | Motivo |
|---|---|---:|---:|---:|---|
| Back_Pre_Yes | NÃO | 23 / 23 / 22 | q10=6.68 | CI90_lb=6.44 | -9.22% | BackPre: clv_q10>0=False, roi_mean>0=True |
| Back_Pre_No | NÃO | 87 / 74 / 79 | q10=6.37 | CI90_lb=6.21 | -10.06% | BackPre: clv_q10>0=True, roi_mean>0=False |
| Back_In_Yes | NÃO | 15 / — / 8 | — | -58.82% | BackIn: roi_q30>0 |
| Back_In_No | SIM | 45 / — / 34 | — | +13.86% | BackIn: roi_q30>0 |
| Lay_Pre_Yes | NÃO | 5 / 4 / 5 | q90=-3.00 | CI90_ub=-2.37 | -22.79% | LayPre: clv_q90<0=False, roi_q30>0=False |
| Lay_Pre_No | NÃO | 16 / 15 / 14 | q90=-3.74 | CI90_ub=-3.13 | -31.71% | LayPre: clv_q90<0=False, roi_q30>0=False |
| Lay_In_Yes | NÃO | 9 / — / 6 | — | +22.01% | In: roi_q30>0 |
| Lay_In_No | NÃO | 25 / — / 23 | — | -14.78% | In: roi_q30>0 |

**Train 2026-02-12→2026-02-14 → Test 2026-02-15→2026-02-15**

| Combinação | Ativa? | Jogos treino (tot/CLV/ROI) | CLV q10/q90 ou CI | ROI q30 | Motivo |
|---|---|---:|---:|---:|---|
| Back_Pre_Yes | SIM | 44 / 39 / 41 | q10=6.71 | CI90_lb=6.58 | -2.48% | BackPre: clv_q10>0=True, roi_mean>0=True |
| Back_Pre_No | SIM | 91 / 83 / 82 | q10=6.19 | CI90_lb=6.06 | +5.30% | BackPre: clv_q10>0=True, roi_mean>0=True |
| Back_In_Yes | NÃO | 26 / — / 19 | — | -38.23% | BackIn: roi_q30>0 |
| Back_In_No | SIM | 72 / — / 61 | — | +6.09% | BackIn: roi_q30>0 |
| Lay_Pre_Yes | NÃO | 8 / 6 / 8 | q90=-3.76 | CI90_ub=-3.24 | -0.87% | LayPre: clv_q90<0=False, roi_q30>0=False |
| Lay_Pre_No | NÃO | 26 / 21 / 23 | q90=-4.25 | CI90_ub=-3.83 | +8.97% | LayPre: clv_q90<0=False, roi_q30>0=False |
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
| Scheme in-match (OOS) | `PROXY` |
| Expansão missing ROI | ON |
| Dias OOS usados | 2 |
| Turnover 30d (proj.) | 27637.22 |
| Turnover 30d (Pre/In) | 40847.80 / 383765.81 |
| Lucro 30d (obs.) | 2836.15 |
| Lucro 30d (obs.) Pre/In | 2681.66 / -85825.03 |
| Lucro 30d (exp.) | 3061.22 |
| Lucro 30d (exp.) Pre/In | 2681.66 / -86734.45 |
| Banca risco p99 (Back+Lay) | 1122.53 |
| Banca liquidez p99 (+buf) | 741.15 |
| Banca recomendada (max) | 1122.53 |
| ROI/banca 30d (obs.) | 252.66% |
| ROI/banca 30d (exp.) | 272.71% |
| DD 30d p95 (obs.) | 368.09 |
| DD 30d p95 (exp.) | 420.67 |

### 12.2 Governança de exposição por jogo (budget por `match_id`) — sensibilidade
Objetivo: evitar concentração quando um mesmo jogo gera muitos sinais. Simulamos um orçamento de exposição por jogo, consumido ao longo do tempo.

- **Back** consome budget em **stake**.
- **Lay** consome budget em **liability**.
- Também aplicamos um cap por sinal como fração do budget do jogo (para não gastar tudo no 1º sinal).

Importante: o budget é parametrizado como fração de uma referência de banca. Aqui usamos:
- se `--kelly-bankroll` estiver setado: essa banca explícita;
- senão: a `Banca recomendada (max)` estimada na 12.1 (baseline).

Referência de banca p/ budget: 1122.53 | budgets por jogo aplicados em stake (Back) e liability (Lay).

| Cenário | Turnover 30d | Lucro 30d (exp.) | Banca rec. (max) | ROI/banca 30d (exp.) | DD 30d p95 (exp.) |
|---|---:|---:|---:|---:|---:|
| BASELINE (sem budget) | 27637.22 | 3061.22 | 1122.53 | 272.71% | 420.67 |
| BUDGET_0.50%/0.25% cap25% | 2983.81 | 303.06 | 116.12 | 260.98% | 19.40 |
| BUDGET_1.00%/0.50% cap33% | 7502.06 | 786.02 | 295.75 | 265.77% | 94.92 |
| BUDGET_2.00%/1.00% cap50% | 20754.19 | 2529.96 | 817.85 | 309.34% | 249.41 |

Leitura:
- Se a curva com budget melhora muito (menos negativo ou mais positivo) com pouca perda de turnover, o problema era **concentração por jogo**.
- Se tudo continuar negativo, o problema é **edge OOS** (principalmente in‑match) e budget só reduz a escala da perda.

**Volume e stake médio por combinação (janela OOS, com budget padrão)**

| Combinação | #steps ativa | Jogos OOS (uniques) | N eventos OOS | Stake eq. médio | Observação |
|---|---:|---:|---:|---:|---|
| Back_In_No | 2 | 71 | 101 | 253.31 | budget reduz concentração por jogo |
| Back_Pre_No | 1 | 31 | 42 | 52.19 | budget reduz concentração por jogo |
| Back_Pre_Yes | 1 | 7 | 8 | 66.39 | budget reduz concentração por jogo |
## 10) Diagnóstico: por que o ROI pode estar zerado
ROI aqui é calculado por placar do jogo (`matches.home_score/away_score`). Se os placares não estiverem preenchidos no banco, a cobertura de ROI será 0.

| Indicador | Valor |
|---|---:|
| Jogos únicos no recorte | 343 |
| Jogos com placar disponível (home_score/away_score não nulos) | 298 |
| Jogos com status='finished' no banco | 298 |

### 10.1 Distribuição por data de kickoff (explica janela da API)
- Kickoff (UTC) no recorte: **2026-02-11 15:00 UTC** até **2026-02-17 21:00 UTC**.

| Kickoff date (UTC) | Jogos | Com placar | Cobertura |
|---|---:|---:|---:|
| 2026-02-17 | 46 | 46 | 100.0% |
| 2026-02-16 | 24 | 19 | 79.2% |
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
