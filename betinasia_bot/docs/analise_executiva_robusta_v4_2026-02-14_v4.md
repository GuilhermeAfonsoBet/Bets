# Analise Executiva Robusta v4.0-api (versao v4 final)

Data da execucao-base: 2026-02-14 10:27:20 UTC  
Fonte principal: `hypothesis_performance_robust.sh --lookback-days 14 --audit-version v4.0-api --combo-min-n 20 --combo-min-n-lay 8`

---

## Sumario executivo (1 pagina)

- Recorte de 14 dias em `v4.0-api` mostra operacao estavel: **92.2% OK** (`2654/2877`).
- Sinal de preco permanece estatisticamente significativo para as duas coortes:
  - **Back edge**: media `+45.268%`, IC95 `[18.558, 71.978]`, `t=3.32`, `sig_95=YES`.
  - **Lay edge**: media `-11.128%`, IC95 `[-12.717, -9.539]`, `t=-13.72`, `sig_95=YES`.
- No Lay, o risco de cauda e elevado:
  - `liability p95=438.23`, `p99=3772.79`, `ES95=2131.85`, `max=4386.23`.
- O monitor de fila/latencia total confirma gargalos episodicos:
  - `queue_wait_avg=19122 ms`, `p95=90673 ms`, `p99=524017 ms`;
  - `queue_depth max=22`;
  - `total_bot_avg=4511 ms`, `p95=6921 ms`, `p99=23703 ms`.
- Cobertura de resultado realizado ainda baixa para algumas hipoteses (especialmente H3B com `pl_cov=1.7%`), o que limita confianca para conclusao economica final de escala.

---

## 1) Escopo da amostra analisada

Esta analise usa:
- somente `audit_version = v4.0-api`;
- janela movel de **14 dias**;
- foco em `hypothesis_type = H3B` na auditoria API.

Nao e a base historica integral: e um recorte temporal + de versao.

Resumo do recorte:

| Metrica | Valor |
|---|---:|
| n_total | 2877 |
| n_ok | 2654 |
| ok_pct | 92.2% |
| n_fail | 223 |
| n_back_edge (`diff >= +2%`) | 1015 |
| n_lay_edge (`diff <= -2%`) | 342 |
| lay_t0_cov_pct | 45.2% |
| lay_temporal_cov_pct | 29.2% |
| finance_cov_pct | 9.4% |

---

## 2) Glossario executivo com formula

### 2.1 Variavel de preco

- `diff_pct`
  - Formula: `diff_pct = (odd_betslip - odd_websocket) / odd_websocket * 100`
  - Leitura:
    - `diff_pct > 0`: candidato Back (BS >> WS)
    - `diff_pct < 0`: candidato Lay (BS << WS)

### 2.2 Coberturas

- `lay_t0_cov_pct = count(lay_t0 nao nulo) / n_total * 100`
- `lay_temporal_cov_pct = count(lay_temporal nao nulo) / n_total * 100`
- `finance_cov_pct = count(finance nao nulo) / n_total * 100`

### 2.3 Variaveis economicas

- `stake_est` (estimada)
  - fallback: `stake_est = 0.25 * limit`
  - quando houver bloco `finance`, usar stake salva no proprio evento.

- `profit_if_win` (Back, potencial)
  - `profit_if_win = stake * (odd - 1)`
  - nao e P/L realizado; e cenario teorico por aposta.

- `liability` (Lay, perda se contra-evento ocorrer)
  - `liability = stake * (odd_lay - 1)`
  - principal variavel de risco de cauda no Lay.

- `ES95` (Expected Shortfall 95%)
  - media da cauda pior que o percentil 95.
  - em termos praticos: perda media dos 5% piores casos.

---

## 3) Performance do robo (fila + tempo total)

| Metrica | Valor |
|---|---:|
| n_ok | 2654 |
| queue_wait_avg_ms | 19122.3 |
| queue_wait_p95_ms | 90672.6 |
| queue_wait_p99_ms | 524017.2 |
| queue_depth_enq_avg | 0.96 |
| queue_depth_enq_p95 | 8.10 |
| queue_depth_enq_max | 22.00 |
| queue_depth_deq_avg | 0.96 |
| queue_depth_deq_p95 | 10.00 |
| queue_depth_deq_max | 22.00 |
| total_bot_avg_ms | 4511.0 |
| total_bot_p95_ms | 6921.2 |
| total_bot_p99_ms | 23703.3 |

Leitura:
- Ha eventos com espera de fila muito longa (cauda operacional), apesar de media de processamento bot em faixa de segundos.
- O monitoramento de profundidade (p95 e max) indica picos pontuais de backlog.

---

## 4) BS vs WS por bucket (inclui faixa neutra)

| Bucket diff | N | Share % | N pre | N in | diff_avg % | diff_p50 % | diff_p95 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| A `<= -10%` | 98 | 3.7 | 25 | 73 | -27.27 | -16.74 | -10.39 |
| B `(-10,-2]%` | 244 | 9.2 | 120 | 124 | -4.64 | -3.86 | -2.12 |
| C `(-2,+2)%` | 1297 | 48.9 | 1006 | 291 | -0.17 | -0.20 | +1.17 |
| D `> +2%` | 1015 | 38.2 | 563 | 452 | +45.27 | +10.87 | +71.24 |

Leitura:
- A analise contempla explicitamente as faixas pedidas:
  - `(-10,-2]`, `(-2,+2)` e `>+2` (alem de `<=-10` para cauda).
- Quase metade dos eventos esta na faixa neutra (`-2` a `+2`), o que ajuda a calibrar filtros de entrada.

---

## 5) Coortes Back e Lay (economia e risco)

### 5.1 Back (BS >> WS)

| Metrica | Valor |
|---|---:|
| N | 1015 |
| diff_avg_pct | +45.27 |
| diff_p50_pct | +10.87 |
| diff_p90_pct | +32.65 |
| odd_avg | 3.505 |
| limit_avg | 648.31 |
| stake_est_avg | 162.08 |
| profit_if_win_avg | 193.01 |

### 5.2 Lay (BS << WS) com foco em cauda

| Metrica | Valor |
|---|---:|
| N | 105 |
| diff_avg_pct | -9.72 |
| diff_p50_pct | -4.00 |
| diff_p10_pct | -22.81 |
| lay_odd_avg | 1.856 |
| lay_limit_avg | 1013.73 |
| lay_stake_avg | 207.57 |
| liability_avg | 160.40 |
| liability_p90 | 202.78 |
| liability_p95 | 413.92 |
| liability_p99 | 3561.36 |
| liability_max | 4386.23 |
| liab_to_stake_avg | 0.86 |

Leitura:
- Back e Lay exigem governanca separada.
- No Lay, media de liability sozinha subestima risco; p95/p99/ES95 sao obrigatorios para decisao.

---

## 6) Risco de cauda Lay e exposicao agregada

| Metrica | Valor |
|---|---:|
| single_liability_p95 | 438.23 |
| single_liability_p99 | 3772.79 |
| single_liability_es95 | 2131.85 |
| bucket_liability_avg | 240.60 |
| bucket_liability_p95 | 843.89 |
| bucket_liability_p99 | 3888.26 |
| bucket_liability_max | 4386.23 |

Top buckets de exposicao (5m) mostram picos isolados com alta concentracao de risco.

---

## 7) Evolucao temporal das odds e descoberta de valor

### 7.1 Back temporal (T+0 -> ultimo ponto)

| Regime | N | diff_t0_avg % | diff_tlast_avg % | delta_avg % | IC95 delta | retention % | loss % | decay % | stable % | improve % |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| IN_MATCH | 301 | 40.89 | 42.56 | +1.67 | [-11.11, 14.44] | 57.8 | 0.7 | 30.9 | 38.2 | 30.9 |
| PRE_MATCH | 690 | 5.91 | 6.22 | +0.31 | [0.13, 0.49] | 50.4 | 0.1 | 7.5 | 75.9 | 16.5 |

### 7.2 Lay temporal (T+0 -> ultimo ponto)

| Regime | N | diff_t0_avg % | diff_tlast_avg % | delta_avg % | IC95 delta | retention % | loss % | gain % | decay_against % | stable % | improve_favor % |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| IN_MATCH | 268 | 33.97 | 42.48 | +8.51 | [-11.39, 28.40] | 20.1 | 2.2 | 3.0 | 28.4 | 37.3 | 34.3 |
| PRE_MATCH | 571 | 1.77 | 1.39 | -0.37 | [-0.51, -0.24] | 31.5 | 0.9 | 4.4 | 6.8 | 72.9 | 20.3 |

Leitura de padrao:
- Em Back PRE_MATCH ha leve melhora media com alta estabilidade.
- Em Lay PRE_MATCH o delta medio e ligeiramente desfavoravel (IC95 totalmente negativo).
- Em regimes IN_MATCH, os intervalos sao mais amplos, sinal de maior volatilidade.

---

## 8) Inferencia estatistica de preco (audit diff_pct)

| Coorte | N | mean_diff_pct | sd_diff_pct | se_diff_pct | IC95 | t_stat_vs_0 | sig_95 |
|---|---:|---:|---:|---:|---|---:|---|
| BACK_EDGE | 1015 | +45.268 | 434.162 | 13.628 | [18.558, 71.978] | +3.32 | YES |
| LAY_EDGE | 342 | -11.128 | 14.994 | 0.811 | [-12.717, -9.539] | -13.72 | YES |

Conclusao:
- Ha evidencia estatistica de descolamento de preco para as duas coortes no recorte.
- A variancia de Back e alta (cauda forte), portanto mediana/percentis devem acompanhar a media.

---

## 9) Inferencia de resultado realizado (P/L e CLV)

| Tabela | pl_cov % | pl_mean_u | IC95 P/L | clv_cov % | clv_mean % | IC95 CLV | win_rate % | IC95 win_rate |
|---|---:|---:|---|---:|---:|---|---:|---|
| h1_pricing_events | 3.6 | +0.4329 | [0.3585, 0.5074] | 3.6 | -0.4677 | [-1.3601, 0.4246] | 72.26 | [68.11, 76.41] |
| h3_line_monotonicity_events | 8.0 | -0.0074 | [-0.2683, 0.2534] | 8.0 | +6.3414 | [-3.7893, 16.4721] | 52.00 | [38.15, 65.85] |
| h3b_temporal_reversal_events | 1.7 | -0.1042 | [-0.1989, -0.0096] | 3.3 | +150.5592 | [41.1827, 259.9357] | 41.26 | [36.10, 46.43] |
| h6_correlation_lag_events | 10.0 | -0.0950 | [-0.1600, -0.0301] | 10.0 | +0.0682 | [-0.3922, 0.5286] | 47.96 | [44.46, 51.46] |

Leitura critica:
- A amostra efetiva para P/L realizado em H3B ainda e pequena (`pl_cov=1.7%`).
- Portanto, a inferencia economica final de escala (ROI/drawdown robusto) ainda exige mais observacoes liquidadas.

---

## 10) Combinacoes H3B com inferencia

### 10.1 Back combinacional (regime x AH x bucket diff)

Principais combinacoes com maior N:

| Regime | AH | Bucket | N | mean_diff % | IC95 | t_stat | sig95 |
|---|---|---|---:|---:|---|---:|---|
| PRE_MATCH | AH_0_1 | B2_[5,10) | 108 | 7.453 | [7.181, 7.724] | 53.74 | YES |
| PRE_MATCH | AH_0_1 | B3_[10,20) | 86 | 13.888 | [13.228, 14.548] | 41.26 | YES |
| IN_MATCH | AH_4_PLUS | B3_[10,20) | 54 | 14.684 | [13.995, 15.374] | 41.75 | YES |
| PRE_MATCH | AH_4_PLUS | B3_[10,20) | 50 | 15.038 | [14.370, 15.706] | 44.14 | YES |
| IN_MATCH | AH_2_4 | B4_[20,+) | 49 | 126.919 | [52.620, 201.219] | 3.35 | YES |
| IN_MATCH | AH_4_PLUS | B4_[20,+) | 48 | 514.412 | [-28.621, 1057.445] | 1.86 | NO |

### 10.2 Lay combinacional (regime x AH x bucket diff)

Com `--combo-min-n-lay 8`, surgiram combinacoes validas:

| Regime | AH | Bucket | N | mean_diff % | IC95 | t_stat | sig95 | liability_avg | liability_p95 |
|---|---|---|---:|---:|---|---:|---|---:|---:|
| PRE_MATCH | AH_0_1 | L1_(-5,-2] | 12 | -3.231 | [-3.649, -2.813] | -15.15 | YES | 425.07 | 2215.44 |
| IN_MATCH | AH_0_1 | L1_(-5,-2] | 8 | -3.031 | [-3.495, -2.567] | -12.80 | YES | 125.15 | 279.42 |

Leitura:
- O Lay combinacional apareceu somente no bucket mais proximo do corte (`-5` a `-2`), com N menor.
- Mesmo com sinal estatistico, a dispersao de liability exige filtro de risco junto da inferencia.

---

## 11) Diagnostico final e recomendacoes

1. Operacao v4.0-api esta funcional para fase de validacao, com nivel de fail atual administravel.
2. O sinal de preco Back e Lay esta presente e estatisticamente detectavel.
3. Lay deve ser governado por risco de cauda (p95/p99/ES95 + limite por janela), nao apenas por media de edge.
4. A secao de fila/tempo total mostra gargalos pontuais; monitor continuo deve permanecer ativo para reduzir latencias extremas.
5. Para conclusao economica robusta (ROI, drawdown, banca), o principal gargalo e elevar cobertura de liquidacao (`n_pl`) nas hipoteses, sobretudo H3B.

---

## 12) Checklist de aderencia ao pedido

- Glossario com formulas: **OK**.
- Definicao clara de amostra (periodo/versao): **OK**.
- Explicacao de `diff`, `stake_est`, `profit_if_win`: **OK**.
- Inferencia robusta com IC e t-stat, incluindo combinacoes H3B: **OK**.
- Buckets BS vs WS incluindo faixa neutra e cortes relevantes: **OK**.
- Evolucao temporal e padroes de valor/nao valor: **OK**.
- Performance com fila e tempo total do bot: **OK**.

