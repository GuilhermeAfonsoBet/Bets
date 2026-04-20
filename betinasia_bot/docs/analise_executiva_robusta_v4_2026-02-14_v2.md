# Analise Executiva Robusta v4.0-api (versao v2)

Data de geracao: 2026-02-14 09:56:17 UTC  
Fonte: `hypothesis_performance_robust.sh --lookback-days 14 --audit-version v4.0-api`

---

## 1) Escopo da amostra (objetivo e periodo analisado)

Esta analise **nao usa toda a base historica**.  
Ela usa:

- `audit_version = v4.0-api`
- `hypothesis_type = H3B`
- janela movel de **14 dias** a partir do horario da execucao.

Logo, o periodo da extração e:
- inicio: **2026-01-31 09:56:17 UTC**
- fim: **2026-02-14 09:56:17 UTC**

Nessa janela:
- `n_total = 2873`
- `n_ok = 2651`
- `ok_pct = 92.3%`

---

## 2) Glossario executivo (com formula)

### 2.1 Variaveis de performance operacional

- **ok_pct**
  - Formula: `ok_pct = n_ok / n_total * 100`
  - Leitura: percentual de auditorias com extração valida no betslip.

- **n_back_edge**
  - Definicao: eventos com `status='OK'` e `diff_pct >= +2%`.
  - Leitura: coorte candidata para Back (BS > WS).

- **n_lay_edge**
  - Definicao: eventos com `status='OK'` e `diff_pct <= -2%`.
  - Leitura: coorte candidata para Lay (BS < WS).

- **lay_t0_cov_pct (cobertura lay T+0)**
  - Formula: `count(lay_t0_presente) / n_total * 100`
  - Onde `lay_t0_presente := hypothesis_details.lay != null`
  - Leitura: quanto da base ja tem captura de lay no instante T+0.

- **lay_temporal_cov_pct (cobertura lay temporal)**
  - Formula: `count(lay_temporal_presente) / n_total * 100`
  - Onde `lay_temporal_presente := hypothesis_details.lay_temporal != null`
  - Leitura: quanto da base tem trilha temporal de lay apos o T+0.

- **finance_cov_pct (cobertura bloco finance)**
  - Formula: `count(finance_presente) / n_total * 100`
  - Onde `finance_presente := hypothesis_details.finance != null`
  - Leitura: percentual com insumos financeiros derivados salvos no audit.

### 2.2 Variaveis de preco e execucao

- **diff_pct**
  - Formula: `diff_pct = (odd_betslip - odd_websocket) / odd_websocket * 100`
  - Leitura:
    - `diff_pct > 0`: betslip melhor que websocket (favoravel para Back)
    - `diff_pct < 0`: betslip pior que websocket (potencial para Lay, dependendo da estrategia)

- **q_avg_ms / q_p95_ms**
  - Definicao: media e p95 de `queue_wait_ms` da telemetria.
  - Leitura: saturacao de fila no caminho critico T+0.

- **pipe_avg_ms / pipe_p95_ms**
  - Definicao: media e p95 de `pipeline_total_ms`.
  - Leitura: latencia fim-a-fim da auditoria.

### 2.3 Variaveis financeiras derivadas (insumo para analise posterior)

- **stake_est_avg**
  - Formula base (fallback): `stake_est = fallback_stake_pct * limite`
  - Nesta rodada: `fallback_stake_pct = 0.25`
  - Leitura: stake de simulacao para medir escala economica.

- **profit_if_win_avg (Back)**
  - Formula: `profit_if_win = stake * (odd - 1)`
  - Leitura: ganho potencial medio por aposta vencedora (na politica de stake escolhida).

- **liability (Lay)**
  - Formula: `liability = stake * (odd_lay - 1)`
  - Leitura: perda maxima por aposta Lay se o desfecho contrario ocorrer.

- **ES95 (Expected Shortfall 95%)**
  - Formula conceitual: media das perdas no pior 5% da distribuicao de perdas.
  - Leitura: risco de cauda (mais informativo que media quando ha extremos).

---

## 3) Performance do robo (v4.0-api)

| Metrica | Valor |
|---|---:|
| n_total | 2873 |
| n_ok | 2651 |
| ok_pct | 92.3% |
| n_fail | 222 |
| n_back_edge (diff >= +2%) | 1014 |
| n_lay_edge (diff <= -2%) | 342 |
| lay_t0_cov_pct | 45.1% |
| lay_temporal_cov_pct | 29.1% |
| finance_cov_pct | 9.2% |

Leitura:
- Consistencia operacional boa para fase de validacao.
- Coberturas lay/finance no agregado de 14d ainda misturam periodos anteriores ao deploy mais novo.

---

## 4) Back e Lay: leitura economica clara

## 4.1 Back (BS >> WS)

| Metrica | Valor |
|---|---:|
| N | 1014 |
| diff medio (%) | +45.31 |
| diff p50 (%) | +10.87 |
| diff p90 (%) | +32.65 |
| odd media | 3.506 |
| limite medio | 648.33 |
| stake estimada media | 162.08 |
| lucro potencial medio se vencer | 193.05 |

Como interpretar:
- `diff` e o descolamento percentual entre preco de execucao e preco WS.
- `stake estimada media` e uma **premissa de simulacao** (25% do limite), nao stake real executada.
- `lucro potencial medio` tambem e cenarial (se a aposta vencedora, pela premissa de stake).

## 4.2 Lay (BS << WS) com risco de cauda

| Metrica | Valor |
|---|---:|
| N | 105 |
| diff medio (%) | -9.72 |
| diff p50 (%) | -4.00 |
| diff p10 (%) | -22.81 |
| odd lay media | 1.856 |
| limite lay medio | 1013.73 |
| stake lay media | 207.57 |
| liability media | 160.40 |
| liability p95 | 413.92 |
| liability p99 | 3561.36 |
| liability max | 4386.23 |

Leitura:
- O risco do Lay nao deve ser lido pela media; deve ser lido pelos extremos (p95/p99/ES95).
- A relacao media `liability/stake = 0.86` esconde cauda muito pesada em poucos casos.

---

## 5) Inferencia estatistica de valor (coortes de preco)

| Coorte | N | Media diff (%) | IC95 | t_stat_vs_0 | Significativo 95% |
|---|---:|---:|---|---:|---|
| BACK_EDGE | 1014 | +45.306 | [18.570, 72.043] | +3.32 | YES |
| LAY_EDGE | 342 | -11.128 | [-12.717, -9.539] | -13.72 | YES |

Conclusao estatistica desta secao:
- Existe sinal estatistico em `diff_pct` para Back e Lay nas coortes definidas.
- Isso e evidencia de **descolamento de preco**, nao prova final de ROI realizado.

---

## 6) Risco de cauda Lay (gestao de banca)

| Metrica | Valor |
|---|---:|
| single_liability_p95 | 438.23 |
| single_liability_p99 | 3772.79 |
| single_liability_es95 | 2131.85 |
| bucket_liability_avg (5m) | 240.60 |
| bucket_liability_p95 (5m) | 843.89 |
| bucket_liability_p99 (5m) | 3888.26 |
| bucket_liability_max (5m) | 4386.23 |

Interpretacao:
- O principal risco economico hoje esta na cauda do Lay.
- Recomendado: limite por aposta + limite agregado por bucket de tempo.

---

## 7) Inferencia de resultado realizado (profit_loss e CLV)

| Tabela | N total | N P/L | Cob. P/L | P/L medio | IC95 P/L | N CLV | Cob. CLV | CLV medio (%) | IC95 CLV | Win rate % | IC95 Win rate |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| h1_pricing_events | 12507 | 447 | 3.6% | +0.4329 | [0.3585, 0.5074] | 447 | 3.6% | -0.4677 | [-1.3601, 0.4246] | 72.26 | [68.11, 76.41] |
| h3_line_monotonicity_events | 626 | 50 | 8.0% | -0.0074 | [-0.2683, 0.2534] | 50 | 8.0% | +6.3414 | [-3.7893, 16.4721] | 52.00 | [38.15, 65.85] |
| h3b_temporal_reversal_events | 21135 | 349 | 1.7% | -0.1042 | [-0.1989, -0.0096] | 693 | 3.3% | +150.5592 | [41.1827, 259.9357] | 41.26 | [36.10, 46.43] |
| h6_correlation_lag_events | 7850 | 784 | 10.0% | -0.0950 | [-0.1600, -0.0301] | 784 | 10.0% | +0.0682 | [-0.3922, 0.5286] | 47.96 | [44.46, 51.46] |

Ponto tecnico essencial sobre `profit_loss`:
- `profit_loss` e metrica de resultado **liquidado** (precisa jogo encerrado e atualizacao de resultado).
- Portanto, `n_pl` e a amostra efetiva de inferencia economica realizada.
- Se `pl_cov_pct` e baixo, os ICs de ROI/lucro/drawdown realizado ficam menos estaveis.

---

## 8) Combinações dentro do H3B (o que faltava e como ficou)

Sua critica foi correta: a versao anterior nao trazia inferencia por combinacoes internas do H3B no mesmo nivel dos reportes antigos.

Ajuste implementado no script:
- foram adicionadas secoes de combinacoes inferenciais:
  - **Back:** `regime x faixa AH x bucket de diff`, com `N`, media, IC95, t-stat, significancia;
  - **Lay:** mesmo desenho + `liability_avg` e `liability_p95`.
- parametro novo: `--combo-min-n` (default 20) para evitar ruido em buckets pequenos.

Comando:
- `bash hypothesis_performance_robust.sh --lookback-days 14 --audit-version v4.0-api --combo-min-n 20`

---

## 9) Conclusao executiva

1. O robo v4.0-api esta operacionalmente consistente no periodo.
2. Ha evidencia estatistica de sinal de preco para Back e Lay nas coortes definidas.
3. O Lay exige controle de risco de cauda obrigatorio (p95/p99/ES95 altos).
4. Para decisao final de escala economica (ROI/drawdown realizado), a qualidade depende de ampliar cobertura de `profit_loss` liquidado.
5. A camada de combinacoes inferenciais dentro do H3B foi incorporada ao script para aproximar o padrao dos reportes anteriores.

