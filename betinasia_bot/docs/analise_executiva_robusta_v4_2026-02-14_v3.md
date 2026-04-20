# Analise Executiva Robusta v4.0-api (versao v3, com combinacoes H3B)

Data da execucao-base: 2026-02-14 10:17:36 UTC  
Fonte principal: `hypothesis_performance_robust.sh --lookback-days 14 --audit-version v4.0-api --combo-min-n 20`

---

## 1) Escopo da amostra analisada

Esta analise usa:
- **somente** `audit_version = v4.0-api`;
- janela movel de **14 dias**;
- foco em `hypothesis_type = H3B` na camada de auditoria API.

Portanto, nao e toda a base historica completa; e um recorte temporal + de versao.

Resumo do recorte:
- `n_total = 2877`
- `n_ok = 2654`
- `ok_pct = 92.2%`

---

## 2) Glossario executivo com formula (variaveis principais)

### 2.1 Coberturas de dados

- **Cobertura Lay T+0 (`lay_t0_cov_pct`)**
  - Formula: `count(hypothesis_details.lay != null) / n_total * 100`
  - Valor no recorte: **45.2%**

- **Cobertura Lay temporal (`lay_temporal_cov_pct`)**
  - Formula: `count(hypothesis_details.lay_temporal != null) / n_total * 100`
  - Valor no recorte: **29.2%**

- **Cobertura bloco finance (`finance_cov_pct`)**
  - Formula: `count(hypothesis_details.finance != null) / n_total * 100`
  - Valor no recorte: **9.4%**
  - Observacao: no agregado de 14 dias mistura periodos antes/depois da ativacao do bloco.

### 2.2 Variavel de preco

- **`diff_pct`**
  - Formula: `diff_pct = (odd_betslip - odd_websocket) / odd_websocket * 100`
  - Leitura:
    - `diff_pct > 0`: betslip acima do websocket (candidato Back BS >> WS)
    - `diff_pct < 0`: betslip abaixo do websocket (candidato Lay BS << WS)

### 2.3 Variaveis economicas derivadas

- **Stake estimada (`stake_est`)**
  - Regra usada no script (fallback): `stake_est = 0.25 * limite`
  - Se existir `hypothesis_details.finance`, o script usa a stake salva ali.

- **Lucro potencial medio (Back, se vencer)**
  - Formula: `profit_if_win = stake * (odd - 1)`
  - E **cenario de potencial**, nao P/L realizado.

- **Liability (Lay, se perder)**
  - Formula: `liability = stake * (odd_lay - 1)`
  - Variavel-chave de risco de cauda no Lay.

---

## 3) Performance operacional do robo (v4.0-api)

| Metrica | Valor |
|---|---:|
| N total | 2877 |
| N OK | 2654 |
| OK % | 92.2% |
| N fail | 223 |
| N Back edge (`diff >= +2%`) | 1015 |
| N Lay edge (`diff <= -2%`) | 342 |

Leitura:
- Operacao consistente para fase de validacao.
- O funil de oportunidades ainda e mais abundante em Back que Lay.

---

## 4) Coortes de valor (Back vs Lay)

## 4.1 Back (BS >> WS)

| Metrica | Valor |
|---|---:|
| N | 1015 |
| diff medio (%) | +45.27 |
| diff p50 (%) | +10.87 |
| diff p90 (%) | +32.65 |
| odd media | 3.505 |
| limite medio | 648.31 |
| stake estimada media | 162.08 |
| lucro potencial medio se vencer | 193.01 |

Interpretacao:
- O `diff` medio esta inflado por cauda (outliers); mediana e percentis sao mais estaveis para leitura operacional.
- `stake estimada` e `lucro potencial` sao metricas de simulacao economica (nao resultado realizado).

## 4.2 Lay (BS << WS)

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

Interpretacao:
- Lay e naturalmente assimetrico: media de risco nao captura a cauda.
- Decisao de alocacao em Lay precisa ser guiada por p95/p99/ES95 e limite de exposicao por janela.

---

## 5) Risco de cauda Lay (core de risco)

| Metrica | Valor |
|---|---:|
| single liability p95 | 438.23 |
| single liability p99 | 3772.79 |
| single liability ES95 | 2131.85 |
| bucket liability p95 (5m) | 843.89 |
| bucket liability p99 (5m) | 3888.26 |
| bucket liability max (5m) | 4386.23 |

Leitura:
- O ES95 alto confirma que os piores 5% de eventos sao economicamente pesados.
- Exposicao agregada em bucket (5m) pode escalar risco rapidamente.

---

## 6) Inferencia estatistica de preco (coortes)

| Coorte | N | Media diff (%) | IC95 | t-stat vs 0 | Significativo 95% |
|---|---:|---:|---|---:|---|
| BACK_EDGE | 1015 | +45.268 | [18.558, 71.978] | +3.32 | YES |
| LAY_EDGE | 342 | -11.128 | [-12.717, -9.539] | -13.72 | YES |

Conclusao:
- As duas coortes apresentam sinal estatistico de preco no recorte atual.
- Isso valida sinal de descolamento, mas nao substitui a inferencia de resultado realizado.

---

## 7) Inferencia de resultado realizado (profit_loss e CLV)

| Tabela | N total | N P/L | Cob. P/L | P/L medio | IC95 P/L | N CLV | Cob. CLV | CLV medio (%) | IC95 CLV | Win rate % | IC95 Win rate |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| h1_pricing_events | 12507 | 447 | 3.6% | +0.4329 | [0.3585, 0.5074] | 447 | 3.6% | -0.4677 | [-1.3601, 0.4246] | 72.26 | [68.11, 76.41] |
| h3_line_monotonicity_events | 626 | 50 | 8.0% | -0.0074 | [-0.2683, 0.2534] | 50 | 8.0% | +6.3414 | [-3.7893, 16.4721] | 52.00 | [38.15, 65.85] |
| h3b_temporal_reversal_events | 21135 | 349 | 1.7% | -0.1042 | [-0.1989, -0.0096] | 693 | 3.3% | +150.5592 | [41.1827, 259.9357] | 41.26 | [36.10, 46.43] |
| h6_correlation_lag_events | 7850 | 784 | 10.0% | -0.0950 | [-0.1600, -0.0301] | 784 | 10.0% | +0.0682 | [-0.3922, 0.5286] | 47.96 | [44.46, 51.46] |

Ponto critico:
- A inferencia de P/L depende de `n_pl` (amostra liquidada), nao de `n_total`.
- Em H3B, `pl_cov_pct = 1.7%`: a amostra efetiva para P/L ainda e limitada para conclusoes finais de escala.

---

## 8) Combinacoes H3B (regime x faixa AH x bucket de diff)

### 8.1 Back combinacional (com inferencia)

Principais combinacoes com `N >= 20`:

| Regime | Faixa AH | Bucket diff | N | Media diff (%) | IC95 | t-stat | Sig95 |
|---|---|---|---:|---:|---|---:|---|
| PRE_MATCH | AH_0_1 | B2 [5,10) | 108 | 7.453 | [7.181, 7.724] | 53.74 | YES |
| PRE_MATCH | AH_0_1 | B3 [10,20) | 86 | 13.888 | [13.228, 14.548] | 41.26 | YES |
| IN_MATCH | AH_4_PLUS | B3 [10,20) | 54 | 14.684 | [13.995, 15.374] | 41.75 | YES |
| PRE_MATCH | AH_4_PLUS | B3 [10,20) | 50 | 15.038 | [14.370, 15.706] | 44.14 | YES |
| IN_MATCH | AH_2_4 | B4 [20,+) | 49 | 126.919 | [52.620, 201.219] | 3.35 | YES |
| IN_MATCH | AH_4_PLUS | B4 [20,+) | 48 | 514.412 | [-28.621, 1057.445] | 1.86 | NO |

Leitura:
- A maioria das combinacoes Back com N suficiente segue estatisticamente positiva.
- Buckets extremos (`B4 [20,+)`) mostram cauda forte e intervalos largos (maior instabilidade).

### 8.2 Lay combinacional

No recorte executado com `--combo-min-n 20`, a secao veio vazia (`0 rows`), indicando que:
- para Lay, N por combinacao e mais esparso nessa granularidade;
- exige limiar menor para leitura combinacional.

Ajuste aplicado no script:
- novo parametro dedicado: `--combo-min-n-lay` (default 8).
- recomendado para Lay: rodar com `--combo-min-n-lay 8` e validar robustez por bloco.

---

## 9) Diagnostico executivo final

1. **Robo v4.0-api**: consistente operacionalmente (92.2% OK).  
2. **Sinal de preco**: Back e Lay com evidencia estatistica no `diff_pct`.  
3. **Lay**: risco de cauda elevado (p99/ES95), exigindo governanca de exposicao.  
4. **Resultado realizado**: ainda limitado por cobertura de liquidacao em algumas hipoteses (especialmente H3B).  
5. **Combinacoes H3B**: Back robustecido com inferencia; Lay requer limiar combinacional menor e monitor dedicado por cauda.

---

## 10) Recomendacoes praticas de proxima rodada

- Manter separacao metodologica Back vs Lay em todos os reportes.
- Para Lay, incluir sempre:
  - p95/p99/ES95 de liability,
  - exposicao por bucket temporal,
  - limite operacional por aposta e por janela.
- Para inferencia de P/L realizado:
  - elevar cobertura `n_pl` antes de qualquer decisao de escala agressiva.
- Para combinacoes:
  - Back: manter `combo_min_n` mais alto;
  - Lay: usar `combo_min_n_lay` menor (ex.: 8) e exigir leitura de cauda no mesmo bloco.

