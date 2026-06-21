# Analise Executiva Robusta - v4.0-api

Data da analise: 2026-02-14  
Fonte: `hypothesis_performance_robust.sh --lookback-days 14 --audit-version v4.0-api`  
Escopo: performance do robo + inferencia estatistica de oportunidades de valor (Back e Lay)

---

## 1) Sumario executivo (direto ao ponto)

- **Operacao do robo (v4.0-api) esta funcional e consistente** no periodo: `N=2873`, `OK=2651`, `ok_pct=92.3%`.
- **Back (BS >> WS)** e **Lay (BS << WS)** mostraram sinal estatistico em `diff_pct` (secao inferencial), mas isso ainda e **sinal de preco**, nao prova final de lucro realizado.
- **Risco de cauda do Lay e materialmente alto**: `liability_p99=3561.36` e `liability_max=4386.23`.
- Na camada de resultado realizado (`profit_loss`), o quadro e misto:
  - `H1`: positivo e significativo no recorte liquidado;
  - `H3`: inconclusivo;
  - `H3B`: negativo no P/L realizado (mesmo com CLV medio alto);
  - `H6`: negativo no P/L realizado.
- **Conclusao executiva**: ha base para continuar validacao de valor, mas com foco em:
  1) governanca de risco para Lay (cauda);
  2) ampliar cobertura de liquidacao (`n_pl`) antes de decisao forte de alocacao.

---

## 2) Por que essas analises ficaram instantaneas agora?

Essas analises ficaram muito mais rapidas porque o fluxo atual usa **consultas SQL agregadas** direto nas tabelas auditadas (com filtros por versao/janela), sem pipeline longo de sincronizacao/log parsing.

Em resumo:
- antes: cadeia de scripts + reconciliacao + etapas de enrich em lote (mais lenta);
- agora: leitura agregada direta de dados ja persistidos (`betslip_audit_results` e tabelas de hipoteses), com metodos estatisticos aplicados no proprio SQL.

---

## 3) Performance operacional do robo (v4.0-api)

| Metrica | Valor |
|---|---:|
| N total auditorias | 2873 |
| N OK | 2651 |
| OK % | 92.3% |
| N fail | 222 |
| N Back edge (`diff >= +2%`) | 1014 |
| N Lay edge (`diff <= -2%`) | 342 |
| Cobertura lay T+0 | 45.1% |
| Cobertura lay temporal | 29.1% |
| Cobertura bloco finance | 9.2% |

### Leitura
- Taxa operacional esta boa para fase de validacao (`92.3% OK`).
- Back teve amostra maior que Lay (normal pelo corte utilizado).
- Cobertura `finance` ainda baixa no agregado de 14 dias por mistura de periodos anteriores; no recorte recente essa cobertura tem melhorado.

---

## 4) Back vs Lay: estatistica descritiva de exposicao e potencial

### 4.1 Back (coorte BS >> WS)

| Metrica | Valor |
|---|---:|
| N | 1014 |
| diff medio (%) | 45.31 |
| diff p50 (%) | 10.87 |
| diff p90 (%) | 32.65 |
| odd media | 3.506 |
| limite medio | 648.33 |
| limite p90 | 1127.19 |
| stake estimada media | 162.08 |
| stake estimada p90 | 281.80 |
| lucro potencial medio se vencer | 193.05 |

### 4.2 Lay (coorte BS << WS)

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
| liability p90 | 202.78 |
| liability p95 | 413.92 |
| liability p99 | 3561.36 |
| liability maxima | 4386.23 |
| relacao liability/stake media | 0.86 |

### Leitura
- Back e Lay sao estruturalmente diferentes: no Lay, a variavel critica e **liability**.
- A media de Lay nao representa bem o risco: a cauda (`p99`, max) domina o risco de capital.

---

## 5) Risco de cauda no Lay (ponto central para gestao de banca)

| Metrica de cauda | Valor |
|---|---:|
| Single liability p95 | 438.23 |
| Single liability p99 | 3772.79 |
| Single liability ES95 | 2131.85 |
| Bucket liability media (5m) | 240.60 |
| Bucket liability p95 (5m) | 843.89 |
| Bucket liability p99 (5m) | 3888.26 |
| Bucket liability max (5m) | 4386.23 |

### Leitura executiva
- **ES95=2131.85** significa que, no pior 5% dos casos de Lay, a perda media potencial por posicao e muito maior que a media geral.
- Existe **concentracao temporal de risco** (buckets de 5 min com exposicao muito alta), o que exige limite por aposta e limite agregado por janela.

---

## 6) Inferencia estatistica de valor (secao de preco)

Analise inferencial sobre `diff_pct` das coortes:

| Coorte | N | Media diff (%) | IC95 | t-stat vs 0 | Significativo 95% |
|---|---:|---:|---|---:|---|
| BACK_EDGE | 1014 | +45.306 | [18.570, 72.043] | +3.32 | YES |
| LAY_EDGE | 342 | -11.128 | [-12.717, -9.539] | -13.72 | YES |

### Interpretacao correta
- Ha evidencia estatistica de que as coortes selecionadas diferem de zero em `diff_pct`.
- Isso valida **sinal de preco** (descolamento BS vs WS) para ambas coortes.
- Mas `diff_pct` significativo **nao e igual** a ROI realizado significativo.

---

## 7) Inferencia de resultado realizado (P/L e CLV)

| Tabela | N total | N P/L | Cob. P/L | P/L medio (u) | IC95 P/L | N CLV | Cob. CLV | CLV medio (%) | IC95 CLV | Win rate % | IC95 Win rate |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| h1_pricing_events | 12507 | 447 | 3.6% | +0.4329 | [0.3585, 0.5074] | 447 | 3.6% | -0.4677 | [-1.3601, 0.4246] | 72.26 | [68.11, 76.41] |
| h3_line_monotonicity_events | 626 | 50 | 8.0% | -0.0074 | [-0.2683, 0.2534] | 50 | 8.0% | +6.3414 | [-3.7893, 16.4721] | 52.00 | [38.15, 65.85] |
| h3b_temporal_reversal_events | 21135 | 349 | 1.7% | -0.1042 | [-0.1989, -0.0096] | 693 | 3.3% | +150.5592 | [41.1827, 259.9357] | 41.26 | [36.10, 46.43] |
| h6_correlation_lag_events | 7850 | 784 | 10.0% | -0.0950 | [-0.1600, -0.0301] | 784 | 10.0% | +0.0682 | [-0.3922, 0.5286] | 47.96 | [44.46, 51.46] |

### Interpretacao executiva
- **H1**: P/L realizado positivo e significativo na amostra liquidada.
- **H3**: inconclusivo no P/L (IC95 cruza zero).
- **H3B**: P/L realizado negativo no recorte liquidado; sinaliza cautela para uso direto.
- **H6**: P/L realizado negativo no recorte liquidado.

### Ponto critico de qualidade estatistica
- Cobertura de liquidacao (`n_pl`) ainda e baixa em algumas hipoteses (ex.: H3B 1.7%).
- Logo, conclusoes de ROI/drawdown realizado ainda tem incerteza relevante por amostra efetiva limitada.

---

## 8) Implicacoes de negocio e risco (proxima fase)

### 8.1 Para Back
- Evidencia de sinal de preco favoravel na coorte.
- Priorizar continuidade de validacao com foco em liquidez/limite e estabilidade de execucao.

### 8.2 Para Lay
- Tratar como estrategia separada, com controles dedicados:
  - limite de liability por aposta;
  - limite de liability agregado por bucket (5m/15m);
  - trava por percentil (ex.: bloquear acima de p95/p99 ate calibracao).

### 8.3 Para inferencia final de valor economico
- Aumentar cobertura de `profit_loss` liquidado antes de decisao definitiva.
- Rodar inferencia periodica por versao com janela movel (7d/14d/30d) e controle de cauda.

---

## 9) Recomendacao executiva consolidada

1. **Manter v4.0-api como baseline operacional** (melhor desempenho entre versoes).  
2. **Continuar separando Back e Lay** em estrategia, risco e governanca.  
3. **Nao escalar Lay sem limite de cauda** (liability p99/ES95 demonstram risco elevado).  
4. **Usar inferencia de P/L realizado com critério de cobertura minima** para decisao de escala.  
5. Preparar proxima iteracao de decisao com foco em:
- ganho incremental de cobertura `n_pl`;
- consistencia de sinais por janela temporal;
- robustez de risco de cauda no Lay.

