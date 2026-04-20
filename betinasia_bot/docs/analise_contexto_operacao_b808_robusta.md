# Análise Estatística Robusta — Contexto Operação (b808)
**Data da execução:** 14/02/2026 23:39 UTC  
**Escopo:** H3B (auditoria), com comparação por `audit_version` e inferência robusta por jogo (cluster bootstrap).  
**Nota:** a robustez aqui significa que intervalos de confiança consideram correlação intra-jogo (múltiplas auditorias por partida).
---
## 0) Sumário executivo (leitura rápida)
- **Recorte**: direction=`up`, lookback_days=`2`, versions=`v4.0-api,v1.0,v1.0-recovered`.
- **Amostra**: 1900 auditorias (jogos únicos=161, média=11.8 obs/jogo); betslip confiável=925.
- **Coortes (status=OK, betslip confiável)**: Back (diff>=2.0%): **326**; Lay (diff<=-2.0%): **120**.
- **Coberturas em `hypothesis_details` (OK)**: temporal(back)=707/925; lay_temporal=659/925; finance=516/925.
- **Cobertura de placar (ROI)**: jogos com placar=130/161 (status finished=130).
- **DOM**: sem dados no recorte atual (N=0), então não há comparação API vs DOM aqui.
- **CLV pre-match (Betslip, API)**: média robusta por jogo +1.966% (IC90 [+1.299%, +2.605%]), com N=435 eventos (jogos=86).
- **Padrão por bucket (CLV PM)**: `BS < WS` -3.854% (sig. negativo), `BS ~ WS` -0.189% (NS), `BS > WS` +6.986% (sig. positivo).
- **Leitura recomendada**: use este relatório para validar **qualidade de execução/CLV**. Para concluir sobre **ROI**, rode após atualizar resultados e/ou use uma janela com jogos já liquidados.

---
## 1) Contexto do corte (b808)
| Indicador | Valor |
|---|---:|
| Auditorias H3B `UP` (match + kickoff passado) | 1900 |
| Betslip bruto | 1535 |
| Betslip confiável (diff -10% a +10%) | 925 |
| Descartados no filtro de qualidade | 610 |
| Jogos únicos (geral) | 161 |
| Média de observações por jogo | 11.8 |
| Jogos únicos com betslip confiável | 155 |

---
## 2) Base comparativa: API vs DOM
| Métrica | API (v4.0-api) | DOM (v1.0) |
|---|---:|---:|
| Total de observações | 1900 | 0 |
| Com betslip confiável | 925 | 0 |
| Com CLV pre-match (betslip) | 435 | 0 |
| Com ROI (betslip) | 789 | 0 |
| Lag médio observado (fim-a-fim) | 15535 ms | — ms |

---
### 2.1 Cobertura temporal (pre-match vs in-match)
| Métrica | Pre-match | In-match | Observação |
|---|---:|---:|---|
| Observações totais com classificação temporal | 721 | 1179 | Contagem bruta do corte |
| ROI Betslip | 440 | 349 | Amostra com resultado do jogo |
| ROI WebSocket | 614 | 995 | Referência de mercado |
| CLV (apenas pre-match) | 435 | — | CLV vs closing pré-jogo não é interpretável in-match |

---
### 2.2 Performance por regime (pre-match vs in-match)
| Regime | N auditorias | N betslip conf. | N OK | Back edge | Lay edge | Diff médio (OK) |
|---|---:|---:|---:|---:|---:|---:|
| PRE_MATCH | 721 | 502 | 502 | 192 | 36 | +2.166% |
| IN_MATCH | 1179 | 423 | 423 | 134 | 84 | +0.995% |

---
## 3) CLV pre-match (núcleo)
### 3.1 CLV com odd do Betslip (execução real)
| Métrica | API | DOM |
|---|---:|---:|
| CLV Bruto BS Pre-Match | +2.379% (sig. positivo, N=435, jogos=86) | — (N/A, N=0, jogos=0) |
| CLV Adicional BS Pre-Match | +1.467% (NS, N=435, jogos=86) | — (N/A, N=0, jogos=0) |
| Taxa de CLV > 0 (bruto) | 60.6% | —% |
| Taxa de CLV > 0 (adicional) | 60.7% | —% |

Notas de robustez (IC 90% por jogo):  
- API CLV bruto (cluster): média +1.966%; IC90 [+1.299%, +2.605%]  
- DOM CLV bruto (cluster): média —; IC90 —  

---
## 4) ROI por modelo
| Métrica | API | DOM |
|---|---:|---:|
| ROI Betslip | +2.563% (NS, N=789) | — (N/A, N=0) |
| ROI WebSocket | -0.675% (NS, N=1603) | — (N/A, N=0) |
| Win rate ROI Betslip | 51.7% | —% |
| Win rate ROI WS | 50.0% | —% |

---
## 5) Diferença de preço BS vs WS
| Métrica | API | DOM |
|---|---:|---:|
| Diff BS vs WS (média) | +1.630% (sig. positivo, N=925) | — (N/A, N=0) |
| BS > WS | 46.4% (429/925) | —% (0/0) |
| BS > WS +2% | 35.2% (326/925) | —% (0/0) |

---
## 6) Combinações de valor
### 6.1 Buckets por diferença BS vs WS
| Bucket | N bucket | CLV BS PM (média) | IC90 (cluster) | N CLV PM | Jogos CLV PM | ROI BS (todos) | IC90 (cluster) |
|---|---:|---:|---|---:|---:|---:|---|
| BS < WS (-10% a -2%) | 120 | -3.854% | [-5.565%, -3.073%] | 30 | 24 | +0.098% | [-16.214%, +14.302%] |
| BS ~ WS (-2% a +2%) | 479 | -0.189% | [-0.658%, +0.345%] | 234 | 79 | +1.973% | [-9.185%, +8.310%] |
| BS > WS (+2% a +10%) | 326 | +6.986% | [+6.580%, +7.652%] | 171 | 63 | +4.345% | [-0.629%, +23.552%] |

---
### 6.2 Combinação por faixa de linha AH
| Faixa AH | CLV BS PM (média) | IC90 (cluster) | ROI BS (todos) | IC90 (cluster) | Diff BS vs WS (média) |
|---|---:|---|---:|---|---:|
| AH 0-1 (líquida) | +2.695% | [+1.714%, +3.409%] | +5.527% | [+3.530%, +24.838%] | +1.802% |
| AH 1-2 (média) | +2.541% | [+1.153%, +4.056%] | +3.384% | [-15.215%, +16.466%] | +2.208% |
| AH 2+ (extrema) | +1.857% | [+0.186%, +2.488%] | +0.196% | [-8.979%, +12.492%] | +1.326% |

---
### 6.3 Combinação por faixa de lag
| Faixa de lag | CLV BS PM (média) | IC90 (cluster) | N CLV PM | Jogos CLV PM | ROI BS (todos) | IC90 (cluster) | Diff BS vs WS (média) |
|---|---:|---|---:|---:|---:|---|---:|
| < 10s | +2.465% | [+1.512%, +2.778%] | 331 | 82 | +3.003% | [-3.762%, +10.043%] | +1.541% |
| 10-20s | +1.493% | [-0.093%, +3.047%] | 4 | 4 | +18.820% | [-23.222%, +64.456%] | +1.011% |
| 20-30s | +1.826% | [+0.452%, +2.907%] | 67 | 45 | +5.015% | [-4.774%, +29.383%] | +1.846% |
| > 30s | +2.745% | [+1.325%, +4.414%] | 33 | 24 | -11.262% | [-35.729%, +10.951%] | +2.390% |

---
## 7) Estimativa financeira (proxy) e risco
Este bloco usa `hypothesis_details.finance` quando existe; se não existir, usa stake fallback = `stake_pct_of_limit × limit`.

### 7.1 Back (BS >> WS)
| Métrica | Valor |
|---|---:|
| Corte (diff_pct) | >= 2.0% |
| N eventos | 326 |
| Stake total (estimado) | 110611.48 |
| Stake médio | 339.30 |
| Profit_if_win total (estimado) | 116319.53 |
| Profit_if_win médio | 356.81 |
| N com ROI realizado | 280 |
| P&L realizado total (estimado) | -14122.39 |
| ROI realizado (ponderado por stake) | -13.47% |

### 7.2 Lay (BS << WS) — risco de cauda
| Métrica | Valor |
|---|---:|
| Corte (diff_pct) | <= -2.0% |
| N eventos | 120 |
| Stake total (estimado) | 12818.81 |
| Liability total (estimada) | 12123.97 |
| Liability média | 101.03 |
| Liability p95 | 440.90 |
| Liability p99 | 968.25 |
| ES95 (liability) | 775.92 |
| Liability max | 1199.86 |
| Proxy de banca (>= p99 liability) | 968.25 |

---
## 8) Curva temporal (pico, reversão e melhor timing)
Esta seção usa `hypothesis_details.temporal` (Back) e `hypothesis_details.lay_temporal` (Lay), coletados em pontos discretos (t≈0,3,6,10,15,20s).

O objetivo é responder: **tempo até o pico/vale**, **% que segue melhorando até t_max**, **% com reversão** e **impacto esperado por timing**. **CLV é reportado somente pre-match** (closing pré-jogo).

### 8.1 Back (pico em diff_pct)
Definições: **pico** = `max(diff_pct)`. **reversão** = após o pico, `diff_pct` cair pelo menos 0.50 p.p. (para Lay, subir 0.50 p.p. após o vale). `t_reversão` é o 1º tempo que cruza esse limiar (logo, não é o pico).

| Regime | N | t_pico médio (s) | t_pico p50 (s) | % melhora até fim | % com reversão | t_reversão médio (s) | Δt (rev - pico) médio (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRE_MATCH | 502 | 4.5 | 5.2 | 73.7% | 19.7% | 12.3 | 7.1 |
| IN_MATCH | 423 | 4.3 | 0.0 | 65.2% | 26.7% | 13.0 | 7.9 |

**Partição 100% (Back)**: categorias exclusivas (somam 100% por regime).

| Regime | % pico no fim, sem reversão | % pico no fim, com reversão (recuperou) | % pico antes, com reversão | % pico antes, sem reversão relevante |
|---|---:|---:|---:|---:|
| PRE_MATCH | 77.7% | 7.6% | 12.2% | 2.6% |
| IN_MATCH | 70.2% | 4.0% | 22.7% | 3.1% |

**Curva média (Back)**: média de `diff_pct` e `odd` por tempo. `CLV` é reportado **somente pre-match** (closing pré-jogo). `ROI` (quando aparece) é o **ROI realizado** se a aposta fosse feita naquele ponto.

| Tempo | N pts | diff_pct médio | odd média | CLV médio (pre-match) | ROI médio (se houver) |
|---|---:|---:|---:|---:|---:|
| t+0s | 925 | +1.63% | 1.991 | +2.38% | 2.56 |
| t+6s | 694 | +2.39% | 2.002 | +2.71% | 2.03 |
| t+10s | 1116 | +2.92% | 2.015 | +2.74% | 3.44 |
| t+15s | 702 | +2.89% | 2.017 | +2.79% | -0.09 |
| t+20s | 961 | +3.49% | 2.018 | +2.84% | 2.82 |

### 8.1b Back — impacto por timing (t0 vs pico vs último) e reversão
Leitura: se a estratégia é **entrar no pico**, compare CLV/ROI no pico vs t0 e veja diferença entre **casos com reversão** vs **sem reversão**. Observação: **CLV só é calculado pre-match**.

| Subcoorte | N total | N CLV (PM) | CLV t0 | CLV pico | CLV último | N ROI | ROI t0 | ROI pico | ROI último |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 713 | 348 | +1.92% | +2.19% | +2.18% | 609 | 3.09 | 5.98 | 5.96 |
| COM_REVERSAO | 212 | 87 | +4.19% | +5.58% | +4.49% | 180 | 0.77 | 3.25 | 0.54 |

### 8.2 Lay (vale em diff_pct)
| Regime | N | t_vale médio (s) | t_vale p50 (s) | % melhora até fim | % com reversão | t_reversão médio (s) | Δt (rev - vale) médio (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRE_MATCH | 443 | 4.6 | 5.2 | 65.9% | 26.0% | 13.1 | 7.0 |
| IN_MATCH | 308 | 5.3 | 0.0 | 51.9% | 37.0% | 13.5 | 8.4 |

**Partição 100% (Lay)**: categorias exclusivas (somam 100% por regime).

| Regime | % vale no fim, sem reversão | % vale no fim, com reversão (recuperou) | % vale antes, com reversão | % vale antes, sem reversão relevante |
|---|---:|---:|---:|---:|
| PRE_MATCH | 69.5% | 10.6% | 15.3% | 4.5% |
| IN_MATCH | 58.4% | 4.5% | 32.5% | 4.5% |

**Curva média (Lay)**: usa `odd=lay_odd`. `CLV` é reportado **somente pre-match** (closing pré-jogo). O ROI mostrado aqui é **ROI por liability** (não por stake), porque Lay é governado por risco.

| Tempo | N pts | diff_pct médio | lay_odd média | CLV médio (pre-match) | ROI/liability médio (se houver) |
|---|---:|---:|---:|---:|---:|
| t+0s | 751 | +0.88% | 1.966 | -0.16% | 7.33 |
| t+6s | 650 | +1.27% | 1.973 | +0.01% | 8.27 |
| t+10s | 1032 | +0.27% | 1.959 | +0.31% | 10.73 |
| t+15s | 659 | +1.54% | 1.978 | -0.27% | 13.93 |
| t+20s | 908 | +2.75% | 2.002 | +0.02% | 5.04 |

### 8.2b Lay — impacto por timing (t0 vs vale vs último) e reversão
Leitura: se a estratégia é **entrar no vale (odd mais baixa)**, compare CLV/ROI no vale vs t0 e veja diferença entre **casos com reversão** vs **sem reversão**. Observação: **CLV só é calculado pre-match**.

| Subcoorte | N total | N CLV (PM) | CLV t0 | CLV vale | CLV último | N ROI | ROI/liab t0 | ROI/liab vale | ROI/liab último |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SEM_REVERSAO | 522 | 279 | -0.46% | -0.04% | -0.07% | 451 | 0.93 | 1.28 | 1.23 |
| COM_REVERSAO | 229 | 98 | +0.67% | +2.28% | +0.99% | 191 | 22.46 | 39.28 | 17.03 |

---
## 9) Combinações de valor (regime × linha AH × lag)
Tabela rankeada por volume (N). Métrica: diff_pct (OK, betslip confiável).

### 9.1 Back combos (diff_pct >= corte)
| Regime | Linha | Lag | N | Jogos | Mean diff % | IC90 cluster |
|---|---|---|---:|---:|---:|---|
| IN_MATCH | AH 2+ (extrema) | < 10s | 69 | 47 | +6.10% | [+5.58%, +6.70%] |
| PRE_MATCH | AH 0-1 (líquida) | < 10s | 67 | 28 | +6.96% | [+6.85%, +7.62%] |
| PRE_MATCH | AH 2+ (extrema) | < 10s | 48 | 24 | +6.91% | [+6.58%, +7.47%] |
| PRE_MATCH | AH 1-2 (média) | < 10s | 35 | 18 | +6.94% | [+6.63%, +7.62%] |
| IN_MATCH | AH 0-1 (líquida) | < 10s | 21 | 16 | +7.05% | [+6.05%, +7.79%] |
| IN_MATCH | AH 1-2 (média) | < 10s | 14 | 14 | +6.56% | [+5.45%, +7.64%] |
| PRE_MATCH | AH 0-1 (líquida) | 20-30s | 13 | 11 | +7.43% | [+6.63%, +8.07%] |
| IN_MATCH | AH 2+ (extrema) | > 30s | 11 | 8 | +7.27% | [+5.01%, +8.34%] |
| PRE_MATCH | AH 0-1 (líquida) | > 30s | 7 | 6 | +6.98% | [+6.11%, +7.71%] |
| PRE_MATCH | AH 2+ (extrema) | 20-30s | 7 | 5 | +7.24% | [+6.33%, +8.65%] |
| PRE_MATCH | AH 2+ (extrema) | > 30s | 6 | 5 | +6.24% | [+4.88%, +7.99%] |
| IN_MATCH | AH 0-1 (líquida) | > 30s | 6 | 4 | +6.91% | [+5.89%, +7.94%] |

### 9.2 Lay combos (diff_pct <= corte) + risco
| Regime | Linha | Lag | N | Jogos | Mean diff % | IC90 cluster | Liability p95 |
|---|---|---|---:|---:|---:|---|---:|
| IN_MATCH | AH 2+ (extrema) | < 10s | 40 | 33 | -4.97% | [-5.73%, -4.52%] | 290.66 |
| IN_MATCH | AH 0-1 (líquida) | < 10s | 23 | 19 | -5.26% | [-6.07%, -4.65%] | 191.72 |
| PRE_MATCH | AH 0-1 (líquida) | < 10s | 13 | 11 | -4.86% | [-6.47%, -4.10%] | 70.95 |
| PRE_MATCH | AH 2+ (extrema) | < 10s | 11 | 11 | -4.53% | [-5.54%, -3.57%] | 1022.80 |
| IN_MATCH | AH 1-2 (média) | < 10s | 6 | 6 | -3.43% | [-4.00%, -3.01%] | 867.76 |
| PRE_MATCH | AH 1-2 (média) | < 10s | 6 | 5 | -4.45% | [-6.25%, -3.07%] | 81.80 |
| IN_MATCH | AH 2+ (extrema) | > 30s | 5 | 5 | -4.39% | [-6.60%, -2.94%] | 182.86 |
| IN_MATCH | AH 2+ (extrema) | 20-30s | 4 | 4 | -4.40% | [-5.69%, -3.05%] | 268.18 |
| IN_MATCH | AH 0-1 (líquida) | 20-30s | 3 | 3 | -3.18% | [-3.50%, -2.85%] | 175.10 |
| IN_MATCH | AH 0-1 (líquida) | > 30s | 3 | 3 | -6.34% | [-8.04%, -4.63%] | 214.41 |
| PRE_MATCH | AH 0-1 (líquida) | > 30s | 2 | 2 | -3.51% | [-3.67%, -3.35%] | 133.35 |
| PRE_MATCH | AH 2+ (extrema) | 20-30s | 2 | 2 | -4.59% | [-6.26%, -2.92%] | 80.99 |

---
## 10) Diagnóstico: por que o ROI pode estar zerado
ROI aqui é calculado por placar do jogo (`matches.home_score/away_score`). Se os placares não estiverem preenchidos no banco, a cobertura de ROI será 0.

| Indicador | Valor |
|---|---:|
| Jogos únicos no recorte | 161 |
| Jogos com placar disponível (home_score/away_score não nulos) | 130 |
| Jogos com status='finished' no banco | 130 |

### 10.1 Distribuição por data de kickoff (explica janela da API)
- Kickoff (UTC) no recorte: **2026-02-12 23:00 UTC** até **2026-02-14 20:05 UTC**.

| Kickoff date (UTC) | Jogos | Com placar | Cobertura |
|---|---:|---:|---:|
| 2026-02-14 | 109 | 97 | 89.0% |
| 2026-02-13 | 50 | 33 | 66.0% |
| 2026-02-12 | 2 | 0 | 0.0% |

**Leitura**: se seu recorte inclui muitos jogos com kickoff antigo, a API-Football **free** pode não retornar fixtures dessa data (limitação por janela recente). Nesse cenário, mesmo com o job rodando, `placar disponível` ficará baixo para jogos fora da janela.

Se `placar disponível` estiver 0 (mesmo para datas recentes), isso geralmente indica que o job de resultados não rodou ou está sem chave válida.  
Sugestão (rodar no servidor): `cd betinasia_bot && python3 -m results.auto_update_results --once` (ou configure o serviço para rodar em loop).

---
## 11) Conclusões, riscos e pontos em aberto
- **Execução (CLV)**: use as seções 3/6 para validar qualidade de execução (especialmente pre-match). Se CLV cluster ficar robustamente positivo, há evidência de boa entrada; se ficar negativo, há erosão estrutural.
- **Pre-match vs in-match**: valide que o comportamento de edge/diff e ROI (quando houver placar) difere entre regimes (seção 2.2). Não é seguro misturar regimes para decisão.
- **Lay**: não pode ser decidido por média. Governança tem que usar p95/p99/ES95 de liability (seção 7.2) e combos com risco (seção 9.2). Se p99/ES95 forem altos, a estratégia precisa limite de exposição por janela.
- **Temporal (retenção de edge)**: se a cobertura `temporal/lay_temporal` for baixa, a inferência de retenção fica limitada (seção 8). Quando há cobertura, delta e retenção indicam se o edge “some” rápido.
- **ROI/resultado realizado**: sem placares no banco, ROI fica 0/ausente e a conclusão financeira final não é possível (seção 10). Primeiro garanta o job de resultados.
- **Pontos em aberto típicos**: (i) trazer DOM para a mesma janela, (ii) garantir atualização de placares, (iii) aumentar cobertura temporal e finance no `hypothesis_details`, (iv) definir política de banca para Lay.

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
