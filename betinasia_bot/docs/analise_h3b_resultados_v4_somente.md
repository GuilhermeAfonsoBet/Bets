# Análise H3B — Somente Dados da Versão v4

**Data da extração:** 12/02/2026  
**Escopo deste documento:** exclusivamente resultados do modelo v4 (API 2-4s vs DOM 15-30s).  
**Regra editorial:** sem comparação com versões anteriores.

---

## 1) Contexto da rodada v4

| Indicador | Valor |
|---|---:|
| Auditorias H3B UP (match + kickoff passado) | 3975 |
| Betslip bruto | 2013 |
| Betslip confiável (diff -10% a +10%) | 892 |
| Descartados no filtro de qualidade | 1121 |
| Jogos únicos (geral) | 349 |
| Média de observações por jogo | 11,4 |
| Jogos únicos com betslip | 155 |

---

## 2) Base comparativa v4: API vs DOM

| Métrica | API (rótulo histórico 2-4s) | DOM (15-30s) |
|---|---:|---:|
| Total de observações | 1053 | 2843 |
| Com betslip | 763 | 129 |
| Com CLV pre-match | 35 | 45 |
| Com ROI | 680 | 105 |
| Lag médio observado (fim-a-fim) | 11522 ms | 15170 ms |

Leitura rápida:
- API tem muito mais cobertura útil de betslip.
- DOM concentra poucos casos com forte erosão de preço.

Nota importante sobre lag:
- O nome "API (2-4s)" é um rótulo operacional legado da arquitetura.
- O valor de 11,5s é o **lag fim-a-fim observado** (detecção -> abertura -> extração no betslip), já incluindo overhead de fila, navegação e latência de execução.
- Ou seja, o sistema API é mais rápido que DOM, mas hoje ainda acima do alvo teórico de 2-4s em média.

### 2.1) Cobertura temporal (pre-match vs in-match) na rodada v4

| Métrica | Pre-match | In-match | Observação |
|---|---:|---:|---|
| Observações totais com classificação temporal | 1568 | 1652 | Contagem bruta do corte v4 |
| ROI Betslip | 379 | 403 | Amostra com resultado de aposta |
| ROI WebSocket | 1430 | 1250 | Referência de mercado |
| CLV Betslip (informativo) | 80 | 42 | Decisão prioriza CLV pre-match |

Leitura:
- Neste corte específico do v4, in-match ficou ligeiramente acima de pre-match no bruto (1652 vs 1568).
- Isso pode ocorrer por cobertura de jogos ao vivo em ciclos longos e por filtros de disponibilidade (betslip/closing/resultados), não necessariamente por maior "densidade de oportunidade" intrínseca.

---

## 3) CLV pre-match (núcleo da comparação v4)

### 3.1 CLV com odd Betslip (execução real)

| Métrica | API (2-4s) | DOM (15-30s) |
|---|---:|---:|
| CLV Bruto BS Pre-Match | +0,839% (NS, N=35) | -2,144% (sig. negativo, N=45) |
| CLV Adicional BS Pre-Match | +0,211% (NS, N=35) | -3,354% (sig. negativo, N=45) |
| Taxa de CLV > 0 (bruto) | 57,1% | 25,0% |
| Taxa de CLV > 0 (adicional) | 51,4% | 26,7% |

### 3.2 CLV WebSocket (referência interna do v4)

| Métrica | API (2-4s) | DOM (15-30s) |
|---|---:|---:|
| CLV Bruto WS Pre-Match | +1,167% (sig. positivo, N=45) | +0,112% (NS, N=981) |
| Taxa de CLV > 0 (WS bruto) | 62,8% | 51,2% |

Leitura:
- No v4, API preserva melhor o valor de entrada no pre-match.
- DOM fica estruturalmente negativo em CLV de execução real.

Nota metodológica:
- A "taxa de CLV > 0" **não** é win rate de aposta.
- Ela mede apenas a proporção de observações com CLV positivo.
- Como `CLV adicional = CLV bruto - baseline`, o sinal pode mudar por observação e, por isso, a taxa de CLV > 0 do bruto e do adicional pode ser diferente.

---

## 4) ROI por modelo (v4)

| Métrica | API (2-4s) | DOM (15-30s) |
|---|---:|---:|
| ROI Betslip | -2,422% (NS, N=680) | +6,893% (NS, N=105) |
| ROI WebSocket | +0,737% (NS, N=953) | -0,276% (NS, N=2263) |
| Win rate ROI Betslip | 50,0% | 60,5% |
| Win rate ROI WS | 49,2% | 49,4% |

Leitura:
- Em ROI, nenhum dos dois modelos mostrou significância estatística robusta nesta rodada.
- O diferencial v4 aparece mais em CLV/diff de execução do que em ROI consolidado.

---

## 5) Diferença de preço BS vs WS (v4)

| Métrica | API (2-4s) | DOM (15-30s) |
|---|---:|---:|
| Diff BS vs WS (média) | +0,185% (NS, N=763) | -2,931% (sig. negativo, N=129) |
| BS > WS | 38,0% (290/763) | 8,5% (11/129) |
| BS > WS +2% | 18,5% (141/763) | 7,0% (9/129) |

Leitura:
- API opera mais perto (ou acima) do preço de referência.
- DOM perde preço de forma recorrente.

---

## 6) Combinações de valor (v4)

### 6.1 Buckets por diferença BS vs WS

| Bucket | N bucket | CLV BS PM | ROI BS (todos) | Leitura |
|---|---:|---|---|---|
| BS < WS (-10% a -2%) | 186 | -4,748% (sig. neg., N=20) | +3,852% (NS, N=159) | CLV fortemente negativo quando BS chega pior que WS |
| BS ~ WS (-2% a +2%) | 556 | -0,100% (NS, N=51) | -4,355% (NS, N=487) | Zona neutra em CLV, ROI sem confirmação |
| BS > WS (+2% a +10%) | 150 | +3,661% (sig. pos., N=9) | +4,208% (NS, N=139) | Melhor combinação de CLV, mas com N pequeno em CLV |

### 6.2 Combinação por faixa de linha AH

| Faixa AH | CLV BS PM | ROI BS | Diff BS vs WS | Leitura |
|---|---|---|---|---|
| AH 0-1 (líquida) | -2,551% (sig. neg., N=46) | -1,349% (NS, N=337) | -0,942% (sig. neg., N=379) | Faixa mais crítica no v4 |
| AH 1-2 (média) | +0,882% (NS, N=8) | +8,983% (NS, N=103) | +0,738% (sig. pos., N=124) | Faixa promissora, ainda sem confirmação estatística em CLV/ROI |
| AH 2+ (extrema) | +1,661% (sig. pos., N=26) | -4,041% (NS, N=345) | +0,074% (NS, N=389) | CLV positivo, mas sem conversão em ROI |

### 6.3 Combinação por faixa de lag

| Faixa de lag | CLV BS PM | ROI BS | Diff BS vs WS | Leitura |
|---|---|---|---|---|
| < 10s | +0,296% (NS, N=33) | -1,938% (NS, N=560) | -0,146% (NS, N=641) | Melhor equilíbrio operacional de preço |
| 10-20s | -2,237% (sig. neg., N=39) | +5,202% (NS, N=87) | -2,393% (sig. neg., N=107) | Faixa com erosão clara de preço |
| 20-30s | +2,878% (NS, N=4) | -14,629% (NS, N=70) | +0,150% (NS, N=75) | Amostra CLV muito pequena, sem robustez |
| > 30s | -0,286% (NS, N=4) | +10,782% (NS, N=68) | +1,477% (sig. pos., N=69) | Resultado heterogêneo e sem confirmação em CLV |

Observação importante:
- As combinações com CLV PM em alguns buckets têm N pequeno (ex.: N=8, N=9, N=4), então o ranking de valor deve ser tratado como indicativo, não conclusivo.

---

## 7) Estimativa econômica do combo BS > WS +2% (v4)

Base usada para estimativa:
- Bucket: **BS > WS (+2% a +10%)**
- ROI Betslip (todos): média **+4,208%**, IC90 **[-9,433%, +17,848%]**, N=139 (não significativo).
- Câmbio de referência: **USD/BRL = 5,20**.

Premissa de stake (ajustada):
- **Não usar stake flat.**
- Para cada aposta: `stake_i = f × stake_max_i`, onde `f` é a fração de execução sobre o stake máximo disponível no betslip.
- Referência empírica de stake máximo (histórico operacional em planilhas da operação): média aproximada **USD 440** por oportunidade elegível.

### 7.1 Validação da estimativa de volume (N/mês)

No corte atual, o bucket BS > WS +2% tem **N=150** ocorrências no período observado.  
A projeção mensal depende diretamente da duração efetiva dessa janela:

`N_mensal ≈ 150 × (30 / dias_da_janela_observada)`

| Janela observada (dias) | Projeção N/mês | Apostas/dia |
|---|---:|---:|
| 10 | 450 | 15,0 |
| 14 | 321 | 10,7 |
| 21 | 214 | 7,1 |
| 30 | 150 | 5,0 |
| 45 | 100 | 3,3 |

Leitura:
- O cenário de **340/mês (~11/dia)** é coerente se a janela efetiva do N=150 for próxima de **13-14 dias**.
- Se a janela real for mais longa (ex.: 30 dias), o volume esperado cai para ~150/mês.

### 7.2 Lucro potencial e turnover mensal (stake dinâmico)

Cenário base de volume: **340 apostas/mês**  
Stake médio executado por aposta: `f × 440`

| Fração do stake máximo (f) | Stake médio (USD) | Turnover/mês (USD) | Lucro esperado/mês (USD) | IC90 lucro/mês (USD) | Lucro esperado/mês (BRL) | IC90 lucro/mês (BRL) |
|---|---:|---:|---:|---:|---:|---:|
| 25% | 110 | 37.400 | +1.574 | [-3.528, +6.675] | +8.184 | [-18.345, +34.712] |
| 35% (base) | 154 | 52.360 | +2.203 | [-4.939, +9.345] | +11.457 | [-25.683, +48.595] |
| 50% | 220 | 74.800 | +3.148 | [-7.056, +13.350] | +16.367 | [-36.691, +69.420] |

Leitura:
- Mesmo com stake dinâmico, o intervalo ainda cruza negativo de forma ampla.
- O combo tem upside, mas continua sem robustez estatística para escalar agressivamente.

### 7.3 Sensibilidade por volume (f = 35% do stake máximo)

| Apostas/mês | Turnover/mês (USD) | Lucro esperado/mês (USD) | IC90 lucro/mês (USD) | Lucro esperado/mês (BRL) | IC90 lucro/mês (BRL) |
|---|---:|---:|---:|---:|---:|
| 150 | 23.100 | +972 | [-2.179, +4.123] | +5.055 | [-11.331, +21.439] |
| 340 | 52.360 | +2.203 | [-4.939, +9.345] | +11.457 | [-25.683, +48.595] |
| 500 | 77.000 | +3.240 | [-7.263, +13.743] | +16.849 | [-37.769, +71.463] |
| 800 | 123.200 | +5.184 | [-11.621, +21.989] | +26.958 | [-60.431, +114.342] |

### 7.4 Banca exigida (sleeve do combo)

Método (aproximação de risco):
- Escalonamento da volatilidade mensal pela stake média executada.
- Referência de downside mensal p99 no cenário base histórico ajustado para stake dinâmico.
- Stress de 3 meses + fator de segurança.

Para 340 apostas/mês:
- `f=25%` -> downside mensal p99 ≈ **-USD 3.038**, stress 3m ≈ **USD 9.115**  
- `f=35%` -> downside mensal p99 ≈ **-USD 4.254**, stress 3m ≈ **USD 12.761**  
- `f=50%` -> downside mensal p99 ≈ **-USD 6.077**, stress 3m ≈ **USD 18.230**

Faixa sugerida de banca (sleeve dedicada):
- **USD 20k a 30k** (aprox. **BRL 104k a 156k**)

Observação:
- Essa faixa é para operar o combo como sleeve dedicada, não para banca total da operação.

---

## 8) Conclusão objetiva da versão v4

1. **API (2-4s) é superior ao DOM (15-30s)** em qualidade de execução.
2. **DOM permanece negativo em CLV de betslip pre-match** com significância estatística.
3. **API ainda não confirmou edge estatístico em CLV adicional de execução**, apesar de sinal melhor que DOM.
4. Em **ROI**, os resultados permanecem não conclusivos para decisão final.
5. Nas combinações, o bucket **BS > WS** concentra o melhor CLV, enquanto **AH 0-1** concentra a pior dinâmica de preço no v4.

---

## 9) Recomendação operacional (somente v4)

- Manter API como trilha principal.
- Reduzir ainda mais latência e aumentar N de CLV pre-match no API.
- Reavaliar quando houver amostra suficiente para fechar significância no CLV adicional de execução.

