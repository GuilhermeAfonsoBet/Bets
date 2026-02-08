# Analise H3B — Resultados WebSocket + Betslip (v2)

**Data:** 08 de Fevereiro de 2026
**Versao:** 2.0
**Status:** Analise preliminar com dados filtrados

---

## 1. Resumo Executivo

Analisamos se a estrategia H3B (Reversao Temporal UP) tem valor na pratica, comparando odds do WebSocket com odds reais do Betslip. Os resultados sao mistos e levantam questoes importantes.

**Achados principais:**

1. **CLV WebSocket Pre-Match: +0.210% (SIGNIFICATIVO, N=906)** — Confirma que H3B UP detecta momentos onde a odd esta acima da closing line.

2. **CLV Betslip Pre-Match: -2.244% (SIGNIFICATIVO NEGATIVO, N=44)** — Quando consideramos a odd real disponivel no betslip (com lag de 10-30s), o valor desaparece.

3. **ROI Betslip Pre-Match: +14.99% (NAO SIGNIFICATIVO, N=34)** — Apesar do CLV negativo, o ROI real (baseado em resultados dos jogos) e positivo, embora nao significativo. Isto contradiz a premissa CLV = ROI.

4. **Erosao media pelo lag: -1.73%** — A odd do betslip e em media 1.73% pior que a do WebSocket.

5. **Correlacao entre apostas do mesmo jogo NAO corrigida** — Multiplas apostas do mesmo jogo inflam a significancia estatistica.

---

## 2. Metodologia

### 2.1. Dados

| Metrica | Valor |
|---------|-------|
| Total de auditorias H3B UP | 2.283 |
| Com betslip extraido (bruto) | 907 |
| Com diff entre -10% e +10% (filtrado) | 118 |
| Descartados (diff fora do range) | 789 (87%) |
| Com CLV WebSocket | 1.481 |
| Com CLV Betslip | 88 |
| Com ROI (resultado do jogo) | 1.839 |
| Pre-match | 982 |
| In-match | 723 |

### 2.2. Definicoes

- **CLV (Closing Line Value):** (odd_apostada - closing_odd) / closing_odd x 100%. Mede quanto a odd estava acima da linha de fechamento.
- **ROI (Return on Investment):** Lucro/prejuizo real baseado no resultado do jogo (gols), aplicando regras Asian Handicap.
- **Diff BS vs WS:** (betslip_odd - websocket_odd) / websocket_odd x 100%. Mede a erosao pelo lag.

### 2.3. Filtro de Qualidade

O extrator de betslip (via DOM parsing) produziu dados com erros significativos: 87% dos registros tinham diferencas maiores que +-10% entre betslip e websocket, indicando extracao incorreta.

Filtramos para diff entre -10% e +10%, resultando em N=118 registros confiaveis. O diagnostico pos-filtro mostra:
- Odds betslip: 1.030 a 2.545 (range razoavel)
- Odds websocket: 1.098 a 2.646
- Diferencas: -9.6% a +9.7% (sem outliers extremos)

**Limitacao:** 87% de descarte introduz possivel vies de selecao. Estamos implementando extracao via API (JSON estruturado) para eliminar esta limitacao.

---

## 3. Resultados

### 3.1. CLV WebSocket (Referencia)

| Metrica | Pre-Match | In-Match | Total |
|---------|-----------|----------|-------|
| N | 906 | 222 | 1.481 |
| CLV medio | +0.210% | +1.093% | +0.551% |
| IC 90% | [+0.067%, +0.352%] | [-0.453%, +2.159%] | [+0.311%, +0.777%] |
| Significancia | POSITIVO | Nao sig. | POSITIVO |

**Nota:** Este CLV e o CLV bruto (nao o "CLV adicional" que subtrai baseline). O CLV adicional da analise v6 anterior era +1.116% com N=273, usando metodologia diferente (com baseline de outras linhas). O valor menor aqui (+0.210%) pode refletir a amostra maior e mais representativa.

### 3.2. CLV Betslip (Realista)

| Metrica | Pre-Match | In-Match |
|---------|-----------|----------|
| N | 44 | 40 |
| CLV medio | -2.244% | -1.750% |
| IC 90% | [-3.291%, -1.196%] | [-4.215%, +0.715%] |
| Significancia | NEGATIVO | Nao sig. |

O CLV com a odd real do betslip e negativo para pre-match. O lag de execucao consome o valor detectado no WebSocket.

### 3.3. ROI Real (Baseado em Resultados dos Jogos)

| Metrica | Pre-Match (BS) | Pre-Match (WS) | In-Match (BS) | In-Match (WS) |
|---------|---------------|----------------|---------------|----------------|
| N | 34 | 746 | 56 | 583 |
| ROI medio | +14.99% | -4.20% | +4.38% | +2.53% |
| IC 90% | [-5.0%, +35.0%] | [-9.8%, +1.4%] | [-9.2%, +18.0%] | [-4.6%, +9.8%] |
| Significancia | Nao sig. | Nao sig. | Nao sig. | Nao sig. |
| N p/ sig. | ~61 | — | ~538 | ~4.433 |

**Observacao critica:** ROI Betslip Pre-Match e +14.99% (positivo) enquanto CLV e -2.24% (negativo). Isto contradiz a premissa usual de que CLV positivo = ROI positivo no longo prazo.

---

## 4. Analise por Dimensoes

### 4.1. Por Diferenca Betslip vs WebSocket

| Bucket | N | CLV BS (PM) | ROI BS |
|--------|---|------------|--------|
| BS < WS (-10% a -2%) | 67 | -5.24% [NEG] | +3.97% (NS) |
| BS ~ WS (-2% a +2%) | 43 | -1.35% [NEG] | -6.81% (NS) |
| BS > WS (+2% a +10%) | 8 | +3.76% [POS] | +94.14% [POS] |

**Achado:** Quando BS > WS (betslip melhor que websocket), tanto CLV quanto ROI sao significativamente positivos. Porem N=8 e muito pequeno.

### 4.2. Pre-Match vs In-Match

| Metrica | Pre-Match | In-Match |
|---------|-----------|----------|
| N betslip | 44 | 69 |
| Diff media BS vs WS | -1.73% [NEG] | -3.51% [NEG] |
| CLV Betslip | -2.24% [NEG] | -1.75% (NS) |
| ROI Betslip | +14.99% (NS) | +4.38% (NS) |

In-match tem erosao maior (-3.51% vs -1.73%) como esperado — odds mudam mais rapido durante o jogo.

### 4.3. Por Faixa de Linha AH

| Faixa | N | CLV BS (PM) | ROI BS | Diff BS vs WS |
|-------|---|------------|--------|---------------|
| AH 0-1 (liquida) | 108 | -3.34% [NEG] | -3.84% (NS) | -3.48% [NEG] |
| AH 1-2 (media) | 1 | Insuficiente | Insuficiente | Insuficiente |
| AH 2+ (extrema) | 9 | +2.41% [POS] | +105.11% [POS] | +3.58% [POS] |

**Atencao:** AH 2+ mostra resultados positivos mas com N=9. Linhas extremas tem baixa liquidez e alta volatilidade — resultados podem nao ser replicaveis.

### 4.4. Por Faixa de Lag

| Lag | N (CLV) | CLV BS (PM) | N (ROI) | ROI BS |
|-----|---------|------------|---------|--------|
| < 10s | 3 | -0.52% (NS) | 11 | +10.60% (NS) |
| 10-20s | 38 | -2.36% [NEG] | 65 | +15.24% [POS] |
| 20-30s | 2 | -4.68% [NEG] | 11 | -30.12% [NEG] |
| > 30s | 1 | Insuficiente | 6 | -6.85% (NS) |

**Achado critico:** No bucket 10-20s, CLV e negativo (-2.36%) mas ROI e positivo (+15.24%). Isto sugere que CLV nao e bom preditor de ROI nestes mercados especificos.

---

## 5. Problemas e Preocupacoes

### 5.1. CLV diferente de ROI

Este e o achado mais importante e preocupante. Possíveis explicacoes:

1. **Closing line nao e eficiente.** Em mercados de AH com linhas especificas, a ultima odd antes do kickoff pode nao refletir a "verdadeira" probabilidade.

2. **Amostra pequena.** ROI tem N=34 (pre-match betslip). Com desvio padrao de 70%, precisamos de ~61 observacoes para significancia. Pode ser ruido.

3. **Vies de selecao.** Filtramos 87% dos dados. Os 13% restantes podem nao ser representativos.

4. **H3B seleciona odds em movimento.** Odds que acabaram de subir (reversao UP) podem correlacionar com outcomes de forma diferente do que CLV sugere.

### 5.2. Correlacao Entre Apostas do Mesmo Jogo

**NAO estamos corrigindo para dependencia intra-jogo.** Se o mesmo jogo gera 5 sinais H3B (AH 0 home, AH 0 away, AH -1 home, etc.), estas 5 observacoes NAO sao independentes — o resultado do jogo afeta todas simultaneamente.

**Impacto:** Os intervalos de confianca podem estar SUBESTIMADOS (muito estreitos). A significancia estatistica pode ser inflada.

**Correcao necessaria:** Usar erros clusterizados por match_id, ou tomar apenas 1 observacao por jogo.

### 5.3. Qualidade dos Dados de Betslip

87% dos dados brutos de betslip foram descartados por terem diferencas > +-10%. Isto indica que o extrator DOM e fundamentalmente nao confiavel.

**Solucao em andamento:** Implementacao de extracao via API REST + WebSocket (PMM messages), que retorna dados em JSON estruturado. Expectativa de 100% de acuracia.

### 5.4. Lag de Execucao

O lag atual (10-30s com proxy + DOM) e alto demais. A nova arquitetura via API deve reduzir para 2-3s, mas ainda nao esta validada em producao.

---

## 6. Conclusoes

### 6.1. O que sabemos com confianca

1. **H3B UP tem CLV positivo no WebSocket** (+0.21%, significativo, N=906). O sinal detecta momentos com valor.

2. **O lag consome parte desse valor.** A erosao media e -1.73% (betslip vs websocket).

3. **CLV Betslip e negativo para pre-match** (-2.24%, significativo, N=44). Com a execucao atual, o edge e consumido.

### 6.2. O que ainda nao sabemos

1. **Se ROI real e positivo.** Indicios de +15% mas com N=34, nao significativo. Precisa de ~61 observacoes.

2. **Se CLV e bom preditor de ROI** nestes mercados. Os dados sugerem que nao, o que e uma descoberta importante.

3. **Se a reducao do lag (2-3s via API) recupera o edge.** Os dados de lag < 10s (N=3) mostram CLV de -0.52% (vs -2.36% em 10-20s), sugerindo que sim, mas amostra insuficiente.

4. **Se as correlacoes intra-jogo afetam a significancia.** Precisa de correcao estatistica.

### 6.3. Recomendacoes

| Prioridade | Acao | Impacto |
|-----------|------|---------|
| 1 | Implementar API betslip (JSON) | Elimina 87% de perda de dados, dados 100% confiaveis |
| 2 | Reduzir lag para 2-3s (via API) | Pode recuperar o edge consumido pelo lag |
| 3 | Corrigir erros clusterizados | Intervalos de confianca mais honestos |
| 4 | Acumular N=100+ betslip confiaveis | Significancia estatistica para ROI |
| 5 | Analisar CLV vs ROI formalmente | Entender se/quando CLV e bom proxy |

---

## 7. Proximos Passos

1. **Curto prazo (esta semana):** Validar API betslip em producao. Coletar dados via API (v4.0-api).

2. **Medio prazo (2-3 semanas):** Acumular 200+ betslips via API. Refazer analise completa com dados confiaveis.

3. **Analise estatistica robusta:** Implementar erros clusterizados por match_id. Analisar formalmente a relacao CLV vs ROI.

4. **Decisao estrategica:** Com dados API + analise robusta, decidir se H3B UP tem edge executavel e em quais condicoes (lag, liga, linha, pre/in-match).
