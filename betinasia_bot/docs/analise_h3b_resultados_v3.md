# Analise H3B — Resultados Completos (v3)

**Data:** 08 de Fevereiro de 2026
**Versao:** 3.0
**Status:** Analise com CLV adicional, ROI mediano e correcoes metodologicas

---

## 1. Resumo Executivo

A estrategia H3B (Reversao Temporal UP), analisada com dados de WebSocket e Betslip real, **nao apresenta edge executavel nas linhas liquidas (AH 0-1)** com a infraestrutura atual.

**Achados principais:**

| Metrica | Valor | Interpretacao |
|---------|-------|---------------|
| CLV Bruto WS Pre-Match | +0.210% (sig.) | H3B detecta odds acima da closing line |
| CLV Adicional WS Pre-Match | **-0.369%** (sig. neg.) | Apos subtrair baseline, edge desaparece |
| CLV Adicional BS Pre-Match | **-3.424%** (sig. neg.) | Com odd real do betslip, erosao e grande |
| ROI BS Pre-Match (media) | +12.43% (nao sig.) | Puxado por outliers AH 2+ |
| ROI BS Pre-Match (mediana) | **0.000%** | Metade das apostas da push — sem consistencia |
| ROI AH 0-1 (media) | **-2.44%** (nao sig.) | Linhas liquidas: negativo |
| ROI AH 0-1 (mediana) | **0.000%** | Sem edge nas linhas que importam |

**Conclusao:** O CLV bruto positivo (+0.210%) reflete um edge minimo no WebSocket, mas o **CLV adicional** (corrigido pelo baseline) e **negativo** (-0.369%). O sinal H3B UP nao tem valor incremental alem do drift natural do mercado. Com o betslip real (lag + erosao), o CLV adicional cai para -3.424%.

---

## 2. Correcoes Metodologicas (vs v2)

| Aspecto | v2 | v3 |
|---------|----|----|
| CLV | Bruto apenas | Bruto + Adicional (com baseline v6) |
| ROI | Media apenas | Media + Mediana + P25/P75 + Win Rate |
| Correlacao | Nao mencionada | Quantificada: 16.6 obs/jogo |
| CLV in-match | Reportado como valido | Corretamente excluido de conclusoes |

---

## 3. Dados e Filtros

| Metrica | Valor |
|---------|-------|
| Total auditorias H3B UP | 2.352 |
| Jogos unicos | 142 |
| Observacoes por jogo (media) | **16.6** |
| Betslip extraido (bruto) | 920 |
| Betslip filtrado (diff -10%/+10%) | 125 |
| Descartados | 795 (86%) |
| Pre-match | 1.007 |
| In-match | 760 |
| Com resultado (gols) | 2.125 |

**Problema de correlacao:** 2.352 observacoes em apenas 142 jogos (16.6 obs/jogo). Os intervalos de confianca estao subestimados. O N efetivo e ~142 (jogos), nao 2.352 (observacoes).

---

## 4. CLV WebSocket — Bruto vs Adicional

### 4.1. CLV Bruto (Pre-Match)

| Metrica | Valor |
|---------|-------|
| N | 928 |
| Media | +0.208% |
| Mediana | 0.000% |
| IC 90% | [+0.068%, +0.348%] |
| Significancia | POSITIVO |
| Win Rate CLV | 52.6% (441/397/90) |

### 4.2. CLV Adicional (Pre-Match, com baseline)

| Metrica | Valor |
|---------|-------|
| N | 928 |
| Media | **-0.369%** |
| Mediana | -0.230% |
| IC 90% | [-0.532%, -0.205%] |
| Significancia | **NEGATIVO** |
| Win Rate CLV | 44.8% (416/512/0) |

**Interpretacao critica:** O CLV bruto e +0.208% (positivo), mas o CLV adicional e -0.369% (negativo). Isto significa que o H3B UP detecta odds que estao acima da closing line, mas **nao mais do que a media das outras linhas do mesmo jogo no mesmo momento**. O "edge" e apenas drift do mercado, nao valor incremental.

**Comparacao com estudo anterior:** O v6 reportou CLV adicional de +1.116% (N=273). Agora com N=928 (3.4x mais dados), o CLV adicional e -0.369%. O resultado anterior era provavelmente ruido com amostra pequena.

---

## 5. CLV Betslip — Realista

### 5.1. CLV Bruto Betslip (Pre-Match)

| Metrica | Valor |
|---------|-------|
| N | 44 |
| Media | -2.244% |
| Mediana | -1.939% |
| IC 90% | [-3.291%, -1.196%] |
| Significancia | NEGATIVO |

### 5.2. CLV Adicional Betslip (Pre-Match)

| Metrica | Valor |
|---------|-------|
| N | 44 |
| Media | **-3.424%** |
| Mediana | -2.607% |
| IC 90% | [-4.846%, -2.003%] |
| Significancia | NEGATIVO |
| Win Rate CLV | 27.3% |

A odd real do betslip esta sistematicamente abaixo da closing line. Erosao media de 1.73% pelo lag (betslip vs websocket), mais 0.37% do baseline = 3.42% de CLV adicional negativo.

---

## 6. ROI Real — Media vs Mediana

### 6.1. ROI Pre-Match Betslip

| Metrica | Valor |
|---------|-------|
| N | 41 |
| Media | +12.43% |
| **Mediana** | **0.000%** |
| P25 / P75 | 0.000% / +78.200% |
| IC 90% | [-4.17%, +29.03%] |
| Significancia | Nao significativo |
| Win/Loss/Push | 11 / 6 / 24 |
| Win Rate | 64.7% |

### 6.2. ROI por Faixa de Linha AH

| Faixa | N | ROI Media | ROI Mediana | Win Rate |
|-------|---|-----------|-------------|----------|
| AH 0-1 (liquida) | 94 | **-2.44%** | **0.000%** | 50.0% |
| AH 2+ (extrema) | 9 | +105.11% | +102.90% | 100.0% |

**Achado fundamental:** O ROI medio geral (+12.43%) e inteiramente puxado por 9 apostas em linhas extremas (AH 2+) que tiveram 100% de win rate. Estas linhas tem:
- Odds altissimas (~2.0 em AH +2)
- Baixa liquidez
- Resultados atipicos (N=9 com 100% win e improvavel de se manter)

Nas linhas liquidas (AH 0-1), onde apostaramos na pratica: ROI = -2.44% (media) e 0.000% (mediana). **Sem edge.**

### 6.3. Paradoxo CLV Negativo + ROI Positivo (Resolvido)

Na v2, reportamos CLV negativo com ROI positivo — contradizendo a teoria de mercados eficientes. A explicacao:

1. **Outliers AH 2+:** 9 apostas com ROI +105% inflam a media para +12.4%
2. **Mediana = 0:** Metade das apostas da push, confirmando ausencia de edge consistente
3. **AH 0-1 (dados confiaveis):** CLV -3.34% E ROI -2.44% — **ambos negativos e consistentes**

O paradoxo desaparece quando olhamos mediana em vez de media, e quando separamos linhas liquidas de extremas.

---

## 7. Impacto do Lag

| Lag | N | Diff BS vs WS | CLV BS (PM) | ROI BS | Win Rate |
|-----|---|--------------|------------|--------|----------|
| < 10s | 24 | -1.19% | -0.52% (NS, N=3) | +10.60% (NS) | 57.1% |
| 10-20s | 81 | -2.62% | -2.36% (neg) | +13.57% (pos) | 70.8% |
| 20-30s | 14 | -4.32% | -4.68% (neg) | -17.31% (NS) | 42.9% |
| > 30s | 6 | -4.67% | N=1 | -6.85% (NS) | 50.0% |

**Padrao claro:** Quanto maior o lag, maior a erosao e pior o ROI. Lag > 20s e destrutivo. Lag < 10s mostra resultados mais promissores mas com N=3 para CLV (insuficiente).

O ROI positivo em 10-20s (+13.57%) e novamente puxado por outliers em linhas extremas.

---

## 8. Pre-Match vs In-Match

| Metrica | Pre-Match | In-Match |
|---------|-----------|----------|
| Diff BS vs WS | -1.73% | -3.11% |
| CLV BS | -2.24% (neg) | -1.75% (NS) |
| ROI BS (media) | +12.43% | +5.57% |
| ROI BS (mediana) | **0.000%** | **0.000%** |
| Win Rate ROI | 64.7% | 60.9% |

In-match tem erosao maior (-3.11% vs -1.73%) mas ambos tem mediana zero. Nenhum dos dois mostra edge consistente.

---

## 9. Espaco Amostral: BS vs WS

| Bucket | N | CLV BS (PM) | ROI BS | Win Rate |
|--------|---|------------|--------|----------|
| BS < WS (-10% a -2%) | 68 | -5.24% (neg) | +4.84% (NS) | 60.0% |
| BS ~ WS (-2% a +2%) | 48 | -1.35% (neg) | -6.42% (NS) | 40.0% |
| **BS > WS (+2% a +10%)** | **9** | **+3.76% (pos)** | **+94.14% (pos)** | **100.0%** |

O unico bucket com edge e BS > WS (betslip melhor que websocket). Mas N=9 com win rate 100% e estatisticamente nao confiavel. Se isto se mantiver com mais dados, seria um filtro poderoso para a estrategia.

---

## 10. Problemas Criticos

### 10.1. Correlacao Intra-Jogo (Nao Corrigida)

16.6 observacoes por jogo em media. Com 142 jogos unicos e 2.352 observacoes, os ICs estao severamente subestimados. O N efetivo para significancia e ~142, nao ~928.

**Correcao necessaria:** Erros clusterizados por match_id, ou 1 observacao por jogo.

### 10.2. Qualidade dos Dados de Betslip

86% dos dados descartados (diff fora de +-10%). A nova API (v4.0, validada hoje) resolve este problema — 100% dos dados sao confiaveis via JSON.

### 10.3. CLV Adicional Inverteu

O CLV adicional do WebSocket Pre-Match **inverteu** de +1.116% (N=273, estudo anterior) para **-0.369%** (N=928, estudo atual). Com 3.4x mais dados, o "edge" do H3B UP desaparece quando corrigido pelo baseline. O resultado anterior era provavelmente ruido estatistico.

### 10.4. AH 2+ (Anomalia)

Os 9 registros em AH 2+ com 100% de win rate e ROI +105% sao uma anomalia. Linhas extremas tem:
- Baixissima liquidez
- Odds imprecisas entre WS e betslip
- Resultados nao replicaveis

Nao devem ser base para decisoes.

---

## 11. Conclusoes

### 11.1. H3B UP como Estrategia Isolada: SEM EDGE

Com os dados atuais (N=928 pre-match, 142 jogos unicos):

- **CLV adicional WS: -0.369%** (significativamente negativo)
- **CLV adicional BS: -3.424%** (significativamente negativo)
- **ROI mediano: 0.000%** em todas as faixas relevantes
- **ROI AH 0-1: -2.44%** (linhas liquidas, onde apostaramos)

A estrategia H3B UP isolada **nao tem edge** nem no WebSocket (quando corrigido pelo baseline) nem no betslip.

### 11.2. H3B UP como Feature de Scoring: POSSIVEL

Apesar de nao ter edge isolado, H3B pode ser util como feature em um modelo que combine multiplos sinais:

- O sinal detecta momentos de movimentacao (reversao)
- Combinado com: diff BS vs WS, liga, line, pre/in-match, num bookmakers
- O bucket BS > WS (+2% a +10%) mostra potencial (N=9, precisa validacao)

### 11.3. H6 (Correlacao/Lag): INVESTIGAR

A hipotese H6 era a unica significativamente positiva no estudo original (CLV adicional +2.3% a +3.1%, N=388-391). Com a infraestrutura API agora operacional (~2.5s de lag), vale investigar H6 com dados de betslip real.

---

## 12. Recomendacoes

| Prioridade | Acao | Status |
|-----------|------|--------|
| 1 | Coletar dados via API betslip (v4.0) | Em producao |
| 2 | Investigar H6 com betslip (unica hipotese com edge significativo anterior) | Proximo passo |
| 3 | Modelo de scoring multi-feature (H3B + H6 + diff BS/WS + liga) | Medio prazo |
| 4 | Corrigir erros clusterizados por match_id | Proximo estudo |
| 5 | Acumular N=500+ via API para analise robusta | 1-2 semanas |
