# Simulação: Tamanho Potencial da Oportunidade H3B

**Data:** 06 de Fevereiro de 2026  
**Tipo:** Estimativa exploratória (alto grau de incerteza)

---

## 1. Premissas Conhecidas (dados reais)

| Parâmetro | Valor | Fonte |
|-----------|-------|-------|
| CLV adicional H3B UP (WebSocket) | +1.116% | Análise v6, N=273 |
| Eventos H3B UP detectados por dia | ~100-200 | Logs do audit |
| Taxa de execução (betslip extraído) | ~40% | Audit, 115/287 |
| Odds média no momento do sinal | ~1.85 | Dados audit |
| Lag médio detecção -> betslip | ~15-20s | Dados audit |
| Linhas extremas (|AH| > 2) na amostra | ~40% | Estimativa dos logs |

---

## 2. Cenários de CLV Realizável

O CLV de 1.116% é do WebSocket. Na prática, há erosão pelo lag e pela diferença betslip. Estimamos 3 cenários:

### Cenário A: Pessimista (erosão alta)
- CLV realizável com scoring: **+0.3%**
- Premissa: lag consome ~70% do valor, scoring recupera pouco

### Cenário B: Base (erosão moderada)
- CLV realizável com scoring: **+1.5%**
- Premissa: scoring filtra melhores sinais, lag consome ~30%, scoring adiciona valor

### Cenário C: Otimista (scoring eficaz + lag reduzido)
- CLV realizável com scoring: **+3.0%**
- Premissa: scoring seleciona top 20% dos sinais, otimização reduz lag para ~5s

---

## 3. Parâmetros da Simulação

### Volume de Apostas

| Parâmetro | Pessimista | Base | Otimista |
|-----------|-----------|------|----------|
| Sinais H3B UP / dia | 150 | 150 | 150 |
| Filtro linhas razoáveis (|AH| <= 1.5) | 60% | 60% | 60% |
| Sinais utilizáveis / dia | 90 | 90 | 90 |
| Taxa execução betslip | 40% | 40% | 50% |
| Filtro scoring (aprovados) | 50% | 30% | 20% |
| **Apostas executadas / dia** | **18** | **11** | **9** |
| **Apostas executadas / mês (30d)** | **540** | **330** | **270** |

### Stake e Banca

| Parâmetro | Valor |
|-----------|-------|
| Banca inicial | $5,000 |
| Stake por aposta (flat) | $50 (1% da banca) |
| Stake por aposta (Kelly fracionário) | Variável, ~0.5-2% da banca |

Para simplificar, usaremos **flat stake de $50** (1% da banca).

---

## 4. Resultados Estimados

### 4.1. Lucro por Aposta

O CLV representa a vantagem sobre a closing line. Em apostas Asian Handicap com odds ~1.85, o lucro esperado por aposta é:

```
Lucro esperado por aposta = Stake x CLV / 100

Onde CLV é a vantagem percentual sobre o mercado.
Nota: em AH com odds ~1.85, o vig (margem da casa) é ~2-3%.
CLV positivo significa que estamos do lado certo do vig.
```

| Cenário | CLV | Lucro/aposta ($50) | Lucro/aposta (%) |
|---------|-----|-------------------|-----------------|
| Pessimista | 0.3% | $0.15 | 0.3% |
| Base | 1.5% | $0.75 | 1.5% |
| Otimista | 3.0% | $1.50 | 3.0% |

### 4.2. Lucro Mensal (Flat Stake $50)

| Cenário | Apostas/mês | Lucro/aposta | **Lucro mensal** | **ROI por $** |
|---------|------------|-------------|-----------------|---------------|
| Pessimista | 540 | $0.15 | **$81** | **0.30%** |
| Base | 330 | $0.75 | **$248** | **1.50%** |
| Otimista | 270 | $1.50 | **$405** | **3.00%** |

### 4.3. ROI sobre Banca ($5,000)

| Cenário | Lucro mensal | **ROI mensal (banca)** | **ROI anual (banca)** |
|---------|-------------|----------------------|---------------------|
| Pessimista | $81 | **1.6%** | **19.4%** |
| Base | $248 | **5.0%** | **59.4%** |
| Otimista | $405 | **8.1%** | **97.2%** |

### 4.4. Turnover (Volume Apostado)

| Cenário | Apostas/mês | Stake | **Turnover mensal** | **Turnover anual** |
|---------|------------|-------|--------------------|--------------------|
| Pessimista | 540 | $50 | $27,000 | $324,000 |
| Base | 330 | $50 | $16,500 | $198,000 |
| Otimista | 270 | $50 | $13,500 | $162,000 |

---

## 5. Análise de Risco

### 5.1. Drawdown Esperado

Em apostas com edge pequeno e odds ~1.85, a variância é alta. Simulando com distribuição binomial:

| Parâmetro | Valor |
|-----------|-------|
| Probabilidade implícita (odds 1.85) | ~54% |
| Com CLV 1.5%, prob real estimada | ~55.5% |
| Desvio padrão por aposta ($50) | ~$49.70 |
| Desvio padrão mensal (330 apostas) | ~$903 |

```
Drawdown máximo esperado (95% confiança):
  = -2 x desvio_padrão_mensal
  = -2 x $903
  = -$1,806

Isso representa ~36% da banca de $5,000.
```

### 5.2. Risco de Ruína

| Banca | Stake (1%) | Risco de ruína (CLV 1.5%) |
|-------|-----------|--------------------------|
| $5,000 | $50 | Muito baixo (< 1%) |
| $2,000 | $50 (2.5%) | Baixo (~3-5%) |
| $1,000 | $50 (5%) | Moderado (~10-15%) |

**Recomendação:** Manter stake <= 1% da banca. Com $5,000 e stake de $50, o risco de ruína é negligível para CLV >= 1%.

### 5.3. Tempo para Lucro Estável

| Cenário | Apostas p/ 95% certeza de lucro |
|---------|-------------------------------|
| CLV 0.3% | ~44,000 apostas (~7 anos) |
| CLV 1.5% | ~1,800 apostas (~5 meses) |
| CLV 3.0% | ~440 apostas (~2 meses) |

Formula: N = (Z x sigma / CLV)^2, Z=1.96 para 95%.

---

## 6. Sensibilidade ao CLV

O CLV é a variável mais importante. Pequenas diferenças mudam tudo:

| CLV realizável | Lucro mensal ($50 flat) | ROI anual (banca $5k) | Viabilidade |
|---------------|------------------------|-----------------------|------------|
| 0.0% (sem edge) | $0 | 0% | Inviável |
| 0.3% | $81 | 19% | Marginal |
| 0.5% | $135 | 32% | Viável |
| 1.0% | $248 | 59% | Bom |
| 1.5% | $371 | 89% | Muito bom |
| 2.0% | $495 | 119% | Excelente |
| 3.0% | $743 | 178% | Excepcional |

---

## 7. Comparação com Benchmarks

| Estratégia | ROI típico | Volume necessário |
|-----------|-----------|-------------------|
| Apostador recreativo | -5% a -10% | Qualquer |
| Value betting manual | +1% a +3% | 500+ apostas/mês |
| **Nossa estimativa (base)** | **+1.5%** | **330 apostas/mês** |
| Fundos profissionais (Pinnacle) | +2% a +5% | 10,000+/mês |
| Arbitragem pura | +0.5% a +1.5% | 1,000+/mês |

A estratégia H3B com scoring estaria no range de **value betting**, o que é realista para uma operação semi-automatizada.

---

## 8. Fatores Não Modelados (riscos adicionais)

| Fator | Impacto | Probabilidade |
|-------|---------|---------------|
| Limites da casa (conta limitada) | Alto — reduz volume | Média-Alta |
| Mudança de estrutura do site | Alto — para operação | Baixa |
| Eficiência crescente do mercado | Médio — reduz CLV ao longo do tempo | Média |
| Custos VPS e API | Baixo — ~$30-50/mês | Certo |
| Erros de execução (clique errado, etc) | Baixo — perdas pontuais | Baixa |
| Correlação entre apostas | Médio — drawdowns maiores | Média |

---

## 9. Resumo: O Tamanho da Oportunidade

### Faixa Estimada (mensal, banca $5,000, stake $50)

| Métrica | Pessimista | Base | Otimista |
|---------|-----------|------|----------|
| **Lucro mensal** | $81 | $248 | $405 |
| **ROI por $ apostado** | 0.3% | 1.5% | 3.0% |
| **ROI sobre banca (mensal)** | 1.6% | 5.0% | 8.1% |
| **ROI sobre banca (anual)** | 19% | 59% | 97% |
| **Apostas por mês** | 540 | 330 | 270 |
| **Turnover mensal** | $27,000 | $16,500 | $13,500 |
| **Drawdown máx esperado** | -$1,200 | -$1,800 | -$1,500 |

### Interpretação

- **Se o scoring funcionar** (cenário base): ~$250/mês com risco controlado. É uma operação viável mas modesta.
- **Se otimizarmos velocidade + scoring** (cenário otimista): ~$400/mês, ROI quase 100% anual. Seria excelente.
- **Se o lag consumir o valor** (cenário pessimista): ~$80/mês. Marginal — pode não compensar o esforço.

### O Que Decide o Cenário

1. **CLV realizável com scoring** — Dado mais importante, ainda não temos. Precisamos de 2-4 semanas de dados.
2. **Otimização do lag** — Reduzir de 20s para 5s pode ser a diferença entre cenário pessimista e base.
3. **Qualidade do filtro de linhas** — Excluir linhas extremas e focar em mercados líquidos.

---

## 10. Escalabilidade

Se a estratégia se confirmar viável:

| Escala | Banca | Stake | Lucro mensal (base) | Considerações |
|--------|-------|-------|--------------------|----|
| Inicial | $5,000 | $50 | $248 | 1 conta, 1 VPS |
| Média | $15,000 | $150 | $743 | Possível limite de conta |
| Avançada | $50,000 | $500 | $2,475 | Múltiplas contas necessárias |

**Limitante principal:** Limites impostos pela casa de apostas. Com volume alto, contas tendem a ser limitadas em semanas/meses.
