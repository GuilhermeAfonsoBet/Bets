# Simulação: Tamanho Potencial da Oportunidade H3B

**Data:** 06 de Fevereiro de 2026  
**Versão:** 2.0  
**Tipo:** Estimativa exploratória para mesa quant

---

## 1. Premissas Conhecidas (dados reais)

| Parâmetro | Valor | Fonte |
|-----------|-------|-------|
| CLV adicional H3B UP (WebSocket) | +1.116% | Análise v6, N=273 |
| Eventos H3B detectados por ciclo (~10s) | ~5-10 | Logs: ~600 em ~60 ciclos |
| Ciclos por dia | ~8,640 (24h) | 10s por ciclo |
| Taxa de execução (betslip extraído) | ~40% | Audit, 115/287 |
| Odds média no momento do sinal | ~1.85 | Dados audit |
| Lag médio detecção -> betslip | ~15-20s | Dados audit |

### Volume Bruto Estimado

```
Eventos H3B por dia (observado):
  ~600 H3B em ~8 horas de monitoramento
  Extrapolando para 24h: ~1,800 H3B/dia
  ~50% são UP: ~900 H3B UP/dia

Nota: o detector roda sobre TODOS os jogos de futebol
simultaneamente (WebSocket captura ~250 jogos por ciclo).
```

---

## 2. Parâmetros Operacionais

### Stake e Estrutura

| Parâmetro | Valor |
|-----------|-------|
| Stake por aposta (flat) | $500 |
| Odds média | 1.85 |
| Mercado | Asian Handicap (principalmente) |
| Operação | 24/7 automatizada via VPS |

### Funil de Sinais (3 cenários)

| Etapa do Funil | Pessimista | Base | Otimista |
|----------------|-----------|------|----------|
| Sinais H3B UP brutos / dia | 900 | 900 | 900 |
| Filtro linhas razoáveis (excl. AH > 2) | 60% | 60% | 60% |
| Sinais filtrados / dia | 540 | 540 | 540 |
| Taxa de execução betslip | 40% | 50% | 60% |
| Sinais executáveis / dia | 216 | 270 | 324 |
| Filtro scoring (aprovados) | 60% | 50% | 40% |
| **Apostas executadas / dia** | **130** | **135** | **130** |
| **Apostas executadas / mês (30d)** | **3,900** | **4,050** | **3,900** |

**Racional do filtro scoring:**
- Pessimista: scoring permissivo (60%), aceita mais apostas com edge menor
- Base: scoring moderado (50%), equilibra volume e qualidade
- Otimista: scoring restritivo (40%), apenas melhores sinais mas com CLV mais alto

---

## 3. Cenários de CLV Realizável

| Cenário | CLV com Scoring | Premissa |
|---------|----------------|----------|
| Pessimista | +0.5% | Lag consome parte do valor, scoring filtra pouco |
| Base | +1.5% | Scoring seleciona bons sinais, lag parcialmente mitigado |
| Otimista | +3.0% | Scoring eficaz + lag reduzido a ~5s + linhas líquidas |

---

## 4. Resultados Estimados

### 4.1. Lucro por Aposta

| Cenário | CLV | Stake | Lucro esperado/aposta |
|---------|-----|-------|----------------------|
| Pessimista | 0.5% | $500 | $2.50 |
| Base | 1.5% | $500 | $7.50 |
| Otimista | 3.0% | $500 | $15.00 |

### 4.2. Resultados Mensais

| Cenário | Apostas/mês | Lucro/aposta | Turnover mensal | **Lucro mensal** | **ROI por $** |
|---------|------------|-------------|----------------|-----------------|---------------|
| Pessimista | 3,900 | $2.50 | $1,950,000 | **$9,750** | **0.50%** |
| Base | 4,050 | $7.50 | $2,025,000 | **$30,375** | **1.50%** |
| Otimista | 3,900 | $15.00 | $1,950,000 | **$58,500** | **3.00%** |

### 4.3. Resultados Anuais

| Cenário | Lucro mensal | **Lucro anual** | Turnover anual |
|---------|-------------|----------------|----------------|
| Pessimista | $9,750 | **$117,000** | $23,400,000 |
| Base | $30,375 | **$364,500** | $24,300,000 |
| Otimista | $58,500 | **$702,000** | $23,400,000 |

---

## 5. Banca Necessária (output)

A banca mínima é determinada pelo drawdown máximo esperado e a tolerância a risco.

### 5.1. Cálculo da Variância

```
Para cada aposta AH com odds ~1.85:
  Prob vitória ≈ 55% (com CLV 1.5%)
  Variância por aposta = p(1-p) × stake² × odds²
  σ por aposta ≈ $495

Para N apostas/mês:
  σ mensal = σ_aposta × √N

  Pessimista (3,900): σ_mensal = $495 × √3,900 = $30,900
  Base (4,050):       σ_mensal = $495 × √4,050 = $31,500
  Otimista (3,900):   σ_mensal = $495 × √3,900 = $30,900
```

### 5.2. Drawdown Máximo Esperado

| Cenário | σ mensal | Drawdown 95% (-2σ) | Drawdown 99% (-2.5σ) |
|---------|---------|--------------------|-----------------------|
| Pessimista | $30,900 | -$52,050 | -$67,500 |
| Base | $31,500 | -$32,625 | -$48,375 |
| Otimista | $30,900 | -$3,300 | -$19,050 |

**Nota:** Drawdown 95% = lucro esperado - 2σ. Mês pessimista com CLV 0.5%: $9,750 - 2×$30,900 = -$52,050.

### 5.3. Banca Recomendada

Critério: banca deve suportar 3 meses de drawdown 95% sem ruir.

| Cenário | Drawdown 95% mensal | **Banca mínima (3 meses)** | **Banca recomendada (margem)** |
|---------|---------------------|---------------------------|-------------------------------|
| Pessimista | -$52,050 | $156,150 | **$200,000** |
| Base | -$32,625 | $97,875 | **$125,000** |
| Otimista | -$3,300 | $9,900 | **$50,000** |

### 5.4. ROI sobre Banca

| Cenário | Banca recomendada | Lucro mensal | **ROI mensal** | **ROI anual** |
|---------|-------------------|-------------|---------------|--------------|
| Pessimista | $200,000 | $9,750 | 4.9% | 58.5% |
| Base | $125,000 | $30,375 | **24.3%** | **291.6%** |
| Otimista | $50,000 | $58,500 | **117.0%** | **1,404%** |

---

## 6. Sensibilidade ao CLV

O CLV realizável é a variável que determina tudo. Com 4,050 apostas/mês e stake $500:

| CLV | Lucro mensal | Lucro anual | Banca necessária | ROI anual (banca) |
|-----|-------------|-------------|------------------|-------------------|
| 0.0% | $0 | $0 | N/A | 0% |
| 0.3% | $6,075 | $72,900 | $220,000 | 33% |
| 0.5% | $10,125 | $121,500 | $200,000 | 61% |
| 1.0% | $20,250 | $243,000 | $150,000 | 162% |
| **1.5%** | **$30,375** | **$364,500** | **$125,000** | **292%** |
| 2.0% | $40,500 | $486,000 | $100,000 | 486% |
| 3.0% | $60,750 | $729,000 | $50,000 | 1,458% |
| 5.0% | $101,250 | $1,215,000 | $30,000 | 4,050% |

---

## 7. Sensibilidade ao Volume de Apostas

Com CLV 1.5% e stake $500:

| Apostas/mês | Lucro mensal | Lucro anual | σ mensal | Banca necessária |
|------------|-------------|-------------|---------|------------------|
| 1,000 | $7,500 | $90,000 | $15,650 | $60,000 |
| 2,000 | $15,000 | $180,000 | $22,130 | $85,000 |
| 3,000 | $22,500 | $270,000 | $27,100 | $100,000 |
| **4,000** | **$30,000** | **$360,000** | **$31,300** | **$125,000** |
| 6,000 | $45,000 | $540,000 | $38,350 | $155,000 |
| 10,000 | $75,000 | $900,000 | $49,500 | $200,000 |

---

## 8. Comparação com Benchmarks (mesa quant)

| Operação | ROI sobre turnover | Lucro anual típico | Capital necessário |
|----------|-------------------|-------------------|--------------------|
| Market making esportivo | 0.5-2% | $200k-$2M | $100k-$500k |
| Value betting (sharp) | 1-3% | $100k-$500k | $50k-$200k |
| Arbitragem pura | 0.3-1% | $50k-$300k | $50k-$200k |
| **H3B + Scoring (est. base)** | **1.5%** | **$364k** | **$125k** |
| CLV betting (Pinnacle) | 2-5% | $500k-$2M+ | $200k+ |

A estimativa base situa a operação H3B no range de uma **operação de value betting sharp**, que é realista e comparável ao mercado.

---

## 9. Fatores Multiplicadores de Escala

### Formas de Aumentar o Volume

| Alavanca | Impacto estimado | Complexidade |
|----------|-----------------|-------------|
| Adicionar mais esportes (basquete, tênis) | +50-100% sinais | Média |
| Múltiplas contas BetinAsia | +100% execução | Alta (custo + risco) |
| Outros agregadores (além do BetinAsia) | +200-500% sinais | Alta |
| Mercados adicionais (1X2, BTTS) | +30-50% sinais | Baixa |
| Operação 24/7 em múltiplos fusos | +20-40% sinais | Baixa (já é 24/7) |

### Limites de Escala

| Fator | Limite |
|-------|--------|
| Limite por conta BetinAsia | ~$500-$2,000/aposta dependendo do mercado |
| Risco de conta limitada | Alto após $50k+ turnover/mês em 1 conta |
| Liquidez do mercado AH | Limitada em ligas menores |
| Capacidade computacional | Baixa (1 VPS é suficiente) |

---

## 10. Resumo Executivo para Investidores

### A Oportunidade

| Métrica | Pessimista | Base | Otimista |
|---------|-----------|------|----------|
| **Lucro mensal** | $9,750 | $30,375 | $58,500 |
| **Lucro anual** | $117,000 | $364,500 | $702,000 |
| **ROI por $ apostado** | 0.50% | 1.50% | 3.00% |
| **ROI sobre banca (anual)** | 59% | 292% | 1,404% |
| **Apostas por mês** | 3,900 | 4,050 | 3,900 |
| **Turnover mensal** | $1.95M | $2.03M | $1.95M |
| **Banca recomendada** | $200,000 | $125,000 | $50,000 |
| **Stake por aposta** | $500 | $500 | $500 |

### Próximos Passos para Validação

| Etapa | Prazo | Objetivo |
|-------|-------|----------|
| Coleta de dados audit H3B | 2-4 semanas | N>500 com closing line para CLV Betslip |
| Modelo de scoring | 4-6 semanas | Estimar CLV realizável com features combinadas |
| Teste de velocidade (1 liga) | 1-2 semanas | Medir impacto de reduzir lag para ~5s |
| Dry-run automatizado | 6-8 semanas | Simular apostas reais em tempo real |
| Go-live | 8-12 semanas | Operação real com capital |

### O Que Precisa Ser Verdade

Para o cenário base ($364k/ano) se materializar:

1. CLV realizável com scoring >= 1.5%
2. Volume de 4,000+ apostas/mês sustentável
3. Contas não limitadas rapidamente
4. Mercado não se torna eficiente demais nos próximos 12 meses
