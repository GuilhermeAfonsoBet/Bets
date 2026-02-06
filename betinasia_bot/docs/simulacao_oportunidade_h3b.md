# Simulacao: Tamanho Potencial da Oportunidade H3B

**Data:** 06 de Fevereiro de 2026  
**Versao:** 3.0  
**Tipo:** Estimativa exploratorio para mesa quant  
**Cambio:** USD 1 = BRL 5,30

**Nota sobre ROI:** Todos os ROIs apresentados sao lineares (nao capitalizados). Em operacao real com reinvestimento de lucros, o retorno composto seria superior.

---

## 1. Premissas Conhecidas (dados reais)

| Parametro | Valor | Fonte |
|-----------|-------|-------|
| CLV adicional H3B UP (WebSocket) | +1,116% | Analise v6, N=273 |
| Eventos H3B detectados por ciclo (~10s) | ~5-10 | Logs do audit |
| Ciclos por dia | ~8.640 (24h) | 10s por ciclo |
| Taxa de execucao (betslip extraido) | ~40% | Audit, 115/287 |
| Odds media no momento do sinal | ~1,85 | Dados audit |
| Lag medio deteccao -> betslip | ~15-20s | Dados audit |

### Volume Bruto Estimado

```
Eventos H3B por dia (observado):
  ~600 H3B em ~8 horas de monitoramento
  Extrapolando para 24h: ~1.800 H3B/dia
  ~50% sao UP: ~900 H3B UP/dia

Nota: o detector roda sobre TODOS os jogos de futebol
simultaneamente (WebSocket captura ~250 jogos por ciclo).
```

---

## 2. Parametros Operacionais

### Stake e Estrutura

| Parametro | Valor USD | Valor BRL |
|-----------|-----------|-----------|
| Stake por aposta (flat) | USD 500 | R$ 2.650 |
| Odds media | 1,85 | 1,85 |
| Mercado | Asian Handicap | Asian Handicap |
| Operacao | 24/7 automatizada via VPS | 24/7 automatizada via VPS |

### Funil de Sinais (3 cenarios)

| Etapa do Funil | Pessimista | Base | Otimista |
|----------------|-----------|------|----------|
| Sinais H3B UP brutos / dia | 900 | 900 | 900 |
| Filtro linhas razoaveis (excl. AH > 2) | 60% | 60% | 60% |
| Sinais filtrados / dia | 540 | 540 | 540 |
| Taxa de execucao betslip | 40% | 50% | 60% |
| Sinais executaveis / dia | 216 | 270 | 324 |
| Filtro scoring (aprovados) | 60% | 50% | 40% |
| Apostas executadas / dia | 130 | 135 | 130 |
| Apostas executadas / mes (30d) | 3.900 | 4.050 | 3.900 |

---

## 3. Cenarios de CLV Realizavel

| Cenario | CLV com Scoring | Premissa |
|---------|----------------|----------|
| Pessimista | +0,5% | Lag consome parte do valor, scoring filtra pouco |
| Base | +1,5% | Scoring seleciona bons sinais, lag parcialmente mitigado |
| Otimista | +3,0% | Scoring eficaz + lag reduzido a ~5s + linhas liquidas |

---

## 4. Resultados Estimados

### 4.1. Lucro por Aposta

| Cenario | CLV | Stake | Lucro/aposta USD | Lucro/aposta BRL |
|---------|-----|-------|-----------------|-----------------|
| Pessimista | 0,5% | USD 500 | USD 2,50 | R$ 13,25 |
| Base | 1,5% | USD 500 | USD 7,50 | R$ 39,75 |
| Otimista | 3,0% | USD 500 | USD 15,00 | R$ 79,50 |

### 4.2. Resultados Mensais

| Cenario | Apostas/mes | Turnover mensal | Lucro mensal USD | Lucro mensal BRL | ROI por USD |
|---------|------------|----------------|-----------------|-----------------|-------------|
| Pessimista | 3.900 | USD 1,95M | USD 9.750 | R$ 51.675 | 0,50% |
| Base | 4.050 | USD 2,03M | USD 30.375 | R$ 160.988 | 1,50% |
| Otimista | 3.900 | USD 1,95M | USD 58.500 | R$ 310.050 | 3,00% |

### 4.3. Resultados Anuais (linear, sem capitalizacao)

| Cenario | Lucro mensal | Lucro anual USD | Lucro anual BRL | Turnover anual |
|---------|-------------|----------------|----------------|----------------|
| Pessimista | USD 9.750 | USD 117.000 | R$ 620.100 | USD 23,4M |
| Base | USD 30.375 | USD 364.500 | R$ 1.931.850 | USD 24,3M |
| Otimista | USD 58.500 | USD 702.000 | R$ 3.720.600 | USD 23,4M |

---

## 5. Banca Necessaria (output)

### 5.1. Calculo da Variancia

```
Para cada aposta AH com odds ~1,85:
  Prob vitoria ~ 55% (com CLV 1,5%)
  Variancia por aposta = p(1-p) x stake^2 x odds^2
  sigma por aposta ~ USD 495

Para N apostas/mes:
  sigma mensal = sigma_aposta x raiz(N)

  Pessimista (3.900): sigma_mensal = USD 495 x raiz(3.900) = USD 30.900
  Base (4.050):       sigma_mensal = USD 495 x raiz(4.050) = USD 31.500
  Otimista (3.900):   sigma_mensal = USD 495 x raiz(3.900) = USD 30.900
```

### 5.2. Drawdown Maximo Esperado

| Cenario | sigma mensal | Drawdown 95% | Drawdown 99% |
|---------|-------------|-------------|-------------|
| Pessimista | USD 30.900 | USD -52.050 | USD -67.500 |
| Base | USD 31.500 | USD -32.625 | USD -48.375 |
| Otimista | USD 30.900 | USD -3.300 | USD -19.050 |

Nota: Drawdown 95% = lucro esperado - 2 sigma.

### 5.3. Banca Recomendada

Criterio: banca deve suportar 3 meses de drawdown 95% sem ruir.

| Cenario | Drawdown 95% mensal | Banca minima (3 meses) USD | Banca recomendada USD | Banca recomendada BRL |
|---------|---------------------|---------------------------|----------------------|-----------------------|
| Pessimista | USD -52.050 | USD 156.150 | USD 200.000 | R$ 1.060.000 |
| Base | USD -32.625 | USD 97.875 | USD 125.000 | R$ 662.500 |
| Otimista | USD -3.300 | USD 9.900 | USD 50.000 | R$ 265.000 |

### 5.4. ROI sobre Banca (linear, sem capitalizacao)

| Cenario | Banca USD | Banca BRL | Lucro mensal | ROI mensal | ROI anual (linear) |
|---------|-----------|-----------|-------------|-----------|-------------------|
| Pessimista | USD 200.000 | R$ 1.060.000 | USD 9.750 | 4,9% | 58,5% |
| Base | USD 125.000 | R$ 662.500 | USD 30.375 | 24,3% | 291,6% |
| Otimista | USD 50.000 | R$ 265.000 | USD 58.500 | 117,0% | 1.404% |

Nota: ROI anual linear = ROI mensal x 12. Com reinvestimento (capitalizado), o retorno composto no cenario base seria (1,243)^12 - 1 = 1.250%, significativamente superior ao linear.

---

## 6. Sensibilidade ao CLV

O CLV realizavel e a variavel que determina tudo. Com 4.050 apostas/mes e stake USD 500:

| CLV | Lucro mensal USD | Lucro mensal BRL | Lucro anual USD | Lucro anual BRL | Banca USD | ROI anual |
|-----|-----------------|-----------------|----------------|----------------|-----------|-----------|
| 0,0% | 0 | 0 | 0 | 0 | N/A | 0% |
| 0,3% | 6.075 | 32.198 | 72.900 | 386.370 | 220.000 | 33% |
| 0,5% | 10.125 | 53.663 | 121.500 | 643.950 | 200.000 | 61% |
| 1,0% | 20.250 | 107.325 | 243.000 | 1.287.900 | 150.000 | 162% |
| 1,5% | 30.375 | 160.988 | 364.500 | 1.931.850 | 125.000 | 292% |
| 2,0% | 40.500 | 214.650 | 486.000 | 2.575.800 | 100.000 | 486% |
| 3,0% | 60.750 | 321.975 | 729.000 | 3.863.700 | 50.000 | 1.458% |
| 5,0% | 101.250 | 536.625 | 1.215.000 | 6.439.500 | 30.000 | 4.050% |

---

## 7. Sensibilidade ao Volume de Apostas

Com CLV 1,5% e stake USD 500:

| Apostas/mes | Lucro mensal USD | Lucro mensal BRL | Lucro anual USD | Lucro anual BRL | Banca USD |
|------------|-----------------|-----------------|----------------|----------------|-----------|
| 1.000 | 7.500 | 39.750 | 90.000 | 477.000 | 60.000 |
| 2.000 | 15.000 | 79.500 | 180.000 | 954.000 | 85.000 |
| 3.000 | 22.500 | 119.250 | 270.000 | 1.431.000 | 100.000 |
| 4.000 | 30.000 | 159.000 | 360.000 | 1.908.000 | 125.000 |
| 6.000 | 45.000 | 238.500 | 540.000 | 2.862.000 | 155.000 |
| 10.000 | 75.000 | 397.500 | 900.000 | 4.770.000 | 200.000 |

---

## 8. Comparacao com Benchmarks e Analise Competitiva

### 8.1. Tipos de Operacao no Mercado de Apostas Esportivas

**Market Making Esportivo**

Operacao que fornece liquidez ao mercado, atuando em ambos os lados de uma aposta (back e lay) simultaneamente, lucrando com o spread. Requer infraestrutura de baixa latencia, capital significativo e acesso a exchanges (Betfair, Smarkets). Players vencedores possuem: infraestrutura de co-location, modelos proprietarios de pricing, capacidade de ajustar spreads em milissegundos, e bancas acima de USD 500k. ROI tipico: 0,5-2% sobre turnover. Lucro anual: USD 200k-2M.

**Value Betting (Sharp)**

Identificacao sistematica de odds que estao acima do valor justo, comparando com linhas de referencia (Pinnacle, mercado de consenso). O apostador sharp aposta ANTES de a linha corrigir, capturando CLV positivo. Players vencedores possuem: modelos de probabilidade proprios, automacao de deteccao e execucao, disciplina de banca rigorosa, e acesso a multiplas casas/brokers. ROI tipico: 1-3% sobre turnover. Lucro anual: USD 100k-500k.

**Arbitragem Pura**

Exploracao de diferencas de odds entre casas de apostas para garantir lucro independente do resultado. Requer velocidade extrema (odds de arbitragem duram segundos) e acesso a muitas contas. Players vencedores possuem: software especializado (RebelBetting, BetBurger), dezenas de contas em casas diferentes, e alta tolerancia a limitacao de contas. ROI tipico: 0,3-1% sobre turnover. Lucro anual: USD 50k-300k.

**CLV Betting (Pinnacle)**

Estrategia focada exclusivamente em bater a closing line da Pinnacle, considerada o mercado mais eficiente. Apostadores que consistentemente obtêm odds melhores que a closing line tem edge comprovado. Players vencedores possuem: modelos sofisticados de previsao, execucao antes do kickoff, acesso direto a Pinnacle. ROI tipico: 2-5% sobre turnover. Lucro anual: USD 500k-2M+.

### 8.2. Onde Nossa Operacao Se Posiciona

| Operacao | ROI turnover | Lucro anual tipico | Capital | Nossa posicao |
|----------|-------------|-------------------|---------|---------------|
| Market making esportivo | 0,5-2% | USD 200k-2M | USD 100k-500k | Nao competimos aqui |
| Value betting (sharp) | 1-3% | USD 100k-500k | USD 50k-200k | Posicao central estimada |
| Arbitragem pura | 0,3-1% | USD 50k-300k | USD 50k-200k | Nao e arbitragem |
| CLV betting (Pinnacle) | 2-5% | USD 500k-2M+ | USD 200k+ | Aspiracional |
| H3B + Scoring (base) | 1,5% | USD 364k | USD 125k | Dentro do range |

### 8.3. Nossas Forcas e Fraquezas

| Dimensao | Status Atual | Avaliacao | Acao Necessaria |
|----------|-------------|-----------|-----------------|
| Deteccao de sinais (WebSocket) | Operacional, 24/7 | FORTE | Manter e expandir |
| Velocidade de execucao | ~15-20s de lag | FRACA | Otimizar para menos de 5s |
| Modelo de scoring | Ainda nao existe | A CONSTRUIR | Prioridade #1 |
| Automacao end-to-end | Parcial (coleta OK, execucao manual) | MEDIA | Automatizar execucao |
| Infraestrutura (VPS) | Funcional | FORTE | Adequada |
| Acesso ao mercado (BetinAsia) | 1 conta ativa | MEDIA | Avaliar contas adicionais |
| Capital disponivel | A definir | N/A | Depende de investidores |
| Dados historicos | ~2 semanas | FRACA (crescendo) | Acumular mais 4-8 semanas |
| Analise estatistica | Metodologia solida | FORTE | Manter rigor |
| Diversificacao de esportes | Apenas futebol | MEDIA | Expandir para basquete, tenis |

### 8.4. O Que Precisamos para Competir com Sharp Value Bettors

1. **Scoring model robusto** - Hoje temos o sinal bruto (H3B). Sharps combinam multiplos sinais.
2. **Execucao em menos de 5 segundos** - Sharps executam em 1-3s via APIs ou pre-posicionamento.
3. **Multiplas fontes de odds** - Nao depender apenas de BetinAsia.
4. **Disciplina de banca** - Gestao profissional com Kelly fracionario e limits.
5. **Dados de pelo menos 3 meses** - Para significancia estatistica robusta.

---

## 9. O Que Precisa Ser Verdade (Analise Detalhada de Riscos)

Para o cenario base (USD 364k/ano, R$ 1,93M/ano) se materializar, quatro premissas devem ser verdadeiras. Abaixo, analise de cada uma com grau de confianca, acoes e riscos.

### 9.1. CLV realizavel com scoring >= 1,5%

| Aspecto | Avaliacao |
|---------|-----------|
| Grau de confianca | 40-50% |
| Justificativa | CLV bruto WebSocket e 1,116% (quase significativo). Scoring pode amplificar filtrando melhores sinais, mas tambem pode nao adicionar valor suficiente |
| Risco principal | O lag de 15-20s pode consumir todo o edge, resultando em CLV realizavel proximo a zero |
| Melhor cenario | Scoring identifica subconjunto com CLV 3%+ filtrando linhas, ligas e momentos |
| Pior cenario | Diferenca betslip vs websocket e sistematicamente negativa, eliminando o valor |
| Acoes para mitigar | (1) Acumular 500+ auditorias com closing line; (2) Teste de velocidade 5s; (3) Modelo de scoring com features combinadas |
| Prazo para validacao | 4-6 semanas |

### 9.2. Volume de 4.000+ apostas/mes sustentavel

| Aspecto | Avaliacao |
|---------|-----------|
| Grau de confianca | 60-70% |
| Justificativa | WebSocket ja detecta ~900 H3B UP/dia. O funil de 40-50% execucao e 50% scoring e conservador. Volume bruto existe |
| Risco principal | Filtro de scoring pode ser mais restritivo que 50%, ou linhas de alta liquidez podem ser poucas |
| Melhor cenario | Com expansao para mais mercados e esportes, volume pode triplicar |
| Pior cenario | Scoring aprovaria apenas 10-20% (e nao 50%), reduzindo para 1.000-2.000 apostas/mes |
| Acoes para mitigar | (1) Monitorar mais esportes; (2) Adicionar mercados 1X2 e BTTS; (3) Multiplas sessoes de WebSocket |
| Prazo para validacao | 2-3 semanas |

### 9.3. Contas nao limitadas rapidamente

| Aspecto | Avaliacao |
|---------|-----------|
| Grau de confianca | 70-80% |
| Justificativa | BetinAsia e um BROKER profissional, nao uma casa de apostas varejista. Brokers existem para servir apostadores sharp e lucram com comissao sobre volume, nao com a perda do cliente. Diferente de Bet365, Betano etc, BetinAsia nao limita por ser vencedor |
| Risco principal | Mesmo brokers podem ter limites de liquidez por mercado/evento. Nao e limitacao de conta, mas limitacao de MERCADO |
| Melhor cenario | Volume de USD 2M/mes absorvido sem problemas em ligas Tier 1 |
| Pior cenario | Limites de USD 200-500 por aposta em ligas menores reduzem turnover efetivo |
| Acoes para mitigar | (1) Focar em ligas Tier 1 (Premier League, La Liga, etc) com maior liquidez; (2) Distribuir apostas ao longo do dia; (3) Considerar segundo broker |
| Prazo para validacao | Verificavel no go-live |

**Nota importante sobre BetinAsia:** Diferente de casas de apostas tradicionais (que lucram com a perda do apostador e limitam vencedores), o BetinAsia e um broker/agregador. Ele agrega odds de multiplos bookmakers (Pinnacle, SBO, ISN, etc) e cobra comissao sobre o volume. O modelo de negocio do BetinAsia e ALINHADO com apostadores de alto volume — quanto mais voce aposta, mais ele ganha. Por isso, o risco de limitacao de conta e significativamente menor do que em casas tradicionais. Porem, os limites de aposta por evento dependem da liquidez oferecida pelos bookmakers subjacentes.

### 9.4. Mercado nao se torna eficiente demais nos proximos 12 meses

| Aspecto | Avaliacao |
|---------|-----------|
| Grau de confianca | 75-85% |
| Justificativa | Ineficiencias em odds existem ha decadas e persistem porque: (1) diferentes bookmakers tem modelos diferentes, (2) informacao assimetrica sempre existira, (3) o mercado de apostas nao e tao eficiente quanto mercados financeiros. A hipotese H3B explora reversoes temporais — um fenomeno estrutural |
| Risco principal | Se muitos players descobrirem e explorarem o mesmo sinal, as reversoes serao arbitradas mais rapidamente |
| Melhor cenario | Nicho permanece pouco explorado e edge persiste por anos |
| Pior cenario | Plataformas de bet analytics popularizam a estrategia e edge cai para 0,3% em 6 meses |
| Acoes para mitigar | (1) Diversificar estrategias (H6 ja e significativo); (2) Evoluir scoring continuamente; (3) Manter vantagem de velocidade |
| Prazo para validacao | Monitoramento continuo |

### Resumo de Riscos

| Premissa | Confianca | Impacto se falhar | Risco geral |
|----------|-----------|-------------------|-------------|
| CLV >= 1,5% com scoring | 40-50% | Critico | ALTO |
| Volume 4.000+/mes | 60-70% | Moderado | MEDIO |
| Contas nao limitadas | 70-80% | Moderado | BAIXO |
| Mercado ineficiente 12 meses | 75-85% | Gradual | BAIXO |

O risco dominante e o item 1 (CLV realizavel). E a variavel que ainda nao temos dado para validar.

---

## 10. Fatores Multiplicadores de Escala

### Formas de Aumentar o Volume

| Alavanca | Impacto estimado | Complexidade |
|----------|-----------------|-------------|
| Adicionar mais esportes (basquete, tenis) | +50-100% sinais | Media |
| Multiplas contas BetinAsia | +100% execucao | Alta (custo + risco) |
| Outros agregadores (alem do BetinAsia) | +200-500% sinais | Alta |
| Mercados adicionais (1X2, BTTS) | +30-50% sinais | Baixa |
| Operacao 24/7 em multiplos fusos | +20-40% sinais | Baixa (ja e 24/7) |

---

## 11. Resumo Executivo

### A Oportunidade

| Metrica | Pessimista | Base | Otimista |
|---------|-----------|------|----------|
| Lucro mensal USD | USD 9.750 | USD 30.375 | USD 58.500 |
| Lucro mensal BRL | R$ 51.675 | R$ 160.988 | R$ 310.050 |
| Lucro anual USD (linear) | USD 117.000 | USD 364.500 | USD 702.000 |
| Lucro anual BRL (linear) | R$ 620.100 | R$ 1.931.850 | R$ 3.720.600 |
| ROI por USD apostado | 0,50% | 1,50% | 3,00% |
| ROI sobre banca (anual, linear) | 59% | 292% | 1.404% |
| Apostas por mes | 3.900 | 4.050 | 3.900 |
| Turnover mensal | USD 1,95M | USD 2,03M | USD 1,95M |
| Banca recomendada USD | USD 200.000 | USD 125.000 | USD 50.000 |
| Banca recomendada BRL | R$ 1.060.000 | R$ 662.500 | R$ 265.000 |
| Stake por aposta | USD 500 | USD 500 | USD 500 |

Nota: ROI anual linear = ROI mensal x 12 meses. Nao inclui capitalizacao (reinvestimento de lucros).

### Proximos Passos para Validacao

| Etapa | Prazo | Objetivo |
|-------|-------|----------|
| Coleta de dados audit H3B | 2-4 semanas | N>500 com closing line para CLV Betslip |
| Modelo de scoring | 4-6 semanas | Estimar CLV realizavel com features combinadas |
| Teste de velocidade (1 liga) | 1-2 semanas | Medir impacto de reduzir lag para ~5s |
| Dry-run automatizado | 6-8 semanas | Simular apostas reais em tempo real |
| Go-live | 8-12 semanas | Operacao real com capital |

### O Risco Central

O maior risco (e maior incognita) e o CLV realizavel com scoring. Temos confianca de 40-50% de que sera >= 1,5%. As proximas 4-6 semanas de coleta de dados vao transformar esta estimativa em dado concreto. Todo o restante (volume, contas, mercado) tem risco controlavel.
