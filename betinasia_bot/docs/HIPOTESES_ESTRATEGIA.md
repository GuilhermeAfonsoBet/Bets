# Hipóteses de Estratégia - BetinAsia Bot

**Data:** Fevereiro 2026  
**Versão:** 3.0  
**Status:** Em validação

---

## Visão Geral

Este documento detalha as hipóteses de estratégia que estamos testando para identificar oportunidades de valor em apostas esportivas. Cada hipótese representa uma possível ineficiência de mercado que pode ser explorada.

### Como Validamos uma Hipótese

Para cada hipótese, coletamos eventos (momentos onde a condição da hipótese é detectada) e depois medimos:

1. **CLV (Closing Line Value):** A odd no momento do evento vs a odd de fechamento
   - CLV positivo = conseguimos uma odd melhor que o mercado final
   - CLV consistentemente positivo = indica edge real

2. **ROI:** Retorno sobre investimento se apostássemos em todos os eventos
   - ROI > 0 com significância estatística = hipótese válida

3. **Significância Estatística:** Quantidade de eventos suficiente para conclusões
   - Mínimo ~100 eventos por segmento
   - p-value < 0.05

---

## H1: Precificação Incorreta

### O que é?

Detecta inconsistências na precificação de mercados, incluindo:
- **Arbitragem:** Overround < 100% (lucro garantido)
- **Mispricing simples:** Odd de um lado significativamente acima da odd justa
- **Mispricing cruzado:** Inconsistências entre diferentes tipos de mercado (1X2, AH, OU)

### Cálculo do Overround

**Para mercados de 2 outcomes (AH, OU):**
```
Overround = (1/odd_home) + (1/odd_away)

Exemplo AH -0.5:
- Home: 2.10 → 1/2.10 = 0.476
- Away: 1.85 → 1/1.85 = 0.541
- Overround = 0.476 + 0.541 = 1.017 (101.7%)
```

**Para mercado 1X2 (3 outcomes):**
```
Overround = (1/odd_home) + (1/odd_draw) + (1/odd_away)

Exemplo:
- Home: 2.50 → 1/2.50 = 0.400
- Draw: 3.20 → 1/3.20 = 0.313
- Away: 2.90 → 1/2.90 = 0.345
- Overround = 0.400 + 0.313 + 0.345 = 1.058 (105.8%)
```

**NOTA:** Nossa implementação atual EXCLUI o mercado 1X2 da detecção H1 porque não coletamos a odd de empate no mesmo registro que Home/Away. Isso será corrigido em versão futura.

### Tipos de Mispricing

#### 1. Arbitragem (Implementado)
Overround < 100% significa lucro garantido apostando em ambos os lados.

```
Exemplo de arbitragem:
- Home: 2.20 → 45.5%
- Away: 1.90 → 52.6%
- TOTAL: 98.1% → ARBITRAGEM!

Aposta proporcional:
- R$100 em Home a 2.20 → retorno R$220
- R$115 em Away a 1.90 → retorno R$218.50
- Investimento: R$215 | Retorno garantido: ~R$219
```

#### 2. Mispricing Simples (Implementado)
Um lado tem odd significativamente acima da odd justa calculada.

```
Odds justas (sem margem):
- Fair_home = odd_home * (1/overround)
- Desvio = (odd_real - odd_justa) / odd_justa

Se desvio > 2%, consideramos mispricing.
```

#### 3. Mispricing Cruzado entre Mercados (Proposta Futura)

Detectar inconsistências comparando diferentes tipos de mercado que deveriam ter probabilidades relacionadas.

##### 3.1 Comparação 1X2 vs Asian Handicap

A probabilidade de Home ganhar no 1X2 deve ser consistente com o AH 0.0:

```
1X2:
- P(Home vencer) = 1/odd_home_1x2

AH 0.0 (Draw No Bet):
- P(Home vencer ou empatar) = 1/odd_home_ah0

Se P(Home) do 1X2 > P(Home ou empate) do AH 0.0:
→ INCONSISTÊNCIA! Home no 1X2 está subprecificado
```

##### 3.2 Comparação 1X2 vs Asian Handicap -0.5

AH -0.5 Home = Home vencer (sem empate):

```
Deveria ser aproximadamente igual:
P(Home vencer) ≈ 1/odd_home_1x2 × [1 - P(empate)]

Onde P(empate) pode ser estimada da odd de empate no 1X2.
```

##### 3.3 Comparação OU vs AH do Favorito

Jogos com expectativa de mais gols geralmente favorecem o time mais forte:

```
Over 2.5 odds baixas → Expectativa de jogo com gols
→ Favorito deveria ter odds menores (mais provável vencer)

Se Over 2.5 = 1.70 (muitos gols esperados)
Mas Favorito AH -0.5 = 2.20 (odds altas para favorito)
→ Possível inconsistência
```

##### 3.4 Triangulação 1X2 completa

Com as três odds do 1X2, podemos validar se são consistentes:

```
P(Home) + P(Draw) + P(Away) = Overround (~105-108%)

Se muito diferente:
→ Alguma odd está incorreta
```

### Métricas Coletadas

| Campo | Descrição |
|-------|-----------|
| `overround` | Soma das probabilidades implícitas |
| `is_arb` | Se overround < 100% |
| `deviation_a/b` | Desvio da odd real vs odd justa |
| `recommended_side` | Lado com maior desvio positivo |
| `recommended_odd` | Odd no momento da detecção |

### Status Atual

[OK] **Implementado e coletando dados**
- Detecta arbitragem e mispricing simples para AH e OU
- Mercado 1X2 excluído temporariamente (precisa coletar odd de empate junto)

**Próximos passos:**
1. Incluir odd de empate na coleta para calcular overround de 1X2 corretamente
2. Implementar detecção de mispricing cruzado entre mercados
3. Análise de CLV dos eventos detectados

---

## H3: Quebra de Monotonicidade entre Linhas Adjacentes

### O que é?

Verifica se a relação de preços entre linhas de handicap adjacentes está correta.

### Por que deve ser monotônica?

```
Linha mais negativa = mais fácil ganhar para o home = odd MENOR

AH -0.75: Home precisa ganhar por 1+ gol (ou empate 0-0 perde metade)
AH -0.5:  Home precisa ganhar por 1+ gol
AH -0.25: Home precisa ganhar (empate perde metade)

Portanto: Odd(-0.75) < Odd(-0.5) < Odd(-0.25) para HOME
```

### Exemplo de Anomalia

```
        | Odd Home (esperado) | Odd Home (real) | Status
--------|---------------------|-----------------|--------
AH -0.25| 2.10                | 2.10            | OK
AH -0.5 | 1.90                | 1.90            | OK  
AH -0.75| 1.70                | 1.95            | ANOMALIA!

A linha -0.75 está pagando MAIS que -0.5, mas deveria pagar MENOS.
→ Apostar em AH -0.75 Home (está "cara demais")
```

### Por que acontece?

- Liquidez diferente entre linhas
- Erro de bookmaker
- Movimento de odds não sincronizado entre linhas

### Métricas Coletadas

| Campo | Descrição |
|-------|-----------|
| `line_a`, `line_b` | Par de linhas com inversão |
| `odd_line_a/b` | Odds das linhas |
| `magnitude` | Tamanho da inversão |
| `recommended_line` | Linha com odd "errada" |
| `recommended_odd` | Odd para apostar |

### Como Validar

- CLV da linha recomendada
- Frequência de correção (a anomalia é corrigida?)
- ROI de apostar na linha anômala

### Status

[OK] **Implementado e coletando dados**
- Detecta inversões entre linhas adjacentes
- Registra apenas primeira detecção (evita duplicatas)
- Próximo passo: análise de CLV

---

## H3b: Reversões Temporais de Odds

### O que é?

Detecta quando uma odd muda de direção ao longo do tempo (estava subindo e começa a descer, ou vice-versa).

### Por que pode haver valor?

- **Correção de exagero:** Mercado moveu demais e está corrigindo
- **Informação conflitante:** Diferentes participantes com visões opostas
- **Oportunidade de mean-reversion:** Apostar que a odd vai "voltar"

### Exemplo Prático

```
Tempo  | Odd Home | Direção | Status
-------|----------|---------|--------
10:00  | 1.85     | -       | -
10:30  | 1.88     | UP      | Tendência de alta
11:00  | 1.92     | UP      | Streak = 2
11:30  | 1.90     | DOWN    | REVERSAO!

A odd estava subindo (mercado duvidando do home) e reverteu.
Possível interpretação: correção de movimento exagerado.
```

### Métricas Coletadas

| Campo | Descrição |
|-------|-----------|
| `direction_before` | Direção antes da reversão (up/down) |
| `direction_after` | Direção depois |
| `reversal_magnitude` | Tamanho do movimento de reversão |
| `streak_before` | Quantos movimentos na direção anterior |
| `oscillation_index` | Frequência de reversões (0-1) |
| `bet_odd` | Odd no momento da reversão |

### Hipóteses a Testar

1. Apostar NA direção da reversão (follow a correção)
2. Apostar CONTRA a reversão (mercado vai continuar)
3. Mercados com alto `oscillation_index` são mais/menos lucrativos?

### Status

[OK] **Implementado e coletando dados**
- Próximo passo: análise de CLV por direção

---

## H6: Inconsistência de Movimento entre Linhas Correlacionadas

### O que é?

Detecta quando uma linha de mercado move significativamente mas linhas correlacionadas (adjacentes) NÃO moveram junto no mesmo período.

### Conceito

Entre dois ciclos de coleta, esperamos que linhas correlacionadas movam de forma semelhante. Se uma linha move 5% e a adjacente não move, pode indicar:
- Ineficiência temporária
- Oportunidade de apostar na linha "atrasada"

### Exemplo Prático

```
Ciclo N:
- AH -1.0 Home: 1.90
- AH -2.0 Home: 1.85

Ciclo N+1:
- AH -1.0 Home: 2.00 (+5.3%)
- AH -2.0 Home: 1.86 (+0.5%)

A linha -1.0 moveu 5.3%, mas a -2.0 moveu apenas 0.5%.
Esperaríamos movimento similar nas duas linhas.
→ Oportunidade: apostar em AH -2.0 Home (ainda não corrigiu)
```

### Implementação Atual

Comparamos **linhas adjacentes do mesmo tipo de mercado:**

| Mercado Líder | Linhas Adjacentes Verificadas |
|---------------|-------------------------------|
| AH -1.0 | AH -2.0, AH 0.0, AH -1.5, AH -0.5 |
| OU 2.5 | OU 3.5, OU 1.5, OU 3.0, OU 2.0 |

**Parâmetros:**
- Movimento mínimo para considerar: 0.3%
- Tempo de lag para considerar "atrasado": 30 segundos

### Proposta Futura: Correlações Entre Tipos de Mercado

Atualmente só comparamos linhas do mesmo tipo (AH com AH, OU com OU). 

Uma extensão seria comparar **tipos diferentes** de mercado que são correlacionados:

| Mercado Líder | Mercado Correlacionado | Correlação |
|---------------|------------------------|------------|
| OU 2.5 Over | AH -0.5 Favorito | Positiva |
| OU 2.5 Under | AH +0.5 Underdog | Positiva |
| 1X2 Home | AH 0.0 Home | Positiva |

**Implementação necessária:**
1. Identificar qual time é favorito (menor odd no AH)
2. Mapear correlações entre tipos de mercado
3. Detectar quando um tipo move e o correlacionado não

### Métricas Coletadas

| Campo | Descrição |
|-------|-----------|
| `leader_market/line` | Mercado que moveu primeiro |
| `lagged_market/line` | Mercado atrasado |
| `lag_seconds` | Tempo desde último movimento do lagged |
| `expected_direction` | Direção esperada do movimento |
| `bet_odd` | Odd do mercado atrasado |

### Status

[OK] **Implementado e coletando dados**
- Detecta inconsistências entre linhas adjacentes do mesmo tipo
- Próximo passo: análise de CLV
- Futuro: adicionar correlações entre tipos de mercado (OU vs AH)

---

## H4: Steam Moves

### O que é?

Movimentos bruscos de odds (>5% em curto período) que indicam informação entrando no mercado.

### Por que pode haver valor?

- **Following sharp money:** Apostadores profissionais moveram a linha
- **Overreaction:** Mercado exagerou e vai corrigir

### Definição

```
Steam Move = Variação > 5% em < 30 minutos
         OU Variação > 3% em < 10 minutos
```

### Estratégias a Testar

1. **Follow:** Apostar na direção do movimento
2. **Fade:** Apostar contra (esperar correção)
3. **Wait:** Esperar estabilizar

### Status

[PENDENTE] **Não implementado ainda**
- Precisa: tracking de variação entre coletas consecutivas

---

## H5: Eficiência por Liga

### O que é?

Ligas menores/menos populares são menos eficientes e podem ter mais oportunidades.

### Por que pode haver valor?

- Menos liquidez = menos apostadores corrigindo erros
- Bookmakers dedicam menos recursos a ligas menores
- Informação menos disponível

### Tiers de Liga

| Tier | Exemplos | Eficiência Esperada |
|------|----------|---------------------|
| 1 | Premier League, La Liga, Serie A | Muito eficiente |
| 2 | Championship, Eredivisie | Eficiente |
| 3 | League One, 2a divisão europeia | Menos eficiente |
| 4 | National League, ligas menores | Potencialmente ineficiente |

### Métricas a Analisar

- Overround médio por tier
- CLV médio por tier
- Frequência de anomalias por tier

### Status

[PENDENTE] **Não implementado ainda**
- Precisa: classificação de ligas por tier

---

## H7: Odds de Abertura

### O que é?

Odds no momento de abertura do mercado podem ter vieses sistemáticos que são corrigidos até o fechamento.

### Por que pode haver valor?

- Favoritos tendem a abrir com odds mais altas
- "Opening Line Value" similar a CLV

### Limitação

Não temos dado explícito de "abertura" no site BetinAsia.

**Proxy:** Primeira coleta de cada jogo como aproximação.

### Status

[PENDENTE] **Não implementado ainda**
- Precisa: tracking de "primeira coleta" por jogo

---

## Resumo de Status

| Hipótese | Implementado | Coletando | Próximo Passo |
|----------|--------------|-----------|---------------|
| H1 | [OK] | [OK] | Análise CLV, incluir 1X2 |
| H3 | [OK] | [OK] | Análise CLV |
| H3b | [OK] | [OK] | Análise CLV |
| H4 | [X] | - | Implementar |
| H5 | [X] | - | Classificar ligas |
| H6 | [OK] | [OK] | Análise CLV, correlações entre tipos |
| H7 | [X] | - | Implementar |

---

## Próximos Passos

1. **Análise de CLV** para H1, H3, H3b, H6
   - Rodar `python -m results.update_hypothesis_results` após jogos terminarem
   - Analisar distribuição de CLV
   - Calcular ROI simulado

2. **Incluir 1X2 corretamente no H1**
   - Coletar odd de empate no mesmo registro
   - Implementar cálculo de overround para 3 outcomes

3. **Implementar mispricing cruzado (H1 avançado)**
   - Comparar 1X2 vs AH 0.0
   - Comparar OU vs AH do favorito

4. **Implementar correlações entre tipos para H6**
   - OU vs AH correlacionados
   - 1X2 vs AH correlacionados

5. **Implementar H4 (Steam Moves)**
   - Tracker de variação entre coletas

6. **Classificar ligas para H5**
   - Criar tabela de tiers

---

*Documento atualizado em 02/02/2026*
