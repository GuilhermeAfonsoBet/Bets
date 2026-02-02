# Hipóteses de Estratégia - BetinAsia Bot

**Data:** Fevereiro 2026  
**Versão:** 2.0  
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

## H1: Precificação Incorreta (Overround Anômalo)

### O que é?

Detecta quando o overround (margem total do mercado) está anormalmente baixo ou quando há arbitragem.

### Por que pode haver valor?

- **Arbitragem (overround < 100%):** Lucro garantido apostando em ambos os lados
- **Mispricing:** Quando um lado está com odd acima da "odd justa"

### Exemplo Prático

```
Mercado AH -0.5:
- Home: 2.10  → Prob implícita: 47.6%
- Away: 1.85  → Prob implícita: 54.1%
- TOTAL: 101.7% (overround normal)

Mercado com anomalia:
- Home: 2.20  → Prob implícita: 45.5%
- Away: 1.90  → Prob implícita: 52.6%
- TOTAL: 98.1% (ARBITRAGEM!)

Odds justas (removendo margem):
- Home justa: 2.10 / 0.981 = 2.14
- Home real: 2.20
- Desvio: +2.8% → APOSTAR EM HOME
```

### Métricas Coletadas

| Campo | Descrição |
|-------|-----------|
| `overround` | Soma das probabilidades implícitas |
| `is_arb` | Se overround < 100% |
| `deviation_a/b` | Desvio da odd real vs odd justa |
| `recommended_side` | Lado com maior desvio positivo |
| `recommended_odd` | Odd no momento da detecção |

### Como Validar

- Calcular CLV médio dos eventos detectados
- Comparar ROI de apostar no `recommended_side`
- Verificar se detecções de arbitragem são reais (confirmar odds no site)

### Status

✅ **Implementado e coletando dados**
- 43k+ eventos detectados
- Próximo passo: análise de CLV

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

✅ **Implementado e coletando dados**
- 585 eventos detectados
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
11:30  | 1.90     | DOWN    | REVERSÃO!

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

✅ **Implementado e coletando dados**
- 570 eventos detectados
- Próximo passo: análise de CLV por direção

---

## H6: Correlação Incompleta entre Mercados (Lag)

### O que é?

Detecta quando um mercado move mas mercados correlacionados NÃO movem junto (ficam "atrasados").

### Por que mercados devem ser correlacionados?

```
Over 2.5 e AH do favorito são correlacionados:

Se Over 2.5 CAI (mercado espera mais gols):
→ Provavelmente o favorito vai ganhar por mais
→ AH -0.5 do favorito deveria CAIR também

Se AH -0.5 do favorito ainda não moveu:
→ Está "atrasado" (lag)
→ Oportunidade de apostar antes que corrija
```

### Exemplo Prático

```
Momento T0:
- Over 2.5:      1.90
- AH -0.5 Home:  1.95

Momento T1 (informação entra no mercado):
- Over 2.5:      1.80  ⬇️ Moveu -5.3%
- AH -0.5 Home:  1.95  ⚠️ Ainda não moveu!

OPORTUNIDADE: Apostar em AH -0.5 Home a 1.95
(esperando que corrija para ~1.85)

Momento T2 (correção):
- Over 2.5:      1.80
- AH -0.5 Home:  1.86  ⬇️ Corrigiu
```

### Correlações Monitoradas

| Mercado Líder | Mercado Correlacionado | Correlação Esperada |
|---------------|------------------------|---------------------|
| AH -0.5 Home | AH -0.75 Home | Mesma direção (+0.90) |
| AH -0.5 Home | AH -0.25 Home | Mesma direção (+0.90) |
| AH -0.5 Home | AH +0.5 Away | Direção oposta (-0.95) |
| OU 2.5 Over | OU 2.0 Over | Mesma direção (+0.85) |

### Limitação Atual ⚠️

**Problema:** Com coleta batch (todos os mercados ao mesmo tempo), não conseguimos detectar o "momento" de lag.

```
Coleta a cada 60s:
- Ciclo 1: Over=1.90, AH=1.95
- Ciclo 2: Over=1.80, AH=1.86 (ambos já moveram)

Nunca capturamos o momento T1 onde Over moveu mas AH ainda não.
```

**Solução necessária:** Coleta em tempo real (streaming) ou intervalo muito menor.

### Métricas Coletadas

| Campo | Descrição |
|-------|-----------|
| `leader_market/line` | Mercado que moveu primeiro |
| `lagged_market/line` | Mercado atrasado |
| `lag_seconds` | Tempo de atraso |
| `expected_direction` | Direção esperada do movimento |
| `bet_odd` | Odd do mercado atrasado |

### Status

⚠️ **Implementado mas com limitações**
- 0 eventos detectados (devido a coleta batch)
- Precisa: streaming ou intervalo < 30s

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

⏳ **Não implementado ainda**
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
| 3 | League One, 2ª divisão europeia | Menos eficiente |
| 4 | National League, ligas menores | Potencialmente ineficiente |

### Métricas a Analisar

- Overround médio por tier
- CLV médio por tier
- Frequência de anomalias por tier

### Status

⏳ **Não implementado ainda**
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

⏳ **Não implementado ainda**
- Precisa: tracking de "primeira coleta" por jogo

---

## Resumo de Status

| Hipótese | Implementado | Coletando | Eventos | Próximo Passo |
|----------|--------------|-----------|---------|---------------|
| H1 | ✅ | ✅ | 43k+ | Análise CLV |
| H3 | ✅ | ✅ | 585 | Análise CLV |
| H3b | ✅ | ✅ | 570 | Análise CLV |
| H4 | ❌ | - | - | Implementar |
| H5 | ❌ | - | - | Classificar ligas |
| H6 | ⚠️ | ❌ | 0 | Precisa streaming |
| H7 | ❌ | - | - | Implementar |

---

## Próximos Passos

1. **Análise de CLV** para H1, H3, H3b
   - Rodar `update_hypothesis_results.py` após jogos terminarem
   - Analisar distribuição de CLV
   - Calcular ROI simulado

2. **Implementar H4 (Steam Moves)**
   - Tracker de variação entre coletas

3. **Classificar ligas para H5**
   - Criar tabela de tiers

4. **Avaliar necessidade de streaming para H6**
   - Alternativa: reduzir intervalo de coleta

---

*Documento atualizado em 02/02/2026*
