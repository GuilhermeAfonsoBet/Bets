# Variáveis Necessárias para Análise de Hipóteses

Este documento especifica as variáveis adicionais que precisam ser coletadas/calculadas para testar as hipóteses H1, H3 e H6 no modelo de apostas.

**IMPORTANTE:** Trabalhamos apenas com **BEST ODDS**, não com odds individuais de cada casa.
A estratégia é **monitorar continuamente** e **gravar eventos** quando detectados,
para posterior merge com a tabela resumo de apostas.

Ver: `monitoramento_hipoteses.py` para implementação.

---

## Variáveis Atuais (10 features)

**Numéricas (6):**
- `Número de casas disponíveis no momento da aposta`
- `Dif % maior odd e segunda maior`
- `Dif % maior odd e odd mediana`
- `Dif Odds RB & BIA`
- `MinutesToMatchStart`
- `TempoApostas.Tempo total bot`

**Categóricas (4):**
- `Subtipo da Aposta`
- `Dia Semana Aposta (UTC)`
- `Turno Aposta (UTC)`
- `Casa aposta vencedora`

---

## H1 - Hipótese de Precificação

**Objetivo:** Detectar momentos de precificação incorreta no mercado.

**Abordagem com Best Odds:**
- Monitorar continuamente os pares de best odds (home/away, over/under)
- Calcular overround/vig a cada atualização
- Detectar e gravar quando: arbitragem (overround < 1) ou desvio significativo

### Variáveis a Gravar (por evento detectado):

| Nome da Variável | Tipo | Descrição |
|------------------|------|-----------|
| `h1_pricing_events_count` | int | Quantos eventos de mispricing detectados antes da aposta |
| `h1_last_mispricing_time` | datetime | Quando foi o último mispricing detectado |
| `h1_time_since_mispricing` | float | Segundos entre mispricing e nossa aposta |
| `h1_overround_at_bet` | float | Overround no momento da aposta |
| `h1_had_arb_opportunity` | bool | Se houve arbitragem detectada |
| `h1_avg_edge_detected` | float | Média de edge nos mispricings detectados |
| `h1_max_edge_detected` | float | Maior edge detectado |
| `h1_mispriced_side` | categorical | Qual lado estava mal precificado |
| `h1_fair_odd_calculated` | float | Odd justa calculada no momento |
| `h1_deviation_from_fair` | float | Desvio da best odd vs odd justa |

### Cálculo de Precificação:
```
prob_implícita_a = 1 / best_odd_a
prob_implícita_b = 1 / best_odd_b
overround = prob_a + prob_b

# Odds justas (removendo margem)
fair_prob_a = prob_a / overround
fair_odd_a = 1 / fair_prob_a

# Desvio
deviation = (best_odd - fair_odd) / fair_odd
```

---

## H3 - Quebra de Monotonicidade entre Linhas Adjacentes

**Objetivo:** Detectar quando a relação de preços entre linhas adjacentes de AH está invertida.

**Conceito:**
- Se AH **-0.5** home paga **1.90**
- Então AH **-0.75** home deveria pagar **MENOS** (ex: 1.70) — mais fácil ganhar
- E AH **-0.25** home deveria pagar **MAIS** (ex: 2.10) — mais difícil ganhar
- **ANOMALIA** = essa relação está invertida

**Abordagem com Best Odds:**
- A cada snapshot, verificar se odds entre linhas adjacentes seguem ordem esperada
- Gravar evento quando inversão detectada

### Variáveis a Gravar (por evento detectado):

| Nome da Variável | Tipo | Descrição |
|------------------|------|-----------|
| `h3_anomaly_count` | int | Quantas anomalias de linha detectadas |
| `h3_last_anomaly_time` | datetime | Quando foi detectada última anomalia |
| `h3_anomaly_lines` | tuple | Par de linhas com inversão (ex: "-0.5, -0.75") |
| `h3_anomaly_odds` | tuple | Odds das linhas (ex: "1.90, 1.95") |
| `h3_anomaly_magnitude` | float | Tamanho da inversão (diferença de odds) |
| `h3_anomaly_side` | categorical | "home" ou "away" |
| `h3_time_since_anomaly` | float | Segundos entre anomalia e nossa aposta |
| `h3_bet_on_anomaly_line` | bool | Se apostamos em uma das linhas anômalas |

### Exemplo de Anomalia:
```
Linha    | Odd Home (esperado) | Odd Home (real) | Status
---------|---------------------|-----------------|--------
AH -0.25 | 2.10                | 2.10            | OK
AH -0.5  | 1.90                | 1.90            | OK  
AH -0.75 | 1.70                | 1.95            | ANOMALIA! Deveria ser < 1.90
```

### Lógica de Detecção:
```python
# Para cada par de linhas adjacentes
for i in range(len(linhas) - 1):
    linha_menor = linhas[i]      # ex: -0.5
    linha_maior = linhas[i+1]    # ex: -0.25
    
    # Se linha menos negativa, odd home deve ser MAIOR
    if odd_home[linha_maior] < odd_home[linha_menor]:
        GRAVAR_EVENTO_ANOMALIA()
```

---

## H3b - Reversões Temporais de Odds (Monotonicidade Temporal)

**Objetivo:** Detectar quando uma odd reverte direção ao longo do tempo (estava subindo, começou a descer ou vice-versa).

**Conceito:**
- Movimento monotônico: odds só sobe OU só desce ao longo do tempo
- **REVERSÃO** = mudança de direção (up → down ou down → up)
- Pode indicar: correção de exagero, informação nova, incerteza do mercado

**Abordagem com Best Odds:**
- Monitorar série temporal de cada best odd
- Detectar mudança de direção
- Gravar evento quando reversão é detectada

### Variáveis a Gravar (por evento detectado):

| Nome da Variável | Tipo | Descrição |
|------------------|------|-----------|
| `h3b_reversal_count` | int | Total de reversões detectadas no evento |
| `h3b_reversals_1h` | int | Reversões na última hora |
| `h3b_last_reversal_time` | datetime | Timestamp da última reversão |
| `h3b_time_since_reversal` | float | Segundos desde última reversão |
| `h3b_is_post_reversal` | bool | Se apostamos logo após reversão (<5 min) |
| `h3b_reversal_magnitude` | float | Tamanho da reversão (diferença de odd) |
| `h3b_direction_before` | categorical | "up" ou "down" antes da reversão |
| `h3b_direction_after` | categorical | "up" ou "down" após reversão |
| `h3b_streak_before_break` | int | Movimentos consecutivos antes de reverter |
| `h3b_oscillation_index` | float | `num_reversões / num_movimentos` (0 = estável, 1 = muito instável) |

### Exemplo de Reversão:
```
Tempo    | Odd    | Direção | Status
---------|--------|---------|--------
10:00    | 1.85   | -       | -
10:05    | 1.87   | UP      | -
10:10    | 1.90   | UP      | Streak = 2
10:15    | 1.88   | DOWN    | REVERSÃO! (up → down)
```

### Lógica de Detecção:
```python
# Para cada atualização de best odd
if odd_atual > odd_anterior:
    direcao_atual = "up"
elif odd_atual < odd_anterior:
    direcao_atual = "down"
else:
    return  # Sem movimento

# Reversão = mudança de direção
if direcao_atual != direcao_anterior:
    GRAVAR_EVENTO_REVERSAO()
```

---

## H6 - Hipótese de Atrasos na Movimentação de Odds Correlacionadas

**Objetivo:** Detectar quando mercados correlacionados não estão sincronizados nas best odds.

**Abordagem com Best Odds:**
- Monitorar mercados que DEVEM se mover juntos (linhas adjacentes, lados opostos)
- Quando um mercado move, verificar se os correlacionados também moveram
- Gravar evento quando há lag significativo (>30s) entre movimentos correlacionados

### Mercados Correlacionados (mesmo evento):

| Mercado Líder | Mercado Correlacionado | Correlação |
|---------------|------------------------|------------|
| AH -0.5 Home | AH -0.25 Home | +0.90 (mesma direção) |
| AH -0.5 Home | AH -0.75 Home | +0.90 (mesma direção) |
| AH -0.5 Home | AH +0.5 Away | -0.95 (direção oposta) |
| OU 2.5 Over | OU 2.0 Over | +0.85 (mesma direção) |
| OU 2.5 Over | OU 2.5 Under | -0.98 (direção oposta) |

### Variáveis a Gravar (por evento detectado):

| Nome da Variável | Tipo | Descrição |
|------------------|------|-----------|
| `h6_lag_events_count` | int | Quantos eventos de lag detectados |
| `h6_avg_lag_seconds` | float | Tempo médio de atraso observado |
| `h6_max_lag_seconds` | float | Maior atraso observado |
| `h6_last_lag_time` | datetime | Quando foi detectado último lag |
| `h6_leader_market` | str | Mercado que moveu primeiro |
| `h6_lagged_market` | str | Mercado que estava atrasado |
| `h6_leader_direction` | categorical | Direção do movimento líder |
| `h6_expected_move` | float | Movimento esperado no mercado atrasado |
| `h6_actual_move` | float | Movimento real (pode ser 0) |
| `h6_markets_with_lag` | list | Lista de mercados que apresentaram lag |
| `h6_bet_on_lagged_market` | bool | Se apostamos no mercado que estava atrasado |

### Lógica de Detecção:
```
# Quando best_odd de mercado A muda:
for mercado_correlacionado in get_correlacionados(A):
    ultimo_movimento_B = get_ultimo_movimento(mercado_correlacionado)
    
    if tempo_desde(ultimo_movimento_B) > 30 segundos:
        GRAVAR_EVENTO_LAG(
            lider=A,
            atrasado=mercado_correlacionado,
            lag=tempo_desde(ultimo_movimento_B)
        )
```

---

---

## Arquitetura de Implementação

### Fluxo de Dados

```
[Stream Best Odds] 
       │
       ▼
┌──────────────────────────────────────────────┐
│         HypothesisMonitor (contínuo)         │
│  ┌─────────┐ ┌─────────┐ ┌─────────────────┐ │
│  │ H1      │ │ H3      │ │ H6              │ │
│  │ Pricing │ │ Monot.  │ │ Correlation Lag │ │
│  └────┬────┘ └────┬────┘ └────────┬────────┘ │
│       │          │               │           │
│       ▼          ▼               ▼           │
│  [pricing_  [monotonicity_  [correlation_   │
│   events.    events.         lag_events.    │
│   jsonl]     jsonl]          jsonl]         │
└──────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────┐
│         Merge com Tabela Resumo              │
│                                              │
│  Para cada aposta realizada:                 │
│  1. Buscar event_id da aposta                │
│  2. Agregar eventos detectados               │
│  3. Calcular variáveis resumo (h1_*, h3_*,   │
│     h6_*)                                    │
│  4. Anexar à linha da aposta                 │
└──────────────────────────────────────────────┘
```

### Variáveis Finais para Tabela Resumo

| Hipótese | Variável | Tipo | Descrição |
|----------|----------|------|-----------|
| H1 | `h1_pricing_events_count` | int | Mispricings detectados |
| H1 | `h1_had_arb` | bool | Se houve oportunidade de arb |
| H1 | `h1_avg_edge` | float | Edge médio detectado |
| H1 | `h1_time_since_mispricing` | float | Segundos desde último mispricing |
| H3 | `h3_anomaly_count` | int | Anomalias de linha detectadas |
| H3 | `h3_bet_on_anomaly_line` | bool | Se apostamos em linha anômala |
| H3 | `h3_anomaly_magnitude` | float | Magnitude da inversão |
| H3 | `h3_time_since_anomaly` | float | Segundos desde última anomalia |
| H3b | `h3b_reversal_count` | int | Total de reversões temporais |
| H3b | `h3b_is_post_reversal` | bool | Se apostamos após reversão |
| H3b | `h3b_oscillation_index` | float | Índice de oscilação (0-1) |
| H3b | `h3b_max_reversal_magnitude` | float | Maior reversão observada |
| H6 | `h6_lag_events_count` | int | Eventos de lag detectados |
| H6 | `h6_avg_lag_seconds` | float | Lag médio em segundos |
| H6 | `h6_max_lag_seconds` | float | Maior lag observado |
| H6 | `h6_bet_on_lagged_market` | bool | Se apostamos em mercado atrasado |

---

## STATUS DE IMPLEMENTAÇÃO

**IMPORTANTE:** O monitoramento ainda **NÃO ESTÁ IMPLEMENTADO** na operação atual de coleta.

O arquivo `monitoramento_hipoteses.py` na raiz do workspace é apenas uma **proposta de código** que precisa ser integrada ao sistema de coleta existente em `betinasia_bot/`.

### O que existe hoje:
- `betinasia_bot/collector/continuous_collector.py` - Coleta odds do site
- `betinasia_bot/storage/database.py` - Armazena em banco de dados
- `betinasia_bot/results/compact_odds.py` - Compacta dados

### O que FALTA fazer:
1. Integrar detectores de hipóteses no fluxo de coleta
2. Criar tabelas para armazenar eventos detectados
3. Implementar merge com tabela resumo

---

## Próximos Passos

1. **Revisar sistema de coleta atual**
   - Entender estrutura do `continuous_collector.py`
   - Identificar ponto de integração

2. **Implementar detectores no coletor**
   - H1: verificar pares de odds a cada snapshot
   - H3: verificar monotonicidade entre linhas a cada snapshot
   - H3b: verificar reversões temporais a cada atualização
   - H6: verificar correlações entre mercados

3. **Criar storage para eventos**
   - Tabela `hypothesis_events` no banco
   - Ou arquivos JSONL por dia

4. **Criar script de merge**
   - Cruzar eventos com tabela resumo de apostas
   - Calcular variáveis agregadas

---

## Arquivos Relevantes

| Arquivo | Descrição | Status |
|---------|-----------|--------|
| `monitoramento_hipoteses.py` (raiz) | Proposta de código para detectores | Proposta |
| `variaveis_hipoteses.md` (raiz) | Este documento | Atualizado |
| `betinasia_bot/docs/HIPOTESES_ESTRATEGIA.md` | Documento principal de hipóteses | Atualizado |
| `betinasia_bot/collector/continuous_collector.py` | Coletor atual | Precisa integração |
