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

## H3 - Hipótese de Quebras de Monotonicidade

**Objetivo:** Detectar quando as best odds revertem direção (quebra de monotonicidade).

**Abordagem com Best Odds:**
- Monitorar continuamente a série temporal de cada best odd
- Detectar mudança de direção (subindo→descendo ou descendo→subindo)
- Gravar evento quando reversão é detectada

### Variáveis a Gravar (por evento detectado):

| Nome da Variável | Tipo | Descrição |
|------------------|------|-----------|
| `h3_total_reversals` | int | Total de reversões detectadas no evento |
| `h3_reversals_1h_before_bet` | int | Reversões na última hora antes da aposta |
| `h3_last_reversal_time` | datetime | Timestamp da última reversão |
| `h3_time_since_reversal` | float | Segundos entre última reversão e nossa aposta |
| `h3_is_post_reversal` | bool | Se apostamos logo após uma reversão (<5 min) |
| `h3_reversal_magnitude` | float | Tamanho da última reversão (em odds) |
| `h3_direction_before` | categorical | "up" ou "down" - direção antes da reversão |
| `h3_direction_after` | categorical | "up" ou "down" - direção após a reversão |
| `h3_streak_before_break` | int | Movimentos consecutivos na direção antes de quebrar |
| `h3_oscillation_index` | float | `num_reversões / num_movimentos` (0-1) |
| `h3_max_reversal_magnitude` | float | Maior reversão observada no evento |

### Lógica de Detecção:
```
# Para cada atualização de best odd
if odd_atual > odd_anterior:
    direção_atual = "up"
else:
    direção_atual = "down"

# Reversão = mudança de direção
if direção_atual != direção_anterior:
    GRAVAR_EVENTO_REVERSÃO()
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
| H3 | `h3_total_reversals` | int | Total de reversões |
| H3 | `h3_is_post_reversal` | bool | Se apostamos após reversão |
| H3 | `h3_oscillation_index` | float | Índice de oscilação (0-1) |
| H3 | `h3_max_reversal_magnitude` | float | Maior reversão observada |
| H6 | `h6_lag_events_count` | int | Eventos de lag detectados |
| H6 | `h6_avg_lag_seconds` | float | Lag médio em segundos |
| H6 | `h6_max_lag_seconds` | float | Maior lag observado |
| H6 | `h6_bet_on_lagged_market` | bool | Se apostamos em mercado atrasado |

---

## Próximos Passos

1. **Integrar `monitoramento_hipoteses.py` ao sistema de coleta**
   - Chamar `monitor.process_odd_update()` a cada atualização de best odd
   - Configurar diretório de saída dos eventos

2. **Criar script de merge**
   - Ler eventos detectados dos arquivos JSONL
   - Cruzar com tabela resumo de apostas por `event_id`
   - Calcular variáveis agregadas

3. **Definir janela de coleta**
   - Recomendação: mínimo 2 semanas de dados para análise inicial
   - Idealmente 1 mês para variabilidade suficiente

4. **Validar detecções**
   - Revisar manualmente alguns eventos detectados
   - Ajustar thresholds se necessário

---

## Arquivo de Implementação

Ver: `monitoramento_hipoteses.py` para código completo com:
- `PricingMonitor` - H1
- `MonotonicityMonitor` - H3  
- `CorrelationLagMonitor` - H6
- `HypothesisMonitor` - Agregador
