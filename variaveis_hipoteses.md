# Variáveis Necessárias para Análise de Hipóteses

Este documento especifica as variáveis adicionais que precisam ser coletadas/calculadas para testar as hipóteses H1, H3 e H6 no modelo de apostas.

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

**Objetivo:** Testar se a qualidade/eficiência da precificação das casas afeta o CLV.

### Variáveis Sugeridas:

| Nome da Variável | Tipo | Descrição |
|------------------|------|-----------|
| `odd_inicial_casa_apostada` | float | Odd inicial da casa onde apostamos |
| `odd_final_casa_apostada` | float | Odd final (fechamento) da casa |
| `odd_inicial_pinnacle` | float | Odd inicial no Pinnacle (referência sharp) |
| `odd_final_pinnacle` | float | Odd final no Pinnacle |
| `spread_pinnacle_casa` | float | `odd_casa - odd_pinnacle` no momento da aposta |
| `vig_casa_apostada` | float | Margem/overround da casa apostada |
| `vig_pinnacle` | float | Margem do Pinnacle (referência de eficiência) |
| `delta_vig` | float | `vig_casa - vig_pinnacle` |
| `movimento_linha_pre_aposta` | float | Quanto a linha moveu antes de apostarmos (últimos N min) |
| `direcao_movimento_linha` | categorical | "favor" / "contra" / "neutro" - relativo ao nosso lado |
| `volatilidade_odds_24h` | float | Desvio padrão das odds nas últimas 24h |
| `consenso_mercado` | float | % de casas oferecendo odd acima de X |
| `posicao_odd_vs_mercado` | float | Percentil da nossa odd em relação ao mercado |
| `tempo_desde_ultimo_movimento` | float | Segundos desde última alteração de linha |
| `magnitude_ultimo_movimento` | float | Tamanho da última variação de odd |
| `is_opening_line` | bool | Se estamos apostando na linha de abertura |
| `diff_vs_opening_line` | float | Diferença entre odd atual e opening line |

---

## H3 - Hipótese de Quebras de Monotonicidade

**Objetivo:** Identificar momentos onde as odds não se movem monotonicamente (reversões), indicando possíveis ineficiências de mercado.

### Variáveis Sugeridas:

| Nome da Variável | Tipo | Descrição |
|------------------|------|-----------|
| `num_reversoes_1h` | int | Contagem de reversões de direção na última hora |
| `num_reversoes_24h` | int | Contagem de reversões nas últimas 24h |
| `amplitude_maior_reversao` | float | Tamanho da maior reversão observada |
| `tempo_desde_ultima_reversao` | float | Minutos desde última reversão |
| `is_pos_reversao` | bool | Se estamos apostando logo após uma reversão |
| `direcao_pre_reversao` | categorical | Direção do movimento antes da reversão |
| `streak_atual_direcao` | int | Quantos movimentos consecutivos na mesma direção |
| `ratio_up_down_moves` | float | Proporção movimentos para cima/baixo nas últimas N horas |
| `tendencia_quebrada` | bool | Se o movimento atual quebrou tendência estabelecida |
| `volatilidade_intraday` | float | Variância dos movimentos no mesmo dia |
| `max_drawdown_odd` | float | Maior queda consecutiva da odd (do pico) |
| `max_runup_odd` | float | Maior subida consecutiva |
| `oscillation_index` | float | Índice de oscilação: `(num_reversoes) / (num_movimentos)` |
| `mean_reversion_tendency` | float | Tendência de reversão à média histórica |
| `crossing_count` | int | Quantas vezes a odd cruzou a média móvel |

---

## H6 - Hipótese de Atrasos na Movimentação de Odds Correlacionadas

**Objetivo:** Detectar quando casas lentas não atualizaram suas odds enquanto casas rápidas já moveram.

### Variáveis Sugeridas:

| Nome da Variável | Tipo | Descrição |
|------------------|------|-----------|
| `lag_pinnacle` | float | Atraso (segundos) da casa apostada vs Pinnacle |
| `lag_betfair` | float | Atraso vs Betfair Exchange |
| `lag_mercado_medio` | float | Atraso vs média do mercado |
| `casas_ja_moveram` | int | Quantas casas já moveram enquanto nossa casa não |
| `pct_casas_alinhadas` | float | % de casas com odds similares (±X%) |
| `spread_max_min_mercado` | float | Diferença entre maior e menor odd do mercado |
| `is_outlier_positivo` | bool | Se nossa odd está significativamente acima do mercado |
| `is_outlier_negativo` | bool | Se nossa odd está significativamente abaixo |
| `z_score_odd` | float | Z-score da nossa odd vs distribuição do mercado |
| `tempo_ultima_atualizacao_casa` | float | Segundos desde última atualização da casa apostada |
| `tempo_ultima_atualizacao_mercado` | float | Segundos desde última atualização de qualquer casa |
| `delta_atualizacao` | float | Diferença entre os dois tempos acima |
| `velocidade_media_casa` | float | Velocidade histórica de atualização da casa |
| `ranking_velocidade_casa` | int | Posição da casa no ranking de velocidade |
| `correlacao_historica_mercado` | float | Correlação histórica entre movimentos da casa e mercado |
| `expected_move` | float | Movimento esperado baseado em outras casas |
| `residual_vs_expected` | float | `odd_atual - expected_move` |
| `leading_indicator` | bool | Se casas sharp já sinalizaram movimento |
| `cluster_timing` | categorical | "líder" / "seguidor" / "outlier" |

---

## Variáveis de Contexto (Úteis para Todas as Hipóteses)

| Nome da Variável | Tipo | Descrição |
|------------------|------|-----------|
| `sport` | categorical | Esporte (futebol, basquete, etc.) |
| `league` | categorical | Liga/competição |
| `is_top_league` | bool | Se é uma liga principal |
| `home_team_rank` | int | Ranking do time da casa |
| `away_team_rank` | int | Ranking do time visitante |
| `market_type` | categorical | Tipo de mercado (AH, OU, ML, etc.) |
| `liquidity_estimate` | float | Estimativa de liquidez do evento |
| `is_live` | bool | Se é aposta ao vivo |
| `event_importance` | categorical | Importância do evento |
| `historical_clv_casa` | float | CLV histórico médio naquela casa |
| `historical_clv_liga` | float | CLV histórico médio naquela liga |
| `hora_utc` | float | Hora em UTC (contínua, 0-24) |
| `dia_do_mes` | int | Dia do mês |
| `is_fim_de_semana` | bool | Se é sábado/domingo |

---

## Fontes de Dados Necessárias

Para coletar essas variáveis, precisamos:

1. **API de Odds em Tempo Real**
   - Histórico de movimentos de odds (timestamps)
   - Odds de múltiplas casas simultaneamente
   - Opening lines

2. **Sistema de Logging Enriquecido**
   - Capturar snapshot do mercado no momento da aposta
   - Registrar odds de todas as casas disponíveis

3. **Base de Referência**
   - Classificação de casas (sharp vs soft)
   - Velocidade média de atualização por casa
   - Correlações históricas entre casas

---

## Priorização Sugerida

### Alta Prioridade (Impacto Esperado Alto)
- `spread_pinnacle_casa` (H1)
- `num_reversoes_1h` (H3)
- `lag_pinnacle` (H6)
- `is_outlier_positivo` (H6)
- `leading_indicator` (H6)

### Média Prioridade
- `direcao_movimento_linha` (H1)
- `oscillation_index` (H3)
- `z_score_odd` (H6)
- `casas_ja_moveram` (H6)

### Baixa Prioridade (Nice to Have)
- Variáveis de contexto
- Métricas de volatilidade de longo prazo

---

## Próximos Passos

1. **Validar disponibilidade de dados** - Verificar quais APIs/fontes conseguem fornecer esses dados
2. **Implementar coleta incremental** - Começar com variáveis de alta prioridade
3. **Modificar payload** - Atualizar estrutura do CSV/JSON para incluir novas variáveis
4. **Período de coleta** - Definir janela mínima de dados para análise estatística
5. **Feature engineering** - Algumas variáveis podem ser derivadas de dados já existentes
