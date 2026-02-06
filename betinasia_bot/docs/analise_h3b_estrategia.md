# Análise da Estratégia H3B — Reversão Temporal de Odds

**Data:** 06 de Fevereiro de 2026  
**Versão:** 1.0  
**Autor:** Equipe BetinAsia Bot

---

## 1. Resumo Executivo

Investigamos se a hipótese H3B (Reversão Temporal de Odds) gera valor para apostas no mercado Asian Handicap. A análise foi conduzida em duas etapas:

1. **WebSocket (teórica):** Detecta reversões de odds em tempo real e calcula o CLV (Closing Line Value) comparando com a closing line.
2. **Betslip (realista):** Verifica se o valor sobrevive quando consideramos a odd real disponível no betslip, incluindo o lag de execução.

### Conclusão Principal

O sinal H3B UP mostra **valor promissor** (CLV +1.116%) mas **ainda não é estatisticamente significativo** (p > 0.10). O lag de ~20 segundos entre detecção e execução provavelmente erode parte desse valor. A recomendação é usar H3B como **feature** num modelo de scoring mais amplo, não como estratégia isolada.

---

## 2. O que é H3B — Reversão Temporal

### Definição

Uma **reversão temporal** ocorre quando uma odd muda de direção após uma sequência de movimentos consecutivos na mesma direção.

**Exemplo — Reversão UP:**
```
Odd Home AH -0.5:  1.850 → 1.830 → 1.810 → 1.840 (↑ reversão!)
```

A odd estava caindo (3 movimentos consecutivos para baixo) e subiu. A hipótese é que essa subida indica que o mercado "corrigiu" um excesso de movimento, e a odd atual (1.840) tem valor.

### Tipos

- **Reversão UP:** Odd subiu após sequência de queda → odd pode estar subvalorizada → potencial aposta
- **Reversão DOWN:** Odd desceu após sequência de alta → odd pode estar sobrevalorizada → potencial fade

---

## 3. Dados Coletados

### 3.1. Infraestrutura

| Componente | Descrição |
|-----------|-----------|
| **Collector** | WebSocket intercepta odds do BetinAsia em tempo real |
| **Detector** | Identifica eventos H3B em todas as linhas AH e OU |
| **Auditor** | Compara odd do WebSocket com odd real do Betslip |
| **Banco** | PostgreSQL com histórico completo de odds |
| **VPS** | DigitalOcean, rodando 24/7 como serviço systemd |

### 3.2. Volume de Dados

| Métrica | Valor |
|---------|-------|
| Período de coleta | Janeiro–Fevereiro 2026 |
| Total de odds coletadas (WebSocket) | ~40.000+ por ciclo |
| Eventos H3B detectados | ~600+ |
| Auditorias de betslip realizadas | 287 |
| Auditorias com betslip extraído | 115 (40.1%) |
| Ligas cobertas | Premier League, Bundesliga, La Liga, Serie A, Ligue 1, e outras |

### 3.3. Mercados Analisados

- Asian Handicap (Full-Time e Half-Time)
- Over/Under (Full-Time e Half-Time)
- Todas as linhas disponíveis

---

## 4. Análise WebSocket — Resultados Teóricos

### 4.1. Metodologia

Para cada evento H3B detectado:

1. **CLV do evento** = (odd_no_momento − closing_odd) / closing_odd × 100%
2. **CLV baseline** = média do CLV de outras linhas do mesmo jogo no mesmo momento
3. **CLV adicional** = CLV do evento − CLV baseline

O CLV adicional mede o **valor incremental** da estratégia, descontando o "drift" natural do mercado.

### 4.2. Resultados — Todas as Hipóteses

| Hipótese | Direção | N | CLV Adicional | IC 90% | Significância |
|----------|---------|---|---------------|--------|---------------|
| **H1** — Precificação | — | 442 | +0.047% | [−0.651%, +0.746%] | ⚪ Não |
| **H3** — Monotonicidade | — | 48 | +0.615% | [−1.809%, +3.039%] | ⚪ Não |
| **H3B** — Reversão UP | UP | 273 | **+1.116%** | [−0.463%, +2.695%] | ⚪ Não (próximo) |
| **H3B** — Reversão DOWN | DOWN | 282 | −1.359% | [−2.981%, +0.262%] | ⚪ Não |
| **H6** — Correlação/Lag | Líder DOWN | 388 | **+2.301%** | [+1.328%, +3.273%] | ✅ **Sim (p<0.10)** |
| **H6** — Correlação/Lag | Líder UP | 391 | **+3.068%** | [+2.066%, +4.070%] | ✅ **Sim (p<0.10)** |

### 4.3. Destaques

- **H6 (Correlação/Lag)** é a única hipótese com significância estatística. CLV de +2.3% a +3.1%.
- **H3B UP** tem CLV de +1.116%, promissor mas precisa de ~547 observações para significância.
- **H3B DOWN** tem CLV negativo — confirma que reversões DOWN não devem ser apostadas.

---

## 5. Análise Betslip — Realidade da Execução

### 5.1. O Problema do Lag

Entre a detecção do sinal H3B no WebSocket e a execução real da aposta no Betslip, há um atraso:

| Etapa | Tempo Típico |
|-------|-------------|
| WebSocket detecta H3B | 0ms (referência) |
| Ciclo WebSocket (coleta + processamento) | ~8-10s |
| Navegar para o jogo no site | ~2s |
| Expandir linhas AH | ~2-3s |
| Clicar na odd e abrir betslip | ~2s |
| Betslip carregar e extrair dados | ~2s |
| **Total médio** | **~15-20s** |

### 5.2. Auditoria Betslip — Resultados Preliminares

| Métrica | Valor |
|---------|-------|
| Total de auditorias | 287 |
| Com betslip extraído com sucesso | 115 (40.1%) |
| Taxa de execução estimada | ~40% |
| Lag médio total | ~15s |
| Pre-match | 120 |
| In-match | 118 |

### 5.3. Causas de Falha (60% das auditorias)

| Causa | Contagem |
|-------|----------|
| MAJOR_DIFF (odds muito diferentes) | 109 |
| LINE_NOT_AVAILABLE | 102 |
| GAME_NOT_FOUND | 5 |
| Outros | 71 |

**Nota:** Muitas "MAJOR_DIFF" são de linhas extremas (AH ±3, ±4, ±5) onde as odds do WebSocket e do Betslip diferem significativamente por baixa liquidez.

### 5.4. CLV Betslip — Dados Preliminares

Com apenas N=3 (jogos com kickoff passado E closing line disponível), os resultados são **estatisticamente irrelevantes**:

| Métrica | Valor |
|---------|-------|
| N | 3 |
| CLV médio Betslip | −13.9% |
| IC 90% | [−33.9%, +6.1%] |
| Significância | Não (N insuficiente) |

**Estes dados não permitem nenhuma conclusão.** É necessário acumular mais dados ao longo de semanas.

---

## 6. Análise Estratégica

### 6.1. H3B como Estratégia Isolada — Ceticismo Fundamentado

O CLV de +1.116% é um edge **pequeno**. Com um lag de ~20 segundos, é provável que o mercado se mova o suficiente para consumir este valor. Argumentos:

1. **Edge pequeno vs lag grande:** Em mercados líquidos, odds se movem frações de segundo após desequilíbrios. 20 segundos é uma eternidade.
2. **Taxa de execução de 40%:** Mesmo que o edge exista, só 4 em cada 10 sinais se convertem em apostas reais.
3. **Linhas extremas:** Muitos sinais H3B ocorrem em linhas de baixa liquidez (AH ±3, ±4), onde odds são voláteis e irrelevantes na prática.

### 6.2. H3B como Feature de Scoring — Abordagem Mais Promissora

O verdadeiro potencial do H3B é como **sinal (feature) dentro de um modelo mais amplo**:

**Features candidatas para scoring:**

| Feature | Descrição |
|---------|-----------|
| `h3b_direction` | UP ou DOWN |
| `h3b_magnitude` | Diferença da odd antes e depois da reversão |
| `betslip_vs_ws` | Razão betslip_odd / websocket_odd |
| `market_type` | AH, OU, AH_HT, etc. |
| `market_period` | Full-time, Half-time |
| `line_value` | Valor absoluto da linha (filtrar extremas) |
| `is_live` | Pre-match vs In-match |
| `league` | Liga (liquidez e eficiência variam) |
| `day_of_week` | Dia da semana |
| `time_to_kickoff` | Tempo até início do jogo |
| `num_bookmakers` | Número de casas de apostas com odds |
| `streak_length` | Tamanho da sequência antes da reversão |

**Hipótese:** Combinando estas features, podemos identificar um **subconjunto** de eventos H3B com CLV significativamente maior que 1.116%, filtrando:

- Apenas linhas líquidas (|AH| ≤ 1.5)
- Apenas ligas Tier 1 (Premier League, La Liga, etc.)
- Apenas quando betslip_odd ≥ websocket_odd
- Apenas pre-match (odds mais estáveis)

### 6.3. Teste de Velocidade — Reduzir Lag para ~5 Segundos

Uma alternativa para capturar mais valor é **reduzir drasticamente o tempo de execução**:

| Abordagem | Tempo Estimado | Complexidade |
|-----------|---------------|-------------|
| Monitorar 1 liga com URLs pré-cadastradas | ~5s | Baixa |
| Múltiplas abas paralelas por liga | ~3-5s | Média |
| Múltiplas contas BetinAsia | ~3s | Alta (custo + risco) |

**Proposta de teste:** Script que monitora apenas a Premier League, com URLs de todos os jogos já cadastradas, medindo em ~5 segundos:

1. WebSocket detecta H3B
2. Script já tem a aba do jogo aberta → clica direto na odd
3. Extrai betslip em ~2 segundos

Se a diferença betslip vs websocket for significativamente menor com 5s de lag vs 20s, isso valida a abordagem de otimização.

---

## 7. Próximos Passos

### Curto Prazo (1-2 semanas)

1. **Continuar coletando dados** — O audit H3B está rodando 24/7 no VPS. Cada dia acumula ~50-100 auditorias com betslip.
2. **Acumular closing lines** — Conforme jogos vão tendo kickoff, a análise de CLV Betslip fica mais robusta.
3. **Re-rodar análise** periodicamente com `python analyze_h3b_websocket_vs_betslip.py`.

### Médio Prazo (2-4 semanas)

4. **Teste de velocidade** — Script monitorando 1 liga, medindo diff em ~5s.
5. **Modelo de scoring** — Quando tiver N>100 no CLV Betslip, treinar modelo com features combinadas.
6. **Análise por subgrupos** — CLV por liga, por tipo de linha, por magnitude de reversão.

### Longo Prazo (1-3 meses)

7. **Backtesting** do modelo de scoring nos dados históricos.
8. **Dry-run** — Executar apostas simuladas em tempo real.
9. **Execução real** — Se dry-run confirmar edge, ativar apostas reais com stake management.

---

## 8. Estimativas de N Necessário

| Análise | N Atual | N Estimado p/ Significância (IC 90%) |
|---------|---------|--------------------------------------|
| H3B UP WebSocket | 273 | ~547 |
| H3B UP Betslip | 3 | ~500-1000 (estimativa) |
| H6 WebSocket | 388-391 | ✅ Já significativo |

**Estimativa de tempo para alcançar N=500 no Betslip:**
- ~50-100 auditorias/dia com betslip
- ~5-10 dias para N=500 auditorias totais
- Mas apenas ~40% com kickoff passado em cada momento
- **Estimativa realista: 2-4 semanas** para análise robusta

---

## Apêndice A — Intervalos de Confiança

Todos os intervalos de confiança foram calculados com:

```
IC = média ± Z × (σ / √n)

Onde:
  Z = 1.645 para IC 90% (p < 0.10)
  Z = 1.960 para IC 95% (p < 0.05)
  σ = desvio padrão da amostra
  n = tamanho da amostra
```

**Critério de significância:** Se o IC não inclui zero, rejeitamos H₀ (sem valor) e concluímos que há evidência estatística de valor.

## Apêndice B — Glossário

| Termo | Definição |
|-------|-----------|
| **CLV** | Closing Line Value — diferença percentual entre a odd apostada e a closing line |
| **Closing Line** | Última odd disponível antes do kickoff do jogo |
| **Asian Handicap** | Mercado com handicap de gols (ex: AH -1.5 = time precisa vencer por 2+) |
| **Edge** | Vantagem estatística sobre o mercado |
| **Reversão Temporal** | Mudança de direção da odd após movimentos consecutivos |
| **WebSocket** | Conexão em tempo real que transmite atualizações de odds |
| **Betslip** | Painel de aposta do site, mostra a odd real disponível para apostar |
