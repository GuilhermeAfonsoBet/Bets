# Hipoteses de Estrategia - BetinAsia Bot

**Data:** Fevereiro 2026  
**Status:** Em validacao

---

## Tabela de Hipoteses

### H1: Incoerencias entre Mercados

| Campo | Descricao |
|-------|-----------|
| **Nome** | Incoerencias entre Mercados |
| **Descricao** | Calcular probabilidades implicitas de diferentes mercados (1X2, AH, OU) e detectar inconsistencias. Se mercados precificam o mesmo evento de forma diferente, ha oportunidade. |
| **Forma de Medicao** | Converter odds para probabilidades. Comparar P(home win) implicita do 1X2 com P(home win) implicita do AH -0.5. Flag se diferenca > X%. |
| **Variaveis** | odds_1x2_h, odds_1x2_d, odds_1x2_a, odds_ah_all_lines, odds_ou_all_lines |
| **Complexidade** | Media |
| **Estudos/Mercado** | Bem documentado em literatura academica de arbitragem. Pinnacle e casas sharp corrigem rapidamente. Oportunidades mais comuns em casas soft ou mercados menos liquidos. |
| **Comentarios** | A BetinAsia agrega multiplos bookmakers, entao as "best odds" ja sao otimizadas. Incoerencias seriam entre mercados (1X2 vs AH), nao entre bookmakers. |

---

### H2: Vieses Sistematicos

| Campo | Descricao |
|-------|-----------|
| **Nome** | Vieses Sistematicos |
| **Descricao** | Identificar combinacoes de fatores (liga, dia, horario, linha) que historicamente geram ROI positivo de forma consistente. |
| **Forma de Medicao** | Segmentar apostas por cada fator. Calcular ROI por segmento. Testar significancia estatistica (n minimo, p-value). |
| **Variaveis** | liga, pais, dia_semana, horario_kickoff, tempo_ate_kickoff, ah_line, odds_range, resultado_jogo |
| **Complexidade** | Media-Alta |
| **Estudos/Mercado** | Favorite-longshot bias e bem documentado (favoritos extremos subvalorizados, underdogs sobrevalorizados). Alguns estudos mostram vieses por dia/horario, mas resultados variam. |
| **Comentarios** | Risco alto de overfitting com multiplas comparacoes. Usar validacao out-of-sample e correcao estatistica. Coletar dados por semanas antes de concluir. |

---

### H3: Quebra de Monotonicidade entre Linhas Adjacentes

| Campo | Descricao |
|-------|-----------|
| **Nome** | Quebra de Monotonicidade entre Linhas Adjacentes |
| **Descricao** | A progressao de odds entre linhas adjacentes de AH deve ser monotonica. Se AH -0.5 home paga 1.90, entao AH -0.75 home deveria pagar MENOS (mais facil ganhar) e AH -0.25 home deveria pagar MAIS (mais dificil). Inversoes indicam mispricing. |
| **Forma de Medicao** | Para cada jogo, ordenar linhas AH. Calcular delta entre linhas adjacentes. Flag se delta tem sinal diferente do esperado ou magnitude anomala. |
| **Variaveis** | odds_ah_all_lines (ordenadas), handicap_values, h3_anomaly_detected, h3_anomaly_lines, h3_anomaly_magnitude |
| **Complexidade** | Baixa-Media |
| **Estudos/Mercado** | Menos documentado formalmente, mas conceito usado por apostadores quantitativos. Relacionado a "arbitragem de linha" em mercados financeiros. |
| **Comentarios** | Abordagem elegante e logicamente solida. Anomalias podem ser raras mas de alto valor. Implementacao relativamente simples. |

**Exemplo de anomalia:**
```
Linha    | Odd Home (esperado) | Odd Home (real) | Status
---------|---------------------|-----------------|--------
AH -0.25 | 2.10                | 2.10            | OK
AH -0.5  | 1.90                | 1.90            | OK  
AH -0.75 | 1.70                | 1.95            | ANOMALIA! Deveria ser < 1.90
```

---

### H3b: Reversoes Temporais de Odds (Monotonicidade Temporal)

| Campo | Descricao |
|-------|-----------|
| **Nome** | Reversoes Temporais de Odds |
| **Descricao** | Quando a odd de um mercado especifico reverte direcao ao longo do tempo (estava subindo e comeca a descer, ou vice-versa), pode indicar incerteza do mercado ou correcao de movimento exagerado. |
| **Forma de Medicao** | Monitorar serie temporal de cada odd. Detectar mudanca de direcao (up->down ou down->up). Contar reversoes, medir magnitude e tempo entre reversoes. |
| **Variaveis** | h3b_reversal_count, h3b_last_reversal_time, h3b_reversal_magnitude, h3b_direction_before, h3b_direction_after, h3b_oscillation_index |
| **Complexidade** | Baixa |
| **Estudos/Mercado** | Relacionado a conceitos de mean-reversion em mercados financeiros. Odds que oscilam muito podem indicar incerteza ou informacao conflitante. |
| **Comentarios** | Hipotese: apostas feitas logo apos uma reversao podem ter valor diferente. Mercados "nervosos" podem ser mais ou menos lucrativos. |

**Definicao de reversao:**
- Movimento UP: odd_atual > odd_anterior
- Movimento DOWN: odd_atual < odd_anterior
- REVERSAO: direcao_atual != direcao_anterior

**Metricas principais:**
- `oscillation_index = num_reversoes / num_movimentos` (0 = totalmente monotonica, 1 = alternando sempre)
- `streak_before_reversal` = quantos movimentos consecutivos na mesma direcao antes de reverter

---

### H4: Steam Moves

| Campo | Descricao |
|-------|-----------|
| **Nome** | Steam Moves |
| **Descricao** | Movimentos bruscos de odds (>5% em curto periodo) indicam informacao entrando no mercado. Seguir ou fade o movimento pode ter valor. |
| **Forma de Medicao** | Calcular variacao percentual entre coletas consecutivas. Flag se abs(variacao) > threshold. Testar ROI de "follow" vs "fade" o movimento. |
| **Variaveis** | odds_t0, odds_t1, timestamp_t0, timestamp_t1, resultado_jogo |
| **Complexidade** | Media |
| **Estudos/Mercado** | Muito usado por apostadores profissionais (sindicatos). "Following sharp money" e estrategia comum. Evidencias mistas - funciona melhor em mercados menos eficientes. |
| **Comentarios** | Requer coleta frequente (idealmente <5min entre coletas). Dificuldade em distinguir "smart money" de "public money". |

---

### H5: Eficiencia por Liga

| Campo | Descricao |
|-------|-----------|
| **Nome** | Eficiencia por Liga |
| **Descricao** | Mercados de ligas menores/menos populares sao menos eficientes e podem ter mais oportunidades de valor. |
| **Forma de Medicao** | Calcular metricas de eficiencia por liga: dispersao de odds, frequencia de movimentos, ROI de estrategias simples. Comparar ligas tier-1 vs tier-2 vs tier-3. |
| **Variaveis** | liga, tier_liga, odds_all, resultado_jogo |
| **Complexidade** | Media |
| **Estudos/Mercado** | Varios estudos academicos confirmam que ligas menores sao menos eficientes. Pinnacle tem margens menores em ligas principais. |
| **Comentarios** | BetinAsia cobre muitas ligas. Podemos comparar Premier League vs England League 2 vs ligas ainda menores. |

---

### H6: Correlacao Incompleta entre Mercados

| Campo | Descricao |
|-------|-----------|
| **Nome** | Correlacao Incompleta entre Mercados |
| **Descricao** | Quando um mercado se move, outros correlacionados deveriam se mover tambem. Se nao se movem, ha lag ou incoerencia exploravel. |
| **Forma de Medicao** | Monitorar movimento de Over 2.5. Verificar se AH do favorito se move na mesma direcao. Flag se correlacao esperada nao ocorre. |
| **Variaveis** | odds_ou_2_5_t0, odds_ou_2_5_t1, odds_ah_fav_t0, odds_ah_fav_t1, timestamp |
| **Complexidade** | Alta |
| **Estudos/Mercado** | Conceito de "cross-market arbitrage" e conhecido. Menos estudado especificamente para futebol. |
| **Comentarios** | Requer entendimento profundo das relacoes teoricas entre mercados. Complexidade alta na implementacao. |

---

### H7: Odds de Abertura

| Campo | Descricao |
|-------|-----------|
| **Nome** | Odds de Abertura |
| **Descricao** | Odds no momento de abertura do mercado podem ter vieses sistematicos que sao corrigidos ate o fechamento. Apostar cedo em direcoes previssiveis pode ter valor. |
| **Forma de Medicao** | Identificar "primeira coleta" de cada jogo como proxy de abertura. Comparar com odds de fechamento. Calcular direcao e magnitude do movimento. Testar se movimento e previsivel. |
| **Variaveis** | odds_primeira_coleta, odds_ultima_coleta, tempo_ate_kickoff, resultado_jogo |
| **Complexidade** | Media-Alta |
| **Estudos/Mercado** | Bem documentado que odds de abertura sao menos eficientes. Casas usam "market makers" que ajustam com base no fluxo. Alguns estudos mostram value em apostar cedo em favoritos. |
| **Comentarios** | **LIMITACAO:** Nao temos dado explicito de "abertura" no site. Proxy: primeira vez que coletamos odds do jogo. Quanto antes comecarmos a coletar, melhor o proxy. |

---

## Detalhamento por Hipotese

### H1: Incoerencias entre Mercados

**Logica teorica:**
```
P(Home Win) deve ser consistente entre:
- 1X2: P(H) = 1/odds_home (ajustado por overround)
- AH -0.5: Se home ganha, AH -0.5 home ganha. Logo P(AH -0.5 H win) ≈ P(Home Win)
- AH 0: P(AH 0 H win) = P(Home Win) + 0.5*P(Draw) [push no empate]
```

**Exemplo de incoerencia:**
```
1X2 Home: 2.50 → P(H) = 40%
AH -0.5 Home: 1.70 → P(H) = 59%
Diferenca: 19% → ANOMALIA
```

**Threshold sugerido:** Diferenca > 5% = flag para investigacao

---

### H2: Vieses Sistematicos

**Segmentos a testar:**

| Categoria | Segmentos |
|-----------|-----------|
| Dia | Segunda, Terca, ..., Domingo |
| Horario | Manha, Tarde, Noite |
| Liga tier | Tier 1 (top 5 europeias), Tier 2, Tier 3 |
| Tempo ate kickoff | <2h, 2-6h, 6-24h, 1-3d, >3d |
| Odds range | <1.30, 1.30-1.60, 1.60-2.00, 2.00-3.00, >3.00 |
| AH line | 0, -0.25, -0.5, -0.75, -1.0, etc |

**Cuidado estatistico:**
- Minimo 100 apostas por segmento para significancia
- Correcao de Bonferroni: p-value < 0.05/n_comparacoes
- Validacao: treino em 70% dados, teste em 30%

---

### H3: Quebra de Monotonicidade entre Linhas Adjacentes

**Logica:**
- Linhas de AH mais negativas (ex: -0.75 vs -0.5) sao mais faceis de ganhar para o home
- Portanto, a odd do home deve ser MENOR em linhas mais negativas
- Se essa relacao se inverte, ha anomalia

**Implementacao:**
```python
def check_line_monotonicity(ah_lines: dict) -> list:
    """
    Verifica se odds entre linhas adjacentes seguem ordem esperada.
    ah_lines: {handicap: {'home': odd, 'away': odd}, ...}
    """
    anomalies = []
    sorted_lines = sorted(ah_lines.items())  # [(−0.75, odds), (−0.5, odds), (−0.25, odds), ...]
    
    for i in range(1, len(sorted_lines)):
        prev_hcap, prev = sorted_lines[i-1]
        curr_hcap, curr = sorted_lines[i]
        
        # Se handicap fica MENOS negativo (ex: -0.5 -> -0.25)
        # Home odds deve AUMENTAR (mais dificil ganhar)
        if curr_hcap > prev_hcap:
            if curr['home'] < prev['home']:  # mas odds DIMINUIU
                anomalies.append({
                    'tipo': 'home_nao_monotonica',
                    'linhas': (prev_hcap, curr_hcap),
                    'odds': (prev['home'], curr['home']),
                    'delta_esperado': 'positivo',
                    'delta_real': curr['home'] - prev['home']
                })
    
    return anomalies
```

**Quando gravar evento:**
- A cada snapshot de odds, verificar monotonicidade entre todas as linhas disponíveis
- Se anomalia detectada, gravar: linhas envolvidas, odds, magnitude da inversao, timestamp

---

### H3b: Reversoes Temporais de Odds

**Logica:**
- Uma odd movendo consistentemente em uma direcao sugere tendencia clara
- Quando a direcao reverte, pode indicar: correcao de exagero, informacao nova, ou incerteza
- Apostas apos reversoes podem ter caracteristicas diferentes

**Implementacao:**
```python
def check_temporal_reversal(odd_history: list) -> dict:
    """
    Detecta reversao de direcao na serie temporal de uma odd.
    odd_history: [(timestamp, odd), (timestamp, odd), ...]
    """
    if len(odd_history) < 3:
        return None
    
    # Calcula direcoes dos ultimos movimentos
    prev_move = odd_history[-2][1] - odd_history[-3][1]
    curr_move = odd_history[-1][1] - odd_history[-2][1]
    
    prev_dir = 'up' if prev_move > 0 else 'down' if prev_move < 0 else 'flat'
    curr_dir = 'up' if curr_move > 0 else 'down' if curr_move < 0 else 'flat'
    
    # Reversao = mudanca de direcao
    if prev_dir != 'flat' and curr_dir != 'flat' and prev_dir != curr_dir:
        return {
            'reversal_detected': True,
            'direction_before': prev_dir,
            'direction_after': curr_dir,
            'magnitude': abs(curr_move),
            'timestamp': odd_history[-1][0]
        }
    
    return None
```

**Quando gravar evento:**
- A cada atualizacao de odd, verificar se houve reversao
- Se sim, gravar: mercado, direcao anterior, direcao nova, magnitude, streak antes da reversao

---

### H4: Steam Moves

**Definicao de steam move:**
```
Variacao > 5% em periodo < 30 minutos
OU
Variacao > 3% em periodo < 10 minutos
```

**Estrategias a testar:**
1. **Follow:** Apostar na direcao do movimento (odds caiu → apostar nesse lado)
2. **Fade:** Apostar contra (mercado overreacted)
3. **Wait:** Esperar estabilizar e apostar se odds voltar

---

### H5: Eficiencia por Liga

**Tiers sugeridos:**

| Tier | Ligas | Expectativa |
|------|-------|-------------|
| 1 | Premier League, La Liga, Serie A, Bundesliga, Ligue 1 | Muito eficiente |
| 2 | Championship, Serie B, Eredivisie, Liga Portugal | Eficiente |
| 3 | League 1, League 2, 2a divisao europeia | Menos eficiente |
| 4 | National League, ligas menores | Potencialmente ineficiente |

**Metricas de eficiencia:**
- Overround medio (menor = mais competitivo/eficiente)
- Frequencia de movimentos de odds
- Dispersao entre bookmakers

---

### H6: Correlacao entre Mercados

**Correlacoes esperadas:**

| Se isso acontece... | Entao isso deveria acontecer... |
|---------------------|--------------------------------|
| Over 2.5 cai (mais gols esperados) | AH favorito cai (vencer por mais) |
| Home 1X2 cai muito | AH 0 home cai, AH -0.5 home cai |
| Draw sobe | AH 0 away sobe (menos chance de vitoria) |

**Medicao:**
```python
correlacao = calcular_correlacao(
    delta_over_2_5,
    delta_ah_favorito
)
# Esperado: correlacao negativa (over cai → ah cai)
# Se correlacao ≈ 0 em alguns momentos → oportunidade
```

---

### H7: Odds de Abertura

**Limitacao importante:**
O site BetinAsia nao fornece explicitamente "odds de abertura". 

**Proxy possivel:**
- Usar a **primeira coleta** do jogo como proxy de abertura
- Quanto mais cedo comecarmos a coletar (ex: 7 dias antes), melhor o proxy
- Comparar primeira coleta vs ultima coleta (fechamento)

**Dado disponivel no site?**
- Nao encontramos API ou WebSocket que forneca odds de abertura historicas
- Teriamos que construir esse historico nos mesmos coletando

**Vieses documentados em abertura:**
- Favoritos tendem a abrir com odds mais altas e sao corrigidos para baixo
- "Opening line value" e conceito similar a CLV mas na abertura

---

## Variaveis Consolidadas para Coleta

| Variavel | Tipo | Fonte | Hipoteses |
|----------|------|-------|-----------|
| event_id | string | WebSocket | Todas |
| home_team | string | WebSocket | Todas |
| away_team | string | WebSocket | Todas |
| liga | string | WebSocket | H2, H5 |
| pais | string | WebSocket | H2, H5 |
| kickoff_time | datetime | WebSocket | H2, H7 |
| collected_at | datetime | Sistema | H4, H6, H7 |
| odds_1x2_h | float | WebSocket | H1, H2 |
| odds_1x2_d | float | WebSocket | H1 |
| odds_1x2_a | float | WebSocket | H1 |
| odds_ah_{line}_h | float | WebSocket | H1, H2, H3, H4 |
| odds_ah_{line}_a | float | WebSocket | H1, H2, H3, H4 |
| odds_ou_{line}_over | float | WebSocket | H1, H6 |
| odds_ou_{line}_under | float | WebSocket | H1, H6 |
| **resultado_jogo** | string | **Externa** | Todas |
| **placar_final** | string | **Externa** | Todas |

**Nota:** Resultado do jogo precisa ser coletado de fonte externa (API de resultados).

---

## Proximos Passos

1. [ ] Implementar coleta continua com todas as variaveis
2. [ ] Implementar scraper/API de resultados de jogos
3. [ ] Coletar dados por 2-4 semanas
4. [ ] Analise exploratoria de cada hipotese
5. [ ] Priorizar hipoteses com maior potencial
6. [ ] Backtesting das estrategias

---

*Documento criado em 01/02/2026*
