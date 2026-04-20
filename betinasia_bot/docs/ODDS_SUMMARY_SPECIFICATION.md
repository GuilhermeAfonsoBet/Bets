# Especificacao da Tabela odds_summary

**Data:** Fevereiro 2026  
**Versao:** 1.0

---

## 1. Visao Geral

A tabela `odds_summary` armazena um resumo compactado de todas as movimentacoes de odds de cada linha/lado de um jogo. Cada registro representa:

- **Um jogo** (ex: Liverpool vs Manchester City)
- **Um mercado** (ex: Asian Handicap)
- **Uma linha** (ex: -1.25)
- **Um lado** (ex: Home)

### Exemplo de Registros para um Jogo

| match_id | market | line | side | opening | closing | ... |
|----------|--------|------|------|---------|---------|-----|
| 123 | AH | -1.25 | home | 1.95 | 1.88 | ... |
| 123 | AH | -1.25 | away | 2.05 | 2.12 | ... |
| 123 | AH | -1.0 | home | 1.75 | 1.72 | ... |
| 123 | AH | -1.0 | away | 2.25 | 2.28 | ... |
| 123 | OU | 2.5 | over | 1.90 | 1.85 | ... |
| 123 | OU | 2.5 | under | 2.00 | 2.05 | ... |
| 123 | 1X2 | - | home | 1.50 | 1.48 | ... |
| 123 | 1X2 | - | draw | 4.20 | 4.30 | ... |
| 123 | 1X2 | - | away | 6.50 | 6.80 | ... |

---

## 2. Estrutura da Tabela

### 2.1 Identificacao do Jogo

| Campo | Tipo | Descricao |
|-------|------|-----------|
| `id` | INTEGER | Chave primaria auto-incremento |
| `match_id` | INTEGER | FK para tabela matches |
| `event_id` | VARCHAR(100) | ID externo do BetinAsia (ex: "2026-02-01,2,12") |
| `home_team` | VARCHAR(200) | Nome do time da casa |
| `away_team` | VARCHAR(200) | Nome do time visitante |
| `league` | VARCHAR(200) | Nome da liga |
| `country` | VARCHAR(50) | Pais da liga |
| `kickoff_time` | TIMESTAMP | Data/hora do inicio do jogo (UTC) |

### 2.2 Identificacao da Linha

| Campo | Tipo | Descricao |
|-------|------|-----------|
| `market_type` | VARCHAR(10) | Tipo de mercado: "AH", "OU", "1X2" |
| `line` | FLOAT | Valor da linha (ex: -1.25, 2.5). NULL para 1X2 |
| `side` | VARCHAR(10) | Lado: "home", "away", "over", "under", "draw" |

### 2.3 Odds de Abertura

| Campo | Tipo | Descricao |
|-------|------|-----------|
| `opening_odds` | FLOAT | Primeira odd coletada para esta linha/lado |
| `opening_time` | TIMESTAMP | Momento da primeira coleta (UTC) |
| `minutes_to_kick_at_open` | INTEGER | Minutos ate o kickoff no momento da abertura |

**Calculo:**
```python
opening_odds = primeira_coleta.odds
opening_time = primeira_coleta.scraped_at
minutes_to_kick_at_open = (kickoff_time - opening_time).total_seconds() / 60
```

### 2.4 Odds de Fechamento (Closing Line)

| Campo | Tipo | Descricao |
|-------|------|-----------|
| `closing_odds` | FLOAT | Ultima odd coletada ANTES do kickoff |
| `closing_time` | TIMESTAMP | Momento da ultima coleta antes do kickoff (UTC) |
| `minutes_to_kick_at_close` | INTEGER | Minutos ate o kickoff no momento do fechamento |

**Calculo:**
```python
# Filtra coletas antes do kickoff
coletas_pre_jogo = [c for c in coletas if c.scraped_at < kickoff_time]

# Pega a ultima
closing_odds = coletas_pre_jogo[-1].odds
closing_time = coletas_pre_jogo[-1].scraped_at
minutes_to_kick_at_close = (kickoff_time - closing_time).total_seconds() / 60
```

**Nota:** A closing line e considerada a odd mais "eficiente" do mercado, pois incorpora toda informacao disponivel ate o inicio do jogo.

### 2.5 Estatisticas de Movimentacao

| Campo | Tipo | Descricao |
|-------|------|-----------|
| `min_odds` | FLOAT | Menor odd registrada |
| `max_odds` | FLOAT | Maior odd registrada |
| `avg_odds` | FLOAT | Media aritmetica das odds |
| `std_odds` | FLOAT | Desvio padrao das odds |
| `num_collections` | INTEGER | Quantidade de coletas |

**Calculos:**

```python
odds_list = [c.odds for c in coletas]

min_odds = min(odds_list)
max_odds = max(odds_list)
avg_odds = sum(odds_list) / len(odds_list)
num_collections = len(odds_list)

# Desvio padrao
variance = sum((x - avg_odds) ** 2 for x in odds_list) / len(odds_list)
std_odds = sqrt(variance)
```

**Interpretacao do Desvio Padrao:**
- `std_odds < 0.02`: Odds muito estavel (pouca movimentacao)
- `std_odds 0.02-0.05`: Movimentacao normal
- `std_odds > 0.05`: Alta volatilidade (muito movimento)

### 2.6 Metricas de Movimentacao

| Campo | Tipo | Descricao |
|-------|------|-----------|
| `movement_pct` | FLOAT | Variacao percentual entre abertura e fechamento |
| `range_pct` | FLOAT | Range (max-min) como percentual da media |
| `direction` | VARCHAR(10) | Direcao do movimento: "up", "down", "stable" |

**Calculos:**

```python
# Variacao percentual total
movement_pct = ((closing_odds - opening_odds) / opening_odds) * 100

# Range como percentual
range_pct = ((max_odds - min_odds) / avg_odds) * 100

# Direcao
if movement_pct > 1.0:
    direction = "up"      # Odds subiu (lado ficou menos favorito)
elif movement_pct < -1.0:
    direction = "down"    # Odds caiu (lado ficou mais favorito)
else:
    direction = "stable"  # Pouca mudanca
```

**Interpretacao:**
- `movement_pct > 0`: Odds subiu = mercado esta apostando CONTRA este lado
- `movement_pct < 0`: Odds caiu = mercado esta apostando A FAVOR deste lado
- Odds que caem geralmente indicam "smart money" entrando

### 2.7 Steam Moves (Movimentos Bruscos)

| Campo | Tipo | Descricao |
|-------|------|-----------|
| `steam_moves_count` | INTEGER | Numero de movimentos > 3% entre coletas consecutivas |
| `max_single_move_pct` | FLOAT | Maior movimento percentual em uma unica coleta |
| `avg_move_per_collection` | FLOAT | Movimento medio entre coletas consecutivas |

**Calculos:**

```python
moves = []
for i in range(1, len(coletas)):
    prev_odds = coletas[i-1].odds
    curr_odds = coletas[i].odds
    move_pct = abs((curr_odds - prev_odds) / prev_odds) * 100
    moves.append(move_pct)

# Conta movimentos bruscos (> 3%)
steam_moves_count = sum(1 for m in moves if m > 3.0)

# Maior movimento individual
max_single_move_pct = max(moves) if moves else 0

# Media de movimento entre coletas
avg_move_per_collection = sum(moves) / len(moves) if moves else 0
```

**Interpretacao:**
- `steam_moves_count > 0`: Houve entrada de dinheiro significativo
- `max_single_move_pct > 5%`: Movimento muito forte (possivel informacao privilegiada)

### 2.8 Resultado do Jogo

| Campo | Tipo | Descricao |
|-------|------|-----------|
| `home_score` | INTEGER | Gols do time da casa |
| `away_score` | INTEGER | Gols do time visitante |
| `bet_result` | VARCHAR(20) | Resultado da aposta neste lado |
| `profit_loss` | FLOAT | Lucro/prejuizo para stake=1 |

**Valores de `bet_result`:**

Para Asian Handicap:
- `win`: Aposta ganha
- `loss`: Aposta perde
- `half_win`: Ganha metade (linhas .25/.75)
- `half_loss`: Perde metade (linhas .25/.75)
- `push`: Empate (devolve stake)

Para 1X2:
- `win`: Acertou o resultado
- `loss`: Errou o resultado

**Calculo do Resultado AH:**

```python
def calculate_ah_result(home_score, away_score, line, side):
    """
    Calcula resultado de aposta Asian Handicap.
    
    Args:
        home_score: Gols do home
        away_score: Gols do away
        line: Linha de handicap (ex: -1.25)
        side: "home" ou "away"
    
    Returns:
        (result, profit_loss) para stake=1
    """
    
    # Ajusta placar com handicap
    if side == "home":
        adjusted_diff = (home_score + line) - away_score
    else:  # away
        adjusted_diff = (away_score - line) - home_score
    
    # Determina resultado
    if adjusted_diff > 0.5:
        return ("win", odds - 1)  # Ganha: lucro = odds - 1
    elif adjusted_diff < -0.5:
        return ("loss", -1)  # Perde: prejuizo = -1
    elif adjusted_diff == 0.5:
        return ("half_win", (odds - 1) / 2)  # Ganha metade
    elif adjusted_diff == -0.5:
        return ("half_loss", -0.5)  # Perde metade
    else:  # adjusted_diff == 0
        return ("push", 0)  # Empate: devolve stake
```

**Exemplos:**

| Placar | Line | Side | Adjusted Diff | Result |
|--------|------|------|---------------|--------|
| 2-1 | -1.0 | home | (2-1)-1 = 0 | push |
| 2-1 | -0.5 | home | (2-0.5)-1 = 0.5 | half_win |
| 2-1 | -1.5 | home | (2-1.5)-1 = -0.5 | half_loss |
| 2-1 | +0.5 | away | (1+0.5)-2 = -0.5 | half_loss |

### 2.9 Metricas de Valor

| Campo | Tipo | Descricao |
|-------|------|-----------|
| `clv` | FLOAT | Closing Line Value |
| `clv_pct` | FLOAT | CLV como percentual |
| `expected_value` | FLOAT | Valor esperado teorico |

**Calculo do CLV:**

```python
# CLV = diferenca entre odds de abertura e fechamento
# CLV positivo = voce "bateu" a closing line (bom!)
# CLV negativo = mercado se moveu contra voce

clv = opening_odds - closing_odds
clv_pct = ((opening_odds - closing_odds) / closing_odds) * 100
```

**Interpretacao:**
- `clv_pct > 0`: Voce apostou em odds melhor que o fechamento (edge potencial)
- `clv_pct < 0`: Voce apostou em odds pior que o fechamento
- CLV positivo consistente = indicador de apostador lucrativo

### 2.10 Metadados

| Campo | Tipo | Descricao |
|-------|------|-----------|
| `created_at` | TIMESTAMP | Quando o resumo foi criado |
| `updated_at` | TIMESTAMP | Ultima atualizacao |

---

## 3. Indices Recomendados

```sql
-- Para buscas por jogo
CREATE INDEX idx_summary_match ON odds_summary(match_id);

-- Para buscas por mercado/linha
CREATE INDEX idx_summary_market ON odds_summary(market_type, line);

-- Para buscas por liga
CREATE INDEX idx_summary_league ON odds_summary(league);

-- Para buscas por data
CREATE INDEX idx_summary_kickoff ON odds_summary(kickoff_time);

-- Para buscas por resultado
CREATE INDEX idx_summary_result ON odds_summary(bet_result);
```

---

## 4. Queries de Exemplo

### 4.1 Jogos com maior CLV positivo

```sql
SELECT home_team, away_team, market_type, line, side,
       opening_odds, closing_odds, clv_pct, bet_result
FROM odds_summary
WHERE clv_pct > 2.0
ORDER BY clv_pct DESC
LIMIT 20;
```

### 4.2 Analise de ROI por liga

```sql
SELECT league,
       COUNT(*) as total_bets,
       SUM(CASE WHEN bet_result = 'win' THEN 1 ELSE 0 END) as wins,
       SUM(profit_loss) as total_profit,
       AVG(profit_loss) * 100 as roi_pct
FROM odds_summary
WHERE market_type = 'AH' AND line = 0
GROUP BY league
ORDER BY roi_pct DESC;
```

### 4.3 Steam moves que resultaram em win

```sql
SELECT home_team, away_team, line, side,
       steam_moves_count, direction, bet_result
FROM odds_summary
WHERE steam_moves_count > 0
  AND direction = 'down'  -- Odds caiu (favorecido)
  AND bet_result IN ('win', 'half_win')
ORDER BY steam_moves_count DESC;
```

### 4.4 Volatilidade por mercado

```sql
SELECT market_type, line,
       AVG(std_odds) as avg_volatility,
       AVG(range_pct) as avg_range
FROM odds_summary
GROUP BY market_type, line
ORDER BY avg_volatility DESC;
```

---

## 5. Fluxo de Processamento

```
1. Jogo termina
   |
2. Busca resultado na API-Football
   |
3. Atualiza tabela matches (home_score, away_score, status='finished')
   |
4. Para cada linha/lado coletado:
   |
   4.1 Busca todas coletas em best_odds_history
   |
   4.2 Calcula metricas (abertura, fechamento, estatisticas)
   |
   4.3 Calcula resultado da aposta (bet_result, profit_loss)
   |
   4.4 Insere em odds_summary
   |
5. (Opcional) Limpa dados antigos de best_odds_history
```

---

## 6. Retencao de Dados

| Tabela | Retencao | Tamanho Estimado |
|--------|----------|------------------|
| `best_odds_history` | 7 dias | ~50-100 MB |
| `odds_summary` | **Permanente** | ~50 MB/ano |
| `matches` | Permanente | ~5 MB/ano |

---

## 7. Validacoes

Antes de inserir um registro, validar:

1. `opening_odds > 1.0` e `closing_odds > 1.0`
2. `opening_time < closing_time < kickoff_time`
3. `num_collections >= 2` (minimo para ter abertura e fechamento)
4. `min_odds <= avg_odds <= max_odds`
5. `std_odds >= 0`

---

*Documento gerado em Fevereiro 2026*
