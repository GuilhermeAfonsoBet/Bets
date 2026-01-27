# Guia de Uso - Script de Extração de Odds BetInAsia para PAD

## Arquivos Disponíveis

1. **extrair_odds_betinasia.js** - Versão principal e otimizada (RECOMENDADO)
2. **extrair_dados_aposta_live_v2.js** - Versão alternativa com mais métodos de busca
3. **extrair_dados_aposta_live.js** - Versão inicial básica

## Como Usar no Power Automate Desktop

### Passo 1: Configurar a Ação "Executar função JavaScript"

No seu fluxo PAD, adicione a ação **"Executar função JavaScript na página da Web"**:

1. **Navegador**: Selecione a instância do navegador onde está o BetInAsia
2. **Função JavaScript**: Cole o conteúdo do arquivo `extrair_odds_betinasia.js`
3. **Resultado**: Armazenar em variável (ex: `dadosApostaLive`)

### Passo 2: Processar o Resultado

Use a ação **"Dividir o texto"** que você já tem configurada:
- **Texto a dividir**: `%dadosApostaLive%`
- **Delimitador**: Nova linha
- **Resultado**: `dadosApostaLiveDividido`

## Formato de Saída

O script retorna dados separados por quebra de linha no formato:

```
TIPO:Futebol : Final Da Partida
EVENTO:AEP Paphos FC vs. Slavia Praha
SELECAO:Slavia Praha +0.5 (Asian)
TOTAL:1.560|$15,731|1.589|$9,120|1.622|$368
3ET:1.585|$255
4CASTERS:1.592|$1,858|1.592|$628|1.595|$346
BDAQ:1.595|$752|1.603|$391|1.606|$722
BETDEX:1.600|$1,007
BF:1.587|$146|1.591|$1,329|1.595|$36
MOLLY:1.575|$348|1.582|$293
```

### Estrutura de Cada Linha

- **TIPO**: Categoria da aposta (Futebol, Basquete, etc.)
- **EVENTO**: Nome do jogo (Time1 vs. Time2)
- **SELECAO**: Aposta selecionada (handicap, over/under, etc.)
- **TOTAL**: Resumo de todos os agentes (odd_total|valor_total|odd_media|valor_media|odd_melhor|valor_melhor)
- **CASA**: Dados de cada casa (odd1|valor1|odd2|valor2|...)

## Acessando Dados no PAD

Após dividir o texto, acesse os elementos da lista:

```
%dadosApostaLiveDividido[0]%  → TIPO:Futebol : Final Da Partida
%dadosApostaLiveDividido[1]%  → EVENTO:AEP Paphos FC vs. Time
%dadosApostaLiveDividido[2]%  → SELECAO:Seleção escolhida
...
```

Para extrair apenas o valor após ":", use outra ação "Dividir texto" com delimitador ":".

## Tratamento de Erros

Se o painel de apostas não estiver aberto, o script retorna:
```
ERRO: Tabela de odds não encontrada. Abra uma aposta primeiro.
```

## Substituindo o Seletor Atual

Este script substitui a necessidade de usar seletores de elemento de interface do usuário (UIAutomation) para capturar dados do painel Group. Vantagens:

1. **Mais confiável**: Não depende de classes CSS que podem mudar
2. **Mais rápido**: Uma única chamada JavaScript vs. múltiplos cliques
3. **Mais dados**: Captura toda a tabela de odds de uma vez
4. **Menos manutenção**: Adaptável a mudanças no layout do site
