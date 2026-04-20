# Guia de Uso - Script de Extração de Odds BetInAsia para PAD

## Arquivos Disponíveis

### COMPATÍVEL COM FLUXO PAD EXISTENTE (RECOMENDADO):
1. **extrair_dados_aposta_live_compativel.js** - Retorna texto bruto igual ao seletor UIAutomation

### PARA DEBUG:
2. **debug_extrair_dados.js** - Execute no Console do navegador (F12) para ver formato exato

### VERSÕES ALTERNATIVAS (formato diferente):
3. **extrair_odds_betinasia.js** - Versão com formato estruturado (prefixos TIPO:, EVENTO:, etc.)
4. **extrair_dados_aposta_live_v2.js** - Versão alternativa estruturada
5. **extrair_dados_aposta_live.js** - Versão inicial básica

## Como Usar no Power Automate Desktop

### Passo 1: Substituir o Seletor UIAutomation

No seu fluxo PAD, **substitua** a ação de captura de elemento de interface por **"Executar função JavaScript na página da Web"**:

1. **Navegador**: Selecione a instância do navegador onde está o BetInAsia
2. **Função JavaScript**: Cole o conteúdo do arquivo `extrair_dados_aposta_live_compativel.js`
3. **Resultado**: Armazenar em variável `dadosApostaLive`

### Passo 2: Manter o Fluxo Existente

O resto do fluxo **NÃO precisa ser alterado**! A ação "Dividir o texto" (linha 304) e todo o processamento subsequente funcionam normalmente:

- **Linha 304**: Dividir `dadosApostaLive` por Nova linha → `dadosApostaLiveDividido`
- **Linhas 305-310**: Definir variáveis (QtdItens, pattern_odd, pattern_stake, etc.)
- **Linhas 311+**: Loops de processamento (buscar "Todos Os Agentes De Apostas", extrair odds, stakes, casas)

### Como Funciona

O script `extrair_dados_aposta_live_compativel.js` retorna o **texto bruto** do painel Group, exatamente como o seletor UIAutomation faria. O formato de saída é idêntico:

```
Clássico
Intercâmbio
Start Acca
Futebol : Final Da Partida
AEP Paphos FC vs. Slavia Praha
Slavia Praha +0.5 (Asian)
Tempo Limite
72 Hours
Aposta
$
...
Todos Os Agentes De Apostas
TOTAL
MÉDIA
MELHOR
1.560
$15,731
1.589
$9,120
1.622
$368
3et
1.585
$255
4casters
1.592
$1,858
...
```

As regexes existentes no PAD funcionam perfeitamente com este formato:
- `pattern_odd` = `'^\s*\d+(?:[.,]\d{2,3})?\s*$'` → captura "1.560", "1.592", etc.
- `pattern_stake` = `'^\s*(?:R\$|\$)\s*\d+(?:[.,]\d{2,3})?\s*$'` → captura "$15,731", "$628", etc.

## Debug - Verificar Formato do Site

Se precisar verificar exatamente qual formato o site está retornando:

1. Abra o site BetInAsia com uma aposta selecionada
2. Pressione F12 para abrir o DevTools
3. Vá para a aba "Console"
4. Cole e execute o conteúdo de `debug_extrair_dados.js`
5. O console mostrará cada linha numerada e classificada (ODD, STAKE, CASA, etc.)

## Tratamento de Erros

Se o painel de apostas não estiver aberto, o script retorna:
```
ERRO: Painel de apostas não encontrado
```

No PAD, você pode verificar se `dadosApostaLive` começa com "ERRO:" para tratar essa situação.

## Vantagens sobre o Seletor UIAutomation

1. **Mais confiável**: Usa múltiplos métodos de busca (classes CSS + conteúdo)
2. **Mais rápido**: Uma única chamada JavaScript vs. interações UIAutomation
3. **Compatível**: Retorna exatamente o mesmo formato que o seletor atual
4. **Menos manutenção**: Se as classes CSS mudarem, busca alternativa por conteúdo

## Troubleshooting

### Script não encontra o painel
- Verifique se o painel "Group" está visível na tela
- Certifique-se de que uma aposta está selecionada (mostrando odds)

### Dados retornados incompletos
- O painel pode ainda estar carregando - adicione um delay antes da execução
- Verifique se a aba "Clássico" está selecionada (não "Intercâmbio")

### Formato diferente do esperado
- Execute `debug_extrair_dados.js` no Console para verificar estrutura real
- Compare com o formato esperado pelo seu fluxo PAD
