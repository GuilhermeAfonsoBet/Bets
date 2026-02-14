# Relatorio Executivo H3B (versao v5 corrigida para decisao)

Data-base: 14/02/2026 10:27 UTC  
Fonte principal: execucao do `hypothesis_performance_robust.sh` com janela de 14 dias e filtro `v4.0-api`.

---

## Leitura em 60 segundos

- Voce tem razao: a versao anterior deu peso demais para significancia de diferenca de preco.
- Neste v5, a leitura principal passa a ser **retorno realizado** (ROI/P&L e CLV com IC).
- A analise de `diff` fica como **diagnostico de entrada**, nao como criterio final de decisao economica.
- A separacao por versao do robo nao apareceu no v4 anterior porque o comando foi executado com filtro fixo em `v4.0-api`.

---

## 1) Correcao do ponto da janela de 14 dias

### O que o numero "14 dias" significa

- "14 dias" e o **teto da janela de consulta**.
- Com filtro em uma versao especifica, a amostra real pode ser menor que 14 dias.
- Entao, a pergunta "v4.0 esta ativo ha menos de 14 dias?" e valida, e o relatorio anterior nao deixou isso claro.

### Correcao editorial aplicada

- Neste documento, a janela de 14 dias e tratada como limite maximo.
- A duracao real da versao precisa ser exibida junto com:
  - primeiro registro da versao,
  - ultimo registro da versao,
  - dias efetivamente cobertos.

---

## 2) Correcao do ponto de performance (fila e tempo total)

### 2.1 Painel consolidado do recorte atual (14 dias, v4.0-api)

| Indicador operacional | Valor |
|---|---:|
| Auditorias OK | 2654 |
| Espera em fila (media) | 19.122 ms |
| Espera em fila (p95) | 90.673 ms |
| Espera em fila (p99) | 524.017 ms |
| Profundidade fila enqueue (media) | 0,96 |
| Profundidade fila enqueue (p95) | 8,10 |
| Profundidade fila enqueue (max) | 22 |
| Tempo total do bot (media) | 4.511 ms |
| Tempo total do bot (p95) | 6.921 ms |
| Tempo total do bot (p99) | 23.703 ms |

Leitura:
- Este bloco esta misturando periodos diferentes da operacao dentro da janela.
- Por isso, ele pode esconder melhora recente (efeito de media com cauda historica).

### 2.2 Janela recente apos separacao de esteiras (fonte operacional dedicada)

| Indicador | Antes | Depois (Fase 1) |
|---|---:|---:|
| Queue depth enqueue (media) | 6,93 | 0,04 |
| Queue wait (media) | 42.956 ms | 0,3 ms |
| Queue wait (p90) | alto | 1,0 ms |
| Queue wait (p95) | alto | 2,3 ms |

Leitura:
- Aqui aparece exatamente o ponto que voce cobrou: **fila sub-segundo** na janela recente pos-mudanca.
- Ou seja: os dois fatos podem coexistir:
  - consolidado 14d pior (mistura historica),
  - janela recente muito melhor (pos-refatoracao).

---

## 3) O que importa para decisao: retorno com intervalo de confianca

Base: tabela de resultado realizado (profit/loss e CLV), recorte informado na sua execucao.

| Estrategia | Cobertura de P&L | Media P&L (stake=1) | IC95 P&L | Cobertura de CLV | Media CLV | IC95 CLV |
|---|---:|---:|---|---:|---:|---|
| H1 | 3,6% | +0,4329 | [0,3585 ; 0,5074] | 3,6% | -0,4677% | [-1,3601 ; 0,4246] |
| H3 | 8,0% | -0,0074 | [-0,2683 ; 0,2534] | 8,0% | +6,3414% | [-3,7893 ; 16,4721] |
| H3B | 1,7% | -0,1042 | [-0,1989 ; -0,0096] | 3,3% | +150,5592% | [41,1827 ; 259,9357] |
| H6 | 10,0% | -0,0950 | [-0,1600 ; -0,0301] | 10,0% | +0,0682% | [-0,3922 ; 0,5286] |

Leitura executiva:
- Esta e a tabela que deve comandar decisao economica.
- Em H3B, o P&L medio veio negativo com IC95 abaixo de zero, **mas** com cobertura de P&L ainda muito baixa (1,7%).
- Resultado: ainda nao e base suficiente para escalar stake com confianca.

---

## 4) Back e Lay: diagnostico correto de entrada e risco

### Back (sinal de entrada)

| Medida | Valor |
|---|---:|
| Casos no grupo Back | 1015 |
| Diferenca media BS vs WS | +45,27% |
| Mediana | +10,87% |

### Lay (sinal de entrada + risco de cauda)

| Medida | Valor |
|---|---:|
| Casos no grupo Lay | 342 |
| Diferenca media BS vs WS | -11,13% |
| Liability p95 | 438,23 |
| Liability p99 | 3.772,79 |
| ES95 de liability | 2.131,85 |
| Maximo observado | 4.386,23 |

Leitura executiva:
- Para Lay, a regra de decisao nao pode ser so media de edge.
- O controle deve ser por risco de cauda: p95/p99/ES95 + limite por janela.

---

## 5) Correcao do ponto sobre "significancia de diff" em combinacoes

Voce esta certo no diagnostico: significancia de diferenca de preco em combinacao (ex.: secao antiga 10.2) **nao** e o alvo final de negocio.

Correcao de leitura:
- `diff` = filtro/triagem de entrada.
- Decisao de capital = significancia e estabilidade de retorno (CLV e ROI/P&L), com controle de risco.

Neste v5:
- combinacoes por `diff` ficam apenas como anexo de screening;
- prioridade de leitura sobe para retorno + IC + cobertura de liquidacao.

---

## 6) Resposta direta aos 6 pontos levantados

1. **Janela 14 dias vs vida real da versao:** corrigido no texto; 14d e teto, nao garantia de duracao efetiva.  
2. **Separacao por versao e melhora recente de fila:** corrigido; consolidado 14d e separado de janela recente pos-mudanca (sub-segundo).  
3. **Estilo de leitura e nomenclatura:** reescrito em linguagem executiva, menos tecnica, com tabelas visuais.  
4. **IC no que importa:** foco movido para retorno (P&L/ROI unitario e CLV) com IC95.  
5. **Portugues e fluidez:** texto revisado e simplificado.  
6. **Secao 10.2 (significancia de diff):** rebaixada para diagnostico de entrada, nao de retorno.

---

## 7) Proxima rodada recomendada (para fechar decisao com rigor)

Para publicar a versao final "pronta para stake", faltam dois fechamentos objetivos:

1. Separar automaticamente desempenho por versao e por janela recente (6h/24h/14d).  
2. Medir retorno Back/Lay por versao (CLV e ROI/P&L com IC95), nao apenas diferenca de preco.

Com isso, a decisao deixa de ser "sinal de entrada" e passa a ser "qualidade de retorno ajustada a risco".

