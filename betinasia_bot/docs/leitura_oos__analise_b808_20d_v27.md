# Leitura crítica (caso concreto) — OOS (Seção 12) do relatório `analise_b808_20d__v27_excl_days`
**Fonte:** relatório `analise_b808_20d__v27_excl_days.md` (execução 23/02/2026 00:01 UTC)  
**Objetivo deste documento:** responder, com base nos números do relatório, se há edge, como é o retorno por risco, e se seleção/sizing/budget estão fazendo sentido.  

---

## 1) Resumo executivo (o que dá para concluir hoje)

- **Edge de execução (pre‑match) existe e é estatisticamente robusto:** o relatório mostra **CLV pre‑match (Betslip vs closing)** com média robusta por jogo **+0,936%** e **IC90 [+0,629%, +1,244%]** (N=1968 eventos; 283 jogos). Isso é um sinal forte de que, quando o pipeline “funciona”, vocês entram em preço melhor que o closing em média.
- **Edge monetizável (P&L OOS) aparece, mas não é robusto a governança por jogo:** no OOS (Seção 12), a extrapolação “30d” (exp.) dá **Lucro +270,75** com **Banca recomendada 2283,94** e **ROI/banca 11,85%** (DD p95 exp. ≈ 141,99). Porém, quando se aplica budget por jogo (12.2), **todas** as parametrizações testadas levam o lucro exp. para **negativo**.
- **Interpretação prática:** hoje dá para dizer “há edge de entrada” (CLV), mas o “edge que vira dinheiro” ainda parece **frágil** e possivelmente **dependente de concentração** (muitos sinais em poucos jogos) ou de uma **parametrização de budget** que está cortando justamente onde o ganho ocorre.

---

## 2) Temos edge nesta operação?

### 2.1 Evidência pró‑edge (forte)
1) **CLV pre‑match positivo e significativo:** +0,936% (IC90 positivo).  
Isso costuma ser um dos melhores indicadores precoces de “edge/execução” em apostas (menos dependente do ruído do resultado final).

2) **OOS agregado (ponderado por turnover) é positivo no recorte mostrado:** somando as 4 janelas do quadro principal (12):
- **Lucro total (estratégia, budget):** \(-14,06 - 45,47 + 53,83 + 77,90 = +72,20\)
- **Turnover total (teste):** \(361,46 + 827,81 + 2499,90 + 1862,80 = 5551,97\)
- **ROI ponderado (lucro/turnover):** \(72,20 / 5551,97 = +1,30%\)

Isso é **edge pequeno**, mas é “dinheiro” no OOS (não é só CLV).

### 2.2 Evidência contra robustez (importante)
1) **Os ROIs OOS por janela têm IC90 muito largos e cruzam 0.** No quadro, 3 de 4 janelas têm ROI médio negativo e 1 positivo, mas **nenhuma** delas tem IC90 “todo acima de 0”.

2) **Budget derruba o lucro para negativo em todos os cenários testados (Seção 12.2).** Isso significa que:
- ou o baseline (sem budget) está capturando lucro por **concentração / correlação intra‑jogo** (não é “free lunch”),
- ou o budget está **mal calibrado** e está cortando o “filé”.

Conclusão concreta: **há sinais de edge**, mas ainda não há robustez suficiente para afirmar “edge replicável em produção com governança por jogo” sem ajustes.

---

## 3) Retorno pelo risco: faz sentido?

Pelo bloco 12.1 (OOS → 30 dias, “exp.”):
- **Lucro 30d (exp.)**: +270,75  
- **Banca recomendada (max)**: 2283,94  
- **ROI/banca 30d (exp.)**: +11,85%  
- **DD 30d p95 (exp.)**: 141,99  

Leitura: **o retorno por risco parece ótimo no baseline** (DD p95 ≈ 6,2% da banca recomendada).  
Mas: como o budget por jogo vira lucro negativo, esse “bom retorno por risco” é **sensível** ao modelo de governança. Para produção, vocês precisam de um budget que **não destrua** o edge (se ele existir) e ao mesmo tempo **controle concentração**.

---

## 4) Tabela “Train window” (Seção 12): respostas diretas

### 4.1 “3 buckets com ROI < 0 e 1 bucket ROI > 0” ⇒ operação sem valor?
**Não dá para concluir isso** apenas pela contagem de sinais, porque:
- os IC90 são largos (amostra pequena por step),
- vocês estão fazendo seleção de combinações no treino e medindo no teste (processo ruidoso por natureza),
- e o resultado **agregado ponderado** nas 4 janelas ficou **positivo** (+1,30% por turnover).

O que dá para concluir: **não está estável**; o edge (se existe) ainda “oscila” no curto prazo.

### 4.2 Por que ROI tem IC (se não é “apenas o realizado do bucket”)?
Porque o relatório está tratando ROI como **estimativa do ROI esperado** (média robusta por jogo), usando bootstrap por cluster (jogo).  
Mesmo com ROI realizado, o IC responde: “se eu repetir essa política em outros jogos, qual faixa plausível do ROI médio?”.

### 4.3 “No último bucket, ROI negativo mas lucro positivo” — o que indica?
Isso indica **mismatch de agregação**, não “mágica”. No quadro, a coluna **ROI OOS (mean; IC90)** é uma métrica **não ponderada por stake** (média por jogo/cluster). Já “Lucro” é **ponderado por sizing** e somado em dinheiro.

Para deixar isso explícito, segue a mesma tabela com o ROI ponderado (\(lucro/turnover\)):

| Train window | Test window | ROI OOS (mean; IC90) | Turnover | Lucro | ROI ponderado (lucro/turnover) |
|---|---|---:|---:|---:|---:|
| 08→09 | 10→11 | -8,43% [-25,57%, +9,30%] | 361,46 | -14,06 | -3,89% |
| 08→11 | 12→13 | -11,14% [-38,51%, +15,94%] | 827,81 | -45,47 | -5,49% |
| 08→13 | 14→15 | +11,17% [-4,49%, +26,97%] | 2499,90 | +53,83 | +2,15% |
| 08→15 | 16→19 | -4,33% [-23,28%, +15,14%] | 1862,80 | +77,90 | +4,18% |

Interpretação do último step: **com sizing/budget aplicado, o capital foi alocado de modo que o P&L ficou positivo**, apesar de a média por jogo (não ponderada) ter sido negativa.

---

## 5) Budget vs Baseline (Seção 12.2): está destruindo valor?

Números do relatório:
- **BASELINE (sem budget)**: Turnover 30d 20819,90 | **Lucro 30d (exp.) +270,75** | ROI/banca +11,85% | DD p95 134,26
- Com budget, o lucro exp. fica **negativo em todos os cenários**, por exemplo:
  - 1,00%/0,50% cap33%: lucro -171,58
  - 4,00%/2,00% cap50%: lucro -283,30

Leitura concreta (duas hipóteses compatíveis com os números):

1) **O lucro baseline depende de concentração intra‑jogo**, e quando você impõe uma governança realista por `match_id`, a expectativa cai para negativo.  
Isso é comum quando o modelo “ganha” por repetir entradas correlacionadas no mesmo jogo.

2) **O budget está mal parametrizado para este edge** (principalmente para Lay), cortando os eventos mais lucrativos.  
Isso também é plausível porque:
- no OOS, Lay aparece como combinação ativa em múltiplos steps (`Lay_In_Yes/No` e `Lay_Pre_Yes/No` entram),
- e, operacionalmente, Lay pode estar capturando reversões onde o payoff é grande, mas o budget em liability é menor (0,5× do Back em vários cenários).

**Conclusão prática:** pelo caso concreto, do jeito que está, **o budget está destruindo valor na simulação**. A pergunta não é “tirar budget”, e sim **recalibrar**:
- testar um grid onde **Lay não seja penalizado por default** (ex.: Lay budget = Back budget), e
- definir budget por jogo por um critério de risco (CVaR/ES por match) em vez de percentuais fixos.

---

## 6) Seleção e sizing no walk-forward: está adequado? como melhorar com os mesmos dados?

### 6.1 O que está bom (concreto)
- O WF com `wf_test_days=2` e `wf_step_days=2` evita sobreposição e dá leitura somável.
- A regra de bloqueio “ROI significativamente negativo ⇒ não ativa” é um bom guard‑rail.
- Separar Pre vs In é essencial (CLV/closing não vale in‑match).

### 6.2 Melhorias objetivas (alto impacto) que atacam os problemas vistos no v27
1) **Reportar (e usar na decisão) ROI ponderado e não‑ponderado.**  
Hoje a tabela principal mistura ROI(mean) (não ponderado) com lucro (ponderado). Isso gera leituras erradas (“ROI negativo com lucro positivo”). Sugestão: mostrar os dois e, para sizing, usar o ponderado.

2) **Excluir regimes comprovadamente ruins de execução:** na Seção 2.3, o bucket **10–20s** tem ROI médio **-20,94%** com IC90 todo < 0 (negativo significativo). Isso é um candidato forte a “bloqueio operacional”.

3) **Shrinkage/Bayes hierárquico para ativação (partial pooling).**  
Pelo próprio relatório, os steps ainda têm N pequeno por combinação em alguns casos; shrinkage reduz “liga/desliga” por ruído e tende a estabilizar o OOS.

4) **Recalibrar Kelly:** a curva (9.4b) sugere que `KELLY_0.25` piora vs `KELLY_0.10` (lucro 30d +59,99 no 0.10 vs -798,36 no 0.25 no bloco PRE). Isso é um sinal de que o sizing está “overbetting” dado o erro de estimação.

5) **Budget por jogo baseado em risco observado (por match_id), não em percentuais fixos.**  
O resultado da 12.2 diz que o budget atual não preserva o edge. O caminho estado‑da‑arte aqui é calibrar budgets para manter a cauda (DD/CVaR) sob controle **sem matar o retorno esperado**.

---

## 7) Próximos passos recomendados (concretos)
1) Ajustar o relatório para sempre exibir **ROI ponderado (lucro/turnover)** lado a lado com **ROI(mean)**.
2) Rodar OOS com um filtro operacional “**proibir 10–20s**” e comparar 12.1/12.2.
3) Rodar uma sensibilidade de budget onde **Lay budget = Back budget** e comparar com baseline.
4) Se a meta é produção: escolher um sizing conservador tipo `KELLY_0.10` + caps, e operar somente buckets de execução rápidos (ex.: `<10s`).

