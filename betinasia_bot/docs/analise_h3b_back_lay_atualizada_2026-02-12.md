# Análise H3B Atualizada — Back e Lay (dados de 12/02/2026)

**Data da consolidação:** 12/02/2026  
**Escopo:** H3B com base nos artefatos mais recentes (`01_sync_audit_matches.log`, `02_update_results.log`, `03_ws_vs_bs.log`, `04_h3b_comprehensive.log`, `05_lay_bucket_analysis.txt`, `06_lay_target_summary.txt`, `runner.log`).  
**Objetivo:** comparar qualidade/valor de **Back** e avaliar a hipótese de **Lay quando BS << WS**.

---

## 1) Resumo executivo

1. **Back via Betslip continua fraco no agregado**: CLV adicional pre-match no BS ficou **negativo e significativo**.
2. **Back tem assimetria por bucket de preço**: quando `BS > WS (+2% a +10%)`, CLV pre-match do Back ficou **positivo**; quando `BS < WS (-10% a -2%)`, CLV do Back ficou **fortemente negativo**.
3. **Lay em BS << WS é promissor em direção, mas ainda inconclusivo em robustez**:
   - sinais positivos de CLV lay em alguns buckets de `BS < WS`;
   - porém a amostra efetiva para a hipótese-alvo (`BS <= -5%`, pre-match) ficou praticamente em **N=1** com closing válido.
4. **Conclusão prática hoje**:
   - **Back H3B sem edge estatístico robusto no agregado**;
   - **Lay H3B ainda em fase de validação**, sem N suficiente para decisão de escala.

---

## 2) Base de dados e qualidade do corte

### 2.1 Tamanho e filtros

| Indicador | Valor |
|---|---:|
| Auditorias H3B UP com match+kickoff passado | 3993 |
| Betslip bruto | 2029 |
| Betslip confiável (`diff` entre -10% e +10%) | 901 |
| Descartados por qualidade (`diff` fora do range) | 1128 |
| Jogos únicos | 352 |
| Média de observações por jogo | 11,3 |

Leitura:
- A base ainda é grande em observações, mas com **alta correlação intra-jogo**.
- O filtro de qualidade removeu mais da metade do bruto de betslip.

### 2.2 Atualização de resultados (limitação operacional)

- O `update_results` rodou em janela limitada do plano free da API-Football.
- Na execução mais recente: **1 jogo atualizado** em 12/02; grande parte permaneceu “não encontrado”.

Implicação:
- métricas de **ROI realizado** ficam mais lentas para maturar;
- CLV segue sendo o melhor termômetro de curto prazo.

---

## 3) Diagnóstico de Back (H3B)

## 3.1 WS vs BS: diferença estrutural

### WebSocket (referência)

| Métrica | Valor |
|---|---:|
| WS reversão UP (CLV adicional) | +1,116% (N=273, NS) |
| WS reversão DOWN (CLV adicional) | -1,359% (N=282, NS) |

### Betslip (execução real)

| Métrica | Valor |
|---|---:|
| BS reversão UP (CLV adicional) | **-17,579%** (N=386, significativo negativo) |
| BS reversão DOWN (CLV adicional) | N=0 |

Leitura:
- O preço observado no WS não se transfere automaticamente para execução no BS.
- Para Back, a erosão de preço no BS segue sendo o gargalo principal.

## 3.2 Back no corte confiável (-10% a +10%)

### Pre-match

| Métrica | Resultado |
|---|---:|
| CLV bruto BS | -0,938% (N=87, sig. negativo) |
| CLV adicional BS | -1,724% (N=87, sig. negativo) |
| ROI BS | -2,877% (N=379, não significativo) |

### In-match

| Métrica | Resultado |
|---|---:|
| CLV BS | -1,443% (N=42, não significativo) |
| ROI BS | +0,409% (N=405, não significativo) |

Leitura:
- Em Back, o quadro mais robusto é o **CLV negativo pre-match** no BS.
- ROI não fechou significância estatística.

## 3.3 Back por bucket de diff BS vs WS

| Bucket | CLV Back pre-match | ROI Back (todos) | Leitura |
|---|---:|---:|---|
| BS < WS (-10% a -2%) | **-4,639%** (N=21, sig. negativo) | +3,852% (N=159, NS) | preço de entrada ruim para Back |
| BS ~ WS (-2% a +2%) | -0,300% (N=57, NS) | -4,185% (N=488, NS) | zona neutra/inconclusiva |
| BS > WS (+2% a +10%) | **+3,661%** (N=9, sig. positivo) | +3,464% (N=140, NS) | melhor região para Back, mas N pequeno em CLV |

---

## 4) Diagnóstico de Lay (hipótese: BS << WS)

## 4.1 Cobertura de Lay no dataset atual

Saída de `05_lay_bucket_analysis.txt`:

| Regime | Bucket | N total | N com lay |
|---|---|---:|---:|
| PRE_MATCH | A <= -10% | 16 | 1 |
| PRE_MATCH | B (-10,-5] | 27 | 1 |
| PRE_MATCH | C (-5,-2] | 58 | 4 |
| PRE_MATCH | D (-2,+2) | 700 | 63 |
| PRE_MATCH | E >= +2% | 200 | 92 |
| IN_MATCH | A..E | 580 | 126 |

Leitura:
- Cobertura de lay ainda heterogênea por bucket.
- Na região de interesse (`BS << WS`), os Ns de lay seguem pequenos.

## 4.2 CLV de Lay por bucket (dados disponíveis)

| Regime | Bucket | CLV Lay médio |
|---|---|---:|
| PRE_MATCH | A <= -10% | +18,069% |
| PRE_MATCH | C (-5,-2] | +5,079% |
| PRE_MATCH | D (-2,+2) | +2,789% |
| PRE_MATCH | E >= +2% | -5,918% |

Leitura direcional:
- Onde o Back sofre mais (`BS < WS`), o sinal de Lay tende a melhorar (sinal de inversão esperado).
- Onde Back tende a ficar melhor (`BS > WS`), Lay fica pior (coerente com simetria econômica).

## 4.3 Hipótese-alvo formal: PRE + BS <= -5%

Saída de `06_lay_target_summary.txt`:

| Métrica | Valor |
|---|---:|
| N total no recorte alvo | 43 |
| N com CLV lay calculável | 1 |
| CLV lay médio | +18,069% |

Interpretação:
- O sinal é positivo, porém **estatisticamente inconclusivo**.
- Não há base para afirmar edge robusto ainda na hipótese de Lay alvo.

---

## 5) API vs DOM no cenário atualizado (Back)

| Métrica | API (rótulo 2-4s) | DOM (15-30s) |
|---|---:|---:|
| Observações totais | 1069 | 2845 |
| Com betslip | 772 | 129 |
| CLV BS pre-match | +0,355% (N=42, NS) | **-2,144% (N=45, sig. negativo)** |
| CLV adicional BS pre-match | +0,023% (N=42, NS) | **-3,354% (N=45, sig. negativo)** |
| ROI BS | -2,446% (N=682, NS) | +6,893% (N=105, NS) |
| Diff BS vs WS | +0,185% (N=772, NS) | **-2,931% (N=129, sig. negativo)** |

Leitura:
- API permanece muito superior ao DOM em qualidade de preço de execução.
- DOM continua estruturalmente desfavorável para Back.

---

## 6) Conclusão operacional (Back + Lay)

### Back H3B (estado atual)
- **Não aprovado para escala por edge estatístico agregado**.
- Melhor sinal pontual segue em `BS > WS`, mas com N pequeno para robustez de CLV.

### Lay H3B (hipótese BS << WS)
- **Promissora em direção econômica**, mas **inconclusiva estatisticamente** hoje.
- A hipótese-alvo ainda não tem massa crítica de observações com closing válido.

### Decisão prática
1. Manter coleta contínua e registro de lay.
2. Reavaliar a hipótese de lay quando `N_clv` do recorte alvo atingir patamar mínimo (ex.: 80-120).
3. Para decisão financeira final, exigir:
   - CLV lay médio > 0 com IC90 acima de zero;
   - ROI lay (com resultados) consistente no mesmo recorte.

---

## 7) Próximo checkpoint recomendado

Rodar novamente a sessão completa quando:
- houver nova atualização de resultados (kickoffs liquidados);
- e/ou o recorte `PRE_MATCH + BS <= -5%` alcançar N substancialmente maior.

Com os dados atuais, o diagnóstico correto é:

**Back: sem edge robusto no agregado**  
**Lay: hipótese viva, mas ainda sem evidência suficiente para escalar**

