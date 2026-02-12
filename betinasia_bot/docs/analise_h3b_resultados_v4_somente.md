# Análise H3B — Somente Dados da Versão v4

**Data da extração:** 12/02/2026  
**Escopo deste documento:** exclusivamente resultados do modelo v4 (API 2-4s vs DOM 15-30s).  
**Regra editorial:** sem comparação com versões anteriores.

---

## 1) Contexto da rodada v4

| Indicador | Valor |
|---|---:|
| Auditorias H3B UP (match + kickoff passado) | 3975 |
| Betslip bruto | 2013 |
| Betslip confiável (diff -10% a +10%) | 892 |
| Descartados no filtro de qualidade | 1121 |
| Jogos únicos (geral) | 349 |
| Média de observações por jogo | 11,4 |
| Jogos únicos com betslip | 155 |

---

## 2) Base comparativa v4: API vs DOM

| Métrica | API (2-4s) | DOM (15-30s) |
|---|---:|---:|
| Total de observações | 1053 | 2843 |
| Com betslip | 763 | 129 |
| Com CLV pre-match | 35 | 45 |
| Com ROI | 680 | 105 |
| Lag médio | 11522 ms | 15170 ms |

Leitura rápida:
- API tem muito mais cobertura útil de betslip.
- DOM concentra poucos casos com forte erosão de preço.

---

## 3) CLV pre-match (núcleo da comparação v4)

### 3.1 CLV com odd Betslip (execução real)

| Métrica | API (2-4s) | DOM (15-30s) |
|---|---:|---:|
| CLV Bruto BS Pre-Match | +0,839% (NS, N=35) | -2,144% (sig. negativo, N=45) |
| CLV Adicional BS Pre-Match | +0,211% (NS, N=35) | -3,354% (sig. negativo, N=45) |
| Win rate CLV bruto | 57,1% | 25,0% |
| Win rate CLV adicional | 51,4% | 26,7% |

### 3.2 CLV WebSocket (referência interna do v4)

| Métrica | API (2-4s) | DOM (15-30s) |
|---|---:|---:|
| CLV Bruto WS Pre-Match | +1,167% (sig. positivo, N=45) | +0,112% (NS, N=981) |
| Win rate CLV WS | 62,8% | 51,2% |

Leitura:
- No v4, API preserva melhor o valor de entrada no pre-match.
- DOM fica estruturalmente negativo em CLV de execução real.

---

## 4) ROI por modelo (v4)

| Métrica | API (2-4s) | DOM (15-30s) |
|---|---:|---:|
| ROI Betslip | -2,422% (NS, N=680) | +6,893% (NS, N=105) |
| ROI WebSocket | +0,737% (NS, N=953) | -0,276% (NS, N=2263) |
| Win rate ROI Betslip | 50,0% | 60,5% |
| Win rate ROI WS | 49,2% | 49,4% |

Leitura:
- Em ROI, nenhum dos dois modelos mostrou significância estatística robusta nesta rodada.
- O diferencial v4 aparece mais em CLV/diff de execução do que em ROI consolidado.

---

## 5) Diferença de preço BS vs WS (v4)

| Métrica | API (2-4s) | DOM (15-30s) |
|---|---:|---:|
| Diff BS vs WS (média) | +0,185% (NS, N=763) | -2,931% (sig. negativo, N=129) |
| BS > WS | 38,0% (290/763) | 8,5% (11/129) |
| BS > WS +2% | 18,5% (141/763) | 7,0% (9/129) |

Leitura:
- API opera mais perto (ou acima) do preço de referência.
- DOM perde preço de forma recorrente.

---

## 6) Conclusão objetiva da versão v4

1. **API (2-4s) é superior ao DOM (15-30s)** em qualidade de execução.
2. **DOM permanece negativo em CLV de betslip pre-match** com significância estatística.
3. **API ainda não confirmou edge estatístico em CLV adicional de execução**, apesar de sinal melhor que DOM.
4. Em **ROI**, os resultados permanecem não conclusivos para decisão final.

---

## 7) Recomendação operacional (somente v4)

- Manter API como trilha principal.
- Reduzir ainda mais latência e aumentar N de CLV pre-match no API.
- Reavaliar quando houver amostra suficiente para fechar significância no CLV adicional de execução.

