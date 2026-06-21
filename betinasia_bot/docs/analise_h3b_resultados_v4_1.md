# Analise H3B - Resultados v4.1 (Atualizacao)

**Data da analise:** 2026-02-12 12:09 UTC  
**Versao:** 4.1  
**Escopo:** CLV + ROI (WebSocket e Betslip), com recortes por pre/in-match, diff BS vs WS, faixa AH, faixa de lag e modelo de execucao.

---

## 1. Resumo Executivo

Base consolidada carregada:
- **3975** auditorias H3B UP com match mapeado e kickoff passado.
- Betslip bruto: **2013**
- Betslip confiavel (diff entre -10% e +10%): **892** (descartados: 1121)
- Jogos unicos: **349** (media **11.4** obs/jogo)
- Jogos unicos com betslip: **155**

Principais achados:

1. **CLV adicional segue negativo no pre-match**:
   - WS pre-match: **-0.356%** (significativo negativo, p<0.10)
   - BS pre-match: **-1.794%** (significativo negativo, p<0.10)

2. **CLV bruto BS pre-match tambem negativo**:
   - BS pre-match: **-0.839%** (significativo negativo, p<0.10)

3. **ROI sem significancia estatistica no betslip**:
   - ROI BS pre-match: **-2.877%** (NS)
   - ROI BS in-match: **+0.464%** (NS)

4. **WebSocket como referencia de ROI**:
   - ROI WS pre-match: **-4.140%** (significativo negativo, p<0.10)
   - ROI WS in-match: **+2.882%** (NS)

5. **API vs DOM (recorte por modelo)**:
   - API (2-4s): CLV BS pre-match **+0.839%** (NS), CLV adicional BS **+0.211%** (NS)
   - DOM (15-30s): CLV BS pre-match **-2.144%** (significativo negativo), CLV adicional BS **-3.354%** (significativo negativo)
   - Diff BS vs WS: API **+0.185%** (NS) vs DOM **-2.931%** (significativo negativo)

Conclusao atual:
- Nao ha evidencia estatistica robusta de edge executavel do H3B UP no agregado.
- O recorte DOM permanece estruturalmente negativo em CLV.
- O recorte API continua mais promissor que DOM, mas ainda sem significancia em CLV adicional BS.

---

## 2. Dados e Amostras

### 2.1 Qualidade e cobertura

| Metrica | Valor |
|---|---:|
| Total match+kickoff | 3975 |
| Betslip bruto | 2013 |
| Betslip confiavel (-10% a +10%) | 892 |
| Descartados no filtro de diff | 1121 |
| Jogos unicos | 349 |
| Observacoes por jogo (media) | 11.4 |
| Jogos unicos (betslip) | 155 |

### 2.2 Cobertura por tipo de metrica

| Metrica | N |
|---|---:|
| CLV WS bruto | 1656 |
| CLV WS adicional | 1657 |
| CLV BS bruto | 126 |
| CLV BS adicional | 125 |
| ROI WS | 3270 |
| ROI BS | 785 |
| Pre-match | 1568 |
| In-match | 1652 |

---

## 3. Revalidacao CLV WebSocket

### 3.1 CLV bruto - WS

| Recorte | N | Media | Mediana | IC 90% | Win rate | Status |
|---|---:|---:|---:|---|---:|---|
| Pre-match | 1051 | +0.172% | +0.000% | [+0.034%, +0.311%] | 51.7% | Significativo positivo |
| In-match | 231 | +1.157% | +0.809% | [+0.129%, +2.185%] | 56.6% | Significativo positivo |

### 3.2 CLV adicional (baseline v6) - WS

| Recorte | N | Media | Mediana | IC 90% | Win rate | Status |
|---|---:|---:|---:|---|---:|---|
| Pre-match | 1051 | -0.356% | -0.254% | [-0.514%, -0.197%] | 44.6% | Significativo negativo |
| In-match | 228 | -0.098% | +0.712% | [-1.396%, +1.199%] | 55.7% | Nao significativo |

---

## 4. CLV e ROI com odd Betslip

### 4.1 Pre-match

| Metrica | N | Media | Mediana | IC 90% | Win rate | Status |
|---|---:|---:|---:|---|---:|---|
| CLV Bruto BS | 80 | -0.839% | -0.825% | [-1.618%, -0.060%] | 39.2% | Significativo negativo |
| CLV Adicional BS | 80 | -1.794% | -1.374% | [-2.768%, -0.820%] | 37.5% | Significativo negativo |
| CLV Adicional WS (ref.) | 1051 | -0.356% | -0.254% | [-0.514%, -0.197%] | 44.6% | Significativo negativo |
| ROI BS | 379 | -2.877% | +0.000% | [-10.311%, +4.558%] | 49.5% | Nao significativo |
| ROI WS (ref.) | 1430 | -4.140% | +0.000% | [-8.125%, -0.155%] | 48.3% | Significativo negativo |

### 4.2 In-match

| Metrica | N | Media | Mediana | IC 90% | Win rate | Status |
|---|---:|---:|---:|---|---:|---|
| ROI BS | 403 | +0.464% | +0.000% | [-6.782%, +7.710%] | 51.9% | Nao significativo |
| ROI WS (ref.) | 1250 | +2.882% | +0.000% | [-3.122%, +8.887%] | 48.9% | Nao significativo |

Obs.: CLV in-match foi calculado no output tecnico, mas para decisao operacional priorizamos ROI no in-match.

---

## 5. Analise por diff Betslip vs WebSocket

| Bucket diff (BS vs WS) | N bucket | CLV BS PM (N) | CLV BS PM media | ROI BS todos (N) | ROI BS media |
|---|---:|---|---:|---|---:|
| BS < WS (-10% a -2%) | 186 | 20 | -4.748% (sig neg) | 159 | +3.852% (NS) |
| BS ~ WS (-2% a +2%) | 556 | 51 | -0.100% (NS) | 487 | -4.355% (NS) |
| BS > WS (+2% a +10%) | 150 | 9 | +3.661% (sig pos) | 139 | +4.208% (NS) |

Leitura:
- O unico CLV BS pre-match claramente positivo esta no bucket **BS > WS**.
- Para ROI, nenhum bucket mostrou significancia nesta amostra.

---

## 6. Pre-match vs In-match (Betslip)

| Recorte | N diff | Diff BS vs WS media | Diff status | N CLV BS | CLV BS media | CLV status | N ROI BS | ROI BS media | ROI status |
|---|---:|---:|---|---:|---:|---|---:|---:|---|
| Pre-match | 436 | -0.021% | NS | 80 | -0.839% | Sig. neg | 379 | -2.877% | NS |
| In-match | 451 | -0.466% | Sig. neg | 42 | -1.443% | NS | 403 | +0.464% | NS |

---

## 7. Top ligas por volume (ROI)

| Liga | N ROI | ROI media | Win rate | Status |
|---|---:|---:|---:|---|
| England National League | 100 | -3.754% | 49.4% | NS |
| England Premier League | 103 | -0.450% | 50.6% | NS |
| England National League North | 82 | -8.452% | 47.1% | NS |
| England National League South | 76 | -3.880% | 49.3% | NS |
| AFC Asian Champions League | 73 | +4.197% | 52.4% | NS |
| England Football League Championship | 62 | +5.526% | 54.0% | NS |
| Scotland Premier League | 37 | -11.257% | 44.1% | NS |
| Club Friendly | 19 | -23.021% | 41.2% | NS |
| England League 1 | 26 | -3.931% | 50.0% | NS |
| England League 2 | 34 | +4.894% | 55.2% | NS |

---

## 8. Por faixa de linha AH

| Faixa AH | N diff | Diff media | Diff status | N CLV BS PM | CLV BS PM media | CLV status | N ROI BS | ROI BS media | ROI status |
|---|---:|---:|---|---:|---:|---|---:|---:|---|
| 0-1 (liquida) | 379 | -0.942% | Sig. neg | 46 | -2.551% | Sig. neg | 337 | -1.349% | NS |
| 1-2 (media) | 124 | +0.738% | Sig. pos | 8 | +0.882% | NS | 103 | +8.983% | NS |
| 2+ (extrema) | 389 | +0.074% | NS | 26 | +1.661% | Sig. pos | 345 | -4.041% | NS |

---

## 9. Por faixa de lag

| Lag | N diff | Diff media | Diff status | N CLV BS PM | CLV BS PM media | CLV status | N ROI BS | ROI BS media | ROI status |
|---|---:|---:|---|---:|---:|---|---:|---:|---|
| < 10s | 641 | -0.146% | NS | 33 | +0.296% | NS | 560 | -1.938% | NS |
| 10-20s | 107 | -2.393% | Sig. neg | 39 | -2.237% | Sig. neg | 87 | +5.202% | NS |
| 20-30s | 75 | +0.150% | NS | 4 | +2.878% | NS | 70 | -14.629% | NS |
| > 30s | 69 | +1.477% | Sig. pos | 4 | -0.286% | NS | 68 | +10.782% | NS |

---

## 10. Por modelo de execucao: API vs DOM

### 10.1 API (2-4s)

| Metrica | Valor |
|---|---|
| Total observacoes | 1053 |
| Com betslip | 763 |
| Com CLV pre-match | 35 |
| Com ROI | 680 |
| Lag medio | 11522 ms |
| CLV Bruto BS PM | +0.839% (NS, N=35) |
| CLV Adicional BS PM | +0.211% (NS, N=35) |
| CLV Bruto WS PM | +1.167% (sig pos, N=45) |
| ROI Betslip | -2.422% (NS, N=680) |
| ROI WS | +0.737% (NS, N=953) |
| Diff BS vs WS | +0.185% (NS, N=763) |
| BS > WS | 290/763 (38.0%) |
| BS > WS +2% | 141/763 (18.5%) |

### 10.2 DOM (15-30s)

| Metrica | Valor |
|---|---|
| Total observacoes | 2843 |
| Com betslip | 129 |
| Com CLV pre-match | 45 |
| Com ROI | 105 |
| Lag medio | 15170 ms |
| CLV Bruto BS PM | -2.144% (sig neg, N=45) |
| CLV Adicional BS PM | -3.354% (sig neg, N=45) |
| CLV Bruto WS PM | +0.112% (NS, N=981) |
| ROI Betslip | +6.893% (NS, N=105) |
| ROI WS | -0.276% (NS, N=2263) |
| Diff BS vs WS | -2.931% (sig neg, N=129) |
| BS > WS | 11/129 (8.5%) |
| BS > WS +2% | 9/129 (7.0%) |

Leitura:
- O gap API vs DOM permanece grande em CLV e diff.
- Com os dados atuais, API ainda precisa de mais N para confirmar CLV adicional positivo.

---

## 11. Diagnostico de qualidade

| Metrica | Valor |
|---|---|
| Betslip odds | min=1.029 med=1.939 max=23.775 |
| WebSocket odds | min=1.068 med=1.943 max=24.725 |
| Diff percentual | min=-10.0% med=-0.2% max=+10.0% |
| Diffs < -30% | 0 (0.0%) |
| Diffs > +30% | 0 (0.0%) |
| Jogos com resultado (gols) | 3270 / 3975 |

---

## 12. Conclusao Final

1. O H3B UP, no agregado atual, **nao mostrou edge robusto** em CLV adicional no pre-match (WS e BS negativos).
2. O recorte API e superior ao DOM em qualidade e degradacao de preco, mas ainda sem confirmacao estatistica em CLV adicional BS.
3. O ROI segue ruidoso e nao significativo na maior parte dos recortes, com mediana frequentemente em 0%.
4. Proximo passo objetivo: ampliar N de CLV BS pre-match no modelo API e reavaliar com erro clusterizado por jogo.

