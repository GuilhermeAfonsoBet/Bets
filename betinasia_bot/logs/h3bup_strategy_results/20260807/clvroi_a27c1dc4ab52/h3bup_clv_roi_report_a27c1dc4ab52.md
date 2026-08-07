# H3BUP_vNext — Análise CLV + ROI (até 2026-08-07)

- **Status:** `CLV_NEGATIVE_ROI_SLIGHTLY_NEGATIVE_INSUFFICIENT_N`
- **Freeze:** `a27c1dc4ab52` · cutoff `2026-08-07T14:25:06Z`
- **Universo:** LIVE_OK Back Pre `H3BUP_vNext_20260629` · N=208 (F=109, NF=99)
- **Eras:** stake10=86 · stake2=122
- **Friendly pack status:** `NO_CLEAR_FRIENDLY_DIFFERENCE`

> Read-only. CLV oficial = **VALID_STRICT** apenas. ROI = P&L settled / stake resolved (void no denominador).

## 1) Veredicto dual (CLV e ROI)

| Métrica | All | stake10 | stake2 | Friendly | Non-Friendly |
|---|---:|---:|---:|---:|---:|
| LIVE_OK | 208 | 86 | 122 | 109 | 99 |
| Settled / Open | 158/22 | 80/0 | 78/22 | 90/5 | 68/17 |
| Stake placed | 1104 | 860 | 244 | 602 | 502 |
| P&L resolved | -30.22 | -31.49 | +1.27 | -15.05 | -15.17 |
| **ROI resolved** | **-2.9%** | -3.7% | 0.7% | -2.6% | -3.3% |
| WR settled | 52.5% | 51.2% | 53.8% | 51.1% | 54.4% |
| CLV POST_5M mean | -1.35% | -0.56% | -1.77% | -2.70% | -0.76% |
| CLV POST_5M cov | 31.7% | 26.7% | 35.2% | 18.3% | 46.5% |
| CLV CLOSING mean | -2.32% | -1.54% | -2.81% | -2.92% | -1.97% |
| CLV CLOSING %pos | 30.6% | 37.5% | 26.3% | 21.7% | 35.9% |

**Leitura:**
1. **ROI** agregado **-2.9%** (P&L -30.22 / stake resolved 1040) — negativo, mas **melhor** que o freeze 01/Ago (−9.9%).
2. **CLV** é consistentemente **negativo** em POST_5M / POST_15M / CLOSING (médias < 0; % positivo baixo). Sinal mais estável: **sem evidência de edge de preço**.
3. Era **stake2** (ROI 0.7%) vs **stake10** (ROI -3.7%): stake10 continua pior; stake2 perto de flat/ligeiramente negativo.
4. Friendly vs NF em ROI quase iguais (-2.6% vs -3.3%). Em CLV, Friendly está **pior** (POST_5M -2.70% vs -0.76%), com coverage mais baixa.

## 2) Inferência (bootstrap)

### ROI (event-cluster, settled)

| Cohort | Média | IC95 | P(ROI>0) |
|---|---:|---|---:|
| ALL | -3.0% | [-17.4%, 11.4%] | 33.0% |
| stake10 | -3.7% | [-20.6%, 13.4%] | 32.9% |
| stake2 | 0.9% | [-17.2%, 19.4%] | 53.5% |
| FRIENDLY | -2.7% | [-20.7%, 16.0%] | 37.7% |
| NON_FRIENDLY | -3.7% | [-25.3%, 18.3%] | 37.0% |

### CLV mean VALID_STRICT

| Cohort | Janela | N | Média | IC95 | P(CLV>0) |
|---|---|---:|---:|---|---:|
| ALL | POST_5M | 66 | -1.35% | [-1.91%, -0.78%] | 0.0% |
| ALL | POST_15M | 60 | -1.60% | [-2.32%, -0.91%] | 0.0% |
| ALL | CLOSING | 62 | -2.32% | [-3.54%, -1.13%] | 0.0% |
| stake10 | POST_5M | 23 | -0.54% | [-1.65%, 0.56%] | 16.6% |
| stake10 | POST_15M | 22 | -0.89% | [-2.41%, 0.43%] | 9.6% |
| stake10 | CLOSING | 24 | -1.54% | [-3.49%, 0.34%] | 5.4% |
| stake2 | POST_5M | 43 | -1.76% | [-2.37%, -1.19%] | 0.0% |
| stake2 | POST_15M | 38 | -2.00% | [-2.77%, -1.30%] | 0.0% |
| stake2 | CLOSING | 38 | -2.79% | [-4.40%, -1.21%] | 0.0% |
| FRIENDLY | POST_5M | 20 | -2.69% | [-3.92%, -1.56%] | 0.0% |
| FRIENDLY | POST_15M | 18 | -3.31% | [-5.05%, -1.63%] | 0.0% |
| FRIENDLY | CLOSING | 23 | -2.89% | [-4.81%, -1.04%] | 0.0% |
| NON_FRIENDLY | POST_5M | 46 | -0.77% | [-1.27%, -0.21%] | 0.5% |
| NON_FRIENDLY | POST_15M | 42 | -0.86% | [-1.41%, -0.24%] | 0.5% |
| NON_FRIENDLY | CLOSING | 39 | -1.97% | [-3.56%, -0.41%] | 0.7% |

Para CLV All com N≥~40, os IC95 ficam **maioritariamente abaixo de zero** — CLV negativo é mais crível que o ROI negativo (cujo IC ainda cruza zero em vários cohorts).

## 3) CLV × ROI realizado (joint)

Só ordens **settled** com CLV VALID_STRICT.

### POST_5M (N=46, corr=-0.063, sign-concordance=50.0%)

| Bucket CLV | N | CLV médio | ROI realizado | WR |
|---|---:|---:|---:|---:|
| CLV < -3% | 9 | -5.18% | 46.4% | 55.6% |
| [-3%, -1%) | 12 | -1.74% | -55.0% | 50.0% |
| [-1%, 0%) | 17 | -0.40% | -18.2% | 47.1% |
| [0%, +1%) | 5 | 0.34% | -8.4% | 40.0% |
| CLV ≥ +1% | 3 | 4.27% | 31.1% | 66.7% |

### POST_15M (N=41, corr=-0.084, sign-concordance=48.8%)

| Bucket CLV | N | CLV médio | ROI realizado | WR |
|---|---:|---:|---:|---:|
| CLV < -3% | 8 | -6.29% | 16.6% | 50.0% |
| [-3%, -1%) | 12 | -1.82% | -31.1% | 66.7% |
| [-1%, 0%) | 13 | -0.49% | -24.1% | 38.5% |
| [0%, +1%) | 4 | 0.33% | -13.7% | 25.0% |
| CLV ≥ +1% | 4 | 3.48% | 34.6% | 75.0% |

### CLOSING (N=51, corr=0.221, sign-concordance=66.7%)

| Bucket CLV | N | CLV médio | ROI realizado | WR |
|---|---:|---:|---:|---:|
| CLV < -3% | 21 | -6.77% | -21.4% | 47.6% |
| [-3%, -1%) | 11 | -1.82% | -84.8% | 27.3% |
| [-1%, 0%) | 3 | -0.56% | -70.3% | 33.3% |
| [0%, +1%) | 6 | 0.40% | 88.9% | 100.0% |
| CLV ≥ +1% | 10 | 4.47% | 72.0% | 80.0% |

Correlação CLV↔ROI é fraca/mista (N CLV limitado). A **massa de CLV < 0** continua o warning principal de mispricing.

## 4) Cobertura CLV

| Cohort | POST_5M | POST_15M | CLOSING |
|---|---:|---:|---:|
| ALL | 31.7% (n=66) | 28.8% (n=60) | 29.8% (n=62) |
| FRIENDLY | 18.3% (n=20) | 16.5% (n=18) | 21.1% (n=23) |
| NON_FRIENDLY | 46.5% (n=46) | 42.4% (n=42) | 39.4% (n=39) |
| stake10 | 26.7% (n=23) | 25.6% (n=22) | 27.9% (n=24) |
| stake2 | 35.2% (n=43) | 31.1% (n=38) | 31.1% (n=38) |

Friendly tem coverage ~2× menor que Non-Friendly em POST_5M — comparar médias F vs NF com cautela.

## 5) Evolução diária (ROI + CLV)

| Dia | N | F/NF | Era | P&L | ROI | CLV5 mean (n) | CLV Close mean (n) |
|---|---:|---:|---|---:|---:|---:|---:|
| 2026-07-28 | 3 | 0/3 | stake10:3 | -0.43 | -1.4% | — (0) | — (0) |
| 2026-07-29 | 24 | 17/7 | stake10:24 | -3.37 | -1.4% | -0.12% (2) | 1.44% (2) |
| 2026-07-30 | 14 | 7/7 | stake10:14 | -12.02 | -8.6% | -2.44% (5) | -3.78% (7) |
| 2026-07-31 | 31 | 15/16 | stake10:31 | +4.76 | 1.5% | 0.12% (13) | -1.02% (11) |
| 2026-08-01 | 45 | 24/21 | stake10:14,stake2:31 | -29.69 | -15.5% | 0.06% (6) | -2.20% (8) |
| 2026-08-02 | 21 | 12/9 | stake2:21 | +0.06 | 0.1% | -2.07% (5) | -1.64% (5) |
| 2026-08-03 | 10 | 1/9 | stake2:10 | +5.09 | 84.8% | -0.87% (4) | -6.49% (2) |
| 2026-08-04 | 19 | 8/11 | stake2:19 | +4.77 | 14.9% | -1.42% (8) | -2.23% (9) |
| 2026-08-05 | 11 | 8/3 | stake2:11 | -4.76 | -21.6% | -2.70% (7) | -4.20% (5) |
| 2026-08-06 | 17 | 10/7 | stake2:17 | +7.36 | 30.7% | -1.89% (8) | -2.36% (11) |
| 2026-08-07 | 13 | 7/6 | stake2:13 | -1.99 | -99.5% | -2.38% (8) | -1.61% (2) |

## 6) Cohorts cruzados stake × Friendly

| Cohort | N | ROI | CLV5 mean | CLV Close mean |
|---|---:|---:|---:|---:|
| stake10_FRIENDLY | 48 | -1.7% | -1.72% | -3.35% |
| stake10_NON_FRIENDLY | 38 | -6.1% | -0.15% | -0.64% |
| stake2_FRIENDLY | 61 | -6.8% | -3.11% | -2.69% |
| stake2_NON_FRIENDLY | 61 | 9.7% | -1.12% | -2.89% |

## 7) Conclusões (diagnóstico)

1. **CLV primeiro:** mercado move-se contra a posição após o fill. Sem CLV positivo sustentado, ROI positivo seria ruído.
2. **ROI melhorou vs 01/Ago** (−9.9% → −2.9%) com maturidade + stake2; **não** prova edge.
3. **Stake2** mitiga $ PnL, mas CLV também negativo — não resolve mispricing.
4. **Friendly filter** não se justifica: ROI F≈NF; CLV Friendly pior (coverage menor).
5. Prioridade: subir coverage CLV (esp. Friendly/POST_5M) e reavaliar com N VALID_STRICT ≥100/janela.

## 8) Artefactos

- Freeze: `logs/h3bup_friendly_analysis/20260807/a27c1dc4ab52/`
- Bundle CLV+ROI: `logs/h3bup_strategy_results/20260807/clvroi_a27c1dc4ab52/`

