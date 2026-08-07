# H3BUP_vNext — Análise robusta de resultados (até freeze 2026-08-01)

- **Status:** `NEGATIVE_INSUFFICIENT_N`
- **Run id:** `strat78c9ext01`
- **Fonte:** freeze Friendly `78c9f53d95df` (checksum classification intacto)
- **Cutoff dados:** `2026-08-01T01:31:39Z`
- **Gerado:** `2026-08-07T14:16:15.296268+00:00`
- **Sample gate:** `INSUFFICIENT_N` (N resolved=56; alvo informal ≥100 para inferência fraca, ≥300 para forte)

> Análise histórica **read-only**. Não altera policy/filtros/Telegram. **Não** inclui a era stake=2 (deploy ~2026-08-01 12:07Z) — refresh VPS pendente.

## 1) Veredicto executivo

No universo operacional exact `H3BUP_vNext_20260629` (Back Pre LIVE_OK, stake 10), de **2026-07-28 → 2026-08-01**, a estratégia está **no vermelho**: P&L resolved **-55.42** USD sobre stake resolved **560** → ROI **-9.9%**.

Pontos-chave:
1. **N pequeno** (74 LIVE_OK / 52 settled) — qualquer conclusão é preliminar.
2. Bootstrap ROI (cluster evento) IC95 **[-30.9% , 9.3%]** — inclui zero; P(ROI>0)≈**14.3%**.
3. **Friendly** ROI -3.3% vs **Non-Friendly** -23.8% — NF concentra a perda, mas coberturas de settlement são assimétricas (NF muitos open no cutoff).
4. Winrate settled **48.1%** vs breakeven aproximado **52.6%** (odd média 1.901) → gap ≈ **-4.5 pp**.
5. Drawdown máximo no equity settled: **-80.12** USD; maior losing streak: **4**.

## 2) Universo e janela

| Campo | Valor |
|---|---|
| Policy | `H3BUP_vNext_20260629` |
| Regime | Back / Pre / LIVE_OK |
| Stake era | 10 USD (100% das 74 ordens) |
| Início observado | 2026-07-28 (pós capacity-fix) |
| Fim | 2026-08-01 01:19Z (última ordem no freeze) |
| LIVE_OK | 74 |
| Eventos únicos | 58 |
| Settled / Void / Open | 52 / 4 / 18 |

Nota: a policy existe desde 2026-06-29, mas o primeiro LIVE_OK exact deste universo no freeze é **2026-07-28** (pós capacity-fix; legado stake 20 excluído).

## 3) Performance agregada

| Métrica | Total | Friendly | Non-Friendly |
|---|---:|---:|---:|
| LIVE_OK | 74 | 39 | 35 |
| Eventos | 58 | 28 | 30 |
| Settled | 52 | 35 | 17 |
| Open (no cutoff) | 18 | 1 | 17 |
| Stake placed | 740 | 390 | 350 |
| Stake resolved | 560 | 380 | 180 |
| P&L resolved | -55.42 | -12.54 | -42.88 |
| ROI resolved | -9.9% | -3.3% | -23.8% |
| Winrate settled | 48.1% | 51.4% | 41.2% |
| Avg win / avg loss | +8.29 / -9.73 | +8.35 / -9.58 | +8.13 / -9.98 |
| Coverage accounting | 75.7% | 97.4% | 51.4% |

### Inferência (bootstrap)

| Método | Média ROI | IC90 | IC95 | P(ROI>0) |
|---|---:|---|---|---:|
| Order resample | -10.5% | [-31.3%, 10.4%] | [-35.5%, 13.9%] | 20.5% |
| Event-cluster | -10.8% | [-27.5%, 6.2%] | [-30.9%, 9.3%] | 14.3% |

Interpretação: a perda pontual é material (~−10% ROI), mas **não é estatisticamente distinguível de zero** com este N (IC cruza zero).

## 4) Evolução diária

| Dia UTC | N | F/NF | Settled | Open | P&L | ROI |
|---|---:|---:|---:|---:|---:|---:|
| 2026-07-28 | 3 | 0/3 | 2 | 0 | -0.43 | -1.4% |
| 2026-07-29 | 24 | 17/7 | 22 | 0 | -3.37 | -1.4% |
| 2026-07-30 | 14 | 7/7 | 13 | 1 | -22.92 | -17.6% |
| 2026-07-31 | 31 | 15/16 | 15 | 15 | -28.70 | -17.9% |
| 2026-08-01 | 2 | 0/2 | 0 | 2 | 0.00 | — |

Padrão: 28–29 Jul quase flat; **30–31 Jul** concentram a perda (−22.9 / −28.7). Em 31/Jul há 15 open no cutoff — ROI do dia subestima o resultado final daquele cohort.

## 5) Friendly vs Non-Friendly (diagnóstico, não filtro)

- Friendly = 100% `Club Friendly` no freeze (N=39).
- Non-Friendly espalhado: UCL, Europa League, MLS, Copa, etc.
- Delta ROI F−NF ≈ **+20.5 pp** (freeze), mas:
  - permutation p≈0.27 (não significativo);
  - NF com coverage 51% vs Friendly 97% → comparação **enviesada** no cutoff;
  - status freeze: `NON_FRIENDLY_WORSE_PRELIMINARY`.
- Slippage mediano mais favorável em Friendly (−3.12% vs −0.64%) — diferença robusta no freeze (IC95 não inclui 0), mas **não traduz** automaticamente em melhor P&L (ver buckets).

## 6) Decomposição por drivers

### 6.1 Ligas (só settled)

| Liga | Classe | N | P&L | ROI |
|---|---|---:|---:|---:|
| UEFA Europa League | NON_FRIENDLY | 4 | -25.76 | -64.4% |
| CONMEBOL Copa Sudamericana | NON_FRIENDLY | 2 | -19.99 | -100.0% |
| UEFA Champions League | NON_FRIENDLY | 3 | -14.29 | -47.6% |
| Club Friendly | FRIENDLY | 35 | -12.54 | -3.6% |
| Scotland Premier League | NON_FRIENDLY | 2 | -0.62 | -3.1% |
| Paraguay Division 1 (Primera División) | NON_FRIENDLY | 2 | -0.30 | -1.5% |
| Colombia Cup | NON_FRIENDLY | 4 | +18.08 | 45.2% |

Maiores detratores settled: **Europa League (−25.8)**, **Sudamericana (−20.0)**, **UCL (−14.3)**, **Club Friendly (−12.5)**.
Único bloco claramente positivo no freeze settled: **Colombia Cup (+18.1, N=4)** — amostra minúscula.

### 6.2 Bookmakers (settled)

| BM | N | P&L | ROI |
|---|---:|---:|---:|
| pin88 | 27 | -29.08 | -10.8% |
| pmk | 10 | -26.98 | -27.0% |
| bf | 4 | -20.23 | -50.6% |
| 4casters | 2 | -19.99 | -100.0% |
| vx | 2 | -0.60 | -3.0% |
| sharp | 1 | +7.62 | 76.2% |
| sing2 | 3 | +8.13 | 27.1% |
| punter_io | 1 | +9.54 | 95.4% |
| sbo | 2 | +16.17 | 80.9% |

Concentração: `pin88` e `pmk` dominam volume e perda. Remover bookmaker dominante no freeze ainda deixa ROI negativo (robustez).

### 6.3 Slippage pré-submit

| Grupo | Bucket | N | P&L | ROI |
|---|---|---:|---:|---:|
| ALL | < -3% | 19 | -41.05 | -21.6% |
| ALL | [-1%, 0%) | 14 | -25.56 | -18.3% |
| ALL | [-3%, -1%) | 19 | +11.19 | 5.9% |
| FRIENDLY | < -3% | 19 | -41.05 | -21.6% |
| FRIENDLY | [-1%, 0%) | 5 | -9.60 | -19.2% |
| FRIENDLY | [-3%, -1%) | 11 | +38.11 | 34.6% |
| NON_FRIENDLY | [-1%, 0%) | 9 | -15.96 | -17.7% |
| NON_FRIENDLY | [-3%, -1%) | 8 | -26.92 | -33.7% |

Achado exploratório: em Friendly, o bucket **mais negativo** (`< -3%`) teve ROI **−21.6%**, enquanto `[−3%,−1%)` foi **+34.7%**. Isto **não** valida “mais slippage favorável = melhor”; sugere ruído / seleção / N baixo. Não usar como regra operacional sem validação OOS.

### 6.4 Odds e linhas

| Odd bin | N | WR | P&L | ROI |
|---|---:|---:|---:|---:|
| 1.85-1.95 | 24 | 54.2% | +4.63 | 1.9% |
| 1.95-2.05 | 10 | 30.0% | -39.58 | -39.6% |
| 2.05-2.15 | 5 | 60.0% | +6.83 | 13.7% |

| |line| bin | N | P&L | ROI |
|---|---:|---:|---:|
| |line|<=1 | 21 | -4.89 | -2.3% |
| 1<|line|<=2 | 14 | -46.82 | -33.4% |
| 2<|line|<=3 | 5 | +7.20 | 14.4% |
| |line|>3 | 12 | -10.91 | -9.1% |

## 7) CLV (qualidade de preço)

Cobertura CLV VALID_STRICT é baixa (Friendly POST_5M ~13%; NF ~46%). Médias CLV são **negativas** na maioria das janelas — consistente com ausência de edge de preço claro, mas N CLV é `VERY_LOW_N`.

| Grupo | Janela | N | Coverage | Média | Mediana | % pos | Status |
|---|---|---:|---:|---:|---:|---:|---|
| FRIENDLY | POST_5M | 5 | 39 | -1.786211353318805 | -1.3705583756345119 | 20.0 | VERY_LOW_N |
| FRIENDLY | POST_15M | 5 | 39 | -3.47169173171594 | -2.801400700350176 | 20.0 | VERY_LOW_N |
| FRIENDLY | CLOSING | 6 | 39 | -3.5568174688262846 | -2.700387843989599 | 33.333333333333336 | VERY_LOW_N |
| NON_FRIENDLY | POST_5M | 16 | 35 | -0.15408912401768254 | -0.43665546916283327 | 25.0 | VERY_LOW_N |
| NON_FRIENDLY | POST_15M | 14 | 35 | -0.029328953280353755 | -0.3509100290359446 | 28.571428571428573 | VERY_LOW_N |
| NON_FRIENDLY | CLOSING | 6 | 35 | -2.0899806315867187 | -1.6767700737800428 | 16.666666666666668 | VERY_LOW_N |

## 8) Robustez (do freeze oficial)

Em **13/13** cenários de remoção (top gains/losses/ligas/bookmaker/só CLV closing), o sinal `sign_vs_base` manteve-se: Friendly melhor que NF em ROI, e total continua negativo na baseline e na maioria dos cortes agressivos.

| Cenário | N | ROI total | ROI F | ROI NF |
|---|---:|---:|---:|---:|
| baseline | 74 | -9.9% | -3.3% | -23.8% |
| remove_top1_gain | 73 | -12.0% | -6.3% | -23.8% |
| remove_top3_gains | 71 | -16.4% | -12.6% | -23.8% |
| remove_top5_gains | 69 | -20.8% | -15.8% | -30.9% |
| remove_top1_loss | 73 | -8.3% | -0.7% | -23.8% |
| remove_top3_losses | 71 | -4.8% | 2.1% | -19.3% |
| remove_top1_league | 69 | -5.7% | -3.3% | -12.2% |
| remove_top3_leagues | 63 | -6.0% | -3.3% | -19.0% |
| remove_dominant_bookmaker | 32 | -9.8% | -0.4% | -25.6% |
| only_accounting_reconciled | 56 | -9.9% | -3.3% | -23.8% |
| only_clv_closing_valid_strict | 12 | -31.7% | -20.0% | -43.5% |
| only_pre_submit_available | 74 | -9.9% | -3.3% | -23.8% |
| only_valid_event_id | 74 | -9.9% | -3.3% | -23.8% |

## 9) Riscos de leitura errada

1. **Open assimétrico:** 17/35 NF open vs 1/39 Friendly no cutoff → ROI NF pode melhorar ou piorar quando liquidar.
2. **N insuficiente** para aprovar/rejeitar edge.
3. **CLV sparse** — não usar como prova de mispricing.
4. **Buckets** (slip/odd/line) são exploratórios; múltiplos testes sem correção.
5. **Era stake=2** (após 2026-08-01 12:07Z) **fora deste relatório** — sizing mudou; precisa cohort separado.
6. Não misturar com legado `bridge_h3b_live_v0` / stake 20.

## 10) Implicações (diagnóstico, não ordem de mudança)

Com a evidência atual:
- A estratégia **não demonstra edge positivo** no período pós capacity-fix; o ponto estimado é negativo (~−10% ROI).
- A perda **não está isolada** num único evento (sobrevive remoção top losses), mas **há concentração** em poucas ligas oficiais.
- Friendly é menos negativo que NF no settled disponível, porém **sem significância** e com coverage desigual.
- Prioridade analítica seguinte: **refresh VPS até 2026-08-07**, liquidar opens do cutoff, separar cohort **stake10 vs stake2**, e recalcular IC event-clustered.

## 11) Próximo refresh

Bloqueado nesta sessão por SSH (`Permission denied`). Chave pública a instalar na VPS:
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFH5MuVrvyEQOKlYnbD/GDgVKENHsK+84mN8x8QNOfPt cursor-agent-20260807-h3bup-results
```
Depois: rerun `python -m ops.h3bup_friendly_analysis` + este extensório strategy_results.

---

Artefactos: `logs/h3bup_strategy_results/20260807/strat78c9ext01/`
Freeze base: `logs/h3bup_friendly_analysis/20260801/78c9f53d95df/`
