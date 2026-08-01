# Relatório — Friendly vs Non-Friendly (H3BUP_vNext)

Data: 2026-08-01  
Run id: `ae996053a99a`  
Cutoff: `2026-08-01T01:22:38Z`  
Classificação: `FRIENDLY_CLASS_V1_20260731`  
Checksum: `1a9706023c55df4244dd2d15a1593137aae073339d2e612728c85972c2df0d8e`  
Status final: **`NON_FRIENDLY_WORSE_PRELIMINARY`**

> Análise histórica read-only. Não altera policy/stake/filtros. Não é recomendação operacional.

## Universo PRIMÁRIO

| Item | Valor |
|---|---|
| LIVE_OK H3BUP_vNext_20260629 Back Pre | **74** |
| Eventos únicos | **58** |
| Friendly | **39** (52.7%) |
| Non-Friendly | **35** (47.3%) |
| Unclassified | **0** |
| Conflict | **0** |
| Classification coverage | **100%** (fonte: `league_name` via audit DB) |

## Performance

| Métrica | Friendly | Non-Friendly | Total |
|---|---:|---:|---:|
| Stake placed | 390 | 350 | 740 |
| Open | 1 | **17** | 18 |
| Settled decided | 35 | 17 | 52 |
| Void/push | 3 | 1 | 4 |
| Stake resolved | 380 | 180 | 560 |
| P&L resolved | **-62.7** | **-214.4** | **-277.1** |
| ROI resolved | **-16.5%** | **-119.1%** | **-49.5%** |
| ROI ex-void | -17.9% | -126.1% | -53.3% |
| Accounting coverage | **97.4%** | **51.4%** | 75.7% |
| Maturity | PARTIALLY_SETTLED | PARTIALLY_SETTLED | PARTIALLY_SETTLED |
| Sample gate | INSUFFICIENT_N | VERY_LOW_N | INSUFFICIENT_N |

### Leitura executiva

1. Friendly está **negativo** (P&L −62.7, ROI −16.5%).
2. Non-Friendly está **negativo** (P&L −214.4, ROI −119.1% no resolvido).
3. A maior parte da perda resolvida está em **Non-Friendly** (−214.4 vs −62.7).
4. CLV closing: Friendly mediana ≈ −2.70% (N=6); Non-Friendly ≈ −1.68% (N=6) — ambos VERY_LOW_N; closing **não confirma** claramente a magnitude da diferença de ROI.
5. A diferença de ROI **sobrevive** a remoção de top ganhos/ligas (sinal estável nos cenários de robustez), mas o IC bootstrap inclui zero.
6. Friendly tem **pior slippage mediano** (−3.12% vs −0.64%) e pre_submit p50 ligeiramente pior; Non-Friendly tem p95 de latência pior.
7. Cobertura **não é comparável**: Friendly 97% accounting vs Non-Friendly 51%; Non-Friendly tem 17 open vs 1.
8. Evidência estatística: **INSUFFICIENT_N** (n_resolved=56); delta ROI F−NF ≈ +1.03; IC95 ≈ [−1.13, +3.19]; permutation p ≈ 0.27.
9. Limitação principal: **amostra pequena + maturidade/cobertura assimétrica** (muitos Non-Friendly ainda open) + concentração em poucas ligas (Europa League / Sudamericana).

## CLV (VALID_STRICT)

| Grupo | Janela | N | Coverage | Média | Mediana | Positivo % | Status |
|---|---|---:|---:|---:|---:|---:|---|
| Friendly | POST_5M | 5 | 12.8% | −1.79 | −1.37 | 20% | VERY_LOW_N |
| Non-Friendly | POST_5M | 15 | 42.9% | −0.13 | −0.39 | 26.7% | VERY_LOW_N |
| Friendly | POST_15M | 5 | 12.8% | −3.47 | −2.80 | 20% | VERY_LOW_N |
| Non-Friendly | POST_15M | 14 | 40.0% | −0.03 | −0.35 | 28.6% | VERY_LOW_N |
| Friendly | CLOSING | 6 | 15.4% | −3.56 | −2.70 | 33.3% | VERY_LOW_N |
| Non-Friendly | CLOSING | 6 | 17.1% | −2.09 | −1.68 | 16.7% | VERY_LOW_N |

## Ligas (destaque)

- **Friendly = 100% Club Friendly** (39/39).
- Maiores perdas Non-Friendly: UEFA Europa League (−128.8), CONMEBOL Sudamericana (−99.95), UEFA Champions League (−71.45).
- Colombia Cup é o principal ganho Non-Friendly (+90.4).

## Alertas

- `ACCOUNTING_COVERAGE_DIFFERENCE` (Friendly 97% vs Non-Friendly 51%)
- `INSUFFICIENT_N` (n_resolved=56)

## Segurança (antes = depois)

| Artefacto | SHA256 |
|---|---|
| wf_policy_current.json | `8009c8fa…725ea1f8` (inalterado) |
| bridge_risk_params.json | `6a38c6dd…27bb3ae6` (inalterado) |
| h3bup_clv_worker.py | `3a190fb5…d3fea2897` (inalterado) |
| accounting_monitor.py | `e7a1872c…6fba1697` (inalterado) |

Telegram / ordens / betslips / timers / env: **Não utilizados / Não alterados**.

## Respostas Q39–Q47

39–47: **Não** (policy, stake, executor, accounting, CLV, timer, Telegram, ordem, betslip).

## Outputs

`logs/h3bup_friendly_analysis/20260801/ae996053a99a/`
