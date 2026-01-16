## Calibração do modelo completo — PnL Previsto -> PnL Teórico Realizado (global_bayes)

- Folds (semanas WF): **16**
- Draws por fold (posterior preditivo): **10000**

### Erros (ponto: média prevista)
- Bias (média do erro): **USD -201.1**
- MAE: **USD 489.5**
- RMSE: **USD 565.0**

### Decomposição do erro (stake vs ROI)
- Identidade: PnL = Stake * ROI.
- Definições no preditivo: P̂ = E[S*R], Ŝ = E[S], R̂ = E[R].
- Termo de dependência (cov): cov = E[S*R] - E[S]E[R].
- Decomposição que fecha a conta do erro y - P̂:
  y - P̂ = (S-Ŝ)R̂ + Ŝ(R-R̂) + (S-Ŝ)(R-R̂) - cov.
- Média componente **stake** ((S-Ŝ)R̂): **USD 65.7**
- Média componente **ROI** (Ŝ(R-R̂)): **USD -44.6**
- Média **interação** ((S-Ŝ)(R-R̂)): **USD -46.8**
- Média **cov** (E[S*R]-E[S]E[R]): **USD 175.4**
- Média total (decomp): **USD -201.1** (deve bater com Bias)

### Diagnóstico direto (stake e ROI, semanas com trade)
- Stake: média (real - previsto): **USD 1,687.6** (positivo => o previsto estava menor que o realizado)
- ROI: média (real - previsto): **-0.01102** (negativo => o previsto estava maior que o realizado)

### Calibração probabilística
- Coverage 80% (p10..p90): **62.5%** (ideal ~80%)
- Coverage 90% (p05..p95): **75.0%** (ideal ~90%)
- PIT médio: **0.417** (ideal ~0.5)
- CRPS médio (aprox): **307.2** (menor é melhor)

### Arquivos
- CSV: `analysis_proba_raw/pro_portfolio_all/forecast_calibration_global_bayes.csv`
