## Calibração do modelo completo — PnL Previsto -> PnL Teórico Realizado (global_bayes)

- Folds (semanas WF): **17**
- Draws por fold (posterior preditivo): **10000**

### Erros (ponto: média prevista)
- Bias (média do erro): **USD -4.7**
- MAE: **USD 179.3**
- RMSE: **USD 236.6**

### Decomposição do erro (stake vs ROI)
- Identidade: PnL = Stake * ROI.
- Definições no preditivo: P̂ = E[S*R], Ŝ = E[S], R̂ = E[R].
- Termo de dependência (cov): cov = E[S*R] - E[S]E[R].
- Decomposição que fecha a conta do erro y - P̂:
  y - P̂ = (S-Ŝ)R̂ + Ŝ(R-R̂) + (S-Ŝ)(R-R̂) - cov.
- Média componente **stake** ((S-Ŝ)R̂): **USD 14.9**
- Média componente **ROI** (Ŝ(R-R̂)): **USD 69.2**
- Média **interação** ((S-Ŝ)(R-R̂)): **USD -57.8**
- Média **cov** (E[S*R]-E[S]E[R]): **USD 31.1**
- Média total (decomp): **USD -4.7** (deve bater com Bias)

### Diagnóstico direto (stake e ROI, semanas com trade)
- Stake: média (real - previsto): **USD 852.4** (positivo => o previsto estava menor que o realizado)
- ROI: média (real - previsto): **0.04110** (negativo => o previsto estava maior que o realizado)

### Calibração probabilística
- Coverage 80% (p10..p90): **82.4%** (ideal ~80%)
- Coverage 90% (p05..p95): **100.0%** (ideal ~90%)
- PIT médio: **0.581** (ideal ~0.5)
- CRPS médio (aprox): **133.3** (menor é melhor)

### Arquivos
- CSV: `analysis_proba_raw/pro_portfolio_all/forecast_calibration_global_bayes_roll12_robust_p10_p70.csv`
