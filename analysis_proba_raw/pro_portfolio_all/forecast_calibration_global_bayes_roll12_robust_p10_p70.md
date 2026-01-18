## Calibração do modelo completo — PnL Previsto -> PnL Teórico Realizado (global_bayes)

- Folds (semanas WF): **16**
- Draws por fold (posterior preditivo): **10000**

### Erros (ponto: média prevista)
- Bias (média do erro): **USD -15.5**
- MAE: **USD 166.1**
- RMSE: **USD 218.3**

### Decomposição do erro (stake vs ROI)
- Identidade: PnL = Stake * ROI.
- Definições no preditivo: P̂ = E[S*R], Ŝ = E[S], R̂ = E[R].
- Termo de dependência (cov): cov = E[S*R] - E[S]E[R].
- Decomposição que fecha a conta do erro y - P̂:
  y - P̂ = (S-Ŝ)R̂ + Ŝ(R-R̂) + (S-Ŝ)(R-R̂) - cov.
- Média componente **stake** ((S-Ŝ)R̂): **USD 19.0**
- Média componente **ROI** (Ŝ(R-R̂)): **USD 76.7**
- Média **interação** ((S-Ŝ)(R-R̂)): **USD -79.2**
- Média **cov** (E[S*R]-E[S]E[R]): **USD 32.0**
- Média total (decomp): **USD -15.5** (deve bater com Bias)

### Diagnóstico direto (stake e ROI, semanas com trade)
- Stake: média (real - previsto): **USD 829.8** (positivo => o previsto estava menor que o realizado)
- ROI: média (real - previsto): **0.05052** (negativo => o previsto estava maior que o realizado)

### Calibração probabilística
- Coverage 80% (p10..p90): **68.8%** (ideal ~80%)
- Coverage 90% (p05..p95): **87.5%** (ideal ~90%)
- PIT médio: **0.596** (ideal ~0.5)
- CRPS médio (aprox): **120.5** (menor é melhor)

### Arquivos
- CSV: `analysis_proba_raw/pro_portfolio_all/forecast_calibration_global_bayes_roll12_robust_p10_p70.csv`
