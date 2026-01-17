## Calibração do modelo completo — PnL Previsto -> PnL Teórico Realizado (global_bayes)

- Folds (semanas WF): **16**
- Draws por fold (posterior preditivo): **10000**

### Erros (ponto: média prevista)
- Bias (média do erro): **USD -52.5**
- MAE: **USD 431.3**
- RMSE: **USD 526.9**

### Decomposição do erro (stake vs ROI)
- Identidade: PnL = Stake * ROI.
- Definições no preditivo: P̂ = E[S*R], Ŝ = E[S], R̂ = E[R].
- Termo de dependência (cov): cov = E[S*R] - E[S]E[R].
- Decomposição que fecha a conta do erro y - P̂:
  y - P̂ = (S-Ŝ)R̂ + Ŝ(R-R̂) + (S-Ŝ)(R-R̂) - cov.
- Média componente **stake** ((S-Ŝ)R̂): **USD 53.1**
- Média componente **ROI** (Ŝ(R-R̂)): **USD 32.8**
- Média **interação** ((S-Ŝ)(R-R̂)): **USD 15.5**
- Média **cov** (E[S*R]-E[S]E[R]): **USD 153.9**
- Média total (decomp): **USD -52.5** (deve bater com Bias)

### Diagnóstico direto (stake e ROI, semanas com trade)
- Stake: média (real - previsto): **USD 1,484.9** (positivo => o previsto estava menor que o realizado)
- ROI: média (real - previsto): **0.03053** (negativo => o previsto estava maior que o realizado)

### Calibração probabilística
- Coverage 80% (p10..p90): **56.2%** (ideal ~80%)
- Coverage 90% (p05..p95): **75.0%** (ideal ~90%)
- PIT médio: **0.555** (ideal ~0.5)
- CRPS médio (aprox): **280.2** (menor é melhor)

### Arquivos
- CSV: `analysis_proba_raw/pro_portfolio_all/forecast_calibration_global_bayes.csv`
