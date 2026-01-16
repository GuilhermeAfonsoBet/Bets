## Calibração do modelo completo — PnL Previsto -> PnL Teórico Realizado (global_bayes)

- Folds (semanas WF): **16**
- Draws por fold (posterior preditivo): **10000**

### Erros (ponto: média prevista)
- Bias (média do erro): **USD -201.1**
- MAE: **USD 489.5**
- RMSE: **USD 565.0**

### Calibração probabilística
- Coverage 80% (p10..p90): **62.5%** (ideal ~80%)
- Coverage 90% (p05..p95): **75.0%** (ideal ~90%)
- PIT médio: **0.417** (ideal ~0.5)
- CRPS médio (aprox): **307.2** (menor é melhor)

### Arquivos
- CSV: `analysis_proba_raw/pro_portfolio_all/forecast_calibration_global_bayes.csv`
