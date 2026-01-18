## Bias-adjusted on-line (walk-forward) — forecast de PnL semanal (global_bayes)

- Modo: **global_bayes_roll12_robust_p10_p70**
- Janela de bias (rolling): **8** semanas passadas

### Métricas (OOS, por semana)
- Bias (real - pred), modelo cru: **USD -15.5**
- Bias (real - pred), bias-adjusted on-line: **USD -6.8**
- MAE, modelo cru: **USD 166.1**
- MAE, bias-adjusted on-line: **USD 177.7**

### Arquivos
- CSV: `analysis_proba_raw/pro_portfolio_all/forecast_calibration_global_bayes_roll12_robust_p10_p70_online_bias.csv`
