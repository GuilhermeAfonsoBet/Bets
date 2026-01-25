## Calibração por combinação (rule_key) — global_bayes

- Observações (linhas): **37**
- Segmentos (rule_key): **8**

### Interpretação rápida
- `error_roi = ROI_real - ROI_previsto`: negativo => previsão otimista de ROI.
- `bias_roi_shrunk`: bias de ROI com shrinkage (pooling) entre segmentos.

### Segmentos com ROI mais otimista (bias shrunken mais negativo)
- **FH|quarta-feira**: bias_roi_shrunk=0.00131, n=6
- **FH|quinta-feira**: bias_roi_shrunk=0.00131, n=3
- **FH|sábado**: bias_roi_shrunk=0.00131, n=1
- **FH|terça-feira**: bias_roi_shrunk=0.00131, n=4
- **FT|quarta-feira**: bias_roi_shrunk=0.00131, n=10
- **FT|segunda-feira**: bias_roi_shrunk=0.00131, n=7
- **FT|sexta-feira**: bias_roi_shrunk=0.00131, n=1

### Arquivos
- CSV (por semana e segmento): `analysis_proba_raw/pro_portfolio_all/forecast_calibration_global_bayes_roll12_robust_p10_p70_by_rule.csv`
- CSV (resumo + shrinkage): `analysis_proba_raw/pro_portfolio_all/forecast_calibration_global_bayes_roll12_robust_p10_p70_by_rule_summary.csv`
