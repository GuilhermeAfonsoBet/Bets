## Calibração por combinação (rule_key) — global_bayes

- Observações (linhas): **101**
- Segmentos (rule_key): **12**

### Interpretação rápida
- `error_roi = ROI_real - ROI_previsto`: negativo => previsão otimista de ROI.
- `bias_roi_shrunk`: bias de ROI com shrinkage (pooling) entre segmentos.

### Segmentos com ROI mais otimista (bias shrunken mais negativo)
- **FH|domingo**: bias_roi_shrunk=-0.01876, n=2
- **FH|quarta-feira**: bias_roi_shrunk=-0.01876, n=8
- **FH|quinta-feira**: bias_roi_shrunk=-0.01876, n=12
- **FH|sexta-feira**: bias_roi_shrunk=-0.01876, n=5
- **FH|sábado**: bias_roi_shrunk=-0.01876, n=13
- **FH|terça-feira**: bias_roi_shrunk=-0.01876, n=7
- **FT|domingo**: bias_roi_shrunk=-0.01876, n=8

### Arquivos
- CSV (por semana e segmento): `analysis_proba_raw/pro_portfolio_all/forecast_calibration_global_bayes_by_rule.csv`
- CSV (resumo + shrinkage): `analysis_proba_raw/pro_portfolio_all/forecast_calibration_global_bayes_by_rule_summary.csv`
