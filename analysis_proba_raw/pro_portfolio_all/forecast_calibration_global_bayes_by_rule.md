## Calibração por combinação (rule_key) — global_bayes

- Observações (linhas): **62**
- Segmentos (rule_key): **12**

### Interpretação rápida
- `error_roi = ROI_real - ROI_previsto`: negativo => previsão otimista de ROI.
- `bias_roi_shrunk`: bias de ROI com shrinkage (pooling) entre segmentos.

### Segmentos com ROI mais otimista (bias shrunken mais negativo)
- **FT|sábado**: bias_roi_shrunk=-0.12504, n=7
- **FT|domingo**: bias_roi_shrunk=-0.10173, n=3
- **FT|quarta-feira**: bias_roi_shrunk=-0.09821, n=11
- **FH|quarta-feira**: bias_roi_shrunk=-0.09808, n=1
- **FH|sábado**: bias_roi_shrunk=-0.09600, n=5
- **FT|terça-feira**: bias_roi_shrunk=-0.09051, n=4
- **FH|sexta-feira**: bias_roi_shrunk=-0.08935, n=1

### Arquivos
- CSV (por semana e segmento): `analysis_proba_raw/pro_portfolio_all/forecast_calibration_global_bayes_by_rule.csv`
- CSV (resumo + shrinkage): `analysis_proba_raw/pro_portfolio_all/forecast_calibration_global_bayes_by_rule_summary.csv`
