## Comparação antes vs depois — calibração por combinação (global_bayes)

- Baseline commit: `a206d56`

### Global (portfólio)
- **Lucro total (OOS WF)**: antes=3086.6, depois=-852.0
- **Lucro médio semanal**: antes=192.9, depois=-53.2
- **Std semanal**: antes=499.3, depois=332.9
- **Sharpe anualizado**: antes=2.786, depois=-1.153
- **ROI por $ (turnover)**: antes=0.0464, depois=-0.0381
- **Forecast Bias (y - pred)**: antes=-201.1, depois=-162.6
- **Coverage 80%**: antes=62.5%, depois=56.2%

### Top mudanças por combinação
**Maior queda de lucro médio semanal por combinação**
- **FT|quarta-feira**: mean_week 18.7 -> -50.3 (Δ -69.0); active_rate 0.69 -> 0.62
- **FT|sábado**: mean_week 55.9 -> -5.8 (Δ -61.7); active_rate 0.62 -> 0.12
- **FT|segunda-feira**: mean_week 62.0 -> 18.3 (Δ -43.7); active_rate 0.56 -> 0.50
- **FT|terça-feira**: mean_week 59.8 -> 27.5 (Δ -32.4); active_rate 0.62 -> 0.31
- **FH|quinta-feira**: mean_week 5.6 -> -17.6 (Δ -23.3); active_rate 0.62 -> 0.50
- **FH|terça-feira**: mean_week -0.4 -> -5.6 (Δ -5.2); active_rate 0.38 -> 0.25
- **FT|quinta-feira**: mean_week -12.2 -> -14.5 (Δ -2.3); active_rate 0.12 -> 0.06

**Combinações mais otimistas em ROI (após shrinkage)**
- **FH|quarta-feira**: bias_roi_shrunk=-0.04027, n_obs=1
- **FH|quinta-feira**: bias_roi_shrunk=-0.04027, n_obs=8
- **FH|terça-feira**: bias_roi_shrunk=-0.04027, n_obs=5
- **FT|quarta-feira**: bias_roi_shrunk=-0.04027, n_obs=11
- **FT|quinta-feira**: bias_roi_shrunk=-0.04027, n_obs=1
- **FT|segunda-feira**: bias_roi_shrunk=-0.04027, n_obs=8
- **FT|sábado**: bias_roi_shrunk=-0.04027, n_obs=1

### Arquivos
- `analysis_proba_raw/pro_portfolio_all/before_after_global_comparison.csv`
- `analysis_proba_raw/pro_portfolio_all/before_after_rule_comparison.csv`
