## Comparação antes vs depois — calibração por combinação (global_bayes)

- Baseline commit: `a206d56`

### Global (portfólio)
- **Lucro total (OOS WF)**: antes=3086.6, depois=814.5
- **Lucro médio semanal**: antes=192.9, depois=50.9
- **Std semanal**: antes=499.3, depois=405.5
- **Sharpe anualizado**: antes=2.786, depois=0.905
- **ROI por $ (turnover)**: antes=0.0464, depois=0.0195
- **Forecast Bias (y - pred)**: antes=-201.1, depois=-181.3
- **Coverage 80%**: antes=62.5%, depois=50.0%

### Top mudanças por combinação
**Maior queda de lucro médio semanal por combinação**
- **FT|terça-feira**: mean_week 59.8 -> -7.3 (Δ -67.1); active_rate 0.62 -> 0.25
- **FT|sábado**: mean_week 55.9 -> -3.3 (Δ -59.1); active_rate 0.62 -> 0.31
- **FH|sábado**: mean_week 28.3 -> 3.3 (Δ -25.1); active_rate 0.75 -> 0.25
- **FH|quarta-feira**: mean_week -14.5 -> -26.0 (Δ -11.5); active_rate 0.38 -> 0.06
- **FT|quarta-feira**: mean_week 18.7 -> 10.1 (Δ -8.5); active_rate 0.69 -> 0.69
- **FT|domingo**: mean_week 2.9 -> -4.3 (Δ -7.2); active_rate 0.25 -> 0.12
- **FH|sexta-feira**: mean_week -5.8 -> -10.8 (Δ -5.0); active_rate 0.25 -> 0.06

**Combinações mais otimistas em ROI (após shrinkage)**
- **FT|sábado**: bias_roi_shrunk=-0.12504, n_obs=7
- **FT|domingo**: bias_roi_shrunk=-0.10173, n_obs=3
- **FT|quarta-feira**: bias_roi_shrunk=-0.09821, n_obs=11
- **FH|quarta-feira**: bias_roi_shrunk=-0.09808, n_obs=1
- **FH|sábado**: bias_roi_shrunk=-0.09600, n_obs=5
- **FT|terça-feira**: bias_roi_shrunk=-0.09051, n_obs=4
- **FH|sexta-feira**: bias_roi_shrunk=-0.08935, n_obs=1

### Arquivos
- `analysis_proba_raw/pro_portfolio_all/before_after_global_comparison.csv`
- `analysis_proba_raw/pro_portfolio_all/before_after_rule_comparison.csv`
