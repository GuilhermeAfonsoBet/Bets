## Comparação antes vs depois — calibração por combinação (global_bayes)

- Baseline commit: `a206d56`

### Global (portfólio)
- **Lucro total (OOS WF)**: antes=3086.6, depois=3980.0
- **Lucro médio semanal**: antes=192.9, depois=248.7
- **Std semanal**: antes=499.3, depois=504.7
- **Sharpe anualizado**: antes=2.786, depois=3.554
- **ROI por $ (turnover)**: antes=0.0464, depois=0.0725
- **Forecast Bias (y - pred)**: antes=-201.1, depois=-52.5
- **Coverage 80%**: antes=62.5%, depois=56.2%

### Top mudanças por combinação
**Maior queda de lucro médio semanal por combinação**
- **FT|quarta-feira**: mean_week 18.7 -> 2.6 (Δ -16.1); active_rate 0.69 -> 0.75
- **FH|terça-feira**: mean_week -0.4 -> -9.5 (Δ -9.1); active_rate 0.38 -> 0.31
- **FT|domingo**: mean_week 2.9 -> 0.0 (Δ -2.9); active_rate 0.25 -> 0.12
- **FT|segunda-feira**: mean_week 62.0 -> 61.2 (Δ -0.8); active_rate 0.56 -> 0.56
- **FT|quinta-feira**: mean_week -12.2 -> -7.5 (Δ 4.7); active_rate 0.12 -> 0.12
- **FT|sábado**: mean_week 55.9 -> 61.6 (Δ 5.8); active_rate 0.62 -> 0.62
- **FH|sábado**: mean_week 28.3 -> 41.1 (Δ 12.8); active_rate 0.75 -> 0.75

**Combinações mais otimistas em ROI (após shrinkage)**
- **FH|quarta-feira**: bias_roi_shrunk=0.01155, n_obs=8
- **FH|quinta-feira**: bias_roi_shrunk=0.01155, n_obs=10
- **FH|sábado**: bias_roi_shrunk=0.01155, n_obs=12
- **FH|terça-feira**: bias_roi_shrunk=0.01155, n_obs=6
- **FT|domingo**: bias_roi_shrunk=0.01155, n_obs=5
- **FT|quarta-feira**: bias_roi_shrunk=0.01155, n_obs=12
- **FT|quinta-feira**: bias_roi_shrunk=0.01155, n_obs=2

### Arquivos
- `analysis_proba_raw/pro_portfolio_all/before_after_global_comparison.csv`
- `analysis_proba_raw/pro_portfolio_all/before_after_rule_comparison.csv`
