## Comparação antes vs depois — calibração por combinação (global_bayes)

- Baseline commit: `a206d56`

### Global (portfólio)
- **Lucro total (OOS WF)**: antes=3086.6, depois=3634.6
- **Lucro médio semanal**: antes=192.9, depois=227.2
- **Std semanal**: antes=499.3, depois=510.0
- **Sharpe anualizado**: antes=2.786, depois=3.212
- **ROI por $ (turnover)**: antes=0.0464, depois=0.0540
- **Forecast Bias (y - pred)**: antes=-201.1, depois=-168.6
- **Coverage 80%**: antes=62.5%, depois=62.5%

### Top mudanças por combinação
**Maior queda de lucro médio semanal por combinação**
- **FH|domingo**: mean_week -7.4 -> -7.4 (Δ 0.0); active_rate 0.12 -> 0.12
- **FH|quarta-feira**: mean_week -14.5 -> -14.5 (Δ 0.0); active_rate 0.38 -> 0.38
- **FH|quinta-feira**: mean_week 5.6 -> 5.6 (Δ 0.0); active_rate 0.62 -> 0.62
- **FH|sexta-feira**: mean_week -5.8 -> -5.8 (Δ 0.0); active_rate 0.25 -> 0.25
- **FH|sábado**: mean_week 28.3 -> 28.3 (Δ 0.0); active_rate 0.75 -> 0.75
- **FT|domingo**: mean_week 2.9 -> 2.9 (Δ 0.0); active_rate 0.25 -> 0.25
- **FT|segunda-feira**: mean_week 62.0 -> 62.0 (Δ 0.0); active_rate 0.56 -> 0.56

**Combinações mais otimistas em ROI (após shrinkage)**
- **FH|domingo**: bias_roi_shrunk=-0.01876, n_obs=2
- **FH|quarta-feira**: bias_roi_shrunk=-0.01876, n_obs=8
- **FH|quinta-feira**: bias_roi_shrunk=-0.01876, n_obs=12
- **FH|sexta-feira**: bias_roi_shrunk=-0.01876, n_obs=5
- **FH|sábado**: bias_roi_shrunk=-0.01876, n_obs=13
- **FH|terça-feira**: bias_roi_shrunk=-0.01876, n_obs=7
- **FT|domingo**: bias_roi_shrunk=-0.01876, n_obs=8

### Arquivos
- `analysis_proba_raw/pro_portfolio_all/before_after_global_comparison.csv`
- `analysis_proba_raw/pro_portfolio_all/before_after_rule_comparison.csv`
