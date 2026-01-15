## Thompson Sampling (bandit) — portfólio com risco global
- Decay=0.970; limiar P(mu>0)>=0.55; MU_SCALE=0.20
- Regras base por segmento: bayes_select=True

### Performance OOS (cap2)
- PnL semanal médio (bootstrap IC95%): USD 98.0 (IC95% -118.8..351.2)
- Std semanal: USD 494.7; P(semana<0)=50.0%
- ROI on stake (ponderado): 0.0280

### Risco no OOS (teste)
- p80(soma stakes/dia)=USD 1220 (limite=USD 1610)
- VaR10%(PnL diário)=USD -245.0 (limite >= USD -575)
- P(PnL diário <= -25% banca)=0.0% (limite <= 10%)

### Segmentos mais usados (peso médio)
- **FH|quinta-feira**: active_rate=100.0%, mean_weight=1.000, mean_Ppos=100.0%
- **FH|terça-feira**: active_rate=100.0%, mean_weight=0.969, mean_Ppos=99.0%
- **FT|domingo**: active_rate=100.0%, mean_weight=0.946, mean_Ppos=99.2%
- **FT|sábado**: active_rate=100.0%, mean_weight=0.936, mean_Ppos=100.0%
- **FT|quarta-feira**: active_rate=100.0%, mean_weight=0.920, mean_Ppos=99.5%
- **FT|quinta-feira**: active_rate=100.0%, mean_weight=0.890, mean_Ppos=99.3%
- **FT|segunda-feira**: active_rate=100.0%, mean_weight=0.860, mean_Ppos=97.0%
- **FT|sexta-feira**: active_rate=100.0%, mean_weight=0.799, mean_Ppos=94.2%
- **FH|domingo**: active_rate=100.0%, mean_weight=0.782, mean_Ppos=94.5%
- **FH|quarta-feira**: active_rate=100.0%, mean_weight=0.613, mean_Ppos=97.5%

### Arquivos
- `analysis_proba_raw/pro_portfolio_all/thompson_global_weekly.csv`
- `analysis_proba_raw/pro_portfolio_all/thompson_global_daily.csv`
- `analysis_proba_raw/pro_portfolio_all/thompson_global_weights.csv`
- `analysis_proba_raw/pro_portfolio_all/thompson_global_weights_summary.csv`
