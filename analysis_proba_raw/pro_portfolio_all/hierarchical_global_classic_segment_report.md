## Modelo hierárquico (partial pooling) — segmentos — `global_classic`
- Observações: ROI semanal por segmento (cap2), ponderado por **n_bets**.
- Posterior: Gibbs (burn=2000, draws=8000).
- Hyperparams (médias posteriores): mu0=0.0574, tau=0.1279, sigma=0.9507

### Top 10 segmentos por P(mu>0)
- **FT|terça-feira**: P(mu>0)=98.2%, mu_post=0.1238, CI90% [0.0256..0.2213], mean_obs=0.1430, n_weeks=9, sum_w=205
- **FH|terça-feira**: P(mu>0)=94.1%, mu_post=0.1642, CI90% [-0.0078..0.3475], mean_obs=0.3611, n_weeks=9, sum_w=32
- **FH|sábado**: P(mu>0)=91.0%, mu_post=0.0711, CI90% [-0.0164..0.1589], mean_obs=0.0744, n_weeks=13, sum_w=260
- **FT|segunda-feira**: P(mu>0)=89.9%, mu_post=0.1274, CI90% [-0.0347..0.2982], mean_obs=0.2321, n_weeks=8, sum_w=39
- **FT|quinta-feira**: P(mu>0)=78.5%, mu_post=0.0567, CI90% [-0.0602..0.1728], mean_obs=0.0586, n_weeks=3, sum_w=121
- **FT|quarta-feira**: P(mu>0)=78.3%, mu_post=0.0638, CI90% [-0.0702..0.1985], mean_obs=0.0687, n_weeks=11, sum_w=77
- **FH|quinta-feira**: P(mu>0)=75.9%, mu_post=0.0593, CI90% [-0.0821..0.2012], mean_obs=0.0626, n_weeks=11, sum_w=68
- **FT|sexta-feira**: P(mu>0)=74.0%, mu_post=0.0754, CI90% [-0.1228..0.2777], mean_obs=0.1723, n_weeks=5, sum_w=11
- **FT|domingo**: P(mu>0)=58.7%, mu_post=0.0239, CI90% [-0.1698..0.2144], mean_obs=-0.0937, n_weeks=4, sum_w=16
- **FH|quarta-feira**: P(mu>0)=55.7%, mu_post=0.0125, CI90% [-0.1329..0.1555], mean_obs=-0.0327, n_weeks=8, sum_w=64

### Arquivos
- CSV: `analysis_proba_raw/pro_portfolio_all/hierarchical_global_classic_segment_posterior.csv`
- Fonte: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_global_classic_weekly_by_segment.csv`
