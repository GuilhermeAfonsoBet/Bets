## Modelo hierárquico (partial pooling) — segmentos — `global_bayes`
- Observações: ROI semanal por segmento (cap2), ponderado por **n_bets**.
- Posterior: Gibbs (burn=2000, draws=8000).
- Hyperparams (médias posteriores): mu0=0.0579, tau=0.1394, sigma=0.9329

### Top 10 segmentos por P(mu>0)
- **FT|terça-feira**: P(mu>0)=99.5%, mu_post=0.1592, CI90% [0.0570..0.2592], mean_obs=0.1851, n_weeks=8, sum_w=192
- **FT|segunda-feira**: P(mu>0)=99.3%, mu_post=0.2149, CI90% [0.0691..0.3645], mean_obs=0.3194, n_weeks=9, sum_w=75
- **FH|sábado**: P(mu>0)=92.5%, mu_post=0.0813, CI90% [-0.0112..0.1736], mean_obs=0.0863, n_weeks=13, sum_w=228
- **FH|terça-feira**: P(mu>0)=84.8%, mu_post=0.1131, CI90% [-0.0638..0.2942], mean_obs=0.2035, n_weeks=9, sum_w=29
- **FT|quarta-feira**: P(mu>0)=82.7%, mu_post=0.0806, CI90% [-0.0627..0.2241], mean_obs=0.0967, n_weeks=11, sum_w=67
- **FH|quinta-feira**: P(mu>0)=80.0%, mu_post=0.0745, CI90% [-0.0744..0.2244], mean_obs=0.0893, n_weeks=11, sum_w=61
- **FT|sexta-feira**: P(mu>0)=73.9%, mu_post=0.0795, CI90% [-0.1319..0.2954], mean_obs=0.1723, n_weeks=5, sum_w=11
- **FT|sábado**: P(mu>0)=63.0%, mu_post=0.0272, CI90% [-0.1084..0.1640], mean_obs=0.0072, n_weeks=11, sum_w=84
- **FH|quarta-feira**: P(mu>0)=52.0%, mu_post=0.0042, CI90% [-0.1430..0.1477], mean_obs=-0.0385, n_weeks=8, sum_w=68
- **FT|domingo**: P(mu>0)=51.9%, mu_post=0.0022, CI90% [-0.2351..0.2222], mean_obs=-0.4167, n_weeks=2, sum_w=6

### Arquivos
- CSV: `analysis_proba_raw/pro_portfolio_all/hierarchical_global_bayes_segment_posterior.csv`
- Fonte: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_global_bayes_weekly_by_segment.csv`
