## OOS pós-filtro por ROI debiased (gating)

Heurística: em cada semana, desligar combinações com ROI_debiased<=0 (estimado apenas com passado).

- Baseline mean/sem=21.1, std=202.0, Sharpe_ann=0.752
- Gating   mean/sem=12.8, std=198.9, Sharpe_ann=0.462

Arquivos:
- `analysis_proba_raw/pro_portfolio_all/oos_postfilter_debiased_roi_global_bayes_roll12_robust_weekly.csv`
- `analysis_proba_raw/pro_portfolio_all/oos_postfilter_debiased_roi_global_bayes_roll12_robust_gating.csv`
- `analysis_proba_raw/pro_portfolio_all/oos_postfilter_debiased_roi_global_bayes_roll12_robust_comparison.csv`
