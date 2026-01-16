## OOS pós-filtro por ROI debiased (gating)

Heurística: em cada semana, desligar combinações com ROI_debiased<=0 (estimado apenas com passado).

- Baseline mean/sem=227.2, std=510.0, Sharpe_ann=3.212
- Gating   mean/sem=179.3, std=410.4, Sharpe_ann=3.151

Arquivos:
- `analysis_proba_raw/pro_portfolio_all/oos_postfilter_debiased_roi_weekly.csv`
- `analysis_proba_raw/pro_portfolio_all/oos_postfilter_debiased_roi_gating.csv`
- `analysis_proba_raw/pro_portfolio_all/oos_postfilter_debiased_roi_comparison.csv`
