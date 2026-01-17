## OOS pós-filtro por ROI debiased (gating)

Heurística: em cada semana, desligar combinações com ROI_debiased<=0 (estimado apenas com passado).

- Baseline mean/sem=248.7, std=504.7, Sharpe_ann=3.554
- Gating   mean/sem=189.1, std=482.9, Sharpe_ann=2.824

Arquivos:
- `analysis_proba_raw/pro_portfolio_all/oos_postfilter_debiased_roi_weekly.csv`
- `analysis_proba_raw/pro_portfolio_all/oos_postfilter_debiased_roi_gating.csv`
- `analysis_proba_raw/pro_portfolio_all/oos_postfilter_debiased_roi_comparison.csv`
