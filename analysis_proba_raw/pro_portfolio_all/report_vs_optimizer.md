## Separação de efeito: Reporte vs Otimizador (global_bayes)

- Baseline commit: `a206d56`

### Global
- **Baseline realizado**: mean/sem=192.9, Sharpe_ann=2.786, ROI/$=0.0464
- **Reporte (baseline)**: forecast mean=394.0; forecast corrigido=192.9
- **Otimizador (versão atual, novo OOS)**: mean/sem=248.7, Sharpe_ann=3.554, ROI/$=0.0725

### Principais diferenças
- Reporte ajustado altera apenas expectativas/projeções; não muda o realizado OOS.
- Otimizador (versão atual) muda seleção/stakes/cutoffs; portanto muda o realizado OOS.

### Arquivos
- `analysis_proba_raw/pro_portfolio_all/report_vs_optimizer_global.csv`
- `analysis_proba_raw/pro_portfolio_all/report_vs_optimizer_rules.csv`
