## Separação de efeito: Reporte vs Otimizador (global_bayes)

- Baseline commit: `a206d56`

### Global
- **Baseline realizado**: mean/sem=192.9, Sharpe_ann=2.786, ROI/$=0.0464
- **Reporte (baseline)**: forecast mean=394.0; forecast corrigido=192.9
- **Otimizador (versão atual, novo OOS)**: mean/sem=-53.2, Sharpe_ann=-1.153, ROI/$=-0.0381

### Principais diferenças
- Reporte ajustado altera apenas expectativas/projeções; não muda o realizado OOS.
- Otimizador (versão atual) muda seleção/stakes/cutoffs; portanto muda o realizado OOS.

### Arquivos
- `analysis_proba_raw/pro_portfolio_all/report_vs_optimizer_global.csv`
- `analysis_proba_raw/pro_portfolio_all/report_vs_optimizer_rules.csv`
