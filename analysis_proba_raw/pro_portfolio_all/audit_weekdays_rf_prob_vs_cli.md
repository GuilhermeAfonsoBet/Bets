## Auditoria — `ApostaLive.rf_prob` vs stdout do `score_logit_weekdays_cli.py`

- Base: `/workspace/pr1_snapshot/dedup_scored_base.csv`
- Modelos: `/workspace`
- `calib_floor` (clip): **0.005**
- parse-mode: **weekdays_cli**
- Janela: últimos **30** dias (coluna `BIA_ApostaUTC`, se disponível)

### Resumo (rf_prob vs proba_cli)
- N (válidos): **453**
- MAE |rf - cli|: **0.216559**
- P95 |rf - cli|: **0.787524**
- % match exato (@6 casas): **6.84%**

### Por dia
- **quarta-feira**: n=168, corr=0.223, MAE=0.266755, p95=0.832165, match6=0.00%
- **segunda-feira**: n=84, corr=0.406, MAE=0.140903, p95=0.428098, match6=36.90%
- **terca-feira**: n=201, corr=0.392, MAE=0.206222, p95=0.821130, match6=0.00%

### Arquivos
- Mismatches (top 200): `/workspace/analysis_proba_raw/pro_portfolio_all/audit_weekdays_rf_prob_vs_cli.csv`
