## Auditoria — alinhamento de scoring vs `ApostaLive.rf_prob`

Este relatório recalcula `proba_cli_like` usando os mesmos artefatos do CLI `score_logit_by_dow_cli.py`:
- Seg..Qui: modelo SegQui + calibração isotônica
- Sex..Dom: modelo SexDom + calibração isotônica
- Piso (calib_floor): 0.005

- Observações: **6625** (rf_prob inválido/fora [0,1]: **7**)
- Correlação global rf_prob vs proba_cli_like: **0.395**
- MAE global |rf_prob - proba_cli_like|: **0.1214**

### Por dia-da-semana
- **segunda-feira**: n=310, corr=0.035, MAE=0.2250
- **terça-feira**: n=648, corr=-0.052, MAE=0.2736
- **quarta-feira**: n=583, corr=-0.005, MAE=0.2190
- **quinta-feira**: n=636, corr=0.358, MAE=0.1730
- **sexta-feira**: n=1486, corr=0.538, MAE=0.0764
- **sábado**: n=2052, corr=0.607, MAE=0.0663
- **domingo**: n=910, corr=0.539, MAE=0.0770

### Diagnóstico: qual score mais se parece com rf_prob?
- **segunda-feira**: melhor corr com **proba_raw_segqui** = **0.049**
- **terça-feira**: melhor corr com **proba_raw_operacional** = **0.612**
- **quarta-feira**: melhor corr com **proba_raw_operacional** = **0.275**
- **quinta-feira**: melhor corr com **proba_raw_operacional** = **0.371**
- **sexta-feira**: melhor corr com **proba_cli_like** = **0.538**
- **sábado**: melhor corr com **proba_cli_like** = **0.607**
- **domingo**: melhor corr com **proba_cli_like** = **0.539**

### Arquivos
- CSV: `analysis_proba_raw/pro_portfolio_all/scoring_alignment_vs_rf_prob.csv`
