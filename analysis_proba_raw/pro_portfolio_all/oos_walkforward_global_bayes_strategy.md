## OOS walk-forward (global_bayes) — estratégia completa
- Dataset: `/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv`
- Semanas totais no dataset: **26**; semanas testadas OOS (WF): **16** (a partir de 10 semanas globais de treino; por-segmento exige >= 6)
- **Modo global**: aplica um fator α (0..1) multiplicando todos os stakes para satisfazer constraints do portfólio agregado no treino de cada passo.

### Performance OOS (cap2) — portfólio agregado
- **PnL semanal médio (bootstrap IC95%)**: **USD 50.9** (IC95% -125.5..257.9)
- **Desvio padrão semanal**: USD 405.5
- **P(semana < 0)**: 56.2%

### Performance OOS (cap2) — excluindo semanas sem trades (stake=0)
- **PnL semanal médio (bootstrap IC95%)**: **USD 54.3** (IC95% -134.9..277.1)
- **Desvio padrão semanal**: USD 419.5
- **P(semana < 0)**: 60.0%
- **ROI on stake agregado (ponderado)**: 0.0195

### Risco no OOS (teste) — portfólio agregado
- p80(soma stakes/dia) = USD 1556 (limite=USD 1610)
- VaR10%(PnL diário) = USD -277.8 (limite >= USD -575)
- P(PnL diário <= -25% banca) = 0.0% (limite <= 10%)

### Ajuste de stake global (α)
- α médio=0.946; p10=0.733; p50=1.000; p90=1.000; P(α<1)=18.8%

### Estabilidade OOS da decisão por segmento (frequência de ativação)
- Arquivo regras: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_global_bayes_selected_rules.csv`

- **FH | domingo**: active_rate=6.2%, ok_rate=6.2%, stake_frac_médio=0.06%, cutoff_médio=0.95
- **FH | quarta-feira**: active_rate=6.2%, ok_rate=6.2%, stake_frac_médio=0.38%, cutoff_médio=0.95
- **FH | quinta-feira**: active_rate=68.8%, ok_rate=68.8%, stake_frac_médio=2.88%, cutoff_médio=0.45
- **FH | segunda-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | sexta-feira**: active_rate=6.2%, ok_rate=6.2%, stake_frac_médio=0.06%, cutoff_médio=0.94
- **FH | sábado**: active_rate=31.2%, ok_rate=31.2%, stake_frac_médio=0.62%, cutoff_médio=0.81
- **FH | terça-feira**: active_rate=43.8%, ok_rate=43.8%, stake_frac_médio=1.25%, cutoff_médio=0.73
- **FT | domingo**: active_rate=18.8%, ok_rate=18.8%, stake_frac_médio=0.19%, cutoff_médio=0.86
- **FT | quarta-feira**: active_rate=68.8%, ok_rate=68.8%, stake_frac_médio=2.94%, cutoff_médio=0.59
- **FT | quinta-feira**: active_rate=12.5%, ok_rate=12.5%, stake_frac_médio=0.25%, cutoff_médio=0.90
- **FT | segunda-feira**: active_rate=56.2%, ok_rate=56.2%, stake_frac_médio=1.31%, cutoff_médio=0.55
- **FT | sexta-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FT | sábado**: active_rate=43.8%, ok_rate=43.8%, stake_frac_médio=0.56%, cutoff_médio=0.78
- **FT | terça-feira**: active_rate=25.0%, ok_rate=25.0%, stake_frac_médio=0.44%, cutoff_médio=0.79

### Segmentos mais estáveis no OOS (por lucro semanal)
- CSV: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_global_bayes_segment_stability.csv`
- **FT|segunda-feira**: mean_week=64.3 (IC95% -6.1..137.6), P(semana>0)=50.0%
- **FH|terça-feira**: mean_week=38.3 (IC95% -31.1..133.7), P(semana>0)=25.0%
- **FT|quarta-feira**: mean_week=10.1 (IC95% -90.8..114.5), P(semana>0)=37.5%
- **FH|sábado**: mean_week=3.3 (IC95% -2.9..11.1), P(semana>0)=12.5%
- **FH|quinta-feira**: mean_week=0.9 (IC95% -64.0..75.5), P(semana>0)=25.0%
- **FH|domingo**: mean_week=-2.2 (IC95% -6.5..0.0), P(semana>0)=0.0%
- **FT|sábado**: mean_week=-3.3 (IC95% -30.7..23.1), P(semana>0)=6.2%
- **FT|domingo**: mean_week=-4.3 (IC95% -12.9..0.0), P(semana>0)=0.0%
