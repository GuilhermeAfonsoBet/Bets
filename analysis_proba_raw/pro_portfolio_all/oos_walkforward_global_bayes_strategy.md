## OOS walk-forward (global_bayes) — estratégia completa
- Dataset: `/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv`
- Semanas totais no dataset: **26**; semanas testadas OOS (WF): **16** (a partir de 10 semanas globais de treino; por-segmento exige >= 6)
- **Modo global**: aplica um fator α (0..1) multiplicando todos os stakes para satisfazer constraints do portfólio agregado no treino de cada passo.

### Performance OOS (cap2) — portfólio agregado
- **PnL semanal médio (bootstrap IC95%)**: **USD 144.5** (IC95% -62.1..374.1)
- **Desvio padrão semanal**: USD 460.8
- **P(semana < 0)**: 43.8%

### Performance OOS (cap2) — excluindo semanas sem trades (stake=0)
- **PnL semanal médio (bootstrap IC95%)**: **USD 154.1** (IC95% -69.6..397.1)
- **Desvio padrão semanal**: USD 475.3
- **P(semana < 0)**: 46.7%
- **ROI on stake agregado (ponderado)**: 0.0345

### Risco no OOS (teste) — portfólio agregado
- p80(soma stakes/dia) = USD 1834 (limite=USD 1610)
- VaR10%(PnL diário) = USD -162.0 (limite >= USD -575)
- P(PnL diário <= -25% banca) = 0.0% (limite <= 10%)

### Ajuste de stake global (α)
- α médio=0.863; p10=0.677; p50=0.952; p90=1.000; P(α<1)=75.0%

### Estabilidade OOS da decisão por segmento (frequência de ativação)
- Arquivo regras: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_global_bayes_selected_rules.csv`

- **FH | domingo**: active_rate=12.5%, ok_rate=12.5%, stake_frac_médio=0.12%, cutoff_médio=0.89
- **FH | quarta-feira**: active_rate=50.0%, ok_rate=50.0%, stake_frac_médio=0.81%, cutoff_médio=0.57
- **FH | quinta-feira**: active_rate=75.0%, ok_rate=75.0%, stake_frac_médio=3.75%, cutoff_médio=0.48
- **FH | segunda-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | sexta-feira**: active_rate=31.2%, ok_rate=31.2%, stake_frac_médio=0.31%, cutoff_médio=0.71
- **FH | sábado**: active_rate=81.2%, ok_rate=81.2%, stake_frac_médio=3.25%, cutoff_médio=0.54
- **FH | terça-feira**: active_rate=37.5%, ok_rate=37.5%, stake_frac_médio=0.75%, cutoff_médio=0.73
- **FT | domingo**: active_rate=50.0%, ok_rate=50.0%, stake_frac_médio=0.75%, cutoff_médio=0.63
- **FT | quarta-feira**: active_rate=75.0%, ok_rate=75.0%, stake_frac_médio=4.12%, cutoff_médio=0.51
- **FT | quinta-feira**: active_rate=12.5%, ok_rate=12.5%, stake_frac_médio=0.25%, cutoff_médio=0.90
- **FT | segunda-feira**: active_rate=56.2%, ok_rate=56.2%, stake_frac_médio=1.38%, cutoff_médio=0.55
- **FT | sexta-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FT | sábado**: active_rate=81.2%, ok_rate=81.2%, stake_frac_médio=1.62%, cutoff_médio=0.51
- **FT | terça-feira**: active_rate=56.2%, ok_rate=56.2%, stake_frac_médio=1.69%, cutoff_médio=0.55

### Segmentos mais estáveis no OOS (por lucro semanal)
- CSV: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_global_bayes_segment_stability.csv`
- **FT|segunda-feira**: mean_week=53.1 (IC95% -10.3..113.6), P(semana>0)=50.0%
- **FT|terça-feira**: mean_week=41.9 (IC95% -16.6..108.9), P(semana>0)=31.2%
- **FT|sábado**: mean_week=34.5 (IC95% -7.2..75.6), P(semana>0)=50.0%
- **FT|quarta-feira**: mean_week=25.2 (IC95% -68.0..118.7), P(semana>0)=31.2%
- **FH|sábado**: mean_week=21.0 (IC95% -40.4..102.7), P(semana>0)=37.5%
- **FH|quinta-feira**: mean_week=9.4 (IC95% -56.2..84.7), P(semana>0)=31.2%
- **FT|domingo**: mean_week=4.2 (IC95% -6.7..14.6), P(semana>0)=18.8%
- **FH|terça-feira**: mean_week=-3.1 (IC95% -37.0..27.6), P(semana>0)=25.0%
