## OOS walk-forward (global_bayes) — estratégia completa
- Dataset: `/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv`
- Semanas totais no dataset: **26**; semanas testadas OOS (WF): **16** (a partir de 10 semanas globais de treino; por-segmento exige >= 6)
- **Modo global**: aplica um fator α (0..1) multiplicando todos os stakes para satisfazer constraints do portfólio agregado no treino de cada passo.

### Performance OOS (cap2) — portfólio agregado
- **PnL semanal médio (bootstrap IC95%)**: **USD 171.6** (IC95% -116.5..500.1)
- **Desvio padrão semanal**: USD 646.5
- **P(semana < 0)**: 50.0%

### Performance OOS (cap2) — excluindo semanas sem trades (stake=0)
- **PnL semanal médio (bootstrap IC95%)**: **USD 171.6** (IC95% -116.5..495.6)
- **Desvio padrão semanal**: USD 646.5
- **P(semana < 0)**: 50.0%
- **ROI on stake agregado (ponderado)**: 0.0425

### Risco no OOS (teste) — portfólio agregado
- p80(soma stakes/dia) = USD 1518 (limite=USD 1610)
- VaR10%(PnL diário) = USD -280.6 (limite >= USD -575)
- P(PnL diário <= -25% banca) = 0.0% (limite <= 10%)

### Ajuste de stake global (α)
- α médio=0.973; p10=0.883; p50=1.000; p90=1.000; P(α<1)=25.0%

### Estabilidade OOS da decisão por segmento (frequência de ativação)
- Arquivo regras: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_global_bayes_selected_rules.csv`

- **FH | domingo**: active_rate=25.0%, ok_rate=25.0%, stake_frac_médio=0.88%, cutoff_médio=0.82
- **FH | quarta-feira**: active_rate=62.5%, ok_rate=62.5%, stake_frac_médio=1.62%, cutoff_médio=0.51
- **FH | quinta-feira**: active_rate=93.8%, ok_rate=93.8%, stake_frac_médio=6.44%, cutoff_médio=0.64
- **FH | segunda-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | sexta-feira**: active_rate=31.2%, ok_rate=31.2%, stake_frac_médio=0.31%, cutoff_médio=0.71
- **FH | sábado**: active_rate=93.8%, ok_rate=93.8%, stake_frac_médio=2.06%, cutoff_médio=0.39
- **FH | terça-feira**: active_rate=68.8%, ok_rate=68.8%, stake_frac_médio=4.62%, cutoff_médio=0.78
- **FT | domingo**: active_rate=87.5%, ok_rate=87.5%, stake_frac_médio=5.75%, cutoff_médio=0.54
- **FT | quarta-feira**: active_rate=75.0%, ok_rate=75.0%, stake_frac_médio=5.25%, cutoff_médio=0.63
- **FT | quinta-feira**: active_rate=93.8%, ok_rate=93.8%, stake_frac_médio=1.81%, cutoff_médio=0.78
- **FT | segunda-feira**: active_rate=56.2%, ok_rate=56.2%, stake_frac_médio=1.31%, cutoff_médio=0.55
- **FT | sexta-feira**: active_rate=37.5%, ok_rate=37.5%, stake_frac_médio=2.63%, cutoff_médio=0.84
- **FT | sábado**: active_rate=100.0%, ok_rate=100.0%, stake_frac_médio=6.31%, cutoff_médio=0.58
- **FT | terça-feira**: active_rate=50.0%, ok_rate=50.0%, stake_frac_médio=1.12%, cutoff_médio=0.60

### Segmentos mais estáveis no OOS (por lucro semanal)
- CSV: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_global_bayes_segment_stability.csv`
- **FT|quarta-feira**: mean_week=67.8 (IC95% -61.7..208.1), P(semana>0)=31.2%
- **FT|segunda-feira**: mean_week=63.5 (IC95% -6.7..136.5), P(semana>0)=50.0%
- **FT|terça-feira**: mean_week=63.0 (IC95% 3.0..131.3), P(semana>0)=37.5%
- **FH|quinta-feira**: mean_week=38.9 (IC95% -47.5..138.9), P(semana>0)=37.5%
- **FH|terça-feira**: mean_week=21.6 (IC95% -63.0..92.0), P(semana>0)=31.2%
- **FT|sexta-feira**: mean_week=17.0 (IC95% 0.0..44.1), P(semana>0)=12.5%
- **FH|sábado**: mean_week=5.4 (IC95% -64.5..68.6), P(semana>0)=43.8%
- **FT|sábado**: mean_week=-2.6 (IC95% -95.9..102.9), P(semana>0)=25.0%
