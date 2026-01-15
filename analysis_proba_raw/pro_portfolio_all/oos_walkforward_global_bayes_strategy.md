## OOS walk-forward (global_bayes) — estratégia completa
- Dataset: `/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv`
- Semanas totais no dataset: **26**; semanas testadas OOS (WF): **16** (a partir de 10 semanas globais de treino; por-segmento exige >= 6)
- **Modo global**: aplica um fator α (0..1) multiplicando todos os stakes para satisfazer constraints do portfólio agregado no treino de cada passo.

### Performance OOS (cap2) — portfólio agregado
- **PnL semanal médio (bootstrap IC95%)**: **USD -6.4** (IC95% -179.8..175.3)
- **Desvio padrão semanal**: USD 375.8
- **P(semana < 0)**: 50.0%

### Performance OOS (cap2) — excluindo semanas sem trades (stake=0)
- **PnL semanal médio (bootstrap IC95%)**: **USD -6.4** (IC95% -179.0..174.9)
- **Desvio padrão semanal**: USD 375.8
- **P(semana < 0)**: 50.0%
- **ROI on stake agregado (ponderado)**: -0.0042

### Risco no OOS (teste) — portfólio agregado
- p80(soma stakes/dia) = USD 772 (limite=USD 1610)
- VaR10%(PnL diário) = USD -334.8 (limite >= USD -575)
- P(PnL diário <= -25% banca) = 0.0% (limite <= 10%)

### Ajuste de stake global (α)
- α médio=0.990; p10=1.000; p50=1.000; p90=1.000; P(α<1)=6.2%

### Estabilidade OOS da decisão por segmento (frequência de ativação)
- Arquivo regras: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_global_bayes_selected_rules.csv`

- **FH | domingo**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | quarta-feira**: active_rate=6.2%, ok_rate=6.2%, stake_frac_médio=0.06%, cutoff_médio=0.95
- **FH | quinta-feira**: active_rate=93.8%, ok_rate=93.8%, stake_frac_médio=5.81%, cutoff_médio=0.69
- **FH | segunda-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | sexta-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | sábado**: active_rate=93.8%, ok_rate=93.8%, stake_frac_médio=6.00%, cutoff_médio=0.70
- **FH | terça-feira**: active_rate=37.5%, ok_rate=37.5%, stake_frac_médio=2.63%, cutoff_médio=0.92
- **FT | domingo**: active_rate=75.0%, ok_rate=75.0%, stake_frac_médio=5.25%, cutoff_médio=0.62
- **FT | quarta-feira**: active_rate=18.8%, ok_rate=18.8%, stake_frac_médio=0.69%, cutoff_médio=0.92
- **FT | quinta-feira**: active_rate=93.8%, ok_rate=93.8%, stake_frac_médio=1.81%, cutoff_médio=0.78
- **FT | segunda-feira**: active_rate=56.2%, ok_rate=56.2%, stake_frac_médio=1.25%, cutoff_médio=0.54
- **FT | sexta-feira**: active_rate=37.5%, ok_rate=37.5%, stake_frac_médio=2.63%, cutoff_médio=0.84
- **FT | sábado**: active_rate=100.0%, ok_rate=100.0%, stake_frac_médio=5.44%, cutoff_médio=0.70
- **FT | terça-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00

### Segmentos mais estáveis no OOS (por lucro semanal)
- CSV: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_global_bayes_segment_stability.csv`
- **FT|segunda-feira**: mean_week=62.9 (IC95% -7.5..136.3), P(semana>0)=50.0%
- **FH|quinta-feira**: mean_week=42.8 (IC95% -29.7..136.9), P(semana>0)=31.2%
- **FT|sexta-feira**: mean_week=17.0 (IC95% 0.0..44.1), P(semana>0)=12.5%
- **FH|quarta-feira**: mean_week=-5.0 (IC95% -15.1..0.0), P(semana>0)=0.0%
- **FH|sábado**: mean_week=-15.5 (IC95% -71.3..21.6), P(semana>0)=12.5%
- **FT|sábado**: mean_week=-21.8 (IC95% -95.3..35.9), P(semana>0)=18.8%
- **FH|terça-feira**: mean_week=-23.2 (IC95% -98.2..33.8), P(semana>0)=12.5%
- **FT|quinta-feira**: mean_week=-30.6 (IC95% -101.2..7.9), P(semana>0)=12.5%
