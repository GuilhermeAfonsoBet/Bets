## OOS walk-forward (global_classic) — estratégia completa
- Dataset: `/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv`
- Semanas totais no dataset: **26**; semanas testadas OOS (WF): **16** (a partir de 10 semanas globais de treino; por-segmento exige >= 6)
- **Modo global**: aplica um fator α (0..1) multiplicando todos os stakes para satisfazer constraints do portfólio agregado no treino de cada passo.

### Performance OOS (cap2) — portfólio agregado
- **PnL semanal médio (bootstrap IC95%)**: **USD -45.5** (IC95% -279.0..189.2)
- **Desvio padrão semanal**: USD 491.2
- **P(semana < 0)**: 56.2%

### Performance OOS (cap2) — excluindo semanas sem trades (stake=0)
- **PnL semanal médio (bootstrap IC95%)**: **USD -56.0** (IC95% -341.9..231.0)
- **Desvio padrão semanal**: USD 548.6
- **P(semana < 0)**: 69.2%
- **ROI on stake agregado (ponderado)**: -0.0204

### Risco no OOS (teste) — portfólio agregado
- p80(soma stakes/dia) = USD 1621 (limite=USD 1610)
- VaR10%(PnL diário) = USD -354.7 (limite >= USD -575)
- P(PnL diário <= -25% banca) = 2.6% (limite <= 10%)

### Ajuste de stake global (α)
- α médio=0.974; p10=0.999; p50=1.000; p90=1.000; P(α<1)=12.5%

### Estabilidade OOS da decisão por segmento (frequência de ativação)
- Arquivo regras: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_global_classic_selected_rules.csv`

- **FH | domingo**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | quarta-feira**: active_rate=6.2%, ok_rate=6.2%, stake_frac_médio=0.12%, cutoff_médio=0.95
- **FH | quinta-feira**: active_rate=50.0%, ok_rate=50.0%, stake_frac_médio=3.25%, cutoff_médio=0.71
- **FH | segunda-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | sexta-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | sábado**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | terça-feira**: active_rate=37.5%, ok_rate=37.5%, stake_frac_médio=0.38%, cutoff_médio=0.80
- **FT | domingo**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FT | quarta-feira**: active_rate=75.0%, ok_rate=75.0%, stake_frac_médio=4.75%, cutoff_médio=0.58
- **FT | quinta-feira**: active_rate=6.2%, ok_rate=6.2%, stake_frac_médio=0.19%, cutoff_médio=0.94
- **FT | segunda-feira**: active_rate=56.2%, ok_rate=56.2%, stake_frac_médio=1.69%, cutoff_médio=0.56
- **FT | sexta-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FT | sábado**: active_rate=18.8%, ok_rate=18.8%, stake_frac_médio=0.38%, cutoff_médio=0.88
- **FT | terça-feira**: active_rate=37.5%, ok_rate=37.5%, stake_frac_médio=0.69%, cutoff_médio=0.69

### Segmentos mais estáveis no OOS (por lucro semanal)
- CSV: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_global_classic_segment_stability.csv`
- **FT|segunda-feira**: mean_week=37.3 (IC95% -27.3..104.8), P(semana>0)=37.5%
- **FT|terça-feira**: mean_week=21.9 (IC95% -10.7..56.0), P(semana>0)=25.0%
- **FH|terça-feira**: mean_week=2.6 (IC95% -11.2..21.7), P(semana>0)=12.5%
- **FH|quarta-feira**: mean_week=-10.0 (IC95% -30.1..0.0), P(semana>0)=0.0%
- **FT|sábado**: mean_week=-13.1 (IC95% -61.8..19.6), P(semana>0)=12.5%
- **FT|quinta-feira**: mean_week=-26.9 (IC95% -80.7..0.0), P(semana>0)=0.0%
- **FH|quinta-feira**: mean_week=-27.6 (IC95% -136.4..60.3), P(semana>0)=37.5%
- **FT|quarta-feira**: mean_week=-29.6 (IC95% -148.9..86.4), P(semana>0)=25.0%
