## OOS walk-forward (global_classic) — estratégia completa
- Dataset: `/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv`
- Semanas totais no dataset: **26**; semanas testadas OOS (WF): **16** (a partir de 10 semanas globais de treino; por-segmento exige >= 6)
- **Modo global**: aplica um fator α (0..1) multiplicando todos os stakes para satisfazer constraints do portfólio agregado no treino de cada passo.

### Performance OOS (cap2) — portfólio agregado
- **PnL semanal médio (bootstrap IC95%)**: **USD 184.7** (IC95% -79.9..463.5)
- **Desvio padrão semanal**: USD 570.1
- **P(semana < 0)**: 43.8%

### Performance OOS (cap2) — excluindo semanas sem trades (stake=0)
- **PnL semanal médio (bootstrap IC95%)**: **USD 211.1** (IC95% -79.9..527.0)
- **Desvio padrão semanal**: USD 607.5
- **P(semana < 0)**: 50.0%
- **ROI on stake agregado (ponderado)**: 0.0465

### Risco no OOS (teste) — portfólio agregado
- p80(soma stakes/dia) = USD 1806 (limite=USD 1610)
- VaR10%(PnL diário) = USD -270.3 (limite >= USD -575)
- P(PnL diário <= -25% banca) = 0.0% (limite <= 10%)

### Ajuste de stake global (α)
- α médio=0.836; p10=0.647; p50=0.808; p90=1.000; P(α<1)=62.5%

### Estabilidade OOS da decisão por segmento (frequência de ativação)
- Arquivo regras: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_global_classic_selected_rules.csv`

- **FH | domingo**: active_rate=6.2%, ok_rate=6.2%, stake_frac_médio=0.06%, cutoff_médio=0.94
- **FH | quarta-feira**: active_rate=56.2%, ok_rate=56.2%, stake_frac_médio=3.50%, cutoff_médio=0.52
- **FH | quinta-feira**: active_rate=62.5%, ok_rate=62.5%, stake_frac_médio=3.69%, cutoff_médio=0.57
- **FH | segunda-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | sexta-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | sábado**: active_rate=75.0%, ok_rate=75.0%, stake_frac_médio=0.94%, cutoff_médio=0.41
- **FH | terça-feira**: active_rate=50.0%, ok_rate=50.0%, stake_frac_médio=2.94%, cutoff_médio=0.69
- **FT | domingo**: active_rate=37.5%, ok_rate=37.5%, stake_frac_médio=0.38%, cutoff_médio=0.72
- **FT | quarta-feira**: active_rate=75.0%, ok_rate=75.0%, stake_frac_médio=4.50%, cutoff_médio=0.56
- **FT | quinta-feira**: active_rate=25.0%, ok_rate=25.0%, stake_frac_médio=0.44%, cutoff_médio=0.78
- **FT | segunda-feira**: active_rate=56.2%, ok_rate=56.2%, stake_frac_médio=1.69%, cutoff_médio=0.56
- **FT | sexta-feira**: active_rate=25.0%, ok_rate=25.0%, stake_frac_médio=0.38%, cutoff_médio=0.80
- **FT | sábado**: active_rate=75.0%, ok_rate=75.0%, stake_frac_médio=1.56%, cutoff_médio=0.53
- **FT | terça-feira**: active_rate=62.5%, ok_rate=62.5%, stake_frac_médio=1.75%, cutoff_médio=0.51

### Segmentos mais estáveis no OOS (por lucro semanal)
- CSV: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_global_classic_segment_stability.csv`
- **FT|terça-feira**: mean_week=75.0 (IC95% 21.1..137.9), P(semana>0)=50.0%
- **FT|sábado**: mean_week=65.9 (IC95% 10.7..135.1), P(semana>0)=50.0%
- **FT|segunda-feira**: mean_week=45.9 (IC95% -14.2..107.3), P(semana>0)=43.8%
- **FH|sábado**: mean_week=37.9 (IC95% 6.4..75.0), P(semana>0)=50.0%
- **FT|quarta-feira**: mean_week=17.4 (IC95% -86.8..120.9), P(semana>0)=37.5%
- **FH|quarta-feira**: mean_week=10.5 (IC95% -49.2..68.5), P(semana>0)=31.2%
- **FT|sexta-feira**: mean_week=-1.7 (IC95% -16.1..9.1), P(semana>0)=12.5%
- **FT|domingo**: mean_week=-1.8 (IC95% -6.9..2.2), P(semana>0)=6.2%
