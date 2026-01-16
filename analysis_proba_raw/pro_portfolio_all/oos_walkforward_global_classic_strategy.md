## OOS walk-forward (global_classic) — estratégia completa
- Dataset: `/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv`
- Semanas totais no dataset: **26**; semanas testadas OOS (WF): **16** (a partir de 10 semanas globais de treino; por-segmento exige >= 6)
- **Modo global**: aplica um fator α (0..1) multiplicando todos os stakes para satisfazer constraints do portfólio agregado no treino de cada passo.

### Performance OOS (cap2) — portfólio agregado
- **PnL semanal médio (bootstrap IC95%)**: **USD 196.8** (IC95% -63.3..471.6)
- **Desvio padrão semanal**: USD 562.8
- **P(semana < 0)**: 43.8%

### Performance OOS (cap2) — excluindo semanas sem trades (stake=0)
- **PnL semanal médio (bootstrap IC95%)**: **USD 224.9** (IC95% -63.8..536.4)
- **Desvio padrão semanal**: USD 598.8
- **P(semana < 0)**: 50.0%
- **ROI on stake agregado (ponderado)**: 0.0524

### Risco no OOS (teste) — portfólio agregado
- p80(soma stakes/dia) = USD 1774 (limite=USD 1610)
- VaR10%(PnL diário) = USD -241.5 (limite >= USD -575)
- P(PnL diário <= -25% banca) = 0.0% (limite <= 10%)

### Ajuste de stake global (α)
- α médio=0.845; p10=0.627; p50=0.845; p90=1.000; P(α<1)=62.5%

### Estabilidade OOS da decisão por segmento (frequência de ativação)
- Arquivo regras: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_global_classic_selected_rules.csv`

- **FH | domingo**: active_rate=6.2%, ok_rate=6.2%, stake_frac_médio=0.06%, cutoff_médio=0.94
- **FH | quarta-feira**: active_rate=56.2%, ok_rate=56.2%, stake_frac_médio=2.56%, cutoff_médio=0.56
- **FH | quinta-feira**: active_rate=62.5%, ok_rate=62.5%, stake_frac_médio=3.69%, cutoff_médio=0.57
- **FH | segunda-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | sexta-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | sábado**: active_rate=75.0%, ok_rate=75.0%, stake_frac_médio=0.94%, cutoff_médio=0.41
- **FH | terça-feira**: active_rate=50.0%, ok_rate=50.0%, stake_frac_médio=2.94%, cutoff_médio=0.68
- **FT | domingo**: active_rate=37.5%, ok_rate=37.5%, stake_frac_médio=0.38%, cutoff_médio=0.72
- **FT | quarta-feira**: active_rate=75.0%, ok_rate=75.0%, stake_frac_médio=4.50%, cutoff_médio=0.56
- **FT | quinta-feira**: active_rate=25.0%, ok_rate=25.0%, stake_frac_médio=0.44%, cutoff_médio=0.78
- **FT | segunda-feira**: active_rate=56.2%, ok_rate=56.2%, stake_frac_médio=1.69%, cutoff_médio=0.56
- **FT | sexta-feira**: active_rate=18.8%, ok_rate=18.8%, stake_frac_médio=0.31%, cutoff_médio=0.85
- **FT | sábado**: active_rate=75.0%, ok_rate=75.0%, stake_frac_médio=1.56%, cutoff_médio=0.53
- **FT | terça-feira**: active_rate=68.8%, ok_rate=68.8%, stake_frac_médio=2.19%, cutoff_médio=0.47

### Segmentos mais estáveis no OOS (por lucro semanal)
- CSV: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_global_classic_segment_stability.csv`
- **FT|sábado**: mean_week=65.6 (IC95% 10.4..134.7), P(semana>0)=50.0%
- **FT|terça-feira**: mean_week=60.6 (IC95% 5.2..122.7), P(semana>0)=43.8%
- **FT|segunda-feira**: mean_week=43.6 (IC95% -18.6..104.8), P(semana>0)=43.8%
- **FT|quarta-feira**: mean_week=33.9 (IC95% -60.6..127.2), P(semana>0)=37.5%
- **FH|sábado**: mean_week=33.3 (IC95% 2.1..70.9), P(semana>0)=43.8%
- **FH|quinta-feira**: mean_week=6.8 (IC95% -67.0..78.4), P(semana>0)=25.0%
- **FT|domingo**: mean_week=-1.7 (IC95% -6.9..2.3), P(semana>0)=6.2%
- **FT|sexta-feira**: mean_week=-3.4 (IC95% -18.8..8.6), P(semana>0)=6.2%
