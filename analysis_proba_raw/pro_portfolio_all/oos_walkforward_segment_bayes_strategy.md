## OOS walk-forward (segment_bayes) — estratégia completa
- Dataset: `/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv`
- Semanas totais no dataset: **26**; semanas testadas OOS (WF): **16** (a partir de 10 semanas globais de treino; por-segmento exige >= 6)
- **Modo segment**: constraints avaliadas por segmento (equivalente ao que já rodamos antes), α=1.

### Performance OOS (cap2) — portfólio agregado
- **PnL semanal médio (bootstrap IC95%)**: **USD 170.4** (IC95% -121.4..499.4)
- **Desvio padrão semanal**: USD 660.4
- **P(semana < 0)**: 50.0%

### Performance OOS (cap2) — excluindo semanas sem trades (stake=0)
- **PnL semanal médio (bootstrap IC95%)**: **USD 170.4** (IC95% -123.9..504.0)
- **Desvio padrão semanal**: USD 660.4
- **P(semana < 0)**: 50.0%
- **ROI on stake agregado (ponderado)**: 0.0407

### Risco no OOS (teste) — portfólio agregado
- p80(soma stakes/dia) = USD 1602 (limite=USD 1610)
- VaR10%(PnL diário) = USD -285.2 (limite >= USD -575)
- P(PnL diário <= -25% banca) = 0.0% (limite <= 10%)

### Ajuste de stake global (α)
- α médio=1.000; p10=1.000; p50=1.000; p90=1.000; P(α<1)=0.0%

### Estabilidade OOS da decisão por segmento (frequência de ativação)
- Arquivo regras: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_segment_bayes_selected_rules.csv`

- **FH | domingo**: active_rate=18.8%, ok_rate=18.8%, stake_frac_médio=0.81%, cutoff_médio=0.86
- **FH | quarta-feira**: active_rate=56.2%, ok_rate=56.2%, stake_frac_médio=1.56%, cutoff_médio=0.55
- **FH | quinta-feira**: active_rate=93.8%, ok_rate=93.8%, stake_frac_médio=6.44%, cutoff_médio=0.64
- **FH | segunda-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | sexta-feira**: active_rate=31.2%, ok_rate=31.2%, stake_frac_médio=0.31%, cutoff_médio=0.71
- **FH | sábado**: active_rate=93.8%, ok_rate=93.8%, stake_frac_médio=2.06%, cutoff_médio=0.39
- **FH | terça-feira**: active_rate=68.8%, ok_rate=68.8%, stake_frac_médio=4.62%, cutoff_médio=0.78
- **FT | domingo**: active_rate=81.2%, ok_rate=81.2%, stake_frac_médio=5.69%, cutoff_médio=0.57
- **FT | quarta-feira**: active_rate=75.0%, ok_rate=75.0%, stake_frac_médio=5.25%, cutoff_médio=0.63
- **FT | quinta-feira**: active_rate=93.8%, ok_rate=93.8%, stake_frac_médio=1.81%, cutoff_médio=0.78
- **FT | segunda-feira**: active_rate=56.2%, ok_rate=56.2%, stake_frac_médio=1.31%, cutoff_médio=0.55
- **FT | sexta-feira**: active_rate=37.5%, ok_rate=37.5%, stake_frac_médio=2.63%, cutoff_médio=0.84
- **FT | sábado**: active_rate=100.0%, ok_rate=100.0%, stake_frac_médio=6.31%, cutoff_médio=0.58
- **FT | terça-feira**: active_rate=50.0%, ok_rate=50.0%, stake_frac_médio=1.12%, cutoff_médio=0.60

### Segmentos mais estáveis no OOS (por lucro semanal)
- CSV: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_segment_bayes_segment_stability.csv`
- **FT|quarta-feira**: mean_week=70.9 (IC95% -59.8..212.8), P(semana>0)=31.2%
- **FT|terça-feira**: mean_week=64.9 (IC95% 3.4..136.5), P(semana>0)=37.5%
- **FT|segunda-feira**: mean_week=64.3 (IC95% -6.1..137.6), P(semana>0)=50.0%
- **FH|quinta-feira**: mean_week=38.2 (IC95% -47.8..138.2), P(semana>0)=37.5%
- **FH|terça-feira**: mean_week=21.2 (IC95% -64.6..92.0), P(semana>0)=31.2%
- **FT|sexta-feira**: mean_week=17.0 (IC95% 0.0..44.1), P(semana>0)=12.5%
- **FH|sábado**: mean_week=7.5 (IC95% -63.4..72.7), P(semana>0)=43.8%
- **FT|sábado**: mean_week=-4.1 (IC95% -97.6..101.6), P(semana>0)=25.0%
