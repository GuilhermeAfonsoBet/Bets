## OOS walk-forward (segment_bayes) — estratégia completa
- Dataset: `/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv`
- Semanas totais no dataset: **26**; semanas testadas OOS (WF): **16** (a partir de 10 semanas globais de treino; por-segmento exige >= 6)
- **Modo segment**: constraints avaliadas por segmento (equivalente ao que já rodamos antes), α=1.

### Performance OOS (cap2) — portfólio agregado
- **PnL semanal médio (bootstrap IC95%)**: **USD 166.2** (IC95% -86.7..447.8)
- **Desvio padrão semanal**: USD 565.7
- **P(semana < 0)**: 31.2%

### Performance OOS (cap2) — excluindo semanas sem trades (stake=0)
- **PnL semanal médio (bootstrap IC95%)**: **USD 177.3** (IC95% -90.6..478.5)
- **Desvio padrão semanal**: USD 583.8
- **P(semana < 0)**: 33.3%
- **ROI on stake agregado (ponderado)**: 0.0336

### Risco no OOS (teste) — portfólio agregado
- p80(soma stakes/dia) = USD 2254 (limite=USD 1610)
- VaR10%(PnL diário) = USD -183.5 (limite >= USD -575)
- P(PnL diário <= -25% banca) = 0.0% (limite <= 10%)

### Ajuste de stake global (α)
- α médio=1.000; p10=1.000; p50=1.000; p90=1.000; P(α<1)=0.0%

### Estabilidade OOS da decisão por segmento (frequência de ativação)
- Arquivo regras: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_segment_bayes_selected_rules.csv`

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
- CSV: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_segment_bayes_segment_stability.csv`
- **FT|segunda-feira**: mean_week=66.8 (IC95% -4.7..140.9), P(semana>0)=50.0%
- **FT|terça-feira**: mean_week=44.2 (IC95% -26.9..125.2), P(semana>0)=31.2%
- **FT|quarta-feira**: mean_week=42.3 (IC95% -75.9..169.1), P(semana>0)=31.2%
- **FT|sábado**: mean_week=37.4 (IC95% -8.1..82.2), P(semana>0)=50.0%
- **FH|sábado**: mean_week=19.3 (IC95% -49.7..107.2), P(semana>0)=31.2%
- **FH|quinta-feira**: mean_week=17.2 (IC95% -64.8..118.4), P(semana>0)=31.2%
- **FT|domingo**: mean_week=4.3 (IC95% -8.6..17.2), P(semana>0)=18.8%
- **FH|terça-feira**: mean_week=-1.1 (IC95% -36.1..30.9), P(semana>0)=25.0%
